"""Synthetic local receipts; no model, credentials or network calls."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
from pathlib import Path
import sqlite3
import threading

import pytest

from scripts.t7_ledger import CampaignLedger, LedgerError


SEAL = "a" * 64


def make_ledger(tmp_path: Path) -> CampaignLedger:
    return CampaignLedger(tmp_path / "campaign.sqlite3", campaign_id="t6_20260930_v1", freeze_seal=SEAL)


def inventory(count: int = 2, reserve: str = "0.0011") -> list[dict]:
    return [{"id": f"slot_{index}", "request_sha256": hashlib.sha256(str(index).encode()).hexdigest(), "reserve_usd": reserve} for index in range(count)]


def register(ledger: CampaignLedger, count: int = 2, *, stage: str = "e1", reserve: str = "0.0011") -> None:
    ledger.register_stage(stage, inventory(count, reserve), limit_usd="0.5" if stage == "e1" else "0.3", max_requests=204 if stage == "e1" else 35)


def finish(ledger: CampaignLedger, item: str = "slot_0", *, stage: str = "e1", **kwargs) -> dict:
    receipt = {
        "cost_usd": "0.0003", "prompt_tokens": 5000, "completion_tokens": 100,
        "actual_model": "qwen/qwen3.7-flash" if stage == "e1" else "anthropic/claude-sonnet-5",
        "provider": "alibaba" if stage == "e1" else "anthropic", "response_sha256": "b" * 64,
    }
    receipt.update(kwargs)
    return ledger.finish(stage, item, **receipt)


def test_success_keeps_all_slots_costs_and_immutable_inventory(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    first_hash = ledger.summary()["stages"]["e1"]["inventory_sha256"]
    claimed = ledger.claim("e1", "slot_0")
    assert claimed["request_sha256"] == inventory()[0]["request_sha256"]
    assert len(claimed["claim_token"]) == 64
    finish(ledger, cost_usd=Decimal("0.0003"))
    register(ledger)
    reopened = make_ledger(tmp_path)
    reopened.claim("e1", "slot_1")
    finish(reopened, "slot_1", cost_usd=0.0004)
    summary = reopened.summary()
    assert summary["known_cost_usd"] == "0.0007"
    assert summary["remaining_reserve_usd"] == "0"
    assert not summary["stopped"]
    assert summary["stages"]["e1"]["counts"] == {"not_started": 0, "started": 0, "completed": 2, "failed": 0}
    assert summary["stages"]["e1"]["inventory_sha256"] == first_hash
    assert "claim_token" not in summary["stages"]["e1"]["items"][0]


def test_frozen_e1_full_inventory_reservation_and_decimal_normalization(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger, count=204)
    assert ledger.summary()["remaining_reserve_usd"] == "0.2244"
    ledger.register_stage("e1", inventory(204, "0.001100"), limit_usd=Decimal("0.500"), max_requests=204)
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 204


def test_judge_waits_for_generation_then_locks_its_own_identity(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger, count=1)
    ledger.claim("e1", "slot_0")
    finish(ledger)
    register(ledger, stage="judge", count=1, reserve="0.02")
    ledger.claim("judge", "slot_0")
    finish(ledger, stage="judge", cost_usd="0.01")
    summary = ledger.summary()
    assert summary["known_cost_usd"] == "0.0103"
    assert summary["stages"]["judge"]["provider"] == "anthropic"


def test_judge_cannot_start_before_e1_complete(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    register(ledger, stage="judge")
    with pytest.raises(LedgerError, match="unfinished_claim"):
        ledger.claim("judge", "slot_0")
    assert ledger.summary()["stopped"]


@pytest.mark.parametrize("change", ["campaign", "seal"])
def test_new_identity_cannot_reset_database(tmp_path, change):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    with pytest.raises(LedgerError, match="campaign_mismatch"):
        CampaignLedger(ledger.path, campaign_id="different" if change == "campaign" else "t6_20260930_v1", freeze_seal="c" * 64 if change == "seal" else SEAL)
    assert ledger.summary()["outstanding_count"] == 1


def test_unfinished_claim_survives_reopen_and_blocks_every_stage(tmp_path):
    original = make_ledger(tmp_path)
    register(original)
    register(original, stage="judge")
    original.claim("e1", "slot_0")
    reopened = make_ledger(tmp_path)
    assert reopened.summary()["unknown_cost_count"] == 1
    with pytest.raises(LedgerError, match="unfinished_claim"):
        reopened.claim("judge", "slot_0")
    assert original.summary()["stopped"]
    # The in-flight owner can still preserve a receipt after a concurrent stop.
    finish(original)
    assert original.summary()["known_cost_usd"] == "0.0003"
    with pytest.raises(LedgerError):
        original.claim("e1", "slot_1")


def test_reopened_instance_cannot_reclassify_an_ambiguous_attempt(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    reopened = make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="unknown_result"):
        finish(reopened)
    assert reopened.summary()["outstanding_count"] == 1


@pytest.mark.parametrize("second_id", ["slot_0", "slot_1"])
def test_atomic_claim_across_connections_prevents_parallel_sends(tmp_path, second_id):
    ledger = make_ledger(tmp_path)
    register(ledger)
    clients = [make_ledger(tmp_path), make_ledger(tmp_path)]
    gate = threading.Barrier(2)

    def attempt(index):
        gate.wait()
        try:
            clients[index].claim("e1", "slot_0" if index == 0 else second_id)
            return "claimed"
        except LedgerError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, range(2)))
    # If slot_1 wins the race, out-of-order stops the campaign before any send.
    assert outcomes.count("claimed") <= 1
    assert ledger.summary()["outstanding_count"] <= 1
    assert ledger.summary()["stopped"]


def test_completed_request_is_never_reissued(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    finish(ledger)
    with pytest.raises(LedgerError, match="already_started"):
        ledger.claim("e1", "slot_0")
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 1


@pytest.mark.parametrize("field,value", [
    ("reserve_usd", "0.0012"), ("request_sha256", "d" * 64), ("id", "changed"),
])
def test_changed_inventory_halts_campaign(tmp_path, field, value):
    ledger = make_ledger(tmp_path)
    register(ledger)
    modified = inventory()
    modified[0][field] = value
    with pytest.raises(LedgerError, match="inventory_changed"):
        ledger.register_stage("e1", modified, limit_usd="0.5", max_requests=204)
    assert ledger.summary()["stopped"]
    assert ledger.summary()["stages"]["e1"]["items"][0]["request_sha256"] == inventory()[0]["request_sha256"]


def test_full_stage_must_fit_before_first_claim_and_retains_denominator(tmp_path):
    ledger = make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="stage_budget_exceeded"):
        register(ledger, count=35, stage="judge", reserve="0.01")
    summary = ledger.summary()
    assert summary["stages"]["judge"]["counts"]["not_started"] == 35
    assert summary["known_cost_usd"] == "0"
    assert summary["stopped"]


@pytest.mark.parametrize("stage", ["e2", "formation", "another_run", "E1"])
def test_unregistered_stages_cannot_bypass_limits(tmp_path, stage):
    ledger = make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="stage_not_authorized"):
        ledger.register_stage(stage, inventory(), limit_usd="2", max_requests=288)
    assert ledger.summary()["stages"] == {}


@pytest.mark.parametrize("value", [True, False, "NaN", "Infinity", "-0.1", None, float("nan")])
def test_invalid_reservation_is_rejected_before_mutation(tmp_path, value):
    ledger = make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="invalid_amount"):
        register(ledger, reserve=value)
    assert ledger.summary()["stages"] == {}


@pytest.mark.parametrize("receipt", [
    {"cost_usd": None}, {"cost_usd": True}, {"cost_usd": "NaN"},
    {"prompt_tokens": True}, {"completion_tokens": -1},
    {"prompt_tokens": 32000}, {"completion_tokens": 1025},
    {"actual_model": None}, {"provider": None}, {"response_sha256": None},
    {"prompt_tokens": 2**80}, {"actual_model": {"secret": "not stored"}},
])
def test_invalid_receipt_preserves_known_charge_and_stops(tmp_path, receipt):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    with pytest.raises(LedgerError):
        finish(ledger, **receipt)
    summary = make_ledger(tmp_path).summary()
    assert summary["stopped"]
    assert summary["stages"]["e1"]["counts"]["failed"] == 1
    assert summary["stages"]["e1"]["counts"]["not_started"] == 1
    if "cost_usd" in receipt:
        assert summary["unknown_cost_count"] == 1
    else:
        assert summary["known_cost_usd"] == "0.0003"
        assert summary["unknown_cost_count"] == 0


@pytest.mark.parametrize("change", [
    {"actual_model": "qwen/qwen3.7-flash-20260727"}, {"provider": "other"},
])
def test_first_actual_identity_is_locked_even_to_allowed_model_alias(tmp_path, change):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    finish(ledger)
    ledger.claim("e1", "slot_1")
    with pytest.raises(LedgerError):
        finish(ledger, "slot_1", **change)
    assert make_ledger(tmp_path).summary()["stopped"]
    assert ledger.summary()["known_cost_usd"] == "0.0006"


@pytest.mark.parametrize("cost,reason", [
    ("0.002", "reservation_exceeded"), ("0.5", "stage_budget_exceeded"),
    ("2", "checkpoint_reached"), ("5", "campaign_budget_exceeded"),
])
def test_unexpected_cost_is_preserved_at_every_stop_line(tmp_path, cost, reason):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    with pytest.raises(LedgerError, match=reason):
        finish(ledger, cost_usd=cost)
    summary = make_ledger(tmp_path).summary()
    assert Decimal(summary["known_cost_usd"]) == Decimal(cost)
    assert summary["stop_reason"] == reason
    assert summary["stages"]["e1"]["counts"]["not_started"] == 1


def test_error_text_never_enters_receipt_and_known_cost_remains(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    with pytest.raises(LedgerError, match="external_error"):
        finish(ledger, error="private raw body with credentials")
    assert "private raw" not in repr(ledger.summary())
    assert ledger.summary()["known_cost_usd"] == "0.0003"


def test_save_failure_poison_is_durable_and_prevents_second_claim(tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    connect = ledger._connect

    def fail_connect(*, readonly=False):
        if not readonly:
            raise sqlite3.OperationalError("secret storage path")
        return connect(readonly=True)

    monkeypatch.setattr(ledger, "_connect", fail_connect)
    with pytest.raises(LedgerError, match="ledger_save_failed"):
        finish(ledger)
    reopened = make_ledger(tmp_path)
    assert reopened.summary()["outstanding_count"] == 1
    with pytest.raises(LedgerError, match="ledger_save_failed"):
        reopened.claim("e1", "slot_1")
    assert b"secret storage" not in ledger._stop_path.read_bytes()


def test_failed_claim_commit_never_returns_a_send_ticket(tmp_path, monkeypatch):
    ledger = make_ledger(tmp_path)
    register(ledger)
    connect = ledger._connect

    class CommitFailure:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def commit(self):
            raise sqlite3.OperationalError("commit failed")

    monkeypatch.setattr(ledger, "_connect", lambda **kwargs: connect(**kwargs) if kwargs.get("readonly") else CommitFailure(connect()))
    with pytest.raises(LedgerError, match="ledger_save_failed"):
        ledger.claim("e1", "slot_0")
    assert ledger._owned == {}
    reopened = make_ledger(tmp_path)
    assert reopened.summary()["stopped"]
    with pytest.raises(LedgerError, match="ledger_save_failed"):
        reopened.claim("e1", "slot_0")


def test_summary_and_reopen_are_byte_read_only_even_when_started(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "slot_0")
    before = ledger.path.read_bytes()
    for _ in range(3):
        assert make_ledger(tmp_path).summary()["outstanding_count"] == 1
    assert ledger.path.read_bytes() == before
    assert not ledger._stop_path.exists()


def test_storage_uses_full_synchronous_and_rejects_mutated_schema(tmp_path):
    ledger = make_ledger(tmp_path)
    with ledger._connect() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        conn.execute("UPDATE campaign SET schema_version='future_incompatible'")
    with pytest.raises(LedgerError, match="campaign_mismatch"):
        make_ledger(tmp_path)


def test_column_schema_drift_fails_without_repair(tmp_path):
    ledger = make_ledger(tmp_path)
    with ledger._connect() as conn:
        conn.execute("ALTER TABLE items ADD COLUMN unregistered_field TEXT")
    before = ledger.path.read_bytes()
    with pytest.raises(LedgerError, match="invalid_ledger_schema"):
        make_ledger(tmp_path)
    assert ledger.path.read_bytes() == before


def test_stop_is_campaign_wide_and_does_not_overwrite_original_failure(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.stop("e1", "source_changed")
    ledger.stop("judge", "manual_stop")
    assert ledger.summary()["stop_reason"] == "source_changed"
    with pytest.raises(LedgerError, match="source_changed"):
        register(ledger, stage="judge")


@pytest.mark.parametrize("max_requests", [True, 205, 0, 1.5])
def test_stage_request_limit_is_strict(tmp_path, max_requests):
    ledger = make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="invalid_stage_limits"):
        ledger.register_stage("e1", inventory(1), limit_usd="0.5", max_requests=max_requests)
