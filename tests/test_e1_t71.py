"""T7.1 integration uses fictional receipts in temporary, non-paid campaigns."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts import run_e1_experiment as legacy
from scripts import run_e1_t71 as runner
from scripts import t71_policy as policy
from scripts import t7_ledger
from tests.test_e1_execution import fake_envelope


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credential_access_forbidden")

    monkeypatch.setattr(runner, "credential", forbidden)
    monkeypatch.setattr(legacy, "credential", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)


@pytest.fixture(scope="module")
def bundle():
    return runner.load_bundle()


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "private-fictional-campaign"
    directory.mkdir()
    ledger = policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="t71_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


class SyntheticTransport:
    def __init__(self, ledger, *, failure=None):
        self.ledger, self.failure, self.calls = ledger, failure, []

    def request(self, payload):
        # A separate read connection sees the FULL SQLite claim before dispatch.
        assert self.ledger.summary()["outstanding_count"] == 1
        self.calls.append(deepcopy(payload))
        envelope = fake_envelope(payload)
        if self.failure == "usage_over_bound":
            estimate = policy.estimate_request(payload, input_price="0.00000003",
                                                output_price="0.00000013")
            envelope["usage"]["prompt_tokens"] = estimate["input_token_plan_bound"] + 1
        elif self.failure == "multiple_tools":
            calls = envelope["choices"][0]["message"]["tool_calls"]
            calls.append(deepcopy(calls[0]))
        elif self.failure == "unknown_cost":
            envelope["usage"].pop("cost")
        return envelope


def run(campaign, bundle, transport, *, source_check=lambda: None):
    directory, ledger = campaign
    return runner.run_generation(directory, ledger, bundle, transport,
                                  source_check=source_check, sleep=lambda _: None)


def test_original_seals_and_exact_registered_document_revision(bundle):
    assert bundle["baseline"]["seal_sha256"] == runner.T6_SEAL
    assert bundle["baseline"]["verified_files"] == 104
    assert bundle["baseline"]["e1_requests"] == 204
    assert legacy.verify_execution(bundle)["seal_sha256"] == runner.T7_SEAL
    revision = bundle["amendment"]["document_revision"]
    old_document = bundle["root"] / revision["original_reference"]
    assert revision["path"] == "docs/VALIDATION.md"
    assert runner.freeze.file_sha256(old_document) == revision["original_sha256"]
    assert runner.freeze.file_sha256(bundle["root"] / revision["path"]) == revision["current_sha256"]
    original = next(entry for entry in bundle["manifest"]["files"] if entry["path"] == revision["path"])
    assert original["sha256"] == revision["original_sha256"]
    assert old_document.stat().st_size == original["bytes"]
    assert revision["current_sha256"] != revision["original_sha256"]


def test_offline_readiness_true_without_credentials_or_campaign(bundle, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "e" * 64})

    def forbidden(*args, **kwargs):
        raise AssertionError("offline_must_not_open_campaign_or_metadata")

    monkeypatch.setattr(runner, "open_campaign", forbidden)
    monkeypatch.setattr(policy, "fetch_metadata", forbidden)
    assert runner.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready_to_send"] is True and report["blocking_reasons"] == []
    assert report["e1_requests"] == 204 and report["model_requests"] == 0
    assert report["live_metadata_checked"] is False
    assert report["provider_hard_bound_verified"] is False
    assert report["full_reserved_cost_usd"] == "0.2244"


def test_execute_simulated_missing_key_stops_before_claim(bundle, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "e" * 64})
    monkeypatch.setattr(policy, "fetch_metadata", lambda **_: {"synthetic_public_metadata": True})
    attempts = []

    def missing_key(*, allow_local_env=False):
        assert allow_local_env is False
        attempts.append("credential_boundary")
        raise legacy.ExecutionError("credential_not_in_environment")

    def forbidden(*args, **kwargs):
        raise AssertionError("missing_credential_must_not_create_or_claim_campaign")

    monkeypatch.setattr(runner, "credential", missing_key)
    monkeypatch.setattr(runner, "open_campaign", forbidden)
    monkeypatch.setattr(runner, "OpenRouterTransport", forbidden)
    assert runner.main(["--execute"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {"ok": False, "reason": "credential_not_in_environment", "no_automatic_retry": True}
    assert attempts == ["credential_boundary"]


def test_complete_204_preserves_frozen_requests_and_legacy_globals(campaign, bundle):
    directory, ledger = campaign
    original_prepare = legacy.prepare_e1
    original_globals_prepare = legacy.run_generation.__globals__["prepare_e1"]
    original_policy = deepcopy(t7_ledger.STAGES)
    transport = SyntheticTransport(ledger)
    checks = []
    result = run(campaign, bundle, transport, source_check=lambda: checks.append(True))
    assert result["status"] == "complete"
    assert len(transport.calls) == len(checks) == result["denominator"] == len(result["rows"]) == 204
    assert all(row["status"] == "generated" for row in result["rows"])
    assert result["model_effect_measured"] is False
    assert ledger.summary()["stages"]["e1"]["counts"]["completed"] == 204
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert len(list((directory / "e1").glob("*.receipt.json"))) == 204
    inventory = legacy.read_json(directory / "e1" / "inventory.json")
    for wire, record, planned in zip(transport.calls, bundle["manifest"]["e1_requests"], inventory):
        assert runner.freeze.canonical_sha256(wire) == record["request_sha256"] == planned["request_sha256"]
        assert wire["parallel_tool_calls"] is False
        assert planned["input_token_plan_bound"] == len(runner.judge.canonical(wire)) + 8192
    assert legacy.load_generation(directory, ledger, bundle) == result["rows"]
    assert legacy.prepare_e1 is original_prepare
    assert legacy.run_generation.__globals__["prepare_e1"] is original_globals_prepare
    assert t7_ledger.STAGES == original_policy
    assert t7_ledger.STAGES["judge"]["limit"] == Decimal("0.3")
    with pytest.raises((OSError, ValueError)):
        run(campaign, bundle, transport)
    assert len(transport.calls) == 204


@pytest.mark.parametrize("failure,reason", [
    ("usage_over_bound", "token_bound_exceeded"), ("multiple_tools", "invalid_schema"),
    ("unknown_cost", "unknown_cost"),
])
def test_first_fault_stops_and_retains_203_unsent_positions(campaign, bundle, failure, reason):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger, failure=failure)
    result = run(campaign, bundle, transport)
    assert len(transport.calls) == 1 and result["status"] == "incomplete"
    assert result["stop_reason"] == reason
    assert result["denominator"] == len(result["rows"]) == 204
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    summary = ledger.summary()
    assert summary["stopped"] and summary["stop_reason"] == reason
    assert summary["stages"]["e1"]["counts"] == {
        "not_started": 203, "started": 0, "completed": 0, "failed": 1}
    assert (directory / "e1" / "results.private.json").is_file()
    assert len(list((directory / "e1").glob("*.receipt.json"))) == 1
    if failure == "unknown_cost":
        assert summary["unknown_cost_count"] == 1
        assert summary["stages"]["e1"]["items"][0]["cost_usd"] is None
    else:
        assert summary["unknown_cost_count"] == 0
        assert Decimal(summary["known_cost_usd"]) == Decimal("0.0001")
    with pytest.raises(policy.LedgerError):
        ledger.claim("e1", summary["stages"]["e1"]["items"][1]["id"])
    assert len(transport.calls) == 1


def test_checked_transport_rejects_unregistered_wire_without_dispatch():
    class NeverCalled:
        def request(self, payload):
            raise AssertionError("unknown_payload_was_dispatched")

    checked = runner.CheckedTransport(NeverCalled(), {})
    with pytest.raises(policy.TransportError, match="request_changed"):
        checked.request({"model": "not_a_frozen_payload"})


def judge_fixture(bundle):
    suite, _, _, items = runner.prepare_e1(bundle)
    rows = [{**legacy.generation_row(item), "status": "generated",
        "raw_decision": {"response": "我听着。"}, "delivered_response": "我听着。",
        "actual_model": "qwen/qwen3.7-flash", "latency_ms": 1.0,
        "deliberation_status": "captured", "selected_in_deliberated_options": True}
        for item in items]
    prepared = policy.build_batches(rows, suite, bundle["plan"], bundle["contract"],
        {"raw_shuffle": 11, "delivered_shuffle": 22, "pair_shuffle": 33, "human_sample": 44})
    return suite, rows, prepared


def test_full_judge_plan_over_half_dollar_is_not_started(campaign, bundle):
    directory, ledger = campaign
    suite, rows, prepared = judge_fixture(bundle)
    assert len(prepared["batches"]) == 19
    assert prepared["status"] == "blocked_budget" and prepared["max_cost_usd"] == "0.5"
    transport = SyntheticTransport(ledger)
    with pytest.raises(legacy.ExecutionError, match="judge_budget_insufficient"):
        runner.run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
                         source_check=lambda: None, sleep=lambda _: None)
    assert not transport.calls and not ledger.summary()["stages"]
    assert not (directory / "judge").exists()


def test_judge_limit_adapter_uses_half_dollar_in_single_batch_software_fixture(campaign, bundle):
    # This intentionally reduced software fixture is never the paid experiment.
    directory, ledger = campaign
    suite, rows, prepared = judge_fixture(bundle)
    batch = next(batch for batch in prepared["batches"] if batch["stage"] == "raw")
    prepared["batches"] = [batch]
    prepared["total_reserve_usd"] = batch["reserve_usd"]
    prepared["status"] = "prepared"
    ledger.register_stage("e1", [{"id": "software_fixture", "request_sha256": "1" * 64,
        "reserve_usd": "0.0011"}], limit_usd="0.5", max_requests=204)
    ledger.claim("e1", "software_fixture")
    ledger.finish("e1", "software_fixture", cost_usd="0", prompt_tokens=1, completion_tokens=1,
        actual_model="qwen/qwen3.7-flash", provider="alibaba", response_sha256="2" * 64)

    class JudgeTransport:
        calls = 0

        def request(self, payload):
            self.calls += 1
            assert ledger.summary()["outstanding_count"] == 1
            assert payload == batch["payload"]
            content = json.dumps([[key, 0, 0, 0, 0] for key in batch["private"]["items"]])
            return {"model": "anthropic/claude-sonnet-5", "provider": "anthropic",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": "0.001"},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    transport = JudgeTransport()
    result = runner.run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
        source_check=lambda: None, sleep=lambda _: None)
    assert transport.calls == result["completed_batches"] == 1
    assert result["status"] == "complete"
    assert ledger.summary()["stages"]["judge"]["limit_usd"] == "0.5"
    assert not ledger.summary()["stopped"]
    assert t7_ledger.STAGES["judge"]["limit"] == Decimal("0.3")
    assert (directory / "judge" / "results.private.json").is_file()


def synthetic_execution(tmp_path):
    # Exact seven-file inventory, independent of the not-yet-sealed real files.
    assert len(runner.FILES) == 7
    entries = []
    for name in sorted(runner.FILES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source for " + name + "\n", encoding="utf-8")
        entries.append({"path": name, "bytes": path.stat().st_size,
                        "sha256": runner.freeze.file_sha256(path)})
    value = {"schema": "aria_t71_execution_v1",
        "inherits": {"t6_seal": runner.T6_SEAL, "t7_seal": runner.T7_SEAL}, "files": entries}
    value["seal_sha256"] = runner.freeze.canonical_sha256(value)
    target = tmp_path / runner.EXECUTION
    target.write_text(json.dumps(value), encoding="utf-8")
    bundle = {"root": tmp_path}
    assert runner.verify_execution(bundle) == value
    return bundle, target, value


@pytest.mark.parametrize("name", sorted(runner.FILES))
def test_each_of_seven_execution_files_is_bound(tmp_path, name):
    bundle, _, _ = synthetic_execution(tmp_path)
    (tmp_path / name).write_text("changed synthetic source\n", encoding="utf-8")
    with pytest.raises(legacy.ExecutionError, match="t71_source_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("mutation", ["seal", "inventory", "inherits"])
def test_execution_seal_and_inventory_and_inherited_seals_reject_drift(tmp_path, mutation):
    bundle, target, value = synthetic_execution(tmp_path)
    if mutation == "seal":
        value["seal_sha256"] = "f" * 64
        reason = "t71_execution_seal_changed"
    else:
        if mutation == "inventory":
            value["files"].pop()
            reason = "t71_inventory_changed"
        else:
            value["inherits"]["t6_seal"] = "f" * 64
            reason = "t71_execution_seal_changed"
        value["seal_sha256"] = runner.freeze.canonical_sha256(
            {key: item for key, item in value.items() if key != "seal_sha256"})
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(legacy.ExecutionError, match=reason):
        runner.verify_execution(bundle)
