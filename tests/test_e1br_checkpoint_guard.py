"""Synthetic inherited attempt, bounded new dispatches, no historical data."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from aria.engine.policy.e1br_calls import E1BRCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped
from scripts.e1br_checkpoint_guard import CheckpointGuardMixin, validate_checkpoint_ledger
from tests.test_e1br_calls import call, http_error, stubs
from tests.test_reserved_calls import client, policy, receipt


class Positioned(E1BRCallGuard):
    def _reserve(self, client, request, payload, state):
        row = super()._reserve(client, request, payload, state)
        row["experiment_position"] = [client.role, "synthetic", 1, "shared", 0, None]
        return row


class Guard(CheckpointGuardMixin, Positioned):
    pass


def configured(path):
    c = client("memory")
    c.reservation_max_price = {"prompt": 2.0, "completion": 10.0}
    captured = []
    def local(request):
        captured.append(request.data)
        return receipt(c)
    original = c._request_json
    c._request_json = local
    c.complete([{"role": "user", "content": "synthetic memory input"}])
    c._request_json = original
    contract = {"schema": "e1br_checkpoint_ledger_v1", "exception_sha256": "a" * 64,
        "parent_receipt_sha256": "d" * 64, "parent_manifest_sha256": "e" * 64,
        "old_no_content_proven": False, "old_physical_attempts_at_position": 1,
        "max_new_attempts_at_position": 2,
        "parent_attempt": {"request_id": "old-failed", "operation_id": "review1", "session_id": "session1",
            "role": "memory", "model": c.model, "request_sha256": hashlib.sha256(captured[0]).hexdigest(),
            "experiment_position": ["memory", "synthetic", 1, "shared", 0, None]}}
    clock = [0.0]
    g = Guard(path, checkpoint_contract=contract, max_requests=2283,
        max_observed_cost_usd="1", reservation_policy=policy(), registration_sha256="b" * 64, authority_check=lambda: None)
    g._pacing_clock = lambda: clock[0]
    g._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    g.attach(c)
    return g, c, contract


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_exact_two_remaining_attempts_with_independent_holds(tmp_path, monkeypatch, failures):
    path = tmp_path / "ledger.sqlite"
    g, c, contract = configured(path)
    with closing(g):
        calls = stubs(monkeypatch, [*[http_error(429) for _ in range(failures)], receipt(c)])
        if failures == 2:
            with pytest.raises(MinimalCallStopped):
                call(g, c)
        else:
            assert call(g, c) == "local fixture"
        state = g.snapshot()
        assert len(calls) == min(failures + 1, 2)
        assert state["unknown_cost_requests"] == failures
        assert [r["checkpoint_retry"]["lineage_attempt_index"] for r in state["attempts"]] == list(range(1, len(calls) + 1))
        assert state["attempts"][0]["checkpoint_retry"]["basis"] == "explicit_single_position_exception"
        if failures == 2:
            assert state["stopped"] and state["stop_reason"] == "inherited_retry_pool_exhausted"
            with pytest.raises(MinimalCallStopped):
                call(g, c)
            assert len(calls) == 2
    assert validate_checkpoint_ledger(path, contract=contract, stage="main", registration_sha256="b" * 64,
        max_requests=2283, reservation_policy=policy()) == state["attempts"]


@pytest.mark.parametrize("failure", [lambda: http_error(429, {"error": {"code": 429}, "choices": []}),
                                    lambda: http_error(401)])
def test_exception_never_waives_next_error_proof(tmp_path, monkeypatch, failure):
    g, c, _ = configured(tmp_path / "ledger.sqlite")
    with closing(g):
        calls = stubs(monkeypatch, [failure(), receipt(c)])
        with pytest.raises(MinimalCallStopped):
            call(g, c)
        assert len(calls) == 1 and g.snapshot()["stopped"]


def test_wrong_first_operation_never_dispatches(tmp_path, monkeypatch):
    g, c, _ = configured(tmp_path / "ledger.sqlite")
    with closing(g):
        calls = stubs(monkeypatch, [receipt(c)])
        with pytest.raises(MinimalCallStopped, match="first_position_unsettled"):
            call(g, c, "another")
        assert not calls and g.snapshot()["calls_used"] == 0


@pytest.mark.parametrize("mutation", ["contract", "row"])
def test_durable_external_contract_tampering_rejected(tmp_path, monkeypatch, mutation):
    path = tmp_path / "ledger.sqlite"
    g, c, contract = configured(path)
    with closing(g):
        stubs(monkeypatch, [receipt(c)])
        call(g, c)
        state = g.snapshot()
    with sqlite3.connect(path) as db:
        if mutation == "contract":
            changed = deepcopy(contract)
            changed["max_new_attempts_at_position"] = 3
            db.execute("UPDATE e1br_checkpoint_policy SET contract=?", (json.dumps(changed, sort_keys=True),))
        else:
            row = deepcopy(state["attempts"][0])
            row["checkpoint_retry"]["old_no_content_proven"] = True
            db.execute("UPDATE attempts SET record=?", (json.dumps(row),))
    with pytest.raises(ValueError, match="checkpoint_"):
        validate_checkpoint_ledger(path, contract=contract, stage="main", registration_sha256="b" * 64,
            max_requests=2283, reservation_policy=policy())
