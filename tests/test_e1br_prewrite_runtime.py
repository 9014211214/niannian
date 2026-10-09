from tests._model_http_stub import stub_model_http
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import ssl
import sqlite3
import urllib.request
import pytest

from scripts.e1br_prewrite_runtime import (PrewriteE1BRCallGuard as CandidateGuard, base, digest,
    validate_contract, validate_prewrite_ledger as validate_ledger, validate_rows)
from aria.engine.policy import minimal_calls
from aria.engine.policy.minimal_calls import MinimalCallStopped
from tests.test_e1br_calls import guard as old_guard
from tests.test_reserved_calls import client, receipt, policy, call
from tests.test_pre_application_rebuild import connect_fixture, ssl_error


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k): raise AssertionError("network forbidden")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    stub_model_http(monkeypatch, denied)


def opener(monkeypatch, c, failures):
    calls = []
    def open_local(request, *, timeout, trace):
        calls.append(request.data)
        count = 2 - trace.preparation_recovery.budget["attempts"]
        actions = [(ssl_error(ssl.SSLEOFError, number=8), .01)] * count if len(calls) <= failures else [(None, .01)]
        conn, _, _, _, _ = connect_fixture(actions, trace=trace, body=json.dumps(receipt(c)).encode())
        conn.request("POST", "/v1/chat/completions", body=request.data)
        return conn.getresponse()
    monkeypatch.setattr(minimal_calls, "open_observed", open_local)
    return calls


def configured(tmp_path, monkeypatch):
    original = tmp_path / "old-software.sqlite3"
    with closing(old_guard(original)) as g:
        c = client(); g.attach(c)
        calls = opener(monkeypatch, c, 2)
        with pytest.raises(MinimalCallStopped): call(g, c)
        state = g.snapshot()
        assert len(calls) == 1 and state["unknown_cost_requests"] == 1
        parent = deepcopy(state["attempts"][0])
    parent["experiment_position"] = ["online_fast", "e1b_e2", 3, "A0", 1, 4]
    contract = {"schema": "e1br_positive_prewrite_retry_v1", "enabled": True,
        "approval_sha256": "a" * 64, "parent_campaign": "artifacts/local/software-only",
        "parent_manifest_sha256": "b" * 64, "parent_receipt_sha256": digest(parent),
        "parent_attempt": parent, "connection_rebuild_used": True,
        "inherited_physical_attempts": 1, "remaining_infrastructure_retries": 2}
    return contract, original, hashlib.sha256(original.read_bytes()).hexdigest()


def new_guard(path, contract):
    g = CandidateGuard(path, prewrite_contract=contract, max_observed_cost_usd="1", reservation_policy=policy(),
                       registration_sha256="c" * 64, authority_check=lambda: None)
    clock = [0.0]
    g._pacing_clock = lambda: clock[0]
    g._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    return g


@pytest.mark.parametrize("failure_count", [0, 1, 2])
def test_cross_campaign_pool_and_rebuild_are_not_reset(tmp_path, monkeypatch, failure_count):
    contract, original, original_sha = configured(tmp_path, monkeypatch)
    path = tmp_path / "new-software.sqlite3"
    with closing(new_guard(path, contract)) as g:
        c = client(); g.attach(c)
        calls = opener(monkeypatch, c, failure_count)
        if failure_count < 2:
            assert call(g, c) == "local fixture"
        else:
            with pytest.raises(MinimalCallStopped): call(g, c)
        state = g.snapshot()
        assert len(calls) == min(failure_count + 1, 2)
        assert state["unknown_cost_requests"] == failure_count
        assert [r["prewrite_retry"]["lineage_attempt_index"] for r in state["attempts"]] == list(range(1, len(calls) + 1))
        assert all(len(r["transport_events"]["connection_preparations"]) == 1
                   and r["transport_events"]["connection_preparations"][0]["ordinal"] == 2 for r in state["attempts"])
        assert state["prewrite_retry"]["remaining_new_attempts_at_position"] == 2 - len(calls)
        if failure_count == 2:
            assert state["stop_reason"] == "inherited_retry_pool_exhausted"
    with closing(new_guard(path, contract)) as reopened:
        assert reopened.snapshot()["attempts"] == state["attempts"]
        assert reopened.snapshot()["reserved_cost_usd"] == state["reserved_cost_usd"]
        c = client(); reopened.attach(c)
        calls = opener(monkeypatch, c, 0)
        with pytest.raises(MinimalCallStopped): call(reopened, c)
        assert not calls
    assert validate_ledger(path, contract=contract, stage="main", registration_sha256="c" * 64,
                           max_requests=3153, reservation_policy=policy()) == state["attempts"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("connection_rebuild_used"),
    lambda c: c.update(connection_rebuild_used=False),
    lambda c: c.update(remaining_infrastructure_retries=3),
    lambda c: c.update(inherited_physical_attempts=0),
    lambda c: c.update(enabled=False),
    lambda c: c.update(parent_receipt_sha256="d" * 64),
    lambda c: c["parent_attempt"]["transport_events"].update(events={}),
])
def test_no_implicit_optin_or_cross_campaign_reset(tmp_path, monkeypatch, mutation):
    contract, _, _ = configured(tmp_path, monkeypatch)
    mutation(contract)
    with pytest.raises(ValueError): new_guard(tmp_path / "rejected.sqlite3", contract)
    assert not (tmp_path / "rejected.sqlite3").exists()


def test_new_wire_change_refused_before_dispatch(tmp_path, monkeypatch):
    contract, _, _ = configured(tmp_path, monkeypatch)
    with closing(new_guard(tmp_path / "new.sqlite3", contract)) as g:
        c = client(); g.attach(c)
        calls = opener(monkeypatch, c, 0)
        with g.scope("turn", "op1", "session1"):
            with pytest.raises(MinimalCallStopped, match="prewrite_failed_wire_changed"):
                c.complete([{"role": "user", "content": "changed synthetic input"}])
        assert not calls and g.snapshot()["calls_used"] == 0


def test_readonly_validator_rejects_link_or_rebuild_pool_tamper(tmp_path, monkeypatch):
    contract, _, _ = configured(tmp_path, monkeypatch)
    with closing(new_guard(tmp_path / "new.sqlite3", contract)) as g:
        c = client(); g.attach(c); opener(monkeypatch, c, 0)
        assert call(g, c) == "local fixture"
        state = g.snapshot()
    for mutate in (lambda r: r.pop("prewrite_retry"),
                   lambda r: r["prewrite_retry"].update(lineage_attempt_index=0),
                   lambda r: r["prewrite_retry"].update(preparation_rebuild_already_used=False),
                   lambda r: r["transport_events"]["connection_preparations"][0].update(ordinal=1)):
        rows = deepcopy(state["attempts"]); mutate(rows[0])
        with pytest.raises(ValueError): validate_rows(rows, contract)


def test_unselected_hook_base_keeps_original_policy(tmp_path, monkeypatch):
    path = tmp_path / "default.sqlite3"
    with closing(base.E1BRCallGuard(path, max_observed_cost_usd="1", reservation_policy=policy(),
                                   registration_sha256="d" * 64, authority_check=lambda: None)) as g:
        c = client(); g.attach(c); calls = opener(monkeypatch, c, 2)
        with pytest.raises(MinimalCallStopped): call(g, c)
        assert len(calls) == 1 and g.snapshot()["stop_reason"] == "provider_cost_unknown"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='e1br_prewrite_policy'").fetchone() is None
