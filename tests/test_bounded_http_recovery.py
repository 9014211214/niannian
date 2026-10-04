"""Bounded recovery uses fake transport/time and preserves every attempt."""
from copy import deepcopy
from decimal import Decimal
import io
import json
import socket
import sqlite3
import urllib.error
import urllib.request

import pytest

from aria.engine.policy import minimal_calls
from aria.engine.policy.bounded_recovery import validate_recovery_records
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_reserved_calls import policy, client, receipt, call


def recovery_policy():
    return {"schema": "bounded_http429_recovery_v1", "max_extra_dispatches": 2, "max_new_unknown": 2,
        "backoff_seconds": [60, 120], "max_attempts_per_operation": 3, "max_logical_requests": 1636,
        "authorization_sha256": "a" * 64}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    blocked = lambda *a, **k: pytest.fail("real network forbidden")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def guarded(path, *, budget="1", recovery=True, clock=None):
    g = ReservedCallGuard(path, max_requests=1638, max_observed_cost_usd=budget,
        reservation_policy=policy(), min_interval_seconds=10, observe_transport=True,
        recovery_policy=recovery_policy() if recovery else None)
    clock = clock if clock is not None else [0.0]
    g._pacing_clock = g._recovery_clock = lambda: clock[0]
    g._pacing_sleep = lambda wait: clock.__setitem__(0, clock[0] + wait)
    return g, clock


def scripted(monkeypatch, actions, clock):
    sent = []
    class Response(io.BytesIO):
        status = 200
        headers = {}
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        clock[0] += .25
        action = actions.pop(0)
        if action == "timeout":
            trace._mark("connection_preparation_started")
            raise TimeoutError("PRIVATE timeout")
        for event in ("connection_preparation_started", "connection_preparation_completed", "tls_completed",
                      "application_write_attempted", "request_write_completed", "response_headers_returned"):
            trace._mark(event)
        payload = json.loads(request.data)
        name = payload["provider"]["only"][0].title()
        if action == "ok":
            data = receipt(client("online_fast" if name == "Alibaba" else "memory"))
            return Response(json.dumps(data).encode())
        if action == "bad_json":
            return Response(b'{"PRIVATE')
        body = {"error": {"code": 429, "message": "PRIVATE", "metadata": {"provider_name": name}}}
        status = 429
        if action == "known429":
            body["usage"] = {"cost": 0}
        elif action == "wrong_provider":
            body["error"]["metadata"]["provider_name"] = "Anthropic" if name == "Alibaba" else "Alibaba"
        elif action == "has_answer":
            body["choices"] = [{"message": {"content": "PRIVATE response"}}]
        elif action == "http500":
            status = body["error"]["code"] = 500
        elif action == "no_error_code":
            body["error"].pop("code")
        raw = json.dumps(body).encode()
        if action == "duplicate":
            raw = b'{"error":{"code":429,"code":429,"metadata":{"provider_name":"Alibaba"}}}'
        elif action == "oversized":
            raw += b" " * (16 * 1024)
        raise urllib.error.HTTPError("https://fixture.invalid", status, "PRIVATE", {}, io.BytesIO(raw))
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    return sent


def checked(g, complete=False):
    state = g.snapshot()
    result = validate_recovery_records(state["attempts"], g.recovery_policy,
        relations=state["recovery"]["relations"], require_complete=complete)
    return state, result


def test_unknown_429_then_success_retains_failed_row_hold_identity_and_latency(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3")
    c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok"], clock)
    assert call(g, c) == "local fixture"
    state, summary = checked(g, True)
    assert len(sent) == c.network_request_count == state["calls_used"] == 2
    assert sent[0] == sent[1]
    failed, success = state["attempts"]
    assert failed["status"] == "failed" and success["status"] == "received"
    assert failed["request_id"] != success["request_id"]
    assert success["recovery_of_request_id"] == failed["request_id"]
    assert success["recovery_wait_seconds"] == 60 and success["recovery_wait_ms"] == 60000
    assert success["logical_elapsed_ms"] >= 60500
    assert summary["logical_calls_used"] == summary["logical_received"] == 1
    assert summary["extra_dispatches"] == summary["authorized_unknown_requests"] == 1
    assert summary["logical_attempts"] == [success] and not state["stopped"]
    assert Decimal(state["reserved_cost_usd"]) == Decimal(failed["reservation"]["reserved_cost_usd"]) > 0
    assert "PRIVATE" not in json.dumps(state)
    g.close()


def test_three_attempts_only_when_prior_errors_have_known_cost(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3")
    c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["known429", "known429", "ok", "known429", "ok"], clock)
    assert call(g, c) == "local fixture"
    state, summary = checked(g, True)
    assert [row["physical_attempt"] for row in state["attempts"]] == [1, 2, 3]
    assert [row["recovery_wait_seconds"] for row in state["attempts"][1:]] == [60, 120]
    assert summary["extra_dispatches"] == 2 and summary["new_unknown_requests"] == 0
    with pytest.raises(MinimalCallStopped):
        call(g, c, "other")
    assert len(sent) == 4 and g.snapshot()["stopped"]
    g.close()


@pytest.mark.parametrize("actions,first_completes", [(["429", "429", "ok"], False), (["429", "ok", "429", "ok"], True)])
def test_second_unknown_stops_before_any_possible_third(tmp_path, monkeypatch, actions, first_completes):
    path = tmp_path / "calls.sqlite3"
    g, clock = guarded(path); c = client(); g.attach(c)
    sent = scripted(monkeypatch, actions, clock)
    if first_completes:
        assert call(g, c) == "local fixture"
    with pytest.raises(MinimalCallStopped):
        call(g, c, "other" if first_completes else "op1")
    state, summary = checked(g)
    assert state["unknown_cost_requests"] == 2 and state["stopped"]
    assert summary["extra_dispatches"] == 1
    before = len(sent)
    g.close()
    g, _ = guarded(path, clock=clock); c = client("memory"); g.attach(c)
    with pytest.raises(MinimalCallStopped):
        call(g, c, "next-stage")
    assert len(sent) == before and g.snapshot()["unknown_cost_requests"] == 2
    g.close()


def test_extra_limit_waits_and_unknown_hold_survive_roles_and_reopening(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    g, clock = guarded(path); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok", "known429", "ok", "known429"], clock)
    assert call(g, c) == "local fixture"
    held = g.snapshot()["reserved_cost_usd"]
    g.close()
    g, _ = guarded(path, clock=clock); c = client("reflection"); g.attach(c)
    assert call(g, c, "next-stage") == "local fixture"
    state, summary = checked(g, True)
    assert summary["logical_calls_used"] == 2 and summary["extra_dispatches"] == 2
    assert state["attempts"][-1]["recovery_wait_seconds"] == 120
    assert state["reserved_cost_usd"] == held and state["unknown_cost_requests"] == 1
    c = client("judge"); g.attach(c)
    with pytest.raises(MinimalCallStopped):
        call(g, c, "judge-position")
    assert len(sent) == 5
    g.close()


@pytest.mark.parametrize("failure", ["timeout", "bad_json", "wrong_provider", "has_answer", "http500", "no_error_code", "duplicate", "oversized"])
def test_only_complete_matching_upstream_no_answer_429_can_recover(tmp_path, monkeypatch, failure):
    g, clock = guarded(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    sent = scripted(monkeypatch, [failure, "ok"], clock)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state, summary = checked(g)
    assert len(sent) == 1 and state["stopped"]
    assert summary["extra_dispatches"] == 0 and summary["relations"] == []
    g.close()


def test_each_additional_attempt_needs_an_additional_budget_reservation(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3", budget="0.0004"); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok"], clock)
    with pytest.raises(MinimalCallStopped, match="budget"):
        call(g, c)
    state = g.snapshot()
    assert len(sent) == state["calls_used"] == 1 and state["unknown_cost_requests"] == 1
    assert state["stopped"] and Decimal(state["reserved_cost_usd"]) > 0
    assert state["recovery"]["relations"][0]["child_request_id"] is None
    g.close()


@pytest.mark.parametrize("damage", ["cancel", "configuration", "policy", "save"])
def test_wait_cancellation_configuration_change_and_save_failure_never_dispatch_again(tmp_path, monkeypatch, damage):
    g, clock = guarded(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok"], clock)
    sleep = g._pacing_sleep
    def interrupt(delay):
        assert delay <= 1
        sleep(delay)
        if damage == "cancel":
            g.halt()
        elif damage == "configuration":
            c.temperature = .8
        elif damage == "policy":
            g.recovery_policy["max_extra_dispatches"] = 3
    if damage == "save":
        monkeypatch.setattr(g, "_finish", lambda row: (_ for _ in ()).throw(OSError("PRIVATE save")))
    else:
        g._pacing_sleep = interrupt
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    assert len(sent) == 1 and g.snapshot()["stopped"]
    g.close()


def test_wait_interrupted_by_new_guard_is_not_automatically_resumed(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    g, clock = guarded(path); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok"], clock)
    def crash(_):
        raise KeyboardInterrupt()
    g._pacing_sleep = crash
    with pytest.raises(KeyboardInterrupt):
        call(g, c)
    g.close()
    g, _ = guarded(path, clock=clock); c = client(); g.attach(c)
    assert g.snapshot()["stop_reason"] == "recovery_wait_interrupted"
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    assert len(sent) == 1
    g.close()


def test_default_guard_does_not_gain_recovery_or_migrate_retained_ledger(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    g, clock = guarded(path, recovery=False); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["429", "ok"], clock)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    assert len(sent) == 1 and "recovery" not in g.snapshot()
    assert "physical_attempt" not in g.snapshot()["attempts"][0]
    g.close()
    with pytest.raises(ValueError, match="no_recovery_policy"):
        guarded(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='bounded_recovery_policy'").fetchone() is None


def test_readonly_projection_rejects_parent_status_wire_and_physical_count_tampering(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    scripted(monkeypatch, ["known429", "ok"], clock)
    call(g, c)
    state, _ = checked(g, True)
    for changes in ({"request_sha256": "b" * 64}, {"physical_attempt": 1}, {"http_dispatch_started": False},
                    {"recovery_wait_seconds": 1}, {"session_id": "another"}, {"e1b_stage": "another"}):
        rows = deepcopy(state["attempts"])
        rows[-1].update(changes)
        with pytest.raises(ValueError):
            validate_recovery_records(rows, g.recovery_policy, relations=state["recovery"]["relations"], require_complete=True)
    g.close()


def test_last_logical_position_can_recover_but_cannot_create_position_1637(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["ok", "429", "ok"], clock)
    assert call(g, c) == "local fixture"
    template = g.snapshot()["attempts"][0]
    # Fill already completed logical positions with explicit software records;
    # only the final position below exercises the live guard/fake transport.
    with g._transaction():
        for number in range(2, 1636):
            row = deepcopy(template)
            row.update(request_id=f"fixture{number}", operation_id=f"op{number}", logical_operation_id=f"op{number}")
            g._db.execute("INSERT INTO attempts(request_id,record) VALUES(?,?)", (row["request_id"], json.dumps(row)))
    assert call(g, c, "last-position") == "local fixture"
    state, summary = checked(g, True)
    assert summary["logical_received"] == summary["logical_calls_used"] == 1636
    assert summary["extra_dispatches"] == 1 and state["calls_used"] == 1637
    assert state["stop_reason"] == "logical_request_limit_reached"
    with pytest.raises(MinimalCallStopped):
        call(g, c, "extra-position")
    assert len(sent) == 3
    g.close()
