"""Pre-send hard failures use only synthetic input and a denied network."""
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.product_calls import ProductCallGuard
from aria.runtime import create_minimal_clients
from aria.voice.minimal import MinimalWebApp
from tests.test_reserved_calls import client, policy, receipt, stub


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))


def guard(path, cls=ProductCallGuard, *, limit=31999):
    p = policy()
    for model, spec in p["models"].items():
        if model.startswith("qwen/"):
            spec["max_input_tokens"] = limit
        spec["max_output_tokens"] = 128000
    return cls(path, max_requests=40, max_observed_cost_usd="1", reservation_policy=p)


def request(c, monkeypatch, *, bound=None):
    captured = []
    with monkeypatch.context() as patch:
        patch.setattr(c, "_request_json", lambda value: (captured.append(value), receipt(c))[1])
        c.complete([{"role": "user", "content": ""}])
        if bound is not None:
            count = bound - len(captured[-1].data) - 8192
            assert count >= 0
            c.complete([{"role": "user", "content": "x" * count}])
    return captured[-1]


def test_exact_input_limit_is_allowed_and_one_byte_over_is_hard_before_claim(tmp_path, monkeypatch):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    accepted = request(c, monkeypatch, bound=31999)
    calls = stub(monkeypatch, receipt(c))
    with g.scope("turn", "at-limit", "synthetic"):
        g.request_json(c, accepted)
    original = deepcopy(g.snapshot()["attempts"])
    rejected = request(c, monkeypatch, bound=32000)
    with pytest.raises(MinimalCallStopped, match="local_request_rejected"):
        with g.scope("turn", "over-limit", "synthetic"):
            g.request_json(c, rejected)
    state = g.snapshot()
    assert len(calls) == 1 and state["attempts"] == original
    assert state["stop_reason"] == "local_request_rejected" and not state["can_resume"]
    assert state["consecutive_failures"] == 0
    assert state["last_request_rejection"]["reason"] == "reservation_token_limit_exceeded"
    assert state["last_request_rejection"]["input_token_plan_bound"] == 32000
    assert state["last_request_rejection"]["request_sent"] is False
    with pytest.raises(MinimalCallStopped, match="product_resume_not_allowed"):
        g.resume("owner-restore", state["pause_id"])
    events = state["product_events"]
    g.close()
    g = guard(tmp_path / "calls.sqlite3")
    assert g.snapshot()["product_events"] == events
    assert g.snapshot()["stop_reason"] == "local_request_rejected"
    assert not g.snapshot()["can_resume"]
    g.close()


@pytest.mark.parametrize("role", ["online_fast", "memory", "reflection"])
def test_configuration_identity_mismatch_is_durable_without_attempt(tmp_path, monkeypatch, role):
    g = guard(tmp_path / "calls.sqlite3"); c = client(role); g.attach(c)
    c.temperature += .1
    req = request(c, monkeypatch)
    with pytest.raises(MinimalCallStopped, match="local_request_rejected"):
        with g.scope("turn" if role == "online_fast" else "review", "identity", "synthetic"):
            g.request_json(c, req)
    state = g.snapshot()
    assert state["calls_used"] == 0 and state["consecutive_failures"] == 0
    assert state["last_request_rejection"]["reason"] == "model_configuration_changed"
    assert c.last_diagnostic["local_rejection_code"] == "model_configuration_changed"
    assert c.last_diagnostic["error_type"] == "local_request_rejected"
    g.close()


def test_caught_duplicate_rejection_still_hard_stops_and_cannot_append_another_rejection(tmp_path, monkeypatch):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    req = request(c, monkeypatch)
    calls = stub(monkeypatch, receipt(c))
    with g.scope("turn", "duplicate", "synthetic"):
        g.request_json(c, req)
        for _ in range(2):
            with pytest.raises(MinimalCallStopped, match="local_request_rejected"):
                g.request_json(c, req)
    state = g.snapshot()
    assert len(calls) == state["calls_used"] == 1
    assert state["stop_reason"] == "local_request_rejected" and state["consecutive_failures"] == 0
    assert state["last_request_rejection"]["reason"] == "duplicate_model_operation"
    assert [e["event"] for e in state["product_events"]] == ["started", "request_rejected", "finished"]
    g.close()


@pytest.mark.parametrize("message,expected", [
    ("e1c_frozen_source_or_history_changed", "e1c_frozen_source_or_history_changed"),
    ("e1cv2_position_rebound", "e1cv2_position_rebound"),
    ("private-user-content-and-key-must-never-be-written", "local_preflight_rejected"),
])
def test_authority_and_unknown_preflight_fail_closed_with_finite_diagnostics(tmp_path, monkeypatch, message, expected):
    class Rejected(ProductCallGuard):
        def _reserve(self, *args):
            raise ValueError(message)
    path = tmp_path / "calls.sqlite3"
    g = guard(path, Rejected); c = client(); g.attach(c)
    req = request(c, monkeypatch)
    with pytest.raises(MinimalCallStopped, match="local_request_rejected"):
        with g.scope("turn", "authority", "synthetic"):
            g.request_json(c, req)
    state = g.snapshot()
    assert state["calls_used"] == 0 and not state["can_resume"]
    assert state["last_request_rejection"]["reason"] == expected
    assert "private-user-content" not in json.dumps(state)
    assert "private-user-content" not in json.dumps(c.last_diagnostic)
    g.close()
    assert b"private-user-content" not in path.read_bytes()


@pytest.mark.parametrize("append_before_failure", [False, True])
def test_stop_event_save_failure_retains_unresolved_operation_across_restart(tmp_path, monkeypatch, append_before_failure):
    path = tmp_path / "calls.sqlite3"
    g = guard(path); c = client(); g.attach(c)
    req = request(c, monkeypatch, bound=32000)
    append = g._append
    def fail_stop(event):
        if event["event"] == "request_rejected":
            if append_before_failure:
                append(event)
            raise OSError("synthetic storage failure")
        return append(event)
    monkeypatch.setattr(g, "_append", fail_stop)
    with pytest.raises(MinimalCallStopped):
        with g.scope("turn", "storage", "synthetic"):
            g.request_json(c, req)
    assert g.snapshot()["stopped"] and g.snapshot()["stop_reason"] == "ledger_unavailable"
    g.close()
    reopened = guard(path)
    assert reopened.snapshot()["stop_reason"] == "unresolved_product_operation"
    assert reopened.snapshot()["calls_used"] == 0 and not reopened.snapshot()["can_resume"]
    assert [e["event"] for e in reopened.snapshot()["product_events"]] == ["started"]
    reopened.close()


def test_real_app_displays_local_rejection_and_does_not_advance_next_turn(tmp_path):
    g = guard(tmp_path / "calls.sqlite3", limit=9000)
    clients = create_minimal_clients(model_calls_enabled=True, api_key="synthetic-inert",
                                     base_url="https://fixture.invalid/v1")
    app = MinimalWebApp(tmp_path / "app", clients=clients, model_call_guard=g)
    sid = app.start()["state"]["session_id"]
    result = app.submit("今天想聊聊手工。", "local-rejection", sid)
    row = app._run["turns"][-1]
    assert not result["ok"] and row["source"] == "local_request_rejected"
    assert row["requires_review"] is True and row["response"] == ""
    assert "未向模型发送请求" in row["error"] and "恢复按钮不能解除" in row["error"]
    assert g.snapshot()["calls_used"] == 0 and g.snapshot()["consecutive_failures"] == 0
    assert not app.submit("下一句也不能发送。", "next", sid)["ok"]
    assert len(app._run["turns"]) == 1
    app.close(); g.close()


def test_budget_refusal_is_not_a_model_failure_and_is_recorded_once(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite3"
    g = ProductCallGuard(path, max_requests=40, max_observed_cost_usd="0.000001", reservation_policy=policy())
    c = client(); g.attach(c); req = request(c, monkeypatch)
    with g.scope("turn", "budget", "synthetic"):
        for _ in range(2):
            with pytest.raises(MinimalCallStopped, match="pre_send_budget_exceeded"):
                g.request_json(c, req)
    state = g.snapshot()
    assert state["stop_reason"] == "pre_send_budget_exceeded" and not state["can_resume"]
    assert state["calls_used"] == state["consecutive_failures"] == 0
    assert [e["event"] for e in state["product_events"]] == ["started", "budget_stopped", "finished"]
    g.close()


@pytest.mark.parametrize("append_before_failure", [False, True])
def test_budget_stop_save_failure_cannot_enable_cheaper_request_after_restart(tmp_path, monkeypatch, append_before_failure):
    path = tmp_path / "budget-write.sqlite3"
    g = ProductCallGuard(path, max_requests=40, max_observed_cost_usd="0.001", reservation_policy=policy())
    c = client(); g.attach(c); req = request(c, monkeypatch, bound=50000)
    append = g._append
    def fail(event):
        if event["event"] == "budget_stopped":
            if append_before_failure:
                append(event)
            raise OSError("synthetic budget save failure")
        return append(event)
    monkeypatch.setattr(g, "_append", fail)
    with pytest.raises(MinimalCallStopped):
        with g.scope("turn", "budget", "synthetic"):
            g.request_json(c, req)
    assert g.snapshot()["stop_reason"] == "ledger_unavailable"
    g.close()
    reopened = ProductCallGuard(path, max_requests=40, max_observed_cost_usd="0.001", reservation_policy=policy())
    assert reopened.snapshot()["stop_reason"] == "unresolved_product_operation"
    assert reopened.snapshot()["calls_used"] == 0 and not reopened.snapshot()["can_resume"]
    with pytest.raises(MinimalCallStopped, match="unresolved_product_operation"):
        with reopened.scope("turn", "cheaper", "synthetic"):
            pytest.fail("cannot start even a cheaper request")
    reopened.close()
