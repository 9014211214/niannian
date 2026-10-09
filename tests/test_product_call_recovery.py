"""Product-only failure recovery, exercised without credentials or a network."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import io
import json
import sqlite3
import urllib.error
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.product_calls import ProductCallGuard
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_reserved_calls import client, policy, receipt, call, stub


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unstubbed network"))


def guard(tmp_path, cls=ProductCallGuard, *, maximum=40, budget="1"):
    g = cls(tmp_path / "calls.sqlite3", max_requests=maximum, max_observed_cost_usd=budget,
            reservation_policy=policy())
    g._pacing_sleep = lambda duration: None
    return g


class ShapeGuard(ProductCallGuard):
    def _record_receipt(self, c, row, data):
        super()._record_receipt(c, row, data)
        if data["choices"][0]["message"]["content"] == "BAD":
            row["validation_error"] = "online_decision_schema_invalid"
            raise ValueError("synthetic private error must not be retained")


def fail_shape(g, c, monkeypatch, operation):
    data = receipt(c)
    data["choices"][0]["message"]["content"] = "BAD"
    calls = stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped):
        call(g, c, operation)
    assert len(calls) == 1


def test_one_bad_shape_does_not_disable_next_turn_or_change_old_attempt(tmp_path, monkeypatch):
    g = guard(tmp_path, ShapeGuard); c = client(); g.attach(c)
    fail_shape(g, c, monkeypatch, "bad")
    state = g.snapshot()
    first = deepcopy(state["attempts"][0])
    assert not state["stopped"] and state["consecutive_failures"] == 1
    assert "synthetic private" not in json.dumps(state)
    calls = stub(monkeypatch, receipt(c))
    assert call(g, c, "good") == "local fixture"
    assert g.snapshot()["attempts"][0] == first
    assert g.snapshot()["consecutive_failures"] == 0 and len(calls) == 1
    g.close()


def test_three_logical_failures_pause_restart_and_explicit_resume_are_append_only(tmp_path, monkeypatch):
    g = guard(tmp_path, ShapeGuard); c = client(); g.attach(c)
    for i in range(3):
        fail_shape(g, c, monkeypatch, f"bad{i}")
    state = g.snapshot()
    assert state["stop_reason"] == "consecutive_failures" and state["can_resume"]
    attempts, events, pause = deepcopy(state["attempts"]), state["product_events"], state["pause_id"]
    with pytest.raises(MinimalCallStopped):
        call(g, c, "blocked")
    g.close()
    reopened = guard(tmp_path, ShapeGuard); c2 = client(); reopened.attach(c2)
    assert reopened.snapshot()["pause_id"] == pause
    with pytest.raises(MinimalCallStopped):
        reopened.resume("resume", "stale-pause")
    reopened.resume("resume", pause)
    after = reopened.snapshot()
    assert after["attempts"] == attempts and after["product_events"][:-1] == events
    assert not after["stopped"] and after["consecutive_failures"] == 0
    reopened.resume("resume", pause)
    assert reopened.snapshot() == after
    stub(monkeypatch, receipt(c2)); assert call(reopened, c2, "next") == "local fixture"
    assert reopened.snapshot()["known_cost_usd"] == "0.00004"
    reopened.close()


def http_failure(c, status, *, cost=0, answer=False, raw=None):
    data = {"error": {"code": status}, "usage": {"cost": cost}}
    if cost is None:
        data.pop("usage")
    if answer:
        data["choices"] = [{"message": {"content": "already received"}}]
    return urllib.error.HTTPError(c.base_url, status, "synthetic message", {},
                                  io.BytesIO(raw if raw is not None else json.dumps(data).encode()))


@pytest.mark.parametrize("status", [429, 500, 503, 599])
def test_known_no_content_retries_twice_at_most_and_counts_as_one_failed_round(tmp_path, monkeypatch, status):
    g = guard(tmp_path); c = client(); g.attach(c)
    sent = []
    def fail(request, **kwargs):
        sent.append(request.data)
        raise http_failure(c, status)
    stub_model_http(monkeypatch, fail)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot()
    assert len(sent) == 3 and len(set(sent)) == 1
    assert state["consecutive_failures"] == 1 and not state["stopped"]
    assert state["known_cost_usd"] == "0" and state["unknown_cost_requests"] == 0
    assert len({r["request_id"] for r in state["attempts"]}) == 3
    assert all("reservation" in r for r in state["attempts"])
    assert [r["product_transport_attempt"] for r in state["attempts"]] == [1, 2, 3]
    assert state["attempts"][1]["product_retry_parent_request_id"] == state["attempts"][0]["request_id"]
    g.close()


def test_retry_then_success_resets_failure_streak(tmp_path, monkeypatch):
    g = guard(tmp_path); c = client(); g.attach(c)
    sent = []
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, size): return json.dumps(receipt(c)).encode()
    def receive(request, **kwargs):
        sent.append(1)
        if len(sent) < 3: raise http_failure(c, 503)
        return Response()
    stub_model_http(monkeypatch, receive)
    assert call(g, c) == "local fixture"
    assert len(sent) == 3 and g.snapshot()["consecutive_failures"] == 0
    assert not g.snapshot()["stopped"]
    g.close()


@pytest.mark.parametrize("kwargs", [{"cost": None}, {"answer": True}, {"raw": b""},
                                    {"raw": b'{"error":'}, {"cost": 99}])
def test_unknown_cost_content_or_incomplete_error_is_never_retried_or_resumable(tmp_path, monkeypatch, kwargs):
    g = guard(tmp_path); c = client(); g.attach(c); calls = []
    def fail(request, **unused):
        calls.append(1); raise http_failure(c, 429, **kwargs)
    stub_model_http(monkeypatch, fail)
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(calls) == 1 and g.snapshot()["stopped"] and not g.snapshot()["can_resume"]
    before = g.snapshot()["attempts"]
    with pytest.raises(MinimalCallStopped): g.resume("no", g.snapshot()["pause_id"])
    assert g.snapshot()["attempts"] == before
    g.close()


@pytest.mark.parametrize("maximum,budget,reason", [(1, "1", "request_limit_reached"),
                                                 (40, "0.000001", "pre_send_budget_exceeded")])
def test_budget_and_request_stops_cannot_be_resumed(tmp_path, monkeypatch, maximum, budget, reason):
    g = guard(tmp_path, maximum=maximum, budget=budget); c = client(); g.attach(c)
    calls = stub(monkeypatch, receipt(c))
    if maximum == 1:
        call(g, c)
    else:
        with pytest.raises(MinimalCallStopped): call(g, c)
    assert g.snapshot()["stop_reason"] == reason
    with pytest.raises(MinimalCallStopped): g.resume("attempt", g.snapshot()["pause_id"])
    assert len(calls) == (1 if maximum == 1 else 0)
    g.close()


def test_persisted_inflight_operation_blocks_other_guard_and_restart(tmp_path):
    g = guard(tmp_path); other = guard(tmp_path)
    with g.scope("turn", "inflight", "session"):
        assert not g.snapshot()["stopped"]
        assert other.snapshot()["stop_reason"] == "unresolved_product_operation"
        with pytest.raises(MinimalCallStopped):
            with other.scope("turn", "other", "session"): pytest.fail("concurrent scope")
    assert not other.snapshot()["stopped"]
    g.close(); other.close()
    with sqlite3.connect(tmp_path / "calls.sqlite3") as db:
        db.execute("INSERT INTO product_events(event) VALUES(?)", (json.dumps({"event": "started",
            "at": "2026-10-06", "token": "interrupted", "kind": "turn", "operation_id": "lost", "session_id": "session"}),))
    reopened = guard(tmp_path)
    assert reopened.snapshot()["stop_reason"] == "unresolved_product_operation"
    with pytest.raises(MinimalCallStopped): reopened.resume("no", None)
    reopened.close()


def test_resume_write_failure_is_fail_closed_and_retains_history(tmp_path, monkeypatch):
    g = guard(tmp_path, ShapeGuard); c = client(); g.attach(c)
    for i in range(3): fail_shape(g, c, monkeypatch, f"bad{i}")
    state = g.snapshot()
    monkeypatch.setattr(g, "_append", lambda event: (_ for _ in ()).throw(OSError("private storage path")))
    with pytest.raises(MinimalCallStopped, match="product_resume_save_failed"):
        g.resume("resume", state["pause_id"])
    assert g.snapshot()["stop_reason"] == "ledger_unavailable"
    assert g.snapshot()["attempts"] == state["attempts"]
    assert g.snapshot()["product_events"] == state["product_events"]
    g.close()


def test_legacy_strict_ledger_is_not_silently_upgraded(tmp_path, monkeypatch):
    legacy = guard(tmp_path, ReservedCallGuard); c = client(); legacy.attach(c)
    stub(monkeypatch, receipt(c)); call(legacy, c); legacy.close()
    before = (tmp_path / "calls.sqlite3").read_bytes()
    with pytest.raises(ValueError, match="retained_product_recovery_contract_missing"):
        guard(tmp_path)
    assert (tmp_path / "calls.sqlite3").read_bytes() == before


def test_durable_reads_and_replay_do_not_change_events_or_fees(tmp_path, monkeypatch):
    from tests.test_minimal_call_integration import app_at, responses, Response, DECISION
    app = app_at(tmp_path)
    calls = responses(monkeypatch, Response("{bad", {"cost": .01}), Response(DECISION, {"cost": .01}))
    sid = app.start()["state"]["session_id"]
    first = app.submit("今天没什么事", "bad", sid)
    assert not first["ok"] and first["state"]["turns"][0]["source"] == "rule_fallback"
    assert first["state"]["turns"][0]["response"] and not first["state"]["requires_review"]
    frozen = app.call_guard.path.read_bytes()
    for _ in range(3): app.state()
    assert not app.resume_models("replay", None, confirmed=True, read_only=True)["ok"]
    assert app.call_guard.path.read_bytes() == frozen and len(calls) == 1
    assert app.submit("还有纸风车", "good", sid)["ok"] and len(calls) == 2
    assert app.state()["model_usage"]["consecutive_failures"] == 0
    app.close()


def test_product_owner_resume_is_explicit_idempotent_and_no_model_request(tmp_path, monkeypatch):
    from tests.test_minimal_call_integration import app_at, responses, Response
    app = app_at(tmp_path); sid = app.start()["state"]["session_id"]
    calls = responses(monkeypatch, *(Response("{bad", {"cost": .01}) for _ in range(3)))
    for i in range(3): assert not app.submit("说一句话", f"bad{i}", sid)["ok"]
    state = app.state(); pause = state["model_usage"]["pause_id"]
    assert state["model_usage"]["can_resume"] and not state["requires_review"]
    assert not app.resume_models("resume", pause)["ok"]
    with pytest.raises(PermissionError): app.resume_models("resume", pause, confirmed=True, actor="caregiver")
    assert not app.resume_models("resume", pause, confirmed=True, read_only=True)["ok"]
    assert app.resume_models("resume", pause, confirmed=True)["ok"]
    assert app.resume_models("resume", pause, confirmed=True)["ok"]
    assert len(calls) == 3
    assert len(app.state()["model_usage"]["product_events"]) == 7
    app.close()


def test_http_resume_requires_local_owner_context_and_is_inert_during_replay(tmp_path, monkeypatch):
    import threading
    from aria.voice.minimal import make_minimal_server
    from tests.test_minimal_care_web import call as http_call
    from tests.test_minimal_call_integration import app_at, responses, Response
    app = app_at(tmp_path); sid = app.start()["state"]["session_id"]
    calls = responses(monkeypatch, *(Response("{bad", {"cost": .01}) for _ in range(3)))
    for i in range(3): app.submit("慢慢说", f"bad{i}", sid)
    state = app.state(); frozen = app.call_guard.path.read_bytes()
    payload = {"request_id": "resume", "pause_id": state["model_usage"]["pause_id"], "confirmed": True}
    server = make_minimal_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        assert http_call(app, server, "/api/minimal/resume-models", data=payload, token=False)[0] == 403
        assert http_call(app, server, "/api/minimal/resume-models", data=payload, origin="https://evil.invalid")[0] == 403
        assert http_call(app, server, "/api/minimal/resume-models", data={**payload, "actor": "caregiver"})[0] == 403
        assert http_call(app, server, "/api/minimal/resume-models", data={**payload, "replay": True})[0] == 409
        assert app.call_guard.path.read_bytes() == frozen
        assert http_call(app, server, "/api/minimal/resume-models", data=payload)[0] == 200
        current = app.call_guard.path.read_bytes()
        assert http_call(app, server, "/api/minimal/resume-models", data=payload)[0] == 200
        assert http_call(app, server, "/api/minimal/state")[0] == 200
        assert app.call_guard.path.read_bytes() == current and len(calls) == 3
    finally:
        server.shutdown(); thread.join(timeout=3); server.server_close(); app.close()


def test_operation_outcome_save_failure_blocks_even_after_process_reopen(tmp_path, monkeypatch):
    g = guard(tmp_path); c = client(); g.attach(c); stub(monkeypatch, receipt(c))
    original = g._append
    def fail_finish(event):
        if event["event"] == "finished": raise OSError("synthetic storage failure")
        original(event)
    monkeypatch.setattr(g, "_append", fail_finish)
    with pytest.raises(MinimalCallStopped, match="product_outcome_save_failed"): call(g, c)
    assert g.snapshot()["stop_reason"] == "ledger_unavailable"
    g.close()
    reopened = guard(tmp_path)
    assert reopened.snapshot()["stop_reason"] == "unresolved_product_operation"
    assert reopened.snapshot()["calls_used"] == 1
    with pytest.raises(MinimalCallStopped): reopened.resume("no", None)
    reopened.close()


def test_received_result_save_failure_keeps_pending_reservation_and_no_retry(tmp_path, monkeypatch):
    g = guard(tmp_path); c = client(); g.attach(c); calls = stub(monkeypatch, receipt(c))
    monkeypatch.setattr(g, "_finish", lambda row: (_ for _ in ()).throw(OSError("synthetic store failed")))
    with pytest.raises(MinimalCallStopped, match="ledger_result_save_failed"): call(g, c)
    assert len(calls) == 1 and g.snapshot()["attempts"][0]["status"] == "pending"
    assert g.snapshot()["unknown_cost_requests"] == 1 and not g.snapshot()["can_resume"]
    g.close()
    reopened = guard(tmp_path)
    assert reopened.snapshot()["stop_reason"] == "unresolved_request"
    reopened.close()


def test_failed_reflection_is_not_reported_as_success_or_permanent_online_halt(tmp_path, monkeypatch):
    from tests.test_minimal_call_integration import app_at, responses, Response, DECISION
    app = app_at(tmp_path); sid = app.start()["state"]["session_id"]
    calls = responses(monkeypatch, Response(DECISION, {"cost": .01}), Response(DECISION, {"cost": .01}),
                      Response(DECISION, {"cost": .01}),
                      Response('{"episodes":[],"relationships":[],"cognitive_privacy":[]}', {"cost": .02}),
                      Response('{"reviews":[]}', {"cost": .01}), Response('{"reviews":[]}', {"cost": .01}))
    assert app.submit("先聊聊", "first", sid)["ok"]
    assert app.submit("还有以前的故事", "second", sid)["ok"]
    assert app.submit("今天先说到这里", "third", sid)["ok"]
    app.seal(sid)
    result = app.reflect("reflection-failed", wait=True)
    assert not result["ok"] and result["state"]["reviews"][0]["status"] == "failed"
    assert not result["state"]["model_usage"]["stopped"]
    assert result["state"]["model_usage"]["consecutive_failures"] == 1
    assert result["state"]["offline"]["pending_count"] == 0
    assert len(calls) == 6
    assert app.reflect("reflection-failed", wait=True)["duplicate"] and len(calls) == 6
    app.close()


def test_success_does_not_clear_unknown_budget_halt(tmp_path, monkeypatch):
    g = guard(tmp_path); c = client(); g.attach(c)
    stub(monkeypatch, receipt(c, usage={"prompt_tokens": 4, "completion_tokens": 2}))
    with pytest.raises(MinimalCallStopped): call(g, c, "unknown")
    before = g.snapshot()
    assert before["stop_reason"] == "provider_cost_unknown" and not before["can_resume"]
    for _ in range(3):
        with pytest.raises(MinimalCallStopped): g.resume("restore", before["pause_id"])
    assert before == g.snapshot()
    g.close()


def test_settled_bad_memory_discards_candidate_skips_reflection_and_drains_next_session():
    from tests.test_offline_queue import setup, context, memory, reflection
    from aria.engine.model.persona import EpisodicMemoryEntry
    queue, store, _ = setup()
    queue.enqueue(context("bad-memory")); queue.enqueue(context("next-session"))
    reflected = []
    def bad(persona, ctx):
        persona.episodic_memory.append(EpisodicMemoryEntry(id="must-not-save", summary="discard candidate"))
        persona.disc.dominance = .99
        return persona, "sealed_failed", "memory_extraction_failed"
    def reflect(persona, ctx):
        reflected.append(ctx.session_id); return reflection(persona, ctx)
    first = queue.drain(bad, reflect)
    assert first["completed_jobs"] == 1 and first["pending_count"] == 1
    audit = first["audit"][0]
    assert audit["outcome"] == "memory_failed" and audit["gap"]
    assert not audit["memory_completed"] and not audit["reflection_completed"]
    assert not queue.persona.episodic_memory and queue.persona.disc.dominance != .99 and not reflected
    second = queue.drain(memory, reflect)
    assert second["pending_count"] == 0 and reflected == ["next-session"]
    assert second["audit"][0] == audit
    assert [e.id for e in store.saved.episodic_memory] == ["next-session"]


def test_memory_failed_seal_storage_error_preserves_pending_job_and_does_not_run_reflection():
    from tests.test_offline_queue import setup, context
    queue, store, _ = setup(fail_on=(3,))
    queue.enqueue(context())
    def bad(persona, ctx): return persona, "sealed_failed", "memory_extraction_failed"
    result = queue.drain(bad, lambda *_: pytest.fail("unscreened reflection"))
    assert result["error_code"] == "offline_store_failed" and result["completed_jobs"] == 0
    assert result["pending_count"] == 1 and result["audit"] == []
    assert not queue.persona.offline_sessions[0].memory_completed


def test_unknown_memory_fee_keeps_queue_stopped_and_never_enters_reflection(tmp_path, monkeypatch):
    from tests.test_minimal_call_integration import app_at, responses, Response, DECISION
    app = app_at(tmp_path); sid = app.start()["state"]["session_id"]
    calls = responses(monkeypatch, Response(DECISION, {"cost": .01}), Response('{}', {}))
    assert app.submit("先说一句", "first", sid)["ok"]
    app.seal(sid)
    result = app.reflect("unknown-memory", wait=True)
    assert not result["ok"] and result["state"]["model_usage"]["stopped"]
    assert result["state"]["offline"]["pending_count"] == 1 and not result["state"]["offline"]["audit"]
    assert len(calls) == 2 and result["state"]["requires_review"]
    app.close()


def test_reflection_transport_retries_and_one_business_correction_are_separate(tmp_path, monkeypatch):
    from aria.engine.policy import reflection_json
    g = guard(tmp_path); c = client("reflection"); c.reflection_return_format = "body_json"; g.attach(c)
    sent = []
    class Response:
        status, headers = 200, {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, size): return json.dumps(self.value).encode()
    def receive(request, **kwargs):
        sent.append(request.data)
        if len(sent) in {1, 2, 4, 5}: raise http_failure(c, 503)
        value = receipt(c)
        value["choices"][0]["message"]["content"] = json.dumps({
            "protocol_version": "2.0", "reviews": "bad" if len(sent) == 3 else [], "changes": []})
        return Response(value)
    stub_model_http(monkeypatch, receive)
    messages = [{"role": "system", "content": reflection_json.reflection_body_system_prompt()},
                {"role": "user", "content": "unchanged local source"}]
    with g.scope("review", "review", "session"):
        c.complete(messages)
        initial = g.snapshot()["attempts"][-1]
        correction = [*messages, reflection_json.reflection_correction_message(
            initial["reflection_body"]["correction_issues"])]
        assert json.loads(c.complete(correction))["reviews"] == []
    state = g.snapshot()
    assert len(sent) == 6 and not state["stopped"] and state["consecutive_failures"] == 0
    assert len(set(sent[:3])) == len(set(sent[3:])) == 1 and sent[0] != sent[3]
    assert [r["product_transport_attempt"] for r in state["attempts"]] == [1, 2, 3, 1, 2, 3]
    assert all(r["format_parent_request_id"] == initial["request_id"] for r in state["attempts"][3:])
    g.close()


def test_settled_no_content_memory_failure_seals_and_next_session_can_review(tmp_path, monkeypatch):
    from tests.test_minimal_call_integration import app_at, responses, Response, DECISION
    app = app_at(tmp_path); app.call_guard._pacing_sleep = lambda _: None
    sid = app.start()["state"]["session_id"]
    responses(monkeypatch, Response(DECISION, {"cost": .001}))
    assert app.submit("普通会话", "first", sid)["ok"]
    app.seal(sid)
    physical = []
    def fail(request, **kwargs):
        physical.append(request.data)
        raise http_failure(app.clients.memory, 503)
    stub_model_http(monkeypatch, fail)
    first = app.reflect("memory-unavailable", wait=True)
    assert not first["ok"] and len(physical) == 3
    assert first["state"]["offline"]["pending_count"] == 0
    assert first["state"]["offline"]["audit"][-1]["outcome"] == "memory_failed"
    assert not first["state"]["model_usage"]["stopped"]
    old = deepcopy(app.call_guard.snapshot()["attempts"])
    # Reopening cannot resurrect the failed memory or poison the FIFO queue.
    app.close(); app = app_at(tmp_path)
    sid = app.start()["state"]["session_id"]
    calls = responses(monkeypatch, *(Response(DECISION, {"cost": .001}) for _ in range(3)),
        Response('{"episodes":[],"relationships":[],"stated_preferences":[],"cognitive_privacy":[]}', {"cost": .001}),
        Response('{"protocol_version":"2.0","reviews":[],"changes":[]}', {"cost": .001}))
    for number in range(3): assert app.submit("再聊一点普通的事", f"next{number}", sid)["ok"]
    app.seal(sid)
    result = app.reflect("next-review", wait=True)
    assert result["ok"] and result["state"]["offline"]["pending_count"] == 0
    assert app.call_guard.snapshot()["attempts"][:len(old)] == old
    assert len(calls) == 5 and result["state"]["model_usage"]["consecutive_failures"] == 0
    app.close()
