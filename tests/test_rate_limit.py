"""Real guard integration with synthetic HTTP and virtual clocks, no network."""
from copy import deepcopy
from email.message import Message
import io
import json
import sqlite3
import urllib.error

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.product_calls import ProductCallGuard
from aria.engine.policy.response_observability import observe_http_error, validate_observation
from tests._model_http_stub import stub_model_http
from tests.test_reserved_calls import client, policy, receipt, call


class Clock:
    def __init__(self): self.now = 0.0; self.sleeps = []
    def time(self): return self.now
    def sleep(self, seconds): self.sleeps.append(seconds); self.now += seconds


def guard(path, clock):
    g = ProductCallGuard(path, max_requests=40, max_observed_cost_usd="1", reservation_policy=policy())
    g._pacing_clock, g._pacing_sleep = clock.time, clock.sleep
    return g


def failure(*, retry="5", cost=0, metadata=None, status=429):
    headers = Message()
    if retry is not None: headers["Retry-After"] = retry
    data = {"error": {"code": status, "message": "PRIVATE", "metadata": metadata or {}}}
    if cost is not None: data["usage"] = {"cost": cost}
    return urllib.error.HTTPError("https://example.invalid", status, "PRIVATE", headers,
                                 io.BytesIO(json.dumps(data).encode()))


@pytest.mark.parametrize("value,status", [("openrouter_in_flight_budget", "recognized"),
    ("openrouter_credits", "recognized"), ("openrouter_key_limit", "recognized"),
    ("openrouter_guardrail_budget", "recognized"), ("openrouter_workspace_budget", "recognized"),
    (None, "null"), (True, "wrong_type"), ({"private": "x"}, "wrong_type"), ("PRIVATE", "unrecognized")])
def test_finite_router_metadata_is_observed_not_a_cost_or_retry_permission(value, status):
    obs, usage = observe_http_error(failure(cost=None, metadata={"limit_source": value,
        "error_type": "provider_overloaded", "remedy_hint": "PRIVATE"}))
    assert obs["version"] == "1.5" and obs["limit_source_status"] == status
    assert obs["router_error_type"] == "provider_overloaded"
    assert ("limit_source" in obs) == (status == "recognized")
    assert not usage and "PRIVATE" not in json.dumps(obs)
    validate_observation(obs)
    wrong = deepcopy(obs); wrong["limit_source"] = "PRIVATE"
    with pytest.raises(ValueError): validate_observation(wrong)


def test_duplicate_json_and_unparseable_body_never_become_metadata():
    e = urllib.error.HTTPError("https://example.invalid", 429, "PRIVATE", {},
        io.BytesIO(b'{"error":{"code":429,"metadata":{"limit_source":"openrouter_credits","limit_source":"PRIVATE"}}}'))
    obs, usage = observe_http_error(e)
    assert obs["body_status"] == "parse_failed" and obs["limit_source_status"] == "absent"
    assert obs["router_error_type_status"] == "absent" and not usage
    validate_observation(obs)


def test_legacy_13_records_remain_valid_without_new_fields():
    obs, _ = observe_http_error(failure(metadata={"provider_name":"Alibaba"}))
    obs["version"] = "1.3"
    for field in ("limit_source_status", "router_error_type_status"): obs.pop(field)
    before = deepcopy(obs); validate_observation(obs); assert obs == before


def test_guard_waits_through_early_wakes_then_resends_identical_wire(tmp_path, monkeypatch):
    clock = Clock(); g = guard(tmp_path / "calls.sqlite3", clock); c = client(); g.attach(c)
    starts, wires = [], []
    class Response:
        status, headers = 200, {}
        def __enter__(self): return self
        def __exit__(self, *unused): return False
        def read(self, size): return json.dumps(receipt(c)).encode()
    def opened(request, **unused):
        starts.append(clock.now); wires.append(request.data)
        if len(starts) == 1: raise failure()
        return Response()
    stub_model_http(monkeypatch, opened)
    g._pacing_sleep = lambda seconds: clock.sleep(min(seconds, .25))
    assert call(g, c) == "local fixture"
    state = g.snapshot()
    assert starts == [0, 5] and wires[0] == wires[1] and len(clock.sleeps) == 20
    assert [x["event"] for x in state["product_retry_waits"]] == ["planned", "completed"]
    assert state["product_retry_waits"][1]["wait_ms"] >= 5000
    original = deepcopy(state["attempts"]); g.close()
    reopened = guard(tmp_path / "calls.sqlite3", Clock())
    assert reopened.snapshot()["attempts"] == original and not reopened.snapshot()["stopped"]
    reopened.close()


@pytest.mark.parametrize("mode", ["cancel", "frozen", "backward", "crash", "store_failed"])
def test_failed_or_interrupted_wait_never_sends_again_and_restart_stays_blocked(tmp_path, monkeypatch, mode):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); c = client(); g.attach(c)
    sends = []
    def opened(request, **unused): sends.append(1); raise failure()
    stub_model_http(monkeypatch, opened)
    def sleep(seconds):
        if mode == "cancel": g._stop_requested.set(); clock.sleep(seconds)
        elif mode == "backward": clock.now -= 1
        elif mode == "crash": raise KeyboardInterrupt()
    if mode != "store_failed": g._pacing_sleep = sleep
    else:
        original = g._append_retry_wait
        def fail_save(event):
            if event["event"] == "completed": raise OSError("PRIVATE")
            original(event)
        g._append_retry_wait = fail_save
    with pytest.raises((MinimalCallStopped, KeyboardInterrupt)): call(g, c)
    assert sends == [1]
    first = deepcopy(g.snapshot()["attempts"])
    g.close(); reopened = guard(path, Clock())
    state = reopened.snapshot()
    assert state["stopped"] and not state["can_resume"] and state["attempts"] == first
    assert "PRIVATE" not in json.dumps(state)
    c2 = client(); reopened.attach(c2)
    with pytest.raises(MinimalCallStopped): call(reopened, c2, "new")
    assert sends == [1]; reopened.close()


@pytest.mark.parametrize("retry", ["31", "999999", "PRIVATE"])
def test_unavailable_wait_not_shortened_or_bypassed_by_new_operation(tmp_path, monkeypatch, retry):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); c = client(); g.attach(c)
    sends = []
    def opened(request, **unused): sends.append(1); raise failure(retry=retry)
    stub_model_http(monkeypatch, opened)
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert sends == [1] and not clock.sleeps
    waits = g.snapshot()["product_retry_waits"]
    assert len(waits) == 1 and waits[0]["plan"]["decision"] == "blocked"
    g.close(); reopened = guard(path, Clock()); assert reopened.snapshot()["stopped"]; reopened.close()


def test_unknown_cost_still_holds_one_attempt_and_never_starts_wait(tmp_path, monkeypatch):
    clock = Clock(); g = guard(tmp_path / "calls.sqlite3", clock); c = client(); g.attach(c); sends = []
    def opened(request, **unused):
        sends.append(1); raise failure(cost=None, metadata={"limit_source":"openrouter_in_flight_budget"})
    stub_model_http(monkeypatch, opened)
    with pytest.raises(MinimalCallStopped): call(g, c)
    state = g.snapshot()
    assert sends == [1] and not clock.sleeps and not state["product_retry_waits"]
    assert state["unknown_cost_requests"] == 1 and float(state["reserved_cost_usd"]) > 0
    assert state["stopped"] and not state["can_resume"]
    g.close()


def test_wait_plan_persistence_failure_keeps_operation_unfinished(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); c = client(); g.attach(c)
    sends = []
    def opened(request, **unused): sends.append(1); raise failure()
    stub_model_http(monkeypatch, opened)
    g._append_retry_wait = lambda event: (_ for _ in ()).throw(OSError("PRIVATE"))
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert sends == [1] and not clock.sleeps
    g.close(); reopened = guard(path, Clock())
    assert reopened.snapshot()["stop_reason"] == "unresolved_product_operation"
    assert len(reopened.snapshot()["product_events"]) == 1
    reopened.close()


def test_completed_wait_tampering_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); c = client(); g.attach(c)
    stub_model_http(monkeypatch, lambda request, **unused: (_ for _ in ()).throw(failure()))
    with pytest.raises(MinimalCallStopped): call(g, c)
    with sqlite3.connect(path) as db:
        row = json.loads(db.execute("SELECT event FROM product_retry_waits WHERE sequence=2").fetchone()[0])
        row["wait_ms"] = 0
        db.execute("UPDATE product_retry_waits SET event=? WHERE sequence=2", (json.dumps(row),))
    assert g.snapshot()["stop_reason"] == "ledger_unavailable"
    g.close()


def test_wait_rechecks_external_durable_halt_before_next_claim(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); c = client(); g.attach(c)
    sends = []
    def opened(request, **unused): sends.append(1); raise failure()
    stub_model_http(monkeypatch, opened)
    def sleep(seconds):
        with sqlite3.connect(path, timeout=.1) as db:
            db.execute("UPDATE settings SET halted='model_or_storage_failed'")
        clock.sleep(seconds)
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert sends == [1] and g.snapshot()["stopped"]
    assert [x["event"] for x in g.snapshot()["product_retry_waits"]] == ["planned"]
    g.close()


def test_concurrent_second_guard_cannot_bypass_in_progress_wait(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"; clock = Clock(); g = guard(path, clock); other = guard(path, Clock())
    c = client(); second = client(); g.attach(c); other.attach(second); sends = []
    def opened(request, **unused): sends.append(1); raise failure()
    stub_model_http(monkeypatch, opened)
    def sleep(seconds):
        assert other.snapshot()["stopped"]
        with pytest.raises(MinimalCallStopped): call(other, second, "parallel")
        clock.sleep(seconds)
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(sends) == 3 and len(g.snapshot()["attempts"]) == 3
    assert len(g.snapshot()["product_retry_waits"]) == 4
    g.close(); other.close()


def test_clock_progress_between_reads_does_not_understate_completed_wait(tmp_path, monkeypatch):
    class AdvancingClock(Clock):
        def time(self):
            self.now += .125
            return self.now
    clock = AdvancingClock()
    path = tmp_path / "calls.sqlite3"
    g = guard(path, clock)
    c = client()
    g.attach(c)
    wires = []
    class Response:
        status, headers = 200, {}
        def __enter__(self): return self
        def __exit__(self, *unused): return False
        def read(self, size): return json.dumps(receipt(c)).encode()
    def opened(request, **unused):
        wires.append(request.data)
        if len(wires) == 1:
            raise failure(retry="2")
        return Response()
    stub_model_http(monkeypatch, opened)
    assert call(g, c) == "local fixture"
    state = g.snapshot()
    assert not state["stopped"] and len(wires) == 2 and wires[0] == wires[1]
    assert state["product_retry_waits"][1]["wait_ms"] >= 2000
    original = deepcopy(state["attempts"])
    g.close()
    reopened = guard(path, Clock())
    assert not reopened.snapshot()["stopped"]
    assert reopened.snapshot()["attempts"] == original
    reopened.close()


def test_fractional_clock_origin_records_full_proven_duration():
    from aria.engine.policy.rate_limit import perform_retry_wait
    clock = Clock()
    clock.now = .3
    result = perform_retry_wait({"decision": "wait", "seconds": 2},
        monotonic=clock.time, sleep=clock.sleep, cancelled=lambda: False)
    assert clock.now >= .3 + 2
    assert result >= 2000
