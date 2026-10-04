"""Approved retries retain every dispatch, first body and inclusive reservation."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import io
import json
import sqlite3
import urllib.error

import pytest

from aria.engine.policy import minimal_calls
from aria.engine.policy.e1br_calls import E1BRCallGuard, logical_attempts
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.model.persona import PersonaProfile
from tests.test_reserved_calls import policy, client, receipt
from tests.test_reflection_evidence_review import context


def guard(path, *, budget="1", authority=lambda: None):
    g = E1BRCallGuard(path, max_observed_cost_usd=budget, reservation_policy=policy(),
        registration_sha256="b" * 64, authority_check=authority)
    clock = [0.0]
    g._pacing_clock = lambda: clock[0]
    g._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    return g


def stubs(monkeypatch, sequence):
    values, calls = iter(sequence), []
    class Response:
        status, headers = 200, {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return json.dumps(self.value).encode()[:size]
    def opened(request, **kwargs):
        calls.append(deepcopy(json.loads(request.data)))
        value = next(values)
        if isinstance(value, Exception): raise value
        return Response(value)
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    return calls


def call(g, c, operation="review1"):
    with g.scope("review", operation, "session1"):
        return c.complete([{"role": "user", "content": "synthetic memory input"}])


def http_error(code, body=None):
    raw = json.dumps({"error": {"code": code}} if body is None else body).encode()
    return urllib.error.HTTPError("https://fixture.invalid", code, "fixture", {}, io.BytesIO(raw))


@pytest.mark.parametrize("failure", [lambda: http_error(429), lambda: http_error(503),
    lambda: TimeoutError(), lambda: {"error": {"code": 400}}])
def test_no_body_failure_retries_once_and_keeps_unknown_hold(tmp_path, monkeypatch, failure):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [failure(), receipt(c)])
        assert call(g, c) == "local fixture"
        state = g.snapshot()
        assert not state["stopped"] and len(calls) == 2 and calls[0] == calls[1]
        assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        assert len(logical_attempts(state)) == 1
        assert state["attempts"][1]["infrastructure_retry_of_request_id"] == state["attempts"][0]["request_id"]
        assert state["attempts"][1]["infrastructure_retry_index"] == 1
        with pytest.raises(MinimalCallStopped, match="duplicate_model_operation"):
            call(g, c)
        assert len(calls) == 2


def test_exhaustion_primary_stops_at_three_dispatches(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [http_error(503) for _ in range(4)])
        with pytest.raises(MinimalCallStopped): call(g, c)
        assert len(calls) == 3
        state = g.snapshot()
        assert state["stopped"] and state["stop_reason"] == "infrastructure_retries_exhausted"
        assert state["unknown_cost_requests"] == 3
        with pytest.raises(MinimalCallStopped): call(g, c, "another")
        assert len(calls) == 3


@pytest.mark.parametrize("failure", [lambda: http_error(401), lambda: http_error(400),
    lambda: http_error(500, {"choices": [{"message": {"content": "FIRST ANSWER"}}], "error": {"code": 500}}),
    lambda: {"choices": [{"message": {"content": "FIRST ANSWER"}}], "error": {"code": 500}},
    lambda: ConnectionResetError(), lambda: http_error(503, ["not an error envelope"])])
def test_unsupported_or_content_bearing_failure_never_retries(tmp_path, monkeypatch, failure):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [failure(), receipt(c)])
        with pytest.raises(MinimalCallStopped): call(g, c)
        assert len(calls) == 1 and g.snapshot()["stopped"]


def test_budget_reserves_before_retry_and_keeps_original_unknown(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite", budget="0.02")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [http_error(503), receipt(c)])
        with pytest.raises(MinimalCallStopped, match="pre_send_budget_exceeded"): call(g, c)
        assert len(calls) == 1 and g.snapshot()["unknown_cost_requests"] == 1
        assert Decimal(g.snapshot()["reserved_cost_usd"]) > 0


def test_reopened_contract_retains_hold_and_tamper_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite"
    with closing(guard(path)) as g:
        c = client("memory"); g.attach(c)
        stubs(monkeypatch, [http_error(503), receipt(c)])
        call(g, c)
        saved = g.snapshot()
    with closing(guard(path)) as g:
        assert g.snapshot()["reserved_cost_usd"] == saved["reserved_cost_usd"]
        assert g.snapshot()["attempts"] == saved["attempts"]
    with sqlite3.connect(path) as db:
        row = saved["attempts"][1]
        row["request_sha256"] = "c" * 64
        db.execute("UPDATE attempts SET record=? WHERE request_id=?", (json.dumps(row), row["request_id"]))
    with pytest.raises(ValueError, match="invalid_e1br_infrastructure_retry"): guard(path)


def test_authority_rechecked_after_retry_wait(tmp_path, monkeypatch):
    allowed = [True]
    def check():
        if not allowed[0]: raise MinimalCallStopped("authorization_changed")
    with closing(guard(tmp_path / "calls.sqlite", authority=check)) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [http_error(429), receipt(c)])
        original_sleep = g._pacing_sleep
        def sleep(seconds):
            original_sleep(seconds); allowed[0] = False
        g._pacing_sleep = sleep
        with pytest.raises(MinimalCallStopped, match="authorization_changed"): call(g, c)
        assert len(calls) == 1 and g.snapshot()["unknown_cost_requests"] == 1


def reflection(g):
    c = client("reflection"); c.reflection_return_format = "body_json"
    g.attach(c)
    return c, PersonaHypothesisUpdater(c, evidence_review_protocol=True, body_json_protocol=True)


def body(c, value):
    data = receipt(c)
    data["choices"][0]["message"]["content"] = json.dumps(value)
    return data


def test_reflection_correction_and_infrastructure_share_two_retries(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c, updater = reflection(g)
        empty = {"protocol_version": "2.0", "reviews": [], "changes": []}
        calls = stubs(monkeypatch, [http_error(503), body(c, {**empty, "reviews": "bad"}),
            http_error(503), http_error(503), body(c, empty)])
        with g.scope("review", "review1", "session1"):
            updater.update(PersonaProfile(user_id="person"), context())
        state = g.snapshot()
        assert len(calls) == 4 and not state["stopped"]
        assert updater.last_status == "failed"
        assert [row["product_step"] for row in state["attempts"]] == ["initial", "initial", "format_correction", "format_correction"]
        assert state["attempts"][-1]["infrastructure_retry_index"] == 2
        assert g.reflection_failure_is_terminal(c)
        assert len(logical_attempts(state)) == 2
        m = client("memory"); g.attach(m)
        stubs(monkeypatch, [receipt(m)])
        assert call(g, m, "review2") == "local fixture"
        assert not g.reflection_failure_is_terminal(c)


def test_reflection_once_corrected_content_cannot_be_replaced(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c, updater = reflection(g)
        empty = {"protocol_version": "2.0", "reviews": [], "changes": []}
        calls = stubs(monkeypatch, [body(c, {**empty, "reviews": "bad"}), body(c, empty)])
        with g.scope("review", "review1", "session1"):
            updater.update(PersonaProfile(user_id="person"), context())
        assert len(calls) == 2 and updater.last_status == "ok"
        assert g.snapshot()["unknown_cost_requests"] == 0
        assert g.snapshot()["attempts"][1]["format_parent_request_id"] == g.snapshot()["attempts"][0]["request_id"]


def test_real_wire_zero_reasoning_usage_roundtrip_stays_readable(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite"
    with closing(guard(path)) as g:
        c = client("memory"); g.attach(c)
        data = receipt(c, usage={"prompt_tokens": 9440, "completion_tokens": 78,
            "total_tokens": 9518, "cost": .00001,
            "completion_tokens_details": {"reasoning_tokens": 0}})
        # The production response's prompt usage must fit this smaller fixture.
        data["usage"]["prompt_tokens"] = 100
        data["usage"]["total_tokens"] = 178
        stubs(monkeypatch, [data])
        assert call(g, c) == "local fixture"
        state = g.snapshot()
        assert not state["stopped"] and state["calls_used"] == 1
        assert state["attempts"][0]["usage"]["reasoning_tokens"] == 0
        assert set(state["attempts"][0]["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens", "cost", "reasoning_tokens"}
    with closing(guard(path)) as g:
        assert g.snapshot()["attempts"] == state["attempts"]
        assert not g.snapshot()["stopped"]


@pytest.mark.parametrize("usage", [{"reasoning_tokens": True}, {"reasoning_tokens": -1},
    {"reasoning_tokens": "0"}, {"reasoning_tokens": float('nan')},
    {"cost": float('inf')}, {"completion_tokens_details": {"reasoning_tokens": 0}}])
def test_stored_usage_rejects_corrupt_or_unsanitized_fields(tmp_path, monkeypatch, usage):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        stubs(monkeypatch, [receipt(c)])
        call(g, c)
        row = g.snapshot()["attempts"][0]
        row["usage"].update(usage)
        g._db.execute("UPDATE attempts SET record=? WHERE request_id=?", (json.dumps(row), row["request_id"]))
        assert g.snapshot()["stop_reason"] == "ledger_unavailable"
