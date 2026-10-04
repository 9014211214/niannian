"""Separate completed HTTP bodies from post-receipt content failures."""
from decimal import Decimal
import http.client
import json
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.response_observability import validate_observation
from tests.test_reserved_calls import client, policy, receipt, call, stub


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unstubbed network"))


class _InnerJSONGuard(ReservedCallGuard):
    def _record_receipt(self, c, row, data):
        super()._record_receipt(c, row, data)
        parsed = json.loads(data["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict) or "response" not in parsed:
            row["validation_error"] = "online_response_missing"
            raise ValueError("private exception text is not a diagnostic code")


def make_guard(tmp_path, cls=ReservedCallGuard):
    return cls(tmp_path / "calls.sqlite3", max_requests=30, max_observed_cost_usd="1",
               reservation_policy=policy())


@pytest.mark.parametrize("content,code", [('{"private inner":', "content_json_invalid"),
                                         ('{"private inner": true}', "online_response_missing")])
def test_inner_content_failure_keeps_completed_outer_evidence_and_known_fee(tmp_path, monkeypatch, content, code):
    g = make_guard(tmp_path, _InnerJSONGuard)
    c = client(); g.attach(c)
    data = receipt(c)
    data["choices"][0]["message"]["content"] = content
    calls = stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped, match="request_failed"):
        call(g, c)
    state = g.snapshot()
    row = state["attempts"][0]
    assert row["failure_phase"] == "receipt_validation" and row["failure_code"] == code
    assert row["observation"]["envelope_kind"] == "success"
    assert row["observation"]["body_status"] == "parsed"
    assert row["observation"]["generation_id"] == "gen-reservationfixture1234"
    assert row["observation"]["usage_source"] == "top_level_usage"
    validate_observation(row["observation"])
    assert row["status"] == "failed" and row["http_status"] == 200
    assert state["unknown_cost_requests"] == 0 and Decimal(state["known_cost_usd"]) == Decimal("0.00001")
    assert state["reserved_cost_usd"] == "0"
    assert "private" not in json.dumps(state) and "synthetic-placeholder" not in json.dumps(state)
    with pytest.raises(MinimalCallStopped):
        call(g, c, "second")
    assert len(calls) == c.network_request_count == 1
    g.close()
    retained = make_guard(tmp_path, _InnerJSONGuard)
    assert retained.snapshot()["attempts"][0] == row
    assert retained.snapshot()["stopped"]
    retained.close()


@pytest.mark.parametrize("kind,code", [("read", "response_read_failed"),
                                       ("decode", "response_decode_failed"),
                                       ("parse", "response_json_invalid")])
def test_real_outer_transport_failures_remain_interrupted_unknown_and_no_retry(tmp_path, monkeypatch, kind, code):
    calls = []
    class Response:
        status = 200
        headers = {"x-request-id": "req-localfixture1234"}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, size):
            if kind == "read":
                raise http.client.IncompleteRead(b"private partial body", 20)
            return b"\xffprivate" if kind == "decode" else b'{"private partial":'
    def receive(*args, **kwargs):
        calls.append(1)
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", receive)
    g = make_guard(tmp_path, _InnerJSONGuard)
    c = client(); g.attach(c)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot()
    row = state["attempts"][0]
    assert row["failure_phase"] == kind and row["failure_code"] == code
    assert row["observation"]["envelope_kind"] == "interrupted"
    assert row["observation"]["body_status"] == kind + "_failed"
    assert state["unknown_cost_requests"] == 1 and state["known_cost_usd"] is None
    assert row["usage"] == {} and "private" not in json.dumps(state)
    validate_observation(row["observation"])
    with pytest.raises(MinimalCallStopped):
        call(g, c, "second")
    assert calls == [1] and c.network_request_count == 1
    g.close()


@pytest.mark.parametrize("change,code", [({"provider": "wrong"}, "response_provider_mismatch"),
    ({"usage": {"prompt_tokens": 5, "completion_tokens": 3}}, "provider_cost_unknown"),
    ({"choices": [{"finish_reason": "length", "message": {"content": "private truncated text"}}]}, "invalid_or_truncated_response")])
def test_route_billing_and_truncation_rejections_are_receipt_validation(tmp_path, monkeypatch, change, code):
    g = make_guard(tmp_path)
    c = client(); g.attach(c)
    calls = stub(monkeypatch, receipt(c, **change))
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert row["failure_phase"] == "receipt_validation" and row["failure_code"] == code
    assert row["validation_error"] == code and row["observation"]["body_status"] == "parsed"
    assert row["observation"]["envelope_kind"] == "success"
    assert state["unknown_cost_requests"] == int(code == "provider_cost_unknown")
    assert "private" not in json.dumps(state) and len(calls) == 1
    g.close()


def test_unknown_receipt_exception_and_unregistered_code_cannot_leak_text(tmp_path, monkeypatch):
    class UnsafeDiagnosticGuard(ReservedCallGuard):
        def _record_receipt(self, c, row, data):
            super()._record_receipt(c, row, data)
            row["validation_error"] = "private diagnostic accidentally copied by a future hook"
            raise RuntimeError("private exception and hidden reasoning must never persist")
    g = make_guard(tmp_path, UnsafeDiagnosticGuard)
    c = client(); g.attach(c)
    stub(monkeypatch, receipt(c))
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert row["failure_phase"] == "receipt_validation"
    assert row["failure_code"] == "receipt_validation_failed"
    assert "validation_error" not in row and "private" not in json.dumps(state)
    assert state["unknown_cost_requests"] == 0
    g.close()


@pytest.mark.parametrize("billing_duplicate", [False, True])
def test_duplicate_outer_fields_are_envelope_validation_without_ambiguous_identity(tmp_path, monkeypatch, billing_duplicate):
    from tests.test_reserved_call_review import raw_response
    g = make_guard(tmp_path)
    c = client(); g.attach(c)
    raw = json.dumps(receipt(c))
    if billing_duplicate:
        raw = raw.replace('"cost": 1e-05', '"cost": 0.1, "cost": 0')
    else:
        raw = raw.replace('"id": "gen-reservationfixture1234"',
                          '"id": "gen-firstfixture1234", "id": "gen-reservationfixture1234"')
    raw_response(monkeypatch, raw.encode())
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert row["failure_phase"] == "envelope_validation"
    assert row["failure_code"] == "duplicate_response_field"
    assert row["observation"]["body_status"] == "parsed"
    assert row["observation"]["envelope_kind"] == "success"
    assert "generation_id" not in row["observation"]
    assert state["unknown_cost_requests"] == int(billing_duplicate)
    validate_observation(row["observation"])
    g.close()
