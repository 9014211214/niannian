"""HTTP 200 error bodies and rejected identity evidence use local fixtures only."""
from copy import deepcopy
from decimal import Decimal
from email.message import Message
import json
import io
import sqlite3
import urllib.request

import pytest

from aria.engine.policy.bounded_recovery import recovery_eligible
from aria.engine.policy import minimal_calls
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.response_observability import (
    observe_success, observe_response_error, receipt_identity, validate_attempt_observation,
    validate_observation, validate_receipt_identity,
)
from tests.test_reserved_calls import call, client, guard, receipt, stub
from tests.test_reserved_call_review import raw_response
from tests.test_bounded_http_recovery import guarded


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unstubbed network"))


@pytest.mark.parametrize("error", [{}, {"code": 502, "type": "api_error"},
    {"code": 429, "type": "rate_limit_error", "metadata": {"provider_name": "Anthropic"}},
    {"code": True, "type": "private unknown error", "message": "private message",
     "metadata": {"provider_name": "private unknown provider", "raw": "private raw",
                  "usage": {"cost": 0}, "reasoning": "private hidden reasoning"}}])
@pytest.mark.parametrize("valid_identity_and_choices", [False, True])
def test_error_object_precedes_model_validation_and_never_authorizes_retry(
        tmp_path, monkeypatch, error, valid_identity_and_choices):
    path = tmp_path / "calls.sqlite3"
    g = guard(path); c = client("reflection"); g.attach(c)
    data = receipt(c, error=error) if valid_identity_and_choices else {
        "id": "gen-localfixture1234", "error": error}
    sent = stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert row["failure_code"] == row["validation_error"] == "provider_error_envelope"
    assert row["failure_phase"] == "receipt_validation" and row["http_status"] == 200
    assert row["receipt_identity"] == {
        "version": "receipt_identity_v1", "model_status": "recognized" if valid_identity_and_choices else "missing",
        "provider_status": "recognized" if valid_identity_and_choices else "missing", "error_field_status": "object"}
    assert row["observation"]["version"] == "1.4"
    assert row["observation"]["envelope_kind"] == "response_error"
    assert "private" not in json.dumps(state)
    assert not recovery_eligible(row)
    assert "http_error_recovery_evidence" not in row
    assert state["unknown_cost_requests"] == int(not valid_identity_and_choices)
    if valid_identity_and_choices:
        assert Decimal(state["known_cost_usd"]) == Decimal("0.00001")
        assert state["reserved_cost_usd"] == "0"
    else:
        assert state["known_cost_usd"] is None and Decimal(state["reserved_cost_usd"]) > 0
    validate_observation(row["observation"])
    validate_receipt_identity(row["receipt_identity"])
    with pytest.raises(MinimalCallStopped):
        call(g, c, "second")
    assert len(sent) == 1
    g.close()
    reopened = guard(path)
    assert reopened.snapshot()["attempts"][0] == row and reopened.snapshot()["stopped"]
    reopened.close()


@pytest.mark.parametrize("field", ["model", "provider"])
@pytest.mark.parametrize("kind,value", [("missing", None), ("null", None), ("wrong_type", []),
    ("wrong_type", False), ("wrong_type", 0), ("unrecognized", "private unknown identity"),
    ("unrecognized", "")])
def test_rejected_identity_status_is_saved_without_unknown_value(tmp_path, monkeypatch, field, kind, value):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    data = receipt(c)
    if kind == "missing":
        del data[field]
    else:
        data[field] = value
    stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    row = g.snapshot()["attempts"][0]
    assert row["validation_error"] == f"response_{field}_mismatch"
    assert row["receipt_identity"][field + "_status"] == kind
    assert "returned_" + field not in row
    assert "private" not in json.dumps(row)
    assert row["usage"]["cost"] == 0.00001
    assert row["observation"] == observe_success(data)
    g.close()


@pytest.mark.parametrize("error,kind", [(None, "null"), (False, "wrong_type"), (0, "wrong_type"),
    ([], "wrong_type"), ("private malformed error", "wrong_type")])
def test_nonobject_error_only_adds_shape_and_preserves_original_success_gate(tmp_path, monkeypatch, error, kind):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    data = receipt(c, error=error)
    stub(monkeypatch, data)
    assert call(g, c) == "local fixture"
    row = g.snapshot()["attempts"][0]
    assert row["receipt_identity"]["error_field_status"] == kind
    assert row["status"] == "received" and row["observation"] == observe_success(data)
    assert "private" not in json.dumps(row)
    g.close()


def test_known_error_metadata_does_not_overwrite_request_header_or_promote_nested_fee(tmp_path, monkeypatch):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    data = {"id": "gen-localfixture1234", "error": {"code": 502, "type": "api_error",
        "metadata": {"provider_name": "Anthropic", "provider_code": 529, "usage": {"cost": 0}}}}
    raw_response(monkeypatch, json.dumps(data).encode())
    fixture_open = urllib.request.urlopen
    def headers_like_http_message(*args, **kwargs):
        response = fixture_open(*args, **kwargs)
        response.headers = Message()
        response.headers["x-request-id"] = "req-reviewfixture1234"
        return response
    monkeypatch.setattr(urllib.request, "urlopen", headers_like_http_message)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    row = g.snapshot()["attempts"][0]; observed = row["observation"]
    assert observed["request_id"] == "req-reviewfixture1234"
    assert observed["provider_error_code"] == 502 and observed["provider_code"] == 529
    assert observed["provider_error_type"] == "api_error" and observed["provider_reported_name"] == "Anthropic"
    assert row["usage"] == {} and "usage_source" not in observed
    validate_observation(observed)
    attempt = {"response_observability": observed, "provider_usage": {}, "http_status": 200,
               "error_type": "response_error"}
    validate_attempt_observation(attempt)
    with pytest.raises(ValueError, match="response_error_mismatch"):
        validate_attempt_observation({**attempt, "error_type": None})
    with pytest.raises(ValueError, match="response_error_mismatch"):
        validate_attempt_observation({**attempt, "http_status": 429})
    g.close()


def test_duplicate_billing_error_stays_ambiguous_and_does_not_get_identity_evidence(tmp_path, monkeypatch):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    raw_response(monkeypatch, b'{"error":{"code":502},"usage":{"cost":0,"cost":1}}')
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert row["failure_code"] == "duplicate_response_field"
    assert row["usage"] == {} and "receipt_identity" not in row
    assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
    g.close()


def test_http200_error_code429_cannot_use_opted_in_recovery(tmp_path, monkeypatch):
    g, clock = guarded(tmp_path / "calls.sqlite3")
    c = client(); g.attach(c)
    sent = []
    class Response(io.BytesIO):
        status = 200
        headers = {}
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        clock[0] += 0.25
        for event in ("connection_preparation_started", "connection_preparation_completed", "tls_completed",
                      "application_write_attempted", "request_write_completed", "response_headers_returned"):
            trace._mark(event)
        return Response(json.dumps({"error": {"code": 429, "metadata": {"provider_name": "Alibaba"}},
                                    "usage": {"cost": 0}}).encode())
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot(); row = state["attempts"][0]
    assert len(sent) == 1 and state["recovery"]["relations"] == []
    assert state["recovery"]["extra_dispatches"] == 0
    assert row["failure_code"] == "provider_error_envelope"
    assert row["observation"]["provider_error_code"] == 429 and not recovery_eligible(row)
    assert Decimal(state["known_cost_usd"]) == 0 and state["unknown_cost_requests"] == 0
    with pytest.raises(MinimalCallStopped):
        call(g, c, "second")
    assert len(sent) == 1
    g.close()


def test_old_saved_rows_remain_compatible_without_diagnostic_migration(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    g = guard(path); c = client(); g.attach(c)
    stub(monkeypatch, receipt(c))
    call(g, c)
    old = g.snapshot()["attempts"][0]
    del old["receipt_identity"]
    g.close()
    # Only this independent local fixture is made into an old-format row.
    with sqlite3.connect(path) as db:
        db.execute("UPDATE attempts SET record=?", (json.dumps(old),))
    reopened = guard(path)
    assert reopened.snapshot()["attempts"] == [old]
    reopened.close()


@pytest.mark.parametrize("field,value", [("version", "private version"), ("model_status", "private identity"),
    ("provider_status", []), ("error_field_status", None), ("raw", "private raw")])
def test_receipt_identity_validator_rejects_untrusted_states(field, value):
    diagnostic = receipt_identity({}, models=[], providers=[])
    diagnostic[field] = value
    with pytest.raises(ValueError, match="invalid_receipt_identity"):
        validate_receipt_identity(diagnostic)


@pytest.mark.parametrize("field,value", [("version", "1.0"), ("envelope_kind", "success"),
    ("error_field_status", "absent"), ("error_field_status", []), ("raw", "private raw"),
    ("provider_error_code", True), ("provider_error_type", "private type"),
    ("error_category", "rate_limited"), ("body_read_error_kind", "timeout")])
def test_response_error_observation_rejects_invalid_or_private_fields(field, value):
    observed = {"version": "1.4", "envelope_kind": "response_error", "body_status": "parsed",
                "request_id_status": "absent", "error_field_status": "object", "router_error_type_status": "absent"}
    mutated = deepcopy(observed); mutated[field] = value
    with pytest.raises(ValueError, match="invalid_response_observability"):
        validate_observation(mutated)


@pytest.mark.parametrize("native,canonical,status", [
    ("private native type", "invalid_request", "recognized"),
    ("invalid_request_error", "timeout", "recognized"),
    ("api_error", "provider_unavailable", "recognized"),
    ("api_error", "private canonical type", "unrecognized"),
    ("api_error", "invalid_request_error", "unrecognized"),
    ("api_error", ["private injected type"], "wrong_type"),
    ("api_error", None, "null"),
])
def test_router_canonical_type_is_independent_of_native_type(tmp_path, monkeypatch, native, canonical, status):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    data = {"error": {"type": native, "message": "private text", "metadata": {"error_type": canonical}}}
    stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    row = g.snapshot()["attempts"][0]; observation = row["observation"]
    assert observation["router_error_type_status"] == status
    if status == "recognized":
        assert observation["router_error_type"] == canonical
        assert observation["router_error_type_source"] == "error.metadata.error_type"
    else:
        assert "router_error_type" not in observation and "router_error_type_source" not in observation
    if native in {"api_error", "invalid_request_error"}:
        assert observation["provider_error_type"] == native
    else:
        assert "provider_error_type" not in observation
    assert row["failure_code"] == "provider_error_envelope" and "private" not in json.dumps(row)
    validate_observation(observation)
    g.close()


@pytest.mark.parametrize("field,value", [("router_error_type", "private unknown"),
    ("router_error_type_source", "error.type"), ("router_error_type_status", "private status"),
    ("router_error_type_status", []), ("provider_metadata_present", False)])
def test_router_error_observation_rejects_injected_canonical_fields(field, value):
    observed = {"version": "1.4", "envelope_kind": "response_error", "body_status": "parsed",
        "request_id_status": "absent", "error_field_status": "object", "provider_metadata_present": True,
        "provider_name_status": "absent", "router_error_type_status": "recognized",
        "router_error_type": "timeout", "router_error_type_source": "error.metadata.error_type"}
    validate_observation(observed)
    observed[field] = value
    with pytest.raises(ValueError, match="invalid_response_observability"):
        validate_observation(observed)


@pytest.mark.parametrize("error", [{"type": "timeout"}, {"error_type": "invalid_request"},
    {"metadata": {"raw": '{"error_type":"provider_unavailable"}'}}])
def test_router_type_is_not_inferred_from_other_paths(error):
    data = {"error": error}
    observed = observe_response_error(data, observe_success(data), error_field_status="object")
    assert observed["router_error_type_status"] == "absent"
    assert "router_error_type" not in observed and "router_error_type_source" not in observed
    validate_observation(observed)


def test_new_error_validation_does_not_mutate_legacy_field_allowlist():
    data = {"error": {"metadata": {"error_type": "timeout"}}}
    current = observe_response_error(data, observe_success(data), error_field_status="object")
    validate_observation(current)
    old = observe_success({})
    validate_observation(old)
    for field, value in (("router_error_type", "timeout"), ("router_error_type_status", "absent"),
                         ("router_error_type_source", "error.metadata.error_type"), ("error_field_status", "object")):
        with pytest.raises(ValueError, match="invalid_response_observability"):
            validate_observation({**old, field: value})
