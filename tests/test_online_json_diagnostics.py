"""Local malformed envelopes identify fixed field paths without leaking values."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy.online_json import parse_json_decision
from aria.engine.policy.online_json_diagnostics import online_shape_diagnostic, sanitize_online_shape_diagnostic
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.runtime import create_minimal_clients
from scripts import run_e1b as run
from tests.test_online_json import DECISION, MESSAGES, envelope
from tests.test_reserved_calls import policy, receipt, stub


VERSION = "online_turn_decision_v1"


@pytest.mark.parametrize("path,value,code,observed", [
    (["current_affect", "confidence"], "PRIVATE_NUMERIC_STRING", "number_type", "string"),
    (["current_affect", "confidence"], True, "number_type", "boolean"),
    (["current_affect", "confidence"], -1, "minimum", "number"),
    (["current_affect", "divergence", "confidence"], 2, "maximum", "number"),
    (["current_affect", "divergence", "possible"], "PRIVATE_BOOLEAN", "boolean_type", "string"),
    (["current_affect", "divergence"], None, "object_type", "null"),
    (["communication_plan", "pace"], "PRIVATE_ENUM_VALUE", "enum", "string"),
    (["communication_plan", "support_order"], "", "min_length", "string"),
    (["communication_plan", "boundary"], "PRIVATE_" * 20, "max_length", "string"),
    (["uncertainty"], float("nan"), "nonfinite_number", "number"),
    (["uncertainty"], float("inf"), "nonfinite_number", "number"),
    (["state_signals"], "PRIVATE_ARRAY", "array_type", "string"),
    (["response"], [], "string_type", "array"),
    (["response"], "  \n ", "empty_response", "string"),
])
def test_field_types_ranges_enums_and_lengths_are_located_without_values(path, value, code, observed):
    payload = deepcopy(DECISION)
    target = payload
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    original = deepcopy(payload)
    raw = json.dumps(payload)
    with pytest.raises(ValueError):
        parse_json_decision(raw, VERSION)
    diagnostic = online_shape_diagnostic(raw, VERSION)
    assert {"type": code, "path": path, "observed_type": observed} in diagnostic["issues"]
    assert "PRIVATE" not in json.dumps(diagnostic)
    assert json.dumps(payload) == json.dumps(original)


def test_missing_nested_fields_and_unknown_names_never_appear_in_diagnostics():
    payload = deepcopy(DECISION)
    del payload["current_affect"]["divergence"]["pattern"]
    payload["current_affect"]["PRIVATE_KEY"] = {"PRIVATE_NESTED": "PRIVATE_VALUE"}
    payload["PRIVATE_TOP"] = "PRIVATE_BODY"
    result = online_shape_diagnostic(json.dumps(payload), VERSION)
    assert {"type": "missing", "path": ["current_affect", "divergence", "pattern"],
            "observed_type": "missing"} in result["issues"]
    assert {"type": "extra_forbidden", "path": ["current_affect", "unknown_field"],
            "observed_type": "object"} in result["issues"]
    assert "PRIVATE" not in json.dumps(result)


def test_list_item_path_and_bounded_issue_count():
    payload = deepcopy(DECISION)
    payload["state_signals"] = [{"id": "PRIVATE_STATE", "confidence": "PRIVATE_CONFIDENCE"}] * 100
    result = online_shape_diagnostic(json.dumps(payload), VERSION)
    assert result["truncated"] is True and len(result["issues"]) == 20
    assert {"type": "enum", "path": ["state_signals", 0, "id"], "observed_type": "string"} in result["issues"]
    assert any(r["type"] == "max_items" for r in result["issues"])
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("raw,kind", [
    ('{"PRIVATE_INVALID":', "json_syntax"),
    ('{"PRIVATE_DUPLICATE":1,"PRIVATE_DUPLICATE":2}', "duplicate_key"),
    (None, "non_text"),
    pytest.param("PRIVATE_" * 20000, "body_too_large", id="bounded-body-limit"),
    ("[", "json_syntax"),
])
def test_malformed_content_diagnostic_never_echoes_text(raw, kind):
    diagnostic = online_shape_diagnostic(raw, VERSION)
    assert diagnostic["kind"] == kind
    assert "PRIVATE" not in json.dumps(diagnostic)


def test_valid_payload_and_v2_use_their_original_schema_without_modification():
    from tests.test_online_decision_v2 import _payload
    for value, version in ((DECISION, VERSION), (_payload(), "online_turn_decision_v2")):
        raw = json.dumps(value)
        assert parse_json_decision(raw, version) == raw
        assert online_shape_diagnostic(raw, version) == {"kind": "schema", "issues": [], "truncated": False}


def test_sanitizer_drops_unknown_field_names_types_and_generation_values():
    value = {"kind": "schema", "PRIVATE": "PRIVATE", "issues": [
        {"type": "PRIVATE_TYPE", "path": ["current_affect", "PRIVATE_KEY", 9999, True],
         "observed_type": "PRIVATE_VALUE", "input": "PRIVATE_INPUT", "message": "PRIVATE_ERROR"}]}
    assert sanitize_online_shape_diagnostic(value) == {"kind": "schema", "issues": [
        {"type": "schema_constraint", "path": ["current_affect", "unknown_field", "unknown_field", "unknown_field"]}]}


def test_client_failure_records_nested_issue_once_without_storing_reply(monkeypatch):
    client = create_minimal_clients(model_calls_enabled=True, api_key="inert-unit-test-key").online
    value = deepcopy(DECISION)
    value["communication_plan"]["pace"] = "PRIVATE_ENUM_VALUE"
    data = envelope(value)
    calls = []
    def receive(request):
        calls.append(request)
        return data
    monkeypatch.setattr(client, "_request_json", receive)
    with pytest.raises(ValueError):
        client.complete(MESSAGES)
    diagnostic = client.last_diagnostic
    assert diagnostic["error_type"] == "invalid_online_json_schema"
    assert diagnostic["online_shape"]["issues"] == [
        {"type": "enum", "path": ["communication_plan", "pace"], "observed_type": "string"}]
    assert len(calls) == 1 and "PRIVATE" not in json.dumps(diagnostic)


def test_bad_carrier_is_distinct_from_body_shape(monkeypatch):
    client = create_minimal_clients(model_calls_enabled=True, api_key="inert-unit-test-key").online
    data = envelope()
    data["choices"][0]["message"].update(content="PRIVATE_INVALID", refusal="PRIVATE_REFUSAL")
    monkeypatch.setattr(client, "_request_json", lambda _: data)
    with pytest.raises(ValueError):
        client.complete(MESSAGES)
    assert client.last_diagnostic["online_shape"] == {"kind": "return_channel", "issues": []}
    assert "PRIVATE" not in json.dumps(client.last_diagnostic)


def test_guard_persists_safe_nested_failure_with_known_cost_and_no_retry(tmp_path, monkeypatch):
    guard = run.EvidenceGuard(tmp_path / "allowance.sqlite3", max_requests=20,
        max_observed_cost_usd="1", reservation_policy=policy(), stage="prescreen", registration_sha256="a" * 64)
    client = run.make_client("online_fast", "inert-unit-test-key")
    guard.attach(client)
    value = deepcopy(DECISION)
    value["current_affect"]["divergence"]["possible"] = "PRIVATE_BOOLEAN"
    content = json.dumps(value)
    data = receipt(client)
    data["choices"][0]["message"]["content"] = content
    calls = stub(monkeypatch, data)
    with guard.scope("turn", "first", "fictional-session"):
        with pytest.raises(MinimalCallStopped):
            client.complete(MESSAGES)
    state = guard.snapshot()
    row, = state["attempts"]
    assert row["reply_shape"]["diagnostic"]["issues"] == [{"type": "boolean_type",
        "path": ["current_affect", "divergence", "possible"], "observed_type": "string"}]
    assert row["reply_shape"]["required_fields_present"] == dict.fromkeys(DECISION, True)
    assert row["validation_error"] == "online_decision_schema_invalid"
    assert row["visible_content_sha256"] == run.text_sha(content)
    assert state["known_cost_usd"] == "0.00001" and state["unknown_cost_requests"] == 0
    assert "PRIVATE" not in json.dumps(state) and "inert-unit-test-key" not in json.dumps(state)
    with guard.scope("turn", "second", "fictional-session"):
        with pytest.raises(MinimalCallStopped):
            client.complete(MESSAGES)
    assert len(calls) == 1
    guard.close()
    reopened = run.EvidenceGuard(tmp_path / "allowance.sqlite3", max_requests=20,
        max_observed_cost_usd="1", reservation_policy=policy(), stage="prescreen", registration_sha256="a" * 64)
    assert reopened.snapshot()["attempts"] == state["attempts"]
    reopened.close()
