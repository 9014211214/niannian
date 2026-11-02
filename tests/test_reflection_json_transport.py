"""Real payload/guard code with synthetic HTTP; never calls an external model."""
from tests._model_http_stub import stub_model_http
from contextlib import closing
from copy import deepcopy
import json

import pytest

from aria.engine.memory.diagnosis import reflection_system_prompt
from aria.engine.memory.reflection_contract import reflection_response_schema
from aria.engine.policy import reflection_json
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.runtime import create_minimal_clients
from tests.test_reserved_calls import guard, receipt, stub

MESSAGES = [{"role": "system", "content": reflection_system_prompt(True)},
            {"role": "user", "content": "本次会话：[turn 0] 用户:今天想安静一会儿。"}]
EMPTY = {"protocol_version": "2.0", "reviews": [], "changes": []}


def client():
    return create_minimal_clients(model_calls_enabled=True, api_key="local-fixture",
                                  reflection_return_format="json_schema").reflection


def response(c, content=None):
    data = receipt(c)
    data["choices"][0]["message"]["content"] = json.dumps(EMPTY) if content is None else content
    return data


def test_explicit_structured_reflection_actual_wire_and_reservation(tmp_path, monkeypatch):
    c = client()
    calls = stub(monkeypatch, response(c))
    with closing(guard(tmp_path / "ledger.sqlite")) as g:
        g.attach(c)
        with g.scope("review", "review1", "session1"):
            assert json.loads(c.complete(MESSAGES)) == EMPTY
        assert g.snapshot()["calls_used"] == 1
    wire, = calls
    assert wire["response_format"] == reflection_json.reflection_request_format()
    assert wire["messages"] == MESSAGES
    assert wire["provider"]["only"] == ["anthropic"]
    assert wire["provider"]["require_parameters"] is True
    assert wire["provider"]["allow_fallbacks"] is False
    assert wire["reasoning"] == {"enabled": False, "exclude": True}
    assert not {"tools", "tool_choice", "temperature"} & wire.keys()
    assert c.timeout == 240 and c.max_retries == 0 and c.fallback_models == []
    assert c.last_diagnostic["output_kind"] == "reflection_json_schema"


@pytest.mark.parametrize("content", [
    'not JSON', '{"protocol_version":2.0,"reviews":[],"changes":[]}',
    '{"protocol_version":"2.0","reviews":[],"changes":[],"reasoning":"private"}',
    '{"protocol_version":"2.0","protocol_version":"2.0","reviews":[],"changes":[]}',
    '{"protocol_version":"2.0","reviews":[],"changes":[],"x":NaN}',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```',
    '{"protocol_version":"2.0","changes":[]}',
])
def test_invalid_first_reply_is_not_repaired_or_retried(tmp_path, monkeypatch, content):
    c = client()
    calls = stub(monkeypatch, response(c, content))
    with closing(guard(tmp_path / "ledger.sqlite")) as g:
        g.attach(c)
        with g.scope("review", "review1", "session1"), pytest.raises(ValueError, match="invalid_reflection_json_schema"):
            c.complete(MESSAGES)
        assert g.snapshot()["stopped"] is True
        assert g.snapshot()["known_cost_usd"] == "0.00001"
        assert g.snapshot()["unknown_cost_requests"] == 0
        assert c.last_diagnostic["error_type"] == "invalid_reflection_json_schema"
        assert "private" not in json.dumps(c.last_diagnostic)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["drop_schema", "weaken_schema", "temperature", "fallback"])
def test_actual_wire_tamper_rejected_before_send(tmp_path, monkeypatch, change):
    c = client()
    calls = stub(monkeypatch, response(c))
    original = c._request_for_model
    def altered(payload, *, model):
        payload = deepcopy(payload)
        if change == "drop_schema": payload.pop("response_format")
        if change == "weaken_schema": payload["response_format"]["json_schema"]["strict"] = False
        if change == "temperature": payload["temperature"] = 0.2
        if change == "fallback": payload["provider"]["allow_fallbacks"] = True
        return original(payload, model=model)
    monkeypatch.setattr(c, "_request_for_model", altered)
    with closing(guard(tmp_path / "ledger.sqlite")) as g:
        g.attach(c)
        with g.scope("review", "review1", "session1"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        assert g.snapshot()["calls_used"] == 0
    assert calls == []


def test_changed_format_cannot_rebind_existing_allowance(tmp_path):
    c = client()
    with closing(guard(tmp_path / "ledger.sqlite")) as g:
        g.attach(c)
        c.reflection_return_format = "native"
        with pytest.raises(ValueError, match="differs from retained"):
            g.attach(c)


def test_custom_reflection_model_requires_explicit_native_mode(monkeypatch):
    import urllib.request
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="verified default model"):
        create_minimal_clients(model_calls_enabled=True, api_key="fixture", reflection_model="custom/model")
    c = create_minimal_clients(model_calls_enabled=True, api_key="fixture",
                               reflection_model="custom/model", reflection_return_format="native").reflection
    assert c.model == "custom/model" and c.reflection_return_format == "native"


@pytest.mark.parametrize("change", ["model", "fallback", "retry", "role", "unbounded", "wrong_prompt"])
def test_unsupported_contract_fails_without_network(monkeypatch, change):
    c = client()
    calls = stub(monkeypatch, response(c))
    messages = deepcopy(MESSAGES)
    if change == "model": c.model = "unverified/route"
    if change == "fallback": c.fallback_models = ["unverified/route"]
    if change == "retry": c.max_retries = 1
    if change == "role": c.role = "memory"
    if change == "unbounded": c.bounded_transport = False
    if change == "wrong_prompt": messages[0]["content"] = "unregistered instructions"
    with pytest.raises(ValueError): c.complete(messages)
    assert calls == []


def test_projection_keeps_shape_and_preserves_original_full_schema():
    full = reflection_response_schema()
    original = deepcopy(full)
    wire = reflection_json._project_wire_keywords(full)
    removed = []
    def compare(a, b, path=""):
        for key, value in a.items():
            if key in reflection_json._LOCAL_ONLY:
                assert key not in b and key in b["description"]
                removed.append(path + key)
            elif key in {"properties", "$defs"}:
                assert set(value) == set(b[key])
                for name, node in value.items(): compare(node, b[key][name], path + key + "/" + name + "/")
            elif key == "items": compare(value, b[key], path + "items/")
            elif key == "anyOf":
                assert len(value) == len(b[key])
                for left, right in zip(value, b[key]): compare(left, right, path + "anyOf/")
            elif key != "description": assert b[key] == value
    compare(full, wire)
    assert len(removed) == 40
    assert full == original == reflection_response_schema()
    altered = deepcopy(full)
    altered["unsupportedKeyword"] = True
    from unittest.mock import patch
    with patch.object(reflection_json, "reflection_response_schema", return_value=altered):
        with pytest.raises(ValueError, match="unsupported_reflection_schema_keyword"):
            reflection_json.reflection_wire_schema()


def test_projection_does_not_weaken_local_full_validation():
    review = {"id": "r1", "hypothesis_id": None, "claim_quote": None, "scope_quote": None,
              "applicability": "unknown", "relation": "neutral", "evidence_basis": "context_only",
              "evidence": [{"turn_index": 0, "evidence_quote": "今天"}],
              "event_evidence": [{"turn_index": 0, "evidence_quote": "今天"}],
              "reason": "仅背景", "event_summary": "本次话题"}
    value = {**EMPTY, "reviews": [review]}
    reflection_json.parse_reflection_json(json.dumps(value))
    value["reviews"][0]["reason"] = "长" * 241
    with pytest.raises(ValueError): reflection_json.parse_reflection_json(json.dumps(value))
    value["reviews"][0]["reason"] = "仅背景"
    value["reviews"] = [review] * 7
    with pytest.raises(ValueError): reflection_json.parse_reflection_json(json.dumps(value))


def test_shape_diagnostics_preserve_bad_field_not_generated_values(monkeypatch):
    c = client()
    content = json.dumps({"protocol_version": "PRIVATE_GENERATED_VALUE", "reviews": [], "changes": [],
                          "PRIVATE_UNKNOWN_KEY": "PRIVATE_GENERATED_VALUE"})
    stub(monkeypatch, response(c, content))
    with pytest.raises(ValueError): c.complete(MESSAGES)
    diagnostic = c.last_diagnostic["reflection_shape"]
    assert diagnostic["kind"] == "schema"
    assert {tuple(row["path"]) for row in diagnostic["issues"]} == {("protocol_version",), ("unknown_field",)}
    assert "PRIVATE" not in json.dumps(c.last_diagnostic)
    from aria.bench.e1b_runner import _reflection_client_diagnostic
    assert _reflection_client_diagnostic(c)["reflection_shape"] == diagnostic
    c.last_diagnostic["reflection_shape"]["issues"].append({"type": "PRIVATE_CODE", "path": ["PRIVATE_FIELD"],
                                                           "input": "PRIVATE_INPUT"})
    assert "PRIVATE" not in json.dumps(_reflection_client_diagnostic(c))


@pytest.mark.parametrize("supported", [[], ["response_format"], ["structured_outputs"],
                                      ["response_format", "structured_outputs"]])
def test_e1b_requires_exact_route_capability_evidence(tmp_path, monkeypatch, supported):
    from scripts import run_e1b as run
    from aria._openrouter import DEFAULT_REFLECTION_MODEL
    name = "sonnet-endpoints.public-v2.json"
    (tmp_path / name).write_text(json.dumps({"data": {"id": DEFAULT_REFLECTION_MODEL,
        "endpoints": [{"tag": "anthropic", "status": 0, "supported_parameters": supported}]}}), encoding="utf-8")
    monkeypatch.setattr(run, "base_metadata_evidence", lambda _: {"files": [{"path": name}]})
    if len(supported) == 2:
        assert run.metadata_evidence(tmp_path)["reflection_structured_output"]["provider"] == "anthropic"
    else:
        with pytest.raises(run.RunError, match="reflection_structured_output_support_unverified"):
            run.metadata_evidence(tmp_path)
