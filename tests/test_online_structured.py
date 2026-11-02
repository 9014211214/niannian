"""Strict candidate transport with synthetic metadata/replies, never real models."""
from copy import deepcopy
import json
import sqlite3
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.online_decision import online_decision_tool_spec
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.policy.online_structured import build_online_structured_route
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.runtime import create_minimal_clients
from tests._model_http_stub import stub_model_http
from tests.test_online_json import DECISION, envelope

MODEL = "qwen/qwen3.8-flash"
VERSION = "online_turn_decision_v3"
MESSAGES = [{"role": "system", "content": "协议版本：" + VERSION},
            {"role": "user", "content": "本地软件夹具。"}]


def metadata():
    models = {"data": [{"id": MODEL, "canonical_slug": "qwen/qwen3.8-flash-20260826",
                         "reasoning": {"mandatory": False}}]}
    endpoints = {"data": {"id": MODEL, "endpoints": [{"model_id": MODEL, "tag": "alibaba",
        "name": "Alibaba | fixture", "provider_name": "Alibaba", "status": 0,
        "supported_parameters": ["structured_outputs", "response_format", "reasoning", "max_tokens", "temperature"]}]}}
    return models, endpoints


def route(models=None, endpoints=None):
    defaults = metadata()
    return build_online_structured_route(json.dumps(models or defaults[0]).encode(),
        json.dumps(endpoints or defaults[1]).encode(), model=MODEL, provider="alibaba")


def client(**kwargs):
    return create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture",
        base_url="https://fixture.invalid/v1", online_model=MODEL,
        online_return_format="json_schema_v1", online_structured_route=kwargs.pop("online_structured_route", route()),
        **kwargs).online


def payload():
    return {**deepcopy(DECISION), "safety_concern": "none"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


@pytest.mark.parametrize("mutation", ["json_only", "inactive", "boolean_status", "mandatory", "missing_reasoning",
    "model_mismatch", "duplicate_endpoint", "missing_provider", "wrong_endpoint_model", "bad_parameters"])
def test_capability_rejection_is_explicit_without_client_or_network(mutation):
    models, endpoints = metadata()
    endpoint = endpoints["data"]["endpoints"][0]
    if mutation == "json_only": endpoint["supported_parameters"].remove("structured_outputs")
    if mutation == "inactive": endpoint["status"] = -1
    if mutation == "boolean_status": endpoint["status"] = False
    if mutation == "mandatory": models["data"][0]["reasoning"]["mandatory"] = True
    if mutation == "missing_reasoning": models["data"][0].pop("reasoning")
    if mutation == "model_mismatch": endpoints["data"]["id"] = "fixture/wrong"
    if mutation == "duplicate_endpoint": endpoints["data"]["endpoints"].append(deepcopy(endpoint))
    if mutation == "missing_provider": endpoint["tag"] = "other"
    if mutation == "wrong_endpoint_model": endpoint["model_id"] = "fixture/wrong"
    if mutation == "bad_parameters": endpoint["supported_parameters"] = {"structured_outputs": True}
    with pytest.raises(ValueError): route(models, endpoints)


@pytest.mark.parametrize("bad", [b'{"data":[],"data":[]}', b'{"data":NaN}', b'[]', b'null', b'not-json'])
def test_ambiguous_capability_metadata_rejected(bad):
    with pytest.raises(ValueError):
        build_online_structured_route(bad, json.dumps(metadata()[1]).encode(), model=MODEL, provider="alibaba")


@pytest.mark.parametrize("version", [VERSION, "online_turn_decision_v4_evidence"])
def test_explicit_candidate_keeps_complete_schema_optional_fields_and_first_bytes(monkeypatch, version):
    c = client(online_decision_version=version)
    requests = []
    value = payload()
    if version == "online_turn_decision_v4_evidence":
        value["safety_evidence"] = {"quote": "", "subject": "not_applicable",
                                    "time": "not_applicable", "urgency": "not_applicable"}
    first = " \n" + json.dumps(value, ensure_ascii=False) + "\n "
    data = envelope(value); data["choices"][0]["message"]["content"] = first
    messages = [{"role": "system", "content": "协议版本：" + version}, MESSAGES[-1]]
    def receive(request):
        requests.append(json.loads(request.data)); return data
    monkeypatch.setattr(c, "_request_json", receive)
    assert c.complete(messages) == first
    wire, = requests
    fmt = wire["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = online_decision_tool_spec(version)["function"]["parameters"]
    assert fmt["json_schema"]["schema"] == schema
    assert not {"preference_control", "cognitive_control"} & set(schema["required"])
    assert schema["properties"]["preference_control"]["type"] == "object"
    assert wire["provider"] == {"allow_fallbacks": False, "require_parameters": True,
                                "data_collection": "deny", "only": ["alibaba"]}
    assert wire["model"] == MODEL and wire["reasoning"] == {"enabled": False, "exclude": True}
    assert wire["stream"] is False and not {"tools", "tool_choice", "plugins", "transforms", "models"} & wire.keys()
    assert json.loads(wire["messages"][-1]["content"]) == {"schema": "online_json_input_v2",
        "conversation_history": [], "current_user_message": MESSAGES[-1]["content"]}
    assert json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")) in wire["messages"][0]["content"]
    assert c.max_retries == 0 and c.fallback_models == []


@pytest.mark.parametrize("mutation", ["state_enum", "optional_null", "extra_field", "boolean_number", "missing", "truncation", "refusal"])
def test_strict_wire_never_repairs_or_retries_invalid_first_content(monkeypatch, mutation):
    c = client(); value = payload(); data = envelope(value)
    if mutation == "state_enum": value["state_signals"] = [{"id": "not-a-state", "relation": "candidate", "confidence": .4,
        "evidence_quote": "本地软件夹具", "rationale": "虚构机械标签"}]
    if mutation == "optional_null": value["preference_control"] = None
    if mutation == "extra_field": value["extra"] = "not permitted"
    if mutation == "boolean_number": value["uncertainty"] = True
    if mutation == "missing": value.pop("safety_concern")
    data["choices"][0]["message"]["content"] = json.dumps(value)
    if mutation == "truncation": data["choices"][0]["finish_reason"] = "length"
    if mutation == "refusal": data["choices"][0]["message"]["refusal"] = "fixture"
    captured = []
    def receive(request): captured.append(request); return data
    monkeypatch.setattr(c, "_request_json", receive)
    with pytest.raises(ValueError): c.complete(MESSAGES)
    assert len(captured) == 1 and c.last_diagnostic["fallback_used"] is False
    if mutation not in {"truncation", "refusal"}:
        with pytest.raises(ValueError): parse_json_decision(json.dumps(value), VERSION)


def test_strict_mode_without_protocol_or_with_changed_route_does_not_dispatch(monkeypatch):
    c = client()
    monkeypatch.setattr(c, "_request_json", lambda request: pytest.fail("must reject before request"))
    with pytest.raises(ValueError, match="integrated_protocol"): c.complete([{"role": "user", "content": "hello"}])
    c.model = "qwen/qwen3.7-flash"
    with pytest.raises(ValueError, match="structured_client"): c.complete(MESSAGES)


def capture_wire(monkeypatch, c):
    requests = []
    def receive(request): requests.append(request); return envelope(payload())
    with monkeypatch.context() as patch:
        patch.setattr(c, "_request_json", receive); c.complete(MESSAGES)
    return requests[0]


@pytest.mark.parametrize("mutation", ["enum", "strict", "optional_required", "provider", "format", "new_option"])
def test_changed_actual_wire_is_rejected_before_claim(tmp_path, monkeypatch, mutation):
    c = client(); request = capture_wire(monkeypatch, c); wire = json.loads(request.data)
    fmt = wire["response_format"]
    if mutation == "enum": fmt["json_schema"]["schema"]["properties"]["state_signals"]["items"]["properties"]["id"]["enum"].append("alias")
    if mutation == "strict": fmt["json_schema"]["strict"] = False
    if mutation == "optional_required": fmt["json_schema"]["schema"]["required"].append("preference_control")
    if mutation == "provider": wire["provider"]["only"] = ["other"]
    if mutation == "format": wire["response_format"] = {"type": "json_object"}
    if mutation == "new_option": wire["plugins"] = [{"id": "response-healing"}]
    g = MinimalCallGuard(tmp_path / "guard.sqlite3", max_requests=2, max_observed_cost_usd="1")
    try:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            with pytest.raises(MinimalCallStopped):
                g.request_json(c, urllib.request.Request(request.full_url, data=json.dumps(wire).encode()))
        assert g.snapshot()["calls_used"] == 0
    finally: g.close()


def policy(c):
    binding = c.online_structured_route.binding()
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {MODEL: {
        "input_usd_per_million": "0.20", "output_usd_per_million": "0.47", "provider": "alibaba",
        "response_models": [MODEL, binding["canonical_slug"]], "response_providers": ["Alibaba"],
        "max_input_tokens": 100000, "max_output_tokens": 4096,
        "metadata_sha256": binding["endpoint_metadata_sha256"]}}}


@pytest.mark.parametrize("mutation", ["metadata", "provider", "receipt_provider", "receipt_model", "extra_provider", "extra_model"])
def test_reservation_must_use_same_route_metadata(tmp_path, mutation):
    c = client(); p = policy(c); spec = p["models"][MODEL]
    if mutation == "metadata": spec["metadata_sha256"] = "a" * 64
    if mutation == "provider": spec["provider"] = "other"
    if mutation == "receipt_provider": spec["response_providers"] = ["Other"]
    if mutation == "receipt_model": spec["response_models"] = ["other/model"]
    if mutation == "extra_provider": spec["response_providers"].append("Other")
    if mutation == "extra_model": spec["response_models"].append("other/model")
    g = ReservedCallGuard(tmp_path / "guard.sqlite3", max_requests=2, max_observed_cost_usd="1", reservation_policy=p)
    try:
        with pytest.raises(ValueError, match="metadata_mismatch"): g.attach(c)
        assert g.snapshot()["calls_used"] == 0
    finally: g.close()


def test_guard_binding_persists_route_metadata_and_complete_schema(tmp_path):
    c = client(); path = tmp_path / "guard.sqlite3"
    g = ReservedCallGuard(path, max_requests=2, max_observed_cost_usd="1", reservation_policy=policy(c))
    try:
        g.attach(c)
        with sqlite3.connect(path) as db:
            binding = json.loads(db.execute("SELECT binding FROM roles WHERE role='online_fast'").fetchone()[0])
        assert binding["online_structured_route"] == c.online_structured_route.binding()
        assert len(binding["online_structured_schema_sha256"]) == 64
        assert binding["online_return_format"] == "json_schema_v1"
    finally: g.close()


def test_metadata_rebinding_after_attach_cannot_dispatch(tmp_path, monkeypatch):
    c = client(); request = capture_wire(monkeypatch, c)
    g = MinimalCallGuard(tmp_path / "guard.sqlite3", max_requests=2, max_observed_cost_usd="1")
    try:
        g.attach(c)
        models, endpoints = metadata()
        endpoints["data"]["endpoints"][0]["name"] = "Alibaba | changed fixture"
        c.online_structured_route = route(models, endpoints)
        with g.scope("turn", "one", "session"):
            with pytest.raises(MinimalCallStopped, match="configuration_changed"):
                g.request_json(c, request)
        assert g.snapshot()["calls_used"] == 0
    finally: g.close()


def test_strict_guard_reserves_before_single_transport_and_retains_fee(tmp_path, monkeypatch):
    from tests.test_reserved_calls import stub
    c = client(); path = tmp_path / "guard.sqlite3"
    g = ReservedCallGuard(path, max_requests=2, max_observed_cost_usd="1", reservation_policy=policy(c))
    data = envelope(payload())
    data.update(id="gen-strictfixture123456", model=MODEL, provider="Alibaba",
                usage={"cost": .00001, "prompt_tokens": 5, "completion_tokens": 3})
    def pending(_request):
        with sqlite3.connect(path) as db:
            row = json.loads(db.execute("SELECT record FROM attempts").fetchone()[0])
        assert row["status"] == "pending" and float(row["reservation"]["reserved_cost_usd"]) > 0
    requests = stub(monkeypatch, data, pending)
    try:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            assert json.loads(c.complete(MESSAGES)) == payload()
        assert len(requests) == g.snapshot()["calls_used"] == 1
        assert float(g.snapshot()["known_cost_usd"]) == .00001
        assert g.snapshot()["reserved_cost_usd"] == "0"
        assert requests[0]["provider"]["max_price"] == {"prompt": .2, "completion": .47}
    finally: g.close()


def test_strict_budget_failure_never_claims_or_dispatches(tmp_path):
    c = client()
    g = ReservedCallGuard(tmp_path / "guard.sqlite3", max_requests=2,
        max_observed_cost_usd="0.000001", reservation_policy=policy(c))
    try:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            with pytest.raises(MinimalCallStopped, match="pre_send_budget_exceeded"):
                c.complete(MESSAGES)
        assert g.snapshot()["calls_used"] == 0
    finally: g.close()


@pytest.mark.parametrize("mutation", ["missing", "state_enum", "boolean_number", "duplicate", "null_optional", "multiple", "refusal"])
def test_v4_native_channel_validates_complete_first_arguments_without_retry(monkeypatch, mutation):
    from aria.engine.policy.llm import OpenRouterLLMClient
    value = payload()
    value["safety_evidence"] = {"quote": "", "subject": "not_applicable",
                                "time": "not_applicable", "urgency": "not_applicable"}
    if mutation == "missing": value.pop("current_affect")
    if mutation == "state_enum": value["state_signals"] = [{"id": "not-a-state", "relation": "candidate", "confidence": .4,
        "evidence_quote": "本地软件夹具", "rationale": "虚构机械标签"}]
    if mutation == "boolean_number": value["uncertainty"] = True
    if mutation == "null_optional": value["preference_control"] = None
    raw = json.dumps(value)
    if mutation == "duplicate": raw = '{"response":"first",' + raw[1:]
    data = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
        "tool_calls": [{"function": {"name": "aria_turn_decision", "arguments": raw}}]}}]}
    if mutation == "multiple": data["choices"].append(deepcopy(data["choices"][0]))
    if mutation == "refusal": data["choices"][0]["message"]["refusal"] = "fixture"
    c = OpenRouterLLMClient(api_key="inert", model=MODEL, role="online_fast", max_retries=0,
                           fallback_models=["fixture/other"])
    calls = []
    def receive(request): calls.append(request); return data
    monkeypatch.setattr(c, "_request_json", receive)
    with pytest.raises(ValueError, match="完整JSON"):
        c.complete([{"role": "system", "content": "协议版本：online_turn_decision_v4_evidence"}, MESSAGES[-1]])
    assert len(calls) == 1 and c.last_diagnostic["fallback_used"] is False


@pytest.mark.parametrize("as_object", [False, True])
def test_v4_native_valid_arguments_keep_original_string_or_full_object(as_object):
    from aria.engine.policy.llm import OpenRouterLLMClient
    value = payload()
    value["safety_evidence"] = {"quote": "", "subject": "not_applicable",
                                "time": "not_applicable", "urgency": "not_applicable"}
    raw = " \n" + json.dumps(value) + "\n "
    data = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
        "tool_calls": [{"function": {"name": "aria_turn_decision", "arguments": value if as_object else raw}}]}}]}
    c = OpenRouterLLMClient(api_key="inert", role="online_fast", max_retries=0, fallback_models=[])
    result = c._extract_online_decision(data, version="online_turn_decision_v4_evidence")
    assert json.loads(result) == value
    if not as_object: assert result == raw


def test_default_is_not_silently_switched():
    c = create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture").online
    assert c.model == "qwen/qwen3.7-flash" and c.online_return_format == "json_object"
    assert c.online_structured_route is None and c.online_decision_version == VERSION
