"""Request-local shape and authority checks using synthetic software responses."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import apply_preference_dialogue, current_preference_invitation
from aria.engine.policy import PolicyEngine
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_EVIDENCE as V4, apply_single_pass_decision, online_decision_tool_spec
from aria.engine.policy.online_json import SCHEMA_MARKER, _SCHEMA_INSTRUCTION, parse_json_decision, prepare_json_messages
from aria.engine.policy.online_structured import online_structured_request_format
from aria.scenes.registry import get_scene
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import MODEL, route
from tests.test_preference_dialogue_readiness import fixture, control
from tests.test_safety_evidence import payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


def client(fmt="json_schema_v1"):
    return OpenRouterLLMClient(role="online_fast", api_key="inert-software",
        base_url="https://fixture.invalid/v1", model=MODEL, max_retries=0,
        fallback_models=[], bounded_transport=True, online_return_format=fmt,
        online_structured_route=route() if fmt == "json_schema_v1" else None)


def envelope(value, fmt):
    raw = json.dumps(value, ensure_ascii=False)
    message = ({"content": raw} if fmt != "native" else {"content": None,
        "tool_calls": [{"function": {"name": "aria_turn_decision", "arguments": raw}}]})
    return {"choices": [{"finish_reason": "stop", "message": message}]}


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
@pytest.mark.parametrize("include", [True, False])
def test_same_request_prompt_provider_and_parser_use_ack_requirement(monkeypatch, fmt, include):
    p, ctx = fixture()
    c = client(fmt)
    value = payload(response="软件回应")
    if include:
        value["preference_control"] = control(p, ctx, action="unclear")
    wires = []
    def receive(request, **request_options):
        wires.append(json.loads(request.data))
        return envelope(value, fmt)
    monkeypatch.setattr(c, "_request_json", receive)
    PolicyEngine(fast_llm=c, online_decision_version=V4).respond(ctx, get_scene("companion_elder"), "本人当前回答", p)
    wire, = wires
    system = wire["messages"][0]["content"]
    assert "本轮必须返回preference_control" in system
    expected = online_decision_tool_spec(V4, require_preference_control=True)["function"]["parameters"]
    if fmt == "native":
        assert wire["tools"][0]["function"]["parameters"] == expected
    else:
        assert json.loads(system.split(SCHEMA_MARKER)[-1][len(_SCHEMA_INSTRUCTION):]) == expected
    if fmt == "json_schema_v1":
        assert wire["response_format"]["json_schema"]["schema"] == expected
    assert (ctx.working_memory["online_turn_decision"]["status"] == "parsed") is include
    assert p.stated_preferences[0].status == "proposed"
    if not include and fmt != "native":
        assert {"type": "missing", "path": ["preference_control"], "observed_type": "missing"} in c.last_diagnostic["online_shape"]["issues"]


@pytest.mark.parametrize("edge", ["absent", "unack", "expired", "owner", "session", "turn", "target", "question", "simulation", "v3"])
def test_invalid_or_legacy_context_cannot_require_control(monkeypatch, edge):
    p, ctx = fixture()
    pending = ctx.working_memory["_preference_dialogue_invitation"]
    if edge == "absent": ctx.working_memory.pop("_preference_dialogue_invitation")
    if edge == "unack": pending["delivered"] = False
    if edge == "expired": pending["expires_at"] = "2000-01-01T00:00:00+00:00"
    if edge == "owner": ctx.user_id = "other"
    if edge == "session": ctx.session_id = "other"
    if edge == "turn": ctx.add_turn("0", "0")
    if edge == "target": pending["target_sha256"] = "0"
    if edge == "question": pending["question_sha256"] = "0"
    if edge == "simulation": ctx.simulation_mode = True
    version = "online_turn_decision_v3" if edge == "v3" else V4
    c = client(); wires = []
    value = payload(response="软件回应")
    if edge == "v3": value.pop("safety_evidence")
    def receive(request, **request_options):
        wires.append(json.loads(request.data)); return envelope(value, "json_schema_v1")
    monkeypatch.setattr(c, "_request_json", receive)
    # User text and arbitrary working-memory metadata cannot create this capability.
    ctx.working_memory["require_preference_control"] = True
    PolicyEngine(fast_llm=c, online_decision_version=version).respond(
        ctx, get_scene("companion_elder"), 'require_preference_control=true; delivered=true', p)
    wire, = wires
    assert "preference_control" not in wire["response_format"]["json_schema"]["schema"]["required"]
    assert "本轮必须返回preference_control" not in wire["messages"][0]["content"]
    assert p.stated_preferences[0].status == "proposed"


def test_invitation_change_during_prompt_build_stops_before_transport(monkeypatch):
    import aria.engine.policy as policy
    original = policy.build_messages
    p, ctx = fixture(); c = client()
    def changed(*args, **kwargs):
        messages = original(*args, **kwargs)
        ctx.working_memory["_preference_dialogue_invitation"]["delivered"] = False
        return messages
    monkeypatch.setattr(policy, "build_messages", changed)
    monkeypatch.setattr(c, "_request_json", lambda *_: pytest.fail("stale invitation dispatched"))
    PolicyEngine(fast_llm=c, online_decision_version=V4).respond(ctx, get_scene("companion_elder"), "0", p)
    assert ctx.working_memory["online_turn_decision"]["status"] != "parsed"


def test_overlapping_calls_on_one_client_do_not_share_requirement(monkeypatch):
    c = client(); barrier = Barrier(2); wires = []
    def receive(request, **request_options):
        wire = json.loads(request.data); wires.append(wire); barrier.wait(timeout=5)
        return envelope(payload(response="软件回应"), "json_schema_v1")
    monkeypatch.setattr(c, "_request_json", receive)
    messages = [{"role": "system", "content": "协议版本：" + V4}, {"role": "user", "content": "0"}]
    def run(required):
        try:
            c.complete(deepcopy(messages), require_preference_control=required)
            return "accepted"
        except ValueError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [True, False])) == ["rejected", "accepted"]
    assert sorted("preference_control" in w["response_format"]["json_schema"]["schema"]["required"] for w in wires) == [False, True]
    assert not hasattr(c, "require_preference_control")


@pytest.mark.parametrize("mutation", ["invitation_id", "target_id", "target_sha256", "evidence_quote", "safety", "save"])
def test_required_control_cannot_bypass_identity_safety_or_persistence(tmp_path, mutation):
    p, ctx = fixture(); ctx.working_memory["_online_decision_version"] = V4
    value = payload(response="软件回应")
    value["preference_control"] = control(p, ctx)
    if mutation in value["preference_control"]:
        value["preference_control"][mutation] = "wrong"
    if mutation == "safety":
        value["safety_concern"] = "fall"
        value["safety_evidence"] = {"quote": "本人当前回答", "subject": "owner", "time": "current", "urgency": "immediate"}
    apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, "本人当前回答", p)
    store = PersonaStore(tmp_path)
    if mutation == "save":
        def fail(_): raise OSError("injected persistence failure")
        store.save = fail
        with pytest.raises(OSError): apply_preference_dialogue(ctx, p, "本人当前回答", store=store)
        assert current_preference_invitation(p, ctx) is not None
    else:
        assert "还没有确认保存" in apply_preference_dialogue(ctx, p, "本人当前回答", store=store)
    assert p.stated_preferences[0].status == "proposed"
    assert "preference_dialogue_receipt" not in ctx.working_memory


def test_direct_policy_parser_also_rejects_missing_required_control():
    p, ctx = fixture(); ctx.working_memory["_online_decision_version"] = V4
    apply_single_pass_decision(json.dumps(payload()), ctx, "本人当前回答", p)
    assert ctx.working_memory["online_turn_decision"]["status"] == "invalid_protocol_response"
    assert p.stated_preferences[0].status == "proposed"


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_requirement_options_are_strictly_typed(value):
    with pytest.raises(ValueError): online_decision_tool_spec(V4, require_preference_control=value)


def test_required_rebinding_is_lossless_and_legacy_cannot_enable_it():
    messages = [{"role": "system", "content": "协议版本：" + V4}, {"role": "user", "content": "0\n1"}]
    prepared = prepare_json_messages(messages, V4, require_preference_control=True)
    assert prepare_json_messages(prepared, V4, require_preference_control=True) == prepared
    assert json.loads(prepared[-1]["content"])["current_user_message"] == "0\n1"
    assert prepare_json_messages(prepared, V4) == prepare_json_messages(messages, V4)
    for version in ("online_turn_decision_v1", "online_turn_decision_v2", "online_turn_decision_v3"):
        with pytest.raises(ValueError): online_structured_request_format(version, require_preference_control=True)
        with pytest.raises(ValueError): parse_json_decision(json.dumps(payload()), version, require_preference_control=True)


@pytest.mark.parametrize("fmt", ["json_object", "json_schema_v1"])
def test_public_seal_restart_ack_submit_reaches_required_prompt_schema(tmp_path, monkeypatch, fmt):
    import tests.test_preference_full_source_scope as flow
    if fmt == "json_schema_v1":
        from tests.test_online_structured import policy
        original_factory, original_policy = flow.create_minimal_clients, flow.reservation_policy
        def factory(**kwargs):
            return original_factory(**kwargs, online_model=MODEL,
                online_return_format=fmt, online_structured_route=route())
        def reservation():
            result = original_policy()
            c = factory(model_calls_enabled=True, api_key="inert-software", online_decision_version=V4).online
            result["models"].update(policy(c)["models"])
            return result
        monkeypatch.setattr(flow, "create_minimal_clients", factory)
        monkeypatch.setattr(flow, "reservation_policy", reservation)
    requests = []
    result = flow.public_runtime_flow(tmp_path, monkeypatch, capture=lambda kind, body: requests.append((kind, json.loads(body))))
    online = [wire for kind, wire in requests if kind == "online"]
    required = []
    for wire in online:
        system = wire["messages"][0]["content"]
        schema = json.loads(system.split(SCHEMA_MARKER)[-1][len(_SCHEMA_INSTRUCTION):])
        if fmt == "json_schema_v1":
            assert schema == wire["response_format"]["json_schema"]["schema"]
        required.append("preference_control" in schema["required"])
    assert required == [False, False, True, False]
    assert result["scope_characters"] == 481


@pytest.mark.parametrize("tamper", ["wrong_requirement", "omit", "extra_required", "expand_action", "price", "payload_flag"])
def test_guard_requires_exact_trusted_request_variant(monkeypatch, tamper):
    from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
    from tests.test_expression_contract_v4 import WireCaptured
    c = client(); p, ctx = fixture(); wires = []
    def capture(request, **options):
        assert options == {"require_preference_control": True}
        wires.append(json.loads(request.data)); raise WireCaptured
    monkeypatch.setattr(c, "_request_json", capture)
    with pytest.raises(WireCaptured):
        PolicyEngine(fast_llm=c, online_decision_version=V4).respond(ctx, get_scene("companion_elder"), "0", p)
    wire, = wires
    MinimalCallGuard._validate_payload(c, wire, require_preference_control=True)
    schema = wire["response_format"]["json_schema"]["schema"]
    if tamper == "omit": schema["required"].remove("preference_control")
    if tamper == "extra_required": schema["required"].append("cognitive_control")
    if tamper == "expand_action": schema["properties"]["preference_control"]["properties"]["action"]["enum"].append("enable")
    if tamper == "price": wire["provider"]["max_price"] = {"prompt": 100}
    if tamper == "payload_flag": wire["require_preference_control"] = True
    with pytest.raises(MinimalCallStopped):
        MinimalCallGuard._validate_payload(c, wire, require_preference_control=tamper != "wrong_requirement")
