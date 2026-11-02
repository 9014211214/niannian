"""Neutral forced-label developer probes; no semantic or clinical accuracy claim."""
from copy import deepcopy
from dataclasses import replace
import io
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy import PolicyEngine
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.online_decision import online_decision_tool_spec
from aria.engine.policy.online_json import parse_json_decision, prepare_json_messages
from aria.engine.policy.safety_instruction import SafetyResponseInstruction, SYSTEM_INSTRUCTION_V2
from aria.engine.risk.safety_task import KEY, current_task_projection
from aria.engine.risk.safety_delivery import SEGMENTS
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from tests._model_http_stub import stub_model_http
from tests.test_preference_required_control import envelope
from tests.test_preference_dialogue_readiness import fixture, control
from tests.test_safety_evidence import payload, evidence, V4
from tests.test_safety_response_instruction import bundle
from tests.test_safety_response_runtime import options

TEXT = "软件请求来源甲。"
TASK = "测试甲\n测试乙"
BODY = "未被选择的原答旁白。\n" + TASK


def projected(concern="fall", row=None, text=TASK, body=None):
    value = payload(concern, row if row is not None else evidence(TEXT, "other", "historical", "uncertain"),
                    BODY if body is None else body)
    value["task_projection"] = {"version": "task_projection_v1", "status": "in_dialogue_task",
                                "request_quote": TEXT, "response_quote": text}
    return value


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


def build(monkeypatch, value, fmt="json_schema_v1"):
    clients = bundle(fmt)
    wires = []
    def receive(request, **kwargs):
        wire = json.loads(request.data)
        MinimalCallGuard._validate_payload(clients.online, wire, **kwargs)
        wires.append((wire, kwargs))
        return envelope(value, fmt)
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
        clients=clients, safety_delivery_version="task_context_v2",
        safety_response_instruction_version="conditional_response_v2")
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires


@pytest.mark.parametrize("concern", ["self_harm", "medical_emergency", "fall", "unsafe_medication"])
@pytest.mark.parametrize("subject", ["owner", "other", "quotation_or_hypothetical", "uncertain"])
def test_task_projection_survives_safety_without_restoring_whole_raw_or_actions(monkeypatch, concern, subject):
    engine, ctx, wires = build(monkeypatch, projected(concern, evidence(TEXT, subject, "current", "immediate")))
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("care"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *a: effects.append("skill"))
    result = engine.process_turn(ctx, TEXT)
    audit = ctx.working_memory["safety_delivery"]
    assert result.endswith(TASK) and "未被选择的原答旁白" not in result
    assert SEGMENTS["B-capability"] in result and not effects and len(wires) == 1
    assert audit["task_status"] == "included_unverified_semantics"
    assert audit["semantic_verified"] is False and audit["grants_permission"] is False
    assert audit["safety_context_status"] == "source_bound"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    assert not engine.persona.episodic_memory and not engine.persona.person_hypotheses
    assert "120" not in result and "12356" not in result


@pytest.mark.parametrize("row", [evidence("not present", "owner", "current", "immediate"),
    evidence(TEXT, "not_applicable", "current", "immediate")])
def test_invalid_safety_source_cannot_destroy_independently_bound_task_or_assert_label(monkeypatch, row):
    engine, ctx, _ = build(monkeypatch, projected(row=row))
    result = engine.process_turn(ctx, TEXT)
    audit = ctx.working_memory["safety_delivery"]
    assert result.endswith(TASK) and SEGMENTS["U-owner / U-other"] in result
    assert audit["subject"] == "unverified" and audit["concern"] is None
    assert audit["safety_context_status"] == "unverified" and not engine.persona.risk_events
    assert ctx.working_memory["online_turn_decision"]["safety_assessment"]["verbatim_source"] is False or row["subject"] == "not_applicable"


@pytest.mark.parametrize("mutation", ["source", "body", "none_with_text", "action", "internal"])
def test_unbound_or_internal_task_rejected_without_losing_conditional_backstop(monkeypatch, mutation):
    value = projected(row=evidence("absent", "owner", "current", "immediate"))
    if mutation == "source": value["task_projection"]["request_quote"] = "missing source"
    elif mutation == "body": value["task_projection"]["response_quote"] = "missing answer"
    elif mutation == "none_with_text": value["task_projection"]["status"] = "none"
    elif mutation == "action":
        value["response"] = value["task_projection"]["response_quote"] = '<action>{"tool":"x","args":{}}</action>'
    elif mutation == "internal":
        value["response"] = value["task_projection"]["response_quote"] = json.dumps(payload())
    engine, ctx, _ = build(monkeypatch, value)
    result = engine.process_turn(ctx, TEXT)
    assert TASK not in result and "<action" not in result and "state_signals" not in result
    assert SEGMENTS["U-owner / U-other"] in result
    assert ctx.working_memory["safety_delivery"]["task_status"] == "missing_or_rejected"


@pytest.mark.parametrize("text", ["我已经帮你发给家人了。", "我可以替你打电话。", "你肯定患了阿尔茨海默症。"])
def test_existing_action_truth_capability_and_medical_guards_apply_to_projected_text(monkeypatch, text):
    engine, ctx, _ = build(monkeypatch, projected(text=text, body="旁白。" + text))
    result = engine.process_turn(ctx, TEXT)
    assert text not in result
    assert ctx.working_memory["safety_delivery"]["task_status"] == "rejected_by_existing_guard"
    assert "_executed_action_signatures" not in ctx.working_memory


def test_pre_model_keyword_remains_timely_without_owner_or_region_assertion(monkeypatch):
    engine, ctx, wires = build(monkeypatch, projected())
    result = engine.process_turn(ctx, "我胸口痛得厉害，喘不上气。")
    assert not wires and SEGMENTS["U-owner / U-other"] in result
    assert "120" not in result and "12356" not in result
    audit = ctx.working_memory["safety_delivery"]
    assert audit["subject"] == audit["time"] == audit["urgency"] == "unverified"
    assert audit["task_status"] == "missing_or_rejected" and not engine.persona.risk_events
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("mutation", ["turn", "utterance", "user", "scene", "session", "body_hash", "projection_hash", "permission"])
def test_projection_cannot_be_reused_for_another_context(monkeypatch, mutation):
    engine, ctx, _ = build(monkeypatch, projected())
    engine.process_turn(ctx, TEXT)
    ctx.turns.pop()  # inspect the same frozen turn before it was appended
    assert current_task_projection(ctx, TEXT, engine.persona) is not None
    utterance = TEXT
    if mutation == "turn": ctx.add_turn("next", "next")
    elif mutation == "utterance": utterance = "other"
    elif mutation == "user": ctx.user_id = "other"
    elif mutation == "scene": ctx.scene_id = "other"
    elif mutation == "session": ctx.session_id = "other"
    elif mutation == "body_hash": ctx.working_memory["online_turn_decision"]["parsed_response_sha256"] = "0" * 64
    elif mutation == "projection_hash": ctx.working_memory["online_turn_decision"]["task_projection"]["response_quote_sha256"] = "0" * 64
    elif mutation == "permission": ctx.working_memory[KEY] = replace(ctx.working_memory[KEY], grants_permission=True)
    assert current_task_projection(ctx, utterance, engine.persona) is None


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
@pytest.mark.parametrize("ack", [False, True])
def test_full_same_call_wire_shape_and_guard_are_request_local(monkeypatch, fmt, ack):
    p, ctx = fixture() if ack else (PersonaProfile(user_id="owner"), EngineContext(user_id="owner", scene_id="companion_elder"))
    c = bundle(fmt).online
    value = projected()
    if ack: value["preference_control"] = control(p, ctx, action="unclear")
    wires = []
    def receive(request, **kwargs):
        wire = json.loads(request.data); wires.append((wire, kwargs))
        MinimalCallGuard._validate_payload(c, wire, **kwargs)
        if fmt != "json_object":
            with pytest.raises(MinimalCallStopped): MinimalCallGuard._validate_payload(c, wire)
        return envelope(value, fmt)
    monkeypatch.setattr(c, "_request_json", receive)
    policy = PolicyEngine(fast_llm=c, online_decision_version=V4, safety_response_instruction_version="conditional_response_v2")
    policy.respond(ctx, get_scene("companion_elder"), TEXT, p)
    wire, kwargs = wires[0]
    assert kwargs == {"require_task_projection": True, **({"require_preference_control": True} if ack else {})}
    schema = online_decision_tool_spec(V4, require_task_projection=True, require_preference_control=ack)["function"]["parameters"]
    assert "task_projection" in schema["required"]
    assert wire["messages"][0]["content"].count(SYSTEM_INSTRUCTION_V2) == 1
    if fmt == "native": assert wire["tools"][0]["function"]["parameters"] == schema
    elif fmt == "json_schema_v1": assert wire["response_format"]["json_schema"]["schema"] == schema
    assert ctx.working_memory["online_turn_decision"]["status"] == "parsed"
    assert not hasattr(c, "require_task_projection")


def test_schema_requires_projection_only_on_explicit_request_and_rebinding_is_exact():
    value = projected()
    assert parse_json_decision(json.dumps(value), V4, require_task_projection=True)
    with pytest.raises(ValueError): parse_json_decision(json.dumps(value), V4)
    value.pop("task_projection")
    assert parse_json_decision(json.dumps(value), V4)
    with pytest.raises(ValueError): parse_json_decision(json.dumps(value), V4, require_task_projection=True)
    messages = [{"role": "system", "content": "协议版本：" + V4}, {"role": "user", "content": TEXT}]
    for preference in (False, True):
        one = prepare_json_messages(messages, V4, require_task_projection=True, require_preference_control=preference)
        assert prepare_json_messages(one, V4, require_task_projection=True, require_preference_control=preference) == one
        assert prepare_json_messages(one, V4) == prepare_json_messages(messages, V4)


@pytest.mark.parametrize("bad", [None, 1, "true", [], {}])
def test_projection_option_is_strict_boolean(bad):
    with pytest.raises(ValueError): online_decision_tool_spec(V4, require_task_projection=bad)


@pytest.mark.parametrize("fmt", ["json_object", "json_schema_v1"])
def test_public_durable_guard_save_and_restart_with_no_sensitive_memory(tmp_path, monkeypatch, fmt):
    calls = []
    def receive(request, **kwargs):
        calls.append(request)
        value = projected()
        data = {"id": "gen-task-projection-fixture", "model": json.loads(request.data)["model"], "provider": "Alibaba",
                "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1}, **envelope(value, fmt)}
        class Response:
            status = 200
            headers = {}
            def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, size=-1): return self.body.read(size)
        return Response()
    stub_model_http(monkeypatch, receive)
    def opts():
        result = options(fmt, "conditional_response_v2")
        result["safety_delivery_version"] = "task_context_v2"
        return result
    app = MinimalWebApp(tmp_path, **opts()); resumed = None
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "task-fixture-1", sid)
        assert result["ok"], result
        assert result["state"]["turns"][-1]["response"].endswith(TASK)
        assert len(app.call_guard.snapshot()["attempts"]) == len(calls) == 1
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.safety_delivery_version == "task_context_v2"
        assert saved.pending_session.safety_response_instruction_version == "conditional_response_v2"
        assert not saved.pending_session.turns and not saved.risk_events
        resumed = MinimalWebApp(tmp_path, **opts())
        assert resumed.start()["state"]["session_id"] == sid
        assert KEY not in resumed._sessions["fictional-chen"][1].working_memory
    finally:
        app._sessions.clear(); app.close()
        if resumed: resumed.close()


def test_candidate_requires_bound_instruction_and_does_not_change_legacy():
    with pytest.raises(ValueError, match="requires_response_v2"):
        create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
                              clients=bundle(), safety_delivery_version="task_context_v2")
    assert SafetyResponseInstruction().block(V4) == ""
    assert "task_projection" not in online_decision_tool_spec(V4)["function"]["parameters"]["properties"]


def test_sensitive_observation_is_not_shared_by_task_projection(monkeypatch):
    from tests.test_cognitive_change_v2 import enabled, record
    text = "给女儿的草稿：我忘了交电费，耽误了事情。"
    engine, ctx, _ = build(monkeypatch, projected(text=text, body="旁白。" + text))
    engine.persona = enabled()
    record(engine.persona, 15, "daily_function", True)
    result = engine.process_turn(ctx, TEXT)
    assert "我忘了交电费" not in result
    assert "cognitive_private_sharing_blocked" in ctx.working_memory["delivery_response_guard"]["violations"]
    assert ctx.working_memory["safety_delivery"]["task_status"] == "rejected_by_existing_guard"


@pytest.mark.parametrize("guard", ["guard_response", "guard_delivery_response"])
def test_required_guard_unavailable_never_releases_model_task(monkeypatch, guard):
    engine, ctx, _ = build(monkeypatch, projected())
    monkeypatch.setattr(engine.scene, guard, None)
    assert TASK not in engine.process_turn(ctx, TEXT)
    assert ctx.working_memory["safety_delivery"]["task_status"] == "required_scene_guard_unavailable"


def test_none_projection_and_missing_protocol_do_not_reuse_a_prior_task(monkeypatch):
    value = projected()
    engine, ctx, calls = build(monkeypatch, value)
    assert TASK in engine.process_turn(ctx, TEXT)
    value["task_projection"] = {"version": "task_projection_v1", "status": "none", "request_quote": "", "response_quote": ""}
    assert TASK not in engine.process_turn(ctx, TEXT)
    assert KEY not in ctx.working_memory
    value.pop("task_projection")
    assert TASK not in engine.process_turn(ctx, TEXT)
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == "unverified"
    assert len(calls) == 3


def test_v2_instruction_is_coherent_and_v1_exact_constant_is_not_rewritten():
    from aria.engine.policy.safety_instruction import SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION_SHA256
    import hashlib
    assert hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest() == SYSTEM_INSTRUCTION_SHA256
    assert SYSTEM_INSTRUCTION.startswith("## 当次安全回应要求（conditional_response_v1）")
    assert SYSTEM_INSTRUCTION_V2.startswith("## 当次安全回应要求（conditional_response_v2）")
    assert "只输出原协议要求的结果，不增加字段" not in SYSTEM_INSTRUCTION_V2
    assert "v4 加 task_projection" in SYSTEM_INSTRUCTION_V2
