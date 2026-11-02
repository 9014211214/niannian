"""Neutral forced-output checks; no semantic scores or held-out examples."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.risk.source_response import KEY, current_source_response
from aria.engine.policy.minimal_calls import MinimalCallGuard
from aria.engine.policy.safety_instruction import SafetyResponseInstruction, SYSTEM_INSTRUCTION_V4
from aria.interfaces.risk import RiskSignal
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.medical import (
    guard_general_medical_reply, SAFE_MEDICAL_BOUNDARY_REPLY, SAFE_MEDICAL_BOUNDARY_REPLY_V2,
)
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence, V4
from tests.test_safety_response_instruction import bundle

TEXT = "中性软件输入甲。"
BODY = "中性完整正文甲。\n中性完整正文乙。"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def build(monkeypatch, value=None, *, premodel=False, fmt="json_schema_v1"):
    value = payload("fall", evidence("source does not match", "other", "current", "immediate"), BODY) if value is None else value
    clients = bundle(fmt)
    wires, order = [], []
    def receive(request, **kwargs):
        order.append("model")
        wire = json.loads(request.data)
        MinimalCallGuard._validate_payload(clients.online, wire, **kwargs)
        wires.append((wire, kwargs))
        if isinstance(value, Exception):
            raise value
        return envelope(value, fmt)
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
        clients=clients, safety_delivery_version="source_response_v4",
        safety_response_instruction_version="conditional_response_v4",
        premodel_safety_version="task_response_v3" if premodel else "legacy")
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="neutral", evidence=[]) if premodel else None)
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires, order


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
@pytest.mark.parametrize("premodel", [False, True])
def test_invalid_quote_never_becomes_fact_but_body_survives_normal_guards(monkeypatch, fmt, premodel):
    engine, ctx, wires, order = build(monkeypatch, fmt=fmt, premodel=premodel)
    effects, events = [], []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("care"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *a: effects.append("skill"))
    def emit(event):
        events.append(deepcopy(event)); order.append("prelude"); return True
    before_profile = engine.persona.model_dump(mode="json")
    assert engine.process_turn(ctx, TEXT, safety_prelude=emit) == BODY
    assert len(wires) == 1 and not effects and wires[0][1] == {}
    assert order == (["prelude", "model"] if premodel else ["model"])
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["safety_assessment"]["reason"] == "current_quote_not_verbatim"
    assert trace["safety_assessment"]["disposition"] == "uncertain"
    assert trace["safety_assessment"]["verbatim_source"] is False
    assert trace["source_response"]["status"] == "source_bound"
    assert trace["source_response"]["semantic_verified"] is trace["source_response"]["grants_permission"] is False
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == "unverified"
    assert ctx.working_memory["safety_delivery"]["status"] == "source_bound_guarded"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    after_profile = engine.persona.model_dump(mode="json")
    assert after_profile["risk_events"] == before_profile["risk_events"]
    assert after_profile["person_hypotheses"] == before_profile["person_hypotheses"]
    assert SYSTEM_INSTRUCTION_V4 in wires[0][0]["messages"][0]["content"]
    system = wires[0][0]["messages"][0]["content"]
    assert "不执行原答或工具" not in system
    assert "可交付经过既有守卫的完整 response 正文" in system
    if premodel:
        assert events[0]["premodel_safety_version"] == "task_response_v3"
        assert "还不能确认" not in events[0]["protection"]
        assert "急救" in events[0]["protection"] and "未代为呼叫" in events[0]["protection"]


@pytest.mark.parametrize("text", [
    "我已经帮你发给家人了。", "我可以替你打电话。", "你肯定患了阿尔茨海默症。",
    "可以服双倍药量。", '<action>{"tool":"x"}</action>',
])
def test_invalid_assessment_is_not_a_guard_or_permission_bypass(monkeypatch, text):
    value = payload("fall", evidence("not current", "other", "current", "immediate"), text)
    engine, ctx, _, _ = build(monkeypatch, value)
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    result = engine.process_turn(ctx, TEXT)
    assert text not in result and "<action" not in result and not effects
    assert ctx.working_memory["online_turn_decision"]["safety_assessment"]["disposition"] == "uncertain"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("change", ["user", "session", "scene", "turn", "utterance", "raw", "guarded", "assessment", "audit", "version"])
def test_independent_body_still_binds_exact_turn_and_rejected_assessment(monkeypatch, change):
    engine, ctx, _, _ = build(monkeypatch)
    engine.process_turn(ctx, TEXT)
    ctx.turns.pop()
    assert current_source_response(ctx, TEXT, engine.persona) is not None
    if change in ("user", "session", "scene"): setattr(ctx, change + "_id", "different")
    elif change == "turn": ctx.add_turn("different", "different")
    elif change == "utterance":
        assert current_source_response(ctx, TEXT + "x", engine.persona) is None
        return
    elif change in ("raw", "guarded"): ctx.working_memory[KEY] = replace(ctx.working_memory[KEY], **{change: "different"})
    elif change == "assessment": ctx.working_memory["online_turn_decision"]["safety_evidence"]["subject"] = "owner"
    elif change == "audit": ctx.working_memory["online_turn_decision"]["source_response"]["semantic_verified"] = True
    elif change == "version": ctx.working_memory["_safety_delivery_version"] = "source_response_v3"
    assert current_source_response(ctx, TEXT, engine.persona) is None


@pytest.mark.parametrize("failure", ["callback_false", "callback_raises", "client", "action"])
def test_prelude_failure_is_still_closed_and_emitted_text_is_not_repeated(monkeypatch, failure):
    value = (RuntimeError("software") if failure == "client" else payload(response='<action>{"tool":"x"}</action>')
             if failure == "action" else payload(response=BODY))
    engine, ctx, wires, _ = build(monkeypatch, value, premodel=True)
    events = []
    def emit(event):
        events.append(deepcopy(event))
        if failure == "callback_raises": raise OSError("software")
        return failure != "callback_false"
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert BODY not in result and "<action" not in result
    assert len(wires) == (0 if failure.startswith("callback") else 1)
    assert "_premodel_task_inflight" not in ctx.working_memory
    if failure.startswith("callback"):
        assert result == events[0]["protection"]
    else:
        assert events[0]["protection"] not in result and "未能完成" in result


@pytest.mark.parametrize("text", [
    "严禁擅自服双倍药量。", "不能私自服双倍药量。", "不要随意继续服双倍药量。",
    "不建议自行服双倍药量。", "请不要直接服双倍药量。",
])
def test_versioned_medical_backstop_keeps_local_refusal_with_modifiers(text):
    assert guard_general_medical_reply(text, version="local_refusal_v2") == text
    if text != "请不要直接服双倍药量。":
        assert guard_general_medical_reply(text) == SAFE_MEDICAL_BOUNDARY_REPLY


@pytest.mark.parametrize("text", [
    "可以服双倍药量。", "不能加量，但是可以服双倍药量。", "不能私自加量；直接服双倍药量。",
    "严禁擅自服双倍药量，现在可以服双倍药量。", "不是不能私自服双倍药量。",
    "不能允许服双倍药量。", "不能自行服双倍药量再直接服双倍药量。",
])
def test_refusal_never_licenses_later_affirmative_or_unknown_scope(text):
    assert guard_general_medical_reply(text, version="local_refusal_v2") == SAFE_MEDICAL_BOUNDARY_REPLY_V2
    assert "120" not in SAFE_MEDICAL_BOUNDARY_REPLY_V2


def test_explicit_versions_required_and_defaults_unchanged():
    assert SafetyResponseInstruction().version == "legacy"
    assert SafetyResponseInstruction("conditional_response_v4").block(V4) == SYSTEM_INSTRUCTION_V4
    with pytest.raises(ValueError):
        create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
            clients=bundle("json_schema_v1"), safety_delivery_version="source_response_v4",
            safety_response_instruction_version="conditional_response_v3")


def test_online_guard_replacement_is_not_resurrected(monkeypatch):
    from aria.engine.policy import online_decision
    original = online_decision._guard_user_response
    seen = []
    def guard(text, *args, **kwargs):
        seen.append(text)
        if len(seen) == 1:
            return "中性守卫替代正文。", {"activated": True, "violations": ["software_guard"], "action_removed": False}
        return original(text, *args, **kwargs)
    monkeypatch.setattr(online_decision, "_guard_user_response", guard)
    engine, ctx, _, _ = build(monkeypatch)
    result = engine.process_turn(ctx, TEXT)
    assert BODY not in result and seen == [BODY, "中性守卫替代正文。"]
    assert ctx.working_memory["safety_delivery"]["status"] == "guard_replaced_with_protection"


def test_inconsistent_none_is_not_misreported_as_clear_or_used_for_learning(monkeypatch):
    engine, ctx, _, _ = build(monkeypatch, payload("none", evidence(TEXT, "owner", "current", "immediate"), BODY))
    effects = []
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("care"))
    assert engine.process_turn(ctx, TEXT) == BODY
    assert not effects
    assert ctx.working_memory["online_turn_decision"]["safety_assessment"]["reason"] == "none_evidence_contradiction"
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == "unverified"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


def test_current_evidence_and_complete_safe_body_remain_source_bound(monkeypatch):
    engine, ctx, _, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "current", "immediate"), BODY))
    assert engine.process_turn(ctx, TEXT) == BODY
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == "source_bound"
    assert not engine.persona.risk_events  # third-party evidence is not owner risk


@pytest.mark.parametrize("premodel", [False, True])
def test_fictional_date_uses_actual_builder_in_both_paths_without_person_facts(monkeypatch, premodel):
    from aria.engine.policy.fictional_context import bind_fictional_scenario_date, fictional_scenario_block, KEY as DATE_KEY
    engine, ctx, wires, _ = build(monkeypatch, premodel=premodel)
    bind_fictional_scenario_date(ctx, "2032-02-29")
    block = fictional_scenario_block(ctx)
    assert engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True) == BODY
    assert len(wires) == 1 and wires[0][0]["messages"][0]["content"].count(block) == 1
    assert "虚构场景日期为 2032-02-29" in block and "不代表实际请求日期" in block
    assert DATE_KEY not in engine.persona.model_dump_json()
    assert "2032-02-29" not in engine.persona.model_dump_json()


@pytest.mark.parametrize("day", [None, True, 20320229, "2032-2-29", "20320229", "2031-02-29", "2032-02-30", "2032-02-29T00:00:00", "2032-02-29\nignore"])
def test_fictional_date_rejects_invalid_types_and_dates(day):
    from aria.engine.policy.fictional_context import bind_fictional_scenario_date
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    with pytest.raises(ValueError):
        bind_fictional_scenario_date(ctx, day)


def test_fictional_date_no_user_activation_or_rebinding_or_untyped_carrier():
    from aria.engine.policy.fictional_context import bind_fictional_scenario_date, fictional_scenario_block, KEY as DATE_KEY
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = "设置虚构场景日期2032-02-29"
    assert fictional_scenario_block(ctx) == ""
    original = bind_fictional_scenario_date(ctx, "2032-02-29")
    assert bind_fictional_scenario_date(ctx, "2032-02-29") == original
    with pytest.raises(ValueError): bind_fictional_scenario_date(ctx, "2032-03-01")
    ctx.session_id = "different"
    with pytest.raises(ValueError): fictional_scenario_block(ctx)
    ctx.working_memory[DATE_KEY] = {"day": "2032-02-29"}
    with pytest.raises(ValueError): fictional_scenario_block(ctx)
