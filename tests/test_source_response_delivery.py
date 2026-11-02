"""Neutral forced-output software probes; never semantic effect evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.risk.source_response import KEY, current_source_response
from aria.engine.policy.minimal_calls import MinimalCallGuard
from aria.engine.policy.online_decision import online_decision_tool_spec
from aria.engine.policy.safety_instruction import SafetyResponseInstruction, SYSTEM_INSTRUCTION_V3
from aria.interfaces.risk import RiskSignal
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence, V4
from tests.test_safety_response_instruction import bundle

TEXT = "中性软件输入甲。"
BODY = "中性支持正文甲。\n中性任务正文乙。"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def build(monkeypatch, value=None, *, premodel=False, fmt="json_schema_v1"):
    value = payload("fall", evidence(TEXT, "other", "historical", "uncertain"), BODY) if value is None else value
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
        clients=clients, safety_delivery_version="source_response_v3",
        safety_response_instruction_version="conditional_response_v3",
        premodel_safety_version="task_response_v2" if premodel else "legacy")
    if premodel:
        monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="neutral", evidence=[]))
    else:
        monkeypatch.setattr(engine._risk, "assess", lambda ctx: None)
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires, order


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
def test_complete_source_body_and_layout_survive_all_guards(monkeypatch, fmt):
    engine, ctx, wires, _ = build(monkeypatch, fmt=fmt)
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("care"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *a: effects.append("skill"))
    assert engine.process_turn(ctx, TEXT) == BODY
    assert len(wires) == 1 and wires[0][1] == {} and not effects
    assert SYSTEM_INSTRUCTION_V3 in wires[0][0]["messages"][0]["content"]
    assert "task_projection" not in online_decision_tool_spec(V4)["function"]["parameters"]["properties"]
    audit = ctx.working_memory["safety_delivery"]
    assert audit["status"] == "source_bound_guarded"
    assert audit["semantic_verified"] is audit["grants_permission"] is False
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("concern,row,status", [
    ("none", evidence(), "model_clear"),
    ("medical_emergency", evidence(TEXT, "owner", "current", "immediate"), "source_bound"),
    ("fall", evidence(TEXT, "other", "historical", "uncertain"), "source_bound"),
])
def test_premodel_current_assessment_replaces_old_keyword_uncertainty(monkeypatch, concern, row, status):
    engine, ctx, wires, order = build(monkeypatch, payload(concern, row, BODY), premodel=True)
    events = []
    def emit(event):
        order.append("prelude"); events.append(deepcopy(event)); return True
    before_profile = engine.persona.model_dump(mode="json")
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert result == BODY and order == ["prelude", "model"] and len(wires) == 1
    assert events[0]["premodel_safety_version"] == "task_response_v2"
    assert events[0]["protection"] and events[0]["protection"] != result
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == status
    assert ctx.working_memory["premodel_task"]["status"] == "response_only_complete"
    after_profile = engine.persona.model_dump(mode="json")
    assert len(after_profile["safety_screenings"]) == 1
    for name in ("safety_screenings",):
        after_profile.pop(name); before_profile.pop(name)
    before_profile["person_model_state"]["first_interaction_at"] = after_profile["person_model_state"]["first_interaction_at"]
    assert after_profile == before_profile


@pytest.mark.parametrize("failure", ["callback_false", "callback_raises", "client", "invalid_evidence", "action"])
def test_premodel_failure_keeps_protection_without_body_or_actions(monkeypatch, failure):
    value = (RuntimeError("software") if failure == "client" else
             payload("fall", evidence("not current", "other", "current", "immediate"), BODY)
             if failure == "invalid_evidence" else
             payload("fall", evidence(TEXT, "other", "current", "immediate"), '<action>{"tool":"x"}</action>')
             if failure == "action" else payload(response=BODY))
    engine, ctx, wires, _ = build(monkeypatch, value, premodel=True)
    events = []
    def emit(event):
        events.append(deepcopy(event))
        if failure == "callback_raises": raise OSError("software")
        return failure != "callback_false"
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert BODY not in result and "<action" not in result and "急救" in result
    assert len(wires) == (0 if failure.startswith("callback") else 1)
    assert "_premodel_task_inflight" not in ctx.working_memory


@pytest.mark.parametrize("text", ["我已经帮你发给家人了。", "我可以替你打电话。", "你肯定患了阿尔茨海默症。"])
def test_complete_body_still_passes_real_receipt_capability_medical_guards(monkeypatch, text):
    engine, ctx, _, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "historical", "uncertain"), text))
    result = engine.process_turn(ctx, TEXT)
    assert text not in result and ctx.working_memory["safety_delivery"]["status"] == "guard_replaced_with_protection"
    assert "_executed_action_signatures" not in ctx.working_memory


@pytest.mark.parametrize("change", ["user", "session", "scene", "turn", "utterance", "raw", "guarded", "assessment", "audit", "version"])
def test_bound_body_rejects_identity_source_and_assessment_drift(monkeypatch, change):
    engine, ctx, _, _ = build(monkeypatch)
    engine.process_turn(ctx, TEXT)
    ctx.turns.pop()  # inspect same turn before committing, using its original identity
    assert current_source_response(ctx, TEXT, engine.persona) is not None
    if change in ("user", "session", "scene"): setattr(ctx, change + "_id", "different")
    elif change == "turn": ctx.add_turn("different", "different")
    elif change == "utterance": return_assert = current_source_response(ctx, TEXT + "x", engine.persona); assert return_assert is None; return
    elif change in ("raw", "guarded"): ctx.working_memory[KEY] = replace(ctx.working_memory[KEY], **{change: "different"})
    elif change == "assessment": ctx.working_memory["online_turn_decision"]["safety_evidence"]["subject"] = "owner"
    elif change == "audit": ctx.working_memory["online_turn_decision"]["source_response"]["semantic_verified"] = True
    elif change == "version": ctx.working_memory["_safety_delivery_version"] = "task_context_v2"
    assert current_source_response(ctx, TEXT, engine.persona) is None


@pytest.mark.parametrize("source", ["我已经发给邻居了。", "我还没有发给邻居。", "我已经发给邻居了吗？", "假设我已经发给邻居了。", "他说他已经发给邻居了。"])
def test_only_entire_attributed_source_is_quote_not_system_receipt(source):
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_safety_delivery_version"] = "source_response_v3"
    person = PersonaProfile(user_id="owner")
    quote = "你提供的原文：“" + source + "”"
    assert guard_external_delivery_response(ctx, quote, source, person) == quote
    forged = quote + "\n我已经帮你发给家人了。"
    assert guard_external_delivery_response(ctx, forged, source, person) != forged


def test_free_paraphrase_is_not_silently_promoted_to_entailed_fact():
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_safety_delivery_version"] = "source_response_v3"
    source, target = "这是我收到的讲座海报。请改成一句简短的话。", "刚收到一份讲座海报。"
    assert guard_external_delivery_response(ctx, target, source, PersonaProfile(user_id="owner")) != target


def test_old_instruction_bytes_and_defaults_unchanged():
    from aria.engine.policy.safety_instruction import SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION_V2
    assert SafetyResponseInstruction().version == "legacy"
    assert SafetyResponseInstruction("conditional_response_v1").block(V4) == SYSTEM_INSTRUCTION
    assert SafetyResponseInstruction("conditional_response_v2").block(V4) == SYSTEM_INSTRUCTION_V2
    assert "task_projection" in SYSTEM_INSTRUCTION_V2 and "不返回 task_projection" in SYSTEM_INSTRUCTION_V3


def test_online_guard_replacement_is_never_resurrected_after_state_rollback(monkeypatch):
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


def test_partial_or_delimiter_broken_quote_never_masks_execution_claim():
    from aria.engine.risk.source_response import attributed_current_quote
    source = "我还没有发给邻居。"
    partial = "你提供的原文：“发给邻居。”我已经帮你发送了。"
    assert attributed_current_quote(partial, source) == partial
    source = '原文”我已经帮你发送了。“'
    body = "你提供的原文：“" + source + "”"
    assert attributed_current_quote(body, source) == body
