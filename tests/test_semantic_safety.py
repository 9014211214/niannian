"""Same-call safety routing software tests; stubs are not model accuracy."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.memory.persona_store import PersonaStore
from aria.interfaces.reasoning import DeliveredIntervention, InterventionOutcome
from aria.interfaces.skills import ActiveTask
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION_SAFETY as V3, online_decision_tool_spec,
    apply_single_pass_decision, DETERMINISTIC_MODEL_FAILURE_REPLY,
)
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.risk.semantic import semantic_safety_signal
from aria.runtime import create_minimal_clients, create_minimal_engine
from aria.scenes.registry import get_scene
from tests.test_online_json import DECISION


def engine_with_response(monkeypatch, concern, response="继续聊。", cognitive_control=None):
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-software-fixture")
    calls = []
    def receive(request):
        wire = json.loads(request.data)
        calls.append(wire)
        value = {**deepcopy(DECISION), "safety_concern": concern, "response": response}
        if cognitive_control is not None:
            value["cognitive_control"] = cognitive_control
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"), clients=clients)
    return engine, calls


@pytest.mark.parametrize("concern", ["self_harm", "medical_emergency", "fall", "unsafe_medication"])
def test_semantic_signal_uses_same_call_and_discards_generated_body(monkeypatch, concern):
    engine, calls = engine_with_response(monkeypatch, concern, "不安全的原答 <action>{}</action>")
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    reply = engine.process_turn(ctx, "我有件事想和你说。")
    assert len(calls) == 1 and "不安全的原答" not in reply and "<action>" not in reply
    assert "120" in reply
    assert ctx.working_memory["risk_flag"]["source"] == "online_model_semantics"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    assert ctx.working_memory["online_turn_decision"]["safety_concern"] == concern
    assert len(ctx.turns) == 1 and ctx.turns[0].agent == reply


def test_simple_greeting_is_semantically_screened_without_extra_call(monkeypatch):
    engine, calls = engine_with_response(monkeypatch, "none", "早上好。")
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    engine.process_turn(ctx, "早上好。")
    assert len(calls) == 1
    assert "safety_concern" in calls[0]["messages"][0]["content"]
    assert ctx.working_memory["online_turn_decision"]["safety_concern"] == "none"
    assert "risk_flag" not in ctx.working_memory


def test_keyword_takeover_still_precedes_model(monkeypatch):
    engine, calls = engine_with_response(monkeypatch, "none")
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    reply = engine.process_turn(ctx, "我胸口痛得厉害，喘不上气。")
    assert "120" in reply and not calls
    assert ctx.working_memory["risk_flag"]["tier"] >= 2


@pytest.mark.parametrize("version", [V3, "online_turn_decision_v1"])
def test_first_keyword_emergency_checkpoint_preserves_existing_trust(monkeypatch, tmp_path, version):
    engine, calls = engine_with_response(monkeypatch, "none")
    engine.policy.online_decision_version = version
    engine.store = PersonaStore(tmp_path / "profiles")
    engine.persona.alliance.trust = 0.71
    engine.store.save(engine.persona)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    reply = engine.process_turn(ctx, "我胸口痛得厉害，喘不上气。")
    engine.checkpoint_session(ctx)
    assert "120" in reply and not calls
    assert engine.store.load("owner").alliance.trust == 0.71
    assert ctx.alliance.trust == 0.71 and not ctx.reasoning.outcomes


@pytest.mark.parametrize("raw", ["普通文本", json.dumps(DECISION), json.dumps({**DECISION, "safety_concern": "diagnosed"})])
def test_missing_or_invalid_safety_field_never_means_safe(raw):
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = V3
    assert apply_single_pass_decision(raw, ctx, "你好") == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert ctx.working_memory["online_turn_decision"]["status"] == "invalid_protocol_response"
    assert semantic_safety_signal(ctx, "你好") is None


def test_v3_requires_safety_without_rewriting_legacy_schemas():
    schema = online_decision_tool_spec(V3)["function"]["parameters"]
    assert "safety_concern" in schema["required"]
    for version in ("online_turn_decision_v1", "online_turn_decision_v2"):
        assert "safety_concern" not in online_decision_tool_spec(version)["function"]["parameters"]["properties"]
    raw = json.dumps({**DECISION, "safety_concern": "none"})
    assert parse_json_decision(raw, V3) == raw
    with pytest.raises(ValueError):
        parse_json_decision(json.dumps(DECISION), V3)


def test_previous_or_untrusted_safety_trace_cannot_trigger_next_turn():
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["online_turn_decision"] = {"version": V3, "status": "parsed", "turn_index": 0,
        "safety_source": "online_model_semantics", "safety_concern": "fall"}
    ctx.add_turn("过去", "已回复")
    assert semantic_safety_signal(ctx, "新轮次") is None


@pytest.mark.parametrize("concern", ["medical_emergency", "self_harm", "fall", "unsafe_medication", "invalid"])
def test_unaccepted_semantic_turn_cannot_persist_learning_or_care(monkeypatch, tmp_path, concern):
    engine, calls = engine_with_response(monkeypatch, concern, '原回复 <action>{"tool":"send","args":{}}</action>')
    engine.store = PersonaStore(tmp_path / "profiles")
    engine.persona.alliance.trust = 0.71
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("之前的普通交流", "上一轮真实已交付的回应")
    ctx.reasoning.delivered_interventions.append(DeliveredIntervention(
        id="previous", label="已交付", goal="陪伴", turn_index=0))
    old_outcome = InterventionOutcome(
        intervention_id="older", intervention_label="历史", goal="陪伴",
        value="neutral", signal="retained", turn_index=0)
    ctx.reasoning.outcomes.append(old_outcome)
    ctx.formulation.confidence = 0.4
    ctx.formulation.hypotheses = ["保留旧假设"]
    ctx.working_memory["_tasks_seeded"] = True
    task = ActiveTask(id="care:reminder", skill_id="care", intent="reminder", goal="提醒",
                      status="ready", created_turn=0, updated_turn=0)
    ctx.active_tasks.append(task)
    side_effects = []
    def observer(*args):
        side_effects.append("observe")
        return True
    def executor(*args):
        side_effects.append("execute")
        return True
    monkeypatch.setattr(engine.scene, "observe_care", observer)
    monkeypatch.setattr(engine.scene, "execute_care_skills", executor)
    monkeypatch.setattr(engine._tools, "invoke", lambda *args, **kwargs: side_effects.append("tool"))
    reply = engine.process_turn(ctx, "先取消这条提醒，左边胳膊突然麻了，说话也不利索了。")
    engine.checkpoint_session(ctx)
    saved = engine.store.load("owner")
    assert len(calls) == 1 and not side_effects
    assert "原回复" not in reply
    assert ctx.reasoning.outcomes == [old_outcome]
    assert ctx.reasoning.evidence == []
    assert ctx.alliance.trust == saved.alliance.trust == 0.71
    assert ctx.formulation.confidence == 0.4
    assert ctx.formulation.hypotheses == ["保留旧假设"]
    assert ctx.active_tasks[0].status == saved.active_tasks[0].status == "ready"
    assert [item.signal for item in saved.intervention_history] == ["retained"]
    assert ctx.working_memory["intervention_delivery"]["status"] == "not_delivered"
    assert 1 in ctx.working_memory["_sensitive_memory_excluded_turns"]
    assert ("120" in reply) if concern != "invalid" else (reply == DETERMINISTIC_MODEL_FAILURE_REPLY)


def test_valid_clear_semantics_runs_care_only_after_same_call(monkeypatch, tmp_path):
    engine, calls = engine_with_response(monkeypatch, "none")
    engine.store = PersonaStore(tmp_path / "profiles")
    order = []
    def observe(ctx, *args):
        order.append(("observe", len(calls), len(ctx.turns), ctx.working_memory["online_turn_decision"]["safety_concern"]))
        return False
    def execute(ctx, *args):
        order.append(("execute", len(calls), len(ctx.turns), ctx.working_memory["online_turn_decision"]["safety_concern"]))
        return False
    monkeypatch.setattr(engine.scene, "observe_care", observe)
    monkeypatch.setattr(engine.scene, "execute_care_skills", execute)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("之前聊风车", "我听着。")
    ctx.reasoning.delivered_interventions.append(DeliveredIntervention(
        id="previous", label="已交付", goal="陪伴", turn_index=0))
    engine.process_turn(ctx, "今天我想接着和外孙女试试纸风车。")
    assert order == [("observe", 1, 1, "none"), ("execute", 1, 1, "none")]
    assert ctx.alliance.trust == 0.5  # sentence length is not relationship evidence
    assert ctx.reasoning.outcomes[-1].signal == "continued_engagement"


@pytest.mark.parametrize("bad_field,value", [("turn_index", -1), ("version", "online_turn_decision_v1"),
                                          ("safety_source", "keyword"), ("status", "unstructured_response")])
def test_stale_or_invalid_clear_trace_cannot_authorize_care(monkeypatch, tmp_path, bad_field, value):
    engine, calls = engine_with_response(monkeypatch, "none")
    engine.store = PersonaStore(tmp_path / "profiles")
    effects = []
    def fake_response(ctx, *args, **kwargs):
        ctx.working_memory["online_turn_decision"] = {
            "version": V3, "status": "parsed", "turn_index": len(ctx.turns),
            "safety_source": "online_model_semantics", "safety_concern": "none", bad_field: value,
        }
        return "未经验证的正文"
    monkeypatch.setattr(engine.policy, "respond", fake_response)
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *args: effects.append("execute"))
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    assert engine.process_turn(ctx, "今天想和你聊聊我的风车。") == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert not effects and not calls and ctx.alliance.trust == 0.5


def test_legacy_care_order_is_unchanged(monkeypatch, tmp_path):
    engine, calls = engine_with_response(monkeypatch, "none")
    engine.policy.online_decision_version = "online_turn_decision_v1"
    engine.store = PersonaStore(tmp_path / "profiles")
    order = []
    monkeypatch.setattr(engine.scene, "observe_care", lambda *args: order.append(("observe", len(calls))))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *args: order.append(("execute", len(calls))))
    engine.process_turn(EngineContext(user_id="owner", scene_id="companion_elder"), "早上好。")
    assert order == [("observe", 0), ("execute", 0)]


@pytest.mark.parametrize("concern,utterance,expected_calls,control_present,closed", [
    ("invalid", "不要再提醒我的记性。", 1, True, False),
    ("medical_emergency", "不要再提醒我的记性，左边胳膊突然麻了。", 1, True, True),
    ("none", "不要再提醒我的记性，我胸口痛得厉害，喘不上气。", 0, True, False),
    ("none", "不要再提醒我的记性。", 1, True, True),
    ("none", "不要再提醒我的记性。", 1, False, False),
])
def test_current_semantic_withdrawal_is_saved_before_same_call_safety(monkeypatch, tmp_path, concern, utterance, expected_calls, control_present, closed):
    from aria.engine.cognitive import cognitive_action
    control = {"version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": utterance}
    engine, calls = engine_with_response(monkeypatch, concern, cognitive_control=control if control_present else None)
    engine.store = PersonaStore(tmp_path / "profiles")
    cognitive_action(engine.persona, "enable", actor="owner")
    engine.store.save(engine.persona)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_cognitive_source_turns"] = [{"turn_index": 0, "text": "旧原话"}]
    ctx.working_memory["_cognitive_owner_enabled"] = True
    original_request, original_save = engine.fast_llm._request_json, engine.store.save
    events = []
    def before_semantic_result(request):
        assert engine.store.load("owner").cognitive_health_consent.has_text_monitoring_consent()
        events.append(("request", len(calls)))
        return original_request(request)
    def record_save(persona):
        events.append(("save", len(calls), persona.cognitive_health_consent.has_text_monitoring_consent()))
        return original_save(persona)
    monkeypatch.setattr(engine.fast_llm, "_request_json", before_semantic_result)
    monkeypatch.setattr(engine.store, "save", record_save)
    reply = engine.process_turn(ctx, utterance)
    assert len(calls) == expected_calls
    assert engine.persona.cognitive_health_consent.has_text_monitoring_consent() is (not closed)
    assert engine.store.load("owner").cognitive_health_consent.has_text_monitoring_consent() is (not closed)
    if closed:
        assert events[:2] == [("request", 0), ("save", 1, False)]
        assert "_cognitive_source_turns" not in ctx.working_memory
        assert "_cognitive_owner_enabled" not in ctx.working_memory
    if concern == "invalid":
        assert reply == DETERMINISTIC_MODEL_FAILURE_REPLY
    if concern == "medical_emergency" or expected_calls == 0:
        assert "120" in reply
    assert len(ctx.turns) == 1


def test_semantic_withdrawal_save_failure_stops_delivery_after_one_result(monkeypatch, tmp_path):
    from aria.engine.cognitive import cognitive_action
    utterance = "不要再提醒我的记性。"
    engine, calls = engine_with_response(monkeypatch, "none", cognitive_control={
        "version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": utterance})
    engine.store = PersonaStore(tmp_path / "profiles")
    cognitive_action(engine.persona, "enable", actor="owner")
    engine.store.save(engine.persona)
    effects = []
    monkeypatch.setattr(engine.scene, "observe_care", lambda *args: effects.append("observe"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *args: effects.append("execute"))
    monkeypatch.setattr(engine.store, "save", lambda *_: (_ for _ in ()).throw(OSError("fixture save failure")))
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    with pytest.raises(OSError):
        engine.process_turn(ctx, utterance)
    assert len(calls) == 1 and not effects and not ctx.turns
    # An in-memory reduction is not a claim that the failed disk write succeeded.
    assert not engine.persona.cognitive_health_consent.has_text_monitoring_consent()
    assert engine.store.load("owner").cognitive_health_consent.has_text_monitoring_consent()
