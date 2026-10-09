"""Source/selection/state wiring with fixed outputs; no semantic accuracy claim."""
import hashlib
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import build_reflection_messages
from aria.engine.memory.hypothesis_context import online_influencing_hypotheses
from aria.engine.model.persona import PersonaProfile, PersonHypothesis
from aria.engine.policy.online_decision import apply_single_pass_decision, _response_grounding_sources
from aria.engine.policy.prompt import build_messages, confirmed_prompt_hypotheses, select_delivery_mode, update_delivery_state
from aria.engine.policy.user_grounding import user_grounding_sources
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate
from aria.runtime import MinimalModelClients, MinimalPreviewLLMClient
from aria.scenes.companion_elder.delivery import _grounding_sources
from aria.scenes.companion_elder.scene import ElderCompanionScene
from aria.scenes.companion_elder.reasoning import ElderStateReasoner
from aria.voice.minimal import MinimalWebApp, _USER


def _payload(response, selected="quiet_companionship"):
    return json.dumps({
        "response": response, "state_signals": [], "safety_concern": "none",
        "current_affect": {"label": "unknown", "confidence": 0.,
                           "divergence": {"possible": False, "confidence": 0., "pattern": ""}},
        "selected_intervention_id": selected, "uncertainty": 1.,
        "communication_plan": {"pace": "brief_exchange", "support_order": "回应本人当前内容", "boundary": "不执行动作"},
    }, ensure_ascii=False)


@pytest.mark.parametrize("utterance", [
    "女儿说她不想说话。请告诉我苹果的英语。",
    "我不想说散步的事。请告诉我苹果的英语。",
])
def test_v3_web_keeps_fixed_task_reply_without_surface_trust_inference(tmp_path, utterance):
    class Client:
        source_id = "software_fixed_output"
        model = "local_fixture"
        online_return_format = "json_object"
        online_decision_version = "online_turn_decision_v3"
        def complete(self, messages):
            return _payload("苹果的英语是 apple。")

    offline = MinimalPreviewLLMClient()
    app = MinimalWebApp(tmp_path / "data", clients=MinimalModelClients(Client(), offline, offline, True))
    try:
        sid = app.start()["state"]["session_id"]
        before = app._sessions[_USER][1].alliance.trust
        result = app.submit(utterance, "local-turn", sid)
        assert result["state"]["turns"][-1]["response"] == "苹果的英语是 apple。"
        ctx = app._sessions[_USER][1]
        assert not ctx.alliance.ruptured
        assert ctx.alliance.trust == before
        # Full output survives both online and final guards; no model call was made.
        assert result["ok"]
    finally:
        app.close()


def _ctx():
    ctx = EngineContext(user_id="core-owner", scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v3"
    return ctx


def test_prompt_and_both_guards_share_exact_history_projection():
    ctx = _ctx()
    ctx.add_turn("儿子已经回复了。", "护士已经回复了。")
    ctx.add_turn("女儿已经回复了。", "收到。")
    build_messages(ElderCompanionScene(), None, ctx, "继续。", history_turns=1)
    expected = ["继续。", "女儿已经回复了。"]
    assert user_grounding_sources(ctx, "继续。") == expected
    assert _response_grounding_sources(ctx, "继续。", None) == expected
    assert _grounding_sources(ctx, "继续。", None) == expected
    scene = ElderCompanionScene()
    assert scene.guard_delivery_response(ctx, "女儿已经回复了。", "继续。") == "女儿已经回复了。"
    assert scene.guard_delivery_response(ctx, "儿子已经回复了。", "继续。") != "儿子已经回复了。"


@pytest.mark.parametrize("source,excluded", [
    ("这件事别提起，女儿已经回复了。", False),
    ("女儿已经回复了。", True),
    ("最近记不住事情，女儿已经回复了。", False),
])
def test_private_or_masked_history_cannot_ground_external_completion(source, excluded):
    ctx = _ctx()
    ctx.add_turn(source, "收到。")
    if excluded:
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    build_messages(ElderCompanionScene(), None, ctx, "继续。")
    assert _grounding_sources(ctx, "继续。", None) == ["继续。"]
    assert ElderCompanionScene().guard_delivery_response(ctx, "女儿已经回复了。", "继续。") != "女儿已经回复了。"


def test_zero_history_does_not_send_or_ground_old_turns_and_stale_session_is_rebuilt():
    ctx = _ctx()
    ctx.add_turn("女儿已经回复了。", "收到。")
    messages = build_messages(ElderCompanionScene(), None, ctx, "现在。", history_turns=0)
    assert len(messages) == 2
    assert user_grounding_sources(ctx, "现在。") == ["现在。"]
    ctx.session_id = "new-session"
    ctx.turns = []
    assert user_grounding_sources(ctx, "现在。") == ["现在。"]


def _delivery_ctx():
    ctx = _ctx()
    candidate = InterventionCandidate(id="quiet_companionship", label="陪伴", goal="回应", guidance="回应本人", selected_turn=0)
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.selected = candidate
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode="ask", question="旧问题？")
    ctx.working_memory["model_route"] = {"reply": {"integrated_decision": True, "source": "software_fixed_output"}}
    response = apply_single_pass_decision(_payload("我们先坐一会儿。"), ctx, "现在先坐一会儿。")
    return ctx, response


def test_v3_actual_strategy_delivery_does_not_require_old_probe_or_imply_outcome():
    ctx, response = _delivery_ctx()
    reasoner = ElderStateReasoner()
    execution = reasoner.record_observation_execution(ctx, response)
    assert not execution.matched_plan
    assert reasoner.record_intervention_delivery(ctx, response) is not None
    assert not ctx.reasoning.outcomes
    assert ctx.working_memory["online_turn_decision"]["parsed_response_sha256"] == hashlib.sha256(response.encode()).hexdigest()


@pytest.mark.parametrize("change", ["stale_turn", "unselected", "different_body", "replaced", "not_candidate", "invalid_status", "missing_trace", "missing_version"])
def test_v3_delivery_requires_current_choice_and_actual_unchanged_body(change):
    ctx, response = _delivery_ctx()
    if change == "stale_turn":
        ctx.working_memory["online_turn_decision"]["turn_index"] = -1
    elif change == "unselected":
        ctx.working_memory["online_turn_decision"]["requested_intervention_id"] = ""
    elif change == "different_body":
        response = "另一个正文。"
    elif change == "not_candidate":
        ctx.reasoning.candidates = []
    elif change == "invalid_status":
        ctx.working_memory["model_route"] = {}
        ctx.working_memory["online_turn_decision"]["status"] = "invalid"
    elif change == "missing_trace":
        ctx.working_memory["model_route"] = {}
        ctx.working_memory.pop("online_turn_decision")
    elif change == "missing_version":
        ctx.working_memory["model_route"] = {}
        ctx.working_memory["online_turn_decision"].pop("version")
    assert ElderStateReasoner().record_intervention_delivery(ctx, response, response_overridden=change == "replaced") is None


def _hypothesis(index, status="candidate", **extra):
    return PersonHypothesis(id=f"h{index}", framework="response", claim=f"有界假设{index}",
                            status=status, first_observed_at="2026-10-08T00:00:00", **extra)


def _reflection_ids(persona):
    content = build_reflection_messages(persona, "当前本人原话", [], True)[1]["content"]
    return [row["id"] for row in json.loads(content.split("\n", 1)[1].split("\n\n本次会话：", 1)[0])]


def test_active_hypothesis_stays_reviewable_when_recent_candidates_fill_window():
    persona = PersonaProfile(user_id="core-owner", person_hypotheses=[_hypothesis(0, "working"), *[_hypothesis(i) for i in range(1, 36)]])
    assert confirmed_prompt_hypotheses(persona)[0].id == "h0"
    assert "h0" in _reflection_ids(persona)
    assert len(_reflection_ids(persona)) == 30


def test_overflow_retains_storage_but_limits_all_current_active_influence():
    persona = PersonaProfile(user_id="core-owner", person_hypotheses=[_hypothesis(i, "working") for i in range(36)])
    expected = [f"h{i}" for i in range(6, 36)]
    assert [h.id for h in online_influencing_hypotheses(persona)] == expected
    assert _reflection_ids(persona) == expected
    assert len(persona.person_hypotheses) == 36


def test_retired_hypothesis_cannot_keep_old_committed_delivery_mode():
    hypothesis = PersonHypothesis(id="general-h", framework="disc", dimension="steadiness", claim="广义稳定偏好", status="stable",
                                  target_value=1., confidence=1., first_observed_at="2026-10-08T00:00:00", scope_tags=["general"])
    persona = PersonaProfile(user_id="core-owner", person_hypotheses=[hypothesis])
    old_mode = update_delivery_state(persona)
    assert old_mode == "low_pressure"
    hypothesis.status = "retired"
    assert select_delivery_mode(persona) == "neutral"
    assert persona.delivery_state.committed == old_mode  # retained record, not current authority
