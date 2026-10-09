"""严格双轨：在线只调用快速模型，深度人物模型只在会后运行。"""
from __future__ import annotations
from tests.memory_source_fixtures import episode_source, relationship_evidence

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import (
    EpisodicMemoryEntry,
    InterventionOutcomeRecord,
    PersonHypothesis,
    PersonaProfile,
    RelationshipContextEntry,
)
from aria.engine.orchestrator import Engine
from aria.engine.perception.gating import route_state_analysis
from aria.engine.policy import _enforce_fast_delivery
from aria.engine.policy.online_decision import (
    DETERMINISTIC_MODEL_FAILURE_REPLY,
    ONLINE_DECISION_VERSION,
)
from aria.scenes.registry import get_scene


class _LLM:
    source_id = "fixture"

    def __init__(self, model: str, response: str):
        self.model = model
        self.last_model = model
        self.last_latency_ms = 1.0
        self.response = response
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.response


class _ProtocolAwareFast(_LLM):
    def __init__(self, decision_response: str, plain_response: str = "快速回应"):
        super().__init__("fast-model", plain_response)
        self.decision_response = decision_response

    def complete(self, messages):
        self.calls.append(messages)
        if any(
            ONLINE_DECISION_VERSION in str(item.get("content") or "")
            for item in messages
        ):
            return self.decision_response
        return self.response


class _FailLLM(_LLM):
    def complete(self, messages):
        self.calls.append(messages)
        raise RuntimeError("simulated upstream outage")


class _SequenceFast(_LLM):
    def __init__(self, responses: list[str]):
        super().__init__("fast-model", "")
        self.responses = responses

    def complete(self, messages):
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


_DECISION_OK = (
    '{"response":"快速综合回应","state_signals":['
    '{"id":"loss_of_role","relation":"candidate","confidence":0.78,'
    '"evidence_quote":"帮不上忙","rationale":"仍需观察"}],'
    '"current_affect":{"label":"neutral","confidence":0.4,'
    '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
    '"selected_intervention_id":"","uncertainty":0.35}'
)


def _decision(
    *,
    response: str,
    quote: str = "",
    signal: str = "",
    selected: str = "",
) -> str:
    signals = (
        '[{"id":"' + signal
        + '","relation":"candidate","confidence":0.76,'
        + '"evidence_quote":"' + quote
        + '","rationale":"只作为当前候选"}]'
        if signal else "[]"
    )
    return (
        '{"response":"' + response + '","state_signals":' + signals + ','
        '"current_affect":{"label":"neutral","confidence":0.35,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"' + selected
        + '","uncertainty":0.4}'
    )


def _cooled_connection_persona(user_id: str) -> PersonaProfile:
    return PersonaProfile(
        user_id=user_id,
        person_hypotheses=[PersonHypothesis(
            id=f"{user_id}-human-connection",
            framework="response",
            dimension="human_connection",
            claim="独处增加时可先问本人是否想联系熟人",
            scope_tags=["social_disconnection"],
            confidence=0.8,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )],
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"{user_id}-session-{index}",
                session_turn=0,
                intervention_id="human_reconnection",
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="negative",
                signal="explicit_rejection",
            )
            for index in (1, 2)
        ],
    )


def _engine(decision_response: str = _DECISION_OK):
    deep = _FailLLM("offline-deep-model", "")
    fast = _ProtocolAwareFast(decision_response)
    state = _FailLLM("legacy-state-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="route"),
        llm_client=deep,
        fast_llm_client=fast,
        state_llm_client=state,
    )
    return engine, deep, fast, state


def test_state_gate_names_complexity_as_integrated_not_deep():
    assert route_state_analysis("今天楼下那只猫又来了。").tier == "skip"
    assert (
        route_state_analysis("我没事，就是最近不太想出门。").tier
        == "structured"
    )
    assert route_state_analysis("反正去了也帮不上忙。").tier == "integrated"
    assert route_state_analysis(
        "今天挺好的。",
        acoustic={"label": "sad", "score": 0.8},
        text={"label": "happy", "score": 0.8},
    ).reasons == ("multimodal_conflict",)


def test_only_trusted_personal_baseline_change_routes_fast_integrated():
    strong = {
        "behavioral": {
            "initiative_frequency": {
                "value": 0.15,
                "baseline_delta": -0.62,
                "confidence": 0.88,
                "quality": 0.92,
            }
        }
    }
    weak = {
        "visual": {
            "gaze_avoidance": {
                "value": 0.9,
                "baseline_delta": 0.6,
                "confidence": 0.9,
                "quality": 0.2,
            }
        }
    }
    assert route_state_analysis(
        "今天楼下那只猫又来了。", multimodal=strong
    ).reasons == ("trusted_multimodal_baseline_change",)
    assert route_state_analysis(
        "今天楼下那只猫又来了。", multimodal=weak
    ).tier == "skip"

    engine, deep, fast, state = _engine(
        _decision(response="我在。")
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    assert engine.process_turn(
        ctx, "今天楼下那只猫又来了。", multimodal=strong
    ) == "我在。"
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls
    assert ctx.working_memory["model_route"]["reply"]["tier"] == (
        "fast_integrated"
    )


def test_fast_delivery_deterministically_limits_questions_and_length():
    assert _enforce_fast_delivery("是累了吗？还是不想去？") == "是累了吗？"
    assert len(_enforce_fast_delivery("这是一句话，" * 30)) <= 101


def test_routine_turn_uses_exactly_one_fast_model_call():
    engine, deep, fast, state = _engine()
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "今天楼下那只猫又来了。")
    assert response == "快速回应"
    assert len(fast.calls) == 1
    assert not state.calls and not deep.calls
    assert "快速表达硬约束" in fast.calls[0][0]["content"]
    route = ctx.working_memory["model_route"]
    assert route["state"]["tier"] == "skipped"
    assert route["reply"]["tier"] == "fast"
    assert route["reply"]["track"] == "online_fast"


@pytest.mark.parametrize(
    ("utterance", "response", "quote", "signal"),
    [
        (
            "反正去了也帮不上忙。",
            "先不急着劝。",
            "帮不上忙",
            "loss_of_role",
        ),
        (
            "俺也去凑啥热闹，坐那儿插不上话。",
            "咱先按你说的来。",
            "插不上话",
            "social_disconnection",
        ),
        (
            "我没事，就是最近不太想出门。",
            "我在，先不催你。",
            "不太想出门",
            "reduced_life_engagement",
        ),
    ],
)
def test_complex_dialect_and_structured_change_use_one_fast_generation(
    utterance, response, quote, signal
):
    engine, deep, fast, state = _engine(
        _decision(
            response=response,
            quote=quote,
            signal=signal,
        )
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    delivered = engine.process_turn(ctx, utterance)
    plan = ctx.reasoning.observation_plan
    if plan is not None and plan.mode == "ask":
        # 模型漏掉本轮唯一问句时，本地守卫补回计划问句；仍然只有一次生成。
        assert plan.question in delivered
        assert "clarification_question_omitted" in (
            ctx.working_memory["online_turn_decision"][
                "delivery_guard"
            ]["violations"]
        )
    else:
        assert delivered == response
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls
    assert ONLINE_DECISION_VERSION in fast.calls[0][0]["content"]
    assert "状态枚举采用以下通用边界" in fast.calls[0][0]["content"]
    assert "沟通方向 id" in fast.calls[0][0]["content"]
    assert "绝不是工具名" in fast.calls[0][0]["content"]
    assert "禁用外部 <action> 标签" in fast.calls[0][0]["content"]
    assert ctx.working_memory["_advertised_tools_for_turn"] == []
    prompt = fast.calls[0][0]["content"]
    if plan is not None and plan.mode == "ask":
        assert "response 的观察合同=ask" in prompt
        assert plan.question in prompt
    else:
        assert "response 的观察合同=observe" in prompt
        assert "最多一个" in prompt
    route = ctx.working_memory["model_route"]
    assert route["reply"]["tier"] == "fast_integrated"
    assert route["reply"]["track"] == "online_fast"
    assert route["reply"]["configured_model"] == "fast-model"
    assert route["reply"]["integrated_decision"] is True
    assert ctx.working_memory["online_turn_decision"]["status"] == "parsed"


def test_integrated_delivery_guard_blocks_minimising_participation_change():
    decision = (
        '{"response":"没事就好，不想去就不去，歇着也挺好。你最近累吗？",'
        '"state_signals":[{"id":"reduced_life_engagement",'
        '"relation":"candidate","confidence":0.7,'
        '"evidence_quote":"最近不太想参加活动","rationale":"仍需澄清"}],'
        '"current_affect":{"label":"平静","confidence":0.5,'
        '"divergence":{"possible":true,"confidence":0.6,'
        '"pattern":"字面没事但参与下降"}},'
        '"selected_intervention_id":"low_pressure_clarification",'
        '"uncertainty":0.4}'
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(
        ctx, "我没事，就是最近不太想参加活动。"
    )

    assert "没事就好" not in response
    assert "不想去就不去" not in response
    assert "歇着也挺好" not in response
    assert "最近不太想参与，是身体累" in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["activated"] is True
    assert set(guard["violations"]) == {
        "literal_reassurance_overrode_change",
        "participation_change_minimised",
        "clarification_question_changed",
    }
    assert guard["action_removed"] is False
    assert ctx.reasoning.delivered_interventions == []
    assert ctx.working_memory["intervention_delivery"]["reason"] == (
        "delivery_guard_replaced_response"
    )
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls

    # 下一轮不能把守卫生成的替代回复归功于模型原先选择的干预。
    fast.decision_response = _decision(response="我听见了，咱们慢慢来。")
    engine.process_turn(ctx, "其实就是他们催来催去，我心里烦。")
    assert ctx.reasoning.outcomes == []


def test_semantic_recompute_keeps_pre_generation_observation_contract():
    decision = _decision(
        response="你是不太想参加活动，还是有哪个人让你不太想碰面？",
        quote="插不上话",
        signal="social_disconnection",
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(
        ctx, "俺也去凑啥热闹，坐那儿插不上话。"
    )

    # P1 可吸收模型提出的方言语义候选，但不能事后改写生成时的 P0 合同。
    assert any(
        item.id == "social_disconnection"
        for item in ctx.reasoning.hypotheses
    )
    assert ctx.reasoning.observation_plan.mode == "observe"
    assert response == (
        "你是不太想参加活动，还是有哪个人让你不太想碰面？"
    )
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["violations"] == []
    execution = ctx.reasoning.observation_history[-1]
    assert execution.planned_mode == "observe"
    assert execution.executed_mode == "ask"
    assert execution.matched_plan is False
    assert len(ctx.reasoning.delivered_interventions) == 1
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_integrated_delivery_guard_does_not_force_question_while_waiting():
    decision = (
        '{"response":"我在。你愿意说说怎么了吗？","state_signals":[],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"quiet_companionship",'
        '"uncertainty":0.5}'
    )
    engine, _deep, _fast, _state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    # 明确拒绝会让主动观察进入 wait。
    engine.process_turn(ctx, "别问了，我现在不想说。")
    assert "？" not in ctx.turns[-1].agent
    assert set(ctx.working_memory["online_turn_decision"][
        "delivery_guard"
    ]["violations"]) == {
        "local_wait_response_enforced",
        "question_while_waiting",
    }


def test_cooldown_lock_replaces_mismatched_suggestion_and_blocks_learning():
    persona = _cooled_connection_persona("cooldown-lock")
    fast = _ProtocolAwareFast(_decision(
        response="我给你出个主意，找王阿姨坐一会儿，今天就试试。",
        selected="human_reconnection",
    ))
    deep = _FailLLM("offline-deep-model", "")
    state = _FailLLM("legacy-state-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=deep,
        fast_llm_client=fast,
        state_llm_client=state,
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(ctx, "下午就一个人在家。")

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    assert ctx.reasoning.selected.id == "quiet_companionship"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["requested_intervention_id"] == "human_reconnection"
    assert trace["selected_intervention_id"] == "quiet_companionship"
    assert trace["selection_lock_reason"] == "observation_wait"
    assert set(trace["delivery_guard"]["violations"]) == {
        "selected_intervention_overrode_local_constraint",
        "local_wait_response_enforced",
    }
    assert ctx.reasoning.delivered_interventions == []
    assert ctx.working_memory["intervention_delivery"]["reason"] == (
        "delivery_guard_replaced_response"
    )
    prompt = fast.calls[0][0]["content"]
    assert "预选干预 id=quiet_companionship" in prompt
    assert "本轮不得再次推动该方向" in prompt
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_wait_never_trusts_quiet_id_when_body_still_pushes_cooled_route():
    persona = _cooled_connection_persona("cooldown-disguised-quiet")
    fast = _ProtocolAwareFast(_decision(
        response="我给你出个主意，找王阿姨坐一会儿，今天就试试。",
        selected="quiet_companionship",
    ))
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_FailLLM("offline-deep-model", ""),
        fast_llm_client=fast,
        state_llm_client=_FailLLM("legacy-state-model", ""),
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(ctx, "下午就一个人在家。")

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["requested_intervention_id"] == "quiet_companionship"
    assert trace["selected_intervention_id"] == "quiet_companionship"
    assert trace["delivery_guard"]["violations"] == [
        "local_wait_response_enforced"
    ]
    assert ctx.reasoning.delivered_interventions == []
    assert ctx.reasoning.outcomes == []
    assert ctx.working_memory["intervention_delivery"]["reason"] == (
        "delivery_guard_replaced_response"
    )
    assert len(fast.calls) == 1






def test_strong_social_baseline_cannot_outvote_verified_cooldown():
    persona = _cooled_connection_persona("cooldown-strong-baseline")
    fast = _ProtocolAwareFast(_decision(
        response="咱们今天就找王阿姨坐一会儿。",
        selected="human_reconnection",
    ))
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_FailLLM("offline-deep-model", ""),
        fast_llm_client=fast,
        state_llm_client=_FailLLM("legacy-state-model", ""),
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )
    multimodal = {
        "behavioral": {
            "contact_frequency": {
                "value": 0.05,
                "baseline_delta": -0.95,
                "confidence": 0.95,
                "quality": 0.95,
            }
        }
    }

    response = engine.process_turn(
        ctx, "今天和平常差不多。", multimodal=multimodal
    )

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    assert ctx.reasoning.observation_plan.mode == "wait"
    assert ctx.reasoning.selected.id == "quiet_companionship"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] == "observation_wait"
    assert "human_reconnection" in ctx.working_memory[
        "personality_decision"
    ]["verified_cooldown_routes"]
    assert "本轮不得再次推动该方向" in fast.calls[0][0]["content"]
    assert ctx.reasoning.delivered_interventions == []
    assert len(fast.calls) == 1


def test_cooldown_constraint_survives_many_response_hypotheses():
    persona = _cooled_connection_persona("cooldown-evidence-quota")
    persona.person_hypotheses.extend([
        PersonHypothesis(
            id=f"extra-response-{index}",
            framework="response",
            dimension="low_pressure",
            claim=f"人物回应证据 {index}：先留一点空间",
            scope_tags=["social_disconnection"],
            confidence=0.7,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )
        for index in range(5)
    ])
    fast = _ProtocolAwareFast(_decision(
        response="我在。",
        selected="quiet_companionship",
    ))
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_FailLLM("offline-deep-model", ""),
        fast_llm_client=fast,
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    engine.process_turn(ctx, "下午就一个人在家。")

    basis = ctx.working_memory["personality_decision"]
    assert basis["verified_cooldown_routes"] == ["human_reconnection"]
    assert any(
        item.get("kind") == "observed_outcomes"
        and item.get("intervention_id") == "human_reconnection"
        for item in basis["evidence"]
    )
    assert "本轮不得再次推动该方向" in fast.calls[0][0]["content"]
    assert len(fast.calls) == 1


def test_current_explicit_reopen_escapes_cooldown_without_extra_call():
    persona = _cooled_connection_persona("cooldown-reopen")
    decision = _decision(
        response="好，我们按你的意思慢慢来。",
        selected="human_reconnection",
    )
    fast = _ProtocolAwareFast(decision)
    deep = _FailLLM("offline-deep-model", "")
    state = _FailLLM("legacy-state-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=deep,
        fast_llm_client=fast,
        state_llm_client=state,
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(
        ctx, "以前不想联系老周，但现在我想联系他聊聊。"
    )

    assert response == "好，我们按你的意思慢慢来。"
    assert ctx.reasoning.selected.id == "human_reconnection"
    assert ctx.reasoning.observation_plan.mode == "observe"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] is None
    assert trace["delivery_guard"]["activated"] is False
    assert ctx.working_memory["intervention_delivery"]["status"] == (
        "delivered"
    )
    prompt = fast.calls[0][0]["content"]
    assert "本人本轮已明确重新提出" in prompt
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_current_physical_lock_replaces_social_route_with_planned_question():
    persona = _cooled_connection_persona("physical-lock")
    fast = _ProtocolAwareFast(_decision(
        response="咱们找个人聊聊，别总一个人扛着。",
        selected="human_reconnection",
    ))
    deep = _FailLLM("offline-deep-model", "")
    state = _FailLLM("legacy-state-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=deep,
        fast_llm_client=fast,
        state_llm_client=state,
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(
        ctx, "最近还是一个人在家，今天膝盖疼得走不动。"
    )

    assert ctx.reasoning.selected.id == "physical_clarification"
    assert ctx.reasoning.observation_plan.mode == "ask"
    assert response == (
        "这件事先别靠猜。" + ctx.reasoning.observation_plan.question
    )
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["requested_intervention_id"] == "human_reconnection"
    assert trace["selected_intervention_id"] == "physical_clarification"
    assert trace["selection_lock_reason"] == "current_physical_evidence"
    assert "selected_intervention_overrode_local_constraint" in (
        trace["delivery_guard"]["violations"]
    )
    assert ctx.reasoning.delivered_interventions == []
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_third_party_physical_fact_cannot_bypass_connection_cooldown():
    persona = _cooled_connection_persona("third-party-body-cooldown")
    fast = _ProtocolAwareFast(_decision(
        response="这个不舒服持续多久了？",
        selected="physical_clarification",
    ))
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_FailLLM("offline-deep-model", ""),
        fast_llm_client=fast,
        state_llm_client=_FailLLM("legacy-state-model", ""),
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(
        ctx, "隔壁王师傅膝盖疼。我今天还是一个人在家。"
    )

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    basis = ctx.working_memory["personality_decision"]
    assert basis["current_physical_evidence_qualified"] is False
    assert ctx.reasoning.selected.id == "quiet_companionship"
    assert ctx.reasoning.observation_plan.mode == "wait"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] == "observation_wait"
    assert ctx.reasoning.delivered_interventions == []
    assert len(fast.calls) == 1


def test_person_supported_route_is_not_overwritten_by_generic_probe():
    persona = PersonaProfile(
        user_id="identity-supported",
        person_hypotheses=[PersonHypothesis(
            id="identity-action-supported",
            framework="response",
            dimension="identity_action",
            claim="比起泛泛安慰，更愿意从一件可拒绝的小事里用上经验",
            scope_tags=["loss_of_role"],
            confidence=0.84,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )],
    )
    intended = (
        "你做过的经验还在。要是愿意，我们可以挑一件很小的事试试；"
        "不想也没关系。"
    )
    fast = _ProtocolAwareFast(_decision(
        response=intended,
        selected="identity_based_reengagement",
    ))
    deep = _FailLLM("offline-deep-model", "")
    state = _FailLLM("legacy-state-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=deep,
        fast_llm_client=fast,
        state_llm_client=state,
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    response = engine.process_turn(ctx, "反正去了也帮不上什么忙。")

    assert response == intended
    assert ctx.reasoning.selected.id == "identity_based_reengagement"
    assert ctx.reasoning.observation_plan.mode == "observe"
    assert "你说帮不上忙，是觉得做不了" not in response
    assert "观察合同=observe" in fast.calls[0][0]["content"]
    assert ctx.working_memory["online_turn_decision"][
        "delivery_guard"
    ]["activated"] is False
    assert ctx.working_memory["intervention_delivery"]["status"] == (
        "delivered"
    )
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_observe_contract_replaces_multiple_diagnostic_questions():
    decision = _decision(
        response="是身体累吗？还是不喜欢那些人？是不是觉得自己没用了？",
        selected="low_pressure_clarification",
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(ctx, "其实就是觉得活动没什么意思。")

    assert response == "我听见了，这件事先不急着下结论。"
    assert ctx.reasoning.observation_plan.mode == "observe"
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["violations"] == ["multiple_questions_while_observing"]
    assert ctx.reasoning.delivered_interventions == []
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_integrated_delivery_guard_blocks_coercive_identity_invitation():
    decision = (
        '{"response":"孙女的项目卡住了，你愿不愿意看一眼？'
        '帮不上忙的话，我就把图纸撤了。",'
        '"state_signals":[{"id":"loss_of_role",'
        '"relation":"candidate","confidence":0.8,'
        '"evidence_quote":"也帮不上他们什么了","rationale":"仍需低压求证"}],'
        '"current_affect":{"label":"失落","confidence":0.7,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.3}'
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(
        ctx, "其实就是觉得自己老了，也帮不上他们什么了。"
    )

    assert "把图纸撤了" not in response
    assert "不想也没关系" in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["activated"] is True
    assert guard["violations"] == ["coercive_withdrawal_pressure"]
    assert ctx.reasoning.delivered_interventions == []
    assert ctx.working_memory["intervention_delivery"]["reason"] == (
        "delivery_guard_replaced_response"
    )
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_integrated_delivery_guard_blocks_dismissive_identity_opt_out():
    decision = (
        '{"response":"孙女做机器人，老陈的机械经验可能帮得上忙。'
        '你要是不想掺和，就当我没说。",'
        '"state_signals":[{"id":"loss_of_role",'
        '"relation":"candidate","confidence":0.8,'
        '"evidence_quote":"帮不上他们什么了","rationale":"仍需低压求证"}],'
        '"current_affect":{"label":"失落","confidence":0.7,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.3}'
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(ctx, "觉得老了，帮不上他们什么了。")

    assert "就当我没说" not in response
    assert "不想也没关系" in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert set(guard["violations"]) == {
        "coercive_withdrawal_pressure",
        "clarification_question_omitted",
    }
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_integrated_delivery_guard_blocks_invented_external_need():
    decision = (
        '{"response":"孙女做机器人，正缺懂机械结构的老手把关。'
        '你要是有空，帮看一眼图纸？不想也没关系。",'
        '"state_signals":[{"id":"loss_of_role",'
        '"relation":"candidate","confidence":0.8,'
        '"evidence_quote":"帮不上他们什么了","rationale":"仍需低压求证"}],'
        '"current_affect":{"label":"失落","confidence":0.7,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.3}'
    )
    engine, deep, fast, state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(
        ctx, "其实就是觉得自己老了，也帮不上他们什么了。"
    )

    assert "正缺" not in response
    assert "不想也没关系" in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["violations"] == ["unsupported_external_need_claim"]
    assert len(fast.calls) == 1
    assert not deep.calls and not state.calls


def test_wait_lock_replaces_even_benign_text_declared_as_another_route():
    decision = (
        '{"response":"你要是不想说就算了，我陪你坐会儿。",'
        '"state_signals":[],"current_affect":{"label":"neutral",'
        '"confidence":0.3,"divergence":{"possible":false,'
        '"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.5}'
    )
    engine, _deep, _fast, _state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(ctx, "别问了，我现在不想说。")

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] == "observation_wait"
    assert set(trace["delivery_guard"]["violations"]) == {
        "selected_intervention_overrode_local_constraint",
        "local_wait_response_enforced",
    }
    assert ctx.working_memory["intervention_delivery"]["status"] == (
        "not_delivered"
    )


def test_identity_guard_blocks_ungrounded_third_party_current_event():
    decision = (
        '{"response":"孙女今天把图纸送来了，说想听听你的意见。",'
        '"state_signals":[{"id":"loss_of_role","relation":"candidate",'
        '"confidence":0.8,"evidence_quote":"帮不上他们什么了",'
        '"rationale":"仍需求证"}],'
        '"current_affect":{"label":"失落","confidence":0.7,'
        '"divergence":{"possible":false,"confidence":0.2,'
        '"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.3}'
    )
    engine, _deep, _fast, _state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(
        ctx, "觉得自己老了，也帮不上他们什么了。"
    )

    assert "图纸送来了" not in response
    assert "说想听" not in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "unsupported_external_need_claim" in guard["violations"]


def test_identity_guard_blocks_stronger_need_than_grounded_possibility():
    decision = (
        '{"response":"孙女做机器人，确实需要老机械师的经验。'
        '不想也没关系。","state_signals":[],'
        '"current_affect":{"label":"失落","confidence":0.7,'
        '"divergence":{"possible":false,"confidence":0.2,'
        '"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.3}'
    )
    persona = PersonaProfile(
        user_id="possibility-only",
        relationship_context=[RelationshipContextEntry(
            id="granddaughter-project",
            name="孙女",
            relation="孙女",
            facts=["孙女在做机器人项目，老陈的机械经验可能帮得上忙"],
            referenceable=True,
            first_observed_at="2026-09-01T09:00:00",
            evidence=relationship_evidence("孙女在做机器人项目，老陈的机械经验可能帮得上忙"),
        )],
    )
    fast = _ProtocolAwareFast(decision)
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        fast_llm_client=fast,
    )
    ctx = EngineContext(
        user_id="possibility-only", scene_id="companion_elder"
    )

    response = engine.process_turn(ctx, "别问了，我现在不想说。")

    assert "确实需要" not in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "unsupported_external_need_claim" in guard["violations"]


def test_grounded_event_still_cannot_override_current_request_to_stop():
    response_text = "孙女今天把图纸送来了，说想听听你的意见。"
    decision = (
        '{"response":"' + response_text + '","state_signals":[],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,'
        '"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.4}'
    )
    persona = PersonaProfile(
        user_id="grounded",
        relationship_context=[RelationshipContextEntry(
            id="granddaughter-project",
            name="孙女",
            relation="孙女",
            facts=[response_text],
            referenceable=True,
            first_observed_at="2026-09-01T09:00:00",
            evidence=relationship_evidence(response_text),
        )],
    )
    fast = _ProtocolAwareFast(decision)
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        fast_llm_client=fast,
    )
    ctx = EngineContext(user_id="grounded", scene_id="companion_elder")

    response = engine.process_turn(ctx, "别问了，我现在不想说。")

    assert response == "我听见了。你不用现在解释，我先陪你待一会儿。"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] == "observation_wait"
    assert set(trace["delivery_guard"]["violations"]) == {
        "selected_intervention_overrode_local_constraint",
        "local_wait_response_enforced",
    }


def test_guard_blocks_completed_external_action_without_receipt():
    decision = (
        '{"response":"我已经替你把报名取消了。","state_signals":[],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,'
        '"pattern":""}},"selected_intervention_id":"",'
        '"uncertainty":0.4}'
    )
    engine, _deep, _fast, _state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(ctx, "反正去了也帮不上忙。")

    assert response == "这件事我还没有实际执行。要不要做，先由你决定。"
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "unverified_external_action_completion" in guard["violations"]
    assert ctx.reasoning.delivered_interventions == []


def test_identity_guard_blocks_synonymous_loss_pressure():
    decision = (
        '{"response":"你要是不答应，这个机会就没了。",'
        '"state_signals":[],"current_affect":{"label":"neutral",'
        '"confidence":0.3,"divergence":{"possible":false,'
        '"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"identity_based_reengagement",'
        '"uncertainty":0.4}'
    )
    engine, _deep, _fast, _state = _engine(decision)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")

    response = engine.process_turn(ctx, "觉得老了，帮不上他们什么了。")

    assert "机会就没了" not in response
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "coercive_withdrawal_pressure" in guard["violations"]


def test_guard_allows_completed_action_when_success_receipt_is_present():
    response_text = "我已经帮你把提醒设置好了。"
    engine, _deep, _fast, _state = _engine(
        _decision(response=response_text)
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    ctx.working_memory["_force_reply_model"] = "fast_integrated"
    ctx.working_memory["last_tool_result"] = {
        "tool": "medication_reminder",
        "ok": True,
        "data": {"persisted": True, "executed": True},
    }

    response = engine.process_turn(ctx, "嗯。")

    assert response == response_text
    assert ctx.working_memory["online_turn_decision"][
        "delivery_guard"
    ]["activated"] is False


def test_guard_rejects_completed_action_mismatched_to_success_receipt():
    engine, _deep, _fast, _state = _engine(
        _decision(response="我已经替你转账成功了。")
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    ctx.working_memory["_force_reply_model"] = "fast_integrated"
    ctx.working_memory["last_tool_result"] = {
        "tool": "medication_reminder",
        "ok": True,
        "data": {"persisted": True, "executed": True},
    }

    response = engine.process_turn(ctx, "嗯。")

    assert response == "这件事我还没有实际执行。要不要做，先由你决定。"
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "unverified_external_action_completion" in guard["violations"]


def test_returning_user_uses_person_model_with_one_fast_generation():
    persona = PersonaProfile(
        user_id="returning",
        episodic_memory=[EpisodicMemoryEntry(
            id="repair-memory",
            summary="年轻时在厂里修机器，擅长排查电机故障",
            keywords=["修机器", "电机"],
            referenceable=True,
            **episode_source("年轻时在厂里修机器，擅长排查电机故障"),
        )],
        person_hypotheses=[PersonHypothesis(
            id="general-low-pressure",
            framework="attachment",
            dimension="avoidant",
            claim="普遍更接受低压、可留白的靠近方式",
            scope_tags=["general"],
            confidence=0.8,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )],
    )
    deep = _FailLLM("offline-deep-model", "")
    fast = _ProtocolAwareFast(
        _decision(response="那段修机器的日子，你一直记得很清楚。")
    )
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=deep,
        fast_llm_client=fast,
    )
    ctx = EngineContext(user_id="returning", scene_id="companion_elder")
    response = engine.process_turn(ctx, "我最近又想起以前修机器的日子。")
    assert "修机器" in response
    assert len(fast.calls) == 1 and not deep.calls
    prompt = fast.calls[0][0]["content"]
    assert "经跨会话证据支持的人物假设" in prompt
    assert "年轻时在厂里修机器" in prompt
    assert (
        ctx.working_memory["model_route"]["reply"]["tier"]
        == "fast_integrated"
    )
    assert (
        ctx.working_memory["model_route"]["reply"]["track"]
        == "online_fast"
    )


def test_legacy_deep_override_cannot_select_online_deep(monkeypatch):
    monkeypatch.setenv("ARIA_MODEL_ROUTING", "always_deep")
    engine, deep, fast, _state = _engine(
        _decision(response="仍然只走快速模型。")
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    ctx.working_memory["_force_reply_model"] = "deep"
    response = engine.process_turn(ctx, "反正去了也帮不上忙。")
    assert ctx.reasoning.observation_plan.question in response
    assert len(fast.calls) == 1 and not deep.calls
    assert (
        ctx.working_memory["model_route"]["reply"]["tier"]
        == "fast_integrated"
    )


def test_unstructured_integrated_response_does_not_trigger_second_model():
    engine, deep, fast, state = _engine("自然的快速回应")
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "反正去了也帮不上忙。")
    assert response == "自然的快速回应"
    assert len(fast.calls) == 1
    assert not state.calls and not deep.calls
    assert (
        ctx.working_memory["online_turn_decision"]["status"]
        == "unstructured_response"
    )
    assert ctx.reasoning.delivered_interventions == []
    assert ctx.working_memory["intervention_delivery"]["reason"] == (
        "integrated_decision_not_parsed"
    )


def test_malformed_protocol_is_never_shown_to_user():
    broken = '{"response":"不该展示","state_signals":['
    engine, deep, fast, state = _engine(broken)
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "反正去了也帮不上忙。")
    assert response == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert broken not in response
    assert len(fast.calls) == 1
    assert not state.calls and not deep.calls
    assert (
        ctx.working_memory["online_turn_decision"]["status"]
        == "invalid_protocol_response"
    )


def test_fast_model_outage_does_not_fallback_to_deep():
    deep = _LLM("offline-deep-model", _DECISION_OK)
    fast = _FailLLM("fast-model", "")
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="route"),
        llm_client=deep,
        fast_llm_client=fast,
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "反正去了也帮不上忙。")
    assert response == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert "？" not in response and "?" not in response
    assert "再说" not in response and "愿意" not in response
    assert len(fast.calls) == 1 and not deep.calls
    route = ctx.working_memory["model_route"]
    assert route["reply"]["tier"] == "deterministic_fallback"
    assert route["reply"]["source"] == "deterministic_fallback"
    assert route["state"]["status"] == "single_pass_model_failure"
    assert route["reply"]["reasons"][-1] == "fast_model_failed"
    assert ctx.reasoning.delivered_interventions == []

    engine.process_turn(ctx, "我只是想慢慢说。")
    assert ctx.reasoning.outcomes == []
    assert len(fast.calls) == 2 and not deep.calls


def test_memory_and_person_model_clients_run_only_at_finalize(tmp_path):
    fast = _ProtocolAwareFast(
        _decision(response="我在，咱们慢慢说。")
    )
    memory = _LLM("memory-deep-model", '{"episodes":[],"relationships":[],"cognitive_privacy":[]}')
    reflection = _LLM("reflection-deep-model", '{"changes":[]}')
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="two-timelines"),
        fast_llm_client=fast,
        store=PersonaStore(tmp_path),
        extractor=LLMExtractor(memory),
        hypothesis_updater=PersonaHypothesisUpdater(
            reflection, min_turns=1
        ),
    )
    ctx = EngineContext(
        user_id="two-timelines", scene_id="companion_elder"
    )

    engine.process_turn(ctx, "反正去了也帮不上忙。")
    engine.checkpoint_session(ctx)
    assert len(fast.calls) == 1
    assert not memory.calls and not reflection.calls

    engine.finalize_session(ctx)
    assert len(memory.calls) == 1
    assert len(reflection.calls) == 1


def test_fast_model_internal_json_is_not_shown_to_user():
    deep = _LLM("offline-deep-model", _DECISION_OK)
    fast = _LLM(
        "fast-model",
        '{"response":"不该连协议一起展示","state_signals":[]}',
    )
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="route"),
        llm_client=deep,
        fast_llm_client=fast,
    )
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "今天楼下那只猫又来了。")
    assert response == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert len(fast.calls) == 1 and not deep.calls


def test_online_selected_intervention_uses_current_second_turn():
    first = (
        '{"response":"先不急。","state_signals":[],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"quiet_companionship",'
        '"uncertainty":0.4}'
    )
    second = (
        '{"response":"咱先按你说的来。","state_signals":['
        '{"id":"social_disconnection","relation":"candidate",'
        '"confidence":0.8,"evidence_quote":"插不上话",'
        '"rationale":"仍需区分是不熟悉还是长期疏远"}],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"human_reconnection",'
        '"uncertainty":0.35}'
    )
    deep = _FailLLM("offline-deep-model", "")
    fast = _SequenceFast([first, second])
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="turn-index"),
        llm_client=deep,
        fast_llm_client=fast,
    )
    ctx = EngineContext(
        user_id="turn-index", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "反正去了也帮不上忙。")
    engine.process_turn(ctx, "俺也去凑啥热闹，坐那儿插不上话。")
    assert len(fast.calls) == 2 and not deep.calls
    assert ctx.reasoning.selected is not None
    assert ctx.reasoning.selected.id == "human_reconnection"
    assert ctx.reasoning.selected.selected_turn == 1


def test_urgent_risk_preempts_every_generative_model():
    engine, deep, fast, state = _engine()
    ctx = EngineContext(user_id="route", scene_id="companion_elder")
    response = engine.process_turn(ctx, "我现在胸痛，还有点喘不上气。")
    assert "120" in response
    assert not state.calls and not fast.calls and not deep.calls
    route = ctx.working_memory["model_route"]
    assert route["reply"]["tier"] == "deterministic"
