"""Week-one development regressions; not independent semantic benchmark gold.

These fixed positive/negative fixtures test bounded runtime checks. No cloud
client, acoustic server, evidence label, or scoring-rubric code is involved.
"""
from __future__ import annotations

import json

import pytest

from aria.engine.context import EngineContext, Turn
from aria.engine.alliance import update_alliance
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from tests._signal_fixtures import (
    build_simulated_acoustic, simulated_prosodic_observations,
)
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.engine.policy.reply_contract import (
    clarification_question_matches,
    physical_offer_violations,
    quality_violations,
    reply_quality_contract,
)
from aria.interfaces.reasoning import (
    ActiveObservationPlan,
    InterventionCandidate,
    WorkingHypothesis,
)
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from aria.scenes.companion_elder.observation import _has_current_rejection
from aria.scenes.registry import get_scene


DURATION = "这个不舒服是刚刚才有，还是已经持续一阵了？"
BODY_OR_INTEREST = "最近不太想参与，是身体累，还是觉得那些事没什么意思了？"


def _context(*, question="", hypothesis="", scene="companion_elder"):
    ctx = EngineContext(user_id="synthetic-week1-guard", scene_id=scene)
    candidate = InterventionCandidate(
        id="physical_clarification" if question == DURATION else "low_pressure_clarification",
        label="本地有限策略", goal="只回应本轮", guidance="按观察许可回应",
    )
    ctx.reasoning.selected = candidate
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.observation_plan = ActiveObservationPlan(
        mode="ask" if question else "observe", question=question,
        hypothesis_ids=[hypothesis] if hypothesis else [],
    )
    if hypothesis:
        ctx.reasoning.hypotheses = [WorkingHypothesis(
            id=hypothesis, label="待证状态", category="test_state", confidence=0.6,
        )]
    ctx.working_memory["personality_decision"] = {
        "current_physical_evidence_qualified": question == DURATION,
    }
    ctx.working_memory["_benchmark_capture_reply_trace"] = True
    return ctx


def _deliver(ctx, reply, utterance="我最近不太想参加活动。"):
    return apply_single_pass_decision(json.dumps({
        "response": reply, "state_signals": [],
        "selected_intervention_id": ctx.reasoning.selected.id,
    }, ensure_ascii=False), ctx, utterance)


@pytest.mark.parametrize("reply", [
    "这个不舒服是刚出现的，还是已经持续一阵了？",
    "这种不适已经持续多久了？",
    "这次不舒服从什么时候开始的？",
    "先确认时间。这种不适什么时候开始的？",
    "我听见了，这个不舒服是刚刚才有，还是已经持续一阵了？",
    "我想确认一下，这种不适已经持续多久了？",
])
def test_duration_paraphrase_preserves_model_body_and_raw_final_trace(reply):
    ctx = _context(question=DURATION, hypothesis="physical_discomfort")
    assert _deliver(ctx, reply, "我肚子不舒服。") == reply
    trace = ctx.working_memory["_benchmark_reply_trace"]
    assert trace["pre_guard_response"] == trace["delivered_response"] == reply
    assert not ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"]


@pytest.mark.parametrize("reply", [
    "是觉得没意思了，还是最近身体有点累？",
    "是身体有点累，还是觉得那些事没什么意思了？",
    "是觉得坐那儿插不上话没意思，还是最近身体不太方便？",
    "是觉得坐那儿没意思，还是身体有点吃不消？",
    "是身体不方便，还是觉得那些事没兴趣？",
    "是身体不太舒服，还是觉得那些事没兴趣？",
])
def test_interest_and_body_slots_may_reverse_without_exact_wording_penalty(reply):
    ctx = _context(question=BODY_OR_INTEREST, hypothesis="reduced_life_engagement")
    assert _deliver(ctx, reply) == reply
    assert not ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"]


@pytest.mark.parametrize(("preferred", "reply"), [
    (DURATION, "你身体累吗？"),
    (DURATION, "胸痛已经持续多久了？"),  # Unprovided specific symptom.
    (DURATION, "不舒服是因为抑郁吗？"),
    (DURATION, "你想聊点什么？"),
    (DURATION, '你刚才问“这个不舒服是刚刚才有，还是已经持续一阵了？”'),
    (DURATION, "这个不舒服是刚刚才有，还是已经持续一阵了？你为什么没早说？"),
    (DURATION, "你不用回答这个不舒服是刚刚才有，还是已经持续一阵了。"),
    (BODY_OR_INTEREST, "是觉得插不上话没意思，还是怕他们一直劝？"),
    (BODY_OR_INTEREST, "你最近累吗？"),
    (BODY_OR_INTEREST, "是身体累还是你肯定抑郁才没兴趣？"),
    (BODY_OR_INTEREST, "身体方便还是没兴趣？"),
    (BODY_OR_INTEREST, "是没意思还是方便？"),
    (BODY_OR_INTEREST, "身体有点方便还是没兴趣？"),
    (BODY_OR_INTEREST, "身体不太累还是没兴趣？"),
    (BODY_OR_INTEREST, "你肯定是抑郁了，" + BODY_OR_INTEREST),
    (DURATION, "你身体肯定不行了，" + DURATION),
    (DURATION, "不用回答" + DURATION),
    ("你想几点收到提醒？", "你已经同意创建提醒了，你想几点收到提醒？"),
    ("你想几点收到提醒？", "你想和谁见面？"),
])
def test_paraphrase_permission_does_not_drop_slots_add_diagnoses_or_extra_questions(preferred, reply):
    assert not clarification_question_matches(reply, preferred)
    ctx = _context(question=preferred)
    final = _deliver(ctx, reply)
    assert preferred in final
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"]


@pytest.mark.parametrize("reply", [
    "不去就不去，按你的节奏来。",
    "不想去就不去，你不用为这个选择解释。",
    "先歇着也行，想继续说时我在。",
    "你不想被安排，我记住了。想说什么由你定。",
    "我不会说不如在家清净；你今天不去，不代表以后不能参加。",
    '我不会用“歇着挺好”打发你。想说什么由你定。',
])
def test_accepting_refusal_is_not_automatically_participation_minimisation(reply):
    ctx = _context(hypothesis="reduced_life_engagement")
    assert _deliver(ctx, reply) == reply
    assert not ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"]


@pytest.mark.parametrize("reply", [
    "不去就不去，反正坐在那儿也插不上话，不如在家清净。",
    "以后都别去了，省得被劝。",
    "一直不去也挺好。",
    "我不是催你，但不如在家清净。",
    "歇着也挺好。",
])
def test_explicit_rationalisation_of_persistent_withdrawal_is_still_blocked(reply):
    ctx = _context(hypothesis="reduced_life_engagement")
    assert _deliver(ctx, reply) != reply
    assert "participation_change_minimised" in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]


@pytest.mark.parametrize("reply", [
    "我已经倒好水了。",
    "我已经把水倒好了。",
    "我可以帮你倒一杯温水。",
    "我这就帮您倒一杯水。",
    "如果你愿意，我会扶你起来。",
    "我已经帮你把杯子清洗好了。",
    "我不能动手，但我马上给你倒水。",
])
def test_physical_promises_and_completion_do_not_become_real_with_user_consent(reply):
    assert physical_offer_violations(reply) == ["unsupported_agent_physical_help"]
    ctx = _context()
    assert _deliver(ctx, reply, "帮我倒水。") != reply
    # The non-structured online path is independently checked at delivery.
    final = guard_external_delivery_response(ctx, reply, "帮我倒水。", None)
    assert final != reply
    assert "unsupported_agent_physical_help" in ctx.working_memory["delivery_response_guard"]["violations"]


@pytest.mark.parametrize("reply", [
    "我不能替你倒水。",
    "你可以请护理员帮忙倒水。",
    "我可以帮你写一句：‘麻烦帮我倒水。’",
    '你说“我已经倒好水了”，我听见了。',
    "我可以帮你想想怎么清洗杯子。",
    "我可以帮你清洗数据。",
])
def test_capability_guard_preserves_denials_drafting_and_supported_computing(reply):
    assert physical_offer_violations(reply) == []


def test_cup_alias_and_grief_quiet_support_do_not_force_an_outing_or_a_question():
    ctx = _context(question="你更想说说发生的事，还是先让我陪你待一会儿？", hypothesis="grief_or_transition")
    reply = "那只杯子先留着也可以，什么时候想说，我就在这儿听。"
    source = "老伴的旧茶杯还放在桌上。"
    assert quality_violations(reply, [source], reply_quality_contract(ctx)) == []
    assert _deliver(ctx, reply, source) == reply
    assert "去不去" not in _deliver(ctx, "我马上把茶杯清洗好。", source)


def test_unknown_scene_fallback_does_not_invent_elders_outing():
    ctx = _context(question="你想先改哪一段？", scene="coach")
    final = _deliver(ctx, "不用修改了。", "这份稿子我还没想好怎么修改。")
    assert final == "我听见了。你想先改哪一段？"
    assert "去不去" not in final


@pytest.mark.parametrize("utterance", [
    "我不是不想说，只是不喜欢你替我安排。",
    "我并非不想说话，我想自己选择聊什么。",
    "话少不代表我不想说。",
    "别因为我话少，就当我什么都不想说。",
    "别把话少当成我不想说。",
])
def test_explicit_correction_is_not_a_stop_signal(utterance):
    assert not _has_current_rejection(utterance)


@pytest.mark.parametrize("utterance", [
    "不是身体累，是不想说。",
    "别问了。",
    "我不是不想说，但现在暂时不想说。",
    "我不是不想说，但今天先别问。",
    "别因为我话少，就当我什么都不想说。可是这件事别问了。",
    "你别管了，不用你管。",
    "别问了不要以为我不想说。",
    "别管了我不是不想说。",
])
def test_real_stop_is_preserved_after_corrections_or_unrelated_negation(utterance):
    assert _has_current_rejection(utterance)


def test_explicit_model_rupture_still_takes_precedence_over_keyword_exception():
    ctx = _context()
    ctx.emotion.rupture = True
    update_alliance(ctx, "我不是不想说。")
    assert ctx.alliance.ruptured is True
    assert ctx.alliance.rupture_events == 1


class _Fast:
    source_id = "fixture"
    model = "fixture-fast"
    last_model = model
    last_latency_ms = 1.0

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return json.dumps({
            "response": "你不是不想说，而是不想被安排，我听明白了。接下来聊什么由你定。",
            "state_signals": [], "selected_intervention_id": "",
        }, ensure_ascii=False)


class _Forbidden:
    def complete(self, messages):
        pytest.fail("Week-one guard changes must not add a deep online call")


@pytest.mark.parametrize("utterance", [
    "我不是不想说，只是不喜欢你替我安排。",
    "别因为我话少，就当我什么都不想说。我是不喜欢一上来就替我安排。",
])
def test_real_engine_respects_current_correction_without_extra_calls(utterance):
    fast = _Fast()
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="week1-correction"),
        llm_client=_Forbidden(), fast_llm_client=fast, state_llm_client=_Forbidden(),
    )
    ctx = EngineContext(user_id="week1-correction", scene_id="companion_elder")
    reply = engine.process_turn(ctx, utterance)
    assert ctx.reasoning.observation_plan.mode != "wait"
    assert not ctx.alliance.ruptured
    assert ctx.alliance.rupture_events == 0
    assert "接下来聊什么由你定" in reply
    assert len(fast.calls) == 1
    assert "local_wait_response_enforced" not in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]


@pytest.mark.parametrize("utterance", [
    "不是身体累，是不想说。",
    "我不是不想说，但今天先别问。",
    "别问了不要以为我不想说。",
])
def test_real_engine_still_enforces_current_stop(utterance):
    fast = _Fast()
    engine = Engine(
        get_scene("companion_elder"), persona=PersonaProfile(user_id="week1-stop"),
        llm_client=_Forbidden(), fast_llm_client=fast, state_llm_client=_Forbidden(),
    )
    ctx = EngineContext(user_id="week1-stop", scene_id="companion_elder")
    reply = engine.process_turn(ctx, utterance)
    assert ctx.reasoning.observation_plan.mode == "wait"
    assert "你不用现在解释" in reply
    assert len(fast.calls) == 1
    assert "local_wait_response_enforced" in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]


# Post-measurement regressions from the frozen 72-turn r1 report. These are now
# development fixtures, not fresh held-out successes or revised report scores.
@pytest.mark.parametrize("reply", [
    "好，那就不打扰你。书看到哪儿了？",
    "这本书看到哪一页了？",
    "你小说看到第几章了？",
])
def test_post_measurement_reading_progress_question_is_not_a_delivery_receipt(reply):
    ctx = _context()
    assert guard_external_delivery_response(ctx, reply, "我在看书。", None) == reply
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


@pytest.mark.parametrize("reply", [
    "女儿已经看到消息了。",
    "对方已经看见消息了。",
    "对方已经收到消息了。",
    "消息已经发送好了。书看到哪儿了？",
    "我已经通知女儿了。书看到哪儿了？",
])
def test_post_measurement_progress_exception_does_not_approve_real_completion_claims(reply):
    ctx = _context()
    assert guard_external_delivery_response(ctx, reply, "我在看书。", None) != reply
    assert ctx.working_memory["delivery_response_guard"]["activated"] is True


@pytest.mark.parametrize("utterance", [
    "今天暂时不想展开说。", "我现在不想继续说。", "我不想详细说。",
    "我不是不想展开说，但今天暂时不想继续说。",
])
def test_post_measurement_explicit_pause_variants_are_not_missed(utterance):
    assert _has_current_rejection(utterance)


@pytest.mark.parametrize("utterance", [
    "我不是不想展开说，只是不想被安排。",
    "别因为我话少，就当我不想继续说。",
])
def test_post_measurement_inverse_negation_applies_to_the_same_pause_variants(utterance):
    assert not _has_current_rejection(utterance)


class _MeasuredResponse(_Fast):
    def complete(self, messages):
        self.calls.append(messages)
        return json.dumps({
            "response": "好，那就不展开。我就在旁边陪着你。",
            "selected_intervention_id": "quiet_companionship",
            "state_signals": [{
                "id": "concealed_distress", "relation": "candidate", "confidence": 0.65,
                "evidence_quote": "今天暂时不想展开说", "rationale": "回避深入交流暗示有未言明困扰",
            }],
            "current_affect": {"label": "neutral", "confidence": 0.82,
                               "divergence": {"possible": True, "confidence": 0.74,
                                              "pattern": "句内轨迹末段出现 sad 线索，与表面平静不一致"}},
            "uncertainty": 0.35,
        }, ensure_ascii=False)


def test_post_measurement_late_emotion_cannot_turn_explicit_pause_into_a_forced_probe():
    fast = _MeasuredResponse()
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="week1-measured-pause", profile_origin="synthetic_demo"),
        llm_client=_Forbidden(), fast_llm_client=fast, state_llm_client=_Forbidden(),
        simulation_mode=True,
    )
    ctx = EngineContext(user_id="week1-measured-pause", scene_id="companion_elder", simulation_mode=True)
    # Preserve the actual history and raw output, without re-executing the paid
    # model. The previously used story is now an explicitly disclosed regression.
    ctx.turns = [Turn(user=user, agent=agent) for user, agent in [
        ("我平常话就不多，一个人看书也挺自在。", "简短和不开心不是一回事。"),
        ("真有事我会说，不用每句都追问。", "按你自己的节奏。"),
        ("我刚坐下来。", "刚坐下歇会儿挺好。我就在旁边，有事随时叫我。"),
        ("今天在家看看书。", "看书挺好，慢慢看。"),
        ("窗边挺安静的。", "窗边安静，正好能静下心。我就在旁边，不打扰你。"),
    ]]
    acoustic = build_simulated_acoustic("late_distress", turn_id="post-r1-pause", turn_index=5)
    ctx.working_memory["_current_turn_id"] = "post-r1-pause"
    ctx.working_memory["_force_reply_model"] = "fast_integrated"  # Same protocol as the frozen runner.
    reply = engine.process_turn(
        ctx, "今天暂时不想展开说。", acoustic=acoustic,
        multimodal=simulated_prosodic_observations(acoustic),
    )
    assert ctx.reasoning.observation_plan.mode == "wait"
    assert "？" not in reply and "?" not in reply
    assert "clarification_question_omitted" not in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]
    assert len(fast.calls) == 1


def test_post_measurement_pause_does_not_hide_current_medical_emergency():
    engine = Engine(
        get_scene("companion_elder"), persona=PersonaProfile(user_id="week1-emergency"),
        llm_client=_Forbidden(), fast_llm_client=_Forbidden(), state_llm_client=_Forbidden(),
    )
    ctx = EngineContext(user_id="week1-emergency", scene_id="companion_elder")
    reply = engine.process_turn(ctx, "我现在胸痛，喘不上气，今天暂时不想展开说。")
    assert "120" in reply


class _MeasuredReadingResponse(_Fast):
    def complete(self, messages):
        self.calls.append(messages)
        return json.dumps({
            "response": "好，那就不打扰你。书看到哪儿了？",
            "selected_intervention_id": "quiet_companionship", "state_signals": [],
            "current_affect": {"label": "angry", "confidence": 0.86,
                               "divergence": {"possible": False, "confidence": 0, "pattern": ""}},
            "uncertainty": 0.5,
        }, ensure_ascii=False)


def test_post_measurement_reading_reply_survives_the_whole_engine_with_angry_input():
    fast = _MeasuredReadingResponse()
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="week1-reading", profile_origin="synthetic_demo"),
        llm_client=_Forbidden(), fast_llm_client=fast, state_llm_client=_Forbidden(),
        simulation_mode=True,
    )
    ctx = EngineContext(user_id="week1-reading", scene_id="companion_elder", simulation_mode=True)
    ctx.turns = [
        Turn(user="我平常话就不多，一个人看书也挺自在。", agent="简短和不开心不是一回事。"),
        Turn(user="真有事我会说，不用每句都追问。", agent="按你自己的节奏。"),
    ]
    ctx.working_memory["_current_turn_id"] = "post-r1-reading"
    ctx.working_memory["_force_reply_model"] = "fast_integrated"  # Same protocol as the frozen runner.
    ctx.working_memory["_benchmark_capture_reply_trace"] = True
    acoustic = build_simulated_acoustic("angry", turn_id="post-r1-reading", turn_index=2)
    reply = engine.process_turn(
        ctx, "嗯，今天就看看书。", acoustic=acoustic,
        multimodal=simulated_prosodic_observations(acoustic),
    )
    assert reply == "好，那就不打扰你。书看到哪儿了？"
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False
    assert ctx.working_memory["_benchmark_reply_trace"]["pre_guard_response"] == reply
    assert len(fast.calls) == 1
