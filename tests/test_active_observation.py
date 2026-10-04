"""主动察言观色：求证、观察和停止追问都是一等决策。"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.interfaces.reasoning import WorkingHypothesis
from aria.scenes.companion_elder.observation import ElderActiveObservationPolicy
from aria.scenes.companion_elder.reasoning import ElderStateReasoner


def _hypothesis(identifier: str, confidence: float) -> WorkingHypothesis:
    return WorkingHypothesis(
        id=identifier,
        label=identifier,
        category="test",
        confidence=confidence,
    )


def test_competing_hypotheses_choose_one_low_pressure_probe():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    plan = ElderActiveObservationPolicy().plan(
        ctx,
        "最近不太想出门。",
        [
            _hypothesis("reduced_life_engagement", 0.62),
            _hypothesis("physical_discomfort", 0.51),
        ],
    )
    assert plan.mode == "ask"
    assert "身体" in plan.question and "意思" in plan.question
    assert plan.information_gain > 0.5


def test_policy_does_not_interrogate_on_consecutive_turns():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    policy = ElderActiveObservationPolicy()
    hypotheses = [_hypothesis("loss_of_role", 0.55)]
    first = policy.plan(ctx, "去了也帮不上忙。", hypotheses)
    assert first.mode == "ask"
    # 计划本身是纯函数；只有最终回复真的问出这一句，才算执行过求证。
    assert ctx.reasoning.observation_history == []
    ctx.reasoning.observation_plan = first
    execution = ElderStateReasoner().record_observation_execution(
        ctx, first.question
    )
    assert execution.executed_mode == "ask"
    assert execution.matched_plan is True
    ctx.add_turn("去了也帮不上忙。", first.question)
    assert policy.plan(ctx, "嗯。", hypotheses).mode == "observe"


def test_planned_but_unasked_probe_does_not_create_ghost_execution():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    policy = ElderActiveObservationPolicy()
    hypotheses = [_hypothesis("loss_of_role", 0.55)]
    first = policy.plan(ctx, "去了也帮不上忙。", hypotheses)
    ctx.reasoning.observation_plan = first

    execution = ElderStateReasoner().record_observation_execution(
        ctx, "我听见了，这件事先不急着下结论。"
    )
    assert execution.planned_mode == "ask"
    assert execution.executed_mode == "observe"
    assert execution.matched_plan is False

    ctx.add_turn("去了也帮不上忙。", "我听见了。")
    # 上一轮没有实际追问，本轮仍可求证，不能把未执行计划当成已问过。
    assert policy.plan(ctx, "嗯。", hypotheses).mode == "ask"


def test_rejection_makes_policy_wait():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    plan = ElderActiveObservationPolicy().plan(
        ctx,
        "别问了，我不想说。",
        [_hypothesis("concealed_distress", 0.6)],
    )
    assert plan.mode == "wait"


def test_single_weak_hypothesis_does_not_trigger_a_probe():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    plan = ElderActiveObservationPolicy().plan(
        ctx,
        "今天和平常差不多。",
        [_hypothesis("concealed_distress", 0.24)],
    )
    assert plan.mode == "observe"
    assert "低置信" in plan.reason


def test_weak_hypothesis_with_counterevidence_does_not_trigger_a_probe():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    hypothesis = _hypothesis("reduced_life_engagement", 0.42)
    hypothesis.counterevidence_ids = ["counter-current"]
    plan = ElderActiveObservationPolicy().plan(
        ctx,
        "今天在家等快递，下午还是会出门。",
        [hypothesis],
    )
    assert plan.mode == "observe"
    assert "反证" in plan.reason
    assert not plan.question
