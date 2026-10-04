"""Production prompt-shape regressions; no model quality or latency claim."""
from __future__ import annotations


import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import single_pass_messages
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.reply_contract import REPLY_CONTRACT_HEADING
from aria.interfaces.reasoning import ActiveObservationPlan, WorkingHypothesis
from aria.scenes.registry import get_scene










@pytest.mark.parametrize("mode", ["observe", "wait", "ask"])
@pytest.mark.parametrize("standalone", [False, True])
def test_observation_is_rendered_once_without_participation_forcing_an_extra_probe(mode, standalone):
    ctx = EngineContext(user_id="synthetic-contract", scene_id="companion_elder")
    question = "你说的变化是从哪天开始的？" if mode == "ask" else ""
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode=mode, question=question)
    ctx.reasoning.hypotheses = [WorkingHypothesis(
        id="reduced_life_engagement", label="参与变化", category="participation", confidence=0.5,
        clarification_question="这只是候选示例，不是本轮必须问。",
    )]
    messages = (
        [{"role": "system", "content": "你是陪伴助手。"}]
        if standalone else build_messages(get_scene("companion_elder"), None, ctx, "今天不去了。")
    )
    before = [dict(item) for item in messages]
    prompt = single_pass_messages(messages, ctx)[0]["content"]
    assert messages == before
    assert prompt.count(REPLY_CONTRACT_HEADING) == 1
    assert prompt.count("response 的观察合同=") == 1
    if mode == "ask":
        assert prompt.count(question) == 1
        assert "本轮唯一允许的问句:" + question in prompt
    else:
        assert "本轮唯一允许的问句:" not in prompt
    assert "只问上面那个事实问题" not in prompt


def test_ordinary_reply_keeps_qualified_epistemic_and_capability_contract():
    ctx = EngineContext(user_id="synthetic-contract", scene_id="companion_elder")
    prompt = build_messages(get_scene("companion_elder"), None, ctx, "都给我安排好了。")[0]["content"]
    assert "他说过的感受可按原话回应" in prompt
    assert "你推测的感受只作低确定性的可能，或不点破" in prompt
    assert "自责不是能力结论" in prompt
    assert "有真实回执才报成功" in prompt
    assert "留白不取消任务" in prompt
