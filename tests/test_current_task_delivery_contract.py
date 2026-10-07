"""Contract wiring regressions, not natural-language or live-model accuracy.

No independently sealed semantic case is read here. The finite legacy draft
diagnostic stays unchanged; it no longer gates the model's delivery guidance.
"""
from copy import deepcopy

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import single_pass_messages
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.reply_contract import contract_prompt, reply_quality_contract
from aria.interfaces.reasoning import ActiveObservationPlan
from aria.scenes.registry import get_scene


def _context(utterance, mode="observe"):
    ctx = EngineContext(user_id="contract-wiring", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = utterance
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode=mode)
    return ctx


@pytest.mark.parametrize("utterance", [
    "",  # Assembly must not depend on a phrase match.
    "这两种说法差在哪儿？",
    "今天不需要你准备内容。",
    "有人以前这样问过，今天只是在讲那件事。",
    "等以后需要时我再找你，现在先放着。",
])
@pytest.mark.parametrize("mode", ["observe", "wait"])
def test_delivery_guidance_reaches_both_reply_paths_without_draft_match(utterance, mode):
    ctx = _context(utterance, mode)
    contract = reply_quality_contract(ctx)
    assert contract["draft_requested"] is False
    messages = build_messages(get_scene("companion_elder"), None, ctx, utterance)
    original = deepcopy(messages)
    structured = single_pass_messages(messages, ctx)
    assert messages == original
    for prompt in (messages[0]["content"], structured[0]["content"]):
        assert "正文就交付实际答案或内容" in prompt
        assert "回答这个范围后停下" in prompt
        assert "不当作现在要做，最新纠正和停止优先" in prompt
        assert "不能假装已有可看的正文" in prompt
        assert "不自动重做历史任务" in prompt


@pytest.mark.parametrize("mode", ["observe", "wait"])
def test_current_task_guidance_keeps_boundaries_and_observation_permission(mode):
    contract = reply_quality_contract(_context("", mode))
    prompt = contract_prompt(contract)
    assert contract["version"] == "1.6.0"
    assert "安全与真实能力 > 本人当前事实/纠正/明确任务" in prompt
    assert "不把未授权的私人背景写给第三人" in prompt
    assert "草稿不等于发送、报名或同意" in prompt
    assert "通知须有相应工具，有真实回执才报成功" in prompt
    assert "本人明确请求的具体内容应保留必要完整性" in prompt
    assert "按步骤交流不等于每轮追加下一步" in prompt
    assert "准备内容、本人审阅、是否发送、允许谁转发是不同决定" in prompt
    assert contract["question_budget"] == (0 if mode == "wait" else 1)
    assert contract["notification_attempt_advertised"] is False


def test_finite_draft_detection_is_not_promoted_to_semantic_classifier():
    positive = reply_quality_contract(_context("帮我写一句话。"))
    negative = reply_quality_contract(_context("不用帮我写一句话了。"))
    assert positive["draft_requested"] is True
    assert negative["draft_requested"] is False
    # The same conditional semantic guidance is sent for either value. A
    # false finite match must not hide task guidance or invent a current task.
    assert contract_prompt(positive) == contract_prompt(negative)
