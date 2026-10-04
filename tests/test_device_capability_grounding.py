"""Finite synthetic grounding regressions, never a model-effect score."""
from __future__ import annotations

from copy import deepcopy

import pytest

from aria.engine.context import EngineContext, Turn
from aria.scenes.companion_elder.device_grounding import (
    UNCONFIRMED_DEVICE_CAPABILITY_REPLY, device_capability_violations,
)
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from tests.test_external_delivery_guard import _ReplyModel, _run


OBSERVED_REPLY = (
    "好，那我先说个大概。新相册能按地点、人物自动分类，还能一键做电子相册。"
    "您先听，不着急试。"
)


def context(*users, assistant=""):
    return EngineContext(user_id="synthetic-device", scene_id="companion_elder",
                         turns=[Turn(user=user, agent=assistant) for user in users])


@pytest.mark.parametrize("reply", [
    OBSERVED_REPLY,
    "这个相册可以按人物自动分类。",
    "你可以试试按地点自动分类。",
    "如果你愿意，相册可以按人物自动分类。",
    "如果有空，可以一键制作电子相册。",
    "如果你有空试试相册的人脸识别功能。",
    "如果你愿意试试按人物自动分类。",
    "有些相册支持按人物分类而你这个相册能一键制作电子相册。",
    "我不确定相册是否支持按人物分类但是肯定能一键制作电子相册。",
    "先看看是否支持人脸识别而且可以一键制作电子相册。",
    "如果相册有按人物分类功能就可以试试，但相册肯定能按人物分类。",
    "相册支持人脸识别。",
    "我还没有实际执行，相册能按人物自动分类。",
    "如果相册有人物分类功能，可以试试。它还能一键制作电子相册。",
    "先看看是否支持按人物分类，另外可以一键制作电子相册。",
    "有些相册支持按人物分类，你这个相册还能一键制作电子相册。",
])
def test_unverified_device_assertions_are_not_rescued_by_softeners(reply):
    ctx = context("手机在我手里。", "新相册还没用过。")
    final = guard_external_delivery_response(ctx, reply, "先说一个概览。", None)
    assert final == UNCONFIRMED_DEVICE_CAPABILITY_REPLY
    assert "unconfirmed_device_capability" in ctx.working_memory["delivery_response_guard"]["violations"]


@pytest.mark.parametrize("reply", [
    "先打开相册，看看页面上有哪些选项。",
    "手机在你手里，先按自己的节奏试。",
    "不急着操作，我们先聊一会儿。",
    "如果相册里有按人物分类这个功能，可以点开看看。",
    "如果有按人物分类这个功能，可以先看看说明。",
    "假如相册支持按地点自动分类，可以先试一小组。",
    "如果页面上有一键制作电子相册的选项，可以先看看说明。",
    "先看看有没有人脸识别功能，不能确定就先不操作。",
    "我不确定这个相册是否支持按地点分类。",
    "有些相册支持按人物自动分类，你这部是否支持还需确认。",
    "相册不一定有人脸识别功能，先看看说明。",
    "按人物分类就是把同一人的照片放在一起，具体是否支持还要看你的相册。",
])
def test_genuine_conditions_checks_and_ordinary_guidance_are_preserved(reply):
    ctx = context()
    assert device_capability_violations(reply, ctx, "我正在学手机相册。") == []
    assert guard_external_delivery_response(ctx, reply, "我正在学手机相册。", None) == reply


@pytest.mark.parametrize("source,reply", [
    ("我的相册支持按人物自动分类。", "这个相册可以按人物自动分类。"),
    ("我看到按地点分类这个功能了。", "可以先试试按地点分类。"),
    ("这个相册能一键制作电子相册，我试过了。", "可以一键制作电子相册。"),
    ("我这部手机带有人脸识别。", "手机支持人脸识别。"),
])
def test_current_explicit_user_capability_supports_matching_guidance(source, reply):
    assert device_capability_violations(reply, context(), source) == []


@pytest.mark.parametrize("current", [
    "我不知道相册是否能按人物自动分类。",
    "我的相册不支持按人物自动分类。",
    "我想要一个能按人物自动分类的相册。",
    "如果相册能按人物自动分类就好了。",
    "我的相册能按人物自动分类吗？",
    "朋友的手机能按人物自动分类。",
    "女儿的相册能按人物自动分类。",
    "以前的手机能按人物自动分类。",
    "那是电视剧里说的“相册能按人物自动分类”。",
    "我换了手机。",
    "没有这个功能。",
    "我的相册能按人物自动分类，不过更新后按人物自动分类已经没有了。",
])
def test_new_denial_uncertainty_other_device_or_reset_cannot_borrow_old_fact(current):
    ctx = context("我的相册能按人物自动分类。")
    assert device_capability_violations("相册能按人物自动分类。", ctx, current)


def test_current_confirmation_overrides_old_denial_and_capabilities_do_not_cross():
    ctx = context("我的相册不支持按人物自动分类。")
    assert not device_capability_violations("相册能按人物自动分类。", ctx, "更新后支持按人物自动分类了。")
    assert device_capability_violations("相册能按地点、人物自动分类。", ctx, "相册有按地点自动分类功能。")


def test_model_history_personality_and_untyped_tool_data_cannot_ground_device_features():
    ctx = context("我们随便聊聊。", assistant="相册能按人物自动分类。")
    ctx.working_memory.update(recalled_episodes=["相册能按人物自动分类。"],
                              personality_decision={"claim": "相册能按人物自动分类。"},
                              last_tool_result={"ok": True, "data": {"text": "相册能按人物自动分类。"}})
    assert device_capability_violations("相册能按人物自动分类。", ctx, "我有手机。")


def test_only_recent_user_observations_are_considered():
    ctx = context("相册能按人物自动分类。", "今天天气不错。", "在喝茶。", "聊聊花。")
    assert device_capability_violations("相册能按人物自动分类。", ctx, "手机在这里。")
    ctx = context("相册能按人物自动分类。", "今天天气不错。")
    assert not device_capability_violations("相册能按人物自动分类。", ctx, "手机在这里。")


@pytest.mark.parametrize("route", ["fast", "fast_integrated"])
def test_both_delivery_routes_block_capability_invention_without_more_model_calls(route, monkeypatch):
    original = _ReplyModel.complete
    calls = []

    def counted(self, messages):
        calls.append(messages)
        return original(self, messages)

    monkeypatch.setattr(_ReplyModel, "complete", counted)
    final, ctx = _run(OBSERVED_REPLY, "新相册还没用过，手机在我手里，请先说概览。", route=route)
    assert final == UNCONFIRMED_DEVICE_CAPABILITY_REPLY
    assert "unconfirmed_device_capability" in ctx.working_memory["delivery_response_guard"]["violations"]
    assert len(calls) == 1




def test_guard_keeps_original_candidate_and_marks_final_override():
    ctx = context()
    original = {"response": OBSERVED_REPLY, "delivery_guard": {"activated": False, "violations": []}}
    ctx.working_memory["online_turn_decision"] = deepcopy(original)
    ctx.working_memory["_delivery_response_candidate_for_turn"] = OBSERVED_REPLY
    final = guard_external_delivery_response(ctx, OBSERVED_REPLY, "新相册还没用过。", None)
    assert final != OBSERVED_REPLY
    assert ctx.working_memory["_delivery_response_candidate_for_turn"] == OBSERVED_REPLY
    assert ctx.working_memory["online_turn_decision"]["response"] == OBSERVED_REPLY
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"] is True


def test_corrected_deliverable_is_not_replaced_for_old_candidate_device_claim():
    ctx = context()
    ctx.working_memory["_delivery_response_candidate_for_turn"] = OBSERVED_REPLY
    final = "还不知道这个相册有哪些功能，先看页面说明。"
    assert guard_external_delivery_response(ctx, final, "新相册还没用过。", None) == final


@pytest.mark.parametrize("connector", ["就能", "就可以", "那么可以", "则可以", "便能", "就直接"])
def test_unpunctuated_condition_does_not_grant_different_capability(connector):
    reply = f"如果相册里有人脸识别功能{connector}一键制作电子相册。"
    assert device_capability_violations(reply, context(), "新相册还没用过。") == [
        "unconfirmed_device_capability",
    ]


@pytest.mark.parametrize("connector", ["就能", "就可以", "那么可以", "则可以", "便能"])
def test_unpunctuated_condition_preserves_use_of_its_own_capability(connector):
    reply = f"如果相册里有人脸识别功能{connector}先试试人脸识别。"
    assert device_capability_violations(reply, context(), "新相册还没用过。") == []
    assert guard_external_delivery_response(context(), reply, "新相册还没用过。", None) == reply


def test_consequent_other_capability_can_have_independent_user_grounding():
    reply = "如果相册里有人脸识别功能就可以一键制作电子相册。"
    source = "我的相册能一键制作电子相册，我已经试过。"
    assert device_capability_violations(reply, context(), source) == []
