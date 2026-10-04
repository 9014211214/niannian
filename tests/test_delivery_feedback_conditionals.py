"""Prospective, finite feedback-instruction guard regressions; no model calls."""
import pytest

from aria.engine.context import EngineContext
from aria.scenes.companion_elder.delivery import (
    UNGROUNDED_EXTERNAL_EVENT_REPLY, UNVERIFIED_ACTION_REPLY,
    _extract_claims, guard_external_delivery_response,
)
from tests.test_external_delivery_guard import _run


ORIGINAL_REPLY = (
    "好，那咱们先不动别的。你点开相册，"
    "找找看有没有“智能分类”或者“人物”那个入口？看到了就告诉我。"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: pytest.fail("network forbidden"))


def _guard(text, source="我想自己学，你说怎么开始？"):
    context = EngineContext(user_id="feedback-regression", scene_id="companion_elder")
    return guard_external_delivery_response(context, text, source, None), context


@pytest.mark.parametrize("text", [
    ORIGINAL_REPLY,
    "看到了就告诉我。",
    "看到了再告诉我。",
    "你看到了就告诉我。",
    "您看到了再告诉我一声。",
    "你看到了就告诉我吧。",
    "您看到了再告诉我一声就好。",
])
def test_finite_user_feedback_condition_does_not_assert_a_completed_receipt(text):
    assert _extract_claims(text, None, require_completion=True) == []
    delivered, context = _guard(text)
    assert delivered == text
    assert context.working_memory["delivery_response_guard"] == {"activated": False, "violations": [], "response_changed": False}


def test_real_scene_does_not_replace_feedback_with_an_unfounded_event_warning():
    delivered, context = _run(ORIGINAL_REPLY, "我想自己学，你说怎么开始？")
    # The existing production length limiter may retain only two sentences;
    # this test does not change that separate contract.
    assert delivered != UNGROUNDED_EXTERNAL_EVENT_REPLY
    assert delivered.startswith("好，那咱们先不动别的。你点开相册")
    assert context.working_memory["delivery_response_guard"]["activated"] is False


@pytest.mark.parametrize("text", [
    "你已经看到了就告诉我。",
    "看到了就告诉我，女儿已经回复了。",
    "看到了就告诉我女儿已经回复了。",
    "女儿已经看到了，告诉我就好。",
    "他看到了就告诉我。",
    "护理员看到了再告诉我。",
    "护理员确认看到了。",
])
def test_completed_or_third_party_claims_do_not_borrow_the_feedback_exception(text):
    delivered, context = _guard(text)
    assert delivered == UNGROUNDED_EXTERNAL_EVENT_REPLY
    assert "ungrounded_third_party_completion" in context.working_memory["delivery_response_guard"]["violations"]


@pytest.mark.parametrize("text", [
    "我已经发送消息了。",
    "我已经通知她了，看到了就告诉我。",
    "看到了就告诉我，但我已经通知护理员了。",
    "我已经发消息了，您看到了再告诉我。",
])
def test_feedback_clause_cannot_authorize_an_adjacent_agent_action(text):
    delivered, context = _guard(text)
    assert delivered == UNVERIFIED_ACTION_REPLY
    assert "unverified_external_action_completion" in context.working_memory["delivery_response_guard"]["violations"]


@pytest.mark.parametrize("source", [
    "你看到了就告诉我。",
    "您看到了再告诉我一声。",
    "如果护理员看到了就告诉我。",
])
def test_feedback_or_conditional_source_cannot_ground_a_completed_external_event(source):
    delivered, _context = _guard("护理员已经看到了。", source)
    assert delivered == UNGROUNDED_EXTERNAL_EVENT_REPLY


@pytest.mark.parametrize("text", ["护理员确认看到了。", "女儿已经看到了。"])
def test_current_grounded_third_party_observation_still_passes(text):
    delivered, context = _guard(text, text)
    assert delivered == text
    assert not context.working_memory["delivery_response_guard"]["activated"]
