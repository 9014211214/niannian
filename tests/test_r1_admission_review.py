"""R1 software checks with explicit model labels, not semantic-effect measurements."""
from copy import deepcopy

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows,
    merge_preferences, transition_preference,
)
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene
from tests.preference_fixtures import communication_review


def propose(text, instruction, *, scope="任何时候", quote=None, confirmation=None, semantic_review=None):
    ctx = EngineContext(user_id="synthetic-owner", scene_id="companion_elder", session_id="source")
    ctx.add_turn(text, instruction + "，这样记下好吗？")
    ctx.add_turn("对，就这么记。", "本地软件替身。")
    profile = PersonaProfile(user_id=ctx.user_id)
    rows, rejected = extract_preference_rows([{
        "do": instruction, "scope": scope, "status": "confirmed",
        "semantic_review": communication_review() if semantic_review is None else semantic_review,
        "quotes": [{"turn_index": 0, "text": quote or text}],
        "confirmation": confirmation,
    }], ctx, profile)
    return ctx, profile, rows, rejected


@pytest.mark.parametrize("source,instruction", [
    ("下回选帽子，我中意哪顶等我自己掂量。", "选帽子留给本人自行掂量的时间"),
    ("逛展览的路线，两条都摆出来，我心里才有底。", "逛展览路线同时呈现两个选项"),
    ("讨论棋谱时，我有点卡壳也让我慢慢来。", "讨论棋谱时给本人慢慢思考的空间"),
    ("这会儿咱们谈围巾，暖不暖和我自己拿捏。", "谈围巾时由本人拿捏保暖程度"),
    ("我答不上来时别急着往下接。", "允许停顿留出考虑时间"),
])
def test_paraphrase_needs_no_request_or_communication_verb_whitelist(source, instruction):
    assert instruction not in source
    _, _, rows, rejected = propose(source, instruction)
    assert not rejected and len(rows) == 1
    assert rows[0].do == instruction
    assert rows[0].scope == source.rstrip("。")
    assert rows[0].quotes[0].text == source
    assert rows[0].status == "proposed"


def test_even_literal_valid_conversation_confirmation_cannot_auto_activate():
    source = "画地图的时候，标记让我自己选。"
    instruction = "画地图时由本人选择标记"
    ctx, profile, rows, rejected = propose(source, instruction,
        confirmation={"turn_index": 1, "text": "对，就这么记。"})
    assert not rejected
    merge_preferences(profile, rows)
    assert rows[0].status == "proposed" and rows[0].confirmation is None
    assert [h.to for h in rows[0].history] == ["proposed"]
    assert confirmed_preference_payloads(profile) == []
    before = deepcopy(profile.model_dump())
    assert 'source="stated_preference_confirmed"' not in build_messages(
        get_scene("companion_elder"), profile, ctx, "继续。"
    )[0]["content"]
    assert profile.model_dump() == before
    transition_preference(profile, rows[0].id, "confirm")
    assert confirmed_preference_payloads(profile)[0]["do"] == instruction


@pytest.mark.parametrize("source,instruction,kind", [
    ("等我做好记录，你再开门。", "等我做好记录，你再开门。", "action_request"),
    ("我不希望等我把疑问列出来再回应。", "我不希望等我把疑问列出来再回应。", "communication_style"),
    ("别等我把疑问列出来再回应。", "别等我把疑问列出来再回应。", "communication_style"),
    ("等我把疑问列出来再回应，你会这样做吗？", "等我把疑问列出来再回应，你会这样做吗？", "communication_style"),
    ("不是等我把疑问列出来再回应。", "不是等我把疑问列出来再回应。", "communication_style"),
    ("我希望先自己找开关，只有我说需要时，你才告诉我下一步。", "告诉我下一步", "communication_style"),
    ("让我自己先看一遍安装图，卡住以后再告诉我下一步。", "告诉我下一步", "communication_style"),
])
def test_former_grammar_inputs_follow_explicit_semantic_kind_without_authorizing_actions(source, instruction, kind):
    # Former inputs are preserved. These are prescribed software labels, not
    # model classification results; action requests must not become preferences.
    ctx, profile, rows, rejected = propose(source, instruction,
        semantic_review=communication_review(kind=kind))
    if kind != "communication_style":
        assert rows == [] and rejected == ["preference_not_communication_style"]
        assert profile.care_plan is None and profile.care_tasks == []
        return
    assert not rejected and rows[0].status == "proposed"
    assert rows[0].scope == source.rstrip("。")
    merge_preferences(profile, rows)
    assert not confirmed_preference_payloads(profile)
    assert profile.care_plan is None and profile.care_tasks == []
    assert ctx.working_memory == {}


@pytest.mark.parametrize("instruction,review", [
    ("你直接通知外甥", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("下次把吃药剂量翻倍", communication_review(kind="action_request")),
    ("对高DISC按人格分类", communication_review(kind="person_inference")),
    ("每次先问我老板", communication_review(kind="action_request")),
    ("转发这次对话", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("不用遵守医嘱", communication_review(kind="action_request")),
])
def test_grounding_and_dangerous_categories_still_reject(instruction, review):
    _, _, rows, rejected = propose("画地图的时候，标记让我自己选。", instruction, semantic_review=review)
    assert not rows and rejected


def test_unrelated_benign_summary_remains_an_unconfirmed_semantic_error():
    # Deliberately wrong model label: semantic self-report is not a truth oracle.
    _, profile, rows, rejected = propose("画地图的时候，标记让我自己选。", "家里灯泡需要更换")
    assert not rejected
    merge_preferences(profile, rows)
    assert rows[0].status == "proposed" and not confirmed_preference_payloads(profile)


def test_long_scope_cannot_drop_any_uncovered_source_qualification():
    source = "今天只聊这一本旅行手册，" + "有些路段我需要慢慢想清楚，" * 4 + "我说可以再往下。"
    _, _, rows, rejected = propose(source, "旅行手册慢慢商量", scope="今天只聊这一本旅行手册")
    assert not rows and rejected == ["preference_full_scope_too_long"]


def test_long_scope_has_same_length_limit_for_literal_and_paraphrased_do():
    condition = "今天只聊这一本旅行手册，有些路段我需要慢慢想清楚，"
    instruction = "你先等我把这几条路各自经过的地方说完，再和我一起核对每个转弯的位置以及什么时候应该停下来。"
    source = condition + instruction
    assert len(source) > 60 and len(condition) <= 60 and len(instruction) <= 60
    for requested_do in (instruction, "逐段讨论路线并尊重停顿"):
        _, _, rows, rejected = propose(source, requested_do, scope=condition)
        assert not rows and rejected == ["preference_full_scope_too_long"]


def test_nonliteral_quote_still_rejected_when_do_is_an_allowed_paraphrase():
    _, _, rows, rejected = propose("画地图的时候，标记让我自己选。", "画地图时由本人选择标记", quote="地图标记我自己选")
    assert not rows and rejected == ["preference_quote_not_verbatim_user_source"]
