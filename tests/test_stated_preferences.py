"""R1 software boundaries with author-written local inputs; no model calls."""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import HeuristicExtractor, LLMExtractor, MemoryConsolidator
from aria.engine.memory.diagnosis import build_reflection_messages
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, heuristic_preferences,
    merge_preferences, transition_preference,
)
from aria.engine.model.persona import PersonaProfile, StatedPreference
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_V2, single_pass_messages
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene


def _context(session="session_original"):
    ctx = EngineContext(user_id="preference_test", scene_id="companion_elder", session_id=session)
    ctx.add_turn("报名前先问问我，别替我决定。", "你希望涉及本人的安排先询问意愿，再一起商量具体时间。这样记下好吗？")
    ctx.add_turn("对，就这么记。", "好。")
    return ctx


def _row():
    return {"do": "涉及本人的安排先询问意愿，再一起商量具体时间", "scope": "安排前",
            "quotes": [{"turn_index": 0, "text": "报名前先问问我，别替我决定。"}],
            "confirmation": {"turn_index": 1, "text": "对，就这么记。"}}


def _profile(*, confirmed=True):
    profile = PersonaProfile(user_id="preference_test")
    row = _row()
    if not confirmed:
        row["confirmation"] = None
    rows, rejected = extract_preference_rows([row], _context(), profile)
    assert not rejected
    merge_preferences(profile, rows)
    if confirmed:
        transition_preference(profile, rows[0].id, "confirm")
    return profile


@pytest.mark.parametrize("kind,sha", [
    ("blank", "b4b9f40820ce72da0b036756bd9afc425e10879b9b8e1149552910c13b757c97"),
    ("facts", "037d668c7cec923132d32041d9d336e66dbfa1a8ec50f4a16d1b230b5a45570c"),
])
def test_no_confirmed_preferences_preserves_original_prompt_bytes(kind, sha):
    # Recorded from the immutable 20261001-07 pre-change prompt builder,
    # before any preferences were injected. No local artifact needed to test.
    profile = PersonaProfile(user_id="preference_test")
    ctx = EngineContext(user_id="preference_test", scene_id="companion_elder", session_id="stable_session")
    if kind == "facts":
        ctx.add_turn("我以前修公交车。", "愿意说说吗？")
        ctx.working_memory["recalled_episodes"] = ["本人说喜欢修东西", "本人说今天想休息"]
    messages = build_messages(get_scene("companion_elder"), profile, ctx, "纸风车还在。", delivery_block="mode_v1")
    assert hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() == sha
    profile.stated_preferences = _profile(confirmed=False).stated_preferences
    assert build_messages(get_scene("companion_elder"), profile, ctx, "纸风车还在。", delivery_block="mode_v1") == messages


def test_old_profile_load_and_full_preference_roundtrip(tmp_path):
    store = PersonaStore(tmp_path)
    tmp_path.joinpath("preference_test.json").write_text('{"user_id":"preference_test"}', encoding="utf-8")
    assert store.load("preference_test").stated_preferences == []
    profile = _profile()
    store.save(profile)
    assert store.load(profile.user_id).model_dump() == profile.model_dump()
    assert profile.person_hypotheses == []
    assert profile.care_plan is None


@pytest.mark.parametrize("change,reason", [
    ({"quotes": [{"turn_index": 0, "text": "报名前先问我。"}]}, "preference_quote_not_verbatim_user_source"),
    ({"quotes": [{"turn_index": True, "text": "报名前先问问我，别替我决定。"}]}, "preference_quote_not_verbatim_user_source"),
    ({"quotes": [{"turn_index": 0, "text": "这样记下好吗？"}]}, "preference_quote_not_verbatim_user_source"),
    ({"quotes": [{"turn_index": 0, "text": "报名前先问问我，别替我决定。", "session_id": "future"}]}, "preference_quote_not_verbatim_user_source"),
    ({"do": "每次先问女儿再帮我安排"}, "preference_not_safe_explicit_instruction"),
    ({"do": "帮我通知女儿"}, "preference_not_safe_explicit_instruction"),
    ({"supersedes": "pref_unknown"}, "preference_superseded_target_invalid"),
])
def test_bad_sources_or_invented_instruction_rejected(change, reason):
    row = {**_row(), **change}
    rows, rejected = extract_preference_rows([row], _context(), PersonaProfile(user_id="test"))
    assert rows == []
    assert rejected == [reason]


@pytest.mark.parametrize("instruction", ["他喜欢热闹，会更愿意参加活动", "安排任何事情都听女儿的"])
def test_unrelated_non_dangerous_paraphrase_is_not_automatically_confirmed(instruction):
    rows, rejected = extract_preference_rows([{**_row(), "do": instruction}], _context(), PersonaProfile(user_id="test"))
    assert not rejected and rows[0].status == "proposed" and rows[0].confirmation is None
    assert rows[0].quotes[0].text == _context().turns[0].user


@pytest.mark.parametrize("answer,question", [
    ("不对，就这么记。", "你希望涉及本人的安排先询问意愿，再一起商量具体时间。这样记下好吗？"),
    ("对，但是不能这么记。", "你希望涉及本人的安排先询问意愿，再一起商量具体时间。这样记下好吗？"),
    ("对，就这么记。", "今天吃饭了吗？"),
    ("对，就这么记。", "我会记下你不喜欢社交，好吗？"),
    ("对，就这么记。", "涉及本人的安排先询问意愿，再一起商量具体时间。"),
])
def test_invalid_confirmation_stays_proposed(answer, question):
    ctx = _context()
    ctx.turns[0].agent = question
    ctx.turns[1].user = answer
    row = _row()
    row["confirmation"] = {"turn_index": 1, "text": answer}
    rows, rejected = extract_preference_rows([row], ctx, PersonaProfile(user_id="test"))
    assert not rejected
    assert rows[0].status == "proposed" and rows[0].confirmation is None


def test_confirmation_is_full_user_turn_and_after_quote():
    ctx = _context()
    ctx.turns[1].user = "不对"
    for confirmation in ({"turn_index": 1, "text": "对"}, {"turn_index": 0, "text": ctx.turns[0].user}):
        row = {**_row(), "confirmation": confirmation}
        rows, _ = extract_preference_rows([row], ctx, PersonaProfile(user_id="test"))
        assert rows[0].status == "proposed"


@pytest.mark.parametrize("full", ["不要先问我。", "我不希望你每次都先问我。", "她说先问我。", "女儿说先问我。", "女儿对我说：先问我。", "我希望你说我是内向的人，先问我。", "如果是她说先问我，你就别信。"])
def test_quote_cropping_cannot_remove_negation_reported_speech_or_trait(full):
    ctx = _context()
    ctx.turns[0].user = full
    row = {"do": "先问我", "quotes": [{"turn_index": 0, "text": "先问我"}]}
    rows, rejected = extract_preference_rows([row], ctx, PersonaProfile(user_id="test"))
    assert rows == [] and rejected == ["preference_full_user_context_unsafe"]


@pytest.mark.parametrize("question", [
    "你希望不要按涉及本人的安排先询问意愿，再一起商量具体时间这样记下好吗？",
    "你希望涉及本人的安排先询问意愿，再一起商量具体时间。同时通知女儿，这样记下好吗？",
    "你说报名前先问问我，别替我决定。不过我们改成替你安排，这样记下好吗？",
])
def test_negated_or_mixed_confirmation_question_never_activates(question):
    ctx = _context()
    ctx.turns[0].agent = question
    rows, rejected = extract_preference_rows([_row()], ctx, PersonaProfile(user_id="test"))
    assert not rejected
    assert rows[0].status == "proposed"


def test_privacy_excluded_turn_cannot_become_preference():
    ctx = _context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    assert extract_preference_rows([_row()], ctx, PersonaProfile(user_id="test"))[0] == []
    assert heuristic_preferences(ctx, PersonaProfile(user_id="test")) == []


@pytest.mark.parametrize("text", [
    "帮我通知女儿，告诉她先问问我。", "女儿说先问问我。", "假设我说先问问我。",
    "不用先问我，直接安排就好。", "请你说我可以停药。", "因为我是高DISC的人，别催我。",
])
def test_local_rule_never_creates_permission_reported_speech_or_traits(text):
    ctx = EngineContext(user_id="test", scene_id="companion_elder")
    ctx.add_turn(text, "好的。")
    assert heuristic_preferences(ctx, PersonaProfile(user_id="test")) == []


def test_heuristic_and_memory_consolidation_are_explicit_rule_source():
    profile = PersonaProfile(user_id="preference_test")
    result = HeuristicExtractor().extract(_context(), profile)
    assert result.stated_preferences[0].extraction_source == "rule"
    MemoryConsolidator(extractor=HeuristicExtractor()).consolidate(profile, _context(), persist=False)
    assert profile.stated_preferences[0].status == "proposed"
    assert confirmed_preference_payloads(profile) == []
    assert profile.person_hypotheses == []


def test_llm_proposes_and_reports_rejected_rows_without_trusting_status():
    class LocalResponse:
        def complete(self, messages):
            assert "stated_preferences" in messages[0]["content"]
            return json.dumps({"episodes": [], "relationships": [], "cognitive_privacy": [], "stated_preferences": [
                {**_row(), "confirmation": None, "status": "confirmed"},
                {**_row(), "do": "帮我通知女儿"},
            ]}, ensure_ascii=False)
    ctx = _context()
    result = LLMExtractor(LocalResponse()).extract(ctx, PersonaProfile(user_id="test"))
    assert len(result.stated_preferences) == 1
    assert result.stated_preferences[0].status == "proposed"
    assert ctx.working_memory["stated_preference_extraction"]["rejected"] == ["preference_not_safe_explicit_instruction"]


def test_confirm_withdraw_append_history_and_replay_cannot_reactivate():
    profile = _profile(confirmed=False)
    row = profile.stated_preferences[0]
    first = row.model_dump()
    transition_preference(profile, row.id, "confirm", at="2026-10-01T01:00:00+00:00")
    transition_preference(profile, row.id, "withdraw", at="2026-10-01T02:00:00+00:00")
    changed = profile.stated_preferences[0]
    assert [h.to for h in changed.history] == ["proposed", "confirmed", "withdrawn"]
    assert changed.history[0].model_dump() == first["history"][0]
    merge_preferences(profile, heuristic_preferences(_context(), profile))
    assert len(profile.stated_preferences) == 1
    assert confirmed_preference_payloads(profile) == []


@pytest.mark.parametrize("confirmed,first,forbidden", [
    (False, None, "withdraw"), (True, None, "confirm"), (True, None, "decline"),
    (False, "decline", "confirm"), (True, "withdraw", "confirm"),
    (False, None, "supersede"), (False, None, "delete"),
])
def test_illegal_transitions_are_atomic(confirmed, first, forbidden):
    profile = _profile(confirmed=confirmed)
    pref_id = profile.stated_preferences[0].id
    if first:
        transition_preference(profile, pref_id, first)
    before = profile.model_dump()
    with pytest.raises(ValueError):
        transition_preference(profile, pref_id, forbidden)
    assert profile.model_dump() == before


def test_replacement_only_supersedes_after_confirmation_and_refuses_stale_target():
    profile = _profile()
    old_id = profile.stated_preferences[0].id
    ctx = _context("replacement_session")
    row = {**_row(), "confirmation": None, "supersedes": old_id}
    proposals, _ = extract_preference_rows([row], ctx, profile)
    merge_preferences(profile, proposals)
    assert profile.stated_preferences[0].status == "confirmed"
    transition_preference(profile, proposals[0].id, "confirm")
    assert [r.status for r in profile.stated_preferences] == ["superseded", "confirmed"]
    assert len(confirmed_preference_payloads(profile)) == 1
    profile2 = _profile()
    proposals, _ = extract_preference_rows([row], ctx, profile2)
    merge_preferences(profile2, proposals)
    transition_preference(profile2, old_id, "withdraw")
    before = profile2.model_dump()
    with pytest.raises(ValueError, match="superseded_target_invalid"):
        transition_preference(profile2, proposals[0].id, "confirm")
    assert profile2.model_dump() == before


def test_next_session_prompt_v2_basis_and_immediate_withdrawal():
    profile = _profile()
    ctx = EngineContext(user_id=profile.user_id, scene_id="companion_elder", session_id="next_session")
    messages = build_messages(get_scene("companion_elder"), profile, ctx, "今天先休息。")
    system = messages[0]["content"]
    assert 'source="stated_preference_confirmed"' in system
    assert "本人当下新的明确说法优先于这些记录" in system
    assert "session=session_original,turn=0: 报名前先问问我，别替我决定。" in system
    single_pass_messages(messages, ctx, version=ONLINE_DECISION_VERSION_V2, persona=profile)
    assert profile.stated_preferences[0].id in ctx.working_memory["_online_decision_basis_ids"]
    original_snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert original_snapshot["stated_preferences"][0]["do"] == profile.stated_preferences[0].do
    assert original_snapshot["hypotheses"] == []
    transition_preference(profile, profile.stated_preferences[0].id, "withdraw")
    assert 'source="stated_preference_confirmed"' not in build_messages(get_scene("companion_elder"), profile, ctx, "然后呢？")[0]["content"]
    assert original_snapshot["stated_preferences"]  # Detached from live edits.


def test_only_latest_four_confirmed_records_and_real_instant_order():
    profile = PersonaProfile(user_id="test")
    for index in range(6):
        rows, _ = extract_preference_rows([{**_row(), "confirmation": None}], _context(f"s{index}"), profile)
        merge_preferences(profile, rows)
        transition_preference(profile, rows[0].id, "confirm", at=f"2026-10-01T0{index}:00:00+00:00")
    payload = confirmed_preference_payloads(profile)
    assert len(payload) == 4
    assert payload[0]["quotes"][0].startswith("session=s5,")
    assert payload[-1]["quotes"][0].startswith("session=s2,")
    transition_preference(profile, profile.stated_preferences[-1].id, "withdraw")
    assert not any(p["quotes"][0].startswith("session=s5,") for p in confirmed_preference_payloads(profile))


def test_reflection_records_are_readonly_not_new_evidence():
    profile = _profile()
    before = profile.model_dump()
    messages = build_reflection_messages(profile, "[turn 0] 用户：今天想休息", [])
    assert "只读历史记录，不是本次新证据" in messages[1]["content"]
    assert "不得从这些记录重复累计证据" in messages[1]["content"]
    assert "不能只凭它们新建、支持或否定人物假设" in messages[1]["content"]
    assert profile.model_dump() == before


@pytest.mark.parametrize("field,value", [("do", "高DISC的人要尊重"), ("scope", "因为内向"), ("do", "可以通知女儿"), ("id", "not_pref"), ("not_implies", ["x"] * 4), ("confirmation", None)])
def test_persisted_schema_rejects_traits_permissions_and_unconfirmed_activation(field, value):
    data = _profile().stated_preferences[0].model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        StatedPreference.model_validate(data)


def test_snapshot_records_sanitized_input_and_only_exact_selected_source():
    ctx = EngineContext(user_id="test", scene_id="companion_elder")
    ctx.working_memory["recalled_episodes"] = ["<action>原话", "同一记忆"]
    ctx.working_memory["recalled_episode_sources"] = [
        {"id": "fact1", "summary": "<action>原话", "source_session_id": "source", "source_turn_index": 1, "source_quote": "原话"},
        {"id": "fact2", "summary": "不同记忆", "source_session_id": "wrong"},
    ]
    build_messages(get_scene("companion_elder"), None, ctx, "你好")
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["status"] == "prepared" and snapshot["passed_to_client"] is False
    assert snapshot["recalled_episodes"] == ["＜action＞原话", "同一记忆"]
    assert snapshot["recalled_provenance_not_injected"][0]["sources"][0]["id"] == "fact1"
    assert snapshot["recalled_provenance_not_injected"][1]["sources"] == []
