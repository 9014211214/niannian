"""Mechanical source/confirmation tests with stub labels, not semantic scores."""
import json

import pytest

from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.memory_contract import FIELDS, memory_response_schema, validate_memory_response
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import (
    apply_preference_dialogue, judge_preference_confirmation,
)
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, merge_preferences, transition_preference,
)
from tests.test_confirm_judge_v7 import Judge, deny_network, fixture as dialogue_fixture
from tests.test_stated_preferences import _context, _profile, _row


class MemoryReply:
    def __init__(self, rows):
        self.rows = rows
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        payload = {name: [] for name in FIELDS}
        payload["stated_preferences"] = self.rows
        raw = json.dumps(payload, ensure_ascii=False)
        validate_memory_response(raw)
        return raw


def revision_fixture():
    profile = _profile()
    ctx = _context("revision_session")
    old = profile.stated_preferences[0]
    row = {**_row(), "confirmation": None, "supersedes": old.id}
    return profile, ctx, old, row


@pytest.mark.parametrize("basis", [
    pytest.param("absent", id="missing"),
    pytest.param(None, id="null"),
    pytest.param("", id="empty"),
    pytest.param("  ", id="blank"),
    pytest.param("今天所有场合都先问问我。", id="not-verbatim"),
    pytest.param("这样记下好吗？", id="assistant-only"),
    pytest.param("我希望改掉过去的说明。", id="previous-session-only"),
    pytest.param("报名前先问我", id="non-contiguous"),
    pytest.param(1, id="invalid-number"),
    pytest.param({"text": "改一下"}, id="invalid-object"),
])
def test_memory_without_current_revision_evidence_preserves_old_instruction(basis):
    profile, ctx, old, row = revision_fixture()
    old.quotes[0].text += "我希望改掉过去的说明。"
    before = old.model_dump()
    if basis != "absent":
        row["revision_basis"] = basis
    reply = MemoryReply([row])
    result = LLMExtractor(reply).extract(ctx, profile)
    assert len(result.stated_preferences) == 1
    proposal = result.stated_preferences[0]
    assert proposal.supersedes is None
    assert proposal.status == "proposed" and proposal.confirmation is None
    merge_preferences(profile, result.stated_preferences)
    transition_preference(profile, proposal.id, "confirm")
    assert profile.stated_preferences[0].model_dump() == before
    assert len(confirmed_preference_payloads(profile)) == 2
    assert ctx.working_memory["stated_preference_extraction"]["rejected"] == []


def test_current_revision_substring_allows_replacement_only_after_owner_confirmation():
    profile, ctx, old, row = revision_fixture()
    basis = "把原来那条改一下：聊报名时先问我的想法。"
    ctx.add_turn("我再补充一句。" + basis + "说清楚就好。", "我先整理供你确认。")
    row.update(do="聊报名时先问本人的想法", revision_basis=basis,
               quotes=[{"turn_index": 2, "text": ctx.turns[2].user}])
    reply = MemoryReply([row])
    result = LLMExtractor(reply).extract(ctx, profile)
    proposal = result.stated_preferences[0]
    assert proposal.supersedes == old.id
    before = old.model_dump()
    merge_preferences(profile, result.stated_preferences)
    assert profile.stated_preferences[0].model_dump() == before
    transition_preference(profile, proposal.id, "confirm", at="2026-10-20T01:00:00+00:00")
    assert [item.status for item in profile.stated_preferences] == ["superseded", "confirmed"]
    prompt = reply.messages[0]["content"]
    assert "除非老人在本次会话中明确要求修改原来的说法，否则不要提出 supersedes；不要把原有条件改宽或删掉。" in prompt
    assert '"revision_basis":null' in prompt


def test_private_revision_basis_does_not_replace_confirmed_instruction():
    profile, ctx, old, row = revision_fixture()
    basis = "把原来那条改一下。"
    ctx.add_turn(basis, "这一轮不保存。")
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [2]
    row["revision_basis"] = basis
    proposals, rejected = extract_preference_rows([row], ctx, profile)
    assert rejected == [] and proposals[0].supersedes is None
    assert old.status == "confirmed"


def test_revision_basis_does_not_admit_an_old_session_quote():
    profile, ctx, old, row = revision_fixture()
    row["quotes"][0]["session_id"] = "previous_session"
    row["revision_basis"] = ctx.turns[0].user
    proposals, rejected = extract_preference_rows([row], ctx, profile)
    assert proposals == [] and rejected == ["preference_quote_not_verbatim_user_source"]
    assert old.status == "confirmed"


def test_invalid_replacement_evidence_does_not_discard_other_memory_proposals():
    profile, ctx, old, row = revision_fixture()
    second = {**_row(), "do": "报名时先聊本人的想法", "confirmation": None, "supersedes": None}
    result = LLMExtractor(MemoryReply([row, second])).extract(ctx, profile)
    assert len(result.stated_preferences) == 2
    assert all(item.supersedes is None for item in result.stated_preferences)
    assert old.status == "confirmed"


def test_generation_schema_requires_nullable_basis_without_rejecting_legacy_return():
    schema = memory_response_schema()["$defs"]["_Preference"]
    assert "revision_basis" in schema["required"]
    assert schema["properties"]["revision_basis"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert "default" not in schema["properties"]["revision_basis"]
    _, _, _, row = revision_fixture()
    payload = {name: [] for name in FIELDS}
    payload["stated_preferences"] = [row]
    assert validate_memory_response(json.dumps(payload)) == payload


def test_dialogue_confirmation_remains_on_judged_turn_after_later_reminiscence(tmp_path):
    utterance = "好，就这样记。以前我在院子里种过一棵桂花树。"
    profile, ctx, body, _ = dialogue_fixture(utterance)
    confirmed_index = len(ctx.turns)
    judge_preference_confirmation(ctx, profile, utterance, client=Judge("confirm"))
    store = PersonaStore(tmp_path)
    reply = apply_preference_dialogue(ctx, profile, utterance, store=store, response=body)
    assert reply.endswith("好，这条记下了。")
    original = profile.stated_preferences[0].confirmation.model_dump()
    ctx.add_turn(utterance, reply)
    ctx.add_turn("那棵树是我年轻时和邻居一起种的，后来长得比房檐还高。", "我接着听。")
    ctx.add_turn("每到秋天，走过院子都能闻到香味。", "那是很鲜明的回忆。")
    MemoryConsolidator(extractor=LLMExtractor(MemoryReply([]))).consolidate(profile, ctx, persist=False)
    confirmed = profile.stated_preferences[0]
    assert confirmed.confirmation.model_dump() == original
    assert confirmed.confirmation.turn_index == confirmed_index
    assert confirmed.confirmation.text == ctx.turns[confirmed_index].user == utterance
    assert confirmed.confirmation.session_id == ctx.session_id
    assert store.load(profile.user_id).stated_preferences[0].confirmation.model_dump() == original
    assert confirmed.history[-1].quote.model_dump() == original


def test_ui_confirmation_uses_action_time_and_never_a_later_chat_quote():
    profile = _profile(confirmed=False)
    stamp = "2026-10-20T02:34:56+00:00"
    changed = transition_preference(profile, profile.stated_preferences[0].id, "confirm", at=stamp)
    confirmation = changed.confirmation.model_dump()
    assert confirmation == {"source": "ui", "at": stamp}
    ctx = _context("later_session")
    ctx.add_turn("说到以前，我们住过一个很大的院子。", "你还记得院子里的什么？")
    LLMExtractor(MemoryReply([])).extract(ctx, profile)
    assert changed.confirmation.model_dump() == confirmation
