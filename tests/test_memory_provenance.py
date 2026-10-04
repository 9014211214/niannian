"""Source lineage checks use deterministic local extraction responses only."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import HeuristicExtractor, LLMExtractor, MemoryConsolidator
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import EpisodicMemoryEntry, PersonaProfile


class LocalExtractorResponse:
    def __init__(self, episodes):
        self.episodes = episodes
        self.messages = []

    def complete(self, messages):
        self.messages = messages
        return json.dumps({"episodes": self.episodes, "relationships": []}, ensure_ascii=False)


def context():
    ctx = EngineContext(user_id="fictional-source", scene_id="companion_elder", session_id="original-session")
    ctx.add_turn("这纸风车翅膀折反了，怪不得不转。", "我听着。")
    ctx.add_turn("有些事，我自己愿意做。", "嗯。")
    return ctx


def row(quote="有些事，我自己愿意做", index=1):
    return {
        "summary": "模型改写不能成为事实来源", "keywords": [], "referenceable": True,
        "evidence_quote": quote, "turn_index": index,
        "source_session_id": "fabricated-session", "source_turn_index": 99,
        "source_quote": "尚未发生的未来台词",
    }


def test_verified_source_is_saved_and_round_trips_without_accepting_model_lineage(tmp_path):
    ctx = context()
    persona = PersonaProfile(user_id=ctx.user_id)
    store = PersonaStore(tmp_path)
    MemoryConsolidator(store, LLMExtractor(LocalExtractorResponse([row()]))).consolidate(persona, ctx)

    entry = store.load(ctx.user_id).episodic_memory[0]
    assert entry.source_session_id == "original-session"
    assert entry.source_turn_index == 1
    assert entry.source_quote == "有些事，我自己愿意做"
    assert entry.source_quote in ctx.turns[entry.source_turn_index].user
    assert "模型改写" not in entry.summary


def test_normalised_matching_retains_user_punctuation_in_source_quote():
    ctx = context()
    entry = LLMExtractor(LocalExtractorResponse([row("有些事我自己愿意做")])).extract(
        ctx, PersonaProfile(user_id=ctx.user_id)
    ).new_episodes[0]
    assert entry.source_quote == "有些事，我自己愿意做"


def test_unicode_case_matching_still_retains_actual_user_text():
    ctx = context()
    ctx.add_turn("本子上写了 ΟΣ 和 İX。", "嗯。")
    entries = LLMExtractor(LocalExtractorResponse([
        row("ος", 2), row("i\u0307x", 2),
    ])).extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes
    # The combining-mark spelling is not equivalent under the existing
    # matcher; only its accepted quote is persisted, without rewriting it.
    assert len(entries) == 1 and entries[0].source_quote == "ΟΣ"


@pytest.mark.parametrize("quote,index", [
    ("明天下午三点半提醒我拿纸和本子", 1),  # Future story suggestion, not spoken.
    ("有些事，我自己愿意做", 5),            # Future turn ID.
    ("有些事，我自己愿意做", 0),            # Existing quote, wrong turn.
    ("我听着", 0),                          # Agent speech is not user evidence.
    ("有些事，我自己愿意做", True),
])
def test_future_wrong_turn_and_agent_quotes_never_create_memories(quote, index):
    ctx = context()
    ctx.working_memory["ui_future_suggestions"] = ["明天下午三点半提醒我拿纸和本子"]
    client = LocalExtractorResponse([row(quote, index)])
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert result.new_episodes == []
    assert "明天下午三点半" not in client.messages[1]["content"]


def test_excluded_source_is_neither_sent_nor_saved_and_original_index_survives():
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    client = LocalExtractorResponse([row("这纸风车翅膀折反了", 0), row()])
    entries = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes
    assert len(entries) == 1 and entries[0].source_turn_index == 1
    assert "[turn 0]" not in client.messages[1]["content"]
    assert "[turn 1]" in client.messages[1]["content"]


def test_legacy_memory_stays_readable_without_invented_provenance(tmp_path):
    legacy = {"id": "old", "summary": "旧记忆", "timestamp": "2026-04-08"}
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(user_id="legacy", episodic_memory=[legacy])
    store.save(persona)
    entry = store.load("legacy").episodic_memory[0]
    assert entry.summary == "旧记忆" and entry.timestamp == "2026-04-08"
    assert entry.source_session_id is None
    assert entry.source_turn_index is None
    assert entry.source_quote is None


def test_heuristic_multi_turn_digest_does_not_pretend_to_be_a_single_quote():
    ctx = context()
    entry = HeuristicExtractor().extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes[0]
    assert entry.type == "session_digest" and entry.referenceable is False
    assert entry.source_session_id == ctx.session_id
    assert entry.source_turn_index is None and entry.source_quote is None


def test_heuristic_single_retained_turn_has_exact_source_and_original_index():
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    entry = HeuristicExtractor().extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes[0]
    assert entry.source_turn_index == 1
    assert entry.source_quote == ctx.turns[1].user


@pytest.mark.parametrize("index", [-1, True, "1"])
def test_source_turn_index_cannot_coerce_an_invalid_identity(index):
    with pytest.raises(ValidationError):
        EpisodicMemoryEntry(id="invalid", summary="fact", source_turn_index=index)
