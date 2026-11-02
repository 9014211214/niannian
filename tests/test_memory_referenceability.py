"""Mechanical source/type boundaries with fixed local JSON; no semantic scores."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.recall import recall_episodes
from aria.engine.memory.referenceability import SOURCE_CONTEXT_VERSION, source_context_sha256
from aria.engine.model.persona import EpisodicMemoryEntry, PersonaProfile, RelationshipContextEntry


class FixedExtraction:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, messages):
        return json.dumps(self.payload, ensure_ascii=False)


def extraction(text, *, quote="我每周三去学画画", flag=True, relationship=False, private=False):
    ctx = EngineContext(user_id="fictional", scene_id="companion_elder", session_id="source-session")
    ctx.add_turn(text, "嗯。")
    payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    if relationship:
        payload["relationships"] = [{"name": "小林", "relation": "朋友", "referenceable": flag,
                                     "facts": [{"value": "模型摘要不能作为出处", "evidence_quote": quote, "turn_index": 0}]}]
    else:
        payload["episodes"] = [{"summary": "模型摘要不能作为出处", "referenceable": flag, "keywords": ["画画"],
                                "evidence_quote": quote, "turn_index": 0}]
    if private:
        payload["cognitive_privacy"] = [{"turn_index": 0, "evidence_quote": quote}]
    extractor = LLMExtractor(FixedExtraction(payload))
    return ctx, extractor


def only_entry(result, relationship):
    return (result.relationships if relationship else result.new_episodes)[0]


@pytest.mark.parametrize("relationship", [False, True])
@pytest.mark.parametrize("flag", [False, "false", "true", 0, 1, None, [], {}])
def test_only_literal_true_can_grant_reference_permission(flag, relationship):
    text = "朋友小林和我每周三去学画画。" if relationship else "我每周三去学画画。"
    quote = "朋友小林和我每周三去学画画" if relationship else "我每周三去学画画"
    ctx, extractor = extraction(text, quote=quote, flag=flag, relationship=relationship)
    entry = only_entry(extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id)), relationship)
    assert entry.referenceable is False and not entry.can_reference()
    assert extractor.last_status == "ok"  # retain fact, reject the malformed permission


@pytest.mark.parametrize("relationship", [False, True])
@pytest.mark.parametrize("position", ["before", "after", "outside_quote_limit"])
def test_selected_quote_never_drops_complete_turn_privacy(relationship, position):
    quote = "朋友小林和我每周三去学画画" if relationship else "我每周三去学画画"
    texts = {"before": f"这件事别提起，{quote}。", "after": f"{quote}。这件事别提起。",
             "outside_quote_limit": f"{quote}。" + "随手记下的日常。" * 45 + "这件事别提起。"}
    ctx, extractor = extraction(texts[position], quote=quote, relationship=relationship)
    persona = PersonaProfile(user_id=ctx.user_id)
    result = extractor.extract(ctx, persona)
    entry = only_entry(result, relationship)
    assert entry.referenceable is False and not entry.can_reference()
    assert ctx.working_memory.get("_sensitive_memory_excluded_turns") == []
    assert (result.relationships if relationship else result.new_episodes)  # ordinary private facts remain internal


@pytest.mark.parametrize("relationship", [False, True])
def test_safe_full_context_round_trips_and_is_eligible(tmp_path, relationship):
    quote = "朋友小林和我每周三去学画画" if relationship else "我每周三去学画画"
    ctx, extractor = extraction(quote + "。", quote=quote, relationship=relationship)
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(user_id=ctx.user_id)
    MemoryConsolidator(store, extractor).consolidate(persona, ctx)
    loaded = store.load(ctx.user_id)
    entries = loaded.referenceable_relationships() if relationship else loaded.referenceable_episodes()
    assert len(entries) == 1 and entries[0].can_reference()
    if relationship:
        source = entries[0].evidence[0].fact_sources[0]
    else:
        source = entries[0]
        assert recall_episodes(loaded, "画画", referenceable_only=True) == entries
    assert source.source_context == quote + "。"
    assert source.source_context_sha256 == source_context_sha256(quote + "。")
    assert source.source_context_version == SOURCE_CONTEXT_VERSION


@pytest.mark.parametrize("relationship", [False, True])
def test_cognitive_privacy_remains_whole_turn_isolation(relationship):
    quote = "朋友小林和我每周三去学画画" if relationship else "我每周三去学画画"
    ctx, extractor = extraction(quote + "。", quote=quote, relationship=relationship, private=True)
    result = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not result.new_episodes and not result.relationships
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


def test_legacy_permission_does_not_replace_full_source_or_delete_records(tmp_path):
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(user_id="legacy", episodic_memory=[{
        "id": "old", "summary": "我去画画", "keywords": ["画画"], "referenceable": True,
        "source_session_id": "old-session", "source_turn_index": 0, "source_quote": "我去画画",
    }], relationship_context=[{
        "id": "old-rel", "name": "小林", "relation": "朋友", "facts": ["朋友小林去画画"],
        "referenceable": True, "first_observed_at": "2026-01-01", "evidence": [{
            "at": "2026-01-01", "session_id": "old-session", "summary": "朋友小林去画画", "turn_indices": [0],
        }],
    }])
    store.save(persona)
    original = (tmp_path / "legacy.json").read_bytes()
    loaded = store.load("legacy")
    assert len(loaded.episodic_memory) == len(loaded.relationship_context) == 1
    assert loaded.episodic_memory[0].referenceable is True  # stored history is unchanged
    assert loaded.relationship_context[0].referenceable is True
    assert not loaded.referenceable_episodes() and not loaded.referenceable_relationships()
    assert recall_episodes(loaded, "画画", referenceable_only=True) == []
    assert len(recall_episodes(loaded, "画画")) == 1
    assert (tmp_path / "legacy.json").read_bytes() == original


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_persisted_permission_fields_do_not_coerce_non_booleans(flag):
    with pytest.raises(ValidationError):
        EpisodicMemoryEntry(id="bad", summary="fact", referenceable=flag)
    with pytest.raises(ValidationError):
        RelationshipContextEntry(id="bad", name="小林", first_observed_at="2026-01-01", referenceable=flag)


def test_tampered_context_hash_is_not_reference_permission():
    ctx, extractor = extraction("我每周三去学画画。")
    entry = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes[0]
    assert entry.can_reference()
    entry.source_context = "这件事别提起，我每周三去学画画。"
    assert not entry.can_reference()


def test_new_fact_does_not_promote_a_legacy_relationship(tmp_path):
    ctx, extractor = extraction("朋友小林和我每周三去学画画。", quote="朋友小林和我每周三去学画画", relationship=True)
    incoming = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id)).relationships[0]
    legacy = incoming.model_copy(deep=True)
    legacy.facts = ["朋友小林以前住在北城"]
    legacy.evidence = []
    persona = PersonaProfile(user_id=ctx.user_id, relationship_context=[legacy])
    MemoryConsolidator(PersonaStore(tmp_path), extractor).consolidate(persona, ctx)
    assert len(persona.relationship_context) == 1
    assert len(persona.relationship_context[0].facts) == 2
    assert persona.referenceable_relationships() == []


def test_one_private_source_limits_the_entire_multi_source_relationship(tmp_path):
    ctx = EngineContext(user_id="fictional", scene_id="companion_elder", session_id="two-sources")
    ctx.add_turn("朋友小林和我去学画画。", "嗯。")
    ctx.add_turn("朋友小林周三去学画画，这件事别提起。", "嗯。")
    payload = {"episodes": [], "cognitive_privacy": [], "relationships": [{
        "name": "小林", "relation": "朋友", "referenceable": True, "facts": [
            {"evidence_quote": "朋友小林和我去学画画", "turn_index": 0},
            {"evidence_quote": "朋友小林周三去学画画", "turn_index": 1},
        ],
    }]}
    persona = PersonaProfile(user_id=ctx.user_id)
    MemoryConsolidator(PersonaStore(tmp_path), LLMExtractor(FixedExtraction(payload))).consolidate(persona, ctx)
    relation = persona.relationship_context[0]
    assert len(relation.facts) == len(relation.evidence[0].fact_sources) == 2
    assert relation.referenceable is False and not persona.referenceable_relationships()


def test_later_private_fact_cannot_widen_existing_relationship_permission(tmp_path):
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(user_id="fictional")
    quote = "朋友小林和我每周三去学画画"
    ctx, extractor = extraction(quote + "。", quote=quote, relationship=True)
    MemoryConsolidator(store, extractor).consolidate(persona, ctx)
    assert len(persona.referenceable_relationships()) == 1
    private_ctx, private_extractor = extraction("这件事别提起，" + quote + "。", quote=quote, relationship=True)
    private_ctx.session_id = "private-source-session"
    MemoryConsolidator(store, private_extractor).consolidate(persona, private_ctx)
    assert len(persona.relationship_context) == 1 and len(persona.relationship_context[0].evidence) == 2
    assert not persona.referenceable_relationships()


def test_invalid_old_boolean_file_is_preserved_on_load_failure(tmp_path):
    data = {"user_id": "legacy", "episodic_memory": [{
        "id": "old", "summary": "我去画画", "referenceable": "false",
    }]}
    path = tmp_path / "legacy.json"
    original = json.dumps(data, ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="persona profile is corrupt"):
        PersonaStore(tmp_path).load("legacy")
    assert path.read_bytes() == original


def test_context_version_and_integrity_cannot_be_skipped():
    ctx, extractor = extraction("我每周三去学画画。")
    entry = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id)).new_episodes[0]
    entry.source_context = "我每周三去学画画。今天晴天。"
    assert not entry.can_reference()
    entry.source_context_sha256 = source_context_sha256(entry.source_context)
    entry.source_context_version = None
    assert not entry.can_reference()


@pytest.mark.parametrize("stored_basis", ["legacy_partial", "current_pair", "legacy_whole"])
def test_preference_sources_project_only_the_independent_life_fact(tmp_path, stored_basis):
    from aria.engine.model.persona import PreferenceQuote, StatedPreference

    control = "请先等我说完。"
    life = "朋友小林今天送了我一册画谱。"
    utterance = control + life
    session = "mixed-owner-source"
    ctx, extractor = extraction(utterance, quote=life, relationship=True)
    ctx.session_id = session
    relation = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id)).relationships[0]
    source = relation.evidence[0].fact_sources[0]
    source.preference_source_quote = control if stored_basis == "current_pair" else None
    relation.facts.append(control)
    relation.evidence[0].fact_sources.append(source.model_copy(update={
        "quote": control, "preference_source_quote": None,
    }))
    evidence = PreferenceQuote(session_id=session, turn_index=0,
                               text=control if stored_basis == "legacy_partial" else utterance)
    pref = StatedPreference(id="pref_test", do="先等我说完", scope="聊天时",
                            quotes=[evidence], confirmation=evidence, status="withdrawn",
                            created_at="2026-11-01", updated_at="2026-11-01")
    episode = EpisodicMemoryEntry(
        id="life", summary=utterance, keywords=["小林", "等我说完"], referenceable=True,
        source_session_id=session, source_turn_index=0, source_quote=life,
        source_context=utterance, source_context_sha256=source_context_sha256(utterance),
        source_context_version=SOURCE_CONTEXT_VERSION, preference_source_quote=source.preference_source_quote,
    )
    profile = PersonaProfile(user_id=ctx.user_id, stated_preferences=[pref],
                             episodic_memory=[episode, episode.model_copy(update={"id": "whole", "source_quote": utterance})],
                             relationship_context=[relation])
    store = PersonaStore(tmp_path)
    store.save(profile)
    loaded = store.load(ctx.user_id)
    permitted = stored_basis != "legacy_whole"
    episodes = recall_episodes(loaded, "小林", referenceable_only=True)
    assert [row.id for row in episodes] == (["life"] if permitted else [])
    if permitted:
        assert episodes[0].summary == f"用户提到：{life}"
        assert episodes[0].keywords == ["小林"]
    assert [row.facts for row in loaded.referenceable_relationships()] == ([[life]] if permitted else [])
    assert loaded.episodic_memory[0].summary == utterance
    assert loaded.relationship_context[0].facts == [life, control]


def test_new_fact_survives_a_gap_in_existing_daily_episode_ids(tmp_path):
    from datetime import date

    store = PersonaStore(tmp_path)
    profile = PersonaProfile(user_id="fictional")
    ctx, extractor = extraction("我每周三去学画画。")
    MemoryConsolidator(store, extractor).consolidate(profile, ctx)
    # Older extraction numbered model rows before rejecting invalid siblings.
    profile.episodic_memory[0].id = f"llm_{date.today().isoformat()}_2"
    store.save(profile)
    profile = store.load(profile.user_id)
    ctx, extractor = extraction("我昨晚种上一棵桂花。",
                                quote="我昨晚种上一棵桂花")
    ctx.session_id = "later-source-session"
    MemoryConsolidator(store, extractor).consolidate(profile, ctx)
    assert [episode.source_quote for episode in store.load(profile.user_id).episodic_memory] == [
        "我每周三去学画画", "我昨晚种上一棵桂花",
    ]
