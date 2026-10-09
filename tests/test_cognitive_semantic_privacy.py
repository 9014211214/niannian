"""Fixed semantic labels test transport and permissions, never model accuracy."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from aria.engine.cognitive import (
    apply_cognitive_withdrawal, capture_cognitive_turn, cognitive_action, cognitive_sensitive, cognitive_sources,
)
from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from tests.test_cognitive_change_v2 import T0, enabled
from tests.test_cognitive_memory_protocol import LocalResponse, complete_payload
from tests.test_offline_queue import Store


NATURAL = "端着碗走到窗边，站了好久才明白是要去厨房盛汤。"
NORMAL = "我喜欢做纸风车。"


def conversation(persona):
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="privacy-session")
    for index, text in enumerate((NATURAL, NORMAL)):
        capture_cognitive_turn(ctx, persona, text, now=T0 + timedelta(days=1, minutes=index))
        ctx.add_turn(text, "回应：" + text)
    return ctx


def labeled_payload(*, observation=True, privacy=True):
    payload = complete_payload()
    payload["cognitive_privacy"] = [{"turn_index": 0, "evidence_quote": NATURAL}] if privacy else []
    if observation:
        payload["cognitive_observations"] = [{"domain": "memory", "evidence_quote": NATURAL,
            "turn_index": 0, "functional_impact": True, "event_kind": "new_event", "prior_event_id": None}]
    # Deliberately contradictory fixed labels: privacy must win over every
    # ordinary extractor even if the model proposes this text in both places.
    payload["episodes"] = [{"summary": "fixture", "keywords": [], "referenceable": True,
        "evidence_quote": text, "turn_index": index} for index, text in enumerate((NATURAL, NORMAL))]
    payload["relationships"] = [{"name": "厨房", "relation": None, "referenceable": True,
        "facts": [{"evidence_quote": NATURAL, "turn_index": 0}]}]
    payload["stated_preferences"] = [{"do": "请留出考虑时间", "scope": "当前交流",
        "quotes": [{"turn_index": 0, "text": NATURAL}], "confirmation": None}]
    return payload


def test_all_consented_turns_are_reviewed_without_positive_keyword_eligibility():
    persona = enabled()
    ctx = conversation(persona)
    assert not cognitive_sensitive(NATURAL)
    assert [row.text for row in cognitive_sources(persona, ctx)] == [NATURAL, NORMAL]
    client = LocalResponse(labeled_payload())
    result = LLMExtractor(client).extract(ctx, persona)
    assert len(client.calls) == 1
    assert NATURAL in client.calls[0][1]["content"] and NORMAL in client.calls[0][1]["content"]
    assert [row.source_quote for row in result.cognitive_observations] == [NATURAL]
    assert result.cognitive_observations[0].functional_impact
    assert [row.source_quote for row in result.new_episodes] == [NORMAL]
    assert not result.relationships and not result.stated_preferences
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


def test_literal_observation_also_masks_source_when_privacy_list_omits_it():
    persona = enabled()
    ctx = conversation(persona)
    result = LLMExtractor(LocalResponse(labeled_payload(privacy=False))).extract(ctx, persona)
    assert len(result.cognitive_observations) == 1
    assert [row.source_quote for row in result.new_episodes] == [NORMAL]
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("revoke_after_capture", [False, True])
def test_disabled_or_revoked_consent_allows_privacy_filter_but_no_observations(revoke_after_capture):
    persona = enabled() if revoke_after_capture else PersonaProfile(user_id="owner")
    ctx = conversation(persona)
    if revoke_after_capture:
        cognitive_action(persona, "disable", actor="owner", delete_records=False, now=T0 + timedelta(days=1, hours=1))
    assert cognitive_sources(persona, ctx) == []
    client = LocalResponse(labeled_payload(observation=False))
    result = LLMExtractor(client).extract(ctx, persona)
    assert not result.cognitive_observations
    assert [row.source_quote for row in result.new_episodes] == [NORMAL]
    assert "cognitive_observations" not in client.calls[0][0]["content"]
    assert not persona.cognitive_observations and not persona.cognitive_trend.session_times


def test_disabled_model_cannot_create_observations_or_ordinary_memory():
    persona = PersonaProfile(user_id="owner")
    ctx = conversation(persona)
    extractor = LLMExtractor(LocalResponse(labeled_payload()))
    result = extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert not result.cognitive_observations and not result.new_episodes
    assert ctx.working_memory.get("_memory_privacy_review_completed") is not True


@pytest.mark.parametrize("bad", [None, "[]", False, [None], [{}],
    [{"turn_index": True, "evidence_quote": NATURAL}],
    [{"turn_index": 2, "evidence_quote": NATURAL}],
    [{"turn_index": 0, "evidence_quote": "他站着想了一会儿"}],
    [{"turn_index": 0, "evidence_quote": NATURAL, "extra": "unsupported"}]])
def test_invalid_privacy_marking_fails_closed_before_all_extraction(bad):
    persona = enabled()
    ctx = conversation(persona)
    payload = labeled_payload()
    payload["cognitive_privacy"] = bad
    extractor = LLMExtractor(LocalResponse(payload))
    result = extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert not result.new_episodes and not result.stated_preferences and not result.cognitive_observations
    assert ctx.working_memory.get("_memory_privacy_review_completed") is not True


def handler(store, client):
    extractor = LLMExtractor(client)
    def memory(persona, ctx):
        MemoryConsolidator(store, extractor).consolidate(persona, ctx, persist=False)
        return persona, extractor.last_status, extractor.last_error_type
    return memory


def test_queue_atomically_persists_mask_and_restart_reflects_filtered_original(tmp_path):
    persona = enabled()
    store = PersonaStore(tmp_path)
    queue = OfflineSessionQueue(persona, store, now=lambda: T0 + timedelta(days=1, hours=2))
    assert queue.enqueue(conversation(persona))["status"] == "queued"
    client = LocalResponse(labeled_payload())
    seen = []
    def reflection(profile, ctx):
        seen.append(deepcopy(ctx))
        assert profile.cognitive_observations == []
        assert "_cognitive_source_turns" not in ctx.working_memory
        assert ctx.working_memory["_memory_privacy_review_completed"] is True
        assert [(row.user, row.agent) for row in ctx.turns] == [("", ""), (NORMAL, "回应：" + NORMAL)]
        return profile, "failed" if len(seen) == 1 else "ok", "person_reflection_failed"
    assert queue.drain(handler(store, client), reflection)["status"] == "failed"
    saved = store.load(persona.user_id)
    job = saved.offline_sessions[0]
    assert job.memory_completed and job.memory_privacy_review_completed
    assert job.excluded_turn_indices == [0] and job.turns[0].user == job.turns[0].agent == ""
    assert [row.source_quote for row in saved.episodic_memory] == [NORMAL]
    restored = OfflineSessionQueue(saved, store, now=lambda: T0 + timedelta(days=1, hours=3))
    assert restored.drain(handler(store, client), reflection)["status"] == "completed"
    assert len(client.calls) == 1 and len(seen) == 2


def test_privacy_save_failure_cannot_partially_commit_memory_or_run_reflection():
    persona = enabled()
    store = Store(fail_on=(3,))  # enqueue, attempt, then atomic memory/mask save
    queue = OfflineSessionQueue(persona, store, now=lambda: T0 + timedelta(days=1, hours=2))
    queue.enqueue(conversation(persona))
    client = LocalResponse(labeled_payload())
    result = queue.drain(handler(store, client), lambda *_: pytest.fail("uncommitted privacy must not reach reflection"))
    assert result["error_code"] == "offline_store_failed"
    assert not store.saved.episodic_memory
    assert not store.saved.offline_sessions[0].memory_completed
    assert not store.saved.offline_sessions[0].memory_privacy_review_completed
    assert len(client.calls) == 1


def test_legacy_completed_memory_without_privacy_receipt_seals_without_client_call(tmp_path):
    persona = PersonaProfile(user_id="owner")
    store = PersonaStore(tmp_path)
    queue = OfflineSessionQueue(persona, store, now=lambda: T0 + timedelta(days=1, hours=2))
    queue.enqueue(conversation(persona), memory_completed=True)
    engine_like = SimpleNamespace(hypothesis_updater=SimpleNamespace(
        llm=object(), update=lambda *_: pytest.fail("unreviewed old LLM job must not be reflected")))
    result = queue.drain(lambda *_: pytest.fail("completed old memory must not be rerun"),
        lambda profile, ctx: Engine._offline_reflection_stage(engine_like, profile, ctx))
    assert result["status"] == "completed" and result["pending_count"] == 0
    audit = queue.persona.offline_session_audit[-1]
    assert audit.outcome == "reflection_failed" and audit.error_code == "memory_privacy_unreviewed"


@pytest.mark.parametrize("completed,flag,expected", [(False, True, False), (True, True, True), (True, None, False), (True, 1, False)])
def test_enqueue_preserves_only_explicit_completed_memory_privacy_receipt(tmp_path, completed, flag, expected):
    persona = PersonaProfile(user_id="owner")
    ctx = conversation(persona)
    ctx.working_memory["_memory_privacy_review_completed"] = flag
    queue = OfflineSessionQueue(persona, PersonaStore(tmp_path), now=lambda: T0 + timedelta(days=1, hours=2))
    queue.enqueue(ctx, memory_completed=completed)
    assert queue.persona.offline_sessions[0].memory_privacy_review_completed is expected


def test_current_semantic_withdrawal_clears_raw_channels_before_safety_handling(tmp_path):
    persona = enabled()
    ctx = conversation(persona)
    queue = OfflineSessionQueue(persona, PersonaStore(tmp_path), now=lambda: T0 + timedelta(days=1, hours=2))
    queue.enqueue(ctx)
    assert queue.persona.offline_sessions[0].cognitive_turns
    import hashlib
    text = "不要再提醒我的记性。"
    ctx.working_memory["online_turn_decision"] = {
        "version": "online_turn_decision_v3", "status": "parsed", "turn_index": len(ctx.turns),
        "owner_message_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "cognitive_control": {"version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": text},
    }
    assert apply_cognitive_withdrawal(ctx, queue.persona, "不要再提醒我的记性。", now=T0 + timedelta(days=1, hours=3))
    assert not queue.persona.cognitive_health_consent.has_text_monitoring_consent()
    assert not queue.persona.offline_sessions[0].cognitive_turns
    assert "_cognitive_source_turns" not in ctx.working_memory
    assert not apply_cognitive_withdrawal(ctx, queue.persona, NORMAL)


@pytest.mark.parametrize("mutation", ["different_owner", "simulation"])
def test_withdrawal_helper_respects_owner_and_simulation_boundaries(mutation):
    persona = enabled()
    ctx = conversation(persona)
    if mutation == "different_owner":
        ctx.user_id = "other"
    else:
        ctx.simulation_mode = True
    assert not apply_cognitive_withdrawal(ctx, persona, "不要再提醒我的记性。")
    assert persona.cognitive_health_consent.has_text_monitoring_consent()
