"""C1 memory protocol failures are missing evidence, not zero observations."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from tests.test_cognitive_change_v2 import T0, context_with, enabled


FIELDS = ("episodes", "relationships", "stated_preferences", "cognitive_observations")


class LocalResponse:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return json.dumps(self.payload, ensure_ascii=False)


def complete_payload():
    return {name: [] for name in FIELDS}


def test_enabled_primary_example_is_one_object_with_four_arrays():
    persona = enabled()
    client = LocalResponse(complete_payload())
    extractor = LLMExtractor(client)
    extractor.extract(context_with(persona), persona)
    system = client.calls[0][0]["content"]
    example, _ = json.JSONDecoder().raw_decode(system.split("输出 JSON：", 1)[1])
    assert set(example) == set(FIELDS)
    assert all(isinstance(example[name], list) for name in FIELDS)
    assert set(example["cognitive_observations"][0]) == {
        "domain", "evidence_quote", "turn_index", "functional_impact", "event_kind", "prior_event_id",
    }
    assert "同一个顶层JSON对象必须同时包含" in system
    assert "不能省略字段" in system
    assert "另输出cognitive_observations数组" not in system
    assert extractor.last_status == "ok" and len(client.calls) == 1


def test_disabled_system_keeps_pre_repair_bytes_and_legacy_envelopes():
    # Captured from the existing C1-off system before this protocol repair.
    expected = "ec916ec8599b1bfc4cd7c69ce23e08ea9e80ea2971526bab804462a6553a0c86"
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("我喜欢纸风车。", "我听着。")
    for payload in ({"episodes": [], "relationships": []}, []):
        client = LocalResponse(payload)
        extractor = LLMExtractor(client)
        extractor.extract(ctx, PersonaProfile(user_id="owner"))
        system = client.calls[0][0]["content"]
        assert hashlib.sha256(system.encode()).hexdigest() == expected
        assert "cognitive_observations" not in system
        assert extractor.last_status == "ok"


@pytest.mark.parametrize("missing", FIELDS)
def test_enabled_missing_required_array_is_failure_without_calibration(tmp_path, missing):
    persona = enabled()
    ctx = context_with(persona)
    before = persona.model_dump()
    payload = complete_payload()
    del payload[missing]
    client = LocalResponse(payload)
    extractor = LLMExtractor(client)
    MemoryConsolidator(PersonaStore(tmp_path), extractor).consolidate(persona, ctx, persist=False)
    assert extractor.last_status == "failed"
    assert extractor.last_error_type == "invalid_json_response"
    assert persona.model_dump() == before
    assert "stated_preference_extraction" not in ctx.working_memory
    assert len(client.calls) == 1


@pytest.mark.parametrize("bad_rows", [None, {}, "[]", False, 0, [None], [False], ["text"], [[]], [{}, "text"]])
def test_enabled_wrong_observation_type_is_failure_not_empty_success(tmp_path, bad_rows):
    persona = enabled()
    ctx = context_with(persona)
    payload = complete_payload()
    payload["cognitive_observations"] = bad_rows
    extractor = LLMExtractor(LocalResponse(payload))
    MemoryConsolidator(PersonaStore(tmp_path), extractor).consolidate(persona, ctx, persist=False)
    assert extractor.last_status == "failed"
    assert extractor.last_error_type == "invalid_json_response"
    assert persona.cognitive_trend.session_times == {}
    assert persona.cognitive_observations == []


def test_enabled_legacy_array_is_failure_even_if_empty():
    persona = enabled()
    extractor = LLMExtractor(LocalResponse([]))
    assert extractor.extract(context_with(persona), persona).cognitive_observations == []
    assert extractor.last_status == "failed"


def test_missing_c1_field_with_seven_episodes_rejects_entire_result(tmp_path):
    # Reproduces the failed real receipt's envelope, using authored local facts.
    # No actual model output, cost receipt or experiment record is rewritten.
    persona = enabled()
    ctx = context_with(persona)
    quote = "我喜欢折纸风车。"
    ctx.add_turn(quote, "我听着。")
    payload = {"episodes": [{"summary": "fact", "keywords": [], "referenceable": True,
                             "evidence_quote": quote, "turn_index": 1} for _ in range(7)],
               "relationships": [], "stated_preferences": []}
    extractor = LLMExtractor(LocalResponse(payload))
    store = PersonaStore(tmp_path)
    MemoryConsolidator(store, extractor).consolidate(persona, ctx)
    saved = store.load(persona.user_id)
    assert extractor.last_status == "failed"
    assert saved.episodic_memory == []
    assert saved.cognitive_observations == []
    assert saved.cognitive_trend.session_times == {}
    assert saved.cognitive_trend.calibration_completed_at is None


def test_explicit_empty_array_is_valid_and_records_the_session(tmp_path):
    persona = enabled()
    ctx = context_with(persona)
    ctx.working_memory["_cognitive_review_at"] = (T0 + timedelta(days=1)).isoformat()
    extractor = LLMExtractor(LocalResponse(complete_payload()))
    MemoryConsolidator(PersonaStore(tmp_path), extractor).consolidate(persona, ctx, persist=False)
    assert extractor.last_status == "ok"
    assert persona.cognitive_observations == []
    assert persona.cognitive_trend.session_times == {ctx.session_id: ctx.working_memory["_cognitive_session_at"]}
    assert persona.cognitive_trend.calibration_completed_at is None


def test_complete_observation_fixture_still_requires_literal_evidence(tmp_path):
    persona = enabled()
    ctx = context_with(persona)
    payload = complete_payload()
    row = {"domain": "memory", "evidence_quote": ctx.turns[0].user, "turn_index": 0,
           "functional_impact": False, "event_kind": "new_event", "prior_event_id": None}
    payload["cognitive_observations"] = [row, {**row, "evidence_quote": "我忘记吃饭了。"}]
    extractor = LLMExtractor(LocalResponse(payload))
    MemoryConsolidator(PersonaStore(tmp_path), extractor).consolidate(persona, ctx, persist=False)
    assert extractor.last_status == "ok"
    assert [r.source_quote for r in persona.cognitive_observations] == [ctx.turns[0].user]
    assert persona.episodic_memory == []


def test_offline_failed_protocol_stays_pending_and_does_not_reach_reflection(tmp_path):
    persona = enabled()
    ctx = context_with(persona)
    store = PersonaStore(tmp_path)
    queue = OfflineSessionQueue(persona, store, now=lambda: T0 + timedelta(days=1))
    assert queue.enqueue(ctx)["status"] == "queued"
    client = LocalResponse({"episodes": [], "relationships": [], "stated_preferences": []})
    extractor = LLMExtractor(client)

    def memory(profile, offline):
        MemoryConsolidator(store, extractor).consolidate(profile, offline, persist=False)
        return profile, extractor.last_status, extractor.last_error_type

    def reflection(_profile, _offline):
        pytest.fail("failed C1 extraction must not reach reflection")

    outcome = queue.drain(memory, reflection)
    assert outcome["status"] == "failed" and outcome["completed_jobs"] == 0
    assert len(client.calls) == 1
    saved = store.load(persona.user_id)
    assert len(saved.offline_sessions) == 1 and not saved.offline_sessions[0].memory_completed
    assert saved.offline_sessions[0].attempts == 1
    assert saved.cognitive_trend.session_times == {}
    assert saved.cognitive_observations == []
    assert saved.person_model_state.sessions_completed == 0
