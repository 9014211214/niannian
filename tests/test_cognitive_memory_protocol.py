"""C1 memory protocol failures are missing evidence, not zero observations."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from tests.test_cognitive_change_v2 import T0, context_with, enabled


FIELDS = ("episodes", "relationships", "stated_preferences", "cognitive_privacy", "cognitive_observations")


class LocalResponse:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return json.dumps(self.payload, ensure_ascii=False)


def complete_payload():
    return {name: [] for name in FIELDS}


def test_enabled_primary_example_is_one_object_with_five_arrays():
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


def test_disabled_system_requires_privacy_classification_and_current_r1_contract():
    # The old byte hashes belong to archived pre-R1-A snapshots. New extraction
    # permits faithful paraphrases and cannot activate its own confirmation.
    addition = (
        "[memory_json_escaping_v2]只输出一个完整、合法的 JSON 对象，不加 Markdown 围栏或说明文字。"
        "所有字符串中的双引号、反斜杠、换行等字符必须按 JSON 语法转义；"
        "evidence_quote 也必须转义，但 JSON 解码后的引文必须保持用户原话逐字不变。"
        "不要为了凑齐结构而编造事实或偏好；无适用内容的数组使用空数组。"
    )
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("我喜欢纸风车。", "我听着。")
    for payload in ({"episodes": [], "relationships": [], "cognitive_privacy": []},
                    {"episodes": [], "relationships": []}, []):
        client = LocalResponse(payload)
        extractor = LLMExtractor(client)
        extractor.extract(ctx, PersonaProfile(user_id="owner"))
        system = client.calls[0][0]["content"]
        assert system.count(addition) == 1
        assert "可忠实转述quotes" in system
        assert "必须逐字引用quotes中的连续原话，不改写" not in system
        assert "quotes保留该轮完整本人原话" in system
        example, _ = json.JSONDecoder().raw_decode(system.split("输出 JSON：", 1)[1])
        assert example["stated_preferences"][0]["confirmation"] is None
        assert "cognitive_observations" not in system
        assert extractor.last_status == ("ok" if isinstance(payload, dict) and "cognitive_privacy" in payload else "failed")


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
