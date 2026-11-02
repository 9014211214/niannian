"""Complete-prefix recovery is a software boundary, not a semantic score."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator, _memory_complete_prefix
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from tests.preference_fixtures import communication_review


FIELDS = ("stated_preferences", "cognitive_privacy", "cognitive_observations", "episodes", "relationships")


class Reply:
    max_completion_tokens = 4000

    def __init__(self, payload, finish="stop"):
        self.payload, self.last_finish_reason = payload, finish
        self.messages = []

    def complete(self, messages):
        self.messages.append(deepcopy(messages))
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)


def setup(payload=None, *, finish="stop", enabled=False):
    persona = PersonaProfile(user_id="prefix_owner")
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="prefix_session")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v5_bound"
    ctx.add_turn("谈起修理的时候，请等我说完再问。", "我听着。")
    ctx.add_turn("今天我在院子里修好了一只纸风车。", "嗯。")
    client = Reply(payload if payload is not None else empty(), finish)
    return persona, ctx, client, LLMExtractor(client, prefix_recovery=True)


def empty():
    return {name: [] for name in FIELDS}


def preference(ctx):
    return {"do": "谈起修理时等我说完再问", "scope": ctx.turns[0].user,
            "quotes": [{"turn_index": 0, "text": ctx.turns[0].user}],
            "confirmation": None, "supersedes": None, "semantic_review": communication_review()}


def prefix(payload, completed=3):
    return json.dumps({key: payload[key] for key in FIELDS[:completed]}, ensure_ascii=False)[:-1]


def test_candidate_prompt_has_priority_order_and_explicit_disabled_observations():
    persona, ctx, client, extractor = setup()
    extractor.extract(ctx, persona)
    prompt = client.messages[0][0]["content"]
    example, _ = json.JSONDecoder().raw_decode(prompt.split("输出 JSON：", 1)[1])
    assert tuple(example) == FIELDS
    assert example["cognitive_observations"] == []
    assert "每项新增关系事实仍必须附逐字出处" in prompt
    assert extractor.last_status == "ok"
    assert ctx.working_memory["_preference_scope_selection"]["protocol"] == "preference_scope_full_source_v1"


def test_completed_privacy_prefix_preserves_proposal_without_inventing_missing_tail():
    persona, ctx, client, extractor = setup(finish="length")
    payload = empty()
    payload["stated_preferences"] = [preference(ctx)]
    client.payload = prefix(payload) + ',"episodes":[{"summary":"unfinished'
    MemoryConsolidator(extractor=extractor).consolidate(persona, ctx, persist=False)
    assert len(persona.stated_preferences) == 1
    assert persona.stated_preferences[0].status == "proposed"
    assert not persona.episodic_memory and not persona.relationship_context
    assert extractor.last_status == "partial"
    assert extractor.last_error_type == "truncated_response"
    assert extractor.last_prefix_summary["missing_fields"] == ["episodes", "relationships"]
    assert extractor.last_prefix_summary["completed_fields"] == list(FIELDS[:3])
    assert ctx.working_memory["_memory_privacy_review_completed"] is True
    assert len(client.messages) == 1


@pytest.mark.parametrize("completed", [1, 2])
def test_either_unfinished_privacy_array_prevents_all_proposals(completed):
    persona, ctx, client, extractor = setup(finish="length")
    payload = empty()
    payload["stated_preferences"] = [preference(ctx)]
    client.payload = prefix(payload, completed)
    before = persona.model_dump()
    MemoryConsolidator(extractor=extractor).consolidate(persona, ctx, persist=False)
    assert persona.model_dump() == before
    assert extractor.last_status == "failed"
    assert not ctx.working_memory.get("_memory_privacy_review_completed")
    assert ctx.working_memory["_memory_prefix_recovery"]["truncated"] is True


@pytest.mark.parametrize("tail", [
    ',"episodes":[{"summary":"bad\\q',
    ',"episodes":[{"a":1,"a":"unfinished',
    ',"episodes":[{"value":01',
    ',"episodes":[trueX',
    ',"episodes":[,',
    ',"episodes":[{"a":1,}',
    ',"episodes":[],"episodes":',
    ',"unexpected":',
    '}trailing',
])
def test_invalid_tail_is_not_repaired_or_ignored(tail):
    persona, ctx, client, extractor = setup(finish="length")
    client.payload = prefix(empty()) + tail
    extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert not ctx.working_memory.get("_memory_privacy_review_completed")


@pytest.mark.parametrize("tail", [',"episodes":[{"a":"partial', ',"episodes":[{"a":-1e+',
                                  ',"episodes":[{"a":"\\u12', ',"episodes":[{"a":tru'])
def test_valid_unfinished_tokens_do_not_become_complete_array_members(tail):
    assert _memory_complete_prefix(prefix(empty()) + tail) == {key: [] for key in FIELDS[:3]}


@pytest.mark.parametrize("finish", ["stop", None, "content_filter"])
def test_partial_recovery_requires_observed_length_termination(finish):
    persona, ctx, client, extractor = setup(prefix(empty()), finish=finish)
    extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert not ctx.working_memory.get("_memory_privacy_review_completed")


def test_length_report_is_preserved_even_if_body_happens_to_be_complete():
    persona, ctx, client, extractor = setup(empty(), finish="length")
    extractor.extract(ctx, persona)
    assert extractor.last_status == "partial"
    assert extractor.last_prefix_summary["missing_fields"] == []
    assert extractor.last_prefix_summary["truncated"] is True


def test_privacy_prefix_masks_proposed_preference_before_merging():
    persona, ctx, client, extractor = setup(finish="length")
    payload = empty()
    payload["stated_preferences"] = [preference(ctx)]
    payload["cognitive_privacy"] = [{"turn_index": 0, "evidence_quote": ctx.turns[0].user}]
    client.payload = prefix(payload)
    MemoryConsolidator(extractor=extractor).consolidate(persona, ctx, persist=False)
    assert not persona.stated_preferences
    assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]


def test_completed_but_invalid_privacy_source_cannot_release_prefix():
    persona, ctx, client, extractor = setup(finish="length")
    payload = empty()
    payload["cognitive_privacy"] = [{"turn_index": 0, "evidence_quote": "invented"}]
    client.payload = prefix(payload)
    extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert extractor.last_prefix_summary["privacy_sources_validated"] is False


def test_candidate_does_not_repair_episode_quote_punctuation():
    persona, ctx, client, extractor = setup()
    payload = empty()
    payload["episodes"] = [{"evidence_quote": ctx.turns[1].user.replace("。", "!"),
                            "turn_index": 1, "referenceable": True}]
    client.payload = payload
    assert extractor.extract(ctx, persona).new_episodes == []


def test_eleven_turn_bounded_fixture_output_is_under_sixty_percent_of_4000_byte_upper_bound():
    # UTF-8 bytes are a deliberately conservative tokenizer-independent bound
    # for this small fixture only; no guarantee for every 11-turn conversation.
    persona, ctx, client, extractor = setup()
    for index in range(9):
        ctx.add_turn(f"第{index + 3}轮，我还在听你说。", "我在。")
    payload = empty()
    payload["stated_preferences"] = [preference(ctx)]
    payload["episodes"] = [{"evidence_quote": ctx.turns[1].user,
                            "turn_index": 1, "referenceable": True, "keywords": ["风车"]}]
    client.payload = payload
    extractor.extract(ctx, persona)
    assert len(ctx.turns) == 11
    assert all(f"[turn {index}]" in client.messages[0][1]["content"] for index in range(11))
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= .6 * client.max_completion_tokens
    assert extractor.last_status == "ok"


def test_partial_memory_audit_survives_queue_completion_and_reload():
    persona, ctx, client, extractor = setup(finish="length")
    payload = empty()
    payload["stated_preferences"] = [preference(ctx)]
    client.payload = prefix(payload)
    queue = OfflineSessionQueue(persona)
    assert queue.enqueue(ctx)["status"] == "queued"
    def memory(candidate, source):
        MemoryConsolidator(extractor=extractor).consolidate(candidate, source, persist=False)
        return candidate, "ok", None
    result = queue.drain(memory, lambda candidate, source: (candidate, "skipped", None))
    assert result["status"] == "completed"
    restored = PersonaProfile.model_validate_json(queue.persona.model_dump_json())
    audit = restored.offline_session_audit[0].memory_result
    assert audit.status == "partial" and audit.truncated is True
    assert audit.missing_fields == ["episodes", "relationships"]
    assert len(restored.stated_preferences) == 1
    assert ctx.turns[0].user not in audit.model_dump_json()


def test_unscreened_prefix_failure_persists_rejection_not_success():
    persona, ctx, client, extractor = setup(prefix(empty(), 2), finish="length")
    queue = OfflineSessionQueue(persona)
    queue.enqueue(ctx)
    def memory(candidate, source):
        extractor.extract(source, candidate)
        return candidate, "sealed_failed", "memory_extraction_failed"
    result = queue.drain(memory, lambda candidate, source: (_ for _ in ()).throw(AssertionError("must not reflect")))
    assert result["status"] == "completed"
    audit = queue.persona.offline_session_audit[0]
    assert audit.memory_result.status == "rejected" and audit.memory_result.truncated
    assert not audit.memory_completed and not audit.reflection_completed
    assert audit.outcome == "memory_failed"


def test_precompleted_memory_keeps_partial_audit_when_enqueued():
    persona, ctx, client, extractor = setup(prefix(empty()), finish="length")
    extractor.extract(ctx, persona)
    queue = OfflineSessionQueue(persona)
    assert queue.enqueue(ctx, memory_completed=True)["status"] == "queued"
    assert queue.persona.offline_sessions[0].memory_result.status == "partial"


def test_observations_also_mask_ordinary_memory_before_partial_commit():
    from tests.test_cognitive_change_v2 import enabled, context_with
    persona = enabled()
    ctx = context_with(persona)
    ctx.working_memory["_sensitive_memory_excluded_turns"] = []
    payload = empty()
    payload["cognitive_observations"] = [{
        "domain": "memory", "evidence_quote": ctx.turns[0].user, "turn_index": 0,
        "functional_impact": False, "event_kind": "new_event", "prior_event_id": None,
    }]
    payload["episodes"] = [{"evidence_quote": ctx.turns[0].user, "turn_index": 0, "referenceable": True}]
    client = Reply(prefix(payload, 4), "length")
    extractor = LLMExtractor(client, prefix_recovery=True)
    result = extractor.extract(ctx, persona)
    assert extractor.last_status == "partial"
    assert result.new_episodes == [] and len(result.cognitive_observations) == 1
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


def test_invalid_completed_observation_source_prevents_ordinary_commit():
    from tests.test_cognitive_change_v2 import enabled, context_with
    persona = enabled()
    ctx = context_with(persona)
    payload = empty()
    payload["cognitive_observations"] = [{
        "domain": "memory", "evidence_quote": "不是本人原话", "turn_index": 0,
        "functional_impact": False, "event_kind": "new_event", "prior_event_id": None,
    }]
    extractor = LLMExtractor(Reply(prefix(payload), "length"), prefix_recovery=True)
    extractor.extract(ctx, persona)
    assert extractor.last_status == "failed"
    assert not ctx.working_memory.get("_memory_privacy_review_completed")
