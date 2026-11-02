"""Deterministic protocol checks, not held-out semantic or model-effect scores."""
from copy import deepcopy
import json

import pytest

from aria.engine.memory.diagnosis import (
    EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
    EVIDENCE_REVIEW_PROTOCOL_VERSION,
    PersonaHypothesisUpdater,
    reflection_system_prompt,
)
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import confirmed_prompt_hypotheses
from tests.test_reflection_body_json import EMPTY, Replies, run
from tests.test_reflection_evidence_review import apply, change, context, review, setup, support
from tests.test_reflection_failure_persistence import _engine
from tests.test_reflection_structured_contract_v23 import payload


def test_contract_assigns_event_identity_to_code_and_requires_reliable_current_evidence():
    assert EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION == "2.5"
    assert EVIDENCE_REVIEW_PROTOCOL_VERSION == "2.0"
    prompt = reflection_system_prompt(True)
    assert "你不输出事件分类或旧事件 ID" in prompt
    assert "重复的原话不会刷新支持会话或置信" in prompt
    assert "新原话也不自动证明事件刚刚发生" in prompt
    assert "当期新证据依据时只记录观察，不增加支持分" in prompt
    assert "reviews 可以没有 change" in prompt


def test_current_event_group_reuses_identity_without_extra_support():
    updater, llm, persona = setup()
    ctx = context()
    first = review(ctx)
    supplement = review(ctx, id="r2")
    supplement["evidence"] = deepcopy(first["evidence"][:1])
    apply(updater, llm, persona, ctx, [first, supplement], [change(first)])
    assert updater.last_status == "ok"
    assert not persona.person_hypotheses
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert [row.event_kind for row in persona.evidence_review_ledger] == ["new_event", "same_event"]
    assert updater.last_proposal_results[0]["reason"] == "event_identity_unknown_no_update"
    assert confirmed_prompt_hypotheses(persona) == []
    single_updater, single_llm, single_persona = setup()
    apply(single_updater, single_llm, single_persona, ctx, [first], [change(first)])
    assert not single_persona.person_hypotheses
    assert single_persona.evidence_review_ledger[0].event_id == persona.evidence_review_ledger[0].event_id
    # Repeating the same event quotes cannot acquire a fresh vote in a later
    # session; the model supplies no event classification or identifier.
    repeated_ctx = context(2, words=[turn.user for turn in ctx.turns])
    repeated = review(repeated_ctx)
    apply(updater, llm, persona, repeated_ctx, [repeated], [change(repeated)])
    assert not persona.person_hypotheses
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert persona.evidence_review_ledger[-1].event_kind == "same_event"
    assert updater.last_proposal_results[0]["reason"] == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("reference", [None, "r1", "unknown_old_event"])
def test_removed_model_event_fields_are_rejected_without_losing_independent_review(reference):
    ctx = context()
    first = review(ctx)
    extra = review(ctx, id="r2", event_kind="same_event", prior_event_id=reference)
    value = {"protocol_version": "2.0", "reviews": [first, extra], "changes": [change(first)]}
    client = Replies([value])
    updater = PersonaHypothesisUpdater(
        client, evidence_review_protocol=True, body_json_protocol=True, itemwise_review=True)
    persona = PersonaProfile(user_id="person")
    updater.update(persona, ctx)
    assert updater.last_status == "partial"
    assert updater.last_review_results == [
        {"review_index": 0, "status": "accepted", "reason": "event_identity_unknown_no_update"},
        {"review_index": 1, "status": "rejected", "reason": "invalid_review_shape"},
    ]
    assert len(client.calls) == 1 and updater.last_format_correction_count == 0
    assert not persona.person_hypotheses and len(persona.evidence_review_ledger) == 1
    assert updater.last_proposal_results[0]["reason"] == "event_identity_unknown_no_update"


@pytest.mark.parametrize("relation", ["support", "unknown"])
def test_unknown_review_is_never_turned_into_within_scope_support(relation):
    ctx = context()
    item = review(ctx, applicability="unknown", relation=relation)
    value = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    updater, client, persona = run([value, EMPTY])
    assert len(client.calls) == 1
    assert not persona.person_hypotheses and confirmed_prompt_hypotheses(persona) == []
    if relation == "support":
        assert updater.last_proposal_results[0]["reason"] == "unknown_scope_no_update"
        assert persona.evidence_review_ledger[0].applicability == "unknown"
    else:
        assert updater.last_error_detail == "change_relation_disagrees_with_review"
        assert not persona.evidence_review_ledger


def test_working_exposure_still_requires_existing_evidence_and_confidence_gates():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    assert confirmed_prompt_hypotheses(persona) == []
    support(updater, llm, persona, 2)
    assert hypothesis.status == "working" and len(hypothesis.supporting_events()) == 2
    assert hypothesis.confidence >= .5
    assert confirmed_prompt_hypotheses(persona) == [hypothesis]


def test_evidence_outcome_conflict_is_shape_failure_without_repair_or_semantic_retry():
    value = payload()
    value["changes"][0]["atomic_claim"]["outcome"] = "PRIVATE_UNTRUSTED_GENERATED_VALUE"
    original = deepcopy(value)
    updater, client, persona = run([value, EMPTY])
    assert value == original
    assert updater.last_error_type == "invalid_json_response"  # terminal compatibility
    assert updater.last_error_detail == "invalid_reflection_shape"
    assert updater.last_format_attempts[0]["shape"]["kind"] == "schema"
    assert len(client.calls) == 1 and updater.last_format_correction_count == 0
    assert not persona.person_hypotheses and not persona.evidence_review_ledger
    assert "PRIVATE" not in json.dumps(updater.last_format_attempts)


@pytest.mark.parametrize("value,expected,calls", [
    ('{"reviews":', "invalid_json_response", 2),
    ({"protocol_version": "2.0", "reviews": []}, "invalid_reflection_shape", 2),
    ("An unstructured answer", "invalid_json_response", 1),
])
def test_terminal_detail_reports_actual_failure_category_and_keeps_retry_bound(value, expected, calls):
    updater, client, persona = run([value, value, EMPTY])
    assert updater.last_status == "failed" and updater.last_error_detail == expected
    assert len(client.calls) == calls
    assert not persona.person_hypotheses and not persona.evidence_review_ledger


def test_shape_detail_survives_real_queue_save_without_raw_content(tmp_path):
    value = payload()
    value["changes"][0]["atomic_claim"]["outcome"] = "PRIVATE_UNTRUSTED_GENERATED_VALUE"
    client = Replies([value])
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    path = tmp_path / "profile.json"

    class Store:
        def save(self, persona):
            path.write_text(persona.model_dump_json(), encoding="utf8")

    queue = OfflineSessionQueue(PersonaProfile(user_id="person"), Store())
    queue.enqueue(context(), memory_completed=True)
    queue.persona.offline_sessions[0].memory_privacy_review_completed = True
    queue.drain(lambda person, _: (person, "skipped", None), _engine(updater)._offline_reflection_stage)
    saved = PersonaProfile.model_validate_json(path.read_text(encoding="utf8"))
    audit = saved.offline_session_audit[-1]
    assert audit.outcome == "reflection_failed" and audit.error_code == "invalid_reflection_shape"
    assert audit.memory_completed and not audit.reflection_completed
    assert not saved.person_hypotheses and not saved.evidence_review_ledger
    assert "PRIVATE" not in path.read_text(encoding="utf8")
    assert len(client.calls) == 1
