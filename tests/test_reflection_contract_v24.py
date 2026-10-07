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


def test_contract_distinguishes_current_event_group_from_old_event_reference():
    assert EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION == "2.4"
    assert EVIDENCE_REVIEW_PROTOCOL_VERSION == "2.0"
    prompt = reflection_system_prompt(True)
    assert "同一批 reviews 中首次描述的一个新事件" in prompt
    assert "逐对象复用同一组 event_evidence" in prompt
    assert "same_event 专指请求中已提供 event_id 的历史事件" in prompt
    assert "不把 unknown 改成" in prompt
    assert "证据类型与结果字段无法如实一致，保留观察、changes=[]" in prompt


def test_current_event_group_reuses_identity_without_extra_support():
    updater, llm, persona = setup()
    ctx = context()
    first = review(ctx)
    supplement = review(ctx, id="r2")
    supplement["evidence"] = deepcopy(first["evidence"][:1])
    apply(updater, llm, persona, ctx, [first, supplement], [change(first)])
    assert updater.last_status == "ok"
    hypothesis = persona.person_hypotheses[0]
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert len(hypothesis.supporting_events()) == 1
    assert hypothesis.status == "candidate"
    assert confirmed_prompt_hypotheses(persona) == []
    single_updater, single_llm, single_persona = setup()
    apply(single_updater, single_llm, single_persona, ctx, [first], [change(first)])
    assert hypothesis.confidence == single_persona.person_hypotheses[0].confidence
    before = hypothesis.model_dump()
    # Repeating the very same event quotes cannot acquire a fresh vote even if
    # the generated event_kind wrongly calls it new again in another session.
    repeated_ctx = context(2, words=[turn.user for turn in ctx.turns])
    repeated = review(repeated_ctx, hypothesis)
    apply(updater, llm, persona, repeated_ctx, [repeated], [change(repeated, hypothesis)])
    assert hypothesis.model_dump() == before
    assert updater.last_proposal_results[0]["reason"] == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("reference", [None, "r1", "unknown_old_event"])
def test_missing_or_fabricated_old_event_still_rejects_whole_transaction(reference):
    ctx = context()
    first = review(ctx)
    extra = review(ctx, id="r2", event_kind="same_event", prior_event_id=reference)
    value = {"protocol_version": "2.0", "reviews": [first, extra], "changes": [change(first)]}
    updater, client, persona = run([value, EMPTY])
    assert updater.last_status == "failed"
    assert updater.last_error_detail == (
        "same_event_requires_known_reference" if reference is None else "unknown_prior_event_reference"
    )
    assert len(client.calls) == 1 and updater.last_format_correction_count == 0
    assert not persona.person_hypotheses and not persona.evidence_review_ledger


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
