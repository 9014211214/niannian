"""Item isolation, dependency binding and transaction safety with local replies."""
from copy import deepcopy
import json

import pytest

from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from tests.test_reflection_evidence_review import Replies, context, review, change, support


def setup(*, body=False):
    client = Replies()
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True,
                                      body_json_protocol=body, itemwise_review=True)
    return updater, client, PersonaProfile(user_id="person")


def execute(updater, client, persona, ctx, reviews, changes):
    client.payload = {"protocol_version": "2.0", "reviews": reviews, "changes": changes}
    updater.update(persona, ctx)


def test_removed_event_field_cannot_discard_independent_valid_review():
    updater, client, persona = setup(body=True)
    ctx = context()
    good = review(ctx)
    bad = review(ctx, id="bad", evidence_basis="context_only", event_kind="same_event")
    execute(updater, client, persona, ctx, [bad, good], [change(good)])
    assert updater.last_status == "partial"
    assert not persona.person_hypotheses
    assert len(persona.evidence_review_ledger) == 1
    assert updater.last_review_results == [
        {"review_index": 0, "status": "rejected", "reason": "invalid_review_shape"},
        {"review_index": 1, "status": "accepted", "reason": "event_identity_unknown_no_update"},
    ]
    assert updater.last_proposal_results[0]["status"] == "deferred"
    assert updater.last_proposal_results[0]["reason"] == "event_identity_unknown_no_update"
    assert updater.last_format_correction_count == 0 and len(client.messages) == 1


@pytest.mark.parametrize("invalid", [None, {}, "not-an-object", {"id": "bad"}])
def test_one_bad_shape_isolated_without_repairing_or_format_retry(invalid):
    updater, client, persona = setup(body=True)
    ctx = context()
    good = review(ctx)
    execute(updater, client, persona, ctx, [invalid, good], [change(good)])
    assert updater.last_status == "partial"
    assert not persona.person_hypotheses
    assert len(persona.evidence_review_ledger) == 1
    assert updater.last_proposal_results[0]["reason"] == "event_identity_unknown_no_update"
    assert updater.last_review_results[0]["reason"] == "invalid_review_shape"
    assert len(client.messages) == 1


def test_bad_review_rejects_its_change_without_rebinding_filtered_indices():
    updater, client, persona = setup()
    ctx = context()
    bad = review(ctx, id="bad", event_kind="same_event")
    good = review(ctx, id="good")
    execute(updater, client, persona, ctx, [bad, good], [change(bad), change(good)])
    assert [row["proposal_index"] for row in updater.last_proposal_results] == [0, 1]
    assert [row["status"] for row in updater.last_proposal_results] == ["rejected", "deferred"]
    assert updater.last_proposal_results[0]["reason"] == "bound_review_rejected"
    assert not persona.person_hypotheses
    assert updater.last_proposal_results[1]["reason"] == "event_identity_unknown_no_update"


def test_bad_change_shape_does_not_reject_independent_later_change():
    updater, client, persona = setup(body=True)
    ctx = context()
    first, second = review(ctx, id="a"), review(ctx, id="b")
    malformed = change(first)
    malformed["framework"] = "bogus"
    execute(updater, client, persona, ctx, [first, second], [malformed, change(second)])
    assert [row["status"] for row in updater.last_proposal_results] == ["rejected", "deferred"]
    assert updater.last_proposal_results[0]["reason"] == "invalid_change_shape"
    assert len(persona.evidence_review_ledger) == 2
    assert not persona.person_hypotheses
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert updater.last_proposal_results[1]["reason"] == "same_event_no_fresh_evidence"


def test_duplicate_review_id_rejects_every_ambiguous_participant():
    updater, client, persona = setup()
    ctx = context()
    a, b = review(ctx, id="duplicate"), review(ctx, id="other")
    duplicate_bad = {"id": "duplicate"}
    execute(updater, client, persona, ctx, [a, duplicate_bad, b], [change(a), change(b)])
    assert [row["status"] for row in updater.last_review_results] == ["rejected", "rejected", "accepted"]
    assert all(row["reason"] == "duplicate_or_invalid_review_id" for row in updater.last_review_results[:2])
    assert [row["status"] for row in updater.last_proposal_results] == ["rejected", "deferred"]
    assert len(persona.evidence_review_ledger) == 1


def test_duplicate_change_binding_rejects_all_not_first_wins():
    updater, client, persona = setup()
    ctx = context()
    a, b = review(ctx, id="a"), review(ctx, id="b")
    execute(updater, client, persona, ctx, [a, b], [change(a), {"review_id": "a"}, change(b)])
    assert [row["status"] for row in updater.last_proposal_results] == ["rejected", "rejected", "deferred"]
    assert all(row["reason"] == "one_review_cannot_duplicate_changes" for row in updater.last_proposal_results[:2])


def test_invalid_counter_change_and_repeated_support_cannot_change_existing_claim():
    updater, client, persona = setup()
    hypothesis = support(updater, client, persona)
    before = hypothesis.model_dump()
    ctx = context(2)
    counter = review(ctx, hypothesis, id="counter", relation="counter")
    supporting = review(ctx, hypothesis, id="support")
    invalid = change(counter, hypothesis, confidence=True)
    execute(updater, client, persona, ctx, [counter, supporting], [invalid, change(supporting, hypothesis)])
    assert persona.person_hypotheses[0].model_dump() == before
    assert updater.last_proposal_results[0]["reason"] == "invalid_change_shape"
    assert updater.last_proposal_results[1]["reason"] == "same_claim_counter_takes_priority"
    assert persona.evidence_review_ledger[-2].disposition == "eligible_not_applied"


def test_invalid_counter_review_does_not_suppress_valid_independent_support():
    updater, client, persona = setup()
    hypothesis = support(updater, client, persona)
    before = hypothesis.model_dump()
    ctx = context(2)
    counter = review(ctx, hypothesis, id="counter", relation="counter", event_kind="same_event")
    supporting = review(ctx, hypothesis, id="support")
    execute(updater, client, persona, ctx, [counter, supporting], [change(counter, hypothesis), change(supporting, hypothesis)])
    assert updater.last_proposal_results[0]["reason"] == "bound_review_rejected"
    assert updater.last_proposal_results[1]["status"] == "deferred"
    assert updater.last_proposal_results[1]["reason"] == "event_identity_unknown_no_update"
    assert persona.person_hypotheses[0].model_dump() == before


def test_same_event_never_reclassified_as_new_or_counted_twice():
    updater, client, persona = setup()
    hypothesis = support(updater, client, persona)
    before = hypothesis.model_dump()
    ctx = context(2, words=[turn.user for turn in context().turns])
    old = persona.evidence_review_ledger[0]
    same = review(ctx, hypothesis)
    execute(updater, client, persona, ctx, [same], [change(same, hypothesis)])
    assert persona.person_hypotheses[0].model_dump() == before
    assert updater.last_proposal_results[0]["reason"] == "same_event_no_fresh_evidence"
    assert persona.evidence_review_ledger[-1].event_id == old.event_id


def test_multiple_unqualified_changes_never_bypass_evidence_gate():
    updater, client, persona = setup()
    ctx = context()
    a, b = review(ctx, id="a"), review(ctx, id="b")
    execute(updater, client, persona, ctx, [a, b], [change(a), change(b)])
    assert not persona.person_hypotheses
    assert updater.last_application_summary["applied_count"] == 0
    assert updater.last_application_summary["deferred_count"] == 2
    assert [row["reason"] for row in updater.last_proposal_results] == [
        "event_identity_unknown_no_update", "same_event_no_fresh_evidence"]


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(protocol_version=2.0),
    lambda p: p.update(extra="not_allowed"),
    lambda p: p.update(reviews="not_an_array"),
    lambda p: p.update(reviews=p["reviews"] * 7),
    lambda p: p.update(changes=p["changes"] * 4),
])
def test_invalid_top_level_rejects_everything(mutate):
    updater, client, persona = setup()
    ctx = context()
    item = review(ctx)
    client.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    mutate(client.payload)
    before = persona.model_dump()
    updater.update(persona, ctx)
    assert updater.last_status == "failed"
    assert persona.model_dump() == before


def test_duplicate_json_key_is_not_silently_overwritten():
    class Raw:
        def complete(self, _):
            return '{"protocol_version":"2.0","reviews":[],"reviews":[],"changes":[]}'
    updater = PersonaHypothesisUpdater(Raw(), evidence_review_protocol=True, itemwise_review=True)
    persona = PersonaProfile(user_id="person")
    updater.update(persona, context())
    assert updater.last_status == "failed"
    assert updater.last_error_detail == "duplicate_reflection_key"


def test_save_failure_has_no_committed_changes_or_successful_audit_receipt():
    updater, client, persona = setup()
    ctx = context()
    item = review(ctx)
    client.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    class Store:
        reject_candidate = False
        def save(self, candidate):
            if self.reject_candidate and candidate.evidence_review_ledger:
                raise OSError("fixture disk failure")
    store = Store()
    queue = OfflineSessionQueue(persona, store)
    queue.enqueue(ctx, memory_completed=True)
    queue.persona.offline_sessions[0].memory_privacy_review_completed = True
    store.reject_candidate = True
    def reflect(candidate, source):
        updater.update(candidate, source)
        source.working_memory["_reflection_itemwise_audit"] = updater.stage_audit_summary()
        return candidate, "ok", None
    result = queue.drain(lambda p, c: (p, "skipped", None), reflect)
    assert result["status"] == "failed" and result["error_code"] == "offline_store_failed"
    assert not queue.persona.person_hypotheses and not queue.persona.evidence_review_ledger
    assert not queue.persona.offline_session_audit
    assert not queue.persona.offline_sessions[0].reflection_completed
    assert queue.persona.offline_sessions[0].reflection_result is None


def test_partial_processing_counts_survive_serialization_without_source_text():
    updater, client, persona = setup()
    ctx = context()
    good = review(ctx)
    bad = review(ctx, id="bad", event_kind="same_event")
    client.payload = {"protocol_version": "2.0", "reviews": [bad, good], "changes": [change(good)]}
    queue = OfflineSessionQueue(persona)
    queue.enqueue(ctx, memory_completed=True)
    queue.persona.offline_sessions[0].memory_privacy_review_completed = True
    def reflect(candidate, source):
        updater.update(candidate, source)
        source.working_memory["_reflection_itemwise_audit"] = updater.stage_audit_summary()
        return candidate, "ok", None
    result = queue.drain(lambda p, c: (p, "skipped", None), reflect)
    assert result["status"] == "completed"
    restored = PersonaProfile.model_validate_json(queue.persona.model_dump_json())
    audit = restored.offline_session_audit[0].reflection_result
    assert audit.status == "partial" and audit.accepted_review_count == 1 and audit.rejected_review_count == 1
    assert audit.applied_proposal_count == 0 and audit.deferred_proposal_count == 1
    assert not restored.person_hypotheses
    assert ctx.turns[0].user not in audit.model_dump_json()
