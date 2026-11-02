"""Code-owned event identity and conservative evidence timing; no model calls."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json

import pytest

from aria.engine.memory.diagnosis import (
    PersonaHypothesisUpdater, _event_signature, reflection_system_prompt,
)
from aria.engine.memory.reflection_contract import (
    reflection_response_schema, validate_reflection_response_shape,
)
from aria.engine.model.persona import PersonHypothesis, PersonHypothesisEvidence, PersonaProfile
from aria.engine.policy.reflection_compact import (
    compact_reflection_schema, compact_reflection_system_prompt,
    decode_compact_reflection, encode_compact_reflection,
)
from aria.engine.policy.reflection_json import reflection_wire_schema
from tests.test_reflection_evidence_review import Replies, change, context, review


def current_review(ctx, hypothesis=None, **overrides):
    item = review(ctx, hypothesis, **overrides)
    item.pop("event_kind", None)
    item.pop("prior_event_id", None)
    return item


def run(persona, ctx, reviews, changes=()):
    client = Replies()
    client.payload = {"protocol_version": "2.0", "reviews": reviews, "changes": list(changes)}
    updater = PersonaHypothesisUpdater(
        client, evidence_review_protocol=True, itemwise_review=True,
        now_provider=lambda: datetime(2026, 10, 20),
    )
    updater.update(persona, ctx)
    return updater


@pytest.mark.parametrize("schema_factory", [
    reflection_response_schema, reflection_wire_schema, compact_reflection_schema,
])
def test_generation_schema_has_no_model_owned_event_identity(schema_factory):
    schema = schema_factory()
    properties = schema["$defs"]["_Review"]["properties"]
    assert "event_kind" not in properties and "prior_event_id" not in properties
    assert set(schema["$defs"]["_Review"]["required"]) == set(properties)
    assert schema["$defs"]["_Review"]["additionalProperties"] is False


def test_prompts_and_compact_slots_match_code_owned_event_contract():
    prompt = reflection_system_prompt(True)
    assert "prompt contract 2.5" in prompt
    assert "不输出事件分类或旧事件 ID" in prompt
    assert "event_kind" not in prompt and "prior_event_id" not in prompt
    assert "以下八处字段" in compact_reflection_system_prompt()
    ctx = context()
    item = current_review(ctx)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    validate_reflection_response_shape(payload)
    encoded = encode_compact_reflection(payload)
    assert json.loads(decode_compact_reflection(json.dumps(encoded))) == payload


@pytest.mark.parametrize("key,value", [
    ("event_kind", "new_event"), ("prior_event_id", None),
])
def test_obsolete_model_event_fields_are_rejected_without_discarding_valid_review(key, value):
    ctx = context()
    valid = current_review(ctx)
    invalid = current_review(ctx, id="bad")
    invalid[key] = value
    persona = PersonaProfile(user_id="person")
    updater = run(persona, ctx, [invalid, valid], [change(invalid), change(valid)])
    assert updater.last_status == "partial"
    assert [row["status"] for row in updater.last_review_results] == ["rejected", "accepted"]
    assert updater.last_review_results[0]["reason"] == "invalid_review_shape"
    assert updater.last_proposal_results[0]["reason"] == "bound_review_rejected"
    assert len(persona.evidence_review_ledger) == 1
    assert not persona.person_hypotheses


def test_signature_normalises_whitespace_order_and_duplicate_quotes_only():
    assert _event_signature(["  one \t two ", "three\nfour"]) == _event_signature(
        ["three four", "one two", "one two"]
    )
    assert _event_signature(["原话甲"]) != _event_signature(["原话乙"])
    assert _event_signature(["ONE two"]) != _event_signature(["one two"])


@pytest.mark.parametrize("basis", ["stated_preference", "observed_response", "repeated_pattern"])
def test_unmatched_signature_is_observation_not_proof_of_current_event(basis):
    persona = PersonaProfile(user_id="person")
    ctx = context()
    item = current_review(ctx, evidence_basis=basis)
    proposal = change(item)
    if basis == "observed_response":
        proposal["atomic_claim"]["outcome"] = "本人描述更愿意继续尝试"
    updater = run(persona, ctx, [item], [proposal])
    assert updater.last_status == "ok"
    saved, = persona.evidence_review_ledger
    assert saved.event_kind == "new_event" and saved.prior_event_id is None
    assert saved.event_id.startswith("evt_")
    assert saved.disposition == "event_identity_unknown_no_update"
    assert saved.independent_event_count == 0
    assert not persona.person_hypotheses
    assert updater.last_belief_change_results == []
    assert updater.last_proposal_results[0]["status"] == "deferred"


def test_distinct_sessions_and_quotes_have_distinct_identities_without_freshness_credit():
    persona = PersonaProfile(user_id="person")
    for index in (1, 2):
        ctx = context(index)
        item = current_review(ctx, evidence_basis="repeated_pattern")
        run(persona, ctx, [item], [change(item)])
    assert len({row.session_id for row in persona.evidence_review_ledger}) == 2
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 2
    assert all(row.independent_event_count == 0 for row in persona.evidence_review_ledger)
    assert not persona.person_hypotheses


def test_repeated_normalised_quote_uses_prior_id_without_new_evidence():
    persona = PersonaProfile(user_id="person")
    first = context(1, words=["one   event", "我更喜欢一步一步来。", "这只是交流节奏。"])
    item = current_review(first)
    run(persona, first, [item], [change(item)])
    original = persona.evidence_review_ledger[0]
    second = context(2, words=["one \t event", "我更喜欢一步一步来。", "这只是交流节奏。"])
    repeated = current_review(second)
    run(persona, second, [repeated], [change(repeated)])
    saved = persona.evidence_review_ledger[-1]
    assert saved.event_id == saved.prior_event_id == original.event_id
    assert saved.event_kind == "same_event"
    assert saved.disposition == "same_event_no_fresh_evidence"
    assert saved.independent_event_count == 0
    assert not persona.person_hypotheses


def test_same_batch_quotes_share_code_identity_without_duplicate_evidence():
    persona = PersonaProfile(user_id="person")
    ctx = context()
    first, second = current_review(ctx), current_review(ctx, id="r2")
    updater = run(persona, ctx, [first, second], [change(first), change(second)])
    assert updater.last_status == "ok"
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert [row.event_kind for row in persona.evidence_review_ledger] == ["new_event", "same_event"]
    assert updater.last_application_summary["applied_count"] == 0


def seeded_hypothesis(*, sessions=2, rows=3, days=1, confidence=.6):
    """Pre-existing software fixture; never presented as a learned model result."""
    hypothesis = PersonHypothesis(
        id="prior", framework="response", dimension="low_pressure",
        claim="在不赶时间的操作教学中可能偏好分步听", scope="不赶时间的操作教学",
        confidence=confidence, status="candidate",
        first_observed_at="2026-01-01T00:00:00",
    )
    for index in range(rows):
        session = index % sessions
        hypothesis.evidence.append(PersonHypothesisEvidence(
            at=f"2026-01-{index % days + 1:02}T00:00:00",
            session_id=f"old_{session}", relation="support", summary=f"旧证据{index}",
            turn_indices=[index], source="conversation_quote", event_id=f"old_event_{session}",
            event_kind="retrospective",
        ))
    return hypothesis


@pytest.mark.parametrize("relation", ["support", "counter"])
def test_unknown_narration_cannot_support_but_grounded_counter_can_correct(relation):
    old = seeded_hypothesis()
    persona = PersonaProfile(user_id="person", person_hypotheses=[old])
    before = old.model_dump()
    ctx = context()
    item = current_review(ctx, old, relation=relation)
    updater = run(persona, ctx, [item], [change(item, old, confidence=1)])
    assert updater.last_status == "ok"
    if relation == "support":
        assert old.model_dump() == before
        assert updater.last_belief_change_results == []
    else:
        assert old.confidence < before["confidence"]
        assert len(updater.last_belief_change_results) == 1
        assert old.claim == before["claim"] and old.scope == before["scope"]
    assert persona.evidence_review_ledger[-1].independent_event_count == 2


@pytest.mark.parametrize("sessions,rows,days,confidence,expected", [
    (1, 3, 1, .6, "candidate"), (2, 2, 1, .6, "candidate"),
    (2, 3, 1, .49, "candidate"), (2, 3, 1, .5, "working"),
    (4, 7, 3, .8, "working"), (5, 6, 3, .8, "working"),
    (5, 7, 2, .8, "working"), (5, 7, 3, .74, "working"),
    (5, 7, 3, .75, "stable"), (2, 3, 1, .15, "retired"),
])
def test_existing_person_thresholds_are_unchanged(sessions, rows, days, confidence, expected):
    old = seeded_hypothesis(sessions=sessions, rows=rows, days=days, confidence=confidence)
    PersonaHypothesisUpdater._refresh_status(old)
    assert old.status == expected


def test_fabricated_quote_rejects_only_its_review_and_bound_change():
    persona = PersonaProfile(user_id="person")
    ctx = context()
    invalid = current_review(ctx, id="bad")
    invalid["event_evidence"][0]["evidence_quote"] = "用户从没说过的事件"
    valid = current_review(ctx)
    updater = run(persona, ctx, [invalid, valid], [change(invalid), change(valid)])
    assert updater.last_status == "partial"
    assert updater.last_review_results[0]["reason"] == "review_quote_not_grounded_in_current_user"
    assert updater.last_review_results[1]["status"] == "accepted"
    assert len(persona.evidence_review_ledger) == 1


def test_retained_signature_works_after_review_ledger_has_aged_out():
    ctx = context()
    old = seeded_hypothesis()
    signature = _event_signature([ctx.turns[0].user])
    old.evidence[0].event_signature = signature
    persona = PersonaProfile(user_id="person", person_hypotheses=[old])
    before = deepcopy(old.model_dump())
    item = current_review(ctx, old)
    run(persona, ctx, [item], [change(item, old)])
    assert old.model_dump() == before
    assert persona.evidence_review_ledger[-1].event_id == old.evidence[0].event_id
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"


def interaction_review(ctx, hypothesis=None):
    """Prescribed semantics and delivery receipts; engineering evidence only."""
    item = current_review(ctx, hypothesis, evidence_basis="observed_response")
    item["event_evidence"] = [deepcopy(item["evidence"][0])]
    proposal = change(item, hypothesis)
    proposal["atomic_claim"]["outcome"] = "本人愿意继续尝试"
    ctx.working_memory["_person_response_deliveries"] = {
        str(index): hashlib.sha256(turn.agent.encode("utf-8")).hexdigest()
        for index, turn in enumerate(ctx.turns)
    }
    return item, proposal


@pytest.mark.parametrize("boundary", ["missing", "changed", "protected_predecessor", "unbound_event", "legacy_partial"])
def test_observed_response_label_cannot_replace_actual_unprotected_delivery(boundary):
    persona = PersonaProfile(user_id="person")
    ctx = context()
    item, proposal = interaction_review(ctx)
    if boundary == "missing":
        ctx.working_memory.pop("_person_response_deliveries")
    elif boundary == "changed":
        ctx.turns[0].agent = "代码替换后的内容"
    elif boundary == "protected_predecessor":
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    elif boundary == "legacy_partial":
        ctx.working_memory["_offline_source_completeness"] = "legacy_tail_unknown_indices"
    else:
        item["event_evidence"] = [{"turn_index": 0, "evidence_quote": ctx.turns[0].user}]
    updater = run(persona, ctx, [item], [proposal])
    assert updater.last_status == "ok" and not persona.person_hypotheses
    assert persona.evidence_review_ledger[-1].disposition == "event_identity_unknown_no_update"


def test_original_source_deduplicates_different_substrings_and_valid_cross_session_support():
    persona = PersonaProfile(user_id="person")
    first = context(1, words=["练习画圆。", "照着你刚才的一步画出了圆，我愿意再试。", "这种节奏让我愿意接着练。"])
    item, proposal = interaction_review(first)
    item["event_evidence"][0]["evidence_quote"] = "画出了圆"
    run(persona, first, [item], [proposal])
    hypothesis, = persona.person_hypotheses
    assert hypothesis.status == "candidate"
    assert hypothesis.confidence == pytest.approx(.48)
    before = hypothesis.model_dump()
    repeated = context(2, words=[turn.user for turn in first.turns])
    item, proposal = interaction_review(repeated, hypothesis)
    item["event_evidence"][0]["evidence_quote"] = "我愿意再试"
    run(persona, repeated, [item], [proposal])
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    second = context(3, words=["练习折纸。", "你刚才分开说，我照着折好第一角了，还想接着折。", "这次我也能跟上这种节奏。"])
    item, proposal = interaction_review(second, hypothesis)
    run(persona, second, [item], [proposal])
    assert hypothesis.status == "working" and hypothesis.confidence == pytest.approx(.656)
    assert len(hypothesis.supporting_sessions()) == len(hypothesis.supporting_events()) == 2


def test_changed_wording_of_unknown_past_story_never_becomes_independent_support():
    persona = PersonaProfile(user_id="person")
    for index, narration in enumerate(("老同事当年按步骤教我，我很喜欢。", "那回同事把步骤拆开，我觉得好学。")):
        ctx = context(index, words=["继续讲以前的事情。", narration, "都是同一回的事情。"])
        item, proposal = interaction_review(ctx)
        # The semantic author identifies a narrated pattern, not a response to
        # the actual preceding Aria text; code must not override that label.
        item["evidence_basis"] = proposal["atomic_claim"]["evidence_basis"] = "repeated_pattern"
        run(persona, ctx, [item], [proposal])
    assert not persona.person_hypotheses
    assert all(row.disposition == "event_identity_unknown_no_update" for row in persona.evidence_review_ledger)


def test_same_batch_changed_excerpt_has_one_original_source_event():
    persona = PersonaProfile(user_id="person")
    ctx = context(1, words=["练习画圆。", "照着你刚才的一步画出了圆，我愿意再试。", "这种节奏让我愿意接着练。"])
    first, proposal = interaction_review(ctx)
    first["event_evidence"][0]["evidence_quote"] = "画出了圆"
    second = deepcopy(first)
    second["id"] = "r2"
    second["event_evidence"][0]["evidence_quote"] = "我愿意再试"
    other = deepcopy(proposal)
    other["review_id"] = second["id"]
    updater = run(persona, ctx, [first, second], [proposal, other])
    assert updater.last_status == "ok"
    assert len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].confidence == pytest.approx(.48)
    assert len({row.event_id for row in persona.evidence_review_ledger}) == 1
    assert [row.disposition for row in persona.evidence_review_ledger] == ["applied", "same_event_no_fresh_evidence"]


@pytest.mark.parametrize("support_first", [True, False])
@pytest.mark.parametrize("itemwise", [True, False])
def test_same_new_source_counter_has_priority_without_review_order_dependency(support_first, itemwise):
    old = seeded_hypothesis(confidence=.6)
    persona = PersonaProfile(user_id="person", person_hypotheses=[old])
    ctx = context(9, words=["本次操作练习。", "照着你说的做完了，但等待节奏不适合我。", "重新考虑之前那个猜测吧。"])
    supporting, support_change = interaction_review(ctx, old)
    supporting["id"] = support_change["review_id"] = "supporting"
    counter = deepcopy(supporting)
    counter.update(id="counter", relation="counter")
    counter_change = change(counter, old)
    counter_change["atomic_claim"]["outcome"] = "本人反馈这次节奏不适合"
    reviews = [supporting, counter] if support_first else [counter, supporting]
    changes = [support_change, counter_change] if support_first else [counter_change, support_change]
    client = Replies()
    client.payload = {"protocol_version": "2.0", "reviews": reviews, "changes": changes}
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, itemwise_review=itemwise)
    updater.update(persona, ctx)
    assert updater.last_status == "ok"
    assert old.confidence == pytest.approx(.415)
    assert len(old.supporting_events()) == 2
    by_relation = {row.relation: row for row in persona.evidence_review_ledger}
    assert by_relation["support"].disposition == "same_claim_counter_takes_priority"
    assert by_relation["counter"].disposition == "applied"
    assert by_relation["support"].independent_event_count == 2
    assert len({row.event_id for row in by_relation.values()}) == 1
    # The same source becomes historical only AFTER this transaction. Neither
    # reversing the next replay nor another review id earns a second debit.
    before = old.model_dump()
    repeated = context(10, words=[turn.user for turn in ctx.turns])
    again, _ = interaction_review(repeated, old)
    again.update(id="another_counter", relation="counter")
    proposal = change(again, old)
    proposal["atomic_claim"]["outcome"] = "本人反馈这次节奏不适合"
    run(persona, repeated, [again], [proposal])
    assert old.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("relation", ["counter", "retire"])
@pytest.mark.parametrize("itemwise", [True, False])
def test_new_quote_corrects_old_support_once_without_fabricating_independent_event(relation, itemwise):
    persona = PersonaProfile(user_id="person")
    first = context(31, words=["练习画圆。", "照着这一步画好了圆，我继续往下做了。", "我完成了你说的下一步。"])
    supporting, proposal = interaction_review(first)
    run(persona, first, [supporting], [proposal])
    hypothesis, = persona.person_hypotheses
    original_events = hypothesis.supporting_events()
    event_quote = supporting["event_evidence"][0]["evidence_quote"]

    def correct(index, suffix):
        ctx = context(index, words=["纠正先前的理解。", f"我说过“{event_quote}”，但那并不表示喜欢等待。", suffix])
        counter = current_review(ctx, hypothesis, relation="counter")
        counter["event_evidence"] = [{"turn_index": 1, "evidence_quote": event_quote}]
        client = Replies()
        client.payload = {"protocol_version": "2.0", "reviews": [counter],
                          "changes": [change(counter, hypothesis, relation=relation)]}
        updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, itemwise_review=itemwise)
        updater.update(persona, ctx)
        return updater

    updater = correct(32, "其实等待让我难以了解完整过程，请改掉那个猜测。")
    assert updater.last_status == "ok"
    assert hypothesis.confidence == pytest.approx(0 if relation == "retire" else .295)
    assert hypothesis.supporting_events() == original_events
    assert {row.event_id for row in hypothesis.evidence if row.relation == "counter"} == original_events
    assert persona.evidence_review_ledger[-1].event_kind == "same_event"
    assert persona.evidence_review_ledger[-1].disposition == "applied"
    after = hypothesis.model_dump()
    # Requoting the correction itself as a different event anchor cannot make
    # a second counter out of identical current testimony.
    repeated = context(34, words=["纠正先前的理解。", f"我说过“{event_quote}”，但那并不表示喜欢等待。",
                                  "其实等待让我难以了解完整过程，请改掉那个猜测。"])
    repeated_review = current_review(repeated, hypothesis, relation="counter")
    repeated_review["event_evidence"] = [deepcopy(repeated_review["evidence"][0])]
    run(persona, repeated, [repeated_review], [change(repeated_review, hypothesis, relation=relation)])
    assert hypothesis.model_dump() == after
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    correct(33, "还是那次练习，我再强调一次不喜欢等待。")
    assert hypothesis.model_dump() == after
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("boundary", ["only_old_quote", "another_old_excerpt", "old_review_missing", "change_excludes_correction"])
def test_old_event_reinterpretation_requires_new_current_quote_in_actual_change(boundary):
    persona = PersonaProfile(user_id="person")
    first = context(41, words=["练习画圆。", "我画好了圆。这个动作已经完成。", "接着我画了下一笔。"])
    supporting, proposal = interaction_review(first)
    old_quote = "我画好了圆。"
    supporting["evidence"][0]["evidence_quote"] = old_quote
    supporting["event_evidence"] = [deepcopy(supporting["evidence"][0])]
    proposal["evidence"] = deepcopy(supporting["evidence"])
    run(persona, first, [supporting], [proposal])
    hypothesis, = persona.person_hypotheses
    before = hypothesis.model_dump()
    ctx = context(42, words=["谈谈先前练习。", "我画好了圆。但这不表示我喜欢等待。", "请不要继续原先的推断。"])
    counter = current_review(ctx, hypothesis, relation="counter")
    counter["event_evidence"] = [{"turn_index": 1, "evidence_quote": old_quote}]
    if boundary == "only_old_quote":
        counter["evidence"] = deepcopy(counter["event_evidence"])
    elif boundary == "another_old_excerpt":
        ctx.turns[1].user = first.turns[1].user
        counter["evidence"] = [{"turn_index": 1, "evidence_quote": "这个动作已经完成。"}]
    elif boundary == "old_review_missing":
        persona.evidence_review_ledger.clear()
    else:
        counter["evidence"].append(deepcopy(counter["event_evidence"][0]))
    proposal = change(counter, hypothesis)
    if boundary == "change_excludes_correction":
        proposal["evidence"] = deepcopy(counter["event_evidence"])
    updater = run(persona, ctx, [counter], [proposal])
    assert updater.last_status == "ok"
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
