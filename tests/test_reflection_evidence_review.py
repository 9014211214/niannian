"""Protocol 2.0 exercises real updater changes; no model/network test doubles score quality."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import (
    PersonaHypothesisUpdater, _review_context, build_reflection_messages,
    reflection_system_prompt,
)
from aria.engine.model.persona import PersonaProfile


class Replies:
    def __init__(self):
        self.payload = {"protocol_version": "2.0", "reviews": [], "changes": []}
        self.messages = []

    def complete(self, messages):
        self.messages.append(deepcopy(messages))
        return json.dumps(self.payload, ensure_ascii=False)


def context(index=1, *, words=None):
    ctx = EngineContext(user_id="person", scene_id="companion_elder", session_id=f"session_{index}")
    for user in words or [f"今天试的是第{index}个新功能。", "先给我一个动作，我更喜欢自己试完再继续。", "这只是我喜欢的沟通节奏。"]:
        ctx.add_turn(user, "我听到了。")
    return ctx


def quotes(ctx, indices):
    return [{"turn_index": i, "evidence_quote": ctx.turns[i].user} for i in indices]


def review(ctx, hypothesis=None, **overrides):
    result = {
        "id": "r1", "hypothesis_id": hypothesis.id if hypothesis else None,
        "claim_quote": hypothesis.claim if hypothesis else None,
        "scope_quote": (hypothesis.scope or "") if hypothesis else None,
        "applicability": "within_scope", "relation": "support", "evidence_basis": "stated_preference",
        "event_kind": "new_event", "prior_event_id": None,
        "evidence": quotes(ctx, [1, 2]), "event_evidence": quotes(ctx, [0]),
        "reason": "本人明确说了特定教学节奏偏好，不据此假定情绪或授权。",
        "event_summary": "本人描述本次教学节奏选择。",
    }
    result.update(overrides)
    return result


def change(item, hypothesis=None, **overrides):
    result = {
        "review_id": item["id"], "id": hypothesis.id if hypothesis else "",
        "framework": hypothesis.framework if hypothesis else "response",
        "dimension": hypothesis.dimension if hypothesis else "low_pressure",
        "claim": hypothesis.claim if hypothesis else "模型自由文本不会绕过原子主张",
        "scope": hypothesis.scope if hypothesis else "不赶时间的新操作教学",
        "scope_tags": ["task_request"], "relation": "support" if hypothesis else "new",
        "lineage": {
            "id": hypothesis.id, "claim_quote": hypothesis.claim, "scope_quote": hypothesis.scope or "",
            "mechanism_relation": "same", "scope_relation": "within",
        } if hypothesis else None,
        "atomic_claim": {
            "aspect": "教学节奏", "conditions": "不赶时间的新操作教学",
            "tendency": "更偏好一次一个动作、自己试完再继续", "outcome": None,
            "evidence_basis": item["evidence_basis"],
        },
        "target_value": None, "confidence": 0.9, "evidence": deepcopy(item["evidence"]),
    }
    if item["relation"] == "counter":
        result["relation"] = "counter"
    result.update(overrides)
    return result


def apply(updater, llm, persona, ctx, reviews, changes):
    llm.payload = {"protocol_version": "2.0", "reviews": reviews, "changes": changes}
    updater.update(persona, ctx)


def setup():
    llm = Replies()
    persona = PersonaProfile(user_id="person")
    updater = PersonaHypothesisUpdater(llm, evidence_review_protocol=True,
                                       now_provider=lambda: datetime(2026, 1, 1))
    return updater, llm, persona


def support(updater, llm, persona, index=1, ctx=None):
    ctx = ctx or context(index)
    hypothesis = persona.person_hypotheses[0] if persona.person_hypotheses else None
    item = review(ctx, hypothesis)
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis)])
    assert updater.last_status == "ok"
    return persona.person_hypotheses[0]


def test_stated_preference_forms_candidate_without_inventing_causal_outcome():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    assert hypothesis.status == "candidate"
    assert hypothesis.atomic_claim.evidence_basis == "stated_preference"
    assert hypothesis.atomic_claim.outcome is None
    assert hypothesis.scope == hypothesis.atomic_claim.conditions
    assert "模型自由文本" not in hypothesis.claim
    assert len(hypothesis.supporting_events()) == 1
    assert all(e.event_id and e.review_id and e.evidence_basis == "stated_preference" for e in hypothesis.evidence)
    assert persona.evidence_review_ledger[-1].disposition == "applied"
    assert updater.last_review_summary["applied_review_count"] == 1
    assert len(llm.messages) == 1


def test_independent_support_counter_and_fresh_recovery_use_unchanged_thresholds():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    first_confidence = hypothesis.confidence
    support(updater, llm, persona, 2)
    assert hypothesis.status == "working" and hypothesis.confidence >= 0.5
    assert hypothesis.confidence > first_confidence
    old_claim = hypothesis.claim
    ctx = context(3, words=["我刚试了另一个功能。", "我改主意了，新功能先给完整路线更舒服。", "不想按一个动作等一个动作了。"])
    item = review(ctx, hypothesis, relation="counter")
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis)])
    assert updater.last_status == "ok"
    assert hypothesis.status == "candidate" and hypothesis.confidence < 0.5
    assert hypothesis.claim == old_claim and len(hypothesis.counter_sessions()) == 1
    support(updater, llm, persona, 4)
    assert hypothesis.status == "working" and hypothesis.claim == old_claim
    assert len(hypothesis.supporting_events()) == 3
    assert len(llm.messages) == 4  # no second model or online follow-up


@pytest.mark.parametrize("applicability,reason", [
    ("outside_scope", "outside_scope_no_update"), ("unknown", "unknown_scope_no_update"),
])
def test_nonapplicable_proposed_counter_is_preserved_but_cannot_change_belief(applicability, reason):
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    before = hypothesis.model_dump()
    ctx = context(2)
    item = review(ctx, hypothesis, applicability=applicability, relation="counter")
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis)])
    assert updater.last_status == "ok"
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == reason


def test_unknown_nonadoption_is_available_to_next_reflection_without_full_history():
    updater, llm, persona = setup()
    ctx = context(1, words=["旧记录只写了一次。", "这不能代表我一直这样。", "今天是在核对那一页，没有新事件。"])
    item = review(ctx, applicability="unknown", relation="unknown", event_kind="unknown", evidence_basis="context_only")
    apply(updater, llm, persona, ctx, [item], [])
    assert persona.person_hypotheses == []
    support(updater, llm, persona, 2)
    actual = llm.messages[-1][1]["content"]
    assert "这不能代表我一直这样" in actual
    assert "unknown_scope_no_update" in actual
    assert "过去的有界证据审查" in actual
    assert "[turn 0] 用户:旧记录只写了一次。" not in actual


def test_first_retrospective_is_allowed_once_but_linked_replay_is_not_fresh_support():
    updater, llm, persona = setup()
    ctx = context(1, words=["两年前第一次学发照片。", "当时先讲一小步，我自己试完再继续。", "那次这个节奏顺手，不代表每次都这样。"])
    item = review(ctx, event_kind="retrospective")
    apply(updater, llm, persona, ctx, [item], [change(item)])
    assert updater.last_status == "ok"
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "candidate"
    before = hypothesis.model_dump()
    prior_event = persona.evidence_review_ledger[-1].event_id
    ctx2 = context(2, words=["再讲的仍是两年前那一次。", "我把那次发照片的记录念了一遍。", "不是第二次新的体验。"])
    replay = review(ctx2, hypothesis, event_kind="same_event", prior_event_id=prior_event)
    apply(updater, llm, persona, ctx2, [replay], [change(replay, hypothesis)])
    assert updater.last_status == "ok"
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    assert len(hypothesis.supporting_events()) == 1


def test_relabeling_same_quoted_event_as_new_does_not_inflate_confidence():
    updater, llm, persona = setup()
    ctx = context(1)
    hypothesis = support(updater, llm, persona, ctx=ctx)
    before = hypothesis.model_dump()
    ctx2 = context(2, words=[t.user for t in ctx.turns])
    replay = review(ctx2, hypothesis, event_kind="new_event", prior_event_id=None)
    apply(updater, llm, persona, ctx2, [replay], [change(replay, hypothesis)])
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].prior_event_id == persona.evidence_review_ledger[0].event_id


def test_mixed_evidence_on_same_claim_applies_true_counter_not_remaining_support():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    support(updater, llm, persona, 2)
    before = hypothesis.confidence
    ctx = context(3, words=["今天试的新扫描功能。", "自己动手我仍喜欢，但讲法要改为先说完整路线。", "不再喜欢一个动作等一个动作。"])
    positive = review(ctx, hypothesis, id="s", evidence=quotes(ctx, [1]))
    negative = review(ctx, hypothesis, id="c", relation="counter", evidence=quotes(ctx, [2]))
    apply(updater, llm, persona, ctx, [positive, negative], [change(positive, hypothesis), change(negative, hypothesis)])
    assert updater.last_status == "ok" and hypothesis.confidence < before
    assert persona.evidence_review_ledger[-2].disposition == "same_claim_counter_takes_priority"
    assert persona.evidence_review_ledger[-1].disposition == "applied"
    assert all(e.relation == "counter" for e in hypothesis.evidence if e.session_id == ctx.session_id)


def test_permission_statement_is_context_not_inferred_personality_or_action():
    updater, llm, persona = setup()
    ctx = context(1, words=["今天这份聚餐通知可以转发。", "就这份通知，别的要另问。", "我不是在介绍自己的性格。"])
    item = review(ctx, relation="neutral", evidence_basis="context_only")
    apply(updater, llm, persona, ctx, [item], [])
    assert updater.last_status == "ok"
    assert persona.person_hypotheses == []
    assert persona.active_tasks == [] and persona.notification_receipts == []
    assert persona.evidence_review_ledger[0].disposition == "context_only_no_update"


def test_multiple_utterances_are_not_independent_repeated_pattern():
    updater, llm, persona = setup()
    ctx = context(1)
    item = review(ctx, evidence_basis="repeated_pattern")
    proposal = change(item, framework="big_five", dimension="conscientiousness")
    apply(updater, llm, persona, ctx, [item], [proposal])
    assert updater.last_status == "ok"
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "candidate"
    assert len(hypothesis.supporting_events()) == 1
    assert persona.evidence_review_ledger[-1].pattern_evidence_status == "provisional_single_event"
    # The provisional slot can accumulate genuinely independent evidence; it
    # is not a protocol dead end that can only ever report no change.
    ctx2 = context(2)
    item2 = review(ctx2, hypothesis, evidence_basis="repeated_pattern")
    apply(updater, llm, persona, ctx2, [item2], [change(item2, hypothesis)])
    assert hypothesis.status == "working"
    assert len(hypothesis.supporting_events()) == 2


def test_strict_unbound_new_claim_cannot_lexically_merge_into_existing_identity():
    updater, llm, persona = setup()
    original = support(updater, llm, persona)
    before = original.model_dump()
    ctx = context(2)
    item = review(ctx)  # intentionally unbound; old heuristic would merge
    proposal = change(item)
    proposal["atomic_claim"]["tendency"] = "更偏好一次一个动作，自己尝试后再继续"
    apply(updater, llm, persona, ctx, [item], [proposal])
    assert updater.last_status == "ok"
    assert original.model_dump() == before
    assert len(persona.person_hypotheses) == 2
    assert persona.person_hypotheses[1].id != original.id
    # Duplicate/near-duplicate meaning remains a semantic quality error for a
    # judge, but never becomes a silent confidence update of the wrong id.


def test_exact_existing_claim_requires_explicit_review_identity():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    before = persona.model_dump()
    ctx = context(2)
    item = review(ctx)
    proposal = change(item)
    apply(updater, llm, persona, ctx, [item], [proposal])
    assert updater.last_status == "failed" and persona.model_dump() == before
    assert hypothesis.supporting_events()


def test_new_explicit_correction_of_old_interpretation_is_not_blocked_as_event_replay():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    support(updater, llm, persona, 2)
    before = hypothesis.confidence
    ctx = context(3, words=[
        "我现在要纠正上次被理解的意思。", "以前只讲一步是临时网络卡顿，不是我偏好的节奏。", "对新操作我其实更喜欢先知道完整路线。",
    ])
    correction = review(ctx, hypothesis, relation="counter", event_kind="new_event",
                        event_evidence=quotes(ctx, [0, 2]))
    apply(updater, llm, persona, ctx, [correction], [change(correction, hypothesis)])
    assert updater.last_status == "ok" and hypothesis.confidence < before
    assert persona.evidence_review_ledger[-1].disposition == "applied"


@pytest.mark.parametrize("tamper", [
    "missing_version", "wrong_version", "unknown_event", "unbound_same_event", "fabricated_quote",
    "assistant_quote", "quote_outside_review", "mixed_causal_outcome", "extra_authorization_field",
    "missing_atomic", "wrong_relation", "nonfinite_confidence",
    "invalid_scope_tags", "out_of_range_scale",
])
def test_strict_protocol_failures_are_not_silently_accepted_as_no_change(tamper):
    updater, llm, persona = setup()
    ctx = context(1)
    item = review(ctx)
    proposal = change(item)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [proposal]}
    if tamper == "missing_version":
        payload.pop("protocol_version")
    elif tamper == "wrong_version":
        payload["protocol_version"] = "1.0"
    elif tamper == "unknown_event":
        item.update(event_kind="same_event", prior_event_id="invented")
    elif tamper == "unbound_same_event":
        item["event_kind"] = "same_event"
    elif tamper == "fabricated_quote":
        item["evidence"][0]["evidence_quote"] = "用户没有讲过的话"
    elif tamper == "assistant_quote":
        item["evidence"][0]["evidence_quote"] = "我听到了。"
    elif tamper == "quote_outside_review":
        proposal["evidence"] = quotes(ctx, [0])
    elif tamper == "mixed_causal_outcome":
        proposal["atomic_claim"]["outcome"] = "因此变得开心且允许自动联系家人"
    elif tamper == "extra_authorization_field":
        proposal["atomic_claim"]["allow_share"] = True
    elif tamper == "missing_atomic":
        proposal.pop("atomic_claim")
    elif tamper == "wrong_relation":
        proposal["relation"] = "counter"
    elif tamper == "invalid_scope_tags":
        proposal["scope_tags"] = ["general", "task_request"]
    elif tamper == "out_of_range_scale":
        proposal["target_value"] = 3
    else:
        proposal["confidence"] = float("nan")
    llm.payload = payload
    updater.update(persona, ctx)
    assert updater.last_status == "failed"
    assert updater.last_error_type == (
        "invalid_json_response" if tamper == "nonfinite_confidence" else "invalid_evidence_review_protocol"
    )
    assert persona.person_hypotheses == [] and persona.evidence_review_ledger == []
    assert ctx.session_id not in persona.person_model_state.reflected_session_ids


def test_stale_or_changed_claim_reference_rejected_without_prior_state_mutation():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    before = persona.model_dump()
    ctx = context(2)
    item = review(ctx, hypothesis, claim_quote="同ID但换了意思")
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis)])
    assert updater.last_status == "failed" and persona.model_dump() == before


def test_sensitive_excluded_turn_not_sent_or_retained_in_any_review():
    updater, llm, persona = setup()
    ctx = context(1, words=["非敏感背景。", "不该存储的私人信息。", "这里只说一般情况。"])
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    item = review(ctx, relation="unknown", applicability="unknown", evidence_basis="context_only")
    apply(updater, llm, persona, ctx, [item], [])
    assert updater.last_status == "failed"
    assert "不该存储的私人信息" not in json.dumps(llm.messages, ensure_ascii=False)
    assert persona.evidence_review_ledger == []


def test_bounded_review_context_and_ledger_do_not_turn_into_all_history():
    updater, llm, persona = setup()
    for index in range(45):
        ctx = context(index + 1)
        updater._now = lambda i=index: datetime(2026, 1, 1) + timedelta(days=i)
        item = review(ctx, relation="unknown", applicability="unknown", event_kind="unknown", evidence_basis="context_only")
        apply(updater, llm, persona, ctx, [item], [])
        assert updater.last_status == "ok"
    assert len(persona.evidence_review_ledger) == 40
    assert len(_review_context(persona)) <= 24
    assert len(llm.messages) == 45
    assert persona.person_hypotheses == []


def test_retained_hypothesis_event_signature_survives_review_ledger_aging():
    updater, llm, persona = setup()
    initial_context = context(1)
    hypothesis = support(updater, llm, persona, ctx=initial_context)
    before = hypothesis.model_dump()
    event_id = hypothesis.evidence[0].event_id
    for index in range(2, 43):
        ctx = context(index)
        item = review(ctx, hypothesis, relation="unknown", applicability="unknown", event_kind="unknown", evidence_basis="context_only")
        apply(updater, llm, persona, ctx, [item], [])
    assert all(r.event_id != event_id for r in persona.evidence_review_ledger)
    current = context(43, words=[t.user for t in initial_context.turns])
    replay = review(current, hypothesis)
    apply(updater, llm, persona, current, [replay], [change(replay, hypothesis)])
    assert hypothesis.model_dump() == before
    assert persona.evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    message = llm.messages[-1][1]["content"]
    assert event_id in message and '"lineage_anchors"' in message


def test_old_profile_and_legacy_updater_remain_compatible():
    persona = PersonaProfile.model_validate({"user_id": "legacy"})
    assert persona.evidence_review_ledger == []
    llm = Replies()
    ctx = context(1)
    legacy = change(review(ctx))
    legacy.pop("review_id")
    legacy.pop("atomic_claim")
    llm.payload = {"changes": [legacy]}
    updater = PersonaHypothesisUpdater(llm)
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].atomic_claim is None
    assert all(e.event_id is None for e in persona.person_hypotheses[0].evidence)
    assert persona.evidence_review_ledger == []
    assert llm.messages[0][0]["content"] == reflection_system_prompt(False)
    assert "过去的有界证据审查" not in llm.messages[0][1]["content"]


def test_exported_input_builder_matches_actual_strict_call():
    updater, llm, persona = setup()
    ctx = context(1)
    before = persona.model_copy(deep=True)
    support(updater, llm, persona, ctx=ctx)
    conversation = "\n".join(f"[turn {i}] 用户:{t.user}\nAria:{t.agent}" for i, t in enumerate(ctx.turns))
    assert llm.messages[0] == build_reflection_messages(before, conversation, [], True)


@pytest.mark.parametrize("raw", [
    'preamble\n```json\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```',
    '{"protocol_version":"2.0","reviews":[],"changes":[]} extra prose',
    '[{"protocol_version":"2.0","reviews":[],"changes":[]}]',
])
def test_strict_input_does_not_strip_preamble_or_trailing_output(raw):
    class Raw:
        def complete(self, messages):
            return raw

    updater = PersonaHypothesisUpdater(Raw(), evidence_review_protocol=True)
    persona = PersonaProfile(user_id="person")
    updater.update(persona, context(1))
    assert updater.last_status == "failed" and updater.last_error_type == "invalid_json_response"
    assert persona.person_model_state.sessions_observed == 0
