"""Deterministic production effects, NOT a rerun/rescore of r8 or model quality.

R8_RETIRE_RAW is the unchanged saved synthetic output from
dev_staged_instructions/free/dsi_s4, initial attempt. The compact prior below
preserves that hypothesis's confidence, lifecycle and three support records;
unrelated profile fields/old non-adoption reviews are intentionally omitted.
Other mutations are authored boundary controls, never edited r8 evidence.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from aria.engine.memory.belief_change import (
    BELIEF_CHANGE_REASONS,
    CONFIDENCE_UPDATE_POLICY_VERSION,
    claim_scope_fingerprint,
    update_confidence,
)
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater, reflection_system_prompt
from aria.engine.model.persona import PersonHypothesis, PersonHypothesisEvidence, PersonaProfile
from tests.test_reflection_evidence_review import apply, change, context, review, setup, support
from tests.test_reflection_review_contract_v22 import RawResponse


R8_RETIRE_SOURCE_FINGERPRINT = "2985b79e7e045943987a4e9ca3fbb7cd10a568d23adcbcf0e07307b6b408163b"
R8_RETIRE_TEXT_SHA256 = "3fc91fd2385c4dba2b46a3588d7c03280bfaa6c626a1963738c4d4678ee6603f"
R8_RETIRE_RAW = '{"protocol_version":"2.0","reviews":[{"id":"r1","hypothesis_id":"ph_d288e998629c","claim_quote":"在学习新软件操作步骤时的情境下，可能倾向于要求分步讲解并自己动手操作，而非让对方代为完成","scope_quote":"学习新软件操作步骤时","applicability":"within_scope","relation":"counter","evidence_basis":"stated_preference","event_kind":"new_event","prior_event_id":null,"evidence":[{"turn_index":1,"evidence_quote":"我想换个教法，这类新操作先给全程概览更适合我。"},{"turn_index":2,"evidence_quote":"不想按一步等一步了，先看全貌再动手。"}],"event_evidence":[{"turn_index":1,"evidence_quote":"我想换个教法，这类新操作先给全程概览更适合我。"}],"reason":"用户在学习新软件操作时明确表示不想再分步等待，而要先看全貌，与旧主张的分步偏好直接矛盾，属于范围内真正反例","event_summary":"用户明确要求改变学习新操作的教法，从分步等待改为先看全程概览"}],"changes":[{"review_id":"r1","id":"ph_d288e998629c","framework":"response","dimension":"autonomy","claim":"在学习新软件操作步骤时的情境下，可能倾向于要求分步讲解并自己动手操作，而非让对方代为完成","scope":"学习新软件操作步骤时","scope_tags":["task_request"],"relation":"retire","lineage":{"id":"ph_d288e998629c","claim_quote":"在学习新软件操作步骤时的情境下，可能倾向于要求分步讲解并自己动手操作，而非让对方代为完成","scope_quote":"学习新软件操作步骤时","mechanism_relation":"same","scope_relation":"within"},"atomic_claim":{"aspect":"操作指导方式","conditions":"学习新软件操作步骤时","tendency":"倾向于先要求全程概览再动手，而非分步等待","outcome":null,"evidence_basis":"stated_preference"},"target_value":null,"confidence":0.5,"evidence":[{"turn_index":1,"evidence_quote":"我想换个教法，这类新操作先给全程概览更适合我。"},{"turn_index":2,"evidence_quote":"不想按一步等一步了，先看全貌再动手。"}]}]}'


def r8_prior():
    raw = json.loads(R8_RETIRE_RAW)["changes"][0]
    hypothesis = PersonHypothesis(
        id=raw["id"], framework="response", dimension="autonomy", claim=raw["claim"],
        scope=raw["scope"], scope_tags=["task_request"], confidence=.473,
        status="candidate", first_observed_at="2026-01-01T00:00:00",
        last_confirmed_at="2026-01-02T00:00:00",
        atomic_claim={
            "aspect": "操作指导方式", "conditions": raw["scope"],
            "tendency": "倾向于要求分步讲解并自己动手操作，而非让对方代为完成",
            "outcome": None, "evidence_basis": "stated_preference",
        },
    )
    # Exactly the retained support quote/event identity projection of r8 s3.
    for session, turn, quote, event, record in [
        (1, 1, "你先只讲第一步，我按好了再听下一步。", "98fbe43ed972104a2876dbfc", "115b08ff6d52b63d7ffecba9"),
        (2, 0, "今天学把字体换大，咱们慢慢来。", "6a386a1fe04d8c6140756b15", "40073f7990afb03355725d8a"),
        (2, 1, "还是一回说一个动作，等我试完。", "6a386a1fe04d8c6140756b15", "40073f7990afb03355725d8a"),
    ]:
        hypothesis.evidence.append(PersonHypothesisEvidence(
            at=f"2026-01-0{session}T00:00:00", session_id=f"week2_person_3_free:session_{session}",
            relation="support", summary=f"用户原话：“{quote}”", turn_indices=[turn],
            source="conversation_quote" if session == 1 else "conversation_quote:lineage:response/autonomy",
            scope=raw["scope"], scope_tags=["task_request"], event_id="evt_" + event,
            review_id="rev_" + record, evidence_basis="stated_preference", event_kind="new_event",
        ))
    return PersonaProfile(user_id="week2_person_3_free", person_hypotheses=[hypothesis])


def r8_context():
    ctx = context(4, words=[
        "新学的扫描功能不急着用。", "我想换个教法，这类新操作先给全程概览更适合我。",
        "不想按一步等一步了，先看全貌再动手。",
    ])
    ctx.session_id = "week2_person_3_free:session_4"
    return ctx


def r8_replay(*, strength=None, confidence=None, status=None, relation=None, fenced=False):
    payload = json.loads(R8_RETIRE_RAW)
    if strength is not None:
        payload["changes"][0]["confidence"] = strength
    if relation is not None:
        payload["changes"][0]["relation"] = relation
        payload["reviews"][0]["relation"] = "support" if relation == "support" else "counter"
    raw = R8_RETIRE_RAW if strength is relation is None else json.dumps(payload, ensure_ascii=False)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    persona = r8_prior()
    if confidence is not None:
        persona.person_hypotheses[0].confidence = confidence
    if status is not None:
        persona.person_hypotheses[0].status = status
    updater = PersonaHypothesisUpdater(RawResponse(raw), evidence_review_protocol=True)
    updater.update(persona, r8_context())
    assert updater.last_status == "ok"
    return updater, persona, persona.person_hypotheses[0]


def test_exact_r8_low_strength_retire_now_weakens_belief_not_falsely_retired():
    assert hashlib.sha256(R8_RETIRE_RAW.encode()).hexdigest() == R8_RETIRE_TEXT_SHA256
    canonical = json.dumps(R8_RETIRE_RAW, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == R8_RETIRE_SOURCE_FINGERPRINT
    updater, persona, hypothesis = r8_replay()
    assert hypothesis.confidence == pytest.approx(.348)  # r8 was unchanged .473
    assert hypothesis.status == "candidate" and hypothesis.activation_epoch == 0
    assert len(hypothesis.evidence) == 5
    assert len(hypothesis.counter_sessions()) == 1
    effect, = updater.last_belief_change_results
    assert effect["proposed_relation"] == "retire" and effect["effective_relation"] == "counter"
    assert effect["evidence_recorded_count"] == 2 and effect["belief_changed"]
    assert not effect["actually_retired"] and not effect["reactivated"]
    assert effect["reason"] == "low_strength_retire_as_counter"
    assert effect["confidence_before"] == .473 and effect["model_evidence_strength"] == .5
    assert effect["claim_scope_before_fingerprint"] == effect["claim_scope_after_fingerprint"]
    assert effect["review_id"] == persona.evidence_review_ledger[-1].id
    assert all(e.review_id == effect["review_id"] for e in hypothesis.evidence[-2:])
    # Historical diagnostics still mean evidence applied, not actual withdrawal.
    assert updater.last_proposal_results[0]["status"] == "applied"
    assert updater.last_proposal_results[0]["proposed_relation"] == "retire"


@pytest.mark.parametrize("strength", [0, .5, .699999, .7, .700001, 1])
def test_retire_threshold_both_sides_and_endpoints(strength):
    updater, _, hypothesis = r8_replay(strength=strength)
    effect, = updater.last_belief_change_results
    explicit = strength >= .7
    expected = 0 if explicit else .473 - (.15 * strength + .05)
    assert hypothesis.confidence == pytest.approx(expected)
    assert effect["actually_retired"] is explicit
    assert effect["effective_relation"] == ("retire" if explicit else "counter")
    assert effect["reason"] == ("explicit_retire" if explicit else "low_strength_retire_as_counter")
    assert hypothesis.claim == json.loads(R8_RETIRE_RAW)["changes"][0]["claim"]


@pytest.mark.parametrize("relation,strength", [("support", 0), ("support", .5), ("counter", 0), ("counter", .7)])
def test_nonretire_relations_preserve_prior_policy_weights(relation, strength):
    updater, _, hypothesis = r8_replay(relation=relation, strength=strength)
    expected = .473 + .14 * strength + .05 if relation == "support" else .473 - (.15 * strength + .05)
    assert hypothesis.confidence == pytest.approx(expected)
    effect, = updater.last_belief_change_results
    assert effect["effective_relation"] == relation
    assert effect["reason"] == f"{relation}_evidence"
    assert effect["model_evidence_strength"] == strength
    assert hypothesis.confidence != strength  # input is never assigned posterior


@pytest.mark.parametrize("strength", [0, .5, 1])
def test_new_belief_has_absent_before_not_fake_zero_existing_person(strength):
    updater, llm, persona = setup()
    ctx = context()
    item = review(ctx)
    apply(updater, llm, persona, ctx, [item], [change(item, confidence=strength)])
    assert updater.last_status == "ok"
    effect, = updater.last_belief_change_results
    assert effect["confidence_before"] is None and effect["status_before"] is None
    assert effect["activation_epoch_before"] is None and effect["target_value_before"] is None
    assert effect["claim_scope_before_fingerprint"] is None
    assert effect["confidence_after"] == pytest.approx(.18 + .30 * strength + .03)
    assert effect["confidence_policy_version"] == CONFIDENCE_UPDATE_POLICY_VERSION == "1.1"
    assert effect["effective_relation"] == "new" and effect["reason"] == "new_evidence"
    assert effect["belief_changed"] and not effect["actually_retired"]


def test_applied_support_at_cap_is_evidence_not_necessarily_changed_belief():
    updater, _, hypothesis = r8_replay(relation="support", strength=.5, confidence=1, status="working")
    effect, = updater.last_belief_change_results
    assert effect["evidence_recorded_count"] == 2 and hypothesis.confidence == 1
    assert not effect["belief_changed"] and not effect["actually_retired"]
    assert effect["status_before"] == effect["status_after"] == "working"


@pytest.mark.parametrize("prior,status,expected", [(.25, "candidate", True), (.3, "candidate", False)])
def test_counter_and_low_retire_can_retire_via_unchanged_confidence_floor(prior, status, expected):
    updater, _, hypothesis = r8_replay(confidence=prior, status=status)
    effect, = updater.last_belief_change_results
    assert effect["effective_relation"] == "counter" and effect["actually_retired"] is expected
    assert hypothesis.confidence == pytest.approx(prior - .125)
    assert (hypothesis.status == "retired") is expected


@pytest.mark.parametrize("confidence,retired", [(.149999, True), (.15, True), (.150001, False)])
def test_lifecycle_floor_itself_is_unchanged(confidence, retired):
    hypothesis = r8_prior().person_hypotheses[0]
    hypothesis.confidence = confidence
    PersonaHypothesisUpdater._refresh_status(hypothesis)
    assert (hypothesis.status == "retired") is retired


@pytest.mark.parametrize("confidence,events,days,rows,counters,expected", [
    (.499999, 2, 2, 3, 0, "candidate"), (.5, 2, 2, 3, 0, "working"),
    (.500001, 2, 2, 3, 0, "working"), (.6, 1, 1, 3, 0, "candidate"),
    (.6, 2, 2, 2, 0, "candidate"), (.6, 2, 2, 3, 0, "working"),
    (.749999, 5, 3, 7, 0, "working"), (.75, 5, 3, 7, 0, "stable"),
    (.750001, 5, 3, 7, 0, "stable"), (.8, 4, 3, 7, 0, "working"),
    (.8, 5, 2, 7, 0, "working"), (.8, 5, 3, 6, 0, "working"),
    (.8, 5, 3, 7, 1, "stable"), (.8, 5, 3, 7, 2, "working"),
])
def test_working_stable_numeric_and_event_gates_remain_unchanged(
    confidence, events, days, rows, counters, expected,
):
    # Authored gate-unit records, not claimed natural-person observations.
    hypothesis = r8_prior().person_hypotheses[0]
    hypothesis.evidence = []
    hypothesis.confidence = confidence
    for index in range(rows):
        event = index % events
        hypothesis.evidence.append(PersonHypothesisEvidence(
            at=f"2026-01-0{event % days + 1}T00:00:00", session_id=f"gate_s{event}",
            event_id=f"gate_e{event}", relation="support", summary="合成门槛检查",
            turn_indices=[index], activation_epoch=0,
        ))
    for index in range(counters):
        hypothesis.evidence.append(PersonHypothesisEvidence(
            at="2026-01-08T00:00:00", session_id=f"counter_s{index}",
            event_id=f"counter_e{index}", relation="counter", summary="合成反证检查",
            turn_indices=[0], activation_epoch=0,
        ))
    PersonaHypothesisUpdater._refresh_status(hypothesis)
    assert hypothesis.status == expected


@pytest.mark.parametrize("relation,strength", [("counter", .5), ("retire", .5), ("retire", .9)])
def test_already_retired_evidence_is_not_another_actual_retirement(relation, strength):
    updater, _, hypothesis = r8_replay(relation=relation, strength=strength, confidence=0, status="retired")
    effect, = updater.last_belief_change_results
    assert hypothesis.confidence == 0 and hypothesis.status == "retired"
    assert effect["evidence_recorded_count"] == 2
    assert not effect["belief_changed"] and not effect["actually_retired"]


def test_retired_reactivation_earns_new_epoch_not_old_support_reuse():
    updater, persona, hypothesis = r8_replay(strength=.9)
    assert hypothesis.status == "retired"
    ctx = context(5, words=["这是另一项新操作。", "现在我还是喜欢分步听，自己试后再继续。", "刚才这样试过，顺手。"])
    item = review(ctx, hypothesis)
    updater.llm = RawResponse(json.dumps({"protocol_version": "2.0", "reviews": [item], "changes": [change(item, hypothesis, confidence=.6)]}))
    updater.update(persona, ctx)
    effect, = updater.last_belief_change_results
    assert hypothesis.activation_epoch == 1 and hypothesis.status == "candidate"
    assert hypothesis.confidence == pytest.approx(.39)
    assert effect["reactivated"] and effect["belief_changed"] and not effect["actually_retired"]
    assert effect["activation_epoch_before"] == 0 and effect["activation_epoch_after"] == 1
    assert len({e.event_id for e in hypothesis.activation_evidence() if e.relation == "support"}) == 1
    assert len(hypothesis.supporting_events()) == 3  # audit history still retained


@pytest.mark.parametrize("kind", ["same_event", "new_event"])
def test_repeated_event_no_evidence_no_effect_even_if_model_requests_retire(kind):
    updater, persona, hypothesis = r8_replay()
    before = hypothesis.model_dump()
    ctx = r8_context()
    ctx.session_id += "_replay"
    payload = json.loads(R8_RETIRE_RAW)
    payload["reviews"][0]["event_kind"] = kind
    if kind == "same_event":
        payload["reviews"][0]["prior_event_id"] = persona.evidence_review_ledger[-1].event_id
    updater.llm = RawResponse(json.dumps(payload, ensure_ascii=False))
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and hypothesis.model_dump() == before
    assert updater.last_belief_change_results == []
    assert updater.last_proposal_results[0]["reason"] == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("scope", ["unknown", "outside_scope"])
@pytest.mark.parametrize("existing", [False, True])
def test_ineligible_review_never_gets_belief_effect_from_strength(scope, existing):
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona) if existing else None
    before = deepcopy(persona.person_hypotheses)
    ctx = context(2)
    item = review(ctx, hypothesis, applicability=scope, relation="counter" if existing else "support")
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis, confidence=1)])
    assert updater.last_status == "ok" and persona.person_hypotheses == before
    assert updater.last_belief_change_results == []


def test_unknown_without_claim_is_valid_and_context_is_not_inferred_personality():
    updater, llm, persona = setup()
    ctx = context(words=["没有真正尝试过。", "我现在还不知道哪种舒服。", "别按别人的喜好替我选。"])
    item = review(ctx, applicability="unknown", relation="unknown", evidence_basis="context_only")
    apply(updater, llm, persona, ctx, [item], [])
    assert updater.last_status == "ok" and not persona.person_hypotheses
    assert updater.last_belief_change_results == [] and not persona.active_tasks
    assert persona.evidence_review_ledger[0].disposition == "unknown_scope_no_update"


def test_invalid_attached_context_rolls_back_counter_and_all_diagnostics():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    before = persona.model_dump()
    ctx = context(2)
    item = review(ctx, hypothesis, relation="counter")
    invalid = review(ctx, id="context", evidence_basis="context_only", relation="neutral", event_evidence=[])
    apply(updater, llm, persona, ctx, [item, invalid], [change(item, hypothesis, relation="retire", confidence=.5)])
    assert updater.last_status == "failed" and persona.model_dump() == before
    assert updater.last_belief_change_results is None
    assert updater.last_proposal_results[0]["status"] == "rejected"


def test_excluded_quote_cannot_leak_into_belief_or_counter_and_failure_resets_effects():
    updater, llm, persona = setup()
    support(updater, llm, persona)
    assert updater.last_belief_change_results
    ctx = context(2, words=["普通背景。", "不要存储的隐私内容。", "我还在想。"])
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    hypothesis = persona.person_hypotheses[0]
    before = persona.model_dump()
    item = review(ctx, hypothesis, relation="counter")
    apply(updater, llm, persona, ctx, [item], [change(item, hypothesis)])
    assert updater.last_status == "failed" and persona.model_dump() == before
    assert updater.last_belief_change_results is None
    assert "不要存储的隐私内容" not in json.dumps(llm.messages[-1], ensure_ascii=False)


def test_diagnostic_scalars_do_not_contain_quote_or_claim_and_hash_matches_contract():
    updater, _, hypothesis = r8_replay()
    effect, = updater.last_belief_change_results
    raw = json.dumps(effect, ensure_ascii=False)
    assert hypothesis.claim not in raw and "全程概览" not in raw
    assert effect["reason"] in BELIEF_CHANGE_REASONS
    assert len(effect) == 21
    assert effect["claim_scope_after_fingerprint"] == claim_scope_fingerprint(hypothesis)
    canonical = json.dumps({k: getattr(hypothesis, k) for k in (
        "claim", "scope", "framework", "dimension", "scope_tags",
    )}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == effect["claim_scope_after_fingerprint"]


def test_single_fence_preserves_raw_and_has_same_actual_belief_effect():
    bare, _, _ = r8_replay()
    wrapped, _, _ = r8_replay(fenced=True)
    assert wrapped.last_belief_change_results == bare.last_belief_change_results
    assert wrapped.last_response_transport["wrapper_removed"]
    assert wrapped.llm.raw == "```json\n" + R8_RETIRE_RAW + "\n```"


def test_skipped_and_legacy_runs_do_not_claim_evaluated_v23_effects():
    updater, persona, _ = r8_replay()
    updater.update(persona, r8_context())
    assert updater.last_status == "skipped" and updater.last_belief_change_results is None
    updater = PersonaHypothesisUpdater(RawResponse('{"changes":[]}'))
    updater.update(PersonaProfile(user_id="legacy"), context())
    assert updater.last_status == "ok" and updater.last_belief_change_results is None


def test_explicit_no_change_keeps_legacy_compatibility_not_new_generation_contract():
    ctx = context()
    proposal = change(review(ctx), relation="no_change")
    raw = json.dumps({"changes": [proposal]})
    persona = PersonaProfile(user_id="legacy")
    updater = PersonaHypothesisUpdater(RawResponse(raw))
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and not persona.person_hypotheses
    assert updater.last_belief_change_results is None
    strict, llm, persona = setup()
    item = review(ctx)
    apply(strict, llm, persona, ctx, [item], [change(item, relation="no_change")])
    assert strict.last_status == "failed" and strict.last_belief_change_results is None


def test_target_value_change_at_confidence_cap_is_real_engineering_state_change():
    updater, llm, persona = setup()
    ctx = context()
    item = review(ctx, evidence_basis="repeated_pattern")
    proposal = change(item, framework="big_five", dimension="conscientiousness", target_value=.4)
    apply(updater, llm, persona, ctx, [item], [proposal])
    hypothesis = persona.person_hypotheses[0]
    ctx2 = context(2)
    item2 = review(ctx2, hypothesis, evidence_basis="repeated_pattern")
    apply(updater, llm, persona, ctx2, [item2], [change(item2, hypothesis, target_value=.4)])
    hypothesis.confidence = 1
    ctx3 = context(3)
    item3 = review(ctx3, hypothesis, evidence_basis="repeated_pattern")
    apply(updater, llm, persona, ctx3, [item3], [change(item3, hypothesis, target_value=.8)])
    effect, = updater.last_belief_change_results
    assert effect["confidence_before"] == effect["confidence_after"] == 1
    assert effect["status_before"] == effect["status_after"] == "working"
    assert effect["target_value_before"] == pytest.approx(.4)
    assert effect["target_value_after"] == pytest.approx(.52)
    assert effect["belief_changed"]


def test_old_persisted_profile_roundtrips_without_confidence_migration():
    old = r8_prior().model_dump()
    restored = PersonaProfile.model_validate(old)
    assert restored.model_dump() == old
    assert restored.person_hypotheses[0].confidence == .473
    updater = PersonaHypothesisUpdater(RawResponse(R8_RETIRE_RAW), evidence_review_protocol=True)
    updater.update(restored, r8_context())
    assert updater.last_belief_change_results[0]["confidence_before"] == .473


def test_prompt_separates_evidence_strength_minimum_scope_and_real_unknown():
    prompt = reflection_system_prompt(True)
    for part in (
        "prompt contract 2.4", "不是人物假设更新后", "不能直接复制为本次 confidence",
        "最小推断范围", "紧急性和阶段", "最多保留该次同类", "对该方式的偏好仍是 unknown",
        "不能为形成标签强改 within_scope", "证据入账、信念实际变化、实际撤回",
        "低于 0.7 按", "不要为了越过阈值抬高 confidence",
    ):
        assert part in prompt


def test_policy_rejects_no_evidence_and_unknown_relations():
    with pytest.raises(ValueError, match="fresh_evidence_required"):
        update_confidence(.473, "counter", .5, 0)
    with pytest.raises(ValueError, match="unsupported_belief_relation"):
        update_confidence(.473, "unknown", .5, 1)
