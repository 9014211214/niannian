"""Contract 2.2 deterministic replay, not a rerun or rescore of r7.

The one source response below is the saved synthetic r7 cold-start response at
dev_private_setback/free/session_1. Its byte fingerprint is retained for audit.
All contrasts are explicitly authored fixture mutations. No private data,
network, frozen report, clinical truth or model-quality score is used here.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from aria.engine.memory.diagnosis import (
    EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
    EVIDENCE_REVIEW_PROTOCOL_VERSION,
    PersonaHypothesisUpdater,
    reflection_system_prompt,
)
from aria.engine.model.persona import PersonaProfile
from tests.test_reflection_evidence_review import (
    apply, change, context, review, setup, support,
)


R7_UNKNOWN_SOURCE_FINGERPRINT = "cbf7f35c0332a94b6e6bf074c6b41b0b008a94309a455e2703886b1f76545da9"
R7_UNKNOWN_SOURCE_TEXT_SHA256 = "d5a13c8db8dc1c550a8968a1088be8e3ceced42118fb86d8f84298884c61c96d"
R7_UNKNOWN_SOURCE_RESPONSE = '''{"protocol_version":"2.0","reviews":[{
"id":"r1","hypothesis_id":null,"claim_quote":null,"scope_quote":null,
"applicability":"unknown","relation":"support","evidence_basis":"stated_preference",
"event_kind":"new_event","prior_event_id":null,
"evidence":[{"turn_index":1,"evidence_quote":"先让我跟你说清楚经过，别急着拉女儿来。"},{"turn_index":2,"evidence_quote":"说完我自己再决定要不要找她。"}],
"event_evidence":[{"turn_index":0,"evidence_quote":"小组的花我没养好，有点不好意思。"},{"turn_index":1,"evidence_quote":"先让我跟你说清楚经过，别急着拉女儿来。"}],
"reason":"用户明确表达希望先说清经过，再自己决定是否联系女儿，属于自主决定的表述性偏好。",
"event_summary":"用户就未养好小组花的事，表达想先自述经过再自行决定是否联系女儿。"
}],"changes":[{
"review_id":"r1","id":"","framework":"response","dimension":"autonomy",
"claim":"","scope":"讨论自己未做好的事情时","scope_tags":["autonomy_threat"],
"relation":"new","lineage":null,
"atomic_claim":{"aspect":"决定是否联系家人","conditions":"讨论自己未做好的事情时",
"tendency":"倾向于先说清经过再自己决定是否联系女儿","outcome":null,
"evidence_basis":"stated_preference"},
"target_value":null,"confidence":0.6,
"evidence":[{"turn_index":1,"evidence_quote":"先让我跟你说清楚经过，别急着拉女儿来。"},{"turn_index":2,"evidence_quote":"说完我自己再决定要不要找她。"}]
}]}'''


class RawResponse:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.raw


def replay(raw, ctx=None):
    llm = RawResponse(raw)
    updater = PersonaHypothesisUpdater(llm, evidence_review_protocol=True)
    persona = PersonaProfile(user_id="person")
    updater.update(persona, ctx or context())
    return updater, persona, llm


def source_context():
    return context(words=[
        "小组的花我没养好，有点不好意思。",
        "先让我跟你说清楚经过，别急着拉女儿来。",
        "说完我自己再决定要不要找她。",
    ])


def test_prompt_declares_new_scope_and_event_context_without_upgrading_schema():
    assert EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION == "2.3"
    assert EVIDENCE_REVIEW_PROTOCOL_VERSION == "2.0"
    prompt = reflection_system_prompt(True)
    for statement in (
        "prompt contract 2.3", "atomic_claim.conditions", "本身不是 unknown",
        "低置信候选", "不能为形成标签强改 within_scope",
        "附属 context_only review 也遵守相同事件合同",
        "没有已知 event_id 时不得填 same_event", "审查完成也不代表人物已经学会",
    ):
        assert statement in prompt
    assert "prompt contract" not in reflection_system_prompt(False)


def test_saved_r7_unknown_new_proposal_remains_unknown_and_is_now_observable():
    assert hashlib.sha256(R7_UNKNOWN_SOURCE_RESPONSE.encode()).hexdigest() == R7_UNKNOWN_SOURCE_TEXT_SHA256
    encoded = json.dumps(R7_UNKNOWN_SOURCE_RESPONSE, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(encoded.encode()).hexdigest() == R7_UNKNOWN_SOURCE_FINGERPRINT
    updater, persona, llm = replay(R7_UNKNOWN_SOURCE_RESPONSE, source_context())
    assert updater.last_status == "ok"  # Backward-compatible review success.
    assert persona.person_hypotheses == []
    assert persona.evidence_review_ledger[0].disposition == "unknown_scope_no_update"
    assert updater.last_proposal_results == [{
        "proposal_index": 0, "proposed_relation": "new", "review_status": "completed",
        "status": "deferred", "reason": "unknown_scope_no_update",
        "scope_basis": "new_atomic_conditions", "contract_warning": "ineligible_review_has_change",
    }]
    assert updater.last_application_summary["applied_count"] == 0
    assert updater.last_application_summary["deferred_count"] == 1
    assert llm.raw == R7_UNKNOWN_SOURCE_RESPONSE and llm.calls == 1


def test_authored_within_scope_contrast_can_create_bounded_candidate_without_old_id():
    # Deliberately a NEW local contrast, never repair/rewrite the source response.
    payload = json.loads(R7_UNKNOWN_SOURCE_RESPONSE)
    payload["reviews"][0]["applicability"] = "within_scope"
    updater, persona, _ = replay(json.dumps(payload, ensure_ascii=False), source_context())
    assert updater.last_status == "ok"
    assert updater.last_application_summary["status"] == "applied"
    assert updater.last_application_summary["applied_count"] == 1
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "candidate"
    assert hypothesis.scope == hypothesis.atomic_claim.conditions
    assert len(hypothesis.supporting_events()) == 1
    assert persona.active_tasks == [] and persona.notification_receipts == []


@pytest.mark.parametrize("scope", ["outside_scope", "unknown"])
def test_nonapplicable_new_contrast_cannot_be_promoted_by_empty_prior_ledger(scope):
    ctx = context()
    item = review(ctx, applicability=scope)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    updater, persona, _ = replay(json.dumps(payload), ctx)
    assert updater.last_status == "ok"
    assert not persona.person_hypotheses
    assert updater.last_proposal_results[0]["status"] == "deferred"
    assert updater.last_proposal_results[0]["reason"] == {
        "outside_scope": "outside_scope_no_update", "unknown": "unknown_scope_no_update",
    }[scope]


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_exact_single_json_wrapper_is_audited_but_content_is_unchanged(eol):
    ctx = context()
    item = review(ctx)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    raw = json.dumps(payload, ensure_ascii=False)
    wrapped = f"  ```json{eol}{raw}{eol}```  "
    updater, persona, llm = replay(wrapped, ctx)
    assert updater.last_status == "ok" and len(persona.person_hypotheses) == 1
    assert updater.last_response_transport == {
        "prompt_contract_version": "2.3", "format": "single_json_fence",
        "wrapper_removed": True, "json_status": "parsed", "content_repaired": False,
    }
    assert llm.raw == wrapped
    assert updater.last_proposal_results[0]["status"] == "applied"


@pytest.mark.parametrize("raw", [
    'Explanation\n```json\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```\nExplanation',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n',
    '```json\n{}\n```\n```json\n{}\n```',
    '```python\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```',
    '```\n{"protocol_version":"2.0","reviews":[],"changes":[]}\n```',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[]} {}\n```',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[],}\n```',
    '```json\n[{"protocol_version":"2.0","reviews":[],"changes":[]}]\n```',
    '```json\n{"protocol_version":"2.0","reviews":[],"changes":[],"extra":NaN}\n```',
])
def test_wrapper_compatibility_never_repairs_or_extracts_json_from_other_text(raw):
    updater, persona, llm = replay(raw)
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_json_response"
    assert updater.last_application_summary["status"] == "not_evaluated"
    assert updater.last_application_summary["proposal_count"] is None
    assert updater.last_proposal_results == []
    assert persona.person_hypotheses == [] and persona.evidence_review_ledger == []
    assert llm.raw == raw


def test_accepted_wrapper_does_not_rescue_fabricated_quote_or_mark_proposal_applied():
    item = review(context())
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    payload["reviews"][0]["evidence"][0]["evidence_quote"] = "PRIVATE_UNGROUNDED_QUOTE"
    updater, persona, _ = replay("```json\n" + json.dumps(payload) + "\n```")
    assert updater.last_response_transport["wrapper_removed"] is True
    assert updater.last_response_transport["json_status"] == "parsed"
    assert updater.last_error_detail == "review_quote_not_grounded_in_current_user"
    assert updater.last_application_summary["review_status"] == "failed"
    assert updater.last_proposal_results[0]["status"] == "rejected"
    assert updater.last_review_summary is None
    assert persona.person_hypotheses == [] and persona.evidence_review_ledger == []
    assert "PRIVATE_" not in json.dumps(updater.last_proposal_results)


def test_invalid_supplementary_context_review_rolls_back_otherwise_valid_new_proposal():
    updater, llm, persona = setup()
    ctx = context()
    item = review(ctx)
    extra = review(ctx, id="context", evidence_basis="context_only", relation="neutral",
                   event_kind="same_event", prior_event_id=None)
    before = persona.model_dump()
    apply(updater, llm, persona, ctx, [item, extra], [change(item)])
    assert updater.last_error_detail == "same_event_requires_known_reference"
    assert persona.model_dump() == before
    assert updater.last_application_summary["review_status"] == "failed"
    assert updater.last_application_summary["applied_count"] == 0
    assert updater.last_proposal_results[0]["reason"] == "same_event_requires_known_reference"
    assert updater.last_proposal_results[0]["status"] == "rejected"


@pytest.mark.parametrize("extra_event", ["new_event", "unknown"])
def test_current_correction_and_valid_supplementary_context_do_not_block_each_other(extra_event):
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    old_confidence = hypothesis.confidence
    ctx = context(2, words=["今天我来纠正上次说的办法。", "现在我改为喜欢先看完整步骤，不想等每一步。", "这只是更喜欢的办法，没有授权替我转发。"])
    correction = review(ctx, hypothesis, relation="counter")
    correction["event_evidence"] = deepcopy(correction["evidence"])
    boundary = review(ctx, id="boundary", evidence_basis="context_only", relation="neutral",
                      event_kind=extra_event, prior_event_id=None)
    apply(updater, llm, persona, ctx, [correction, boundary], [change(correction, hypothesis)])
    assert updater.last_status == "ok"
    assert hypothesis.confidence < old_confidence
    assert len(hypothesis.counter_sessions()) == 1
    assert updater.last_proposal_results[0]["scope_basis"] == "existing_claim_scope"
    assert updater.last_proposal_results[0]["status"] == "applied"
    assert persona.evidence_review_ledger[-1].disposition == "context_only_no_update"
    assert persona.active_tasks == [] and persona.notification_receipts == []


def test_known_same_event_remains_nonfresh_even_when_proposal_is_wrapped():
    updater, llm, persona = setup()
    hypothesis = support(updater, llm, persona)
    before = hypothesis.model_dump()
    ctx = context(2)
    item = review(ctx, hypothesis, event_kind="same_event", prior_event_id=hypothesis.evidence[0].event_id)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item, hypothesis)]}
    updater.llm = RawResponse("```json\n" + json.dumps(payload) + "\n```")
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and hypothesis.model_dump() == before
    assert updater.last_proposal_results[0]["reason"] == "same_event_no_fresh_evidence"


@pytest.mark.parametrize("prior", ["r1", "controlled_candidate", "invented_event"])
def test_no_inference_of_event_id_from_review_hypothesis_or_unknown_text(prior):
    item = review(context(), event_kind="same_event", prior_event_id=prior)
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    updater, persona, _ = replay(json.dumps(payload))
    assert updater.last_error_detail == "unknown_prior_event_reference"
    assert persona.evidence_review_ledger == []
    assert updater.last_proposal_results[0]["status"] == "rejected"


@pytest.mark.parametrize("next_action", ["no_proposal", "short_session", "duplicate_session", "bad_json", "model_error"])
def test_diagnostics_reset_and_no_proposal_is_not_learning_success(next_action):
    updater, llm, persona = setup()
    support(updater, llm, persona)
    ctx = context(2)
    if next_action == "no_proposal":
        llm.payload = {"protocol_version": "2.0", "reviews": [], "changes": []}
    elif next_action == "short_session":
        ctx = context(2, words=["你好。"])
    elif next_action == "duplicate_session":
        ctx = context(1)
    elif next_action == "bad_json":
        updater.llm = RawResponse("PRIVATE_INVALID_CONTENT")
    else:
        class Failing:
            def complete(self, messages):
                raise RuntimeError("PRIVATE_PROVIDER_MESSAGE")
        updater.llm = Failing()
    updater.update(persona, ctx)
    assert updater.last_proposal_results == []
    if next_action in {"short_session", "duplicate_session"}:
        assert updater.last_application_summary is None and updater.last_response_transport is None
    elif next_action == "no_proposal":
        assert updater.last_status == "ok"
        assert updater.last_application_summary["status"] == "no_proposal"
        assert updater.last_application_summary["applied_count"] == 0
    else:
        assert updater.last_application_summary["status"] == "not_evaluated"
        assert updater.last_application_summary["review_status"] == "not_evaluated"
    assert "PRIVATE_" not in json.dumps([updater.last_proposal_results, updater.last_application_summary])


def test_untrusted_relation_object_does_not_break_or_leak_diagnostics():
    item = review(context())
    proposal = change(item)
    proposal["relation"] = {"PRIVATE_PAYLOAD": "not a valid relation"}
    updater, persona, _ = replay(json.dumps({"protocol_version": "2.0", "reviews": [item], "changes": [proposal]}))
    assert updater.last_status == "failed"
    assert updater.last_proposal_results[0]["proposed_relation"] is None
    assert "PRIVATE_PAYLOAD" not in json.dumps(updater.last_proposal_results)
    assert persona.person_hypotheses == []


def test_excessive_transaction_has_bounded_rows_but_counts_every_rejected_proposal():
    item = review(context())
    payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)] * 4}
    updater, persona, _ = replay(json.dumps(payload))
    assert updater.last_error_detail == "review_or_change_bound_exceeded"
    assert len(updater.last_proposal_results) == 3
    assert updater.last_application_summary["proposal_count"] == 4
    assert updater.last_application_summary["rejected_count"] == 4
    assert updater.last_application_summary["omitted_proposal_results_count"] == 1
    assert persona.person_hypotheses == []


def test_legacy_multiple_proposals_compatibility_still_selects_one_and_accounts_for_all():
    updater, llm, persona = setup()
    ctx = context()
    one = review(ctx, id="r1")
    two = review(ctx, id="r2")
    apply(updater, llm, persona, ctx, [one, two], [change(one), change(two)])
    assert updater.last_status == "ok"
    assert len(persona.person_hypotheses) == 1
    assert updater.last_application_summary["proposal_count"] == 2
    assert updater.last_application_summary["applied_count"] == 1
    assert updater.last_application_summary["deferred_count"] == 1
    assert updater.last_proposal_results[1]["reason"] == "not_selected_for_application"


def test_legacy_nonreview_parser_is_unchanged_and_has_no_review_diagnostics():
    llm = RawResponse('harmless preamble\n{"changes":[]}')
    updater = PersonaHypothesisUpdater(llm)
    updater.update(PersonaProfile(user_id="person"), context())
    assert updater.last_status == "ok"
    assert updater.last_response_transport is None
    assert updater.last_application_summary is None and updater.last_proposal_results == []
