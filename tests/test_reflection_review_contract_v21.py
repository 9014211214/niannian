"""Wire-contract regression tests, not model-quality or psychological evidence.

All responses are local fixtures. No model, network, production profile or frozen
benchmark artifact is read or changed by this module.
"""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import (
    EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
    EVIDENCE_REVIEW_PROTOCOL_VERSION,
    PersonaHypothesisUpdater,
    _REFLECTION_ERROR_DETAILS,
    _review_quotes,
    reflection_system_prompt,
)
from aria.engine.model.persona import PersonaProfile


class LocalResponse:
    def __init__(self):
        self.payload = {"protocol_version": "2.0", "reviews": [], "changes": []}
        self.raw = None
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.raw if self.raw is not None else json.dumps(self.payload, ensure_ascii=False)


def setup():
    llm = LocalResponse()
    updater = PersonaHypothesisUpdater(
        llm, evidence_review_protocol=True,
        now_provider=lambda: datetime(2026, 1, 1),
    )
    return updater, llm, PersonaProfile(user_id="contract-test")


def fixture(index=1, old=None, words=None):
    ctx = EngineContext(user_id="contract-test", scene_id="companion_elder", session_id=f"wire-{index}")
    for word in words or [f"这回是第{index}个新操作。", "我喜欢先说一个动作，让我自己试完再继续。", "这是我喜欢的教学节奏。"]:
        ctx.add_turn(word, "我听到了。")
    evidence = [{"turn_index": i, "evidence_quote": ctx.turns[i].user} for i in (1, 2)]
    review = {
        "id": "r1", "hypothesis_id": old.id if old else None,
        "claim_quote": old.claim if old else None,
        "scope_quote": (old.scope or "") if old else None,
        "applicability": "within_scope", "relation": "support",
        "evidence_basis": "stated_preference", "event_kind": "new_event", "prior_event_id": None,
        "evidence": evidence,
        "event_evidence": [{"turn_index": 0, "evidence_quote": ctx.turns[0].user}],
        "reason": "本人明确描述了教学沟通偏好。", "event_summary": "本次新操作的教学节奏选择。",
    }
    change = {
        "review_id": "r1", "id": old.id if old else "",
        "framework": old.framework if old else "response",
        "dimension": old.dimension if old else "low_pressure",
        "claim": old.claim if old else "新主张按原子结构生成",
        "scope": old.scope if old else "学习新操作时",
        "scope_tags": ["task_request"], "relation": "support" if old else "new",
        "lineage": {"id": old.id, "claim_quote": old.claim, "scope_quote": old.scope or "",
                    "mechanism_relation": "same", "scope_relation": "within"} if old else None,
        "atomic_claim": {"aspect": "教学节奏", "conditions": "学习新操作时",
                         "tendency": "偏好一次一个动作，自己试完再继续",
                         "outcome": None, "evidence_basis": "stated_preference"},
        "target_value": None, "confidence": 0.8, "evidence": deepcopy(evidence),
    }
    return ctx, {"protocol_version": "2.0", "reviews": [review], "changes": [change]}


def test_prompt_contract_revision_does_not_change_data_protocol_or_legacy_prompt():
    assert EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION == "2.3"
    assert EVIDENCE_REVIEW_PROTOCOL_VERSION == "2.0"
    prompt = reflection_system_prompt(True)
    for statement in (
        "prompt contract 2.3", 'protocol_version 仍必须为字符串 "2.0"',
        "response/big_five/disc/attachment/cbt", "general 只能单独使用",
        "每个引用对象必须完全复制", "不能缩短成另一子串", "1..3 项数组",
        "2..240 个字符", "各最多 100", "总长度不得超过 180", "最多 240 个字符",
        "全部字段，不能增删字段", "outcome 字段必须", "不能省略",
        "observed_response 必须为用户有据的反应字符串", "只允许 change.relation=support",
    ):
        assert statement in prompt
    assert "prompt contract 2.3" not in reflection_system_prompt(False)


@pytest.mark.parametrize(("mutation", "expected"), [
    ("shorter_quote", "change_quote_not_in_bound_review"),
    ("general_plus_context", "invalid_explicit_scope_tags"),
    ("duplicate_tags", "invalid_explicit_scope_tags"),
    ("missing_outcome", "one_atomic_claim_required"),
    ("observed_without_outcome", "observed_response_requires_reported_outcome"),
    ("big_five_display_name", "invalid_reviewed_change_shape"),
    ("uppercase_disc_scale", "invalid_or_inapplicable_person_scale"),
    ("aspect101", "invalid_atomic_claim_text"),
    ("conditions101", "invalid_atomic_claim_text"),
    ("tendency101", "invalid_atomic_claim_text"),
    ("combined181", "atomic_claim_exceeds_retained_claim_bound"),
    ("outcome101", "invalid_atomic_outcome"),
    ("reason241", "invalid_bounded_review_explanation"),
    ("summary241", "invalid_bounded_review_explanation"),
    ("four_event_quotes", "bounded_current_quotes_required"),
    ("missing_review_field", "invalid_review_fields"),
    ("missing_change_field", "change_requires_exact_review_binding"),
    ("string_null", "review_prior_identity_mismatch"),
    ("new_claim_old_text", "new_review_has_unbound_prior_text"),
    ("bool_confidence", "invalid_change_confidence"),
])
def test_declared_limits_reject_without_mutation_and_report_safe_detail(mutation, expected):
    updater, llm, persona = setup()
    ctx, payload = fixture()
    review, change = payload["reviews"][0], payload["changes"][0]
    atomic = change["atomic_claim"]
    if mutation == "shorter_quote":
        change["evidence"][0]["evidence_quote"] = "让我自己试完再继续"
    elif mutation == "general_plus_context":
        change["scope_tags"] = ["general", "task_request"]
    elif mutation == "duplicate_tags":
        change["scope_tags"] = ["task_request", "task_request"]
    elif mutation == "missing_outcome":
        del atomic["outcome"]
    elif mutation == "observed_without_outcome":
        review["evidence_basis"] = atomic["evidence_basis"] = "observed_response"
    elif mutation == "big_five_display_name":
        change.update(framework="Big Five", dimension="conscientiousness")
    elif mutation == "uppercase_disc_scale":
        change.update(framework="DISC", dimension="conscientiousness", target_value=0.6)
    elif mutation in {"aspect101", "conditions101", "tendency101"}:
        atomic[mutation[:-3]] = "甲" * 101
    elif mutation == "combined181":
        atomic["conditions"] = "甲" * 80
        atomic["tendency"] = "乙" * (181 - len(f'在{atomic["conditions"]}的情境下，可能'))
        assert len(atomic["conditions"]) <= 100 and len(atomic["tendency"]) <= 100
    elif mutation == "outcome101":
        review["evidence_basis"] = atomic["evidence_basis"] = "observed_response"
        atomic["outcome"] = "甲" * 101
    elif mutation in {"reason241", "summary241"}:
        review["reason" if mutation == "reason241" else "event_summary"] = "甲" * 241
    elif mutation == "four_event_quotes":
        review["event_evidence"] = [{"turn_index": 0, "evidence_quote": text}
                                    for text in ("这回", "这回是", "新操作", "个新操作")]
    elif mutation == "missing_review_field":
        del review["event_summary"]
    elif mutation == "missing_change_field":
        del change["scope_tags"]
    elif mutation == "string_null":
        review["hypothesis_id"] = "null"
    elif mutation == "new_claim_old_text":
        review["claim_quote"] = "不存在的旧主张"
    else:
        change["confidence"] = True
    before = persona.model_dump()
    llm.payload = payload
    updater.update(persona, ctx)
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_evidence_review_protocol"
    assert updater.last_error_detail == expected
    assert updater.last_error_detail in _REFLECTION_ERROR_DETAILS
    assert persona.model_dump() == before
    assert updater.last_review_summary is None


def test_exact_boundary_lengths_and_subset_quote_binding_still_form_candidate():
    updater, llm, persona = setup()
    ctx, payload = fixture()
    review, change = payload["reviews"][0], payload["changes"][0]
    review["reason"] = review["event_summary"] = "甲" * 240
    atomic = change["atomic_claim"]
    atomic["aspect"] = "甲" * 100
    atomic["conditions"] = "甲" * 80
    atomic["tendency"] = "乙" * (180 - len(f'在{atomic["conditions"]}的情境下，可能'))
    change["evidence"] = [deepcopy(review["evidence"][1])]
    llm.payload = payload
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and updater.last_error_detail is None
    assert len(persona.person_hypotheses[0].claim) == 180
    assert persona.person_hypotheses[0].status == "candidate"


@pytest.mark.parametrize("length", [1, 2, 240, 241])
def test_current_quote_length_boundary_is_not_relaxed(length):
    ctx, _ = fixture(words=["甲" * 241, "本人原话甲", "本人原话乙"])
    rows = [{"turn_index": 0, "evidence_quote": "甲" * length}]
    if 2 <= length <= 240:
        assert len(_review_quotes(rows, ctx)) == 1
    else:
        with pytest.raises(ValueError, match="review_quote_not_grounded_in_current_user"):
            _review_quotes(rows, ctx)


def test_three_distinct_quote_objects_are_accepted_but_duplicates_are_not():
    ctx, _ = fixture()
    rows = [{"turn_index": i, "evidence_quote": ctx.turns[i].user} for i in range(3)]
    assert len(_review_quotes(rows, ctx)) == 3
    with pytest.raises(ValueError, match="review_quote_not_grounded_in_current_user"):
        _review_quotes([rows[0], rows[0]], ctx)


@pytest.mark.parametrize(("framework", "dimension"), [
    ("response", "low_pressure"), ("big_five", "conscientiousness"),
    ("disc", "conscientiousness"), ("attachment", "secure"), ("cbt", "protective_factor"),
])
def test_canonical_framework_wire_names_remain_valid(framework, dimension):
    updater, llm, persona = setup()
    ctx, payload = fixture()
    payload["changes"][0].update(framework=framework, dimension=dimension)
    if framework in {"big_five", "disc"}:
        payload["changes"][0]["target_value"] = 0.6
    llm.payload = payload
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and updater.last_error_detail is None
    assert persona.person_hypotheses[0].framework == framework


def test_reported_response_with_grounded_outcome_is_not_forced_to_no_change():
    updater, llm, persona = setup()
    ctx, payload = fixture(words=["这次第一次试这个功能。", "一次只说一步，我照着做，感觉轻松。", "刚才自己完成了。"])
    payload["reviews"][0]["evidence_basis"] = "observed_response"
    payload["changes"][0]["atomic_claim"].update(
        evidence_basis="observed_response", outcome="本人描述感觉轻松并完成本次操作",
    )
    llm.payload = payload
    updater.update(persona, ctx)
    assert updater.last_status == "ok"
    assert persona.person_hypotheses[0].atomic_claim.outcome


def test_existing_support_cannot_silently_disappear_as_new_but_support_and_counter_work():
    updater, llm, persona = setup()
    first, llm.payload = fixture()
    updater.update(persona, first)
    hypothesis = persona.person_hypotheses[0]
    second, payload = fixture(2, hypothesis)
    payload["changes"][0]["relation"] = "new"
    llm.payload = payload
    before = persona.model_dump()
    updater.update(persona, second)
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_evidence_review_protocol"
    assert updater.last_error_detail == "existing_support_requires_support_change"
    assert persona.model_dump() == before
    payload["changes"][0]["relation"] = "support"
    updater.update(persona, second)
    assert updater.last_status == "ok" and updater.last_error_detail is None
    assert len(hypothesis.supporting_events()) == 2 and hypothesis.status == "working"
    confidence = hypothesis.confidence
    third, payload = fixture(3, hypothesis, words=["这次试了另一个功能。", "我改主意了，现在要先看完整步骤。", "不想一次等一个动作了。"])
    payload["reviews"][0]["relation"] = payload["changes"][0]["relation"] = "counter"
    llm.payload = payload
    updater.update(persona, third)
    assert updater.last_status == "ok" and updater.last_error_detail is None
    assert hypothesis.confidence < confidence and len(hypothesis.counter_sessions()) == 1


@pytest.mark.parametrize("next_action", ["success", "no_change", "short_session", "already_reflected"])
def test_error_detail_resets_before_success_and_both_skip_paths(next_action):
    updater, llm, persona = setup()
    first, llm.payload = fixture()
    updater.update(persona, first)
    failed, llm.payload = fixture(2, persona.person_hypotheses[0])
    llm.payload["changes"][0]["scope_tags"] = ["general", "task_request"]
    updater.update(persona, failed)
    assert updater.last_error_detail == "invalid_explicit_scope_tags"
    next_ctx, llm.payload = fixture(3, persona.person_hypotheses[0])
    if next_action == "no_change":
        llm.payload = {"protocol_version": "2.0", "reviews": [], "changes": []}
    elif next_action == "short_session":
        next_ctx = EngineContext(user_id="contract-test", scene_id="companion_elder", session_id="short")
        next_ctx.add_turn("你好。", "你好。")
    elif next_action == "already_reflected":
        next_ctx = first
    calls = llm.calls
    updater.update(persona, next_ctx)
    assert updater.last_status == ("skipped" if next_action in {"short_session", "already_reflected"} else "ok")
    assert updater.last_error_type is None and updater.last_error_detail is None
    if updater.last_status == "skipped":
        assert llm.calls == calls


class UnrenderableValueError(ValueError):
    def __str__(self):
        raise AssertionError("Untrusted exception text must not be rendered")


@pytest.mark.parametrize("error", [
    ValueError("SENSITIVE_PRIVATE_SENTENCE"), KeyError("SENSITIVE_PRIVATE_SENTENCE"),
    TypeError("SENSITIVE_PRIVATE_SENTENCE"),
    ValueError("invalid_review_fields: SENSITIVE_PRIVATE_SENTENCE"),
    ValueError("invalid_review_fields", "SENSITIVE_PRIVATE_SENTENCE"),
    UnrenderableValueError("SENSITIVE_PRIVATE_SENTENCE"),
])
def test_unknown_validation_errors_never_expose_exception_text(monkeypatch, error):
    updater, llm, persona = setup()
    ctx, llm.payload = fixture()
    def fail(*args):
        raise error
    monkeypatch.setattr(updater, "_reviewed_changes", fail)
    updater.update(persona, ctx)
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_evidence_review_protocol"
    assert updater.last_error_detail == "invalid_evidence_review_protocol"
    assert updater.last_review_summary is None
    assert persona.person_hypotheses == [] and persona.evidence_review_ledger == []


@pytest.mark.parametrize(("raw", "detail"), [
    ("SENSITIVE_PRIVATE_SENTENCE: not JSON", "invalid_json_response"),
    ('{"protocol_version":"2.0","changes":[],"reviews":[],"extra":NaN}', "nonfinite_json_number"),
    ('["SENSITIVE_PRIVATE_SENTENCE"]', "reflection_object_required"),
    ('{"SENSITIVE_PRIVATE_SENTENCE":1}', "changes_array_required"),
])
def test_json_errors_keep_legacy_category_and_only_safe_details(raw, detail):
    updater, llm, persona = setup()
    ctx, _ = fixture()
    llm.raw = raw
    updater.update(persona, ctx)
    assert updater.last_error_type == "invalid_json_response"
    assert updater.last_error_detail == detail
    assert "SENSITIVE_PRIVATE_SENTENCE" not in updater.last_error_detail


def test_transport_error_does_not_expose_provider_exception(monkeypatch):
    updater, llm, persona = setup()
    ctx, _ = fixture()
    def fail(messages):
        raise RuntimeError("SENSITIVE_PRIVATE_SENTENCE: provider request content")
    monkeypatch.setattr(llm, "complete", fail)
    updater.update(persona, ctx)
    assert updater.last_status == "failed"
    assert updater.last_error_type == updater.last_error_detail == "model_call_failed"


def test_legacy_success_still_works_and_starts_without_diagnostics():
    llm = LocalResponse()
    llm.payload = {"changes": []}
    updater = PersonaHypothesisUpdater(llm)
    persona = PersonaProfile(user_id="contract-test")
    ctx, _ = fixture()
    assert updater.last_error_detail is None
    updater.update(persona, ctx)
    assert updater.last_status == "ok" and updater.last_error_detail is None
