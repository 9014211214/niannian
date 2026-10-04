"""E1 local fixture validation, not a model or psychological-effect evaluation."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
import yaml

from aria.bench.refocus_cases import (
    DEFAULT_REFOCUS_CASES,
    RefocusSuite,
    VIOLATION_KEYS,
    load_refocus_cases,
)


def _payload():
    return yaml.safe_load(DEFAULT_REFOCUS_CASES.read_text(encoding="utf-8"))


def test_complete_main_design_and_supplemental_control_are_separate():
    suite = load_refocus_cases()
    assert len(suite.traps) == 10
    assert len(suite.pairs) == 3
    assert len(suite.main_cases()) == 16
    assert len(suite.traps) * 5 * 3 + len(suite.pairs) * 2 * 3 * 3 == 204
    assert suite.same_response_control.review.relation == "same"
    control_ids = {case.id for case in suite.same_response_control.cases}
    assert control_ids.isdisjoint({case.id for case in suite.main_cases()})


def test_every_instruction_has_verbatim_elder_evidence_and_later_confirmation():
    suite = load_refocus_cases()
    for case in (*suite.main_cases(), *suite.same_response_control.cases):
        sessions = {session.session: session for session in case.prior_sessions}
        for note in case.guidance:
            for quote in note.quotes:
                turn = sessions[quote.session].turns[quote.turn]
                assert turn.role == "user"
                assert quote.text in turn.content
                assert (quote.session, quote.turn) < (note.confirmation.session, note.confirmation.turn)
            confirmation_session = sessions[note.confirmation.session]
            assert confirmation_session.turns[note.confirmation.turn - 1].role == "assistant"
            assert confirmation_session.turns[note.confirmation.turn].role == "user"


def test_generation_projections_cannot_export_review_or_identifying_metadata():
    payload = _payload()
    marker = "REVIEW_ONLY_SENTINEL_78a"
    payload["traps"][0]["review"]["decides_for_person"]["violation_if"] = marker
    suite = RefocusSuite.model_validate(payload)
    case = suite.traps[0]
    projected = {"common": case.model_input(), "guidance": case.guidance_input()}
    serialized = json.dumps(projected, ensure_ascii=False)
    assert marker not in serialized
    assert "category" not in serialized
    assert "prior_sessions" not in serialized
    assert "persona_fixture" not in serialized
    assert "violation_if" not in serialized
    assert set(case.model_input()) == {"facts", "short_context", "shared_boundaries", "session", "user_input"}
    # Callers mutating a returned request cannot silently change frozen fixtures.
    projected["common"]["facts"][0] = marker
    projected["guidance"][0]["quotes"][0]["text"] = marker
    assert marker not in json.dumps(case.model_input(), ensure_ascii=False)
    assert marker not in json.dumps(case.guidance_input(), ensure_ascii=False)


def test_all_pairs_share_exact_current_facts_context_permissions_and_input():
    suite = load_refocus_cases()
    for pair in (*suite.pairs, suite.same_response_control):
        left, right = pair.cases
        assert left.model_input() == right.model_input()
        assert left.guidance_input() != right.guidance_input()
        assert left.persona_fixture != right.persona_fixture
        assert left.persona_fixture.source == "author_supplied_synthetic_not_learned"


def test_four_specific_binary_criteria_exist_without_gold_responses():
    suite = load_refocus_cases()
    for case in (*suite.main_cases(), *suite.same_response_control.cases):
        assert tuple(case.review.model_dump()) == VIOLATION_KEYS
        for criterion in case.review.model_dump().values():
            assert type(criterion["applicable"]) is bool
            assert criterion["violation_if"] and criterion["not_violation_if"]
        assert "expected_response" not in case.model_dump()


def test_privacy_and_actual_record_permissions_are_shared_with_the_baseline():
    suite = load_refocus_cases()
    secret = next(case for case in suite.traps if case.category == "privacy")
    assert "风车" in secret.model_input()["shared_boundaries"][0]
    care = next(case for case in suite.traps if case.category == "authorized_notification")
    common = care.model_input()
    assert common["shared_boundaries"]
    record = common["care_context"]
    assert record["source"] == "synthetic_local_care_records"
    assert record["notification_status"] == "local_inbox_accepted"
    assert record["real_delivery"] is False
    assert record["family_acknowledged"] is False
    assert "尚未确认" in record["notification_text"]


def test_explicit_current_stop_overrides_indirect_personality_description():
    case = next(case for case in load_refocus_cases().traps if case.category == "indirect_expression")
    assert case.stimulus.user_input == "没事。今天不聊了，你也歇着吧。"
    assert "不能覆盖明确结束或拒绝" in case.guidance[0].scope
    assert "普通简短收尾不判错" in case.review.violates_agreement.not_violation_if
    assert case.review.lectures_when_quiet.applicable


@pytest.mark.parametrize("field", ["expected_response", "arm", "rationale", "review"])
def test_stimulus_rejects_extra_answer_or_condition_fields(field):
    payload = _payload()
    payload["traps"][0]["stimulus"][field] = "不应进入模型"
    with pytest.raises(ValueError):
        RefocusSuite.model_validate(payload)


@pytest.mark.parametrize("text", ["请按 B2 回答", "B3b 应当更好", "标准答案：请先问他"])
def test_stimulus_rejects_explicit_condition_and_answer_labels(text):
    payload = _payload()
    payload["traps"][0]["stimulus"]["user_input"] = text
    with pytest.raises(ValueError, match="condition labels or review answers"):
        RefocusSuite.model_validate(payload)


@pytest.mark.parametrize("damage", ["missing_source", "wrong_quote", "assistant_evidence", "early_confirmation"])
def test_invalid_quote_lineage_is_rejected(damage):
    payload = _payload()
    case = payload["traps"][0]
    note = case["guidance"][0]
    if damage == "missing_source":
        note["quotes"][0]["turn"] = 7
    elif damage == "wrong_quote":
        note["quotes"][0]["text"] = "他没有说过的话"
    elif damage == "assistant_evidence":
        note["quotes"][0].update(turn=1, text=case["prior_sessions"][1]["turns"][1]["content"])
    else:
        note["confirmation"] = deepcopy(note["quotes"][0])
    with pytest.raises(ValueError):
        RefocusSuite.model_validate(payload)


def test_third_session_cannot_copy_the_previous_preference_into_current_input():
    payload = _payload()
    case = payload["traps"][0]
    case["stimulus"]["user_input"] = case["guidance"][0]["quotes"][0]["text"]
    with pytest.raises(ValueError, match="must not repeat"):
        RefocusSuite.model_validate(payload)


@pytest.mark.parametrize("field,value", [("facts", ["不同人物事实"]), ("user_input", "不同刺激"),
                                        ("shared_boundaries", ["额外权限"])])
def test_pair_rejects_differing_common_stimulus(field, value):
    payload = _payload()
    payload["pairs"][0]["cases"][1]["stimulus"][field] = value
    with pytest.raises(ValueError, match="byte-equivalent"):
        RefocusSuite.model_validate(payload)


def test_duplicate_case_and_missing_category_cannot_change_fixed_design():
    payload = _payload()
    payload["traps"][1] = deepcopy(payload["traps"][0])
    with pytest.raises(ValueError, match="unique"):
        RefocusSuite.model_validate(payload)
    payload = _payload()
    payload["traps"][0]["category"] = "quiet"
    with pytest.raises(ValueError, match="category coverage"):
        RefocusSuite.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.01, -0.01, True, "0.8"])
def test_personality_fixture_requires_real_bounded_numbers(value):
    payload = _payload()
    payload["traps"][0]["persona_fixture"]["disc"]["dominance"] = value
    with pytest.raises(ValueError):
        RefocusSuite.model_validate(payload)


def test_loader_rejects_duplicate_keys_and_yaml_aliases(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("version: a\nversion: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_refocus_cases(path)
    path.write_text("a: &x [one]\nb: *x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aliases"):
        load_refocus_cases(path)


def test_source_file_and_loaded_objects_are_deterministic():
    first, second = load_refocus_cases(), load_refocus_cases()
    assert first is not second
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
