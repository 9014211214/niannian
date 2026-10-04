"""Synthetic metadata only: never load or quote a CPED source file."""
from dataclasses import replace
import json

import pytest

from aria.bench.cped_preflight import (
    HoldoutCaseAudit,
    HoldoutRunPlan,
    REQUIRED_CASE_AUDITS,
    assert_aggregate_only_preflight,
    preflight_cped_holdout,
)


def _cases(count=48, elderly=0):
    return [HoldoutCaseAudit(
        role_key=(f"private-tv-{index}", f"private-role-{index}"),
        formation_dialogues=("private-first", "private-second", "private-third"),
        holdout_dialogue="private-fourth", formation_utterances=12,
        context_turns=2, correct_position=index % 4, elderly=index < elderly,
        completed_audits=REQUIRED_CASE_AUDITS,
    ) for index in range(count)]


def _plan(**changes):
    return replace(HoldoutRunPlan(
        terms_review_acknowledged=True, aggregate_only_accepted=True,
        frozen_audit_and_fingerprints_retained=True, budget_approved=True,
        logical_call_budget=288, network_call_budget=576, max_retries_per_call=1,
        estimated_cost_usd=2, cost_ceiling_usd=3,
    ), **changes)


def _run(cases=None, surface=None, plan=None):
    return preflight_cped_holdout(
        _cases() if cases is None else cases,
        surface_correct_counts=(
            {"length": 12, "punctuation": 13, "lexical": 16, "position": 12}
            if surface is None else surface
        ), plan=plan or _plan(),
    )


def test_valid_metadata_does_not_become_effect_evidence_or_execute_models():
    report = _run()
    assert report["status"] == "metadata_preflight_passed"
    assert report["executed_model_calls"] == 0
    assert report["planned_logical_calls"] == 288
    assert "not_source_audit_or_effect" in report["evidence_status"]
    assert "private-" not in json.dumps(report)
    assert_aggregate_only_preflight(report)


@pytest.mark.parametrize("field", [
    "terms_review_acknowledged", "aggregate_only_accepted",
])
def test_terms_gate_does_not_iterate_private_metadata(field):
    class DoNotRead:
        def __iter__(self):
            raise AssertionError("metadata was read without acknowledgement")
    report = _run(cases=DoNotRead(), plan=_plan(**{field: False}))
    assert report["status"] == "stopped_terms"
    assert report["case_count"] == 0


def test_thirteen_roles_stop_normally_and_do_not_emit_elder_accuracy():
    report = _run(cases=_cases(13, elderly=13))
    assert report["status"] == "stopped_feasibility"
    assert report["independent_role_count"] == 13
    assert report["elderly_role_count"] == 13
    assert report["elderly_slice_sample_gate_passed"] is False
    assert "accuracy" not in json.dumps(report)


@pytest.mark.parametrize("elderly,expected", [(23, False), (24, True)])
def test_elder_sample_gate_is_separate_and_does_not_produce_a_score(elderly, expected):
    report = _run(cases=_cases(elderly=elderly))
    assert report["elderly_slice_sample_gate_passed"] is expected


@pytest.mark.parametrize("change,reason", [
    ({"role_key": ("private-tv-1", "private-role-1")}, "repeated_role"),
    ({"formation_dialogues": ("a", "a", "b")}, "invalid_dialogue_separation"),
    ({"holdout_dialogue": "private-first"}, "invalid_dialogue_separation"),
    ({"formation_utterances": 11}, "insufficient_formation_utterances"),
    ({"formation_utterances": True}, "insufficient_formation_utterances"),
    ({"context_turns": 7}, "invalid_holdout_context"),
    ({"context_turns": 1}, "invalid_holdout_context"),
    ({"correct_position": True}, "invalid_candidate_position"),
    ({"correct_position": 4}, "invalid_candidate_position"),
    ({"elderly": "yes"}, "invalid_age_slice"),
    ({"completed_audits": frozenset()}, "case_audit_incomplete"),
])
def test_case_metadata_constraints_fail_closed(change, reason):
    cases = _cases()
    cases[0] = replace(cases[0], **change)
    report = _run(cases=cases)
    assert report["failure_counts"][reason] >= 1
    assert report["status"] == "stopped_feasibility"


@pytest.mark.parametrize("count,reason", [
    (17, "surface_shortcut_above_35_percent"),
    (True, "invalid_surface_count"),
    (-1, "invalid_surface_count"),
    (49, "invalid_surface_count"),
])
def test_surface_counts_use_fixed_denominator_without_bool_coercion(count, reason):
    report = _run(surface={
        "length": count, "punctuation": 12, "lexical": 12, "position": 12,
    })
    assert reason in report["failure_counts"]


def test_missing_surface_and_mismatched_position_are_not_silently_accepted():
    assert "surface_audit_incomplete" in _run(surface={})["failure_counts"]
    report = _run(surface={
        "length": 12, "punctuation": 12, "lexical": 12, "position": 0,
    })
    assert "position_baseline_count_mismatch" in report["failure_counts"]


@pytest.mark.parametrize("change", [
    {"budget_approved": False}, {"logical_call_budget": 287},
    {"network_call_budget": 287}, {"network_call_budget": 577},
    {"max_retries_per_call": -1}, {"estimated_cost_usd": float("nan")},
    {"cost_ceiling_usd": float("inf")}, {"estimated_cost_usd": 4},
    {"cost_ceiling_usd": True},
])
def test_invalid_or_unapproved_budget_stops(change):
    assert "budget_plan_not_ready" in _run(plan=_plan(**change))["failure_counts"]


def test_frozen_source_audit_must_be_retained_but_gate_cannot_validate_it():
    report = _run(plan=_plan(frozen_audit_and_fingerprints_retained=False))
    assert "frozen_audit_not_retained" in report["failure_counts"]


@pytest.mark.parametrize("key,value", [
    ("cases", [{"quote": "private text"}]),
    ("case_count", "private text"),
    ("failure_counts", {"private-role": 1}),
    ("surface_correct_counts", {"length": "private text"}),
    ("status", "private text"),
    ("status", ["private text"]),
    ("status", {"private": "text"}),
    ("position_counts", [12, 12, 12, "private text"]),
])
def test_aggregate_export_rejects_unknown_fields_keys_or_free_text(key, value):
    report = _run()
    report[key] = value
    with pytest.raises(ValueError, match="aggregate-only"):
        assert_aggregate_only_preflight(report)
