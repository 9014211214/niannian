"""Data-free, pre-call checks for the CPED holdout *proposal*.

This is not a dataset loader, candidate builder, source audit, or experiment
runner. The caller must retain the local audit supporting its metadata. Passing
this gate grants no rights and proves neither source correctness nor quality.
No role identifiers, quotes, case hashes, or arbitrary metadata are exported.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


COHORT_SIZE = 48
LOGICAL_CALLS = COHORT_SIZE * 6  # Formation plus five behavior conditions.
SURFACE_BASELINES = frozenset({"length", "punctuation", "lexical", "position"})
REQUIRED_CASE_AUDITS = frozenset({
    "consistent_big_five",
    "target_is_next_speaker_turn",
    "three_distractors_other_roles_and_dialogues",
    "matched_emotion_act_length_punctuation_topic",
    "no_duplicate_text_or_fragment_overlap",
    "identities_and_gold_removed_from_model_inputs",
})
_BOUNDARY = "caller_metadata_only_not_source_audit_or_effect_evidence"


@dataclass(frozen=True)
class HoldoutCaseAudit:
    """Private metadata from an independently retained candidate audit.

    ``role_key`` is the local (TV_ID, Speaker) pair. Dialogue identifiers must
    be scoped to that show. The audit flags require actual local checks; this
    module cannot verify their truth without reading the restricted source.
    """

    role_key: tuple[str, str] = field(repr=False)
    formation_dialogues: tuple[str, ...] = field(repr=False)
    holdout_dialogue: str = field(repr=False)
    formation_utterances: int
    context_turns: int
    correct_position: int
    elderly: bool = False
    completed_audits: frozenset[str] = field(default_factory=frozenset, repr=False)


@dataclass(frozen=True)
class HoldoutRunPlan:
    terms_review_acknowledged: bool = False
    aggregate_only_accepted: bool = False
    frozen_audit_and_fingerprints_retained: bool = False
    budget_approved: bool = False
    logical_call_budget: int = 0
    network_call_budget: int = 0
    max_retries_per_call: int = 0
    estimated_cost_usd: float = 0.0
    cost_ceiling_usd: float = 0.0


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _positive_finite(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(value)
        and value > 0
    )


def preflight_cped_holdout(
    cases: Sequence[HoldoutCaseAudit],
    *,
    surface_correct_counts: Mapping[str, int],
    plan: HoldoutRunPlan,
) -> dict[str, Any]:
    """Check a preselected cohort without reading files or calling models.

    Insufficient independent roles is a normal feasibility stop, not a software
    exception. Selection, matching and surface-baseline computation remain the
    responsibility of the future local candidate builder. Fixed protocol gates
    are not relaxed to make a small corpus pass. Terms acknowledgement is
    checked before iterating even the private metadata.
    """
    if (
        plan.terms_review_acknowledged is not True
        or plan.aggregate_only_accepted is not True
    ):
        return _report("stopped_terms", {}, 0, 0, 0, [0] * 4, {})

    failures: Counter[str] = Counter()
    roles: set[tuple[str, str]] = set()
    positions = [0] * 4
    elderly_roles: set[tuple[str, str]] = set()
    for case in cases:
        role = case.role_key
        role_valid = (
            isinstance(role, tuple) and len(role) == 2
            and all(isinstance(part, str) and part.strip() for part in role)
        )
        if not role_valid:
            failures["invalid_role_key"] += 1
        elif role in roles:
            failures["repeated_role"] += 1
        else:
            roles.add(role)
            if case.elderly is True:
                elderly_roles.add(role)
        dialogues = case.formation_dialogues
        if (
            not isinstance(dialogues, tuple)
            or not all(isinstance(item, str) and item.strip() for item in dialogues)
            or len(set(dialogues)) < 3
            or len(set(dialogues)) != len(dialogues)
            or not isinstance(case.holdout_dialogue, str)
            or not case.holdout_dialogue.strip()
            or case.holdout_dialogue in dialogues
        ):
            failures["invalid_dialogue_separation"] += 1
        if not _integer(case.formation_utterances, 12):
            failures["insufficient_formation_utterances"] += 1
        if not _integer(case.context_turns, 2) or case.context_turns > 6:
            failures["invalid_holdout_context"] += 1
        if not _integer(case.correct_position) or case.correct_position > 3:
            failures["invalid_candidate_position"] += 1
        else:
            positions[case.correct_position] += 1
        if type(case.elderly) is not bool:
            failures["invalid_age_slice"] += 1
        if (
            not isinstance(case.completed_audits, frozenset)
            or case.completed_audits != REQUIRED_CASE_AUDITS
        ):
            failures["case_audit_incomplete"] += 1

    if len(cases) != COHORT_SIZE or len(roles) != COHORT_SIZE:
        failures["cohort_size_not_48_independent_roles"] += 1
    if positions != [12] * 4:
        failures["candidate_positions_not_balanced"] += 1

    surface: dict[str, int] = {}
    if set(surface_correct_counts) != SURFACE_BASELINES:
        failures["surface_audit_incomplete"] += 1
    else:
        for name in sorted(SURFACE_BASELINES):
            count = surface_correct_counts[name]
            if not _integer(count) or count > len(cases):
                failures["invalid_surface_count"] += 1
            else:
                surface[name] = count
                if count * 100 > 35 * len(cases):
                    failures["surface_shortcut_above_35_percent"] += 1
        if surface.get("position") != max(positions):
            failures["position_baseline_count_mismatch"] += 1

    if plan.frozen_audit_and_fingerprints_retained is not True:
        failures["frozen_audit_not_retained"] += 1
    budget_valid = (
        plan.budget_approved is True
        and _integer(plan.logical_call_budget)
        and plan.logical_call_budget == LOGICAL_CALLS
        and _integer(plan.network_call_budget, LOGICAL_CALLS)
        and _integer(plan.max_retries_per_call)
        and plan.network_call_budget <= (
            LOGICAL_CALLS * (plan.max_retries_per_call + 1)
        )
        and _positive_finite(plan.estimated_cost_usd)
        and _positive_finite(plan.cost_ceiling_usd)
        and plan.estimated_cost_usd <= plan.cost_ceiling_usd
    )
    if not budget_valid:
        failures["budget_plan_not_ready"] += 1
    status = "stopped_feasibility" if failures else "metadata_preflight_passed"
    return _report(
        status, dict(failures), len(cases), len(roles), len(elderly_roles),
        positions, surface,
    )


def _report(
    status: str, failures: dict[str, int], cases: int, roles: int, elderly: int,
    positions: list[int], surface: dict[str, int],
) -> dict[str, Any]:
    report = {
        "suite": "cped_holdout_metadata_preflight_v1",
        "status": status,
        "evidence_status": _BOUNDARY,
        "artifact_scope": "aggregate_only_no_case_level_derivatives",
        "case_count": cases,
        "independent_role_count": roles,
        "elderly_role_count": elderly,
        "elderly_slice_sample_gate_passed": (
            status == "metadata_preflight_passed" and elderly >= 24
        ),
        "required_role_count": COHORT_SIZE,
        "planned_logical_calls": LOGICAL_CALLS,
        "executed_model_calls": 0,
        "position_counts": positions,
        "surface_correct_counts": dict(sorted(surface.items())),
        "failure_counts": dict(sorted(failures.items())),
    }
    assert_aggregate_only_preflight(report)
    return report


_FAILURE_CODES = frozenset({
    "invalid_role_key", "repeated_role", "invalid_dialogue_separation",
    "insufficient_formation_utterances", "invalid_holdout_context",
    "invalid_candidate_position", "invalid_age_slice", "case_audit_incomplete",
    "cohort_size_not_48_independent_roles", "candidate_positions_not_balanced",
    "surface_audit_incomplete", "invalid_surface_count",
    "surface_shortcut_above_35_percent", "position_baseline_count_mismatch",
    "frozen_audit_not_retained", "budget_plan_not_ready",
})


def assert_aggregate_only_preflight(report: Mapping[str, Any]) -> None:
    """Reject additions, unbounded text, case lists and identifiers on export.

    This validates this module's exact preflight schema, not arbitrary Aria
    reports. It is intentionally fail-closed rather than a keyword scrubber.
    """
    constants = {
        "suite": "cped_holdout_metadata_preflight_v1",
        "evidence_status": _BOUNDARY,
        "artifact_scope": "aggregate_only_no_case_level_derivatives",
    }
    counts = {
        "case_count", "independent_role_count", "elderly_role_count",
        "required_role_count", "planned_logical_calls", "executed_model_calls",
    }
    expected = set(constants) | counts | {
        "status", "elderly_slice_sample_gate_passed", "position_counts",
        "surface_correct_counts", "failure_counts",
    }
    valid = (
        set(report) == expected
        and all(report.get(key) == value for key, value in constants.items())
        and all(_integer(report.get(key)) for key in counts)
        and isinstance(report.get("status"), str)
        and report.get("status") in {
            "stopped_terms", "stopped_feasibility", "metadata_preflight_passed",
        }
        and type(report.get("elderly_slice_sample_gate_passed")) is bool
        and isinstance(report.get("position_counts"), list)
        and len(report["position_counts"]) == 4
        and all(_integer(value) for value in report["position_counts"])
    )
    for key, allowed in (
        ("surface_correct_counts", SURFACE_BASELINES),
        ("failure_counts", _FAILURE_CODES),
    ):
        values = report.get(key)
        valid = valid and (
            isinstance(values, dict)
            and set(values) <= allowed
            and all(_integer(value) for value in values.values())
        )
    if not valid:
        raise ValueError("Invalid aggregate-only CPED preflight report")
