"""Deterministic engineering belief updates, not calibrated psychology scores.

The wire field ``change.confidence`` is retained for old callers. It means the
model's strength assessment of THIS quoted evidence relation, not the desired
posterior value of ``PersonHypothesis.confidence``. The latter is a bounded
engineering accumulator with evidence floors and lifecycle gates; neither is
a probability of the person's true personality. Policy 1.1 preserves the prior
weights and gates, but a retire proposal below .7 now weakens the old belief as
counter evidence instead of recording a counter with no numerical response.
"""
from __future__ import annotations

import hashlib
import json

from ..model.persona import PersonHypothesis


CONFIDENCE_UPDATE_POLICY_VERSION = "1.1"
EXPLICIT_RETIRE_MIN_STRENGTH = 0.7
BELIEF_CHANGE_REASONS = frozenset({
    "new_evidence", "support_evidence", "counter_evidence",
    "low_strength_retire_as_counter", "explicit_retire",
})


def claim_scope_fingerprint(hypothesis: PersonHypothesis) -> str:
    """Hash exact retained content without copying quotes into diagnostics."""
    content = {key: getattr(hypothesis, key) for key in (
        "claim", "scope", "framework", "dimension", "scope_tags",
    )}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def belief_state(hypothesis: PersonHypothesis | None) -> dict:
    """Only auditable state scalars; absent is not a zero-confidence old claim."""
    return {
        "confidence": hypothesis.confidence if hypothesis is not None else None,
        "status": hypothesis.status if hypothesis is not None else None,
        "activation_epoch": hypothesis.activation_epoch if hypothesis is not None else None,
        "target_value": hypothesis.target_value if hypothesis is not None else None,
        "claim_scope_fingerprint": (
            claim_scope_fingerprint(hypothesis) if hypothesis is not None else None
        ),
    }


def update_confidence(
    prior: float, relation: str, evidence_strength: float, evidence_count: int,
) -> tuple[float, str, str]:
    """Return accumulator, effective relation and a content-free policy code.

    Call only after quote, applicability, event novelty and lineage checks.
    A low strength never turns unknown or repeated evidence into a fresh vote.
    Existing evidence floors are deliberate engineering weights, not estimates
    fitted to population data. Retirement/status gates remain caller-owned.
    """
    if relation not in {"new", "support", "counter", "retire"}:
        raise ValueError("unsupported_belief_relation")
    if evidence_count < 1:
        raise ValueError("fresh_evidence_required")
    effective = relation
    reason = f"{relation}_evidence"
    if relation == "retire":
        if evidence_strength >= EXPLICIT_RETIRE_MIN_STRENGTH:
            return 0.0, "retire", "explicit_retire"
        effective = "counter"
        reason = "low_strength_retire_as_counter"
    if effective in {"new", "support"}:
        value = (
            .18 + evidence_strength * .30 + min(.08, max(0, evidence_count - 1) * .03)
            if prior == 0 else
            prior + .14 * evidence_strength + min(.075, evidence_count * .025)
        )
    else:
        value = prior - (.15 * evidence_strength + min(.08, evidence_count * .025))
    return max(0.0, min(1.0, value)), effective, reason
