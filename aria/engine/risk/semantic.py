"""Map a validated same-turn model signal to deterministic safety handling.

This is a routing hint, not a diagnosis or permission to contact another person.
The keyword monitor still runs first and cannot be downgraded by this signal.
"""
from ...interfaces.risk import RiskSignal
from ..policy.online_decision import ONLINE_DECISION_VERSION_SAFETY, EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE, SAFETY_DECISION_VERSIONS


def semantic_safety_signal(ctx, utterance):
    trace = ctx.working_memory.get("online_turn_decision") or {}
    if (trace.get("version") not in SAFETY_DECISION_VERSIONS
            or trace.get("status") != "parsed"
            or trace.get("turn_index") != len(ctx.turns)
            or trace.get("safety_source") != "online_model_semantics"):
        return None
    if ctx.working_memory.get("_online_decision_version", trace.get("version")) != trace.get("version"):
        return None
    evidence = [utterance]
    if trace["version"] in EVIDENCE_DECISION_VERSIONS:
        from .safety_evidence import assess_evidence
        assessment = assess_evidence(trace.get("safety_concern"), trace.get("safety_evidence"), utterance)
        trace["safety_assessment"] = assessment
        mode = assessment["disposition"]
        if mode == "clear":
            return None
        if ctx.working_memory.get("_safety_delivery_version") == "minimal_response_v1":
            # Historic, quoted or uncertain labels do not justify a crisis
            # replacement. Engine still isolates their actions and learning.
            if mode not in ("urgent_owner", "urgent_other"):
                return None
        elif mode != "urgent_owner":
            return RiskSignal(tier=2, kind="semantic_safety_" + mode, evidence=[], confidence=0.0)
        evidence = [trace["safety_evidence"]["quote"]]
    kind = {
        "self_harm": "semantic_self_harm_concern",
        "medical_emergency": "medical_emergency_unspecified",
        "fall": "medical_emergency_fall",
        "unsafe_medication": "semantic_unsafe_medication_concern",
    }.get(trace.get("safety_concern"))
    if kind is None:
        return None
    # No invented probability and no inferred observation. The evidence is the
    # actual current text; the saved trace explicitly records the model source.
    return RiskSignal(tier=2 if trace["safety_concern"] in {"self_harm", "unsafe_medication"} else 3,
                      kind=kind, evidence=evidence, confidence=0.0)
