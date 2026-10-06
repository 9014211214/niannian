"""Map a validated same-turn model signal to deterministic safety handling.

This is a routing hint, not a diagnosis or permission to contact another person.
The keyword monitor still runs first and cannot be downgraded by this signal.
"""
from ...interfaces.risk import RiskSignal
from ..policy.online_decision import ONLINE_DECISION_VERSION_SAFETY


def semantic_safety_signal(ctx, utterance):
    trace = ctx.working_memory.get("online_turn_decision") or {}
    if (trace.get("version") != ONLINE_DECISION_VERSION_SAFETY
            or trace.get("status") != "parsed"
            or trace.get("turn_index") != len(ctx.turns)
            or trace.get("safety_source") != "online_model_semantics"):
        return None
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
                      kind=kind, evidence=[utterance], confidence=0.0)
