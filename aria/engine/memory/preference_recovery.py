"""Renew the exact displayed proposal after a validated unclear transaction.

No text classification or proposal editing happens here. The first hook is
called only inside the existing fully validated ``action == unclear`` branch;
the second runs after that owner turn has actually entered EngineContext.
Neither hook acknowledges display, confirms a preference, or saves a profile.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from aria.engine.memory.preference_candidate import enabled as scoped_enabled
from aria.engine.memory.preference_dialogue import (
    _KEY, _REQUEST_KEY, _sha, current_preference_invitation,
    preference_fingerprint, preference_invitation_view,
    prepare_preference_invitation,
)
from aria.engine.memory.stated_preferences import preference_admission

VERSION = "preference_unclear_recovery_v1"
CONTEXT_KEY = "_preference_unclear_recovery_version"
TICKET_KEY = "_preference_unclear_recovery_ticket"
RECEIPT_KEY = "_preference_unclear_recovery_receipt"


def enabled(ctx):
    return ctx.working_memory.get(CONTEXT_KEY) == VERSION and scoped_enabled(ctx)



def validate_preference_recovery_client(client):
    """An explicit fresh online client capability; never read from user data."""
    from .preference_candidate import validate_preference_candidate_client
    if (getattr(client, "preference_recovery_version", None) != VERSION
            or client.role != "online_fast"
            or getattr(client, "online_decision_version", None) not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")):
        raise ValueError("invalid_preference_recovery_client")
    validate_preference_candidate_client(client)
    return {"version": VERSION, "target": "unchanged_displayed_proposal",
            "fresh_display_ack_required": True, "extends_original_expiry": False,
            "automatic_confirmation": False, "semantic_verified": False}


def bind_context(client, ctx):
    """Bind on every dispatch from the trusted client and discard stale tickets."""
    ctx.working_memory.pop(CONTEXT_KEY, None)
    ctx.working_memory.pop(TICKET_KEY, None)
    if getattr(client, "preference_recovery_version", None) is None:
        return
    validate_preference_recovery_client(client)
    if not scoped_enabled(ctx):
        raise ValueError("preference_recovery_protocol_mismatch")
    ctx.working_memory[CONTEXT_KEY] = VERSION


def stage_validated_unclear(persona, ctx, pending, utterance):
    """Internal hook after the old validator accepted precisely ``unclear``.

    Keep only the already displayed transaction and a hash of the current owner
    answer. This ticket is not a new invitation and conveys no confirmation.
    """
    ctx.working_memory.pop(TICKET_KEY, None)
    if not enabled(ctx):
        return
    if current_preference_invitation(persona, ctx) != pending:
        return
    binding = ctx.working_memory.get(_REQUEST_KEY)
    if binding != {"invitation": pending, "owner_message_sha256": _sha(utterance)}:
        return
    ctx.working_memory[TICKET_KEY] = {
        "version": VERSION, "invitation": deepcopy(pending),
        "owner_message_sha256": _sha(utterance),
    }


def complete_unclear_turn(persona, ctx, *, now=None):
    """Issue at most one fresh, unacknowledged invitation for the same row.

    Call once immediately after ctx.add_turn on the preference response path,
    under the same owner/session lock. Invalid or replayed tickets are consumed.
    A new UI display acknowledgement is always necessary. The original expiry
    is retained, so this mechanical continuation never extends permission TTL.
    """
    ticket = ctx.working_memory.pop(TICKET_KEY, None)
    if not enabled(ctx) or type(ticket) is not dict or set(ticket) != {
        "version", "invitation", "owner_message_sha256"
    } or ticket.get("version") != VERSION:
        return None
    pending = ticket.get("invitation")
    if type(pending) is not dict or set(pending) != {
        "version", "id", "owner_id", "session_id", "turn_index", "target_id",
        "target_sha256", "question", "question_sha256", "expires_at", "delivered"
    }:
        return None
    clock = now or datetime.now(timezone.utc)
    try:
        expiry = datetime.fromisoformat(pending["expires_at"])
    except (TypeError, ValueError):
        return None
    origin = pending.get("turn_index")
    trace = ctx.working_memory.get("online_turn_decision") or {}
    control = trace.get("preference_control") or {}
    row = next((item for item in persona.stated_preferences
                if item.id == pending.get("target_id")), None) if persona else None
    binding = ctx.working_memory.get(_REQUEST_KEY)
    if (persona is None or ctx.simulation_mode or _KEY in ctx.working_memory
            or pending.get("version") != "preference_confirmation_v1"
            or pending.get("delivered") is not True
            or type(origin) is not int or origin < 0 or len(ctx.turns) != origin + 1
            or pending.get("owner_id") != ctx.user_id or ctx.user_id != persona.user_id
            or pending.get("session_id") != ctx.session_id
            or clock.tzinfo is None or expiry.tzinfo is None or clock >= expiry
            or row is None or row.status != "proposed"
            or preference_admission(row) != "model_reviewed"
            or preference_fingerprint(row) != pending.get("target_sha256")
            or _sha(pending.get("question")) != pending.get("question_sha256")
            or _sha(ctx.turns[origin].user) != ticket.get("owner_message_sha256")
            or binding != {"invitation": pending,
                           "owner_message_sha256": ticket.get("owner_message_sha256")}
            or trace.get("version") not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")
            or trace.get("status") != "parsed"
            or type(trace.get("turn_index")) is not int or trace["turn_index"] != origin
            or trace.get("preference_control_status") != "validated"
            or trace.get("safety_source") != "online_model_semantics"
            or trace.get("safety_concern") != "none"
            or trace.get("owner_message_sha256") != ticket.get("owner_message_sha256")
            or control != {"version": "preference_confirmation_v1", "action": "unclear",
                "invitation_id": pending.get("id"), "target_id": row.id,
                "target_sha256": pending.get("target_sha256"),
                "evidence_quote": ctx.turns[origin].user}
            or (trace.get("version") != "online_turn_decision_v7"
                and origin not in ctx.working_memory.get("_sensitive_memory_excluded_turns", []))):
        return None
    # Re-check the existing structural safety result, never classify owner text.
    from aria.engine.risk.safety_evidence import assess_evidence
    if assess_evidence(trace.get("safety_concern"), trace.get("safety_evidence"),
                       ctx.turns[origin].user)["disposition"] != "clear":
        return None
    prepare_preference_invitation(persona, ctx, row.id, actor=persona.user_id, now=clock)
    fresh = ctx.working_memory[_KEY]
    if (fresh["id"] == pending["id"] or fresh["target_sha256"] != pending["target_sha256"]
            or fresh["question_sha256"] != pending["question_sha256"]):
        ctx.working_memory.pop(_KEY, None)
        return None
    fresh["expires_at"] = pending["expires_at"]
    # Only the normal trusted UI acknowledgement may change delivered to true.
    ctx.working_memory[RECEIPT_KEY] = {
        "version": VERSION, "status": "awaiting_display", "source_action": "unclear",
        "source_invitation_id": pending["id"], "invitation_id": fresh["id"],
        "owner_id": persona.user_id, "session_id": ctx.session_id,
        "source_turn_index": origin, "turn_index": len(ctx.turns),
        "target_id": row.id, "target_sha256": fresh["target_sha256"],
        "question_sha256": fresh["question_sha256"], "expires_at": fresh["expires_at"],
        "profile_changed": False, "confirmed": False, "semantic_verified": False,
    }
    return preference_invitation_view(persona, ctx, now=clock)
