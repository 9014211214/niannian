"""Same-call task projection: provenance only, never semantic safety proof.

The immutable candidate is ephemeral. Audits contain hashes and bounded codes,
not a second copy of the generated task or the user's private source text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

VERSION = "task_projection_v1"
INSTRUCTION_VERSION = "conditional_response_v2"
KEY = "_safety_task_projection_for_turn"


def task_projection_schema():
    return {"type": "object", "properties": {
        "version": {"type": "string", "enum": [VERSION]},
        "status": {"type": "string", "enum": ["none", "in_dialogue_task"]},
        "request_quote": {"type": "string", "maxLength": 240},
        "response_quote": {"type": "string", "maxLength": 2000},
    }, "required": ["version", "status", "request_quote", "response_quote"],
        "additionalProperties": False}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskProjection:
    user_id: str
    session_id: str
    scene_id: str
    turn_index: int
    utterance_sha256: str
    raw_response_sha256: str
    request_quote_sha256: str
    response_quote_sha256: str
    text: str
    semantic_verified: bool = False
    grants_permission: bool = False


def bind_task_projection(ctx, utterance, payload, trace):
    """Validate independent task provenance even when safety provenance fails."""
    ctx.working_memory.pop(KEY, None)
    audit = {"version": VERSION, "status": "rejected", "reason": "invalid_projection",
             "semantic_verified": False, "grants_permission": False}
    row = payload.get("task_projection")
    if (ctx.working_memory.get("_safety_response_instruction_version") != INSTRUCTION_VERSION
            or type(utterance) is not str or type(row) is not dict
            or set(row) != {"version", "status", "request_quote", "response_quote"}
            or row.get("version") != VERSION or type(row.get("status")) is not str
            or any(type(row.get(key)) is not str for key in ("request_quote", "response_quote"))):
        return audit
    source, text = row["request_quote"], row["response_quote"]
    if row["status"] == "none" and source == text == "":
        return {**audit, "status": "not_requested", "reason": "model_declared_none"}
    body = payload.get("response")
    if (row["status"] != "in_dialogue_task" or not source.strip() or not text.strip()
            or len(source) > 240 or len(text) > 2000 or source not in utterance
            or type(body) is not str or text not in body
            or trace.get("parsed_response_sha256") != digest(body.strip())
            or type(trace.get("turn_index")) is not int or trace["turn_index"] != len(ctx.turns)
            or trace.get("status") != "parsed"
            or trace.get("version") not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")):
        return audit
    # Action syntax and internal carriers cannot become a display-only task.
    # This is a protocol boundary, not a vocabulary-based semantic classifier.
    from ..policy.online_decision import is_internal_protocol_response
    if "<action" in text.lower() or "</action" in text.lower() or is_internal_protocol_response(text):
        return {**audit, "reason": "task_contains_internal_or_action_protocol"}
    frozen = TaskProjection(ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns),
                            digest(utterance), digest(body.strip()), digest(source), digest(text), text)
    ctx.working_memory[KEY] = frozen
    return {**audit, "status": "source_bound", "reason": "exact_request_and_response_quotes",
            "request_quote_sha256": frozen.request_quote_sha256,
            "response_quote_sha256": frozen.response_quote_sha256,
            "raw_response_sha256": frozen.raw_response_sha256}


def current_task_projection(ctx, utterance, persona):
    frozen = ctx.working_memory.get(KEY)
    trace = ctx.working_memory.get("online_turn_decision") or {}
    audit = trace.get("task_projection") or {}
    if (type(frozen) is not TaskProjection or type(utterance) is not str
            or ctx.working_memory.get("_safety_response_instruction_version") != INSTRUCTION_VERSION
            or ctx.working_memory.get("_online_decision_version") not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
            or trace.get("version") not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
            or trace.get("status") != "parsed" or type(trace.get("turn_index")) is not int
            or trace["turn_index"] != len(ctx.turns)
            or (persona is not None and persona.user_id != ctx.user_id)
            or (frozen.user_id, frozen.session_id, frozen.scene_id, frozen.turn_index) !=
               (ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns))
            or frozen.utterance_sha256 != digest(utterance)
            or frozen.raw_response_sha256 != trace.get("parsed_response_sha256")
            or frozen.response_quote_sha256 != digest(frozen.text)
            or audit.get("status") != "source_bound"
            or audit.get("request_quote_sha256") != frozen.request_quote_sha256
            or audit.get("response_quote_sha256") != frozen.response_quote_sha256
            or audit.get("raw_response_sha256") != frozen.raw_response_sha256
            or frozen.semantic_verified is not False or frozen.grants_permission is not False
            or audit.get("semantic_verified") is not False or audit.get("grants_permission") is not False):
        return None
    return frozen
