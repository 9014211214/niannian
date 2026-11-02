"""Ephemeral provenance for a complete same-call response, never semantic proof.

This candidate only establishes which received body and assessment belong to
this turn. Every delivered body still passes the normal response/scene/receipt
guards. Neither a model assessment nor a source quote grants any action power.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

VERSION = "source_response_v3"
INSTRUCTION_VERSION = "conditional_response_v3"
VERSION_V4 = "source_response_v4"
INSTRUCTION_VERSION_V4 = "conditional_response_v4"
MINIMAL_VERSION = "minimal_response_v1"
COMPACT_INSTRUCTION = "compact_response_v1"
KEY = "_source_response_for_turn"
ONLINE = "online_turn_decision_v4_evidence"
ONLINE_VERSIONS = (ONLINE, "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceResponse:
    user_id: str
    session_id: str
    scene_id: str
    turn_index: int
    utterance_sha256: str
    raw: str
    raw_sha256: str
    guarded: str
    guarded_sha256: str
    assessment_sha256: str
    version: str = VERSION
    instruction_version: str = INSTRUCTION_VERSION


def _assessment_digest(trace, utterance, *, independent_body=False):
    import json
    from .safety_evidence import assess_evidence
    assessment = assess_evidence(trace.get("safety_concern"), trace.get("safety_evidence"), utterance)
    if (trace.get("safety_source") != "online_model_semantics"
            or trace.get("safety_assessment") != assessment
            or (not independent_body and assessment.get("reason") not in {
                "model_none_structurally_consistent", "model_current_immediate_with_source",
                "current_subject_or_urgency_not_established"})):
        return None
    return digest(json.dumps({"concern": trace.get("safety_concern"),
        "evidence": trace.get("safety_evidence"), "assessment": assessment},
        sort_keys=True, ensure_ascii=False))


def bind_source_response(ctx, utterance, raw, guarded, trace):
    ctx.working_memory.pop(KEY, None)
    instruction = ctx.working_memory.get("_safety_response_instruction_version")
    independent_body = instruction in (INSTRUCTION_VERSION_V4, COMPACT_INSTRUCTION)
    version = MINIMAL_VERSION if instruction == COMPACT_INSTRUCTION else VERSION_V4 if independent_body else VERSION
    audit = {"version": version, "status": "rejected", "semantic_verified": False,
             "grants_permission": False}
    from ..policy.online_decision import is_internal_protocol_response
    assessment = _assessment_digest(trace, utterance, independent_body=independent_body)
    if (instruction not in (INSTRUCTION_VERSION, INSTRUCTION_VERSION_V4, COMPACT_INSTRUCTION)
            or trace.get("version") not in ONLINE_VERSIONS or trace.get("status") != "parsed"
            or type(trace.get("turn_index")) is not int or trace["turn_index"] != len(ctx.turns)
            or type(raw) is not str or type(guarded) is not str or not raw.strip() or not guarded.strip()
            or trace.get("parsed_response_sha256") != digest(raw)
            or assessment is None
            or (trace.get("version") != "online_turn_decision_v7" and is_internal_protocol_response(raw))
            or "<action" in raw.lower() or "</action" in raw.lower()):
        return audit
    frozen = SourceResponse(ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns),
                            digest(utterance), raw, digest(raw), guarded, digest(guarded), assessment, version, instruction)
    ctx.working_memory[KEY] = frozen
    return {**audit, "status": "source_bound", "raw_response_sha256": frozen.raw_sha256,
            "guarded_response_sha256": frozen.guarded_sha256,
            "assessment_sha256": assessment}


def current_source_response(ctx, utterance, persona):
    frozen = ctx.working_memory.get(KEY)
    instruction = ctx.working_memory.get("_safety_response_instruction_version")
    version = MINIMAL_VERSION if instruction == COMPACT_INSTRUCTION else VERSION_V4 if instruction == INSTRUCTION_VERSION_V4 else VERSION
    trace = ctx.working_memory.get("online_turn_decision") or {}
    audit = trace.get("source_response") or {}
    if (type(frozen) is not SourceResponse
            or ctx.working_memory.get("_safety_delivery_version") != version
            or instruction not in (INSTRUCTION_VERSION, INSTRUCTION_VERSION_V4, COMPACT_INSTRUCTION)
            or ctx.working_memory.get("_online_decision_version") not in ONLINE_VERSIONS
            or trace.get("version") not in ONLINE_VERSIONS or trace.get("status") != "parsed"
            or type(trace.get("turn_index")) is not int or trace["turn_index"] != len(ctx.turns)
            or (persona is not None and persona.user_id != ctx.user_id)
            or (frozen.user_id, frozen.session_id, frozen.scene_id, frozen.turn_index) !=
               (ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns))
            or frozen.version != version or frozen.instruction_version != instruction
            or frozen.utterance_sha256 != digest(utterance)
            or frozen.raw_sha256 != digest(frozen.raw)
            or frozen.guarded_sha256 != digest(frozen.guarded)
            or frozen.raw_sha256 != trace.get("parsed_response_sha256")
            or frozen.assessment_sha256 != _assessment_digest(trace, utterance, independent_body=version in (VERSION_V4, MINIMAL_VERSION))
            or audit != {"version": version, "status": "source_bound", "semantic_verified": False,
                         "grants_permission": False, "raw_response_sha256": frozen.raw_sha256,
                         "guarded_response_sha256": frozen.guarded_sha256,
                         "assessment_sha256": frozen.assessment_sha256}):
        return None
    return deepcopy(frozen)


def attributed_current_quote(text, utterance):
    """Mask only an entire current message inside an explicit attribution.

    No paraphrase entailment or actor/action-category inference is performed.
    Exact whole-message equality preserves negation, questions and conditions.
    Adjacent claims stay outside the mask and remain subject to receipt checks.
    """
    if (type(utterance) is not str or not utterance.strip()
            or any(mark in utterance for mark in ("“", "”"))):
        return text
    token = "你提供的原文：“" + utterance + "”"
    return text.replace(token, "（当前用户原文逐字引用）")
