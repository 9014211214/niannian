"""Explicit response-only continuation after an immediately emitted backstop.

This is a transport/state boundary, not semantic task admission. The same model
call proposes the task; exact quotations do not establish semantic correctness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256

LEGACY = "legacy"
TASK_RESPONSE_V1 = "task_response_v1"
TASK_RESPONSE_V2 = "task_response_v2"
TASK_RESPONSE_V3 = "task_response_v3"
TASK_RESPONSE_V4 = "task_response_v4"
PRELUDE_VERSION = "safety_prelude_v1"


@dataclass(frozen=True)
class DeliveredSafetyPrelude:
    """Exact event captured only after the trusted transport callback succeeds.

    This ephemeral receipt is a Python capability, never user-request metadata.
    Its text identifies the protection actually shown, not later generated text.
    """
    version: str
    premodel_safety_version: str
    user_id: str
    session_id: str
    scene_id: str
    turn_index: int
    utterance_sha256: str
    protection: str
    protection_sha256: str


def current_delivered_prelude(ctx, utterance, receipt):
    if type(receipt) is not DeliveredSafetyPrelude:
        return None
    if type(receipt.turn_index) is not int or receipt.turn_index < 0:
        return None
    try:
        expected = make_prelude(ctx, utterance, receipt.protection)
    except (TypeError, ValueError):
        return None
    return asdict(receipt) if asdict(receipt) == expected else None


def validate_premodel_safety_version(value):
    if type(value) is not str or value not in (LEGACY, TASK_RESPONSE_V1, TASK_RESPONSE_V2, TASK_RESPONSE_V3, TASK_RESPONSE_V4):
        raise ValueError("invalid_premodel_safety_version")
    return value


def validate_premodel_configuration(version, delivery, instruction, online):
    validate_premodel_safety_version(version)
    if version == TASK_RESPONSE_V1 and (
        delivery != "task_context_v2" or instruction != "conditional_response_v2"
        or online not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
    ):
        raise ValueError("premodel_task_requires_task_context_v2_response_v2_online_v4")

    if version == TASK_RESPONSE_V2 and (
        delivery != "source_response_v3" or instruction != "conditional_response_v3"
        or online not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
    ):
        raise ValueError("premodel_task_requires_source_response_v3_online_v4")

    if version == TASK_RESPONSE_V3 and (
        delivery != "source_response_v4" or instruction != "conditional_response_v4"
        or online not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
    ):
        raise ValueError("premodel_task_requires_source_response_v4_online_v4")

    if version == TASK_RESPONSE_V4 and (
        delivery != "minimal_response_v1" or instruction != "compact_response_v1"
        or online not in ("online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
    ):
        raise ValueError("premodel_task_requires_minimal_response_v1_online_v4")


def make_prelude(ctx, utterance, protection):
    if (type(protection) is not str or not 1 <= len(protection) <= 4096
            or type(utterance) is not str
            or any(type(value) is not str or not value
                   for value in (ctx.user_id, ctx.session_id, ctx.scene_id))):
        raise ValueError("invalid_safety_prelude_identity")
    return {"version": PRELUDE_VERSION, "premodel_safety_version": ctx.working_memory.get("_premodel_safety_version", TASK_RESPONSE_V1),
            "user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
            "turn_index": len(ctx.turns), "utterance_sha256": sha256(utterance.encode()).hexdigest(),
            "protection": protection, "protection_sha256": sha256(protection.encode()).hexdigest()}


def response_only_context(ctx, persona):
    """Detach all mutable data; retain actual history and privacy boundaries.

    No scene observer, Skill, memory, action or profile commit is invoked on this
    path. Removed carriers cannot advertise a pending authority to the model.
    """
    detached, profile = deepcopy(ctx), deepcopy(persona)
    detached.active_tasks = []
    detached.reasoning.hypotheses = []
    detached.reasoning.candidates = []
    detached.reasoning.selected = None
    detached.reasoning.observation_plan = None
    for key in (
        "_pending_external_action", "_preference_dialogue_invitation",
        "_cognitive_text_readiness", "_cognitive_readiness_dispatch",
        "_cognitive_readiness_prepared", "_cognitive_block_selected",
        "_local_preference_preview", "last_care_action", "last_tool_result",
        "_consumed_tool_result_for_turn", "skill_proposals", "personality_decision",
        "domain_knowledge", "recalled_episodes", "recalled_episode_sources", "world_knowledge",
    ):
        detached.working_memory.pop(key, None)
    return detached, profile


def transfer_delivery_audits(source, target):
    """Return only current delivery evidence; never model controls or learning."""
    trace = source.working_memory.get("online_turn_decision")
    if type(trace) is dict:
        keys = ("version", "status", "turn_index", "parsed_response_sha256",
                "safety_concern", "safety_source", "safety_evidence",
                "safety_assessment", "task_projection", "source_response")
        target.working_memory["online_turn_decision"] = {
            key: deepcopy(trace[key]) for key in keys if key in trace}
    for key in ("model_route", "prompt_evidence_snapshot", "safety_response_instruction",
                "safety_delivery", "delivery_response_guard", "_reply_delivery_stages",
                "safety_review",
                "_safety_task_projection_for_turn", "_source_response_for_turn", "_benchmark_reply_trace"):
        if key in source.working_memory:
            target.working_memory[key] = deepcopy(source.working_memory[key])
