"""Record visible-response transformations without model reasoning or plaintext."""
from __future__ import annotations

import hashlib
import re


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_reply_stage(ctx, stage: str, before: str, after: str, *, violations=()) -> None:
    """Append current-turn evidence; a later unchanged stage cannot erase it."""
    index = len(ctx.turns)
    record = ctx.working_memory.get("_reply_delivery_stages")
    if not isinstance(record, dict) or record.get("turn_index") != index:
        record = {"turn_index": index, "stages": []}
        ctx.working_memory["_reply_delivery_stages"] = record
    reasons = list(dict.fromkeys(value for value in violations
                                if isinstance(value, str) and re.fullmatch(r"[a-z_]{1,80}", value)))[:16]
    record["stages"].append({
        "stage": stage, "activated": bool(reasons) or before != after,
        "response_changed": before != after, "violations": reasons,
        "before_sha256": _digest(before), "after_sha256": _digest(after),
    })


def delivery_protection(ctx) -> dict:
    """Project every observed stage, separately from source/safety routing."""
    record = ctx.working_memory.get("_reply_delivery_stages") or {}
    # Called after Engine.add_turn. Older/missing traces must not masquerade as
    # proof of an unchanged new response (notably the early safety path).
    stages = list(record.get("stages") or []) if record.get("turn_index") == len(ctx.turns) - 1 else []
    reasons = list(dict.fromkeys(value for stage in stages for value in stage["violations"]))
    return {
        "version": "delivery_protection_v2",
        "status": "observed" if stages else "not_observed",
        "activated": any(stage["activated"] for stage in stages),
        "response_changed": any(stage["response_changed"] for stage in stages),
        "violations": reasons[:16],
        "stages": [dict(stage) for stage in stages],
        "model_response_sha256": stages[0]["before_sha256"] if stages else None,
        "delivered_response_sha256": stages[-1]["after_sha256"] if stages else None,
    }
