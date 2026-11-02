"""Auditable product wait decisions; no transport, account access or permission.

A Retry-After hint does not establish zero cost or non-execution. Callers must
first apply their existing receipt, permission and budget checks.
"""
from __future__ import annotations

import hashlib
import json
import math

from .response_observability import validate_observation
from .request_pacing import monotonic_value, wait_until

SCHEMA = "product_retry_wait_v1"


def retry_wait_plan(observation, *, retry_index, unknown_429_hold=False):
    validate_observation(observation)
    if type(retry_index) is not int or retry_index not in {0, 1}:
        raise ValueError("invalid_product_retry_index")
    if type(unknown_429_hold) is not bool:
        raise ValueError("invalid_unknown_429_wait_mode")
    minimum = 900 if unknown_429_hold else 1 + retry_index
    maximum = 3600 if unknown_429_hold else 30
    status = observation.get("retry_after_status", "absent")
    seconds = observation.get("retry_after_seconds", minimum)
    reason = ("retry_after_rejected" if status == "rejected" else
              "retry_after_exceeds_wait_budget" if seconds > maximum else None)
    return {"schema": "product_unknown_429_wait_v1" if unknown_429_hold else SCHEMA, "retry_index": retry_index,
            "decision": "blocked" if reason else "wait", "reason": reason,
            "seconds": None if reason else max(minimum, seconds),
            "observation_sha256": hashlib.sha256(json.dumps(observation, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def perform_retry_wait(plan, *, monotonic, sleep, cancelled):
    if plan["decision"] != "wait":
        raise ValueError("product_retry_wait_blocked")
    seconds = plan["seconds"]
    minimum, maximum = ((900, 3600) if plan.get("schema") == "product_unknown_429_wait_v1" else (1, 30))
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not minimum <= seconds <= maximum:
        raise ValueError("invalid_product_retry_wait")
    started = monotonic_value(monotonic)
    # wait_until has its own initial clock read. Its elapsed value excludes
    # the interval between that read and our deadline origin, so it cannot
    # certify this plan's total duration.
    wait_until(started + seconds, monotonic=monotonic, sleep=sleep, cancelled=cancelled)
    finished = monotonic_value(monotonic)
    if cancelled():
        raise ValueError("pacing_cancelled")
    if finished < started + seconds:
        raise ValueError("pacing_wait_not_satisfied")
    # The deadline comparison above is authoritative. Binary subtraction
    # can round a satisfied 2 s interval to 1999.9999999999998 ms; do not
    # persist an apparently short duration after proving the full deadline.
    return max(seconds * 1000, (finished - started) * 1000)
