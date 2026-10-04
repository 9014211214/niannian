"""Finite reliability values and explicit turn binding for optional observations."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any, Mapping


def unit_interval(value: Any, default: float = 0.0) -> float:
    """Missing values may use a legacy default; malformed values fail closed."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else 0.0


def is_simulated(value: Any) -> bool:
    if isinstance(value, Mapping):
        return (
            value.get("source_kind") == "simulated"
            # Legacy Web story buttons used synthetic_demo_timeline.  Treat
            # that explicit demo namespace like the newer simulated_* signals,
            # without classifying unrelated synthetic_* providers as demos.
            or str(value.get("source", "")).startswith((
                "simulated_", "synthetic_demo_",
            ))
            or "simulation_id" in value
            or any(is_simulated(item) for item in value.values())
        )
    return isinstance(value, list) and any(is_simulated(item) for item in value)


def observation_rejections(
    observation: Mapping[str, Any], *, turn_index: int,
    turn_id: str | None = None, now: datetime | None = None,
) -> list[str]:
    """Absent legacy clocks are allowed; supplied clocks must match/be current."""
    reasons = []
    index = observation.get("turn_index")
    if "turn_index" in observation and (
        isinstance(index, bool) or not isinstance(index, int) or index != turn_index
    ):
        reasons.append("turn_index_mismatch")
    if "turn_id" in observation and turn_id is not None and observation["turn_id"] != turn_id:
        reasons.append("turn_id_mismatch")
    if is_simulated(observation) and (
        index is None or not observation.get("turn_id")
    ):
        reasons.append("simulation_turn_binding_missing")
    clock = now or datetime.now(timezone.utc)
    for field in ("observed_at", "expires_at"):
        if field not in observation:
            continue
        try:
            timestamp = datetime.fromisoformat(str(observation[field]))
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
            if field == "expires_at" and timestamp <= clock:
                reasons.append("expired")
            if field == "observed_at":
                age = (clock - timestamp).total_seconds()
                if age < -5 or age > 120:
                    reasons.append("observation_not_current")
        except (ValueError, TypeError):
            reasons.append("invalid_timestamp")
    return reasons


def prepare_acoustic(
    acoustic: Mapping[str, Any] | None, *, turn_index: int,
    turn_id: str | None = None,
) -> dict[str, Any] | None:
    if acoustic is None:
        return None
    if not isinstance(acoustic, Mapping):
        raise ValueError("acoustic observation must be a mapping")
    result = deepcopy(dict(acoustic))
    reasons = observation_rejections(result, turn_index=turn_index, turn_id=turn_id)
    result["score"] = unit_interval(result.get("score"))
    result["quality_known"] = "quality" in result and result["quality"] is not None
    result["quality"] = unit_interval(result.get("quality"), 1.0)
    for child_key in ("emotion_trace", "prosodic"):
        child = result.get(child_key)
        if isinstance(child, dict):
            reasons.extend(
                f"{child_key}:{reason}"
                for reason in observation_rejections(child, turn_index=turn_index, turn_id=turn_id)
            )
    if is_simulated(result):
        if (
            result.get("source") != "simulated_acoustic"
            or result.get("source_kind") != "simulated"
            or result.get("simulation_version") != "acoustic_simulation_v1"
            or not result.get("simulation_id")
            or not result["quality_known"]
        ):
            reasons.append("invalid_simulation_provenance")
        distribution = result.get("all_scores")
        if not isinstance(distribution, dict) or not distribution:
            reasons.append("invalid_label_distribution")
        else:
            scores = {key: unit_interval(value) for key, value in distribution.items()}
            label = result.get("label")
            if (
                any(scores[key] != value for key, value in distribution.items())
                or not isinstance(label, str)
                or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-6)
                or not math.isclose(scores.get(label, -1), result["score"], abs_tol=1e-6)
                or max(scores.values()) > result["score"] + 1e-6
            ):
                reasons.append("invalid_label_distribution")
    if isinstance(result.get("all_scores"), dict):
        result["all_scores"] = {
            str(key): unit_interval(value)
            for key, value in result["all_scores"].items()
        }
    if reasons:
        result["quality"] = 0.0
        result["prosodic"] = {}
        result["emotion_trace"] = {}
    result["rejection_reasons"] = sorted(set(reasons))
    return result


def prepare_multimodal(
    multimodal: Mapping[str, Any] | None, *, turn_index: int,
    turn_id: str | None = None,
) -> dict[str, Any] | None:
    if not multimodal:
        return None
    result = deepcopy(dict(multimodal))
    for payload in result.values():
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "signal" in payload:
            rows = [payload]
        elif isinstance(payload, dict):
            rows = list(payload.values())
        else:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            reasons = observation_rejections(row, turn_index=turn_index, turn_id=turn_id)
            if isinstance(row.get("value"), (int, float)) and not isinstance(row["value"], bool):
                try:
                    valid_number = math.isfinite(float(row["value"]))
                except (ValueError, OverflowError):
                    valid_number = False
                if not valid_number:
                    row["value"] = None
                    reasons.append("invalid_numeric_observation")
            if row.get("baseline_delta") is not None:
                try:
                    delta = float(row["baseline_delta"])
                    if not math.isfinite(delta) or isinstance(row["baseline_delta"], bool) or abs(delta) > 1:
                        raise ValueError("invalid baseline delta")
                    row["baseline_delta"] = delta
                except (ValueError, TypeError, OverflowError):
                    row.pop("baseline_delta", None)
                    reasons.append("invalid_baseline_delta")
            if reasons:
                row["quality"] = 0.0
                row["rejection_reasons"] = reasons
            elif "quality" in row:
                row["quality"] = unit_interval(row["quality"])
            if "confidence" in row:
                row["confidence"] = unit_interval(row["confidence"])
    return result
