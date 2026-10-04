"""Explicit, deterministic synthetic acoustic observations for isolated research.

These fixtures exercise conditional reasoning, not audio recognition. No audio,
network request, expected action, person hypothesis or judge answer is generated.
Scenario titles stay in the selector; only sensor-shaped values enter the engine.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SIMULATION_VERSION = "acoustic_simulation_v1"
SIMULATION_SOURCE = "simulated_acoustic"
_LABELS = (
    "angry", "happy", "sad", "fearful", "disgusted", "surprised",
    "neutral", "other", "unknown",
)
# Values are controlled test stimuli, not clinically calibrated voice profiles.
_SCENARIOS = {
    "neutral": ("平静", "中性声学观测", "neutral", .86, .90, 4.0, .12),
    "happy": ("开心", "积极声学观测", "happy", .86, .90, 4.8, .10),
    "sad": ("低落", "低落声学观测", "sad", .86, .90, 2.4, .40),
    "angry": ("烦躁", "愤怒声学观测", "angry", .86, .90, 5.0, .16),
    "fearful": ("紧张", "恐惧或紧张声学观测", "fearful", .86, .90, 4.4, .30),
    "disgusted": ("厌烦", "厌恶声学观测", "disgusted", .86, .90, 3.6, .22),
    "surprised": ("惊讶", "惊讶声学观测", "surprised", .86, .90, 4.6, .20),
    "unknown": ("无法归类", "有输入但情绪分类未知", "unknown", .86, .90, 4.0, .12),
    "low_confidence": ("低置信", "分类不确定，不能视为确定的低落", "sad", .22, .90, 4.0, .12),
    "low_quality": ("低质量", "质量差的低落信号，不应主导判断", "sad", .90, .08, 2.4, .40),
    "late_distress": ("末段低落", "前段平静、末段低落的合成轨迹", "neutral", .82, .90, 3.2, .32),
    "recovery": ("逐渐平静", "由低落恢复平静的合成轨迹", "neutral", .86, .90, 4.0, .12),
}


def list_emotion_scenarios() -> list[dict[str, str]]:
    return [
        {"id": key, "title": row[0], "description": row[1]}
        for key, row in _SCENARIOS.items()
    ]


def build_simulated_acoustic(
    scenario_id: str, *, turn_id: str, turn_index: int,
) -> dict[str, Any]:
    """Build a fresh value; the same arguments always yield identical input.

    No wall-clock timestamp is fabricated. The caller binds this observation to
    a server-owned turn ID and index. Optional actual capture timestamps belong
    to real adapters, not a deterministic synthetic fixture.
    """
    if scenario_id not in _SCENARIOS:
        raise ValueError("unknown emotion simulation scenario")
    if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 128:
        raise ValueError("simulation requires a nonempty turn_id up to 128 chars")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("simulation requires a nonnegative integer turn_index")
    _, _, label, score, quality, pace, pause = _SCENARIOS[scenario_id]
    distribution = {
        item: score if item == label else (1.0 - score) / (len(_LABELS) - 1)
        for item in _LABELS
    }
    observation = {
        "label": label,
        "score": score,
        "quality": quality,
        "all_scores": distribution,
        "source": SIMULATION_SOURCE,
        "source_kind": "simulated",
        "simulation_version": SIMULATION_VERSION,
        # Opaque provenance, not the human-readable scenario or a gold label.
        "simulation_id": f"s{tuple(_SCENARIOS).index(scenario_id) + 1:02d}",
        "turn_id": turn_id,
        "turn_index": turn_index,
        "fallback": False,
        "prosodic": {
            "speech_pace": pace, "pause_ratio": pause,
            "volume": .08, "volume_envelope": "stable",
            "confidence": .80, "quality": quality,
        },
    }
    if scenario_id in {"late_distress", "recovery"}:
        observation["emotion_trace"] = {
            "pattern": scenario_id,
            "trace": (
                ["neutral", "neutral", "sad"]
                if scenario_id == "late_distress"
                else ["sad", "neutral", "neutral"]
            ),
            "confidence": .82, "quality": quality,
        }
    return deepcopy(observation)


def simulated_prosodic_observations(acoustic: dict[str, Any]) -> dict[str, Any]:
    """Convert only explicit simulated prosody; never relabel real recordings."""
    if acoustic.get("source_kind") != "simulated":
        raise ValueError("expected explicit simulated acoustic observation")
    prosodic = acoustic.get("prosodic") or {}
    rows = {
        signal: {
            "value": prosodic[signal],
            "confidence": prosodic.get("confidence", 0.0),
            "quality": min(prosodic.get("quality", 0.0), acoustic.get("quality", 0.0)),
            "source": "simulated_prosody",
            "source_kind": "simulated",
            "turn_id": acoustic.get("turn_id"),
            "turn_index": acoustic.get("turn_index"),
            "detail": "合成节律观测，非真实音频测量",
        }
        for signal in ("speech_pace", "pause_ratio", "volume")
        if signal in prosodic
    }
    return {"prosodic": rows} if rows else {}
