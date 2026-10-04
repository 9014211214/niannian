"""把可选模态输入归一成可追溯证据。

调用方可以只传文本/声音，也可以追加:

    multimodal={
        "visual": [
            {"signal": "gaze_avoidance", "value": 0.8,
             "confidence": 0.7, "quality": 0.9, "baseline_delta": 0.35}
        ],
        "behavioral": {
            "initiative_frequency": {
                "value": 0.1, "baseline_delta": -0.7, "confidence": 0.9
            }
        },
    }

缺失的模态不会生成占位证据，也不会被当作反证。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ...interfaces.reasoning import ObservationEvidence
from ..context import EmotionState
from .signals import unit_interval


def collect_evidence(
    utterance: str,
    emotion: EmotionState,
    *,
    turn_index: int,
    multimodal: Optional[Mapping[str, Any]] = None,
) -> list[ObservationEvidence]:
    """收集本轮文本、融合情绪与外部可选模态。"""
    evidence = [
        ObservationEvidence(
            id=f"t{turn_index}:text:utterance",
            modality="text",
            signal="utterance",
            value=utterance,
            source="user",
            confidence=1.0,
            quality=1.0,
            turn_index=turn_index,
        )
    ]

    if emotion.external_reading is None and (
        emotion.label not in ("neutral", "unknown") or emotion.divergence
    ):
        evidence.append(ObservationEvidence(
            id=f"t{turn_index}:affect:fused",
            modality="affect",
            signal="divergence" if emotion.divergence else "emotion",
            value=emotion.pattern if emotion.divergence else emotion.label,
            source=emotion.inference_source or "aria_fusion",
            confidence=emotion.score or (0.45 if emotion.divergence else 0.4),
            quality=1.0,
            turn_index=turn_index,
        ))

    # 文本模型提出的是“有原文锚点的候选解释”，不是新的感官观测。
    # 质量固定降权，后续必须与规则、个人基线、可选多模态和反证竞争。
    for index, signal in enumerate(emotion.semantic_hypotheses):
        evidence_turn = (
            signal.evidence_turn_index
            if signal.evidence_turn_index is not None
            else turn_index
        )
        evidence_turn = max(0, min(turn_index, evidence_turn))
        if signal.counter_only:
            evidence.append(ObservationEvidence(
                id=(
                    f"t{turn_index}:semantic:counter:{signal.id}:{index}:"
                    f"source{evidence_turn}"
                ),
                modality="semantic",
                signal=f"counter:{signal.id}",
                value=signal.evidence_quote,
                source=signal.source,
                confidence=signal.confidence,
                quality=0.80,
                detail=signal.rationale or "语境中的明确反证",
                turn_index=evidence_turn,
            ))
            continue
        evidence.append(ObservationEvidence(
            id=(
                f"t{turn_index}:semantic:{signal.id}:{index}:"
                f"source{evidence_turn}"
            ),
            modality="semantic",
            signal=signal.id,
            value=signal.evidence_quote,
            source=signal.source,
            confidence=signal.confidence,
            quality=0.80,
            detail=signal.rationale,
            turn_index=evidence_turn,
        ))
        if signal.counterevidence_quote:
            counter_turn = (
                signal.counterevidence_turn_index
                if signal.counterevidence_turn_index is not None
                else turn_index
            )
            counter_turn = max(0, min(turn_index, counter_turn))
            evidence.append(ObservationEvidence(
                id=(
                    f"t{turn_index}:semantic:counter:{signal.id}:{index}:"
                    f"source{counter_turn}"
                ),
                modality="semantic",
                signal=f"counter:{signal.id}",
                value=signal.counterevidence_quote,
                source=signal.source,
                confidence=signal.confidence,
                quality=0.80,
                detail="同一语境中的候选反证",
                turn_index=counter_turn,
            ))

    # acoustic 中保存的是引擎已见到的原始信号；只在确实存在时生成证据。
    acoustic = emotion.acoustic or {}
    if acoustic.get("acoustic_label") not in (None, "unknown"):
        evidence.append(ObservationEvidence(
            id=f"t{turn_index}:acoustic:emotion",
            modality="acoustic",
            signal="emotion",
            value=acoustic.get("acoustic_label"),
            source=str(acoustic.get("source") or "unknown"),
            fallback=bool(acoustic.get("fallback", False)),
            confidence=unit_interval(acoustic.get("acoustic_score")),
            quality=unit_interval(acoustic.get("acoustic_quality"), 1.0),
            detail=(
                "合成情绪观测，非真实声音识别"
                if acoustic.get("source_kind") == "simulated" else ""
            ),
            turn_index=turn_index,
        ))

    for modality, payload in (multimodal or {}).items():
        evidence.extend(_normalise_modality(str(modality), payload, turn_index))
    return evidence


def _normalise_modality(
    modality: str, payload: Any, turn_index: int
) -> list[ObservationEvidence]:
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and "signal" in payload:
        rows = [payload]
    elif isinstance(payload, Mapping):
        rows = []
        for signal, value in payload.items():
            if isinstance(value, Mapping):
                rows.append({"signal": signal, **value})
            else:
                rows.append({"signal": signal, "value": value})
    else:
        return []

    out: list[ObservationEvidence] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        signal = str(row.get("signal") or "").strip()
        value = row.get("value")
        if not signal or value is None:
            continue
        confidence = unit_interval(row.get("confidence"), 0.5)
        quality = unit_interval(row.get("quality"), 1.0)
        baseline_delta = row.get("baseline_delta")
        if baseline_delta is not None:
            baseline_delta = _number(baseline_delta, 0.0)
        out.append(ObservationEvidence(
            id=f"t{turn_index}:{modality}:{signal}:{index}",
            modality=modality,
            signal=signal,
            value=value,
            source=str(row.get("source") or modality),
            fallback=bool(row.get("fallback", False)),
            confidence=max(0.0, min(1.0, confidence)),
            quality=max(0.0, min(1.0, quality)),
            baseline_delta=baseline_delta,
            detail=str(row.get("detail") or ""),
            turn_index=turn_index,
        ))
    return out


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default
