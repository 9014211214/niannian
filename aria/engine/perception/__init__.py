"""Bounded text affect and evidence handling; no observation service clients."""
from __future__ import annotations

from typing import Any, Optional

from ..context import EmotionState
from ...interfaces.reasoning import SemanticStateSignal
from .fusion import FusedEmotion, fuse
from .gating import (
    StateAnalysisRoute,
    needs_deep_analysis,
    needs_integrated_decision,
    route_state_analysis,
)
from .evidence import collect_evidence
from .longitudinal import (
    enrich_with_personal_baseline,
)

__all__ = [
    "PerceptionEngine", "FusedEmotion", "fuse", "needs_deep_analysis",
    "needs_integrated_decision",
    "StateAnalysisRoute", "route_state_analysis",
    "collect_evidence",
    "enrich_with_personal_baseline",
]


class PerceptionEngine:
    def perceive(
        self,
        utterance: str,
        acoustic: Optional[dict[str, Any]] = None,
        text: Optional[dict[str, Any]] = None,
        history=None,
        *,
        prosodic: Optional[dict] = None,
        emotion_trace: Optional[dict] = None,
        analysis_route: StateAnalysisRoute | None = None,
    ) -> EmotionState:
        """产出本轮情绪状态与有原文依据的语义状态候选。

        - 有声学 / 文本情绪信号(语音模式 / bench)→ 走 fusion(声学×文本分歧)。
        - 复杂表达交由同一次在线决策解析；没有单独的文本探针调用。
        """
        semantic = None
        route = analysis_route or route_state_analysis(
            utterance,
            history,
            acoustic=acoustic,
            text=text,
        )
        if route.tier == "structured":
            semantic = _structured_emotion_state(
                utterance, len(history or [])
            )
        if acoustic is None and text is None:
            return semantic or EmotionState(label="unknown")

        fused: FusedEmotion = fuse(
            acoustic, text, prosodic=prosodic, emotion_trace=emotion_trace
        )
        semantic = semantic or EmotionState(label="unknown")
        acoustic_observed = fused.acoustic_label != "unknown"
        use_fused_affect = acoustic_observed or fused.divergence or fused.text_label != "unknown"
        label = (
            fused.primary_label
            if use_fused_affect
            else semantic.label
        )
        divergence = fused.divergence or semantic.divergence
        pattern = (
            fused.divergence_pattern
            if fused.divergence
            else semantic.pattern
        )
        score = (
            fused.primary_score
            if use_fused_affect
            else semantic.score
        )
        source = (
            "multimodal_fusion"
            if fused.divergence or acoustic_observed
            else semantic.inference_source
        )
        if (acoustic or {}).get("source_kind") == "simulated" and (
            fused.divergence or acoustic_observed
        ):
            source = "simulated_multimodal_fusion"
        elif not acoustic_observed and fused.text_label != "unknown":
            source = "provided_text_emotion"
        return EmotionState(
            label=label,
            divergence=divergence,
            pattern=pattern,
            score=score,
            acoustic={
                "acoustic_label": fused.acoustic_label,
                "acoustic_score": fused.acoustic_score,
                "acoustic_quality": fused.acoustic_quality,
                "acoustic_effective_score": fused.acoustic_effective_score,
                "quality_known": bool((acoustic or {}).get("quality_known", "quality" in (acoustic or {}))),
                "text_label": fused.text_label,
                "text_score": fused.text_score,
                "prosodic": fused.prosodic,
                "emotion_trace": fused.emotion_trace,
                "source": (acoustic or {}).get("source", ""),
                "source_kind": (acoustic or {}).get("source_kind", "unspecified"),
                "simulation_version": (acoustic or {}).get("simulation_version"),
                "simulation_id": (acoustic or {}).get("simulation_id"),
                "turn_id": (acoustic or {}).get("turn_id"),
                "turn_index": (acoustic or {}).get("turn_index"),
                "rejection_reasons": list((acoustic or {}).get("rejection_reasons") or []),
                "fusion_strategy": fused.fusion_strategy,
                "fallback": bool((acoustic or {}).get("fallback", False)),
                "attempted_sources": list(
                    (acoustic or {}).get("attempted_sources") or []
                ),
            },
            rupture=semantic.rupture,
            inference_source=source,
            semantic_hypotheses=semantic.semantic_hypotheses,
        )


def _structured_emotion_state(
    utterance: str,
    turn_index: int,
) -> EmotionState:
    """仅处理门控已确认的“报平安 + 当前参与变化”证据对。

    这是可审计的低置信候选，不是对隐藏情绪的确定判断。更隐晦的表达仍由状态
    在线快速模型的单次综合决策提出竞争解释。
    """
    return EmotionState(
        label="可能的困扰或参与变化",
        divergence=True,
        pattern="字面报平安与当前参与变化同时出现，仍需澄清原因",
        score=0.55,
        inference_source="text_structured_observation",
        semantic_hypotheses=[
            SemanticStateSignal(
                id="reduced_life_engagement",
                confidence=0.68,
                evidence_quote=utterance[:120],
                rationale="原话明确描述近期参与或外出意愿下降，原因尚未确认。",
                evidence_turn_index=turn_index,
                source="text_structured_observation",
            ),
            SemanticStateSignal(
                id="concealed_distress",
                confidence=0.42,
                evidence_quote=utterance[:120],
                rationale="报平安与具体变化并存，只保留为待澄清的低置信候选。",
                evidence_turn_index=turn_index,
                source="text_structured_observation",
            ),
        ],
    )
