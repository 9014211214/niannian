"""情绪融合 —— 声学与文本观测 → 有可靠性边界的情绪候选与分歧。

(从旧 repo 移植,逻辑保持一致。纯规则,不调 LLM,零延迟。)

divergence 仅指观测不一致，不证明隐藏心理、欺瞒或疾病。
身体状况、环境与识别错误也可能解释分歧，不能用声学覆盖本人明确意愿。

声学标签集(emotion2vec_plus_large 9 类):
  angry / happy / sad / fearful / disgusted / surprised / neutral / other / unknown
文本标签集(实测):happy / sad / angry / neutral
两者由 _normalize_label 统一映射。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from .signals import unit_interval

# ── 标签归一 ────────────────────────────────────────────────
_NORMALIZE_MAP: Dict[str, str] = {
    "angry": "angry", "anger": "angry", "怒": "angry", "愤怒": "angry", "生气": "angry",
    "happy": "happy", "happiness": "happy", "joy": "happy", "喜": "happy", "开心": "happy", "高兴": "happy",
    "sad": "sad", "sadness": "sad", "悲": "sad", "悲伤": "sad", "难过": "sad",
    "fearful": "fearful", "fear": "fearful", "anxious": "fearful", "惧": "fearful", "害怕": "fearful", "恐惧": "fearful",
    "disgusted": "disgusted", "disgust": "disgusted", "hate": "disgusted", "恶": "disgusted", "厌恶": "disgusted",
    "surprised": "surprised", "surprise": "surprised", "惊讶": "surprised", "吃惊": "surprised",
    "neutral": "neutral", "calm": "neutral", "中性": "neutral", "中立": "neutral", "平静": "neutral",
    "other": "unknown", "unknown": "unknown", "": "unknown",
}


def _normalize_label(raw: Optional[str]) -> str:
    if not isinstance(raw, str) or not raw:
        return "unknown"
    key = raw.strip().lower()
    if key in _NORMALIZE_MAP:
        return _NORMALIZE_MAP[key]
    if "/" in key:
        parts = [p.strip() for p in key.split("/") if p.strip()]
        for p in reversed(parts):  # 末段(英文)优先
            if p in _NORMALIZE_MAP:
                return _NORMALIZE_MAP[p]
    return "unknown"


# ── 分歧规则表 ──────────────────────────────────────────────
# (acoustic, text) -> (primary, divergence_pattern)
_DIVERGENCE_RULES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("happy", "sad"): ("sad", "语调轻快但用词悲伤(可能强忍/苦笑)"),
    ("happy", "angry"): ("angry", "语调轻快但用词强硬(可能反讽/嘲讽)"),
    ("sad", "happy"): ("sad", "语调低沉但用词积极(可能客套/勉强)"),
    ("sad", "neutral"): ("sad", "语调低沉但用词中性(可能压抑情绪)"),
    ("sad", "angry"): ("sad", "语调低沉但用词强硬(可能委屈/无力反抗)"),
    ("angry", "happy"): ("angry", "语调愤怒但用词积极(可能反讽)"),
    ("angry", "neutral"): ("angry", "语调愤怒但用词中性(可能压抑/强忍)"),
    ("angry", "sad"): ("angry", "语调愤怒但用词悲伤(委屈+愤怒并行)"),
    ("fearful", "happy"): ("fearful", "语调紧张但用词积极(可能强装镇定)"),
    ("fearful", "neutral"): ("fearful", "语调紧张但用词中性(可能压抑焦虑)"),
    ("fearful", "angry"): ("fearful", "语调紧张但用词强硬(可能虚张声势)"),
    ("disgusted", "happy"): ("disgusted", "语调厌恶但用词积极(可能反讽/讽刺)"),
    ("disgusted", "neutral"): ("disgusted", "语调厌恶但用词中性(可能克制不满)"),
}


class FusedEmotion(BaseModel):
    """实时融合输出。"""
    primary_label: str = "unknown"
    primary_score: float = 0.0

    divergence: bool = False
    divergence_pattern: str = ""

    acoustic_label: str = "unknown"
    acoustic_score: float = 0.0
    acoustic_quality: float = 1.0
    acoustic_effective_score: float = 0.0
    text_label: str = "unknown"
    text_score: float = 0.0

    fusion_strategy: str = "quality_aware_v2"

    # 声学证据 + 子句级情绪轨迹(给下游当物理证据,不只复读标签)。
    prosodic: Dict[str, Any] = Field(default_factory=dict)
    emotion_trace: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        return self.primary_score >= 0.5 and self.primary_label not in ("neutral", "unknown")


def fuse(
    acoustic: Optional[dict],
    text: Optional[dict],
    *,
    acoustic_threshold: float = 0.4,
    text_threshold: float = 0.4,
    prosodic: Optional[dict] = None,
    emotion_trace: Optional[dict] = None,
) -> FusedEmotion:
    """声学 + 文本 → FusedEmotion(永不为 None)。

    acoustic / text: {"label","score",...} | None;置信度低于阈值视为 unknown。
    prosodic / emotion_trace: 可选,用于 divergence 增强(末段崩塌 / 物理证据)。
    """
    a_label = _normalize_label((acoustic or {}).get("label"))
    a_raw_score = unit_interval((acoustic or {}).get("score"))
    a_quality = unit_interval((acoustic or {}).get("quality"), 1.0)
    a_score = a_raw_score * a_quality
    if a_score < acoustic_threshold:
        a_label = "unknown"

    t_label = _normalize_label((text or {}).get("label"))
    t_score = (
        unit_interval((text or {}).get("score"))
        * unit_interval((text or {}).get("quality"), 1.0)
    )
    if t_score < text_threshold:
        t_label = "unknown"

    fused = FusedEmotion(
        acoustic_label=a_label, acoustic_score=a_raw_score,
        acoustic_quality=a_quality, acoustic_effective_score=a_score,
        text_label=t_label, text_score=t_score,
        prosodic=prosodic if isinstance(prosodic, dict) else {},
        emotion_trace=emotion_trace if isinstance(emotion_trace, dict) else {},
    )

    if a_label == "unknown" and t_label == "unknown":
        pass
    elif a_label == "unknown":
        fused.primary_label, fused.primary_score = t_label, t_score
    elif t_label == "unknown":
        fused.primary_label, fused.primary_score = a_label, a_score
    elif a_label == t_label:
        fused.primary_label = a_label
        fused.primary_score = max(a_score, t_score)
    elif a_label == "neutral" and t_label != "neutral":
        # 声学中性但文本有明确情绪 → 采信文本(弱信号,非强分歧)。
        fused.primary_label, fused.primary_score = t_label, t_score
    else:
        rule = _DIVERGENCE_RULES.get((a_label, t_label))
        if rule:
            primary, pattern = rule
            fused.primary_label = primary
            fused.primary_score = a_score if primary == a_label else t_score
            fused.divergence = True
            fused.divergence_pattern = pattern
        else:
            fused.primary_label, fused.primary_score = a_label, a_score
            fused.divergence = True
            fused.divergence_pattern = f"声学={a_label} 文本={t_label} 不一致(未列入规则表)"

    _apply_acoustic_evidence(fused)
    if fused.divergence:
        fused.divergence_pattern += "；仅为观测分歧，亦可能受身体/环境或识别误差影响，需本人确认"
    return fused


def _apply_acoustic_evidence(fused: FusedEmotion) -> None:
    """用 prosodic 五维 + emotion_trace 三段再做一次 divergence 增强(只追加,不覆盖)。"""
    pros: dict = fused.prosodic or {}
    trace: dict = fused.emotion_trace or {}

    # Segment labels require their own reliability; an unscored pattern string
    # cannot force distress even when the whole-utterance classifier is strong.
    trace_score = (
        unit_interval(trace.get("confidence"))
        * min(fused.acoustic_quality, unit_interval(trace.get("quality")))
    )
    segments = trace.get("trace")
    late_label = (
        _normalize_label(segments[-1])
        if isinstance(segments, list) and len(segments) >= 2 else "unknown"
    )
    if (
        trace.get("pattern") == "late_distress"
        and trace_score >= .4
        and late_label in {"sad", "angry", "fearful", "disgusted"}
        and not fused.divergence
    ):
        fused.divergence = True
        fused.divergence_pattern = f"句内轨迹末段出现 {late_label} 线索，原因待确认"
        if fused.primary_label in ("unknown", "neutral"):
            fused.primary_label = late_label
            fused.primary_score = trace_score

    # 2) 物理证据追加(只在已有 divergence 时附加,避免误判)。
    evidence_bits: List[str] = []
    pros_quality = min(fused.acoustic_quality, unit_interval(pros.get("quality"), 1.0))
    if pros_quality * unit_interval(pros.get("confidence"), 1.0) < .4:
        return
    if unit_interval(pros.get("voice_break_ratio")) > 0.2:
        evidence_bits.append(f"声音破裂率 {float(pros['voice_break_ratio']):.2f}")
    if pros.get("volume_envelope") == "decaying":
        evidence_bits.append("尾音衰减")
    if unit_interval(pros.get("pitch_jitter")) > 0.25:
        evidence_bits.append(f"基频抖动 {float(pros['pitch_jitter']):.2f}")
    if unit_interval(pros.get("pause_ratio")) > 0.35:
        evidence_bits.append(f"长停顿占比 {float(pros['pause_ratio']):.2f}")

    if evidence_bits and fused.divergence:
        fused.divergence_pattern = (
            (fused.divergence_pattern + " · " if fused.divergence_pattern else "")
            + "声学证据:" + "、".join(evidence_bits)
        )
