"""感知层:情绪融合 + 口是心非(divergence)。"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.orchestrator import Engine
from aria.engine.perception import PerceptionEngine, fuse
from aria.scenes.registry import get_scene


def test_divergence_suppressed_sadness():
    """声学 sad + 用词 neutral → 口是心非(压抑),primary=sad。"""
    f = fuse({"label": "sad", "score": 0.6}, {"label": "neutral", "score": 0.85})
    assert f.divergence is True
    assert f.primary_label == "sad"
    assert "压抑" in f.divergence_pattern


def test_no_divergence_when_consistent():
    f = fuse({"label": "angry", "score": 0.8}, {"label": "angry", "score": 0.8})
    assert f.divergence is False
    assert f.primary_label == "angry"


def test_reliable_late_distress_trace_supports_divergence():
    """Only a trace with its own reliability may add a divergence candidate."""
    f = fuse(
        {"label": "neutral", "score": 0.7},
        {"label": "neutral", "score": 0.7},
        emotion_trace={"pattern": "late_distress", "trace": ["calm", "calm", "sad"],
                       "confidence": .8, "quality": .9},
    )
    assert f.divergence is True
    assert "末段" in f.divergence_pattern


def test_low_confidence_drops_to_unknown():
    f = fuse({"label": "angry", "score": 0.2}, {"label": "neutral", "score": 0.2})
    assert f.primary_label == "unknown"
    assert f.divergence is False


def test_perception_engine_maps_to_emotion_state():
    emo = PerceptionEngine().perceive(
        "我挺好的", {"label": "sad", "score": 0.6}, {"label": "neutral", "score": 0.85}
    )
    assert emo.label == "sad" and emo.divergence is True and emo.pattern


def test_engine_surfaces_divergence_in_elder_demo():
    """端到端:老陈「我挺好的」(声学 sad/文本 neutral)→ 感知判定口是心非。"""
    engine = Engine(get_scene("companion_elder"))
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    out = engine.process_turn(
        ctx, "我挺好的，你别老问。",
        {"label": "sad", "score": 0.6}, {"label": "neutral", "score": 0.85},
    )
    assert isinstance(out, str) and out          # 回应来自 LLM(mock 时为占位)
    assert ctx.emotion.divergence is True and ctx.emotion.pattern
