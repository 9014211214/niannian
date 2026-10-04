"""P1:心理策略矩阵 —— 资产加载 / 情绪归一化 / 规则检索 / prompt 渲染。"""
from __future__ import annotations

from aria.engine.context import EmotionState, EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.strategy import (
    load_strategy_matrix,
    normalize_emotion,
    select_strategies,
)
from aria.scenes.registry import get_scene


# ── 资产加载 ────────────────────────────────────────────────
def test_load_matrix():
    version, entries = load_strategy_matrix()
    assert version.startswith("strategy-matrix-v1")
    assert len(entries) >= 12
    ids = [s.id for s in entries]
    assert len(ids) == len(set(ids))                        # id 唯一
    for s in entries:
        assert s.emotions and s.do and s.mechanism          # 机制(文献依据)是 IP,必填


# ── 情绪归一化(中文自由词 + 英文规范标签)───────────────────
def test_normalize_emotion():
    assert normalize_emotion("失落") == "sad"
    assert normalize_emotion("嘴上说没事其实很委屈") == "sad"
    assert normalize_emotion("烦") == "angry"
    assert normalize_emotion("fearful") == "anxious"
    assert normalize_emotion("欣慰但疲惫") == "mixed"       # 多类别命中 → 并存情绪
    assert normalize_emotion("neutral") == "neutral"
    assert normalize_emotion("平平无奇的火星词") == "neutral"


# ── 规则检索:越具体越优先,显式不匹配直接排除 ──────────────
def test_select_divergent_sad_warm():
    got = select_strategies(EmotionState(label="难受", divergence=True), "warm_take_over")
    assert got and got[0].id == "gently_name_hidden_feeling"


def test_select_divergent_angry_firm_prefers_shortcut():
    got = select_strategies(EmotionState(label="angry", divergence=True), "firm_with_options")
    ids = [s.id for s in got]
    assert ids[0] == "shortcut_for_suppressed_anger"        # 情绪+分歧+模式三键全中
    assert "detect_sarcasm" in ids                          # 两键命中,次优


def test_select_low_pressure_sad_not_divergent():
    got = select_strategies(EmotionState(label="sad", divergence=False), "low_pressure")
    assert [s.id for s in got][0] == "low_pressure_checkin"
    # 显式 divergence=true 的条目(温和点破)绝不能命中表里一致的情况
    assert "gently_name_hidden_feeling" not in [s.id for s in got]


def test_select_neutral_hits_nothing():
    assert select_strategies(EmotionState(label="neutral"), "neutral") == []


# ── prompt 渲染:只渲染命中的条目 ───────────────────────────
def test_prompt_renders_selected_strategy():
    scene = get_scene("companion_elder")
    persona = load_demo_persona("laochen_demo")             # 回避型 → low_pressure
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    ctx.emotion = EmotionState(label="失落", divergence=False)
    system = build_messages(scene, persona, ctx, "相亲没聊下去。也正常。")[0]["content"]
    assert "这一拍怎么接" in system
    assert "低压陪伴" in system                              # low_pressure × sad 命中
    assert "温和点破" not in system                          # divergence=true 条目未命中
