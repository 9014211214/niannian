"""心理策略矩阵 —— 规则检索「这一拍怎么接」(BACKLOG P1-#6)。

策略(情绪 × 交付模式 × 口是心非 → 动作)是结构化资产
`assets/strategy_matrix.yaml`,不是 prompt 散文:可单测、可累积、可分层开源。
policy 在这里规则命中条目,prompt 只负责渲染选中的 1-2 条。

情绪归一化:语音 fusion 给规范英文标签(sad/angry/...),文字探针给自由
中文词(「失落」「烦」),先归到统一类别再匹配。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from ..context import EmotionState

_MATRIX_PATH = Path(__file__).resolve().parents[2] / "assets" / "strategy_matrix.yaml"

# 归一化类别 → 触发词(中文自由词 + 英文规范标签都收)。
# 顺序即扫描顺序;命中多个类别 → mixed(并存情绪)。
_EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    "tired": ("累", "疲惫", "疲倦", "耗竭", "倦", "tired", "exhausted", "burnout", "burned_out"),
    "anxious": ("焦虑", "担心", "紧张", "慌", "害怕", "恐惧", "不安", "心虚",
                "anxious", "fearful", "fear", "worried", "nervous"),
    # 注意:单字触发词误伤面大(「火」中「火车」、「气」中「天气」),只收双字以上,
    # 例外是口语高频且歧义低的「烦」「恼」「怨」。
    "angry": ("生气", "愤怒", "烦", "恼火", "上火", "窝火", "不满", "恼", "怨",
              "不耐烦", "气愤", "气恼", "来气", "angry", "anger", "frustrated",
              "annoyed", "sarcas"),
    "sad": ("难过", "难受", "失落", "伤心", "委屈", "孤独", "低落", "沮丧", "失望",
            "想念", "思念", "寂寞", "悲", "不甘", "sad", "lonely", "down", "grief",
            "disappointed", "discouraged", "resigned", "hurt"),
    "happy": ("开心", "高兴", "自豪", "骄傲", "欣慰", "喜悦", "兴奋", "满足", "得意",
              "happy", "proud", "joy", "excited", "pleased", "relief", "relieved"),
}


def normalize_emotion(label: str) -> str:
    """自由情绪词 → 规范类别(sad/angry/anxious/happy/tired/mixed/neutral)。"""
    s = (label or "").strip().lower()
    if not s or s in ("neutral", "unknown"):
        return "neutral"
    hits = [cat for cat, words in _EMOTION_LEXICON.items() if any(w in s for w in words)]
    if len(hits) > 1:
        return "mixed"
    return hits[0] if hits else "neutral"


@dataclass
class Strategy:
    id: str
    label: str
    mechanism: str
    do: str
    avoid: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)      # 必填:适用情绪类别
    divergence: Optional[bool] = None                       # None = 不限
    modes: Optional[list[str]] = None                       # None = 不限


@lru_cache(maxsize=1)
def load_strategy_matrix(path: str | None = None) -> tuple[str, tuple[Strategy, ...]]:
    p = Path(path) if path else _MATRIX_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    entries = []
    for e in data["entries"]:
        when = e.get("when") or {}
        entries.append(Strategy(
            id=e["id"], label=e["label"], mechanism=e.get("mechanism", ""),
            do=str(e["do"]).strip(), avoid=list(e.get("avoid") or []),
            emotions=list(when.get("emotions") or []),
            divergence=when.get("divergence"),
            modes=list(when["modes"]) if when.get("modes") else None,
        ))
    return str(data["version"]), tuple(entries)


def select_strategies(emotion: EmotionState, mode: str, *, top_n: int = 2) -> list[Strategy]:
    """规则检索:显式条件全匹配才候选,越具体(显式键越多)越优先。

    neutral 情绪不命中任何策略 —— 平常对话不需要「策略」,别过度干预。
    """
    category = normalize_emotion(emotion.label)
    if category == "neutral":
        return []
    _, entries = load_strategy_matrix()
    scored: list[tuple[int, int, Strategy]] = []
    for i, s in enumerate(entries):
        if category not in s.emotions:
            continue
        score = 1                                           # 情绪命中(必要条件)
        if s.divergence is not None:
            if s.divergence != emotion.divergence:
                continue
            score += 2                                      # 显式匹配口是心非更具体
        if s.modes is not None:
            if mode not in s.modes:
                continue
            score += 1
        scored.append((-score, i, s))                       # 同分按文件顺序稳定排序
    scored.sort()
    return [s for _, _, s in scored[:top_n]]
