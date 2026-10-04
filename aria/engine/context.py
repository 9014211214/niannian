"""EngineContext —— 取代客服时代的 FSMState,成为引擎唯一的状态家。

情绪(快变量)、联盟(关系状态)、个案假设(慢变量)、工作记忆都挂这里,
与任何 FSM / 任务流彻底解耦。任务流(若有)只是读写它的一个插件。

这一步是整个重构的地基:旧项目里情绪寄生在 FSMState 上,导致引擎离不开 FSM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..interfaces.reasoning import ReasoningState, SemanticStateSignal
from ..interfaces.skills import ActiveTask


@dataclass
class Turn:
    user: str
    agent: str


@dataclass
class EmotionState:
    """快变量,每轮重算。

    `divergence` 可能来自声学×文字的观测分歧，也可能只是文本模型提出的低权重
    假设；`inference_source` 明确区分两者。语义状态候选仍要进入证据层校准，
    不能把模型的一次解释直接当作事实。
    """
    label: str = "neutral"
    divergence: bool = False
    pattern: str = ""                                       # 口是心非的描述(给策略/调试)
    score: float = 0.0                                      # 主情绪置信度
    acoustic: dict[str, Any] = field(default_factory=dict)  # 原始留痕
    rupture: bool | None = None                             # 探针顺带判定:对「我」不耐烦?
                                                            # None=本轮无判定(走关键词兜底)
    inference_source: str = ""
    # Optional external classifier output, kept apart from observational evidence.
    external_reading: dict[str, Any] | None = None
    semantic_hypotheses: list[SemanticStateSignal] = field(
        default_factory=list
    )


@dataclass
class AllianceState:
    """关系 / 治疗联盟状态 —— 大师持续盯的东西。裂了优先修复,而非继续推内容。"""
    trust: float = 0.5
    ruptured: bool = False
    rupture_events: int = 0                                 # 本会话破裂次数(结束写回账本)


@dataclass
class Formulation:
    """演进中的个案假设(慢变量,带置信)。不是静态档案 —— 它会被证据更新。"""
    hypotheses: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EngineContext:
    user_id: str
    scene_id: str
    session_id: str = field(default_factory=lambda: uuid4().hex)
    turns: list[Turn] = field(default_factory=list)
    emotion: EmotionState = field(default_factory=EmotionState)
    alliance: AllianceState = field(default_factory=AllianceState)
    formulation: Formulation = field(default_factory=Formulation)
    reasoning: ReasoningState = field(default_factory=ReasoningState)
    active_tasks: list[ActiveTask] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    # Sticky provenance: a context used for synthetic observations must never
    # be checkpointed or finalized by a normal persistent engine.
    simulation_mode: bool = False
    # Python-only capability for persistent fictional experiments; never Web data.
    external_emotion_experiment: Any = None

    def add_turn(self, user: str, agent: str) -> None:
        self.turns.append(Turn(user=user, agent=agent))
