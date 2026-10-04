"""状态推理契约 —— 把「看到了什么」与「这一拍做什么」分开。

设计目标:
- 多模态是可选证据源；缺一种模态时正常退化，绝不补造观测。
- 所有观测都带来源、质量与置信度，视觉/声学只作为证据，不冒充事实。
- 假设保持竞争关系，并用澄清问题继续求证。
- 干预产生可观察结果，供会后反思与长期个性化使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class ObservationEvidence:
    """一条可追溯的观测，不等同于情绪或诊断结论。"""

    id: str
    modality: str
    signal: str
    value: Any
    source: str = ""
    fallback: bool = False
    confidence: float = 0.5
    quality: float = 1.0
    baseline_delta: Optional[float] = None
    detail: str = ""
    turn_index: int = 0

    @property
    def weight(self) -> float:
        """实际进入推理的证据权重。低画质/低置信会自然降权。"""
        return _clamp01(self.confidence) * _clamp01(self.quality)

    def summary(self) -> str:
        delta = ""
        if self.baseline_delta is not None:
            delta = f",相对个人基线 {self.baseline_delta:+.2f}"
        detail = f"({self.detail})" if self.detail else ""
        provenance = f",来源 {self.source}" if self.source else ""
        if self.fallback:
            provenance += ",回退信号"
        return (
            f"{self.modality}/{self.signal}={self.value}{delta}"
            f",有效置信 {self.weight:.2f}{provenance}{detail}"
        )


@dataclass
class SemanticStateSignal:
    """文本模型提出的当前状态候选。

    它不是观测到的情绪，也不是诊断。`evidence_quote` 必须能在当前话语或近期
    用户原话中找到；否则感知层会丢弃这条候选，避免模型用解释冒充证据。
    """

    id: str
    confidence: float
    evidence_quote: str
    rationale: str = ""
    counterevidence_quote: str = ""
    counter_only: bool = False
    # 引用究竟来自当前轮还是近期历史。没有这两个字段时，模型每轮重复引用
    # 同一句旧话会把旧证据伪装成“刚刚又发生了一次”。
    evidence_turn_index: Optional[int] = None
    counterevidence_turn_index: Optional[int] = None
    source: str = "text_llm_hypothesis"


@dataclass
class WorkingHypothesis:
    """关于当前需要/状态的可证伪工作假设，不是临床诊断。"""

    id: str
    label: str
    category: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    counterevidence_ids: list[str] = field(default_factory=list)
    clarification_question: str = ""


@dataclass
class InterventionCandidate:
    """一个可以比较的回应/行动方向。"""

    id: str
    label: str
    goal: str
    guidance: str
    avoid: str = ""
    score: float = 0.0
    consent_required: bool = False
    hypothesis_ids: list[str] = field(default_factory=list)
    selected_turn: int = 0


@dataclass
class InterventionOutcome:
    """用户对一次干预的后续反应。"""

    intervention_id: str
    intervention_label: str
    goal: str
    value: str                              # positive / neutral / negative
    signal: str
    note: str = ""
    turn_index: int = 0
    hypothesis_ids: list[str] = field(default_factory=list)


@dataclass
class ActiveObservationPlan:
    """为了减少不确定性而选择的低打扰观察动作；问问题只是其中一种。"""

    mode: str = "observe"                  # ask / observe / wait
    reason: str = ""
    question: str = ""
    hypothesis_ids: list[str] = field(default_factory=list)
    information_gain: float = 0.0
    intrusiveness: str = "low"


@dataclass
class ObservationExecution:
    """本轮真正交付的观察动作；与生成前后的计划明确分开。"""

    turn_index: int
    planned_mode: str = "none"
    executed_mode: str = "observe"       # ask / observe / wait
    planned_question: str = ""
    executed_question: str = ""
    # observe 下出现一个日常邀请会记录为 ask + matched_plan=False：它不是
    # 预先计划的诊断式 probe，但仍计入本轮提问成本，避免下一轮连续追问。
    matched_plan: bool = False
    hypothesis_ids: list[str] = field(default_factory=list)


@dataclass
class DeliveredIntervention:
    """真正随最终回复交付的干预；候选被选中并不自动等于已交付。"""

    id: str
    label: str
    goal: str
    turn_index: int
    hypothesis_ids: list[str] = field(default_factory=list)
    source: str = "final_response"


@dataclass
class ReasoningState:
    """会话内的证据—假设—决策—结果状态。"""

    evidence: list[ObservationEvidence] = field(default_factory=list)
    hypotheses: list[WorkingHypothesis] = field(default_factory=list)
    candidates: list[InterventionCandidate] = field(default_factory=list)
    selected: Optional[InterventionCandidate] = None
    outcomes: list[InterventionOutcome] = field(default_factory=list)
    observation_plan: Optional[ActiveObservationPlan] = None
    observation_history: list[ObservationExecution] = field(
        default_factory=list
    )
    delivered_interventions: list[DeliveredIntervention] = field(
        default_factory=list
    )


@runtime_checkable
class StateReasoner(Protocol):
    """场景可选的状态推理插件。"""

    def observe_response(self, ctx, utterance: str) -> None: ...

    def update(self, ctx, persona, utterance: str,
               evidence: list[ObservationEvidence]) -> None: ...

    def record_observation_execution(self, ctx, response: str): ...

    def record_intervention_delivery(
        self, ctx, response: str, *, response_overridden: bool = False
    ): ...

    def record_outcome(self, ctx, value: str, *, signal: str,
                       note: str = "") -> None: ...

    def finalize(self, ctx, persona) -> None: ...
