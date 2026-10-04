"""风险预警接口 —— 非临床筛查与确定性升级的稳定契约。

设计:引擎每轮调用 RiskMonitor.assess()；BaseScene 默认启发式心理风险，
养老场景叠加身体急症信号。产品可替换为经验证的检测器而不改主循环。

分级(tier):
  0 无接管   1 对话内降速与继续观察
  2 紧急真人/专业评估   3 立即急救或危机升级
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from ..engine.context import EngineContext


@dataclass
class RiskSignal:
    tier: int                                  # 0/1/2/3,见模块文档
    kind: str                                  # ideation / depression_baseline_shift / medical / abuse ...
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


class RiskMonitor(Protocol):
    def assess(self, ctx: "EngineContext") -> Optional[RiskSignal]: ...


class NullRiskMonitor:
    """显式关闭风险筛查时使用的零行为实现。"""

    def assess(self, ctx: "EngineContext") -> Optional[RiskSignal]:
        return None
