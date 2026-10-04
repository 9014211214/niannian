"""BaseScene —— 给「薄皮肤」的默认实现。

场景只覆盖它关心的口子,其余走默认(无工具、无任务流)。
这样新增一张皮的成本极低 —— 这正是「可插拔」要的效果。

安全底线例外:风险筛查默认是**开启的**(启发式、非临床)。
一张皮可以没有工具没有知识库,但不该没有安全网;确实要关
(如离线评测)显式覆盖 risk_monitor() 返回 NullRiskMonitor。
"""
from __future__ import annotations

from typing import Optional

from ..engine.risk.heuristic import HeuristicRiskMonitor
from ..interfaces.channels import Channel
from ..interfaces.knowledge import KnowledgeSource
from ..interfaces.risk import RiskMonitor
from ..interfaces.reasoning import StateReasoner
from ..interfaces.skills import Skill
from ..interfaces.scene import AgentPreset, Redline
from ..interfaces.task_flow import TaskFlow
from ..interfaces.tools import Tool


class BaseScene:
    id: str = "base"

    def agent_persona(self) -> AgentPreset:
        return AgentPreset(name="Aria", role="对话伙伴")

    def safety_rules(self) -> list[Redline]:
        return []

    def tools(self) -> list[Tool]:
        return []

    def skills(self) -> list[Skill]:
        return []

    def task_flow(self) -> Optional[TaskFlow]:
        return None

    def channels(self) -> list[Channel]:
        return []

    def knowledge(self) -> Optional[KnowledgeSource]:
        return None

    def risk_monitor(self) -> RiskMonitor:
        return HeuristicRiskMonitor()

    def state_reasoner(self) -> Optional[StateReasoner]:
        return None

    def observe_care(self, ctx, persona, utterance: str, risk_signal=None) -> bool:
        """可选的场景照护观测钩子；默认没有任何副作用。"""
        return False

    def execute_care_skills(self, ctx, persona) -> bool:
        """可选的确定性 Skill 执行钩子；默认不执行外部动作。"""
        return False

    def guard_delivery_response(
        self, ctx, response: str, utterance: str, persona=None
    ) -> str:
        """动作执行后的最终交付守卫；默认不改写。"""
        return response
