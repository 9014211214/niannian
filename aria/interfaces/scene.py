"""Scene 协议 —— 可插拔皮肤的统一契约。

引擎只依赖本协议；当前产品场景是养老，其他实现仅用于验证可插拔边界。
关系身份与安全边界由 Scene 提供；场景内新增任务优先使用 Skill，而不是新建皮肤。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .channels import Channel
from .knowledge import KnowledgeSource
from .risk import RiskMonitor
from .reasoning import StateReasoner
from .skills import Skill
from .task_flow import TaskFlow
from .tools import Tool


@dataclass
class AgentPreset:
    """Agent 在该场景下的身份 / 口吻。"""
    name: str
    role: str
    description: str = ""


@dataclass
class Redline:
    """一条场景级安全红线。"""
    rule: str


@runtime_checkable
class Scene(Protocol):
    """一张「皮肤」。引擎拿到一个 Scene 实例即可运行,不关心具体是哪个场景。"""

    id: str

    def agent_persona(self) -> AgentPreset: ...
    def safety_rules(self) -> list[Redline]: ...
    def tools(self) -> list[Tool]: ...
    def skills(self) -> list[Skill]: ...
    def task_flow(self) -> Optional[TaskFlow]: ...      # 客服=FSM插件;养老/陪伴=None
    def channels(self) -> list[Channel]: ...
    def knowledge(self) -> Optional[KnowledgeSource]: ...
    def risk_monitor(self) -> RiskMonitor: ...          # 默认 NullRiskMonitor
    def state_reasoner(self) -> Optional[StateReasoner]: ...  # 可选:证据→假设→干预→结果
    def observe_care(self, ctx, persona, utterance: str, risk_signal=None) -> bool: ...
    def execute_care_skills(self, ctx, persona) -> bool: ...
    def guard_delivery_response(
        self, ctx, response: str, utterance: str, persona=None
    ) -> str: ...
