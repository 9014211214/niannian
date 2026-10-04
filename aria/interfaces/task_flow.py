"""TaskFlow 协议 —— Skill 需要多步骤、等待和恢复时使用。

普通闲聊与原子工具不需要任务流；预约、设备控制、持续提醒等养老 Skill 可以
在同一陪伴关系里挂载 TaskFlow，而不是切换成另一张“客服皮”。

关键:TaskFlow 只读写 EngineContext,**不持有**情绪 / 联盟状态。
那些是引擎的一等公民,与任何任务流解耦。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..engine.context import EngineContext


class TaskFlow(Protocol):
    def advance(self, ctx: "EngineContext") -> None: ...
    def pause(self, ctx: "EngineContext", task_id: str) -> None: ...
    def resume(self, ctx: "EngineContext", task_id: str) -> None: ...
    def cancel(self, ctx: "EngineContext", task_id: str) -> None: ...
