"""Tool 协议 —— 引擎可调用的「行动」。

Tool 只表示真实的原子副作用；Skill 负责把用户目标、授权和这些动作组织起来。
工具框架通用,具体工具由各场景注册。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, str]

    def invoke(self, **kwargs: Any) -> ToolResult: ...
