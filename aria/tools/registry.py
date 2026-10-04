"""通用工具注册表(框架)。

具体工具由各场景通过 Scene.tools() 提供;policy 决定何时调用。
框架本身通用,与场景无关 —— 客服 / 养老 / 教练 用同一套注册与调用。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..interfaces.tools import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Optional[list[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[str]:
        """名字、说明和参数约束 —— 降低模型漏参数与错参数。"""
        descriptions: list[str] = []
        for tool in self._tools.values():
            parameters = getattr(tool, "parameters", {}) or {}
            schema = (
                "；参数:"
                + json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
                if parameters else ""
            )
            descriptions.append(f"{tool.name}:{tool.description}{schema}")
        return descriptions

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, data={"error": f"unknown tool: {name}"})
        return tool.invoke(**kwargs)
