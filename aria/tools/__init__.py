"""tools —— 通用工具框架。

`ToolRegistry` 是框架(通用);具体工具由各场景通过 `Scene.tools()` 注册。
客服专用工具(地址 / 高德 / 工单)不在此处 —— 那些属于将来的 service 皮。
"""
from __future__ import annotations

from .registry import ToolRegistry

__all__ = ["ToolRegistry"]
