"""Aria 产品 Skill 协议。

Skill 不是一个原子 API，也不是新的 Scene。它把同一养老关系中的用户目标、
对话指导、所需能力、授权要求与可选工具组织在一起；多步流程仍可交给 TaskFlow。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .tools import Tool


@dataclass
class SkillProposal:
    skill_id: str
    intent: str
    goal: str
    guidance: str
    priority: float = 0.5
    consent_required: bool = False
    required_capability: Optional[str] = None
    capability_available: bool = True
    track_task: bool = False
    task_status: str = "detected"
    slots: dict[str, str] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)


@dataclass
class ActiveTask:
    id: str
    skill_id: str
    intent: str
    goal: str
    status: str
    created_turn: int
    updated_turn: int
    slots: dict[str, str] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    required_capability: Optional[str] = None
    capability_available: bool = False
    consent_required: bool = False
    created_at: str = ""
    updated_at: str = ""
    resume_status: Optional[str] = None
    history: list[str] = field(default_factory=list)


@runtime_checkable
class Skill(Protocol):
    id: str
    description: str

    def propose(self, ctx, persona, utterance: str) -> Optional[SkillProposal]: ...

    def continue_task(
        self, ctx, persona, task: ActiveTask, utterance: str
    ) -> Optional[SkillProposal]: ...

    def tools(self) -> list[Tool]: ...
