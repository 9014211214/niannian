"""Skill 注册、同轮提议与轻量任务状态。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import re
from typing import Iterable

from ..interfaces.skills import ActiveTask, Skill, SkillProposal
from ..interfaces.tools import Tool

_TERMINAL = {"completed", "failed", "cancelled"}
_CANCEL = ("取消", "算了", "先不用", "不用提醒", "别提醒")
_PAUSE = ("暂停", "先放一放", "等会再说")
_RESUME = ("继续", "接着", "恢复")
_TOPIC_CONTROL = ("不聊", "别聊", "不说这个", "换个话题")
_CONTROL_NEGATIONS = ("不", "别", "不要", "不能", "不想", "没有", "没", "未")
_CONTROL_NON_CURRENT = (
    "昨天", "以前", "上次", "之前", "刚才", "如果", "假如", "要是",
    "他说", "她说", "医生说", "护理员说", "新闻说", "电视里",
)
_NEGATIVE_FORM_CANCELS = {"先不用", "不用提醒", "别提醒"}


def _has_current_control(
    utterance: str,
    markers: tuple[str, ...],
) -> bool:
    """任务控制只接受本人当下指令，避免否定、旧事或转述改变任务。"""
    clauses = [
        item.strip()
        for item in re.split(r"[，,。！？?；;\n]+", utterance)
        if item.strip()
    ]
    decision: bool | None = None
    for clause in clauses:
        occurrences: list[tuple[int, str]] = []
        for marker in markers:
            start = 0
            while (at := clause.find(marker, start)) >= 0:
                occurrences.append((at, marker))
                start = at + len(marker)
        for at, marker in sorted(occurrences):
            prefix = clause[:at]
            if any(item in prefix for item in _CONTROL_NON_CURRENT):
                continue
            negated = (
                marker not in _NEGATIVE_FORM_CANCELS
                and any(
                    item in prefix[-7:] for item in _CONTROL_NEGATIONS
                )
            )
            # 同句有纠正时以后项为准：“取消……不，别取消”不得执行前项。
            decision = not negated
    return decision is True


class SkillRegistry:
    def __init__(
        self,
        skills: Iterable[Skill] = (),
        *,
        capabilities: Iterable[str] = (),
    ):
        self._skills = {skill.id: skill for skill in skills}
        self._capabilities = set(capabilities)

    def tools(self) -> list[Tool]:
        result: list[Tool] = []
        for skill in self._skills.values():
            provider = getattr(skill, "tools", None)
            if callable(provider):
                result.extend(provider())
        return result

    def has_capability(self, capability: str | None) -> bool:
        return bool(capability and capability in self._capabilities)

    def restored_status(
        self,
        status: str,
        *,
        required_capability: str | None,
        missing_slots: list[str],
    ) -> str:
        """部署能力变化后重算未完成任务状态，不沿用过期的可执行声明。"""
        if status in _TERMINAL or status == "paused":
            return status
        if missing_slots:
            return "needs_details"
        if required_capability:
            return (
                "needs_confirmation"
                if self.has_capability(required_capability)
                else "needs_capability"
            )
        return status

    def evaluate(self, ctx, persona, utterance: str) -> list[SkillProposal]:
        controlled = self._apply_control(ctx, utterance)
        if controlled is not None:
            ctx.working_memory["skill_proposals"] = [controlled]
            return [controlled]

        proposals: list[SkillProposal] = []
        continued_skill_ids: set[str] = set()
        for task in reversed(ctx.active_tasks):
            if task.status in _TERMINAL or task.status == "paused":
                continue
            skill = self._skills.get(task.skill_id)
            continuation = getattr(skill, "continue_task", None)
            if not callable(continuation):
                continue
            proposal = continuation(ctx, persona, task, utterance)
            if proposal is not None:
                proposals.append(self._resolve_capability(proposal))
                continued_skill_ids.add(skill.id)

        for skill in self._skills.values():
            if skill.id in continued_skill_ids:
                continue
            proposal = skill.propose(ctx, persona, utterance)
            if proposal is not None:
                proposals.append(self._resolve_capability(proposal))
        proposals.sort(key=lambda item: item.priority, reverse=True)
        ctx.working_memory["skill_proposals"] = proposals[:3]
        for proposal in proposals:
            if proposal.track_task:
                self._upsert_task(ctx, proposal)
        return proposals

    def describe(self) -> list[dict]:
        return [
            {"id": skill.id, "description": skill.description}
            for skill in self._skills.values()
        ]

    @staticmethod
    def serialise(proposal: SkillProposal) -> dict:
        return asdict(proposal)

    def _resolve_capability(
        self, proposal: SkillProposal
    ) -> SkillProposal:
        required = proposal.required_capability
        if not required:
            proposal.capability_available = True
            return proposal
        proposal.capability_available = required in self._capabilities
        if (
            proposal.capability_available
            and proposal.task_status == "needs_capability"
        ):
            proposal.task_status = (
                "needs_confirmation"
                if proposal.consent_required else "ready"
            )
        elif (
            not proposal.capability_available
            and proposal.task_status
            not in {"needs_details", "unavailable"}
        ):
            proposal.task_status = "needs_capability"
        return proposal

    @staticmethod
    def _upsert_task(ctx, proposal: SkillProposal) -> None:
        task_id = f"{proposal.skill_id}:{proposal.intent}"
        current = next(
            (
                item for item in reversed(ctx.active_tasks)
                if item.id == task_id and item.status not in _TERMINAL
            ),
            None,
        )
        turn = len(ctx.turns)
        now = datetime.now().isoformat(timespec="seconds")
        if current is None:
            ctx.active_tasks.append(ActiveTask(
                id=task_id,
                skill_id=proposal.skill_id,
                intent=proposal.intent,
                goal=proposal.goal,
                status=proposal.task_status,
                created_turn=turn,
                updated_turn=turn,
                slots=dict(proposal.slots),
                missing_slots=list(proposal.missing_slots),
                required_capability=proposal.required_capability,
                capability_available=proposal.capability_available,
                consent_required=proposal.consent_required,
                created_at=now,
                updated_at=now,
                history=[f"{now}:{proposal.task_status}"],
            ))
        else:
            current.status = proposal.task_status
            current.updated_turn = turn
            current.slots.update(proposal.slots)
            current.missing_slots = list(proposal.missing_slots)
            current.required_capability = proposal.required_capability
            current.capability_available = proposal.capability_available
            current.consent_required = proposal.consent_required
            current.updated_at = now
            current.history.append(f"{now}:{proposal.task_status}")
            current.history = current.history[-20:]
        ctx.active_tasks = ctx.active_tasks[-20:]

    @staticmethod
    def _mentions_task(current: ActiveTask, utterance: str) -> bool:
        intent = current.intent.lower()
        goal = current.goal
        terms: set[str] = set()
        if "reminder" in intent or "提醒" in goal:
            terms.update(("提醒", "闹钟", "到点叫"))
        if "hydration" in intent or "喝水" in goal or "饮水" in goal:
            terms.update(("喝水", "饮水", "水提醒"))
        if "water_delivery" in intent or "倒水" in goal or "送水" in goal:
            terms.update(("倒水", "送水", "送过来"))
        return any(term in utterance for term in terms)

    @classmethod
    def _control_is_scoped(cls, ctx, current: ActiveTask, utterance: str) -> bool:
        if cls._mentions_task(current, utterance):
            return True
        # “不聊电视了”控制的是谈话主题；即使上一轮刚好创建了提醒，
        # 也不能把一个泛化的“算了”解释成取消外部任务。
        if any(marker in utterance for marker in _TOPIC_CONTROL):
            return False
        adjacent_focus = current.updated_turn >= max(0, len(ctx.turns) - 1)
        if adjacent_focus:
            return True
        # 暂停状态本身提供了窄焦点，但只接受简短的恢复指令，避免
        # “继续聊电视”意外恢复后台任务。
        compact = utterance.strip(" ，,。！？!?吧呢啊")
        return (
            current.status == "paused"
            and len(compact) <= 4
            and any(compact.startswith(marker) for marker in _RESUME)
        )

    @staticmethod
    def _apply_control(
        ctx, utterance: str
    ) -> SkillProposal | None:
        current = next(
            (
                item for item in reversed(ctx.active_tasks)
                if item.status not in _TERMINAL
            ),
            None,
        )
        if current is None:
            return None
        now = datetime.now().isoformat(timespec="seconds")
        if _has_current_control(utterance, _CANCEL):
            if not SkillRegistry._control_is_scoped(ctx, current, utterance):
                return None
            current.status = "cancelled"
            guidance = "确认已经取消当前任务，不执行任何工具；不要继续追问。"
        elif _has_current_control(utterance, _PAUSE):
            if not SkillRegistry._control_is_scoped(ctx, current, utterance):
                return None
            current.resume_status = current.status
            current.status = "paused"
            guidance = "确认任务已暂停；保留上下文，但不执行任何动作。"
        elif current.status == "paused" and _has_current_control(
            utterance, _RESUME
        ):
            if not SkillRegistry._control_is_scoped(ctx, current, utterance):
                return None
            current.status = current.resume_status or "needs_details"
            current.resume_status = None
            guidance = "确认继续之前的任务，从尚未完成的步骤接着处理。"
        else:
            return None
        current.updated_at = now
        current.updated_turn = len(ctx.turns)
        current.history.append(f"{now}:{current.status}")
        current.history = current.history[-20:]
        return SkillProposal(
            skill_id=current.skill_id,
            intent=f"task_{current.status}",
            goal=current.goal,
            guidance=guidance,
            priority=0.95,
            task_status=current.status,
            slots=dict(current.slots),
            missing_slots=list(current.missing_slots),
            required_capability=current.required_capability,
            capability_available=current.capability_available,
            consent_required=current.consent_required,
        )
