"""结构化动作解析(P3-#13)—— 让「懂」接到「做」。

LLM 需要替用户做事时,在回应正文后另起一行输出:
    <action>{"tool":"medication_reminder","args":{"medication":"降压药"}}</action>
引擎解析后剥离标签(用户绝不会看到)、调用 ToolRegistry、把结果写回
working_memory。只有 Tool 返回成功回执才能称为执行；默认能力占位返回失败时，
当轮就纠正模型可能说出的完成承诺。

解析必须绝对稳健:JSON 坏 / 字段缺 → 当普通文本处理(标签照样剥掉),
绝不让格式问题污染对用户的回应。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional
from uuid import uuid4

from ..context import EngineContext
from ..model.persona import (
    AD_HOC_NOTIFICATION_CONSENT_SCOPES,
    PersonaProfile,
)

_ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_DIRECT_NOTIFY_REQUEST = ("联系", "通知", "打电话给", "发消息给")
_TELL_ME_RE = re.compile(r"告诉(?:一下)?(?:我|我们|咱们)")
_TELL_OUTBOUND_RE = re.compile(
    r"告诉(?:一下)?(?:给)?(?!我|我们|咱们)[^，。！？?]{1,16}"
)
_INFORMATION_QUERY_MARKERS = (
    "告诉我", "告诉一下我", "跟我说", "问问", "询问", "查一下", "查询",
    "几点", "什么时候", "是否", "有没有", "怎么样", "怎么回事", "回复了吗",
)
_EXPLICIT_CONFIRMATION = (
    "确认执行", "确认创建", "确认提醒", "就这么办", "就按这个",
    "按这个来", "可以执行", "现在执行",
)
_REMINDER_REQUEST = ("提醒", "闹钟", "到点叫", "到时候叫")
_MEDICATION_CONTEXT = ("药", "服药", "吃药", "用药")
_PENDING_ACTION_KEY = "_pending_external_action"
_PENDING_ACTION_TTL_TURNS = 2
_BUILTIN_PENDING_ARG_ALLOWLIST = {
    "medication_reminder": {"medication", "when"},
    "notify_family": {"contact"},
}
_ACTION_VERB_PATTERN = (
    r"(?:联系|通知|告诉|打电话给|发消息给|提醒|闹钟|到点叫|到时候叫|"
    r"执行|创建)"
)
_NEGATED_ACTION_RE = re.compile(
    rf"(?:不|别|不要|不用|无需|不能|先别|暂不|暂时别|取消|停止)"
    rf"[^，。！？?]{{0,6}}{_ACTION_VERB_PATTERN}"
)
_NON_CURRENT_ACTION_RE = re.compile(
    rf"(?:如果|要是|假如|万一|昨天|前天|以前|上次|之前|当时|"
    rf"刚才|刚刚|已经|曾经)[^，。！？?]{{0,18}}{_ACTION_VERB_PATTERN}"
)
_REPORTED_ACTION_RE = re.compile(
    rf"(?:他|她|医生|护士|护理员|女儿|儿子|新闻|电视)"
    rf"[^，。！？?]{{0,8}}(?:说|提到|让我|叫我)"
    rf"[^，。！？?]{{0,8}}{_ACTION_VERB_PATTERN}"
)


def _action_language_is_current_request(utterance: str) -> bool:
    compact = str(utterance or "").strip()
    return not (
        _NEGATED_ACTION_RE.search(compact)
        or _NON_CURRENT_ACTION_RE.search(compact)
        or _REPORTED_ACTION_RE.search(compact)
    )


def _is_confirmation(utterance: str) -> bool:
    if not _action_language_is_current_request(utterance):
        return False
    decision: bool | None = None
    clauses = [
        item.strip()
        for item in re.split(r"[，,。！？?；;\n]+", utterance)
        if item.strip()
    ]
    for clause in clauses:
        occurrences: list[tuple[int, str]] = []
        for marker in _EXPLICIT_CONFIRMATION:
            start = 0
            while (at := clause.find(marker, start)) >= 0:
                occurrences.append((at, marker))
                start = at + len(marker)
        for at, _marker in sorted(occurrences):
            prefix = clause[max(0, at - 7):at]
            decision = not any(
                negation in prefix
                for negation in (
                    "不", "别", "不要", "不能", "先别", "暂不", "取消",
                )
            )
    return decision is True


def _has_outbound_notification_language(utterance: str) -> bool:
    """区分“通知女儿”与“告诉我女儿几点来”这类信息查询。"""
    compact = str(utterance or "").strip()
    if (
        _TELL_ME_RE.search(compact)
        or not _action_language_is_current_request(compact)
    ):
        return False
    return (
        any(marker in compact for marker in _DIRECT_NOTIFY_REQUEST)
        or bool(_TELL_OUTBOUND_RE.search(compact))
    )


def _is_information_query(utterance: str) -> bool:
    compact = str(utterance or "").strip()
    return (
        any(marker in compact for marker in _INFORMATION_QUERY_MARKERS)
    )


def _contact_is_outbound_target(utterance: str, contact: str) -> bool:
    compact = str(utterance or "").strip()
    escaped = re.escape(contact)
    linked = re.search(
        rf"(?:联系(?:一下)?|通知(?:一下)?|打电话给|发消息给|"
        rf"告诉(?:一下)?(?:给)?)"
        rf"\s*{escaped}",
        compact,
    )
    return bool(linked) and not _is_information_query(compact)


def _has_direct_medication_reminder(utterance: str) -> bool:
    compact = str(utterance or "").strip()
    return (
        _action_language_is_current_request(compact)
        and any(marker in compact for marker in _REMINDER_REQUEST)
        and any(marker in compact for marker in _MEDICATION_CONTEXT)
    )


def stage_pending_action(
    ctx: EngineContext,
    tool: str,
    args: dict[str, Any],
    *,
    action_id: str | None = None,
    ttl_turns: int = _PENDING_ACTION_TTL_TURNS,
    allowed_args: set[str] | None = None,
) -> dict[str, Any]:
    """由可信服务端暂存待确认动作；模型输出本身不能创建这条记录。"""
    clean_tool = str(tool or "").strip()
    if not clean_tool:
        raise ValueError("pending action tool is required")
    if not isinstance(args, dict):
        raise TypeError("pending action args must be a dict")
    allowlist = (
        set(allowed_args)
        if allowed_args is not None
        else _BUILTIN_PENDING_ARG_ALLOWLIST.get(clean_tool, set())
    )
    clean_args = {
        key: value for key, value in args.items()
        if key in allowlist
    }
    current_turn = len(ctx.turns)
    record = {
        "id": str(action_id or uuid4().hex),
        "tool": clean_tool,
        "args": clean_args,
        "created_turn": current_turn,
        "expires_turn": current_turn + max(1, int(ttl_turns)),
    }
    ctx.working_memory[_PENDING_ACTION_KEY] = record
    return {**record, "args": dict(record["args"])}


def current_pending_action(ctx: EngineContext) -> dict[str, Any] | None:
    """只返回结构完整且尚未过期的服务端待确认动作。"""
    value = ctx.working_memory.get(_PENDING_ACTION_KEY)
    if not isinstance(value, dict):
        return None
    required = ("id", "tool", "args", "created_turn", "expires_turn")
    if any(key not in value for key in required):
        ctx.working_memory.pop(_PENDING_ACTION_KEY, None)
        return None
    if (
        not isinstance(value.get("id"), str)
        or not value["id"].strip()
        or not isinstance(value.get("tool"), str)
        or not value["tool"].strip()
        or not isinstance(value.get("args"), dict)
        or not isinstance(value.get("created_turn"), int)
        or not isinstance(value.get("expires_turn"), int)
        or value["created_turn"] > len(ctx.turns)
        or value["expires_turn"] < len(ctx.turns)
    ):
        ctx.working_memory.pop(_PENDING_ACTION_KEY, None)
        return None
    return {**value, "args": dict(value["args"])}


def consume_pending_action(ctx: EngineContext, action_id: str) -> None:
    """成功回执后才消费；失败时保留，让用户可重试或取消。"""
    pending = current_pending_action(ctx)
    if pending is not None and pending["id"] == action_id:
        ctx.working_memory.pop(_PENDING_ACTION_KEY, None)


def is_pending_confirmation(
    utterance: str,
    pending_action: dict[str, Any] | None,
) -> bool:
    """泛确认只有在服务端确有待确认动作时才具有动作含义。"""
    compact = str(utterance or "").strip()
    direct_medication = _has_direct_medication_reminder(compact)
    return bool(
        pending_action is not None
        and _is_confirmation(compact)
        and not _has_outbound_notification_language(compact)
        and not direct_medication
    )


def has_current_action_intent(
    utterance: str,
    pending_action: dict[str, Any] | None = None,
) -> bool:
    """只有当轮明确动作意图才向生成模型暴露工具协议。"""
    compact = str(utterance or "").strip()
    notify = _has_outbound_notification_language(compact)
    medication_reminder = _has_direct_medication_reminder(compact)
    confirmed = is_pending_confirmation(compact, pending_action)
    return notify or medication_reminder or confirmed


def parse_action(raw: str, *, source_positions=None) -> tuple[str, Optional[dict[str, Any]]]:
    """返回 (用户可见文本, 动作 dict 或 None)。动作仅取第一个。"""
    matches = list(_ACTION_RE.finditer(raw))
    if source_positions is not None:
        kept = [index for index in range(len(raw))
                if not any(match.start() <= index < match.end() for match in matches)]
        untrimmed = "".join(raw[index] for index in kept)
        start = len(untrimmed) - len(untrimmed.lstrip())
        source_positions[:] = kept[start:start + len(untrimmed.strip())]
    if not matches:
        return raw.strip(), None
    m = matches[0]
    # 模型偶尔会重复发射协议标签；只解析第一个，但所有完整标签都从
    # 用户可见文本剥离，既不多次执行，也不把内部 JSON 泄露到界面。
    clean = _ACTION_RE.sub("", raw).strip()
    try:
        data = json.loads(m.group(1))
    except Exception:
        return clean, None
    if not isinstance(data, dict) or not isinstance(data.get("tool"), str) or not data["tool"]:
        return clean, None
    args = data.get("args")
    parsed = {
        "tool": data["tool"],
        "args": args if isinstance(args, dict) else {},
    }
    pending_action_id = data.get("pending_action_id")
    if isinstance(pending_action_id, str) and pending_action_id.strip():
        parsed["pending_action_id"] = pending_action_id.strip()
    return clean, parsed


def prepare_action(
    action: dict[str, Any],
    utterance: str,
    persona: Optional[PersonaProfile],
    *,
    executed: Optional[set[str]] = None,
    pending_action: dict[str, Any] | None = None,
    duplicate_is_attempt: bool = False,
) -> tuple[Optional[dict[str, Any]], str]:
    """确定性授权、补齐参数并去重；绝不只信 prompt 或模型自报的同意。"""
    tool = str(action.get("tool") or "")
    args = dict(action.get("args") or {})
    compact = utterance.strip()
    direct_notify = _has_outbound_notification_language(compact)
    direct_medication = _has_direct_medication_reminder(compact)
    generic_confirmation = (
        _is_confirmation(compact)
        and not direct_notify
        and not direct_medication
    )
    bound_pending_id = ""
    if generic_confirmation:
        if pending_action is None:
            return None, "没有可核对的待确认动作，请重新说清要做什么"
        supplied_id = str(action.get("pending_action_id") or "").strip()
        expected_id = str(pending_action.get("id") or "").strip()
        if not supplied_id or supplied_id != expected_id:
            return None, "确认未绑定到服务端待确认动作"
        if tool != str(pending_action.get("tool") or ""):
            return None, "确认的动作类型与待确认动作不一致"
        pending_args = pending_action.get("args")
        if not isinstance(pending_args, dict):
            return None, "服务端待确认动作参数无效"
        if args and args != pending_args:
            return None, "模型动作参数与服务端待确认参数不一致"
        args = dict(pending_args)
        bound_pending_id = expected_id

    if tool == "notify_family":
        if not direct_notify and not generic_confirmation:
            return None, "需要当轮明确说出要联系或通知谁"
        if direct_notify and _is_information_query(compact):
            return None, "通知工具不能代替查询联系人信息或获取对方回复"
        contacts = (
            persona.care_plan.emergency_contacts
            if persona is not None and persona.care_plan is not None else []
        )
        consented = [
            item for item in contacts
            if item.has_auditable_notification_consent(
                AD_HOC_NOTIFICATION_CONSENT_SCOPES
            )
        ]
        if not consented:
            return None, "档案中没有范围、来源和时间均有效的通知授权"
        requested_contact = args.get("contact")
        if not isinstance(requested_contact, str):
            return None, "通知动作必须明确指定联系人，不能自动替换"
        requested_contact = requested_contact.strip()
        if not requested_contact:
            return None, "通知动作必须明确指定联系人，不能自动替换"
        match = next(
            (item for item in consented if item.name == requested_contact),
            None,
        )
        if match is None:
            return None, "模型指定的联系人不在有效授权名单中"
        if direct_notify and not _contact_is_outbound_target(
            compact, match.name
        ):
            return None, "当轮没有明确要求联系模型指定的联系人"
        # 模型自由生成的 reason/message 可能夹带未授权隐私。对外工具只接收
        # 服务端白名单字段和固定原因码，不转发模型原文或完整对话。
        args = {
            "contact": match.name,
            "reason_code": "user_requested_contact",
        }
    elif tool == "medication_reminder":
        if not (direct_medication or generic_confirmation):
            return None, "需要当轮明确提出用药提醒或确认执行"
    elif not generic_confirmation:
        return None, "需要当轮给出与该工具相关的意图或明确确认"

    if tool == "medication_reminder":
        args = {
            key: value.strip() for key, value in args.items()
            if key in {"medication", "when"}
            and isinstance(value, str) and value.strip()
        }
        medications = (
            persona.care_plan.medications
            if persona is not None and persona.care_plan is not None else []
        )
        if direct_medication and len(medications) == 1:
            args.setdefault("medication", medications[0].name)
            args.setdefault(
                "when", medications[0].at or medications[0].schedule or "按计划"
            )
        if direct_medication:
            medication = str(args.get("medication") or "").strip()
            when = str(args.get("when") or "").strip()
            if not medication:
                return None, "无法从当轮原话或唯一护理计划核对药物名称"
            matching_medications = [
                item for item in medications if item.name == medication
            ]
            if medication not in compact and not (
                len(medications) == 1 and matching_medications
            ):
                return None, "模型给出的药物名称无法由当轮原话或护理计划核对"
            if not when:
                return None, "无法从当轮原话或护理计划核对提醒时间"
            selected_times = {
                value for item in matching_medications
                for value in (item.at, item.schedule)
                if value
            }
            if when not in compact and when not in selected_times:
                return None, "模型给出的提醒时间无法由当轮原话或护理计划核对"
        elif generic_confirmation and (
            not str(args.get("medication") or "").strip()
            or not str(args.get("when") or "").strip()
        ):
            return None, "服务端待确认的用药提醒缺少药物或时间"

    signature = json.dumps(
        {"tool": tool, "args": args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if executed is not None and signature in executed:
        if duplicate_is_attempt:
            return None, "本会话已有同一动作的尝试记录，不重复提交；这不证明先前动作已完成"
        return None, "同一动作本会话已经执行，不重复提交"
    if executed is not None:
        executed.add(signature)
    prepared = {"tool": tool, "args": args}
    if bound_pending_id:
        prepared["pending_action_id"] = bound_pending_id
    return prepared, ""
