"""Ground final claims about real-world actions and third-party events.

This guard deliberately runs *after* tools and deterministic scene skills.  A
model may therefore say an action completed only when the matching adapter has
already returned a successful receipt.  Claims about what a relative or
professional did are a different trust domain: a notification receipt proves
only that Aria sent something, never that the recipient replied or acted.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ...engine.context import EngineContext
from ...engine.model.persona import PersonaProfile
from ...engine.policy.reply_contract import (
    UNAVAILABLE_CONTACT_REPLY, contact_offer_violations, reply_quality_contract,
    physical_offer_violations,
)
from .device_grounding import (
    UNCONFIRMED_DEVICE_CAPABILITY_REPLY, device_capability_violations,
)


UNVERIFIED_ACTION_REPLY = (
    "这件事我还没有实际执行。要不要做，先由你决定。"
)
UNGROUNDED_EXTERNAL_EVENT_REPLY = (
    "我没有依据确认这件事已经发生，先不把它说成事实。"
)

_GENERIC_EXTERNAL_ACTORS = (
    "工作人员", "护理人员", "护理员", "照护人", "家里人", "家属",
    "孙女", "孙子", "外孙女", "外孙", "女儿", "儿子", "老伴", "爱人",
    "护士", "医生", "药师", "社工", "社区", "邻居", "朋友", "对方",
    "收件人",
)
_THIRD_PERSON_PRONOUNS = ("他们", "她们", "他", "她")
_AGENT_ASSISTANCE = ("我", "替你", "帮你", "给你", "为你")
_MANAGED_SUBJECTS = ("提醒", "闹钟", "通知", "消息", "报名", "订单")
_PAST_MARKERS = ("已经", "已", "刚刚", "刚才", "刚", "早已")
_NEGATIONS = ("没有", "没", "未", "尚未", "不曾", "并未", "还没")
_UNCERTAIN_OR_HYPOTHETICAL = (
    "如果", "假如", "要是", "可能", "也许", "或许", "大概", "听说",
    "据说", "预计", "准备", "打算", "将会", "会不会",
)
_TERMINAL_NOTIFICATION_STATUSES = {"delivered", "notified"}
_EXPLICIT_NON_COMPLETION = (
    "这个动作还没有执行", "还没有实际执行", "没有创建成功",
    "尚未执行", "未执行",
)

_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "reminder",
        re.compile(
            r"(?:设置|创建|建立).{0,8}(?:提醒|闹钟)|"
            r"(?:提醒|闹钟).{0,8}(?:设置|创建|建立)"
        ),
    ),
    (
        "notification",
        re.compile(
            r"通知|联系|发消息|发送消息|打电话|叫(?:人|工作人员)|"
            r"(?:消息|通知).{0,6}(?:发出|发送)"
        ),
    ),
    (
        "arrangement",
        re.compile(r"安排|预约|约好|挂号|报名|取消|撤回|撤掉"),
    ),
    ("financial", re.compile(r"转账|付款|支付|下单")),
    ("reply", re.compile(r"回复|回话|回信|回电|答应")),
    ("recipient_receipt", re.compile(r"收到|看见|看到|已读")),
    ("delivery", re.compile(r"送来|拿来|带来|送到|拿到")),
)
_CLAUSE_SPLIT = re.compile(r"[。！？；;，,\n]+")
_READING_PROGRESS_QUERY = re.compile(
    r"(?:你|您)?(?:(?:这|那)(?:本|份))?(?:书|小说|报纸|杂志|文章|漫画)(?:你|您)?"
    r"看到(?:哪(?:儿|里|一页|页|一章|章)|第几(?:页|章)|多少页)(?:了|呢)?"
)
_USER_FEEDBACK_CONDITION = re.compile(
    r"(?:你|您)?看到了(?:就|再)告诉我(?:一声)?(?:吧|就好|就行)?"
)


def _actor_aliases(persona: PersonaProfile | None) -> dict[str, str]:
    aliases = {item: item for item in _GENERIC_EXTERNAL_ACTORS}
    if persona is not None:
        for item in persona.relationship_context[-6:]:
            if not item.referenceable:
                continue
            canonical = item.name.strip() or item.relation.strip()
            if not canonical:
                continue
            if item.name.strip():
                aliases[item.name.strip()] = canonical
            if item.relation.strip():
                aliases[item.relation.strip()] = canonical
    return aliases


def _is_completed_claim(clause: str, match: re.Match[str]) -> bool:
    # "书看到哪儿了" asks reading progress; its final 了 is not an external
    # delivery/receipt assertion. Do not exempt every question: a sentence can
    # still assert "我已通知女儿了" before asking something else. The complete
    # clause must match this finite non-receipt use of 看到.
    if match.group(0) == "看到" and _READING_PROGRESS_QUERY.fullmatch(clause):
        return False
    # In this complete, finite feedback instruction, 了 belongs to the
    # prospective condition ("when you see it"), not a claim of receipt.
    # Do not exempt a whole response, explicit 已经, or a third-party subject;
    # adjacent completed actions still go through their own receipt checks.
    if match.group(0) == "看到" and _USER_FEEDBACK_CONDITION.fullmatch(clause):
        return False
    prefix = clause[max(0, match.start() - 12):match.start()]
    suffix = clause[match.end():match.end() + 14]
    if any(marker in prefix for marker in _NEGATIONS):
        return False
    if any(
        marker in prefix for marker in _UNCERTAIN_OR_HYPOTHETICAL
    ):
        return False
    matched = match.group(0)
    return (
        any(marker in prefix for marker in _PAST_MARKERS)
        or any(marker in matched for marker in _PAST_MARKERS)
        or re.search(r"(?:好|完|妥|成功|完成|到)?了", suffix) is not None
        or suffix.startswith(("好", "完", "妥", "成功", "完成"))
    )


def _actor_mentions(
    clause: str,
    aliases: dict[str, str],
) -> list[tuple[int, str, str]]:
    mentions: list[tuple[int, str, str]] = []
    for alias, canonical in aliases.items():
        start = clause.find(alias)
        while start >= 0:
            mentions.append((start, alias, canonical))
            start = clause.find(alias, start + len(alias))
    for pronoun in _THIRD_PERSON_PRONOUNS:
        start = clause.find(pronoun)
        while start >= 0:
            mentions.append((start, pronoun, pronoun))
            start = clause.find(pronoun, start + len(pronoun))
    return sorted(set(mentions), key=lambda item: (item[0], -len(item[1])))


def _normalise_pronoun(actor: str, active_actor: str) -> str:
    if actor in _THIRD_PERSON_PRONOUNS and active_actor:
        return active_actor
    return actor


def _extract_claims(
    text: str,
    persona: PersonaProfile | None,
    *,
    require_completion: bool,
) -> list[dict[str, str]]:
    """Extract action/event claims while retaining the grammatical actor."""
    aliases = _actor_aliases(persona)
    claims: list[dict[str, str]] = []
    active_actor = ""
    for clause in (
        item.strip() for item in _CLAUSE_SPLIT.split(text) if item.strip()
    ):
        mentions = _actor_mentions(clause, aliases)
        explicit_mentions = [
            item for item in mentions
            if item[1] not in _THIRD_PERSON_PRONOUNS
        ]
        if explicit_mentions:
            active_actor = explicit_mentions[-1][2]
        elif mentions:
            active_actor = _normalise_pronoun(
                mentions[-1][2], active_actor
            )

        for category, pattern in _ACTION_PATTERNS:
            for match in pattern.finditer(clause):
                prefix = clause[max(0, match.start() - 10):match.start()]
                if any(marker in prefix for marker in _NEGATIONS):
                    continue
                if require_completion and not _is_completed_claim(
                    clause, match
                ):
                    continue

                subjects = [item for item in mentions if item[0] < match.start()]
                actor = (
                    _normalise_pronoun(subjects[-1][2], active_actor)
                    if subjects else ""
                )
                objects = [
                    item for item in mentions if item[0] >= match.end()
                ]
                target = (
                    _normalise_pronoun(objects[0][2], active_actor)
                    if objects else ""
                )
                agent_before = any(
                    marker in clause[:match.start()]
                    for marker in _AGENT_ASSISTANCE
                )
                if actor:
                    owner = "third_party"
                elif agent_before or any(
                    marker in clause for marker in _MANAGED_SUBJECTS
                ):
                    owner = "agent"
                elif require_completion:
                    owner = "external_fact"
                else:
                    owner = ""
                claims.append({
                    "owner": owner,
                    "actor": actor,
                    "target": target,
                    "category": category,
                })
    return claims


def _grounding_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _grounding_sources(
    ctx: EngineContext,
    utterance: str,
    persona: PersonaProfile | None,
) -> list[str]:
    sources = [str(utterance or "")]
    sources.extend(
        _grounding_text(item)
        for item in list(
            ctx.working_memory.get("recalled_episodes") or []
        )[:2]
    )
    if persona is not None:
        for item in persona.relationship_context[-6:]:
            if item.referenceable:
                sources.extend(str(fact) for fact in item.facts[:3])
    return sources


def _third_party_claim_is_grounded(
    claim: dict[str, str],
    sources: list[str],
    persona: PersonaProfile | None,
) -> bool:
    for source in sources:
        for grounded in _extract_claims(
            source,
            persona,
            require_completion=True,
        ):
            if grounded["category"] != claim["category"]:
                continue
            if claim["actor"] and grounded["actor"] != claim["actor"]:
                continue
            return True
    return False


def _notification_delivery_confirmed(receipt: dict, data: dict) -> bool:
    # ok only describes the adapter call. Explicit pending/unknown/conflicting
    # statuses cannot be promoted into terminal delivery by a truthy flag.
    statuses = [source["status"] for source in (receipt, data)
                if "status" in source]
    if any(not isinstance(status, str)
           or status.strip().lower() not in _TERMINAL_NOTIFICATION_STATUSES
           for status in statuses):
        return False
    if statuses:
        return len({status.strip().lower() for status in statuses}) == 1
    # Compatibility for adapters with an explicit terminal boolean receipt.
    return data.get("notified") is True or data.get("delivered") is True


def _successful_receipts(
    ctx: EngineContext,
    persona: PersonaProfile | None,
) -> list[dict[str, str]]:
    successful: list[dict[str, str]] = []
    aliases = _actor_aliases(persona)
    receipts = [
        ctx.working_memory.get("_consumed_tool_result_for_turn"),
        ctx.working_memory.get("last_tool_result"),
    ]
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("ok") is not True:
            continue
        data = receipt.get("data")
        data = data if isinstance(data, dict) else {}
        if any(
            data.get(field) is False
            for field in ("executed", "persisted", "notified", "delivered")
        ):
            continue
        category = {
            "medication_reminder": "reminder",
            "notify_family": "notification",
        }.get(str(receipt.get("tool") or ""))
        if not category:
            continue
        if category == "notification" and not _notification_delivery_confirmed(receipt, data):
            continue
        args = receipt.get("args")
        args = args if isinstance(args, dict) else {}
        target = str(args.get("contact") or "").strip()
        successful.append({
            "category": category,
            "target": aliases.get(target, target),
        })
    care_action = ctx.working_memory.get("last_care_action")
    if isinstance(care_action, dict) and care_action.get("ok") is True:
        category = {
            "reminder_created": "reminder",
        }.get(str(care_action.get("action") or ""))
        if category:
            successful.append({"category": category, "target": ""})
    return successful


def _receipt_matches_claim(
    claim: dict[str, str],
    receipts: list[dict[str, str]],
) -> bool:
    for receipt in receipts:
        if receipt["category"] != claim["category"]:
            continue
        if claim.get("target"):
            # A named outbound target must be present in, and match, the
            # server-side prepared args.  Category-only success is not proof.
            if not receipt.get("target"):
                continue
            if receipt["target"] != claim["target"]:
                continue
        return True
    return False


def _minimal_care_record_claims(ctx: EngineContext, candidate: str) -> tuple[str, bool]:
    """Exclude only narrowly verified record descriptions from action claims.

    A saved care snapshot is never an action receipt. Only whole state clauses
    after an explicit record prefix can be checked here; adjacent agent or
    third-party claims stay in the original guard. Labels are not templates.
    """
    snapshot = ctx.working_memory.get("minimal_care_context")
    if not isinstance(snapshot, dict):
        return candidate, False
    available = (
        snapshot.get("source") == "minimal_care_records"
        and snapshot.get("status") == "available"
        and snapshot.get("read_only") is True
        and snapshot.get("external_enabled") is False
    )

    def unique(kind):
        rows = snapshot.get(kind)
        counts = snapshot.get("counts")
        count = counts.get(kind) if isinstance(counts, dict) else None
        if not available or not isinstance(rows, list) or len(rows) != 1:
            return None
        if not isinstance(count, dict) or any(
            type(count.get(key)) is not int or count[key] != expected
            for key, expected in (("total", 1), ("included", 1), ("omitted", 0))
        ):
            return None
        row = rows[0]
        if not isinstance(row, dict) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", str(row.get("id") or "")
        ):
            return None
        return row

    reminder = unique("reminders")
    consent = unique("consents")
    if (consent is not None and (
        reminder is None or consent.get("reminder_id") != reminder["id"]
        or consent.get("source") != "simulation"
    )):
        consent = None
    supported = {
        "cancelled": reminder is not None and reminder.get("status") == "cancelled",
        "created": reminder is not None and reminder.get("status") == "active",
        "revoked": consent is not None and consent.get("status") == "revoked"
                   and consent.get("valid_now") is False,
        "valid": consent is not None and consent.get("status") == "active"
                 and consent.get("valid_now") is True,
    }
    patterns = (
        ("cancelled", r"(?:这个)?提醒(?:已经|已)取消(?:了)?"),
        ("created", r"(?:这个)?提醒(?:已经|已)(?:创建|设置)(?:好)?(?:了)?"),
        ("revoked", r"(?:这条)?授权(?:已经|已)撤回(?:了)?"),
        ("valid", r"(?:这条)?授权(?:当前有效|仍有效|已生效|已经生效)"),
    )
    invalid = False
    # Keep sentence boundaries: a record prefix cannot cover a later sentence.
    sentences = re.split(r"([。！？；;\n]+)", candidate)
    for index in range(0, len(sentences), 2):
        sentence = sentences[index]
        prefix = re.match(r"\s*(?:页面记录|护理记录)显示\s*[，,:：]\s*", sentence)
        if prefix is None:
            continue
        clauses = re.split(r"([，,])", sentence[prefix.end():])
        for clause_index in range(0, len(clauses), 2):
            clause = clauses[clause_index].strip()
            for state, pattern in patterns:
                if not re.fullmatch(pattern, clause):
                    continue
                if supported[state]:
                    # Remove only this checked clause, never the entire reply.
                    clauses[clause_index] = "（已核对记录状态）"
                else:
                    invalid = True
                break
        sentences[index] = sentence[:prefix.end()] + "".join(clauses)
    return "".join(sentences), invalid


def _record_guard(
    ctx: EngineContext,
    violations: list[str],
) -> None:
    trace = {
        "activated": bool(violations),
        "response_changed": False,
        "violations": list(dict.fromkeys(violations)),
    }
    ctx.working_memory["delivery_response_guard"] = trace
    online = ctx.working_memory.get("online_turn_decision")
    if not isinstance(online, dict) or not violations:
        return
    integrated = online.get("delivery_guard")
    if not isinstance(integrated, dict):
        integrated = {
            "activated": False,
            "violations": [],
            "action_removed": False,
        }
    integrated["activated"] = True
    integrated["violations"] = list(dict.fromkeys([
        *list(integrated.get("violations") or []),
        *violations,
    ]))
    online["delivery_guard"] = integrated


def guard_external_delivery_response(
    ctx: EngineContext,
    response: str,
    utterance: str,
    persona: PersonaProfile | None,
) -> str:
    """Enforce receipts for Aria actions and grounding for third parties."""
    compact = " ".join(str(response or "").strip().split())
    if not compact:
        _record_guard(ctx, [])
        return compact
    candidate = str(
        ctx.working_memory.get("_delivery_response_candidate_for_turn")
        or compact
    )
    checked_candidate, invalid_care_status = _minimal_care_record_claims(ctx, candidate)
    claims = _extract_claims(checked_candidate, persona, require_completion=True)
    receipts = _successful_receipts(ctx, persona)
    sources = _grounding_sources(ctx, utterance, persona)
    # Use the actual deliverable, not a superseded model candidate, for future
    # offers. This also covers the unstructured fast path without another LLM.
    future_violations = contact_offer_violations(compact, reply_quality_contract(ctx))
    physical_violations = physical_offer_violations(compact)
    device_violations = device_capability_violations(compact, ctx, utterance)
    violations: list[str] = [*future_violations, *physical_violations, *device_violations]
    if invalid_care_status:
        violations.append("unverified_minimal_care_record_status")
    for claim in claims:
        if claim["owner"] == "agent":
            if not _receipt_matches_claim(claim, receipts):
                violations.append("unverified_external_action_completion")
        elif claim["owner"] in {"third_party", "external_fact"}:
            if not _third_party_claim_is_grounded(
                claim, sources, persona
            ):
                violations.append("ungrounded_third_party_completion")
    _record_guard(ctx, violations)
    if violations and not invalid_care_status and not future_violations and not physical_violations and not device_violations and any(
        marker in compact for marker in _EXPLICIT_NON_COMPLETION
    ):
        # 工具失败路径已经给出了更具体、真实的失败原因，不要再用通用
        # 模板覆盖它；候选中的虚假完成态仍保留在审计 trace 中。
        return compact
    safe = compact
    if "unverified_external_action_completion" in violations:
        safe = UNVERIFIED_ACTION_REPLY
    elif ("ungrounded_third_party_completion" in violations
          or "unverified_minimal_care_record_status" in violations):
        safe = UNGROUNDED_EXTERNAL_EVENT_REPLY
    elif future_violations:
        contract = reply_quality_contract(ctx)
        safe = (
            "我听见了。你不用现在解释，我先陪你待一会儿。"
            if contract["observation_mode"] == "wait" or contract["person_requires_space"]
            else UNAVAILABLE_CONTACT_REPLY
        )
    elif physical_violations:
        safe = "我没有身体，也没有接入能执行这件事的设备，不能替你动手完成。"
    elif device_violations:
        safe = UNCONFIRMED_DEVICE_CAPABILITY_REPLY
    if safe != compact:
        ctx.working_memory["delivery_response_guard"]["response_changed"] = True
        from ...engine.policy.online_decision import _controlled_wait_task_communication

        # A compound violation must not return before the independent task is
        # restored. Never reuse the discarded model's action or completion.
        task_message, task_trace = _controlled_wait_task_communication(ctx)
        if task_message:
            safe = f"{safe}\n{task_message}"
            ctx.working_memory["delivery_response_guard"]["task_communication"] = task_trace
        return safe
    return compact


__all__ = [
    "UNGROUNDED_EXTERNAL_EVENT_REPLY",
    "UNVERIFIED_ACTION_REPLY",
    "guard_external_delivery_response",
]
