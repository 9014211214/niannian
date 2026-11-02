"""Shared action receipt identity and terminal-status checks."""
from __future__ import annotations

from ..context import EngineContext
from ..model.persona import PersonaProfile


_GENERIC_EXTERNAL_ACTORS = (
    "工作人员", "护理人员", "护理员", "照护人", "家里人", "家属",
    "孙女", "孙子", "外孙女", "外孙", "女儿", "儿子", "老伴", "爱人",
    "护士", "医生", "药师", "社工", "社区", "邻居", "朋友", "对方",
    "收件人",
)


_TERMINAL_NOTIFICATION_STATUSES = {"delivered", "notified"}


def _actor_aliases(persona: PersonaProfile | None) -> dict[str, str]:
    aliases = {item: item for item in _GENERIC_EXTERNAL_ACTORS}
    if persona is not None:
        for item in persona.referenceable_relationships()[-6:]:
            canonical = item.name.strip() or item.relation.strip()
            if not canonical:
                continue
            if item.name.strip():
                aliases[item.name.strip()] = canonical
            if item.relation.strip():
                aliases[item.relation.strip()] = canonical
    return aliases


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
