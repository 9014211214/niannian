"""Bounded, read-only care facts for the minimal companion's current turn.

The care service remains the only owner of schedules, consent and deliveries.
This projection never authorizes or executes an action and never exposes contact
details, free-form follow-up notes, authentication data or notification bodies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re

_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_REMINDER_STATUSES = {"active", "cancelled", "completed"}
_OCCURRENCE_STATUSES = {
    "pending", "unconfirmed", "snoozed", "acknowledged", "declined",
    "cancelled", "superseded",
}
_DELIVERY_STATUSES = {
    "queued", "sending", "accepted", "delivered", "delivered_local",
    "family_ack", "unknown", "failed", "cancelled",
}


def _choice(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def _identifier(value):
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _channel(value):
    # Arbitrary adapter names can contain account/endpoint information. Keep
    # only known channel classes; do not forward a custom adapter's identifier.
    return _choice(value, {"local_care_inbox", "https_webhook"})


def _occurrence(row):
    if row is None:
        return None
    return {
        "id": _identifier(row.get("id")),
        "reminder_id": _identifier(row.get("reminder_id")),
        "status": _choice(row.get("status"), _OCCURRENCE_STATUSES),
        "timezone": _choice(row.get("timezone"), {"UTC", "Asia/Shanghai"}),
        "delivery_status": _choice(row.get("delivery_status"), {"offered_local"}),
        **{key: _timestamp(row.get(key)) for key in (
            "due_at", "offered_at", "deadline_at", "responded_at",
            "snoozed_until", "superseded_at",
        )},
    }


def _consent(row, now):
    start, expiry = _timestamp(row.get("valid_from")), _timestamp(row.get("expires_at"))
    valid_now = (row.get("status") == "active" and start is not None and expiry is not None
                 and datetime.fromisoformat(start) <= now < datetime.fromisoformat(expiry))
    return {
        "id": _identifier(row.get("id")),
        "contact_id": _identifier(row.get("contact_id")),
        "reminder_id": _identifier(row.get("reminder_id")),
        "status": _choice(row.get("status"), {"active", "revoked"}),
        "source": _choice(row.get("source"), {"simulation", "real"}),
        "trigger": _choice(row.get("trigger"), {"unconfirmed"}),
        "scope": _choice(row.get("scope"), {"reminder_status"}),
        "valid_from": start, "expires_at": expiry,
        "revoked_at": _timestamp(row.get("revoked_at")),
        # This is a time/status fact, never an execution permission.
        "valid_now": valid_now,
    }


def _delivery(row):
    history = row.get("receipt_history")
    last = history[-1] if isinstance(history, list) and history else {}
    last = last if isinstance(last, dict) else {}
    return {
        "id": _identifier(row.get("id")), "event_id": _identifier(row.get("event_id")),
        "occurrence_id": _identifier(row.get("occurrence_id")),
        "contact_id": _identifier(row.get("contact_id")),
        "consent_id": _identifier(row.get("consent_id")),
        "status": _choice(row.get("status"), _DELIVERY_STATUSES),
        "source": _choice(row.get("source"), {"simulation", "real"}),
        "channel": _channel(row.get("channel")),
        "created_at": _timestamp(row.get("created_at")),
        "attempted_at": _timestamp(row.get("attempted_at")),
        "receipt_at": _timestamp(last.get("at")),
        "receipt_source": _choice(row.get("receipt_source", last.get("source")),
                                  {"local_adapter", "gateway", "caregiver_report"}),
    }


def _project(view, elder_id, now):
    if type(view.get("configured")) is not bool or type(view.get("external_enabled")) is not bool:
        raise ValueError("invalid care view")
    rows = {}
    for kind in ("reminders", "occurrences", "consents", "deliveries", "tasks"):
        values = view[kind]
        if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
            raise ValueError("invalid care records")
        # The service already scopes elder_view. Retain a second explicit
        # boundary so an accidentally broader view cannot reach this prompt.
        rows[kind] = [row for row in values if row.get("elder_id") == elder_id]

    projected = {kind: [] for kind in rows}
    for row in reversed(rows["reminders"][-3:]):
        reminder_id = row.get("id")
        occurrences = [item for item in rows["occurrences"] if item.get("reminder_id") == reminder_id]
        latest = occurrences[-1] if occurrences else None
        occurrence_id = latest.get("id") if latest else None
        consents = [item for item in rows["consents"] if item.get("reminder_id") == reminder_id]
        deliveries = [item for item in rows["deliveries"]
                      if occurrence_id is not None and item.get("occurrence_id") == occurrence_id]
        tasks = [item for item in rows["tasks"]
                 if occurrence_id is not None and item.get("occurrence_id") == occurrence_id]
        label = row.get("label")
        projected["reminders"].append({
            "id": _identifier(reminder_id),
            "label": _CONTROL.sub("", label)[:120] if isinstance(label, str) else None,
            "status": _choice(row.get("status"), _REMINDER_STATUSES),
            "schedule": _choice(row.get("schedule"), {"once", "daily"}),
            "date": row.get("date") if isinstance(row.get("date"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]) else None,
            "time": row.get("time") if isinstance(row.get("time"), str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", row["time"]) else None,
            "timezone": _choice(row.get("timezone"), {"UTC", "Asia/Shanghai"}),
            "next_due_at": _timestamp(row.get("next_due_at")),
            "grace_minutes": _count(row.get("grace_minutes")),
            "skipped_total": _count(row.get("skipped_total", 0)),
        })
        if latest:
            projected["occurrences"].append(_occurrence(latest))
        projected["consents"].extend(_consent(item, now) for item in reversed(consents[-2:]))
        projected["deliveries"].extend(_delivery(item) for item in reversed(deliveries[-2:]))
        if tasks:
            projected["tasks"].append({
                "id": _identifier(tasks[-1].get("id")), "occurrence_id": _identifier(occurrence_id),
                "status": _choice(tasks[-1].get("status"), {"open", "following_up", "closed"}),
            })
    return {
        "status": "available" if view["configured"] else "not_configured",
        "channel": _channel(view.get("channel")), "external_enabled": view["external_enabled"],
        **projected,
        "counts": {kind: {"total": len(rows[kind]), "included": len(projected[kind]),
                          "omitted": len(rows[kind]) - len(projected[kind])} for kind in rows},
        "scope_notice": "仅含最近3条提醒、每条最新发生记录、最多2条授权和2条当前通知、最新护理任务；未展示的旧记录不能视为未发生。",
    }


def build_care_context(service, elder_id) -> dict:
    """Read one bounded snapshot; failures are unknown, never empty success.

    observed_at describes this read, not a delivery time. A failed service clock
    uses the host UTC time only to timestamp the unavailable observation.
    """
    result = {"source": "minimal_care_records", "status": "unavailable",
              "read_only": True, "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        with service._lock:
            now = service._now()
            observed_at = _timestamp(now.isoformat())
            if observed_at is None:
                return result
            result["observed_at"] = observed_at
            snapshot = service.elder_view(elder_id)
            projected = _project(snapshot, elder_id, now)
        return {**result, **projected}
    except Exception:
        return result
