"""Notification contract and local simulated delivery for the first release."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass(frozen=True)
class NotificationDelivery:
    status: str
    provider_id: Optional[str] = None
    error: Optional[str] = None


class NotificationAdapter(Protocol):
    channel: str

    def send(
        self,
        *,
        recipient: str,
        message: str,
        severity: str,
        idempotency_key: str,
        subject_id: str | None = None,
        recipient_id: str | None = None,
        recipient_channel: str | None = None,
    ) -> NotificationDelivery: ...


class LocalInboxNotificationAdapter:
    """默认只投递到本地参考工作台，不声称已经联系真实护理员。"""

    channel = "local_care_inbox"

    def send(
        self,
        *,
        recipient: str,
        message: str,
        severity: str,
        idempotency_key: str,
        subject_id: str | None = None,
        recipient_id: str | None = None,
        recipient_channel: str | None = None,
    ) -> NotificationDelivery:
        del (
            recipient, message, severity, subject_id,
            recipient_id, recipient_channel,
        )
        return NotificationDelivery(
            status="delivered_local",
            provider_id=f"local:{idempotency_key}",
        )


