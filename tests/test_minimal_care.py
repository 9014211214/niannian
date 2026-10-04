"""Local care contracts: no HTTP, actual recipients, or model calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from aria.engine.care import LocalInboxNotificationAdapter, NotificationDelivery
from aria.engine.minimal_care import MinimalCareService

UTC = timezone.utc
START = datetime(2026, 9, 20, 7, 0, tzinfo=UTC)
PASSWORD = "local-test-password-2026"  # release-fixture: not-a-deployment-secret
ELDER = "fictional-chen"


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now


class Gateway:
    channel = "fixture_gateway"

    def __init__(self, outcome="accepted"):
        self.calls = []
        self.outcome = outcome

    def send(self, **payload):
        self.calls.append(payload)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return NotificationDelivery(self.outcome, provider_id="fixture-receipt")


@pytest.fixture
def care(tmp_path):
    clock = Clock()
    service = MinimalCareService(tmp_path / "care.sqlite", clock=clock)
    service.bootstrap("carer", PASSWORD, [ELDER])
    token = service.login("carer", PASSWORD)
    yield service, token, clock
    service.close()


def command(service, action, request_id, **payload):
    return service.elder_action(ELDER, action, {"request_id": request_id, **payload})["record"]


def reminder(service, request_id="rem", **extra):
    return command(service, "reminder.create", request_id, **{
        "label": "准备纸和本子", "schedule": "once", "date": "2026-09-20",
        "time": "15:30", "timezone": "Asia/Shanghai", "grace_minutes": 30, **extra})


def setup_rule(service, *, token=None, source="simulation", **extra):
    rule = reminder(service, **extra)
    contact = command(service, "contact.create", "contact", name="虚构女儿", relation="女儿")
    if source == "real":
        service.mutate(token, ELDER, "contact.verify", {"request_id": "verify", "contact_id": contact["id"],
                                                        "evidence_source": "local test fixture verification"})
    consent = command(service, "consent.grant", "grant", contact_id=contact["id"], reminder_id=rule["id"],
                      trigger="unconfirmed", scope="reminder_status", source=source, confirmed=True,
                      expires_at="2026-12-01T00:00:00+00:00")
    return rule, contact, consent


def advance(service, clock, minutes):
    clock.now = START + timedelta(minutes=minutes)
    return service.tick()


def queue_without_dispatch(service, clock):
    dispatch = service.dispatch_pending
    service.dispatch_pending = lambda **kwargs: {"ok": True, "processed": 0}
    try:
        advance(service, clock, 30)
        advance(service, clock, 60)
    finally:
        service.dispatch_pending = dispatch


def test_default_ignores_notification_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_URL", "https://example.invalid/never")
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_TOKEN", "not-real")
    import aria.engine.care as old_care
    import urllib.request
    assert not hasattr(old_care, "create_notification_adapter")
    assert not hasattr(old_care, "WebhookNotificationAdapter")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("external notification forbidden"))
    service = MinimalCareService(tmp_path / "care.sqlite")
    assert isinstance(service.notifier, LocalInboxNotificationAdapter)
    view = service.elder_view(ELDER)
    assert view["configured"] is False and not view["elders"]
    with pytest.raises(PermissionError):
        reminder(service)
    service.close()


def test_readonly_views_and_reopen_do_not_change_database(care, tmp_path):
    service, token, _clock = care
    path = tmp_path / "care.sqlite"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(3):
        service.view(token)
        service.elder_view(ELDER)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    service.close()
    reopened = MinimalCareService(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    reopened.close()


def test_passwords_tokens_scopes_and_bootstrap_are_explicit(care):
    service, token, _clock = care
    service.bootstrap("second", PASSWORD, ["other-elder"])
    other = service.login("second", PASSWORD)
    assert service.view(token)["elders"] == [{"id": ELDER}]
    assert service.view(other)["elders"] == [{"id": "other-elder"}]
    with pytest.raises(PermissionError):
        service.view(token, "other-elder")
    with pytest.raises(PermissionError):
        service.mutate(other, ELDER, "reminder.create", {})
    with pytest.raises(ValueError, match="已存在"):
        service.bootstrap("carer", "different-password", ["other-elder"])
    raw = " ".join(str(tuple(row)) for row in service._db.execute("SELECT * FROM users"))
    raw += " ".join(str(tuple(row)) for row in service._db.execute("SELECT * FROM sessions"))
    assert PASSWORD not in raw and token not in raw
    service.logout(token)
    with pytest.raises(PermissionError):
        service.view(token)


def test_login_ttl_rate_limit_and_uniform_error(care):
    service, token, clock = care
    messages = []
    for name in ["unknown", *(["carer"] * 5), "carer"]:
        with pytest.raises(PermissionError) as error:
            service.login(name, PASSWORD if len(messages) == 6 else "incorrect")
        messages.append(str(error.value))
    assert len(set(messages)) == 1
    clock.now += timedelta(minutes=16)
    assert service.login("carer", PASSWORD)
    clock.now += timedelta(hours=8)
    with pytest.raises(PermissionError):
        service.view(token)


def test_command_ids_survive_restart_and_reject_reuse(care, tmp_path):
    service, _token, clock = care
    first = reminder(service)
    assert reminder(service)["id"] == first["id"]
    with pytest.raises(ValueError, match="不同操作"):
        reminder(service, label="另一件事")
    service.close()
    restored = MinimalCareService(tmp_path / "care.sqlite", clock=clock)
    assert reminder(restored)["id"] == first["id"]
    assert len(restored.elder_view(ELDER)["reminders"]) == 1
    restored.close()


def test_once_is_not_repeated_and_late_offer_gets_real_window(care):
    service, _token, clock = care
    reminder(service)
    advance(service, clock, 10)
    assert not service.elder_view(ELDER)["occurrences"]
    advance(service, clock, 90)
    occurrence = service.elder_view(ELDER)["occurrences"][0]
    assert occurrence["due_at"] == "2026-09-20T07:30:00+00:00"
    assert occurrence["offered_at"] == "2026-09-20T08:30:00+00:00"
    assert occurrence["deadline_at"] == "2026-09-20T09:00:00+00:00"
    assert occurrence["status"] == "pending"
    command(service, "occurrence.ack", "ack", occurrence_id=occurrence["id"])
    advance(service, clock, 24 * 60 + 90)
    view = service.elder_view(ELDER)
    assert len(view["occurrences"]) == 1
    assert view["reminders"][0]["status"] == "completed"
    assert not view["tasks"] and not view["deliveries"]


@pytest.mark.parametrize("action", ["occurrence.ack", "occurrence.decline"])
def test_response_prevents_unconfirmed_notification_and_terminal_changes(care, action):
    service, _token, clock = care
    setup_rule(service)
    advance(service, clock, 30)
    occurrence = service.elder_view(ELDER)["occurrences"][0]
    command(service, action, "response", occurrence_id=occurrence["id"])
    with pytest.raises(ValueError, match="状态转换"):
        command(service, "occurrence.snooze", "invalid", occurrence_id=occurrence["id"], minutes=10)
    advance(service, clock, 90)
    assert not service.elder_view(ELDER)["deliveries"]


@pytest.mark.parametrize("operation", ["ack", "decline", "cancel", "revoke", "snooze", "expire"])
def test_queued_outbox_rechecks_response_revoke_and_expiry(care, operation):
    service, _token, clock = care
    rule, _contact, consent = setup_rule(service)
    queue_without_dispatch(service, clock)
    view = service.elder_view(ELDER)
    occurrence = view["occurrences"][0]
    assert view["deliveries"][0]["status"] == "queued"
    if operation in {"ack", "decline", "snooze"}:
        command(service, "occurrence." + operation, "response", occurrence_id=occurrence["id"], minutes=10)
    elif operation == "cancel":
        command(service, "reminder.cancel", "cancel", reminder_id=rule["id"])
    elif operation == "revoke":
        command(service, "consent.revoke", "revoke", consent_id=consent["id"])
    else:
        clock.now = datetime(2026, 12, 1, tzinfo=UTC)
    assert service.dispatch_pending()["processed"] == 0
    assert service.elder_view(ELDER)["deliveries"][0]["status"] == "cancelled"


def test_snooze_has_distinct_occurrence_generation_not_network_retry(care):
    service, _token, clock = care
    setup_rule(service)
    queue_without_dispatch(service, clock)
    view = service.elder_view(ELDER)
    occurrence = view["occurrences"][0]
    old_id = view["deliveries"][0]["id"]
    command(service, "occurrence.snooze", "later", occurrence_id=occurrence["id"], minutes=10)
    advance(service, clock, 70)
    assert service.elder_view(ELDER)["occurrences"][0]["status"] == "pending"
    advance(service, clock, 100)
    view = service.elder_view(ELDER)
    assert [row["status"] for row in view["deliveries"]] == ["cancelled", "delivered_local"]
    assert view["deliveries"][1]["id"] != old_id


def test_daily_31_day_outage_offers_latest_only_and_preserves_skips(care):
    service, _token, clock = care
    setup_rule(service, schedule="daily")
    advance(service, clock, 31 * 24 * 60 + 30)
    view = service.elder_view(ELDER)
    assert len(view["occurrences"]) == 1
    assert view["occurrences"][0]["due_at"] == "2026-10-21T07:30:00+00:00"
    catchup = view["reminders"][0]["catchup"]
    assert catchup["skipped_count"] == 31 and catchup["status"] == "not_offered"
    assert catchup["first_skipped_due_at"] == "2026-09-20T07:30:00+00:00"
    advance(service, clock, 31 * 24 * 60 + 60)
    assert len(service.elder_view(ELDER)["deliveries"]) == 1
    advance(service, clock, 32 * 24 * 60 + 30)
    assert len(service.elder_view(ELDER)["occurrences"]) == 2


def test_new_daily_instance_does_not_dispatch_stale_queued_day(care):
    service, _token, clock = care
    setup_rule(service, schedule="daily")
    queue_without_dispatch(service, clock)
    advance(service, clock, 31 * 24 * 60 + 30)
    view = service.elder_view(ELDER)
    assert [row["status"] for row in view["occurrences"]] == ["superseded", "pending"]
    assert view["deliveries"][0]["status"] == "cancelled"
    advance(service, clock, 31 * 24 * 60 + 60)
    view = service.elder_view(ELDER)
    assert [row["status"] for row in view["deliveries"]] == ["cancelled", "delivered_local"]


def test_unconfirmed_is_not_medical_failure_and_manual_care_is_separate(care):
    service, token, clock = care
    setup_rule(service, label="私人内容绝不外发")
    advance(service, clock, 30)
    advance(service, clock, 60)
    view = service.view(token)
    assert view["occurrences"][0]["status"] == "unconfirmed"
    delivery, task = view["deliveries"][0], view["tasks"][0]
    assert delivery["status"] == "delivered_local"
    assert "私人内容" not in delivery["message"]
    assert "尚未收到确认" in delivery["message"]
    with pytest.raises(ValueError):
        service.mutate(token, ELDER, "task.close", {"request_id": "close-first", "task_id": task["id"], "note": "不能直接关闭"})
    service.mutate(token, ELDER, "task.follow_up", {"request_id": "follow", "task_id": task["id"], "note": "模拟护理员开始核查"})
    service.mutate(token, ELDER, "receipt.record", {"request_id": "receipt", "delivery_id": delivery["id"], "status": "family_ack",
                                                    "evidence_source": "模拟家属角色扮演", "note": "仅本地演练"})
    service.mutate(token, ELDER, "task.close", {"request_id": "closed", "task_id": task["id"], "note": "模拟核查完成"})
    view = service.view(token)
    assert view["deliveries"][0]["status"] == "family_ack"
    assert view["deliveries"][0]["receipt_history"][-1]["source"] == "caregiver_report"
    assert view["tasks"][0]["status"] == "closed"
    with pytest.raises(PermissionError):
        command(service, "receipt.record", "elder-hack", delivery_id=delivery["id"], status="delivered")


@pytest.mark.parametrize("extra", [
    {"timezone": "America/New_York"}, {"timezone": "Europe/London"},
    {"timezone": "UTC+8"}, {"time": "24:00"}, {"date": "2026-02-30"},
    {"date": "1991-09-20"}, {"grace_minutes": True}, {"grace_minutes": 0},
    {"date": "2026-09-19"}, {"schedule": "weekly"},
])
def test_time_and_dst_domains_fail_explicitly(care, extra):
    service, _token, _clock = care
    with pytest.raises(ValueError):
        reminder(service, **extra)
    assert not service.elder_view(ELDER)["reminders"]


def test_utc_rollover_daily_and_naive_tick_rejected(care):
    service, _token, clock = care
    reminder(service, schedule="daily", timezone="UTC", time="23:59")
    clock.now = datetime(2026, 9, 20, 23, 59, tzinfo=UTC)
    service.tick()
    assert service.elder_view(ELDER)["reminders"][0]["next_due_at"] == "2026-09-21T23:59:00+00:00"
    with pytest.raises(ValueError, match="时区"):
        service.tick(datetime(2026, 9, 21))


@pytest.mark.parametrize("scope,trigger", [("chat", "unconfirmed"), ("reminder_status", "sadness")])
def test_consent_cannot_expand_scope_or_trigger(care, scope, trigger):
    service, _token, _clock = care
    rule = reminder(service)
    contact = command(service, "contact.create", "contact", name="虚构联系人")
    with pytest.raises(ValueError):
        command(service, "consent.grant", "bad", contact_id=contact["id"], reminder_id=rule["id"],
                scope=scope, trigger=trigger, source="simulation", confirmed=True, expires_at="2026-10-01T00:00:00Z")
    assert not service.elder_view(ELDER)["consents"]


def make_external(tmp_path, outcome="accepted"):
    gateway = Gateway(outcome)
    clock = Clock()
    service = MinimalCareService(tmp_path / "external.sqlite", notifier=gateway, allow_external=True, clock=clock)
    service.bootstrap("carer", PASSWORD, [ELDER])
    token = service.login("carer", PASSWORD)
    return service, token, clock, gateway


def test_external_requires_explicit_setting_and_never_accepts_simulation(tmp_path):
    with pytest.raises(ValueError, match="显式"):
        MinimalCareService(tmp_path / "bad.sqlite", notifier=Gateway())
    service, _token, clock, gateway = make_external(tmp_path)
    setup_rule(service)
    advance(service, clock, 30)
    advance(service, clock, 60)
    assert not gateway.calls
    assert service.elder_view(ELDER)["tasks"] and not service.elder_view(ELDER)["deliveries"]
    service.close()


def test_gateway_acceptance_and_family_ack_never_mean_human_handled(tmp_path):
    service, token, clock, gateway = make_external(tmp_path)
    setup_rule(service, token=token, source="real")
    advance(service, clock, 30)
    advance(service, clock, 60)
    service.tick()
    view = service.view(token)
    delivery = view["deliveries"][0]
    assert delivery["status"] == "accepted" and len(gateway.calls) == 1
    assert gateway.calls[0]["idempotency_key"] == delivery["event_id"]
    service.mutate(token, ELDER, "receipt.record", {"request_id": "r", "delivery_id": delivery["id"], "status": "delivered",
                                                    "evidence_source": "fixture provider audit", "note": "local test"})
    assert service.view(token)["tasks"][0]["status"] == "open"
    service.close()


@pytest.mark.parametrize("outcome", [TimeoutError("not public"), "failed", "nonsense", "delivered_local"])
def test_unknown_or_legacy_failure_does_not_retry(tmp_path, outcome):
    service, token, clock, gateway = make_external(tmp_path, outcome)
    setup_rule(service, token=token, source="real")
    advance(service, clock, 30)
    advance(service, clock, 60)
    service.tick()
    view = service.view(token)
    assert len(gateway.calls) == 1
    assert view["deliveries"][0]["status"] == "unknown"
    assert "not public" not in json.dumps(view)
    assert "未知" in view["tasks"][0]["delivery_attention"]
    service.close()
    restored = MinimalCareService(tmp_path / "external.sqlite", notifier=gateway, allow_external=True, clock=clock)
    restored.tick()
    assert len(gateway.calls) == 1
    restored.close()


def test_crash_between_sending_and_receipt_recovers_unknown(tmp_path):
    class Crash(BaseException):
        pass
    service, token, clock, gateway = make_external(tmp_path, Crash())
    setup_rule(service, token=token, source="real")
    advance(service, clock, 30)
    with pytest.raises(Crash):
        advance(service, clock, 60)
    assert service.view(token)["deliveries"][0]["status"] == "sending"
    service.close()
    restored = MinimalCareService(tmp_path / "external.sqlite", notifier=gateway, allow_external=True, clock=clock)
    assert restored.view(token)["deliveries"][0]["status"] == "unknown"
    assert "未知" in restored.view(token)["tasks"][0]["delivery_attention"]
    restored.tick()
    assert len(gateway.calls) == 1
    restored.close()


def test_receipt_commit_failure_keeps_durable_sending_and_never_resends(tmp_path, monkeypatch):
    service, token, clock, gateway = make_external(tmp_path)
    setup_rule(service, token=token, source="real")
    advance(service, clock, 30)
    original = service._put
    def fail(kind, row):
        if kind == "deliveries" and row["status"] == "accepted":
            raise sqlite3.OperationalError("local write failure fixture")
        return original(kind, row)
    monkeypatch.setattr(service, "_put", fail)
    with pytest.raises(sqlite3.OperationalError):
        advance(service, clock, 60)
    assert service.view(token)["deliveries"][0]["status"] == "sending"
    service.tick()
    assert len(gateway.calls) == 1
    service.close()


def test_concurrent_ticks_do_not_duplicate_outbox_send(tmp_path):
    service, token, clock, gateway = make_external(tmp_path)
    setup_rule(service, token=token, source="real")
    advance(service, clock, 30)
    clock.now = START + timedelta(minutes=60)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: service.tick(), range(4)))
    assert len(gateway.calls) == 1
    assert len(service.view(token)["deliveries"]) == 1
    service.close()


def test_two_service_connections_cannot_claim_same_delivery(tmp_path):
    service, token, clock, gateway = make_external(tmp_path)
    setup_rule(service, token=token, source="real")
    other = MinimalCareService(tmp_path / "external.sqlite", notifier=gateway, allow_external=True, clock=clock)
    queue_without_dispatch(service, clock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda instance: instance.dispatch_pending(), [service, other]))
    assert len(gateway.calls) == 1
    assert service.view(token)["deliveries"][0]["status"] == "accepted"
    other.close()
    service.close()


def test_each_queued_delivery_uses_fresh_expiry_check(tmp_path):
    service, token, clock, gateway = make_external(tmp_path)
    rule, _contact, _consent = setup_rule(service, token=token, source="real")
    second = command(service, "contact.create", "second", name="另一虚构联系人")
    service.mutate(token, ELDER, "contact.verify", {"request_id": "verify-second", "contact_id": second["id"],
                                                    "evidence_source": "local test fixture"})
    command(service, "consent.grant", "grant-second", contact_id=second["id"], reminder_id=rule["id"],
            trigger="unconfirmed", scope="reminder_status", source="real", confirmed=True,
            expires_at="2026-09-20T08:00:01+00:00")
    queue_without_dispatch(service, clock)
    original = gateway.send
    def slow_send(**kwargs):
        receipt = original(**kwargs)
        clock.now += timedelta(seconds=2)
        return receipt
    gateway.send = slow_send
    service.dispatch_pending()
    view = service.view(token)
    assert len(gateway.calls) == 1
    assert [row["status"] for row in view["deliveries"]] == ["accepted", "cancelled"]
    service.close()


def test_scope_validation_is_atomic_on_foreign_record(care):
    service, token, _clock = care
    rule = reminder(service)
    service.bootstrap("second", PASSWORD, ["other-elder"])
    second = service.login("second", PASSWORD)
    with pytest.raises(ValueError, match="负责范围"):
        service.mutate(second, "other-elder", "reminder.cancel", {"request_id": "x", "reminder_id": rule["id"]})
    assert service.view(token)["reminders"][0]["status"] == "active"
    assert not service.view(second)["reminders"]


def test_local_forget_preserves_login_and_assignments(care):
    service, token, clock = care
    setup_rule(service)
    advance(service, clock, 30)
    advance(service, clock, 60)
    service.forget_elder(ELDER)
    view = service.view(token)
    assert view["elders"] == [{"id": ELDER}]
    assert all(not view[key] for key in ("reminders", "occurrences", "consents", "contacts", "tasks", "deliveries"))
