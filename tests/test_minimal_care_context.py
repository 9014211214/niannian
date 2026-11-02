"""Actual local care records projected without mutations or private fields."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json

import pytest

from aria.engine.minimal_care_context import build_care_context, compact_empty_care_context
from tests.test_minimal_care import (
    ELDER, START, advance_visible_page, care, command, queue_without_dispatch, reminder, setup_rule,
)


def item(service):
    return build_care_context(service, ELDER)


def test_unconfigured_is_distinct_from_configured_without_reminders(care):
    service, _token, _clock = care
    result = build_care_context(service, ELDER)
    assert result["status"] == "available" and result["counts"]["reminders"]["total"] == 0
    assert build_care_context(service, "not-assigned")["status"] == "not_configured"
    for elder_id, status in ((ELDER, "available"), ("not-assigned", "not_configured")):
        raw = build_care_context(service, elder_id)
        before = deepcopy(raw)
        view = compact_empty_care_context(raw)
        assert raw == before and build_care_context(service, elder_id) == before
        assert view["status"] == status
        assert set(raw) - set(view) == {*raw["counts"], "scope_notice"}
        assert {key: value for key, value in view.items() if key != "counts"} == {
            key: value for key, value in raw.items() if key in view and key != "counts"}
        assert {kind: view["counts"]["each"] for kind in view["counts"]["kinds"]} == raw["counts"]
        assert len(json.dumps(view, ensure_ascii=False).encode("utf-8")) < len(
            json.dumps(raw, ensure_ascii=False).encode("utf-8"))
        view["counts"]["each"]["total"] = 1
        assert raw == before


def test_read_projects_actual_schedule_and_not_a_completed_action(care):
    service, _token, _clock = care
    rule = reminder(service)
    result = build_care_context(service, ELDER)
    assert result["source"] == "minimal_care_records" and result["read_only"] is True
    assert result["observed_at"] == START.isoformat()
    row = result["reminders"][0]
    assert row["id"] == rule["id"] and row["label"] == "准备纸和本子"
    assert row["schedule"] == "once" and row["timezone"] == "Asia/Shanghai"
    assert row["next_due_at"] == "2026-09-20T07:30:00+00:00"
    assert not result["occurrences"] and result["counts"]["deliveries"]["total"] == 0
    assert compact_empty_care_context(result) == result


def test_snapshot_does_not_tick_deliver_or_change_database(care, tmp_path, monkeypatch):
    service, _token, clock = care
    setup_rule(service)
    queue_without_dispatch(service, clock)
    before = hashlib.sha256((tmp_path / "care.sqlite").read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail("read projection must not execute care")
    monkeypatch.setattr(service, "tick", forbidden)
    monkeypatch.setattr(service, "dispatch_pending", forbidden)
    monkeypatch.setattr(service.notifier, "send", forbidden)
    for _ in range(3):
        row = item(service)
        assert row["occurrences"][0]["status"] == "unconfirmed"
        assert row["occurrences"][0]["timezone"] == "Asia/Shanghai"
        assert row["deliveries"][0]["status"] == "queued"
    assert hashlib.sha256((tmp_path / "care.sqlite").read_bytes()).hexdigest() == before


def test_private_contact_auth_message_and_followup_never_project(care):
    service, token, clock = care
    setup_rule(service)
    advance_visible_page(service, clock, 30)
    advance_visible_page(service, clock, 60)
    view = service.elder_view(ELDER)
    task = view["tasks"][0]
    delivery = view["deliveries"][0]
    service.mutate(token, ELDER, "task.follow_up", {"request_id": "follow", "task_id": task["id"], "note": "PRIVATE_FOLLOWUP"})
    service.mutate(token, ELDER, "receipt.record", {"request_id": "receipt", "delivery_id": delivery["id"], "status": "family_ack",
                                                    "evidence_source": "PRIVATE_EVIDENCE", "note": "PRIVATE_RECEIPT"})
    contact = view["contacts"][0]
    with service._lock, service._tx():
        contact.update(name="PRIVATE_CONTACT", relation="PRIVATE_RELATION", channel="PRIVATE_ADDRESS")
        service._put("contacts", contact)
    result = build_care_context(service, ELDER)
    wire = json.dumps(result, ensure_ascii=False)
    assert "PRIVATE" not in wire and token not in wire and "carer" not in wire
    for key in ("message", "provider_id", "note", "history", "created_by", "responded_by", "recorded_by", "contacts", "evidence_source"):
        assert f'"{key}"' not in wire
    sent = result["deliveries"][0]
    assert sent["status"] == "family_ack" and sent["source"] == "simulation"
    assert sent["channel"] == "local_care_inbox" and sent["receipt_source"] == "caregiver_report"


def test_expired_future_and_revoked_consent_remain_distinct(care):
    service, _token, clock = care
    _rule, _contact, consent = setup_rule(service)
    assert item(service)["consents"][0]["valid_now"] is True
    clock.now = START - timedelta(days=1)
    assert item(service)["consents"][0]["valid_now"] is False
    clock.now = START + timedelta(days=100)
    expired = item(service)["consents"][0]
    assert expired["status"] == "active" and expired["valid_now"] is False
    clock.now = START
    command(service, "consent.revoke", "revoke", consent_id=consent["id"])
    revoked = item(service)["consents"][0]
    assert revoked["status"] == "revoked" and revoked["valid_now"] is False


def test_consent_validity_preserves_subsecond_boundaries(care):
    service, _token, clock = care
    _rule, _contact, consent = setup_rule(service)
    with service._lock, service._tx():
        consent["valid_from"] = (START + timedelta(microseconds=500000)).isoformat()
        consent["expires_at"] = (START + timedelta(seconds=1, microseconds=500000)).isoformat()
        service._put("consents", consent)
    assert not item(service)["consents"][0]["valid_now"]
    clock.now = START + timedelta(seconds=1)
    assert item(service)["consents"][0]["valid_now"]
    # The care scheduler's existing _now() operates at whole-second precision.
    # Projection must use that same clock and not round stored expiry down.
    clock.now = START + timedelta(seconds=2)
    assert not item(service)["consents"][0]["valid_now"]


def test_bounds_and_daily_history_omissions_are_explicit(care):
    service, _token, clock = care
    for index in range(4):
        reminder(service, request_id=f"rem-{index}", label=f"earlier-{index}")
    rule, _contact, _consent = setup_rule(service, schedule="daily")
    advance_visible_page(service, clock, 30)
    advance_visible_page(service, clock, 60)
    advance_visible_page(service, clock, 24 * 60 + 30)
    result = build_care_context(service, ELDER)
    assert result["counts"]["reminders"] == {"total": 5, "included": 3, "omitted": 2}
    latest = result["reminders"][0]
    assert latest["id"] == rule["id"]
    assert result["counts"]["occurrences"] == {"total": 6, "included": 3, "omitted": 3}
    assert result["occurrences"][0]["status"] == "pending"
    assert result["counts"]["deliveries"] == {"total": 1, "included": 0, "omitted": 1}


def test_related_consent_and_delivery_limits_keep_omission_counts(care):
    service, _token, clock = care
    rule = reminder(service)
    for index in range(3):
        contact = command(service, "contact.create", f"contact-{index}", name=f"SECRET-{index}")
        command(service, "consent.grant", f"grant-{index}", contact_id=contact["id"], reminder_id=rule["id"],
                trigger="unconfirmed", scope="reminder_status", source="simulation", confirmed=True,
                expires_at="2026-12-01T00:00:00+00:00")
    advance_visible_page(service, clock, 30)
    advance_visible_page(service, clock, 60)
    row = item(service)
    for kind in ("consents", "deliveries"):
        assert row["counts"][kind] == {"total": 3, "included": 2, "omitted": 1}
        assert len(row[kind]) == 2
    assert "SECRET" not in json.dumps(row)


def test_cross_elder_records_cannot_enter_projection_even_with_broad_view(care, monkeypatch):
    service, _token, clock = care
    own = reminder(service)
    advance_visible_page(service, clock, 30)
    view = service.elder_view(ELDER)
    view["reminders"].append({**own, "elder_id": "other", "label": "OTHER_ELDER_SECRET"})
    view["occurrences"].append({**view["occurrences"][0], "elder_id": "other", "status": "acknowledged"})
    view["deliveries"].append({"elder_id": "other", "occurrence_id": view["occurrences"][0]["id"],
                               "status": "family_ack", "message": "OTHER_ELDER_SECRET"})
    monkeypatch.setattr(service, "elder_view", lambda _elder: view)
    result = build_care_context(service, ELDER)
    assert result["counts"]["reminders"]["total"] == 1
    assert result["counts"]["occurrences"]["total"] == 1
    assert result["occurrences"][0]["status"] == "pending" and not result["deliveries"]
    assert "OTHER_ELDER_SECRET" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["view", "clock", "malformed"])
def test_read_failure_is_unavailable_without_raw_errors_or_zero_counts(care, monkeypatch, failure):
    service, _token, _clock = care
    def fail(*args):
        raise RuntimeError("PRIVATE_DATABASE_OR_CLOCK_ERROR")
    if failure == "view":
        monkeypatch.setattr(service, "elder_view", fail)
    elif failure == "clock":
        monkeypatch.setattr(service, "_now", fail)
    else:
        monkeypatch.setattr(service, "elder_view", lambda _elder: {"configured": True})
    result = build_care_context(service, ELDER)
    assert result["status"] == "unavailable" and result["read_only"] is True
    assert "reminders" not in result and "PRIVATE" not in json.dumps(result)
    assert result["observed_at"].endswith("+00:00")


def test_unknown_delivery_cannot_become_success(care):
    service, _token, clock = care
    setup_rule(service)
    queue_without_dispatch(service, clock)
    row = service.elder_view(ELDER)["deliveries"][0]
    with service._lock, service._tx():
        row.update(status="unknown", error="PRIVATE_TRANSPORT_ERROR", provider_id="PRIVATE_PROVIDER")
        service._put("deliveries", row)
    result = item(service)["deliveries"][0]
    assert result["status"] == "unknown" and "PRIVATE" not in json.dumps(result)


def test_clock_and_snapshot_are_read_under_same_service_lock(care, monkeypatch):
    service, _token, clock = care
    original = service.elder_view
    calls = []
    def checked_view(elder_id):
        assert service._lock._is_owned()
        calls.append(elder_id)
        return original(elder_id)
    def checked_clock():
        assert service._lock._is_owned()
        return clock.now
    monkeypatch.setattr(service, "elder_view", checked_view)
    monkeypatch.setattr(service, "clock", checked_clock)
    assert build_care_context(service, ELDER)["status"] == "available"
    assert calls == [ELDER]


@pytest.mark.parametrize("case", ["source", "status", "read_only", "nonempty",
    "count_bool", "count_positive", "count_missing", "extra_count", "notice", "missing_list",
    "status_type", "counts_type", "list_type"])
def test_empty_prompt_view_never_hides_unknown_or_omitted_records(care, case):
    service, _token, _clock = care
    raw = build_care_context(service, ELDER)
    if case == "source":
        raw["source"] = "unknown"
    elif case == "status":
        raw["status"] = "unavailable"
    elif case == "read_only":
        raw["read_only"] = False
    elif case == "nonempty":
        raw["deliveries"] = [{"status": "unknown"}]
    elif case == "count_bool":
        raw["counts"]["consents"]["total"] = False
    elif case == "count_positive":
        raw["counts"]["consents"]["total"] = 1
        raw["counts"]["consents"]["omitted"] = 1
    elif case == "count_missing":
        del raw["counts"]["tasks"]["omitted"]
    elif case == "extra_count":
        raw["counts"]["other"] = {"total": 0, "included": 0, "omitted": 0}
    elif case == "notice":
        raw["scope_notice"] = "未知范围。"
    elif case == "missing_list":
        del raw["occurrences"]
    elif case == "status_type":
        raw["status"] = []
    elif case == "counts_type":
        raw["counts"] = None
    elif case == "list_type":
        raw["tasks"] = None
    before = deepcopy(raw)
    assert compact_empty_care_context(raw) == before
    assert raw == before
