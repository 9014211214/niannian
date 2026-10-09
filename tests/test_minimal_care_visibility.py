"""Reminder visibility contracts with fictional local software receipts only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json

import pytest

from aria.engine.minimal_care import MinimalCareService
from aria.engine.minimal_care_context import build_care_context
from tests.test_minimal_care import Clock, ELDER, PASSWORD, START, command, reminder, setup_rule


@pytest.fixture
def unseen_care(tmp_path):
    clock = Clock()
    path = tmp_path / "care.sqlite"
    service = MinimalCareService(path, clock=clock)
    service.bootstrap("carer", PASSWORD, [ELDER])
    token = service.login("carer", PASSWORD)
    yield service, token, clock, path
    service.close()


def due(service, clock):
    clock.now = START + timedelta(minutes=30)
    service.tick()
    return service.elder_view(ELDER)["occurrences"][0]


def displayed(service, occurrence, request_id="display", **extra):
    return command(service, "occurrence.displayed", request_id, occurrence_id=occurrence["id"],
                   generation=occurrence["generation"], **extra)


def test_generated_reminder_never_starts_unconfirmed_grace_when_unseen(unseen_care):
    service, _token, clock, _path = unseen_care
    setup_rule(service)
    row = due(service, clock)
    assert row["delivery_status"] == "awaiting_display"
    assert row["generated_at"] == "2026-09-20T07:30:00+00:00"
    assert row["offered_at"] is None and row["deadline_at"] is None and row["displayed_at"] is None
    clock.now += timedelta(days=60)
    for _ in range(3):
        service.tick()
    view = service.elder_view(ELDER)
    assert view["occurrences"][0]["status"] == "pending"
    assert not view["tasks"] and not view["deliveries"]


def test_polling_both_views_is_readonly_and_does_not_acknowledge_display(unseen_care):
    service, token, clock, path = unseen_care
    reminder(service)
    due(service, clock)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(5):
        service.elder_view(ELDER)
        service.view(token)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not service.elder_view(ELDER)["occurrences"][0]["display_history"]


def test_prompt_care_facts_distinguish_generated_and_page_displayed(unseen_care):
    service, _token, clock, _path = unseen_care
    reminder(service)
    row = due(service, clock)
    before = build_care_context(service, ELDER)["occurrences"][0]
    assert before["delivery_status"] == "awaiting_display" and not before["page_displayed"]
    assert before["displayed_at"] is None and before["deadline_at"] is None
    displayed(service, row)
    after = build_care_context(service, ELDER)["occurrences"][0]
    assert after["delivery_status"] == "displayed_local" and after["page_displayed"]
    assert after["displayed_generation"] == after["generation"] == 0
    wire = json.dumps(after)
    assert "displayed_by" not in wire and "display_history" not in wire


def test_late_display_receipt_starts_server_window_then_only_missing_response(unseen_care):
    service, _token, clock, _path = unseen_care
    setup_rule(service)
    row = due(service, clock)
    clock.now = START + timedelta(hours=2)
    receipt = displayed(service, row, displayed_at="1992-01-01T00:00:00Z",
                        deadline_at="1992-01-01T00:01:00Z", actor="caregiver:carer")
    assert receipt["displayed_at"] == "2026-09-20T09:00:00+00:00"
    assert receipt["deadline_at"] == "2026-09-20T09:30:00+00:00"
    assert receipt["displayed_by"] == "elder:" + ELDER
    assert receipt["display_source"] == "elder_page_receipt_v1"
    assert receipt["status"] == "pending" and "responded_at" not in receipt
    clock.now += timedelta(minutes=29, seconds=59)
    service.tick()
    assert not service.elder_view(ELDER)["tasks"]
    clock.now += timedelta(seconds=1)
    service.tick()
    view = service.elder_view(ELDER)
    assert view["occurrences"][0]["status"] == "unconfirmed"
    assert view["tasks"][0]["visibility_evidence"] == "elder_page_receipt_v1"
    assert len(view["deliveries"]) == 1 and view["deliveries"][0]["status"] == "delivered_local"


def test_repeated_display_commands_do_not_extend_grace_or_duplicate_event(unseen_care):
    service, _token, clock, _path = unseen_care
    reminder(service)
    row = due(service, clock)
    first = displayed(service, row)
    clock.now += timedelta(minutes=20)
    assert displayed(service, row) == first
    assert displayed(service, row, "display-second") == first
    assert len(service.elder_view(ELDER)["occurrences"][0]["display_history"]) == 1


def test_snooze_requires_new_visible_generation_and_stale_receipt_cannot_rearm(unseen_care):
    service, _token, clock, _path = unseen_care
    setup_rule(service)
    row = due(service, clock)
    displayed(service, row)
    snoozed = command(service, "occurrence.snooze", "snooze", occurrence_id=row["id"], minutes=10)
    assert snoozed["generation"] == 1 and snoozed["deadline_at"] is None
    with pytest.raises(ValueError, match="代次"):
        displayed(service, row, "stale")
    with pytest.raises(ValueError, match="状态转换"):
        displayed(service, snoozed, "too-early")
    clock.now += timedelta(minutes=40)
    service.tick()
    ready = service.elder_view(ELDER)["occurrences"][0]
    assert ready["status"] == "pending" and ready["delivery_status"] == "awaiting_display"
    assert ready["deadline_at"] is None and not service.elder_view(ELDER)["tasks"]
    new_receipt = displayed(service, ready, "new-generation")
    assert len(new_receipt["display_history"]) == 2
    assert new_receipt["display_history"][0]["generation"] == 0
    assert new_receipt["display_history"][1]["generation"] == 1
    clock.now += timedelta(minutes=30)
    service.tick()
    assert len(service.elder_view(ELDER)["deliveries"]) == 1


@pytest.mark.parametrize("generation", [True, False, "0", None, -1, 1])
def test_display_generation_is_exact_and_strict(unseen_care, generation):
    service, _token, clock, _path = unseen_care
    reminder(service)
    row = due(service, clock)
    before = service.elder_view(ELDER)
    with pytest.raises(ValueError, match="代次"):
        command(service, "occurrence.displayed", "invalid-display", occurrence_id=row["id"], generation=generation)
    assert service.elder_view(ELDER) == before


def test_caregiver_cannot_forge_display_or_replay_legacy_cached_result(unseen_care):
    service, token, clock, _path = unseen_care
    reminder(service)
    row = due(service, clock)
    payload = {"request_id": "forged", "occurrence_id": row["id"], "generation": 0,
               "actor": "elder:" + ELDER, "role": "elder", "caregiver": False}
    action = "occurrence.displayed"
    wire = json.dumps([action, payload], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    service._db.execute("INSERT INTO commands(actor,elder_id,request_id,digest,result) VALUES(?,?,?,?,?)",
        ("caregiver:carer", ELDER, "forged", hashlib.sha256(wire.encode()).hexdigest(),
         json.dumps({"ok": True, "record": {"delivery_status": "historical-only"}})))
    before = service.elder_view(ELDER)
    with pytest.raises(PermissionError, match="只能由本人"):
        service.mutate(token, ELDER, action, payload)
    assert service.elder_view(ELDER) == before


def test_terminal_unseen_reminder_cannot_gain_display_receipt(unseen_care):
    service, _token, clock, _path = unseen_care
    reminder(service)
    row = due(service, clock)
    command(service, "occurrence.ack", "ack", occurrence_id=row["id"])
    with pytest.raises(ValueError, match="状态转换"):
        displayed(service, row)
    current = service.elder_view(ELDER)["occurrences"][0]
    assert current["status"] == "acknowledged" and current["displayed_at"] is None


def test_upgrade_retains_old_offer_evidence_and_stops_undisplayed_outbox(unseen_care):
    service, _token, clock, path = unseen_care
    setup_rule(service)
    row = due(service, clock)
    displayed(service, row)
    original_dispatch = service.dispatch_pending
    service.dispatch_pending = lambda **kw: {"ok": True, "processed": 0}
    clock.now += timedelta(minutes=30)
    service.tick()
    service.dispatch_pending = original_dispatch
    old = service.elder_view(ELDER)["occurrences"][0]
    legacy = {k: v for k, v in old.items() if not k.startswith("display") and k != "generated_at"}
    legacy["delivery_status"] = "offered_local"
    with service._tx():
        service._put("occurrences", legacy)
    original_offer = legacy["offered_at"]
    original_deadline = legacy["deadline_at"]
    service.close()
    restored = MinimalCareService(path, clock=clock)
    view = restored.elder_view(ELDER)
    upgraded = view["occurrences"][0]
    assert upgraded["status"] == "pending" and upgraded["deadline_at"] is None
    assert upgraded["legacy_visibility"] == {"status": "unconfirmed", "delivery_status": "offered_local",
        "offered_at": original_offer, "deadline_at": original_deadline, "generation": 0}
    assert upgraded["generation"] == 1
    assert view["deliveries"][0]["status"] == "cancelled"
    assert view["tasks"][0]["visibility_evidence"] == "legacy_unverified"
    clock.now += timedelta(days=5)
    restored.tick()
    assert restored.elder_view(ELDER)["occurrences"][0]["status"] == "pending"
    assert restored.dispatch_pending()["processed"] == 0
    displayed(restored, upgraded, "fresh-after-upgrade")
    clock.now += timedelta(minutes=30)
    restored.tick()
    new_view = restored.elder_view(ELDER)
    assert [row["status"] for row in new_view["deliveries"]] == ["cancelled", "delivered_local"]
    assert new_view["tasks"][1]["visibility_evidence"] == "elder_page_receipt_v1"
    assert new_view["tasks"][1]["generation"] == 1
    restored.close()


def test_two_connections_display_only_once_and_preserve_first_clock(unseen_care):
    service, _token, clock, path = unseen_care
    reminder(service)
    row = due(service, clock)
    other = MinimalCareService(path, clock=clock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda pair: displayed(pair[0], row, pair[1]),
                               [(service, "one"), (other, "two")]))
    assert values[0] == values[1]
    assert len(service.elder_view(ELDER)["occurrences"][0]["display_history"]) == 1
    other.close()
