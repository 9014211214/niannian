"""Identity, permissions, persistence and local receipts; no semantic or model score."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from aria.engine.explicit_sharing import (
    ExplicitSharingService, LocalSharingAdapter, LocalSharingReceipt, SharingContext,
)


OWNER = "fictional-chen"
START = datetime(2026, 10, 9, 0, tzinfo=timezone.utc)


class Adapter(LocalSharingAdapter):
    def __init__(self, outcome="delivered_local"):
        self.calls = []
        self.outcome = outcome

    def send(self, **payload):
        self.calls.append(payload)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return LocalSharingReceipt(self.outcome, payload["sharing_id"],
                                   payload["preview_sha256"], "local_" + "a" * 24)


@pytest.fixture
def fixture(tmp_path):
    clock = [START]
    adapter = Adapter()
    path = tmp_path / "sharing.sqlite3"
    service = ExplicitSharingService(path, owner_id=OWNER, adapter=adapter, clock=lambda: clock[0])
    context = SharingContext(OWNER, "run_1", "active_1", True,
        ({"turn_id": "turn_1", "session_id": "active_1", "turn_index": 0,
          "text": "周六我愿意和你一起折纸风车。", "status": "completed"},),
        ({"id": "contact_1", "elder_id": OWNER, "name": "虚构女儿", "relation": "女儿", "channel": "local"},))
    yield service, context, adapter, clock, path
    service.close()


def action(service, context, name, request_id, **payload):
    return service.action(context, name, {"request_id": request_id, **payload}, actor=OWNER)["record"]


def prepare(service, context, request_id="prepare"):
    return action(service, context, "prepare", request_id, source_id="turn_1", contact_id="contact_1")


def approved(service, context):
    row = prepare(service, context)
    return action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)


def send(service, context, row, request_id="send"):
    return action(service, context, "send", request_id, id=row["id"], preview_sha256=row["preview_sha256"])


def test_default_has_no_dispatch_or_model_and_readonly_bytes(fixture, monkeypatch):
    service, context, adapter, _, path = fixture
    monkeypatch.setenv("ARIA_SHARING_URL", "https://example.invalid/unused")
    before = path.read_bytes()
    for _ in range(3):
        assert not service.view(context)["records"]
    assert path.read_bytes() == before and not adapter.calls
    assert service.view(context)["external_enabled"] is False


def test_exact_preview_approval_distinct_send_and_local_receipt(fixture):
    service, context, adapter, *_ = fixture
    row = prepare(service, context)
    assert row["message"] == context.sources[0]["text"] and row["status"] == "draft"
    assert not adapter.calls
    with pytest.raises(ValueError):
        send(service, context, row)
    row = action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    assert not adapter.calls and row["status"] == "approved"
    result = send(service, context, row)
    assert result["status"] == "delivered_local" and result["attempts"] == 1
    assert result["receipt"]["source"] == "local_simulation"
    assert adapter.calls[0]["message"] == context.sources[0]["text"]
    assert adapter.calls[0]["recipient"]["id"] == "contact_1"


@pytest.mark.parametrize("name", ["prepare", "approve", "send", "revoke"])
@pytest.mark.parametrize("role", ["caregiver:test", "other_owner", None])
def test_authority_checked_before_any_saved_command_replay(fixture, name, role):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    if name == "send":
        send(service, context, row)
    payloads = {"prepare": {"request_id": "prepare", "source_id": "turn_1", "contact_id": "contact_1"},
                "approve": {"request_id": "approve", "id": row["id"], "preview_sha256": row["preview_sha256"], "confirmed": True},
                "send": {"request_id": "send", "id": row["id"], "preview_sha256": row["preview_sha256"]},
                "revoke": {"request_id": "revoke", "id": row["id"]}}
    previous = len(adapter.calls)
    with pytest.raises(PermissionError):
        service.action(context, name, payloads[name], actor=role)
    assert len(adapter.calls) == previous


@pytest.mark.parametrize("read_only", [True, "false", 1, None])
def test_replay_flag_must_be_explicit_false(fixture, read_only):
    service, context, adapter, *_ = fixture
    with pytest.raises(PermissionError):
        service.action(context, "prepare", {"request_id": "p", "source_id": "turn_1", "contact_id": "contact_1"}, actor=OWNER, read_only=read_only)
    assert not adapter.calls


@pytest.mark.parametrize("change", ["owner", "session", "run", "source", "index", "contact", "recipient", "channel", "expiry", "hash"])
def test_changed_source_target_scope_or_expiry_invalidates_approval(fixture, change):
    service, context, adapter, clock, *_ = fixture
    row = approved(service, context)
    digest = row["preview_sha256"]
    if change == "owner": context = replace(context, owner_id="someone_else")
    if change == "session": context = replace(context, session_id="other_session")
    if change == "run": context = replace(context, run_id="other_run")
    if change == "source": context = replace(context, sources=({**context.sources[0], "text": "只换一个字。"},))
    if change == "index": context = replace(context, sources=({**context.sources[0], "turn_index": 1},))
    if change == "contact": context = replace(context, contacts=({**context.contacts[0], "name": "新称呼"},))
    if change == "recipient": context = replace(context, contacts=({**context.contacts[0], "elder_id": "someone_else"},))
    if change == "channel": context = replace(context, contacts=({**context.contacts[0], "channel": "external"},))
    if change == "expiry": clock[0] += timedelta(minutes=5)
    if change == "hash": digest = "0" * 64
    with pytest.raises((ValueError, PermissionError)):
        action(service, context, "send", "changed", id=row["id"], preview_sha256=digest)
    assert not adapter.calls


@pytest.mark.parametrize("extra", [{"message": "改写原话"}, {"scope": "all_chat"}, {"confirmed": True}, {"elder_id": "other"}])
def test_prepare_rejects_arbitrary_text_or_permissions(fixture, extra):
    service, context, adapter, *_ = fixture
    with pytest.raises(ValueError):
        action(service, context, "prepare", "p", source_id="turn_1", contact_id="contact_1", **extra)
    assert not adapter.calls


@pytest.mark.parametrize("source", [(), ({"turn_id": "turn_1", "status": "pending"},), ({"turn_id": "turn_1", "status": "completed", "text": ""},)])
def test_only_whole_completed_saved_owner_sources_are_eligible(fixture, source):
    service, context, adapter, *_ = fixture
    with pytest.raises(ValueError):
        prepare(service, replace(context, sources=source))
    assert not adapter.calls


def test_old_saved_session_is_not_a_source_for_current_owner_grant(fixture):
    service, context, adapter, *_ = fixture
    old = replace(context, sources=({**context.sources[0], "session_id": "old_session"},))
    with pytest.raises(ValueError): prepare(service, old)
    assert not service.view(context)["records"] and not adapter.calls


def test_duplicate_send_returns_saved_receipt_without_second_attempt(fixture):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    result = send(service, context, row)
    assert send(service, context, row) == result
    with pytest.raises(ValueError):
        send(service, context, row, "send_changed_id")
    assert len(adapter.calls) == 1


def test_concurrent_senders_make_exactly_one_adapter_call(fixture):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    def attempt(number):
        try: return send(service, context, row, "send_" + str(number))["status"]
        except ValueError: return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert results.count("delivered_local") == 1 and results.count("rejected") == 1
    assert len(adapter.calls) == 1


def test_new_prepare_invalidates_prior_approved_selection(fixture):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    prepare(service, context, "new_selection")
    with pytest.raises(ValueError): send(service, context, row)
    assert not adapter.calls


def test_withdraw_before_send_and_before_final_dispatch_prevents_adapter(fixture, monkeypatch):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    original = service._dispatch
    def withdraw(sharing_id, current):
        action(service, context, "revoke", "withdraw", id=sharing_id)
        return original(sharing_id, current)
    monkeypatch.setattr(service, "_dispatch", withdraw)
    assert send(service, context, row)["status"] == "revoked"
    assert not adapter.calls


def test_revalidate_current_source_immediately_before_attempt(fixture):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    changed = replace(context, sources=())
    result = service.action(context, "send", {"request_id": "send", "id": row["id"], "preview_sha256": row["preview_sha256"]}, actor=OWNER, current_context=lambda: changed)
    assert result["record"]["status"] == "revoked" and not adapter.calls


@pytest.mark.parametrize("change", ["owner", "session", "run", "contact", "writable"])
def test_final_dispatch_rechecks_current_host_permissions_and_target(fixture, change):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    options = {"owner": {"owner_id": "other"}, "session": {"session_id": "other_session"},
               "run": {"run_id": "other_run"}, "contact": {"contacts": ()}, "writable": {"writable": False}}
    changed = replace(context, **options[change])
    result = service.action(context, "send", {"request_id": "send", "id": row["id"], "preview_sha256": row["preview_sha256"]}, actor=OWNER, current_context=lambda: changed)
    assert result["record"]["status"] == "revoked" and result["record"]["attempts"] == 0
    assert not adapter.calls


@pytest.mark.parametrize("fault", ["id", "hash", "receipt_id", "shape"])
def test_forged_or_unbound_adapter_receipts_are_unknown(fixture, monkeypatch, fault):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    receipt = LocalSharingReceipt("delivered_local", row["id"], row["preview_sha256"], "local_" + "a" * 24)
    if fault == "id": receipt = replace(receipt, sharing_id="other_share")
    if fault == "hash": receipt = replace(receipt, preview_sha256="0" * 64)
    if fault == "receipt_id": receipt = replace(receipt, receipt_id="claimed_real_delivery")
    if fault == "shape": receipt = {"status": "delivered_local"}
    monkeypatch.setattr(adapter, "send", lambda **_: receipt)
    result = send(service, context, row)
    assert result["status"] == "unknown" and result["receipt"] is None and result["attempts"] == 1


@pytest.mark.parametrize("outcome", ["accepted", "delivered", "unknown", TimeoutError("local_fixture_timeout")])
def test_receipts_never_upgrade_accepted_or_forge_real_delivery(fixture, outcome):
    service, context, adapter, *_ = fixture
    adapter.outcome = outcome
    row = approved(service, context)
    result = send(service, context, row)
    assert result["status"] == ("accepted" if outcome == "accepted" else "unknown")
    assert result["status"] != "delivered" and len(adapter.calls) == 1
    with pytest.raises(ValueError): send(service, context, row, "other_send")
    if result["status"] == "unknown":
        with pytest.raises(ValueError): prepare(service, context, "new_run_attempt")
        assert action(service, context, "revoke", "revoke", id=row["id"])["status"] == "unknown"


def test_crash_after_durable_attempt_reopens_unknown_never_retries(fixture):
    service, context, adapter, clock, path = fixture
    adapter.outcome = KeyboardInterrupt()
    row = approved(service, context)
    with pytest.raises(KeyboardInterrupt): send(service, context, row)
    assert service.view(context)["records"][0]["status"] == "sending"
    service.close()
    fresh = Adapter()
    reopened = ExplicitSharingService(path, owner_id=OWNER, adapter=fresh, clock=lambda: clock[0])
    try:
        assert reopened.view(context)["records"][0]["status"] == "unknown"
        assert reopened.view(context)["records"][0]["attempts"] == 1
        assert not fresh.calls
        with pytest.raises(ValueError): prepare(reopened, context, "new_id")
    finally: reopened.close()


def test_restart_invalidates_unattempted_approval(fixture):
    service, context, adapter, clock, path = fixture
    row = approved(service, context)
    service.close()
    reopened = ExplicitSharingService(path, owner_id=OWNER, clock=lambda: clock[0])
    try:
        assert reopened.view(context)["records"][0]["status"] == "expired"
        with pytest.raises(ValueError): send(reopened, context, row)
        assert not adapter.calls
    finally: reopened.close()


def test_failure_before_attempt_rolls_back_and_stops_store(fixture, monkeypatch):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    monkeypatch.setattr(service, "_put", lambda _: (_ for _ in ()).throw(sqlite3.OperationalError("fixture_save_error")))
    with pytest.raises(RuntimeError): send(service, context, row)
    assert service.failed and not adapter.calls
    assert service.view(context)["records"][0]["status"] == "approved"
    with pytest.raises(RuntimeError): prepare(service, context, "fresh")


def test_result_save_failure_keeps_sending_and_prevents_retry(fixture, monkeypatch):
    service, context, adapter, clock, path = fixture
    row = approved(service, context)
    original = service._put
    def fail_result(value):
        if value["status"] == "delivered_local": raise sqlite3.OperationalError("fixture_receipt_store_error")
        original(value)
    monkeypatch.setattr(service, "_put", fail_result)
    with pytest.raises(RuntimeError): send(service, context, row)
    assert service.failed and len(adapter.calls) == 1
    assert service.view(context)["records"][0]["status"] == "sending"
    service.close()
    reopened = ExplicitSharingService(path, owner_id=OWNER, clock=lambda: clock[0])
    try: assert reopened.view(context)["records"][0]["status"] == "unknown"
    finally: reopened.close()


def test_forget_erases_body_recipient_but_keeps_nonreplayable_tombstone(fixture):
    service, context, adapter, *_ = fixture
    row = approved(service, context)
    service.forget()
    value = service.view(context)["records"][0]
    assert value["forgotten"] and value["status"] == "revoked"
    assert "message" not in value and "source" not in value and "recipient" not in value
    with pytest.raises(ValueError): send(service, context, row)
    assert not adapter.calls
    raw = " ".join(row[0] for row in service._db.execute("SELECT data FROM sharing"))
    assert context.sources[0]["text"] not in raw and "虚构女儿" not in raw


def test_unknown_hold_survives_forget_and_new_run_identity(fixture):
    service, context, adapter, *_ = fixture
    adapter.outcome = TimeoutError("local_fixture_timeout")
    row = approved(service, context)
    assert send(service, context, row)["status"] == "unknown"
    service.forget()
    fresh = replace(context, run_id="fresh_run", session_id="fresh_session")
    value = service.view(fresh)
    assert value["requires_review"] and not value["records"]
    assert value["holds"] == [{"id": row["id"], "status": "unknown", "attempts": 1}]
    assert "message" not in value["holds"][0] and "recipient" not in value["holds"][0]
    with pytest.raises(ValueError): prepare(service, fresh, "fresh_request")
    assert len(adapter.calls) == 1


def test_external_adapter_and_wrong_owner_view_are_rejected(fixture, tmp_path):
    service, context, adapter, *_ = fixture
    class External(LocalSharingAdapter): external = True
    with pytest.raises(ValueError): ExplicitSharingService(tmp_path / "external.sqlite", owner_id=OWNER, adapter=External())
    with pytest.raises(PermissionError): service.view(replace(context, owner_id="other"))
    assert not adapter.calls


def test_same_request_id_cannot_be_repurposed_and_view_is_defensive(fixture):
    service, context, adapter, *_ = fixture
    row = prepare(service, context)
    with pytest.raises(ValueError): action(service, context, "revoke", "prepare", id=row["id"])
    copy = service.view(context);copy["records"][0]["message"] = "changed"
    assert service.view(context)["records"][0]["message"] == context.sources[0]["text"]
    assert hashlib.sha256(json.dumps(copy).encode()).hexdigest()
