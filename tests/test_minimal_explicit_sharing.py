"""Actual minimal app/HTTP integration using fictional saved utterances only."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from aria.voice.minimal import MinimalWebApp, make_minimal_server
from tests._model_http_stub import stub_model_http


OWNER = "fictional-chen"


@pytest.fixture
def app(tmp_path, monkeypatch):
    stub_model_http(monkeypatch, lambda *args, **kwargs: pytest.fail("model network forbidden"))
    value = MinimalWebApp(tmp_path / "data")
    value.care.bootstrap("carer", "local-sharing-fixture-2026", [OWNER, "other_elder"])
    value.care.elder_action(OWNER, "contact.create", {"request_id": "contact", "name": "虚构女儿", "relation": "女儿", "channel": "local"})
    value.care.elder_action("other_elder", "contact.create", {"request_id": "other_contact", "name": "别人的联系人", "channel": "local"})
    started = value.start()
    session_id = started["state"]["session_id"]
    assert value.submit("我愿意周六和你一起折纸风车。", "own_turn", session_id)["ok"]
    yield value, session_id
    value.close()


def command(app, session_id, action, request_id, **payload):
    return app.sharing_action(action, {"request_id": request_id, **payload}, session_id)


def draft(app, session_id):
    contact = app.sharing_state()["contacts"][0]
    return command(app, session_id, "prepare", "prepare", source_id="own_turn", contact_id=contact["id"])["record"]


def test_actual_saved_source_and_contact_are_owner_bound_no_person_model_copy(app):
    value, session_id = app
    state = value.state()
    assert state["sharing"]["sources"][0]["text"] == state["turns"][0]["text"]
    assert len(state["sharing"]["contacts"]) == 1
    assert state["sharing"]["contacts"][0]["elder_id"] == OWNER
    profile_before = deepcopy(state["profile"])
    row = draft(value, session_id)
    assert row["message"] == state["turns"][0]["text"]
    assert value.state()["profile"] == profile_before
    assert not value.state()["model_calls_enabled"]
    assert not value._sessions[OWNER][1].working_memory.get("sharing")


def test_distinct_approve_send_durable_local_receipt_no_reminder_delivery(app):
    value, session_id = app
    row = draft(value, session_id)
    row = command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)["record"]
    assert row["status"] == "approved"
    result = command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])
    assert result["record"]["status"] == "delivered_local"
    assert not value.care_state()["deliveries"]
    assert result["state"]["sharing"]["records"][0]["receipt"]["source"] == "local_simulation"
    assert result["state"]["services"]["external_notifications"] is False
    assert result["state"]["profile"]["facts"] == []


@pytest.mark.parametrize("actor", ["caregiver:carer", "other_elder"])
def test_caregiver_or_different_owner_cannot_replay_saved_owner_command(app, actor):
    value, session_id = app
    row = draft(value, session_id)
    contact_id = row["recipient"]["id"]
    with pytest.raises(PermissionError):
        value.sharing_action("prepare", {"request_id": "prepare", "source_id": "own_turn", "contact_id": contact_id}, session_id, actor=actor)
    token = value.care.login("carer", "local-sharing-fixture-2026")
    with pytest.raises(PermissionError):
        value.caregiver_action(token, OWNER, "sharing.send", {"request_id": "x", "id": row["id"]})
    assert value.sharing_state()["records"][0]["status"] == "draft"


def test_history_replay_and_stale_session_reject_saved_command(app):
    value, session_id = app
    row = draft(value, session_id)
    with pytest.raises(PermissionError):
        value.sharing_action("approve", {"request_id": "approve", "id": row["id"], "preview_sha256": row["preview_sha256"], "confirmed": True}, session_id, read_only=True)
    assert value.seal(session_id)["ok"]
    with pytest.raises(ValueError):
        command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    assert value.start()["ok"]
    current = value.state()["session_id"]
    with pytest.raises(ValueError):
        command(value, current, "approve", "newapprove", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)


def test_old_run_turn_remains_history_but_cannot_be_selected_for_sharing(app):
    value, session_id = app
    assert value.seal(session_id)["ok"]
    assert not value.sharing_state()["sources"]
    assert value.start()["ok"]
    current = value.state()["session_id"]
    assert not value.sharing_state()["sources"]
    contact = value.sharing_state()["contacts"][0]
    with pytest.raises(ValueError):
        command(value, current, "prepare", "old_history", source_id="own_turn", contact_id=contact["id"])
    assert value.state()["turns"][0]["turn_id"] == "own_turn"


def test_reminder_consent_does_not_approve_or_create_content_share(app):
    value, session_id = app
    contact = value.sharing_state()["contacts"][0]
    reminder = value.care.elder_action(OWNER, "reminder.create", {"request_id": "reminder", "label": "拿好纸风车", "schedule": "once", "date": "2027-01-01", "time": "15:00", "timezone": "Asia/Shanghai"})["record"]
    value.care.elder_action(OWNER, "consent.grant", {"request_id": "consent", "contact_id": contact["id"], "reminder_id": reminder["id"], "trigger": "unconfirmed", "scope": "reminder_status", "source": "simulation", "confirmed": True, "expires_at": "2027-02-01T00:00:00+00:00"})
    assert not value.sharing_state()["records"]
    row = draft(value, session_id)
    with pytest.raises(ValueError):
        command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])


def test_source_and_contact_current_bytes_checked_again_before_send(app):
    value, session_id = app
    row = draft(value, session_id)
    command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    value._run["turns"][0]["text"] = "来源已经改变。"
    with pytest.raises(ValueError):
        command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])
    assert value.sharing_state()["records"][0]["attempts"] == 0


def test_unreadable_contacts_disable_sharing_without_caching_or_blocking_chat(app, monkeypatch):
    value, session_id = app
    row = draft(value, session_id)
    command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    def unavailable(*args): raise OSError("private_storage_error_must_not_escape")
    monkeypatch.setattr(value.care, "elder_view", unavailable)
    state = value.state()
    assert not state["sharing"]["contacts"] and state["sharing"]["writable"] is False
    assert state["sharing"]["contacts_available"] is False
    assert "private_storage_error_must_not_escape" not in json.dumps(state)
    with pytest.raises(ValueError): command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])
    assert value.submit("我们接着聊纸风车。", "ordinary_next_turn", session_id)["ok"]
    assert value.sharing_state()["records"][0]["attempts"] == 0


def test_unreadable_sharing_is_unknown_stops_permissions_and_preserves_chat(app, monkeypatch):
    value, session_id = app
    row = draft(value, session_id)
    def unavailable(): raise sqlite3.OperationalError("private_sharing_store_error")
    monkeypatch.setattr(value.sharing, "_rows", unavailable)
    state = value.state()
    sharing = state["sharing"]
    assert sharing["available"] is False and sharing["requires_review"] is True
    assert sharing["storage_failed"] is True and sharing["writable"] is False
    assert not sharing["records"] and not sharing["commands"] and not sharing["sources"] and not sharing["contacts"]
    assert "private_sharing_store_error" not in json.dumps(state)
    assert value.sharing.failed
    with pytest.raises(RuntimeError): command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])
    assert value.submit("继续聊纸风车。", "ordinary_after_sharing_fault", session_id)["ok"]
    assert value.state()["turns"][-1]["status"] == "completed"


def test_forget_clears_only_local_bodies_and_new_run_cannot_replay_grant(app):
    value, session_id = app
    row = draft(value, session_id)
    command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    assert value.forget_run()["ok"]
    assert not value.sharing_state()["records"]
    assert not value.sharing_state()["sources"]
    assert value.start()["ok"]
    with pytest.raises(ValueError):
        command(value, value.state()["session_id"], "send", "oldsend", id=row["id"], preview_sha256=row["preview_sha256"])


def test_default_app_rejects_any_injected_sharing_gateway(tmp_path):
    with pytest.raises(ValueError, match="不能注入"):
        MinimalWebApp(tmp_path, sharing_service=object())


def test_shutdown_retains_readonly_sharing_state_without_any_send(app):
    value, session_id = app
    row = draft(value, session_id)
    value.close()
    assert value.state()["sharing"]["records"][0]["id"] == row["id"]
    assert value.state()["sharing"]["writable"] is False
    assert value.state()["sharing"]["records"][0]["attempts"] == 0
    with pytest.raises(ValueError): command(value, session_id, "send", "post_close", id=row["id"], preview_sha256=row["preview_sha256"])


def test_sharing_adds_zero_model_requests_to_actual_fixture_clients(tmp_path):
    from tests.test_minimal_web import live_fixture, start_and_submit
    value, clients = live_fixture(tmp_path)
    try:
        value.care.bootstrap("carer", "local-sharing-fixture-2026", [OWNER])
        value.care.elder_action(OWNER, "contact.create", {"request_id": "contact", "name": "虚构女儿", "channel": "local"})
        session_id = start_and_submit(value, ("我愿意周六一起折纸风车。",))
        source = value.sharing_state()["sources"][0]
        contact = value.sharing_state()["contacts"][0]
        before = tuple(len(client.calls) for client in (clients.online, clients.memory, clients.reflection))
        row = command(value, session_id, "prepare", "prepare", source_id=source["turn_id"], contact_id=contact["id"])["record"]
        command(value, session_id, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
        assert command(value, session_id, "send", "send", id=row["id"], preview_sha256=row["preview_sha256"])["record"]["status"] == "delivered_local"
        assert tuple(len(client.calls) for client in (clients.online, clients.memory, clients.reflection)) == before
        assert before == (1, 0, 0)
    finally: value.close()


def test_http_requires_demo_token_and_fixed_owner_scope_and_get_is_readonly(app):
    value, session_id = app
    server = make_minimal_server(value, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        def request(method, path, body=None, token=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            headers = {"Content-Type": "application/json"}
            if token: headers["X-Aria-Demo-Token"] = value.demo_token
            raw = None if body is None else json.dumps(body).encode("utf-8")
            connection.request(method, path, raw, headers)
            response = connection.getresponse()
            data = response.read()
            connection.close()
            return response.status, json.loads(data)
        path = value.run_dir / "sharing.sqlite3"
        before = path.read_bytes()
        assert request("GET", "/api/minimal/sharing", token=False)[0] == 403
        assert request("GET", "/api/minimal/sharing?elder_id=other_elder")[1]["contacts"][0]["elder_id"] == OWNER
        assert path.read_bytes() == before
        contact = value.sharing_state()["contacts"][0]
        payload = {"action": "prepare", "session_id": session_id, "payload": {"request_id": "http_prepare", "source_id": "own_turn", "contact_id": contact["id"]}}
        assert request("POST", "/api/minimal/sharing", payload, token=False)[0] == 403
        assert not value.sharing_state()["records"]
        status, result = request("POST", "/api/minimal/sharing", payload)
        assert status == 200 and result["record"]["status"] == "draft"
        payload["actor"] = "caregiver:carer"
        assert request("POST", "/api/minimal/sharing", payload)[0] == 403
        payload.pop("actor");payload["replay"] = True
        assert request("POST", "/api/minimal/sharing", payload)[0] == 403
    finally:
        server.shutdown();server.server_close();worker.join()


def test_page_has_exact_preview_separate_owner_approval_send_and_text_nodes():
    import aria.voice.minimal as minimal
    page = Path(minimal._PAGE).read_text(encoding="utf-8")
    assert 'id="sharing-source"' in page and 'id="sharing-contact"' in page
    assert "sharingAction('approve'" in page and "sharingAction('send'" in page
    sharing_js = page.split("    function sharingControls()", 1)[1].split("    function renderChoices()", 1)[0]
    assert "innerHTML" not in sharing_js
    assert "viewingHistory()" in sharing_js and "replayMode" in sharing_js
    assert "sharingSelectionDirty" in sharing_js and "preview_sha256" in sharing_js
    assert "setInterval" not in sharing_js and "local_simulation" in sharing_js
