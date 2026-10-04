"""Actual Engine with local recording clients; no model-quality claims."""
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.voice.minimal import MinimalWebApp
from tests.test_minimal_care import ELDER, care, command, setup_rule, advance  # noqa: F401
from tests.test_minimal_web import live_fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("external request forbidden")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def send(app, text, turn_id="first"):
    sid = app.start()["state"]["session_id"]
    result = app.submit(text, turn_id, sid)
    assert result["ok"], result.get("error")
    return result["state"]["turns"][-1]


def prompt_snapshot(client):
    prompt = client.calls[-1][0]["content"]
    marker = '<aria_untrusted_data source="minimal_care_records">\n'
    return json.loads(prompt.split(marker)[1].split("\n＜/aria_untrusted_data＞")[0])


def test_saved_form_state_reaches_actual_prompt_without_executing_care(tmp_path, care):
    service, _token, _clock = care
    rule, _contact, _consent = setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    writes_before = service._db.total_changes
    original = deepcopy(app.care_state())
    row = send(app, "现在是什么状态？告诉她了吗？")
    supplied = prompt_snapshot(clients.online)
    assert supplied["source"] == "minimal_care_records"
    assert supplied["reminders"][0]["id"] == rule["id"]
    assert supplied["reminders"][0]["label"] == "准备纸和本子"
    assert supplied["reminders"][0]["status"] == "active"
    assert supplied == row["care_context"]
    assert app.care_state() == original
    assert service._db.total_changes == writes_before
    assert not clients.memory.calls and not clients.reflection.calls
    assert "minimal_care_context" not in app._sessions[ELDER][1].working_memory


def test_next_turn_reads_current_status_but_duplicate_and_replay_keep_old_basis(tmp_path, care):
    service, _token, clock = care
    rule, _contact, consent = setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    first = deepcopy(send(app, "提醒安排好了没有？"))
    command(service, "consent.revoke", "revoke", consent_id=consent["id"])
    command(service, "reminder.cancel", "cancel", reminder_id=rule["id"])
    duplicate = app.submit(first["text"], first["turn_id"], first["session_id"])
    assert duplicate["duplicate"] and len(clients.online.calls) == 1
    assert duplicate["state"]["turns"][0] == first
    second = send(app, "再看看提醒和授权状态。", "second")
    assert second["care_context"]["reminders"][0]["status"] == "cancelled"
    assert second["care_context"]["consents"][0]["status"] == "revoked"
    assert prompt_snapshot(clients.online)["consents"][0]["valid_now"] is False
    before = app.run_path.read_bytes()
    assert app.state()["turns"][0] == first
    assert app.run_path.read_bytes() == before
    assert len(clients.online.calls) == 2


def test_local_delivery_and_family_roleplay_stay_local_in_prompt(tmp_path, care):
    service, token, clock = care
    setup_rule(service)
    advance(service, clock, 30)
    advance(service, clock, 60)
    delivery = service.elder_view(ELDER)["deliveries"][0]
    service.mutate(token, ELDER, "receipt.record", {
        "request_id": "roleplay", "delivery_id": delivery["id"], "status": "family_ack",
        "evidence_source": "private-roleplay-evidence", "note": "private-followup-note",
    })
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    send(app, "这个通知现在怎么样？")
    snap = prompt_snapshot(clients.online)
    assert snap["occurrences"][0]["status"] == "unconfirmed"
    assert snap["deliveries"][0]["status"] == "family_ack"
    assert snap["deliveries"][0]["source"] == "simulation"
    assert snap["deliveries"][0]["channel"] == "local_care_inbox"
    assert not snap["external_enabled"]
    prompt = str(clients.online.calls[-1])
    assert "不证明真实家属收到" in prompt
    for private in ("虚构女儿", "private-roleplay-evidence", "private-followup-note"):
        assert private not in prompt


def test_unrelated_chat_drops_snapshot_and_offline_does_not_receive_raw_care_data(tmp_path, care):
    service, _token, _clock = care
    setup_rule(service, label="护理私有标签ZXQ")
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    send(app, "提醒现在是什么状态？")
    assert "护理私有标签ZXQ" in str(clients.online.calls[-1])
    second = send(app, "今天我只想坐一会儿。", "second")
    send(app, "窗外的风还很大。", "third")
    assert "care_context" not in second
    assert "minimal_care_records" not in str(clients.online.calls[-1])
    assert "护理私有标签ZXQ" not in str(clients.online.calls[-1])
    app.seal(second["session_id"])
    assert app.reflect("explicit-review")["ok"]
    assert "护理私有标签ZXQ" not in str(clients.memory.calls + clients.reflection.calls)
    assert not app.state()["profile"]["hypotheses"]


def test_unreadable_care_is_unknown_and_does_not_reuse_last_snapshot(tmp_path, care, monkeypatch):
    service, _token, _clock = care
    setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    send(app, "提醒现在是什么状态？")
    def fail(*args):
        raise OSError("private-storage-error-text")
    monkeypatch.setattr(service, "elder_view", fail)
    row = send(app, "再看看提醒的状态。", "unreadable")
    snap = prompt_snapshot(clients.online)
    assert snap["status"] == row["care_context"]["status"] == "unavailable"
    assert "reminders" not in snap and "counts" not in snap
    assert "private-storage-error-text" not in str(row) + str(clients.online.calls[-1])
    assert "准备纸和本子" not in str(clients.online.calls[-1])


def test_preview_and_restart_keep_historical_snapshot_without_model_or_care_write(tmp_path):
    app = MinimalWebApp(tmp_path)
    row = deepcopy(send(app, "提醒现在是什么状态？"))
    assert row["source"] == "local_preview"
    assert row["care_context"]["status"] == "not_configured"
    assert not app.state()["profile"]["facts"]
    app.close()
    resumed = MinimalWebApp(tmp_path)
    assert resumed.state()["turns"][0] == row
    assert resumed.state()["model_calls_enabled"] is False
    resumed.close()


def test_care_text_cannot_close_unexecuted_action_or_grant_authority(tmp_path, care):
    service, _token, _clock = care
    setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    clients.online.payload = None
    clients.online.complete = lambda messages: "我已经通知你女儿了。"
    before = deepcopy(app.care_state())
    row = send(app, "请告诉女儿提醒没确认。")
    assert row["response"] != "我已经通知你女儿了。"
    assert row["source"] == "rule_protection"
    assert row["delivery_protection"]["response_changed"] is True
    assert row["delivery_protection"]["activated"] is True
    assert app.care_state() == before
    assert not before["deliveries"]


def test_verified_record_description_remains_model_text_not_an_action_receipt(tmp_path, care):
    service, _token, _clock = care
    rule, _contact, consent = setup_rule(service)
    command(service, "reminder.cancel", "cancel", reminder_id=rule["id"])
    command(service, "consent.revoke", "revoke", consent_id=consent["id"])
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    response = "页面记录显示，提醒已取消，授权已撤回。"
    clients.online.complete = lambda messages: response
    row = send(app, "提醒和授权现在怎样？")
    assert row["response"] == response
    assert row["source"] == "live"  # Recording client is a fixture, not an effect result.
    assert row["delivery_protection"]["response_changed"] is False


def test_snapshot_persistence_fails_before_any_model_call(tmp_path, care, monkeypatch):
    service, _token, _clock = care
    setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    sid = app.start()["state"]["session_id"]
    def fail():
        raise OSError("fixture write failure")
    monkeypatch.setattr(app, "_save_run", fail)
    before = deepcopy(app.care_state())
    with pytest.raises(OSError):
        app.submit("提醒现在怎样？", "write-failure", sid)
    assert not clients.online.calls
    assert app.care_state() == before
    assert "minimal_care_context" not in app._sessions[ELDER][1].working_memory
