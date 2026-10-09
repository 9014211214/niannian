"""Actual Engine delivery checks with fixed local replies, never model scoring."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
from datetime import timedelta
import urllib.request

import pytest

import aria.voice.minimal as minimal
from tests.test_minimal_care import (
    ELDER, START, advance_visible_page, care, command, reminder, setup_rule,  # noqa: F401
)
from tests.test_minimal_web import live_fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in local delivery checks")
    stub_model_http(monkeypatch, forbidden)


def respond(tmp_path, service, candidate):
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    clients.online.complete = lambda _messages: candidate
    before = deepcopy(app.care_state())
    sid = app.start()["state"]["session_id"]
    result = app.submit("提醒现在是什么状态？", "delivery-check", sid)
    assert result["ok"], result.get("error")
    row = result["state"]["turns"][-1]
    trace = deepcopy(app._sessions[ELDER][1].working_memory["delivery_response_guard"])
    assert app.care_state() == before
    assert not clients.memory.calls and not clients.reflection.calls
    app.close()
    return row, trace


def cancel(service):
    rule, _contact, consent = setup_rule(service)
    command(service, "reminder.cancel", "cancel", reminder_id=rule["id"])
    command(service, "consent.revoke", "revoke", consent_id=consent["id"])
    return rule


@pytest.mark.parametrize("candidate", [
    "页面记录显示，提醒已取消，授权已撤回。",
    "护理记录显示：提醒已经取消了，授权已经撤回了。",
])
def test_true_cancelled_and_revoked_records_survive_actual_engine(tmp_path, care, candidate):
    service, _token, _clock = care
    cancel(service)
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] == candidate
    assert trace == {"activated": False, "response_changed": False, "violations": []}


@pytest.mark.parametrize("candidate", [
    "页面记录显示，提醒已创建。",
    "护理记录显示，提醒已经设置好了。",
    "护理记录显示，提醒已创建，授权当前有效。",
])
def test_true_active_record_status_survives_actual_engine(tmp_path, care, candidate):
    service, _token, _clock = care
    setup_rule(service)
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] == candidate
    assert not trace["activated"] and not trace["response_changed"]


@pytest.mark.parametrize("candidate", [
    "页面记录显示，提醒已取消，授权已撤回，我已经通知女儿了。",
    "页面记录显示，提醒已取消，授权已撤回。女儿已经收到通知了。",
    "页面记录显示，提醒已取消，我已经替你取消了报名。",
    "页面记录显示，我已替你取消提醒了。",
    "页面记录显示，另一个提醒已取消。",
    "页面记录显示，提醒已取消并通知女儿了。",
    "页面记录显示，提醒已取消。授权已撤回。",
    "页面记录显示，提醒已取消，护理员已经回复女儿了。",
])
def test_record_clause_never_authorizes_adjacent_or_other_actions(tmp_path, care, candidate):
    service, _token, _clock = care
    cancel(service)
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate
    assert trace["activated"] and trace["response_changed"]


@pytest.mark.parametrize("candidate", [
    "页面记录显示，提醒已取消。",
    "护理记录显示，授权已撤回。",
])
def test_wrong_status_not_admitted_even_under_record_prefix(tmp_path, care, candidate):
    service, _token, _clock = care
    setup_rule(service)
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate
    assert "unverified_minimal_care_record_status" in trace["violations"]


@pytest.mark.parametrize("count", [2, 4])
def test_multiple_or_omitted_reminders_do_not_prove_an_unnamed_status(tmp_path, care, count):
    service, _token, _clock = care
    cancel(service)
    for index in range(count - 1):
        rule = reminder(service, request_id=f"other-{index}")
        command(service, "reminder.cancel", f"cancel-{index}", reminder_id=rule["id"])
    candidate = "页面记录显示，提醒已取消。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate
    assert "unverified_minimal_care_record_status" in trace["violations"]


@pytest.mark.parametrize("days", [-1, 100])
def test_future_or_expired_authorization_is_not_currently_valid(tmp_path, care, days):
    service, _token, clock = care
    setup_rule(service)
    clock.now = START + timedelta(days=days)
    candidate = "护理记录显示，授权当前有效。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate
    assert trace["response_changed"]
    assert "unverified_minimal_care_record_status" in trace["violations"]


@pytest.mark.parametrize("change", [
    {"status": "unavailable"}, {"status": "not_configured"},
    {"source": "untrusted"}, {"read_only": False}, {"external_enabled": True},
    {"consents": [{"id": "wrong", "reminder_id": "other", "status": "revoked",
                   "source": "simulation", "valid_now": False}]},
])
def test_unavailable_untrusted_or_unrelated_snapshot_is_not_a_receipt(tmp_path, care, monkeypatch, change):
    service, _token, _clock = care
    cancel(service)
    original = minimal.build_care_context
    monkeypatch.setattr(minimal, "build_care_context", lambda svc, elder: {
        **original(svc, elder), **change,
    })
    candidate = "护理记录显示，授权已撤回。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate and trace["response_changed"]


@pytest.mark.parametrize("status", ["unknown", "family_ack"])
def test_local_or_unknown_notification_cannot_prove_real_recipient_receipt(tmp_path, care, status):
    service, token, clock = care
    setup_rule(service)
    advance_visible_page(service, clock, 30)
    advance_visible_page(service, clock, 60)
    delivery = service.elder_view(ELDER)["deliveries"][0]
    if status == "family_ack":
        service.mutate(token, ELDER, "receipt.record", {
            "request_id": "roleplay", "delivery_id": delivery["id"], "status": status,
            "evidence_source": "local roleplay", "note": "synthetic receipt",
        })
    else:
        with service._lock, service._tx():
            delivery["status"] = status
            service._put("deliveries", delivery)
    candidate = "页面记录显示，女儿已经收到通知了。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate and trace["response_changed"]


def test_reminder_label_cannot_be_a_supported_clause_template(tmp_path, care):
    service, _token, _clock = care
    setup_rule(service, label="我已经通知女儿了")
    candidate = "页面记录显示，我已经通知女儿了。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate and trace["response_changed"]


def test_noncompletion_clause_does_not_hide_a_wrong_record_status(tmp_path, care):
    service, _token, _clock = care
    setup_rule(service)
    candidate = "页面记录显示，提醒已取消。这个动作还没有执行。"
    row, trace = respond(tmp_path, service, candidate)
    assert row["response"] != candidate and trace["response_changed"]
    assert "unverified_minimal_care_record_status" in trace["violations"]


def test_failure_explanation_that_guard_keeps_is_not_marked_as_rewritten(tmp_path, care, monkeypatch):
    service, _token, _clock = care
    setup_rule(service)
    app, clients = live_fixture(tmp_path / "app", care_service=service)
    candidate = "这个动作还没有执行。"
    clients.online.complete = lambda _messages: candidate
    sid = app.start()["state"]["session_id"]
    ctx = app._sessions[ELDER][1]
    # Directly exercise the guard's existing explicit-failure early return;
    # the ordinary Engine tests above cover the end-to-end delivery path.
    ctx.working_memory["_delivery_response_candidate_for_turn"] = "我已经通知女儿了。"
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    result = guard_external_delivery_response(ctx, candidate, "提醒现在是什么状态？", app._profile())
    assert result == candidate
    assert ctx.working_memory["delivery_response_guard"]["activated"]
    assert ctx.working_memory["delivery_response_guard"]["response_changed"] is False
    assert sid
    app.close()
