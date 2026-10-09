"""Automatic scheduling with recording software fixtures, never live models."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import threading

import pytest

from aria.voice.auto_review import AutoReviewQuota, default_settings, local_day, stable_request_id, view
from aria.engine.model.persona import StatedPreference, PreferenceQuote
from aria.engine.policy.minimal_calls import MinimalCallGuard
from tests.preference_fixtures import communication_review
from aria.voice.minimal import MinimalWebApp
from tests._model_http_stub import stub_model_http
from tests.test_minimal_web import live_fixture, start_and_submit


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("automatic review must not reach a real provider")
    stub_model_http(monkeypatch, forbidden)


def enable(app, daily_limit=1, request_id="enable"):
    assert app.auto_review_action(True, daily_limit, confirmed=True, request_id=request_id)["ok"]


def sealed_queue(app):
    sid = start_and_submit(app, ["这纸风车翅膀折反了。", "今天先修轮轴。", "然后再把叶片装上。"])
    with app._user_lock("fictional-chen"):
        app._seal_active_session("fictional-chen", app._sessions["fictional-chen"])
    return sid


def wait_worker(app, worker=None):
    worker = worker or app._review_thread
    if worker:
        worker.join(5)
        assert not worker.is_alive(), "software fixture review did not finish"
    return app.state()


def test_old_run_migrates_disabled_without_read_writes(tmp_path):
    app = MinimalWebApp(tmp_path)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    app.seal(sid)
    legacy = deepcopy(app._run)
    del legacy["auto_review"]
    app.run_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    before = app.run_path.read_bytes()
    reopened = MinimalWebApp(tmp_path)
    assert reopened.state()["auto_review"]["enabled"] is False
    assert reopened._run["auto_review"] == default_settings()
    reopened.state()
    assert reopened.run_path.read_bytes() == before
    reopened.close()
    app.close()


@pytest.mark.parametrize("value,limit", [("true", 1), (1, 1), (True, True), (True, 0), (True, 9), (True, 1.0)])
def test_settings_strict_input(tmp_path, value, limit):
    app = MinimalWebApp(tmp_path)
    assert not app.auto_review_action(value, limit, confirmed=True)["ok"]
    assert not app.state()["auto_review"]["enabled"]
    app.close()


def test_setting_requires_owner_confirmation_and_forbids_replay(tmp_path):
    app = MinimalWebApp(tmp_path)
    assert not app.auto_review_action(True)["ok"]
    assert not app.auto_review_action(True, confirmed=True, read_only=True)["ok"]
    with pytest.raises(PermissionError):
        app.auto_review_action(True, confirmed=True, actor="caregiver")
    assert not app.state()["auto_review"]["enabled"]
    app.close()


def test_persistent_setting_commands_are_idempotent_without_reactivating_old_request(tmp_path):
    app = MinimalWebApp(tmp_path)
    enable(app, 2)
    first = app.run_path.read_bytes()
    duplicate = app.auto_review_action(True, 2, confirmed=True, request_id="enable")
    assert duplicate["ok"] and duplicate["duplicate"] and app.run_path.read_bytes() == first
    assert not app.auto_review_action(True, 3, confirmed=True, request_id="enable")["ok"]
    assert app.auto_review_action(False, 2, request_id="disable")["ok"]
    closed = app.run_path.read_bytes()
    assert app.auto_review_action(True, 2, confirmed=True, request_id="enable")["duplicate"]
    assert not app.state()["auto_review"]["enabled"] and app.run_path.read_bytes() == closed
    reopened = MinimalWebApp(tmp_path)
    assert not reopened.state()["auto_review"]["enabled"]
    assert reopened.state()["auto_review"]["last_command_id"] == "disable"
    reopened.close()
    app.close()


def test_automatic_first_review_follows_seal_and_never_repeats(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。", "今天先修轮轴。", "然后再把叶片装上。"])
    assert not clients.memory.calls and not clients.reflection.calls
    assert app.seal(sid)["ok"]
    state = wait_worker(app)
    event = state["auto_review"]["events"][0]
    assert event["status"] == "completed" and state["auto_review"]["today_started"] == 1
    assert event["request_id"] == stable_request_id(state["run_id"], sid)
    assert state["reviews"][0]["request_id"] == event["request_id"]
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    assert app.end()["ok"] and not app.tick_auto_review()["started"]
    reopened = MinimalWebApp(tmp_path, clients=clients)
    assert reopened.state()["auto_review"]["events"][0]["status"] == "completed"
    assert not reopened.tick_auto_review()["started"]
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    reopened.close()
    app.close()


def test_idle_uses_the_same_seal_event_and_scheduler(tmp_path):
    app, clients = live_fixture(tmp_path, session_idle_seconds=1)
    enable(app)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    assert app.finalize_idle_sessions(now=app._session_activity["fictional-chen"] + 2) == ["fictional-chen"]
    assert not clients.memory.calls
    assert app.state()["auto_review"]["events"][0]["session_id"] == sid
    assert app.tick_auto_review()["started"]
    assert wait_worker(app)["auto_review"]["events"][0]["status"] == "completed"
    app.close()


def test_preview_blocks_without_manufacturing_learning(tmp_path):
    app = MinimalWebApp(tmp_path)
    enable(app)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    app.seal(sid)
    state = app.state()
    assert state["auto_review"]["events"][0]["reason"] == "models_disabled"
    assert not state["reviews"] and not state["profile"]["facts"] and not state["profile"]["hypotheses"]
    assert state["auto_review"]["today_started"] == 0
    assert not app.tick_auto_review()["started"]
    app.close()


def test_enabling_does_not_authorize_old_sealed_head(tmp_path):
    app, clients = live_fixture(tmp_path)
    old = sealed_queue(app)
    enable(app)
    new = sealed_queue(app)
    result = app.tick_auto_review()
    assert result["reason"] == "earlier_manual_job" and not result["started"]
    assert not clients.memory.calls
    assert [row["session_id"] for row in app.state()["auto_review"]["events"]] == [new]
    assert app.reflect("manual-old", wait=True)["ok"]
    assert app.state()["reviews"][0]["session_id"] == old
    assert app.tick_auto_review()["started"]
    state = wait_worker(app)
    assert state["auto_review"]["events"][0]["status"] == "completed"
    assert len(clients.memory.calls) == 2
    app.close()


def test_daily_limit_defers_second_job_without_model_call(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    first = sealed_queue(app)
    assert app.tick_auto_review()["started"]
    wait_worker(app)
    second = sealed_queue(app)
    result = app.tick_auto_review()
    assert result["reason"] == "daily_job_limit" and not result["started"]
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    state = app.state()["auto_review"]
    assert state["today_started"] == 1
    assert [row["session_id"] for row in state["events"]] == [first, second]
    assert state["events"][1]["status"] == "deferred"
    assert app.auto_review_action(False, request_id="disable-limit")["ok"]
    assert app.state()["auto_review"]["events"][1]["reason"] == "owner_disabled"
    app.close()


def test_daily_counter_uses_shanghai_date_and_counts_failed_started_jobs():
    settings = default_settings()
    for day in ("2026-10-08", "2026-10-09"):
        settings["events"].append({"status": "failed", "reason": "review_not_completed", "started_on": day})
    clock = datetime(2026, 10, 8, 16, 0, tzinfo=timezone.utc)
    assert local_day(clock) == "2026-10-09"
    assert view(settings, now=clock)["today_started"] == 1
    with pytest.raises(ValueError):
        local_day(clock.replace(tzinfo=None))


def test_daily_quota_survives_forget_and_restart_without_chat_text(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    assert app.tick_auto_review()["started"]
    wait_worker(app)
    assert app.forget_run()["ok"]
    state = app.state()["auto_review"]
    assert not state["enabled"] and state["today_started"] == 1 and not state["events"]
    app.close()
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert restarted.state()["auto_review"]["today_started"] == 1
    enable(restarted)
    sealed_queue(restarted)
    assert restarted.tick_auto_review()["reason"] == "daily_job_limit"
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    assert restarted.state()["auto_review"]["quota_scope"] == "current_data_directory_and_mode_not_account_wide"
    restarted.close()


def test_daily_quota_is_atomic_across_instances_and_new_day_is_independent(tmp_path):
    path = tmp_path / "quota.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: AutoReviewQuota(path).reserve(f"request-{index}", "2026-10-09", 1), range(4)))
    assert results.count("reserved") == 1 and results.count("daily_limit") == 3
    quota = AutoReviewQuota(path)
    assert quota.used("2026-10-09") == 1
    assert quota.reserve("next-day", "2026-10-10", 1) == "reserved"
    assert quota.reserve("next-day", "2026-10-10", 1) == "duplicate"
    assert quota.used("2026-10-10") == 1


def test_read_only_disabled_quota_does_not_create_file(tmp_path):
    path = tmp_path / "not-created.sqlite3"
    quota = AutoReviewQuota(path)
    assert quota.used("2026-10-09") == 0
    quota.validate_requests({})
    assert not path.exists()


def test_missing_retained_quota_fails_closed_without_modifying_run(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    assert app.tick_auto_review()["started"]
    wait_worker(app)
    run_before = app.run_path.read_bytes()
    # Named isolated test file, not an operational cleanup or history deletion.
    path = app._auto_review_quota.path
    quota_before = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(ValueError, match="quota_missing_or_changed"):
            MinimalWebApp(tmp_path, clients=clients)
        assert app.run_path.read_bytes() == run_before
    finally:
        path.write_bytes(quota_before)
    app.close()


@pytest.mark.parametrize("fault,reason", [
    ("attempt", "previous_attempt"), ("started", "previous_attempt"),
    ("incomplete", "source_incomplete_or_expired"), ("expired", "source_incomplete_or_expired"),
    ("partial", "prior_stage_result"), ("changed", "sealed_source_changed"),
])
def test_prior_or_bad_source_is_blocked_without_retry(tmp_path, fault, reason):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    profile = app._profile()
    job = profile.offline_sessions[0]
    if fault == "attempt":
        job.attempts = 1
    elif fault == "started":
        job.state = "started"
    elif fault == "incomplete":
        job.source_completeness = "legacy_tail_unknown_indices"
    elif fault == "expired":
        job.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    elif fault == "partial":
        job.memory_completed = True
    else:
        job.turns[0].user = "changed software source"
    app.store.save(profile)
    assert app.tick_auto_review()["reason"] == reason
    assert not app.tick_auto_review()["started"]
    assert not clients.memory.calls and not clients.reflection.calls
    assert app.state()["auto_review"]["today_started"] == 0
    app.close()


@pytest.mark.parametrize("fault,reason", [("storage", "storage_failed"), ("unknown", "result_requires_review"), ("stop", "models_stopped")])
def test_stop_gates_block_and_later_change_never_retries(tmp_path, monkeypatch, fault, reason):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    if fault == "storage":
        app._storage_failed = True
    elif fault == "unknown":
        app._run["turns"][0]["status"] = "pending"
    else:
        monkeypatch.setattr(app, "_models_stopped", lambda: True)
    assert app.tick_auto_review()["reason"] == reason
    app._storage_failed = False
    app._run["turns"][0]["status"] = "completed"
    monkeypatch.setattr(app, "_models_stopped", lambda: False)
    assert not app.tick_auto_review()["started"] and not clients.memory.calls
    app.close()


def test_owner_can_disable_while_unknown_and_cannot_reenable(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    app._run["turns"][0]["status"] = "pending"
    app._save_run()
    assert app.auto_review_action(False, request_id="stop-unknown")["ok"]
    assert not app.auto_review_action(True, request_id="restart-unknown", confirmed=True)["ok"]
    assert not app.tick_auto_review()["started"] and not clients.memory.calls
    app.close()


def test_setting_save_failure_does_not_activate(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path)
    original = app._save_run
    def fail():
        app._storage_failed = True
        raise RuntimeError("injected save failure")
    monkeypatch.setattr(app, "_save_run", fail)
    assert not app.auto_review_action(True, confirmed=True)["ok"]
    assert not app.state()["auto_review"]["enabled"]
    monkeypatch.setattr(app, "_save_run", original)
    app.close()


def test_shutdown_seals_without_starting_and_restart_does_not_launch(tmp_path):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    app.close()
    assert not clients.memory.calls and not clients.reflection.calls
    reopened = MinimalWebApp(tmp_path, clients=clients)
    event = reopened.state()["auto_review"]["events"][0]
    assert event["session_id"] == sid and event["reason"] == "application_closing"
    assert not reopened.tick_auto_review()["started"]
    assert not clients.memory.calls
    reopened.close()


def test_actual_background_restart_is_pending_not_replayed(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sid = sealed_queue(app)
    entered, release = threading.Event(), threading.Event()
    original = clients.memory.complete
    def delayed(messages):
        entered.set()
        assert release.wait(5)
        return original(messages)
    monkeypatch.setattr(clients.memory, "complete", delayed)
    assert app.tick_auto_review()["started"]
    worker = app._review_thread
    try:
        assert entered.wait(2)
        reopened = MinimalWebApp(tmp_path, clients=clients)
        retained = reopened.run_path.read_bytes()
        state = reopened.state()
        assert state["requires_review"]
        assert state["auto_review"]["events"][0]["status"] == "pending"
        assert state["auto_review"]["events"][0]["session_id"] == sid
        assert not reopened.tick_auto_review()["started"]
        assert not reopened.start()["ok"] and not reopened.reflect("new-review")["ok"]
        assert reopened.run_path.read_bytes() == retained
        assert not clients.reflection.calls
        reopened.close()
    finally:
        release.set()
        wait_worker(app, worker)
    app.close()


@pytest.mark.parametrize("mutation", ["enabled_string", "limit_bool", "fake_request", "repeat_session", "orphan_permission", "setting_digest", "event_authorization"])
def test_malformed_retained_settings_fail_closed_and_keep_original(tmp_path, mutation):
    app = MinimalWebApp(tmp_path)
    enable(app)
    sealed_queue(app)
    value = deepcopy(app._run)
    settings = value["auto_review"]
    if mutation == "enabled_string":
        settings["enabled"] = "true"
    elif mutation == "limit_bool":
        settings["daily_limit"] = True
    elif mutation == "fake_request":
        settings["events"][0]["request_id"] = "spoof"
    elif mutation == "repeat_session":
        settings["events"].append(deepcopy(settings["events"][0]))
    elif mutation == "orphan_permission":
        settings["commands"] = []
    elif mutation == "setting_digest":
        settings["commands"][-1]["digest"] = "a" * 64
    else:
        settings["events"][0]["authorized_at"] = "2026-10-08T00:00:00+00:00"
    app.run_path.write_text(json.dumps(value), encoding="utf-8")
    retained = app.run_path.read_bytes()
    with pytest.raises(ValueError):
        MinimalWebApp(tmp_path)
    assert app.run_path.read_bytes() == retained
    # Return the valid source to let cleanup close without overwriting the bad
    # artifact being inspected above; this is an isolated software fixture.
    app._save_run()
    app.close()


def seed_dialogue_pref(app, sid):
    profile = app._profile()
    stamp = datetime.now(timezone.utc).isoformat()
    profile.stated_preferences = [StatedPreference(id="pref_dialogue", do="先询问本人意愿", scope="安排之前先问我",
        quotes=[PreferenceQuote(session_id=sid, turn_index=0, text="安排之前先问我。")],
        semantic_review=communication_review(), created_at=stamp, updated_at=stamp)]
    app.store.save(profile)


def test_dialogue_web_context_invitation_receipt_is_current_owner_only(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["安排之前先问我。"])
    seed_dialogue_pref(app, sid)
    before = len(clients.online.calls)
    assert not app.preference_invitation("pref_dialogue", "wrong-session")["ok"]
    assert not app.preference_invitation("pref_dialogue", sid, read_only=True)["ok"]
    with pytest.raises(PermissionError):
        app.preference_invitation("pref_dialogue", sid, actor="caregiver")
    result = app.preference_invitation("pref_dialogue", sid)
    invitation = result["state"]["preference_invitation"]
    assert result["ok"] and invitation["status"] == "awaiting_display"
    assert not app.preference_invitation_ack(invitation["id"], "bad-hash", sid)["ok"]
    assert not app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], "other-session")["ok"]
    assert app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], sid)["ok"]
    assert app.state()["preference_invitation"]["status"] == "delivered"
    assert len(clients.online.calls) == before and not clients.memory.calls
    assert app._profile().stated_preferences[0].status == "proposed"
    app.close()


def test_dialogue_web_restart_does_not_retain_invitation_authority(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["安排之前先问我。"])
    seed_dialogue_pref(app, sid)
    invitation = app.preference_invitation("pref_dialogue", sid)["state"]["preference_invitation"]
    app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], sid)
    # A checkpoint stores ordinary source but not the next-turn grant.
    app.checkpoint("fictional-chen")
    reopened = MinimalWebApp(tmp_path, clients=clients)
    reopened.start()
    assert reopened.state()["preference_invitation"] is None
    assert not reopened.preference_invitation_ack(invitation["id"], invitation["question_sha256"], sid)["ok"]
    reopened.close()
    app.close()


@pytest.mark.parametrize("action", ["preference", "readiness"])
def test_dialogue_web_pending_or_disabled_models_do_not_stage(tmp_path, action):
    app = MinimalWebApp(tmp_path)
    sid = start_and_submit(app, ["安排之前先问我。"])
    result = app.preference_invitation("pref_dialogue", sid) if action == "preference" else app.cognitive_readiness(sid)
    assert not result["ok"]
    assert app.state()["preference_invitation"] is None
    app.close()


def test_readiness_web_wrong_session_replay_and_no_qualification_are_rejected(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["安排之前先问我。"])
    assert not app.cognitive_readiness("wrong-session")["ok"]
    assert not app.cognitive_readiness(sid, read_only=True)["ok"]
    assert not app.cognitive_readiness(sid)["ok"]
    with pytest.raises(PermissionError):
        app.cognitive_readiness(sid, actor="caregiver")
    assert len(clients.online.calls) == 1 and not clients.memory.calls
    app.close()


def test_thread_start_failure_keeps_unknown_quota_and_closes_owned_resources(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sid = sealed_queue(app)
    # A real local accounting connection, with no network client attached.
    guard = MinimalCallGuard(app.run_dir / "thread-failure-calls.sqlite3", max_requests=8,
                             max_observed_cost_usd="0.5")
    app.call_guard = guard
    closed = []
    original_close = guard.close
    def close_guard():
        closed.append(True)
        original_close()
    monkeypatch.setattr(guard, "close", close_guard)
    def fail_start(thread):
        raise RuntimeError("injected thread-start failure")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    assert not app.tick_auto_review()["started"]
    state = app.state()
    assert state["requires_review"] and app._review_thread is None
    assert state["reviews"][0]["status"] == "pending" and state["reviews"][0]["requires_review"]
    assert state["reviews"][0]["session_id"] == sid
    assert state["auto_review"]["events"][0]["status"] == "pending"
    assert state["auto_review"]["today_started"] == 1
    assert not app.tick_auto_review()["started"] and not clients.memory.calls
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert restarted.state()["requires_review"] and not restarted.tick_auto_review()["started"]
    restarted.close()
    app.close()
    assert app.care._closed and closed == [True]
    assert not clients.memory.calls and not clients.reflection.calls


def test_thread_started_before_start_exception_is_retained_until_completion(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    enable(app)
    sealed_queue(app)
    entered, release = threading.Event(), threading.Event()
    original_complete = clients.memory.complete
    def delayed(messages):
        entered.set()
        assert release.wait(5)
        return original_complete(messages)
    monkeypatch.setattr(clients.memory, "complete", delayed)
    original_start = threading.Thread.start
    def start_then_fail(thread):
        original_start(thread)
        if thread.name == "aria-offline-review":
            raise RuntimeError("injected post-start exception")
    monkeypatch.setattr(threading.Thread, "start", start_then_fail)
    assert not app.tick_auto_review()["started"]
    worker = app._review_thread
    try:
        assert entered.wait(2) and worker is not None and worker.is_alive()
        assert app.state()["reviews"][0]["status"] == "pending"
        assert app.state()["auto_review"]["today_started"] == 1
    finally:
        release.set()
        wait_worker(app, worker)
    assert not app.tick_auto_review()["started"]
    app.close()
    assert app.care._closed
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
