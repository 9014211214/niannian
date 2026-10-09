"""Deterministic scheduling/commit checks, never model quality measurements."""
from copy import deepcopy
import json
import threading
import time

import pytest

from aria.engine.model.persona import (
    CognitiveObservationRecord, PreferenceQuote, PreferenceUIConfirmation, StatedPreference,
)
from aria.engine.memory.offline_merge import merge_offline_stage
from aria.voice.minimal import MinimalWebApp
from tests._model_http_stub import stub_model_http
from tests.preference_fixtures import communication_review
from tests.test_minimal_web import live_fixture, start_and_submit
from tests.test_minimal_call_integration import app_at, DECISION, Response, responses, send_one


def block(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = client.complete

    def slow(messages):
        entered.set()
        assert release.wait(5), "test barrier never released"
        return original(messages)

    monkeypatch.setattr(client, "complete", slow)
    return entered, release


def finish(app, worker, release):
    release.set()
    worker.join(5)
    assert not worker.is_alive(), "offline worker failed to finish"
    return app.state()


def seed_owner_records(app, session_id):
    profile = app._profile()
    profile.stated_preferences = [StatedPreference(
        id="pref_owner", do="先问本人意愿", scope="安排时", quotes=[PreferenceQuote(
            session_id=session_id, turn_index=0, text="安排之前先问我。")],
        confirmation=PreferenceUIConfirmation(at="2026-10-09T01:00:00+00:00"),
        status="confirmed", semantic_review=communication_review(),
        created_at="2026-10-09T01:00:00+00:00", updated_at="2026-10-09T01:00:00+00:00")]
    profile.cognitive_observations = [CognitiveObservationRecord(
        id="obs_owner", at="2026-10-09T01:00:00+00:00", session_id=session_id,
        domain="memory", signal="owner fixture", source_turn_index=0)]
    app.store.save(profile)


def test_background_returns_and_next_chat_checkpoint_survives(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    assert app.seal(sid)["ok"]
    entered, release = block(clients.memory, monkeypatch)
    result = app.reflect("background")
    worker = app._review_thread
    try:
        assert result["ok"] and result["accepted"] and result["review_pending"]
        assert entered.wait(2)
        started = time.monotonic()
        state = app.state()
        assert state["offline_review_running"] and not state["requires_review"]
        assert app.reflect("background")["duplicate"]
        assert not app.reflect("second-button")["ok"]
        new_sid = app.start()["state"]["session_id"]
        assert new_sid != sid
        assert app.submit("今天聊新的事情。", "new-turn", new_sid)["ok"]
        assert time.monotonic() - started < 1
    finally:
        state = finish(app, worker, release)
    assert state["reviews"][-1]["status"] == "completed"
    assert state["session_id"] == new_sid and state["turns"][-1]["turn_id"] == "new-turn"
    stored = app.store.load("fictional-chen")
    assert stored.pending_session.session_id == new_sid
    assert stored.pending_session.turns[-1].user == "今天聊新的事情。"
    assert sid in stored.person_model_state.completed_session_ids
    assert len(stored.episodic_memory) == 1
    app.close()


@pytest.mark.parametrize("owner_action", ["delete", "disable"])
def test_concurrent_owner_withdrawal_and_c1_delete_win(tmp_path, monkeypatch, owner_action):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["安排之前先问我。"])
    assert app.cognitive_action("enable")["ok"]
    seed_owner_records(app, sid)
    assert app.seal(sid)["ok"]
    clients.memory.payload = {key: [] for key in (
        "episodes", "relationships", "stated_preferences", "cognitive_privacy", "cognitive_observations")}
    entered, release = block(clients.memory, monkeypatch)
    original_engine = app._engine_for_persona

    def engine_for(persona):
        engine = original_engine(persona)
        original = engine._offline_memory_stage

        def memory_stage(profile, ctx):
            proposed, status, error = original(profile, ctx)
            proposed.cognitive_observations.append(CognitiveObservationRecord(
                id="obs_stale", at="2026-10-09T01:00:00+00:00", session_id=sid,
                domain="memory", signal="late fixture", source_turn_index=0))
            return proposed, status, error

        engine._offline_memory_stage = memory_stage
        return engine

    monkeypatch.setattr(app, "_engine_for_persona", engine_for)
    assert app.reflect("withdraw-during-work")["ok"]
    worker = app._review_thread
    try:
        assert entered.wait(2)
        assert app.preference_action("withdraw", "pref_owner")["ok"]
        if owner_action == "delete":
            assert app.cognitive_action("delete", observation_id="obs_owner")["ok"]
        else:
            assert app.cognitive_action("disable", delete_records=True)["ok"]
    finally:
        state = finish(app, worker, release)
    assert state["reviews"][-1]["status"] == "completed"
    profile = app.store.load("fictional-chen")
    assert profile.stated_preferences[0].status == "withdrawn"
    assert not any(row.get("category") == "stated_preferences" for row in state["reviews"][-1]["changes"])
    assert not profile.cognitive_observations
    if owner_action == "disable":
        assert not profile.cognitive_health_consent.has_text_monitoring_consent()
    else:
        assert "obs_owner" in profile.cognitive_trend.deleted_observation_ids
    app.close()


def test_forget_invalidates_background_and_does_not_reintroduce_files(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    app.seal(sid)
    entered, release = block(clients.memory, monkeypatch)
    old_run = app.state()["run_id"]
    app.reflect("forget-during-work")
    worker = app._review_thread
    try:
        assert entered.wait(2)
        assert app.forget_run()["ok"]
        assert app.state()["run_id"] != old_run
    finally:
        state = finish(app, worker, release)
    assert not app.store.exists("fictional-chen") and not app.run_path.exists()
    assert not state["reviews"] and not state["profile"]["facts"]
    assert not clients.reflection.calls
    app.close()


def test_restart_read_is_unknown_and_never_restarts_running_review(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    app.seal(sid)
    entered, release = block(clients.memory, monkeypatch)
    app.reflect("inflight-restart")
    worker = app._review_thread
    try:
        assert entered.wait(2)
        restarted = MinimalWebApp(tmp_path, clients=clients)
        assert restarted.state()["requires_review"]
        assert not restarted.start()["ok"]
        assert not restarted.reflect("inflight-restart")["ok"]
        assert not restarted.reflect("new-after-restart")["ok"]
        assert not clients.reflection.calls
        restarted.close()
    finally:
        finish(app, worker, release)
    app.close()


def test_previously_deleted_c1_source_cannot_be_added_back_by_worker(tmp_path):
    app, _clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["这纸风车翅膀折反了。"])
    assert app.cognitive_action("enable")["ok"]
    seed_owner_records(app, sid)
    app.seal(sid)
    removed = deepcopy(app._profile().cognitive_observations[0])
    assert app.cognitive_action("delete", observation_id=removed.id)["ok"]
    before = app._profile().model_copy(deep=True)
    proposed = before.model_copy(deep=True)
    proposed.cognitive_observations.append(removed.model_copy(update={"id": "new-id-same-source"}))
    merged = merge_offline_stage(before, proposed, before, sid)
    assert not merged.cognitive_observations
    assert merged.cognitive_trend.deleted_observation_ids == before.cognitive_trend.deleted_observation_ids
    app.close()


def test_actual_product_guard_barrier_does_not_block_state_or_owner_operations(tmp_path, monkeypatch):
    app = app_at(tmp_path)
    responses(monkeypatch, *(Response(deepcopy(DECISION), {"cost": .001}) for _ in range(3)))
    sid = send_one(app)
    assert app.submit("今天来修另一个风车。", "second", sid)["ok"]
    assert app.submit("我想先把轮轴安好。", "third", sid)["ok"]
    assert app.cognitive_action("enable")["ok"]
    seed_owner_records(app, sid)
    app.seal(sid)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            output = json.dumps({key: [] for key in (
                "episodes", "relationships", "stated_preferences", "cognitive_privacy", "cognitive_observations")})
        else:
            output = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
        response = Response(output, {"cost": .001, "prompt_tokens": 10, "completion_tokens": 12})
        response.data.update(model=payload["model"], provider="Anthropic")
        return response

    stub_model_http(monkeypatch, receive)
    assert app.reflect("guarded-background")["accepted"]
    worker = app._review_thread
    try:
        assert entered.wait(2)
        started = time.monotonic()
        state = app.state()
        assert state["model_usage"]["snapshot_status"] == "observed_before_background"
        assert state["model_usage"]["offline_call_inflight"]
        assert app.preference_action("withdraw", "pref_owner")["ok"]
        assert app.cognitive_action("disable", delete_records=True)["ok"]
        next_sid = app.start()["state"]["session_id"]
        turns = deepcopy(app._run["turns"])
        rejected = app.submit("暂存我的输入。", "guarded-new-turn", next_sid)
        assert rejected["transient"] and rejected["error_code"] == "offline_call_inflight"
        assert rejected["request_sent"] is False and app._run["turns"] == turns
        assert app.care_state()["ok"] and app.seal(next_sid)["ok"]
        assert time.monotonic() - started < 1
        assert len(calls) == 1
    finally:
        state = finish(app, worker, release)
    assert state["reviews"][-1]["status"] == "completed"
    assert "snapshot_status" not in state["model_usage"]
    assert not state["model_usage"]["stopped"] and len(calls) == 2
    assert not app._profile().cognitive_observations
    assert app._profile().stated_preferences[0].status == "withdrawn"
    app.close()
