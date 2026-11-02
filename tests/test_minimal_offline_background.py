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
        elif payload["provider"]["only"] == ["alibaba"]:
            output = json.dumps({**DECISION, "safety_concern": "none"}, ensure_ascii=False)
        else:
            output = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
        response = Response(output, {"cost": .001, "prompt_tokens": 10, "completion_tokens": 12})
        response.data.update(model=payload["model"], provider="Alibaba" if payload["provider"]["only"] == ["alibaba"] else "Anthropic")
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
        accepted = threading.Event()
        outputs = []
        sender = threading.Thread(target=lambda: outputs.append(app.submit(
            "暂存我的输入。", "guarded-new-turn", next_sid,
            on_waiting_input=lambda frame: accepted.set() or True)))
        sender.start()
        assert accepted.wait(2)
        queued = app.state()["waiting_input"]
        assert queued["status"] == "waiting" and queued["request_sent"] is False
        duplicate = app.submit("暂存我的输入。", "guarded-new-turn", next_sid)
        assert duplicate["duplicate"] and duplicate["accepted"] and duplicate["request_sent"] is False
        assert app.care_state()["ok"]
        assert time.monotonic() - started < 1
        assert len(calls) == 1
    finally:
        state = finish(app, worker, release)
        sender.join(5)
        assert not sender.is_alive()
    assert outputs[0]["ok"], outputs
    assert calls[1]["provider"]["only"] == ["alibaba"]
    assert state["reviews"][-1]["status"] == "completed"
    assert "snapshot_status" not in state["model_usage"]
    assert not state["model_usage"]["stopped"] and len(calls) == 3
    assert not app._profile().cognitive_observations
    assert app._profile().stated_preferences[0].status == "withdrawn"
    app.close()


def release_app(tmp_path, monkeypatch, *, maximum=40, unknown_memory_fee=False, first_settled_failure=False):
    """Actual installed DeepSeek/Together stack with synthetic HTTP receipts."""
    import socket
    from aria.engine.policy.max_release import tls_dispatch
    from tests.test_deepseek_release import clients, prices
    from tests.test_installable_max_release import response, Response as WireResponse
    def forbidden(*args, **kwargs):
        raise AssertionError("network and real credentials forbidden")
    original_connect = socket.socket.connect
    def local_only(sock, address):
        if address[0] not in {"127.0.0.1", "localhost", "::1"}:
            forbidden()
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)
    selected = clients()
    calls, entered, release = [], threading.Event(), threading.Event()
    blocked = [False]
    def opened(request, **kwargs):
        payload = json.loads(request.data)
        schema = payload.get("response_format", {}).get("json_schema", {}).get("name", "")
        if "memory" in schema:
            client = selected.memory
        elif payload.get("stream"):
            client = selected.reflection
        elif "safety" in schema:
            client = selected.safety_review
        else:
            client = selected.online
        calls.append((client.role, payload))
        trace = kwargs["trace"]
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        if client.role == "memory" and blocked[0]:
            entered.set()
            assert release.wait(10), "background fixture was not released"
        data = response(client)
        if first_settled_failure and len(calls) == 1:
            data["choices"][0]["message"]["content"] = '{"response_segments":[],"safety_concern":"none"}'
        if client.role == "memory" and unknown_memory_fee:
            data["usage"].pop("cost")
        if client.role in {"online_fast", "safety_review"}:
            data.update(model=client.model, provider="Together")
            data["usage"] = {"cost": .00001, "prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60}
        return WireResponse(data, client.role == "reflection")
    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    app = MinimalWebApp(tmp_path, clients=selected, model_max_requests=maximum,
        model_stop_cost_usd="4", model_reservation_policy=prices())
    clock = [1000.]
    app.call_guard._pacing_clock = lambda: clock[0]
    app.call_guard._pacing_sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    sid = app.start()["state"]["session_id"]
    if first_settled_failure:
        failed = app.submit("格式拒绝软件输入。", "settled-before-review", sid)
        assert failed["settled_validation_failure"] and failed["continue_next_input"]
    for index in range(3):
        result = app.submit("中性软件输入。", "old-" + str(index), sid)
        assert result["ok"], result
    assert app.cognitive_action("enable")["ok"]
    seed_owner_records(app, sid)
    assert app.seal(sid)["ok"]
    blocked[0] = True
    assert app.reflect("release-background")["accepted"]
    assert entered.wait(2)
    return app, calls, release


def test_default_release_background_yields_to_saved_online_input(tmp_path, monkeypatch):
    app, calls, release = release_app(tmp_path, monkeypatch)
    worker = app._review_thread
    next_sid = app.start()["state"]["session_id"]
    accepted, outputs = threading.Event(), []
    sender = threading.Thread(target=lambda: outputs.append(app.submit(
        "普通软件输入。", "waiting-turn", next_sid,
        on_waiting_input=lambda frame: accepted.set() or True)))
    sender.start()
    try:
        assert accepted.wait(2)
        started = time.monotonic()
        state = app.state()
        assert state["waiting_input"]["status"] == "waiting"
        assert not state["waiting_input"].get("protection_prelude")
        assert not state["requires_review"]
        assert app.preference_action("withdraw", "pref_owner")["ok"]
        assert app.cognitive_action("disable", delete_records=True)["ok"]
        assert time.monotonic() - started < 1
        assert calls[-1][0] == "memory"
    finally:
        release.set()
        sender.join(10)
        worker.join(10)
    assert not sender.is_alive() and not worker.is_alive()
    assert outputs[0]["ok"], outputs
    assert "request_sent" not in outputs[0]["state"]["waiting_input"]
    tail = [role for role, _ in calls][-4:]
    assert tail == ["memory", "online_fast", "safety_review", "reflection"]
    assert app._profile().stated_preferences[0].status == "withdrawn"
    assert not app._profile().cognitive_observations
    rows = app.call_guard.snapshot()["attempts"]
    assert [row["operation_id"] for row in rows[-4:]] == [
        "release-background", "waiting-turn", "waiting-turn", "release-background"]
    assert not app.call_guard.snapshot()["stopped"]
    for row in rows:
        from pathlib import Path
        receipt = json.loads(Path(row["public_first_output"]["path"]).read_text(encoding="utf8"))
        assert receipt["request_id"] == row["request_id"] and receipt["role"] == row["role"]
        bindings = [json.loads(path.read_text(encoding="utf8")) for path in
            app.call_guard.physical_dir.glob("*/attempt-binding.json")]
        assert sum(any(item["request_id"] == row["request_id"] and item["operation_id"] == row["operation_id"]
            for item in binding["new_attempts"]) for binding in bindings) == 1
    app.close()


@pytest.mark.parametrize("change", ["seal", "forget", "input", "halt"])
def test_waiting_input_rechecks_state_and_never_sends_changed_turn(tmp_path, monkeypatch, change):
    app, calls, release = release_app(tmp_path, monkeypatch)
    worker = app._review_thread
    next_sid = app.start()["state"]["session_id"]
    accepted, outputs = threading.Event(), []
    sender = threading.Thread(target=lambda: outputs.append(app.submit(
        "普通软件输入。", "waiting-turn", next_sid,
        on_waiting_input=lambda frame: accepted.set() or True)))
    sender.start()
    try:
        assert accepted.wait(2)
        if change == "seal":
            assert app.seal(next_sid)["ok"]
        elif change == "forget":
            assert app.forget_run()["ok"]
        elif change == "input":
            with app._user_lock("fictional-chen"):
                app._run["waiting_input"]["text"] = "changed"
                app._save_run()
        else:
            app._halt_models()
    finally:
        release.set()
        sender.join(10)
        worker.join(10)
    assert not sender.is_alive() and not worker.is_alive()
    assert not outputs[0]["ok"] and outputs[0]["request_sent"] is False
    assert not any(row["operation_id"] == "waiting-turn" for row in app.call_guard.snapshot()["attempts"])
    app.close()



def test_default_http_waiting_turn_emits_protection_before_background_finishes(tmp_path, monkeypatch):
    import http.client
    from aria.voice.minimal import make_minimal_server
    app, calls, release = release_app(tmp_path, monkeypatch)
    review = app._review_thread
    sid = app.start()["state"]["session_id"]
    server = make_minimal_server(app, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        body = {"text": "我现在喘不上气。", "turn_id": "waiting-crisis", "session_id": sid}
        connection.request("POST", "/api/minimal/text-stream", body=json.dumps(body),
            headers={"X-Aria-Demo-Token": app.demo_token, "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        first = json.loads(response.readline())
        assert first["type"] == "input_waiting"
        prelude = first["state"]["waiting_input"]["protection_prelude"]
        assert prelude["protection"] and prelude["status"] == "saved"
        assert not release.is_set() and calls[-1][0] == "memory"
        assert app.state()["waiting_input"]["protection_prelude"]["status"] == "transport_returned"
        release.set()
        rest = [json.loads(line) for line in response.read().splitlines()]
        assert [frame["type"] for frame in rest] == ["result"]
        result = rest[-1]["result"]
        assert result["ok"], result
        turn = result["state"]["turns"][-1]
        assert turn["protection_prelude"]["protection"] == prelude["protection"]
        assert prelude["protection"] not in turn["response"]
        assert turn["turn_id"] == "waiting-crisis"
        assert [row["role"] for row in app.call_guard.snapshot()["attempts"]
            if row["operation_id"] == "waiting-crisis"] == ["online_fast", "safety_review"]
    finally:
        release.set()
        connection.close()
        review.join(10)
        server.shutdown()
        worker.join(5)
        server.server_close()


@pytest.mark.parametrize("stop", ["request_limit", "unknown_fee"])
def test_waiting_turn_uses_fresh_ledger_after_background_receipt(tmp_path, monkeypatch, stop):
    app, calls, release = release_app(tmp_path, monkeypatch,
        maximum=7 if stop == "request_limit" else 40, unknown_memory_fee=stop == "unknown_fee")
    worker = app._review_thread
    sid = app.start()["state"]["session_id"]
    accepted, outputs = threading.Event(), []
    sender = threading.Thread(target=lambda: outputs.append(app.submit(
        "普通软件输入。", "waiting-turn", sid,
        on_waiting_input=lambda frame: accepted.set() or True)))
    sender.start()
    try:
        assert accepted.wait(2)
        assert app.state()["model_usage"]["snapshot_status"] == "observed_before_background"
        assert not app.state()["model_usage"]["stopped"]
    finally:
        release.set()
        sender.join(10)
        worker.join(10)
    assert not sender.is_alive() and not worker.is_alive()
    assert not outputs[0]["ok"] and outputs[0]["request_sent"] is False
    state = app.call_guard.snapshot()
    assert state["stopped"] and state["calls_used"] == 7
    assert not any(row["operation_id"] == "waiting-turn" for row in state["attempts"])
    if stop == "unknown_fee":
        assert state["unknown_cost_requests"] == 1
        assert float(state["reserved_cost_usd"]) > 0
    app.close()


def test_orphan_waiting_record_is_not_automatically_dispatched_after_restart(tmp_path):
    app = MinimalWebApp(tmp_path)
    sid = app.start()["state"]["session_id"]
    app._run["waiting_input"] = {"turn_id": "queued", "session_id": sid,
        "text": "保存但尚未发送的软件输入。", "status": "waiting", "request_sent": False,
        "run_id": app._run["run_id"]}
    app._save_run()
    restarted = MinimalWebApp(tmp_path)
    assert restarted.state()["requires_review"]
    assert not restarted.start()["ok"]
    assert not restarted.submit("保存但尚未发送的软件输入。", "queued", sid)["ok"]
    assert restarted._run["turns"] == []
    restarted.close()



def test_settled_history_does_not_block_state_start_or_immediate_waiting_protection(tmp_path, monkeypatch):
    app, calls, release = release_app(tmp_path, monkeypatch, first_settled_failure=True)
    worker = app._review_thread
    sender = None
    try:
        started = time.monotonic()
        state = app.state()
        assert state["settled_validation_failure_ids"] == ["settled-before-review"]
        assert not state["requires_review"]
        sid = app.start()["state"]["session_id"]
        assert time.monotonic() - started < .5
        frames, results, accepted = [], [], threading.Event()
        def received(frame):
            frames.append(frame)
            accepted.set()
            return True
        sender = threading.Thread(target=lambda: results.append(app.submit(
            "我现在喘不上气。", "urgent-with-history", sid, on_waiting_input=received,
            on_safety_prelude=lambda frame: True)))
        sender.start()
        assert accepted.wait(.5), "historical validation waited for the physical HTTP call"
        assert frames[0]["state"]["waiting_input"]["protection_prelude"]["protection"]
        assert calls[-1][0] == "memory" and not release.is_set()
        assert app.preference_action("withdraw", "pref_owner")["ok"]
        assert app.cognitive_action("disable", delete_records=True)["ok"]
    finally:
        release.set()
        if sender:
            sender.join(10)
        worker.join(10)
    assert sender is not None and not sender.is_alive() and not worker.is_alive()
    assert results[0]["ok"], results
    assert app._profile().stated_preferences[0].status == "withdrawn"
    app.close()


@pytest.mark.parametrize("damage", ["disposition", "public_output", "ledger_row", "ledger_identity", "ledger_read", "ledger_locked", "run"])
def test_settled_history_revalidated_during_background_and_bad_reads_cannot_send(tmp_path, monkeypatch, damage):
    import sqlite3
    from pathlib import Path
    app, calls, release = release_app(tmp_path, monkeypatch, first_settled_failure=True)
    worker = app._review_thread
    sid = app.start()["state"]["session_id"]
    metadata_path = next(app._settled_validation_directory().glob("*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    writer = None
    try:
        if damage == "disposition":
            metadata["business_after_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf8")
        elif damage == "public_output":
            output = Path(metadata["first_output"]["path"])
            output.write_bytes(output.read_bytes() + b" ")
        elif damage in {"ledger_row", "ledger_identity", "ledger_locked"}:
            writer = sqlite3.connect(app.call_guard.path, isolation_level=None)
            if damage == "ledger_locked":
                writer.execute("BEGIN EXCLUSIVE")
            elif damage == "ledger_identity":
                writer.execute("UPDATE reservation_identity SET ledger_id=? WHERE id=1", ("f" * 32,))
            else:
                raw = writer.execute("SELECT record FROM attempts WHERE request_id=?", (metadata["request_id"],)).fetchone()[0]
                row = json.loads(raw)
                row["usage"]["cost"] += .0001
                writer.execute("UPDATE attempts SET record=? WHERE request_id=?", (json.dumps(row), metadata["request_id"]))
        elif damage == "ledger_read":
            def unreadable():
                raise OSError("synthetic historical receipt read failure")
            monkeypatch.setattr(app.call_guard, "read_committed_attempts", unreadable)
        else:
            run = json.loads(app.run_path.read_text(encoding="utf8"))
            run["run_id"] = "changed-on-disk"
            app.run_path.write_text(json.dumps(run), encoding="utf8")
        started = time.monotonic()
        state = app.state()
        assert time.monotonic() - started < 1, "bad historical source blocked on live HTTP"
        assert state["requires_review"] and state["settled_validation_failure_ids"] == []
        # No cached set of valid IDs may exempt the failed source on another read.
        assert app.state()["requires_review"]
        rejected = app.submit("不可在历史来源失效后发送。", "blocked-after-damage", sid)
        assert not rejected["ok"] and rejected["request_sent"] is False
        assert calls[-1][0] == "memory" and not release.is_set()
        # Reducing permissions remains a bounded owner operation even when the
        # historical source is invalid; run-file corruption itself blocks save.
        if damage != "run":
            assert app.preference_action("withdraw", "pref_owner")["ok"]
    finally:
        if writer is not None:
            writer.rollback()
            writer.close()
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not any(row["turn_id"] == "blocked-after-damage" for row in app._run["turns"])
    try:
        app.close()
    except (RuntimeError, ValueError):
        assert damage in {"run", "ledger_identity"}
