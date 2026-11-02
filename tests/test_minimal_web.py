"""Local Engine/HTTP checks; recording clients are software fixtures, not models."""
from __future__ import annotations
from tests._model_http_stub import stub_model_http

from copy import deepcopy
import http.client
import json
from pathlib import Path
import threading
import urllib.request

import pytest

import aria.runtime as runtime
from aria.engine.orchestrator import Engine
from aria.runtime import MinimalModelClients
from aria.voice.minimal import MinimalWebApp, make_minimal_server
from tests.test_reflection_evidence_review import change, context, review


STORY = (
    (
        "这纸风车翅膀折反了，怪不得不转。",
        "囡囡她妈小时候，什么坏了都拿来找我。",
        "现在她老说别折腾，买一个就行。",
        "她对我好，我知道。可有些事我自己愿意做。",
        "今天先聊到这儿，这事先别跟她讲。",
    ),
    (
        "社区周末也有手作，女儿给我报了。我还没说想去呢。",
        "不是不喜欢别人教，上回竹编老师讲的我就爱听。",
        "我是想，写我名字以前，总得问问我。",
        "去一次可以，就怕答应了以后回回都得去。",
        "今天我只是想坐会儿，不用再帮我找活动。",
    ),
    (
        "周六我想和囡囡试试那个风车，不一定去上课。",
        "明天下午三点半提醒我拿纸和本子，先提醒一次。",
        "过半小时我还没确认，就告诉女儿提醒没确认，别发咱们聊天。",
        "我改主意了，取消这个提醒和后续通知。",
        "现在是什么状态？告诉她了吗？",
    ),
)


def forbid_external(*args, **kwargs):
    raise AssertionError("unexpected external service or configuration lookup")


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    stub_model_http(monkeypatch, forbid_external)


class RecordingClient:
    source_id = "local_test_fixture"
    model = "local-fixture"

    def __init__(self, role):
        self.role = role
        self.calls = []
        self.failure = None
        self.payload = None

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        if self.failure:
            raise self.failure
        if self.payload is not None:
            return json.dumps(self.payload, ensure_ascii=False)
        if self.role == "memory":
            quote = messages[1]["content"].split("用户:", 1)[1].split("\n", 1)[0]
            return json.dumps({"episodes": [{
                "summary": "软件夹具事实", "evidence_quote": quote,
                "turn_index": 0, "keywords": ["风车"], "referenceable": True,
            }], "relationships": [], "cognitive_privacy": []}, ensure_ascii=False)
        if self.role == "reflection":
            return '{"protocol_version":"2.0","reviews":[],"changes":[]}'
        return "本地测试替身的输出，仅用于验证软件流程。"


def live_fixture(tmp_path, **options):
    clients = MinimalModelClients(*(RecordingClient(role) for role in (
        "online", "memory", "reflection")), model_calls_enabled=True)
    return MinimalWebApp(tmp_path, clients=clients, **options), clients


def start_and_submit(app, words=STORY[0]):
    started = app.start()
    assert started["ok"]
    session_id = started["state"]["session_id"]
    for index, text in enumerate(words):
        assert app.submit(text, f"turn-{session_id}-{index}", session_id)["ok"]
    return session_id


def snapshot_files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_default_preview_ignores_all_service_keys_and_factories(tmp_path, monkeypatch):
    for name in ("OPENROUTER_API_KEY", "ARIA_LLM_API_KEY", "DASHSCOPE_API_KEY",
                 "ARIA_EMBED_API_KEY", "ARIA_VISUAL_API_KEY", "VOLC_TTS_API_KEY"):
        monkeypatch.setenv(name, "inert-test-key")
    monkeypatch.setenv("ARIA_OMNI_REALTIME_PROVIDER", "bailian")
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_URL", "https://invalid.example/never-used")
    for name in ("openrouter_api_key", "OpenRouterLLMClient"):
        monkeypatch.setattr(runtime, name, forbid_external)
    app = MinimalWebApp(tmp_path)
    session_id = start_and_submit(app)
    assert app.seal(session_id)["ok"]
    assert not app.reflect("disabled-review", wait=True)["ok"]
    state = app.state()
    assert state["mode"] == "local_preview" and not state["model_calls_enabled"]
    assert not any(state["services"].values())
    assert state["profile"]["facts"] == state["profile"]["hypotheses"] == []
    assert all(not hasattr(app, name) for name in ("asr", "tts", "emotion", "visual", "realtime_gateway"))
    app.close()


def test_three_sessions_use_actual_engine_only_disclosed_turns_and_explicit_reflection(tmp_path):
    app, clients = live_fixture(tmp_path)
    all_words = [text for session in STORY for text in session]
    consumed = 0
    ids = []
    for number, words in enumerate(STORY):
        result = app.start()
        session_id = result["state"]["session_id"]
        ids.append(session_id)
        assert isinstance(app._sessions["fictional-chen"][0], Engine)
        for index, text in enumerate(words):
            result = app.submit(text, f"session-{number}-turn-{index}", session_id)
            assert result["ok"]
            consumed += 1
            assert len(clients.online.calls) == consumed
            assert len(clients.memory.calls) == len(clients.reflection.calls) == number
            prompt = str(clients.online.calls[-1])
            assert text in prompt
            assert all(future not in prompt for future in all_words[consumed:])
        assert app.seal(session_id)["ok"]
        assert len(clients.memory.calls) == len(clients.reflection.calls) == number
        result = app.reflect(f"review-{number}", wait=True)
        assert result["ok"], result
        assert len(clients.memory.calls) == len(clients.reflection.calls) == number + 1
        for client in (clients.memory, clients.reflection):
            prompt = str(client.calls[-1])
            assert all(word in prompt for word in words)
            assert all(future not in prompt for future in all_words[consumed:])
        assert all(row["source_session_id"] in ids for row in result["state"]["profile"]["facts"])
    state = app.state()
    assert len(set(ids)) == len(state["sessions"]) == 3
    assert len(state["turns"]) == 15
    assert len(state["profile"]["facts"]) == 3
    assert not state["profile"]["hypotheses"]
    persona = app._profile()
    assert persona.person_model_state.sessions_completed == 3
    assert persona.person_model_state.sessions_observed == 3
    assert not persona.reminder_rules and not persona.notification_receipts
    app.close()


def test_state_and_restarted_replay_do_not_mutate_or_call_clients(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app)
    app.seal(session_id)
    app.reflect("review", wait=True)
    before = snapshot_files(tmp_path)
    counts = [len(client.calls) for client in (clients.online, clients.memory, clients.reflection)]
    first = app.state()
    first["turns"][0]["text"] = "browser cannot mutate saved transcript"
    restarted = MinimalWebApp(tmp_path, clients=clients)
    for _ in range(3):
        state = restarted.state()
        assert state["turns"][0]["text"] == STORY[0][0]
        assert not restarted._sessions
    assert snapshot_files(tmp_path) == before
    assert [len(client.calls) for client in (clients.online, clients.memory, clients.reflection)] == counts


@pytest.mark.parametrize("applicability", ["within_scope", "outside_scope"])
def test_real_updater_candidate_correction_and_nonadoption_are_visible_without_promotion(tmp_path, applicability):
    """Unknown positive observations defer; grounded in-scope corrections apply."""
    from tests.test_reflection_evidence_review import support
    app, clients = live_fixture(tmp_path)
    first_words = [turn.user for turn in context().turns]
    first_id = start_and_submit(app, first_words)
    current = app._sessions["fictional-chen"][1]
    item = review(current)
    clients.reflection.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}
    app.seal(first_id)
    first = app.reflect("form-candidate", wait=True)
    assert first["ok"]
    assert first["state"]["profile"]["hypotheses"] == []
    first_fact = first["state"]["profile"]["facts"][0]
    assert first_fact["source_session_id"] == first_id
    assert first_fact["source_turn_index"] == 0 and first_fact["source_quote"] == first_words[0]
    first_review = first["state"]["profile"]["evidence"][0]
    assert first_review["session_id"] == first_id
    assert first_review["disposition"] == "event_identity_unknown_no_update"
    assert {row["turn_index"]: row["quote"] for row in first_review["evidence"]} == {
        i: first_words[i] for i in (1, 2)}
    assert not any(row.get("category") == "hypotheses" and row["kind"] == "added"
                   for row in first["state"]["profile"]["changes"])
    assert app.store.load("fictional-chen").person_hypotheses == []
    first_context = current

    second_words = ["今天我来纠正上次说的办法。", "现在我改为喜欢先看完整步骤，不想等每一步。",
                    "这只是更喜欢的办法，没有授权替我转发。"]
    second_id = start_and_submit(app, second_words)
    engine, current = app._sessions["fictional-chen"]
    # This is previously accepted evidence inserted only as a software fixture.
    # It invokes no updater/model and cannot be attributed to the first review.
    prior_call_counts = (len(clients.memory.calls), len(clients.reflection.calls))
    hypothesis = support(engine.hypothesis_updater, clients.reflection, engine.persona, ctx=first_context)
    app.store.save(engine.persona)
    assert (len(clients.memory.calls), len(clients.reflection.calls)) == prior_call_counts == (1, 1)
    before = app.state()["profile"]["hypotheses"][0]
    assert before["status"] == "candidate"
    assert before["scope"] == "不赶时间的新操作教学"
    assert all(evidence["session_id"] == first_id for evidence in before["evidence"])
    quoted_turns = {row["turn_index"]: row["quote"]
                    for evidence in before["evidence"] for row in evidence["quotes"]}
    assert quoted_turns == {i: first_words[i] for i in (1, 2)}
    baseline_evidence = deepcopy(hypothesis.model_dump())
    item = review(current, hypothesis, relation="counter", applicability=applicability)
    item["event_evidence"] = deepcopy(item["evidence"])
    clients.reflection.payload = {"protocol_version": "2.0", "reviews": [item],
                                  "changes": [change(item, hypothesis)]}
    app.seal(second_id)
    corrected = app.reflect("review-correction", wait=True)
    assert corrected["ok"]
    profile = corrected["state"]["profile"]
    after = profile["hypotheses"][0]
    assert after["status"] == "candidate" and after["scope"] == before["scope"]
    if applicability == "within_scope":
        assert after["confidence"] < before["confidence"]
        assert any(row.get("category") == "hypotheses" and row["kind"] == "revised"
                   for row in corrected["state"]["reviews"][-1]["changes"])
        assert profile["evidence"][-1]["disposition"] == "applied"
    else:
        assert after == before
        assert not any(row.get("category") == "hypotheses" and row["kind"] == "revised"
                       for row in corrected["state"]["reviews"][-1]["changes"])
        assert profile["evidence"][-1]["disposition"] == "outside_scope_no_update"
    assert profile["evidence"][-1]["session_id"] == second_id
    assert profile["evidence"][-1]["evidence"][0]["quote"] == second_words[1]
    saved = app.store.load("fictional-chen").person_hypotheses[0]
    assert saved.confidence < baseline_evidence["confidence"] if applicability == "within_scope" else saved.model_dump() == baseline_evidence

    third_id = app.start()["state"]["session_id"]
    app._sessions["fictional-chen"][1].working_memory["ui_future_suggestions"] = [
        "未发送的未来台词：今晚才谈旧本子"]
    assert app.submit("这次我想先看看新的折法。", "third-conversation", third_id)["ok"]
    prompt = str(clients.online.calls[-1])
    assert after["claim"] not in prompt  # Candidate evidence has not earned online use.
    assert "未发送的未来台词：今晚才谈旧本子" not in prompt
    assert app.state()["profile"]["hypotheses"][0]["status"] == "candidate"
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 2
    app.close()


def test_restart_resumes_exact_active_session_without_running_a_model(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app, STORY[0][:2])
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert restarted.state()["session_id"] == session_id
    assert restarted.start()["state"]["session_id"] == session_id
    assert len(clients.online.calls) == 2
    assert restarted.submit(STORY[0][2], "after-restart", session_id)["ok"]
    assert STORY[0][0] in str(clients.online.calls[-1])
    restarted.close()


def test_restart_can_seal_saved_checkpoint_before_start_and_open_fresh_session(tmp_path):
    app, clients = live_fixture(tmp_path)
    old_id = start_and_submit(app, STORY[0][:1])
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert not restarted._sessions
    sealed = restarted.seal(old_id)
    assert sealed["ok"]
    assert sealed["state"]["offline"]["pending_count"] == 1
    assert sealed["state"]["offline"]["jobs"][0]["session_id"] == old_id
    new_id = restarted.start()["state"]["session_id"]
    assert new_id != old_id
    assert restarted.submit(STORY[1][0], "fresh-session", new_id)["ok"]
    assert len(clients.online.calls) == 2
    assert not clients.memory.calls and not clients.reflection.calls
    restarted.close()


def test_idle_empty_session_has_no_active_session_and_never_learns(tmp_path):
    app, clients = live_fixture(tmp_path, session_idle_seconds=2)
    old_id = app.start()["state"]["session_id"]
    app.finalize_idle_sessions(now=app._session_activity["fictional-chen"] + 3)
    assert app.state()["session_id"] is None
    assert app.state()["offline"]["pending_count"] == 0
    assert app.start()["state"]["session_id"] != old_id
    assert not clients.online.calls and not clients.memory.calls and not clients.reflection.calls
    app.close()


@pytest.mark.parametrize("closing", ["end", "idle", "close"])
def test_passive_closure_only_seals_without_offline_calls(tmp_path, closing):
    app, clients = live_fixture(tmp_path, session_idle_seconds=2)
    session_id = start_and_submit(app)
    if closing == "idle":
        app.finalize_idle_sessions(now=app._session_activity["fictional-chen"] + 3)
    else:
        getattr(app, closing)()
    assert not clients.memory.calls and not clients.reflection.calls
    state = app.state()
    assert state["session_id"] is None
    assert state["offline"]["jobs"][0]["session_id"] == session_id
    assert state["offline"]["jobs"][0]["attempts"] == 0


def test_duplicate_turn_and_conflicting_reuse_never_generate_twice(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = app.start()["state"]["session_id"]
    assert app.submit(STORY[0][0], "one-id", session_id)["ok"]
    assert app.submit(STORY[0][0], "one-id", session_id)["duplicate"]
    assert not app.submit(STORY[0][1], "one-id", session_id)["ok"]
    app.seal(session_id)
    assert app.submit(STORY[0][0], "one-id", session_id)["duplicate"]
    assert len(clients.online.calls) == 1


def test_stale_session_cannot_submit_or_seal_new_session(tmp_path):
    app, clients = live_fixture(tmp_path)
    old_id = start_and_submit(app, STORY[0][:1])
    app.seal(old_id)
    new_id = app.start()["state"]["session_id"]
    assert old_id != new_id
    assert not app.submit(STORY[1][0], "stale", old_id)["ok"]
    assert not app.seal(old_id)["ok"]
    assert app.state()["session_id"] == new_id
    assert len(clients.online.calls) == 1
    app.close()


def test_model_failure_is_failed_with_no_successful_story_reply(tmp_path):
    app, clients = live_fixture(tmp_path)
    clients.online.failure = RuntimeError("local fixture failure")
    session_id = app.start()["state"]["session_id"]
    result = app.submit(STORY[0][0], "failed", session_id)
    row = result["state"]["turns"][0]
    assert not result["ok"] and row["status"] == "failed"
    assert row["response"] == "" and row["source"] == "model_failed"
    assert "模型" in row["error"]
    assert app.submit(STORY[0][0], "failed", session_id)["duplicate"]
    assert row["requires_review"]
    assert result["state"]["requires_review"]
    assert not app.submit(STORY[0][0], "new-failed-id", session_id)["ok"]
    assert len(clients.online.calls) == 1


def test_pending_crash_record_blocks_new_id_after_restart(tmp_path):
    app, clients = live_fixture(tmp_path)
    clients.online.failure = KeyboardInterrupt("simulated process termination")
    session_id = app.start()["state"]["session_id"]
    with pytest.raises(KeyboardInterrupt):
        app.submit(STORY[0][0], "interrupted", session_id)
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert restarted.state()["turns"][0]["status"] == "pending"
    assert not restarted.start()["ok"]
    assert not restarted.submit(STORY[0][0], "new-id", session_id)["ok"]
    assert not restarted.submit(STORY[0][0], "interrupted", session_id)["ok"]
    assert len(clients.online.calls) == 1


@pytest.mark.parametrize("failed_save", [1, 2])
def test_run_save_failure_prevents_automatic_reissue(tmp_path, monkeypatch, failed_save):
    app, clients = live_fixture(tmp_path)
    session_id = app.start()["state"]["session_id"]
    original = Path.replace
    saves = 0

    def fail_once(path, target):
        nonlocal saves
        if path == app.run_path.with_suffix(".json.tmp"):
            saves += 1
            if saves == failed_save:
                raise OSError("fixture disk failure")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_once)
        with pytest.raises(RuntimeError, match="minimal_run_store_failed"):
            app.submit(STORY[0][0], "save-failure", session_id)
    expected_calls = 0 if failed_save == 1 else 1
    assert len(clients.online.calls) == expected_calls
    assert app.state()["storage_failed"]
    assert not app.submit(STORY[0][0], "new-id", session_id)["ok"]
    if failed_save == 2:
        restarted = MinimalWebApp(tmp_path, clients=clients)
        assert not restarted.start()["ok"]
        assert not restarted.submit(STORY[0][0], "newer-id", session_id)["ok"]
    assert len(clients.online.calls) == expected_calls


def test_profile_save_after_generation_failure_keeps_pending_and_blocks_retry(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    session_id = app.start()["state"]["session_id"]
    with monkeypatch.context() as patch:
        patch.setattr(app.store, "save", lambda *args: (_ for _ in ()).throw(OSError("fixture save failure")))
        result = app.submit(STORY[0][0], "profile-save-failure", session_id)
    assert not result["ok"] and result["state"]["turns"][0]["status"] == "pending"
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert not restarted.submit(STORY[0][0], "new-attempt", session_id)["ok"]
    assert len(clients.online.calls) == 1


def test_close_propagates_seal_save_failure_and_retains_ram_and_last_checkpoint(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app, STORY[0][:1])
    pair = app._sessions["fictional-chen"]
    saved = snapshot_files(tmp_path)

    def fail_save(*args, **kwargs):
        raise OSError("fixture shutdown save failure")

    with monkeypatch.context() as patch:
        patch.setattr(app.store, "save", fail_save)
        with pytest.raises(RuntimeError, match="尚未安全封存"):
            app.close()
    state = app.state()
    assert state["storage_failed"] and state["session_id"] == session_id
    assert app._sessions["fictional-chen"] is pair
    assert pair[1].turns[0].user == STORY[0][0]
    assert state["turns"][0]["text"] == STORY[0][0]
    assert snapshot_files(tmp_path) == saved
    assert not app.submit(STORY[0][1], "after-failed-close", session_id)["ok"]
    assert len(clients.online.calls) == 1
    assert not clients.memory.calls and not clients.reflection.calls


def test_server_close_closes_socket_even_when_durable_seal_fails(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app, STORY[0][:1])
    server = make_minimal_server(app, port=0)

    def fail_save(*args, **kwargs):
        raise OSError("fixture server close save failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(app.store, "save", fail_save)
            with pytest.raises(RuntimeError, match="尚未安全封存"):
                server.server_close()
        assert server.socket.fileno() == -1
        server._reaper_thread.join(timeout=3)
        assert not server._reaper_thread.is_alive()
        state = app.state()
        assert state["storage_failed"] and state["session_id"] == session_id
        assert app._sessions["fictional-chen"][1].turns[0].user == STORY[0][0]
        assert not clients.memory.calls and not clients.reflection.calls
    finally:
        server._reaper_stop.set()
        server.socket.close()
        server._reaper_thread.join(timeout=3)


def test_failed_reflection_is_not_retried_by_another_button_id_or_close(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app)
    app.seal(session_id)
    clients.reflection.failure = RuntimeError("fixture reflection failure")
    result = app.reflect("failed-review", wait=True)
    assert not result["ok"]
    assert result["state"]["reviews"][0]["status"] == "failed"
    assert app.reflect("failed-review", wait=True)["duplicate"]
    assert not app.reflect("new-review-id", wait=True)["ok"]
    app.close()
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert not restarted.reflect("after-restart", wait=True)["ok"]
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1


def test_successful_review_duplicate_is_read_only(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = start_and_submit(app)
    app.seal(session_id)
    assert app.reflect("once", wait=True)["ok"]
    before = snapshot_files(tmp_path)
    assert app.reflect("once", wait=True)["duplicate"]
    assert not app.reflect("nothing-left", wait=True)["ok"]
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    assert snapshot_files(tmp_path) == before


@pytest.fixture
def local_server(tmp_path):
    app = MinimalWebApp(tmp_path)
    server = make_minimal_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, server.server_address[1]
    server.shutdown()
    thread.join(timeout=3)
    server.server_close()


def request(port, route, *, method="GET", token=None, host=None, origin=None,
            data=None, content_length=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    headers = {}
    if token is not None:
        headers["X-Aria-Demo-Token"] = token
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    body = (data if isinstance(data, str) else json.dumps(data)) if data is not None else None
    conn.request(method, route, body=body, headers=headers)
    response = conn.getresponse()
    result = response.status, response.read()
    conn.close()
    return result


def test_http_requires_token_and_same_origin_and_rejects_untrusted_host(local_server):
    app, port = local_server
    assert request(port, "/api/minimal/state")[0] == 403
    assert request(port, "/api/minimal/state", token=app.demo_token)[0] == 200
    assert request(port, "/api/minimal/state", token=app.demo_token,
                   host="evil.invalid")[0] == 421
    assert request(port, "/api/minimal/start", method="POST", token=app.demo_token,
                   origin="https://evil.invalid", data="")[0] == 403
    assert app.state()["sessions"] == []


def test_http_does_not_expose_legacy_media_simulation_or_care_endpoints(local_server):
    app, port = local_server
    for route in ("/api/text", "/api/turn", "/api/tts", "/api/realtime/offer",
                  "/api/emotion-simulation", "/api/care/contact", "/api/end"):
        assert request(port, route, method="POST", token=app.demo_token, data="")[0] == 404
    for route in ("/api/me", "/api/care", "/api/realtime/session"):
        assert request(port, route, token=app.demo_token)[0] == 404
    assert app.state()["sessions"] == []


def test_http_rejects_bad_bodies_and_allows_only_explicit_start(local_server):
    app, port = local_server
    status, payload = request(port, "/api/minimal/start", method="POST", token=app.demo_token, data={})
    assert status == 200 and json.loads(payload)["state"]["session_id"]
    # Send only oversized length, proving rejection happens before body reads.
    status, _ = request(port, "/api/minimal/text", method="POST", token=app.demo_token,
                        content_length=17001)
    assert status == 400
    assert app.state()["turns"] == []



@pytest.mark.parametrize("binding", ["valid", "missing", "wrong", "legacy_indices"])
def test_checkpoint_response_provenance_restores_only_hash_bound_original_indices(tmp_path, binding):
    import hashlib
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, STORY[0][:3])
    profile = app.store.load("fictional-chen")
    checkpoint = profile.pending_session
    kept = [checkpoint.turns[0], checkpoint.turns[2]]
    for row in kept:
        row.person_response_sha256 = (None if binding == "missing" else "0" * 64
            if binding == "wrong" else hashlib.sha256(row.agent.encode()).hexdigest())
    checkpoint.turns = kept
    checkpoint.turn_indices = [0, 2] if binding != "legacy_indices" else []
    checkpoint.original_turn_count = 3
    app.store.save(profile)
    restarted = MinimalWebApp(tmp_path, clients=clients)
    assert restarted.start()["state"]["session_id"] == sid
    ctx = restarted._sessions["fictional-chen"][1]
    restored = ctx.working_memory.get("_person_response_deliveries", {})
    if binding == "valid":
        assert restored == {str(index): row.person_response_sha256 for index, row in zip([0, 2], kept)}
        assert ctx.turns[1].agent == ""
    else:
        assert restored == {}
    restarted.close()
