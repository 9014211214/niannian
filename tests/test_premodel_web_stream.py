"""Neutral staged transport tests; scripted output is not semantic evidence."""
import http.client
import io
import json
import threading
import urllib.error

import pytest

from aria.voice.minimal import MinimalWebApp, make_minimal_server
from tests._model_http_stub import stub_model_http
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload
from tests.test_safety_response_runtime import options

TEXT = "软件记录：喘不上气。任务来源甲。"
TASK = "测试任务内容甲。"


def opts():
    value = options(version="conditional_response_v2")
    value.update(safety_delivery_version="task_context_v2", premodel_safety_version="task_response_v1")
    return value


def response(request):
    value = payload(response=TASK)
    value["task_projection"] = {"version": "task_projection_v1", "status": "in_dialogue_task",
                                "request_quote": TEXT, "response_quote": TASK}
    data = {"id": "gen-neutral-staged-fixture", "model": json.loads(request.data)["model"],
            "provider": "Alibaba", "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
            **envelope(value, "json_schema_v1")}
    class Response:
        status = 200
        headers = {}
        def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size=-1): return self.body.read(size)
    return Response()


@pytest.fixture(autouse=True)
def forbid_model_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected model transport"))


def test_prelude_saved_and_transport_returned_before_single_physical_attempt(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path, **opts())
    frames, requests = [], []
    def receive(request, **kwargs):
        requests.append(request)
        saved = json.loads(app.run_path.read_text(encoding="utf-8"))["turns"][-1]
        assert saved["status"] == "pending"
        assert saved["protection_prelude"]["status"] == "transport_returned"
        assert len(frames) == 1 and not frames[0]["state"]["turns"][-1]["response"]
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "staged-1", sid, on_safety_prelude=lambda frame: frames.append(frame) or True)
        row = result["state"]["turns"][-1]
        assert result["ok"] and row["response"].endswith(TASK)
        assert row["response"].startswith(row["protection_prelude"]["protection"])
        assert len(requests) == len(app.call_guard.snapshot()["attempts"]) == 1
        assert app.submit(TEXT, "staged-1", sid, on_safety_prelude=lambda frame: pytest.fail("duplicate callback"))["duplicate"]
        assert len(requests) == 1
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.premodel_safety_version == "task_response_v1"
        assert not saved.person_hypotheses and not saved.episodic_memory and not saved.risk_events
    finally:
        app.close()


def test_missing_channel_retains_protection_without_model_or_fake_task(tmp_path):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "no-channel", sid)
        row = result["state"]["turns"][-1]
        assert result["ok"] and row["response"] and TASK not in row["response"]
        assert not app.call_guard.snapshot()["attempts"]
        assert row["premodel_task"]["client_entered"] is False
    finally:
        app.close()


@pytest.mark.parametrize("ack", [False, None, 1, "true"])
def test_non_true_transport_return_cannot_start_model_and_keeps_saved_protection(tmp_path, ack):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "bad-channel", sid, on_safety_prelude=lambda frame: ack)
        row = result["state"]["turns"][-1]
        assert not result["ok"] and row["requires_review"]
        assert row["protection_prelude"]["protection"]
        assert not app.call_guard.snapshot()["attempts"]
        assert not app.submit(TEXT, "fresh-id", sid, on_safety_prelude=lambda frame: True)["ok"]
    finally:
        app.close()


def test_callback_exception_keeps_partial_protection_and_restart_never_resumes(tmp_path):
    app = MinimalWebApp(tmp_path, **opts())
    sid = app.start()["state"]["session_id"]
    def disconnected(frame):
        raise BrokenPipeError("neutral disconnected transport")
    try:
        result = app.submit(TEXT, "broken-channel", sid, on_safety_prelude=disconnected)
        assert not result["ok"] and not app.call_guard.snapshot()["attempts"]
        protection = result["state"]["turns"][-1]["protection_prelude"]["protection"]
    finally:
        app.close()
    resumed = MinimalWebApp(tmp_path, **opts())
    try:
        before = resumed.run_path.read_bytes()
        state = resumed.state()
        assert state["turns"][-1]["protection_prelude"]["protection"] == protection
        assert before == resumed.run_path.read_bytes()
        assert not resumed.submit(TEXT, "fresh-id", sid, on_safety_prelude=lambda frame: True)["ok"]
        assert not resumed.call_guard.snapshot()["attempts"]
    finally:
        resumed.close()


@pytest.mark.parametrize("failure_stage", ["before_transport", "after_transport"])
def test_prelude_save_failure_never_starts_model(tmp_path, monkeypatch, failure_stage):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        real_save = app._save_run
        frames = []
        def save():
            row = app._run["turns"][-1] if app._run["turns"] else {}
            phase = row.get("protection_prelude", {}).get("status")
            if phase == ("saved" if failure_stage == "before_transport" else "transport_returned"):
                raise OSError("neutral save failure")
            return real_save()
        monkeypatch.setattr(app, "_save_run", save)
        result = app.submit(TEXT, "save-fail", sid, on_safety_prelude=lambda frame: frames.append(frame) or True)
        assert not result["ok"] and not app.call_guard.snapshot()["attempts"]
        assert len(frames) == (0 if failure_stage == "before_transport" else 1)
        assert result["state"]["turns"][-1]["protection_prelude"]["protection"]
    finally:
        app.close()


def test_unknown_429_retains_partial_protection_without_retry(tmp_path, monkeypatch):
    requests = []
    def limited(request, **kwargs):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, "fixture", {}, io.BytesIO(b'{"error":{"code":429}}'))
    stub_model_http(monkeypatch, limited)
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "limited-1", sid, on_safety_prelude=lambda frame: True)
        row = result["state"]["turns"][-1]
        assert not result["ok"] and row["protection_prelude"]["protection"]
        assert row["status"] == "failed" and not row["response"]
        assert len(requests) == len(app.call_guard.snapshot()["attempts"]) == 1
        assert not app.submit(TEXT, "limited-2", sid, on_safety_prelude=lambda frame: True)["ok"]
        assert len(requests) == 1
    finally:
        app.close()


def test_real_http_first_frame_arrives_while_model_is_blocked(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    requests = []
    def receive(request, **kwargs):
        requests.append(request)
        started.set()
        assert release.wait(8), "test must release its model fixture"
        return response(request)
    stub_model_http(monkeypatch, receive)
    app = MinimalWebApp(tmp_path, **opts())
    server = make_minimal_server(app, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        sid = app.start()["state"]["session_id"]
        body = json.dumps({"text": TEXT, "turn_id": "stream-1", "session_id": sid})
        connection.request("POST", "/api/minimal/text-stream", body=body,
                           headers={"Content-Type": "application/json", "X-Aria-Demo-Token": app.demo_token})
        received = connection.getresponse()
        assert received.status == 200 and "application/x-ndjson" in received.getheader("Content-Type")
        first = json.loads(received.readline())
        assert first["type"] == "safety_prelude"
        assert first["state"]["turns"][-1]["status"] == "pending"
        assert first["state"]["turns"][-1]["protection_prelude"]["protection"]
        assert started.wait(2) and not release.is_set()
        release.set()
        last = json.loads(received.readline())
        assert last["type"] == "result" and last["result"]["ok"]
        assert last["result"]["state"]["turns"][-1]["response"].endswith(TASK)
        assert received.read() == b"" and len(requests) == 1
    finally:
        release.set()
        connection.close()
        server.shutdown()
        worker.join(5)
        server.server_close()


def test_run_cannot_reopen_with_changed_premodel_version(tmp_path):
    app = MinimalWebApp(tmp_path, **opts())
    app.start()
    app.close()
    alternate = opts()
    alternate.pop("premodel_safety_version")
    with pytest.raises(ValueError, match="minimal_run_premodel_safety_version_mismatch"):
        MinimalWebApp(tmp_path, **alternate)


def test_callback_cannot_reenter_a_second_turn(tmp_path, monkeypatch):
    calls = []
    def receive(request, **kwargs):
        calls.append(request)
        return response(request)
    stub_model_http(monkeypatch, receive)
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        def callback(frame):
            second = app.submit(TEXT, "recursive-2", sid, on_safety_prelude=lambda frame: True)
            assert not second["ok"] and len(second["state"]["turns"]) == 1
            return True
        assert app.submit(TEXT, "recursive-1", sid, on_safety_prelude=callback)["ok"]
        assert len(calls) == 1
    finally:
        app.close()


def test_callback_changing_saved_turn_identity_cannot_start_model(tmp_path):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        def changed(frame):
            app._run["turns"][-1]["turn_id"] = "different-id"
            return True
        result = app.submit(TEXT, "original-id", sid, on_safety_prelude=changed)
        assert not result["ok"] and not app.call_guard.snapshot()["attempts"]
        assert result["state"]["turns"][-1]["protection_prelude"]["protection"]
    finally:
        app.close()


@pytest.mark.parametrize("target", ["memory", "disk"])
def test_transport_cannot_change_saved_protection_and_original_is_retained(tmp_path, target):
    app = MinimalWebApp(tmp_path, **opts())
    original = []
    try:
        sid = app.start()["state"]["session_id"]
        def changed(frame):
            original.append(frame["state"]["turns"][-1]["protection_prelude"]["protection"])
            if target == "memory":
                app._run["turns"][-1]["protection_prelude"]["protection"] = "changed transport copy"
            else:
                saved = json.loads(app.run_path.read_text(encoding="utf-8"))
                saved["turns"][-1]["protection_prelude"]["protection"] = "changed durable copy"
                app.run_path.write_text(json.dumps(saved), encoding="utf-8")
            return True
        result = app.submit(TEXT, "changed-protection", sid, on_safety_prelude=changed)
        assert not result["ok"] and not app.call_guard.snapshot()["attempts"]
        assert result["state"]["turns"][-1]["protection_prelude"]["protection"] == original[0]
        assert result["state"]["turns"][-1]["requires_review"]
    finally:
        app.close()
