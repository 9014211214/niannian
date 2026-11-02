"""Versioned complete-response path through durable local transport fixtures."""
import io
import json
import pytest

from aria.voice.minimal import MinimalWebApp
from tests._model_http_stub import stub_model_http
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload
from tests.test_safety_response_runtime import options

TEXT = "软件前置输入：喘不上气。"
BODY = "中性完整支持甲。\n中性任务乙。"


def opts():
    values = options(version="conditional_response_v3")
    values.update(safety_delivery_version="source_response_v3", premodel_safety_version="task_response_v2")
    return values


def response(request):
    data = {"id": "gen-software-v3", "model": json.loads(request.data)["model"], "provider": "Alibaba",
            "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
            **envelope(payload(response=BODY), "json_schema_v1")}
    class Response:
        status = 200
        headers = {}
        def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size=-1): return self.body.read(size)
    return Response()


@pytest.fixture(autouse=True)
def no_model_network(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected model transport"))


def test_saved_prelude_precedes_model_and_new_body_survives_with_old_evidence(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path, **opts())
    frames, requests = [], []
    def receive(request, **kwargs):
        requests.append(request)
        saved = json.loads(app.run_path.read_text(encoding="utf8"))["turns"][-1]
        assert saved["protection_prelude"]["status"] == "transport_returned"
        assert saved["status"] == "pending" and len(frames) == 1
        assert not frames[0]["state"]["turns"][-1]["response"]
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "v3-1", sid, on_safety_prelude=lambda frame: frames.append(frame) or True)
        row = result["state"]["turns"][-1]
        assert result["ok"] and row["response"] == BODY
        assert row["protection_prelude"]["protection"] == frames[0]["state"]["turns"][-1]["protection_prelude"]["protection"]
        assert row["protection_prelude"]["premodel_safety_version"] == "task_response_v2"
        assert len(app.call_guard.snapshot()["attempts"]) == len(requests) == 1
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.premodel_safety_version == "task_response_v2"
        assert "_source_response_for_turn" not in saved.model_dump_json()
        assert app.submit(TEXT, "v3-1", sid, on_safety_prelude=lambda frame: pytest.fail("duplicate"))["duplicate"]
        assert len(requests) == 1
    finally:
        app._sessions.clear()  # preserve the checkpoint to simulate process restart
        app.close()
    resumed = MinimalWebApp(tmp_path, **opts())
    try:
        assert resumed.start()["state"]["session_id"] == sid
        assert "_source_response_for_turn" not in resumed._sessions["fictional-chen"][1].working_memory
    finally:
        resumed.close()


@pytest.mark.parametrize("mutate", ["text", "hash", "identity", "version", "disk"])
def test_v3_prelude_mutation_cannot_reach_model(tmp_path, mutate):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        def emit(frame):
            row = app._run["turns"][-1]
            if mutate == "text": row["protection_prelude"]["protection"] = "changed"
            elif mutate == "hash": row["protection_prelude"]["protection_sha256"] = "changed"
            elif mutate == "identity": row["turn_id"] = "changed"
            elif mutate == "version": app.premodel_safety_version = "task_response_v1"
            else: app.run_path.write_bytes(b"{}")
            return True
        result = app.submit(TEXT, "mutation", sid, on_safety_prelude=emit)
        assert not result["ok"] and result["state"]["turns"][-1]["protection_prelude"]["protection"]
        assert not app.call_guard.snapshot()["attempts"]
    finally:
        app.premodel_safety_version = "task_response_v2"
        app.close()


def test_v3_client_failure_retains_already_displayed_prelude_and_never_resumes(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path, **opts())
    frames, calls = [], []
    def broken(*args, **kwargs): calls.append(1); raise OSError("software transport failure")
    stub_model_http(monkeypatch, broken)
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "failure", sid, on_safety_prelude=lambda frame: frames.append(frame) or True)
        row = result["state"]["turns"][-1]
        assert not result["ok"] and row["protection_prelude"]["protection"]
        assert len(frames) == len(calls) == 1
    finally:
        app.close()
    resumed = MinimalWebApp(tmp_path, **opts())
    try:
        assert not resumed.submit(TEXT, "new-id", sid, on_safety_prelude=lambda frame: True)["ok"]
        assert len(calls) == 1
    finally:
        resumed.close()
