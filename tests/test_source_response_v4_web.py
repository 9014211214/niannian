"""Actual durable transport for the explicit candidate; inert model fixture."""
from copy import deepcopy
import json

import pytest

from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.fictional_context import bind_fictional_scenario_date, KEY as DATE_KEY
from tests._model_http_stub import stub_model_http
from tests.test_source_response_web import TEXT, BODY, response, no_model_network  # noqa: F401
from tests.test_safety_response_runtime import options


def opts():
    values = options(version="conditional_response_v4")
    values.update(safety_delivery_version="source_response_v4", premodel_safety_version="task_response_v3")
    return values


def test_actual_saved_prelude_date_and_body_survive_checkpoint_without_binding_persistence(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path, **opts())
    frames, requests = [], []
    def receive(request, **kwargs):
        requests.append(request)
        saved = json.loads(app.run_path.read_text(encoding="utf8"))["turns"][-1]
        assert saved["protection_prelude"]["status"] == "transport_returned"
        assert saved["status"] == "pending" and len(frames) == 1
        assert "虚构场景日期为 2032-02-29" in json.loads(request.data)["messages"][0]["content"]
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        sid = app.start()["state"]["session_id"]
        bind_fictional_scenario_date(app._sessions["fictional-chen"][1], "2032-02-29")
        result = app.submit(TEXT, "v4-1", sid, on_safety_prelude=lambda frame: frames.append(deepcopy(frame)) or True)
        row = result["state"]["turns"][-1]
        assert result["ok"] and row["response"] == BODY
        assert row["protection_prelude"]["protection"] == frames[0]["state"]["turns"][-1]["protection_prelude"]["protection"]
        assert row["protection_prelude"]["premodel_safety_version"] == "task_response_v3"
        assert "还不能确认" not in row["protection_prelude"]["protection"]
        assert len(app.call_guard.snapshot()["attempts"]) == len(requests) == 1
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.premodel_safety_version == "task_response_v3"
        assert DATE_KEY not in saved.model_dump_json()
        assert "_source_response_for_turn" not in saved.model_dump_json()
        assert app.submit(TEXT, "v4-1", sid, on_safety_prelude=lambda frame: pytest.fail("duplicate"))["duplicate"]
        assert len(requests) == 1
    finally:
        app._sessions.clear()
        app.close()
    resumed = MinimalWebApp(tmp_path, **opts())
    try:
        assert resumed.start()["state"]["session_id"] == sid
        assert DATE_KEY not in resumed._sessions["fictional-chen"][1].working_memory
    finally:
        resumed.close()


@pytest.mark.parametrize("mutate", ["text", "hash", "identity", "version", "disk"])
def test_prelude_drift_still_cannot_enter_model(tmp_path, mutate):
    app = MinimalWebApp(tmp_path, **opts())
    try:
        sid = app.start()["state"]["session_id"]
        def emit(frame):
            row = app._run["turns"][-1]
            if mutate == "text": row["protection_prelude"]["protection"] = "changed"
            elif mutate == "hash": row["protection_prelude"]["protection_sha256"] = "changed"
            elif mutate == "identity": row["turn_id"] = "changed"
            elif mutate == "version": app.premodel_safety_version = "task_response_v2"
            else: app.run_path.write_bytes(b"{}")
            return True
        result = app.submit(TEXT, "mutation", sid, on_safety_prelude=emit)
        assert not result["ok"] and result["state"]["turns"][-1]["protection_prelude"]["protection"]
        assert not app.call_guard.snapshot()["attempts"]
    finally:
        app.premodel_safety_version = "task_response_v3"
        app.close()
