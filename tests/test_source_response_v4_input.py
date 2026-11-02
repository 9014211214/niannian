"""Actual v4 input fidelity and existing boundaries, using inert HTTP responses."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from aria.voice.minimal import MinimalWebApp
from aria.voice.web import TextWebApp
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.policy.fictional_context import bind_fictional_conversation_history
from tests._model_http_stub import stub_model_http
from tests.test_source_response_web import BODY, response, no_model_network  # noqa: F401
from tests.test_source_response_v4_web import opts
from tests.test_source_response_web import opts as v3_opts
from tests.test_safety_response_instruction import bundle


SPACES = ["  ", "\n", "\r\n", "\t", "\u00a0", "\u3000", "\u2003"]


@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("premodel", [False, True])
def test_actual_web_raw_text_wire_hash_persistence_and_duplicate_identity(tmp_path, monkeypatch, space, premodel):
    text = space + ("软件输入：喘不上气。" if premodel else "中性输入甲。") + space + "中性输入乙。" + space
    app = MinimalWebApp(tmp_path, **opts())
    requests, frames = [], []
    original_history = [{"role": "user", "content": "中性历史甲。\n\t中性历史乙。"}]
    def receive(request, **kwargs):
        wire = json.loads(request.data)
        carrier = json.loads(wire["messages"][1]["content"])
        assert carrier["current_user_message"] == text
        assert carrier["conversation_history"] == original_history
        requests.append(wire)
        row = json.loads(app.run_path.read_text(encoding="utf8"))["turns"][-1]
        assert row["text"] == text and row["status"] == "pending"
        if premodel:
            assert row["protection_prelude"]["utterance_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert row["protection_prelude"]["status"] == "transport_returned"
            assert len(frames) == 1 and frames[0]["state"]["turns"][-1]["text"] == text
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        state = app.start()["state"]
        assert state["preserve_input_text"] is True
        sid = state["session_id"]
        ctx = app._sessions["fictional-chen"][1]
        bind_fictional_conversation_history(ctx, original_history)
        result = app.submit(text, "exact-input", sid, on_safety_prelude=lambda frame: frames.append(deepcopy(frame)) or True)
        row = result["state"]["turns"][-1]
        assert result["ok"] and row["text"] == text and row["response"] == BODY
        assert ctx.turns[-1].user == text
        assert len(requests) == len(app.call_guard.snapshot()["attempts"]) == 1
        assert len(frames) == int(premodel)
        saved = app.store.load("fictional-chen")
        assert "FictionalConversationHistory" not in saved.model_dump_json()
        if not premodel:
            assert saved.pending_session.turns[-1].user == text
        assert app.submit(text, "exact-input", sid)["duplicate"]
        different = app.submit(text.strip(), "exact-input", sid)
        assert not different["ok"] and not different.get("duplicate") and len(requests) == 1
        disk = json.loads(app.run_path.read_text(encoding="utf8"))["turns"][-1]
        assert disk["text"] == text
    finally:
        app._sessions.clear()
        app.close()
    resumed = MinimalWebApp(tmp_path, **opts())
    try:
        resumed_state = resumed.start()["state"]
        assert resumed_state["turns"][-1]["text"] == text
        assert resumed.submit(text, "exact-input", sid)["duplicate"]
        assert len(requests) == 1
    finally:
        resumed.close()


@pytest.mark.parametrize("version", ["legacy", "conditional_context_v1", "task_context_v2", "source_response_v3", "source_response_v4"])
def test_text_service_explicit_version_boundary(version):
    app = object.__new__(TextWebApp)
    app.safety_delivery_version = version
    original = " \n\t中性甲\r\n\u3000中性乙\u00a0 "
    assert app._text_utterance(original) == (original if version == "source_response_v4" else "中性甲 中性乙")


@pytest.mark.parametrize("text", [None, 3, "", " \t\n\u3000", "甲" * 501, " " * 500 + "甲"])
def test_actual_v4_text_service_rejects_invalid_or_raw_overlong_input_before_session(tmp_path, monkeypatch, text):
    app = TextWebApp(store=PersonaStore(tmp_path), clients=bundle(), safety_delivery_version="source_response_v4",
                     safety_response_instruction_version="conditional_response_v4")
    monkeypatch.setattr(app, "_session", lambda *args: pytest.fail("invalid input entered session"))
    result = app.text_turn("owner", text)
    assert not result["ok"] and app._sessions == {}


def test_old_web_prelude_and_wire_still_use_original_normalization(tmp_path, monkeypatch):
    text = " \n软件输入：喘不上气。\t中性信息。\u3000"
    normalized = " ".join(text.strip().split())
    app = MinimalWebApp(tmp_path, **v3_opts())
    requests = []
    def receive(request, **kwargs):
        requests.append(request)
        carrier = json.loads(json.loads(request.data)["messages"][1]["content"])
        assert carrier["current_user_message"] == normalized
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        state = app.start()["state"]
        assert "preserve_input_text" not in state
        result = app.submit(text, "old-v3", state["session_id"], on_safety_prelude=lambda frame: True)
        assert result["ok"] and len(requests) == 1
        row = result["state"]["turns"][-1]
        assert row["text"] == text
        assert row["protection_prelude"]["utterance_sha256"] == hashlib.sha256(normalized.encode()).hexdigest()
    finally:
        app.close()


@pytest.mark.parametrize("premodel", [False, True])
def test_input_preservation_does_not_disable_actual_receipt_and_medical_guards(tmp_path, monkeypatch, premodel):
    text = " \n" + ("软件输入：喘不上气。" if premodel else "中性信息甲。") + "\t中性信息乙。\u3000"
    app = MinimalWebApp(tmp_path, **opts())
    requests = []
    def receive(request, **kwargs):
        requests.append(request)
        result = response(request)
        data = json.loads(result.body.getvalue())
        answer = json.loads(data["choices"][0]["message"]["content"])
        answer["response"] = "我已经帮你发给家人了。可以服双倍药量。"
        data["choices"][0]["message"]["content"] = json.dumps(answer, ensure_ascii=False)
        import io
        result.body = io.BytesIO(json.dumps(data).encode())
        return result
    stub_model_http(monkeypatch, receive)
    try:
        sid = app.start()["state"]["session_id"]
        engine = app._sessions["fictional-chen"][0]
        effects = []
        monkeypatch.setattr(engine._tools, "invoke", lambda *args, **kwargs: effects.append("tool"))
        result = app.submit(text, "guards", sid, on_safety_prelude=lambda frame: True)
        assert result["ok"] and len(requests) == 1 and not effects
        delivered = result["state"]["turns"][-1]["response"]
        assert "已经帮你发" not in delivered and "可以服双倍" not in delivered
    finally:
        app.close()


@pytest.mark.parametrize("mode", [True, False, None, "true"])
@pytest.mark.parametrize("raw", [" \n甲\t乙\u3000", " \n\t\u3000"])
def test_actual_composer_preserves_only_server_selected_v4_text(mode, raw):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    html = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf8")
    handler = next(line for line in html.splitlines() if "$('composer').addEventListener('submit'" in line)
    script = """
const input={value:RAW};let captured=[],handler;
const state={preserve_input_text:MODE,session_id:'s'};
const crypto={randomUUID:()=> 't'},isCurrentOpen=()=>true,viewingHistory=()=>false;
const $=id=>id==='message-input'?input:{addEventListener:(event,fn)=>{handler=fn}};
const mutate=(endpoint,body,done)=>{captured.push({endpoint,body});done()};
HANDLER
handler({preventDefault(){}});
console.log(JSON.stringify({captured,value:input.value}));
""".replace("RAW", json.dumps(raw)).replace("MODE", json.dumps(mode)).replace("HANDLER", handler)
    executed = subprocess.run([node, "-"], input=script, encoding="utf8", capture_output=True, timeout=10)
    assert executed.returncode == 0, executed.stderr
    result = json.loads(executed.stdout)
    if not raw.strip():
        assert result["captured"] == [] and result["value"] == raw
    else:
        assert result["captured"] == [{"endpoint": "text", "body": {"text": raw if mode is True else raw.strip(), "turn_id": "t", "session_id": "s"}}]
        assert result["value"] == ""
