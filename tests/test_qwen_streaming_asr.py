"""Protocol/resource tests with in-memory WebSockets; no provider requests."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import io
import json
from pathlib import Path
import wave

import pytest

from aria.voice import asr, asr_cli


PCM = b"\x00\x01" * 16


def sentence(text="最终原文。", *, identity=1, final=True):
    return {"output": {"sentence": {"sentence_id": identity, "text": text,
            "sentence_end": final, "begin_time": 0,
            "end_time": 1 if final else None}}}


class Socket:
    def __init__(self, results=None, *, early=None, send_error=False, close_hangs=False):
        self.results = results if results is not None else [
            ("result-generated", sentence("中间修订", final=False)),
            ("result-generated", sentence()), ("task-finished", {})]
        self.early, self.send_error, self.close_hangs = early, send_error, close_hangs
        self.queue = asyncio.Queue()
        self.sent = []
        self.task_id = None
        self.closed = False
        self.aborted = False
        self.transport = self
        self.recv_active = 0

    def abort(self):
        self.aborted = True

    async def send(self, value):
        self.sent.append(value)
        if isinstance(value, bytes):
            if self.send_error:
                raise RuntimeError("Authorization: Bearer not-a-real-key")
            if self.early is not None:
                self.put(*self.early)
                self.early = None
            return
        request = json.loads(value)
        if request["header"]["action"] == "run-task":
            self.task_id = request["header"]["task_id"]
            self.put("task-started", {})
        else:
            for kind, payload in self.results:
                self.put(kind, payload)

    def put(self, kind, payload):
        if kind == "raw":
            self.queue.put_nowait(payload)
            return
        header = {"event": kind, "task_id": self.task_id}
        if kind == "task-failed":
            header.update(error_code="Throttling.RateQuota",
                          error_message="Authorization: Bearer not-a-real-key")
        self.queue.put_nowait(json.dumps({"header": header, "payload": payload}))

    async def recv(self):
        self.recv_active += 1
        try:
            return await self.queue.get()
        finally:
            self.recv_active -= 1

    async def close(self):
        self.closed = True
        if self.close_hangs:
            await asyncio.Event().wait()


@pytest.fixture
def transport(monkeypatch):
    # The optional library is installed in the development environment only.
    websockets_client = pytest.importorskip("websockets.asyncio.client")
    calls = []
    sockets = []
    factory = {"build": Socket}

    class Connection:
        def __init__(self, url, **kwargs):
            calls.append((url, kwargs))
            self.socket = factory["build"]()
            sockets.append(self.socket)
            self.process_redirect = lambda error: "must-be-overridden"

        def __await__(self):
            async def opened():
                error = RuntimeError("redirect fixture")
                assert self.process_redirect(error) is error
                return self.socket
            return opened().__await__()

    monkeypatch.setattr(websockets_client, "connect", Connection)
    return calls, sockets, factory


def recognize(**kwargs):
    config = kwargs.pop("config", asr.ASRConfig(enabled=True))
    return asyncio.run(asr.transcribe_pcm(PCM, api_key="not-a-real-key",
                                         config=config, **kwargs))


def test_full_duplex_single_task_final_only_and_close(transport):
    calls, sockets, _ = transport
    updates = []
    result = recognize(on_sentence=updates.append)
    assert result.status == "completed"
    assert result.transcript == "最终原文。"
    assert [item.final for item in updates] == [False, True]
    assert result.task_started and result.task_finished
    assert result.audio_bytes_sent == len(PCM)
    assert result.automatic_retries == 0 and result.actual_charge_cny is None
    assert len(calls) == len(sockets) == 1
    assert calls[0][0] == asr.ENDPOINT
    assert calls[0][1]["max_size"] == asr.MAX_MESSAGE_BYTES
    assert not calls[0][1]["logger"].isEnabledFor(50)
    socket = sockets[0]
    assert socket.closed and not socket.aborted and socket.recv_active == 0
    commands = [json.loads(item) for item in socket.sent if isinstance(item, str)]
    assert [item["header"]["action"] for item in commands] == ["run-task", "finish-task"]
    assert {item["header"]["task_id"] for item in commands} == {result.task_id}
    assert commands[0]["payload"]["model"] == asr.MODEL
    assert commands[0]["payload"]["parameters"] == {"format": "pcm", "sample_rate": 16000}
    assert b"".join(item for item in socket.sent if isinstance(item, bytes)) == PCM


def test_cumulative_usage_is_not_summed_or_promoted_to_final(transport):
    _, _, factory = transport
    first = sentence("中间", final=False)
    first["usage"] = {"input_tokens": 48, "output_tokens": 5}
    last = sentence()
    last["usage"] = {"input_tokens": 48, "output_tokens": 8}
    factory["build"] = lambda: Socket([
        ("result-generated", first), ("result-generated", last), ("task-finished", {})])
    result = recognize()
    assert result.final_usage is None
    assert result.last_observed_usage == {"input_tokens": 48, "output_tokens": 8}
    assert result.last_usage_event == "result-generated"
    assert result.actual_charge_cny is None


def test_task_finished_usage_keeps_its_own_provenance(transport):
    _, _, factory = transport
    factory["build"] = lambda: Socket([
        ("result-generated", sentence()),
        ("task-finished", {"usage": {"duration": 1, "input_tokens": 12, "output_tokens": 2}})])
    result = recognize()
    assert result.final_usage == result.last_observed_usage
    assert result.last_usage_event == "task-finished"


@pytest.mark.parametrize("events,code", [
    ([("result-generated", sentence("不能冒充最终", final=False)), ("task-finished", {})], "asr_final_text_missing"),
    ([("result-generated", sentence("  ")), ("task-finished", {})], "asr_final_text_missing"),
    ([("result-generated", sentence()), ("result-generated", sentence("未完", identity=2, final=False)),
      ("task-finished", {})], "asr_unfinished_sentence"),
    ([("result-generated", sentence()), ("result-generated", sentence("另一个终稿"))], "asr_final_sentence_changed"),
    ([("result-generated", sentence(identity=2)), ("result-generated", sentence(identity=1))], "asr_sentence_order_changed"),
    ([("task-started", {})], "asr_duplicate_task_start"),
    ([("result-generated", {"output": {}})], "missing_asr_sentence"),
    ([("result-generated", {**sentence(), "usage": {"input_tokens": True}})], "invalid_asr_usage"),
    ([("task-failed", {})], "asr_provider_task_failed"),
    ([("raw", json.dumps({"header": {"event": "result-generated", "task_id": "other"}, "payload": sentence()}))],
     "asr_task_identity_mismatch"),
    ([("raw", "{\"header\":{},\"header\":{},\"payload\":{}}")], "duplicate_json_key"),
    ([("raw", b"binary")], "invalid_asr_event_encoding"),
])
def test_bad_or_partial_response_never_becomes_a_transcript(transport, events, code):
    calls, sockets, factory = transport
    factory["build"] = lambda: Socket(events)
    with pytest.raises(asr.ASRError) as failure:
        recognize()
    assert failure.value.code == code
    assert failure.value.result.status == "failed"
    assert failure.value.result.transcript == ""
    assert sockets[0].closed and len(calls) == 1 and sockets[0].recv_active == 0
    assert "not-a-real-key" not in str(failure.value)
    assert "not-a-real-key" not in json.dumps(asdict(failure.value.result))


def test_exact_duplicate_final_deduplicates_without_losing_sentence_order(transport):
    _, _, factory = transport
    factory["build"] = lambda: Socket([
        ("result-generated", sentence("第一。")),
        ("result-generated", sentence("第一。")),
        ("result-generated", sentence("第二。", identity=2)), ("task-finished", {})])
    result = recognize()
    assert result.transcript == "第一。第二。"
    assert len(result.final_sentences) == 2


def test_provider_request_id_is_separate_and_heartbeat_is_not_a_sentence(transport):
    _, _, factory = transport
    final = sentence()
    final["output"]["request_id"] = "provider-request-42"
    heartbeat = {"output": {"sentence": {"heartbeat": True, "sentence_id": 0,
                                        "text": "", "sentence_end": False}}}
    factory["build"] = lambda: Socket([
        ("result-generated", final), ("result-generated", heartbeat), ("task-finished", {})])
    result = recognize()
    assert result.transcript == "最终原文。"
    assert len(result.sentences) == 1
    assert result.provider_request_ids == ["provider-request-42"]


@pytest.mark.parametrize("early,code", [
    (("task-finished", {}), "asr_premature_task_finish"),
    (("task-failed", {}), "asr_provider_task_failed"),
])
def test_early_terminal_cancels_sender_and_never_finishes_task(transport, early, code):
    calls, sockets, factory = transport
    factory["build"] = lambda: Socket(early=early)
    with pytest.raises(asr.ASRError, match=code):
        asyncio.run(asr.transcribe_pcm(b"\0\0" * 3200, api_key="fake", config=asr.ASRConfig(enabled=True)))
    assert len(calls) == 1 and sockets[0].closed
    assert len([item for item in sockets[0].sent if isinstance(item, bytes)]) == 1
    assert len([item for item in sockets[0].sent if isinstance(item, str)]) == 1


def test_send_error_surfaces_without_waiting_for_the_receiver(transport):
    calls, sockets, factory = transport
    factory["build"] = lambda: Socket(send_error=True)
    with pytest.raises(asr.ASRError, match="asr_transport_or_callback_error"):
        recognize()
    assert len(calls) == 1 and sockets[0].closed and sockets[0].recv_active == 0


def test_timeout_keeps_received_partial_and_cleans_up(transport):
    calls, sockets, factory = transport
    factory["build"] = lambda: Socket([("result-generated", sentence("只有中间", final=False))])
    with pytest.raises(asr.ASRError, match="asr_timeout") as failure:
        recognize(config=asr.ASRConfig(enabled=True, timeout_seconds=0.05))
    assert len(failure.value.result.sentences) == 1
    assert failure.value.result.transcript == ""
    assert len(calls) == 1 and (sockets[0].closed or sockets[0].aborted)
    assert sockets[0].recv_active == 0


def test_hanging_close_aborts_in_overall_timeout(transport):
    _, sockets, factory = transport
    factory["build"] = lambda: Socket(close_hangs=True)
    with pytest.raises(asr.ASRError, match="asr_close_failed"):
        recognize(config=asr.ASRConfig(enabled=True, timeout_seconds=0.05))
    assert sockets[0].closed and sockets[0].aborted


def test_external_cancellation_closes_and_cancels_pending_receive(transport):
    _, sockets, factory = transport
    factory["build"] = lambda: Socket([])
    async def cancel():
        task = asyncio.create_task(asr.transcribe_pcm(PCM, api_key="fake", config=asr.ASRConfig(enabled=True)))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel())
    assert sockets[0].closed and sockets[0].recv_active == 0


@pytest.mark.parametrize("kwargs", [
    {"enabled": "true"}, {"max_audio_seconds": 61}, {"max_audio_seconds": float("nan")},
    {"timeout_seconds": 91}, {"timeout_seconds": True}, {"timeout_seconds": 0},
])
def test_config_rejects_unbounded_or_truthy_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        asr.ASRConfig(**kwargs)


@pytest.mark.parametrize("pcm", [b"", b"1", bytearray(PCM), b"\0" * 320002],
                         ids=["empty", "odd-byte", "mutable-buffer", "too-long"])
def test_audio_invalid_before_connect(transport, pcm):
    with pytest.raises(ValueError):
        asyncio.run(asr.transcribe_pcm(pcm, api_key="fake", config=asr.ASRConfig(enabled=True)))
    assert transport[0] == []


def test_key_presence_and_legacy_env_never_enable_or_route(transport, monkeypatch):
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ARIA_ASR_BASE_URL", "wss://example.invalid/")
    monkeypatch.setenv("ALIYUN_ASR_MODEL", "old-model")
    with pytest.raises(ValueError, match="asr_calls_not_enabled"):
        asyncio.run(asr.transcribe_pcm(PCM, api_key="fake"))
    assert transport[0] == []
    result = recognize()
    assert transport[0][0][0] == asr.ENDPOINT and result.model == asr.MODEL


def wav_bytes(pcm=PCM, *, rate=16000, channels=1, width=2):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["wav", "pcm"])
def test_cli_preview_reads_audio_but_not_key_or_network(tmp_path, monkeypatch, transport, format, capsys):
    path = tmp_path / ("audio." + format)
    path.write_bytes(wav_bytes() if format == "wav" else PCM)
    monkeypatch.setattr(asr_cli, "_project_key", lambda *_: pytest.fail("key must not be read"))
    assert asr_cli.main([str(path), "--format", format]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "local_preview_no_call"
    assert transport[0] == []


@pytest.mark.parametrize("spec", [{"rate": 8000}, {"channels": 2}, {"width": 1}])
def test_wrong_wav_spec_rejected_without_resampling(tmp_path, spec):
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(**spec))
    with pytest.raises(ValueError, match="asr_wav_requires"):
        asr_cli.read_audio(path, "wav", asr.ASRConfig())


def test_truncated_wav_and_oversize_pcm_are_rejected(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes()[:-2])
    with pytest.raises(ValueError, match="asr_wav_truncated"):
        asr_cli.read_audio(path, "wav", asr.ASRConfig())
    path.write_bytes(b"\0" * 320002)
    with pytest.raises(ValueError, match="asr_audio_file_limit_exceeded"):
        asr_cli.read_audio(path, "pcm", asr.ASRConfig())


def test_cli_one_task_preserves_receipts_and_refuses_reuse(tmp_path, monkeypatch, transport, capsys):
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes())
    output = tmp_path / "actual"
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "not-a-real-key")
    args = [str(path), "--enable-asr-calls", "--output-dir", str(output), "--fee-reserve-cny", "0.10"]
    assert asr_cli.main(args) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    closed = json.loads((output / "single-task-closure.json").read_text(encoding="utf-8"))
    assert result["transcript"] == "最终原文。"
    assert closed["unknown_fee_reserve_cny"] == "0.10" and closed["actual_charge_cny"] is None
    assert closed["dialogue_submitted"] is False and closed["can_resume"] is False
    prior = {item.name: item.read_bytes() for item in output.iterdir()}
    with pytest.raises(SystemExit):
        asr_cli.main(args)
    assert prior == {item.name: item.read_bytes() for item in output.iterdir()}
    assert len(transport[0]) == 1
    assert "not-a-real-key" not in "".join(value.decode() for value in prior.values())
    assert "not-a-real-key" not in capsys.readouterr().out


def test_cli_failure_closes_and_never_assumes_zero_fee(tmp_path, monkeypatch, transport):
    transport[2]["build"] = lambda: Socket([("task-failed", {})])
    path = tmp_path / "audio.pcm"
    path.write_bytes(PCM)
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "not-a-real-key")
    output = tmp_path / "failed"
    assert asr_cli.main([str(path), "--format", "pcm", "--enable-asr-calls", "--output-dir", str(output),
                         "--fee-reserve-cny", "0.10"]) == 1
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed" and result["transcript"] == ""
    assert result["provider_error_code"] == "Throttling.RateQuota"
    assert result["actual_charge_cny"] is None
    assert json.loads((output / "single-task-closure.json").read_text(encoding="utf-8"))["can_resume"] is False


def test_key_reader_extracts_only_selected_name_without_setting_env(tmp_path, monkeypatch):
    env = tmp_path / "fixture.env"
    env.write_text("OTHER_PRIVATE_KEY=do-not-load\nALIYUN_DASHSCOPE_API_KEY='fixture-value'\n", encoding="utf-8")
    assert asr_cli._project_key(env) == "fixture-value"
    import os
    assert os.environ.get("OTHER_PRIVATE_KEY") is None
    monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "process-value")
    assert asr_cli._project_key(env) == "process-value"


def test_bridge_requires_completion_and_explicit_submission(transport):
    result = recognize()
    class TextInput:
        def __init__(self):
            self.calls = []
        def submit(self, *args):
            self.calls.append(args)
            return "text-return"
    app = TextInput()
    with pytest.raises(ValueError, match="not_confirmed"):
        asr.submit_final_transcript(result, app, session_id="session")
    assert app.calls == []
    assert asr.submit_final_transcript(result, app, session_id="session", confirmed=True) == "text-return"
    assert app.calls == [("最终原文。", "asr_" + result.task_id.replace("-", ""), "session")]
    result.sentences.append(asr.ASRSentence(3, "tampered", True))
    with pytest.raises(ValueError, match="not_locally_verified"):
        asr.submit_final_transcript(result, app, session_id="session", confirmed=True)


def test_reconstructed_receipt_is_not_provider_proof_or_bridge_capability(transport):
    result = recognize()
    forged = asr.ASRResult(result.task_id, result.model, result.audio_sha256,
        result.audio_bytes, result.audio_seconds, status="completed",
        task_started=True, task_finished=True, sentences=list(result.sentences))
    with pytest.raises(ValueError, match="not_locally_verified"):
        asr.submit_final_transcript(forged, object(), session_id="session", confirmed=True)


def test_real_text_preview_submission_is_explicit_and_durably_idempotent(transport, tmp_path):
    from aria.voice.minimal import MinimalWebApp
    result = recognize()
    app = MinimalWebApp(tmp_path / "preview")
    try:
        before = app.state()
        assert not before["session_id"]
        app.start()
        session_id = app.state()["session_id"]
        first = asr.submit_final_transcript(result, app, session_id=session_id, confirmed=True)
        assert first["ok"]
        second = asr.submit_final_transcript(result, app, session_id=session_id, confirmed=True)
        assert second["ok"] and second["duplicate"]
        assert len(app.state()["turns"]) == 1
        assert app.state()["turns"][0]["text"] == "最终原文。"
        assert not app.clients.model_calls_enabled
        wrong_session = asr.submit_final_transcript(result, app, session_id="wrong", confirmed=True)
        assert not wrong_session["ok"]
    finally:
        app.close()
