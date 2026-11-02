"""Explicit, single-task Alibaba speech recognition; no dialogue-model changes.

The optional transport is imported only after an enabled call is validated.
Final transcripts are never submitted to an Aria conversation automatically.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
import re
import uuid
import weakref
from typing import Any, Callable


MODEL = "qwen-audio-3.1-asr-flash-streaming"
ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
SAMPLE_RATE = 16000
PCM_BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_AUDIO_SECONDS = 60
MAX_MESSAGE_BYTES = 1024 * 1024
_CHUNK_BYTES = 3200
_MAX_EVENTS = 10000
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_completed_results: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class ASRConfig:
    enabled: bool = False
    max_audio_seconds: float = 10.0
    timeout_seconds: float = 90.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("ASR enabled must be an explicit boolean")
        for name, upper in (("max_audio_seconds", MAX_AUDIO_SECONDS),
                            ("timeout_seconds", 90)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= upper):
                raise ValueError(f"invalid_asr_{name}")


@dataclass(frozen=True)
class ASRSentence:
    sentence_id: int
    text: str
    final: bool
    begin_time_ms: int | None = None
    end_time_ms: int | None = None


@dataclass(eq=False)
class ASRResult:
    task_id: str
    model: str
    audio_sha256: str
    audio_bytes: int
    audio_seconds: float
    status: str = "starting"
    task_started: bool = False
    task_finished: bool = False
    run_task_sent: bool = False
    finish_task_sent: bool = False
    audio_bytes_sent: int = 0
    events_received: int = 0
    sentences: list[ASRSentence] = field(default_factory=list)
    final_usage: dict[str, int | float] | None = None
    last_observed_usage: dict[str, int | float] | None = None
    last_usage_event: str | None = None
    error_code: str | None = None
    provider_error_code: str | None = None
    provider_request_ids: list[str] = field(default_factory=list)
    actual_charge_cny: str | None = None
    automatic_retries: int = 0
    elapsed_ms: int = 0

    @property
    def final_sentences(self) -> tuple[ASRSentence, ...]:
        # Exact duplicate final events are omitted while receiving, not joined.
        return tuple(sentence for sentence in self.sentences if sentence.final)

    @property
    def transcript(self) -> str:
        if self.status != "completed" or not self.task_finished:
            return ""
        return "".join(sentence.text for sentence in self.final_sentences)


class ASRError(RuntimeError):
    """A bounded diagnostic and partial receipt, never an auth/header dump."""

    def __init__(self, code: str, result: ASRResult):
        super().__init__(code)
        self.code = code
        self.result = result


def validate_pcm(pcm: bytes, config: ASRConfig) -> None:
    """Accept only explicitly supplied mono signed-16-bit little-endian PCM."""
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError("invalid_asr_pcm_bytes")
    if len(pcm) > int(config.max_audio_seconds * PCM_BYTES_PER_SECOND):
        raise ValueError("asr_audio_limit_exceeded")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("nonfinite_json_number")


def _parse_event(raw: Any, task_id: str) -> tuple[str, dict, dict]:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("invalid_asr_event_encoding")
    event = json.loads(raw, object_pairs_hook=_unique_object,
                       parse_constant=_invalid_constant)
    if not isinstance(event, dict):
        raise ValueError("invalid_asr_event")
    header, payload = event.get("header"), event.get("payload")
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise ValueError("invalid_asr_event_envelope")
    if header.get("task_id") != task_id:
        raise ValueError("asr_task_identity_mismatch")
    kind = header.get("event")
    if kind not in ("task-started", "result-generated", "task-finished", "task-failed"):
        raise ValueError("unknown_asr_event")
    output = payload.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("invalid_asr_output")
    # output.request_id belongs to the provider and need not equal task_id.
    return kind, header, payload


def _usage(payload: dict) -> dict[str, int | float] | None:
    if "usage" not in payload:
        return None
    raw = payload["usage"]
    if not isinstance(raw, dict):
        raise ValueError("invalid_asr_usage")
    clean = {}
    for key in ("duration", "input_tokens", "output_tokens", "total_tokens"):
        if key not in raw:
            continue
        value = raw[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "duration" and type(value) is not int)):
            raise ValueError("invalid_asr_usage")
        clean[key] = value
    return clean or None


def _sentence(payload: dict) -> ASRSentence:
    raw = payload.get("output", {}).get("sentence")
    if not isinstance(raw, dict):
        raise ValueError("missing_asr_sentence")
    identity, text, final = (raw.get("sentence_id"), raw.get("text"),
                             raw.get("sentence_end"))
    if (type(identity) is not int or identity < 0 or not isinstance(text, str)
            or type(final) is not bool):
        raise ValueError("invalid_asr_sentence")
    times = []
    for key in ("begin_time", "end_time"):
        value = raw.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid_asr_sentence_time")
        times.append(value)
    if times[0] is not None and times[1] is not None and times[1] < times[0]:
        raise ValueError("invalid_asr_sentence_time")
    return ASRSentence(identity, text, final, *times)


def _completion_binding(result: ASRResult) -> str:
    # Local capability only: JSON/reconstructed or altered results cannot submit.
    value = {"task_id": result.task_id, "model": result.model,
             "audio_sha256": result.audio_sha256, "audio_bytes": result.audio_bytes,
             "audio_seconds": result.audio_seconds, "status": result.status,
             "task_started": result.task_started, "task_finished": result.task_finished,
             "error_code": result.error_code,
             "final_sentences": [asdict(item) for item in result.final_sentences]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


async def transcribe_pcm(
    pcm: bytes, *, api_key: str, config: ASRConfig = ASRConfig(),
    on_sentence: Callable[[ASRSentence], None] | None = None,
) -> ASRResult:
    """Recognize one bounded 16 kHz PCM buffer, with no retry or fallback.

    Intermediate callbacks are revisions, not dialogue input. Only a completed
    result's ``transcript`` contains final text. The caller retains responsibility
    for consent and an applicable fee reservation; no free quota is assumed.
    """
    if not config.enabled:
        raise ValueError("asr_calls_not_enabled")
    validate_pcm(pcm, config)
    if not isinstance(api_key, str) or not api_key.strip() or "\r" in api_key or "\n" in api_key:
        raise ValueError("asr_api_key_missing_or_invalid")
    # Importing Aria's text preview never imports this optional networking stack.
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        raise RuntimeError("asr_optional_dependency_missing_install_asr_extra") from None

    result = ASRResult(str(uuid.uuid4()), MODEL, hashlib.sha256(pcm).hexdigest(),
                       len(pcm), len(pcm) / PCM_BYTES_PER_SECOND)
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    # Reserve a bounded closing window within the requested overall timeout.
    close_seconds = min(5.0, config.timeout_seconds / 10)
    work_seconds = config.timeout_seconds - close_seconds
    ws = None
    sender = None
    failure = None
    finals: dict[int, ASRSentence] = {}
    latest: dict[int, ASRSentence] = {}
    highest_sentence_id = -1

    async def receive_event():
        raw = await ws.recv()
        result.events_received += 1
        if result.events_received > _MAX_EVENTS:
            raise ValueError("asr_event_limit_exceeded")
        kind, header, payload = _parse_event(raw, result.task_id)
        provider_id = payload.get("output", {}).get("request_id")
        if (isinstance(provider_id, str) and _SAFE_CODE.fullmatch(provider_id)
                and api_key not in provider_id and provider_id not in result.provider_request_ids):
            result.provider_request_ids.append(provider_id)
        observed_usage = _usage(payload)
        if observed_usage is not None:
            result.last_observed_usage = observed_usage
            result.last_usage_event = kind
        if kind == "task-failed":
            provider_code = header.get("error_code")
            if (isinstance(provider_code, str) and _SAFE_CODE.fullmatch(provider_code)
                    and api_key not in provider_code):
                result.provider_error_code = provider_code
            raise ValueError("asr_provider_task_failed")
        return kind, payload

    async def send_audio():
        for offset in range(0, len(pcm), _CHUNK_BYTES):
            chunk = pcm[offset:offset + _CHUNK_BYTES]
            await ws.send(chunk)
            result.audio_bytes_sent += len(chunk)
            await asyncio.sleep(len(chunk) / PCM_BYTES_PER_SECOND)
        finish = {"header": {"action": "finish-task", "task_id": result.task_id,
                              "streaming": "duplex"}, "payload": {"input": {}}}
        await ws.send(json.dumps(finish))
        result.finish_task_sent = True

    async def run():
        nonlocal ws, sender, highest_sentence_id
        # A single await, not the library's reconnecting async iterator.
        quiet_logger = logging.Logger("aria.asr.transport", level=logging.CRITICAL + 1)
        quiet_logger.propagate = False
        connection = connect(ENDPOINT,
            additional_headers={"Authorization": "Bearer " + api_key},
            open_timeout=min(20.0, work_seconds), close_timeout=close_seconds,
            max_size=MAX_MESSAGE_BYTES, max_queue=16, logger=quiet_logger)
        # Even same-host redirects are a second attempt and aren't authorized.
        connection.process_redirect = lambda exc: exc
        ws = await connection
        request = {"header": {"action": "run-task", "task_id": result.task_id,
                               "streaming": "duplex"},
                   "payload": {"task_group": "audio", "task": "asr",
                               "function": "recognition", "model": MODEL,
                               "parameters": {"format": "pcm", "sample_rate": SAMPLE_RATE},
                               "input": {}}}
        await ws.send(json.dumps(request))
        result.run_task_sent = True
        kind, _ = await receive_event()
        if kind != "task-started":
            raise ValueError("asr_task_start_missing")
        result.task_started = True
        sender = asyncio.create_task(send_audio())
        while True:
            # Receive and send run together: an early provider failure stops audio.
            receiving = asyncio.create_task(receive_event())
            try:
                if not sender.done():
                    done, _ = await asyncio.wait((sender, receiving),
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if sender in done:
                        await sender  # surface send errors immediately
                else:
                    await sender
                kind, payload = await receiving
            finally:
                if not receiving.done():
                    receiving.cancel()
                await asyncio.gather(receiving, return_exceptions=True)
            if kind == "task-started":
                raise ValueError("asr_duplicate_task_start")
            if kind == "task-finished":
                if not result.finish_task_sent or result.audio_bytes_sent != len(pcm):
                    raise ValueError("asr_premature_task_finish")
                await sender
                result.task_finished = True
                result.final_usage = _usage(payload)
                if not finals or not any(item.text.strip() for item in finals.values()):
                    raise ValueError("asr_final_text_missing")
                if any(item.text.strip() and not item.final for item in latest.values()):
                    raise ValueError("asr_unfinished_sentence")
                result.status = "completed"
                return
            raw_sentence = payload.get("output", {}).get("sentence")
            if isinstance(raw_sentence, dict) and raw_sentence.get("heartbeat") is True:
                continue
            item = _sentence(payload)
            if item.sentence_id in finals:
                if item != finals[item.sentence_id]:
                    raise ValueError("asr_final_sentence_changed")
                continue
            if item.sentence_id < highest_sentence_id:
                raise ValueError("asr_sentence_order_changed")
            highest_sentence_id = item.sentence_id
            latest[item.sentence_id] = item
            if item.final:
                finals[item.sentence_id] = item
            result.sentences.append(item)
            if on_sentence is not None:
                on_sentence(item)

    try:
        await asyncio.wait_for(run(), timeout=work_seconds)
    except asyncio.CancelledError:
        result.status, result.error_code = "failed", "asr_cancelled"
        raise
    except asyncio.TimeoutError:
        failure = "asr_timeout"
    except ValueError as exc:
        code = str(exc)
        failure = code if _SAFE_CODE.fullmatch(code) and api_key not in code else "asr_protocol_error"
    except Exception:
        # Provider/transport exceptions may contain headers or response bodies.
        failure = "asr_transport_or_callback_error"
    finally:
        if sender is not None:
            if not sender.done():
                sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        if ws is not None:
            remaining = config.timeout_seconds - (loop.time() - started_at)
            try:
                if remaining <= 0:
                    raise asyncio.TimeoutError
                await asyncio.wait_for(ws.close(), timeout=remaining)
            except Exception:
                # Abort is local cleanup, never a reconnect or another request.
                transport = getattr(ws, "transport", None)
                if transport is not None:
                    transport.abort()
                failure = failure or "asr_close_failed"
        result.elapsed_ms = round((loop.time() - started_at) * 1000)
    if failure is not None:
        result.status, result.error_code = "failed", failure
        raise ASRError(failure, result) from None
    _completed_results[result] = _completion_binding(result)
    return result


def submit_final_transcript(result: ASRResult, text_app: Any, *,
                            session_id: str, confirmed: bool = False):
    """Explicitly submit final text through an already configured text app.

    Does not start a session, create clients, or enable model calls. The existing
    submit method enforces session, length, budget and durable turn-ID deduplication.
    Only an unchanged result completed by this process is accepted, not a
    deserialized receipt. Duplicate calls retain the same durable Aria turn ID.
    This is local source binding, not cryptographic proof from the provider.
    """
    if confirmed is not True:
        raise ValueError("asr_transcript_submission_not_confirmed")
    if not isinstance(result, ASRResult) or result.model != MODEL:
        raise ValueError("invalid_asr_result")
    if _completed_results.get(result) != _completion_binding(result):
        raise ValueError("asr_result_not_locally_verified")
    try:
        task_uuid = uuid.UUID(result.task_id)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid_asr_task_id") from None
    if (not result.task_started or not result.task_finished or result.error_code is not None
            or not result.transcript.strip()):
        raise ValueError("asr_final_transcript_unavailable")
    return text_app.submit(result.transcript, "asr_" + task_uuid.hex, session_id)
