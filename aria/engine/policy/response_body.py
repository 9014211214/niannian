"""Bounded response consumption; never repair framing or retry a request.

Only byte counts, hashes and closed categories leave this reader on failure.
A complete JSON prefix is not a complete HTTP response. Failed partial bytes
are never passed to capture hooks or used to infer a fee.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import io
import json
import math
import re
import time


SCHEMA = "bounded_response_body_v1"
CHUNK_BYTES = 64 * 1024
_KINDS = {"capture_failed", "body_limit", "framing_invalid", "incomplete_read", "remote_disconnected",
    "connection_reset", "timeout", "http_protocol_error", "other_read_error"}


def validate_body_read_observation(value):
    """Validate evidence shape only; this never grants recovery permission."""
    fields = {"schema", "max_bytes", "bytes_received", "body_sha256", "framing", "content_length",
        "framing_checked", "complete", "exception_kind", "json_status", "read_mode", "capture_status"}
    if (type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA
            or type(value["max_bytes"]) is not int or value["max_bytes"] < 1
            or type(value["bytes_received"]) is not int or not 0 <= value["bytes_received"] <= value["max_bytes"] + 1
            or type(value["body_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", value["body_sha256"])
            or type(value["framing"]) is not str or value["framing"] not in {"unobserved", "chunked", "content_length", "close_delimited"}
            or value["content_length"] is not None and (type(value["content_length"]) is not int or not 0 <= value["content_length"] < 10**20)
            or type(value["complete"]) is not bool or type(value["framing_checked"]) is not bool
            or value["exception_kind"] is not None and (type(value["exception_kind"]) is not str or value["exception_kind"] not in _KINDS)
            or type(value["json_status"]) is not str or value["json_status"] not in {"not_inspected", "complete_object", "non_object", "invalid_or_incomplete"}
            or type(value["read_mode"]) is not str or value["read_mode"] not in {"unobserved", "read1", "single_read_fallback"}
            or type(value["capture_status"]) is not str or value["capture_status"] not in {"not_requested", "complete", "failed"}):
        raise ValueError("invalid_body_read_observation")
    if ((value["framing"] == "content_length") != (value["content_length"] is not None)
            or value["framing_checked"] != (value["framing"] != "unobserved")
            or value["bytes_received"] == 0 and value["body_sha256"] != hashlib.sha256(b"").hexdigest()
            or value["complete"] and (not value["framing_checked"] or value["bytes_received"] > value["max_bytes"]
                or value["read_mode"] == "unobserved" or value["exception_kind"] not in {None, "capture_failed"}
                or value["content_length"] is not None and value["bytes_received"] != value["content_length"])
            or (value["capture_status"] == "failed") != (value["exception_kind"] == "capture_failed")):
        raise ValueError("invalid_body_read_observation")
    return value


class ResponseDeadline:
    def __init__(self, expires_at, *, clock=time.perf_counter):
        if type(expires_at) not in (float, int) or not math.isfinite(expires_at) or not callable(clock):
            raise ValueError("invalid_response_deadline")
        self.expires_at, self.clock, self.last = expires_at, clock, None

    def remaining(self):
        now = self.clock()
        if (type(now) not in (int, float) or not math.isfinite(now)
                or self.last is not None and now < self.last):
            raise ValueError("invalid_response_clock")
        self.last = now
        remaining = self.expires_at - now
        if remaining <= 0:
            raise TimeoutError("response_deadline_exhausted")
        return remaining


class DeadlineRawReader(io.RawIOBase):
    """Apply the same deadline before every socket read, including headers.

Buffered readline/read1 may perform several raw reads. Updating the socket
timeout at this boundary prevents a slow trickle from restarting its budget.
"""
    def __init__(self, raw, sock, deadline):
        self.raw, self.sock, self.deadline = raw, sock, deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        self.sock.settimeout(self.deadline.remaining())
        result = self.raw.readinto(buffer)
        self.deadline.remaining()
        return result

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class ResponseFramingError(http.client.HTTPException):
    pass


class ResponseBodyLimitError(http.client.HTTPException):
    pass


class ResponseCaptureError(http.client.HTTPException):
    pass


def _framing(response):
    headers = getattr(response, "headers", None)
    def values(name):
        if headers is None:
            return []
        get_all = getattr(headers, "get_all", None)
        if callable(get_all):
            return get_all(name, [])
        value = headers.get(name)
        return [] if value is None else [value]
    lengths, transfers = values("Content-Length"), values("Transfer-Encoding")
    if (len(lengths) > 1 or len(transfers) > 1 or lengths and transfers
            or any(type(value) is not str for value in lengths + transfers)):
        raise ResponseFramingError("ambiguous_response_framing")
    if transfers:
        if transfers[0].strip().lower() != "chunked":
            raise ResponseFramingError("unsupported_response_framing")
        return "chunked", None
    if lengths:
        value = lengths[0].strip()
        if not re.fullmatch(r"[0-9]{1,20}", value):
            raise ResponseFramingError("invalid_response_length")
        return "content_length", int(value)
    return "close_delimited", None


def _json_status(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_member")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite_json")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        return "complete_object" if type(value) is dict else "non_object"
    except (UnicodeError, ValueError, TypeError, RecursionError):
        return "invalid_or_incomplete"


def _kind(error):
    for cls, label in ((ResponseCaptureError, "capture_failed"), (ResponseBodyLimitError, "body_limit"),
            (ResponseFramingError, "framing_invalid"), (http.client.IncompleteRead, "incomplete_read"),
            (http.client.RemoteDisconnected, "remote_disconnected"),
            (ConnectionResetError, "connection_reset"), (TimeoutError, "timeout"),
            (http.client.HTTPException, "http_protocol_error")):
        if isinstance(error, cls):
            return label
    return "other_read_error"


def read_response_body(response, *, limit, deadline, observation):
    """Return only a complete bounded body, with its original bytes intact.

The raw transport's framing remains authoritative, and all JSON/identity/fee
validation remains with the caller. No partial body is accepted or exposed.
Fallback read exists for inert/custom response adapters; production observed
HTTP responses always implement read1 and a deadline at the socket boundary.
"""
    if type(limit) is not int or limit < 1 or type(deadline) is not ResponseDeadline or type(observation) is not dict:
        raise ValueError("invalid_response_reader_configuration")
    observation.update(schema=SCHEMA, max_bytes=limit, bytes_received=0,
        body_sha256=hashlib.sha256(b"").hexdigest(), framing="unobserved", content_length=None,
        framing_checked=False, complete=False, exception_kind=None, json_status="not_inspected",
        read_mode="unobserved", capture_status="not_requested")
    body = bytearray()
    try:
        deadline.remaining()
        framing, length = _framing(response)
        observation.update(framing=framing, content_length=length, framing_checked=True)
        if length is not None and length > limit:
            raise ResponseBodyLimitError("response_body_limit")
        read1 = getattr(response, "read1", None)
        incremental = callable(read1)
        observation["read_mode"] = "read1" if incremental else "single_read_fallback"
        while True:
            deadline.remaining()
            try:
                part = read1(min(CHUNK_BYTES, limit + 1 - len(body))) if incremental else response.read(limit + 1)
            except http.client.IncompleteRead as error:
                # Only bytes accounted within the existing bound; no exception
                # message or partial body is persisted or passed to a capture.
                # read1 exposes data in prior successful calls; an exception
                # partial can instead be an incomplete chunk delimiter.
                if not incremental and type(error.partial) is bytes:
                    body.extend(error.partial[:max(0, limit + 1 - len(body))])
                raise
            if type(part) is not bytes:
                raise ResponseFramingError("non_bytes_response_body")
            body.extend(part[:max(0, limit + 1 - len(body))])
            if len(body) > limit or len(part) > limit:
                raise ResponseBodyLimitError("response_body_limit")
            deadline.remaining()
            if not incremental or not part:
                break
        if length is not None and len(body) != length:
            raise http.client.IncompleteRead(b"", max(0, length - len(body)))
        observation.update(complete=True, bytes_received=len(body), body_sha256=hashlib.sha256(body).hexdigest())
        capture = getattr(response, "observe_complete_body", None)
        if callable(capture):
            try:
                capture(bytes(body), observation=deepcopy(observation))
            except Exception:
                observation["capture_status"] = "failed"
                raise ResponseCaptureError("response_capture_failed") from None
            observation["capture_status"] = "complete"
        return bytes(body)
    except Exception as error:
        observation.update(bytes_received=len(body), body_sha256=hashlib.sha256(body).hexdigest(), exception_kind=_kind(error))
        if len(body) <= limit:
            observation["json_status"] = _json_status(body)
        if observation["capture_status"] != "failed":
            capture = getattr(response, "observe_failed_body", None)
            if callable(capture):
                try:
                    capture(observation=deepcopy(observation))
                except Exception:
                    observation.update(capture_status="failed", exception_kind="capture_failed")
                    raise ResponseCaptureError("response_capture_failed") from None
                observation["capture_status"] = "complete"
        raise
