"""Real loopback sockets exercise framing, deadlines and single-dispatch holds."""
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import http.client
import json
import socket
import threading
import time
import urllib.request

import pytest

from aria.engine.policy import observed_transport as transport
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.response_body import (
    ResponseDeadline, ResponseBodyLimitError, ResponseCaptureError, ResponseFramingError,
    read_response_body,
)
from tests.test_reserved_calls import client, call, policy, receipt


@pytest.fixture(autouse=True)
def no_external_proxy(monkeypatch):
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})


@contextmanager
def server(parts):
    """One local connection; no credentials and no external traffic."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    seen, errors = [], []
    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(3)
                header = bytearray()
                while b"\r\n\r\n" not in header:
                    piece = conn.recv(4096)
                    if not piece:
                        break
                    header.extend(piece)
                header_bytes, _, received_body = bytes(header).partition(b"\r\n\r\n")
                content_length = next((int(line.split(b":", 1)[1].strip()) for line in header_bytes.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")), 0)
                while len(received_body) < content_length:
                    piece = conn.recv(min(4096, content_length - len(received_body)))
                    if not piece:
                        break
                    received_body += piece
                seen.append(bytes(header))
                for delay, data in parts:
                    if delay:
                        time.sleep(delay)
                    conn.sendall(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Expected when the bounded client closes a rejected response.
        except Exception as error:
            errors.append(type(error).__name__)
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", seen
    finally:
        listener.close()
        worker.join(3)
        assert not worker.is_alive()
        assert errors == []


def fixed(body, extra=0, headers=b""):
    return b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)+extra).encode() + b"\r\n" + headers + b"\r\n" + body


def chunked(body, ending=b"\r\n0\r\n\r\n", delimiter=b"\r\n"):
    return b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + format(len(body), "x").encode() + delimiter + body + ending


def read(parts, *, timeout=2, limit=1024, wrapper=lambda x:x):
    audit = {}
    with server(parts) as (url, seen):
        deadline = ResponseDeadline(time.perf_counter() + timeout)
        trace = transport.TransportTrace(diagnostics=True, response_deadline=deadline)
        with transport.open_observed(urllib.request.Request(url), timeout=timeout, trace=trace) as raw:
            value = read_response_body(wrapper(raw), limit=limit, deadline=deadline, observation=audit)
        assert len(seen) == 1
    return value, audit


@pytest.mark.parametrize("wire", [
    fixed(b'{"ok":true}'), chunked(b'{"ok":true}'),
    b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{"ok":true}',
    chunked(b'{"ok":true}', ending=b"\r\n0\r\nX-Test: ignored\r\n\r\n"),
])
def test_complete_http_bodies_keep_exact_bytes(wire):
    value, audit = read([(0, wire)])
    assert value == b'{"ok":true}'
    assert audit["complete"] and audit["framing_checked"]
    assert audit["body_sha256"] == hashlib.sha256(value).hexdigest()
    assert audit["read_mode"] == "read1"


@pytest.mark.parametrize("wire,exception", [
    (fixed(b'{"ok":true}', extra=20), http.client.IncompleteRead),
    (chunked(b'{"ok":true}', ending=b"\r\n"), http.client.IncompleteRead),
    (chunked(b'{"ok":true}', ending=b"\r\n0\r\n"), http.client.IncompleteRead),
    (chunked(b'{"ok":true}', ending=b"\r"), http.client.IncompleteRead),
    (chunked(b'{"ok":true}', ending=b"xx0\r\n\r\n"), ResponseFramingError),
    (chunked(b'{"ok":true}', delimiter=b"\n"), http.client.IncompleteRead),
    (fixed(b'{"ok":true}', headers=b"Content-Length: 999\r\n"), ResponseFramingError),
    (fixed(b'{"ok":true}', headers=b"Transfer-Encoding: chunked\r\n"), ResponseFramingError),
    (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n{}", ResponseFramingError),
    (b"HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n{}", ResponseFramingError),
])
def test_incomplete_or_ambiguous_framing_never_returns_complete_json_prefix(wire, exception):
    with pytest.raises(exception):
        read([(0, wire)])


@pytest.mark.parametrize("wire", [fixed(b"x"*65), chunked(b"x"*65), b"HTTP/1.1 200 OK\r\n\r\n" + b"x"*65])
def test_body_limit_applies_to_each_framing(wire):
    with pytest.raises(ResponseBodyLimitError):
        read([(0, wire)], limit=64)


def test_multiple_socket_reads_one_body_one_complete_callback():
    captures = []
    class Capture:
        def __init__(self, raw): self.raw = raw
        def __getattr__(self, name): return getattr(self.raw, name)
        def observe_complete_body(self, raw, *, observation): captures.append((raw, observation))
        def observe_failed_body(self, *, observation): pytest.fail("unexpected failed body")
    body = b'{"test":"' + b"a"*70000 + b'"}'
    wire = chunked(body)
    value, audit = read([(0, wire[:80]), (.01, wire[80:30000]), (.01, wire[30000:])], limit=100000, wrapper=Capture)
    assert value == body and len(captures) == 1
    assert captures[0][0] == body and audit["capture_status"] == "complete"


def test_failed_capture_gets_metadata_only_and_no_hidden_or_partial_text():
    captures = []
    secret = b'{"reasoning_content":"HIDDEN_TRANSPORT_SENTINEL"}'
    class Capture:
        def __init__(self, raw): self.raw = raw
        def __getattr__(self, name): return getattr(self.raw, name)
        def observe_complete_body(self, raw, *, observation): pytest.fail("partial accepted")
        def observe_failed_body(self, *, observation): captures.append(observation)
    with pytest.raises(http.client.IncompleteRead):
        read([(0, chunked(secret, ending=b"\r\n"))], wrapper=Capture)
    assert len(captures) == 1 and not captures[0]["complete"]
    assert captures[0]["bytes_received"] == len(secret)
    assert captures[0]["json_status"] == "complete_object"
    assert "HIDDEN" not in json.dumps(captures)


@pytest.mark.parametrize("failure", ["complete", "partial"])
def test_capture_storage_failure_is_distinct_and_never_recoverable(failure):
    class Capture:
        def __init__(self, raw): self.raw = raw
        def __getattr__(self, name): return getattr(self.raw, name)
        def observe_complete_body(self, *args, **kwargs): raise ConnectionResetError("private storage")
        def observe_failed_body(self, **kwargs): raise ConnectionResetError("private storage")
    wire = fixed(b"{}", extra=3 if failure == "partial" else 0)
    with pytest.raises(ResponseCaptureError):
        read([(0, wire)], wrapper=Capture)


@pytest.mark.parametrize("phase", ["body", "headers", "chunk_size", "trailer"])
def test_slow_trickle_cannot_reset_absolute_deadline(phase):
    prefix = {"body": b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\n\r\n",
        "headers": b"HTTP/1.1 200 OK\r\nX-Slow: ",
        "chunk_size": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        "trailer": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX-Slow: "}[phase]
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        read([(0, prefix)] + [(.20, b"1")]*8, timeout=1.0)
    assert time.perf_counter() - started < 2.5


def test_pre_header_elapsed_time_is_deducted_from_body_budget():
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        read([(.40, b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n"), (.70,b"1234")], timeout=1.0)
    assert time.perf_counter() - started < 2.5


@pytest.mark.parametrize("bad_framing", [False, True])
def test_real_http_guard_keeps_fee_hold_and_never_dispatches_again(tmp_path, bad_framing):
    c = client()
    body = json.dumps(receipt(c)).encode()
    wire = fixed(body, extra=20 if bad_framing else 0)
    with server([(0, wire)]) as (url, seen):
        c.base_url = url
        c.timeout = 2
        g = ReservedCallGuard(tmp_path / "real-http.sqlite3", max_requests=5, max_observed_cost_usd="1",
            reservation_policy=policy(), observe_transport=True, transport_diagnostics=True)
        g.attach(c)
        try:
            if bad_framing:
                with pytest.raises(MinimalCallStopped): call(g, c)
                state = g.snapshot()
                assert state["calls_used"] == 1 and state["unknown_cost_requests"] == 1
                assert Decimal(state["reserved_cost_usd"]) > 0
                assert not state["attempts"][0]["body_read"]["complete"]
                assert state["attempts"][0]["observation"]["body_read_error_kind"] == "incomplete_read"
                with pytest.raises(MinimalCallStopped): call(g, c, "no-retry")
            else:
                assert call(g, c) == "local fixture"
                assert g.snapshot()["attempts"][0]["body_read"]["complete"]
            assert len(seen) == 1
        finally:
            g.close()
