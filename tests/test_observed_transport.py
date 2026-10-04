"""Fake sockets exercise opt-in stages without network, credentials or sleeps."""
import io
import json
import socket
import ssl
import urllib.error
import urllib.request

import pytest

from aria.engine.policy import minimal_calls, observed_transport as observed
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_reserved_calls import call, client, policy, receipt


def response(body=b"{}", status=b"200 OK", extra=b""):
    return b"HTTP/1.1 " + status + b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n" + extra + b"\r\n" + body


class FakeSocket:
    def __init__(self, trace, responses=None, *, write_failure=None):
        self.trace, self.responses = trace, iter(responses or [response()])
        self.writes, self.write_failure, self.closed = [], write_failure, False

    def sendall(self, data):
        self.writes.append((bytes(data), self.trace.snapshot()))
        if self.write_failure == len(self.writes):
            raise BrokenPipeError("PRIVATE partial write; bytes may already have left")

    def makefile(self, *args, **kwargs):
        value = next(self.responses)
        if isinstance(value, BaseException):
            class BrokenReader(io.BytesIO):
                def readline(self, *args):
                    raise value
            return BrokenReader()
        return io.BytesIO(value)

    def setsockopt(self, *args):
        pass

    def close(self):
        self.closed = True


class FakeTLS:
    # Python 3.10 checks these SSLContext attributes during construction.
    verify_mode = ssl.CERT_REQUIRED
    check_hostname = True

    def __init__(self, *, fail=False):
        self.fail, self.calls = fail, 0

    def wrap_socket(self, sock, *, server_hostname):
        self.calls += 1
        if self.fail:
            raise ssl.SSLCertVerificationError("PRIVATE cert")
        return sock


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("unstubbed socket"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected global opener"))
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})


def connection(*, https=True, tunnel=False, failure=None):
    trace = observed.TransportTrace()
    tls = FakeTLS(fail=failure == "tls")
    responses = [b"HTTP/1.1 200 Connection established\r\n\r\n"] if tunnel else []
    responses.append(ConnectionResetError("PRIVATE headers") if failure == "headers" else response())
    sock = FakeSocket(trace, responses, write_failure=(1 if failure == "header_write" else
        2 if failure == "write" and not tunnel else 3 if failure == "write" else None))
    cls = observed._ObservedHTTPSConnection if https else observed._ObservedHTTPConnection
    conn = cls("fixture.invalid", trace=trace, **({"context": tls} if https else {}))
    conn._create_connection = lambda *a, **kw: sock
    if failure == "prepare":
        conn._create_connection = lambda *a, **kw: (_ for _ in ()).throw(socket.gaierror(-2, "PRIVATE host"))
    if tunnel:
        conn.set_tunnel("fixture.invalid", headers={"Proxy-Authorization": "PRIVATE proxy"})
    return conn, trace, sock, tls


@pytest.mark.parametrize("https,tunnel", [(False, False), (True, False), (True, True)])
def test_ordered_local_events_proxy_connect_is_not_application_write(https, tunnel):
    conn, trace, sock, tls = connection(https=https, tunnel=tunnel)
    conn.request("POST", "/synthetic", body=b"PRIVATE_BODY", headers={"Authorization": "PRIVATE_KEY"})
    result = conn.getresponse()
    assert result.read() == b"{}"
    events = trace.snapshot()["events"]
    assert events == dict.fromkeys({"connection_preparation_started", "connection_preparation_completed",
        "application_write_attempted", "request_write_completed", "response_headers_returned"} | ({"tls_completed"} if https else set()), True)
    if tunnel:
        assert sock.writes[0][0].startswith(b"CONNECT ")
        assert sock.writes[0][1]["events"] == {"connection_preparation_started": True}
    first_application = sock.writes[1 if tunnel else 0][1]["events"]
    assert first_application["application_write_attempted"] and first_application["connection_preparation_completed"]
    assert "request_write_completed" not in first_application
    assert tls.calls == int(https)
    assert "PRIVATE" not in json.dumps(trace.snapshot())
    assert "fixture.invalid" not in json.dumps(trace.snapshot())
    conn.close()


@pytest.mark.parametrize("failure", ["prepare", "tls", "write", "headers"])
def test_partial_or_failed_stages_are_not_inferred_completed(failure):
    conn, trace, sock, _ = connection(failure=failure)
    with pytest.raises((OSError, ssl.SSLError)):
        conn.request("POST", "/synthetic", body=b"PRIVATE")
        conn.getresponse()
    events = trace.snapshot()["events"]
    assert events["connection_preparation_started"]
    assert "response_headers_returned" not in events
    if failure in {"prepare", "tls"}:
        assert "connection_preparation_completed" not in events and "application_write_attempted" not in events
        assert sock.writes == []
    else:
        assert events["application_write_attempted"]
        assert ("request_write_completed" in events) is (failure == "headers")
    assert trace.snapshot()["absent_events"] == "not_observed"
    assert "PRIVATE" not in json.dumps(trace.snapshot())
    conn.close()


def test_proxy_connect_failure_does_not_mark_origin_write():
    conn, trace, sock, tls = connection(tunnel=True)
    sock.responses = iter([b"HTTP/1.1 407 Proxy authorization required\r\n\r\n"])
    with pytest.raises(OSError):
        conn.request("POST", "/synthetic", body=b"PRIVATE")
    assert trace.snapshot()["events"] == {"connection_preparation_started": True}
    assert len(sock.writes) == 1 and tls.calls == 0


def test_proxy_connect_success_then_tls_failure_never_marks_application_write():
    conn, trace, sock, tls = connection(tunnel=True, failure="tls")
    with pytest.raises(ssl.SSLCertVerificationError):
        conn.request("POST", "/synthetic", body=b"PRIVATE")
    assert trace.snapshot()["events"] == {"connection_preparation_started": True}
    assert len(sock.writes) == 1 and sock.writes[0][0].startswith(b"CONNECT ")
    assert tls.calls == 1
    conn.close()


def test_per_request_opener_uses_default_verified_tls_proxy_and_no_install(monkeypatch):
    trace = observed.TransportTrace()
    sock = FakeSocket(trace, [b"HTTP/1.1 200 Connection established\r\n\r\n", response()])
    seen = []
    real_class = observed._ObservedHTTPSConnection
    def fake_connection(*args, **kwargs):
        context = kwargs["context"]
        # Some older supported Python versions leave default context creation
        # to HTTPSConnection; inspect the resulting standard context as well.
        conn = real_class(*args, **kwargs)
        verified = conn._context if context is None else context
        assert verified.verify_mode == ssl.CERT_REQUIRED and verified.check_hostname is True
        conn._context = FakeTLS()  # Test only: handshake without a real socket.
        conn._create_connection = lambda *a, **kw: sock
        seen.append(conn)
        return conn
    monkeypatch.setattr(observed, "_ObservedHTTPSConnection", fake_connection)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://synthetic-proxy.invalid:8888"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(urllib.request, "install_opener", lambda _: pytest.fail("global install"))
    previous = urllib.request._opener
    request = urllib.request.Request("https://fixture.invalid/synthetic", data=b"PRIVATE", method="POST")
    with observed.open_observed(request, timeout=2, trace=trace) as result:
        assert result.read() == b"{}"
    assert urllib.request._opener is previous and len(seen) == 1
    assert sock.writes[0][0].startswith(b"CONNECT ")
    assert "application_write_attempted" not in sock.writes[0][1]["events"]
    assert "synthetic-proxy" not in json.dumps(trace.snapshot())


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_never_sends_a_second_application_request(monkeypatch, status):
    trace = observed.TransportTrace()
    sock = FakeSocket(trace, [response(status=str(status).encode() + b" Moved", extra=b"Location: http://other.invalid/PRIVATE\r\n")])
    connections = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: (connections.append(1) or sock))
    request = urllib.request.Request("http://fixture.invalid/synthetic", data=b"PRIVATE", method="POST")
    with pytest.raises(urllib.error.HTTPError) as failure:
        observed.open_observed(request, timeout=2, trace=trace)
    assert failure.value.code == status and len(connections) == 1
    assert len(sock.writes) == 2  # one header write + one body write, one POST
    assert trace.snapshot()["events"]["response_headers_returned"]


@pytest.mark.parametrize("failure", [False, "header_write", "write"])
def test_enabled_guard_persists_events_but_never_releases_unknown_or_retries(tmp_path, monkeypatch, failure):
    current = ReservedCallGuard(tmp_path / "observed.sqlite3", max_requests=30, max_observed_cost_usd="1",
        reservation_policy=policy(), observe_transport=True)
    model = client()
    current.attach(model)
    calls = []
    def opened(request, *, timeout, trace):
        calls.append(request.data)
        conn, local_trace, sock, _ = connection(failure=failure if failure else None)
        conn._trace = trace
        sock.trace = trace
        if not failure:
            sock.responses = iter([response(json.dumps(receipt(model)).encode())])
        conn.request("POST", "/synthetic", body=request.data)
        return conn.getresponse()
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    if failure:
        with pytest.raises(MinimalCallStopped):
            call(current, model)
        assert current.snapshot()["unknown_cost_requests"] == 1
        assert current.snapshot()["reserved_cost_usd"] != "0"
        with pytest.raises(MinimalCallStopped):
            call(current, model, "no-retry")
    else:
        assert call(current, model) == "local fixture"
        assert current.snapshot()["unknown_cost_requests"] == 0
    row = current.snapshot()["attempts"][0]
    assert row["transport_events"]["events"]["application_write_attempted"]
    assert ("request_write_completed" in row["transport_events"]["events"]) is not bool(failure)
    assert len(calls) == 1
    current.close()


def test_opt_in_cannot_be_added_to_used_ledger_or_silently_disabled(tmp_path, monkeypatch):
    path = tmp_path / "new.sqlite3"
    g = ReservedCallGuard(path, max_requests=30, max_observed_cost_usd="1", reservation_policy=policy(), observe_transport=True)
    g.close()
    with pytest.raises(ValueError, match="transport_observation_changed"):
        ReservedCallGuard(path, max_requests=30, max_observed_cost_usd="1", reservation_policy=policy())
    legacy = tmp_path / "legacy.sqlite3"
    old = ReservedCallGuard(legacy, max_requests=30, max_observed_cost_usd="1", reservation_policy=policy())
    model = client(); old.attach(model)
    from tests.test_reserved_calls import stub
    stub(monkeypatch, receipt(model))
    call(old, model)
    old.close()
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="no_transport_observation_contract"):
        ReservedCallGuard(legacy, max_requests=30, max_observed_cost_usd="1", reservation_policy=policy(), observe_transport=True)
    assert legacy.read_bytes() == before
