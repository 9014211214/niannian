"""Default guard/opener redirect boundaries, using only synthetic sockets."""
from decimal import Decimal
import io
import socket
import urllib.error
import urllib.request

import pytest

from aria.engine.policy import observed_transport as transport
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.product_calls import ProductCallGuard
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_reserved_calls import call, client, policy


class SyntheticSocket:
    def __init__(self, status, target):
        self.writes = []
        body = b'{"error":{"message":"synthetic redirect"}}'
        self.response = (f"HTTP/1.1 {status} Redirect\r\nLocation: {target}\r\n"
                         f"Content-Length: {len(body)}\r\n\r\n").encode() + body

    def sendall(self, data):
        self.writes.append(bytes(data))

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.response)

    def setsockopt(self, *args):
        pass

    def close(self):
        pass


def fake_connection(monkeypatch, status, target):
    sock = SyntheticSocket(status, target)
    connections = []
    def connect(address, *args, **kwargs):
        connections.append(address)
        assert len(connections) == 1, "a redirect must never issue another request"
        return sock
    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("global opener used"))
    monkeypatch.setattr(urllib.request, "install_opener", lambda *a: pytest.fail("global opener changed"))
    return sock, connections


@pytest.mark.parametrize("status", range(300, 309))
@pytest.mark.parametrize("target", [
    "http://other.invalid/second", "https://other.invalid/second",
    "http://fixture.invalid/other", "ftp://other.invalid/second",
])
@pytest.mark.parametrize("observed", [False, True])
def test_no_redirect_or_forwarded_credentials_in_either_transport(monkeypatch, status, target, observed):
    sock, connections = fake_connection(monkeypatch, status, target)
    request = urllib.request.Request("http://fixture.invalid/first", data=b"synthetic-body",
                                     headers={"Authorization": "Bearer synthetic-only"}, method="POST")
    prior_opener = urllib.request._opener
    with pytest.raises(urllib.error.HTTPError) as caught:
        if observed:
            transport.open_observed(request, timeout=1, trace=transport.TransportTrace())
        else:
            transport.open_unredirected(request, timeout=1)
    assert caught.value.code == status
    caught.value.close()
    assert connections == [("fixture.invalid", 80)]
    assert len(sock.writes) == 2
    assert sock.writes[0].startswith(b"POST /first HTTP/1.1\r\n")
    assert b"Authorization: Bearer synthetic-only\r\n" in sock.writes[0]
    assert sock.writes[1] == b"synthetic-body"
    assert urllib.request._opener is prior_opener


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("cls", [MinimalCallGuard, ReservedCallGuard, ProductCallGuard])
def test_default_guards_record_only_first_failure_and_keep_unknown_hold(tmp_path, monkeypatch, status, cls):
    _sock, connections = fake_connection(monkeypatch, status, "https://unregistered.invalid/second")
    c = client()
    c.base_url = "http://fixture.invalid/v1"
    options = {"max_requests": 20, "max_observed_cost_usd": "1"}
    if cls is not MinimalCallGuard:
        options["reservation_policy"] = policy()
    path = tmp_path / "redirect.sqlite3"
    g = cls(path, **options)
    try:
        g.attach(c)
        with pytest.raises(MinimalCallStopped):
            call(g, c)
        state = g.snapshot()
        assert len(connections) == c.network_request_count == state["calls_used"] == 1
        assert state["stopped"] and state["unknown_cost_requests"] == 1
        row = state["attempts"][0]
        assert row["http_status"] == status and row["failure_phase"] == "http_status"
        assert row["status"] == "failed" and "cost" not in row["usage"]
        if cls is not MinimalCallGuard:
            assert Decimal(state["reserved_cost_usd"]) > 0
        if cls is ProductCallGuard:
            assert not state["can_resume"]
            with pytest.raises(MinimalCallStopped):
                g.resume("cannot-clear-unknown", state.get("pause_id"))
        with pytest.raises(MinimalCallStopped):
            call(g, c, "second-operation")
        assert len(connections) == 1 and g.snapshot()["attempts"] == state["attempts"]
    finally:
        g.close()
    reopened = cls(path, **options)
    try:
        restored = reopened.snapshot()
        assert restored["attempts"] == state["attempts"]
        assert restored["stopped"] and restored["unknown_cost_requests"] == 1
        if cls is not MinimalCallGuard:
            assert restored["reserved_cost_usd"] == state["reserved_cost_usd"]
    finally:
        reopened.close()
