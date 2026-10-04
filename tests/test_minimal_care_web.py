"""HTTP authentication and assembly checks, with no paid/external services."""
from __future__ import annotations

from datetime import datetime, timezone
import http.client
import json
import select
import socket
import threading
import urllib.request

import pytest

from aria.engine.minimal_care import MinimalCareService
from aria.voice.minimal import MinimalWebApp, make_minimal_server
import aria.voice.minimal as minimal


@pytest.fixture
def care_server(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("external request"))
    care = MinimalCareService(tmp_path / "care.sqlite3", clock=lambda: datetime(2026, 9, 20, tzinfo=timezone.utc))
    care.bootstrap("nurse", "fixture-only-long-password", ["fictional-chen"])
    care.bootstrap("other", "another-fixture-password", ["other-elder"])
    app = MinimalWebApp(tmp_path / "app", care_service=care)
    server = make_minimal_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, server
    server.shutdown()
    thread.join(timeout=3)
    server.server_close()


def call(app, server, route, *, data=None, cookie=None, token=True, origin=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Aria-Demo-Token"] = app.demo_token
    if cookie:
        headers["Cookie"] = cookie
    if origin:
        headers["Origin"] = origin
    connection.request("GET" if data is None else "POST", route,
                       None if data is None else json.dumps(data), headers)
    response = connection.getresponse()
    body = response.read()
    result = response.status, dict(response.getheaders()), json.loads(body)
    connection.close()
    return result


def login(app, server):
    status, headers, body = call(app, server, "/api/care/login", data={
        "username": "nurse", "password": "fixture-only-long-password"})  # release-fixture: not-a-deployment-secret
    assert status == 200 and body == {"ok": True}
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert "Path=/api/care" in cookie
    return cookie.split(";", 1)[0]


def test_login_scope_logout_and_no_chat_exposure(care_server):
    app, server = care_server
    assert call(app, server, "/api/care/state")[0] == 403
    assert call(app, server, "/api/care/login", data={"username": "nurse", "password": "bad"})[0] == 403
    cookie = login(app, server)
    status, _, body = call(app, server, "/api/care/state", cookie=cookie)
    assert status == 200
    assert "other-elder" not in json.dumps(body)
    assert not ({"turns", "profile", "hypotheses", "password", "token"} & body.keys())
    assert call(app, server, "/api/care/state?elder_id=other-elder", cookie=cookie)[0] == 403
    assert call(app, server, "/api/care/action", cookie=cookie, data={
        "elder_id": "other-elder", "action": "reminder.create", "payload": {"request_id": "forbidden"}})[0] == 403
    assert call(app, server, "/api/care/logout", cookie=cookie, data={})[0] == 200
    assert call(app, server, "/api/care/state", cookie=cookie)[0] == 403


def test_care_login_and_writes_require_local_token_same_origin(care_server):
    app, server = care_server
    payload = {"username": "nurse", "password": "fixture-only-long-password"}  # release-fixture: not-a-deployment-secret
    assert call(app, server, "/api/care/login", data=payload, token=False)[0] == 403
    assert call(app, server, "/api/care/login", data=payload, origin="https://evil.invalid")[0] == 403
    assert call(app, server, "/api/minimal/care", data={"action": "reminder.create", "payload": {}}, token=False)[0] == 403


@pytest.mark.parametrize("mode,status", [("allowed", 200), ("no_token", 403), ("wrong_origin", 403)])
def test_delayed_post_body_delivers_response_without_premature_action(care_server, mode, status):
    """The request body arrives after headers, as in the Windows reset failure."""
    app, server = care_server
    body = b"{}"
    headers = ["POST /api/minimal/start HTTP/1.1", f"Host: 127.0.0.1:{server.server_port}",
               "Content-Length: 2", "Content-Type: application/json"]
    if mode != "no_token":
        headers.append(f"X-Aria-Demo-Token: {app.demo_token}")
    if mode == "wrong_origin":
        headers.append("Origin: https://evil.invalid")
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as connection:
        connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
        # No rejection may close the connection while a permitted-size body is pending.
        assert not select.select([connection], [], [], 0.03)[0]
        assert app.state()["sessions"] == []
        connection.sendall(body)
        response = http.client.HTTPResponse(connection)
        response.begin()
        assert response.status == status
        result = json.loads(response.read())
        assert result["ok"] is (mode == "allowed")
    assert len(app.state()["sessions"]) == (1 if mode == "allowed" else 0)


@pytest.mark.parametrize("mode", ["missing_length", "oversized", "short_body", "read_timeout"])
def test_invalid_or_incomplete_body_closes_without_business_action(care_server, monkeypatch, mode):
    app, server = care_server
    monkeypatch.setattr(minimal, "_POST_BODY_TIMEOUT_SECONDS", 0.05)
    headers = ["POST /api/minimal/start HTTP/1.1", f"Host: 127.0.0.1:{server.server_port}",
               f"X-Aria-Demo-Token: {app.demo_token}"]
    if mode != "missing_length":
        headers.append("Content-Length: " + ("17001" if mode == "oversized" else "2"))
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as connection:
        connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
        if mode == "short_body":
            connection.sendall(b"{")
            connection.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(connection)
        response.begin()
        assert response.status == 400
        assert json.loads(response.read())["ok"] is False
        assert connection.recv(1) == b""
    assert app.state()["sessions"] == []


def test_public_elder_view_is_fixed_and_simulation_cannot_become_real_consent(care_server):
    app, server = care_server
    status, _, body = call(app, server, "/api/minimal/care?elder_id=other-elder")
    assert status == 200 and "other-elder" not in json.dumps(body)
    payload = {"action": "consent.grant", "payload": {"source": "real", "request_id": "not-real"}}
    assert call(app, server, "/api/minimal/care", data=payload)[0] == 400
    cookie = login(app, server)
    assert call(app, server, "/api/care/action", cookie=cookie,
                data={**payload, "elder_id": "fictional-chen"})[0] == 400


def test_care_scheduler_keeps_running_after_chat_failure_and_seal(care_server, monkeypatch):
    app, server = care_server
    sid = app.start()["state"]["session_id"]
    app.submit("今天先安静一会儿", "quiet", sid)
    assert app.seal(sid)["ok"]
    called = []
    monkeypatch.setattr(app.care, "tick", lambda: called.append("care"))
    monkeypatch.setattr(app, "finalize_idle_sessions", lambda: (_ for _ in ()).throw(OSError("private details")))
    server._maintenance_once()
    assert called == ["care"]
    assert server.maintenance_errors == {"finalize_idle_sessions": "OSError"}
    assert app.state()["session_id"] is None
    assert app.state()["offline"]["pending_count"] == 1
    assert not app.state()["model_calls_enabled"]


def test_fixed_elder_care_actions_idempotent_and_reads_side_effect_free(care_server, tmp_path):
    app, server = care_server
    payload = {"action": "reminder.create", "payload": {
        "request_id": "one-request", "label": "拿纸和本子", "schedule": "once",
        "date": "2026-09-21", "time": "15:30", "timezone": "Asia/Shanghai", "grace_minutes": 30}}
    assert call(app, server, "/api/minimal/care", data=payload)[0] == 200
    assert call(app, server, "/api/minimal/care", data=payload)[0] == 200
    status, _, view = call(app, server, "/api/minimal/care")
    assert status == 200 and len(view["reminders"]) == 1
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for _ in range(3):
        assert call(app, server, "/api/minimal/care")[0] == 200
        assert call(app, server, "/api/minimal/state")[0] == 200
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert app.state()["turns"] == []


def test_shutdown_drains_accepted_http_start_before_sealing(tmp_path, monkeypatch):
    """A real handler waits at a barrier until listener shutdown has started."""
    app = MinimalWebApp(tmp_path)
    entered, release = threading.Event(), threading.Event()
    close_started, app_closed = threading.Event(), threading.Event()
    start, close = app.start, app.close
    result, errors = {}, []

    def held_start():
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test barrier timed out")
        return start()

    def observed_close():
        close()
        app_closed.set()

    monkeypatch.setattr(app, "start", held_start)
    monkeypatch.setattr(app, "close", observed_close)
    server = make_minimal_server(app, port=0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()

    def send_request():
        try:
            result["response"] = call(app, server, "/api/minimal/start", data={})
        except Exception as exc:
            errors.append(type(exc).__name__)

    def close_server():
        close_started.set()
        try:
            server.server_close()
        except Exception as exc:
            errors.append(type(exc).__name__)

    requesting = threading.Thread(target=send_request, daemon=True)
    closing = threading.Thread(target=close_server, daemon=True)
    try:
        requesting.start()
        assert entered.wait(timeout=2)
        server.shutdown()
        serving.join(timeout=2)
        closing.start()
        assert close_started.wait(timeout=2)
        assert not app_closed.wait(timeout=0.1)
        assert closing.is_alive()  # The accepted request is still outstanding.
        release.set()
        requesting.join(timeout=3)
        closing.join(timeout=3)
        assert not requesting.is_alive() and not closing.is_alive()
        assert not errors and result["response"][0] == 200
        assert app_closed.is_set()
        assert app.state()["session_id"] is None
        assert app.state()["sessions"][0]["status"] == "sealed"
        assert not app._sessions
        assert server.socket.fileno() == -1
    finally:
        release.set()
        if serving.is_alive():
            server.shutdown()
        serving.join(timeout=3)
        if requesting.ident is not None:
            requesting.join(timeout=3)
        if closing.ident is not None:
            closing.join(timeout=3)
        server.server_close()
