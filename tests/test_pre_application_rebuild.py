"""All connections, TLS handshakes, time and model responses are local fakes."""
from copy import deepcopy
from decimal import Decimal
import http.client
import json
import socket
import ssl
import urllib.request

import pytest

from aria.engine.policy import minimal_calls, observed_transport as transport
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.transport_failure import transport_failure, validate_transport_failure
from aria.engine.policy.max_release import tls_transport as candidate
from aria.engine.policy.response_body import ResponseDeadline
from tests.test_observed_transport import FakeSocket, response
from tests.test_reserved_calls import call, client, policy, receipt


def continuation_policy():
    return {"schema": "bounded_http429_continuation_v1", "max_extra_dispatches": 1, "max_new_unknown": 1,
        "backoff_seconds": [120], "max_attempts_per_operation": 2, "max_logical_requests": 1246,
        "authorization_sha256": "a" * 64}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    blocked = lambda *a, **kw: pytest.fail("real network forbidden")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})


class TimedSocket(FakeSocket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


def connect_fixture(actions, *, clock=None, trace=None, check=lambda: None, tunnel=False,
                    body=b"{}", status=b"200 OK", write_failure=None, budget=None,
                    connection_class=transport._ObservedHTTPSConnection):
    clock = clock if clock is not None else [0.0]
    budget = budget if budget is not None else {"attempts": 0}
    trace = trace or transport.TransportTrace(diagnostics=True, preparation_recovery=transport.PreparationRecovery(
        check=check, timeout=120, budget=budget, monotonic=lambda: clock[0]))
    sockets, connection_timeouts = [], []
    class TLS:
        # Python 3.10 checks these SSLContext attributes during construction.
        verify_mode = ssl.CERT_REQUIRED
        check_hostname = True

        def wrap_socket(self, sock, *, server_hostname):
            error, elapsed = actions.pop(0)
            clock[0] += elapsed
            if error is not None:
                raise error
            return sock
    conn = connection_class("fixture.invalid", trace=trace, context=TLS())
    def created(*args, **kwargs):
        connection_timeouts.append(conn.timeout)
        responses = [b"HTTP/1.1 200 Connection established\r\n\r\n"] if tunnel else []
        responses.append(response(body, status=status))
        sock = TimedSocket(trace, responses, write_failure=write_failure)
        sockets.append(sock)
        return sock
    conn._create_connection = created
    if tunnel:
        conn.set_tunnel("fixture.invalid", headers={"Proxy-Authorization": "PRIVATE"})
    return conn, trace, sockets, connection_timeouts, clock


def ssl_error(cls=ssl.SSLError, reason="UNEXPECTED_EOF_WHILE_READING", number=1):
    error = cls(number, "PRIVATE certificate/address/secret")
    error.reason, error.library = reason, "SSL"
    return error


def candidate_fixture(actions, *, timeout=90, check=lambda: None, on_sleep=lambda: None,
                      oversleep=0, write_failure=None):
    clock, sleeps = [0.0], []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds + oversleep
        on_sleep()
    preparation = candidate.TLSPreparation(check=check,
        deadline=ResponseDeadline(timeout, clock=lambda: clock[0]),
        allowance=candidate.LogicalTLSAllowance(), sleep=sleep)
    preparation.bind_request_headers(object())
    trace = candidate.CandidateTrace(preparation=preparation, deadline=preparation.deadline)
    conn, _, sockets, timeouts, _ = connect_fixture(actions, clock=clock, trace=trace,
        tunnel=True, write_failure=write_failure, connection_class=candidate._HTTPSConnection)
    return conn, trace, sockets, timeouts, clock, sleeps


@pytest.mark.parametrize("timeout", [90, 240])
@pytest.mark.parametrize("error", [ssl_error(ssl.SSLEOFError), ssl_error()])
def test_candidate_tls_rebuild_waits_once_within_original_deadline(timeout, error):
    conn, trace, sockets, timeouts, clock, sleeps = candidate_fixture(
        [(error, 2), (None, 3)], timeout=timeout)
    try:
        conn.request("POST", "/fixture", body=b"synthetic")
        assert conn.getresponse().read() == b"{}"
        transport.validate_trace(trace.snapshot(), require_complete=True)
        assert sleeps == [1.0] and clock[0] == 6
        assert timeouts == [timeout, timeout - 3]
        assert sockets[0].closed and len(sockets[0].writes) == 1
        assert sockets[0].writes[0][0].startswith(b"CONNECT ")
        assert sum(data.startswith(b"POST ") for sock in sockets for data, _ in sock.writes) == 1
        assert trace.preparation_recovery.receipt()["extra_consumed_in_physical"] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("failure_time", [89, 89.5])
def test_candidate_no_rebuild_when_deadline_cannot_cover_pause(failure_time):
    conn, trace, sockets, _, _, sleeps = candidate_fixture(
        [(ssl_error(ssl.SSLEOFError), failure_time), (None, 0)])
    try:
        with pytest.raises(ssl.SSLEOFError):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [] and len(sockets) == 1 and sockets[0].closed
        assert trace.preparation_recovery.allowance.used == 0
        transport.validate_trace(trace.snapshot())
    finally:
        conn.close()


@pytest.mark.parametrize("change", ["cancel", "deadline"])
def test_candidate_wait_rechecks_permission_and_original_deadline(change):
    cancelled = [False]
    def check():
        if cancelled[0]:
            raise PermissionError("synthetic cancellation")
    conn, trace, sockets, _, _, sleeps = candidate_fixture(
        [(ssl_error(ssl.SSLEOFError), 1), (None, 0)], check=check,
        on_sleep=lambda: cancelled.__setitem__(0, change == "cancel"),
        oversleep=90 if change == "deadline" else 0)
    try:
        with pytest.raises(PermissionError if change == "cancel" else TimeoutError):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [1.0] and len(sockets) == 1 and sockets[0].closed
        assert trace.preparation_recovery.allowance.used == 0
        assert "application_request_started" not in trace.snapshot()["events"]
        transport.validate_trace(trace.snapshot())
    finally:
        conn.close()


def test_candidate_permission_revoked_before_pause_prevents_wait_and_rebuild():
    checks = []
    def check():
        checks.append(True)
        if len(checks) == 2:
            raise PermissionError("synthetic revoked allowance")
    conn, trace, sockets, _, _, sleeps = candidate_fixture(
        [(ssl_error(ssl.SSLEOFError), 1), (None, 0)], check=check)
    try:
        with pytest.raises(PermissionError):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [] and len(sockets) == 1 and sockets[0].closed
        assert trace.preparation_recovery.allowance.used == 0
    finally:
        conn.close()


@pytest.mark.parametrize("error", [ssl_error(ssl.SSLCertVerificationError),
    ssl_error(reason="WRONG_VERSION_NUMBER"), TimeoutError("synthetic"),
    ConnectionResetError("synthetic"), http.client.RemoteDisconnected("synthetic")])
def test_candidate_pause_does_not_expand_rebuild_eligibility(error):
    conn, trace, sockets, _, _, sleeps = candidate_fixture([(error, 1), (None, 0)])
    try:
        with pytest.raises(type(error)):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [] and len(sockets) == 1 and sockets[0].closed
        assert trace.preparation_recovery.allowance.used == 0
    finally:
        conn.close()


def test_candidate_second_tls_failure_does_not_wait_or_connect_again():
    conn, trace, sockets, _, _, sleeps = candidate_fixture(
        [(ssl_error(ssl.SSLEOFError), 1), (ssl_error(ssl.SSLEOFError), 1), (None, 0)])
    try:
        with pytest.raises(ssl.SSLEOFError):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [1.0] and len(sockets) == 2 and all(s.closed for s in sockets)
        assert trace.preparation_recovery.allowance.used == 1
        assert all(not data.startswith(b"POST ") for sock in sockets for data, _ in sock.writes)
        transport.validate_trace(trace.snapshot())
    finally:
        conn.close()


def test_candidate_application_write_failure_does_not_pause_or_reconnect():
    # CONNECT is write 1; the application request starts at write 2.
    conn, trace, sockets, _, _, sleeps = candidate_fixture([(None, 1), (None, 0)], write_failure=2)
    try:
        with pytest.raises(BrokenPipeError):
            conn.request("POST", "/fixture", body=b"synthetic")
        assert sleeps == [] and len(sockets) == 1
        assert trace.preparation_recovery.allowance.used == 0
        assert trace.snapshot()["events"]["application_write_attempted"]
    finally:
        conn.close()


@pytest.mark.parametrize("cls,kind", [(ssl.SSLEOFError, "tls_eof_error"),
    (ssl.SSLZeroReturnError, "tls_clean_close"), (ssl.SSLSyscallError, "tls_syscall_error"),
    (ssl.SSLWantReadError, "tls_want_read"), (ssl.SSLWantWriteError, "tls_want_write"),
    (ssl.SSLCertVerificationError, "tls_certificate_error")])
def test_ssl_finite_subtypes_and_legacy_record_are_both_valid(cls, kind):
    error = ssl_error(cls)
    error.verify_code = 10
    legacy = transport_failure(error, opener_invoked=True, response_returned=False)
    detailed = transport_failure(error, opener_invoked=True, response_returned=False, diagnostics=True)
    assert validate_transport_failure(legacy) == legacy
    assert validate_transport_failure(detailed) == detailed
    assert legacy["version"] == "transport_failure_v1" and "tls_details" not in legacy
    assert detailed["exception_kind"] == kind
    assert detailed["tls_details"]["reason_code"] == "UNEXPECTED_EOF_WHILE_READING"
    assert ("verify_code" in detailed["tls_details"]) is (cls is ssl.SSLCertVerificationError)
    assert "PRIVATE" not in json.dumps(detailed)


def test_ssl_private_properties_values_names_and_unknown_numbers_are_not_retained():
    class Hostile(ssl.SSLError):
        @property
        def reason(self):
            raise AssertionError("property read")
        @property
        def library(self):
            raise AssertionError("property read")
        def __str__(self):
            raise AssertionError("stringified")
    Hostile.__name__ = "PRIVATE_CLASS"
    error = Hostile(999999999, "PRIVATE")
    error.__dict__.update(reason="PRIVATE_REASON", library=["PRIVATE"], verify_code=999999999)
    value = transport_failure(error, opener_invoked=True, response_returned=False, diagnostics=True)
    assert validate_transport_failure(value) == value
    assert "PRIVATE" not in json.dumps(value) and "999999999" not in json.dumps(value)
    for key, bad in (("reason_code", "PRIVATE"), ("verify_code", True), ("library_code", [])):
        changed = deepcopy(value)
        changed["tls_details"].update({key: bad, key + "_status": "recognized"})
        with pytest.raises(ValueError):
            validate_transport_failure(changed)


@pytest.mark.parametrize("first", [ssl_error(ssl.SSLEOFError), ssl_error(), ConnectionResetError(10054, "PRIVATE"),
    ConnectionAbortedError(10053, "PRIVATE")])
def test_typed_prepare_failure_rebuilds_once_and_preserves_each_attempt(first):
    conn, trace, sockets, timeouts, _ = connect_fixture([(first, 40), (None, 10)], tunnel=True)
    conn.request("POST", "/fixture", body=b"PRIVATE")
    assert conn.getresponse().read() == b"{}"
    value = trace.snapshot()
    assert transport.validate_trace(value, require_complete=True) == value
    assert timeouts == [120, 80] and sockets[0].closed
    assert sockets[1].timeouts and max(sockets[1].timeouts) == 70
    assert len(sockets[0].writes) == 1  # Proxy CONNECT, no application HTTP.
    assert sockets[0].writes[0][1]["events"] == {"connection_preparation_started": True}
    assert len(value["connection_preparations"]) == 2
    assert value["connection_preparations"][0]["events"]["connection_preparation_failed_before_request"]
    assert "failure" in value["connection_preparations"][0]
    assert "connection_preparation_failed_before_request" not in value["events"]
    assert "PRIVATE" not in json.dumps(value)
    conn.close()


@pytest.mark.parametrize("error", [ssl_error(ssl.SSLCertVerificationError), ssl_error(reason="WRONG_VERSION_NUMBER"),
    ssl_error(reason="PRIVATE"), TimeoutError("PRIVATE"), KeyboardInterrupt(), SystemExit()])
def test_certificate_unknown_tls_timeout_and_process_interrupt_never_rebuild(error):
    conn, trace, sockets, _, _ = connect_fixture([(error, 1), (None, 1)])
    with pytest.raises(type(error)):
        conn.request("POST", "/fixture", body=b"PRIVATE")
    value = trace.snapshot()
    transport.validate_trace(value)
    assert len(sockets) == 1 and sockets[0].closed and not sockets[0].writes
    assert value["events"] == {"connection_preparation_started": True, "connection_preparation_failed_before_request": True}


def test_second_prepare_failure_has_no_third_attempt():
    conn, trace, sockets, _, _ = connect_fixture([(ssl_error(), 1), (ssl_error(ssl.SSLEOFError), 1), (None, 1)])
    with pytest.raises(ssl.SSLError):
        conn.request("POST", "/fixture", body=b"PRIVATE")
    transport.validate_trace(trace.snapshot())
    assert len(sockets) == 2 and all(sock.closed for sock in sockets)
    assert len(trace.snapshot()["connection_preparations"]) == 2


def test_remote_disconnected_does_not_gain_rebuild_from_reset_base_class():
    conn, trace, sockets, _, _ = connect_fixture([(http.client.RemoteDisconnected("PRIVATE"), 1), (None, 1)])
    with pytest.raises(http.client.RemoteDisconnected):
        conn.request("POST", "/fixture", body=b"PRIVATE")
    value = trace.snapshot()
    transport.validate_trace(value)
    assert len(sockets) == 1 and sockets[0].closed
    assert value["connection_preparations"][0]["failure"]["exception_kind"] == "remote_disconnected"


@pytest.mark.parametrize("at", ["after_failure", "after_success", "deadline"])
def test_callback_or_original_deadline_blocks_rebuild_or_application_write(at):
    clock = [0.0]
    def check():
        if at == "after_failure" and clock[0] >= 1 or at == "after_success" and clock[0] >= 2:
            raise PermissionError("PRIVATE revoked")
    actions = [(ssl_error(), 120 if at == "deadline" else 1), (None, 1)]
    conn, trace, sockets, _, _ = connect_fixture(actions, clock=clock, check=check)
    with pytest.raises((PermissionError, TimeoutError)):
        conn.request("POST", "/fixture", body=b"PRIVATE")
    transport.validate_trace(trace.snapshot())
    assert len(sockets) == (2 if at == "after_success" else 1)
    assert all(not sock.writes for sock in sockets)
    assert "application_request_started" not in trace.snapshot()["events"]
    conn.close()


def test_partial_first_send_never_reconnects_and_does_not_prove_nondelivery():
    conn, trace, sockets, _, _ = connect_fixture([(None, 1), (None, 1)], write_failure=1)
    with pytest.raises(BrokenPipeError):
        conn.request("POST", "/fixture", body=b"PRIVATE")
    value = trace.snapshot()
    transport.validate_trace(value)
    assert len(sockets) == len(sockets[0].writes) == 1
    assert value["events"]["application_request_started"] and value["events"]["application_write_attempted"]
    assert "request_write_completed" not in value["events"]
    conn.close()


def make_guard(path, clock, check=lambda: None):
    g = ReservedCallGuard(path, max_requests=1247, max_observed_cost_usd="1", reservation_policy=policy(),
        min_interval_seconds=10, observe_transport=True, transport_diagnostics=True,
        connection_preparation_recovery=True, pre_application_check=check, recovery_policy=continuation_policy())
    g._pacing_clock = g._recovery_clock = lambda: clock[0]
    g._pacing_sleep = lambda delay: clock.__setitem__(0, clock[0] + delay)
    return g


def test_guard_single_http_with_rebuild_retains_success_wire_cost_and_first_error(tmp_path, monkeypatch):
    clock = [0.0]
    g = make_guard(tmp_path / "guard.sqlite3", clock); model = client(); g.attach(model)
    sent = []
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        conn, _, _, _, _ = connect_fixture([(ssl_error(), 1), (None, 1)], clock=clock, trace=trace,
            body=json.dumps(receipt(model)).encode())
        conn.request("POST", "/fixture", body=request.data)
        return conn.getresponse()
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    assert call(g, model) == "local fixture"
    state = g.snapshot()
    assert state["calls_used"] == model.network_request_count == len(sent) == 1
    assert state["unknown_cost_requests"] == 0
    assert len(state["attempts"][0]["transport_events"]["connection_preparations"]) == 2
    assert state["recovery"]["extra_dispatches"] == 0
    assert json.loads(sent[0])["messages"] == [{"role": "user", "content": "synthetic input"}]
    g.close()


@pytest.mark.parametrize("damage", ["callback", "configuration", "budget", "cancel", "partial_write", "tls"])
def test_guard_rechecks_pending_permission_configuration_budget_and_keeps_unknown_hold(tmp_path, monkeypatch, damage):
    clock = [0.0]
    def check():
        if clock[0] > 0 and damage == "callback":
            raise PermissionError("PRIVATE revoked")
    g = make_guard(tmp_path / "guard.sqlite3", clock, check); model = client(); g.attach(model)
    sent, sockets = [], []
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        failure = ssl_error(ssl.SSLCertVerificationError) if damage == "tls" else ssl_error()
        actions = [(None, 1)] if damage == "partial_write" else [(failure, 1), (None, 1)]
        conn, _, local_sockets, _, _ = connect_fixture(actions, clock=clock, trace=trace,
            write_failure=1 if damage == "partial_write" else None)
        real_wrap = conn._context.wrap_socket
        def changed(sock, **kwargs):
            try:
                return real_wrap(sock, **kwargs)
            finally:
                if damage == "configuration":
                    model.temperature = .9
                elif damage == "budget":
                    g._db.execute("UPDATE settings SET threshold='0.00000001' WHERE id=1")
                elif damage == "cancel":
                    g._stop_requested.set()
        conn._context.wrap_socket = changed
        try:
            conn.request("POST", "/fixture", body=request.data)
            return conn.getresponse()
        finally:
            sockets.extend(local_sockets)
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    with pytest.raises(MinimalCallStopped):
        call(g, model)
    state = g.snapshot()
    assert len(sent) == len(sockets) == 1
    if damage == "budget":
        # An externally corrupted allowance must not yield a trusted snapshot.
        # The original failed attempt and hold remain durable, not zeroed.
        assert state["stop_reason"] == "ledger_unavailable" and state["unknown_cost_requests"] is None
        saved = json.loads(g._db.execute("SELECT record FROM attempts").fetchone()[0])
        assert saved["status"] == "failed" and saved["usage"] == {}
        assert Decimal(saved["reservation"]["reserved_cost_usd"]) > 0
        assert g._db.execute("SELECT COUNT(*) FROM authorized_recoveries").fetchone() == (0,)
    else:
        assert state["unknown_cost_requests"] == 1
        assert Decimal(state["reserved_cost_usd"]) > 0
        assert state["recovery"]["relations"] == []
    with pytest.raises(MinimalCallStopped):
        call(g, model, "never-repeat")
    assert len(sent) == 1
    g.close()


def test_exhausted_connect_budget_does_not_claim_impossible_http429_recovery(tmp_path, monkeypatch):
    clock = [0.0]
    g = make_guard(tmp_path / "guard.sqlite3", clock); model = client(); g.attach(model)
    sent = []
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        body = {"error": {"code": 429, "metadata": {"provider_name": "Alibaba"}}, "usage": {"cost": 0}}
        conn, _, _, _, _ = connect_fixture([(ssl_error(), 1), (None, 1)], trace=trace, clock=clock)
        conn.request("POST", "/fixture", body=request.data)
        conn.getresponse().close()
        import io
        import urllib.error
        raise urllib.error.HTTPError("https://fixture.invalid", 429, "PRIVATE", {}, io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    with pytest.raises(MinimalCallStopped):
        call(g, model)
    state = g.snapshot()
    assert len(sent) == state["calls_used"] == 1 and state["unknown_cost_requests"] == 0
    assert state["recovery"]["relations"] == [] and clock[0] == 2
    g.close()


def test_known429_can_use_second_connection_without_resetting_logical_preparations(tmp_path, monkeypatch):
    clock = [0.0]
    g = make_guard(tmp_path / "guard.sqlite3", clock); model = client(); g.attach(model)
    sent = []
    def opened(request, *, timeout, trace):
        sent.append(request.data)
        conn, _, _, _, _ = connect_fixture([(None, .25)], trace=trace, clock=clock,
            body=json.dumps(receipt(model)).encode())
        conn.request("POST", "/fixture", body=request.data)
        result = conn.getresponse()
        if len(sent) == 1:
            result.close()
            import io
            import urllib.error
            body = {"error": {"code": 429, "metadata": {"provider_name": "Alibaba"}}, "usage": {"cost": 0}}
            raise urllib.error.HTTPError("https://fixture.invalid", 429, "PRIVATE", {}, io.BytesIO(json.dumps(body).encode()))
        return result
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    assert call(g, model) == "local fixture"
    state = g.snapshot()
    assert state["calls_used"] == len(sent) == 2 and state["unknown_cost_requests"] == 0
    assert [row["transport_events"]["connection_preparations"][0]["ordinal"] for row in state["attempts"]] == [1, 2]
    rows = deepcopy(state["attempts"])
    rows[-1]["transport_events"]["connection_preparations"][0]["ordinal"] = 1
    with pytest.raises(ValueError, match="preparation_limit"):
        g.validate_recovery_records(rows, g.recovery_policy, relations=state["recovery"]["relations"], require_complete=True)
    g.close()


def test_explicit_rebuild_requires_callback_and_cannot_migrate_old_observation(tmp_path):
    options = dict(max_requests=1247, max_observed_cost_usd="1", reservation_policy=policy(),
        min_interval_seconds=10, observe_transport=True, transport_diagnostics=True, recovery_policy=continuation_policy())
    with pytest.raises(ValueError, match="invalid_connection_preparation_recovery"):
        ReservedCallGuard(tmp_path / "absent.sqlite3", **options, connection_preparation_recovery=True)
    assert not (tmp_path / "absent.sqlite3").exists()
    path = tmp_path / "old.sqlite3"
    options["transport_diagnostics"] = False
    ReservedCallGuard(path, **options).close()
    before = path.read_bytes()
    options["transport_diagnostics"] = True
    with pytest.raises(ValueError, match="transport_observation_changed"):
        ReservedCallGuard(path, **options, connection_preparation_recovery=True, pre_application_check=lambda: None)
    assert path.read_bytes() == before


def test_trace_validation_preserves_v1_and_rejects_impossible_or_private_v2():
    legacy = transport.TransportTrace().snapshot()
    assert transport.validate_trace(legacy) == legacy
    conn, trace, _, _, _ = connect_fixture([(None, 1)])
    conn.request("POST", "/fixture", body=b"fixture")
    conn.getresponse().close()
    value = trace.snapshot()
    transport.validate_trace(value, require_complete=True)
    for mutation in (lambda v: v["events"].update(connection_preparation_failed_before_request=True),
            lambda v: v["events"].update(PRIVATE=True),
            lambda v: v["connection_preparations"][0].update(ordinal=3),
            lambda v: v["connection_preparations"][0]["events"].update(tls_completed=False)):
        changed = deepcopy(value); mutation(changed)
        with pytest.raises(ValueError):
            transport.validate_trace(changed)
    conn.close()
