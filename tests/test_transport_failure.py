"""Synthetic exceptions only; categories never imply non-dispatch or no charge."""
from tests._model_http_stub import stub_model_http
from decimal import Decimal
import errno
import http.client
import json
import socket
import ssl
import urllib.error
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped, _record_failure_stage
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.transport_failure import transport_failure
from tests.test_reserved_calls import call, client, guard, policy, receipt, stub


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail("unstubbed network"))


ERRORS = [
    (urllib.error.URLError(socket.gaierror(-2, "PRIVATE")), "name_resolution_error", "not_applicable"),
    (urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "PRIVATE")), "connection_refused", "connection_refused"),
    (urllib.error.URLError(ssl.SSLError(1, "PRIVATE")), "tls_error", "not_applicable"),
    (urllib.error.URLError(BrokenPipeError(32, "PRIVATE")), "broken_pipe", "broken_pipe"),
    (http.client.RemoteDisconnected("PRIVATE"), "remote_disconnected", "not_available"),
    (ValueError("PRIVATE"), "other_exception", "not_available"),
    (urllib.error.URLError(TimeoutError(10060, "PRIVATE")), "timeout", "timeout"),
    (ssl.SSLCertVerificationError(1, "PRIVATE"), "tls_certificate_error", "not_applicable"),
]


@pytest.mark.parametrize("error,kind,number", ERRORS, ids=[f"category-{n}" for n in range(len(ERRORS))])
def test_future_first_failure_classification_keeps_fee_unknown_and_never_retries(tmp_path, monkeypatch, error, kind, number):
    current, model = guard(tmp_path / "calls.sqlite3"), client()
    current.attach(model)
    sent = []
    def failed(*args, **kwargs):
        sent.append(1)
        raise error
    stub_model_http(monkeypatch, failed)
    with pytest.raises(MinimalCallStopped):
        call(current, model)
    state = current.snapshot()
    row = state["attempts"][0]
    assert row["failure_phase"] == "pre_response_open" and row["failure_code"] == "response_open_failed"
    assert row["transport_failure"]["exception_kind"] == kind
    assert row["transport_failure"]["errno_category"] == number
    assert row["transport_failure"]["opener_call"] == "invoked"
    assert row["transport_failure"]["application_write"] == "unobserved"
    assert row["transport_failure"]["billing_inference"] == "none"
    assert row["usage"] == {} and row["observation"] == {}
    assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
    assert "PRIVATE" not in json.dumps(state)
    with pytest.raises(MinimalCallStopped):
        call(current, model, "no-retry")
    assert len(sent) == 1
    current.close()


@pytest.mark.parametrize("boundary", ["bookkeeping", "late_stop", "late_pacing"])
def test_failure_before_opener_is_explicit_without_releasing_reserve(tmp_path, monkeypatch, boundary):
    current, model = guard(tmp_path / "calls.sqlite3"), client()
    if boundary == "late_pacing":
        current.close()
        current = ReservedCallGuard(tmp_path / "paced.sqlite3", max_requests=30, max_observed_cost_usd="1",
            reservation_policy=policy(), min_interval_seconds=10)
        current._pacing_clock = lambda: 0
        current._pacing_sleep = lambda _: pytest.fail("first request needs no wait")
    current.attach(model)
    before = current._begin
    def changed(*args, **kwargs):
        row = before(*args, **kwargs)
        if boundary == "late_stop":
            current._stop_requested.set()
        elif boundary == "late_pacing":
            current._pacing_clock = lambda: float("nan")
        return row
    monkeypatch.setattr(current, "_begin", changed)
    if boundary == "bookkeeping":
        monkeypatch.setattr(model, "_record_network_request", lambda: (_ for _ in ()).throw(ValueError("PRIVATE")))
    with pytest.raises(MinimalCallStopped):
        call(current, model)
    state = current.snapshot()
    row = state["attempts"][0]
    expected = {"bookkeeping": "local_bookkeeping", "late_stop": "local_stop_check", "late_pacing": "local_pacing_check"}
    assert row["failure_phase"] == expected[boundary]
    assert row["transport_failure"]["opener_call"] == "not_invoked"
    assert row["transport_failure"]["application_write"] == "unobserved"
    assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
    current.close()


def test_returned_response_context_failure_is_not_claimed_as_connect_failure(tmp_path, monkeypatch):
    current, model = guard(tmp_path / "calls.sqlite3"), client()
    current.attach(model)
    class BrokenContext:
        def __enter__(self):
            raise RuntimeError("PRIVATE")
        def __exit__(self, *args):
            return False
    stub_model_http(monkeypatch, lambda *a, **kw: BrokenContext())
    with pytest.raises(MinimalCallStopped):
        call(current, model)
    row = current.snapshot()["attempts"][0]
    assert row["failure_phase"] == "response_context"
    assert row["transport_failure"]["response_object"] == "returned"
    current.close()


def test_success_has_original_reply_wire_and_no_diagnostic_fields(tmp_path, monkeypatch):
    current, model = guard(tmp_path / "calls.sqlite3"), client()
    current.attach(model)
    sent = stub(monkeypatch, receipt(model))
    assert call(current, model) == "local fixture"
    row = current.snapshot()["attempts"][0]
    assert "transport_failure" not in row and "failure_phase" not in row and "failure_code" not in row
    assert sent[0]["messages"] == [{"role": "user", "content": "synthetic input"}]
    assert len(sent) == 1 and current.snapshot()["unknown_cost_requests"] == 0
    current.close()


def test_historical_connect_failure_row_is_read_unchanged_and_stays_stopped(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    current, model = guard(path), client()
    current.attach(model)
    stub(monkeypatch, receipt(model))
    call(current, model)
    row = current.snapshot()["attempts"][0]
    row.update(status="failed", usage={}, observation={}, failure_phase="connect", failure_code="connection_failed")
    current._finish(row)
    current.close()
    before = path.read_bytes()
    reopened = guard(path)
    state = reopened.snapshot()
    assert state["attempts"] == [row]
    assert "transport_failure" not in row
    assert state["stopped"] and state["unknown_cost_requests"] == 1
    reopened.close()
    assert path.read_bytes() == before
    old = {}
    _record_failure_stage(old, "connect", ValueError("PRIVATE"))
    assert old == {"failure_phase": "connect", "failure_code": "connection_failed"}


def test_malicious_exception_text_names_properties_and_cycles_are_not_retained():
    class PrivateException(Exception):
        def __str__(self):
            raise AssertionError("must not stringify")
        def __repr__(self):
            raise AssertionError("must not repr")
    PrivateException.__name__ = "PRIVATE_CLASS"
    class BrokenReason(urllib.error.URLError):
        @property
        def reason(self):
            raise ValueError("PRIVATE_REASON")
        @reason.setter
        def reason(self, _):
            pass
    cycle = urllib.error.URLError("PRIVATE_CYCLE")
    cycle.reason = cycle
    deep = PrivateException("PRIVATE_SECRET")
    for _ in range(10):
        deep = urllib.error.URLError(deep)
    exceptions = [PrivateException("PRIVATE_SECRET"), urllib.error.URLError("https://PRIVATE/key"),
                  BrokenReason("PRIVATE"), cycle, deep, OSError(999999999, "PRIVATE_ERRNO")]
    for error in exceptions:
        value = transport_failure(error, opener_invoked=True, response_returned=False)
        assert "PRIVATE" not in json.dumps(value) and "999999999" not in json.dumps(value)
        assert value["application_write"] == value["tls_handshake"] == "unobserved"
    assert transport_failure(cycle, opener_invoked=True, response_returned=False)["reason_status"] == "cycle"
    assert transport_failure(deep, opener_invoked=True, response_returned=False)["reason_status"] == "depth_limit"


@pytest.mark.parametrize("boundary", [(None, False), (True, 1), (False, True)])
def test_invalid_boundary_is_rejected(boundary):
    with pytest.raises(ValueError, match="transport_boundary"):
        transport_failure(ValueError(), opener_invoked=boundary[0], response_returned=boundary[1])
