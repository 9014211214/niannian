"""Shared pacing uses injected clocks and fake HTTP; no real sleeps or keys."""
from tests._model_http_stub import stub_model_http
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
import threading
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.request_pacing import pacing_interval, wait_until
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_minimal_call_guard import _Response, _call, _client
from tests.test_reserved_calls import call as reserved_call, client as reserved_client, policy, receipt, stub


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail("unstubbed network"))


def guard(path, clock, *, interval=10, reserved=False):
    cls = ReservedCallGuard if reserved else MinimalCallGuard
    kwargs = {"reservation_policy": policy()} if reserved else {}
    result = cls(path, max_requests=30, max_observed_cost_usd="1",
                 min_interval_seconds=interval, **kwargs)
    result._pacing_clock, result._pacing_sleep = clock.monotonic, clock.sleep
    return result


@pytest.mark.parametrize("value", [True, -1, "10", None, float("nan"), float("inf"), 3601])
def test_invalid_intervals_rejected_before_ledger_creation(tmp_path, value):
    with pytest.raises(ValueError, match="interval"):
        guard(tmp_path / "bad.sqlite3", Clock(), interval=value)
    assert not (tmp_path / "bad.sqlite3").exists()


def test_pure_wait_rechecks_early_wake_without_sending():
    clock = Clock()
    def early(seconds):
        clock.sleep(seconds / 2 if seconds >= .5 else seconds)
    assert pacing_interval(10) == 10.0
    assert wait_until(2.5, monotonic=clock.monotonic, sleep=early) == 2500.0
    assert clock.now == 2.5 and len(clock.sleeps) > 3


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: -1, lambda: True])
def test_bad_monotonic_clock_rejected(clock):
    with pytest.raises(ValueError, match="pacing_clock"):
        wait_until(10, monotonic=clock, sleep=lambda _: pytest.fail("must not sleep"))


def test_default_zero_does_not_wait_or_change_attempt_contract(tmp_path, monkeypatch):
    clock = Clock(); current = guard(tmp_path / "zero.sqlite3", clock, interval=0)
    current._pacing_clock = lambda: pytest.fail("default needs no pacing clock")
    current._pacing_sleep = lambda _: pytest.fail("default must not sleep")
    client = _client(); current.attach(client)
    stub_model_http(monkeypatch, lambda *a, **kw: _Response())
    _call(current, client, "one"); _call(current, client, "two")
    assert all("pacing" not in r for r in current.snapshot()["attempts"])
    assert current._db.execute("SELECT name FROM sqlite_master WHERE name='request_pacing'").fetchone() is None
    current.close()


def test_all_reserved_roles_share_one_start_interval_and_original_identities(tmp_path, monkeypatch):
    clock = Clock(); current = guard(tmp_path / "roles.sqlite3", clock, reserved=True)
    starts = []; original_operations = []
    for index, role in enumerate(("online_fast", "memory", "reflection", "judge")):
        client = reserved_client(role); current.attach(client)
        requests = stub(monkeypatch, receipt(client), lambda _: starts.append(clock.now))
        operation = f"role-{index}"; original_operations.append(operation)
        assert reserved_call(current, client, operation) == "local fixture"
        assert len(requests) == 1
        assert getattr(client, "benchmark_request_control", None) is None
    rows = current.snapshot()["attempts"]
    assert starts == [0, 10, 20, 30]
    assert [r["operation_id"] for r in rows] == original_operations
    assert [r["pacing"]["wait_ms"] for r in rows] == [0, 10000, 10000, 10000]
    assert [r["pacing"]["start_gap_ms"] for r in rows] == [None, 10000, 10000, 10000]
    assert all(r["elapsed_ms"] < 1000 for r in rows)  # Fake HTTP; wait is measured separately.
    assert current.snapshot()["reserved_cost_usd"] == "0"
    current.close()


def test_concurrent_callers_cannot_race_shared_pacing_or_operations(tmp_path, monkeypatch):
    clock = Clock(); current = guard(tmp_path / "threads.sqlite3", clock)
    starts = []
    clients = [_client(role) for role in ("online_fast", "memory", "reflection")]
    for client in clients:
        current.attach(client)
    def opened(*args, **kwargs):
        starts.append(clock.now)
        return _Response()
    stub_model_http(monkeypatch, opened)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: _call(current, item[1], f"thread-{item[0]}"), enumerate(clients)))
    assert len(results) == 3 and starts == [0, 10, 20]
    assert len({r["operation_id"] for r in current.snapshot()["attempts"]}) == 3
    current.close()


@pytest.mark.parametrize("stop", ["halt", "close"])
def test_stop_from_another_thread_interrupts_wait_before_claim(tmp_path, monkeypatch, stop):
    clock = Clock(); current = guard(tmp_path / "cancel.sqlite3", clock)
    client = _client(); current.attach(client); sends = []
    stub_model_http(monkeypatch, lambda *a, **kw: (sends.append(1) or _Response()))
    _call(current, client, "first")
    waiting = threading.Event()
    def pause(seconds):
        waiting.set()
        assert current._stop_requested.wait(timeout=2)
        clock.sleep(seconds)
    current._pacing_sleep = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        sending = pool.submit(_call, current, client, "second")
        assert waiting.wait(timeout=2)
        stopping = pool.submit(getattr(current, stop))
        with pytest.raises(MinimalCallStopped):
            sending.result(timeout=2)
        stopping.result(timeout=2)
    assert len(sends) == current.snapshot()["calls_used"] == 1
    assert current.snapshot()["unknown_cost_requests"] == 0
    current.close()


def test_second_instance_constructed_while_empty_checks_current_history(tmp_path, monkeypatch):
    path = tmp_path / "instances.sqlite3"; clock = Clock()
    first, second = guard(path, clock), guard(path, clock)
    client_a, client_b = _client(), _client("memory")
    first.attach(client_a); second.attach(client_b)
    starts = []
    stub_model_http(monkeypatch, lambda *a, **kw: (starts.append(clock.now) or _Response()))
    _call(first, client_a, "one"); _call(second, client_b, "two"); _call(first, client_a, "three")
    assert starts == [0, 10, 20]
    assert second.snapshot()["attempts"][1]["pacing"]["wait_ms"] == 10000
    assert first.snapshot()["attempts"][2]["pacing"]["start_gap_ms"] is None
    first.close(); second.close()


def test_ledger_change_during_wait_is_rejected_by_atomic_claim(tmp_path, monkeypatch):
    path = tmp_path / "prefix.sqlite3"; clock = Clock(); first, second = guard(path, clock), guard(path, clock)
    client_a, client_b = _client(), _client("memory")
    first.attach(client_a); second.attach(client_b); starts = []
    stub_model_http(monkeypatch, lambda *a, **kw: (starts.append(clock.now) or _Response()))
    _call(first, client_a, "one")
    inserted = False
    def insert_other_call(seconds):
        nonlocal inserted
        if not inserted:
            inserted = True
            _call(second, client_b, "two")
        clock.sleep(seconds)
    first._pacing_sleep = insert_other_call
    with pytest.raises(MinimalCallStopped, match="pacing_ledger_changed"):
        _call(first, client_a, "three")
    assert len(starts) == first.snapshot()["calls_used"] == 2
    assert first.snapshot()["unknown_cost_requests"] == 0
    first.close(); second.close()


@pytest.mark.parametrize("failure", ["raise", "clock_frozen", "clock_backward", "interrupt"])
def test_wait_failure_stops_without_another_claim_or_send(tmp_path, monkeypatch, failure):
    clock = Clock(); current = guard(tmp_path / "wait.sqlite3", clock)
    client = _client(); current.attach(client); sends = []
    stub_model_http(monkeypatch, lambda *a, **kw: (sends.append(1) or _Response()))
    _call(current, client, "first")
    before = deepcopy(current.snapshot()["attempts"])
    def broken(_):
        if failure == "raise":
            raise RuntimeError("private exception text")
        if failure == "interrupt":
            raise KeyboardInterrupt("private interrupt")
        if failure == "clock_backward":
            clock.now -= 1
    current._pacing_sleep = broken
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else MinimalCallStopped):
        _call(current, client, "second")
    state = current.snapshot()
    assert state["stopped"] and state["attempts"] == before and state["unknown_cost_requests"] == 0
    assert len(sends) == client.network_request_count == 1
    assert "private" not in json.dumps(state)
    current.close()


def test_wait_does_not_hold_db_transaction_and_rechecks_halt_before_claim(tmp_path, monkeypatch):
    clock = Clock(); path = tmp_path / "halt.sqlite3"; current = guard(path, clock)
    client = _client(); current.attach(client); sends = []
    stub_model_http(monkeypatch, lambda *a, **kw: (sends.append(1) or _Response()))
    _call(current, client, "first")
    def stop_during_wait(seconds):
        with sqlite3.connect(path, timeout=.1) as db:
            db.execute("UPDATE settings SET halted='model_or_storage_failed'")
        clock.sleep(seconds)
    current._pacing_sleep = stop_during_wait
    with pytest.raises(MinimalCallStopped, match="model_or_storage_failed"):
        _call(current, client, "second")
    assert current.snapshot()["calls_used"] == len(sends) == 1
    current.close()


def test_unknown_reservation_still_stops_all_roles_without_wait_or_retry(tmp_path, monkeypatch):
    clock = Clock(); current = guard(tmp_path / "unknown.sqlite3", clock, reserved=True)
    online, memory = reserved_client(), reserved_client("memory")
    current.attach(online); current.attach(memory)
    requests = stub(monkeypatch, receipt(online, usage={}))
    with pytest.raises(MinimalCallStopped):
        reserved_call(current, online)
    before = current.snapshot()
    assert before["unknown_cost_requests"] == 1 and float(before["reserved_cost_usd"]) > 0
    with pytest.raises(MinimalCallStopped):
        reserved_call(current, memory, "next")
    assert len(requests) == 1 and not clock.sleeps and current.snapshot() == before
    current.close()


@pytest.mark.parametrize("replacement", [0, 5, 20])
def test_registered_pacing_cannot_change_on_same_ledger(tmp_path, replacement):
    path = tmp_path / "bound.sqlite3"; current = guard(path, Clock()); current.close()
    with pytest.raises(ValueError, match="pacing_changed"):
        guard(path, Clock(), interval=replacement)


def test_reopen_keeps_accounting_and_waits_full_interval_without_old_monotonic(tmp_path, monkeypatch):
    path = tmp_path / "reopen.sqlite3"; clock = Clock(); current = guard(path, clock)
    client = _client(); current.attach(client)
    starts = []
    stub_model_http(monkeypatch, lambda *a, **kw: (starts.append(clock.now) or _Response()))
    _call(current, client, "before")
    before = current.snapshot()["attempts"]; current.close()
    # A new clock origin proves no serialized monotonic value is re-used.
    clock = Clock(); current = guard(path, clock); client = _client(); current.attach(client)
    _call(current, client, "after")
    rows = current.snapshot()["attempts"]
    assert starts == [0, 10] and rows[:1] == before
    assert rows[1]["pacing"]["wait_ms"] == 10000 and rows[1]["pacing"]["start_gap_ms"] is None
    current.close()


def test_existing_unpaced_attempts_cannot_be_reinterpreted_as_paced(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite3"; current = guard(path, Clock(), interval=0)
    client = _client(); current.attach(client)
    stub_model_http(monkeypatch, lambda *a, **kw: _Response())
    _call(current, client); current.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no_pacing_contract"):
        guard(path, Clock())
    assert path.read_bytes() == before
