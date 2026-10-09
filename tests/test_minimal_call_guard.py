"""Persistent live-demo call limits, using synthetic transport only."""
from __future__ import annotations
from tests._model_http_stub import stub_model_http

from contextlib import contextmanager
from decimal import Decimal
import http.client
import io
import json
import sqlite3
import urllib.error
import urllib.request

import pytest

from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped


class _Response:
    status = 200
    headers = {"x-request-id": "req-localfixture1234"}

    def __init__(self, usage=None, *, raw=None, error=None):
        self.raw = raw if raw is not None else json.dumps({
            "id": "gen-localfixture1234", "model": "fixture/model",
            "usage": {"cost": 0.01, "prompt_tokens": 4, "completion_tokens": 2}
            if usage is None else usage,
            "choices": [{"finish_reason": "stop", "message": {"content": "本地测试回复"}}],
        }).encode()
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if self.error is not None:
            raise self.error
        return self.raw if size < 0 else self.raw[:size]


@pytest.fixture(autouse=True)
def _deny_unstubbed_transport(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("a call-guard test attempted unstubbed network access")
    stub_model_http(monkeypatch, denied)


def _client(role="online_fast", *, retries=0, fallbacks=None, model="fixture/model"):
    return OpenRouterLLMClient(
        role=role, api_key="not-a-real-key", base_url="https://fixture.invalid/v1",
        model=model, max_retries=retries, fallback_models=fallbacks or [],
    )


@contextmanager
def _guard(tmp_path, *, count=21, cost="1.00"):
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=count,
                             max_observed_cost_usd=cost)
    try:
        yield guard
    finally:
        guard.close()


def _call(guard, client, operation="turn-1"):
    kind = "turn" if client.role == "online_fast" else "review"
    with guard.scope(kind, operation, "session-1"):
        return client.complete([{"role": "user", "content": "虚构测试输入"}])


def _stub(monkeypatch, *responses):
    calls = []
    remaining = iter(responses)

    def open_response(request, **kwargs):
        calls.append(request)
        value = next(remaining)
        if isinstance(value, BaseException):
            raise value
        return value

    stub_model_http(monkeypatch, open_response)
    return calls


def test_constructor_and_attach_never_contact_provider(tmp_path):
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        assert client.model_request_control is guard
        state = guard.snapshot()
        assert state["configured"] is True
        assert state["calls_used"] == 0
        assert state["unknown_cost_requests"] == 0
        assert not state["stopped"]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "21"])
def test_invalid_request_ceiling_rejected(tmp_path, count):
    with pytest.raises((TypeError, ValueError)):
        MinimalCallGuard(tmp_path / "bad.sqlite3", max_requests=count,
                         max_observed_cost_usd="1")


@pytest.mark.parametrize("cost", ["0", "-1", "NaN", "Infinity", True, "invalid"])
def test_invalid_cost_threshold_rejected(tmp_path, cost):
    with pytest.raises((TypeError, ValueError)):
        MinimalCallGuard(tmp_path / "bad.sqlite3", max_requests=21,
                         max_observed_cost_usd=cost)


@pytest.mark.parametrize("kwargs", [
    {"role": "judge"}, {"retries": 1}, {"fallbacks": ["fixture/fallback"]},
])
def test_attach_rejects_unapproved_role_retry_or_fallback(tmp_path, kwargs):
    with _guard(tmp_path) as guard:
        with pytest.raises((TypeError, ValueError)):
            guard.attach(_client(**kwargs))


def test_network_request_requires_explicit_operation_context(tmp_path):
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        with pytest.raises((MinimalCallStopped, ValueError)):
            client.complete([{"role": "user", "content": "虚构输入"}])
        assert guard.snapshot()["calls_used"] == 0


def test_three_roles_share_network_ceiling_and_record_context(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response(), _Response(), _Response())
    with _guard(tmp_path, count=3) as guard:
        clients = [_client(role) for role in ("online_fast", "memory", "reflection")]
        for client in clients:
            guard.attach(client)
        for index, client in enumerate(clients):
            assert _call(guard, client, f"operation-{index}") == "本地测试回复"
        with pytest.raises(MinimalCallStopped):
            _call(guard, clients[0], "operation-4")
        state = guard.snapshot()
        assert len(requests) == state["calls_used"] == 3
        assert state["stopped"]
        assert [row["role"] for row in state["attempts"]] == [
            "online_fast", "memory", "reflection",
        ]
        assert [row["kind"] for row in state["attempts"]] == ["turn", "review", "review"]
        assert [row["operation_id"] for row in state["attempts"]] == [
            "operation-0", "operation-1", "operation-2",
        ]
        assert all(row["session_id"] == "session-1" for row in state["attempts"])
        assert len({row["request_id"] for row in state["attempts"]}) == 3
        assert Decimal(state["known_cost_usd"]) == Decimal("0.03")


def test_observed_cost_threshold_stops_next_request_and_keeps_overshoot(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response({"cost": 0.03}), _Response({"cost": 0.04}))
    with _guard(tmp_path, cost="0.05") as guard:
        client = _client()
        guard.attach(client)
        _call(guard, client, "first")
        _call(guard, client, "second")
        with pytest.raises(MinimalCallStopped):
            _call(guard, client, "third")
        state = guard.snapshot()
        assert state["calls_used"] == len(requests) == 2
        assert Decimal(state["known_cost_usd"]) == Decimal("0.07")
        assert Decimal(state["max_observed_cost_usd"]) == Decimal("0.05")
        assert state["stopped"]


@pytest.mark.parametrize("bad_usage", [{}, {"cost": None}, {"cost": True},
                                       {"cost": -0.1}, {"cost": "0.01"}])
def test_unknown_cost_blocks_other_roles_even_after_http_success(tmp_path, monkeypatch, bad_usage):
    requests = _stub(monkeypatch, _Response(bad_usage))
    with _guard(tmp_path) as guard:
        memory, reflection = _client("memory"), _client("reflection")
        guard.attach(memory)
        guard.attach(reflection)
        try:
            _call(guard, memory, "review-1")
        except MinimalCallStopped:
            pass
        with pytest.raises(MinimalCallStopped):
            _call(guard, reflection, "review-1")
        state = guard.snapshot()
        assert state["stopped"]
        assert state["unknown_cost_requests"] == 1
        assert len(requests) == state["calls_used"] == 1
        assert "cost" not in state["attempts"][0]["usage"]


def test_explicit_zero_cost_is_known_and_allows_next_request(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response({"cost": 0}), _Response({"cost": 0.01}))
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        _call(guard, client, "zero")
        assert guard.snapshot()["unknown_cost_requests"] == 0
        assert not guard.snapshot()["stopped"]
        _call(guard, client, "next")
        assert len(requests) == 2


@pytest.mark.parametrize("response", [
    urllib.error.URLError("synthetic secret-like error must not be stored"),
    _Response(error=http.client.IncompleteRead(b"private partial output", 20)),
    _Response(raw=b"not json private output"),
    _Response(raw=b"\xff"),
])
def test_transport_failure_stops_all_future_calls_without_retry(tmp_path, monkeypatch, response):
    requests = _stub(monkeypatch, response)
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        with pytest.raises(Exception):
            _call(guard, client)
        with pytest.raises(MinimalCallStopped):
            _call(guard, client, "next")
        state = guard.snapshot()
        assert state["stopped"]
        assert state["calls_used"] == len(requests) == 1
        assert state["unknown_cost_requests"] == 1
        serialized = json.dumps(state)
        assert "private" not in serialized
        assert "secret-like" not in serialized
        assert "not-a-real-key" not in serialized


def test_http_failure_cost_and_ids_retained_but_message_is_not(tmp_path, monkeypatch):
    error = urllib.error.HTTPError(
        "https://fixture.invalid/v1/chat/completions", 429, "private message",
        {"x-request-id": "req-localfixture1234"},
        io.BytesIO(json.dumps({"error": {"code": 429, "message": "private message"},
                               "usage": {"cost": 0.01}}).encode()),
    )
    requests = _stub(monkeypatch, error)
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        with pytest.raises(Exception):
            _call(guard, client)
        with pytest.raises(MinimalCallStopped):
            _call(guard, client, "next")
        state = guard.snapshot()
        assert state["stopped"]
        assert len(requests) == 1
        assert state["unknown_cost_requests"] == 0
        assert Decimal(state["known_cost_usd"]) == Decimal("0.01")
        assert "private message" not in json.dumps(state)


def test_restart_keeps_known_accounting_and_limit(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response({"cost": 0.03}))
    with _guard(tmp_path, count=1) as guard:
        client = _client()
        guard.attach(client)
        _call(guard, client)
    with _guard(tmp_path, count=1) as reopened:
        client = _client()
        reopened.attach(client)
        with pytest.raises(MinimalCallStopped):
            _call(reopened, client, "after-restart")
        state = reopened.snapshot()
        assert state["calls_used"] == len(requests) == 1
        assert Decimal(state["known_cost_usd"]) == Decimal("0.03")


def test_intent_is_persisted_before_transport_and_crash_blocks_restart(tmp_path, monkeypatch):
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)

        def interrupt(request, **kwargs):
            state = guard.snapshot()
            assert state["calls_used"] == 1
            assert state["attempts"][0]["status"] == "pending"
            raise KeyboardInterrupt("synthetic abrupt stop")

        stub_model_http(monkeypatch, interrupt)
        with pytest.raises(KeyboardInterrupt):
            _call(guard, client)
    with _guard(tmp_path) as reopened:
        state = reopened.snapshot()
        assert state["stopped"]
        assert state["calls_used"] == 1
        assert state["unknown_cost_requests"] == 1
        client = _client()
        reopened.attach(client)
        with pytest.raises(MinimalCallStopped):
            _call(reopened, client, "after-crash")


@pytest.mark.parametrize("count,cost", [(22, "1"), (21, "2")])
def test_restart_cannot_silently_replace_registered_limits(tmp_path, count, cost):
    with _guard(tmp_path):
        pass
    with pytest.raises(ValueError):
        MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=count,
                         max_observed_cost_usd=cost)


def test_explicit_halt_survives_restart(tmp_path):
    with _guard(tmp_path) as guard:
        guard.halt()
        assert guard.snapshot()["stopped"]
    with _guard(tmp_path) as reopened:
        client = _client()
        reopened.attach(client)
        assert reopened.snapshot()["stopped"]
        with pytest.raises(MinimalCallStopped):
            _call(reopened, client)
        assert reopened.snapshot()["calls_used"] == 0


def test_snapshot_cannot_mutate_durable_accounting(tmp_path, monkeypatch):
    _stub(monkeypatch, _Response({"cost": 0.03}))
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        _call(guard, client)
        exported = guard.snapshot()
        exported["attempts"][0]["usage"]["cost"] = 0
        exported["calls_used"] = 0
        unchanged = guard.snapshot()
        assert unchanged["calls_used"] == 1
        assert Decimal(unchanged["known_cost_usd"]) == Decimal("0.03")


def test_second_guard_sees_committed_pending_before_network(tmp_path, monkeypatch):
    requests = []
    with _guard(tmp_path, count=2) as first, _guard(tmp_path, count=2) as second:
        first_client, second_client = _client(), _client("memory")
        first.attach(first_client)
        second.attach(second_client)

        def first_response(request, **kwargs):
            requests.append(request)
            pending = second.snapshot()
            assert pending["calls_used"] == 1
            assert pending["attempts"][0]["status"] == "pending"
            assert pending["stopped"]
            with pytest.raises(MinimalCallStopped):
                _call(second, second_client, "concurrent-review")
            assert second_client.network_request_count == 0
            return _Response({"cost": 0.01})

        stub_model_http(monkeypatch, first_response)
        _call(first, first_client, "first-turn")
        assert len(requests) == 1
        for state in (first.snapshot(), second.snapshot()):
            assert state["calls_used"] == 1
            assert state["attempts"][0]["status"] == "received"
            assert state["unknown_cost_requests"] == 0
            assert Decimal(state["known_cost_usd"]) == Decimal("0.01")
            assert not state["stopped"]


def test_pending_insert_failure_never_connects(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response())
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)

        def deny_pending_insert(action, table, *unused):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == "attempts" else sqlite3.SQLITE_OK

        guard._db.set_authorizer(deny_pending_insert)
        with pytest.raises(MinimalCallStopped):
            _call(guard, client)
        assert requests == []
        assert client.network_request_count == 0
        state = guard.snapshot()
        assert state["stopped"] and state["calls_used"] == 0
        assert state["attempts"] == []


def test_result_write_failure_preserves_unknown_pending_after_restart(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response({"cost": 0.01}))
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)

        def deny_result_update(action, table, *unused):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_UPDATE and table == "attempts" else sqlite3.SQLITE_OK

        guard._db.set_authorizer(deny_result_update)
        with pytest.raises(MinimalCallStopped):
            _call(guard, client)
        assert len(requests) == 1
        state = guard.snapshot()
        assert state["stopped"]
        assert state["attempts"][0]["status"] == "pending"
        assert state["unknown_cost_requests"] == 1
        assert state["known_cost_usd"] is None
    with _guard(tmp_path) as reopened:
        client = _client()
        reopened.attach(client)
        with pytest.raises(MinimalCallStopped):
            _call(reopened, client, "after-restart")
        state = reopened.snapshot()
        assert len(requests) == state["calls_used"] == 1
        assert state["attempts"][0]["status"] == "pending"
        assert state["unknown_cost_requests"] == 1
        assert state["known_cost_usd"] is None


def test_duplicate_role_operation_rejected_but_review_roles_share_identity(tmp_path, monkeypatch):
    requests = _stub(monkeypatch, _Response(), _Response())
    with _guard(tmp_path) as guard:
        memory, reflection = _client("memory"), _client("reflection")
        guard.attach(memory)
        guard.attach(reflection)
        _call(guard, memory, "review-once")
        with pytest.raises(MinimalCallStopped):
            _call(guard, memory, "review-once")
        assert len(requests) == 1
        _call(guard, reflection, "review-once")
        state = guard.snapshot()
        assert state["calls_used"] == len(requests) == 2
        assert [row["role"] for row in state["attempts"]] == ["memory", "reflection"]
        assert all(row["operation_id"] == "review-once" for row in state["attempts"])


def test_closed_guard_retains_readable_stopped_snapshot(tmp_path, monkeypatch):
    _stub(monkeypatch, _Response({"cost": 0.03}))
    with _guard(tmp_path) as guard:
        client = _client()
        guard.attach(client)
        _call(guard, client)
        guard.close()
        state = guard.snapshot()
        assert state["stopped"]
        assert state["calls_used"] == 1
        assert Decimal(state["known_cost_usd"]) == Decimal("0.03")
