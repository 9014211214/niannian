"""Bound client options must match the actual request before a paid attempt."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import json
import sqlite3
import urllib.request

import pytest

from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.reflection_json import reflection_body_system_prompt
from aria.runtime import create_minimal_clients
from tests.test_online_json import MESSAGES, envelope
from tests.test_minimal_call_guard import _Response, _call, _client, _stub


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


def wire_client(monkeypatch, *, role="online_fast", integrated=True):
    clients = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture",
                                     base_url="https://fixture.invalid/v1")
    client = {"online_fast": clients.online, "memory": clients.memory, "reflection": clients.reflection}[role]
    requests = []
    def capture(request):
        requests.append(request)
        if role == "reflection":
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                {"protocol_version": "2.0", "reviews": [], "changes": []})}}]}
        return envelope() if integrated and role == "online_fast" else {
            "choices": [{"finish_reason": "stop", "message": {"content": "fixture"}}]}
    with monkeypatch.context() as patch:
        patch.setattr(client, "_request_json", capture)
        messages = MESSAGES if integrated else [{"role": "user", "content": "虚构输入"}]
        if role == "reflection":
            messages = [{"role": "system", "content": reflection_body_system_prompt()}, *messages]
        client.complete(messages)
    return client, requests[0]


@pytest.mark.parametrize("mutation", [
    "provider_privacy", "provider_fallback", "provider_route", "temperature", "reasoning",
    "stream", "tools", "tool_choice", "parallel", "response_format", "token_alias",
    "token_float", "token_wrong", "model", "models", "extra_option", "empty_messages",
])
def test_actual_json_wire_drift_rejected_before_claim_and_network(tmp_path, monkeypatch, mutation):
    client, request = wire_client(monkeypatch)
    payload = json.loads(request.data)
    if mutation == "provider_privacy": payload["provider"]["data_collection"] = "allow"
    if mutation == "provider_fallback": payload["provider"]["allow_fallbacks"] = True
    if mutation == "provider_route": payload["provider"]["only"] = ["unapproved-provider"]
    if mutation == "temperature": payload["temperature"] = 1.5
    if mutation == "reasoning": payload["reasoning"] = {"effort": "high", "exclude": False}
    if mutation == "stream": payload["stream"] = True
    if mutation == "tools": payload["tools"] = []
    if mutation == "tool_choice": payload["tool_choice"] = "auto"
    if mutation == "parallel": payload["parallel_tool_calls"] = False
    if mutation == "response_format": payload.pop("response_format")
    if mutation == "token_alias": payload["max_completion_tokens"] = payload.pop("max_tokens")
    if mutation == "token_float": payload["max_tokens"] = float(payload["max_tokens"])
    if mutation == "token_wrong": payload["max_tokens"] += 1
    if mutation == "model": payload["model"] = "fixture/other"
    if mutation == "models": payload["models"] = []
    if mutation == "extra_option": payload["transforms"] = ["middle-out"]
    if mutation == "empty_messages": payload["messages"] = []
    changed = urllib.request.Request(request.full_url, data=json.dumps(payload).encode())
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=20, max_observed_cost_usd="1")
    try:
        guard.attach(client)
        with guard.scope("turn", "attempt", "session"):
            with pytest.raises(MinimalCallStopped):
                guard.request_json(client, changed)
        assert guard.snapshot()["calls_used"] == 0
    finally:
        guard.close()


@pytest.mark.parametrize("attribute,value,payload_field", [
    ("temperature", .7, "temperature"), ("max_completion_tokens", 2048, "max_tokens"),
    ("model", "fixture/other", "model"), ("online_return_format", "native", None),
    ("reasoning_effort", "high", None), ("timeout", 900, None),
])
def test_mutating_client_and_wire_together_cannot_rebind_allowance(tmp_path, monkeypatch, attribute, value, payload_field):
    client, request = wire_client(monkeypatch)
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=20, max_observed_cost_usd="1")
    try:
        guard.attach(client)
        payload = json.loads(request.data)
        setattr(client, attribute, value)
        if payload_field:
            payload[payload_field] = value
        changed = urllib.request.Request(request.full_url, data=json.dumps(payload).encode())
        with guard.scope("turn", "attempt", "session"):
            with pytest.raises(MinimalCallStopped, match="model_configuration_changed"):
                guard.request_json(client, changed)
        assert guard.snapshot()["calls_used"] == 0
    finally:
        guard.close()


@pytest.mark.parametrize("role,integrated", [("online_fast", True), ("online_fast", False), ("memory", False), ("reflection", False)])
def test_actual_unchanged_role_wire_accepted_once(tmp_path, monkeypatch, role, integrated):
    client, request = wire_client(monkeypatch, role=role, integrated=integrated)
    calls = _stub(monkeypatch, _Response())
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=20, max_observed_cost_usd="1")
    try:
        guard.attach(client)
        with guard.scope("turn" if role == "online_fast" else "review", "attempt", "session"):
            guard.request_json(client, request)
        assert len(calls) == guard.snapshot()["calls_used"] == 1
    finally:
        guard.close()


def test_native_full_tool_contract_mutation_rejected(tmp_path, monkeypatch):
    client = _client()
    requests = []
    with monkeypatch.context() as patch:
        def capture(request):
            requests.append(request)
            raise ValueError("fixture stops after wire capture")
        patch.setattr(client, "_request_json", capture)
        with pytest.raises(ValueError): client.complete(MESSAGES)
    payload = json.loads(requests[0].data)
    payload["tools"][0]["function"]["parameters"]["additionalProperties"] = True
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=20, max_observed_cost_usd="1")
    try:
        guard.attach(client)
        with guard.scope("turn", "attempt", "session"):
            with pytest.raises(MinimalCallStopped):
                guard.request_json(client, urllib.request.Request(requests[0].full_url, data=json.dumps(payload).encode()))
        assert guard.snapshot()["calls_used"] == 0
    finally:
        guard.close()


@pytest.mark.parametrize("mutation", ["duplicate_key", "invalid_json", "non_string_role", "non_post"])
def test_ambiguous_or_malformed_wire_is_configuration_failure_not_ledger_failure(tmp_path, monkeypatch, mutation):
    client, request = wire_client(monkeypatch)
    data = request.data
    if mutation == "duplicate_key": data = b'{"temperature":100,' + data[1:]
    if mutation == "invalid_json": data = b'{'
    if mutation == "non_string_role":
        payload = json.loads(data)
        payload["messages"][0]["role"] = []
        data = json.dumps(payload).encode()
    changed = urllib.request.Request(request.full_url, data=data,
                                     method="GET" if mutation == "non_post" else "POST")
    guard = MinimalCallGuard(tmp_path / "calls.sqlite3", max_requests=20, max_observed_cost_usd="1")
    try:
        guard.attach(client)
        with guard.scope("turn", "attempt", "session"):
            with pytest.raises(MinimalCallStopped, match="model_(request_configuration|endpoint)_changed"):
                guard.request_json(client, changed)
        assert guard.snapshot()["calls_used"] == 0
        assert guard.snapshot()["stop_reason"] is None
    finally:
        guard.close()


def test_legacy_binding_rejection_keeps_existing_spend_and_attempts(tmp_path, monkeypatch):
    path = tmp_path / "calls.sqlite3"
    client = _client()
    calls = _stub(monkeypatch, _Response())
    guard = MinimalCallGuard(path, max_requests=20, max_observed_cost_usd="1")
    guard.attach(client)
    _call(guard, client)
    saved = guard.snapshot()
    guard.close()
    del client.model_request_control
    with sqlite3.connect(path) as db:
        binding = json.loads(db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone()[0])
        for key in ("request_contract_version", "role", "online_return_format"):
            binding.pop(key)
        legacy = json.dumps(binding, sort_keys=True, allow_nan=False)
        db.execute("UPDATE roles SET binding=? WHERE role=?", (legacy, client.role))
    reopened = MinimalCallGuard(path, max_requests=20, max_observed_cost_usd="1")
    try:
        with pytest.raises(ValueError, match="configuration differs"):
            reopened.attach(client)
        after = reopened.snapshot()
        assert after["calls_used"] == saved["calls_used"] == len(calls) == 1
        assert after["known_cost_usd"] == saved["known_cost_usd"] == "0.01"
        assert after["attempts"] == saved["attempts"]
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone()[0] == legacy
    finally:
        reopened.close()
