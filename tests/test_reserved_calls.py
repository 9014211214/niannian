"""Reservations are verified with local transport, never a live credential."""
from copy import deepcopy
from decimal import Decimal
import json
import sqlite3
import urllib.request

import pytest

from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_ONLINE_FAST_MODEL
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard, validate_reservation_policy


def policy():
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {
        model: {"input_usd_per_million": price, "output_usd_per_million": output,
                "provider": provider, "response_models": [model],
                "response_providers": [provider.title()], "max_input_tokens": 100000,
                "max_output_tokens": 4096, "metadata_sha256": "a" * 64}
        for model, price, output, provider in [
            (DEFAULT_ONLINE_FAST_MODEL, "0.03", "0.13", "alibaba"),
            (DEFAULT_MEMORY_MODEL, "2", "10", "anthropic")]
    }}


def client(role="online_fast"):
    c = OpenRouterLLMClient(api_key="synthetic-placeholder", base_url="https://fixture.invalid/v1",
        model=DEFAULT_ONLINE_FAST_MODEL if role == "online_fast" else DEFAULT_MEMORY_MODEL,
        role=role, max_retries=0, fallback_models=[], bounded_transport=True,
        online_return_format="json_object" if role == "online_fast" else "native")
    c.max_completion_tokens = 128
    return c


def receipt(c, **changes):
    data = {"id": "gen-reservationfixture1234", "model": c.model,
            "provider": "Alibaba" if c.role == "online_fast" else "Anthropic",
            "usage": {"cost": 0.00001, "prompt_tokens": 5, "completion_tokens": 3},
            "choices": [{"finish_reason": "stop", "message": {"content": "local fixture"}}]}
    data.update(changes)
    return data


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("unstubbed network")
    monkeypatch.setattr(urllib.request, "urlopen", denied)


def stub(monkeypatch, data, on_open=None):
    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size=-1):
            raw = json.dumps(data).encode()
            return raw if size < 0 else raw[:size]
    calls = []
    def opened(request, **kwargs):
        calls.append(json.loads(request.data))
        if on_open:
            on_open(request)
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", opened)
    return calls


def guard(path, budget="1", p=None):
    return ReservedCallGuard(path, max_requests=30, max_observed_cost_usd=budget,
                             reservation_policy=p or policy())


def call(g, c, operation="op1"):
    with g.scope("turn" if c.role == "online_fast" else "review", operation, "session1"):
        return c.complete([{"role": "user", "content": "synthetic input"}])


def test_insufficient_reserve_stops_before_any_network_or_claim(tmp_path):
    g = guard(tmp_path / "budget.sqlite3", "0.000001")
    c = client(); g.attach(c)
    with pytest.raises(MinimalCallStopped, match="pre_send_budget_exceeded"):
        call(g, c)
    assert g.snapshot()["calls_used"] == 0
    assert c.network_request_count == 0
    g.close()


@pytest.mark.parametrize("role", ["online_fast", "memory", "reflection", "judge"])
def test_all_roles_commit_reservation_before_post_and_check_wire(tmp_path, monkeypatch, role):
    path = tmp_path / "budget.sqlite3"
    g = guard(path); c = client(role); g.attach(c)
    def check_pending(request):
        with sqlite3.connect(path) as db:
            row = json.loads(db.execute("SELECT record FROM attempts").fetchone()[0])
        assert row["status"] == "pending"
        assert row["reservation"]["input_token_plan_bound"] == len(request.data) + 8192
        assert Decimal(row["reservation"]["reserved_cost_usd"]) > 0
    requests = stub(monkeypatch, receipt(c), check_pending)
    assert call(g, c) == "local fixture"
    wire = requests[0]
    assert wire["max_tokens"] == 128 and "max_completion_tokens" not in wire
    assert wire["reasoning"] == {"enabled": False, "exclude": True}
    if role != "online_fast":
        assert "temperature" not in wire
    assert wire["provider"]["allow_fallbacks"] is False
    assert wire["provider"]["data_collection"] == "deny"
    assert wire["provider"]["max_price"]["prompt"] == (0.03 if role == "online_fast" else 2)
    assert g.snapshot()["reserved_cost_usd"] == "0"
    g.close()


@pytest.mark.parametrize("change,reason", [
    ({"usage": {"prompt_tokens": 5, "completion_tokens": 3}}, "provider_cost_unknown"),
    ({"usage": {"cost": 0.00001}}, "provider_token_usage_missing"),
    ({"usage": {"cost": 0.1, "prompt_tokens": 5, "completion_tokens": 3}}, "reservation_cost_exceeded"),
    ({"usage": {"cost": 0.00001, "prompt_tokens": 100001, "completion_tokens": 3}}, "reservation_token_bound_exceeded"),
    ({"usage": {"cost": 0.00001, "prompt_tokens": 5, "completion_tokens": 129}}, "reservation_token_bound_exceeded"),
    ({"model": "wrong/model"}, "response_model_mismatch"),
    ({"provider": "wrong-provider"}, "response_provider_mismatch"),
    ({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}, "invalid_or_truncated_response"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": "reply", "reasoning": "synthetic hidden"}}]}, "disabled_reasoning_returned"),
])
def test_bad_receipt_persists_actual_fee_and_halts_without_retry(tmp_path, monkeypatch, change, reason):
    path = tmp_path / "budget.sqlite3"
    g = guard(path); c = client(); g.attach(c)
    calls = stub(monkeypatch, receipt(c, **change))
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot()
    assert state["attempts"][0]["validation_error"] == reason
    assert state["attempts"][0]["status"] == "failed"
    assert "synthetic hidden" not in json.dumps(state)
    if "cost" not in change.get("usage", {"cost": 0}):
        assert state["unknown_cost_requests"] == 1
        assert Decimal(state["reserved_cost_usd"]) > 0
    with pytest.raises(MinimalCallStopped):
        call(g, c, "op2")
    assert len(calls) == 1
    g.close()
    g = guard(path)
    assert g.snapshot()["stopped"]
    g.close()


def test_retained_policy_change_or_old_ledger_never_reset(tmp_path):
    path = tmp_path / "reserved.sqlite3"
    guard(path).close()
    changed = policy(); changed["models"][DEFAULT_ONLINE_FAST_MODEL]["input_usd_per_million"] = "0.01"
    with pytest.raises(ValueError, match="retained_reservation_policy_changed"):
        guard(path, p=changed)
    old = tmp_path / "old.sqlite3"
    MinimalCallGuard(old, max_requests=30, max_observed_cost_usd="1").close()
    original = old.read_bytes()
    with pytest.raises(ValueError, match="retained_ledger_has_no_reservation_policy"):
        guard(old)
    assert old.read_bytes() == original


def test_pending_request_is_visible_to_second_connection(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite3"
    g = guard(path); c = client(); g.attach(c)
    other = guard(path); c2 = client(); other.attach(c2)
    def competing(request):
        with pytest.raises(MinimalCallStopped, match="unresolved_request"):
            call(other, c2, "op2")
    calls = stub(monkeypatch, receipt(c), competing)
    call(g, c)
    assert len(calls) == 1
    other.close(); g.close()


@pytest.mark.parametrize("field,value", [("framing_tokens", 0), ("framing_tokens", True), ("schema", "other")])
def test_policy_cannot_silently_weaken_framing(field, value):
    p = policy(); p[field] = value
    with pytest.raises(ValueError):
        validate_reservation_policy(p)


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "-1"])
def test_policy_rejects_invalid_prices(value):
    p = policy(); p["models"][DEFAULT_MEMORY_MODEL]["input_usd_per_million"] = value
    with pytest.raises(ValueError):
        validate_reservation_policy(p)


def test_price_mutation_after_attach_blocks_before_post(tmp_path):
    g = guard(tmp_path / "budget.sqlite3"); c = client(); g.attach(c)
    c.reservation_max_price["prompt"] = 100
    with pytest.raises(MinimalCallStopped, match="model_configuration_changed"):
        call(g, c)
    assert g.snapshot()["calls_used"] == 0
    g.close()


def test_ledger_save_failure_keeps_committed_pending_reserve(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite3"
    g = guard(path); c = client(); g.attach(c)
    stub(monkeypatch, receipt(c))
    monkeypatch.setattr(g, "_finish", lambda row: (_ for _ in ()).throw(OSError("synthetic write failure")))
    with pytest.raises(MinimalCallStopped, match="ledger_result_save_failed"):
        call(g, c)
    g.close()
    restarted = guard(path)
    state = restarted.snapshot()
    assert state["stopped"] and state["unknown_cost_requests"] == 1
    assert Decimal(state["reserved_cost_usd"]) > 0
    restarted.close()
