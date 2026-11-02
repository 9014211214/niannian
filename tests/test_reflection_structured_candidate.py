"""Synthetic transport and strict source checks; no model-effect evidence."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from aria.engine.policy.reflection_structured import validate_reflection_structured_client
from aria.engine.policy.reflection_json import reflection_request_format
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from tests.test_memory_structured_candidate import clients as previous_clients, route, policy
from tests.test_reflection_json_transport import MESSAGES, response, EMPTY
from tests.test_reserved_calls import stub
from tests._model_http_stub import stub_model_http


def client():
    c = previous_clients().reflection
    c.reflection_structured_route = route()
    c.reflection_return_format = "json_schema_v2"
    c.timeout = 90
    c.reasoning_effort = "none"
    return c


def prices(c):
    data = policy(previous_clients().memory)
    data["models"][c.model]["max_output_tokens"] = 2800
    return data


def guard(path, c, reservation=None):
    return ReservedCallGuard(path, max_requests=1, max_observed_cost_usd="1",
                             reservation_policy=reservation or prices(c))


@pytest.fixture(autouse=True)
def forbid_network_and_credentials(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setattr("aria.runtime.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))


def test_explicit_route_binds_exact_reflection_schema_and_limits(tmp_path, monkeypatch):
    c = client()
    calls = stub(monkeypatch, response(c))
    with closing(guard(tmp_path / "ledger.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "reflection-check", "fictional"):
            assert json.loads(c.complete(MESSAGES)) == EMPTY
        assert json.loads(g._binding(c))["reflection_structured_route"] == validate_reflection_structured_client(c)
        assert g.snapshot()["calls_used"] == 1
    wire, = calls
    assert wire["response_format"] == reflection_request_format("json_schema_v2")
    assert wire["messages"] == MESSAGES
    assert wire["provider"]["only"] == ["anthropic"]
    assert wire["provider"]["allow_fallbacks"] is False
    assert wire["reasoning"] == {"enabled": False, "exclude": True}
    assert wire["max_tokens"] == 2800 and c.timeout == 90
    assert not {"tools", "tool_choice", "temperature"} & wire.keys()


@pytest.mark.parametrize("field,value", [
    ("role", "memory"), ("model", "other/model"), ("max_completion_tokens", 2801),
    ("max_retries", 1), ("fallback_models", ["other/model"]), ("timeout", 91),
    ("reflection_return_format", "body_json"), ("reflection_itemwise_review", False),
    ("reasoning_effort", "low"), ("reflection_structured_route", object()),
])
def test_route_configuration_cannot_drift(field, value):
    c = client()
    setattr(c, field, value)
    with pytest.raises(ValueError):
        validate_reflection_structured_client(c)


def test_unguarded_and_unpriced_calls_are_refused(tmp_path):
    c = client()
    with pytest.raises(ValueError, match="candidate_requires_reserved_receipt_guard"):
        c.complete(MESSAGES)
    with closing(MinimalCallGuard(tmp_path / "plain.sqlite", max_requests=1, max_observed_cost_usd=1)) as g:
        with pytest.raises(ValueError, match="candidate_requires_reserved_receipt_guard"):
            g.attach(c)


@pytest.mark.parametrize("field", ["input_usd_per_million", "output_usd_per_million"])
def test_highest_applicable_price_cannot_be_understated(tmp_path, field):
    c = client()
    data = prices(c)
    data["models"][c.model][field] = "1"
    with closing(guard(tmp_path / "under.sqlite", c, data)) as g:
        with pytest.raises(ValueError, match="reservation_reflection_price_understated"):
            g.attach(c)


@pytest.mark.parametrize("raw", [
    '{"protocol_version":"2.0","protocol_version":"2.0","reviews":[],"changes":[]}',
    'preface {"protocol_version":"2.0","reviews":[],"changes":[]}',
    '{"protocol_version":"2.0","reviews":[],"changes":[],"x":NaN}',
])
def test_first_invalid_reply_is_rejected_without_retry_or_correction(tmp_path, monkeypatch, raw):
    c = client()
    calls = stub(monkeypatch, response(c, raw))
    with closing(guard(tmp_path / "invalid.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "reflection-check", "fictional"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        assert g.snapshot()["stopped"] is True
        assert g.snapshot()["known_cost_usd"] == "0.00001"
    assert len(calls) == 1


def test_schema_tamper_is_rejected_before_physical_request(tmp_path, monkeypatch):
    c = client()
    calls = stub(monkeypatch, response(c))
    original = c._request_for_model
    def altered(payload, *, model):
        data = deepcopy(payload)
        data["response_format"]["json_schema"]["strict"] = False
        return original(data, model=model)
    monkeypatch.setattr(c, "_request_for_model", altered)
    with closing(guard(tmp_path / "tamper.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "reflection-check", "fictional"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        assert g.snapshot()["calls_used"] == 0
    assert calls == []


@pytest.mark.parametrize("fault", ["model", "provider", "error_type", "cost", "tokens"])
def test_untrusted_receipt_keeps_reservation_and_stops(fault, tmp_path, monkeypatch):
    c = client()
    data = response(c)
    if fault == "model": data["model"] = "other/model"
    elif fault == "provider": data["provider"] = "Other"
    elif fault == "error_type": data["error"] = "malformed"
    elif fault == "cost": data["usage"].pop("cost")
    elif fault == "tokens": data["usage"]["completion_tokens"] = "3"
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "untrusted.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "reflection-check", "fictional"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        state = g.snapshot()
        assert state["stopped"] is True
        assert state["unknown_cost_requests"] == 1
        assert state["known_cost_usd"] is None
        assert float(state["reserved_cost_usd"]) > 0
    assert len(calls) == 1


@pytest.mark.parametrize("channel", ["reasoning", "reasoning_details", "reasoning_content"])
def test_disabled_reasoning_channels_stop_with_verified_fee_without_retaining_text(channel, tmp_path, monkeypatch):
    c = client()
    data = response(c)
    marker = "SYNTHETIC_PRIVATE_CHANNEL_NOT_RETAINED"
    data["choices"][0]["message"][channel] = marker
    data["usage"]["provider_private_metadata"] = marker
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "channel.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "first-reflection", "fictional"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        state = g.snapshot()
        assert state["stopped"] is True and state["unknown_cost_requests"] == 0
        assert state["known_cost_usd"] == "0.00001" and state["reserved_cost_usd"] == "0"
        with g.scope("review", "second-reflection", "fictional"), pytest.raises(MinimalCallStopped):
            c.complete(MESSAGES)
        assert marker not in json.dumps(c.last_usage)
        assert marker not in json.dumps(c.last_diagnostic)
        assert marker not in json.dumps(state)
    assert len(calls) == 1
    assert all(marker.encode() not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())


def test_successful_strict_reflection_keeps_only_allowlisted_usage(tmp_path, monkeypatch):
    c = client()
    data = response(c)
    data["usage"]["provider_private_metadata"] = "SYNTHETIC_PROVIDER_PRIVATE_DATA"
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "usage.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "first-reflection", "fictional"):
            assert json.loads(c.complete(MESSAGES)) == EMPTY
        assert "provider_private_metadata" not in c.last_usage
    assert len(calls) == 1
