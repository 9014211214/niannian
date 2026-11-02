"""Exact DeepSeek route/wire tests with local receipts, never model effects."""
from contextlib import closing
import json
import socket
import sqlite3
import urllib.request

import pytest

from aria.engine.memory.preference_dialogue import confirm_judge_messages, confirm_judge_request_format
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_structured import (
    DEEPSEEK_MODEL, DEEPSEEK_PROVIDER, build_online_structured_route,
    validate_release_online_client,
)
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.runtime import create_confirm_judge_client
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import metadata as flash_metadata
from tests.test_qwen_max_candidate import messages as safety_messages, response as safety_response
from tests.test_reserved_calls import stub


@pytest.fixture(autouse=True)
def no_network_credentials(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("real network and credential access forbidden")
    stub_model_http(monkeypatch, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", denied)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", denied)


def route(provider=DEEPSEEK_PROVIDER, provider_name=None):
    provider_name = provider_name or {"together": "Together", "fireworks": "Fireworks"}.get(provider, provider)
    models, endpoints = flash_metadata()
    models["data"][0].update(id=DEEPSEEK_MODEL, canonical_slug=DEEPSEEK_MODEL + "-fixture")
    endpoints["data"]["id"] = DEEPSEEK_MODEL
    endpoints["data"]["endpoints"][0].update(model_id=DEEPSEEK_MODEL,
        tag=provider, name=provider_name + " | fixture", provider_name=provider_name)
    return build_online_structured_route(json.dumps(models).encode(), json.dumps(endpoints).encode(),
        model=DEEPSEEK_MODEL, provider=provider)


def client(role="online_fast", provider=DEEPSEEK_PROVIDER):
    if role == "confirm_judge":
        return create_confirm_judge_client(client(provider=provider))
    kwargs = {"online_structured_route" if role == "online_fast" else "safety_structured_route": route(provider)}
    value = OpenRouterLLMClient(role=role, api_key="inert-fixture", base_url="https://fixture.invalid/v1",
        model=DEEPSEEK_MODEL, online_return_format="json_schema_v1" if role == "online_fast" else "native",
        max_retries=0, fallback_models=[], timeout=90, bounded_transport=True, **kwargs)
    if role == "online_fast":
        value.max_completion_tokens = 4096
        value.online_decision_version = "online_turn_decision_v7"
    return value


def messages(c):
    if c.role == "confirm_judge":
        return confirm_judge_messages("聊天先听我说完。", "仅限闲聊。", "行，就这么记。")
    if c.role == "safety_review":
        return safety_messages(c)
    return [{"role": "system", "content": "协议版本：online_turn_decision_v7"},
            {"role": "user", "content": "中性软件输入。"}]


def response(c):
    if c.role == "safety_review":
        value = safety_response(c)
    else:
        body = {"action": "confirm"} if c.role == "confirm_judge" else {
            "response_segments": [{"text": "本地软件答复。"}], "safety_concern": "none"}
        value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]}
    value.update(id="gen-deepseekfixture123456", model=DEEPSEEK_MODEL,
        provider=validate_release_online_client(c)["provider_name"],
        usage={"cost": .00001, "prompt_tokens": 20, "completion_tokens": 20,
               "total_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 0}})
    return value


def policy(provider=DEEPSEEK_PROVIDER):
    binding = route(provider).binding()
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {DEEPSEEK_MODEL: {
        "input_usd_per_million": "0.15", "output_usd_per_million": "0.6", "provider": provider,
        "response_models": [DEEPSEEK_MODEL, binding["canonical_slug"]], "response_providers": [binding["provider_name"]],
        "max_input_tokens": 100000, "max_output_tokens": 4096,
        "metadata_sha256": binding["endpoint_metadata_sha256"]}}}


def guard(path, budget="1", provider=DEEPSEEK_PROVIDER):
    return ReservedCallGuard(path, max_requests=5, max_observed_cost_usd=budget, reservation_policy=policy(provider))


@pytest.mark.parametrize("role,limit", [("online_fast", 4096), ("safety_review", 1024), ("confirm_judge", 60)])
@pytest.mark.parametrize("provider", ["together", "fireworks"])
def test_actual_guard_wire_fee_and_disabled_reasoning(tmp_path, monkeypatch, role, limit, provider):
    c = client(role, provider)
    path = tmp_path / "ledger.sqlite3"
    def pending(request):
        with sqlite3.connect(path) as db:
            row = json.loads(db.execute("SELECT record FROM attempts").fetchone()[0])
        assert row["status"] == "pending" and row["reservation"]["output_token_limit"] == limit
        assert row["reservation"]["input_token_plan_bound"] == len(request.data) + 8192
        assert float(row["reservation"]["reserved_cost_usd"]) > 0
        assert "billable_completion_limit" not in row["reservation"]
    calls = stub(monkeypatch, response(c), pending)
    with closing(guard(path, provider=provider)) as g:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            c.complete(messages(c))
        state = g.snapshot()
        row, = state["attempts"]
        assert row["status"] == "received" and row["receipt_identity"]["provider_status"] == "recognized"
        assert state["known_cost_usd"] == "0.00001" and state["reserved_cost_usd"] == "0"
        assert not state["stopped"] and len(calls) == 1
    wire, = calls
    assert wire["model"] == DEEPSEEK_MODEL and wire["max_tokens"] == limit
    assert wire["reasoning"] == {"enabled": False, "exclude": True}
    assert wire["provider"] == {"only": [provider], "allow_fallbacks": False,
        "require_parameters": True, "data_collection": "deny", "max_price": {"prompt": .15, "completion": .6}}
    assert wire["response_format"]["type"] == "json_schema"
    if role == "confirm_judge":
        assert wire["temperature"] == 0 and wire["messages"] == messages(c)
        assert wire["response_format"] == confirm_judge_request_format()
    elif role == "online_fast":
        assert wire["response_format"]["json_schema"]["schema"]["required"] == ["response_segments", "safety_concern"]


@pytest.mark.parametrize("role", ["online_fast", "safety_review", "confirm_judge"])
def test_environment_cannot_enable_reasoning_for_deepseek(monkeypatch, role):
    monkeypatch.setenv("OPENROUTER_" + role.upper() + "_REASONING_EFFORT", "high")
    c = client(role)
    assert c.reasoning_effort == "none"
    assert validate_release_online_client(c)["provider"] == DEEPSEEK_PROVIDER


@pytest.mark.parametrize("field,value", [("model", "deepseek/deepseek-v4.1-flash"),
    ("max_completion_tokens", 61), ("max_completion_tokens", True), ("temperature", 1),
    ("temperature", False), ("reasoning_effort", "minimal"), ("max_retries", 1),
    ("fallback_models", ["qwen/qwen3.8-max-0902"]), ("timeout", 91), ("timeout", True),
    ("bounded_transport", False), ("online_return_format", "json_object")])
def test_changed_confirmation_contract_rejected_before_dispatch(field, value):
    c = client("confirm_judge")
    setattr(c, field, value)
    with pytest.raises(ValueError, match="invalid_release_online_client"):
        validate_release_online_client(c)
    assert c.network_request_count == 0


@pytest.mark.parametrize("role", ["online_fast", "safety_review", "confirm_judge"])
def test_other_provider_not_silently_accepted(role):
    c = client(role)
    attribute = {"online_fast": "online_structured_route", "safety_review": "safety_structured_route",
                 "confirm_judge": "confirm_structured_route"}[role]
    setattr(c, attribute, route("deepseek"))
    with pytest.raises(ValueError, match="invalid_release_online_client"):
        validate_release_online_client(c)


@pytest.mark.parametrize("role", ["online_fast", "safety_review", "confirm_judge"])
@pytest.mark.parametrize("provider,name", [("together", "Fireworks"), ("fireworks", "Together")])
def test_selected_provider_cannot_claim_other_accepted_provider_name(role, provider, name):
    c = client(role, provider)
    attribute = {"online_fast": "online_structured_route", "safety_review": "safety_structured_route",
                 "confirm_judge": "confirm_structured_route"}[role]
    setattr(c, attribute, route(provider, provider_name=name))
    with pytest.raises(ValueError, match="invalid_release_online_client"):
        validate_release_online_client(c)
    assert c.network_request_count == 0


@pytest.mark.parametrize("mutation", ["reasoning", "provider", "max_tokens", "schema"])
def test_changed_actual_confirmation_wire_rejected_before_claim(tmp_path, monkeypatch, mutation):
    c = client("confirm_judge")
    with closing(guard(tmp_path / "ledger.sqlite3")) as g:
        g.attach(c)
        captured = []
        def capture(request):
            captured.append(request)
            return response(c)
        monkeypatch.setattr(c, "_request_json", capture)
        c.complete(messages(c))
        request, = captured
        wire = json.loads(request.data)
        if mutation == "reasoning": wire["reasoning"] = {"effort": "minimal", "exclude": True}
        if mutation == "provider": wire["provider"]["only"] = ["alibaba"]
        if mutation == "max_tokens": wire["max_tokens"] = 4096
        if mutation == "schema": wire["response_format"] = {"type": "json_object"}
        changed = urllib.request.Request(request.full_url, data=json.dumps(wire).encode())
        with g.scope("turn", "tampered", "session"):
            with pytest.raises(MinimalCallStopped, match="configuration_changed"):
                g.request_json(c, changed)
        assert g.snapshot()["calls_used"] == 0


@pytest.mark.parametrize("mutation", ["unknown_fee", "reasoning_used", "budget"])
def test_confirm_unknown_fee_and_reasoning_violation_remain_visible(tmp_path, monkeypatch, mutation):
    c = client("confirm_judge")
    data = response(c)
    if mutation == "unknown_fee": del data["usage"]["cost"]
    if mutation == "reasoning_used": data["usage"]["completion_tokens_details"]["reasoning_tokens"] = 20
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger.sqlite3", "0.000001" if mutation == "budget" else "1")) as g:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            with pytest.raises((ValueError, MinimalCallStopped)):
                c.complete(messages(c))
        state = g.snapshot()
        assert len(calls) == (0 if mutation == "budget" else 1)
        if mutation != "budget":
            assert state["stopped"]
        else:
            assert state["calls_used"] == 0
        if mutation == "unknown_fee":
            assert state["known_cost_usd"] is None and float(state["reserved_cost_usd"]) > 0


@pytest.mark.parametrize("role", ["online_fast", "safety_review", "confirm_judge"])
@pytest.mark.parametrize("field", ["reasoning", "reasoning_details", "reasoning_content"])
def test_all_disabled_reasoning_fields_rejected_in_guard_without_retaining_text(tmp_path, monkeypatch, role, field):
    c = client(role)
    data = response(c)
    marker = "synthetic-forbidden-field-marker"
    data["choices"][0]["message"][field] = [{"text": marker}] if field == "reasoning_details" else marker
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger.sqlite3")) as g:
        g.attach(c)
        with g.scope("turn", "one", "session"):
            with pytest.raises(MinimalCallStopped):
                c.complete(messages(c))
        state = g.snapshot()
        row, = state["attempts"]
        assert len(calls) == 1 and row["status"] == "failed"
        assert row["validation_error"] == "disabled_reasoning_returned"
        assert marker not in json.dumps(state)
        assert marker not in json.dumps(c.last_diagnostic)
        assert marker not in json.dumps(c.last_usage)
