"""Max transport software fixtures, never live capability or semantic evidence."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import io
import json
import urllib.error

import pytest

from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_REFLECTION_MODEL
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_structured import build_online_structured_route, online_structured_request_format
from aria.engine.policy.qwen_max_structured import (
    MODEL, build_qwen_max_structured_route, validate_qwen_max_client, validate_qwen_max_response,
)
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.safety_structured import safety_request_format
from aria.engine.policy.semantic_delivery import VERSION
from aria.engine.risk.safety_review_v2 import safety_review_messages
from aria.release import create_qwen_max_candidate_clients, create_semantic_candidate_clients
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import metadata as flash_metadata, route as flash_route
from tests.test_reserved_calls import stub
from tests.test_safety_evidence import payload
from tests.test_semantic_delivery_v6 import segment


def metadata():
    models, endpoints = flash_metadata()
    models["data"][0].update(id=MODEL, canonical_slug=MODEL + "-fixture",
        reasoning={"mandatory": True, "supported_efforts": ["minimal", "low", "medium", "high", "xhigh"]})
    endpoints["data"]["id"] = MODEL
    endpoints["data"]["endpoints"][0].update(model_id=MODEL, max_completion_tokens=131072,
        pricing={"prompt": "0.000002", "completion": "0.000006", "input_cache_read": "0.00000025",
                 "input_cache_write": "0.0000025", "discount": 0})
    return models, endpoints


def route(models=None, endpoints=None, **kwargs):
    default = metadata()
    return build_qwen_max_structured_route(json.dumps(models or default[0]).encode(),
        json.dumps(endpoints or default[1]).encode(), **kwargs)


def clients(**kwargs):
    return create_qwen_max_candidate_clients(model_calls_enabled=True, api_key="inert-software-key",
        base_url="https://fixture.invalid/v1", online_structured_route=kwargs.pop("online_structured_route", route()), **kwargs)


def policy(selected):
    binding = selected.online.online_structured_route.binding()
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {MODEL: {
        "input_usd_per_million": "2.5", "output_usd_per_million": "6", "provider": "alibaba",
        "response_models": [MODEL, binding["canonical_slug"]], "response_providers": ["Alibaba"],
        "max_input_tokens": 100000, "max_output_tokens": 8202,
        "metadata_sha256": binding["endpoint_metadata_sha256"]}}}


def messages(client):
    if client.role == "online_fast":
        return [{"role": "system", "content": "协议版本：" + VERSION},
                {"role": "user", "content": "中性软件装配输入。"}]
    evidence = {"concern": "fall", "subject": "owner", "time": "current", "urgency": "immediate",
        "quote": "中性软件装配输入。", "verbatim_source": True, "disposition": "urgent_owner",
        "semantic_verified": False, "conversation_history": []}
    from aria.engine.risk.safety_review_v2 import uses_owner_sources
    return safety_review_messages("中性软件装配输入。", "保留的软件正文。", evidence,
        None, final_protection="软件条件保护。", owner_sources=uses_owner_sources(client))


def response(client):
    answer = payload(response="中性软件正文。")
    answer.update(applied_preference_ids=[], response_segments=[segment("中性软件正文。")])
    if client.role == "safety_review":
        answer = {"risk_assessment": {"concern": "none", "subject": "uncertain", "time": "uncertain",
                  "urgency": "uncertain"}, "conflicts": False, "delete_sentence_ids": []}
    return {"id": "gen-qwenmaxfixture123456", "model": MODEL, "provider": "Alibaba",
        "usage": {"cost": 0.0001, "prompt_tokens": 20, "completion_tokens": 4500,
                  "completion_tokens_details": {"reasoning_tokens": 4096}},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer, ensure_ascii=False)}}]}


def guard(path, selected, *, budget="1", prices=None):
    return ReservedCallGuard(path, max_requests=4, max_observed_cost_usd=budget,
        reservation_policy=prices or policy(selected))


@pytest.fixture(autouse=True)
def deny_network_and_credentials(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr("aria.runtime.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))


def test_explicit_max_factory_is_inert_and_keeps_offline_models_and_original_default():
    selected = clients()
    old = create_semantic_candidate_clients(model_calls_enabled=True, api_key="inert-software-key",
        base_url="https://fixture.invalid/v1", online_structured_route=flash_route())
    assert old.online.model == old.safety_review.model == "qwen/qwen3.8-flash"
    assert selected.online.model == selected.safety_review.model == MODEL
    assert selected.online.online_decision_version == VERSION
    assert selected.memory.model == DEFAULT_MEMORY_MODEL and selected.reflection.model == DEFAULT_REFLECTION_MODEL
    assert selected.reflection.reflection_return_format == "body_json"
    for role in ("memory", "reflection"):
        for field in ("model", "max_completion_tokens", "timeout", "reasoning_effort", "memory_prefix_recovery", "reflection_itemwise_review"):
            assert getattr(getattr(selected, role), field) == getattr(getattr(old, role), field)
    for client in (selected.online, selected.safety_review, selected.memory, selected.reflection):
        assert client.network_request_count == 0 and client.max_retries == 0 and client.fallback_models == []
    for client, visible, billable in ((selected.online, 4096, 8202), (selected.safety_review, 1024, 5130)):
        binding = validate_qwen_max_client(client)
        assert (binding["wire_visible_limit"], binding["billable_completion_limit"]) == (visible, billable)
        assert binding["gateway_mapping_observed"] is False
        assert client.timeout == 90 and client.reasoning_effort == "minimal"


def test_preview_never_reads_key_or_requires_route():
    preview = create_qwen_max_candidate_clients()
    assert preview.model_calls_enabled is False and preview.safety_review is None


@pytest.mark.parametrize("enabled", [None, 0, 1, "true"])
def test_nonboolean_enable_rejected(enabled):
    with pytest.raises(TypeError):
        create_qwen_max_candidate_clients(model_calls_enabled=enabled)


def test_old_route_does_not_accept_mandatory_reasoning_and_new_factory_rejects_flash():
    models, endpoints = metadata()
    with pytest.raises(ValueError, match="endpoint_not_supported"):
        build_online_structured_route(json.dumps(models).encode(), json.dumps(endpoints).encode(),
            model=MODEL, provider="alibaba")
    with pytest.raises(ValueError, match="max_candidate_requires"):
        clients(online_structured_route=flash_route())


@pytest.mark.parametrize("kwargs", [{"model": "qwen/qwen3.8-max"}, {"provider": "alibaba/us-east-1"},
                                    {"effort": "none"}, {"effort": "high"}])
def test_route_cannot_switch_model_provider_or_effort(kwargs):
    with pytest.raises(ValueError, match="unsupported_qwen_max_route"):
        route(**kwargs)


@pytest.mark.parametrize("mutation", ["optional", "missing_effort", "no_schema", "inactive", "duplicate", "short_cap", "bad_price"])
def test_incomplete_public_capability_is_rejected(mutation):
    models, endpoints = metadata()
    endpoint = endpoints["data"]["endpoints"][0]
    if mutation == "optional": models["data"][0]["reasoning"]["mandatory"] = False
    if mutation == "missing_effort": models["data"][0]["reasoning"]["supported_efforts"].remove("minimal")
    if mutation == "no_schema": endpoint["supported_parameters"].remove("structured_outputs")
    if mutation == "inactive": endpoint["status"] = -1
    if mutation == "duplicate": endpoints["data"]["endpoints"].append(deepcopy(endpoint))
    if mutation == "short_cap": endpoint["max_completion_tokens"] = 1024
    if mutation == "bad_price": endpoint["pricing"]["completion"] = "NaN"
    with pytest.raises(ValueError): route(models, endpoints)


@pytest.mark.parametrize("role", ["online", "safety_review"])
def test_actual_local_transport_reserves_reasoning_without_increasing_visible_wire(tmp_path, monkeypatch, role):
    selected = clients(); client = getattr(selected, role)
    calls = stub(monkeypatch, response(client))
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(client)
        with g.scope("turn", "op1", "session1"):
            raw = client.complete(messages(client))
        assert json.loads(raw) == json.loads(response(client)["choices"][0]["message"]["content"])
        row, = g.snapshot()["attempts"]
        bound = validate_qwen_max_client(client)
        assert row["reservation"]["output_token_limit"] == bound["billable_completion_limit"]
        assert row["reservation"]["wire_visible_limit"] == client.max_completion_tokens
        assert row["qwen_max_token_accounting"]["reasoning_tokens"] == 4096
        assert g.snapshot()["reserved_cost_usd"] == "0"
    wire, = calls
    assert wire["model"] == MODEL and wire["max_tokens"] == client.max_completion_tokens
    assert wire["reasoning"] == {"effort": "minimal", "exclude": True}
    assert wire["provider"] == {"only": ["alibaba"], "allow_fallbacks": False,
        "require_parameters": True, "data_collection": "deny", "max_price": {"prompt": 2.5, "completion": 6}}
    assert wire["response_format"] == (online_structured_request_format(VERSION) if role == "online"
                                       else safety_request_format("safety_review_v2"))
    assert not {"models", "tools", "thinking_budget", "max_completion_tokens"} & wire.keys()


@pytest.mark.parametrize("field,value", [("input_usd_per_million", "2.49"), ("output_usd_per_million", "5.9"),
                                        ("max_output_tokens", 4096), ("metadata_sha256", "a" * 64)])
def test_incomplete_reservation_policy_blocks_before_send(tmp_path, field, value):
    selected = clients(); prices = policy(selected); prices["models"][MODEL][field] = value
    with closing(guard(tmp_path / "ledger", selected, prices=prices)) as g:
        with pytest.raises(ValueError): g.attach(selected.online)
        assert g.snapshot()["calls_used"] == 0


def test_missing_guard_and_insufficient_budget_do_not_send(tmp_path):
    selected = clients()
    with pytest.raises(ValueError, match="reserved_receipt_guard"):
        selected.online.complete(messages(selected.online))
    with closing(guard(tmp_path / "ledger", selected, budget="0.000001")) as g:
        g.attach(selected.online)
        with g.scope("turn", "op1", "session1"), pytest.raises(MinimalCallStopped, match="pre_send_budget"):
            selected.online.complete(messages(selected.online))
        assert g.snapshot()["calls_used"] == 0


@pytest.mark.parametrize("mutation", ["missing_reasoning", "boolean_reasoning", "reasoning_gt_total", "reasoning_cap", "visible_cap", "total_cap"])
def test_inconsistent_usage_retains_full_hold_without_counting_unverified_cost(tmp_path, monkeypatch, mutation):
    selected = clients(); client = selected.online; data = response(client); usage = data["usage"]
    if mutation == "missing_reasoning": usage.pop("completion_tokens_details")
    if mutation == "boolean_reasoning": usage["completion_tokens_details"]["reasoning_tokens"] = True
    if mutation == "reasoning_gt_total": usage["completion_tokens"] = 1
    if mutation == "reasoning_cap": usage["completion_tokens_details"]["reasoning_tokens"] = 4107
    if mutation == "visible_cap": usage.update(completion_tokens=4107, completion_tokens_details={"reasoning_tokens": 0})
    if mutation == "total_cap": usage["completion_tokens"] = 8203
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(client)
        with g.scope("turn", "op1", "session1"), pytest.raises(MinimalCallStopped): client.complete(messages(client))
        state = g.snapshot(); row, = state["attempts"]
        assert state["known_cost_usd"] in (None, "0") and Decimal(state["reserved_cost_usd"]) > 0
        assert "cost" not in row["usage"] and row["unverified_cost_usd"] == .0001
        assert row["fee_verification_error"] == "qwen_max_usage_unverified"
        with g.scope("turn", "op2", "session1"), pytest.raises(MinimalCallStopped): client.complete(messages(client))
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["truncated", "bad_schema", "hidden_text"])
def test_response_failure_with_verified_fee_keeps_known_without_double_hold(tmp_path, monkeypatch, mutation):
    selected = clients(); client = selected.online; data = response(client)
    if mutation == "truncated": data["choices"][0]["finish_reason"] = "length"
    if mutation == "bad_schema": data["choices"][0]["message"]["content"] = "{}"
    if mutation == "hidden_text": data["choices"][0]["message"]["reasoning"] = "synthetic hidden fixture"
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(client)
        with g.scope("turn", "op1", "session1"), pytest.raises((ValueError, MinimalCallStopped)):
            client.complete(messages(client))
        state = g.snapshot()
        assert Decimal(state["known_cost_usd"]) == Decimal("0.0001") and state["reserved_cost_usd"] == "0"
        assert "synthetic hidden fixture" not in json.dumps(state) + json.dumps(client.last_diagnostic)
    assert len(calls) == 1


@pytest.mark.parametrize("error_cost", [None, 0])
def test_429_has_one_attempt_no_fallback_and_full_unknown_hold(tmp_path, monkeypatch, error_cost):
    selected = clients(); calls = []
    def limited(request, **kwargs):
        calls.append(request)
        body = None if error_cost is None else io.BytesIO(json.dumps({"error": {"code": 429},
            "usage": {"cost": error_cost}, "model": MODEL, "provider": "Alibaba"}).encode())
        raise urllib.error.HTTPError(request.full_url, 429, "fixture", {}, body)
    stub_model_http(monkeypatch, limited)
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(selected.online)
        with g.scope("turn", "op1", "session1"), pytest.raises(Exception):
            selected.online.complete(messages(selected.online))
        state = g.snapshot()
        assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        assert state["stopped"]
        if error_cost is not None:
            row, = state["attempts"]
            assert row["unverified_cost_usd"] == error_cost and "cost" not in row["usage"]
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["model", "provider", "error", "malformed_error"])
def test_unattributed_cost_zero_does_not_release_reservation(tmp_path, monkeypatch, mutation):
    selected = clients(); client = selected.online; data = response(client)
    data["usage"]["cost"] = 0
    if mutation == "model": data["model"] = "qwen/qwen3.8-flash"
    if mutation == "provider": data["provider"] = "Other"
    if mutation == "error": data["error"] = {"code": 429, "message": "synthetic fixture"}
    if mutation == "malformed_error": data["error"] = "synthetic fixture"
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(client)
        with g.scope("turn", "op1", "session1"), pytest.raises(MinimalCallStopped): client.complete(messages(client))
        state = g.snapshot(); row, = state["attempts"]
        assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        assert "cost" not in row["usage"] and row["unverified_cost_usd"] == 0
        assert state["known_cost_usd"] in (None, "0")
    assert len(calls) == 1


@pytest.mark.parametrize("field,unknown", [("model", True), ("provider", True), ("finish_reason", False)])
def test_duplicate_identity_preserves_hold_but_nonidentity_format_error_keeps_known(tmp_path, monkeypatch, field, unknown):
    selected = clients(); client = selected.online
    raw = json.dumps(response(client))
    value = {"model": MODEL, "provider": "Alibaba", "finish_reason": "stop"}[field]
    member = json.dumps(field) + ": " + json.dumps(value)
    raw = raw.replace(member, member + ", " + member, 1).encode()
    calls = []
    class Response:
        status = 200
        headers = {}
        def __init__(self): self.body = io.BytesIO(raw)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size=-1): return self.body.read(size)
    def receive(request, **kwargs):
        calls.append(request)
        return Response()
    stub_model_http(monkeypatch, receive)
    with closing(guard(tmp_path / "ledger", selected)) as g:
        g.attach(client)
        with g.scope("turn", "op1", "session1"), pytest.raises(MinimalCallStopped): client.complete(messages(client))
        state = g.snapshot(); row, = state["attempts"]
        assert bool(Decimal(state["reserved_cost_usd"])) is unknown
        assert ("cost" not in row["usage"]) is unknown
        if unknown:
            assert row["fee_verification_error"] == "duplicate_response_identity"
        else:
            assert Decimal(state["known_cost_usd"]) == Decimal("0.0001")
    assert len(calls) == 1


@pytest.mark.parametrize("total,reasoning,valid", [(8202, 4096, True), (8202, 4106, True), (8212, 4106, False),
                                                (4106, 0, True), (4107, 0, False)])
def test_ten_token_margin_is_shared_not_doubled(total, reasoning, valid):
    client = clients().online; data = response(client)
    data["usage"].update(completion_tokens=total, completion_tokens_details={"reasoning_tokens": reasoning})
    if valid:
        assert validate_qwen_max_response(client, data)["completion_tokens"] == total
    else:
        with pytest.raises(ValueError, match="usage_invalid"): validate_qwen_max_response(client, data)
