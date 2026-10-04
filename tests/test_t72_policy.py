"""No-network checks of T7.2 actual wire support and immutable SHA binding."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import t7_transport, t72_policy as policy


ROOT = Path(__file__).resolve().parents[1]


def payload(model="qwen/qwen3.7-flash"):
    qwen = model.startswith("qwen/")
    result = {"model": model, "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 1024 if qwen else 512, "stream": False,
        "provider": {"only": ["alibaba" if qwen else "anthropic"],
            "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False, "exclude": True}}
    if qwen:
        result.update(temperature=0.2, parallel_tool_calls=False,
            tools=[{"type": "function", "function": {"name": "aria_turn_decision"}}],
            tool_choice={"type": "function", "function": {"name": "aria_turn_decision"}})
    return result


class Response:
    def __init__(self, value, url):
        self.body = io.BytesIO(t7_transport._json_bytes(value))
        self.url, self.headers = url, {}

    def read1(self, size):
        return self.body.read(size)

    def close(self):
        self.body.close()

    def geturl(self):
        return self.url

    def getcode(self):
        return 200


@pytest.fixture
def metadata():
    frozen = json.loads((ROOT / "experiments/t6_20260930/public_metadata.json").read_text(encoding="utf-8"))
    documents = {t7_transport.MODELS_URL: {"data": deepcopy(frozen["models"])}}
    for row in frozen["selected_endpoints"]:
        documents[row["source_url"]] = {"data": deepcopy(row["data"])}
    calls = []

    def opener(request, *, timeout):
        assert request.get_method() == "GET" and request.get_header("Authorization") is None
        assert timeout == 120
        calls.append(request)
        return Response(documents[request.full_url], request.full_url)

    return frozen, documents, calls, opener


def endpoint(documents, model="qwen/qwen3.7-flash"):
    return documents[t7_transport.MODELS_URL + "/" + model + "/endpoints"]["data"]["endpoints"][0]


def check(metadata, wire=None, bound=20000):
    frozen, _, _, opener = metadata
    wire = payload() if wire is None else wire
    return policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
        payload=wire, prompt_token_bound=bound, opener=opener)


def test_false_parallel_field_is_checked_and_blocks_after_two_public_gets(metadata):
    frozen, _, calls, _ = metadata
    wire = payload()
    before = deepcopy((wire, frozen))
    with pytest.raises(policy.TransportError,
                       match="parameter_support_unverified_parallel_tool_calls") as caught:
        check(metadata, wire)
    assert (wire, frozen) == before
    result = caught.value.receipt
    assert result["missing_fields"] == ["parallel_tool_calls"]
    assert result["supported"] is False and not caught.value.outcome_unknown
    assert result["actual_payload_sha256"] == hashlib.sha256(t7_transport._json_bytes(wire)).hexdigest()
    assert len(calls) == 2 and result["inference_requests"] == 0
    assert "ready_to_send" not in result


def test_advertised_parallel_passes_without_changing_any_request_bytes(metadata):
    _, documents, calls, _ = metadata
    endpoint(documents)["supported_parameters"].append("parallel_tool_calls")
    wire = payload()
    before = deepcopy(wire)
    result = check(metadata, wire)
    assert result["supported"] and not result["missing_fields"]
    assert result["actual_wire_field_removed"] is False
    assert wire == before and len(calls) == 2
    assert policy.verify_payload_binding(wire, result) == result["actual_payload_sha256"]


@pytest.mark.parametrize("field", ["tools", "tool_choice", "temperature", "reasoning", "max_tokens"])
def test_each_other_sent_optional_field_still_requires_evidence(metadata, field):
    _, documents, calls, _ = metadata
    endpoint(documents)["supported_parameters"].append("parallel_tool_calls")
    endpoint(documents)["supported_parameters"].remove(field)
    with pytest.raises(policy.TransportError, match="parameter_support_unverified_" + field) as caught:
        check(metadata)
    assert caught.value.receipt["missing_fields"] == [field] and len(calls) == 2


@pytest.mark.parametrize("change,code", [
    ("snapshot", "metadata_snapshot_changed"), ("price", "metadata_price_changed"),
    ("provider", "metadata_provider_changed"), ("route", "metadata_provider_changed"),
    ("limit", "metadata_token_limit"), ("parameters", "invalid_metadata"),
])
def test_original_identity_price_provider_and_limit_gates_remain(metadata, change, code):
    _, documents, calls, _ = metadata
    row = endpoint(documents)
    row["supported_parameters"].append("parallel_tool_calls")
    if change == "snapshot":
        documents[t7_transport.MODELS_URL]["data"][0]["canonical_slug"] += "-changed"
    elif change == "price":
        row["pricing"]["prompt"] = "1"
    elif change == "provider":
        row["provider_name"] = "another_provider"
    elif change == "route":
        row["status"] = 1
    elif change == "limit":
        row["context_length"] = 10
    else:
        row["supported_parameters"] = {"parallel_tool_calls": True}
    with pytest.raises(policy.TransportError, match=code) as caught:
        check(metadata)
    assert caught.value.receipt["supported"] is False and len(calls) == 2


@pytest.mark.parametrize("bound,price,tier", [
    (31999, "0.00000003", 0), (32000, "0.0000001", 32000), (256000, "0.0000002", 256000)])
def test_original_actual_tier_selection_stays_exact(metadata, bound, price, tier):
    endpoint(metadata[1])["supported_parameters"].append("parallel_tool_calls")
    result = check(metadata, bound=bound)
    assert Decimal(result["prompt_price_usd"]) == Decimal(price)
    assert result["tier_min_prompt_tokens"] == tier


def test_anthropic_model_uses_its_original_route_without_parallel_or_temperature(metadata):
    result = check(metadata, payload("anthropic/claude-sonnet-5"))
    assert result["supported"] and result["provider"] == "anthropic"
    assert "parallel_tool_calls" not in result["required_fields"]
    assert "temperature" not in result["required_fields"]
    assert Decimal(result["prompt_price_usd"]) == Decimal("0.000002")


@pytest.mark.parametrize("change,code", [
    ({"private-key-in-field-name": False}, "invalid_payload_fields"),
    ({"parallel_tool_calls": True}, "unfrozen_parallel_tool_calls"),
    ({"parallel_tool_calls": 0}, "unfrozen_parallel_tool_calls"),
    ({"parallel_tool_calls": None}, "unfrozen_parallel_tool_calls"),
    ({"stream": 0}, "unfrozen_route"),
    ({"model": "qwen/another-model"}, "unfrozen_model"),
    ({"provider": {"only": ["alibaba"], "allow_fallbacks": True}}, "unfrozen_route"),
    ({"reasoning": {"enabled": True, "exclude": True}}, "unfrozen_route"),
])
def test_unknown_fields_and_changed_frozen_semantics_fail_before_any_get(metadata, change, code):
    with pytest.raises(policy.TransportError, match=code) as caught:
        check(metadata, payload() | change)
    assert metadata[2] == [] and not caught.value.outcome_unknown
    assert "private-key-in-field-name" not in repr(caught.value.receipt)


def test_parallel_field_cannot_be_removed_for_route_support(metadata):
    wire = payload()
    del wire["parallel_tool_calls"]
    with pytest.raises(policy.TransportError, match="unfrozen_parallel_tool_calls"):
        check(metadata, wire)
    assert metadata[2] == []


@pytest.mark.parametrize("field", ["messages", "tools", "temperature", "max_tokens"])
def test_full_nested_request_fingerprint_changes_invalidate_prior_support(metadata, field):
    endpoint(metadata[1])["supported_parameters"].append("parallel_tool_calls")
    wire = payload()
    result = check(metadata, wire)
    if field == "messages":
        wire[field][0]["content"] = "不同原话"
    elif field == "tools":
        wire[field][0]["function"]["description"] = "changed"
    elif field == "temperature":
        wire[field] = 0
    else:
        wire[field] = 512
    with pytest.raises(policy.TransportError, match="actual_payload_changed"):
        policy.verify_payload_binding(wire, result)


def test_mutation_during_metadata_read_cannot_change_checked_snapshot_or_get_support(metadata):
    frozen, documents, calls, opener = metadata
    endpoint(documents)["supported_parameters"].append("parallel_tool_calls")
    wire = payload()

    def mutating_opener(request, *, timeout):
        wire["messages"][0]["content"] = "changed during GET"
        return opener(request, timeout=timeout)

    with pytest.raises(policy.TransportError, match="actual_payload_changed"):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen, payload=wire,
            prompt_token_bound=20000, opener=mutating_opener)
    assert len(calls) == 2


def test_offline_assessment_lists_all_missing_sent_fields_without_network(metadata, monkeypatch):
    frozen, _, calls, _ = metadata
    monkeypatch.setattr(t7_transport, "_opener", lambda: pytest.fail("network opener created"))
    selected = frozen["selected_endpoints"][0]["data"]["endpoints"][0]
    selected["supported_parameters"].remove("tools")
    before = deepcopy(frozen)
    wire = payload()
    result = policy.assess_payload_support(model=wire["model"], frozen_metadata=frozen,
        payload=wire, prompt_token_bound=20000)
    assert result["missing_fields"] == ["parallel_tool_calls", "tools"]
    assert not result["supported"] and result["network_requests"] == 0
    assert result["assessment_scope"] == "offline_frozen_metadata"
    assert result["inference_requests"] == 0 and not result["authenticated"]
    assert "ready_to_send" not in result and calls == [] and frozen == before


def test_offline_supported_metadata_is_only_static_evidence(metadata):
    frozen = metadata[0]
    wire = payload("anthropic/claude-sonnet-5")
    result = policy.assess_payload_support(model=wire["model"], frozen_metadata=frozen,
        payload=wire, prompt_token_bound=20000)
    assert result["supported"] and result["stop_reason"] is None
    assert result["assessment_scope"] == "offline_frozen_metadata"
    assert not result["provider_hard_token_bound_verified"]
    assert not result["provider_billing_guarantee"] and "ready_to_send" not in result


@pytest.mark.parametrize("fault", ["missing_models", "unknown_parameters", "wrong_provider"])
def test_malformed_static_metadata_fails_closed(metadata, fault):
    frozen = metadata[0]
    if fault == "missing_models":
        del frozen["models"]
    elif fault == "unknown_parameters":
        frozen["selected_endpoints"][0]["data"]["endpoints"][0]["supported_parameters"] = [None]
    else:
        frozen["selected_endpoints"][0]["data"]["endpoints"][0]["tag"] = "other"
    result = policy.assess_payload_support(model=payload()["model"], frozen_metadata=frozen,
        payload=payload(), prompt_token_bound=20000)
    assert not result["supported"] and result["stop_reason"] == "invalid_metadata"
    assert result["network_requests"] == 0


def test_public_metadata_network_requires_explicit_opt_in(metadata, monkeypatch):
    monkeypatch.setattr(t7_transport, "_opener", lambda: pytest.fail("network opener created"))
    with pytest.raises(policy.TransportError, match="network_disabled"):
        policy.fetch_metadata(model=payload()["model"], frozen_metadata=metadata[0],
            payload=payload(), prompt_token_bound=20000)
