"""T7.3 tests use frozen fictional inventory and in-memory metadata only."""
from copy import deepcopy
from decimal import Decimal

import pytest

from scripts import run_e1_t71 as previous
from scripts import t7_transport, t72_policy, t73_policy as policy


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_and_credentials_forbidden")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


@pytest.fixture(scope="module")
def original(bundle):
    return previous.prepare_e1(bundle)


@pytest.fixture(scope="module")
def revised(bundle):
    return policy.prepare_e1(bundle)


@pytest.fixture
def wire(revised):
    return deepcopy(next(iter(revised[2].values()))["wire"])


@pytest.fixture
def metadata(bundle):
    frozen = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    documents = {t7_transport.MODELS_URL: {"data": deepcopy(frozen["models"])}}
    for row in frozen["selected_endpoints"]:
        documents[row["source_url"]] = {"data": deepcopy(row["data"])}
    calls = []

    def opener(request, *, timeout):
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") is None and timeout == 120
        calls.append(request.full_url)
        return t72_policy._StaticResponse(documents[request.full_url], request.full_url)

    return frozen, documents, calls, opener


def check(metadata, wire):
    return policy.fetch_metadata(model=wire["model"], frozen_metadata=metadata[0],
        payload=wire, prompt_token_bound=25300, opener=metadata[3])


def test_all_204_positions_link_old_and_new_exact_single_field_difference(original, revised):
    assert len(original[3]) == len(revised[3]) == 204
    assert len(original[2]) == len(revised[2]) == 68
    assert len({item["request_sha256"] for item in revised[3]}) == 65
    assert sum((Decimal(item["reserve_usd"]) for item in revised[3]), Decimal(0)) == Decimal("0.2244")
    for old, new in zip(original[3], revised[3]):
        assert new["original_request_sha256"] == old["request_sha256"]
        assert new["request_sha256"] != old["request_sha256"]
        assert new["input_token_plan_bound"] == old["input_token_plan_bound"] - 28
        assert {k: v for k, v in new.items() if k not in {
            "original_request_sha256", "request_sha256", "input_token_plan_bound"}} == {
            k: v for k, v in old.items() if k not in {"request_sha256", "input_token_plan_bound"}}
        key = old["case_id"], old["arm"]
        new_wire, old_wire = revised[2][key]["wire"], original[2][key]["wire"]
        assert "parallel_tool_calls" not in new_wire
        assert {**new_wire, "parallel_tool_calls": False} == old_wire
        assert previous.freeze.canonical_sha256(new_wire) == new["request_sha256"]
        assert new_wire["provider"] == {"only": ["alibaba"], "allow_fallbacks": False,
            "require_parameters": True, "data_collection": "deny"}
        assert new_wire["tool_choice"]["function"]["name"] == "aria_turn_decision"


def test_preparation_does_not_mutate_original_objects_or_return_aliased_wires(bundle, original, monkeypatch):
    old = deepcopy(original)
    before = deepcopy(old)
    monkeypatch.setattr(previous, "prepare_e1", lambda _: old)
    revised = policy.prepare_e1(bundle)
    assert old == before
    trial = next(iter(revised[2].values()))
    trial["wire"]["messages"][0]["content"] = "changed caller copy"
    revised[3][0]["case_id"] = "changed"
    assert old == before


def test_actual_amended_request_passes_saved_support_without_projection(metadata, wire):
    before = deepcopy(wire)
    result = check(metadata, wire)
    assert wire == before and len(metadata[2]) == 2
    assert result["supported"] and result["missing_fields"] == []
    assert "parallel_tool_calls" not in result["required_fields"]
    assert result["actual_payload_sha256"] == previous.freeze.canonical_sha256(wire)
    assert result["original_request_sha256"] == previous.freeze.canonical_sha256(
        {**wire, "parallel_tool_calls": False})
    assert result["single_native_tool_receipt_required"]
    assert "ready_to_send" not in result


def test_all_new_inventory_payloads_are_statically_supported_without_network(metadata, revised):
    by_sha = {}
    for trial in revised[2].values():
        result = policy.assess_payload_support(model=trial["wire"]["model"],
            frozen_metadata=metadata[0], payload=trial["wire"],
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"])
        assert result["supported"] and result["network_requests"] == result["inference_requests"] == 0
        assert result["assessment_scope"] == "offline_frozen_metadata"
        assert "ready_to_send" not in result
        by_sha[result["actual_payload_sha256"]] = result
    assert len(by_sha) == 65 and metadata[2] == []
    assert all(item["request_sha256"] in by_sha for item in revised[3])


@pytest.mark.parametrize("field", ["max_tokens", "reasoning", "temperature", "tool_choice", "tools"])
def test_missing_any_remaining_actual_field_blocks(metadata, wire, field):
    endpoint = metadata[1][t7_transport.MODELS_URL + "/" + wire["model"] + "/endpoints"]["data"]["endpoints"][0]
    endpoint["supported_parameters"].remove(field)
    with pytest.raises(policy.TransportError, match="parameter_support_unverified_" + field) as error:
        check(metadata, wire)
    assert error.value.receipt["missing_fields"] == [field]
    assert error.value.receipt["supported"] is False and len(metadata[2]) == 2


@pytest.mark.parametrize("change", ["messages", "tools", "temperature", "max_tokens", "parallel",
                                  "tool_choice", "model", "privacy", "fallback", "reasoning"])
def test_any_other_wire_change_is_rejected_before_metadata_get(metadata, wire, change):
    if change == "messages":
        wire["messages"][0]["content"] += "changed"
    elif change == "tools":
        wire["tools"][0]["function"]["description"] = "changed"
    elif change == "temperature":
        wire["temperature"] = 0.7
    elif change == "max_tokens":
        wire["max_tokens"] = 512
    elif change == "parallel":
        wire["parallel_tool_calls"] = False
    elif change == "tool_choice":
        wire["tool_choice"] = "auto"
    elif change == "model":
        wire["model"] = "anthropic/claude-sonnet-5"
    elif change == "privacy":
        wire["provider"]["data_collection"] = "allow"
    elif change == "fallback":
        wire["provider"]["allow_fallbacks"] = True
    else:
        wire["reasoning"]["enabled"] = True
    with pytest.raises(policy.TransportError):
        check(metadata, wire)
    assert metadata[2] == []


def test_another_registered_payload_cannot_reuse_first_payload_proof(metadata, wire, revised):
    result = check(metadata, wire)
    other = next(trial["wire"] for trial in revised[2].values() if trial["wire"] != wire)
    with pytest.raises(policy.TransportError, match="actual_payload_changed"):
        policy.verify_payload_binding(other, result)


def test_mutation_during_metadata_read_cannot_change_assessed_payload(metadata, wire):
    def opener(request, *, timeout):
        wire["messages"][0]["content"] += "changed during GET"
        return metadata[3](request, timeout=timeout)

    with pytest.raises(policy.TransportError, match="unregistered_wire_amendment"):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=metadata[0], payload=wire,
            prompt_token_bound=25300, opener=opener)
    assert len(metadata[2]) == 2


def test_changed_original_manifest_cannot_authorize_an_arbitrary_request(metadata, wire, monkeypatch):
    manifest = previous.legacy.read_json(previous.ROOT / previous.legacy.T6_PATH)
    manifest["e1_requests"][0]["request_sha256"] = "0" * 64
    monkeypatch.setattr(previous.legacy, "read_json", lambda _: manifest)
    with pytest.raises(policy.TransportError, match="original_inventory_unverified"):
        check(metadata, wire)
    assert metadata[2] == []


@pytest.mark.parametrize("damage", ["drop_position", "reorder", "coordinate", "other_wire", "swap_wire"])
def test_prepare_rejects_changed_old_inventory_or_wire(bundle, original, monkeypatch, damage):
    old = deepcopy(original)
    if damage == "drop_position":
        old[3].pop()
    elif damage == "reorder":
        old[3][0], old[3][1] = old[3][1], old[3][0]
    elif damage == "coordinate":
        old[3][0]["case_id"] = "changed"
    elif damage == "swap_wire":
        first, second = list(old[2].values())[:2]
        first["wire"], second["wire"] = second["wire"], first["wire"]
    else:
        next(iter(old[2].values()))["wire"]["temperature"] = 0.7
    monkeypatch.setattr(previous, "prepare_e1", lambda _: old)
    with pytest.raises(policy.TransportError):
        policy.prepare_e1(bundle)


def envelope():
    return {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": "0.0001"},
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None,
            "tool_calls": [{"type": "function", "function": {"name": "aria_turn_decision",
                            "arguments": '{"response":"好的。"}'}}]}}]}


@pytest.mark.parametrize("fault,code", [("multi_tool", "invalid_schema"),
    ("wrong_tool", "invalid_schema"), ("prompt_over", "token_bound_exceeded"),
    ("output_over", "token_bound_exceeded")])
def test_receipt_still_requires_one_native_tool_and_observed_usage_bounds(wire, fault, code):
    reply = envelope()
    calls = reply["choices"][0]["message"]["tool_calls"]
    if fault == "multi_tool":
        calls.append(deepcopy(calls[0]))
    elif fault == "wrong_tool":
        calls[0]["function"]["name"] = "other_tool"
    elif fault == "prompt_over":
        reply["usage"]["prompt_tokens"] = 101
    else:
        reply["usage"]["completion_tokens"] = 1025
    with pytest.raises(policy.TransportError, match=code) as error:
        policy.validate_receipt(wire, reply, prompt_token_bound=100)
    assert error.value.receipt is reply


def test_exactly_one_native_tool_valid_receipt_remains_accepted(wire):
    reply = envelope()
    assert policy.validate_receipt(wire, reply, prompt_token_bound=100) is reply


def test_live_metadata_requires_explicit_opt_in(metadata, wire):
    with pytest.raises(policy.TransportError, match="network_disabled"):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=metadata[0],
            payload=wire, prompt_token_bound=25300)
