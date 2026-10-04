"""JSON-object E1 integration: exact actual-wire gates, no keys or network."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from aria.bench.prompt_footprint import measure_messages
from aria.engine.policy.online_return_channel import bind_online_return_channel
from scripts import e1_compat as previous, e1_json_mode as execution
from scripts import run_e1_t71, t71_policy, t76_transport
from tests.test_e1_execution import fake_envelope
from tests.test_e1_compat import metadata, opener_for
from tests.test_t7_transport import Response


@pytest.fixture(autouse=True)
def no_real_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credentials_forbidden")

    monkeypatch.setattr(run_e1_t71, "credential", forbidden)
    monkeypatch.setattr(run_e1_t71.legacy, "credential", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    text, binary = Path.read_text, Path.read_bytes

    def safe_text(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return text(path, *args, **kwargs)

    def safe_bytes(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return binary(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", safe_text)
    monkeypatch.setattr(Path, "read_bytes", safe_bytes)


@pytest.fixture(scope="module")
def bundle():
    return run_e1_t71.load_bundle()


@pytest.fixture(scope="module")
def prepared(bundle):
    return execution.prepare_e1(bundle)


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "synthetic-json-mode"
    directory.mkdir()
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="json_mode_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


def receipt(payload):
    result = fake_envelope(payload)
    message = result["choices"][0]["message"]
    message["content"] = message["tool_calls"][0]["function"]["arguments"]
    del message["tool_calls"]
    result["choices"][0]["finish_reason"] = "stop"
    result["has_hidden_reasoning"] = False
    return result


def test_all_fixed_coordinates_and_original_schemas_preserved_with_only_one_format_system_change(bundle, prepared):
    old = previous.prepare_e1(bundle)
    assert len(prepared[3]) == 204 and len(prepared[2]) == 68
    assert len({item["request_sha256"] for item in prepared[3]}) == 65
    for item, ancestor in zip(prepared[3], old[3]):
        key = item["case_id"], item["arm"]
        before, trial = old[2][key], prepared[2][key]
        assert item["t74_request_sha256"] == ancestor["request_sha256"]
        assert item["original_request_sha256"] == ancestor["original_request_sha256"]
        assert all(item[name] == ancestor[name] for name in ("id", "order", "case_id", "arm", "repeat", "reserve_usd"))
        wire = trial["wire"]
        assert wire["response_format"] == {"type": "json_object"}
        assert not set(wire) & {"tools", "tool_choice", "parallel_tool_calls"}
        assert trial["response_field_schema"] == before["wire"]["tools"][0]["function"]["parameters"]
        assert item["field_schema_sha256"] == execution.canonical_sha256(trial["response_field_schema"])
        assert bind_online_return_channel(wire["messages"], channel="native") == before["wire"]["messages"]
        changes = [(a, b) for a, b in zip(before["wire"]["messages"], wire["messages"]) if a != b]
        assert len(changes) == 1 and changes[0][0]["role"] == changes[0][1]["role"] == "system"
        assert "唯一通道=native" in changes[0][0]["content"]
        assert "唯一通道=plain_json" in changes[0][1]["content"]
        assert trial["input_measurement"] == measure_messages(wire["messages"])
        assert trial["input_measurement"]["tool_spec_compact_json_characters"] == 0
        assert item["input_token_plan_bound"] == len(t76_transport.validate_json_mode_payload(wire)) + 8192
    assert old == previous.prepare_e1(bundle) and prepared == execution.prepare_e1(bundle)


def test_65_actual_response_format_payloads_are_checked_and_no_schema_enforcement_is_claimed(bundle, prepared):
    calls, opener = opener_for(metadata(bundle))
    actuals = []
    metadata_check = execution._fetch_actual_metadata

    def observe(**kwargs):
        actuals.append(deepcopy(kwargs["payload"]))
        return metadata_check(**kwargs)

    # Replace this one private function only for an observation, never alter
    # the old metadata module or omit the actual response_format field.
    with pytest.MonkeyPatch.context() as context:
        context.setattr(execution, "_fetch_actual_metadata", observe)
        report = execution.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and len(actuals) == 65
    assert report["ready_to_send"] and report["parameter_compatibility_verified"]
    assert all(payload["response_format"] == {"type": "json_object"} for payload in actuals)
    assert all("response_format" in row["required_fields"] for row in report["assessments"])
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert not report["server_field_schema_enforced"]
    assert report["contract_json_object"] == execution.CONTRACT
    assert report["response_format_evidence"]["json_schema_enforcement_supported"] is False
    assert report["provider_privacy_filter_retained"] == "deny"
    assert not report["account_route_verified"] and not report["privacy_filter_acceptance_verified"]
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    execution.CheckedTransport(None, bounds, qualification=report)


@pytest.mark.parametrize("mutation,code", [
    ("response_format_missing", "parameter_support_unverified_response_format"),
    ("mandatory", "reasoning_disable_support_unverified"),
    ("status_boolean", "metadata_provider_changed"),
    ("prompt_missing", "metadata_prompt_limit_unverified"),
    ("prompt_boolean", "metadata_prompt_limit_unverified"),
    ("prompt_small", "metadata_token_limit"),
    ("context", "metadata_token_limit"),
    ("price", "metadata_price_changed"),
    ("canonical", "metadata_snapshot_changed"),
])
def test_exact_current_route_and_response_format_evidence_fail_closed_for_all_positions(bundle, mutation, code):
    documents = metadata(bundle)
    endpoint = documents[previous.choices.ENDPOINT]["data"]["endpoints"][0]
    model = next(row for row in documents[execution.t7_transport.MODELS_URL]["data"]
                 if row["id"] == previous.choices.MODEL)
    if mutation == "response_format_missing":
        endpoint["supported_parameters"].remove("response_format")
    elif mutation == "mandatory":
        model["reasoning"]["mandatory"] = True
    elif mutation == "status_boolean":
        endpoint["status"] = False
    elif mutation == "prompt_missing":
        endpoint.pop("max_prompt_tokens")
    elif mutation == "prompt_boolean":
        endpoint["max_prompt_tokens"] = True
    elif mutation == "prompt_small":
        endpoint["max_prompt_tokens"] = 1
    elif mutation == "context":
        endpoint["context_length"] = 1
    elif mutation == "price":
        endpoint["pricing"]["prompt"] = "0.01"
    else:
        model["canonical_slug"] = "unregistered_snapshot"
    _, opener = opener_for(documents)
    report = execution.preflight(bundle, online=True, opener=opener)
    assert not report["ready_to_send"] and report["passed_payload_sha256"] == []
    assert report["distinct_payloads_checked"] == 65 and code in report["blocking_reasons"]


@pytest.mark.parametrize("mutation", ["tools", "choice", "parallel", "format", "temperature", "tokens", "message", "deny"])
def test_unregistered_actual_wire_never_reaches_transport(prepared, mutation):
    trial = next(iter(prepared[2].values()))
    payload = deepcopy(trial["wire"])
    if mutation == "tools":
        payload["tools"] = []
    elif mutation == "choice":
        payload["tool_choice"] = "none"
    elif mutation == "parallel":
        payload["parallel_tool_calls"] = False
    elif mutation == "format":
        payload["response_format"] = {"type": "json_schema"}
    elif mutation == "temperature":
        payload["temperature"] = 0.3
    elif mutation == "tokens":
        payload["max_tokens"] = 1025
    elif mutation == "message":
        payload["messages"][0]["content"] += "invented change"
    else:
        payload["provider"]["data_collection"] = "allow"
    with pytest.raises(execution.TransportError):
        execution._snapshot(payload)


@pytest.mark.parametrize("arm", ["B0", "B2"])
@pytest.mark.parametrize("mutation", ["valid", "missing", "extra", "bad_state", "bool_confidence", "markdown",
                                      "duplicate", "malformed", "hidden", "native", "mixed", "truncated"])
def test_body_only_carrier_and_exact_original_v1_v2_fields(prepared, arm, mutation):
    trial = next(value for (case_id, trial_arm), value in prepared[2].items() if trial_arm == arm)
    payload = trial["wire"]
    envelope = receipt(payload)
    message = envelope["choices"][0]["message"]
    decision = json.loads(message["content"])
    if mutation == "missing":
        decision.pop("communication_plan")
    elif mutation == "extra":
        decision["arbitrary"] = "invented"
    elif mutation == "bad_state":
        decision["state_signals"] = [{"id": "wrong_state", "relation": "candidate", "confidence": 0.5,
            "evidence_quote": "合成引用", "rationale": "合成依据"}]
    elif mutation == "bool_confidence":
        decision["current_affect"]["confidence"] = True
    elif mutation == "hidden":
        decision["analysis"] = "invented hidden value"
    raw = json.dumps(decision, ensure_ascii=False)
    if mutation == "markdown":
        raw = "```json\n" + raw + "\n```"
    elif mutation == "duplicate":
        raw = raw[:-1] + ',"response":"duplicate"}'
    elif mutation == "malformed":
        raw = raw[:-1]
    message["content"] = raw
    if mutation in {"native", "mixed"}:
        message["tool_calls"] = fake_envelope(payload)["choices"][0]["message"]["tool_calls"]
        if mutation == "native":
            message["content"] = None
            envelope["choices"][0]["finish_reason"] = "tool_calls"
    elif mutation == "truncated":
        envelope["choices"][0]["finish_reason"] = "length"
    original = deepcopy(envelope)
    if mutation == "valid":
        carrier, received, visible = execution.extract_decision(payload, envelope)
        assert carrier == "content_json" and received == raw and visible == decision
        assert execution.validate_receipt(payload, envelope,
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"]) is envelope
    else:
        with pytest.raises(execution.TransportError) as error:
            execution.extract_decision(payload, envelope)
        assert error.value.code == "invalid_schema" and error.value.receipt is envelope
        assert error.value.receipt["usage"]["cost"] == "0.0001"
    assert envelope == original


class AuthoredHTTP:
    def __init__(self, ledger, items, *, failure=None):
        self.ledger, self.items, self.failure = ledger, items, failure
        self.calls = []

    def __call__(self, request, *, timeout):
        assert request.get_method() == "POST" and timeout == 120
        assert self.ledger.summary()["outstanding_count"] == 1
        payload = json.loads(request.data)
        item = self.items[len(self.calls)]
        assert execution.canonical_sha256(payload) == item["request_sha256"]
        assert payload["response_format"] == {"type": "json_object"}
        assert not set(payload) & {"tools", "tool_choice", "parallel_tool_calls"}
        self.calls.append(payload)
        envelope = receipt(payload)
        if self.failure == "schema":
            message = envelope["choices"][0]["message"]
            value = json.loads(message["content"])
            value.pop("uncertainty")
            message["content"] = json.dumps(value)
        elif self.failure == "unknown":
            envelope["usage"].pop("cost")
        elif self.failure == "timeout":
            raise TimeoutError
        elif self.failure == "native":
            envelope = fake_envelope(payload)
        return Response(envelope)


def run(campaign, bundle, prepared, *, failure=None):
    directory, ledger = campaign
    opener = AuthoredHTTP(ledger, prepared[3], failure=failure)
    transport = t76_transport.OpenRouterTransport("synthetic-json-mode-credential", opener=opener)
    result = execution.run_generation(directory, ledger, bundle, transport,
        source_check=lambda: None, sleep=lambda _: None)
    return result, opener


def test_full_204_json_mode_actual_serialization_preserves_ancestry_receipts_and_legacy_globals(campaign, bundle, prepared):
    legacy = run_e1_t71.legacy
    globals_before = {name: legacy.run_generation.__globals__[name]
                      for name in ("prepare_e1", "parse_generation", "write_private", "generation_row")}
    result, opener = run(campaign, bundle, prepared)
    directory, ledger = campaign
    assert len(opener.calls) == result["denominator"] == len(result["rows"]) == 204
    assert result["status"] == "complete" and result["schema"] == "aria_e1_json_object_live_result_v1"
    assert not result["model_effect_measured"] and not result["server_field_schema_enforced"]
    assert not result["historical_receipts_reclassified"]
    for item, row, sent in zip(prepared[3], result["rows"], opener.calls):
        request = legacy.read_json(directory / "e1" / (item["id"] + ".request.json"))
        saved = legacy.read_json(directory / "e1" / (item["id"] + ".receipt.json"))
        assert sent == request and execution.canonical_sha256(request) == item["request_sha256"]
        assert saved["row"] == row and saved["error"] is None and row["status"] == "generated"
        assert row["response_carrier"] == "content_json" and row["response_policy_id"] == execution.POLICY_ID
        assert "tool_calls" not in saved["envelope"]["choices"][0]["message"]
        assert row["field_schema_sha256"] == item["field_schema_sha256"]
    assert legacy.read_json(directory / "e1/results.private.json") == result
    assert all(legacy.run_generation.__globals__[name] is value for name, value in globals_before.items())
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("failure,unknown,cost", [
    ("schema", 0, "0.0001"), ("native", 0, "0.0001"), ("unknown", 1, "0"), ("timeout", 1, "0"),
])
def test_failed_new_output_is_never_repaired_or_retried_and_fixed_inventory_remains(campaign, bundle, prepared, failure, unknown, cost):
    result, opener = run(campaign, bundle, prepared, failure=failure)
    assert len(opener.calls) == 1 and result["status"] == "incomplete"
    assert result["denominator"] == len(result["rows"]) == 204
    assert result["rows"][0]["status"] == "generation_failed"
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    summary = campaign[1].summary()
    assert summary["stopped"] and summary["unknown_cost_count"] == unknown
    assert Decimal(summary["known_cost_usd"]) == Decimal(cost)
    with pytest.raises(t71_policy.LedgerError):
        campaign[1].claim("e1", "e1_002")


def test_save_failure_is_durable_known_cost_stop(campaign, bundle, prepared, monkeypatch):
    original = run_e1_t71.legacy.write_private

    def failed_save(path, value):
        if path.name.endswith(".receipt.json"):
            raise OSError("synthetic_save_failure")
        return original(path, value)

    monkeypatch.setattr(run_e1_t71.legacy, "write_private", failed_save)
    result, opener = run(campaign, bundle, prepared)
    assert len(opener.calls) == 1 and result["stop_reason"] == "result_save_failed"
    assert Decimal(campaign[1].summary()["known_cost_usd"]) == Decimal("0.0001")
    assert campaign[1].summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("mutation", ["required", "boolean_bound", "forged_sha"])
def test_mutated_local_schema_cannot_relax_field_acceptance(prepared, mutation):
    trial = deepcopy(next(iter(prepared[2].values())))
    if mutation == "required":
        trial["response_field_schema"]["required"] = []
        trial["field_schema_sha256"] = execution.canonical_sha256(trial["response_field_schema"])
    elif mutation == "boolean_bound":
        trial["response_field_schema"]["properties"]["uncertainty"]["maximum"] = True
        trial["field_schema_sha256"] = execution.canonical_sha256(trial["response_field_schema"])
    else:
        trial["field_schema_sha256"] = "f" * 64
    envelope = receipt(trial["wire"])
    case = prepared[1][next(iter(prepared[2]))[0]]
    with pytest.raises(execution.TransportError, match="response_schema_changed"):
        execution.parse_generation(envelope, trial, case)
