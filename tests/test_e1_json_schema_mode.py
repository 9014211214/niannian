"""Schema-prompt E1: all actual wires and durable faults, no paid calls."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from aria.bench.prompt_footprint import measure_messages
from scripts import e1_json_mode as previous, e1_json_schema_mode as execution
from scripts import run_e1_t71, t71_policy, t76_transport
from tests.test_e1_compat import metadata, opener_for
from tests.test_e1_execution import fake_envelope
from tests.test_e1_json_mode import no_real_network_or_credentials, receipt  # noqa: F401
from tests.test_t7_transport import Response


@pytest.fixture(scope="module")
def bundle():
    return run_e1_t71.load_bundle()


@pytest.fixture(scope="module")
def prepared(bundle):
    return execution.prepare_e1(bundle)


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "synthetic-json-schema-prompt"
    directory.mkdir()
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="json_schema_prompt_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


def test_all_204_positions_only_append_the_exact_frozen_schema_to_one_format_system(bundle, prepared):
    old = previous.prepare_e1(bundle)
    assert len(prepared[3]) == 204 and len(prepared[2]) == 68
    assert len({item["request_sha256"] for item in prepared[3]}) == 65
    for item, ancestor in zip(prepared[3], old[3]):
        key = item["case_id"], item["arm"]
        before, trial = old[2][key], prepared[2][key]
        assert item["t76_request_sha256"] == ancestor["request_sha256"]
        assert all(item[name] == ancestor[name] for name in (
            "id", "order", "case_id", "arm", "repeat", "reserve_usd", "t74_request_sha256",
            "original_request_sha256", "field_schema_sha256"))
        assert trial["response_field_schema"] == before["response_field_schema"]
        wire = trial["wire"]
        assert wire["response_format"] == {"type": "json_object"}
        assert not set(wire) & {"tools", "tool_choice", "parallel_tool_calls"}
        restored = deepcopy(wire)
        changed = [(index, a, b) for index, (a, b) in
                   enumerate(zip(before["wire"]["messages"], wire["messages"])) if a != b]
        assert len(changed) == 1
        index, a, b = changed[0]
        assert a["role"] == b["role"] == "system" and "唯一通道=plain_json" in a["content"]
        suffix = execution._schema_suffix(before["response_field_schema"])
        assert b["content"] == a["content"] + suffix
        schema_text = suffix.split("\n")[-1]
        assert json.loads(schema_text) == before["response_field_schema"]
        assert hashlib.sha256(schema_text.encode("utf-8")).hexdigest() == item["field_schema_sha256"]
        assert "所有enum值必须逐字匹配" in suffix and "不得拼接中文解释" in suffix
        restored["messages"][index] = deepcopy(a)
        assert restored == before["wire"]
        assert trial["messages"] == wire["messages"]
        assert trial["input_measurement"] == measure_messages(wire["messages"])
        assert trial["input_measurement"]["tool_spec_compact_json_characters"] == 0
        actual_bytes = len(t76_transport.validate_json_mode_payload(wire))
        assert item["input_token_plan_bound"] == actual_bytes + 8192
        assert trial["cost_plan"]["payload_utf8_bytes"] == actual_bytes
        assert item["input_token_plan_bound"] > ancestor["input_token_plan_bound"]
        assert Decimal(trial["cost_plan"]["reserved_cost_usd"]) <= Decimal(item["reserve_usd"])
    assert old == previous.prepare_e1(bundle) and prepared == execution.prepare_e1(bundle)


def test_all_65_actual_payload_metadata_gates_include_appended_schema_and_new_measurement(bundle, prepared, monkeypatch):
    calls, opener = opener_for(metadata(bundle))
    actuals = []
    checker = execution._fetch_actual_metadata

    def observe(**kwargs):
        actuals.append((deepcopy(kwargs["payload"]), kwargs["prompt_token_bound"]))
        return checker(**kwargs)

    monkeypatch.setattr(execution, "_fetch_actual_metadata", observe)
    report = execution.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and len(actuals) == 65
    assert report["ready_to_send"] and report["parameter_compatibility_verified"]
    assert report["schema"] == "aria_e1_json_schema_prompt_preflight_v1"
    assert report["response_policy_id"] == execution.POLICY_ID
    assert report["contract_json_object"] == execution.CONTRACT
    assert report["prompt_field_schema_embedded"] is True
    assert report["server_field_schema_enforced"] is False
    assert report["response_format_evidence"]["json_schema_enforcement_supported"] is False
    assert report["provider_privacy_filter_retained"] == "deny"
    assert report["post_response_single_native_tool_required"] is False
    assert not report["account_route_verified"] and not report["privacy_filter_acceptance_verified"]
    expected = {item["request_sha256"]: item for item in prepared[3]}
    assert set(report["passed_payload_sha256"]) == set(expected)
    for payload, bound in actuals:
        assert payload["response_format"] == {"type": "json_object"}
        assert sum(execution.SCHEMA_SECTION in message["content"] for message in payload["messages"]) == 1
        assert bound == expected[execution.canonical_sha256(payload)]["input_token_plan_bound"]
    for assessment in report["assessments"]:
        item = expected[assessment["actual_payload_sha256"]]
        assert assessment["t76_request_sha256"] == item["t76_request_sha256"]
        assert assessment["t74_request_sha256"] == item["t74_request_sha256"]
        assert assessment["original_request_sha256"] == item["original_request_sha256"]
        assert "response_format" in assessment["required_fields"]
    bounds = {key: item["input_token_plan_bound"] for key, item in expected.items()}
    execution.CheckedTransport(None, bounds, qualification=report)
    old_report = previous.preflight(bundle, online=True, opener=opener)
    with pytest.raises(execution.TransportError, match="actual_payload_not_qualified"):
        execution.CheckedTransport(None, bounds, qualification=old_report)


@pytest.mark.parametrize("mutation,code", [
    ("format", "parameter_support_unverified_response_format"),
    ("mandatory", "reasoning_disable_support_unverified"),
    ("status", "metadata_provider_changed"),
    ("prompt", "metadata_prompt_limit_unverified"),
    ("context", "metadata_token_limit"),
    ("price", "metadata_price_changed"),
])
def test_original_metadata_guards_still_block_all_65_actual_payloads(bundle, mutation, code):
    documents = metadata(bundle)
    endpoint = documents[previous.previous.choices.ENDPOINT]["data"]["endpoints"][0]
    model = next(row for row in documents[execution.t7_transport.MODELS_URL]["data"]
                 if row["id"] == previous.previous.choices.MODEL)
    if mutation == "format":
        endpoint["supported_parameters"].remove("response_format")
    elif mutation == "mandatory":
        model["reasoning"]["mandatory"] = True
    elif mutation == "status":
        endpoint["status"] = False
    elif mutation == "prompt":
        endpoint["max_prompt_tokens"] = True
    elif mutation == "context":
        endpoint["context_length"] = 1
    else:
        endpoint["pricing"]["prompt"] = "0.01"
    _, opener = opener_for(documents)
    report = execution.preflight(bundle, online=True, opener=opener)
    assert not report["ready_to_send"] and report["passed_payload_sha256"] == []
    assert report["distinct_payloads_checked"] == 65 and code in report["blocking_reasons"]


@pytest.mark.parametrize("mutation", ["schema", "format", "user", "core", "tools", "deny"])
def test_unregistered_prompt_or_wire_never_reaches_the_transport(prepared, mutation):
    payload = deepcopy(next(iter(prepared[2].values()))["wire"])
    if mutation == "schema":
        index = next(i for i, message in enumerate(payload["messages"])
                     if execution.SCHEMA_SECTION in message["content"])
        payload["messages"][index]["content"] += "extra"
    elif mutation == "format":
        payload["response_format"] = {"type": "json_schema"}
    elif mutation == "user":
        payload["messages"][-1]["content"] += "invented"
    elif mutation == "core":
        payload["messages"][0]["content"] += "invented"
    elif mutation == "tools":
        payload["tools"] = []
    else:
        payload["provider"]["data_collection"] = "allow"
    with pytest.raises(execution.TransportError):
        execution._snapshot(payload)


@pytest.mark.parametrize("arm", ["B0", "B2"])
@pytest.mark.parametrize("mutation", ["pause", "pause_suffix", "missing", "extra", "boolean",
                                      "duplicate", "malformed", "hidden", "markdown", "native"])
def test_exact_original_schema_fields_and_only_real_json_body(prepared, arm, mutation):
    trial = next(value for (_, trial_arm), value in prepared[2].items() if trial_arm == arm)
    envelope = receipt(trial["wire"])
    message = envelope["choices"][0]["message"]
    value = json.loads(message["content"])
    if mutation in {"pause", "pause_suffix"}:
        value["communication_plan"]["pace"] = "pause" if mutation == "pause" else "pause留白"
    elif mutation == "missing":
        value.pop("uncertainty")
    elif mutation == "extra":
        value["extra"] = "synthetic"
    elif mutation == "boolean":
        value["uncertainty"] = True
    elif mutation == "hidden":
        value["analysis"] = "synthetic hidden"
    raw = json.dumps(value, ensure_ascii=False)
    if mutation == "duplicate":
        raw = raw[:-1] + ',"response":"duplicate"}'
    elif mutation == "malformed":
        raw = raw[:-1]
    elif mutation == "markdown":
        raw = "```json\n" + raw + "\n```"
    message["content"] = raw
    if mutation == "native":
        message["tool_calls"] = fake_envelope(trial["wire"])["choices"][0]["message"]["tool_calls"]
    before = deepcopy(envelope)
    if mutation == "pause":
        carrier, received, visible = execution.extract_decision(trial["wire"], envelope)
        assert carrier == "content_json" and received == raw and visible == value
        assert execution.validate_receipt(trial["wire"], envelope,
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"]) is envelope
    else:
        with pytest.raises(execution.TransportError) as error:
            execution.validate_receipt(trial["wire"], envelope,
                prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"])
        assert error.value.code == "invalid_schema" and error.value.receipt is envelope
        assert envelope["usage"]["cost"] == "0.0001"
    assert envelope == before


def test_saved_t76_failure_stays_invalid_and_unchanged_under_the_new_prompt(prepared):
    path = Path(__file__).resolve().parents[1] / "artifacts/local/e1_t76_20261001_v1/e1/e1_001.receipt.json"
    if not path.is_file():
        pytest.skip("private historical receipt is absent outside this local experiment workspace")
    before = path.read_bytes()
    saved = run_e1_t71.legacy.read_json(path)
    item = prepared[3][0]
    trial = prepared[2][item["case_id"], item["arm"]]
    with pytest.raises(execution.TransportError, match="invalid_schema") as error:
        execution.validate_receipt(trial["wire"], saved["envelope"],
            prompt_token_bound=item["input_token_plan_bound"])
    assert error.value.receipt is saved["envelope"]
    assert saved["error"] == saved["row"]["reason"] == "invalid_schema"
    assert saved["envelope"]["usage"]["cost"] == "0.00009469"
    assert path.read_bytes() == before


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
        assert sum(execution.SCHEMA_SECTION in message["content"] for message in payload["messages"]) == 1
        assert payload["response_format"] == {"type": "json_object"}
        assert not set(payload) & {"tools", "tool_choice", "parallel_tool_calls"}
        self.calls.append(payload)
        envelope = receipt(payload)
        if self.failure == "schema":
            message = envelope["choices"][0]["message"]
            value = json.loads(message["content"])
            value["communication_plan"]["pace"] = "pause留白"
            message["content"] = json.dumps(value)
        elif self.failure == "unknown":
            envelope["usage"].pop("cost")
        elif self.failure == "timeout":
            raise TimeoutError
        return Response(envelope)


def run(campaign, bundle, prepared, *, failure=None):
    directory, ledger = campaign
    opener = AuthoredHTTP(ledger, prepared[3], failure=failure)
    transport = t76_transport.OpenRouterTransport("synthetic-schema-prompt-credential", opener=opener)
    result = execution.run_generation(directory, ledger, bundle, transport,
        source_check=lambda: None, sleep=lambda _: None)
    return result, opener


def test_full_204_actual_fake_http_serialization_receipts_cost_and_original_globals(campaign, bundle, prepared):
    legacy = run_e1_t71.legacy
    globals_before = {name: legacy.run_generation.__globals__[name]
                      for name in ("prepare_e1", "parse_generation", "write_private", "generation_row")}
    prior_snapshot = previous._snapshot
    result, opener = run(campaign, bundle, prepared)
    directory, ledger = campaign
    assert len(opener.calls) == result["denominator"] == len(result["rows"]) == 204
    assert result["status"] == "complete" and result["schema"] == "aria_e1_json_schema_prompt_live_result_v1"
    assert result["prompt_field_schema_embedded"] is True
    assert not result["model_effect_measured"] and not result["server_field_schema_enforced"]
    assert not result["historical_receipts_reclassified"]
    for item, row, sent in zip(prepared[3], result["rows"], opener.calls):
        request = legacy.read_json(directory / "e1" / (item["id"] + ".request.json"))
        saved = legacy.read_json(directory / "e1" / (item["id"] + ".receipt.json"))
        assert sent == request and execution.canonical_sha256(request) == item["request_sha256"]
        assert saved["row"] == row and saved["error"] is None and row["status"] == "generated"
        assert row["response_carrier"] == "content_json" and row["response_policy_id"] == execution.POLICY_ID
        assert row["field_schema_sha256"] == item["field_schema_sha256"]
        assert row["input_measurement"] == prepared[2][item["case_id"], item["arm"]]["input_measurement"]
        assert "tool_calls" not in saved["envelope"]["choices"][0]["message"]
    assert legacy.read_json(directory / "e1/results.private.json") == result
    assert all(legacy.run_generation.__globals__[name] is value for name, value in globals_before.items())
    assert previous._snapshot is prior_snapshot
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("failure,unknown,cost", [
    ("schema", 0, "0.0001"), ("unknown", 1, "0"), ("timeout", 1, "0"),
])
def test_schema_fee_and_network_failures_permanently_stop_without_repair(campaign, bundle, prepared, failure, unknown, cost):
    result, opener = run(campaign, bundle, prepared, failure=failure)
    assert len(opener.calls) == 1 and result["status"] == "incomplete"
    assert result["denominator"] == len(result["rows"]) == 204
    assert result["rows"][0]["status"] == "generation_failed"
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    state = campaign[1].summary()
    assert state["stopped"] and state["unknown_cost_count"] == unknown
    assert Decimal(state["known_cost_usd"]) == Decimal(cost)
    with pytest.raises(t71_policy.LedgerError):
        campaign[1].claim("e1", "e1_002")


def test_save_failure_retains_known_fee_and_permanent_stop(campaign, bundle, prepared, monkeypatch):
    original = run_e1_t71.legacy.write_private

    def fail(path, value):
        if path.name.endswith(".receipt.json"):
            raise OSError("synthetic_save_failure")
        return original(path, value)

    monkeypatch.setattr(run_e1_t71.legacy, "write_private", fail)
    result, opener = run(campaign, bundle, prepared)
    assert len(opener.calls) == 1 and result["stop_reason"] == "result_save_failed"
    assert Decimal(campaign[1].summary()["known_cost_usd"]) == Decimal("0.0001")
    assert campaign[1].summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("mutation", ["required", "boolean_bound", "sha"])
def test_caller_local_schema_cannot_weaken_acceptance(prepared, mutation):
    trial = deepcopy(next(iter(prepared[2].values())))
    if mutation == "required":
        trial["response_field_schema"]["required"] = []
        trial["field_schema_sha256"] = execution.canonical_sha256(trial["response_field_schema"])
    elif mutation == "boolean_bound":
        trial["response_field_schema"]["properties"]["uncertainty"]["maximum"] = True
        trial["field_schema_sha256"] = execution.canonical_sha256(trial["response_field_schema"])
    else:
        trial["field_schema_sha256"] = "f" * 64
    case = prepared[1][next(iter(prepared[2]))[0]]
    with pytest.raises(execution.TransportError, match="response_schema_changed"):
        execution.parse_generation(receipt(trial["wire"]), trial, case)
