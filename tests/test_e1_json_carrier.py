"""Prospective dual-carrier E1 tests: fictional receipts, no keys or network."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from scripts import e1_compat as previous, e1_json_carrier as execution
from scripts import run_e1_t71, t71_policy, t74_transport
from tests.test_e1_execution import fake_envelope
from tests.test_e1_compat import metadata, opener_for
from tests.test_t7_transport import Response

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_real_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credentials_forbidden")

    monkeypatch.setattr(run_e1_t71, "credential", forbidden)
    monkeypatch.setattr(run_e1_t71.legacy, "credential", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def guarded_text(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return read_text(path, *args, **kwargs)

    def guarded_bytes(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)


@pytest.fixture(scope="module")
def bundle():
    return run_e1_t71.load_bundle()


@pytest.fixture(scope="module")
def prepared(bundle):
    return execution.prepare_e1(bundle)


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "synthetic-dual-carrier"
    directory.mkdir()
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="dual_carrier_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


def receipt(payload, carrier):
    envelope = fake_envelope(payload)
    envelope["has_hidden_reasoning"] = False
    if carrier == "content_json":
        message = envelope["choices"][0]["message"]
        message["content"] = message["tool_calls"][0]["function"]["arguments"]
        del message["tool_calls"]
        envelope["choices"][0]["finish_reason"] = "stop"
    return envelope


def raw_decision(envelope):
    message = envelope["choices"][0]["message"]
    return (message["tool_calls"][0]["function"]["arguments"]
            if message.get("tool_calls") else message["content"])


def set_raw(envelope, raw):
    message = envelope["choices"][0]["message"]
    if message.get("tool_calls"):
        message["tool_calls"][0]["function"]["arguments"] = raw
    else:
        message["content"] = raw


@pytest.mark.parametrize("carrier", execution.CARRIERS)
def test_exact_field_contract_accepts_each_real_carrier_without_modifying_envelope(prepared, carrier):
    trial = next(iter(prepared[2].values()))
    envelope = receipt(trial["wire"], carrier)
    original = deepcopy(envelope)
    extracted, raw, decision = execution.extract_decision(trial["wire"], envelope)
    assert extracted == carrier and raw == raw_decision(original)
    assert decision == json.loads(raw)
    assert execution.validate_receipt(trial["wire"], envelope,
        prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"]) is envelope
    assert envelope == original
    assert ("tool_calls" in envelope["choices"][0]["message"]) == (carrier == "native_tool")


@pytest.mark.parametrize("carrier", execution.CARRIERS)
@pytest.mark.parametrize("mutation", [
    "missing_response", "extra_root", "extra_nested", "bad_state", "state_relation",
    "state_confidence_boolean", "state_confidence_over", "four_states",
    "affect_missing", "affect_confidence_string", "divergence_boolean_string",
    "uncertainty_nan", "uncertainty_infinity", "uncertainty_boolean", "uncertainty_negative",
    "pace_invalid", "plan_extra", "plan_missing", "plan_too_long", "response_empty",
    "markdown", "prose", "malformed", "duplicate_key", "duplicate_nested_key", "array",
    "trailer", "hidden_key", "hidden_tag", "plain_text",
])
def test_bad_fields_and_json_fail_both_carriers_and_retain_known_receipt(prepared, carrier, mutation):
    trial = next(iter(prepared[2].values()))
    envelope = receipt(trial["wire"], carrier)
    decision = json.loads(raw_decision(envelope))
    state = {"id": "autonomy_threat", "relation": "candidate", "confidence": 0.5,
             "evidence_quote": "合成引用", "rationale": "合成依据"}
    if mutation == "missing_response":
        decision.pop("response")
    elif mutation == "extra_root":
        decision["additional"] = "invented"
    elif mutation == "extra_nested":
        decision["current_affect"]["additional"] = "invented"
    elif mutation.startswith("state_") or mutation in {"bad_state", "four_states"}:
        decision["state_signals"] = [state]
        if mutation == "bad_state":
            state["id"] = "unregistered_state"
        elif mutation == "state_relation":
            state["relation"] = "observed"
        elif mutation == "state_confidence_boolean":
            state["confidence"] = True
        elif mutation == "state_confidence_over":
            state["confidence"] = 1.01
        else:
            decision["state_signals"] = [deepcopy(state) for _ in range(4)]
    elif mutation == "affect_missing":
        decision["current_affect"].pop("divergence")
    elif mutation == "affect_confidence_string":
        decision["current_affect"]["confidence"] = "0.5"
    elif mutation == "divergence_boolean_string":
        decision["current_affect"]["divergence"]["possible"] = "false"
    elif mutation.startswith("uncertainty_"):
        decision["uncertainty"] = {"nan": float("nan"), "infinity": float("inf"),
            "boolean": True, "negative": -0.01}[mutation.removeprefix("uncertainty_")]
    elif mutation == "pace_invalid":
        decision["communication_plan"]["pace"] = "not_in_enum"
    elif mutation == "plan_extra":
        decision["communication_plan"]["additional"] = "invented"
    elif mutation == "plan_missing":
        decision["communication_plan"].pop("boundary")
    elif mutation == "plan_too_long":
        decision["communication_plan"]["boundary"] = "界" * 81
    elif mutation == "response_empty":
        decision["response"] = ""
    elif mutation == "hidden_key":
        decision["analysis"] = "invented hidden value"
    elif mutation == "hidden_tag":
        decision["response"] = "<analysis>invented</analysis>"
    raw = json.dumps(decision, ensure_ascii=False)
    if mutation == "markdown":
        raw = "```json\n" + raw + "\n```"
    elif mutation == "prose":
        raw = "A decision: " + raw
    elif mutation == "malformed":
        raw = raw[:-1]
    elif mutation == "duplicate_key":
        raw = raw[:-1] + ',"response":"duplicate"}'
    elif mutation == "duplicate_nested_key":
        raw = raw.replace('"possible": false', '"possible": false,"possible": true')
    elif mutation == "array":
        raw = "[" + raw + "]"
    elif mutation == "trailer":
        raw += "{}"
    elif mutation == "plain_text":
        raw = "我听着。"
    set_raw(envelope, raw)
    original = deepcopy(envelope)
    with pytest.raises(execution.TransportError) as error:
        execution.extract_decision(trial["wire"], envelope)
    assert error.value.code == "invalid_schema"
    assert error.value.receipt is envelope and envelope == original
    assert error.value.receipt["usage"]["cost"] == "0.0001"


@pytest.mark.parametrize("mutation", ["mixed", "two_tools", "wrong_tool", "tool_type", "empty",
                                      "wrong_finish", "body_tool_finish", "multiple_choices", "wrong_role"])
def test_ambiguous_or_invalid_carrier_is_not_a_tool_fallback(prepared, mutation):
    trial = next(iter(prepared[2].values()))
    envelope = receipt(trial["wire"], "native_tool")
    message = envelope["choices"][0]["message"]
    if mutation == "mixed":
        message["content"] = raw_decision(envelope)
    elif mutation == "two_tools":
        message["tool_calls"].append(deepcopy(message["tool_calls"][0]))
    elif mutation == "wrong_tool":
        message["tool_calls"][0]["function"]["name"] = "invented_other_tool"
    elif mutation == "tool_type":
        message["tool_calls"][0]["type"] = "other"
    elif mutation == "empty":
        message["tool_calls"] = []
        envelope["choices"][0]["finish_reason"] = "stop"
    elif mutation == "wrong_finish":
        envelope["choices"][0]["finish_reason"] = "length"
    elif mutation == "body_tool_finish":
        message["content"] = raw_decision(envelope)
        del message["tool_calls"]
    elif mutation == "multiple_choices":
        envelope["choices"].append(deepcopy(envelope["choices"][0]))
    elif mutation == "wrong_role":
        message["role"] = "user"
    with pytest.raises(execution.TransportError) as error:
        execution.extract_decision(trial["wire"], envelope)
    assert error.value.code == "invalid_schema" and error.value.receipt is envelope


def test_new_preflight_reuses_all_wire_gates_and_requires_new_qualification(bundle, prepared):
    documents = metadata(bundle)
    calls, opener = opener_for(documents)
    report = execution.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and report["distinct_payloads_checked"] == 65
    assert report["fixed_positions"] == 204 and report["ready_to_send"]
    assert report["response_policy_id"] == execution.POLICY_ID
    assert report["post_response_single_native_tool_required"] is False
    assert report["contract_dual_carrier"]["synthetic_tool_calls_permitted"] is False
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    execution.CheckedTransport(None, bounds, qualification=report)
    _, old_opener = opener_for(documents)
    old = previous.preflight(bundle, online=True, opener=old_opener)
    with pytest.raises(execution.TransportError, match="actual_payload_not_qualified"):
        execution.CheckedTransport(None, bounds, qualification=old)
    assert execution.prepare_e1(bundle) == previous.prepare_e1(bundle)


class AuthoredHTTP:
    def __init__(self, ledger, items, *, failure=None):
        self.ledger, self.items, self.failure = ledger, items, failure
        self.calls, self.envelopes = [], []

    def __call__(self, request, *, timeout):
        assert request.get_method() == "POST" and timeout == 120
        assert self.ledger.summary()["outstanding_count"] == 1
        payload = json.loads(request.data)
        item = self.items[len(self.calls)]
        assert previous.canonical_sha256(payload) == item["request_sha256"]
        assert payload["tool_choice"] == "auto" and "parallel_tool_calls" not in payload
        self.calls.append(payload)
        carrier = "content_json" if len(self.calls) % 2 else "native_tool"
        envelope = receipt(payload, carrier)
        if self.failure == "malformed":
            set_raw(envelope, raw_decision(envelope)[:-1])
        elif self.failure == "missing_field":
            decision = json.loads(raw_decision(envelope))
            decision.pop("communication_plan")
            set_raw(envelope, json.dumps(decision))
        elif self.failure == "timeout":
            raise TimeoutError
        elif self.failure == "unknown_cost":
            envelope["usage"].pop("cost")
        self.envelopes.append(deepcopy(envelope))
        return Response(envelope)


def run(campaign, bundle, prepared, *, failure=None):
    directory, ledger = campaign
    opener = AuthoredHTTP(ledger, prepared[3], failure=failure)
    transport = t74_transport.OpenRouterTransport("synthetic-dual-carrier-credential", opener=opener)
    result = execution.run_generation(directory, ledger, bundle, transport,
        source_check=lambda: None, sleep=lambda _: None)
    return result, opener


def test_all_204_mixed_real_carriers_preserve_wire_receipts_fees_and_original_globals(campaign, bundle, prepared):
    legacy = run_e1_t71.legacy
    frozen_globals = {key: legacy.run_generation.__globals__[key]
                      for key in ("prepare_e1", "parse_generation", "generation_row", "write_private")}
    result, opener = run(campaign, bundle, prepared)
    directory, ledger = campaign
    assert result["status"] == "complete" and len(opener.calls) == len(result["rows"]) == 204
    assert result["schema"] == "aria_e1_dual_carrier_live_result_v1"
    assert result["historical_receipts_reclassified"] is False and not result["model_effect_measured"]
    assert {row["response_carrier"] for row in result["rows"]} == set(execution.CARRIERS)
    for item, row in zip(prepared[3], result["rows"]):
        saved = legacy.read_json(directory / "e1" / (item["id"] + ".receipt.json"))
        message = saved["envelope"]["choices"][0]["message"]
        assert ("tool_calls" in message) == (row["response_carrier"] == "native_tool")
        assert row["response_policy_id"] == execution.POLICY_ID and row["status"] == "generated"
    assert legacy.read_json(directory / "e1/results.private.json") == result
    assert all(legacy.run_generation.__globals__[key] is value for key, value in frozen_globals.items())
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("failure,unknown,known", [
    ("missing_field", 0, "0.0001"), ("malformed", 0, "0.0001"),
    ("timeout", 1, "0"), ("unknown_cost", 1, "0"),
])
def test_bad_receipt_preserves_cost_and_stops_without_retry(campaign, bundle, prepared, failure, unknown, known):
    result, opener = run(campaign, bundle, prepared, failure=failure)
    directory, ledger = campaign
    assert len(opener.calls) == 1 and len(result["rows"]) == result["denominator"] == 204
    assert result["status"] == "incomplete" and result["rows"][0]["status"] == "generation_failed"
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    assert ledger.summary()["stopped"] and ledger.summary()["unknown_cost_count"] == unknown
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal(known)
    assert (directory / "e1/e1_001.receipt.json").is_file()
    with pytest.raises(t71_policy.LedgerError):
        ledger.claim("e1", "e1_002")


def test_receipt_save_failure_retains_known_cost_and_no_retry(campaign, bundle, prepared, monkeypatch):
    legacy = run_e1_t71.legacy
    original = legacy.write_private

    def failing_write(path, value):
        if path.name.endswith(".receipt.json"):
            raise OSError("synthetic_save_failure")
        return original(path, value)

    monkeypatch.setattr(legacy, "write_private", failing_write)
    result, opener = run(campaign, bundle, prepared)
    assert len(opener.calls) == 1 and result["stop_reason"] == "result_save_failed"
    assert Decimal(campaign[1].summary()["known_cost_usd"]) == Decimal("0.0001")
    assert campaign[1].summary()["unknown_cost_count"] == 0


@pytest.mark.parametrize("mutation,code", [
    ("prompt", "token_bound_exceeded"), ("completion", "token_bound_exceeded"),
    ("hidden", "reasoning_not_disabled"), ("reasoning_tokens", "reasoning_not_disabled"),
    ("provider", "provider_mismatch"), ("model", "model_mismatch"),
])
def test_usage_identity_and_reasoning_gates_remain(prepared, mutation, code):
    trial = next(iter(prepared[2].values()))
    envelope = receipt(trial["wire"], "content_json")
    bound = trial["cost_plan"]["input_token_plan_bound"]
    if mutation == "prompt":
        envelope["usage"]["prompt_tokens"] = bound + 1
    elif mutation == "completion":
        envelope["usage"]["completion_tokens"] = trial["wire"]["max_tokens"] + 1
    elif mutation == "hidden":
        envelope["has_hidden_reasoning"] = True
    elif mutation == "reasoning_tokens":
        envelope["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1}
    else:
        envelope[mutation] = "unregistered_identity"
    with pytest.raises(execution.TransportError) as error:
        execution.validate_receipt(trial["wire"], envelope, prompt_token_bound=bound)
    assert error.value.code == code and error.value.receipt is envelope


def test_saved_real_failed_body_is_diagnostic_only_and_all_old_receipts_stay_unchanged(prepared):
    stage = ROOT / "artifacts/local/e1_t74_20261001_v1/e1"
    paths = sorted(stage.glob("*.receipt.json"))
    if not paths:
        pytest.skip("private real receipts excluded from release artifacts")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert len(paths) == 12
    for path in paths:
        saved = run_e1_t71.legacy.read_json(path)
        row = saved["row"]
        trial = prepared[2][row["case_id"], row["arm"]]
        execution.validate_receipt(trial["wire"], saved["envelope"],
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"])
        parsed = execution.parse_generation(saved["envelope"], trial, prepared[1][row["case_id"]])
        assert parsed["status"] == "generated"
        if path.name == "e1_012.receipt.json":
            assert saved["error"] == "invalid_schema" and row["status"] == "generation_failed"
            assert parsed["response_carrier"] == "content_json"
            assert "tool_calls" not in saved["envelope"]["choices"][0]["message"]
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
