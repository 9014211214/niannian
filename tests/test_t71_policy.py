"""Offline T7.1 policy checks: synthetic receipts, no credentials or network."""
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path

import pytest

from aria.bench.refocus_cases import load_refocus_cases
from aria.bench.refocus_e1_runner import inventory
from scripts import t71_policy as policy
from scripts import t7_judge, t7_ledger, t7_transport


ROOT = Path(__file__).resolve().parents[1]
SEAL = "a" * 64


def payload(model="qwen/qwen3.7-flash"):
    result = {"model": model, "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 1024 if model.startswith("qwen/") else 512, "stream": False,
        "provider": {"only": ["alibaba" if model.startswith("qwen/") else "anthropic"],
                     "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"},
        "reasoning": {"enabled": False, "exclude": True}}
    if model.startswith("qwen/"):
        result.update(temperature=0.2, parallel_tool_calls=False,
            tools=[{"type": "function", "function": {"name": "aria_turn_decision"}}],
            tool_choice={"type": "function", "function": {"name": "aria_turn_decision"}})
    return result


def receipt():
    return {"model": "qwen/qwen3.7-flash", "provider": "alibaba",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": "0.0001"},
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"type": "function", "function": {
                "name": "aria_turn_decision", "arguments": '{"response":"我听着。"}'}}]}}]}


def make_ledger(tmp_path, cls=policy.CampaignLedger, name="campaign.sqlite3"):
    return cls(tmp_path / name, campaign_id="t71_test", freeze_seal=SEAL)


def register(ledger, stage="e1", reserve="0.0011", limit="0.5"):
    ledger.register_stage(stage, [{"id": "first", "request_sha256": "b" * 64,
        "reserve_usd": reserve}], limit_usd=limit, max_requests=204 if stage == "e1" else 35)


def finish(ledger, stage="e1", **overrides):
    data = {"cost_usd": "0.0001", "prompt_tokens": 100, "completion_tokens": 20,
        "actual_model": "qwen/qwen3.7-flash" if stage == "e1" else "anthropic/claude-sonnet-5",
        "provider": "alibaba" if stage == "e1" else "anthropic", "response_sha256": "c" * 64}
    data.update(overrides)
    return ledger.finish(stage, "first", **data)


def test_new_judge_ceiling_does_not_mutate_original_ledger(tmp_path):
    original_stages = deepcopy(t7_ledger.STAGES)
    revised = make_ledger(tmp_path)
    register(revised)
    revised.claim("e1", "first")
    finish(revised)
    register(revised, "judge", reserve="0.4")
    revised.claim("judge", "first")
    finish(revised, "judge", cost_usd="0.31")
    restored = make_ledger(tmp_path)
    summary = restored.summary()
    assert summary["known_cost_usd"] == "0.3101"
    assert summary["stages"]["judge"]["limit_usd"] == "0.5"
    assert summary["pause_limit_usd"] == "2" and summary["total_limit_usd"] == "5"
    assert not summary["stopped"]
    legacy = make_ledger(tmp_path, t7_ledger.CampaignLedger, "legacy.sqlite3")
    with pytest.raises(t7_ledger.LedgerError, match="invalid_stage_limits"):
        register(legacy, "judge", reserve="0.4")
    assert t7_ledger.STAGES == original_stages
    assert t7_ledger.STAGES["judge"]["limit"] == Decimal("0.3")


@pytest.mark.parametrize("stage,limit,count", [
    ("e1", "0.51", 204), ("judge", "0.51", 35), ("judge", "0.5", 36),
    ("e1", "0.5", 205), ("e2", "0.5", 1),
])
def test_original_stage_restrictions_preserved(tmp_path, stage, limit, count):
    ledger = make_ledger(tmp_path)
    with pytest.raises(policy.LedgerError):
        ledger.register_stage(stage, [{"id": "first", "request_sha256": "b" * 64,
            "reserve_usd": "0.001"}], limit_usd=limit, max_requests=count)
    assert not ledger.summary()["stages"]


def test_judge_whole_reservation_over_half_dollar_stops_without_claim(tmp_path):
    ledger = make_ledger(tmp_path)
    with pytest.raises(policy.LedgerError, match="stage_budget_exceeded"):
        register(ledger, "judge", reserve="0.500001")
    summary = ledger.summary()
    assert summary["stopped"]
    assert summary["stages"]["judge"]["counts"]["not_started"] == 1
    assert summary["known_cost_usd"] == "0"


def test_unknown_fee_keeps_unknown_and_stops_reopen(tmp_path):
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "first")
    with pytest.raises(policy.LedgerError, match="unknown_cost"):
        finish(ledger, cost_usd=None)
    summary = make_ledger(tmp_path).summary()
    assert summary["unknown_cost_count"] == 1
    assert summary["stages"]["e1"]["items"][0]["cost_usd"] is None
    with pytest.raises(policy.LedgerError, match="unknown_cost"):
        make_ledger(tmp_path).claim("e1", "first")


def test_estimate_counts_entire_actual_wire_and_disclaims_guarantee():
    wire = payload()
    actual_bytes = len(t7_transport._json_bytes(wire))
    result = policy.estimate_request(wire, input_price="0.00000003", output_price="0.00000013")
    assert result["payload_utf8_bytes"] == actual_bytes
    assert result["input_token_plan_bound"] == actual_bytes + 8192
    assert Decimal(result["reserved_cost_usd"]) == (Decimal(actual_bytes + 8192)
        * Decimal("0.00000003") + 1024 * Decimal("0.00000013"))
    assert not result["provider_hard_token_bound_verified"]
    assert not result["provider_billing_guarantee"]
    assert result["post_response_usage_check_required"]


@pytest.mark.parametrize("framing", [0, 8191, 8193, True])
def test_framing_policy_cannot_silently_change(framing):
    with pytest.raises(policy.TransportError, match="unfrozen_framing_policy"):
        policy.estimate_request(payload(), input_price="0.00000003", output_price="0.00000013",
                                framing_tokens=framing)


@pytest.mark.parametrize("field,value", [("prompt_tokens", 1001), ("completion_tokens", 1025)])
def test_actual_usage_over_plan_stops_ledger_with_known_cost(tmp_path, field, value):
    envelope = receipt()
    envelope["usage"][field] = value
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "first")
    with pytest.raises(policy.TransportError, match="token_bound_exceeded") as caught:
        policy.validate_receipt(payload(), envelope, prompt_token_bound=1000)
    assert caught.value.receipt is envelope
    with pytest.raises(policy.LedgerError, match="token_bound_exceeded"):
        finish(ledger, prompt_tokens=envelope["usage"]["prompt_tokens"],
            completion_tokens=envelope["usage"]["completion_tokens"], error=caught.value.code)
    summary = make_ledger(tmp_path).summary()
    assert summary["stopped"] and summary["known_cost_usd"] == "0.0001"
    with pytest.raises(policy.LedgerError, match="token_bound_exceeded"):
        make_ledger(tmp_path).claim("e1", "first")


@pytest.mark.parametrize("mutation", ["two_tools", "wrong_tool", "no_tools", "mixed_content", "two_choices"])
def test_native_tool_receipt_structure_cannot_bypass_policy(tmp_path, mutation):
    envelope = receipt()
    message = envelope["choices"][0]["message"]
    if mutation == "two_tools":
        message["tool_calls"].append(deepcopy(message["tool_calls"][0]))
    elif mutation == "wrong_tool":
        message["tool_calls"][0]["function"]["name"] = "another_tool"
    elif mutation == "no_tools":
        message["tool_calls"] = []
    elif mutation == "mixed_content":
        message["content"] = "another response"
    else:
        envelope["choices"].append(deepcopy(envelope["choices"][0]))
    ledger = make_ledger(tmp_path)
    register(ledger)
    ledger.claim("e1", "first")
    with pytest.raises(policy.TransportError, match="invalid_schema") as caught:
        policy.validate_receipt(payload(), envelope, prompt_token_bound=1000)
    with pytest.raises(policy.LedgerError, match="invalid_schema"):
        finish(ledger, error=caught.value.code)
    assert ledger.summary()["stopped"]


@pytest.mark.parametrize("value", [-1, True, None, 100.5, "100"])
def test_invalid_usage_not_accepted(value):
    envelope = receipt()
    envelope["usage"]["prompt_tokens"] = value
    with pytest.raises(policy.TransportError, match="invalid_receipt"):
        policy.validate_receipt(payload(), envelope, prompt_token_bound=1000)


def test_single_tool_exact_bound_and_judge_text_are_accepted():
    envelope = receipt()
    assert policy.validate_receipt(payload(), envelope, prompt_token_bound=100) is envelope
    envelope["choices"][0]["message"] = {"content": '[["opaque",0,0,0,0]]'}
    assert policy.validate_receipt(payload("anthropic/claude-sonnet-5"), envelope,
                                  prompt_token_bound=100) is envelope


class FakeResponse:
    def __init__(self, value, url):
        self.body, self.url, self.headers = io.BytesIO(json.dumps(value).encode()), url, {}

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
        assert request.method == "GET" and request.get_header("Authorization") is None
        calls.append(request)
        return FakeResponse(documents[request.full_url], request.full_url)

    return frozen, documents, calls, opener


def test_only_parallel_evidence_is_waived_and_actual_payload_unchanged(metadata):
    frozen, _, calls, opener = metadata
    wire = payload()
    before = deepcopy(wire)
    result = policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
        payload=wire, prompt_token_bound=20000, opener=opener)
    assert wire == before and wire["parallel_tool_calls"] is False
    assert result["parallel_tool_calls"] == {"sent": True, "value": False,
        "metadata_status": "unverified", "fallback": "exactly_one_aria_turn_decision_or_stop",
        "actual_wire_field_removed": False}
    assert len(calls) == 2 and result["inference_requests"] == 0
    with pytest.raises(t7_transport.TransportError, match="parameter_support_unverified_parallel_tool_calls"):
        t7_transport.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
            payload=wire, prompt_token_bound=20000, opener=opener)


@pytest.mark.parametrize("fault,code", [
    ("tools", "parameter_support_unverified_tools"), ("reasoning", "parameter_support_unverified_reasoning"),
    ("price", "metadata_price_changed"), ("identity", "metadata_snapshot_changed"),
    ("provider", "metadata_provider_changed"),
])
def test_other_parameter_identity_price_gates_stay_closed(metadata, fault, code):
    frozen, documents, _, opener = metadata
    wire = payload()
    endpoint = documents[t7_transport.MODELS_URL + "/" + wire["model"] + "/endpoints"]["data"]["endpoints"][0]
    if fault in ("tools", "reasoning"):
        endpoint["supported_parameters"].remove(fault)
    elif fault == "price":
        endpoint["pricing"]["prompt"] = "0.99"
    elif fault == "identity":
        documents[t7_transport.MODELS_URL]["data"][0]["canonical_slug"] = "different_model"
    else:
        endpoint["provider_name"] = "different_provider"
    with pytest.raises(policy.TransportError, match=code):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
            payload=wire, prompt_token_bound=20000, opener=opener)


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_parallel_override_rejects_changed_wire_without_get(metadata, value):
    frozen, _, calls, opener = metadata
    wire = payload()
    wire["parallel_tool_calls"] = value
    with pytest.raises(policy.TransportError, match="unfrozen_parallel_tool_calls"):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
            payload=wire, prompt_token_bound=20000, opener=opener)
    assert calls == []


def test_parallel_field_cannot_be_removed_to_pass_preflight(metadata):
    frozen, _, calls, opener = metadata
    wire = payload()
    del wire["parallel_tool_calls"]
    with pytest.raises(policy.TransportError, match="unfrozen_parallel_tool_calls"):
        policy.fetch_metadata(model=wire["model"], frozen_metadata=frozen,
            payload=wire, prompt_token_bound=20000, opener=opener)
    assert calls == []


@pytest.fixture
def design():
    suite = load_refocus_cases()
    plan = json.loads((ROOT / "experiments/t6_20260930/plan.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "experiments/t6_20260930/judge_contract.json").read_text(encoding="utf-8"))
    rows = [{**coordinate, "order": order, "status": "generated",
             "raw_decision": {"response": "我听着。"}, "delivered_response": "我听着。",
             "actual_model": "qwen/qwen3.7-flash"}
            for order, coordinate in enumerate(inventory(suite), 1)]
    seeds = {name: index for index, name in enumerate(
        ("raw_shuffle", "delivered_shuffle", "pair_shuffle", "human_sample"), 1)}
    return rows, suite, plan, contract, seeds


@pytest.mark.parametrize("change_delivered", [False, True])
def test_new_budget_preserves_all_batches_and_contract_bytes(design, change_delivered):
    if change_delivered:
        for row in design[0]:
            row["delivered_response"] = "我听着，请慢慢说。"
    before = deepcopy(design[2:])
    previous = t7_judge.build_batches(*design)
    result = policy.build_batches(*design)
    assert design[2:] == before
    assert result["max_cost_usd"] == "0.5" and previous["max_cost_usd"] == "0.30"
    assert result["batches"] == previous["batches"]
    assert result["slots"] == previous["slots"] and len(result["slots"]) == 408
    assert result["pair_slots"] == previous["pair_slots"] and len(result["pair_slots"]) == 27
    assert len(result["batches"]) == (35 if change_delivered else 19)
    expected = "blocked_budget" if Decimal(result["total_reserve_usd"]) > Decimal("0.5") else "prepared"
    assert result["status"] == expected
    assert result["provider_token_bound_verified"] is False
    assert result["post_response_usage_check_required"] is True


def test_missing_generation_positions_are_still_rejected(design):
    rows, suite, plan, contract, seeds = design
    with pytest.raises(ValueError, match="judge_inventory_requires_all_204_positions"):
        policy.build_batches(rows[:-1], suite, plan, contract, seeds)
