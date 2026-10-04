"""Prospective T7.1 policy adapters; frozen T6/T7 modules remain unchanged.

The input count is a planning assumption, not a provider token guarantee. Every
response still needs the actual usage check, and a failed check stops the durable
campaign. No function reads credentials, submits a model request or retries one.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
from types import FunctionType, MappingProxyType

from scripts import t7_judge, t7_ledger, t7_transport


POLICY_ID = "aria_t71_execution_policy_v1"
JUDGE_LIMIT_USD = Decimal("0.5")
FRAMING_TOKENS = 8192
LedgerError = t7_ledger.LedgerError
TransportError = t7_transport.TransportError


def _ledger_method_with_policy(method):
    """Bind only the two budget-aware methods to an isolated, immutable policy.

    The original function code and every other ledger safety check are reused.
    Neither t7_ledger.STAGES nor any original function globals are mutated. The
    caller's execution seal must bind both the frozen module and this adapter.
    """
    stages = deepcopy(t7_ledger.STAGES)
    stages["judge"]["limit"] = JUDGE_LIMIT_USD
    for spec in stages.values():
        spec["models"] = frozenset(spec["models"])
    policy = MappingProxyType({key: MappingProxyType(value) for key, value in stages.items()})
    namespace = dict(method.__globals__, STAGES=policy)
    adapted = FunctionType(method.__code__, namespace, method.__name__,
                           method.__defaults__, method.__closure__)
    adapted.__kwdefaults__ = deepcopy(method.__kwdefaults__)
    adapted.__annotations__ = dict(method.__annotations__)
    adapted.__doc__ = method.__doc__
    return adapted


class CampaignLedger(t7_ledger.CampaignLedger):
    """T7 ledger with the separately authorized $0.50 judge stage ceiling.

    E1 $0.50, 204/35 request caps, $2 pause, $5 total, unknown costs and E2 denial
    keep the original behavior. Existing campaigns are never rebound here.
    """

    register_stage = _ledger_method_with_policy(t7_ledger.CampaignLedger.register_stage)
    _validate = _ledger_method_with_policy(t7_ledger.CampaignLedger._validate)


def estimate_request(payload, *, input_price, output_price, framing_tokens=FRAMING_TOKENS):
    """Count the complete canonical wire JSON bytes, then add the template plan."""
    if type(framing_tokens) is not int or framing_tokens != FRAMING_TOKENS:
        raise TransportError("unfrozen_framing_policy", outcome_unknown=False)
    previous = t7_transport.estimate_request(payload, input_price=input_price,
        output_price=output_price, framing_tokens=framing_tokens)
    size = len(t7_transport._json_bytes(payload))
    bound = size + previous["framing_token_allowance"]
    reserve = (Decimal(bound) * t7_transport._decimal(input_price)
               + Decimal(payload["max_tokens"]) * t7_transport._decimal(output_price))
    return {**previous, "payload_utf8_bytes": size, "input_token_plan_bound": bound,
            "reserved_cost_usd": str(reserve), "policy_id": POLICY_ID,
            "token_bound_method": "full_payload_utf8_bytes_plus_8192_template_plan",
            "tokenizer_assumption": "at_least_one_utf8_byte_per_token_not_provider_verified",
            "post_response_usage_check_required": True,
            "provider_billing_guarantee": False}


def validate_receipt(payload, envelope, *, prompt_token_bound):
    """Check observed usage and the single native E1 tool receipt before scoring.

    Return the same already-sanitized envelope. On failure retain that receipt so
    the caller can durably account for known cost, then stop with no retry.
    """
    t7_transport._validate_payload(payload)
    if type(prompt_token_bound) is not int or prompt_token_bound < 0:
        raise TransportError("invalid_prompt_bound", outcome_unknown=False)
    if not isinstance(envelope, dict):
        raise TransportError("invalid_receipt")
    usage = envelope.get("usage")
    if (not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0
                   for key in ("prompt_tokens", "completion_tokens"))):
        raise TransportError("invalid_receipt", receipt=envelope)
    if (usage["prompt_tokens"] > prompt_token_bound
            or usage["completion_tokens"] > payload["max_tokens"]):
        raise TransportError("token_bound_exceeded", receipt=envelope)
    if payload["model"] == "qwen/qwen3.7-flash":
        choices = envelope.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], dict)):
            raise TransportError("invalid_schema", receipt=envelope)
        choice = choices[0]
        if choice.get("finish_reason") not in ("tool_calls", "stop"):
            raise TransportError("truncated_output", receipt=envelope)
        message = choice.get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if (not isinstance(message, dict) or message.get("content") not in (None, "")
                or not isinstance(calls, list) or len(calls) != 1
                or not isinstance(calls[0], dict)):
            raise TransportError("invalid_schema", receipt=envelope)
        call = calls[0]
        function = call.get("function")
        if (call.get("type") != "function" or not isinstance(function, dict)
                or function.get("name") != "aria_turn_decision"
                or not isinstance(function.get("arguments"), (str, dict))):
            raise TransportError("invalid_schema", receipt=envelope)
    return envelope


def fetch_metadata(*, model, frozen_metadata, payload, prompt_token_bound,
                   opener=None, enable_network=False, timeout=120):
    """Check all old metadata gates except the explicitly accepted parallel flag.

    A separate copy is used solely for required-parameter evidence. The actual
    payload remains byte-for-byte unchanged, including parallel_tool_calls=False.
    Missing evidence is reported as unverified, never as provider support.
    """
    t7_transport._validate_payload(payload)
    projection = deepcopy(payload)
    parallel_present = "parallel_tool_calls" in payload
    if model == "qwen/qwen3.7-flash" and payload.get("parallel_tool_calls") is not False:
        raise TransportError("unfrozen_parallel_tool_calls", outcome_unknown=False)
    if parallel_present:
        if model != "qwen/qwen3.7-flash" or payload["parallel_tool_calls"] is not False:
            raise TransportError("unfrozen_parallel_tool_calls", outcome_unknown=False)
        del projection["parallel_tool_calls"]
    receipt = t7_transport.fetch_metadata(model=model, frozen_metadata=frozen_metadata,
        payload=projection, prompt_token_bound=prompt_token_bound, opener=opener,
        enable_network=enable_network, timeout=timeout)
    supported = "parallel_tool_calls" in receipt["supported_parameters"]
    receipt.update({"policy_id": POLICY_ID,
        "actual_payload_sha256": hashlib.sha256(t7_transport._json_bytes(payload)).hexdigest(),
        "parallel_tool_calls": {
            "sent": parallel_present, "value": payload.get("parallel_tool_calls"),
            "metadata_status": ("advertised" if supported else "unverified")
                if parallel_present else "not_applicable",
            "fallback": "exactly_one_aria_turn_decision_or_stop" if parallel_present else None,
            "actual_wire_field_removed": False,
        },
        "post_response_usage_check_required": True,
        "provider_hard_token_bound_verified": False,
        "provider_billing_guarantee": False})
    return receipt


def build_batches(rows, suite, plan, contract, seeds):
    """Keep every frozen judge batch and evaluate its unchanged reserve at $0.50."""
    prepared = t7_judge.build_batches(rows, suite, plan, contract, seeds)
    for batch in prepared["batches"]:
        request = contract["model_request"]
        estimate = estimate_request(batch["payload"],
            input_price=Decimal(request["input_price_usd_per_million_tokens"]) / 1000000,
            output_price=Decimal(request["output_price_usd_per_million_tokens"]) / 1000000)
        # The frozen builder already counts full payload bytes. Differences here
        # indicate a programming drift, not permission to reprice/rebatch outputs.
        if (batch["input_token_upper_bound"] != estimate["input_token_plan_bound"]
                or Decimal(batch["reserve_usd"]) != Decimal(estimate["reserved_cost_usd"])):
            raise ValueError("judge_reservation_method_changed")
    total = sum((Decimal(batch["reserve_usd"]) for batch in prepared["batches"]), Decimal(0))
    prepared.update({"schema": "aria_t71_judge_preparation_v1", "policy_id": POLICY_ID,
        "status": "blocked_budget" if total > JUDGE_LIMIT_USD else "prepared",
        "max_cost_usd": str(JUDGE_LIMIT_USD), "original_max_cost_usd": "0.30",
        "total_reserve_usd": str(total),
        "requires_verified_provider_bound_before_send": False,
        "post_response_usage_check_required": True,
        "tokenizer_assumption": "at_least_one_utf8_byte_per_token_not_provider_verified"})
    return prepared
