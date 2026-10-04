"""Registered E1 response carriers with the frozen decision field contract.

T7.4's actual wire and all 65 parameter gates stay unchanged. This prospective
adapter accepts a real native tool call or a real JSON message body, never
manufactures a tool call, and never changes any historical receipt or ledger.
Both carriers must satisfy the exact frozen tool-parameter schema before the
unchanged production parser and delivery guards run. No credentials or CLI.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time
from types import FunctionType, MappingProxyType

from scripts import e1_compat as previous
from scripts import t7_transport
from scripts.verify_experiment_freeze import canonical_sha256

TransportError = t7_transport.TransportError
POLICY_ID = "aria_e1_dual_response_carrier_v1"
CARRIERS = ("native_tool", "content_json")
prepare_e1 = previous.prepare_e1
_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "enum", "minLength", "maxLength", "minimum", "maximum",
}


def _validate_fields(value, schema):
    """Fail closed for the small JSON Schema vocabulary in the frozen tools.

    JSON Schema's number type excludes booleans and non-finite JSON numbers.
    There is no coercion, truncation, default insertion, or field deletion.
    """
    if not isinstance(schema, dict) or set(schema) - _SCHEMA_KEYS:
        raise ValueError("unsupported_frozen_schema")
    kind = schema.get("type")
    if kind == "object":
        if type(value) is not dict:
            raise ValueError("invalid_schema")
        properties = schema.get("properties", {})
        if (not set(schema.get("required", [])).issubset(value)
                or (schema.get("additionalProperties") is False and set(value) - set(properties))):
            raise ValueError("invalid_schema")
        for key, child in value.items():
            if key in properties:
                _validate_fields(child, properties[key])
    elif kind == "array":
        if (type(value) is not list or len(value) < schema.get("minItems", 0)
                or len(value) > schema.get("maxItems", math.inf)):
            raise ValueError("invalid_schema")
        for child in value:
            _validate_fields(child, schema["items"])
    elif kind == "string":
        if (type(value) is not str or len(value) < schema.get("minLength", 0)
                or len(value) > schema.get("maxLength", math.inf)):
            raise ValueError("invalid_schema")
    elif kind == "number":
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < schema.get("minimum", -math.inf)
                or value > schema.get("maximum", math.inf)):
            raise ValueError("invalid_schema")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ValueError("invalid_schema")
    else:
        raise ValueError("unsupported_frozen_schema")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("invalid_schema")


def extract_decision(payload, envelope):
    """Return (actual carrier, unmodified JSON text, parsed visible fields).

    A body must be a single complete JSON object; Markdown, prose, mixed
    carriers, duplicate keys, hidden reasoning, and fabricated tool fields fail.
    Empty tool_calls is allowed only with a real JSON body, meaning no call.
    """
    previous._snapshot(payload)
    try:
        choices = envelope["choices"]
        if type(choices) is not list or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        if (type(choice) is not dict or type(choice.get("index")) is not int
                or choice["index"] != 0 or type(message) is not dict
                or message.get("role") != "assistant"):
            raise ValueError
        calls = message.get("tool_calls", [])
        content = message.get("content")
        if type(calls) is not list:
            raise ValueError
        if calls:
            if (len(calls) != 1 or type(calls[0]) is not dict
                    or content not in (None, "")
                    or choice.get("finish_reason") not in ("tool_calls", "stop")):
                raise ValueError
            call = calls[0]
            function = call.get("function")
            if (call.get("type") != "function" or type(function) is not dict
                    or function.get("name") != "aria_turn_decision"):
                raise ValueError
            raw = function.get("arguments")
            if type(raw) is dict:
                raw = t7_transport._json_bytes(raw).decode("utf-8")
            carrier = "native_tool"
        else:
            if choice.get("finish_reason") != "stop":
                raise ValueError
            raw, carrier = content, "content_json"
        if type(raw) is not str or len(raw.encode("utf-8")) > t7_transport.MAX_RESPONSE_BYTES:
            raise ValueError
        # The transport decoder rejects duplicate/escaped keys, NaN, nested
        # excess, trailers, and non-object JSON without repairing the text.
        value = t7_transport._decode(raw.encode("utf-8"))
        if t7_transport._has_hidden(value):
            raise ValueError
        schema = payload["tools"][0]["function"]["parameters"]
        _validate_fields(value, schema)
        from aria.bench.refocus_e1_runner import _visible_payload
        visible = _visible_payload(raw)
        return carrier, raw, visible
    except (TransportError, KeyError, IndexError, TypeError, ValueError, OverflowError, RecursionError):
        raise TransportError("invalid_schema", receipt=envelope) from None


def validate_receipt(payload, envelope, *, prompt_token_bound):
    """Retain identity, reasoning, cost, token, and unchanged actual-wire gates."""
    previous._snapshot(payload)
    if type(prompt_token_bound) is not int or prompt_token_bound < 0:
        raise TransportError("invalid_prompt_bound", outcome_unknown=False)
    if type(envelope) is not dict:
        raise TransportError("invalid_receipt")
    # The network transport already sanitizes these fields. Rechecking here
    # also prevents an injected transport from bypassing the registered gates.
    if envelope.get("has_hidden_reasoning") is not False or t7_transport._has_hidden(envelope):
        raise TransportError("reasoning_not_disabled", receipt=envelope)
    canonical, provider = t7_transport._MODELS[payload["model"]]
    if envelope.get("model") not in (payload["model"], canonical):
        raise TransportError("model_mismatch", receipt=envelope)
    if envelope.get("provider") != provider:
        raise TransportError("provider_mismatch", receipt=envelope)
    usage = envelope.get("usage")
    if (type(usage) is not dict or any(type(usage.get(key)) is not int or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens"))):
        raise TransportError("invalid_receipt", receipt=envelope)
    try:
        t7_transport._decimal(usage.get("cost"))
    except ValueError:
        raise TransportError("invalid_usage_cost", receipt=envelope) from None
    details = usage.get("completion_tokens_details", {})
    if (type(details) is not dict or type(details.get("reasoning_tokens", 0)) is not int
            or details.get("reasoning_tokens", 0) != 0):
        raise TransportError("reasoning_not_disabled", receipt=envelope)
    if (usage["prompt_tokens"] > prompt_token_bound
            or usage["completion_tokens"] > payload["max_tokens"]):
        raise TransportError("token_bound_exceeded", receipt=envelope)
    extract_decision(payload, envelope)
    return envelope


def preflight(bundle, *, online=False, opener=None, documents=None):
    report = previous.preflight(bundle, online=online, opener=opener, documents=documents)
    report.update(schema="aria_e1_dual_carrier_preflight_v1", response_policy_id=POLICY_ID,
        post_response_single_native_tool_required=False,
        contract_dual_carrier={"carriers": list(CARRIERS), "mutually_exclusive": True,
            "field_schema": "unchanged_actual_tool_parameters", "strict_fields_for_both": True,
            "synthetic_tool_calls_permitted": False, "retrospective_old_success_permitted": False})
    return report


class CheckedTransport:
    def __init__(self, transport, bounds, *, qualification=None):
        self.transport, self.bounds = transport, MappingProxyType(dict(bounds))
        self.qualified = None
        if qualification is not None:
            contract = qualification.get("contract_dual_carrier")
            if (qualification.get("schema") != "aria_e1_dual_carrier_preflight_v1"
                    or qualification.get("response_policy_id") != POLICY_ID
                    or qualification.get("post_response_single_native_tool_required") is not False
                    or contract != {"carriers": list(CARRIERS), "mutually_exclusive": True,
                        "field_schema": "unchanged_actual_tool_parameters", "strict_fields_for_both": True,
                        "synthetic_tool_calls_permitted": False, "retrospective_old_success_permitted": False}
                    or qualification.get("ready_to_send") is not True
                    or qualification.get("parameter_compatibility_verified") is not True
                    or set(qualification.get("passed_payload_sha256", [])) != set(bounds)
                    or qualification.get("metadata_scope") != "live_public_get"):
                raise TransportError("actual_payload_not_qualified", outcome_unknown=False)
            self.qualified = frozenset(qualification["passed_payload_sha256"])

    def request(self, payload):
        snapshot = deepcopy(payload)
        digest = canonical_sha256(snapshot)
        if digest not in self.bounds or (self.qualified is not None and digest not in self.qualified):
            raise TransportError("request_changed", outcome_unknown=False)
        bound = self.bounds[digest]
        if type(bound) is not int or bound < 0:
            raise TransportError("invalid_prompt_bound", outcome_unknown=False)
        previous._snapshot(snapshot)
        envelope = self.transport.request(snapshot)
        if canonical_sha256(snapshot) != digest:
            raise TransportError("request_changed", receipt=envelope)
        return validate_receipt(snapshot, envelope, prompt_token_bound=bound)


def parse_generation(envelope, trial, case):
    """Apply the original field, production, and delivery pipeline to real JSON."""
    from aria.bench.prompt_footprint import measure_output_fields
    from aria.engine.policy.online_decision import apply_single_pass_decision
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    carrier, raw, visible = extract_decision(trial["wire"], envelope)
    ctx, persona = deepcopy(trial["ctx"]), deepcopy(trial["persona"])
    delivered = apply_single_pass_decision(raw, ctx, case.stimulus.user_input, persona)
    delivered = guard_external_delivery_response(ctx, delivered, case.stimulus.user_input, persona)
    trace = ctx.working_memory.get("online_turn_decision") or {}
    return {
        "raw_decision": visible, "delivered_response": delivered,
        "status": "generated" if trace.get("status") == "parsed" else "invalid_protocol",
        "deliberation_status": trace.get("deliberation_status", "not_applicable"),
        "selected_in_deliberated_options": trace.get("selected_in_deliberated_options"),
        "selected_intervention_id": trace.get("selected_intervention_id"),
        "rule_response_changed": delivered != visible["response"],
        "output_measurement": measure_output_fields([visible]),
        "response_carrier": carrier, "response_policy_id": POLICY_ID,
        "field_schema_sha256": canonical_sha256(trial["wire"]["tools"][0]["function"]["parameters"]),
    }


def run_generation(directory, ledger, bundle, transport, *, source_check,
                   sleep=time.sleep, qualification=None):
    """Clone only the old durable loop's globals; never mutate frozen modules."""
    function = previous.previous.legacy.run_generation
    original_write = function.__globals__["write_private"]

    def registered_write(path, value):
        if path.name == "results.private.json":
            value.update(schema="aria_e1_dual_carrier_live_result_v1", response_policy_id=POLICY_ID,
                response_carriers=list(CARRIERS), historical_receipts_reclassified=False)
        return original_write(path, value)

    def registered_row(item):
        return function.__globals__["generation_row"](item) | {
            "response_policy_id": POLICY_ID, "response_carrier": None}

    adapted = FunctionType(function.__code__, {**function.__globals__,
        "prepare_e1": prepare_e1, "parse_generation": parse_generation,
        "generation_row": registered_row, "write_private": registered_write},
        function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    _, _, _, items = prepare_e1(bundle)
    checked = CheckedTransport(transport,
        {item["request_sha256"]: item["input_token_plan_bound"] for item in items},
        qualification=qualification)
    return adapted(directory, ledger, bundle, checked, source_check=source_check, sleep=sleep)
