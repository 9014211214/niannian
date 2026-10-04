"""Prospective E1 JSON-object wire with the unchanged local decision schema.

Only the dedicated return-channel system message changes from native to
plain_json. Task/person/stimulus messages and their order stay byte-identical.
The provider guarantees JSON syntax, not field correctness. The original frozen
tool-parameter schema is retained locally and remains a strict acceptance gate.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
import json
import time
from types import FunctionType, MappingProxyType, SimpleNamespace

from aria.bench.prompt_footprint import measure_messages
from aria.engine.policy.online_return_channel import bind_online_return_channel
from scripts import e1_compat as previous, e1_json_carrier as fields
from scripts import run_e1_t71, t7_transport
from scripts.t76_transport import validate_json_mode_payload
from scripts.verify_experiment_freeze import canonical_sha256

TransportError = t7_transport.TransportError
POLICY_ID = "aria_e1_json_object_response_v1"
FORMAT = {"type": "json_object"}
CONTRACT = {
    "carrier": "content_json", "response_format": FORMAT,
    "server_field_schema_enforced": False, "local_exact_frozen_field_schema_required": True,
    "native_tools_permitted": False, "synthetic_tool_calls_permitted": False,
}
FORMAT_EVIDENCE = {
    "model_page": "https://openrouter.ai/qwen/qwen3.7-flash",
    "parameter_reference": "https://openrouter.ai/docs/api/reference/parameters",
    "checked_on": "2026-10-01", "json_object_supported": True,
    "json_schema_enforcement_supported": False,
    "scope": "the exact OpenRouter model/Alibaba endpoint, not a direct Alibaba API promise",
}


def _clone(function, **bindings):
    copied = FunctionType(function.__code__, {**function.__globals__, **bindings},
        function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return copied


def _prepare(bundle):
    suite, cases, originals, old_items = previous.prepare_e1(bundle)
    prepared = {}
    for key, original in originals.items():
        trial = deepcopy(original)
        old_wire = original["wire"]
        trial["response_field_schema"] = deepcopy(old_wire["tools"][0]["function"]["parameters"])
        trial["field_schema_sha256"] = canonical_sha256(trial["response_field_schema"])
        trial["t74_request_sha256"] = canonical_sha256(old_wire)
        trial["wire"]["messages"] = bind_online_return_channel(old_wire["messages"], channel="plain_json")
        new_messages = trial["wire"]["messages"]
        changed = [(old, new) for old, new in zip(old_wire["messages"], new_messages) if old != new]
        if (len(old_wire["messages"]) != len(new_messages) or len(changed) != 1
                or changed[0][0]["role"] != "system" or changed[0][1]["role"] != "system"
                or "唯一通道=native" not in changed[0][0]["content"]
                or "唯一通道=plain_json" not in changed[0][1]["content"]
                or bind_online_return_channel(new_messages, channel="native") != old_wire["messages"]):
            raise TransportError("unregistered_prompt_channel_change", outcome_unknown=False)
        del trial["wire"]["tools"]
        del trial["wire"]["tool_choice"]
        trial["wire"]["response_format"] = deepcopy(FORMAT)
        raw = validate_json_mode_payload(trial["wire"])
        bound = len(raw) + 8192
        reserve = Decimal(bound) * Decimal("0.00000003") + Decimal(1024) * Decimal("0.00000013")
        if bound > bundle["plan"]["e1"]["max_prompt_tokens"] or reserve > Decimal("0.0011"):
            raise TransportError("e1_reservation_insufficient", outcome_unknown=False)
        trial["cost_plan"] = {
            "payload_utf8_bytes": len(raw), "input_token_plan_bound": bound,
            "framing_token_allowance": 8192, "reserved_cost_usd": str(reserve),
            "policy_id": POLICY_ID, "max_output_tokens": 1024,
            "token_bound_method": "full_payload_utf8_bytes_plus_8192_template_plan",
            "provider_billing_guarantee": False, "post_response_usage_check_required": True,
        }
        trial["messages"] = deepcopy(new_messages)
        trial["input_measurement"] = measure_messages(new_messages)
        prepared[key] = trial
    items = deepcopy(old_items)
    for item in items:
        trial = prepared[item["case_id"], item["arm"]]
        item["t74_request_sha256"] = item["request_sha256"]
        item["request_sha256"] = canonical_sha256(trial["wire"])
        item["field_schema_sha256"] = trial["field_schema_sha256"]
        item["input_token_plan_bound"] = trial["cost_plan"]["input_token_plan_bound"]
        if (item["t74_request_sha256"] != trial["t74_request_sha256"]
                or item["original_request_sha256"] != trial["original_request_sha256"]):
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
    if len(items) != 204 or len({item["request_sha256"] for item in items}) != 65:
        raise TransportError("amended_inventory_changed", outcome_unknown=False)
    return suite, cases, prepared, items


def prepare_e1(bundle):
    return _prepare(bundle)


@lru_cache(maxsize=1)
def _bindings():
    # These derive only from the inherited frozen files. Source verification is
    # still performed before each paid claim by the execution layer.
    _, _, prepared, items = _prepare(run_e1_t71.load_bundle())
    result = {}
    for item in items:
        trial = prepared[item["case_id"], item["arm"]]
        result[item["request_sha256"]] = (
            item["t74_request_sha256"], item["original_request_sha256"],
            t7_transport._json_bytes(trial["response_field_schema"]),
        )
    return MappingProxyType(result)


def _snapshot(payload):
    raw = validate_json_mode_payload(payload)
    snapshot = t7_transport._decode(raw)
    digest = canonical_sha256(snapshot)
    binding = _bindings().get(digest)
    if binding is None:
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    t74_sha, original_sha, schema_bytes = binding
    return snapshot, digest, t74_sha, original_sha, t7_transport._decode(schema_bytes)


def extract_decision(payload, envelope, *, field_schema=None):
    """Accept only a real JSON body; never synthesize an absent native tool."""
    _, _, _, _, frozen_schema = _snapshot(payload)
    if field_schema is not None and canonical_sha256(field_schema) != canonical_sha256(frozen_schema):
        raise TransportError("response_schema_changed", outcome_unknown=False)
    try:
        choices = envelope["choices"]
        if type(choices) is not list or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        if (type(choice) is not dict or type(choice.get("index")) is not int or choice["index"] != 0
                or choice.get("finish_reason") != "stop" or type(message) is not dict
                or message.get("role") != "assistant" or message.get("tool_calls") not in (None, [])):
            raise ValueError
        raw = message.get("content")
        if type(raw) is not str or len(raw.encode("utf-8")) > t7_transport.MAX_RESPONSE_BYTES:
            raise ValueError
        value = t7_transport._decode(raw.encode("utf-8"))
        if t7_transport._has_hidden(value):
            raise ValueError
        fields._validate_fields(value, frozen_schema)
        from aria.bench.refocus_e1_runner import _visible_payload
        return "content_json", raw, _visible_payload(raw)
    except (TransportError, KeyError, IndexError, TypeError, ValueError, OverflowError, RecursionError):
        raise TransportError("invalid_schema", receipt=envelope) from None


_validate_receipt = _clone(fields.validate_receipt,
    previous=SimpleNamespace(_snapshot=_snapshot), extract_decision=extract_decision)


def validate_receipt(payload, envelope, *, prompt_token_bound):
    return _validate_receipt(payload, envelope, prompt_token_bound=prompt_token_bound)


_fetch_actual_metadata = _clone(t7_transport.fetch_metadata, _validate_payload=validate_json_mode_payload)


def preflight(bundle, *, online=False, opener=None, documents=None):
    """Check all 65 complete actual JSON wires, including response_format."""
    _, _, prepared, items = prepare_e1(bundle)
    report = {
        "schema": "aria_e1_json_object_preflight_v1", "policy_id": POLICY_ID,
        "response_policy_id": POLICY_ID, "contract_json_object": deepcopy(CONTRACT),
        "response_format_evidence": deepcopy(FORMAT_EVIDENCE),
        "post_response_single_native_tool_required": False,
        "fixed_positions": len(items), "distinct_payloads_checked": 0,
        "linked_inventory_sha256": canonical_sha256(items),
        "model_requests": 0, "credentials_read": False, "ready_to_send": False,
        "parameter_compatibility_verified": False, "account_route_verified": False,
        "provider_privacy_filter_retained": "deny", "privacy_filter_acceptance_verified": False,
        "provider_cost_guarantee": False, "server_field_schema_enforced": False,
        "planned_full_e1_reserve_usd": str(sum((Decimal(i["reserve_usd"]) for i in items), Decimal(0))),
        "budget_usd": deepcopy(bundle["amendment"]["budget_usd"]),
        "blocking_reasons": [], "passed_payload_sha256": [], "assessments": [],
    }
    if online:
        if documents is not None:
            raise TransportError("live_gate_cannot_use_saved_documents", outcome_unknown=False)
        documents, receipt = previous._documents(opener=opener, enable_network=True)
        report["metadata"], report["metadata_scope"] = receipt, "live_public_get"
    elif documents is None:
        report["blocking_reasons"].append("live_metadata_not_checked")
        return report
    else:
        report["metadata_scope"] = "saved_public_snapshot"
        report["blocking_reasons"].append("live_metadata_not_checked")
    frozen = run_e1_t71.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")

    def static_open(request, *, timeout):
        if request.get_method() != "GET" or request.get_header("Authorization") is not None:
            raise TransportError("invalid_static_metadata_request", outcome_unknown=False)
        return previous.t72_policy._StaticResponse(documents[request.full_url], request.full_url)

    assessments = {}
    for trial in prepared.values():
        snapshot, digest, t74_sha, original_sha, _ = _snapshot(trial["wire"])
        if digest in assessments:
            continue
        assessment = {"actual_payload_sha256": digest, "t74_request_sha256": t74_sha,
            "original_request_sha256": original_sha, "supported": False, "blocking_reason": None}
        try:
            selected = [row for row in documents[previous.choices.ENDPOINT]["data"]["endpoints"]
                        if row.get("tag") == previous.choices.PROVIDER]
            if len(selected) != 1 or type(selected[0].get("status")) is not int or selected[0]["status"] != 0:
                raise TransportError("metadata_provider_changed", outcome_unknown=False)
            prompt_limit = selected[0].get("max_prompt_tokens")
            if type(prompt_limit) is not int or prompt_limit <= 0:
                raise TransportError("metadata_prompt_limit_unverified", outcome_unknown=False)
            if trial["cost_plan"]["input_token_plan_bound"] > prompt_limit:
                raise TransportError("metadata_token_limit", outcome_unknown=False)
            evidence = _fetch_actual_metadata(model=snapshot["model"], frozen_metadata=frozen,
                payload=snapshot, prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"],
                opener=static_open, timeout=20)
            models = [row for row in documents[t7_transport.MODELS_URL]["data"]
                      if row["id"] == snapshot["model"]]
            reasoning = models[0].get("reasoning") if len(models) == 1 else None
            if type(reasoning) is not dict or reasoning.get("mandatory") is not False:
                raise TransportError("reasoning_disable_support_unverified", outcome_unknown=False)
            if (Decimal(evidence["prompt_price_usd"]) != Decimal("0.00000003")
                    or Decimal(evidence["completion_price_usd"]) != Decimal("0.00000013")):
                raise TransportError("amended_price_tier_changed", outcome_unknown=False)
            assessment.update(supported=True, response_format_mode="json_object",
                json_schema_enforcement_claimed=False,
                required_fields=sorted(set(snapshot) - {"model", "messages", "provider", "stream"}))
            report["passed_payload_sha256"].append(digest)
        except TransportError as error:
            assessment["blocking_reason"] = error.code
            report["blocking_reasons"].append(error.code)
        assessments[digest] = assessment
    if any(item["request_sha256"] not in assessments for item in items):
        raise TransportError("amended_payload_unchecked", outcome_unknown=False)
    report["assessments"] = list(assessments.values())
    report["distinct_payloads_checked"] = len(assessments)
    report["blocking_reasons"] = sorted(set(report["blocking_reasons"]))
    report["passed_payload_sha256"].sort()
    report["parameter_compatibility_verified"] = len(report["passed_payload_sha256"]) == 65
    report["ready_to_send"] = bool(online) and not report["blocking_reasons"]
    return report


class CheckedTransport:
    def __init__(self, transport, bounds, *, qualification=None):
        self.transport, self.bounds = transport, MappingProxyType(dict(bounds))
        self.qualified = None
        if qualification is not None:
            if (qualification.get("schema") != "aria_e1_json_object_preflight_v1"
                    or qualification.get("response_policy_id") != POLICY_ID
                    or qualification.get("contract_json_object") != CONTRACT
                    or qualification.get("server_field_schema_enforced") is not False
                    or qualification.get("ready_to_send") is not True
                    or qualification.get("parameter_compatibility_verified") is not True
                    or set(qualification.get("passed_payload_sha256", [])) != set(bounds)
                    or qualification.get("metadata_scope") != "live_public_get"):
                raise TransportError("actual_payload_not_qualified", outcome_unknown=False)
            self.qualified = frozenset(qualification["passed_payload_sha256"])

    def request(self, payload):
        snapshot, digest, _, _, _ = _snapshot(deepcopy(payload))
        if digest not in self.bounds or (self.qualified is not None and digest not in self.qualified):
            raise TransportError("request_changed", outcome_unknown=False)
        bound = self.bounds[digest]
        if type(bound) is not int or bound < 0:
            raise TransportError("invalid_prompt_bound", outcome_unknown=False)
        envelope = self.transport.request(snapshot)
        if canonical_sha256(snapshot) != digest:
            raise TransportError("request_changed", receipt=envelope)
        return validate_receipt(snapshot, envelope, prompt_token_bound=bound)


def parse_generation(envelope, trial, case):
    from aria.bench.prompt_footprint import measure_output_fields
    from aria.engine.policy.online_decision import apply_single_pass_decision
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    schema_sha = canonical_sha256(trial["response_field_schema"])
    if trial["field_schema_sha256"] != schema_sha:
        raise TransportError("response_schema_changed", outcome_unknown=False)
    carrier, raw, visible = extract_decision(trial["wire"], envelope,
                                             field_schema=trial["response_field_schema"])
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
        "field_schema_sha256": schema_sha,
    }


def run_generation(directory, ledger, bundle, transport, *, source_check,
                   sleep=time.sleep, qualification=None):
    function = run_e1_t71.legacy.run_generation
    original_write = function.__globals__["write_private"]

    def registered_write(path, value):
        if path.name == "results.private.json":
            value.update(schema="aria_e1_json_object_live_result_v1", response_policy_id=POLICY_ID,
                response_carriers=["content_json"], historical_receipts_reclassified=False,
                server_field_schema_enforced=False)
        return original_write(path, value)

    def registered_row(item):
        return function.__globals__["generation_row"](item) | {
            "response_policy_id": POLICY_ID, "response_carrier": None}

    adapted = _clone(function, prepare_e1=prepare_e1, parse_generation=parse_generation,
        generation_row=registered_row, write_private=registered_write)
    _, _, _, items = prepare_e1(bundle)
    checked = CheckedTransport(transport,
        {item["request_sha256"]: item["input_token_plan_bound"] for item in items},
        qualification=qualification)
    return adapted(directory, ledger, bundle, checked, source_check=source_check, sleep=sleep)
