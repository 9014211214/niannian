"""Prospective JSON-object E1 with its original field schema in the prompt.

Only the existing plain_json return-channel system message gains the canonical
frozen parameters schema and a uniform exact-enum instruction. No response is
repaired, old receipt upgraded, reference exposed, or provider guarantee added.
The JSON-object API mode still guarantees syntax only; local strict acceptance,
production delivery, fees, metadata gates, and permanent stopping remain intact.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
import json
import time
from types import MappingProxyType, SimpleNamespace

from aria.bench.prompt_footprint import measure_messages
from scripts import e1_json_mode as previous, run_e1_t71, t7_transport
from scripts.t76_transport import validate_json_mode_payload
from scripts.verify_experiment_freeze import canonical_sha256

TransportError = t7_transport.TransportError
POLICY_ID = "aria_e1_json_schema_prompt_response_v1"
SCHEMA_SECTION = "\n\n## 原冻结返回字段Schema：e1_json_schema_prompt_v1\n"
SCHEMA_INSTRUCTION = (
    "以下是aria_turn_decision原冻结parameters的完整JSON Schema，仅限定本次正文JSON的字段格式；"
    "它不是回复示例，不改变人物理解、回应策略、医疗规则或外部动作权限。"
    "输出对象必须满足所有原字段、required、类型、范围、数组约束与additionalProperties限制。"
    "所有enum值必须逐字匹配列出的一个值；不得拼接中文解释、别名、空格或其他后缀。"
    "解释只能写入Schema允许的说明字段；不要输出Schema本身或额外字段。\n"
)
CONTRACT = deepcopy(previous.CONTRACT) | {
    "prompt_field_schema_embedded": True,
    "prompt_field_schema_source": "unchanged_frozen_tool_parameters",
    "prompt_field_schema_serialization": "canonical_utf8_json",
    "exact_enum_values_required": True,
    "field_schema_prompt_is_provider_enforcement": False,
}
_clone = previous._clone
_fetch_actual_metadata = previous._fetch_actual_metadata


def _schema_suffix(schema):
    text = json.dumps(schema, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return (SCHEMA_SECTION + SCHEMA_INSTRUCTION
            + "原字段Schema SHA256=" + canonical_sha256(schema) + "\n" + text)


def _prepare(bundle):
    suite, cases, originals, old_items = previous.prepare_e1(bundle)
    prepared = {}
    for key, original in originals.items():
        trial = deepcopy(original)
        schema = trial["response_field_schema"]
        if canonical_sha256(schema) != trial["field_schema_sha256"]:
            raise TransportError("response_schema_changed", outcome_unknown=False)
        old_messages = original["wire"]["messages"]
        positions = [index for index, message in enumerate(old_messages)
                     if message.get("role") == "system"
                     and "## 客户端指定的唯一返回通道：" in message.get("content", "")
                     and "唯一通道=plain_json" in message.get("content", "")]
        if len(positions) != 1 or any(SCHEMA_SECTION in message.get("content", "")
                                     for message in old_messages):
            raise TransportError("unregistered_prompt_schema_change", outcome_unknown=False)
        position = positions[0]
        trial["wire"]["messages"][position]["content"] += _schema_suffix(schema)
        new_messages = trial["wire"]["messages"]
        changed = [index for index, pair in enumerate(zip(old_messages, new_messages))
                   if pair[0] != pair[1]]
        if (len(old_messages) != len(new_messages) or changed != [position]
                or {k: v for k, v in trial["wire"].items() if k != "messages"}
                   != {k: v for k, v in original["wire"].items() if k != "messages"}):
            raise TransportError("unregistered_prompt_schema_change", outcome_unknown=False)
        raw = validate_json_mode_payload(trial["wire"])
        bound = len(raw) + 8192
        reserve = Decimal(bound) * Decimal("0.00000003") + Decimal(1024) * Decimal("0.00000013")
        if bound > bundle["plan"]["e1"]["max_prompt_tokens"] or reserve > Decimal("0.0011"):
            raise TransportError("e1_reservation_insufficient", outcome_unknown=False)
        trial["t76_request_sha256"] = canonical_sha256(original["wire"])
        trial["prompt_field_schema_sha256"] = trial["field_schema_sha256"]
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
        item["t76_request_sha256"] = item["request_sha256"]
        item["request_sha256"] = canonical_sha256(trial["wire"])
        item["input_token_plan_bound"] = trial["cost_plan"]["input_token_plan_bound"]
        if (item["t76_request_sha256"] != trial["t76_request_sha256"]
                or item["t74_request_sha256"] != trial["t74_request_sha256"]
                or item["original_request_sha256"] != trial["original_request_sha256"]
                or item["field_schema_sha256"] != trial["field_schema_sha256"]):
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
    if len(items) != 204 or len({item["request_sha256"] for item in items}) != 65:
        raise TransportError("amended_inventory_changed", outcome_unknown=False)
    return suite, cases, prepared, items


def prepare_e1(bundle):
    return _prepare(bundle)


@lru_cache(maxsize=1)
def _bindings():
    # Only inherited frozen inputs are cached. The entry point verifies sources
    # and history before each claim; caller-provided schemas never define this.
    _, _, prepared, items = _prepare(run_e1_t71.load_bundle())
    return MappingProxyType({item["request_sha256"]: (
        item["t76_request_sha256"], item["t74_request_sha256"], item["original_request_sha256"],
        t7_transport._json_bytes(prepared[item["case_id"], item["arm"]]["response_field_schema"]))
        for item in items})


def _snapshot(payload):
    raw = validate_json_mode_payload(payload)
    snapshot = t7_transport._decode(raw)
    digest = canonical_sha256(snapshot)
    binding = _bindings().get(digest)
    if binding is None:
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    _, t74_sha, original_sha, schema_bytes = binding
    return snapshot, digest, t74_sha, original_sha, t7_transport._decode(schema_bytes)


extract_decision = _clone(previous.extract_decision, _snapshot=_snapshot)
_validate_receipt = _clone(previous._validate_receipt,
    previous=SimpleNamespace(_snapshot=_snapshot), extract_decision=extract_decision)


def validate_receipt(payload, envelope, *, prompt_token_bound):
    return _validate_receipt(payload, envelope, prompt_token_bound=prompt_token_bound)


parse_generation = _clone(previous.parse_generation,
    extract_decision=extract_decision, POLICY_ID=POLICY_ID)


def preflight(bundle, *, online=False, opener=None, documents=None):
    adapted = _clone(previous.preflight, prepare_e1=prepare_e1, _snapshot=_snapshot,
        _fetch_actual_metadata=_fetch_actual_metadata, POLICY_ID=POLICY_ID, CONTRACT=CONTRACT)
    report = adapted(bundle, online=online, opener=opener, documents=documents)
    report["schema"] = "aria_e1_json_schema_prompt_preflight_v1"
    report["prompt_field_schema_embedded"] = True
    report["all_non_format_messages_unchanged"] = True
    for assessment in report["assessments"]:
        assessment["t76_request_sha256"] = _bindings()[assessment["actual_payload_sha256"]][0]
    return report


class CheckedTransport(previous.CheckedTransport):
    def __init__(self, transport, bounds, *, qualification=None):
        self.transport, self.bounds = transport, MappingProxyType(dict(bounds))
        self.qualified = None
        if qualification is not None:
            if (qualification.get("schema") != "aria_e1_json_schema_prompt_preflight_v1"
                    or qualification.get("response_policy_id") != POLICY_ID
                    or qualification.get("contract_json_object") != CONTRACT
                    or qualification.get("prompt_field_schema_embedded") is not True
                    or qualification.get("server_field_schema_enforced") is not False
                    or qualification.get("ready_to_send") is not True
                    or qualification.get("parameter_compatibility_verified") is not True
                    or qualification.get("post_response_single_native_tool_required") is not False
                    or set(qualification.get("passed_payload_sha256", [])) != set(bounds)
                    or qualification.get("metadata_scope") != "live_public_get"):
                raise TransportError("actual_payload_not_qualified", outcome_unknown=False)
            self.qualified = frozenset(qualification["passed_payload_sha256"])

    request = _clone(previous.CheckedTransport.request,
                     _snapshot=_snapshot, validate_receipt=validate_receipt)


def run_generation(directory, ledger, bundle, transport, *, source_check,
                   sleep=time.sleep, qualification=None):
    function = run_e1_t71.legacy.run_generation
    original_write = function.__globals__["write_private"]

    def registered_write(path, value):
        if path.name == "results.private.json":
            value.update(schema="aria_e1_json_schema_prompt_live_result_v1", response_policy_id=POLICY_ID,
                response_carriers=["content_json"], historical_receipts_reclassified=False,
                server_field_schema_enforced=False, prompt_field_schema_embedded=True)
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
