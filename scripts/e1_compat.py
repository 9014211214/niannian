"""Actual E1 auto-tool wire and complete, credential-free compatibility gate.

The original frozen stimuli and all old attempts stay immutable. Only the two
registered wire differences are accepted. Public metadata qualifies the exact
payloads; successful metadata is never a promise of account routing or billing.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import time
from types import FunctionType, MappingProxyType
from urllib.request import Request

from scripts import check_e1_tool_choice as choices
from scripts import run_e1_t71 as previous
from scripts import t7_transport, t71_policy, t72_policy, t73_policy
from scripts.verify_experiment_freeze import canonical_sha256

TransportError = t7_transport.TransportError
POLICY_ID = "aria_e1_auto_tool_compat_v1"
ORIGINAL_CHOICE = {"type": "function", "function": {"name": "aria_turn_decision"}}
WIRE_CHANGES = [
    {"field": "parallel_tool_calls", "from": False, "to": "omitted"},
    {"field": "tool_choice", "from": ORIGINAL_CHOICE, "to": "auto"},
]


def _snapshot(payload):
    raw = t7_transport._validate_payload(payload)
    snapshot = json.loads(raw)
    if (snapshot["model"] != choices.MODEL or "parallel_tool_calls" in snapshot
            or snapshot.get("tool_choice") != "auto"):
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    restored = {**snapshot, "parallel_tool_calls": False, "tool_choice": ORIGINAL_CHOICE}
    original_sha = canonical_sha256(restored)
    if original_sha not in {r["request_sha256"] for r in t73_policy._original_records()}:
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    return snapshot, hashlib.sha256(raw).hexdigest(), original_sha


def prepare_e1(bundle):
    suite, cases, original, old_items = previous.prepare_e1(bundle)
    prepared = {}
    for key, old in original.items():
        trial = deepcopy(old)
        if (trial["wire"].get("parallel_tool_calls") is not False
                or trial["wire"].get("tool_choice") != ORIGINAL_CHOICE):
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
        del trial["wire"]["parallel_tool_calls"]
        trial["wire"]["tool_choice"] = "auto"
        _, _, original_sha = _snapshot(trial["wire"])
        if original_sha != canonical_sha256(old["wire"]):
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
        trial["original_request_sha256"] = original_sha
        trial["cost_plan"] = t71_policy.estimate_request(trial["wire"],
            input_price="0.00000003", output_price="0.00000013")
        if (trial["cost_plan"]["input_token_plan_bound"] > bundle["plan"]["e1"]["max_prompt_tokens"]
                or Decimal(trial["cost_plan"]["reserved_cost_usd"]) > Decimal("0.0011")):
            raise TransportError("e1_reservation_insufficient", outcome_unknown=False)
        prepared[key] = trial
    items = deepcopy(old_items)
    for item in items:
        trial = prepared[item["case_id"], item["arm"]]
        item["original_request_sha256"] = item["request_sha256"]
        if item["original_request_sha256"] != trial["original_request_sha256"]:
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
        item["request_sha256"] = canonical_sha256(trial["wire"])
        item["input_token_plan_bound"] = trial["cost_plan"]["input_token_plan_bound"]
    if len(items) != 204 or len({i["request_sha256"] for i in items}) != 65:
        raise TransportError("amended_inventory_changed", outcome_unknown=False)
    return suite, cases, prepared, items


def validate_receipt(payload, envelope, *, prompt_token_bound):
    _snapshot(payload)
    return t71_policy.validate_receipt(payload, envelope, prompt_token_bound=prompt_token_bound)


def _documents(*, opener=None, enable_network=False):
    if opener is None and enable_network is not True:
        raise TransportError("network_disabled", outcome_unknown=False)
    documents, sources = {}, []
    started = datetime.now(timezone.utc).isoformat()
    for url in (t7_transport.MODELS_URL, choices.ENDPOINT):
        request = Request(url, method="GET", headers={
            "Accept": "application/json", "Accept-Encoding": "identity"})
        raw = t7_transport._exchange(request, opener=opener or t7_transport._opener(),
                                     timeout=20, limit=choices.MAX_METADATA_BYTES)
        documents[url] = t7_transport._decode(raw)
        sources.append({"url": url, "response_sha256": hashlib.sha256(raw).hexdigest()})
    return documents, {"started_at_utc": started,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "network_requests": 2, "authenticated": False, "inference_requests": 0}


def preflight(bundle, *, online=False, opener=None, documents=None):
    """Check each distinct actual payload using the same two public responses.

    Saved documents permit offline fault tests and diagnosis, never a live gate.
    No credential reader, POST or campaign mutation is reachable here.
    """
    _, _, prepared, items = prepare_e1(bundle)
    report = {"schema": "aria_e1_auto_tool_preflight_v1", "policy_id": POLICY_ID,
        "fixed_positions": len(items), "distinct_payloads_checked": 0,
        "linked_inventory_sha256": canonical_sha256(items), "wire_changes": deepcopy(WIRE_CHANGES),
        "model_requests": 0, "credentials_read": False, "ready_to_send": False,
        "parameter_compatibility_verified": False, "account_route_verified": False,
        "provider_privacy_filter_retained": "deny", "privacy_filter_acceptance_verified": False,
        "provider_cost_guarantee": False, "post_response_single_native_tool_required": True,
        "planned_full_e1_reserve_usd": str(sum((Decimal(i["reserve_usd"]) for i in items), Decimal(0))),
        "budget_usd": deepcopy(bundle["amendment"]["budget_usd"]),
        "blocking_reasons": [], "passed_payload_sha256": [], "assessments": []}
    if online:
        if documents is not None:
            raise TransportError("live_gate_cannot_use_saved_documents", outcome_unknown=False)
        documents, receipt = _documents(opener=opener, enable_network=True)
        report["metadata"] = receipt
        report["metadata_scope"] = "live_public_get"
    elif documents is None:
        report["blocking_reasons"].append("live_metadata_not_checked")
        return report
    else:
        report["metadata_scope"] = "saved_public_snapshot"
        report["blocking_reasons"].append("live_metadata_not_checked")
    frozen = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")

    def static_open(request, *, timeout):
        if request.get_method() != "GET" or request.get_header("Authorization") is not None:
            raise TransportError("invalid_static_metadata_request", outcome_unknown=False)
        return t72_policy._StaticResponse(documents[request.full_url], request.full_url)

    assessments = {}
    for trial in prepared.values():
        snapshot, digest, original_sha = _snapshot(trial["wire"])
        if digest in assessments:
            continue
        assessment = {"actual_payload_sha256": digest, "original_request_sha256": original_sha,
                      "supported": False, "blocking_reason": None}
        try:
            selected = [row for row in documents[choices.ENDPOINT]["data"]["endpoints"]
                        if row.get("tag") == choices.PROVIDER]
            if (len(selected) != 1 or type(selected[0].get("status")) is not int
                    or selected[0]["status"] != 0):
                raise TransportError("metadata_provider_changed", outcome_unknown=False)
            prompt_limit = selected[0].get("max_prompt_tokens")
            if type(prompt_limit) is not int or prompt_limit <= 0:
                raise TransportError("metadata_prompt_limit_unverified", outcome_unknown=False)
            if trial["cost_plan"]["input_token_plan_bound"] > prompt_limit:
                raise TransportError("metadata_token_limit", outcome_unknown=False)
            evidence = t7_transport.fetch_metadata(model=snapshot["model"], frozen_metadata=frozen,
                payload=snapshot, prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"],
                opener=static_open, timeout=20)
            mode = choices.assess_tool_choice(snapshot, documents[choices.ENDPOINT])
            if not mode["supported"]:
                raise TransportError(mode["blocking_reason"], outcome_unknown=False)
            models = [row for row in documents[t7_transport.MODELS_URL]["data"]
                      if row["id"] == snapshot["model"]]
            reasoning = models[0].get("reasoning") if len(models) == 1 else None
            if not isinstance(reasoning, dict) or reasoning.get("mandatory") is not False:
                raise TransportError("reasoning_disable_support_unverified", outcome_unknown=False)
            if (Decimal(evidence["prompt_price_usd"]) != Decimal("0.00000003")
                    or Decimal(evidence["completion_price_usd"]) != Decimal("0.00000013")):
                raise TransportError("amended_price_tier_changed", outcome_unknown=False)
            assessment.update(supported=True, tool_choice_mode=mode["required_mode"],
                mode_supported=mode["advertised_support"],
                required_fields=sorted(set(snapshot) - {"model", "messages", "provider", "stream"}))
            report["passed_payload_sha256"].append(digest)
        except TransportError as error:
            assessment["blocking_reason"] = error.code
            report["blocking_reasons"].append(error.code)
        assessments[digest] = assessment
    if any(i["request_sha256"] not in assessments for i in items):
        raise TransportError("amended_payload_unchecked", outcome_unknown=False)
    report["assessments"] = list(assessments.values())
    report["distinct_payloads_checked"] = len(assessments)
    report["blocking_reasons"] = sorted(set(report["blocking_reasons"]))
    report["passed_payload_sha256"].sort()
    report["parameter_compatibility_verified"] = len(report["passed_payload_sha256"]) == 65
    # The execution layer separately evaluates historical costs and credentials.
    report["ready_to_send"] = bool(online) and not report["blocking_reasons"]
    return report


class CheckedTransport:
    def __init__(self, transport, bounds, *, qualification=None):
        self.transport = transport
        self.bounds = MappingProxyType(dict(bounds))
        self.qualified = None
        if qualification is not None:
            if (qualification.get("ready_to_send") is not True
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
        _snapshot(snapshot)
        envelope = self.transport.request(snapshot)
        if canonical_sha256(snapshot) != digest:
            raise TransportError("request_changed", receipt=envelope)
        return validate_receipt(snapshot, envelope, prompt_token_bound=bound)


def run_generation(directory, ledger, bundle, transport, *, source_check,
                   sleep=time.sleep, qualification=None):
    function = previous.legacy.run_generation
    adapted = FunctionType(function.__code__, {**function.__globals__, "prepare_e1": prepare_e1},
        function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    _, _, _, items = prepare_e1(bundle)
    checked = CheckedTransport(transport,
        {item["request_sha256"]: item["input_token_plan_bound"] for item in items},
        qualification=qualification)
    return adapted(directory, ledger, bundle, checked, source_check=source_check, sleep=sleep)
