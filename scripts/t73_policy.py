"""Prospective E1 wire amendment: omit only the parallel-tool request field.

The original 204 positions and stopped campaign stay unchanged. This module
prepares linked payloads and validates them, but never reads credentials, writes
a campaign, sends inference, or resolves a historical unknown charge.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json

from scripts import run_e1_t71 as previous
from scripts import t7_transport, t71_policy, t72_policy


POLICY_ID = "aria_t73_omit_parallel_field_v1"
TransportError = t7_transport.TransportError
_MODEL = "qwen/qwen3.7-flash"
_ROUTING_FIELDS = frozenset({"model", "messages", "provider", "stream"})


def _original_records():
    """Trust only the original manifest's fixed digest, not caller-supplied IDs."""
    try:
        manifest = previous.legacy.read_json(previous.ROOT / previous.legacy.T6_PATH)
        if (manifest.get("seal_sha256") != previous.T6_SEAL
                or previous.freeze.canonical_sha256(
                    {k: v for k, v in manifest.items() if k != "seal_sha256"}) != previous.T6_SEAL
                or len(manifest["e1_requests"]) != 204):
            raise ValueError
        return manifest["e1_requests"]
    except (ValueError, TypeError, KeyError, OSError):
        raise TransportError("original_inventory_unverified", outcome_unknown=False) from None


def _snapshot(payload, *, original_hashes=None):
    try:
        raw = t7_transport._validate_payload(payload)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        if isinstance(error, TransportError):
            raise
        raise TransportError("invalid_payload", outcome_unknown=False) from None
    snapshot = json.loads(raw)
    if snapshot["model"] != _MODEL or "parallel_tool_calls" in snapshot:
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    restored = {**snapshot, "parallel_tool_calls": False}
    original_sha = previous.freeze.canonical_sha256(restored)
    if original_hashes is None:
        original_hashes = {r["request_sha256"] for r in _original_records()}
    if original_sha not in original_hashes:
        raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
    return snapshot, hashlib.sha256(raw).hexdigest(), original_sha


def _binding(snapshot, digest, original_sha):
    return {"policy_id": POLICY_ID, "actual_payload_sha256": digest,
            "original_request_sha256": original_sha,
            "wire_changes": [{"field": "parallel_tool_calls", "from": False, "to": "omitted"}],
            "required_fields": sorted(set(snapshot) - _ROUTING_FIELDS),
            "provider_hard_token_bound_verified": False,
            "provider_billing_guarantee": False,
            "post_response_usage_check_required": True,
            "single_native_tool_receipt_required": True}


def verify_payload_binding(payload, assessment):
    """Require both the exact amended SHA and its fixed original ancestor."""
    _, digest, original_sha = _snapshot(payload)
    if (not isinstance(assessment, dict) or assessment.get("policy_id") != POLICY_ID
            or assessment.get("actual_payload_sha256") != digest
            or assessment.get("original_request_sha256") != original_sha):
        raise TransportError("actual_payload_changed", outcome_unknown=False)
    return digest


def prepare_e1(bundle):
    """Verify the old inventory first; return new copies with exactly one omission."""
    suite, cases, original_prepared, original_items = previous.prepare_e1(bundle)
    records = _original_records()
    if (len(original_items) != len(records)
            or any(any(item.get(key) != value for key, value in record.items())
                   for item, record in zip(original_items, records))):
        raise TransportError("original_inventory_unverified", outcome_unknown=False)
    original_hashes = {r["request_sha256"] for r in records}
    expected_by_trial = {(r["case_id"], r["arm"]): r["request_sha256"] for r in records}
    if set(original_prepared) != set(expected_by_trial):
        raise TransportError("original_inventory_unverified", outcome_unknown=False)
    prepared = {}
    for key, original_trial in original_prepared.items():
        trial = deepcopy(original_trial)
        if trial["wire"].get("parallel_tool_calls") is not False:
            raise TransportError("unregistered_wire_amendment", outcome_unknown=False)
        del trial["wire"]["parallel_tool_calls"]
        _, _, original_sha = _snapshot(trial["wire"], original_hashes=original_hashes)
        if original_sha != expected_by_trial[key]:
            raise TransportError("original_inventory_unverified", outcome_unknown=False)
        trial["original_request_sha256"] = original_sha
        trial["cost_plan"] = t71_policy.estimate_request(trial["wire"],
            input_price="0.00000003", output_price="0.00000013")
        if (trial["cost_plan"]["input_token_plan_bound"] > bundle["plan"]["e1"]["max_prompt_tokens"]
                or Decimal(trial["cost_plan"]["reserved_cost_usd"]) > Decimal("0.0011")):
            raise TransportError("e1_reservation_insufficient", outcome_unknown=False)
        prepared[key] = trial
    items = deepcopy(original_items)
    for item in items:
        trial = prepared[item["case_id"], item["arm"]]
        item["original_request_sha256"] = item["request_sha256"]
        item["request_sha256"] = previous.freeze.canonical_sha256(trial["wire"])
        item["input_token_plan_bound"] = trial["cost_plan"]["input_token_plan_bound"]
    return suite, cases, prepared, items


def validate_receipt(payload, envelope, *, prompt_token_bound):
    """Keep T7.1's usage bound and exactly-one aria_turn_decision requirement."""
    _snapshot(payload)
    return t71_policy.validate_receipt(payload, envelope, prompt_token_bound=prompt_token_bound)


def fetch_metadata(*, model, frozen_metadata, payload, prompt_token_bound,
                   opener=None, enable_network=False, timeout=120):
    """Check the new actual wire directly, never a smaller preflight projection."""
    snapshot, digest, original_sha = _snapshot(payload)
    binding = _binding(snapshot, digest, original_sha)
    try:
        result = t7_transport.fetch_metadata(model=model, frozen_metadata=deepcopy(frozen_metadata),
            payload=snapshot, prompt_token_bound=prompt_token_bound, opener=opener,
            enable_network=enable_network, timeout=timeout)
    except TransportError as error:
        prefix = "parameter_support_unverified_"
        missing = [error.code[len(prefix):]] if error.code.startswith(prefix) else []
        verify_payload_binding(payload, binding)
        raise TransportError(error.code, outcome_unknown=False,
            receipt={**error.receipt, **binding, "supported": False, "missing_fields": missing,
                     "assessment_scope": "public_metadata_preflight"}) from None
    verify_payload_binding(payload, binding)
    return {**result, **binding, "supported": True, "missing_fields": [],
            "assessment_scope": "public_metadata_preflight"}


def assess_payload_support(*, model, frozen_metadata, payload, prompt_token_bound):
    """Reuse all metadata gates on saved documents, with no network or send gate."""
    snapshot, digest, original_sha = _snapshot(payload)
    binding = _binding(snapshot, digest, original_sha)
    metadata = deepcopy(frozen_metadata)
    missing = []
    try:
        documents = {t7_transport.MODELS_URL: {"data": metadata["models"]}}
        saved = [row["data"] for row in metadata["selected_endpoints"] if row["data"]["id"] == model]
        if len(saved) != 1:
            raise ValueError
        documents[t7_transport.MODELS_URL + "/" + model + "/endpoints"] = {"data": saved[0]}
        endpoints = [row for row in saved[0]["endpoints"] if row["tag"] == "alibaba"]
        if len(endpoints) != 1:
            raise ValueError
        supported = endpoints[0]["supported_parameters"]
        if not isinstance(supported, list) or any(not isinstance(x, str) for x in supported):
            raise ValueError
        missing = sorted(set(binding["required_fields"]) - set(supported))
    except (ValueError, TypeError, KeyError, IndexError):
        return {**binding, "supported": False, "missing_fields": missing,
                "stop_reason": "invalid_metadata", "network_requests": 0,
                "authenticated": False, "inference_requests": 0,
                "assessment_scope": "offline_frozen_metadata"}

    def static_opener(request, *, timeout):
        if request.get_method() != "GET" or request.get_header("Authorization") is not None:
            raise TransportError("invalid_static_metadata_request", outcome_unknown=False)
        return t72_policy._StaticResponse(documents[request.full_url], request.full_url)

    try:
        result = fetch_metadata(model=model, frozen_metadata=metadata, payload=snapshot,
            prompt_token_bound=prompt_token_bound, opener=static_opener)
        reason = None
    except TransportError as error:
        result, reason = error.receipt, error.code
    verify_payload_binding(payload, binding)
    return {**result, **binding, "supported": reason is None, "missing_fields": missing,
            "stop_reason": reason, "network_requests": 0, "authenticated": False,
            "inference_requests": 0, "assessment_scope": "offline_frozen_metadata"}
