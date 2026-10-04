"""T7.2 actual-payload metadata checks, separate from every frozen executor.

This module never reads credentials or submits inference. Public metadata checks
reuse every T7 identity, route, price, parameter and token-limit gate. A saved
metadata assessment is only static evidence, never permission to send a request.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json

from scripts import t7_transport


POLICY_ID = "aria_t72_actual_payload_policy_v1"
TransportError = t7_transport.TransportError
_ROUTING_FIELDS = frozenset({"model", "messages", "provider", "stream"})


def _snapshot(payload):
    """Capture exactly the canonical bytes that the original transport sends."""
    try:
        raw = t7_transport._validate_payload(payload)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        if isinstance(error, TransportError):
            raise
        raise TransportError("invalid_payload", outcome_unknown=False) from None
    snapshot = json.loads(raw)
    if snapshot["model"] == "qwen/qwen3.7-flash":
        if snapshot.get("parallel_tool_calls") is not False:
            raise TransportError("unfrozen_parallel_tool_calls", outcome_unknown=False)
    elif "parallel_tool_calls" in snapshot:
        raise TransportError("unfrozen_parallel_tool_calls", outcome_unknown=False)
    return snapshot, hashlib.sha256(raw).hexdigest()


def verify_payload_binding(payload, assessment):
    """Reject any change to assessed wire bytes; this is not a send gate itself."""
    _, digest = _snapshot(payload)
    if (not isinstance(assessment, dict)
            or assessment.get("policy_id") != POLICY_ID
            or assessment.get("actual_payload_sha256") != digest):
        raise TransportError("actual_payload_changed", outcome_unknown=False)
    return digest


def _binding(snapshot, digest):
    return {"policy_id": POLICY_ID, "actual_payload_sha256": digest,
            "required_fields": sorted(set(snapshot) - _ROUTING_FIELDS),
            "actual_wire_field_removed": False,
            "provider_hard_token_bound_verified": False,
            "provider_billing_guarantee": False,
            "post_response_usage_check_required": True}


def fetch_metadata(*, model, frozen_metadata, payload, prompt_token_bound,
                   opener=None, enable_network=False, timeout=120):
    """Check the exact request, including false-valued fields, with T7 gates.

    The snapshot passed to the original checker is unprojected and independent
    of caller-owned objects. The returned SHA must still be matched immediately
    before any future sender serializes its request. Failure carries bounded
    metadata diagnostics; it does not turn missing evidence into support.
    """
    snapshot, digest = _snapshot(payload)
    binding = _binding(snapshot, digest)
    try:
        result = t7_transport.fetch_metadata(model=model,
            frozen_metadata=deepcopy(frozen_metadata), payload=snapshot,
            prompt_token_bound=prompt_token_bound, opener=opener,
            enable_network=enable_network, timeout=timeout)
    except TransportError as error:
        prefix = "parameter_support_unverified_"
        missing = [error.code[len(prefix):]] if error.code.startswith(prefix) else []
        receipt = {**error.receipt, **binding, "supported": False,
                   "missing_fields": missing,
                   "assessment_scope": "public_metadata_preflight"}
        # A callback/concurrent owner may change the caller's request during GET.
        verify_payload_binding(payload, binding)
        raise TransportError(error.code, outcome_unknown=False, receipt=receipt) from None
    verify_payload_binding(payload, binding)
    return {**result, **binding, "supported": True, "missing_fields": [],
            "assessment_scope": "public_metadata_preflight"}


class _StaticResponse:
    """An in-memory HTTP-shaped source for the unchanged T7 validation code."""

    def __init__(self, document, url):
        self._body = io.BytesIO(t7_transport._json_bytes(document))
        self._url = url
        self.headers = {}

    def read1(self, size):
        return self._body.read(size)

    def close(self):
        self._body.close()

    def geturl(self):
        return self._url

    def getcode(self):
        return 200


def assess_payload_support(*, model, frozen_metadata, payload, prompt_token_bound):
    """Assess saved frozen metadata locally, without credentials or any network.

    The original T7 checker also verifies the saved model/provider/pricing tier
    and limits. The complete missing-field list is a static inventory finding;
    success is explicitly historical metadata support, not live routability.
    """
    snapshot, digest = _snapshot(payload)
    binding = _binding(snapshot, digest)
    metadata = deepcopy(frozen_metadata)
    required = binding["required_fields"]
    missing = []
    documents = {}
    try:
        documents[t7_transport.MODELS_URL] = {"data": metadata["models"]}
        saved = [row["data"] for row in metadata["selected_endpoints"]
                 if row["data"]["id"] == model]
        if len(saved) != 1:
            raise ValueError
        endpoint_url = t7_transport.MODELS_URL + "/" + model + "/endpoints"
        documents[endpoint_url] = {"data": saved[0]}
        provider = snapshot["provider"]["only"][0]
        endpoints = [row for row in saved[0]["endpoints"] if row["tag"] == provider]
        if len(endpoints) != 1:
            raise ValueError
        supported = endpoints[0]["supported_parameters"]
        if not isinstance(supported, list) or any(not isinstance(x, str) for x in supported):
            raise ValueError
        missing = sorted(set(required) - set(supported))
    except (ValueError, TypeError, KeyError, IndexError):
        return {**binding, "supported": False, "missing_fields": missing,
                "stop_reason": "invalid_metadata", "network_requests": 0,
                "authenticated": False, "inference_requests": 0,
                "assessment_scope": "offline_frozen_metadata"}

    def static_opener(request, *, timeout):
        if request.get_method() != "GET" or request.get_header("Authorization") is not None:
            raise TransportError("invalid_static_metadata_request", outcome_unknown=False)
        return _StaticResponse(documents[request.full_url], request.full_url)

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
