"""Explicit provider-schema candidate with immutable public capability evidence.

This module neither fetches metadata nor enables a model call. Public support
is a preparation prerequisite, not proof of provider compilation or effect.
The provider receives the complete local schema, including optional fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .online_decision import EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE, ONLINE_DECISION_VERSIONS, online_decision_tool_spec

ONLINE_STRUCTURED_FORMAT = "json_schema_v1"
ONLINE_JSON_FORMATS = frozenset({"json_object", ONLINE_STRUCTURED_FORMAT})
DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash-1017"
DEEPSEEK_PROVIDER = "together"
DEEPSEEK_PROVIDER_NAMES = {"together": "Together", "fireworks": "Fireworks"}
_REQUIRED = frozenset({"structured_outputs", "response_format", "max_tokens", "reasoning", "temperature"})


def _metadata(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_online_capability_field")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite_online_capability_value")

    if type(raw) is not bytes or not 1 <= len(raw) <= 10 * 1024 * 1024:
        raise ValueError("invalid_online_capability_metadata")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, TypeError, RecursionError, json.JSONDecodeError):
        raise ValueError("invalid_online_capability_metadata") from None
    if type(value) is not dict:
        raise ValueError("invalid_online_capability_metadata")
    return value


@dataclass(frozen=True)
class OnlineStructuredRoute:
    """Saved public metadata and an explicit model/provider selection."""

    models_raw: bytes
    endpoints_raw: bytes
    model: str
    provider: str

    def __post_init__(self):
        self.binding()

    def binding(self) -> dict:
        if (type(self.model) is not str or not re.fullmatch(r"[a-zA-Z0-9_./:-]{1,160}", self.model)
                or type(self.provider) is not str or not re.fullmatch(r"[a-z0-9_/-]{1,80}", self.provider)):
            raise ValueError("invalid_online_structured_route")
        models = _metadata(self.models_raw).get("data")
        data = _metadata(self.endpoints_raw).get("data")
        if (type(models) is not list or any(type(row) is not dict for row in models)
                or type(data) is not dict or data.get("id") != self.model):
            raise ValueError("invalid_online_capability_metadata")
        matches = [row for row in models if row.get("id") == self.model]
        endpoints = data.get("endpoints")
        if len(matches) != 1 or type(endpoints) is not list or any(type(row) is not dict for row in endpoints):
            raise ValueError("invalid_online_capability_metadata")
        selected = [row for row in endpoints if row.get("tag") == self.provider]
        if len(selected) != 1:
            raise ValueError("unique_online_structured_endpoint_required")
        model, endpoint = matches[0], selected[0]
        parameters = endpoint.get("supported_parameters")
        if (endpoint.get("model_id") != self.model or type(endpoint.get("status")) is not int
                or endpoint["status"] != 0 or type(parameters) is not list
                or any(type(p) is not str for p in parameters) or not _REQUIRED <= set(parameters)
                or type(model.get("reasoning")) is not dict or model["reasoning"].get("mandatory") is not False
                or type(model.get("canonical_slug")) is not str or not model["canonical_slug"]
                or type(endpoint.get("name")) is not str or not endpoint["name"]
                or type(endpoint.get("provider_name")) is not str or not endpoint["provider_name"]):
            raise ValueError("online_structured_endpoint_not_supported")
        return {"format": ONLINE_STRUCTURED_FORMAT, "model": self.model, "provider": self.provider,
                "endpoint_name": endpoint["name"], "canonical_slug": model["canonical_slug"],
                "provider_name": endpoint["provider_name"], "required_parameters": sorted(_REQUIRED),
                "models_metadata_sha256": sha256(self.models_raw).hexdigest(),
                "endpoint_metadata_sha256": sha256(self.endpoints_raw).hexdigest()}


def build_online_structured_route(models_raw: bytes, endpoints_raw: bytes, *, model: str,
                                 provider: str) -> OnlineStructuredRoute:
    return OnlineStructuredRoute(models_raw, endpoints_raw, model, provider)


def online_structured_request_format(version: str, *, require_preference_control: bool = False, require_task_projection: bool = False) -> dict:
    # A current acknowledged v4 invitation has a distinct, request-local schema.
    # No other optional fields, nulls or constraints are changed.
    schema = online_decision_tool_spec(
        version, require_preference_control=require_preference_control, require_task_projection=require_task_projection,
    )["function"]["parameters"]
    return {"type": "json_schema", "json_schema": {
        "name": "aria_" + version, "strict": True, "schema": schema,
    }}


def validate_online_structured_client(client) -> dict:
    route = getattr(client, "online_structured_route", None)
    from .qwen_max_structured import QwenMaxStructuredRoute, validate_qwen_max_client
    if type(route) is QwenMaxStructuredRoute:
        if client.role != "online_fast":
            raise ValueError("invalid_online_structured_client")
        return validate_qwen_max_client(client)
    if client.model == DEEPSEEK_MODEL or getattr(route, "model", None) == DEEPSEEK_MODEL:
        if client.role != "online_fast":
            raise ValueError("invalid_online_structured_client")
        return validate_release_online_client(client)
    if (type(route) is not OnlineStructuredRoute or client.role != "online_fast"
            or client.model != route.model or not client.bounded_transport
            or client.max_retries != 0 or client.fallback_models or client.reasoning_effort != "none"):
        raise ValueError("invalid_online_structured_client")
    return route.binding()


def validate_release_online_client(client) -> dict:
    """Validate the explicit release route without changing historical Max."""
    from .qwen_max_structured import qwen_max_route, validate_qwen_max_client
    if qwen_max_route(client) is not None:
        return validate_qwen_max_client(client)
    role = getattr(client, "role", None)
    attribute = {"online_fast": "online_structured_route",
                 "safety_review": "safety_structured_route",
                 "confirm_judge": "confirm_structured_route"}.get(role)
    route = getattr(client, attribute, None) if attribute else None
    tokens = getattr(client, "max_completion_tokens", None)
    valid_tokens = (type(tokens) is int and
                    (64 <= tokens <= 4096 if role == "online_fast" else
                     tokens == 1024 if role == "safety_review" else tokens == 60))
    if (attribute is None or type(route) is not OnlineStructuredRoute
            or client.model != DEEPSEEK_MODEL or route.model != DEEPSEEK_MODEL
            or route.provider not in DEEPSEEK_PROVIDER_NAMES or client.reasoning_effort != "none"
            or client.bounded_transport is not True
            or type(client.max_retries) is not int or client.max_retries != 0
            or client.fallback_models != [] or not valid_tokens
            or type(client.timeout) not in (int, float) or not 0 < client.timeout <= 90
            or (role == "confirm_judge" and (type(client.temperature) not in (int, float)
                                             or client.temperature != 0))
            or client.reflection_return_format != "native"
            or client.online_return_format != (ONLINE_STRUCTURED_FORMAT if role == "online_fast" else "native")):
        raise ValueError("invalid_release_online_client")
    binding = route.binding()
    if binding["provider_name"] != DEEPSEEK_PROVIDER_NAMES[route.provider]:
        raise ValueError("invalid_release_online_client")
    return binding


def online_structured_schema_binding() -> str:
    # Preserve all supported protocol shapes in the durable option binding.
    # Adding or changing any contract requires a newly registered allowance.
    formats = {version: online_structured_request_format(version) for version in ONLINE_DECISION_VERSIONS}
    for version in EVIDENCE_DECISION_VERSIONS:
        formats[version + ":preference_control_required"] = (
            online_structured_request_format(version, require_preference_control=True))
        for preference in (False, True):
            formats[version + ":task_projection_required:" + str(preference)] = (
                online_structured_request_format(version,
                    require_preference_control=preference, require_task_projection=True))
    return sha256(json.dumps(formats, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
