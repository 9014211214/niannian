"""Explicit Max capability evidence, separate from non-reasoning Flash routes.

This module does not fetch metadata, qualify an actual request, or authorize
model calls. Mandatory reasoning is kept explicit; metadata alone cannot prove
the provider's visible/reasoning token accounting or an executable fee bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256

from .online_structured import _metadata

MODEL = "qwen/qwen3.8-max-0902"
PROVIDER = "alibaba"
EFFORT = "minimal"
REASONING_TOKEN_PLAN = 4096
TOKEN_MARGIN = 10
_REQUIRED = frozenset({"structured_outputs", "response_format", "max_tokens", "reasoning", "temperature"})


def _prices(value):
    if type(value) is not dict:
        raise ValueError("invalid_qwen_max_pricing")
    prices = {}
    for key, raw in value.items():
        if key == "discount":
            # A discount never reduces the pre-send reservation price.
            continue
        if type(key) is not str or type(raw) is not str:
            raise ValueError("invalid_qwen_max_pricing")
        try:
            amount = Decimal(raw)
        except InvalidOperation:
            raise ValueError("invalid_qwen_max_pricing") from None
        if not amount.is_finite() or amount < 0:
            raise ValueError("invalid_qwen_max_pricing")
        prices[key] = str(amount)
    if not {"prompt", "completion"} <= prices.keys() or any(
            Decimal(prices[key]) <= 0 for key in ("prompt", "completion")):
        raise ValueError("invalid_qwen_max_pricing")
    return prices


@dataclass(frozen=True)
class QwenMaxStructuredRoute:
    """A dated, fixed-provider candidate with mandatory minimal reasoning."""

    models_raw: bytes
    endpoints_raw: bytes
    model: str = MODEL
    provider: str = PROVIDER
    effort: str = EFFORT

    def __post_init__(self):
        self.binding()

    def binding(self):
        if self.model != MODEL or self.provider != PROVIDER or self.effort != EFFORT:
            raise ValueError("unsupported_qwen_max_route")
        models = _metadata(self.models_raw).get("data")
        data = _metadata(self.endpoints_raw).get("data")
        if (type(models) is not list or any(type(row) is not dict for row in models)
                or type(data) is not dict or data.get("id") != self.model
                or type(data.get("endpoints")) is not list
                or any(type(row) is not dict for row in data["endpoints"])):
            raise ValueError("invalid_qwen_max_metadata")
        matches = [row for row in models if row.get("id") == self.model]
        endpoints = [row for row in data["endpoints"] if row.get("tag") == self.provider]
        if len(matches) != 1 or len(endpoints) != 1:
            raise ValueError("unique_qwen_max_model_endpoint_required")
        model, endpoint = matches[0], endpoints[0]
        reasoning = model.get("reasoning")
        parameters = endpoint.get("supported_parameters")
        if (type(reasoning) is not dict or reasoning.get("mandatory") is not True
                or type(reasoning.get("supported_efforts")) is not list
                or any(type(effort) is not str for effort in reasoning["supported_efforts"])
                or self.effort not in reasoning["supported_efforts"]
                or endpoint.get("model_id") != self.model
                or type(endpoint.get("status")) is not int or endpoint["status"] != 0
                or type(parameters) is not list or any(type(p) is not str for p in parameters)
                or not _REQUIRED <= set(parameters)
                or type(endpoint.get("max_completion_tokens")) is not int
                or endpoint["max_completion_tokens"] < 4096
                or any(type(value) is not str or not value for value in (
                    model.get("canonical_slug"), endpoint.get("name"), endpoint.get("provider_name")))):
            raise ValueError("qwen_max_endpoint_not_supported")
        return {"version": "qwen_max_structured_route_v1", "format": "json_schema_v1",
                "model": self.model, "provider": self.provider, "effort": self.effort,
                "exclude": True, "reasoning_mandatory": True,
                "reasoning_token_plan": REASONING_TOKEN_PLAN, "shared_token_margin": TOKEN_MARGIN,
                "token_accounting_contract": "visible_plus_reasoning_documented_bound_v1",
                "gateway_mapping_observed": False,
                "canonical_slug": model["canonical_slug"], "endpoint_name": endpoint["name"],
                "provider_name": endpoint["provider_name"],
                "endpoint_max_completion_tokens": endpoint["max_completion_tokens"],
                "pricing": _prices(endpoint.get("pricing")),
                "required_parameters": sorted(_REQUIRED),
                "models_metadata_sha256": sha256(self.models_raw).hexdigest(),
                "endpoint_metadata_sha256": sha256(self.endpoints_raw).hexdigest()}


def build_qwen_max_structured_route(models_raw, endpoints_raw, *, model=MODEL,
                                    provider=PROVIDER, effort=EFFORT):
    return QwenMaxStructuredRoute(models_raw, endpoints_raw, model, provider, effort)


def qwen_max_route(client):
    """Only an exact explicit route opts into mandatory reasoning."""
    attribute = {"online_fast": "online_structured_route", "safety_review": "safety_structured_route",
                 "confirm_judge": "confirm_structured_route"}.get(client.role)
    route = getattr(client, attribute, None) if attribute else None
    return route if type(route) is QwenMaxStructuredRoute else None


def validate_qwen_max_client(client):
    route = qwen_max_route(client)
    limit = 4096 if client.role == "online_fast" else 1024
    valid_tokens = (type(client.max_completion_tokens) is int
                    and (client.max_completion_tokens == 60 if client.role == "confirm_judge"
                         else 64 <= client.max_completion_tokens <= limit))
    if (route is None or client.model != MODEL or client.reasoning_effort != EFFORT
            or not client.bounded_transport or client.max_retries != 0 or client.fallback_models
            or type(client.max_completion_tokens) is not int
            or not valid_tokens
            or (client.role == "confirm_judge" and client.temperature != 0)
            or type(client.timeout) not in (int, float) or not 0 < client.timeout <= 90
            or client.reflection_return_format != "native"
            or client.online_return_format != ("json_schema_v1" if client.role == "online_fast" else "native")):
        raise ValueError("invalid_qwen_max_client")
    binding = route.binding()
    total = client.max_completion_tokens + REASONING_TOKEN_PLAN + TOKEN_MARGIN
    if total > binding["endpoint_max_completion_tokens"]:
        raise ValueError("qwen_max_endpoint_completion_limit_exceeded")
    return {**binding, "wire_visible_limit": client.max_completion_tokens,
            "billable_completion_limit": total}


def validate_qwen_max_usage(client, data):
    binding = validate_qwen_max_client(client)
    if type(data) is not dict or type(data.get("usage")) is not dict:
        raise ValueError("qwen_max_usage_missing")
    usage = data["usage"]
    total, details = usage.get("completion_tokens"), usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if type(details) is dict else None
    if (type(total) is not int or not 0 <= total <= binding["billable_completion_limit"]
            or type(reasoning) is not int or not 0 <= reasoning <= min(total, REASONING_TOKEN_PLAN + TOKEN_MARGIN)
            or total - reasoning > binding["wire_visible_limit"] + TOKEN_MARGIN):
        raise ValueError("qwen_max_usage_invalid")
    return {"completion_tokens": total, "reasoning_tokens": reasoning,
            "visible_tokens": total - reasoning,
            "wire_visible_limit": binding["wire_visible_limit"],
            "billable_completion_limit": binding["billable_completion_limit"],
            "gateway_mapping_observed": False, "reasoning_text_retained": False}


def validate_qwen_max_response(client, data):
    counts = validate_qwen_max_usage(client, data)
    choices = data.get("choices")
    if type(choices) is not list or not choices:
        raise ValueError("qwen_max_response_invalid")
    for choice in choices:
        message = choice.get("message") if type(choice) is dict else None
        if (type(message) is not dict or message.get("reasoning") not in (None, "")
                or message.get("reasoning_details") not in (None, "", [])
                or message.get("reasoning_content") not in (None, "")):
            raise ValueError("excluded_reasoning_returned")
    return counts
