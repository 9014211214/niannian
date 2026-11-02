"""Explicit, metadata-bound offline reasoning; no network or model activation.

The completion ceiling includes reasoning. Only token counts may leave the
response boundary; excluded reasoning text is rejected, never persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256

from .online_structured import _metadata

MODEL = "google/gemini-4-argon"
PROVIDER = "google-vertex/global"
_REQUIRED = {"reasoning", "max_tokens", "temperature", "response_format", "structured_outputs"}


def _price(value):
    try:
        if type(value) is not str:
            raise ValueError
        result = Decimal(value)
        if not result.is_finite() or result <= 0:
            raise ValueError
        return result
    except (ValueError, InvalidOperation):
        raise ValueError("invalid_offline_reasoning_price") from None


@dataclass(frozen=True)
class OfflineReasoningRoute:
    model_raw: bytes
    endpoints_raw: bytes
    model: str = MODEL
    provider: str = PROVIDER
    effort: str = "medium"

    def __post_init__(self):
        self.binding()

    def binding(self):
        if self.model != MODEL or self.provider != PROVIDER or self.effort != "medium":
            raise ValueError("unsupported_offline_reasoning_route")
        model = _metadata(self.model_raw).get("data")
        data = _metadata(self.endpoints_raw).get("data")
        if (type(model) is not dict or model.get("id") != self.model
                or type(data) is not dict or data.get("id") != self.model
                or type(data.get("endpoints")) is not list
                or any(type(row) is not dict for row in data["endpoints"])):
            raise ValueError("invalid_offline_reasoning_metadata")
        selected = [row for row in data["endpoints"] if row.get("tag") == self.provider]
        if len(selected) != 1:
            raise ValueError("unique_offline_reasoning_endpoint_required")
        endpoint = selected[0]
        reasoning = model.get("reasoning")
        parameters = endpoint.get("supported_parameters")
        if (type(reasoning) is not dict or reasoning.get("mandatory") is not True
                or type(reasoning.get("supported_efforts")) is not list
                or self.effort not in reasoning["supported_efforts"]
                or endpoint.get("model_id") != self.model
                or type(endpoint.get("status")) is not int or endpoint["status"] != 0
                or type(parameters) is not list or any(type(p) is not str for p in parameters)
                or not _REQUIRED <= set(parameters)
                or type(endpoint.get("max_completion_tokens")) is not int
                or endpoint["max_completion_tokens"] < 64
                or any(type(value) is not str or not value for value in (
                    model.get("canonical_slug"), endpoint.get("name"), endpoint.get("provider_name")))):
            raise ValueError("offline_reasoning_endpoint_not_supported")
        prices = endpoint.get("pricing")
        if type(prices) is not dict:
            raise ValueError("invalid_offline_reasoning_price")
        prompt, completion, reasoning_price = (_price(prices.get(k))
                                              for k in ("prompt", "completion", "internal_reasoning"))
        # A separate or higher reasoning tariff cannot inherit the ordinary
        # completion reservation. This candidate only supports equal tariffs.
        if reasoning_price != completion:
            raise ValueError("unsupported_offline_reasoning_tariff")
        return {"version": "offline_reasoning_route_v1", "model": self.model,
                "provider": self.provider, "effort": self.effort, "exclude": True,
                "completion_bound_includes_reasoning": True,
                "max_completion_tokens": endpoint["max_completion_tokens"],
                "canonical_slug": model["canonical_slug"], "endpoint_name": endpoint["name"],
                "provider_name": endpoint["provider_name"],
                "input_usd_per_million": str(prompt * 1000000),
                "output_usd_per_million": str(completion * 1000000),
                "required_parameters": sorted(_REQUIRED),
                "model_metadata_sha256": sha256(self.model_raw).hexdigest(),
                "endpoint_metadata_sha256": sha256(self.endpoints_raw).hexdigest()}


def build_offline_reasoning_route(model_raw, endpoints_raw, *, model=MODEL,
                                  provider=PROVIDER, effort="medium"):
    return OfflineReasoningRoute(model_raw, endpoints_raw, model, provider, effort)


def validate_offline_reasoning_client(client):
    route = getattr(client, "offline_reasoning_route", None)
    if (type(route) is not OfflineReasoningRoute or client.role != "reflection"
            or client.model != route.model or client.reasoning_effort != route.effort
            or not client.bounded_transport or client.max_retries != 0 or client.fallback_models
            or client.reflection_return_format != "body_json" or client.online_return_format != "native"):
        raise ValueError("invalid_offline_reasoning_client")
    binding = route.binding()
    if (type(client.max_completion_tokens) is not int
            or not 64 <= client.max_completion_tokens <= binding["max_completion_tokens"]):
        raise ValueError("invalid_offline_reasoning_completion_bound")
    return binding


def validate_offline_reasoning_response(data, *, max_completion_tokens):
    if type(data) is not dict or type(data.get("usage")) is not dict:
        raise ValueError("offline_reasoning_usage_missing")
    usage = data["usage"]
    total = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if type(details) is dict else None
    if (type(total) is not int or not 0 <= total <= max_completion_tokens
            or type(reasoning) is not int or not 0 <= reasoning <= total):
        raise ValueError("offline_reasoning_usage_invalid")
    choices = data.get("choices")
    if type(choices) is not list:
        raise ValueError("offline_reasoning_response_invalid")
    for choice in choices:
        message = choice.get("message") if type(choice) is dict else None
        if (type(message) is not dict or message.get("reasoning") not in (None, "")
                or message.get("reasoning_details") not in (None, "", [])):
            raise ValueError("excluded_reasoning_returned")
    return {"completion_tokens": total, "reasoning_tokens": reasoning,
            "completion_bound_includes_reasoning": True, "reasoning_text_retained": False}
