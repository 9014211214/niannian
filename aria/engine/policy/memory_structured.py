"""Explicit same-Sonnet memory schema capability, never live qualification."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json

from .online_structured import _metadata, DEEPSEEK_MODEL
from ..memory.memory_contract import FACT_PROTOCOLS, VERSION, SOURCE_REFS_VERSION, memory_response_schema, validate_memory_response
from ..memory.preference_sources import LEGACY, SOURCE_REFS, source_protocol

MODEL = "anthropic/claude-sonnet-5"
PROVIDER = "anthropic"
OFFLINE_ROUTES = {(MODEL, PROVIDER): "Anthropic", (DEEPSEEK_MODEL, "together"): "Together"}
_REQUIRED = frozenset({"structured_outputs", "response_format", "max_tokens", "reasoning"})
_INPUT_PRICES = ("prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h")
_PRICES = frozenset({*_INPUT_PRICES, "completion", "internal_reasoning", "web_search"})


def _prices(raw):
    if type(raw) is not dict:
        raise ValueError("invalid_memory_structured_pricing")
    result = {}
    for key, value in raw.items():
        if key == "discount":
            continue  # A discount cannot reduce the reservation.
        if type(key) is not str or type(value) is not str:
            raise ValueError("invalid_memory_structured_pricing")
        try:
            amount = Decimal(value)
        except InvalidOperation:
            raise ValueError("invalid_memory_structured_pricing") from None
        if not amount.is_finite() or amount < 0 or key not in _PRICES and amount != 0:
            raise ValueError("unsupported_memory_structured_pricing")
        result[key] = str(amount)
    if not {"prompt", "completion"} <= result.keys() or any(Decimal(result[k]) <= 0 for k in ("prompt", "completion")):
        raise ValueError("invalid_memory_structured_pricing")
    return result


@dataclass(frozen=True)
class MemoryStructuredRoute:
    models_raw: bytes
    endpoints_raw: bytes
    model: str = MODEL
    provider: str = PROVIDER

    def __post_init__(self):
        self.binding()

    def binding(self):
        if (self.model, self.provider) not in OFFLINE_ROUTES:
            raise ValueError("unsupported_memory_structured_route")
        models = _metadata(self.models_raw).get("data")
        data = _metadata(self.endpoints_raw).get("data")
        if (type(models) is not list or any(type(row) is not dict for row in models)
                or type(data) is not dict or data.get("id") != self.model
                or type(data.get("endpoints")) is not list
                or any(type(row) is not dict for row in data["endpoints"])):
            raise ValueError("invalid_memory_structured_metadata")
        matches = [row for row in models if row.get("id") == self.model]
        endpoints = [row for row in data["endpoints"] if row.get("tag") == self.provider]
        if len(matches) != 1 or len(endpoints) != 1:
            raise ValueError("unique_memory_structured_endpoint_required")
        model, endpoint = matches[0], endpoints[0]
        parameters = endpoint.get("supported_parameters")
        if (endpoint.get("model_id") != self.model or type(endpoint.get("status")) is not int
                or endpoint["status"] != 0 or type(parameters) is not list
                or any(type(p) is not str for p in parameters) or not _REQUIRED <= set(parameters)
                or type(model.get("reasoning")) is not dict or model["reasoning"].get("mandatory") is not False
                or type(endpoint.get("max_completion_tokens")) is not int or endpoint["max_completion_tokens"] < 4000
                or any(type(v) is not str or not v for v in (model.get("canonical_slug"), endpoint.get("name")))
                or endpoint.get("provider_name") != OFFLINE_ROUTES[(self.model, self.provider)]):
            raise ValueError("memory_structured_endpoint_not_supported")
        prices = _prices(endpoint.get("pricing"))
        return {"version": "memory_structured_route_v1", "format": VERSION, "model": self.model,
            "provider": self.provider, "provider_name": endpoint["provider_name"], "canonical_slug": model["canonical_slug"],
            "endpoint_name": endpoint["name"], "required_parameters": sorted(_REQUIRED),
            "pricing": prices,
            "input_usd_per_million": str(max(Decimal(prices.get(k, "0")) for k in _INPUT_PRICES) * 1000000),
            "output_usd_per_million": str(max(Decimal(prices["completion"]), Decimal(prices.get("internal_reasoning", "0"))) * 1000000),
            "web_search_enabled": False, "provider_schema_compilation_observed": False,
            "models_metadata_sha256": sha256(self.models_raw).hexdigest(),
            "endpoint_metadata_sha256": sha256(self.endpoints_raw).hexdigest()}


def build_memory_structured_route(models_raw, endpoints_raw, *, model=MODEL, provider=PROVIDER):
    return MemoryStructuredRoute(models_raw, endpoints_raw, model, provider)


def memory_schema_protocol(client):
    protocol = source_protocol(client)
    selected = getattr(client, "memory_fact_source_protocol", None)
    if selected is None:
        return protocol
    route = memory_structured_route(client)
    from .max_release.reflection_sse import STRUCTURED_STREAM_VERSION
    if (selected not in FACT_PROTOCOLS or protocol != SOURCE_REFS or route is None
            or client.model != DEEPSEEK_MODEL or route.model != DEEPSEEK_MODEL or route.provider != "together"
            or getattr(client, "structured_stream_protocol", None) != STRUCTURED_STREAM_VERSION):
        raise ValueError("unsupported_memory_fact_source_protocol")
    return selected


def memory_request_format(client=None):
    protocol = memory_schema_protocol(client)
    version = {LEGACY: VERSION, SOURCE_REFS: SOURCE_REFS_VERSION}.get(protocol, protocol)
    return {"type": "json_schema", "json_schema": {
        "name": "aria_" + version, "strict": True, "schema": memory_response_schema(source_protocol=protocol)}}


def memory_structured_route(client):
    route = getattr(client, "memory_structured_route", None)
    return route if type(route) is MemoryStructuredRoute else None


def validate_memory_structured_client(client):
    route = memory_structured_route(client)
    if (route is None or client.role != "memory" or client.model != route.model
            or client.bounded_transport is not True or type(client.max_retries) is not int or client.max_retries != 0
            or client.fallback_models != [] or client.reasoning_effort != "none"
            or client.online_return_format != "native" or client.reflection_return_format != "native"
            or client.memory_prefix_recovery is not True or getattr(client, "offline_reasoning_route", None) is not None
            or type(client.max_completion_tokens) is not int or not 64 <= client.max_completion_tokens <= 4000
            or type(client.timeout) not in (int, float) or client.timeout != 90):
        raise ValueError("invalid_memory_structured_client")
    selected = memory_request_format(client)
    binding = route.binding()
    if source_protocol(client) == SOURCE_REFS:
        binding.update(format=selected["json_schema"]["name"].removeprefix("aria_"), preference_source_protocol=SOURCE_REFS)
    return {**binding, "memory_schema_sha256": sha256(json.dumps(selected,
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def validate_memory_structured_response(data, client=None):
    choices = data.get("choices") if type(data) is dict else None
    if (type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict
            or choices[0].get("finish_reason") != "stop"):
        raise ValueError("invalid_memory_structured_channel")
    message = choices[0].get("message")
    if (type(message) is not dict or message.get("tool_calls") not in (None, [])
            or message.get("function_call") is not None or message.get("refusal") not in (None, "")
            or any(message.get(k) not in (None, "", []) for k in ("reasoning", "reasoning_details", "reasoning_content"))):
        raise ValueError("invalid_memory_structured_channel")
    validate_memory_response(message.get("content"), source_protocol=memory_schema_protocol(client))
    return message["content"]
