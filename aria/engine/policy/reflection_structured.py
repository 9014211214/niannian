"""Explicit Sonnet reflection schema, bound to supplied public capability data.

Reuses the same model/provider capability proof as memory, while binding the
different reflection schema and local validator. No metadata fetch or call.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .memory_structured import MemoryStructuredRoute, MODEL
from .reflection_json import reflection_request_format


def validate_reflection_structured_client(client):
    route = getattr(client, "reflection_structured_route", None)
    if (type(route) is not MemoryStructuredRoute or client.role != "reflection"
            or client.model != route.model or client.bounded_transport is not True
            or type(client.max_retries) is not int or client.max_retries != 0
            or client.fallback_models != [] or client.reasoning_effort != "none"
            or client.online_return_format != "native"
            or client.reflection_return_format not in {"json_schema_v2", "json_schema_v3", "json_schema_v4"}
            or client.reflection_itemwise_review is not True
            or client.memory_prefix_recovery is not False
            or getattr(client, "memory_structured_route", None) is not None
            or getattr(client, "offline_reasoning_route", None) is not None
            or type(client.max_completion_tokens) is not int or client.max_completion_tokens != 2800
            or type(client.timeout) not in (int, float) or client.timeout != 90):
        raise ValueError("invalid_reflection_structured_client")
    version = client.reflection_return_format
    return {**route.binding(), "version": ("reflection_structured_route_v3" if version == "json_schema_v4" else
                                          "reflection_structured_route_v2" if version == "json_schema_v3"
                                           else "reflection_structured_route_v1"),
            "format": "reflection_" + version,
            "reflection_schema_sha256": sha256(json.dumps(
                reflection_request_format(version), sort_keys=True,
                ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def validate_reflection_structured_channel(data):
    """The disabled reasoning contract applies to every known text channel."""
    choices = data.get("choices") if type(data) is dict else None
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("invalid_reflection_structured_channel")
    message = choices[0].get("message")
    if (type(message) is not dict or any(message.get(key) not in (None, "", [])
            for key in ("reasoning", "reasoning_details", "reasoning_content"))):
        raise ValueError("invalid_reflection_structured_channel")
