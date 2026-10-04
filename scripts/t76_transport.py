"""Single-attempt transport with content-free choice shape diagnostics.

No raw rejected reply, tool id/name, provider trace, or hidden text is added to
the diagnostic. Numeric usage and already sanitized identities survive a
choice failure. Only actual object arguments and actual null tool_calls gain
explicit representation normalization; malformed JSON is never repaired.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from types import SimpleNamespace

from scripts import t7_transport as frozen
from scripts import t74_transport as previous

TransportError = previous.TransportError
ENDPOINT = previous.ENDPOINT
SHAPE_SCHEMA = "aria_choice_shape_v1"
MAX_SHAPE_ITEMS = 8
MAX_SHAPE_COUNT = 1000000
_MISSING = object()
_FINISH = frozenset({"stop", "tool_calls", "length", "content_filter", "error"})
_PART_TYPES = frozenset({"text", "reasoning", "reasoning_details", "reasoning_content",
    "thinking", "analysis", "image_url", "audio", "input_audio", "video_url", "refusal"})
_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _kind(value):
    if value is _MISSING:
        return "missing"
    return {type(None): "null", str: "string", dict: "object", list: "array",
        bool: "boolean", int: "number", float: "number"}.get(type(value), "unsupported")


def _count(value):
    return min(len(value), MAX_SHAPE_COUNT)


def _category(value, allowed):
    kind = _kind(value)
    if kind != "string":
        return kind
    return value if value in allowed else "unsupported"


def _identifier(value, pattern):
    kind = _kind(value)
    if kind != "string":
        return kind
    return "valid" if pattern.fullmatch(value) else "invalid_format"


class _JsonShapeError(ValueError):
    pass


def _json_shape(value):
    """Only fixed reasons, with strict duplicate/nonfinite/depth inspection."""
    if type(value) not in (str, dict):
        return "not_object_json"
    try:
        if type(value) is str:
            encoded = value.encode("utf-8")
            if len(encoded) > frozen.MAX_RESPONSE_BYTES:
                return "too_large"

            def pairs(items):
                result = {}
                for key, item in items:
                    if key in result:
                        raise _JsonShapeError("duplicate_key")
                    result[key] = item
                return result

            def invalid(_):
                raise _JsonShapeError("nonfinite_number")

            parsed = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)
        else:
            # Objects came from strict outer JSON decoding. This also rejects
            # injected non-JSON objects, nonfinite numbers and bad Unicode.
            encoded = frozen._json_bytes(value)
            if len(encoded) > frozen.MAX_RESPONSE_BYTES:
                return "too_large"
            parsed = value
        if type(parsed) is not dict:
            return "not_object_json"
        pending = [(parsed, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 64:
                return "depth_limit"
            if type(item) is dict:
                pending.extend((part, depth + 1) for part in item.values())
            elif type(item) is list:
                pending.extend((part, depth + 1) for part in item)
        return "hidden_field_or_marker" if frozen._has_hidden(parsed) else "object"
    except _JsonShapeError as exc:
        return str(exc)
    except UnicodeError:
        return "invalid_unicode"
    except RecursionError:
        return "depth_limit"
    except (ValueError, TypeError, OverflowError):
        return "malformed_json"


def _content_shape(content):
    result = {"kind": _kind(content)}
    if type(content) is str:
        result.update(empty=content == "", json_shape=_json_shape(content))
    elif type(content) is list:
        result.update(parts_count=_count(content), parts_truncated=len(content) > MAX_SHAPE_ITEMS,
            parts=[{"kind": _kind(part),
                "type": _category(part.get("type", _MISSING), _PART_TYPES),
                "text_kind": _kind(part.get("text", _MISSING))}
                if type(part) is dict else {"kind": _kind(part)}
                for part in content[:MAX_SHAPE_ITEMS]])
    return result


def _call_shape(call):
    result = {"kind": _kind(call)}
    if type(call) is not dict:
        return result
    result.update(type=_category(call.get("type", _MISSING), {"function"}),
        id=_identifier(call.get("id", _MISSING), _ID),
        function_kind=_kind(call.get("function", _MISSING)))
    function = call.get("function")
    if type(function) is dict:
        name = function.get("name", _MISSING)
        name_shape = _identifier(name, _NAME)
        if name_shape == "valid":
            name_shape = "expected_decision_name" if name == "aria_turn_decision" else "other_valid_name"
        arguments = function.get("arguments", _MISSING)
        result.update(name=name_shape, arguments_kind=_kind(arguments),
            arguments_json_shape=_json_shape(arguments))
    return result


def choice_shape(value):
    """Bounded counts/categories only; never copy untrusted field values."""
    choices = value.get("choices", _MISSING) if type(value) is dict else _MISSING
    result = {"schema": SHAPE_SCHEMA, "choices_kind": _kind(choices), "normalizations": []}
    if type(choices) is not list:
        return result
    result.update(choices_count=_count(choices), choices_truncated=len(choices) > MAX_SHAPE_ITEMS,
                  choices=[])
    for choice in choices[:MAX_SHAPE_ITEMS]:
        shape = {"kind": _kind(choice)}
        result["choices"].append(shape)
        if type(choice) is not dict:
            continue
        index = choice.get("index", _MISSING)
        shape.update(index="valid_zero" if type(index) is int and index == 0 else
            "nonzero" if type(index) is int else _kind(index),
            finish_reason=_category(choice.get("finish_reason", _MISSING), _FINISH),
            message_kind=_kind(choice.get("message", _MISSING)))
        message = choice.get("message")
        if type(message) is not dict:
            continue
        shape.update(role=_category(message.get("role", _MISSING), {"assistant"}),
            content=_content_shape(message.get("content", _MISSING)))
        calls = message.get("tool_calls", _MISSING)
        shape["tool_calls_kind"] = _kind(calls)
        if type(calls) is list:
            shape.update(tool_calls_count=_count(calls), tool_calls_truncated=len(calls) > MAX_SHAPE_ITEMS,
                         calls=[_call_shape(call) for call in calls[:MAX_SHAPE_ITEMS]])
    return result


def _compatible_choices(value):
    """Normalize representations, not meaning or the response field schema."""
    copied = deepcopy(value)
    if type(copied) is list:
        for choice in copied:
            if type(choice) is not dict or type(choice.get("message")) is not dict:
                continue
            message = choice["message"]
            if "tool_calls" in message and message["tool_calls"] is None:
                # null states that no native calls were emitted. Removing the
                # empty representation never creates a call or a JSON body.
                del message["tool_calls"]
            calls = message.get("tool_calls")
            if type(calls) is list:
                for call in calls:
                    if type(call) is not dict or type(call.get("function")) is not dict:
                        continue
                    function = call["function"]
                    if type(function.get("arguments")) is dict:
                        try:
                            function["arguments"] = frozen._json_bytes(function["arguments"]).decode("utf-8")
                        except (ValueError, TypeError, UnicodeError, RecursionError):
                            raise ValueError from None
    return frozen._choices(copied)


_sanitize_core = previous._clone(frozen.sanitize_response, _choices=_compatible_choices)


def _normalizations(value):
    result = set()
    choices = value.get("choices") if type(value) is dict else None
    if type(choices) is not list:
        return []
    for choice in choices[:MAX_SHAPE_ITEMS]:
        message = choice.get("message") if type(choice) is dict else None
        if type(message) is not dict:
            continue
        if "tool_calls" in message and message["tool_calls"] is None:
            result.add("null_tool_calls_as_absent")
        calls = message.get("tool_calls")
        if type(calls) is list:
            for call in calls[:MAX_SHAPE_ITEMS]:
                function = call.get("function") if type(call) is dict else None
                if type(function) is dict and type(function.get("arguments")) is dict:
                    result.add("object_arguments_serialized_as_json")
    return sorted(result)


def sanitize_response(value, *, model):
    shape = choice_shape(value)
    shape["normalizations"] = _normalizations(value)
    try:
        clean = _sanitize_core(value, model=model)
    except TransportError as exc:
        receipt = dict(exc.receipt)
        receipt["choice_shape"] = shape
        raise TransportError(exc.code, outcome_unknown=exc.outcome_unknown, receipt=receipt) from None
    clean["choice_shape"] = shape
    return clean


def _decode(body):
    try:
        return frozen._decode(body)
    except TransportError as exc:
        try:
            status = _json_shape(body.decode("utf-8"))
        except UnicodeError:
            status = "invalid_unicode"
        raise TransportError(exc.code, outcome_unknown=exc.outcome_unknown, receipt={
            "choice_shape": {"schema": SHAPE_SCHEMA, "body_json_shape": status,
                "choices_kind": "unavailable", "normalizations": []}}) from None


def validate_json_mode_payload(payload):
    """Validate one exact prospective wire; return the actual bytes to send.

    Projection exists only to reuse the old route/type gate. The JSON-mode
    field remains in the returned bytes and must pass the caller's full actual
    parameter preflight. No tool or alternate JSON format can use this path.
    """
    keys = {"model", "messages", "max_tokens", "stream", "provider", "reasoning",
            "temperature", "response_format"}
    if (type(payload) is not dict or set(payload) != keys
            or payload.get("model") != "qwen/qwen3.7-flash"
            or payload.get("response_format") != {"type": "json_object"}
            or type(payload.get("max_tokens")) is not int or payload["max_tokens"] != 1024
            or type(payload.get("temperature")) not in (int, float) or payload["temperature"] != 0.2):
        raise TransportError("invalid_json_mode_payload", outcome_unknown=False)
    frozen._validate_payload({key: value for key, value in payload.items() if key != "response_format"})
    return frozen._json_bytes(payload)


def _validate_payload(payload):
    if type(payload) is dict and "response_format" in payload:
        return validate_json_mode_payload(payload)
    return frozen._validate_payload(payload)


# The original request's single open, bounded reads, credential-echo rejection,
# HTTP error categories and safe identities stay intact. No frozen global is
# changed; only a private view chooses this prospective response sanitizer.
_private_frozen = SimpleNamespace(**vars(frozen))
_private_frozen.sanitize_response = sanitize_response
_private_frozen._decode = _decode
_private_frozen._validate_payload = _validate_payload


class OpenRouterTransport(previous.OpenRouterTransport):
    request = previous._clone(previous.OpenRouterTransport.request, frozen=_private_frozen)
