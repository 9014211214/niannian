"""Bounded online JSON shape diagnostics, never response values or repairs.

This is an observer of the existing decision schema. The product parser remains
the acceptance authority; this module is called only when recording a failure.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math

from .online_decision import ONLINE_DECISION_VERSIONS, online_decision_tool_spec

_KINDS = frozenset({"schema", "json_syntax", "duplicate_key", "non_text", "body_too_large",
                    "return_channel", "unsupported_schema", "unknown"})
_TYPES = frozenset({"missing", "extra_forbidden", "object_type", "array_type", "string_type",
    "number_type", "boolean_type", "minimum", "maximum", "min_length", "max_length", "min_items",
    "max_items", "enum", "nonfinite_number", "number_overflow", "empty_response", "schema_constraint"})
_OBSERVED = frozenset({"object", "array", "string", "number", "boolean", "null", "missing", "unknown"})
_SCHEMA_KEYS = frozenset({"type", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "enum", "minLength", "maxLength", "minimum", "maximum"})


@lru_cache(maxsize=1)
def _fields():
    fields = {"unknown_field"}
    def visit(schema):
        properties = schema.get("properties", {})
        fields.update(properties)
        for child in properties.values():
            visit(child)
        if "items" in schema:
            visit(schema["items"])
    for version in ONLINE_DECISION_VERSIONS:
        visit(online_decision_tool_spec(version)["function"]["parameters"])
    return frozenset(fields)


def sanitize_online_shape_diagnostic(value):
    """Retain only fixed categories and schema-owned path components."""
    if type(value) is not dict or type(value.get("kind")) is not str or value["kind"] not in _KINDS:
        return {}
    result = {"kind": value["kind"], "issues": []}
    if type(value.get("offset")) is int and 0 <= value["offset"] <= 131072:
        result["offset"] = value["offset"]
    if type(value.get("truncated")) is bool:
        result["truncated"] = value["truncated"]
    rows = value.get("issues")
    for row in rows[:20] if type(rows) is list else []:
        if type(row) is not dict or type(row.get("path")) is not list:
            continue
        code = row.get("type")
        code = code if type(code) is str and code in _TYPES else "schema_constraint"
        path = [item if (type(item) is str and item in _fields()
                        or type(item) is int and 0 <= item <= 63) else "unknown_field"
                for item in row["path"][:12]]
        output = {"type": code, "path": path}
        if type(row.get("observed_type")) is str and row["observed_type"] in _OBSERVED:
            output["observed_type"] = row["observed_type"]
        result["issues"].append(output)
    return result


def online_shape_diagnostic(raw, version):
    """Report violated field constraints without echoing any model field value.

    Object paths come exclusively from the trusted schema, including for
    missing fields. Unknown generated key names are always ``unknown_field``.
    """
    if type(raw) is not str:
        return {"kind": "non_text", "issues": []}
    try:
        if len(raw.encode("utf-8")) > 128 * 1024:
            return {"kind": "body_too_large", "issues": []}
        duplicates = False
        def unique(pairs):
            nonlocal duplicates
            result = {}
            for key, value in pairs:
                duplicates = duplicates or key in result
                result[key] = value
            return result
        try:
            parsed = json.loads(raw, object_pairs_hook=unique)
        except json.JSONDecodeError as error:
            return sanitize_online_shape_diagnostic({"kind": "json_syntax", "offset": error.pos})
        if duplicates:
            return {"kind": "duplicate_key", "issues": []}
        if version not in ONLINE_DECISION_VERSIONS:
            return {"kind": "unsupported_schema", "issues": []}
        schema = online_decision_tool_spec(version)["function"]["parameters"]
        result = {"kind": "schema", "issues": [], "truncated": False}

        def typename(value):
            return {dict: "object", list: "array", str: "string", int: "number",
                    float: "number", bool: "boolean", type(None): "null"}.get(type(value), "unknown")

        def issue(code, path, value, *, missing=False):
            if len(result["issues"]) >= 20:
                result["truncated"] = True
                return
            result["issues"].append({"type": code, "path": path,
                "observed_type": "missing" if missing else typename(value)})

        def visit(value, node, path):
            if len(result["issues"]) >= 20:
                result["truncated"] = True
                return
            if type(node) is not dict or set(node) - _SCHEMA_KEYS:
                result["kind"] = "unsupported_schema"
                return
            kind = node.get("type")
            expected = {"object": (dict,), "array": (list,), "string": (str,),
                        "number": (int, float), "boolean": (bool,)}.get(kind)
            if expected is None:
                result["kind"] = "unsupported_schema"
                return
            if type(value) not in expected:
                issue(kind + "_type", path, value)
                return
            if "enum" in node and value not in node["enum"]:
                issue("enum", path, value)
            if kind == "object":
                properties = node.get("properties", {})
                for key in node.get("required", []):
                    if key not in value:
                        issue("missing", path + [key], None, missing=True)
                if node.get("additionalProperties") is False and set(value) - set(properties):
                    issue("extra_forbidden", path + ["unknown_field"], value)
                for key, child in properties.items():
                    if key in value:
                        visit(value[key], child, path + [key])
            elif kind in {"array", "string"}:
                suffix = "Items" if kind == "array" else "Length"
                if len(value) < node.get("min" + suffix, 0):
                    issue("min_items" if kind == "array" else "min_length", path, value)
                if len(value) > node.get("max" + suffix, math.inf):
                    issue("max_items" if kind == "array" else "max_length", path, value)
                if kind == "array":
                    for index, child in enumerate(value[:64]):
                        visit(child, node["items"], path + [index])
                    result["truncated"] = result["truncated"] or len(value) > 64
            elif kind == "number":
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    issue("number_overflow", path, value)
                    return
                if not finite:
                    issue("nonfinite_number", path, value)
                elif value < node.get("minimum", -math.inf):
                    issue("minimum", path, value)
                elif value > node.get("maximum", math.inf):
                    issue("maximum", path, value)
        visit(parsed, schema, [])
        if type(parsed) is dict and type(parsed.get("response")) is str and not parsed["response"].strip():
            issue("empty_response", ["response"], parsed["response"])
        return sanitize_online_shape_diagnostic(result)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return {"kind": "unknown", "issues": []}
