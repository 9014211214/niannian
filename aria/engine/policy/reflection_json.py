"""Reflection body JSON and explicitly archived provider schema formats.

The provider enforces its supported JSON Schema subset. All omitted numeric,
length and count constraints still run in the original local shape validator;
the updater remains responsible for evidence truth, scope and event identity.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pydantic import ValidationError

from ..memory.reflection_contract import (
    reflection_response_schema, validate_reflection_response_shape,
)

REFLECTION_JSON_PROTOCOL_VERSION = "reflection_json_schema_v2"
REFLECTION_BODY_PROTOCOL_VERSION = "reflection_body_json_v1"
_CORRECTION_PREFIX = "本次仅更正 JSON 格式。保持原会话证据边界，不增补证据或改写任务；重新返回完整 JSON。字段路径及错误类型：\n"
_LOCAL_ONLY = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "maxItems", "uniqueItems",
})
_WIRE_KEYWORDS = frozenset({
    "$defs", "$ref", "type", "properties", "required", "additionalProperties",
    "items", "anyOf", "const", "enum", "pattern", "minItems", "title", "description",
})
_DIAGNOSTIC_FIELDS = frozenset({
    "protocol_version", "reviews", "changes", "id", "hypothesis_id", "claim_quote", "scope_quote",
    "applicability", "relation", "evidence_basis", "event_kind", "prior_event_id", "evidence",
    "event_evidence", "reason", "event_summary", "turn_index", "evidence_quote", "review_id",
    "framework", "dimension", "claim", "scope", "scope_tags", "lineage", "mechanism_relation",
    "scope_relation", "atomic_claim", "aspect", "conditions", "tendency", "outcome", "target_value",
    "confidence", "_StatedPreference", "_ObservedResponse", "_RepeatedPattern", "unknown_field",
})
_DIAGNOSTIC_TYPES = frozenset({
    "missing", "extra_forbidden", "literal_error", "string_type", "list_type", "dict_type", "model_type",
    "string_pattern_mismatch", "string_too_short", "string_too_long", "greater_than_equal",
    "less_than_equal", "finite_number", "int_type", "float_type", "float_parsing", "int_parsing",
    "too_long", "too_short", "value_error", "schema_constraint",
})


def sanitize_reflection_shape_diagnostic(value) -> dict:
    """Allowlisted shape positions only, never input values/messages/context."""
    if type(value) is not dict:
        return {}
    kind = value.get("kind")
    if type(kind) is not str or kind not in {"schema", "json_syntax", "duplicate_key", "nonfinite_number",
                                             "non_text", "return_channel", "unknown"}:
        return {}
    result = {"kind": kind, "issues": []}
    for key in ("offset", "omitted_count"):
        if type(value.get(key)) is int and 0 <= value[key] <= 10_000_000:
            result[key] = value[key]
    issues = value.get("issues")
    for row in issues[:20] if type(issues) is list else []:
        if type(row) is not dict or type(row.get("path")) is not list:
            continue
        code = row.get("type")
        code = code if type(code) is str and code in _DIAGNOSTIC_TYPES else "schema_constraint"
        path = [item if (type(item) is str and item in _DIAGNOSTIC_FIELDS
                        or type(item) is int and 0 <= item <= 63) else "unknown_field"
                for item in row["path"][:12]]
        result["issues"].append({"type": code, "path": path})
    return result


def reflection_shape_diagnostic(error: Exception) -> dict:
    if isinstance(error, ValidationError):
        errors = error.errors(include_url=False, include_context=False, include_input=False)
        value = {"kind": "schema", "omitted_count": max(0, len(errors) - 20),
                 "issues": [{"type": row["type"], "path": list(row["loc"])} for row in errors[:20]]}
    elif type(error) is json.JSONDecodeError:
        value = {"kind": "json_syntax", "offset": error.pos}
    elif type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str:
        value = {"kind": {"duplicate_reflection_key": "duplicate_key",
                          "nonfinite_reflection_number": "nonfinite_number",
                          "reflection_text_required": "non_text",
                          "invalid_reflection_return_channel": "return_channel"}.get(error.args[0], "unknown")}
    else:
        value = {"kind": "unknown"}
    return sanitize_reflection_shape_diagnostic(value)


def _project_wire_keywords(node: dict) -> dict:
    """Project only schema keywords; never walk property names as keywords."""
    def project(node):
        if type(node) is not dict or set(node) - _WIRE_KEYWORDS - _LOCAL_ONLY:
            raise ValueError("unsupported_reflection_schema_keyword")
        result = deepcopy(node)
        local = {}
        for key in _LOCAL_ONLY:
            if key in result:
                local[key] = result.pop(key)
        if "minItems" in result and result["minItems"] not in (0, 1):
            local["minItems"] = result.pop("minItems")
        if local:
            constraint = "Local validation also requires " + json.dumps(local, sort_keys=True) + "."
            result["description"] = (result.get("description", "") + " " + constraint).strip()
        for group in ("properties", "$defs"):
            if group in result:
                result[group] = {key: project(value) for key, value in result[group].items()}
        if isinstance(result.get("items"), dict):
            result["items"] = project(result["items"])
        for group in ("anyOf",):
            if group in result:
                result[group] = [project(value) for value in result[group]]
        return result
    return project(node)


def _schema_constraints(value):
    """Ignore annotations only while proving the atomic branch factorization."""
    if isinstance(value, dict):
        return {key: _schema_constraints(item) for key, item in value.items()
                if key not in {"title", "description"}}
    if isinstance(value, list):
        return [_schema_constraints(item) for item in value]
    return value


def _factor_atomic_outcomes(schema: dict) -> dict:
    """Factor equivalent null/string branches without widening legal output.

    Required fields, common text constraints and closed objects must agree.
    A future incompatible local contract fails explicitly instead of silently
    making the provider grammar more permissive. Full local validation stays
    unchanged and still checks all omitted and cross-field constraints.
    """
    result = deepcopy(schema)
    try:
        definitions = result["$defs"]
        old_names = ("_StatedPreference", "_ObservedResponse", "_RepeatedPattern")
        stated, observed, repeated = (definitions[name] for name in old_names)
        branches = (stated, observed, repeated)
        common = []
        for branch, basis in zip(branches, ("stated_preference", "observed_response", "repeated_pattern")):
            constraints = _schema_constraints(branch)
            if (constraints["type"] != "object" or constraints["additionalProperties"] is not False
                    or set(constraints["required"]) != set(constraints["properties"])
                    or constraints["properties"]["evidence_basis"] != {"const": basis, "type": "string"}):
                raise ValueError
            del constraints["properties"]["outcome"]
            del constraints["properties"]["evidence_basis"]
            common.append(constraints)
        null_shape = _schema_constraints(stated["properties"]["outcome"])
        text_shape = _schema_constraints(observed["properties"]["outcome"])
        repeated_shape = _schema_constraints(repeated["properties"]["outcome"])
        atomic = definitions["_Change"]["properties"]["atomic_claim"]
        old_refs = {"#/$defs/" + name for name in old_names}
        def references(value):
            if isinstance(value, dict):
                return ([value["$ref"]] if value.get("$ref") in old_refs else []) + [
                    ref for item in value.values() for ref in references(item)]
            if isinstance(value, list):
                return [ref for item in value for ref in references(item)]
            return []
        if (common[0] != common[1] or common[0] != common[2]
                or null_shape != {"type": "null"} or text_shape.get("type") != "string"
                or repeated_shape != {"anyOf": [text_shape, null_shape]}
                or atomic["anyOf"] != [{"$ref": "#/$defs/" + name} for name in old_names]
                or sorted(references(result)) != sorted(old_refs)):
            raise ValueError
        new_names = ("_AtomicClaimNullOutcome", "_AtomicClaimObservedOutcome")
        if any(name in definitions for name in new_names):
            raise ValueError
        for branch, name, title, basis in (
            (stated, new_names[0], "AtomicClaimNullOutcome", ["stated_preference", "repeated_pattern"]),
            (observed, new_names[1], "AtomicClaimObservedOutcome", ["observed_response", "repeated_pattern"]),
        ):
            factored = deepcopy(branch)
            factored["title"] = title
            factored["properties"]["evidence_basis"].pop("const")
            factored["properties"]["evidence_basis"]["enum"] = basis
            definitions[name] = factored
        for name in old_names:
            del definitions[name]
        atomic["anyOf"] = [{"$ref": "#/$defs/" + name} for name in new_names]
    except (KeyError, TypeError, ValueError):
        raise ValueError("incompatible_reflection_atomic_schema") from None
    return result


def reflection_wire_schema() -> dict:
    return _factor_atomic_outcomes(_project_wire_keywords(reflection_response_schema()))


def reflection_request_format(version: str = "json_schema") -> dict:
    if version not in {"json_schema", "json_schema_v1", "json_schema_v2"}:
        raise ValueError("invalid_reflection_return_format")
    schema = (_project_wire_keywords(reflection_response_schema())
              if version == "json_schema_v1" else reflection_wire_schema())
    return {"type": "json_schema", "json_schema": {
        "name": "aria_reflection_v1", "strict": True, "schema": schema,
    }}


def reflection_body_system_prompt() -> str:
    from ..memory.diagnosis import reflection_system_prompt
    return (reflection_system_prompt(True)
            + "\n\n完整本地 JSON Schema（正文只返回完整 JSON，不使用工具、不省略 nullable 字段）：\n"
            + json.dumps(reflection_response_schema(), ensure_ascii=False, sort_keys=True))


def reflection_correction_issues(error: Exception, raw: str) -> list[dict]:
    """Only syntactic/field-shape errors qualify, never evidence or refusals."""
    if isinstance(error, ValidationError) and any(
            row["type"] in {"value_error", "assertion_error"}
            for row in error.errors(include_url=False, include_context=False, include_input=False)):
        return []
    diagnostic = reflection_shape_diagnostic(error)
    kind = diagnostic.get("kind")
    if kind == "schema":
        # Cross-field validators can encode domain meaning. Do not label their
        # rejection a format failure or solicit a different substantive answer.
        if any(row["type"] in {"value_error", "schema_constraint"}
               for row in diagnostic["issues"]):
            return []
        return diagnostic["issues"]
    if kind == "json_syntax" and type(raw) is str and raw.lstrip().startswith(("{", "[")):
        return [{"path": [], "type": "json_syntax"}]
    if kind in {"duplicate_key", "nonfinite_number"}:
        return [{"path": [], "type": kind}]
    return []


def reflection_correction_message(issues: list[dict]) -> dict:
    """No generated content, exception message or arbitrary field names."""
    if type(issues) is not list or not 1 <= len(issues) <= 20:
        raise ValueError("invalid_reflection_correction")
    for row in issues:
        if (type(row) is not dict or set(row) != {"path", "type"}
                or row["type"] not in _DIAGNOSTIC_TYPES | {"json_syntax", "duplicate_key", "nonfinite_number"}
                or type(row["path"]) is not list or len(row["path"]) > 12
                or any(not (type(part) is str and part in _DIAGNOSTIC_FIELDS
                            or type(part) is int and 0 <= part <= 63) for part in row["path"])):
            raise ValueError("invalid_reflection_correction")
    return {"role": "user", "content": _CORRECTION_PREFIX + json.dumps(issues, ensure_ascii=False, sort_keys=True)}


def reflection_product_step(messages) -> str:
    """Identify the bounded product step from a validated request, not a flag."""
    if (type(messages) is not list or len(messages) not in {2, 3}
            or messages[0] != {"role": "system", "content": reflection_body_system_prompt()}
            or type(messages[1]) is not dict or set(messages[1]) != {"role", "content"}
            or messages[1]["role"] != "user" or type(messages[1]["content"]) is not str):
        raise ValueError("reflection_json_requires_evidence_review_contract")
    if len(messages) == 2:
        return "initial"
    last = messages[2]
    if (type(last) is not dict or set(last) != {"role", "content"}
            or last["role"] != "user" or type(last["content"]) is not str
            or not last["content"].startswith(_CORRECTION_PREFIX)):
        raise ValueError("invalid_reflection_correction")
    try:
        issues = json.loads(last["content"][len(_CORRECTION_PREFIX):])
        if reflection_correction_message(issues) != last:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise ValueError("invalid_reflection_correction") from None
    return "format_correction"


def reflection_body_observation(data: dict) -> dict:
    """Content-free, receipt-bound permission for a single format correction."""
    from ..memory.diagnosis import _unwrap_review_response
    result = {"channel_valid": False, "correction_issues": []}
    choices = data.get("choices") if type(data) is dict else None
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        return result
    choice = choices[0]
    message = choice.get("message")
    if (type(message) is not dict or choice.get("finish_reason") != "stop"
            or message.get("tool_calls") not in (None, []) or message.get("function_call") is not None
            or message.get("refusal") not in (None, "")
            or message.get("reasoning") not in (None, "")
            or message.get("reasoning_details") not in (None, "", [])
            or type(message.get("content")) is not str or not message["content"].strip()):
        return result
    usage = data.get("usage")
    details = usage.get("completion_tokens_details") if type(usage) is dict else None
    if type(details) is dict and details.get("reasoning_tokens") is not None and (
            type(details["reasoning_tokens"]) is not int or details["reasoning_tokens"] != 0):
        return result
    result["channel_valid"] = True
    body = message["content"]
    try:
        body, _ = _unwrap_review_response(body)
        parse_reflection_json(body)
    except (ValueError, TypeError, RecursionError) as error:
        result["correction_issues"] = reflection_correction_issues(error, body)
    return result


def validate_reflection_request(client, messages) -> None:
    from ..._openrouter import DEFAULT_REFLECTION_MODEL
    from ..memory.diagnosis import reflection_system_prompt

    if (client.role != "reflection" or not client.bounded_transport
            or client.model != DEFAULT_REFLECTION_MODEL
            or client.max_retries != 0 or client.fallback_models):
        raise ValueError("unsupported_reflection_json_route")
    if client.reflection_return_format == "body_json":
        reflection_product_step(messages)
        return
    if (type(messages) is not list or len(messages) != 2
            or messages[0] != {"role": "system", "content": reflection_system_prompt(True)}
            or type(messages[1]) is not dict or set(messages[1]) != {"role", "content"}
            or messages[1]["role"] != "user" or type(messages[1]["content"]) is not str):
        raise ValueError("reflection_json_requires_evidence_review_contract")


def parse_reflection_json(raw: str) -> str:
    """Validate the full original shape and return unchanged JSON content."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_reflection_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite_reflection_number")

    if type(raw) is not str:
        raise ValueError("reflection_text_required")
    parsed = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    validate_reflection_response_shape(parsed)
    return raw.strip()
