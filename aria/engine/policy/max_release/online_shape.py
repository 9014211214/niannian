"""Opt-in representation projection for one exact v6 empty-response shape.

This module never changes semantic labels, source references, safety or consent.
The current v7 stream can also project complete object members when only
the final outer delimiter is missing. Original public bytes and the native
finish reason are retained separately from the parser's view.
"""
from __future__ import annotations

from copy import copy, deepcopy
from hashlib import sha256
import json

from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.online_decision import online_decision_tool_spec, ONLINE_DECISION_VERSION_V7
from aria.engine.policy.online_json import _unique_object, _validate, parse_json_decision, parse_v7_payload
from aria.engine.policy.semantic_delivery import VERSION as DECISION_VERSION, valid_segments

VERSION = "online_empty_response_segments_v1"
ONLINE_STREAM_VERSION = "online_sse_application_v1"
OUTER_CLOSE = "complete_members_missing_outer_close"
PADDED_OBJECT = "complete_object_with_terminal_padding"
MAX_BYTES = 128 * 1024


def _bytes(value):
    if type(value) is not str:
        raise ValueError("invalid_online_projection_text")
    try:
        raw = value.encode("utf-8")
    except UnicodeError:
        raise ValueError("invalid_online_projection_unicode") from None
    if len(raw) > MAX_BYTES:
        raise ValueError("online_projection_size_exceeded")
    return raw


def online_content_projection(raw, version, *, require_preference_control=False,
                              require_task_projection=False, padded_length=False):
    """Return a strict canonical view; no source data is modified in place.

    Valid v7 fields pass unchanged. For an explicitly padded length terminal,
    require the whole strict schema, with at most a missing outer delimiter.
    The v6 empty-response projection is unchanged.
    Source/receipt authorization remains the downstream product's responsibility.
    """
    if (version not in (DECISION_VERSION, ONLINE_DECISION_VERSION_V7) or type(require_preference_control) is not bool
            or type(require_task_projection) is not bool):
        raise ValueError("unsupported_online_projection_contract")
    raw_bytes = _bytes(raw)
    if padded_length and version != ONLINE_DECISION_VERSION_V7:
        raise ValueError("invalid_online_outer_close")
    if version == ONLINE_DECISION_VERSION_V7:
        canonical = raw
        if padded_length:
            # Accept one structural byte only. A partial string/array, dangling
            # comma, duplicate field or omitted required member cannot qualify.
            if not raw.endswith((" ", "\t", "\r", "\n")) or not raw.rstrip(" \t\r\n").endswith(("]", "}")):
                raise ValueError("invalid_online_outer_close")
            try:
                fields = json.loads(raw, object_pairs_hook=_unique_object)
            except json.JSONDecodeError:
                canonical = raw + "}"
                fields = json.loads(canonical, object_pairs_hook=_unique_object)
            _validate(fields, online_decision_tool_spec(version, require_task_projection=require_task_projection)
                      ["function"]["parameters"])
        value, dropped = parse_v7_payload(canonical, require_task_projection=require_task_projection)
        if padded_length and dropped:
            raise ValueError("invalid_online_outer_close")
        body = value["response"]
        mode = "segments_only"
        if padded_length:
            mode = PADDED_OBJECT if canonical == raw else OUTER_CLOSE
        canonical_bytes = _bytes(canonical) if canonical != raw else raw_bytes
        return {"canonical": canonical, "metadata": {
            "version": VERSION, "status": "validated", "decision_version": version,
            "mode": mode, "raw_public_sha256": sha256(raw_bytes).hexdigest(),
            "canonical_sha256": sha256(canonical_bytes).hexdigest(),
            "segments_body_sha256": sha256(body.encode("utf-8")).hexdigest(),
            "raw_public_bytes": len(raw_bytes), "canonical_bytes": len(canonical_bytes),
            "segment_count": len(value["response_segments"]), "body_characters": len(body),
            "raw_response_contract": mode if padded_length else "not_requested",
            "raw_segments_match_response": False,
            "derived_fields": ["response"], "dropped_fields": dropped,
            "derivation": "ordered_exact_concat_no_separator_no_strip",
            "representation_only": True, "semantic_verified": False,
            "semantic_content_repaired": False, "source_or_identifier_repaired": False,
            "safety_or_permission_repaired": False,
        }}
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
        _validate(value, online_decision_tool_spec(version,
            require_preference_control=require_preference_control,
            require_task_projection=require_task_projection)["function"]["parameters"])
        segments = value["response_segments"]
        # Reuse the original kind/action/source/state-ref cross-field contract.
        if not valid_segments(segments):
            raise ValueError("invalid_segments")
        body = "".join(row["text"] for row in segments)
        if not body.strip():
            raise ValueError("empty_segments_body")
        derived = value["response"] == ""
        if not derived and value["response"] != body:
            raise ValueError("conflicting_response")
        if derived:
            canonical_value = deepcopy(value)
            canonical_value["response"] = body
            canonical = json.dumps(canonical_value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
        else:
            canonical = raw
        canonical_bytes = _bytes(canonical)
        parse_json_decision(canonical, version,
            require_preference_control=require_preference_control,
            require_task_projection=require_task_projection)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        raise ValueError("invalid_online_segments_projection") from None
    return {"canonical": canonical, "metadata": {
        "version": VERSION, "status": "validated", "decision_version": DECISION_VERSION,
        "mode": "empty_response_derived" if derived else "identity",
        "raw_public_sha256": sha256(raw_bytes).hexdigest(),
        "canonical_sha256": sha256(canonical_bytes).hexdigest(),
        "segments_body_sha256": sha256(body.encode("utf-8")).hexdigest(),
        "raw_public_bytes": len(raw_bytes), "canonical_bytes": len(canonical_bytes),
        "segment_count": len(segments), "body_characters": len(body),
        "raw_response_contract": "fail_empty_response" if derived else "pass",
        "raw_segments_match_response": not derived,
        "derived_fields": ["response"] if derived else [],
        "derivation": "ordered_exact_concat_no_separator_no_strip",
        "representation_only": True, "semantic_verified": False,
        "semantic_content_repaired": False, "source_or_identifier_repaired": False,
        "safety_or_permission_repaired": False,
    }}


def online_envelope_projection(data, version, *, allow_padded_length=False, **requirements):
    """Keep envelope/usage/identity intact; content is the only possible change."""
    try:
        choices = data["choices"]
        if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        padded_length = allow_padded_length and choice.get("finish_reason") == "length"
        if (type(message) is not dict or (not padded_length and choice.get("finish_reason") != "stop")
                or message.get("tool_calls") not in (None, [])
                or message.get("function_call") is not None
                or message.get("refusal") not in (None, "")):
            raise ValueError
        result = online_content_projection(message["content"], version, padded_length=padded_length, **requirements)
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ValueError("invalid_online_projection_envelope") from None
    view = deepcopy(data)
    view["choices"][0]["message"]["content"] = result["canonical"]
    return view, result["metadata"]


def validate_online_projection_client(client):
    from aria.engine.policy.online_structured import validate_release_online_client
    if (type(client) is not SegmentsProjectedOnlineClient
            or getattr(client, "online_response_projection_protocol", None) != VERSION
            or client.role != "online_fast"
            or client.online_decision_version not in (DECISION_VERSION, ONLINE_DECISION_VERSION_V7)
            or client.online_return_format != "json_schema_v1"):
        raise ValueError("invalid_online_projection_client")
    validate_release_online_client(client)
    v7 = client.online_decision_version == ONLINE_DECISION_VERSION_V7
    return {"version": VERSION, "role": "online_fast", "decision_version": client.online_decision_version,
        "wire_changed": False, "representation_only": True, "semantic_verified": False,
        "only_empty_response": not v7, "conflicting_nonempty_response_rejected": not v7,
        "raw_first_output_required": True, "downstream_original_guards_required": True,
        "maximum_public_and_canonical_bytes": MAX_BYTES}


def validate_online_sse_client(client):
    """Bind one explicit release transport; the JSON product contract stays exact."""
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL, validate_release_online_client
    validate_online_projection_client(client)
    route = validate_release_online_client(client)
    if (getattr(client, "online_stream_protocol", None) != ONLINE_STREAM_VERSION
            or client.model != DEEPSEEK_MODEL or route["provider"] != "together"
            or client.online_decision_version != ONLINE_DECISION_VERSION_V7
            or client.timeout != 90 or type(client.max_retries) is not int or client.max_retries != 0
            or client.fallback_models != [] or client.max_completion_tokens != 4096
            or client.bounded_transport is not True):
        raise ValueError("invalid_online_sse_client")
    return _online_sse_contract()


def _online_sse_contract():
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL
    from .reflection_sse import VERSION as READER_VERSION
    return {"schema": ONLINE_STREAM_VERSION, "reader_protocol": READER_VERSION,
        "receipt_mode": ONLINE_STREAM_VERSION,
        "role": "online_fast", "model": DEEPSEEK_MODEL, "provider": "together",
        "stream": True, "physical_timeout_seconds": 90,
        "provider_schema_applied": True, "response_format_preserved": True,
        "terminal_receipt_finishes": ["stop", "length"], "required_delivery_finish": "stop",
        "length_projection": OUTER_CLOSE,
        "required_terminal_usage": True, "required_done": True,
        "partial_submission": False, "identity_and_fee_validation_required": True,
        "response_envelope_source": "validated_sse_public_content_assembly",
        "http_message_completion_required": False, "current_chunk_termination_required": True}


def online_receipt_body_complete(row, public_content=None):
    """Prove the current SSE receipt terminal; never grant delivery or known fees.

    A length/empty receipt can be accounted for and rejected by the original
    product validator. Its HTTP completeness flag is never rewritten.
    """
    from .reflection_sse import VERSION as READER_VERSION, validate_stream_body_observation
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL
    from aria.engine.policy.response_observability import safe_generation_id
    try:
        expected_contract = _online_sse_contract()
        if 'length_projection' not in row['online_stream_contract'] and 'online_length_projection' not in row:
            del expected_contract['length_projection']  # Historical receipts remain verifiable.
        if (type(row) is not dict or row.get('role') != 'online_fast'
                or row.get('model') != DEEPSEEK_MODEL or row.get('expected_provider') != 'Together'
                or json.dumps(row.get('online_stream_contract'), sort_keys=True, allow_nan=False)
                    != json.dumps(expected_contract, sort_keys=True, allow_nan=False)):
            return False
        body = validate_stream_body_observation(row['body_read'])
        stream, shape = row['stream_read'], row['online_receipt_shape']
        if (body['application_complete'] is not True or body['exception_kind'] is not None
                or body['capture_status'] != 'complete' or type(stream) is not dict or type(shape) is not dict
                or stream.get('schema') != READER_VERSION or stream.get('receipt_mode') != ONLINE_STREAM_VERSION
                or any(stream.get(k) is not True for k in ('complete', 'application_complete',
                    'finish_observed', 'usage_observed', 'done_observed', 'closed_after_done'))
                or any(stream.get(k) is not False for k in ('partial_output_submitted',
                    'usage_is_verified_fee', 'deadline_extended'))
                or any(type(stream.get(k)) is not int or stream[k] != 1 for k in ('usage_events', 'done_events'))
                or any(type(stream.get(k)) is not bool or stream[k] != body[k] for k in
                    ('http_message_complete', 'http_eof_observed', 'current_chunk_terminated'))
                or type(stream.get('stream_bytes')) is not int or stream['stream_bytes'] != body['bytes_received']
                or stream.get('stream_sha256') != body['body_sha256']
                or type(stream.get('content_characters')) is not int or stream['content_characters'] < 0
                or safe_generation_id(stream.get('generation_id')) is None
                or stream['generation_id'] != row.get('observation', {}).get('generation_id')
                or shape.get('decision_version') != ONLINE_DECISION_VERSION_V7
                or stream.get('finish_reason') not in ('stop', 'length')
                or stream['finish_reason'] != shape.get('finish_reason')
                or stream.get('public_content_sha256') != shape.get('public_content_sha256')):
            return False
        for name in ('stream_sha256', 'assembled_sha256', 'public_content_sha256'):
            value = stream.get(name)
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                return False
        if public_content is not None and (type(public_content) is not str
                or stream['public_content_sha256'] != sha256(public_content.encode('utf-8')).hexdigest()
                or stream['content_characters'] != len(public_content)):
            return False
        return True
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return False


def enable_online_sse(client):
    """Enable the shared SSE reader on a fresh, already projected release client."""
    validate_online_projection_client(client)
    if (getattr(client, "online_stream_protocol", None) is not None
            or getattr(client, "model_request_control", None) is not None
            or any(getattr(client, key, 0) != 0 for key in
                   ("call_count", "network_request_count", "generation_decision_count"))):
        raise ValueError("fresh_online_sse_client_required")
    result = copy(client)
    result.online_stream_protocol = ONLINE_STREAM_VERSION
    validate_online_sse_client(result)
    return result


class SegmentsProjectedOnlineClient(OpenRouterLLMClient):
    def _request_for_model(self, payload, *, model, require_preference_control=False,
                           require_task_projection=False):
        if getattr(self, "online_stream_protocol", None) is not None:
            validate_online_sse_client(self)
            from .sse_adapter import ReflectionSSEGuardMixin
            if (not isinstance(getattr(self, "model_request_control", None), ReflectionSSEGuardMixin)
                    or not hasattr(self.model_request_control, "policy")):
                raise ValueError("online_sse_reserved_guard_required")
            if (type(payload) is not dict or model != self.model or payload.get("stream") is not False
                    or type(payload.get("response_format")) is not dict
                    or payload["response_format"].get("type") != "json_schema"):
                raise ValueError("online_sse_builder_changed")
            # No prompt/schema transform: only transport framing changes. The
            # existing guard checks every actual wire field before reservation.
            payload = deepcopy(payload)
            payload["stream"] = True
        return super()._request_for_model(payload, model=model,
            require_preference_control=require_preference_control,
            require_task_projection=require_task_projection)

    def _extract_json_decision(self, data, version, *, require_preference_control=False,
                               require_task_projection=False):
        validate_online_projection_client(self)
        # The reserved guard has already persisted the unmodified public first
        # response. Its identity, cost, reasoning, channel and truncation checks
        # remain on that original envelope; this is solely the parser's view.
        try:
            view, metadata = online_envelope_projection(data, version,
                allow_padded_length=(getattr(self, "online_stream_protocol", None) == ONLINE_STREAM_VERSION
                                     and "online_length_projection" in data),
                require_preference_control=require_preference_control,
                require_task_projection=require_task_projection)
            if metadata["mode"] in (OUTER_CLOSE, PADDED_OBJECT) and data["online_length_projection"] != metadata:
                raise ValueError("online_length_projection_binding_changed")
        except ValueError:
            self._update_diagnostic(online_response_projection={"version": VERSION,
                "status": "rejected", "representation_only": True, "semantic_verified": False})
            # Retain the original strict diagnostic for bad raw JSON. If raw
            # JSON passes but segments conflict, still fail instead of return it.
            super()._extract_json_decision(data, version,
                require_preference_control=require_preference_control,
                require_task_projection=require_task_projection)
            raise ValueError("invalid_online_segments_projection") from None
        result = super()._extract_json_decision(view, version,
            validated_padded_length=metadata["mode"] in (OUTER_CLOSE, PADDED_OBJECT),
            require_preference_control=require_preference_control,
            require_task_projection=require_task_projection)
        if sha256(result.encode("utf-8")).hexdigest() != metadata["canonical_sha256"]:
            raise ValueError("online_projection_canonical_mismatch")
        self._update_diagnostic(online_response_projection=deepcopy(metadata))
        return result


def make_segments_projected_online_client(client):
    """Explicit fresh online-only opt-in; original client/factories untouched."""
    from aria.engine.policy.online_structured import validate_release_online_client
    validate_release_online_client(client)
    if (type(client) is not OpenRouterLLMClient or client.role != "online_fast"
            or getattr(client, "model_request_control", None) is not None
            or any(getattr(client, key, 0) != 0 for key in
                   ("call_count", "network_request_count", "generation_decision_count"))):
        raise ValueError("fresh_online_projection_client_required")
    result = copy(client)
    result.__class__ = SegmentsProjectedOnlineClient
    result.online_response_projection_protocol = VERSION
    validate_online_projection_client(result)
    return result


def completion_projection_metadata(client, result):
    """Validate the logical recorder's explicit derived-output label and hashes."""
    validate_online_projection_client(client)
    metadata = deepcopy(getattr(client, "last_diagnostic", {}).get("online_response_projection"))
    if client.online_decision_version == ONLINE_DECISION_VERSION_V7:
        mode = metadata.get("mode") if isinstance(metadata, dict) else None
        close_outer = mode == OUTER_CLOSE
        raw = result[:-1] if close_outer and result.endswith("}") else result
        expected = online_content_projection(raw, ONLINE_DECISION_VERSION_V7,
            padded_length=mode in (OUTER_CLOSE, PADDED_OBJECT))["metadata"]
        if metadata != expected:
            raise ValueError("online_projection_completion_receipt_missing_or_changed")
        return metadata
    if (type(metadata) is not dict or metadata.get("version") != VERSION
            or metadata.get("status") != "validated"
            or metadata.get("canonical_sha256") != sha256(_bytes(result)).hexdigest()
            or metadata.get("mode") not in ("identity", "empty_response_derived")
            or any(type(metadata.get(k)) is not str or len(metadata[k]) != 64
                   or any(c not in "0123456789abcdef" for c in metadata[k])
                   for k in ("raw_public_sha256", "canonical_sha256", "segments_body_sha256"))):
        raise ValueError("online_projection_completion_receipt_missing_or_changed")
    derived = metadata["mode"] == "empty_response_derived"
    if (metadata.get("representation_only") is not True
            or any(metadata.get(k) is not False for k in ("semantic_verified", "semantic_content_repaired",
                "source_or_identifier_repaired", "safety_or_permission_repaired"))
            or metadata.get("derived_fields") != (["response"] if derived else [])
            or metadata.get("raw_response_contract") != ("fail_empty_response" if derived else "pass")
            or metadata.get("raw_segments_match_response") is not (not derived)
            or not derived and metadata["raw_public_sha256"] != metadata["canonical_sha256"]):
        raise ValueError("online_projection_completion_receipt_missing_or_changed")
    return metadata
