"""Bounded public transport evidence, never raw provider messages or billing guesses.

Only documented envelope metadata is inspected. A valid ``usage.cost: 0`` is an
explicit observation; HTTP status, absent usage, and account balance are not.
Generation IDs are useful for a separately authorized read-only invoice lookup.
They do not authorize retries or change historical accounting.
"""
from __future__ import annotations

import http.client
import json
import math
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any
import urllib.error


VERSION = "1.0"
INTERRUPTED_VERSION = "1.1"
LEGACY_HTTP_ERROR_VERSION = "1.2"
HTTP_ERROR_VERSION = "1.3"
RESPONSE_ERROR_VERSION = "1.4"
MAX_ERROR_BODY_BYTES = 16 * 1024
MAX_PROVIDER_RAW_BYTES = 4096
MAX_RETRY_AFTER_SECONDS = 86400
_MAX_HTTP_TIMESTAMP = 253402300799
_GENERATION_ID = re.compile(r"gen-(?:[A-Za-z0-9]{8,80}|[0-9]{9,13}-[A-Za-z0-9]{8,64})\Z")
_REQUEST_ID = re.compile(
    r"(?:[0-9a-fA-F]{16,64}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|req[-_][A-Za-z0-9]{8,64}|"
    r"req-[0-9]{9,13}-[A-Za-z0-9]{8,64})\Z"
)
_BODY_STATUSES = {
    "parsed", "empty", "oversized", "non_bytes", "read_failed", "decode_failed",
    "parse_failed", "non_object",
}
_ERROR_CATEGORIES = {
    "authentication", "payment_required", "permission_denied", "request_invalid",
    "rate_limited", "server_http_error", "client_http_error", "other_http_error",
}
_ERROR_TYPES = {
    "rate_limit_exceeded", "rate_limit_error", "insufficient_quota",
    "insufficient_credits", "authentication_error", "invalid_request_error",
    "permission_error", "overloaded_error", "api_error",
}
_PROVIDER_NAMES = {"Alibaba", "Baidu", "DeepSeek", "Anthropic"}
# Exact Alibaba codes documented at https://help.aliyun.com/en/model-studio/error-code.
# These are observations only: AllocationQuota has multiple causes, and
# insufficient_quota must not be interpreted as an Alibaba code for other providers.
_ALIBABA_NATIVE_CODES = {
    "Throttling.RateQuota", "LimitRequests", "limit_requests",
    "Throttling.BurstRate", "limit_burst_rate", "Throttling.AllocationQuota", "insufficient_quota",
}
_NATIVE_SOURCES = {
    "metadata.provider_code", "metadata.raw.code", "metadata.raw.type",
    "metadata.raw.error.code", "metadata.raw.error.type",
}
_NATIVE_FIELDS = {"provider_native_code_status", "provider_raw_status",
                  "provider_native_code", "provider_native_code_sources"}
_NATIVE_STATUSES = {"not_inspected", "absent", "invalid", "recognized", "unrecognized", "conflicting"}
_RAW_STATUSES = {"not_inspected", "absent", "invalid", "parsed"}
_IDENTITY_STATUSES = {"missing", "null", "wrong_type", "unrecognized", "recognized"}
_ERROR_FIELD_STATUSES = {"absent", "null", "wrong_type", "object"}
_PROVIDER_ERROR_FIELDS = {
    "provider_error_code", "provider_error_type", "provider_error_type_source", "provider_code",
    "provider_metadata_present", "provider_reported_name", "provider_name_status",
}
# OpenRouter's canonical metadata.error_type is separate from a provider's
# native error.type. Unknown native values must not mask this finite category.
# https://openrouter.ai/docs/api_reference/errors-and-debugging (2026-10-04)
_ROUTER_ERROR_TYPES = {
    "context_length_exceeded", "max_tokens_exceeded", "token_limit_exceeded", "string_too_long",
    "authentication", "permission_denied", "payment_required", "rate_limit_exceeded",
    "provider_overloaded", "provider_unavailable", "invalid_request", "invalid_prompt", "not_found",
    "precondition_failed", "payload_too_large", "unprocessable", "content_policy_violation", "refusal",
    "invalid_image", "image_too_large", "image_too_small", "unsupported_image_format", "image_not_found",
    "image_download_failed", "server", "timeout", "unmapped",
}
_ROUTER_ERROR_FIELDS = {"router_error_type", "router_error_type_status", "router_error_type_source"}
_READ_KINDS = {
    "incomplete_read", "remote_disconnected", "timeout", "url_timeout", "url_error",
    "connection_error", "http_protocol_error", "other_read_error",
}
_REQUEST_SOURCES = {"header_x_request_id", "header_request_id", "header_x_openrouter_request_id", "body_request_id"}
_FIELDS = {
    "version", "envelope_kind", "body_status", "error_category", "provider_error_code",
    "provider_error_type", "generation_id", "request_id", "request_id_source",
    "request_id_status", "body_read_error_kind", "body_close_failed", "usage_source",
    "provider_error_type_source", "provider_code", "rate_limit_limit",
    "rate_limit_remaining", "rate_limit_reset",
    "provider_metadata_present", "provider_reported_name", "provider_name_status",
}
_CANDIDATE_SOURCES = {"header_x_generation_id", "header_x_openrouter_generation_id"}
_INTERRUPTED_FIELDS = {
    "generation_id_candidate", "generation_id_candidate_source", "generation_id_candidate_status",
}
_RETRY_AFTER_FIELDS = {
    "retry_after_status", "retry_after_seconds", "retry_after_source",
    "retry_after_reference", "retry_after_reference_epoch_seconds", "retry_after_target_epoch_seconds",
}
_HTTP_DATE = re.compile(
    r"(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT|"
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"(?: [1-9]|[0-9]{2}) [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4})\Z"
)


def safe_generation_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _GENERATION_ID.fullmatch(value) else None


def _safe_request_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


def safe_usage(data: Any) -> dict[str, float | int]:
    """Only numeric fields from the top-level provider usage envelope.

    Nested provider metadata may describe upstream charges, not the amount
    charged by this endpoint, and is deliberately not promoted to an invoice.
    """
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    clean = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        value = usage.get(key)
        if _nonnegative_number(value):
            clean[key] = value
    details = usage.get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if _nonnegative_number(value):
        clean["reasoning_tokens"] = value
    return clean


def _nonnegative_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _category(status: int) -> str:
    if type(status) is not int:
        return "other_http_error"
    if status == 401:
        return "authentication"
    if status == 402:
        return "payment_required"
    if status == 403:
        return "permission_denied"
    if status in {400, 404, 405, 413, 415, 422}:
        return "request_invalid"
    if status == 429:
        return "rate_limited"
    if 500 <= status <= 599:
        return "server_http_error"
    if 400 <= status <= 499:
        return "client_http_error"
    return "other_http_error"


def _header_id_pairs(headers: Any, keys: tuple[str, ...], *, reject_getter_failure: bool = False) -> list[tuple[str, Any]]:
    """Fixed header names only; repeated fields are ambiguous, even if equal.

    Email/HTTPMessage ``get`` silently picks the first repeated field. Prefer
    ``get_all`` so a conflicting second value cannot become a trusted trace ID.
    A broken header implementation never hides the original transport failure.
    """
    pairs = []
    for key in keys:
        try:
            getter = getattr(headers, "get_all", None)
            if callable(getter):
                values = getter(key)
                if values is None or values == []:
                    continue
                if not isinstance(values, (list, tuple)) or len(values) != 1:
                    pairs.append(("header_" + key.lower().replace("-", "_"), None))
                    continue
                value = values[0]
            else:
                value = headers.get(key) if headers is not None else None
        except Exception:
            if reject_getter_failure:
                pairs.append(("header_" + key.lower().replace("-", "_"), None))
            value = None
        if value is not None:
            pairs.append(("header_" + key.lower().replace("-", "_"), value))
    return pairs


def _headers_request_id(headers: Any, *, reject_getter_failure: bool = False) -> list[tuple[str, Any]]:
    return _header_id_pairs(headers, ("X-Request-ID", "Request-ID", "X-OpenRouter-Request-ID"),
                            reject_getter_failure=reject_getter_failure)


def _generation_header_candidates(headers: Any) -> dict:
    result = {}
    pairs = _header_id_pairs(headers, ("X-Generation-ID", "X-OpenRouter-Generation-ID"),
                            reject_getter_failure=True)
    valid = [(source, safe_generation_id(value)) for source, value in pairs]
    if not valid:
        result["generation_id_candidate_status"] = "absent"
    elif any(value is None for _, value in valid):
        result["generation_id_candidate_status"] = "rejected"
    elif len({value for _, value in valid}) != 1:
        result["generation_id_candidate_status"] = "conflicting"
    else:
        result["generation_id_candidate_status"] = "accepted"
        result["generation_id_candidate_source"], result["generation_id_candidate"] = valid[0]
    return result


def _single_header(headers: Any, name: str) -> tuple[str, Any]:
    """Do not pick a value from repeated headers, including equal duplicates."""
    try:
        getter = getattr(headers, "get_all", None)
        if callable(getter):
            values = getter(name)
            if values is None or values == []:
                return "absent", None
            if not isinstance(values, (list, tuple)) or len(values) != 1:
                return "rejected", None
            value = values[0]
        else:
            value = headers.get(name) if headers is not None else None
            if value is None:
                return "absent", None
    except Exception:
        return "rejected", None
    if type(value) is not str or len(value) > 128:
        return "rejected", None
    return "present", value.strip(" \t")


def _http_date_epoch(value: str) -> int | None:
    """Accept complete full-year HTTP dates; ambiguous two-digit years reject."""
    match = _HTTP_DATE.fullmatch(value)
    if match is None:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:  # The asctime HTTP form is implicitly GMT.
            parsed = parsed.replace(tzinfo=timezone.utc)
        weekday = next(part for part in match.groups() if part is not None)[:3]
        if weekday != ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[parsed.weekday()]:
            return None
        epoch = int(parsed.timestamp())
        return epoch if 0 <= epoch <= _MAX_HTTP_TIMESTAMP else None
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _retry_after_observation(headers: Any, *, observed_at: float | int | None) -> dict:
    status, raw = _single_header(headers, "Retry-After")
    if status != "present":
        return {"retry_after_status": status}
    rejected = {"retry_after_status": "rejected"}
    if re.fullmatch(r"[0-9]{1,6}", raw):
        seconds = int(raw)
        return ({"retry_after_status": "accepted", "retry_after_seconds": seconds,
                 "retry_after_source": "delta_seconds"}
                if seconds <= MAX_RETRY_AFTER_SECONDS else rejected)
    target = _http_date_epoch(raw)
    if target is None:
        return rejected
    date_status, raw_date = _single_header(headers, "Date")
    if date_status == "present":
        reference, origin = _http_date_epoch(raw_date), "response_date"
        if reference is None:
            return rejected
    elif date_status == "absent":
        try:
            reference = time.time() if observed_at is None else observed_at
        except Exception:
            return rejected
        origin = "observed_at"
    else:
        return rejected
    if not _nonnegative_number(reference) or reference > _MAX_HTTP_TIMESTAMP:
        return rejected
    seconds = max(0, target - reference)
    if seconds > MAX_RETRY_AFTER_SECONDS:
        return rejected
    return {"retry_after_status": "accepted", "retry_after_seconds": seconds,
            "retry_after_source": "http_date", "retry_after_reference": origin,
            "retry_after_reference_epoch_seconds": reference, "retry_after_target_epoch_seconds": target}


def capture_response_headers(response: Any) -> dict:
    """Capture finite metadata *before* reading; never keep a live header object.

    Header generation IDs are only lookup candidates, not body-envelope IDs or
    invoices. They deliberately use a distinct field that billing does not
    consume automatically. Missing IDs cannot be recovered from partial text.
    """
    try:
        headers = getattr(response, "headers", None)
    except Exception:
        headers = None
    result = {"envelope_kind": "success"}
    _metadata(None, headers, result, reject_header_failure=True)
    result.pop("envelope_kind")
    result.update(_generation_header_candidates(headers))
    return result


def observe_interrupted(header_observation: dict, *, phase: str, error: BaseException) -> dict:
    """No partial-response parsing, scoring, token usage, or cost extraction."""
    statuses = {"read": "read_failed", "decode": "decode_failed", "parse": "parse_failed"}
    if phase not in statuses:
        raise ValueError("invalid_interrupted_response_phase")
    result = {**header_observation, "version": INTERRUPTED_VERSION,
              "envelope_kind": "interrupted", "body_status": statuses[phase]}
    if phase == "read":
        result["body_read_error_kind"] = _read_error_kind(error)
    validate_observation(result)
    return result


def _metadata(data: Any, headers: Any, result: dict, *, reject_header_failure: bool = False) -> None:
    pairs = _headers_request_id(headers, reject_getter_failure=reject_header_failure)
    for header, field in (("X-RateLimit-Limit", "rate_limit_limit"),
                          ("X-RateLimit-Remaining", "rate_limit_remaining"),
                          ("X-RateLimit-Reset", "rate_limit_reset")):
        try:
            raw = headers.get(header) if headers is not None else None
            number = float(raw) if isinstance(raw, str) and len(raw) <= 30 else None
        except Exception:
            # A nonstandard header object must not hide the original failure.
            number = None
        if _nonnegative_number(number) and number <= 10 ** 18:
            result[field] = number
    if isinstance(data, dict):
        generation_id = safe_generation_id(data.get("id"))
        if generation_id:
            result["generation_id"] = generation_id
        if data.get("request_id") is not None:
            pairs.append(("body_request_id", data["request_id"]))
        if safe_usage(data):
            result["usage_source"] = "top_level_usage"
        error = data.get("error")
        if result["envelope_kind"] in {"http_error", "response_error"} and isinstance(error, dict):
            code = error.get("code")
            if type(code) is int and 0 <= code <= 999999:
                result["provider_error_code"] = code
            error_type = error.get("type")
            type_source = "error_type"
            metadata = error.get("metadata")
            if isinstance(metadata, dict):
                result["provider_metadata_present"] = True
                provider_name = metadata.get("provider_name")
                if provider_name is None:
                    result["provider_name_status"] = "absent"
                elif isinstance(provider_name, str) and provider_name in _PROVIDER_NAMES:
                    result["provider_name_status"] = "recognized"
                    result["provider_reported_name"] = provider_name
                else:
                    result["provider_name_status"] = "unrecognized"
                provider_code = metadata.get("provider_code")
                if type(provider_code) is int and 0 <= provider_code <= 999999:
                    result["provider_code"] = provider_code
                if error_type is None:
                    error_type, type_source = metadata.get("error_type"), "error_metadata_type"
            if isinstance(error_type, str) and error_type in _ERROR_TYPES:
                result["provider_error_type"] = error_type
                result["provider_error_type_source"] = type_source
    valid = [(source, _safe_request_id(value)) for source, value in pairs]
    if not valid:
        result["request_id_status"] = "absent"
    elif any(value is None for _, value in valid):
        result["request_id_status"] = "rejected"
    elif len({value for _, value in valid}) != 1:
        result["request_id_status"] = "conflicting"
    else:
        result["request_id_status"] = "accepted"
        result["request_id_source"], result["request_id"] = valid[0]


def observe_success(data: Any, headers: Any = None, *, header_observation: dict | None = None) -> dict:
    result = {"version": VERSION, "envelope_kind": "success",
              "body_status": "parsed" if isinstance(data, dict) else "non_object"}
    _metadata(data, headers, result)
    if header_observation is not None:
        # Prefer the pre-read snapshot to a header object that may no longer be
        # readable. Header generation candidates are not promoted on success.
        for key in ("rate_limit_limit", "rate_limit_remaining", "rate_limit_reset"):
            if key in header_observation:
                result[key] = header_observation[key]
        header_status = header_observation["request_id_status"]
        body_status = result["request_id_status"]
        if "rejected" in (header_status, body_status):
            combined = "rejected"
        elif "conflicting" in (header_status, body_status):
            combined = "conflicting"
        elif header_status == "accepted" and body_status == "accepted":
            combined = "accepted" if header_observation["request_id"] == result["request_id"] else "conflicting"
        else:
            combined = header_status if header_status != "absent" else body_status
        result["request_id_status"] = combined
        if combined != "accepted":
            result.pop("request_id", None)
            result.pop("request_id_source", None)
        elif header_status == "accepted":
            result["request_id"] = header_observation["request_id"]
            result["request_id_source"] = header_observation["request_id_source"]
    return result


def receipt_identity(data: dict, *, models: list[str], providers: list[str]) -> dict:
    """Finite states only; unrecognized identities may contain private text."""
    def identity(field, allowed):
        if field not in data:
            return "missing"
        value = data[field]
        if value is None:
            return "null"
        if type(value) is not str:
            return "wrong_type"
        return "recognized" if value in allowed else "unrecognized"

    error = ("absent" if "error" not in data else "null" if data["error"] is None
             else "object" if type(data["error"]) is dict else "wrong_type")
    return {"version": "receipt_identity_v1", "model_status": identity("model", models),
            "provider_status": identity("provider", providers), "error_field_status": error}


def validate_receipt_identity(value: Any) -> None:
    if (type(value) is not dict
            or set(value) != {"version", "model_status", "provider_status", "error_field_status"}
            or value["version"] != "receipt_identity_v1"
            or any(type(value[field]) is not str or value[field] not in _IDENTITY_STATUSES
                   for field in ("model_status", "provider_status"))
            or type(value["error_field_status"]) is not str
            or value["error_field_status"] not in _ERROR_FIELD_STATUSES):
        raise ValueError("invalid_receipt_identity")


def observe_response_error(data: dict, observation: dict, *, error_field_status: str) -> dict:
    """Classify a present top-level error independently of HTTP success.

    Preserve already captured IDs and unambiguous usage evidence. Never copy
    raw error messages, provider raw payloads, unknown identities, or choices.
    This evidence does not grant retry or release an unknown-cost reservation.
    """
    if type(data) is not dict or type(data.get("error")) is not dict or error_field_status != "object":
        raise ValueError("invalid_response_error_field_status")
    result = {**observation, "version": RESPONSE_ERROR_VERSION, "envelope_kind": "response_error",
              "error_field_status": error_field_status}
    metadata = {"envelope_kind": "response_error"}
    _metadata(data, None, metadata)
    result.update({key: metadata[key] for key in _PROVIDER_ERROR_FIELDS if key in metadata})
    router_metadata = data["error"].get("metadata")
    status = "absent"
    if type(router_metadata) is dict and "error_type" in router_metadata:
        category = router_metadata["error_type"]
        status = ("null" if category is None else "wrong_type" if type(category) is not str
                  else "recognized" if category in _ROUTER_ERROR_TYPES else "unrecognized")
        if status == "recognized":
            result.update(router_error_type=category, router_error_type_source="error.metadata.error_type")
    result["router_error_type_status"] = status
    validate_observation(result)
    return result


def _read_error_kind(exc: BaseException) -> str:
    if isinstance(exc, http.client.IncompleteRead):
        return "incomplete_read"
    if isinstance(exc, http.client.RemoteDisconnected):
        return "remote_disconnected"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "url_timeout" if isinstance(exc.reason, TimeoutError) else "url_error"
    if isinstance(exc, ConnectionError):
        return "connection_error"
    if isinstance(exc, http.client.HTTPException):
        return "http_protocol_error"
    return "other_read_error"


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _bounded_depth(value: Any, *, max_depth: int = 64) -> bool:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > max_depth:
            return False
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
    return True


def _reject_json_constant(value):
    raise ValueError("invalid_json_constant")


def _native_provider_error(data: Any, result: dict) -> None:
    """Finite native codes, never raw messages, guessed quotas, fees or retries."""
    result.update(provider_raw_status="not_inspected", provider_native_code_status="not_inspected")
    if (result.get("error_category") != "rate_limited"
            or result.get("provider_reported_name") != "Alibaba" or result.get("body_status") != "parsed"):
        return
    metadata = data["error"]["metadata"]
    candidates = []
    direct = metadata.get("provider_code")
    # Preserve the existing numeric provider_code separately. HTTP 429 is not
    # a native string code and cannot conflict with valid raw code evidence.
    if type(direct) is str:
        candidates.append(("metadata.provider_code", direct))
    raw_status = "absent"
    if "raw" in metadata:
        raw_status = "invalid"
        raw = metadata["raw"]
        if type(raw) is str and len(raw) <= MAX_PROVIDER_RAW_BYTES:
            try:
                if len(raw.encode("utf-8")) > MAX_PROVIDER_RAW_BYTES:
                    raise ValueError("provider_raw_too_large")
                parsed = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_json_constant)
                if type(parsed) is not dict or not _bounded_depth(parsed, max_depth=8):
                    raise ValueError("invalid_provider_raw_shape")
                nested = parsed.get("error", {})
                if type(nested) is not dict:
                    raise ValueError("invalid_provider_raw_error")
            except (ValueError, UnicodeError, RecursionError):
                pass
            else:
                raw_status = "parsed"
                for prefix, obj in (("metadata.raw", parsed), ("metadata.raw.error", nested)):
                    for field in ("code", "type"):
                        if field in obj:
                            candidates.append((prefix + "." + field, obj[field]))
    result["provider_raw_status"] = raw_status
    recognized = [(source, code) for source, code in candidates
                  if type(code) is str and code in _ALIBABA_NATIVE_CODES]
    # Do not select one of two disagreeing fields. Unknown values never leave
    # this function, including when another field contains an allowed code.
    if recognized and (len({code for _, code in recognized}) > 1 or len(recognized) != len(candidates)):
        status = "conflicting"
    elif recognized:
        status = "recognized"
        result["provider_native_code"] = recognized[0][1]
        result["provider_native_code_sources"] = sorted(source for source, _ in recognized)
    elif raw_status == "invalid":
        status = "invalid"
    elif candidates:
        status = "unrecognized"
    else:
        status = "absent"
    result["provider_native_code_status"] = status


def observe_http_error(exc: urllib.error.HTTPError, *, observed_at: float | int | None = None,
                       recovery_evidence: dict | None = None,
                       no_content_evidence: dict | None = None) -> tuple[dict, dict]:
    """Consume at most limit+1 bytes, retaining only finite metadata and usage.

    Duplicate keys, incomplete bodies, oversized bodies and invalid envelopes
    are not parsed partially. Read/close failures are recorded without replacing
    the caller's original HTTPError. No provider exception message is persisted.
    """
    result = {"version": HTTP_ERROR_VERSION, "envelope_kind": "http_error",
              "error_category": _category(exc.code), "body_status": "read_failed"}
    try:
        headers = exc.headers
    except Exception:
        headers = None
        result.update(generation_id_candidate_status="rejected", retry_after_status="rejected")
    else:
        # Snapshot these fields before reading/closing. Header IDs are independent
        # lookup candidates; no candidate or waiting hint becomes usage evidence.
        result.update(_generation_header_candidates(headers))
        result.update(_retry_after_observation(headers, observed_at=observed_at))
    data = None
    try:
        try:
            raw = exc.read(MAX_ERROR_BODY_BYTES + 1)
        except Exception as read_exc:
            result["body_read_error_kind"] = _read_error_kind(read_exc)
        else:
            if not isinstance(raw, bytes):
                result["body_status"] = "non_bytes"
            elif len(raw) > MAX_ERROR_BODY_BYTES:
                result["body_status"] = "oversized"
            elif not raw:
                result["body_status"] = "empty"
            else:
                try:
                    decoded = raw.decode("utf-8")
                except UnicodeDecodeError:
                    result["body_status"] = "decode_failed"
                else:
                    try:
                        parsed = json.loads(decoded, object_pairs_hook=_unique_keys)
                    except (ValueError, RecursionError):
                        result["body_status"] = "parse_failed"
                    else:
                        if not _bounded_depth(parsed):
                            result["body_status"] = "parse_failed"
                        elif isinstance(parsed, dict):
                            result["body_status"], data = "parsed", parsed
                        else:
                            result["body_status"] = "non_object"
    finally:
        try:
            exc.close()
        except Exception:
            result["body_close_failed"] = True
    _metadata(data, headers, result)
    _native_provider_error(data, result)
    if (recovery_evidence is not None and type(recovery_evidence) is dict
            and exc.code == 429 and result["body_status"] == "parsed" and not result.get("body_close_failed")
            and type(data) is dict and set(data) <= {"error", "id", "request_id", "usage"}
            and type(data.get("error")) is dict and type(data["error"].get("code")) is int
            and data["error"]["code"] == 429):
        recovery_evidence.update(schema="http429_error_envelope_v1", complete_error_envelope=True, no_model_answer=True)
    finite_error = False
    if type(no_content_evidence) is dict and type(data) is dict:
        try:
            json.dumps(data, allow_nan=False)
        except (ValueError, TypeError, RecursionError):
            pass
        else:
            code = data.get("error", {}).get("code") if type(data.get("error")) is dict else None
            finite_error = code is None or type(code) is int and (code in {400, 408, 429} or 500 <= code <= 599)
    if (type(no_content_evidence) is dict and type(exc.code) is int
            and (exc.code == 429 or 500 <= exc.code <= 599)
            and not result.get("body_close_failed")
            and (result["body_status"] == "empty" or (
                result["body_status"] == "parsed" and type(data) is dict
                and set(data) <= {"error", "id", "request_id", "usage"}
                and type(data.get("error")) is dict and finite_error))):
        no_content_evidence.update(schema="e1br_no_content_v1", kind="http_error",
                                   http_status=exc.code, body_status=result["body_status"])
    return result, safe_usage(data)


def validate_observation(value: Any) -> None:
    """Reject injected/free-form metadata in resumed journal evidence."""
    def invalid():
        raise ValueError("invalid_response_observability")
    if not isinstance(value, dict):
        invalid()
    interrupted = value.get("version") == INTERRUPTED_VERSION and value.get("envelope_kind") == "interrupted"
    response_error = value.get("version") == RESPONSE_ERROR_VERSION and value.get("envelope_kind") == "response_error"
    extended_error = value.get("version") in (LEGACY_HTTP_ERROR_VERSION, HTTP_ERROR_VERSION) and value.get("envelope_kind") == "http_error"
    native_error = extended_error and value["version"] == HTTP_ERROR_VERSION
    fields = _FIELDS | _INTERRUPTED_FIELDS if interrupted or extended_error else set(_FIELDS)
    if extended_error:
        fields |= _RETRY_AFTER_FIELDS
    if native_error:
        fields |= _NATIVE_FIELDS
    if response_error:
        fields |= {"error_field_status"} | _ROUTER_ERROR_FIELDS
    kinds = {"interrupted"} if interrupted else {"response_error"} if response_error else {"success", "http_error"}
    if (not set(value) <= fields
            or value.get("version") != (INTERRUPTED_VERSION if interrupted else value["version"] if extended_error
                                       else RESPONSE_ERROR_VERSION if response_error else VERSION)
            or not isinstance(value.get("envelope_kind"), str)
            or value["envelope_kind"] not in kinds
            or not isinstance(value.get("body_status"), str)
            or value["body_status"] not in _BODY_STATUSES
            or not isinstance(value.get("request_id_status"), str)
            or value["request_id_status"] not in {"absent", "rejected", "conflicting", "accepted"}):
        invalid()
    if value["envelope_kind"] == "http_error":
        if not isinstance(value.get("error_category"), str) or value["error_category"] not in _ERROR_CATEGORIES:
            invalid()
    elif response_error:
        if (value["body_status"] != "parsed" or type(value.get("error_field_status")) is not str
                or value["error_field_status"] != "object"
                or any(key in value for key in ("error_category", "body_close_failed", "body_read_error_kind"))):
            invalid()
        router_status = value.get("router_error_type_status")
        if type(router_status) is not str or router_status not in {"absent", "null", "wrong_type", "unrecognized", "recognized"}:
            invalid()
        recognized = router_status == "recognized"
        if ("router_error_type" in value) is not recognized or ("router_error_type_source" in value) is not recognized:
            invalid()
        if recognized and (type(value["router_error_type"]) is not str
                           or value["router_error_type"] not in _ROUTER_ERROR_TYPES
                           or value["router_error_type_source"] != "error.metadata.error_type"):
            invalid()
        if router_status != "absent" and value.get("provider_metadata_present") is not True:
            invalid()
    elif any(key in value for key in ("error_category", "provider_error_code", "provider_error_type", "provider_code", "provider_error_type_source", "body_close_failed", "provider_metadata_present", "provider_reported_name", "provider_name_status")):
        invalid()
    if not interrupted and value["envelope_kind"] == "success" and "body_read_error_kind" in value:
        invalid()
    if interrupted:
        if (value["body_status"] not in {"read_failed", "decode_failed", "parse_failed"}
                or any(key in value for key in ("usage_source", "generation_id"))
                or value.get("request_id_source") == "body_request_id"):
            invalid()
    if interrupted or extended_error:
        candidate_status = value.get("generation_id_candidate_status")
        if not isinstance(candidate_status, str) or candidate_status not in {"absent", "rejected", "conflicting", "accepted"}:
            invalid()
        candidate_accepted = candidate_status == "accepted"
        if (("generation_id_candidate" in value) is not candidate_accepted
                or ("generation_id_candidate_source" in value) is not candidate_accepted):
            invalid()
        if candidate_accepted and (
            safe_generation_id(value["generation_id_candidate"]) is None
            or not isinstance(value["generation_id_candidate_source"], str)
            or value["generation_id_candidate_source"] not in _CANDIDATE_SOURCES
        ):
            invalid()
    if extended_error:
        status = value.get("retry_after_status")
        if type(status) is not str or status not in {"absent", "rejected", "accepted"}:
            invalid()
        retry_fields = set(value) & _RETRY_AFTER_FIELDS
        expected_fields = {"retry_after_status"}
        if status == "accepted":
            expected_fields |= {"retry_after_seconds", "retry_after_source"}
            seconds, source = value.get("retry_after_seconds"), value.get("retry_after_source")
            if not _nonnegative_number(seconds) or seconds > MAX_RETRY_AFTER_SECONDS:
                invalid()
            if source == "http_date":
                expected_fields |= {"retry_after_reference", "retry_after_reference_epoch_seconds", "retry_after_target_epoch_seconds"}
                reference, target = value.get("retry_after_reference_epoch_seconds"), value.get("retry_after_target_epoch_seconds")
                if (value.get("retry_after_reference") not in ("response_date", "observed_at")
                        or not _nonnegative_number(reference) or reference > _MAX_HTTP_TIMESTAMP
                        or type(target) is not int or not 0 <= target <= _MAX_HTTP_TIMESTAMP
                        or value["retry_after_reference"] == "response_date" and type(reference) is not int
                        or seconds != max(0, target - reference)):
                    invalid()
            elif source != "delta_seconds" or type(seconds) is not int:
                invalid()
        if retry_fields != expected_fields:
            invalid()
    if native_error:
        status, raw_status = value.get("provider_native_code_status"), value.get("provider_raw_status")
        if type(status) is not str or status not in _NATIVE_STATUSES or type(raw_status) is not str or raw_status not in _RAW_STATUSES:
            invalid()
        inspected = (value.get("error_category") == "rate_limited"
                     and value.get("provider_reported_name") == "Alibaba" and value["body_status"] == "parsed")
        if (status == "not_inspected") is inspected or (raw_status == "not_inspected") is inspected:
            invalid()
        recognized = status == "recognized"
        if ("provider_native_code" in value) is not recognized or ("provider_native_code_sources" in value) is not recognized:
            invalid()
        if recognized:
            code, sources = value["provider_native_code"], value["provider_native_code_sources"]
            if (type(code) is not str or code not in _ALIBABA_NATIVE_CODES or type(sources) is not list or not sources
                    or any(type(source) is not str or source not in _NATIVE_SOURCES for source in sources)
                    or sources != sorted(set(sources))
                    or raw_status != "parsed" and sources != ["metadata.provider_code"]):
                invalid()
        if status == "invalid" and raw_status != "invalid":
            invalid()
        if (raw_status == "invalid" and status not in {"invalid", "recognized"}
                or raw_status == "absent" and status not in {"absent", "recognized", "unrecognized"}):
            invalid()
    for key in ("provider_error_code", "provider_code"):
        if key in value and (type(value[key]) is not int or not 0 <= value[key] <= 999999):
            invalid()
    for key in ("rate_limit_limit", "rate_limit_remaining", "rate_limit_reset"):
        if key in value and (not _nonnegative_number(value[key]) or value[key] > 10 ** 18):
            invalid()
    for key, allowed in (("provider_error_type", _ERROR_TYPES), ("body_read_error_kind", _READ_KINDS), ("request_id_source", _REQUEST_SOURCES)):
        if key in value and (not isinstance(value[key], str) or value[key] not in allowed):
            invalid()
    if ("provider_error_type" in value) != ("provider_error_type_source" in value):
        invalid()
    if "provider_error_type_source" in value and value["provider_error_type_source"] not in ("error_type", "error_metadata_type"):
        invalid()
    if "provider_metadata_present" in value:
        if (value["provider_metadata_present"] is not True
                or value.get("provider_name_status") not in ("absent", "recognized", "unrecognized")):
            invalid()
        if ("provider_reported_name" in value) is not (value["provider_name_status"] == "recognized"):
            invalid()
    elif "provider_reported_name" in value or "provider_name_status" in value:
        invalid()
    if "provider_reported_name" in value and (not isinstance(value["provider_reported_name"], str)
                                             or value["provider_reported_name"] not in _PROVIDER_NAMES):
        invalid()
    if ("generation_id" in value and safe_generation_id(value["generation_id"]) is None
            or "request_id" in value and _safe_request_id(value["request_id"]) is None
            or "body_close_failed" in value and value["body_close_failed"] is not True
            or "usage_source" in value and value["usage_source"] != "top_level_usage"):
        invalid()
    accepted = value["request_id_status"] == "accepted"
    if ("request_id" in value) is not accepted or ("request_id_source" in value) is not accepted:
        invalid()
    if ("body_read_error_kind" in value) is not (value["body_status"] == "read_failed"):
        invalid()
    if value["body_status"] != "parsed" and any(key in value for key in ("generation_id", "usage_source", "provider_error_code", "provider_error_type", "provider_code", "provider_metadata_present", "provider_reported_name", "provider_name_status")):
        invalid()


def validate_attempt_observation(attempt: dict) -> None:
    """Bind additive metadata to its enclosing attempt; old rows stay supported."""
    if "response_observability" not in attempt:
        return
    value = attempt["response_observability"]
    validate_observation(value)
    usage = attempt.get("provider_usage")
    status, error_type = attempt.get("http_status"), attempt.get("error_type")
    if not isinstance(usage, dict):
        raise ValueError("response_observability_attempt_mismatch")
    if bool(usage) is not (value.get("usage_source") == "top_level_usage"):
        raise ValueError("response_observability_usage_mismatch")
    if usage and value["body_status"] != "parsed":
        raise ValueError("response_observability_usage_without_parsed_body")
    if value["envelope_kind"] == "interrupted":
        phase = attempt.get("failure_phase")
        if (type(status) is not int or not 200 <= status <= 299
                or usage or "output_limit_reported" in attempt):
            raise ValueError("response_observability_interrupted_mismatch")
        if value["body_status"] == "read_failed":
            if (phase != "read" or error_type not in {"network_error", "timeout"}
                    or value["body_read_error_kind"] != attempt.get("network_error_kind")
                    or (error_type == "timeout") is not (value["body_read_error_kind"] == "timeout")):
                raise ValueError("response_observability_interrupted_read_mismatch")
        elif (phase != {"decode_failed": "decode", "parse_failed": "parse"}[value["body_status"]]
              or error_type != "invalid_json_response" or "network_error_kind" in attempt):
            raise ValueError("response_observability_interrupted_decode_mismatch")
    elif value["envelope_kind"] == "http_error":
        if (error_type != "http_error" or type(status) is not int or not 300 <= status <= 599
                or value["error_category"] != _category(status)):
            raise ValueError("response_observability_http_error_mismatch")
    elif value["envelope_kind"] == "response_error":
        if error_type != "response_error" or type(status) is not int or not 200 <= status <= 299:
            raise ValueError("response_observability_response_error_mismatch")
    elif (error_type is not None or status is not None and (type(status) is not int or not 200 <= status <= 299)
          or value["body_status"] not in {"parsed", "non_object"}):
        raise ValueError("response_observability_success_mismatch")
