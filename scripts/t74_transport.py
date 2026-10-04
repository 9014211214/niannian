"""T7.4 keeps useful, bounded HTTP error categories without retaining messages.

The T7.2 single-attempt exchange and request implementation are reused through
private function globals. Only the error-body parser is extended; their frozen
module, payload validation, successful sanitizer, deadlines and retry policy
remain untouched. A category records a provider message pattern or HTTP status,
not proof of a root cause, an unbilled attempt or permission to retry.
"""
from __future__ import annotations

import re
from types import FunctionType

from scripts import t72_transport as previous


TransportError = previous.TransportError
ENDPOINT = previous.ENDPOINT
MAX_ERROR_BYTES = previous.MAX_ERROR_BYTES
MAX_REASON_MESSAGE_BYTES = 8192
ERROR_REASONS = frozenset({
    "no_endpoints_parameter_support", "no_endpoints_data_policy",
    "no_endpoints_privacy", "no_endpoints_available", "model_not_found",
    "authentication_failed", "insufficient_credits", "rate_limited",
    "permission_denied", "invalid_request", "upstream_unavailable",
    "request_timeout", "redirect_rejected", "unclassified_http_error",
    "ambiguous_api_error",
})
_STATUS_REASONS = {
    400: "invalid_request", 401: "authentication_failed",
    402: "insufficient_credits", 403: "permission_denied",
    408: "request_timeout", 422: "invalid_request", 429: "rate_limited",
}
_NO_ENDPOINTS = re.compile(
    r"\bno\s+(?:(?:available|eligible)\s+)?(?:endpoints|providers)\b"
)
_PARAMETERS = re.compile(
    r"\b(?:parameters?|tool(?:s|\s+(?:use|calling|choice))?|"
    r"reasoning|response\s+format|json\s+mode)\b"
)
_DATA_POLICY = re.compile(r"\b(?:data\s+(?:policy|collection|retention\s+policy))\b")
_PRIVACY = re.compile(r"\b(?:privacy|zero\s+data\s+retention|zdr)\b")
_MODEL = re.compile(
    r"\b(?:model(?:\s+id)?\s+(?:not\s+found|does\s+not\s+exist|is\s+invalid)|"
    r"(?:invalid|unknown|nonexistent)\s+model|not\s+a\s+valid\s+model\s+id)\b"
)
_MESSAGE_REASONS = (
    ("authentication_failed", re.compile(
        r"\b(?:invalid\s+(?:api\s+)?key|(?:api\s+)?key\s+(?:is\s+)?invalid|"
        r"authentication\s+(?:failed|error)|unauthorized|missing\s+(?:api\s+)?key)\b")),
    ("insufficient_credits", re.compile(
        r"\b(?:insufficient\s+(?:credits|balance|funds)|payment\s+required|"
        r"credit\s+balance\s+(?:is\s+)?(?:too\s+low|exhausted))\b")),
    ("rate_limited", re.compile(r"\b(?:rate\s+limit(?:ed|\s+exceeded)?|too\s+many\s+requests)\b")),
    ("permission_denied", re.compile(r"\b(?:permission\s+denied|access\s+denied|forbidden)\b")),
)


def classify_error_message(message):
    """Return an allowlisted category or None; never return any input text.

    Classification reads only error.message. Nested provider traces, URLs and
    hidden reasoning are not used as evidence. Malformed/oversized messages are
    left to the HTTP-status fallback. Multiple competing patterns are marked
    ambiguous, rather than arbitrarily choosing a diagnosis.
    """
    if not isinstance(message, str) or len(message) > MAX_REASON_MESSAGE_BYTES:
        return None
    try:
        if len(message.encode("utf-8")) > MAX_REASON_MESSAGE_BYTES:
            return None
    except UnicodeError:
        return None
    if previous.frozen._THINK_MARKER.search(message):
        return None
    normalized = re.sub(r"[a-z]+://\S+", " ", message.lower())
    normalized = re.sub(r"\s+", " ", normalized.replace("_", " ").replace("-", " "))
    if _NO_ENDPOINTS.search(normalized):
        reasons = []
        if _PARAMETERS.search(normalized):
            reasons.append("no_endpoints_parameter_support")
        if _DATA_POLICY.search(normalized):
            reasons.append("no_endpoints_data_policy")
        elif _PRIVACY.search(normalized):
            reasons.append("no_endpoints_privacy")
        if len(reasons) > 1:
            return "ambiguous_api_error"
        return reasons[0] if reasons else "no_endpoints_available"
    reasons = [reason for reason, pattern in _MESSAGE_REASONS if pattern.search(normalized)]
    if _MODEL.search(normalized):
        reasons.append("model_not_found")
    if len(reasons) > 1:
        return "ambiguous_api_error"
    return reasons[0] if reasons else None


def _body_fields(body, key):
    fields = previous._body_fields(body, key)
    if fields.get("body_status") != "json":
        return fields
    # The previous parser already performed bounded, strict decoding and
    # credential-echo rejection. The message itself is never returned.
    value = previous.frozen._decode(body)
    error = value.get("error")
    reason = classify_error_message(error.get("message")) if isinstance(error, dict) else None
    if reason is not None:
        fields["api_error_reason"] = reason
        fields["api_error_reason_source"] = "message_pattern"
    return fields


def _clone(function, **bindings):
    cloned = FunctionType(function.__code__, {**function.__globals__, **bindings},
                          function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return cloned


_body_exchange = _clone(previous._exchange, _body_fields=_body_fields)


def _exchange(request, *, opener, timeout, key):
    try:
        return _body_exchange(request, opener=opener, timeout=timeout, key=key)
    except TransportError as exc:
        receipt = dict(exc.receipt)
        diagnostic = dict(receipt.get("http_diagnostic", {}))
        status = diagnostic.get("http_status")
        if type(status) is int and status != 200 and "api_error_reason" not in diagnostic:
            if 300 <= status < 400:
                reason = "redirect_rejected"
            elif status >= 500:
                reason = "upstream_unavailable"
            else:
                reason = _STATUS_REASONS.get(status, "unclassified_http_error")
            diagnostic["api_error_reason"] = reason
            diagnostic["api_error_reason_source"] = "http_status"
        if diagnostic:
            receipt["http_diagnostic"] = diagnostic
        raise TransportError(exc.code, outcome_unknown=exc.outcome_unknown,
                             receipt=receipt) from None


class OpenRouterTransport(previous.OpenRouterTransport):
    request = _clone(previous.OpenRouterTransport.request, _exchange=_exchange)
