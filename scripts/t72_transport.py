"""Independent T7.2 HTTP diagnostics; no credentials/files/automatic retries.

The frozen T7 payload validator and successful-response sanitizer are reused.
HTTP failures retain status and narrowly typed identifiers, never response text,
arbitrary headers or hidden reasoning. Status categories describe HTTP only:
even 401/404 and a reported cost of zero do NOT establish an unbilled attempt.
Only the caller's separately reconciled evidence can resolve that unknown cost.
header_generation_id and generation_id retain header/body provenance separately;
generation_id_conflict flags disagreement and never selects an authoritative ID.
Injected openers are trusted test doubles; production networking is opt-in.
"""
from __future__ import annotations

import base64
import re
import socket
import time
from urllib.error import HTTPError
from urllib.request import Request

from scripts import t7_transport as frozen


TransportError = frozen.TransportError
ENDPOINT = frozen.ENDPOINT
MAX_ERROR_BYTES = 64 * 1024
_GENERATION = re.compile(r"gen-[0-9]{10,16}-[A-Za-z0-9]{8,64}\Z")
_REQUEST = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_CATEGORIES = {400: "bad_request", 401: "authentication_error",
               402: "payment_required", 403: "forbidden", 404: "not_found",
               408: "request_timeout", 422: "unprocessable_entity",
               429: "rate_limited"}


def _category(status):
    if status == 200:
        return "ok"
    if 300 <= status < 400:
        return "redirect"
    if status >= 500:
        return "server_error"
    return _CATEGORIES.get(status, "other_http_status")


def _secret_in_text(value, key, *, fragments=False):
    if key in value:
        return True
    if not fragments:
        return False
    # IDs must not retain recognizable key fragments, including punctuation-
    # stripped or common base64 encodings. Short keys use their whole value.
    needles = (key, re.sub(r"[^A-Za-z0-9]", "", key),
               base64.b64encode(key.encode("ascii")).decode().rstrip("="),
               base64.urlsafe_b64encode(key.encode("ascii")).decode().rstrip("="))
    haystack = re.sub(r"[^A-Za-z0-9]", "", value).lower()
    for needle in needles:
        needle = re.sub(r"[^A-Za-z0-9]", "", needle).lower()
        width = min(8, len(needle))
        if width and any(needle[i:i + width] in haystack
                         for i in range(len(needle) - width + 1)):
            return True
    return False


def _secret_in_json(value, key):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and _secret_in_text(item, key):
            return True
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _safe_id(value, pattern, key):
    return (value if isinstance(value, str) and pattern.fullmatch(value)
            and not _secret_in_text(value, key, fragments=True) else None)


def _header_fields(headers, key):
    result = {}
    request_id = _safe_id(headers.get("X-Request-ID"), _REQUEST, key)
    if request_id is not None:
        result["request_id"] = request_id
    generation_id = _safe_id(headers.get("X-Generation-Id"), _GENERATION, key)
    if generation_id is not None:
        result["header_generation_id"] = generation_id
    retry = headers.get("Retry-After")
    if (isinstance(retry, str) and re.fullmatch(r"[0-9]{1,5}", retry)
            and int(retry) <= 86400 and not _secret_in_text(retry, key)):
        result["retry_after_seconds"] = int(retry)
    return result


def _merge_body_fields(diagnostic, fields):
    diagnostic.update(fields)
    header_id = diagnostic.get("header_generation_id")
    body_id = diagnostic.get("generation_id")
    if header_id is not None and body_id is not None and header_id != body_id:
        diagnostic["generation_id_conflict"] = True


def _body_fields(body, key):
    if key.encode("ascii") in body:
        return {"body_status": "credential_echo_rejected"}
    try:
        value = frozen._decode(body)
    except TransportError:
        return {"body_status": "invalid_response_json"}
    if _secret_in_json(value, key):
        return {"body_status": "credential_echo_rejected"}
    result = {"body_status": "json"}
    generation_id = _safe_id(value.get("id"), _GENERATION, key)
    if generation_id is not None:
        result["generation_id"] = generation_id
    error = value.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if type(code) is int and 100 <= code <= 599:
            result["api_error_code"] = code
    return result


def _set_read_timeout(response, seconds):
    # HTTPError.fp wraps HTTPResponse, adding one layer to normal HTTPS reads.
    cursor = response
    for _ in range(4):
        raw = getattr(cursor, "raw", None)
        sock = getattr(raw, "_sock", None) or getattr(cursor, "_sock", None)
        if sock is not None:
            sock.settimeout(seconds)
            return
        cursor = getattr(cursor, "fp", None)
        if cursor is None:
            return


def _read_body(response, *, deadline, limit):
    headers = response.headers
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise TransportError("encoded_response_rejected")
    length = headers.get("Content-Length")
    expected = None
    if length is not None:
        if not isinstance(length, str) or not re.fullmatch(r"[0-9]{1,10}", length):
            raise TransportError("invalid_response_length")
        expected = int(length)
        if expected > limit:
            raise TransportError("response_too_large")
    parts, size = [], 0
    read = getattr(response, "read1", None) or response.read
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportError("response_timeout")
        _set_read_timeout(response, remaining)
        block = read(min(65536, limit + 1 - size))
        if time.monotonic() > deadline:
            raise TransportError("response_timeout")
        if not isinstance(block, bytes):
            raise TransportError("invalid_response_bytes")
        if not block:
            break
        size += len(block)
        if size > limit:
            raise TransportError("response_too_large")
        parts.append(block)
    if expected is not None and size != expected:
        raise TransportError("response_truncated")
    return b"".join(parts)


def _exchange(request, *, opener, timeout, key):
    deadline = time.monotonic() + timeout
    response, diagnostic = None, {}
    try:
        try:
            response = opener(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        status = response.getcode()
        if type(status) is not int or not 100 <= status <= 599:
            raise TransportError("http_status_error")
        diagnostic = {"http_status": status, "http_category": _category(status)}
        diagnostic.update(_header_fields(response.headers, key))
        if response.geturl() != request.full_url:
            raise TransportError("redirect_rejected")
        if status != 200:
            code = "redirect_rejected" if 300 <= status < 400 else "http_status_error"
            if 300 <= status < 400:
                diagnostic["body_status"] = "not_read_redirect"
            else:
                try:
                    body = _read_body(response, deadline=deadline, limit=MAX_ERROR_BYTES)
                    _merge_body_fields(diagnostic, _body_fields(body, key))
                except TransportError as exc:
                    diagnostic["body_status"] = exc.code
                except (TimeoutError, socket.timeout):
                    diagnostic["body_status"] = "response_timeout"
                except Exception:
                    diagnostic["body_status"] = "transport_failure"
            raise TransportError(code)
        return (_read_body(response, deadline=deadline, limit=frozen.MAX_RESPONSE_BYTES),
                diagnostic)
    except TransportError as exc:
        raise TransportError(exc.code, receipt={"http_diagnostic": diagnostic}
                             if diagnostic else {}) from None
    except (TimeoutError, socket.timeout):
        raise TransportError("response_timeout", receipt={"http_diagnostic": diagnostic}
                             if diagnostic else {}) from None
    except Exception:
        raise TransportError("transport_failure", receipt={"http_diagnostic": diagnostic}
                             if diagnostic else {}) from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


class OpenRouterTransport:
    def __init__(self, api_key, *, timeout=120, opener=None, enable_network=False):
        self._key = api_key
        self._timeout = frozen._timeout(timeout)
        self._open = opener
        self._enabled = enable_network is True

    def request(self, payload):
        raw = frozen._validate_payload(payload)
        if self._open is None and not self._enabled:
            raise TransportError("network_disabled", outcome_unknown=False)
        if (not isinstance(self._key, str) or not self._key or not self._key.isascii()
                or any(ord(c) < 33 or ord(c) > 126 for c in self._key)):
            raise TransportError("invalid_api_key", outcome_unknown=False)
        request = Request(ENDPOINT, data=raw, method="POST", headers={
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
            "Accept": "application/json", "Accept-Encoding": "identity",
        })
        body, diagnostic = _exchange(request, opener=self._open or frozen._opener(),
                                     timeout=self._timeout, key=self._key)
        try:
            # Decoding is needed to detect an escaped credential echo.
            if self._key.encode("ascii") in body:
                raise TransportError("credential_echo_rejected")
            decoded = frozen._decode(body)
            if _secret_in_json(decoded, self._key):
                raise TransportError("credential_echo_rejected")
            generation_id = _safe_id(decoded.get("id"), _GENERATION, self._key)
            if generation_id is not None:
                _merge_body_fields(diagnostic, {"generation_id": generation_id})
            return frozen.sanitize_response(decoded, model=payload["model"])
        except TransportError as exc:
            receipt = dict(exc.receipt)
            receipt["http_diagnostic"] = diagnostic
            raise TransportError(exc.code, receipt=receipt) from None
