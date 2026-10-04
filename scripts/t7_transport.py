"""Explicit, single-attempt T7 transport; never loads credentials or writes files.

Injected openers are trusted test doubles. Production networking requires an
explicit opt-in and uses a credential-safe opener without redirects or proxies.
The caller owns durable claims, identity locks, budgets and stopping the run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
import socket
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MODELS = {
    "qwen/qwen3.7-flash": ("qwen/qwen3.7-flash-20260727", "alibaba"),
    "anthropic/claude-sonnet-5": (
        "anthropic/claude-sonnet-5-20260630", "anthropic"
    ),
}
_WIRE_KEYS = {
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "temperature", "max_tokens", "reasoning", "provider", "stream",
}
_HIDDEN_KEYS = {
    "reasoning", "reasoning_details", "reasoning_content", "thinking", "analysis"
}
_THINK = re.compile(r"<(think|thinking|reasoning|analysis)>.*?</\1>", re.I | re.S)
_THINK_MARKER = re.compile(r"</?(?:think|thinking|reasoning|analysis)\b", re.I)


class TransportError(ValueError):
    """Only bounded codes and sanitized observations; never raw exceptions."""

    def __init__(self, code, *, outcome_unknown=True, receipt=None):
        self.code = code
        self.outcome_unknown = outcome_unknown
        self.receipt = receipt or {}
        self.observation = self.receipt
        self.response = self.receipt
        super().__init__(code)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    # No ambient proxy/redirect may receive Authorization or alter the endpoint.
    return build_opener(ProxyHandler({}), _NoRedirect()).open


def _timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 120):
        raise TransportError("invalid_timeout", outcome_unknown=False)
    return float(value)


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError from None
    if not result.is_finite() or result < 0:
        raise ValueError
    return result


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError
    return value


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def invalid(_):
        raise ValueError

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        if not isinstance(result, dict):
            raise ValueError
        pending = [(result, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 64:
                raise ValueError
            if isinstance(item, dict):
                pending.extend((part, depth + 1) for part in item.values())
            elif isinstance(item, list):
                pending.extend((part, depth + 1) for part in item)
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TransportError("invalid_response_json") from None


def _set_read_timeout(response, seconds):
    # urllib HTTPS responses expose their socket through HTTPResponse.fp.
    stream = getattr(response, "fp", None)
    raw = getattr(stream, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(seconds)


def _exchange(request, *, opener, timeout, limit=MAX_RESPONSE_BYTES):
    """A single open and bounded reads under one absolute response deadline."""
    deadline = time.monotonic() + timeout
    response = None
    try:
        response = opener(request, timeout=timeout)
        if response.geturl() != request.full_url:
            raise TransportError("redirect_rejected")
        status = response.getcode()
        if type(status) is not int or status != 200:
            raise TransportError("http_status_error")
        headers = response.headers
        if headers.get("Content-Encoding", "identity").lower() != "identity":
            raise TransportError("encoded_response_rejected")
        size_header = headers.get("Content-Length")
        expected = None
        if size_header is not None:
            if not str(size_header).isascii() or not str(size_header).isdigit():
                raise TransportError("invalid_response_length")
            expected = int(size_header)
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
    except TransportError:
        raise
    except HTTPError as exc:
        code = "redirect_rejected" if 300 <= exc.code < 400 else "http_status_error"
        exc.close()
        raise TransportError(code) from None
    except (TimeoutError, socket.timeout):
        raise TransportError("response_timeout") from None
    except Exception:
        # URLError/repr/remote bodies may contain the key or sensitive text.
        raise TransportError("transport_failure") from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _has_hidden(value):
    if isinstance(value, dict):
        return any(key.lower() in _HIDDEN_KEYS and item not in (None, "", [], {})
                   or _has_hidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_hidden(item) for item in value)
    if isinstance(value, str):
        return bool(_THINK_MARKER.search(value))
    return False


def _visible_text(value):
    hidden = False
    if value is None:
        return None, hidden
    if isinstance(value, list):
        texts = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError
            if item.get("type") in _HIDDEN_KEYS:
                hidden = True
            elif item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
            else:
                raise ValueError
        value = "".join(texts)
    if not isinstance(value, str):
        raise ValueError
    if _THINK_MARKER.search(value):
        hidden = True
        value = _THINK.sub("", value)
        if _THINK_MARKER.search(value):
            raise ValueError  # Incomplete hidden span must never be persisted.
    if value.lstrip().startswith(("{", "[")):
        # Content can itself be a JSON judge/tool reply. Decode it inside a
        # wrapper so duplicate/escaped keys cannot bypass the hidden-field check.
        try:
            parsed = _decode(("{\"content\":" + value + "}").encode("utf-8"))["content"]
        except TransportError:
            raise ValueError from None
        if _has_hidden(parsed):
            hidden = True
            value = _json_bytes(_strip_hidden_structure(parsed)).decode("utf-8")
    return value, hidden


def _strip_hidden_structure(item):
    if isinstance(item, dict):
        return {key: _strip_hidden_structure(value) for key, value in item.items()
                if key.lower() not in _HIDDEN_KEYS}
    if isinstance(item, list):
        return [_strip_hidden_structure(value) for value in item]
    if isinstance(item, str):
        return _visible_text(item)[0]
    return item


def _visible_arguments(value):
    """Application deliberation is evidence; hidden provider traces are not."""
    try:
        parsed = _decode(value.encode("utf-8"))
    except TransportError:
        # Duplicate/escaped keys or incomplete JSON cannot safely be inspected.
        raise ValueError from None
    hidden = _has_hidden(parsed) or bool(_THINK_MARKER.search(value))
    return (_json_bytes(_strip_hidden_structure(parsed)).decode("utf-8") if hidden else value), hidden


def _choices(value):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError
    result, hidden = [], False
    for choice in value:
        if (not isinstance(choice, dict) or type(choice.get("index")) is not int
                or choice["index"] != 0):
            raise ValueError
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError
        content, inner_hidden = _visible_text(message.get("content"))
        hidden |= inner_hidden
        clean = {"role": "assistant", "content": content}
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError
            clean_calls = []
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError
                if (not isinstance(call.get("id"), str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", call["id"])
                        or not isinstance(function.get("name"), str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", function["name"])
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError
                arguments, arguments_hidden = _visible_arguments(function["arguments"])
                hidden |= arguments_hidden
                clean_calls.append({"id": call["id"], "type": "function",
                                    "function": {"name": function["name"],
                                                 "arguments": arguments}})
            clean["tool_calls"] = clean_calls
        reason = choice.get("finish_reason")
        safe_reason = reason if isinstance(reason, str) and reason in {
            "stop", "tool_calls", "length", "content_filter", "error"
        } else None
        result.append({"index": 0, "finish_reason": safe_reason,
                       "message": clean})
    return result, hidden


def sanitize_response(value, *, model):
    """Keep visible evidence and numeric receipts even on an invalid outcome."""
    clean = {"has_hidden_reasoning": _has_hidden(value)}
    faults = []
    for name in ("id", "model"):
        item = value.get(name)
        if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", item):
            clean[name] = item
        else:
            faults.append("missing_" + name)
    provider = value.get("provider")
    if isinstance(provider, str) and provider.lower() in {"alibaba", "anthropic"}:
        clean["provider"] = provider.lower()
    else:
        faults.append("invalid_provider")
    usage = value.get("usage")
    valid_usage = {}
    for name in ("prompt_tokens", "completion_tokens", "cost"):
        try:
            item = usage[name]
            valid_usage[name] = (str(_decimal(item)) if name == "cost"
                                 else _integer(item))
        except (KeyError, TypeError, ValueError):
            faults.append("invalid_usage_" + name)
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and "reasoning_tokens" in details:
            try:
                tokens = _integer(details["reasoning_tokens"])
                valid_usage["completion_tokens_details"] = {"reasoning_tokens": tokens}
                if tokens > valid_usage.get("completion_tokens", -1):
                    faults.append("reasoning_tokens_exceed_completion")
            except ValueError:
                faults.append("invalid_reasoning_tokens")
    clean["usage"] = valid_usage
    try:
        clean["choices"], hidden = _choices(value.get("choices"))
        clean["has_hidden_reasoning"] |= hidden
        if hidden:
            faults.append("hidden_reasoning_in_visible_output")
        reason = clean["choices"][0]["finish_reason"]
        if reason not in {"stop", "tool_calls"}:
            # Persist safe visible text but never trust an arbitrary reason body.
            clean["choices"][0]["finish_reason"] = (
                reason if reason in {"length", "content_filter", "error"} else None
            )
            faults.append("incomplete_completion")
    except (ValueError, TypeError):
        faults.append("invalid_choices")
    reasoning_tokens = valid_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if clean["has_hidden_reasoning"] or reasoning_tokens > 0:
        # The frozen request disables reasoning. A cleaned trace is evidence of
        # a contract violation, never permission to count this trial as valid.
        clean["has_hidden_reasoning"] = True
        faults.append("reasoning_not_disabled")
    canonical, expected_provider = _MODELS[model]
    if clean.get("model") not in {model, canonical}:
        faults.append("model_mismatch")
    if clean.get("provider") != expected_provider:
        faults.append("provider_mismatch")
    if "error" in value:
        faults.append("provider_error")
    if faults:
        raise TransportError(faults[0], receipt=clean)
    return clean


def _validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) - _WIRE_KEYS:
        raise TransportError("invalid_payload_fields", outcome_unknown=False)
    model = payload.get("model")
    if not isinstance(model, str) or model not in _MODELS:
        raise TransportError("unfrozen_model", outcome_unknown=False)
    expected_provider = {"only": [_MODELS[model][1]], "allow_fallbacks": False,
                         "require_parameters": True, "data_collection": "deny"}
    if (_json_bytes(payload.get("provider")) != _json_bytes(expected_provider)
            or _json_bytes(payload.get("reasoning")) != _json_bytes({"enabled": False, "exclude": True})
            or payload.get("stream") is not False):
        raise TransportError("unfrozen_route", outcome_unknown=False)
    try:
        if not _integer(payload.get("max_tokens")):
            raise ValueError
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise ValueError
        return _json_bytes(payload)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TransportError("invalid_payload", outcome_unknown=False) from None


class OpenRouterTransport:
    def __init__(self, api_key, *, timeout=120, opener=None, enable_network=False):
        self._key = api_key
        self._timeout = _timeout(timeout)
        self._open = opener
        self._enabled = enable_network is True

    def request(self, payload):
        raw = _validate_payload(payload)
        if self._open is None and not self._enabled:
            raise TransportError("network_disabled", outcome_unknown=False)
        if (not isinstance(self._key, str) or not self._key or not self._key.isascii()
                or any(ord(c) < 33 or ord(c) > 126 for c in self._key)):
            raise TransportError("invalid_api_key", outcome_unknown=False)
        request = Request(ENDPOINT, data=raw, method="POST", headers={
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
            "Accept": "application/json", "Accept-Encoding": "identity",
        })
        body = _exchange(request, opener=self._open or _opener(), timeout=self._timeout)
        if self._key.encode("ascii") in body:
            raise TransportError("credential_echo_rejected")
        return sanitize_response(_decode(body), model=payload["model"])


def estimate_request(payload, *, input_price, output_price, framing_tokens=8192):
    """A conservative local planning bound, NOT a provider token-count guarantee."""
    _validate_payload(payload)
    try:
        allowance = _integer(framing_tokens)
        if allowance < 8192:
            raise ValueError
        data = {"messages": payload["messages"], "tools": payload.get("tools", [])}
        size = len(_json_bytes(data))
        bound = size + allowance
        price = bound * _decimal(input_price) + payload["max_tokens"] * _decimal(output_price)
    except (ValueError, TypeError):
        raise TransportError("invalid_estimate_parameters", outcome_unknown=False) from None
    return {"messages_and_tools_utf8_bytes": size, "framing_token_allowance": allowance,
            "input_token_plan_bound": bound, "max_output_tokens": payload["max_tokens"],
            "reserved_cost_usd": str(price), "provider_hard_token_bound_verified": False}


def fetch_metadata(*, model, frozen_metadata, payload, prompt_token_bound,
                   opener=None, enable_network=False, timeout=120):
    """Fetch public catalog/endpoint data without credentials; no inference call.

    Missing parameter evidence (including parallel_tool_calls) is a closed gate,
    never permission to remove a frozen field or silently route elsewhere.
    """
    _validate_payload(payload)
    if model != payload["model"] or model not in _MODELS:
        raise TransportError("metadata_model_mismatch", outcome_unknown=False)
    if opener is None and enable_network is not True:
        raise TransportError("network_disabled", outcome_unknown=False)
    try:
        bound = _integer(prompt_token_bound)
    except ValueError:
        raise TransportError("invalid_prompt_bound", outcome_unknown=False) from None
    sources, documents = [], []
    for url in (MODELS_URL, MODELS_URL + "/" + model + "/endpoints"):
        request = Request(url, method="GET", headers={
            "Accept": "application/json", "Accept-Encoding": "identity"})
        raw = _exchange(request, opener=opener or _opener(), timeout=_timeout(timeout))
        sources.append({"url": url, "response_sha256": hashlib.sha256(raw).hexdigest()})
        documents.append(_decode(raw))
    receipt = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "sources": sources, "authenticated": False, "inference_requests": 0}
    try:
        current = [row for row in documents[0]["data"] if row["id"] == model]
        frozen = [row for row in frozen_metadata["models"] if row["id"] == model]
        if len(current) != 1 or len(frozen) != 1:
            raise ValueError
        canonical, provider = _MODELS[model]
        if current[0]["canonical_slug"] != canonical or frozen[0]["canonical_slug"] != canonical:
            raise TransportError("metadata_snapshot_changed", receipt=receipt)
        endpoint_data = documents[1]["data"]
        saved_data = next(row["data"] for row in frozen_metadata["selected_endpoints"]
                          if row["data"]["id"] == model)
        selected = [row for row in endpoint_data["endpoints"] if row["tag"] == provider]
        saved = [row for row in saved_data["endpoints"] if row["tag"] == provider]
        if endpoint_data["id"] != model or len(selected) != 1 or len(saved) != 1:
            raise ValueError
        endpoint = selected[0]
        if (endpoint["status"] != 0 or endpoint["model_id"] != model
                or endpoint["provider_name"].lower() != provider
                or not endpoint["name"].endswith(" | " + canonical)):
            raise TransportError("metadata_provider_changed", receipt=receipt)
        if (endpoint["pricing"] != saved[0]["pricing"]
                or current[0]["pricing"] != frozen[0]["pricing"]):
            raise TransportError("metadata_price_changed", receipt=receipt)
        supported = endpoint["supported_parameters"]
        if not isinstance(supported, list) or any(not isinstance(x, str) for x in supported):
            raise ValueError
        required = set(payload) - {"model", "messages", "provider", "stream"}
        for parameter in sorted(required):
            if parameter not in supported:
                raise TransportError("parameter_support_unverified_" + parameter, receipt=receipt)
        pricing = endpoint["pricing"]
        tiers = [{"min_prompt_tokens": 0, **pricing}, *pricing.get("overrides", [])]
        tiers.sort(key=lambda item: _integer(item["min_prompt_tokens"]))
        applicable = [tier for tier in tiers if tier["min_prompt_tokens"] <= bound]
        tier = applicable[-1]
        next_tiers = [tier["min_prompt_tokens"] for tier in tiers
                      if tier["min_prompt_tokens"] > bound]
        if (bound + payload["max_tokens"] > endpoint["context_length"]
                or payload["max_tokens"] > endpoint["max_completion_tokens"]):
            raise TransportError("metadata_token_limit", receipt=receipt)
        receipt.update({"model": model, "canonical_slug": canonical, "provider": provider,
                        "prompt_price_usd": str(_decimal(tier["prompt"])),
                        "completion_price_usd": str(_decimal(tier["completion"])),
                        "tier_min_prompt_tokens": tier["min_prompt_tokens"],
                        "tier_next_prompt_tokens": min(next_tiers) if next_tiers else None,
                        "supported_parameters": sorted(supported)})
        return receipt
    except TransportError:
        raise
    except (ValueError, TypeError, KeyError, StopIteration, IndexError):
        raise TransportError("invalid_metadata", receipt=receipt) from None
