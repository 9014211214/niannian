"""Offline transport tests: authored replies and injected openers, no API calls."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from email.message import Message
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, build_opener
from urllib.response import addinfourl

import pytest

from scripts import t7_transport as transport


KEY = "authored-unit-test-key"
ROOT = Path(__file__).resolve().parents[1]


def payload(model="qwen/qwen3.7-flash"):
    provider = "alibaba" if model.startswith("qwen/") else "anthropic"
    result = {"model": model, "messages": [{"role": "user", "content": "你好"}],
              "max_tokens": 1024, "stream": False,
              "provider": {"only": [provider], "allow_fallbacks": False,
                           "require_parameters": True, "data_collection": "deny"},
              "reasoning": {"enabled": False, "exclude": True}}
    if provider == "alibaba":
        result["temperature"] = 0.2
    return result


def reply():
    return {"id": "gen-authored", "model": "qwen/qwen3.7-flash", "provider": "Alibaba",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.00001},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "好的。"}}]}


class Response:
    def __init__(self, body, *, status=200, url=transport.ENDPOINT, headers=None):
        self.body = io.BytesIO(body if isinstance(body, bytes)
                               else json.dumps(body, ensure_ascii=False).encode())
        self.status, self.url = status, url
        self.headers = headers or {}
        self.closed = False

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status

    def read1(self, size):
        return self.body.read(size)

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, response=None, error=None):
        self.response = response or Response(reply())
        self.error, self.calls = error, []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def run(value=None, **kwargs):
    opener = Opener(Response(value or reply()))
    return transport.OpenRouterTransport(KEY, opener=opener, **kwargs), opener


def test_network_default_is_closed_and_does_not_construct_opener(monkeypatch):
    monkeypatch.setattr(transport, "_opener", lambda: pytest.fail("network opener created"))
    with pytest.raises(transport.TransportError, match="network_disabled") as caught:
        transport.OpenRouterTransport(KEY).request(payload())
    assert caught.value.outcome_unknown is False


def test_one_frozen_post_with_no_environment_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://authored.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "https://authored.invalid")
    client, opener = run()
    result = client.request(payload())
    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == transport.ENDPOINT
    assert request.method == "POST" and timeout == 120
    assert request.get_header("Authorization") == "Bearer " + KEY
    assert json.loads(request.data) == payload()
    assert result["provider"] == "alibaba"
    assert Decimal(result["usage"]["cost"]) == Decimal("0.00001")
    assert opener.response.closed


@pytest.mark.parametrize("change", [
    {"model": "other/model"}, {"stream": True}, {"stream": 0},
    {"reasoning": {"enabled": True, "exclude": True}},
    {"provider": {"only": ["alibaba"], "allow_fallbacks": True}},
    {"models": ["other/model"]}, {"max_tokens": True}, {"max_tokens": 0},
])
def test_invalid_payload_never_opens(change):
    client, opener = run()
    request = payload() | change
    with pytest.raises(transport.TransportError) as caught:
        client.request(request)
    assert not caught.value.outcome_unknown
    assert opener.calls == []


@pytest.mark.parametrize("value", [0, -1, 121, True, float("nan"), float("inf"), "120"])
def test_timeout_range_is_bounded(value):
    with pytest.raises(transport.TransportError, match="invalid_timeout"):
        transport.OpenRouterTransport(KEY, timeout=value)


@pytest.mark.parametrize("error,code", [
    (TimeoutError("secret authored trace"), "response_timeout"),
    (URLError("Bearer " + KEY), "transport_failure"),
    (RuntimeError("Bearer " + KEY), "transport_failure"),
    (HTTPError(transport.ENDPOINT, 429, "Bearer " + KEY, {}, io.BytesIO(b"private")),
     "http_status_error"),
    (HTTPError(transport.ENDPOINT, 302, "Bearer " + KEY, {}, io.BytesIO(b"private")),
     "redirect_rejected"),
])
def test_errors_stop_after_one_attempt_without_secret_or_body(error, code):
    opener = Opener(error=error)
    with pytest.raises(transport.TransportError) as caught:
        transport.OpenRouterTransport(KEY, opener=opener).request(payload())
    assert caught.value.code == code and caught.value.outcome_unknown
    assert str(caught.value) == code and caught.value.receipt == {}
    assert len(opener.calls) == 1


def test_urllib_redirect_does_not_forward_authorization():
    seen = []

    class AuthoredHTTPS(HTTPSHandler):
        def https_open(self, request):
            seen.append(request)
            headers = Message()
            headers["Location"] = "https://authored.invalid/stolen"
            result = addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
            result.msg = "Found"
            return result

    opener = build_opener(ProxyHandler({}), transport._NoRedirect(), AuthoredHTTPS())
    with pytest.raises(transport.TransportError, match="redirect_rejected"):
        transport.OpenRouterTransport(KEY, opener=opener.open).request(payload())
    assert len(seen) == 1
    assert seen[0].full_url == transport.ENDPOINT
    assert seen[0].get_header("Authorization") == "Bearer " + KEY


def test_unexpected_final_url_rejected_and_closed():
    opener = Opener(Response(reply(), url="https://authored.invalid/redirect"))
    with pytest.raises(transport.TransportError, match="redirect_rejected"):
        transport.OpenRouterTransport(KEY, opener=opener).request(payload())
    assert opener.response.closed and len(opener.calls) == 1


def test_slow_drip_read_uses_absolute_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])

    class Slow(Response):
        def read1(self, size):
            clock[0] += 0.6
            return self.body.read(1)

    opener = Opener(Slow(reply()))
    with pytest.raises(transport.TransportError, match="response_timeout"):
        transport.OpenRouterTransport(KEY, timeout=1, opener=opener).request(payload())
    assert clock[0] == 1.2 and opener.response.closed
    assert len(opener.calls) == 1


@pytest.mark.parametrize("body,headers,code", [
    (b"{}", {"Content-Length": "5"}, "response_truncated"),
    (b"{}", {"Content-Length": "-2"}, "invalid_response_length"),
    (b"{}", {"Content-Length": str(transport.MAX_RESPONSE_BYTES + 1)}, "response_too_large"),
    (b"x" * (transport.MAX_RESPONSE_BYTES + 1), {}, "response_too_large"),
    (b"{}", {"Content-Encoding": "gzip"}, "encoded_response_rejected"),
    (b'{"id":1,"id":2}', {}, "invalid_response_json"),
    (b'{"cost": NaN}', {}, "invalid_response_json"),
    (b"broken", {}, "invalid_response_json"),
], ids=["short", "negative-length", "declared-too-large", "body-too-large",
        "encoding", "duplicate-json", "nonfinite-json", "invalid-json"])
def test_body_boundaries_do_not_return_raw_content(body, headers, code):
    opener = Opener(Response(body, headers=headers))
    with pytest.raises(transport.TransportError) as caught:
        transport.OpenRouterTransport(KEY, opener=opener).request(payload())
    assert caught.value.code == code and caught.value.receipt == {}
    assert opener.response.closed


@pytest.mark.parametrize("field,value", [
    ("cost", True), ("cost", -0.1), ("cost", "NaN"), ("cost", "Infinity"),
    ("prompt_tokens", True), ("prompt_tokens", 2.2), ("prompt_tokens", "123"),
    ("completion_tokens", -1), ("completion_tokens", None),
])
def test_usage_strictness_preserves_other_known_receipts(field, value):
    source = reply()
    source["usage"][field] = value
    client, _ = run(source)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    receipt = caught.value.receipt
    assert field not in receipt["usage"]
    assert receipt["choices"][0]["message"]["content"] == "好的。"
    if field != "cost":
        assert Decimal(receipt["usage"]["cost"]) == Decimal("0.00001")


def test_missing_usage_stops_and_retains_visible_output():
    source = reply()
    del source["usage"]
    client, _ = run(source)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    assert caught.value.receipt["usage"] == {}
    assert caught.value.receipt["choices"]


@pytest.mark.parametrize("model,provider", [
    ("qwen/qwen3.7-flash", "Alibaba"),
    ("qwen/qwen3.7-flash-20260727", "alibaba"),
])
def test_model_alias_and_canonical_are_accepted_for_caller_lock(model, provider):
    source = reply() | {"model": model, "provider": provider}
    client, _ = run(source)
    assert client.request(payload())["model"] == model


@pytest.mark.parametrize("change", [
    {"provider": "Anthropic"}, {"provider": "unknown"}, {"provider": None},
    {"model": "qwen/qwen3.7-flash-other"}, {"model": None},
])
def test_provider_model_drift_is_a_stop_with_known_cost(change):
    client, opener = run(reply() | change)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    assert "cost" in caught.value.receipt["usage"] and len(opener.calls) == 1


def test_hidden_provider_trace_stops_with_clean_visible_output_and_known_cost():
    source = reply() | {"debug": "must-discard", "reasoning": "hidden-sentinel"}
    source["usage"].update({"extra": "must-discard", "completion_tokens_details": {
        "reasoning_tokens": 40, "hidden_text": "hidden-sentinel"}})
    source["choices"][0]["message"].update({"reasoning": "hidden-sentinel", "audio": "discard"})
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="reasoning_not_disabled") as caught:
        client.request(payload())
    result = caught.value.receipt
    assert set(result) == {"id", "model", "provider", "usage", "choices", "has_hidden_reasoning"}
    assert result["has_hidden_reasoning"] is True
    assert "hidden-sentinel" not in json.dumps(result) and "must-discard" not in json.dumps(result)
    assert result["usage"]["completion_tokens"] == 50
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 40
    assert Decimal(result["usage"]["cost"]) == Decimal("0.00001")


@pytest.mark.parametrize("reasoning_tokens", [1, 50])
def test_positive_reasoning_token_receipt_stops_even_without_hidden_text(reasoning_tokens):
    source = reply()
    source["usage"]["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="reasoning_not_disabled") as caught:
        client.request(payload())
    result = caught.value.receipt
    assert result["has_hidden_reasoning"]
    assert result["choices"][0]["message"]["content"] == "好的。"
    assert result["usage"]["completion_tokens"] == 50
    assert Decimal(result["usage"]["cost"]) == Decimal("0.00001")


def test_zero_reasoning_tokens_can_remain_a_valid_receipt():
    source = reply()
    source["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
    client, _ = run(source)
    result = client.request(payload())
    assert result["has_hidden_reasoning"] is False
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


@pytest.mark.parametrize("content", [
    "<think>hidden-sentinel</think>好的。", "<analysis>hidden-sentinel</analysis>好的。",
    [{"type": "thinking", "text": "hidden-sentinel"}, {"type": "text", "text": "好的。"}],
])
def test_hidden_text_in_visible_output_is_sanitized_then_stopped(content):
    source = reply()
    source["choices"][0]["message"]["content"] = content
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="hidden_reasoning_in_visible_output") as caught:
        client.request(payload())
    assert "hidden-sentinel" not in json.dumps(caught.value.receipt)
    assert caught.value.receipt["choices"][0]["message"]["content"] == "好的。"


def test_function_hidden_fields_removed_without_removing_application_deliberation():
    source = reply()
    source["choices"][0]["finish_reason"] = "tool_calls"
    source["choices"][0]["message"]["tool_calls"] = [{"id": "call-1", "type": "function",
        "function": {"name": "aria_turn_decision", "arguments": json.dumps({
            "analysis": "hidden-sentinel", "deliberation": {"person_read": "fixture",
            "options": ["continue"], "reasoning_content": "hidden-sentinel"},
            "response": "好的。"})}}]
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="hidden_reasoning_in_visible_output") as caught:
        client.request(payload())
    result = caught.value.receipt
    arguments = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert arguments["deliberation"] == {"person_read": "fixture", "options": ["continue"]}
    assert result["has_hidden_reasoning"] and "hidden-sentinel" not in json.dumps(result)


def test_content_json_hidden_fields_are_removed_and_stopped():
    source = reply()
    source["choices"][0]["message"]["content"] = json.dumps({
        "thinking": "hidden-sentinel", "answer": [["q_fixture", 0, 1, 0, 0]]})
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="hidden_reasoning_in_visible_output") as caught:
        client.request(payload())
    assert "hidden-sentinel" not in json.dumps(caught.value.receipt)
    assert json.loads(caught.value.receipt["choices"][0]["message"]["content"]) == {
        "answer": [["q_fixture", 0, 1, 0, 0]]}


def test_incomplete_hidden_span_is_not_persisted():
    source = reply()
    source["choices"][0]["message"]["content"] = "<think>hidden-sentinel"
    client, _ = run(source)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    assert "choices" not in caught.value.receipt
    assert "hidden-sentinel" not in json.dumps(caught.value.receipt)
    assert caught.value.receipt["has_hidden_reasoning"]


@pytest.mark.parametrize("arguments", [
    '{"reasoning":"hidden-sentinel", "response":',
    '{"\\u0072easoning":"hidden-sentinel","response":"ok","response":"other"}',
])
def test_malformed_arguments_cannot_hide_uninspectable_reasoning(arguments):
    source = reply()
    source["choices"][0]["message"]["tool_calls"] = [{"id": "call-1", "type": "function",
        "function": {"name": "aria_turn_decision", "arguments": arguments}}]
    client, _ = run(source)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    assert "choices" not in caught.value.receipt
    assert "hidden-sentinel" not in json.dumps(caught.value.receipt)


def test_finish_reason_cannot_smuggle_unknown_object_fields():
    source = reply()
    source["choices"][0]["finish_reason"] = {"reasoning": "hidden-sentinel"}
    client, _ = run(source)
    with pytest.raises(transport.TransportError) as caught:
        client.request(payload())
    assert caught.value.receipt["choices"][0]["finish_reason"] is None
    assert "hidden-sentinel" not in json.dumps(caught.value.receipt)


def test_reasoning_tokens_cannot_be_discounted_out_of_completion_usage():
    source = reply()
    source["usage"]["completion_tokens_details"] = {"reasoning_tokens": 51}
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="reasoning_tokens_exceed_completion") as caught:
        client.request(payload())
    assert caught.value.receipt["usage"]["completion_tokens"] == 50
    assert caught.value.receipt["usage"]["completion_tokens_details"]["reasoning_tokens"] == 51


@pytest.mark.parametrize("reason", ["length", "content_filter", "unknown body sentinel"])
def test_truncated_or_unknown_finish_preserves_visible_reply_and_stops(reason):
    source = reply()
    source["choices"][0]["finish_reason"] = reason
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="incomplete_completion") as caught:
        client.request(payload())
    assert caught.value.receipt["choices"][0]["message"]["content"] == "好的。"
    assert "unknown body sentinel" not in json.dumps(caught.value.receipt)


def test_credentials_echo_is_never_exposed():
    source = reply()
    source["choices"][0]["message"]["content"] = KEY
    client, _ = run(source)
    with pytest.raises(transport.TransportError, match="credential_echo_rejected") as caught:
        client.request(payload())
    assert caught.value.receipt == {} and KEY not in repr(caught.value)


def test_estimate_is_utf8_plus_framing_with_no_cache_or_reasoning_discount():
    request = payload()
    estimate = transport.estimate_request(request, input_price="0.00000003",
                                           output_price="0.00000013")
    encoded = json.dumps({"messages": request["messages"], "tools": []}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode()
    assert estimate["input_token_plan_bound"] == len(encoded) + 8192
    expected = Decimal(len(encoded) + 8192) * Decimal("0.00000003") + 1024 * Decimal("0.00000013")
    assert Decimal(estimate["reserved_cost_usd"]) == expected
    assert estimate["provider_hard_token_bound_verified"] is False


def metadata_fixture():
    frozen = json.loads((ROOT / "experiments/t6_20260930/public_metadata.json").read_text(encoding="utf-8"))
    documents = {transport.MODELS_URL: {"data": deepcopy(frozen["models"])}}
    for row in frozen["selected_endpoints"]:
        documents[row["source_url"]] = {"data": deepcopy(row["data"])}
    calls = []

    def opener(request, *, timeout):
        calls.append(request)
        assert request.method == "GET" and request.get_header("Authorization") is None
        assert request.full_url in documents and timeout == 120
        return Response(documents[request.full_url], url=request.full_url)

    return frozen, documents, calls, opener


def test_public_metadata_has_no_key_and_receipts_cover_both_sources():
    frozen, _, calls, opener = metadata_fixture()
    result = transport.fetch_metadata(model=payload()["model"], frozen_metadata=frozen,
        payload=payload(), prompt_token_bound=20000, opener=opener)
    assert len(calls) == 2 and len(result["sources"]) == 2
    assert all(len(row["response_sha256"]) == 64 for row in result["sources"])
    assert result["provider"] == "alibaba" and result["tier_next_prompt_tokens"] == 32000
    assert result["tier_min_prompt_tokens"] == 0
    assert result["authenticated"] is False and result["inference_requests"] == 0


@pytest.mark.parametrize("bound,price,tier", [(31999, "0.00000003", 0),
    (32000, "0.0000001", 32000), (256000, "0.0000002", 256000)])
def test_metadata_reports_actual_price_tier(bound, price, tier):
    frozen, _, _, opener = metadata_fixture()
    result = transport.fetch_metadata(model=payload()["model"], frozen_metadata=frozen,
        payload=payload(), prompt_token_bound=bound, opener=opener)
    assert Decimal(result["prompt_price_usd"]) == Decimal(price)
    assert result["tier_min_prompt_tokens"] == tier


@pytest.mark.parametrize("change,code", [
    ("snapshot", "metadata_snapshot_changed"), ("price", "metadata_price_changed"),
    ("provider", "metadata_provider_changed"),
    ("parameter", "parameter_support_unverified_temperature"),
])
def test_catalog_changes_are_closed_gates(change, code):
    frozen, documents, calls, opener = metadata_fixture()
    model = payload()["model"]
    endpoint = documents[transport.MODELS_URL + "/" + model + "/endpoints"]["data"]["endpoints"][0]
    if change == "snapshot":
        documents[transport.MODELS_URL]["data"][0]["canonical_slug"] += "-new"
    elif change == "price":
        endpoint["pricing"]["prompt"] = "0.01"
    elif change == "provider":
        endpoint["status"] = 1
    else:
        endpoint["supported_parameters"].remove("temperature")
    with pytest.raises(transport.TransportError, match=code) as caught:
        transport.fetch_metadata(model=model, frozen_metadata=frozen, payload=payload(),
                                 prompt_token_bound=20000, opener=opener)
    assert len(calls) == 2 and len(caught.value.receipt["sources"]) == 2


def test_frozen_parallel_tool_calls_is_preserved_and_unverified_support_blocks():
    frozen, _, _, opener = metadata_fixture()
    request = payload() | {"parallel_tool_calls": False}
    original = deepcopy(request)
    with pytest.raises(transport.TransportError, match="parameter_support_unverified_parallel_tool_calls"):
        transport.fetch_metadata(model=request["model"], frozen_metadata=frozen, payload=request,
                                 prompt_token_bound=20000, opener=opener)
    assert request == original


def test_anthropic_selected_provider_and_no_temperature():
    frozen, _, calls, opener = metadata_fixture()
    request = payload("anthropic/claude-sonnet-5")
    result = transport.fetch_metadata(model=request["model"], frozen_metadata=frozen,
        payload=request, prompt_token_bound=20000, opener=opener)
    assert result["provider"] == "anthropic" and len(calls) == 2
    assert Decimal(result["prompt_price_usd"]) == Decimal("0.000002")


def test_metadata_network_also_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(transport, "_opener", lambda: pytest.fail("network created"))
    with pytest.raises(transport.TransportError, match="network_disabled"):
        transport.fetch_metadata(model=payload()["model"], frozen_metadata={},
                                 payload=payload(), prompt_token_bound=20000)
