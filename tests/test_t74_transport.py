"""T7.4 uses authored HTTP responses; no credentials, .env or network access."""
from copy import deepcopy
from email.message import Message
import io
import json
from urllib.error import HTTPError

import pytest

from scripts import t7_transport as frozen
from scripts import t72_transport as previous
from scripts import t74_transport as transport
from tests.test_t7_transport import KEY, Opener, Response, payload, reply


GENERATION_ID = "gen-1790777949000-AbCdEfGh12345678"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(frozen, "_opener", lambda: pytest.fail("network forbidden"))


def failure(body, *, status=404, headers=None, error=False, key=KEY):
    if error:
        stream = io.BytesIO(json.dumps(body).encode())
        parsed_headers = Message()
        for name, value in (headers or {}).items():
            parsed_headers[name] = value
        response = HTTPError(transport.ENDPOINT, status, "private HTTP reason",
                             parsed_headers, stream)
        opener = Opener(error=response)
    else:
        response = Response(body, status=status, headers=headers or {})
        opener = Opener(response)
    with pytest.raises(transport.TransportError) as caught:
        transport.OpenRouterTransport(key, opener=opener).request(payload())
    assert len(opener.calls) == 1
    assert caught.value.outcome_unknown is True
    assert response.closed if not error else stream.closed
    return caught.value


@pytest.mark.parametrize("message,reason", [
    ("No endpoints found that support tool use.", "no_endpoints_parameter_support"),
    ("No endpoints found for the provided parameters.", "no_endpoints_parameter_support"),
    ("No endpoints found that support tool_choice=function.", "no_endpoints_parameter_support"),
    ("No endpoints found that support reasoning.", "no_endpoints_parameter_support"),
    ("No endpoints found that match your data policy. Configure: https://openrouter.ai/settings/privacy",
     "no_endpoints_data_policy"),
    ("No endpoints found matching data_collection=deny", "no_endpoints_data_policy"),
    ("No endpoints available with your privacy settings", "no_endpoints_privacy"),
    ("No eligible providers available with ZDR", "no_endpoints_privacy"),
    ("No endpoints found", "no_endpoints_available"),
    ("Unknown model", "model_not_found"),
    ("qwen/example is not a valid model ID", "model_not_found"),
    ("Model does not exist", "model_not_found"),
    ("Invalid API key", "authentication_failed"),
    ("Authentication failed", "authentication_failed"),
    ("Insufficient credits", "insufficient_credits"),
    ("Payment required", "insufficient_credits"),
    ("Rate limit exceeded", "rate_limited"),
    ("Too many requests", "rate_limited"),
    ("Access denied", "permission_denied"),
    ("No endpoints found matching data policy and parameters", "ambiguous_api_error"),
    ("Invalid API key; insufficient credits", "ambiguous_api_error"),
])
@pytest.mark.parametrize("error", [False, True])
def test_same_failure_retains_fixed_reason_and_no_message(message, reason, error):
    body = {"id": GENERATION_ID, "error": {"code": 404,
            "message": message + " private sentinel",
            "metadata": {"raw": "provider private trace", "reasoning": "hidden sentinel"}},
            "usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}
    exc = failure(body, error=error, headers={"X-Generation-Id": GENERATION_ID})
    diagnostic = exc.receipt["http_diagnostic"]
    assert diagnostic["api_error_reason"] == reason
    assert diagnostic["api_error_reason_source"] == "message_pattern"
    assert diagnostic["http_status"] == 404
    assert diagnostic["api_error_code"] == 404
    assert diagnostic["header_generation_id"] == diagnostic["generation_id"] == GENERATION_ID
    assert exc.code == "http_status_error"
    serialized = json.dumps(exc.receipt)
    assert all(value not in serialized for value in (
        message, "private sentinel", "provider private trace", "hidden sentinel",
        KEY, "usage", "cost", "private HTTP reason"))


@pytest.mark.parametrize("status,reason", [
    (400, "invalid_request"), (401, "authentication_failed"),
    (402, "insufficient_credits"), (403, "permission_denied"),
    (404, "unclassified_http_error"), (408, "request_timeout"),
    (422, "invalid_request"), (429, "rate_limited"),
    (502, "upstream_unavailable"), (418, "unclassified_http_error"),
])
def test_status_fallback_preserves_uncertainty(status, reason):
    exc = failure({"error": {"code": status, "message": "opaque private message"}}, status=status)
    assert exc.receipt["http_diagnostic"]["api_error_reason"] == reason
    assert exc.receipt["http_diagnostic"]["api_error_reason_source"] == "http_status"


@pytest.mark.parametrize("message", [None, True, {}, [], "x" * 8193,
    "密" * 3000, "\ud800", "<think>No endpoints found for tool use</think>",
    "<thinking>No endpoints found matching privacy</thinking>"] ,
    ids=["none", "boolean", "object", "array", "long-ascii", "long-utf8",
         "invalid-unicode", "hidden-think", "hidden-thinking"])
def test_classifier_ignores_non_surface_or_oversized_messages(message):
    assert transport.classify_error_message(message) is None


@pytest.mark.parametrize("body,body_status", [
    (b"{", "invalid_response_json"),
    (b'{"error":1,"error":2}', "invalid_response_json"),
    (b"x" * (transport.MAX_ERROR_BYTES + 1), "response_too_large"),
    (json.dumps({"error": {"message": "No endpoints found for tool use " + KEY}}).encode(),
     "credential_echo_rejected"),
    (json.dumps({"error": {"message": "No endpoints found for tool use"},
                 "unexpected": KEY}).encode(), "credential_echo_rejected"),
], ids=["invalid-json", "duplicate-json", "oversize", "message-key", "nested-key"])
def test_unsafe_or_unreadable_body_keeps_http_only(body, body_status):
    exc = failure(body, status=404)
    diagnostic = exc.receipt["http_diagnostic"]
    assert diagnostic["body_status"] == body_status
    assert diagnostic["api_error_reason"] == "unclassified_http_error"
    assert diagnostic["api_error_reason_source"] == "http_status"
    assert KEY not in json.dumps(exc.receipt)


def test_escaped_key_echo_cannot_be_classified_or_retained():
    key = "synthetic-key-<&>"
    message = "No endpoints found for tool use " + key
    raw = json.dumps({"error": {"message": message}}).encode().replace(b"<", b"\\u003c")
    diagnostic = failure(raw, key=key).receipt["http_diagnostic"]
    assert diagnostic["body_status"] == "credential_echo_rejected"
    assert diagnostic["api_error_reason_source"] == "http_status"
    assert key not in json.dumps(diagnostic)


def test_nested_trace_and_urls_do_not_supply_reason():
    body = {"error": {"message": "private https://example.invalid/no-endpoints-found-tool-use",
                     "metadata": {"raw": "No endpoints found matching privacy",
                                  "reasoning": "Invalid API key"}}}
    diagnostic = failure(body).receipt["http_diagnostic"]
    assert diagnostic["api_error_reason"] == "unclassified_http_error"
    assert diagnostic["api_error_reason_source"] == "http_status"


def test_success_wire_and_result_are_unchanged_and_frozen_globals_intact():
    original_exchange = previous.OpenRouterTransport.request.__globals__["_exchange"]
    original_parser = previous._exchange.__globals__["_body_fields"]
    request = payload() | {"tool_choice": "auto"}
    original = deepcopy(request)
    opener = Opener(Response(reply()))
    result = transport.OpenRouterTransport(KEY, opener=opener).request(request)
    assert request == original
    assert opener.calls[0][0].data == frozen._validate_payload(original)
    assert result == frozen.sanitize_response(reply(), model=request["model"])
    assert len(opener.calls) == 1 and opener.response.closed
    assert previous.OpenRouterTransport.request.__globals__["_exchange"] is original_exchange
    assert previous._exchange.__globals__["_body_fields"] is original_parser


def test_http_200_sanitizer_failure_keeps_original_semantics():
    source = reply() | {"reasoning": "hidden private sentinel"}
    exc = failure(source, status=200, headers={"X-Generation-Id": GENERATION_ID})
    diagnostic = exc.receipt["http_diagnostic"]
    assert diagnostic["http_status"] == 200
    assert diagnostic["header_generation_id"] == GENERATION_ID
    assert "api_error_reason" not in diagnostic
    assert "hidden private sentinel" not in json.dumps(exc.receipt)


def test_no_network_or_invalid_credential_before_single_attempt():
    with pytest.raises(transport.TransportError, match="network_disabled") as caught:
        transport.OpenRouterTransport(KEY).request(payload())
    assert caught.value.outcome_unknown is False
    opener = Opener()
    with pytest.raises(transport.TransportError, match="invalid_api_key") as caught:
        transport.OpenRouterTransport("", opener=opener).request(payload())
    assert caught.value.outcome_unknown is False and not opener.calls


def test_redirect_body_is_not_read_and_no_second_call():
    diagnostic = failure(b"No endpoints found matching privacy", status=307).receipt["http_diagnostic"]
    assert diagnostic["body_status"] == "not_read_redirect"
    assert diagnostic["api_error_reason"] == "redirect_rejected"
    assert diagnostic["api_error_reason_source"] == "http_status"


def test_same_attempt_reason_persists_without_resolving_unknown_cost(tmp_path):
    from scripts import run_e1_experiment as runner
    from scripts.t7_ledger import CampaignLedger, LedgerError

    exc = failure({"id": GENERATION_ID, "error": {
        "code": 404, "message": "No endpoints found that support tool_choice=function"},
        "usage": {"cost": 0}})
    ledger = CampaignLedger(tmp_path / "ledger.sqlite3", campaign_id="t74_authored_fixture",
                            freeze_seal="f" * 64)
    ledger.register_stage("e1", [{"id": "e1_001", "request_sha256": "a" * 64,
                                 "reserve_usd": "0.0011"}], limit_usd="0.5", max_requests=204)
    ledger.claim("e1", "e1_001")
    destination = tmp_path / "receipt.json"
    digest = runner.write_private(destination, {"envelope": exc.receipt, "error": exc.code})
    with pytest.raises(LedgerError):
        runner.finish_receipt(ledger, "e1", "e1_001", exc.receipt, digest, exc.code)
    saved = json.loads(destination.read_text())
    assert saved["envelope"]["http_diagnostic"]["api_error_reason"] == "no_endpoints_parameter_support"
    assert "tool_choice=function" not in destination.read_text()
    summary = ledger.summary()
    assert summary["unknown_cost_count"] == 1 and summary["known_cost_usd"] == "0"
    assert summary["stopped"] is True
