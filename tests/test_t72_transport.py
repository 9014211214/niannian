"""T7.2 diagnostics use authored HTTP responses, never the network or .env."""
import base64
from copy import deepcopy
from email.message import Message
import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, build_opener
from urllib.response import addinfourl

import pytest

from scripts import t7_transport as frozen
from scripts import t72_transport as transport
from tests.test_t7_transport import KEY, Opener, Response, payload, reply


GENERATION_ID = "gen-1790777949000-AbCdEfGh12345678"
REQUEST_ID = "12345678-1234-4abc-8abc-1234567890ab"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(frozen, "_opener", lambda: pytest.fail("network forbidden"))


def failure(response=None, *, error=None, key=KEY, timeout=120):
    opener = Opener(response, error=error)
    with pytest.raises(frozen.TransportError) as caught:
        transport.OpenRouterTransport(key, opener=opener, timeout=timeout).request(payload())
    assert len(opener.calls) == 1
    assert caught.value.outcome_unknown is True
    if response is not None:
        assert response.closed
    return caught.value


def test_default_closed_before_constructing_network():
    with pytest.raises(frozen.TransportError, match="network_disabled") as caught:
        transport.OpenRouterTransport(KEY).request(payload())
    assert caught.value.outcome_unknown is False


@pytest.mark.parametrize("value", ["", "not ascii 密钥", "has space", None])
def test_invalid_credential_never_opens(value):
    opener = Opener()
    with pytest.raises(frozen.TransportError, match="invalid_api_key"):
        transport.OpenRouterTransport(value, opener=opener).request(payload())
    assert not opener.calls


def test_unchanged_wire_and_success_semantics():
    request = payload() | {"parallel_tool_calls": False}
    original = deepcopy(request)
    opener = Opener(Response(reply()))
    result = transport.OpenRouterTransport(KEY, opener=opener).request(request)
    assert request == original
    sent, timeout = opener.calls[0]
    assert sent.data == frozen._validate_payload(original)
    assert sent.method == "POST" and sent.full_url == frozen.ENDPOINT and timeout == 120
    assert sent.get_header("Authorization") == "Bearer " + KEY
    assert result == frozen.sanitize_response(reply(), model=request["model"])
    assert len(opener.calls) == 1 and opener.response.closed


def test_invalid_frozen_payload_never_opens():
    opener = Opener()
    with pytest.raises(frozen.TransportError, match="unfrozen_route"):
        transport.OpenRouterTransport(KEY, opener=opener).request(payload() | {"stream": True})
    assert not opener.calls


@pytest.mark.parametrize("status,category", [(400, "bad_request"), (401, "authentication_error"),
    (402, "payment_required"), (403, "forbidden"), (404, "not_found"),
    (408, "request_timeout"), (422, "unprocessable_entity"), (429, "rate_limited"),
    (502, "server_error"), (418, "other_http_status")])
def test_http_error_retains_status_and_safe_fields_but_not_cost_or_messages(status, category):
    raw = {"id": GENERATION_ID, "error": {"code": status, "message": "private sentinel",
           "metadata": {"raw": "provider trace", "reasoning": "hidden sentinel"}},
           "usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}
    headers = Message()
    headers["X-Request-ID"] = REQUEST_ID
    headers["Retry-After"] = "12"
    headers["Authorization"] = "Bearer " + KEY
    stream = io.BytesIO(json.dumps(raw).encode())
    exc = failure(error=HTTPError(transport.ENDPOINT, status, "private sentinel", headers, stream))
    assert exc.code == "http_status_error"
    assert exc.receipt == {"http_diagnostic": {
        "http_status": status, "http_category": category, "body_status": "json",
        "request_id": REQUEST_ID, "generation_id": GENERATION_ID,
        "retry_after_seconds": 12, "api_error_code": status}}
    assert stream.closed
    assert all(s not in json.dumps(exc.receipt) for s in
               [KEY, "private sentinel", "provider trace", "hidden sentinel", "usage", "cost"])


def test_returned_error_response_matches_raised_http_error():
    exc = failure(Response({"error": {"code": 404, "message": "unknown private body"}}, status=404))
    assert exc.receipt["http_diagnostic"] == {
        "http_status": 404, "http_category": "not_found", "body_status": "json", "api_error_code": 404}


def test_generation_response_header_survives_missing_or_unreadable_body_id():
    exc = failure(Response(b"unreadable error", status=503,
                          headers={"X-Generation-Id": GENERATION_ID}))
    assert exc.receipt["http_diagnostic"] == {"http_status": 503,
        "http_category": "server_error", "body_status": "invalid_response_json",
        "header_generation_id": GENERATION_ID}


@pytest.mark.parametrize("status", [200, 404])
@pytest.mark.parametrize("body_id,conflict", [(GENERATION_ID, False),
    ("gen-1790777949001-ZyXwVuTs98765432", True)])
def test_generation_header_and_body_identifiers_keep_provenance_and_conflicts(status, body_id, conflict):
    source = reply() | {"id": body_id, "reasoning": "hidden sentinel"}
    exc = failure(Response(source, status=status, headers={"X-Generation-Id": GENERATION_ID}))
    diagnostic = exc.receipt["http_diagnostic"]
    assert diagnostic["header_generation_id"] == GENERATION_ID
    assert diagnostic["generation_id"] == body_id
    assert diagnostic.get("generation_id_conflict", False) is conflict


@pytest.mark.parametrize("body,headers", [(b"{", {}),
    (b"{}", {"Content-Length": "10"}),
    (json.dumps(reply() | {"debug": KEY}).encode(), {})])
def test_http_200_failures_retain_safe_header_for_reconciliation(body, headers):
    exc = failure(Response(body, headers=headers | {
        "X-Generation-Id": GENERATION_ID, "X-Request-ID": REQUEST_ID}))
    assert exc.receipt["http_diagnostic"]["header_generation_id"] == GENERATION_ID
    assert exc.receipt["http_diagnostic"]["request_id"] == REQUEST_ID
    assert exc.receipt["http_diagnostic"]["http_status"] == 200
    assert KEY not in json.dumps(exc.receipt)


@pytest.mark.parametrize("value", [KEY, "gen-1790777949000-unittestkey",
    "gen-1790777949000-" + base64.b64encode(KEY.encode()).decode().rstrip("="),
    GENERATION_ID + "\n", "gen-1790777949000-%61%62%63%64%65%66%67%68"])
def test_generation_header_rejects_malicious_or_secret_identifiers(value):
    exc = failure(Response({}, status=404, headers={"X-Generation-Id": value}))
    assert "header_generation_id" not in exc.receipt["http_diagnostic"]


@pytest.mark.parametrize("value", ["NaN", {}, True, -1, 600, "404", "private sentinel"])
def test_arbitrary_remote_error_code_not_persisted(value):
    exc = failure(Response({"error": {"code": value}}, status=400))
    assert "api_error_code" not in exc.receipt["http_diagnostic"]


@pytest.mark.parametrize("body,headers,body_status", [
    (b"{", {}, "invalid_response_json"),
    (b'{"error":1,"error":2}', {}, "invalid_response_json"),
    (b'{"error":NaN}', {}, "invalid_response_json"),
    (b"\xff", {}, "invalid_response_json"),
    (b"{}", {"Content-Length": "9"}, "response_truncated"),
    (b"{}", {"Content-Length": "-1"}, "invalid_response_length"),
    (b"{}", {"Content-Length": "9999999999999999999999"}, "invalid_response_length"),
    (b"{}", {"Content-Length": str(transport.MAX_ERROR_BYTES + 1)}, "response_too_large"),
    (b"x" * (transport.MAX_ERROR_BYTES + 1), {}, "response_too_large"),
    (b"{}", {"Content-Encoding": "gzip"}, "encoded_response_rejected"),
], ids=["invalid-json", "duplicate-json", "nonfinite", "non-utf8", "truncated",
        "negative-length", "length-too-long", "declared-oversize", "actual-oversize", "gzip"])
def test_error_body_failures_keep_original_http_status(body, headers, body_status):
    exc = failure(Response(body, status=503, headers=headers))
    assert exc.code == "http_status_error"
    assert exc.receipt == {"http_diagnostic": {"http_status": 503,
        "http_category": "server_error", "body_status": body_status}}


def test_error_slow_drip_respects_absolute_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])

    class Slow(Response):
        def read1(self, size):
            clock[0] += 0.6
            return self.body.read(1)

    exc = failure(Slow(b"private sentinel", status=429), timeout=1)
    assert clock[0] == 1.2
    assert exc.receipt["http_diagnostic"]["body_status"] == "response_timeout"


def test_http_error_nested_socket_gets_remaining_deadline():
    seen = []
    sock = SimpleNamespace(settimeout=seen.append)
    response = SimpleNamespace(fp=SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock))))
    transport._set_read_timeout(response, 0.75)
    assert seen == [0.75]


@pytest.mark.parametrize("exception,code", [(TimeoutError("private sentinel"), "response_timeout"),
    (URLError("Bearer " + KEY), "transport_failure"),
    (RuntimeError("Bearer " + KEY), "transport_failure")])
def test_pre_response_errors_do_not_invent_http_status(exception, code):
    exc = failure(error=exception)
    assert exc.code == code and exc.receipt == {} and str(exc) == code


def test_read_exception_keeps_status_and_suppresses_exception_text():
    class Broken(Response):
        def read1(self, size):
            raise RuntimeError("Bearer " + KEY)

    exc = failure(Broken(b"", status=500))
    assert exc.receipt == {"http_diagnostic": {"http_status": 500,
        "http_category": "server_error", "body_status": "transport_failure"}}


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_error_is_not_followed_or_read(status):
    class Unread(Response):
        def read1(self, size):
            pytest.fail("redirect body must not be read")

    exc = failure(Unread(b"private sentinel", status=status,
                          headers={"Location": "https://invalid.test/" + KEY}))
    assert exc.code == "redirect_rejected"
    assert exc.receipt == {"http_diagnostic": {"http_status": status,
        "http_category": "redirect", "body_status": "not_read_redirect"}}


def test_real_urllib_redirect_handler_does_not_forward_auth():
    seen = []

    class AuthoredHTTPS(HTTPSHandler):
        def https_open(self, request):
            seen.append(request)
            headers = Message()
            headers["Location"] = "https://authored.invalid/stolen"
            response = addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
            response.msg = "Found"
            return response

    opener = build_opener(ProxyHandler({}), frozen._NoRedirect(), AuthoredHTTPS())
    with pytest.raises(frozen.TransportError, match="redirect_rejected") as caught:
        transport.OpenRouterTransport(KEY, opener=opener.open).request(payload())
    assert len(seen) == 1 and seen[0].full_url == frozen.ENDPOINT
    assert caught.value.receipt["http_diagnostic"]["http_status"] == 302


def test_unexpected_response_url_is_rejected():
    exc = failure(Response(reply(), url="https://authored.invalid/stolen"))
    assert exc.code == "redirect_rejected"
    assert exc.receipt["http_diagnostic"]["http_status"] == 200


@pytest.mark.parametrize("value", [KEY, "private-person@example.test", "private\nheader",
    "12345678-1234-4abc-8abc-1234567890ab\n", "x" * 1000,
    "%31%32%33%34%35%36%37%38-1234-4abc-8abc-1234567890ab"])
def test_malicious_header_identifiers_are_omitted(value):
    exc = failure(Response({}, status=400, headers={"X-Request-ID": value}))
    assert "request_id" not in exc.receipt["http_diagnostic"]


@pytest.mark.parametrize("value", ["Fri, 1 Jan 2027", "-1", "86401", "999999", "1.5", KEY])
def test_only_bounded_numeric_retry_after_is_stored(value):
    exc = failure(Response({}, status=429, headers={"Retry-After": value}))
    assert "retry_after_seconds" not in exc.receipt["http_diagnostic"]


@pytest.mark.parametrize("encode", [False, True])
def test_full_key_echo_in_json_never_retains_remote_fields(encode):
    body = json.dumps({"id": GENERATION_ID, "error": {"message": KEY}})
    if encode:
        body = body.replace(KEY, "".join("\\u%04x" % ord(c) for c in KEY))
    exc = failure(Response(body.encode(), status=401))
    assert exc.receipt == {"http_diagnostic": {"http_status": 401,
        "http_category": "authentication_error", "body_status": "credential_echo_rejected"}}


@pytest.mark.parametrize("suffix", ["unit-test-key", "unittestkey",
    base64.b64encode(KEY.encode()).decode().rstrip("=")])
def test_key_fragments_and_encoded_echo_cannot_be_saved_in_generation_id(suffix):
    exc = failure(Response({"id": "gen-1790777949000-" + suffix}, status=404))
    assert "generation_id" not in exc.receipt["http_diagnostic"]


def test_key_fragment_cannot_be_saved_as_uuid_request_id():
    key = "synthetic-credential-1234567812344abc8abc1234567890abcdef"
    exc = failure(Response({}, status=404, headers={"X-Request-ID": REQUEST_ID}), key=key)
    assert "request_id" not in exc.receipt["http_diagnostic"]


@pytest.mark.parametrize("value", ["gen-authored", "gen-1790777949000-1234567",
    "gen-1790777949000-" + "a" * 65, "private sentinel", REQUEST_ID])
def test_body_generation_id_requires_specific_bounded_format(value):
    exc = failure(Response({"id": value}, status=400))
    assert "generation_id" not in exc.receipt["http_diagnostic"]


def test_escaped_success_key_echo_is_rejected_without_output_or_cost():
    source = reply()
    source["choices"][0]["message"]["content"] = KEY
    body = json.dumps(source).replace(KEY, "".join("\\u%04x" % ord(c) for c in KEY)).encode()
    exc = failure(Response(body))
    assert exc.code == "credential_echo_rejected"
    assert exc.receipt == {"http_diagnostic": {"http_status": 200, "http_category": "ok"}}


def test_success_semantic_violation_reuses_safe_receipt_and_known_usage():
    source = reply() | {"reasoning": "hidden sentinel"}
    exc = failure(Response(source))
    assert exc.code == "reasoning_not_disabled"
    expected = {"http_status": 200, "http_category": "ok"}
    assert exc.receipt.pop("http_diagnostic") == expected
    with pytest.raises(frozen.TransportError) as original:
        frozen.sanitize_response(source, model=payload()["model"])
    assert exc.receipt == original.value.receipt
    assert "hidden sentinel" not in json.dumps(exc.receipt)


def test_http_diagnostic_persists_with_frozen_runner_and_unknown_ledger_cost(tmp_path):
    from scripts import run_e1_experiment as runner
    from scripts.t7_ledger import CampaignLedger, LedgerError

    exc = failure(Response({"id": GENERATION_ID, "usage": {"cost": 0}}, status=404))
    ledger = CampaignLedger(tmp_path / "ledger.sqlite3", campaign_id="t72_authored_fixture",
                            freeze_seal="f" * 64)
    ledger.register_stage("e1", [{"id": "e1_001", "request_sha256": "a" * 64,
                                 "reserve_usd": "0.0011"}], limit_usd="0.5", max_requests=204)
    ledger.claim("e1", "e1_001")
    destination = tmp_path / "receipt.json"
    digest = runner.write_private(destination, {"envelope": exc.receipt, "error": exc.code})
    with pytest.raises(LedgerError):
        runner.finish_receipt(ledger, "e1", "e1_001", exc.receipt, digest, exc.code)
    assert json.loads(destination.read_text())["envelope"] == exc.receipt
    summary = ledger.summary()
    assert summary["unknown_cost_count"] == 1 and summary["known_cost_usd"] == "0"
    assert summary["stopped"] is True
    with pytest.raises(LedgerError):
        ledger.claim("e1", "e1_001")
