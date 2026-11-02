"""Safe evidence from synthetic HTTP envelopes; never a live provider test."""
from copy import deepcopy
from email.message import Message
import http.client
import io
import json
import urllib.error

import pytest

from aria.bench.evaluation_journal import JournalError, _accounting
from aria.bench.request_control import BenchmarkRequestControlStopped
from aria.engine.policy.response_observability import (
    MAX_ERROR_BODY_BYTES, observe_http_error, observe_success,
    safe_generation_id, validate_observation,
)
from tests.test_bench_request_control import client, controller_fixture, envelope


GEN_ID = "gen-1789480804-sQdrvXdaT3WhCO3pBtai"
REQ_ID = "11111111-2222-4333-8444-555555555555"


def error(body, *, status=429, headers=None, stream=None):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    return urllib.error.HTTPError("https://PRIVATE.invalid", status, "PRIVATE-reason",
                                  headers or Message(), stream or io.BytesIO(body))


def _run(error_obj, **kwargs):
    control, _, starts = controller_fixture([error_obj], **kwargs)
    llm = client(retries=3)
    control.attach(llm)
    with pytest.raises(urllib.error.HTTPError) as caught:
        llm.complete([])
    assert caught.value is error_obj
    return control, llm, starts


def test_error_captures_only_finite_fields_and_explicit_zero_cost():
    headers = Message()
    headers["X-Request-ID"] = REQ_ID
    headers["Retry-After"] = "4"
    headers["X-RateLimit-Limit"] = "20"
    headers["X-RateLimit-Remaining"] = "0"
    headers["X-RateLimit-Reset"] = "1789480810"
    headers["Authorization"] = "Bearer PRIVATE-secret"
    body = {"id": GEN_ID, "request_id": REQ_ID,
            "usage": {"cost": 0, "prompt_tokens": 12, "private": "PRIVATE-usage"},
            "error": {"code": 429, "message": "PRIVATE-prompt api_key=PRIVATE-key",
                      "metadata": {"error_type": "rate_limit_exceeded", "provider_code": 429,
                                   "provider_name": "PRIVATE-provider", "raw": "PRIVATE-raw"}}}
    control, llm, starts = _run(error(body, headers=headers),
                               stop_on_rate_limit=True, max_observed_cost_usd=.04)
    account = control.accounting()
    row = account["attempts"][0]
    observation = row["response_observability"]
    assert row["provider_usage"] == {"cost": 0, "prompt_tokens": 12}
    assert row["retry_after_seconds"] == 4
    assert observation == {
        "version": "1.5", "envelope_kind": "http_error", "body_status": "parsed",
        "error_category": "rate_limited", "provider_error_code": 429,
        "provider_error_type": "rate_limit_exceeded", "provider_error_type_source": "error_metadata_type",
        "provider_code": 429, "rate_limit_limit": 20.0, "rate_limit_remaining": 0.0,
        "provider_metadata_present": True, "provider_name_status": "unrecognized",
        "rate_limit_reset": 1789480810.0, "request_id": REQ_ID,
        "request_id_source": "header_x_request_id", "request_id_status": "accepted",
        "generation_id": GEN_ID, "usage_source": "top_level_usage",
        "generation_id_candidate_status": "absent", "retry_after_status": "accepted",
        "retry_after_seconds": 4, "retry_after_source": "delta_seconds",
        "provider_raw_status": "not_inspected", "provider_native_code_status": "not_inspected",
        "limit_source_status": "absent", "router_error_type_status": "recognized",
        "router_error_type": "rate_limit_exceeded", "router_error_type_source": "error.metadata.error_type",
    }
    assert account["provider_cost_complete"] is True
    assert account["provider_observed_cost_usd"] == 0
    assert account["stopped_reason"] == "http_429_requires_new_registration"
    assert "PRIVATE" not in json.dumps(account)
    assert "PRIVATE" not in json.dumps(llm.last_diagnostic)
    validate_observation(observation)
    assert _accounting(account)[1]["unknown_cost_requests"] == 0
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1


@pytest.mark.parametrize("usage", [None, {}, {"cost": None}, {"cost": False}, {"cost": "0"},
                                   {"cost": -1}, {"cost": float("nan")}, {"cost": float("inf")}])
def test_missing_or_invalid_cost_stays_unknown_and_never_retries(usage):
    control, llm, starts = _run(error({"usage": usage, "error": {"code": 429}}),
                               max_observed_cost_usd=.04)
    account = control.accounting()
    assert account["provider_cost_complete"] is False
    assert account["provider_observed_cost_usd"] is None
    assert account["provider_cost_observed_requests"] == 0
    assert account["stopped_reason"] == "provider_cost_unknown_with_cost_guard"
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1


@pytest.mark.parametrize("body,status", [
    (b"", "empty"), (b"PRIVATE not json", "parse_failed"), (b"\xff", "decode_failed"),
    (b"[]", "non_object"), (b'0', "non_object"),
    (b'{"usage":{"cost":0,"cost":.1}}', "parse_failed"),
    (b'{"usage":{"cost":0,"cost":1}}', "parse_failed"),
    (b'{"usage":{"cost":0}}' + b" " * MAX_ERROR_BODY_BYTES, "oversized"),
    ((b"[" * 2000) + b"0" + (b"]" * 2000), "parse_failed"),
])
def test_unusable_body_does_not_produce_cost_or_leak_content(body, status):
    observation, usage = observe_http_error(error(body))
    assert observation["body_status"] == status
    assert usage == {}
    assert "PRIVATE" not in json.dumps(observation)
    validate_observation(observation)


def test_oversized_body_is_read_once_with_a_fixed_limit():
    class Bounded(io.BytesIO):
        calls = []
        def read(self, amount):
            self.calls.append(amount)
            return super().read(amount)
    stream = Bounded(b"PRIVATE" * MAX_ERROR_BODY_BYTES)
    observation, usage = observe_http_error(error(b"", stream=stream))
    assert stream.calls == [MAX_ERROR_BODY_BYTES + 1]
    assert stream.closed
    assert observation["body_status"] == "oversized" and usage == {}


@pytest.mark.parametrize("failure,kind", [
    (http.client.IncompleteRead(b"PRIVATE-partial"), "incomplete_read"),
    (TimeoutError("PRIVATE-timeout"), "timeout"),
    (ConnectionResetError("PRIVATE-reset"), "connection_error"),
    (RuntimeError("PRIVATE-unexpected"), "other_read_error"),
])
def test_error_body_read_failure_preserves_original_http_exception(failure, kind):
    class Broken(io.BytesIO):
        def read(self, amount):
            raise failure
    original = error(b"", stream=Broken())
    control, _, starts = _run(original, max_observed_cost_usd=.1)
    row = control.accounting()["attempts"][0]
    assert row["error_type"] == "http_error" and row["http_status"] == 429
    assert row["failure_phase"] == "connect"
    assert row["response_observability"]["body_read_error_kind"] == kind
    assert row["response_observability"]["body_status"] == "read_failed"
    assert row["provider_usage"] == {} and len(starts) == 1
    assert "PRIVATE" not in json.dumps(row)
    validate_observation(row["response_observability"])


def test_error_body_close_failure_does_not_replace_http_error():
    class BadClose(io.BytesIO):
        failed = False
        def close(self):
            if not self.failed:
                self.failed = True
                raise RuntimeError("PRIVATE-close")
            return super().close()
    original = error(b"", stream=BadClose(b'{"usage":{"cost":0}}'))
    control, _, _ = _run(original, stop_on_rate_limit=True)
    assert control.accounting()["attempts"][0]["response_observability"]["body_close_failed"] is True


def test_nested_provider_cost_is_not_promoted_to_endpoint_invoice():
    body = {"error": {"metadata": {"usage": {"cost": 0}, "cost": 0}}}
    observation, usage = observe_http_error(error(body))
    assert usage == {} and "usage_source" not in observation


def test_broken_header_reader_cannot_replace_http_error():
    class BrokenHeaders:
        def get(self, key):
            raise RuntimeError("PRIVATE-header")
    original = error({"error": {"code": 429}}, headers=BrokenHeaders())
    observation, usage = observe_http_error(original)
    assert observation["body_status"] == "parsed" and usage == {}
    assert observation["request_id_status"] == "absent"
    validate_observation(observation)


def test_broken_retry_after_reader_preserves_original_http_error_in_controller():
    class BrokenHeaders:
        def get(self, key):
            raise RuntimeError("PRIVATE-header")
    original = error({"usage": {"cost": 0}}, headers=BrokenHeaders())
    control, llm, starts = _run(original, stop_on_rate_limit=True)
    row = control.accounting()["attempts"][0]
    assert row["retry_after_seconds"] is None
    assert row["http_status"] == 429 and row["error_type"] == "http_error"
    assert row["provider_usage"]["cost"] == 0
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1
    assert "PRIVATE" not in json.dumps(control.accounting())


def test_error_message_and_nested_ids_never_become_classification_or_lookup_id():
    body = {"error": {"message": "insufficient credits PRIVATE-key",
                      "type": "PRIVATE-custom-type", "code": "PRIVATE-code",
                      "metadata": {"id": GEN_ID, "generation_id": GEN_ID,
                                   "provider_name": "PRIVATE-provider"}}}
    observation, _ = observe_http_error(error(body, status=503))
    assert observation["error_category"] == "server_http_error"
    assert "provider_error_type" not in observation
    assert "provider_error_code" not in observation
    assert "generation_id" not in observation
    assert "PRIVATE" not in json.dumps(observation)


@pytest.mark.parametrize("name", ["Alibaba", "Baidu", "DeepSeek", "Anthropic"])
def test_only_known_reported_provider_name_is_preserved_without_cause_inference(name):
    observation, usage = observe_http_error(error({"error": {"metadata": {"provider_name": name}}}))
    assert observation["provider_metadata_present"] is True
    assert observation["provider_name_status"] == "recognized"
    assert observation["provider_reported_name"] == name
    assert observation["error_category"] == "rate_limited"
    assert "origin_hint" not in observation and usage == {}
    validate_observation(observation)


@pytest.mark.parametrize("name", ["PRIVATE-provider", "alibaba", {"private": "data"}, 123])
def test_unknown_provider_name_is_presence_only(name):
    observation, _ = observe_http_error(error({"error": {"metadata": {"provider_name": name}}}))
    assert observation["provider_name_status"] == "unrecognized"
    assert "provider_reported_name" not in observation
    assert "PRIVATE" not in json.dumps(observation)
    validate_observation(observation)


@pytest.mark.parametrize("tamper", ["success_status", "success_error", "error_status", "error_kind",
                                   "category", "missing_usage", "missing_usage_source", "unparsed_cost"])
def test_new_observation_must_correlate_with_enclosing_attempt(tamper):
    if tamper.startswith("success"):
        control, _, _ = controller_fixture([envelope()])
        llm = client()
        control.attach(llm)
        llm.complete([])
    else:
        control, _, _ = _run(error({"usage": {"cost": 0}}), stop_on_rate_limit=True)
    saved = control.accounting()
    row = saved["attempts"][0]
    observation = row["response_observability"]
    if tamper == "success_status":
        row["http_status"] = 429
    elif tamper == "success_error":
        row["error_type"] = "http_error"
    elif tamper == "error_status":
        row["http_status"] = 200
    elif tamper == "error_kind":
        row["error_type"] = None
    elif tamper == "category":
        observation["error_category"] = "authentication"
    elif tamper == "missing_usage":
        row["provider_usage"] = {}
    elif tamper == "missing_usage_source":
        observation.pop("usage_source")
    else:
        observation["body_status"] = "empty"
        observation.pop("usage_source")
    with pytest.raises(JournalError):
        _accounting(saved)


@pytest.mark.parametrize("generation_id", [GEN_ID, "gen-3bhGkxlo4XFrqiabUM7NDtwDzWwG", "gen-1234567890"])
def test_documented_generation_ids_can_support_separate_invoice_lookup(generation_id):
    body = {**envelope(), "id": generation_id}
    control, _, _ = controller_fixture([body])
    llm = client()
    control.attach(llm)
    assert llm.complete([]) == "ok"
    observation = control.accounting()["attempts"][0]["response_observability"]
    assert observation["generation_id"] == generation_id
    assert safe_generation_id(generation_id) == generation_id
    validate_observation(observation)
    _accounting(control.accounting())


@pytest.mark.parametrize("unsafe", [None, [], {}, True, "gen-short", "gen-" + "x" * 81,
                                   "gen-sk-or-v1-PRIVATE", "gen-abcd/../../../PRIVATE",
                                   "gen-1234567890-PRIVATE\n", "Bearer PRIVATE-secret", "https://PRIVATE"])
def test_identifier_sanitizer_rejects_non_opaque_forms(unsafe):
    assert safe_generation_id(unsafe) is None
    observation = observe_success({"id": unsafe, "request_id": unsafe})
    assert "generation_id" not in observation and "request_id" not in observation
    assert "PRIVATE" not in json.dumps(observation)
    validate_observation(observation)


def test_conflicting_request_ids_are_not_selected_arbitrarily():
    headers = Message()
    headers["X-Request-ID"] = REQ_ID
    observation = observe_success({"request_id": "req-abcdefghijk"}, headers)
    assert observation["request_id_status"] == "conflicting"
    assert "request_id" not in observation
    validate_observation(observation)


def test_documented_timestamped_request_id_is_preserved():
    request_id = "req-1727282430-aBcDeFgHiJkLmNoPqRsT"
    observation = observe_success({"request_id": request_id})
    assert observation["request_id"] == request_id
    validate_observation(observation)


@pytest.mark.parametrize("tamper", ["extra_request", "reason", "false_policy", "invented_stop"])
def test_journal_rate_limit_policy_has_enforced_evidence(tamper):
    control, llm, _ = _run(error({"usage": {"cost": 0}}), stop_on_rate_limit=True)
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    saved = control.accounting()
    assert saved["stopped_reason"] == "http_429_requires_new_registration"
    if tamper == "extra_request":
        saved["attempts"].append(deepcopy(saved["attempts"][0]))
        saved["actual_network_requests"] += 1
        saved["provider_cost_observed_requests"] += 1
    elif tamper == "reason":
        saved["stopped_reason"] = None
    elif tamper == "false_policy":
        saved["stop_on_rate_limit"] = False
    else:
        saved["attempts"][0]["http_status"] = 503
    with pytest.raises(JournalError):
        _accounting(saved)


@pytest.mark.parametrize("field,value", [
    ("provider_error_code", "PRIVATE"), ("provider_error_code", True),
    ("provider_error_type", "PRIVATE"), ("body_status", {}),
    ("request_id_status", []), ("envelope_kind", {}), ("error_category", {}),
    ("generation_id", "gen-sk-or-v1-PRIVATE"), ("request_id_source", "PRIVATE"),
    ("provider_name", "PRIVATE"), ("usage_source", "PRIVATE"),
    ("rate_limit_remaining", float("nan")), ("rate_limit_reset", True),
    ("body_read_error_kind", "PRIVATE"),
])
def test_journal_rejects_untrusted_observation_fields(field, value):
    control, _, _ = _run(error({"usage": {"cost": 0}}), stop_on_rate_limit=True)
    saved = control.accounting()
    saved["attempts"][0]["response_observability"][field] = value
    with pytest.raises(JournalError):
        _accounting(saved)


def test_successful_prefix_resume_preserves_observability_and_explicit_policy():
    opts = {"stop_on_rate_limit": True, "max_observed_cost_usd": .04}
    first, _, _ = controller_fixture([{**envelope(), "id": GEN_ID}], **opts)
    llm = client()
    first.attach(llm)
    llm.complete([])
    saved = first.accounting()
    second, _, starts = controller_fixture([], **opts)
    second.restore_accounting(saved)
    assert second.accounting() == saved and starts == []
    saved["attempts"][0]["response_observability"]["generation_id"] = "gen-1234567890"
    assert second.accounting()["attempts"][0]["response_observability"]["generation_id"] == GEN_ID
    without, _, _ = controller_fixture([], max_observed_cost_usd=.04)
    with pytest.raises(ValueError):
        without.restore_accounting(deepcopy(first.accounting()))


def test_legacy_accounting_without_observability_still_restores():
    first, _, _ = controller_fixture([envelope()])
    llm = client()
    first.attach(llm)
    llm.complete([])
    saved = first.accounting()
    del saved["attempts"][0]["response_observability"]
    second, _, _ = controller_fixture([])
    second.restore_accounting(saved)
    assert second.accounting() == saved
