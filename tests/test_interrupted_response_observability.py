"""Strictly local interrupted-transport fixtures; no provider/API traffic."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
from email.message import Message
import http.client
import json
import urllib.error

import pytest

from aria.bench.evaluation_journal import JournalError, _accounting
from aria.bench.request_control import BenchmarkRequestControl, BenchmarkRequestControlStopped
from aria.engine.policy.response_observability import (
    capture_response_headers, observe_interrupted, observe_success,
    validate_attempt_observation, validate_observation,
)
from tests.test_bench_request_control import Clock, client, envelope


GEN_A = "gen-1789480804-abcdefgh12345678"
GEN_B = "gen-1789480807-ijklmnop12345678"
REQ_A = "55c10856-1a24-4e6e-90cc-60b9915f7510"
REQ_B = "req-1789480804-abcdefgh12345678"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr("socket.socket.connect", lambda *_a, **_k: pytest.fail("network forbidden"))
    stub_model_http(monkeypatch, lambda *_a, **_k: pytest.fail("network forbidden"))


class InterruptedResponse:
    status = 200

    def __init__(self, outcome, headers=None):
        self.outcome, self._headers = outcome, headers
        self.header_reads, self.body_reads = 0, 0

    @property
    def headers(self):
        self.header_reads += 1
        if self.body_reads:
            raise RuntimeError("PRIVATE-headers-after-read")
        return self._headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        self.body_reads += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def run_response(response, *, retries=3, guarded=True):
    clock, calls = Clock(), []

    def opener(_request, **_kwargs):
        calls.append(True)
        clock.now += .25
        if isinstance(response, BaseException):
            raise response
        return response

    control = BenchmarkRequestControl(
        min_interval_seconds=0, max_observed_cost_usd=.1 if guarded else None,
        monotonic=clock.monotonic, wall_time=clock.wall_time, sleep=clock.sleep,
        opener=opener,
    )
    llm = client(retries=retries)
    control.attach(llm)
    return control, llm, calls


def headers(*pairs):
    value = Message()
    for key, item in pairs:
        value[key] = item
    return value


def read_failure(*, header_values=None, partial=b"PRIVATE-partial"):
    response = InterruptedResponse(http.client.IncompleteRead(partial), header_values)
    control, llm, calls = run_response(response)
    with pytest.raises(http.client.IncompleteRead) as caught:
        llm.complete([])
    assert caught.value is response.outcome
    assert response.header_reads == response.body_reads == 1
    return control, llm, calls


def test_header_trace_is_captured_before_read_and_candidate_is_not_an_invoice():
    value = headers(("X-Request-ID", REQ_A), ("X-Generation-ID", GEN_A),
                    ("Authorization", "Bearer PRIVATE-key"))
    control, llm, calls = read_failure(header_values=value)
    row = control.accounting()["attempts"][0]
    observation = row["response_observability"]
    assert observation == {
        "version": "1.1", "envelope_kind": "interrupted", "body_status": "read_failed",
        "body_read_error_kind": "incomplete_read", "request_id_status": "accepted",
        "request_id_source": "header_x_request_id", "request_id": REQ_A,
        "generation_id_candidate_status": "accepted", "generation_id_candidate": GEN_A,
        "generation_id_candidate_source": "header_x_generation_id",
    }
    assert "generation_id" not in observation and row["provider_usage"] == {}
    assert control.accounting()["stopped_reason"] == "provider_cost_unknown_with_cost_guard"
    assert _accounting(control.accounting())[1]["unknown_cost_requests"] == 1
    assert llm.last_diagnostic["response_observability"] == observation
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(calls) == 1
    assert "PRIVATE" not in json.dumps(control.accounting())
    assert "PRIVATE" not in json.dumps(llm.last_diagnostic)


@pytest.mark.parametrize("partial", [
    b'{"id":"' + GEN_A.encode() + b'","usage":{"cost":0},"choices":[',
    b'{"id":"' + GEN_A.encode() + b'","id":"' + GEN_B.encode() + b'",',
    b'{"id":"' + GEN_A.encode() + b'","usage":{"cost":0}}',
    b'{"choices":[{"message":{"reasoning":"PRIVATE-thought","content":"PRIVATE"}}]',
    b"PRIVATE" * 100000,
], ids=["id-and-cost-prefix", "duplicate-id-prefix", "looks-complete", "private-reasoning", "oversized"])
def test_partial_bytes_are_never_parsed_even_if_they_look_complete(partial):
    control, _, calls = read_failure(partial=partial)
    row = control.accounting()["attempts"][0]
    assert row["provider_usage"] == {}
    assert row["response_observability"]["generation_id_candidate_status"] == "absent"
    assert "generation_id" not in row["response_observability"]
    assert "PRIVATE" not in json.dumps(row)
    assert len(calls) == 1
    _accounting(control.accounting())


@pytest.mark.parametrize("pairs,field,status", [
    ((("X-Request-ID", REQ_A), ("X-Request-ID", REQ_B)), "request_id", "rejected"),
    ((("X-Request-ID", REQ_A), ("X-Request-ID", REQ_A)), "request_id", "rejected"),
    ((("X-Request-ID", REQ_A), ("Request-ID", REQ_B)), "request_id", "conflicting"),
    ((("X-Request-ID", "Bearer PRIVATE-key"),), "request_id", "rejected"),
    ((("X-Generation-ID", GEN_A), ("X-Generation-ID", GEN_B)), "generation_id_candidate", "rejected"),
    ((("X-Generation-ID", GEN_A), ("X-OpenRouter-Generation-ID", GEN_B)), "generation_id_candidate", "conflicting"),
    ((("X-Generation-ID", "gen-sk-or-v1-PRIVATE"),), "generation_id_candidate", "rejected"),
    ((("X-Generation-ID", "https://PRIVATE.invalid"),), "generation_id_candidate", "rejected"),
])
def test_duplicate_conflicting_or_malicious_headers_never_become_an_identifier(pairs, field, status):
    control, _, _ = read_failure(header_values=headers(*pairs))
    observation = control.accounting()["attempts"][0]["response_observability"]
    assert observation[field + "_status"] == status
    assert field not in observation and "PRIVATE" not in json.dumps(observation)
    validate_observation(observation)


def test_header_getter_failure_does_not_replace_read_error():
    class BrokenHeaders:
        def get(self, _key):
            raise RuntimeError("PRIVATE-header")

    control, _, _ = read_failure(header_values=BrokenHeaders())
    observation = control.accounting()["attempts"][0]["response_observability"]
    assert observation["body_read_error_kind"] == "incomplete_read"
    assert observation["request_id_status"] == observation["generation_id_candidate_status"] == "rejected"


def test_inaccessible_conflicting_header_cannot_be_ignored_when_one_id_is_valid():
    class PartlyBrokenHeaders:
        def get(self, key):
            if key == "X-Request-ID":
                return REQ_A
            if key == "X-Generation-ID":
                return GEN_A
            raise RuntimeError("PRIVATE-header")

    control, _, _ = read_failure(header_values=PartlyBrokenHeaders())
    observation = control.accounting()["attempts"][0]["response_observability"]
    assert observation["request_id_status"] == observation["generation_id_candidate_status"] == "rejected"
    assert "request_id" not in observation and "generation_id_candidate" not in observation


def test_headers_property_failure_does_not_replace_read_error():
    class BrokenProperty(InterruptedResponse):
        @property
        def headers(self):
            raise RuntimeError("PRIVATE-property")

    response = BrokenProperty(TimeoutError("PRIVATE-timeout"))
    control, llm, calls = run_response(response)
    with pytest.raises(TimeoutError):
        llm.complete([])
    row = control.accounting()["attempts"][0]
    assert row["response_observability"]["body_read_error_kind"] == "timeout"
    assert len(calls) == 1 and row["provider_usage"] == {}
    _accounting(control.accounting())


@pytest.mark.parametrize("outcome,phase,status", [
    (b"\xffPRIVATE", "decode", "decode_failed"),
    (b'{"id":"gen-1789480804-abcdefgh12345678","usage":{"cost":0},', "parse", "parse_failed"),
])
def test_decode_parse_failure_preserves_headers_but_not_partial_usage(outcome, phase, status):
    response = InterruptedResponse(outcome, headers(("X-Request-ID", REQ_A)))
    control, llm, calls = run_response(response)
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        llm.complete([])
    row = control.accounting()["attempts"][0]
    assert row["failure_phase"] == phase and row["error_type"] == "invalid_json_response"
    assert row["response_observability"]["body_status"] == status
    assert row["response_observability"]["request_id"] == REQ_A
    assert row["provider_usage"] == {} and len(calls) == 1
    assert control.accounting()["stopped_reason"] == "provider_cost_unknown_with_cost_guard"
    _accounting(control.accounting())


@pytest.mark.parametrize("failure", [TimeoutError("PRIVATE"), urllib.error.URLError("PRIVATE"),
                                      ConnectionResetError("PRIVATE")])
def test_connect_failure_immediately_sets_sticky_unknown_cost_stop(failure):
    control, llm, calls = run_response(failure)
    with pytest.raises(type(failure)):
        llm.complete([])
    account = control.accounting()
    row = account["attempts"][0]
    assert row["http_status"] is None and row["failure_phase"] == "connect"
    assert "response_observability" not in row
    assert account["stopped_reason"] == "provider_cost_unknown_with_cost_guard"
    assert account["provider_cost_complete"] is False
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(calls) == 1


def test_no_guard_keeps_explicit_legacy_retry_policy():
    response = InterruptedResponse(TimeoutError("PRIVATE"))
    control, llm, calls = run_response(response, retries=1, guarded=False)
    with pytest.raises(TimeoutError):
        llm.complete([])
    assert len(calls) == 2 and control.accounting()["stopped_reason"] is None


@pytest.mark.parametrize("tamper", [
    "known_cost", "usage_source", "trusted_id", "body_request_id", "version", "phase",
    "status", "error", "read_kind", "output_limit", "candidate_source", "private_field",
])
def test_forged_interrupted_observation_cannot_relabel_cost_or_success(tamper):
    control, _, _ = read_failure(header_values=headers(("X-Generation-ID", GEN_A), ("X-Request-ID", REQ_A)))
    row = deepcopy(control.accounting()["attempts"][0])
    observation = row["response_observability"]
    if tamper == "known_cost":
        row["provider_usage"] = {"cost": 0}
    elif tamper == "usage_source":
        observation["usage_source"] = "top_level_usage"
    elif tamper == "trusted_id":
        observation["generation_id"] = GEN_A
    elif tamper == "body_request_id":
        observation["request_id_source"] = "body_request_id"
    elif tamper == "version":
        observation["version"] = "1.0"
    elif tamper == "phase":
        row["failure_phase"] = "connect"
    elif tamper == "status":
        row["http_status"] = 429
    elif tamper == "error":
        row["error_type"] = None
    elif tamper == "read_kind":
        row["network_error_kind"] = "timeout"
    elif tamper == "output_limit":
        row["output_limit_reported"] = False
    elif tamper == "candidate_source":
        observation["generation_id_candidate_source"] = "body_id"
    else:
        observation["raw_body"] = "PRIVATE"
    with pytest.raises(ValueError):
        validate_attempt_observation(row)
    account = control.accounting()
    account["attempts"][0] = row
    with pytest.raises(JournalError):
        _accounting(account)


def test_success_uses_pre_read_headers_and_keeps_version_one_contract():
    response = InterruptedResponse(json.dumps(envelope()).encode(), headers(("X-Request-ID", REQ_A)))
    control, llm, calls = run_response(response)
    assert llm.complete([]) == "ok"
    observation = control.accounting()["attempts"][0]["response_observability"]
    assert observation["version"] == "1.0" and observation["request_id"] == REQ_A
    assert "generation_id_candidate_status" not in observation
    assert response.header_reads == 1 and len(calls) == 1
    _accounting(control.accounting())


def test_success_body_and_preread_request_id_conflict_is_not_selected():
    snapshot = capture_response_headers(InterruptedResponse(b"", headers(("X-Request-ID", REQ_A))))
    observation = observe_success({"request_id": REQ_B}, header_observation=snapshot)
    assert observation["request_id_status"] == "conflicting" and "request_id" not in observation
    validate_observation(observation)


def test_invalid_phase_rejected_without_body_access():
    snapshot = capture_response_headers(InterruptedResponse(b""))
    with pytest.raises(ValueError, match="invalid_interrupted_response_phase"):
        observe_interrupted(snapshot, phase="connect", error=TimeoutError())


def test_unknown_prefix_cannot_restore_a_live_controller():
    control, _, _ = read_failure(header_values=headers(("X-Generation-ID", GEN_A)))
    fresh, _, calls = run_response(InterruptedResponse(json.dumps(envelope()).encode()))
    with pytest.raises(ValueError, match="accounting_restore_unsafe_or_stopped_prefix"):
        fresh.restore_accounting(control.accounting())
    assert calls == []
