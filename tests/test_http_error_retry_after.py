"""Finite HTTP error metadata only; no real network or billing inference."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
from datetime import datetime, timezone
from email.message import Message
import io
import json

import pytest

from aria.bench.evaluation_journal import _accounting
from aria.bench.request_control import BenchmarkRequestControlStopped
from aria.engine.policy.response_observability import (
    MAX_RETRY_AFTER_SECONDS, observe_http_error, validate_observation,
)
from tests.test_http_error_observability import GEN_ID, error, _run


DATE = "Sat, 03 Oct 2026 12:30:00 GMT"
TARGET = "Sat, 03 Oct 2026 12:30:30 GMT"
EPOCH = int(datetime(2026, 10, 3, 12, 30, tzinfo=timezone.utc).timestamp())


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr("socket.socket.connect", lambda *_a, **_k: pytest.fail("network forbidden"))
    stub_model_http(monkeypatch, lambda *_a, **_k: pytest.fail("network forbidden"))


def headers(*pairs):
    result = Message()
    for key, value in pairs:
        result[key] = value
    return result


def observe(*pairs, observed_at=EPOCH):
    result, usage = observe_http_error(error({"error": {"message": "PRIVATE"}}, headers=headers(*pairs)),
                                      observed_at=observed_at)
    assert usage == {} and "PRIVATE" not in json.dumps(result)
    validate_observation(result)
    return result


@pytest.mark.parametrize("raw,seconds", [("0", 0), ("4", 4), ("  15\t", 15), ("000030", 30),
                                          (str(MAX_RETRY_AFTER_SECONDS), MAX_RETRY_AFTER_SECONDS)])
def test_numeric_wait_is_bounded_metadata_without_billing_or_raw_headers(raw, seconds):
    value = observe(("Retry-After", raw), ("Authorization", "Bearer PRIVATE-key"))
    assert value["version"] == "1.3"
    assert value["retry_after_status"] == "accepted"
    assert value["retry_after_seconds"] == seconds
    assert value["retry_after_source"] == "delta_seconds"
    assert not any(k.startswith("retry_after_reference") for k in value)
    assert "generation_id" not in value and "usage_source" not in value


@pytest.mark.parametrize("raw", ["", "nan", "inf", "-1", "+5", "5.0", "1e2", "１２", "4\n", "4\r\n",
                                 "3,4", "PRIVATE", "https://PRIVATE.invalid", "9" * 129,
                                 str(MAX_RETRY_AFTER_SECONDS + 1), "999999999999999999999"])
def test_invalid_numeric_or_free_form_wait_is_rejected(raw):
    value = observe(("Retry-After", raw))
    assert value["retry_after_status"] == "rejected"
    assert not set(value) & {"retry_after_seconds", "retry_after_source", "retry_after_reference"}


@pytest.mark.parametrize("second", ["4", "5", TARGET])
def test_duplicate_retry_after_is_rejected_even_when_equal(second):
    value = observe(("Retry-After", "4"), ("Retry-After", second))
    assert value["retry_after_status"] == "rejected" and "retry_after_seconds" not in value


@pytest.mark.parametrize("target", [TARGET, "Sat Oct  3 12:30:30 2026"])
def test_http_date_uses_response_date_with_auditable_numeric_reference(target):
    value = observe(("Retry-After", target), ("Date", DATE), observed_at=EPOCH + 1000)
    assert value["retry_after_status"] == "accepted" and value["retry_after_seconds"] == 30
    assert value["retry_after_source"] == "http_date"
    assert value["retry_after_reference"] == "response_date"
    assert value["retry_after_reference_epoch_seconds"] == EPOCH
    assert value["retry_after_target_epoch_seconds"] == EPOCH + 30
    assert "Oct" not in json.dumps(value)


def test_http_date_missing_response_date_uses_explicit_observation_time():
    value = observe(("Retry-After", TARGET), observed_at=EPOCH + .5)
    assert value["retry_after_reference"] == "observed_at"
    assert value["retry_after_reference_epoch_seconds"] == EPOCH + .5
    assert value["retry_after_seconds"] == 29.5


def test_default_observation_time_is_captured_and_saved_for_date_conversion(monkeypatch):
    monkeypatch.setattr("aria.engine.policy.response_observability.time.time", lambda: EPOCH + .25)
    value = observe(("Retry-After", TARGET), observed_at=None)
    assert value["retry_after_reference_epoch_seconds"] == EPOCH + .25
    assert value["retry_after_seconds"] == 29.75


def test_past_http_date_is_recorded_as_zero_wait_and_never_zero_cost():
    value = observe(("Retry-After", DATE), observed_at=EPOCH + 10)
    assert value["retry_after_seconds"] == 0 and "usage_source" not in value


@pytest.mark.parametrize("pairs", [
    (("Date", DATE), ("Date", DATE)), (("Date", DATE), ("Date", TARGET)),
    (("Date", "PRIVATE"),), (("Date", "Sun, 03 Oct 2026 12:30:00 GMT"),),
])
def test_invalid_or_duplicate_date_cannot_silently_fall_back_to_local_clock(pairs):
    value = observe(("Retry-After", TARGET), *pairs)
    assert value["retry_after_status"] == "rejected"


@pytest.mark.parametrize("target", ["Sun, 03 Oct 2026 12:30:30 GMT", TARGET + " PRIVATE", TARGET + "\r\n",
                                    "2026-10-03T12:30:30Z", "Sat, 03 Oct 2026 12:30:30 +0000",
                                    "Sat, 03 Oct 2026 12:30:60 GMT", "Tue, 03 Oct 2028 12:30:30 GMT",
                                    "Saturday, 03-Oct-26 12:30:30 GMT"])
def test_http_date_must_be_complete_valid_and_within_wait_bound(target):
    value = observe(("Retry-After", target))
    assert value["retry_after_status"] == "rejected"


@pytest.mark.parametrize("observed_at", [-1, True, "PRIVATE", float("nan"), float("inf"), 10 ** 1000])
def test_invalid_observation_clock_cannot_create_a_wait(observed_at):
    value = observe(("Retry-After", TARGET), observed_at=observed_at)
    assert value["retry_after_status"] == "rejected"


def test_clock_failure_does_not_replace_http_error(monkeypatch):
    def unavailable():
        raise RuntimeError("PRIVATE-clock")
    monkeypatch.setattr("aria.engine.policy.response_observability.time.time", unavailable)
    value = observe(("Retry-After", TARGET), observed_at=None)
    assert value["retry_after_status"] == "rejected"


def test_header_generation_candidate_is_separate_from_body_id_and_usage():
    body_id = "gen-1789480807-ijklmnop12345678"
    value, usage = observe_http_error(error({"id": body_id}, headers=headers(("X-Generation-ID", GEN_ID))))
    assert value["generation_id"] == body_id
    assert value["generation_id_candidate"] == GEN_ID
    assert value["generation_id_candidate_source"] == "header_x_generation_id"
    assert usage == {} and "usage_source" not in value
    validate_observation(value)


@pytest.mark.parametrize("pairs,status", [
    ((("X-Generation-ID", GEN_ID),), "accepted"),
    ((("X-OpenRouter-Generation-ID", GEN_ID),), "accepted"),
    ((("X-Generation-ID", GEN_ID), ("X-OpenRouter-Generation-ID", GEN_ID)), "accepted"),
    ((("X-Generation-ID", GEN_ID), ("X-Generation-ID", GEN_ID)), "rejected"),
    ((("X-Generation-ID", GEN_ID), ("X-OpenRouter-Generation-ID", "gen-abcdefgh98765432")), "conflicting"),
    ((("X-Generation-ID", "gen-sk-or-v1-PRIVATE"),), "rejected"),
    ((("X-Generation-ID", "https://PRIVATE.invalid"),), "rejected"),
])
def test_http_generation_candidates_reject_ambiguity_and_unsafe_values(pairs, status):
    value = observe(*pairs)
    assert value["generation_id_candidate_status"] == status
    assert ("generation_id_candidate" in value) is (status == "accepted")
    assert "generation_id" not in value


def test_header_metadata_is_snapshotted_before_body_close():
    values = headers(("Retry-After", TARGET), ("Date", DATE), ("X-Generation-ID", GEN_ID))
    class Closing(io.BytesIO):
        def close(self):
            for key in ("Retry-After", "Date", "X-Generation-ID"):
                del values[key]
            return super().close()
    value, usage = observe_http_error(error(b"", headers=values, stream=Closing(b"{}")), observed_at=EPOCH)
    assert value["retry_after_seconds"] == 30 and value["generation_id_candidate"] == GEN_ID
    assert usage == {}
    validate_observation(value)


def test_error_body_read_failure_keeps_wait_and_candidate_without_parsing_partial_body():
    class Broken(io.BytesIO):
        def read(self, _amount):
            raise TimeoutError("PRIVATE")
    value, usage = observe_http_error(error(b"", headers=headers(("Retry-After", "7"), ("X-Generation-ID", GEN_ID)),
                                          stream=Broken()))
    assert value["body_status"] == "read_failed" and value["retry_after_seconds"] == 7
    assert value["generation_id_candidate"] == GEN_ID and usage == {}
    assert "PRIVATE" not in json.dumps(value)
    validate_observation(value)


def test_broken_or_nonstandard_header_reader_does_not_replace_http_error():
    class Broken:
        def get(self, _key):
            raise RuntimeError("PRIVATE")
    value, usage = observe_http_error(error({}, headers=Broken()))
    assert value["retry_after_status"] == value["generation_id_candidate_status"] == "rejected"
    assert usage == {} and "PRIVATE" not in json.dumps(value)
    validate_observation(value)


@pytest.mark.parametrize("values", [["4", "4"], "4", [None], [True], {"PRIVATE": "4"}])
def test_nonstandard_multi_header_accessor_is_not_trusted(values):
    class Odd:
        def get_all(self, key):
            return values if key == "Retry-After" else None
        def get(self, _key):
            return None
    value, _ = observe_http_error(error({}, headers=Odd()))
    assert value["retry_after_status"] == "rejected"
    validate_observation(value)


def test_wait_hint_and_candidate_do_not_clear_unknown_cost_or_allow_a_second_request():
    control, llm, starts = _run(error({}, headers=headers(("Retry-After", "0"), ("X-Generation-ID", GEN_ID))),
                               max_observed_cost_usd=.1)
    row = control.accounting()["attempts"][0]
    assert row["response_observability"]["generation_id_candidate"] == GEN_ID
    assert row["provider_usage"] == {} and _accounting(control.accounting())[1]["unknown_cost_requests"] == 1
    assert control.accounting()["stopped_reason"] == "provider_cost_unknown_with_cost_guard"
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1


def test_legacy_error_and_interrupted_versions_remain_supported():
    current = observe(("Retry-After", "4"))
    prior_error = {k: v for k, v in current.items() if not k.startswith(("provider_native_", "provider_raw_"))}
    prior_error["version"] = "1.2"
    before = deepcopy(prior_error)
    validate_observation(prior_error)
    assert prior_error == before
    old = {k: v for k, v in prior_error.items() if not k.startswith(("retry_after_", "generation_id_candidate"))}
    old["version"] = "1.0"
    before = deepcopy(old)
    validate_observation(old)
    assert old == before
    interrupted = {"version": "1.1", "envelope_kind": "interrupted", "body_status": "read_failed",
                   "body_read_error_kind": "timeout", "request_id_status": "absent",
                   "generation_id_candidate_status": "absent"}
    validate_observation(interrupted)


@pytest.mark.parametrize("field,value", [
    ("retry_after_status", "PRIVATE"), ("retry_after_status", {}),
    ("retry_after_seconds", True), ("retry_after_seconds", float("inf")),
    ("retry_after_seconds", -1), ("retry_after_seconds", MAX_RETRY_AFTER_SECONDS + 1),
    ("retry_after_seconds", 31), ("retry_after_source", "PRIVATE"),
    ("retry_after_reference", "PRIVATE"), ("retry_after_reference_epoch_seconds", "PRIVATE"),
    ("retry_after_reference_epoch_seconds", True), ("retry_after_reference_epoch_seconds", EPOCH + .5),
    ("retry_after_target_epoch_seconds", float(EPOCH + 30)), ("retry_after_target_epoch_seconds", -1),
    ("generation_id_candidate_status", "PRIVATE"), ("generation_id_candidate_source", "body_id"),
    ("raw_header", "PRIVATE"), ("version", "1.0"), ("version", "1.1"),
])
def test_validator_rejects_tampered_new_fields_and_cross_version_data(field, value):
    result = observe(("Retry-After", TARGET), ("Date", DATE), ("X-Generation-ID", GEN_ID))
    result[field] = value
    with pytest.raises(ValueError, match="invalid_response_observability"):
        validate_observation(result)


@pytest.mark.parametrize("key", ["retry_after_status", "retry_after_seconds", "retry_after_source",
                                "retry_after_reference", "retry_after_reference_epoch_seconds",
                                "retry_after_target_epoch_seconds", "generation_id_candidate_status"])
def test_validator_requires_all_fields_for_the_observed_conversion(key):
    result = observe(("Retry-After", TARGET), ("Date", DATE))
    del result[key]
    with pytest.raises(ValueError, match="invalid_response_observability"):
        validate_observation(result)


def test_rejected_or_absent_wait_cannot_carry_a_numeric_value():
    result = observe()
    result["retry_after_seconds"] = 0
    with pytest.raises(ValueError, match="invalid_response_observability"):
        validate_observation(result)


def test_delta_seconds_must_be_integer_and_cannot_claim_a_clock_reference():
    for tamper in ({"retry_after_seconds": 4.0}, {"retry_after_reference": "observed_at"}):
        result = observe(("Retry-After", "4"))
        result.update(tamper)
        with pytest.raises(ValueError, match="invalid_response_observability"):
            validate_observation(result)
