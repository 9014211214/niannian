"""Bounded HTTP-error evidence through real accounting; all responses synthetic."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
import urllib.error

import pytest

from aria.engine.policy.e1br_calls import no_content_failure, validate_e1br_records
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.response_observability import (
    MAX_ERROR_BODY_BYTES, observe_http_error, validate_error_envelope_audit,
)
from tests.test_e1br_calls import call, client, guard, receipt, stubs


PRIVATE = "synthetic-private-value-never-in-audit"


def envelope(**extras):
    return {"error": {"code": 429, "message": PRIVATE,
            "metadata": {"provider_name": "Alibaba", "raw": PRIVATE}}, **extras}


def error(data, *, status=429, stream=None, headers=None):
    raw = json.dumps(data).encode() if type(data) is not bytes else data
    return urllib.error.HTTPError("https://fixture.invalid", status, PRIVATE, headers or {}, stream or io.BytesIO(raw))


def inspect(exc):
    proof, audit = {}, {}
    observation, usage = observe_http_error(exc, no_content_evidence=proof, envelope_audit=audit)
    validate_error_envelope_audit(audit)
    assert PRIVATE not in json.dumps(audit)
    return proof, audit, observation, usage


@pytest.mark.parametrize("user_id", [None, "user_123-ABC", "a" * 128])
@pytest.mark.parametrize("channel", ["http429", "http200"])
def test_bounded_user_metadata_retries_and_persists_audit_hold(tmp_path, monkeypatch, user_id, channel):
    data = envelope(user_id=user_id)
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [error(data) if channel == "http429" else data, receipt(c)])
        assert call(g, c) == "local fixture"
        state = g.snapshot()
        assert len(calls) == 2 and calls[0] == calls[1] and not state["stopped"]
        assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        first, second = state["attempts"]
        audit = first["error_envelope_audit"]
        validate_error_envelope_audit(audit)
        assert no_content_failure(first)
        assert audit["rejection_reasons"] == [] and audit["read_complete"] and audit["close_ok"]
        assert audit["user_id_status"] == ("null" if user_id is None else "bounded_identifier")
        assert audit["source"] == ("http_error_bytes" if channel == "http429" else "parsed_response_canonical")
        if channel == "http429":
            raw = json.dumps(data).encode()
            assert audit["captured_sha256"] == hashlib.sha256(raw).hexdigest()
            assert audit["captured_bytes"] == len(raw)
        assert second["infrastructure_retry_of_request_id"] == first["request_id"]
        assert PRIVATE not in json.dumps(state)
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        assert g.snapshot()["attempts"] == state["attempts"]


@pytest.mark.parametrize("extras,reason", [
    ({"user_id": 7}, "user_id_metadata_invalid"),
    ({"user_id": True}, "user_id_metadata_invalid"),
    ({"user_id": {"content": PRIVATE}}, "user_id_metadata_invalid"),
    ({"user_id": "x" * 129}, "user_id_metadata_invalid"),
    ({"user_id": "contains spaces"}, "user_id_metadata_invalid"),
    ({"user_id": ""}, "user_id_metadata_invalid"),
    ({"unknown_secret_key": PRIVATE}, "unknown_top_keys"),
    ({"choices": []}, "model_content_channel_present"),
    ({"choices": [{"message": {"content": PRIVATE}}]}, "model_content_channel_present"),
    ({"content": None}, "model_content_channel_present"),
    ({"refusal": PRIVATE}, "model_content_channel_present"),
    ({"usage": {"cost": float("nan")}}, "nonfinite_or_unserializable"),
])
@pytest.mark.parametrize("channel", ["http429", "http200"])
def test_mixed_unknown_invalid_or_nonfinite_envelopes_never_retry(tmp_path, monkeypatch, extras, reason, channel):
    data = envelope(**extras)
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [error(data) if channel == "http429" else data, receipt(c)])
        with pytest.raises(MinimalCallStopped):
            call(g, c)
        state = g.snapshot()
        assert len(calls) == 1 and state["stopped"] and state["unknown_cost_requests"] == 1
        row = state["attempts"][0]
        assert not no_content_failure(row) and "no_content_evidence" not in row
        assert reason in row["error_envelope_audit"]["rejection_reasons"]
        assert PRIVATE not in json.dumps(state) and "unknown_secret_key" not in json.dumps(state)


@pytest.mark.parametrize("http_status,code", [(429,401),(503,403),(401,429),(403,503)])
def test_authentication_never_becomes_retryable(http_status, code):
    data = envelope(user_id="user_1")
    data["error"]["code"] = code
    proof, audit, _, _ = inspect(error(data, status=http_status))
    assert not proof
    assert ("http_status_not_eligible" if http_status in {401,403} else "error_code_not_eligible") in audit["rejection_reasons"]


@pytest.mark.parametrize("mode", ["tail", "length_mismatch", "close_failed", "read_failed", "oversized"])
def test_incomplete_capture_never_grants_proof(mode):
    raw = json.dumps(envelope(user_id="user_1")).encode()
    headers = {}
    class Stream(io.BytesIO):
        def read(self, size=-1):
            if mode == "read_failed":
                raise TimeoutError("synthetic interruption")
            if mode == "tail" and self.tell() == 0:
                return super().read(len(raw))
            return super().read(size)
        def close(self):
            super().close()
            if mode == "close_failed":
                raise OSError("synthetic close failure")
    if mode == "length_mismatch": headers["Content-Length"] = str(len(raw) + 1)
    supplied = raw + (b"X" if mode == "tail" else b"")
    if mode == "oversized": supplied = b"x" * (MAX_ERROR_BODY_BYTES + 20)
    proof, audit, _, _ = inspect(error(supplied, stream=Stream(supplied), headers=headers))
    assert not proof
    assert ("body_close_failed" if mode == "close_failed" else "capture_not_complete") in audit["rejection_reasons"]
    assert audit["captured_bytes"] is None or audit["captured_bytes"] <= MAX_ERROR_BODY_BYTES + 1


def test_new_audit_does_not_migrate_old_row_and_rejects_tampering(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        stubs(monkeypatch, [error(envelope(user_id="user_1")), receipt(c)])
        call(g,c)
        rows = g.snapshot()["attempts"]
    old = deepcopy(rows)
    del old[0]["error_envelope_audit"]
    validate_e1br_records(old, stage="main", registration_sha256="b" * 64)
    assert "error_envelope_audit" not in old[0]
    for field, value in [("captured_sha256", PRIVATE), ("user_id_status", PRIVATE),
                         ("rejection_reasons", [PRIVATE]), ("unknown_top_key_count", True),
                         ("known_top_keys", ["error", "unknown_secret_key"]), ("http_status", 401)]:
        corrupt = deepcopy(rows)
        corrupt[0]["error_envelope_audit"][field] = value
        with pytest.raises(ValueError):
            validate_e1br_records(corrupt, stage="main", registration_sha256="b" * 64)


def test_default_legacy_observation_is_not_implicitly_upgraded():
    raw = json.dumps(envelope(user_id="user_1")).encode()
    legacy = {}
    observation, _ = observe_http_error(error(raw), recovery_evidence=legacy)
    assert observation["body_status"] == "parsed" and legacy == {}


def test_three_failures_keep_shared_retry_cap_and_three_unknown_holds(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client("memory"); g.attach(c)
        calls = stubs(monkeypatch, [error(envelope(user_id="user_1")) for _ in range(4)])
        with pytest.raises(MinimalCallStopped): call(g,c)
        state = g.snapshot()
        assert len(calls) == 3 and state["unknown_cost_requests"] == 3
        assert state["e1br"]["infrastructure_retries_used"] == 2
        assert state["stop_reason"] == "infrastructure_retries_exhausted"
