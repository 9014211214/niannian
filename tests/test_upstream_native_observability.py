"""Finite Alibaba HTTP429 evidence, using synthetic bodies and no network."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy import response_observability as observation
from tests.test_http_error_observability import error, _run
from aria.bench.request_control import BenchmarkRequestControlStopped


CODES = ("Throttling.RateQuota", "LimitRequests", "limit_requests", "Throttling.BurstRate",
         "limit_burst_rate", "Throttling.AllocationQuota", "insufficient_quota")
MISSING = object()


def native(raw=MISSING, *, direct=MISSING, provider="Alibaba", status=429, usage=None):
    metadata = {"provider_name": provider}
    if raw is not MISSING:
        metadata["raw"] = raw
    if direct is not MISSING:
        metadata["provider_code"] = direct
    body = {"error": {"code": status, "message": "PRIVATE-secret", "metadata": metadata}}
    if usage is not None:
        body["usage"] = usage
    result, cost = observation.observe_http_error(error(body, status=status))
    observation.validate_observation(result)
    assert "PRIVATE" not in json.dumps(result)
    return result, cost


@pytest.mark.parametrize("code", CODES)
def test_documented_codes_are_retained_exactly_without_inferred_cause(code):
    result, usage = native(json.dumps({"code": code, "message": "PRIVATE-prompt"}))
    assert result["provider_native_code_status"] == "recognized"
    assert result["provider_native_code"] == code
    assert result["provider_native_code_sources"] == ["metadata.raw.code"]
    assert result["provider_raw_status"] == "parsed" and usage == {}
    assert not any(field in result for field in ("quota_type", "tpm", "retry_permitted", "cost"))


@pytest.mark.parametrize("location", ["direct", "code", "type", "error.code", "error.type"])
def test_every_supported_origin_has_a_fixed_source(location):
    code = "Throttling.RateQuota"
    if location == "direct":
        result, _ = native(direct=code)
        source = "metadata.provider_code"
    else:
        raw = {location: code} if "." not in location else {"error": {location.split(".")[1]: code}}
        result, _ = native(json.dumps(raw))
        source = "metadata.raw." + location
    assert result["provider_native_code_sources"] == [source]
    assert result["provider_native_code"] == code


def test_numeric_http_code_does_not_mask_raw_and_same_sources_agree():
    result, _ = native(json.dumps({"code": "Throttling.BurstRate", "type": "Throttling.BurstRate"}), direct=429)
    assert result["provider_code"] == 429
    assert result["provider_native_code"] == "Throttling.BurstRate"
    assert result["provider_native_code_sources"] == ["metadata.raw.code", "metadata.raw.type"]


@pytest.mark.parametrize("provider,status", [("Anthropic", 429), ("PRIVATE-provider", 429),
                                             (None, 429), ("Alibaba", 503), ("Alibaba", 400)])
def test_other_provider_or_status_never_inspects_raw(provider, status, monkeypatch):
    original = observation.json.loads
    raw = '{"code":"insufficient_quota"}'
    def bounded(text, **kwargs):
        assert text != raw, "raw from another supplier/status must not be parsed"
        return original(text, **kwargs)
    monkeypatch.setattr(observation.json, "loads", bounded)
    result, _ = native(raw, provider=provider, status=status)
    assert result["provider_raw_status"] == result["provider_native_code_status"] == "not_inspected"
    assert "provider_native_code" not in result


@pytest.mark.parametrize("raw", [None, {}, [], 429, "PRIVATE not JSON", "[]", "0",
    '{"code":"Throttling.RateQuota","code":"Throttling.BurstRate"}',
    '{"code":"Throttling.RateQuota","hidden":{"key":1,"key":2}}',
    '{"code":"Throttling.RateQuota","hidden":NaN}',
    '{"code":"Throttling.RateQuota","hidden":Infinity}',
    '{"code":"Throttling.RateQuota","error":"PRIVATE"}',
    json.dumps({"code": "Throttling.RateQuota", "message": "x" * 4096}),
    json.dumps({"code": "Throttling.RateQuota", "message": "中" * 1500}, ensure_ascii=False),
    '{"code":"Throttling.RateQuota","hidden":' + '[' * 10 + '0' + ']' * 10 + '}',
    '{"code":"Throttling.RateQuota","hidden":"\ud800"}',
])
def test_invalid_raw_is_never_partly_used_but_independent_direct_code_can_survive(raw):
    result, usage = native(raw)
    assert result["provider_raw_status"] == result["provider_native_code_status"] == "invalid"
    assert "provider_native_code" not in result and usage == {}
    result, _ = native(raw, direct="Throttling.AllocationQuota")
    assert result["provider_raw_status"] == "invalid"
    assert result["provider_native_code"] == "Throttling.AllocationQuota"
    assert result["provider_native_code_sources"] == ["metadata.provider_code"]


@pytest.mark.parametrize("raw,direct,status", [
    (MISSING, MISSING, "absent"), ("{}", MISSING, "absent"),
    ('{"code":"PRIVATE-dynamic-code"}', MISSING, "unrecognized"),
    ('{"code":429}', MISSING, "unrecognized"),
    ('{"code":"Throttling.RateQuota","type":"Throttling.BurstRate"}', MISSING, "conflicting"),
    ('{"code":"Throttling.RateQuota","type":"PRIVATE-unknown"}', MISSING, "conflicting"),
    ('{"code":"Throttling.RateQuota"}', "Throttling.AllocationQuota", "conflicting"),
])
def test_missing_unknown_and_conflicting_values_never_choose_a_code(raw, direct, status):
    result, _ = native(raw, direct=direct)
    assert result["provider_native_code_status"] == status
    assert "provider_native_code" not in result and "provider_native_code_sources" not in result


@pytest.mark.parametrize("field,value", [
    ("provider_native_code", "PRIVATE-url-or-token"), ("provider_native_code_status", "PRIVATE"),
    ("provider_native_code_sources", ["PRIVATE-source"]),
    ("provider_native_code_sources", ["metadata.raw.code", "metadata.raw.code"]),
    ("provider_native_code_sources", "metadata.raw.code"), ("provider_raw_status", "absent"),
    ("provider_native_code_status", "conflicting"), ("provider_reported_name", "Anthropic"),
    ("provider_raw_status", "PRIVATE"), ("raw", "PRIVATE-raw"),
])
def test_saved_native_metadata_cannot_inject_values_or_relabel_origins(field, value):
    result, _ = native('{"code":"Throttling.RateQuota"}')
    result[field] = value
    with pytest.raises(ValueError, match="invalid_response_observability"):
        observation.validate_observation(result)


def test_prior_12_observation_is_accepted_unchanged_and_cannot_gain_native_fields():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "artifacts/local/e1b-formal-v6/prescreen/execution-receipt.json"
    # Historical rows are not required for a portable software test.
    result, _ = native()
    result["version"] = "1.2"
    for field in ("provider_native_code_status", "provider_raw_status"):
        result.pop(field)
    before = deepcopy(result)
    observation.validate_observation(result)
    assert result == before
    if path.exists():
        saved = path.read_bytes()
        old = json.loads(saved)["accounting"]["attempts"][-1]["observation"]
        assert old["version"] == "1.2"
        observation.validate_observation(old)
        assert path.read_bytes() == saved
    result["provider_raw_status"] = "absent"
    with pytest.raises(ValueError):
        observation.validate_observation(result)


def test_native_rate_code_does_not_change_unknown_billing_stop_or_authorize_retry():
    body = {"error": {"code": 429, "metadata": {"provider_name": "Alibaba",
        "raw": '{"code":"Throttling.RateQuota","usage":{"cost":0},"message":"PRIVATE"}'}}}
    control, client, calls = _run(error(body), max_observed_cost_usd=.04)
    account = control.accounting()
    assert account["provider_cost_complete"] is False and account["provider_observed_cost_usd"] is None
    assert account["attempts"][0]["response_observability"]["provider_native_code"] == "Throttling.RateQuota"
    with pytest.raises(BenchmarkRequestControlStopped):
        client.complete([])
    assert len(calls) == 1


def test_explicit_top_level_zero_usage_remains_separate_from_native_code():
    result, usage = native('{"code":"insufficient_quota","cost":99}', usage={"cost": 0})
    assert usage == {"cost": 0}
    assert result["provider_native_code"] == "insufficient_quota"
