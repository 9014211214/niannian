"""Synthetic price-table boundaries; no model requests, secrets or private artifacts."""
from copy import deepcopy
from contextlib import closing
import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from aria._openrouter import DEFAULT_ONLINE_FAST_MODEL, DEFAULT_MEMORY_MODEL
from scripts import prepare_model_policy as policy_tool
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped




def inputs():
    models = {"data": [{"id": model, "canonical_slug": model + "-fixture",
                        "reasoning": {"mandatory": False}}
                       for model in (DEFAULT_ONLINE_FAST_MODEL, DEFAULT_MEMORY_MODEL)]}
    endpoints = []
    for model, provider in ((DEFAULT_ONLINE_FAST_MODEL, "alibaba"), (DEFAULT_MEMORY_MODEL, "anthropic")):
        endpoints.append({"data": {"id": model, "endpoints": [{
            "tag": provider, "provider_name": provider.title(), "status": 0,
            "supported_parameters": ["max_tokens", "reasoning", "response_format", "temperature"],
            "context_length": 1000000, "max_completion_tokens": 65536 if provider == "alibaba" else 128000,
            "max_prompt_tokens": 983616 if provider == "alibaba" else None,
            "pricing": {"prompt": "0.00000003", "completion": "0.00000013", "discount": 0,
                        "input_cache_write": "0.000000038", "overrides": [
                            {"min_prompt_tokens": 32000, "prompt": "0.0000001", "completion": "0.0000004"},
                            {"min_prompt_tokens": 256000, "prompt": "0.0000002", "completion": "0.0000008",
                             "input_cache_write": "0.00000025"}]} if provider == "alibaba" else {
                "prompt": "0.000002", "completion": "0.00001", "input_cache_write_1h": "0.000004",
                "web_search": "0.01", "request": "0", "discount": 0}
        }]}})
    return [models, *endpoints]


def raw(rows):
    return [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in rows]


def derive(rows, **kwargs):
    return policy_tool.derive_policy(*raw(rows), **kwargs)


def pricing(rows):
    return rows[1]["data"]["endpoints"][0]["pricing"]


def test_default_retains_lowest_tier_and_explicit_false_contract():
    rows = inputs()
    assert derive(rows) == derive(rows, cover_all_price_tiers=False)
    assert derive(rows)[0]["models"][DEFAULT_ONLINE_FAST_MODEL]["max_input_tokens"] == 31999


def test_explicit_all_tiers_uses_maximum_cache_input_output_and_advertised_caps():
    rows = inputs()
    before = deepcopy(rows)
    policy, proof = derive(rows, cover_all_price_tiers=True)
    online, offline = policy["models"][DEFAULT_ONLINE_FAST_MODEL], policy["models"][DEFAULT_MEMORY_MODEL]
    assert online["max_input_tokens"] == 934464
    assert online["input_usd_per_million"] == "0.25"
    assert online["output_usd_per_million"] == "0.8"
    assert online["max_output_tokens"] == 65536
    assert offline["input_usd_per_million"] == "4"
    assert offline["output_usd_per_million"] == "1E+1"
    assert offline["max_input_tokens"] == 872000
    assert policy["framing_tokens"] == 8192
    assert online["metadata_sha256"] == hashlib.sha256(raw(rows)[1]).hexdigest()
    assert offline["metadata_sha256"] == hashlib.sha256(raw(rows)[2]).hexdigest()
    assert proof[0]["price_tier_boundaries"] == [32000, 256000]
    assert proof[0]["plain_input_usd_per_million"] == "0.03000000"
    assert all(p["price_tier_restricted"] is False and p["provider_billing_guarantee"] is False for p in proof)
    assert rows == before


def test_maximum_is_across_all_tiers_not_last_and_partial_rates_inherit_safely():
    rows = inputs()
    tiers = pricing(rows)["overrides"]
    tiers[0].update(input_cache_read="0.0000009", completion="0.0000013")
    tiers[1] = {"min_prompt_tokens": 256000, "input_cache_write_1h": "0.0000004"}
    online = derive(rows, cover_all_price_tiers=True)[0]["models"][DEFAULT_ONLINE_FAST_MODEL]
    assert online["input_usd_per_million"] == "0.9"
    assert online["output_usd_per_million"] == "1.3"


def test_smaller_separate_prompt_limit_is_not_widened():
    rows = inputs(); rows[1]["data"]["endpoints"][0]["max_prompt_tokens"] = 200000
    assert derive(rows, cover_all_price_tiers=True)[0]["models"][DEFAULT_ONLINE_FAST_MODEL]["max_input_tokens"] == 200000


@pytest.mark.parametrize("boundary", [True, False, "32000", 32000.0, 0, -1, None, 1000001])
def test_rejects_unverified_tier_boundaries(boundary):
    rows = inputs(); pricing(rows)["overrides"][0]["min_prompt_tokens"] = boundary
    with pytest.raises(ValueError, match="unverified_price_tier"):
        derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("boundary", [32000, 10000])
def test_rejects_duplicate_or_unsorted_tier_boundaries(boundary):
    rows = inputs(); pricing(rows)["overrides"][1]["min_prompt_tokens"] = boundary
    with pytest.raises(ValueError, match="unverified_price_tier"):
        derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("overrides", [None, {}, "tiers", [None], [[]], [{"min_prompt_tokens": 32000}]])
def test_rejects_unverified_tier_collection_shape(overrides):
    rows = inputs(); pricing(rows)["overrides"] = overrides
    with pytest.raises(ValueError):
        derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("where", ["base", "tier"])
@pytest.mark.parametrize("field", ["prompt", "completion", "input_cache_read", "input_cache_write", "input_cache_write_1h", "discount", "web_search"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", -1, True, None])
def test_every_published_price_is_finite_nonnegative_scalar(where, field, value):
    rows = inputs(); price = pricing(rows)
    (price if where == "base" else price["overrides"][0])[field] = value
    with pytest.raises(ValueError):
        derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("where", ["base", "tier"])
def test_nonzero_request_and_unknown_additional_charges_are_rejected(where):
    for name in ("request", "new_surcharge"):
        rows = inputs(); price = pricing(rows)
        (price if where == "base" else price["overrides"][0])[name] = "0.1"
        with pytest.raises(ValueError):
            derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("field,value", [("max_prompt_tokens", 0), ("overrides", []), ("multiplier", 0)])
def test_unknown_tier_structure_cannot_be_treated_as_zero_price(field, value):
    rows = inputs(); pricing(rows)["overrides"][0][field] = value
    with pytest.raises(ValueError, match="unverified_price_tier_field"):
        derive(rows, cover_all_price_tiers=True)


def test_discount_does_not_reduce_maximum_and_empty_overrides_keep_context():
    rows = inputs(); price = pricing(rows); price["discount"] = "0.5"; price["overrides"] = []
    policy, proof = derive(rows, cover_all_price_tiers=True)
    assert policy["models"][DEFAULT_ONLINE_FAST_MODEL]["input_usd_per_million"] == "0.038"
    assert proof[0]["discount_applied"] is False
    assert proof[0]["price_tier_boundaries"] == []


@pytest.mark.parametrize("extra", ['"prompt":"0",', '"prompt":NaN,', '"prompt":Infinity,'])
def test_duplicate_and_nonfinite_raw_metadata_are_rejected(extra):
    values = raw(inputs())
    values[1] = values[1].replace(b'"pricing": {', ('"pricing": {' + extra).encode(), 1)
    with pytest.raises(ValueError):
        policy_tool.derive_policy(*values, cover_all_price_tiers=True)


@pytest.mark.parametrize("mode", [None, "all", 1, 0])
def test_all_tiers_requires_explicit_boolean(mode):
    with pytest.raises(ValueError, match="invalid_price_tier_mode"):
        derive(inputs(), cover_all_price_tiers=mode)


@pytest.mark.parametrize("invalid", [None, [], {}, {"data": None}, {"data": [None]},
                                     {"data": [{"id": "missing/selected"}]}])
def test_selected_models_shape_is_validated(invalid):
    rows = inputs(); rows[0] = invalid
    with pytest.raises(ValueError):
        derive(rows, cover_all_price_tiers=True)


@pytest.mark.parametrize("field,value", [("max_completion_tokens", None), ("provider_name", None),
                                         ("supported_parameters", [None]), ("pricing", None)])
def test_selected_endpoint_shape_is_validated(field, value):
    rows = inputs(); rows[1]["data"]["endpoints"][0][field] = value
    with pytest.raises(ValueError):
        derive(rows, cover_all_price_tiers=True)


def test_cli_all_tiers_creates_separate_policy_and_proof_without_overwrite(tmp_path):
    paths = []
    for index, value in enumerate(raw(inputs())):
        path = tmp_path / (str(index) + ".json"); path.write_bytes(value); paths.append(str(path))
    target = tmp_path / "all-tiers.json"
    args = ["--models-json", paths[0], "--online-endpoints-json", paths[1],
            "--offline-endpoints-json", paths[2], "--output", str(target), "--cover-all-price-tiers"]
    assert policy_tool.main(args) == 0
    expected = target.read_bytes()
    assert json.loads(expected)["models"][DEFAULT_ONLINE_FAST_MODEL]["max_input_tokens"] == 934464
    assert json.loads(target.with_suffix(".proof.json").read_bytes())["models"][0]["price_tier_boundaries"] == [32000, 256000]
    with pytest.raises(FileExistsError):
        policy_tool.main(args)
    assert target.read_bytes() == expected


def test_long_actual_wire_keeps_byte_margin_and_more_expensive_budget_reservation():
    payload = {"provider": {"only": ["alibaba"]}, "max_tokens": 1024,
               "messages": [{"role": "user", "content": "示" * 9000}]}
    wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request, client = SimpleNamespace(data=wire), SimpleNamespace(model=DEFAULT_ONLINE_FAST_MODEL)
    state = {"known_cost_usd": "0", "reserved_cost_usd": "0"}
    old = SimpleNamespace(policy=derive(inputs())[0], threshold="1")
    expanded = SimpleNamespace(policy=derive(inputs(), cover_all_price_tiers=True)[0], threshold="1")
    expanded.policy_sha256 = hashlib.sha256(json.dumps(expanded.policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(MinimalCallStopped, match="reservation_token_limit_exceeded"):
        ReservedCallGuard._reserve(old, client, request, payload, state)
    reservation = ReservedCallGuard._reserve(expanded, client, request, payload, state)["reservation"]
    assert reservation["input_token_plan_bound"] == len(wire) + 8192
    expected = (Decimal(len(wire) + 8192) * Decimal("0.25") + Decimal(1024) * Decimal("0.8")) / 1000000
    assert Decimal(reservation["reserved_cost_usd"]) == expected
    expanded.threshold = str(expected - Decimal("0.00000001"))
    with pytest.raises(MinimalCallStopped, match="pre_send_budget_exceeded"):
        ReservedCallGuard._reserve(expanded, client, request, payload, state)


def test_all_tier_policy_cannot_replace_the_old_ledger_contract(tmp_path):
    path = tmp_path / "retained.sqlite3"
    old = derive(inputs())[0]
    with closing(ReservedCallGuard(path, max_requests=5, max_observed_cost_usd="1", reservation_policy=old)):
        pass
    # Metadata SHA stays the same; changed rates/limits still cannot overwrite
    # an existing policy. This does not rely only on changing the source hash.
    before = path.read_bytes()
    new = derive(inputs(), cover_all_price_tiers=True)[0]
    with pytest.raises(ValueError, match="retained_reservation_policy_changed"):
        ReservedCallGuard(path, max_requests=5, max_observed_cost_usd="1", reservation_policy=new)
    assert path.read_bytes() == before
