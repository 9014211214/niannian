from copy import deepcopy
import json

import pytest

from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_ONLINE_FAST_MODEL
from scripts.prepare_model_policy import derive_policy


def inputs():
    models = {"data": [{"id": model, "canonical_slug": model + "-fixture",
                        "reasoning": {"mandatory": False}}
                       for model in (DEFAULT_ONLINE_FAST_MODEL, DEFAULT_MEMORY_MODEL)]}
    endpoints = []
    for model, provider in ((DEFAULT_ONLINE_FAST_MODEL, "alibaba"), (DEFAULT_MEMORY_MODEL, "anthropic")):
        endpoints.append({"data": {"id": model, "endpoints": [{
            "tag": provider, "provider_name": provider.title(), "status": 0,
            "supported_parameters": ["max_tokens", "reasoning", "response_format"]
                + (["temperature"] if provider == "alibaba" else []),
            "context_length": 1000000, "max_completion_tokens": 128000,
            "max_prompt_tokens": None,
            "pricing": {"prompt": "0.00000003", "completion": "0.00000013",
                        "overrides": [{"min_prompt_tokens": 32000, "prompt": "0.0000001"}]}
                if provider == "alibaba" else {"prompt": "0.000002", "completion": "0.00001"}
        }]}})
    return [models, *endpoints]


def derive(rows):
    return derive_policy(*(json.dumps(row).encode() for row in rows))


def test_policy_keeps_low_price_tier_and_records_missing_separate_prompt_bound():
    policy, proof = derive(inputs())
    assert policy["models"][DEFAULT_ONLINE_FAST_MODEL]["max_input_tokens"] == 31999
    assert policy["models"][DEFAULT_MEMORY_MODEL]["max_input_tokens"] == 872000
    assert policy["models"][DEFAULT_MEMORY_MODEL]["input_usd_per_million"] == "2"
    assert proof[1]["separate_max_prompt_tokens"] is None
    assert "temperature" not in proof[1]["required_parameters"]
    assert all(row["provider_billing_guarantee"] is False for row in proof)


@pytest.mark.parametrize("status", [False, True, 1, "0", None])
def test_status_must_be_integer_zero(status):
    rows=inputs(); rows[1]["data"]["endpoints"][0]["status"] = status
    with pytest.raises(ValueError):
        derive(rows)


def test_missing_actual_parameter_support_blocks():
    rows=inputs(); rows[1]["data"]["endpoints"][0]["supported_parameters"].remove("response_format")
    with pytest.raises(ValueError, match="actual_wire_parameters_not_verified"):
        derive(rows)


def test_mandatory_reasoning_blocks_before_live():
    rows=inputs(); rows[0]["data"][1]["reasoning"]["mandatory"] = True
    with pytest.raises(ValueError):
        derive(rows)


def test_wrong_model_or_ambiguous_provider_blocks():
    rows=inputs(); rows[2]["data"]["id"] = "wrong/model"
    with pytest.raises(ValueError):
        derive(rows)
    rows=inputs(); rows[2]["data"]["endpoints"].append(deepcopy(rows[2]["data"]["endpoints"][0]))
    with pytest.raises(ValueError):
        derive(rows)


def test_price_and_metadata_changes_do_not_reuse_policy_binding():
    rows=inputs(); before,_=derive(rows)
    rows[2]["data"]["endpoints"][0]["pricing"]["prompt"]="0.000003"
    after,_=derive(rows)
    assert before["models"][DEFAULT_MEMORY_MODEL]["metadata_sha256"] != after["models"][DEFAULT_MEMORY_MODEL]["metadata_sha256"]
    assert after["models"][DEFAULT_MEMORY_MODEL]["input_usd_per_million"] == "3"
