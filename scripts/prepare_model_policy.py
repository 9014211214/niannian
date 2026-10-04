"""Derive a priced, bounded policy from saved unauthenticated public metadata.

This CLI is offline and never obtains credentials or sends model requests.
The three source files must be refreshed and sealed by the execution owner.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_ONLINE_FAST_MODEL
from aria.engine.policy.reserved_calls import validate_reservation_policy


def _price(value):
    if type(value) not in (str, int, float):
        raise ValueError("invalid_metadata_price")
    try:
        number = Decimal(str(value))
    except Exception:
        raise ValueError("invalid_metadata_price") from None
    if not number.is_finite() or number < 0:
        raise ValueError("invalid_metadata_price")
    return number


def derive_policy(models_raw: bytes, online_raw: bytes, offline_raw: bytes):
    models = {item["id"]: item for item in json.loads(models_raw)["data"]}
    result = {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {}}
    proof = []
    for model, provider, raw in ((DEFAULT_ONLINE_FAST_MODEL, "alibaba", online_raw),
                                  (DEFAULT_MEMORY_MODEL, "anthropic", offline_raw)):
        item = models[model]
        endpoint_data = json.loads(raw)["data"]
        if endpoint_data.get("id") != model:
            raise ValueError("metadata_model_mismatch")
        candidates = [e for e in endpoint_data["endpoints"] if e.get("tag") == provider]
        if len(candidates) != 1:
            raise ValueError("unique_selected_provider_required")
        endpoint = candidates[0]
        required = {"max_tokens", "reasoning"}
        if model == DEFAULT_ONLINE_FAST_MODEL:
            required |= {"temperature", "response_format"}
        if (type(endpoint.get("status")) is not int or endpoint["status"] != 0
                or not required <= set(endpoint.get("supported_parameters", []))
                or item.get("reasoning", {}).get("mandatory") is not False):
            raise ValueError("actual_wire_parameters_not_verified")
        price = endpoint["pricing"]
        # Fixed request charges are outside a token-only reservation. Tool
        # search is not enabled by this strictly text-only, no-tools wire.
        if _price(price.get("request", 0)) != 0:
            raise ValueError("unreserved_fixed_request_price")
        known_prices = {"prompt", "completion", "request", "discount", "overrides", "web_search",
                        "input_cache_read", "input_cache_write", "input_cache_write_1h"}
        if any(_price(value) != 0 for key, value in price.items() if key not in known_prices):
            raise ValueError("unreserved_additional_price")
        prompt_price = _price(price["prompt"])
        # Cache writes may be more expensive than plain input even if the app
        # does not request caching. Reserve the highest published input rate.
        input_price = max([prompt_price] + [_price(price[key]) for key in (
            "input_cache_read", "input_cache_write", "input_cache_write_1h") if key in price])
        context = endpoint["context_length"]
        max_output = endpoint["max_completion_tokens"]
        if any(type(n) is not int or n < 64 for n in (context, max_output)):
            raise ValueError("invalid_context_limits")
        # A missing separate prompt ceiling is recorded; context minus the
        # advertised maximum output is the more restrictive planning ceiling.
        max_input = context - max_output
        separate_prompt = endpoint.get("max_prompt_tokens")
        if separate_prompt is not None:
            if type(separate_prompt) is not int or separate_prompt < 64:
                raise ValueError("invalid_prompt_limit")
            max_input = min(max_input, separate_prompt)
        overrides = price.get("overrides", [])
        for tier in overrides:
            boundary = tier.get("min_prompt_tokens")
            if type(boundary) is not int or boundary < 1:
                raise ValueError("unverified_price_tier")
            max_input = min(max_input, boundary - 1)
        result["models"][model] = {
            "input_usd_per_million": str(input_price * 1000000),
            "output_usd_per_million": str(_price(price["completion"]) * 1000000),
            "provider": provider,
            "response_models": list(dict.fromkeys([model, item["canonical_slug"]])),
            "response_providers": [endpoint["provider_name"]],
            "max_input_tokens": max_input, "max_output_tokens": max_output,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        }
        proof.append({"model": model, "provider": provider,
                      "required_parameters": sorted(required),
                      "separate_max_prompt_tokens": separate_prompt,
                      "planning_max_input_tokens": max_input,
                      "plain_input_usd_per_million": str(prompt_price * 1000000),
                      "reserved_input_usd_per_million": str(input_price * 1000000),
                      "price_tier_restricted": bool(overrides),
                      "provider_billing_guarantee": False})
    return validate_reservation_policy(result), proof


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--online-endpoints-json", type=Path, required=True)
    p.add_argument("--offline-endpoints-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    policy, proof = derive_policy(a.models_json.read_bytes(), a.online_endpoints_json.read_bytes(),
                                  a.offline_endpoints_json.read_bytes())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    with a.output.with_suffix(".proof.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"models": proof, "models_sha256": hashlib.sha256(a.models_json.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps({"status": "prepared_offline", "models": len(proof), "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
