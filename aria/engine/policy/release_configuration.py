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


_INPUT_PRICES = ("prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h")
_TOKEN_PRICES = frozenset((*_INPUT_PRICES, "completion"))
_SCALAR_PRICES = _TOKEN_PRICES | {"request", "discount", "web_search"}


def _strict_metadata(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_metadata_field")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("nonfinite_metadata_value")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (TypeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid_metadata_json") from None


def _all_tier_prices(price, context):
    """Reserve the maximum known rate, without assuming an unverified tier shape."""
    if type(price) is not dict or not {"prompt", "completion"} <= price.keys():
        raise ValueError("invalid_metadata_pricing")
    tiers = price.get("overrides", [])
    if type(tiers) is not list:
        raise ValueError("unverified_price_tier")
    input_rates, output_rates, boundaries = [], [], []
    previous = 0
    for index, tier in enumerate([price, *tiers]):
        if type(tier) is not dict:
            raise ValueError("unverified_price_tier")
        if index:
            boundary = tier.get("min_prompt_tokens")
            if type(boundary) is not int or not previous < boundary <= context:
                raise ValueError("unverified_price_tier")
            if not _TOKEN_PRICES.intersection(tier):
                raise ValueError("empty_price_tier")
            previous = boundary
            boundaries.append(boundary)
        for key, value in tier.items():
            if (index == 0 and key == "overrides") or (index and key == "min_prompt_tokens"):
                continue
            if index and key not in _SCALAR_PRICES:
                # Unknown structural dimensions cannot be interpreted as a
                # simple input threshold, even when their value happens to be 0.
                raise ValueError("unverified_price_tier_field")
            number = _price(value)
            if key == "request" and number != 0:
                raise ValueError("unreserved_fixed_request_price")
            if key not in _SCALAR_PRICES and number != 0:
                raise ValueError("unreserved_additional_price")
            if key == "discount" and number > 1:
                raise ValueError("invalid_metadata_discount")
            if key in _TOKEN_PRICES and number > 1:
                raise ValueError("invalid_metadata_price")
            if key in _INPUT_PRICES:
                input_rates.append(number)
            elif key == "completion":
                output_rates.append(number)
            # Discounts are not subtracted. Web search is disabled by the
            # frozen no-tools request, but its published value is still checked.
    return max(input_rates), max(output_rates), boundaries


def derive_policy(models_raw: bytes, online_raw: bytes, offline_raw: bytes, *,
                  cover_all_price_tiers=False, release_profile="legacy",
                  offline_model=DEFAULT_MEMORY_MODEL, offline_provider="anthropic"):
    if type(cover_all_price_tiers) is not bool:
        raise ValueError("invalid_price_tier_mode")
    if type(release_profile) is not str or release_profile not in {"legacy", "release_v1", "release_max_v1", "release_deepseek_v1"}:
        raise ValueError("invalid_release_profile")
    if release_profile in ("release_max_v1", "release_deepseek_v1"):
        if not cover_all_price_tiers:raise ValueError("release_max_v1_requires_all_price_tiers")
        return derive_max_policy(models_raw,online_raw,offline_raw,release_profile=release_profile,
                                 offline_model=offline_model,offline_provider=offline_provider)
    if (offline_model, offline_provider) != (DEFAULT_MEMORY_MODEL, "anthropic"):
        raise ValueError("explicit_offline_selection_requires_deepseek_release")
    online_model = DEFAULT_ONLINE_FAST_MODEL
    route = None
    if release_profile == "release_v1":
        from aria.release import ONLINE_MODEL, ONLINE_PROVIDER
        from aria.engine.policy.online_structured import build_online_structured_route
        if not cover_all_price_tiers:
            raise ValueError("release_v1_requires_all_price_tiers")
        online_model = ONLINE_MODEL
        route = build_online_structured_route(
            models_raw, online_raw, model=online_model, provider=ONLINE_PROVIDER)
    derive = _derive_all_tier_policy if cover_all_price_tiers else _derive_lowest_tier_policy
    policy, proof = derive(models_raw, online_raw, offline_raw, online_model=online_model)
    if route is not None:
        binding = route.binding()
        proof[0].update(required_parameters=binding["required_parameters"],
                        online_return_format=binding["format"], release_profile=release_profile)
    return policy, proof


def _derive_all_tier_policy(models_raw, online_raw, offline_raw, *,
                            online_model=DEFAULT_ONLINE_FAST_MODEL):
    # Reuse the original identity/parameter/context validation without
    # changing its default behavior or proof. Validate every extra tier before
    # constructing a legacy-compatible base view with no tier restriction.
    model_data = _strict_metadata(models_raw)
    if (type(model_data) is not dict or type(model_data.get("data")) is not list
            or any(type(row) is not dict or type(row.get("id")) is not str
                   for row in model_data["data"])
            or len({row["id"] for row in model_data["data"]}) != len(model_data["data"])):
        raise ValueError("invalid_metadata_models")
    selected_models = {row["id"]: row for row in model_data["data"]}
    for model in (online_model, DEFAULT_MEMORY_MODEL):
        item = selected_models.get(model)
        if (item is None or type(item.get("canonical_slug")) is not str
                or not item["canonical_slug"] or type(item.get("reasoning")) is not dict):
            raise ValueError("invalid_metadata_models")
    transformed, tier_proofs = [], []
    for raw, model, provider in ((online_raw, online_model, "alibaba"),
                                (offline_raw, DEFAULT_MEMORY_MODEL, "anthropic")):
        data = _strict_metadata(raw)
        if type(data) is not dict or type(data.get("data")) is not dict:
            raise ValueError("invalid_metadata_endpoints")
        if data["data"].get("id") != model:
            raise ValueError("metadata_model_mismatch")
        endpoints = data.get("data", {}).get("endpoints")
        if type(endpoints) is not list or any(type(row) is not dict for row in endpoints):
            raise ValueError("invalid_metadata_endpoints")
        matches = [row for row in endpoints if row.get("tag") == provider]
        if len(matches) != 1:
            raise ValueError("unique_selected_provider_required")
        endpoint = matches[0]
        context = endpoint.get("context_length")
        max_output = endpoint.get("max_completion_tokens")
        if any(type(value) is not int or value < 64 for value in (context, max_output)):
            raise ValueError("invalid_context_limits")
        if (type(endpoint.get("provider_name")) is not str or not endpoint["provider_name"]
                or type(endpoint.get("supported_parameters")) is not list
                or any(type(value) is not str for value in endpoint["supported_parameters"])):
            raise ValueError("invalid_metadata_endpoint_fields")
        rates = _all_tier_prices(endpoint.get("pricing"), context)
        tier_proofs.append((rates, hashlib.sha256(raw).hexdigest(),
                            _price(endpoint["pricing"]["prompt"])))
        # Keep all token rates, but the selected maxima replace prompt and
        # completion. The legacy validator then enforces the ordinary caps.
        endpoint["pricing"]["prompt"] = str(rates[0])
        endpoint["pricing"]["completion"] = str(rates[1])
        endpoint["pricing"]["overrides"] = []
        transformed.append(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    policy, proof = _derive_lowest_tier_policy(
        json.dumps(model_data, ensure_ascii=False, allow_nan=False).encode("utf-8"), *transformed,
        online_model=online_model)
    for (rates, source_sha, plain), row in zip(tier_proofs, proof):
        model = row["model"]
        policy["models"][model]["metadata_sha256"] = source_sha
        row.update(plain_input_usd_per_million=str(plain * 1000000),
                   price_tier_restricted=False, price_tier_coverage="all_published_input_thresholds",
                   price_tier_boundaries=rates[2], reserved_output_usd_per_million=str(rates[1] * 1000000),
                   discount_applied=False, web_search_enabled=False)
    return validate_reservation_policy(policy), proof


def _derive_lowest_tier_policy(models_raw: bytes, online_raw: bytes, offline_raw: bytes, *,
                               online_model=DEFAULT_ONLINE_FAST_MODEL):
    models = {item["id"]: item for item in json.loads(models_raw)["data"]}
    result = {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {}}
    proof = []
    for model, provider, raw in ((online_model, "alibaba", online_raw),
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
        if model == online_model:
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


def derive_max_policy(models_raw,online_raw,offline_raw,*,release_profile="release_max_v1",
                      offline_model=DEFAULT_MEMORY_MODEL,offline_provider="anthropic"):
    from aria.release import build_release_online_route
    from aria.engine.policy.memory_structured import build_memory_structured_route
    online=build_release_online_route(models_raw,online_raw,release_profile=release_profile).binding()
    offline=build_memory_structured_route(models_raw,offline_raw,
        model=offline_model,provider=offline_provider).binding()
    if (offline_model,offline_provider)!=(DEFAULT_MEMORY_MODEL,"anthropic"):
        if (release_profile!="release_deepseek_v1" or offline_model!=online['model']
                or offline_provider!=online['provider'] or online_raw!=offline_raw):
            raise ValueError("deepseek_offline_requires_same_together_metadata")
    policy={"schema":"aria_reservation_v1","framing_tokens":8192,"models":{}};proof=[]
    online_output = 4096 if release_profile == "release_deepseek_v1" else 8202
    for raw,model,provider,binding,output in ((online_raw,online['model'],online['provider'],online,online_output),
            (offline_raw,offline_model,offline_provider,offline,4000)):
        data=_strict_metadata(raw)["data"]
        endpoints=[e for e in data["endpoints"] if e.get("tag")==provider]
        if len(endpoints)!=1:raise ValueError("unique_selected_provider_required")
        e=endpoints[0];context=e.get("context_length");prompt=e.get("max_prompt_tokens") or context
        if (type(context) is not int or type(prompt) is not int or prompt<64 or prompt>context
            or type(e.get("max_completion_tokens")) is not int or e["max_completion_tokens"]<output):
            raise ValueError("invalid_context_limits")
        price=e["pricing"];rates=_all_tier_prices(price,context)
        if _price(price.get("request",0))!=0:raise ValueError("unreserved_fixed_request_price")
        known={"prompt","completion","request","discount","overrides","web_search","input_cache_read","input_cache_write","input_cache_write_1h"}
        if any(_price(v)!=0 for k,v in price.items() if k not in known):raise ValueError("unreserved_additional_price")
        spec={"input_usd_per_million":str(rates[0]*1000000),
            "output_usd_per_million":str(rates[1]*1000000),"provider":provider,
            "response_models":list(dict.fromkeys([model,binding["canonical_slug"]])),
            "response_providers":[e["provider_name"]],"max_input_tokens":prompt,"max_output_tokens":output,
            "metadata_sha256":hashlib.sha256(raw).hexdigest()}
        previous=policy["models"].get(model)
        if previous is not None:
            if any(previous[key]!=value for key,value in spec.items() if key!="max_output_tokens"):
                raise ValueError("shared_model_reservation_route_mismatch")
            spec["max_output_tokens"]=max(previous["max_output_tokens"],spec["max_output_tokens"])
        policy["models"][model]=spec
        proof.append({"model":model,"provider":provider,"release_profile":release_profile,
            "discount_applied":False,"price_tier_coverage":"all_published_input_thresholds",
            "price_tier_boundaries":rates[2],"reserved_output_token_limit":output,
            "gateway_mapping_observed":False,"provider_billing_guarantee":False})
    return validate_reservation_policy(policy),proof


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--online-endpoints-json", type=Path, required=True)
    p.add_argument("--offline-endpoints-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release-profile", choices=("legacy", "release_v1", "release_max_v1", "release_deepseek_v1"), default="legacy",
                   help="Select the exact model bundle explicitly; release profiles require all price tiers.")
    from .memory_structured import OFFLINE_ROUTES
    p.add_argument("--offline-model", choices=sorted({model for model,_ in OFFLINE_ROUTES}),default=DEFAULT_MEMORY_MODEL)
    p.add_argument("--offline-provider", choices=sorted({provider for _,provider in OFFLINE_ROUTES}),default="anthropic")
    p.add_argument("--cover-all-price-tiers", action="store_true",
                   help="Explicitly reserve the highest published input/cache/output tier and full context caps; create a new policy/ledger.")
    a = p.parse_args(argv)
    policy, proof = derive_policy(a.models_json.read_bytes(), a.online_endpoints_json.read_bytes(),
                                  a.offline_endpoints_json.read_bytes(), cover_all_price_tiers=a.cover_all_price_tiers,
                                  release_profile=a.release_profile,offline_model=a.offline_model,
                                  offline_provider=a.offline_provider)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    with a.output.with_suffix(".proof.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"models": proof, "models_sha256": hashlib.sha256(a.models_json.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps({"status": "prepared_offline", "models": len(policy["models"]), "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
