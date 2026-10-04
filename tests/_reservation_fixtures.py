"""Synthetic high unit prices cover historical fake response costs in tests."""
from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_ONLINE_FAST_MODEL


def fixture_reservation_policy():
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {
        model: {"input_usd_per_million": "10", "output_usd_per_million": "50",
                "provider": provider, "response_models": [model],
                "response_providers": [provider.title()], "max_input_tokens": 100000,
                "max_output_tokens": 10000, "metadata_sha256": "a" * 64}
        for model, provider in [(DEFAULT_ONLINE_FAST_MODEL, "alibaba"), (DEFAULT_MEMORY_MODEL, "anthropic")]
    }}
