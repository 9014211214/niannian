"""Backward-compatible source entry; installed users can run aria-model-policy."""
from aria.engine.policy.release_configuration import (
    derive_policy, derive_max_policy, main, _price, _strict_metadata,
    _all_tier_prices, _derive_all_tier_policy, _derive_lowest_tier_policy,
)

if __name__ == "__main__":
    raise SystemExit(main())
