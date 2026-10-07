"""Explicit model-output fixtures; these labels do not measure semantic accuracy."""


def communication_review(**changes):
    return {"version": "preference_semantics_v1", "kind": "communication_style",
            "owner_requested": True, "quote_supported": True,
            "scope_preserved": True, "no_permission_claim": True, **changes}
