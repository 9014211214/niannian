"""Explicit memory storage contract; model interpretation is not consent."""
from __future__ import annotations

from ..model.persona import PREFERENCE_SCOPE_MODEL, preference_scope_max_characters


def validate_model_scope_client(client):
    if (getattr(client, "preference_scope_protocol", None) != PREFERENCE_SCOPE_MODEL
            or getattr(client, "role", None) != "memory"):
        raise ValueError("invalid_model_preference_scope_client")
    return {"protocol": PREFERENCE_SCOPE_MODEL,
            "max_characters": preference_scope_max_characters(PREFERENCE_SCOPE_MODEL),
            "interpretation": "model_proposal", "evidence": "verbatim_owner_quotes",
            "automatic_confirmation": False, "semantic_verified": False}
