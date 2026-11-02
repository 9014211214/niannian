"""Machine-readable output shape for offline reflection prompt contract 2.5.

This schema constrains syntax, enums, counts and preference/outcome shape. It
cannot establish quote ownership, scope entailment, old identity or event
novelty. The updater's exact current-quote/lineage transaction validation remains
authoritative. No missing key, citation, identifier or output text is repaired.
The generation contract asks for <=1 change; the old wire parser deliberately
continues to accept <=3 well-formed proposals and deterministically select one.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_Text100 = Annotated[str, Field(min_length=1, max_length=100, pattern=r"\S")]
_Text240 = Annotated[str, Field(min_length=1, max_length=240, pattern=r"\S")]
_Strength = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _Object(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Quote(_Object):
    turn_index: Annotated[int, Field(ge=0)]
    evidence_quote: Annotated[str, Field(min_length=2, max_length=240)]


_Quotes = Annotated[list[_Quote], Field(min_length=1, max_length=3)]


class _Review(_Object):
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")]
    hypothesis_id: str | None
    claim_quote: str | None
    scope_quote: str | None
    applicability: Literal["within_scope", "outside_scope", "unknown"]
    relation: Literal["support", "counter", "neutral", "unknown"]
    evidence_basis: Literal[
        "stated_preference", "observed_response", "repeated_pattern", "context_only",
    ]
    evidence: _Quotes
    event_evidence: _Quotes
    reason: _Text240
    event_summary: _Text240


class _AtomicBase(_Object):
    aspect: _Text100
    conditions: _Text100
    tendency: _Text100


class _StatedPreference(_AtomicBase):
    outcome: None
    evidence_basis: Literal["stated_preference"]


class _ObservedResponse(_AtomicBase):
    outcome: _Text100
    evidence_basis: Literal["observed_response"]


class _RepeatedPattern(_AtomicBase):
    outcome: _Text100 | None
    evidence_basis: Literal["repeated_pattern"]


class _Lineage(_Object):
    id: str
    claim_quote: str
    scope_quote: str
    mechanism_relation: Literal["same"]
    scope_relation: Literal["within"]


class _Change(_Object):
    review_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")]
    id: str | None
    framework: Literal["response", "big_five", "disc", "attachment", "cbt"]
    dimension: str
    claim: str
    scope: str | None
    scope_tags: Annotated[list[str], Field(min_length=1, max_length=8)]
    relation: Literal["new", "support", "counter", "retire"]
    lineage: _Lineage | None
    atomic_claim: _StatedPreference | _ObservedResponse | _RepeatedPattern
    target_value: _Strength | None
    confidence: _Strength = Field(description=(
        "Strength of this quoted evidence relation, NOT desired accumulated "
        "person confidence or a calibrated personality probability."
    ))
    evidence: _Quotes

    @model_validator(mode="after")
    def _declared_shape(self):
        # Import lazily so the production enum source cannot drift or create
        # an import cycle when diagnosis exports the schema helper.
        from .diagnosis import _FRAMEWORK_DIMENSIONS, _SCOPE_TAGS

        if self.dimension not in _FRAMEWORK_DIMENSIONS[self.framework]:
            raise ValueError("invalid_contract_dimension")
        if (len(set(self.scope_tags)) != len(self.scope_tags)
                or any(tag not in _SCOPE_TAGS for tag in self.scope_tags)
                or ("general" in self.scope_tags and len(self.scope_tags) > 1)):
            raise ValueError("invalid_contract_scope_tags")
        if self.framework not in {"big_five", "disc"} and self.target_value is not None:
            raise ValueError("invalid_contract_target_value")
        return self


class _Response(_Object):
    protocol_version: Literal["2.0"]
    reviews: Annotated[list[_Review], Field(max_length=6)]
    changes: Annotated[list[_Change], Field(max_length=1)]


def reflection_response_schema() -> dict:
    """Return a fresh JSON Schema; provider support is not implied or probed.

    All properties are required (nullable fields still must be present) and
    objects forbid additional fields. Local validators and the prompt also
    cover cross-field rules not expressible by all provider schema subsets.
    A provider adapter must fail explicitly if it cannot honor this contract;
    it must not silently drop the schema or claim schema use for text-only calls.
    """
    from .diagnosis import _FRAMEWORK_DIMENSIONS, _SCOPE_TAGS

    schema = _Response.model_json_schema()
    change = schema["$defs"]["_Change"]
    change["properties"]["dimension"]["enum"] = sorted(set().union(*_FRAMEWORK_DIMENSIONS.values()))
    change["properties"]["scope_tags"]["items"]["enum"] = sorted(_SCOPE_TAGS)
    change["properties"]["scope_tags"]["uniqueItems"] = True
    # Event identity is updater-owned in this generation contract.
    schema["title"] = "AriaOfflineEvidenceReview2_5"
    return schema


def validate_reflection_response_shape(data: dict) -> None:
    """Validate the NEW generation contract locally, without altering input.

    This is stricter than legacy wire compatibility (at most one change). It
    validates structure only, NOT source truth, permissions or scope meaning.
    Existing wire replay should use the updater, not this new generation check.
    """
    _Response.model_validate(data)


class _ItemwiseEnvelope(_Object):
    """Candidate transaction envelope; each item is validated separately later."""

    protocol_version: Literal["2.0"]
    reviews: Annotated[list[object], Field(max_length=6)]
    # Preserve the archived updater's three-proposal compatibility and its
    # deterministic single-change selection. Generation still asks for <=1.
    changes: Annotated[list[object], Field(max_length=3)]


def validate_reflection_response_envelope(data: dict) -> None:
    _ItemwiseEnvelope.model_validate(data)


def validate_reflection_review_shape(data: object) -> None:
    _Review.model_validate(data)


def validate_reflection_change_shape(data: object) -> None:
    _Change.model_validate(data)
