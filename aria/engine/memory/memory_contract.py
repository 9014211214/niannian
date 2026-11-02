"""Mechanical shape of the explicit five-array memory generation contract.

Reception constrains types, keys and existing enums. The generation schema
also exposes existing preference capacities; reception remains itemwise for
those bounds. The extractor owns privacy, sources, scope and confirmation.
No field default supplies a missing statement or permission.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from .preference_sources import LEGACY, SOURCE_REFS

from ..model.persona import (
    PREFERENCE_DO_MAX_CHARACTERS, PREFERENCE_QUOTE_MAX_CHARACTERS,
    PREFERENCE_QUOTE_MAX_COUNT,
)

VERSION = "memory_json_schema_v1"
SOURCE_REFS_VERSION = "memory_source_refs_schema_v1"
LEGACY_FACT_SOURCE_REFS = "memory_fact_sources_v2"
FACT_SOURCE_REFS = "memory_fact_sources_v3"
FACT_PROTOCOLS = (LEGACY_FACT_SOURCE_REFS, FACT_SOURCE_REFS)
FIELDS = ("stated_preferences", "cognitive_privacy", "cognitive_observations", "episodes", "relationships")


class _Object(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _PreferenceQuote(_Object):
    turn_index: int
    text: str


class _SemanticReview(_Object):
    version: Literal["preference_semantics_v1"]
    kind: Literal["communication_style", "permission_boundary", "action_request", "fact", "person_inference", "unclear"]
    owner_requested: bool
    quote_supported: bool
    scope_preserved: bool
    no_permission_claim: bool


class _Preference(_Object):
    do: str
    scope: str
    quotes: list[_PreferenceQuote]
    confirmation: None
    supersedes: str | None
    revision_basis: str | None = None
    semantic_review: _SemanticReview

    @field_validator("revision_basis", mode="before")
    @classmethod
    def optional_revision_evidence(cls, value):
        # A malformed replacement basis cannot reject unrelated proposals.
        # The extractor still receives the raw value and drops supersedes.
        return value if isinstance(value, str) else None


class _Privacy(_Object):
    turn_index: int
    evidence_quote: str


class _PreferenceRefs(_Object):
    do: str
    scope: str
    source_turn_indices: list[int]
    confirmation: None
    supersedes: str | None
    revision_turn_index: int | None
    semantic_review: _SemanticReview


class _Observation(_Object):
    domain: Literal["memory", "language", "orientation", "daily_function", "repetition"]
    evidence_quote: str
    turn_index: int
    functional_impact: bool
    event_kind: Literal["new_event", "same_event"]
    prior_event_id: str | None
    signal: Literal["explicit_change", "context"] | None = None

    @field_validator("signal", mode="before")
    @classmethod
    def optional_semantic_signal(cls, value):
        # Extraction rejects the raw malformed item after privacy masking;
        # unrelated facts and observations must not fail with it.
        return value if value in ("explicit_change", "context") else None


class _Episode(_Object):
    summary: str
    keywords: list[str]
    referenceable: bool
    evidence_quote: str
    turn_index: int


class _RelationshipFact(_Object):
    value: str
    evidence_quote: str
    turn_index: int


class _Relationship(_Object):
    name: str
    relation: str
    facts: list[_RelationshipFact]
    referenceable: bool


class _Response(_Object):
    stated_preferences: list[_Preference]
    cognitive_privacy: list[_Privacy]
    cognitive_observations: list[_Observation]
    episodes: list[_Episode]
    relationships: list[_Relationship]


class _ResponseRefs(_Response):
    stated_preferences: list[_PreferenceRefs]


class _SelectedEpisode(_Object):
    source_id: str
    keywords: list[str]
    referenceable: bool
    preference_source_id: str | None


class _SelectedRelationship(_Object):
    name: str
    relation: str
    source_ids: list[str]
    referenceable: bool
    preference_source_id: str | None


class _ResponseFactRefs(_ResponseRefs):
    episodes: list[_SelectedEpisode]
    relationships: list[_SelectedRelationship]


class _ResponseRefsReception(_Response):
    # Per-item source failures belong to the extractor, after privacy review.
    # Do not discard unrelated proposals merely for an invalid reference type.
    stated_preferences: list[object]


class _FactRefsReception(_ResponseRefsReception):
    # Invalid source selections affect one item, after the privacy mask is known.
    episodes: list[object]
    relationships: list[object]


def memory_response_schema(*, source_protocol: str = LEGACY) -> dict:
    """Fresh schema; mutating a caller's copy cannot weaken a later request."""
    if source_protocol not in (LEGACY, SOURCE_REFS, *FACT_PROTOCOLS):
        raise ValueError("unknown_preference_source_protocol")
    model = {LEGACY: _Response, SOURCE_REFS: _ResponseRefs}.get(source_protocol, _ResponseFactRefs)
    schema = model.model_json_schema()
    if source_protocol == LEGACY_FACT_SOURCE_REFS:
        # The previously registered contract used global integer positions.
        schema["$defs"]["_SelectedEpisode"]["properties"]["source_id"]["type"] = "integer"
        schema["$defs"]["_SelectedRelationship"]["properties"]["source_ids"]["items"]["type"] = "integer"
        for name in ("_SelectedEpisode", "_SelectedRelationship"):
            schema["$defs"][name]["properties"].pop("preference_source_id")
            schema["$defs"][name]["required"].remove("preference_source_id")
    observation = schema["$defs"]["_Observation"]
    observation["properties"]["signal"] = {
        "type": "string", "enum": ["explicit_change", "context"],
    }
    observation["required"].append("signal")
    preference = schema["$defs"]["_Preference" if source_protocol == LEGACY else "_PreferenceRefs"]
    # Generation always requests an explicit nullable source. Local parsing
    # permits its absence so extraction can keep the independent proposal.
    if source_protocol == LEGACY:
        preference["properties"]["revision_basis"].pop("default", None)
        preference["required"].append("revision_basis")
    # Project existing storage/extraction bounds into generation only. Putting
    # these on _Response would reject unrelated valid rows on receipt.
    preference["properties"]["do"].update(minLength=1, maxLength=PREFERENCE_DO_MAX_CHARACTERS)
    if source_protocol != LEGACY:
        preference["properties"]["source_turn_indices"].update(
            minItems=1, maxItems=PREFERENCE_QUOTE_MAX_COUNT)
        preference["properties"]["source_turn_indices"]["items"]["minimum"] = 0
        preference["properties"]["revision_turn_index"]["anyOf"][0]["minimum"] = 0
    else:
        preference["properties"]["quotes"].update(minItems=1, maxItems=PREFERENCE_QUOTE_MAX_COUNT)
        quote = schema["$defs"]["_PreferenceQuote"]["properties"]
        quote["text"].update(minLength=1, maxLength=PREFERENCE_QUOTE_MAX_CHARACTERS)
        quote["turn_index"]["minimum"] = 0
    return schema


def validate_memory_response(raw: str, *, source_protocol: str = LEGACY) -> dict:
    """One complete JSON body; duplicate keys are never collapsed or repaired."""
    from .consolidator import _unique_object, _finite_float, _reject_constant
    if type(raw) is not str or not raw.strip() or len(raw.encode("utf-8")) > 262144:
        raise ValueError("memory_schema_text_required")
    # The legacy parser's fence/preamble compatibility is unchanged. This
    # explicit provider-schema path requires the complete JSON value itself.
    data = json.loads(raw, object_pairs_hook=_unique_object,
                      parse_float=_finite_float, parse_constant=_reject_constant)
    if source_protocol not in (LEGACY, SOURCE_REFS, *FACT_PROTOCOLS):
        raise ValueError("unknown_preference_source_protocol")
    model = {LEGACY: _Response, SOURCE_REFS: _ResponseRefsReception}.get(source_protocol, _FactRefsReception)
    model.model_validate(data)
    return data
