"""Complete owner-source fixtures for software permission/recall tests only."""
from aria.engine.memory.referenceability import SOURCE_CONTEXT_VERSION, source_context_sha256
from aria.engine.model.persona import RelationshipEvidence, RelationshipFactSource


def episode_source(text, *, session_id="source-fixture", turn_index=0):
    return dict(source_session_id=session_id, source_turn_index=turn_index, source_quote=text,
                source_context=text, source_context_sha256=source_context_sha256(text),
                source_context_version=SOURCE_CONTEXT_VERSION)


def relationship_evidence(quote, *, context=None, session_id="source-fixture", turn_index=0):
    text = context if context is not None else quote
    return [RelationshipEvidence(
        at="2026-09-01T08:00:00", session_id=session_id, summary=quote, turn_indices=[turn_index],
        fact_sources=[RelationshipFactSource(
            quote=quote, turn_index=turn_index, source_context=text,
            source_context_sha256=source_context_sha256(text), source_context_version=SOURCE_CONTEXT_VERSION,
        )],
    )]
