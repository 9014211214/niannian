"""Reconcile a sealed worker's staged changes with current owner state.

The caller holds the owner lock only while loading, merging and saving. Model
work never shares the active Engine profile. Concurrent owner changes win any
conflict; a deleted run or changed source invalidates the worker altogether.
"""
from __future__ import annotations

from copy import deepcopy

from .offline_queue import _OFFLINE_RESULT_FIELDS


class OfflineReviewCancelled(RuntimeError):
    """The original run/source no longer authorizes committing this result."""


def _merge_records(before, proposed, current):
    old = {row.id: row for row in before}
    new = {row.id: row for row in proposed}
    result = []
    present = {row.id for row in current}
    for row in current:
        original = old.get(row.id)
        updated = new.get(row.id)
        if original is None or row != original or updated == original:
            result.append(deepcopy(row))
        elif updated is not None:
            result.append(deepcopy(updated))
    for row in proposed:
        # Existing IDs deleted by the owner cannot be reintroduced.
        if row.id not in present and row.id not in old:
            result.append(deepcopy(row))
    return result


def merge_offline_stage(before, proposed, current, session_id):
    """Three-way merge one claimed session, retaining all live runtime fields."""
    prior_job = next((job for job in before.offline_sessions if job.session_id == session_id), None)
    live_job = next((job for job in current.offline_sessions if job.session_id == session_id), None)
    if prior_job is None or live_job is None:
        raise OfflineReviewCancelled("offline_source_removed")
    # C1 raw sources may be withdrawn while ordinary sealed text remains valid.
    for field in ("scene_id", "created_at", "expires_at", "turns", "excluded_turn_indices",
                  "source_completeness", "attempts", "state", "active_stage", "memory_completed",
                  "reflection_completed", "memory_privacy_review_completed"):
        if getattr(prior_job, field) != getattr(live_job, field):
            raise OfflineReviewCancelled("offline_source_changed")
    merged = current.model_copy(deep=True)
    for field in _OFFLINE_RESULT_FIELDS:
        original, updated, latest = (getattr(profile, field) for profile in (before, proposed, current))
        if updated == original:
            continue
        if isinstance(original, list) and all(hasattr(row, "id") for row in [*original, *updated, *latest]):
            setattr(merged, field, _merge_records(original, updated, latest))
        elif latest == original:
            setattr(merged, field, deepcopy(updated))

    # Completion and reflection counters are session-deduplicated independently
    # of current chat state. Online first-interaction metadata always survives.
    state = merged.person_model_state
    for ids_field, count_field in (("completed_session_ids", "sessions_completed"),
                                   ("reflected_session_ids", "sessions_observed")):
        prior_ids = getattr(before.person_model_state, ids_field)
        added = [value for value in getattr(proposed.person_model_state, ids_field)
                 if value not in prior_ids and value not in getattr(current.person_model_state, ids_field)]
        setattr(state, ids_field, list(dict.fromkeys([*getattr(current.person_model_state, ids_field), *added]))[-100:])
        setattr(state, count_field, getattr(current.person_model_state, count_field) + len(added))
    if before.person_model_state.last_reflected_at != proposed.person_model_state.last_reflected_at:
        state.last_reflected_at = max(value for value in (
            current.person_model_state.last_reflected_at, proposed.person_model_state.last_reflected_at) if value)

    cognitive_unchanged = (current.cognitive_health_consent == before.cognitive_health_consent
                          and current.cognitive_trend == before.cognitive_trend)
    if cognitive_unchanged and current.cognitive_health_consent.has_text_monitoring_consent():
        merged.cognitive_observations = _merge_records(before.cognitive_observations,
                                                      proposed.cognitive_observations, current.cognitive_observations)
        deleted = set(current.cognitive_trend.deleted_observation_ids)
        merged.cognitive_observations = [row for row in merged.cognitive_observations
            if row.id not in deleted and f"source:{row.session_id}:{row.source_turn_index}" not in deleted]
        merged.cognitive_trend = proposed.cognitive_trend.model_copy(deep=True)
        merged.cognitive_trend.deleted_observation_ids = list(current.cognitive_trend.deleted_observation_ids)

    updated_job = next((job for job in proposed.offline_sessions if job.session_id == session_id), None)
    jobs = []
    for job in current.offline_sessions:
        if job.session_id != session_id:
            jobs.append(deepcopy(job))
        elif updated_job is not None:
            value = updated_job.model_copy(deep=True)
            # A worker cannot restore withdrawn/deleted sensitive raw sources.
            value.cognitive_turns = deepcopy(job.cognitive_turns)
            value.cognitive_session_at = job.cognitive_session_at
            jobs.append(value)
    merged.offline_sessions = jobs
    added_audit = [row for row in proposed.offline_session_audit
                   if row.session_id == session_id and row not in before.offline_session_audit]
    merged.offline_session_audit = [*current.offline_session_audit, *deepcopy(added_audit)][-64:]
    # Never overwrite or remove the checkpoint of the next active conversation.
    return merged
