"""Durable FIFO for text-only offline work, with bounded replay and retention.

The owner must serialize writes to one user's profile and adopt ``queue.persona``
after every operation, including failed drains: an attempt may already be durably
recorded. The global lock protects shared offline component calls, not enqueue or
online checkpointing. Handlers must not persist their own candidate profile.
De-duplication is bounded by the existing 100 completed IDs and 64 audit records;
it is not a promise of permanent exactly-once processing across deleted history.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable

from ..context import EngineContext
from ..model.persona import (
    OfflineSessionAuditRecord, OfflineSessionRecord, PersonaProfile, SessionTurnSnapshot,
)
from ..perception.signals import is_simulated
from ..perception.external_emotion import validate_experiment_context
from .diagnosis import _REFLECTION_ERROR_DETAILS
from ..cognitive import cognitive_sources


OFFLINE_SESSION_LOCK = RLock()
OFFLINE_QUEUE_CAPACITY = 8
OFFLINE_MAX_ATTEMPTS = 3
OFFLINE_TTL_HOURS = 24
OFFLINE_MAX_TURNS = 120
OFFLINE_MAX_CHARACTERS = 60_000
OFFLINE_AUDIT_LIMIT = 64

_HANDLER_ERROR_CODES = _REFLECTION_ERROR_DETAILS | {
    "offline_handler_failed", "offline_handler_result_invalid",
    "offline_memory_handler_exception", "offline_reflection_handler_exception",
    "invalid_memory_payload", "invalid_memory_protocol", "memory_extraction_failed",
    "person_reflection_failed",
    "memory_privacy_unreviewed",
}
# A handler cannot overwrite the active checkpoint, queue, task state, consent,
# trust or other live runtime data while committing offline understanding.
_OFFLINE_RESULT_FIELDS = (
    "episodic_memory", "relationship_context", "person_hypotheses", "evidence_review_ledger",
    "stated_preferences",
    "person_model_state", "disc", "big_five", "attachment_style", "cognitive_profile",
    "case_conceptualization", "behavioral_markers", "emotional_baseline",
    "aria_strategy_hints", "delivery_state",
)


def _stamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse(value: str | None, *, legacy: bool = False) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            # Legacy checkpoints used local datetime.now(), not UTC timestamps.
            parsed = parsed.astimezone() if legacy else parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _safe_code(value, default="offline_handler_failed") -> str:
    return value if type(value) is str and value in _HANDLER_ERROR_CODES else default


def _simulated_context(ctx) -> bool:
    # Perception's *derived* acoustic audit always contains simulation_id=None,
    # even for real input. It is not a raw observation. Only strip that known
    # empty audit placeholder; raw perception_input keeps the stronger ingress
    # rule that even an explicitly supplied null simulation_id is synthetic.
    acoustic = dict(ctx.emotion.acoustic)
    fused_audit_fields = {"acoustic_label", "text_label", "fusion_strategy", "source_kind", "rejection_reasons"}
    if fused_audit_fields <= acoustic.keys() and acoustic.get("simulation_id") is None:
        acoustic.pop("simulation_id", None)
    # This new, separately validated classifier field is never sealed. All old
    # acoustic/multimodal simulation checks retain their original strictness.
    perception_input = ctx.working_memory.get("perception_input")
    if isinstance(perception_input, dict):
        perception_input = {k: v for k, v in perception_input.items() if k != "external_emotion"}
    return (
        ctx.simulation_mode or "_simulation_demo" in ctx.working_memory
        or is_simulated(perception_input) or is_simulated(acoustic)
        or str(ctx.emotion.inference_source).startswith("simulated_")
        or any(is_simulated({"source": item.source, "value": item.value}) for item in ctx.reasoning.evidence)
    )


class OfflineSessionQueue:
    def __init__(self, persona: PersonaProfile, store=None, now: Callable[[], datetime] | None = None):
        self.persona = persona
        self.store = store
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _clock(self) -> datetime:
        return _parse(_stamp(self._now()))

    def _commit(self, candidate: PersonaProfile) -> bool:
        try:
            if self.store is not None:
                self.store.save(candidate)
        except Exception:
            return False
        self.persona = candidate
        return True

    @staticmethod
    def _metadata(job: OfflineSessionRecord, now: datetime) -> dict:
        expires = _parse(job.expires_at)
        return {
            "session_id": job.session_id, "scene_id": job.scene_id,
            "created_at": job.created_at, "expires_at": job.expires_at,
            "source_completeness": job.source_completeness, "recovery_mode": "text_only",
            "turn_count": len(job.turns), "excluded_turn_count": len(job.excluded_turn_indices),
            "memory_completed": job.memory_completed, "reflection_completed": job.reflection_completed,
            "attempts": job.attempts, "state": job.state, "active_stage": job.active_stage,
            "expired": expires is None or expires <= now,
            "last_error_code": _safe_code(job.last_error_code) if job.last_error_code else None,
        }

    def status(self) -> dict:
        """Read metadata only; never renew TTL, clean up, persist or call a model."""
        now = self._clock()
        jobs = [self._metadata(job, now) for job in self.persona.offline_sessions]
        head = jobs[0] if jobs else None
        return {
            "status": "idle" if not head else "blocked" if (
                head["state"] in {"exhausted", "needs_review"} or head["attempts"] >= OFFLINE_MAX_ATTEMPTS
            ) else "pending",
            "pending_count": len(jobs),
            "exhausted_count": sum(job["state"] == "exhausted" or job["attempts"] >= OFFLINE_MAX_ATTEMPTS for job in jobs),
            "needs_review_count": sum(job["state"] == "needs_review" for job in jobs),
            "expired_count": sum(job["expired"] for job in jobs),
            "earliest_session_id": head["session_id"] if head else None,
            "jobs": jobs,
            "audit": [row.model_dump() for row in self.persona.offline_session_audit[-OFFLINE_AUDIT_LIMIT:]],
            "limits": {"capacity": OFFLINE_QUEUE_CAPACITY, "max_attempts": OFFLINE_MAX_ATTEMPTS,
                       "ttl_hours": OFFLINE_TTL_HOURS, "max_turns": OFFLINE_MAX_TURNS,
                       "max_characters": OFFLINE_MAX_CHARACTERS},
        }

    def _result(self, status: str, error_code=None, **values) -> dict:
        return {**self.status(), "status": status, "error_code": error_code, **values}

    @staticmethod
    def _clear_matching_checkpoint(persona, session_id):
        if persona.pending_session and persona.pending_session.session_id == session_id:
            persona.pending_session = None

    @staticmethod
    def _audit(persona, job, now, outcome, error_code=None):
        persona.offline_session_audit.append(OfflineSessionAuditRecord(
            session_id=job.session_id, scene_id=job.scene_id,
            created_at=job.created_at, expires_at=job.expires_at, finished_at=_stamp(now),
            outcome=outcome, source_completeness=job.source_completeness,
            turn_count=len(job.turns), excluded_turn_count=len(job.excluded_turn_indices),
            memory_completed=job.memory_completed, reflection_completed=job.reflection_completed,
            attempts=job.attempts, gap=outcome != "completed", error_code=error_code,
        ))
        persona.offline_session_audit = persona.offline_session_audit[-OFFLINE_AUDIT_LIMIT:]

    def _finish(self, candidate, job, now, *, outcome="completed", error_code=None):
        job.state, job.active_stage, job.last_error_code = "pending", None, None
        state = candidate.person_model_state
        if job.turns and job.session_id not in state.completed_session_ids:
            state.completed_session_ids.append(job.session_id)
            state.completed_session_ids = state.completed_session_ids[-100:]
            state.sessions_completed += 1
        self._audit(candidate, job, now, outcome, error_code)
        candidate.offline_sessions = [row for row in candidate.offline_sessions if row.session_id != job.session_id]
        self._clear_matching_checkpoint(candidate, job.session_id)

    def prune(self) -> dict:
        """Expire raw jobs in one save, preserving a bounded record of missing work."""
        now = self._clock()
        candidate = self.persona.model_copy(deep=True)
        expired = [job for job in candidate.offline_sessions
                   if _parse(job.expires_at) is None or _parse(job.expires_at) <= now]
        for job in expired:
            self._audit(candidate, job, now, "expired", "offline_session_expired")
            self._clear_matching_checkpoint(candidate, job.session_id)
        if not expired:
            return self._result("unchanged", expired_jobs=0)
        expired_ids = {job.session_id for job in expired}
        candidate.offline_sessions = [job for job in candidate.offline_sessions if job.session_id not in expired_ids]
        if not self._commit(candidate):
            return self._result("failed", "offline_store_failed", expired_jobs=0)
        return self._result("pruned", expired_jobs=len(expired))

    def enqueue(self, ctx, memory_completed=False, reflection_completed=False, source_completeness="complete") -> dict:
        external_input = (ctx.working_memory.get("perception_input") or {}).get("external_emotion")
        experiment_profile = self.persona.profile_origin.startswith("synthetic_emotion_experiment:")
        if (ctx.external_emotion_experiment is not None or experiment_profile
                or is_simulated(ctx.emotion.external_reading) or is_simulated(external_input)):
            if not validate_experiment_context(ctx, self.store, self.persona):
                return self._result("rejected", "offline_simulation_rejected")
        if _simulated_context(ctx):
            return self._result("rejected", "offline_simulation_rejected")
        if (ctx.user_id != self.persona.user_id or not isinstance(ctx.session_id, str)
                or not 1 <= len(ctx.session_id) <= 128 or not isinstance(ctx.scene_id, str)
                or not 1 <= len(ctx.scene_id) <= 128):
            return self._result("rejected", "offline_context_identity_invalid")
        if source_completeness not in {"complete", "legacy_tail_unknown_indices"}:
            return self._result("rejected", "offline_source_completeness_invalid")
        if type(memory_completed) is not bool or type(reflection_completed) is not bool:
            return self._result("rejected", "offline_stage_flags_invalid")
        if len(ctx.turns) > OFFLINE_MAX_TURNS:
            return self._result("rejected", "offline_turn_limit")
        if any(not isinstance(turn.user, str) or not isinstance(turn.agent, str) for turn in ctx.turns):
            return self._result("rejected", "offline_turn_text_invalid")
        if sum(len(turn.user) + len(turn.agent) for turn in ctx.turns) > OFFLINE_MAX_CHARACTERS:
            return self._result("rejected", "offline_character_limit")
        excluded = ctx.working_memory.get("_sensitive_memory_excluded_turns", [])
        if (not isinstance(excluded, (list, tuple, set)) or any(
                type(index) is not int or not 0 <= index < len(ctx.turns) for index in excluded)):
            return self._result("rejected", "offline_exclusion_mask_invalid")
        excluded = sorted(set(excluded))
        sealed_turns = [SessionTurnSnapshot(user="" if index in excluded else turn.user,
                                            agent="" if index in excluded else turn.agent)
                        for index, turn in enumerate(ctx.turns)]
        result = self.prune()
        if result["error_code"]:
            return result
        candidate = self.persona.model_copy(deep=True)
        existing = next((job for job in candidate.offline_sessions if job.session_id == ctx.session_id), None)
        if existing:
            if (existing.scene_id != ctx.scene_id or existing.turns != sealed_turns
                    or existing.excluded_turn_indices != excluded or existing.source_completeness != source_completeness):
                return self._result("rejected", "offline_session_already_sealed", session_id=ctx.session_id)
            self._clear_matching_checkpoint(candidate, ctx.session_id)
            if candidate.pending_session != self.persona.pending_session and not self._commit(candidate):
                return self._result("failed", "offline_store_failed", session_id=ctx.session_id)
            return self._result("existing", session_id=ctx.session_id)
        if (ctx.session_id in candidate.person_model_state.completed_session_ids
                or any(row.session_id == ctx.session_id and row.outcome == "completed"
                       for row in candidate.offline_session_audit)):
            self._clear_matching_checkpoint(candidate, ctx.session_id)
            if candidate.pending_session != self.persona.pending_session and not self._commit(candidate):
                return self._result("failed", "offline_store_failed", session_id=ctx.session_id)
            return self._result("completed", session_id=ctx.session_id)
        if any(row.session_id == ctx.session_id for row in candidate.offline_session_audit):
            return self._result("rejected", "offline_session_expired", session_id=ctx.session_id)
        if len(candidate.offline_sessions) >= OFFLINE_QUEUE_CAPACITY:
            return self._result("rejected", "offline_queue_full", session_id=ctx.session_id)
        now = self._clock()
        candidate.offline_sessions.append(OfflineSessionRecord(
            session_id=ctx.session_id, scene_id=ctx.scene_id, created_at=_stamp(now),
            expires_at=_stamp(now + timedelta(hours=OFFLINE_TTL_HOURS)), turns=sealed_turns,
            cognitive_turns=cognitive_sources(candidate, ctx),
            cognitive_session_at=(ctx.working_memory.get("_cognitive_session_at")
                                  if candidate.cognitive_health_consent.has_text_monitoring_consent() else None),
            excluded_turn_indices=excluded, source_completeness=source_completeness,
            memory_completed=memory_completed, reflection_completed=reflection_completed,
            memory_privacy_review_completed=(memory_completed
                and ctx.working_memory.get("_memory_privacy_review_completed") is True),
            state="pending" if source_completeness == "complete" else "needs_review",
        ))
        self._clear_matching_checkpoint(candidate, ctx.session_id)
        if not self._commit(candidate):
            return self._result("failed", "offline_store_failed", session_id=ctx.session_id)
        return self._result("queued", session_id=ctx.session_id)

    @staticmethod
    def _context(persona, job) -> EngineContext:
        ctx = EngineContext(user_id=persona.user_id, scene_id=job.scene_id, session_id=job.session_id)
        for turn in job.turns:
            ctx.add_turn(turn.user, turn.agent)
        ctx.working_memory.update({
            "_offline_recovery": True, "_offline_recovery_mode": "text_only",
            "_offline_source_completeness": job.source_completeness,
            "_sensitive_memory_excluded_turns": list(job.excluded_turn_indices),
            "_memory_consolidated": job.memory_completed, "_reflection_completed": job.reflection_completed,
            "_memory_privacy_review_completed": job.memory_privacy_review_completed,
        })
        if persona.cognitive_health_consent.has_text_monitoring_consent():
            ctx.working_memory.update({
                "_cognitive_source_turns": [row.model_dump() for row in job.cognitive_turns],
                "_cognitive_session_at": job.cognitive_session_at,
                "_cognitive_review_at": job.created_at,
            })
        return ctx

    def _merge_result(self, proposed, *, stage=None):
        candidate = self.persona.model_copy(deep=True)
        proposed = proposed.model_copy(deep=True)
        for field in _OFFLINE_RESULT_FIELDS:
            setattr(candidate, field, getattr(proposed, field))
        if stage == "memory" and candidate.cognitive_health_consent.has_text_monitoring_consent():
            candidate.cognitive_observations = proposed.cognitive_observations
            candidate.cognitive_trend = proposed.cognitive_trend
        # Queue completion, not handlers, owns these counters and de-dup IDs.
        candidate.person_model_state.sessions_completed = self.persona.person_model_state.sessions_completed
        candidate.person_model_state.completed_session_ids = list(self.persona.person_model_state.completed_session_ids)
        return candidate

    def drain(self, memory_handler, reflection_handler, max_jobs=1) -> dict:
        if type(max_jobs) is not int or not 1 <= max_jobs <= OFFLINE_QUEUE_CAPACITY:
            return self._result("rejected", "offline_job_limit_invalid", attempted_jobs=0, completed_jobs=0)
        cleanup = self.prune()
        if cleanup["error_code"]:
            return {**cleanup, "attempted_jobs": 0, "completed_jobs": 0}
        attempted = completed = 0
        for _ in range(max_jobs):
            if not self.persona.offline_sessions:
                break
            job = self.persona.offline_sessions[0]
            if job.memory_completed and job.reflection_completed:
                candidate = self.persona.model_copy(deep=True)
                self._finish(candidate, candidate.offline_sessions[0], self._clock())
                if not self._commit(candidate):
                    return self._result("failed", "offline_store_failed", attempted_jobs=attempted, completed_jobs=completed)
                completed += 1
                continue
            if job.source_completeness != "complete" or job.state == "needs_review":
                return self._result("blocked", "offline_source_needs_review", attempted_jobs=attempted, completed_jobs=completed)
            if job.attempts >= OFFLINE_MAX_ATTEMPTS:
                return self._result("blocked", "offline_retry_exhausted", attempted_jobs=attempted, completed_jobs=completed)
            # Save the attempt BEFORE calling an external component. A crash in
            # the call consumes this attempt rather than granting unlimited retries.
            candidate = self.persona.model_copy(deep=True)
            active = candidate.offline_sessions[0]
            active.attempts += 1
            active.state, active.last_attempt_at, active.last_error_code = "started", _stamp(self._clock()), None
            active.active_stage = "memory" if not active.memory_completed else "reflection"
            if not self._commit(candidate):
                return self._result("failed", "offline_store_failed", attempted_jobs=attempted, completed_jobs=completed)
            attempted += 1
            for stage, handler in (("memory", memory_handler), ("reflection", reflection_handler)):
                job = self.persona.offline_sessions[0]
                if getattr(job, stage + "_completed"):
                    continue
                copy_persona = self.persona.model_copy(deep=True)
                ctx = self._context(copy_persona, job)
                if stage == "reflection":
                    # Neither private raw sources nor extracted observations
                    # are available to the personality reviewer.
                    ctx.working_memory.pop("_cognitive_source_turns", None)
                    ctx.working_memory.pop("_cognitive_session_at", None)
                    copy_persona.cognitive_observations = []
                # Future jobs and a different active checkpoint are not input
                # evidence for this session, even if a future handler chooses
                # to serialize the whole profile. Queue state is merged only
                # from the committed owner profile after the call.
                copy_persona.offline_sessions = []
                copy_persona.offline_session_audit = []
                copy_persona.pending_session = None
                error = None
                try:
                    with OFFLINE_SESSION_LOCK:
                        result = handler(copy_persona, ctx)
                        if (not isinstance(result, tuple) or len(result) != 3
                                or not isinstance(result[0], PersonaProfile)
                                or result[0].user_id != self.persona.user_id
                                or type(result[1]) is not str or result[1] not in {"ok", "skipped", "failed", "sealed_failed"}):
                            error = "offline_handler_result_invalid"
                        elif result[1] == "failed":
                            error = _safe_code(result[2])
                        elif (stage == "memory" and result[1] in {"ok", "skipped"}
                              and ctx.working_memory.get("_memory_privacy_review_required")
                              and not ctx.working_memory.get("_memory_privacy_review_completed")):
                            error = "offline_memory_privacy_unreviewed"
                except Exception:
                    error = f"offline_{stage}_handler_exception"
                # Late results cannot outlive the fixed raw-data retention window.
                if _parse(job.expires_at) <= self._clock():
                    cleanup = self.prune()
                    return self._result("failed", cleanup["error_code"] or "offline_session_expired",
                                        attempted_jobs=attempted, completed_jobs=completed)
                if error:
                    candidate = self.persona.model_copy(deep=True)
                    active = candidate.offline_sessions[0]
                    active.state = "exhausted" if active.attempts >= OFFLINE_MAX_ATTEMPTS else "pending"
                    active.last_error_code, active.active_stage = error, stage
                    if not self._commit(candidate):
                        error = "offline_store_failed"
                    return self._result("failed", error, attempted_jobs=attempted, completed_jobs=completed)
                if result[1] == "sealed_failed":
                    # Discard the handler candidate entirely. A settled memory
                    # body failure also seals this session without reflection:
                    # its privacy classification was not established. No stage
                    # is falsely marked complete and future sessions can drain.
                    # Reflection failure preserves previously committed memory.
                    candidate = self.persona.model_copy(deep=True)
                    active = candidate.offline_sessions[0]
                    self._finish(candidate, active, self._clock(), outcome=stage + "_failed",
                                 error_code=_safe_code(result[2]))
                    if not self._commit(candidate):
                        return self._result("failed", "offline_store_failed", attempted_jobs=attempted,
                                            completed_jobs=completed)
                    completed += 1
                    break
                candidate = self._merge_result(result[0], stage=stage)
                active = candidate.offline_sessions[0]
                if stage == "memory":
                    # The semantic privacy decision and extracted memory share
                    # the same durable commit. Reflection reconstructs the
                    # original conversation using this persisted exclusion map.
                    mask = ctx.working_memory.get("_sensitive_memory_excluded_turns", [])
                    if (not isinstance(mask, (list, tuple, set)) or any(
                            type(index) is not int or not 0 <= index < len(active.turns) for index in mask)):
                        return self._result("failed", "offline_exclusion_mask_invalid",
                                            attempted_jobs=attempted, completed_jobs=completed)
                    active.excluded_turn_indices = sorted(set(active.excluded_turn_indices) | set(mask))
                    for index in active.excluded_turn_indices:
                        active.turns[index] = SessionTurnSnapshot(user="", agent="")
                    active.memory_privacy_review_completed = (
                        ctx.working_memory.get("_memory_privacy_review_completed") is True)
                setattr(active, stage + "_completed", True)
                active.last_error_code = None
                if active.memory_completed and active.reflection_completed:
                    self._finish(candidate, active, self._clock())
                else:
                    # This same atomic save checkpoints memory and marks the next
                    # stage started before the next external call.
                    active.state, active.active_stage = "started", "reflection"
                if not self._commit(candidate):
                    return self._result("failed", "offline_store_failed", attempted_jobs=attempted, completed_jobs=completed)
                if not any(row.session_id == job.session_id for row in self.persona.offline_sessions):
                    completed += 1
                    break
        return self._result("completed" if completed else "idle" if not self.persona.offline_sessions else "pending",
                            attempted_jobs=attempted, completed_jobs=completed)

    def migrate_legacy_pending(self) -> dict:
        cleanup = self.prune()
        if cleanup["error_code"]:
            return cleanup
        checkpoint = self.persona.pending_session
        if checkpoint is None or not checkpoint.offline_finalize_pending:
            return self._result("unchanged")
        candidate = self.persona.model_copy(deep=True)
        session_id = checkpoint.session_id
        previous = next((row for row in reversed(candidate.offline_session_audit) if row.session_id == session_id), None)
        if (session_id in candidate.person_model_state.completed_session_ids
                or any(job.session_id == session_id for job in candidate.offline_sessions) or previous is not None):
            self._clear_matching_checkpoint(candidate, session_id)
            if not self._commit(candidate):
                return self._result("failed", "offline_store_failed", session_id=session_id)
            if session_id in candidate.person_model_state.completed_session_ids or (previous and previous.outcome == "completed"):
                return self._result("completed", session_id=session_id)
            if previous:
                return self._result("expired", "offline_session_expired", session_id=session_id)
            return self._result("existing", session_id=session_id)
        now = self._clock()
        created = _parse(checkpoint.updated_at, legacy=True)
        expires = _parse(checkpoint.expires_at, legacy=True)
        if created is None:
            # Unknown legacy clocks cannot earn a new retention window.
            created = datetime(1970, 1, 1, tzinfo=timezone.utc)
        deadline = created + timedelta(hours=OFFLINE_TTL_HOURS)
        expires = min(expires, deadline) if expires is not None else deadline
        job = OfflineSessionRecord(
            session_id=session_id, scene_id=checkpoint.scene_id, created_at=_stamp(created), expires_at=_stamp(expires),
            turns=[turn.model_copy(deep=True) for turn in checkpoint.turns],
            source_completeness="legacy_tail_unknown_indices", state="needs_review",
            memory_completed=checkpoint.memory_consolidated, reflection_completed=checkpoint.reflection_completed,
        )
        oversized = len(job.turns) > OFFLINE_MAX_TURNS or sum(len(t.user) + len(t.agent) for t in job.turns) > OFFLINE_MAX_CHARACTERS
        if expires <= now or oversized:
            self._audit(candidate, job, now, "expired" if expires <= now else "legacy_rejected",
                        "offline_session_expired" if expires <= now else "offline_legacy_bounds_exceeded")
            self._clear_matching_checkpoint(candidate, session_id)
            if not self._commit(candidate):
                return self._result("failed", "offline_store_failed", session_id=session_id)
            return self._result("expired" if expires <= now else "rejected",
                                "offline_session_expired" if expires <= now else "offline_legacy_bounds_exceeded", session_id=session_id)
        if len(candidate.offline_sessions) >= OFFLINE_QUEUE_CAPACITY:
            return self._result("rejected", "offline_queue_full", session_id=session_id)
        candidate.offline_sessions.append(job)
        self._clear_matching_checkpoint(candidate, session_id)
        if not self._commit(candidate):
            return self._result("failed", "offline_store_failed", session_id=session_id)
        return self._result("queued", session_id=session_id)
