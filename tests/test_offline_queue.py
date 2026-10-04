"""No-network queue persistence, provenance, retry and lifecycle contracts."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from threading import Event, Thread

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.offline_queue import (
    OFFLINE_AUDIT_LIMIT, OFFLINE_MAX_ATTEMPTS, OFFLINE_SESSION_LOCK,
    OfflineSessionQueue,
)
from aria.engine.model.persona import (
    EpisodicMemoryEntry, OfflineSessionRecord, PersonaProfile,
    SessionCheckpointRecord, SessionTurnSnapshot,
)
from aria.interfaces.reasoning import ObservationEvidence


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 12, 8, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


class Store:
    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.saved = None
        self.snapshots = []

    def save(self, persona):
        self.calls += 1
        if self.calls in self.fail_on:
            raise OSError("PRIVATE_RAW_SENTENCE must not become a diagnostic")
        self.saved = persona.model_copy(deep=True)
        self.snapshots.append(self.saved.model_dump())


def context(session="s1", *, words=None, user="u"):
    ctx = EngineContext(user_id=user, scene_id="companion_elder", session_id=session)
    for word in words if words is not None else ["这是本次新操作。", "我喜欢一次只说一步。", "让我自己试完再继续。"]:
        ctx.add_turn(word, "我听到了。")
    return ctx


def setup(*, fail_on=()):
    clock, store = Clock(), Store(fail_on)
    return OfflineSessionQueue(PersonaProfile(user_id="u"), store, clock), store, clock


def memory(persona, ctx):
    persona.episodic_memory.append(EpisodicMemoryEntry(id=ctx.session_id, summary="一次教学节奏交流"))
    return persona, "ok", None


def reflection(persona, ctx):
    persona.person_model_state.reflected_session_ids.append(ctx.session_id)
    persona.person_model_state.sessions_observed += 1
    return persona, "ok", None


def skipped(persona, ctx):
    return persona, "skipped", None


def failed(persona, ctx):
    persona.disc.dominance = 0.99
    persona.episodic_memory.append(EpisodicMemoryEntry(id="must-not-commit", summary="PRIVATE_RAW_SENTENCE"))
    return persona, "failed", "person_reflection_failed"


def checkpoint(session, clock):
    return SessionCheckpointRecord(
        session_id=session, scene_id="companion_elder", updated_at=clock().isoformat(),
        expires_at=(clock() + timedelta(hours=24)).isoformat(),
        turns=[SessionTurnSnapshot(user="active checkpoint", agent="reply")],
    )


def test_enqueue_seals_copy_and_recovery_preserves_redacted_indices_without_modalities():
    queue, store, clock = setup()
    ctx = context(words=["安全前文", "PRIVATE_RAW_SENTENCE", "真正偏好在原编号2"])
    ctx.turns[1].agent = "PRIVATE_RAW_SENTENCE mirrored by assistant"
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    ctx.working_memory["secret_online_state"] = "PRIVATE_RAW_SENTENCE"
    ctx.reasoning.evidence.append(ObservationEvidence("camera", "visual", "face", "raw-observation"))
    ctx.emotion.acoustic = {"label": "sad", "provider": "real_audio"}
    assert queue.enqueue(ctx)["status"] == "queued"
    ctx.turns[2].user = "changed after sealing"
    assert "PRIVATE_RAW_SENTENCE" not in json.dumps(store.snapshots)
    restored = []
    def capture(persona, recovered):
        restored.append(deepcopy(recovered))
        return persona, "skipped", None
    result = queue.drain(capture, capture)
    assert result["status"] == "completed" and result["pending_count"] == 0
    assert [t.user for t in restored[0].turns] == ["安全前文", "", "真正偏好在原编号2"]
    assert restored[0].turns[1].agent == ""
    assert restored[0].working_memory["_sensitive_memory_excluded_turns"] == [1]
    assert restored[0].working_memory["_offline_recovery"] is True
    assert restored[0].working_memory["_offline_recovery_mode"] == "text_only"
    assert "secret_online_state" not in restored[0].working_memory
    assert not restored[0].reasoning.evidence and not restored[0].emotion.acoustic
    assert not queue.persona.offline_sessions
    assert queue.persona.person_model_state.sessions_completed == 1
    assert queue.persona.person_model_state.sessions_observed == 0
    assert "真正偏好" not in json.dumps(queue.status(), ensure_ascii=False)


def test_different_active_checkpoint_is_preserved_even_if_successful_handler_mutates_it():
    queue, store, clock = setup()
    queue.persona.pending_session = checkpoint("active-other", clock)
    original = queue.persona.pending_session.model_dump()
    queue.persona.alliance.trust = 0.6
    queue.enqueue(context())
    def overwrite(persona, ctx):
        persona.pending_session = None
        persona.offline_sessions.clear()
        persona.alliance.trust = 0.99
        persona.person_model_state.sessions_completed = 999
        return memory(persona, ctx)
    queue.drain(overwrite, skipped)
    assert queue.persona.pending_session.model_dump() == original
    assert queue.persona.alliance.trust == 0.6
    assert queue.persona.person_model_state.sessions_completed == 1


def test_handler_never_receives_other_jobs_or_active_checkpoint_raw_text():
    queue, store, clock = setup()
    queue.persona.pending_session = checkpoint("active-other", clock)
    queue.persona.pending_session.turns[0].user = "ACTIVE_PRIVATE_TEXT"
    queue.enqueue(context("head"))
    queue.enqueue(context("later", words=["FUTURE_PRIVATE_TEXT"]))
    seen = []
    def isolated(persona, ctx):
        dumped = persona.model_dump_json()
        assert "ACTIVE_PRIVATE_TEXT" not in dumped and "FUTURE_PRIVATE_TEXT" not in dumped
        assert persona.pending_session is None
        assert persona.offline_sessions == [] and persona.offline_session_audit == []
        assert ctx.session_id == "head"
        seen.append(ctx.session_id)
        return persona, "skipped", None
    assert queue.drain(isolated, isolated)["completed_jobs"] == 1
    assert seen == ["head", "head"]
    assert queue.persona.pending_session.turns[0].user == "ACTIVE_PRIVATE_TEXT"
    assert queue.persona.offline_sessions[0].turns[0].user == "FUTURE_PRIVATE_TEXT"


def test_matching_checkpoint_clears_only_after_enqueue_commit_and_existing_does_not_renew():
    queue, store, clock = setup(fail_on={1})
    ctx = context()
    queue.persona.pending_session = checkpoint(ctx.session_id, clock)
    original = queue.persona
    result = queue.enqueue(ctx)
    assert result["error_code"] == "offline_store_failed" and queue.persona is original
    assert queue.persona.pending_session is not None and not queue.persona.offline_sessions
    assert queue.enqueue(ctx)["status"] == "queued"
    assert queue.persona.pending_session is None
    first_deadline = queue.persona.offline_sessions[0].expires_at
    clock.advance(hours=3)
    assert queue.enqueue(ctx, memory_completed=True, reflection_completed=True)["status"] == "existing"
    assert queue.persona.offline_sessions[0].expires_at == first_deadline
    assert not queue.persona.offline_sessions[0].memory_completed
    ctx.add_turn("new text after sealing", "reply")
    assert queue.enqueue(ctx)["error_code"] == "offline_session_already_sealed"


def test_failure_then_recovery_resumes_only_missing_stage_and_fifo_stops_on_failure():
    queue, store, clock = setup()
    queue.enqueue(context("a"))
    queue.enqueue(context("b"))
    calls = []
    def reflect_fail(persona, ctx):
        calls.append(ctx.session_id)
        return failed(persona, ctx)
    result = queue.drain(memory, reflect_fail, max_jobs=8)
    assert result["status"] == "failed" and result["attempted_jobs"] == 1
    assert calls == ["a"] and queue.persona.offline_sessions[0].memory_completed
    assert [e.id for e in queue.persona.episodic_memory] == ["a"]
    assert queue.persona.disc.dominance == 0.5
    reloaded = OfflineSessionQueue(store.saved, store, clock)
    result = reloaded.drain(memory, reflection)
    assert result["completed_jobs"] == 1 and result["pending_count"] == 1
    assert [e.id for e in reloaded.persona.episodic_memory] == ["a"]
    assert reloaded.persona.offline_session_audit[-1].attempts == 2
    reloaded.drain(memory, reflection)
    assert [e.id for e in reloaded.persona.episodic_memory] == ["a", "b"]
    assert reloaded.persona.person_model_state.completed_session_ids == ["a", "b"]
    assert reloaded.enqueue(context("a"))["status"] == "completed"
    assert reloaded.persona.person_model_state.sessions_completed == 2


@pytest.mark.parametrize(("failure_save", "memory_done", "memory_entries"), [(2, False, 0), (3, False, 0), (4, True, 1)])
def test_attempt_and_each_stage_commit_are_atomic(failure_save, memory_done, memory_entries):
    queue, store, clock = setup(fail_on={failure_save})
    queue.enqueue(context())
    result = queue.drain(memory, reflection)
    assert result["error_code"] == "offline_store_failed"
    assert queue.persona.model_dump() == store.saved.model_dump()
    job = queue.persona.offline_sessions[0]
    assert job.memory_completed == memory_done and not job.reflection_completed
    assert len(queue.persona.episodic_memory) == memory_entries
    assert queue.persona.person_model_state.sessions_completed == 0
    assert not queue.persona.person_model_state.reflected_session_ids
    assert job.attempts == (0 if failure_save == 2 else 1)
    result = queue.drain(memory, reflection)
    assert result["completed_jobs"] == 1
    assert len(queue.persona.episodic_memory) == 1
    assert queue.persona.person_model_state.sessions_completed == 1


@pytest.mark.parametrize(("memory_done", "reflection_done", "expected_calls"), [
    (True, False, ["reflection"]), (False, True, ["memory"]), (True, True, []),
])
def test_precompleted_stage_flags_are_honored_without_repeating_work(memory_done, reflection_done, expected_calls):
    queue, store, clock = setup()
    queue.enqueue(context(), memory_completed=memory_done, reflection_completed=reflection_done)
    calls = []
    def stage(name):
        def handle(persona, ctx):
            calls.append(name)
            assert ctx.working_memory["_memory_consolidated"] == memory_done
            assert ctx.working_memory["_reflection_completed"] == reflection_done
            return persona, "skipped", None
        return handle
    result = queue.drain(stage("memory"), stage("reflection"))
    assert calls == expected_calls and result["completed_jobs"] == 1
    assert queue.persona.person_model_state.sessions_completed == 1
    assert queue.persona.person_model_state.sessions_observed == 0
    assert queue.persona.offline_session_audit[-1].attempts == (1 if expected_calls else 0)


def test_failure_record_save_failure_keeps_precharged_attempt_and_no_handler_mutation():
    queue, store, clock = setup(fail_on={3})
    queue.enqueue(context())
    result = queue.drain(failed, reflection)
    assert result["error_code"] == "offline_store_failed"
    assert queue.persona.offline_sessions[0].state == "started"
    assert queue.persona.offline_sessions[0].attempts == 1
    assert not queue.persona.episodic_memory and queue.persona.disc.dominance == 0.5
    assert queue.persona.model_dump() == store.saved.model_dump()
    assert queue.drain(memory, reflection)["completed_jobs"] == 1


def test_three_total_attempts_exhaust_head_until_original_ttl_then_clear_raw_gap():
    queue, store, clock = setup()
    queue.enqueue(context("a"))
    clock.advance(hours=1)
    queue.enqueue(context("b"))
    deadline = queue.persona.offline_sessions[0].expires_at
    for count in range(1, OFFLINE_MAX_ATTEMPTS + 1):
        assert queue.drain(failed, reflection, max_jobs=8)["status"] == "failed"
        assert queue.persona.offline_sessions[0].attempts == count
        assert queue.persona.offline_sessions[0].expires_at == deadline
    assert queue.persona.offline_sessions[0].state == "exhausted"
    assert queue.drain(memory, reflection)["error_code"] == "offline_retry_exhausted"
    assert queue.status()["exhausted_count"] == 1
    clock.advance(hours=23)
    assert queue.prune()["expired_jobs"] == 1
    assert queue.status()["earliest_session_id"] == "b"
    audit = queue.persona.offline_session_audit[-1]
    assert audit.session_id == "a" and audit.outcome == "expired" and audit.gap
    assert not hasattr(audit, "turns")
    assert queue.enqueue(context("a"))["error_code"] == "offline_session_expired"
    assert queue.drain(memory, reflection)["completed_jobs"] == 1


def test_crash_before_result_cannot_reset_attempt_limit():
    queue, store, clock = setup()
    queue.enqueue(context())
    def crash(persona, ctx):
        raise KeyboardInterrupt()
    for attempt in range(1, 4):
        with pytest.raises(KeyboardInterrupt):
            queue.drain(crash, reflection)
        assert store.saved.offline_sessions[0].attempts == attempt
        queue = OfflineSessionQueue(store.saved, store, clock)
    assert queue.drain(crash, reflection)["error_code"] == "offline_retry_exhausted"
    assert queue.status()["exhausted_count"] == 1


def test_late_model_result_does_not_extend_ttl_or_commit_derived_mutations():
    queue, store, clock = setup()
    queue.enqueue(context())
    def late(persona, ctx):
        clock.advance(hours=24)
        return memory(persona, ctx)
    assert queue.drain(late, reflection)["error_code"] == "offline_session_expired"
    assert not queue.persona.offline_sessions and not queue.persona.episodic_memory
    assert queue.persona.offline_session_audit[-1].gap


def test_failed_expiry_save_does_not_claim_raw_was_removed():
    queue, store, clock = setup(fail_on={2})
    queue.enqueue(context())
    clock.advance(hours=24)
    before = queue.persona
    result = queue.prune()
    assert result["error_code"] == "offline_store_failed" and result["expired_jobs"] == 0
    assert queue.persona is before and queue.status()["expired_count"] == 1
    assert queue.persona.offline_sessions[0].turns
    assert queue.prune()["expired_jobs"] == 1
    assert not queue.persona.offline_sessions


def test_capacity_and_complete_source_bounds_are_explicit_rejections():
    queue, store, clock = setup()
    for index in range(8):
        assert queue.enqueue(context(f"s{index}"))["status"] == "queued"
    assert queue.enqueue(context("ninth"))["error_code"] == "offline_queue_full"
    assert len(queue.persona.offline_sessions) == 8
    fresh, _, _ = setup()
    assert fresh.enqueue(context(words=["x"] * 121))["error_code"] == "offline_turn_limit"
    assert fresh.enqueue(context(words=["x" * 60_001]))["error_code"] == "offline_character_limit"
    ctx = context(words=["x"] * 120)
    assert fresh.enqueue(ctx)["status"] == "queued"
    assert len(fresh.persona.offline_sessions[0].turns) == 120
    exact, _, _ = setup()
    ctx = context(words=["x" * 60_000])
    ctx.turns[0].agent = ""
    assert exact.enqueue(ctx)["status"] == "queued"


@pytest.mark.parametrize("source", ["sticky", "demo", "acoustic", "evidence"])
def test_simulation_never_enters_real_queue(source):
    queue, store, clock = setup()
    ctx = context()
    if source == "sticky":
        ctx.simulation_mode = True
    elif source == "demo":
        ctx.working_memory["_simulation_demo"] = {}
    elif source == "acoustic":
        ctx.emotion.acoustic = {"source_kind": "simulated"}
    else:
        ctx.reasoning.evidence.append(ObservationEvidence("x", "acoustic", "emotion", "sad", source="simulated_acoustic"))
    assert queue.enqueue(ctx)["error_code"] == "offline_simulation_rejected"
    assert store.calls == 0 and not queue.persona.offline_sessions


def test_real_fusion_audit_null_simulation_id_is_not_confused_with_raw_synthetic_input():
    queue, store, clock = setup()
    ctx = context()
    ctx.emotion.acoustic = {
        "acoustic_label": "neutral", "text_label": "neutral", "fusion_strategy": "weighted",
        "source_kind": "unspecified", "rejection_reasons": [], "simulation_id": None,
    }
    ctx.working_memory["perception_fusion"] = {"acoustic": dict(ctx.emotion.acoustic)}
    assert queue.enqueue(ctx)["status"] == "queued"
    for raw_location in ("perception_input", "acoustic"):
        fresh, _, _ = setup()
        ctx = context()
        if raw_location == "perception_input":
            ctx.working_memory["perception_input"] = {"acoustic": {"simulation_id": None}}
        else:
            ctx.emotion.acoustic = {"simulation_id": None}
        assert fresh.enqueue(ctx)["error_code"] == "offline_simulation_rejected"


@pytest.mark.parametrize("mask", [[True], [-1], [3], "1"])
def test_invalid_exclusion_mask_fails_closed(mask):
    queue, store, clock = setup()
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = mask
    assert queue.enqueue(ctx)["error_code"] == "offline_exclusion_mask_invalid"
    assert store.calls == 0


def test_handler_exception_and_unknown_returned_error_do_not_expose_text():
    queue, store, clock = setup()
    queue.enqueue(context())
    def explode(persona, ctx):
        raise RuntimeError("PRIVATE_RAW_SENTENCE")
    result = queue.drain(explode, reflection)
    assert result["error_code"] == "offline_memory_handler_exception"
    assert "PRIVATE_RAW_SENTENCE" not in json.dumps(queue.status())
    def unknown(persona, ctx):
        return persona, "failed", "PRIVATE_RAW_SENTENCE"
    assert queue.drain(unknown, reflection)["error_code"] == "offline_handler_failed"


@pytest.mark.parametrize("malformed", ["not_tuple", "wrong_user", "unknown_status", "non_string_status"])
def test_invalid_handler_contract_fails_without_committing_candidate(malformed):
    queue, store, clock = setup()
    queue.enqueue(context())
    def handler(persona, ctx):
        persona.disc.dominance = 0.99
        if malformed == "not_tuple":
            return {"secret": "PRIVATE_RAW_SENTENCE"}
        if malformed == "wrong_user":
            persona.user_id = "another-user"
        return persona, (["PRIVATE_RAW_SENTENCE"] if malformed == "non_string_status"
                         else "unrecognized" if malformed == "unknown_status" else "ok"), None
    result = queue.drain(handler, reflection)
    assert result["error_code"] == "offline_handler_result_invalid"
    assert queue.persona.user_id == "u" and queue.persona.disc.dominance == 0.5
    assert "PRIVATE_RAW_SENTENCE" not in json.dumps(queue.status())


def test_empty_session_completes_without_inflating_completed_session_count():
    queue, store, clock = setup()
    ctx = context(words=[])
    queue.enqueue(ctx)
    queue.drain(skipped, skipped)
    assert queue.persona.person_model_state.sessions_completed == 0
    assert queue.persona.offline_session_audit[-1].outcome == "completed"
    assert queue.enqueue(ctx)["status"] == "completed"


def test_legacy_pending_has_unknown_indices_preserves_flags_and_original_expiry():
    queue, store, clock = setup()
    queue.persona.pending_session = checkpoint("legacy", clock)
    queue.persona.pending_session.updated_at = (clock() - timedelta(hours=10)).isoformat()
    queue.persona.pending_session.expires_at = (clock() + timedelta(hours=2)).isoformat()
    queue.persona.pending_session.offline_finalize_pending = True
    queue.persona.pending_session.memory_consolidated = True
    assert queue.migrate_legacy_pending()["status"] == "queued"
    job = queue.persona.offline_sessions[0]
    assert job.source_completeness == "legacy_tail_unknown_indices" and job.state == "needs_review"
    assert job.memory_completed and not job.reflection_completed
    assert datetime.fromisoformat(job.expires_at) == clock() + timedelta(hours=2)
    assert queue.persona.pending_session is None
    def no_call(persona, ctx):
        pytest.fail("Legacy unknown turn indices must not be used for automatic reflection")
    assert queue.drain(no_call, no_call)["error_code"] == "offline_source_needs_review"
    assert queue.persona.offline_sessions[0].attempts == 0
    clock.advance(hours=2)
    assert queue.prune()["expired_jobs"] == 1
    assert queue.persona.offline_session_audit[-1].source_completeness == "legacy_tail_unknown_indices"


def test_legacy_expired_missing_clock_or_overlong_raw_is_never_given_new_ttl():
    queue, store, clock = setup()
    queue.persona.pending_session = checkpoint("legacy", clock)
    queue.persona.pending_session.offline_finalize_pending = True
    queue.persona.pending_session.updated_at = "unknown"
    assert queue.migrate_legacy_pending()["status"] == "expired"
    assert not queue.persona.offline_sessions and queue.persona.pending_session is None
    queue.persona.pending_session = checkpoint("oversized", clock)
    queue.persona.pending_session.offline_finalize_pending = True
    queue.persona.pending_session.turns = [SessionTurnSnapshot(user="x", agent="y")] * 121
    assert queue.migrate_legacy_pending()["error_code"] == "offline_legacy_bounds_exceeded"
    assert not queue.persona.offline_sessions and queue.persona.pending_session is None


def test_legacy_migration_does_not_touch_active_checkpoint_and_failed_save_keeps_raw():
    queue, store, clock = setup(fail_on={1})
    queue.persona.pending_session = checkpoint("legacy", clock)
    assert queue.migrate_legacy_pending()["status"] == "unchanged"
    queue.persona.pending_session.offline_finalize_pending = True
    before = queue.persona.model_dump()
    assert queue.migrate_legacy_pending()["error_code"] == "offline_store_failed"
    assert queue.persona.model_dump() == before


def test_prune_audit_and_completed_ids_remain_bounded():
    queue, store, clock = setup()
    queue.persona.person_model_state.completed_session_ids = [f"previous-{i}" for i in range(100)]
    for index in range(OFFLINE_AUDIT_LIMIT + 2):
        queue.enqueue(context(f"s{index}"))
        queue.drain(skipped, skipped)
    assert len(queue.persona.offline_session_audit) == OFFLINE_AUDIT_LIMIT
    assert len(queue.persona.person_model_state.completed_session_ids) == 100
    before = store.calls
    queue.status()
    assert store.calls == before


def test_enqueue_and_status_do_not_wait_for_other_users_global_handler_lock():
    queue, store, clock = setup()
    entered, release, done = Event(), Event(), Event()
    def holder():
        with OFFLINE_SESSION_LOCK:
            entered.set()
            release.wait(3)
    def seal():
        queue.enqueue(context())
        queue.status()
        queue.prune()
        done.set()
    holding = Thread(target=holder, daemon=True)
    holding.start()
    assert entered.wait(1)
    worker = Thread(target=seal, daemon=True)
    worker.start()
    try:
        assert done.wait(1), "A different user's model call must not block sealing"
    finally:
        release.set()
        worker.join(3)
        holding.join(3)


def test_old_profiles_load_with_empty_queue_defaults():
    persona = PersonaProfile.model_validate({"user_id": "old"})
    assert persona.offline_sessions == [] and persona.offline_session_audit == []
    assert OfflineSessionRecord.model_fields["recovery_mode"].default == "text_only"
