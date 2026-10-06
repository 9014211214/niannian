"""Real Web/Engine/queue integration with local models only, no network calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import threading

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.offline_queue import (
    OFFLINE_MAX_ATTEMPTS, OFFLINE_QUEUE_CAPACITY, OFFLINE_TTL_HOURS, OfflineSessionQueue,
)
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import (
    PersonaProfile, PersistentTaskRecord, SessionCheckpointRecord, SessionTurnSnapshot,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.llm import MockLLMClient
from aria.voice.web import TextWebApp
from aria.runtime import MinimalModelClients
from tests.test_model_routing import _DECISION_OK, _ProtocolAwareFast


class LocalModel:
    model = "synthetic/local-only"

    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def complete(self, messages):
        self.messages.append(deepcopy(messages))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


GOOD_REFLECTION = '{"protocol_version":"2.0","reviews":[],"changes":[]}'


def _app(store, *, failing=True, reflection=None, checkpoint_turns=1):
    memory = LocalModel('{"episodes":[],"relationships":[],"cognitive_privacy":[]}')
    reflection = reflection or LocalModel(RuntimeError("PRIVATE_PROVIDER_FAILURE") if failing else GOOD_REFLECTION)
    fast = _ProtocolAwareFast(_DECISION_OK)
    app = TextWebApp(store=store, clients=MinimalModelClients(fast, memory, reflection, True),
                     checkpoint_turns=checkpoint_turns, session_idle_seconds=10)
    original_factory = app._engine_for_persona
    def fixture_engine(persona):
        engine = original_factory(persona)
        engine.hypothesis_updater = PersonaHypothesisUpdater(reflection, min_turns=1, evidence_review_protocol=True)
        return engine
    app._engine_for_persona = fixture_engine
    return app, fast, memory, reflection


def _failed_close(app, user_id="person"):
    assert app.text_turn(user_id, "OLD_ONLY 我愿意按自己的节奏聊。")["ok"]
    old_id = app._sessions[user_id][1].session_id
    result = app.end(user_id, process_offline=True)
    assert result["ok"] and result["offline_processing_pending"]
    assert user_id not in app._sessions
    return old_id, result


def test_old_failure_and_new_chat_have_distinct_ids_and_no_online_offline_call(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    old_id, result = _failed_close(app)
    assert "OLD_ONLY" not in json.dumps(result["offline_processing"])
    assert "PRIVATE_PROVIDER_FAILURE" not in json.dumps(result)
    before = len(memory.messages), len(reflection.messages), len(fast.calls)
    app.text_turn("person", "NEW_ONLY 今天想换个话题。")
    new_id = app._sessions["person"][1].session_id
    assert new_id != old_id
    assert [t.user for t in app._sessions["person"][1].turns] == ["NEW_ONLY 今天想换个话题。"]
    assert (len(memory.messages), len(reflection.messages)) == before[:2]
    assert len(fast.calls) == before[2] + 1
    saved = store.load("person")
    assert saved.pending_session.session_id == new_id
    assert saved.offline_sessions[0].session_id == old_id

    reflection.payload = GOOD_REFLECTION
    app.end("person", process_offline=True)  # Seal new, recover only old (FIFO).
    assert len(memory.messages) == 1
    assert "NEW_ONLY" not in json.dumps(reflection.messages[-1], ensure_ascii=False)
    saved = store.load("person")
    assert [j.session_id for j in saved.offline_sessions] == [new_id]
    assert old_id in saved.person_model_state.completed_session_ids
    app.end("person", process_offline=True)  # No active chat: recover the second sealed job.
    assert len(memory.messages) == 2
    assert "NEW_ONLY" in json.dumps(memory.messages[-1], ensure_ascii=False)
    assert "OLD_ONLY" not in json.dumps(memory.messages[-1], ensure_ascii=False)
    assert store.load("person").offline_sessions == []
    assert store.load("person").person_model_state.sessions_completed == 2
    assert "person" not in app._sessions
    app.close()


def test_repeated_end_recovers_exact_job_without_new_chat_or_duplicate_memory(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    old_id, _ = _failed_close(app)
    reflection.payload = GOOD_REFLECTION
    result = app.end("person", process_offline=True)
    assert result["ok"] and not result["offline_processing_pending"]
    assert len(fast.calls) == 1 and len(memory.messages) == 1 and len(reflection.messages) == 2
    assert store.load("person").person_model_state.completed_session_ids == [old_id]
    assert app.end("person", process_offline=True)["memories"] is None
    assert len(reflection.messages) == 2
    assert app._sessions == {}
    app.close()


def test_restarted_app_can_recover_without_opening_a_conversation(tmp_path):
    store = PersonaStore(tmp_path)
    first, _, _, _ = _app(store)
    old_id, _ = _failed_close(first)
    first.close()
    restarted, fast, memory, reflection = _app(store, failing=False)
    result = restarted.end("person", process_offline=True)
    assert result["ok"] and not result["offline_processing_pending"]
    assert restarted._sessions == {} and fast.calls == [] and memory.messages == []
    assert len(reflection.messages) == 1
    assert store.load("person").person_model_state.completed_session_ids == [old_id]
    restarted.close()


def test_recovery_preserves_durable_trust_and_active_tasks(tmp_path):
    store = PersonaStore(tmp_path)
    app, _, _, reflection = _app(store)
    _failed_close(app)
    p = store.load("person")
    p.alliance.trust = 0.83
    p.active_tasks = [_task()]
    store.save(p)
    original_tasks = [t.model_dump() for t in p.active_tasks]
    reflection.payload = GOOD_REFLECTION
    assert app.end("person", process_offline=True)["ok"]
    saved = store.load("person")
    assert saved.alliance.trust == 0.83
    assert [t.model_dump() for t in saved.active_tasks] == original_tasks
    app.close()


def _task():
    return PersistentTaskRecord(id="existing-task", skill_id="reminder", intent="reminder",
                                goal="已有生活提醒", status="pending",
                                created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
                                slots={"label": "浇花"})


def test_restored_live_checkpoint_can_end_without_wiping_runtime_state(tmp_path):
    store = PersonaStore(tmp_path)
    p = PersonaProfile(user_id="live", active_tasks=[_task()], pending_session=SessionCheckpointRecord(
        session_id="live-session", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        turns=[SessionTurnSnapshot(user="今天想聊自己的节奏。", agent="你慢慢说。")],
        original_turn_count=1, turn_indices=[0],
    ))
    p.alliance.trust = 0.83
    store.save(p)
    app, fast, _, _ = _app(store, failing=False)
    _, ctx = app._session("live")
    assert ctx.session_id == "live-session" and ctx.alliance.trust == 0.83
    assert ctx.active_tasks[0].id == "existing-task"
    assert app.end("live", process_offline=True)["ok"]
    saved = store.load("live")
    assert saved.alliance.trust == 0.83
    assert len(saved.active_tasks) == 1 and saved.active_tasks[0].slots == {"label": "浇花"}
    assert fast.calls == []
    app.close()


def test_expired_closed_text_is_pruned_without_cloud_calls_or_active_chat(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    _failed_close(app)
    p = store.load("person")
    p.offline_sessions[0].expires_at = "2000-01-01T00:00:00+00:00"
    store.save(p)
    counts = len(fast.calls), len(memory.messages), len(reflection.messages)
    result = app.prune_offline_sessions()
    assert result == {"inspected_profiles": 1, "failed_profiles": 0}
    saved = store.load("person")
    assert saved.offline_sessions == [] and "OLD_ONLY" not in saved.model_dump_json()
    assert (len(fast.calls), len(memory.messages), len(reflection.messages)) == counts
    assert app._sessions == {}
    app.close()


def test_idle_only_seals_and_never_starts_paid_processing(tmp_path):
    app, _, memory, reflection = _app(PersonaStore(tmp_path))
    app.text_turn("idle", "OLD_ONLY 我愿意慢一点聊。")
    now = app._session_activity["idle"] + 11
    assert app.finalize_idle_sessions(now=now) == ["idle"]
    assert len(memory.messages) == 0 and len(reflection.messages) == 0
    for _ in range(3):
        assert app.finalize_idle_sessions(now=now + 20) == []
    assert len(reflection.messages) == 0
    app.close()


def test_seal_store_failure_retains_live_pair_and_never_calls_offline_models(tmp_path, monkeypatch):
    store = PersonaStore(tmp_path)
    app, _, memory, reflection = _app(store, failing=False)
    app.text_turn("person", "这段对话还不能丢。")
    original_pair = app._sessions["person"]
    original_persona = original_pair[0].persona
    original_state = original_persona.model_dump()
    original_working = deepcopy(original_pair[1].working_memory)
    original_save = store.save

    def fail_save(_):
        raise OSError("PRIVATE_STORAGE_ERROR")

    monkeypatch.setattr(store, "save", fail_save)
    result = app.end("person", process_offline=True)
    assert not result["ok"] and result["active_session_retained"]
    assert app._sessions["person"] is original_pair
    assert original_pair[0].persona is original_persona
    assert original_pair[0].persona.model_dump() == original_state
    assert original_pair[1].working_memory == original_working
    assert memory.messages == reflection.messages == []
    assert "PRIVATE_STORAGE_ERROR" not in json.dumps(result)
    monkeypatch.setattr(store, "save", original_save)
    assert app.end("person", process_offline=True)["ok"]
    assert "person" not in app._sessions
    app.close()


def test_idle_storage_failure_isolated_from_other_users(tmp_path, monkeypatch):
    store = PersonaStore(tmp_path)
    app, _, _, reflection = _app(store, failing=False)
    app.text_turn("bad", "第一段合成对话。")
    app.text_turn("good", "第二段合成对话。")
    original_save = store.save

    def conditional_save(persona):
        if persona.user_id == "bad":
            raise OSError("private")
        return original_save(persona)

    monkeypatch.setattr(store, "save", conditional_save)
    now = max(app._session_activity.values()) + 11
    assert app.finalize_idle_sessions(now=now) == ["good"]
    assert "bad" in app._sessions and "good" not in app._sessions
    assert len(reflection.messages) == 0
    with pytest.raises(RuntimeError, match="尚未安全封存"):
        app.close()
    assert "bad" in app._sessions


def test_post_seal_exception_does_not_lose_job_or_poison_idle_worker(tmp_path, monkeypatch):
    store = PersonaStore(tmp_path)
    app, _, _, _ = _app(store, failing=False)
    app.text_turn("person", "原话应在模型调用前安全封存。")
    engine = app._sessions["person"][0]

    def fail_recover(**_):
        raise RuntimeError("PRIVATE_UNEXPECTED_ERROR")

    monkeypatch.setattr(engine, "recover_offline_sessions", fail_recover)
    result = app.end("person", process_offline=True)
    assert not result["ok"] and "PRIVATE_UNEXPECTED_ERROR" not in json.dumps(result)
    assert "person" not in app._sessions
    assert len(store.load("person").offline_sessions) == 1
    assert app.end("person", process_offline=True)["ok"]  # A fresh engine reads the durable job.
    assert store.load("person").offline_sessions == []
    app.close()


def test_reflection_commit_save_failure_exposes_only_committed_profile_and_can_recover(tmp_path, monkeypatch):
    store = PersonaStore(tmp_path)
    app, _, memory, reflection = _app(store, failing=False)
    app.text_turn("person", "这段完整合成会话等待事务提交。")
    engine, ctx = app._sessions["person"]
    original_persona = engine.persona
    original_save = store.save

    def fail_final_commit(persona):
        if persona.person_model_state.sessions_completed:
            raise OSError("PRIVATE_ATOMIC_SAVE_FAILURE")
        return original_save(persona)

    monkeypatch.setattr(store, "save", fail_final_commit)
    failed = app.end("person", process_offline=True)
    assert failed["ok"] is False
    assert "PRIVATE_ATOMIC_SAVE_FAILURE" not in json.dumps(failed)
    assert engine.persona is original_persona
    assert engine.persona.person_model_state.sessions_completed == 0
    assert ctx.session_id not in engine.persona.person_model_state.reflected_session_ids
    saved = store.load("person")
    assert saved.model_dump() == engine.persona.model_dump()
    assert saved.offline_sessions[0].memory_completed is True
    assert saved.offline_sessions[0].reflection_completed is False
    assert len(memory.messages) == len(reflection.messages) == 1
    monkeypatch.setattr(store, "save", original_save)
    assert app.end("person", process_offline=True)["ok"]
    assert len(memory.messages) == 1 and len(reflection.messages) == 2
    saved = store.load("person")
    assert saved.offline_sessions == [] and saved.person_model_state.sessions_completed == 1
    app.close()


def test_attempt_limit_is_visible_and_repeated_end_does_not_reset_it(tmp_path):
    app, _, _, reflection = _app(PersonaStore(tmp_path))
    _failed_close(app)
    for _ in range(OFFLINE_MAX_ATTEMPTS + 2):
        result = app.end("person", process_offline=True)
    assert len(reflection.messages) == OFFLINE_MAX_ATTEMPTS
    assert result["offline_processing"]["exhausted_count"] == 1
    assert result["offline_processing_pending"]
    app.close()


def test_forget_removes_pending_text_and_audit_without_recreating_profile(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    _failed_close(app)
    before = len(fast.calls), len(memory.messages), len(reflection.messages)
    result = app.forget("person")
    assert result["deleted"] and "会后待处理会话原文与恢复记录" in result["local_scope"]
    assert app.end("person", process_offline=True)["memories"] is None
    app.prune_offline_sessions()
    assert store.load("person") is None and not list(tmp_path.glob("person.json*"))
    assert (len(fast.calls), len(memory.messages), len(reflection.messages)) == before
    assert app._sessions == {}
    app.close()


def test_end_unknown_user_does_not_seed_profile(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    assert app.end("unknown", process_offline=True)["memories"] is None
    assert store.list_user_ids() == [] and app._sessions == {}
    assert fast.calls == memory.messages == reflection.messages == []
    app.close()


def test_legacy_closed_tail_is_not_reopened_or_guessed_as_complete_context(tmp_path):
    store = PersonaStore(tmp_path)
    store.save(PersonaProfile(user_id="legacy", pending_session=SessionCheckpointRecord(
        session_id="legacy-closed", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(), offline_finalize_pending=True,
        memory_consolidated=True,
        turns=[SessionTurnSnapshot(user="LEGACY_PRIVATE_OLD_TEXT", agent="以前的回应。")],
    )))
    app, _, memory, reflection = _app(store, failing=False)
    app.text_turn("legacy", "NEW_ONLY 新会话不拼接旧文字。")
    ctx = app._sessions["legacy"][1]
    assert ctx.session_id != "legacy-closed" and len(ctx.turns) == 1
    saved = store.load("legacy")
    assert saved.offline_sessions[0].source_completeness == "legacy_tail_unknown_indices"
    assert saved.offline_sessions[0].state == "needs_review"
    assert memory.messages == reflection.messages == []
    app.close()


def test_full_queue_cannot_silently_delete_unmigrated_legacy_checkpoint(tmp_path):
    store = PersonaStore(tmp_path)
    app, fast, memory, reflection = _app(store)
    engine = app._engine_for_persona(PersonaProfile(user_id="full"))
    for index in range(OFFLINE_QUEUE_CAPACITY):
        ctx = EngineContext(user_id="full", scene_id="companion_elder", session_id=f"closed_{index}")
        ctx.add_turn("完整封存的合成话。", "我听到了。")
        engine.seal_session(ctx)
    p = store.load("full")
    p.pending_session = SessionCheckpointRecord(
        session_id="legacy-still-pending", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(), offline_finalize_pending=True,
        turns=[SessionTurnSnapshot(user="不能静默丢掉的旧检查点", agent="以前的回应。")],
    )
    store.save(p)
    with pytest.raises(RuntimeError):
        app._session("full")
    saved = store.load("full")
    assert saved.pending_session.session_id == "legacy-still-pending"
    assert len(saved.offline_sessions) == OFFLINE_QUEUE_CAPACITY
    assert "full" not in app._sessions
    assert fast.calls == memory.messages == reflection.messages == []
    app.close()


def test_empty_context_finalization_does_not_report_pending_or_call_models(tmp_path):
    app, fast, memory, reflection = _app(PersonaStore(tmp_path), failing=False)
    engine = app._engine_for_persona(PersonaProfile(user_id="empty", active_tasks=[_task()]))
    engine.persona.alliance.trust = 0.83
    ctx = EngineContext(user_id="empty", scene_id="companion_elder")
    engine.finalize_session(ctx)
    assert engine.offline_status()["pending_count"] == 0
    assert not ctx.working_memory.get("_offline_finalize_pending")
    assert engine.persona.person_model_state.sessions_completed == 0
    assert engine.persona.alliance.trust == 0.83 and len(engine.persona.active_tasks) == 1
    assert fast.calls == memory.messages == reflection.messages == []
    app.close()


def test_empty_restored_checkpoint_is_cleared_on_web_end_without_cloud_calls(tmp_path):
    store = PersonaStore(tmp_path)
    p = PersonaProfile(user_id="empty", active_tasks=[_task()], pending_session=SessionCheckpointRecord(
        session_id="empty-checkpoint", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(), turns=[],
        original_turn_count=0, turn_indices=[],
    ))
    p.alliance.trust = 0.83
    store.save(p)
    app, fast, memory, reflection = _app(store, failing=False)
    app._session("empty")
    result = app.end("empty", process_offline=True)
    assert result["ok"] and not result["offline_processing_pending"]
    saved = store.load("empty")
    assert saved.pending_session is None and saved.offline_sessions == []
    assert saved.alliance.trust == 0.83 and len(saved.active_tasks) == 1
    assert fast.calls == memory.messages == reflection.messages == []
    assert saved.person_model_state.sessions_completed == 0
    app.close()


def test_closed_context_refuses_further_online_and_checkpoint_writes(tmp_path):
    app, fast, memory, reflection = _app(PersonaStore(tmp_path))
    app.text_turn("closed", "这轮应封存。")
    engine, ctx = app._sessions["closed"]
    engine.seal_session(ctx)
    before = deepcopy(ctx.turns)
    with pytest.raises(ValueError):
        engine.process_turn(ctx, "不应拼接进入封存的会话。")
    with pytest.raises(ValueError):
        engine.checkpoint_session(ctx)
    assert ctx.turns == before and len(fast.calls) == 1
    assert memory.messages == reflection.messages == []
    app.close()


def test_long_complete_checkpoint_keeps_first_evidence_and_original_indices_after_restart(tmp_path):
    store = PersonaStore(tmp_path)
    first, _, _, _ = _app(store, failing=False)
    engine, ctx = first._session("long")
    for index in range(17):
        ctx.add_turn(f"ORIGINAL_{index} 本轮合成的独立对话。", "我听到了。")
    original_id = ctx.session_id
    engine.checkpoint_session(ctx)
    saved = store.load("long").pending_session
    assert saved.original_turn_count == 17
    assert saved.turn_indices == list(range(17)) and len(saved.turns) == 17
    assert saved.turns[0].user.startswith("ORIGINAL_0 ")
    # Simulate process loss after the durable checkpoint; no graceful seal.

    restarted, fast, memory, reflection = _app(store, failing=False)
    _, restored = restarted._session("long")
    assert restored.session_id == original_id and len(restored.turns) == 17
    assert restored.turns[0].user.startswith("ORIGINAL_0 ")
    assert restored.working_memory["_offline_source_completeness"] == "complete"
    assert restarted.end("long", process_offline=True)["ok"]
    assert fast.calls == [] and len(memory.messages) == len(reflection.messages) == 1
    prompt = json.dumps(reflection.messages[0], ensure_ascii=False)
    assert "[turn 0] 用户:ORIGINAL_0 " in prompt
    assert "[turn 16] 用户:ORIGINAL_16 " in prompt
    audit = store.load("long").offline_session_audit[-1]
    assert audit.source_completeness == "complete" and audit.turn_count == 17
    restarted.close()


def test_sensitive_middle_turn_is_removed_but_source_indices_survive_restart(tmp_path):
    store = PersonaStore(tmp_path)
    first, _, _, _ = _app(store, failing=False)
    engine, ctx = first._session("sensitive")
    ctx.add_turn("SOURCE_ZERO 合成的普通经历。", "我听到了。")
    ctx.add_turn("PRIVATE_MIDDLE_TURN 不能保存的敏感原话。", "PRIVATE_MIDDLE_REPLY")
    ctx.add_turn("SOURCE_TWO 愿意慢一点听说明。", "可以慢一点。")
    ctx.add_turn("SOURCE_THREE 更喜欢有选择。", "你来选。")
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    engine.checkpoint_session(ctx)
    saved = store.load("sensitive")
    assert saved.pending_session.original_turn_count == 4
    assert saved.pending_session.turn_indices == [0, 2, 3]
    assert "PRIVATE_MIDDLE" not in saved.model_dump_json()
    # Simulate process loss after the durable checkpoint; no graceful seal.

    restarted, fast, memory, reflection = _app(store, failing=False)
    _, restored = restarted._session("sensitive")
    assert len(restored.turns) == 4
    assert restored.turns[1].user == restored.turns[1].agent == ""
    assert restored.turns[2].user.startswith("SOURCE_TWO ")
    assert restored.working_memory["_sensitive_memory_excluded_turns"] == [1]
    assert restarted.end("sensitive", process_offline=True)["ok"]
    assert fast.calls == [] and len(memory.messages) == len(reflection.messages) == 1
    for calls in (memory.messages, reflection.messages):
        prompt = json.dumps(calls, ensure_ascii=False)
        assert "[turn 2] 用户:SOURCE_TWO " in prompt
        assert "[turn 1]" not in prompt and "PRIVATE_MIDDLE" not in prompt
    saved = store.load("sensitive")
    assert "PRIVATE_MIDDLE" not in saved.model_dump_json()
    assert saved.offline_session_audit[-1].excluded_turn_count == 1
    restarted.close()


def test_legacy_active_tail_stays_incomplete_after_new_turn_and_checkpoint(tmp_path):
    store = PersonaStore(tmp_path)
    store.save(PersonaProfile(user_id="legacy_active", pending_session=SessionCheckpointRecord(
        session_id="old-live-tail", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(),
        turns=[SessionTurnSnapshot(user=f"OLD_TAIL_{i} 旧版本保留的尾部。", agent="以前的回应。")
               for i in range(12)],
    )))
    app, fast, memory, reflection = _app(store, failing=False)
    _, ctx = app._session("legacy_active")
    assert ctx.working_memory["_offline_source_completeness"] == "legacy_tail_unknown_indices"
    assert app.text_turn("legacy_active", "NEW_AFTER_LEGACY 接着聊了一句。")["ok"]
    pending = store.load("legacy_active").pending_session
    assert pending.original_turn_count is None and pending.turn_indices == []
    assert len(pending.turns) == 12
    result = app.end("legacy_active", process_offline=True)
    assert result["ok"] and result["offline_processing_pending"]
    assert result["offline_processing"]["needs_review_count"] == 1
    assert memory.messages == reflection.messages == [] and len(fast.calls) == 1
    assert store.load("legacy_active").offline_sessions[0].source_completeness == "legacy_tail_unknown_indices"
    app.close()


@pytest.mark.parametrize("count,indices", [
    (None, []), (-1, [0, 1]), (121, [0, 1]), (2, [0]),
    (2, [0, 0]), (2, [1, 0]), (2, [0, 2]), (0, [0, 1]),
])
def test_invalid_checkpoint_index_contract_cannot_be_upgraded_to_complete(tmp_path, count, indices):
    store = PersonaStore(tmp_path)
    store.save(PersonaProfile(user_id="bad_indices", pending_session=SessionCheckpointRecord(
        session_id="bad-source-indices", scene_id="companion_elder",
        updated_at=datetime.now(timezone.utc).isoformat(), original_turn_count=count,
        turn_indices=indices,
        turns=[SessionTurnSnapshot(user="第一条合成原话。", agent="回应。"),
               SessionTurnSnapshot(user="第二条合成原话。", agent="回应。")],
    )))
    app, fast, memory, reflection = _app(store, failing=False)
    _, ctx = app._session("bad_indices")
    assert ctx.working_memory["_offline_source_completeness"] == "legacy_tail_unknown_indices"
    result = app.end("bad_indices", process_offline=True)
    assert result["offline_processing"]["needs_review_count"] == 1
    assert fast.calls == memory.messages == reflection.messages == []
    app.close()


def test_prune_is_bounded_rotates_and_isolates_corrupt_profiles(tmp_path, monkeypatch):
    store = PersonaStore(tmp_path)
    for name in ("a", "b", "c"):
        store.save(PersonaProfile(user_id=name))
    app, fast, memory, reflection = _app(store)
    original_load = store.load
    seen = []

    def checked_load(user_id):
        seen.append(user_id)
        if user_id == "b":
            raise RuntimeError("PRIVATE_CORRUPT_TEXT")
        return original_load(user_id)

    monkeypatch.setattr(store, "load", checked_load)
    assert app.prune_offline_sessions(max_profiles=2) == {"inspected_profiles": 2, "failed_profiles": 1}
    assert seen == ["a", "b"]
    seen.clear()
    assert app.prune_offline_sessions(max_profiles=2) == {"inspected_profiles": 2, "failed_profiles": 0}
    assert seen == ["c", "a"]
    assert fast.calls == memory.messages == reflection.messages == []
    app.close()


def test_capabilities_describe_recovery_retention_and_cost_limits(tmp_path):
    app, _, _, _ = _app(PersonaStore(tmp_path))
    caps = app.capabilities()
    assert caps["data_practices"]["offline_session_text_retention_hours"] == OFFLINE_TTL_HOURS
    assert caps["data_practices"]["offline_queue_deleted_with_profile"] is True
    assert caps["session"]["offline_max_attempts_per_session"] == OFFLINE_MAX_ATTEMPTS
    assert caps["session"]["offline_recovery_during_chat"] is False
    assert caps["session"]["periodic_paid_retry_of_closed_sessions"] is False
    app.close()


@pytest.mark.parametrize("max_profiles", [0, 129, True, "2"])
def test_prune_rejects_invalid_or_unbounded_scan_sizes(tmp_path, max_profiles):
    app, _, _, _ = _app(PersonaStore(tmp_path))
    with pytest.raises(ValueError):
        app.prune_offline_sessions(max_profiles=max_profiles)
    app.close()


def test_forget_waits_for_inflight_same_user_recovery_and_prevents_resurrection(tmp_path):
    entered, release, forgotten = threading.Event(), threading.Event(), threading.Event()

    class BlockingReflection(LocalModel):
        def complete(self, messages):
            entered.set()
            assert release.wait(5)
            return super().complete(messages)

    store = PersonaStore(tmp_path)
    reflection = BlockingReflection(GOOD_REFLECTION)
    app, _, _, _ = _app(store, reflection=reflection)
    app.text_turn("person", "PRIVATE_CLOSED_TEXT 待处理会话。")
    results, errors = {}, []

    def finish():
        try:
            results["end"] = app.end("person", process_offline=True)
        except BaseException as error:
            errors.append(error)

    def forget():
        try:
            results["forget"] = app.forget("person")
            forgotten.set()
        except BaseException as error:
            errors.append(error)

    finishing = threading.Thread(target=finish)
    deleting = threading.Thread(target=forget)
    finishing.start()
    try:
        assert entered.wait(2)
        deleting.start()
        assert not forgotten.wait(0.05)
    finally:
        release.set()
        finishing.join(5)
        if deleting.ident is not None:
            deleting.join(5)
    assert not errors and forgotten.is_set()
    assert results["end"]["ok"] and results["forget"]["deleted"]
    assert store.load("person") is None
    assert app.end("person", process_offline=True)["memories"] is None
    assert len(reflection.messages) == 1
    assert app._sessions == {}
    app.close()


def test_other_user_chat_is_not_blocked_by_inflight_offline_model(tmp_path):
    entered, release, online_done = threading.Event(), threading.Event(), threading.Event()

    class BlockingReflection(LocalModel):
        def complete(self, messages):
            entered.set()
            assert release.wait(5)
            return super().complete(messages)

    reflection = BlockingReflection(GOOD_REFLECTION)
    app, fast, _, _ = _app(PersonaStore(tmp_path), reflection=reflection)
    app.text_turn("closing", "这一段要做会后处理。")
    errors = []

    def close_old():
        try:
            assert app.end("closing", process_offline=True)["ok"]
        except BaseException as error:
            errors.append(error)

    def new_chat():
        try:
            assert app.text_turn("another", "另一位用户现在聊天。")["ok"]
            online_done.set()
        except BaseException as error:
            errors.append(error)

    closer = threading.Thread(target=close_old)
    chatter = threading.Thread(target=new_chat)
    closer.start()
    try:
        assert entered.wait(2)
        chatter.start()
        assert online_done.wait(2), "unrelated online chat waited for an offline model"
    finally:
        release.set()
        closer.join(5)
        if chatter.ident is not None:
            chatter.join(5)
        app.close()
    assert not errors and len(fast.calls) == 2
