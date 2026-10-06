"""会话增量检查点、幂等结束与空闲自动收口。"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import (
    PersonaProfile,
    SessionCheckpointRecord,
    SessionTurnSnapshot,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.llm import MockLLMClient
from aria.scenes.registry import get_scene
from aria.voice.web import TextWebApp


class _CountingUpdater:
    def __init__(self):
        self.calls = 0

    def diagnose(self, persona, _ctx):
        self.calls += 1
        return persona


class _PayloadLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, _messages):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_checkpoint_is_lightweight_and_finalize_is_idempotent(tmp_path):
    store = PersonaStore(tmp_path)
    updater = _CountingUpdater()
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="u"),
        llm_client=MockLLMClient(),
        store=store,
        diagnoser=updater,
    )
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    engine.process_turn(ctx, "最近不太想出门。")
    engine.checkpoint_session(ctx)
    assert store.exists("u")
    assert updater.calls == 0
    assert store.load("u").pending_session is not None

    engine.finalize_session(ctx)
    memories = len(engine.persona.episodic_memory)
    assert engine.persona.person_model_state.sessions_completed == 1
    engine.finalize_session(ctx)
    assert updater.calls == 1
    assert len(engine.persona.episodic_memory) == memories
    assert engine.persona.person_model_state.sessions_completed == 1
    assert store.load("u").pending_session is None


def test_web_idle_session_finalizes_and_persists(tmp_path):
    store = PersonaStore(tmp_path)
    app = TextWebApp(
        store=store,
        session_idle_seconds=10,
        checkpoint_turns=1,
    )
    result = app.text_turn("idle_user", "今天聊了一会儿。")
    assert result["ok"] and store.exists("idle_user")
    active_at = app._session_activity["idle_user"]

    assert app.finalize_idle_sessions(now=active_at + 11) == ["idle_user"]
    assert "idle_user" not in app._sessions
    saved = store.load("idle_user")
    assert saved.offline_sessions and not saved.episodic_memory


def test_web_restores_unfinished_text_session_from_checkpoint(tmp_path):
    store = PersonaStore(tmp_path)
    first = TextWebApp(
        store=store,
        checkpoint_turns=1,
    )
    first.text_turn("resume_user", "这是检查点前说的话。")
    saved = store.load("resume_user")
    assert saved.pending_session is not None
    previous_session_id = saved.pending_session.session_id

    restarted = TextWebApp(
        store=store,
        checkpoint_turns=5,
    )
    _, context = restarted._session("resume_user")
    assert context.session_id == previous_session_id
    assert [turn.user for turn in context.turns] == [
        "这是检查点前说的话。"
    ]
    restarted.text_turn("resume_user", "这是重启后继续的话。")
    restarted.end("resume_user")
    reloaded = store.load("resume_user")
    assert reloaded.pending_session is None
    assert reloaded.person_model_state.sessions_completed == 0
    assert len(reloaded.offline_sessions) == 1


def test_expired_checkpoint_text_is_not_restored(tmp_path):
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(
        user_id="expired",
        pending_session=SessionCheckpointRecord(
            session_id="old-session",
            scene_id="companion_elder",
            updated_at="2020-01-01T00:00:00",
            expires_at="2020-01-02T00:00:00",
            turns=[SessionTurnSnapshot(user="旧的私密原话", agent="回应")],
        ),
    )
    store.save(persona)
    app = TextWebApp(store=store)
    _engine, context = app._session("expired")
    assert context.session_id != "old-session"
    assert context.turns == []
    assert store.load("expired").pending_session is None


def test_failed_offline_memory_is_persisted_for_retry(tmp_path):
    store = PersonaStore(tmp_path)
    llm = _PayloadLLM(RuntimeError("temporary outage"))
    extractor = LLMExtractor(llm)
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="memory_retry"),
        llm_client=MockLLMClient(),
        store=store,
        extractor=extractor,
    )
    ctx = EngineContext(
        user_id="memory_retry", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "我年轻时在厂里修机器。")
    engine.finalize_session(ctx)

    pending = store.load("memory_retry").offline_sessions[0]
    assert pending.session_id == ctx.session_id
    assert pending.memory_completed is False
    assert engine.persona.person_model_state.sessions_completed == 0

    llm.payload = (
        '{"cognitive_privacy":[],"relationships":[],"episodes":[{"summary":"年轻时修机器","keywords":["修机器"],'
        '"referenceable":true,"evidence_quote":"我年轻时在厂里修机器",'
        '"turn_index":0}]}'
    )
    engine.finalize_session(ctx)
    saved = store.load("memory_retry")
    assert saved.pending_session is None
    assert saved.offline_sessions == []
    assert saved.person_model_state.sessions_completed == 1
    assert any("修机器" in row.summary for row in saved.episodic_memory)


def test_failed_offline_reflection_retries_without_duplicate_memory(tmp_path):
    store = PersonaStore(tmp_path)
    reflection_llm = _PayloadLLM(RuntimeError("temporary outage"))
    updater = PersonaHypothesisUpdater(reflection_llm, min_turns=1)
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="reflection_retry"),
        llm_client=MockLLMClient(),
        store=store,
        extractor=LLMExtractor(_PayloadLLM('{"episodes":[],"relationships":[],"cognitive_privacy":[]}')),
        hypothesis_updater=updater,
    )
    ctx = EngineContext(
        user_id="reflection_retry", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "我不喜欢别人替我做决定。")
    engine.finalize_session(ctx)

    pending = store.load("reflection_retry").offline_sessions[0]
    assert pending.session_id == ctx.session_id
    assert pending.memory_completed is True
    memory_count = len(engine.persona.episodic_memory)
    assert engine.persona.person_model_state.sessions_completed == 0

    reflection_llm.payload = '{"changes":[]}'
    engine.finalize_session(ctx)
    saved = store.load("reflection_retry")
    assert saved.pending_session is None
    assert saved.offline_sessions == []
    assert saved.person_model_state.sessions_completed == 1
    assert len(saved.episodic_memory) == memory_count


def test_invalid_memory_protocol_stays_pending_then_retries_successfully(
    tmp_path,
):
    store = PersonaStore(tmp_path)
    llm = _PayloadLLM("{}")
    extractor = LLMExtractor(llm)
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="memory_protocol_retry"),
        llm_client=MockLLMClient(),
        store=store,
        extractor=extractor,
    )
    ctx = EngineContext(
        user_id="memory_protocol_retry", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "我年轻时在厂里修机器。")

    engine.finalize_session(ctx)

    pending = store.load("memory_protocol_retry").offline_sessions[0]
    assert pending.session_id == ctx.session_id
    assert pending.memory_completed is False
    assert extractor.last_error_type == "invalid_json_response"

    llm.payload = '{"episodes":[],"relationships":[],"cognitive_privacy":[]}'
    engine.finalize_session(ctx)

    saved = store.load("memory_protocol_retry")
    assert saved.pending_session is None
    assert saved.offline_sessions == []
    assert saved.person_model_state.sessions_completed == 1
    assert extractor.last_status == "ok"


def test_invalid_reflection_protocol_stays_pending_then_retries_successfully(
    tmp_path,
):
    store = PersonaStore(tmp_path)
    reflection_llm = _PayloadLLM('{"change":[]}')
    updater = PersonaHypothesisUpdater(reflection_llm, min_turns=1)
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="reflection_protocol_retry"),
        llm_client=MockLLMClient(),
        store=store,
        extractor=LLMExtractor(_PayloadLLM('{"episodes":[],"relationships":[],"cognitive_privacy":[]}')),
        hypothesis_updater=updater,
    )
    ctx = EngineContext(
        user_id="reflection_protocol_retry", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "我不喜欢别人替我做决定。")

    engine.finalize_session(ctx)

    pending = store.load("reflection_protocol_retry").offline_sessions[0]
    assert pending.session_id == ctx.session_id
    assert pending.memory_completed is True
    assert pending.reflection_completed is False
    assert updater.last_error_type == "invalid_json_response"
    assert ctx.session_id not in (
        engine.persona.person_model_state.reflected_session_ids
    )

    reflection_llm.payload = '{"changes":[]}'
    engine.finalize_session(ctx)

    saved = store.load("reflection_protocol_retry")
    assert saved.pending_session is None
    assert saved.offline_sessions == []
    assert saved.person_model_state.sessions_completed == 1
    assert ctx.session_id in saved.person_model_state.reflected_session_ids
    assert updater.last_status == "ok"
