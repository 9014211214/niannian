"""Text-only session lifecycle and local HTTP infrastructure for the first release."""
from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..engine.context import EngineContext, Turn
from ..engine.memory.offline_queue import OFFLINE_MAX_ATTEMPTS, OFFLINE_MAX_TURNS, OFFLINE_TTL_HOURS
from ..engine.memory.persona_store import PersonaStore
from ..engine.model.persona import PersonaProfile
from ..engine.orchestrator import Engine
from ..engine.perception.external_emotion import (
    SyntheticEmotionExperiment, parse_external_emotion, external_emotion_payload,
)
from ..engine.perception.signals import is_simulated
from ..engine.policy.online_decision import configured_online_decision_version
from ..engine.policy.prompt import configured_delivery_block, prompt_version_for
from ..runtime import create_minimal_clients, create_minimal_engine
from ..scenes.registry import get_scene


class TextWebApp:
    """Bounded text sessions; idle and shutdown only seal their original text."""

    def __init__(self, *, store: PersonaStore, clients=None,
                 session_idle_seconds=1200, checkpoint_turns=1,
                 demo_token=None, delivery_block=None):
        self.store = store
        self.clients = clients or create_minimal_clients()
        self.scene_id = "companion_elder"
        self.delivery_block = configured_delivery_block(delivery_block)
        self.demo_token = demo_token or secrets.token_urlsafe(24)
        self.allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        self.session_idle_seconds = float(session_idle_seconds)
        self.checkpoint_turns = max(1, int(checkpoint_turns))
        self._sessions: dict[str, tuple[Engine, EngineContext]] = {}
        self._session_activity: dict[str, float] = {}
        self._lock = threading.RLock()
        self._user_locks: dict[str, threading.RLock] = {}
        self._offline_prune_cursor = None

    def _seed_persona(self, user_id):
        return PersonaProfile(user_id=user_id, profile_origin="blank")

    def _engine_for_persona(self, persona):
        return create_minimal_engine(get_scene(self.scene_id), persona,
                                     clients=self.clients, store=self.store,
                                     delivery_block=self.delivery_block)

    def _user_lock(self, user_id: str) -> threading.RLock:
        with self._lock:
            lock = self._user_locks.get(user_id)
            if lock is None:
                lock = threading.RLock()
                self._user_locks[user_id] = lock
            return lock


    @staticmethod
    def _checkpoint_is_current(pending) -> bool:
        raw = pending.expires_at
        if not raw:
            try:
                updated = datetime.fromisoformat(pending.updated_at)
            except (TypeError, ValueError):
                return False
            return (
                datetime.now(tz=updated.tzinfo) - updated
            ).total_seconds() <= 86_400
        try:
            expires = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return False
        return expires >= datetime.now(tz=expires.tzinfo)


    def _session(self, user_id: str) -> tuple[Engine, EngineContext]:
        with self._lock:
            if user_id not in self._sessions:
                persona = self.store.load(user_id) or self._seed_persona(user_id)
                engine = self._engine_for_persona(persona)
                # Migration/expiry only: opening a chat must never drain an
                # older failed offline job or append new text to its session.
                engine.prepare_offline_sessions()
                persona = engine.persona
                pending = persona.pending_session
                if pending is not None and pending.offline_finalize_pending:
                    # Capacity/storage refusal must not silently discard an
                    # unmigrated legacy job just to open a fresh chat.
                    raise RuntimeError("offline_session_migration_incomplete")
                if (
                    pending is not None
                    and not pending.offline_finalize_pending
                    and pending.scene_id == self.scene_id
                    and self._checkpoint_is_current(pending)
                ):
                    count = pending.original_turn_count
                    indices = pending.turn_indices
                    complete = (
                        type(count) is int and 0 <= count <= OFFLINE_MAX_TURNS
                        and isinstance(indices, list) and len(indices) == len(pending.turns)
                        and all(type(index) is int and 0 <= index < count for index in indices)
                        and indices == sorted(set(indices))
                    )
                    if complete:
                        restored_turns = [Turn(user="", agent="") for _ in range(count)]
                        for index, item in zip(indices, pending.turns):
                            restored_turns[index] = Turn(user=item.user, agent=item.agent)
                    else:
                        # Old checkpoints were only a tail with unknown source
                        # indices. Keep that provenance sticky through future
                        # checkpoints and sealing, never pretend it is complete.
                        restored_turns = [Turn(user=item.user, agent=item.agent) for item in pending.turns]
                    ctx = EngineContext(
                        user_id=user_id,
                        scene_id=self.scene_id,
                        session_id=pending.session_id,
                        turns=restored_turns,
                    )
                    ctx.working_memory["_offline_source_completeness"] = (
                        "complete" if complete else "legacy_tail_unknown_indices"
                    )
                    if complete:
                        kept_indices = set(indices)
                        ctx.working_memory["_sensitive_memory_excluded_turns"] = [
                            index for index in range(count) if index not in kept_indices
                        ]
                    ctx.working_memory["_last_checkpoint_turn"] = len(
                        ctx.turns
                    )
                    ctx.working_memory["_restored_checkpoint"] = True
                    if persona.cognitive_health_consent.has_text_monitoring_consent():
                        ctx.working_memory["_cognitive_source_turns"] = [
                            row.model_dump() for row in pending.cognitive_turns]
                        ctx.working_memory["_cognitive_session_at"] = pending.cognitive_session_at
                    # An owner may restore a checkpoint and immediately end it
                    # without process_turn. Do not write default trust/empty
                    # active tasks back over the durable runtime state.
                    ctx.alliance.trust = persona.alliance.trust
                    ctx.working_memory["_alliance_seeded"] = True
                    engine._seed_persistent_tasks(ctx)
                    if pending.memory_consolidated:
                        ctx.working_memory["_memory_consolidated"] = True
                    if pending.reflection_completed:
                        ctx.working_memory["_reflection_completed"] = True
                else:
                    if pending is not None:
                        persona.pending_session = None
                        self.store.save(persona)
                    ctx = EngineContext(
                        user_id=user_id, scene_id=self.scene_id
                    )
                self._sessions[user_id] = (engine, ctx)
            self._session_activity[user_id] = time.monotonic()
            return self._sessions[user_id]


    def text_turn(self, user_id, text, *, turn_id=None, external_emotion=None):
        """Only text becomes evidence; optional classifier data stays a soft hint."""
        reading = parse_external_emotion(external_emotion)
        if is_simulated(reading) and not isinstance(
            getattr(self, "external_emotion_experiment", None),
            SyntheticEmotionExperiment,
        ):
            raise ValueError("simulated_external_emotion_requires_isolated_experiment")
        utterance = " ".join((text or "").strip().split())
        if not utterance or len(utterance) > 500:
            return {"ok": False, "error": "请输入500字以内文字。"}
        with self._user_lock(user_id):
            engine, ctx = self._session(user_id)
            started = time.perf_counter()
            response = (engine.process_turn(ctx, utterance) if reading is None else
                        engine.process_turn(ctx, utterance, external_emotion=reading))
            with self._lock:
                self._session_activity[user_id] = time.monotonic()
            self._maybe_checkpoint(engine, ctx)
            return {
                "ok": True, "transcript": utterance, "response": response,
                "turn_id": turn_id,
                "timings_ms": {"engine_llm": round((time.perf_counter() - started) * 1000)},
                "emotion": {"label": ctx.emotion.label, "divergence": ctx.emotion.divergence,
                            "pattern": ctx.emotion.pattern,
                            **({"external_reading": external_emotion_payload(reading)}
                               if reading is not None else {})},
                "profile": _profile_view(engine),
                "evidence": [item.summary() for item in ctx.reasoning.evidence
                             if item.turn_index == len(ctx.turns) - 1],
                "reasoning": _reasoning_view(ctx),
            }

    def end(self, user_id: str, *, process_offline: bool = False) -> dict:
        with self._user_lock(user_id):
            with self._lock:
                pair = self._sessions.get(user_id)
            try:
                if pair is not None:
                    engine, _ctx = pair
                    self._seal_active_session(user_id, pair)
                else:
                    # Repeated end is an explicit, bounded recovery request.
                    # It must not seed a missing user or reopen an old chat.
                    persona = self.store.load(user_id)
                    if persona is None:
                        return {"ok": True, "memories": None,
                                "offline_processing_pending": False}
                    engine = self._engine_for_persona(persona)
                    engine.prepare_offline_sessions()
            except Exception:
                return {"ok": False, "error": "会话尚未安全保存，请稍后重试结束。",
                        "active_session_retained": pair is not None}
            had_pending = self._offline_pending(engine.offline_status())
            if not process_offline:
                return {
                    "ok": True, "sealed": True,
                    "offline_processing_pending": had_pending,
                    "offline_processing": engine.offline_status(),
                    "profile": _profile_view(engine),
                }
            try:
                processing = engine.recover_offline_sessions(max_jobs=1)
            except Exception:
                return {"ok": False, "error": "会后处理暂未完成，已保存的记录可稍后重试。",
                        "offline_processing_pending": True,
                        "offline_processing": engine.offline_status()}
            return {
                "ok": True,
                "offline_processing_pending": self._offline_pending(processing),
                "offline_processing": processing,
                "memories": (
                    len(engine.persona.episodic_memory)
                    if engine.persona and (pair is not None or had_pending) else None
                ),
                "profile": _profile_view(engine),
            }


    @staticmethod
    def _offline_pending(status: dict) -> bool:
        return bool(status.get("pending_count") or status.get("exhausted_count")
                    or status.get("needs_review_count"))


    def _seal_active_session(self, user_id: str, pair: tuple[Engine, EngineContext]) -> None:
        """Caller owns the user lock; discard RAM only after a durable seal."""
        engine, ctx = pair
        engine.seal_session(ctx)
        with self._lock:
            self._sessions.pop(user_id, None)
            self._session_activity.pop(user_id, None)


    def checkpoint(self, user_id: str) -> dict:
        """页面离开或长会话中的轻量保存；不触发人格反思大模型。"""
        with self._user_lock(user_id):
            with self._lock:
                pair = self._sessions.get(user_id)
            if pair is None:
                return {"ok": True, "turns": None}
            engine, ctx = pair
            engine.checkpoint_session(ctx)
            with self._lock:
                self._session_activity[user_id] = time.monotonic()
            return {"ok": True, "turns": len(ctx.turns)}


    def finalize_idle_sessions(
        self, *, now: float | None = None
    ) -> list[str]:
        """Seal expired chats without invoking either after-session model."""
        current = time.monotonic() if now is None else now
        finalized: list[str] = []
        with self._lock:
            expired = [
                user_id for user_id, active_at in self._session_activity.items()
                if current - active_at >= self.session_idle_seconds
            ]
        for user_id in expired:
            with self._user_lock(user_id):
                with self._lock:
                    active_at = self._session_activity.get(user_id)
                    if (
                        active_at is None
                        or current - active_at < self.session_idle_seconds
                    ):
                        continue
                    pair = self._sessions.get(user_id)
                if pair is None:
                    continue
                engine, _ctx = pair
                try:
                    self._seal_active_session(user_id, pair)
                except Exception:
                    # Retain the live pair; a local storage/queue problem must
                    # not discard text or kill other users' maintenance work.
                    continue
                finalized.append(user_id)
        self.prune_offline_sessions()
        return finalized


    def prune_offline_sessions(self, *, max_profiles: int = 32) -> dict:
        """Bounded rotating expiry/migration maintenance; never invokes models."""
        if type(max_profiles) is not int or not 1 <= max_profiles <= 128:
            raise ValueError("max_profiles must be an integer from 1 to 128")
        with self._lock:
            active_ids = set(self._sessions)
            cursor = self._offline_prune_cursor
        try:
            user_ids = sorted(active_ids | set(self.store.list_user_ids()))
        except Exception:
            return {"inspected_profiles": 0, "failed_profiles": 1}
        start = next((i for i, uid in enumerate(user_ids) if cursor is None or uid > cursor), 0)
        selected = (user_ids[start:] + user_ids[:start])[:max_profiles]
        failed = 0
        for user_id in selected:
            with self._user_lock(user_id):
                try:
                    with self._lock:
                        pair = self._sessions.get(user_id)
                    if pair is not None:
                        engine = pair[0]
                    else:
                        persona = self.store.load(user_id)
                        if persona is None:
                            continue
                        engine = self._engine_for_persona(persona)
                    engine.prepare_offline_sessions()
                except Exception:
                    # No exception content is public; one corrupt profile
                    # cannot prevent expiry/deletion for other people.
                    failed += 1
                finally:
                    with self._lock:
                        self._offline_prune_cursor = user_id
        return {"inspected_profiles": len(selected), "failed_profiles": failed}


    def _maybe_checkpoint(
        self, engine: Engine, ctx: EngineContext
    ) -> None:
        previous = int(ctx.working_memory.get("_last_checkpoint_turn", 0))
        if len(ctx.turns) - previous >= self.checkpoint_turns:
            engine.checkpoint_session(ctx)


    def forget(self, user_id):
        """Delete durable data before discarding the last in-memory copy."""
        with self._user_lock(user_id):
            deleted = self.store.delete(user_id)
            with self._lock:
                active = self._sessions.pop(user_id, None)
                self._session_activity.pop(user_id, None)
            return {"ok": True, "deleted": bool(deleted),
                    "active_session_cleared": active is not None,
                    "local_scope": ["文字检查点", "会后待处理会话原文与恢复记录", "记忆", "人物假设", "会后证据审查与不采纳记录"],
                    "external_data_retracted": False}

    def close(self):
        for user_id in list(self._sessions):
            if not self.end(user_id)["ok"]:
                raise RuntimeError("会话尚未安全封存；请检查本地存储。")

    def capabilities(self):
        return {
            "reasoning_architecture": {
                "delivery_block": self.delivery_block,
                "prompt_version": prompt_version_for(self.delivery_block),
                "online_decision_schema": configured_online_decision_version(),
            },
            "data_practices": {
                "offline_session_text_retention_hours": OFFLINE_TTL_HOURS,
                "offline_queue_deleted_with_profile": True,
            },
            "session": {
                "offline_max_attempts_per_session": OFFLINE_MAX_ATTEMPTS,
                "offline_recovery_during_chat": False,
                "periodic_paid_retry_of_closed_sessions": False,
            },
        }


def make_handler(app: TextWebApp):
    class Handler(BaseHTTPRequestHandler):
        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy", "frame-ancestors 'none'"
            )


        def _request_allowed(self, *, api: bool) -> bool:
            raw_host = str(self.headers.get("Host") or "").strip()
            parsed_host = urlparse(f"//{raw_host}")
            hostname = (parsed_host.hostname or "").lower()
            if hostname not in app.allowed_hosts:
                self._json(
                    {"ok": False, "error": "请求主机不受信任"}, 421
                )
                return False
            if not api:
                return True
            supplied = str(
                self.headers.get("X-Aria-Demo-Token") or ""
            )
            if not hmac.compare_digest(supplied, app.demo_token):
                self._json(
                    {"ok": False, "error": "本地体验令牌无效"}, 403
                )
                return False
            origin = str(self.headers.get("Origin") or "").strip()
            if origin:
                parsed_origin = urlparse(origin)
                if (
                    parsed_origin.scheme not in {"http", "https"}
                    or parsed_origin.netloc.lower() != raw_host.lower()
                ):
                    self._json(
                        {"ok": False, "error": "跨站请求已拒绝"}, 403
                    )
                    return False
            return True


        def _json(self, obj: dict, status: int = 200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._security_headers()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


        def log_message(self, *a):                   # 安静;需要时看客户端 UI
            pass


    return Handler


class AriaWebServer(ThreadingHTTPServer):
    """带空闲会话收口线程的本地参考服务器。"""

    # ThreadingHTTPServer otherwise makes request threads daemons and silently
    # excludes them from server_close's join. Accepted writes must finish before
    # we seal chats or close the care database.
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, handler, app: TextWebApp):
        super().__init__(address, handler)
        self._aria_app = app
        self.maintenance_errors: dict[str, str] = {}
        self._reaper_stop = threading.Event()
        interval = max(1.0, min(30.0, app.session_idle_seconds / 4.0))
        self._reaper_thread = threading.Thread(
            target=self._reap_loop,
            args=(interval,),
            daemon=True,
            name="aria-session-reaper",
        )
        self._reaper_thread.start()

    def _reap_loop(self, interval: float) -> None:
        while not self._reaper_stop.wait(interval):
            self._maintenance_once()

    def _maintenance_once(self) -> None:
        # A chat checkpoint failure must not stop an independently enabled
        # care scheduler. Expose sanitized health instead of killing the loop.
        for name in ("finalize_idle_sessions", "tick_care_all"):
            try:
                getattr(self._aria_app, name)()
                self.maintenance_errors.pop(name, None)
            except Exception as exc:
                self.maintenance_errors[name] = type(exc).__name__
        # Hosts without an automatic-review option retain their existing
        # maintenance contract. A review failure cannot stop care scheduling.
        auto_review = getattr(self._aria_app, "tick_auto_review", None)
        if callable(auto_review):
            try:
                auto_review()
                self.maintenance_errors.pop("tick_auto_review", None)
            except Exception as exc:
                self.maintenance_errors["tick_auto_review"] = type(exc).__name__

    def shutdown(self) -> None:
        self._reaper_stop.set()
        super().shutdown()

    def server_close(self) -> None:
        self._reaper_stop.set()
        try:
            # Close the listener and drain accepted request handlers first. A
            # handler delayed before app.start must not reopen a sealed app.
            super().server_close()
        finally:
            if threading.current_thread() is not self._reaper_thread:
                self._reaper_thread.join()
            self._aria_app.close()


def _reasoning_view(ctx: EngineContext) -> dict:
    """公开可审计的状态摘要，不暴露模型私有思维链。"""
    selected = ctx.reasoning.selected
    return {
        "hypotheses": [
            {
                "id": item.id,
                "label": item.label,
                "confidence": round(item.confidence, 2),
                "evidence_count": len(item.evidence_ids),
                "has_counterevidence": bool(item.counterevidence_ids),
                "next_check": item.clarification_question,
            }
            for item in ctx.reasoning.hypotheses[:4]
        ],
        "selected": ({
            "id": selected.id,
            "label": selected.label,
            "goal": selected.goal,
            "consent_required": selected.consent_required,
            "score": round(selected.score, 2),
        } if selected is not None else None),
        "observation": ({
            "mode": ctx.reasoning.observation_plan.mode,
            "reason": ctx.reasoning.observation_plan.reason,
            "question": ctx.reasoning.observation_plan.question,
            "information_gain": round(
                ctx.reasoning.observation_plan.information_gain, 2
            ),
        } if ctx.reasoning.observation_plan is not None else None),
        "skills": [
            {
                "id": item.skill_id,
                "intent": item.intent,
                "goal": item.goal,
                "status": item.task_status,
                "missing_slots": list(item.missing_slots),
                "capability_available": item.capability_available,
            }
            for item in (
                ctx.working_memory.get("skill_proposals") or []
            )[:3]
        ],
        "active_tasks": [
            {
                "id": item.id,
                "status": item.status,
                "goal": item.goal,
            }
            for item in ctx.active_tasks[-5:]
        ],
        "last_outcome": ({
            "value": ctx.reasoning.outcomes[-1].value,
            "signal": ctx.reasoning.outcomes[-1].signal,
        } if ctx.reasoning.outcomes else None),
        "personality_decision": ctx.working_memory.get(
            "personality_decision"
        ),
        "online_turn_decision": ctx.working_memory.get(
            "online_turn_decision"
        ),
        "intervention_delivery": ctx.working_memory.get(
            "intervention_delivery"
        ),
        "model_route": ctx.working_memory.get("model_route"),
        "risk": ctx.working_memory.get("risk_flag"),
        "care_events": ctx.working_memory.get("care_events") or [],
    }


def _profile_view(engine: Engine) -> dict:
    persona = engine.persona
    if persona is None:
        return {"stage": "cold_start", "origin": "blank"}
    return {
        "stage": persona.understanding_stage(),
        "origin": persona.profile_origin,
        "sessions_completed": (
            persona.person_model_state.sessions_completed
        ),
        "sessions_reflected": (
            persona.person_model_state.sessions_observed
        ),
    }
