"""A fictional, persistent companion demo using the existing Engine.

The story belongs to the presentation layer. Only a submitted utterance reaches
the engine. Reading a saved run never creates a session or invokes a model.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timezone
from http.cookies import SimpleCookie
import json
from pathlib import Path
import re
import secrets
import time
from urllib.parse import parse_qs, urlparse
import uuid

from ..engine.memory.crypto import decrypt_if_needed, encrypt_if_keyed
from ..engine.memory.atomic_file import replace_with_retry
from ..engine.memory.offline_queue import OfflineSessionQueue
from ..engine.memory.persona_store import PersonaStore
from ..engine.memory.stated_preferences import (
    heuristic_preferences, merge_preferences, preference_admission, preference_is_usable,
    transition_preference,
)
from ..engine.minimal_care import MinimalCareService
from ..engine.minimal_care_context import build_care_context
from ..engine.policy.llm import OpenRouterLLMClient
from ..engine.policy.minimal_calls import MinimalCallGuard, validate_limits
from ..engine.policy.product_calls import ProductCallGuard
from ..engine.model.persona import PersonaProfile
from ..engine.cognitive import cognitive_action, cognitive_owner_view
from ..engine.perception.external_emotion import parse_external_emotion, external_emotion_payload
from ..engine.perception.signals import is_simulated
from ..runtime import create_minimal_clients, create_minimal_engine
from ..scenes.registry import get_scene
from .web import AriaWebServer, TextWebApp, make_handler

_PAGE = Path(__file__).resolve().parents[1] / "assets" / "voice" / "minimal.html"
_CARE_PAGE = _PAGE.with_name("care.html")
_COOKIE = "aria_care_session"
_POST_BODY_LIMIT = 16384
_POST_BODY_TIMEOUT_SECONDS = 3.0
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_USER = "fictional-chen"
_CARE_QUERY = re.compile(
    r"提醒|通知|授权|撤回|延后|推迟|取消|确认|回执|护理|"
    r"(?:现在|目前|当前|什么).{0,6}状态|告诉.{0,6}(?:她|他|家人|女儿|儿子)"
)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


class MinimalWebApp(TextWebApp):
    """One local fictional run; its directory must not be a real profile store."""

    def __init__(self, data_dir: str | Path, *, clients=None,
                 session_idle_seconds=1200, care_service=None,
                 model_max_requests=None, model_stop_cost_usd=None,
                 model_reservation_policy=None, model_call_guard=None):
        self.clients = clients or create_minimal_clients()
        self.call_guard = None
        self._owns_call_guard = model_call_guard is None
        network_clients = [client for client in (
            self.clients.online, self.clients.memory, self.clients.reflection,
        ) if isinstance(client, OpenRouterLLMClient)]
        if network_clients and model_call_guard is None:
            validate_limits(model_max_requests, model_stop_cost_usd)
            if model_reservation_policy is None:
                raise ValueError("真实模型需要发送前费用预留策略。")
        elif not network_clients and (model_max_requests is not None or model_stop_cost_usd is not None or model_call_guard is not None):
            raise ValueError("模型调用限额只用于真实模型装配。")
        self.mode = "live" if self.clients.model_calls_enabled else "local_preview"
        # Separate preview/live profiles even when a developer changes launch flags.
        self.run_dir = Path(data_dir).resolve() / self.mode
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = self.run_dir / "run.json"
        self._run = self._load_run()
        self._storage_failed = False
        if care_service is not None and (care_service.external or care_service.allow_external):
            raise ValueError("最小虚构Web只允许本地护理通知；真实外发服务须独立配置与验收。")
        if network_clients:
            from ..engine.policy.reserved_calls import ReservedCallGuard, validate_reservation_policy
            ledger_path = self.run_dir / "model_calls.sqlite3"
            if model_call_guard is None and not ledger_path.exists() and (self._run["turns"] or self._run["reviews"]):
                raise ValueError("已有真实模式记录缺少调用台账；须先核查旧费用，不能自动从零计费。")
            if model_call_guard is not None and not isinstance(model_call_guard, ReservedCallGuard):
                raise ValueError("共享调用账本必须使用发送前费用预留。")
            if model_call_guard is not None:
                if model_max_requests is not None or model_stop_cost_usd is not None:
                    limits = validate_limits(model_max_requests, model_stop_cost_usd)
                    if limits != (model_call_guard.max_requests, model_call_guard.threshold):
                        raise ValueError("显式限额与共享调用账本不一致。")
                if model_reservation_policy is not None and validate_reservation_policy(model_reservation_policy) != model_call_guard.policy:
                    raise ValueError("显式费用预留策略与共享调用账本不一致。")
            guard = model_call_guard or ProductCallGuard(
                ledger_path, max_requests=model_max_requests,
                max_observed_cost_usd=model_stop_cost_usd,
                reservation_policy=model_reservation_policy)
            try:
                binding = {"ledger_id": guard.ledger_id, "policy_sha256": guard.policy_sha256}
                # Accounting identity is deliberately outside deletable chat
                # history. Forgetting a fictional conversation cannot reset its
                # shared allowance or attach a newly empty ledger.
                binding_path = self.run_dir / "model_allowance.json"
                if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                    raise ValueError("已有真实体验绑定了不同调用账本，不能从新账本重新计费。")
                prior_binding = self._run.get("model_allowance")
                if prior_binding is not None and prior_binding != binding:
                    raise ValueError("已有真实体验绑定了不同调用账本，不能从新账本重新计费。")
                if prior_binding is None and (self._run["turns"] or self._run["reviews"]):
                    raise ValueError("已有真实体验缺少账本归属，须先核对费用，不能补绑空账本。")
                for client in network_clients:
                    guard.attach(client)
                try:
                    with binding_path.open("x", encoding="utf-8") as stream:
                        json.dump(binding, stream, sort_keys=True, allow_nan=False)
                except FileExistsError:
                    if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                        raise ValueError("调用账本归属在打开期间发生变化，未启动模型。") from None
                # Persists with the first ordinary run save; cloned experiment
                # branches retain the same durable shared allowance identity.
                self._run["model_allowance"] = binding
            except Exception:
                if self._owns_call_guard:
                    guard.close()
                raise
            self.call_guard = guard
        try:
            self.care = care_service or MinimalCareService(self.run_dir / "care.sqlite3")
            super().__init__(
                store=PersonaStore(self.run_dir / "profiles"), clients=self.clients,
                checkpoint_turns=1,
                session_idle_seconds=session_idle_seconds,
                demo_token=secrets.token_urlsafe(24),
            )
        except BaseException:
            if self.call_guard and self._owns_call_guard:
                self.call_guard.close()
            if care_service is None and hasattr(self, "care"):
                self.care.close()
            raise
        # Local demo token is not caregiver authentication.
        self.allowed_hosts = {"127.0.0.1", "localhost", "::1"}

    def _load_run(self):
        if not self.run_path.exists():
            return {"version": 1, "mode": self.mode, "run_id": uuid.uuid4().hex,
                    "sessions": [], "turns": [], "reviews": [], "preference_reviews": []}
        value = json.loads(decrypt_if_needed(
            self.run_path.read_text(encoding="utf-8"), self.run_dir, who="minimal-run"))
        if (not isinstance(value, dict) or value.get("version") != 1
                or value.get("mode") != self.mode
                or any(not isinstance(value.get(key), list)
                       for key in ("sessions", "turns", "reviews"))):
            raise ValueError("minimal run is invalid; original files were retained")
        # Older runs have no local-rule preference review history.
        if "preference_reviews" not in value:
            value["preference_reviews"] = []
        if not isinstance(value["preference_reviews"], list):
            raise ValueError("minimal preference reviews are invalid; original files were retained")
        return value

    def _save_run(self):
        tmp = self.run_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(encrypt_if_keyed(json.dumps(
                self._run, ensure_ascii=False, indent=2), self.run_dir), encoding="utf-8")
            replace_with_retry(tmp, self.run_path)
        except Exception:
            self._storage_failed = True
            self._halt_models()
            raise RuntimeError("minimal_run_store_failed") from None

    def _halt_models(self):
        if self.call_guard:
            try:
                self.call_guard.halt()
            except Exception:
                # Its own fail-closed flag and durable pending record remain.
                pass

    def _model_scope(self, kind, operation_id, session_id):
        return (self.call_guard.scope(kind, operation_id, session_id)
                if self.call_guard else nullcontext())

    def _models_stopped(self):
        return self.call_guard is not None and self.call_guard.snapshot()["stopped"]

    def resume_models(self, request_id, pause_id, *, confirmed=False, read_only=False, actor=_USER):
        """The explicit local owner surface; replay never changes the ledger."""
        with self._user_lock(_USER):
            if read_only:
                return self._result(False, "回放只读，不能恢复模型调用。")
            if actor != _USER:
                raise PermissionError("只有本体验的本人可以恢复调用。")
            if confirmed is not True or not isinstance(self.call_guard, ProductCallGuard):
                return self._result(False, "请本人明确确认恢复；严格实验账本不支持产品恢复。")
            if self._unresolved():
                return self._result(False, "存在保存失败或结果不明，不能通过恢复按钮继续调用。")
            try:
                self.call_guard.resume(request_id, pause_id)
            except (ValueError, RuntimeError):
                return self._result(False, "当前暂停不能恢复，或状态已变化；请核查费用、限额与保存记录。")
            return self._result()

    def _engine_for_persona(self, persona):
        return create_minimal_engine(get_scene(self.scene_id), persona,
                                     clients=self.clients, store=self.store,
                                     delivery_block=self.delivery_block)

    def _seed_persona(self, user_id):
        # No biography, psychological interpretation, synthetic observations,
        # future dialogue, or desired ending is seeded into this profile.
        return PersonaProfile(user_id=user_id, profile_origin="fictional_roleplay")

    def _profile(self):
        pair = self._sessions.get(_USER)
        return pair[0].persona if pair else self.store.load(_USER)

    def _understanding(self, persona):
        if persona is None:
            return {"facts": [], "hypotheses": [], "stated_preferences": [],
                    "evidence": [], "changes": []}
        hypotheses = [item.model_dump() for item in persona.person_hypotheses]
        sources = {(row["session_id"], row.get("turn_index")): row["text"]
                   for row in self._run["turns"]}
        for item in hypotheses:
            for evidence in item["evidence"]:
                evidence["quotes"] = [
                    {"turn_index": index, "quote": sources[(evidence["session_id"], index)]}
                    for index in evidence["turn_indices"]
                    if (evidence["session_id"], index) in sources]
        return {
            "facts": [item.model_dump() for item in persona.episodic_memory],
            "hypotheses": hypotheses,
            "stated_preferences": [{**item.model_dump(), "admission": preference_admission(item),
                                    "can_confirm": preference_is_usable(item, allow_rule_preview=self.mode == "local_preview"),
                                    "usable": item.status == "confirmed" and preference_is_usable(
                                        item, allow_rule_preview=self.mode == "local_preview")}
                                   for item in persona.stated_preferences],
            "evidence": [item.model_dump() for item in persona.evidence_review_ledger],
            "changes": [change for review in [*self._run["reviews"], *self._run["preference_reviews"]]
                        for change in review.get("changes", [])],
            "confidence_notice": "置信值是工程证据累计量，不是准确率或诊断概率。",
        }

    def state(self):
        """Strictly read-only, including after restart and during playback."""
        with self._user_lock(_USER):
            persona = self._profile()
            sessions = deepcopy(self._run["sessions"])
            queued = {job.session_id for job in persona.offline_sessions} if persona else set()
            completed = set(persona.person_model_state.completed_session_ids) if persona else set()
            for row in sessions:
                if row["session_id"] in completed:
                    row["status"] = "reviewed"
                elif row["session_id"] in queued:
                    row["status"] = "sealed"
                elif row["status"] == "active" and _USER not in self._sessions:
                    pending = persona.pending_session if persona else None
                    if (pending is None or pending.session_id != row["session_id"]
                            or not self._checkpoint_is_current(pending)):
                        row["status"] = "expired"
            active = next((row for row in reversed(sessions) if row["status"] == "active"), None)
            return {
                "ok": True, "mode": self.mode, "run_id": self._run["run_id"],
                "model_calls_enabled": self.clients.model_calls_enabled,
                "model_usage": self.call_guard.snapshot() if self.call_guard else {"configured": False},
                "session_id": active["session_id"] if active else None,
                "session_number": active["number"] if active else len(sessions),
                "sessions": sessions, "turns": deepcopy(self._run["turns"]),
                "reviews": deepcopy(self._run["reviews"]),
                "preference_reviews": deepcopy(self._run["preference_reviews"]),
                "profile": self._understanding(persona),
                "cognitive": cognitive_owner_view(persona),
                "offline": OfflineSessionQueue(persona).status() if persona else {"pending_count": 0},
                "storage_failed": self._storage_failed,
                "requires_review": self._unresolved(),
                "notice": ("本地流程示意：回复为占位；未执行模型、事实抽取或人物反思。"
                           if self.mode == "local_preview" else
                           "真实生成：在线和手动会后整理会调用云模型；输入为虚构角色扮演。"),
                "care_notice": "护理相关提问会附本人近期状态供本轮参考；动作仍按结构表单执行。通知仅写本地收件箱，未联系真实家属。关闭聊天后护理调度继续，停止服务后暂停。",
                "services": {name: False for name in (
                    "tts", "asr", "embedding", "acoustic", "visual", "omni", "external_notifications")},
            }

    def care_state(self):
        # A public fictional elder surface never accepts a caller-selected ID.
        return {"ok": True, **self.care.elder_view(_USER)}

    def care_action(self, action, payload):
        if not isinstance(payload, dict):
            raise ValueError("护理请求格式无效。")
        if action == "consent.grant" and payload.get("source") != "simulation":
            raise ValueError("虚构体验只接受模拟授权，不能用它真实外发。")
        result = self.care.elder_action(_USER, action, payload)
        return {"ok": True, **result, "state": self.care_state()}

    def caregiver_action(self, token, elder_id, action, payload):
        if not isinstance(payload, dict):
            raise ValueError("护理请求格式无效。")
        if action == "consent.grant" and payload.get("source") != "simulation":
            raise ValueError("虚构体验只接受模拟授权，不能用它真实外发。")
        return {"ok": True, **self.care.mutate(token, elder_id, action, payload)}

    def tick_care_all(self):
        # Independent of an active chat, post-session models, or personality.
        self.care.tick()
        return [_USER]

    def _result(self, ok=True, error=None, **extra):
        return {"ok": ok, **({"error": error} if error else {}), **extra, "state": self.state()}

    def _unresolved(self):
        return self._storage_failed or any(row["status"] == "pending" or row.get("requires_review")
            for row in [*self._run["turns"], *self._run["reviews"], *self._run["preference_reviews"]])

    def preference_action(self, action, preference_id, *, read_only=False):
        """An explicit owner operation; care authorization is never changed."""
        with self._user_lock(_USER):
            if read_only:
                return self._result(False, "回放只读，不能改变相处说明。")
            if (not isinstance(action, str) or action not in {"confirm", "decline", "withdraw"}
                    or not isinstance(preference_id, str)):
                return self._result(False, "相处说明操作无效。")
            if self._storage_failed or (self._unresolved() and action != "withdraw"):
                return self._result(False, "存在保存失败或结果不明，请先核查记录。")
            persona = self._profile()
            if persona is None:
                return self._result(False, "没有已保存的相处说明。")
            candidate = persona.model_copy(deep=True)
            try:
                transition_preference(candidate, preference_id, action, allow_rule_preview=self.mode == "local_preview")
            except (ValueError, KeyError):
                return self._result(False, "这条说明已变化或操作不适用，请刷新后查看。")
            try:
                self.store.save(candidate)
            except Exception:
                self._storage_failed = True
                self._halt_models()
                return self._result(False, "相处说明保存失败，未显示为生效；请先核查本地存储。")
            pair = self._sessions.get(_USER)
            if pair is not None:
                pair[0]._accept_offline_profile(candidate)
            return self._result()

    def cognitive_action(self, action, *, observation_id=None, delete_records=None, read_only=False, actor=_USER):
        """The single fictional owner surface; caregiver routes do not expose it."""
        with self._user_lock(_USER):
            if read_only:
                return self._result(False, "回放只读，不能改变变化提醒。")
            if actor != _USER:
                raise PermissionError("只有本人可以操作变化提醒，护理员不能代开。")
            if (self._storage_failed or self._unresolved()) and action not in {"disable", "delete"}:
                return self._result(False, "请先核查保存失败或结果不明的操作。")
            persona = self._profile() or self._seed_persona(_USER)
            candidate = persona.model_copy(deep=True)
            try:
                cognitive_action(candidate, action, actor=_USER, observation_id=observation_id,
                                 delete_records=delete_records)
            except ValueError as exc:
                return self._result(False, str(exc))
            try:
                self.store.save(candidate)
            except Exception:
                self._storage_failed = True
                self._halt_models()
                return self._result(False, "变化提醒保存失败，未显示为生效；请核查本地存储。")
            pair = self._sessions.get(_USER)
            if pair:
                pair[0]._accept_offline_profile(candidate)
                if action in {"enable", "disable"}:
                    for key in ("_cognitive_source_turns", "_cognitive_session_at", "_cognitive_owner_enabled"):
                        pair[1].working_memory.pop(key, None)
            return self._result()

    def preview_preferences(self, request_id):
        """Explicit rule-only demonstration, separate from model offline work."""
        with self._user_lock(_USER):
            if self.mode != "local_preview":
                return self._result(False, "规则提取仅用于本地预览；真实模式请明确启动会后整理。")
            if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
                return self._result(False, "提取请求标识无效。")
            old = next((row for row in self._run["preference_reviews"]
                        if row["request_id"] == request_id), None)
            if old:
                return self._result(old["status"] == "completed", old.get("error"), duplicate=True)
            if self._unresolved() or self.state()["session_id"]:
                return self._result(False, "请先封存会话，并核查结果不明的请求。")
            persona = self._profile()
            reviewed = {row["session_id"] for row in self._run["preference_reviews"]}
            job = next((job for job in persona.offline_sessions if job.session_id not in reviewed), None) if persona else None
            if job is None:
                return self._result(False, "没有尚未提取相处说明的已封存会话。")
            info = next(row for row in OfflineSessionQueue(persona).status()["jobs"]
                        if row["session_id"] == job.session_id)
            if info["expired"] or job.source_completeness != "complete":
                return self._result(False, "原会话已到期或出处不完整，不能用于提取相处说明。")
            before = self._understanding(persona)
            row = {"request_id": request_id, "session_id": job.session_id,
                   "source": "local_rule", "status": "pending", "created_at": _stamp()}
            self._run["preference_reviews"].append(row)
            self._save_run()
            try:
                candidate = persona.model_copy(deep=True)
                ctx = OfflineSessionQueue._context(candidate, job)
                merge_preferences(candidate, heuristic_preferences(ctx, candidate))
                # Do not complete/remove offline jobs, create facts, or run reflection.
                self.store.save(candidate)
                row.update(status="completed", finished_at=_stamp(),
                           changes=_changes(before, self._understanding(candidate), job.session_id),
                           notice="只运行本地规则提议；未执行模型、事实抽取或人物反思。")
            except Exception:
                self._storage_failed = True
                row["requires_review"] = True
                row["error"] = "本地规则提取或保存未完成，结果需核查；不会自动重试。"
            self._save_run()
            return self._result(row["status"] == "completed", row.get("error"))

    def start(self):
        with self._user_lock(_USER):
            if self._unresolved():
                return self._result(False, "上次请求或保存结果不明，已停止继续调用；可查看原记录或清除本地虚构资料。")
            engine, ctx = self._session(_USER)
            if not any(row["session_id"] == ctx.session_id for row in self._run["sessions"]):
                self._run["sessions"].append({
                    "session_id": ctx.session_id, "number": len(self._run["sessions"]) + 1,
                    "status": "active", "started_at": _stamp(),
                })
                # A restart must resume this exact session, even before its first turn.
                engine.checkpoint_session(ctx)
                self._save_run()
            return self._result()

    def submit(self, text, turn_id, session_id, *, external_emotion=None):
        with self._user_lock(_USER):
            try:
                external_emotion = parse_external_emotion(external_emotion)
            except ValueError:
                return self._result(False, "外部情绪读数格式无效，未启动模型。")
            external_payload = external_emotion_payload(external_emotion) if external_emotion is not None else None
            if (not isinstance(turn_id, str) or not _ID.fullmatch(turn_id)
                    or not isinstance(text, str) or not text.strip() or len(text) > 500):
                return self._result(False, "请输入500字以内文字及有效轮次标识。")
            old = next((row for row in self._run["turns"] if row["turn_id"] == turn_id), None)
            if old:
                if (old["text"] != text or old["session_id"] != session_id
                        or old.get("external_emotion") != external_payload):
                    return self._result(False, "轮次标识已用于不同输入，未再次执行。")
                return self._result(old["status"] == "completed", old.get("error"), duplicate=True)
            if self._unresolved():
                return self._result(False, "请求或保存结果不明，不能自动重发或继续调用。")
            if self._models_stopped():
                return self._result(False, "后续模型调用已停止；请核查调用台账、未知费用或限额。封存、回放和护理仍可使用。")
            view = self.state()
            if not session_id or session_id != view["session_id"]:
                return self._result(False, "会话已封存或变化，请先开始下一次会话。")
            engine, ctx = self._session(_USER)
            if ctx.session_id != session_id:
                return self._result(False, "会话已过期，请重新开始。")
            if is_simulated(external_emotion):
                capability = getattr(engine, "external_emotion_experiment", None)
                if capability is None:
                    return self._result(False, "合成情绪仅可用于独立实验，不能写入普通体验。")
                try:
                    capability.validate(engine.store, engine.persona)
                except (ValueError, AttributeError):
                    return self._result(False, "合成情绪实验范围无效，未启动模型。")
            row = {"turn_id": turn_id, "session_id": session_id,
                   "session_number": view["session_number"], "turn_index": len(ctx.turns),
                   "text": text, "response": "", "status": "pending", "created_at": _stamp(),
                   "source": "local_preview" if self.mode == "local_preview" else "live"}
            if external_payload is not None:
                row["external_emotion"] = external_payload
            # Read a bounded snapshot from the same service used by the forms.
            # It is neither an action receipt for this turn nor user evidence.
            # Duplicate turns return above without refreshing their old basis.
            if _CARE_QUERY.search(text):
                row["care_context"] = build_care_context(self.care, _USER)
            self._run["turns"].append(row)
            # Persist the stable attempt before any model or business operation.
            self._save_run()
            started = time.perf_counter()
            try:
                ctx.working_memory.pop("prompt_evidence_snapshot", None)
                # Only the explicitly model-disabled preview can expose local
                # rule proposals. Reopening the same profile live resets this.
                ctx.working_memory["_local_preference_preview"] = self.mode == "local_preview"
                if "care_context" in row:
                    ctx.working_memory["minimal_care_context"] = deepcopy(row["care_context"])
                with self._model_scope("turn", turn_id, session_id):
                    result = super().text_turn(_USER, text, turn_id=turn_id, external_emotion=external_emotion)
                    if (isinstance(self.call_guard, ProductCallGuard) and
                            (ctx.working_memory.get("model_route") or {}).get("reply", {}).get("source") == "deterministic_fallback"):
                        self.call_guard.note_output_failure()
                if not result["ok"]:
                    raise RuntimeError("turn_failed")
                route = (ctx.working_memory.get("model_route") or {}).get("reply", {})
                failed = route.get("source") == "deterministic_fallback"
                from ..engine.policy.delivery_provenance import delivery_protection
                protection = delivery_protection(ctx)
                product_fallback = failed and isinstance(self.call_guard, ProductCallGuard)
                hard_stop = product_fallback and self.call_guard.snapshot()["stop_reason"] not in {None, "consecutive_failures"}
                local_rejection = (self.call_guard.snapshot().get("last_request_rejection", {})
                                   if product_fallback else {})
                local_rejected = local_rejection.get("operation_id") == turn_id
                row.update(status="failed" if failed else "completed",
                           response=(result["response"] if product_fallback and not hard_stop else "") if failed else result["response"],
                           source=("local_request_rejected" if local_rejected else
                                   "rule_fallback" if product_fallback and not hard_stop else "model_failed") if failed else (
                               "rule_safety" if ctx.working_memory.get("risk_flag") else
                               "rule_protection" if protection["protection_response_changed"] else row["source"]),
                           delivery_protection=protection,
                           emotion_summary={"label": ctx.emotion.label,
                                            "source": ctx.emotion.inference_source or "unavailable",
                                            "acoustic": "未观测", "visual": "未观测"},
                           strategy_summary=({"label": ctx.reasoning.selected.label,
                                              "goal": ctx.reasoning.selected.goal}
                                             if ctx.reasoning.selected else None),
                           elapsed_ms=round((time.perf_counter() - started) * 1000))
                if failed:
                    if product_fallback:
                        row["requires_review"] = bool(hard_stop)
                        row["error"] = ("本地发送前检查未通过，本轮未向模型发送请求。已停止后续调用；请核查输入额度、配置或记录完整性，恢复按钮不能解除。"
                                        if local_rejected else
                                        "本轮使用系统兜底回复，不是模型生成；连续三次失败已暂停，请本人明确恢复。"
                                        if self.call_guard.snapshot().get("can_resume") else
                                        "模型调用受费用、限额或完整性保护停止；请核查记录，不能用恢复按钮解除。" if hard_stop else
                                        "本轮使用系统兜底回复，不是模型生成；可继续下一轮，不重发本轮内容。")
                    else:
                        self._halt_models()
                        row["requires_review"] = True
                        row["error"] = "模型调用或返回失败，本轮没有可用的模型回复；已停止后续调用，需先核查结果与费用。"
            except Exception:
                self._halt_models()
                # The model might already have run. Keep the durable attempt pending;
                # a read or restart must not retry it under a new ID.
                row["error"] = "生成或保存未完成，结果可能不明；已停止自动重试。"
            finally:
                # Freeze what this turn's builder selected, including failures.
                # Never reconstruct old input from a later profile revision.
                evidence = ctx.working_memory.get("prompt_evidence_snapshot")
                row["input_evidence"] = deepcopy(evidence) if isinstance(evidence, dict) else {
                    "schema_version": "prompt_evidence_v1", "status": "not_built", "passed_to_client": False,
                    "recalled_episodes": [], "hypotheses": [], "stated_preferences": [],
                }
                sources = {(turn["session_id"], turn.get("turn_index")): turn["text"]
                           for turn in self._run["turns"]}
                for hypothesis in row["input_evidence"].get("hypotheses", []):
                    for source in hypothesis.get("provenance_not_injected", []):
                        source["quotes"] = [
                            {"turn_index": index, "quote": sources[(source["session_id"], index)]}
                            for index in source.get("turn_indices", [])
                            if (source["session_id"], index) in sources]
                row["input_evidence"].update(
                    recorded_at=_stamp(),
                    generation_status=("result_unknown" if row["status"] == "pending" else row["source"]),
                    turn_status=row["status"],
                )
                # Never reuse a snapshot in the next chat or offline reasoning.
                ctx.working_memory.pop("minimal_care_context", None)
            self._save_run()
            return self._result(row["status"] == "completed", row.get("error"))

    def end(self, user_id=_USER, *, process_offline=None):
        # Closing a tab/process always seals. Explicit review has its own endpoint.
        with self._user_lock(_USER):
            if _USER not in self._sessions:
                persona = self.store.load(_USER)
                pending = persona.pending_session if persona else None
                if (pending is not None and not pending.offline_finalize_pending
                        and self._checkpoint_is_current(pending)):
                    # A restarted process has a durable checkpoint but no RAM
                    # pair. Restore that exact context before sealing it.
                    self._session(_USER)
            result = super().end(_USER, process_offline=False)
            if not result["ok"]:
                self._storage_failed = True
            return result

    def _seal_active_session(self, user_id, pair):
        session_id = pair[1].session_id
        super()._seal_active_session(user_id, pair)
        for row in self._run["sessions"]:
            if row["session_id"] == session_id:
                row.update(status="sealed", sealed_at=_stamp())
        self._save_run()

    def seal(self, session_id):
        with self._user_lock(_USER):
            known = next((row for row in self._run["sessions"]
                          if row["session_id"] == session_id), None)
            if known is None:
                return self._result(False, "会话标识无效。")
            # An old tab must not seal the next conversation.
            active_id = self.state()["session_id"]
            if active_id and active_id != session_id:
                return self._result(False, "会话已变化，未结束新的会话。")
            result = self.end()
            return self._result(result["ok"], result.get("error"))

    def reflect(self, request_id):
        with self._user_lock(_USER):
            if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
                return self._result(False, "整理请求标识无效。")
            old = next((row for row in self._run["reviews"] if row["request_id"] == request_id), None)
            if old:
                return self._result(old["status"] == "completed", old.get("error"), duplicate=True)
            if not self.clients.model_calls_enabled:
                return self._result(False, "本地示意未启用模型；已封存原话，事实抽取和人物反思均未执行。")
            if self._models_stopped():
                return self._result(False, "后续模型调用已停止；请先核查调用台账与限额，不会启动会后整理。")
            if self._unresolved() or self.state()["session_id"]:
                return self._result(False, "请先安全封存当前会话，并处理结果不明的请求。")
            persona = self._profile()
            if persona is None or not persona.offline_sessions:
                return self._result(False, "没有待整理会话。")
            job = persona.offline_sessions[0]
            if job.attempts:
                return self._result(False, "上次会后整理未完成，不会盲目重试；请先核查失败记录与费用。")
            before = self._understanding(persona)
            row = {"request_id": request_id, "session_id": job.session_id,
                   "status": "pending", "created_at": _stamp()}
            self._run["reviews"].append(row)
            self._save_run()
            try:
                engine = self._engine_for_persona(persona)
                with self._model_scope("review", request_id, job.session_id):
                    result = engine.recover_offline_sessions(max_jobs=1)
                    success = result.get("last_run", {}).get("completed_jobs", 0) == 1
                    if isinstance(self.call_guard, ProductCallGuard):
                        audit = next((item for item in reversed(engine.persona.offline_session_audit)
                                      if item.session_id == job.session_id), None)
                        success = success and audit is not None and audit.outcome == "completed"
                        if not success:
                            self.call_guard.note_output_failure()
                        if result.get("last_run", {}).get("error_code") == "offline_store_failed":
                            self._storage_failed = True
                            self._halt_models()
                after = self._understanding(engine.persona)
                row.update(status="completed" if success else "failed", result=result,
                           changes=_changes(before, after, job.session_id), finished_at=_stamp())
                if not success:
                    if isinstance(self.call_guard, ProductCallGuard):
                        reason = self.call_guard.snapshot()["stop_reason"]
                        row["requires_review"] = reason not in {None, "consecutive_failures"}
                        row["error"] = "本次会后整理未完成，阶段记录保留且不自动重发；其他对话是否可用以调用状态为准。"
                    else:
                        row["requires_review"] = True
                        self._halt_models()
                        row["error"] = "会后整理未完成；请查看阶段状态，不会自动重试。"
            except Exception:
                self._halt_models()
                row["error"] = "会后整理或保存结果不明；已停止继续调用。"
            self._save_run()
            return self._result(row["status"] == "completed", row.get("error"))

    def forget_run(self):
        with self._user_lock(_USER):
            self.care.forget_elder(_USER)
            # Delete durable files first. A failed deletion retains the RAM copy.
            self.store.delete(_USER)
            for path in (self.run_path, self.run_path.with_suffix(".json.tmp")):
                path.unlink(missing_ok=True)
            self._sessions.clear()
            self._session_activity.clear()
            self._run = self._load_run()
            if self.call_guard:
                self._run["model_allowance"] = {"ledger_id": self.call_guard.ledger_id,
                                                "policy_sha256": self.call_guard.policy_sha256}
            self._storage_failed = False
            return self._result()

    def close(self):
        # Preserve failures in RAM until the owner explicitly retries sealing.
        result = None
        if self._sessions:
            result = self.end()
        self.care.close()
        if self.call_guard and self._owns_call_guard:
            self.call_guard.close()
        if result is not None and not result["ok"]:
            raise RuntimeError("会话尚未安全封存；请检查本地存储，最近成功的检查点保持不变。")


def _changes(before, after, session_id):
    changes = []
    for category in ("facts", "hypotheses", "stated_preferences"):
        prior = {item["id"]: item for item in before.get(category, [])}
        current = {item["id"]: item for item in after.get(category, [])}
        for key in prior.keys() | current.keys():
            old, new = prior.get(key), current.get(key)
            if old == new:
                continue
            changes.append({"session_id": session_id, "category": category, "id": key,
                            "kind": "added" if old is None else "removed" if new is None else "revised",
                            "before": old, "after": new})
    return changes or [{"session_id": session_id, "kind": "unchanged",
                       "summary": "本次没有新增或修订事实、人物假设或相处说明；不代表推理错误。"}]


def make_minimal_handler(app):
    base = make_handler(app)

    class Handler(base):
        def _read_post_body(self):
            """Consume a bounded body before a refusal closes the local socket.

            Headers and body can arrive separately. Closing an HTTP/1.0
            connection with its body unread can discard our refusal on Windows.
            Reading bytes is not authentication, parsing, or permission to act.
            """
            lengths = self.headers.get_all("Content-Length", [])
            raw_length = lengths[0].strip() if len(lengths) == 1 else ""
            if (not raw_length or not raw_length.isascii() or not raw_length.isdecimal()
                    or len(raw_length) > 8 or self.headers.get("Transfer-Encoding") is not None):
                raise ValueError("invalid_body_framing")
            size = int(raw_length)
            if size > _POST_BODY_LIMIT:
                raise ValueError("body_too_large")
            chunks = []
            remaining = size
            deadline = time.monotonic() + _POST_BODY_TIMEOUT_SECONDS
            previous_timeout = self.connection.gettimeout()
            try:
                while remaining:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        raise TimeoutError("body_read_timeout")
                    self.connection.settimeout(wait)
                    chunk = self.rfile.read1(remaining)
                    if not chunk:
                        raise ValueError("incomplete_body")
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                self.connection.settimeout(previous_timeout)
            return b"".join(chunks)

        def _care_token(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except Exception:
                return ""
            return cookie[_COOKIE].value if _COOKIE in cookie else ""

        def _care_json(self, obj, *, token=None, clear=False, status=200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._security_headers()
            if token is not None or clear:
                value = "" if clear else token
                ttl = 0 if clear else 8 * 3600
                self.send_header("Set-Cookie", f"{_COOKIE}={value}; Path=/api/care; HttpOnly; SameSite=Strict; Max-Age={ttl}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            route = urlparse(self.path).path
            if not self._request_allowed(api=route.startswith("/api/")):
                return
            if route in {"/", "/care"}:
                page = _PAGE if route == "/" else _CARE_PAGE
                body = page.read_text(encoding="utf-8").replace(
                    "__ARIA_DEMO_TOKEN__", app.demo_token).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._security_headers()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/minimal/state":
                try:
                    self._json(app.state())
                except Exception:
                    self._json({"ok": False, "error": "本地资料读取失败，原文件未覆盖。"}, 500)
            elif route in {"/api/minimal/care", "/api/care/state"}:
                try:
                    if route == "/api/minimal/care":
                        result = app.care_state()
                    else:
                        query = parse_qs(urlparse(self.path).query)
                        elder_id = (query.get("elder_id") or [None])[0]
                        result = {"ok": True, **app.care.view(self._care_token(), elder_id)}
                    result["maintenance_errors"] = dict(getattr(self.server, "maintenance_errors", {}))
                    self._json(result)
                except PermissionError:
                    self._json({"ok": False, "error": "请登录有对应老人权限的护理员账号。"}, 403)
                except ValueError as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                except Exception:
                    self._json({"ok": False, "error": "护理资料读取失败，请检查本地存储。"}, 500)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            route = urlparse(self.path).path
            actions = {
                "/api/minimal/start": lambda body: app.start(),
                "/api/minimal/text": lambda body: app.submit(body.get("text"), body.get("turn_id"), body.get("session_id"), external_emotion=body.get("external_emotion")),
                "/api/minimal/seal": lambda body: app.seal(body.get("session_id")),
                "/api/minimal/reflect": lambda body: app.reflect(body.get("request_id")),
                "/api/minimal/resume-models": lambda body: app.resume_models(
                    body.get("request_id"), body.get("pause_id"), confirmed=body.get("confirmed"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/preference": lambda body: app.preference_action(
                    body.get("action"), body.get("id"), read_only=body.get("replay", False)),
                "/api/minimal/preference-review": lambda body: app.preview_preferences(body.get("request_id")),
                "/api/minimal/cognitive": lambda body: app.cognitive_action(
                    body.get("action"), observation_id=body.get("id"), delete_records=body.get("delete_records"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/forget": lambda body: app.forget_run(),
                "/api/minimal/care": lambda body: app.care_action(body.get("action"), body.get("payload")),
                "/api/care/action": lambda body: app.caregiver_action(
                    self._care_token(), body.get("elder_id"), body.get("action"), body.get("payload")),
            }
            try:
                raw_body = self._read_post_body()
            except (ValueError, OSError):
                self.close_connection = True
                try:
                    self._json({"ok": False, "error": "请求格式、大小或读取时限无效。"}, 400)
                except OSError:
                    pass  # The client may already have closed its connection.
                return
            if route not in actions and route not in {"/api/care/login", "/api/care/logout"}:
                self.send_error(404)
                return
            if not self._request_allowed(api=True):
                return
            try:
                body = json.loads(raw_body.decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self._json({"ok": False, "error": "请求格式或大小无效。"}, 400)
                return
            try:
                if route == "/api/care/login":
                    token = app.care.login(body.get("username"), body.get("password"))
                    self._care_json({"ok": True}, token=token)
                    return
                if route == "/api/care/logout":
                    app.care.logout(self._care_token())
                    self._care_json({"ok": True}, clear=True)
                    return
                result = actions[route](body)
                self._json(result, 200 if result["ok"] else 409)
            except PermissionError:
                self._json({"ok": False, "error": "账号验证失败或无权访问该老人。"}, 403)
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except Exception:
                self._json({"ok": False, "error": "本地处理或保存失败；请求结果可能不明，请只刷新查看，不要重发。"}, 500)

    return Handler


def make_minimal_server(app, host="127.0.0.1", port=8080):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("最小Demo目前只允许本机访问")
    return AriaWebServer((host, port), make_minimal_handler(app), app)
