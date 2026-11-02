"""A fictional, persistent companion demo using the existing Engine.

The story belongs to the presentation layer. Only a submitted utterance reaches
the engine. Reading a saved run never creates a session or invokes a model.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timezone
from http.cookies import SimpleCookie
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

from ..engine.memory.crypto import decrypt_if_needed, encrypt_if_keyed
from ..engine.memory.atomic_file import replace_with_retry
from ..engine.memory.offline_queue import OfflineSessionQueue
from ..engine.memory.offline_merge import merge_offline_stage, OfflineReviewCancelled
from ..engine.memory.persona_store import PersonaStore
from ..engine.memory.stated_preferences import (
    heuristic_preferences, merge_preferences, preference_admission, preference_is_usable,
    transition_preference,
)
from ..engine.memory.preference_dialogue import (
    prepare_preference_invitation, acknowledge_preference_invitation, preference_invitation_view,
)
from ..engine.explicit_sharing import ExplicitSharingService, SharingContext
from ..engine.minimal_care import MinimalCareService
from ..engine.minimal_care_context import build_care_context
from ..engine.policy.llm import OpenRouterLLMClient
from ..engine.policy.minimal_calls import MinimalCallGuard, validate_limits
from ..engine.policy.product_calls import ProductCallGuard
from ..engine.policy.release_prompt import CompactPromptBudgetExceeded
from ..engine.model.persona import PersonaProfile
from ..engine.cognitive import cognitive_action, cognitive_owner_view, stage_cognitive_readiness
from ..engine.perception.external_emotion import parse_external_emotion, external_emotion_payload
from ..engine.perception.signals import is_simulated
from ..runtime import create_minimal_clients, create_minimal_engine
from ..scenes.registry import get_scene
from .web import AriaWebServer, TextWebApp, make_handler
from .auto_review import (
    default_settings as default_auto_review, validate_settings as validate_auto_review,
    view as auto_review_view, stable_request_id as auto_review_request_id,
    source_digest as auto_review_source_digest, command_digest as auto_review_command_digest,
    local_day as auto_review_day, AutoReviewQuota,
)

_PAGE = Path(__file__).resolve().parents[1] / "assets" / "voice" / "minimal.html"
_CARE_PAGE = _PAGE.with_name("care.html")
_COOKIE = "aria_care_session"
_POST_BODY_LIMIT = 16384
_POST_BODY_TIMEOUT_SECONDS = 3.0
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_USER = "fictional-chen"
_DEFAULT_VERSION = object()
_CARE_QUERY = re.compile(
    r"提醒|通知|授权|撤回|延后|推迟|取消|确认|回执|护理|"
    r"(?:现在|目前|当前|什么).{0,6}状态|告诉.{0,6}(?:她|他|家人|女儿|儿子)"
)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


from .settled_validation import SettledValidationAppMixin
from .settled_reflection import SettledReflectionAppMixin
from .held_input import HeldInputAppMixin


class MinimalWebApp(HeldInputAppMixin,SettledReflectionAppMixin,SettledValidationAppMixin,TextWebApp):
    """One local fictional run; its directory must not be a real profile store."""
    _settled_owner = _USER

    def __init__(self, data_dir: str | Path, *, clients=None,
                 session_idle_seconds=1200, care_service=None, sharing_service=None,
                 model_max_requests=None, model_stop_cost_usd=None,
                 model_reservation_policy=None, model_call_guard=None,
                 safety_delivery_version=_DEFAULT_VERSION,
                 safety_response_instruction_version=_DEFAULT_VERSION,
                 premodel_safety_version=_DEFAULT_VERSION):
        self.clients = clients or create_minimal_clients()
        from ..release import engine_options
        defaults = engine_options() if self.clients.release_profile in ("release_v1", "release_v2", "release_max_v1", "release_deepseek_v1") else {}
        safety_delivery_version = defaults.get("safety_delivery_version", "legacy") if safety_delivery_version is _DEFAULT_VERSION else safety_delivery_version
        safety_response_instruction_version = defaults.get("safety_response_instruction_version", "legacy") if safety_response_instruction_version is _DEFAULT_VERSION else safety_response_instruction_version
        premodel_safety_version = defaults.get("premodel_safety_version", "legacy") if premodel_safety_version is _DEFAULT_VERSION else premodel_safety_version
        from ..engine.risk.premodel_task import validate_premodel_safety_version
        self.premodel_safety_version = validate_premodel_safety_version(premodel_safety_version)
        from ..engine.risk.safety_delivery import validate_delivery_version
        self.safety_delivery_version = validate_delivery_version(safety_delivery_version)
        from ..engine.policy.safety_instruction import SafetyResponseInstruction
        self._safety_response_instruction = SafetyResponseInstruction(safety_response_instruction_version)
        if self.premodel_safety_version != "legacy":
            from ..engine.risk.premodel_task import validate_premodel_configuration
            validate_premodel_configuration(self.premodel_safety_version, self.safety_delivery_version,
                self.safety_response_instruction_version, "online_turn_decision_v4_evidence")
        if self.safety_delivery_version != "legacy":
            from ..engine.policy import PolicyEngine
            from ..engine.policy.online_decision import EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE
            if PolicyEngine(fast_llm=self.clients.online).online_decision_version not in EVIDENCE_DECISION_VERSIONS:
                raise ValueError("conditional_safety_delivery_requires_online_v4")
        if self.safety_response_instruction_version != "legacy":
            from ..engine.policy import PolicyEngine
            self._safety_response_instruction.validate(
                PolicyEngine(fast_llm=self.clients.online).online_decision_version)
        self.call_guard = None
        self._owns_call_guard = model_call_guard is None
        network_clients = [client for client in (
            self.clients.online, self.clients.memory, self.clients.reflection, self.clients.safety_review,
            self.clients.confirm_judge,
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
        self._auto_review_quota = AutoReviewQuota(self.run_dir / "auto_review_jobs.sqlite3")
        self._auto_review_quota.validate_requests({row["request_id"]: row["started_on"]
            for row in self._run["auto_review"]["events"] if row["started_on"] is not None})
        self._storage_failed = False
        self._profile_write_conflict = False
        if care_service is not None and (care_service.external or care_service.allow_external):
            raise ValueError("最小虚构Web只允许本地护理通知；真实外发服务须独立配置与验收。")
        if sharing_service is not None:
            raise ValueError("虚构Web固定使用独立本地转达；不能注入外部发送服务。")
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
            guard_type=ProductCallGuard
            if self.clients.release_profile in ("release_max_v1", "release_deepseek_v1"):
                from ..engine.policy.max_release.guard import MaxReleaseCallGuard
                guard_type=MaxReleaseCallGuard
            guard = model_call_guard or guard_type(
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
            self.sharing = ExplicitSharingService(self.run_dir / "sharing.sqlite3", owner_id=_USER)
            super().__init__(
                store=PersonaStore(self.run_dir / "profiles"), clients=self.clients,
                checkpoint_turns=1,
                session_idle_seconds=session_idle_seconds,
                demo_token=secrets.token_urlsafe(24),
                **({"safety_delivery_version": self.safety_delivery_version}
                   if self.safety_delivery_version != "legacy" else {}),
                **({"safety_response_instruction_version": self.safety_response_instruction_version}
                   if self.safety_response_instruction_version != "legacy" else {}),
                **({"premodel_safety_version": self.premodel_safety_version}
                   if self.premodel_safety_version != "legacy" else {}),
            )
        except BaseException:
            if self.call_guard and self._owns_call_guard:
                self.call_guard.close()
            if hasattr(self, "sharing"):
                self.sharing.close()
            if care_service is None and hasattr(self, "care"):
                self.care.close()
            raise
        # Local demo token is not caregiver authentication.
        self.allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        self._review_thread = None
        self._waiting_submission = None
        self._review_usage_snapshot = None
        self._auto_review_dispatching = None
        self._closing = False
        self._sharing_closed_view = None

    def _load_run(self):
        if not self.run_path.exists():
            return {"version": 1, "mode": self.mode, "run_id": uuid.uuid4().hex,
                    "sessions": [], "turns": [], "reviews": [], "preference_reviews": [],
                    "auto_review": default_auto_review(),
                    **({"safety_delivery_version": self.safety_delivery_version}
                       if self.safety_delivery_version != "legacy" else {}),
                    **({"safety_response_instruction_version": self.safety_response_instruction_version}
                       if self.safety_response_instruction_version != "legacy" else {}),
                    **({"premodel_safety_version": self.premodel_safety_version}
                       if self.premodel_safety_version != "legacy" else {})}
        value = json.loads(decrypt_if_needed(
            self.run_path.read_text(encoding="utf-8"), self.run_dir, who="minimal-run"))
        if (not isinstance(value, dict) or value.get("version") != 1
                or value.get("mode") != self.mode
                or any(not isinstance(value.get(key), list)
                       for key in ("sessions", "turns", "reviews"))):
            raise ValueError("minimal run is invalid; original files were retained")
        if value.get("safety_delivery_version", "legacy") != self.safety_delivery_version:
            raise ValueError("minimal_run_safety_delivery_version_mismatch")
        if value.get("safety_response_instruction_version", "legacy") != self.safety_response_instruction_version:
            raise ValueError("minimal_run_safety_response_instruction_version_mismatch")
        if value.get("premodel_safety_version", "legacy") != self.premodel_safety_version:
            raise ValueError("minimal_run_premodel_safety_version_mismatch")
        # Older runs have no local-rule preference review history.
        if "preference_reviews" not in value:
            value["preference_reviews"] = []
        if not isinstance(value["preference_reviews"], list):
            raise ValueError("minimal preference reviews are invalid; original files were retained")
        if "auto_review" not in value:
            value["auto_review"] = default_auto_review()
        validate_auto_review(value["auto_review"], value["run_id"],
                             {row["session_id"] for row in value["sessions"]})
        return value

    def _save_run(self):
        tmp = self.run_path.with_suffix(".json.tmp")
        try:
            self._held_before_run_save()
            tmp.write_text(encrypt_if_keyed(json.dumps(
                self._run, ensure_ascii=False, indent=2), self.run_dir), encoding="utf-8")
            replace_with_retry(tmp, self.run_path)
            self._held_after_run_save()
        except Exception:
            self._storage_failed = True
            self._halt_models()
            raise RuntimeError("minimal_run_store_failed") from None

    def _halt_models(self):
        self._halt_models_with_mode(hard=False)

    def _halt_held_models(self):
        # Integrity failures must not be softened by an experiment subclass's
        # ordinary model-output recovery override.
        self._halt_models_with_mode(hard=True)

    def _halt_models_with_mode(self, *, hard):
        if self.call_guard:
            if self._review_is_running() and threading.current_thread() is not self._review_thread:
                # Set the inherited dispatch stop immediately, without waiting
                # for the worker's ledger lock while holding the owner lock.
                self.call_guard._stop_requested.set()
                threading.Thread(target=self._halt_models_when_free, kwargs={"hard": hard}, daemon=True).start()
                return
            self._halt_models_when_free(hard=hard)

    def _halt_models_when_free(self, *, hard=False):
        if self.call_guard:
            try:
                if hard and hasattr(self.call_guard, "hard_stop_held"):
                    self.call_guard.hard_stop_held()
                else:
                    self.call_guard.halt()
            except Exception:
                # Its own fail-closed flag and durable pending record remain.
                pass

    def _review_is_running(self):
        return self._review_thread is not None and self._review_thread.is_alive()

    def _model_usage(self):
        if self._review_is_running() and self.call_guard:
            state = {**deepcopy(self._review_usage_snapshot),
                     "snapshot_status": "observed_before_background", "offline_call_inflight": True}
            # The fees remain an explicitly old display snapshot. A newly
            # signalled stop is immediate and cannot be masked by that snapshot.
            if self.call_guard._stop_requested.is_set():
                state.update(stopped=True, stop_reason=state.get("stop_reason") or "stop_requested")
            return state
        return self.call_guard.snapshot() if self.call_guard else {"configured": False}

    def _model_scope(self, kind, operation_id, session_id):
        return (self.call_guard.scope(kind, operation_id, session_id)
                if self.call_guard else nullcontext())

    def _models_stopped(self):
        return self.call_guard is not None and self._model_usage()["stopped"]

    def resume_models(self, request_id, pause_id, *, confirmed=False, read_only=False, actor=_USER):
        """The explicit local owner surface; replay never changes the ledger."""
        with self._user_lock(_USER):
            if self._review_is_running():
                return self._result(False, "会后整理仍在运行，当前不能改变模型调用状态。")
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
                                     delivery_block=self.delivery_block,
                                     **({"safety_delivery_version": self.safety_delivery_version}
                                        if self.safety_delivery_version != "legacy" else {}),
                                     **({"safety_response_instruction_version": self.safety_response_instruction_version}
                                        if self.safety_response_instruction_version != "legacy" else {}),
                                     **({"premodel_safety_version": self.premodel_safety_version}
                                        if self.premodel_safety_version != "legacy" else {}))

    def _seed_persona(self, user_id):
        # No biography, psychological interpretation, synthetic observations,
        # future dialogue, or desired ending is seeded into this profile.
        return PersonaProfile(user_id=user_id, profile_origin="fictional_roleplay")

    def _profile(self):
        pair = self._sessions.get(_USER)
        persona = pair[0].persona if pair else self.store.load(_USER)
        if persona is not None and getattr(self.clients.online, "online_decision_version", None) in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
            from ..engine.memory.preference_status import adopt_ui_result
            adopt_ui_result(persona, getattr(self, "_preference_last_ui_result", None))
        return persona

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
                **({"preserve_input_text": True}
                   if self.safety_delivery_version in ("source_response_v4", "minimal_response_v1") else {}),
                **({"premodel_safety_version": self.premodel_safety_version}
                   if self.premodel_safety_version != "legacy" else {}),
                "model_usage": self._model_usage(),
                "offline_review_running": self._review_is_running(),
                "auto_review": auto_review_view(self._run["auto_review"],
                    running=self._review_is_running(), dispatching=self._auto_review_dispatching,
                    quota_used=self._auto_review_quota.used(auto_review_day())),
                "session_id": active["session_id"] if active else None,
                "session_number": active["number"] if active else len(sessions),
                "sessions": sessions, "turns": deepcopy(self._run["turns"]),
                "waiting_input": deepcopy(self._run.get("waiting_input")),
                "reviews": deepcopy(self._run["reviews"]),
                "preference_reviews": deepcopy(self._run["preference_reviews"]),
                "preference_invitation": (preference_invitation_view(persona, self._sessions[_USER][1])
                    if _USER in self._sessions else None),
                "profile": self._understanding(persona),
                "cognitive": cognitive_owner_view(persona),
                "offline": OfflineSessionQueue(persona).status() if persona else {"pending_count": 0},
                "storage_failed": self._storage_failed,
                "requires_review": self._unresolved(),
                "settled_validation_failure_ids": sorted(self._safe_settled_validation_ids()),
                "settled_reflection_failure_ids": sorted(self._safe_settled_reflection_ids()),
                "held_new_input": self._held_view(),
                "held_new_input_failure_ids": self._safe_held_turn_ids(),
                "notice": ("本地流程示意：回复为占位；未执行模型、事实抽取或人物反思。"
                           if self.mode == "local_preview" else
                           "真实生成：在线、手动会后整理以及明确开启的自动会后整理会调用云模型；输入为虚构角色扮演。"),
                "sharing": self.sharing_state(),
                "care_notice": "护理相关提问会附本人近期状态供本轮参考；动作仍按结构表单执行。通知仅写本地收件箱，未联系真实家属。关闭聊天后护理调度继续，停止服务后暂停。",
                "services": {name: False for name in (
                    "tts", "asr", "embedding", "acoustic", "visual", "omni", "external_notifications")},
            }

    def _sharing_context(self):
        pair = self._sessions.get(_USER)
        session_id = pair[1].session_id if pair is not None else None
        try:
            contacts = tuple(deepcopy(row) for row in self.care.elder_view(_USER)["contacts"])
            contacts_available = True
        except Exception:
            # A care read failure stops this independent action only. Never
            # reuse old recipients or expose private storage error details.
            contacts, contacts_available = (), False
        return SharingContext(_USER, self._run["run_id"], session_id,
            bool(pair and contacts_available and not self._closing
                 and not self._storage_failed and not self._unresolved()),
            tuple(deepcopy(row) for row in self._run["turns"]
                  if row.get("status") == "completed" and row.get("session_id") == session_id),
            contacts, contacts_available)

    def sharing_state(self):
        with self._user_lock(_USER):
            if self._sharing_closed_view is not None:
                return deepcopy(self._sharing_closed_view)
            try:
                context = self._sharing_context()
                value = self.sharing.view(context)
                value["sources"] = [{key: row[key] for key in ("turn_id", "session_id", "turn_index", "text")}
                                    for row in context.sources]
                value["contacts"] = [row for row in context.contacts if row.get("channel") == "local"]
                value["writable"] = context.writable
                value["contacts_available"] = context.contacts_available
                value["available"] = True
                if not context.contacts_available:
                    value["notice"] += " 联系人当前无法读取；转达暂停，不复用旧联系人。"
                return value
            except Exception:
                # Unknown is not absence. Keep the independent permission
                # stopped, without reusing cached records or blocking chat.
                self.sharing.failed = True
                return {"channel": "local_sharing_inbox", "external_enabled": False,
                        "records": [], "commands": [], "sources": [], "contacts": [], "holds": [],
                        "writable": False, "available": False, "contacts_available": False,
                        "requires_review": True, "storage_failed": True,
                        "notice": "转达记录读取失败，结果未知；停止转达，不复用旧记录或重发。普通聊天仍按原有保存与费用边界处理。"}

    def sharing_action(self, action, payload, session_id, *, read_only=False, actor=_USER):
        with self._user_lock(_USER):
            if self._closing:
                raise ValueError("服务已停止；转达记录只读，不再批准或发送。")
            context = self._sharing_context()
            if session_id != context.session_id:
                raise ValueError("转达请求不属于当前本人会话。")
            result = self.sharing.action(context, action, payload, actor=actor,
                read_only=read_only, current_context=self._sharing_context)
            return {"ok": True, **result, "state": self.state()}

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
        # The service binds the authenticated caregiver and enforces action
        # permissions; role or actor fields inside payload grant no authority.
        return {"ok": True, **self.care.mutate(token, elder_id, action, payload)}

    def tick_care_all(self):
        # Independent of an active chat, post-session models, or personality.
        self.care.tick()
        return [_USER]

    def _result(self, ok=True, error=None, **extra):
        return {"ok": ok, **({"error": error} if error else {}), **extra, "state": self.state()}

    def _unresolved(self):
        waiting = self._run.get("waiting_input")
        if (waiting and waiting.get("status") in {"waiting", "admitted"}
                and waiting is not self._waiting_submission
                and not any(row.get("turn_id") == waiting.get("turn_id") for row in self._run["turns"])):
            return True
        try:
            settled = self._settled_validation_ids()
            settled |= self._held_accepted_turn_ids()
            reflections = self._resolved_reflection_ids()
        except Exception:
            return True
        auto_uncertain = auto_review_view(self._run["auto_review"],
            running=self._review_is_running(), dispatching=self._auto_review_dispatching)["requires_review"]
        return self._storage_failed or auto_uncertain or any(row["status"] == "pending" or row.get("requires_review")
            or (row["status"] == "running" and not self._review_is_running())
            for row in [*self._run["turns"], *self._run["reviews"], *self._run["preference_reviews"]]
            if row.get("turn_id") not in settled and row.get("request_id") not in reflections)

    def auto_review_action(self, enabled, daily_limit=1, *, request_id=None,
                           confirmed=False, read_only=False, actor=_USER):
        """Persist an explicit owner's setting; it does not review old sessions."""
        with self._user_lock(_USER):
            if actor != _USER:
                raise PermissionError("只有本体验的本人可以设置自动会后整理。")
            if type(read_only) is not bool or read_only:
                return self._result(False, "回放只读，不能设置自动会后整理。")
            if (type(enabled) is not bool or type(daily_limit) is not int
                    or not 1 <= daily_limit <= 8 or (enabled and confirmed is not True)):
                return self._result(False, "请明确选择开关和每日1至8个任务上限；开启须本人确认云模型费用。")
            request_id = request_id if request_id is not None else "auto-setting_" + uuid.uuid4().hex
            if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
                return self._result(False, "自动整理设置请求标识无效。")
            settings = self._run["auto_review"]
            digest = auto_review_command_digest(enabled, daily_limit)
            old = next((row for row in settings["commands"] if row["request_id"] == request_id), None)
            if old:
                return self._result(old["digest"] == digest,
                    None if old["digest"] == digest else "请求标识已用于不同设置，未执行。", duplicate=True)
            if self._storage_failed or (enabled and self._unresolved()):
                return self._result(False, "存在保存失败或结果不明；未开启自动整理，请先核查记录。")
            candidate = deepcopy(settings)
            stamp = _stamp()
            candidate.update(enabled=enabled, daily_limit=daily_limit,
                             enabled_at=stamp if enabled else None)
            candidate["commands"].append({"request_id": request_id, "digest": digest, "at": stamp})
            if not enabled:
                for row in candidate["events"]:
                    if row["status"] in {"queued", "deferred"}:
                        row.update(status="blocked", reason="owner_disabled")
            self._run["auto_review"] = candidate
            try:
                self._save_run()
            except Exception:
                self._run["auto_review"] = settings
                return self._result(False, "自动整理设置保存失败，未显示为生效；请核查本地存储。")
            return self._result()

    def _queue_auto_review(self, session_id):
        settings = self._run["auto_review"]
        if not settings["enabled"] or any(row["session_id"] == session_id for row in settings["events"]):
            return
        persona = self._profile()
        job = next((job for job in persona.offline_sessions if job.session_id == session_id), None) if persona else None
        # Empty/omitted sessions create no billable automatic job. A seal which
        # has no eligible source is still recorded as blocked, without raw text.
        digest = auto_review_source_digest(job, _USER)
        settings["events"].append({"request_id": auto_review_request_id(self._run["run_id"], session_id),
            "session_id": session_id, "source_sha256": digest,
            "authorized_at": settings["enabled_at"], "created_at": _stamp(),
            "status": "blocked" if self._closing or job is None else "queued",
            "reason": "application_closing" if self._closing else "no_eligible_source" if job is None else "",
            "started_on": None})

    def tick_auto_review(self):
        """Start at most one first attempt. Reads, restart, and replay never tick."""
        with self._user_lock(_USER):
            settings = self._run["auto_review"]
            waiting = [row for row in settings["events"] if row["status"] in {"queued", "deferred"}]
            if not waiting:
                return {"ok": True, "started": False}

            def finish(row, status, reason):
                if row["status"] != status or row["reason"] != reason:
                    row.update(status=status, reason=reason)
                    self._save_run()
                return {"ok": True, "started": False, "status": status, "reason": reason}

            if self._closing or not settings["enabled"]:
                reason = "application_closing" if self._closing else "owner_disabled"
                for row in waiting:
                    row.update(status="blocked", reason=reason)
                self._save_run()
                return {"ok": True, "started": False, "status": "blocked", "reason": reason}
            # A worker has a snapshot of its current budget. Do not acquire a
            # transport/ledger lock while holding the owner lock during a call.
            if self._review_is_running():
                return finish(waiting[0], "deferred", "review_inflight")
            if self._storage_failed or self._unresolved() or self._models_stopped():
                reason = ("storage_failed" if self._storage_failed else
                          "result_requires_review" if self._unresolved() else "models_stopped")
                for row in waiting:
                    row.update(status="blocked", reason=reason)
                self._save_run()
                return {"ok": True, "started": False, "status": "blocked", "reason": reason}
            if not self.clients.model_calls_enabled:
                return finish(waiting[0], "blocked", "models_disabled")
            if self.state()["session_id"]:
                return finish(waiting[0], "deferred", "active_session")
            persona = self._profile()
            status = OfflineSessionQueue(persona).status() if persona else {"jobs": []}
            jobs = status["jobs"]
            head = jobs[0] if jobs else None
            row = next((value for value in waiting if head and value["session_id"] == head["session_id"]), None)
            if row is None:
                if not head:
                    for value in waiting:
                        value.update(status="blocked", reason="source_no_longer_queued")
                    self._save_run()
                    return {"ok": True, "started": False, "status": "blocked", "reason": "source_no_longer_queued"}
                return finish(waiting[0], "deferred", "earlier_manual_job")
            if head["attempts"] != 0 or head["state"] != "pending" or head["active_stage"] is not None:
                return finish(row, "blocked", "previous_attempt")
            if head["expired"] or head["source_completeness"] != "complete" or not head["turn_count"]:
                return finish(row, "blocked", "source_incomplete_or_expired")
            if head["memory_completed"] or head["reflection_completed"]:
                return finish(row, "blocked", "prior_stage_result")
            job = persona.offline_sessions[0]
            if auto_review_source_digest(job, _USER) != row["source_sha256"]:
                return finish(row, "blocked", "sealed_source_changed")
            day = auto_review_day()
            try:
                reservation = self._auto_review_quota.reserve(row["request_id"], day, settings["daily_limit"])
            except Exception:
                self._storage_failed = True
                self._halt_models()
                raise RuntimeError("auto_review_quota_store_failed") from None
            if reservation == "daily_limit":
                return finish(row, "deferred", "daily_job_limit")
            if reservation == "duplicate":
                return finish(row, "blocked", "previous_dispatch")
            # Consume a job slot durably before dispatch; a crash at any later
            # point is a review-required attempt, never an automatic replay.
            row.update(status="dispatching", reason="", started_on=day)
            self._save_run()
            self._auto_review_dispatching = row["request_id"]
            try:
                result = self.reflect(row["request_id"])
            finally:
                self._auto_review_dispatching = None
            if result.get("accepted"):
                row.update(status="running", reason="")
            else:
                review = next((value for value in self._run["reviews"]
                               if value["request_id"] == row["request_id"]), None)
                if review is not None and review["status"] == "pending":
                    row.update(status="pending", reason="result_requires_review")
                else:
                    row.update(status="blocked", reason="start_rejected")
            self._save_run()
            return {"ok": True, "started": bool(result.get("accepted")),
                    "request_id": row["request_id"], "status": row["status"]}

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
            pair = self._sessions.get(_USER)
            error = None
            try:
                transition_preference(candidate, preference_id, action, allow_rule_preview=self.mode == "local_preview")
            except (ValueError, KeyError):
                error = "这条说明已变化或操作不适用，请刷新后查看。"
            else:
                try:
                    self.store.save(candidate)
                except Exception:
                    self._storage_failed = True
                    self._halt_models()
                    error = "相处说明保存失败，未显示为生效；请先核查本地存储。"
            if getattr(self.clients.online, "online_decision_version", None) in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
                from ..engine.memory.preference_status import record_preference_ui_result
                observed = candidate if error is None else persona
                record_preference_ui_result(observed, persona, preference_id, action, persisted=error is None,
                                            rejected=error is not None and not self._storage_failed,
                                            ctx=pair[1] if pair else None)
                self._preference_last_ui_result = deepcopy(observed._preference_ui_result)
            if error is not None:
                return self._result(False, error)
            if pair is not None:
                pair[0]._accept_offline_profile(candidate)
            return self._result()

    def _owner_dialogue_context(self, session_id, *, read_only, actor):
        if actor != _USER:
            raise PermissionError("只有本体验的本人可以操作当前对话。")
        if type(read_only) is not bool or read_only:
            raise ValueError("回放只读，不能授权下一轮对话。")
        if (self._closing or self._unresolved() or self._models_stopped()
                or not self.clients.model_calls_enabled):
            raise ValueError("当前未启用模型，或调用/保存需核查；未开启下一轮语义操作。")
        pair = self._sessions.get(_USER)
        if (pair is None or not isinstance(session_id, str) or session_id != pair[1].session_id
                or pair[1].user_id != _USER or pair[0].persona is None
                or pair[0].persona.user_id != _USER):
            raise ValueError("当前本人会话已变化；未授权其他会话。")
        return pair

    def preference_invitation(self, preference_id, session_id, *, read_only=False, actor=_USER):
        with self._user_lock(_USER):
            try:
                engine, ctx = self._owner_dialogue_context(session_id, read_only=read_only, actor=actor)
                value = prepare_preference_invitation(engine.persona, ctx, preference_id, actor=_USER)
            except ValueError as exc:
                return self._result(False, str(exc))
            return self._result(preference_invitation=value)

    def preference_invitation_ack(self, invitation_id, question_sha256, session_id,
                                  *, read_only=False, actor=_USER):
        with self._user_lock(_USER):
            try:
                engine, ctx = self._owner_dialogue_context(session_id, read_only=read_only, actor=actor)
                value = acknowledge_preference_invitation(engine.persona, ctx, invitation_id,
                    question_sha256, actor=_USER)
            except ValueError as exc:
                return self._result(False, str(exc))
            return self._result(preference_invitation=value)

    def cognitive_readiness(self, session_id, *, read_only=False, actor=_USER):
        with self._user_lock(_USER):
            try:
                engine, ctx = self._owner_dialogue_context(session_id, read_only=read_only, actor=actor)
                value = stage_cognitive_readiness(engine.persona, ctx, actor=_USER)
            except ValueError as exc:
                return self._result(False, str(exc))
            return self._result(**value)

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

    def load_fictional_context(self, execution_input, session_id, *, fictional=False,
                               read_only=False, actor=_USER):
        """Explicit in-process evaluation input; deliberately has no HTTP route."""
        with self._user_lock(_USER):
            if fictional is not True or read_only is not False or actor != _USER:
                return self._result(False, "虚构评测上下文仅限明确授权的本地本人入口。")
            persona = self._profile()
            if (self._closing or self._unresolved() or self._storage_failed
                    or self._run["auto_review"]["enabled"] or len(self._run["sessions"]) != 1
                    or self._run["turns"] or self._run["reviews"]
                    or persona is None or persona.profile_origin != "fictional_roleplay"
                    or persona.episodic_memory or persona.person_hypotheses or persona.stated_preferences):
                return self._result(False, "虚构评测上下文必须使用全新、未预填的虚构体验。")
            try:
                bound = self._bind_fictional_context(_USER, session_id, execution_input)
            except ValueError as exc:
                return self._result(False, str(exc))
            self._run["sessions"][0]["fictional_input"] = bound
            try:
                self._save_run()
            except Exception:
                self._storage_failed = True
                self._halt_models()
                return self._result(False, "虚构评测上下文保存失败，未授权提交。")
            return self._result(fictional_input=deepcopy(bound))

    def submit(self, text, turn_id, session_id, *, external_emotion=None,
               on_safety_prelude=None, on_waiting_input=None):
        with self.call_guard.online_interest() if self.call_guard else nullcontext():
            return self._submit_scheduled(text, turn_id, session_id,
                external_emotion=external_emotion, on_safety_prelude=on_safety_prelude,
                on_waiting_input=on_waiting_input)

    def _submit_scheduled(self, text, turn_id, session_id, *, external_emotion=None,
                          on_safety_prelude=None, on_waiting_input=None):
        guard = self.call_guard
        options = dict(external_emotion=external_emotion, on_safety_prelude=on_safety_prelude)
        if guard is None:
            return self._submit_with_held_input(text, turn_id, session_id, **options)
        waiting = None
        with self._user_lock(_USER):
            old_wait = self._run.get("waiting_input")
            if old_wait and old_wait.get("status") == "waiting":
                same = (old_wait.get("text"), old_wait.get("turn_id"), old_wait.get("session_id"),
                        old_wait.get("external_emotion")) == (text, turn_id, session_id, external_emotion)
                active = old_wait is self._waiting_submission
                return self._result(False, "这句话已保存并正在等待当前请求结束。" if same and active else
                    "已有一句话的等待状态需要核查；不会自动重新发送。", duplicate=same,
                    accepted=same and active, request_sent=False, waiting_for_model=active)
            if self._review_is_running():
                try:
                    parse_external_emotion(external_emotion)
                except ValueError:
                    return self._result(False, "外部情绪读数格式无效，未启动模型。", request_sent=False)
                view = self.state()
                if (not isinstance(turn_id, str) or not _ID.fullmatch(turn_id)
                        or not isinstance(text, str) or not text.strip() or len(text) > 500
                        or self._closing or self._unresolved() or self._models_stopped()
                        or not session_id or session_id != view["session_id"]
                        or any(row["turn_id"] == turn_id for row in self._run["turns"])):
                    return self._result(False, "当前输入、会话或调用状态不能接受新请求；未发送模型。", request_sent=False)
                waiting = {"turn_id": turn_id, "session_id": session_id, "text": text,
                    "session_number": view["session_number"],
                    "turn_index": len(self._sessions[_USER][1].turns),
                    "external_emotion": deepcopy(external_emotion), "status": "waiting",
                    "created_at": _stamp(), "run_id": self._run["run_id"], "request_sent": False}
                # The existing deterministic backstop must be emitted before
                # waiting for a background HTTP request, never after it.
                waiting_transport = on_waiting_input if callable(on_waiting_input) else on_safety_prelude
                if callable(waiting_transport) and self.premodel_safety_version != "legacy":
                    engine, ctx = self._session(_USER)
                    event = engine.pending_safety_prelude(ctx, self._text_utterance(text))
                    if event is not None:
                        waiting["protection_prelude"] = {**event, "status": "saved",
                            "saved_at": _stamp(), "model_generated": False}
                self._run["waiting_input"] = waiting
                self._waiting_submission = waiting
                self._save_run()
                expected = deepcopy(waiting)
                if callable(waiting_transport):
                    try:
                        if waiting_transport({"type": "input_waiting", "turn_id": turn_id,
                                "session_id": session_id, "state": self.state()}) is not True:
                            raise ValueError("waiting_input_transport_not_confirmed")
                        if self._run.get("waiting_input") is not waiting or waiting != expected:
                            raise ValueError("waiting_input_identity_changed")
                        if "protection_prelude" in waiting:
                            waiting["protection_prelude"].update(status="transport_returned",
                                transport_returned_at=_stamp())
                            self._save_run()
                            expected = deepcopy(waiting)
                    except Exception:
                        self._cancel_waiting_input(waiting, "waiting_delivery_failed")
                        self._waiting_submission = None
                        return self._result(False, "等待状态未送达，这句话未发送模型。", request_sent=False)
        # No owner lock is held while the already-started physical request
        # finishes. The same accepted turn takes priority over reflection.
        try:
            with guard.dispatch_slot(online=True):
                settle=getattr(guard,'wait_for_offline_failure_settlement',None)
                if callable(settle):settle()
                with self._user_lock(_USER):
                    if waiting is not None:
                        if (self._run.get("run_id") != expected["run_id"]
                                or self._run.get("waiting_input") is not waiting):
                            return self._result(False, "等待期间体验资料已改变，这句话未发送模型。", request_sent=False)
                        valid = (waiting == expected and not self._closing and not self._unresolved()
                            and not self._storage_failed and not guard.snapshot()["stopped"]
                            and self.state()["session_id"] == session_id)
                        if not valid:
                            self._cancel_waiting_input(waiting, "waiting_state_changed")
                            return self._result(False, "等待期间会话、权限或调用状态已改变，这句话未发送模型。", request_sent=False)
                        # Persist the admission before entering the original
                        # held/validation transaction. Its own pending turn,
                        # provenance and fee checks remain unchanged.
                        waiting["status"] = "admitted"
                        waiting.pop("request_sent", None)  # The ordinary turn now owns send status.
                        self._save_run()
                    result = self._submit_with_held_input(text, turn_id, session_id, **options)
                    if waiting is not None and not any(row["turn_id"] == turn_id for row in self._run["turns"]):
                        self._cancel_waiting_input(waiting, "waiting_input_rejected")
                        result.update(request_sent=False, state=self.state())
                    return result
        finally:
            if waiting is not None:
                with self._user_lock(_USER):
                    if self._waiting_submission is waiting:
                        self._waiting_submission = None

    def _cancel_waiting_input(self, waiting, code):
        waiting.update(status="cancelled", error_code=code, request_sent=False)
        if self._run.get("waiting_input") is waiting:
            if not any(row["turn_id"] == waiting["turn_id"] for row in self._run["turns"]):
                self._run["turns"].append({**deepcopy(waiting), "status": "failed", "response": "",
                    "source": "local_request_rejected", "requires_review": False,
                    "error": "等待期间状态已改变或等待消息未送达，这句话未发送模型。"})
            self._save_run()

    def _submit_original(self, text, turn_id, session_id, *, external_emotion=None, on_safety_prelude=None):
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
                return self._result(old["status"] == "completed", old.get("error"), duplicate=True,
                                    **{key: old[key] for key in ("error_code", "request_sent") if key in old})
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
            session = next(row for row in self._run["sessions"] if row["session_id"] == session_id)
            stimulus = session.get("fictional_input")
            if stimulus is not None and (ctx.turns
                    or ctx.working_memory.get("_fictional_web_input") != stimulus
                    or text != stimulus["messages"][-1]["content"]):
                return self._result(False, "虚构评测上下文仅允许原样提交绑定的当前一句一次。",
                                    error_code="fictional_context_mismatch", request_sent=False)
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
            prelude_failed = False
            run_identity = self._run["run_id"]
            attempts_before = (self.call_guard.snapshot().get("calls_used")
                               if self.call_guard else None)

            def deliver_prelude(event):
                # This is a trusted in-process capability, never a request-body
                # permission. Persist before transport; persist its return before
                # allowing the engine to start the one model request.
                nonlocal prelude_failed
                saved_prelude = None
                try:
                    expected = {"version", "premodel_safety_version", "user_id", "session_id",
                                "scene_id", "turn_index", "utterance_sha256", "protection",
                                "protection_sha256"}
                    utterance = self._text_utterance(text)
                    if (self.premodel_safety_version not in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4")
                            or type(event) is not dict or set(event) != expected
                            or event.get("version") != "safety_prelude_v1"
                            or event.get("premodel_safety_version") != self.premodel_safety_version
                            or (event.get("user_id"), event.get("session_id"), event.get("scene_id"))
                               != (_USER, session_id, ctx.scene_id)
                            or type(event.get("turn_index")) is not int
                            or event["turn_index"] != row["turn_index"]
                            or event.get("utterance_sha256") != hashlib.sha256(utterance.encode()).hexdigest()
                            or type(event.get("protection")) is not str
                            or not 0 < len(event["protection"]) <= 4096
                            or event.get("protection_sha256") != hashlib.sha256(event["protection"].encode()).hexdigest()
                            or "protection_prelude" in row):
                        raise ValueError("invalid_current_safety_prelude")
                    waiting = self._run.get("waiting_input") or {}
                    early = waiting.get("protection_prelude")
                    if (early is not None and waiting.get("turn_id") == turn_id
                            and waiting.get("session_id") == session_id):
                        if (waiting.get("status") != "admitted" or early.get("status") != "transport_returned"
                                or {key: early.get(key) for key in expected} != event):
                            raise ValueError("waiting_safety_prelude_changed")
                        row["protection_prelude"] = deepcopy(early)
                        self._save_run()
                        return True  # Already transported before the wait.
                    row["protection_prelude"] = {**deepcopy(event), "status": "saved",
                                                "saved_at": _stamp(), "model_generated": False}
                    saved_prelude = deepcopy(row["protection_prelude"])
                    self._save_run()
                    saved_bytes = self.run_path.read_bytes()
                    saved_row = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    frame = {"type": "safety_prelude", "turn_id": turn_id,
                             "session_id": session_id, "state": self.state()}
                    if on_safety_prelude(deepcopy(frame)) is not True:
                        raise ValueError("safety_prelude_transport_not_confirmed")
                    if (self._run.get("run_id") != run_identity
                            or not any(item is row for item in self._run["turns"])
                            or json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) != saved_row
                            or self.run_path.read_bytes() != saved_bytes
                            or (row.get("turn_id"), row.get("session_id"), row.get("text"), row.get("status"))
                               != (turn_id, session_id, text, "pending")
                            or self._sessions.get(_USER, (None, None))[1] is not ctx
                            or self._storage_failed or row.get("requires_review")
                            or self.premodel_safety_version not in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4")
                            or self.premodel_safety_version != saved_prelude["premodel_safety_version"]
                            or self._models_stopped()):
                        raise ValueError("safety_prelude_identity_or_state_changed")
                    row["protection_prelude"].update(status="transport_returned",
                                                     transport_returned_at=_stamp())
                    self._save_run()
                    return True
                except Exception:
                    prelude_failed = True
                    row["requires_review"] = True
                    if saved_prelude is not None:
                        row["protection_prelude"] = saved_prelude
                    if "protection_prelude" in row:
                        row["protection_prelude"]["status"] = "transport_or_save_unknown"
                    self._halt_models()
                    raise

            try:
                ctx.working_memory.pop("prompt_evidence_snapshot", None)
                # Only the explicitly model-disabled preview can expose local
                # rule proposals. Reopening the same profile live resets this.
                ctx.working_memory["_local_preference_preview"] = self.mode == "local_preview"
                if "care_context" in row:
                    ctx.working_memory["minimal_care_context"] = deepcopy(row["care_context"])
                local_budget_error = None
                with self._model_scope("turn", turn_id, session_id):
                    prelude_options = ({"safety_prelude": deliver_prelude}
                        if self.premodel_safety_version in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4") and callable(on_safety_prelude) else {})
                    try:
                        result = super().text_turn(_USER, text, turn_id=turn_id,
                                                  external_emotion=external_emotion, **prelude_options)
                    except CompactPromptBudgetExceeded as error:
                        # The scope deliberately halts on any escaping error.
                        # Defer this one typed construction failure until its
                        # zero-attempt classification below; never clear a halt.
                        local_budget_error = error
                    if (local_budget_error is None and isinstance(self.call_guard, ProductCallGuard) and
                            (ctx.working_memory.get("model_route") or {}).get("reply", {}).get("source") == "deterministic_fallback"):
                        self.call_guard.note_output_failure()
                if local_budget_error is not None:
                    raise local_budget_error
                if not result["ok"]:
                    raise RuntimeError("turn_failed")
                if prelude_failed:
                    raise RuntimeError("safety_prelude_not_confirmed")
                route = (ctx.working_memory.get("model_route") or {}).get("reply", {})
                failed = route.get("source") == "deterministic_fallback"
                from ..engine.policy.delivery_provenance import delivery_protection
                protection = delivery_protection(ctx)
                product_fallback = failed and isinstance(self.call_guard, ProductCallGuard)
                hard_stop = product_fallback and self.call_guard.snapshot()["stop_reason"] not in {None, "consecutive_failures"}
                local_rejection = (self.call_guard.snapshot().get("last_request_rejection", {})
                                   if product_fallback else {})
                local_rejected = local_rejection.get("operation_id") == turn_id
                if failed and not product_fallback and self._held_run_guard_ready:
                    # The guard validates accepted holds against the persisted
                    # pending turn. Stop before changing that turn in memory.
                    self._halt_models()
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
                        if not self._held_run_guard_ready:
                            self._halt_models()
                        row["requires_review"] = True
                        row["error"] = "模型调用或返回失败，本轮没有可用的模型回复；已停止后续调用，需先核查结果与费用。"
            except CompactPromptBudgetExceeded as error:
                # A typed construction error is recoverable only when the
                # durable product ledger proves this turn sent no request.
                current = self.call_guard.snapshot() if self.call_guard else {}
                local_only = (type(error) is CompactPromptBudgetExceeded
                    and ctx.working_memory.get("_online_decision_version") == "online_turn_decision_v7"
                    and type(attempts_before) is int and current.get("calls_used") == attempts_before
                    and not current.get("stopped") and current.get("unknown_cost_requests") == 0
                    and not self._storage_failed and not prelude_failed)
                if local_only:
                    row.update(status="failed", response="", source="local_request_rejected",
                        requires_review=False, request_sent=False,
                        error_code="compact_online_system_byte_limit_exceeded",
                        error="本轮必要上下文超过发送上限，未向模型发送请求；可继续下一轮，本轮不会自动重发。",
                        elapsed_ms=round((time.perf_counter() - started) * 1000))
                else:
                    self._halt_models()
                    row["error"] = "生成或保存未完成，结果可能不明；已停止自动重试。"
            except Exception:
                self._halt_models()
                # The model might already have run. Keep the durable attempt pending;
                # a read or restart must not retry it under a new ID.
                row["error"] = "生成或保存未完成，结果可能不明；已停止自动重试。"
            finally:
                premodel = ctx.working_memory.get("premodel_task")
                if (self.premodel_safety_version != "legacy" and type(premodel) is dict
                        and premodel.get("turn_index") == row["turn_index"]
                        and premodel.get("session_id") == session_id):
                    row["premodel_task"] = deepcopy(premodel)
                # A render audit is not proof of successful visible delivery.
                # Keep only this turn's hash/enum metadata, never a stale turn.
                safety_delivery = ctx.working_memory.get("safety_delivery")
                if (self.safety_delivery_version != "legacy" and type(safety_delivery) is dict
                        and safety_delivery.get("turn_index") == row["turn_index"]
                        and safety_delivery.get("session_id") == session_id
                        and safety_delivery.get("user_id") == _USER):
                    row["safety_delivery_render"] = deepcopy(safety_delivery)
                safety_review = ctx.working_memory.get("safety_review")
                if (type(safety_review) is dict and safety_review.get("turn_index") == row["turn_index"]
                        and ctx.session_id == session_id and ctx.user_id == _USER):
                    row["safety_review"] = deepcopy(safety_review)
                instruction = ctx.working_memory.get("safety_response_instruction")
                if (self.safety_response_instruction_version != "legacy" and type(instruction) is dict
                        and instruction.get("turn_index") == row["turn_index"]
                        and instruction.get("session_id") == session_id
                        and instruction.get("user_id") == _USER):
                    row["safety_response_instruction"] = deepcopy(instruction)
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
            return self._result(row["status"] == "completed", row.get("error"),
                                **{key: row[key] for key in ("error_code", "request_sent") if key in row})

    def end(self, user_id=_USER, *, process_offline=None):
        # Seal first. Only the separately confirmed automatic option may start
        # a first background review; process shutdown suppresses that option.
        with self._user_lock(_USER):
            if self._profile_write_conflict:
                return {"ok": False,
                    "error": "本地资料与当前会话来源不一致，原记录保持；核查前不会用旧会话覆盖保存。",
                    "active_session_retained": _USER in self._sessions}
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
            elif not self._closing:
                self.tick_auto_review()
            return result

    def _seal_active_session(self, user_id, pair):
        session_id = pair[1].session_id
        super()._seal_active_session(user_id, pair)
        for row in self._run["sessions"]:
            if row["session_id"] == session_id:
                row.update(status="sealed", sealed_at=_stamp())
        self._queue_auto_review(session_id)
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

    def reflect(self, request_id, *, wait=False):
        """Explicitly start one background review; optional wait is for batch callers."""
        with self._user_lock(_USER):
            if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
                return self._result(False, "整理请求标识无效。")
            old = next((row for row in self._run["reviews"] if row["request_id"] == request_id), None)
            if old:
                running = old["status"] == "running" and self._review_is_running()
                return self._result(running or old["status"] == "completed", old.get("error"),
                                    duplicate=True, accepted=running)
            if self._closing or self._review_is_running():
                return self._result(False, "会后整理正在运行，请查看当前进度，不会重复启动。")
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
            row = {"request_id": request_id, "session_id": job.session_id,
                   "status": "running", "created_at": _stamp()}
            self._run["reviews"].append(row)
            self._save_run()
            self._review_usage_snapshot = self.call_guard.snapshot() if self.call_guard else None
            arguments = (request_id, self._run["run_id"], job.session_id, persona.model_copy(deep=True))
            worker = (threading.current_thread() if wait else threading.Thread(
                target=self._run_review, args=arguments, name="aria-offline-review", daemon=True))
            self._review_thread = worker
            if not wait:
                try:
                    worker.start()
                except Exception:
                    # A raised start does not prove that no work ran. Retain
                    # the durable request and quota as uncertain; never replay.
                    # An unstarted Thread cannot be joined during shutdown.
                    if worker.ident is None:
                        self._review_thread = None
                        self._review_usage_snapshot = None
                    row.update(status="pending", requires_review=True,
                               error="会后任务启动结果不明；请核查原记录，不会自动重发。")
                    self._halt_models()
                    self._save_run()
                    return self._result(False, row["error"], accepted=False, review_pending=True)
                return self._result(accepted=True, review_pending=True)
        # Registered batch callers keep their existing thread-local experiment
        # scope. They still release the owner lock before slow work.
        self._run_review(*arguments)
        with self._user_lock(_USER):
            final = next((value for value in self._run["reviews"] if value["request_id"] == request_id), None)
            if final is None:
                return self._result(False, "原体验已清除，旧整理结果未写回。")
            return self._result(final["status"] == "completed", final.get("error"))

    def _run_review(self, request_id, run_id, session_id, persona):
        return self._run_review_with_settled_reflection(request_id,run_id,session_id,persona)

    def _run_review_original(self, request_id, run_id, session_id, persona):
        stage_store = _OfflineStageStore(self, run_id, session_id, persona)
        values = {}
        halt_requested = False
        try:
            engine = self._engine_for_persona(persona)
            engine.store = stage_store
            with self._model_scope("review", request_id, session_id):
                result = engine.recover_offline_sessions(max_jobs=1)
                success = result.get("last_run", {}).get("completed_jobs", 0) == 1
                if isinstance(self.call_guard, ProductCallGuard):
                    audit = next((item for item in reversed(engine.persona.offline_session_audit)
                                  if item.session_id == session_id), None)
                    success = success and audit is not None and audit.outcome == "completed"
                    if not success and not stage_store.cancelled:
                        self.call_guard.note_output_failure()
                    if not stage_store.cancelled and result.get("last_run", {}).get("error_code") == "offline_store_failed":
                        self._storage_failed = True
                        self._halt_models()
            values.update(status="cancelled" if stage_store.cancelled else "completed" if success else "failed",
                          result=result, finished_at=_stamp())
            if stage_store.cancelled:
                values["error"] = "本人状态或封存来源已变化，旧整理结果未写回，也不会自动重发。"
            elif not success:
                if isinstance(self.call_guard, ProductCallGuard):
                    reason = self.call_guard.snapshot()["stop_reason"]
                    values["requires_review"] = reason not in {None, "consecutive_failures"}
                    values["error"] = "本次会后整理未完成，阶段记录保留且不自动重发；其他对话是否可用以调用状态为准。"
                else:
                    values["requires_review"] = True
                    halt_requested = True
                    values["error"] = "会后整理未完成；请查看阶段状态，不会自动重试。"
        except Exception:
            if stage_store.cancelled:
                values.update(status="cancelled", finished_at=_stamp(),
                              error="本人状态或封存来源已变化，旧整理结果未写回，也不会自动重发。")
            else:
                halt_requested = True
                values.update(status="pending", requires_review=True,
                              error="会后整理或保存结果不明；已停止继续调用。")
        # Scope/transport locks have been released before owner state is read.
        with self._user_lock(_USER):
            self._review_thread = None
            self._review_usage_snapshot = None
            if self._run["run_id"] != run_id:
                return
            row = next(value for value in self._run["reviews"] if value["request_id"] == request_id)
            row.update(values, changes=stage_store.changes or [{"session_id": session_id, "kind": "unchanged",
                "summary": "本次没有提交事实、人物假设或相处说明的修改。"}])
            event = next((value for value in self._run["auto_review"]["events"]
                          if value["request_id"] == request_id), None)
            if event is not None:
                event.update(status=row["status"], reason="" if row["status"] == "completed" else "review_not_completed")
            # Experiment-only terminal handlers inspect the installed failed
            # row before deciding whether its known body failure may settle.
            if halt_requested:
                self._halt_models()
            try:
                self._save_run()
            except Exception:
                row.update(status="pending", requires_review=True,
                           error="会后记录保存未完成，需核查；不会自动重发。")
                if event is not None:
                    event.update(status="pending", reason="result_requires_review")

    def _forget_run_original(self):
        with self._user_lock(_USER):
            held_markers=self._begin_held_forget()
            self.sharing.forget()
            self.care.forget_elder(_USER)
            # Delete durable files first. A failed deletion retains the RAM copy.
            self.store.delete(_USER)
            self._forget_settled_validation_metadata()
            self._forget_settled_reflection_metadata()
            for path in (self.run_path, self.run_path.with_suffix(".json.tmp")):
                path.unlink(missing_ok=True)
            self._sessions.clear()
            self._session_activity.clear()
            self._run = self._load_run()
            if self.call_guard:
                self._run["model_allowance"] = {"ledger_id": self.call_guard.ledger_id,
                                                "policy_sha256": self.call_guard.policy_sha256}
            self._storage_failed = False
            self._profile_write_conflict = False
            self._complete_held_forget(held_markers)
            return None

    def close(self):
        # Preserve failures in RAM until the owner explicitly retries sealing.
        # Set this before end(): shutdown must never start an automatic model.
        with self._user_lock(_USER):
            self._closing = True
        result = None
        if self._sessions:
            result = self.end()
        with self._user_lock(_USER):
            self.tick_auto_review()
            worker = self._review_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._user_lock(_USER):
            self._sharing_closed_view = self.sharing_state()
        self.sharing.close()
        self.care.close()
        if self.call_guard and self._owns_call_guard:
            self.call_guard.close()
        if result is not None and not result["ok"]:
            raise RuntimeError("会话尚未安全封存；请检查本地存储，最近成功的检查点保持不变。")


class _OfflineStageStore:
    """Publish only reconciled stage commits; model handlers use a private copy."""

    def __init__(self, app, run_id, session_id, before):
        self.app, self.run_id, self.session_id = app, run_id, session_id
        self.before = before.model_copy(deep=True)
        self.cancelled = False
        self.changes = []

    def save(self, proposed):
        with self.app._user_lock(_USER):
            try:
                if (self.app._run["run_id"] != self.run_id
                        or self.app._closing or self.app._storage_failed):
                    raise OfflineReviewCancelled("offline_run_removed")
                current = self.app.store.load(_USER)
                pair = self.app._sessions.get(_USER)
                if (current is None or pair is not None
                        and pair[0].persona.model_dump(mode="json") != current.model_dump(mode="json")):
                    # An owner API updates its checkpoint and active cache in
                    # one owner-locked operation. A different durable value is
                    # not this worker's state to overwrite or bless as consent.
                    self.app._profile_write_conflict = True
                    self.app._storage_failed = True
                    self.app._halt_held_models()
                    raise OfflineReviewCancelled("offline_owner_profile_changed")
                merged = merge_offline_stage(self.before, proposed, current, self.session_id)
            except OfflineReviewCancelled:
                self.cancelled = True
                raise
            changes = _changes(self.app._understanding(current), self.app._understanding(merged), self.session_id)
            self.app.store.save(merged)
            self.changes.extend(row for row in changes if row["kind"] != "unchanged")
            pair = self.app._sessions.get(_USER)
            if pair:
                pair[0]._accept_offline_profile(merged)
            # Queue adopts this object after saving, so its next stage sees
            # current owner withdrawals and the reconciled live profile.
            for field in type(proposed).model_fields:
                setattr(proposed, field, deepcopy(getattr(merged, field)))
            from ..engine.memory.preference_status import copy_preference_provenance
            copy_preference_provenance(proposed, merged)
            self.before = merged.model_copy(deep=True)


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

        def _stream_text(self, body):
            # Existing localhost/token/body checks run before this method.
            # HTTP/1.0 close delimits the stream; every frame is complete JSON.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self._security_headers()
            self.send_header("Connection", "close")
            self.end_headers()

            def write_frame(frame):
                self.wfile.write((json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                self.wfile.flush()
                # This records only transport return, never user reading.
                return True

            try:
                result = app.submit(body.get("text"), body.get("turn_id"), body.get("session_id"),
                                    external_emotion=body.get("external_emotion"),
                                    on_safety_prelude=write_frame, on_waiting_input=write_frame)
            except Exception:
                try:
                    result = app._result(False, "本地处理或保存未完成；已发送的保护内容保留，请刷新核查，不要重发。")
                except Exception:
                    result = {"ok": False, "error": "本地保存状态不可读取，请核查原记录；不要重发。"}
            try:
                write_frame({"type": "result", "turn_id": body.get("turn_id"),
                             "session_id": body.get("session_id"), "result": result})
            except (OSError, ValueError):
                pass  # A disconnected client must not restart the saved turn.

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
            elif route == "/api/minimal/sharing":
                try:
                    self._json({"ok": True, **app.sharing_state()})
                except Exception:
                    self._json({"ok": False, "error": "转达记录读取失败，原文件未覆盖。"}, 500)
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
                "/api/minimal/auto-review": lambda body: app.auto_review_action(
                    body.get("enabled"), body.get("daily_limit", 1), request_id=body.get("request_id"),
                    confirmed=body.get("confirmed"), read_only=body.get("replay", False),
                    actor=body.get("actor", _USER)),
                "/api/minimal/resume-models": lambda body: app.resume_models(
                    body.get("request_id"), body.get("pause_id"), confirmed=body.get("confirmed"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/continue-new-input": lambda body: app.continue_new_input(
                    body.get("request_id"),body.get("failure_id"),body.get("session_id"),
                    confirmed=body.get("confirmed"),read_only=body.get("replay",False),actor=body.get("actor",_USER)),
                "/api/minimal/preference": lambda body: app.preference_action(
                    body.get("action"), body.get("id"), read_only=body.get("replay", False)),
                "/api/minimal/preference-invitation": lambda body: app.preference_invitation(
                    body.get("id"), body.get("session_id"), read_only=body.get("replay", False),
                    actor=body.get("actor", _USER)),
                "/api/minimal/preference-invitation-ack": lambda body: app.preference_invitation_ack(
                    body.get("id"), body.get("question_sha256"), body.get("session_id"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/cognitive-readiness": lambda body: app.cognitive_readiness(
                    body.get("session_id"), read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/preference-review": lambda body: app.preview_preferences(body.get("request_id")),
                "/api/minimal/cognitive": lambda body: app.cognitive_action(
                    body.get("action"), observation_id=body.get("id"), delete_records=body.get("delete_records"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
                "/api/minimal/forget": lambda body: app.forget_run(),
                "/api/minimal/sharing": lambda body: app.sharing_action(
                    body.get("action"), body.get("payload"), body.get("session_id"),
                    read_only=body.get("replay", False), actor=body.get("actor", _USER)),
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
            if route not in actions and route not in {"/api/care/login", "/api/care/logout", "/api/minimal/text-stream"}:
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
                if route == "/api/minimal/text-stream":
                    if app.premodel_safety_version not in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4"):
                        self._json({"ok": False, "error": "当前运行未启用分阶段回应。"}, 409)
                        return
                    self._stream_text(body)
                    return
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
