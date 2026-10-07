"""引擎主循环 —— 无 FSM。

每轮:感知 → 更新关系/假设 → 主动观察/Skill → 检索知识 → 风险筛查 →
       策略(带知识+工具)→ (可选)任务流 → 写回。
会话结束:finalize_session → 复利沉淀(对话→更新档案→落盘)。
首发由用户发起对话，护理调度独立运行。

只认 Scene 协议,绝不 import 任何具体场景。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
import json
import time
from typing import Any, Optional

from ..interfaces.risk import RiskSignal
from ..interfaces.skills import ActiveTask, Skill
from ..interfaces.knowledge import WorldKnowledgeItem, WorldKnowledgeSource
from ..interfaces.scene import Scene
from ..interfaces.tools import Tool
from ..skills import SkillRegistry
from ..tools.registry import ToolRegistry
from .alliance import update_alliance
from .context import EmotionState, EngineContext
from .cognitive import apply_cognitive_withdrawal, cognitive_sources
from .formulation import update_formulation
from .memory.consolidator import Extractor, MemoryConsolidator
from .memory.diagnosis import PersonaHypothesisUpdater, _REFLECTION_ERROR_DETAILS
from .memory.persona_store import PersonaStore
from .memory.offline_queue import (
    OFFLINE_MAX_CHARACTERS, OFFLINE_MAX_TURNS, OFFLINE_SESSION_LOCK, OfflineSessionQueue,
)
from .memory.recall import recall_episodes
from .model.persona import (
    PersonaProfile,
    PersistentTaskRecord,
    RiskEvent,
    SessionCheckpointRecord,
    SessionTurnSnapshot,
)
from .risk.safety import safety_response
from .risk.semantic import semantic_safety_signal
from .perception import PerceptionEngine, route_state_analysis
from .perception.evidence import collect_evidence
from .perception.longitudinal import enrich_with_personal_baseline
from .perception.signals import is_simulated, prepare_acoustic, prepare_multimodal
from .perception.external_emotion import SyntheticEmotionExperiment, parse_external_emotion
from .policy import PolicyEngine
from .policy.actions import (
    consume_pending_action,
    current_pending_action,
    parse_action,
    prepare_action,
    stage_pending_action,
)
from .policy.llm import LLMClient, create_llm_client
from .policy.online_decision import (
    DETERMINISTIC_MODEL_FAILURE_REPLY, ONLINE_DECISION_VERSION_SAFETY,
)
from .policy.prompt import update_delivery_state
from .routing import route_reply_model


def _checkpoint_turns(ctx: EngineContext) -> list[SessionTurnSnapshot]:
    """短期恢复检查点也遵守场景标记的敏感回合排除。"""
    raw = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
    excluded = {
        item for item in raw
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }
    start = max(0, len(ctx.turns) - 12)
    return [
        SessionTurnSnapshot(user=turn.user, agent=turn.agent)
        for index, turn in enumerate(ctx.turns[start:], start=start)
        if index not in excluded
    ]


class Engine:
    def __init__(
        self,
        scene: Scene,
        persona: Optional[PersonaProfile] = None,
        llm_client: Optional[LLMClient] = None,
        store: Optional[PersonaStore] = None,
        extractor: Optional[Extractor] = None,
        diagnoser: Optional[PersonaHypothesisUpdater] = None,
        hypothesis_updater: Optional[PersonaHypothesisUpdater] = None,
        capabilities: Optional[set[str]] = None,
        world_knowledge: Optional[WorldKnowledgeSource] = None,
        skills: Optional[list[Skill]] = None,
        tools: Optional[list[Tool]] = None,
        state_llm_client: Optional[LLMClient] = None,
        fast_llm_client: Optional[LLMClient] = None,
        simulation_mode: bool = False,
        delivery_block: Optional[str] = None,
        external_emotion_experiment: SyntheticEmotionExperiment | None = None,
    ) -> None:
        if external_emotion_experiment is not None:
            if type(external_emotion_experiment) is not SyntheticEmotionExperiment:
                raise ValueError("external_emotion_experiment_requires_python_capability")
            if simulation_mode or capabilities or tools or skills or world_knowledge:
                raise ValueError("external_emotion_experiment_cannot_enable_external_actions")
            external_emotion_experiment.validate(store, persona)
        elif persona is not None and persona.profile_origin.startswith("synthetic_emotion_experiment:"):
            raise ValueError("synthetic_experiment_profile_requires_isolated_engine")
        self.external_emotion_experiment = external_emotion_experiment
        if simulation_mode:
            if store is not None:
                raise ValueError("emotion simulation must use an isolated memory-only engine")
            if persona is not None and persona.profile_origin != "synthetic_demo":
                raise ValueError("emotion simulation requires a synthetic_demo persona")
            if capabilities or tools or skills or extractor or hypothesis_updater or diagnoser or world_knowledge:
                raise ValueError("emotion simulation cannot enable external tools or reflection")
        self.simulation_mode = simulation_mode
        self.scene = scene
        self.persona = deepcopy(persona) if simulation_mode else persona
        # 严格双轨：process_turn 只装配快速在线模型。llm_client 是旧注入名，
        # 只有未显式传 fast_llm_client 时才作为快速客户端使用；不存在在线
        # Claude 升级或失败回退。深度模型只归 consolidator / updater 所有。
        fast_llm = (
            fast_llm_client
            or llm_client
            or create_llm_client("online_fast")
        )
        self.fast_llm = fast_llm
        # 仅为旧版集成保留公开属性。在线主链路不再创建或调用独立状态模型；
        # 深度人物模型由会后 hypothesis_updater 负责。
        self.state_llm = state_llm_client
        self.policy = PolicyEngine(fast_llm=fast_llm, delivery_block=delivery_block)
        self.perception = PerceptionEngine()
        skills_provider = getattr(scene, "skills", None)
        scene_skills = (
            skills_provider() if callable(skills_provider) and not simulation_mode
            and external_emotion_experiment is None else []
        )
        self._skills = SkillRegistry(
            [*scene_skills, *(skills or [])],
            capabilities=capabilities or set(),
        )
        self._tools = ToolRegistry(
            [] if simulation_mode or external_emotion_experiment is not None
            else [*scene.tools(), *self._skills.tools(), *(tools or [])]
        )
        self._knowledge = scene.knowledge()          # 该场景的领域 KB(可空)
        self._world_knowledge = world_knowledge     # 公共现状；不写入 persona
        self._risk = scene.risk_monitor()
        reasoner_factory = getattr(scene, "state_reasoner", None)
        self._state_reasoner = reasoner_factory() if callable(reasoner_factory) else None
        self.store = store
        # 复利沉淀器:有 store 才落盘。extractor 可注入(生产用 LLMExtractor 深抽取;
        # 不传则用 HeuristicExtractor,无 LLM 也能跑通闭环)。
        self._consolidator = (
            MemoryConsolidator(store, extractor) if store is not None else None
        )
        self.hypothesis_updater = hypothesis_updater or diagnoser
        self.diagnoser = self.hypothesis_updater     # 兼容 0.1 版公开属性

    def _validate_context_mode(self, ctx: EngineContext) -> None:
        external_input = (ctx.working_memory.get("perception_input") or {}).get("external_emotion")
        if (not self.simulation_mode and self.external_emotion_experiment is None
                and (is_simulated(ctx.emotion.external_reading) or is_simulated(external_input))):
            raise ValueError("simulated_external_emotion_requires_isolated_experiment")
        if ctx.external_emotion_experiment is not None and (
            ctx.external_emotion_experiment != self.external_emotion_experiment
        ):
            raise ValueError("synthetic_experiment_context_requires_its_isolated_engine")
        if self.external_emotion_experiment is not None:
            self.external_emotion_experiment.validate(self.store, self.persona)
            ctx.external_emotion_experiment = self.external_emotion_experiment
        if ctx.simulation_mode and not self.simulation_mode:
            raise ValueError("synthetic session cannot enter a real engine or store")
        if self.simulation_mode:
            ctx.simulation_mode = True

    def stage_pending_action(
        self,
        ctx: EngineContext,
        tool: str,
        args: dict[str, Any],
        *,
        action_id: str | None = None,
        ttl_turns: int = 2,
    ) -> dict[str, Any]:
        """可信服务端为后续泛确认暂存动作；不接受模型自行创建的 pending。"""
        self._validate_context_mode(ctx)
        registered = self._tools.get(tool)
        if registered is None:
            raise ValueError(f"unknown tool: {tool}")
        allowed_args = set(
            (getattr(registered, "parameters", {}) or {}).keys()
        )
        return stage_pending_action(
            ctx,
            tool,
            args,
            action_id=action_id,
            ttl_turns=ttl_turns,
            allowed_args=allowed_args,
        )

    def process_turn(
        self,
        ctx: EngineContext,
        utterance: str,
        acoustic: Optional[dict[str, Any]] = None,
        text_emotion: Optional[dict[str, Any]] = None,
        multimodal: Optional[dict[str, Any]] = None,
        *,
        external_emotion: Optional[dict[str, Any]] = None,
    ) -> str:
        if ctx.working_memory.get("_offline_session_sealed"):
            raise ValueError("closed_session_requires_a_new_context")
        external_reading = parse_external_emotion(external_emotion)
        if (is_simulated(external_reading) and not self.simulation_mode
                and self.external_emotion_experiment is None):
            raise ValueError("simulated_external_emotion_requires_isolated_experiment")
        # Explicit research opt-in is mandatory; reject before modifying a real
        # context, persona, evidence ledger, or baseline. Missing-input controls
        # remain valid in the isolated simulation engine.
        if not self.simulation_mode and (
            is_simulated(acoustic) or is_simulated(multimodal) or is_simulated(text_emotion)
        ):
            raise ValueError("simulated observations require an isolated simulation engine")
        self._validate_context_mode(ctx)
        semantic_screening = (
            self.policy.online_decision_version == ONLINE_DECISION_VERSION_SAFETY
        )
        # Reducing privacy permission must never wait for a model's safety
        # answer. This narrow helper does not collect, remind or notify anyone.
        if semantic_screening and apply_cognitive_withdrawal(ctx, self.persona, utterance):
            if self.store is not None and self.persona is not None:
                self.store.save(self.persona)
        current_turn_id = ctx.working_memory.pop("_current_turn_id", None)
        acoustic = prepare_acoustic(
            acoustic, turn_index=len(ctx.turns), turn_id=current_turn_id,
        )
        text_emotion = prepare_acoustic(
            text_emotion, turn_index=len(ctx.turns), turn_id=current_turn_id,
        )
        multimodal = prepare_multimodal(
            multimodal, turn_index=len(ctx.turns), turn_id=current_turn_id,
        )
        ctx.working_memory["current_utterance"] = utterance
        # 这些字段只描述“这一轮”。高风险会在常规感知/推理之前早退，因此若不
        # 先清理，Web 审计视图会把上一轮的情绪、干预、Skill 或反诈判断错误地
        # 展示成急症回合的依据。会话累计的 evidence/outcomes/delivery history
        # 仍然保留，结果学习不会被抹掉。
        for key in (
            "risk_flag",
            "last_care_action",
            "skill_proposals",
            "personality_decision",
            "online_turn_decision",
            "intervention_delivery",
            "observation_execution",
            "fraud_assessment",
            "cognitive_assessment",
            "care_events",
            "_advertised_tools_for_turn",
            "_consumed_tool_result_for_turn",
            "_delivery_response_candidate_for_turn",
            "_benchmark_reply_trace",
            "delivery_response_guard",
            "_reply_delivery_stages",
            "perception_input",
            "perception_fusion",
            "prompt_evidence_snapshot",
        ):
            ctx.working_memory.pop(key, None)
        ctx.reasoning.hypotheses = []
        ctx.reasoning.candidates = []
        ctx.reasoning.selected = None
        ctx.reasoning.observation_plan = None
        ctx.emotion = EmotionState(
            label="unknown",
            inference_source="awaiting_current_turn_evidence",
        )
        self._seed_persistent_tasks(ctx)

        # 所有可选时间序列先与这个人自己的历史基线比较。只保存聚合统计，
        # 不保存原始音视频；上游已经给 baseline_delta 时保持原值。
        multimodal = enrich_with_personal_baseline(self.persona, multimodal)
        ctx.working_memory["perception_input"] = {
            "acoustic": deepcopy(acoustic),
            "text_emotion": deepcopy(text_emotion),
            "multimodal": deepcopy(multimodal),
        }
        if external_reading is not None:
            ctx.working_memory["perception_input"]["external_emotion"] = deepcopy(external_reading)
        ctx.working_memory["model_route"] = {}

        # Every early-return path can be checkpointed by the UI. Preserve the
        # existing relationship baseline before either safety route can return.
        if self.persona is not None and not ctx.working_memory.get("_alliance_seeded"):
            ctx.alliance.trust = self.persona.alliance.trust
            ctx.working_memory["_alliance_seeded"] = True

        # 明确高风险先于任何生成式调用接管，不能让胸痛/呼吸困难先排队等状态模型。
        signal = self._risk.assess(ctx)
        if signal is not None and signal.tier >= 2:
            self._record_risk(ctx, signal)
            ctx.working_memory["risk_flag"] = {
                "tier": signal.tier,
                "kind": signal.kind,
            }
            ctx.working_memory["model_route"] = {
                "state": {
                    "tier": "skipped",
                    "reasons": ["deterministic_safety_takeover"],
                    "latency_ms": 0,
                },
                "reply": {
                    "tier": "deterministic",
                    "reasons": ["risk_tier_2_or_3"],
                    "configured_model": None,
                    "effective_model": None,
                    "latency_ms": 0,
                },
            }
            care_observer = getattr(self.scene, "observe_care", None)
            care_changed = (
                bool(care_observer(ctx, self.persona, utterance, signal))
                if callable(care_observer) else False
            )
            response = self._handle_risk(ctx, signal)
            ctx.add_turn(utterance, response)
            self._mark_interaction_started()
            if (
                care_changed
                and self.store is not None
                and self.persona is not None
            ):
                self.store.save(self.persona)
            return response

        learning_before_screening = None
        if semantic_screening:
            # The same online call still needs tentative state for its prompt.
            # Commit none of that learning until its safety field is validated.
            learning_before_screening = deepcopy((
                ctx.reasoning, ctx.alliance, ctx.formulation, ctx.active_tasks,
            ))

        # Keyword safety already passed; v3 changes here remain tentative until
        # same-turn semantic safety also passes. Emergencies are never learned
        # as successful continued engagement with the previous intervention.
        if self._state_reasoner is not None:
            self._state_reasoner.observe_response(ctx, utterance)

        # 1. 感知只融合当轮可观察信号。复杂语义不会在这里另调一次模型，
        #    而是在策略阶段与沟通决策、回复合并为一次在线生成决策；底层
        #    瞬时错误重试或同职责回退可能产生额外 HTTP 请求。
        state_route = route_state_analysis(
            utterance,
            ctx.turns,
            acoustic=acoustic,
            text=text_emotion,
            multimodal=multimodal,
        )
        state_started = time.perf_counter()
        ctx.emotion = self.perception.perceive(
            utterance,
            acoustic,
            text_emotion,
            history=ctx.turns,
            prosodic=(acoustic or {}).get("prosodic"),
            emotion_trace=(acoustic or {}).get("emotion_trace"),
            analysis_route=state_route,
        )
        if external_reading is not None:
            # A soft online hint, never an acoustic observation or personality
            # evidence. Keep the actual perception audit separate and explicit.
            ctx.emotion.external_reading = deepcopy(external_reading)
            ctx.emotion.label = external_reading["label"]
            ctx.emotion.score = external_reading["confidence"]
            ctx.emotion.inference_source = "external_model"
            ctx.emotion.divergence = False
            ctx.emotion.pattern = ""
        ctx.working_memory["perception_fusion"] = asdict(ctx.emotion)
        state_elapsed = (time.perf_counter() - state_started) * 1000
        if state_route.needs_integrated_decision:
            state_info = {
                "tier": "online_integrated",
                "reasons": list(state_route.reasons),
                "source": "online_decision",
                "configured_model": None,
                "effective_model": None,
                "latency_ms": 0,
                "status": "pending_single_pass",
            }
        elif state_route.tier == "structured":
            state_info = {
                "tier": "structured",
                "reasons": list(state_route.reasons),
                "source": "deterministic_evidence_pair",
                "configured_model": None,
                "effective_model": None,
                "latency_ms": round(state_elapsed),
                "status": "structured_evidence",
            }
        else:
            state_info = {
                "tier": "skipped",
                "reasons": list(state_route.reasons),
                "source": "deterministic_gate",
                "configured_model": None,
                "effective_model": None,
                "latency_ms": round(state_elapsed),
                "status": "skipped",
            }
        ctx.working_memory["model_route"]["state"] = state_info

        # 2. 更新结构化状态(非 LLM):关系联盟 + 个案假设
        #    Existing trust was seeded before safety handling, including early returns.
        update_alliance(ctx, utterance)
        update_formulation(ctx, self.persona)

        # 2a. 当前轮风险优先于主动观察和任务。tier1 先留下“降速”标志，
        #     让观察策略知道这一拍不要继续收集信息；tier≥2 直接确定性接管。
        if signal is not None:
            self._record_risk(ctx, signal)
            # tier≥2 已在任何模型调用前接管；这里仅剩 tier1 的降速姿态。
            ctx.working_memory["risk_flag"] = {
                "tier": signal.tier,
                "kind": signal.kind,
            }
        care_observer = getattr(self.scene, "observe_care", None)
        care_changed = (
            bool(care_observer(ctx, self.persona, utterance, signal))
            if callable(care_observer) and not semantic_screening else False
        )
        # 2b. 场景状态推理:文本永远有;声音/视觉/行为/环境等有输入才作为证据加入。
        #     这一层维护竞争假设和干预候选，不做疾病诊断。
        reasoning_before_current = None
        observation_plan_for_turn = None
        if self._state_reasoner is not None:
            # 先形成可解释的候选供在线模型选择；模型只补充当轮语义证据。
            # 若模型返回了有效候选，稍后从本轮前快照重算一次，避免重复计证。
            reasoning_before_current = deepcopy(ctx.reasoning)
            evidence = collect_evidence(
                utterance, ctx.emotion,
                turn_index=len(ctx.turns),
                multimodal=multimodal,
            )
            self._state_reasoner.update(ctx, self.persona, utterance, evidence)
            # P0 是生成当前回复时使用的观察合同。在线模型提出的语义候选
            # 可在稍后重算证据/假设/候选，但不能事后把本轮计划改成 P1。
            observation_plan_for_turn = deepcopy(
                ctx.reasoning.observation_plan
            )

        # 2c. 同轮 Skill:陪伴关系不切场景；显式任务作为第二条目标并行进入策略。
        self._skills.evaluate(ctx, self.persona, utterance)
        care_skill_executor = getattr(
            self.scene, "execute_care_skills", None
        )
        care_skill_changed = (
            bool(care_skill_executor(ctx, self.persona))
            if (callable(care_skill_executor) and self.store is not None
                and not semantic_screening)
            else False
        )
        if (
            care_skill_changed
            and self.store is not None
            and self.persona is not None
        ):
            self._persist_runtime_state(ctx)
            self.store.save(self.persona)

        # 3. 检索知识:领域 KB + 个人记忆召回 → working_memory
        self._retrieve_knowledge(ctx, utterance)

        # 5. 策略 → 回应(persona × 情绪,带着检索到的知识和可用工具)
        reply_route = route_reply_model(
            ctx,
            utterance,
            state_route=state_route,
        )
        raw = self.policy.respond(
            ctx, self.scene, utterance,
            persona=self.persona,
            available_tools=self._tools.describe(),
            model_route=reply_route,
        )
        online_trace = ctx.working_memory.get("online_turn_decision") or {}
        model_signal = semantic_safety_signal(ctx, utterance)
        semantic_clear = (
            online_trace.get("version") == ONLINE_DECISION_VERSION_SAFETY
            and online_trace.get("status") == "parsed"
            and online_trace.get("turn_index") == len(ctx.turns)
            and online_trace.get("safety_source") == "online_model_semantics"
            and online_trace.get("safety_concern") == "none"
        )
        if semantic_screening and not semantic_clear:
            # Keep the received protocol/route audit, but roll back tentative
            # outcomes, trust, formulation and task controls before ANY save.
            # A later Web checkpoint must not persist a false intervention win.
            (ctx.reasoning, ctx.alliance, ctx.formulation,
             ctx.active_tasks) = learning_before_screening
            for key in ("personality_decision", "skill_proposals",
                        "_delivery_response_candidate_for_turn", "_cognitive_block_selected"):
                ctx.working_memory.pop(key, None)
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
            reason = "semantic_safety_takeover" if model_signal else "semantic_safety_unavailable"
            ctx.working_memory["intervention_delivery"] = {
                "status": "not_delivered", "turn_index": len(ctx.turns),
                "intervention_id": None, "reason": reason,
            }
            if model_signal is None:
                # Missing/invalid safety is not a clear result. Deliver only
                # fallback, without scene observers, actions or task flows.
                ctx.working_memory["model_route"]["safety"] = {
                    "source": "online_model_semantics", "status": "unavailable",
                    "response_source": "deterministic_fallback",
                }
                ctx.add_turn(utterance, DETERMINISTIC_MODEL_FAILURE_REPLY)
                self._mark_interaction_started()
                return DETERMINISTIC_MODEL_FAILURE_REPLY
        if model_signal is not None:
            self._record_risk(ctx, model_signal)
            ctx.working_memory["risk_flag"] = {
                "tier": model_signal.tier, "kind": model_signal.kind,
                "source": "online_model_semantics", "diagnosis": False,
            }
            # A safety concern never becomes ordinary memory/personality evidence.
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
            if "_cognitive_source_turns" in ctx.working_memory:
                ctx.working_memory["_cognitive_source_turns"] = [
                    row for row in ctx.working_memory["_cognitive_source_turns"]
                    if row.get("turn_index") != len(ctx.turns)
                ]
            ctx.working_memory.pop("_delivery_response_candidate_for_turn", None)
            ctx.working_memory["model_route"]["safety"] = {
                "source": "online_model_semantics", "response_source": "deterministic_safety",
                "concern": online_trace["safety_concern"],
            }
            # Do not parse or execute the generated action/body on this branch.
            response = self._handle_risk(ctx, model_signal)
            from .policy.delivery_provenance import record_reply_stage
            record_reply_stage(ctx, "semantic_safety", raw, response,
                               violations=["semantic_safety_takeover"])
            ctx.add_turn(utterance, response)
            self._mark_interaction_started()
            if self.store is not None and self.persona is not None:
                self._persist_runtime_state(ctx)
                self.store.save(self.persona)
            return response
        if semantic_screening:
            # These extension hooks can mutate consent, tasks or external care
            # state, so v3 may invoke them only after a current validated none.
            care_changed = (
                bool(care_observer(ctx, self.persona, utterance, signal))
                if callable(care_observer) else False
            )
            care_skill_changed = (
                bool(care_skill_executor(ctx, self.persona))
                if callable(care_skill_executor) and self.store is not None else False
            )
            if (care_changed or care_skill_changed) and self.store is not None and self.persona is not None:
                self._persist_runtime_state(ctx)
                self.store.save(self.persona)
        if (
            self._state_reasoner is not None
            and reasoning_before_current is not None
            and online_trace.get("status") == "parsed"
        ):
            selected_id = str(
                online_trace.get("selected_intervention_id") or ""
            )
            ctx.reasoning = reasoning_before_current
            evidence = collect_evidence(
                utterance,
                ctx.emotion,
                turn_index=len(ctx.turns),
                multimodal=multimodal,
            )
            self._state_reasoner.update(
                ctx, self.persona, utterance, evidence
            )
            ctx.reasoning.observation_plan = observation_plan_for_turn
            selected = next(
                (
                    item for item in ctx.reasoning.candidates
                    if item.id == selected_id
                ),
                None,
            )
            if selected is not None:
                selected.selected_turn = len(ctx.turns)
                ctx.reasoning.selected = selected
        # 5b. 结构化动作(P3-#13):解析 → 执行 → 回执留给下一轮 prompt。
        consumed_tool_result = ctx.working_memory.pop(
            "last_tool_result", None
        )
        if isinstance(consumed_tool_result, dict):
            # 上一轮回执已经进入本轮 prompt，但仍须保留到最终交付校验；
            # 否则模型如实复述上次成功动作也会在 pop 后被误判为无回执。
            ctx.working_memory["_consumed_tool_result_for_turn"] = (
                consumed_tool_result
            )
        response, action = parse_action(raw)
        ctx.working_memory.setdefault(
            "_delivery_response_candidate_for_turn", response
        )
        from .policy.delivery_provenance import record_reply_stage
        response_overridden = False
        response_guard = getattr(self.scene, "guard_response", None)
        if callable(response_guard):
            guarded = response_guard(ctx, response)
            record_reply_stage(ctx, "scene_guard", response, guarded,
                               violations=["scene_response_replaced"] if guarded != response else [])
            if guarded != response:
                response = guarded
                action = None
                response_overridden = True
        before_tool = response
        if action is not None:
            pending_action = current_pending_action(ctx)
            prepared, denied_reason = prepare_action(
                action,
                utterance,
                self.persona,
                executed=ctx.working_memory.setdefault(
                    "_executed_action_signatures", set()
                ),
                pending_action=pending_action,
            )
            if prepared is None:
                ctx.working_memory["last_tool_result"] = {
                    "tool": action["tool"],
                    "ok": False,
                    "data": {"reason": denied_reason, "executed": False},
                }
                response = (
                    "这个动作还没有执行："
                    f"{denied_reason}。"
                )
                response_overridden = True
            else:
                result = self._tools.invoke(
                    prepared["tool"], **prepared["args"]
                )
                ctx.working_memory["last_tool_result"] = {
                    "tool": prepared["tool"],
                    "ok": result.ok,
                    "args": dict(prepared["args"]),
                    "data": result.data,
                }
                pending_action_id = str(
                    prepared.get("pending_action_id") or ""
                )
                if result.ok and pending_action_id:
                    consume_pending_action(ctx, pending_action_id)
                if not result.ok:
                    signatures = ctx.working_memory.setdefault(
                        "_executed_action_signatures", set()
                    )
                    signatures.discard(json.dumps(
                        {
                            "tool": prepared["tool"],
                            "args": prepared["args"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ))
                    note = str(
                        result.data.get("note")
                        or result.data.get("error")
                        or "外部能力没有返回成功回执"
                    )
                    response = (
                        "这个动作还没有执行："
                        f"{note.rstrip('。')}。"
                    )
                    response_overridden = True

        record_reply_stage(ctx, "tool_result", before_tool, response,
                           violations=["tool_response_replaced"] if before_tool != response else [])

        # 6. 可选多步任务流；养老 Skill 需要等待/恢复时也可使用，不切换关系场景。
        flow = self.scene.task_flow()
        if flow is not None:
            flow.advance(ctx)

        # 工具与确定性 Skill 都执行完后再检查完成态。综合推理的 pre-guard
        # 不能在真实动作尚未取得回执时提前删除合法 <action>。
        delivery_response_guard = getattr(
            self.scene, "guard_delivery_response", None
        )
        if callable(delivery_response_guard):
            guarded = delivery_response_guard(
                ctx, response, utterance, self.persona
            )
            audit = ctx.working_memory.get("delivery_response_guard") or {}
            record_reply_stage(ctx, "delivery_guard", response, guarded,
                               violations=audit.get("violations", []))
            if guarded != response:
                response = guarded
                response_overridden = True
        ctx.working_memory.pop("_consumed_tool_result_for_turn", None)
        ctx.working_memory.pop("_delivery_response_candidate_for_turn", None)

        # 7. 写回上下文
        if self._state_reasoner is not None:
            record_observation = getattr(
                self._state_reasoner,
                "record_observation_execution",
                None,
            )
            if callable(record_observation):
                record_observation(ctx, response)
            record_delivery = getattr(
                self._state_reasoner,
                "record_intervention_delivery",
                None,
            )
            if callable(record_delivery):
                record_delivery(
                    ctx,
                    response,
                    response_overridden=response_overridden,
                )
        ctx.add_turn(utterance, response)
        self._mark_interaction_started()
        return response

    def _mark_interaction_started(self) -> None:
        """首轮 prompt 保持真冷启动；回应完成后才进入观察阶段。"""
        if (
            self.persona is not None
            and self.persona.person_model_state.first_interaction_at is None
        ):
            self.persona.person_model_state.first_interaction_at = (
                datetime.now().isoformat(timespec="seconds")
            )

    def finalize_session(self, ctx: EngineContext) -> None:
        """Close durably, then make one bounded, chronological offline pass.

        Retrying this method never appends new conversation to a failed job.
        Online callers use checkpoint_session, never this deep-model path.
        """
        self._validate_context_mode(ctx)
        if self.simulation_mode or self.persona is None:
            return
        if ctx.working_memory.get("_session_finalized"):
            return
        self.seal_session(ctx)
        if ctx.working_memory.get("_session_finalized"):
            return
        recovery = self.recover_offline_sessions(max_jobs=8)
        completed = ctx.session_id in self.persona.person_model_state.completed_session_ids
        if completed:
            ctx.working_memory["_session_finalized"] = True
            ctx.working_memory.pop("_offline_finalize_pending", None)
        else:
            ctx.working_memory["_offline_finalize_pending"] = True
        for job in self.persona.offline_sessions:
            if job.session_id == ctx.session_id:
                ctx.working_memory["_memory_consolidated"] = job.memory_completed
                ctx.working_memory["_reflection_completed"] = job.reflection_completed

    def _accept_offline_profile(self, persona: PersonaProfile) -> None:
        """Publish only committed state, retaining the public profile identity."""
        if self.persona is None:
            self.persona = persona
        elif self.persona is not persona:
            for name in type(persona).model_fields:
                setattr(self.persona, name, deepcopy(getattr(persona, name)))

    def _offline_queue_result(self, queue: OfflineSessionQueue, result: dict) -> dict:
        self._accept_offline_profile(queue.persona)
        if result.get("error_code") == "offline_store_failed":
            raise RuntimeError("offline_store_failed")
        return result

    def prepare_offline_sessions(self) -> dict:
        """Migrate/prune storage only; never wait for or call a cloud model."""
        if self.persona is None or self.simulation_mode:
            return {"pending_count": 0}
        queue = OfflineSessionQueue(self.persona, store=self.store)
        migration = queue.migrate_legacy_pending()
        self._offline_queue_result(queue, migration)
        if (migration.get("error_code") and self.persona.pending_session is not None
                and self.persona.pending_session.offline_finalize_pending):
            # A full queue cannot silently erase the older legacy checkpoint.
            raise RuntimeError("offline_legacy_migration_pending")
        self._offline_queue_result(queue, queue.prune())
        return queue.status()

    def offline_status(self) -> dict:
        if self.persona is None or self.simulation_mode:
            return {"pending_count": 0}
        return OfflineSessionQueue(self.persona, store=self.store).status()

    def seal_session(self, ctx: EngineContext) -> dict:
        """Persist runtime once and freeze text before any offline model call.

        Reconstructed offline contexts must never overwrite trust or active
        tasks. Queue/storage refusal raises before Web releases the live pair.
        """
        self._validate_context_mode(ctx)
        if self.simulation_mode or self.persona is None:
            return {"status": "not_required", "pending_count": 0}
        self.prepare_offline_sessions()
        if ctx.working_memory.get("_offline_session_sealed"):
            return self.offline_status()
        if ctx.session_id in self.persona.person_model_state.completed_session_ids:
            ctx.working_memory["_session_finalized"] = True
            return self.offline_status()
        if not ctx.turns:
            if self.persona.pending_session and self.persona.pending_session.session_id == ctx.session_id:
                candidate = deepcopy(self.persona)
                candidate.pending_session = None
                if self.store is not None:
                    try:
                        self.store.save(candidate)
                    except Exception as exc:
                        raise RuntimeError("offline_store_failed") from exc
                self._accept_offline_profile(candidate)
            ctx.working_memory["_offline_session_sealed"] = True
            ctx.working_memory["_session_finalized"] = True
            ctx.working_memory.pop("_offline_finalize_pending", None)
            return self.offline_status()
        original = self.persona
        working_memory = deepcopy(ctx.working_memory)
        self.persona = deepcopy(original)
        try:
            already_queued = any(job.session_id == ctx.session_id
                                 for job in self.persona.offline_sessions)
            if not already_queued and not ctx.working_memory.get("_offline_recovery"):
                self._persist_runtime_state(ctx)
            queue = OfflineSessionQueue(self.persona, store=self.store)
            result = queue.enqueue(
                ctx,
                memory_completed=bool(ctx.working_memory.get("_memory_consolidated")),
                reflection_completed=bool(ctx.working_memory.get("_reflection_completed")),
                source_completeness=ctx.working_memory.get("_offline_source_completeness", "complete"),
            )
            if result.get("error_code"):
                raise RuntimeError("offline_session_seal_failed")
        except Exception:
            self.persona = original
            ctx.working_memory.clear()
            ctx.working_memory.update(working_memory)
            raise
        self.persona = original
        self._accept_offline_profile(queue.persona)
        ctx.working_memory["_offline_session_sealed"] = True
        ctx.working_memory["_offline_finalize_pending"] = bool(self.persona.offline_sessions)
        return self.offline_status()

    def _offline_memory_stage(self, persona: PersonaProfile, ctx: EngineContext):
        if self._consolidator is None:
            return persona, "skipped", None
        with OFFLINE_SESSION_LOCK:
            updated = self._consolidator.consolidate(persona, ctx, persist=False)
            extractor = self._consolidator.extractor
            status = getattr(extractor, "last_status", "ok")
            if status == "failed":
                client = getattr(extractor, "llm", None)
                control = getattr(client, "model_request_control", None)
                terminal = getattr(control, "memory_failure_is_terminal", None)
                if callable(terminal) and terminal(client):
                    # Production may close this failed session without repeating
                    # its received body or exposing unscreened text to reflection.
                    # Strict experimental guards retain their own original rule.
                    return persona, "sealed_failed", "memory_extraction_failed"
                return persona, "failed", "memory_extraction_failed"
            return updated, "ok", None

    def _offline_reflection_stage(self, persona: PersonaProfile, ctx: EngineContext):
        if self.hypothesis_updater is None:
            update_delivery_state(persona)
            return persona, "skipped", None
        reflection_client = (
            getattr(self.hypothesis_updater, "reflection_metered_client", None)
            or getattr(self.hypothesis_updater, "llm", None)
        )
        if (reflection_client is not None
                and ctx.working_memory.get("_memory_privacy_review_completed") is not True):
            # Old persisted memory-completed jobs have no semantic privacy
            # receipt. Never expose their raw conversation or silently re-run it.
            return persona, "sealed_failed", "memory_privacy_unreviewed"
        with OFFLINE_SESSION_LOCK:
            update = getattr(self.hypothesis_updater, "update", None)
            if not callable(update):
                update = self.hypothesis_updater.diagnose
            updated = update(persona, ctx)
            status = getattr(self.hypothesis_updater, "last_status", "ok")
            if status == "failed":
                detail = getattr(self.hypothesis_updater, "last_error_detail", None)
                # Preserve a reviewed diagnostic across queue persistence. Never
                # stringify provider exceptions or arbitrary updater values.
                error_code = (
                    detail if type(detail) is str and detail in _REFLECTION_ERROR_DETAILS
                    else "person_reflection_failed"
                )
                if getattr(self.hypothesis_updater, "body_json_protocol", False):
                    client = getattr(self.hypothesis_updater, "reflection_metered_client",
                                     getattr(self.hypothesis_updater, "llm", None))
                    control = getattr(client, "model_request_control", None)
                    diagnostic = getattr(client, "last_diagnostic", {})
                    terminal = getattr(self.hypothesis_updater, "last_error_type", None) in {
                        "invalid_json_response", "invalid_evidence_review_protocol"}
                    terminal = terminal or (type(diagnostic) is dict and diagnostic.get("error_type") in {
                        "timeout", "network_error", "http_error", "invalid_reflection_body_channel"})
                    if not terminal and control is not None:
                        settled_failure = getattr(control, "reflection_failure_is_terminal", None)
                        if callable(settled_failure):
                            try:
                                terminal = settled_failure(client) is True
                            except Exception:
                                terminal = False
                    # A received audit failure may finish this optional stage.
                    # Storage/authorization/budget failures and unresolved
                    # transport claims still require the caller's hard stop.
                    if terminal and control is not None:
                        try:
                            accounting = control.snapshot()
                            terminal = (accounting.get("stopped") is False
                                        and not any(row.get("status") == "pending"
                                                    for row in accounting.get("attempts", [])))
                        except Exception:
                            terminal = False
                    if terminal:
                        return persona, "sealed_failed", error_code
                return persona, "failed", error_code
            update_delivery_state(updated)
            return updated, "skipped" if status == "skipped" else "ok", None

    def recover_offline_sessions(self, *, max_jobs: int = 1) -> dict:
        """Explicit offline recovery, FIFO and bounded; no runtime-state replay."""
        if self.persona is None or self.simulation_mode:
            return {"pending_count": 0}
        self.prepare_offline_sessions()
        queue = OfflineSessionQueue(self.persona, store=self.store)
        result = queue.drain(
            self._offline_memory_stage, self._offline_reflection_stage,
            max_jobs=max_jobs,
        )
        self._offline_queue_result(queue, result)
        return {**self.offline_status(), "last_run": result}

    def _persist_finalize_pending(self, ctx: EngineContext) -> None:
        """Legacy private entry delegates to the durable closed-session queue."""
        self.seal_session(ctx)

    def checkpoint_session(self, ctx: EngineContext) -> None:
        """轻量增量检查点。

        不调用会后大模型，也不把单次在线假设升级为长期人格；只保存已经形成的
        个人基线、关系状态和可去重的干预结果。浏览器意外关闭或服务重启时，
        至少不会丢掉这些结构化进展。深度记忆与人物反思仍在语义会话结束时执行。
        """
        self._validate_context_mode(ctx)
        if self.persona is None or self.store is None:
            return
        if ctx.working_memory.get("_offline_session_sealed"):
            raise ValueError("closed_session_cannot_be_checkpointed_as_live")
        self._persist_runtime_state(ctx)
        complete = (len(ctx.turns) <= OFFLINE_MAX_TURNS
                    and sum(len(t.user) + len(t.agent) for t in ctx.turns) <= OFFLINE_MAX_CHARACTERS
                    and ctx.working_memory.get("_offline_source_completeness", "complete") == "complete")
        excluded = {item for item in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])
                    if type(item) is int and item >= 0}
        indices = [index for index in range(len(ctx.turns)) if index not in excluded] if complete else []
        self.persona.pending_session = SessionCheckpointRecord(
            session_id=ctx.session_id,
            scene_id=ctx.scene_id,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            turns=([SessionTurnSnapshot(user=ctx.turns[i].user, agent=ctx.turns[i].agent)
                    for i in indices] if complete else _checkpoint_turns(ctx)),
            cognitive_turns=cognitive_sources(self.persona, ctx) if complete else [],
            cognitive_session_at=(ctx.working_memory.get("_cognitive_session_at")
                                  if self.persona.cognitive_health_consent.has_text_monitoring_consent() else None),
            original_turn_count=len(ctx.turns) if complete else None,
            turn_indices=indices,
            expires_at=(
                datetime.now() + timedelta(hours=24)
            ).isoformat(timespec="seconds"),
            memory_consolidated=bool(
                ctx.working_memory.get("_memory_consolidated")
            ),
            reflection_completed=bool(
                ctx.working_memory.get("_reflection_completed")
            ),
            offline_finalize_pending=bool(
                ctx.working_memory.get("_offline_finalize_pending")
            ),
        )
        self.store.save(self.persona)
        ctx.working_memory["_last_checkpoint_turn"] = len(ctx.turns)

    def _persist_runtime_state(self, ctx: EngineContext) -> None:
        self._validate_context_mode(ctx)
        if self.persona is None:
            return
        # 干预结果可多次检查点写回；reasoner 以 session/turn/strategy 去重。
        if self._state_reasoner is not None:
            self._state_reasoner.finalize(ctx, self.persona)
        self.persona.alliance.trust = ctx.alliance.trust
        persisted_ruptures = int(
            ctx.working_memory.get("_persisted_rupture_events", 0)
        )
        new_ruptures = max(
            0, ctx.alliance.rupture_events - persisted_ruptures
        )
        self.persona.alliance.rupture_count += new_ruptures
        ctx.working_memory["_persisted_rupture_events"] = (
            ctx.alliance.rupture_events
        )
        now = datetime.now().isoformat(timespec="seconds")
        self.persona.last_interaction_at = now
        state = self.persona.person_model_state
        if state.first_interaction_at is None:
            state.first_interaction_at = now
        self.persona.active_tasks = [
            PersistentTaskRecord(
                id=item.id,
                skill_id=item.skill_id,
                intent=item.intent,
                goal=item.goal,
                status=item.status,
                created_at=item.created_at or now,
                updated_at=item.updated_at or now,
                slots={
                    key: value for key, value in item.slots.items()
                    if not key.startswith("_")
                },
                missing_slots=list(item.missing_slots),
                required_capability=item.required_capability,
                capability_available=item.capability_available,
                consent_required=item.consent_required,
                resume_status=item.resume_status,
                history=list(item.history[-20:]),
            )
            for item in ctx.active_tasks
            if item.status not in {"completed", "failed", "cancelled"}
        ][-20:]

    def _mark_session_completed(self, ctx: EngineContext) -> None:
        if self.persona is None or not ctx.turns:
            return
        state = self.persona.person_model_state
        if ctx.session_id in state.completed_session_ids:
            return
        state.completed_session_ids.append(ctx.session_id)
        state.completed_session_ids = state.completed_session_ids[-100:]
        state.sessions_completed += 1

    def _seed_persistent_tasks(self, ctx: EngineContext) -> None:
        if ctx.working_memory.get("_tasks_seeded"):
            return
        ctx.working_memory["_tasks_seeded"] = True
        if self.persona is None:
            return
        ctx.active_tasks = [
            ActiveTask(
                id=item.id,
                skill_id=item.skill_id,
                intent=item.intent,
                goal=item.goal,
                status=self._skills.restored_status(
                    item.status,
                    required_capability=item.required_capability,
                    missing_slots=item.missing_slots,
                ),
                created_turn=-1,
                updated_turn=-1,
                slots=dict(item.slots),
                missing_slots=list(item.missing_slots),
                required_capability=item.required_capability,
                capability_available=self._skills.has_capability(
                    item.required_capability
                ),
                consent_required=item.consent_required,
                created_at=item.created_at,
                updated_at=item.updated_at,
                resume_status=item.resume_status,
                history=[
                    *item.history[-19:],
                    f"{item.updated_at}:restored",
                ],
            )
            for item in self.persona.active_tasks
        ]

    def record_intervention_outcome(
        self,
        ctx: EngineContext,
        value: str,
        *,
        signal: str = "explicit_feedback",
        note: str = "",
    ) -> None:
        """显式记录真实结果。

        例:家属端确认完成视频、老人参加活动、本人拒绝建议。没有状态推理插件时
        安全 no-op，调用方无需按场景分支。
        """
        if self._state_reasoner is not None:
            self._state_reasoner.record_outcome(
                ctx, value, signal=signal, note=note
            )





    def _retrieve_knowledge(self, ctx: EngineContext, utterance: str) -> None:
        """领域知识 + 个人记忆 → ctx.working_memory。

        个人记忆只召回「可主动引用」的(私密 episode 不进 policy 的可说区)。
        """
        domain = self._knowledge.retrieve(utterance) if self._knowledge else []
        episodes: list[str] = []
        episode_sources: list[dict] = []
        if self.persona is not None:
            eps = recall_episodes(
                self.persona, utterance, top_k=2, referenceable_only=True
            )
            episodes = [e.summary for e in eps]
            episode_sources = [e.model_dump() for e in eps]
        ctx.working_memory["domain_knowledge"] = domain
        ctx.working_memory["recalled_episodes"] = episodes
        # Parallel audit provenance from the actual recall result. Prompt text
        # still contains only summaries, and no later profile lookup is needed.
        ctx.working_memory["recalled_episode_sources"] = episode_sources
        world: list[WorldKnowledgeItem] = []
        if self._world_knowledge is not None:
            try:
                world = [
                    item
                    for item in self._world_knowledge.retrieve(
                        utterance, top_k=3
                    )
                    if isinstance(item, WorldKnowledgeItem)
                    and item.content.strip()
                    and item.source.strip()
                    and item.observed_at.strip()
                    and item.is_current()
                ]
            except Exception:                       # 公共信息失败不影响陪伴主链
                world = []
        ctx.working_memory["world_knowledge"] = world

    def _handle_risk(self, ctx: EngineContext, signal: RiskSignal) -> str:
        """tier ≥2 接管:确定性安全回应(热线 12356 / 紧急 120 / 推向真人)。

        不过 LLM——安全底线不交给生成模型抽卡。通知照护人只做建议,
        自动通知需要真实通道 + 当次同意,接入口在此分流。
        """
        return safety_response(signal, self.persona)

    def _record_risk(self, ctx: EngineContext, signal: RiskSignal) -> None:
        """风险事件落账(每会话每 kind 最多一条,避免刷屏)——纵向基线的原始信号。"""
        if self.persona is None:
            return
        logged: set = ctx.working_memory.setdefault("_risk_logged_kinds", set())
        if signal.kind in logged:
            return
        logged.add(signal.kind)
        note = (signal.evidence[0][:60] if signal.evidence else None)
        self.persona.risk_events.append(RiskEvent(
            at=datetime.now().isoformat(timespec="seconds"),
            tier=signal.tier, kind=signal.kind, note=note,
        ))
