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
import hashlib
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
    SafetyScreeningRecord,
    SessionCheckpointRecord,
    SessionTurnSnapshot,
)
from .risk.safety import safety_response
from .risk.semantic import semantic_safety_signal
from .risk.safety_delivery import (
    LEGACY, CONDITIONAL_CONTEXT_V1, TASK_CONTEXT_V2, SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1, capture_delivery_context,
    render_safety_delivery, render_unverified_safety, validate_delivery_version,
)
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
    EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE, SAFETY_DECISION_VERSIONS,
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
        SessionTurnSnapshot(user=turn.user, agent=turn.agent,
            person_response_sha256=(ctx.working_memory.get("_person_response_deliveries") or {}).get(str(index)))
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
        safety_review_llm_client: Optional[LLMClient] = None,
        confirm_judge_llm_client: Optional[LLMClient] = None,
        safety_review_version: str = "safety_review_v2",
        simulation_mode: bool = False,
        delivery_block: Optional[str] = None,
        external_emotion_experiment: SyntheticEmotionExperiment | None = None,
        safety_delivery_version: str = LEGACY,
        safety_response_instruction_version: str = "legacy",
        premodel_safety_version: str = "legacy",
    ) -> None:
        from .risk.premodel_task import validate_premodel_safety_version
        self.premodel_safety_version = validate_premodel_safety_version(premodel_safety_version)
        self.safety_delivery_version = validate_delivery_version(safety_delivery_version)
        from .policy.safety_instruction import SafetyResponseInstruction
        self._safety_response_instruction = SafetyResponseInstruction(safety_response_instruction_version)
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
        # An explicitly budgeted, deletion-only client; never auto-created.
        self.safety_review_llm = safety_review_llm_client
        self.confirm_judge_llm = confirm_judge_llm_client
        if safety_review_version not in {"safety_review_v1", "safety_review_v2"}:
            raise ValueError("unsupported_safety_review_version")
        self.safety_review_version = safety_review_version
        # 仅为旧版集成保留公开属性。在线主链路不再创建或调用独立状态模型；
        # 深度人物模型由会后 hypothesis_updater 负责。
        self.state_llm = state_llm_client
        self.policy = PolicyEngine(fast_llm=fast_llm, delivery_block=delivery_block,
            **({"safety_response_instruction_version": self.safety_response_instruction_version}
               if self.safety_response_instruction_version != "legacy" else {}))
        if (self.safety_delivery_version != LEGACY
                and self.policy.online_decision_version not in EVIDENCE_DECISION_VERSIONS):
            raise ValueError("conditional_safety_delivery_requires_online_v4")
        if (self.safety_delivery_version == TASK_CONTEXT_V2
                and self.safety_response_instruction_version != "conditional_response_v2"):
            raise ValueError("task_safety_delivery_requires_response_v2")
        if ((self.safety_delivery_version == SOURCE_RESPONSE_V3)
                != (self.safety_response_instruction_version == "conditional_response_v3")):
            raise ValueError("source_response_requires_matching_instruction_v3")
        if ((self.safety_delivery_version == SOURCE_RESPONSE_V4)
                != (self.safety_response_instruction_version == "conditional_response_v4")):
            raise ValueError("source_response_requires_matching_instruction_v4")
        if ((self.safety_delivery_version == MINIMAL_RESPONSE_V1)
                != (self.safety_response_instruction_version == "compact_response_v1")):
            raise ValueError("minimal_response_requires_matching_compact_instruction")
        from .risk.premodel_task import validate_premodel_configuration
        validate_premodel_configuration(self.premodel_safety_version, self.safety_delivery_version,
                                       self.safety_response_instruction_version, self.policy.online_decision_version)
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

    @property
    def safety_response_instruction_version(self):
        return self._safety_response_instruction.version

    def _validate_context_mode(self, ctx: EngineContext) -> None:
        from .risk.premodel_task import validate_premodel_configuration
        validate_premodel_configuration(self.premodel_safety_version, self.safety_delivery_version,
                                       self.safety_response_instruction_version, self.policy.online_decision_version)
        bound_premodel = ctx.working_memory.get("_premodel_safety_version")
        if bound_premodel is not None and bound_premodel != self.premodel_safety_version:
            raise ValueError("session_premodel_safety_version_mismatch")
        if getattr(self.policy, "safety_response_instruction_version", "legacy") != self.safety_response_instruction_version:
            raise ValueError("policy_safety_response_instruction_version_mismatch")
        validate_delivery_version(self.safety_delivery_version)
        if (self.safety_delivery_version != LEGACY
                and self.policy.online_decision_version not in EVIDENCE_DECISION_VERSIONS):
            raise ValueError("conditional_safety_delivery_requires_online_v4")
        if (self.safety_delivery_version == TASK_CONTEXT_V2
                and self.safety_response_instruction_version != "conditional_response_v2"):
            raise ValueError("task_safety_delivery_requires_response_v2")
        if ((self.safety_delivery_version == SOURCE_RESPONSE_V3)
                != (self.safety_response_instruction_version == "conditional_response_v3")):
            raise ValueError("source_response_requires_matching_instruction_v3")
        if ((self.safety_delivery_version == SOURCE_RESPONSE_V4)
                != (self.safety_response_instruction_version == "conditional_response_v4")):
            raise ValueError("source_response_requires_matching_instruction_v4")
        if ((self.safety_delivery_version == MINIMAL_RESPONSE_V1)
                != (self.safety_response_instruction_version == "compact_response_v1")):
            raise ValueError("minimal_response_requires_matching_compact_instruction")
        bound_delivery = ctx.working_memory.get("_safety_delivery_version")
        if bound_delivery is not None and bound_delivery != self.safety_delivery_version:
            raise ValueError("session_safety_delivery_version_mismatch")
        self._safety_response_instruction.bind_context(ctx, getattr(self.policy, "online_decision_version", None))
        ctx.working_memory["_safety_delivery_version"] = self.safety_delivery_version
        ctx.working_memory["_premodel_safety_version"] = self.premodel_safety_version
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
        safety_prelude=None,
    ) -> str:
        from .policy.fictional_context import fictional_conversation_history
        fictional_conversation_history(ctx)
        if ctx.working_memory.get("_premodel_task_inflight") is not None:
            raise ValueError("premodel_task_reentry_forbidden")
        if ctx.working_memory.get("_safety_review_inflight") is not None:
            raise ValueError("safety_review_reentry_forbidden")
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
            self.policy.online_decision_version in SAFETY_DECISION_VERSIONS
        )
        # The observation planner runs before reply generation; give it the
        # same explicit contract, including on the first turn of a session.
        ctx.working_memory["_online_decision_version"] = self.policy.online_decision_version
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
        # Runtime-only routing: not serialized in working_memory/checkpoints,
        # not copied from model data, and reset even on an early-return turn.
        ctx._semantic_delivery_review_v2 = (
            (ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns),
             hashlib.sha256(utterance.encode("utf-8")).hexdigest())
            if self._reviews_clear_body() else None)
        # 这些字段只描述“这一轮”。高风险会在常规感知/推理之前早退，因此若不
        # 先清理，Web 审计视图会把上一轮的情绪、干预、Skill 或反诈判断错误地
        # 展示成急症回合的依据。会话累计的 evidence/outcomes/delivery history
        # 仍然保留，结果学习不会被抹掉。
        for key in (
            "risk_flag", "_safety_task_projection_for_turn", "_source_response_for_turn",
            "safety_delivery", "safety_response_instruction",
            "safety_review",
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
            "_semantic_current_action_receipt",
            "_delivery_response_candidate_for_turn",
            "_benchmark_reply_trace",
            "delivery_response_guard",
            "_reply_delivery_stages",
            "perception_input",
            "perception_fusion",
            "prompt_evidence_snapshot",
            "safety_screening_audit",
            "premodel_task",
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
            from .cognitive import abandon_cognitive_readiness
            abandon_cognitive_readiness(ctx, "keyword_safety_takeover")
            # A keyword backstop protects immediately but does not establish
            # whose symptoms these are. Keep a separate source-only audit.
            self._record_safety_screening(ctx, signal, utterance)
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
            ctx.working_memory["intervention_delivery"] = {
                "status": "not_delivered", "turn_index": len(ctx.turns),
                "intervention_id": None, "reason": "keyword_safety_takeover",
            }
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
            if self.premodel_safety_version in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4"):
                return self._premodel_task_response(ctx, utterance, signal, safety_prelude)
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
        update_alliance(ctx, utterance, allow_surface_inference=not semantic_screening)
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
        # C1 reduction is checked before safety branches; a valid current
        # semantic withdrawal can stop capture even if the safety answer is
        # concerning. The model never gains enable, deletion or share power.
        if semantic_screening and apply_cognitive_withdrawal(ctx, self.persona, utterance):
            if self.store is not None and self.persona is not None:
                self.store.save(self.persona)
        v7_without_optional_evidence = (
            self.policy.online_decision_version == "online_turn_decision_v7"
            and online_trace.get("version") == "online_turn_decision_v7"
            and online_trace.get("status") == "parsed"
            and online_trace.get("turn_index") == len(ctx.turns)
            and online_trace.get("safety_source") == "online_model_semantics"
            and online_trace.get("safety_concern") == "none"
            and online_trace.get("safety_evidence") is None
        )
        model_signal = None if v7_without_optional_evidence else semantic_safety_signal(ctx, utterance)
        delivery_context = capture_delivery_context(
            ctx, utterance, delivery_version=self.safety_delivery_version,
            online_version=self.policy.online_decision_version, persona=self.persona,
        ) if self.safety_delivery_version != LEGACY else None
        semantic_clear = (
            online_trace.get("version") == self.policy.online_decision_version
            and online_trace.get("status") == "parsed"
            and online_trace.get("turn_index") == len(ctx.turns)
            and online_trace.get("safety_source") == "online_model_semantics"
            and online_trace.get("safety_concern") == "none"
            and model_signal is None
            and (online_trace.get("version") not in EVIDENCE_DECISION_VERSIONS
                 or v7_without_optional_evidence
                 or (online_trace.get("safety_assessment") or {}).get("disposition") == "clear")
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
                # A valid v7 owner invitation is a separate transaction from
                # optional safety metadata. Keep this turn's safety rollback
                # and assessment, and use the same source-bound review below;
                # never send the original body without that configured review.
                from .memory.preference_dialogue import current_judged_preference_invitation
                if (self._reviews_clear_body()
                        and online_trace.get("safety_concern") == "none"
                        and current_judged_preference_invitation(ctx, self.persona, utterance) is not None):
                    preference_reply = self._respond_to_preference_invitation(
                        ctx, utterance, raw, learning_before_screening=learning_before_screening,
                        safety_prelude=safety_prelude)
                    if preference_reply is not None:
                        return preference_reply
                # Missing/invalid safety is not a clear result. Deliver only
                # fallback, without scene observers, actions or task flows.
                ctx.working_memory["model_route"]["safety"] = {
                    "source": "online_model_semantics", "status": "unavailable",
                    "response_source": "deterministic_fallback",
                }
                response = DETERMINISTIC_MODEL_FAILURE_REPLY
                if self.policy.online_decision_version in EVIDENCE_DECISION_VERSIONS:
                    from .risk.safety_evidence import UNCERTAIN_REPLY
                    response = UNCERTAIN_REPLY
                if self.safety_delivery_version == TASK_CONTEXT_V2:
                    response = self._compose_safety_task(ctx, None, None)
                if self.safety_delivery_version == MINIMAL_RESPONSE_V1:
                    response = self._compose_source_response(ctx, None, delivery_context,
                                                             safety_prelude=safety_prelude)
                    ctx.working_memory["model_route"]["safety"]["response_source"] = (
                        "source_bound_body" if (ctx.working_memory.get("safety_delivery") or {}).get("status")
                        in {"source_bound_guarded", "guard_replaced_without_crisis_prelude"}
                        else "deterministic_fallback")
                ctx.add_turn(utterance, response)
                self._mark_interaction_started()
                return response
        if model_signal is not None:
            if (online_trace.get("version") not in EVIDENCE_DECISION_VERSIONS
                    or (online_trace.get("safety_assessment") or {}).get("disposition") == "urgent_owner"):
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
            if self.policy.online_decision_version in ("online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7"):
                response = self._handle_risk(ctx, model_signal, delivery_context=delivery_context,
                                             safety_prelude=safety_prelude)
            else:
                response = (self._handle_risk(ctx, model_signal, delivery_context=delivery_context)
                            if delivery_context is not None else self._handle_risk(ctx, model_signal))
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
            preference_reply = self._respond_to_preference_invitation(
                ctx, utterance, raw, learning_before_screening=learning_before_screening,
                safety_prelude=safety_prelude)
            if preference_reply is not None:
                return preference_reply
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
        from .policy.delivery_spans import parse_action_projected
        response, action = parse_action_projected(ctx, raw)
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
                duplicate_is_attempt=self.policy.online_decision_version in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"),
            )
            if prepared is None:
                ctx.working_memory["last_tool_result"] = {
                    "tool": action["tool"],
                    "ok": False,
                    "data": {"reason": denied_reason, "executed": False},
                }
                response = (("本次未提交动作：" if self.policy.online_decision_version in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")
                             else "这个动作还没有执行：") + f"{denied_reason}。")
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
                from .policy.semantic_delivery import bind_current_action_receipt
                bind_current_action_receipt(ctx, utterance, ctx.working_memory["last_tool_result"])
                pending_action_id = str(
                    prepared.get("pending_action_id") or ""
                )
                if result.ok and pending_action_id:
                    consume_pending_action(ctx, pending_action_id)
                if not result.ok:
                    semantic_outcome = self.policy.online_decision_version in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")
                    definitely_not_executed = type(result.data) is dict and result.data.get("executed") is False
                    signatures = ctx.working_memory.setdefault(
                        "_executed_action_signatures", set()
                    )
                    if not semantic_outcome or definitely_not_executed:
                        signatures.discard(json.dumps(
                            {"tool": prepared["tool"], "args": prepared["args"]},
                            ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    if semantic_outcome:
                        from .policy.semantic_delivery import failed_action_status
                        response = failed_action_status(result.data)
                    else:
                        note = str(result.data.get("note") or result.data.get("error")
                                   or "外部能力没有返回成功回执")
                        response = "这个动作还没有执行：" + f"{note.rstrip('。')}。"
                    response_overridden = True

        record_reply_stage(ctx, "tool_result", before_tool, response,
                           violations=["tool_response_replaced"] if before_tool != response else [])
        if before_tool != response and self.policy.online_decision_version in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
            from .policy.delivery_spans import publish_code_response
            publish_code_response(ctx, response)

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
        reviewed = self._review_clear_source_body(ctx, utterance, response, safety_prelude=safety_prelude)
        if reviewed is not None:
            reviewed_response, clear = reviewed
            response_overridden = response_overridden or reviewed_response != response or not clear
            response = reviewed_response
            if not clear:
                if learning_before_screening is not None:
                    ctx.reasoning, ctx.alliance, ctx.formulation = deepcopy(learning_before_screening[:3])
                # Preserve genuine transaction/task state, but do not record
                # crisis or failed-review text as an observation intervention.
                ctx.working_memory["intervention_delivery"] = {
                    "status": "not_delivered", "turn_index": len(ctx.turns),
                    "intervention_id": None, "reason": "body_safety_review"}
                ctx.add_turn(utterance, response)
                self._mark_interaction_started()
                return response

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
        # Source provenance for a later observed-response hypothesis. It does
        # not turn the model's interpretation into a fact. Code receipts,
        # safety takeovers and failures cannot manufacture person evidence.
        from .policy.delivery_spans import model_positions
        from .risk.source_response import current_source_response
        trace = ctx.working_memory.get("online_turn_decision") or {}
        positions = model_positions(ctx, response)
        if (trace.get("status") == "parsed" and trace.get("safety_concern") == "none"
                and not ctx.working_memory.get("risk_flag")
                and current_source_response(ctx, utterance, self.persona) is not None
                and positions and all(origin is not None for origin in positions)
                and all(trace["response_segments"][origin].get("kind") not in {"action", "safety_core"}
                        for origin in set(positions))):
            ctx.working_memory.setdefault("_person_response_deliveries", {})[str(len(ctx.turns))] = hashlib.sha256(response.encode()).hexdigest()
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
            from .memory.preference_status import copy_preference_provenance
            copy_preference_provenance(self.persona, persona)

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
            stage_summary = getattr(self.hypothesis_updater, "stage_audit_summary", None)
            if callable(stage_summary):
                summary = stage_summary()
                if summary is not None:
                    ctx.working_memory["_reflection_itemwise_audit"] = summary
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
            premodel_safety_version=self.premodel_safety_version,
            safety_delivery_version=self.safety_delivery_version,
            safety_response_instruction_version=self.safety_response_instruction_version,
            session_id=ctx.session_id,
            scene_id=ctx.scene_id,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            turns=([SessionTurnSnapshot(user=ctx.turns[i].user, agent=ctx.turns[i].agent,
                        person_response_sha256=(ctx.working_memory.get("_person_response_deliveries") or {}).get(str(i)))
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

    def _handle_risk(self, ctx: EngineContext, signal: RiskSignal, *, delivery_context=None,
                     safety_prelude=None) -> str:
        """tier ≥2 接管:确定性安全回应(热线 12356 / 紧急 120 / 推向真人)。

        不过 LLM——安全底线不交给生成模型抽卡。通知照护人只做建议,
        自动通知需要真实通道 + 当次同意,接入口在此分流。
        """
        if self.safety_delivery_version in (SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1):
            return self._compose_source_response(ctx, signal, delivery_context, safety_prelude=safety_prelude)
        if self.safety_delivery_version == TASK_CONTEXT_V2:
            return self._compose_safety_task(ctx, signal, delivery_context)
        if delivery_context is not None:
            current = capture_delivery_context(
                ctx, ctx.working_memory.get("current_utterance"),
                delivery_version=self.safety_delivery_version,
                online_version=self.policy.online_decision_version, persona=self.persona,
            )
            if (current is not None and current == delivery_context
                    and signal == semantic_safety_signal(
                        ctx, ctx.working_memory.get("current_utterance"))):
                response, segment_ids = render_safety_delivery(current)
                ctx.working_memory["safety_delivery"] = {
                    "version": self.safety_delivery_version, "turn_index": current.turn_index,
                    "user_id": current.user_id, "session_id": current.session_id,
                    "scene_id": current.scene_id, "status": "rendered",
                    "source": current.source, "utterance_sha256": current.utterance_sha256,
                    "quote_sha256": current.quote_sha256, "concern": current.concern,
                    "subject": current.subject, "time": current.time, "urgency": current.urgency,
                    "disposition": current.disposition, "reason": current.reason,
                    "semantic_verified": False, "grants_permission": False,
                    "delivery_only": True, "new_model_understanding_exposure": False,
                    "segment_ids": list(segment_ids),
                }
                return response
        return safety_response(signal, self.persona)

    def pending_safety_prelude(self, ctx, utterance):
        """Prepare the existing immediate backstop without mutating a session.

        Used while an accepted web input waits for the current physical call.
        The transport must rebind the exact event when the turn is admitted;
        this method grants neither a model call nor a delivered capability.
        """
        from .risk.premodel_task import make_prelude, response_only_context
        if self.premodel_safety_version not in ("task_response_v1", "task_response_v2", "task_response_v3", "task_response_v4"):
            return None
        detached, profile = response_only_context(ctx, self.persona)
        self._validate_context_mode(detached)
        detached.working_memory["current_utterance"] = utterance
        detached.working_memory["_online_decision_version"] = self.policy.online_decision_version
        detached.working_memory["_premodel_safety_version"] = self.premodel_safety_version
        for key in ("online_turn_decision", "_source_response_for_turn", "safety_review", "safety_delivery"):
            detached.working_memory.pop(key, None)
        signal = deepcopy(self._risk).assess(detached)
        if signal is None or signal.tier < 2:
            return None
        protection = (self._compose_source_response(detached, signal, None, persona=profile)
                      if self.safety_delivery_version in (SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1)
                      else self._compose_safety_task(detached, signal, None, persona=profile))
        return make_prelude(detached, utterance, protection)

    def _premodel_task_response(self, ctx, utterance, signal, safety_prelude):
        """Preserve immediate protection, then one isolated response-only call."""
        from .policy.fictional_context import fictional_conversation_history
        from .risk.premodel_task import (DeliveredSafetyPrelude, make_prelude,
                                         response_only_context, transfer_delivery_audits)
        from .routing import ReplyModelRoute
        protection = (self._compose_source_response(ctx, signal, None)
                      if self.safety_delivery_version in (SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1)
                      else self._compose_safety_task(ctx, signal, None))
        audit = {"version": self.premodel_safety_version, "turn_index": len(ctx.turns),
                 "session_id": ctx.session_id,
                 "status": "prelude_unavailable", "client_entered": False,
                 "semantic_verified": False, "grants_permission": False}
        ctx.working_memory["premodel_task"] = audit
        delivered = False
        delivered_prelude = None
        original_turns = deepcopy(ctx.turns)
        original_fictional_history = fictional_conversation_history(ctx)
        inflight = object()
        ctx.working_memory["_premodel_task_inflight"] = inflight
        try:
            if callable(safety_prelude):
                try:
                    expected = make_prelude(ctx, utterance, protection)
                    event = dict(expected)
                    audit["protection_sha256"] = expected["protection_sha256"]
                    audit["utterance_sha256"] = expected["utterance_sha256"]
                    delivered = safety_prelude(event) is True
                    self._validate_context_mode(ctx)
                    if (event != expected or make_prelude(ctx, utterance, protection) != expected
                            or ctx.working_memory.get("current_utterance") != utterance
                            or ctx.turns != original_turns
                            or fictional_conversation_history(ctx) != original_fictional_history
                            or ctx.working_memory.get("_premodel_task_inflight") is not inflight):
                        delivered = False
                        audit["status"] = "prelude_identity_changed"
                    if delivered:
                        delivered_prelude = DeliveredSafetyPrelude(**expected)
                except Exception:
                    audit["status"] = "prelude_failed"
                    delivered = False
            response = protection
            if delivered:
                audit["status"] = "prelude_emitted"
                try:
                    detached, profile = response_only_context(ctx, self.persona)
                    scene = deepcopy(self.scene)
                    # Fresh stateless policy; only the transport/fee client is shared.
                    # No normal observers, learning, Skills or action commits run.
                    policy = PolicyEngine(fast_llm=self.policy.fast_llm,
                        online_decision_version=self.policy.online_decision_version,
                        delivery_block=self.policy.delivery_block,
                        safety_response_instruction_version=self.safety_response_instruction_version)
                    returned = policy.respond(detached, scene, utterance, persona=profile,
                                   available_tools=[], model_route=ReplyModelRoute(
                                       "fast_integrated", ("premodel_response_only",)))
                    audit["client_entered"] = (detached.working_memory.get("safety_response_instruction") or {}).get("passed_to_client") is True
                    if self.premodel_safety_version in ("task_response_v2", "task_response_v3", "task_response_v4"):
                        current_signal = semantic_safety_signal(detached, utterance)
                        current_context = capture_delivery_context(
                            detached, utterance, delivery_version=self.safety_delivery_version,
                            online_version=policy.online_decision_version, persona=profile)
                        response = self._compose_source_response(
                            detached, current_signal, current_context, persona=profile,
                            scene=scene, expected_return=returned,
                            delivered_prelude=delivered_prelude)
                    else:
                        response = self._compose_safety_task(detached, signal, None, persona=profile, scene=scene)
                    transfer_delivery_audits(detached, ctx)
                    audit["status"] = "response_only_complete"
                    if (ctx.working_memory.get("model_route") or {}).get("reply", {}).get("source") == "deterministic_fallback":
                        audit["status"] = "response_only_failed"
                except Exception:
                    audit["status"] = "response_only_failed"
                    ctx.working_memory["model_route"]["reply"] = {
                        "tier": "deterministic_fallback", "source": "deterministic_fallback",
                        "reasons": ["premodel_response_only_failed"],
                        "configured_model": None, "effective_model": None}
            if self.premodel_safety_version in ("task_response_v3", "task_response_v4") and delivered and response == protection:
                response = "本次具体回应未能完成；若有现实中的紧急危险，请按先行提示及时求助。"
                audit["duplicate_protection_avoided"] = True
            ctx.add_turn(utterance, response)
            self._mark_interaction_started()
            return response
        finally:
            if ctx.working_memory.get("_premodel_task_inflight") is inflight:
                ctx.working_memory.pop("_premodel_task_inflight", None)

    def _compose_source_response(self, ctx, signal, delivery_context, *, persona=None,
                                 scene=None, expected_return=None, safety_prelude=None,
                                 delivered_prelude=None):
        """Deliver a same-turn complete response through all normal guards.

        V3 also requires an accepted assessment; V4 independently binds the
        received body while preserving assessment rejection and isolation.
        Neither path grants actions, learning or semantic validity. Missing provenance retains
        deterministic protection. Already emitted preludes remain in transport
        evidence; this function never rewrites their saved events.
        """
        from .risk.source_response import current_source_response, digest
        from .policy.delivery_provenance import record_reply_stage
        from .policy.online_decision import _guard_user_response
        profile = self.persona if persona is None else persona
        guard_scene = self.scene if scene is None else scene
        utterance = ctx.working_memory.get("current_utterance")
        current = capture_delivery_context(ctx, utterance,
            delivery_version=self.safety_delivery_version,
            online_version=self.policy.online_decision_version, persona=profile)
        context_bound = (current is not None and current == delivery_context
                         and signal == semantic_safety_signal(ctx, utterance))
        protection, segment_ids = (render_safety_delivery(current) if context_bound
                                   else render_unverified_safety())
        minimal = self.safety_delivery_version == MINIMAL_RESPONSE_V1
        independent_body = self.safety_delivery_version in (SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1)
        if independent_body and not context_bound:
            from .risk.safety_delivery import render_unverified_safety_v2
            protection, segment_ids = render_unverified_safety_v2()
        if minimal and signal is None:
            # Missing source is a generation failure, not evidence of danger.
            protection, segment_ids = DETERMINISTIC_MODEL_FAILURE_REPLY, ()
        frozen = current_source_response(ctx, utterance, profile)
        trace = ctx.working_memory.get("online_turn_decision") or {}
        from .risk.safety_review import current_review_evidence
        review_evidence = current_review_evidence(ctx, utterance, persona=profile,
            include_clear=self._reviews_clear_body()) if minimal else None
        if review_evidence is not None:
            from .risk.safety_delivery import render_review_protection
            protection, segment_ids = (("", ()) if review_evidence["concern"] == "none"
                                       else render_review_protection(review_evidence))
        elif (minimal and self.policy.online_decision_version in ("online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7")
              and signal is not None and frozen is None):
            # The existing keyword backstop must be visible before its isolated
            # model continuation. It is not a model semantic classification.
            # A lexical backstop cannot establish subject, time or reality.
            # Keep immediate protection conditional until the model has read
            # the complete request; do not assert a personal current crisis.
            if self.premodel_safety_version == "task_response_v4":
                from .risk.safety_delivery import render_unverified_safety_v2
                protection, segment_ids = render_unverified_safety_v2()
            else:
                protection, segment_ids = safety_response(signal, profile), ("keyword-safety-existing",)
        assessment_current = context_bound or (
            frozen is not None and trace.get("safety_concern") == "none"
            and signal is None and delivery_context is None
            and semantic_safety_signal(ctx, utterance) is None)
        if independent_body and frozen is not None:
            assessment_current = signal == semantic_safety_signal(ctx, utterance)
        response = protection
        status = "source_unavailable"
        if (frozen is not None and assessment_current
                and (not self._reviews_clear_body() or review_evidence is not None)
                and (expected_return is None or expected_return == frozen.guarded)):
            response_guard = getattr(guard_scene, "guard_response", None)
            delivery_guard = getattr(guard_scene, "guard_delivery_response", None)
            if callable(response_guard) and callable(delivery_guard):
                # Recheck the original candidate, including completion claims
                # that an earlier guard may have removed. No action parser runs.
                if minimal:
                    guarded, guard_audit = frozen.guarded, {"violations": [], "policy": "minimal_response_v1"}
                else:
                    guarded, guard_audit = _guard_user_response(frozen.guarded, ctx, utterance, profile)
                record_reply_stage(ctx, "source_response_online_guard", frozen.guarded, guarded,
                                   violations=guard_audit.get("violations", []))
                before = guarded
                guarded = response_guard(ctx, guarded)
                record_reply_stage(ctx, "source_response_scene_guard", before, guarded,
                    violations=["source_scene_replaced"] if before != guarded else [])
                ctx.working_memory["_delivery_response_candidate_for_turn"] = frozen.raw
                try:
                    before = guarded
                    guarded = delivery_guard(ctx, guarded, utterance, profile)
                    record_reply_stage(ctx, "source_response_delivery_guard", before, guarded,
                        violations=(ctx.working_memory.get("delivery_response_guard") or {}).get("violations", []))
                finally:
                    ctx.working_memory.pop("_delivery_response_candidate_for_turn", None)
                if review_evidence is not None:
                    from .policy.delivery_spans import model_positions
                    body_positions = model_positions(ctx, guarded)
                    body_segments = deepcopy(trace.get("response_segments"))
                    if self.safety_review_version == "safety_review_v2":
                        reviewed, protection, segment_ids = self._review_source_risk_v2(
                            ctx, utterance, guarded, profile=profile, frozen=frozen,
                            evidence=review_evidence, protection=protection, segment_ids=segment_ids,
                            safety_prelude=safety_prelude, delivered_prelude=delivered_prelude)
                    else:
                        reviewed = self._review_source_risk_body(ctx, utterance, guarded,
                            profile=profile, frozen=frozen, evidence=review_evidence,
                            protection=protection, safety_prelude=safety_prelude,
                            delivered_prelude=delivered_prelude)
                    reviewed = self._remove_repeated_safety_core(
                        ctx, guarded, reviewed, protection, body_positions, body_segments)
                    reviewed = self._project_reviewed_ui_state(ctx, utterance, profile, reviewed,
                                                               candidate=guarded, positions=body_positions)
                    remaining_protection = self._undelivered_review_protection(ctx, protection)
                    response = "\n".join(piece for piece in (remaining_protection, reviewed) if piece)
                    if not response:
                        response = "我在。" if "U-safe-contact" in segment_ids else DETERMINISTIC_MODEL_FAILURE_REPLY
                    status = ("risk_reviewed_with_protection" if protection else "risk_reviewed_no_applicable_template") if reviewed else "risk_protection_only"
                    record_reply_stage(ctx, "risk_body_review", guarded, response,
                        violations=["risk_protection_required"])
                elif guarded == frozen.raw:
                    response, status, segment_ids = guarded, "source_bound_guarded", ()
                elif minimal and signal is None:
                    response, status, segment_ids = guarded, "guard_replaced_without_crisis_prelude", ()
                else:
                    # A finite guard rejection cannot certify that essential
                    # safety content survived. Keep the existing safe fallback
                    # plus the truthful guard explanation for this exceptional
                    # path, instead of silently resurrecting the rejected body.
                    response = protection + ("\n" + guarded if guarded else "")
                    status = "guard_replaced_with_protection"
            else:
                status = "required_scene_guard_unavailable"
        trace = ctx.working_memory.get("online_turn_decision") or {}
        ctx.working_memory["safety_delivery"] = {
            "version": self.safety_delivery_version, "turn_index": len(ctx.turns),
            "user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
            "status": status, "source": "online_model_semantics" if frozen else "unverified_backstop",
            "utterance_sha256": digest(utterance) if type(utterance) is str else None,
            "safety_context_status": "source_bound" if context_bound else
                ("model_clear" if frozen and trace.get("safety_concern") == "none"
                 and (trace.get("safety_assessment") or {}).get("disposition") == "clear" else "unverified"),
            "concern": trace.get("safety_concern") if frozen else None,
            "source_response": trace.get("source_response"),
            "response_sha256": digest(response), "segment_ids": list(segment_ids),
            "semantic_verified": False, "grants_permission": False,
            "delivery_only": True, "new_model_understanding_exposure": False,
        }
        return response

    def _respond_to_preference_invitation(self, ctx, utterance, raw, *,
                                          learning_before_screening, safety_prelude):
        """Commit a bound preference, then review only its actual model spans."""
        from .memory.preference_dialogue import apply_preference_dialogue, judge_preference_confirmation
        before_commit = judge_preference_confirmation(ctx, self.persona, utterance, client=self.confirm_judge_llm)
        preference_reply = apply_preference_dialogue(
            ctx, self.persona, utterance, store=self.store, response=raw, before_commit=before_commit)
        if preference_reply is not None:
            from .policy.delivery_provenance import record_reply_stage
            record_reply_stage(ctx, "preference_confirmation_commit", raw, preference_reply)
            if self.policy.online_decision_version in ("online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7"):
                # The retained companion text has no authority to execute
                # actions and must still pass normal truth/safety guards.
                from .policy.delivery_spans import parse_action_projected
                preference_reply, _ignored_action = parse_action_projected(ctx, preference_reply)
                ctx.working_memory["_delivery_response_candidate_for_turn"] = preference_reply
                for guard_name in ("guard_response", "guard_delivery_response"):
                    guard = getattr(self.scene, guard_name, None)
                    if callable(guard):
                        before = preference_reply
                        preference_reply = (guard(ctx, before) if guard_name == "guard_response"
                                            else guard(ctx, before, utterance, self.persona))
                        record_reply_stage(ctx, "preference_" + guard_name, before, preference_reply)
            ctx.working_memory.pop("_delivery_response_candidate_for_turn", None)
            reviewed = self._review_clear_source_body(
                ctx, utterance, preference_reply, safety_prelude=safety_prelude)
            if reviewed is not None:
                preference_reply, clear = reviewed
                if not clear and learning_before_screening is not None:
                    ctx.reasoning, ctx.alliance, ctx.formulation = deepcopy(learning_before_screening[:3])
            ctx.working_memory["intervention_delivery"] = {
                "status": "not_delivered", "turn_index": len(ctx.turns),
                "intervention_id": None, "reason": "owner_preference_confirmation",
            }
            ctx.add_turn(utterance, preference_reply)
            from .memory.preference_recovery import complete_unclear_turn
            complete_unclear_turn(self.persona, ctx)
            self._mark_interaction_started()
            return preference_reply
        return None

    def _reviews_clear_body(self):
        return (self.policy.online_decision_version == "online_turn_decision_v7"
                and self.safety_review_version == "safety_review_v2"
                and self.safety_delivery_version == MINIMAL_RESPONSE_V1
                and getattr(self.safety_review_llm, "role", None) == "safety_review")

    def _review_clear_source_body(self, ctx, utterance, candidate, *, safety_prelude):
        """Review final bound model spans without editing genuine code receipts.

        Normal permission and transaction guards retain their authority. This
        delivery-only check neither executes actions nor grants permissions.
        Unconfigured previews and historical versions retain their old path.
        """
        if not self._reviews_clear_body():
            return None
        from .risk.source_response import current_source_response
        from .risk.safety_review import current_review_evidence
        from .policy.delivery_spans import model_positions, publish_positions
        from .policy.delivery_provenance import record_reply_stage
        trace = ctx.working_memory.get("online_turn_decision") or {}
        if trace.get("status") != "parsed" or trace.get("safety_concern") != "none":
            ctx.working_memory["safety_review"] = {
                "version": "safety_review_v2", "turn_index": len(ctx.turns),
                "status": "source_unavailable", "passed_to_client": False,
                "actual_send_verified": False, "semantic_verified": False,
                "grants_permission": False, "model_body_delivered": False}
            return DETERMINISTIC_MODEL_FAILURE_REPLY, False
        evidence = current_review_evidence(ctx, utterance, persona=self.persona, include_clear=True)
        frozen = current_source_response(ctx, utterance, self.persona)
        positions = model_positions(ctx, candidate)
        current_trace = (trace.get("turn_index") == len(ctx.turns)
                         and trace.get("owner_message_sha256") == hashlib.sha256(utterance.encode()).hexdigest())
        if positions is not None and current_trace and all(origin is None for origin in positions):
            return None
        if positions is None or evidence is None or frozen is None:
            ctx.working_memory["safety_review"] = {
                "version": "safety_review_v2", "turn_index": len(ctx.turns),
                "status": "source_unavailable", "passed_to_client": False,
                "actual_send_verified": False, "semantic_verified": False,
                "grants_permission": False, "model_body_delivered": False}
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
            code = ("".join(char for char, origin in zip(candidate, positions) if origin is None)
                    if positions is not None and current_trace else "")
            response = DETERMINISTIC_MODEL_FAILURE_REPLY + ("\n" + code if code else "")
            publish_positions(ctx, response, [None] * len(response))
            record_reply_stage(ctx, "clear_body_review", candidate, response,
                               violations=["body_safety_review_source_unavailable"])
            return response, False
        model_offsets = [index for index, origin in enumerate(positions) if origin is not None]
        model_body = "".join(candidate[index] for index in model_offsets)
        if not model_body.strip():
            return None
        body, protection, segment_ids = self._review_source_risk_v2(
            ctx, utterance, model_body, profile=self.persona, frozen=frozen,
            evidence=evidence, protection="", segment_ids=(),
            safety_prelude=safety_prelude, delivered_prelude=None)
        review = ctx.working_memory.get("safety_review") or {}
        clear = (review.get("status") == "reviewed" and bool(body)
                 and (review.get("risk_assessment") or {}).get("concern") == "none")
        kept_offsets = set()
        if body and review.get("status") == "reviewed":
            offsets = self._retained_review_offsets(model_body, body, review)
            if offsets is None:
                review.update(status="source_changed", model_body_delivered=False)
                body, clear = "", False
            else:
                kept_offsets = {model_offsets[index] for index in offsets}
        selected_offsets = [index for index, origin in enumerate(positions)
                            if origin is None or index in kept_offsets]
        remainder = "".join(candidate[index] for index in selected_offsets)
        prefix = self._undelivered_review_protection(ctx, protection)
        if not prefix and not body:
            prefix = "我在。" if "U-safe-contact" in segment_ids else DETERMINISTIC_MODEL_FAILURE_REPLY
        separator = "\n" if prefix and remainder else ""
        response = prefix + separator + remainder
        publish_positions(ctx, response, [None] * (len(prefix) + len(separator))
                          + [positions[index] for index in selected_offsets])
        # This route already has a source-bound, parsed online "none". A failed
        # body review withholds the generated reply, not the owner's original
        # input. Preserve prior exclusions; a changed source or an actual new
        # risk/uncertainty assessment can still exclude this turn from learning.
        assessment = review.get("risk_assessment") or {}
        if review.get("status") == "reviewed":
            # The review can advise conditional protection yet explicitly find
            # no owner signal. On this bound online-none path that distinction
            # preserves the owner's words. Missing/invalid references do not.
            protection_only = assessment.get("concern") == "uncertain" and review.get("owner_source_turns") == []
            input_excluded = assessment.get("concern") != "none" and not protection_only
        elif review.get("status") in {"review_failed", "unavailable"}:
            input_excluded = bool(assessment) and assessment.get("concern") != "none"
        else:
            input_excluded = True
        if input_excluded:
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
        review["code_receipts_excluded_from_review"] = any(origin is None for origin in positions)
        record_reply_stage(ctx, "clear_body_review", candidate, response,
                           violations=[] if clear else ["body_safety_review"])
        return self._project_reviewed_ui_state(ctx, utterance, self.persona, response), clear

    @staticmethod
    def _retained_review_offsets(candidate, body, review):
        """Exact source positions of the reviewer's retained and edge-stripped text."""
        from .risk.safety_review_v2 import response_sentences
        retained = set(review.get("retained_sentence_ids", []))
        offsets, offset = [], 0
        for index, sentence in enumerate(response_sentences(candidate)):
            if index in retained:
                offsets.extend(range(offset, offset + len(sentence)))
            offset += len(sentence)
        text = "".join(candidate[i] for i in offsets)
        start = len(text) - len(text.lstrip())
        return offsets[start:start + len(body)] if text.strip() == body else None

    def _project_reviewed_ui_state(self, ctx, utterance, profile, body, *, candidate=None, positions=None):
        """Use the same transaction evidence after either route's safety review."""
        from .memory.preference_status import preceding_failed_ui_state
        from .policy.delivery_spans import model_positions, publish_positions
        from .policy.delivery_provenance import record_reply_stage
        from .policy.semantic_delivery import guard_semantic_delivery
        review = ctx.working_memory.get("safety_review") or {}
        trace = ctx.working_memory.get("online_turn_decision") or {}
        if (not body or review.get("status") != "reviewed" or trace.get("version") != "online_turn_decision_v7"
                or not any(row["kind"] == "preference_state" for row in trace.get("response_segments") or [])):
            return body
        failed_state = preceding_failed_ui_state(profile, ctx, utterance)
        if not failed_state:
            return body
        if model_positions(ctx, body) is None and candidate is not None and positions is not None:
            offsets = self._retained_review_offsets(candidate, body, review)
            if offsets is not None:
                publish_positions(ctx, body, [positions[i] for i in offsets])
        projected = guard_semantic_delivery(ctx, body, utterance, profile, privacy_predicate=None,
                                            failed_state=failed_state)
        if projected != body:
            record_reply_stage(ctx, "preference_ui_failure", body, projected,
                               violations=["preference_status_from_failed_ui"])
            review["model_body_delivered"] = any(
                origin is not None for origin in model_positions(ctx, projected) or [])
        return projected

    @staticmethod
    def _undelivered_review_protection(ctx, protection):
        """Remove only the code-owned prefix acknowledged by this transport."""
        review = ctx.working_memory.get("safety_review") or {}
        count = review.get("delivered_protection_prefix_length")
        if (review.get("turn_index") == len(ctx.turns)
                and review.get("final_protection_sha256") == hashlib.sha256(protection.encode()).hexdigest()
                and type(count) is int and 0 < count <= len(protection)
                and review.get("prelude_status") in {"already_delivered", "delivered_after_review"}):
            return protection[count:].lstrip("\n")
        return protection

    @staticmethod
    def _remove_repeated_safety_core(ctx, candidate, reviewed, protection, positions, segments):
        """Delete only exact repeated sentences from bound marked segments."""
        from .risk.source_response import digest
        from .risk.safety_review_v2 import response_sentences

        trace = ctx.working_memory.get("online_turn_decision") or {}
        if trace.get("version") != "online_turn_decision_v7" or not protection or not reviewed:
            return reviewed
        audit = {"status": "source_unavailable", "removed_segment_indexes": [],
                 "semantic_verified": False, "grants_permission": False}
        review = ctx.working_memory.get("safety_review") or {}
        review["safety_core_deduplication"] = audit
        if (type(segments) is not list or segments != trace.get("response_segments")
                or positions is None or len(positions) != len(candidate)
                or review.get("status") != "reviewed"
                or review.get("turn_index") != len(ctx.turns)
                or review.get("candidate_sha256") != digest(candidate)
                or review.get("retained_body_sha256") != digest(reviewed)
                or review.get("prelude_status") not in {"already_delivered", "delivered", "delivered_after_review"}
                or review.get("final_protection_sha256") != digest(protection)
                or (review.get("prelude_status") != "delivered_after_review"
                    and review.get("protection_sha256") != digest(protection))):
            return reviewed
        marked = {index for index, row in enumerate(segments) if row.get("kind") == "safety_core"}
        if not marked:
            audit["status"] = "no_marked_segments"
            return reviewed
        # Review units may be comma clauses, but exact deduplication must not
        # delete a shared condition and leave a different consequent behind.
        # First recover the reviewer's exact retained offsets, then only remove
        # an entire original sentence which survived review in full.
        sentences = response_sentences(candidate)
        retained = review.get("retained_sentence_ids")
        if (type(retained) is not list
                or any(type(index) is not int or not 0 <= index < len(sentences) for index in retained)
                or retained != sorted(set(retained))):
            return reviewed
        retained_set = set(retained)
        retained_offsets, offset = set(), 0
        for index, unit in enumerate(sentences):
            if index in retained_set:
                retained_offsets.update(range(offset, offset + len(unit)))
            offset += len(unit)
        retained_text = "".join(candidate[index] for index in sorted(retained_offsets))
        if retained_text.strip() != reviewed:
            return reviewed
        from .risk.safety_review import response_sentences as complete_sentences
        covered = {sentence.strip() for sentence in complete_sentences(protection)}
        removed_offsets, offset = set(), 0
        removed = set()
        for sentence in complete_sentences(candidate):
            sentence_offsets = set(range(offset, offset + len(sentence)))
            if sentence_offsets.issubset(retained_offsets):
                origins = positions[offset:offset + len(sentence)]
                source_indexes = {origin for char, origin in zip(sentence, origins) if not char.isspace()}
                if (sentence.strip() in covered and source_indexes
                        and source_indexes.issubset(marked)):
                    removed.update(source_indexes)
                    removed_offsets.update(sentence_offsets)
            offset += len(sentence)
        kept = sorted(retained_offsets - removed_offsets)
        remaining = "".join(candidate[index] for index in kept)
        result = remaining.strip()
        if removed:
            from .policy.delivery_spans import publish_positions
            start = len(remaining) - len(remaining.lstrip())
            publish_positions(ctx, result, [positions[i] for i in kept[start:start + len(result)]])
        audit.update(status="removed" if removed else "no_proven_duplicates",
                     removed_segment_indexes=sorted(removed), delivered_body_sha256=digest(result))
        review["model_body_delivered"] = bool(result)
        return result

    def _review_source_risk_v2(self, ctx, utterance, candidate, *, profile, frozen,
                               evidence, protection, segment_ids, safety_prelude,
                               delivered_prelude):
        """Verify model-only applicability before displaying a category template.

        Keyword protection is immutable. Only this risk-only request may select
        a different template; no receipt, action, memory or permission is added.
        Waiting includes the product's pacing delay as well as model timeout.
        """
        from .risk.source_response import current_source_response
        from .risk.safety_review import current_review_evidence
        from .risk.safety_review_v2 import SAFETY_REVIEW_VERSION, review_safety_body
        from .risk.safety_delivery import render_review_protection
        from .risk.premodel_task import current_delivered_prelude, make_prelude
        fallback = ("", protection, segment_ids)
        previous = ctx.working_memory.get("safety_review")
        if type(previous) is dict and previous.get("turn_index") == len(ctx.turns):
            return fallback
        audit = {"version": SAFETY_REVIEW_VERSION, "turn_index": len(ctx.turns),
                 "status": "unavailable", "passed_to_client": False,
                 "actual_send_verified": False, "semantic_verified": False,
                 "grants_permission": False, "model_body_delivered": False}
        ctx.working_memory["safety_review"] = audit
        original_turns = deepcopy(ctx.turns)
        inflight = object()
        ctx.working_memory["_safety_review_inflight"] = inflight
        try:
            actual = current_delivered_prelude(ctx, utterance, delivered_prelude)
            if delivered_prelude is not None and actual is None:
                audit.update(status="prelude_invalid", prelude_status="source_changed")
                return fallback
            actual_text = actual["protection"] if actual is not None else None
            event_hash = (hashlib.sha256(json.dumps(actual, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest() if actual is not None else None)
            # The already-delivered conditional backstop remains valid while
            # the reviewer judges applicability and the complete body. Different
            # template wording is not evidence that the body must be discarded.
            body, result = review_safety_body(ctx, utterance, candidate,
                client=self.safety_review_llm, protection=actual_text, final_protection=protection,
                evidence=evidence, persona=profile)
            result.update(prelude_status="already_delivered" if actual is not None else "not_yet_delivered",
                          prelude_event_sha256=event_hash, model_body_delivered=False)
            ctx.working_memory["safety_review"] = result
            if (ctx.turns != original_turns or current_source_response(ctx, utterance, profile) != frozen
                    or current_review_evidence(ctx, utterance, persona=profile, include_clear=True) != evidence
                    or ctx.working_memory.get("_safety_review_inflight") is not inflight
                    or actual is not None and current_delivered_prelude(ctx, utterance, delivered_prelude) != actual):
                result["status"] = "source_changed"
                return fallback
            selected, selected_ids = protection, segment_ids
            if result["status"] in {"reviewed", "review_failed"} and "risk_assessment" in result:
                assessment = result["risk_assessment"]
                concern = assessment["concern"]
                if concern == "none" or (concern == "other_danger" and body):
                    selected, selected_ids = (actual_text, ("keyword-safety-existing",)) if actual_text else ("", ())
                elif concern in {"uncertain", "other_danger"}:
                    from .risk.safety_delivery import render_unverified_safety_v2
                    safe_contact = concern == "other_danger" or result.get("risk_source_unbound", False)
                    selected, selected_ids = ((actual_text, ("keyword-safety-existing",))
                                              if actual_text else render_unverified_safety_v2(safe_contact=safe_contact))
                else:
                    selected, selected_ids = render_review_protection(assessment, reviewed_fields=True)
                    if actual_text and selected != actual_text:
                        selected = actual_text + "\n" + selected
                        selected_ids = ("keyword-safety-existing", *selected_ids)
            # An independently valid risk assessment still requires protection
            # even when deletion validation failed and no body can be released.
            if not body and "risk_assessment" not in result:
                selected, selected_ids = protection, segment_ids
            if selected and actual is None and callable(safety_prelude):
                try:
                    expected = make_prelude(ctx, utterance, selected)
                    event = deepcopy(expected)
                    acknowledged = safety_prelude(event) is True
                    self._validate_context_mode(ctx)
                    if (not acknowledged or event != expected or make_prelude(ctx, utterance, selected) != expected
                            or ctx.turns != original_turns or current_source_response(ctx, utterance, profile) != frozen
                            or current_review_evidence(ctx, utterance, persona=profile, include_clear=True) != evidence
                            or ctx.working_memory.get("_safety_review_inflight") is not inflight):
                        result.update(status="prelude_failed", prelude_status="failed")
                        body = ""
                    else:
                        result["prelude_status"] = "delivered_after_review"
                        result["delivered_protection_prefix_length"] = len(selected)
                        result["prelude_event_sha256"] = hashlib.sha256(json.dumps(expected, sort_keys=True,
                            ensure_ascii=False).encode()).hexdigest()
                except Exception:
                    result.update(status="prelude_failed", prelude_status="failed")
                    body = ""
            if actual_text and (selected == actual_text or selected.startswith(actual_text + "\n")):
                result["delivered_protection_prefix_length"] = len(actual_text)
            result.update(final_protection_sha256=hashlib.sha256(selected.encode()).hexdigest() if selected else None,
                          model_body_delivered=bool(body))
            return body, selected, selected_ids
        finally:
            if ctx.working_memory.get("_safety_review_inflight") is inflight:
                ctx.working_memory.pop("_safety_review_inflight", None)

    def _review_source_risk_body(self, ctx, utterance, candidate, *, profile, frozen,
                                evidence, protection, safety_prelude,
                                delivered_prelude):
        """Emit protection before the sole risk-only request, then retain a subset."""
        from .risk.source_response import current_source_response
        from .risk.safety_review import (SAFETY_REVIEW_VERSION, current_review_evidence,
                                         review_safety_body)
        from .risk.premodel_task import (DeliveredSafetyPrelude, current_delivered_prelude,
                                         make_prelude)
        previous = ctx.working_memory.get("safety_review")
        if type(previous) is dict and previous.get("turn_index") == len(ctx.turns):
            # No second attempt if this internal composer is entered again.
            return ""
        audit = {"version": SAFETY_REVIEW_VERSION, "turn_index": len(ctx.turns),
                 "status": "prelude_unavailable", "passed_to_client": False,
                 "actual_send_verified": False, "semantic_verified": False,
                 "grants_permission": False}
        ctx.working_memory["safety_review"] = audit
        original_turns = deepcopy(ctx.turns)
        inflight = object()
        ctx.working_memory["_safety_review_inflight"] = inflight
        try:
            receipt = delivered_prelude
            actual = current_delivered_prelude(ctx, utterance, receipt)
            delivered = actual is not None
            audit["prelude_status"] = "already_delivered" if delivered else "unavailable"
            if receipt is not None and not delivered:
                audit.update(status="prelude_invalid", prelude_status="source_changed")
                return ""
            if not delivered and callable(safety_prelude):
                try:
                    expected = make_prelude(ctx, utterance, protection)
                    event = deepcopy(expected)
                    delivered = safety_prelude(event) is True
                    self._validate_context_mode(ctx)
                    delivered = bool(delivered and event == expected
                        and make_prelude(ctx, utterance, protection) == expected)
                    if delivered:
                        receipt = DeliveredSafetyPrelude(**expected)
                        actual = current_delivered_prelude(ctx, utterance, receipt)
                        delivered = actual is not None
                    audit["prelude_status"] = "delivered" if delivered else "failed"
                except Exception:
                    delivered = False
                    audit["prelude_status"] = "failed"
            if (ctx.turns != original_turns
                    or current_source_response(ctx, utterance, profile) != frozen
                    or current_review_evidence(ctx, utterance, persona=profile) != evidence
                    or ctx.working_memory.get("_safety_review_inflight") is not inflight):
                delivered = False
                audit.update(status="source_changed", prelude_status="source_changed")
            if not delivered:
                return ""
            # A keyword prelude can differ from the protection warranted by a
            # later model label. Pass exactly what the callback displayed;
            # planned final protection is a separate, explicitly future field.
            actual_text = actual["protection"]
            event_hash = hashlib.sha256(
                json.dumps(actual, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if actual_text != protection:
                # This UI admits one acknowledged prelude per turn. Do not
                # delay newly necessary protection behind another model call,
                # and do not release the unreviewed body. The caller returns
                # the new protection immediately as the final response.
                audit.update(status="additional_protection_required",
                    reason="new_protection_not_yet_delivered",
                    protection_sha256=actual["protection_sha256"],
                    final_protection_sha256=hashlib.sha256(protection.encode()).hexdigest(),
                    prelude_event_sha256=event_hash, review_call_count=0,
                    model_body_delivered=False)
                return ""
            body, result = review_safety_body(ctx, utterance, candidate,
                client=self.safety_review_llm, protection=actual_text,
                final_protection=protection, evidence=evidence, persona=profile)
            result["prelude_status"] = audit["prelude_status"]
            result["prelude_event_sha256"] = event_hash
            if current_delivered_prelude(ctx, utterance, receipt) != actual:
                result["status"] = "source_changed"
                body = ""
            ctx.working_memory["safety_review"] = result
            return body
        finally:
            if ctx.working_memory.get("_safety_review_inflight") is inflight:
                ctx.working_memory.pop("_safety_review_inflight", None)

    def _compose_safety_task(self, ctx, signal, delivery_context, *, persona=None, scene=None):
        """Compose independently bounded task content; never execute its plan.

        Same-call model labels and exact quotes are not semantic proof. The
        existing scene guards are finite backstops, not a completeness judge.
        """
        from .risk.safety_task import current_task_projection, digest
        from .policy.delivery_provenance import record_reply_stage
        profile = self.persona if persona is None else persona
        guard_scene = self.scene if scene is None else scene
        utterance = ctx.working_memory.get("current_utterance")
        current = capture_delivery_context(ctx, utterance,
            delivery_version=self.safety_delivery_version,
            online_version=self.policy.online_decision_version, persona=profile)
        verified_context = (current is not None and current == delivery_context
                            and signal == semantic_safety_signal(ctx, utterance))
        if verified_context:
            protection, segment_ids = render_safety_delivery(current)
        else:
            protection, segment_ids = render_unverified_safety()
        task = current_task_projection(ctx, utterance, profile)
        task_status, task_text = "missing_or_rejected", ""
        if task is not None:
            from .policy.online_decision import _guard_user_response
            # Only the projected user-visible slice enters the existing guards.
            # Do not restore the original body or use communication_plan text.
            guarded, audit = _guard_user_response(task.text, ctx, utterance, profile)
            record_reply_stage(ctx, "safety_task_online_guard", task.text, guarded,
                               violations=audit.get("violations", []))
            response_guard = getattr(guard_scene, "guard_response", None)
            delivery_guard = getattr(guard_scene, "guard_delivery_response", None)
            if callable(response_guard) and callable(delivery_guard):
                before = guarded
                guarded = response_guard(ctx, guarded)
                record_reply_stage(ctx, "safety_task_scene_guard", before, guarded,
                                   violations=["task_scene_replaced"] if before != guarded else [])
                ctx.working_memory["_delivery_response_candidate_for_turn"] = task.text
                try:
                    before = guarded
                    guarded = delivery_guard(ctx, guarded, utterance, profile)
                    record_reply_stage(ctx, "safety_task_delivery_guard", before, guarded,
                        violations=(ctx.working_memory.get("delivery_response_guard") or {}).get("violations", []))
                finally:
                    ctx.working_memory.pop("_delivery_response_candidate_for_turn", None)
                if guarded == task.text:
                    task_status, task_text = "included_unverified_semantics", task.text
                else:
                    task_status = "rejected_by_existing_guard"
            else:
                task_status = "required_scene_guard_unavailable"
        response = protection + ("\n" + task_text if task_text else "")
        trace = ctx.working_memory.get("online_turn_decision") or {}
        ctx.working_memory["safety_delivery"] = {
            "version": TASK_CONTEXT_V2, "turn_index": len(ctx.turns),
            "user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
            "status": "composed", "source": "online_model_semantics" if trace.get("status") == "parsed" else "unverified_backstop",
            "utterance_sha256": digest(utterance) if type(utterance) is str else None,
            "safety_context_status": "source_bound" if verified_context else "unverified",
            "concern": current.concern if verified_context else None,
            "subject": current.subject if verified_context else "unverified",
            "time": current.time if verified_context else "unverified",
            "urgency": current.urgency if verified_context else "unverified",
            "task_status": task_status, "task_projection": trace.get("task_projection"),
            "task_response_sha256": digest(task_text) if task_text else None,
            "response_sha256": digest(response), "segment_ids": list(segment_ids),
            "semantic_verified": False, "grants_permission": False,
            "delivery_only": True, "new_model_understanding_exposure": False,
        }
        return response

    def _record_risk(self, ctx: EngineContext, signal: RiskSignal) -> None:
        """历史/模型分流事件落账；不据此断言主体或自动推导纵向变化。"""
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

    def _record_safety_screening(self, ctx: EngineContext, signal: RiskSignal,
                                 utterance: str) -> None:
        """Record provenance without classifying an unverified subject."""
        if self.persona is None:
            return
        identity = (ctx.session_id, len(ctx.turns), signal.kind)
        if any((row.session_id, row.turn_index, row.kind) == identity
               for row in self.persona.safety_screenings):
            return
        try:
            row = SafetyScreeningRecord(
                at=datetime.now().isoformat(timespec="seconds"),
                session_id=ctx.session_id, turn_index=len(ctx.turns),
                tier=signal.tier, kind=signal.kind,
                evidence_sha256=hashlib.sha256(utterance.encode("utf-8", errors="surrogatepass")).hexdigest(),
            )
        except (ValueError, TypeError, UnicodeError):
            # Audit identity failure may never delay an urgent response.
            ctx.working_memory["safety_screening_audit"] = {
                "status": "not_recorded", "reason": "invalid_source_identity",
            }
            return
        self.persona.safety_screenings.append(row)
        ctx.working_memory["safety_screening_audit"] = {
            "status": "recorded", "subject": "unverified",
            "source": "keyword_backstop",
        }
