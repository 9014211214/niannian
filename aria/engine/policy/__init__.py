"""策略引擎 —— persona × 情绪 + 知识 + 可用工具 → 回应。

在线只装配一个快速 LLM：普通回合做短表达，复杂回合用同一次 Function Call
产出当轮候选、沟通方向和回复。失败时走确定性兜底，不升级到会后深度模型。
「说什么」由 LLM 出;「怎么说 / 带什么上下文」由 prompt 组装决定(护城河在这)。
真实回应优先使用 OPENROUTER_API_KEY；也兼容 ARIA_LLM_API_KEY 指向本地端点。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ...interfaces.scene import Scene
from ..context import EngineContext
from ..model.persona import PersonaProfile
from ..routing import ReplyModelRoute, client_route_info
from .llm import LLMClient, MockLLMClient
from .online_decision import (
    DETERMINISTIC_MODEL_FAILURE_REPLY,
    ONLINE_DECISION_VERSIONS,
    EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE,
    ONLINE_DECISION_VERSION_SAFETY, SAFETY_DECISION_VERSIONS,
    apply_single_pass_decision,
    configured_online_decision_version,
    is_internal_protocol_response,
    record_online_decision_failure,
    single_pass_messages,
)
from .actions import (
    current_pending_action,
    has_current_action_intent,
    is_pending_confirmation,
)
from .prompt import build_messages, configured_delivery_block, select_delivery_mode
from .release_prompt import CompactPromptBudgetExceeded

__all__ = ["PolicyEngine", "build_messages", "select_delivery_mode"]

_ACTION_TAG_RE = re.compile(
    r"<action>\s*\{.*?\}\s*</action>", re.DOTALL
)


class PolicyEngine:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        fast_llm: Optional[LLMClient] = None,
        online_decision_version: Optional[str] = None,
        delivery_block: Optional[str] = None,
        safety_response_instruction_version: str = "legacy",
    ):
        # ``llm`` 仅作旧注入名；在线策略永远只有这一条 fast client。
        selected = fast_llm or llm or MockLLMClient(role="online_fast")
        self.fast_llm: LLMClient = selected
        self.llm: LLMClient = selected
        # The minimal real client declares v3; explicitly frozen experiments
        # and legacy clients retain their selected v1/v2 contract.
        version = (online_decision_version or getattr(selected, "online_decision_version", None)
                   or configured_online_decision_version())
        if version not in ONLINE_DECISION_VERSIONS:
            raise ValueError("unknown_online_decision_version")
        self.online_decision_version = version
        self.delivery_block = configured_delivery_block(delivery_block)
        from .safety_instruction import SafetyResponseInstruction
        self._safety_response_instruction = SafetyResponseInstruction(safety_response_instruction_version)
        self._safety_response_instruction.validate(version)

    @property
    def safety_response_instruction_version(self):
        return self._safety_response_instruction.version

    def respond(self, ctx, scene, utterance, persona=None, available_tools=None, model_route=None):
        guard = getattr(self.fast_llm, "model_request_control", None)
        admission = getattr(guard, "online_context_admission", None)
        if callable(admission):
            with admission(self.fast_llm, ctx, persona, utterance):
                return self._respond_admitted(ctx, scene, utterance, persona, available_tools, model_route)
        return self._respond_admitted(ctx, scene, utterance, persona, available_tools, model_route)

    def _respond_admitted(
        self,
        ctx: EngineContext,
        scene: Scene,
        utterance: str,
        persona: Optional[PersonaProfile] = None,
        available_tools: Optional[list[str]] = None,
        model_route: ReplyModelRoute | None = None,
    ) -> str:
        instruction = self._safety_response_instruction
        instruction.bind_context(ctx, self.online_decision_version)
        ctx.working_memory.pop("safety_response_instruction", None)
        # Select the current contract before building the base prompt.
        # Clear stale v3 semantics when an explicit historical policy is used.
        ctx.working_memory["_online_decision_version"] = self.online_decision_version
        from ..memory.preference_candidate import bind_context as bind_preference_candidate
        bind_preference_candidate(self.fast_llm, ctx)
        from ..memory.preference_recovery import bind_context as bind_preference_recovery
        bind_preference_recovery(self.fast_llm, ctx)
        from ..memory.preference_dialogue import current_preference_invitation
        evidence_candidate = self.online_decision_version in EVIDENCE_DECISION_VERSIONS
        preference_invitation = (current_preference_invitation(persona, ctx)
                                 if evidence_candidate else None)
        from ..memory.preference_dialogue import bind_preference_request
        bind_preference_request(persona, ctx, utterance)
        pending_action = current_pending_action(ctx)
        advertised_tools = (
            list(available_tools or [])
            if has_current_action_intent(utterance, pending_action) else []
        )
        if is_pending_confirmation(utterance, pending_action):
            pending_tool = str(pending_action.get("tool") or "")
            advertised_tools = [
                item for item in advertised_tools
                if item.split(":", 1)[0] == pending_tool
            ]
        ctx.working_memory["_advertised_tools_for_turn"] = list(
            advertised_tools
        )
        messages = build_messages(
            scene, persona, ctx, utterance, advertised_tools,
            delivery_block=self.delivery_block,
            **({"safety_response_instruction_version": instruction.version}
               if instruction.version != "legacy" else {}),
        )
        if is_pending_confirmation(utterance, pending_action) and messages:
            pending_payload = {
                "id": pending_action["id"],
                "tool": pending_action["tool"],
                "args": pending_action["args"],
            }
            messages[0]["content"] = (
                str(messages[0].get("content") or "")
                + "\n\n## 服务端待确认动作（不可修改）\n"
                + json.dumps(
                    pending_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n本轮若确认执行，只能原样输出上述 tool/args，并在 action "
                "对象顶层带 pending_action_id；不得补写、改写或另选动作。"
            )
        route = model_route or ReplyModelRoute(
            "fast_integrated", ("routing_not_provided",)
        )
        selected = self.fast_llm
        if self.online_decision_version in SAFETY_DECISION_VERSIONS:
            # Semantic safety cannot depend on the lexical fast/complex router.
            # This changes the shape of the same call, never adds a classifier call.
            route = ReplyModelRoute("fast_integrated", (*route.reasons, "same_call_semantic_safety"))
        single_pass = route.integrated_decision
        if single_pass:
            messages = single_pass_messages(
                messages, ctx,
                version=self.online_decision_version, persona=persona,
                delivery_block=self.delivery_block,
                format_style=("schema_guided" if getattr(selected, "online_return_format", None)
                              in {"json_object", "json_schema_v1"} else "legacy"),
            )
        else:
            messages = _fast_delivery_messages(messages, ctx)
        instruction.validate_messages(messages, self.online_decision_version)
        if instruction.version != "legacy":
            from .safety_instruction import SCENE_CLARIFICATION_SHA256
            ctx.working_memory["safety_response_instruction"] = {
                "version": instruction.version, "system_block_sha256": instruction.block_sha256,
                "scene_clarification_sha256": (None if instruction.version == "compact_response_v1"
                                               else SCENE_CLARIFICATION_SHA256),
                "user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
                "turn_index": len(ctx.turns), "online_version": self.online_decision_version,
                "passed_to_client": False, "actual_send_verified": False,
                "semantic_verified": False, "grants_permission": False,
            }
        deterministic_fallback = False
        readiness_dispatch = None
        readiness_outcome = "abandoned"
        try:
            from ..cognitive import begin_cognitive_readiness_dispatch
            readiness_dispatch = begin_cognitive_readiness_dispatch(persona, ctx, utterance)
            snapshot = ctx.working_memory.get("prompt_evidence_snapshot")
            if isinstance(snapshot, dict):
                # This is the local client boundary, not a provider receipt.
                snapshot["passed_to_client"] = True
            if (evidence_candidate
                    and current_preference_invitation(persona, ctx) != preference_invitation):
                raise ValueError("preference_invitation_changed_before_dispatch")
            if self._safety_response_instruction is not instruction:
                raise ValueError("safety_response_instruction_changed_before_dispatch")
            instruction.validate_messages(messages, self.online_decision_version)
            if instruction.version != "legacy":
                ctx.working_memory["safety_response_instruction"]["passed_to_client"] = True
            # A request option, not a client flag or a user-text instruction.
            # A client lacking this explicit capability fails closed.
            requirements = {}
            if preference_invitation is not None:
                requirements["require_preference_control"] = True
            if instruction.version == "conditional_response_v2":
                requirements["require_task_projection"] = True
            guard = getattr(selected, "model_request_control", None)
            bind_context = getattr(guard, "bind_online_context", None)
            if callable(bind_context):
                bind_context(selected, ctx, persona, utterance, messages, requirements, preference_invitation)
            response = selected.complete(messages, **requirements)
            readiness_outcome = "client_returned"
        except CompactPromptBudgetExceeded:
            # This typed local failure precedes HTTP Request construction.
            # Preserve it for the product's no-attempt check; do not turn it
            # into a model/transport failure and lose the distinction.
            readiness_outcome = "local_request_rejected"
            raise
        except Exception:
            readiness_outcome = "client_error"
            response = DETERMINISTIC_MODEL_FAILURE_REPLY
            deterministic_fallback = True
            single_pass = False
            route = ReplyModelRoute(
                "deterministic_fallback",
                (*route.reasons, "fast_model_failed"),
            )
        finally:
            from ..cognitive import finish_cognitive_readiness_dispatch
            finish_cognitive_readiness_dispatch(ctx, readiness_dispatch, readiness_outcome)
        if deterministic_fallback and (
            self.online_decision_version in SAFETY_DECISION_VERSIONS
            or (ctx.working_memory.get("model_route") or {})
            .get("state", {}).get("tier") == "online_integrated"
        ):
            record_online_decision_failure(ctx)
        if single_pass:
            response = apply_single_pass_decision(
                response, ctx, utterance, persona
            )
            trace = ctx.working_memory.get("online_turn_decision") or {}
            if trace.get("status") == "invalid_protocol_response":
                deterministic_fallback = True
                route = ReplyModelRoute(
                    "deterministic_fallback",
                    ("invalid_online_protocol",),
                )
        elif is_internal_protocol_response(response):
            response = DETERMINISTIC_MODEL_FAILURE_REPLY
            deterministic_fallback = True
            route = ReplyModelRoute(
                "deterministic_fallback",
                ("invalid_model_protocol",),
            )
        if not route.integrated_decision and not deterministic_fallback:
            from .delivery_provenance import record_reply_stage
            before_format = response
            response = _enforce_fast_delivery(response)
            record_reply_stage(ctx, "fast_format", before_format, response)
        route_info = client_route_info(
            selected,
            tier=route.tier,
            reasons=route.reasons,
        )
        if deterministic_fallback:
            route_info.update({
                "attempted_source": route_info.get("source"),
                "attempted_model": route_info.get("effective_model"),
                "source": "deterministic_fallback",
                "effective_model": None,
            })
        routes = ctx.working_memory.setdefault("model_route", {})
        routes["reply"] = route_info
        state_info = routes.get("state") or {}
        if state_info.get("tier") == "online_integrated":
            trace = ctx.working_memory.get("online_turn_decision") or {}
            state_info.update({
                "source": route_info.get("source"),
                "configured_model": route_info.get("configured_model"),
                "effective_model": route_info.get("effective_model"),
                "latency_ms": route_info.get("latency_ms"),
                "status": (
                    "single_pass_" + str(
                        trace.get("status") or "unavailable"
                    )
                ),
            })
        return response


def _fast_delivery_messages(
    messages: list[dict],
    ctx: EngineContext,
) -> list[dict]:
    """快速模型只负责表达既有决策，不允许重新展开一轮心理分析。"""
    copied = [dict(item) for item in messages]
    if not copied:
        return copied
    observation = ctx.reasoning.observation_plan
    preferred_question = (
        observation.question
        if observation is not None
        and observation.mode == "ask"
        and observation.question
        else ""
    )
    dynamic = ""
    if preferred_question:
        dynamic += (
            "\n本轮已经决定的唯一问题是："
            f"「{preferred_question}」"
            "。必须原样使用它；前面最多加一句短承接，不得改写或追加第二个问题。"
        )
    else:
        dynamic += (
            "\n本轮没有指定需要追问：不要自行提问，用一句短陈述自然接住即可。"
        )
    if ctx.reasoning.hypotheses or ctx.emotion.divergence:
        dynamic += (
            "\n本轮存在仍待验证的状态假设：不得说“没事就好”“那就好”或把字面"
            "报平安直接当结论，也不得把假设说成已经看穿对方。"
        )
    if any(
        item.id in {
            "reduced_life_engagement",
            "social_disconnection",
            "loss_of_role",
        }
        for item in ctx.reasoning.hypotheses
    ):
        dynamic += (
            "\n本轮可能涉及生活参与收缩：尊重“不想去”，但不能用“不去就不去”"
            "“歇着挺好”直接结束探索，也不能反过来劝他必须参加。"
        )
    copied[0]["content"] = (
        str(copied[0].get("content") or "")
        + "\n\n## 快速表达硬约束\n"
        "你只负责自然说出上面已经决定的回应方向，不重新分析人物，不解释你的"
        "推理过程。只输出 1-2 个短句，尽量不超过 70 个汉字；最多一个问题、"
        "最多一个问号。若上文给了“优先使用这个问题”，只能问那个问题。"
        "不要复述“他说了某词所以我觉得他其实怎样”，不要列出多个可能原因，"
        "不要臆测输入中没有的动作或事实，不要使用 Markdown，不要加开场标签。"
        + dynamic
    )
    return copied


def _enforce_fast_delivery(response: str) -> str:
    """模型偶尔无视长度/问题数约束时做最小、确定性的交付收口。"""
    raw = str(response).strip()
    action_matches = list(_ACTION_TAG_RE.finditer(raw))
    action_suffix = action_matches[0].group(0) if action_matches else ""
    visible = _ACTION_TAG_RE.sub("", raw)
    value = " ".join(visible.split())
    first_question = value.find("？")
    if first_question < 0:
        first_question = value.find("?")
    if first_question >= 0:
        value = value[: first_question + 1]
    if len(value) > 100:
        candidates = [
            value.rfind(mark, 0, 101)
            for mark in ("。", "！", "？", "；", "，")
        ]
        cut = max(candidates)
        value = value[: cut + 1] if cut >= 20 else value[:100].rstrip()
    if action_suffix:
        return value + "\n" + action_suffix
    return value
