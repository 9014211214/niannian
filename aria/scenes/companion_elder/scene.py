"""Elder companionship with evidence-based understanding and bounded safety."""
from __future__ import annotations

from typing import Optional
from ...interfaces.knowledge import KnowledgeSource
from ...interfaces.scene import AgentPreset, Redline
from ..base import BaseScene
from .knowledge import ElderKnowledge
from .reasoning import ElderStateReasoner
from .risk import ElderRiskMonitor
from .medical_privacy import guard_cognitive_health_reply, is_cognitive_health_sensitive_utterance
from .medical import guard_general_medical_reply
from .delivery import guard_external_delivery_response
from ...engine.cognitive import capture_cognitive_turn, cognitive_sensitive
import re


class ElderCompanionScene(BaseScene):
    id = "companion_elder"
    default_persona_id = "laochen_demo"
    grounding_privacy_predicate = staticmethod(is_cognitive_health_sensitive_utterance)

    def agent_persona(self) -> AgentPreset:
        return AgentPreset(
            name="小予",
            role="懂情绪、会陪伴的家人式伙伴",
            description=(
                "能结合语气、停顿、用词提出老人可能没直说的解释，并以低打扰方式求证;"
                "克制、不黏腻,惦记但不催;记得他的人和事,在对的时刻出现。"
            ),
        )


    def safety_rules(self) -> list[Redline]:
        return [
            Redline("不诊断、不扮演医生;识别到危机只做筛查并升级到人。"),
            Redline("尊重老人自主;通知家人需遵守预设的同意与阈值,不偷偷上报。"),
            Redline("明确承认自己是 AI 助手;不伪装成某个真实亲人。"),
        ]


    def knowledge(self) -> Optional[KnowledgeSource]:
        return ElderKnowledge()


    def state_reasoner(self) -> ElderStateReasoner:
        return ElderStateReasoner()


    def risk_monitor(self) -> ElderRiskMonitor:
        return ElderRiskMonitor()


    def guard_response(self, ctx, response: str) -> str:
        """全局医疗声明守卫；防止普通回合中模型自发越过医疗边界。"""
        cognitive_safe = guard_cognitive_health_reply(response)
        semantic = ctx is not None and ctx.working_memory.get("_online_decision_version") in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")
        if semantic:
            from ...engine.policy.delivery_spans import project_exact_transform, publish_code_response
            if cognitive_safe == " ".join(str(response or "").strip().split()):
                cognitive_safe = response  # Retain exact layout/source positions.
            else:
                publish_code_response(ctx, cognitive_safe)
        medical_version = ("local_refusal_v2" if ctx is not None
                           and ctx.working_memory.get("_safety_delivery_version") in ("source_response_v4", "minimal_response_v1")
                           else "legacy")
        if ctx is not None and ctx.working_memory.get("_online_decision_version") in ("online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7"):
            medical_version = "sentence_refusal_v3"
        source_positions = [] if semantic else None
        safe = (guard_general_medical_reply(cognitive_safe, version=medical_version, source_positions=source_positions)
                if semantic else guard_general_medical_reply(cognitive_safe, version=medical_version))
        if semantic:
            if safe == cognitive_safe.strip():
                safe = cognitive_safe
                source_positions = list(range(len(cognitive_safe)))
            project_exact_transform(ctx, cognitive_safe, safe, source_positions)
        if ctx is not None and ctx.working_memory.get("_cognitive_owner_enabled") and re.search(
            r"阿尔[茨兹]海默|阿兹海默|痴呆|认知障碍|(?:风险|认知|记忆).{0,8}(?:\d+分|概率|高风险|低风险)|\d+(?:\.\d+)?\s*%", safe
        ):
            safe = "我不能通过聊天判断疾病或给出个人评估。反复出现的变化可能有多种原因，是否在下次体检时向医生提一句，由您决定；不愿意也可以。"
            if semantic:
                publish_code_response(ctx, safe)
            return safe
        # All safety/receipt checks above keep their existing compact view.
        # Restore layout only when those checks did not replace content.
        if (ctx is not None
                and ctx.working_memory.get("_online_decision_version") in {"online_turn_decision_v3", "online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7"}
                and safe == " ".join(str(response or "").strip().split())):
            return response
        return safe


    def guard_delivery_response(
        self,
        ctx,
        response: str,
        utterance: str,
        persona=None,
    ) -> str:
        """动作执行后统一核验现实完成态和第三方新事件。"""
        safe = guard_external_delivery_response(
            ctx, response, utterance, persona,
            privacy_predicate=self.grounding_privacy_predicate,
        )
        private_observation_response = bool(ctx.working_memory.pop("_cognitive_block_selected", False))
        if persona is not None and persona.cognitive_observations:
            private_observation_response = private_observation_response or cognitive_sensitive(safe) or any(
                row.source_quote and row.source_quote in safe for row in persona.cognitive_observations)
        if private_observation_response:
            # The assistant may repeat a private observation after a harmless
            # greeting. Its own reply must not carry those records into the
            # ordinary memory or personality-review conversation either.
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
        if (persona is not None and (persona.cognitive_health_consent.has_text_monitoring_consent()
                                     or persona.cognitive_observations)
                and re.search(r"转达|告诉|发给|写给|短信|草稿|家人|女儿|儿子", safe)
                and (cognitive_sensitive(safe) or any(
                    row.source_quote and row.source_quote in safe for row in persona.cognitive_observations))):
            # First-release notification scopes do not authorize cognitive
            # notes. A monitoring checkbox is never third-party sharing consent.
            audit = ctx.working_memory.setdefault("delivery_response_guard", {})
            audit.update(activated=True, response_changed=True)
            audit.setdefault("violations", []).append("cognitive_private_sharing_blocked")
            return "这些变化记录只对您本人展示。若您想告诉家人，请先选出愿意分享的那一句，并明确授权分享范围；我还没有转达或发送。"
        # All safety/receipt checks above keep their existing compact view.
        # Restore layout only when those checks did not replace content.
        if (ctx is not None
                and ctx.working_memory.get("_online_decision_version") in {"online_turn_decision_v3", "online_turn_decision_v4_evidence", "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", "online_turn_decision_v7"}
                and safe == " ".join(str(response or "").strip().split())):
            return response
        return safe


    def observe_care(self, ctx, persona, utterance: str, risk_signal=None) -> bool:
        """Ordinary memory stays separate from explicit owner-controlled C1."""
        if is_cognitive_health_sensitive_utterance(utterance) or cognitive_sensitive(utterance):
            excluded = ctx.working_memory.setdefault("_sensitive_memory_excluded_turns", [])
            if len(ctx.turns) not in excluded:
                excluded.append(len(ctx.turns))
        capture_cognitive_turn(ctx, persona, utterance, risk_signal)
        return False
