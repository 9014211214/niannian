"""严格双轨路由：在线只用快速模型，深度人物理解只在会后运行。"""
from __future__ import annotations

from dataclasses import dataclass

from .context import EngineContext
from .perception.gating import StateAnalysisRoute


@dataclass(frozen=True)
class ReplyModelRoute:
    tier: str
    reasons: tuple[str, ...]

    @property
    def integrated_decision(self) -> bool:
        """是否让快速模型在一次调用中同时输出状态候选、方向与回复。"""
        return self.tier == "fast_integrated"


def route_reply_model(
    ctx: EngineContext,
    utterance: str,
    *,
    state_route: StateAnalysisRoute,
) -> ReplyModelRoute:
    """选择快速表达或快速综合决策；绝不选择会后深度模型。

    已沉淀的人物模型、确定性证据和当前情绪先在本地进入上下文。复杂回合只是
    改用同一个快速模型的结构化单次决策协议，不会升级到 Claude，也不会串联
    第二次生成调用。明确高风险仍在进入本函数前由确定性安全层接管。
    """
    del utterance  # 路由只读取已经结构化的当轮状态，避免重复做文本推断。
    forced = str(
        ctx.working_memory.get("_force_reply_model") or ""
    ).strip().lower()
    if forced == "fast":
        return ReplyModelRoute("fast", ("benchmark_or_test_override",))
    if forced in {"integrated", "fast_integrated"}:
        return ReplyModelRoute(
            "fast_integrated", ("benchmark_or_test_override",)
        )
    if forced:
        # 历史 deep override 不再拥有线上升级能力。
        return ReplyModelRoute(
            "fast_integrated",
            ("unsupported_override_mapped_to_fast_integrated",),
        )

    risk = ctx.working_memory.get("risk_flag") or {}
    if int(risk.get("tier") or 0) == 1:
        return ReplyModelRoute("fast_integrated", ("risk_tier_1",))
    if ctx.alliance.ruptured:
        return ReplyModelRoute(
            "fast_integrated", ("relationship_rupture",)
        )
    if state_route.tier == "structured":
        return ReplyModelRoute(
            "fast_integrated",
            ("structured_core_requires_integrated_delivery",),
        )

    skills = ctx.working_memory.get("skill_proposals") or []
    if any(
        bool(getattr(item, "consent_required", False))
        or getattr(item, "task_status", "") in {
            "ready_for_confirmation", "unavailable"
        }
        for item in skills
    ):
        return ReplyModelRoute(
            "fast_integrated", ("action_or_consent",)
        )

    if state_route.needs_integrated_decision:
        return ReplyModelRoute(
            "fast_integrated", ("complex_turn_fast_single_pass",)
        )

    decision = ctx.working_memory.get("personality_decision") or {}
    margin = decision.get("decision_margin")
    if (
        isinstance(margin, (int, float))
        and margin < 0.04
        and len(ctx.reasoning.hypotheses) >= 3
    ):
        return ReplyModelRoute(
            "fast_integrated", ("high_decision_uncertainty",)
        )

    return ReplyModelRoute("fast", ("structured_decision_ready",))


def client_route_info(client, *, tier: str, reasons: tuple[str, ...]) -> dict:
    """不暴露密钥的单轮模型可观测信息。"""
    result = {
        "tier": tier,
        "track": "online_fast",
        "integrated_decision": tier == "fast_integrated",
        "reasons": list(reasons),
        "source": getattr(client, "source_id", type(client).__name__),
        "configured_model": getattr(client, "model", None),
        "effective_model": (
            getattr(client, "last_model", None)
            or getattr(client, "model", None)
        ),
        "latency_ms": (
            round(float(client.last_latency_ms))
            if isinstance(
                getattr(client, "last_latency_ms", None), (int, float)
            )
            else None
        ),
    }
    diagnostic = getattr(client, "last_diagnostic", None)
    if isinstance(diagnostic, dict) and diagnostic:
        # 只由客户端写入固定的状态、响应形态和 token 数；不含 prompt、
        # response、异常消息或供应商原始 body。
        result["request_diagnostic"] = dict(diagnostic)
    return result
