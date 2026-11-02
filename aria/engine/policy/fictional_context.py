"""Explicit ephemeral fictional-date input, separate from every runtime clock.

Only a trusted Python caller can bind this typed value. User text is never
scanned for activation. It is not a person fact, evidence or an action grant.
"""
from dataclasses import dataclass
from datetime import date

KEY = "_fictional_scenario_date_v1"
HISTORY_KEY = "_fictional_conversation_history_v1"


@dataclass(frozen=True)
class FictionalMessage:
    role: str
    content: str

    def __post_init__(self):
        if (type(self.role) is not str or self.role not in ("user", "assistant")
                or type(self.content) is not str):
            raise ValueError("invalid_fictional_history_message")


@dataclass(frozen=True)
class FictionalConversationHistory:
    messages: tuple[FictionalMessage, ...]
    user_id: str
    session_id: str
    scene_id: str

    def __post_init__(self):
        if (type(self.messages) is not tuple
                or any(type(item) is not FictionalMessage for item in self.messages)):
            raise ValueError("untrusted_fictional_history_messages")
        for item in self.messages:
            item.__post_init__()
        if any(type(value) is not str or not value
               for value in (self.user_id, self.session_id, self.scene_id)):
            raise ValueError("invalid_fictional_history_identity")


def bind_fictional_conversation_history(ctx, history: list[dict]) -> FictionalConversationHistory:
    """Bind exact ordered stimulus once to a fresh context, never to its Turn log.

    This explicit Python-only input replaces the builder's usual paired history
    window for this one fictional response. It creates no prior assistant reply,
    person record, observation, permission, invitation, or action receipt.
    """
    if type(history) is not list or any(
        type(row) is not dict or set(row) != {"role", "content"} for row in history
    ):
        raise ValueError("invalid_fictional_history")
    if ctx.turns:
        raise ValueError("fictional_history_requires_fresh_context")
    bound = FictionalConversationHistory(
        tuple(FictionalMessage(**row) for row in history),
        ctx.user_id, ctx.session_id, ctx.scene_id,
    )
    existing = ctx.working_memory.get(HISTORY_KEY)
    if HISTORY_KEY in ctx.working_memory and (
        type(existing) is not FictionalConversationHistory or existing != bound
    ):
        raise ValueError("fictional_history_binding_mismatch")
    ctx.working_memory[HISTORY_KEY] = bound
    return bound


def fictional_conversation_history(ctx) -> list[dict] | None:
    """Validate binding and return an independent copy; None means default off.

    Empty and missing history remain distinct. Paired, unpaired, consecutive and
    leading assistant messages retain their original roles and exact strings.
    A real completed Turn makes this single-response stimulus stale and unusable.
    """
    if HISTORY_KEY not in ctx.working_memory:
        return None
    bound = ctx.working_memory[HISTORY_KEY]
    if type(bound) is not FictionalConversationHistory:
        raise ValueError("untrusted_fictional_history")
    bound.__post_init__()
    if (bound.user_id, bound.session_id, bound.scene_id) != (
        ctx.user_id, ctx.session_id, ctx.scene_id
    ):
        raise ValueError("fictional_history_identity_mismatch")
    if ctx.turns:
        raise ValueError("fictional_history_requires_fresh_context")
    return [{"role": item.role, "content": item.content} for item in bound.messages]


@dataclass(frozen=True)
class FictionalScenarioDate:
    day: str
    user_id: str
    session_id: str
    scene_id: str

    def __post_init__(self):
        if type(self.day) is not str:
            raise ValueError("invalid_fictional_scenario_date")
        try:
            parsed = date.fromisoformat(self.day)
        except (ValueError, TypeError):
            raise ValueError("invalid_fictional_scenario_date") from None
        if parsed.isoformat() != self.day:
            raise ValueError("invalid_fictional_scenario_date")
        if any(type(value) is not str or not value for value in (self.user_id, self.session_id, self.scene_id)):
            raise ValueError("invalid_fictional_scenario_identity")


def bind_fictional_scenario_date(ctx, day: str) -> FictionalScenarioDate:
    """Bind once to this context; no persistence, clock changes or permissions."""
    bound = FictionalScenarioDate(day, ctx.user_id, ctx.session_id, ctx.scene_id)
    existing = ctx.working_memory.get(KEY)
    if KEY in ctx.working_memory and (type(existing) is not FictionalScenarioDate or existing != bound):
        raise ValueError("fictional_scenario_date_binding_mismatch")
    ctx.working_memory[KEY] = bound
    return bound


def fictional_scenario_block(ctx) -> str:
    if KEY not in ctx.working_memory:
        return ""
    bound = ctx.working_memory[KEY]
    if type(bound) is not FictionalScenarioDate:
        raise ValueError("untrusted_fictional_scenario_date")
    bound.__post_init__()
    if (bound.user_id, bound.session_id, bound.scene_id) != (ctx.user_id, ctx.session_id, ctx.scene_id):
        raise ValueError("fictional_scenario_date_identity_mismatch")
    return ("## 虚构场景日期（fictional_scenario_date_v1）\n"
            f"本案例的虚构场景日期为 {bound.day}；只用于理解场景中的相对日期，不代表实际请求日期。"
            "这不是本人的真实经历或身份事实，不改变审计、费用、等待、同意或校准时钟，也不授予任何权限。")
