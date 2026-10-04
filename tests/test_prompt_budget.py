"""生产在线 prompt 的硬预算，防止重新滑回完整历史分析。"""
from __future__ import annotations

from aria.engine.context import EngineContext, Turn
from aria.engine.model.persona import (
    PersonHypothesis,
    PersonaProfile,
    RelationshipContextEntry,
)
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene


def test_online_prompt_has_bounded_history_person_state_and_relationships():
    persona = PersonaProfile(
        user_id="budget",
        person_hypotheses=[
            PersonHypothesis(
                id=f"hyp-{index}",
                framework="response",
                dimension="low_pressure",
                claim=f"人物假设-{index}",
                confidence=0.7,
                status="working",
                first_observed_at="2026-01-01T10:00:00+08:00",
            )
            for index in range(8)
        ],
        relationship_context=[
            RelationshipContextEntry(
                id=f"rel-{index}",
                name=f"关系-{index}",
                relation="朋友",
                facts=[f"关系事实-{index}"],
                referenceable=True,
                first_observed_at="2026-01-01T10:00:00+08:00",
            )
            for index in range(10)
        ],
    )
    ctx = EngineContext(
        user_id="budget",
        scene_id="companion_elder",
        turns=[
            Turn(user=f"历史用户-{index}", agent=f"历史助手-{index}")
            for index in range(10)
        ],
    )
    ctx.working_memory["recalled_episodes"] = [
        f"事件记忆-{index}" for index in range(7)
    ]

    messages = build_messages(
        get_scene("companion_elder"), persona, ctx, "现在这一轮"
    )
    system = messages[0]["content"]
    history = messages[1:-1]

    assert len(history) == 8
    assert "历史用户-5" not in str(history)
    assert "历史用户-6" in str(history)
    assert system.count("经跨会话证据支持的人物假设:") == 4
    assert "人物假设-3" not in system
    assert "人物假设-4" in system
    assert "事件记忆-0" in system and "事件记忆-1" in system
    assert "事件记忆-2" not in system
    assert "关系-3" not in system and "关系-4" in system
    assert "关系-9" in system
