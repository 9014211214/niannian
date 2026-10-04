"""知识 & 工具接进主循环的验证。"""
from __future__ import annotations

from datetime import datetime

from aria.engine.context import EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.orchestrator import Engine
from aria.interfaces.knowledge import WorldKnowledgeItem
from aria.scenes.companion_elder.knowledge import ElderKnowledge
from aria.scenes.registry import get_scene
from aria.tools.registry import ToolRegistry


def _elder_engine() -> Engine:
    scene = get_scene("companion_elder")
    persona = load_demo_persona("laochen_demo")
    return Engine(scene, persona=persona)


def test_domain_kb_retrieval():
    kb = ElderKnowledge()
    hits = kb.retrieve("我那个降压药今天还没吃")
    assert hits and "降压药" in hits[0]
    assert kb.retrieve("天气怎么样") == []        # 无关 query 不召回




def test_loop_populates_working_memory_and_recall():
    """主循环:领域知识 + 个人记忆召回都写进 working_memory。"""
    engine = _elder_engine()
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    # 「那只猫」应召回 ep_c_001(referenceable),且不泄露私密 episode。
    engine.process_turn(ctx, "今天那只猫没来。",
                        {"label": "sad", "score": 0.5}, {"label": "neutral", "score": 0.8})
    recalled = ctx.working_memory["recalled_episodes"]
    assert any("橘猫" in r for r in recalled)
    # 可用工具暴露给 policy(通过引擎的工具注册表)。
    assert engine._tools.names() == []


def test_recall_respects_privacy():
    """召回只给可引用的;老伴 / 未接电话等私密记忆不能被召回进可说区。"""
    engine = _elder_engine()
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    engine.process_turn(ctx, "想起以前的事。",
                        {"label": "sad", "score": 0.6}, {"label": "sad", "score": 0.6})
    recalled = " ".join(ctx.working_memory["recalled_episodes"])
    assert "老伴" not in recalled and "电话" not in recalled


def test_domain_knowledge_in_loop():
    engine = _elder_engine()
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    engine.process_turn(ctx, "降压药还没吃呢。")
    domain = ctx.working_memory["domain_knowledge"]
    assert domain and any("降压药" in d for d in domain)


def test_world_knowledge_is_time_scoped_and_never_written_to_persona():
    class World:
        def retrieve(self, _query, top_k=3):
            return [
                WorldKnowledgeItem(
                    content="社区周五有公开棋社活动",
                    source="社区公告",
                    observed_at="2026-07-26T09:00:00",
                    expires_at="2099-07-27T18:00:00",
                ),
                WorldKnowledgeItem(
                    content="已经结束的活动",
                    source="旧公告",
                    observed_at="2020-01-01T09:00:00",
                    expires_at="2020-01-02T09:00:00",
                ),
                WorldKnowledgeItem(
                    content="来自未来的错误公告",
                    source="坏时钟",
                    observed_at="2099-01-01T09:00:00",
                    expires_at="2099-01-02T09:00:00",
                ),
            ]

    persona = load_demo_persona("laochen_demo")
    before = {
        "episodes": persona.episodic_memory.copy(),
        "relationships": persona.relationship_context.copy(),
        "hypotheses": persona.person_hypotheses.copy(),
        "disc": persona.disc.model_copy(deep=True),
        "big_five": persona.big_five.model_copy(deep=True),
    }
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        world_knowledge=World(),
    )
    ctx = EngineContext(
        user_id="laochen_demo", scene_id="companion_elder"
    )
    engine.process_turn(ctx, "社区最近有什么活动？")
    world = ctx.working_memory["world_knowledge"]
    assert [item.content for item in world] == ["社区周五有公开棋社活动"]
    # 公共现状是当轮外部上下文，不进入个人记忆、关系或人格。
    assert persona.episodic_memory == before["episodes"]
    assert persona.relationship_context == before["relationships"]
    assert persona.person_hypotheses == before["hypotheses"]
    assert persona.disc == before["disc"]
    assert persona.big_five == before["big_five"]
