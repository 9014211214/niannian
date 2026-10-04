"""补审计缺口②:让 alliance + formulation 真正被用 + 个案假设复利。"""
from __future__ import annotations

from aria.engine.alliance import update_alliance
from aria.engine.context import EngineContext
from aria.engine.formulation import update_formulation
from aria.engine.memory.consolidator import MemoryConsolidator
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene


def test_prompt_consumes_formulation_and_alliance():
    scene = get_scene("companion_elder")
    persona = load_demo_persona("laochen_demo")
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    ctx.emotion.divergence = True
    update_formulation(ctx, persona)               # 注入依恋 + 个案模式 + 口是心非假设
    update_alliance(ctx, "你别管了,烦不烦。")      # 触发 ruptured

    system = build_messages(scene, persona, ctx, "随便说", available_tools=[])[0]["content"]
    assert "你对他的理解" in system                # formulation 进了 prompt(之前是死代码)
    assert "口是心非" in system
    assert "关系信号" in system                    # alliance ruptured 进了 prompt


def test_online_formulation_does_not_bypass_long_term_evidence_gate(tmp_path):
    """单次在线工作假设不能直接升级为长期个案结论。"""
    store = PersonaStore(base_dir=tmp_path)
    persona = PersonaProfile(user_id="u1")          # 全新用户,无任何个案模式
    ctx = EngineContext(user_id="u1", scene_id="s")
    ctx.emotion.divergence = True
    update_formulation(ctx, persona)                # 产生在线假设

    MemoryConsolidator(store).consolidate(persona, ctx)
    assert not any(
        "口是心非" in cp
        for cp in persona.case_conceptualization.core_patterns
    )

    reloaded = store.load("u1")
    assert reloaded is not None
    assert not any(
        "口是心非" in cp
        for cp in reloaded.case_conceptualization.core_patterns
    )
