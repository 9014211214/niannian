"""大师五要素的结构化层 + 复利 + 风险 + 主动性 + 评测框架。"""
from __future__ import annotations

from datetime import date

from aria.engine.alliance import update_alliance
from aria.engine.context import EngineContext
from aria.engine.formulation import update_formulation
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import EpisodicMemoryEntry, PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.policy.llm import MockLLMClient
from aria.engine.risk.heuristic import HeuristicRiskMonitor
from aria.scenes.base import BaseScene
from aria.scenes.registry import get_scene


# ── 联盟(关系)─────────────────────────────────────────────
def test_alliance_rupture_and_engagement():
    ctx = EngineContext(user_id="u", scene_id="s")
    update_alliance(ctx, "你别管了,烦不烦。")
    assert ctx.alliance.ruptured is True and ctx.alliance.trust < 0.5
    ctx2 = EngineContext(user_id="u", scene_id="s")
    update_alliance(ctx2, "今天那只猫又来蹭饭了,我给它留了点吃的。")
    assert ctx2.alliance.ruptured is False and ctx2.alliance.trust > 0.5


# ── 个案假设 ────────────────────────────────────────────────
def test_formulation_accumulates_from_divergence():
    ctx = EngineContext(user_id="u", scene_id="s")
    persona = load_demo_persona("laochen_demo")
    ctx.emotion.divergence = True
    update_formulation(ctx, persona)
    assert any("依恋" in h for h in ctx.formulation.hypotheses)
    assert any("口是心非" in h for h in ctx.formulation.hypotheses)
    c1 = ctx.formulation.confidence
    update_formulation(ctx, persona)              # 再来一轮证据,置信上升
    assert ctx.formulation.confidence > c1


# ── 复利闭环(护城河):对话 → 沉淀 → 落盘 → 下次读到 ──────────
def test_compounding_loop_persists_across_sessions(tmp_path):
    store = PersonaStore(base_dir=tmp_path)
    persona = load_demo_persona("laochen_demo")
    n0 = len(persona.episodic_memory)

    # Session 1
    engine = Engine(get_scene("companion_elder"), persona=persona,
                    llm_client=MockLLMClient(), store=store)
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    engine.process_turn(ctx, "今天那只猫没来。")
    engine.finalize_session(ctx)                  # 复利沉淀

    # 档案长大了,且落盘了
    assert len(persona.episodic_memory) == n0 + 1
    reloaded = store.load("laochen_demo")
    assert reloaded is not None and len(reloaded.episodic_memory) == n0 + 1
    # 「下次会话」冷启动也能从 store 读到积累 —— 无法冷启动的证据
    assert any(e.type == "session_digest" for e in reloaded.episodic_memory)


# ── 风险预警(启发式框架)──────────────────────────────────
def test_heuristic_risk_tiers():
    ctx = EngineContext(user_id="u", scene_id="s")
    ctx.working_memory["current_utterance"] = "我有时候觉得活不下去了。"
    sig = HeuristicRiskMonitor().assess(ctx)
    assert sig is not None and sig.tier == 3 and sig.kind == "ideation"


def test_engine_routes_risk_when_monitor_active():
    class RiskyElder(BaseScene):
        id = "risky_elder"
        def risk_monitor(self):
            return HeuristicRiskMonitor()
    engine = Engine(RiskyElder(), llm_client=MockLLMClient())
    ctx = EngineContext(user_id="u", scene_id="risky_elder")
    out = engine.process_turn(ctx, "撑不下去了。")
    # tier ≥2 接管:确定性安全回应(不过 LLM),必含热线
    assert "12356" in out and not out.startswith("[mock-llm")


# ── 主动性 / 时机 ──────────────────────────────────────────


# ── 评测框架(骨架)────────────────────────────────────────
