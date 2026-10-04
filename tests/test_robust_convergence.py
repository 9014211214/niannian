"""P2:判断稳健收敛 —— 意图分流 / 置信加权 / 滞回 / 联盟持久化 / rupture / 反证。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from aria.engine.alliance import update_alliance
from aria.engine.context import EmotionState, EngineContext
from aria.engine.formulation import _H_DIVERGENCE, update_formulation
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.perception import PerceptionEngine, needs_deep_analysis
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.prompt import select_delivery_mode, update_delivery_state
from aria.scenes.registry import get_scene


class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.payload


# ── #7 意图分流:寒暄 / 事实查询不烧探针 ─────────────────────
def test_gating_rules():
    assert not needs_deep_analysis("你好")
    assert not needs_deep_analysis("今天天气怎么样")
    assert not needs_deep_analysis("提醒我下午三点吃药")
    assert not needs_deep_analysis("现在几点了")
    assert needs_deep_analysis("连几点出门都给我定好了")
    assert needs_deep_analysis("唉,今天没什么劲。")
    assert needs_deep_analysis("没事。")                    # 嘴硬高频词必走深度
    assert needs_deep_analysis("都行。")
    assert needs_deep_analysis("孙女走了家里空落落的")


def test_text_perception_has_no_independent_model_call():
    pe = PerceptionEngine()
    assert pe.perceive("现在几点了").label == "unknown"
    assert pe.perceive("算了,不说了。").label == "unknown"


# ── #8 诊断置信加权融合:单次断言不覆盖画像 ──────────────────
def _diagnose(persona, conf, framework="disc"):
    grounded_evidence = [{
        "turn_index": 0,
        "evidence_quote": "本次证据 0",
    }, {
        "turn_index": 1,
        "evidence_quote": "本次证据 1",
    }]
    changes = {
        "disc": {
            "framework": "disc",
            "dimension": "steadiness",
            "claim": "在普遍交流中偏好稳定节奏",
            "scope_tags": ["general"],
            "relation": "support",
            "target_value": 0.9,
            "confidence": conf,
            "evidence": grounded_evidence,
        },
        "big_five": {
            "framework": "big_five",
            "dimension": "extraversion",
            "claim": "在普遍交流中较少主动扩展话题",
            "scope_tags": ["general"],
            "relation": "support",
            "target_value": 0.1,
            "confidence": conf,
            "evidence": grounded_evidence,
        },
        "attachment": {
            "framework": "attachment",
            "dimension": "anxious",
            "claim": "在普遍关系互动中更需要及时回应",
            "scope_tags": ["general"],
            "relation": "support",
            "confidence": conf,
            "evidence": grounded_evidence,
        },
    }
    # 会后协议一次只允许一个最高价值变化；各框架分别验证收敛。
    payload = json.dumps({"changes": [changes[framework]]})
    observed_at = datetime(2026, 1, 1) + timedelta(
        days=persona.person_model_state.sessions_observed
    )
    d = PersonaHypothesisUpdater(
        _FakeLLM(payload),
        min_turns=0,
        now_provider=lambda: observed_at,
    )
    ctx = EngineContext(user_id="u", scene_id="s")
    for index in range(3):
        ctx.add_turn(f"本次证据 {index}", "收到")
    return d.update(persona, ctx)


def test_diagnosis_confidence_weighted_merge():
    p = _diagnose(PersonaProfile(user_id="u"), conf=1.0)
    assert p.disc.steadiness == 0.5                         # 单次高置信仍只是候选
    assert all(item.status == "candidate" for item in p.person_hypotheses)
    p = _diagnose(p, conf=1.0)
    assert p.disc.steadiness == 0.5                         # 假设不污染基线画像
    assert any(item.status == "working" for item in p.person_hypotheses)

    p2 = _diagnose(PersonaProfile(user_id="u"), conf=0.2)
    p2 = _diagnose(p2, conf=0.2)
    assert p2.disc.steadiness == 0.5                        # 低置信不够生效


def test_diagnosis_attachment_needs_confidence_to_overwrite():
    p = PersonaProfile(user_id="u", attachment_style="avoidant")
    p = _diagnose(p, conf=0.2, framework="attachment")
    assert p.attachment_style == "avoidant"
    for _ in range(3):
        p = _diagnose(p, conf=0.8, framework="attachment")
    assert p.attachment_style == "avoidant"                 # 假设不覆盖导入基线
    p = _diagnose(p, conf=0.8, framework="attachment")
    assert p.attachment_style == "avoidant"                 # 稳定假设也保持可撤销

    p3 = PersonaProfile(user_id="u")
    p3 = _diagnose(p3, conf=0.8, framework="attachment")
    assert p3.attachment_style is None
    p3 = _diagnose(p3, conf=0.8, framework="attachment")
    assert p3.attachment_style is None
    assert any(
        item.framework == "attachment"
        and item.dimension == "anxious"
        and item.status == "working"
        for item in p3.person_hypotheses
    )


# ── #9 交付模式滞回:换挡需连续两次会话一致 ──────────────────
def test_delivery_hysteresis():
    p = PersonaProfile(user_id="u", attachment_style="avoidant")
    assert update_delivery_state(p) == "low_pressure"       # 首次定挡直接生效
    # 画像漂成高 D 高 C(阈值另一侧)
    p.attachment_style = None
    p.disc.dominance, p.disc.conscientiousness = 0.9, 0.9
    p.big_five.extraversion = 0.5
    assert update_delivery_state(p) == "low_pressure"       # 第一次新证据:不换
    assert p.delivery_state.candidate == "firm_with_options"
    assert update_delivery_state(p) == "firm_with_options"  # 连续第二次:换挡
    assert select_delivery_mode(p) == "firm_with_options"


def test_delivery_hysteresis_flicker_resets():
    p = PersonaProfile(user_id="u", attachment_style="avoidant")
    update_delivery_state(p)                                # 定挡 low_pressure
    p.attachment_style, p.disc.dominance, p.disc.conscientiousness = None, 0.9, 0.9
    p.big_five.extraversion = 0.5
    update_delivery_state(p)                                # 候选 firm,streak=1
    p.disc.dominance = 0.5
    p.big_five.extraversion = 0.2                           # 证据又回到 low_pressure
    assert update_delivery_state(p) == "low_pressure"
    assert p.delivery_state.candidate is None               # 候选作废,不带病换挡


# ── #10 联盟持久化:trust 跨会话延续 ─────────────────────────
def test_alliance_persists_across_sessions(tmp_path):
    store = PersonaStore(base_dir=tmp_path)
    persona = PersonaProfile(user_id="u1")
    persona.alliance.trust = 0.8
    engine = Engine(get_scene("companion_elder"), persona=persona,
                    llm_client=MockLLMClient(), store=store)
    ctx = EngineContext(user_id="u1", scene_id="companion_elder")
    engine.process_turn(ctx, "今天去公园走了一大圈,腿都酸了")
    assert ctx.alliance.trust > 0.8                         # 播种 0.8 起步而非 0.5
    engine.finalize_session(ctx)
    assert store.load("u1").alliance.trust == ctx.alliance.trust  # 写回落盘


def test_rupture_history_accumulates(tmp_path):
    store = PersonaStore(base_dir=tmp_path)
    engine = Engine(get_scene("companion_elder"), persona=PersonaProfile(user_id="u2"),
                    llm_client=MockLLMClient(), store=store)
    ctx = EngineContext(user_id="u2", scene_id="companion_elder")
    engine.process_turn(ctx, "你别管了,烦不烦。")
    engine.finalize_session(ctx)
    assert store.load("u2").alliance.rupture_count == 1


# ── #11 rupture 由探针判定,关键词只兜底且收紧 ───────────────
def test_rupture_probe_overrides_keywords():
    ctx = EngineContext(user_id="u", scene_id="s")
    ctx.emotion = EmotionState(rupture=True)
    update_alliance(ctx, "嗯,知道了。")                     # 无关键词,探针说裂了
    assert ctx.alliance.ruptured
    ctx2 = EngineContext(user_id="u", scene_id="s")
    ctx2.emotion = EmotionState(rupture=False)
    update_alliance(ctx2, "你别管了,烦不烦。")              # 有关键词,探针说没裂
    assert not ctx2.alliance.ruptured


def test_rupture_keyword_no_longer_hits_weather():
    ctx = EngineContext(user_id="u", scene_id="s")          # 探针无判定 → 关键词兜底
    update_alliance(ctx, "今天天气好烦,一直下雨。")
    assert not ctx.alliance.ruptured                        # 「烦」不再误伤
    update_alliance(ctx, "烦不烦,别问了。")
    assert ctx.alliance.ruptured


def test_self_worth_correction_is_not_mistaken_for_agent_rupture():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    update_alliance(ctx, "我不是觉得自己没用，是膝盖疼。")
    assert not ctx.alliance.ruptured

    update_alliance(ctx, "跟你说也没用，别问了。")
    assert ctx.alliance.ruptured


# ── #12 formulation 反证:假设可被推翻 ───────────────────────
def test_formulation_refutation():
    ctx = EngineContext(user_id="u", scene_id="s")
    ctx.emotion = EmotionState(label="难受", divergence=True)
    update_formulation(ctx)
    update_formulation(ctx)
    assert _H_DIVERGENCE in ctx.formulation.hypotheses
    assert ctx.formulation.confidence > 0.25
    # 之后他反复表里一致地直接说出情绪 → 「压抑」假设被反证撤销
    ctx.emotion = EmotionState(label="难受", divergence=False)
    for _ in range(4):
        update_formulation(ctx)
    assert ctx.formulation.confidence < 0.15
    assert _H_DIVERGENCE not in ctx.formulation.hypotheses
