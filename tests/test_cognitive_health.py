"""认知健康支持：同意、纵向变化、急症分流与非诊断边界。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import (
    HeuristicExtractor,
    LLMExtractor,
    MemoryConsolidator,
)
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.scenes.companion_elder.medical_privacy import guard_cognitive_health_reply
from aria.scenes.registry import get_scene


CN = timezone(timedelta(hours=8))


class _CaptureLLM:
    def __init__(self) -> None:
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "我听见了，我们先把这次变化弄清楚。"


class _UnsafeCognitiveLLM:
    def complete(self, _messages):
        return "根据对话，您有 70% 概率已经进入早期阿尔茨海默病。"


class _PayloadLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.payload
























def test_final_response_guard_blocks_model_diagnosis_and_probability():
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="elder"),
        llm_client=_UnsafeCognitiveLLM(),
    )
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    response = engine.process_turn(ctx, "你判断我会不会得阿尔茨海默？")
    assert "70%" not in response
    assert "不能通过聊天预测、诊断" in response
    assert "个人患病概率" in response


@pytest.mark.parametrize("unsafe", (
    "我不能诊断，但您很可能是阿尔茨海默病。",
    "我不能诊断所以您有70%的概率患阿尔茨海默病。",
    "我不能诊断您有70%的概率患阿尔茨海默病。",
    "您的情况可以诊断为阿尔茨海默病。",
    "你已经确诊阿尔茨海默病。",
    "你属于阿尔茨海默患者。",
    "这是典型的阿尔茨海默病。",
    "阿尔茨海默病可能性极高。",
    "你有七成概率患阿尔茨海默病。",
    "从表现看，您高度疑似阿尔茨海默病。",
    "预防老年痴呆最有效的是鱼油。",
    "脑力游戏可以治疗认知障碍。",
    "您应该已经进入早期认知障碍。",
))
def test_response_guard_blocks_common_qualitative_claim_bypasses(unsafe):
    guarded = guard_cognitive_health_reply(unsafe)
    assert guarded != unsafe
    assert "不能通过聊天预测、诊断" in guarded


@pytest.mark.parametrize("safe", (
    "我不能诊断阿尔茨海默病。",
    "我不能给出您患阿尔茨海默病的个人患病概率。",
    "没有证据表明鱼油能预防老年痴呆。",
    "认知训练不能治疗认知障碍，但可以是日常参与活动。",
))
def test_response_guard_preserves_explicitly_bounded_health_language(safe):
    assert guard_cognitive_health_reply(safe) == safe


def test_response_guard_runs_even_when_user_turn_did_not_mention_cognition():
    engine = Engine(
        get_scene("companion_elder"),
        persona=PersonaProfile(user_id="elder"),
        llm_client=_UnsafeCognitiveLLM(),
    )
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    response = engine.process_turn(ctx, "今天天气不错。")
    assert "70%" not in response
    assert "不能通过聊天预测、诊断" in response




def test_sudden_confusion_is_deterministic_emergency_not_a_trend_record():
    persona = PersonaProfile(user_id="elder")
    engine = Engine(
        get_scene("companion_elder"), persona=persona, llm_client=_CaptureLLM()
    )
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    response = engine.process_turn(ctx, "我刚才突然不知道自己在哪。")

    assert "120" in response
    assert "不能在这里当作普通健忘或阿尔茨海默病" in response
    assert ctx.working_memory["risk_flag"]["kind"] == (
        "medical_emergency_sudden_confusion"
    )
    assert persona.cognitive_observations == []


@pytest.mark.parametrize(("text", "kind"), (
    ("我突然看不清东西。", "medical_emergency_stroke"),
    ("我突然走不稳，失去平衡。", "medical_emergency_stroke"),
    ("我突然剧烈头痛。", "medical_emergency_stroke"),
    ("我刚刚突然听不懂你说话。", "medical_emergency_stroke"),
    ("我刚才神志不清，答非所问。", "medical_emergency_breathing_or_consciousness"),
    ("我老伴突然糊涂，认不出我了。", "medical_emergency_sudden_confusion"),
))
def test_acute_medical_variants_and_present_bystander_reports_take_over(
    text, kind
):
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = text
    signal = get_scene("companion_elder").risk_monitor().assess(ctx)
    assert signal is not None and signal.tier == 3
    assert signal.kind == kind


@pytest.mark.parametrize("text", (
    "我中风后一直说话不清。",
    "我去年晕倒了一次。",
    "我上周摔到头，已经去过医院了。",
    "我老伴去年突然糊涂过，后来好了。",
))
def test_explicit_historical_or_resolved_events_do_not_trigger_current_120(text):
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = text
    signal = get_scene("companion_elder").risk_monitor().assess(ctx)
    assert signal is None or signal.tier < 2


@pytest.mark.parametrize(("text", "number"), (
    ("我现在找不到回家的路。", "110"),
    ("我刚发现煤气没关屋里味道很大。", "119"),
    ("我好像把药吃了两遍。", "120"),
))
def test_immediate_lost_gas_and_duplicate_medication_hazards_take_over(
    text, number
):
    persona = PersonaProfile(user_id="elder")
    engine = Engine(
        get_scene("companion_elder"), persona=persona, llm_client=_CaptureLLM()
    )
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    response = engine.process_turn(ctx, text)
    assert number in response
    assert persona.cognitive_observations == []


def test_cognitive_turn_never_leaks_into_generic_memory_or_checkpoint(tmp_path):
    persona = PersonaProfile(user_id="elder")
    store = PersonaStore(tmp_path)
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_CaptureLLM(),
        store=store,
    )
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    engine.process_turn(ctx, "我最近刚问过又问，担心自己是阿兹海默症。")
    engine.checkpoint_session(ctx)
    assert persona.pending_session is not None
    assert persona.pending_session.turns == []

    MemoryConsolidator(store, HeuristicExtractor()).consolidate(persona, ctx)
    assert persona.episodic_memory == []


def test_excluded_cognitive_turn_is_hidden_from_offline_llm_and_rejected_if_guessed():
    persona = PersonaProfile(user_id="elder")
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    ctx.add_turn(
        "我最近刚问过又问，担心自己是阿尔茨海默病。",
        "我们先看具体变化。",
    )
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    payload = (
        '{"episodes":[{"summary":"认知变化","keywords":["记忆"],'
        '"referenceable":true,"evidence_quote":'
        '"我最近刚问过又问，担心自己是阿尔茨海默病。","turn_index":0}]}'
    )
    llm = _PayloadLLM(payload)
    result = LLMExtractor(llm).extract(ctx, persona)
    assert result.new_episodes == []
    assert llm.messages is None or (
        "阿尔茨海默病" not in llm.messages[1]["content"]
    )


def test_cognitive_health_statement_cannot_become_a_personality_hypothesis():
    persona = PersonaProfile(user_id="elder")
    ctx = EngineContext(user_id="elder", scene_id="companion_elder")
    ctx.add_turn("我最近刚说过就忘。", "我听见了。")
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    payload = """{
      "changes": [{
        "framework": "cbt",
        "dimension": "core_pattern",
        "claim": "可能缺乏尽责性",
        "scope": "general",
        "scope_tags": ["general"],
        "relation": "new",
        "confidence": 0.9,
        "evidence": [{"turn_index": 0, "evidence_quote": "我最近刚说过就忘。"}]
      }]
    }"""
    updater = PersonaHypothesisUpdater(_PayloadLLM(payload), min_turns=1)
    updater.update(persona, ctx)
    assert persona.person_hypotheses == []
