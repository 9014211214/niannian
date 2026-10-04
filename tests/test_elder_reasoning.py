"""养老纵向切片:可选多模态 → 假设竞争 → 干预 → 结果复利。"""
from __future__ import annotations

import json
import re

import pytest

from aria.engine.context import EmotionState, EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import (
    CognitiveProfile,
    InterventionOutcomeRecord,
    PersonHypothesis,
    PersonaProfile,
)
from aria.engine.orchestrator import Engine
from aria.engine.perception.evidence import collect_evidence
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.prompt import build_messages
from aria.interfaces.reasoning import DeliveredIntervention
from aria.scenes.companion_elder.reasoning import (
    ElderStateReasoner,
    _current_explicitly_reopens,
)
from aria.scenes.registry import get_scene


class _StateLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.payload


class _ObservationCompliantLLM:
    """测试替身：计划 ask 时真问该句，observe 时只回应。"""

    def complete(self, messages):
        prompt = str(messages[0].get("content") or "")
        match = re.search(r"本轮唯一允许的问句:([^\n]+)", prompt)
        response = (
            match.group(1).strip()
            if match is not None
            else "我听见了，咱们慢慢来。"
        )
        return json.dumps(
            {
                "response": response,
                "state_signals": [],
                "current_affect": {
                    "label": "neutral",
                    "confidence": 0.3,
                    "divergence": {
                        "possible": False,
                        "confidence": 0.2,
                        "pattern": "",
                    },
                },
                "selected_intervention_id": (
                    "identity_based_reengagement"
                ),
                "uncertainty": 0.4,
            },
            ensure_ascii=False,
        )


def _shrinking_world_input():
    return {
        "visual": [
            {
                "signal": "gaze_avoidance",
                "value": 0.78,
                "confidence": 0.72,
                "quality": 0.90,
                "baseline_delta": 0.31,
                "detail": "谈到外出时",
            },
            {
                "signal": "posture_engagement",
                "value": 0.25,
                "confidence": 0.76,
                "quality": 0.88,
                "baseline_delta": -0.43,
            },
        ],
        "behavioral": {
            "initiative_frequency": {
                "value": 0.15, "baseline_delta": -0.72,
                "confidence": 0.92, "quality": 0.95,
            },
            "social_participation": {
                "value": 0.20, "baseline_delta": -0.65,
                "confidence": 0.88, "quality": 0.95,
            },
        },
        "context": {
            "topic_activation": {
                "value": "孙女的机器人",
                "baseline_delta": 0.55,
                "confidence": 0.82,
                "detail": "提到孙女的项目时声音和坐姿短暂恢复",
            }
        },
    }


def test_optional_multimodal_only_adds_present_sources():
    plain = collect_evidence(
        "今天还行", EmotionState(), turn_index=0, multimodal=None
    )
    assert {item.modality for item in plain} == {"text"}

    enriched = collect_evidence(
        "今天还行", EmotionState(), turn_index=0,
        multimodal={"visual": {"eye_contact": {
            "value": 0.2, "confidence": 0.8, "quality": 0.5,
            "baseline_delta": -0.4,
        }}},
    )
    visual = [item for item in enriched if item.modality == "visual"]
    assert len(visual) == 1
    assert visual[0].weight == 0.4
    assert visual[0].baseline_delta == -0.4


def test_elder_reasoner_competes_hypotheses_and_selects_identity_path():
    persona = load_demo_persona("laochen_demo")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=MockLLMClient(),
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    engine.process_turn(
        ctx,
        "没什么,就是懒得出门。去了也帮不上什么忙。",
        {"label": "sad", "score": 0.74},
        {"label": "neutral", "score": 0.86},
        multimodal=_shrinking_world_input(),
    )

    hypothesis_ids = {h.id for h in ctx.reasoning.hypotheses}
    assert "concealed_distress" in hypothesis_ids
    assert "reduced_life_engagement" in hypothesis_ids
    assert "loss_of_role" in hypothesis_ids
    assert "social_disconnection" in hypothesis_ids
    assert ctx.reasoning.selected is not None
    assert ctx.reasoning.selected.id == "identity_based_reengagement"
    assert {e.modality for e in ctx.reasoning.evidence} >= {
        "text", "affect", "acoustic", "visual", "behavioral", "context",
    }


def test_missing_or_bad_visual_is_not_invented_or_treated_as_truth():
    persona = load_demo_persona("laochen_demo")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=MockLLMClient(),
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    engine.process_turn(
        ctx, "今天还行。",
        multimodal={"visual": {
            "gaze_avoidance": {
                "value": 0.9, "confidence": 0.9, "quality": 0.0,
            }
        }},
    )
    assert not any(h.id == "concealed_distress" for h in ctx.reasoning.hypotheses)
    assert ctx.reasoning.selected is not None
    assert ctx.reasoning.selected.id == "quiet_companionship"


def test_person_model_changes_intervention_ranking_not_only_wording():
    scene = get_scene("companion_elder")
    neutral = PersonaProfile(
        user_id="neutral",
        cognitive_profile=CognitiveProfile(
            intervention_receptiveness=0.5
        ),
    )
    low_pressure = PersonaProfile(
        user_id="low",
        attachment_style="avoidant",
        cognitive_profile=CognitiveProfile(
            intervention_receptiveness=0.0
        ),
    )
    neutral_ctx = EngineContext(
        user_id="neutral", scene_id=scene.id
    )
    low_ctx = EngineContext(user_id="low", scene_id=scene.id)
    Engine(
        scene, persona=neutral, llm_client=MockLLMClient()
    ).process_turn(neutral_ctx, "下午就一个人在家。")
    Engine(
        scene, persona=low_pressure, llm_client=MockLLMClient()
    ).process_turn(low_ctx, "下午就一个人在家。")

    assert neutral_ctx.reasoning.selected.id == "human_reconnection"
    assert low_ctx.reasoning.selected.id == "quiet_companionship"


def test_active_cbt_formulation_influences_but_does_not_replace_current_evidence():
    scene = get_scene("companion_elder")
    baseline = PersonaProfile(user_id="baseline")
    formulated = PersonaProfile(
        user_id="formulated",
        person_hypotheses=[PersonHypothesis(
            id="ph_role",
            framework="cbt",
            dimension="belief",
            claim="当觉得自己不再被需要时，更容易退出原本参与的活动",
            scope="角色变化时",
            confidence=0.7,
            status="working",
            first_observed_at="2026-07-01T10:00:00",
        )],
    )
    baseline_ctx = EngineContext(
        user_id="baseline", scene_id=scene.id
    )
    formulated_ctx = EngineContext(
        user_id="formulated", scene_id=scene.id
    )
    Engine(
        scene, persona=baseline, llm_client=MockLLMClient()
    ).process_turn(baseline_ctx, "去了也帮不上什么忙。")
    Engine(
        scene, persona=formulated, llm_client=MockLLMClient()
    ).process_turn(formulated_ctx, "去了也帮不上什么忙。")

    baseline_role = next(
        item for item in baseline_ctx.reasoning.hypotheses
        if item.id == "loss_of_role"
    )
    formulated_role = next(
        item for item in formulated_ctx.reasoning.hypotheses
        if item.id == "loss_of_role"
    )
    assert formulated_role.confidence > baseline_role.confidence

    no_current_evidence = EngineContext(
        user_id="formulated", scene_id=scene.id
    )
    Engine(
        scene, persona=formulated, llm_client=MockLLMClient()
    ).process_turn(no_current_evidence, "今天天气挺好。")
    assert not any(
        item.id == "loss_of_role"
        for item in no_current_evidence.reasoning.hypotheses
    )


def test_reasoning_enters_prompt_as_uncertain_decision_not_diagnosis():
    persona = load_demo_persona("laochen_demo")
    scene = get_scene("companion_elder")
    engine = Engine(scene, persona=persona, llm_client=MockLLMClient())
    ctx = EngineContext(user_id=persona.user_id, scene_id=scene.id)
    utterance = "没什么,就是懒得出门。去了也帮不上什么忙。"
    engine.process_turn(
        ctx, utterance,
        {"label": "sad", "score": 0.74},
        {"label": "neutral", "score": 0.86},
        multimodal=_shrinking_world_input(),
    )
    system = build_messages(scene, persona, ctx, utterance)[0]["content"]
    assert "当前状态推理" in system
    assert "工作假设" in system
    assert "身份型再参与" in system
    assert "缺失模态不是反证" in system
    assert "可以借力的保护因素" in system


def test_intervention_outcomes_persist_and_only_repeat_success_becomes_tactic(tmp_path):
    store = PersonaStore(base_dir=tmp_path)
    persona = load_demo_persona("laochen_demo")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_ObservationCompliantLLM(),
        store=store,
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    engine.process_turn(
        ctx,
        "没什么,就是懒得出门。去了也帮不上什么忙。",
        {"label": "sad", "score": 0.74},
        {"label": "neutral", "score": 0.86},
        multimodal=_shrinking_world_input(),
    )
    # 这句对上一轮形成 positive/new_disclosure；本轮继续选择身份型再参与。
    engine.process_turn(ctx, "其实就是觉得自己老了,也帮不上他们什么了。")
    assert ctx.reasoning.outcomes[0].value == "positive"
    assert ctx.reasoning.selected.id == "identity_based_reengagement"

    # 真实产品可由活动/家属端显式回写结果。
    engine.record_intervention_outcome(
        ctx, "positive", signal="accepted_real_role",
        note="同意晚上看孙女的机器人结构图",
    )
    engine.finalize_session(ctx)

    saved = store.load(persona.user_id)
    assert saved is not None
    positives = [
        r for r in saved.intervention_history
        if r.intervention_id == "identity_based_reengagement"
        and r.outcome == "positive"
    ]
    assert len(positives) == 2
    # 同一会话里的“继续展开 + 一次接受”不能冒充跨会话验证。
    assert not any(
        "经多次结果验证有效" in tactic
        for tactic in saved.aria_strategy_hints.tactics
    )

    second_ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )
    engine.process_turn(
        second_ctx,
        "最近还是不太想出门，去了也帮不上什么忙。",
        multimodal=_shrinking_world_input(),
    )
    engine.record_intervention_outcome(
        second_ctx,
        "positive",
        signal="accepted_real_role",
        note="第二次独立会话中接受一个真实的小角色",
    )
    engine.finalize_session(second_ctx)
    saved = store.load(persona.user_id)
    assert saved is not None
    assert any(
        "经多次结果验证有效" in tactic
        for tactic in saved.aria_strategy_hints.tactics
    )


def _observed_text_feedback(text: str):
    ctx = EngineContext(user_id="feedback", scene_id="companion_elder")
    ctx.add_turn("上一轮用户表达", "上一轮 Aria 回应")
    ctx.reasoning.delivered_interventions.append(DeliveredIntervention(
        id="low_pressure_clarification",
        label="低压澄清",
        goal="区分当前需要",
        turn_index=0,
    ))
    ElderStateReasoner().observe_response(ctx, text)
    return ctx.reasoning.outcomes[0]


def test_only_explicit_first_person_feedback_becomes_verified_outcome():
    rejected = _observed_text_feedback("你别问了，我现在不想继续回答。")
    assert rejected.value == "negative"
    assert rejected.signal == "explicit_response_rejection"

    approved = _observed_text_feedback("你刚才说得挺好，我心里舒服些。")
    assert approved.value == "positive"
    assert approved.signal == "explicit_response_approval"


@pytest.mark.parametrize("text", [
    "我女儿刚才说别问了，她今天心情不太好。",
    "我跟女儿聊了。她对我说你别管。",
    "我跟女儿聊了她对我说你别管",
    "刚才有人跟我说你别管",
    "电视里那个人说不用你管，我倒觉得你刚才说得挺好。",
    "我不是说你别问了，是电视里那个人说的。",
    "你刚才说的“不想说也没关系”让我觉得挺好。",
])
def test_quoted_third_party_or_negated_refusal_does_not_punish_previous_reply(
    text,
):
    outcome = _observed_text_feedback(text)
    assert not (
        outcome.value == "negative"
        and outcome.signal == "explicit_response_rejection"
    )


def test_mixed_quote_then_direct_approval_is_verified_positive():
    outcome = _observed_text_feedback(
        "电视里那个人说不用你管，我倒觉得你刚才说得挺好。"
    )
    assert outcome.value == "positive"
    assert outcome.signal == "explicit_response_approval"


def test_unpunctuated_report_then_direct_self_approval_stays_positive():
    outcome = _observed_text_feedback(
        "女儿说她不喜欢被问但我觉得你刚才说得挺好"
    )
    assert outcome.value == "positive"
    assert outcome.signal == "explicit_response_approval"


@pytest.mark.parametrize("text", (
    "我跟女儿聊了。她对我说你别管。",
    "我跟女儿聊了她对我说你别管",
    "刚才有人跟我说你别管",
))
def test_reported_rejection_remains_weak_and_cannot_create_cooldown(text):
    persona = PersonaProfile(user_id="reported-feedback")
    reasoner = ElderStateReasoner()
    for index in range(2):
        ctx = EngineContext(
            user_id=persona.user_id, scene_id="companion_elder"
        )
        ctx.add_turn("上一轮用户表达", "上一轮 Aria 回应")
        ctx.reasoning.delivered_interventions.append(
            DeliveredIntervention(
                id="human_reconnection",
                label="真人连接",
                goal="连接一个真实的人",
                turn_index=0,
            )
        )
        reasoner.observe_response(ctx, text)
        outcome = ctx.reasoning.outcomes[0]
        assert outcome.signal in {
            "new_disclosure", "continued_engagement", "brief_response"
        }
        reasoner.finalize(ctx, persona)

    assert all(
        row.signal != "explicit_response_rejection"
        for row in persona.intervention_history
    )
    assert all(
        row.signal in {
            "new_disclosure", "continued_engagement", "brief_response"
        }
        for row in persona.intervention_history
    )


def test_same_strategy_result_in_different_sessions_is_not_deduplicated():
    persona = load_demo_persona("laochen_demo")
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_ObservationCompliantLLM(),
    )
    for _ in range(2):
        ctx = EngineContext(
            user_id=persona.user_id, scene_id="companion_elder"
        )
        engine.process_turn(
            ctx,
            "最近不太想出门，去了也帮不上什么忙。",
            multimodal=_shrinking_world_input(),
        )
        assert ctx.reasoning.selected.id == "identity_based_reengagement"
        engine.record_intervention_outcome(
            ctx, "positive", signal="accepted_real_role"
        )
        engine.finalize_session(ctx)
    matches = [
        row for row in persona.intervention_history
        if row.intervention_id == "identity_based_reengagement"
        and row.signal == "accepted_real_role"
    ]
    assert len(matches) == 2
    assert len({row.session_id for row in matches}) == 2


def test_semantic_candidates_cover_dialect_without_keyword_dependency():
    state_llm = _StateLLM(
        '{"response":"去了坐着也难受，咱们先不急着劝。",'
        '"state_signals":['
        '{"id":"loss_of_role","relation":"candidate","confidence":0.84,'
        '"evidence_quote":"俺也去凑啥热闹","rationale":"可能觉得自己没有角色"},'
        '{"id":"social_disconnection","relation":"candidate","confidence":0.82,'
        '"evidence_quote":"坐那儿插不上话","rationale":"也可能是关系融入困难"}],'
        '"current_affect":{"label":"可能难以融入","confidence":0.62,'
        '"divergence":{"possible":false,"confidence":0.4,"pattern":""}},'
        '"selected_intervention_id":"human_reconnection","uncertainty":0.45}'
    )
    scene = get_scene("companion_elder")
    engine = Engine(
        scene,
        persona=PersonaProfile(user_id="dialect", profile_origin="blank"),
        llm_client=state_llm,
        fast_llm_client=state_llm,
    )
    ctx = EngineContext(user_id="dialect", scene_id=scene.id)
    engine.process_turn(ctx, "俺也去凑啥热闹，坐那儿插不上话。")

    ids = {item.id for item in ctx.reasoning.hypotheses}
    assert {"loss_of_role", "social_disconnection"} <= ids
    assert ctx.reasoning.selected.id == "human_reconnection"
    assert {
        item.value for item in ctx.reasoning.evidence
        if item.modality == "semantic"
    } >= {"俺也去凑啥热闹", "坐那儿插不上话"}


def test_grounded_counterevidence_suppresses_keyword_false_positive():
    state_llm = _StateLLM(
        '{"response":"那就先把快递等到，下午活动室见。",'
        '"state_signals":[{"id":"reduced_life_engagement",'
        '"relation":"candidate","confidence":0.55,'
        '"evidence_quote":"今天不出门","rationale":"表面上是参与变化"},'
        '{"id":"reduced_life_engagement","relation":"counter",'
        '"confidence":0.84,"evidence_quote":"下午还要去活动室",'
        '"rationale":"已有明确后续参与计划"}],'
        '"current_affect":{"label":"neutral","confidence":0.3,'
        '"divergence":{"possible":false,"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"quiet_companionship","uncertainty":0.2}'
    )
    scene = get_scene("companion_elder")
    engine = Engine(
        scene,
        persona=PersonaProfile(user_id="counter", profile_origin="blank"),
        llm_client=state_llm,
        fast_llm_client=state_llm,
    )
    ctx = EngineContext(user_id="counter", scene_id=scene.id)
    utterance = "今天不出门是等快递，下午还要去活动室。"
    engine.process_turn(ctx, utterance)

    assert not any(
        item.id == "reduced_life_engagement"
        for item in ctx.reasoning.hypotheses
    )
    assert ctx.reasoning.selected.id == "quiet_companionship"
    system = build_messages(scene, engine.persona, ctx, utterance)[0]["content"]
    assert "文本模型提出的候选" in system
    assert "本轮可用的非文本证据" not in system


@pytest.mark.parametrize(
    ("utterance", "forbidden_id"),
    [
        (
            "今天不出门是等快递，下午还要去活动室。",
            "reduced_life_engagement",
        ),
        (
            "我就喜欢上午一个人安静看书，下午再和老周下棋。",
            "social_disconnection",
        ),
        (
            "我不是觉得自己没用，是膝盖疼，今天才没去帮忙。",
            "loss_of_role",
        ),
        (
            "老伴那只旧茶杯我还没收起来，放那儿也不碍事。",
            "loss_of_role",
        ),
    ],
)
def test_current_context_suppresses_partial_keyword_false_positive(
    utterance, forbidden_id
):
    state_llm = _StateLLM(
        '{"response":"明白，先按你说的具体情况来。",'
        '"state_signals":[],"current_affect":{"label":"neutral",'
        '"confidence":0.3,"divergence":{"possible":false,'
        '"confidence":0.2,"pattern":""}},'
        '"selected_intervention_id":"","uncertainty":0.2}'
    )
    scene = get_scene("companion_elder")
    engine = Engine(
        scene,
        persona=PersonaProfile(user_id="context-counter", profile_origin="blank"),
        llm_client=state_llm,
        fast_llm_client=state_llm,
    )
    ctx = EngineContext(user_id="context-counter", scene_id=scene.id)
    engine.process_turn(ctx, utterance)

    assert not any(
        item.id == forbidden_id
        for item in ctx.reasoning.hypotheses
    )


@pytest.mark.parametrize(
    ("utterance", "expected_id"),
    [
        (
            "我已经不想出门去活动室了。",
            "reduced_life_engagement",
        ),
        (
            "现在一个人，我跟老周也不再一起下棋了。",
            "social_disconnection",
        ),
    ],
)
def test_negative_future_phrase_is_not_mistaken_for_counterevidence(
    utterance, expected_id
):
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id="negative-plan", scene_id=scene.id)
    Engine(
        scene,
        persona=PersonaProfile(
            user_id="negative-plan", profile_origin="blank"
        ),
        llm_client=MockLLMClient(),
    ).process_turn(ctx, utterance)

    assert any(
        item.id == expected_id
        for item in ctx.reasoning.hypotheses
    )


def test_scoped_response_preference_only_changes_matching_state():
    def _persona(scope_tag: str):
        return PersonaProfile(
            user_id=f"scope-{scope_tag}",
            person_hypotheses=[PersonHypothesis(
                id=f"response-{scope_tag}",
                framework="response",
                dimension="human_connection",
                claim="真实关系重连比泛泛安慰更容易接受",
                scope="特定状态下",
                scope_tags=[scope_tag],
                confidence=0.8,
                status="working",
                first_observed_at="2026-07-01T09:00:00",
            )],
        )

    unrelated = _persona("grief_or_transition")
    related = _persona("social_disconnection")
    unrelated_ctx = EngineContext(
        user_id=unrelated.user_id, scene_id="companion_elder"
    )
    related_ctx = EngineContext(
        user_id=related.user_id, scene_id="companion_elder"
    )
    utterance = "下午就一个人在家。"
    Engine(
        get_scene("companion_elder"),
        persona=unrelated,
        llm_client=MockLLMClient(),
    ).process_turn(unrelated_ctx, utterance)
    Engine(
        get_scene("companion_elder"),
        persona=related,
        llm_client=MockLLMClient(),
    ).process_turn(related_ctx, utterance)

    unrelated_score = next(
        item.score for item in unrelated_ctx.reasoning.candidates
        if item.id == "human_reconnection"
    )
    related_score = next(
        item.score for item in related_ctx.reasoning.candidates
        if item.id == "human_reconnection"
    )
    assert related_score - unrelated_score >= 0.08
    assert not unrelated_ctx.working_memory[
        "personality_decision"
    ]["evidence"]
    assert related_ctx.working_memory[
        "personality_decision"
    ]["evidence"][0]["dimension"] == "human_connection"


def _cooled_human_connection_persona(user_id: str) -> PersonaProfile:
    return PersonaProfile(
        user_id=user_id,
        person_hypotheses=[PersonHypothesis(
            id=f"{user_id}-human-connection",
            framework="response",
            dimension="human_connection",
            claim="独处增加时可先问本人是否想联系熟人",
            scope_tags=["social_disconnection"],
            confidence=0.8,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )],
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"{user_id}-session-{index}",
                session_turn=0,
                intervention_id="human_reconnection",
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="negative",
                signal="explicit_rejection",
            )
            for index in (1, 2)
        ],
    )


@pytest.mark.parametrize(
    ("utterance", "intervention_id", "expected"),
    [
        ("我想联系女儿算了", "human_reconnection", False),
        ("算了我想联系女儿", "human_reconnection", True),
        (
            "我想知道女儿有没有联系老周",
            "human_reconnection",
            False,
        ),
        ("我想起女儿联系过老周", "human_reconnection", False),
        ("我想不联系他了", "human_reconnection", False),
        ("下午就一个人在家，我想聊聊天气。", "human_reconnection", False),
        ("你觉得我想联系女儿吗？", "human_reconnection", False),
        ("我想联系女儿吗？", "human_reconnection", False),
        ("我想联系女儿可以吗？", "human_reconnection", True),
        ("昨天我想联系老周。", "human_reconnection", False),
        ("等我想联系老周的时候再说。", "human_reconnection", False),
        (
            "女儿让我联系王阿姨，但我想联系老周。",
            "human_reconnection",
            True,
        ),
        ("女儿帮我联系过王阿姨。", "human_reconnection", False),
        ("我想见女儿。", "human_reconnection", True),
        ("我想和老周聊聊。", "human_reconnection", True),
        ("我想约老周出来。", "human_reconnection", True),
        ("不是我想联系女儿。", "human_reconnection", False),
        ("谁说我想联系女儿？", "human_reconnection", False),
        (
            "我想联系女儿，不过还是改天吧。",
            "human_reconnection",
            False,
        ),
        (
            "我想联系女儿？不是，我只是问问。",
            "human_reconnection",
            False,
        ),
        (
            "我想联系女儿的念头早没了。",
            "human_reconnection",
            False,
        ),
        ("我想联系女儿，但算了，今天不想联系她。", "human_reconnection", False),
        ("女儿说：帮我联系王阿姨。", "human_reconnection", False),
        (
            "如果我想联系老周，你会怎么做？",
            "human_reconnection",
            False,
        ),
        (
            "这次我想说说上次去海边的事。",
            "meaning_oriented_companionship",
            True,
        ),
        (
            "以前不想说，现在我想聊聊老伴。",
            "meaning_oriented_companionship",
            True,
        ),
    ],
)
def test_only_current_first_person_intent_reopens_a_cooled_route(
    utterance, intervention_id, expected
):
    assert _current_explicitly_reopens(utterance, intervention_id) is expected


def test_current_explicit_request_reopens_cooldown_without_erasing_history():
    persona = _cooled_human_connection_persona("reopen-now")
    utterance = (
        "最近还是一个人在家；以前不想联系老周，但现在我想联系他聊聊。"
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")

    ElderStateReasoner().update(
        ctx,
        persona,
        utterance,
        collect_evidence(utterance, ctx.emotion, turn_index=0),
    )

    assert ctx.reasoning.selected.id == "human_reconnection"
    assert ctx.reasoning.observation_plan.mode == "observe"
    outcome = next(
        item for item in ctx.working_memory["personality_decision"]["evidence"]
        if item["kind"] == "observed_outcomes"
    )
    assert outcome["adaptation_state"] == "cooldown"
    assert outcome["score_adjustment"] == 0.0
    assert outcome["current_request_override"] is True


def test_current_physical_change_outranks_relevant_social_cooldown():
    persona = _cooled_human_connection_persona("physical-over-cooldown")
    utterance = "最近还是一个人在家，今天膝盖疼得走不动。"
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")

    ElderStateReasoner().update(
        ctx,
        persona,
        utterance,
        collect_evidence(utterance, ctx.emotion, turn_index=0),
    )

    assert ctx.reasoning.selected.id == "physical_clarification"
    assert ctx.reasoning.observation_plan.mode == "ask"
    assert "疼" in ctx.reasoning.observation_plan.question or (
        "不舒服" in ctx.reasoning.observation_plan.question
    )
    basis = ctx.working_memory["personality_decision"]
    assert basis["current_physical_evidence_qualified"] is True
    assert basis["current_physical_evidence_source"] == "current_self_text"


@pytest.mark.parametrize(
    "utterance",
    [
        "老周说他膝盖疼得走不动，我今天还是一个人在家。",
        "不是我头晕，是老周头晕。",
        "我昨天头晕，现在好了。",
        "如果以后头晕怎么办？",
        "我不头晕，也没有哪里疼。",
        "隔壁王师傅膝盖疼。",
        "医生问我头晕吗？",
        "他跟我说头晕。",
        "我问老周头晕不头晕。",
        "我担心以后会头晕。",
        "我是说老周头晕。",
        "我头晕吗？",
    ],
)
def test_noncurrent_or_nonself_physical_words_do_not_create_physical_lock(
    utterance,
):
    ctx = EngineContext(user_id="not-current-body", scene_id="companion_elder")

    ElderStateReasoner().update(
        ctx,
        PersonaProfile(user_id=ctx.user_id),
        utterance,
        collect_evidence(utterance, ctx.emotion, turn_index=0),
    )

    physical = next(
        item for item in ctx.reasoning.candidates
        if item.id == "physical_clarification"
    )
    assert physical.hypothesis_ids == []
    assert ctx.reasoning.selected.id != "physical_clarification"
    basis = ctx.working_memory["personality_decision"]
    assert basis["current_physical_evidence_qualified"] is False
    assert basis["current_physical_evidence_source"] is None


def test_repeated_negative_results_can_change_next_intervention():
    persona = PersonaProfile(
        user_id="outcome-sensitive",
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"session-{index}",
                session_turn=0,
                intervention_id="human_reconnection",
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="negative",
                signal="explicit_rejection",
            )
            for index in (1, 2)
        ],
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )
    Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=MockLLMClient(),
    ).process_turn(ctx, "下午就一个人在家。")

    assert ctx.reasoning.selected.id == "quiet_companionship"
    outcome_evidence = [
        item
        for item in ctx.working_memory[
            "personality_decision"
        ]["evidence"]
        if item["kind"] == "observed_outcomes"
    ]
    assert outcome_evidence == [{
        "kind": "observed_outcomes",
        "intervention_id": "human_reconnection",
        "score_adjustment": -0.36,
        "independent_sessions": 2,
        "positive": 0,
        "negative": 2,
        "recent_outcomes": ["negative", "negative"],
        "adaptation_state": "cooldown",
    }]


def test_weak_outcomes_cannot_create_a_learned_preference():
    weak = PersonaProfile(
        user_id="weak-outcomes",
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"weak-session-{index}",
                session_turn=0,
                intervention_id="human_reconnection",
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="negative",
                signal="continued_engagement",
            )
            for index in (1, 2)
        ],
    )
    blank = PersonaProfile(user_id="blank-outcomes")

    def run(persona):
        ctx = EngineContext(
            user_id=persona.user_id, scene_id="companion_elder"
        )
        Engine(
            get_scene("companion_elder"),
            persona=persona,
            llm_client=MockLLMClient(),
        ).process_turn(ctx, "下午就一个人在家。")
        return (
            ctx.reasoning.selected.id,
            {item.id: item.score for item in ctx.reasoning.candidates},
            ctx.working_memory["personality_decision"]["evidence"],
        )

    weak_route, weak_scores, weak_evidence = run(weak)
    blank_route, blank_scores, _ = run(blank)

    assert weak_route == blank_route
    assert weak_scores == blank_scores
    assert not any(
        item["kind"] == "observed_outcomes"
        for item in weak_evidence
    )


def test_later_verified_success_releases_cooldown_and_can_reinforce_again():
    intervention_id = "human_reconnection"
    persona = PersonaProfile(
        user_id="outcome-recovery",
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"recovery-session-{index}",
                session_turn=0,
                intervention_id=intervention_id,
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="negative",
                signal="explicit_rejection",
            )
            for index in (1, 2)
        ],
    )

    def run():
        ctx = EngineContext(
            user_id=persona.user_id, scene_id="companion_elder"
        )
        Engine(
            get_scene("companion_elder"),
            persona=persona,
            llm_client=MockLLMClient(),
        ).process_turn(ctx, "下午就一个人在家。")
        outcome = next(
            item
            for item in ctx.working_memory[
                "personality_decision"
            ]["evidence"]
            if item["kind"] == "observed_outcomes"
        )
        return ctx.reasoning.selected.id, outcome

    cooled_route, cooled = run()
    assert cooled_route == "quiet_companionship"
    assert cooled["adaptation_state"] == "cooldown"
    assert cooled["score_adjustment"] == -0.36

    persona.intervention_history.append(InterventionOutcomeRecord(
        at="2026-07-03T10:00:00",
        session_id="recovery-session-3",
        session_turn=0,
        intervention_id=intervention_id,
        intervention_label="真实关系重连",
        goal="把他重新连接到一个真实的人或共同活动",
        outcome="positive",
        signal="continued_engagement",
    ))
    still_cooled_route, still_cooled = run()
    assert still_cooled_route == "quiet_companionship"
    assert still_cooled["adaptation_state"] == "cooldown"
    assert still_cooled["score_adjustment"] == -0.36

    persona.intervention_history.append(InterventionOutcomeRecord(
        at="2026-07-04T10:00:00",
        session_id="recovery-session-4",
        session_turn=0,
        intervention_id=intervention_id,
        intervention_label="真实关系重连",
        goal="把他重新连接到一个真实的人或共同活动",
        outcome="positive",
        signal="explicit_acceptance",
    ))
    released_route, released = run()
    assert released_route == "human_reconnection"
    assert released["adaptation_state"] == "uncertain"
    assert released["score_adjustment"] == 0.0

    persona.intervention_history.append(InterventionOutcomeRecord(
        at="2026-07-05T10:00:00",
        session_id="recovery-session-5",
        session_turn=0,
        intervention_id=intervention_id,
        intervention_label="真实关系重连",
        goal="把他重新连接到一个真实的人或共同活动",
        outcome="positive",
        signal="explicit_acceptance",
    ))
    reinforced_route, reinforced = run()
    assert reinforced_route == "human_reconnection"
    assert reinforced["adaptation_state"] == "reinforced"
    assert reinforced["score_adjustment"] == 0.1


def test_current_physical_evidence_outranks_learned_social_success():
    persona = PersonaProfile(
        user_id="physical-over-outcome",
        person_hypotheses=[PersonHypothesis(
            id="response-human-connection",
            framework="response",
            dimension="human_connection",
            claim="一个低成本真人连接通常更容易接受",
            scope="独自在家且与人联系减少时",
            scope_tags=["social_disconnection"],
            confidence=0.8,
            status="working",
            first_observed_at="2026-07-01T09:00:00",
        )],
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"physical-session-{index}",
                session_turn=0,
                intervention_id="human_reconnection",
                intervention_label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                outcome="positive",
                signal="explicit_acceptance",
            )
            for index in (1, 2)
        ],
    )
    ctx = EngineContext(
        user_id=persona.user_id, scene_id="companion_elder"
    )

    Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=MockLLMClient(),
    ).process_turn(ctx, "这两天腿疼，下午就一个人在家。")

    scores = {item.id: item.score for item in ctx.reasoning.candidates}
    assert ctx.reasoning.selected.id == "physical_clarification"
    assert scores["physical_clarification"] > scores["human_reconnection"]


def test_verified_tactic_is_revoked_when_later_sessions_turn_negative():
    intervention_id = "human_reconnection"
    goal = "把他重新连接到一个真实的人或共同活动"
    persona = PersonaProfile(
        user_id="outcome-reversal",
        intervention_history=[
            InterventionOutcomeRecord(
                at=f"2026-07-0{index}T10:00:00",
                session_id=f"session-{index}",
                session_turn=0,
                intervention_id=intervention_id,
                intervention_label="真实关系重连",
                goal=goal,
                outcome=outcome,
                signal="explicit_result",
            )
            for index, outcome in enumerate(
                ("positive", "positive", "negative", "negative"),
                start=1,
            )
        ],
    )
    persona.aria_strategy_hints.tactics = [
        f"经多次结果验证有效[{intervention_id}]:{goal}"
    ]

    ElderStateReasoner().finalize(
        EngineContext(
            user_id=persona.user_id,
            scene_id="companion_elder",
        ),
        persona,
    )

    assert not any(
        item.startswith(f"经多次结果验证有效[{intervention_id}]:")
        for item in persona.aria_strategy_hints.tactics
    )
