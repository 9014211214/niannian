"""Finite delivery/measurement regressions, not semantic-quality benchmark gold."""
from __future__ import annotations

import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.engine.policy.reply_contract import (
    contract_prompt,
    quality_violations,
    reply_quality_contract,
)
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate


CUP_UTTERANCE = "老伴那只旧茶杯我还没收起来，放那儿也不碍事。"


def _context(*, mode="observe", ids=(), physical=False):
    ctx = EngineContext(user_id="synthetic-fairness", scene_id="companion_elder")
    ctx.reasoning.observation_plan = ActiveObservationPlan(
        mode=mode, hypothesis_ids=list(ids),
        question="你更想说说发生的事，还是先让我陪你待一会儿？" if mode == "ask" else "",
    )
    candidate = InterventionCandidate(
        id="physical_clarification" if physical else "meaning_oriented_companionship",
        label="陪伴", goal="由本人决定要不要说", guidance="可拒绝的邀请",
    )
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.selected = candidate
    ctx.working_memory["personality_decision"] = {
        "current_physical_evidence_qualified": physical,
    }
    return ctx


def _deliver(ctx, reply, utterance=CUP_UTTERANCE):
    return apply_single_pass_decision(json.dumps({
        "response": reply, "state_signals": [],
        "selected_intervention_id": ctx.reasoning.selected.id,
    }, ensure_ascii=False), ctx, utterance)


@pytest.mark.parametrize("reply", [
    "那只杯子先放着也可以。",
    "想聊聊这只杯子吗？不想聊也没关系。",
    "那只旧茶杯先放着，不用急着决定。",
])
def test_known_tea_cup_can_be_referred_to_as_cup(reply):
    assert "unsupported_concrete_referent" not in quality_violations(
        reply, [CUP_UTTERANCE], reply_quality_contract(_context()),
    )


@pytest.mark.parametrize("current", [
    "杯子已经丢了。", "我没有杯子了。", "茶杯已经送走了。",
    "假如那个杯子还在就好了。", "如果它还在，我会留着那个杯子。",
    "那是故事里的茶杯。",
])
def test_alias_cannot_resurrect_a_missing_or_hypothetical_current_object(current):
    assert "unsupported_concrete_referent" in quality_violations(
        "那只杯子先放着。", [current, "我有一只茶杯。"],
        reply_quality_contract(_context()),
    )


def test_general_cup_does_not_establish_a_tea_cup_but_denial_covers_the_family():
    contract = reply_quality_contract(_context())
    assert "unsupported_concrete_referent" in quality_violations(
        "那只茶杯先放着。", ["桌上有个杯子。"], contract,
    )
    assert "unsupported_concrete_referent" in quality_violations(
        "那只茶杯先放着。", ["杯子已经扔掉了。", "茶杯还在。"], contract,
    )
    assert "unsupported_concrete_referent" in quality_violations(
        "那只杯子先放着。", ["今天收拾了一会儿柜子。"], contract,
    )


def test_quoted_fiction_is_not_a_source_or_a_denial_of_an_independent_real_object():
    contract = reply_quality_contract(_context())
    fiction = '电视剧里说“我的杯子还在桌上”。'
    assert "unsupported_concrete_referent" in quality_violations(
        "那只杯子先放着。", [fiction], contract,
    )
    assert "unsupported_concrete_referent" not in quality_violations(
        "那只杯子先放着。", [fiction, "我有一只茶杯。"], contract,
    )


@pytest.mark.parametrize("reply", [
    "需要我帮您把它擦干净吗？",
    "如果你愿意，我可以帮您把那只旧茶杯擦干净。",
    "我可以替你清洗杯子。",
    "我帮你把桌子搬走。",
    "我不能擦杯子，但我可以帮您把它擦干净。",
])
def test_object_grounding_and_consent_do_not_create_physical_help_capability(reply):
    ctx = _context(mode="ask", ids=("grief_or_transition",))
    assert "unsupported_agent_physical_help" in quality_violations(
        reply, [CUP_UTTERANCE], reply_quality_contract(ctx),
    )
    final = _deliver(ctx, reply)
    assert final != reply
    assert "擦干净" not in final and "清洗" not in final and "搬走" not in final
    assert "去不去" not in final


@pytest.mark.parametrize("reply", [
    "我不能替你清洗杯子。",
    "我可以帮您想想怎么清洗杯子。",
    "我可以教你把它擦干净。",
    "我可以帮您写一句：‘请把它擦干净。’",
    "你可以请护理员帮忙擦干净。",
    '你说“我帮你把它擦干净”，这是你刚才的原话。',
    "假设以后有机械臂，我可以把它擦干净；现在并没有。",
    "我可以帮你清洗数据。",
])
def test_physical_help_guard_distinguishes_drafting_denial_quotes_and_user_action(reply):
    assert "unsupported_agent_physical_help" not in quality_violations(
        reply, [CUP_UTTERANCE], reply_quality_contract(_context()),
    )


@pytest.mark.parametrize("reply", [
    "留着也可以，什么时候想说，我就在这儿听。",
    "想聊聊这只杯子吗？不想聊也没关系。",
])
def test_grief_permission_is_optional_and_does_not_force_a_fixed_question(reply):
    ctx = _context(mode="ask", ids=("grief_or_transition",))
    contract = reply_quality_contract(ctx)
    assert contract["question_requirement"] == "invitation_optional"
    assert "不强制逐字复述" in contract_prompt(contract)
    assert _deliver(ctx, reply) == reply
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"] is False


def test_optional_permission_does_not_depend_on_persona_or_model_selected_route():
    ctx = _context(mode="ask", ids=("grief_or_transition",))
    initial = reply_quality_contract(ctx)["question_requirement"]
    ctx.working_memory["personality_decision"]["evidence"] = [{
        "kind": "response_hypothesis", "status": "working",
        "dimension": "meaning_memory", "claim": "愿意保留物品慢慢说。",
    }]
    ctx.reasoning.selected.id = "identity_based_reengagement"
    assert reply_quality_contract(ctx)["question_requirement"] == initial == "invitation_optional"


@pytest.mark.parametrize("ids", [(), ("physical_discomfort",), ("grief_or_transition", "physical_discomfort")])
def test_other_fact_questions_are_not_silently_relaxed(ids):
    ctx = _context(mode="ask", ids=ids)
    assert reply_quality_contract(ctx)["question_requirement"] == "exact"
    final = _deliver(ctx, "我听见了。")
    assert ctx.reasoning.observation_plan.question in final
    assert "去不去" not in final


def test_qualified_physical_evidence_keeps_exact_question_even_with_grief_context():
    ctx = _context(mode="ask", ids=("grief_or_transition",), physical=True)
    ctx.reasoning.observation_plan.question = "这个不舒服是刚刚才有，还是已经持续一阵了？"
    assert reply_quality_contract(ctx)["question_requirement"] == "exact"
    final = _deliver(ctx, "我先陪你。", "我肚子一直不舒服。")
    assert ctx.reasoning.observation_plan.question in final
    assert "clarification_question_omitted" in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]


def test_optional_invitation_is_not_permission_for_two_questions():
    ctx = _context(mode="ask", ids=("grief_or_transition",))
    _deliver(ctx, "想聊聊这只杯子吗？还想聊其他的吗？")
    assert "multiple_questions_during_optional_invitation" in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]


@pytest.mark.parametrize("mode", ["wait", "observe"])
def test_wait_and_person_space_still_disallow_optional_companionship_questions(mode):
    ctx = _context(mode=mode, ids=("grief_or_transition",))
    if mode == "observe":
        ctx.working_memory["personality_decision"]["evidence"] = [{
            "kind": "response_hypothesis", "status": "working", "dimension": "low_pressure",
            "claim": "一次说明后留白，不要求解释。",
        }]
    contract = reply_quality_contract(ctx)
    assert contract["question_requirement"] == "none"
    assert contract["question_budget"] == 0
    final = _deliver(ctx, "想聊聊这只杯子吗？不想聊也可以。")
    assert "？" not in final


def test_alias_permission_does_not_disable_existing_contact_or_dignity_guards():
    contract = reply_quality_contract(_context())
    assert "unsupported_future_contact_offer" in quality_violations(
        "我帮你问问护理员。", [CUP_UTTERANCE], contract,
    )
    assert "dignity_undermining_agreement" in quality_violations(
        "这种活确实不适合你做。", [CUP_UTTERANCE], contract,
    )
