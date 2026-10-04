"""Finite delivery regressions and negative controls; not empathy accuracy."""
from __future__ import annotations

import json

import pytest

from aria.engine.context import EngineContext, Turn
from aria.engine.policy.online_decision import apply_single_pass_decision, single_pass_messages
from aria.engine.policy.reply_contract import quality_violations, reply_quality_contract
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate


def _context(*, space=False, physical=False):
    ctx = EngineContext(user_id="synthetic-contract", scene_id="companion_elder")
    candidate = InterventionCandidate(
        id="identity_based_reengagement", label="参与", goal="保留真实作用",
        guidance="可拒绝的小邀请", score=0.7,
    )
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.selected = candidate
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode="observe")
    ctx.working_memory["personality_decision"] = {
        "current_physical_evidence_qualified": physical,
        "evidence": [{
            "kind": "response_hypothesis", "dimension": "low_pressure",
            "claim": "普通邀请时一次说明后留白", "status": "working",
        }] if space else [],
    }
    return ctx


@pytest.mark.parametrize(("reply", "reason"), [
    ("我这里有一把螺丝刀，拿去试试。", "unsupported_agent_physical_presence"),
    ("如果你愿意，我手里有个木盒。", "unsupported_agent_physical_presence"),
    ("可以把那本相册翻开。", "unsupported_concrete_referent"),
    ("要不要修那个旧木盒？", "unsupported_concrete_referent"),
    ("在旁边看着反而容易碍事。", "dignity_undermining_agreement"),
    ("这种粗活儿确实不适合你插手。", "dignity_undermining_agreement"),
    ("是身上不舒服，还是觉得不想动弹？", "diagnostic_probe_while_observing"),
])
def test_known_delivery_failures_are_explicit_not_quality_scores(reply, reason):
    assert reason in quality_violations(reply, [], reply_quality_contract(_context()))


@pytest.mark.parametrize("reply", [
    "你担心自己会碍事，我听到了。",
    "不代表你确实没用。",
    "我不觉得你真的没用。",
    "你不是只会添乱。",
    "你说“我这里有块木料”，想怎么处理由你决定。",
    "如果有一本相册，也许可以选一张你喜欢的照片。",
    "假如我这里有木头，这只是举例，不代表现场有材料。",
    "我这里有一个建议，不想试也没关系。",
    "想聊聊刚才提到的花吗？不想说也没关系。",
    "你喜欢喝茶还是喝水？",
])
def test_negation_quotes_hypothetical_and_daily_invitation_controls(reply):
    assert quality_violations(reply, [], reply_quality_contract(_context())) == []


@pytest.mark.parametrize("current", ["我没有木盒。", "如果我有木盒就好了。", "木盒已经扔掉了。",
                                      "如果以后能找回来，我就修木盒。"])
def test_current_denial_or_hypothetical_cannot_be_grounded_by_old_mention(current):
    assert "unsupported_concrete_referent" in quality_violations(
        "你可以修那个旧木盒。", [current, "我有一个木盒。"],
        reply_quality_contract(_context()),
    )


def test_fictional_quotation_is_not_a_physical_object_source():
    assert "unsupported_concrete_referent" in quality_violations(
        "修那个旧木盒。", ['那是电视剧里说的“我有一个木盒”。'],
        reply_quality_contract(_context()),
    )


def test_current_user_fact_allows_reference_but_agent_history_does_not():
    ctx = _context()
    ctx.turns = [Turn(user="今天随便聊聊。", agent="有个木盒。")]
    payload = json.dumps({"response": "可以修那个旧木盒。", "state_signals": [],
                          "selected_intervention_id": "identity_based_reengagement"})
    assert "旧木盒" not in apply_single_pass_decision(payload, ctx, "我今天不想去。")
    assert "旧木盒" in apply_single_pass_decision(payload, _context(), "我有一个旧木盒。")


def test_space_contract_stops_personal_questions_but_physical_evidence_overrides():
    ctx = _context(space=True)
    contract = reply_quality_contract(ctx)
    assert contract["question_budget"] == 0
    assert "question_overrode_person_space" in quality_violations(
        "想聊聊花吗？", [], contract,
    )
    assert reply_quality_contract(_context(space=True, physical=True))["question_budget"] == 1


@pytest.mark.parametrize("space", [True, False])
def test_person_space_does_not_swallow_task_confirmation_even_without_model_question(space):
    from aria.interfaces.skills import SkillProposal
    ctx = _context(space=space)
    ctx.working_memory["skill_proposals"] = [SkillProposal(
        skill_id="hydration_support", intent="create_hydration_reminder",
        goal="喝水提醒", guidance="等待确认", task_status="needs_confirmation",
        slots={"schedule": "20:00"},
    )]
    payload = json.dumps({"response": "可以修那个旧木盒。", "state_signals": [],
                          "selected_intervention_id": "identity_based_reengagement"})
    final = apply_single_pass_decision(payload, ctx, "先别追问，晚上八点提醒喝水。")
    assert "20:00" in final and "要现在创建吗" in final
    assert "已经创建" not in final
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["task_communication"]["mode"] == "ask"


def test_original_reply_capture_is_explicit_and_removed_on_invalid_next_turn():
    ctx = _context()
    payload = json.dumps({"response": "我这里有一块木料。", "state_signals": [],
                          "selected_intervention_id": "identity_based_reengagement"})
    apply_single_pass_decision(payload, ctx, "我帮不上什么了。")
    assert "_benchmark_reply_trace" not in ctx.working_memory
    ctx.working_memory["_benchmark_capture_reply_trace"] = True
    final = apply_single_pass_decision(payload, ctx, "我帮不上什么了。")
    trace = ctx.working_memory["_benchmark_reply_trace"]
    assert trace["pre_guard_response"] == "我这里有一块木料。"
    assert trace["delivered_response"] == final
    assert trace["guard_applied"] is True
    assert "木料" not in final
    apply_single_pass_decision('{"broken', ctx, "下一轮。")
    assert "_benchmark_reply_trace" not in ctx.working_memory


@pytest.mark.parametrize("reply", [
    "我帮你问问老周周末有没有空。",
    "如果你愿意我可以替你联系朋友。",
    "要是你同意，我这就打电话。",
    "我会通知护理员。",
    "我可以尝试通知护理员。",
    "我替你打听一下。",
    "我还没有实际执行，稍后我会联系老周。",
    "我帮你问问老周就行了。",
    "我不能打电话但我可以帮你问问老周。",
    "我这就联系老周了。",
])
def test_consent_and_tentative_language_do_not_create_contact_capability(reply):
    assert "unsupported_future_contact_offer" in quality_violations(
        reply, ["用户同意联系。"], reply_quality_contract(_context()),
    )


@pytest.mark.parametrize("reply", [
    "你愿意的话，可以自己问问老周。",
    "我可以陪你斟酌怎么开口。",
    "我问问你今天想聊些什么。",
    "我不能替你联系对方。",
    "你说“我帮你问问老周”，这是你之前的原话。",
    "假设将来接通通知功能，我可以尝试通知工作人员；现在还没有接通。",
    "我联系女儿了。",  # Success is separately checked against receipts.
])
def test_drafting_user_questions_negation_and_fiction_are_not_outreach(reply):
    assert "unsupported_future_contact_offer" not in quality_violations(
        reply, [], reply_quality_contract(_context()),
    )


def test_advertised_notification_only_allows_attempt_not_call_or_guarantee():
    ctx = _context()
    ctx.working_memory["_advertised_tools_for_turn"] = ["notify_family:参考通知器"]
    contract = reply_quality_contract(ctx)
    assert quality_violations("我可以尝试通知护理员。", [], contract) == []
    for reply in ("我会通知护理员。", "我可以尝试联系老周。", "我帮你问问老周。"):
        assert "unsupported_future_contact_offer" in quality_violations(reply, [], contract)
    ctx.working_memory["_advertised_tools_for_turn"] = ["other:notify_family"]
    assert not reply_quality_contract(ctx)["notification_attempt_advertised"]


@pytest.mark.parametrize("wait", [False, True])
def test_contact_guard_keeps_wait_and_same_turn_task(wait):
    from aria.interfaces.skills import SkillProposal
    ctx = _context()
    if wait:
        ctx.reasoning.observation_plan = ActiveObservationPlan(mode="wait")
    ctx.working_memory["skill_proposals"] = [SkillProposal(
        skill_id="hydration_support", intent="create_hydration_reminder",
        goal="喝水提醒", guidance="待确认", task_status="needs_confirmation",
        slots={"schedule": "20:00"},
    )]
    payload = json.dumps({"response": "我帮你问问老周。", "state_signals": [],
                          "selected_intervention_id": "identity_based_reengagement"})
    final = apply_single_pass_decision(payload, ctx, "先陪我，八点提醒我喝水。")
    assert "20:00" in final and "问问老周" not in final
    if wait:
        assert "斟酌" not in final
    assert final.count("？") == 1  # Only the explicitly requested task.


def test_prompt_distinguishes_guidance_from_observation_and_capability_evidence():
    from aria.engine.policy.prompt import build_messages
    from aria.scenes.registry import get_scene
    ctx = _context(space=True)
    ctx.emotion.divergence = True
    ctx.emotion.pattern = "口头没事，但描述参与减少"
    prompt = single_pass_messages(
        build_messages(get_scene("companion_elder"), None, ctx, "今天先不去了。"), ctx,
    )[0]["content"]
    assert "按口是心非处理" not in prompt
    assert "低置信时先澄清" not in prompt
    assert "观察/留白许可优先" in prompt
    assert "不代表他说的不是真心话" in prompt
    assert "不因年龄、手艺身份或活动类型推断体力" in prompt
    assert "征得同意也不会凭空接通能力" in prompt
