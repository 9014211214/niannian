"""Post-action grounding for real-world completion claims."""
from __future__ import annotations
from tests.memory_source_fixtures import relationship_evidence

import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import (
    PersonaProfile,
    RelationshipContextEntry,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.online_decision import uses_online_decision_protocol
from aria.interfaces.tools import ToolResult
from aria.scenes.companion_elder.delivery import (
    UNGROUNDED_EXTERNAL_EVENT_REPLY,
    UNVERIFIED_ACTION_REPLY,
    guard_external_delivery_response,
)
from aria.scenes.registry import get_scene


class _ReplyModel:
    def __init__(self, response: str):
        self.response = response

    def complete(self, messages):
        if uses_online_decision_protocol(messages):
            return json.dumps({
                "response": self.response,
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
                "selected_intervention_id": "",
                "uncertainty": 0.4,
            }, ensure_ascii=False)
        return self.response


class _ReminderAdapter:
    name = "medication_reminder"
    description = "测试提醒适配器"
    parameters = {"medication": "药物", "when": "时间"}

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(
            ok=self.ok,
            data={
                **kwargs,
                "persisted": self.ok,
                "executed": self.ok,
                "note": "测试适配器未执行" if not self.ok else "",
            },
        )


def _run(
    response: str,
    utterance: str,
    *,
    route: str = "fast",
    persona: PersonaProfile | None = None,
    tools=None,
    previous_receipt=None,
):
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=_ReplyModel(response),
        tools=list(tools or []),
    )
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    ctx.working_memory["_force_reply_model"] = route
    if previous_receipt is not None:
        ctx.working_memory["last_tool_result"] = previous_receipt
    output = engine.process_turn(ctx, utterance)
    return output, ctx


@pytest.mark.parametrize("route", ["fast", "fast_integrated"])
def test_all_reply_routes_block_agent_completion_without_receipt(route: str):
    output, ctx = _run(
        "我已经通知女儿了。",
        "今天天气不错。",
        route=route,
    )
    assert output == UNVERIFIED_ACTION_REPLY
    assert ctx.working_memory["delivery_response_guard"] == {
        "activated": True,
        "violations": ["unverified_external_action_completion"],
        "response_changed": True,
    }


@pytest.mark.parametrize("route", ["fast", "fast_integrated"])
def test_all_reply_routes_reject_unavailable_future_outreach(route):
    output, ctx = _run("我帮你问问老周这周末有没有空。", "今天天气不错。", route=route)
    assert "问问老周" not in output and "不能直接替你联系" in output
    trace = ctx.working_memory.get("online_turn_decision") or {}
    reasons = trace.get("delivery_guard", {}).get("violations", [])
    reasons += ctx.working_memory["delivery_response_guard"]["violations"]
    assert "unsupported_future_contact_offer" in reasons


def test_noncompletion_disclaimer_does_not_authorize_future_outreach():
    output, _ctx = _run(
        "我还没有实际执行，明天我会联系老周。", "今天天气不错。", route="fast",
    )
    assert "我会联系" not in output


@pytest.mark.parametrize("reply", [
    "我帮你问问老周。",
    "我已经通知女儿了。我帮你问问老周。",
    "女儿已经安排好复诊。我帮你问问老周。",
])
def test_final_contact_guard_keeps_parallel_hydration_status(reply):
    from aria.interfaces.skills import SkillProposal
    from aria.interfaces.reasoning import ActiveObservationPlan
    ctx = EngineContext(user_id="synthetic-contact", scene_id="companion_elder")
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode="wait")
    ctx.working_memory["skill_proposals"] = [SkillProposal(
        skill_id="hydration_support", intent="create_hydration_reminder",
        goal="喝水提醒", guidance="待确认", task_status="needs_confirmation",
        slots={"schedule": "20:00"},
    )]
    final = guard_external_delivery_response(ctx, reply, "八点提醒喝水。", None)
    assert "20:00" in final and "要现在创建吗" in final
    assert "已经创建" not in final


@pytest.mark.parametrize("route", ["fast", "fast_integrated"])
def test_current_success_receipt_allows_matching_completion_and_action(
    route: str,
):
    adapter = _ReminderAdapter(ok=True)
    visible = "我已经帮你把提醒设置好了。"
    response = (
        visible
        + '\n<action>{"tool":"medication_reminder","args":'
        '{"medication":"降压药","when":"明早"}}</action>'
    )
    output, ctx = _run(
        response,
        "明早提醒我吃降压药。",
        route=route,
        tools=[adapter],
    )
    assert output == visible
    assert len(adapter.calls) == 1
    assert ctx.working_memory["last_tool_result"]["ok"] is True
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


def test_failed_current_action_keeps_specific_failure_and_never_claims_success():
    adapter = _ReminderAdapter(ok=False)
    response = (
        "我已经帮你把提醒设置好了。"
        '\n<action>{"tool":"medication_reminder","args":'
        '{"medication":"降压药","when":"明早"}}</action>'
    )
    output, ctx = _run(
        response,
        "明早提醒我吃降压药。",
        tools=[adapter],
    )
    assert "还没有执行" in output
    assert "测试适配器未执行" in output
    assert ctx.working_memory["last_tool_result"]["ok"] is False
    assert ctx.working_memory["delivery_response_guard"]["activated"] is True


def test_consumed_previous_receipt_is_available_only_for_final_validation():
    receipt = {
        "tool": "medication_reminder",
        "ok": True,
        "data": {"persisted": True, "executed": True},
    }
    response = "我已经帮你把提醒设置好了。"
    output, ctx = _run(
        response,
        "嗯。",
        previous_receipt=receipt,
    )
    assert output == response
    assert "_consumed_tool_result_for_turn" not in ctx.working_memory
    assert "_delivery_response_candidate_for_turn" not in ctx.working_memory


def test_wrong_receipt_category_cannot_authorize_completion_claim():
    reminder_receipt = {
        "tool": "medication_reminder",
        "ok": True,
        "data": {"persisted": True, "executed": True},
    }
    output, _ctx = _run(
        "我已经通知女儿了。",
        "嗯。",
        previous_receipt=reminder_receipt,
    )
    assert output == UNVERIFIED_ACTION_REPLY


def test_notification_receipt_never_proves_relative_replied_or_acted():
    notification_receipt = {
        "tool": "notify_family",
        "ok": True,
        "data": {"notified": True, "executed": True},
    }
    output, ctx = _run(
        "女儿已经回复了，她已经安排好复诊。",
        "我今天有点累。",
        previous_receipt=notification_receipt,
    )
    assert output == UNGROUNDED_EXTERNAL_EVENT_REPLY
    assert "ungrounded_third_party_completion" in ctx.working_memory[
        "delivery_response_guard"
    ]["violations"]


@pytest.mark.parametrize(
    "uncertain_source",
    [
        "女儿可能已经回复了。",
        "如果女儿明天回复，我就告诉你。",
        "听说女儿已经安排了复诊。",
    ],
)
def test_uncertain_or_hypothetical_source_cannot_ground_completed_event(
    uncertain_source: str,
):
    output, _ctx = _run(
        "女儿已经回复了。",
        uncertain_source,
    )
    assert output == UNGROUNDED_EXTERNAL_EVENT_REPLY


@pytest.mark.parametrize(
    "utterance",
    [
        "女儿已经回复了，她已经安排好复诊。",
        "我女儿刚才回复了，复诊已经约好了。",
    ],
)
def test_current_user_statement_grounds_third_party_completion(
    utterance: str,
):
    output, ctx = _run(utterance, utterance)
    assert output == utterance
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


def test_referenceable_relationship_fact_grounds_third_party_completion():
    fact = "陈小芳已经回复了，她安排好了复诊。"
    persona = PersonaProfile(
        user_id="u",
        relationship_context=[RelationshipContextEntry(
            id="daughter",
            name="陈小芳",
            relation="女儿",
            facts=[fact],
            referenceable=True,
            first_observed_at="2026-09-01T08:00:00",
            evidence=relationship_evidence(fact, context="女儿" + fact),
        )],
    )
    response = "女儿已经回复了，她安排好了复诊。"
    output, ctx = _run(response, "嗯。", persona=persona)
    assert output == response
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


def test_negated_user_statement_does_not_ground_positive_third_party_claim():
    output, _ctx = _run(
        "女儿已经回复了。",
        "女儿还没有回复。",
    )
    assert output == UNGROUNDED_EXTERNAL_EVENT_REPLY


def test_asr_without_punctuation_keeps_actor_across_consecutive_events():
    response = "女儿已经回复了她已经安排好复诊了"
    blocked, _ctx = _run(response, "我今天有点累")
    assert blocked == UNGROUNDED_EXTERNAL_EVENT_REPLY

    grounded, ctx = _run(response, response)
    assert grounded == response
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


@pytest.mark.parametrize(
    "contradiction",
    [
        {"executed": False},
        {"persisted": False},
        {"notified": False},
        {"delivered": False},
    ],
)
def test_ok_receipt_with_explicit_false_field_is_not_success(contradiction):
    receipt = {
        "tool": "notify_family",
        "ok": True,
        "args": {"contact": "女儿"},
        "data": contradiction,
    }
    output, _ctx = _run(
        "我已经通知女儿了。",
        "嗯。",
        previous_receipt=receipt,
    )
    assert output == UNVERIFIED_ACTION_REPLY


def test_notification_receipt_must_match_named_target():
    persona = PersonaProfile(
        user_id="u",
        relationship_context=[
            RelationshipContextEntry(
                id="daughter",
                name="陈小芳",
                relation="女儿",
                facts=["陈小芳是我女儿。"],
                referenceable=True,
                first_observed_at="2026-09-01T08:00:00",
                evidence=relationship_evidence("陈小芳是我女儿。"),
            ),
            RelationshipContextEntry(
                id="carer",
                name="王护理员",
                relation="护理员",
                facts=["王护理员是负责我的护理员。"],
                referenceable=True,
                first_observed_at="2026-09-01T08:00:00",
                evidence=relationship_evidence("王护理员是负责我的护理员。"),
            ),
        ],
    )
    receipt = {
        "tool": "notify_family",
        "ok": True,
        "args": {"contact": "王护理员"},
        "data": {"notified": True, "status": "delivered"},
    }
    blocked, _ctx = _run(
        "我已经通知女儿了。",
        "嗯。",
        persona=persona,
        previous_receipt=receipt,
    )
    assert blocked == UNVERIFIED_ACTION_REPLY

    allowed, ctx = _run(
        "我已经通知王护理员了。",
        "嗯。",
        persona=persona,
        previous_receipt=receipt,
    )
    assert allowed == "我已经通知王护理员了。"
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


@pytest.mark.parametrize("status", ["accepted", "delivered_local", "sent_local"])
def test_non_terminal_or_local_receipt_never_claims_real_delivery(status: str):
    receipt = {
        "tool": "notify_family",
        "ok": True,
        "args": {"contact": "女儿"},
        "data": {"status": status},
    }
    output, _ctx = _run(
        "消息已经发出了。",
        "嗯。",
        previous_receipt=receipt,
    )
    assert output == UNVERIFIED_ACTION_REPLY


def test_terminal_delivery_receipt_can_support_message_sent_claim():
    receipt = {
        "tool": "notify_family",
        "ok": True,
        "args": {"contact": "女儿"},
        "data": {"notified": True, "status": "delivered"},
    }
    response = "消息已经发出了。"
    output, ctx = _run(response, "嗯。", previous_receipt=receipt)
    assert output == response
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


@pytest.mark.parametrize(
    "recipient_claim",
    ["对方已经收到消息了。", "对方已经看见消息了。", "对方已经已读。"],
)
def test_delivery_receipt_never_proves_recipient_received_or_read(
    recipient_claim: str,
):
    receipt = {
        "tool": "notify_family",
        "ok": True,
        "args": {"contact": "女儿"},
        "data": {"notified": True, "status": "delivered"},
    }
    output, _ctx = _run(
        recipient_claim,
        "嗯。",
        previous_receipt=receipt,
    )
    assert output == UNGROUNDED_EXTERNAL_EVENT_REPLY


def test_user_statement_can_ground_recipient_received_claim():
    response = "女儿已经收到消息了。"
    output, ctx = _run(response, response)
    assert output == response
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False


def test_deterministic_skill_receipt_allows_only_matching_reminder_claim():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    ctx.working_memory["last_care_action"] = {
        "ok": True,
        "action": "reminder_created",
    }
    response = "喝水提醒已经设置好了。"
    assert guard_external_delivery_response(
        ctx, response, "确认创建喝水提醒。", None
    ) == response
    assert guard_external_delivery_response(
        ctx, "我已经预约好复诊了。", "嗯。", None
    ) == UNVERIFIED_ACTION_REPLY


def test_explicit_non_completion_is_not_mistaken_for_success():
    response = "我还没有通知女儿，需要先得到你的同意。"
    output, ctx = _run(response, "嗯。")
    assert output == response
    assert ctx.working_memory["delivery_response_guard"]["activated"] is False
