"""Synthetic semantic labels verify permission transport, never NLU accuracy."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json

import pytest

from aria.engine.cognitive import apply_cognitive_withdrawal, cognitive_extraction_prompt, extract_cognitive_rows
from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import apply_single_pass_decision, single_pass_messages, online_decision_tool_spec
from tests.test_cognitive_change_v2 import T0, enabled, context_with


def decision(text, action="disable_observation"):
    return {"response": "你可在面板核对保存状态。", "state_signals": [],
            "current_affect": {"label": "neutral", "confidence": .5,
                               "divergence": {"possible": False, "confidence": 0, "pattern": ""}},
            "selected_intervention_id": "", "communication_plan": {
                "pace": "brief_exchange", "support_order": "回应当前要求", "boundary": "只减少权限"},
            "safety_concern": "none", "uncertainty": .5,
            "cognitive_control": {"version": "cognitive_permission_v1", "action": action,
                                  "evidence_quote": text if action == "disable_observation" else ""}}


def parsed(persona, text, value=None):
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v3"
    apply_single_pass_decision(json.dumps(value or decision(text), ensure_ascii=False), ctx, text, persona)
    return ctx


def test_valid_current_owner_reduction_clears_raw_and_retains_saved_records():
    p = enabled()
    text = "停止观察我的日常变化。"  # Development example, not sealed item.
    ctx = parsed(p, text)
    ctx.working_memory["_cognitive_source_turns"] = [{"turn_index": 0, "text": "private"}]
    assert apply_cognitive_withdrawal(ctx, p, text, now=T0 + timedelta(days=1))
    assert not p.cognitive_health_consent.has_text_monitoring_consent()
    assert "_cognitive_source_turns" not in ctx.working_memory
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("mutation", ["old_turn", "wrong_owner", "simulation", "wrong_hash", "old_version", "not_parsed"])
def test_stale_or_unauthorized_control_cannot_reduce(mutation):
    p = enabled(); text = "停止观察我的日常变化。"; ctx = parsed(p, text)
    trace = ctx.working_memory["online_turn_decision"]
    if mutation == "old_turn": ctx.add_turn("earlier", "earlier")
    elif mutation == "wrong_owner": ctx.user_id = "caregiver"
    elif mutation == "simulation": ctx.simulation_mode = True
    elif mutation == "wrong_hash": trace["owner_message_sha256"] = "0" * 64
    elif mutation == "old_version": trace["version"] = "online_turn_decision_v2"
    else: trace["status"] = "invalid_protocol_response"
    assert not apply_cognitive_withdrawal(ctx, p, text)
    assert p.cognitive_health_consent.has_text_monitoring_consent()


@pytest.mark.parametrize("control", [None, "disable_observation", {},
    {"version": "cognitive_permission_v1", "action": [], "evidence_quote": ""},
    {"version": "cognitive_permission_v1", "action": {}, "evidence_quote": ""},
    {"version": "cognitive_permission_v1", "action": "enable", "evidence_quote": ""},
    {"version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": "片段"},
    {"version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": "当前原话", "extra": True}])
def test_invalid_control_fails_closed(control):
    p = enabled(); text = "当前原话"; value = decision(text); value["cognitive_control"] = control
    ctx = parsed(p, text, value)
    assert ctx.working_memory["online_turn_decision"]["cognitive_control_status"] == "missing_or_invalid"
    assert not apply_cognitive_withdrawal(ctx, p, text)


def test_no_permission_change_label_does_not_apply_keyword_withdrawal():
    p = enabled(); text = "我不是说不要再提醒我的记性。"
    ctx = parsed(p, text, decision(text, "no_permission_change"))
    assert not apply_cognitive_withdrawal(ctx, p, text)
    assert p.cognitive_health_consent.has_text_monitoring_consent()


def test_schema_and_prompt_bound_current_semantics_without_enable_share_power():
    schema = online_decision_tool_spec("online_turn_decision_v3")["function"]["parameters"]
    assert schema["properties"]["cognitive_control"]["properties"]["action"]["enum"] == ["no_permission_change", "disable_observation"]
    assert "cognitive_control" not in online_decision_tool_spec("online_turn_decision_v1")["function"]["parameters"]["properties"]
    p = enabled(); ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    messages = single_pass_messages([{"role": "system", "content": "trusted"}], ctx,
                                    version="online_turn_decision_v3", persona=p)
    assert "C1本人权限语义" in messages[0]["content"]


def test_empty_personality_ledger_accepts_only_third_literal_same_event_in_c1():
    p = enabled(); words = "早饭后的报纸放在哪里了？"
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    from aria.engine.cognitive import capture_cognitive_turn
    for index in range(3):
        capture_cognitive_turn(ctx, p, words, now=T0 + timedelta(days=1, minutes=index))
        ctx.add_turn(words, "回应")
    rows = extract_cognitive_rows([{"domain": "repetition", "turn_index": index,
             "evidence_quote": words, "event_kind": "same_event"} for index in range(3)], p, ctx)
    assert [row.counting for row in rows] == [False, False, True]
    assert len({row.event_id for row in rows}) == 1 and not p.evidence_review_ledger
    assert all(row.event_id.startswith("cog_event_") for row in rows)
    assert "已知同一事件出处" not in cognitive_extraction_prompt(p, ctx)[1]


def test_c1_same_event_prompt_uses_private_c1_records_only():
    p = enabled(); ctx = context_with(p)
    row = extract_cognitive_rows([{"domain": "repetition", "turn_index": 0,
             "evidence_quote": ctx.turns[0].user, "event_kind": "same_event"}], p, ctx)[0]
    p.cognitive_observations.append(deepcopy(row))
    assert row.event_id in cognitive_extraction_prompt(p, ctx)[1]
    assert not p.evidence_review_ledger
