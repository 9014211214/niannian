"""online_turn_decision_v2：推演在前、response 在最后的可选在线协议。

全部使用本地确定性回复，只检查协议、校验与 trace；不代表推演改善了回复质量，
也不证明模型真的先推演后作答（字段顺序只是降低“先写回复再补理由”的机会）。
"""
from __future__ import annotations

import io
import json
import urllib.request

import pytest

from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.model.persona import PersonHypothesis
from aria.engine.policy import PolicyEngine
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.online_decision import (
    DELIBERATION_VERSION,
    ONLINE_DECISION_VERSION,
    ONLINE_DECISION_VERSION_ENV,
    ONLINE_DECISION_VERSION_V2,
    apply_single_pass_decision,
    configured_online_decision_version,
    online_decision_tool_spec,
    protocol_version,
    single_pass_messages,
    validate_deliberation,
)
from tests.test_online_communication_plan import _PLAN, _ROUTE, _context

_IDS = {_ROUTE, "quiet_companionship", "physical_clarification"}
_MESSAGES = [
    {"role": "system", "content": "基础提示"},
    {"role": "user", "content": "今天有点闷。"},
]


def _persona():
    persona = load_demo_persona("laochen_demo")
    persona.person_hypotheses.append(PersonHypothesis(
        id="h_ask_first", framework="response", claim="涉及自己的安排，先问再定",
        confidence=0.7, status="working", scope_tags=["general"],
        first_observed_at="2026-09-01T09:00:00+08:00",
    ))
    return persona


def _deliberation(**overrides):
    value = {
        "person_read": "想自己拿主意，不想被安排",
        "applied_basis": ["agency", "h_ask_first"],
        "options": [
            {"intervention_id": _ROUTE, "predicted_reaction": "可能愿意说说以前的手艺"},
            {"intervention_id": "quiet_companionship", "predicted_reaction": "未知"},
        ],
    }
    value.update(overrides)
    return value


def _payload(**deliberation_overrides):
    return {
        "state_signals": [],
        "current_affect": {
            "label": "neutral", "confidence": 0.4,
            "divergence": {"possible": False, "confidence": 0.0, "pattern": ""},
        },
        "deliberation": _deliberation(**deliberation_overrides),
        "selected_intervention_id": _ROUTE,
        "communication_plan": dict(_PLAN),
        "uncertainty": 0.4,
        "response": "那本旧本子还在吗？想翻就翻翻。",
    }


def _v2_ctx(persona=None):
    ctx = _context()
    single_pass_messages(
        _MESSAGES, ctx, version=ONLINE_DECISION_VERSION_V2,
        persona=persona or _persona(), delivery_block="profile_v1",
    )
    return ctx


def test_v1_spec_is_unchanged_and_v2_puts_response_last():
    v1 = online_decision_tool_spec()["function"]["parameters"]
    assert list(v1["properties"])[0] == "response"
    assert "deliberation" not in v1["properties"]
    assert online_decision_tool_spec(ONLINE_DECISION_VERSION) == online_decision_tool_spec()
    v2 = online_decision_tool_spec(ONLINE_DECISION_VERSION_V2)["function"]["parameters"]
    order = list(v2["properties"])
    assert order[-1] == "response"
    assert order.index("deliberation") < order.index("selected_intervention_id")
    assert v2["required"] == order
    assert v2["properties"]["deliberation"]["additionalProperties"] is False
    with pytest.raises(ValueError):
        online_decision_tool_spec("online_turn_decision_v9")


def test_protocol_version_requires_trusted_explicit_declaration():
    assert protocol_version([{"content": ONLINE_DECISION_VERSION}]) is None
    for version in (ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2):
        assert protocol_version([{"role": "system", "content": "协议版本：" + version}]) == version
    assert protocol_version([{"content": "普通请求"}]) is None
    with pytest.raises(ValueError, match="conflicting_online_protocol"):
        protocol_version([{"role": "system", "content": "协议版本：" + version}
                          for version in (ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2)])


def test_v1_prompt_ignores_persona_argument():
    plain = single_pass_messages(_MESSAGES, _context())
    with_persona = single_pass_messages(_MESSAGES, _context(), persona=_persona())
    assert plain == with_persona
    assert ONLINE_DECISION_VERSION in plain[0]["content"]


def test_v2_prompt_lists_offline_basis_and_asks_response_last():
    ctx = _context()
    messages = single_pass_messages(
        _MESSAGES, ctx, version=ONLINE_DECISION_VERSION_V2, persona=_persona(),
        delivery_block="profile_v1",
    )
    content = messages[0]["content"]
    assert ONLINE_DECISION_VERSION_V2 in content
    assert ONLINE_DECISION_VERSION not in content
    assert DELIBERATION_VERSION in content
    assert content.index('"deliberation"') < content.index('"response"')
    assert "h_ask_first" in content and '"agency"' in content
    assert "不重算 DISC" in content
    assert "API有aria_turn_decision时" not in content
    assert ctx.working_memory["_online_decision_version"] == ONLINE_DECISION_VERSION_V2
    basis = ctx.working_memory["_online_decision_basis_ids"]
    assert basis[:5] == ["pace", "agency", "structure", "closeness", "warmth"]
    assert basis[-1] == "h_ask_first" and len(basis) <= 5 + 4
    assert messages[1:] == _MESSAGES[1:]
    with pytest.raises(ValueError):
        single_pass_messages(_MESSAGES, _context(), version="bad")


def test_v2_prompt_without_person_understanding_allows_empty_basis():
    ctx = _context()
    content = single_pass_messages(
        _MESSAGES, ctx, version=ONLINE_DECISION_VERSION_V2, persona=None,
    )[0]["content"]
    assert "applied_basis 给空数组" in content
    assert ctx.working_memory["_online_decision_basis_ids"] == []


@pytest.mark.parametrize("overrides", [
    {"applied_basis": ["made_up_basis"]},
    {"applied_basis": ["agency", "agency"]},
    {"person_read": "长" * 41},
    {"options": []},
    {"options": [{"intervention_id": "send_message", "predicted_reaction": "会高兴"}]},
    {"options": [{"intervention_id": _ROUTE, "predicted_reaction": "长" * 31}]},
    {"options": [
        {"intervention_id": _ROUTE, "predicted_reaction": "未知"},
        {"intervention_id": _ROUTE, "predicted_reaction": "未知"},
    ]},
    {"options": [{"intervention_id": _ROUTE, "predicted_reaction": "未知", "x": 1}]},
    {"extra": "field"},
])
def test_invalid_deliberation_is_rejected_whole(overrides):
    value = _deliberation(**overrides)
    assert validate_deliberation(
        value, intervention_ids=_IDS, basis_ids={"agency", "h_ask_first"},
    ) is None


def test_valid_deliberation_is_copied():
    value = _deliberation()
    copied = validate_deliberation(
        value, intervention_ids=_IDS, basis_ids={"agency", "h_ask_first"},
    )
    assert copied == value and copied is not value
    empty = validate_deliberation(
        _deliberation(applied_basis=[]), intervention_ids=_IDS, basis_ids=set(),
    )
    assert empty is not None


def test_v2_apply_records_captured_deliberation():
    ctx = _v2_ctx()
    reply = apply_single_pass_decision(
        json.dumps(_payload(), ensure_ascii=False), ctx, "今天有点闷。",
    )
    assert reply == "那本旧本子还在吗？想翻就翻翻。"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["version"] == ONLINE_DECISION_VERSION_V2
    assert trace["deliberation_status"] == "captured"
    assert trace["deliberation"]["applied_basis"] == ["agency", "h_ask_first"]
    assert trace["selected_in_deliberated_options"] is True


def test_v2_apply_keeps_reply_when_deliberation_missing_or_invalid():
    ctx = _v2_ctx()
    payload = _payload()
    payload.pop("deliberation")
    assert apply_single_pass_decision(json.dumps(payload), ctx, "今天有点闷。")
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["deliberation_status"] == "missing"
    assert trace["deliberation"] is None
    assert trace["selected_in_deliberated_options"] is None

    ctx = _v2_ctx()
    payload = _payload(applied_basis=["invented"])
    assert apply_single_pass_decision(json.dumps(payload), ctx, "今天有点闷。")
    assert ctx.working_memory["online_turn_decision"]["deliberation_status"] == "invalid"


def test_v2_selection_outside_deliberated_options_is_visible():
    ctx = _v2_ctx()
    payload = _payload(options=[
        {"intervention_id": "quiet_companionship", "predicted_reaction": "未知"},
    ])
    apply_single_pass_decision(json.dumps(payload, ensure_ascii=False), ctx, "今天有点闷。")
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["deliberation_status"] == "captured"
    assert trace["selected_in_deliberated_options"] is False


def test_v1_apply_trace_has_no_deliberation_fields():
    ctx = _context()
    single_pass_messages(_MESSAGES, ctx)
    payload = _payload()
    payload.pop("deliberation")
    apply_single_pass_decision(json.dumps(payload, ensure_ascii=False), ctx, "今天有点闷。")
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["version"] == ONLINE_DECISION_VERSION
    assert "deliberation_status" not in trace


def test_version_switch_defaults_to_v1(monkeypatch):
    monkeypatch.delenv(ONLINE_DECISION_VERSION_ENV, raising=False)
    assert configured_online_decision_version() == ONLINE_DECISION_VERSION
    assert PolicyEngine().online_decision_version == ONLINE_DECISION_VERSION
    monkeypatch.setenv(ONLINE_DECISION_VERSION_ENV, "not-a-version")
    assert configured_online_decision_version() == ONLINE_DECISION_VERSION
    monkeypatch.setenv(ONLINE_DECISION_VERSION_ENV, ONLINE_DECISION_VERSION_V2)
    assert PolicyEngine().online_decision_version == ONLINE_DECISION_VERSION_V2
    assert PolicyEngine(
        online_decision_version=ONLINE_DECISION_VERSION
    ).online_decision_version == ONLINE_DECISION_VERSION
    with pytest.raises(ValueError):
        PolicyEngine(online_decision_version="bad")


def test_client_sends_the_v2_tool_spec_for_v2_messages(monkeypatch):
    calls = []
    arguments = json.dumps(_payload(), ensure_ascii=False)

    def opener(request, **_kwargs):
        calls.append(json.loads(request.data))
        body = {
            "model": "qwen/chat-test",
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "aria_turn_decision", "arguments": arguments},
            }]}}],
        }
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    client = OpenRouterLLMClient(
        api_key="test-key", base_url="https://example.test",
        model="qwen/chat-test", role="online_fast",
    )
    messages = single_pass_messages(
        _MESSAGES, _context(), version=ONLINE_DECISION_VERSION_V2, persona=_persona(),
    )
    raw = client.complete(messages)
    assert json.loads(raw)["deliberation"]["person_read"]
    assert calls[0]["tools"] == [online_decision_tool_spec(ONLINE_DECISION_VERSION_V2)]
    system = [m["content"] for m in calls[0]["messages"] if m["role"] == "system"]
    assert any("唯一通道=native" in text for text in system)
