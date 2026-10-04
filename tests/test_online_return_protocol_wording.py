"""Prompt/transport regressions, not evidence of improved live compliance.

The HTTP boundary is an in-memory fixture; no external service is called.
"""
from copy import deepcopy
import io
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.policy.llm import MockLLMClient, OpenAILLMClient, OpenRouterLLMClient
from aria.engine.policy.online_decision import (
    online_decision_tool_spec, single_pass_messages,
)
from aria.engine.policy.reply_contract import contract_prompt, reply_quality_contract
from aria.scenes.registry import get_scene


def _context(advertised):
    ctx = EngineContext(user_id="return-contract", scene_id="companion_elder")
    ctx.working_memory["_advertised_tools_for_turn"] = deepcopy(advertised)
    return ctx


@pytest.mark.parametrize("advertised", [[], [{"name": "fixture_external_action"}]])
def test_return_function_is_separate_from_external_action_permissions(advertised):
    ctx = _context(advertised)
    messages = [
        {"role": "system", "content": "陪伴说明", "name": "preserved"},
        {"role": "user", "content": "今天想随便聊聊。"},
    ]
    original = deepcopy(messages)
    original_memory = deepcopy(ctx.working_memory)
    emitted = single_pass_messages(messages, ctx)
    assert messages == original
    assert len(emitted) == len(original)
    assert emitted[1:] == original[1:]
    assert set(emitted[0]) == set(original[0])
    assert emitted[0]["name"] == "preserved"
    assert ctx.working_memory == original_memory
    text = emitted[0]["content"]
    assert "\n只输出一个 JSON 对象：API有aria_turn_decision时，仅调用一次" in text
    assert "JSON放function.arguments，勿另附content；否则输出JSON正文" in text
    assert "禁Markdown" in text
    assert "内部返回不执行/授权外部动作" in text
    assert "短句仅指response字段" in text
    assert "情绪/症状本身不授权" in text
    assert "本轮工具协议未开放" not in text
    assert "本轮没有向模型开放可执行工具" not in text
    if advertised:
        assert "上文列出且当前明确请求/确认并授权的工具" in text
        assert "标签置 response 末尾；拿不准就省略" in text
    else:
        assert "禁用外部 <action> 标签" in text




def test_shared_natural_reply_rule_is_not_an_envelope_format_instruction():
    text = contract_prompt(reply_quality_contract(_context([])))
    assert "用户可见正文用1–2个自然短句" in text
    assert "在线决策时仅指response字段" in text
    assert "本轮外部行动工具" in text
    assert "通知须有相应工具" in text
    assert "有真实回执才报成功" in text


@pytest.mark.parametrize("client_type", [OpenRouterLLMClient, OpenAILLMClient])
def test_one_turn_preserves_native_function_or_plain_json_transport(monkeypatch, client_type):
    declaration = {
        "response": "我在听，可以慢慢说。",
        "state_signals": [],
        "current_affect": {"label": "neutral", "confidence": .3,
            "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
        "selected_intervention_id": "",
        "communication_plan": {"pace": "open_listening", "support_order": "先听本人说",
                               "boundary": "不强加话题或行动"},
        "uncertainty": .6,
    }
    requests = []

    class Response(io.BytesIO):
        status = 200

    def opener(request, **_kwargs):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append(payload)
        encoded = json.dumps(declaration, ensure_ascii=False)
        if client_type is OpenRouterLLMClient:
            assert payload["tools"] == [online_decision_tool_spec()]
            assert payload["tool_choice"] == {
                "type": "function", "function": {"name": "aria_turn_decision"},
            }
            assert payload["parallel_tool_calls"] is False
            message = {"content": None, "tool_calls": [{"type": "function", "id": "fixture",
                "function": {"name": "aria_turn_decision", "arguments": encoded}}]}
        else:
            assert "tools" not in payload and "tool_choice" not in payload
            message = {"content": encoded}
        return Response(json.dumps({"model": "qwen/fixture", "choices": [{
            "message": message, "finish_reason": "tool_calls" if "tools" in payload else "stop",
        }]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    kwargs = {"api_key": "fixture-only", "base_url": "https://fixture.invalid",
              "model": "qwen/fixture", "role": "online_fast", "max_retries": 0}
    if client_type is OpenRouterLLMClient:
        kwargs["fallback_models"] = []
    client = client_type(**kwargs)
    engine = Engine(get_scene("companion_elder"), persona=PersonaProfile(user_id="return-contract"),
        llm_client=MockLLMClient("reflection"), state_llm_client=MockLLMClient("state"),
        fast_llm_client=client)
    ctx = _context([])
    reply = engine.process_turn(ctx, "反正去了也帮不上忙。")
    assert len(requests) == client.generation_decision_count == client.network_request_count == 1
    assert ctx.working_memory["online_turn_decision"]["status"] == "parsed"
    assert ctx.working_memory["_advertised_tools_for_turn"] == []
    assert "aria_turn_decision" not in reply and "<action>" not in reply
    assert ctx.working_memory["online_turn_decision"]["communication_plan"] == declaration["communication_plan"]
