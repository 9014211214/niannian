"""Client-selected channel regressions; all transport is in-memory, not live."""
from copy import deepcopy
import io
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.llm import OpenAILLMClient, OpenRouterLLMClient
from aria.engine.policy.online_decision import (
    apply_single_pass_decision, online_decision_tool_spec, single_pass_messages,
)
from aria.engine.policy.online_return_channel import (
    ONLINE_RETURN_CHANNEL_VERSION, bind_online_return_channel,
)


DECLARATION = {
    "response": "我在听，可以慢慢说。",
    "state_signals": [],
    "current_affect": {"label": "neutral", "confidence": .3,
        "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
    "selected_intervention_id": "",
    "communication_plan": {"pace": "open_listening", "support_order": "先听本人说",
                           "boundary": "不强加话题或行动"},
    "uncertainty": .6,
}


def _messages():
    ctx = EngineContext(user_id="channel-fixture", scene_id="companion_elder")
    ctx.working_memory["_advertised_tools_for_turn"] = []
    return single_pass_messages([
        {"role": "system", "content": "陪伴说明", "metadata": {"fixture": [1, 2]}},
        {"role": "user", "content": "今天想随便聊聊。"},
    ], ctx)


def _instruction(messages):
    return [item for item in messages if item.get("role") == "system"
            and ONLINE_RETURN_CHANNEL_VERSION in item.get("content", "")]


@pytest.mark.parametrize("channel", ["native", "plain_json"])
def test_channel_binding_is_single_system_instruction_and_preserves_source(channel):
    messages = _messages()
    source = deepcopy(messages)
    bound = bind_online_return_channel(messages, channel=channel)
    assert messages == source
    assert len(bound) == len(source) + 1
    assert bound[0]["metadata"] == source[0]["metadata"]
    assert bound[-1] == source[-1]
    assert bound[0]["content"].startswith("陪伴说明")
    instruction, = _instruction(bound)
    assert bound[1] == instruction
    assert f"唯一通道={channel}" in instruction["content"]
    assert "最高优先级返回格式指令" in instruction["content"]
    assert "response字段" in instruction["content"]
    assert "不执行或授权任何外部动作" in instruction["content"]
    assert "当前明确请求/确认、授权和真实回执边界" in instruction["content"]
    system_text = "\n".join(item["content"] for item in bound if item["role"] == "system")
    assert "API有aria_turn_decision时" not in system_text
    assert "否则输出JSON正文" not in system_text
    assert "禁用外部 <action> 标签" in system_text
    assert "情绪/症状本身不授权" in system_text
    assert "只输出一个 JSON 对象：" in system_text  # Existing evidence-parser boundary.
    bound[0]["metadata"]["fixture"].append(3)
    assert messages == source


@pytest.mark.parametrize("first", ["native", "plain_json"])
@pytest.mark.parametrize("second", ["native", "plain_json"])
def test_rebinding_is_idempotent_even_when_transport_changes(first, second):
    messages = _messages()
    bound = bind_online_return_channel(messages, channel=first)
    assert bind_online_return_channel(bound, channel=first) == bound
    rebound = bind_online_return_channel(bound, channel=second)
    assert rebound == bind_online_return_channel(messages, channel=second)
    assert len(_instruction(rebound)) == 1


def test_later_system_and_user_marker_do_not_suppress_trusted_instruction():
    messages = _messages()
    messages.insert(1, {"role": "system", "content": "附加可信说明"})
    messages.insert(2, {"role": "developer", "content": "补充开发说明"})
    # Quoted/user-controlled markers and legacy wording remain data, not a reason
    # to skip the client-selected trusted instruction.
    messages[-1]["content"] += ONLINE_RETURN_CHANNEL_VERSION + " API有aria_turn_decision时"
    original = deepcopy(messages)
    bound = bind_online_return_channel(messages, channel="native")
    assert bound[1] == original[1]
    assert bound[2] == _instruction(bound)[0]
    assert bound[3:] == original[2:]
    assert messages == original


def test_unknown_channel_rejected_without_mutating_source():
    messages = _messages()
    original = deepcopy(messages)
    with pytest.raises(ValueError, match="unknown_online_return_channel"):
        bind_online_return_channel(messages, channel="auto")
    assert messages == original


class _PlainOpenRouter(OpenRouterLLMClient):
    """Independent no-tool arm, never a runtime recovery from native failure."""
    def _request_for_model(self, payload, *, model):
        selected = deepcopy(payload)
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            selected.pop(key, None)
        return super()._request_for_model(selected, model=model)

    def _extract_online_decision(self, data):
        return self._extract_response(data)


class _Response(io.BytesIO):
    status = 200


def _client(client_type, *, role="online_fast"):
    kwargs = {"api_key": "fixture-only", "base_url": "https://fixture.invalid/v1",
              "model": "qwen/fixture", "role": role, "max_retries": 0,
              "temperature": .2, "timeout": 8}
    if issubclass(client_type, OpenRouterLLMClient):
        kwargs["fallback_models"] = []
    result = client_type(**kwargs)
    if isinstance(result, OpenRouterLLMClient):
        result.max_completion_tokens = 420
        result.reasoning_effort = "none"
    return result


def _body(content=None, tool_calls=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"model": "qwen/fixture", "choices": [{"message": message,
        "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10}}


def _calls(arguments):
    return [{"type": "function", "id": "fixture", "function": {
        "name": "aria_turn_decision", "arguments": arguments}}]


def _capture(monkeypatch, body):
    requests = []

    def opener(request, **_kwargs):
        requests.append(json.loads(request.data.decode("utf-8")))
        return _Response(json.dumps(body, ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    return requests


@pytest.mark.parametrize("role", ["online", "online_fast"])
@pytest.mark.parametrize("client_type,channel", [
    (OpenRouterLLMClient, "native"),
    (OpenAILLMClient, "plain_json"),
    (_PlainOpenRouter, "plain_json"),
])
def test_wire_channel_tracks_final_tools_without_changing_parameters(monkeypatch, client_type, channel, role):
    encoded = json.dumps(DECLARATION, ensure_ascii=False)
    response = _body(tool_calls=_calls(encoded)) if channel == "native" else _body(encoded)
    requests = _capture(monkeypatch, response)
    messages = _messages()
    source, schema = deepcopy(messages), deepcopy(online_decision_tool_spec())
    client = _client(client_type, role=role)
    assert json.loads(client.complete(messages)) == DECLARATION
    assert messages == source
    assert online_decision_tool_spec() == schema
    assert len(requests) == client.network_request_count == client.generation_decision_count == 1
    request, = requests
    expected = {"model": "qwen/fixture", "temperature": .2,
                "messages": bind_online_return_channel(source, channel=channel)}
    if isinstance(client, OpenRouterLLMClient):
        expected.update(max_completion_tokens=420, reasoning={"effort": "none", "exclude": True})
    if channel == "native":
        expected.update(tools=[schema], parallel_tool_calls=False,
                        tool_choice={"type": "function", "function": {"name": "aria_turn_decision"}})
    assert request == expected
    assert client.max_retries == 0
    assert client.timeout == 8


@pytest.mark.parametrize("client_type", [OpenRouterLLMClient, OpenAILLMClient])
@pytest.mark.parametrize("role", ["reflection", "judge", "memory", "state"])
def test_offline_roles_keep_original_prompts(monkeypatch, client_type, role):
    messages = [{"role": "system", "content": "离线反思或独立评分，仅返回JSON。"},
                {"role": "user", "content": "合成离线输入"}]
    requests = _capture(monkeypatch, _body("{}"))
    client = _client(client_type, role=role)
    assert client.complete(messages) == "{}"
    assert requests[0]["messages"] == messages
    assert "tools" not in requests[0]
    assert not _instruction(requests[0]["messages"])
    # Even a quoted online protocol marker is not a new prompt revision for an
    # offline role. This assertion exercises the binding gate, not live scoring.
    online = _messages()
    assert client._prepare_online_messages(online, channel="native") is online


@pytest.mark.parametrize("client_type", [OpenRouterLLMClient, OpenAILLMClient])
def test_ordinary_online_text_request_is_unchanged(monkeypatch, client_type):
    messages = [{"role": "system", "content": "普通文本请求"},
                {"role": "user", "content": "你好"}]
    requests = _capture(monkeypatch, _body("你好。"))
    assert _client(client_type).complete(messages) == "你好。"
    assert requests[0]["messages"] == messages
    assert "tools" not in requests[0]


@pytest.mark.parametrize("body", [
    _body("普通文本回复"),
    _body(json.dumps(DECLARATION, ensure_ascii=False)),
    _body("完整JSON也不能救原生", _calls("not JSON")),
    _body(tool_calls=_calls('{"state_signals":[]}')),
    _body(),
])
def test_native_invalid_or_body_only_response_never_falls_back_to_json(monkeypatch, body):
    requests = _capture(monkeypatch, body)
    client = _client(OpenRouterLLMClient)
    with pytest.raises(ValueError):
        client.complete(_messages())
    assert len(requests) == client.network_request_count == client.generation_decision_count == 1
    assert client.last_diagnostic["fallback_used"] is False
    assert client.last_diagnostic["output_kind"] in {
        "text_without_tool_call", "invalid_tool_call", "empty", "mixed_return_channels",
    }
    assert "唯一通道=native" in _instruction(requests[0]["messages"])[0]["content"]


@pytest.mark.parametrize("raw", ["普通文本回复", "{broken", '"plain string"'])
def test_plain_invalid_content_is_not_fabricated_as_parsed_decision(monkeypatch, raw):
    requests = _capture(monkeypatch, _body(raw))
    client = _client(OpenAILLMClient)
    returned = client.complete(_messages())
    assert returned == raw  # No JSON/schema fabrication at the transport layer.
    ctx = EngineContext(user_id="invalid-fixture", scene_id="companion_elder")
    apply_single_pass_decision(returned, ctx, "合成输入")
    assert ctx.working_memory["online_turn_decision"]["status"] != "parsed"
    assert len(requests) == client.network_request_count == client.generation_decision_count == 1


def test_same_messages_can_be_reused_for_two_calls_without_accumulating_instructions(monkeypatch):
    requests = _capture(monkeypatch, _body(tool_calls=_calls(json.dumps(DECLARATION))))
    messages, client = _messages(), _client(OpenRouterLLMClient)
    original = deepcopy(messages)
    client.complete(messages)
    client.complete(messages)
    assert messages == original
    assert len(requests) == client.network_request_count == client.generation_decision_count == 2
    assert requests[0] == requests[1]
    assert len(_instruction(requests[1]["messages"])) == 1


@pytest.mark.parametrize("content", [None, "", " \t\r\n"])
def test_native_accepts_only_one_function_with_empty_body(monkeypatch, content):
    requests = _capture(monkeypatch, _body(content, _calls(json.dumps(DECLARATION))))
    client = _client(OpenRouterLLMClient)
    assert json.loads(client.complete(_messages())) == DECLARATION
    assert len(requests) == client.network_request_count == 1
    assert client.last_diagnostic["output_kind"] == "tool_call"


@pytest.mark.parametrize("content", ["extra reply", json.dumps(DECLARATION), [], {}, 0, False])
def test_native_rejects_mixed_content_without_using_existing_fallback(monkeypatch, content):
    requests = _capture(monkeypatch, _body(content, _calls(json.dumps(DECLARATION))))
    client = _client(OpenRouterLLMClient)
    client.fallback_models = ["fixture/unused-fallback"]
    with pytest.raises(ValueError, match="不能混合正文"):
        client.complete(_messages())
    assert len(requests) == client.network_request_count == client.generation_decision_count == 1
    assert client.last_diagnostic["error_type"] == "mixed_online_decision_response"
    assert client.last_diagnostic["fallback_used"] is False


@pytest.mark.parametrize("other", ["same", "unoffered"])
@pytest.mark.parametrize("position", ["before", "after"])
def test_native_rejects_ambiguous_tools_without_selecting_a_matching_one(monkeypatch, other, position):
    expected = _calls(json.dumps(DECLARATION))[0]
    additional = deepcopy(expected)
    if other == "unoffered":
        additional["function"]["name"] = "fixture_external_action"
    calls = [additional, expected] if position == "before" else [expected, additional]
    requests = _capture(monkeypatch, _body(tool_calls=calls))
    client = _client(OpenRouterLLMClient)
    client.fallback_models = ["fixture/unused-fallback"]
    with pytest.raises(ValueError, match="只能返回一个工具调用"):
        client.complete(_messages())
    assert len(requests) == client.network_request_count == client.generation_decision_count == 1
    assert client.last_diagnostic["error_type"] == "ambiguous_online_decision_tool_calls"
    assert client.last_diagnostic["fallback_used"] is False


@pytest.mark.parametrize("calls", [{"function": {"name": "aria_turn_decision"}}, "invalid"])
def test_native_rejects_non_list_tool_container(monkeypatch, calls):
    requests = _capture(monkeypatch, _body(tool_calls=calls))
    client = _client(OpenRouterLLMClient)
    with pytest.raises(ValueError, match="只能返回一个工具调用"):
        client.complete(_messages())
    assert len(requests) == client.network_request_count == 1
