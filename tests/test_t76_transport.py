"""Invented HTTP-200 choice shapes; no real network, credentials or retries."""
from copy import deepcopy
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import t7_transport as frozen, t74_transport as previous
from scripts import t76_transport as transport
from tests.test_t7_transport import KEY, Opener, Response, payload, reply

GENERATION_ID = "gen-1790777949000-AbCdEfGh12345678"
VISIBLE = "invented-visible-sentinel"
PRIVATE = "private-provider-sentinel"
HIDDEN = "hidden-provider-sentinel"


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("real_network_or_credentials_forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(frozen, "_opener", forbidden)
    monkeypatch.setattr(transport._private_frozen, "_opener", forbidden)
    old_text, old_bytes = Path.read_text, Path.read_bytes
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        forbidden() if path.name == ".env" else old_text(path, *args, **kwargs))
    monkeypatch.setattr(Path, "read_bytes", lambda path, *args, **kwargs:
                        forbidden() if path.name == ".env" else old_bytes(path, *args, **kwargs))


def tool_reply():
    value = reply()
    value["id"] = GENERATION_ID
    value["choices"][0]["finish_reason"] = "tool_calls"
    value["choices"][0]["message"] = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "authored_call", "type": "function", "function": {
            "name": "aria_turn_decision", "arguments": json.dumps({"response": VISIBLE})}}],
    }
    return value


def json_payload():
    return payload() | {"response_format": {"type": "json_object"}}


def call(value, *, wire=None):
    response = Response(value, headers={"X-Generation-Id": GENERATION_ID})
    opener = Opener(response)
    client = transport.OpenRouterTransport(KEY, opener=opener)
    try:
        result = client.request(wire or payload())
    except transport.TransportError as exc:
        result = exc
    assert len(opener.calls) == 1 and response.closed
    request, timeout = opener.calls[0]
    assert request.get_method() == "POST" and request.full_url == transport.ENDPOINT
    assert timeout == 120
    assert json.loads(request.data) == (wire or payload())
    return result


def broken_shape(fault):
    value = tool_reply()
    choice = value["choices"][0]
    message = choice["message"]
    native = message["tool_calls"][0]
    function = native["function"]
    mutations = {
        "choices_missing": lambda: value.pop("choices"),
        "choices_null": lambda: value.update(choices=None),
        "choices_object": lambda: value.update(choices={"private": PRIVATE}),
        "choices_empty": lambda: value.update(choices=[]),
        "choices_multiple": lambda: value.update(choices=[deepcopy(choice), deepcopy(choice)]),
        "choice_null": lambda: value.update(choices=[None]),
        "choice_text": lambda: value.update(choices=[PRIVATE]),
        "index_missing": lambda: choice.pop("index"),
        "index_boolean": lambda: choice.update(index=True),
        "index_string": lambda: choice.update(index=PRIVATE),
        "index_nonzero": lambda: choice.update(index=1),
        "message_missing": lambda: choice.pop("message"),
        "message_null": lambda: choice.update(message=None),
        "message_string": lambda: choice.update(message=PRIVATE),
        "role_missing": lambda: message.pop("role"),
        "role_wrong": lambda: message.update(role=PRIVATE),
        "role_object": lambda: message.update(role={"secret": PRIVATE}),
        "content_object": lambda: message.update(content={"private": PRIVATE}),
        "content_boolean": lambda: message.update(content=True),
        "content_number": lambda: message.update(content=123),
        "content_bad_part": lambda: message.update(content=[PRIVATE]),
        "content_bad_part_type": lambda: message.update(content=[{"type": PRIVATE, "text": PRIVATE}]),
        "content_bad_text": lambda: message.update(content=[{"type": "text", "text": {"private": PRIVATE}}]),
        "content_duplicate_json": lambda: message.update(content='{"response":"first","response":"' + PRIVATE + '"}'),
        "calls_object": lambda: message.update(tool_calls={"private": PRIVATE}),
        "calls_string": lambda: message.update(tool_calls=PRIVATE),
        "calls_boolean": lambda: message.update(tool_calls=True),
        "call_null": lambda: message.update(tool_calls=[None]),
        "call_string": lambda: message.update(tool_calls=[PRIVATE]),
        "call_type_missing": lambda: native.pop("type"),
        "call_type_wrong": lambda: native.update(type=PRIVATE),
        "function_missing": lambda: native.pop("function"),
        "function_null": lambda: native.update(function=None),
        "function_string": lambda: native.update(function=PRIVATE),
        "id_missing": lambda: native.pop("id"),
        "id_null": lambda: native.update(id=None),
        "id_number": lambda: native.update(id=123),
        "id_bad_format": lambda: native.update(id="private id: " + PRIVATE),
        "name_missing": lambda: function.pop("name"),
        "name_null": lambda: function.update(name=None),
        "name_number": lambda: function.update(name=123),
        "name_bad_format": lambda: function.update(name="private name: " + PRIVATE),
        "arguments_missing": lambda: function.pop("arguments"),
        "arguments_null": lambda: function.update(arguments=None),
        "arguments_array": lambda: function.update(arguments=[PRIVATE]),
        "arguments_boolean": lambda: function.update(arguments=True),
        "arguments_number": lambda: function.update(arguments=123),
        "arguments_malformed": lambda: function.update(arguments='{"response":"' + PRIVATE + '"'),
        "arguments_duplicate": lambda: function.update(arguments='{"response":"first","response":"' + PRIVATE + '"}'),
        "arguments_escaped_duplicate": lambda: function.update(arguments='{"response":1,"re\\u0073ponse":"' + PRIVATE + '"}'),
        "arguments_json_array": lambda: function.update(arguments='["' + PRIVATE + '"]'),
        "arguments_json_number": lambda: function.update(arguments="123"),
        "arguments_nan": lambda: function.update(arguments='{"uncertainty":NaN}'),
        "arguments_hidden": lambda: function.update(arguments=json.dumps({"response": VISIBLE, "analysis": HIDDEN})),
        "arguments_hidden_object": lambda: function.update(arguments={"response": VISIBLE, "reasoning": HIDDEN}),
        "arguments_hidden_marker": lambda: function.update(arguments=json.dumps({"response": "<think>" + HIDDEN + "</think>"})),
        "content_hidden_part": lambda: message.update(content=[{"type": "reasoning", "text": HIDDEN}]),
        "finish_missing": lambda: choice.pop("finish_reason"),
        "finish_unsupported": lambda: choice.update(finish_reason=PRIVATE),
        "finish_length": lambda: choice.update(finish_reason="length"),
        "finish_filter": lambda: choice.update(finish_reason="content_filter"),
    }
    mutations[fault]()
    return value


BAD_SHAPES = [
    "choices_missing", "choices_null", "choices_object", "choices_empty", "choices_multiple",
    "choice_null", "choice_text", "index_missing", "index_boolean", "index_string", "index_nonzero",
    "message_missing", "message_null", "message_string", "role_missing", "role_wrong", "role_object",
    "content_object", "content_boolean", "content_number", "content_bad_part", "content_bad_part_type",
    "content_bad_text", "content_duplicate_json", "calls_object", "calls_string", "calls_boolean",
    "call_null", "call_string", "call_type_missing", "call_type_wrong", "function_missing", "function_null",
    "function_string", "id_missing", "id_null", "id_number", "id_bad_format", "name_missing", "name_null",
    "name_number", "name_bad_format", "arguments_missing", "arguments_null", "arguments_array",
    "arguments_boolean", "arguments_number", "arguments_malformed", "arguments_duplicate",
    "arguments_escaped_duplicate", "arguments_json_array", "arguments_json_number", "arguments_nan",
    "arguments_hidden", "arguments_hidden_object", "arguments_hidden_marker", "content_hidden_part",
    "finish_missing", "finish_unsupported", "finish_length", "finish_filter",
]


@pytest.mark.parametrize("fault", BAD_SHAPES)
def test_http_200_invalid_shape_keeps_known_usage_and_fixed_diagnosis_without_retry(fault):
    value = broken_shape(fault)
    result = call(value)
    assert isinstance(result, transport.TransportError)
    assert result.outcome_unknown is True
    receipt = result.receipt
    assert receipt["model"] == "qwen/qwen3.7-flash" and receipt["provider"] == "alibaba"
    assert receipt["id"] == GENERATION_ID
    assert receipt["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "cost": "0.00001"}
    assert receipt["http_diagnostic"]["http_status"] == 200
    assert receipt["http_diagnostic"]["generation_id"] == GENERATION_ID
    shape = receipt["choice_shape"]
    assert shape["schema"] == transport.SHAPE_SCHEMA
    assert all(text not in json.dumps(shape) for text in (PRIVATE, VISIBLE, HIDDEN, KEY))
    assert all(text not in json.dumps(receipt) for text in (PRIVATE, HIDDEN, KEY))
    if fault.startswith("finish_"):
        assert result.code == "incomplete_completion"
    elif "hidden" in fault:
        assert result.code in {"hidden_reasoning_in_visible_output", "reasoning_not_disabled"}
    else:
        assert result.code == "invalid_choices"


@pytest.mark.parametrize("fault,reason", [
    ("arguments_duplicate", "duplicate_key"), ("arguments_escaped_duplicate", "duplicate_key"),
    ("arguments_malformed", "malformed_json"), ("arguments_json_array", "not_object_json"),
    ("arguments_json_number", "not_object_json"), ("arguments_nan", "nonfinite_number"),
    ("arguments_hidden", "hidden_field_or_marker"), ("arguments_hidden_object", "hidden_field_or_marker"),
    ("arguments_hidden_marker", "hidden_field_or_marker"),
])
def test_argument_reason_distinguishes_structural_failure_without_string_echo(fault, reason):
    result = call(broken_shape(fault))
    shape = result.receipt["choice_shape"]["choices"][0]["calls"][0]
    assert shape["arguments_json_shape"] == reason


def test_actual_object_arguments_are_serialized_with_original_representation_recorded():
    value = tool_reply()
    arguments = {"response": VISIBLE, "uncertainty": 0.5}
    value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = deepcopy(arguments)
    before = deepcopy(value)
    result = call(value)
    assert isinstance(result, dict) and value == before
    actual = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(actual) == arguments
    assert result["choice_shape"]["normalizations"] == ["object_arguments_serialized_as_json"]
    assert result["choice_shape"]["choices"][0]["calls"][0]["arguments_kind"] == "object"
    assert result["has_hidden_reasoning"] is False


@pytest.mark.parametrize("representation", ["missing", "null", "empty"])
def test_json_content_never_creates_native_tool_when_no_call_exists(representation):
    value = reply()
    value["id"] = GENERATION_ID
    message = value["choices"][0]["message"]
    message["content"] = json.dumps({"response": VISIBLE})
    if representation == "null":
        message["tool_calls"] = None
    elif representation == "empty":
        message["tool_calls"] = []
    result = call(value, wire=json_payload())
    assert isinstance(result, dict)
    assert not result["choices"][0]["message"].get("tool_calls")
    assert result["choice_shape"]["choices"][0]["tool_calls_kind"] == {
        "missing": "missing", "null": "null", "empty": "array"}[representation]
    assert result["choice_shape"]["normalizations"] == (["null_tool_calls_as_absent"]
        if representation == "null" else [])


def test_no_body_and_null_calls_are_not_changed_into_json_or_fake_decision():
    value = reply()
    message = value["choices"][0]["message"]
    message.update(content=None, tool_calls=None)
    result = call(value)
    assert result["choices"][0]["message"] == {"role": "assistant", "content": None}
    assert result["choice_shape"]["choices"][0]["content"]["kind"] == "null"


def test_text_parts_shape_is_recorded_while_original_visible_join_stays():
    value = reply()
    value["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    result = call(value)
    assert result["choices"][0]["message"]["content"] == "onetwo"
    shape = result["choice_shape"]["choices"][0]["content"]
    assert shape["kind"] == "array" and shape["parts_count"] == 2
    assert all(part["type"] == "text" for part in shape["parts"])
    assert all(text not in json.dumps(shape) for text in ("one", "two"))


def test_diagnostic_is_bounded_even_when_response_has_many_choices_parts_or_calls():
    value = tool_reply()
    message = value["choices"][0]["message"]
    message["tool_calls"] *= 20
    message["content"] = [{"type": PRIVATE, "text": PRIVATE}] * 20
    value["choices"] *= 20
    result = call(value)
    shape = result.receipt["choice_shape"]
    assert shape["choices_count"] == 20 and len(shape["choices"]) == transport.MAX_SHAPE_ITEMS
    assert shape["choices_truncated"] is True
    assert all(len(choice["calls"]) == len(choice["content"]["parts"]) == transport.MAX_SHAPE_ITEMS
               for choice in shape["choices"])
    assert len(json.dumps(shape)) < 30000
    assert PRIVATE not in json.dumps(shape)


@pytest.mark.parametrize("body,reason", [
    (b'{"choices":[],"choices":[]}', "duplicate_key"),
    (b'{"response":', "malformed_json"), (b'[]', "not_object_json"),
    (b'{"uncertainty":NaN}', "nonfinite_number"), (b'\xff', "invalid_unicode"),
])
def test_bad_outer_json_cannot_be_repaired_or_given_untrusted_usage(body, reason):
    result = call(body)
    assert result.code == "invalid_response_json"
    assert result.receipt["choice_shape"]["body_json_shape"] == reason
    assert result.receipt["choice_shape"]["choices_kind"] == "unavailable"
    assert "usage" not in result.receipt
    assert result.receipt["http_diagnostic"]["http_status"] == 200


@pytest.mark.parametrize("escaped", [False, True])
def test_credential_echo_is_rejected_before_shape_or_usage_is_persisted(escaped):
    value = tool_reply()
    value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"response": KEY})
    body = json.dumps(value)
    if escaped:
        body = body.replace(KEY, "".join("\\u%04x" % ord(char) for char in KEY))
    result = call(body.encode())
    assert result.code == "credential_echo_rejected"
    assert "usage" not in result.receipt and "choice_shape" not in result.receipt
    assert KEY not in json.dumps(result.receipt)


def test_http_failure_retains_t74_safe_category_without_trusting_claimed_cost():
    body = {"id": GENERATION_ID, "error": {"code": 404,
        "message": "No endpoints found for response format " + PRIVATE},
        "usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}
    stream = io.BytesIO(json.dumps(body).encode())
    error = HTTPError(transport.ENDPOINT, 404, PRIVATE, {}, stream)
    opener = Opener(error=error)
    with pytest.raises(transport.TransportError) as caught:
        transport.OpenRouterTransport(KEY, opener=opener).request(json_payload())
    assert len(opener.calls) == 1 and stream.closed
    receipt = caught.value.receipt
    assert receipt["http_diagnostic"]["api_error_reason"] == "no_endpoints_parameter_support"
    assert "usage" not in receipt and "choice_shape" not in receipt
    assert all(text not in json.dumps(receipt) for text in (PRIVATE, KEY))


def test_json_mode_sends_actual_parameter_and_preserves_route_without_input_mutation():
    wire = json_payload()
    before = deepcopy(wire)
    serialized = transport.validate_json_mode_payload(wire)
    assert json.loads(serialized) == before and wire == before
    value = reply()
    value["choices"][0]["message"]["content"] = json.dumps({"response": VISIBLE})
    result = call(value, wire=wire)
    assert result["choice_shape"]["choices"][0]["content"]["json_shape"] == "object"
    assert wire["provider"] == {"only": ["alibaba"], "allow_fallbacks": False,
                               "require_parameters": True, "data_collection": "deny"}


@pytest.mark.parametrize("change", [
    {"response_format": {"type": "json_schema"}}, {"response_format": {"type": "json_object", "strict": True}},
    {"response_format": "json_object"}, {"tools": []}, {"tool_choice": "auto"},
    {"parallel_tool_calls": False}, {"model": "anthropic/claude-sonnet-5"}, {"max_tokens": 2048},
    {"max_tokens": True}, {"temperature": 0.3}, {"temperature": True},
    {"reasoning": {"enabled": True, "exclude": True}},
    {"reasoning": {"enabled": False, "exclude": False}}, {"stream": True},
    {"provider": {"only": ["alibaba"], "allow_fallbacks": True,
                  "require_parameters": True, "data_collection": "deny"}},
    {"provider": {"only": ["alibaba"], "allow_fallbacks": False,
                  "require_parameters": True, "data_collection": "allow"}},
    {"provider": {"only": ["anthropic"], "allow_fallbacks": False,
                  "require_parameters": True, "data_collection": "deny"}},
    {"messages": []}, {"unused": PRIVATE},
])
def test_json_mode_rejects_any_route_budget_or_representation_expansion_before_post(change):
    opener = Opener(Response(reply()))
    with pytest.raises(transport.TransportError) as caught:
        transport.OpenRouterTransport(KEY, opener=opener).request(json_payload() | change)
    assert caught.value.outcome_unknown is False and opener.calls == []
    assert all(text not in str(caught.value) for text in (PRIVATE, KEY))


def test_default_network_gate_remains_closed_and_original_globals_unchanged():
    choices, sanitizer, request = frozen._choices, frozen.sanitize_response, previous.OpenRouterTransport.request
    original_view = request.__globals__["frozen"]
    with pytest.raises(transport.TransportError, match="network_disabled") as caught:
        transport.OpenRouterTransport(KEY).request(json_payload())
    assert caught.value.outcome_unknown is False
    assert frozen._choices is choices and frozen.sanitize_response is sanitizer
    assert previous.OpenRouterTransport.request is request
    assert request.__globals__["frozen"] is original_view is frozen
    assert transport.OpenRouterTransport.request.__globals__["frozen"] is transport._private_frozen
