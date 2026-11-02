"""Real in-memory HTTP framing through the current factory/guard; no model effects."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import http.client
import io
import json
from pathlib import Path
import socket
import threading
import time

import pytest

from aria.engine.policy.max_release import tls_dispatch
from aria.engine.policy.max_release.online_shape import (
    ONLINE_STREAM_VERSION, enable_online_sse, validate_online_sse_client, online_receipt_body_complete,
)
from aria.engine.policy.max_release.reflection_sse import (
    read_reflection_sse, validate_stream_body_observation, parse_reflection_sse, SSEProtocolError,
)
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.observed_transport import _ObservedSocket, _ObservedHTTPResponse, TransportTrace
from aria.engine.policy.response_body import ResponseDeadline
from tests import test_deepseek_offline_route as current
from tests import test_deepseek_release_guard as previous
from tests import test_installable_max_release as historical


class MemorySocket:
    """Only HTTPResponse's byte source; never constructs or opens a socket."""
    def __init__(self, raw):
        self.raw = raw

    def makefile(self, mode, *, buffering=0):
        assert mode == 'rb' and buffering == 0
        return io.BytesIO(self.raw)

    def settimeout(self, timeout):
        assert 0 < timeout <= 90


def http_response(body, *, trace=None, chunk_size=None, current_crlf=True, http_tail=False, on_body=None):
    if trace is None:
        deadline = ResponseDeadline(time.perf_counter() + 90)
        trace = TransportTrace(diagnostics=True, response_deadline=deadline)
    size = chunk_size or len(body)
    parts = [body[i:i + size] for i in range(0, len(body), size)]
    encoded = b''.join(f'{len(part):x}\r\n'.encode() + part +
        (b'\r\n' if current_crlf or i < len(parts) - 1 else b'')
        for i, part in enumerate(parts))
    if http_tail:
        encoded += b'0\r\n\r\n'
    raw = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
           b'Transfer-Encoding: chunked\r\n\r\n' + encoded)
    class ObservedMemoryResponse(_ObservedHTTPResponse):
        def read1(self, size=-1):
            part = super().read1(size)
            if part and not getattr(self, '_body_callback_completed', False):
                self._body_callback_completed = True
                if on_body:
                    on_body()
            return part
    response = ObservedMemoryResponse(_ObservedSocket(MemorySocket(raw), trace))
    response.begin()
    return response, trace.response_deadline


def stream_frames(data):
    identity = {key: data[key] for key in ('id', 'model', 'provider')}
    text = data['choices'][0]['message']['content']
    return [
        {**identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0,
            'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}]},
        {**identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0,
            'delta': {}, 'finish_reason': data['choices'][0]['finish_reason']}]},
        {**identity, 'object': 'chat.completion.chunk', 'choices': [], 'usage': data['usage']},
    ]


def stream_bytes(frames, *, done=True):
    raw = b''.join(('data: ' + json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n\n').encode()
                   for frame in frames)
    return raw + (b'data: [DONE]\n\n' if done else b'')


def install_memory_http(monkeypatch, body, *, chunk_size=None, current_crlf=True, before_response=None):
    calls = []
    def opened(request, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append((json.loads(request.data), timeout))
        if before_response:
            before_response()
        return http_response(body, trace=trace, chunk_size=chunk_size, current_crlf=current_crlf)[0]
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    return calls


@pytest.fixture(autouse=True)
def no_network_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network and credentials forbidden in in-memory HTTP tests')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria._env.load_local_env', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


def test_only_current_same_route_factory_enables_online_stream():
    bundle = current.clients()
    contract = validate_online_sse_client(bundle.online)
    assert contract['schema'] == ONLINE_STREAM_VERSION
    assert contract['provider_schema_applied'] is True
    assert contract['response_format_preserved'] is True
    for client in (bundle.memory, bundle.reflection, bundle.safety_review, bundle.confirm_judge,
                   previous.clients('together').online, previous.clients('fireworks').online,
                   historical.make_clients().online):
        assert getattr(client, 'online_stream_protocol', None) is None
    with pytest.raises(ValueError):
        enable_online_sse(bundle.online)


@pytest.mark.parametrize('chunk_size', [None, 1, 7, 113])
def test_current_factory_uses_one_guard_and_accepts_application_terminal_without_http_tail(tmp_path, monkeypatch, chunk_size):
    client = current.clients().online
    data = current.response(client)
    content = json.dumps({'response_segments': [{'text': '逐字软件正文：春天🌱。'}],
                         'safety_concern': 'none'}, ensure_ascii=False)
    data['choices'][0]['message']['content'] = content
    body = stream_bytes(stream_frames(data))
    calls = install_memory_http(monkeypatch, body, chunk_size=chunk_size)
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        result = previous.invoke(guard, client)
        state = guard.snapshot()
        row, = state['attempts']
        assert result == content
        assert len(calls) == state['calls_used'] == 1 and not state['stopped']
        wire, timeout = calls[0]
        assert timeout == row['effective_physical_timeout_seconds'] == 90
        assert wire['stream'] is True and wire['response_format']['type'] == 'json_schema'
        assert wire['response_format']['json_schema']['strict'] is True
        assert wire['provider']['only'] == ['together']
        assert wire['provider']['allow_fallbacks'] is False
        assert wire['reasoning'] == {'enabled': False, 'exclude': True}
        assert wire['max_tokens'] == row['reservation']['output_token_limit'] == 4096
        assert row['online_stream_contract'] == validate_online_sse_client(client)
        assert online_receipt_body_complete(row, content)
        assert row['receipt_identity']['model_status'] == row['receipt_identity']['provider_status'] == 'recognized'
        assert state['known_cost_usd'] == str(Decimal(str(data['usage']['cost'])))
        assert state['reserved_cost_usd'] == '0' and state['paid_response_replays'] == 0
        observed = validate_stream_body_observation(row['body_read'])
        assert observed['application_complete'] is True and observed['current_chunk_terminated'] is True
        assert observed['http_message_complete'] is False and observed['complete'] is False
        assert observed['bytes_received'] == len(body) and observed['body_sha256'] == sha256(body).hexdigest()
        public = json.loads(Path(row['public_first_output']['path']).read_bytes())
        assert public['public_first_output'][0]['content'] == content
        physical, = guard.physical_dir.iterdir()
        receipt = json.loads((physical / 'first-body-receipt.json').read_bytes())
        assert receipt['unread_http_tail_possible'] is True
        assert receipt['byte_identical_response_stored'] is False
        assert receipt['raw_stream_scope'] == 'observed_sse_prefix_through_current_chunk'
        frames = json.loads((physical / 'stream-public-frames.json').read_bytes())
        assert ''.join(frame['content'] for frame in frames['frames']) == content
        assert frames['raw_stream_saved'] is False and frames['hidden_reasoning_saved'] is False
        assert 'provider_schema_applied' not in row['stream_read']  # No reflection schema-off label.


@pytest.mark.parametrize('failure', ['missing_done', 'missing_usage', 'missing_cost', 'missing_chunk_crlf',
    'provider_drift', 'generation_drift', 'model_drift', 'choice_drift', 'after_done_event',
    'finish_other', 'refusal', 'tool_call', 'hidden_reasoning', 'wrong_provider', 'wrong_model',
    'cost_over_bound', 'tokens_over_bound'])
def test_incomplete_or_unverified_stream_keeps_full_hold_and_never_replays(tmp_path, monkeypatch, failure):
    client = current.clients().online
    data = current.response(client)
    if failure == 'wrong_provider':
        data['provider'] = 'Fireworks'
    if failure == 'wrong_model':
        data['model'] = 'synthetic/unexpected-model'
    if failure == 'cost_over_bound':
        data['usage']['cost'] = 1
    if failure == 'tokens_over_bound':
        data['usage'].update(completion_tokens=4097, total_tokens=4117)
    if failure == 'missing_cost':
        del data['usage']['cost']
    frames = stream_frames(data)
    if failure == 'missing_usage':
        frames.pop()
    for case, key, value in [('provider_drift', 'provider', 'Fireworks'),
                             ('generation_drift', 'id', 'gen-otherfixture123'),
                             ('model_drift', 'model', 'synthetic/changed-model')]:
        if failure == case:
            frames[1][key] = value
    if failure == 'choice_drift':
        frames[1]['choices'][0]['index'] = 1
    if failure == 'finish_other':
        frames[1]['choices'][0]['finish_reason'] = 'content_filter'
    if failure == 'refusal':
        frames[0]['choices'][0]['delta']['refusal'] = 'synthetic refusal'
    if failure == 'tool_call':
        frames[0]['choices'][0]['delta']['tool_calls'] = [{'index': 0}]
    if failure == 'hidden_reasoning':
        frames[0]['choices'][0]['delta']['reasoning'] = 'synthetic hidden sentinel'
    body = stream_bytes(frames, done=failure != 'missing_done')
    if failure == 'after_done_event':
        body += b'data: {"error":{"code":500}}\n\n'
    calls = install_memory_http(monkeypatch, body, current_crlf=failure != 'missing_chunk_crlf')
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client)
        state = guard.snapshot()
        row, = state['attempts']
        assert row['status'] == 'failed' and state['stopped']
        assert len(calls) == state['calls_used'] == 1 and state['paid_response_replays'] == 0
        assert state['known_cost_usd'] is None and 'cost' not in row['usage']
        assert state['unknown_cost_requests'] == 1
        assert Decimal(state['reserved_cost_usd']) == Decimal(row['reservation']['reserved_cost_usd']) > 0
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client, operation='forbidden-replay')
        assert len(calls) == 1
        if failure not in ('wrong_provider', 'wrong_model', 'cost_over_bound', 'tokens_over_bound'):
            assert 'public_first_output' not in row
            physical, = guard.physical_dir.iterdir()
            assert not (physical / 'first-visible.json').exists()
            assert not (physical / 'stream-public-frames.json').exists()
            receipt = json.loads((physical / 'first-body-receipt.json').read_bytes())
            assert receipt['partial'] is True and receipt['visible_projection_saved'] is False


@pytest.mark.parametrize('failure', ['stream_false', 'schema_removed', 'provider_changed', 'model_changed'])
def test_bound_online_wire_cannot_select_weaker_transport_or_route(tmp_path, failure):
    client = current.clients().online
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        # Capture the real built request before any claim, then check mutation
        # against the same guard rather than hand-building a matching schema.
        captured = []
        class Capture(BaseException):
            pass
        def capture(client, request, **requirements):
            captured.append(json.loads(request.data))
            raise Capture
        guard._request_once = capture
        with pytest.raises(Capture):
            previous.invoke(guard, client)
        wire, = captured
        guard._validate_payload(client, wire)
        changed = deepcopy(wire)
        if failure == 'stream_false':
            changed['stream'] = False
        elif failure == 'schema_removed':
            del changed['response_format']
        elif failure == 'provider_changed':
            changed['provider']['only'] = ['fireworks']
        else:
            changed['model'] = 'synthetic/changed-model'
        with pytest.raises(MinimalCallStopped):
            guard._validate_payload(client, changed)
        assert guard.snapshot()['calls_used'] == 0


def test_done_inside_incomplete_declared_chunk_is_not_an_application_terminal():
    client = current.clients().online
    body = stream_bytes(stream_frames(current.response(client)))
    # DONE is complete, but the enclosing chunk promises bytes that never arrive.
    raw = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n'
           + f'{len(body) + 1:x}\r\n'.encode() + body)
    deadline = ResponseDeadline(time.perf_counter() + 90)
    trace = TransportTrace(diagnostics=True, response_deadline=deadline)
    response = _ObservedHTTPResponse(_ObservedSocket(MemorySocket(raw), trace)); response.begin()
    observed, stream = {}, {}
    with pytest.raises(http.client.IncompleteRead):
        read_reflection_sse(response, limit=2097152, deadline=deadline, observation=observed,
                            stream_observation=stream, header_observation={})
    assert stream['done_observed'] is True
    assert observed['application_complete'] is False and stream['partial_output_submitted'] is False


@pytest.mark.parametrize('finish,usage_finish,accepted', [
    ('stop', 'stop', True), ('length', 'length', True),
    ('stop', 'length', False), ('length', 'stop', False),
])
def test_terminal_usage_may_repeat_but_never_change_the_original_finish(finish, usage_finish, accepted):
    client = current.clients().online
    data = current.response(client)
    data['choices'][0]['finish_reason'] = finish
    frames = stream_frames(data)
    frames[-1]['choices'] = [{'index': 0, 'delta': {}, 'finish_reason': usage_finish}]
    raw = stream_bytes(frames)
    if accepted:
        assembled, _ = parse_reflection_sse(raw, receipt_mode=ONLINE_STREAM_VERSION)
        assert json.loads(assembled)['choices'][0]['finish_reason'] == finish
    else:
        with pytest.raises(SSEProtocolError) as error:
            parse_reflection_sse(raw, receipt_mode=ONLINE_STREAM_VERSION)
        assert error.value.code == 'finish_changed'


@pytest.mark.parametrize('kind', ['length', 'empty'])
def test_accountable_invalid_online_content_does_not_weaken_reflection_reader(kind):
    data = current.response(current.clients().online)
    if kind == 'length':
        data['choices'][0]['finish_reason'] = 'length'
    else:
        data['choices'][0]['message']['content'] = ''
    raw = stream_bytes(stream_frames(data))
    with pytest.raises(SSEProtocolError):
        parse_reflection_sse(raw)
    assembled, _ = parse_reflection_sse(raw, receipt_mode=ONLINE_STREAM_VERSION)
    value = json.loads(assembled)
    assert value['choices'][0] == {'index': 0, 'finish_reason': data['choices'][0]['finish_reason'],
        'message': {'role': 'assistant', 'content': data['choices'][0]['message']['content']}}


@pytest.mark.parametrize('kind', ['length', 'empty'])
def test_actual_app_isolates_fully_accounted_length_or_empty_then_accepts_only_new_input(tmp_path, monkeypatch, kind):
    from aria.voice.minimal import MinimalWebApp
    from tests.test_safety_review_v2 import answer
    bundle = current.clients()
    calls = []
    def opened(request, *, timeout, trace):
        wire = json.loads(request.data)
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append(wire)
        if wire['max_tokens'] == 4096:
            data = current.response(bundle.online)
            if len(calls) == 1:
                if kind == 'length':
                    data['choices'][0]['finish_reason'] = 'length'
                else:
                    data['choices'][0]['message']['content'] = ''
            return http_response(stream_bytes(stream_frames(data)), trace=trace)[0]
        data = current.response(bundle.safety_review)
        data['choices'][0]['message']['content'] = json.dumps(answer(wire=wire))
        return http_response(stream_bytes(stream_frames(data)), trace=trace)[0]
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    app = MinimalWebApp(tmp_path, clients=bundle, model_max_requests=10,
        model_stop_cost_usd='1', model_reservation_policy=current.policy())
    clock = [1000.0]
    app.call_guard._pacing_clock = lambda: clock[0]
    app.call_guard._pacing_sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    sid = app.start()['state']['session_id']
    try:
        first = app.submit('首个中性软件输入。', 'stream-invalid-1', sid)
        assert not first['ok'] and first['settled_validation_failure']
        assert not first['response_delivered'] and first['continue_next_input']
        assert app._run['turns'][-1]['response'] == ''
        state = app.call_guard.snapshot()
        row, = state['attempts']
        assert row['status'] == 'failed' and not state['stopped']
        assert state['known_cost_usd'] == '0.00001' and state['reserved_cost_usd'] == '0'
        assert state['unknown_cost_requests'] == 0
        assert row['failure_phase'] == 'receipt_validation'
        assert row['validation_error'] == ('invalid_or_truncated_response' if kind == 'length' else 'invalid_response_message')
        public = json.loads(Path(row['public_first_output']['path']).read_bytes())
        assert online_receipt_body_complete(row, public['public_first_output'][0]['content'])
        assert row['body_read']['application_complete'] is True and row['body_read']['http_message_complete'] is False
        assert app.submit('首个中性软件输入。', 'stream-invalid-1', sid)['duplicate'] and len(calls) == 1
        assert app.submit('后续中性软件输入。', 'stream-valid-2', sid)['ok']
        assert len(calls) == 3 and app.call_guard.snapshot()['paid_response_replays'] == 0
    finally:
        app.close()


@pytest.mark.parametrize("outer_present", [False, True])
def test_length_padded_complete_members_reach_original_body_review(tmp_path, monkeypatch, outer_present):
    from aria.voice.minimal import MinimalWebApp
    from aria.engine.policy.max_release.online_shape import OUTER_CLOSE, PADDED_OBJECT, completion_projection_metadata
    owner = "我想请人帮我办件事。"
    safe, rejected = "可以写下你的请求。", "我已经替你发出去了。"
    text = json.dumps({"safety_concern": "none", "response_segments": [{"text": safe}, {"text": rejected}]})
    raw = (text if outer_present else text[:-1]) + "\n " * 800
    bundle, calls = current.clients(), []

    def opened(request, *, timeout, trace):
        wire = json.loads(request.data)
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append(wire)
        if len(calls) == 1:
            data = current.response(bundle.online)
            data["choices"][0].update(finish_reason="length", message={"content": raw})
        else:
            data = current.response(bundle.safety_review)
            data["choices"][0]["message"]["content"] = json.dumps(
                {"risk_assessment": [], "keep_each_span": [True, False]})
        return http_response(stream_bytes(stream_frames(data)), trace=trace)[0]

    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    app = MinimalWebApp(tmp_path, clients=bundle, model_max_requests=5,
                        model_stop_cost_usd="1", model_reservation_policy=current.policy())
    clock = [1000.0]
    app.call_guard._pacing_clock = lambda: clock[0]
    app.call_guard._pacing_sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    try:
        sid = app.start()["state"]["session_id"]
        assert app.submit(owner, "padded-online", sid)["ok"]
        first, review = app.call_guard.snapshot()["attempts"]
        assert first["status"] == review["status"] == "received"
        assert first["online_receipt_shape"]["finish_reason"] == "length"
        assert online_receipt_body_complete(first, raw)
        assert json.loads(Path(first["public_first_output"]["path"]).read_bytes())["public_first_output"][0]["content"] == raw
        canonical = raw if outer_present else raw + "}"
        receipt = completion_projection_metadata(bundle.online, canonical)
        assert receipt == first["online_length_projection"]
        assert receipt["mode"] == (PADDED_OBJECT if outer_present else OUTER_CLOSE)
        assert json.loads(calls[1]["messages"][1]["content"])["candidate"] == safe + rejected
        assert app._run["turns"][-1]["response"] == safe
        assert len(calls) == 2 and not app.call_guard.snapshot()["stopped"]
    finally:
        app.close()


@pytest.mark.parametrize("raw", [
    '{"safety_concern":"none","response_segments":[{"text":"safe"}',
    '{"safety_concern":"none","response_segments":[{"text":"unclosed',
    '{"safety_concern":"none","response_segments":[{"text":"safe"}],',
    '{"safety_concern":"none","response_segments":[{"text":"safe"}]} {}',
    '{"safety_concern":"none","safety_concern":"fall","response_segments":[{"text":"safe"}]',
    '{"response_segments":[{"text":"safe"}]',
    '{"safety_concern":"unsupported","response_segments":[{"text":"safe"}]',
    '{"safety_concern":"none","response_segments":[{"text":"safe","kind":"assistant_action","action":"wire_transfer"}]',
    '{"safety_concern":"none","response_segments":[{"text":"safe"}],"unexpected":[]',
    '{"safety_concern":"none","response_segments":' + "[" * 1500 + "]" * 1500,
])
def test_padded_length_rejects_truncated_members_and_invalid_full_schema(raw):
    from aria.engine.policy.max_release.online_shape import online_envelope_projection
    data = {"choices": [{"finish_reason": "length", "message": {"content": raw + " \n"}}]}
    with pytest.raises(ValueError):
        online_envelope_projection(data, "online_turn_decision_v7", allow_padded_length=True)


@pytest.mark.parametrize("fault", ["wrong_provider", "missing_cost", "missing_done", "refusal", "tool_call"])
def test_padded_length_never_substitutes_for_a_verified_native_response(tmp_path, monkeypatch, fault):
    client = current.clients().online
    data = current.response(client)
    raw = '{"safety_concern":"none","response_segments":[{"text":"软件测试。"}]  \n'
    data["choices"][0].update(finish_reason="length", message={"content": raw})
    if fault == "wrong_provider":
        data["provider"] = "Elsewhere"
    if fault == "missing_cost":
        data["usage"].pop("cost")
    frames = stream_frames(data)
    if fault == "refusal":
        frames[0]["choices"][0]["delta"]["refusal"] = "declined"
    if fault == "tool_call":
        frames[0]["choices"][0]["delta"]["tool_calls"] = [{"id": "call_1"}]
    install_memory_http(monkeypatch, stream_bytes(frames, done=fault != "missing_done"))
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client)
        state = guard.snapshot()
        row, = state["attempts"]
        assert row["status"] == "failed" and state["stopped"]
        assert not client.last_diagnostic.get("online_response_projection")


@pytest.mark.parametrize('damage', ['contract_type', 'contract_mode', 'body_hash', 'body_complete',
    'stream_hash', 'stream_bytes', 'content_hash', 'shape_hash', 'generation', 'finish', 'missing_stream'])
def test_application_receipt_proof_rejects_missing_or_crossed_sources(tmp_path, monkeypatch, damage):
    client = current.clients().online
    data = current.response(client)
    content = data['choices'][0]['message']['content']
    install_memory_http(monkeypatch, stream_bytes(stream_frames(data)))
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        previous.invoke(guard, client)
        row, = guard.snapshot()['attempts']
        assert online_receipt_body_complete(row, content)
        changed = deepcopy(row)
        if damage == 'contract_type':
            changed['online_stream_contract']['stream'] = 1
        elif damage == 'contract_mode':
            changed['online_stream_contract']['receipt_mode'] = 'other'
        elif damage == 'body_hash':
            changed['body_read']['body_sha256'] = '0' * 64
        elif damage == 'body_complete':
            changed['body_read']['application_complete'] = False
        elif damage == 'stream_hash':
            changed['stream_read']['stream_sha256'] = '0' * 64
        elif damage == 'stream_bytes':
            changed['stream_read']['stream_bytes'] += 1
        elif damage == 'content_hash':
            changed['stream_read']['public_content_sha256'] = '0' * 64
        elif damage == 'shape_hash':
            changed['online_receipt_shape']['public_content_sha256'] = '0' * 64
        elif damage == 'generation':
            changed['stream_read']['generation_id'] = 'gen-otherfixture123'
        elif damage == 'finish':
            changed['stream_read']['finish_reason'] = 'length'
        else:
            del changed['stream_read']
        assert not online_receipt_body_complete(changed, content)
        assert not online_receipt_body_complete(row, content + 'changed')


@pytest.mark.parametrize('mutation', ['none', 'stop', 'owner', 'session', 'forget'])
def test_actual_app_does_not_deliver_stale_stream_after_authority_changes(tmp_path, monkeypatch, mutation):
    from aria.voice.minimal import MinimalWebApp
    from tests.test_safety_review_v2 import answer
    bundle = current.clients()
    calls = []
    original_content = '仅用于传输回归的逐字正文。'
    def changed():
        if mutation == 'stop':
            app.call_guard.halt()
        elif mutation == 'owner':
            ctx.user_id = 'different-owner'
        elif mutation == 'session':
            ctx.session_id = 'different-session'
        elif mutation == 'forget':
            app.forget_run()
    def opened(request, *, timeout, trace):
        wire = json.loads(request.data)
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append(wire)
        if wire['max_tokens'] == 4096:
            data = current.response(bundle.online)
            data['choices'][0]['message']['content'] = json.dumps({
                'response_segments': [{'text': original_content}], 'safety_concern': 'none'}, ensure_ascii=False)
            return http_response(stream_bytes(stream_frames(data)), trace=trace, on_body=changed)[0]
        assert wire['max_tokens'] == 1024
        data = current.response(bundle.safety_review)
        data['choices'][0]['message']['content'] = json.dumps(answer(wire=wire), ensure_ascii=False)
        return http_response(stream_bytes(stream_frames(data)), trace=trace)[0]
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    app = MinimalWebApp(tmp_path, clients=bundle, model_max_requests=10,
        model_stop_cost_usd='1', model_reservation_policy=current.policy())
    clock = [1000.0]
    app.call_guard._pacing_clock = lambda: clock[0]
    app.call_guard._pacing_sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    sid = app.start()['state']['session_id']
    _, ctx = app._sessions[app._settled_owner]
    try:
        try:
            result = app.submit('中性软件输入。', 'stream-app-1', sid)
        except (RuntimeError, ValueError):
            if mutation == 'none':
                raise
            result = {'ok': False}
        if mutation == 'none':
            assert result['ok'] and app._run['turns'][-1]['response'] == original_content
            assert len(calls) == 2
        else:
            assert all(row.get('response') != original_content for row in app._run['turns'])
        state = app.call_guard.snapshot()
        assert state['paid_response_replays'] == 0
        first = state['attempts'][0]
        assert first['status'] == 'received'
        assert first['usage']['cost'] == 0.00001
        assert first['body_read']['application_complete'] is True
        assert first['body_read']['http_message_complete'] is False
        assert first['delivery_context_validation']['context_bound'] is True
        if mutation != 'none':
            assert first['delivery_context_validation']['status'] == 'rejected'
            assert first['failure_phase'] == 'delivery_context_validation'
            assert first['failure_code'] == 'online_delivery_context_changed'
        if mutation == 'forget':
            assert not list(app.call_guard.public_dir.glob('*.json'))
            assert not list(app.call_guard.physical_dir.glob('*/first-visible.json'))
    finally:
        try:
            app.close()
        except (RuntimeError, ValueError, OSError):
            pass


def test_actual_close_waits_for_same_owner_turn_without_cancelling_a_verified_receipt(tmp_path, monkeypatch):
    from aria.voice.minimal import MinimalWebApp
    from tests.test_safety_review_v2 import answer
    bundle = current.clients()
    entered, release, closing, closed = [threading.Event() for _ in range(4)]
    calls, results, failures = [], [], []
    def body_pause():
        entered.set()
        assert release.wait(5), 'test did not release the in-memory response'
    def opened(request, *, timeout, trace):
        wire = json.loads(request.data)
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append(wire)
        data = current.response(bundle.online if wire['max_tokens'] == 4096 else bundle.safety_review)
        if wire['max_tokens'] == 4096:
            return http_response(stream_bytes(stream_frames(data)), trace=trace, on_body=body_pause)[0]
        data['choices'][0]['message']['content'] = json.dumps(answer(wire=wire))
        return http_response(stream_bytes(stream_frames(data)), trace=trace)[0]
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    app = MinimalWebApp(tmp_path, clients=bundle, model_max_requests=10,
        model_stop_cost_usd='1', model_reservation_policy=current.policy())
    clock = [1000.0]
    app.call_guard._pacing_clock = lambda: clock[0]
    app.call_guard._pacing_sleep = lambda n: clock.__setitem__(0, clock[0] + n)
    sid = app.start()['state']['session_id']
    def submit():
        try:
            results.append(app.submit('中性软件输入。', 'stream-close-1', sid))
        except BaseException as error:
            failures.append(error)
    def close():
        closing.set()
        try:
            app.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()
    sender, closer = threading.Thread(target=submit), threading.Thread(target=close)
    try:
        sender.start()
        assert entered.wait(5)
        closer.start()
        assert closing.wait(5)
        # close() takes the existing owner lock before it changes closing state.
        assert not closed.wait(.05) and app._closing is False
    finally:
        release.set()
        sender.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not sender.is_alive() and not closer.is_alive() and not failures
    assert closed.is_set() and len(results) == 1 and results[0]['ok']
    assert len(calls) == 2  # One online answer and its existing safety review.
    state = app.call_guard.snapshot()
    assert state['stop_reason'] == 'ledger_closed' and state['paid_response_replays'] == 0
    assert state['attempts'][0]['delivery_context_validation']['status'] == 'passed'
    assert state['attempts'][0]['body_read']['application_complete'] is True


@pytest.mark.parametrize('role', ['memory', 'safety_review', 'confirm_judge'])
@pytest.mark.parametrize('chunk_size', [None, 1, 113])
def test_other_current_roles_share_reader_and_native_schema(tmp_path, monkeypatch, role, chunk_size):
    from aria.engine.policy.max_release.sse_adapter import validate_structured_sse_client
    from aria.engine.policy.max_release.reflection_sse import structured_receipt_body_complete
    client = getattr(current.clients(), role)
    data = current.response(client)
    calls = install_memory_http(monkeypatch, stream_bytes(stream_frames(data)), chunk_size=chunk_size)
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        result = previous.invoke(guard, client)
        row, = guard.snapshot()['attempts']
        assert result and len(calls) == 1 and row['status'] == 'received'
        assert calls[0][0]['stream'] is True and calls[0][0]['response_format']['type'] == 'json_schema'
        assert calls[0][0]['provider']['only'] == ['together']
        assert calls[0][1] == row['effective_physical_timeout_seconds'] == 90
        assert row['structured_stream_contract'] == validate_structured_sse_client(client)
        assert structured_receipt_body_complete(row, data['choices'][0]['message']['content'])
        assert row['body_read']['complete'] is False and row['body_read']['current_chunk_terminated'] is True
        assert guard.snapshot()['known_cost_usd'] == '0.00001' and guard.snapshot()['reserved_cost_usd'] == '0'


@pytest.mark.parametrize('role', ['memory', 'safety_review', 'confirm_judge', 'reflection'])
@pytest.mark.parametrize('failure', ['missing_done', 'missing_usage', 'missing_cost', 'missing_chunk_crlf',
    'identity_changed', 'wrong_model', 'wrong_provider', 'half_event'])
def test_other_current_roles_hold_incomplete_or_unverified_stream(tmp_path, monkeypatch, role, failure):
    client = getattr(current.clients(), role)
    data = current.response(client)
    if failure == 'missing_cost':
        del data['usage']['cost']
    if failure == 'wrong_model':
        data['model'] = 'synthetic/other-model'
    if failure == 'wrong_provider':
        data['provider'] = 'Other'
    frames = stream_frames(data)
    if failure == 'missing_usage':
        frames.pop()
    if failure == 'identity_changed':
        frames[1]['id'] = 'gen-otherfixture123'
    body = stream_bytes(frames, done=failure != 'missing_done')
    if failure == 'half_event':
        body = body[:len(body) // 2]
    calls = install_memory_http(monkeypatch, body, current_crlf=failure != 'missing_chunk_crlf')
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client)
        state = guard.snapshot(); row, = state['attempts']
        assert state['stopped'] and state['unknown_cost_requests'] == 1 and len(calls) == 1
        assert state['known_cost_usd'] is None and 'cost' not in row['usage']
        assert Decimal(state['reserved_cost_usd']) == Decimal(row['reservation']['reserved_cost_usd'])
        assert state['paid_response_replays'] == 0 and row['physical_attempt'] == 1


@pytest.mark.parametrize('role', ['safety_review', 'confirm_judge'])
@pytest.mark.parametrize('kind', ['length', 'empty'])
@pytest.mark.parametrize('alias', [False, True])
def test_accounted_other_online_role_failure_stays_local(tmp_path, monkeypatch, role, kind, alias):
    client = getattr(current.clients(), role)
    data = current.response(client)
    if alias:
        data['model'] = 'deepseek/deepseek-v4.1-flash-20261017'
    if kind == 'length':
        data['choices'][0]['finish_reason'] = 'length'
    else:
        data['choices'][0]['message']['content'] = ''
    calls = install_memory_http(monkeypatch, stream_bytes(stream_frames(data)))
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with guard.scope('turn', 'invalid-content', 'fixture-session'):
            with pytest.raises(MinimalCallStopped):
                client.complete(previous.messages(client))
        state = guard.snapshot(); row, = state['attempts']
        assert row['status'] == 'failed' and not state['stopped']
        assert state['known_cost_usd'] == '0.00001' and state['reserved_cost_usd'] == '0'
        assert state['unknown_cost_requests'] == 0 and len(calls) == 1
        assert row['body_read']['complete'] is False and row['body_read']['application_complete'] is True
        assert (guard._settled_safety_review_format_failure(row) if role == 'safety_review'
                else guard._settled_confirm_format_failure(row))
        changed = deepcopy(row); changed['stream_read']['generation_id'] = 'gen-crossedfixture123'
        assert not (guard._settled_safety_review_format_failure(changed) if role == 'safety_review'
                    else guard._settled_confirm_format_failure(changed))
        valid = current.response(client)
        install_memory_http(monkeypatch, stream_bytes(stream_frames(valid)))
        assert previous.invoke(guard, client, operation='only-new-call')
        assert guard.snapshot()['attempts'][0] == row


@pytest.mark.parametrize('kind', ['length', 'empty', 'schema'])
@pytest.mark.parametrize('alias', [False, True])
@pytest.mark.parametrize('framing', ['chunked', 'content_length'])
def test_real_memory_http_failure_seals_once_and_reopen_does_not_replay(tmp_path, monkeypatch, kind, alias, framing):
    from tests import test_installable_settled_reflection as helpers
    monkeypatch.setattr(helpers, 'make_clients', current.clients)
    monkeypatch.setattr(helpers, 'prices', current.policy)
    def fixture_response(client):
        value = current.response(client)
        if alias and client.role == 'memory':
            value['model'] = 'deepseek/deepseek-v4.1-flash-20261017'
        return value
    monkeypatch.setattr(helpers, 'response', fixture_response)
    app, calls = helpers.setup(tmp_path, monkeypatch, '' if kind == 'empty' else '{"episodes":[',
        fault='length' if kind == 'length' else None, failure_role='memory')
    existing = tls_dispatch.open_observed
    def opened(request, **kwargs):
        fake = existing(request, **kwargs)
        assert json.loads(request.data)['stream'] is True
        return (content_length_response(fake.body.getvalue(), trace=kwargs['trace'])
                if framing == 'content_length' else http_response(fake.body.getvalue(), trace=kwargs['trace'])[0])
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    try:
        assert not helpers.one_session(app, 1)['ok']
        before = app.call_guard.snapshot(); row = before['attempts'][-1]
        assert row['role'] == 'memory' and row['status'] == 'failed' and not before['stopped']
        assert before['unknown_cost_requests'] == 0 and row['body_read']['complete'] is (framing == 'content_length')
        from aria.engine.policy.max_release.reflection_failure import validate_receipt
        validate_receipt(row)
        if framing == 'content_length':
            for missing in ('structured_stream_contract', 'structured_stream_receipt', 'stream_read'):
                changed = deepcopy(row); del changed[missing]
                with pytest.raises(ValueError):
                    validate_receipt(changed)
        assert app.store.load(app._settled_owner).offline_session_audit[-1].outcome == 'memory_failed'
    finally:
        app.close()
    app = helpers.reopen(tmp_path)
    try:
        assert not app.call_guard.snapshot()['stopped'] and app.start()['ok']
        assert app.call_guard.snapshot()['attempts'] == before['attempts']
        assert calls.count('memory') == 1 and 'reflection' not in calls
    finally:
        app.close()


@pytest.mark.parametrize('role', ['memory', 'safety_review', 'confirm_judge'])
@pytest.mark.parametrize('damage', ['role', 'generation', 'request', 'body_hash', 'finish'])
def test_other_role_receipt_cannot_cross_request_or_source(tmp_path, monkeypatch, role, damage):
    from aria.engine.policy.max_release.reflection_sse import structured_receipt_body_complete
    client = getattr(current.clients(), role)
    data = current.response(client); content = data['choices'][0]['message']['content']
    install_memory_http(monkeypatch, stream_bytes(stream_frames(data)))
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client); previous.invoke(guard, client)
        row, = guard.snapshot()['attempts']
        assert structured_receipt_body_complete(row, content)
        changed = deepcopy(row)
        if damage == 'role':
            changed['structured_stream_contract']['role'] = 'reflection'
        elif damage == 'generation':
            changed['stream_read']['generation_id'] = 'gen-crossedfixture123'
        elif damage == 'request':
            changed['structured_stream_receipt']['request_sha256'] = '0' * 64
        elif damage == 'body_hash':
            changed['body_read']['body_sha256'] = '0' * 64
        else:
            changed['structured_stream_receipt']['finish_reason'] = 'length'
        assert not structured_receipt_body_complete(changed, content)
        assert not structured_receipt_body_complete(row, content + 'source changed')


@pytest.mark.parametrize('role', ['memory', 'safety_review', 'confirm_judge'])
def test_other_role_response_closed_mid_chunk_never_uses_partial_content(tmp_path, monkeypatch, role):
    client = getattr(current.clients(), role)
    data = current.response(client); calls = []
    def opened(request, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        calls.append(json.loads(request.data))
        holder = {}
        def close_source():
            holder['response'].close()
        response, _ = http_response(stream_bytes(stream_frames(data)), trace=trace,
            chunk_size=7, on_body=close_source)
        holder['response'] = response
        return response
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client)
        state = guard.snapshot(); row, = state['attempts']
        assert state['stopped'] and state['unknown_cost_requests'] == 1
        assert state['known_cost_usd'] is None and len(calls) == 1
        assert Decimal(state['reserved_cost_usd']) == Decimal(row['reservation']['reserved_cost_usd'])
        assert 'public_first_output' not in row and state['paid_response_replays'] == 0


def content_length_response(body, *, trace):
    raw = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: '
           + str(len(body)).encode() + b'\r\n\r\n' + body)
    response = _ObservedHTTPResponse(_ObservedSocket(MemorySocket(raw), trace))
    response.begin()
    return response


@pytest.mark.parametrize('role', ['memory', 'safety_review', 'confirm_judge', 'reflection'])
@pytest.mark.parametrize('kind', ['length', 'empty'])
def test_complete_length_sse_cannot_downgrade_to_finite_receipt(tmp_path, monkeypatch, role, kind):
    from aria.engine.policy.max_release.reflection_sse import structured_receipt_body_complete
    client = getattr(current.clients(), role); data = current.response(client)
    if kind == 'length':
        data['choices'][0]['finish_reason'] = 'length'
    else:
        data['choices'][0]['message']['content'] = ''
    def opened(request, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        return content_length_response(stream_bytes(stream_frames(data)), trace=trace)
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(current.guard(tmp_path)) as guard:
        guard.attach(client)
        with guard.scope('review' if role in ('memory', 'reflection') else 'turn', 'complete-bad', 'fixture-session'):
            with pytest.raises(MinimalCallStopped):
                client.complete(previous.messages(client))
        row, = guard.snapshot()['attempts']
        assert row['body_read']['complete'] is True
        assert structured_receipt_body_complete(row)
        assert row['usage']['cost'] == 0.00001
        for missing in ('structured_stream_contract', 'structured_stream_receipt', 'stream_read'):
            changed = deepcopy(row); del changed[missing]
            assert not structured_receipt_body_complete(changed)
            if role == 'safety_review':
                assert not guard._settled_safety_review_format_failure(changed)
            elif role == 'confirm_judge':
                assert not guard._settled_confirm_format_failure(changed)


@pytest.mark.parametrize('kind', ['length', 'empty', 'null_content'])
@pytest.mark.parametrize('framing', ['chunked', 'content_length'])
def test_current_reflection_accounted_rejection_preserves_memory_and_next_session(
        tmp_path, monkeypatch, kind, framing):
    from aria.voice.minimal import MinimalWebApp
    from aria.engine.policy.max_release.reflection_failure import validate_receipt
    from tests import test_installable_settled_reflection as helpers
    from tests.test_safety_review_v2 import answer
    bundle = current.clients(); calls = []; saved_memory = []
    def opened(request, *, timeout, trace):
        wire = json.loads(request.data)
        role = {4096: 'online', 4000: 'memory', 2800: 'reflection', 1024: 'safety_review'}[wire['max_tokens']]
        calls.append(role); data = current.response(getattr(bundle, role))
        if role == 'safety_review':
            data['choices'][0]['message']['content'] = json.dumps(answer(wire=wire))
        if role == 'reflection':
            saved_memory.append(deepcopy(app.store.load(app._settled_owner).episodic_memory))
            data['model'] = 'deepseek/deepseek-v4.1-flash-20261017'
            if kind == 'length':
                # A complete-looking JSON is still content failure when the
                # provider declares length; no person update may be inferred.
                data['choices'][0]['finish_reason'] = 'length'
            else:
                data['choices'][0]['message']['content'] = None if kind == 'null_content' else ''
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        body = stream_bytes(stream_frames(data))
        return (content_length_response(body, trace=trace) if framing == 'content_length'
                else http_response(body, trace=trace)[0])
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    app = MinimalWebApp(tmp_path, clients=bundle, model_max_requests=30,
        model_stop_cost_usd='1', model_reservation_policy=current.policy())
    helpers.pace(app.call_guard)
    try:
        assert not helpers.one_session(app, 1)['ok']
        state = app.call_guard.snapshot(); row = state['attempts'][-1]
        assert row['role'] == 'reflection' and row['status'] == 'failed' and not state['stopped']
        assert row['validation_error'] == ('invalid_or_truncated_response' if kind == 'length' else 'invalid_response_message')
        assert state['known_cost_usd'] == '0.00008' and state['reserved_cost_usd'] == '0'
        validate_receipt(row)
        if framing == 'content_length':
            changed = deepcopy(row); del changed['structured_stream_contract']
            with pytest.raises(ValueError):
                validate_receipt(changed)
        profile = app.store.load(app._settled_owner)
        assert profile.episodic_memory == saved_memory[0] and not profile.person_hypotheses
        assert profile.offline_session_audit[-1].memory_completed
        assert profile.offline_session_audit[-1].outcome == 'reflection_failed'
        assert not profile.offline_sessions and not app.state()['requires_review']
    finally:
        app.close()
    reopened = MinimalWebApp(tmp_path, clients=current.clients(), model_max_requests=30,
        model_stop_cost_usd='1', model_reservation_policy=current.policy())
    helpers.pace(reopened.call_guard)
    try:
        assert not reopened.call_guard.snapshot()['stopped']
        sid = reopened.start()['state']['session_id']
        assert reopened.submit('后续中性软件输入。', 'after-reflection-failure', sid)['ok']
        assert calls.count('reflection') == 1 and calls.count('memory') == 1
        assert reopened.call_guard.snapshot()['attempts'][:len(state['attempts'])] == state['attempts']
    finally:
        reopened.close()


@pytest.mark.parametrize('kind', ['length', 'empty', 'null_content'])
def test_legacy_schemafree_reflection_keeps_original_strict_terminal(tmp_path, monkeypatch, kind):
    client = previous.clients().reflection; data = previous.response(client)
    if kind == 'length':
        data['choices'][0]['finish_reason'] = 'length'
    else:
        data['choices'][0]['message']['content'] = None if kind == 'null_content' else ''
    calls = install_memory_http(monkeypatch, stream_bytes(stream_frames(data)))
    with closing(previous.guard(tmp_path)) as guard:
        guard.attach(client)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(guard, client)
        state = guard.snapshot(); row, = state['attempts']
        assert state['stopped'] and state['unknown_cost_requests'] == 1 and len(calls) == 1
        assert 'structured_stream_contract' not in row and 'receipt_mode' not in row['reflection_provider_contract']
        assert state['known_cost_usd'] is None and Decimal(state['reserved_cost_usd']) > 0
