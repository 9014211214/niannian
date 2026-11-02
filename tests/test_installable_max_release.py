"""Mechanical installation/wiring checks with synthetic responses; zero model calls."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
from email.message import Message
import io
import json
from pathlib import Path
import socket
import ssl
import urllib.error

import pytest

import aria
from aria.release import create_max_release_clients, create_release_clients
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from aria.engine.policy.max_release import tls_dispatch
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.qwen_max_structured import build_qwen_max_structured_route
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.engine.memory.diagnosis import reflection_system_prompt
from tests import test_qwen_max_candidate as q
from tests import test_memory_structured_candidate as m
from aria.engine.policy.release_configuration import derive_policy


def metadata():
    qm, qe = q.metadata(); mm, me = m.metadata()
    for e in (qe, me):
        e['data']['endpoints'][0].update(context_length=262144, max_prompt_tokens=100000)
    return tuple(json.dumps(x, ensure_ascii=False).encode() for x in
                 ({'data': qm['data'] + mm['data']}, qe, me))


def make_clients():
    models, online, offline = metadata()
    return create_max_release_clients(model_calls_enabled=True,
        api_key='synthetic-inert-key', base_url='https://openrouter.ai/api/v1',
        online_structured_route=build_qwen_max_structured_route(models, online),
        memory_structured_route=build_memory_structured_route(models, offline))


def prices():
    return derive_policy(*metadata(), cover_all_price_tiers=True,
                         release_profile='release_max_v1')[0]


def messages(client):
    if client.role in ('online_fast', 'safety_review'): return q.messages(client)
    if client.role == 'memory': return m.messages()
    return [{'role': 'system', 'content': reflection_system_prompt(True)},
            {'role': 'user', 'content': '本次会话：[turn 0] 用户:中性软件输入。'}]


def response(client):
    if client.role in ('online_fast', 'safety_review'): return q.response(client)
    raw = json.dumps(m.empty() if client.role == 'memory' else
                     {'protocol_version': '2.0', 'reviews': [], 'changes': []})
    if client.role == 'reflection': raw = '```json\n' + raw + '\n```'
    return {'id': 'gen-syntheticfixture123456', 'model': client.model, 'provider': 'Anthropic',
        'usage': {'cost': .0001, 'prompt_tokens': 20, 'completion_tokens': 40, 'total_tokens': 60},
        'choices': [{'finish_reason': 'stop', 'message': {'content': raw}}]}


def stream(data):
    identity = {k: data[k] for k in ('id', 'model', 'provider')}
    choice=data['choices'][0]
    frames = [{**identity, 'choices': [{'index': 0, 'delta': deepcopy(choice['message']),
              'finish_reason': None}]},
        {**identity, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}]},
        {**identity, 'choices': [], 'usage': data['usage']}]
    return (''.join('data: ' + json.dumps(f) + '\n\n' for f in frames) + 'data: [DONE]\n\n').encode()


class Response:
    status = 200
    def __init__(self, data, streaming=False):
        self.body = io.BytesIO(stream(data) if streaming else json.dumps(data).encode())
        self.headers = Message()
        self.headers['Content-Length'] = str(len(self.body.getvalue()))
        self.headers['Content-Type'] = 'text/event-stream' if streaming else 'application/json'
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size=-1): return self.body.read(size)
    def read1(self, size=-1): return self.body.read1(size)
    def close(self): self.body.close()


@pytest.fixture(autouse=True)
def no_network_credentials(monkeypatch):
    def forbidden(*a, **k): raise AssertionError('network or credential access forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


def guard(tmp_path, budget='1.00'):
    g = MaxReleaseCallGuard(tmp_path / 'calls.sqlite3', max_requests=8,
                           max_observed_cost_usd=budget, reservation_policy=prices())
    clock = [1000.0]
    g._pacing_clock = lambda: clock[0]
    g._pacing_sleep = lambda delta: clock.__setitem__(0, clock[0] + delta)
    return g


def invoke(g, c, operation='fixture-op'):
    with g.scope('turn' if c.role in ('online_fast', 'safety_review') else 'review', operation, 'fixture-session'):
        return c.complete(messages(c))


def malformed_compact_slot(collection, field, value):
    from aria.engine.policy.reflection_compact import encode_compact_reflection
    from tests.test_reflection_structured_contract_v23 import payload
    body = encode_compact_reflection(payload())
    item = body[collection][0]
    for name in field.split('.')[:-1]:
        item = item[name]
    item[field.split('.')[-1]] = deepcopy(value)
    return json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize('route', ['sonnet', 'deepseek'])
@pytest.mark.parametrize('collection,field,value', [
    ('reviews', 'hypothesis_id', ['first', 'second']),
    ('reviews', 'claim_quote', ['first', 'second']),
    ('reviews', 'scope_quote', ['first', 'second']),
    ('changes', 'id', ['first', 'second']),
    ('changes', 'scope', ['条件一', '条件二']),
    ('changes', 'lineage', [{}, {}]),
    ('changes', 'target_value', [0, 1]),
    ('changes', 'atomic_claim.outcome', ['first', 'second']),
    ('changes', 'scope', 'not-an-array'),
    ('changes', 'scope', None),
    ('changes', 'scope', [None]),
])
def test_compact_slot_rejection_keeps_receipt_and_respects_version_isolation(
        tmp_path, monkeypatch, route, collection, field, value):
    from aria.engine.policy.reflection_compact import (
        decode_compact_reflection, decode_indexed_compact_reflection, encode_indexed_compact_reflection)
    from aria.engine.policy.reflection_json import parse_reflection_json
    from tests.test_reflection_structured_contract_v23 import payload
    from tests import test_deepseek_offline_route as deepseek
    client = (make_clients() if route == 'sonnet' else deepseek.clients()).reflection
    data = (response if route == 'sonnet' else deepseek.response)(client)
    indexed=client.reflection_return_format=='json_schema_v4'
    raw = malformed_compact_slot(collection, field, value)
    if indexed:
        body=encode_indexed_compact_reflection(payload())
        target=body[collection][0]
        for part in field.split('.')[:-1]:target=target[part]
        target[field.split('.')[-1]]=deepcopy(value)
        raw=json.dumps(body,ensure_ascii=False)
    assert parse_reflection_json(raw, itemwise_review=True) == raw
    if indexed:
        canonical=decode_indexed_compact_reflection(raw,itemwise_review=True)
        rejected=json.loads(canonical)[collection][0]
        assert ('id' if collection=='reviews' else 'review_id') not in rejected
        # The malformed item stays invalid; a complete transport is not a
        # successful review or a grant to create a hypothesis.
        with pytest.raises(ValueError):parse_reflection_json(canonical,itemwise_review=False)
    else:
        with pytest.raises(ValueError, match='invalid_compact_reflection_slot'):
            decode_compact_reflection(raw, itemwise_review=True)
    data['choices'][0]['message']['content'] = raw
    before = deepcopy(data); calls = []
    def opened(req, **kw):
        calls.append(req)
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        return Response(data, True)
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    selected = guard(tmp_path) if route == 'sonnet' else deepseek.guard(tmp_path)
    with closing(selected) as g:
        g.attach(client)
        if indexed:
            assert invoke(g,client)==canonical
        else:
            with pytest.raises(MinimalCallStopped):invoke(g, client)
        state = g.snapshot(); row, = state['attempts']
        if indexed:
            assert row['status']=='received' and row['reflection_wrapper']['status']=='validated'
            assert 'validation_error' not in row and not state['stopped']
        else:
            assert row['status'] == 'failed'
            assert row['failure_phase'] == 'receipt_validation'
            assert row['validation_error'] == row['failure_code'] == 'reflection_schema_invalid'
            assert row['reflection_wrapper']['status'] == 'rejected'
        assert row['usage']['cost'] == data['usage']['cost']
        assert Decimal(state['known_cost_usd']) == Decimal(str(data['usage']['cost']))
        assert state['unknown_cost_requests'] == 0 and state['reserved_cost_usd'] == '0'
        assert not row.get('fee_verification_error') and not row.get('unverified_cost_usd')
        first = json.loads(Path(row['public_first_output']['path']).read_text(encoding='utf-8'))
        assert first['public_first_output'][0]['content'] == raw and data == before
        assert len(calls)==1
        # A v4 item rejection is left to the existing updater. A legacy
        # whole-response rejection still cannot bypass the App transaction.
        if not indexed:
            assert state['stopped']
            with pytest.raises(MinimalCallStopped):invoke(g, client, 'not-a-replay')
        assert len(calls) == 1


def test_import_origin_preview_and_legacy_default():
    assert Path(aria.__file__).resolve().is_relative_to(Path.cwd().resolve())
    assert not create_max_release_clients().model_calls_enabled
    assert not create_release_clients().model_calls_enabled
    from aria.release import ONLINE_MODEL
    assert ONLINE_MODEL == 'qwen/qwen3.8-flash'


@pytest.mark.parametrize('role', ['online', 'memory', 'reflection', 'safety_review'])
def test_four_roles_actual_dispatch_decode_and_first_output(tmp_path, monkeypatch, role):
    client = getattr(make_clients(), role); data = response(client); calls = []
    def opened(req, **kw):
        calls.append((json.loads(req.data), kw['timeout']))
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        return Response(data, client.role == 'reflection')
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(guard(tmp_path)) as g:
        g.attach(client)
        result = invoke(g, client)
        state = g.snapshot(); row, = state['attempts']
        assert row['status'] == 'received' and state['known_cost_usd'] == '0.0001'
        assert 'stop_diagnostic' not in state
        assert state['reserved_cost_usd'] == '0' and len(calls) == 1
        assert row['pacing']['min_interval_seconds'] == 120
        assert row['pacing']['wait_ms'] >= 120000
        first = json.loads(Path(row['public_first_output']['path']).read_text(encoding='utf-8'))
        assert first['public_first_output'][0]['content'] == data['choices'][0]['message']['content']
        assert first['hidden_reasoning_retained'] is False
        assert 'synthetic-inert-key' not in json.dumps(state)
        wire, timeout = calls[0]
        assert wire['model'] == client.model and timeout == 90
        assert row['effective_physical_timeout_seconds'] == timeout
        assert wire['provider']['allow_fallbacks'] is False
        assert wire['provider']['data_collection'] == 'deny'
        if role == 'reflection':
            assert wire['stream'] is True and 'response_format' not in wire
            assert json.loads(result) == {'protocol_version': '2.0', 'reviews': [], 'changes': []}
            assert row['reflection_wrapper']['wrapper'] == 'single_json_fence'
        elif role == 'online':
            assert json.loads(result)['response'] == '中性软件正文。'
        else: assert json.loads(result) == json.loads(data['choices'][0]['message']['content'])


@pytest.mark.parametrize('failure', ['429', 'missing_cost', 'truncated', 'wrong_provider'])
def test_unknown_or_failed_response_no_replay_preserves_hold(tmp_path, monkeypatch, failure):
    c = make_clients().online; data = response(c); calls = []
    if failure == 'missing_cost': del data['usage']['cost']
    if failure == 'wrong_provider': data['provider'] = 'Other'
    def opened(req, **kw):
        calls.append(req)
        if failure == '429': raise urllib.error.HTTPError(req.full_url, 429, 'fixture', {}, None)
        r = Response(data)
        if failure == 'truncated': r.body = io.BytesIO(r.body.getvalue()[:19])
        return r
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(guard(tmp_path)) as g:
        g.attach(c)
        with pytest.raises(MinimalCallStopped): invoke(g, c)
        state = g.snapshot()
        assert state['stopped'] and len(calls) == 1 and state['calls_used'] == 1
        assert Decimal(state['reserved_cost_usd']) > 0 and Decimal(state['known_cost_usd'] or '0') == 0
        with pytest.raises(MinimalCallStopped): invoke(g, c, 'another-op')
        assert len(calls) == 1
    with closing(guard(tmp_path)) as reopened:
        assert reopened.snapshot()['stopped']
        assert reopened.snapshot()['reserved_cost_usd'] == state['reserved_cost_usd']


def test_full_reservation_stops_before_any_dispatch(tmp_path, monkeypatch):
    c = make_clients().online
    with closing(guard(tmp_path, '0.000001')) as g:
        g.attach(c)
        with pytest.raises(MinimalCallStopped): invoke(g, c)
        assert g.snapshot()['calls_used'] == 0


@pytest.mark.parametrize('status,expected', [
    (None, 'finite_no_http_response_cost_unverified'),
    (429, 'finite_http_error_cost_unverified'),
    (503, 'finite_http_error_cost_unverified'),
    (200, None),
])
def test_fee_classification_distinguishes_absent_response_from_http_error(status, expected):
    row = {'http_status': status,
           'receipt_identity': {'model_status': 'recognized', 'provider_status': 'recognized',
                                'error_field_status': 'absent'},
           'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
           'reservation': {'input_token_plan_bound': 10, 'output_token_limit': 10}}
    assert MaxReleaseCallGuard._fee_attribution_error(row) == expected


@pytest.mark.parametrize('role', ['online', 'memory', 'reflection', 'safety_review'])
def test_no_response_stop_explains_transport_failure_and_keeps_hold(tmp_path, monkeypatch, role):
    client = getattr(make_clients(), role); calls = []
    def opened(request, **kwargs):
        calls.append(request)
        raise urllib.error.URLError(ssl.SSLEOFError(8, 'synthetic private exception text'))
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(guard(tmp_path)) as g:
        g.attach(client)
        with pytest.raises(MinimalCallStopped): invoke(g, client)
        state = g.snapshot(); row, = state['attempts']
        assert state['stopped'] and state['stop_reason'] == 'model_or_storage_failed'
        assert row.get('http_status') is None and row['usage'] == {}
        assert row['fee_verification_error'] == 'finite_no_http_response_cost_unverified'
        assert state['stop_diagnostic'] == {
            'source': 'latest_recorded_request', 'request_id': row['request_id'],
            'failure_phase': 'pre_response_open', 'failure_code': 'response_open_failed',
            'http_status': None, 'fee_verification_error': 'finite_no_http_response_cost_unverified',
            'exception_kind': 'tls_eof_error', 'errno_category': 'not_applicable'}
        assert state['unknown_cost_requests'] == 1 and Decimal(state['reserved_cost_usd']) > 0
        assert 'synthetic private exception text' not in json.dumps(state)
        with pytest.raises(MinimalCallStopped): invoke(g, client, 'blocked-op')
        assert len(calls) == 1
    with closing(guard(tmp_path)) as reopened:
        after = reopened.snapshot()
        assert after['stopped'] and after['stop_reason'] == state['stop_reason']
        assert after['attempts'] == state['attempts']
        assert after['reserved_cost_usd'] == state['reserved_cost_usd']
        assert after['stop_diagnostic'] == state['stop_diagnostic']


def test_runtime_allowance_mutation_cannot_dispatch(tmp_path):
    c = make_clients().online
    with closing(guard(tmp_path)) as g:
        g.attach(c); g.max_requests += 1
        with pytest.raises(MinimalCallStopped): invoke(g, c)


def test_max_policy_all_prices_and_private_campaign_absence():
    p = prices()
    assert Decimal(p['models'][q.MODEL]['input_usd_per_million']) == Decimal('2.5')
    assert p['models'][q.MODEL]['max_output_tokens'] == 8202
    assert Decimal(p['models'][m.MODEL]['input_usd_per_million']) == Decimal('4')
    assert p['models'][m.MODEL]['max_output_tokens'] == 4000
    with pytest.raises(ValueError, match='requires_all_price_tiers'):
        derive_policy(*metadata(), release_profile='release_max_v1')
    raw = Path('aria/engine/policy/max_release/guard.py').read_text(encoding='utf-8')
    for private in ('prior_global', 'shared_recovery_slots', 'active-live'):
        assert private not in raw


@pytest.mark.parametrize('live', [False, True])
def test_actual_cli_assembles_profile_without_opening_a_socket(tmp_path, monkeypatch, live):
    from aria import cli
    from aria.voice import minimal
    seen = {}; key_reads = []
    monkeypatch.setattr(cli, 'load_local_env', lambda: key_reads.append('explicit-env-load'))
    monkeypatch.setattr('aria.runtime.openrouter_api_key', lambda: key_reads.append('key') or 'inert-fixture-key')
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', lambda: key_reads.append('key') or 'inert-fixture-key')
    class Server:
        server_port = 8080
        def __init__(self, app): self.app = app
        def serve_forever(self):
            seen['mode'] = self.app.mode
            seen['clients'] = self.app.clients
            if live:
                assert isinstance(self.app.call_guard, MaxReleaseCallGuard)
                assert self.app.call_guard.snapshot()['calls_used'] == 0
                assert self.app.call_guard.threshold == '1'
                assert self.app.call_guard.max_requests == 8
            else: assert self.app.call_guard is None
        def server_close(self): self.app.close()
    monkeypatch.setattr(minimal, 'make_minimal_server', lambda app, *a: Server(app))
    args = ['--release-profile', 'release_max_v1', '--minimal-data-dir', str(tmp_path / 'data')]
    if live:
        for name, raw in zip(('models', 'online', 'offline'), metadata()):
            (tmp_path / (name + '.json')).write_bytes(raw)
        (tmp_path / 'policy.json').write_text(json.dumps(prices()), encoding='utf-8')
        args += ['--enable-model-calls', '--model-max-requests', '8', '--model-stop-cost-usd', '1.00',
            '--model-reservation-policy', str(tmp_path / 'policy.json'),
            '--online-models-metadata', str(tmp_path / 'models.json'),
            '--online-endpoints-metadata', str(tmp_path / 'online.json'),
            '--offline-endpoints-metadata', str(tmp_path / 'offline.json')]
    assert cli.web_main(args) == 0
    assert seen['mode'] == ('live' if live else 'local_preview')
    assert bool(key_reads) is live
    if live: assert seen['clients'].release_profile == 'release_max_v1'


def test_policy_module_is_available_without_repository_script():
    from aria.engine.policy import release_configuration
    assert release_configuration.derive_policy(*metadata(), cover_all_price_tiers=True,
        release_profile='release_max_v1')[0] == prices()
