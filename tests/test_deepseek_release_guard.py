"""Offline model-switch integration; synthetic HTTP receipts are not effects."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from aria.engine.memory.preference_dialogue import confirm_judge_messages
from aria.engine.policy.max_release import sse_dispatch, tls_dispatch, tls_transport
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard, MODELS, BILLABLE_OUTPUTS
from aria.engine.policy.max_release.public_capture import projection
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_structured import (
    DEEPSEEK_MODEL, DEEPSEEK_PROVIDER, build_online_structured_route, validate_release_online_client)
from aria.engine.policy.release_configuration import derive_policy
from aria.engine.policy.response_body import ResponseDeadline
from aria.release import create_deepseek_release_clients
from tests import test_installable_max_release as old


CANONICAL = 'deepseek/deepseek-v4.1-flash-20261017'


def metadata(provider=DEEPSEEK_PROVIDER):
    models, online, offline = [json.loads(raw) for raw in old.metadata()]
    models['data'][0].update(id=DEEPSEEK_MODEL, canonical_slug=CANONICAL,
                            reasoning={'mandatory': False, 'supported_efforts': ['none', 'high']})
    online['data']['id'] = DEEPSEEK_MODEL
    online['data']['endpoints'][0].update(
        model_id=DEEPSEEK_MODEL, tag=provider,
        provider_name={'together': 'Together', 'fireworks': 'Fireworks'}[provider],
        pricing={'prompt': '0.0000003', 'completion': '0.0000012',
                 'input_cache_read': '0.000000006', 'discount': 0})
    return tuple(json.dumps(value).encode() for value in (models, online, offline))


def clients(provider=DEEPSEEK_PROVIDER):
    models, online, offline = metadata(provider)
    return create_deepseek_release_clients(model_calls_enabled=True,
        api_key='inert-offline-fixture', base_url='https://openrouter.ai/api/v1',
        online_structured_route=build_online_structured_route(models, online,
            model=DEEPSEEK_MODEL, provider=provider),
        memory_structured_route=build_memory_structured_route(models, offline))


def policy(provider=DEEPSEEK_PROVIDER):
    result = derive_policy(*metadata(), cover_all_price_tiers=True,
                           release_profile='release_deepseek_v1')[0]
    if provider != DEEPSEEK_PROVIDER:
        # Explicit historical policy fixture; rates and limits match both synthetic endpoints.
        models, online, _ = metadata(provider)
        binding = build_online_structured_route(models, online, model=DEEPSEEK_MODEL,
                                               provider=provider).binding()
        result['models'][DEEPSEEK_MODEL].update(provider=provider,
            response_providers=[binding['provider_name']], metadata_sha256=binding['endpoint_metadata_sha256'])
    return result


def guard(path, prices=None, provider=DEEPSEEK_PROVIDER):
    result = MaxReleaseCallGuard(path / 'calls.sqlite3', max_requests=8,
        max_observed_cost_usd='1', reservation_policy=prices if prices is not None else policy(provider))
    clock = [1000.0]
    result._pacing_clock = lambda: clock[0]
    result._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    return result


def messages(client):
    if client.role == 'online_fast':
        return [{'role': 'system', 'content': '协议版本：online_turn_decision_v7'},
                {'role': 'user', 'content': '中性软件输入。'}]
    if client.role == 'confirm_judge':
        return confirm_judge_messages('中性说明', '软件范围', '中性标签输入')
    return old.messages(client)


def response(client):
    if client.role in ('memory', 'reflection'):
        return old.response(client)
    if client.role == 'online_fast':
        answer = {'response_segments': [{'text': '中性软件正文。'}], 'safety_concern': 'none'}
    elif client.role == 'confirm_judge':
        answer = {'action': 'confirm'}
    else:
        answer = json.loads(old.response(client)['choices'][0]['message']['content'])
    return {'id': 'gen-offlinefixture123456', 'model': DEEPSEEK_MODEL,
            'provider': validate_release_online_client(client)['provider_name'],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 40, 'total_tokens': 60,
                      'completion_tokens_details': {'reasoning_tokens': 0}, 'cost': 0.00001},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}


def invoke(guard, client, operation='fixture-op'):
    with guard.scope('review' if client.role in ('memory', 'reflection') else 'turn',
                     operation, 'fixture-session'):
        return client.complete(messages(client))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network and credential access forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


@pytest.mark.parametrize('role,limit', [('online', 4096), ('safety_review', 1024),
                                     ('confirm_judge', 60), ('memory', 4000), ('reflection', 2800)])
@pytest.mark.parametrize('provider', ['together', 'fireworks'])
def test_five_roles_use_existing_dispatch_and_exact_reservations(tmp_path, monkeypatch, role, limit, provider):
    client = getattr(clients(provider), role)
    data = response(client)
    seen, timeouts = [], []
    def built(handler, redirect):
        def opened(request, *, timeout):
            seen.append(json.loads(request.data))
            timeouts.append(timeout)
            trace = handler._trace
            assert trace.response_deadline is trace.preparation_recovery.deadline
            assert 0 < trace.response_deadline.remaining() <= 90
            trace.preparation_recovery.bind_request_headers(request)
            trace.preparation_recovery.before_preparation()
            return old.Response(data, role == 'reflection')
        return SimpleNamespace(open=opened)
    monkeypatch.setattr(urllib.request, 'build_opener', built)
    monkeypatch.setattr(tls_dispatch, 'open_observed', tls_transport.open_candidate)
    with closing(guard(tmp_path, provider=provider)) as selected:
        selected.attach(client)
        result = invoke(selected, client)
        state = selected.snapshot()
        row, = state['attempts']
        assert row['status'] == 'received' and not state['stopped']
        assert state['runtime_profile'] == 'release_deepseek_v1'
        assert state['known_cost_usd'] == str(Decimal(str(data['usage']['cost'])))
        assert state['reserved_cost_usd'] == '0'
        assert row['reservation']['output_token_limit'] == limit
        assert row['pacing']['min_interval_seconds'] == 120
        assert client.timeout == row['effective_physical_timeout_seconds'] == 90
        assert timeouts == [90]
        if role == 'reflection':
            assert row['reflection_provider_contract']['physical_timeout_seconds'] == 90
        assert len(seen) == 1
        assert seen[0]['max_tokens'] == limit
        assert seen[0]['reasoning'] == {'enabled': False, 'exclude': True}
        assert seen[0]['provider']['only'] == (['anthropic'] if role in ('memory', 'reflection') else [provider])
        assert seen[0]['provider']['allow_fallbacks'] is False
        assert row['receipt_identity']['provider_status'] == 'recognized'
        first = json.loads(Path(row['public_first_output']['path']).read_bytes())
        assert first['hidden_reasoning_retained'] is False
        assert first['public_first_output'][0]['content'] == data['choices'][0]['message']['content']
        if role == 'confirm_judge':
            assert json.loads(result) == {'action': 'confirm'}


@pytest.mark.parametrize('field', ['timeout', 'reflection_physical_timeout_seconds'])
def test_reflection_timeout_drift_is_rejected_before_dispatch(tmp_path, field):
    client = clients().reflection
    setattr(client, field, 240)
    with closing(guard(tmp_path)) as selected:
        with pytest.raises(ValueError):
            selected.attach(client)
        assert selected.snapshot()['calls_used'] == 0


def test_reflection_physical_deadline_expiry_keeps_hold_without_replay(tmp_path, monkeypatch):
    client = clients().reflection
    clock, opens = [1000.0], []
    monkeypatch.setattr(sse_dispatch, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(sse_dispatch, 'ResponseDeadline',
                        lambda expires: ResponseDeadline(expires, clock=lambda: clock[0]))
    class TimedResponse(old.Response):
        def read1(self, size=-1):
            clock[0] = 1090.001
            return super().read1(size)
    def built(handler, redirect):
        def opened(request, *, timeout):
            opens.append(timeout)
            trace = handler._trace
            assert trace.response_deadline.expires_at == 1090.0
            trace.preparation_recovery.bind_request_headers(request)
            trace.preparation_recovery.before_preparation()
            return TimedResponse(response(client), streaming=True)
        return SimpleNamespace(open=opened)
    monkeypatch.setattr(urllib.request, 'build_opener', built)
    monkeypatch.setattr(tls_dispatch, 'open_observed', tls_transport.open_candidate)
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(MinimalCallStopped):
            invoke(selected, client)
        state = selected.snapshot()
        row, = state['attempts']
        assert opens == [90] and row['effective_physical_timeout_seconds'] == 90
        assert row['status'] == 'failed' and state['stopped']
        assert state['unknown_cost_requests'] == 1
        assert Decimal(state['reserved_cost_usd']) > 0
        assert state['paid_response_replays'] == 0
        with pytest.raises(MinimalCallStopped):
            invoke(selected, client, 'blocked-after-timeout')
        assert opens == [90]


@pytest.mark.parametrize('timeout', [89, 91, 240])
def test_sonnet_stream_opener_rejects_other_physical_timeouts(monkeypatch, timeout):
    deadline = ResponseDeadline(90, clock=lambda: 0)
    preparation = tls_transport.TLSPreparation(check=lambda: None, deadline=deadline,
        allowance=tls_transport.LogicalTLSAllowance())
    trace = tls_transport.CandidateTrace(preparation=preparation, deadline=deadline)
    request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps({'model': 'anthropic/claude-sonnet-5', 'stream': True}).encode())
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid timeout must fail before opener construction')
    monkeypatch.setattr(urllib.request, 'build_opener', forbidden)
    with pytest.raises(ValueError, match='invalid_candidate_open_contract'):
        tls_transport.open_candidate(request, timeout=timeout, trace=trace)


@pytest.mark.parametrize('failure', ['missing_cost', 'wrong_provider', 'other_accepted_provider', 'reasoning_returned', 'output_over_bound'])
def test_failed_identity_or_usage_keeps_cost_or_hold_without_replay(tmp_path, monkeypatch, failure):
    client = clients().confirm_judge
    data = response(client)
    if failure == 'missing_cost':
        del data['usage']['cost']
    elif failure == 'wrong_provider':
        data['provider'] = 'Alibaba'
    elif failure == 'other_accepted_provider':
        data['provider'] = 'Fireworks'
    elif failure == 'reasoning_returned':
        data['usage']['completion_tokens_details']['reasoning_tokens'] = 1
    else:
        data['usage'].update(completion_tokens=61, total_tokens=81)
    seen = []
    def opened(request, **kwargs):
        seen.append(request)
        return old.Response(data)
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(MinimalCallStopped):
            invoke(selected, client)
        state = selected.snapshot()
        assert state['stopped'] and len(seen) == 1
        if failure == 'reasoning_returned':
            # Identity and bounded token/cost usage remain attributable; this
            # pre-existing global format halt is deliberately not fixed here.
            assert state['known_cost_usd'] == '0.00001'
            assert state['reserved_cost_usd'] == '0'
        else:
            assert Decimal(state['reserved_cost_usd']) > 0
        with pytest.raises(MinimalCallStopped):
            invoke(selected, client, 'another-fixture-op')
        assert len(seen) == 1


@pytest.mark.parametrize('change', ['underpriced_input', 'underpriced_output', 'wrong_provider', 'mixed_models'])
def test_new_policy_rejects_understatement_or_mixed_route(tmp_path, change):
    prices = policy()
    spec = prices['models'][DEEPSEEK_MODEL]
    if change == 'underpriced_input':
        spec['input_usd_per_million'] = '0.01'
    elif change == 'underpriced_output':
        spec['output_usd_per_million'] = '0.01'
    elif change == 'wrong_provider':
        spec['provider'] = 'alibaba'
    else:
        prices['models'][MODELS['online_fast']] = deepcopy(old.prices()['models'][MODELS['online_fast']])
    with pytest.raises(ValueError):
        with closing(guard(tmp_path, prices)) as selected:
            selected.attach(clients().online)


@pytest.mark.parametrize('provider,other', [('together', 'fireworks'), ('fireworks', 'together')])
@pytest.mark.parametrize('role', ['online', 'safety_review', 'confirm_judge'])
def test_accepted_providers_still_require_exact_policy_route_binding(tmp_path, provider, other, role):
    with closing(guard(tmp_path, provider=provider)) as selected:
        with pytest.raises(ValueError, match='reservation_structured_metadata_mismatch'):
            selected.attach(getattr(clients(other), role))
        assert selected.snapshot()['calls_used'] == 0


@pytest.mark.parametrize('provider,name', [('together', 'Fireworks'), ('fireworks', 'Together')])
def test_policy_provider_names_cannot_be_interchanged(tmp_path, provider, name):
    prices = policy(provider)
    prices['models'][DEEPSEEK_MODEL]['response_providers'] = [name]
    with pytest.raises(ValueError, match='deepseek_release_policy_mismatch'):
        guard(tmp_path, prices)


def test_existing_fireworks_ledger_cannot_be_reopened_as_together(tmp_path):
    with closing(guard(tmp_path, provider='fireworks')) as selected:
        selected.attach(clients('fireworks').online)
        original = selected.snapshot()
    with pytest.raises(ValueError, match='reservation_policy_changed'):
        guard(tmp_path, provider='together')
    with closing(guard(tmp_path, provider='fireworks')) as selected:
        assert selected.snapshot() == original


def test_old_ledger_and_clients_cannot_be_rebound_to_new_model(tmp_path):
    with closing(old.guard(tmp_path)) as selected:
        selected.attach(old.make_clients().online)
        original = selected.snapshot()
        assert original['runtime_profile'] == 'release_max_v1'
        with pytest.raises(ValueError):
            selected.attach(clients().online)
    with pytest.raises(ValueError, match='reservation_policy_changed'):
        guard(tmp_path)
    with closing(old.guard(tmp_path)) as selected:
        assert selected.snapshot() == original
    assert BILLABLE_OUTPUTS['confirm_judge'] == 4166
    with closing(guard(tmp_path / 'new')) as selected:
        with pytest.raises(ValueError):
            selected.attach(old.make_clients().online)


@pytest.mark.parametrize('model', [DEEPSEEK_MODEL, CANONICAL])
@pytest.mark.parametrize('provider', ['together', 'fireworks'])
def test_public_capture_recognizes_new_identity_without_retaining_reasoning(model, provider):
    data = response(clients(provider).online)
    data['model'] = model
    data['choices'][0]['message']['reasoning'] = 'synthetic-hidden-field'
    public, observation = projection(json.dumps(data).encode())
    assert observation['identity_states']['model_status'] == 'recognized'
    assert observation['identity_states']['provider_status'] == 'recognized'
    assert 'synthetic-hidden-field' not in json.dumps([public, observation])
