"""Independent offline switch review: synthetic receipts, never model effects."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from aria.release import create_deepseek_release_clients
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.engine.policy.online_structured import DEEPSEEK_MODEL, build_online_structured_route
from aria.engine.policy.release_configuration import derive_policy
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from aria.engine.policy.max_release import tls_dispatch, tls_transport
from aria.engine.policy.minimal_calls import MinimalCallStopped
from tests import test_deepseek_release_guard as legacy


def materials():
    models, online, _ = legacy.metadata()
    return models, online, online


def policy():
    return derive_policy(*materials(), cover_all_price_tiers=True,
        release_profile='release_deepseek_v1', offline_model=DEEPSEEK_MODEL,
        offline_provider='together')[0]


def bundle():
    models, online, offline = materials()
    return create_deepseek_release_clients(model_calls_enabled=True,
        api_key='synthetic-key', base_url='https://openrouter.ai/api/v1',
        online_structured_route=build_online_structured_route(models, online,
            model=DEEPSEEK_MODEL, provider='together'),
        memory_structured_route=build_memory_structured_route(models, offline,
            model=DEEPSEEK_MODEL, provider='together'))


def guard(path, selected_policy=None, budget='0.4'):
    result = MaxReleaseCallGuard(path / 'review.sqlite3', max_requests=20,
        max_observed_cost_usd=budget,
        reservation_policy=policy() if selected_policy is None else selected_policy)
    clock = [1000.0]
    result._pacing_clock = lambda: clock[0]
    result._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    return result


@pytest.fixture(autouse=True)
def forbid_network_and_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('independent tests cannot access network or credentials')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr('aria._env.load_local_env', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


def receive(monkeypatch, data, seen):
    def built(handler, redirect):
        def opened(request, *, timeout):
            payload = json.loads(request.data)
            seen.append((payload, timeout))
            trace = handler._trace
            trace.preparation_recovery.bind_request_headers(request)
            trace.preparation_recovery.before_preparation()
            result = legacy.old.Response(data, payload.get('stream') is True)
            if payload.get('stream') is True:
                identity = {key: data[key] for key in ('id', 'model', 'provider')}
                frames = [{**identity, 'choices': [{'index': 0,
                    'delta': data['choices'][0]['message'], 'finish_reason': None}]},
                    {**identity, 'choices': [{'index': 0, 'delta': {},
                        'finish_reason': data['choices'][0]['finish_reason']}]},
                    {**identity, 'choices': [], 'usage': data['usage']}]
                raw = (''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames)
                    + 'data: [DONE]\n\n').encode()
                result.body = io.BytesIO(raw)
                result.headers.replace_header('Content-Length', str(len(raw)))
            return result
        return SimpleNamespace(open=opened)
    monkeypatch.setattr(urllib.request, 'build_opener', built)
    monkeypatch.setattr(tls_dispatch, 'open_observed', tls_transport.open_candidate)


def response(client):
    result = legacy.response(client)
    if client.role == 'reflection' and client.reflection_return_format == 'json_schema_v4':
        result['choices'][0]['message']['content'] = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
    result['model'] = DEEPSEEK_MODEL
    result['provider'] = 'Together'
    return result


def test_five_roles_share_one_policy_and_ledger_without_output_cap_collision(tmp_path, monkeypatch):
    selected = bundle()
    rates = policy()
    assert set(rates['models']) == {DEEPSEEK_MODEL}
    spec = rates['models'][DEEPSEEK_MODEL]
    assert spec['max_output_tokens'] == 4096
    assert spec['input_usd_per_million'] == '0.3'
    assert spec['output_usd_per_million'] == '1.2'
    seen = []
    with closing(guard(tmp_path, rates)) as g:
        for ordinal, (role, limit) in enumerate((('online', 4096), ('memory', 4000),
                ('reflection', 2800), ('safety_review', 1024), ('confirm_judge', 60))):
            client = getattr(selected, role)
            g.attach(client)
            receive(monkeypatch, response(client), seen)
            legacy.invoke(g, client, f'independent-{ordinal}')
            payload, timeout = seen[-1]
            row = g.snapshot()['attempts'][-1]
            assert payload['model'] == DEEPSEEK_MODEL
            assert payload['provider']['only'] == ['together']
            assert payload['provider']['allow_fallbacks'] is False
            assert payload['reasoning'] == {'enabled': False, 'exclude': True}
            assert payload['max_tokens'] == row['reservation']['output_token_limit'] == limit
            assert timeout == client.timeout == row['effective_physical_timeout_seconds'] == 90
            assert client.max_retries == 0 and client.fallback_models == []
            assert row['status'] == 'received'
            assert row['expected_provider'] == 'Together'
            assert row['pacing']['min_interval_seconds'] == 5
            assert Decimal(row['reservation']['reserved_cost_usd']) == (
                Decimal(row['reservation']['input_token_plan_bound']) * Decimal('0.3')
                + Decimal(limit) * Decimal('1.2')) / 1000000
            if role == 'memory':
                assert payload['response_format']['type'] == 'json_schema'
                assert payload['response_format']['json_schema']['strict'] is True
            if role == 'reflection':
                assert payload['stream'] is True
                assert payload['response_format']['json_schema']['strict'] is True
                assert payload['response_format']['json_schema']['name'] == 'aria_reflection_compact_v4'
                assert row['reflection_provider_contract']['provider_schema_applied'] is True
        state = g.snapshot()
        assert len(seen) == len(state['attempts']) == state['calls_used'] == 5
        assert state['known_cost_usd'] == '0.00023'
        assert state['reserved_cost_usd'] == '0'
        assert state['paid_response_replays'] == 0


@pytest.mark.parametrize('role', ['memory', 'reflection'])
@pytest.mark.parametrize('bad', ['provider', 'model', 'unknown_cost', 'reasoning_tokens', 'truncated'])
def test_offline_failed_receipts_do_not_replay_or_change_provider(tmp_path, monkeypatch, role, bad):
    client = getattr(bundle(), role)
    data = response(client)
    if bad == 'provider':
        data['provider'] = 'Anthropic'
    elif bad == 'model':
        data['model'] = 'anthropic/claude-sonnet-5'
    elif bad == 'unknown_cost':
        data['usage'].pop('cost')
    elif bad == 'reasoning_tokens':
        data['usage']['completion_tokens_details'] = {'reasoning_tokens': 1}
    else:
        data['choices'][0]['finish_reason'] = 'length'
    seen = []
    receive(monkeypatch, data, seen)
    with closing(guard(tmp_path)) as g:
        g.attach(client)
        with pytest.raises((MinimalCallStopped, ValueError)):
            legacy.invoke(g, client)
        state = g.snapshot()
        assert len(seen) == len(state['attempts']) == 1
        assert state['attempts'][0]['status'] == 'failed'
        assert state['paid_response_replays'] == 0
        if bad in {'provider', 'model', 'unknown_cost'}:
            assert Decimal(state['reserved_cost_usd']) > 0
        with pytest.raises((MinimalCallStopped, ValueError)):
            legacy.invoke(g, client, 'must-not-replay')
        assert len(seen) == 1


@pytest.mark.parametrize('role', ['memory', 'reflection'])
@pytest.mark.parametrize('field,value', [('max_completion_tokens', 4001), ('timeout', 240),
    ('max_retries', 1), ('fallback_models', ['anthropic/claude-sonnet-5']),
    ('reasoning_effort', 'high')])
def test_offline_role_drift_rejected_before_any_attempt(tmp_path, role, field, value):
    client = getattr(bundle(), role)
    setattr(client, field, value)
    with closing(guard(tmp_path)) as g:
        with pytest.raises(ValueError):
            g.attach(client)
        assert g.snapshot()['calls_used'] == 0


@pytest.mark.parametrize('field', ['provider', 'metadata_sha256', 'max_output_tokens', 'input_usd_per_million'])
def test_same_model_policy_cannot_undercut_or_change_the_verified_route(tmp_path, field):
    rates = deepcopy(policy())
    rates['models'][DEEPSEEK_MODEL][field] = {
        'provider': 'fireworks', 'metadata_sha256': '0' * 64,
        'max_output_tokens': 4000, 'input_usd_per_million': '0.006'}[field]
    with pytest.raises(ValueError):
        with closing(guard(tmp_path, rates)) as g:
            for role in ('online', 'memory', 'reflection', 'safety_review', 'confirm_judge'):
                g.attach(getattr(bundle(), role))


def test_explicit_offline_selection_rejects_different_metadata_and_old_profile():
    models, online, _ = materials()
    altered = json.dumps(json.loads(online), indent=2).encode()
    with pytest.raises(ValueError):
        derive_policy(models, online, altered, cover_all_price_tiers=True,
            release_profile='release_deepseek_v1', offline_model=DEEPSEEK_MODEL,
            offline_provider='together')
    with pytest.raises(ValueError):
        derive_policy(*legacy.old.metadata(), cover_all_price_tiers=True,
            release_profile='release_max_v1', offline_model=DEEPSEEK_MODEL,
            offline_provider='together')


@pytest.mark.parametrize('role', ['memory', 'reflection'])
def test_hidden_offline_channel_is_rejected_and_never_written(tmp_path, monkeypatch, role):
    client = getattr(bundle(), role)
    data = response(client)
    sentinel = 'independent-hidden-sentinel-never-persist-2313'
    data['choices'][0]['message']['reasoning_content'] = sentinel
    seen = []
    receive(monkeypatch, data, seen)
    with closing(guard(tmp_path)) as g:
        g.attach(client)
        with pytest.raises((MinimalCallStopped, ValueError)):
            legacy.invoke(g, client)
        assert len(seen) == 1
    for path in Path(tmp_path).rglob('*'):
        if path.is_file():
            assert sentinel.encode() not in path.read_bytes()
