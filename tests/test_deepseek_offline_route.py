"""Explicit offline switch through the existing driver; all receipts are synthetic."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import json
import io
from pathlib import Path
import socket
import sqlite3

import pytest

from aria.engine.policy.memory_structured import build_memory_structured_route, memory_request_format
from aria.engine.policy import memory_structured
from aria.engine.memory.memory_contract import _Response
from aria.engine.policy.online_structured import DEEPSEEK_MODEL
from aria.engine.policy.release_configuration import derive_policy, main as policy_main
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from aria.engine.policy.max_release import tls_dispatch
from aria.engine.policy.max_release.reflection_failure import validate_receipt_metadata
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.release import build_release_online_route, create_deepseek_release_clients
from aria.runtime import create_minimal_clients
from scripts import run_story
from tests import test_deepseek_release_guard as previous
from tests import test_installable_max_release as old


def metadata():
    models, endpoint, _ = previous.metadata()
    return models, endpoint


def clients(models=None, endpoint=None):
    saved_models, saved_endpoint = metadata()
    models = saved_models if models is None else models
    endpoint = saved_endpoint if endpoint is None else endpoint
    return create_deepseek_release_clients(model_calls_enabled=True, api_key='inert-software-key',
        online_structured_route=build_release_online_route(models, endpoint, release_profile='release_deepseek_v1'),
        memory_structured_route=build_memory_structured_route(models, endpoint,
            model=DEEPSEEK_MODEL, provider='together'))


def policy(models=None, endpoint=None):
    saved_models, saved_endpoint = metadata()
    models = saved_models if models is None else models
    endpoint = saved_endpoint if endpoint is None else endpoint
    return derive_policy(models, endpoint, endpoint, cover_all_price_tiers=True,
        release_profile='release_deepseek_v1', offline_model=DEEPSEEK_MODEL, offline_provider='together')[0]


def guard(path, *, selected=None, limit='1'):
    result = MaxReleaseCallGuard(path / 'calls.sqlite3', max_requests=10,
        max_observed_cost_usd=limit, reservation_policy=policy() if selected is None else selected)
    clock = [1000.0]
    result._pacing_clock = lambda: clock[0]
    result._pacing_sleep = lambda delta: clock.__setitem__(0, clock[0] + delta)
    return result


def response(client):
    result = previous.response(client)
    if client.role == 'reflection' and client.reflection_return_format == 'json_schema_v4':
        result['choices'][0]['message']['content'] = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
    result.update(model=DEEPSEEK_MODEL, provider='Together')
    result['usage']['cost'] = 0.00001
    return result


@pytest.fixture(autouse=True)
def no_network_credentials(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('network and credentials are forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria._env.load_local_env', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


def dispatch(monkeypatch, data, *, reflection=False):
    seen = []
    def opened(request, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        seen.append((json.loads(request.data), timeout))
        return old.Response(data, json.loads(request.data).get('stream') is True)
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    return seen


def test_five_roles_share_one_existing_guard_and_exact_route(tmp_path, monkeypatch):
    bundle = clients()
    assert bundle.release_profile == 'release_deepseek_v1'
    with closing(guard(tmp_path)) as selected:
        for name, limit in [('online', 4096), ('memory', 4000), ('reflection', 2800),
                            ('safety_review', 1024), ('confirm_judge', 60)]:
            client = getattr(bundle, name)
            selected.attach(client)
            seen = dispatch(monkeypatch, response(client), reflection=name == 'reflection')
            answer = previous.invoke(selected, client, operation='fixture-' + name)
            assert answer
            wire, timeout = seen[0]
            assert len(seen) == 1 and timeout == client.timeout == 90
            assert wire['model'] == DEEPSEEK_MODEL and wire['max_tokens'] == limit
            assert wire['provider']['only'] == ['together']
            assert wire['provider']['allow_fallbacks'] is False
            assert wire['provider']['require_parameters'] is True
            assert wire['provider']['data_collection'] == 'deny'
            assert wire['reasoning'] == {'enabled': False, 'exclude': True}
            assert client.max_retries == 0 and client.fallback_models == []
            if name == 'memory':
                assert wire['response_format'] == memory_request_format(client)
                assert wire['response_format']['json_schema']['strict'] is True
                assert wire['stream'] is True
            elif name == 'reflection':
                assert wire['stream'] is True and wire['response_format']['json_schema']['strict'] is True
                assert wire['response_format']['json_schema']['name'] == 'aria_reflection_compact_v4'
                assert client.reflection_itemwise_review is True
            state = selected.snapshot()
            row = state['attempts'][-1]
            assert row['status'] == 'received' and not state['stopped']
            assert row['reservation']['output_token_limit'] == limit
            assert row['expected_provider'] == row['returned_provider'] == 'Together'
            assert row['pacing']['min_interval_seconds'] == 5
            if state['calls_used'] == 1:
                assert row['pacing']['wait_ms'] == 0
            else:
                assert row['pacing']['start_gap_ms'] >= 5000
            assert row['effective_physical_timeout_seconds'] == 90
            assert row['receipt_identity']['model_status'] == row['receipt_identity']['provider_status'] == 'recognized'
            captured = json.loads(Path(row['public_first_output']['path']).read_bytes())
            assert captured['hidden_reasoning_retained'] is False
        state = selected.snapshot()
        assert state['calls_used'] == 5 and len(state['attempts']) == 5
        assert Decimal(state['known_cost_usd']) == Decimal('.00005')
        assert state['reserved_cost_usd'] == '0'


@pytest.mark.parametrize('interval', [5, 120])
def test_deepseek_reopen_keeps_interval_and_unknown_fee_stop(tmp_path, monkeypatch, interval):
    with closing(guard(tmp_path)):
        pass
    path = tmp_path / 'calls.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        # An empty allowance registered by either generation of this route.
        db.execute('UPDATE request_pacing SET min_interval_seconds=? WHERE id=1', (interval,))
    client = clients().memory
    with closing(guard(tmp_path)) as first:
        first.attach(client)
        dispatch(monkeypatch, response(client))
        previous.invoke(first, client, operation='first')
        prior = first.snapshot()['attempts'][0]
        assert prior['pacing']['min_interval_seconds'] == interval
    with closing(guard(tmp_path)) as second:
        client = clients().memory
        second.attach(client)
        bad = response(client)
        bad['provider'] = 'unrecognized'
        seen = dispatch(monkeypatch, bad)
        with pytest.raises((MinimalCallStopped, ValueError)):
            previous.invoke(second, client, operation='second')
        state = second.snapshot()
        assert len(seen) == 1 and state['calls_used'] == 2 and state['attempts'][0] == prior
        assert state['attempts'][1]['pacing']['wait_ms'] >= interval * 1000
        assert state['stopped'] and Decimal(state['reserved_cost_usd']) > 0
    with closing(guard(tmp_path)) as stopped:
        restored = stopped.snapshot()
        assert restored['stopped'] and restored['reserved_cost_usd'] == state['reserved_cost_usd']
        assert restored['attempts'] == state['attempts']
    if interval == 120:
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('UPDATE request_pacing SET min_interval_seconds=5 WHERE id=1')
        with pytest.raises(ValueError, match='retained_request_pacing_changed'):
            guard(tmp_path)


def test_same_model_policy_merges_caps_and_preserves_metadata_and_rates():
    models, endpoint = metadata()
    data, proof = derive_policy(models, endpoint, endpoint, cover_all_price_tiers=True,
        release_profile='release_deepseek_v1', offline_model=DEEPSEEK_MODEL, offline_provider='together')
    assert set(data['models']) == {DEEPSEEK_MODEL}
    spec = data['models'][DEEPSEEK_MODEL]
    assert spec['max_output_tokens'] == 4096
    assert spec['input_usd_per_million'] == '0.3' and spec['output_usd_per_million'] == '1.2'
    assert spec['provider'] == 'together' and spec['response_providers'] == ['Together']
    assert {item['reserved_output_token_limit'] for item in proof} == {4000, 4096}
    assert all(item['discount_applied'] is False for item in proof)


def test_current_allowance_requires_owner_source_contract_before_first_binding(tmp_path):
    from aria.engine.policy.max_release.sse_adapter import enable_structured_sse
    from aria.engine.risk.safety_review_v2 import safety_review_messages
    with closing(guard(tmp_path)) as current:
        weak = clients().safety_review
        weak.structured_stream_protocol = None
        memory = clients().memory
        memory.memory_fact_source_protocol = None
        for wrong in (weak, memory):
            with pytest.raises(ValueError, match='release_owner_source_contract_mismatch'):
                current.attach(wrong)
        review = clients().safety_review
        current.attach(review)
        evidence = {'concern': 'none', 'subject': None, 'time': None, 'urgency': None, 'quote': None,
                    'verbatim_source': False, 'disposition': None, 'semantic_verified': False,
                    'conversation_history': []}
        old_messages = safety_review_messages('本人原话。', '候选正文。', evidence, None, final_protection='')
        with current.scope('turn', 'wrong-contract', 'session'):
            with pytest.raises(ValueError, match='invalid_owner_source_review_route'):
                review.complete(old_messages)
        assert current.snapshot()['calls_used'] == 0
    mixed_path = tmp_path / 'mixed'
    mixed_path.mkdir()
    with closing(guard(mixed_path, selected=previous.policy())) as mixed:
        with pytest.raises(ValueError, match='release_owner_source_contract_mismatch'):
            mixed.attach(enable_structured_sse(previous.clients().safety_review))
        assert mixed.snapshot()['calls_used'] == 0


def test_current_reflection_wire_keeps_preferences_out_of_positive_person_inference(tmp_path, monkeypatch):
    """Actual fake HTTP wire, not a model understanding score."""
    from aria.engine.policy.max_release.sse_adapter import (
        model_visible_contract_binding, model_visible_schema_text,
    )
    client = clients().reflection
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        seen = dispatch(monkeypatch, response(client), reflection=True)
        previous.invoke(selected, client, operation='fixture-current-t4-prompt')
        wire, = [payload for payload, _ in seen]
        prompt = wire['messages'][0]['content']
        assert '本人明确在这些条件下表达偏好，可为 within_scope support 并形成' not in prompt
        assert '首次行为证据可提出 repeated_pattern 方向的暂定候选' not in prompt
        assert 'response用于用户明确的沟通偏好或实际互动反应' not in prompt
        assert '交流要求由 memory' in prompt and '本人确认' in prompt
        assert '本会话真实交付后的 observed_response' in prompt
        assert '原独立事件数量与状态阈值保持' in prompt
        assert '有据反证或明确纠正' in prompt
        assert 'reviews 为数组，可为空，最多 6 项' in prompt
        assert model_visible_schema_text() not in prompt
        assert '字段与类型 Schema（' not in prompt
        # The actual current wire no longer asks the model to translate a
        # historical null-valued prompt or choose among internal version names.
        assert 'prompt contract' not in prompt and 'v3' not in prompt and '2.5' not in prompt
        assert 'protocol_version 固定为字符串 "2.0"' in prompt
        assert len(prompt.encode()) < 7500  # Schema is supplied once through response_format.
        assert 'atomic_claim 只含一个 aspect、conditions 和 tendency' not in prompt
        assert 'atomic_claim 恰有五个字段：aspect、conditions、tendency、outcome、evidence_basis' in prompt
        assert '不要为填满数组写占位 review' in prompt
        # Prompt cleanup cannot silently widen the source/shape contract.
        assert sha256(model_visible_schema_text().encode()).hexdigest() == (
            '481d5996f6472b443fd0e21f17817557583d5cbe6acd0340cd1000d79a2d20cc')
        assert wire['response_format']['json_schema']['strict'] is True and wire['stream'] is True
        schema = wire['response_format']['json_schema']['schema']
        assert 'id' not in schema['$defs']['_Review']['properties']
        assert 'review_id' not in schema['$defs']['_Change']['properties']
        assert 'review_index' in schema['$defs']['_Change']['properties']
        assert wire['max_tokens'] == 2800
        row, = selected.snapshot()['attempts']
        assert row['status'] == 'received'
        assert row['reflection_provider_contract']['model_visible_system_sha256'] == sha256(prompt.encode()).hexdigest()
        assert row['reflection_provider_contract']['model_visible_system_sha256'] == model_visible_contract_binding(client)['model_visible_system_sha256']


def test_current_reflection_example_has_all_original_required_fields_and_decodes():
    from aria.engine.policy.reflection_compact import (
        indexed_compact_reflection_system_prompt, decode_indexed_compact_reflection)
    from aria.engine.memory.reflection_contract import validate_reflection_response_shape

    marker = '格式示例（仅展示一个完整观察的形状，不是本次证据）：\n'
    example, _ = json.JSONDecoder().raw_decode(indexed_compact_reflection_system_prompt().split(marker, 1)[1])
    canonical = json.loads(decode_indexed_compact_reflection(json.dumps(example, ensure_ascii=False)))
    validate_reflection_response_shape(canonical)
    assert set(canonical) == {'protocol_version', 'reviews', 'changes'}
    assert canonical['protocol_version'] == '2.0' and canonical['changes'] == []
    assert len(canonical['reviews']) == 1
    review = canonical['reviews'][0]
    assert review['hypothesis_id'] is review['claim_quote'] is review['scope_quote'] is None
    assert review['evidence'] == example['reviews'][0]['evidence']
    assert review['event_evidence'] == example['reviews'][0]['event_evidence']


@pytest.mark.parametrize('duplicate_key', ['reason', 'event_summary'])
def test_current_reflection_duplicate_key_remains_failed_first_output(
        tmp_path, monkeypatch, duplicate_key):
    """Exercise the actual current factory/receipt path, with synthetic content."""
    from aria.engine.policy.reflection_compact import indexed_compact_reflection_system_prompt
    from aria.engine.policy.minimal_calls import MinimalCallStopped

    marker = '格式示例（仅展示一个完整观察的形状，不是本次证据）：\n'
    example, _ = json.JSONDecoder().raw_decode(indexed_compact_reflection_system_prompt().split(marker, 1)[1])
    first_review = example['reviews'][0]
    # Even equal duplicate values are ambiguous input, not a successful repair.
    review_json = json.dumps(first_review, ensure_ascii=False)
    duplicate = json.dumps({duplicate_key: first_review[duplicate_key]}, ensure_ascii=False)
    raw = '{"protocol_version":"2.0","reviews":[' + review_json[:-1] + ',' + duplicate[1:] + '],"changes":[]}'
    client = clients().reflection
    data = response(client)
    data['choices'][0]['message']['content'] = raw
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        seen = dispatch(monkeypatch, data, reflection=True)
        with pytest.raises(MinimalCallStopped):
            previous.invoke(selected, client, operation='fixture-duplicate-reflection-key')
        state = selected.snapshot()
        row, = state['attempts']
        assert len(seen) == 1
        assert row['status'] == 'failed' and row['validation_error'] == 'reflection_schema_invalid'
        assert row['reflection_wrapper']['status'] == 'rejected'
        assert state['known_cost_usd'] == '0.00001' and state['reserved_cost_usd'] == '0'
        assert state['unknown_cost_requests'] == 0
        captured = json.loads(Path(row['public_first_output']['path']).read_bytes())
        assert captured['public_first_output'][0]['content'] == raw
        assert captured['hidden_reasoning_retained'] is False


@pytest.mark.parametrize('version,expected', [
    ('domain', '0bf559aa051ddf3fd3d54493a72a7a703fc6b96da19552423aa4ad4fa2519dc6'),
    ('body', 'c28865e7a58b4acb55502bdda5ae855596ff9e20b45c0d911aa35383950ee7e4'),
    ('compact', 'd839747221689ba658e2ae0a518989b951486ec6e0600a34ae902757261a7edc'),
])
def test_current_t4_prompt_does_not_rewrite_historical_explicit_prompts(version, expected):
    from aria.engine.memory.diagnosis import reflection_system_prompt
    from aria.engine.policy.reflection_json import reflection_body_system_prompt
    from aria.engine.policy.reflection_compact import compact_reflection_system_prompt
    prompts = {'domain': lambda: reflection_system_prompt(True),
               'body': reflection_body_system_prompt, 'compact': compact_reflection_system_prompt}
    assert sha256(prompts[version]().encode()).hexdigest() == expected


@pytest.mark.parametrize('fault', ['metadata_bytes', 'price', 'canonical', 'provider'])
def test_shared_model_metadata_cannot_be_mixed(fault):
    models, endpoint = metadata()
    altered = json.loads(endpoint)
    value = altered['data']['endpoints'][0]
    altered_models = models
    if fault == 'metadata_bytes':
        altered['checked_at'] = 'different-capture'
    elif fault == 'price':
        value['pricing']['prompt'] = '.0000004'
    elif fault == 'provider':
        value.update(tag='fireworks', provider_name='Fireworks')
    else:
        model_value = json.loads(models)
        model_value['data'][0]['canonical_slug'] += '-changed'
        altered_models = json.dumps(model_value).encode()
    raw = json.dumps(altered).encode()
    with pytest.raises(ValueError):
        create_deepseek_release_clients(model_calls_enabled=True, api_key='inert-software-key',
            online_structured_route=build_release_online_route(models, endpoint, release_profile='release_deepseek_v1'),
            memory_structured_route=build_memory_structured_route(altered_models, raw,
                model=DEEPSEEK_MODEL, provider='fireworks' if fault == 'provider' else 'together'))
    if fault != 'canonical':
        with pytest.raises(ValueError):
            derive_policy(models, endpoint, raw, cover_all_price_tiers=True,
                release_profile='release_deepseek_v1', offline_model=DEEPSEEK_MODEL,
                offline_provider='fireworks' if fault == 'provider' else 'together')


@pytest.mark.parametrize('missing', ['structured_outputs', 'response_format', 'max_tokens', 'reasoning'])
def test_offline_route_requires_every_advertised_capability(missing):
    models, raw = metadata()
    value = json.loads(raw)
    value['data']['endpoints'][0]['supported_parameters'].remove(missing)
    with pytest.raises(ValueError):
        build_memory_structured_route(models, json.dumps(value).encode(), model=DEEPSEEK_MODEL, provider='together')


@pytest.mark.parametrize('mutation', ['mandatory_reasoning', 'provider_name', 'inactive', 'output_cap'])
def test_offline_route_rejects_incompatible_capabilities(mutation):
    models, raw = map(json.loads, metadata())
    endpoint = raw['data']['endpoints'][0]
    if mutation == 'mandatory_reasoning': models['data'][0]['reasoning']['mandatory'] = True
    elif mutation == 'provider_name': endpoint['provider_name'] = 'Anthropic'
    elif mutation == 'inactive': endpoint['status'] = 1
    else: endpoint['max_completion_tokens'] = 3999
    with pytest.raises(ValueError):
        build_memory_structured_route(json.dumps(models).encode(), json.dumps(raw).encode(),
            model=DEEPSEEK_MODEL, provider='together')


@pytest.mark.parametrize('role,fault', [(role, fault) for role in ('memory', 'reflection')
    for fault in ('model', 'provider', 'tokens')] + [('memory', 'reasoning')])
def test_wrong_receipt_is_rejected_without_retry_or_releasing_hold(tmp_path, monkeypatch, role, fault):
    client = getattr(clients(), role)
    data = response(client)
    if fault == 'model': data['model'] = 'anthropic/claude-sonnet-5'
    elif fault == 'provider': data['provider'] = 'Anthropic'
    elif fault == 'tokens': data['usage']['completion_tokens'] = 5000
    else: data['choices'][0]['message']['reasoning'] = 'synthetic-hidden-text'
    seen = dispatch(monkeypatch, data, reflection=role == 'reflection')
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(Exception): previous.invoke(selected, client)
        state = selected.snapshot()
        assert len(seen) == state['calls_used'] == 1
        assert state['attempts'][0]['status'] == 'failed' and state['stopped']
        assert Decimal(state['reserved_cost_usd']) > 0
        if fault == 'reasoning':
            # The shared SSE reader rejects this forbidden first frame before
            # terminal usage arrives; no finite-envelope fee can be inherited.
            row = state['attempts'][0]
            assert row['stream_read']['failure_code'] == 'hidden_channel'
            assert state['known_cost_usd'] is None and 'public_first_output' not in row
            assert all('synthetic-hidden-text' not in path.read_text(encoding='utf-8')
                       for path in selected.physical_dir.rglob('*.json'))


def test_memory_missing_schema_field_is_not_repaired(tmp_path, monkeypatch):
    client = clients().memory
    data = response(client)
    value = json.loads(data['choices'][0]['message']['content'])
    del value[next(iter(value))]
    data['choices'][0]['message']['content'] = json.dumps(value)
    seen = dispatch(monkeypatch, data)
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(Exception): previous.invoke(selected, client)
        assert len(seen) == 1 and selected.snapshot()['attempts'][0]['status'] == 'failed'


@pytest.mark.parametrize('fault', ['done', 'usage', 'reasoning'])
def test_reflection_stream_requires_terminal_receipt_and_excludes_reasoning(tmp_path, monkeypatch, fault):
    client = clients().reflection
    data = response(client)
    raw = old.stream(data)
    if fault == 'done': raw = raw.replace(b'data: [DONE]\n\n', b'')
    elif fault == 'usage':
        raw = b'\n\n'.join(frame for frame in raw.split(b'\n\n') if b'"usage"' not in frame)
    else:
        raw = raw.replace(b'"delta": {"content":', b'"delta": {"reasoning": "synthetic-hidden-text", "content":', 1)
    seen = []
    def opened(request, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        seen.append(json.loads(request.data))
        result = old.Response(data, True)
        result.body = io.BytesIO(raw)
        result.headers.replace_header('Content-Length', str(len(raw)))
        return result
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(Exception): previous.invoke(selected, client)
        state = selected.snapshot()
        assert len(seen) == state['calls_used'] == 1
        assert state['attempts'][0]['status'] == 'failed' and state['stopped']
        assert Decimal(state['reserved_cost_usd']) > 0
    assert all('synthetic-hidden-text' not in path.read_text(encoding='utf-8')
               for path in tmp_path.rglob('*.json'))


@pytest.mark.parametrize('mutation', ['output_cap', 'provider', 'metadata_sha', 'underpriced'])
def test_five_role_binding_rejects_changed_policy_before_dispatch(tmp_path, mutation):
    value = policy()
    spec = value['models'][DEEPSEEK_MODEL]
    if mutation == 'output_cap': spec['max_output_tokens'] = 4000
    elif mutation == 'provider': spec.update(provider='fireworks', response_providers=['Fireworks'])
    elif mutation == 'metadata_sha': spec['metadata_sha256'] = '0' * 64
    else: spec['input_usd_per_million'] = '0.001'
    with pytest.raises(ValueError), closing(guard(tmp_path, selected=value)) as selected:
        selected.attach(clients().online)


def test_old_sonnet_ledger_and_incomplete_settlement_are_not_inherited(tmp_path):
    old_policy = previous.policy()
    with closing(guard(tmp_path, selected=old_policy)) as selected:
        selected.attach(previous.clients().memory)
        old_identity = selected.ledger_id
    with pytest.raises(ValueError, match='retained_reservation_policy_changed'):
        guard(tmp_path)
    with closing(guard(tmp_path, selected=old_policy)) as selected:
        assert selected.ledger_id == old_identity and selected.snapshot()['calls_used'] == 0
    with pytest.raises(ValueError, match='exact_settled_reflection_rejection_required'):
        validate_receipt_metadata({'role': 'reflection', 'kind': 'review', 'status': 'failed',
            'model': DEEPSEEK_MODEL, 'returned_model': DEEPSEEK_MODEL})


@pytest.mark.parametrize('fault',['version','extra'])
def test_known_reflection_format_failure_without_app_transaction_still_stops(tmp_path,monkeypatch,fault):
    client=clients().reflection;data=response(client)
    body={'protocol_version':'2.0','reviews':[],'changes':[]}
    if fault=='version':body['protocol_version']='v3'
    else:body['contract']='prompt_contract 2.5'
    data['choices'][0]['message']['content']=json.dumps(body)
    seen=dispatch(monkeypatch,data,reflection=True)
    with closing(guard(tmp_path)) as selected:
        selected.attach(client)
        with pytest.raises(Exception):previous.invoke(selected,client)
        state=selected.snapshot();row=state['attempts'][-1]
        assert len(seen)==state['calls_used']==1 and state['stopped']
        assert row['status']=='failed' and row['validation_error']=='reflection_schema_invalid'
        assert Decimal(state['known_cost_usd'])==Decimal('.00001') and state['reserved_cost_usd']=='0'
        with pytest.raises((KeyError,ValueError)):validate_receipt_metadata(row)
        with pytest.raises(ValueError,match='exact_settled_reflection_rejection_required'):
            validate_receipt_metadata(row,historical=True)


def test_nondefault_reflection_cannot_bypass_verified_route():
    with pytest.raises(ValueError, match='structured reflection requires'):
        create_minimal_clients(model_calls_enabled=True, api_key='inert-software-key',
            reflection_model=DEEPSEEK_MODEL, reflection_return_format='json_schema_v3')


def test_original_policy_and_story_clis_accept_same_explicit_selection(tmp_path, monkeypatch):
    models, endpoint = metadata()
    source = tmp_path / 'models.json'; source.write_bytes(models)
    route = tmp_path / 'route.json'; route.write_bytes(endpoint)
    prices = tmp_path / 'policy.json'
    selection = ['--offline-model', DEEPSEEK_MODEL, '--offline-provider', 'together']
    assert policy_main(['--models-json', str(source), '--online-endpoints-json', str(route),
        '--offline-endpoints-json', str(route), '--output', str(prices), '--cover-all-price-tiers',
        '--release-profile', 'release_deepseek_v1', *selection]) == 0
    args = run_story.parser().parse_args(['--story', 'unused.json', '--out', str(tmp_path / 'run'),
        '--enable-model-calls', '--model-max-requests', '10', '--model-stop-cost-usd', '0.4',
        '--model-reservation-policy', str(prices), '--online-models-metadata', str(source),
        '--online-endpoints-metadata', str(route), '--offline-endpoints-metadata', str(route), *selection])
    monkeypatch.setattr('aria.runtime.openrouter_api_key', lambda: 'inert-software-key')
    factory, close = run_story.configured_factory(args)
    try:
        app = factory(tmp_path / 'profile')
        assert app.clients.memory.model == app.clients.reflection.model == DEEPSEEK_MODEL
        assert app.clients.release_profile == 'release_deepseek_v1'
        assert app.call_guard.min_interval_seconds == 5
        assert app.call_guard.snapshot()['calls_used'] == 0
        app.close()
    finally:
        close()
    assert run_story.parser().parse_args(['--story', 'unused', '--out', 'unused']).offline_model == 'anthropic/claude-sonnet-5'


def generation_schema_before_capacity_projection(*, source_protocol="preference_quotes_v1"):
    """The prior projection over unchanged receiving models, for wire comparison."""
    from aria.engine.memory.memory_contract import FACT_SOURCE_REFS, _ResponseFactRefs, _ResponseRefs
    from aria.engine.memory.preference_sources import LEGACY, SOURCE_REFS
    schema = {LEGACY: _Response, SOURCE_REFS: _ResponseRefs,
              FACT_SOURCE_REFS: _ResponseFactRefs}[source_protocol].model_json_schema()
    if source_protocol == LEGACY:
        preference = schema['$defs']['_Preference']
        preference['properties']['revision_basis'].pop('default', None)
        preference['required'].append('revision_basis')
    return schema


def capture_five_role_requests(path, monkeypatch, offline):
    bundle = clients() if offline == 'deepseek' else previous.clients()
    prices = policy() if offline == 'deepseek' else previous.policy()
    requests, bindings = {}, {}
    with closing(guard(path, selected=prices)) as selected:
        for name in ('online', 'memory', 'reflection', 'safety_review', 'confirm_judge'):
            client = getattr(bundle, name)
            selected.attach(client)
            bindings[client.role] = json.loads(selected._binding(client))
            data = response(client) if offline == 'deepseek' else previous.response(client)
            def capture(request, *, timeout, trace):
                trace.preparation_recovery.bind_request_headers(request)
                trace.preparation_recovery.before_preparation()
                requests[client.role] = request.data
                assert timeout == 90
                return old.Response(data, json.loads(request.data).get('stream') is True)
            monkeypatch.setattr(tls_dispatch, 'open_observed', capture)
            previous.invoke(selected, client, operation='schema-wire-' + name)
        assert selected.snapshot()['calls_used'] == 5 and not selected.snapshot()['stopped']
    return requests, bindings, prices


@pytest.mark.parametrize('offline', ['deepseek', 'sonnet'])
def test_generation_projection_changes_only_strict_memory_schema_and_binding(tmp_path, monkeypatch, offline):
    with monkeypatch.context() as historical:
        historical.setattr(memory_structured, 'memory_response_schema', generation_schema_before_capacity_projection)
        before, old_bindings, old_prices = capture_five_role_requests(tmp_path / 'before', historical, offline)
    after, new_bindings, new_prices = capture_five_role_requests(tmp_path / 'after', monkeypatch, offline)
    assert old_prices == new_prices
    assert set(before) == set(after) == {'online_fast', 'memory', 'reflection', 'safety_review', 'confirm_judge'}
    for role in before:
        if role != 'memory':
            assert before[role] == after[role] and old_bindings[role] == new_bindings[role]
            continue
        old_wire, new_wire = json.loads(before[role]), json.loads(after[role])
        old_schema = old_wire['response_format']['json_schema'].pop('schema')
        new_schema = new_wire['response_format']['json_schema'].pop('schema')
        assert old_wire == new_wire  # Includes original model, provider, messages and role cap.
        assert old_schema != new_schema and new_wire['response_format']['json_schema']['strict'] is True
        for definition, field, keys in [
            ('_PreferenceRefs', 'do', ('minLength', 'maxLength')),
            ('_PreferenceRefs', 'source_turn_indices', ('minItems', 'maxItems')),
        ]:
            for key in keys: new_schema['$defs'][definition]['properties'][field].pop(key)
        properties = new_schema['$defs']['_PreferenceRefs']['properties']
        properties['source_turn_indices']['items'].pop('minimum')
        properties['revision_turn_index']['anyOf'][0].pop('minimum')
        # Current generation also requires the existing observation signal;
        # reception keeps missing/null compatible as non-counting context.
        observation = new_schema['$defs']['_Observation']
        assert observation['properties']['signal'] == {'type': 'string', 'enum': ['explicit_change', 'context']}
        assert 'signal' in observation['required']
        observation['properties']['signal'] = old_schema['$defs']['_Observation']['properties']['signal']
        observation['required'].remove('signal')
        assert new_schema == old_schema
        old_hash = old_bindings[role]['memory_structured_route'].pop('memory_schema_sha256')
        new_hash = new_bindings[role]['memory_structured_route'].pop('memory_schema_sha256')
        assert old_hash != new_hash and old_bindings[role] == new_bindings[role]
    reflection = json.loads(after['reflection'])
    assert reflection['stream'] is True
    assert ('response_format' in reflection) is (offline == 'deepseek')


@pytest.mark.parametrize('offline', ['deepseek', 'sonnet'])
def test_new_memory_schema_cannot_reuse_old_bound_allowance(tmp_path, monkeypatch, offline):
    factory = clients if offline == 'deepseek' else previous.clients
    prices = policy() if offline == 'deepseek' else previous.policy()
    with monkeypatch.context() as historical:
        historical.setattr(memory_structured, 'memory_response_schema', generation_schema_before_capacity_projection)
        with closing(guard(tmp_path, selected=prices)) as selected:
            selected.attach(factory().memory)
            identity = selected.ledger_id
            retained = selected._db.execute('SELECT role,binding FROM roles ORDER BY role').fetchall()
            assert selected.snapshot()['calls_used'] == 0
    with closing(guard(tmp_path, selected=prices)) as selected:
        with pytest.raises(ValueError, match='model configuration differs from retained allowance'):
            selected.attach(factory().memory)
        assert selected.ledger_id == identity and selected.snapshot()['calls_used'] == 0
        assert selected._db.execute('SELECT role,binding FROM roles ORDER BY role').fetchall() == retained
    # A historical software fixture still reattaches with its exact old schema;
    # no retained record is rewritten to impersonate the new contract.
    with monkeypatch.context() as historical:
        historical.setattr(memory_structured, 'memory_response_schema', generation_schema_before_capacity_projection)
        with closing(guard(tmp_path, selected=prices)) as selected:
            selected.attach(factory().memory)
            assert selected.ledger_id == identity and selected.snapshot()['calls_used'] == 0
