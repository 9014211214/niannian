"""Offline assembly checks; synthetic replies do not establish model effects."""
import json
import socket
from contextlib import closing

import pytest

from aria.release import build_release_online_route, create_deepseek_release_clients, create_max_release_clients
from aria.engine.policy.release_configuration import derive_policy
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.engine.policy.online_structured import DEEPSEEK_MODEL, DEEPSEEK_PROVIDER
from tests.test_installable_max_release import metadata as max_metadata, make_clients as max_clients, prices as max_prices, messages
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from scripts import run_story


def metadata():
    models, online, offline = (json.loads(raw) for raw in max_metadata())
    row = models['data'][0]
    row.update(id=DEEPSEEK_MODEL, canonical_slug='deepseek/deepseek-v4.1-flash-20261017',
               reasoning={'mandatory': False, 'supported_efforts': ['none', 'low', 'high']})
    online['data']['id'] = DEEPSEEK_MODEL
    endpoint = online['data']['endpoints'][0]
    endpoint.update(model_id=DEEPSEEK_MODEL, tag=DEEPSEEK_PROVIDER,
                    provider_name='Together', name='Together | DeepSeek 1017')
    endpoint['pricing'] = {'prompt': '0.0000003', 'completion': '0.0000012', 'input_cache_read': '0.000000006'}
    return tuple(json.dumps(value, ensure_ascii=False).encode() for value in (models, online, offline))


def prices():
    return derive_policy(*metadata(), cover_all_price_tiers=True, release_profile='release_deepseek_v1')[0]


def clients():
    models, online, offline = metadata()
    return create_deepseek_release_clients(model_calls_enabled=True, api_key='inert-test-key',
        online_structured_route=build_release_online_route(models, online, release_profile='release_deepseek_v1'),
        memory_structured_route=build_memory_structured_route(models, offline))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network and real credentials forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria._env.load_local_env', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)


def test_new_profile_all_online_roles_and_sonnet_protocols():
    selected = clients()
    assert DEEPSEEK_PROVIDER == 'together'
    assert selected.release_profile == 'release_deepseek_v1'
    for role, limit in [('online', 4096), ('safety_review', 1024), ('confirm_judge', 60)]:
        client = getattr(selected, role)
        assert client.model == DEEPSEEK_MODEL
        assert client.reasoning_effort == 'none'
        assert client.max_completion_tokens == limit
        assert client.max_retries == 0 and client.fallback_models == []
        assert client.network_request_count == 0
        route = getattr(client, {'online': 'online_structured_route',
            'safety_review': 'safety_structured_route', 'confirm_judge': 'confirm_structured_route'}[role])
        assert route.provider == 'together' and route.binding()['provider_name'] == 'Together'
    assert selected.online.online_decision_version == 'online_turn_decision_v7'
    assert selected.memory.model == selected.reflection.model == 'anthropic/claude-sonnet-5'
    assert selected.reflection.reflection_return_format == 'json_schema_v3'
    assert selected.memory.max_completion_tokens == 4000
    assert selected.reflection.max_completion_tokens == 2800


def test_price_policy_is_bound_to_new_model_without_reasoning_reservation():
    policy, proof = derive_policy(*metadata(), cover_all_price_tiers=True, release_profile='release_deepseek_v1')
    assert set(policy['models']) == {DEEPSEEK_MODEL, 'anthropic/claude-sonnet-5'}
    online = policy['models'][DEEPSEEK_MODEL]
    assert online['provider'] == 'together'
    assert online['input_usd_per_million'] == '0.3'
    assert online['output_usd_per_million'] == '1.2'
    assert online['max_output_tokens'] == 4096
    assert online['response_providers'] == ['Together']
    assert all(row['release_profile'] == 'release_deepseek_v1' for row in proof)


@pytest.mark.parametrize('missing', ['structured_outputs', 'reasoning', 'max_tokens'])
def test_missing_advertised_capability_does_not_inherit_max_support(missing):
    models, online, _ = metadata()
    endpoint = json.loads(online)
    endpoint['data']['endpoints'][0]['supported_parameters'].remove(missing)
    with pytest.raises(ValueError):
        build_release_online_route(models, json.dumps(endpoint).encode(), release_profile='release_deepseek_v1')


def test_old_metadata_cannot_silently_select_old_model_for_new_default():
    models, online, _ = max_metadata()
    with pytest.raises(ValueError):
        build_release_online_route(models, online, release_profile='release_deepseek_v1')
    assert build_release_online_route(models, online, release_profile='release_max_v1').model.startswith('qwen/')


def test_existing_driver_assembles_new_profile_without_sending(tmp_path, monkeypatch):
    paths = []
    for name, raw in zip(('models', 'online', 'offline'), metadata()):
        path = tmp_path / (name + '.json'); path.write_bytes(raw); paths.append(path)
    policy = tmp_path / 'policy.json'; policy.write_text(json.dumps(prices()))
    args = run_story.parser().parse_args(['--story', 'unused.json', '--out', str(tmp_path / 'new-run'),
        '--enable-model-calls', '--model-max-requests', '10', '--model-stop-cost-usd', '0.4',
        '--model-reservation-policy', str(policy), '--online-models-metadata', str(paths[0]),
        '--online-endpoints-metadata', str(paths[1]), '--offline-endpoints-metadata', str(paths[2])])
    monkeypatch.setattr('aria.runtime.openrouter_api_key', lambda: 'inert-test-key')
    factory, close = run_story.configured_factory(args)
    try:
        app = factory(tmp_path / 'profile')
        assert app.clients.release_profile == 'release_deepseek_v1'
        assert app.clients.confirm_judge.model == DEEPSEEK_MODEL
        assert app.call_guard.snapshot()['calls_used'] == 0
        assert app.call_guard.snapshot()['runtime_profile'] == 'release_deepseek_v1'
        app.close()
    finally:
        close()


def test_preview_is_inert():
    assert create_deepseek_release_clients().model_calls_enabled is False


@pytest.mark.parametrize('role', ['memory', 'reflection'])
@pytest.mark.parametrize('source_protocol', ['preference_quotes_v1', 'preference_source_refs_v1'])
def test_sonnet_wire_changes_only_the_explicit_memory_contract(tmp_path, monkeypatch, role, source_protocol):
    class Captured(BaseException):
        pass
    actual = []
    for index, (bundle, policy) in enumerate(((max_clients(), max_prices()), (clients(), prices()))):
        client = getattr(bundle, role)
        if role == 'memory' and index == 1:
            client.preference_source_protocol = source_protocol
        with closing(MaxReleaseCallGuard(tmp_path / str(index) / 'ledger.sqlite3',
                max_requests=1, max_observed_cost_usd='1', reservation_policy=policy)) as guard:
            guard.attach(client)
            def capture(c, request, **options):
                actual.append(request.data)
                raise Captured()
            monkeypatch.setattr(guard, '_request_once', capture)
            with pytest.raises(Captured), guard.scope('review', 'neutral-op', 'neutral-session'):
                client.complete(messages(client))
            assert guard.snapshot()['calls_used'] == 0
    if role == 'memory' and source_protocol == 'preference_source_refs_v1':
        old, new = map(json.loads, actual)
        assert old.pop('response_format') != new.pop('response_format')
        assert old == new
    else:
        assert actual[0] == actual[1]

