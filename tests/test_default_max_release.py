"""Default release selection and explicit historical compatibility, offline only."""
import json
import socket

import pytest

from aria import cli
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from aria.engine.policy.max_release import tls_dispatch
from aria.voice import minimal
from tests.test_installable_max_release import metadata, messages, prices
from tests.test_release_assembly import release_policy
from tests.test_online_structured import metadata as flash_metadata
from tests.test_reserved_calls import policy as historical_policy
from tests.test_deepseek_release import metadata as deepseek_metadata, prices as deepseek_prices


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network or actual credential access forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)
    monkeypatch.setattr(cli, 'load_local_env', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)


def arguments(tmp_path, profile=None, live=True):
    args = ['--minimal-data-dir', str(tmp_path / 'data')]
    if profile is not None:
        args += ['--release-profile', profile]
    if not live:
        return args
    if profile == 'legacy':
        p = historical_policy()
    elif profile == 'release_v1':
        p = release_policy()
        for name, value in zip(('models', 'online'), flash_metadata()):
            (tmp_path / (name + '.json')).write_text(json.dumps(value), encoding='utf8')
        args += ['--online-models-metadata', str(tmp_path / 'models.json'),
                 '--online-endpoints-metadata', str(tmp_path / 'online.json')]
    else:
        deepseek = profile in (None, 'release_deepseek_v1')
        p = deepseek_prices() if deepseek else prices()
        for name, raw in zip(('models', 'online', 'offline'), deepseek_metadata() if deepseek else metadata()):
            (tmp_path / (name + '.json')).write_bytes(raw)
        args += ['--online-models-metadata', str(tmp_path / 'models.json'),
                 '--online-endpoints-metadata', str(tmp_path / 'online.json'),
                 '--offline-endpoints-metadata', str(tmp_path / 'offline.json')]
    (tmp_path / 'policy.json').write_text(json.dumps(p), encoding='utf8')
    return args + ['--enable-model-calls', '--model-max-requests', '8',
        '--model-stop-cost-usd', '1', '--model-reservation-policy', str(tmp_path / 'policy.json')]


def run_entry(monkeypatch, args, live, inspect_app=None):
    seen = {}
    if live:
        monkeypatch.setattr(cli, 'load_local_env', lambda: None)
        monkeypatch.setattr('aria.runtime.openrouter_api_key', lambda: 'inert-software-fixture')
        monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', lambda: 'inert-software-fixture')
    class Server:
        server_port = 8080
        def __init__(self, app): self.app = app
        def serve_forever(self):
            seen['clients'] = self.app.clients
            seen['mode'] = self.app.mode
            seen['guard'] = type(self.app.call_guard)
            if live:
                assert self.app.call_guard.snapshot()['calls_used'] == 0
            else:
                assert self.app.call_guard is None
            if inspect_app is not None:
                inspect_app(self.app)
        def server_close(self): self.app.close()
    monkeypatch.setattr(minimal, 'make_minimal_server', lambda app, *a: Server(app))
    assert cli.web_main(args) == 0
    return seen


@pytest.mark.parametrize('profile', [None, 'release_deepseek_v1', 'release_max_v1', 'release_v1', 'legacy'])
def test_preview_never_reads_key_and_stays_inert(tmp_path, monkeypatch, profile):
    seen = run_entry(monkeypatch, arguments(tmp_path, profile, live=False), False)
    assert seen['mode'] == 'local_preview'
    assert seen['clients'].model_calls_enabled is False


@pytest.mark.parametrize('profile', [None, 'release_deepseek_v1', 'release_max_v1', 'release_v1', 'legacy'])
def test_actual_entry_selects_requested_bundle_without_dispatch(tmp_path, monkeypatch, profile):
    seen = run_entry(monkeypatch, arguments(tmp_path, profile), True)
    expected = profile or 'release_deepseek_v1'
    assert seen['mode'] == 'live'
    if expected == 'legacy':
        assert seen['clients'].online.online_decision_version == 'online_turn_decision_v3'
    else:
        assert seen['clients'].release_profile == expected
        if expected in ('release_max_v1', 'release_deepseek_v1'):
            assert seen['guard'] is MaxReleaseCallGuard
            assert seen['clients'].online.model == ('deepseek/deepseek-v4.1-flash-1017'
                if expected == 'release_deepseek_v1' else 'qwen/qwen3.8-max-0902')
            assert seen['clients'].memory.model == 'anthropic/claude-sonnet-5'


class Captured(BaseException): pass


@pytest.mark.parametrize('role', ['online', 'memory', 'reflection', 'safety_review'])
def test_default_actual_factory_wire_equals_explicit_deepseek(tmp_path, monkeypatch, role):
    records = []
    for index, profile in enumerate((None, 'release_deepseek_v1')):
        d = tmp_path / str(index); d.mkdir()
        def inspect_app(app):
            client = getattr(app.clients, role)
            guard = app.call_guard
            captured = []
            def stop_at_wire(c, request, **options):
                captured.append((request.data, options))
                raise Captured()
            monkeypatch.setattr(guard, '_request_once', stop_at_wire)
            with pytest.raises(Captured):
                with guard.scope('turn' if role in ('online', 'safety_review') else 'review',
                                 'neutral-op', 'neutral-session'):
                    client.complete(messages(client))
            assert guard.snapshot()['calls_used'] == 0
            records.append((captured, guard._binding(client)))
        run_entry(monkeypatch, arguments(d, profile), True, inspect_app)
    assert records[0] == records[1]


def test_implicit_historical_protocol_keeps_compatibility(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, '_minimal_main', lambda args: seen.append(args) or 0)
    # Historical explicit protocol selection must still use the historical path.
    args = arguments(tmp_path, 'legacy')
    offset = args.index('--release-profile'); del args[offset:offset + 2]
    args += ['--online-decision-version', 'online_turn_decision_v3']
    assert cli.web_main(args) == 0
    assert seen[0].release_profile == 'legacy'
    assert seen[0].online_decision_version == 'online_turn_decision_v3'


@pytest.mark.parametrize('extra', [[], ['--release-profile', 'release_max_v1'],
    ['--release-profile', 'release_max_v1', '--online-decision-version', 'online_turn_decision_v3']])
def test_missing_metadata_or_conflicting_max_configuration_stops_before_key(extra):
    with pytest.raises(SystemExit) as raised:
        cli.web_main(['--enable-model-calls'] + extra)
    assert raised.value.code == 2
