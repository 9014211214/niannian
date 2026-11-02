"""Real product paths with fake HTTP; these do not measure model semantics."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket

import pytest

from aria.engine.policy.max_release import tls_dispatch
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
from aria.voice.minimal import MinimalWebApp
from tests import test_deepseek_release_guard as deepseek
from tests import test_installable_max_release as original
from tests.test_confirm_judge_v7 import judge_response
from tests.test_stated_preferences import _profile


BODY = '中性软件正文。'
INPUT = '中性软件输入。'
DISPOSITION = 'unclear_known_cost_format_failure_v1'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network and credential access forbidden')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key', forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', forbidden)
    monkeypatch.setattr(tls_dispatch, 'open_observed', forbidden)


def setup(tmp_path, monkeypatch, *, profile='deepseek', outcome='empty', max_requests=8,
          budget='1', bill_reserved=False):
    fixtures = deepseek if profile == 'deepseek' else original
    clients = fixtures.clients() if profile == 'deepseek' else fixtures.make_clients()
    selected = MaxReleaseCallGuard(tmp_path / 'calls.sqlite3', max_requests=max_requests,
        max_observed_cost_usd=budget,
        reservation_policy=fixtures.policy() if profile == 'deepseek' else fixtures.prices())
    clock = [1000.0]
    selected._pacing_clock = lambda: clock[0]
    selected._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    app = MinimalWebApp(tmp_path / 'app', clients=clients, model_call_guard=selected)
    wires = []
    delivered = []
    def opened(request, **kwargs):
        wire = json.loads(request.data)
        wires.append(wire)
        kwargs['trace'].preparation_recovery.bind_request_headers(request)
        kwargs['trace'].preparation_recovery.before_preparation()
        is_judge = wire['max_tokens'] == 60
        is_review = wire['max_tokens'] == 1024
        client = clients.confirm_judge if is_judge else clients.safety_review if is_review else clients.online
        data = (deepseek.response(client) if profile == 'deepseek' else
                judge_response(client) if is_judge else original.response(client))
        if bill_reserved:
            data['usage']['cost'] = float(selected.snapshot()['attempts'][-1]['reservation']['reserved_cost_usd'])
        choice = data['choices'][0]
        if is_review:
            from tests.test_safety_review_v2 import answer
            # Fixed reviewer label for the fixture's first saved-status claim;
            # the production reviewer now receives it instead of a coarse
            # pre-review deletion. This is not a semantic accuracy assertion.
            choice['message']['content'] = json.dumps(answer(delete=[0]))
        elif not is_judge:
            choice['message']['content'] = json.dumps({
                'response_segments': [{'text': '我已经记下了。', 'kind': 'preference_state'},
                                      {'text': BODY}], 'safety_concern': 'none'}, ensure_ascii=False)
        else:
            if outcome in {'empty', 'unknown_cost', 'wrong_provider', 'wrong_model', 'cancelled',
                           'reasoning_returned', 'reasoning_content', 'over_budget', 'bad_total', 'over_tokens'}:
                choice['message']['content'] = None
            elif outcome == 'truncated':
                choice['finish_reason'] = 'length'
                choice['message']['content'] = '{"action":'
            elif outcome == 'invalid_json':
                choice['message']['content'] = 'not json'
            elif outcome == 'invalid_action':
                choice['message']['content'] = '{"action":"save"}'
            else:
                choice['message']['content'] = json.dumps({'action': outcome})
            if outcome == 'unknown_cost':
                del data['usage']['cost']
            elif outcome == 'wrong_provider':
                data['provider'] = 'Other Provider'
            elif outcome == 'wrong_model':
                data['model'] = 'other/model'
            elif outcome == 'cancelled':
                selected.halt()
            elif outcome == 'reasoning_returned':
                data['usage']['completion_tokens_details']['reasoning_tokens'] = 1
            elif outcome == 'reasoning_content':
                choice['message']['reasoning_content'] = 'synthetic-hidden-marker'
            elif outcome == 'over_budget':
                data['usage']['cost'] = 10
            elif outcome == 'bad_total':
                data['usage']['total_tokens'] += 1
            elif outcome == 'over_tokens':
                data['usage'].update(completion_tokens=61, total_tokens=81)
        delivered.append(deepcopy(data))
        return original.Response(data)
    monkeypatch.setattr(tls_dispatch, 'open_observed', opened)
    session = app.start()['state']['session_id']
    engine, ctx = app._sessions[app._settled_owner]
    engine.persona.stated_preferences = deepcopy(_profile(confirmed=False).stated_preferences)
    app.store.save(engine.persona)
    invitation = app.preference_invitation(engine.persona.stated_preferences[0].id, session)['preference_invitation']
    assert app.preference_invitation_ack(invitation['id'], invitation['question_sha256'], session)['ok']
    return app, selected, session, ctx, wires, delivered


@pytest.mark.parametrize('profile', ['deepseek', 'max'])
@pytest.mark.parametrize('outcome', ['empty', 'truncated', 'invalid_json', 'invalid_action'])
def test_format_failure_keeps_first_reply_cost_and_next_real_product_call(tmp_path, monkeypatch, profile, outcome):
    app, guard, sid, ctx, wires, delivered = setup(tmp_path, monkeypatch, profile=profile, outcome=outcome)
    try:
        assert app.submit(INPUT, 'turn1', sid)['ok']
        first = deepcopy(app._run['turns'][0])
        assert first['response'] == BODY + '\n这条我先不记，你想记的时候再说一声。'
        assert '记下了' not in first['response']
        assert app.store.load(app._settled_owner).stated_preferences[0].status == 'proposed'
        assert ctx.working_memory['online_turn_decision']['confirm_judge']['status'] == 'failed'
        state = guard.snapshot()
        assert not state['stopped'] and state['unknown_cost_requests'] == 0
        assert state['reserved_cost_usd'] == '0' and state['calls_used'] == 3
        assert Decimal(state['known_cost_usd']) == sum(Decimal(str(v['usage']['cost'])) for v in delivered)
        online, judge, review = state['attempts']
        assert review['role'] == 'safety_review' and review['status'] == 'received'
        assert online['status'] == 'received' and judge['role'] == 'confirm_judge'
        receipt = json.loads(Path(judge['public_first_output']['path']).read_bytes())
        assert receipt['public_first_output'][0]['content'] == delivered[1]['choices'][0]['message']['content']
        if outcome in {'empty', 'truncated'}:
            assert judge['status'] == 'failed' and judge['confirm_judge_disposition'] == DISPOSITION
            assert judge['failure_phase'] == 'receipt_validation' and judge['physical_attempt'] == 1
        else:
            assert judge['status'] == 'received' and 'confirm_judge_disposition' not in judge
        assert [v['max_tokens'] for v in wires] == [4096, 60, 1024]
        assert app.submit(INPUT, 'turn1', sid)['duplicate'] and len(wires) == 3
        assert app.submit('另一条中性输入。', 'turn2', sid)['ok']
        assert [v['max_tokens'] for v in wires] == [4096, 60, 1024, 4096, 1024]
        after = guard.snapshot()
        assert after['attempts'][:3] == state['attempts']
        assert app._run['turns'][0] == first
        assert after['attempts'][3]['status'] == 'received' and not after['stopped']
        assert after['attempts'][3]['pacing']['start_gap_ms'] >= 120000
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize('action', ['confirm', 'decline', 'defer', 'unclear'])
def test_legal_actions_retain_the_actual_product_commit_semantics(tmp_path, monkeypatch, action):
    app, guard, sid, ctx, wires, _ = setup(tmp_path, monkeypatch, outcome=action)
    try:
        assert app.submit(INPUT, 'turn1', sid)['ok']
        saved = app.store.load(app._settled_owner).stated_preferences[0]
        assert saved.status == {'confirm': 'confirmed', 'decline': 'declined'}.get(action, 'proposed')
        reply = app._run['turns'][0]['response']
        assert reply.startswith(BODY) and ('好，这条记下了。' in reply) is (action == 'confirm')
        trace = ctx.working_memory['online_turn_decision']['confirm_judge']
        assert trace['status'] == 'parsed' and trace['action'] == action
        state = guard.snapshot()
        assert all(row['status'] == 'received' and 'confirm_judge_disposition' not in row for row in state['attempts'])
        assert not state['stopped'] and len(wires) == 3
        assert app.submit('另一条中性输入。', 'turn2', sid)['ok'] and len(wires) == 5
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize('failure', ['unknown_cost', 'wrong_provider', 'wrong_model', 'cancelled',
    'reasoning_returned', 'reasoning_content', 'over_budget', 'bad_total', 'over_tokens'])
def test_nonformat_failures_still_stop_without_replay_or_preference_save(tmp_path, monkeypatch, failure):
    app, guard, sid, _, wires, _ = setup(tmp_path, monkeypatch, outcome=failure)
    try:
        app.submit(INPUT, 'turn1', sid)
        state = guard.snapshot()
        assert state['stopped'] and len(wires) == 2
        judge = state['attempts'][1]
        assert judge['status'] == 'failed' and 'confirm_judge_disposition' not in judge
        assert app.store.load(app._settled_owner).stated_preferences[0].status == 'proposed'
        assert '记下了' not in app._run['turns'][0]['response']
        if failure in {'unknown_cost', 'wrong_provider', 'wrong_model', 'over_budget', 'bad_total', 'over_tokens'}:
            assert state['unknown_cost_requests'] == 1 and Decimal(state['reserved_cost_usd']) > 0
        else:
            assert state['unknown_cost_requests'] == 0 and state['reserved_cost_usd'] == '0'
        assert not app.submit('另一条中性输入。', 'turn2', sid)['ok']
        assert len(wires) == 2 and guard.snapshot()['attempts'] == state['attempts']
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize('boundary', ['request_limit', 'budget'])
def test_format_failure_does_not_hide_reached_limits(tmp_path, monkeypatch, boundary):
    budget = '1'
    if boundary == 'budget':
        # Obtain this neutral fixture's exact existing reservation, without a
        # network or a provider call, then test the boundary in a fresh store.
        probe, probe_guard, sid, _, _, _ = setup(tmp_path / 'reservation', monkeypatch)
        try:
            assert probe.submit(INPUT, 'turn1', sid)['ok']
            budget = str(sum(Decimal(row['reservation']['reserved_cost_usd'])
                             for row in probe_guard.snapshot()['attempts']))
        finally:
            probe.close()
            probe_guard.close()
    app, guard, sid, _, wires, _ = setup(tmp_path / 'boundary', monkeypatch,
        max_requests=2 if boundary == 'request_limit' else 8,
        budget=budget, bill_reserved=boundary == 'budget')
    try:
        assert app.submit(INPUT, 'turn1', sid)['ok']
        state = guard.snapshot()
        expected_calls = 2 if boundary == 'request_limit' else 3
        assert len(wires) == expected_calls and state['attempts'][1]['confirm_judge_disposition'] == DISPOSITION
        assert state['stopped']
        assert state['stop_reason'] == ('request_limit_reached' if boundary == 'request_limit'
                                       else 'observed_cost_threshold_reached')
        assert not app.submit('另一条中性输入。', 'turn2', sid)['ok'] and len(wires) == expected_calls
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize('failure', ['public_capture', 'ledger_write', 'turn_save'])
def test_storage_failure_cannot_be_downgraded_to_confirmation_format(tmp_path, monkeypatch, failure):
    app, guard, sid, _, wires, _ = setup(tmp_path, monkeypatch)
    def failed_write(*args, **kwargs):
        raise OSError('synthetic storage failure')
    if failure == 'public_capture':
        write = guard.content_erasure.write
        def capture(path, value, directory):
            if value.get('role') == 'confirm_judge':
                failed_write()
            return write(path, value, directory)
        monkeypatch.setattr(guard.content_erasure, 'write', capture)
    elif failure == 'ledger_write':
        from aria.engine.policy.minimal_calls import MinimalCallGuard
        finish = MinimalCallGuard._finish
        def save(instance, row):
            if row.get('role') == 'confirm_judge' and row.get('status') == 'failed':
                failed_write()
            return finish(instance, row)
        monkeypatch.setattr(MinimalCallGuard, '_finish', save)
    else:
        write = Path.write_text
        def persist(path, *args, **kwargs):
            if path.parent == app.run_path.parent and path.suffix == '.tmp' and len(wires) >= 2:
                failed_write()
            return write(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'write_text', persist)
    try:
        try:
            app.submit(INPUT, 'turn1', sid)
        except RuntimeError as error:
            assert failure == 'turn_save'
            assert str(error) == 'minimal_run_store_failed'
        state = guard.snapshot()
        expected_calls = 3 if failure == 'turn_save' else 2
        assert state['stopped'] and len(wires) == expected_calls
        assert app.store.load(app._settled_owner).stated_preferences[0].status == 'proposed'
        assert not app.submit('另一条中性输入。', 'turn2', sid)['ok'] and len(wires) == expected_calls
    finally:
        monkeypatch.undo()
        try:
            app.close()
        except RuntimeError:
            # A failed turn checkpoint can also refuse normal session sealing.
            assert failure == 'turn_save'
        finally:
            guard.close()
