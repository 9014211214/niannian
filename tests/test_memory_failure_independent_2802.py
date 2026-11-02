"""Independent installed-App failure lifecycle checks; synthetic HTTP only."""
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import json
import sqlite3
import threading

import pytest

from aria.engine.memory.offline_queue import OfflineSessionQueue, OFFLINE_AUDIT_LIMIT
from aria.engine.model.persona import EpisodicMemoryEntry, OfflineSessionRecord
from aria.engine.policy.max_release import tls_dispatch
from tests import test_installable_settled_reflection as helpers
from tests.test_installable_settled_reflection import offline


def test_failed_memory_then_reopen_and_complete_next_session(tmp_path, monkeypatch):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[',
                               fault='length', failure_role='memory')
    original_open = tls_dispatch.open_observed
    def once_failed(request, **kwargs):
        result = original_open(request, **kwargs)
        wire = json.loads(request.data)
        if wire['max_tokens'] == 4000 and calls.count('memory') > 1:
            return helpers.Response(helpers.response(app.clients.memory), wire.get('stream') is True)
        return result
    monkeypatch.setattr(tls_dispatch, 'open_observed', once_failed)
    prior = app._seed_persona(app._settled_owner)
    prior.episodic_memory = [EpisodicMemoryEntry(id='independent-old-fact', summary='已有的本地软件夹具。')]
    app.store.save(prior)
    assert not helpers.one_session(app, 1)['ok']
    before = deepcopy(app.call_guard.snapshot())
    original_raw = helpers.read(before['attempts'][-1]['public_first_output']['path'])
    app.close()
    app = helpers.reopen(tmp_path)
    try:
        assert helpers.one_session(app, 5)['ok']
        after = app.call_guard.snapshot()
        assert not after['stopped']
        assert after['attempts'][:len(before['attempts'])] == before['attempts']
        assert Decimal(after['known_cost_usd']) == sum(Decimal(str(row['usage']['cost'])) for row in after['attempts'])
        profile = app.store.load(app._settled_owner)
        assert [record.outcome for record in profile.offline_session_audit] == ['memory_failed', 'completed']
        assert profile.episodic_memory[0].id == 'independent-old-fact'
        assert not profile.offline_sessions
        assert calls.count('memory') == 2 and calls.count('reflection') == 1
        assert helpers.read(before['attempts'][-1]['public_first_output']['path']) == original_raw
    finally:
        app.close()


@pytest.mark.parametrize('kind', ['memory_binding', 'terminal_audit'])
def test_memory_failure_cannot_settle_changed_identity_or_terminal(tmp_path, monkeypatch, kind):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[', failure_role='memory')
    if kind == 'memory_binding':
        original = app.call_guard._finish
        def changed(row):
            if row['role'] == 'memory':
                row['memory_validation_contract_binding']['session_id'] = 'wrong-session'
            return original(row)
        monkeypatch.setattr(app.call_guard, '_finish', changed)
    try:
        assert not helpers.one_session(app, 1)['ok']
        if kind == 'terminal_audit':
            profile = app.store.load(app._settled_owner)
            profile.offline_session_audit[0].outcome = 'completed'
            app.store.save(profile)
        assert app.call_guard.snapshot()['stopped']
        assert not app.start()['ok']
        assert calls.count('memory') == 1 and 'reflection' not in calls
    finally:
        app.close()


def test_normal_bounded_audit_retention_does_not_revoke_settled_memory_failure(tmp_path, monkeypatch):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[', failure_role='memory')
    try:
        assert not helpers.one_session(app, 1)['ok']
        original = deepcopy(app.call_guard.snapshot())
        profile = app.store.load(app._settled_owner)
        receipt = helpers.read(next(app._settled_reflections.glob('*.json')))
        job = OfflineSessionRecord.model_validate(receipt['original_job'])
        at = datetime.fromisoformat(receipt['audit']['finished_at'])
        # Exercise the queue's real bounded-retention operation, rather than
        # deleting history directly. These are software state fixtures only.
        for index in range(OFFLINE_AUDIT_LIMIT):
            later = job.model_copy(deep=True)
            later.session_id = 'later-retention-' + str(index)
            OfflineSessionQueue._audit(profile, later, at + timedelta(seconds=index + 1),
                                       'expired', 'offline_session_expired')
        app.store.save(profile)
        assert len(profile.offline_session_audit) == OFFLINE_AUDIT_LIMIT
        assert receipt['audit'] not in [row.model_dump(mode='json') for row in profile.offline_session_audit]
        state = app.call_guard.snapshot()
        assert not state['stopped']
        assert state['attempts'] == original['attempts']
        assert calls.count('memory') == 1
        app.close()
        app = helpers.reopen(tmp_path)
        assert not app.call_guard.snapshot()['stopped']
        assert app.start()['ok']
        assert calls.count('memory') == 1
    finally:
        app.close()


@pytest.mark.parametrize('mutation', ['early_delete', 'old_time', 'reordered',
                                     'duplicate', 'naive_time', 'requeued'])
def test_missing_memory_audit_requires_real_later_retention_window(tmp_path, monkeypatch, mutation):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[', failure_role='memory')
    try:
        assert not helpers.one_session(app, 1)['ok']
        profile = app.store.load(app._settled_owner)
        receipt = helpers.read(next(app._settled_reflections.glob('*.json')))
        job = OfflineSessionRecord.model_validate(receipt['original_job'])
        at = datetime.fromisoformat(receipt['audit']['finished_at'])
        for index in range(OFFLINE_AUDIT_LIMIT):
            later = job.model_copy(deep=True)
            later.session_id = 'later-retention-' + str(index)
            OfflineSessionQueue._audit(profile, later, at + timedelta(seconds=index + 1),
                                       'expired', 'offline_session_expired')
        if mutation == 'early_delete':
            profile.offline_session_audit.pop()
        elif mutation == 'old_time':
            profile.offline_session_audit[0].finished_at = (at - timedelta(seconds=1)).isoformat()
        elif mutation == 'reordered':
            profile.offline_session_audit.reverse()
        elif mutation == 'duplicate':
            profile.offline_session_audit[0].session_id = profile.offline_session_audit[1].session_id
        elif mutation == 'naive_time':
            profile.offline_session_audit[0].finished_at = at.replace(tzinfo=None).isoformat()
        else:
            profile.offline_sessions.append(job)
        app.store.save(profile)
        assert app.call_guard.snapshot()['stopped']
        assert not app.start()['ok']
        assert calls.count('memory') == 1 and 'reflection' not in calls
    finally:
        app.close()


def test_auto_memory_failure_tolerance_does_not_allow_other_state_changes(tmp_path, monkeypatch):
    from aria.engine.policy.max_release.reflection_failure import failure_binding, validate_state_pair
    from tests.test_minimal_auto_review import enable, wait_worker
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[', failure_role='memory')
    try:
        enable(app, 2)
        sid = app.start()['state']['session_id']
        assert app.submit('独立软件状态检查。', 'auto-input', sid)['ok']
        assert app.seal(sid)['ok']
        wait_worker(app)
        assert not app.call_guard.snapshot()['stopped']
        receipt = helpers.read(next(app._settled_reflections.glob('*.json')))
        row = receipt['ledger_row']
        before = helpers.read(failure_binding(row)['before_state']['path'])
        after = helpers.read(receipt['after_failed_state']['path'])
        request_id = receipt['failed_review']['request_id']
        for mutation in ('enabled', 'limit', 'session', 'other_event'):
            changed = deepcopy(after)
            auto = changed['run']['auto_review']
            if mutation == 'enabled':
                auto['enabled'] = False
            elif mutation == 'limit':
                auto['daily_limit'] += 1
            elif mutation == 'session':
                auto['events'][0]['session_id'] = 'another-session'
            else:
                other = deepcopy(auto['events'][0])
                other['request_id'] = 'unrelated-request'
                auto['events'].append(other)
            with pytest.raises(ValueError):
                validate_state_pair(before, changed, request_id, sid, 'memory')
        assert calls.count('memory') == 1 and 'reflection' not in calls
    finally:
        app.close()


@pytest.mark.parametrize('stage', ['memory', 'reflection'])
@pytest.mark.parametrize('action', ['start', 'waiting_submit', 'waiting_during_terminal'])
def test_background_content_failure_preserves_new_session_and_waiting_input(
        tmp_path, monkeypatch, stage, action):
    app, calls = helpers.setup(tmp_path, monkeypatch,
        '{"episodes":[' if stage == 'memory' else helpers.failure_body('envelope'),
        fault='length' if stage == 'memory' else None, failure_role=stage)
    entered, release, accepted = (threading.Event() for _ in range(3))
    terminal_entered, terminal_release = threading.Event(), threading.Event()
    if action == 'waiting_during_terminal':
        import aria.voice.settled_reflection as integration
        original_seal = integration.seal_failed_job
        def pause_terminal(*args, **kwargs):
            terminal_entered.set()
            assert terminal_release.wait(8), 'test did not release terminal write'
            return original_seal(*args, **kwargs)
        monkeypatch.setattr(integration, 'seal_failed_job', pause_terminal)
    original_open = tls_dispatch.open_observed
    limit = 4000 if stage == 'memory' else 2800
    def delayed(request, **kwargs):
        if json.loads(request.data)['max_tokens'] == limit:
            entered.set()
            assert release.wait(8), 'test did not release synthetic response'
        return original_open(request, **kwargs)
    monkeypatch.setattr(tls_dispatch, 'open_observed', delayed)
    worker = sender = None
    outputs = []
    text = '新的会话原话须完整保留。'
    try:
        old_sid = app.start()['state']['session_id']
        for index in range(3):
            assert app.submit('先前会话软件输入。', 'old-input-' + str(index), old_sid)['ok']
        assert app.seal(old_sid)['ok']
        assert app.reflect('background-failure', wait=False)['ok']
        worker = app._review_thread
        assert entered.wait(5)
        start = app.start()
        assert start['ok']
        new_sid = start['state']['session_id']
        assert new_sid != old_sid
        if action.startswith('waiting_'):
            sender = threading.Thread(target=lambda: outputs.append(app.submit(
                text, 'new-input', new_sid,
                on_waiting_input=lambda frame: accepted.set() or True)))
            sender.start()
            assert accepted.wait(5)
            assert app.state()['waiting_input']['status'] == 'waiting'
            assert app.state()['waiting_input']['request_sent'] is False
            assert calls.count('online_fast') == 3
        release.set()
        if action == 'waiting_during_terminal':
            assert terminal_entered.wait(5)
            assert sender.is_alive()
            assert calls.count('online_fast') == 3
            terminal_release.set()
        worker.join(8)
        assert not worker.is_alive()
        if sender is not None:
            sender.join(8)
            assert not sender.is_alive()
        else:
            outputs.append(app.submit(text, 'new-input', new_sid))
        assert outputs and outputs[0]['ok'], outputs
        usage = app.call_guard.snapshot()
        assert not usage['stopped'] and usage['unknown_cost_requests'] == 0
        failed, = [row for row in usage['attempts'] if row['status'] == 'failed']
        assert failed['role'] == stage and failed['usage']['cost'] > 0
        assert all(row['physical_attempt'] == 1 for row in usage['attempts'])
        assert len(usage['attempts']) == len(calls)
        profile = app.store.load(app._settled_owner)
        assert profile.pending_session.session_id == new_sid
        assert profile.pending_session.turns[-1].user == text
        assert profile.offline_session_audit[-1].outcome == stage + '_failed'
        assert not profile.offline_sessions
        assert app.state()['reviews'][-1]['status'] == 'failed'
        expected = ['online_fast', 'safety_review'] * 3 + ['memory']
        if stage == 'reflection':
            expected += ['reflection']
        assert calls == expected + ['online_fast', 'safety_review']
    finally:
        release.set()
        terminal_release.set()
        if worker is not None:
            worker.join(8)
        if sender is not None:
            sender.join(8)
        app.close()


@pytest.mark.parametrize('mutation', ['business', 'permission', 'sealed_source', 'unknown_fee', 'forget'])
def test_legitimate_new_session_cannot_mask_unrelated_failed_stage_mutations(
        tmp_path, monkeypatch, mutation):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[',
                               fault='length', failure_role='memory', unknown=mutation == 'unknown_fee')
    entered, release = threading.Event(), threading.Event()
    original_open = tls_dispatch.open_observed
    errors = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: errors.append(args.exc_type))
    def delayed(request, **kwargs):
        if json.loads(request.data)['max_tokens'] == 4000:
            entered.set()
            assert release.wait(8)
        return original_open(request, **kwargs)
    monkeypatch.setattr(tls_dispatch, 'open_observed', delayed)
    worker = None
    try:
        old_sid = app.start()['state']['session_id']
        for index in range(3):
            assert app.submit('原会话软件输入。', 'before-' + str(index), old_sid)['ok']
        assert app.seal(old_sid)['ok']
        assert app.reflect('background-failure', wait=False)['ok']
        worker = app._review_thread
        assert entered.wait(5)
        new_sid = app.start()['state']['session_id']
        assert new_sid != old_sid
        if mutation == 'forget':
            assert app.forget_run()['ok']
        elif mutation == 'permission':
            with sqlite3.connect(app.run_dir / 'care.sqlite3') as db:
                db.execute('CREATE TABLE independent_unrelated_change(value INTEGER)')
        elif mutation != 'unknown_fee':
            profile = app.store.load(app._settled_owner)
            if mutation == 'business':
                profile.disc.dominance = .99
            else:
                profile.offline_sessions[0].turns[0].user = '未经授权替换的旧来源。'
            app.store.save(profile)
            # The installed App owns an active in-memory profile. Inject the
            # same committed corruption there so the stage merge observes it;
            # a disk-only write would be overwritten before this boundary.
            app._sessions[app._settled_owner][0]._accept_offline_profile(profile)
        release.set()
        worker.join(8)
        assert not worker.is_alive()
        assert app.call_guard.snapshot()['stopped']
        assert not app.submit('不能越过失败的状态核验。', 'after', new_sid)['ok']
        assert calls == ['online_fast', 'safety_review'] * 3 + ['memory']
        profile = app.store.load(app._settled_owner)
        if mutation == 'forget':
            assert profile is None
            assert not app.state()['reviews'] and not app.state()['turns']
        else:
            assert not profile.offline_session_audit
    finally:
        release.set()
        if worker is not None:
            worker.join(8)
        app.close()


def test_online_between_successful_memory_and_failed_reflection_keeps_own_receipt_scope(tmp_path, monkeypatch):
    app, calls = helpers.setup(tmp_path, monkeypatch, helpers.failure_body('envelope'))
    entered, release, accepted = (threading.Event() for _ in range(3))
    original_open = tls_dispatch.open_observed
    def delayed(request, **kwargs):
        if json.loads(request.data)['max_tokens'] == 4000:
            entered.set()
            assert release.wait(8)
        return original_open(request, **kwargs)
    monkeypatch.setattr(tls_dispatch, 'open_observed', delayed)
    worker = sender = None
    outputs = []
    text = '优先处理这次新输入。'
    try:
        old_sid = app.start()['state']['session_id']
        for index in range(3):
            assert app.submit('原会话软件输入。', 'old-' + str(index), old_sid)['ok']
        assert app.seal(old_sid)['ok']
        assert app.reflect('background-review', wait=False)['ok']
        worker = app._review_thread
        assert entered.wait(5)
        new_sid = app.start()['state']['session_id']
        sender = threading.Thread(target=lambda: outputs.append(app.submit(text, 'new-online', new_sid,
            on_waiting_input=lambda frame: accepted.set() or True)))
        sender.start()
        assert accepted.wait(5)
        release.set()
        sender.join(8)
        worker.join(8)
        assert not sender.is_alive() and not worker.is_alive()
        assert outputs[0]['ok']
        assert calls == ['online_fast', 'safety_review'] * 3 + [
            'memory', 'online_fast', 'safety_review', 'reflection']
        state = app.call_guard.snapshot()
        assert not state['stopped'] and state['unknown_cost_requests'] == 0
        failed, = [row for row in state['attempts'] if row['status'] == 'failed']
        assert failed['role'] == 'reflection' and failed['session_id'] == old_sid
        profile = app.store.load(app._settled_owner)
        assert profile.pending_session.session_id == new_sid
        assert profile.pending_session.turns[-1].user == text
        assert profile.offline_session_audit[-1].outcome == 'reflection_failed'
        assert profile.offline_session_audit[-1].memory_completed is True
        assert not profile.offline_sessions
        assert app.submit('继续这个新会话。', 'new-second', new_sid)['ok']
        assert calls.count('memory') == calls.count('reflection') == 1
    finally:
        release.set()
        if sender is not None:
            sender.join(8)
        if worker is not None:
            worker.join(8)
        app.close()


@pytest.mark.parametrize('changed_field', ['business', 'sealed_source', 'pending_session'])
def test_external_persisted_change_is_not_overwritten_by_cached_offline_profile(
        tmp_path, monkeypatch, changed_field):
    app, calls = helpers.setup(tmp_path, monkeypatch, '{"episodes":[',
                               fault='length', failure_role='memory')
    entered, release = threading.Event(), threading.Event()
    original_open = tls_dispatch.open_observed
    errors = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: errors.append(args.exc_type))
    def delayed(request, **kwargs):
        if json.loads(request.data)['max_tokens'] == 4000:
            entered.set()
            assert release.wait(8)
        return original_open(request, **kwargs)
    monkeypatch.setattr(tls_dispatch, 'open_observed', delayed)
    worker = None
    def persisted_value(profile):
        if changed_field == 'business':
            return profile.disc.dominance
        if changed_field == 'sealed_source':
            return profile.offline_sessions[0].turns[0].user
        return profile.pending_session.scene_id
    try:
        old_sid = app.start()['state']['session_id']
        for index in range(3):
            assert app.submit('原会话软件输入。', 'old-' + str(index), old_sid)['ok']
        assert app.seal(old_sid)['ok']
        assert app.reflect('background-review', wait=False)['ok']
        worker = app._review_thread
        assert entered.wait(5)
        new_sid = app.start()['state']['session_id']
        profile = app.store.load(app._settled_owner)
        if changed_field == 'business':
            profile.disc.dominance = .99
        elif changed_field == 'sealed_source':
            profile.offline_sessions[0].turns[0].user = '持久来源已被外部改变。'
        else:
            profile.pending_session.scene_id = 'changed-persistent-scene'
        app.store.save(profile)
        changed = persisted_value(profile)
        durable = profile.model_dump(mode='json')
        # Deliberately do not update the active cache: this is the independent
        # persistent-writer conflict, not a simulated App-owned transaction.
        assert persisted_value(app._sessions[app._settled_owner][0].persona) != changed
        release.set()
        worker.join(8)
        assert not worker.is_alive()
        assert app.call_guard.snapshot()['stopped']
        assert persisted_value(app.store.load(app._settled_owner)) == changed
        assert app.store.load(app._settled_owner).model_dump(mode='json') == durable
        assert not app.submit('后续请求不能写回旧缓存。', 'new-input', new_sid)['ok']
        assert persisted_value(app.store.load(app._settled_owner)) == changed
        assert app.store.load(app._settled_owner).model_dump(mode='json') == durable
        assert calls == ['online_fast', 'safety_review'] * 3 + ['memory']
    finally:
        release.set()
        if worker is not None:
            worker.join(8)
        try:
            app.close()
        except RuntimeError as error:
            assert '会话尚未安全封存' in str(error)
    assert persisted_value(app.store.load(app._settled_owner)) == changed
    assert app.store.load(app._settled_owner).model_dump(mode='json') == durable


@pytest.mark.parametrize('owner_action', ['start', 'confirm', 'withdraw'])
def test_successful_offline_stage_accepts_normal_new_checkpoint_and_owner_action(
        tmp_path, monkeypatch, owner_action):
    from tests.test_minimal_offline_background import seed_owner_records
    app, calls = helpers.setup(tmp_path, monkeypatch, helpers.BARE)
    entered, release = threading.Event(), threading.Event()
    original_open = tls_dispatch.open_observed
    def delayed(request, **kwargs):
        if json.loads(request.data)['max_tokens'] == 4000:
            entered.set()
            assert release.wait(8)
        return original_open(request, **kwargs)
    monkeypatch.setattr(tls_dispatch, 'open_observed', delayed)
    worker = None
    try:
        old_sid = app.start()['state']['session_id']
        for index in range(3):
            assert app.submit('安排之前先问我。', 'old-' + str(index), old_sid)['ok']
        if owner_action != 'start':
            seed_owner_records(app, old_sid)
            if owner_action == 'confirm':
                proposal = app._profile()
                proposal.stated_preferences[0].status = 'proposed'
                proposal.stated_preferences[0].confirmation = None
                app.store.save(proposal)
        assert app.seal(old_sid)['ok']
        assert app.reflect('background-success', wait=False)['ok']
        worker = app._review_thread
        assert entered.wait(5)
        new_sid = app.start()['state']['session_id']
        assert new_sid != old_sid
        if owner_action != 'start':
            assert app.preference_action(owner_action, 'pref_owner')['ok']
        release.set()
        worker.join(8)
        assert not worker.is_alive()
        assert app.state()['reviews'][-1]['status'] == 'completed'
        assert not app.call_guard.snapshot()['stopped']
        profile = app.store.load(app._settled_owner)
        assert profile.pending_session.session_id == new_sid
        assert profile.offline_session_audit[-1].outcome == 'completed'
        if owner_action != 'start':
            assert profile.stated_preferences[0].status == ('withdrawn' if owner_action == 'withdraw' else 'confirmed')
        assert app.submit('新会话继续正常保存。', 'new-input', new_sid)['ok']
        assert app.store.load(app._settled_owner).pending_session.turns[-1].user == '新会话继续正常保存。'
        assert calls.count('memory') == calls.count('reflection') == 1
    finally:
        release.set()
        if worker is not None:
            worker.join(8)
        app.close()
