"""Opt-in retained-hold 429 recovery, with fake time and no live network."""
from copy import deepcopy
from decimal import Decimal
import json
import sqlite3
import threading

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.product_calls import ProductCallGuard
from tests._model_http_stub import stub_model_http
from tests.test_product_call_recovery import http_failure
from tests.test_reserved_calls import client, policy, receipt, call, stub


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unstubbed network"))


def contract(prior=1):
    return {"schema": "unknown_429_hold_recovery_v1", "prior_unknown_429_count": prior,
            "prior_evidence_sha256": "a" * 64, "source_manifest_sha256": "b" * 64}


def guard(path, *, prior=1, budget="1", check=lambda: True, minimum=0, hold=True):
    options = {"unknown_429_recovery": contract(prior), "recovery_pre_send_check": check} if hold else {}
    g = ProductCallGuard(path, max_requests=40, max_observed_cost_usd=budget,
                         reservation_policy=policy(), min_interval_seconds=minimum, **options)
    clock = [0.0]
    g._pacing_clock = lambda: clock[0]
    g._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    return g, clock


class Response:
    status, headers = 200, {}
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self, size): return json.dumps(self.data).encode()


def replies(monkeypatch, c, failures=1, *, status=429, kwargs=None, headers=None):
    sent = []
    def receive(request, **_):
        sent.append(request.data)
        if len(sent) <= failures:
            error = http_failure(c, status, cost=None, **(kwargs or {}))
            error.headers = headers or {}
            raise error
        return Response(receipt(c))
    stub_model_http(monkeypatch, receive)
    return sent


def test_unknown_hold_survives_recovery_same_wire_and_distinct_attempts(tmp_path, monkeypatch):
    checks = []
    g, clock = guard(tmp_path / 'calls.db', check=lambda: checks.append(True) or True)
    c = client(); g.attach(c)
    sent = replies(monkeypatch, c, failures=2)
    assert call(g, c) == 'local fixture'
    state = g.snapshot()
    assert len(sent) == 3 and len(set(sent)) == 1 and clock[0] == 1800
    assert len(checks) == 5  # initial claim, after each wait, next physical claim
    assert state['unknown_cost_requests'] == 2 and not state['stopped']
    assert state['unknown_429_recovery']['task_unknown_429_count'] == 3
    assert Decimal(state['reserved_cost_usd']) == sum(
        Decimal(row['reservation']['reserved_cost_usd']) for row in state['attempts'][:2])
    assert [row['product_transport_attempt'] for row in state['attempts']] == [1, 2, 3]
    assert len({row['request_id'] for row in state['attempts']}) == 3
    assert all('cost' not in row['usage'] for row in state['attempts'][:2])
    g.close()


@pytest.mark.parametrize('kwargs,status', [({'answer': True}, 429), ({'raw': b''}, 429),
    ({'raw': b'{"error":'}, 429), ({'raw': b'{"error":{"code":429},"other":1}'}, 429),
    ({'raw': b'{"error":{"code":429},"error":{"code":429}}'}, 429), ({}, 503)])
def test_ambiguous_or_output_or_unknown_5xx_never_retry(tmp_path, monkeypatch, kwargs, status):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c)
    sent = replies(monkeypatch, c, kwargs=kwargs, status=status)
    with pytest.raises(MinimalCallStopped): call(g, c)
    state = g.snapshot()
    assert len(sent) == 1 and clock[0] == 0 and state['stopped']
    assert Decimal(state['reserved_cost_usd']) > 0
    g.close()


def test_timeout_unknown_never_retries(tmp_path, monkeypatch):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c); sent = []
    def fail(*a, **k): sent.append(1); raise TimeoutError('fixture')
    stub_model_http(monkeypatch, fail)
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(sent) == 1 and clock[0] == 0 and g.snapshot()['stopped']
    g.close()


@pytest.mark.parametrize('hint,seconds', [('1200', 1200), ('3600', 3600), ('5', 900)])
def test_valid_hint_honored_with_local_floor(tmp_path, monkeypatch, hint, seconds):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c)
    replies(monkeypatch, c, headers={'Retry-After': hint})
    assert call(g, c) == 'local fixture'
    assert clock[0] == seconds
    g.close()


@pytest.mark.parametrize('hint', ['3601', 'nonsense', '-1'])
def test_rejected_hint_leaves_pending_stop_no_send(tmp_path, monkeypatch, hint):
    path = tmp_path / 'calls.db'; g, clock = guard(path); c = client(); g.attach(c)
    sent = replies(monkeypatch, c, headers={'Retry-After': hint})
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(sent) == 1 and clock[0] == 0
    g.close()
    reopened, _ = guard(path)
    assert reopened.snapshot()['stop_reason'] == 'unresolved_product_retry_wait'
    reopened.close()


def test_original_ledger_contract_cannot_be_upgraded_and_v2_binding_immutable(tmp_path):
    old = tmp_path / 'old.db'; g, _ = guard(old, hold=False); g.close(); before = old.read_bytes()
    with pytest.raises(ValueError, match='product_recovery_contract_changed'): guard(old)
    assert old.read_bytes() == before
    fresh = tmp_path / 'new.db'; g, _ = guard(fresh); g.close(); before = fresh.read_bytes()
    with pytest.raises(ValueError, match='unknown_429_recovery_contract_changed'): guard(fresh, prior=0)
    assert fresh.read_bytes() == before
    with pytest.raises(ValueError, match='product_recovery_contract_changed'): guard(fresh, hold=False)


def test_inherited_task_count_stops_sixth_unknown_and_cannot_reset(tmp_path, monkeypatch):
    path = tmp_path / 'calls.db'; g, clock = guard(path, prior=5); c = client(); g.attach(c)
    sent = replies(monkeypatch, c)
    with pytest.raises(MinimalCallStopped): call(g, c)
    state = g.snapshot()
    assert len(sent) == 1 and clock[0] == 0
    assert state['unknown_429_recovery']['task_unknown_429_count'] == 6
    assert state['stop_reason'] == 'unknown_429_task_limit_reached'
    assert not state['can_resume']
    g.close(); reopened, _ = guard(path, prior=5)
    assert reopened.snapshot()['stop_reason'] == 'unknown_429_task_limit_reached'
    reopened.close()


def test_third_unknown_stops_logical_request_no_fourth_attempt(tmp_path, monkeypatch):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c)
    sent = replies(monkeypatch, c, failures=4)
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(sent) == 3 and clock[0] == 1800
    assert g.snapshot()['stop_reason'] == 'unknown_429_request_attempts_exhausted'
    assert g.snapshot()['unknown_cost_requests'] == 3
    g.close()


def test_next_attempt_requires_new_full_reservation_with_prior_hold(tmp_path, monkeypatch):
    g, clock = guard(tmp_path / 'calls.db', budget='0.0004'); c = client(); g.attach(c)
    sent = replies(monkeypatch, c)
    with pytest.raises(MinimalCallStopped): call(g, c)
    state = g.snapshot()
    assert len(sent) == 1 and state['stop_reason'] == 'pre_send_budget_exceeded'
    assert state['unknown_cost_requests'] == 1 and Decimal(state['reserved_cost_usd']) > 0
    assert len(state['attempts']) == 1
    g.close()


def test_source_change_during_wait_stops_before_retry_and_pending_on_reopen(tmp_path, monkeypatch):
    path = tmp_path / 'calls.db'; valid = [True]
    g, clock = guard(path, check=lambda: valid[0]); c = client(); g.attach(c)
    sent = replies(monkeypatch, c)
    def sleep(seconds): clock[0] += seconds; valid[0] = False
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(sent) == 1
    g.close(); reopened, _ = guard(path)
    assert reopened.snapshot()['stop_reason'] == 'unresolved_product_retry_wait'
    reopened.close()


def test_source_rechecked_after_ordinary_pacing_before_claim(tmp_path, monkeypatch):
    valid = [True]; g, clock = guard(tmp_path / 'calls.db', minimum=120, check=lambda: valid[0])
    c = client(); g.attach(c); sent = stub(monkeypatch, receipt(c))
    assert call(g, c) == 'local fixture'
    def sleep(seconds): clock[0] += seconds; valid[0] = False
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c, 'next')
    assert len(sent) == 1 and len(g.snapshot()['attempts']) == 1
    g.close()


def test_wire_mutation_during_wait_is_rejected(tmp_path, monkeypatch):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c); held = []
    def fail(request, **_): held.append(request); raise http_failure(c, 429, cost=None)
    stub_model_http(monkeypatch, fail)
    def sleep(seconds): clock[0] += seconds; held[0].data += b' '
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert len(held) == 1 and g.snapshot()['stop_reason'] == 'local_request_rejected'
    g.close()


def test_wait_does_not_hold_global_lock_and_halt_is_responsive(tmp_path, monkeypatch):
    g, clock = guard(tmp_path / 'calls.db'); c = client(); g.attach(c)
    sent = replies(monkeypatch, c); observed = []; errors = []
    def inspect_and_stop():
        observed.append(g.snapshot()['calls_used'])
        g.halt()
    def sleep(seconds):
        observer = threading.Thread(target=inspect_and_stop)
        observer.start(); observer.join(timeout=2)
        assert not observer.is_alive(), 'global lock held through cooldown'
        clock[0] += seconds
    g._pacing_sleep = sleep
    with pytest.raises(MinimalCallStopped): call(g, c)
    assert observed == [1] and len(sent) == 1
    g.close()


def test_pending_wait_interruption_is_not_resumable_after_reopen(tmp_path, monkeypatch):
    path = tmp_path / 'calls.db'; g, _ = guard(path); c = client(); g.attach(c)
    sent = replies(monkeypatch, c)
    def interrupt(_): raise KeyboardInterrupt
    g._pacing_sleep = interrupt
    with pytest.raises((MinimalCallStopped, KeyboardInterrupt)): call(g, c)
    before = deepcopy(g.snapshot()['attempts']); g.close()
    reopened, _ = guard(path)
    assert reopened.snapshot()['attempts'] == before
    assert reopened.snapshot()['stop_reason'] == 'unresolved_product_retry_wait'
    assert not reopened.snapshot()['can_resume'] and len(sent) == 1
    reopened.close()


@pytest.mark.parametrize("role", ["safety_review", "reflection"])
def test_zero_retry_roles_keep_unknown429_hard_stopped(tmp_path, monkeypatch, role):
    from tests.test_candidate_role_transport import client as role_client, policy as role_policy, messages
    c = role_client(role)
    g = ProductCallGuard(tmp_path / 'calls.db', max_requests=40, max_observed_cost_usd='1',
        reservation_policy=role_policy(c), unknown_429_recovery=contract(),
        recovery_pre_send_check=lambda: True)
    g.attach(c); sent = replies(monkeypatch, c)
    with pytest.raises(MinimalCallStopped), g.scope('turn' if role == 'safety_review' else 'review', 'op', 'session'):
        c.complete(messages(c))
    state = g.snapshot()
    assert len(sent) == 1 and state['stopped'] and not state['can_resume']
    assert state['unknown_cost_requests'] == 1 and Decimal(state['reserved_cost_usd']) > 0
    assert not state['product_retry_waits']
    with pytest.raises(MinimalCallStopped):
        with g.scope('turn' if role == 'safety_review' else 'review', 'next', 'session'):
            c.complete(messages(c))
    assert len(sent) == 1
    g.close()
