"""Fixed residual allowance: no reset of the original activity's counters."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from tests.test_bounded_http_recovery import scripted, checked, no_network
from tests.test_pre_application_rebuild import continuation_policy
from tests.test_reserved_calls import policy, client, call


def remaining_guard(path, clock):
    g = ReservedCallGuard(path, max_requests=1247, max_observed_cost_usd="1", reservation_policy=policy(),
        min_interval_seconds=10, observe_transport=True, recovery_policy=continuation_policy())
    g._pacing_clock = g._recovery_clock = lambda: clock[0]
    g._pacing_sleep = lambda delay: clock.__setitem__(0, clock[0] + delay)
    return g


def test_remaining_extra_uses120_and_stays_consumed_after_role_and_reopen(tmp_path, monkeypatch):
    clock = [0.0]; path = tmp_path / "residual.sqlite3"
    g = remaining_guard(path, clock); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["known429", "ok", "known429", "ok"], clock)
    assert call(g, c) == "local fixture"
    state, summary = checked(g, True)
    assert summary["extra_dispatches"] == 1 and state["attempts"][-1]["recovery_wait_seconds"] == 120
    assert state["attempts"][-1]["recovery_wait_ms"] == 120000
    g.close()
    g = remaining_guard(path, clock); c = client("memory"); g.attach(c)
    with pytest.raises(MinimalCallStopped):
        call(g, c, "next-stage")
    assert len(sent) == 3 and len(g.snapshot()["recovery"]["relations"]) == 1
    g.close()


def test_final_remaining_logical_position_recovers_but_new_position1247_cannot_exist(tmp_path, monkeypatch):
    clock = [0.0]
    g = remaining_guard(tmp_path / "residual.sqlite3", clock); c = client(); g.attach(c)
    sent = scripted(monkeypatch, ["ok", "known429", "ok"], clock)
    assert call(g, c) == "local fixture"
    template = g.snapshot()["attempts"][0]
    with g._transaction():
        for number in range(2, 1246):
            row = deepcopy(template)
            row.update(request_id=f"fixture{number}", operation_id=f"op{number}", logical_operation_id=f"op{number}")
            g._db.execute("INSERT INTO attempts(request_id,record) VALUES(?,?)", (row["request_id"], json.dumps(row)))
    assert call(g, c, "final-position") == "local fixture"
    state, summary = checked(g, True)
    assert summary["logical_received"] == summary["logical_calls_used"] == 1246
    assert state["calls_used"] == 1247 and summary["extra_dispatches"] == 1
    assert state["stop_reason"] == "logical_request_limit_reached"
    with pytest.raises(MinimalCallStopped):
        call(g, c, "extra-position")
    assert len(sent) == 3
    g.close()
