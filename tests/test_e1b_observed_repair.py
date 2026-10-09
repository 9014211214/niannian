"""Formal-v6 cost/identity checks with synthetic parents and fake HTTP only."""
from tests._model_http_stub import stub_model_http
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import socket
import urllib.request

import pytest

from scripts import run_e1b as original
from scripts import run_e1b_reflection_repair as previous
from scripts import run_e1b_observed_repair as repair
from tests.test_e1b_repair import prior  # noqa: F401
from tests.test_e1b_reflection_repair import history  # noqa: F401
from tests.test_e1b_structure_repair import history3  # noqa: F401
from tests.test_e1b_pacing_repair import history4  # noqa: F401


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected real network"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("unexpected real network"))


@pytest.fixture(autouse=True)
def simulated_guard_clock(monkeypatch):
    from aria.engine.policy.minimal_calls import MinimalCallGuard
    initialize = MinimalCallGuard.__init__
    clock = [0.0]
    def fake_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda delay: clock.__setitem__(0, clock[0] + delay)
    monkeypatch.setattr(MinimalCallGuard, "__init__", fake_initialize)
    return clock


def tree(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": original.byte_sha(p)}
            for p in sorted((root / repair.PARENT).rglob("*")) if p.is_file()]


@pytest.fixture
def history5(history4, monkeypatch):
    root, policy = history4
    monkeypatch.setattr(repair, "CAMPAIGN", Path("v6"))
    monkeypatch.setattr(repair, "PARENT", Path("v5"))
    monkeypatch.setattr(repair, "ARCHIVE", Path("archive-v5"))
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", {"human_quote": "Synthetic budget decision only",
        "human_message_reference": "fictional test reference", "budget_choice": previous.BUDGET_CHOICE})
    monkeypatch.setattr(previous, "HUMAN_BUDGET_APPROVAL", dict(repair.HUMAN_BUDGET_APPROVAL))
    sources = []
    for index in range(121):
        relative = f"aria/source{index}.py"
        path = root / repair.ARCHIVE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic fifth-version source\n")
        sources.append({"path": relative, "sha256": original.byte_sha(path)})
    registration = {"source": sources, "metadata": {"policy": policy}, "stage": "prescreen", "fixture": True}
    registration["seal_sha256"] = original.digest(registration)
    monkeypatch.setattr(repair, "PARENT_SEAL", registration["seal_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/registration.json", registration)
    original.write_once(root / repair.ARCHIVE / "registration.json", registration)
    attempts = []
    for index in range(56):
        relative = Path("prescreen/wire-requests") / f"{index + 1:04d}.json"
        path = root / repair.PARENT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
        attempts.append({"request_id": str(index) if index < 55 else repair.PARENT_UNKNOWN_REQUEST, "status": "received" if index < 55 else "failed",
            "usage": {"cost": str(repair.PARENT_COST) if index == 0 else "0"} if index < 55 else {}, "e1b_stage": "prescreen",
            "role": "online_fast", "registration_sha256": registration["seal_sha256"],
            "wire_request_path": relative.as_posix(), "request_sha256": original.byte_sha(path), "http_status": 200 if index < 55 else None,
            "reservation": {"reserved_cost_usd": str(repair.PARENT_UNKNOWN_HOLD)}})
    attempts[-1].update(failure_phase="connect", failure_code="connection_failed", observation={})
    monkeypatch.setattr(repair, "PARENT_UNKNOWN_WIRE", attempts[-1]["request_sha256"])
    anchor = dict(repair.HUMAN_UNKNOWN_APPROVAL)
    anchor.update(campaign=repair.PARENT.as_posix(), request_sha256=repair.PARENT_UNKNOWN_WIRE,
        human_quote="Synthetic unknown-charge decision only", human_message_reference="fictional new exception")
    monkeypatch.setattr(repair, "HUMAN_UNKNOWN_APPROVAL", anchor)
    encoded_policy = json.dumps(policy, sort_keys=True, allow_nan=False)
    state = {"attempts": attempts, "ledger_id": "synthetic-fifth-ledger", "max_requests": 1636,
        "max_observed_cost_usd": "1.430191268", "calls_used": 56,
        "reservation_policy_sha256": hashlib.sha256(encoded_policy.encode()).hexdigest(),
        "unknown_cost_requests": 1, "stopped": True, "stop_reason": "model_or_storage_failed",
        "reserved_cost_usd": str(repair.PARENT_UNKNOWN_HOLD), "known_cost_usd": str(repair.PARENT_COST)}
    value = {"accounting": state, "registration_sha256": registration["seal_sha256"], "stage": "prescreen",
        "status": "not_evaluable", "old_unknown_requests_retained": 3, "historical_unknown_reserved_usd": str(repair.PRIOR_UNKNOWN_HOLD),
        "cumulative_prior_known_usd": "1.461468254"}
    value["receipt_sha256"] = original.digest(value)
    monkeypatch.setattr(repair, "PARENT_RECEIPT", value["receipt_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/execution-receipt.json", value)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts (sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?,?,?)", [(i, row["request_id"], json.dumps(row))
                                                              for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings (id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES (1,1,1636,'1.430191268','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy (id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES (1,?)", (encoded_policy,))
        db.execute("CREATE TABLE reservation_identity (id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES (1,'synthetic-fifth-ledger')")
        db.execute("CREATE TABLE request_pacing (id INTEGER,min_interval_seconds REAL)")
        db.execute("INSERT INTO request_pacing VALUES (1,10)")
    for index in range(37):
        original.write_once(root / repair.PARENT / f"fixture{index}.json", {"fictional": index})
    assert len(tree(root)) == 96
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    return root, policy


def prepared(history5, monkeypatch):
    root, policy = history5
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_observed_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    path = root / "freeze-v6.json"
    return entry, path, entry.prepare(root, root, path)


def authorize(root, entry, frozen):
    anchor = repair.HUMAN_BUDGET_APPROVAL
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=previous.BUDGET_CHOICE, user_limit_usd="1.5",
        cumulative_stop_usd="3", human_quote=anchor["human_quote"], human_message_reference=anchor["human_message_reference"],
        human_budget_approval_sha256=original.digest(anchor),
        unknown_human_quote=repair.HUMAN_UNKNOWN_APPROVAL["human_quote"],
        unknown_human_message_reference=repair.HUMAN_UNKNOWN_APPROVAL["human_message_reference"],
        human_unknown_approval_sha256=original.digest(repair.HUMAN_UNKNOWN_APPROVAL))
    path = entry.campaign(root) / "authorization.json"
    if path.exists():
        assert entry.read_json(path) == auth
    else:
        entry.write_once(path, auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": original.digest(auth)}
    entry.write_once(entry.campaign(root) / f"{frozen['stage']}.authorization.json", phase)
    return {"allow_local_env": True, "budget_choice": previous.BUDGET_CHOICE,
        "campaign_authorization_sha256": original.digest(auth), "stage_authorization_sha256": original.digest(phase)}


def test_five_campaigns_all_costs_holds_and_frozen_mode(history5, monkeypatch):
    root, _ = history5
    before = tree(root)
    parent = repair.parent_evidence(root)["fifth_campaign"]
    assert parent["calls"] == 56 and parent["archived_sources"] == 121
    assert parent["known_cost_usd"] == "0.00350335"
    assert parent["retained_unknown_reserve_usd"] == "0.001200844"
    entry, registration, frozen = prepared(history5, monkeypatch)
    assert entry.verify(root, root, registration) == frozen
    args = authorize(root, entry, frozen)
    _, limit = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert limit == Decimal("1.425487074")
    budget = frozen["protocol"]["budget"]
    assert budget["prior_e1b_known_usd"] == "0.072592432"
    assert budget["prior_known_floor_usd"] == "1.464971604"
    assert budget["accepted_old_unknown"] == 4
    assert budget["historical_unknown_reserved_usd"] == "0.001920494"
    assert frozen["protocol"]["planned_total_calls"] == 1636
    assert frozen["protocol"]["transport"]["min_interval_seconds"] == 10
    assert frozen["protocol"]["transport"]["observe_transport"] is True
    assert tree(root) == before


@pytest.mark.parametrize("part", ["wire", "archive", "journal", "v4archive"])
def test_changed_parent_or_archive_stops_before_new_namespace(history5, part):
    from scripts import run_e1b_pacing_repair as v5
    root, _ = history5
    path = {"wire": repair.PARENT / "prescreen/wire-requests/0056.json",
        "archive": repair.ARCHIVE / "aria/source1.py", "journal": repair.PARENT / "allowance.sqlite3-wal",
        "v4archive": v5.ARCHIVE / "aria/source1.py"}[part]
    (root / path).write_bytes(b"synthetic mutation")
    with pytest.raises(repair.RunError):
        repair.build_engine(root)
    assert not (root / repair.CAMPAIGN).exists()


@pytest.mark.parametrize("field,value", [("accepted_old_unknown", 3), ("historical_unknown_reserved_usd", "0.00071965"),
    ("effective_new_allowance_usd", "1.427407568"), ("unknown_human_quote", "old third exception"),
    ("human_unknown_approval_sha256", "b" * 64), ("prior_known_usd", "1.461468254")])
def test_old_exception_and_reset_hold_rejected_before_key(history5, monkeypatch, field, value):
    root, _ = history5
    entry, registration, frozen = prepared(history5, monkeypatch)
    args = authorize(root, entry, frozen)
    path = entry.campaign(root) / "authorization.json"
    auth = entry.read_json(path)
    auth[field] = value
    path.write_text(json.dumps(auth), encoding="utf-8")
    args["campaign_authorization_sha256"] = original.digest(auth)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("must reject before key"))
    with pytest.raises(repair.RunError, match="explicitly_authorized"):
        entry.execute(root, root, registration, **args)


@pytest.mark.parametrize("point", ["before_preparation", "tls", "partial_write", "headers", "missing_success_events"])
def test_real_entry_observation_holds_unknown_and_never_retries(history5, monkeypatch, point):
    from aria.engine.policy import minimal_calls
    from tests.test_e1b_execution import fake_http
    root, _ = history5
    entry, registration, frozen = prepared(history5, monkeypatch)
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    fake_http(monkeypatch)
    calls = []
    def observed(request, *, timeout, trace):
        calls.append(request.data)
        events = {"before_preparation": [], "tls": ["connection_preparation_started"],
            "partial_write": ["connection_preparation_started", "connection_preparation_completed", "tls_completed", "application_write_attempted"],
            "headers": ["connection_preparation_started", "connection_preparation_completed", "tls_completed", "application_write_attempted", "request_write_completed"],
            "missing_success_events": []}[point]
        for event in events:
            trace._mark(event)
        if point == "missing_success_events":
            return urllib.request.urlopen(request, timeout=timeout)
        raise TimeoutError("synthetic private text must not persist")
    monkeypatch.setattr(minimal_calls, "open_observed", observed)
    result = entry.execute(root, root, registration, **args)
    receipt = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")
    state = receipt["accounting"]
    assert result["status"] == "not_evaluable" and len(calls) == 1
    assert state["calls_used"] == 1 and state["stopped"]
    assert receipt["old_unknown_requests_retained"] == 4
    assert receipt["historical_unknown_reserved_usd"] == "0.001920494"
    row = state["attempts"][0]
    assert "synthetic private" not in json.dumps(row)
    assert row["pacing"]["min_interval_seconds"] == 10
    if point != "missing_success_events":
        assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        repair.validate_transport_records([row], entry.protocol())
    else:
        with pytest.raises(repair.RunError, match="transport_record"):
            repair.validate_transport_records([row], entry.protocol())
    with sqlite3.connect(entry.campaign(root) / "allowance.sqlite3") as db:
        assert db.execute("SELECT version FROM transport_observation WHERE id=1").fetchone() == ("connection_trace_v1",)
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", [None, "table", "missing_table", "protocol", "missing_event", "false_event", "extra_event"])
def test_readonly_assessment_requires_same_observation_contract(tmp_path, monkeypatch, damage):
    from aria.engine.policy.observed_transport import TransportTrace
    from scripts import e1b_assessment as assessment
    from tests.test_e1b_assessment import ledger_fixture
    monkeypatch.setattr(assessment.execution, "OBSERVE_TRANSPORT", True, raising=False)
    monkeypatch.setattr(assessment.execution, "validate_transport_records", repair.validate_transport_records, raising=False)
    path, state, frozen = ledger_fixture(tmp_path)
    frozen["protocol"] = {"transport": repair.transport_contract()}
    if damage == "protocol":
        frozen["protocol"]["transport"]["observe_transport"] = False
    for row in state["attempts"]:
        trace = TransportTrace()
        for event in ("connection_preparation_started", "connection_preparation_completed", "tls_completed", "application_write_attempted", "request_write_completed", "response_headers_returned"):
            trace._mark(event)
        row["transport_events"] = trace.snapshot()
    event = state["attempts"][0]["transport_events"]["events"]
    if damage == "missing_event":
        event.pop("tls_completed")
    if damage == "false_event":
        event["tls_completed"] = False
    if damage == "extra_event":
        event["synthetic_private_url"] = True
    with sqlite3.connect(path) as db:
        if damage != "missing_table":
            db.execute("CREATE TABLE transport_observation(id INTEGER PRIMARY KEY,version TEXT)")
            db.execute("INSERT INTO transport_observation VALUES(1,?)", ("other" if damage == "table" else "connection_trace_v1",))
        for index, row in enumerate(state["attempts"]):
            db.execute("UPDATE attempts SET record=? WHERE sequence=?", (json.dumps(row), index))
    before = path.read_bytes()
    if damage:
        with pytest.raises((assessment.AssessmentError, repair.RunError), match="transport"):
            assessment._ledger(tmp_path, state, frozen)
    else:
        assert assessment._ledger(tmp_path, state, frozen) == original.byte_sha(path)
    assert path.read_bytes() == before


