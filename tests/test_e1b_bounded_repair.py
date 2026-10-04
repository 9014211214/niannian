"""V7 entry integration: synthetic history, fake transport, no credentials."""
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import urllib.error
import urllib.request

import pytest

from scripts import run_e1b as original
from scripts import run_e1b_bounded_repair as repair
from scripts import run_e1b_observed_repair as observed
from tests.test_e1b_observed_repair import (  # noqa: F401
    prior, history, history3, history4, history5, offline, simulated_guard_clock,
)

EVENTS = ("connection_preparation_started", "connection_preparation_completed", "tls_completed",
          "application_write_attempted", "request_write_completed", "response_headers_returned")


@pytest.fixture
def history6(history5, monkeypatch):
    root, policy = history5
    for name, value in {"CAMPAIGN": Path("v7"), "PARENT": Path("v6"), "ARCHIVE": Path("archive-v6")}.items():
        monkeypatch.setattr(repair, name, value)
    sources = []
    for index in range(124):
        relative = f"aria/source{index}.py"
        path = root / repair.ARCHIVE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic sixth-version source\n")
        sources.append({"path": relative, "sha256": original.byte_sha(path)})
    registration = {"source": sources, "metadata": {"policy": policy}, "stage": "prescreen", "fixture": True,
                    "protocol": {"transport": observed.transport_contract()}}
    registration["seal_sha256"] = original.digest(registration)
    monkeypatch.setattr(repair, "PARENT_SEAL", registration["seal_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/registration.json", registration)
    original.write_once(root / repair.ARCHIVE / "registration.json", registration)
    attempts = []
    for index in range(200):
        relative = Path("prescreen/wire-requests") / f"{index + 1:04d}.json"
        path = root / repair.PARENT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
        attempts.append({"request_id": str(index) if index < 199 else repair.PARENT_UNKNOWN_REQUEST,
            "status": "received" if index < 199 else "failed",
            "usage": {"cost": str(repair.PARENT_COST) if index == 0 else "0"} if index < 199 else {},
            "e1b_stage": "prescreen", "role": "online_fast", "registration_sha256": registration["seal_sha256"],
            "wire_request_path": relative.as_posix(), "request_sha256": original.byte_sha(path),
            "http_status": 200 if index < 199 else 429,
            "reservation": {"reserved_cost_usd": str(repair.PARENT_UNKNOWN_HOLD)},
            "transport_events": {"version": "connection_trace_v1", "events": dict.fromkeys(EVENTS, True),
                "absent_events": "not_observed", "scope": "local_events_not_delivery_or_billing"}})
    attempts[-1].update(failure_phase="http_status", failure_code="http_error", observation={"provider_reported_name": "Alibaba"})
    monkeypatch.setattr(repair, "PARENT_UNKNOWN_WIRE", attempts[-1]["request_sha256"])
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", dict(observed.HUMAN_BUDGET_APPROVAL))
    anchor = {**repair.HUMAN_UNKNOWN_APPROVAL, "campaign": repair.PARENT.as_posix(),
              "request_sha256": repair.PARENT_UNKNOWN_WIRE, "human_quote": "Synthetic fifth unknown only"}
    monkeypatch.setattr(repair, "HUMAN_UNKNOWN_APPROVAL", anchor)
    monkeypatch.setattr(repair, "HUMAN_RECOVERY_APPROVAL", {**repair.HUMAN_RECOVERY_APPROVAL,
        "campaign": repair.CAMPAIGN.as_posix(), "human_quote": "Synthetic bounded recovery only"})
    encoded = json.dumps(policy, sort_keys=True, allow_nan=False)
    state = {"attempts": attempts, "ledger_id": "synthetic-sixth-ledger", "max_requests": 1636,
        "max_observed_cost_usd": "1.425487074", "calls_used": 200,
        "reservation_policy_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "unknown_cost_requests": 1, "stopped": True, "stop_reason": "model_or_storage_failed",
        "reserved_cost_usd": str(repair.PARENT_UNKNOWN_HOLD), "known_cost_usd": str(repair.PARENT_COST)}
    receipt = {"accounting": state, "registration_sha256": registration["seal_sha256"], "stage": "prescreen",
        "status": "not_evaluable", "old_unknown_requests_retained": 4,
        "historical_unknown_reserved_usd": str(repair.PRIOR_UNKNOWN_HOLD), "cumulative_prior_known_usd": "1.464971604"}
    receipt["receipt_sha256"] = original.digest(receipt)
    monkeypatch.setattr(repair, "PARENT_RECEIPT", receipt["receipt_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/execution-receipt.json", receipt)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts(sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES(?,?,?)", [(i, row["request_id"], json.dumps(row)) for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings(id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES(1,1,1636,'1.425487074','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy(id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES(1,?)", (encoded,))
        db.execute("CREATE TABLE reservation_identity(id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES(1,'synthetic-sixth-ledger')")
        db.execute("CREATE TABLE request_pacing(id INTEGER,min_interval_seconds REAL)")
        db.execute("INSERT INTO request_pacing VALUES(1,10)")
        db.execute("CREATE TABLE transport_observation(id INTEGER,version TEXT)")
        db.execute("INSERT INTO transport_observation VALUES(1,'connection_trace_v1')")
    for index in range(112):
        original.write_once(root / repair.PARENT / f"fixture{index}.json", {"fictional": index})
    files = [{"path": p.relative_to(root).as_posix(), "sha256": original.byte_sha(p)}
             for p in sorted((root / repair.PARENT).rglob("*")) if p.is_file()]
    assert len(files) == 315
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(files))
    return root, policy


def prepared(history6, monkeypatch):
    root, policy = history6
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_bounded_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    path = root / "freeze-v7.json"
    return entry, path, entry.prepare(root, root, path)


def authorize(root, entry, frozen):
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=repair.BUDGET_CHOICE, user_limit_usd="1.5", cumulative_stop_usd="3")
    for prefix, anchor in (("", repair.HUMAN_BUDGET_APPROVAL), ("unknown_", repair.HUMAN_UNKNOWN_APPROVAL),
                           ("recovery_", repair.HUMAN_RECOVERY_APPROVAL)):
        auth[prefix + "human_quote"] = anchor["human_quote"]
        auth[prefix + "human_message_reference"] = anchor["human_message_reference"]
        auth["human_" + (prefix or "budget_") + "approval_sha256"] = original.digest(anchor)
    path = entry.campaign(root) / "authorization.json"
    if path.exists():
        assert entry.read_json(path) == auth
    else:
        entry.write_once(path, auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": original.digest(auth)}
    entry.write_once(entry.campaign(root) / f"{frozen['stage']}.authorization.json", phase)
    return {"allow_local_env": True, "budget_choice": repair.BUDGET_CHOICE,
        "campaign_authorization_sha256": original.digest(auth), "stage_authorization_sha256": original.digest(phase)}


def test_six_histories_budget_and_explicit_recovery_bound(history6, monkeypatch):
    root, _ = history6
    assert repair.parent_evidence(root)["sixth_campaign"]["calls"] == 200
    entry, registration, frozen = prepared(history6, monkeypatch)
    assert entry.verify(root, root, registration) == frozen
    args = authorize(root, entry, frozen)
    _, allowance = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert allowance == Decimal("1.410501882")
    assert frozen["protocol"]["planned_total_calls"] == 1636
    assert frozen["protocol"]["max_wire_requests"] == 1638
    assert frozen["protocol"]["budget"]["accepted_old_unknown"] == 5
    assert frozen["protocol"]["budget"]["historical_unknown_reserved_usd"] == "0.002638320"
    assert frozen["protocol"]["bounded_recovery"] == repair.recovery_policy()


@pytest.mark.parametrize("field", ["recovery_human_quote", "human_recovery_approval_sha256", "accepted_old_unknown"])
def test_missing_or_changed_specific_authorization_precedes_key(history6, monkeypatch, field):
    root, _ = history6
    entry, registration, frozen = prepared(history6, monkeypatch)
    args = authorize(root, entry, frozen)
    path = entry.campaign(root) / "authorization.json"
    auth = entry.read_json(path)
    auth[field] = None
    path.write_text(json.dumps(auth), encoding="utf-8")
    args["campaign_authorization_sha256"] = original.digest(auth)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("rejection must precede credential"))
    with pytest.raises(original.RunError, match="explicitly_authorized"):
        entry.execute(root, root, registration, **args)


def test_entry_one_recovery_keeps_failed_wire_and_first_actual_answer(history6, monkeypatch):
    from aria.engine.policy import minimal_calls
    from tests.test_e1b_execution import fake_http
    root, _ = history6
    checked = repair.parent_evidence(root)
    monkeypatch.setattr(repair, "parent_evidence", lambda _: checked)
    entry, registration, frozen = prepared(history6, monkeypatch)
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    fake_http(monkeypatch)
    calls = []
    def transport(request, *, timeout, trace):
        calls.append(request.data)
        for event in EVENTS:
            trace._mark(event)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "", {}, io.BytesIO(json.dumps(
                {"error": {"code": 429, "message": "synthetic", "metadata": {"provider_name": "Alibaba"}}}).encode()))
        return urllib.request.urlopen(request, timeout=timeout)
    monkeypatch.setattr(minimal_calls, "open_observed", transport)
    real_case = entry.E1bRunner.run_case
    def single_case(self, *a, **kw):
        real_case(self, *a, **kw)
        raise RuntimeError("synthetic stop after one complete case")
    monkeypatch.setattr(entry.E1bRunner, "run_case", single_case)
    result = entry.execute(root, root, registration, **args)
    state = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")["accounting"]
    assert result["status"] == "not_evaluable" and len(calls) == 31
    assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
    assert state["recovery"]["logical_received"] == 30 and state["recovery"]["extra_dispatches"] == 1
    assert state["attempts"][0]["status"] == "failed" and state["attempts"][1]["physical_attempt"] == 2
    assert calls[0] == calls[1] and len(entry.logical_attempts(state)) == 30
