"""Prior-run qualification with authored records, never private experiment data."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import run_ctest_repair as run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    base = root / run.BASE
    live = base / "ctest-live-v1"
    experiment = live / "synthetic-cognitive-experiments/ctest-v1-20261003"
    archived = base / "ctest-frozen-v1-source"
    experiment.mkdir(parents=True)
    sources = []
    for name in run.previous.FILES:
        raw = ("authored old source: " + name + "\n").encode()
        for path in (root / name, archived / name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        sources.append({"path": name, "bytes": len(raw), "sha256": run.previous.digest(raw)})
    for name in run.EXTRA:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("authored repair source\n", encoding="utf-8")
    case_manifest = {"source_sha256": "authored-script-v1"}
    monkeypatch.setattr(run.previous, "protocol", lambda: {"case_manifest": case_manifest})
    policy = {"schema": "authored-policy-fixture"}
    registration = {
        "protocol": {"case_manifest": case_manifest,
                     "budget": {"effective_new_allowance_usd": str(run.previous.LIMIT)}},
        "source": sources, "metadata": {"policy": policy},
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    seal = run.previous.digest(run.previous.canonical(registration))
    registration["seal_sha256"] = seal
    monkeypatch.setattr(run, "PARENT_SEAL", seal)
    write_json(base / "ctest-registration-v2.json", registration)
    write_json(archived / "registration.json", registration)
    attempts, operations = [], []
    for index in range(40):
        case, visit = index // 8 + 1, index % 8 + 1
        session_id = f"c{case:02d}-s{visit}"
        cost = Decimal("0.01") if index < 39 else Decimal("0.010796")
        attempts.append({"request_id": f"authored-request-{index:02d}", "kind": "review",
                         "operation_id": "ctest-" + session_id, "session_id": session_id,
                         "role": "memory", "model": "anthropic/claude-sonnet-5",
                         "status": "received", "usage": {"cost": str(cost)}})
        operation = {"case_id": f"c{case:02d}", "session": visit, "session_id": session_id}
        operations.append({**operation, "event": "pending"})
        if index < 39:
            operations.append({**operation, "event": "completed"})
    operations.append({"event": "stopped", "error_type": "CTestRunStopped", "real_model_gate": "not_evaluable"})
    journal = experiment / "operations.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in operations), encoding="utf-8")
    response = experiment / "c05-s8-response.json"
    write_json(response, {"status": "cognitive_protocol_missing_or_invalid",
                          "memory_json": {"episodes": [], "relationships": [], "stated_preferences": []}})
    policy_json = json.dumps(policy, sort_keys=True, allow_nan=False)
    ledger_id = "a" * 32
    accounting = {"configured": True, "calls_used": 40, "max_requests": 96,
                  "known_cost_usd": str(run.PARENT_COST), "unknown_cost_requests": 0,
                  "stopped": True, "stop_reason": "model_or_storage_failed",
                  "max_observed_cost_usd": str(run.previous.LIMIT), "attempts": attempts,
                  "reserved_cost_usd": "0", "ledger_id": ledger_id,
                  "reservation_policy_sha256": run.previous.digest(policy_json.encode())}
    receipt = live / "execution-receipt.json"
    write_json(receipt, {"seal_sha256": seal, "real_model_gate": "not_evaluable",
                         "failure_type": "CTestRunStopped", "accounting": accounting,
                         "old_unknown_requests_retained": 2})
    ledger = live / "allowance.sqlite3"
    with sqlite3.connect(ledger) as db:
        db.executescript("""
            CREATE TABLE settings (id INTEGER PRIMARY KEY,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT);
            CREATE TABLE attempts (sequence INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT,record TEXT);
            CREATE TABLE reservation_policy (id INTEGER PRIMARY KEY,policy TEXT);
            CREATE TABLE reservation_identity (id INTEGER PRIMARY KEY,ledger_id TEXT);
        """)
        db.execute("INSERT INTO settings VALUES(1,1,96,?,?)", (str(run.previous.LIMIT), "model_or_storage_failed"))
        db.executemany("INSERT INTO attempts(request_id,record) VALUES(?,?)",
                       [(row["request_id"], json.dumps(row)) for row in attempts])
        db.execute("INSERT INTO reservation_policy VALUES(1,?)", (policy_json,))
        db.execute("INSERT INTO reservation_identity VALUES(1,?)", (ledger_id,))
    return {"root": root, "base": base, "archived": archived, "ledger": ledger,
            "receipt": receipt, "journal": journal, "operations": operations, "response": response}


def test_authored_parent_qualifies_without_reading_key_or_mutating_evidence(history, monkeypatch):
    monkeypatch.setattr(run.previous, "local_key", lambda _: pytest.fail("history cannot read a key"))
    originals = {path: path.read_bytes() for path in history["root"].rglob("*") if path.is_file()}
    result = run.parent_evidence(history["root"])
    assert result["calls"] == 40 and result["valid_sessions"] == 39 and result["unissued_sessions"] == 56
    assert result["known_cost_usd"] == "0.400796" and result["new_unknown_requests"] == 0
    assert result["prior_seal_recomputed"] and result["ledger_receipt_consistent"]
    assert originals == {path: path.read_bytes() for path in history["root"].rglob("*") if path.is_file()}


@pytest.mark.parametrize("part", ["protocol", "source", "metadata"])
def test_old_seal_constant_does_not_hide_changed_registration_body(history, part):
    path = history["base"] / "ctest-registration-v2.json"
    value = json.loads(path.read_bytes())
    value[part] = [] if part == "source" else {"changed": True}
    write_json(path, value)
    with pytest.raises(run.RunError, match="prior_ctest_seal_changed"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("part", ["attempt", "request_id", "policy", "ledger_id", "threshold", "halted"])
def test_sqlite_changes_cannot_be_hidden_by_an_unchanged_receipt(history, part):
    with sqlite3.connect(history["ledger"]) as db:
        if part == "attempt":
            row = json.loads(db.execute("SELECT record FROM attempts WHERE sequence=1").fetchone()[0])
            row["usage"]["cost"] = "0.02"
            db.execute("UPDATE attempts SET record=? WHERE sequence=1", (json.dumps(row),))
        elif part == "request_id":
            db.execute("UPDATE attempts SET request_id='changed' WHERE sequence=1")
        elif part == "policy":
            db.execute("UPDATE reservation_policy SET policy='{}'")
        elif part == "ledger_id":
            db.execute("UPDATE reservation_identity SET ledger_id='changed'")
        elif part == "threshold":
            db.execute("UPDATE settings SET threshold='2'")
        else:
            db.execute("UPDATE settings SET halted=NULL")
    with pytest.raises(run.RunError, match="prior_ctest_ledger_receipt_mismatch"):
        run.parent_evidence(history["root"])


def test_receipt_changes_cannot_hide_unchanged_sqlite(history):
    value = json.loads(history["receipt"].read_bytes())
    value["accounting"]["attempts"][0]["request_id"] = "invented-request"
    write_json(history["receipt"], value)
    with pytest.raises(run.RunError, match="prior_ctest_ledger_receipt_mismatch"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("part", ["completed_missing", "completed_duplicate", "pending_missing", "pending_reordered", "stopped_missing"])
def test_original_39_of_40_event_sequence_is_required(history, part):
    rows = list(history["operations"])
    if part == "completed_missing":
        rows.pop(1)
    elif part == "completed_duplicate":
        rows.insert(2, rows[1])
    elif part == "pending_missing":
        rows.pop(0)
    elif part == "pending_reordered":
        rows[0], rows[2] = rows[2], rows[0]
    else:
        rows.pop()
    history["journal"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(run.RunError, match="prior_ctest_failure_changed"):
        run.parent_evidence(history["root"])


def test_archived_failure_response_must_still_lack_fourth_field(history):
    value = json.loads(history["response"].read_bytes())
    value["memory_json"]["cognitive_observations"] = []
    write_json(history["response"], value)
    with pytest.raises(run.RunError, match="prior_ctest_failure_changed"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("name", run.previous.FILES)
def test_repair_cannot_rewrite_original_entry_or_protocol(history, name):
    (history["root"] / name).write_text("changed old execution source", encoding="utf-8")
    with pytest.raises(run.RunError, match="prior_execution_source_changed"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("part", ["source", "registration"])
def test_archived_frozen_bytes_are_checked(history, part):
    path = history["archived"] / (run.previous.FILES[0] if part == "source" else "registration.json")
    path.write_text("changed archived bytes", encoding="utf-8")
    with pytest.raises(run.RunError, match="prior_frozen_source_missing_or_changed|prior_archived_registration_changed"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_unresolved_sqlite_sidecars_block_history_qualification(history, suffix):
    Path(str(history["ledger"]) + suffix).write_bytes(b"authored pending journal")
    with pytest.raises(run.RunError, match="prior_ctest_ledger_not_closed"):
        run.parent_evidence(history["root"])


def test_changed_case_script_manifest_is_not_a_repair(history, monkeypatch):
    monkeypatch.setattr(run.previous, "protocol", lambda: {"case_manifest": {"changed": True}})
    with pytest.raises(run.RunError, match="repair_must_not_change_test_scripts"):
        run.parent_evidence(history["root"])


@pytest.mark.parametrize("name", run.EXTRA)
def test_extra_repair_source_links_are_rejected(history, name):
    path = history["root"] / name
    target = history["root"] / "authored-link-target.txt"
    target.write_text("authored source", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("current Windows account cannot create test symlinks")
    with pytest.raises(run.RunError, match="evidence_or_source_link_rejected"):
        run.inventory(history["root"])
