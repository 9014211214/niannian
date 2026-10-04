"""Prospective C-test after the missing cognitive-field protocol repair.

The failed first version is immutable, fully charged, and never resumed. This
entry is one new complete 96-session measurement inside the same user budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ctest as previous
from aria.bench.ctest_runner import CTestRunner
from aria.engine.policy.reserved_calls import ReservedCallGuard

BASE = Path("artifacts/local/tasks/20261003-02")
PARENT_SEAL = "61cbf5bbbf04711d5db962741d556b34e584e5ccf09fc0dd3635a25d92e6da49"
PARENT_COST = Decimal("0.400796")
LIMIT = previous.LIMIT - PARENT_COST
EXPERIMENT = "ctest-v2-20261003"
MARKER = BASE / "ctest-repair-v2-authorization.started.json"
EXTRA = ("scripts/run_ctest_repair.py", "docs/CTEST_REPAIR.md")
RunError = previous.RunError


def read_original(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise RunError("evidence_or_source_link_rejected")
    return path.read_bytes()


def check_parent_ledger(path, registration, accounting):
    # A stopped, closed ledger must have no unresolved journal. Immutable mode
    # prevents inspection from changing the preserved database or creating WAL.
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("prior_ctest_ledger_not_closed")
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
            records = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
            policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
            identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
    except sqlite3.DatabaseError:
        raise RunError("prior_ctest_ledger_unreadable") from None
    attempts = [json.loads(raw) for _, raw in records]
    expected_policy = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    expected_threshold = registration["protocol"]["budget"]["effective_new_allowance_usd"]
    if (settings != (1, 96, expected_threshold, "model_or_storage_failed")
            or Decimal(expected_threshold) != previous.LIMIT
            or attempts != accounting["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(records, attempts))
            or policy != (expected_policy,)
            or identity != (accounting["ledger_id"],)
            or accounting["reservation_policy_sha256"] != previous.digest(expected_policy.encode())
            or accounting["max_requests"] != 96
            or accounting["max_observed_cost_usd"] != expected_threshold
            or accounting["stopped"] is not True
            or accounting["stop_reason"] != "model_or_storage_failed"
            or Decimal(accounting["reserved_cost_usd"]) != 0):
        raise RunError("prior_ctest_ledger_receipt_mismatch")


def parent_evidence(root):
    paths = [BASE / "ctest-registration-v2.json", BASE / "ctest-live-v1/execution-receipt.json",
             BASE / "ctest-live-v1/allowance.sqlite3",
             BASE / "ctest-live-v1/synthetic-cognitive-experiments/ctest-v1-20261003/operations.jsonl",
             BASE / "ctest-live-v1/synthetic-cognitive-experiments/ctest-v1-20261003/c05-s8-response.json"]
    originals = [read_original(root, name) for name in paths]
    registration, receipt = (json.loads(raw) for raw in originals[:2])
    unsigned = {key: value for key, value in registration.items() if key != "seal_sha256"}
    if (set(registration) != {"protocol", "source", "metadata", "frozen_at", "seal_sha256"}
            or previous.digest(previous.canonical(unsigned)) != PARENT_SEAL
            or registration["seal_sha256"] != PARENT_SEAL):
        raise RunError("prior_ctest_seal_changed")
    accounting = receipt["accounting"]
    if (registration["seal_sha256"] != PARENT_SEAL or receipt["seal_sha256"] != PARENT_SEAL
            or receipt["real_model_gate"] != "not_evaluable" or accounting["calls_used"] != 40
            or accounting["unknown_cost_requests"] != 0 or Decimal(accounting["known_cost_usd"]) != PARENT_COST
            or len(accounting["attempts"]) != 40
            or any(row["status"] != "received" or "cost" not in row["usage"] for row in accounting["attempts"])
            or sum((Decimal(str(row["usage"]["cost"])) for row in accounting["attempts"]), Decimal(0)) != PARENT_COST):
        raise RunError("prior_ctest_accounting_changed")
    check_parent_ledger(root / paths[2], registration, accounting)
    if read_original(root, paths[2]) != originals[2]:
        raise RunError("prior_ctest_ledger_changed_during_read")
    operations = [json.loads(line) for line in originals[3].decode().splitlines()]
    pending = [row["session_id"] for row in operations if row["event"] == "pending"]
    completed = [row["session_id"] for row in operations if row["event"] == "completed"]
    expected_sessions = [f"c{case:02d}-s{session}" for case in range(1, 6) for session in range(1, 9)]
    response = json.loads(originals[4])
    if (pending != expected_sessions or completed != expected_sessions[:-1]
            or [row["session_id"] for row in accounting["attempts"]] != expected_sessions
            or [row["operation_id"] for row in accounting["attempts"]] != ["ctest-" + sid for sid in expected_sessions]
            or operations[-1] != {"event": "stopped", "error_type": "CTestRunStopped", "real_model_gate": "not_evaluable"}
            or response["status"] != "cognitive_protocol_missing_or_invalid"
            or type(response["memory_json"]) is not dict
            or "cognitive_observations" in response["memory_json"]):
        raise RunError("prior_ctest_failure_changed")
    archived_registration = BASE / "ctest-frozen-v1-source/registration.json"
    if read_original(root, archived_registration) != originals[0]:
        raise RunError("prior_archived_registration_changed")
    for row in registration["source"]:
        archived = BASE / "ctest-frozen-v1-source" / row["path"]
        if previous.digest(read_original(root, archived)) != row["sha256"]:
            raise RunError("prior_frozen_source_missing_or_changed")
        if row["path"] in previous.FILES and previous.digest(read_original(root, row["path"])) != row["sha256"]:
            raise RunError("prior_execution_source_changed")
    if previous.canonical(registration["protocol"]["case_manifest"]) != previous.canonical(previous.protocol()["case_manifest"]):
        raise RunError("repair_must_not_change_test_scripts")
    return {"seal_sha256": PARENT_SEAL, "calls": 40, "valid_sessions": 39,
            "known_cost_usd": str(PARENT_COST), "new_unknown_requests": 0,
            "outcome": "not_evaluable", "unissued_sessions": 56,
            "prior_seal_recomputed": True, "ledger_receipt_consistent": True,
            "archived_source_files_checked": len(registration["source"]),
            "files": [{"path": name.as_posix(), "sha256": previous.digest(raw)} for name, raw in zip(paths, originals)]}


def inventory(root):
    rows = previous.inventory(root)
    for name in EXTRA:
        raw = read_original(root, name)
        rows.append({"path": name, "bytes": len(raw), "sha256": previous.digest(raw)})
    return sorted(rows, key=lambda row: row["path"])


def protocol():
    value = previous.protocol()
    value.update(experiment=EXPERIMENT, schema="aria_ctest_repair_execution_v1",
                 repair="C1-on unified four-array example plus required output validation; C1-off bytes unchanged",
                 measurement="new complete 96; do not reuse 39 prior valid sessions or select better outputs")
    value["budget"].update(prior_ctest_known_usd=str(PARENT_COST), effective_new_allowance_usd=str(LIMIT),
                           ctest_total_user_limit_usd="2")
    return value


def prepare(root, metadata, destination):
    value = {"protocol": protocol(), "source": inventory(root), "parent": parent_evidence(root),
             "metadata": previous.metadata_evidence(metadata), "frozen_at": datetime.now(timezone.utc).isoformat()}
    value["seal_sha256"] = previous.digest(previous.canonical(value))
    previous.write_new(destination, value)
    return value


def verify(root, metadata, registration):
    value = json.loads(registration.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    if (set(value) != {"protocol", "source", "parent", "metadata", "frozen_at", "seal_sha256"}
            or value["seal_sha256"] != previous.digest(previous.canonical(unsigned))):
        raise RunError("repair_seal_changed")
    if value["protocol"] != protocol() or value["source"] != inventory(root):
        raise RunError("repair_source_or_protocol_changed")
    if value["parent"] != parent_evidence(root) or value["metadata"] != previous.metadata_evidence(metadata):
        raise RunError("repair_parent_or_metadata_changed")
    stamp = datetime.fromisoformat(value["frozen_at"])
    if stamp.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 86400:
        raise RunError("repair_registration_expired")
    return value


def execute(root, metadata, registration, output, *, allow_local_env):
    if allow_local_env is not True:
        raise RunError("explicit_local_key_permission_required")
    frozen = verify(root, metadata, registration)
    marker = root.resolve() / MARKER
    if not marker.resolve().is_relative_to(root.resolve()):
        raise RunError("authorization_marker_outside_workspace")
    marker.parent.mkdir(parents=True, exist_ok=True)
    previous.write_new(marker, {"seal_sha256": frozen["seal_sha256"], "experiment": EXPERIMENT,
                        "output": str(output.resolve()), "automatic_resume": False})
    output.mkdir(parents=True, exist_ok=False)
    previous.write_new(output / "STARTED.json", {"seal_sha256": frozen["seal_sha256"],
                        "started_at": datetime.now(timezone.utc).isoformat()})
    guard = ReservedCallGuard(output / "allowance.sqlite3", max_requests=96,
                              max_observed_cost_usd=str(LIMIT), reservation_policy=frozen["metadata"]["policy"])
    status, failure = "not_evaluable", None
    try:
        client = previous.make_client("local-configuration-placeholder")
        guard.attach(client)
        client.api_key = previous.local_key(root / ".env")
        runner = CTestRunner(output, experiment_id=EXPERIMENT, memory_client=client, guard=guard,
                             allow_model_calls=True, registration_sha256=frozen["seal_sha256"])
        status = runner.run()["real_model_gate"]
    except Exception as error:
        failure = type(error).__name__
        guard.halt()
    finally:
        state = guard.snapshot()
        if state["calls_used"] is None or state["unknown_cost_requests"] is None:
            status, failure = "not_evaluable", failure or "AccountingUnavailable"
        guard.close()
        known = None if state["known_cost_usd"] is None else Decimal(state["known_cost_usd"])
        previous.write_new(output / "execution-receipt.json", {"seal_sha256": frozen["seal_sha256"],
             "real_model_gate": status, "failure_type": failure, "accounting": state,
             "prior_ctest_known_cost_usd": str(PARENT_COST), "old_unknown_requests_retained": 2,
             "ctest_total_known_cost_usd": None if known is None else str(PARENT_COST + known),
             "cumulative_e1_ctest_known_usd": None if known is None else str(previous.PRIOR_KNOWN + PARENT_COST + known),
             "completed_at": datetime.now(timezone.utc).isoformat()})
    return {"real_model_gate": status, "failure_type": failure, "calls_used": state["calls_used"],
            "known_cost_usd": state["known_cost_usd"], "unknown_cost_requests": state["unknown_cost_requests"],
            "prior_ctest_known_cost_usd": str(PARENT_COST)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-local-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            if args.output is None:
                raise RunError("output_required")
            result = execute(ROOT, args.metadata, args.registration, args.output, allow_local_env=args.allow_local_env)
        else:
            frozen = (prepare if args.prepare else verify)(ROOT, args.metadata, args.registration)
            result = {"status": "prepared" if args.prepare else "verified", "seal_sha256": frozen["seal_sha256"]}
        print(json.dumps(result))
        return 2 if result.get("real_model_gate") == "not_evaluable" else 0
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "error": str(error) if isinstance(error, RunError) else type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
