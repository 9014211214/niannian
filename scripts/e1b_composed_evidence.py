"""Fixed v7 donor evidence for a prospective, separately authorized v8.

Only byte-identical complete cases travel into a report view. Neither a donor
receipt nor a donor row is rewritten as successful v8 accounting.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts import run_e1b as base
from scripts import run_e1b_repair as common
from scripts import run_e1b_bounded_repair as v7
from scripts import run_e1b_observed_repair as observed

PARENT = Path("artifacts/local/e1b-formal-v7")
ARCHIVE = Path("artifacts/local/tasks/20261004-01/source-frozen-v7")
PARENT_FILES = 613
PARENT_TREE = "8d33c5a5cb4f64c9c28f99fe115c2ce5c335697b3fd82b2469bbbc6be582f012"
PARENT_SEAL = "a33922227b23c619945a5dc729b9be1cef30076dc4f2a69b54dd774442e6276f"
PARENT_RECEIPT = "f77ee47b98a365f8f16c3f68a7e395f4f14e7acc8b4f13b658ac6932d5021a6d"
FAILED_REQUEST = "63208ea0c62547b5934237decbefa765"
FAILED_WIRE = "890ad7632a4bcc0d681ed870caf81baad0e3ad27b12f9679f6f1d58c38b95c6d"
PARENT_COST = Decimal("0.02481454")
PARENT_HOLD = Decimal("0.000692822")
RETAINED_CASES = tuple(f"e1b_{group}{number}" for group in "abcd" for number in (1, 2, 3)) + ("e1b_e1",)
REMAINING_CASES = ("e1b_e2", "e1b_e3", "e1b_f1", "e1b_f2", "e1b_f3")
RUN = Path("prescreen/synthetic-emotion-experiments/e1b-prescreen-v1/run")
# These files change only orchestration, bounded transport, diagnostics or docs.
# Everything else in the donor inventory must remain byte-identical, including
# prompts, models, parsing/Schema, cases, perception and product delivery.
ALLOWED_SOURCE_CHANGES = frozenset({
    "aria/engine/policy/bounded_recovery.py", "aria/engine/policy/minimal_calls.py",
    "aria/engine/policy/transport_failure.py", "aria/engine/policy/reserved_calls.py",
    "aria/engine/policy/observed_transport.py", "scripts/run_e1b.py",
    "scripts/e1b_assessment.py", "docs/E1B_EXECUTION.md", "docs/VALIDATION.md",
})


def _require(ok, code):
    if not ok:
        raise base.RunError(code)


def originals(root):
    root = Path(root).resolve()
    return [{"path": p.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(common.read_original(root, p.relative_to(root))).hexdigest()}
            for p in sorted((root / PARENT).rglob("*")) if p.is_file()]


def fingerprint(root):
    """Cheap change detection within an already fully checked local execution."""
    root = Path(root).resolve()
    paths = [p for folder in (PARENT, ARCHIVE) for p in sorted((root / folder).rglob("*")) if p.is_file()]
    _require(all(not p.is_symlink() and not any(q.is_symlink() for q in p.parents) for p in paths), "donor_link_rejected")
    return tuple((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def validate_product_compatibility(root, registration):
    changed = []
    for item in registration["source"]:
        archived = common.read_original(root, ARCHIVE / item["path"])
        _require(hashlib.sha256(archived).hexdigest() == item["sha256"], "v7_frozen_source_changed")
        current = hashlib.sha256(common.read_original(root, Path(item["path"]))).hexdigest()
        if current != item["sha256"]:
            _require(item["path"] in ALLOWED_SOURCE_CHANGES, "composed_product_source_incompatible")
            changed.append({"path": item["path"], "old_sha256": item["sha256"], "new_sha256": current})
    return changed


def donor_qualification(root, registration, state):
    """Validate an explicitly fixed received prefix, not an arbitrary filter."""
    _require(len(state["attempts"]) == 391
             and [r["status"] for r in state["attempts"]] == ["received"] * 390 + ["failed"], "donor_prefix_changed")
    module = common._load_module("scripts._e1b_v8_donor_qualification", "run_e1b.py")
    module.COUNTS = {**module.COUNTS, "prescreen": {"online_fast": 390}}
    scope = {**registration, "inputs": {"case_ids": list(RETAINED_CASES)}}
    result = module.qualify(Path(root) / PARENT / RUN, scope, {**state, "attempts": state["attempts"][:390]})
    expected = [(c, "A0", r, t) for c in RETAINED_CASES for r in (1, 2, 3) for t in range(10)]
    _require([(p["case_id"], p["arm"], p["repeat"], p["turn"]) for p in result["positions"]] == expected,
             "donor_positions_changed")
    for case in RETAINED_CASES:
        report = base.read_json(Path(root) / PARENT / RUN / "prescreen" / case / "report.json")
        _require(report["case_id"] == case and report["scored_reply_count"] == 9 and report["frozen"] is False,
                 "donor_report_identity_changed")
    _require(not (Path(root) / PARENT / "prescreen/blind-bundle").exists(), "donor_already_blind_scored_or_exported")
    return result


def load_parent(root):
    """Full independent bytes/ledger check, called at build and final qualify."""
    root = Path(root).resolve()
    earlier = v7.parent_evidence(root)
    files = originals(root)
    _require(len(files) == PARENT_FILES and base.digest(files) == PARENT_TREE, "v7_originals_changed")
    raw = common.read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw)
    _require(registration["seal_sha256"] == PARENT_SEAL
             == base.digest({k: v for k, v in registration.items() if k != "seal_sha256"})
             and common.read_original(root, ARCHIVE / "registration.json") == raw
             and len(registration["source"]) == 126, "v7_registration_changed")
    changes = validate_product_compatibility(root, registration)
    receipt = json.loads(common.read_original(root, PARENT / "prescreen/execution-receipt.json"))
    _require(receipt["receipt_sha256"] == PARENT_RECEIPT
             == base.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
             and receipt["registration_sha256"] == PARENT_SEAL and receipt["stage"] == "prescreen"
             and receipt["status"] == "not_evaluable" and receipt["old_unknown_requests_retained"] == 5
             and receipt["historical_unknown_reserved_usd"] == "0.002638320", "v7_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    _require(not any(Path(str(path) + s).exists() for s in ("-wal", "-shm", "-journal")), "v7_ledger_not_closed")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        recovery = db.execute("SELECT policy FROM bounded_recovery_policy WHERE id=1").fetchone()
        relations = db.execute("SELECT * FROM authorized_recoveries").fetchall()
        pacing = db.execute("SELECT min_interval_seconds FROM request_pacing WHERE id=1").fetchone()
        transport = db.execute("SELECT version FROM transport_observation WHERE id=1").fetchone()
    encoded = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    _require([json.loads(raw) for _, raw in rows] == state["attempts"]
             and all(key == row["request_id"] for (key, _), row in zip(rows, state["attempts"]))
             and settings == (1, 1638, "1.410501882", "model_or_storage_failed")
             and identity == (state["ledger_id"],) and policy == (encoded,)
             and recovery == (json.dumps(v7.recovery_policy(), sort_keys=True, allow_nan=False),)
             and not relations and pacing == (10.0,) and transport == ("connection_trace_v1",)
             and state["calls_used"] == 391 and state["unknown_cost_requests"] == 1
             and state["stopped"] is True and state["stop_reason"] == "model_or_storage_failed"
             and Decimal(state["reserved_cost_usd"]) == PARENT_HOLD, "v7_ledger_receipt_changed")
    from aria.engine.policy.bounded_recovery import validate_recovery_records
    projected = validate_recovery_records(state["attempts"], v7.recovery_policy(), relations=[])
    _require(projected["logical_received"] == 390 and projected["extra_dispatches"] == 0
             and projected["new_unknown_requests"] == 1, "v7_recovery_state_changed")
    observed.validate_transport_records(state["attempts"], {"transport": observed.transport_contract()})
    for index, row in enumerate(state["attempts"], 1):
        relative = Path("prescreen/wire-requests") / f"{index:04d}.json"
        _require(row["registration_sha256"] == PARENT_SEAL and row["role"] == "online_fast"
                 and row["e1b_stage"] == "prescreen" and row["wire_request_path"] == relative.as_posix()
                 and hashlib.sha256(common.read_original(root, PARENT / relative)).hexdigest() == row["request_sha256"],
                 "v7_wire_evidence_changed")
    failed = state["attempts"][-1]
    _require(failed["request_id"] == FAILED_REQUEST and failed["request_sha256"] == FAILED_WIRE
             and failed["usage"] == {} and failed.get("http_status") is None
             and failed["transport_events"]["events"] == {"connection_preparation_started": True}
             and Decimal(failed["reservation"]["reserved_cost_usd"]) == PARENT_HOLD, "v7_tls_failure_changed")
    cost = sum((Decimal(str(r["usage"]["cost"])) for r in state["attempts"][:390]), Decimal(0))
    _require(cost == PARENT_COST == Decimal(state["known_cost_usd"])
             and Decimal(receipt["cumulative_prior_known_usd"]) == v7.PRIOR_KNOWN, "v7_cost_changed")
    qualification = donor_qualification(root, registration, state)
    emotion_sha = hashlib.sha256(common.read_original(root, PARENT / RUN / "emotion-plan.json")).hexdigest()
    summary = {"earlier_campaigns": earlier, "donor": {
        "campaign": PARENT.as_posix(), "tree_sha256": PARENT_TREE, "files": PARENT_FILES,
        "registration_sha256": PARENT_SEAL, "receipt_sha256": PARENT_RECEIPT,
        "status": "not_evaluable", "physical_attempts": 391, "retained_received": 390,
        "retained_case_ids": list(RETAINED_CASES), "remaining_case_ids": list(REMAINING_CASES),
        "retained_scoring_positions": 117, "remaining_scoring_positions": 45,
        "known_cost_usd": str(PARENT_COST), "active_unknown_requests": 1,
        "unknown_reserved_usd": str(PARENT_HOLD), "ledger_id": state["ledger_id"],
        "failed_request_id": FAILED_REQUEST, "failed_request_sha256": FAILED_WIRE,
        "emotion_plan_sha256": emotion_sha, "source_report_sha256": qualification["source_report_sha256"],
        "qualification_sha256": base.digest(qualification), "allowed_source_changes": changes,
        "old_labels_reused": 0, "old_records_rewritten": False}}
    return {"summary": summary, "registration": registration, "state": state,
            "qualification": qualification, "fingerprint": fingerprint(root)}


def compose_view(root, run_root, donor):
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    _require(run_root.is_relative_to(root) and not any(p.is_symlink() for p in (run_root, *run_root.parents)), "composition_view_path_invalid")
    _require(base.byte_sha(run_root / "emotion-plan.json") == donor["summary"]["donor"]["emotion_plan_sha256"], "composed_emotion_plan_changed")
    files = []
    for case in RETAINED_CASES:
        relative = RUN / "prescreen" / case / "report.json"
        raw = common.read_original(root, PARENT / relative)
        target = run_root / "prescreen" / case / "report.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        with target.open("xb") as stream:
            stream.write(raw)
        files.append({"case_id": case, "source": (PARENT / relative).as_posix(),
                      "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"schema": "e1b_composed_report_view_v1", "donor": donor["summary"]["donor"],
                "copied_reports": files, "new_case_ids": list(REMAINING_CASES), "old_bytes_rewritten": False}
    base.write_once(run_root / "composition.json", manifest)
    return run_root


def qualify_view(root, run_root, frozen, state, new_qualification, donor):
    run_root = Path(run_root)
    manifest = base.read_json(run_root / "composition.json")
    _require(manifest["donor"] == donor["summary"]["donor"]
             and manifest["new_case_ids"] == list(REMAINING_CASES) and manifest["old_bytes_rewritten"] is False,
             "composed_manifest_changed")
    expected_files = [{"case_id": case, "source": (PARENT / RUN / "prescreen" / case / "report.json").as_posix(),
                       "sha256": donor["qualification"]["source_report_sha256"][case]} for case in RETAINED_CASES]
    _require(manifest["copied_reports"] == expected_files
             and all(base.byte_sha(run_root / "prescreen" / row["case_id"] / "report.json") == row["sha256"] for row in expected_files),
             "composed_donor_copy_changed")
    expected_new = [(c, "A0", r, t) for c in REMAINING_CASES for r in (1, 2, 3) for t in range(10)]
    _require([(p["case_id"], p["arm"], p["repeat"], p["turn"]) for p in new_qualification["positions"]] == expected_new,
             "composed_new_positions_changed")
    first = state["attempts"][0]
    _require(first.get("continuation_of_request_id") == FAILED_REQUEST
             and first["request_sha256"] == FAILED_WIRE
             and first.get("continuation_coordinate") == {"case_id": "e1b_e2", "arm": "A0", "repeat": 1, "turn": 0},
             "specific_tls_position_not_bound")
    result = deepcopy(new_qualification)
    result.update(schema="e1b_composed_transport_qualification_v1", actual_counts={"online_fast": 540},
                  new_dispatch_counts={"online_fast": 150}, historical_received_counts={"online_fast": 390},
                  source_report_sha256={**donor["qualification"]["source_report_sha256"], **result["source_report_sha256"]},
                  positions=deepcopy(donor["qualification"]["positions"]) + result["positions"],
                  provenance={"donor": donor["summary"]["donor"], "new_registration_sha256": frozen["seal_sha256"],
                              "new_ledger_id": state["ledger_id"], "new_qualification_sha256": base.digest(new_qualification),
                              "view_sha256": base.byte_sha(run_root / "composition.json")},
                  measurement_claim="prospectively_registered_composition_not_v7_completion",
                  old_receipt_status="not_evaluable", first_answer_basis="fixed_donor_prefix_and_new_first_answers_no_selection")
    return result
