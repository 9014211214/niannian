"""One prospective E1b repair inside the already approved, shared USD 1 budget.

The stopped formal-v1 measurement is immutable and charged in full. This entry
uses a fixed formal-v2 ledger for a complete new measurement, never a resume or
a selectable campaign. Prepare/verify/assess are offline and never read a key.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1b as original
from aria.engine.policy import online_json

CAMPAIGN = Path("artifacts/local/e1b-formal-v2")
PARENT = Path("artifacts/local/e1b-formal-v1")
ARCHIVE = Path("artifacts/local/tasks/20261003-04/frozen-source")
PARENT_SEAL = "c2e03bf79c15289e7986a49badcdcfa765822d2660300df9fdee63604aa79be4"
PARENT_RECEIPT = "d38634b8c6720a2d9938d4eb9ea2d132d5071f0d53d851ba7a62e1b834e6b6b3"
PARENT_TREE = "e5f9f14bd005846f9aa7581a1d8551fce5e2391b59b37c0a73ec0ed47c7cb8ee"
PARENT_COST = Decimal("0.00065169")
PRIOR_KNOWN = Decimal("1.393030862")
E1B_REMAINING = Decimal("1") - PARENT_COST
RunError = original.RunError


def read_original(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if (not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise RunError("prior_evidence_or_source_link_rejected")
    return path.read_bytes()


def parent_evidence(root):
    """Read the first ledger immutably; reject any altered original or archive."""
    root = Path(root).resolve()
    paths = sorted(path for path in (root / PARENT).rglob("*") if path.is_file())
    rows = [{"path": path.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(read_original(root, path.relative_to(root))).hexdigest()} for path in paths]
    if len(rows) != 14 or original.digest(rows) != PARENT_TREE:
        raise RunError("prior_campaign_originals_changed")
    raw_registration = read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw_registration)
    unsigned = {key: value for key, value in registration.items() if key != "seal_sha256"}
    if (registration["seal_sha256"] != PARENT_SEAL or original.digest(unsigned) != PARENT_SEAL
            or len(registration["source"]) != 114
            or read_original(root, ARCHIVE / "registration.json") != raw_registration):
        raise RunError("prior_registration_or_archive_changed")
    for item in registration["source"]:
        if hashlib.sha256(read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("prior_frozen_source_changed")
    case_path = "aria/bench/cases/e1b_v1_draft.yaml"
    case = next(row for row in registration["source"] if row["path"] == case_path)
    if hashlib.sha256(read_original(root, case_path)).hexdigest() != case["sha256"]:
        raise RunError("repair_must_not_change_test_cases")
    receipt = json.loads(read_original(root, PARENT / "prescreen/execution-receipt.json"))
    if (receipt["receipt_sha256"] != PARENT_RECEIPT
            or original.digest({key: value for key, value in receipt.items() if key != "receipt_sha256"}) != PARENT_RECEIPT
            or receipt["registration_sha256"] != PARENT_SEAL
            or receipt["status"] != "not_evaluable"
            or receipt["stage"] != "prescreen" or receipt["old_unknown_requests_retained"] != 2):
        raise RunError("prior_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("prior_ledger_not_closed")
    before = original.byte_sha(path)
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            records = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
            settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
            policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
            identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
    except sqlite3.DatabaseError:
        raise RunError("prior_ledger_unreadable") from None
    attempts = [json.loads(raw) for _, raw in records]
    encoded_policy = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    if (before != original.byte_sha(path) or attempts != state["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(records, attempts))
            or settings != (1, 1636, "0.607620828", "model_or_storage_failed")
            or policy != (encoded_policy,) or identity != (state["ledger_id"],)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded_policy.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "0.607620828"
            or state["calls_used"] != 10 or len(attempts) != 10
            or [row["status"] for row in attempts] != ["received"] * 9 + ["failed"]
            or state["unknown_cost_requests"] != 0 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed"
            or Decimal(state["reserved_cost_usd"]) != 0):
        raise RunError("prior_ledger_receipt_mismatch")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts), Decimal(0))
    if (not total.is_finite() or total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipt["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN):
        raise RunError("prior_cost_cannot_be_reset")
    return {"campaign": PARENT.as_posix(), "registration_sha256": PARENT_SEAL,
            "receipt_sha256": PARENT_RECEIPT, "files": rows, "archived_source_files": 114,
            "calls": 10, "delivered_turns": 9, "failed_calls": 1, "unissued_prescreen_calls": 530,
            "known_cost_usd": str(total), "unknown_cost_requests": 0, "historical_unknown_retained": 2,
            "ledger_id": state["ledger_id"], "measurement": "not_evaluable", "reused_answers": 0}


def _load_module(name, filename):
    # A separate namespace avoids changing run_e1b or assessment globals when
    # this entry is imported by tests, a CLI, or another local review tool.
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wire_body(path, body):
    """Save only the already-validated request body, never request headers."""
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RunError("wire_request_path_link_rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def build_engine(root):
    """Bind the reused executor to one root, parent, and fixed repair ledger."""
    root = Path(root).resolve()
    parent = parent_evidence(root)
    entry = _load_module("scripts._e1b_repair_execution", "run_e1b.py")
    entry.RunError = RunError
    entry.CAMPAIGN, entry.PRIOR_KNOWN = CAMPAIGN, PRIOR_KNOWN
    entry.FILES = (*entry.FILES, "scripts/run_e1b_repair.py")
    base_protocol = entry.protocol
    base_prepare, base_verify = entry.prepare, entry.verify
    base_authorize = entry.validate_authorization
    base_receipt, base_qualify = entry.read_receipt, entry.qualify

    def check_parent(current=root):
        if Path(current).resolve() != root or parent_evidence(root) != parent:
            raise entry.RunError("repair_parent_or_workspace_changed")

    def protocol():
        version = getattr(online_json, "ONLINE_JSON_PROTOCOL_VERSION", None)
        if version != "online_json_protocol_v2":
            raise entry.RunError("revised_online_json_protocol_required")
        value = base_protocol()
        value.update(schema="aria_e1b_repair_execution_v1", campaign="e1b-formal-v2", parent=parent,
                     measurement="new_complete_measurement_no_prior_answers_no_best_retry",
                     request_evidence="fictional_experiment_exact_body_bytes_before_send_no_headers")
        value["product_configuration"]["online_json_protocol_version"] = version
        schema = online_json.online_decision_tool_spec(value["product_configuration"]["online_decision_version"])
        value["product_configuration"]["online_json_schema_sha256"] = original.digest(schema["function"]["parameters"])
        value["budget"].update(prior_e1b_known_usd=str(PARENT_COST),
            e1b_total_user_limit_usd="1", e1b_remaining_usd=str(E1B_REMAINING),
            effective_new_allowance_usd=str(min(E1B_REMAINING, entry.CUMULATIVE_STOP - PRIOR_KNOWN)))
        return value

    def prepare(current, metadata, destination, *, stage="prescreen"):
        check_parent(current)
        return base_prepare(current, metadata, destination, stage=stage)

    def verify(current, metadata, registration):
        check_parent(current)
        return base_verify(current, metadata, registration)

    def validate_authorization(current, frozen, **kwargs):
        check_parent(current)
        authorization, limit = base_authorize(current, frozen, **kwargs)
        return authorization, min(E1B_REMAINING, limit)

    def check_wire_evidence(state):
        # Derive the filename from the immutable stage/sequence inventory,
        # rather than following a ledger-supplied path. Include earlier phases.
        counts = dict.fromkeys(entry.STAGES, 0)
        for row in state["attempts"]:
            stage = row.get("e1b_stage")
            if stage not in counts:
                raise entry.RunError("wire_request_stage_invalid")
            counts[stage] += 1
            relative = Path(stage) / "wire-requests" / f"{counts[stage]:04d}.json"
            if row.get("wire_request_path") != relative.as_posix():
                raise entry.RunError("wire_request_identity_changed")
            try:
                raw = read_original(root, CAMPAIGN / relative)
            except OSError:
                raise entry.RunError("wire_request_evidence_missing") from None
            if hashlib.sha256(raw).hexdigest() != row["request_sha256"]:
                raise entry.RunError("wire_request_evidence_changed")

    def read_receipt(current, stage):
        check_parent(current)
        value = base_receipt(current, stage)
        check_wire_evidence(value["accounting"])
        return value

    def qualify(run_root, frozen, state):
        check_wire_evidence(state)
        return base_qualify(run_root, frozen, state)

    class ParentCheckedGuard(entry.EvidenceGuard):
        def _reserve(self, client, request, payload, state):
            check_parent()
            reservation = super()._reserve(client, request, payload, state)
            sequence = 1 + sum(row.get("e1b_stage") == self.stage for row in state["attempts"])
            relative = Path(self.stage) / "wire-requests" / f"{sequence:04d}.json"
            # Budget/route/body validation has passed; the enclosing transaction
            # has not inserted a pending row and the HTTP send has not begun.
            # A partial write remains visible and the campaign cannot restart.
            write_wire_body(entry.campaign(root) / relative, request.data)
            return {**reservation, "wire_request_path": relative.as_posix()}

    entry.protocol, entry.prepare, entry.verify = protocol, prepare, verify
    entry.validate_authorization, entry.EvidenceGuard = validate_authorization, ParentCheckedGuard
    entry.read_receipt, entry.qualify, entry.check_wire_evidence = read_receipt, qualify, check_wire_evidence
    return entry


def assess(root):
    try:
        entry = build_engine(root)
        module = _load_module("scripts._e1b_repair_assessment", "e1b_assessment.py")
        module.execution = entry
        result = module.assess(root)
        result["prior_e1b_known_cost_usd"] = str(PARENT_COST)
        if "known_campaign_cost_usd" in result:
            known = Decimal(result["known_campaign_cost_usd"])
            result["total_e1b_known_cost_usd"] = str(PARENT_COST + known)
            result["cumulative_known_cost_usd"] = str(PRIOR_KNOWN + known)
        return result
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        return {"schema": "e1b_formal_assessment_v1", "status": "not_evaluable",
                "main_gate": "not_evaluable", "formation_gate": "not_evaluable", "overall_gate": "not_evaluable",
                "effect_claim_eligible": False, "model_calls": 0,
                "reason": str(error) if isinstance(error, RunError) else type(error).__name__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assessment = commands.add_parser("assess")
    assessment.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--stage", choices=original.STAGES, default="prescreen")
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
            command.add_argument("--budget-choice", choices=("12_usd1",), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "assess":
        result = assess(ROOT)
        original.write_once(args.output, result)
    else:
        entry = build_engine(ROOT)
        if args.command == "prepare":
            frozen = entry.prepare(ROOT, args.metadata, args.registration, stage=args.stage)
            result = {"stage": frozen["stage"], "seal_sha256": frozen["seal_sha256"],
                      "authorization": entry.authorization_template(ROOT), "paid_calls": 0}
        elif args.command == "verify":
            frozen = entry.verify(ROOT, args.metadata, args.registration)
            result = {"verified": True, "stage": frozen["stage"], "seal_sha256": frozen["seal_sha256"], "paid_calls": 0}
        else:
            result = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
                budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
                stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
