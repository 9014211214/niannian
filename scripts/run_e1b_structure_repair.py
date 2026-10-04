"""Fixed formal-v4 under the existing inclusive USD 1.5/3 human approval.

All three stopped campaigns, sources and charges remain immutable. A complete
new measurement uses one new fixed ledger; no old answers, resumes or retries.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts import run_e1b_reflection_repair as previous
from aria.engine.policy import online_json

CAMPAIGN = Path("artifacts/local/e1b-formal-v4")
PARENT = Path("artifacts/local/e1b-formal-v3")
ARCHIVE = Path("artifacts/local/tasks/20261003-06/source-frozen-v3")
PARENT_FILES = 37
PARENT_TREE = "330b32d5e0bf6b91c0dab81ed439d90397131e718f9f1c28ecdcf32366aa7ebb"
PARENT_SEAL = "93970bee5d192603b81a2a89e1fe3b4bbef125697ca134ef54faf669ba8172cd"
PARENT_RECEIPT = "ddf47b791686636e9390dedc3143c18acf79da91f91bf75c3c98486d1abfd256"
PARENT_COST = Decimal("0.00091593")
PRIOR_E1B = Decimal("0.068908612")
PRIOR_KNOWN = Decimal("1.461287784")
ALLOWANCE = min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN)
HUMAN_BUDGET_APPROVAL = dict(previous.HUMAN_BUDGET_APPROVAL)
RunError = original.RunError


def parent_evidence(root):
    root = Path(root).resolve()
    earlier = previous.parent_evidence(root)
    files = [{"path": p.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(common.read_original(root, p.relative_to(root))).hexdigest()}
        for p in sorted((root / PARENT).rglob("*")) if p.is_file()]
    if len(files) != PARENT_FILES or original.digest(files) != PARENT_TREE:
        raise RunError("third_campaign_originals_changed")
    raw = common.read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw)
    if (registration["seal_sha256"] != PARENT_SEAL
            or original.digest({k: v for k, v in registration.items() if k != "seal_sha256"}) != PARENT_SEAL
            or registration["stage"] != "prescreen" or len(registration["source"]) != 117
            or common.read_original(root, ARCHIVE / "registration.json") != raw):
        raise RunError("third_campaign_registration_changed")
    for item in registration["source"]:
        if hashlib.sha256(common.read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("third_campaign_frozen_source_changed")
    receipt = json.loads(common.read_original(root, PARENT / "prescreen/execution-receipt.json"))
    if (receipt["receipt_sha256"] != PARENT_RECEIPT
            or original.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != PARENT_RECEIPT
            or receipt["registration_sha256"] != PARENT_SEAL or receipt["stage"] != "prescreen"
            or receipt["status"] != "not_evaluable" or receipt["old_unknown_requests_retained"] != 2):
        raise RunError("third_campaign_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("third_campaign_ledger_not_closed")
    before = original.byte_sha(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        records = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
    attempts = [json.loads(raw) for _, raw in records]
    encoded_policy = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    if (before != original.byte_sha(path) or attempts != state["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(records, attempts))
            or settings != (1, 1636, "1.432007318", "model_or_storage_failed")
            or policy != (encoded_policy,) or identity != (state["ledger_id"],)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded_policy.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "1.432007318"
            or state["calls_used"] != 18 or len(attempts) != 18
            or [r["status"] for r in attempts] != ["received"] * 17 + ["failed"]
            or state["unknown_cost_requests"] != 0 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed" or Decimal(state["reserved_cost_usd"]) != 0):
        raise RunError("third_campaign_ledger_receipt_mismatch")
    for number, row in enumerate(attempts, 1):
        relative = Path("prescreen/wire-requests") / f"{number:04d}.json"
        if (row.get("e1b_stage") != "prescreen" or row.get("role") != "online_fast"
                or row.get("registration_sha256") != PARENT_SEAL
                or row.get("wire_request_path") != relative.as_posix()
                or hashlib.sha256(common.read_original(root, PARENT / relative)).hexdigest() != row["request_sha256"]):
            raise RunError("third_campaign_request_identity_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts), Decimal(0))
    earlier_e1b = Decimal(earlier["first_campaign"]["known_cost_usd"]) + Decimal(earlier["second_campaign"]["known_cost_usd"])
    if (total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipt["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN
            or earlier_e1b + total != PRIOR_E1B):
        raise RunError("third_campaign_cost_cannot_be_reset")
    return {"earlier_campaigns": earlier, "third_campaign": {"campaign": PARENT.as_posix(),
        "files": PARENT_FILES, "tree_sha256": PARENT_TREE, "registration_sha256": PARENT_SEAL,
        "receipt_sha256": PARENT_RECEIPT, "archived_sources": 117, "calls": 18,
        "received": 17, "failed": 1, "unissued_prescreen": 522, "known_cost_usd": str(total),
        "unknown_cost_requests": 0, "historical_unknown_retained": 2, "reused_answers": 0,
        "measurement": "not_evaluable", "ledger_id": state["ledger_id"]}}


def build_engine(root):
    wrapper = common._load_module("scripts._e1b_structure_repair_wrapper", "run_e1b_reflection_repair.py")
    wrapper.CAMPAIGN, wrapper.PRIOR_KNOWN, wrapper.PRIOR_E1B = CAMPAIGN, PRIOR_KNOWN, PRIOR_E1B
    wrapper.PROPOSED_ALLOWANCE = ALLOWANCE
    wrapper.HUMAN_BUDGET_APPROVAL = HUMAN_BUDGET_APPROVAL
    wrapper.parent_evidence = parent_evidence
    entry = wrapper.build_engine(root)
    entry.FILES = (*entry.FILES, "scripts/run_e1b_structure_repair.py")
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight

    def check_structure():
        if getattr(online_json, "ONLINE_JSON_STRUCTURE_VERSION", None) != "online_json_structure_v1":
            raise RunError("revised_online_json_structure_required")

    def protocol():
        check_structure()
        value = base_protocol()
        value.update(schema="aria_e1b_structure_repair_execution_v1", campaign="e1b-formal-v4")
        value["product_configuration"].update(
            online_json_structure_version=online_json.ONLINE_JSON_STRUCTURE_VERSION,
            online_return_format="json_object", online_prompt_format_style="schema_guided",
            online_shape_diagnostic="fixed_schema_paths_types_only")
        return value

    def preflight():
        base_preflight()
        check_structure()
        client = entry.make_client("online_fast", "local-configuration-placeholder")
        if client.online_return_format != "json_object" or client.role != "online_fast":
            raise RunError("structure_repair_requires_actual_json_object_client")

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    return entry


def assess(root):
    try:
        entry = build_engine(root)
        module = common._load_module("scripts._e1b_structure_repair_assessment", "e1b_assessment.py")
        module.execution = previous.assessment_engine(entry)
        result = module.assess(root)
        result["prior_e1b_known_cost_usd"] = str(PRIOR_E1B)
        if "known_campaign_cost_usd" in result:
            known = Decimal(result["known_campaign_cost_usd"])
            result.update(total_e1b_known_cost_usd=str(PRIOR_E1B + known), cumulative_known_cost_usd=str(PRIOR_KNOWN + known))
        return result
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        return {"schema": "e1b_formal_assessment_v1", "status": "not_evaluable", "effect_claim_eligible": False,
            "main_gate": "not_evaluable", "formation_gate": "not_evaluable", "overall_gate": "not_evaluable",
            "model_calls": 0, "reason": str(error) if isinstance(error, RunError) else type(error).__name__}


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
            command.add_argument("--budget-choice", choices=(previous.BUDGET_CHOICE,), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "assess":
        result = assess(ROOT)
        original.write_once(args.output, result)
    else:
        entry = build_engine(ROOT)
        if args.command == "prepare":
            value = entry.prepare(ROOT, args.metadata, args.registration, stage=args.stage)
            result = {"seal_sha256": value["seal_sha256"], "stage": value["stage"],
                "authorization": entry.authorization_template(ROOT), "paid_calls": 0}
        elif args.command == "verify":
            value = entry.verify(ROOT, args.metadata, args.registration)
            result = {"verified": True, "seal_sha256": value["seal_sha256"], "paid_calls": 0}
        else:
            result = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
                budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
                stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
