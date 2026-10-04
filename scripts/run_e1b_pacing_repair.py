"""Fixed formal-v5 with the exact accepted third unknown and shared pacing.

All four stopped campaigns, sources and charges remain immutable. A complete
new measurement uses one new fixed ledger; no old answers, resumes or retries.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts import run_e1b_structure_repair as previous
from scripts import run_e1b_reflection_repair as reflection

CAMPAIGN = Path("artifacts/local/e1b-formal-v5")
PARENT = Path("artifacts/local/e1b-formal-v4")
ARCHIVE = Path("artifacts/local/tasks/20261003-06/source-frozen-v4")
PARENT_FILES = 19
PARENT_TREE = "67b974f1f6fdbbe011c81b7b383a6e594da0705a03f08b4f4c6514e42891d88c"
PARENT_SEAL = "fd9f5f5fe8fa8b413044d9a555fa6d778158099474b60e2536a169e46e566d9b"
PARENT_RECEIPT = "4b5520bf9b159a7421009720b3c447aaf05bdb5db103336ce1670ff0f3245c3e"
PARENT_COST = Decimal("0.00018047")
PARENT_UNKNOWN_REQUEST = "bb6f3ab761aa4908a3a2cafb680febd8"
PARENT_UNKNOWN_WIRE = "e5d6819446b316b7b3ec546469524ef620972fa3f0309dd735f28e896d968efe"
PRIOR_E1B = Decimal("0.069089082")
PRIOR_KNOWN = Decimal("1.461468254")
RETAINED_UNKNOWN_HOLD = Decimal("0.00071965")
ACCEPTED_OLD_UNKNOWN = 3
MIN_INTERVAL_SECONDS = 10
ALLOWANCE = min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD
BUDGET_CHOICE = reflection.BUDGET_CHOICE
HUMAN_BUDGET_APPROVAL = dict(previous.HUMAN_BUDGET_APPROVAL)
HUMAN_UNKNOWN_APPROVAL = {
    "human_quote": "接受新增这 1 笔暂时未知（合计 3 笔），继续执行",
    "human_message_reference": "20261003-07 current human reply in this project conversation accepting the new formal-v4 HTTP429 unknown charge, three unresolved charges total, and continuing execution",
    "campaign": PARENT.as_posix(), "request_id": PARENT_UNKNOWN_REQUEST,
    "request_sha256": PARENT_UNKNOWN_WIRE, "accepted_total_unknown": ACCEPTED_OLD_UNKNOWN,
    "retained_unknown_reserve_usd": str(RETAINED_UNKNOWN_HOLD),
}
RunError = original.RunError


def parent_evidence(root):
    root = Path(root).resolve()
    earlier = previous.parent_evidence(root)
    files = [{"path": p.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(common.read_original(root, p.relative_to(root))).hexdigest()}
        for p in sorted((root / PARENT).rglob("*")) if p.is_file()]
    if len(files) != PARENT_FILES or original.digest(files) != PARENT_TREE:
        raise RunError("fourth_campaign_originals_changed")
    raw = common.read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw)
    if (registration["seal_sha256"] != PARENT_SEAL
            or original.digest({k: v for k, v in registration.items() if k != "seal_sha256"}) != PARENT_SEAL
            or registration["stage"] != "prescreen" or len(registration["source"]) != 119
            or common.read_original(root, ARCHIVE / "registration.json") != raw):
        raise RunError("fourth_campaign_registration_changed")
    for item in registration["source"]:
        if hashlib.sha256(common.read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("fourth_campaign_frozen_source_changed")
    receipt = json.loads(common.read_original(root, PARENT / "prescreen/execution-receipt.json"))
    if (receipt["receipt_sha256"] != PARENT_RECEIPT
            or original.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != PARENT_RECEIPT
            or receipt["registration_sha256"] != PARENT_SEAL or receipt["stage"] != "prescreen"
            or receipt["status"] != "not_evaluable" or receipt["old_unknown_requests_retained"] != 2):
        raise RunError("fourth_campaign_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("fourth_campaign_ledger_not_closed")
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
            or settings != (1, 1636, "1.431091388", "model_or_storage_failed")
            or policy != (encoded_policy,) or identity != (state["ledger_id"],)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded_policy.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "1.431091388"
            or state["calls_used"] != 5 or len(attempts) != 5
            or [r["status"] for r in attempts] != ["received"] * 4 + ["failed"]
            or state["unknown_cost_requests"] != 1 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed" or Decimal(state["reserved_cost_usd"]) != RETAINED_UNKNOWN_HOLD):
        raise RunError("fourth_campaign_ledger_receipt_mismatch")
    for number, row in enumerate(attempts, 1):
        relative = Path("prescreen/wire-requests") / f"{number:04d}.json"
        if (row.get("e1b_stage") != "prescreen" or row.get("role") != "online_fast"
                or row.get("registration_sha256") != PARENT_SEAL
                or row.get("wire_request_path") != relative.as_posix()
                or hashlib.sha256(common.read_original(root, PARENT / relative)).hexdigest() != row["request_sha256"]):
            raise RunError("fourth_campaign_request_identity_changed")
    unknown = attempts[-1]
    if (unknown["request_id"] != PARENT_UNKNOWN_REQUEST or unknown["request_sha256"] != PARENT_UNKNOWN_WIRE
            or unknown.get("http_status") != 429 or unknown.get("usage") != {}
            or unknown.get("reservation", {}).get("reserved_cost_usd") != str(RETAINED_UNKNOWN_HOLD)):
        raise RunError("fourth_campaign_unknown_charge_identity_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts[:-1]), Decimal(0))
    first_two = earlier["earlier_campaigns"]
    earlier_e1b = (Decimal(first_two["first_campaign"]["known_cost_usd"])
        + Decimal(first_two["second_campaign"]["known_cost_usd"])
        + Decimal(earlier["third_campaign"]["known_cost_usd"]))
    if (total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipt["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN
            or earlier_e1b + total != PRIOR_E1B):
        raise RunError("fourth_campaign_cost_cannot_be_reset")
    return {"earlier_campaigns": earlier, "fourth_campaign": {"campaign": PARENT.as_posix(),
        "files": PARENT_FILES, "tree_sha256": PARENT_TREE, "registration_sha256": PARENT_SEAL,
        "receipt_sha256": PARENT_RECEIPT, "archived_sources": 119, "calls": 5,
        "received": 4, "failed": 1, "unissued_prescreen": 535, "known_cost_usd": str(total),
        "unknown_cost_requests": 1, "historical_unknown_retained": 2,
        "retained_unknown_reserve_usd": str(RETAINED_UNKNOWN_HOLD),
        "unknown_request_id": PARENT_UNKNOWN_REQUEST, "unknown_request_sha256": PARENT_UNKNOWN_WIRE, "reused_answers": 0,
        "measurement": "not_evaluable", "ledger_id": state["ledger_id"]}}


def _unknown_anchor():
    value = HUMAN_UNKNOWN_APPROVAL
    expected = {"campaign": PARENT.as_posix(), "request_id": PARENT_UNKNOWN_REQUEST,
        "request_sha256": PARENT_UNKNOWN_WIRE, "accepted_total_unknown": ACCEPTED_OLD_UNKNOWN,
        "retained_unknown_reserve_usd": str(RETAINED_UNKNOWN_HOLD)}
    if (type(value) is not dict or set(value) != {*expected, "human_quote", "human_message_reference"}
            or any(value.get(k) != v for k, v in expected.items())
            or any(type(value.get(k)) is not str or not value[k].strip()
                   for k in ("human_quote", "human_message_reference"))):
        raise RunError("specific_unknown_charge_human_approval_required")
    return value


def build_engine(root):
    root = Path(root).resolve()
    wrapper = common._load_module("scripts._e1b_pacing_repair_wrapper", "run_e1b_structure_repair.py")
    wrapper.CAMPAIGN, wrapper.PRIOR_KNOWN, wrapper.PRIOR_E1B = CAMPAIGN, PRIOR_KNOWN, PRIOR_E1B
    wrapper.ALLOWANCE = ALLOWANCE
    wrapper.HUMAN_BUDGET_APPROVAL = HUMAN_BUDGET_APPROVAL
    wrapper.parent_evidence = parent_evidence
    entry = wrapper.build_engine(root)
    entry.ACCEPTED_OLD_UNKNOWN = ACCEPTED_OLD_UNKNOWN
    entry.RETAINED_UNKNOWN_HOLD = RETAINED_UNKNOWN_HOLD
    entry.MIN_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
    entry.FILES = (*entry.FILES, "scripts/run_e1b_pacing_repair.py")
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight

    def protocol():
        value = base_protocol()
        value.update(schema="aria_e1b_pacing_repair_execution_v1", campaign="e1b-formal-v5")
        value["transport"].update(min_interval_seconds=MIN_INTERVAL_SECONDS,
            pacing_scope="one_shared_durable_guard_all_roles", pacing_clock="monotonic",
            retry_after="diagnostic_only_no_automatic_retry")
        value["budget"].update(accepted_old_unknown=ACCEPTED_OLD_UNKNOWN,
            historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD),
            hold_is_provider_invoice_ceiling=False,
            known_only_remaining_usd=str(ALLOWANCE + RETAINED_UNKNOWN_HOLD),
            effective_new_allowance_usd=str(ALLOWANCE),
            human_unknown_approval_sha256=original.digest(_unknown_anchor()))
        return value

    def preflight():
        base_preflight()
        if (entry.MIN_INTERVAL_SECONDS != MIN_INTERVAL_SECONDS or MIN_INTERVAL_SECONDS != 10
                or entry.ACCEPTED_OLD_UNKNOWN != ACCEPTED_OLD_UNKNOWN or ACCEPTED_OLD_UNKNOWN != 3
                or entry.RETAINED_UNKNOWN_HOLD != RETAINED_UNKNOWN_HOLD
                or ALLOWANCE != min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD):
            raise RunError("paced_inclusive_budget_configuration_changed")
        _unknown_anchor()

    def authorization_template(current=None):
        return {"schema": "e1b_pacing_repair_authorization_v1", "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": None, "prior_known_usd": str(PRIOR_KNOWN),
            "prior_e1b_known_usd": str(PRIOR_E1B), "cumulative_stop_usd": None,
            "accepted_old_unknown": ACCEPTED_OLD_UNKNOWN, "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD),
            "effective_new_allowance_usd": str(ALLOWANCE),
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None,
            "human_quote": None, "human_message_reference": None, "human_budget_approval_sha256": None,
            "unknown_human_quote": None, "unknown_human_message_reference": None, "human_unknown_approval_sha256": None}

    def validate_authorization(current, frozen, *, budget_choice, campaign_authorization_sha256,
                               stage_authorization_sha256):
        preflight()
        if Path(current).resolve() != root:
            raise RunError("repair_parent_or_workspace_changed")
        if budget_choice != BUDGET_CHOICE:
            raise RunError("explicit_new_budget_pair_required")
        if not all(type(s) is str and original.SHA.fullmatch(s)
                   for s in (campaign_authorization_sha256, stage_authorization_sha256)):
            raise RunError("independently_confirmed_authorization_hashes_required")
        budget, unknown = HUMAN_BUDGET_APPROVAL, _unknown_anchor()
        expected = authorization_template(root)
        expected.update(status="approved", budget_choice=BUDGET_CHOICE, user_limit_usd="1.5",
            cumulative_stop_usd="3", human_quote=budget["human_quote"],
            human_message_reference=budget["human_message_reference"], human_budget_approval_sha256=original.digest(budget),
            unknown_human_quote=unknown["human_quote"], unknown_human_message_reference=unknown["human_message_reference"],
            human_unknown_approval_sha256=original.digest(unknown))
        auth = entry.read_json(entry.campaign(root) / "authorization.json")
        if auth != expected or original.digest(auth) != campaign_authorization_sha256:
            raise RunError("campaign_not_explicitly_authorized")
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
            "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": campaign_authorization_sha256}
        if (frozen.get("workspace_identity") != entry.workspace_identity(root)
                or frozen["protocol"] != protocol()
                or entry.read_json(entry.campaign(root) / f"{frozen['stage']}.authorization.json") != phase
                or original.digest(phase) != stage_authorization_sha256):
            raise RunError("stage_not_explicitly_authorized")
        return auth, ALLOWANCE

    class ApprovedPacedGuard(entry.EvidenceGuard):
        def _reserve(self, client, request, payload, state):
            # The shared guard calls this again after its bounded wait, before
            # reserving or dispatching. Revocation during that wait must win.
            base = entry.campaign(root)
            frozen = entry.read_json(base / self.stage / "registration.json")
            binding = entry.read_json(base / "budget-binding.json")
            started = entry.read_json(base / f"{self.stage}.started.json")
            validate_authorization(root, frozen, budget_choice=BUDGET_CHOICE,
                campaign_authorization_sha256=binding["campaign_authorization_sha256"],
                stage_authorization_sha256=started["stage_authorization_sha256"])
            return super()._reserve(client, request, payload, state)

    base_receipt = entry.read_receipt

    def read_receipt(current, stage):
        value = base_receipt(current, stage)
        if (value.get("old_unknown_requests_retained") != ACCEPTED_OLD_UNKNOWN
                or value.get("historical_unknown_reserved_usd") != str(RETAINED_UNKNOWN_HOLD)):
            raise RunError("historical_unknown_charge_carry_changed")
        return value

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    entry.authorization_template, entry.validate_authorization = authorization_template, validate_authorization
    entry.EvidenceGuard, entry.read_receipt = ApprovedPacedGuard, read_receipt
    return entry


def assess(root):
    try:
        entry = build_engine(root)
        module = common._load_module("scripts._e1b_pacing_repair_assessment", "e1b_assessment.py")
        module.execution = reflection.assessment_engine(entry)
        result = module.assess(root)
        result["prior_e1b_known_cost_usd"] = str(PRIOR_E1B)
        result["historical_unknown_requests_retained"] = ACCEPTED_OLD_UNKNOWN
        result["historical_unknown_reserved_usd"] = str(RETAINED_UNKNOWN_HOLD)
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
            command.add_argument("--budget-choice", choices=(BUDGET_CHOICE,), required=True)
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
