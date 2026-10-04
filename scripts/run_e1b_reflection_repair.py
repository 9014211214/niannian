"""Fixed formal-v3 entry with the human-approved inclusive USD 1.5/3 limits.

No budget-reset/new-campaign switch exists. Both prior campaigns and source
archives are checked before each send. The anchored decision still requires a
new source freeze and matching campaign/stage authorization before execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1b as original
from scripts import run_e1b_repair as previous
from aria.engine.memory.reflection_contract import reflection_response_schema
from aria.engine.policy.reflection_json import REFLECTION_JSON_PROTOCOL_VERSION, reflection_wire_schema

CAMPAIGN = Path("artifacts/local/e1b-formal-v3")
PARENT = previous.CAMPAIGN
ARCHIVE = Path("artifacts/local/tasks/20261003-05/source-frozen-v2")
PARENT_TREE = "09578a35e4cc07b944e59e6e7fcd73369c1eb00a03fa67597962f6cb3729510a"
PARENT_FILES = 1196
PARENT_SEALS = {
    "prescreen": "61de48bde416ef0bfaf818f23759bd19ca8935c9e529a5a4fc5f833e0ce6bed8",
    "main": "ce7414dc24284ddff0f30da35025326693ecc9f0483736ef66aabd971b285664",
}
PARENT_RECEIPTS = {
    "prescreen": "3a514a5b7f7355f43648350cbf4fa732926fcb7baa58f701a2f3eeac80939dd6",
    "main": "6e594c3dfe9f367c4d2d8d69b03e88445b86d6a4590997058044a591ad5645cf",
}
PARENT_COST = Decimal("0.067340992")
PRIOR_E1B = Decimal("0.067992682")
PRIOR_KNOWN = Decimal("1.460371854")
PROPOSED_E1B_LIMIT = Decimal("1.5")
PROPOSED_CUMULATIVE_STOP = Decimal("3")
PROPOSED_ALLOWANCE = min(PROPOSED_E1B_LIMIT - PRIOR_E1B, PROPOSED_CUMULATIVE_STOP - PRIOR_KNOWN)
BUDGET_CHOICE = "12_usd1_5_cumulative3"
# Only a newly received human reply can replace this; a local authorization
# file, CLI flag or proposed protocol is never sufficient by itself.
HUMAN_BUDGET_APPROVAL = {
    "human_quote": "同意将E1b总上限提高至1.5美元、累计停报线提高至3美元（均包含已花费用）",
    "human_message_reference": "20261003-06 current human reply in this project conversation approving E1b USD1.5 and cumulative USD3, both inclusive of prior spend",
    "budget_choice": BUDGET_CHOICE,
}
RunError = original.RunError


def parent_evidence(root):
    root = Path(root).resolve()
    first = previous.parent_evidence(root)
    files = [{"path": p.relative_to(root).as_posix(),
              "sha256": hashlib.sha256(previous.read_original(root, p.relative_to(root))).hexdigest()}
             for p in sorted((root / PARENT).rglob("*")) if p.is_file()]
    if len(files) != PARENT_FILES or original.digest(files) != PARENT_TREE:
        raise RunError("second_campaign_originals_changed")
    registrations, receipts = {}, {}
    for stage in ("prescreen", "main"):
        raw = previous.read_original(root, PARENT / stage / "registration.json")
        value = json.loads(raw)
        if (value["seal_sha256"] != PARENT_SEALS[stage]
                or original.digest({k: v for k, v in value.items() if k != "seal_sha256"}) != PARENT_SEALS[stage]
                or value["stage"] != stage or len(value["source"]) != 115):
            raise RunError("second_campaign_registration_changed")
        if stage == "prescreen" and previous.read_original(root, ARCHIVE / "registration.json") != raw:
            raise RunError("second_campaign_archive_registration_changed")
        registrations[stage] = value
        receipt = json.loads(previous.read_original(root, PARENT / stage / "execution-receipt.json"))
        if (receipt["receipt_sha256"] != PARENT_RECEIPTS[stage]
                or original.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != PARENT_RECEIPTS[stage]
                or receipt["registration_sha256"] != PARENT_SEALS[stage]
                or receipt["stage"] != stage or receipt["old_unknown_requests_retained"] != 2
                or receipt["status"] != ("completed" if stage == "prescreen" else "not_evaluable")):
            raise RunError("second_campaign_receipt_changed")
        receipts[stage] = receipt
    if registrations["prescreen"]["source"] != registrations["main"]["source"]:
        raise RunError("second_campaign_sources_disagree")
    for item in registrations["prescreen"]["source"]:
        if hashlib.sha256(previous.read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("second_campaign_frozen_source_changed")
    state = receipts["main"]["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("second_campaign_ledger_not_closed")
    before = original.byte_sha(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        records = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
    attempts = [json.loads(raw) for _, raw in records]
    encoded_policy = json.dumps(registrations["main"]["metadata"]["policy"], sort_keys=True, allow_nan=False)
    if (before != original.byte_sha(path) or attempts != state["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(records, attempts))
            or settings != (1, 1636, "0.606969138", "model_or_storage_failed")
            or policy != (encoded_policy,) or identity != (state["ledger_id"],)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded_policy.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "0.606969138"
            or state["calls_used"] != 554 or len(attempts) != 554
            or any(r["status"] != "received" for r in attempts)
            or state["unknown_cost_requests"] != 0 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed" or Decimal(state["reserved_cost_usd"]) != 0
            or attempts[:540] != receipts["prescreen"]["accounting"]["attempts"]):
        raise RunError("second_campaign_ledger_receipt_mismatch")
    counts = Counter()
    for row in attempts:
        stage = row.get("e1b_stage")
        if stage not in PARENT_SEALS or row.get("registration_sha256") != PARENT_SEALS[stage]:
            raise RunError("second_campaign_request_stage_changed")
        counts[(stage, row.get("role"))] += 1
        sequence = sum(counts[(stage, role)] for role in ("online_fast", "memory", "reflection"))
        relative = Path(stage) / "wire-requests" / f"{sequence:04d}.json"
        if (row.get("wire_request_path") != relative.as_posix()
                or hashlib.sha256(previous.read_original(root, PARENT / relative)).hexdigest() != row["request_sha256"]):
            raise RunError("second_campaign_wire_changed")
    if counts != Counter({("prescreen", "online_fast"): 540, ("main", "online_fast"): 12,
                          ("main", "memory"): 1, ("main", "reflection"): 1}):
        raise RunError("second_campaign_actual_counts_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts), Decimal(0))
    if (total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipts["main"]["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN
            or Decimal(first["known_cost_usd"]) + total != PRIOR_E1B):
        raise RunError("second_campaign_cost_cannot_be_reset")
    return {"first_campaign": first, "second_campaign": {
        "campaign": PARENT.as_posix(), "files": PARENT_FILES, "tree_sha256": PARENT_TREE,
        "registration_sha256": PARENT_SEALS, "receipt_sha256": PARENT_RECEIPTS,
        "archived_sources": 115, "calls": 554, "known_cost_usd": str(total),
        "main_measurement": "not_evaluable", "reused_answers": 0,
        "unknown_cost_requests": 0, "historical_unknown_retained": 2, "ledger_id": state["ledger_id"]}}


def _approval_anchor():
    value = HUMAN_BUDGET_APPROVAL
    if (type(value) is not dict or set(value) != {"human_quote", "human_message_reference", "budget_choice"}
            or value.get("budget_choice") != BUDGET_CHOICE
            or any(type(value.get(k)) is not str or not value[k].strip()
                   for k in ("human_quote", "human_message_reference"))):
        raise RunError("new_human_budget_approval_pending")
    return value


def reflection_originals(run_root, frozen, state):
    """Join all 24 first replies to exact requests and updater outcomes."""
    if frozen["stage"] != "main":
        return []
    rows = [r for r in state["attempts"] if r.get("e1b_stage") == "main" and r.get("role") == "reflection"]
    if len(rows) != 24:
        raise RunError("reflection_first_answer_denominator_mismatch")
    results, used = [], set()
    for case_id in frozen["inputs"]["case_ids"]:
        folder = Path(run_root) / "main" / case_id
        for number in (1, 2):
            paths = {kind: folder / f"reflection-s{number}.{kind}.json" for kind in ("attempt", "first", "product")}
            values = {kind: json.loads(previous.read_original(run_root, path.relative_to(run_root)))
                      for kind, path in paths.items()}
            attempt, first, product = (values[k] for k in ("attempt", "first", "product"))
            session = attempt.get("session_id")
            matching = [r for r in rows if r.get("session_id") == session]
            if len(matching) != 1:
                raise RunError("reflection_first_answer_identity_mismatch")
            row = matching[0]
            if (session in used or attempt.get("case_id") != case_id or attempt.get("session") != number
                    or attempt.get("source_sha256") != frozen["protocol"]["case_source_sha256"]
                    or attempt.get("status") != "started"
                    or first.get("schema") != "e1b_reflection_first_v1"
                    or first.get("case_id") != case_id or first.get("session") != number
                    or first.get("source_sha256") != attempt["source_sha256"]
                    or first.get("session_id") != session or product.get("session_id") != session
                    or product.get("session") != number or product.get("schema") != "e1b_reflection_product_v1"
                    or row.get("operation_id") != f"e1b-review-{session}"
                    or first.get("request_hash_kind") != "actual_wire"
                    or first.get("request_sha256") != row["request_sha256"]
                    or first.get("wire_visible_content_sha256") != row["visible_content_sha256"]
                    or first.get("visible_response_sha256") != row["client_return_content_sha256"]
                    or first.get("response_hash_source") != "client_return_content"
                    or first.get("request_identity_missing") or first.get("receipt_status") != "received"
                    or first.get("parse_status") != "object" or first.get("duplicate_key_count") != 0
                    or first.get("known_fields_present") != dict.fromkeys(("protocol_version", "reviews", "changes"), True)
                    or first.get("client_diagnostic", {}).get("output_kind") != "reflection_json_schema"
                    or type(first.get("public_projection_dropped_fields")) is not int
                    or first["public_projection_dropped_fields"] < 0
                    or type(first.get("public_projection")) is not dict
                    or product.get("first_available") is not True or product.get("updater_raised") is not False
                    or product.get("capture_error") is not None or product.get("product_status") != "ok"
                    or product.get("last_error_type") is not None or product.get("last_error_detail") is not None
                    or product.get("first_sha256") != original.byte_sha(paths["first"])):
                raise RunError("reflection_first_answer_or_product_invalid")
            used.add(session)
            results.append({"case_id": case_id, "session": number, "receipt_sha256": original.digest(row),
                "public_projection_dropped_fields": first["public_projection_dropped_fields"],
                **{kind + "_sha256": original.byte_sha(path) for kind, path in paths.items()}})
    if len(results) != 24 or len(used) != 24:
        raise RunError("reflection_first_answer_denominator_mismatch")
    return results


def build_engine(root):
    root = Path(root).resolve()
    wrapper = previous._load_module("scripts._e1b_reflection_repair_wrapper", "run_e1b_repair.py")
    wrapper.CAMPAIGN, wrapper.PRIOR_KNOWN = CAMPAIGN, PRIOR_KNOWN
    wrapper.PARENT_COST, wrapper.E1B_REMAINING = PRIOR_E1B, PROPOSED_ALLOWANCE
    wrapper.parent_evidence = parent_evidence
    entry = wrapper.build_engine(root)
    entry.CUMULATIVE_STOP = PROPOSED_CUMULATIVE_STOP
    entry.FILES = (*entry.FILES, "scripts/run_e1b_reflection_repair.py")
    base_protocol, base_qualify, base_receipt = entry.protocol, entry.qualify, entry.read_receipt

    def protocol():
        value = base_protocol()
        if (REFLECTION_JSON_PROTOCOL_VERSION != "reflection_json_schema_v1"
                or value["product_configuration"].get("reflection_return_format") != "json_schema"):
            raise RunError("revised_reflection_json_protocol_required")
        value.update(schema="aria_e1b_reflection_repair_execution_v1", campaign="e1b-formal-v3")
        value["product_configuration"].update(reflection_return_format="json_schema",
            reflection_json_protocol_version=REFLECTION_JSON_PROTOCOL_VERSION,
            reflection_wire_schema_sha256=original.digest(reflection_wire_schema()),
            reflection_local_schema_sha256=original.digest(reflection_response_schema()),
            reflection_first_answer_capture="e1b_reflection_first_v1")
        value["budget"].update(status="pending" if HUMAN_BUDGET_APPROVAL is None else "human_decision_anchored",
            cumulative_stop_usd="2" if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_CUMULATIVE_STOP),
            current_remaining_usd=str((Decimal("2") if HUMAN_BUDGET_APPROVAL is None else PROPOSED_CUMULATIVE_STOP) - PRIOR_KNOWN),
            e1b_total_user_limit_usd=None if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_E1B_LIMIT),
            e1b_remaining_usd=None if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_E1B_LIMIT - PRIOR_E1B),
            effective_new_allowance_usd=None if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_ALLOWANCE),
            currently_approved_e1b_limit_usd="1" if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_E1B_LIMIT),
            currently_approved_cumulative_stop_usd="2" if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_CUMULATIVE_STOP),
            currently_approved_remaining_usd=str(Decimal("2") - PRIOR_KNOWN) if HUMAN_BUDGET_APPROVAL is None else str(PROPOSED_ALLOWANCE),
            proposed_e1b_limit_usd=str(PROPOSED_E1B_LIMIT), proposed_cumulative_stop_usd=str(PROPOSED_CUMULATIVE_STOP),
            proposed_new_allowance_usd=str(PROPOSED_ALLOWANCE),
            human_budget_approval_sha256=None if HUMAN_BUDGET_APPROVAL is None else original.digest(_approval_anchor()))
        return value

    def authorization_template(current=None):
        return {"schema": "e1b_reflection_repair_authorization_v1", "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": None, "prior_known_usd": str(PRIOR_KNOWN),
            "prior_e1b_known_usd": str(PRIOR_E1B), "cumulative_stop_usd": None, "accepted_old_unknown": 2,
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None,
            "human_quote": None, "human_message_reference": None, "human_budget_approval_sha256": None}

    def validate_authorization(current, frozen, *, budget_choice, campaign_authorization_sha256,
                               stage_authorization_sha256):
        anchor = _approval_anchor()
        if Path(current).resolve() != root:
            raise RunError("repair_parent_or_workspace_changed")
        if budget_choice != BUDGET_CHOICE:
            raise RunError("explicit_new_budget_pair_required")
        if not all(type(s) is str and original.SHA.fullmatch(s)
                   for s in (campaign_authorization_sha256, stage_authorization_sha256)):
            raise RunError("independently_confirmed_authorization_hashes_required")
        expected = authorization_template(root)
        expected.update(status="approved", budget_choice=BUDGET_CHOICE, user_limit_usd=str(PROPOSED_E1B_LIMIT),
            cumulative_stop_usd=str(PROPOSED_CUMULATIVE_STOP), human_quote=anchor["human_quote"],
            human_message_reference=anchor["human_message_reference"], human_budget_approval_sha256=original.digest(anchor))
        auth = entry.read_json(entry.campaign(root) / "authorization.json")
        if auth != expected or original.digest(auth) != campaign_authorization_sha256:
            raise RunError("campaign_not_explicitly_authorized")
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
            "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": campaign_authorization_sha256}
        if (frozen.get("workspace_identity") != entry.workspace_identity(root)
                or entry.read_json(entry.campaign(root) / f"{frozen['stage']}.authorization.json") != phase
                or original.digest(phase) != stage_authorization_sha256):
            raise RunError("stage_not_explicitly_authorized")
        return auth, PROPOSED_ALLOWANCE

    def qualify(run_root, frozen, state):
        result = base_qualify(run_root, frozen, state)
        result["reflection_originals"] = reflection_originals(run_root, frozen, {**state, "attempts": entry.logical_attempts(state)})
        return result

    def read_receipt(current, stage):
        value = base_receipt(current, stage)
        if stage == "main":
            saved = entry.campaign(root) / "main"
            frozen = entry.read_json(saved / "registration.json")
            run = saved / "synthetic-emotion-experiments/e1b-main-v1/run"
            if (frozen["seal_sha256"] != value["registration_sha256"]
                    or original.digest({k: v for k, v in frozen.items() if k != "seal_sha256"}) != frozen["seal_sha256"]
                    or reflection_originals(run, frozen, {**value["accounting"], "attempts": entry.logical_attempts(value["accounting"])}) != value["qualification"]["reflection_originals"]):
                raise RunError("saved_reflection_originals_changed")
        return value

    entry.protocol, entry.authorization_template = protocol, authorization_template
    entry.validate_authorization, entry.qualify, entry.read_receipt = validate_authorization, qualify, read_receipt
    return entry


def assessment_engine(entry):
    # The older read-only assessor has a fixed legacy choice literal. Translate
    # only its internal call; execute still rejects that old budget choice.
    adapter = SimpleNamespace(**vars(entry))
    def validate_authorization(root, frozen, *, budget_choice, **kwargs):
        if budget_choice != "12_usd1":
            raise RunError("legacy_assessment_budget_contract_changed")
        return entry.validate_authorization(root, frozen, budget_choice=BUDGET_CHOICE, **kwargs)
    adapter.validate_authorization = validate_authorization
    return adapter


def assess(root):
    try:
        entry = build_engine(root)
        module = previous._load_module("scripts._e1b_reflection_repair_assessment", "e1b_assessment.py")
        module.execution = assessment_engine(entry)
        result = module.assess(root)
        result["prior_e1b_known_cost_usd"] = str(PRIOR_E1B)
        if "known_campaign_cost_usd" in result:
            cost = Decimal(result["known_campaign_cost_usd"])
            result.update(total_e1b_known_cost_usd=str(PRIOR_E1B + cost), cumulative_known_cost_usd=str(PRIOR_KNOWN + cost))
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
