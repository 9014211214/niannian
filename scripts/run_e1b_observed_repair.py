"""Fixed formal-v6 with the accepted fourth unknown and observed transport.

All five stopped campaigns, sources and charges remain immutable. A complete
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
from scripts import run_e1b_pacing_repair as previous
from scripts import run_e1b_reflection_repair as reflection

CAMPAIGN = Path("artifacts/local/e1b-formal-v6")
PARENT = Path("artifacts/local/e1b-formal-v5")
ARCHIVE = Path("artifacts/local/tasks/20261003-07/source-frozen-v5")
PARENT_FILES = 96
PARENT_TREE = "85721fe375ed04e986b8388dc60a94352117bdbed55853fdcb328f7fd1f9b691"
PARENT_SEAL = "5618d67ab77db5e3a54cb3b7d2f0c225ca18e9218440d6c67f435e3d627801fd"
PARENT_RECEIPT = "384c77c69abc51c2f3a34b19a5a02c41bbc4c5fee0044ca8946a9032593240c0"
PARENT_COST = Decimal("0.00350335")
PARENT_UNKNOWN_REQUEST = "3ed6221fedbe4721b03e07047731517c"
PARENT_UNKNOWN_WIRE = "8d0810902c70cb725374de654b13cbd88140f17602ce212b47b69e26e39b2041"
PRIOR_E1B = Decimal("0.072592432")
PRIOR_KNOWN = Decimal("1.464971604")
PRIOR_UNKNOWN_HOLD = Decimal("0.00071965")
PARENT_UNKNOWN_HOLD = Decimal("0.001200844")
RETAINED_UNKNOWN_HOLD = PRIOR_UNKNOWN_HOLD + PARENT_UNKNOWN_HOLD
ACCEPTED_OLD_UNKNOWN = 4
EXPECTED_ACCEPTED_OLD_UNKNOWN = 4
EXECUTION_SCHEMA = "aria_e1b_observed_repair_execution_v1"
AUTHORIZATION_SCHEMA = "e1b_observed_repair_authorization_v1"
CAMPAIGN_NAME = "e1b-formal-v6"
MIN_INTERVAL_SECONDS = 10
OBSERVE_TRANSPORT = True
ALLOWANCE = min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD
BUDGET_CHOICE = reflection.BUDGET_CHOICE
HUMAN_BUDGET_APPROVAL = dict(previous.HUMAN_BUDGET_APPROVAL)
HUMAN_UNKNOWN_APPROVAL = {
    "human_quote": "接受这新增第4笔暂时未知，按原1.5／3美元上限继续",
    "human_message_reference": "20261003-08 current human reply in this project conversation accepting the formal-v5 fourth unknown charge and continuing within the prior inclusive E1b USD1.5/cumulative USD3 limits",
    "campaign": PARENT.as_posix(), "request_id": PARENT_UNKNOWN_REQUEST,
    "request_sha256": PARENT_UNKNOWN_WIRE, "accepted_total_unknown": ACCEPTED_OLD_UNKNOWN,
    "retained_unknown_reserve_usd": str(RETAINED_UNKNOWN_HOLD),
}
RunError = original.RunError


def transport_contract():
    return {"observe_transport": True, "observation_version": "connection_trace_v1",
        "observation_scope": "local_events_not_delivery_or_billing", "absent_events": "not_observed",
        "automatic_redirects": False, "application_write_reconnect": False,
        "connection_preparation_retries": 0, "application_request_retries": 0}


def validate_transport_records(records, protocol):
    contract = transport_contract()
    if any(type(protocol.get("transport", {}).get(k)) is not type(v)
           or protocol["transport"][k] != v for k, v in contract.items()):
        raise RunError("observed_transport_protocol_changed")
    expected = {"connection_preparation_started", "connection_preparation_completed", "tls_completed",
        "application_write_attempted", "request_write_completed", "response_headers_returned"}
    required = {"connection_preparation_completed": {"connection_preparation_started", "tls_completed"},
        "tls_completed": {"connection_preparation_started", "connection_preparation_completed"},
        "application_write_attempted": {"connection_preparation_started", "connection_preparation_completed", "tls_completed"},
        "request_write_completed": {"application_write_attempted"},
        "response_headers_returned": {"request_write_completed"}}
    for row in records:
        value = row.get("transport_events")
        if (type(value) is not dict or set(value) != {"version", "events", "absent_events", "scope"}
                or value.get("version") != "connection_trace_v1" or value.get("absent_events") != "not_observed"
                or value.get("scope") != "local_events_not_delivery_or_billing" or type(value.get("events")) is not dict):
            raise RunError("observed_transport_record_changed")
        events = value["events"]
        if (not set(events) <= expected or any(flag is not True for flag in events.values())
                or any(event in events and not dependencies <= events.keys() for event, dependencies in required.items())
                or row.get("status") not in {"received", "failed"}
                or row["status"] == "received" and events.keys() != expected):
            raise RunError("observed_transport_record_changed")
        # A failed attempt may have no events or only a prefix. Neither implies
        # non-delivery, zero cost, permission to release its hold, or a retry.


def parent_evidence(root):
    root = Path(root).resolve()
    earlier = previous.parent_evidence(root)
    files = [{"path": p.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(common.read_original(root, p.relative_to(root))).hexdigest()}
        for p in sorted((root / PARENT).rglob("*")) if p.is_file()]
    if len(files) != PARENT_FILES or original.digest(files) != PARENT_TREE:
        raise RunError("fifth_campaign_originals_changed")
    raw = common.read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw)
    if (registration["seal_sha256"] != PARENT_SEAL
            or original.digest({k: v for k, v in registration.items() if k != "seal_sha256"}) != PARENT_SEAL
            or registration["stage"] != "prescreen" or len(registration["source"]) != 121
            or common.read_original(root, ARCHIVE / "registration.json") != raw):
        raise RunError("fifth_campaign_registration_changed")
    for item in registration["source"]:
        if hashlib.sha256(common.read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("fifth_campaign_frozen_source_changed")
    receipt = json.loads(common.read_original(root, PARENT / "prescreen/execution-receipt.json"))
    if (receipt["receipt_sha256"] != PARENT_RECEIPT
            or original.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != PARENT_RECEIPT
            or receipt["registration_sha256"] != PARENT_SEAL or receipt["stage"] != "prescreen"
            or receipt["status"] != "not_evaluable" or receipt["old_unknown_requests_retained"] != 3
            or receipt.get("historical_unknown_reserved_usd") != str(PRIOR_UNKNOWN_HOLD)):
        raise RunError("fifth_campaign_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("fifth_campaign_ledger_not_closed")
    before = original.byte_sha(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        records = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
        pacing = db.execute("SELECT min_interval_seconds FROM request_pacing WHERE id=1").fetchone()
    attempts = [json.loads(raw) for _, raw in records]
    encoded_policy = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    if (before != original.byte_sha(path) or attempts != state["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(records, attempts))
            or settings != (1, 1636, "1.430191268", "model_or_storage_failed")
            or pacing != (10.0,) or policy != (encoded_policy,) or identity != (state["ledger_id"],)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded_policy.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "1.430191268"
            or state["calls_used"] != 56 or len(attempts) != 56
            or [r["status"] for r in attempts] != ["received"] * 55 + ["failed"]
            or state["unknown_cost_requests"] != 1 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed" or Decimal(state["reserved_cost_usd"]) != PARENT_UNKNOWN_HOLD):
        raise RunError("fifth_campaign_ledger_receipt_mismatch")
    for number, row in enumerate(attempts, 1):
        relative = Path("prescreen/wire-requests") / f"{number:04d}.json"
        if (row.get("e1b_stage") != "prescreen" or row.get("role") != "online_fast"
                or row.get("registration_sha256") != PARENT_SEAL
                or row.get("wire_request_path") != relative.as_posix()
                or hashlib.sha256(common.read_original(root, PARENT / relative)).hexdigest() != row["request_sha256"]):
            raise RunError("fifth_campaign_request_identity_changed")
    unknown = attempts[-1]
    if (unknown["request_id"] != PARENT_UNKNOWN_REQUEST or unknown["request_sha256"] != PARENT_UNKNOWN_WIRE
            or unknown.get("http_status") is not None or unknown.get("usage") != {}
            or unknown.get("failure_phase") != "connect" or unknown.get("failure_code") != "connection_failed"
            or unknown.get("observation") != {}
            or unknown.get("reservation", {}).get("reserved_cost_usd") != str(PARENT_UNKNOWN_HOLD)):
        raise RunError("fifth_campaign_unknown_charge_identity_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts[:-1]), Decimal(0))
    first_two = earlier["earlier_campaigns"]["earlier_campaigns"]
    earlier_e1b = (Decimal(first_two["first_campaign"]["known_cost_usd"])
        + Decimal(first_two["second_campaign"]["known_cost_usd"])
        + Decimal(earlier["earlier_campaigns"]["third_campaign"]["known_cost_usd"])
        + Decimal(earlier["fourth_campaign"]["known_cost_usd"]))
    if (total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipt["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN
            or earlier_e1b + total != PRIOR_E1B):
        raise RunError("fifth_campaign_cost_cannot_be_reset")
    return {"earlier_campaigns": earlier, "fifth_campaign": {"campaign": PARENT.as_posix(),
        "files": PARENT_FILES, "tree_sha256": PARENT_TREE, "registration_sha256": PARENT_SEAL,
        "receipt_sha256": PARENT_RECEIPT, "archived_sources": 121, "calls": 56,
        "received": 55, "failed": 1, "unissued_prescreen": 484, "known_cost_usd": str(total),
        "unknown_cost_requests": 1, "historical_unknown_retained": 3,
        "retained_unknown_reserve_usd": str(PARENT_UNKNOWN_HOLD),
        "prior_unknown_reserve_usd": str(PRIOR_UNKNOWN_HOLD),
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


def authorization_extensions(*, approved=False):
    """Version-specific extensions are empty for the immutable v6 protocol."""
    return {}


def extend_protocol(value):
    return value


def build_engine(root):
    root = Path(root).resolve()
    wrapper = common._load_module("scripts._e1b_observed_repair_wrapper", "run_e1b_structure_repair.py")
    wrapper.CAMPAIGN, wrapper.PRIOR_KNOWN, wrapper.PRIOR_E1B = CAMPAIGN, PRIOR_KNOWN, PRIOR_E1B
    wrapper.ALLOWANCE = ALLOWANCE
    wrapper.HUMAN_BUDGET_APPROVAL = HUMAN_BUDGET_APPROVAL
    wrapper.parent_evidence = parent_evidence
    entry = wrapper.build_engine(root)
    entry.ACCEPTED_OLD_UNKNOWN = ACCEPTED_OLD_UNKNOWN
    entry.RETAINED_UNKNOWN_HOLD = RETAINED_UNKNOWN_HOLD
    entry.MIN_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
    entry.OBSERVE_TRANSPORT = OBSERVE_TRANSPORT
    entry.validate_transport_records = validate_transport_records
    entry.FILES = (*entry.FILES, "scripts/run_e1b_pacing_repair.py", "scripts/run_e1b_observed_repair.py")
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight

    def protocol():
        value = base_protocol()
        value.update(schema=EXECUTION_SCHEMA, campaign=CAMPAIGN_NAME)
        value["transport"].update(min_interval_seconds=MIN_INTERVAL_SECONDS,
            pacing_scope="one_shared_durable_guard_all_roles", pacing_clock="monotonic",
            retry_after="diagnostic_only_no_automatic_retry", **transport_contract())
        value["budget"].update(accepted_old_unknown=ACCEPTED_OLD_UNKNOWN,
            historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD),
            hold_is_provider_invoice_ceiling=False,
            known_only_remaining_usd=str(ALLOWANCE + RETAINED_UNKNOWN_HOLD),
            effective_new_allowance_usd=str(ALLOWANCE),
            human_unknown_approval_sha256=original.digest(_unknown_anchor()))
        return extend_protocol(value)

    def preflight():
        base_preflight()
        if (entry.MIN_INTERVAL_SECONDS != MIN_INTERVAL_SECONDS or MIN_INTERVAL_SECONDS != 10
                or entry.ACCEPTED_OLD_UNKNOWN != ACCEPTED_OLD_UNKNOWN or ACCEPTED_OLD_UNKNOWN != EXPECTED_ACCEPTED_OLD_UNKNOWN
                or entry.OBSERVE_TRANSPORT is not True or OBSERVE_TRANSPORT is not True
                or entry.RETAINED_UNKNOWN_HOLD != RETAINED_UNKNOWN_HOLD
                or RETAINED_UNKNOWN_HOLD != PRIOR_UNKNOWN_HOLD + PARENT_UNKNOWN_HOLD
                or ALLOWANCE != min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD):
            raise RunError("observed_inclusive_budget_configuration_changed")
        _unknown_anchor()

    def authorization_template(current=None):
        return {"schema": AUTHORIZATION_SCHEMA, "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": None, "prior_known_usd": str(PRIOR_KNOWN),
            "prior_e1b_known_usd": str(PRIOR_E1B), "cumulative_stop_usd": None,
            "accepted_old_unknown": ACCEPTED_OLD_UNKNOWN, "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD),
            "effective_new_allowance_usd": str(ALLOWANCE),
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None,
            "human_quote": None, "human_message_reference": None, "human_budget_approval_sha256": None,
            "unknown_human_quote": None, "unknown_human_message_reference": None, "human_unknown_approval_sha256": None,
            **authorization_extensions()}

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
            human_unknown_approval_sha256=original.digest(unknown), **authorization_extensions(approved=True))
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

    class ApprovedObservedGuard(entry.EvidenceGuard):
        def __init__(self, *args, **kwargs):
            # Every product/judge role shares this same explicitly observed guard.
            if kwargs.pop("observe_transport", True) is not True or entry.OBSERVE_TRANSPORT is not True:
                raise RunError("observed_transport_required")
            super().__init__(*args, observe_transport=True, **kwargs)

        def _finish(self, row):
            # Preserve the first record even if its evidence contract is broken;
            # the caller then stops, without a replay, reset or fee exemption.
            super()._finish(row)
            validate_transport_records([row], protocol())

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
        validate_transport_records(value["accounting"]["attempts"], protocol())
        return value

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    entry.authorization_template, entry.validate_authorization = authorization_template, validate_authorization
    entry.EvidenceGuard, entry.read_receipt = ApprovedObservedGuard, read_receipt
    return entry


def assess(root):
    try:
        entry = build_engine(root)
        module = common._load_module("scripts._e1b_observed_repair_assessment", "e1b_assessment.py")
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
