"""Fixed formal-v7: explicitly authorized, bounded no-answer HTTP429 recovery.

All six stopped campaigns stay immutable. Logical denominators and first actual
answers stay fixed; each additional HTTP dispatch retains its own cost evidence.
"""
from __future__ import annotations

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
from scripts import run_e1b_observed_repair as previous

CAMPAIGN = Path("artifacts/local/e1b-formal-v7")
PARENT = Path("artifacts/local/e1b-formal-v6")
ARCHIVE = Path("artifacts/local/tasks/20261003-08/source-frozen-v6")
PARENT_FILES = 315
PARENT_TREE = "da4a29150bc249bccc62fe503b729ce6cb61b7e4e3bb1df5f71f30e94045c80c"
PARENT_SEAL = "b5d5cb8989029406103fd807163a81446544ce4728c513ac0ff6f0e94c23f51b"
PARENT_RECEIPT = "bf8bb40bfe474f61de566ac8c2bb9b196396c1a779d0933149533a6e62372c38"
PARENT_COST = Decimal("0.014267366")
PARENT_UNKNOWN_REQUEST = "ae115194c3cd4f49bb11ab2a36021a56"
PARENT_UNKNOWN_WIRE = "8f32b49da5500c5c8816e1c5e9754be617e588095a7f4a2af6269e5f30f41d72"
PRIOR_E1B = Decimal("0.086859798")
PRIOR_KNOWN = Decimal("1.479238970")
PRIOR_UNKNOWN_HOLD = Decimal("0.001920494")
PARENT_UNKNOWN_HOLD = Decimal("0.000717826")
RETAINED_UNKNOWN_HOLD = PRIOR_UNKNOWN_HOLD + PARENT_UNKNOWN_HOLD
ACCEPTED_OLD_UNKNOWN = 5
MIN_INTERVAL_SECONDS = 10
MAX_LOGICAL_REQUESTS = 1636
MAX_WIRE_REQUESTS = 1638
ALLOWANCE = min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD
BUDGET_CHOICE = previous.BUDGET_CHOICE
HUMAN_BUDGET_APPROVAL = dict(previous.HUMAN_BUDGET_APPROVAL)
HUMAN_QUOTE = "接受第5笔未知，并批准该方案可能再产生最多2笔未知、逐笔保留预留   尽快开始主实验，现在进度太慢了"
HUMAN_REFERENCE = "human-current-message-20261004-01"
HUMAN_UNKNOWN_APPROVAL = {
    "human_quote": HUMAN_QUOTE, "human_message_reference": HUMAN_REFERENCE,
    "campaign": PARENT.as_posix(), "request_id": PARENT_UNKNOWN_REQUEST,
    "request_sha256": PARENT_UNKNOWN_WIRE, "accepted_total_unknown": ACCEPTED_OLD_UNKNOWN,
    "retained_unknown_reserve_usd": str(RETAINED_UNKNOWN_HOLD),
}
HUMAN_RECOVERY_APPROVAL = {
    "human_quote": HUMAN_QUOTE, "human_message_reference": HUMAN_REFERENCE,
    "campaign": CAMPAIGN.as_posix(), "max_extra_dispatches": 2, "max_new_unknown": 2,
    "backoff_seconds": [60, 120], "max_attempts_per_operation": 3,
    "max_logical_requests": MAX_LOGICAL_REQUESTS, "max_wire_requests": MAX_WIRE_REQUESTS,
    "unknown_holds_retained": True, "stop_before_dispatch_when_unknown_limit_reached": True,
    "scope": "complete_no_model_answer_http429_only_no_resume_no_content_repair",
}
RunError = original.RunError


def recovery_policy():
    anchor = HUMAN_RECOVERY_APPROVAL
    expected = {"campaign": CAMPAIGN.as_posix(), "max_extra_dispatches": 2, "max_new_unknown": 2,
        "backoff_seconds": [60, 120], "max_attempts_per_operation": 3,
        "max_logical_requests": MAX_LOGICAL_REQUESTS, "max_wire_requests": MAX_WIRE_REQUESTS,
        "unknown_holds_retained": True, "stop_before_dispatch_when_unknown_limit_reached": True,
        "scope": "complete_no_model_answer_http429_only_no_resume_no_content_repair"}
    if (set(anchor) != {*expected, "human_quote", "human_message_reference"}
            or any(type(anchor.get(k)) is not type(v) or anchor[k] != v for k, v in expected.items())
            or any(type(anchor.get(k)) is not str or not anchor[k].strip()
                   for k in ("human_quote", "human_message_reference"))):
        raise RunError("specific_bounded_recovery_approval_required")
    return {"schema": "bounded_http429_recovery_v1", "max_extra_dispatches": 2,
        "max_new_unknown": 2, "backoff_seconds": [60, 120], "max_attempts_per_operation": 3,
        "max_logical_requests": MAX_LOGICAL_REQUESTS, "authorization_sha256": original.digest(anchor)}


def parent_evidence(root):
    root = Path(root).resolve()
    earlier = previous.parent_evidence(root)
    files = [{"path": p.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(common.read_original(root, p.relative_to(root))).hexdigest()}
        for p in sorted((root / PARENT).rglob("*")) if p.is_file()]
    if len(files) != PARENT_FILES or original.digest(files) != PARENT_TREE:
        raise RunError("sixth_campaign_originals_changed")
    raw = common.read_original(root, PARENT / "prescreen/registration.json")
    registration = json.loads(raw)
    if (registration["seal_sha256"] != PARENT_SEAL
            or original.digest({k: v for k, v in registration.items() if k != "seal_sha256"}) != PARENT_SEAL
            or registration["stage"] != "prescreen" or len(registration["source"]) != 124
            or common.read_original(root, ARCHIVE / "registration.json") != raw):
        raise RunError("sixth_campaign_registration_changed")
    for item in registration["source"]:
        if hashlib.sha256(common.read_original(root, ARCHIVE / item["path"])).hexdigest() != item["sha256"]:
            raise RunError("sixth_campaign_frozen_source_changed")
    receipt = json.loads(common.read_original(root, PARENT / "prescreen/execution-receipt.json"))
    if (receipt["receipt_sha256"] != PARENT_RECEIPT
            or original.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != PARENT_RECEIPT
            or receipt["registration_sha256"] != PARENT_SEAL or receipt["stage"] != "prescreen"
            or receipt["status"] != "not_evaluable" or receipt["old_unknown_requests_retained"] != 4
            or receipt.get("historical_unknown_reserved_usd") != str(PRIOR_UNKNOWN_HOLD)):
        raise RunError("sixth_campaign_failed_receipt_changed")
    state = receipt["accounting"]
    path = root / PARENT / "allowance.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RunError("sixth_campaign_ledger_not_closed")
    before = original.byte_sha(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
        pacing = db.execute("SELECT min_interval_seconds FROM request_pacing WHERE id=1").fetchone()
        observed = db.execute("SELECT version FROM transport_observation WHERE id=1").fetchone()
    attempts = [json.loads(raw) for _, raw in rows]
    encoded = json.dumps(registration["metadata"]["policy"], sort_keys=True, allow_nan=False)
    if (before != original.byte_sha(path) or attempts != state["attempts"]
            or any(request_id != row["request_id"] for (request_id, _), row in zip(rows, attempts))
            or settings != (1, 1636, "1.425487074", "model_or_storage_failed")
            or policy != (encoded,) or identity != (state["ledger_id"],) or pacing != (10.0,)
            or observed != ("connection_trace_v1",)
            or state["reservation_policy_sha256"] != hashlib.sha256(encoded.encode()).hexdigest()
            or state["max_requests"] != 1636 or state["max_observed_cost_usd"] != "1.425487074"
            or state["calls_used"] != 200 or len(attempts) != 200
            or [row["status"] for row in attempts] != ["received"] * 199 + ["failed"]
            or state["unknown_cost_requests"] != 1 or state["stopped"] is not True
            or state["stop_reason"] != "model_or_storage_failed" or Decimal(state["reserved_cost_usd"]) != PARENT_UNKNOWN_HOLD):
        raise RunError("sixth_campaign_ledger_receipt_mismatch")
    previous.validate_transport_records(attempts, registration["protocol"])
    for number, row in enumerate(attempts, 1):
        relative = Path("prescreen/wire-requests") / f"{number:04d}.json"
        if (row.get("e1b_stage") != "prescreen" or row.get("role") != "online_fast"
                or row.get("registration_sha256") != PARENT_SEAL
                or row.get("wire_request_path") != relative.as_posix()
                or hashlib.sha256(common.read_original(root, PARENT / relative)).hexdigest() != row["request_sha256"]):
            raise RunError("sixth_campaign_request_identity_changed")
    failed = attempts[-1]
    if (failed["request_id"] != PARENT_UNKNOWN_REQUEST or failed["request_sha256"] != PARENT_UNKNOWN_WIRE
            or failed.get("http_status") != 429 or failed.get("usage") != {}
            or failed.get("failure_phase") != "http_status" or failed.get("failure_code") != "http_error"
            or failed.get("observation", {}).get("provider_reported_name") != "Alibaba"
            or failed.get("reservation", {}).get("reserved_cost_usd") != str(PARENT_UNKNOWN_HOLD)):
        raise RunError("sixth_campaign_unknown_charge_identity_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in attempts[:-1]), Decimal(0))
    if (total != PARENT_COST or Decimal(state["known_cost_usd"]) != total
            or Decimal(receipt["cumulative_prior_known_usd"]) + total != PRIOR_KNOWN
            or previous.PRIOR_E1B + total != PRIOR_E1B):
        raise RunError("sixth_campaign_cost_cannot_be_reset")
    return {"earlier_campaigns": earlier, "sixth_campaign": {"campaign": PARENT.as_posix(),
        "files": PARENT_FILES, "tree_sha256": PARENT_TREE, "registration_sha256": PARENT_SEAL,
        "receipt_sha256": PARENT_RECEIPT, "archived_sources": 124, "calls": 200,
        "received": 199, "failed": 1, "unissued_prescreen": 340, "known_cost_usd": str(total),
        "unknown_cost_requests": 1, "historical_unknown_retained": 4,
        "retained_unknown_reserve_usd": str(PARENT_UNKNOWN_HOLD),
        "prior_unknown_reserve_usd": str(PRIOR_UNKNOWN_HOLD), "unknown_request_id": PARENT_UNKNOWN_REQUEST,
        "unknown_request_sha256": PARENT_UNKNOWN_WIRE, "reused_answers": 0,
        "measurement": "not_evaluable", "ledger_id": state["ledger_id"]}}


def build_engine(root):
    from aria.engine.policy.bounded_recovery import validate_recovery_records

    root = Path(root).resolve()
    wrapper = common._load_module("scripts._e1b_bounded_observed", "run_e1b_observed_repair.py")
    for name in ("CAMPAIGN", "PARENT", "PRIOR_KNOWN", "PRIOR_E1B", "ALLOWANCE", "PRIOR_UNKNOWN_HOLD",
            "PARENT_UNKNOWN_HOLD", "RETAINED_UNKNOWN_HOLD", "ACCEPTED_OLD_UNKNOWN", "PARENT_UNKNOWN_REQUEST",
            "PARENT_UNKNOWN_WIRE", "HUMAN_BUDGET_APPROVAL", "HUMAN_UNKNOWN_APPROVAL", "parent_evidence"):
        setattr(wrapper, name, globals()[name])
    wrapper.EXPECTED_ACCEPTED_OLD_UNKNOWN = 5
    wrapper.EXECUTION_SCHEMA = "aria_e1b_bounded_repair_execution_v1"
    wrapper.AUTHORIZATION_SCHEMA = "e1b_bounded_repair_authorization_v1"
    wrapper.CAMPAIGN_NAME = "e1b-formal-v7"
    wrapper.transport_contract = lambda: {**previous.transport_contract(), "application_request_retries": 2}

    def extensions(*, approved=False):
        recovery_policy()
        anchor = HUMAN_RECOVERY_APPROVAL
        return {"recovery_human_quote": anchor["human_quote"] if approved else None,
            "recovery_human_message_reference": anchor["human_message_reference"] if approved else None,
            "human_recovery_approval_sha256": original.digest(anchor) if approved else None}

    def extended_protocol(value):
        value.update(planned_total_calls=MAX_LOGICAL_REQUESTS, max_wire_requests=MAX_WIRE_REQUESTS,
            logical_denominators=dict(original.COUNTS), bounded_recovery=recovery_policy(),
            measurement="new_complete_measurement_first_model_answer_bounded_no_answer_429_recovery")
        value["transport"]["retry_after"] = "bounded_authorized_429_only_no_model_answer"
        return value

    wrapper.authorization_extensions, wrapper.extend_protocol = extensions, extended_protocol
    entry = wrapper.build_engine(root)
    entry.FILES = (*entry.FILES, "scripts/run_e1b_bounded_repair.py")
    entry.MAX_WIRE_REQUESTS = MAX_WIRE_REQUESTS
    entry.RECOVERY_POLICY = recovery_policy()
    base_guard, base_qualify = entry.EvidenceGuard, entry.qualify

    def projection(state, *, complete=False):
        return validate_recovery_records(state["attempts"], recovery_policy(),
            relations=state.get("recovery", {}).get("relations", []), require_complete=complete)

    def assert_clean(state, *, expected_calls=None, final=False):
        projected = projection(state, complete=True)
        acceptable = final and state.get("stop_reason") in {"request_limit_reached", "logical_request_limit_reached"}
        if ((state.get("stopped") and not acceptable) or type(state.get("calls_used")) is not int
                or len(state["attempts"]) != state["calls_used"] or state["calls_used"] > MAX_WIRE_REQUESTS
                or state.get("unknown_cost_requests") != projected["authorized_unknown_requests"]
                or projected["logical_calls_used"] != projected["logical_received"]
                or any(state.get("recovery", {}).get(k) != v for k, v in projected.items() if k != "logical_attempts")
                or state.get("recovery", {}).get("policy_sha256") != hashlib.sha256(
                    json.dumps(recovery_policy(), sort_keys=True, allow_nan=False).encode()).hexdigest()
                or expected_calls is not None and projected["logical_received"] != expected_calls
                or any("cost" not in row.get("usage", {}) for row in projected["logical_attempts"])):
            raise RunError("accounting_pending_unapproved_unknown_failed_or_wrong_denominator")

    class ApprovedBoundedGuard(base_guard):
        def __init__(self, *args, **kwargs):
            if kwargs.pop("recovery_policy", recovery_policy()) != recovery_policy():
                raise RunError("bounded_recovery_policy_changed")
            super().__init__(*args, recovery_policy=recovery_policy(), **kwargs)

    def qualify(run_root, frozen, state):
        result = base_qualify(run_root, frozen, state)
        result["physical_attempts"] = sum(row.get("e1b_stage") == frozen["stage"] for row in state["attempts"])
        result["bounded_recovery"] = {k: v for k, v in projection(state, complete=True).items() if k != "logical_attempts"}
        result["first_answer_basis"] = "unique_received_after_validated_no_answer_429_relations"
        return result

    entry.logical_attempts = lambda state: projection(state)["logical_attempts"]
    entry.assert_clean, entry.EvidenceGuard, entry.qualify = assert_clean, ApprovedBoundedGuard, qualify
    return entry


def assess(root):
    wrapper = common._load_module("scripts._e1b_bounded_assess", "run_e1b_observed_repair.py")
    for name in ("build_engine", "PRIOR_E1B", "PRIOR_KNOWN", "ACCEPTED_OLD_UNKNOWN", "RETAINED_UNKNOWN_HOLD"):
        setattr(wrapper, name, globals()[name])
    return wrapper.assess(root)


def main(argv=None):
    wrapper = common._load_module("scripts._e1b_bounded_cli", "run_e1b_observed_repair.py")
    wrapper.ROOT, wrapper.build_engine, wrapper.assess = ROOT, build_engine, assess
    return wrapper.main(argv)


if __name__ == "__main__":
    main()
