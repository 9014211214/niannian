"""Preserved 300-answer checkpoint with one separately approved exception.

The old HTTP 429 has no saved no-content proof. Missing explicit exception
approval blocks execution before Key access. No ancestor stop is cleared.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1br_checkpoint as checkpoint_module
from scripts import run_e1br as base
from scripts import run_e1br_received_continuation as parent
from scripts import run_e1b_repair as common
from scripts.e1br_checkpoint_guard import CheckpointGuardMixin, validate_checkpoint_ledger

CAMPAIGN = Path("artifacts/local/e1br-formal-v3")
PRIOR_E1B, PRIOR_KNOWN = Decimal("0.475402924"), Decimal("1.867782096")
RETAINED_UNKNOWN_HOLD = Decimal("0.321493014")
ALLOWANCE = Decimal("0.703104062")
PRIOR_DISPATCHES, RETAINED_MAIN, PRIOR_UNKNOWN = 844, 300, 9
MAX_WIRE_REQUESTS = 2283
NEW_COUNTS = checkpoint_module.NEW_COUNTS
SELECTED, BUDGET_CHOICE, RunError = base.SELECTED, base.BUDGET_CHOICE, base.RunError
EXCEPTION_APPROVAL_PATH = Path("artifacts/local/tasks/20261004-07/e1br-checkpoint-exception-approval.json")


def refuse_execution(*args, **kwargs):
    raise RunError("checkpoint_prepare_only_old_429_has_no_content_proof_no_execute")


def exception_template(root, *, status="pending", human_quote=""):
    checkpoint = checkpoint_module.load_checkpoint(root)
    failed = checkpoint["failed"]
    return {"schema": "e1br_checkpoint_single_position_exception_v1", "status": status,
        "human_quote": human_quote, "parent_campaign": parent.CAMPAIGN.as_posix(),
        "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
        "parent_registration_sha256": checkpoint["registration"]["seal_sha256"],
        "parent_receipt_sha256": base.original.digest(failed),
        "parent_attempt": {key: failed[key] for key in ("request_id", "operation_id", "session_id", "role",
            "model", "request_sha256", "experiment_position")},
        "old_no_content_proven": False, "old_physical_attempts_at_position": 1,
        "max_new_attempts_at_position": 2, "future_retry_policy_waived": False,
        "user_limit_usd": "1.5", "cumulative_stop_usd": "3", "prior_e1b_known_usd": str(PRIOR_E1B),
        "prior_known_usd": str(PRIOR_KNOWN), "accepted_old_unknown": PRIOR_UNKNOWN,
        "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD),
        "effective_new_allowance_usd": str(ALLOWANCE), "max_physical_calls": MAX_WIRE_REQUESTS}


def exception_anchor(root, template):
    path = Path(root) / EXCEPTION_APPROVAL_PATH
    if not path.exists():
        return None
    raw = common.read_original(Path(root), EXCEPTION_APPROVAL_PATH)
    value = json.loads(raw)
    expected = {**template, "status": value.get("status"), "human_quote": value.get("human_quote")}
    checkpoint_module.require(value == expected and value["status"] == "approved"
        and type(value["human_quote"]) is str and 0 < len(value["human_quote"].strip()) <= 4096,
        "checkpoint_specific_exception_approval_required")
    return {"path": EXCEPTION_APPROVAL_PATH.as_posix(), "sha256": checkpoint_module.sha(raw), "value": value}


def build_engine(root):
    root = Path(root).resolve()
    checkpoint = checkpoint_module.load_checkpoint(root)
    template = exception_template(root)
    anchor = exception_anchor(root, template)
    received = parent.load_received(root)
    revision = common._load_module("scripts._e1br_checkpoint_revision", "run_e1br.py")
    revision.CAMPAIGN, revision.PRIOR_E1B, revision.PRIOR_KNOWN = CAMPAIGN, PRIOR_E1B, PRIOR_KNOWN
    revision.ALLOWANCE, revision.PRIOR_DISPATCHES, revision.RETAINED_MAIN = ALLOWANCE, PRIOR_DISPATCHES, RETAINED_MAIN
    revision.PRIOR_UNKNOWN, revision.RETAINED_UNKNOWN_HOLD = PRIOR_UNKNOWN, RETAINED_UNKNOWN_HOLD
    revision.MAX_WIRE_REQUESTS, revision.NEW_COUNTS = MAX_WIRE_REQUESTS, NEW_COUNTS
    revision.PARENT_MANIFEST_SHA = checkpoint_module.MANIFEST_SHA
    revision.parent_originals = lambda current: checkpoint_module.inherited_files(current, checkpoint)
    revision.retained_extra_rows = lambda _: [deepcopy(received["receipt"]), *deepcopy(checkpoint["accepted"])]
    revision.EXTRA_FILES = ("scripts/run_e1br_received_continuation.py", "scripts/e1br_received_body.py",
        "docs/E1BR_RECEIVED_CONTINUATION.md", "scripts/e1br_checkpoint.py",
        "scripts/run_e1br_checkpoint_continuation.py", "scripts/e1br_checkpoint_guard.py",
        "docs/E1BR_CHECKPOINT_CONTINUATION.md")
    positions = (base.position_inventory() - {("reflection", "e1b_a1", 1, "shared", 0, None)}
                 - {tuple(row["experiment_position"]) for row in checkpoint["accepted"]})
    checkpoint_module.require(len(positions) == 756, "checkpoint_remaining_position_count_changed")
    revision.position_inventory = lambda: positions
    revision.continuation_details = lambda _: {"stopped_checkpoint": deepcopy(checkpoint["summary"]),
        "execution_readiness": "specific_exception_approved" if anchor else "prepare_only",
        "execute_authorized": anchor is not None, "checkpoint_exception": deepcopy(anchor),
        "max_infrastructure_retries_total": 1512,
        "max_additional_dispatches_after_first_new_at_each_position": 1511,
        "retry_total_definition": "cross_campaign_infrastructure_retries_includes_the_first_new_attempt_at_the_old_failed_position",
        "retained_physical_attempts_from_v2": 287, "remaining_max_physical_requests": MAX_WIRE_REQUESTS,
        "remaining_physical_derivation": "755_fresh_positions_times_3_plus_2_for_old_failed_position_plus_16_format_corrections",
        "unresolved_single_position": {"request_id": checkpoint_module.FAILED_REQUEST,
            "old_no_content_proven": False, "remaining_attempts_at_position": 2,
            "explicit_exception_authorization_required": True}}
    revision.runner_for = lambda current, prefix: checkpoint_module.runner_class(checkpoint)

    def first(client, payload, scope, key, prefix, current):
        # This is a preparation invariant, not authorization to send this body.
        checkpoint_module.require(key == ("online_fast", checkpoint_module.CASE, 3, "A1", 1, 8)
            and scope["operation_id"] == checkpoint["failed"]["operation_id"]
            and scope["session_id"] == checkpoint_module.SESSION
            and checkpoint_module.sha(json.dumps(payload, ensure_ascii=False).encode("utf8")) == checkpoint_module.FAILED_WIRE,
            "checkpoint_first_new_wire_must_equal_old_failed_wire")
        return {"continuation_of_request_id": checkpoint_module.FAILED_REQUEST,
            "parent_campaign": parent.CAMPAIGN.as_posix(), "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
            "old_no_content_proven": False, "old_physical_attempts_consumed_at_position": 1}
    revision.first_request_lineage = first
    entry = revision.build_engine(root)

    def require_exception(expected_sha=None):
        if anchor is None:
            refuse_execution()
        checkpoint_module.require(exception_anchor(root, template) == anchor
            and (expected_sha is None or expected_sha == anchor["sha256"]),
            "checkpoint_exception_binding_changed")

    contract = {"schema": "e1br_checkpoint_ledger_v1", "exception_sha256": anchor["sha256"] if anchor else None,
        **{key: template[key] for key in ("parent_attempt", "parent_receipt_sha256", "parent_manifest_sha256",
            "old_no_content_proven", "old_physical_attempts_at_position", "max_new_attempts_at_position")}}
    original_execute, original_validate = entry.execute, entry.validate_authorization
    def execute(*args, checkpoint_exception_sha256=None, **kwargs):
        if not checkpoint_exception_sha256:
            refuse_execution()
        require_exception(checkpoint_exception_sha256)
        return original_execute(*args, **kwargs)
    def validate_authorization(*args, **kwargs):
        require_exception()
        return original_validate(*args, **kwargs)
    class CheckpointGuard(CheckpointGuardMixin, entry.EvidenceGuard):
        def __init__(self, *args, **kwargs):
            require_exception()
            super().__init__(*args, checkpoint_contract=contract, **kwargs)
        def check_authority(self):
            require_exception()
            super().check_authority()
    def validate_closed_ledger(current, frozen, state):
        require_exception()
        checkpoint_module.require(Path(current).resolve() == root, "checkpoint_workspace_changed")
        rows = validate_checkpoint_ledger(entry.campaign(root) / "allowance.sqlite3", contract=contract,
            stage="main", registration_sha256=frozen["seal_sha256"], max_requests=MAX_WIRE_REQUESTS,
            reservation_policy=frozen["metadata"]["policy"])
        checkpoint_module.require(rows == state["attempts"], "checkpoint_snapshot_rows_changed")
        return rows
    old_qualify = entry.qualify
    def qualify(run_root, frozen, state):
        validate_closed_ledger(root, frozen, state)
        result = old_qualify(run_root, frozen, state)
        result["retained_main_prefix"] = {"original_thirteen": entry.retained_prefix["summary"],
            "checkpoint": deepcopy(checkpoint["summary"]), "total_retained": 300}
        result["source_lineage"] = checkpoint_module.qualification_lineage(root, entry.retained_prefix,
            received, checkpoint, frozen, state, entry.logical_attempts, CAMPAIGN)
        return result
    entry.execute, entry.validate_authorization = execute, validate_authorization
    entry.EvidenceGuard, entry.qualify = CheckpointGuard, qualify
    entry.validate_closed_ledger = validate_closed_ledger
    entry.checkpoint_contract, entry.checkpoint_exception = contract, anchor
    entry.stopped_checkpoint = checkpoint
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("assess")
    report.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
            command.add_argument("--budget-choice", choices=(BUDGET_CHOICE,), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
            command.add_argument("--checkpoint-exception-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "assess":
        assessment = common._load_module("scripts._e1br_checkpoint_assessment", "e1br_assessment.py")
        assessment.revision = sys.modules[__name__]
        value = assessment.assess(ROOT)
        base.original.write_once(args.output, value)
        print(json.dumps(value, indent=2))
        return
    entry = build_engine(ROOT)
    if args.command == "prepare":
        value = entry.prepare(ROOT, args.metadata, args.registration, stage="main")
    elif args.command == "verify":
        value = entry.verify(ROOT, args.metadata, args.registration)
    else:
        value = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
            budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
            stage_authorization_sha256=args.stage_authorization_sha256,
            checkpoint_exception_sha256=args.checkpoint_exception_sha256)
        print(json.dumps(value, indent=2))
        return
    print(json.dumps({"status": value["protocol"]["execution_readiness"],
        "execute_authorized": entry.checkpoint_exception is not None, "paid_calls": 0,
        "seal_sha256": value["seal_sha256"], "retained_first_answers": 300,
        "remaining_basic_positions": 756, "remaining_physical_cap": MAX_WIRE_REQUESTS}, indent=2))


if __name__ == "__main__":
    main()
