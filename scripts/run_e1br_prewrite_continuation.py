"""Continue the approved no-content retry pool after a proven prewrite failure.

Preserves 834 first results and dispatches 222 remaining logical positions.
The old failure and reserve remain; its one connection rebuild is already used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1br as base
from scripts import run_e1b_repair as common
from scripts import e1br_prewrite_checkpoint as checkpoint_module
from scripts import e1br_memory_qualification as qualification
from scripts.e1br_memory_runtime import MemoryGuardMixin, MemoryExperimentApp, validate_memory_ledger, SCHEMA
from scripts.e1br_format_runtime import terminal_format_failure, validate_format_ledger
from scripts.e1br_prewrite_runtime import (
    PrewriteGuardMixin, validate_prewrite_ledger, SCHEMA as PREWRITE_SCHEMA, digest as prewrite_digest,
    validate_contract as validate_prewrite_contract,
)
from aria.engine.policy.online_decision import ONLINE_ID_NAMESPACE_VERSION

CAMPAIGN = Path("artifacts/local/e1br-formal-v6")
PRIOR_E1B, PRIOR_KNOWN = Decimal("0.992574442"), Decimal("2.384953614")
E1B_LIMIT, CUMULATIVE_STOP = Decimal("2"), Decimal("3.5")
RETAINED_UNKNOWN_HOLD = Decimal("0.325713486")
ALLOWANCE = Decimal("0.681712072")
PRIOR_DISPATCHES, RETAINED_MAIN, PRIOR_UNKNOWN = 1381, 834, 11
MAX_WIRE_REQUESTS = 669
NEW_COUNTS = {"main": {"online_fast": 214, "memory": 4, "reflection": 4}}
BUDGET_CHOICE = "e1br_12_codex_only_usd2_cumulative3_5_positive_prewrite"
SELECTED, RunError = base.SELECTED, base.RunError
APPROVAL_PATH = Path("artifacts/local/tasks/20261005-03/prewrite-existing-authorization-interpretation-v1.json")
APPROVAL_SHA = "42796c80bad70369690ed4ca000dbfcb05bb6831868ab5158ca1b81ff99d0234"
PROMPT_SOURCES = {
    "aria/engine/policy/online_json.py": "a5f653b43d01bfe73ff915f9a716bf996876bcb981c2dafe258ff5c1f6b63bf7",
    "aria/engine/memory/consolidator.py": "9f07bea3196a16c3fa9abaddaa339179c91966522cf4a17181d03b606343dde2",
}


def approval(root):
    raw = common.read_original(Path(root), APPROVAL_PATH)
    if hashlib.sha256(raw).hexdigest() != APPROVAL_SHA:
        raise RunError("prewrite_existing_authority_interpretation_changed")
    value = json.loads(raw)
    expected = {"e1b_inclusive_cap_usd": str(E1B_LIMIT), "cumulative_inclusive_stop_usd": str(CUMULATIVE_STOP),
        "prior_e1b_known_usd": str(PRIOR_E1B), "prior_cumulative_known_usd": str(PRIOR_KNOWN),
        "inherited_unknown_count": PRIOR_UNKNOWN, "inherited_unknown_hold_usd": str(RETAINED_UNKNOWN_HOLD),
        "new_allowance_usd": str(ALLOWANCE), "retained_initial_results": RETAINED_MAIN,
        "remaining_basic_positions": 222, "remaining_counts": NEW_COUNTS["main"],
        "max_new_physical_requests": MAX_WIRE_REQUESTS, "first_new_infrastructure_retry_index": 1,
        "maximum_retry_index_in_same_position": 2, "preparation_rebuild_already_consumed": True}
    if any(value.get(key) != expected_value or type(value.get(key)) is not type(expected_value)
           for key, expected_value in expected.items()):
        raise RunError("prewrite_existing_authorized_scope_changed")
    for key in ("prior_user_approval", "latest_user_approval", "approved_proposal_source"):
        linked = value[key]
        if hashlib.sha256(common.read_original(Path(root), Path(linked["path"]))).hexdigest() != linked["sha256"]:
            raise RunError("prewrite_original_human_approval_or_proposal_changed")
    latest = json.loads(common.read_original(Path(root), Path(value["latest_user_approval"]["path"])))
    if latest["human_message_verbatim"] != value["latest_human_message_verbatim"]:
        raise RunError("prewrite_existing_human_quote_changed")
    return {"approval_file_sha256": APPROVAL_SHA, "human_quote": value["latest_human_message_verbatim"],
        "human_message_reference": "20261005-03", "new_human_approval_claimed": False,
        "authority_kind": value["authority_kind"], "scope": value["root_interpretation"],
        "approved_no_content_scope_source_sha256": value["approved_proposal_source"]["sha256"]}


def build_engine(root):
    root = Path(root).resolve()
    checkpoint = checkpoint_module.load_checkpoint(root)
    revision = common._load_module("scripts._e1br_prewrite_revision", "run_e1br.py")
    revision.CAMPAIGN, revision.PRIOR_E1B, revision.PRIOR_KNOWN = CAMPAIGN, PRIOR_E1B, PRIOR_KNOWN
    revision.E1B_LIMIT, revision.CUMULATIVE_STOP = E1B_LIMIT, CUMULATIVE_STOP
    revision.ALLOWANCE, revision.PRIOR_DISPATCHES = ALLOWANCE, PRIOR_DISPATCHES
    revision.RETAINED_MAIN, revision.PRIOR_UNKNOWN = RETAINED_MAIN, PRIOR_UNKNOWN
    revision.RETAINED_UNKNOWN_HOLD, revision.BUDGET_CHOICE = RETAINED_UNKNOWN_HOLD, BUDGET_CHOICE
    revision.MAX_WIRE_REQUESTS, revision.NEW_COUNTS = MAX_WIRE_REQUESTS, NEW_COUNTS
    revision.PARENT_MANIFEST_SHA = checkpoint_module.MANIFEST_SHA
    revision.approval = approval
    revision.parent_originals = lambda current: checkpoint_module.inherited_files(current, checkpoint)
    revision.retained_extra_rows = lambda _: deepcopy(checkpoint["retained_attempts"][13:])
    revision.EXTRA_FILES = (
        "scripts/run_e1br_received_continuation.py", "scripts/e1br_received_body.py",
        "scripts/e1br_checkpoint.py", "scripts/e1br_checkpoint_guard.py",
        "scripts/run_e1br_checkpoint_continuation.py", "scripts/e1br_format_checkpoint.py",
        "scripts/e1br_format_runtime.py", "scripts/e1br_format_legacy.py",
        "scripts/e1br_format_qualification.py", "scripts/run_e1br_format_continuation.py",
        "scripts/e1br_memory_checkpoint.py", "scripts/e1br_memory_runtime.py",
        "scripts/e1br_memory_legacy.py", "scripts/e1br_memory_qualification.py",
        "scripts/run_e1br_memory_continuation.py", "docs/E1BR_MEMORY_CONTINUATION.md",
        "scripts/e1br_prewrite_checkpoint.py", "scripts/e1br_prewrite_runtime.py",
        "scripts/run_e1br_prewrite_continuation.py", "docs/E1BR_PREWRITE_CONTINUATION.md",
    )
    positions = frozenset(checkpoint["remaining_positions"])
    if (len(checkpoint["retained_attempts"]) != RETAINED_MAIN or len(positions) != 222
            or Counter(row[0] for row in positions) != Counter(NEW_COUNTS["main"])):
        raise RunError("prewrite_continuation_remaining_position_count_changed")
    revision.position_inventory = lambda: positions
    revision.continuation_details = lambda _: {
        "stopped_checkpoint": deepcopy(checkpoint["summary"]),
        "execution_readiness": "approved_existing_no_content_pool_positive_prewrite_continuation",
        "prewrite_retry_scope": "validated_positive_pre_application_write_failure_original_two_retry_pool",
        "inherited_failed_position_retry_indices": [1, 2],
        "inherited_connection_rebuild_already_used": True,
        "max_infrastructure_retries_total": 443,
        "inherited_target_first_dispatch_is_retry": True,
        "continuation_measurement": "exploratory_mixed_versions_original_gates_unchanged",
        "memory_terminal_policy": "saved_complete_known_cost_json_syntax_failure_no_update_no_retry",
        "memory_format_corrections": 0,
        "future_memory_same_narrow_failure_terminal": True,
        "online_format_failure_policy": "known_cost_complete_first_body_strict_schema_rejected_terminal_fallback_no_retry",
        "remaining_physical_derivation": "222_remaining_including_old_retry1_plus_4_corrections_plus_442_other_retries_plus_old_retry2",
        "unknown_cost_policy": "eleven_unresolved_retained_each_bounded_no_body_attempt_keeps_reservation",
        "measurement_denominators": {"logical_positions": 1056, "reply_positions": 216, "formation_cases": 12},
        "same_final_configuration_effect_claim_eligible": False,
        "prospective_prompt_sources": dict(PROMPT_SOURCES),
    }
    revision.runner_for = lambda current, prefix: checkpoint_module.runner_class(checkpoint)

    def first(client, payload, scope, key, prefix, current):
        checkpoint_module.validate_first_request(checkpoint, payload, scope, key)
        if client.role != "online_fast" or payload.get("response_format") is not None:
            raise RunError("prewrite_first_original_text_route_changed")
        return {"continued_after_prewrite_request_id": checkpoint["failed"]["request_id"],
                "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
                "historical_unknown_hold_retained": True, "historical_body_received": False}

    revision.first_request_lineage = first
    entry = revision.build_engine(root)
    original_preflight = entry.product_configuration_preflight

    def preflight():
        original_preflight()
        for path, expected in PROMPT_SOURCES.items():
            if hashlib.sha256(common.read_original(root, Path(path))).hexdigest() != expected:
                raise RunError("prewrite_continuation_reviewed_prompt_changed")

    format_contract = {"schema": "e1br_known_format_terminal_v1", "retry_answer_bearing_failure": False,
        "effect_claim_eligible": False, "approval_sha256": APPROVAL_SHA,
        "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA, "prompt_namespace_version": ONLINE_ID_NAMESPACE_VERSION}
    memory_contract = {"schema": SCHEMA, "retry_answer_bearing_failure": False, "effect_claim_eligible": False,
        "approval_sha256": APPROVAL_SHA, "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
        "scope": "saved_complete_known_cost_json_syntax_failure_only", "memory_paid_corrections": 0}

    prewrite_contract = {"schema": PREWRITE_SCHEMA, "enabled": True, "approval_sha256": APPROVAL_SHA,
        "parent_campaign": Path(checkpoint_module.PARENT).as_posix(), "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
        "parent_receipt_sha256": prewrite_digest(checkpoint["failed"]), "parent_attempt": deepcopy(checkpoint["failed"]),
        "connection_rebuild_used": True, "inherited_physical_attempts": 1, "remaining_infrastructure_retries": 2}
    validate_prewrite_contract(prewrite_contract)

    class PrewriteGuard(PrewriteGuardMixin, MemoryGuardMixin, entry.EvidenceGuard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, prewrite_contract=deepcopy(prewrite_contract),
                format_contract=deepcopy(format_contract), format_evidence_root=entry.campaign(root),
                memory_contract=deepcopy(memory_contract), memory_evidence_root=entry.campaign(root), **kwargs)

    def assert_clean(state, *, expected_calls=None, final=False):
        rows = [row for row in entry.logical_attempts(state) if row["product_step"] == "initial"]
        complete = len(rows) == 222 and Counter(row["role"] for row in rows) == Counter(NEW_COUNTS["main"])
        final_limit = complete and state.get("stop_reason") in {"request_limit_reached", "inclusive_cost_limit_reached"}
        if ((state.get("stopped") and not final_limit) or len(state["attempts"]) != state["calls_used"]
                or any(row["status"] == "pending" for row in state["attempts"])
                or any(row["status"] != "received" and not terminal_format_failure(row)
                       for row in rows if row["role"] != "reflection")
                or expected_calls is not None and (len(rows) != expected_calls or not complete)):
            raise RunError("prewrite_continuation_primary_or_accounting_incomplete")
        for row in state["attempts"]:
            revision.validate_position_binding(tuple(row.get("experiment_position", ())), row, state["attempts"])
        if Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) > ALLOWANCE:
            raise RunError("prewrite_continuation_inclusive_budget_exceeded")

    def validate_closed_ledger(current, frozen, state):
        if Path(current).resolve() != root:
            raise RunError("prewrite_continuation_workspace_changed")
        ledger = entry.campaign(root) / "allowance.sqlite3"
        rows = validate_prewrite_ledger(ledger, contract=prewrite_contract, stage="main", registration_sha256=frozen["seal_sha256"],
            max_requests=MAX_WIRE_REQUESTS, reservation_policy=frozen["metadata"]["policy"])
        if rows != state["attempts"]:
            raise RunError("prewrite_continuation_closed_ledger_changed")
        validate_format_ledger(ledger, format_contract=format_contract, attempts=rows)
        validate_memory_ledger(ledger, memory_contract=memory_contract, attempts=rows)
        return rows

    def qualify(run_root, frozen, state):
        checkpoint_module.assert_originals(checkpoint, full=True)
        validate_closed_ledger(root, frozen, state)
        entry.check_wire_evidence(state)
        return qualification.qualify(run_root, frozen, state,
            retained_attempts=deepcopy(checkpoint["retained_attempts"]),
            retained_physical_attempts=deepcopy(checkpoint["retained_physical_attempts"]),
            retained_summary=deepcopy(checkpoint["summary"]), logical_attempts=entry.logical_attempts,
            retained_position_bindings=deepcopy(checkpoint["retained_position_bindings"]), expected_retained=RETAINED_MAIN)

    entry.product_configuration_preflight = preflight
    entry.EvidenceGuard, entry.E1bExperimentApp = PrewriteGuard, MemoryExperimentApp
    entry.assert_clean, entry.qualify = assert_clean, qualify
    entry.validate_closed_ledger, entry.stopped_checkpoint = validate_closed_ledger, checkpoint
    entry.format_contract, entry.memory_contract = format_contract, memory_contract
    entry.prewrite_contract = prewrite_contract
    entry.validate_retry_ledger = partial(validate_prewrite_ledger, contract=prewrite_contract)
    inherited_extensions = entry.receipt_extensions
    entry.receipt_extensions = lambda: {**inherited_extensions(), "generation_scope": "exploratory_mixed_versions",
        "old_frozen_gates_replaced": False, "same_final_configuration_effect_claim_eligible": False,
        "memory_failure_policy": "terminal_missing_without_memory_update_or_retry",
        "prewrite_recovery_policy": "original_two_retry_pool_positive_evidence_with_inherited_rebuild_exhaustion"}
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "assess"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-local-env", action="store_true")
    parser.add_argument("--budget-choice", choices=(BUDGET_CHOICE,))
    parser.add_argument("--campaign-authorization-sha256")
    parser.add_argument("--stage-authorization-sha256")
    args = parser.parse_args(argv)
    if args.command == "assess":
        if args.output is None:
            parser.error("assess requires --output")
        value = qualification.assess(ROOT, revision=sys.modules[__name__])
        base.original.write_once(args.output, value)
        print(json.dumps({k: value.get(k) for k in ("status", "main_gate", "formation_gate", "overall_gate")}, indent=2))
        return
    if args.metadata is None or args.registration is None:
        parser.error("prepare/verify/execute require --metadata and --registration")
    entry = build_engine(ROOT)
    if args.command == "prepare":
        value = entry.prepare(ROOT, args.metadata, args.registration, stage="main")
    elif args.command == "verify":
        value = entry.verify(ROOT, args.metadata, args.registration)
    else:
        value = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
            budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
            stage_authorization_sha256=args.stage_authorization_sha256)
        print(json.dumps(value, indent=2))
        return
    print(json.dumps({"seal_sha256": value["seal_sha256"], "paid_calls": 0,
                      "retained_first_results": RETAINED_MAIN, "remaining_basic_positions": 222}))


if __name__ == "__main__":
    main()
