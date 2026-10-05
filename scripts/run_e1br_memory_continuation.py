"""Approved, separately frozen continuation with missing-memory terminals.

Keeps 731 first results and dispatches only 325 remaining basic positions.
Known, saved JSON syntax failures stay missing; no memory answer is retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1br as base
from scripts import run_e1b_repair as common
from scripts import e1br_memory_checkpoint as checkpoint_module
from scripts import e1br_memory_qualification as qualification
from scripts.e1br_memory_runtime import MemoryGuardMixin, MemoryExperimentApp, validate_memory_ledger, SCHEMA
from scripts.e1br_format_runtime import terminal_format_failure, validate_format_ledger
from aria.engine.policy.e1br_calls import validate_e1br_ledger
from aria.engine.policy.online_decision import ONLINE_ID_NAMESPACE_VERSION

CAMPAIGN = Path("artifacts/local/e1br-formal-v5")
PRIOR_E1B, PRIOR_KNOWN = Decimal("0.880939320"), Decimal("2.273318492")
E1B_LIMIT, CUMULATIVE_STOP = Decimal("2"), Decimal("3.5")
RETAINED_UNKNOWN_HOLD = Decimal("0.324948616")
ALLOWANCE = Decimal("0.794112064")
PRIOR_DISPATCHES, RETAINED_MAIN, PRIOR_UNKNOWN = 1277, 731, 10
MAX_WIRE_REQUESTS = 982
NEW_COUNTS = {"main": {"online_fast": 312, "memory": 6, "reflection": 7}}
BUDGET_CHOICE = "e1br_12_codex_only_usd2_cumulative3_5_memory_terminal"
SELECTED, RunError = base.SELECTED, base.RunError
APPROVAL_PATH = Path("artifacts/local/tasks/20261005-03/human-memory-continuation-approval-v1.json")
APPROVAL_SHA = "b64402143e1ba2e1fc54f48cf896f723984649b12454b6cb7fb1c18c0da4728e"
PROMPT_SOURCES = {
    "aria/engine/policy/online_json.py": "a5f653b43d01bfe73ff915f9a716bf996876bcb981c2dafe258ff5c1f6b63bf7",
    "aria/engine/memory/consolidator.py": "9f07bea3196a16c3fa9abaddaa339179c91966522cf4a17181d03b606343dde2",
}


def approval(root):
    raw = common.read_original(Path(root), APPROVAL_PATH)
    if hashlib.sha256(raw).hexdigest() != APPROVAL_SHA:
        raise RunError("memory_continuation_human_approval_changed")
    value = json.loads(raw)
    if (not value.get("human_message_verbatim") or value.get("e1b_inclusive_cap_usd") != str(E1B_LIMIT)
            or value.get("cumulative_inclusive_stop_usd") != str(CUMULATIVE_STOP)
            or value.get("effective_new_allowance_usd") != str(ALLOWANCE)
            or value.get("remaining_basic") != 325 or value.get("retained_first_results") != RETAINED_MAIN):
        raise RunError("memory_continuation_approved_scope_changed")
    proposal = common.read_original(Path(root), Path(value["approved_proposal_path"]))
    if hashlib.sha256(proposal).hexdigest() != value["approved_proposal_sha256"]:
        raise RunError("memory_continuation_approved_proposal_changed")
    return {"approval_file_sha256": APPROVAL_SHA, "human_quote": value["human_message_verbatim"],
            "human_message_reference": "20261005-03", "proposal_sha256": value["approved_proposal_sha256"],
            "scope": value["scope"]}


def build_engine(root):
    root = Path(root).resolve()
    checkpoint = checkpoint_module.load_checkpoint(root)
    revision = common._load_module("scripts._e1br_memory_revision", "run_e1br.py")
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
    )
    positions = frozenset(checkpoint["remaining_positions"])
    if (len(checkpoint["retained_attempts"]) != RETAINED_MAIN or len(positions) != 325
            or Counter(row[0] for row in positions) != Counter(NEW_COUNTS["main"])):
        raise RunError("memory_continuation_remaining_position_count_changed")
    revision.position_inventory = lambda: positions
    revision.continuation_details = lambda _: {
        "stopped_checkpoint": deepcopy(checkpoint["summary"]),
        "execution_readiness": "approved_memory_terminal_missing_undispatched_only",
        "continuation_measurement": "exploratory_mixed_versions_original_gates_unchanged",
        "memory_terminal_policy": "saved_complete_known_cost_json_syntax_failure_no_update_no_retry",
        "memory_format_corrections": 0,
        "future_memory_same_narrow_failure_terminal": True,
        "online_format_failure_policy": "known_cost_complete_first_body_strict_schema_rejected_terminal_fallback_no_retry",
        "remaining_physical_derivation": "325_initial_plus_7_reflection_corrections_plus_650_shared_no_body_retries",
        "unknown_cost_policy": "ten_unresolved_retained_each_bounded_no_body_attempt_keeps_reservation",
        "measurement_denominators": {"logical_positions": 1056, "reply_positions": 216, "formation_cases": 12},
        "same_final_configuration_effect_claim_eligible": False,
        "prospective_prompt_sources": dict(PROMPT_SOURCES),
    }
    revision.runner_for = lambda current, prefix: checkpoint_module.runner_class(checkpoint)

    def first(client, payload, scope, key, prefix, current):
        checkpoint_module.validate_first_request(checkpoint, payload, scope, key)
        if client.role != "reflection" or payload.get("response_format") is not None:
            raise RunError("memory_continuation_first_reflection_protocol_changed")
        return {"continued_after_terminal_request_id": checkpoint["failed"]["request_id"],
                "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
                "old_memory_retried": False, "old_memory_failure_replaced": False}

    revision.first_request_lineage = first
    entry = revision.build_engine(root)
    original_preflight = entry.product_configuration_preflight

    def preflight():
        original_preflight()
        for path, expected in PROMPT_SOURCES.items():
            if hashlib.sha256(common.read_original(root, Path(path))).hexdigest() != expected:
                raise RunError("memory_continuation_reviewed_prompt_changed")

    format_contract = {"schema": "e1br_known_format_terminal_v1", "retry_answer_bearing_failure": False,
        "effect_claim_eligible": False, "approval_sha256": APPROVAL_SHA,
        "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA, "prompt_namespace_version": ONLINE_ID_NAMESPACE_VERSION}
    memory_contract = {"schema": SCHEMA, "retry_answer_bearing_failure": False, "effect_claim_eligible": False,
        "approval_sha256": APPROVAL_SHA, "parent_manifest_sha256": checkpoint_module.MANIFEST_SHA,
        "scope": "saved_complete_known_cost_json_syntax_failure_only", "memory_paid_corrections": 0}

    class MemoryGuard(MemoryGuardMixin, entry.EvidenceGuard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, format_contract=deepcopy(format_contract), format_evidence_root=entry.campaign(root),
                memory_contract=deepcopy(memory_contract), memory_evidence_root=entry.campaign(root), **kwargs)

    def assert_clean(state, *, expected_calls=None, final=False):
        rows = [row for row in entry.logical_attempts(state) if row["product_step"] == "initial"]
        complete = len(rows) == 325 and Counter(row["role"] for row in rows) == Counter(NEW_COUNTS["main"])
        final_limit = complete and state.get("stop_reason") in {"request_limit_reached", "inclusive_cost_limit_reached"}
        if ((state.get("stopped") and not final_limit) or len(state["attempts"]) != state["calls_used"]
                or any(row["status"] == "pending" for row in state["attempts"])
                or any(row["status"] != "received" and not terminal_format_failure(row)
                       for row in rows if row["role"] != "reflection")
                or expected_calls is not None and (len(rows) != expected_calls or not complete)):
            raise RunError("memory_continuation_primary_or_accounting_incomplete")
        for row in state["attempts"]:
            revision.validate_position_binding(tuple(row.get("experiment_position", ())), row, state["attempts"])
        if Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) > ALLOWANCE:
            raise RunError("memory_continuation_inclusive_budget_exceeded")

    def validate_closed_ledger(current, frozen, state):
        if Path(current).resolve() != root:
            raise RunError("memory_continuation_workspace_changed")
        ledger = entry.campaign(root) / "allowance.sqlite3"
        rows = validate_e1br_ledger(ledger, stage="main", registration_sha256=frozen["seal_sha256"],
            max_requests=MAX_WIRE_REQUESTS, reservation_policy=frozen["metadata"]["policy"])
        if rows != state["attempts"]:
            raise RunError("memory_continuation_closed_ledger_changed")
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
            retained_position_bindings=deepcopy(checkpoint["retained_position_bindings"]))

    entry.product_configuration_preflight = preflight
    entry.EvidenceGuard, entry.E1bExperimentApp = MemoryGuard, MemoryExperimentApp
    entry.assert_clean, entry.qualify = assert_clean, qualify
    entry.validate_closed_ledger, entry.stopped_checkpoint = validate_closed_ledger, checkpoint
    entry.format_contract, entry.memory_contract = format_contract, memory_contract
    inherited_extensions = entry.receipt_extensions
    entry.receipt_extensions = lambda: {**inherited_extensions(), "generation_scope": "exploratory_mixed_versions",
        "old_frozen_gates_replaced": False, "same_final_configuration_effect_claim_eligible": False,
        "memory_failure_policy": "terminal_missing_without_memory_update_or_retry"}
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
                      "retained_first_results": RETAINED_MAIN, "remaining_basic_positions": 325}))


if __name__ == "__main__":
    main()
