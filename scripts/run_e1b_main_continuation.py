"""Approved v9: retain the complete prescreen and thirteen main first answers.

Only the failed reflection position is dispatched again, under the equivalent
v2 schema. A fresh bounded ledger covers1043 main and40 judge requests. The
failed v8 ledger and all its originals remain immutable; no automatic resume.
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

import pydantic
import pydantic_core

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts import run_e1b_reflection_repair as reflection
from scripts import e1b_evaluation as evaluation
from aria.engine.policy.bounded_recovery import MAIN_CONTINUATION_SCHEMA, validate_recovery_records
from aria.engine.policy.observed_transport import validate_trace
from aria.engine.policy.reflection_json import REFLECTION_JSON_PROTOCOL_VERSION, reflection_wire_schema
from aria.engine.memory.reflection_contract import reflection_response_schema

CAMPAIGN = Path("artifacts/local/e1b-formal-v9")
PARENT = Path("artifacts/local/e1b-formal-v8")
TASK = Path("artifacts/local/tasks/20261004-03")
PRIOR_KNOWN = Decimal("1.527273986")
PRIOR_E1B = Decimal("0.134894814")
RETAINED_UNKNOWN_HOLD = Decimal("0.163635142")
ALLOWANCE = Decimal("1.201470044")
NEW_COUNTS = {"main": {"online_fast": 996, "memory": 23, "reflection": 24}, "judge": {"judge": 40}}
MAX_WIRE_REQUESTS = 1083
BUDGET_CHOICE = "12_usd1_5_cumulative3"
CANDIDATE_SHA = "15f3553324acf254b4bd184db3eaec072084e238ddbde3b61a57412421022012"
APPROVAL_SHA = "e7a8103caeddf1fd4a63f6b8e2af35bb70ab84ffc55647243819bf8c7ae575c7"
PROPOSAL_SHA = "f1ffa6206ff2fd1f35cf0905a78af6aa25a49210b64b9c9b5acdca06b50f0a8a"
PARENT_MANIFEST = Path("artifacts/local/tasks/20261004-02/v8-main-failure-originals-v1.manifest.json")
PARENT_MANIFEST_SHA = "8cfb6332f0a9dcff3a1db2409f3ec790eadf65344474ee9b3547fc2ba217f996"
RunError = original.RunError


def approval(root):
    path = TASK / "human-main-continuation-approval.json"
    raw = common.read_original(root, path)
    value = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != APPROVAL_SHA or value.get("human_quote") != "批准"
            or value.get("proposal_sha256") != PROPOSAL_SHA):
        raise RunError("specific_main_continuation_approval_changed")
    return {"human_quote": "批准", "human_message_reference": value["human_message_reference"],
            "approved_question": value["approved_question"], "approval_file_sha256": APPROVAL_SHA,
            "proposal_sha256": PROPOSAL_SHA}


def recovery_policy(root):
    return {"schema": MAIN_CONTINUATION_SCHEMA, "max_extra_dispatches": 0, "max_new_unknown": 1,
        "backoff_seconds": [], "max_attempts_per_operation": 1, "max_logical_requests": 1083,
        "authorization_sha256": original.digest(approval(root))}


def parent_originals(root):
    """Bind the complete prescreen, first labels, ledger and frozen v8 source."""
    raw = common.read_original(root, PARENT_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != PARENT_MANIFEST_SHA:
        raise RunError("v8_original_manifest_changed")
    manifest = json.loads(raw)
    records = manifest["files"] + manifest["frozen_archive_files"]
    if len(manifest["files"]) != 615 or len(manifest["frozen_archive_files"]) != 129:
        raise RunError("v8_original_inventory_changed")
    for row in records:
        if hashlib.sha256(common.read_original(root, Path(row["path"]))).hexdigest() != row["sha256"]:
            raise RunError("v8_original_changed:" + row["path"])
    receipt = original.read_json(Path(root) / PARENT / "main/execution-receipt.json")
    state = receipt["accounting"]
    cost = sum((Decimal(str(r["usage"]["cost"])) for r in state["attempts"] if "cost" in r["usage"]), Decimal(0))
    unresolved = [r for r in state["attempts"] if "cost" not in r["usage"]]
    holds = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in unresolved), Decimal(0))
    if (len(state["attempts"]) != 164 or len(unresolved) != 1 or cost != Decimal("0.023220476")
            or holds != Decimal("0.160304") or Decimal(state["known_cost_usd"]) != cost
            or PRIOR_KNOWN != Decimal("1.504053510") + cost or PRIOR_E1B != Decimal("0.111674338") + cost
            or RETAINED_UNKNOWN_HOLD != Decimal("0.003331142") + holds):
        raise RunError("v8_cost_carry_changed")
    selection = evaluation.aggregate(Path(root) / PARENT / "prescreen/blind-bundle")
    if (selection["status"] != "prescreen_labels_complete" or selection["valid_primary_labels"] != 162
            or selection["planned_reply_positions"] != 162 or selection["missing_generation_positions"] != 0
            or selection["selection"]["case_ids"] != [f"e1b_{g}{n}" for g in "abcd" for n in (1, 2, 3)]):
        raise RunError("v8_first_prescreen_labels_or_selection_changed")
    paths = [Path(root) / PARENT_MANIFEST, *(Path(root) / r["path"] for r in records)]
    return paths, tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def build_engine(root):
    from scripts import e1b_main_continuation as retained

    root = Path(root).resolve()
    anchor = approval(root)
    parent_paths, parent_fingerprint = parent_originals(root)
    prefix = retained.load_prefix(root)
    entry = common._load_module("scripts._e1b_main_continuation_execution", "run_e1b.py")
    qualifier = common._load_module("scripts._e1b_main_continuation_qualification", "run_e1b.py")
    entry.RunError = qualifier.RunError = RunError
    entry.CAMPAIGN, entry.PRIOR_KNOWN, entry.CUMULATIVE_STOP = CAMPAIGN, PRIOR_KNOWN, Decimal("3")
    entry.STAGES, entry.COUNTS = ("main", "judge"), {k: dict(v) for k, v in NEW_COUNTS.items()}
    entry.MAX_WIRE_REQUESTS, entry.MIN_INTERVAL_SECONDS = 1083, 10
    entry.ACCEPTED_OLD_UNKNOWN, entry.RETAINED_UNKNOWN_HOLD = 5, RETAINED_UNKNOWN_HOLD
    entry.OBSERVE_TRANSPORT, entry.TRANSPORT_VERSION = True, "connection_trace_v2"
    entry.CONNECTION_PREPARATION_RECOVERY, entry.RECOVERY_POLICY = True, recovery_policy(root)
    entry.FILES = (*entry.FILES, "scripts/run_e1b_repair.py", "scripts/run_e1b_reflection_repair.py",
        "scripts/run_e1b_main_continuation.py", "scripts/e1b_main_continuation.py")
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight
    base_receipt, base_guard, base_clean = entry.read_receipt, entry.EvidenceGuard, entry.assert_clean
    old_attempts = prefix["retained_attempts"]
    old_ids = {r["request_id"]: r for r in old_attempts}
    changed_sources = {"aria/engine/policy/reflection_json.py", "aria/engine/policy/minimal_calls.py",
        "aria/engine/policy/reserved_calls.py", "aria/engine/policy/response_observability.py",
        "aria/engine/policy/bounded_recovery.py", "scripts/run_e1b.py", "docs/E1B_EXECUTION.md"}

    def check_parent(current=root, *, full=False):
        if Path(current).resolve() != root or approval(root) != anchor:
            raise RunError("main_continuation_workspace_or_approval_changed")
        retained.assert_prefix_unchanged(prefix, full=full)
        if (any(p.is_symlink() or any(q.is_symlink() for q in p.parents) for p in parent_paths)
                or tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in parent_paths) != parent_fingerprint):
            raise RunError("v8_originals_changed_during_continuation")
        if full:
            parent_originals(root)

    def preflight():
        base_preflight()
        if (REFLECTION_JSON_PROTOCOL_VERSION != "reflection_json_schema_v2"
                or original.digest(reflection_wire_schema()) != CANDIDATE_SHA
                or sys.version_info[:3] != (3, 13, 11) or pydantic.__version__ != "2.13.4"
                or pydantic_core.__version__ != "2.46.4"
                or ALLOWANCE != min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD):
            raise RunError("approved_schema_or_budget_changed")
        for row in prefix["registration"]["source"]:
            if row["path"] not in changed_sources and original.byte_sha(root / row["path"]) != row["sha256"]:
                raise RunError("retained_generation_source_changed:" + row["path"])
        check_parent()

    def protocol():
        value = base_protocol()
        value.update(schema="aria_e1b_main_continuation_v1", campaign="e1b-formal-v9",
            planned_calls=NEW_COUNTS, planned_total_calls=1083, measurement_calls=original.COUNTS,
            measurement_total_calls=1636, lineage_max_wire_requests=1638, prior_lineage_dispatches=555,
            retained_prescreen=540, retained_main=13, continuation=prefix["summary"], approval=anchor,
            parent_manifest_sha256=PARENT_MANIFEST_SHA,
            runtime={"python": "3.13.11", "pydantic": "2.13.4", "pydantic_core": "2.46.4"},
            bounded_recovery=entry.RECOVERY_POLICY, request_evidence="exact_fictional_body_before_send_no_headers")
        value["product_configuration"].update(reflection_json_protocol_version=REFLECTION_JSON_PROTOCOL_VERSION,
            reflection_wire_schema_sha256=CANDIDATE_SHA,
            reflection_local_schema_sha256=original.digest(reflection_response_schema()),
            reflection_first_answer_capture="e1b_reflection_first_v1")
        value["transport"].update(min_interval_seconds=10, observe_transport=True, observation_version="connection_trace_v2",
            transport_diagnostics=True, application_request_retries=0, connection_preparation_retries=1,
            max_connection_preparations_per_logical_request=2,
            connection_preparation_policy="one_positive_prewrite_eof_or_reset_reconnect_original_deadline_no_http_retry")
        value["budget"].update(status="human_decision_anchored", choice=BUDGET_CHOICE, approved_limit_usd="1.5",
            prior_e1b_known_usd=str(PRIOR_E1B), effective_new_allowance_usd=str(ALLOWANCE),
            accepted_old_unknown=5, inherited_activity_unknown=2, lineage_activity_unknown_limit=3,
            remaining_new_unknown=1, historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD))
        return value

    def authorization_template(current=None):
        return {"schema": "e1b_main_continuation_authorization_v1", "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": "1.5", "cumulative_stop_usd": "3",
            "prior_known_usd": str(PRIOR_KNOWN), "prior_e1b_known_usd": str(PRIOR_E1B),
            "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD), "effective_new_allowance_usd": str(ALLOWANCE),
            "accepted_old_unknown": 5, "inherited_activity_unknown": 2, "max_new_unknown": 1,
            "approval": anchor, "parent_sha256": original.digest(prefix["summary"]),
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None}

    def validate_authorization(current, frozen, *, budget_choice, campaign_authorization_sha256, stage_authorization_sha256):
        preflight()
        check_parent(current)
        if budget_choice != BUDGET_CHOICE or not all(type(v) is str and original.SHA.fullmatch(v)
                for v in (campaign_authorization_sha256, stage_authorization_sha256)):
            raise RunError("specific_main_continuation_hashes_required")
        expected = authorization_template(root)
        expected.update(status="approved", budget_choice=BUDGET_CHOICE)
        auth = entry.read_json(entry.campaign(root) / "authorization.json")
        if auth != expected or original.digest(auth) != campaign_authorization_sha256:
            raise RunError("main_continuation_not_authorized")
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
            "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": campaign_authorization_sha256}
        if (frozen["workspace_identity"] != entry.workspace_identity(root) or frozen["protocol"] != protocol()
                or frozen["source"] != entry.inventory(root)
                or entry.read_json(entry.campaign(root) / f"{frozen['stage']}.authorization.json") != phase
                or original.digest(phase) != stage_authorization_sha256):
            raise RunError("main_continuation_stage_not_authorized")
        return auth, ALLOWANCE

    def stage_inputs(current, stage):
        check_parent(current)
        if stage == "main":
            return {"case_ids": prefix["selected_case_ids"], "retained": prefix["summary"]}
        if stage != "judge":
            raise RunError("prescreen_already_complete_no_new_prescreen")
        receipt = read_receipt(root, "main")
        bundle = entry.campaign(root) / "main/blind-bundle"
        packet, key = evaluation.load_bundle(bundle)
        if original.digest(packet) != receipt["packet_sha256"] or original.digest(key) != receipt["mapping_sha256"]:
            raise RunError("main_bundle_not_bound_to_receipt")
        sample = evaluation._load_secondary(bundle, packet)
        return {"packet_sha256": original.digest(packet), "sample_sha256": original.digest(sample),
            "main_receipt_sha256": receipt["receipt_sha256"], "case_ids": key["protocol"]["case_ids"]}

    def assert_clean(state, *, expected_calls=None, final=False):
        projection = validate_recovery_records(state["attempts"], entry.RECOVERY_POLICY,
            relations=state["recovery"]["relations"], require_complete=True)
        if projection["extra_dispatches"] or state["unknown_cost_requests"]:
            raise RunError("main_continuation_has_no_http_retry_or_unknown_capacity")
        view = dict(state)
        if final and state["stop_reason"] == "logical_request_limit_reached":
            view.update(stopped=False, stop_reason=None)
        base_clean(view, expected_calls=expected_calls, final=final)

    def validate_transport_records(records, frozen_protocol):
        if frozen_protocol["transport"]["connection_preparation_retries"] != 1:
            raise RunError("main_continuation_connection_policy_changed")
        for row in records:
            trace = row.get("transport_events", {})
            if trace.get("version") != "connection_trace_v2" or "connection_preparations" not in trace:
                raise RunError("main_continuation_trace_required")
            validate_trace(trace, require_complete=row["status"] == "received", require_https=True)

    def check_wire_evidence(state):
        counts = Counter()
        for row in state["attempts"]:
            stage = row.get("e1b_stage")
            if stage not in entry.STAGES:
                raise RunError("main_continuation_wire_stage_changed")
            counts[stage] += 1
            rel = Path(stage) / "wire-requests" / f"{counts[stage]:04d}.json"
            if (row.get("wire_request_path") != rel.as_posix()
                    or hashlib.sha256(common.read_original(root, CAMPAIGN / rel)).hexdigest() != row["request_sha256"]):
                raise RunError("main_continuation_wire_changed")
        validate_transport_records(state["attempts"], protocol())

    def combined(state):
        new = [r for r in state["attempts"] if r.get("e1b_stage") == "main"]
        return [*old_attempts, *new, *(r for r in state["attempts"] if r.get("e1b_stage") != "main")]

    def validate_attempt_registration(row, frozen):
        prior = old_ids.get(row.get("request_id"))
        return row == prior if prior is not None else row.get("registration_sha256") == frozen["seal_sha256"]

    qualifier.validate_attempt_registration = validate_attempt_registration

    def qualify(run_root, frozen, state):
        check_parent(full=True)
        check_wire_evidence(state)
        composed = {**state, "attempts": combined(state)}
        result = qualifier.qualify(run_root, frozen, composed)
        result["reflection_originals"] = reflection.reflection_originals(run_root, frozen, composed)
        result["retained_main_prefix"] = prefix["summary"]
        result["new_physical_attempts"] = sum(r["e1b_stage"] == "main" for r in state["attempts"])
        result["retained_first_answers"] = 13
        return result

    def read_receipt(current, stage):
        check_parent(current)
        value = base_receipt(current, stage)
        check_wire_evidence(value["accounting"])
        if value.get("inherited_activity_unknown_requests") != 2 or value["historical_unknown_reserved_usd"] != str(RETAINED_UNKNOWN_HOLD):
            raise RunError("main_continuation_historical_budget_changed")
        return value

    class MainGuard(base_guard):
        def __init__(self, *args, **kwargs):
            for key, wanted in (("recovery_policy", entry.RECOVERY_POLICY), ("observe_transport", True),
                    ("transport_diagnostics", True), ("connection_preparation_recovery", True)):
                if kwargs.pop(key, wanted) != wanted:
                    raise RunError("main_continuation_guard_policy_changed")
                kwargs[key] = wanted
            if "pre_application_check" in kwargs:
                raise RunError("main_continuation_authority_check_replaced")
            kwargs["pre_application_check"] = self.check_authority
            super().__init__(*args, **kwargs)

        def check_authority(self):
            base = entry.campaign(root)
            frozen = entry.read_json(base / self.stage / "registration.json")
            binding = entry.read_json(base / "budget-binding.json")
            started = entry.read_json(base / f"{self.stage}.started.json")
            validate_authorization(root, frozen, budget_choice=BUDGET_CHOICE,
                campaign_authorization_sha256=binding["campaign_authorization_sha256"],
                stage_authorization_sha256=started["stage_authorization_sha256"])

        def _reserve(self, client, request, payload, state):
            self.check_authority()
            if state["unknown_cost_requests"] >= 1:
                raise RunError("main_continuation_unknown_limit_reached")
            first = self.stage == "main" and not state["attempts"]
            if first:
                prior = prefix["failed_attempt"]
                scope = self._local.scope
                if (client.role != "reflection" or scope["operation_id"] != prior["operation_id"]
                        or scope["session_id"] != prior["session_id"]
                        or {k: v for k, v in payload.items() if k != "response_format"}
                        != {k: v for k, v in prefix["failed_payload"].items() if k != "response_format"}):
                    raise RunError("specific_reflection_continuation_request_changed")
            row = super()._reserve(client, request, payload, state)
            sequence = 1 + sum(r.get("e1b_stage") == self.stage for r in state["attempts"])
            rel = Path(self.stage) / "wire-requests" / f"{sequence:04d}.json"
            common.write_wire_body(entry.campaign(root) / rel, request.data)
            row["wire_request_path"] = rel.as_posix()
            if first:
                row.update(continuation_of_request_id=prefix["failed_attempt"]["request_id"],
                    continuation_parent_sha256=original.digest(prefix["summary"]),
                    continuation_schema_sha256=CANDIDATE_SHA)
            return row

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    entry.authorization_template, entry.validate_authorization = authorization_template, validate_authorization
    entry.stage_inputs, entry.assert_clean = stage_inputs, assert_clean
    entry.qualify, entry.read_receipt, entry.EvidenceGuard = qualify, read_receipt, MainGuard
    entry.check_wire_evidence, entry.validate_transport_records = check_wire_evidence, validate_transport_records
    entry.E1bRunner = retained.runner_class(prefix)
    entry.receipt_extensions = lambda: {"inherited_activity_unknown_requests": 2,
        "retained_main_prefix_sha256": original.digest(prefix["summary"]), "prior_main_first_answers": 13}
    entry.retained_prefix, entry.combined_attempts = prefix, combined
    return entry


def assess(root):
    """Re-use the original fixed scoring rules after checking the new ledger."""
    from scripts import e1b_formation as formation

    result = {"schema": "e1b_formal_assessment_v1", "status": "not_evaluable", "main_gate": "not_evaluable",
        "formation_gate": "not_evaluable", "overall_gate": "not_evaluable", "effect_claim_eligible": False,
        "model_calls": 0, "scope": "authored_fictional_explicit_preferences_single_online_model_not_human_calibrated"}
    try:
        entry = build_engine(root)
        module = common._load_module("scripts._e1b_main_continuation_assessment", "e1b_assessment.py")
        module.execution = entry
        base = entry.campaign(root)
        receipts = {s: entry.read_receipt(root, s) for s in entry.STAGES}
        frozen = {s: module._registration(root, s, receipts[s]) for s in entry.STAGES}
        before, after = (receipts[s]["accounting"] for s in entry.STAGES)
        if (after["attempts"][:len(before["attempts"])] != before["attempts"] or before["ledger_id"] != after["ledger_id"]
                or before["reservation_policy_sha256"] != after["reservation_policy_sha256"]):
            raise RunError("main_continuation_prior_accounting_changed")
        for stage, state in (("main", before), ("judge", after)):
            rows = [r for r in state["attempts"] if r.get("e1b_stage") == stage]
            if (Counter(r["role"] for r in rows) != Counter(NEW_COUNTS[stage])
                    or any(r.get("registration_sha256") != frozen[stage]["seal_sha256"] for r in rows)):
                raise RunError("main_continuation_phase_registration_or_count_changed")
        auth, binding = (entry.read_json(base / name) for name in ("authorization.json", "budget-binding.json"))
        for stage in entry.STAGES:
            phase = entry.read_json(base / f"{stage}.authorization.json")
            _, limit = entry.validate_authorization(root, frozen[stage], budget_choice=BUDGET_CHOICE,
                campaign_authorization_sha256=original.digest(auth), stage_authorization_sha256=original.digest(phase))
            if (Decimal(receipts[stage]["accounting"]["max_observed_cost_usd"]) != limit
                    or binding["campaign_authorization_sha256"] != original.digest(auth)
                    or binding["source_sha256"] != original.digest(frozen[stage]["source"])
                    or binding["protocol_sha256"] != original.digest(frozen[stage]["protocol"])
                    or binding["policy_sha256"] != original.digest(frozen[stage]["metadata"]["policy"])):
                raise RunError("main_continuation_budget_binding_changed")
        result["ledger_sha256"] = module._ledger(root, after, frozen["judge"])
        qualification, packet, key = module._qualification(root, "main", receipts["main"], frozen["main"])
        formed, formed_key = formation.load_bundle(base / "main/formation-bundle")
        if (original.digest(formed) != receipts["main"]["formation_packet_sha256"]
                or original.digest(formed_key) != receipts["main"]["formation_mapping_sha256"]
                or formed_key["protocol"] != key["protocol"]):
            raise RunError("formation_generation_binding_changed")
        replies = evaluation.aggregate(base / "main/blind-bundle")
        formed_result = formation.aggregate(base / "main/formation-bundle")
        result.update(reply_evaluation=replies, formation_evaluation=formed_result)
        if (replies.get("status") != "labels_complete" or replies.get("valid_primary_labels") != 216
                or replies.get("missing_generation_positions") != 0 or formed_result.get("status") != "local_labels_complete"):
            raise RunError("independent_first_labels_incomplete")
        module._secondary_labels(root, packet, after)
        secondary = replies.get("secondary", {})
        if secondary.get("status") != "complete":
            raise RunError("secondary_labels_incomplete_or_not_independent")
        result["formation_gate"] = formed_result["mechanical_formation_gate"]
        scored = [row for row in qualification["positions"] if row["arm"] in {"A0", "A1"} and row["turn"] in (7, 8, 9)]
        if len(scored) != 216 or any(row["delivery"] not in {"model_unmodified", "guard_modified"} for row in scored):
            raise RunError("scored_transport_denominator_or_delivery_changed")
        result.update(scored_delivery_counts=dict(Counter(row["delivery"] for row in scored)),
            mechanical_delivered_main_gate=replies["mechanical_main_gate"],
            raw_model_effect_claim_eligible=all(row["model_reply_eligible"] for row in scored),
            effect_layer="delivered_product_configuration_including_recorded_guard_modifications")
        if not secondary.get("sufficient_agreement"):
            result.update(status="inconclusive", main_gate="inconclusive_judge_disagreement", overall_gate="inconclusive",
                reason="secondary_exact_agreement_below_34_of_40")
            return result
        cost = Decimal(after["known_cost_usd"])
        result.update(status="evaluated", main_gate=replies["mechanical_main_gate"], effect_claim_eligible=True,
            known_campaign_cost_usd=str(cost), total_e1b_known_cost_usd=str(PRIOR_E1B + cost),
            cumulative_known_cost_usd=str(PRIOR_KNOWN + cost), historical_unknown_requests_retained=7,
            historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD), physical_attempts=after["calls_used"],
            lineage_physical_attempts=555 + after["calls_used"], measurement_logical_calls=1636)
        result["overall_gate"] = "pass" if result["main_gate"] == result["formation_gate"] == "pass" else "fail"
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        result["reason"] = str(error) if isinstance(error, (RunError, module.AssessmentError if 'module' in locals() else RunError)) else type(error).__name__
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("assess")
    report.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--stage", choices=("main", "judge"), default="main")
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
            result = {"stage": args.stage, "seal_sha256": value["seal_sha256"], "authorization": entry.authorization_template(ROOT), "paid_calls": 0}
        elif args.command == "verify":
            value = entry.verify(ROOT, args.metadata, args.registration)
            result = {"stage": value["stage"], "seal_sha256": value["seal_sha256"], "paid_calls": 0}
        else:
            result = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
                budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
                stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
