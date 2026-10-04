"""Authorized formal-v8: preserve 390 v7 answers and generate only the missing 150.

The explicit human continuation decision is bound below; each actual stage
still requires matching frozen source and durable authorization records.
"""
from __future__ import annotations

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
from scripts import run_e1b_observed_repair as observed
from scripts import run_e1b_reflection_repair as reflection
from scripts import run_e1b_bounded_repair as v7
from scripts import e1b_composed_evidence as evidence

CAMPAIGN = Path("artifacts/local/e1b-formal-v8")
PRIOR_KNOWN = Decimal("1.504053510")
PRIOR_E1B = Decimal("0.111674338")
RETAINED_UNKNOWN_HOLD = Decimal("0.003331142")
ACCEPTED_OLD_UNKNOWN = 5
INHERITED_ACTIVITY_UNKNOWN = 1
ALLOWANCE = Decimal("1.384994520")
MAX_LOGICAL_REQUESTS = 1246
MAX_WIRE_REQUESTS = 1247
MIN_INTERVAL_SECONDS = 10
NEW_COUNTS = {"prescreen": {"online_fast": 150}, "main": dict(original.COUNTS["main"]), "judge": {"judge": 40}}
MEASUREMENT_COUNTS = {k: dict(v) for k, v in original.COUNTS.items()}
BUDGET_CHOICE = v7.BUDGET_CHOICE
HUMAN_BUDGET_APPROVAL = dict(v7.HUMAN_BUDGET_APPROVAL)
HUMAN_CONTINUATION_APPROVAL = {
    "status": "approved",
    "terms_sha256": "f0f96eec09a340bd6633d45426a40e6f5d479801cad999da56dbf7f264ce948a",
    "human_quote": "保留390首答，补150轮（含失败位置再发），启用限定的写入前一次连接重建，随后继续主实验和Claude复核。原费用上限与未知额度不变。我需要你进入主实验",
    "human_message_reference": "20261004-02:user:explicit_composed_continuation_and_main",
}
RunError = original.RunError


def continuation_terms():
    return {"schema": "e1b_v8_continuation_terms_v1", "campaign": CAMPAIGN.as_posix(),
        "donor_campaign": evidence.PARENT.as_posix(), "donor_receipt_sha256": evidence.PARENT_RECEIPT,
        "donor_tree_sha256": evidence.PARENT_TREE, "retained_received": 390,
        "retained_case_ids": list(evidence.RETAINED_CASES), "remaining_case_ids": list(evidence.REMAINING_CASES),
        "specific_failed_request_id": evidence.FAILED_REQUEST, "specific_failed_wire_sha256": evidence.FAILED_WIRE,
        "specific_coordinate": {"case_id": "e1b_e2", "arm": "A0", "repeat": 1, "turn": 0},
        "measurement_calls": 1636, "measurement_prescreen_calls": 540, "prescreen_scoring_positions": 162,
        "new_calls": 1246, "new_wire_limit": 1247, "lineage_wire_limit": 1638,
        "manual_reattempt_extra_consumed": 1, "remaining_429_extras": 1, "remaining_backoff_seconds": [120],
        "connection_preparation_policy": "one_positive_prewrite_eof_or_reset_reconnect_original_deadline_no_http_retry",
        "max_connection_preparations_per_logical_request": 2,
        "inherited_activity_unknown": 1, "remaining_new_unknown": 1, "lineage_activity_unknown_limit": 2,
        "stop_before_dispatch_at_unknown_limit": True, "all_holds_retained": True,
        "user_limit_usd": "1.5", "cumulative_stop_usd": "3", "effective_new_allowance_usd": str(ALLOWANCE),
        "scope": "specific_tls_position_and_fixed_390_plus_150_composition_no_general_tls_retry_no_label_reuse"}


def _continuation_anchor(*, required=False):
    anchor = HUMAN_CONTINUATION_APPROVAL
    if anchor is None and not required:
        return {"status": "pending", "terms_sha256": original.digest(continuation_terms()),
                "human_quote": None, "human_message_reference": None}
    if (type(anchor) is not dict or set(anchor) != {"status", "terms_sha256", "human_quote", "human_message_reference"}
            or anchor.get("status") != "approved" or anchor.get("terms_sha256") != original.digest(continuation_terms())
            or any(type(anchor.get(k)) is not str or not anchor[k].strip() for k in ("human_quote", "human_message_reference"))):
        raise RunError("specific_composition_and_tls_reattempt_approval_required")
    return anchor


def recovery_policy():
    return {"schema": "bounded_http429_continuation_v1", "max_extra_dispatches": 1,
        "max_new_unknown": 1, "backoff_seconds": [120], "max_attempts_per_operation": 2,
        "max_logical_requests": 1246, "authorization_sha256": original.digest(_continuation_anchor())}


def build_engine(root):
    from aria.engine.policy.bounded_recovery import validate_recovery_records
    from aria.engine.policy.observed_transport import validate_trace

    root = Path(root).resolve()
    donor = evidence.load_parent(root)
    checked_fingerprint = donor["fingerprint"]

    def parent_evidence(current):
        if Path(current).resolve() != root or evidence.fingerprint(root) != checked_fingerprint:
            raise RunError("retained_donor_changed_during_execution")
        return donor["summary"]

    wrapper = common._load_module("scripts._e1b_composed_structure", "run_e1b_structure_repair.py")
    for key in ("CAMPAIGN", "PRIOR_KNOWN", "PRIOR_E1B", "ALLOWANCE", "HUMAN_BUDGET_APPROVAL"):
        setattr(wrapper, key, globals()[key])
    wrapper.parent_evidence = parent_evidence
    entry = wrapper.build_engine(root)
    entry.COUNTS = {k: dict(v) for k, v in NEW_COUNTS.items()}
    entry.MAX_WIRE_REQUESTS = MAX_WIRE_REQUESTS
    entry.ACCEPTED_OLD_UNKNOWN = ACCEPTED_OLD_UNKNOWN
    entry.RETAINED_UNKNOWN_HOLD = RETAINED_UNKNOWN_HOLD
    entry.MIN_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
    entry.OBSERVE_TRANSPORT = True
    entry.TRANSPORT_DIAGNOSTICS = True
    entry.TRANSPORT_VERSION = "connection_trace_v2"
    entry.CONNECTION_PREPARATION_RECOVERY = True
    entry.RECOVERY_POLICY = recovery_policy()
    entry.FILES = (*entry.FILES, "scripts/run_e1b_pacing_repair.py", "scripts/run_e1b_observed_repair.py",
                   "scripts/run_e1b_bounded_repair.py", "scripts/run_e1b_composed_repair.py", "scripts/e1b_composed_evidence.py")
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight
    base_inputs, base_qualify, base_receipt, base_guard = entry.stage_inputs, entry.qualify, entry.read_receipt, entry.EvidenceGuard

    def protocol():
        value = base_protocol()
        value.update(schema="aria_e1b_composed_execution_v1", campaign="e1b-formal-v8",
            planned_calls=NEW_COUNTS, planned_total_calls=MAX_LOGICAL_REQUESTS,
            measurement_calls=MEASUREMENT_COUNTS, measurement_total_calls=1636,
            max_wire_requests=MAX_WIRE_REQUESTS, lineage_max_wire_requests=1638,
            measurement="prospectively_registered_fixed_donor_prefix_plus_remaining_cases",
            continuation=continuation_terms(), continuation_approval=_continuation_anchor(), bounded_recovery=recovery_policy())
        value["transport"].update(observed.transport_contract(), observation_version="connection_trace_v2",
            transport_diagnostics=True, min_interval_seconds=MIN_INTERVAL_SECONDS,
            pacing_scope="one_shared_durable_guard_all_roles", pacing_clock="monotonic",
            application_request_retries=1, retry_after="remaining_authorized_429_only_120_seconds",
            connection_preparation_retries=1, max_connection_preparations_per_logical_request=2,
            connection_preparation_policy=continuation_terms()["connection_preparation_policy"])
        value["budget"].update(accepted_old_unknown=5, inherited_activity_unknown=1,
            old_accepted_unknown_hold_usd="0.002638320", inherited_activity_unknown_hold_usd="0.000692822",
            historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD), effective_new_allowance_usd=str(ALLOWANCE),
            remaining_new_unknown=1, lineage_activity_unknown_limit=2, completion_within_budget="not_guaranteed")
        return value

    def preflight():
        base_preflight()
        evidence.validate_product_compatibility(root, donor["registration"])
        if (ALLOWANCE != min(Decimal("1.5") - PRIOR_E1B, Decimal("3") - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD
                or PRIOR_KNOWN != v7.PRIOR_KNOWN + evidence.PARENT_COST
                or PRIOR_E1B != v7.PRIOR_E1B + evidence.PARENT_COST
                or RETAINED_UNKNOWN_HOLD != v7.RETAINED_UNKNOWN_HOLD + evidence.PARENT_HOLD
                or entry.COUNTS != NEW_COUNTS or entry.MAX_WIRE_REQUESTS != 1247
                or tuple(c.id for c in entry.load_e1b_cases().items) != evidence.RETAINED_CASES + evidence.REMAINING_CASES):
            raise RunError("fixed_continuation_inventory_or_budget_changed")
        _continuation_anchor()

    def authorization_template(current=None):
        return {"schema": "e1b_composed_authorization_v1", "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": None, "cumulative_stop_usd": None,
            "prior_known_usd": str(PRIOR_KNOWN), "prior_e1b_known_usd": str(PRIOR_E1B),
            "accepted_old_unknown": 5, "inherited_activity_unknown": 1,
            "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD), "effective_new_allowance_usd": str(ALLOWANCE),
            "continuation_terms_sha256": original.digest(continuation_terms()),
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None,
            "human_quote": None, "human_message_reference": None, "human_budget_approval_sha256": None,
            "continuation_human_quote": None, "continuation_human_message_reference": None,
            "human_continuation_approval_sha256": None}

    def validate_authorization(current, frozen, *, budget_choice, campaign_authorization_sha256, stage_authorization_sha256):
        anchor = _continuation_anchor(required=True)
        preflight()
        parent_evidence(current)
        if budget_choice != BUDGET_CHOICE or not all(type(v) is str and original.SHA.fullmatch(v)
                for v in (campaign_authorization_sha256, stage_authorization_sha256)):
            raise RunError("explicit_continuation_budget_and_hashes_required")
        expected = authorization_template(root)
        expected.update(status="approved", budget_choice=BUDGET_CHOICE, user_limit_usd="1.5", cumulative_stop_usd="3",
            human_quote=HUMAN_BUDGET_APPROVAL["human_quote"], human_message_reference=HUMAN_BUDGET_APPROVAL["human_message_reference"],
            human_budget_approval_sha256=original.digest(HUMAN_BUDGET_APPROVAL),
            continuation_human_quote=anchor["human_quote"], continuation_human_message_reference=anchor["human_message_reference"],
            human_continuation_approval_sha256=original.digest(anchor))
        auth = entry.read_json(entry.campaign(root) / "authorization.json")
        if auth != expected or original.digest(auth) != campaign_authorization_sha256:
            raise RunError("campaign_not_explicitly_authorized")
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
                 "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": campaign_authorization_sha256}
        if (frozen["workspace_identity"] != entry.workspace_identity(root) or frozen["protocol"] != protocol()
                or entry.read_json(entry.campaign(root) / f"{frozen['stage']}.authorization.json") != phase
                or original.digest(phase) != stage_authorization_sha256):
            raise RunError("stage_not_explicitly_authorized")
        return auth, ALLOWANCE

    def stage_inputs(current, stage):
        if stage == "prescreen":
            return {"case_ids": list(evidence.REMAINING_CASES),
                "measurement_case_ids": list(evidence.RETAINED_CASES + evidence.REMAINING_CASES),
                "donor_qualification_sha256": original.digest(donor["qualification"]),
                "specific_reattempt": continuation_terms()["specific_coordinate"]}
        return base_inputs(current, stage)

    def projection(state, *, complete=False):
        return validate_recovery_records(state["attempts"], recovery_policy(),
            relations=state.get("recovery", {}).get("relations", []), require_complete=complete)

    def assert_clean(state, *, expected_calls=None, final=False):
        projected = projection(state, complete=True)
        acceptable = final and state.get("stop_reason") in {"request_limit_reached", "logical_request_limit_reached"}
        if ((state.get("stopped") and not acceptable) or len(state["attempts"]) != state["calls_used"]
                or state["calls_used"] > MAX_WIRE_REQUESTS or state.get("unknown_cost_requests") != 0
                or projected["new_unknown_requests"] != 0
                or any(state.get("recovery", {}).get(k) != v for k, v in projected.items() if k != "logical_attempts")
                or state["recovery"]["policy_sha256"] != hashlib.sha256(json.dumps(recovery_policy(), sort_keys=True, allow_nan=False).encode()).hexdigest()
                or expected_calls is not None and projected["logical_received"] != expected_calls
                or any("cost" not in row.get("usage", {}) for row in projected["logical_attempts"])):
            raise RunError("continuation_incomplete_unknown_or_inventory_changed")

    def validate_transport_records(records, frozen_protocol):
        if (frozen_protocol["transport"].get("observation_version") != "connection_trace_v2"
                or frozen_protocol["transport"].get("transport_diagnostics") is not True
                or frozen_protocol["transport"].get("connection_preparation_retries") != 1
                or frozen_protocol["transport"].get("max_connection_preparations_per_logical_request") != 2
                or frozen_protocol["transport"].get("connection_preparation_policy") != continuation_terms()["connection_preparation_policy"]):
            raise RunError("continuation_transport_diagnostics_required")
        for row in records:
            trace = row.get("transport_events")
            if (not isinstance(trace, dict) or trace.get("version") != "connection_trace_v2"
                    or "connection_preparations" not in trace):
                raise RunError("continuation_trace_version_changed")
            validate_trace(trace, require_complete=row["status"] == "received", require_https=True)

    class ContinuationGuard(base_guard):
        def __init__(self, *args, **kwargs):
            for name, value in (("recovery_policy", recovery_policy()), ("observe_transport", True),
                                ("transport_diagnostics", True), ("connection_preparation_recovery", True)):
                if kwargs.pop(name, value) != value:
                    raise RunError("continuation_guard_contract_changed")
                kwargs[name] = value
            if "pre_application_check" in kwargs:
                raise RunError("continuation_source_check_cannot_be_replaced")
            kwargs["pre_application_check"] = self._check_frozen_authority
            super().__init__(*args, **kwargs)

        def _check_frozen_authority(self):
            base = entry.campaign(root)
            frozen = entry.read_json(base / self.stage / "registration.json")
            if frozen["source"] != entry.inventory(root):
                raise RunError("frozen_source_changed_before_application_write")
            binding = entry.read_json(base / "budget-binding.json")
            started = entry.read_json(base / f"{self.stage}.started.json")
            validate_authorization(root, frozen, budget_choice=BUDGET_CHOICE,
                campaign_authorization_sha256=binding["campaign_authorization_sha256"],
                stage_authorization_sha256=started["stage_authorization_sha256"])

        def _reserve(self, client, request, payload, state):
            self._check_frozen_authority()
            if state["unknown_cost_requests"] >= 1:
                raise RunError("inherited_activity_unknown_limit_reached")
            first = self.stage == "prescreen" and not state["attempts"]
            if first and (client.role != "online_fast" or hashlib.sha256(request.data).hexdigest() != evidence.FAILED_WIRE):
                raise RunError("specific_tls_reattempt_wire_changed")
            value = super()._reserve(client, request, payload, state)
            if first:
                value.update(continuation_of_request_id=evidence.FAILED_REQUEST,
                             continuation_coordinate=continuation_terms()["specific_coordinate"],
                             continuation_terms_sha256=original.digest(continuation_terms()))
            return value

        def _finish(self, row):
            super()._finish(row)
            validate_transport_records([row], protocol())

    def refresh_donor():
        refreshed = evidence.load_parent(root)
        if refreshed["summary"] != donor["summary"]:
            raise RunError("composed_donor_changed_at_final_check")
        return refreshed

    def prepare_report_view(run_root, frozen, state):
        if frozen["stage"] == "prescreen":
            return evidence.compose_view(root, run_root, refresh_donor())
        return run_root

    def qualify(run_root, frozen, state):
        result = base_qualify(run_root, frozen, state)
        if frozen["stage"] == "prescreen":
            result = evidence.qualify_view(root, run_root, frozen, state, result, donor)
        result["physical_attempts"] = sum(r["e1b_stage"] == frozen["stage"] for r in state["attempts"])
        result["inherited_activity_unknown_requests"] = 1
        return result

    def read_receipt(current, stage):
        value = base_receipt(current, stage)
        if (value["old_unknown_requests_retained"] != 5
                or value["historical_unknown_reserved_usd"] != str(RETAINED_UNKNOWN_HOLD)
                or value.get("inherited_activity_unknown_requests") != 1):
            raise RunError("continuation_unknown_carry_changed")
        validate_transport_records(value["accounting"]["attempts"], protocol())
        return value

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    entry.authorization_template, entry.validate_authorization = authorization_template, validate_authorization
    entry.stage_inputs, entry.logical_attempts = stage_inputs, lambda state: projection(state)["logical_attempts"]
    entry.assert_clean, entry.EvidenceGuard = assert_clean, ContinuationGuard
    entry.validate_transport_records = validate_transport_records
    entry.prepare_report_view, entry.qualify, entry.read_receipt = prepare_report_view, qualify, read_receipt
    entry.export_protocol = lambda frozen: entry.evaluation.draft_protocol(entry.load_e1b_cases(), phase=frozen["stage"],
        case_ids=list(evidence.RETAINED_CASES + evidence.REMAINING_CASES) if frozen["stage"] == "prescreen" else frozen["inputs"]["case_ids"])
    entry.receipt_extensions = lambda: {"inherited_activity_unknown_requests": 1, "old_accepted_unknown_reserved_usd": "0.002638320",
        "inherited_activity_unknown_reserved_usd": "0.000692822", "donor_receipt_sha256": evidence.PARENT_RECEIPT}
    return entry


def assess(root):
    try:
        entry = build_engine(root)
        module = common._load_module("scripts._e1b_composed_assessment", "e1b_assessment.py")
        module.execution = reflection.assessment_engine(entry)
        result = module.assess(root)
        result.update(prior_e1b_known_cost_usd=str(PRIOR_E1B), historical_unknown_requests_retained=5,
            inherited_activity_unknown_requests=1, historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD),
            composition="v7_fixed_390_plus_v8_remaining_150_not_v7_completion")
        if "known_campaign_cost_usd" in result:
            cost = Decimal(result["known_campaign_cost_usd"])
            result.update(total_e1b_known_cost_usd=str(PRIOR_E1B + cost), cumulative_known_cost_usd=str(PRIOR_KNOWN + cost),
                new_logical_calls=1246, measurement_logical_calls=1636,
                lineage_physical_attempts=391 + result["physical_attempts"])
        return result
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        return {"schema": "e1b_formal_assessment_v1", "status": "not_evaluable", "effect_claim_eligible": False,
                "main_gate": "not_evaluable", "formation_gate": "not_evaluable", "overall_gate": "not_evaluable",
                "model_calls": 0, "reason": str(error) if isinstance(error, RunError) else type(error).__name__}


def main(argv=None):
    wrapper = common._load_module("scripts._e1b_composed_cli", "run_e1b_observed_repair.py")
    wrapper.ROOT, wrapper.build_engine, wrapper.assess = ROOT, build_engine, assess
    return wrapper.main(argv)


if __name__ == "__main__":
    main()
