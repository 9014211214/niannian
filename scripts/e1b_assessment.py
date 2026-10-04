"""Read-only E1b assessment after frozen generation and independent labels.

Re-check originals and the retained SQLite ledger before joining local scoring
arithmetic. No client, credential, registration, recovery or paid call is made.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from scripts import e1b_evaluation as evaluation
from scripts import e1b_formation as formation
from scripts import run_e1b as execution


class AssessmentError(ValueError):
    pass


def _require(condition, code):
    if not condition:
        raise AssessmentError(code)


def _registration(root, stage, receipt):
    frozen = evaluation.read_json(execution.campaign(root) / stage / "registration.json")
    _require(frozen["seal_sha256"] == evaluation.digest({k: v for k, v in frozen.items() if k != "seal_sha256"})
             == receipt["registration_sha256"], "saved_registration_seal_mismatch")
    _require(frozen["stage"] == stage and frozen["protocol"] == execution.protocol()
             and frozen["workspace_identity"] == execution.workspace_identity(root), "saved_registration_identity_changed")
    _require(frozen["source"] == execution.inventory(root), "source_changed_use_frozen_checkout")
    _require(frozen["inputs"] == execution.stage_inputs(root, stage), "frozen_phase_inputs_changed")
    return frozen


def _ledger(root, state, frozen):
    path = execution.campaign(root) / "allowance.sqlite3"
    _require(path.is_file() and not path.is_symlink(), "retained_ledger_missing_or_linked")
    _require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
             "retained_ledger_not_closed")
    before = execution.byte_sha(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT request_id,record FROM attempts ORDER BY sequence").fetchall()
        settings = db.execute("SELECT version,max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        policy = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
        identity = db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
        interval = execution.MIN_INTERVAL_SECONDS
        pacing = db.execute("SELECT min_interval_seconds FROM request_pacing WHERE id=1").fetchone() if interval else None
        observed = getattr(execution, "OBSERVE_TRANSPORT", False)
        if observed:
            table = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transport_observation'").fetchone()
            observation = db.execute("SELECT version FROM transport_observation WHERE id=1").fetchone() if table else None
        preparation_recovery = getattr(execution, "CONNECTION_PREPARATION_RECOVERY", False)
        if preparation_recovery:
            table = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='connection_preparation_recovery'").fetchone()
            preparation = db.execute("SELECT version FROM connection_preparation_recovery WHERE id=1").fetchone() if table else None
        recovery = getattr(execution, "RECOVERY_POLICY", None)
        if recovery is not None:
            recovery_policy = db.execute("SELECT policy FROM bounded_recovery_policy WHERE id=1").fetchone()
            fields = ("ordinal", "parent_request_id", "child_request_id", "status", "wait_seconds", "wait_ms")
            relations = [dict(zip(fields, row)) for row in db.execute(
                "SELECT ordinal,parent_request_id,child_request_id,status,wait_seconds,wait_ms FROM authorized_recoveries ORDER BY ordinal")]
    _require(before == execution.byte_sha(path), "ledger_changed_during_assessment")
    records = [json.loads(raw) for _, raw in rows]
    _require(records == state["attempts"] and all(request_id == row["request_id"]
             for (request_id, _), row in zip(rows, records)), "ledger_receipt_mismatch")
    if interval:
        _require(pacing == (interval,) and frozen["protocol"]["transport"]["min_interval_seconds"] == interval
                 and all(row.get("pacing", {}).get("min_interval_seconds") == interval
                         and row["pacing"].get("scope") == "one_guard_all_roles"
                         and row["pacing"].get("clock") == "monotonic" for row in records),
                 "ledger_pacing_contract_changed")
    if observed:
        _require(observed is True and observation == (getattr(execution, "TRANSPORT_VERSION", "connection_trace_v1"),),
                 "ledger_transport_observation_changed")
        execution.validate_transport_records(records, frozen["protocol"])
    if preparation_recovery:
        _require(preparation == ("pre_application_rebuild_v1",)
                 and frozen["protocol"]["transport"].get("connection_preparation_retries") == 1,
                 "ledger_preparation_recovery_changed")
    if recovery is not None:
        _require(recovery_policy == (json.dumps(recovery, sort_keys=True, allow_nan=False),)
                 and frozen["protocol"]["bounded_recovery"] == recovery
                 and relations == state["recovery"]["relations"], "ledger_bounded_recovery_changed")
        execution.assert_clean(state, expected_calls=sum(sum(v.values()) for v in execution.COUNTS.values()), final=True)
    expected_policy = json.dumps(frozen["metadata"]["policy"], sort_keys=True, allow_nan=False)
    _require(policy == (expected_policy,) and identity == (state["ledger_id"],)
             and settings == (1, execution.MAX_WIRE_REQUESTS, state["max_observed_cost_usd"], None), "ledger_policy_or_identity_changed")
    total = sum((Decimal(str(row["usage"]["cost"])) for row in records if "cost" in row.get("usage", {})), Decimal(0))
    held = sum((Decimal(row["reservation"]["reserved_cost_usd"]) for row in records if "cost" not in row.get("usage", {})), Decimal(0))
    _require(total == Decimal(state["known_cost_usd"]) and held == Decimal(state.get("reserved_cost_usd", "0"))
             and total + held <= Decimal(state["max_observed_cost_usd"]),
             "ledger_cost_sum_or_limit_mismatch")
    return before


def _qualification(root, stage, receipt, frozen):
    base = execution.campaign(root) / stage
    run = base / "synthetic-emotion-experiments" / f"e1b-{stage}-v1" / "run"
    checked = execution.qualify(run, frozen, receipt["accounting"])
    _require(checked == receipt["qualification"] == evaluation.read_json(base / "qualification.json"),
             "saved_generation_qualification_changed")
    packet, key = evaluation.load_bundle(base / "blind-bundle")
    _require(evaluation.digest(packet) == receipt["packet_sha256"]
             and evaluation.digest(key) == receipt["mapping_sha256"]
             and key["source_report_sha256"] == checked["source_report_sha256"], "scored_packet_source_mismatch")
    return checked, packet, key


def _secondary_labels(root, packet, state):
    base = execution.campaign(root)
    bundle = base / "main" / "blind-bundle"
    sample = evaluation._load_secondary(bundle, packet)
    labels, _, _ = evaluation._labels(bundle, packet, "secondary")
    items = {item["id"]: item for item in packet["items"]}
    rows = {row["operation_id"]: row for row in execution.logical_attempts(state) if row["role"] == "judge"}
    _require(len(rows) == 40 and set(rows) == {"e1b-judge-" + opaque for opaque in sample["ids"]},
             "secondary_receipt_positions_mismatch")
    for opaque in sample["ids"]:
        first = evaluation.read_json(base / "judge" / "judge" / (opaque + ".first.json"))
        ledger = rows["e1b-judge-" + opaque]
        evaluation._validate_label(first["row"], items[opaque])
        _require(first["id"] == opaque and first["visible_response_sha256"] == ledger["client_return_content_sha256"]
                 and execution.text_sha(first["visible_response"]) == first["visible_response_sha256"]
                 and json.loads(first["visible_response"], object_pairs_hook=evaluation._unique) == first["row"]
                 and first["row"]["id"] == opaque and labels.get(opaque) == first["row"]["labels"],
                 "secondary_first_label_receipt_changed")


def assess(root):
    """Return not_evaluable for incomplete or changed evidence, never zero."""
    result = {"schema": "e1b_formal_assessment_v1", "status": "not_evaluable",
              "main_gate": "not_evaluable", "formation_gate": "not_evaluable",
              "overall_gate": "not_evaluable", "effect_claim_eligible": False,
              "model_calls": 0, "scope": "authored_fictional_explicit_preferences_single_online_model_not_human_calibrated"}
    try:
        root = Path(root).resolve()
        base = execution.campaign(root)
        receipts = {stage: execution.read_receipt(root, stage) for stage in execution.STAGES}
        frozen = {stage: _registration(root, stage, receipts[stage]) for stage in execution.STAGES}
        states = {stage: receipts[stage]["accounting"] for stage in execution.STAGES}
        expected = 0
        prior = []
        for stage in execution.STAGES:
            expected += sum(execution.COUNTS[stage].values())
            state = states[stage]
            execution.assert_clean(state, expected_calls=expected, final=stage == "judge")
            _require(state["attempts"][:len(prior)] == prior, "earlier_phase_accounting_rewritten")
            _require(state["ledger_id"] == states["prescreen"]["ledger_id"]
                     and state["reservation_policy_sha256"] == states["prescreen"]["reservation_policy_sha256"],
                     "phase_ledger_identity_changed")
            rows = [row for row in execution.logical_attempts(state) if row["e1b_stage"] == stage]
            _require(Counter(row["role"] for row in rows) == Counter(execution.COUNTS[stage])
                     and all(row["e1b_stage"] == stage and row["registration_sha256"] == frozen[stage]["seal_sha256"]
                             for row in rows), "phase_inventory_or_registration_changed")
            prior = state["attempts"]
        authorization = evaluation.read_json(base / "authorization.json")
        binding = evaluation.read_json(base / "budget-binding.json")
        for stage in execution.STAGES:
            phase = evaluation.read_json(base / f"{stage}.authorization.json")
            _, limit = execution.validate_authorization(root, frozen[stage], budget_choice="12_usd1",
                campaign_authorization_sha256=evaluation.digest(authorization),
                stage_authorization_sha256=evaluation.digest(phase))
            _require(Decimal(states[stage]["max_observed_cost_usd"]) == limit
                     and binding["campaign_authorization_sha256"] == evaluation.digest(authorization)
                     and binding["source_sha256"] == evaluation.digest(frozen[stage]["source"])
                     and binding["protocol_sha256"] == evaluation.digest(frozen[stage]["protocol"])
                     and binding["policy_sha256"] == evaluation.digest(frozen[stage]["metadata"]["policy"]),
                     "durable_budget_binding_changed")
        result["ledger_sha256"] = _ledger(root, states["judge"], frozen["judge"])
        _qualification(root, "prescreen", receipts["prescreen"], frozen["prescreen"])
        qualification, packet, key = _qualification(root, "main", receipts["main"], frozen["main"])
        formed, formed_key = formation.load_bundle(base / "main" / "formation-bundle")
        _require(evaluation.digest(formed) == receipts["main"]["formation_packet_sha256"]
                 and evaluation.digest(formed_key) == receipts["main"]["formation_mapping_sha256"]
                 and formed_key["protocol"] == key["protocol"], "formation_generation_binding_changed")
        replies = evaluation.aggregate(base / "main" / "blind-bundle")
        formed_result = formation.aggregate(base / "main" / "formation-bundle")
        result.update(reply_evaluation=replies, formation_evaluation=formed_result)
        _require(replies.get("status") == "labels_complete" and replies.get("valid_primary_labels") == 216
                 and replies.get("missing_generation_positions") == 0
                 and formed_result.get("status") == "local_labels_complete", "independent_first_labels_incomplete")
        _secondary_labels(root, packet, states["judge"])
        secondary = replies.get("secondary", {})
        _require(secondary.get("status") == "complete", "secondary_labels_incomplete_or_not_independent")
        result["formation_gate"] = formed_result["mechanical_formation_gate"]
        # Preserve the fixed denominator. A rule-modified reply is not replaced
        # by a new model sample or quietly excluded from the original positions.
        scored = [row for row in qualification["positions"]
                  if row["arm"] in {"A0", "A1"} and row["turn"] in (7, 8, 9)]
        _require(len(scored) == 216, "scored_transport_denominator_changed")
        result["scored_delivery_counts"] = dict(Counter(row["delivery"] for row in scored))
        result["mechanical_delivered_main_gate"] = replies["mechanical_main_gate"]
        _require(all(row["delivery"] in {"model_unmodified", "guard_modified"} for row in scored),
                 "non_model_or_missing_scored_delivery")
        result["raw_model_effect_claim_eligible"] = all(row["model_reply_eligible"] for row in scored)
        result["effect_layer"] = "delivered_product_configuration_including_recorded_guard_modifications"
        if not secondary.get("sufficient_agreement"):
            result.update(status="inconclusive", main_gate="inconclusive_judge_disagreement",
                          overall_gate="inconclusive", reason="secondary_exact_agreement_below_34_of_40")
            return result
        result.update(status="evaluated", main_gate=replies["mechanical_main_gate"], effect_claim_eligible=True)
        result["overall_gate"] = "pass" if result["main_gate"] == result["formation_gate"] == "pass" else "fail"
        result["known_campaign_cost_usd"] = states["judge"]["known_cost_usd"]
        result["historical_unknown_requests_retained"] = execution.ACCEPTED_OLD_UNKNOWN
        result["historical_unknown_reserved_usd"] = str(execution.RETAINED_UNKNOWN_HOLD)
        if getattr(execution, "RECOVERY_POLICY", None) is not None:
            result.update(physical_attempts=states["judge"]["calls_used"], logical_calls=sum(sum(v.values()) for v in execution.COUNTS.values()),
                authorized_recovery=states["judge"]["recovery"], retained_campaign_unknown_usd=states["judge"]["reserved_cost_usd"])
        return result
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        result["reason"] = str(error) if isinstance(error, (AssessmentError, execution.RunError)) else type(error).__name__
        return result
