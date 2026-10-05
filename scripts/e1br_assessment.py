"""Read-only E1b-R assessment; no secondary-model gate or paid calls."""
from collections import Counter
from decimal import Decimal
from pathlib import Path
import sqlite3
from aria.engine.policy.e1br_calls import validate_e1br_ledger

from scripts import e1b_evaluation as evaluation
from scripts import e1b_formation as formation
from scripts import run_e1b_repair as common
from scripts import run_e1br as revision


def retained_accounting(frozen, authorization, receipt):
    """Use the frozen lineage; a later continuation cannot inherit count eight."""
    budget = frozen["protocol"]["budget"]
    count = budget["accepted_old_unknown"]
    hold = budget["historical_unknown_reserved_usd"]
    if (type(count) is not int or count < 0
            or type(authorization.get("accepted_old_unknown")) is not int
            or type(receipt.get("inherited_unknown_requests")) is not int
            or count != authorization["accepted_old_unknown"]
            or count != receipt["inherited_unknown_requests"]
            or type(hold) is not str or hold != authorization.get("historical_unknown_reserved_usd")
            or hold != receipt.get("historical_unknown_reserved_usd")
            or hold != str(revision.RETAINED_UNKNOWN_HOLD)):
        raise revision.RunError("e1br_retained_accounting_binding_changed")
    return count, hold


def require_real_generation(campaign, receipt):
    """Software generation may test plumbing, never pass an effect gate."""
    if ((Path(campaign) / "SOFTWARE_ONLY.json").exists()
            or "software_fixture_only" in receipt and receipt["software_fixture_only"] is not False
            or "new_calls_are_real_models" in receipt and receipt["new_calls_are_real_models"] is not True):
        raise revision.RunError("software_receipt_cannot_enter_real_model_assessment")


def assess(root):
    result = {"schema": "e1br_formal_assessment_v1", "status": "not_evaluable",
              "main_gate": "not_evaluable", "formation_gate": "not_evaluable", "overall_gate": "not_evaluable",
              "effect_claim_eligible": False, "model_calls": 0,
              "scope": "authored_fictional_explicit_preferences_single_online_model_codex_only_no_second_judge_calibration"}
    try:
        root = Path(root).resolve()
        entry = revision.build_engine(root)
        verifier = common._load_module("scripts._e1br_assessment_checks", "e1b_assessment.py")
        verifier.execution = entry
        base = entry.campaign(root)
        receipt = entry.read_receipt(root, "main")
        require_real_generation(base, receipt)
        frozen = verifier._registration(root, "main", receipt)
        state = receipt["accounting"]
        entry.assert_clean(state, expected_calls=sum(revision.NEW_COUNTS["main"].values()), final=True)
        authorization = evaluation.read_json(base / "authorization.json")
        phase = evaluation.read_json(base / "main.authorization.json")
        binding = evaluation.read_json(base / "budget-binding.json")
        _, limit = entry.validate_authorization(root, frozen, budget_choice=revision.BUDGET_CHOICE,
            campaign_authorization_sha256=evaluation.digest(authorization),
            stage_authorization_sha256=evaluation.digest(phase))
        inherited_count, inherited_hold = retained_accounting(frozen, authorization, receipt)
        verifier._require(Decimal(state["max_observed_cost_usd"]) == limit
            and binding["campaign_authorization_sha256"] == evaluation.digest(authorization)
            and binding["source_sha256"] == evaluation.digest(frozen["source"])
            and binding["protocol_sha256"] == evaluation.digest(frozen["protocol"])
            and binding["policy_sha256"] == evaluation.digest(frozen["metadata"]["policy"]),
            "e1br_budget_binding_changed")
        result["ledger_sha256"] = verifier._ledger(root, state, frozen)
        retry_validator = getattr(entry, "validate_retry_ledger", validate_e1br_ledger)
        verifier._require(retry_validator(base / "allowance.sqlite3", stage="main",
            registration_sha256=frozen["seal_sha256"], max_requests=revision.MAX_WIRE_REQUESTS,
            reservation_policy=frozen["metadata"]["policy"]) == state["attempts"],
            "e1br_retry_policy_or_records_changed")
        if hasattr(entry, "validate_closed_ledger"):
            entry.validate_closed_ledger(root, frozen, state)
        qualification, packet, key = verifier._qualification(root, "main", receipt, frozen)
        formed, formed_key = formation.load_bundle(base / "main/formation-bundle")
        verifier._require(evaluation.digest(formed) == receipt["formation_packet_sha256"]
            and evaluation.digest(formed_key) == receipt["formation_mapping_sha256"]
            and formed_key["protocol"] == key["protocol"]
            and key["protocol"]["secondary_positions"] == 0
            and key["protocol"]["case_ids"] == revision.SELECTED, "e1br_primary_or_formation_binding_changed")
        replies = evaluation.aggregate(base / "main/blind-bundle")
        formed_result = formation.aggregate(base / "main/formation-bundle")
        result.update(reply_evaluation=replies, formation_evaluation=formed_result,
                      reflection_secondary=qualification["reflection_secondary"])
        if "source_lineage" in qualification:
            result["source_lineage"] = qualification["source_lineage"]
        verifier._require(replies.get("status") == "labels_complete" and replies.get("valid_primary_labels") == 216
            and replies.get("missing_generation_positions") == 0 and formed_result.get("status") == "local_labels_complete",
            "e1br_independent_first_labels_incomplete")
        verifier._require(replies["secondary"]["status"] == "not_planned_by_approved_revision"
                          and not any(row["role"] == "judge" for row in state["attempts"]), "e1br_unapproved_second_judge")
        scored = [row for row in qualification["positions"] if row["arm"] in {"A0", "A1"} and row["turn"] in (7, 8, 9)]
        verifier._require(len(scored) == 216 and all(row["delivery"] in {"model_unmodified", "guard_modified"} for row in scored),
                          "e1br_scored_deliveries_missing_or_rule_only")
        cost = Decimal(state["known_cost_usd"])
        result.update(status="evaluated", effect_claim_eligible=True,
            main_gate=replies["mechanical_main_gate"], formation_gate=formed_result["mechanical_formation_gate"],
            scored_delivery_counts=dict(Counter(row["delivery"] for row in scored)),
            raw_model_effect_claim_eligible=all(row["model_reply_eligible"] for row in scored),
            effect_layer="delivered_product_configuration_including_recorded_guard_modifications",
            known_campaign_cost_usd=str(cost), total_e1b_known_cost_usd=str(revision.PRIOR_E1B + cost),
            cumulative_known_cost_usd=str(revision.PRIOR_KNOWN + cost),
            inherited_unknown_count=inherited_count, inherited_unknown_reserved_usd=inherited_hold,
            new_unknown_count=state["unknown_cost_requests"], new_unknown_reserved_usd=state["reserved_cost_usd"],
            physical_attempts=state["calls_used"], lineage_physical_attempts=revision.PRIOR_DISPATCHES + state["calls_used"],
            primary_logical_positions=1056, second_judge_calls=0)
        result["overall_gate"] = "pass" if result["main_gate"] == result["formation_gate"] == "pass" else "fail"
    except (ValueError, KeyError, TypeError, OSError, sqlite3.DatabaseError) as error:
        result["reason"] = str(error) if isinstance(error, revision.RunError) else type(error).__name__
    return result
