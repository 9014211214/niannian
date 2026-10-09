"""Versioned exploratory qualification; never converts the original E1b gate.

Retain every planned position and distinguish the raw model body, actual rule
fallback and missing evidence. Aggregation remains in the existing first-label
engines; this adapter never scores answers or substitutes a reply field from a
schema-rejected JSON envelope.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from aria.bench.e1b_cases import load_e1b_cases
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from scripts import e1b_evaluation as evaluation
from scripts import run_e1b_repair as common
from scripts.e1br_format_runtime import SCHEMA, terminal_format_failure
from scripts.e1br_qualification import COUNTS, reflection_metrics, require
from scripts.run_e1b import SHA, byte_sha, text_sha


def classify_online_position(turn, row, *, proof=None, report_evidence_kind=None):
    """Require a settled first body and an independently persisted delivery."""
    persisted_turn = turn
    if report_evidence_kind is not None:
        require(report_evidence_kind in {"real_model", "software_fixture"}
                and turn.get("evidence_kind") == report_evidence_kind,
                "format_report_turn_evidence_kind_changed")
        # E1bExperimentApp.state adds only this display marker to a copy.
        # Runtime proofs deliberately hash the unmodified persisted turn.
        persisted_turn = {key: value for key, value in turn.items() if key != "evidence_kind"}
    visible = row.get("visible_content_sha256")
    require(isinstance(visible, str) and SHA.fullmatch(visible), "format_visible_first_body_missing")
    channel = row.get("reply_return_channel")
    require(channel in {"text", "json_object"}, "format_online_return_channel_changed")
    if terminal_format_failure(row):
        require(turn.get("status") == "failed" and turn.get("source") == "model_failed"
                and turn.get("response") == "" and type(proof) is dict,
                "format_failed_turn_or_proof_missing")
        require(proof.get("schema") == SCHEMA and proof.get("request_id") == row["request_id"]
                and proof.get("receipt_sha256") == evaluation.digest(row)
                and proof.get("turn_id") == turn["turn_id"] and proof.get("session_id") == turn["session_id"]
                and proof.get("turn_sha256") == evaluation.digest(persisted_turn)
                and proof.get("fallback_sha256") == text_sha(DETERMINISTIC_MODEL_FAILURE_REPLY)
                and proof.get("delivered_response") == DETERMINISTIC_MODEL_FAILURE_REPLY
                and proof.get("model_reply_eligible") is False and proof.get("model_metrics") is None
                and type(proof.get("historical")) is bool and proof.get("body_retried") is False
                and proof.get("fixed_denominator_retained") is True,
                "format_terminal_delivery_binding_changed")
        return {"delivery": "rule_fallback", "origin": "rule_fallback", "raw_model_status": "schema_rejected",
                "reply_return_channel": channel,
                "visible_content_sha256": visible, "delivered_reply_sha256": proof["fallback_sha256"],
                "terminal_proof_sha256": evaluation.digest(proof), "model_reply_eligible": False,
                "model_metrics": None, "visible_model_body_received": True,
                "missing_qualified_model_response": True, "body_retried": False}
    require(proof is None and row.get("status") == "received" and turn.get("status") == "completed"
            and isinstance(turn.get("response"), str) and bool(turn["response"].strip())
            and isinstance(row.get("model_reply_sha256"), str) and SHA.fullmatch(row["model_reply_sha256"]),
            "format_primary_or_delivery_unqualified")
    sha = text_sha(turn["response"])
    delivery = ("rule_only" if turn.get("source") == "rule_safety" else
                "guard_modified" if turn.get("source") == "rule_protection" else
                "model_unmodified" if sha == row["model_reply_sha256"] else "guard_modified")
    return {"delivery": delivery, "origin": "rule_only" if delivery == "rule_only" else "model",
            "raw_model_status": "decision_json_schema_accepted" if channel == "json_object" else "plain_text_received",
            "reply_return_channel": channel, "visible_content_sha256": visible,
            "delivered_reply_sha256": sha, "model_reply_eligible": delivery == "model_unmodified",
            "model_metrics": None, "visible_model_body_received": True,
            "missing_qualified_model_response": False, "body_retried": False}


def scored_inventory(positions):
    rows = [p for p in positions if p["arm"] in {"A0", "A1"} and p["turn"] in (7, 8, 9)]
    require(len(rows) == 216 and len({(p["case_id"], p["arm"], p["repeat"], p["turn"]) for p in rows}) == 216,
            "format_fixed_scoring_inventory_changed")
    return [{"case_id": p["case_id"], "arm": p["arm"], "repeat": p["repeat"], "turn": p["turn"],
             "origin": p["origin"], "raw_model_status": p["raw_model_status"],
             "model_label_eligible": p["delivery"] in {"model_unmodified", "guard_modified"},
             "model_metrics": None} for p in rows]


def reliability_counts(positions):
    counts = Counter(p["raw_model_status"] for p in positions)
    return {"planned_online_positions": 1008, "recorded_positions": len(positions),
            "decision_json_positions": counts["decision_json_schema_accepted"] + counts["schema_rejected"],
            "decision_json_schema_accepted": counts["decision_json_schema_accepted"],
            "decision_json_schema_rejected": counts["schema_rejected"],
            "plain_text_received": counts["plain_text_received"],
            "no_body": 0, "not_dispatched": 1008 - len(positions),
            "delivery_counts": dict(Counter(p["delivery"] for p in positions)),
            "reliability_pass_claim": False,
            "denominator_rule": "Only json_object first bodies enter the decision-JSON schema denominator; received plain text is a separate legitimate path. No-body, unattempted and rule delivery are not semantic zero scores."}


def exploratory_assessment_view(original):
    """Keep original gates closed even when the reused mechanical comparison passes."""
    result = deepcopy(original)
    evaluated = original.get("status") == "evaluated"
    result.update(schema="e1br_format_exploratory_assessment_v1",
                  status="exploratory_labels_complete" if evaluated else "not_evaluable_exploratory",
                  main_gate="not_evaluable", formation_gate="not_evaluable", overall_gate="not_evaluable",
                  effect_claim_eligible=False, raw_model_effect_claim_eligible=False,
                  measurement_kind="mixed_source_exploratory_continuation",
                  original_frozen_experiment_gate="not_evaluable",
                  single_final_configuration_effect_proven=False)
    replies = result.get("reply_evaluation", {})
    formation = result.get("formation_evaluation", {})
    result["exploratory"] = {
        "fixed_cases": 12, "fixed_reply_positions": 216, "fixed_logical_positions": 1056,
        "main_mechanical_comparison": original.get("main_gate") if evaluated else None,
        "formation_mechanical_comparison": formation.get("mechanical_formation_gate")
            if formation.get("status") == "local_labels_complete" else None,
        "not_an_original_gate_pass": True,
        "raw_model_effect_claim": False,
        "interpretation": "Qualified recorded model delivery, including separately reported guard modifications; mixed source versions and preserved fallback history prevent a final-configuration efficacy claim.",
    }
    # An old aggregator may compute a delivered-text result before the stricter
    # transport gate rejects rule-only or failed positions. Keep it diagnostic;
    # do not expose it as the model comparison after that rejection.
    if not evaluated and "mechanical_main_gate" in replies:
        replies["diagnostic_delivered_text_comparison"] = replies.pop("mechanical_main_gate")
        replies["model_comparison"] = None
    return result


def assess(root, *, revision):
    """Reuse original integrity, finance and first-label checks, then relabel scope."""
    module = common._load_module("scripts._e1br_format_assessment_base", "e1br_assessment.py")
    module.revision = revision
    return exploratory_assessment_view(module.assess(root))


def _position_bindings(retained_attempts, bindings):
    """Only the sealed first 13 receipts predate recorded position metadata."""
    bindings = {} if bindings is None else bindings
    missing = {row["request_id"]: row for row in retained_attempts if "experiment_position" not in row}
    require(type(bindings) is dict and set(bindings) == set(missing), "format_legacy_position_bindings_missing")
    if not missing:
        return bindings
    require(len(missing) == 13, "format_legacy_position_inventory_changed")
    expected = {("online_fast", "e1b_a1", 1, "shared", 0, index) for index in range(12)}
    expected.add(("memory", "e1b_a1", 1, "shared", 0, None))
    positions = set()
    for request_id, row in missing.items():
        bound = bindings[request_id]
        require(type(bound) is dict and bound.get("receipt_sha256") == evaluation.digest(row)
                and type(bound.get("position")) is list and len(bound["position"]) == 6
                and bound["position"][0] == row["role"], "format_legacy_position_binding_changed")
        positions.add(tuple(bound["position"]))
    require(positions == expected, "format_legacy_position_inventory_changed")
    return bindings


def _require_position(row, expected, bindings):
    position = row.get("experiment_position")
    if position is None:
        bound = bindings.get(row["request_id"], {})
        require(bound.get("receipt_sha256") == evaluation.digest(row), "format_legacy_position_binding_changed")
        position = bound.get("position")
    require(position == expected, "format_receipt_position_changed")


def qualify(run_root, frozen, state, *, retained_attempts, retained_summary, logical_attempts,
            retained_position_bindings=None):
    bindings = _position_bindings(retained_attempts, retained_position_bindings)
    attempts = [*retained_attempts, *(row for row in logical_attempts(state)
                                    if row["product_step"] == "initial")]
    require(Counter(row["role"] for row in attempts) == Counter(COUNTS), "e1br_logical_denominator_changed")
    old = {row["request_id"]: row for row in retained_attempts}
    require(all(row == old[row["request_id"]] if row["request_id"] in old
                else row.get("registration_sha256") == frozen["seal_sha256"] for row in attempts),
            "e1br_receipt_registration_changed")
    require(len({row["request_id"] for row in attempts}) == len(attempts), "format_duplicate_logical_request")
    require(all(row.get("status") == "received" and SHA.fullmatch(row.get("visible_content_sha256", ""))
                for row in attempts if row["role"] == "memory"), "e1br_primary_content_missing")
    require(all(row.get("status") == "received" and SHA.fullmatch(row.get("visible_content_sha256", ""))
                or terminal_format_failure(row) for row in attempts if row["role"] == "online_fast"),
                "format_unqualified_primary_failure")
    require(len(frozen["inputs"]["case_ids"]) == len(set(frozen["inputs"]["case_ids"])) == 12,
            "format_fixed_case_denominator_changed")
    lookup = {(row["operation_id"], row["session_id"]): row for row in attempts if row["role"] == "online_fast"}
    require(len(lookup) == 1008, "e1br_online_identity_duplicate")
    positions, reports, memory, reflection, used = [], {}, [], [], set()
    for case_id in frozen["inputs"]["case_ids"]:
        directory = Path(run_root) / "main" / case_id
        report = evaluation.read_json(directory / "report.json")
        reports[case_id] = byte_sha(directory / "report.json")
        require(report.get("case_id") == case_id and report.get("evidence_kind") == "real_model" and report.get("phase") == "main",
                "e1br_report_generation_changed")
        capture = evaluation.read_json(directory / "raw-proposals.json")
        evaluation._validate_raw_capture_v2(capture, directory, case_id, load_e1b_cases().source_sha256)
        require(len(capture["sessions"]) == 2 and all(row["complete"] for row in capture["sessions"]),
                "e1br_two_memory_originals_required")
        for number in (1, 2):
            first_path = directory / f"memory-s{number}.first.json"
            first = evaluation.read_json(first_path)
            attempted = evaluation.read_json(directory / f"memory-s{number}.attempt.json")
            session = attempted["session_id"]
            linked = [row for row in attempts if row["role"] == "memory" and row["session_id"] == session]
            reviewed = [row for row in attempts if row["role"] == "reflection" and row["session_id"] == session]
            require(len(linked) == len(reviewed) == 1 and first["request_hash_kind"] == "actual_wire"
                    and linked[0]["request_sha256"] == first["request_sha256"]
                    and linked[0]["client_return_content_sha256"] == first["visible_response_sha256"]
                    and linked[0]["operation_id"] == reviewed[0]["operation_id"], "e1br_memory_binding_changed")
            _require_position(linked[0], ["memory", case_id, number, "shared", 0, None], bindings)
            _require_position(reviewed[0], ["reflection", case_id, number, "shared", 0, None], bindings)
            memory.append({"case_id": case_id, "session": number, "first_sha256": byte_sha(first_path),
                           "receipt_sha256": evaluation.digest(linked[0]),
                           "reflection_receipt_sha256": evaluation.digest(reviewed[0])})
            product_path = directory / f"reflection-s{number}.product.json"
            product = evaluation.read_json(product_path)
            require(product.get("schema") == "e1br_reflection_product_v1"
                    and product.get("session_id") == session and product.get("capture_error") is None
                    and product.get("updater_raised") is False and product.get("product_status") in {"ok", "failed"},
                    "e1br_reflection_terminal_evidence_missing")
            steps = product["product_steps"]
            require([row["product_step"] for row in steps] in (["initial"], ["initial", "format_correction"])
                    and product["format_correction_count"] == len(steps) - 1,
                    "e1br_reflection_product_steps_changed")
            first_records = []
            for step in steps:
                suffix = "" if step["product_step"] == "initial" else ".correction"
                filename = f"reflection-s{number}{suffix}.first.json"
                require(step["first_path"] == filename and step["first_sha256"] == byte_sha(directory / filename),
                        "e1br_reflection_first_bytes_changed")
                original = evaluation.read_json(directory / filename)
                physical = [row for row in [*retained_attempts, *state["attempts"]] if row["role"] == "reflection"
                            and row["session_id"] == session and row["product_step"] == step["product_step"]]
                bound = [row for row in physical if row["request_id"] == original.get("request_id")]
                require(len(bound) == 1 and not original.get("request_identity_missing")
                        and original["physical_request_ids"] == [row["request_id"] for row in physical]
                        and original["request_sha256"] == bound[0]["request_sha256"]
                        and original.get("visible_response_sha256") == bound[0].get("client_return_content_sha256"),
                        "e1br_reflection_physical_binding_changed")
                first_records.append({"product_step": step["product_step"], "first_sha256": step["first_sha256"],
                                      "physical_receipt_sha256": evaluation.digest(bound[0]),
                                      "physical_attempts": len(physical)})
            require(product["first_sha256"] == steps[0]["first_sha256"], "e1br_initial_reflection_replaced")
            reflection.append({"case_id": case_id, "session": number, "product_sha256": byte_sha(product_path),
                               "status": product["product_status"], "format_correction_count": len(steps) - 1,
                               "format_attempts": product.get("format_attempts", []),
                               "steps": first_records, "secondary_only": True})
        branches = [{"arm": "shared", "repeat": 0, "state": report["shared"]["state"]}, *report["branches"]]
        require(len(branches) == 7 and {(b["arm"], b["repeat"]) for b in branches} ==
                {("shared", 0), *((arm, repeat) for arm in ("A0", "A1") for repeat in (1, 2, 3))},
                "format_branch_inventory_changed")
        for branch in branches:
            turns = branch["state"]["turns"]
            require(len(turns) == (24 if branch["arm"] == "shared" else 10), "format_turn_inventory_changed")
            proofs = branch["state"].get("format_terminal_positions", [])
            require(type(proofs) is list and all(type(p) is dict for p in proofs), "format_terminal_proof_invalid")
            proof_lookup = {(p.get("turn_id"), p.get("session_id")): p for p in proofs}
            require(len(proof_lookup) == len(proofs), "format_duplicate_terminal_proof")
            consumed = set()
            for index, turn in enumerate(turns):
                identity = turn["turn_id"], turn["session_id"]
                require(identity not in used, "e1br_reply_receipt_reused")
                used.add(identity)
                row = lookup.get(identity)
                require(row is not None, "e1br_reply_missing")
                _require_position(row, ["online_fast", case_id,
                    index // 12 + 1 if branch["arm"] == "shared" else 3,
                    branch["arm"], branch["repeat"], index % 12 if branch["arm"] == "shared" else index], bindings)
                proof = proof_lookup.get(identity)
                classified = classify_online_position(turn, row, proof=proof,
                    report_evidence_kind=report["evidence_kind"])
                if proof is not None:
                    consumed.add(identity)
                positions.append({"case_id": case_id, "arm": branch["arm"], "repeat": branch["repeat"],
                                  "turn": index, "turn_id": identity[0], "session_id": identity[1],
                                  "request_id": row["request_id"], "receipt_sha256": evaluation.digest(row),
                                  **classified})
            require(consumed == set(proof_lookup), "format_unbound_terminal_proof")
    require(set(lookup) == used, "e1br_unbound_online_receipts")
    return {"schema": "e1br_format_exploratory_qualification_v1", "stage": "main",
            "registration_sha256": frozen["seal_sha256"], "ledger_id": state["ledger_id"],
            "source_report_sha256": reports, "actual_counts": dict(COUNTS),
            "positions": positions, "memory_originals": memory, "reflection_originals": reflection,
            "reflection_secondary": reflection_metrics(reflection),
            "retained_main_prefix": retained_summary, "retained_first_answers": len(retained_attempts),
            "new_physical_attempts": len(state["attempts"]), "effect_claim_eligible": False,
            "fixed_case_denominator": 12, "fixed_reply_positions": 216,
            "original_main_gate": "not_evaluable", "original_formation_gate": "not_evaluable",
            "original_overall_gate": "not_evaluable", "measurement_kind": "mixed_source_exploratory_continuation",
            "online_reliability": reliability_counts(positions),
            "scored_position_inventory": scored_inventory(positions),
            "scope": "Preserved raw model failures and actual deliveries; complete logical inventory is not a model-effect pass."}
