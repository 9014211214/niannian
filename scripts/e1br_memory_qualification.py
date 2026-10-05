"""Approved memory-failure continuation qualification, retaining original gates.

Only complete, known-cost, saved JSON-syntax failures may be terminal missing.
Every original reply, failed capture, source and physical correction stays bound.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from aria.bench.e1b_cases import load_e1b_cases
from scripts import e1b_evaluation as evaluation
from scripts import e1b_formation as formation
from scripts import run_e1b_repair as common
from scripts.e1br_format_qualification import (
    _position_bindings, _require_position, classify_online_position,
    exploratory_assessment_view, reliability_counts, scored_inventory,
)
from scripts.e1br_format_runtime import terminal_format_failure
from scripts.e1br_qualification import COUNTS, reflection_metrics, require
from scripts.run_e1b import SHA, byte_sha


def validate_memory_terminal_proof(directory, proof, row):
    # Import lazily so the same public validator is used by runtime and finalization.
    from scripts.e1br_memory_runtime import validate_memory_terminal_proof as validate
    return validate(directory, proof, row)


def qualify(run_root, frozen, state, *, retained_attempts, retained_summary, logical_attempts,
            retained_position_bindings=None, retained_physical_attempts=None, expected_retained=731):
    require(type(expected_retained) is int and 0 < expected_retained < 1056
            and len(retained_attempts) == expected_retained, "memory_retained_logical_denominator_changed")
    physical_retained = retained_attempts if retained_physical_attempts is None else retained_physical_attempts
    require(len({row["request_id"] for row in physical_retained}) == len(physical_retained), "memory_duplicate_retained_physical_request")
    retained_by_id = {row["request_id"]: row for row in physical_retained}
    require(all(retained_by_id.get(row["request_id"]) == row for row in retained_attempts), "memory_logical_first_missing_from_physical_lineage")
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
    missing_memory = []
    for case_id in frozen["inputs"]["case_ids"]:
        directory = Path(run_root) / "main" / case_id
        report = evaluation.read_json(directory / "report.json")
        reports[case_id] = byte_sha(directory / "report.json")
        require(report.get("case_id") == case_id and report.get("evidence_kind") == "real_model" and report.get("phase") == "main",
                "e1br_report_generation_changed")
        capture = evaluation.read_json(directory / "raw-proposals.json")
        evaluation._validate_raw_capture_v2(capture, directory, case_id, load_e1b_cases().source_sha256)
        require(len(capture["sessions"]) == 2, "memory_two_original_stages_required")
        proofs = report["shared"]["state"].get("memory_terminal_positions", [])
        require(type(proofs) is list and all(type(p) is dict for p in proofs), "memory_terminal_proofs_invalid")
        proof_map = {p.get("request_id"): p for p in proofs}
        require(len(proof_map) == len(proofs), "memory_duplicate_terminal_proof")
        consumed_memory_proofs = set()
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
            captured_session = capture["sessions"][number - 1]
            missing = not captured_session["complete"]
            proof = proof_map.get(linked[0]["request_id"])
            if missing:
                require(proof is not None, "memory_incomplete_without_terminal_proof")
                historical = linked[0]["request_id"] in old
                require(type(proof.get("historical")) is bool and proof["historical"] is historical
                        and (type(retained_summary.get("memory_source_baseline")) is dict
                             and proof.get("source_baseline") == retained_summary["memory_source_baseline"]
                             if historical else proof.get("source_baseline") is None),
                        "memory_terminal_frozen_baseline_changed")
                require(proof.get("first_capture_sha256") == evaluation.digest(first)
                        and proof.get("product_capture_sha256") == evaluation.digest(captured_session["product_outcome"]),
                        "memory_terminal_original_capture_changed")
                validate_memory_terminal_proof(directory, proof, linked[0])
                consumed_memory_proofs.add(linked[0]["request_id"])
                missing_memory.append({"case_id": case_id, "session": number,
                    "position": ["memory", case_id, number, "shared", 0, None],
                    "request_id": linked[0]["request_id"], "proof_sha256": evaluation.digest(proof),
                    "receipt_sha256": evaluation.digest(linked[0]), "raw_proposals": None,
                    "accepted_count": None, "memory_completed": False, "model_metrics": None,
                    "formation_result": "not_evaluable", "retry_allowed": False})
            else:
                require(proof is None, "memory_terminal_proof_on_successful_capture")
            memory.append({"case_id": case_id, "session": number, "first_sha256": byte_sha(first_path),
                           "missing": missing, "model_metrics": None,
                           "receipt_sha256": evaluation.digest(linked[0]),
                           "reflection_receipt_sha256": evaluation.digest(reviewed[0])})
            product_path = directory / f"reflection-s{number}.product.json"
            product = evaluation.read_json(product_path)
            require(product.get("schema") == "e1br_reflection_product_v1"
                    and product.get("session_id") == session and product.get("capture_error") is None
                    and product.get("updater_raised") is False and product.get("product_status") in {"ok", "failed"},
                    "e1br_reflection_terminal_evidence_missing")
            if missing:
                completion = evaluation.read_json(directory / f"memory-s{number}.terminal.reflection.json")
                audit = completion.get("offline_audit", {})
                require(completion.get("schema") == "e1br_memory_reflection_completion_v1"
                        and completion.get("memory_proof_sha256") == evaluation.digest(proof)
                        and completion.get("memory_completed") is False
                        and audit.get("session_id") == session and audit.get("memory_completed") is False
                        and audit.get("gap") is True and audit.get("outcome") == "memory_terminal_missing"
                        and completion.get("reflection_status") in ({"sealed_failed"} if product["product_status"] == "failed" else {"ok", "skipped"})
                        and audit in report["shared"].get("profile", {}).get("offline_session_audit", []),
                        "memory_missing_completion_or_gap_changed")
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
                physical = [row for row in [*physical_retained, *state["attempts"]] if row["role"] == "reflection"
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
        require(consumed_memory_proofs == set(proof_map), "memory_unbound_terminal_proof")
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
    registrations = {row["request_id"]: row["registration_sha256"] for row in attempts}
    require(all(type(value) is str and SHA.fullmatch(value) for value in registrations.values()),
            "memory_generation_registration_missing")
    for position in positions:
        position["generation_registration_sha256"] = registrations[position["request_id"]]
    lineage = {"schema": "e1br_memory_source_lineage_v1", "retained_first_answers": expected_retained,
        "retained_source_registration": deepcopy(retained_summary),
        "new_registration_sha256": frozen["seal_sha256"],
        "all_initial_request_registrations": registrations,
        "initial_positions_by_registration": dict(Counter(registrations.values())),
        "initial_positions": [{"request_id": row["request_id"],
            "position": deepcopy(row.get("experiment_position", bindings.get(row["request_id"], {}).get("position"))),
            "receipt_sha256": evaluation.digest(row), "registration_sha256": registrations[row["request_id"]],
            "retained": row["request_id"] in old} for row in attempts],
        "mixed_version": True, "old_frozen_gates_replaced": False}
    return {"schema": "e1br_memory_exploratory_qualification_v1", "stage": "main",
            "registration_sha256": frozen["seal_sha256"], "ledger_id": state["ledger_id"],
            "source_report_sha256": reports, "actual_counts": dict(COUNTS),
            "positions": positions, "memory_originals": memory, "reflection_originals": reflection,
            "reflection_secondary": reflection_metrics(reflection),
            "missing_memory_positions": missing_memory,
            "memory_missing_metrics": None,
            "formation_expected_cases": 12,
            "retained_physical_attempts": len(physical_retained),
            "source_lineage": lineage,
            "retained_main_prefix": retained_summary, "retained_first_answers": len(retained_attempts),
            "new_physical_attempts": len(state["attempts"]), "effect_claim_eligible": False,
            "fixed_case_denominator": 12, "fixed_reply_positions": 216,
            "original_main_gate": "not_evaluable", "original_formation_gate": "not_evaluable",
            "original_overall_gate": "not_evaluable", "measurement_kind": "mixed_source_memory_terminal_continuation",
            "online_reliability": reliability_counts(positions),
            "scored_position_inventory": scored_inventory(positions),
            "scope": "Preserved first bodies, actual deliveries and null failed memories. Fixed complete inventory is neither a raw-model nor original-effect pass."}


def validate_formation_missing(qualification, mapping):
    """Only proved failed memory originals may remain unavailable to the judge."""
    require(qualification.get("schema") == "e1br_memory_exploratory_qualification_v1"
            and qualification.get("fixed_case_denominator") == 12
            and qualification.get("fixed_reply_positions") == 216, "memory_formation_qualification_changed")
    expected = {}
    seen = set()
    for row in qualification["missing_memory_positions"]:
        case, session = row["case_id"], row["session"]
        require((case, session) not in seen and session in (1, 2)
                and row.get("raw_proposals") is None and row.get("accepted_count") is None
                and row.get("memory_completed") is False and row.get("model_metrics") is None
                and row.get("formation_result") == "not_evaluable", "memory_formation_missing_proof_changed")
        seen.add((case, session))
        expected.setdefault(case, []).append(f"memory_{session}_array_unavailable")
    inventory = mapping.get("inventory", [])
    require(len(inventory) == 12 and len({row["case_id"] for row in inventory}) == 12,
            "memory_formation_fixed_cases_changed")
    require(set(expected) <= {row["case_id"] for row in inventory}, "memory_missing_case_not_in_formation")
    for row in inventory:
        require(sorted(row.get("problems", [])) == sorted(expected.get(row["case_id"], [])),
                "memory_formation_unqualified_problem")
        capture = row.get("raw_capture")
        require(type(capture) is dict and len(capture.get("sessions", [])) == 2,
                "memory_formation_originals_missing")
        for session in capture["sessions"]:
            missing = (row["case_id"], session["session"]) in seen
            require((session.get("parse_status") == "parse_error" and session.get("raw_proposals") is None
                     and session.get("complete") is False) if missing else
                    (session.get("parse_status") == "present_array" and type(session.get("raw_proposals")) is list
                     and session.get("complete") is True), "memory_formation_null_or_complete_changed")
    return {"fixed_case_denominator": 12, "missing_memory_positions": len(seen),
            "missing_cases": sorted(expected), "missing_metrics": None,
            "observed_candidates": len(mapping.get("mapping", {})), "formation_gate": "not_evaluable"}


def export_packets(run_root, reply_output, formation_output, protocol, *, qualification,
                   suite=None, registered_sha256=None, blind_seed=20261003):
    """Existing exporters already preserve unavailable slots and null captures."""
    replies = evaluation.export_packet(run_root, reply_output, protocol, suite=suite,
        registered_sha256=registered_sha256, blind_seed=blind_seed)
    formed = formation.export_packet(run_root, formation_output, protocol, suite=suite,
        registered_sha256=registered_sha256, blind_seed=blind_seed)
    _, mapping = formation.load_bundle(formation_output, suite=suite)
    missing = validate_formation_missing(qualification, mapping)
    return {"reply": replies, "formation": formed, "memory_missing": missing,
            "effect_claim_eligible": False}


def assess(root, *, revision):
    """Reuse original source/finance/first-label audit, never loosen its gate."""
    module = common._load_module("scripts._e1br_memory_assessment_base", "e1br_assessment.py")
    module.revision = revision
    original = module.assess(root)
    result = exploratory_assessment_view(original)
    result.update(schema="e1br_memory_exploratory_assessment_v1",
                  measurement_kind="mixed_source_memory_terminal_continuation")
    # Incomplete originals intentionally keep the original aggregate unavailable.
    # Available first labels remain visible as counts; no missing case is zero.
    if "formation_evaluation" in original:
        formed = original["formation_evaluation"]
        result["exploratory"]["formation_missing_cases"] = sorted(formed.get("raw_problem_cases", {}))
        result["exploratory"]["formation_missing_metrics"] = None
        result["exploratory"]["available_formation_first_labels"] = formed.get("valid_labels")
    if "reply_evaluation" in original:
        replies = original["reply_evaluation"]
        result["exploratory"]["available_reply_first_labels"] = replies.get("valid_primary_labels")
        result["exploratory"]["missing_reply_positions"] = replies.get("missing_generation_positions")
    if original.get("reason") == "e1br_independent_first_labels_incomplete":
        # The reused original gate correctly rejects approved missing evidence.
        # Distinguish finished first-label collection from a complete effect study.
        base = Path(root) / revision.CAMPAIGN
        packet, _ = evaluation.load_bundle(base / "main/blind-bundle")
        formed_packet, formed_key = formation.load_bundle(base / "main/formation-bundle")
        qualification = evaluation.read_json(base / "main/qualification.json")
        missing = validate_formation_missing(qualification, formed_key)
        replies, formed = original["reply_evaluation"], original["formation_evaluation"]
        labels_complete = (replies.get("valid_primary_labels") == len(packet["items"])
            and replies.get("missing_or_invalid_primary_labels") == 0
            and formed.get("valid_labels") == len(formed_packet["items"])
            and formed.get("missing_or_invalid_labels") == 0 and not formed.get("changed_originals"))
        result["exploratory"]["available_first_labels_complete"] = labels_complete
        result["exploratory"]["memory_missing"] = missing
        if labels_complete:
            result["status"] = "exploratory_first_labels_complete_with_missing_evidence"
    return result
