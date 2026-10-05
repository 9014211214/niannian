"""Synthetic metadata exercises the versioned exploratory qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aria.bench.e1b_cases import load_e1b_cases
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from scripts import e1b_evaluation as evaluation
from scripts import e1br_format_qualification as subject
from scripts.e1br_format_runtime import SCHEMA, FormatExperimentApp
from scripts.run_e1b import RunError, byte_sha, text_sha


def saved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return byte_sha(path)


def rejected(row):
    return {**row, "status": "failed", "http_status": 200,
            "failure_phase": "receipt_validation", "reply_return_channel": "json_object",
            "failure_code": "online_decision_schema_invalid", "validation_error": "online_decision_schema_invalid",
            "visible_original_sha256": "e" * 64, "model_reply_sha256": None,
            "observation": {"body_status": "parsed", "envelope_kind": "success"},
            "receipt_identity": {"version": "receipt_identity_v1", "error_field_status": "absent",
                                 "model_status": "recognized", "provider_status": "recognized"},
            "usage": {"cost": 0.00001, "prompt_tokens": 1, "completion_tokens": 1},
            "reservation": {"reserved_cost_usd": "0.001", "input_token_plan_bound": 100, "output_token_limit": 100}}


def terminal_proof(turn, row):
    return {"schema": SCHEMA, "request_id": row["request_id"], "receipt_sha256": evaluation.digest(row),
            "turn_id": turn["turn_id"], "session_id": turn["session_id"],
            "turn_sha256": evaluation.digest(turn), "fallback_sha256": text_sha(DETERMINISTIC_MODEL_FAILURE_REPLY),
            "delivered_response": DETERMINISTIC_MODEL_FAILURE_REPLY,
            "model_reply_eligible": False, "model_metrics": None, "historical": True,
            "body_retried": False, "fixed_denominator_retained": True}


@pytest.fixture
def full_metadata(tmp_path):
    suite = load_e1b_cases()
    ids = [f"e1b_{letter}{number}" for letter in "abcdef" for number in (1, 2)]
    cases = {case.id: case for case in suite.items}
    rows, reports, failed_turn, failed_row = [], {}, None, None
    for case_id in ids:
        directory = tmp_path / "main" / case_id
        report = {"case_id": case_id, "phase": "main", "evidence_kind": "real_model",
                  "synthetic_emotion": True, "shared": {"state": {"turns": []}}, "branches": []}
        reports[case_id] = report
        captures = []
        def receipt(role, session, operation):
            row = {"request_id": f"request-{len(rows)}", "role": role, "session_id": session,
                   "operation_id": operation, "product_step": "initial", "status": "received",
                   "reply_return_channel": "json_object" if role == "online_fast" else "text",
                   "request_sha256": "a" * 64, "client_return_content_sha256": "b" * 64,
                   "visible_content_sha256": "b" * 64}
            rows.append(row)
            return row
        for number in (1, 2):
            session = f"{case_id}-s{number}"
            for index in range(12):
                operation = f"{session}-{index}"
                row = receipt("online_fast", session, operation)
                row["experiment_position"] = ["online_fast", case_id, number, "shared", 0, index]
                turn = {"turn_id": operation, "session_id": session, "status": "completed",
                        "source": "model", "response": "软件生成占位", "text": cases[case_id].current_input(number, index)}
                row["model_reply_sha256"] = text_sha(turn["response"])
                if case_id == "e1b_c1" and number == 2 and index == 5:
                    row.update(rejected(row))
                    turn.update(status="failed", source="model_failed", response="", requires_review=True)
                    failed_turn, failed_row = turn, row
                report["shared"]["state"]["turns"].append(turn)
            mem = receipt("memory", session, session + "-offline")
            ref = receipt("reflection", session, session + "-offline")
            mem["experiment_position"] = ["memory", case_id, number, "shared", 0, None]
            ref["experiment_position"] = ["reflection", case_id, number, "shared", 0, None]
            first = {"session": number, "request_sha256": mem["request_sha256"], "request_hash_kind": "actual_wire",
                     "visible_response_sha256": mem["client_return_content_sha256"], "field_present": True,
                     "raw_proposals": [], "parse_status": "present_array", "public_projection_dropped_fields": 0}
            first_sha = saved(directory / f"memory-s{number}.first.json", first)
            outcome = {"product_status": "ok"}
            outcome_sha = saved(directory / f"memory-s{number}.product.json", outcome)
            saved(directory / f"memory-s{number}.attempt.json", {"session_id": session})
            captures.append({**first, "complete": True, "product_outcome": outcome,
                             "first_attempt_sha256": first_sha, "product_outcome_sha256": outcome_sha})
            ref_first = {"request_id": ref["request_id"], "physical_request_ids": [ref["request_id"]],
                         "request_sha256": ref["request_sha256"], "visible_response_sha256": ref["client_return_content_sha256"]}
            filename = f"reflection-s{number}.first.json"
            ref_sha = saved(directory / filename, ref_first)
            saved(directory / f"reflection-s{number}.product.json", {
                "schema": "e1br_reflection_product_v1", "session_id": session, "capture_error": None,
                "updater_raised": False, "product_status": "ok", "format_correction_count": 0,
                "product_steps": [{"product_step": "initial", "first_path": filename, "first_sha256": ref_sha}],
                "first_sha256": ref_sha, "format_attempts": [{"status": "shape_valid"}]})
        saved(directory / "raw-proposals.json", {"schema": "e1b_raw_proposals_v2", "case_id": case_id,
                                                 "source_sha256": suite.source_sha256, "sessions": captures})
        for arm in ("A0", "A1"):
            for repeat in (1, 2, 3):
                branch = {"arm": arm, "repeat": repeat, "state": {"turns": []}, "scored_replies": []}
                for index in range(10):
                    session = f"{case_id}-{arm}-{repeat}"
                    row = receipt("online_fast", session, f"{session}-{index}")
                    row["experiment_position"] = ["online_fast", case_id, 3, arm, repeat, index]
                    turn = {"turn_id": row["operation_id"], "session_id": session, "status": "completed",
                            "source": "model", "response": "软件生成占位", "text": cases[case_id].current_input(3, index)}
                    row["model_reply_sha256"] = text_sha(turn["response"])
                    branch["state"]["turns"].append(turn)
                    if index in (7, 8, 9):
                        branch["scored_replies"].append({"turn": index, "response": turn["response"]})
                report["branches"].append(branch)
    assert len(rows) == 1056 and rows[371] is failed_row
    for index, row in enumerate(rows):
        row["registration_sha256"] = "d" * 64 if index < 372 else "c" * 64
    legacy = {}
    for row in rows[:13]:
        position = row.pop("experiment_position")
        legacy[row["request_id"]] = {"receipt_sha256": evaluation.digest(row), "position": position}
    reports["e1b_c1"]["shared"]["state"]["format_terminal_positions"] = [terminal_proof(failed_turn, failed_row)]
    for case_id, report in reports.items():
        displayed = deepcopy(report)
        for state in [displayed["shared"]["state"], *(b["state"] for b in displayed["branches"])]:
            for turn in state["turns"]:
                turn["evidence_kind"] = displayed["evidence_kind"]
        saved(tmp_path / "main" / case_id / "report.json", displayed)
    frozen = {"inputs": {"case_ids": ids}, "seal_sha256": "c" * 64}
    state = {"ledger_id": "software-ledger", "attempts": rows[372:]}
    return SimpleNamespace(root=tmp_path, frozen=frozen, state=state, retained=rows[:372], rows=rows,
                           reports=reports, failed_turn=failed_turn, failed_row=failed_row, legacy=legacy)


def qualify(data):
    return subject.qualify(data.root, data.frozen, data.state, retained_attempts=data.retained,
                           retained_summary={"software_only": True}, logical_attempts=lambda s: s["attempts"],
                           retained_position_bindings=data.legacy)


def test_complete_inventory_retains_failure_without_model_zero(full_metadata):
    result = qualify(full_metadata)
    assert result["actual_counts"] == {"online_fast": 1008, "memory": 24, "reflection": 24}
    assert result["retained_first_answers"] == 372 and result["new_physical_attempts"] == 684
    assert len(result["positions"]) == 1008 and len(result["scored_position_inventory"]) == 216
    failure = next(p for p in result["positions"] if p["raw_model_status"] == "schema_rejected")
    assert failure["origin"] == "rule_fallback" and failure["model_metrics"] is None
    assert not failure["model_reply_eligible"] and result["online_reliability"]["decision_json_schema_rejected"] == 1
    assert failure["visible_model_body_received"] and failure["missing_qualified_model_response"]
    assert result["original_main_gate"] == result["original_formation_gate"] == "not_evaluable"
    assert result["effect_claim_eligible"] is False


def test_actual_runtime_proof_writer_matches_qualification(full_metadata, tmp_path):
    app = SimpleNamespace(_run={"turns": [full_metadata.failed_turn]},
        _sessions={"fictional-chen": (None, SimpleNamespace(turns=[SimpleNamespace(agent=DETERMINISTIC_MODEL_FAILURE_REPLY)]))},
        _storage_failed=False, _format_terminal={}, run_dir=tmp_path)
    proof = FormatExperimentApp.retain_terminal(app, full_metadata.failed_row, historical=True)
    assert proof["receipt_sha256"] == evaluation.digest(full_metadata.failed_row)
    assert proof["turn_sha256"] == evaluation.digest(full_metadata.failed_turn)
    result = subject.classify_online_position(full_metadata.failed_turn, full_metadata.failed_row, proof=proof)
    assert result["origin"] == "rule_fallback" and result["model_metrics"] is None
    assert evaluation.read_json(tmp_path / f"format-terminal-{full_metadata.failed_row['request_id']}.json") == proof


def test_old_zero_based_branch_shape_is_rejected(full_metadata):
    path = full_metadata.root / "main/e1b_a1/report.json"
    report = evaluation.read_json(path)
    for branch in report["branches"]:
        branch["repeat"] -= 1
    saved(path, report)
    with pytest.raises(RunError, match="format_branch_inventory_changed"):
        qualify(full_metadata)


def test_swapping_completed_branch_labels_cannot_reassign_model_receipts(full_metadata):
    path = full_metadata.root / "main/e1b_a1/report.json"
    report = evaluation.read_json(path)
    for branch in report["branches"]:
        branch["arm"] = "A1" if branch["arm"] == "A0" else "A0"
    saved(path, report)
    with pytest.raises(RunError, match="format_receipt_position_changed"):
        qualify(full_metadata)


def test_missing_sealed_legacy_mapping_is_not_inferred_from_report(full_metadata):
    full_metadata.legacy.clear()
    with pytest.raises(RunError, match="format_legacy_position_bindings_missing"):
        qualify(full_metadata)


def test_legacy_mapping_is_bound_to_exact_original_receipt(full_metadata):
    next(iter(full_metadata.legacy.values()))["receipt_sha256"] = "f" * 64
    with pytest.raises(RunError, match="format_legacy_position_binding_changed"):
        qualify(full_metadata)


def test_offline_receipts_cannot_change_case_or_session_number(full_metadata):
    row = next(r for r in full_metadata.rows if r["role"] == "reflection")
    row["experiment_position"][2] = 2
    with pytest.raises(RunError, match="format_receipt_position_changed"):
        qualify(full_metadata)


def test_received_plain_text_is_not_counted_as_decision_json_schema_success(full_metadata):
    rows = [r for r in full_metadata.rows if r["role"] == "online_fast" and r["status"] == "received"
            and "experiment_position" in r][:33]
    for row in rows:
        row["reply_return_channel"] = "text"
    result = qualify(full_metadata)
    counts = result["online_reliability"]
    assert counts["plain_text_received"] == 33
    assert counts["decision_json_positions"] == 975
    assert counts["decision_json_schema_accepted"] == 974 and counts["decision_json_schema_rejected"] == 1
    assert "shape_accepted" not in counts
    assert counts["reliability_pass_claim"] is False


def test_unrecorded_online_return_channel_cannot_be_assumed_json(full_metadata):
    full_metadata.rows[14].pop("reply_return_channel")
    with pytest.raises(RunError, match="format_online_return_channel_changed"):
        qualify(full_metadata)


@pytest.mark.parametrize("field,value", [
    ("body_retried", True), ("model_reply_eligible", True), ("model_metrics", 0),
    ("delivered_response", "software counterfeit fallback"), ("turn_sha256", "f" * 64),
    ("receipt_sha256", "f" * 64), ("historical", "true"),
])
def test_mutated_terminal_proof_rejected(field, value):
    row = rejected({"role": "online_fast", "product_step": "initial", "request_id": "fixture",
                    "request_sha256": "a" * 64, "visible_content_sha256": "b" * 64,
                    "client_return_content_sha256": "b" * 64})
    turn = {"turn_id": "turn", "session_id": "session", "status": "failed", "source": "model_failed", "response": ""}
    proof = terminal_proof(turn, row)
    proof[field] = value
    with pytest.raises(RunError):
        subject.classify_online_position(turn, row, proof=proof)


def test_no_body_unknown_cost_and_memory_failure_not_accepted(full_metadata):
    full_metadata.failed_row["usage"].pop("cost")
    with pytest.raises(RunError, match="format_unqualified_primary_failure"):
        qualify(full_metadata)
    full_metadata.failed_row["usage"]["cost"] = 0.00001
    next(r for r in full_metadata.rows if r["role"] == "memory" and "experiment_position" in r)["status"] = "failed"
    with pytest.raises(RunError, match="e1br_primary_content_missing"):
        qualify(full_metadata)


def test_assessment_never_reopens_original_gate_or_claims_new_configuration():
    original = {"status": "evaluated", "main_gate": "pass", "formation_gate": "pass", "overall_gate": "pass",
                "effect_claim_eligible": True, "raw_model_effect_claim_eligible": True,
                "formation_evaluation": {"status": "local_labels_complete", "mechanical_formation_gate": "pass"}}
    before = deepcopy(original)
    result = subject.exploratory_assessment_view(original)
    assert original == before
    assert result["main_gate"] == result["formation_gate"] == result["overall_gate"] == "not_evaluable"
    assert result["exploratory"]["main_mechanical_comparison"] == "pass"
    assert result["effect_claim_eligible"] is result["raw_model_effect_claim_eligible"] is False


def test_rule_only_partial_result_is_not_model_improvement():
    result = subject.exploratory_assessment_view({"status": "not_evaluable", "reason": "rule_only",
        "reply_evaluation": {"mechanical_main_gate": "pass", "status": "labels_complete"}})
    assert result["exploratory"]["main_mechanical_comparison"] is None
    assert "mechanical_main_gate" not in result["reply_evaluation"]
    assert result["reply_evaluation"]["model_comparison"] is None


def test_existing_export_preserves_null_target_and_fixed_inventory(full_metadata):
    suite = load_e1b_cases()
    proto = evaluation.draft_protocol(suite, phase="main", case_ids=full_metadata.frozen["inputs"]["case_ids"], revision="e1br_v1")
    saved(full_metadata.root / "emotion-plan.json", {"source_sha256": suite.source_sha256, "synthetic": True})
    path = full_metadata.root / "main/e1b_c1/report.json"
    report = evaluation.read_json(path)
    report["branches"][0]["state"]["turns"][7].update(status="failed", source="model_failed", response="")
    saved(path, report)
    output = full_metadata.root / "blind-bundle"
    result = evaluation.export_packet(full_metadata.root, output, proto, suite=suite)
    _, key = evaluation.load_bundle(output)
    assert result["planned_positions"] == 216 and result["missing_positions"] == 3
    assert len(key["inventory"]) == 216
    missing = [row for row in key["inventory"] if row["status"] != "available"]
    assert len(missing) == 3 and all(row["id"] is None for row in missing)
