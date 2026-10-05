"""Portable metadata and export contracts; no provider or historical artifacts."""
from copy import deepcopy

import pytest

from scripts import e1b_evaluation as evaluation
from scripts import e1b_formation as formation
from scripts import e1br_memory_qualification as subject
from scripts.run_e1b import RunError, byte_sha
from tests.test_e1br_format_qualification import full_metadata, saved


@pytest.fixture
def memory_metadata(full_metadata, monkeypatch):
    data = full_metadata
    data.retained = data.rows[:731]
    data.state["attempts"] = data.rows[731:]
    data.checked_proofs = []
    data.retained_summary = {"software_only": True, "memory_source_baseline": {"synthetic_baseline": "bound"}}

    def check(directory, proof, row):
        # This fixture isolates the inventory/export adapter. The runtime tests
        # separately exercise full body/baseline/settled-cost validation.
        assert proof["request_id"] == row["request_id"]
        assert proof["receipt_sha256"] == evaluation.digest(row)
        assert proof["raw_proposals"] is None and proof["memory_completed"] is False
        data.checked_proofs.append(proof["request_id"])
    monkeypatch.setattr(subject, "validate_memory_terminal_proof", check)
    return data


def fail_memory(data, case="e1b_e1", number=2):
    path = data.root / "main" / case
    row = next(r for r in data.rows if r.get("experiment_position") == ["memory", case, number, "shared", 0, None])
    first_path = path / f"memory-s{number}.first.json"
    first = evaluation.read_json(first_path)
    first.update(parse_status="parse_error", field_present=None, raw_proposals=None)
    first_sha = saved(first_path, first)
    product = {"product_status": "failed", "accepted_count": None, "accepted_preferences": []}
    product_sha = saved(path / f"memory-s{number}.product.json", product)
    capture = evaluation.read_json(path / "raw-proposals.json")
    capture["sessions"][number - 1] = {**first, "complete": False, "product_outcome": product,
        "first_attempt_sha256": first_sha, "product_outcome_sha256": product_sha}
    saved(path / "raw-proposals.json", capture)
    proof = {"request_id": row["request_id"], "receipt_sha256": evaluation.digest(row),
             "raw_proposals": None, "memory_completed": False,
             "first_capture_sha256": evaluation.digest(first), "product_capture_sha256": evaluation.digest(product)}
    historical = row in data.retained
    proof.update(historical=historical, source_baseline=deepcopy(data.retained_summary["memory_source_baseline"]) if historical else None)
    report = evaluation.read_json(path / "report.json")
    report["shared"]["state"].setdefault("memory_terminal_positions", []).append(proof)
    audit = {"session_id": row["session_id"], "memory_completed": False, "gap": True, "outcome": "memory_terminal_missing"}
    report["shared"].setdefault("profile", {}).setdefault("offline_session_audit", []).append(audit)
    saved(path / f"memory-s{number}.terminal.reflection.json", {
        "schema": "e1br_memory_reflection_completion_v1", "memory_proof_sha256": evaluation.digest(proof),
        "memory_completed": False, "offline_audit": audit, "reflection_status": "ok"})
    saved(path / "report.json", report)
    return proof


def qualify(data, physical=None):
    return subject.qualify(data.root, data.frozen, data.state, retained_attempts=data.retained,
        retained_summary=data.retained_summary, logical_attempts=lambda state: state["attempts"],
        retained_position_bindings=data.legacy, retained_physical_attempts=physical)


def test_current_and_future_missing_memories_keep_fixed_denominator(memory_metadata):
    data = memory_metadata
    fail_memory(data)
    fail_memory(data, "e1b_e2", 1)
    result = qualify(data)
    assert result["retained_first_answers"] == 731 and result["new_physical_attempts"] == 325
    assert result["actual_counts"] == {"online_fast": 1008, "memory": 24, "reflection": 24}
    assert len(result["positions"]) == 1008 and len(result["scored_position_inventory"]) == 216
    assert len(result["missing_memory_positions"]) == 2
    assert all(row["raw_proposals"] is None and row["accepted_count"] is None
               and row["model_metrics"] is None for row in result["missing_memory_positions"])
    assert len(data.checked_proofs) == 2
    assert result["original_main_gate"] == result["original_formation_gate"] == "not_evaluable"


def test_missing_proof_is_not_inferred_from_failed_product(memory_metadata):
    data = memory_metadata
    fail_memory(data)
    path = data.root / "main/e1b_e1/report.json"
    report = evaluation.read_json(path)
    report["shared"]["state"].pop("memory_terminal_positions")
    saved(path, report)
    with pytest.raises(RunError, match="memory_incomplete_without_terminal_proof"):
        qualify(data)


def test_duplicate_or_success_proof_rejected(memory_metadata):
    data = memory_metadata
    proof = fail_memory(data)
    path = data.root / "main/e1b_e1/report.json"
    report = evaluation.read_json(path)
    report["shared"]["state"]["memory_terminal_positions"].append(deepcopy(proof))
    saved(path, report)
    with pytest.raises(RunError, match="memory_duplicate_terminal_proof"):
        qualify(data)


def test_old_reflection_correction_is_separate_from_731_logical(memory_metadata):
    data = memory_metadata
    first = next(row for row in data.retained if row["role"] == "reflection")
    correction = {**first, "request_id": "old-correction-physical", "product_step": "format_correction"}
    path = data.root / "main/e1b_a1"
    initial_path = path / "reflection-s1.first.json"
    body = {"request_id": correction["request_id"], "physical_request_ids": [correction["request_id"]],
            "request_sha256": correction["request_sha256"], "visible_response_sha256": correction["client_return_content_sha256"]}
    name = "reflection-s1.correction.first.json"
    checksum = saved(path / name, body)
    product = evaluation.read_json(path / "reflection-s1.product.json")
    product.update(format_correction_count=1,
        product_steps=[{"product_step": "initial", "first_path": initial_path.name, "first_sha256": byte_sha(initial_path)},
                       {"product_step": "format_correction", "first_path": name, "first_sha256": checksum}],
        format_attempts=[{"status": "shape_failed"}, {"status": "shape_valid"}])
    saved(path / "reflection-s1.product.json", product)
    with pytest.raises(RunError, match="e1br_reflection_physical_binding_changed"):
        qualify(data)
    result = qualify(data, [*data.retained, correction])
    assert result["retained_first_answers"] == 731 and result["retained_physical_attempts"] == 732
    assert result["reflection_secondary"]["format_corrections"] == 1


def test_physical_source_cannot_substitute_old_logical_receipt(memory_metadata):
    physical = deepcopy(memory_metadata.retained)
    physical[0]["status"] = "failed"
    with pytest.raises(RunError, match="memory_logical_first_missing_from_physical_lineage"):
        qualify(memory_metadata, physical)


def test_guard_modified_delivery_is_not_raw_model_eligible(memory_metadata):
    path = memory_metadata.root / "main/e1b_a1/report.json"
    report = evaluation.read_json(path)
    report["branches"][0]["state"]["turns"][7]["source"] = "rule_protection"
    saved(path, report)
    result = qualify(memory_metadata)
    row = next(row for row in result["positions"] if row["case_id"] == "e1b_a1"
               and row["arm"] == "A0" and row["repeat"] == 1 and row["turn"] == 7)
    assert row["delivery"] == "guard_modified" and row["model_reply_eligible"] is False


def test_existing_formation_export_keeps_failed_memory_null(memory_metadata):
    data = memory_metadata
    fail_memory(data)
    qualified = qualify(data)
    suite = subject.load_e1b_cases()
    protocol = evaluation.draft_protocol(suite, phase="main", case_ids=data.frozen["inputs"]["case_ids"], revision="e1br_v1")
    out = data.root / "formation"
    formation.export_packet(data.root, out, protocol, suite=suite)
    _, mapping = formation.load_bundle(out)
    result = subject.validate_formation_missing(qualified, mapping)
    assert result["fixed_case_denominator"] == 12 and result["missing_cases"] == ["e1b_e1"]
    assert result["missing_metrics"] is None
    aggregated = formation.aggregate(out)
    assert aggregated["formation_gate"] == "not_evaluable" and aggregated["proposal_rate"] is None
    assert aggregated["raw_problem_cases"] == {"e1b_e1": ["memory_2_array_unavailable"]}


def test_extra_raw_problem_not_exempted(memory_metadata):
    fail_memory(memory_metadata)
    qualified = qualify(memory_metadata)
    suite = subject.load_e1b_cases()
    protocol = evaluation.draft_protocol(suite, phase="main", case_ids=memory_metadata.frozen["inputs"]["case_ids"], revision="e1br_v1")
    out = memory_metadata.root / "formation"
    formation.export_packet(memory_metadata.root, out, protocol, suite=suite)
    _, mapping = formation.load_bundle(out)
    mapping["inventory"][0]["problems"].append("unblindable_candidate_content")
    with pytest.raises(RunError, match="memory_formation_unqualified_problem"):
        subject.validate_formation_missing(qualified, mapping)


def test_model_unknown_cost_cannot_enter_terminal_adapter(memory_metadata):
    fail_memory(memory_metadata)
    memory_metadata.rows[730]["status"] = "failed"
    with pytest.raises(RunError, match="e1br_primary_content_missing"):
        qualify(memory_metadata)


def test_completion_cannot_turn_missing_memory_into_success(memory_metadata):
    fail_memory(memory_metadata)
    path = memory_metadata.root / "main/e1b_e1/memory-s2.terminal.reflection.json"
    completion = evaluation.read_json(path)
    completion["memory_completed"] = True
    saved(path, completion)
    with pytest.raises(RunError, match="memory_missing_completion_or_gap_changed"):
        qualify(memory_metadata)


def test_historical_proof_cannot_choose_another_baseline(memory_metadata):
    fail_memory(memory_metadata)
    memory_metadata.retained_summary["memory_source_baseline"] = {"different": "baseline"}
    with pytest.raises(RunError, match="memory_terminal_frozen_baseline_changed"):
        qualify(memory_metadata)


def actual_terminal_fixture(data):
    """Create a complete synthetic actual-wire receipt for the shared validator."""
    from scripts import e1br_memory_runtime as runtime
    fail_memory(data)
    directory = data.root / "main/e1b_e1"
    row = data.rows[730]
    content = '{"stated_preferences":[{"evidence_quote":"bad "quote""}]}'
    visible = {"request_id": row["request_id"], "role": "memory", "content": content}
    visible_sha = saved(directory / "memory-s2.terminal.visible.json", visible)
    row.update(http_status=200, observation={"body_status": "parsed", "envelope_kind": "success"},
        receipt_identity={"version": "receipt_identity_v1", "error_field_status": "absent",
                          "model_status": "recognized", "provider_status": "recognized"},
        visible_original_sha256=visible_sha, visible_content_sha256=runtime.sha(content.encode()),
        client_return_content_sha256=runtime.sha(content.encode()), usage={"cost": .0001, "prompt_tokens": 1,
            "completion_tokens": 1, "reasoning_tokens": 0},
        reservation={"reserved_cost_usd": "0.001", "input_token_plan_bound": 100, "output_token_limit": 100})
    first = evaluation.read_json(directory / "memory-s2.first.json")
    first["visible_response_sha256"] = row["visible_content_sha256"]
    first_sha = saved(directory / "memory-s2.first.json", first)
    product = evaluation.read_json(directory / "memory-s2.product.json")
    product.update(session=2, diagnostics_available=False)
    product_sha = saved(directory / "memory-s2.product.json", product)
    capture = evaluation.read_json(directory / "raw-proposals.json")
    capture["sessions"][1] = {**first, "complete": False, "product_outcome": product,
        "first_attempt_sha256": first_sha, "product_outcome_sha256": product_sha}
    saved(directory / "raw-proposals.json", capture)
    report = evaluation.read_json(directory / "report.json")
    turns = [{"text": turn["text"]} for turn in report["shared"]["state"]["turns"]]
    job = {"session_id": row["session_id"], "memory_completed": False, "reflection_completed": False,
           "source_completeness": "complete", "active_stage": "memory", "last_error_code": "memory_extraction_failed",
           "created_at": "2026-10-05T00:00:00+00:00", "expires_at": "2026-10-06T00:00:00+00:00",
           "turns": [{"user": turn["text"]} for turn in turns[-12:]]}
    profile = {"offline_sessions": [job], "pending_session": None, "episodic_memory": ["synthetic old fact"]}
    run = {"run_id": "synthetic-retained-run", "sessions": [{"session_id": "e1b_e1-s1", "status": "sealed"},
        {"session_id": row["session_id"], "status": "sealed"}], "turns": turns,
        "reviews": [{"status": "failed", "session_id": row["session_id"], "request_id": row["operation_id"]}]}
    proof = runtime.terminal_memory_evidence(row, visible, first, product, profile, run, row["experiment_position"])
    baseline = {key: proof[key] for key in ("original_profile_sha256", "original_job_sha256", "original_run_sha256", "old_failed_review_sha256")}
    baseline.update(run_id=run["run_id"], model_allowance=None, source_manifest_sha256="9" * 64)
    data.retained_summary["memory_source_baseline"] = baseline
    proof.update(historical=True, source_baseline=deepcopy(baseline),
        proof_path="memory-s2.terminal.json", visible_path="memory-s2.terminal.visible.json",
        first_path="memory-s2.first.json", product_path="memory-s2.product.json",
        baseline_profile_path="memory-s2.terminal.profile.json", baseline_run_path="memory-s2.terminal.run.json")
    saved(directory / proof["baseline_profile_path"], profile)
    saved(directory / proof["baseline_run_path"], run)
    saved(directory / proof["proof_path"], proof)
    report["shared"]["state"]["memory_terminal_positions"] = [proof]
    saved(directory / "report.json", report)
    completion = evaluation.read_json(directory / "memory-s2.terminal.reflection.json")
    completion["memory_proof_sha256"] = evaluation.digest(proof)
    saved(directory / "memory-s2.terminal.reflection.json", completion)
    return row, directory, proof


def test_actual_shared_proof_validator_is_used_by_qualification(memory_metadata, monkeypatch):
    from scripts.e1br_memory_runtime import validate_memory_terminal_proof
    actual_terminal_fixture(memory_metadata)
    monkeypatch.setattr(subject, "validate_memory_terminal_proof", validate_memory_terminal_proof)
    result = qualify(memory_metadata)
    assert len(result["missing_memory_positions"]) == 1
    assert result["missing_memory_positions"][0]["model_metrics"] is None


def test_actual_proof_without_known_cost_is_rejected(memory_metadata, monkeypatch):
    from scripts.e1br_memory_runtime import validate_memory_terminal_proof
    row, _, _ = actual_terminal_fixture(memory_metadata)
    row["usage"].pop("cost")
    monkeypatch.setattr(subject, "validate_memory_terminal_proof", validate_memory_terminal_proof)
    with pytest.raises(ValueError, match="memory_complete_known_received_required"):
        qualify(memory_metadata)
