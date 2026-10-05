"""All 12 disclosed metadata cases; these are not transport/model evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from aria.bench.e1c_runner import E1cRunner
from scripts import e1b_evaluation as common
from tests.test_e1c_execution import entry, qualification, no_network_or_env  # noqa: F401
from tests.e1c_evaluation_fixture import data, save  # noqa: F401


@pytest.fixture
def full(data):
    for case in data.submission.suite.items:
        directory = data.root / "main" / case.id
        report = common.read_json(directory / "report.json")
        report.update(E1cRunner._report_metadata(SimpleNamespace(suite=data.submission.suite,
            submission=data.submission), case, "main"))
        report.update(evidence_kind="real_model", phase="main")
        for state in [report["shared"]["state"], *(b["state"] for b in report["branches"])]:
            for turn in state["turns"]:
                turn["evidence_kind"] = "real_model"
        save(directory / "report.json", report)
        capture = common.read_json(directory / "raw-proposals.json")
        for number in (1, 2):
            memory = next(r for r in data.rows if r["experiment_position"] == ["memory", case.id, number, "shared", 0, None])
            ref = next(r for r in data.rows if r["experiment_position"] == ["reflection", case.id, number, "shared", 0, None])
            first = common.read_json(directory / f"memory-s{number}.first.json")
            first["request_hash_kind"] = "actual_wire"
            first_sha = save(directory / f"memory-s{number}.first.json", first)
            capture["sessions"][number - 1].update(request_hash_kind="actual_wire", first_attempt_sha256=first_sha)
            save(directory / f"memory-s{number}.attempt.json", {"session_id": memory["session_id"]})
            ref_first = {"request_id": ref["request_id"], "physical_request_ids": [ref["request_id"]],
                "request_sha256": ref["request_sha256"], "visible_response_sha256": ref["client_return_content_sha256"]}
            filename = f"reflection-s{number}.first.json"
            ref_sha = save(directory / filename, ref_first)
            save(directory / f"reflection-s{number}.product.json", {
                "schema": "e1br_reflection_product_v1", "session_id": ref["session_id"], "capture_error": None,
                "updater_raised": False, "product_status": "ok", "format_correction_count": 0,
                "product_steps": [{"product_step": "initial", "first_path": filename, "first_sha256": ref_sha}],
                "first_sha256": ref_sha, "format_attempts": [{"status": "shape_valid"}]})
        save(directory / "raw-proposals.json", capture)
    data.state["ledger_id"] = "SOFTWARE_METADATA_ONLY_NOT_ACTUAL_TRANSPORT"
    data.frozen = {"seal_sha256": "a" * 64, "case_ids": [c.id for c in data.submission.suite.items]}
    return data


def bind(full):
    return qualification.bind_reports(full.root, full.frozen, full.submission, full.state)


def test_all_twelve_capture_and_first_body_bindings(full):
    shared, positions, captures, initial = bind(full)
    assert len(initial) == 1056 and len(positions) == 1008 and len(captures) == 12
    assert len(shared["scored_position_inventory"]) == 216
    assert len(shared["reflection_originals"]) == 24
    assert shared["online_reliability"]["plain_text_received"] == 1008
    assert shared["online_reliability"]["decision_json_positions"] == 0
    assert shared["effect_claim_eligible"] is False


@pytest.mark.parametrize("change", ["input", "arm", "reflection_request", "reflection_steps", "source", "failed_primary", "drop"])
def test_full_candidate_binding_rejects_meaningful_corruption(full, change):
    directory = full.root / "main/e1c_a1"
    path = directory / "report.json"
    report = common.read_json(path)
    if change == "input":
        report["shared"]["state"]["turns"][0]["text"] = "not the first source"
    elif change == "arm":
        for branch in report["branches"]:
            branch["arm"] = "A1" if branch["arm"] == "A0" else "A0"
    elif change == "source":
        report["author_provenance_sha256"] = "f" * 64
    elif change == "failed_primary":
        full.rows[0]["status"] = "failed"
    elif change == "drop":
        full.rows.pop()
    elif change == "reflection_request":
        p = directory / "reflection-s1.first.json"
        value = common.read_json(p)
        value["physical_request_ids"].append("phantom")
        first_sha = save(p, value)
        p = directory / "reflection-s1.product.json"
        value = common.read_json(p); value["first_sha256"] = first_sha
        value["product_steps"][0]["first_sha256"] = first_sha
        save(p, value)
    else:
        p = directory / "reflection-s1.product.json"
        value = common.read_json(p); value["format_correction_count"] = 1
        save(p, value)
    save(path, report)
    with pytest.raises((ValueError, RuntimeError)):
        bind(full)
