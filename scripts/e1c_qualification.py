"""Dedicated E1c receipt/source/closed-ledger qualification, before scoring.

Only a private instance of the shared complete capture verifier is adapted to
the E1c source. No original E1b globals, datasets or qualification gate change.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

from scripts import e1b_evaluation as common

HERE = Path(__file__).resolve().parent


def entry_module():
    from scripts import e1c_execution
    return e1c_execution


def require(condition, code):
    if not condition:
        raise ValueError(code)


def check_case_source(directory, submission, case):
    """Bind every actual disclosed input, not just a reconstructed answer key."""
    report = common.read_json(directory / "report.json")
    require(report.get("experiment") == "e1c" and report.get("case_id") == case.id
            and report.get("source_sha256") == submission.suite.source_sha256
            and report.get("first_receipt_sha256") == submission.receipt_sha256
            and report.get("author_provenance_sha256") == submission.provenance_sha256
            and report.get("evidence_kind") == "real_model" and report.get("phase") == "main"
            and report.get("case_order") == [c.id for c in submission.suite.items]
            and report.get("preference_sources") == [s.model_dump(mode="json") for s in case.preference_sources]
            and report.get("planned_case_denominator") == 12
            and report.get("planned_basic_positions") == 1056
            and report.get("planned_reply_denominator") == 216
            and report.get("planned_formation_denominator") == 12
            and report.get("retained_experimental_answers") == 0,
            "e1c_report_first_source_changed")
    branches = [{"arm": "shared", "repeat": 0, "state": report["shared"]["state"]}, *report["branches"]]
    require(len(branches) == 7 and {(b["arm"], b["repeat"]) for b in branches} ==
            {("shared", 0), *((a, r) for a in ("A0", "A1") for r in (1, 2, 3))},
            "e1c_branch_inventory_changed")
    for branch in branches:
        turns = branch["state"]["turns"]
        require(len(turns) == (24 if branch["arm"] == "shared" else 10), "e1c_turn_inventory_changed")
        require(not branch["state"].get("format_terminal_positions"), "e1c_unapproved_online_failure_continuation")
        for index, turn in enumerate(turns):
            session, number = (index // 12 + 1, index % 12) if branch["arm"] == "shared" else (3, index)
            require(turn.get("text") == case.current_input(session, number), "e1c_disclosed_input_changed")
    capture = common.read_json(directory / "raw-proposals.json")
    common._validate_raw_capture_v2(capture, directory, case.id, submission.suite.source_sha256)
    return report, capture


def bind_reports(run_root, frozen, source, state):
    """Complete capture/inventory layer; does not itself grant paid eligibility."""
    entry = entry_module()
    run_root = Path(run_root)
    logical = entry.guard_code.CALLS.logical_attempts(state)
    initial = [r for r in logical if r["product_step"] == "initial"]
    require(len(initial) == 1056
            and {tuple(r["experiment_position"]) for r in initial} == entry.guard_code.positions(source.suite),
            "e1c_fixed_inventory_incomplete")
    require(all(r["status"] == "received" for r in initial if r["role"] != "reflection"),
            "e1c_primary_first_body_failed")
    captures = {}
    for case in source.suite.items:
        directory = run_root / "main" / case.id
        entry.regular(directory / "report.json")
        _, captures[case.id] = check_case_source(directory, source, case)

    name = "scripts._e1c_complete_capture_qualification"
    spec = importlib.util.spec_from_file_location(name, entry.ROOT / "scripts/e1br_format_qualification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_e1b_cases = lambda: source.suite
    shared = module.qualify(run_root, {"seal_sha256": frozen["seal_sha256"], "inputs": {"case_ids": frozen["case_ids"]}},
        state, retained_attempts=[], retained_summary={}, logical_attempts=entry.guard_code.CALLS.logical_attempts)
    positions = deepcopy(shared["positions"])
    rows = {r["request_id"]: r for r in initial}
    for position in positions:
        position["position"] = rows[position["request_id"]]["experiment_position"]
    return shared, positions, captures, initial


def qualify(campaign, *, authorization_sha256, write=True):
    entry = entry_module()
    root = Path(campaign).resolve()
    frozen, source = entry.verify_authority(root, authorization_sha256)
    receipt = common.read_json(root / "execution-receipt.json")
    require(receipt["receipt_sha256"] == common.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            and receipt["status"] == "generated" and receipt["complete_cases"] == 12
            and receipt["registration_sha256"] == frozen["seal_sha256"]
            and receipt["authorization_sha256"] == authorization_sha256
            and receipt["inherited_finance"] == frozen["finance"]
            and receipt["mode"] == frozen["mode"]
            and receipt["software_fixture_only"] == (frozen["mode"] == "software_fixture")
            and receipt["retained_experimental_answers"] == 0,
            "e1c_complete_receipt_required")
    state = receipt["accounting"]
    ledger_sha = entry.guard_code.closed_ledger(root, frozen, state)
    run_root = Path(receipt["run_root"]).resolve()
    require(run_root.is_relative_to(root) and run_root.is_dir(), "e1c_run_root_outside_campaign")
    shared, positions, captures, initial = bind_reports(run_root, frozen, source, state)
    result = {"schema": "e1c_generation_qualification_v1", "status": "qualified",
        "mode": frozen["mode"], "evidence_kind": frozen["mode"],
        "software_fixture_only": frozen["mode"] == "software_fixture", "effect_claim_eligible": False,
        "source_sha256": source.suite.source_sha256, "first_receipt_sha256": source.receipt_sha256,
        "case_ids": frozen["case_ids"], "source_report_sha256": shared["source_report_sha256"],
        "positions": positions, "raw_captures": captures,
        "memory_originals": shared["memory_originals"], "reflection_originals": shared["reflection_originals"],
        "reflection_secondary": shared["reflection_secondary"], "online_reliability": shared["online_reliability"],
        "fixed_logical_positions": 1056, "fixed_reply_positions": 216, "fixed_formation_cases": 12,
        "retained_experimental_answers": 0, "new_physical_attempts": len(state["attempts"]),
        "registration_sha256": frozen["seal_sha256"], "authorization_sha256": authorization_sha256,
        "evaluation_protocol": frozen["evaluation_protocol"],
        "evaluation_protocol_sha256": frozen["evaluation_protocol_sha256"],
        "ledger_sha256": ledger_sha, "execution_receipt_sha256": entry.sha(root / "execution-receipt.json"),
        "finance": frozen["finance"], "new_accounting": {k: v for k, v in state.items() if k != "attempts"},
        "source_lineage": {"single_registered_source": True, "inherited_experimental_answers": 0,
            "source_inventory_sha256": common.digest(frozen["source"]),
            "all_initial_request_registrations": {r["request_id"]: r["registration_sha256"] for r in initial}},
        "scope": "Generation integrity only. Software results cannot pass real-model effect gates; semantic first labels are still required."}
    result["qualification_digest"] = common.digest(result)
    if write:
        common.write_once(root / "qualification.json", result)
    return result
