"""Disclosed 12-case software metadata, not a transport qualification or model result."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from aria.bench.e1c_source import seal_first_submission, load_first_submission
from aria.engine.policy.e1br_calls import logical_attempts
from scripts import e1b_evaluation as common, e1c_scoring as scoring  # noqa: F401
from scripts.e1br_format_qualification import classify_online_position
from scripts.run_e1b import text_sha, byte_sha
from tests import e1c_software_fixture as fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("software fixture must never access network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(common.canonical(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


@pytest.fixture
def data(tmp_path):
    source = tmp_path / "first"
    seal_first_submission(json.dumps(fixture.fixture_payload()).encode(), fixture.fixture_author(), source)
    first_sha = hashlib.sha256((source / "receipt.json").read_bytes()).hexdigest()
    submission = load_first_submission(source, expected_receipt_sha256=first_sha, allow_software_fixture=True)
    root = tmp_path / "software-run"
    rows = []
    for case in submission.suite.items:
        directory = root / "main" / case.id
        report = {"experiment": "e1c", "case_id": case.id, "source_sha256": submission.suite.source_sha256,
            "first_receipt_sha256": first_sha, "evidence_kind": "software_fixture",
            "case_order": [c.id for c in submission.suite.items],
            "preference_sources": [s.model_dump(mode="json") for s in case.preference_sources],
            "shared": {"state": {"turns": []}}, "branches": []}
        def row(role, session, arm, repeat, index, session_id, operation_id):
            value = {"request_id": f"software-{len(rows)}", "registration_sha256": "a" * 64,
                "role": role, "session_id": session_id, "operation_id": operation_id,
                "product_step": "initial", "status": "received", "request_sha256": "b" * 64,
                "visible_content_sha256": "c" * 64, "client_return_content_sha256": "c" * 64,
                "model_reply_sha256": text_sha("软件回复"), "reply_return_channel": "text",
                "experiment_position": [role, case.id, session, arm, repeat, index]}
            rows.append(value)
            return value
        captures = []
        for session in (1, 2):
            sid = f"{case.id}-s{session}"
            for index in range(12):
                receipt = row("online_fast", session, "shared", 0, index, sid, f"{sid}-t{index}")
                report["shared"]["state"]["turns"].append({"turn_id": receipt["operation_id"], "session_id": sid,
                    "status": "completed", "source": "model", "response": "软件回复",
                    "evidence_kind": "software_fixture", "text": case.current_input(session, index)})
            memory = row("memory", session, "shared", 0, None, sid, f"review-{sid}")
            row("reflection", session, "shared", 0, None, sid, f"review-{sid}")
            proposals = [None, {"do": "软件候选", "quotes": [{"turn_index": 1, "text": case.current_input(session, 1)}]}]
            first = {"session": session, "request_sha256": memory["request_sha256"], "request_hash_kind": "software_messages",
                "visible_response_sha256": memory["client_return_content_sha256"], "field_present": True,
                "raw_proposals": proposals, "parse_status": "present_array", "public_projection_dropped_fields": 0}
            first_hash = save(directory / f"memory-s{session}.first.json", first)
            product = {"product_status": "ok"}
            product_hash = save(directory / f"memory-s{session}.product.json", product)
            captures.append({**first, "complete": True, "product_outcome": product,
                "first_attempt_sha256": first_hash, "product_outcome_sha256": product_hash})
        save(directory / "raw-proposals.json", {"schema": "e1b_raw_proposals_v2", "case_id": case.id,
            "source_sha256": submission.suite.source_sha256, "sessions": captures})
        for arm in ("A0", "A1"):
            for repeat in (1, 2, 3):
                sid = f"{case.id}-{arm}-{repeat}"
                branch = {"arm": arm, "repeat": repeat, "state": {"turns": []}}
                for index in range(10):
                    receipt = row("online_fast", 3, arm, repeat, index, sid, f"{sid}-t{index}")
                    branch["state"]["turns"].append({"turn_id": receipt["operation_id"], "session_id": sid,
                        "status": "completed", "source": "model", "response": "软件回复",
                        "evidence_kind": "software_fixture", "text": case.current_input(3, index)})
                report["branches"].append(branch)
        save(directory / "report.json", report)
    assert len(rows) == 1056
    return SimpleNamespace(root=root, source=source, first_sha=first_sha, submission=submission,
                           state={"attempts": rows}, rows=rows)


def _position(case_id, arm, repeat, index):
    return ["online_fast", case_id, index // 12 + 1 if arm == "shared" else 3,
            arm, repeat, index % 12 if arm == "shared" else index]


def bind_inventory(run_root, submission, state, *, registration_sha256):
    """Local source/inventory precheck, deliberately not full transport qualification.

    Production receipt/authorization/wire/ledger validation must precede this
    after integration. The separate E1c runtime currently accepts software only.
    """
    require(submission.suite.provenance == "software_fixture_only", "candidate_inventory_software_only")
    current = load_first_submission(submission.directory, expected_receipt_sha256=submission.receipt_sha256,
                                    allow_software_fixture=True)
    require(current == submission, "candidate_source_changed")
    rows = [row for row in logical_attempts(state) if row["product_step"] == "initial"]
    require(all(row.get("status") in {"received", "failed"} for row in state["attempts"]),
            "candidate_pending_or_unknown_attempt_status")
    require(Counter(r["role"] for r in rows) == {"online_fast": 1008, "memory": 24, "reflection": 24},
            "candidate_fixed_logical_denominator_changed")
    require(len({r["request_id"] for r in rows}) == 1056
            and all(r["registration_sha256"] == registration_sha256 for r in rows),
            "candidate_receipt_registration_changed")
    by_position = {tuple(r["experiment_position"]): r for r in rows}
    require(len(by_position) == 1056, "candidate_position_reused")
    positions, hashes, captures, used = [], {}, {}, set()
    for case in submission.suite.items:
        directory = Path(run_root) / "main" / case.id
        report = common.read_json(directory / "report.json")
        require(report.get("experiment") == "e1c" and report.get("case_id") == case.id
                and report.get("source_sha256") == submission.suite.source_sha256
                and report.get("first_receipt_sha256") == submission.receipt_sha256
                and report.get("evidence_kind") == "software_fixture"
                and report.get("case_order") == [c.id for c in submission.suite.items]
                and report.get("preference_sources") == [s.model_dump(mode="json") for s in case.preference_sources],
                "candidate_report_source_or_provenance_changed")
        hashes[case.id] = byte_sha(directory / "report.json")
        raw = common.read_json(directory / "raw-proposals.json")
        common._validate_raw_capture_v2(raw, directory, case.id, submission.suite.source_sha256)
        require(len(raw["sessions"]) == 2 and all(r["complete"] for r in raw["sessions"]),
                "candidate_two_memory_originals_required")
        captures[case.id] = deepcopy(raw)
        branches = [{"arm": "shared", "repeat": 0, "state": report["shared"]["state"]}, *report["branches"]]
        require(len(branches) == 7 and {(b["arm"], b["repeat"]) for b in branches} ==
                {("shared", 0), *((a, r) for a in ("A0", "A1") for r in (1, 2, 3))},
                "candidate_branch_inventory_changed")
        session_ids = {}
        for branch in branches:
            turns = branch["state"]["turns"]
            require(len(turns) == (24 if branch["arm"] == "shared" else 10), "candidate_turn_inventory_changed")
            for index, turn in enumerate(turns):
                key = tuple(_position(case.id, branch["arm"], branch["repeat"], index))
                require(turn.get("text") == case.current_input(key[2], key[5]),
                        "candidate_disclosed_input_changed")
                row = by_position.get(key)
                require(row is not None and row["operation_id"] == turn["turn_id"]
                        and row["session_id"] == turn["session_id"] and row["status"] == "received",
                        "candidate_turn_receipt_binding_changed")
                if branch["arm"] == "shared":
                    number = index // 12 + 1
                    session_ids.setdefault(number, row["session_id"])
                    require(session_ids[number] == row["session_id"], "candidate_session_changed")
                classified = classify_online_position(turn, row, report_evidence_kind="software_fixture")
                used.add(key)
                positions.append({"position": list(key), "request_id": row["request_id"], **classified})
        require(len(set(session_ids.values())) == 2, "candidate_shared_sessions_not_distinct")
        for number in (1, 2):
            memory = by_position[("memory", case.id, number, "shared", 0, None)]
            reflection = by_position[("reflection", case.id, number, "shared", 0, None)]
            first = common.read_json(directory / f"memory-s{number}.first.json")
            require(memory["status"] == "received" and memory["session_id"] == reflection["session_id"] == session_ids[number]
                    and memory["operation_id"] == reflection["operation_id"]
                    and memory["request_sha256"] == first["request_sha256"]
                    and memory["client_return_content_sha256"] == first["visible_response_sha256"],
                    "candidate_memory_receipt_binding_changed")
            used.update((tuple(memory["experiment_position"]), tuple(reflection["experiment_position"])))
    require(used == set(by_position), "candidate_unbound_receipt")
    return {"schema": "e1c_inventory_candidate_v1", "status": "software_inventory_bound_not_transport_qualified",
        "source_sha256": submission.suite.source_sha256, "first_receipt_sha256": submission.receipt_sha256,
        "case_ids": [c.id for c in submission.suite.items], "source_report_sha256": hashes,
        "positions": positions, "raw_captures": captures, "fixed_logical_positions": 1056,
        "fixed_reply_positions": 216, "fixed_formation_cases": 12,
        "retained_experimental_answers": 0, "effect_claim_eligible": False, "paid_evaluable": False,
        "missing_qualification_layers": ["actual_wire_and_provider_receipts", "closed_ledger_and_authorizations",
            "reflection_first_and_format_correction_physical_bindings", "complete_source_inventory"]}


def bound(data):
    return bind_inventory(data.root, data.submission, data.state, registration_sha256="a" * 64)
