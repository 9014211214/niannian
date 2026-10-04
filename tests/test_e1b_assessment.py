"""Read-only assessment rejects incomplete/tampered evidence without billing."""
from copy import deepcopy
from decimal import Decimal
import json
import sqlite3
import urllib.request

import pytest

from scripts import e1b_assessment as assessment
from scripts import e1b_evaluation as evaluation


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("assessment must not contact a model")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def ledger_fixture(tmp_path):
    base = assessment.execution.campaign(tmp_path)
    base.mkdir(parents=True)
    path = base / "allowance.sqlite3"
    policy = {"synthetic_fixture": True}
    records = [{"request_id": f"synthetic-{n}", "status": "received", "usage": {"cost": 0.1}} for n in range(2)]
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE attempts(sequence INTEGER PRIMARY KEY,request_id TEXT,record TEXT);"
                         "CREATE TABLE settings(id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT);"
                         "CREATE TABLE reservation_policy(id INTEGER,policy TEXT);"
                         "CREATE TABLE reservation_identity(id INTEGER,ledger_id TEXT);")
        db.execute("INSERT INTO settings VALUES(1,1,1636,'0.5',NULL)")
        db.execute("INSERT INTO reservation_policy VALUES(1,?)", (json.dumps(policy, sort_keys=True),))
        db.execute("INSERT INTO reservation_identity VALUES(1,'synthetic-ledger')")
        for index, row in enumerate(records):
            db.execute("INSERT INTO attempts VALUES(?,?,?)", (index, row["request_id"], json.dumps(row)))
    return path, {"attempts": records, "ledger_id": "synthetic-ledger", "max_observed_cost_usd": "0.5",
                  "known_cost_usd": "0.2"}, {"metadata": {"policy": policy}}


def test_closed_ledger_audit_is_read_only(tmp_path):
    path, state, frozen = ledger_fixture(tmp_path)
    before = path.read_bytes()
    observed = assessment._ledger(tmp_path, state, frozen)
    assert observed == assessment.execution.byte_sha(path)
    assert path.read_bytes() == before and len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("damage", ["row", "policy", "identity", "sum", "stopped", "journal"])
def test_ledger_corruption_never_becomes_qualified(tmp_path, damage):
    path, state, frozen = ledger_fixture(tmp_path)
    if damage == "sum":
        state["known_cost_usd"] = "0.1"
    elif damage == "journal":
        path.with_name(path.name + "-journal").write_text("pending transaction")
    else:
        with sqlite3.connect(path) as db:
            if damage == "row":
                db.execute("UPDATE attempts SET record='{}' WHERE sequence=0")
            elif damage == "policy":
                db.execute("UPDATE reservation_policy SET policy='{}'")
            elif damage == "identity":
                db.execute("UPDATE reservation_identity SET ledger_id='new-zero-budget'")
            else:
                db.execute("UPDATE settings SET halted='model_or_storage_failed'")
    with pytest.raises(assessment.AssessmentError):
        assessment._ledger(tmp_path, state, frozen)


def test_missing_campaign_is_not_evaluable_and_creates_nothing(tmp_path):
    assert list(tmp_path.iterdir()) == []
    result = assessment.assess(tmp_path)
    assert result["overall_gate"] == "not_evaluable" and result["effect_claim_eligible"] is False
    assert result["model_calls"] == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changed", ["source", "identity", "phase_inputs", "seal"])
def test_saved_registration_cannot_be_reinterpreted_after_mutation(tmp_path, monkeypatch, changed):
    run = assessment.execution
    source, inputs = [{"path": "synthetic.py", "sha256": "a" * 64}], {"case_ids": ["synthetic"]}
    frozen = {"stage": "main", "protocol": run.protocol(), "workspace_identity": run.workspace_identity(tmp_path),
              "source": source, "inputs": inputs}
    frozen["seal_sha256"] = evaluation.digest(frozen)
    receipt = {"registration_sha256": frozen["seal_sha256"]}
    evaluation.write_once(run.campaign(tmp_path) / "main/registration.json", frozen)
    monkeypatch.setattr(run, "inventory", lambda root: deepcopy(source))
    monkeypatch.setattr(run, "stage_inputs", lambda root, stage: deepcopy(inputs))
    assert assessment._registration(tmp_path, "main", receipt) == frozen
    if changed == "source":
        monkeypatch.setattr(run, "inventory", lambda root: [])
    elif changed == "identity":
        monkeypatch.setattr(run, "workspace_identity", lambda root: "another-checkout")
    elif changed == "phase_inputs":
        monkeypatch.setattr(run, "stage_inputs", lambda root, stage: {"case_ids": ["chosen-afterwards"]})
    else:
        receipt["registration_sha256"] = "0" * 64
    with pytest.raises(assessment.AssessmentError):
        assessment._registration(tmp_path, "main", receipt)


@pytest.mark.parametrize("changed", ["none", "guard_modified", "rule_only", "disagreement", "formation_fail", "missing_labels"])
def test_registered_product_gate_and_raw_effect_scope_remain_separate(tmp_path, monkeypatch, changed):
    # Provenance checks have separate tamper tests. Here inject validated
    # software evidence to exercise only final gate joining, never a model score.
    run = assessment.execution
    rows, receipts, registrations = [], {}, {}
    for stage in run.STAGES:
        for role, count in run.COUNTS[stage].items():
            rows.extend({"role": role, "e1b_stage": stage, "registration_sha256": stage,
                         "status": "received", "usage": {"cost": 0.000001}} for _ in range(count))
        state = {"attempts": deepcopy(rows), "calls_used": len(rows), "unknown_cost_requests": 0,
                 "stopped": stage == "judge", "stop_reason": "request_limit_reached" if stage == "judge" else None,
                 "ledger_id": "synthetic", "reservation_policy_sha256": "synthetic-policy",
                 "max_observed_cost_usd": "0.5", "known_cost_usd": "0.001636"}
        receipts[stage] = {"accounting": state, "formation_packet_sha256": evaluation.digest({}),
                           "formation_mapping_sha256": evaluation.digest({"protocol": {}})}
        registrations[stage] = {"seal_sha256": stage, "source": [], "protocol": {}, "metadata": {"policy": {}}}
    authorization = {"status": "test_fixture_only"}
    binding = {"campaign_authorization_sha256": evaluation.digest(authorization),
               "source_sha256": evaluation.digest([]), "protocol_sha256": evaluation.digest({}),
               "policy_sha256": evaluation.digest({})}
    monkeypatch.setattr(run, "read_receipt", lambda root, stage: receipts[stage])
    monkeypatch.setattr(assessment, "_registration", lambda root, stage, receipt: registrations[stage])
    monkeypatch.setattr(run, "validate_authorization", lambda *a, **k: (authorization, Decimal("0.5")))
    monkeypatch.setattr(evaluation, "read_json", lambda path: binding if path.name == "budget-binding.json" else authorization)
    monkeypatch.setattr(assessment, "_ledger", lambda *a: "fixture-ledger-hash")
    positions = [{"arm": "A1", "turn": 7, "delivery": "model_unmodified", "model_reply_eligible": True}
                 for _ in range(216)]
    if changed in {"guard_modified", "rule_only"}:
        positions[0].update(delivery=changed, model_reply_eligible=False)
    monkeypatch.setattr(assessment, "_qualification", lambda *a: ({"positions": positions}, {}, {"protocol": {}}))
    monkeypatch.setattr(assessment, "_secondary_labels", lambda *a: None)
    monkeypatch.setattr(assessment.formation, "load_bundle", lambda *a: ({}, {"protocol": {}}))
    replies = {"status": "labels_complete", "valid_primary_labels": 216, "missing_generation_positions": 0,
               "mechanical_main_gate": "pass", "secondary": {"status": "complete", "sufficient_agreement": True}}
    formed = {"status": "local_labels_complete", "mechanical_formation_gate": "pass"}
    if changed == "disagreement":
        replies["secondary"]["sufficient_agreement"] = False
    elif changed == "formation_fail":
        formed["mechanical_formation_gate"] = "fail"
    elif changed == "missing_labels":
        replies["valid_primary_labels"] = 215
    monkeypatch.setattr(evaluation, "aggregate", lambda *a: replies)
    monkeypatch.setattr(assessment.formation, "aggregate", lambda *a: formed)
    result = assessment.assess(tmp_path)
    if changed in {"none", "guard_modified"}:
        assert result["overall_gate"] == "pass"
        assert result["raw_model_effect_claim_eligible"] is (changed == "none")
        assert result["effect_layer"].startswith("delivered_product_configuration")
    elif changed == "formation_fail":
        assert result["overall_gate"] == "fail" and result["main_gate"] == "pass"
    elif changed == "disagreement":
        assert result["overall_gate"] == "inconclusive" and result["effect_claim_eligible"] is False
    else:
        assert result["overall_gate"] == "not_evaluable" and result["effect_claim_eligible"] is False
