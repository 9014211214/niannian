"""Pending formal-v3 authorization and immutable cost-chain software checks."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request

import pytest

from scripts import run_e1b as original
from scripts import run_e1b_repair as old
from scripts import run_e1b_reflection_repair as repair
from aria.engine.policy import reflection_json
from tests.test_e1b_repair import prior  # noqa: F401 - provides the isolated v1 evidence fixture


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))


def tree(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": original.byte_sha(p)}
            for p in sorted((root / repair.PARENT).rglob("*")) if p.is_file()]


def _historical_reflection_protocol(monkeypatch):
    """Run synthetic v3-v7 entry tests under their actual v1 wire contract.

    Production uses v2. Only these historical software fixtures restore the
    unfactored three-branch schema; changing a version label alone would mix
    contracts. The original local content/evidence validators remain active.
    """
    def wire_v1():
        return reflection_json._project_wire_keywords(
            reflection_json.reflection_response_schema())

    monkeypatch.setattr(reflection_json, "REFLECTION_JSON_PROTOCOL_VERSION", "reflection_json_schema_v1")
    monkeypatch.setattr(reflection_json, "reflection_wire_schema", wire_v1)
    # This entry copied both imports at module load; dynamically loaded wrappers
    # import the same patched fixture protocol while the fixture is active.
    monkeypatch.setattr(repair, "REFLECTION_JSON_PROTOCOL_VERSION", "reflection_json_schema_v1")
    monkeypatch.setattr(repair, "reflection_wire_schema", wire_v1)


@pytest.fixture
def history(prior, monkeypatch):
    _historical_reflection_protocol(monkeypatch)
    root, policy = prior
    monkeypatch.setattr(repair, "PARENT", Path("v2"))
    monkeypatch.setattr(repair, "CAMPAIGN", Path("v3"))
    monkeypatch.setattr(repair, "ARCHIVE", Path("archive-v2"))
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", None)
    sources = []
    for index in range(115):
        relative = f"aria/source{index}.py"
        path = root / repair.ARCHIVE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic second-version source\n")
        sources.append({"path": relative, "sha256": original.byte_sha(path)})
    seals, receipts, attempts = {}, {}, []
    for stage in ("prescreen", "main"):
        registration = {"source": sources, "metadata": {"policy": policy}, "stage": stage, "fixture": True}
        registration["seal_sha256"] = original.digest(registration)
        seals[stage] = registration["seal_sha256"]
        original.write_once(root / repair.PARENT / stage / "registration.json", registration)
        if stage == "prescreen":
            original.write_once(root / repair.ARCHIVE / "registration.json", registration)
        roles = ["online_fast"] * 540 if stage == "prescreen" else ["online_fast"] * 12 + ["memory", "reflection"]
        for sequence, role in enumerate(roles, 1):
            relative = Path(stage) / "wire-requests" / f"{sequence:04d}.json"
            wire = root / repair.PARENT / relative
            wire.parent.mkdir(parents=True, exist_ok=True)
            wire.write_bytes(b"{}")
            cost = "0.032192042" if stage == "prescreen" else "0.03514895"
            attempts.append({"request_id": str(len(attempts)), "status": "received", "e1b_stage": stage,
                "role": role, "registration_sha256": seals[stage], "wire_request_path": relative.as_posix(),
                "request_sha256": original.byte_sha(wire), "usage": {"cost": cost if sequence == 1 else "0"}})
        encoded_policy = json.dumps(policy, sort_keys=True, allow_nan=False)
        state = {"attempts": deepcopy(attempts), "ledger_id": "synthetic-second-ledger", "max_requests": 1636,
            "max_observed_cost_usd": "0.606969138", "calls_used": len(attempts),
            "reservation_policy_sha256": hashlib.sha256(encoded_policy.encode()).hexdigest(),
            "unknown_cost_requests": 0, "stopped": stage == "main",
            "stop_reason": "model_or_storage_failed" if stage == "main" else None,
            "reserved_cost_usd": "0", "known_cost_usd": "0.032192042" if stage == "prescreen" else str(repair.PARENT_COST)}
        value = {"accounting": state, "registration_sha256": seals[stage], "stage": stage,
            "status": "completed" if stage == "prescreen" else "not_evaluable",
            "old_unknown_requests_retained": 2, "cumulative_prior_known_usd": str(old.PRIOR_KNOWN)}
        value["receipt_sha256"] = original.digest(value)
        receipts[stage] = value["receipt_sha256"]
        original.write_once(root / repair.PARENT / stage / "execution-receipt.json", value)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts (sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?,?,?)", [(i, row["request_id"], json.dumps(row))
                                                              for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings (id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES (1,1,1636,'0.606969138','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy (id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES (1,?)", (encoded_policy,))
        db.execute("CREATE TABLE reservation_identity (id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES (1,'synthetic-second-ledger')")
    monkeypatch.setattr(repair, "PARENT_SEALS", seals)
    monkeypatch.setattr(repair, "PARENT_RECEIPTS", receipts)
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    monkeypatch.setattr(repair, "PARENT_FILES", len(tree(root)))
    return root, policy


def test_historical_fixture_restores_actual_v1_wire_without_admitting_production_v2(tmp_path, monkeypatch):
    current = reflection_json.reflection_request_format()
    assert reflection_json.REFLECTION_JSON_PROTOCOL_VERSION == "reflection_json_schema_v2"
    monkeypatch.setattr(repair, "parent_evidence", lambda _: {"software_fixture": True})
    entry = repair.build_engine(tmp_path)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("must reject before credentials"))
    destination = tmp_path / "must-not-register.json"
    with pytest.raises(original.RunError, match="revised_reflection_json_protocol_required"):
        entry.prepare(tmp_path, tmp_path, destination)
    assert not destination.exists()
    with monkeypatch.context() as historical:
        _historical_reflection_protocol(historical)
        protocol = entry.protocol()
        schema = reflection_json.reflection_request_format()["json_schema"]["schema"]
        assert protocol["product_configuration"]["reflection_json_protocol_version"] == "reflection_json_schema_v1"
        assert protocol["product_configuration"]["reflection_wire_schema_sha256"] == original.digest(schema)
        assert schema["$defs"]["_Change"]["properties"]["atomic_claim"]["anyOf"] == [
            {"$ref": "#/$defs/" + name}
            for name in ("_StatedPreference", "_ObservedResponse", "_RepeatedPattern")]
        assert "_AtomicClaimNullOutcome" not in schema["$defs"]
        assert schema != current["json_schema"]["schema"]
    assert reflection_json.reflection_request_format() == current
    assert reflection_json.REFLECTION_JSON_PROTOCOL_VERSION == "reflection_json_schema_v2"
    with pytest.raises(original.RunError, match="revised_reflection_json_protocol_required"):
        entry.protocol()


def prepare(history, monkeypatch, *, approved=False):
    root, policy = history
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", {
        "human_quote": "SYNTHETIC authorization fixture, not a real user approval",
        "human_message_reference": "synthetic fixture", "budget_choice": repair.BUDGET_CHOICE} if approved else None)
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_reflection_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    registration = root / "freeze.json"
    frozen = entry.prepare(root, root, registration)
    return entry, registration, frozen


def authorize(root, entry, frozen):
    anchor = repair.HUMAN_BUDGET_APPROVAL or {"human_quote": "FORGED", "human_message_reference": "FORGED"}
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=repair.BUDGET_CHOICE, user_limit_usd="1.5",
        cumulative_stop_usd="3", human_quote=anchor["human_quote"], human_message_reference=anchor["human_message_reference"],
        human_budget_approval_sha256=original.digest(anchor))
    entry.write_once(entry.campaign(root) / "authorization.json", auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": original.digest(auth)}
    entry.write_once(entry.campaign(root) / f"{frozen['stage']}.authorization.json", phase)
    return {"allow_local_env": True, "budget_choice": repair.BUDGET_CHOICE,
        "campaign_authorization_sha256": original.digest(auth), "stage_authorization_sha256": original.digest(phase)}


def test_history_cost_chain_and_all_archives_are_read_only(history):
    root, _ = history
    before = tree(root)
    evidence = repair.parent_evidence(root)
    assert evidence["first_campaign"]["known_cost_usd"] == "0.00065169"
    assert evidence["second_campaign"]["calls"] == 554
    assert evidence["second_campaign"]["known_cost_usd"] == "0.067340992"
    assert evidence["second_campaign"]["reused_answers"] == 0
    assert evidence["second_campaign"]["archived_sources"] == 115
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


@pytest.mark.parametrize("part", ["v1", "v1archive", "v2wire", "v2archive", "journal"])
def test_each_parent_evidence_mutation_blocks_without_new_campaign(history, part):
    root, _ = history
    paths = {"v1": old.PARENT / "synthetic-extra0.json", "v1archive": old.ARCHIVE / "aria/source1.py",
        "v2wire": repair.PARENT / "main/wire-requests/0014.json", "v2archive": repair.ARCHIVE / "aria/source1.py",
        "journal": repair.PARENT / "allowance.sqlite3-wal"}
    (root / paths[part]).write_bytes(b"synthetic changed")
    with pytest.raises((repair.RunError, OSError)):
        repair.build_engine(root)
    assert not (root / repair.CAMPAIGN).exists()


def test_sqlite_receipt_still_checked_if_outer_tree_is_reanchored(history, monkeypatch):
    root, _ = history
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("UPDATE settings SET threshold='1.5'")
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    with pytest.raises(repair.RunError, match="ledger_receipt"):
        repair.parent_evidence(root)


def test_pending_proposal_is_not_available_budget_and_forged_approved_file_cannot_read_key(history, monkeypatch):
    root, _ = history
    entry, registration, frozen = prepare(history, monkeypatch)
    assert entry.verify(root, root, registration) == frozen
    budget = frozen["protocol"]["budget"]
    assert budget["status"] == "pending" and budget["effective_new_allowance_usd"] is None
    assert budget["currently_approved_e1b_limit_usd"] == "1"
    assert budget["currently_approved_cumulative_stop_usd"] == "2"
    assert budget["currently_approved_remaining_usd"] == budget["current_remaining_usd"] == "0.539628146"
    assert budget["cumulative_stop_usd"] == "2" and budget["proposed_new_allowance_usd"] == "1.432007318"
    assert entry.authorization_template(root)["status"] == "pending"
    assert "scripts/run_e1b_reflection_repair.py" in entry.FILES
    assert frozen["protocol"]["planned_total_calls"] == 1636
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("pending approval cannot read key"))
    with pytest.raises(repair.RunError, match="human_budget_approval_pending"):
        entry.execute(root, root, registration, **args)
    assert not (entry.campaign(root) / "prescreen.started.json").exists()
    assert not (entry.campaign(root) / "allowance.sqlite3").exists()


def test_exact_anchored_budget_carries_all_previous_costs_and_keeps_old_namespace(history, monkeypatch):
    root, _ = history
    entry, registration, frozen = prepare(history, monkeypatch, approved=True)
    args = authorize(root, entry, frozen)
    _, limit = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert limit == Decimal("1.432007318")
    assert frozen["protocol"]["budget"]["currently_approved_e1b_limit_usd"] == "1.5"
    assert frozen["protocol"]["budget"]["currently_approved_cumulative_stop_usd"] == "3"
    assert frozen["protocol"]["budget"]["currently_approved_remaining_usd"] == str(limit)
    assert frozen["protocol"]["budget"]["prior_e1b_known_usd"] == "0.067992682"
    assert frozen["protocol"]["budget"]["prior_known_floor_usd"] == "1.460371854"
    assert frozen["protocol"]["product_configuration"]["reflection_json_protocol_version"] == "reflection_json_schema_v1"
    assert original.PRIOR_KNOWN == Decimal("1.392379172") and old.PRIOR_KNOWN == Decimal("1.393030862")
    assert entry.verify(root, root, registration) == frozen
    checks = {k: v for k, v in args.items() if k not in {"allow_local_env", "budget_choice"}}
    with pytest.raises(repair.RunError, match="explicit_new_budget_pair"):
        entry.validate_authorization(root, frozen, budget_choice="12_usd1", **checks)
    _, assessed_limit = repair.assessment_engine(entry).validate_authorization(
        root, frozen, budget_choice="12_usd1", **checks)
    assert assessed_limit == limit


@pytest.mark.parametrize("field,value", [
    ("prior_known_usd", "1.393030862"), ("prior_e1b_known_usd", "0"),
    ("human_quote", "altered"), ("human_message_reference", "another"),
    ("user_limit_usd", "1"), ("cumulative_stop_usd", "2"), ("accepted_old_unknown", 0),
    ("workspace_identity", "a" * 64), ("ledger_identity", "a" * 64),
])
def test_altered_cost_quote_workspace_or_limit_denied_before_key(history, monkeypatch, field, value):
    root, _ = history
    entry, registration, frozen = prepare(history, monkeypatch, approved=True)
    args = authorize(root, entry, frozen)
    path = entry.campaign(root) / "authorization.json"
    auth = entry.read_json(path)
    auth[field] = value
    path.write_text(json.dumps(auth), encoding="utf-8")
    args["campaign_authorization_sha256"] = original.digest(auth)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("invalid authorization cannot read key"))
    with pytest.raises(repair.RunError, match="explicitly_authorized"):
        entry.execute(root, root, registration, **args)
    assert not (entry.campaign(root) / "prescreen.started.json").exists()


def test_stage_hash_and_copied_workspace_cannot_borrow_authorization(history, monkeypatch, tmp_path_factory):
    root, _ = history
    entry, _, frozen = prepare(history, monkeypatch, approved=True)
    args = authorize(root, entry, frozen)
    checks = {k: v for k, v in args.items() if k != "allow_local_env"}
    with pytest.raises(repair.RunError, match="workspace_changed"):
        entry.validate_authorization(tmp_path_factory.mktemp("copy"), frozen, **checks)
    checks["stage_authorization_sha256"] = "a" * 64
    with pytest.raises(repair.RunError, match="stage_not_explicitly"):
        entry.validate_authorization(root, frozen, **checks)


def test_new_wire_saved_before_one_fake_failure_and_campaign_cannot_restart(history, monkeypatch):
    from tests.test_e1b_execution import fake_http
    root, _ = history
    entry, registration, frozen = prepare(history, monkeypatch, approved=True)
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    calls, _ = fake_http(monkeypatch, failure="timeout")
    network = urllib.request.urlopen
    def check(request, **kwargs):
        raw = (entry.campaign(root) / "prescreen/wire-requests/0001.json").read_bytes()
        assert raw == request.data and b"synthetic-inert-key" not in raw
        return network(request, **kwargs)
    monkeypatch.setattr(urllib.request, "urlopen", check)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == 1
    value = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")
    assert value["accounting"]["max_observed_cost_usd"] == "1.432007318"
    assert value["cumulative_prior_known_usd"] == "1.460371854" and value["old_unknown_requests_retained"] == 2
    entry.check_wire_evidence(value["accounting"])
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert len(calls) == 1


def captured_twentyfour(root):
    frozen = {"stage": "main", "inputs": {"case_ids": [f"c{i}" for i in range(12)]},
        "protocol": {"case_source_sha256": "b" * 64}}
    rows = []
    for case in frozen["inputs"]["case_ids"]:
        folder = root / "main" / case
        for number in (1, 2):
            session = f"{case}-s{number}"
            row = {"e1b_stage": "main", "role": "reflection", "session_id": session,
                "operation_id": f"e1b-review-{session}", "request_sha256": "c" * 64,
                "visible_content_sha256": "d" * 64, "client_return_content_sha256": "e" * 64}
            rows.append(row)
            first = {"schema": "e1b_reflection_first_v1", "case_id": case, "session": number,
                "session_id": session, "source_sha256": "b" * 64, "request_hash_kind": "actual_wire",
                "request_sha256": "c" * 64, "wire_visible_content_sha256": "d" * 64,
                "visible_response_sha256": "e" * 64, "response_hash_source": "client_return_content",
                "receipt_status": "received", "parse_status": "object", "duplicate_key_count": 0,
                "known_fields_present": dict.fromkeys(("protocol_version", "reviews", "changes"), True),
                "client_diagnostic": {"output_kind": "reflection_json_schema"},
                "public_projection_dropped_fields": 0, "public_projection": {"protocol_version": "2.0", "reviews": [], "changes": []}}
            attempt = {"case_id": case, "session": number, "session_id": session, "source_sha256": "b" * 64, "status": "started"}
            original.write_once(folder / f"reflection-s{number}.attempt.json", attempt)
            first_path = folder / f"reflection-s{number}.first.json"
            original.write_once(first_path, first)
            product = {"schema": "e1b_reflection_product_v1", "session": number, "session_id": session,
                "first_available": True, "updater_raised": False, "capture_error": None, "product_status": "ok",
                "last_error_type": None, "last_error_detail": None, "first_sha256": original.byte_sha(first_path)}
            original.write_once(folder / f"reflection-s{number}.product.json", product)
    return frozen, {"attempts": rows}


def test_reflection_all24_exact_requests_and_public_first_outcomes_required(tmp_path):
    frozen, state = captured_twentyfour(tmp_path)
    result = repair.reflection_originals(tmp_path, frozen, state)
    assert len(result) == 24 and len({(r["case_id"], r["session"]) for r in result}) == 24
    assert all(len(r["first_sha256"]) == 64 and len(r["product_sha256"]) == 64 for r in result)


def test_diagnostic_privacy_projection_drop_is_not_a_new_product_rejection_gate(tmp_path):
    frozen, state = captured_twentyfour(tmp_path)
    first_path = tmp_path / "main/c0/reflection-s1.first.json"
    first = original.read_json(first_path)
    # The real strict client/updater can accept a long proposed claim and rebuild
    # its atomic claim; the diagnostic projection deliberately omits that text.
    first["public_projection_dropped_fields"] = 1
    first["public_projection"]["changes"] = [{"claim": None}]
    first_path.write_text(json.dumps(first), encoding="utf-8")
    product_path = first_path.with_name("reflection-s1.product.json")
    product = original.read_json(product_path)
    product["first_sha256"] = original.byte_sha(first_path)
    product_path.write_text(json.dumps(product), encoding="utf-8")
    result = repair.reflection_originals(tmp_path, frozen, state)
    assert len(result) == 24 and result[0]["public_projection_dropped_fields"] == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wire", "trimmed", "failed", "raw_missing", "changed_first"])
def test_reflection_capture_tamper_or_failed_result_never_qualifies(tmp_path, mutation):
    frozen, state = captured_twentyfour(tmp_path)
    if mutation == "missing":
        state["attempts"].pop()
    elif mutation == "duplicate":
        state["attempts"][-1] = deepcopy(state["attempts"][0])
    elif mutation in {"wire", "trimmed"}:
        state["attempts"][0]["request_sha256" if mutation == "wire" else "client_return_content_sha256"] = "f" * 64
    else:
        name = "product" if mutation == "failed" else "first"
        path = tmp_path / "main/c0" / f"reflection-s1.{name}.json"
        value = original.read_json(path)
        if mutation == "failed":
            value["product_status"] = "failed"
        elif mutation == "raw_missing":
            value.update(public_projection=None, response_hash_source="receipt_only")
        else:
            value["public_projection"]["reviews"] = [{"id": "edited"}]
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(repair.RunError, match="reflection_first_answer"):
        repair.reflection_originals(tmp_path, frozen, state)


def test_assessment_uses_formal_v3_namespace_without_touching_prior_versions(history):
    root, _ = history
    before = tree(root)
    result = repair.assess(root)
    assert result["status"] == "not_evaluable" and result["model_calls"] == 0
    assert result["prior_e1b_known_cost_usd"] == "0.067992682"
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


def test_full_reflection_repair_fake_http_chain(history, monkeypatch):
    """One full product/phase chain; immutable history is covered above.

    Use an already-validated synthetic parent snapshot to avoid hashing 700
    unrelated historical fixture files thousands of times. This does not change
    product requests, ledger reservations, counters, phase seals or first replies.
    """
    import tests.test_e1b_execution as existing
    root, _ = history
    checked_parent = repair.parent_evidence(root)
    monkeypatch.setattr(repair, "parent_evidence", lambda _: deepcopy(checked_parent))
    entry, registration, frozen = prepare(history, monkeypatch, approved=True)
    monkeypatch.setattr(existing, "run", entry)
    def approve_stage(current, value):
        auth_path = entry.campaign(current) / "authorization.json"
        if not auth_path.exists():
            return authorize(current, entry, value)
        auth_sha = original.digest(entry.read_json(auth_path))
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": value["stage"],
            "registration_sha256": value["seal_sha256"], "campaign_authorization_sha256": auth_sha}
        entry.write_once(entry.campaign(current) / f"{value['stage']}.authorization.json", phase)
        return {"allow_local_env": True, "budget_choice": repair.BUDGET_CHOICE,
            "campaign_authorization_sha256": auth_sha, "stage_authorization_sha256": original.digest(phase)}
    monkeypatch.setattr(existing, "approve", approve_stage)
    existing.test_full_18_by_3_by_10_prescreen_real_assembly_fake_http(
        (root, registration, frozen), monkeypatch)
    main = entry.read_receipt(root, "main")
    assert len(main["qualification"]["reflection_originals"]) == 24
    final = entry.read_receipt(root, "judge")
    assert final["accounting"]["calls_used"] == 1636
    assert final["accounting"]["max_observed_cost_usd"] == "1.432007318"
    assert final["cumulative_prior_known_usd"] == "1.460371854"
