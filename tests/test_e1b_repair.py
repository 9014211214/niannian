"""Repair authorization/accounting tests. All evidence is local and synthetic."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request

import pytest

from _reservation_fixtures import fixture_reservation_policy
from scripts import run_e1b as original
from scripts import run_e1b_repair as repair


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))


def tree(root):
    return [{"path": path.relative_to(root).as_posix(), "sha256": original.byte_sha(path)}
            for path in sorted((root / repair.PARENT).rglob("*")) if path.is_file()]


@pytest.fixture
def prior(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("r")
    monkeypatch.setattr(repair, "CAMPAIGN", Path("v2"))
    monkeypatch.setattr(repair.online_json, "ONLINE_JSON_PROTOCOL_VERSION", "online_json_protocol_v2", raising=False)
    policy = fixture_reservation_policy()
    for spec in policy["models"].values():
        spec.update(input_usd_per_million="0.01", output_usd_per_million="0.02")
    rows = []
    for index in range(114):
        name = "aria/bench/cases/e1b_v1_draft.yaml" if index == 0 else f"aria/source{index}.py"
        raw = b"# synthetic frozen source\n"
        archive = root / repair.ARCHIVE / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(raw)
        if index == 0:
            current = root / name
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_bytes(raw)
        rows.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    registration = {"source": rows, "metadata": {"policy": policy}, "fixture": True}
    registration["seal_sha256"] = original.digest(registration)
    monkeypatch.setattr(repair, "PARENT_SEAL", registration["seal_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/registration.json", registration)
    original.write_once(root / repair.ARCHIVE / "registration.json", registration)
    attempts = [{"request_id": str(index), "status": "received" if index < 9 else "failed",
                 "usage": {"cost": "0.00006000" if index < 9 else "0.00011169"}}
                for index in range(10)]
    encoded_policy = json.dumps(policy, sort_keys=True, allow_nan=False)
    state = {"attempts": attempts, "ledger_id": "synthetic-old-ledger", "max_requests": 1636,
             "max_observed_cost_usd": "0.607620828", "calls_used": 10,
             "reservation_policy_sha256": hashlib.sha256(encoded_policy.encode()).hexdigest(),
             "unknown_cost_requests": 0, "stopped": True, "stop_reason": "model_or_storage_failed",
             "reserved_cost_usd": "0", "known_cost_usd": str(repair.PARENT_COST)}
    receipt = {"accounting": state, "registration_sha256": registration["seal_sha256"],
               "status": "not_evaluable", "stage": "prescreen", "old_unknown_requests_retained": 2,
               "cumulative_prior_known_usd": "1.392379172"}
    receipt["receipt_sha256"] = original.digest(receipt)
    monkeypatch.setattr(repair, "PARENT_RECEIPT", receipt["receipt_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/execution-receipt.json", receipt)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts (sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?,?,?)", [(i, row["request_id"], json.dumps(row))
                                                              for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings (id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES (1,1,1636,'0.607620828','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy (id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES (1,?)", (encoded_policy,))
        db.execute("CREATE TABLE reservation_identity (id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES (1,'synthetic-old-ledger')")
    for index in range(11):
        original.write_once(root / repair.PARENT / f"synthetic-extra{index}.json", {"local_fixture": index})
    assert len(tree(root)) == 14
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    return root, policy


@pytest.fixture
def prepared(prior, monkeypatch):
    root, policy = prior
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    registration = root / "repair.json"
    frozen = entry.prepare(root, root, registration)
    return root, entry, registration, frozen


def authorize(root, entry, frozen):
    value = entry.authorization_template(root)
    value.update(status="approved", budget_choice="12_usd1", user_limit_usd="1",
                 human_authorization_reference="synthetic test only")
    entry.write_once(entry.campaign(root) / "authorization.json", value)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
             "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": entry.digest(value)}
    entry.write_once(entry.campaign(root) / "prescreen.authorization.json", phase)
    return {"allow_local_env": True, "budget_choice": "12_usd1",
            "campaign_authorization_sha256": entry.digest(value), "stage_authorization_sha256": entry.digest(phase)}


def test_parent_actual_accounting_and_sources_checked_without_writing(prior):
    root, _ = prior
    before = tree(root)
    value = repair.parent_evidence(root)
    assert value["known_cost_usd"] == "0.00065169"
    assert value["calls"] == 10 and value["delivered_turns"] == 9
    assert value["archived_source_files"] == 114 and value["reused_answers"] == 0
    assert value["historical_unknown_retained"] == 2
    assert tree(root) == before
    assert not any((root / repair.PARENT).glob("*-wal"))


@pytest.mark.parametrize("target", ["tree", "archive", "case", "ledger_journal", "archived_registration"])
def test_parent_mutation_blocks_before_key(prior, target):
    root, _ = prior
    if target == "tree":
        path = root / repair.PARENT / "synthetic-extra0.json"
    elif target == "archive":
        path = root / repair.ARCHIVE / "aria/source1.py"
    elif target == "case":
        path = root / "aria/bench/cases/e1b_v1_draft.yaml"
    elif target == "ledger_journal":
        path = root / repair.PARENT / "allowance.sqlite3-wal"
    else:
        path = root / repair.ARCHIVE / "registration.json"
    path.write_bytes(b"synthetic changed evidence")
    with pytest.raises(repair.RunError):
        repair.build_engine(root)
    assert not (root / repair.CAMPAIGN).exists()


def test_ledger_is_checked_against_receipt_even_if_file_hash_is_reanchored(prior, monkeypatch):
    root, _ = prior
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("UPDATE settings SET threshold='1'")
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    with pytest.raises(repair.RunError, match="ledger_receipt"):
        repair.parent_evidence(root)


def test_new_protocol_budget_source_and_original_namespace_are_separate(prepared):
    root, entry, registration, frozen = prepared
    assert entry.verify(root, root, registration) == frozen
    assert original.CAMPAIGN == Path("artifacts/local/e1b-formal-v1")
    assert original.PRIOR_KNOWN == Decimal("1.392379172")
    assert entry.campaign(root) != original.campaign(root)
    protocol = frozen["protocol"]
    assert protocol["campaign"] == "e1b-formal-v2"
    assert protocol["product_configuration"]["online_json_protocol_version"] == "online_json_protocol_v2"
    assert len(protocol["product_configuration"]["online_json_schema_sha256"]) == 64
    assert protocol["budget"]["prior_known_floor_usd"] == "1.393030862"
    assert protocol["budget"]["e1b_remaining_usd"] == "0.99934831"
    assert protocol["budget"]["effective_new_allowance_usd"] == "0.606969138"
    assert protocol["planned_total_calls"] == 1636
    assert "scripts/run_e1b_repair.py" in entry.FILES
    args = authorize(root, entry, frozen)
    _, limit = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert limit == Decimal("0.606969138")


def test_old_json_contract_cannot_be_run_as_repair(prior, monkeypatch):
    root, _ = prior
    entry = repair.build_engine(root)
    monkeypatch.setattr(repair.online_json, "ONLINE_JSON_PROTOCOL_VERSION", "online_json_protocol_v1")
    with pytest.raises(repair.RunError, match="revised_online_json"):
        entry.prepare(root, root, root / "invalid.json")


@pytest.mark.parametrize("prior_cost", ["1.392379172", "0"])
def test_authorization_cannot_reset_parent_cost(prepared, monkeypatch, prior_cost):
    root, entry, registration, frozen = prepared
    args = authorize(root, entry, frozen)
    path = entry.campaign(root) / "authorization.json"
    auth = entry.read_json(path)
    auth["prior_known_usd"] = prior_cost
    path.write_text(json.dumps(auth), encoding="utf-8")
    args["campaign_authorization_sha256"] = entry.digest(auth)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("must reject before reading key"))
    with pytest.raises(repair.RunError, match="prior_cost"):
        entry.execute(root, root, registration, **args)
    assert not (entry.campaign(root) / "prescreen.started.json").exists()


def test_parent_rechecked_before_each_actual_send_and_no_restart(prepared, monkeypatch):
    root, entry, registration, frozen = prepared
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    from test_e1b_execution import fake_http
    calls, _ = fake_http(monkeypatch)
    network = urllib.request.urlopen
    def mutate_after_first(request, **kwargs):
        response = network(request, **kwargs)
        (root / repair.ARCHIVE / "aria/source1.py").write_text("changed after first call", encoding="utf-8")
        return response
    monkeypatch.setattr(urllib.request, "urlopen", mutate_after_first)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == 1
    receipt = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")
    assert receipt["accounting"]["calls_used"] == 1
    assert receipt["accounting"]["known_cost_usd"] == "0.000001"
    assert receipt["accounting"]["unknown_cost_requests"] == 0
    assert receipt["cumulative_prior_known_usd"] == "1.393030862"
    (root / repair.ARCHIVE / "aria/source1.py").write_bytes(b"# synthetic frozen source\n")
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert len(calls) == 1


def test_assess_uses_fixed_repair_engine_without_mutating_original(prior, monkeypatch):
    root, _ = prior
    original_path = original.campaign(root)
    before = deepcopy(repair.parent_evidence(root))
    result = repair.assess(root)
    assert result["status"] == "not_evaluable" and result["model_calls"] == 0
    assert result["prior_e1b_known_cost_usd"] == "0.00065169"
    assert original.campaign(root) == original_path
    assert repair.parent_evidence(root) == before


def test_receipt_cannot_mix_another_workspace_with_bound_wire_files(prepared, tmp_path_factory):
    _, entry, _, _ = prepared
    with pytest.raises(repair.RunError, match="workspace_changed"):
        entry.read_receipt(tmp_path_factory.mktemp("other"), "prescreen")


def test_full_repair_prescreen_main_judge_fake_http(prepared, monkeypatch):
    # Exercise the unchanged 540/1056/40 phase chain using this entry's stricter
    # parent checks and shared remaining budget. Every HTTP reply is synthetic.
    import test_e1b_execution as existing
    root, entry, registration, frozen = prepared
    old_tree = tree(root)
    monkeypatch.setattr(existing, "run", entry)
    existing.test_full_18_by_3_by_10_prescreen_real_assembly_fake_http(
        (root, registration, frozen), monkeypatch)
    final = entry.read_receipt(root, "judge")
    assert final["accounting"]["max_observed_cost_usd"] == "0.606969138"
    assert final["cumulative_prior_known_usd"] == "1.393030862"
    assert final["old_unknown_requests_retained"] == 2
    assert tree(root) == old_tree


def test_wire_body_saved_exactly_before_send_without_headers(prepared, monkeypatch):
    root, entry, registration, frozen = prepared
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-secret-must-not-be-saved")
    from test_e1b_execution import fake_http
    calls, _ = fake_http(monkeypatch, failure="timeout")
    network, observed = urllib.request.urlopen, []
    def inspect_send(request, **kwargs):
        path = entry.campaign(root) / "prescreen/wire-requests/0001.json"
        observed.append(path.read_bytes())
        assert path.read_bytes() == request.data
        assert b"synthetic-secret-must-not-be-saved" not in path.read_bytes()
        assert b"Authorization" not in path.read_bytes()
        return network(request, **kwargs)
    monkeypatch.setattr(urllib.request, "urlopen", inspect_send)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == len(observed) == 1
    state = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")["accounting"]
    assert state["attempts"][0]["wire_request_path"] == "prescreen/wire-requests/0001.json"
    assert hashlib.sha256(observed[0]).hexdigest() == state["attempts"][0]["request_sha256"]
    entry.check_wire_evidence(state)


def test_wire_fsync_failure_keeps_original_but_never_sends_or_restarts(prepared, monkeypatch):
    root, entry, registration, frozen = prepared
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    from test_e1b_execution import fake_http
    calls, _ = fake_http(monkeypatch)
    writer = repair.write_wire_body
    def failing_fsync(path, body):
        # Patch only the writer's actual fsync; subsequent receipt persistence
        # keeps working and records the halted, unissued request honestly.
        original_fsync = repair.os.fsync
        try:
            repair.os.fsync = lambda fd: (_ for _ in ()).throw(OSError("synthetic fsync failure"))
            writer(path, body)
        finally:
            repair.os.fsync = original_fsync
    monkeypatch.setattr(repair, "write_wire_body", failing_fsync)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and calls == []
    base = entry.campaign(root)
    assert (base / "prescreen.started.json").is_file()
    assert (base / "prescreen/wire-requests/0001.json").is_file()
    with sqlite3.connect((base / "allowance.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert calls == []


@pytest.mark.parametrize("mutation", ["body", "path", "missing"])
def test_wire_sequence_is_per_stage_and_tampering_prior_prefix_blocks_qualification(prepared, monkeypatch, mutation):
    root, entry, _, frozen = prepared
    from test_e1b_execution import fake_http
    calls, _ = fake_http(monkeypatch)
    policy = frozen["metadata"]["policy"]
    base = entry.campaign(root)
    for stage, role, kind in (("prescreen", "online_fast", "turn"), ("main", "memory", "review")):
        guard = entry.EvidenceGuard(base / "allowance.sqlite3", max_requests=1636,
            max_observed_cost_usd="0.606969138", reservation_policy=policy,
            stage=stage, registration_sha256=frozen["seal_sha256"])
        client = entry.make_client(role, "synthetic-inert-key")
        guard.attach(client)
        with guard.scope(kind, "op-" + stage, "session-" + stage):
            client.complete([{"role": "user", "content": "本地软件替身材料"}])
        state = guard.snapshot()
        guard.close()
    assert len(calls) == 2
    assert [row["wire_request_path"] for row in state["attempts"]] == [
        "prescreen/wire-requests/0001.json", "main/wire-requests/0001.json"]
    entry.check_wire_evidence(state)
    first = base / "prescreen/wire-requests/0001.json"
    if mutation == "body":
        first.write_bytes(b"{}")
    elif mutation == "missing":
        first.unlink()
    else:
        state["attempts"][0]["wire_request_path"] = "../outside.json"
    with pytest.raises(repair.RunError, match="wire_request"):
        entry.qualify(root, {"stage": "main"}, state)
