"""Formal-v4 cost/identity checks with synthetic parents and fake HTTP only."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request

import pytest

from scripts import run_e1b as original
from scripts import run_e1b_reflection_repair as previous
from scripts import run_e1b_structure_repair as repair
from tests.test_e1b_repair import prior  # noqa: F401
from tests.test_e1b_reflection_repair import history  # noqa: F401


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected real network"))


def tree(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": original.byte_sha(p)}
            for p in sorted((root / repair.PARENT).rglob("*")) if p.is_file()]


@pytest.fixture
def history3(history, monkeypatch):
    root, policy = history
    monkeypatch.setattr(repair, "CAMPAIGN", Path("v4"))
    monkeypatch.setattr(repair, "PARENT", Path("v3"))
    monkeypatch.setattr(repair, "ARCHIVE", Path("archive-v3"))
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", {"human_quote": "Synthetic budget decision only",
        "human_message_reference": "fictional test reference", "budget_choice": previous.BUDGET_CHOICE})
    sources = []
    for index in range(117):
        relative = f"aria/source{index}.py"
        path = root / repair.ARCHIVE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic third-version source\n")
        sources.append({"path": relative, "sha256": original.byte_sha(path)})
    registration = {"source": sources, "metadata": {"policy": policy}, "stage": "prescreen", "fixture": True}
    registration["seal_sha256"] = original.digest(registration)
    monkeypatch.setattr(repair, "PARENT_SEAL", registration["seal_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/registration.json", registration)
    original.write_once(root / repair.ARCHIVE / "registration.json", registration)
    attempts = []
    for index in range(18):
        relative = Path("prescreen/wire-requests") / f"{index + 1:04d}.json"
        path = root / repair.PARENT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
        attempts.append({"request_id": str(index), "status": "received" if index < 17 else "failed",
            "usage": {"cost": "0" if index < 17 else str(repair.PARENT_COST)}, "e1b_stage": "prescreen",
            "role": "online_fast", "registration_sha256": registration["seal_sha256"],
            "wire_request_path": relative.as_posix(), "request_sha256": original.byte_sha(path)})
    encoded_policy = json.dumps(policy, sort_keys=True, allow_nan=False)
    state = {"attempts": attempts, "ledger_id": "synthetic-third-ledger", "max_requests": 1636,
        "max_observed_cost_usd": "1.432007318", "calls_used": 18,
        "reservation_policy_sha256": hashlib.sha256(encoded_policy.encode()).hexdigest(),
        "unknown_cost_requests": 0, "stopped": True, "stop_reason": "model_or_storage_failed",
        "reserved_cost_usd": "0", "known_cost_usd": str(repair.PARENT_COST)}
    value = {"accounting": state, "registration_sha256": registration["seal_sha256"], "stage": "prescreen",
        "status": "not_evaluable", "old_unknown_requests_retained": 2,
        "cumulative_prior_known_usd": "1.460371854"}
    value["receipt_sha256"] = original.digest(value)
    monkeypatch.setattr(repair, "PARENT_RECEIPT", value["receipt_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/execution-receipt.json", value)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts (sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?,?,?)", [(i, row["request_id"], json.dumps(row))
                                                              for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings (id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES (1,1,1636,'1.432007318','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy (id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES (1,?)", (encoded_policy,))
        db.execute("CREATE TABLE reservation_identity (id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES (1,'synthetic-third-ledger')")
    for index in range(16):
        original.write_once(root / repair.PARENT / f"fixture{index}.json", {"fictional": index})
    assert len(tree(root)) == 37
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    return root, policy


def prepared(history3, monkeypatch):
    root, policy = history3
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_structure_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    path = root / "freeze-v4.json"
    return entry, path, entry.prepare(root, root, path)


def authorize(root, entry, frozen):
    anchor = repair.HUMAN_BUDGET_APPROVAL
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=previous.BUDGET_CHOICE, user_limit_usd="1.5",
        cumulative_stop_usd="3", human_quote=anchor["human_quote"], human_message_reference=anchor["human_message_reference"],
        human_budget_approval_sha256=original.digest(anchor))
    path = entry.campaign(root) / "authorization.json"
    if path.exists():
        assert entry.read_json(path) == auth
    else:
        entry.write_once(path, auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": original.digest(auth)}
    entry.write_once(entry.campaign(root) / f"{frozen['stage']}.authorization.json", phase)
    return {"allow_local_env": True, "budget_choice": previous.BUDGET_CHOICE,
        "campaign_authorization_sha256": original.digest(auth), "stage_authorization_sha256": original.digest(phase)}


def test_three_campaign_chain_and_archived_sources_carry_all_costs(history3):
    root, _ = history3
    before = tree(root)
    value = repair.parent_evidence(root)
    assert value["third_campaign"]["known_cost_usd"] == "0.00091593"
    assert value["third_campaign"]["calls"] == 18 and value["third_campaign"]["archived_sources"] == 117
    assert value["earlier_campaigns"]["first_campaign"]["calls"] == 10
    assert value["earlier_campaigns"]["second_campaign"]["calls"] == 554
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


@pytest.mark.parametrize("part", ["wire", "archive", "journal", "v2archive", "v1original"])
def test_alteration_in_any_generation_blocks_new_entry(history3, part):
    from scripts import run_e1b_repair as v2
    root, _ = history3
    paths = {"wire": repair.PARENT / "prescreen/wire-requests/0018.json",
        "archive": repair.ARCHIVE / "aria/source1.py", "journal": repair.PARENT / "allowance.sqlite3-wal",
        "v2archive": previous.ARCHIVE / "aria/source1.py", "v1original": v2.PARENT / "synthetic-extra0.json"}
    (root / paths[part]).write_bytes(b"synthetic alteration")
    with pytest.raises(repair.RunError):
        repair.build_engine(root)
    assert not (root / repair.CAMPAIGN).exists()


def test_reanchored_outer_tree_does_not_override_sqlite_cost_guard(history3, monkeypatch):
    root, _ = history3
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("UPDATE settings SET threshold='1.5'")
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    with pytest.raises(repair.RunError, match="ledger_receipt"):
        repair.parent_evidence(root)


def test_new_version_existing_approval_and_remaining_allowance_are_bound(history3, monkeypatch):
    root, _ = history3
    entry, registration, frozen = prepared(history3, monkeypatch)
    assert entry.verify(root, root, registration) == frozen
    config = frozen["protocol"]["product_configuration"]
    assert config["online_json_protocol_version"] == "online_json_protocol_v2"
    assert config["online_json_structure_version"] == "online_json_structure_v1"
    assert config["online_return_format"] == "json_object" and config["online_prompt_format_style"] == "schema_guided"
    assert frozen["protocol"]["planned_total_calls"] == 1636
    args = authorize(root, entry, frozen)
    _, limit = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert limit == Decimal("1.431091388")
    assert frozen["protocol"]["budget"]["prior_e1b_known_usd"] == "0.068908612"
    assert frozen["protocol"]["budget"]["prior_known_floor_usd"] == "1.461287784"
    assert previous.PRIOR_E1B == Decimal("0.067992682") and previous.PRIOR_KNOWN == Decimal("1.460371854")


@pytest.mark.parametrize("field,value", [("prior_known_usd", "1.460371854"),
    ("prior_e1b_known_usd", "0.067992682"), ("campaign", "v3"), ("human_quote", "unmatched"),
    ("accepted_old_unknown", 0), ("user_limit_usd", "2")])
def test_old_budget_baseline_campaign_or_replaced_quote_cannot_reauthorize(history3, monkeypatch, field, value):
    root, _ = history3
    entry, registration, frozen = prepared(history3, monkeypatch)
    args = authorize(root, entry, frozen)
    path = entry.campaign(root) / "authorization.json"
    auth = entry.read_json(path)
    auth[field] = value
    path.write_text(json.dumps(auth), encoding="utf-8")
    args["campaign_authorization_sha256"] = original.digest(auth)
    monkeypatch.setattr(entry, "local_key", lambda _: pytest.fail("invalid identity cannot read key"))
    with pytest.raises(repair.RunError, match="explicitly_authorized"):
        entry.execute(root, root, registration, **args)
    assert not (entry.campaign(root) / "prescreen.started.json").exists()


def test_wrong_online_client_format_is_rejected_before_registration(history3, monkeypatch):
    root, _ = history3
    entry = repair.build_engine(root)
    make = entry.make_client
    def changed(role, key):
        client = make(role, key)
        client.online_return_format = "native"
        return client
    monkeypatch.setattr(entry, "make_client", changed)
    with pytest.raises(repair.RunError, match="actual_json_object"):
        entry.prepare(root, root, root / "bad.json")
    assert not (root / "bad.json").exists()


def test_parent_is_rechecked_at_each_send_and_no_restart(history3, monkeypatch):
    from tests.test_e1b_execution import fake_http
    root, _ = history3
    entry, registration, frozen = prepared(history3, monkeypatch)
    args = authorize(root, entry, frozen)
    monkeypatch.setattr(entry, "local_key", lambda _: "synthetic-inert-key")
    calls, _ = fake_http(monkeypatch)
    network = urllib.request.urlopen
    changed = root / repair.ARCHIVE / "aria/source1.py"
    old = changed.read_bytes()
    def mutate_after_first(request, **kwargs):
        assert (entry.campaign(root) / "prescreen/wire-requests/0001.json").read_bytes() == request.data
        reply = network(request, **kwargs)
        changed.write_bytes(b"synthetic mutation")
        return reply
    monkeypatch.setattr(urllib.request, "urlopen", mutate_after_first)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == 1
    changed.write_bytes(old)
    value = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")
    assert value["accounting"]["known_cost_usd"] == "0.000001"
    assert value["accounting"]["max_observed_cost_usd"] == "1.431091388"
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert len(calls) == 1


def test_assess_new_namespace_is_read_only(history3):
    root, _ = history3
    before = tree(root)
    result = repair.assess(root)
    assert result["status"] == "not_evaluable" and result["model_calls"] == 0
    assert result["prior_e1b_known_cost_usd"] == "0.068908612"
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


def test_full_structure_repair_fake_http_chain(history3, monkeypatch):
    import tests.test_e1b_execution as existing
    root, _ = history3
    checked = repair.parent_evidence(root)
    # Synthetic historical originals are verified separately by the short tests;
    # this long test covers product/phase/wire/first-answer behavior exactly once.
    monkeypatch.setattr(repair, "parent_evidence", lambda _: deepcopy(checked))
    entry, registration, frozen = prepared(history3, monkeypatch)
    monkeypatch.setattr(existing, "run", entry)
    monkeypatch.setattr(existing, "approve", lambda current, value: authorize(current, entry, value))
    existing.test_full_18_by_3_by_10_prescreen_real_assembly_fake_http((root, registration, frozen), monkeypatch)
    main = entry.read_receipt(root, "main")
    assert len(main["qualification"]["reflection_originals"]) == 24
    final = entry.read_receipt(root, "judge")
    assert final["accounting"]["calls_used"] == 1636
    assert final["accounting"]["max_observed_cost_usd"] == "1.431091388"
    assert final["cumulative_prior_known_usd"] == "1.461287784"
