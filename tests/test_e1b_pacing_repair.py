"""Formal-v5 cost/identity checks with synthetic parents and fake HTTP only."""
from tests._model_http_stub import stub_model_http
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
from scripts import run_e1b_pacing_repair as repair
from tests.test_e1b_repair import prior  # noqa: F401
from tests.test_e1b_reflection_repair import history  # noqa: F401
from tests.test_e1b_structure_repair import history3  # noqa: F401


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected real network"))


@pytest.fixture(autouse=True)
def simulated_guard_clock(monkeypatch):
    from aria.engine.policy.minimal_calls import MinimalCallGuard
    initialize = MinimalCallGuard.__init__
    clock = [0.0]
    def fake_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda delay: clock.__setitem__(0, clock[0] + delay)
    monkeypatch.setattr(MinimalCallGuard, "__init__", fake_initialize)
    return clock


def tree(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": original.byte_sha(p)}
            for p in sorted((root / repair.PARENT).rglob("*")) if p.is_file()]


@pytest.fixture
def history4(history3, monkeypatch):
    root, policy = history3
    monkeypatch.setattr(repair, "CAMPAIGN", Path("v5"))
    monkeypatch.setattr(repair, "PARENT", Path("v4"))
    monkeypatch.setattr(repair, "ARCHIVE", Path("archive-v4"))
    monkeypatch.setattr(repair, "HUMAN_BUDGET_APPROVAL", {"human_quote": "Synthetic budget decision only",
        "human_message_reference": "fictional test reference", "budget_choice": previous.BUDGET_CHOICE})
    monkeypatch.setattr(previous, "HUMAN_BUDGET_APPROVAL", dict(repair.HUMAN_BUDGET_APPROVAL))
    sources = []
    for index in range(119):
        relative = f"aria/source{index}.py"
        path = root / repair.ARCHIVE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic fourth-version source\n")
        sources.append({"path": relative, "sha256": original.byte_sha(path)})
    registration = {"source": sources, "metadata": {"policy": policy}, "stage": "prescreen", "fixture": True}
    registration["seal_sha256"] = original.digest(registration)
    monkeypatch.setattr(repair, "PARENT_SEAL", registration["seal_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/registration.json", registration)
    original.write_once(root / repair.ARCHIVE / "registration.json", registration)
    attempts = []
    for index in range(5):
        relative = Path("prescreen/wire-requests") / f"{index + 1:04d}.json"
        path = root / repair.PARENT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
        attempts.append({"request_id": str(index) if index < 4 else repair.PARENT_UNKNOWN_REQUEST, "status": "received" if index < 4 else "failed",
            "usage": {"cost": str(repair.PARENT_COST) if index == 0 else "0"} if index < 4 else {}, "e1b_stage": "prescreen",
            "role": "online_fast", "registration_sha256": registration["seal_sha256"],
            "wire_request_path": relative.as_posix(), "request_sha256": original.byte_sha(path), "http_status": 200 if index < 4 else 429,
            "reservation": {"reserved_cost_usd": str(repair.RETAINED_UNKNOWN_HOLD)}})
    monkeypatch.setattr(repair, "PARENT_UNKNOWN_WIRE", attempts[-1]["request_sha256"])
    anchor = dict(repair.HUMAN_UNKNOWN_APPROVAL)
    anchor.update(campaign=repair.PARENT.as_posix(), request_sha256=repair.PARENT_UNKNOWN_WIRE,
        human_quote="Synthetic unknown-charge decision only", human_message_reference="fictional new exception")
    monkeypatch.setattr(repair, "HUMAN_UNKNOWN_APPROVAL", anchor)
    encoded_policy = json.dumps(policy, sort_keys=True, allow_nan=False)
    state = {"attempts": attempts, "ledger_id": "synthetic-fourth-ledger", "max_requests": 1636,
        "max_observed_cost_usd": "1.431091388", "calls_used": 5,
        "reservation_policy_sha256": hashlib.sha256(encoded_policy.encode()).hexdigest(),
        "unknown_cost_requests": 1, "stopped": True, "stop_reason": "model_or_storage_failed",
        "reserved_cost_usd": str(repair.RETAINED_UNKNOWN_HOLD), "known_cost_usd": str(repair.PARENT_COST)}
    value = {"accounting": state, "registration_sha256": registration["seal_sha256"], "stage": "prescreen",
        "status": "not_evaluable", "old_unknown_requests_retained": 2,
        "cumulative_prior_known_usd": "1.461287784"}
    value["receipt_sha256"] = original.digest(value)
    monkeypatch.setattr(repair, "PARENT_RECEIPT", value["receipt_sha256"])
    original.write_once(root / repair.PARENT / "prescreen/execution-receipt.json", value)
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("CREATE TABLE attempts (sequence INTEGER,request_id TEXT,record TEXT)")
        db.executemany("INSERT INTO attempts VALUES (?,?,?)", [(i, row["request_id"], json.dumps(row))
                                                              for i, row in enumerate(attempts)])
        db.execute("CREATE TABLE settings (id INTEGER,version INTEGER,max_requests INTEGER,threshold TEXT,halted TEXT)")
        db.execute("INSERT INTO settings VALUES (1,1,1636,'1.431091388','model_or_storage_failed')")
        db.execute("CREATE TABLE reservation_policy (id INTEGER,policy TEXT)")
        db.execute("INSERT INTO reservation_policy VALUES (1,?)", (encoded_policy,))
        db.execute("CREATE TABLE reservation_identity (id INTEGER,ledger_id TEXT)")
        db.execute("INSERT INTO reservation_identity VALUES (1,'synthetic-fourth-ledger')")
    for index in range(11):
        original.write_once(root / repair.PARENT / f"fixture{index}.json", {"fictional": index})
    assert len(tree(root)) == 19
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    return root, policy


def prepared(history4, monkeypatch):
    root, policy = history4
    entry = repair.build_engine(root)
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "scripts/run_e1b_pacing_repair.py", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    path = root / "freeze-v5.json"
    return entry, path, entry.prepare(root, root, path)


def authorize(root, entry, frozen):
    anchor = repair.HUMAN_BUDGET_APPROVAL
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=previous.BUDGET_CHOICE, user_limit_usd="1.5",
        cumulative_stop_usd="3", human_quote=anchor["human_quote"], human_message_reference=anchor["human_message_reference"],
        human_budget_approval_sha256=original.digest(anchor),
        unknown_human_quote=repair.HUMAN_UNKNOWN_APPROVAL["human_quote"],
        unknown_human_message_reference=repair.HUMAN_UNKNOWN_APPROVAL["human_message_reference"],
        human_unknown_approval_sha256=original.digest(repair.HUMAN_UNKNOWN_APPROVAL))
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


def test_four_campaign_chain_and_archived_sources_carry_all_costs(history4):
    root, _ = history4
    before = tree(root)
    value = repair.parent_evidence(root)
    assert value["fourth_campaign"]["known_cost_usd"] == "0.00018047"
    assert value["fourth_campaign"]["calls"] == 5 and value["fourth_campaign"]["archived_sources"] == 119
    assert value["earlier_campaigns"]["earlier_campaigns"]["first_campaign"]["calls"] == 10
    assert value["earlier_campaigns"]["earlier_campaigns"]["second_campaign"]["calls"] == 554
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


@pytest.mark.parametrize("part", ["wire", "archive", "journal", "v2archive", "v1original"])
def test_alteration_in_any_generation_blocks_new_entry(history4, part):
    from scripts import run_e1b_repair as v2
    root, _ = history4
    paths = {"wire": repair.PARENT / "prescreen/wire-requests/0005.json",
        "archive": repair.ARCHIVE / "aria/source1.py", "journal": repair.PARENT / "allowance.sqlite3-wal",
        "v2archive": previous.ARCHIVE / "aria/source1.py", "v1original": v2.PARENT / "synthetic-extra0.json"}
    (root / paths[part]).write_bytes(b"synthetic alteration")
    with pytest.raises(repair.RunError):
        repair.build_engine(root)
    assert not (root / repair.CAMPAIGN).exists()


def test_reanchored_outer_tree_does_not_override_sqlite_cost_guard(history4, monkeypatch):
    root, _ = history4
    with sqlite3.connect(root / repair.PARENT / "allowance.sqlite3") as db:
        db.execute("UPDATE settings SET threshold='1.5'")
    monkeypatch.setattr(repair, "PARENT_TREE", original.digest(tree(root)))
    with pytest.raises(repair.RunError, match="ledger_receipt"):
        repair.parent_evidence(root)


def test_new_version_existing_approval_and_remaining_allowance_are_bound(history4, monkeypatch):
    root, _ = history4
    entry, registration, frozen = prepared(history4, monkeypatch)
    assert entry.verify(root, root, registration) == frozen
    config = frozen["protocol"]["product_configuration"]
    assert config["online_json_protocol_version"] == "online_json_protocol_v2"
    assert config["online_json_structure_version"] == "online_json_structure_v1"
    assert config["online_return_format"] == "json_object" and config["online_prompt_format_style"] == "schema_guided"
    assert frozen["protocol"]["planned_total_calls"] == 1636
    args = authorize(root, entry, frozen)
    _, limit = entry.validate_authorization(root, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    assert limit == Decimal("1.430191268")
    assert frozen["protocol"]["budget"]["prior_e1b_known_usd"] == "0.069089082"
    assert frozen["protocol"]["budget"]["prior_known_floor_usd"] == "1.461468254"
    assert frozen["protocol"]["budget"]["accepted_old_unknown"] == 3
    assert frozen["protocol"]["budget"]["historical_unknown_reserved_usd"] == "0.00071965"
    assert frozen["protocol"]["transport"]["min_interval_seconds"] == 10
    assert previous.PRIOR_E1B == Decimal("0.067992682") and previous.PRIOR_KNOWN == Decimal("1.460371854")


@pytest.mark.parametrize("field,value", [("prior_known_usd", "1.460371854"),
    ("prior_e1b_known_usd", "0.067992682"), ("campaign", "v3"), ("human_quote", "unmatched"),
    ("accepted_old_unknown", 2), ("user_limit_usd", "2"),
    ("unknown_human_quote", "old acceptance is not this exception"),
    ("historical_unknown_reserved_usd", "0"), ("effective_new_allowance_usd", "1.430910918"),
    ("human_unknown_approval_sha256", "b" * 64)])
def test_old_budget_baseline_campaign_or_replaced_quote_cannot_reauthorize(history4, monkeypatch, field, value):
    root, _ = history4
    entry, registration, frozen = prepared(history4, monkeypatch)
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


def test_wrong_online_client_format_is_rejected_before_registration(history4, monkeypatch):
    root, _ = history4
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


def test_parent_is_rechecked_at_each_send_and_no_restart(history4, monkeypatch):
    from tests.test_e1b_execution import fake_http
    root, _ = history4
    entry, registration, frozen = prepared(history4, monkeypatch)
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
    stub_model_http(monkeypatch, mutate_after_first)
    result = entry.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == 1
    changed.write_bytes(old)
    value = entry.read_json(entry.campaign(root) / "prescreen/execution-receipt.json")
    assert value["accounting"]["known_cost_usd"] == "0.000001"
    assert value["accounting"]["max_observed_cost_usd"] == "1.430191268"
    with pytest.raises(repair.RunError, match="already_started"):
        entry.execute(root, root, registration, **args)
    assert len(calls) == 1


def test_assess_new_namespace_is_read_only(history4):
    root, _ = history4
    before = tree(root)
    result = repair.assess(root)
    assert result["status"] == "not_evaluable" and result["model_calls"] == 0
    assert result["prior_e1b_known_cost_usd"] == "0.069089082"
    assert tree(root) == before and not (root / repair.CAMPAIGN).exists()


def test_full_pacing_repair_fake_http_chain(history4, monkeypatch):
    import tests.test_e1b_execution as existing
    root, _ = history4
    checked = repair.parent_evidence(root)
    # Synthetic historical originals are verified separately by the short tests;
    # this long test covers product/phase/wire/first-answer behavior exactly once.
    monkeypatch.setattr(repair, "parent_evidence", lambda _: deepcopy(checked))
    entry, registration, frozen = prepared(history4, monkeypatch)
    monkeypatch.setattr(existing, "run", entry)
    monkeypatch.setattr(existing, "approve", lambda current, value: authorize(current, entry, value))
    existing.test_full_18_by_3_by_10_prescreen_real_assembly_fake_http((root, registration, frozen), monkeypatch)
    main = entry.read_receipt(root, "main")
    assert len(main["qualification"]["reflection_originals"]) == 24
    final = entry.read_receipt(root, "judge")
    assert final["accounting"]["calls_used"] == 1636
    assert final["accounting"]["max_observed_cost_usd"] == "1.430191268"
    assert final["cumulative_prior_known_usd"] == "1.461468254"
    assert final["old_unknown_requests_retained"] == 3
    assert final["historical_unknown_reserved_usd"] == "0.00071965"
    rows = final["accounting"]["attempts"]
    assert all(row["pacing"]["min_interval_seconds"] == 10 for row in rows)
    assert all(row["pacing"]["start_gap_ms"] is None or row["pacing"]["start_gap_ms"] >= 10000 for row in rows)
    assert all(row["pacing"]["scope"] == "one_guard_all_roles" for row in rows)


@pytest.mark.parametrize("damage", [None, "table", "protocol", "row"])
def test_readonly_assessment_binds_actual_pacing(tmp_path, monkeypatch, damage):
    from scripts import e1b_assessment as assessment
    from tests.test_e1b_assessment import ledger_fixture
    monkeypatch.setattr(assessment.execution, "MIN_INTERVAL_SECONDS", 10)
    path, state, frozen = ledger_fixture(tmp_path)
    frozen["protocol"] = {"transport": {"min_interval_seconds": 9 if damage == "protocol" else 10}}
    for index, row in enumerate(state["attempts"]):
        row["pacing"] = {"min_interval_seconds": 9 if damage == "row" and index == 0 else 10,
            "scope": "one_guard_all_roles", "clock": "monotonic"}
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE request_pacing(id INTEGER PRIMARY KEY,min_interval_seconds REAL)")
        db.execute("INSERT INTO request_pacing VALUES(1,?)", (9 if damage == "table" else 10,))
        for index, row in enumerate(state["attempts"]):
            db.execute("UPDATE attempts SET record=? WHERE sequence=?", (json.dumps(row), index))
    before = path.read_bytes()
    if damage:
        with pytest.raises(assessment.AssessmentError, match="pacing_contract"):
            assessment._ledger(tmp_path, state, frozen)
    else:
        assert assessment._ledger(tmp_path, state, frozen) == original.byte_sha(path)
    assert path.read_bytes() == before
