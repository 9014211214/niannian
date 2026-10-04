"""T7.2 integration uses fictional HTTP failures and temporary ledgers only."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import check_e1_t72 as check
from scripts import run_e1_t71 as previous
from scripts import t71_policy, t72_policy, t72_transport


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credentials_forbidden")

    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def guarded_text(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return read_text(path, *args, **kwargs)

    def guarded_bytes(path, *args, **kwargs):
        if path.name == ".env":
            forbidden()
        return read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


@pytest.fixture(scope="module")
def prepared(bundle):
    return previous.prepare_e1(bundle)


@pytest.fixture
def failed_campaign(bundle, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    local_bundle = {**bundle, "root": root}
    metadata = "experiments/t6_20260930/public_metadata.json"
    (root / metadata).parent.mkdir(parents=True)
    shutil.copyfile(bundle["root"] / metadata, root / metadata)
    directory = previous.legacy.campaign_directory(root)
    directory.mkdir(parents=True)
    previous.legacy.write_private(directory / "campaign.json", {
        "campaign_id": previous.legacy.CAMPAIGN_ID, "t6_seal": previous.T6_SEAL,
        "t7_seal": previous.T7_SEAL, "execution_seal": check.T71_SEAL})
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id=previous.legacy.CAMPAIGN_ID, freeze_seal=check.T71_SEAL)
    calls = []

    def opener(request, *, timeout):
        assert ledger.summary()["outstanding_count"] == 1
        calls.append(request.get_method())
        raise HTTPError(request.full_url, 404, "provider text must not persist", {},
            io.BytesIO(b'{"error":{"code":404,"message":"arbitrary private message"},"usage":{"cost":0}}'))

    result = previous.run_generation(directory, ledger, local_bundle,
        t72_transport.OpenRouterTransport("synthetic-test-credential", opener=opener),
        source_check=lambda: None, sleep=lambda _: None)
    return local_bundle, directory, ledger, result, calls


def tree_hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def test_http_diagnostic_persists_without_resolving_unknown_fee(failed_campaign):
    _, directory, ledger, result, calls = failed_campaign
    receipt = previous.legacy.read_json(directory / "e1/e1_001.receipt.json")
    assert receipt["envelope"]["http_diagnostic"]["http_status"] == 404
    assert receipt["error"] == "http_status_error"
    assert "usage" not in receipt["envelope"]
    text = (directory / "e1/e1_001.receipt.json").read_text(encoding="utf-8")
    assert "arbitrary private message" not in text
    assert "synthetic-test-credential" not in text
    state = ledger.summary()
    assert state["unknown_cost_count"] == 1 and state["known_cost_usd"] == "0"
    assert state["stopped"] and calls == ["POST"]
    assert len(result["rows"]) == result["denominator"] == 204
    assert state["stages"]["e1"]["counts"] == {
        "failed": 1, "completed": 0, "started": 0, "not_started": 203}
    with pytest.raises(t71_policy.LedgerError):
        ledger.claim("e1", "e1_002")


def test_offline_qualification_retains_old_failure_and_actual_parameters(failed_campaign):
    local_bundle, directory, _, _, _ = failed_campaign
    before = tree_hashes(directory)
    report = check.build_report(local_bundle)
    assert report["missing_parameter_evidence"] == ["parallel_tool_calls"]
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["planned_full_e1_reserve_usd"] == "0.2244"
    assert report["model_requests"] == 0 and report["credentials_read"] is False
    assert report["ready_to_send"] is False and report["execution_entry_enabled"] is False
    assert report["prior_campaign"]["unknown_cost_count"] == 1
    assert report["prior_campaign"]["same_campaign_resume_allowed"] is False
    assert "prior_campaign_permanently_stopped" in report["blocking_reasons"]
    assert "prior_cost_unresolved" in report["blocking_reasons"]
    assert report["judge"]["eligible"] is False
    assert tree_hashes(directory) == before


def test_public_metadata_failure_is_bound_to_actual_wire(failed_campaign):
    local_bundle, directory, _, _, _ = failed_campaign
    metadata = previous.legacy.read_json(local_bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    calls = []

    def opener(request, *, timeout):
        assert request.get_method() == "GET" and request.get_header("Authorization") is None
        calls.append(request.full_url)
        data = ({"data": metadata["models"]} if request.full_url.endswith("/models") else
                {"data": metadata["selected_endpoints"][0]["data"]})
        return t72_policy._StaticResponse(data, request.full_url)

    before = tree_hashes(directory)
    report = check.build_report(local_bundle, online=True, opener=opener)
    assert len(calls) == 2
    assert report["live_metadata_error"] == "parameter_support_unverified_parallel_tool_calls"
    assert report["live_metadata"]["actual_wire_field_removed"] is False
    assert report["ready_to_send"] is False
    assert tree_hashes(directory) == before


@pytest.mark.parametrize("damage", ["marker", "inventory", "request", "receipt", "database", "wal"])
def test_readonly_check_rejects_damaged_or_live_campaign(failed_campaign, prepared, damage):
    local_bundle, directory, _, _, _ = failed_campaign
    paths = {"marker": "campaign.json", "inventory": "e1/inventory.json",
             "request": "e1/e1_001.request.json", "receipt": "e1/e1_001.receipt.json"}
    if damage in paths:
        (directory / paths[damage]).write_text("{}", encoding="utf-8")
    elif damage == "database":
        with sqlite3.connect(directory / "campaign.sqlite3") as conn:
            conn.execute("UPDATE campaign SET freeze_seal=?", ("0" * 64,))
    else:
        (directory / "campaign.sqlite3-wal").write_bytes(b"synthetic marker")
    before = tree_hashes(directory)
    with pytest.raises((ValueError, KeyError, t71_policy.LedgerError)):
        check.inspect_campaign(local_bundle, prepared[3])
    assert tree_hashes(directory) == before


def test_missing_campaign_never_creates_one(bundle, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    local_bundle = {**bundle, "root": root}
    before = tree_hashes(tmp_path)
    with pytest.raises(check.QualificationError):
        check.inspect_campaign(local_bundle, [])
    assert tree_hashes(tmp_path) == before
    assert not previous.legacy.campaign_directory(root).exists()


def test_readonly_ledger_cannot_claim(failed_campaign):
    _, directory, _, _, _ = failed_campaign
    before = tree_hashes(directory)
    reader = check._ExistingLedger(directory / "campaign.sqlite3")
    with pytest.raises(check.QualificationError, match="qualification_is_read_only"):
        reader.claim("e1", "e1_002")
    assert tree_hashes(directory) == before


@pytest.mark.parametrize("args", [["--execute"], ["--allow-local-env"], ["--stage", "judge"]])
def test_cli_does_not_offer_execution_or_key_access(args):
    with pytest.raises(SystemExit) as error:
        check.main(args)
    assert error.value.code == 2


def test_cli_reports_block_without_opening_an_execution_path(failed_campaign, monkeypatch, capsys):
    local_bundle, directory, _, _, _ = failed_campaign
    monkeypatch.setattr(previous, "load_bundle", lambda: local_bundle)
    monkeypatch.setattr(check, "verify_repair", lambda _: {"seal_sha256": "a" * 64})
    before = tree_hashes(directory)
    assert check.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["ready_to_send"] is False
    assert tree_hashes(directory) == before


def test_cli_never_prints_exception_contents(monkeypatch, capsys):
    def fail():
        raise ValueError("synthetic private contents must not appear")

    monkeypatch.setattr(previous, "load_bundle", fail)
    assert check.main([]) == 1
    output = capsys.readouterr().out
    assert "private contents" not in output
    assert json.loads(output)["reason"] == "qualification_check_failed"


def test_repair_and_original_seals(bundle):
    assert check.verify_repair(bundle)["inherits"] == check.INHERITS
    assert previous.verify_execution(bundle)["seal_sha256"] == check.T71_SEAL


@pytest.mark.parametrize("relative", sorted(check.FILES))
def test_repair_seal_rejects_each_changed_source(bundle, tmp_path, monkeypatch, relative):
    for name in check.FILES | {check.EXECUTION}:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle["root"] / name, target)
    with (tmp_path / relative).open("ab") as file:
        file.write(b"\n")
    monkeypatch.setattr(previous, "verify_execution", lambda _: {"seal_sha256": check.T71_SEAL})
    with pytest.raises(check.QualificationError):
        check.verify_repair({**deepcopy(bundle), "root": tmp_path})
