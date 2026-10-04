"""Compatible CLI uses only temporary histories, synthetic keys and fake HTTP."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import e1_compat as compat, run_e1_compat as runner
from scripts import run_e1_t71 as previous, t71_policy, t73_policy, t74_transport
from tests.test_e1_compat import metadata, opener_for


REAL_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_KEY = "synthetic-compat-fixture-credential"


@pytest.fixture(autouse=True)
def no_real_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("real_network_or_credentials_forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)
    monkeypatch.setattr(runner.old_run, "credential", forbidden)
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
def source_bundle():
    return previous.load_bundle()


@pytest.fixture(scope="module")
def inventories(source_bundle):
    return (previous.prepare_e1(source_bundle), t73_policy.prepare_e1(source_bundle))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
                    encoding="utf-8")


def tree_hashes(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def artifacts(directory):
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(directory.rglob("*")) if path.is_file()]


def failed_history(directory, identifier, seal, prepared, allowed):
    _, _, trials, items = prepared
    directory.mkdir(parents=True)
    write_json(directory / "campaign.json", {"campaign_id": identifier, "synthetic_fixture": True})
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3", campaign_id=identifier,
                                     freeze_seal=seal)
    ledger.register_stage("e1", [{k: item[k] for k in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    write_json(directory / "e1/inventory.json", items)
    first = items[0]
    write_json(directory / "e1/e1_001.request.json", trials[first["case_id"], first["arm"]]["wire"])
    receipt = {"envelope": {"http_diagnostic": {"http_status": 404}}, "error": "http_status_error"}
    digest = previous.legacy.write_private(directory / "e1/e1_001.receipt.json", receipt)
    ledger.claim("e1", "e1_001")
    with pytest.raises(t71_policy.LedgerError):
        previous.legacy.finish_receipt(ledger, "e1", "e1_001", receipt["envelope"], digest,
                                      receipt["error"])
    write_json(directory / "e1/results.private.json", {"denominator": 204,
        "rows": [{"id": item["id"], "status": "generation_failed" if i == 0 else "not_started"}
                 for i, item in enumerate(items)]})
    for relative in allowed - set(tree_hashes(directory)):
        write_json(directory / relative, {"synthetic_fixture": True})
    assert set(tree_hashes(directory)) == allowed


@pytest.fixture
def histories(tmp_path, source_bundle, inventories, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    bundle = {**source_bundle, "root": root}
    public = "experiments/t6_20260930/public_metadata.json"
    (root / public).parent.mkdir(parents=True)
    shutil.copyfile(REAL_ROOT / public, root / public)
    # The exact original seven-file exception has its own T7.3 tests. This
    # fixture isolates the additional two-history verifier without reading any
    # real private campaign or changing the original source registration.
    original = runner.old_run.REGISTRATION
    (root / original).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REAL_ROOT / original, root / original)
    monkeypatch.setattr(runner.old_run, "verify_history", lambda *args: None)
    directories = []
    for identifier, seal, planned, allowed in zip(runner.HISTORY_IDS, runner.HISTORY_SEALS,
                                                 inventories, runner.HISTORY_FILES):
        directory = root.parent / "aria-private-experiments" / identifier
        failed_history(directory, identifier, seal, planned, allowed)
        directories.append(directory)
    registration = {"schema": "aria_e1_auto_tool_amendment_v1",
        "campaign_id": runner.CAMPAIGN_ID, "wire_changes": deepcopy(compat.WIRE_CHANGES),
        "budget_usd": deepcopy(bundle["amendment"]["budget_usd"]),
        "unknown_cost_policy": "retain_and_block_unaccepted",
        "accepted_unknown_attempts": [list(a) for a in runner.ACCEPTED_UNKNOWN_ATTEMPTS],
        "fixed_generation_positions": 204, "parent_execution_seal": runner.HISTORY_SEALS[1],
        "prior_campaigns": [{"campaign_id": identifier, "artifacts": artifacts(directory)}
                            for identifier, directory in zip(runner.HISTORY_IDS, directories)]}
    write_json(root / runner.REGISTRATION, registration)
    return bundle, directories, registration


def refresh_history(bundle, directories, registration):
    for row, directory in zip(registration["prior_campaigns"], directories):
        row["artifacts"] = artifacts(directory)
    write_json(bundle["root"] / runner.REGISTRATION, registration)


def patch_cli(monkeypatch, bundle, *, documents=None):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(previous, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    calls, opener = opener_for(documents if documents is not None else metadata(bundle))
    original_preflight = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda bundle, online=False, **kwargs:
                        original_preflight(bundle, online=online, opener=opener))
    return calls


def test_two_stopped_histories_are_read_only_and_original_exception_is_not_expanded(histories):
    bundle, directories, _ = histories
    before = [tree_hashes(directory) for directory in directories]
    report = runner.verify_history(bundle)
    assert report["unknown_cost_count"] == 2 and report["total_actual_cost_usd"] is None
    assert report["unaccepted_unknown_attempts"] == [[runner.HISTORY_IDS[1], "e1_001"]]
    assert runner.ACCEPTED_UNKNOWN_ATTEMPTS == ((runner.HISTORY_IDS[0], "e1_001"),)
    for directory, identifier, seal in zip(directories, runner.HISTORY_IDS, runner.HISTORY_SEALS):
        read_only = runner._ReadOnlyLedger(directory / "campaign.sqlite3", identifier, seal)
        with pytest.raises(runner.RunError, match="history_is_read_only"):
            read_only.stop("e1", "manual_stop")
        with sqlite3.connect((directory / "campaign.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
            assert conn.execute("SELECT cost_usd FROM items WHERE id='e1_001'").fetchone() == (None,)
            assert conn.execute("SELECT stopped FROM campaign").fetchone() == (1,)
    assert [tree_hashes(directory) for directory in directories] == before


@pytest.mark.parametrize("index,relative", [(0, "campaign.json"), (1, "e1/e1_001.receipt.json")])
def test_bound_history_tampering_is_rejected(histories, index, relative):
    bundle, directories, _ = histories
    with (directories[index] / relative).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(runner.RunError, match="historical_artifact_changed"):
        runner.verify_history(bundle)


@pytest.mark.parametrize("status", ["failed", "started"])
def test_more_unknown_or_outstanding_is_rejected_even_after_updating_hashes(histories, status):
    bundle, directories, registration = histories
    with sqlite3.connect(directories[1] / "campaign.sqlite3") as conn:
        conn.execute("UPDATE items SET status=?,claim_token=?,error=? WHERE id='e1_002'",
                     (status, "c" * 64, "external_error" if status == "failed" else None))
    refresh_history(bundle, directories, registration)
    with pytest.raises((runner.RunError, t71_policy.LedgerError)):
        runner.verify_history(bundle)


def test_changed_fixed_inventory_is_rejected_after_updating_file_hash(histories):
    bundle, directories, registration = histories
    path = directories[1] / "e1/inventory.json"
    value = previous.legacy.read_json(path)
    value[1]["arm"] = "B0"
    value[1]["request_sha256"] = "d" * 64
    write_json(path, value)
    refresh_history(bundle, directories, registration)
    with pytest.raises(runner.RunError, match="historical_campaign_state_changed"):
        runner.verify_history(bundle)


def test_linked_history_directory_is_rejected_before_read(histories, monkeypatch):
    bundle, directories, _ = histories
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path:
                        path == directories[1] or original(path))
    with pytest.raises(runner.RunError, match="linked_path_rejected"):
        runner.verify_history(bundle)


def test_additional_file_is_not_silently_ignored(histories):
    bundle, directories, registration = histories
    write_json(directories[1] / "unexpected.json", {"synthetic": True})
    refresh_history(bundle, directories, registration)
    with pytest.raises(runner.RunError, match="historical_artifact_inventory_changed"):
        runner.verify_history(bundle)


def test_live_journal_is_rejected_without_changing_ledger(histories):
    bundle, directories, _ = histories
    database = directories[1] / "campaign.sqlite3"
    database.with_name(database.name + "-wal").write_bytes(b"synthetic journal")
    before = tree_hashes(directories[1])
    with pytest.raises(runner.RunError):
        runner.verify_history(bundle)
    assert tree_hashes(directories[1]) == before


def test_registration_cannot_expand_the_single_human_fee_exception(histories):
    bundle, _, registration = histories
    registration["accepted_unknown_attempts"].append([runner.HISTORY_IDS[1], "e1_001"])
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    with pytest.raises(runner.RunError, match="compatibility_registration_changed"):
        runner.verify_history(bundle)


def test_offline_main_never_reads_key_creates_campaign_or_claims_live_compatibility(histories, monkeypatch, capsys):
    bundle, directories, _ = histories
    before = [tree_hashes(directory) for directory in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert calls == [] and not report["ready_to_send"]
    assert not report["credentials_read"] and report["model_requests"] == 0
    assert "live_metadata_not_checked" in report["blocking_reasons"]
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(directory) for directory in directories] == before


@pytest.mark.parametrize("arguments", [["--online-preflight"], ["--execute", "--allow-local-env"]])
def test_actual_cli_checks_new_wire_but_unaccepted_unknown_blocks_before_key_and_post(histories, monkeypatch, capsys, arguments):
    bundle, directories, _ = histories
    before = [tree_hashes(directory) for directory in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(arguments) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["parameter_ready_to_send"] is True
    assert report["parameter_compatibility_verified"] is True
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["history"]["unaccepted_unknown_attempts"] == [[runner.HISTORY_IDS[1], "e1_001"]]
    assert report["blocking_reasons"] == ["new_historical_unknown_cost_requires_resolution"]
    assert not report["credentials_read"] and report["model_requests"] == 0
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(directory) for directory in directories] == before


def test_execute_without_local_env_flag_stops_before_public_checks(histories, monkeypatch, capsys):
    bundle, _, _ = histories
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("argument", ["--ignore-unknown-cost", "--retry"])
def test_no_retry_or_cost_override_cli_flag(argument):
    with pytest.raises(SystemExit) as caught:
        runner.main([argument])
    assert caught.value.code == 2


def test_mode_failure_and_new_unknown_both_block_before_credentials(histories, monkeypatch, capsys):
    bundle, _, _ = histories
    documents = metadata(bundle)
    documents[compat.choices.ENDPOINT]["data"]["endpoints"][0]["supports_tool_choice"]["auto"] = False
    calls = patch_cli(monkeypatch, bundle, documents=documents)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and not report["ready_to_send"]
    assert set(report["blocking_reasons"]) == {"tool_choice_mode_not_supported",
        "new_historical_unknown_cost_requires_resolution"}
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_synthetic_complete_authorization_reaches_t74_sender_and_claim_then_stops_without_retry(histories, monkeypatch, capsys):
    bundle, directories, registration = histories
    before = [tree_hashes(directory) for directory in directories]
    # Fictional double approval is local test input; production keeps only the
    # real original single approval, verified separately above.
    synthetic_approvals = tuple((identifier, "e1_001") for identifier in runner.HISTORY_IDS)
    monkeypatch.setattr(runner, "ACCEPTED_UNKNOWN_ATTEMPTS", synthetic_approvals)
    registration["accepted_unknown_attempts"] = [list(a) for a in synthetic_approvals]
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    calls = patch_cli(monkeypatch, bundle)
    key_reads, posts = [], []

    def credential(root):
        assert root == bundle["root"]
        key_reads.append(True)
        return SYNTHETIC_KEY

    def failed_http(request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t74_transport.ENDPOINT
        directory = runner.campaign_directory(bundle["root"])
        ledger = runner._ReadOnlyLedger(directory / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64)
        assert ledger.summary()["outstanding_count"] == 1
        assert (directory / "e1/e1_001.request.json").is_file()
        posts.append(json.loads(request.data))
        body = {"error": {"code": 404,
                "message": "No endpoints found that match your data policy. private sentinel"}}
        raise HTTPError(request.full_url, 404, "private HTTP reason", {},
                        io.BytesIO(json.dumps(body).encode()))

    def transport(key, *, enable_network):
        assert key == SYNTHETIC_KEY and enable_network is True
        return t74_transport.OpenRouterTransport(key, opener=failed_http)

    monkeypatch.setattr(runner.old_run, "credential", credential)
    monkeypatch.setattr(runner, "OpenRouterTransport", transport)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    outcome = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(posts) == len(key_reads) == 1
    assert posts[0]["tool_choice"] == "auto" and "parallel_tool_calls" not in posts[0]
    assert posts[0]["provider"]["data_collection"] == "deny"
    assert outcome["historical_unknown_cost_count"] == 2 and outcome["new_unknown_cost_count"] == 1
    assert outcome["total_actual_cost_usd"] is None and outcome["denominator"] == 204
    assert outcome["stage_counts"] == {"failed": 1, "completed": 0, "started": 0, "not_started": 203}
    directory = runner.campaign_directory(bundle["root"])
    receipt = previous.legacy.read_json(directory / "e1/e1_001.receipt.json")
    assert receipt["envelope"]["http_diagnostic"]["api_error_reason"] == "no_endpoints_data_policy"
    result = previous.legacy.read_json(directory / "e1/results.private.json")
    assert result["denominator"] == len(result["rows"]) == 204
    assert all(row["status"] == "not_started" for row in result["rows"][1:])
    assert all(text not in (directory / "e1/e1_001.receipt.json").read_text()
               for text in ("private sentinel", "private HTTP reason", SYNTHETIC_KEY))
    assert [tree_hashes(old) for old in directories] == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    repeated = json.loads(capsys.readouterr().out)
    assert "new_campaign_already_exists" in repeated["blocking_reasons"]
    assert len(posts) == len(key_reads) == 1
    assert [tree_hashes(old) for old in directories] == before
