"""Exact continuation authorization uses temporary ledgers and fake HTTP only."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import run_e1_authorized as runner
from scripts import run_e1_compat as parent, run_e1_t71 as original, t74_transport
from tests.test_e1_compat import metadata, opener_for
from tests.test_t74_run import (  # noqa: F401 -- pytest fixture dependencies
    REAL_ROOT, SYNTHETIC_KEY, artifacts, histories, inventories,
    no_real_network_or_credentials, source_bundle, tree_hashes, write_json,
)


@pytest.fixture
def authorized_history(histories):
    bundle, directories, registration = histories
    write_json(bundle["root"] / runner.REGISTRATION, runner.expected_registration(bundle))
    return bundle, directories, registration


def patch_cli(monkeypatch, bundle, *, documents=None):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(original, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    calls, opener = opener_for(documents if documents is not None else metadata(bundle))
    fetch = parent.compat._documents
    monkeypatch.setattr(parent.compat, "_documents", lambda *, enable_network=False, **kwargs:
                        fetch(opener=opener, enable_network=enable_network))
    return calls


def test_exact_two_unknowns_are_accepted_but_original_ledgers_remain_null_and_stopped(authorized_history):
    bundle, directories, _ = authorized_history
    before = [tree_hashes(path) for path in directories]
    state = runner.verify_history(bundle)
    assert state["unknown_cost_count"] == 2
    assert state["unaccepted_unknown_attempts"] == []
    assert state["total_actual_cost_usd"] is None
    assert runner.ACCEPTED_UNKNOWN_ATTEMPTS == tuple((name, "e1_001") for name in parent.HISTORY_IDS)
    for directory in directories:
        with sqlite3.connect((directory / "campaign.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
            assert conn.execute("SELECT cost_usd FROM items WHERE id='e1_001'").fetchone() == (None,)
            assert conn.execute("SELECT stopped FROM campaign").fetchone() == (1,)
    assert [tree_hashes(path) for path in directories] == before


def test_authorized_registration_binds_original_wire_inventory_budget_and_scope(authorized_history):
    bundle, _, registration = authorized_history
    accepted = runner._registration(bundle)
    for key in ("wire_changes", "fixed_generation_positions", "prior_campaigns", "budget_usd"):
        assert accepted[key] == registration[key]
    assert accepted["authorization"]["permits_future_unknown_cost_exception"] is False
    assert accepted["authorization"]["permits_automatic_retry"] is False
    assert accepted["authorization"]["retain_original_null_costs"] is True
    assert accepted["compatibility_seal_sha256"] == runner.COMPATIBILITY_SEAL


@pytest.mark.parametrize("change", ["third_unknown", "future_unknown", "retry", "budget", "wire", "statement"])
def test_authorization_cannot_expand_to_future_unknowns_or_other_work(authorized_history, change):
    bundle, _, _ = authorized_history
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    if change == "third_unknown":
        value["accepted_unknown_attempts"].append([parent.HISTORY_IDS[1], "e1_002"])
    elif change == "future_unknown":
        value["authorization"]["permits_future_unknown_cost_exception"] = True
    elif change == "retry":
        value["authorization"]["permits_automatic_retry"] = True
    elif change == "budget":
        value["budget_usd"]["e1"] = "0.6"
    elif change == "wire":
        value["wire_changes"] = []
    else:
        value["authorization"]["user_statement"] = "synthetic broader approval"
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises(runner.RunError, match="exact_continuation_authorization_changed"):
        runner.verify_history(bundle)


def test_project_private_directory_and_original_entry_globals_are_unchanged(authorized_history):
    bundle, _, _ = authorized_history
    old = (parent.ROOT, parent._registration, parent.verify_execution, parent.verify_history,
           parent.preflight, parent.main, parent.campaign_directory, parent.ACCEPTED_UNKNOWN_ATTEMPTS)
    namespace = runner._namespace()
    root = bundle["root"]
    directory = runner.campaign_directory(root)
    assert directory == root.absolute() / "artifacts/local/e1_t74_20261001_v1"
    assert directory.is_relative_to(root.absolute())
    assert namespace["ACCEPTED_UNKNOWN_ATTEMPTS"] == runner.ACCEPTED_UNKNOWN_ATTEMPTS
    assert namespace["verify_execution"] is runner.verify_execution
    assert namespace["verify_history"].__globals__ is namespace
    assert namespace["preflight"].__globals__ is namespace
    assert namespace["main"].__globals__ is namespace
    assert old == (parent.ROOT, parent._registration, parent.verify_execution, parent.verify_history,
                   parent.preflight, parent.main, parent.campaign_directory, parent.ACCEPTED_UNKNOWN_ATTEMPTS)
    assert parent.ACCEPTED_UNKNOWN_ATTEMPTS == ((parent.HISTORY_IDS[0], "e1_001"),)


def test_offline_default_never_reads_key_or_creates_campaign(authorized_history, monkeypatch, capsys):
    bundle, directories, _ = authorized_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert calls == [] and report["ready_to_send"] is False
    assert report["history"]["unaccepted_unknown_attempts"] == []
    assert report["blocking_reasons"] == ["live_metadata_not_checked"]
    assert report["credentials_read"] is False and report["model_requests"] == 0
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(path) for path in directories] == before


def test_live_public_preflight_passes_all_204_positions_without_credentials(authorized_history, monkeypatch, capsys):
    bundle, directories, _ = authorized_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--online-preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is True
    assert report["parameter_ready_to_send"] is True
    assert report["parameter_compatibility_verified"] is True
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["planned_full_e1_reserve_usd"] == "0.2244"
    assert report["history"]["unknown_cost_count"] == 2
    assert report["history"]["unaccepted_unknown_attempts"] == []
    assert report["credentials_read"] is False and report["model_requests"] == 0
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(path) for path in directories] == before


def test_parameter_mode_failure_remains_a_gate_before_credentials(authorized_history, monkeypatch, capsys):
    bundle, _, _ = authorized_history
    documents = metadata(bundle)
    documents[parent.compat.choices.ENDPOINT]["data"]["endpoints"][0]["supports_tool_choice"]["auto"] = False
    calls = patch_cli(monkeypatch, bundle, documents=documents)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is False
    assert report["blocking_reasons"] == ["tool_choice_mode_not_supported"]
    assert report["history"]["unaccepted_unknown_attempts"] == []
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_stage_reservation_gate_is_not_waived_by_fee_exceptions(authorized_history, monkeypatch, capsys):
    bundle, _, _ = authorized_history
    patch_cli(monkeypatch, bundle)
    assess = parent.compat.preflight

    def excessive(*args, **kwargs):
        report = assess(*args, **kwargs)
        report["planned_full_e1_reserve_usd"] = "0.5001"
        return report

    monkeypatch.setattr(parent.compat, "preflight", excessive)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blocking_reasons"] == ["new_stage_reservation_failed"]
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_without_explicit_local_env_flag_stops_before_public_checks(authorized_history, monkeypatch, capsys):
    bundle, _, _ = authorized_history
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("argument", ["--ignore-unknown-cost", "--retry", "--budget", "--key"])
def test_no_cost_override_retry_budget_or_key_command_line(argument):
    with pytest.raises(SystemExit) as caught:
        runner.main([argument])
    assert caught.value.code == 2


def test_history_mutation_blocks_before_credentials_and_public_get(authorized_history, monkeypatch, capsys):
    bundle, directories, _ = authorized_history
    calls = patch_cli(monkeypatch, bundle)
    with (directories[1] / "e1/e1_001.receipt.json").open("ab") as file:
        file.write(b"\n")
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "historical_artifact_changed"
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


def test_linked_project_private_directory_is_rejected_before_credentials(authorized_history, monkeypatch, capsys):
    bundle, _, _ = authorized_history
    calls = patch_cli(monkeypatch, bundle)
    path = runner.campaign_directory(bundle["root"]).parent
    old = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == path or old(p))
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "linked_path_rejected"
    assert len(calls) == 2 and not runner.campaign_directory(bundle["root"]).exists()


def test_claim_is_durable_before_single_post_and_new_unknown_stops_without_retry(authorized_history, monkeypatch, capsys):
    bundle, directories, _ = authorized_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    keys, posts = [], []

    def credential(root):
        assert root == bundle["root"]
        keys.append(True)
        return SYNTHETIC_KEY

    def failed_http(request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t74_transport.ENDPOINT
        directory = runner.campaign_directory(bundle["root"])
        ledger = parent._ReadOnlyLedger(directory / "campaign.sqlite3", parent.CAMPAIGN_ID, "a" * 64)
        assert ledger.summary()["outstanding_count"] == 1
        assert (directory / "e1/e1_001.request.json").is_file()
        posts.append(json.loads(request.data))
        body = {"error": {"code": 404, "message":
            "No endpoints found that match your data policy. private sentinel"}}
        raise HTTPError(request.full_url, 404, "private HTTP reason", {},
                        io.BytesIO(json.dumps(body).encode()))

    def transport(key, *, enable_network):
        assert key == SYNTHETIC_KEY and enable_network is True
        return t74_transport.OpenRouterTransport(key, opener=failed_http)

    monkeypatch.setattr(parent.old_run, "credential", credential)
    monkeypatch.setattr(parent, "OpenRouterTransport", transport)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    outcome = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(keys) == len(posts) == 1
    assert posts[0]["tool_choice"] == "auto" and "parallel_tool_calls" not in posts[0]
    assert posts[0]["provider"]["data_collection"] == "deny"
    assert outcome["historical_unknown_cost_count"] == 2
    assert outcome["new_unknown_cost_count"] == 1
    assert outcome["new_known_cost_usd"] == "0" and outcome["total_actual_cost_usd"] is None
    assert outcome["stage_counts"] == {"failed": 1, "completed": 0, "started": 0, "not_started": 203}
    directory = runner.campaign_directory(bundle["root"])
    marker = original.legacy.read_json(directory / "campaign.json")
    assert marker["accepted_unknown_attempts"] == [list(a) for a in runner.ACCEPTED_UNKNOWN_ATTEMPTS]
    assert marker["execution_seal"] == "a" * 64
    result = original.legacy.read_json(directory / "e1/results.private.json")
    assert result["denominator"] == len(result["rows"]) == 204
    assert all(row["status"] == "not_started" for row in result["rows"][1:])
    receipt_path = directory / "e1/e1_001.receipt.json"
    receipt = original.legacy.read_json(receipt_path)
    assert receipt["envelope"]["http_diagnostic"]["api_error_reason"] == "no_endpoints_data_policy"
    assert all(value not in receipt_path.read_text() for value in ("private sentinel", "private HTTP reason", SYNTHETIC_KEY))
    assert [tree_hashes(path) for path in directories] == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    retry = json.loads(capsys.readouterr().out)
    assert "new_campaign_already_exists" in retry["blocking_reasons"]
    assert len(keys) == len(posts) == 1
    assert [tree_hashes(path) for path in directories] == before


def sealed_fixture(authorized_history, monkeypatch):
    bundle, directories, _ = authorized_history
    monkeypatch.setattr(parent, "verify_execution", lambda _: {"seal_sha256": runner.COMPATIBILITY_SEAL})
    for relative in ("scripts/run_e1_authorized.py", "tests/test_e1_authorized.py"):
        target = bundle["root"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / relative, target)
    seal = {"schema": "aria_t74_exact_continuation_v1",
        "compatibility_seal_sha256": runner.COMPATIBILITY_SEAL,
        "files": [{"path": relative, "bytes": (bundle["root"] / relative).stat().st_size,
            "sha256": hashlib.sha256((bundle["root"] / relative).read_bytes()).hexdigest()}
            for relative in sorted(runner.FILES)]}
    seal["seal_sha256"] = parent.compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    return bundle, directories, seal


def test_new_seal_verifies_and_binds_parent_seal(authorized_history, monkeypatch):
    bundle, _, seal = sealed_fixture(authorized_history, monkeypatch)
    assert runner.verify_execution(bundle) == seal
    monkeypatch.setattr(parent, "verify_execution", lambda _: {"seal_sha256": "f" * 64})
    with pytest.raises(runner.RunError, match="compatibility_parent_seal_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("relative", sorted(runner.FILES))
def test_every_new_source_or_registration_tamper_is_rejected(authorized_history, monkeypatch, relative):
    bundle, _, _ = sealed_fixture(authorized_history, monkeypatch)
    with (bundle["root"] / relative).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(runner.RunError, match="continuation_source_changed"):
        runner.verify_execution(bundle)


def test_new_seal_cannot_drop_source_inventory(authorized_history, monkeypatch):
    bundle, _, seal = sealed_fixture(authorized_history, monkeypatch)
    seal["files"] = seal["files"][:-1]
    seal.pop("seal_sha256")
    seal["seal_sha256"] = parent.compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    with pytest.raises(runner.RunError, match="continuation_source_inventory_changed"):
        runner.verify_execution(bundle)
