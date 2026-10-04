"""Fast schema-prompt entry tests; only invented first-item histories and HTTP."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import run_e1_json_schema_mode as runner, run_e1_t71 as original, t71_policy, t76_transport
from tests.test_e1_json_mode_run import json_body
from tests.test_e1_compat import metadata, opener_for
from tests.test_t74_run import (  # noqa: F401 -- imported pytest dependencies
    REAL_ROOT, SYNTHETIC_KEY, artifacts, no_real_network_or_credentials,
    source_bundle, tree_hashes, write_json,
)
from tests.test_t7_transport import Response


PARENT_STATE = {
    "known_cost_usd": "0.00592184", "unknown_cost_count": 2,
    "unaccepted_unknown_attempts": [], "total_actual_cost_usd": None,
    "campaigns": [
        {"campaign_id": "t6_20260930_v1", "known_cost_usd": "0", "unknown_cost_count": 1,
         "stopped": True, "files_verified": 7},
        {"campaign_id": "e1_t73_20261001_v1", "known_cost_usd": "0", "unknown_cost_count": 1,
         "stopped": True, "files_verified": 8},
        {"campaign_id": "e1_t74_20261001_v1", "known_cost_usd": "0.00171618", "unknown_cost_count": 0,
         "stopped": True, "files_verified": 30},
        {"campaign_id": "e1_t75_20261001_v1", "known_cost_usd": "0.00420566", "unknown_cost_count": 0,
         "stopped": True, "files_verified": 66},
    ],
}


@pytest.fixture
def schema_history(tmp_path, source_bundle, monkeypatch):
    """The parent verifier has its own tests; isolate this new eight-file scope."""
    root = tmp_path / "project"
    root.mkdir()
    bundle = {**source_bundle, "root": root}
    public = "experiments/t6_20260930/public_metadata.json"
    (root / public).parent.mkdir(parents=True)
    shutil.copyfile(REAL_ROOT / public, root / public)
    monkeypatch.setattr(runner.parent, "verify_history", lambda _: deepcopy(PARENT_STATE))
    directory = root / "artifacts/local" / runner.PARENT_CAMPAIGN
    directory.mkdir(parents=True)
    _, _, trials, items = runner.parent.compat.prepare_e1(bundle)
    first = items[0]
    trial = trials[first["case_id"], first["arm"]]
    envelope = json_body(trial["wire"])
    fields = json.loads(envelope["choices"][0]["message"]["content"])
    fields["communication_plan"]["pace"] = "pause留白"
    envelope["choices"][0]["message"]["content"] = json.dumps(fields, ensure_ascii=False)
    envelope["usage"]["cost"] = runner.PARENT_KNOWN_COST
    envelope["has_hidden_reasoning"] = False
    write_json(directory / "campaign.json", {"synthetic_fixture": True})
    write_json(directory / "preflight.json", {"synthetic_fixture": True})
    write_json(directory / "e1/inventory.json", items)
    write_json(directory / "e1/e1_001.request.json", trial["wire"])
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3", campaign_id=runner.PARENT_CAMPAIGN,
                                      freeze_seal=runner.PARENT_EXECUTION_SEAL)
    ledger.register_stage("e1", [{key: item[key] for key in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.49407816", max_requests=204)
    rows = [original.legacy.generation_row(item) for item in items]
    rows[0].update(status="generation_failed", reason="invalid_schema")
    digest = original.legacy.write_private(directory / "e1/e1_001.receipt.json",
        {"envelope": envelope, "error": "invalid_schema", "row": rows[0]})
    ledger.claim("e1", "e1_001")
    with pytest.raises(t71_policy.LedgerError):
        original.legacy.finish_receipt(ledger, "e1", "e1_001", envelope, digest, "invalid_schema")
    write_json(directory / "e1/results.private.json", {"denominator": 204, "rows": rows,
        "status": "incomplete", "stop_reason": "invalid_schema", "campaign": ledger.summary()})
    write_json(directory / "outcome.aggregate.json", {"synthetic_fixture": True})
    assert len(artifacts(directory)) == 8
    write_json(root / runner.REGISTRATION, runner.expected_registration(bundle, artifacts=artifacts(directory)))
    return bundle, directory


def patch_cli(monkeypatch, bundle, *, documents=None):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(original, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    calls, opener = opener_for(documents if documents is not None else metadata(bundle))
    preflight = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda bundle, online=False, **kwargs:
                        preflight(bundle, online=online, opener=opener))
    return calls


def test_parent_failed_enum_receipt_stays_failed_and_history_is_read_only(schema_history):
    bundle, directory = schema_history
    before = tree_hashes(directory)
    state = runner.verify_history(bundle)
    assert state["known_cost_usd"] == "0.00601653" and state["unknown_cost_count"] == 2
    assert state["unaccepted_unknown_attempts"] == [] and state["total_actual_cost_usd"] is None
    assert sum(row["files_verified"] for row in state["campaigns"]) == 119
    assert state["campaigns"][-1]["stage_counts"] == {
        "completed": 0, "failed": 1, "started": 0, "not_started": 203}
    assert tree_hashes(directory) == before


@pytest.mark.parametrize("relative", ["campaign.json", "campaign.sqlite3", "e1/e1_001.receipt.json"])
def test_prior_byte_change_blocks_before_get_and_credentials(schema_history, monkeypatch, capsys, relative):
    bundle, directory = schema_history
    calls = patch_cli(monkeypatch, bundle)
    with (directory / relative).open("ab") as stream:
        stream.write(b"\n")
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert "changed" in json.loads(capsys.readouterr().out)["reason"]
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("change", ["unknown", "reopened", "outstanding", "fee"])
def test_prior_state_change_is_not_accepted_by_updating_its_hashes(schema_history, change):
    bundle, directory = schema_history
    with sqlite3.connect(directory / "campaign.sqlite3") as conn:
        if change == "unknown":
            conn.execute("UPDATE items SET cost_usd=NULL WHERE id='e1_001'")
        elif change == "reopened":
            conn.execute("UPDATE campaign SET stopped=0,stop_reason=NULL")
        elif change == "outstanding":
            conn.execute("UPDATE items SET status='started',claim_token=? WHERE id='e1_002'", ("c" * 64,))
        else:
            conn.execute("UPDATE items SET cost_usd='0.0001' WHERE id='e1_001'")
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    value["prior_known_campaign"]["artifacts"] = artifacts(directory)
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises((runner.RunError, t71_policy.LedgerError)):
        runner.verify_history(bundle)


@pytest.mark.parametrize("change", ["budget", "third_unknown", "retry", "carry", "wire", "selection"])
def test_registration_cannot_expand_scope_or_upgrade_old_failure(schema_history, change):
    bundle, _ = schema_history
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    if change == "budget":
        value["budget_usd"]["e1"] = "0.6"
    elif change == "third_unknown":
        value["accepted_unknown_attempts"].append([runner.PARENT_CAMPAIGN, "e1_001"])
    elif change == "retry":
        value["authorization"]["permits_automatic_retry"] = True
    elif change == "carry":
        value["cumulative_prior_known_e1_cost_usd"] = "0"
    elif change == "wire":
        value["wire_changes_from_t76"]["all_task_persona_user_messages_unchanged"] = False
    else:
        value["measurement"] = "reuse_previous_successes_and_fix_invalid_enum"
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises(runner.RunError, match="json_schema_mode_registration_changed"):
        runner.verify_history(bundle)


def test_default_and_missing_permission_never_read_key_or_create_campaign(schema_history, monkeypatch, capsys):
    bundle, directory = schema_history
    before = tree_hashes(directory)
    calls = patch_cli(monkeypatch, bundle)
    path = runner.campaign_directory(bundle["root"])
    assert path == bundle["root"].absolute() / "artifacts/local/e1_t77_20261001_v1"
    assert runner.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert calls == [] and report["credentials_read"] is False and report["model_requests"] == 0
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"
    assert calls == [] and not path.exists() and tree_hashes(directory) == before


def test_live_preflight_retains_all_positions_and_original_shared_allowance(schema_history, monkeypatch, capsys):
    bundle, directory = schema_history
    before = tree_hashes(directory)
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--online-preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is True
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["remaining_e1_budget_usd"] == "0.49398347"
    assert report["combined_e1_reserve_usd"] == "0.23041653"
    assert report["contract_json_object"]["prompt_field_schema_embedded"] is True
    assert report["contract_json_object"]["server_field_schema_enforced"] is False
    assert report["response_policy_id"] == runner.compat.POLICY_ID
    assert not runner.campaign_directory(bundle["root"]).exists() and tree_hashes(directory) == before


def test_new_stage_persists_remaining_cap_and_reservation_still_blocks(schema_history, monkeypatch, capsys, tmp_path):
    bundle, _ = schema_history
    items = runner.compat.prepare_e1(bundle)[3]
    ledger = runner.CarryLedger(tmp_path / "cap.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="a" * 64)
    ledger.register_stage("e1", [{key: item[key] for key in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    assert ledger.summary()["stages"]["e1"]["limit_usd"] == "0.49398347"
    patch_cli(monkeypatch, bundle)
    preflight = runner.compat.preflight

    def excessive(*args, **kwargs):
        report = preflight(*args, **kwargs)
        report["planned_full_e1_reserve_usd"] = "0.494"
        return report

    monkeypatch.setattr(runner.compat, "preflight", excessive)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blocking_reasons"] == ["shared_e1_reservation_failed"]
    assert report["combined_e1_reserve_usd"] == "0.50001653"
    assert not runner.campaign_directory(bundle["root"]).exists()


def patch_execution(monkeypatch, bundle, fault, after_post=None):
    keys, posts = [], []
    generation = runner.compat.run_generation
    monkeypatch.setattr(runner.compat, "run_generation", lambda *args, **kwargs:
                        generation(*args, **kwargs, sleep=lambda _: None))

    def credential(root):
        assert root == bundle["root"]
        keys.append(True)
        return SYNTHETIC_KEY

    def invented_http(request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t76_transport.ENDPOINT
        path = runner.campaign_directory(bundle["root"])
        state = runner.core._ReadOnlyLedger(path / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64).summary()
        assert state["outstanding_count"] == 1 and state["stages"]["e1"]["limit_usd"] == "0.49398347"
        payload = json.loads(request.data)
        assert (path / "e1/e1_001.request.json").is_file()
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        posts.append(payload)
        if fault == "http":
            raise HTTPError(request.full_url, 404, "private reason", {}, io.BytesIO(b'{"error":{"code":404}}'))
        envelope = json_body(payload)
        if fault == "unknown":
            envelope["usage"].pop("cost")
        elif fault == "enum":
            fields = json.loads(envelope["choices"][0]["message"]["content"])
            fields["communication_plan"]["pace"] = "pause留白"
            envelope["choices"][0]["message"]["content"] = json.dumps(fields, ensure_ascii=False)
        if after_post is not None:
            after_post()
        return Response(envelope)

    def transport(key, *, enable_network):
        assert key == SYNTHETIC_KEY and enable_network is True
        return t76_transport.OpenRouterTransport(key, opener=invented_http)

    monkeypatch.setattr(runner.core.old_run, "credential", credential)
    monkeypatch.setattr(runner, "OpenRouterTransport", transport)
    return keys, posts


@pytest.mark.parametrize("fault,unknown", [("http", 1), ("unknown", 1), ("enum", 0)])
def test_single_failure_retains_original_schema_and_fee_then_blocks_retry(schema_history, monkeypatch, capsys, fault, unknown):
    bundle, directory = schema_history
    before = tree_hashes(directory)
    calls = patch_cli(monkeypatch, bundle)
    keys, posts = patch_execution(monkeypatch, bundle, fault)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(keys) == len(posts) == 1
    assert result["stage_counts"] == {"completed": 0, "failed": 1, "started": 0, "not_started": 203}
    assert result["new_unknown_cost_count"] == unknown and result["historical_unknown_cost_count"] == 2
    assert Decimal(result["cumulative_known_e1_cost_usd"]) == Decimal(runner.CARRY_KNOWN_COST) + (
        Decimal("0.0001") if fault == "enum" else Decimal(0))
    new = runner.campaign_directory(bundle["root"])
    marker = original.legacy.read_json(new / "campaign.json")
    assert len(marker["parent_campaign_ids"]) == 5 and marker["response_carriers"] == ["content_json"]
    assert marker["remaining_e1_budget_usd"] == "0.49398347" and marker["server_field_schema_enforced"] is False
    assert tree_hashes(directory) == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    assert "new_campaign_already_exists" in json.loads(capsys.readouterr().out)["blocking_reasons"]
    assert len(keys) == len(posts) == 1 and tree_hashes(directory) == before


def test_prior_mutation_after_successful_first_reply_stops_next_post(schema_history, monkeypatch, capsys):
    bundle, directory = schema_history
    patch_cli(monkeypatch, bundle)

    def mutate():
        with (directory / "e1/e1_001.receipt.json").open("ab") as stream:
            stream.write(b"\n")

    keys, posts = patch_execution(monkeypatch, bundle, None, mutate)
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "json_schema_mode_historical_bytes_changed"
    assert len(keys) == len(posts) == 1
    path = runner.campaign_directory(bundle["root"])
    assert (path / "e1/e1_001.receipt.json").is_file()
    state = runner.core._ReadOnlyLedger(path / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64).summary()
    assert state["stopped"] and state["known_cost_usd"] == "0.0001"


def sealed_fixture(schema_history, monkeypatch):
    bundle, directory = schema_history
    monkeypatch.setattr(runner.parent, "verify_execution", lambda _: {"seal_sha256": runner.PARENT_EXECUTION_SEAL})
    for relative in runner.FILES - {runner.REGISTRATION, runner.INVENTORY}:
        path = bundle["root"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / relative, path)
    items = runner.compat.prepare_e1(bundle)[3]
    write_json(bundle["root"] / runner.INVENTORY, {"items": items,
        "items_sha256": runner.compat.canonical_sha256(items)})
    seal = {"schema": "aria_e1_json_schema_mode_execution_v1", "parent_execution_seal": runner.PARENT_EXECUTION_SEAL,
        "files": [{"path": relative, "bytes": (bundle["root"] / relative).stat().st_size,
            "sha256": hashlib.sha256((bundle["root"] / relative).read_bytes()).hexdigest()}
            for relative in sorted(runner.FILES)]}
    seal["seal_sha256"] = runner.compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    return bundle, directory, seal


def test_execution_seal_binds_all_nine_and_parent_version(schema_history, monkeypatch):
    bundle, _, seal = sealed_fixture(schema_history, monkeypatch)
    assert runner.verify_execution(bundle) == seal and len(seal["files"]) == 9
    monkeypatch.setattr(runner.parent, "verify_execution", lambda _: {"seal_sha256": "f" * 64})
    with pytest.raises(runner.RunError, match="json_schema_mode_parent_seal_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("relative", sorted(runner.FILES))
def test_every_new_sealed_source_or_inventory_change_is_refused(schema_history, monkeypatch, relative):
    bundle, _, _ = sealed_fixture(schema_history, monkeypatch)
    with (bundle["root"] / relative).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(runner.RunError, match="json_schema_mode_source_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("flag", ["--retry", "--ignore-unknown-cost", "--budget", "--key"])
def test_no_retry_fee_budget_or_direct_key_override(flag):
    with pytest.raises(SystemExit) as caught:
        runner.main([flag])
    assert caught.value.code == 2
