"""Prospective carrier entry uses invented receipts and temporary ledgers only."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import e1_compat, run_e1_authorized as authorized
from scripts import run_e1_carrier as runner
from scripts import run_e1_compat as parent, run_e1_t71 as original, t71_policy, t74_transport
from tests.test_e1_compat import metadata, opener_for
from tests.test_e1_execution import fake_envelope
from tests.test_t7_transport import Response
from tests.test_t74_run import (  # noqa: F401 -- pytest fixture dependencies
    REAL_ROOT, SYNTHETIC_KEY, artifacts, histories, inventories,
    no_real_network_or_credentials, source_bundle, tree_hashes, write_json,
)


def known_stopped_history(directory, bundle):
    """Twelve invented visible responses represent the stopped parent shape."""
    _, cases, trials, items = e1_compat.prepare_e1(bundle)
    directory.mkdir(parents=True)
    write_json(directory / "campaign.json", {
        "campaign_id": runner.PARENT_CAMPAIGN,
        "execution_seal": runner.PARENT_AUTHORIZATION_SEAL,
        "parent_campaign_ids": list(parent.HISTORY_IDS),
        "historical_unknown_cost_count": 2,
        "accepted_unknown_attempts": [list(a) for a in authorized.ACCEPTED_UNKNOWN_ATTEMPTS],
        "linked_inventory_sha256": e1_compat.canonical_sha256(items),
    })
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id=runner.PARENT_CAMPAIGN, freeze_seal=runner.PARENT_AUTHORIZATION_SEAL)
    ledger.register_stage("e1", [{k: item[k] for k in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    write_json(directory / "preflight.json", {"synthetic_fixture": True})
    write_json(directory / "e1/inventory.json", items)
    rows = [original.legacy.generation_row(item) for item in items]
    for index, item in enumerate(items[:12]):
        trial = trials[item["case_id"], item["arm"]]
        envelope = fake_envelope(trial["wire"])
        envelope["usage"]["cost"] = "0.0001" if index < 11 else "0.00061618"
        error = None
        if index < 11:
            rows[index].update(original.legacy.parse_generation(envelope, trial, cases[item["case_id"]]))
        else:
            message = envelope["choices"][0]["message"]
            message["content"] = message.pop("tool_calls")[0]["function"]["arguments"]
            envelope["choices"][0]["finish_reason"] = "stop"
            error = "invalid_schema"
            rows[index].update(status="generation_failed", reason=error)
        write_json(directory / "e1" / (item["id"] + ".request.json"), trial["wire"])
        digest = original.legacy.write_private(directory / "e1" / (item["id"] + ".receipt.json"), {
            "envelope": envelope, "error": error, "row": rows[index],
        })
        ledger.claim("e1", item["id"])
        if error:
            with pytest.raises(t71_policy.LedgerError):
                original.legacy.finish_receipt(ledger, "e1", item["id"], envelope, digest, error)
        else:
            original.legacy.finish_receipt(ledger, "e1", item["id"], envelope, digest)
    write_json(directory / "e1/results.private.json", {
        "schema": "aria_e1_live_result_v1", "denominator": 204, "status": "incomplete",
        "stop_reason": "invalid_schema", "rows": rows, "campaign": ledger.summary(),
        "model_effect_measured": False,
    })
    write_json(directory / "outcome.aggregate.json", {
        "schema": "aria_e1_compatible_outcome_v1", "campaign_id": runner.PARENT_CAMPAIGN,
        "ok": False, "status": "incomplete", "stop_reason": "invalid_schema", "denominator": 204,
        "stage_counts": ledger.summary()["stages"]["e1"]["counts"],
        "new_known_cost_usd": runner.CARRY_KNOWN_COST, "new_unknown_cost_count": 0,
        "historical_unknown_cost_count": 2, "total_actual_cost_usd": None,
        "model_effect_measured": False, "no_automatic_retry": True,
    })
    assert len(artifacts(directory)) == 30


@pytest.fixture
def carrier_history(histories):
    bundle, directories, _ = histories
    write_json(bundle["root"] / authorized.REGISTRATION, authorized.expected_registration(bundle))
    known = bundle["root"] / "artifacts/local" / runner.PARENT_CAMPAIGN
    known_stopped_history(known, bundle)
    write_json(bundle["root"] / runner.REGISTRATION,
               runner.expected_registration(bundle, artifacts=artifacts(known)))
    return bundle, [*directories, known]


def patch_cli(monkeypatch, bundle, *, documents=None):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(original, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    calls, opener = opener_for(documents if documents is not None else metadata(bundle))
    fetch = e1_compat._documents
    monkeypatch.setattr(e1_compat, "_documents", lambda *, enable_network=False, **kwargs:
                        fetch(opener=opener, enable_network=enable_network))
    return calls


def test_three_stopped_histories_remain_read_only_with_two_exact_unknowns(carrier_history):
    bundle, directories = carrier_history
    before = [tree_hashes(path) for path in directories]
    state = runner.verify_history(bundle)
    assert state["unknown_cost_count"] == 2
    assert state["known_cost_usd"] == runner.CARRY_KNOWN_COST
    assert state["unaccepted_unknown_attempts"] == []
    assert state["total_actual_cost_usd"] is None
    assert [row["campaign_id"] for row in state["campaigns"]] == [*parent.HISTORY_IDS, runner.PARENT_CAMPAIGN]
    with sqlite3.connect((directories[2] / "campaign.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT stopped,stop_reason FROM campaign").fetchone() == (1, "invalid_schema")
        assert dict(conn.execute("SELECT status,COUNT(*) FROM items GROUP BY status")) == {
            "completed": 11, "failed": 1, "not_started": 192}
    assert [tree_hashes(path) for path in directories] == before


@pytest.mark.parametrize("index,relative", [(0, "campaign.json"), (1, "e1/e1_001.receipt.json"),
                                           (2, "e1/e1_012.receipt.json")])
def test_bound_parent_mutation_blocks_before_credentials_or_get(carrier_history, monkeypatch, capsys, index, relative):
    bundle, directories = carrier_history
    calls = patch_cli(monkeypatch, bundle)
    with (directories[index] / relative).open("ab") as file:
        file.write(b"\n")
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "changed" in report["reason"]
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("change", ["unknown", "reopened", "outstanding", "reordered"])
def test_parent_state_must_match_even_with_fresh_file_hashes(carrier_history, change):
    bundle, directories = carrier_history
    directory = directories[2]
    with sqlite3.connect(directory / "campaign.sqlite3") as conn:
        if change == "unknown":
            conn.execute("UPDATE items SET cost_usd=NULL WHERE id='e1_012'")
        elif change == "reopened":
            conn.execute("UPDATE campaign SET stopped=0,stop_reason=NULL")
        elif change == "outstanding":
            conn.execute("UPDATE items SET status='started',claim_token=? WHERE id='e1_013'", ("c" * 64,))
        else:
            conn.execute("UPDATE items SET status='failed',error='invalid_schema' WHERE id='e1_011'")
            conn.execute("UPDATE items SET status='completed',error=NULL WHERE id='e1_012'")
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    value["prior_known_campaign"]["artifacts"] = artifacts(directory)
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises((runner.RunError, t71_policy.LedgerError)):
        runner.verify_history(bundle)


@pytest.mark.parametrize("argument", ["--retry", "--ignore-unknown-cost", "--budget", "--key"])
def test_cli_has_no_retry_fee_budget_or_credential_override(argument):
    with pytest.raises(SystemExit) as caught:
        runner.main([argument])
    assert caught.value.code == 2


def test_new_campaign_is_inside_cleaned_project_without_parent_reuse(carrier_history):
    bundle, _ = carrier_history
    directory = runner.campaign_directory(bundle["root"])
    assert directory == bundle["root"].absolute() / "artifacts/local/e1_t75_20261001_v1"
    assert directory.is_relative_to(bundle["root"].absolute())
    assert directory != bundle["root"] / "artifacts/local" / runner.PARENT_CAMPAIGN


def test_default_offline_does_not_read_credentials_or_create_campaign(carrier_history, monkeypatch, capsys):
    bundle, directories = carrier_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert calls == [] and report["ready_to_send"] is False
    assert report["credentials_read"] is False and report["model_requests"] == 0
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(path) for path in directories] == before


def test_missing_local_env_permission_stops_before_public_get(carrier_history, monkeypatch, capsys):
    bundle, _ = carrier_history
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


def test_fresh_preflight_reserves_whole_inventory_and_carries_known_fee(carrier_history, monkeypatch, capsys):
    bundle, directories = carrier_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--online-preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is True
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["history"]["known_cost_usd"] == runner.CARRY_KNOWN_COST
    assert Decimal(report["planned_full_e1_reserve_usd"]) + Decimal(runner.CARRY_KNOWN_COST) <= Decimal("0.5")
    assert report["credentials_read"] is False and report["model_requests"] == 0
    assert not runner.campaign_directory(bundle["root"]).exists()
    assert [tree_hashes(path) for path in directories] == before


@pytest.mark.parametrize("change", ["budget", "third_unknown", "future_unknown", "retry", "carry", "wire", "measurement"])
def test_amendment_cannot_expand_budget_or_rewrite_history_authorization(carrier_history, change):
    bundle, _ = carrier_history
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    if change == "budget":
        value["budget_usd"]["e1"] = "0.6"
    elif change == "third_unknown":
        value["accepted_unknown_attempts"].append([runner.PARENT_CAMPAIGN, "e1_012"])
    elif change == "future_unknown":
        value["authorization"]["permits_future_unknown_cost_exception"] = True
    elif change == "retry":
        value["authorization"]["permits_automatic_retry"] = True
    elif change == "carry":
        value["prior_known_campaign"]["known_cost_usd"] = "0"
    elif change == "wire":
        value["wire_changes_from_t74"] = [{"field": "max_tokens", "to": 2048}]
    else:
        value["measurement"] = "reuse_first_11_and_retry_only_failure"
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises(runner.RunError, match="carrier_registration_changed"):
        runner.verify_history(bundle)


def test_stage_cap_is_persisted_after_subtracting_prior_cost(carrier_history, tmp_path):
    bundle, _ = carrier_history
    items = e1_compat.prepare_e1(bundle)[3]
    ledger = runner.CarryLedger(tmp_path / "carry.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="a" * 64)
    ledger.register_stage("e1", [{k: item[k] for k in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    assert ledger.summary()["stages"]["e1"]["limit_usd"] == "0.49828382"
    with sqlite3.connect(ledger.path.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT limit_usd FROM stages WHERE name='e1'").fetchone() == ("0.49828382",)


@pytest.mark.parametrize("stage,limit,maximum", [("judge", "0.5", 204), ("e1", "0.6", 204),
                                               ("e1", "0.49828382", 204), ("e1", "0.5", 205)])
def test_carried_ledger_cannot_expand_stage_scope(tmp_path, stage, limit, maximum):
    ledger = runner.CarryLedger(tmp_path / "scope.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="a" * 64)
    with pytest.raises(runner.RunError, match="carrier_stage_scope_changed"):
        ledger.register_stage(stage, [{"id": "e1_001", "request_sha256": "b" * 64,
            "reserve_usd": "0.0011"}], limit_usd=limit, max_requests=maximum)


def test_carried_full_reservation_blocks_before_credentials(carrier_history, monkeypatch, capsys):
    bundle, _ = carrier_history
    patch_cli(monkeypatch, bundle)
    assess = runner.compat.preflight

    def excessive(*args, **kwargs):
        report = assess(*args, **kwargs)
        report["planned_full_e1_reserve_usd"] = "0.4983"
        return report

    monkeypatch.setattr(runner.compat, "preflight", excessive)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blocking_reasons"] == ["shared_e1_reservation_failed"]
    assert report["combined_e1_reserve_usd"] == "0.50001618"
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_mode_capability_failure_still_blocks_before_credentials(carrier_history, monkeypatch, capsys):
    bundle, _ = carrier_history
    documents = metadata(bundle)
    documents[e1_compat.choices.ENDPOINT]["data"]["endpoints"][0]["supports_tool_choice"]["auto"] = False
    calls = patch_cli(monkeypatch, bundle, documents=documents)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is False
    assert report["blocking_reasons"] == ["tool_choice_mode_not_supported"]
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_existing_new_campaign_never_reads_credentials_or_restarts(carrier_history, monkeypatch, capsys):
    bundle, _ = carrier_history
    calls = patch_cli(monkeypatch, bundle)
    directory = runner.campaign_directory(bundle["root"])
    directory.mkdir()
    write_json(directory / "existing.json", {"immutable": True})
    before = tree_hashes(directory)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and "new_campaign_already_exists" in report["blocking_reasons"]
    assert tree_hashes(directory) == before


def patch_execution(monkeypatch, bundle, *, fault=None, after_post=None):
    key_reads, posts = [], []
    planned = e1_compat.prepare_e1(bundle)[3]
    generation = runner.compat.run_generation
    monkeypatch.setattr(runner.compat, "run_generation", lambda *args, **kwargs:
                        generation(*args, **kwargs, sleep=lambda _: None))

    def credential(root):
        assert root == bundle["root"]
        key_reads.append(True)
        return SYNTHETIC_KEY

    def invented_http(request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t74_transport.ENDPOINT
        directory = runner.campaign_directory(bundle["root"])
        ledger = parent._ReadOnlyLedger(directory / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64)
        state = ledger.summary()
        assert state["outstanding_count"] == 1
        assert state["stages"]["e1"]["limit_usd"] == "0.49828382"
        item = planned[len(posts)]
        payload = json.loads(request.data)
        assert (directory / "e1" / (item["id"] + ".request.json")).is_file()
        assert e1_compat.canonical_sha256(payload) == item["request_sha256"]
        assert payload["tool_choice"] == "auto" and "parallel_tool_calls" not in payload
        assert payload["provider"]["data_collection"] == "deny"
        posts.append(payload)
        if fault == "http":
            raise HTTPError(request.full_url, 404, "private reason", {},
                            io.BytesIO(b'{"error":{"code":404,"message":"private sentinel"}}'))
        response = fake_envelope(payload)
        if len(posts) % 2:
            message = response["choices"][0]["message"]
            message["content"] = message.pop("tool_calls")[0]["function"]["arguments"]
            response["choices"][0]["finish_reason"] = "stop"
        if fault == "unknown":
            response["usage"].pop("cost")
        elif fault == "schema":
            response["choices"][0]["message"]["content"] = "invented non-JSON prose"
        if after_post is not None:
            after_post(len(posts))
        return Response(response)

    def transport(key, *, enable_network):
        assert key == SYNTHETIC_KEY and enable_network is True
        return t74_transport.OpenRouterTransport(key, opener=invented_http)

    monkeypatch.setattr(parent.old_run, "credential", credential)
    monkeypatch.setattr(runner, "OpenRouterTransport", transport)
    return key_reads, posts


def test_complete_new_204_positions_preserve_real_carriers_and_shared_budget(carrier_history, monkeypatch, capsys):
    bundle, directories = carrier_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    keys, posts = patch_execution(monkeypatch, bundle)
    assert runner.main(["--execute", "--allow-local-env"]) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(keys) == 1 and len(posts) == 204
    assert outcome["stage_counts"] == {"completed": 204, "failed": 0, "started": 0, "not_started": 0}
    assert outcome["denominator"] == 204 and outcome["model_effect_measured"] is False
    assert outcome["new_known_cost_usd"] == "0.0204"
    assert outcome["cumulative_known_e1_cost_usd"] == "0.02211618"
    assert outcome["historical_unknown_cost_count"] == 2 and outcome["new_unknown_cost_count"] == 0
    assert outcome["total_actual_cost_usd"] is None
    directory = runner.campaign_directory(bundle["root"])
    result = original.legacy.read_json(directory / "e1/results.private.json")
    assert len(result["rows"]) == 204 and all(row["status"] == "generated" for row in result["rows"])
    marker = original.legacy.read_json(directory / "campaign.json")
    assert marker["parent_campaign_ids"] == [*parent.HISTORY_IDS, runner.PARENT_CAMPAIGN]
    assert marker["prior_known_e1_cost_usd"] == runner.CARRY_KNOWN_COST
    assert marker["remaining_e1_budget_usd"] == "0.49828382"
    assert marker["accepted_unknown_attempts"] == [list(a) for a in runner.ACCEPTED_UNKNOWN_ATTEMPTS]
    first = original.legacy.read_json(directory / "e1/e1_001.receipt.json")
    second = original.legacy.read_json(directory / "e1/e1_002.receipt.json")
    assert first["envelope"]["choices"][0]["message"]["content"]
    assert not first["envelope"]["choices"][0]["message"].get("tool_calls")
    assert second["envelope"]["choices"][0]["message"]["tool_calls"]
    assert [tree_hashes(path) for path in directories] == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    assert "new_campaign_already_exists" in json.loads(capsys.readouterr().out)["blocking_reasons"]
    assert len(keys) == 1 and len(posts) == 204


@pytest.mark.parametrize("fault,expected_unknown", [("http", 1), ("unknown", 1), ("schema", 0)])
def test_new_failure_is_persistent_without_retry_or_extending_old_fee_exception(carrier_history, monkeypatch, capsys, fault, expected_unknown):
    bundle, directories = carrier_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    keys, posts = patch_execution(monkeypatch, bundle, fault=fault)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    outcome = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(keys) == len(posts) == 1
    assert outcome["historical_unknown_cost_count"] == 2 and outcome["new_unknown_cost_count"] == expected_unknown
    assert outcome["stage_counts"] == {"completed": 0, "failed": 1, "started": 0, "not_started": 203}
    assert outcome["no_automatic_retry"] is True and outcome["total_actual_cost_usd"] is None
    assert Decimal(outcome["cumulative_known_e1_cost_usd"]) == Decimal(runner.CARRY_KNOWN_COST) + (
        Decimal("0.0001") if fault == "schema" else Decimal("0"))
    directory = runner.campaign_directory(bundle["root"])
    result = original.legacy.read_json(directory / "e1/results.private.json")
    assert len(result["rows"]) == 204 and all(row["status"] == "not_started" for row in result["rows"][1:])
    raw = (directory / "e1/e1_001.receipt.json").read_text(encoding="utf-8")
    assert all(text not in raw for text in (SYNTHETIC_KEY, "private sentinel", "private reason"))
    assert [tree_hashes(path) for path in directories] == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    assert "new_campaign_already_exists" in json.loads(capsys.readouterr().out)["blocking_reasons"]
    assert len(keys) == len(posts) == 1


def test_receipt_save_failure_keeps_observed_known_fee_and_stops(carrier_history, monkeypatch, capsys):
    bundle, _ = carrier_history
    patch_cli(monkeypatch, bundle)
    keys, posts = patch_execution(monkeypatch, bundle)
    write = original.legacy.write_private
    new_directory = runner.campaign_directory(bundle["root"])

    def failed_save(path, value):
        if path.is_relative_to(new_directory) and path.name.endswith(".receipt.json"):
            raise OSError("synthetic private path")
        return write(path, value)

    monkeypatch.setattr(original.legacy, "write_private", failed_save)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    outcome = json.loads(capsys.readouterr().out)
    assert len(keys) == len(posts) == 1
    assert outcome["stop_reason"] == "result_save_failed"
    assert outcome["new_known_cost_usd"] == "0.0001" and outcome["new_unknown_cost_count"] == 0
    assert outcome["cumulative_known_e1_cost_usd"] == "0.00181618"
    assert outcome["stage_counts"]["not_started"] == 203


def test_history_changed_after_one_post_blocks_second_and_retains_first_receipt(carrier_history, monkeypatch, capsys):
    bundle, directories = carrier_history
    patch_cli(monkeypatch, bundle)

    def mutate(count):
        if count == 1:
            with (directories[2] / "e1/e1_012.receipt.json").open("ab") as file:
                file.write(b"\n")

    keys, posts = patch_execution(monkeypatch, bundle, after_post=mutate)
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "carrier_historical_bytes_changed"
    assert len(keys) == len(posts) == 1
    directory = runner.campaign_directory(bundle["root"])
    assert (directory / "e1/e1_001.receipt.json").is_file()
    state = parent._ReadOnlyLedger(directory / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64).summary()
    assert state["stopped"] and state["known_cost_usd"] == "0.0001"
    assert state["stages"]["e1"]["counts"]["not_started"] == 203


def sealed_fixture(carrier_history, monkeypatch):
    bundle, directories = carrier_history
    monkeypatch.setattr(authorized, "verify_execution", lambda _: {"seal_sha256": runner.PARENT_AUTHORIZATION_SEAL})
    for relative in runner.FILES - {runner.REGISTRATION}:
        target = bundle["root"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / relative, target)
    seal = {"schema": "aria_e1_dual_carrier_execution_v1",
        "parent_authorization_seal": runner.PARENT_AUTHORIZATION_SEAL,
        "files": [{"path": relative, "bytes": (bundle["root"] / relative).stat().st_size,
            "sha256": hashlib.sha256((bundle["root"] / relative).read_bytes()).hexdigest()}
            for relative in sorted(runner.FILES)]}
    seal["seal_sha256"] = e1_compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    return bundle, directories, seal


def test_new_carrier_seal_binds_parent_authorization_and_all_new_source(carrier_history, monkeypatch):
    bundle, _, seal = sealed_fixture(carrier_history, monkeypatch)
    assert runner.verify_execution(bundle) == seal
    monkeypatch.setattr(authorized, "verify_execution", lambda _: {"seal_sha256": "f" * 64})
    with pytest.raises(runner.RunError, match="carrier_parent_seal_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("relative", sorted(runner.FILES))
def test_every_new_source_is_sealed(carrier_history, monkeypatch, relative):
    bundle, _, _ = sealed_fixture(carrier_history, monkeypatch)
    with (bundle["root"] / relative).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(runner.RunError, match="carrier_source_changed"):
        runner.verify_execution(bundle)


def test_new_seal_cannot_omit_entry_or_receipt_parser(carrier_history, monkeypatch):
    bundle, _, seal = sealed_fixture(carrier_history, monkeypatch)
    seal["files"] = seal["files"][:-1]
    seal.pop("seal_sha256")
    seal["seal_sha256"] = e1_compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    with pytest.raises(runner.RunError, match="carrier_source_inventory_changed"):
        runner.verify_execution(bundle)
