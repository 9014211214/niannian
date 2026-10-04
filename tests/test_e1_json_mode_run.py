"""JSON-mode entry tests use invented histories, synthetic keys and fake HTTP."""
from decimal import Decimal
import hashlib
import io
import json
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import run_e1_json_mode as runner, run_e1_t71 as original, t71_policy, t76_transport
from tests.test_e1_carrier_run import (  # noqa: F401 -- imported pytest dependencies
    REAL_ROOT, SYNTHETIC_KEY, artifacts, carrier_history, histories, inventories,
    no_real_network_or_credentials, source_bundle, tree_hashes, write_json,
)
from tests.test_e1_compat import metadata, opener_for
from tests.test_e1_execution import fake_envelope
from tests.test_t7_transport import Response


def json_body(payload):
    envelope = fake_envelope(payload)
    message = envelope["choices"][0]["message"]
    message["content"] = message.pop("tool_calls")[0]["function"]["arguments"]
    envelope["choices"][0]["finish_reason"] = "stop"
    return envelope


def known_parent(directory, bundle):
    _, cases, trials, items = runner.parent.compat.prepare_e1(bundle)
    directory.mkdir(parents=True)
    write_json(directory / "campaign.json", {"synthetic_fixture": True})
    write_json(directory / "preflight.json", {"synthetic_fixture": True})
    write_json(directory / "e1/inventory.json", items)
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3", campaign_id=runner.PARENT_CAMPAIGN,
                                      freeze_seal=runner.PARENT_EXECUTION_SEAL)
    ledger.register_stage("e1", [{key: item[key] for key in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.49828382", max_requests=204)
    rows = [original.legacy.generation_row(item) for item in items]
    for index, item in enumerate(items[:30]):
        trial = trials[item["case_id"], item["arm"]]
        envelope = json_body(trial["wire"])
        envelope["usage"]["cost"] = "0.0001" if index < 29 else "0.00130566"
        error = None
        if index < 29:
            envelope["has_hidden_reasoning"] = False
            rows[index].update(runner.parent.compat.parse_generation(envelope, trial, cases[item["case_id"]]))
        else:
            envelope.pop("choices")
            error = "invalid_choices"
            rows[index].update(status="generation_failed", reason=error)
        write_json(directory / "e1" / (item["id"] + ".request.json"), trial["wire"])
        digest = original.legacy.write_private(directory / "e1" / (item["id"] + ".receipt.json"),
            {"envelope": envelope, "error": error, "row": rows[index]})
        ledger.claim("e1", item["id"])
        if error:
            with pytest.raises(t71_policy.LedgerError):
                original.legacy.finish_receipt(ledger, "e1", item["id"], envelope, digest, error)
        else:
            original.legacy.finish_receipt(ledger, "e1", item["id"], envelope, digest)
    write_json(directory / "e1/results.private.json", {"denominator": 204, "status": "incomplete",
        "stop_reason": "external_error", "rows": rows, "campaign": ledger.summary()})
    write_json(directory / "outcome.aggregate.json", {"synthetic_fixture": True,
        "new_known_cost_usd": runner.PARENT_KNOWN_COST})
    assert len(artifacts(directory)) == 66


@pytest.fixture
def json_history(carrier_history):
    bundle, directories = carrier_history
    directory = bundle["root"] / "artifacts/local" / runner.PARENT_CAMPAIGN
    known_parent(directory, bundle)
    write_json(bundle["root"] / runner.REGISTRATION,
               runner.expected_registration(bundle, artifacts=artifacts(directory)))
    return bundle, [*directories, directory]


def patch_cli(monkeypatch, bundle, *, documents=None):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(original, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    calls, opener = opener_for(documents if documents is not None else metadata(bundle))
    preflight = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda bundle, online=False, **kwargs:
                        preflight(bundle, online=online, opener=opener))
    return calls


def test_four_stopped_histories_are_read_only_and_full_known_fee_is_carried(json_history):
    bundle, directories = json_history
    before = [tree_hashes(path) for path in directories]
    state = runner.verify_history(bundle)
    assert state["known_cost_usd"] == "0.00592184" and state["unknown_cost_count"] == 2
    assert state["unaccepted_unknown_attempts"] == [] and state["total_actual_cost_usd"] is None
    assert sum(row["files_verified"] for row in state["campaigns"]) == 111
    assert state["campaigns"][-1]["stage_counts"] == {
        "completed": 29, "failed": 1, "started": 0, "not_started": 174}
    assert [tree_hashes(path) for path in directories] == before


@pytest.mark.parametrize("index,relative", [(0, "campaign.json"), (1, "e1/e1_001.receipt.json"),
    (2, "e1/e1_012.receipt.json"), (3, "e1/e1_030.receipt.json")])
def test_each_prior_attempt_is_bound_before_get_or_credential(json_history, monkeypatch, capsys, index, relative):
    bundle, directories = json_history
    calls = patch_cli(monkeypatch, bundle)
    with (directories[index] / relative).open("ab") as stream:
        stream.write(b"\n")
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert "changed" in json.loads(capsys.readouterr().out)["reason"]
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("change", ["unknown", "reopened", "outstanding", "fee"])
def test_prior_state_changes_rejected_even_after_refreshing_artifact_hashes(json_history, change):
    bundle, directories = json_history
    directory = directories[-1]
    with sqlite3.connect(directory / "campaign.sqlite3") as conn:
        if change == "unknown":
            conn.execute("UPDATE items SET cost_usd=NULL WHERE id='e1_030'")
        elif change == "reopened":
            conn.execute("UPDATE campaign SET stopped=0,stop_reason=NULL")
        elif change == "outstanding":
            conn.execute("UPDATE items SET status='started',claim_token=? WHERE id='e1_031'", ("c" * 64,))
        else:
            conn.execute("UPDATE items SET cost_usd='0.0001' WHERE id='e1_030'")
    registration = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    registration["prior_known_campaign"]["artifacts"] = artifacts(directory)
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    with pytest.raises((runner.RunError, t71_policy.LedgerError)):
        runner.verify_history(bundle)


@pytest.mark.parametrize("change", ["budget", "third_unknown", "retry", "carry", "wire", "measurement"])
def test_registration_cannot_expand_budget_or_select_old_results(json_history, change):
    bundle, _ = json_history
    value = original.legacy.read_json(bundle["root"] / runner.REGISTRATION)
    if change == "budget":
        value["budget_usd"]["e1"] = "0.6"
    elif change == "third_unknown":
        value["accepted_unknown_attempts"].append([runner.PARENT_CAMPAIGN, "e1_030"])
    elif change == "retry":
        value["authorization"]["permits_automatic_retry"] = True
    elif change == "carry":
        value["cumulative_prior_known_e1_cost_usd"] = "0"
    elif change == "wire":
        value["wire_changes_from_t75"]["all_non_format_messages_unchanged"] = False
    else:
        value["measurement"] = "reuse_successes_only"
    write_json(bundle["root"] / runner.REGISTRATION, value)
    with pytest.raises(runner.RunError, match="json_mode_registration_changed"):
        runner.verify_history(bundle)


def test_new_campaign_and_offline_default_use_cleaned_project_without_key_or_mutation(json_history, monkeypatch, capsys):
    bundle, directories = json_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    directory = runner.campaign_directory(bundle["root"])
    assert directory == bundle["root"].absolute() / "artifacts/local/e1_t76_20261001_v1"
    assert runner.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert calls == [] and report["ready_to_send"] is False
    assert report["credentials_read"] is False and report["model_requests"] == 0
    assert not directory.exists() and [tree_hashes(path) for path in directories] == before


def test_explicit_local_permission_flag_is_required_before_public_get(json_history, monkeypatch, capsys):
    bundle, _ = json_history
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"
    assert calls == [] and not runner.campaign_directory(bundle["root"]).exists()


def test_live_preflight_checks_new_65_wires_and_shared_whole_stage_budget(json_history, monkeypatch, capsys):
    bundle, _ = json_history
    calls = patch_cli(monkeypatch, bundle)
    assert runner.main(["--online-preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is True
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["remaining_e1_budget_usd"] == "0.49407816"
    assert report["combined_e1_reserve_usd"] == "0.23032184"
    assert report["response_policy_id"] == runner.compat.POLICY_ID
    assert report["credentials_read"] is False and not runner.campaign_directory(bundle["root"]).exists()


def test_persisted_stage_cap_and_budget_gate_subtract_all_previous_known_fees(json_history, monkeypatch, capsys, tmp_path):
    bundle, _ = json_history
    items = runner.compat.prepare_e1(bundle)[3]
    ledger = runner.CarryLedger(tmp_path / "budget.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="a" * 64)
    ledger.register_stage("e1", [{key: item[key] for key in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    assert ledger.summary()["stages"]["e1"]["limit_usd"] == "0.49407816"
    patch_cli(monkeypatch, bundle)
    assess = runner.compat.preflight

    def excessive(*args, **kwargs):
        report = assess(*args, **kwargs)
        report["planned_full_e1_reserve_usd"] = "0.4941"
        return report

    monkeypatch.setattr(runner.compat, "preflight", excessive)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blocking_reasons"] == ["shared_e1_reservation_failed"]
    assert report["combined_e1_reserve_usd"] == "0.50002184"
    assert not runner.campaign_directory(bundle["root"]).exists()


@pytest.mark.parametrize("stage,limit,maximum", [("judge", "0.5", 204), ("e1", "0.6", 204), ("e1", "0.5", 205)])
def test_ledger_stage_scope_cannot_expand(tmp_path, stage, limit, maximum):
    ledger = runner.CarryLedger(tmp_path / "scope.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="a" * 64)
    with pytest.raises(runner.RunError, match="json_mode_stage_scope_changed"):
        ledger.register_stage(stage, [{"id": "e1_001", "request_sha256": "b" * 64,
            "reserve_usd": "0.0011"}], limit_usd=limit, max_requests=maximum)


def test_missing_response_format_capability_remains_a_gate_before_key(json_history, monkeypatch, capsys):
    bundle, _ = json_history
    documents = metadata(bundle)
    endpoint = documents[runner.compat.previous.choices.ENDPOINT]["data"]["endpoints"][0]
    endpoint["supported_parameters"] = [name for name in endpoint["supported_parameters"] if name != "response_format"]
    calls = patch_cli(monkeypatch, bundle, documents=documents)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and report["ready_to_send"] is False
    assert not runner.campaign_directory(bundle["root"]).exists()


def patch_execution(monkeypatch, bundle, *, fault=None, after_post=None):
    keys, posts = [], []
    planned = runner.compat.prepare_e1(bundle)[3]
    generation = runner.compat.run_generation
    monkeypatch.setattr(runner.compat, "run_generation", lambda *args, **kwargs:
                        generation(*args, **kwargs, sleep=lambda _: None))

    def credential(root):
        assert root == bundle["root"]
        keys.append(True)
        return SYNTHETIC_KEY

    def invented_http(request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t76_transport.ENDPOINT
        directory = runner.campaign_directory(bundle["root"])
        ledger = runner.parent.parent._ReadOnlyLedger(directory / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64)
        assert ledger.summary()["outstanding_count"] == 1
        assert ledger.summary()["stages"]["e1"]["limit_usd"] == "0.49407816"
        item = planned[len(posts)]
        payload = json.loads(request.data)
        assert (directory / "e1" / (item["id"] + ".request.json")).is_file()
        assert runner.compat.canonical_sha256(payload) == item["request_sha256"]
        assert payload["response_format"] == {"type": "json_object"}
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["provider"]["data_collection"] == "deny"
        posts.append(payload)
        if fault == "http":
            raise HTTPError(request.full_url, 404, "private reason", {},
                            io.BytesIO(b'{"error":{"code":404,"message":"private sentinel"}}'))
        response = json_body(payload)
        if fault == "unknown":
            response["usage"].pop("cost")
        elif fault == "schema":
            response["choices"][0]["message"]["content"] = "synthetic invalid prose"
        if after_post is not None:
            after_post(len(posts))
        return Response(response)

    def transport(key, *, enable_network):
        assert key == SYNTHETIC_KEY and enable_network is True
        return t76_transport.OpenRouterTransport(key, opener=invented_http)

    monkeypatch.setattr(runner.parent.parent.old_run, "credential", credential)
    monkeypatch.setattr(runner, "OpenRouterTransport", transport)
    return keys, posts


@pytest.mark.parametrize("fault,unknown", [("http", 1), ("unknown", 1), ("schema", 0)])
def test_claim_then_single_failure_is_persistent_and_cannot_restart(json_history, monkeypatch, capsys, fault, unknown):
    bundle, directories = json_history
    before = [tree_hashes(path) for path in directories]
    calls = patch_cli(monkeypatch, bundle)
    keys, posts = patch_execution(monkeypatch, bundle, fault=fault)
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    outcome = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(keys) == len(posts) == 1
    assert outcome["new_unknown_cost_count"] == unknown and outcome["historical_unknown_cost_count"] == 2
    assert outcome["stage_counts"] == {"completed": 0, "failed": 1, "started": 0, "not_started": 203}
    assert Decimal(outcome["cumulative_known_e1_cost_usd"]) == Decimal(runner.CARRY_KNOWN_COST) + (
        Decimal("0.0001") if fault == "schema" else Decimal(0))
    assert outcome["response_carriers"] == ["content_json"] and outcome["total_actual_cost_usd"] is None
    directory = runner.campaign_directory(bundle["root"])
    marker = original.legacy.read_json(directory / "campaign.json")
    assert marker["prior_known_e1_cost_usd"] == runner.CARRY_KNOWN_COST
    assert marker["remaining_e1_budget_usd"] == "0.49407816"
    assert marker["parent_campaign_ids"] == [*runner.parent.parent.HISTORY_IDS,
        runner.parent.PARENT_CAMPAIGN, runner.PARENT_CAMPAIGN]
    result = original.legacy.read_json(directory / "e1/results.private.json")
    assert len(result["rows"]) == 204 and all(row["status"] == "not_started" for row in result["rows"][1:])
    assert [tree_hashes(path) for path in directories] == before
    assert runner.main(["--execute", "--allow-local-env"]) == 2
    assert "new_campaign_already_exists" in json.loads(capsys.readouterr().out)["blocking_reasons"]
    assert len(keys) == len(posts) == 1 and [tree_hashes(path) for path in directories] == before


def test_history_changed_after_first_post_stops_before_second_and_keeps_receipt(json_history, monkeypatch, capsys):
    bundle, directories = json_history
    patch_cli(monkeypatch, bundle)

    def mutate(count):
        if count == 1:
            with (directories[-1] / "e1/e1_030.receipt.json").open("ab") as stream:
                stream.write(b"\n")

    keys, posts = patch_execution(monkeypatch, bundle, after_post=mutate)
    assert runner.main(["--execute", "--allow-local-env"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "json_mode_historical_bytes_changed"
    assert len(keys) == len(posts) == 1
    directory = runner.campaign_directory(bundle["root"])
    assert (directory / "e1/e1_001.receipt.json").is_file()
    state = runner.parent.parent._ReadOnlyLedger(directory / "campaign.sqlite3", runner.CAMPAIGN_ID, "a" * 64).summary()
    assert state["stopped"] and state["known_cost_usd"] == "0.0001"
    assert state["stages"]["e1"]["counts"]["not_started"] == 203


def sealed_fixture(json_history, monkeypatch):
    bundle, directories = json_history
    monkeypatch.setattr(runner.parent, "verify_execution", lambda _: {"seal_sha256": runner.PARENT_EXECUTION_SEAL})
    for relative in runner.FILES - {runner.REGISTRATION, runner.INVENTORY}:
        destination = bundle["root"] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / relative, destination)
    items = runner.compat.prepare_e1(bundle)[3]
    write_json(bundle["root"] / runner.INVENTORY,
               {"items": items, "items_sha256": runner.compat.canonical_sha256(items)})
    seal = {"schema": "aria_e1_json_mode_execution_v1", "parent_execution_seal": runner.PARENT_EXECUTION_SEAL,
        "files": [{"path": relative, "bytes": (bundle["root"] / relative).stat().st_size,
            "sha256": hashlib.sha256((bundle["root"] / relative).read_bytes()).hexdigest()}
            for relative in sorted(runner.FILES)]}
    seal["seal_sha256"] = runner.compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    return bundle, directories, seal


def test_seal_binds_parent_and_all_nine_new_source_files(json_history, monkeypatch):
    bundle, _, seal = sealed_fixture(json_history, monkeypatch)
    assert runner.verify_execution(bundle) == seal
    assert len(seal["files"]) == 9
    monkeypatch.setattr(runner.parent, "verify_execution", lambda _: {"seal_sha256": "f" * 64})
    with pytest.raises(runner.RunError, match="json_mode_parent_seal_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("relative", sorted(runner.FILES))
def test_each_new_source_or_inventory_is_sealed(json_history, monkeypatch, relative):
    bundle, _, _ = sealed_fixture(json_history, monkeypatch)
    with (bundle["root"] / relative).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(runner.RunError, match="json_mode_source_changed"):
        runner.verify_execution(bundle)


def test_inventory_coordinates_cannot_change_even_with_rehashed_seal(json_history, monkeypatch):
    bundle, _, seal = sealed_fixture(json_history, monkeypatch)
    inventory = original.legacy.read_json(bundle["root"] / runner.INVENTORY)
    inventory["items"][1]["arm"] = "B0"
    inventory["items_sha256"] = runner.compat.canonical_sha256(inventory["items"])
    write_json(bundle["root"] / runner.INVENTORY, inventory)
    for row in seal["files"]:
        if row["path"] == runner.INVENTORY:
            path = bundle["root"] / runner.INVENTORY
            row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    seal.pop("seal_sha256")
    seal["seal_sha256"] = runner.compat.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    with pytest.raises(runner.RunError, match="json_mode_inventory_changed"):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("argument", ["--retry", "--ignore-unknown-cost", "--budget", "--key"])
def test_no_retry_budget_cost_or_direct_key_override(argument):
    with pytest.raises(SystemExit) as caught:
        runner.main([argument])
    assert caught.value.code == 2
