"""T7.3 qualification tests never read keys, bill models, or modify real ledgers."""
from copy import deepcopy
import json
import shutil

import pytest

from scripts import check_e1_t73 as check
from scripts import check_e1_t72 as prior
from scripts import run_e1_t71 as previous
from scripts import t72_policy
from tests.test_e1_t72 import failed_campaign, no_network_or_credentials, tree_hashes  # noqa: F401


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


@pytest.fixture
def linked_campaign(failed_campaign):
    bundle, directory, ledger, result, calls = failed_campaign
    registration = previous.legacy.read_json(previous.ROOT / check.REGISTRATION)
    registration["prior_campaign"] = prior.inspect_campaign(bundle, previous.prepare_e1(bundle)[3])
    path = bundle["root"] / check.REGISTRATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registration), encoding="utf-8")
    return bundle, directory


def static_metadata(bundle, *, failure=None):
    metadata = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    calls = []

    def opener(request, *, timeout):
        assert request.get_method() == "GET" and request.get_header("Authorization") is None
        calls.append(request.full_url)
        value = ({"data": metadata["models"]} if request.full_url.endswith("/models")
                 else {"data": deepcopy(metadata["selected_endpoints"][0]["data"])})
        if failure and not request.full_url.endswith("/models"):
            value["data"]["endpoints"][0]["supported_parameters"].remove("tools")
        return t72_policy._StaticResponse(value, request.full_url)

    return calls, opener


def test_wire_metadata_repair_passes_but_unknown_cost_remains(linked_campaign):
    bundle, directory = linked_campaign
    before = tree_hashes(directory)
    calls, opener = static_metadata(bundle)
    report = check.build_report(bundle, online=True, opener=opener)
    assert len(calls) == 2
    assert report["parameter_repair_verified"] is True
    assert report["actual_required_fields"] == ["max_tokens", "reasoning", "temperature", "tool_choice", "tools"]
    assert report["missing_parameter_evidence"] == []
    assert report["blocking_reasons"] == ["prior_cost_unresolved"]
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert report["planned_full_e1_reserve_usd"] == "0.2244"
    assert report["historical_cost_carry_usd"] is None
    assert not report["ready_to_send"] and not report["combined_budget_qualified"]
    assert not report["credentials_read"] and report["model_requests"] == 0
    assert report["prior_campaign"]["unknown_cost_count"] == 1
    assert tree_hashes(directory) == before


def test_offline_report_does_not_claim_live_support(linked_campaign):
    report = check.build_report(linked_campaign[0])
    assert report["offline_parameter_evidence_sufficient"]
    assert not report["live_metadata_attempted"] and not report["parameter_repair_verified"]
    assert report["blocking_reasons"] == ["live_metadata_not_checked", "prior_cost_unresolved"]


def test_missing_remaining_parameter_fails_before_any_inference(linked_campaign):
    calls, opener = static_metadata(linked_campaign[0], failure=True)
    report = check.build_report(linked_campaign[0], online=True, opener=opener)
    assert len(calls) == 2 and report["live_metadata_attempted"]
    assert not report["parameter_repair_verified"] and not report["ready_to_send"]
    assert "live_metadata_gate_failed" in report["blocking_reasons"]


def test_replacing_historical_snapshot_does_not_unlock_new_wire(linked_campaign):
    bundle, directory = linked_campaign
    path = bundle["root"] / check.REGISTRATION
    value = json.loads(path.read_text(encoding="utf-8"))
    value["prior_campaign"]["unknown_cost_count"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")
    before = tree_hashes(directory)
    report = check.build_report(bundle)
    assert "prior_campaign_evidence_unavailable_or_invalid" in report["blocking_reasons"]
    assert not report["ready_to_send"] and tree_hashes(directory) == before


@pytest.mark.parametrize("args", [["--execute"], ["--allow-local-env"], ["--ignore-unknown-cost"]])
def test_no_cli_cost_override_or_inference_entry(args):
    with pytest.raises(SystemExit) as error:
        check.main(args)
    assert error.value.code == 2


def test_cli_reports_only_safe_aggregate(linked_campaign, monkeypatch, capsys):
    monkeypatch.setattr(previous, "load_bundle", lambda: linked_campaign[0])
    monkeypatch.setattr(check, "verify_repair", lambda _: {"seal_sha256": "a" * 64})
    assert check.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and not report["ready_to_send"]


def test_cli_suppresses_exception_contents(monkeypatch, capsys):
    def failure():
        raise ValueError("synthetic private message")

    monkeypatch.setattr(previous, "load_bundle", failure)
    assert check.main([]) == 1
    output = capsys.readouterr().out
    assert "synthetic private message" not in output


def test_all_seals_and_linked_inventory(bundle):
    assert check.verify_repair(bundle)["inherits"] == check.INHERITS


@pytest.mark.parametrize("relative", sorted(check.FILES))
def test_repair_seal_catches_each_changed_file(bundle, tmp_path, monkeypatch, relative):
    for name in check.FILES | {check.EXECUTION}:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle["root"] / name, target)
    with (tmp_path / relative).open("ab") as file:
        file.write(b"\n")
    monkeypatch.setattr(prior, "verify_repair", lambda _: {"seal_sha256": check.T72_SEAL})
    with pytest.raises(prior.QualificationError):
        check.verify_repair({**bundle, "root": tmp_path})
