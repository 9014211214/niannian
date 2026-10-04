"""Whole-stage compatibility and paid-entry guards, with fake public metadata."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import e1_compat as compat, run_e1_compat as runner
from scripts import run_e1_t71 as previous, t7_transport, t72_policy

REAL_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_real_network_or_keys(monkeypatch):
    import urllib.request
    def forbidden(*args, **kwargs):
        pytest.fail("real_network_or_credential_read")
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(runner.old_run, "credential", forbidden)
    read_text, read_bytes = Path.read_text, Path.read_bytes
    def safe_text(path, *args, **kwargs):
        if path.absolute() == REAL_ROOT / ".env":
            forbidden()
        return read_text(path, *args, **kwargs)
    def safe_bytes(path, *args, **kwargs):
        if path.absolute() == REAL_ROOT / ".env":
            forbidden()
        return read_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", safe_text)
    monkeypatch.setattr(Path, "read_bytes", safe_bytes)


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


def metadata(bundle):
    saved = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    documents = {t7_transport.MODELS_URL: {"data": deepcopy(saved["models"])},
                 compat.choices.ENDPOINT: {"data": deepcopy(saved["selected_endpoints"][0]["data"])}}
    documents[compat.choices.ENDPOINT]["data"]["endpoints"][0]["supports_tool_choice"] = {
        "none": True, "auto": True, "required": False, "function": False}
    # T6 didn't record this limit; this is synthetic current-capability input.
    documents[compat.choices.ENDPOINT]["data"]["endpoints"][0]["max_prompt_tokens"] = 983616
    return documents


def opener_for(documents):
    calls = []
    def open_request(request, *, timeout):
        assert request.get_method() == "GET" and request.get_header("Authorization") is None
        calls.append(request.full_url)
        return t72_policy._StaticResponse(documents[request.full_url], request.full_url)
    return calls, open_request


def test_entire_actual_inventory_qualified_with_only_two_public_gets(bundle):
    calls, opener = opener_for(metadata(bundle))
    report = compat.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2
    assert report["ready_to_send"] and report["parameter_compatibility_verified"]
    assert report["fixed_positions"] == 204 and report["distinct_payloads_checked"] == 65
    assert len(report["assessments"]) == len(report["passed_payload_sha256"]) == 65
    assert all(row["tool_choice_mode"] == "auto" for row in report["assessments"])
    assert report["planned_full_e1_reserve_usd"] == "0.2244"
    assert report["provider_privacy_filter_retained"] == "deny"
    assert not report["privacy_filter_acceptance_verified"] and not report["account_route_verified"]
    assert report["model_requests"] == 0 and not report["credentials_read"]


@pytest.mark.parametrize("mutation,reason", [
    ("mode_false", "tool_choice_mode_not_supported"),
    ("mode_missing", "tool_choice_mode_support_unverified"),
    ("mode_string", "tool_choice_mode_support_unverified"),
    ("field_missing", "parameter_support_unverified_tools"),
    ("endpoint_down", "metadata_provider_changed"),
    ("status_boolean", "metadata_provider_changed"),
    ("status_string", "metadata_provider_changed"),
    ("status_missing", "metadata_provider_changed"),
    ("canonical", "metadata_snapshot_changed"),
    ("price", "metadata_price_changed"),
    ("token_limit", "metadata_token_limit"),
    ("prompt_limit", "metadata_token_limit"),
    ("prompt_limit_missing", "metadata_prompt_limit_unverified"),
    ("prompt_limit_boolean", "metadata_prompt_limit_unverified"),
    ("prompt_limit_zero", "metadata_prompt_limit_unverified"),
    ("mandatory", "reasoning_disable_support_unverified"),
    ("mandatory_missing", "reasoning_disable_support_unverified"),
    ("mandatory_nonboolean", "reasoning_disable_support_unverified"),
])
def test_incompatible_actual_values_block_all_positions_before_credentials(bundle, mutation, reason):
    documents = metadata(bundle)
    endpoint = documents[compat.choices.ENDPOINT]["data"]["endpoints"][0]
    model = next(row for row in documents[t7_transport.MODELS_URL]["data"]
                 if row["id"] == compat.choices.MODEL)
    if mutation == "mode_false":
        endpoint["supports_tool_choice"]["auto"] = False
    elif mutation == "mode_missing":
        endpoint.pop("supports_tool_choice")
    elif mutation == "mode_string":
        endpoint["supports_tool_choice"]["auto"] = "true"
    elif mutation == "field_missing":
        endpoint["supported_parameters"].remove("tools")
    elif mutation == "endpoint_down":
        endpoint["status"] = 1
    elif mutation == "status_boolean":
        endpoint["status"] = False
    elif mutation == "status_string":
        endpoint["status"] = "0"
    elif mutation == "status_missing":
        endpoint.pop("status")
    elif mutation == "canonical":
        model["canonical_slug"] = "changed"
    elif mutation == "price":
        endpoint["pricing"]["prompt"] = "0.001"
    elif mutation == "token_limit":
        endpoint["max_completion_tokens"] = 1
    elif mutation == "prompt_limit":
        endpoint["max_prompt_tokens"] = 1
    elif mutation == "prompt_limit_missing":
        endpoint.pop("max_prompt_tokens")
    elif mutation == "prompt_limit_boolean":
        endpoint["max_prompt_tokens"] = True
    elif mutation == "prompt_limit_zero":
        endpoint["max_prompt_tokens"] = 0
    elif mutation == "mandatory":
        model["reasoning"]["mandatory"] = True
    elif mutation == "mandatory_nonboolean":
        model["reasoning"]["mandatory"] = 0
    else:
        model["reasoning"].pop("mandatory")
    calls, opener = opener_for(documents)
    report = compat.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and not report["ready_to_send"]
    assert report["blocking_reasons"] == [reason]
    assert report["passed_payload_sha256"] == [] and len(report["assessments"]) == 65


def test_saved_success_cannot_supply_a_live_gate(bundle):
    report = compat.preflight(bundle, documents=metadata(bundle))
    assert report["parameter_compatibility_verified"] and not report["ready_to_send"]
    assert report["blocking_reasons"] == ["live_metadata_not_checked"]
    with pytest.raises(compat.TransportError, match="live_gate_cannot_use_saved_documents"):
        compat.preflight(bundle, online=True, documents=metadata(bundle))


def test_offline_default_does_not_create_public_requests(bundle):
    report = compat.preflight(bundle)
    assert not report["ready_to_send"] and report["distinct_payloads_checked"] == 0
    assert report["blocking_reasons"] == ["live_metadata_not_checked"]


def test_only_two_exact_wire_changes_and_positions_survive(bundle):
    original = previous.prepare_e1(bundle)
    prepared = compat.prepare_e1(bundle)
    assert len(prepared[2]) == 68 and len(prepared[3]) == 204
    for key, trial in prepared[2].items():
        old = original[2][key]
        restored = {**trial["wire"], "parallel_tool_calls": False,
                    "tool_choice": compat.ORIGINAL_CHOICE}
        assert restored == old["wire"]
        assert trial["original_request_sha256"] == compat.canonical_sha256(old["wire"])
        assert old["wire"]["tool_choice"] == compat.ORIGINAL_CHOICE
    assert prepared[3] == compat.prepare_e1(bundle)[3]


def test_actual_launcher_keeps_new_unknown_block_before_key_or_directory(bundle, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})
    history = {"unknown_cost_count": 2, "unaccepted_unknown_attempts": [[runner.HISTORY_IDS[1], "e1_001"]]}
    monkeypatch.setattr(runner, "verify_history", lambda _: history)
    monkeypatch.setattr(runner, "campaign_directory", lambda _: tmp_path / "not-created")
    calls, opener = opener_for(metadata(bundle))
    report = runner.preflight(bundle, online=True, opener=opener)
    assert report["parameter_ready_to_send"] and not report["ready_to_send"]
    assert report["blocking_reasons"] == ["new_historical_unknown_cost_requires_resolution"]
    assert len(calls) == 2 and not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("args", [["--ignore-unknown-cost"], ["--retry"], ["--accept-unknown-cost", "2"]])
def test_no_command_can_waive_fee_or_retry(args):
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2


def test_execute_without_local_flag_never_reads_key(capsys):
    assert runner.main(["--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "local_env_permission_required"


def test_sealed_registration_can_be_checked_without_any_private_campaign(bundle, monkeypatch):
    monkeypatch.setattr(runner, "verify_history", lambda *args: pytest.fail("private history read"))
    runner.verify_execution(bundle)
