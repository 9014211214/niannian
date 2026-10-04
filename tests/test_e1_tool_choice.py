"""Read-only value-level capability tests; only public-shaped local fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_e1_tool_choice as check
from scripts import run_e1_t71 as previous, t72_policy, t73_policy


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("real_network_or_credentials_forbidden")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


@pytest.fixture(scope="module")
def prepared(bundle):
    return t73_policy.prepare_e1(bundle)


@pytest.fixture
def wire(prepared):
    return deepcopy(next(iter(prepared[2].values()))["wire"])


@pytest.fixture
def metadata():
    return {"data": {"id": check.MODEL, "endpoints": [{"model_id": check.MODEL,
        "tag": check.PROVIDER, "provider_name": "Alibaba",
        "name": "Alibaba | " + check.CANONICAL_MODEL,
        "supported_parameters": ["tools", "tool_choice"],
        "supports_tool_choice": {"none": True, "auto": True, "required": False,
                                 "function": False}}]}}


def test_parameter_name_support_cannot_hide_function_mode_false(wire, metadata):
    original = deepcopy((wire, metadata))
    result = check.assess_tool_choice(wire, metadata)
    assert result["required_mode"] == "function" and result["advertised_support"] is False
    assert result["blocking_reason"] == "tool_choice_mode_not_supported"
    assert not result["supported"] and not result["ready_to_send"]
    assert result["actual_payload_sha256"] == previous.freeze.canonical_sha256(wire)
    assert (wire, metadata) == original


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", {}, []])
def test_missing_or_nonboolean_mode_support_is_unverified(wire, metadata, value):
    metadata["data"]["endpoints"][0]["supports_tool_choice"]["function"] = value
    result = check.assess_tool_choice(wire, metadata)
    assert not result["supported"] and result["advertised_support"] is None
    assert result["blocking_reason"] == "tool_choice_mode_support_unverified"


@pytest.mark.parametrize("change", ["missing_map", "missing_mode", "missing_parameter", "missing_tools"])
def test_absent_required_evidence_blocks_even_if_other_modes_work(wire, metadata, change):
    endpoint = metadata["data"]["endpoints"][0]
    if change == "missing_map":
        del endpoint["supports_tool_choice"]
    elif change == "missing_mode":
        del endpoint["supports_tool_choice"]["function"]
    else:
        endpoint["supported_parameters"].remove("tool_choice" if change == "missing_parameter" else "tools")
    assert not check.assess_tool_choice(wire, metadata)["supported"]


@pytest.mark.parametrize("change", ["document_id", "endpoint_id", "provider_tag", "provider_name", "duplicate", "canonical"])
def test_endpoint_identity_and_provider_cannot_drift(wire, metadata, change):
    endpoint = metadata["data"]["endpoints"][0]
    endpoint["supports_tool_choice"]["function"] = True
    if change == "document_id":
        metadata["data"]["id"] = "other/model"
    elif change == "endpoint_id":
        endpoint["model_id"] = "other/model"
    elif change == "provider_tag":
        endpoint["tag"] = "other"
    elif change == "provider_name":
        endpoint["provider_name"] = "Other"
    elif change == "canonical":
        endpoint["name"] = "Alibaba | qwen/qwen3.7-flash-20990101"
    else:
        metadata["data"]["endpoints"].append(deepcopy(endpoint))
    result = check.assess_tool_choice(wire, metadata)
    assert not result["supported"] and result["blocking_reason"] == "endpoint_identity_unverified"


@pytest.mark.parametrize("value", [None, [], "alibaba", 0])
def test_invalid_provider_object_returns_closed_request_diagnosis(wire, metadata, value):
    wire["provider"] = value
    result = check.assess_tool_choice(wire, metadata)
    assert not result["supported"] and result["blocking_reason"] == "invalid_tool_choice_request"


@pytest.mark.parametrize("mode", ["function", "auto", "none", "required"])
def test_only_exact_requested_mode_true_is_capability_evidence(wire, metadata, mode):
    if mode != "function":
        wire["tool_choice"] = mode
    metadata["data"]["endpoints"][0]["supports_tool_choice"] = {mode: True}
    result = check.assess_tool_choice(wire, metadata)
    assert result["supported"] and result["advertised_support"] is True
    assert result["required_mode"] == mode and result["blocking_reason"] is None
    assert result["ready_to_send"] is False


def test_all_204_fixed_requests_block_without_changing_the_inventory(bundle, prepared, metadata):
    before = deepcopy(prepared[3])
    report = check.check_inventory(bundle, metadata)
    assert report["fixed_positions"] == report["blocked_positions"] == 204
    assert report["distinct_payloads_checked"] == 65 and report["supported_positions"] == 0
    assert report["requested_modes"] == ["function"]
    assert report["blocking_reason_counts"] == {"tool_choice_mode_not_supported": 204}
    assert report["wire_changes"] == [] and report["ready_to_send"] is False
    assert not report["historical_http_cause_confirmed"] and not report["credentials_read"]
    assert t73_policy.prepare_e1(bundle)[3] == before


def test_bounded_local_public_metadata_round_trip(tmp_path, metadata):
    path = tmp_path / "endpoint.json"
    raw = json.dumps(metadata).encode()
    path.write_bytes(raw)
    document, digest = check.read_metadata_file(path)
    assert document == metadata and digest == hashlib.sha256(raw).hexdigest()


def test_env_path_is_rejected_before_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("file opened"))
    with pytest.raises(check.CheckError, match="metadata_json_file_required"):
        check.read_metadata_file(tmp_path / ".env")


def test_local_metadata_link_and_oversize_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / "endpoint.json"
    path.write_bytes(b"x" * (check.MAX_METADATA_BYTES + 1))
    with pytest.raises(check.CheckError, match="metadata_file_unavailable"):
        check.read_metadata_file(path)
    is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == path or is_symlink(p))
    with pytest.raises(check.CheckError, match="metadata_link_rejected"):
        check.read_metadata_file(path)


def test_public_fetch_uses_exactly_one_get_without_authorization(metadata):
    calls = []

    def opener(request, *, timeout):
        assert request.get_method() == "GET" and request.data is None
        assert request.get_header("Authorization") is None and timeout == 20
        assert request.full_url == check.ENDPOINT
        calls.append(request)
        return t72_policy._StaticResponse(metadata, request.full_url)

    document, digest = check.fetch_endpoint(opener=opener)
    assert document == metadata and len(calls) == 1 and len(digest) == 64
    with pytest.raises(check.CheckError, match="network_disabled"):
        check.fetch_endpoint()


def test_cli_default_does_not_automatically_read_or_fetch_metadata(monkeypatch, capsys):
    monkeypatch.setattr(previous, "load_bundle", lambda *a: pytest.fail("unnecessary source load"))
    assert check.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "metadata_file_required" and result["model_requests"] == 0


def test_cli_saved_capabilities_block_without_printing_source_text(tmp_path, metadata, capsys):
    path = tmp_path / "endpoint.json"
    metadata["private_unused_field"] = "source-text-must-not-appear"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert check.main(["--metadata-file", str(path)]) == 2
    output = capsys.readouterr().out
    assert "source-text-must-not-appear" not in output
    assert json.loads(output)["blocked_positions"] == 204


def test_cli_does_not_offer_execution_or_credentials():
    with pytest.raises(SystemExit) as error:
        check.main(["--execute", "--allow-local-env"])
    assert error.value.code == 2
