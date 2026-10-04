"""T7.4 actual wire integration using invented receipts and temporary ledgers.

These checks exercise serialization and failure accounting. They do not call a
model, read credentials, or establish that tool_choice=auto will produce good
replies on a live provider.
"""
from copy import deepcopy
from decimal import Decimal
from email.message import Message
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import e1_compat as execution
from scripts import run_e1_t71 as previous
from scripts import t71_policy, t72_transport
from tests.test_e1_execution import fake_envelope
from tests.test_t7_transport import Response


GENERATION_ID = "gen-1790777949000-AbCdEfGh12345678"
TEST_CREDENTIAL = "synthetic-compatibility-credential"
FORCED_TOOL = {"type": "function", "function": {"name": "aria_turn_decision"}}


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credentials_forbidden")

    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
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
    return execution.prepare_e1(bundle)


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "synthetic-t74-campaign"
    directory.mkdir()
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="t74_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


def restored_original(payload):
    return {**deepcopy(payload), "tool_choice": deepcopy(FORCED_TOOL),
            "parallel_tool_calls": False}


class AuthoredHTTP:
    def __init__(self, ledger, items, *, fault=None):
        self.ledger, self.items, self.fault = ledger, items, fault
        self.calls = []

    def __call__(self, request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t72_transport.ENDPOINT
        assert timeout == 120
        assert self.ledger.summary()["outstanding_count"] == 1
        item = self.items[len(self.calls)]
        payload = json.loads(request.data)
        assert payload["tool_choice"] == "auto"
        assert "parallel_tool_calls" not in payload
        assert execution.canonical_sha256(payload) == item["request_sha256"]
        assert execution.canonical_sha256(restored_original(payload)) == item["original_request_sha256"]
        self.calls.append(payload)
        if self.fault == "http":
            headers = Message()
            headers["X-Generation-Id"] = GENERATION_ID
            raise HTTPError(request.full_url, 404, "private provider text", headers,
                io.BytesIO(b'{"error":{"code":404,"message":"private provider text"},"usage":{"cost":0}}'))
        if self.fault == "timeout":
            raise TimeoutError("private provider timeout")
        response = fake_envelope(payload)
        message = response["choices"][0]["message"]
        if self.fault == "multiple_tools":
            message["tool_calls"].append(deepcopy(message["tool_calls"][0]))
        elif self.fault == "wrong_tool":
            message["tool_calls"][0]["function"]["name"] = "different_tool"
        elif self.fault == "no_tools":
            message["tool_calls"] = []
        elif self.fault == "plain_text":
            message["content"] = "This is invented fixture prose."
            message.pop("tool_calls")
            response["choices"][0]["finish_reason"] = "stop"
        elif self.fault == "text_and_tool":
            message["content"] = "This is invented fixture prose."
        elif self.fault == "prompt_over":
            response["usage"]["prompt_tokens"] = item["input_token_plan_bound"] + 1
        elif self.fault == "output_over":
            response["usage"]["completion_tokens"] = payload["max_tokens"] + 1
        elif self.fault == "missing_cost":
            response["usage"].pop("cost")
        elif self.fault == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        elif self.fault == "missing_prompt_usage":
            response["usage"].pop("prompt_tokens")
        return Response(response)


def run(campaign, bundle, prepared, *, fault=None, source_check=lambda: None):
    directory, ledger = campaign
    opener = AuthoredHTTP(ledger, prepared[3], fault=fault)
    transport = t72_transport.OpenRouterTransport(TEST_CREDENTIAL, opener=opener)
    result = execution.run_generation(directory, ledger, bundle, transport,
        source_check=source_check, sleep=lambda _: None)
    return result, opener


def test_prepare_preserves_all_frozen_positions_and_only_changes_compatible_wire(bundle, prepared):
    original = previous.prepare_e1(bundle)
    assert len(prepared[3]) == len(original[3]) == 204
    assert len(prepared[2]) == len(original[2]) == 68
    assert len({item["request_sha256"] for item in prepared[3]}) == 65
    assert set(prepared[2]) == set(original[2])
    for item, old_item in zip(prepared[3], original[3]):
        assert item["original_request_sha256"] == old_item["request_sha256"]
        old_trial = original[2][item["case_id"], item["arm"]]
        trial = prepared[2][item["case_id"], item["arm"]]
        assert restored_original(trial["wire"]) == old_trial["wire"]
        for key in old_item.keys() - {"request_sha256", "input_token_plan_bound"}:
            assert item[key] == old_item[key]
        cost = t71_policy.estimate_request(trial["wire"],
            input_price="0.00000003", output_price="0.00000013")
        assert item["input_token_plan_bound"] == cost["input_token_plan_bound"]
        assert item["input_token_plan_bound"] < old_item["input_token_plan_bound"]
    assert original == previous.prepare_e1(bundle)


def test_preparation_is_repeatable_and_does_not_share_mutable_originals(bundle, prepared):
    independent = execution.prepare_e1(bundle)
    assert independent == prepared
    trial = next(iter(independent[2].values()))
    trial["wire"]["messages"][0]["content"] += "fixture mutation"
    independent[3][0]["request_sha256"] = "f" * 64
    assert execution.prepare_e1(bundle) == prepared


def test_all_204_posts_use_auto_and_link_to_original_without_polluting_legacy_globals(campaign, bundle, prepared):
    original_prepare = previous.legacy.prepare_e1
    original_globals = previous.legacy.run_generation.__globals__["prepare_e1"]
    checks = []
    result, opener = run(campaign, bundle, prepared, source_check=lambda: checks.append(True))
    directory, ledger = campaign
    assert result["status"] == "complete" and len(opener.calls) == len(checks) == 204
    assert result["denominator"] == len(result["rows"]) == 204
    assert all(row["status"] == "generated" for row in result["rows"])
    inventory = previous.legacy.read_json(directory / "e1/inventory.json")
    assert inventory == prepared[3]
    for item, sent in zip(inventory, opener.calls):
        saved = previous.legacy.read_json(directory / "e1" / (item["id"] + ".request.json"))
        assert sent == saved and execution.canonical_sha256(saved) == item["request_sha256"]
        assert saved["tool_choice"] == "auto" and "parallel_tool_calls" not in saved
        assert execution.canonical_sha256(restored_original(saved)) == item["original_request_sha256"]
    assert previous.legacy.prepare_e1 is original_prepare
    assert previous.legacy.run_generation.__globals__["prepare_e1"] is original_globals
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["unknown_cost_count"] == 0
    assert result["model_effect_measured"] is False


def test_http_failure_records_actual_wire_diagnostic_and_stops_remaining(campaign, bundle, prepared):
    result, opener = run(campaign, bundle, prepared, fault="http")
    directory, ledger = campaign
    receipt = previous.legacy.read_json(directory / "e1/e1_001.receipt.json")
    assert receipt["error"] == "http_status_error"
    assert receipt["envelope"]["http_diagnostic"]["http_status"] == 404
    assert receipt["envelope"]["http_diagnostic"]["header_generation_id"] == GENERATION_ID
    assert "usage" not in receipt["envelope"]
    saved = (directory / "e1/e1_001.receipt.json").read_text(encoding="utf-8")
    assert "private provider text" not in saved and TEST_CREDENTIAL not in saved
    assert len(opener.calls) == 1 and len(result["rows"]) == result["denominator"] == 204
    assert ledger.summary()["unknown_cost_count"] == 1
    assert ledger.summary()["known_cost_usd"] == "0"
    assert ledger.summary()["stages"]["e1"]["counts"] == {
        "failed": 1, "completed": 0, "started": 0, "not_started": 203}
    assert len(list((directory / "e1").glob("*.request.json"))) == 1
    with pytest.raises(t71_policy.LedgerError):
        ledger.claim("e1", "e1_002")


@pytest.mark.parametrize("fault,code", [
    ("multiple_tools", "invalid_schema"), ("wrong_tool", "invalid_schema"),
    ("no_tools", "invalid_schema"), ("plain_text", "invalid_schema"),
    ("text_and_tool", "invalid_schema"), ("prompt_over", "token_bound_exceeded"),
    ("output_over", "token_bound_exceeded"), ("missing_cost", "invalid_usage_cost"),
    ("truncated", "incomplete_completion"), ("missing_prompt_usage", "invalid_usage_prompt_tokens"),
])
def test_invalid_receipt_is_not_faked_or_retried(campaign, bundle, prepared, fault, code):
    result, opener = run(campaign, bundle, prepared, fault=fault)
    directory, ledger = campaign
    receipt = previous.legacy.read_json(directory / "e1/e1_001.receipt.json")
    assert receipt["error"] == code
    assert len(opener.calls) == 1 and len(result["rows"]) == 204
    assert result["rows"][0]["status"] == "generation_failed"
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    state = ledger.summary()
    assert state["stopped"] and state["stages"]["e1"]["counts"]["not_started"] == 203
    assert state["unknown_cost_count"] == (1 if fault == "missing_cost" else 0)
    assert Decimal(state["known_cost_usd"]) == (Decimal(0) if fault == "missing_cost" else Decimal("0.0001"))


def test_network_timeout_stops_and_preserves_unknown_fee(campaign, bundle, prepared):
    result, opener = run(campaign, bundle, prepared, fault="timeout")
    directory, ledger = campaign
    assert len(opener.calls) == 1 and result["status"] == "incomplete"
    assert ledger.summary()["unknown_cost_count"] == 1
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 203
    assert "private provider timeout" not in json.dumps(result)
    assert len(list((directory / "e1").glob("*.request.json"))) == 1


def test_receipt_save_failure_retains_observed_fee_without_second_post(campaign, bundle, prepared, monkeypatch):
    directory, ledger = campaign
    write = previous.legacy.write_private

    def failed_receipt(path, value):
        if str(path).endswith(".receipt.json"):
            raise OSError("private fixture path must not be reported")
        return write(path, value)

    monkeypatch.setattr(previous.legacy, "write_private", failed_receipt)
    result, opener = run(campaign, bundle, prepared)
    assert len(opener.calls) == 1 and result["status"] == "incomplete"
    state = ledger.summary()
    assert state["stopped"] and state["stop_reason"] == "result_save_failed"
    assert Decimal(state["known_cost_usd"]) == Decimal("0.0001")
    assert state["unknown_cost_count"] == 0
    assert state["stages"]["e1"]["counts"]["not_started"] == 203
    assert not list((directory / "e1").glob("*.receipt.json"))
    assert "private fixture path" not in json.dumps(result)


def test_source_failure_does_not_claim_or_post(campaign, bundle, prepared):
    def changed_source():
        raise ValueError("synthetic source fingerprint changed")

    result, opener = run(campaign, bundle, prepared, source_check=changed_source)
    directory, ledger = campaign
    assert opener.calls == [] and result["status"] == "incomplete"
    assert result["stop_reason"] == "source_or_claim_failed"
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 204
    assert ledger.summary()["unknown_cost_count"] == 0
    assert list((directory / "e1").glob("*.request.json")) == []


@pytest.mark.parametrize("change", ["old_parallel", "forced_tool", "required_tool", "text", "other_parameter"])
def test_checked_transport_rejects_unbound_wire_before_post(prepared, change):
    wire = deepcopy(next(iter(prepared[2].values()))["wire"])
    if change == "old_parallel":
        wire["parallel_tool_calls"] = False
    elif change == "forced_tool":
        wire["tool_choice"] = deepcopy(FORCED_TOOL)
    elif change == "required_tool":
        wire["tool_choice"] = "required"
    elif change == "text":
        wire["messages"][0]["content"] += "changed"
    else:
        wire["temperature"] = 0.3

    def must_not_open(*args, **kwargs):
        pytest.fail("an unbound wire reached POST")

    checked = execution.CheckedTransport(t72_transport.OpenRouterTransport(
        TEST_CREDENTIAL, opener=must_not_open),
        {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]})
    with pytest.raises(execution.TransportError, match="request_changed") as error:
        checked.request(wire)
    assert error.value.outcome_unknown is False


def test_transport_mutation_is_detected_after_send_with_observed_fee_retained(prepared):
    class MutatingTransport:
        def request(self, payload):
            response = fake_envelope(payload)
            payload["tool_choice"] = "none"
            return response

    wire = deepcopy(next(iter(prepared[2].values()))["wire"])
    original = deepcopy(wire)
    checked = execution.CheckedTransport(MutatingTransport(),
        {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]})
    with pytest.raises(execution.TransportError, match="request_changed") as error:
        checked.request(wire)
    assert error.value.receipt["usage"]["cost"] == "0.0001"
    assert wire == original


def test_bounds_are_immutable_after_binding(prepared):
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    checked = execution.CheckedTransport(None, bounds)
    digest = next(iter(bounds))
    before = checked.bounds[digest]
    bounds[digest] += 1000
    assert checked.bounds[digest] == before
    with pytest.raises(TypeError):
        checked.bounds[digest] = before + 1000


@pytest.mark.parametrize("change", ["missing", "not_ready", "not_verified", "saved", "incomplete", "unexpected"])
def test_live_qualification_must_cover_exact_actual_inventory_before_any_post(prepared, change):
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    qualification = {"ready_to_send": True, "parameter_compatibility_verified": True,
                     "passed_payload_sha256": sorted(bounds), "metadata_scope": "live_public_get"}
    if change == "missing":
        qualification = {}
    elif change == "not_ready":
        qualification["ready_to_send"] = False
    elif change == "not_verified":
        qualification["parameter_compatibility_verified"] = False
    elif change == "saved":
        qualification["metadata_scope"] = "saved_public_snapshot"
    elif change == "incomplete":
        qualification["passed_payload_sha256"].pop()
    else:
        qualification["passed_payload_sha256"].append("f" * 64)

    def must_not_open(*args, **kwargs):
        pytest.fail("an unqualified wire reached POST")

    with pytest.raises(execution.TransportError, match="actual_payload_not_qualified") as error:
        execution.CheckedTransport(t72_transport.OpenRouterTransport(
            TEST_CREDENTIAL, opener=must_not_open), bounds, qualification=qualification)
    assert error.value.outcome_unknown is False


def test_qualification_inventory_is_immutable_after_binding(prepared):
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    qualification = {"ready_to_send": True, "parameter_compatibility_verified": True,
                     "passed_payload_sha256": sorted(bounds), "metadata_scope": "live_public_get"}
    checked = execution.CheckedTransport(None, bounds, qualification=qualification)
    qualification["passed_payload_sha256"].clear()
    assert checked.qualified == frozenset(bounds)


def test_stopped_campaign_cannot_be_reopened_by_compatible_transport(campaign, bundle, prepared):
    directory, ledger = campaign
    ledger.stop("e1", "manual_stop")
    opener = AuthoredHTTP(ledger, prepared[3])
    transport = t72_transport.OpenRouterTransport(TEST_CREDENTIAL, opener=opener)
    with pytest.raises(t71_policy.LedgerError):
        execution.run_generation(directory, ledger, bundle, transport,
            source_check=lambda: None, sleep=lambda _: None)
    assert opener.calls == [] and ledger.summary()["stopped"]
