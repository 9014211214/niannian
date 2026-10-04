"""Actual T7.3 serialization with fake HTTP and temporary, authored ledgers."""
from copy import deepcopy
from decimal import Decimal
from email.message import Message
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import run_e1_t71 as previous
from scripts import t71_policy, t72_transport, t73_policy
from scripts import t73_execution as execution
from tests.test_e1_execution import fake_envelope
from tests.test_t7_transport import Response


GENERATION_ID = "gen-1790777949000-AbCdEfGh12345678"
TEST_CREDENTIAL = "synthetic-execution-credential"


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
    return t73_policy.prepare_e1(bundle)


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "synthetic-t73-campaign"
    directory.mkdir()
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3",
        campaign_id="t73_software_fixture", freeze_seal="e" * 64)
    return directory, ledger


class AuthoredHTTP:
    def __init__(self, ledger, items, *, fault=None):
        self.ledger, self.items, self.fault = ledger, items, fault
        self.calls = []

    def __call__(self, request, *, timeout):
        assert request.get_method() == "POST" and request.full_url == t72_transport.ENDPOINT
        assert timeout == 120
        state = self.ledger.summary()
        assert state["outstanding_count"] == 1
        item = self.items[len(self.calls)]
        payload = json.loads(request.data)
        assert "parallel_tool_calls" not in payload
        assert execution.canonical_sha256(payload) == item["request_sha256"]
        assert execution.canonical_sha256({**payload, "parallel_tool_calls": False}) == item["original_request_sha256"]
        self.calls.append(payload)
        if self.fault == "http":
            headers = Message()
            headers["X-Generation-Id"] = GENERATION_ID
            raise HTTPError(request.full_url, 404, "private provider text", headers,
                io.BytesIO(b'{"error":{"code":404,"message":"private provider text"},"usage":{"cost":0}}'))
        response = fake_envelope(payload)
        if self.fault == "multiple_tools":
            calls = response["choices"][0]["message"]["tool_calls"]
            calls.append(deepcopy(calls[0]))
        elif self.fault == "wrong_tool":
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "different_tool"
        elif self.fault == "prompt_over":
            response["usage"]["prompt_tokens"] = item["input_token_plan_bound"] + 1
        elif self.fault == "output_over":
            response["usage"]["completion_tokens"] = payload["max_tokens"] + 1
        elif self.fault == "missing_cost":
            response["usage"].pop("cost")
        return Response(response)


def run(campaign, bundle, prepared, *, fault=None, source_check=lambda: None):
    directory, ledger = campaign
    opener = AuthoredHTTP(ledger, prepared[3], fault=fault)
    transport = t72_transport.OpenRouterTransport(TEST_CREDENTIAL, opener=opener)
    result = execution.run_generation(directory, ledger, bundle, transport,
        source_check=source_check, sleep=lambda _: None)
    return result, opener


def test_all_204_posts_use_new_inventory_with_original_lineage(campaign, bundle, prepared):
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
        assert "parallel_tool_calls" not in saved
    assert previous.legacy.prepare_e1 is original_prepare
    assert previous.legacy.run_generation.__globals__["prepare_e1"] is original_globals
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["unknown_cost_count"] == 0
    assert result["model_effect_measured"] is False


def test_http_failure_records_new_wire_diagnostic_and_stops_all_remaining(campaign, bundle, prepared):
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


@pytest.mark.parametrize("fault,code", [("multiple_tools", "invalid_schema"),
    ("wrong_tool", "invalid_schema"), ("prompt_over", "token_bound_exceeded"),
    ("output_over", "token_bound_exceeded"), ("missing_cost", "invalid_usage_cost")])
def test_invalid_observed_receipt_stops_without_resending(campaign, bundle, prepared, fault, code):
    result, opener = run(campaign, bundle, prepared, fault=fault)
    directory, ledger = campaign
    receipt = previous.legacy.read_json(directory / "e1/e1_001.receipt.json")
    assert receipt["error"] == code
    assert len(opener.calls) == 1 and len(result["rows"]) == 204
    state = ledger.summary()
    assert state["stopped"] and state["stages"]["e1"]["counts"]["not_started"] == 203
    assert state["unknown_cost_count"] == (1 if fault == "missing_cost" else 0)
    assert Decimal(state["known_cost_usd"]) == (Decimal(0) if fault == "missing_cost" else Decimal("0.0001"))


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


@pytest.mark.parametrize("change", ["old_parallel", "text", "other_parameter"])
def test_checked_transport_rejects_unbound_wire_before_post(prepared, change):
    wire = deepcopy(next(iter(prepared[2].values()))["wire"])
    if change == "old_parallel":
        wire["parallel_tool_calls"] = False
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


def test_bounds_are_not_mutable_after_binding(prepared):
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in prepared[3]}
    checked = execution.CheckedTransport(None, bounds)
    digest = next(iter(bounds))
    before = checked.bounds[digest]
    bounds[digest] += 1000
    assert checked.bounds[digest] == before
    with pytest.raises(TypeError):
        checked.bounds[digest] = before + 1000


def test_stopped_temporary_campaign_is_never_reopened(campaign, bundle, prepared):
    directory, ledger = campaign
    ledger.stop("e1", "manual_stop")
    opener = AuthoredHTTP(ledger, prepared[3])
    transport = t72_transport.OpenRouterTransport(TEST_CREDENTIAL, opener=opener)
    with pytest.raises(t71_policy.LedgerError):
        execution.run_generation(directory, ledger, bundle, transport,
            source_check=lambda: None, sleep=lambda _: None)
    assert opener.calls == [] and ledger.summary()["stopped"]


def test_adapter_has_no_cli_or_campaign_factory():
    assert not hasattr(execution, "main")
    assert not hasattr(execution, "open_campaign")
    assert not hasattr(execution, "credential")
