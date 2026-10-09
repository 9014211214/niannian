"""Prospective repair keeps the failed campaign and its fee; no network."""
from tests._model_http_stub import stub_model_http
from datetime import datetime, timezone, timedelta
import io
import json
import urllib.request

import pytest

from scripts import run_ctest_repair as repair
from _reservation_fixtures import fixture_reservation_policy
from aria.bench.ctest_runner import CTestLocalMemoryFixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *_a, **_k: pytest.fail("network not injected"))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(repair, "inventory", lambda _: [{"path": "source.py", "sha256": "unchanged"}])
    monkeypatch.setattr(repair, "parent_evidence", lambda _: {"calls": 40, "valid_sessions": 39,
                           "known_cost_usd": "0.400796", "outcome": "not_evaluable"})
    monkeypatch.setattr(repair.previous, "metadata_evidence", lambda _: {"policy": fixture_reservation_policy()})
    registration = tmp_path / "repair.json"
    value = repair.prepare(root, root, registration)
    return root, registration, value


def test_repair_carries_failed_cost_without_changing_denominators():
    value, prior = repair.protocol(), repair.previous.protocol()
    assert value["denominators"] == prior["denominators"]
    assert value["case_manifest"] == prior["case_manifest"]
    assert value["decision"] == prior["decision"]
    assert value["budget"]["effective_new_allowance_usd"] == "1.572484828"
    assert value["budget"]["prior_ctest_known_usd"] == "0.400796"
    assert value["experiment"] != prior["experiment"]


@pytest.mark.parametrize("change", ["parent", "source", "metadata", "expiry", "seal"])
def test_changed_evidence_blocks_before_key(frozen, monkeypatch, change):
    root, registration, value = frozen
    monkeypatch.setattr(repair.previous, "local_key", lambda _: pytest.fail("preflight must precede key"))
    if change == "parent":
        monkeypatch.setattr(repair, "parent_evidence", lambda _: {"known_cost_usd": "0"})
    elif change == "source":
        monkeypatch.setattr(repair, "inventory", lambda _: [])
    elif change == "metadata":
        monkeypatch.setattr(repair.previous, "metadata_evidence", lambda _: {})
    else:
        if change == "expiry":
            value["frozen_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            value["seal_sha256"] = repair.previous.digest(repair.previous.canonical(
                {k: v for k, v in value.items() if k != "seal_sha256"}))
        else:
            value["seal_sha256"] = "changed"
        registration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(repair.RunError):
        repair.execute(root, root, registration, root / "out", allow_local_env=True)
    assert not (root / "out").exists()


def test_explicit_permission_and_fixed_marker(frozen, monkeypatch):
    root, registration, _ = frozen
    monkeypatch.setattr(repair.previous, "local_key", lambda _: pytest.fail("must not read key"))
    with pytest.raises(repair.RunError, match="permission"):
        repair.execute(root, root, registration, root / "out", allow_local_env=False)
    marker = root / repair.MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("retained prior attempt", encoding="utf-8")
    copied = registration.with_name("copied.json")
    copied.write_bytes(registration.read_bytes())
    with pytest.raises(FileExistsError):
        repair.execute(root, root, copied, root / "different-out", allow_local_env=True)


@pytest.mark.parametrize("failure", [None, "unknown", "missing_field"])
def test_real_client_with_local_receipts_preserves_both_campaigns(frozen, monkeypatch, failure):
    root, registration, _ = frozen
    (root / ".env").write_text("OPENROUTER_API_KEY=local-repair-test-fixture\n", encoding="utf-8")
    memory, calls = CTestLocalMemoryFixture(), []

    class Response(io.BytesIO):
        status = 200
        headers = {}

    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        content = memory.complete(payload["messages"])
        if failure == "missing_field":
            content = '{"episodes":[],"relationships":[],"stated_preferences":[]}'
        usage = {"cost": .0001, "prompt_tokens": 100, "completion_tokens": 100}
        if failure == "unknown":
            usage.pop("cost")
        return Response(json.dumps({"id": "gen-fixturerepair", "model": payload["model"],
            "provider": "Anthropic", "usage": usage,
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())
    stub_model_http(monkeypatch, receive)
    output = root / "out"
    result = repair.execute(root, root, registration, output, allow_local_env=True)
    assert result["calls_used"] == len(calls) == (96 if failure is None else 1)
    assert result["real_model_gate"] == ("pass" if failure is None else "not_evaluable")
    receipt = json.loads((output / "execution-receipt.json").read_text(encoding="utf-8"))
    assert receipt["prior_ctest_known_cost_usd"] == "0.400796"
    assert receipt["old_unknown_requests_retained"] == 2
    if failure is None:
        assert receipt["ctest_total_known_cost_usd"] == "0.410396"
        assert receipt["cumulative_e1_ctest_known_usd"] == "0.437115172"
    with pytest.raises(FileExistsError):
        repair.execute(root, root, registration, root / "other", allow_local_env=True)
