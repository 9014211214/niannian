"""Execution qualification and credential boundaries; all calls are local."""
from datetime import datetime, timezone, timedelta
import json
import io
import urllib.request

import pytest

from scripts import run_ctest as run
from _reservation_fixtures import fixture_reservation_policy
from aria.bench.ctest_runner import CTestLocalMemoryFixture


def test_fixed_denominators_budget_and_component_scope():
    value = run.protocol()
    assert value["denominators"] == {"people": 12, "change": 6, "control": 6, "sessions": 96,
                                      "memory_calls": 96, "online_calls": 0, "reflection_calls": 0}
    assert value["decision"]["change_min_triggers"] == 4
    assert value["decision"]["control_false_triggers"] == 0
    assert value["budget"]["effective_new_allowance_usd"] == "1.973280828"
    assert value["budget"]["prior_accepted_unknown_requests"] == 2


def test_target_key_only_no_env_mutation(tmp_path, monkeypatch):
    monkeypatch.delenv("UNRELATED_SETTING", raising=False)
    path = tmp_path / "input.env"
    path.write_text("UNRELATED_SETTING=ignored\nOPENROUTER_API_KEY='local-unit-fixture-value'\n", encoding="utf-8")
    assert run.local_key(path) == "local-unit-fixture-value"
    assert "UNRELATED_SETTING" not in run.os.environ
    path.write_text("OPENROUTER_API_KEY=one\nOPENROUTER_API_KEY=two\n", encoding="utf-8")
    with pytest.raises(run.RunError, match="missing_or_invalid"):
        run.local_key(path)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(run, "inventory", lambda _root: [{"path": "source.py", "sha256": "original"}])
    monkeypatch.setattr(run, "metadata_evidence", lambda _path: {"policy": {}, "files": []})
    registration = tmp_path / "registration.json"
    value = run.prepare(root, tmp_path, registration)
    return root, registration, value


def test_freeze_verify_without_key_access(frozen, monkeypatch):
    root, registration, expected = frozen
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("offline cannot read key"))
    assert run.verify(root, root, registration) == expected
    with pytest.raises(FileExistsError):
        run.prepare(root, root, registration)


@pytest.mark.parametrize("part", ["source", "protocol", "metadata", "seal", "expiry"])
def test_changes_block_before_key(frozen, monkeypatch, part):
    root, registration, value = frozen
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("must not read key"))
    if part == "source":
        monkeypatch.setattr(run, "inventory", lambda _: [])
    elif part == "metadata":
        monkeypatch.setattr(run, "metadata_evidence", lambda _: {})
    elif part == "protocol":
        monkeypatch.setattr(run, "protocol", lambda: {})
    elif part == "expiry":
        value["frozen_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        value["seal_sha256"] = run.digest(run.canonical({k: v for k, v in value.items() if k != "seal_sha256"}))
        registration.write_text(json.dumps(value), encoding="utf-8")
    else:
        value["seal_sha256"] = "changed"
        registration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(run.RunError):
        run.execute(root, root, registration, root / "output", allow_local_env=True)
    assert not (root / "output").exists()


def test_explicit_permission_and_replay_marker_block_before_key(frozen, monkeypatch):
    root, registration, _ = frozen
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("must not read key"))
    with pytest.raises(run.RunError, match="permission"):
        run.execute(root, root, registration, root / "output", allow_local_env=False)
    run.write_new(registration.with_suffix(".started.json"), {"previous_attempt": True})
    with pytest.raises(FileExistsError):
        run.execute(root, root, registration, root / "different-output", allow_local_env=True)


@pytest.mark.parametrize("registration_change", ["copy", "new_freeze"])
def test_same_authorization_cannot_be_reused_with_another_registration(frozen, monkeypatch, registration_change):
    root, registration, value = frozen
    marker = root / run.AUTHORIZATION_MARKER
    marker.parent.mkdir(parents=True)
    run.write_new(marker, {"seal_sha256": value["seal_sha256"], "previous_attempt": True})
    another = registration.with_name("another-registration.json")
    if registration_change == "copy":
        another.write_bytes(registration.read_bytes())
    else:
        run.prepare(root, root, another)
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("must not read key"))
    with pytest.raises(FileExistsError):
        run.execute(root, root, another, root / "new-ledger", allow_local_env=True)
    assert not (root / "new-ledger").exists()


def test_public_metadata_requires_fresh_hash_bound_observation(tmp_path):
    now = datetime.now(timezone.utc)
    records = {}
    for key, name in zip(("models", "qwen-endpoints", "sonnet-endpoints"), run.METADATA):
        raw = b"{}"
        (tmp_path / name).write_bytes(raw)
        records[key] = {"sha256": run.digest(raw), "observed_at": (now - timedelta(days=2)).isoformat()}
    (tmp_path / "public-model-metadata-v2.json").write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(run.RunError, match="expired"):
        run.metadata_evidence(tmp_path)


def test_metadata_observation_original_and_source_are_frozen(tmp_path, monkeypatch):
    records = {}
    for key, name in zip(run.METADATA_URLS, run.METADATA):
        raw = b"{}"
        (tmp_path / name).write_bytes(raw)
        records[key] = {"sha256": run.digest(raw), "observed_at": datetime.now(timezone.utc).isoformat(),
                        "url": run.METADATA_URLS[key], "selected": {"fixture": True}}
    observation = tmp_path / run.OBSERVATIONS
    observation.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(run, "derive_policy", lambda *raws: ({}, []))
    original = run.metadata_evidence(tmp_path)
    assert original["observation_file"]["sha256"] == run.digest(observation.read_bytes())
    records["models"]["selected"] = {"fixture": "changed"}
    observation.write_text(json.dumps(records), encoding="utf-8")
    assert original != run.metadata_evidence(tmp_path)
    records["models"]["url"] = "https://unverified.example/models"
    observation.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(run.RunError, match="source_mismatch"):
        run.metadata_evidence(tmp_path)
    records["models"]["url"] = run.METADATA_URLS["models"]
    records["models"]["sha256"] = "changed"
    observation.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(run.RunError, match="observation_mismatch"):
        run.metadata_evidence(tmp_path)


def test_explicit_client_configuration_overrides_ambient_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MEMORY_MAX_COMPLETION_TOKENS", "9000")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://invalid.local")
    monkeypatch.setenv("OPENROUTER_MEMORY_RETRY", "8")
    client = run.make_client("local-unit-fixture-value")
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.max_completion_tokens == 1800
    assert client.bounded_transport and client.max_retries == 0
    assert client.fallback_models == []


@pytest.mark.parametrize("response_kind", ["complete", "unknown_cost", "timeout", "accounting_unavailable"])
def test_full_execute_actual_transport_with_local_receipts(tmp_path, monkeypatch, response_kind):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text("OPENROUTER_API_KEY=local-execution-test-fixture\n", encoding="utf-8")
    monkeypatch.setattr(run, "inventory", lambda _: [])
    monkeypatch.setattr(run, "metadata_evidence", lambda _: {"policy": fixture_reservation_policy()})
    fixture = CTestLocalMemoryFixture()
    calls = []
    if response_kind == "accounting_unavailable":
        original_snapshot = run.ReservedCallGuard.snapshot

        def unreadable_snapshot(guard):
            state = original_snapshot(guard)
            if len(calls) == 96:
                state.update(calls_used=None, known_cost_usd=None, unknown_cost_requests=None,
                             stopped=True, stop_reason="ledger_unavailable")
            return state

        monkeypatch.setattr(run.ReservedCallGuard, "snapshot", unreadable_snapshot)

    class Response:
        status = 200
        headers = {}

        def __init__(self, raw):
            self.stream = io.BytesIO(raw)

        def read(self, size=-1):
            return self.stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        assert payload["provider"]["only"] == ["anthropic"]
        assert payload["provider"]["require_parameters"] is True
        assert payload["max_tokens"] == 1800 and "temperature" not in payload
        if response_kind == "timeout":
            raise TimeoutError("local test transport timeout")
        content = fixture.complete(payload["messages"])
        usage = {"prompt_tokens": 100, "completion_tokens": 100}
        if response_kind in {"complete", "accounting_unavailable"}:
            usage["cost"] = .0001
        return Response(json.dumps({"id": "gen-ctestfixture", "model": payload["model"],
                                    "provider": "Anthropic", "usage": usage,
                                    "choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", receive)
    registration = tmp_path / "freeze.json"
    run.prepare(root, root, registration)
    output = tmp_path / "execution"
    result = run.execute(root, root, registration, output, allow_local_env=True)
    expected_calls = 96 if response_kind in {"complete", "accounting_unavailable"} else 1
    assert len(calls) == expected_calls
    assert result["calls_used"] == (None if response_kind == "accounting_unavailable" else expected_calls)
    if response_kind == "complete":
        assert result["known_cost_usd"] == "0.0096" and result["unknown_cost_requests"] == 0
        assert result["real_model_gate"] == "pass"  # injected responses, not real effect evidence
    elif response_kind == "accounting_unavailable":
        assert result["unknown_cost_requests"] is None and result["known_cost_usd"] is None
        assert result["real_model_gate"] == "not_evaluable"
        assert result["failure_type"] == "AccountingUnavailable"
    else:
        assert result["unknown_cost_requests"] == 1
        assert result["real_model_gate"] == "not_evaluable"
    receipt = json.loads((output / "execution-receipt.json").read_text())
    assert receipt["old_unknown_requests_retained"] == 2
    assert (root / run.AUTHORIZATION_MARKER).is_file()
    with pytest.raises(FileExistsError):
        run.execute(root, root, registration, tmp_path / "another-output", allow_local_env=True)
    copied_registration = tmp_path / "copied-freeze.json"
    copied_registration.write_bytes(registration.read_bytes())
    with pytest.raises(FileExistsError):
        run.execute(root, root, copied_registration, tmp_path / "fresh-ledger", allow_local_env=True)
    assert len(calls) == expected_calls
