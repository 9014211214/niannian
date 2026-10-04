"""Frozen E1 integration with synthetic receipts only; no keys or network."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts import run_e1_experiment as runner
from scripts.t7_ledger import CampaignLedger
from scripts.t7_transport import TransportError


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network_or_credential_access_forbidden")

    monkeypatch.setattr(runner, "credential", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)


@pytest.fixture(scope="module")
def bundle():
    return runner.load_bundle()


@pytest.fixture
def campaign(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir()
    ledger = CampaignLedger(directory / "campaign.sqlite3", campaign_id=runner.CAMPAIGN_ID, freeze_seal="e" * 64)
    return directory, ledger


def fake_envelope(payload):
    system = payload["messages"][0]["content"]
    routes = json.loads(system.split("当前可选干预：", 1)[1].split("\n", 1)[0])
    selected = routes[0]["id"]
    decision = {
        "response": "我听着。", "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": 0.1,
                           "divergence": {"possible": False, "confidence": 0.0, "pattern": ""}},
        "selected_intervention_id": selected,
        "communication_plan": {"pace": "open_listening", "support_order": "先听他说", "boundary": "不替他决定"},
        "uncertainty": 0.5,
    }
    if "online_turn_decision_v2" in system:
        decision["deliberation"] = {
            "person_read": "未知", "applied_basis": [],
            "options": [{"intervention_id": selected, "predicted_reaction": "未知"}],
        }
    return {
        "id": "synthetic_receipt", "model": "qwen/qwen3.7-flash", "provider": "alibaba",
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "cost": "0.0001"},
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "synthetic_tool", "type": "function", "function": {
                "name": "aria_turn_decision", "arguments": json.dumps(decision, ensure_ascii=False),
            }}],
        }}],
    }


class SyntheticTransport:
    def __init__(self, ledger, *, failure=None):
        self.ledger = ledger
        self.failure = failure
        self.calls = []

    def request(self, payload):
        # This observation comes from a fresh read connection, proving the
        # durable claim is visible before the injected network boundary.
        summary = self.ledger.summary()
        assert summary["outstanding_count"] == 1
        self.calls.append(deepcopy(payload))
        envelope = fake_envelope(payload)
        if self.failure == "timeout":
            raise TransportError("response_timeout")
        if self.failure == "unknown_cost":
            envelope["usage"].pop("cost")
            raise TransportError("invalid_usage_cost", receipt=envelope)
        if self.failure == "known_cost_invalid_tokens":
            envelope["usage"].pop("prompt_tokens")
            raise TransportError("invalid_usage_prompt_tokens", receipt=envelope)
        return envelope


def run(directory, ledger, bundle, transport, *, source_check=lambda: None):
    return runner.run_generation(directory, ledger, bundle, transport, source_check=source_check, sleep=lambda _: None)


def test_complete_204_frozen_positions_are_generated_and_journaled(campaign, bundle):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)
    checks = []
    result = run(directory, ledger, bundle, transport, source_check=lambda: checks.append(True))
    assert len(transport.calls) == len(checks) == result["denominator"] == len(result["rows"]) == 204
    assert result["status"] == "complete"
    assert result["model_effect_measured"] is False
    assert all(row["status"] == "generated" for row in result["rows"])
    assert all(row["raw_decision"]["response"] == "我听着。" for row in result["rows"])
    assert all(row["deliberation_status"] == "captured" for row in result["rows"] if row["arm"] in {"B2", "B3a", "B3b"})
    assert all(row["output_measurement"]["samples"] == 1 for row in result["rows"])
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.0204")
    assert ledger.summary()["stages"]["e1"]["counts"]["completed"] == 204
    assert len(list((directory / "e1").glob("*.request.json"))) == 204
    assert len(list((directory / "e1").glob("*.receipt.json"))) == 204
    for sent, frozen in zip(transport.calls, bundle["manifest"]["e1_requests"]):
        assert runner.canonical_sha256(sent) == frozen["request_sha256"]
    with pytest.raises((OSError, ValueError)):
        run(directory, ledger, bundle, transport)
    assert len(transport.calls) == 204


@pytest.mark.parametrize("failure", ["timeout", "unknown_cost", "known_cost_invalid_tokens"])
def test_first_failure_preserves_203_unissued_and_never_retries(campaign, bundle, failure):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger, failure=failure)
    result = run(directory, ledger, bundle, transport)
    assert len(transport.calls) == 1
    assert result["status"] == "incomplete"
    assert result["denominator"] == len(result["rows"]) == 204
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 203
    summary = ledger.summary()
    assert summary["stopped"]
    assert summary["stages"]["e1"]["counts"]["failed"] == 1
    assert summary["stages"]["e1"]["counts"]["not_started"] == 203
    assert (directory / "e1" / "results.private.json").is_file()
    if failure == "known_cost_invalid_tokens":
        assert Decimal(summary["known_cost_usd"]) == Decimal("0.0001")
        assert summary["unknown_cost_count"] == 0
    else:
        assert summary["unknown_cost_count"] == 1


def test_receipt_save_failure_stops_with_known_fee_and_no_second_call(campaign, bundle, monkeypatch):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)
    write = runner.write_private

    def failed_receipt(path, value):
        if str(path).endswith(".receipt.json"):
            raise OSError("private path deliberately not reported")
        return write(path, value)

    monkeypatch.setattr(runner, "write_private", failed_receipt)
    result = run(directory, ledger, bundle, transport)
    assert len(transport.calls) == 1
    assert result["status"] == "incomplete"
    summary = ledger.summary()
    assert summary["stopped"]
    assert summary["stop_reason"] == "result_save_failed"
    assert Decimal(summary["known_cost_usd"]) == Decimal("0.0001")
    assert summary["stages"]["e1"]["counts"]["not_started"] == 203
    assert not list((directory / "e1").glob("*.receipt.json"))
    assert "private path deliberately" not in json.dumps(result)


def test_source_check_failure_sends_nothing_and_preserves_all_positions(campaign, bundle):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)

    def changed():
        raise runner.ExecutionError("execution_source_changed")

    result = run(directory, ledger, bundle, transport, source_check=changed)
    assert transport.calls == []
    assert result["status"] == "incomplete"
    assert all(row["status"] == "not_started" for row in result["rows"])
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 204
    assert ledger.summary()["stopped"]


def test_preflight_and_default_cli_do_not_read_credential_or_open_network(bundle, monkeypatch, capsys):
    # Execution seal creation is owned by root; this tests the default branch's
    # IO boundary while load_bundle has already verified the real T6 baseline.
    monkeypatch.setattr(runner, "load_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "e" * 64})

    def forbidden(*args, **kwargs):
        raise AssertionError("default must not inspect or mutate paid campaign")

    monkeypatch.setattr(runner, "open_campaign", forbidden)
    assert runner.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_requests"] == 0
    assert report["e1_requests"] == 204
    assert report["live_metadata_checked"] is False
    assert report["provider_hard_token_bound_verified"] is False


def test_parse_generation_uses_real_v1_v2_protocol_and_isolates_trial_state(bundle):
    _, cases, prepared, _ = runner.prepare_e1(bundle)
    case_id = next(iter(cases))
    for arm in ("B0", "B2", "B3b"):
        trial = prepared[case_id, arm]
        original = deepcopy(trial["ctx"].working_memory)
        parsed = runner.parse_generation(fake_envelope(trial["wire"]), trial, cases[case_id])
        assert parsed["status"] == "generated"
        assert parsed["deliberation_status"] == ("not_applicable" if arm == "B0" else "captured")
        assert trial["ctx"].working_memory == original


def test_open_campaign_uses_fixed_id_and_rejects_partial_initialization(tmp_path, bundle):
    local = {**bundle, "root": tmp_path / "source"}
    execution = {"seal_sha256": "e" * 64}
    directory, ledger = runner.open_campaign(local, execution)
    assert directory == tmp_path / "aria-private-experiments" / runner.CAMPAIGN_ID
    assert ledger.path == (directory / "campaign.sqlite3").resolve()
    assert runner.open_campaign(local, execution)[0] == directory
    with pytest.raises(runner.ExecutionError, match="existing_campaign_requires_reconciliation"):
        runner.open_campaign(local, {"seal_sha256": "f" * 64})
    other = {**bundle, "root": tmp_path / "another" / "source"}
    failed = runner.campaign_directory(other["root"])
    failed.mkdir(parents=True)
    with pytest.raises(runner.ExecutionError, match="existing_campaign_requires_reconciliation"):
        runner.open_campaign(other, execution)


def test_private_artifacts_are_immutable(campaign):
    directory, _ = campaign
    path = directory / "evidence.json"
    digest = runner.write_private(path, {"synthetic": True})
    assert len(digest) == 64
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_private(path, {"overwritten": True})
    assert path.read_bytes() == before


def test_inventory_save_failure_retains_full_unissued_ledger(campaign, bundle, monkeypatch):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)
    write = runner.write_private

    def fail_inventory(path, value):
        if Path(path).name == "inventory.json":
            raise OSError("synthetic disk failure")
        return write(path, value)

    monkeypatch.setattr(runner, "write_private", fail_inventory)
    with pytest.raises(OSError):
        run(directory, ledger, bundle, transport)
    assert transport.calls == []
    summary = ledger.summary()
    assert summary["stopped"]
    assert summary["stop_reason"] == "result_save_failed"
    assert summary["stages"]["e1"]["counts"]["not_started"] == 204


def test_final_result_save_failure_stops_even_after_all_receipts_saved(campaign, bundle, monkeypatch):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)
    write = runner.write_private

    def fail_final(path, value):
        if Path(path).name == "results.private.json":
            raise OSError("synthetic disk failure")
        return write(path, value)

    monkeypatch.setattr(runner, "write_private", fail_final)
    with pytest.raises(OSError):
        run(directory, ledger, bundle, transport)
    assert len(transport.calls) == 204
    summary = ledger.summary()
    assert summary["stopped"]
    assert summary["stop_reason"] == "result_save_failed"
    assert summary["stages"]["e1"]["counts"]["completed"] == 204
    assert len(list((directory / "e1").glob("*.receipt.json"))) == 204


def test_payload_is_rechecked_after_preparation_before_claim(campaign, bundle, monkeypatch):
    directory, ledger = campaign
    prepared = runner.prepare_e1(bundle)
    first = prepared[3][0]
    target = prepared[2][first["case_id"], first["arm"]]
    monkeypatch.setattr(runner, "prepare_e1", lambda _: prepared)
    transport = SyntheticTransport(ledger)

    def changed_memory():
        target["wire"]["messages"][0]["content"] += "modified after preparation"

    result = run(directory, ledger, bundle, transport, source_check=changed_memory)
    assert transport.calls == []
    assert result["status"] == "incomplete"
    assert ledger.summary()["stages"]["e1"]["counts"]["not_started"] == 204


def test_execute_with_unverified_token_bound_stops_before_key_or_claim(bundle, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "e" * 64})
    monkeypatch.setattr(runner, "fetch_metadata", lambda **_: {"synthetic_public_metadata": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked gate must not open campaign")

    monkeypatch.setattr(runner, "open_campaign", forbidden)
    assert runner.main(["--execute"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked_before_credentials"
    assert report["ready_to_send"] is False
    assert "provider_input_token_bound_unverified" in report["blocking_reasons"]


def test_incomplete_execution_has_nonzero_exit_code(campaign, bundle, monkeypatch, capsys):
    # Pure command-branch fixture, not authorization or an executable gate bypass.
    monkeypatch.setattr(runner, "load_bundle", lambda: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "e" * 64})
    monkeypatch.setattr(runner, "preflight_e1", lambda *_, **__: {"ready_to_send": True})
    monkeypatch.setattr(runner, "credential", lambda: "synthetic-placeholder")
    monkeypatch.setattr(runner, "open_campaign", lambda *args: campaign)
    monkeypatch.setattr(runner, "OpenRouterTransport", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "run_generation", lambda *args, **kwargs: {
        "status": "incomplete", "rows": [], "stop_reason": "unknown_cost", "campaign": {},
    })
    assert runner.main(["--execute"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["status"] == "incomplete"


def synthetic_judge_preparation(bundle):
    """Real 204 row shapes with entirely invented software-fixture replies."""
    from scripts.t7_judge import build_batches

    suite, _, _, items = runner.prepare_e1(bundle)
    rows = [{**runner.generation_row(item), "status": "generated",
             "raw_decision": {"response": "我听着。"}, "delivered_response": "我听着。",
             "actual_model": "qwen/qwen3.7-flash", "latency_ms": 1.0,
             "deliberation_status": "captured", "selected_in_deliberated_options": True}
            for item in items]
    prepared = build_batches(rows, suite, bundle["plan"], bundle["contract"],
                             {"raw_shuffle": 11, "delivered_shuffle": 22,
                              "pair_shuffle": 33, "human_sample": 44})
    return suite, rows, prepared


def test_judge_full_real_batch_plan_is_blocked_without_a_call(campaign, bundle):
    directory, ledger = campaign
    suite, rows, prepared = synthetic_judge_preparation(bundle)
    transport = SyntheticTransport(ledger)
    assert prepared["status"] == "blocked_budget"
    assert Decimal(prepared["total_reserve_usd"]) > Decimal("0.30")
    with pytest.raises(runner.ExecutionError, match="judge_budget_insufficient"):
        runner.run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
                         source_check=lambda: None, sleep=lambda _: None)
    assert transport.calls == []
    assert ledger.summary()["stages"] == {}
    assert not (directory / "judge").exists()


@pytest.mark.parametrize("failure", [None, "invalid_labels", "apply_results", "aggregate"])
def test_synthetic_judge_one_batch_save_and_schema_failure(campaign, bundle, failure, monkeypatch):
    # Deliberately narrowed *software fixture*; never a full or paid experiment.
    directory, ledger = campaign
    suite, rows, original = synthetic_judge_preparation(bundle)
    prepared = deepcopy(original)
    prepared["batches"] = [prepared["batches"][0]]
    prepared["total_reserve_usd"] = prepared["batches"][0]["reserve_usd"]
    prepared["status"] = "prepared"
    ledger.register_stage("e1", [{"id": "fixture", "request_sha256": "1" * 64, "reserve_usd": "0.0011"}],
                          limit_usd="0.5", max_requests=204)
    ledger.claim("e1", "fixture")
    ledger.finish("e1", "fixture", cost_usd="0", prompt_tokens=1, completion_tokens=1,
                  actual_model="qwen/qwen3.7-flash", provider="alibaba", response_sha256="2" * 64)
    if failure in {"apply_results", "aggregate"}:
        from scripts import t7_judge

        def fail_processing(*args, **kwargs):
            raise ValueError("synthetic processing failure")

        monkeypatch.setattr(t7_judge, failure, fail_processing)

    class JudgeTransport:
        calls = 0

        def request(self, payload):
            self.calls += 1
            assert ledger.summary()["outstanding_count"] == 1
            batch = prepared["batches"][0]
            content = [[key, True if failure == "invalid_labels" else 0, 0, 0, 0] for key in batch["private"]["items"]]
            return {"model": "anthropic/claude-sonnet-5", "provider": "anthropic",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": "0.001"},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}]}

    transport = JudgeTransport()
    if failure in {"apply_results", "aggregate"}:
        with pytest.raises((ValueError, runner.ExecutionError)):
            runner.run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
                             source_check=lambda: None, sleep=lambda _: None)
        assert transport.calls == 1
        assert ledger.summary()["stopped"]
        assert ledger.summary()["stop_reason"] == "invalid_schema"
        assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.001")
        assert len(list((directory / "judge").glob("*.receipt.json"))) == 1
        return
    result = runner.run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
                              source_check=lambda: None, sleep=lambda _: None)
    assert transport.calls == 1
    invalid = failure == "invalid_labels"
    assert result["completed_batches"] == (0 if invalid else 1)
    assert result["status"] == ("incomplete" if invalid else "complete")
    assert ledger.summary()["stopped"] is invalid
    assert Decimal(ledger.summary()["known_cost_usd"]) == Decimal("0.001")
    assert (directory / "judge" / "results.private.json").is_file()


def test_load_generation_validates_full_204_and_rejects_three_evidence_tampers(campaign, bundle):
    directory, ledger = campaign
    transport = SyntheticTransport(ledger)
    result = run(directory, ledger, bundle, transport)
    assert len(transport.calls) == 204
    assert runner.load_generation(directory, ledger, bundle) == result["rows"]
    first = ledger.summary()["stages"]["e1"]["items"][0]["id"]
    targets = [
        (directory / "e1" / "results.private.json", "row"),
        (directory / "e1" / f"{first}.receipt.json", "receipt"),
        (directory / "e1" / f"{first}.request.json", "request"),
    ]
    for path, kind in targets:
        original = path.read_bytes()
        changed = json.loads(original)
        if kind == "row":
            changed["rows"][0]["raw_decision"]["response"] = "changed only in final result"
        elif kind == "receipt":
            changed["envelope"]["usage"]["cost"] = "0"
        else:
            changed["messages"][0]["content"] += " altered request"
        # Fault injection is confined to pytest's temporary fictional campaign;
        # actual evidence is never rewritten by the application.
        try:
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with pytest.raises(runner.ExecutionError, match="generation_evidence_changed"):
                runner.load_generation(directory, ledger, bundle)
        finally:
            path.write_bytes(original)
        assert runner.load_generation(directory, ledger, bundle) == result["rows"]
        assert len(transport.calls) == 204
    assert "judge" not in ledger.summary()["stages"]
    assert not (directory / "judge").exists()


@pytest.mark.parametrize("mutation", ["source", "seal", "inventory"])
def test_execution_revision_seal_is_independent_and_rejects_drift(tmp_path, mutation):
    # This synthetic manifest intentionally never reads the real execution.json
    # or hashes this test file, avoiding a self-referential seal assertion.
    names = set(runner.IMPLEMENTATION_FILES) | {
        "tests/test_t7_ledger.py", "tests/test_t7_transport.py", "tests/test_t7_judge.py",
        "tests/test_e1_execution.py", "docs/E1_EXECUTION.md",
        "experiments/t7_20260930/preflight.json",
    }
    assert len(names) == 10
    records = []
    for name in sorted(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source for " + name + "\n", encoding="utf-8")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": runner.file_sha256(path)})
    manifest = {"schema": "aria_t7_execution_v1", "campaign_id": runner.CAMPAIGN_ID,
                "t6_seal_sha256": "a" * 64, "files": records}
    manifest["seal_sha256"] = runner.canonical_sha256(manifest)
    target = tmp_path / runner.EXECUTION_PATH
    target.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = {"root": tmp_path, "baseline": {"seal_sha256": "a" * 64}}
    assert runner.verify_execution(bundle) == manifest
    if mutation == "source":
        (tmp_path / "scripts/t7_ledger.py").write_text("changed source\n", encoding="utf-8")
        reason = "execution_source_changed"
    elif mutation == "seal":
        manifest["seal_sha256"] = "b" * 64
        target.write_text(json.dumps(manifest), encoding="utf-8")
        reason = "execution_seal_mismatch"
    else:
        manifest["files"].pop()
        manifest["seal_sha256"] = runner.canonical_sha256({k: v for k, v in manifest.items() if k != "seal_sha256"})
        target.write_text(json.dumps(manifest), encoding="utf-8")
        reason = "execution_file_inventory_changed"
    with pytest.raises(runner.ExecutionError, match=reason):
        runner.verify_execution(bundle)
