"""Neutral mechanical fixtures; zero external connections or real safety items."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import socket
import shutil
import uuid

import pytest

from aria.bench import max_safety_regression as runner
from aria.bench import safety_regression as fixed
from aria.engine.policy.max_release import tls_dispatch
from tests.test_installable_max_release import metadata, make_clients, response, Response


@pytest.fixture(scope="module", autouse=True)
def virtual_guard_pacing():
    """Exercise the real 120-second pacing with elapsed time supplied by tests."""
    original_init = runner.SafetyRunGuard.__init__
    clocks = []

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        clock = [1000.0]
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        clocks.append(clock)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.SafetyRunGuard, "__init__", init)
        yield clocks


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("local_fixture_network_or_key_forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(tls_dispatch, "open_observed", forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)


def fixtures(path):
    path.mkdir(parents=True, exist_ok=True)
    suite = path / "suite"; suite.mkdir()
    cases = [{"regression_id": f"FS-{i:03d}", "execution_input": {
        "messages": [{"role": "assistant", "content": "中性软件历史。"},
                     {"role": "user", "content": "中性软件装配输入。"}],
        "setup": {"language": "zh-CN", "today": "2001-01-02"}, "context_material": []}} for i in range(1, 88)]
    fixed.write_new(suite / "inputs.json", {"fictional_only": True, "cases": cases})
    for name in ("references.json", "protocol.json"):
        fixed.write_new(suite / name, {"fixture_only": True, "semantic_evidence": False})
    ids = [c["regression_id"] for c in cases]
    fixed.write_new(suite / "manifest.json", {"schema": fixed.SCHEMA, "version": "neutral-mechanical-fixture-v1",
        "fictional_only": True, "fixture_only": True, "case_count": 87, "cell_count": 696,
        "ordered_case_ids": ids, "files": {name: fixed.digest((suite / name).read_bytes())
            for name in ("inputs.json", "references.json", "protocol.json")},
        "cells": [{"cell_id": f"{identity}-{layer}-{n}", "case_id": identity, "layer": layer, "criterion": str(n)}
                  for identity in ids for layer in ("raw", "delivery") for n in range(4)]})
    paths = [path / name for name in ("models.json", "online.json", "offline.json")]
    for target, raw in zip(paths, metadata()):
        target.write_bytes(raw)
    return (suite, *paths)


def prepare_one(tmp_path):
    paths = fixtures(tmp_path)
    online, offline, policy, _, _ = runner.configuration(*paths[1:])
    case = fixed.load_suite(paths[0])[1][0]
    return paths, online, offline, policy, case


def guard(tmp_path, policy, prepare=False, cap="1"):
    value = runner.SafetyRunGuard(tmp_path / "ledger.sqlite3", max_requests=174,
        max_observed_cost_usd=cap, reservation_policy=policy)
    value.prepare_only, value.expected_source = prepare, fixed.source_binding()
    value.wire_limit = 100000
    return value


def opened_response(monkeypatch, clients, *, conditional=False, missing_cost=False):
    calls = []
    def opened(request, **kwargs):
        wire = json.loads(request.data)
        carrier = json.loads(wire["messages"][-1]["content"])
        role = "online_fast" if carrier.get("schema") == "online_json_input_v2" else "safety_review"
        client = clients.online if role == "online_fast" else clients.safety_review
        data = response(client)
        if conditional and role == "online_fast":
            value = json.loads(data["choices"][0]["message"]["content"])
            value["safety_concern"] = "fall"
            value["safety_evidence"] = {"quote": "中性软件装配输入。", "subject": "owner", "time": "current", "urgency": "immediate"}
            data["choices"][0]["message"]["content"] = json.dumps(value, ensure_ascii=False)
        if missing_cost:
            data["usage"].pop("cost")
        calls.append(wire)
        kwargs["trace"].preparation_recovery.bind_request_headers(request)
        kwargs["trace"].preparation_recovery.before_preparation()
        return Response(data)
    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    return calls


def test_full87_actual_max_builder_no_key_or_network(tmp_path, monkeypatch, virtual_guard_pacing):
    paths = fixtures(tmp_path)
    original = runner.os.environ.get
    def no_key(name, *args):
        if "API_KEY" in name:
            pytest.fail("offline_preparation_read_key")
        return original(name, *args)
    monkeypatch.setattr(runner.os.environ, "get", no_key)
    clock_count = len(virtual_guard_pacing)
    plan = runner.prepare(*paths, tmp_path / "prepared")
    assert len(virtual_guard_pacing) - clock_count == 87
    assert len(plan["positions"]) == 87 and plan["model_requests"] == 0
    assert plan["release_profile"] == "release_max_v1" and not plan["estimate_is_execution_admission_gate"]
    assert plan["transport"]["physical_attempt_limit"] == 174
    assert set(plan["policy"]["models"]) == {runner.MODEL, "anthropic/claude-sonnet-5"}
    wire = fixed.read(tmp_path / "prepared/prepared-requests/FS-001.json")
    assert wire["model"] == runner.MODEL and wire["reasoning"]["effort"] == "minimal"
    assert wire["max_tokens"] == 4096
    carrier = json.loads(wire["messages"][-1]["content"])
    assert carrier["conversation_history"] == [{"role": "assistant", "content": "中性软件历史。"}]
    assert carrier["current_user_message"] == "中性软件装配输入。"


@pytest.mark.parametrize("conditional", [False, True])
def test_actual_product_normal_and_conditional_review_wires(tmp_path, monkeypatch, conditional):
    _, online, offline, policy, case = prepare_one(tmp_path)
    calls = opened_response(monkeypatch, make_clients(), conditional=conditional)
    g = guard(tmp_path, policy)
    try:
        result = runner.position(case, tmp_path / "case", g, online, offline, "inert-local-key")
        rows = g.snapshot()["attempts"]
        assert result["submit_ok"], fixed.read(tmp_path / "case/result-frame.json")
        assert len(rows) == len(calls) == 2
        assert result["request_roles"] == ["online_fast", "safety_review"]
        assert all(row["status"] == "received" and row["pacing"]["wait_ms"] >= 120000 for row in rows)
        assert rows[0]["reservation"]["output_token_limit"] == 8202
        assert (tmp_path / "case/first-visible.json").is_file()
        assert rows[1]["reservation"]["output_token_limit"] == 5130
        assert (tmp_path / "case/safety-review-first-visible.json").is_file()
        assert calls[1]["max_tokens"] == 1024
        if not conditional:
            assert json.loads(calls[1]["messages"][-1]["content"])["final_protection"] == ""
        assert g.snapshot()["paid_response_replays"] == 0
    finally:
        g.close()


def test_unknown_fee_preserved_and_halts_without_retry(tmp_path, monkeypatch):
    _, online, offline, policy, case = prepare_one(tmp_path)
    calls = opened_response(monkeypatch, make_clients(), missing_cost=True)
    g = guard(tmp_path, policy)
    try:
        result = runner.position(case, tmp_path / "case", g, online, offline, "inert-local-key")
        assert not result["submit_ok"] and len(calls) == 1
        state = g.snapshot()
        assert state["stopped"] and Decimal(state["reserved_cost_usd"]) > 0
        assert "cost" not in state["attempts"][0]["usage"]
    finally:
        g.close()


def test_global_cap_refuses_before_dispatch(tmp_path, monkeypatch):
    _, online, offline, policy, case = prepare_one(tmp_path)
    calls = opened_response(monkeypatch, make_clients())
    g = guard(tmp_path, policy, cap="0.000001")
    try:
        result = runner.position(case, tmp_path / "case", g, online, offline, "inert-local-key")
        assert not result["submit_ok"] and calls == [] and g.snapshot()["attempts"] == []
    finally:
        g.close()


def review(plan):
    return {"fixture": False, "reviewer": "neutral-software-fixture", "verified_at": datetime.now(timezone.utc).isoformat(),
        "billing_mode": "standard_credits", "account_and_workspace_mode_reviewed": True,
        "all_applicable_prices_reviewed": True, "non_token_surcharges_usd": "0",
        **{key: plan[key] for key in ("models_sha256", "endpoints_sha256", "offline_endpoints_sha256")},
        "known_usd": "0.1", "unknown_holds": [{"id": "old-unknown", "reserved_usd": "0.2"}]}


def test_held_baseline_not_released_and_estimate_not_admission_gate():
    plan = {key: "0" * 64 for key in ("models_sha256", "endpoints_sha256", "offline_endpoints_sha256")}
    plan["worst_case_reserved_usd"] = "1000"
    value = review(plan)
    assert runner.validate_billing(value, plan, Decimal("0.4")) == Decimal("0.3")
    with pytest.raises(ValueError):
        runner.validate_billing(value, plan, Decimal("0.3"))
    value["offline_endpoints_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="offline_billing"):
        runner.validate_billing(value, plan, Decimal("0.4"))


def test_cli_default_max_and_explicit_flash_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *args: calls.append(args) or {"worst_case_reserved_usd": "1"})
    monkeypatch.setattr(fixed, "prepare", lambda *args: calls.append(args) or {"worst_case_reserved_usd": "1"})
    args = ["--suite", "s", "--models-json", "m", "--endpoints-json", "o", "--output", "x"]
    with pytest.raises(SystemExit):
        fixed.main(args)
    assert calls == []
    assert fixed.main([*args, "--offline-endpoints-json", "f"]) == 0 and len(calls[-1]) == 5
    assert fixed.main([*args, "--release-profile", "release_v1"]) == 0 and len(calls[-1]) == 4


@pytest.fixture(scope="module")
def prepared_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("max-safety-neutral-template")
    paths = fixtures(path)
    runner.prepare(*paths, path / "prepared")
    return path


@pytest.fixture
def prepared(prepared_template, tmp_path):
    path = tmp_path / "identity"
    shutil.copytree(prepared_template, path)
    plan = fixed.read(path / "prepared/plan.json")
    plan.update(run_id=uuid.uuid4().hex, preparation_directory=str((path / "prepared").resolve()))
    (path / "prepared/plan.json").write_bytes(fixed.encoded(plan))
    return path, plan


def execute_fixture(prepared, monkeypatch, tmp_path, *, conditional=False, missing_cost=False):
    path, plan = prepared
    # Test-only in-memory waiver to exercise the paid path with a forbidden
    # network and synthetic transport. The published fixture remains marked.
    original_load = fixed.load_suite
    def fixture_waiver(directory):
        manifest, cases = original_load(directory)
        assert manifest.get("fixture_only") is True
        return {**manifest, "fixture_only": False}, cases
    monkeypatch.setattr(fixed, "load_suite", fixture_waiver)
    calls = opened_response(monkeypatch, make_clients(), conditional=conditional, missing_cost=missing_cost)
    monkeypatch.setenv("OPENROUTER_API_KEY", "inert-mechanical-execution")
    fixed.write_new(path / "review.json", review(plan))
    args = (path / "suite", path / "models.json", path / "online.json", path / "offline.json",
            path / "prepared/plan.json", path / "review.json", tmp_path / "run", Decimal("10"), "neutral-software-only")
    runner.execute(*args)
    return args, calls


def test_all87_actual_loop_finance_capture_verify_and_unique_claim(prepared, monkeypatch, tmp_path):
    args, calls = execute_fixture(prepared, monkeypatch, tmp_path, conditional=True)
    summary = runner.verify_run(args[6])
    assert len(calls) == summary["physical_attempts"] == 174
    assert summary["fixed_cases"] == 87 and summary["fixed_cells"] == 696
    assert Decimal(summary["new_known_usd"]) == Decimal("0.0174")
    assert Decimal(summary["total_committed_usd"]) == Decimal("0.3174")
    assert summary["old_holds_released_usd"] == "0" and summary["paid_judges"] == 0
    results = fixed.read(args[6] / "results.json")
    assert all(row["status"] == "completed" for row in results["positions"])
    replay = (*args[:6], tmp_path / "attempted-replay", *args[7:])
    with pytest.raises(FileExistsError):
        runner.execute(*replay)
    assert len(calls) == 174 and not replay[6].exists()


def test_real_loop_unknown_holds_first_missing_and_preserves_remaining86(prepared, monkeypatch, tmp_path):
    args, calls = execute_fixture(prepared, monkeypatch, tmp_path, missing_cost=True)
    summary = runner.verify_run(args[6])
    assert len(calls) == 1 and summary["physical_attempts"] == 1
    assert Decimal(summary["new_unknown_hold_usd"]) > 0
    results = fixed.read(args[6] / "results.json")
    assert len([r for r in results["positions"] if r["status"] == "not_executed"]) == 86
    assert results["old_unknown_holds"] == [{"id": "old-unknown", "reserved_usd": "0.2"}]


def test_fixture_cannot_claim_or_read_key(prepared, monkeypatch, tmp_path):
    path, plan = prepared
    fixed.write_new(path / "review.json", review(plan))
    with pytest.raises(ValueError, match="fixture_suite"):
        runner.execute(path / "suite", path / "models.json", path / "online.json", path / "offline.json",
            path / "prepared/plan.json", path / "review.json", tmp_path / "run", Decimal("10"), "neutral-test")
    assert not (path / "prepared/execution-claim.json").exists()


@pytest.mark.parametrize("fault", ["offline_metadata", "source", "quote", "wire", "transport", "plan_copy"])
def test_prepared_mutations_stop_before_claim(prepared, monkeypatch, tmp_path, fault):
    path, plan = prepared
    original_load = fixed.load_suite
    monkeypatch.setattr(fixed, "load_suite", lambda d: ({**original_load(d)[0], "fixture_only": False}, original_load(d)[1]))
    if fault == "offline_metadata":
        (path / "offline.json").write_bytes((path / "offline.json").read_bytes() + b" ")
    elif fault == "source": plan["source"]["runner_sha256"] = "0" * 64
    elif fault == "quote": plan["positions"][0]["online_reserved_estimate_usd"] = "0"
    elif fault == "wire": (path / "prepared/prepared-requests/FS-001.json").write_bytes(b"{}")
    elif fault == "transport": plan["transport"]["minimum_start_gap_seconds"] = 0
    (path / "prepared/plan.json").write_bytes(fixed.encoded(plan))
    plan_path = path / "prepared/plan.json"
    if fault == "plan_copy":
        plan_path = tmp_path / "copy.json"
        plan_path.write_bytes(fixed.encoded({**plan, "run_id": "a" * 32}))
    with pytest.raises(ValueError):
        runner.validate_preparation(path / "suite", path / "models.json", path / "online.json", path / "offline.json", plan_path, "neutral-test")
    assert not (path / "prepared/execution-claim.json").exists()
