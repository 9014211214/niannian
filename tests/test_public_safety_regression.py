"""Neutral software fixtures only; no real safety questions or model calls."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import socket
import shutil
import urllib.error
import uuid

import pytest

MODULE = Path(__file__).parents[1] / "aria/bench/safety_regression.py"
spec = importlib.util.spec_from_file_location("safety_candidate", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def save(path, value):
    path.write_bytes(runner.encoded(value))


def files(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    cases = [{"regression_id": f"FS-{i:03d}", "execution_input": {
        "messages": [{"role": "assistant", "content": "中性软件夹具的有序历史。"},
                     {"role": "user", "content": "只用于本地接口检查。"}],
        "setup": {"language": "zh-CN", "today": "2001-01-02"}, "context_material": []}}
        for i in range(1, 88)]
    save(suite / "inputs.json", {"fictional_only": True, "cases": cases})
    save(suite / "references.json", {"software_fixture_only": True, "no_semantic_reference": True})
    save(suite / "protocol.json", {"software_fixture_only": True})
    ids = [c["regression_id"] for c in cases]
    manifest = {"schema": runner.SCHEMA, "version": "neutral-software-fixture-v1", "fictional_only": True,
        "case_count": 87, "cell_count": 696, "ordered_case_ids": ids,
        "files": {n: runner.digest((suite / n).read_bytes()) for n in ("inputs.json", "references.json", "protocol.json")},
        "cells": [{"cell_id": f"{id_}-{layer}-{n}", "case_id": id_, "layer": layer, "criterion": str(n)}
                  for id_ in ids for layer in ("raw", "delivery") for n in range(4)]}
    save(suite / "manifest.json", manifest)
    models = {"data": [{"id": runner.ONLINE_MODEL, "canonical_slug": "fixture/qwen38", "reasoning": {"mandatory": False}}]}
    endpoints = {"data": {"id": runner.ONLINE_MODEL, "endpoints": [{"model_id": runner.ONLINE_MODEL,
        "tag": "alibaba", "name": "Alibaba | software fixture", "provider_name": "Alibaba", "status": 0,
        "supported_parameters": ["structured_outputs", "response_format", "max_tokens", "reasoning", "temperature"],
        "context_length": 262144, "max_prompt_tokens": 262144, "max_completion_tokens": 8192,
        "pricing": {"prompt": "0.0000002", "completion": "0.00000047", "request": "0",
                    "input_cache_write": "0.00000025", "overrides": [{"min_prompt_tokens": 32000, "prompt": "0.0000003"}]}}]}}
    save(tmp_path / "models.json", models)
    save(tmp_path / "endpoints.json", endpoints)
    return suite, tmp_path / "models.json", tmp_path / "endpoints.json"


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("software_fixture_network_forbidden")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(runner.minimal_calls, "open_unredirected", blocked)


_NEUTRAL_TEMPLATE = None


@pytest.fixture
def prepared(tmp_path_factory, tmp_path):
    global _NEUTRAL_TEMPLATE
    if _NEUTRAL_TEMPLATE is None:
        template = tmp_path_factory.mktemp("neutral-preparation")
        suite, models, endpoints = files(template)
        # Run all actual builders once; each software test below creates its own
        # new fixture identity. execute_fixture copies that identity unchanged.
        runner.prepare(suite, models, endpoints, template / "prepared")
        _NEUTRAL_TEMPLATE = template
    path = tmp_path / "neutral-fixture-identity"
    shutil.copytree(_NEUTRAL_TEMPLATE, path)
    plan = runner.read(path / "prepared/plan.json")
    plan.update(run_id=uuid.uuid4().hex, preparation_directory=str((path / "prepared").resolve()))
    save(path / "prepared/plan.json", plan)
    suite, models, endpoints = path / "suite", path / "models.json", path / "endpoints.json"
    return path, suite, models, endpoints, plan


def review_file(path, plan):
    value = {"fixture": False, "reviewer": "software-fixture-only-not-live-approval",
        "verified_at": datetime.now(timezone.utc).isoformat(), "billing_mode": "standard_credits",
        "account_and_workspace_mode_reviewed": True, "all_applicable_prices_reviewed": True,
        "non_token_surcharges_usd": "0", "models_sha256": plan["models_sha256"], "endpoints_sha256": plan["endpoints_sha256"],
        "known_usd": "0.01", "unknown_holds": [{"id": "neutral-old-unknown", "reserved_usd": "0.02"}]}
    save(path, value)
    return path


def local_plan(prepared, tmp_path):
    target = tmp_path / "prepared-copy"
    if not target.exists():
        shutil.copytree(prepared[0] / "prepared", target)
    return target / "plan.json"


def decision():
    return {"response": "已收到这条中性软件夹具。", "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": 0.2,
            "divergence": {"possible": False, "confidence": 0, "pattern": ""}},
        "selected_intervention_id": "", "uncertainty": 0.8,
        "communication_plan": {"pace": "open_listening", "support_order": "只检查接口", "boundary": "无真实判断"},
        "safety_concern": "none", "safety_evidence": {"quote": "", "subject": "not_applicable",
            "time": "not_applicable", "urgency": "not_applicable"}}


def response_raw(**changes):
    data = {"id": "gen-neutralsoftwarefixture1234", "model": runner.ONLINE_MODEL, "provider": "Alibaba",
        "usage": {"prompt_tokens": 8, "completion_tokens": 8, "cost": 0.00001},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision(), ensure_ascii=False),
            "reasoning": "SYNTHETIC_HIDDEN_CHANNEL_MUST_NOT_PERSIST"}}]}
    data.update(changes)
    # A nonempty reasoning field is intentionally rejected by the real guard.
    return runner.encoded(data)


class Response:
    status = 200
    headers = {}
    def __init__(self, raw):
        self.raw = raw
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, size=-1):
        raw, self.raw = self.raw, b""
        return raw if size < 0 else raw[:size]


def execute_fixture(prepared, tmp_path, monkeypatch, opener):
    path, suite, models, endpoints, plan = prepared
    monkeypatch.setenv("OPENROUTER_API_KEY", "inert-software-fixture")
    monkeypatch.setattr(runner.minimal_calls, "open_unredirected", opener)
    # Never wait on wall time in mechanical tests; production keeps 120 seconds.
    original_init = runner.RunGuard.__init__
    def init(self, *a, **k):
        original_init(self, *a, **k)
        clock = [0.0]
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    monkeypatch.setattr(runner.RunGuard, "__init__", init)
    review = review_file(tmp_path / "review.json", plan)
    out = tmp_path / "run"
    runner.execute(suite, models, endpoints, local_plan(prepared, tmp_path), review, out, Decimal("2"), "neutral-test-operator")
    return out


def test_all87_dry_preparation_uses_actual_builder_and_preserves_exact_history(prepared):
    path, suite, _, _, plan = prepared
    assert plan["model_requests"] == 0 and plan["credentials_read"] is False
    assert len(plan["positions"]) == 87
    assert plan["transport"]["automatic_retries"] == 0
    policy = plan["policy"]["models"][runner.ONLINE_MODEL]
    assert Decimal(policy["input_usd_per_million"]) == Decimal("0.3")
    for row in plan["positions"]:
        wire_path = path / "prepared/prepared-requests" / (row["case_id"] + ".json")
        wire = runner.read(wire_path)
        assert runner.digest(wire_path.read_bytes()) == row["prepared_request_sha256"]
        assert wire["provider"]["only"] == ["alibaba"]
        assert wire["max_tokens"] == 4096 and wire["stream"] is False
        assert wire["response_format"]["type"] == "json_schema"
        carrier = json.loads(wire["messages"][-1]["content"])
        assert carrier["conversation_history"] == [{"role": "assistant", "content": "中性软件夹具的有序历史。"}]
        assert carrier["current_user_message"] == "只用于本地接口检查。"
        assert sum(len(m["content"].encode()) for m in wire["messages"] if m["role"] == "system") <= 12000
        assert "2001-01-02" in wire["messages"][0]["content"]


def test_default_preparation_never_reads_key(tmp_path, monkeypatch):
    suite, models, endpoints = files(tmp_path)
    original = runner.os.environ.get
    def get(key, *a):
        if "API_KEY" in key:
            pytest.fail("dry_run_read_credentials")
        return original(key, *a)
    monkeypatch.setattr(runner.os.environ, "get", get)
    runner.prepare(suite, models, endpoints, tmp_path / "prepared")


@pytest.mark.parametrize("kind", ["input_bytes", "missing_case", "duplicate_case", "missing_cell", "duplicate_cell", "path_escape", "context_injection"])
def test_fixed_set_integrity_rejections(tmp_path, kind):
    suite, _, _ = files(tmp_path)
    manifest = runner.read(suite / "manifest.json")
    inputs = runner.read(suite / "inputs.json")
    if kind == "input_bytes":
        (suite / "inputs.json").write_bytes(b"changed")
    elif kind in {"missing_case", "duplicate_case", "context_injection"}:
        if kind == "missing_case": inputs["cases"].pop()
        if kind == "duplicate_case": inputs["cases"][1] = inputs["cases"][0]
        if kind == "context_injection": inputs["cases"][0]["execution_input"]["context_material"] = ["injected"]
        save(suite / "inputs.json", inputs)
        manifest["files"]["inputs.json"] = runner.digest((suite / "inputs.json").read_bytes())
    elif kind == "missing_cell": manifest["cells"].pop()
    elif kind == "duplicate_cell": manifest["cells"][1] = manifest["cells"][0]
    elif kind == "path_escape": manifest["files"]["../unexpected"] = "0" * 64
    save(suite / "manifest.json", manifest)
    with pytest.raises(ValueError): runner.load_suite(suite)


def test_first_visible_preserved_unknown_cost_held_and_remaining85_not_replayed(prepared, tmp_path, monkeypatch):
    count = []
    def open_(request, **kwargs):
        count.append(request)
        data = json.loads(response_raw())
        del data["choices"][0]["message"]["reasoning"]
        if len(count) == 2:
            data["usage"].pop("cost")
        return Response(runner.encoded(data))
    out = execute_fixture(prepared, tmp_path, monkeypatch, open_)
    summary = runner.verify_run(out)
    assert len(count) == 2 and summary["physical_attempts"] == 2
    assert Decimal(summary["new_known_usd"]) == Decimal("0.00001")
    assert Decimal(summary["new_unknown_hold_usd"]) > 0
    results = runner.read(out / "results.json")
    assert len(results["positions"]) == 87
    assert sum(r["status"] == "not_executed" for r in results["positions"]) == 85
    assert results["old_unknown_holds"] == [{"id": "neutral-old-unknown", "reserved_usd": "0.02"}]
    assert (out / "FS-002/first-visible.json").is_file()
    assert runner.read(out / "closure.json")["can_resume"] is False
    with pytest.raises(FileExistsError):
        execute_fixture(prepared, tmp_path, monkeypatch, open_)
    _, suite, models, endpoints, plan = prepared
    with pytest.raises(FileExistsError):
        runner.execute(suite, models, endpoints, local_plan(prepared, tmp_path), tmp_path / "review.json",
                       tmp_path / "duplicate-other-directory", Decimal("2"), "fixture")
    assert len(count) == 2


@pytest.mark.parametrize("failure", ["429", "tls", "malformed", "duplicate", "hidden", "interrupt"])
def test_failures_lock_first_attempt_and_close_with_no_hidden_channel(prepared, tmp_path, monkeypatch, failure):
    calls = []
    def open_(request, **kwargs):
        calls.append(request)
        if failure == "429":
            raise urllib.error.HTTPError(request.full_url, 429, "fixture", {}, io.BytesIO(b'{"error":{"code":429}}'))
        if failure == "tls": raise ConnectionResetError("neutral fixture")
        if failure == "interrupt": raise KeyboardInterrupt()
        if failure == "malformed": return Response(b"not-json-fixture")
        if failure == "duplicate":
            return Response(b'{"usage":{"cost":0.1,"cost":0.2},"choices":[{"message":{"content":"first","content":"second"}}]}')
        return Response(response_raw())
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt): execute_fixture(prepared, tmp_path, monkeypatch, open_)
        out = tmp_path / "run"
    else:
        out = execute_fixture(prepared, tmp_path, monkeypatch, open_)
    assert len(calls) == 1
    assert (out / "closure.json").is_file()
    summary = runner.verify_run(out)
    if failure != "hidden": assert Decimal(summary["new_unknown_hold_usd"]) > 0
    for p in out.rglob("*"):
        if p.is_file() and p.suffix != ".sqlite3":
            assert b"SYNTHETIC_HIDDEN_CHANNEL_MUST_NOT_PERSIST" not in p.read_bytes()
            assert b"inert-software-fixture" not in p.read_bytes()
    if failure == "duplicate":
        captured = (out / "FS-001/first-visible.json").read_text("utf-8")
        assert "first" in captured and "second" in captured


@pytest.mark.parametrize("change", ["expired", "wrong_mode", "price_sha", "surcharge", "hold_budget"])
def test_billing_rejects_before_claim_and_key(prepared, tmp_path, change):
    path, suite, models, endpoints, plan = prepared
    review = review_file(tmp_path / "review.json", plan)
    data = runner.read(review)
    if change == "expired": data["verified_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    if change == "wrong_mode": data["billing_mode"] = "unreviewed_byok"
    if change == "price_sha": data["endpoints_sha256"] = "0" * 64
    if change == "surcharge": data["non_token_surcharges_usd"] = "0.1"
    if change == "hold_budget": data["unknown_holds"][0]["reserved_usd"] = "2"
    save(review, data)
    with pytest.raises(ValueError):
        runner.execute(suite, models, endpoints, path / "prepared/plan.json", review, tmp_path / "run", Decimal("2"), "fixture")
    assert not (tmp_path / "run").exists()


def test_missing_key_closes_zero_request_claim(prepared, tmp_path, monkeypatch):
    path, suite, models, endpoints, plan = prepared
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="missing_environment_key"):
        runner.execute(suite, models, endpoints, local_plan(prepared, tmp_path), review_file(tmp_path / "review.json", plan),
                       tmp_path / "run", Decimal("2"), "fixture")
    assert runner.verify_run(tmp_path / "run")["physical_attempts"] == 0


def test_manual_scoring_preserves696_and_missing_ne(prepared, tmp_path, monkeypatch):
    out = execute_fixture(prepared, tmp_path, monkeypatch, lambda *a, **k: Response(response_raw()))
    scores = runner.read(out / "scores-template.json")
    assert len(scores["cells"]) == 696 and all(c["status"] is None for c in scores["cells"])
    scores["grader"] = "independent-software-fixture"
    scores["grader_is_neither_author_nor_implementer"] = True
    for cell in scores["cells"]:
        cell.update(status="NE", reason="neutral mechanical fixture; not semantic grading", evidence=[])
    save(tmp_path / "scores.json", scores)
    summary = runner.verify_run(out, tmp_path / "scores.json")
    assert summary["scores"] == {"P": 0, "F": 0, "NE": 696}
    scores["cells"][-1]["status"] = "P"
    save(tmp_path / "scores.json", scores)
    with pytest.raises(ValueError, match="missing_evidence"):
        runner.verify_run(out, tmp_path / "scores.json")
    (out / "results.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="closed_evidence_changed"):
        runner.verify_run(out)


def test_all87_successful_mechanical_outputs_not_automatically_effect_pass(prepared, tmp_path, monkeypatch):
    raw = json.loads(response_raw())
    del raw["choices"][0]["message"]["reasoning"]
    calls = []
    def opened(request, **kwargs):
        calls.append(request)
        return Response(runner.encoded(raw))
    out = execute_fixture(prepared, tmp_path, monkeypatch, opened)
    summary = runner.verify_run(out)
    assert len(calls) == 87 and summary["physical_attempts"] == 87
    assert summary["effect_status"] == "awaiting_independent_scoring"
    assert runner.read(out / "closure.json")["status"] == "all_positions_recorded"
    results = runner.read(out / "results.json")
    assert all(r["status"] == "completed" for r in results["positions"])
    assert all(a["pacing"]["start_gap_ms"] >= 120000 for a in results["ledger"]["attempts"][1:])


def test_source_change_stops_before_claim(prepared, tmp_path, monkeypatch):
    path, suite, models, endpoints, plan = prepared
    monkeypatch.setattr(runner, "source_binding", lambda: {"changed": True})
    with pytest.raises(ValueError, match="unchanged_prepared_source"):
        runner.execute(suite, models, endpoints, path / "prepared/plan.json", review_file(tmp_path / "review.json", plan),
                       tmp_path / "run", Decimal("2"), "fixture")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("flag", ["--help", "missing_execute_budget"])
def test_cli_never_infers_paid_permission(tmp_path, flag):
    with pytest.raises(SystemExit) as result:
        runner.main(["--help"] if flag == "--help" else ["--execute", "--suite", str(tmp_path),
            "--models-json", "models.json", "--endpoints-json", "endpoints.json", "--output", str(tmp_path / "run")])
    assert result.value.code == (0 if flag == "--help" else 2)
    assert not (tmp_path / "run").exists()


def test_empty429_projection_cannot_get_raw_pass(prepared, tmp_path, monkeypatch):
    def failure(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 429, "fixture", {}, io.BytesIO(b'{"error":{"code":429}}'))
    out = execute_fixture(prepared, tmp_path, monkeypatch, failure)
    scores = runner.read(out / "scores-template.json")
    scores.update(grader="independent-neutral-fixture", grader_is_neither_author_nor_implementer=True)
    for cell in scores["cells"]:
        cell.update(status="NE", reason="no model reply", evidence=[])
    scores["cells"][0].update(status="P", reason="invalid pass fixture", evidence=["FS-001/first-visible.json"])
    save(tmp_path / "scores.json", scores)
    with pytest.raises(ValueError, match="missing_evidence"):
        runner.verify_run(out, tmp_path / "scores.json")


def test_copy_before_first_claim_still_uses_original_anchor(prepared, tmp_path, monkeypatch):
    original = prepared[0] / "prepared"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    # Both copies exist before either claim; no marker is copied into them.
    local_plan(prepared, first)
    local_plan(prepared, second)
    assert not (first / "prepared-copy/execution-claim.json").exists()
    assert not (second / "prepared-copy/execution-claim.json").exists()
    calls = []
    def interrupted(request, **kwargs):
        calls.append(request)
        raise ConnectionResetError("neutral fixture")
    execute_fixture(prepared, first, monkeypatch, interrupted)
    assert (original / "execution-claim.json").is_file()
    with pytest.raises(FileExistsError):
        execute_fixture(prepared, second, monkeypatch, interrupted)
    assert len(calls) == 1 and not (second / "run").exists()


@pytest.mark.parametrize("amount", ["0", "-1", "NaN", "Infinity", "-Infinity", "1e-1000000", "0e-1000000", "1e1000000", "0.0000000000001", "0.123456789012345678901", "1" * 1000])
def test_unknown_hold_requires_positive_bounded_exact_amount(prepared, tmp_path, amount):
    plan = prepared[-1]
    review = runner.read(review_file(tmp_path / "review.json", plan))
    review["unknown_holds"][0]["reserved_usd"] = amount
    with pytest.raises(ValueError):
        runner.validate_billing(review, plan, Decimal("2"))


def test_smallest_allowed_hold_is_not_rounded_away(prepared, tmp_path):
    plan = prepared[-1]
    review = runner.read(review_file(tmp_path / "review.json", plan))
    review["known_usd"] = "0.01"
    review["unknown_holds"][0]["reserved_usd"] = "0.000000000001"
    baseline = runner.validate_billing(review, plan, Decimal("2"))
    assert baseline == Decimal("0.010000000001")
    assert baseline != Decimal("0.01")


def test_changed_anchor_without_original_plan_is_rejected(prepared, tmp_path, monkeypatch):
    path, suite, models, endpoints, plan = prepared
    altered = runner.read(path / "prepared/plan.json")
    altered["preparation_directory"] = str((tmp_path / "nonexistent-anchor").resolve())
    changed_plan = tmp_path / "changed-plan.json"
    save(changed_plan, altered)
    with pytest.raises(FileNotFoundError):
        runner.execute(suite, models, endpoints, changed_plan, review_file(tmp_path / "review.json", plan),
                       tmp_path / "run", Decimal("2"), "fixture")
    assert not (tmp_path / "run").exists()
