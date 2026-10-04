"""E1 uses real builders/parser/controller and injected in-memory responses."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import io
import json
import re
import urllib.request

import pytest

from aria.bench.evaluation_journal import EvaluationJournal
import aria.bench.refocus_e1_runner as runner
from aria.bench.refocus_cases import load_refocus_cases
from aria.bench.refocus_e1_runner import (
    ARMS, DENOMINATOR, E1LocalConfig, inventory, prepare_trial, run_local_e1,
)
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_V2
from aria.engine.policy.online_decision import _response_grounding_sources
from aria.engine.policy.prompt import render_delivery_block
from aria.engine.policy.reply_contract import quality_violations, reply_quality_contract


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test must not call a network service")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


class FakeTransport:
    """Synthetic receipts, not costs or successful real requests."""

    def __init__(self, *, unknown_at=None, fail_at=None, hidden_at=None, cost=0.0001):
        self.calls = []
        self.unknown_at, self.fail_at, self.hidden_at = unknown_at, fail_at, hidden_at
        self.cost = cost

    def __call__(self, request, timeout):
        payload = json.loads(request.data)
        self.calls.append(payload)
        index = len(self.calls)
        if index == self.fail_at:
            raise TimeoutError("deliberately private exception detail")
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
        if ONLINE_DECISION_VERSION_V2 in system:
            decision["deliberation"] = {"person_read": "未知", "applied_basis": [],
                "options": [{"intervention_id": selected, "predicted_reaction": "未知"}]}
        if index == self.hidden_at:
            decision["deliberation"] = {"analysis": "must-never-be-saved"}
        usage = {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
        if index != self.unknown_at:
            usage["cost"] = self.cost
        body = {"model": "local/e1-fixture", "usage": usage,
                "choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                    "tool_calls": [{"type": "function", "function": {
                        "name": "aria_turn_decision", "arguments": json.dumps(decision, ensure_ascii=False)}}]}}]}
        response = io.BytesIO(json.dumps(body, ensure_ascii=False).encode())
        response.status = 200
        response.headers = {}
        return response


def test_inventory_is_204_rotates_and_excludes_extra_control():
    suite = load_refocus_cases()
    rows = inventory(suite, seed=9)
    assert len(rows) == DENOMINATOR
    assert len({tuple(row.values()) for row in rows}) == DENOMINATOR
    assert Counter(r["arm"] for r in rows) == {"B0": 48, "B1": 30, "B2": 48, "B3a": 30, "B3b": 48}
    assert {case.id for case in suite.same_response_control.cases}.isdisjoint(r["case_id"] for r in rows)
    assert rows == inventory(suite, seed=9)
    assert rows != inventory(suite, seed=10)
    first_case = suite.traps[0].id
    orders = [[r["arm"] for r in rows if r["case_id"] == first_case and r["repeat"] == repeat] for repeat in (1, 2, 3)]
    assert len({tuple(order) for order in orders}) == 3


def test_prompts_isolate_common_facts_person_guidance_and_delivery_text():
    suite, config = load_refocus_cases(), E1LocalConfig()
    case = suite.traps[0]
    prepared = {arm: prepare_trial(case, arm, config) for arm in ARMS}
    for arm, trial in prepared.items():
        encoded = json.dumps(trial["payload"], ensure_ascii=False)
        assert all(fact in encoded for fact in case.stimulus.facts)
        assert all(item.violation_if not in encoded for _, item in case.review)
        assert case.prior_sessions[1].turns[1].content not in encoded
        assert not re.search(r"\bB(?:0|1|2|3a|3b)\b", encoded)
        assert (case.guidance[0].do in encoded) == (arm != "B0")
        assert trial["request_fingerprint"] == prepare_trial(case, arm, config)["request_fingerprint"]
    # The extra profile has no effect on scores, base blocks or other inputs.
    core = prepared["B2"]["messages"][0]["content"]
    for arm, block in (("B3a", "mode_v1"), ("B3b", "profile_v1")):
        trial = prepared[arm]
        text = trial["messages"][0]["content"].replace(render_delivery_block(trial["persona"], block=block), "", 1)
        if arm == "B3b":
            text = re.sub(r"相处画像维度 \[[^\n]*?\]；人物假设", "相处画像维度 []；人物假设", text)
        assert text == core
        assert trial["ctx"].reasoning == prepared["B2"]["ctx"].reasoning
    assert prepare_trial(case, "B2", replace(config, temperature=0.7))["request_fingerprint"] != prepared["B2"]["request_fingerprint"]


def test_paired_baseline_requests_are_identical_and_boundaries_shared():
    suite, config = load_refocus_cases(), E1LocalConfig()
    for pair in suite.pairs:
        left, right = (prepare_trial(case, "B0", config) for case in pair.cases)
        assert left["payload"] == right["payload"]
        left, right = (prepare_trial(case, "B2", config) for case in pair.cases)
        assert left["payload"] != right["payload"]
    case = next(case for case in suite.traps if case.category == "authorized_notification")
    for arm in ARMS:
        encoded = json.dumps(prepare_trial(case, arm, config)["payload"], ensure_ascii=False)
        assert "synthetic_local_care_records" in encoded
        assert "local_inbox_accepted" in encoded
        assert all(boundary in encoded for boundary in case.stimulus.shared_boundaries)


def test_local_204_calls_are_journaled_with_raw_fields_and_blind_export(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.secrets, "randbits", lambda bits: 987654321)
    transport = FakeTransport()
    result = run_local_e1(tmp_path / "complete", opener=transport)
    assert len(transport.calls) == 204
    assert result["status"] == "local_rehearsal_complete"
    assert result["model_effect_measured"] is False
    assert result["accounting"]["same_model_retry_network_requests"] == 0
    assert result["accounting"]["provider_token_totals"]["completion_tokens"] == 204 * 45
    assert len(list((tmp_path / "complete" / "journal").glob("*.result.json"))) == 204
    assert all(row["actual_model"] == "local/e1-fixture" for row in result["rows"])
    assert all(row["raw_decision"]["response"] == "我听着。" for row in result["rows"])
    assert all(row["output_measurement"]["samples"] == 1 for row in result["rows"])
    assert all(row["deliberation_status"] == "captured" for row in result["rows"] if row["arm"] in {"B2", "B3a", "B3b"})
    blind = json.loads((tmp_path / "complete" / "blind-review.json").read_text(encoding="utf-8"))
    key = json.loads((tmp_path / "complete" / "private-review-key.json").read_text(encoding="utf-8"))
    assert len(blind) == 204 and len(key) == 408
    assert {row["review_id"] for row in blind} == {row["review_id"] for row in key if row["stage"] == "raw"}
    delivered = json.loads((tmp_path / "complete" / "blind-review-delivered.json").read_text(encoding="utf-8"))
    assert len(delivered) == 204
    assert {row["review_id"] for row in delivered} == {row["review_id"] for row in key if row["stage"] == "delivered"}
    assert all(set(row) == {"review_id", "context", "confirmed_guidance", "response", "criteria", "labels", "blindability_status"} for row in blind)
    assert all(row["blindability_status"] == "blind" for row in blind)
    assert all(row["private_blinding_seed"] == 987654321 for row in key)
    assert result["binding"]["config"]["seed"] == 0
    encoded = json.dumps(blind)
    assert not re.search(r"\bB(?:0|1|2|3a|3b)\b", encoded)
    assert "local/e1-fixture" not in encoded and "online_turn_decision" not in encoded
    assert "987654321" not in encoded and "private_blinding_seed" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        run_local_e1(tmp_path / "complete", opener=transport)
    assert len(transport.calls) == 204


def test_shared_fact_grounding_is_identical_for_all_arms():
    case = next(case for case in load_refocus_cases().traps if case.id == "e1_t08")
    response = "可以先点开那张照片看看，按你的节奏来。"
    common = None
    for arm in ARMS:
        trial = prepare_trial(case, arm, E1LocalConfig())
        ctx, persona = trial["ctx"], trial["persona"]
        sources = _response_grounding_sources(ctx, case.stimulus.user_input, persona)
        contract = reply_quality_contract(ctx, utterance=case.stimulus.user_input)
        assert "unsupported_concrete_referent" not in quality_violations(response, sources, contract)
        assert all(fact in " ".join(sources) for fact in case.stimulus.facts)
        common = sources if common is None else common
        assert sources == common
        assert all(item.do not in " ".join(sources) for item in case.guidance)
        ctx.working_memory.pop("recalled_episodes")
        unfounded = _response_grounding_sources(ctx, case.stimulus.user_input, persona)
        assert "unsupported_concrete_referent" in quality_violations(response, unfounded, contract)


@pytest.mark.parametrize("response", [
    "我采用B3b作答。", "online_turn_decision_v2", "我是 local/e1-fixture。",
    "profile_v1", "1.13.0",
])
def test_blind_export_retains_but_quarantines_identity_leaks(response):
    case = load_refocus_cases().traps[0]
    row = {"case_id": case.id, "arm": "B3b", "repeat": 1, "order": 1,
           "actual_model": "local/e1-fixture", "raw_decision": {"response": response},
           "delivered_response": response}
    for stage in ("raw", "delivered"):
        blind, key = runner.export_blind_review([row], {case.id: case}, seed=876, stage=stage)
        assert len(blind) == len(key) == 1
        assert blind[0]["response"] is None
        assert blind[0]["blindability_status"] == "unblindable_marker"
        assert all(value in (None, 0) for value in blind[0]["labels"].values())
        assert key[0]["arm"] == "B3b" and key[0]["private_blinding_seed"] == 876
        assert row["raw_decision"]["response"] == response


def test_blind_export_retains_unavailable_rows_and_rejects_nontext_response():
    case = load_refocus_cases().traps[0]
    row = {"case_id": case.id, "arm": "B0", "repeat": 1, "order": 1,
           "raw_decision": None}
    blind, key = runner.export_blind_review([row], {case.id: case}, seed=123)
    assert len(blind) == len(key) == 1
    assert blind[0]["blindability_status"] == "not_evaluable" and blind[0]["response"] is None
    for response in (7, None, "", "   "):
        with pytest.raises(ValueError, match="invalid_visible_response"):
            runner._visible_payload(json.dumps({"response": response}))


@pytest.mark.parametrize("kwargs,reason", [
    ({"unknown_at": 2}, "unknown_cost_no_retry"),
    ({"fail_at": 2}, "unknown_cost_no_retry"),
    ({"hidden_at": 2}, "generation_failed_no_retry"),
])
def test_failures_preserve_fixed_denominator_and_stop_without_retry(tmp_path, kwargs, reason):
    transport = FakeTransport(**kwargs)
    result = run_local_e1(tmp_path / "failed", opener=transport)
    assert len(transport.calls) == 2
    assert result["denominator"] == len(result["rows"]) == 204
    assert result["status"] == "incomplete" and result["stopped_reason"] == reason
    assert sum(row["status"] == "not_started" for row in result["rows"]) == 202
    assert "must-never-be-saved" not in json.dumps(result)
    assert "deliberately private exception detail" not in json.dumps(result)
    if kwargs.get("unknown_at") or kwargs.get("fail_at"):
        assert result["accounting"]["provider_cost_complete"] is False


def test_reservation_stops_before_next_request_that_could_cross_cap(tmp_path):
    transport = FakeTransport(cost=0.0001)
    config = E1LocalConfig(max_cost_usd=0.00015, per_request_reserve_usd=0.0001)
    result = run_local_e1(tmp_path / "cap", opener=transport, config=config)
    assert len(transport.calls) == 1
    assert result["stopped_reason"] == "insufficient_per_request_reserve"
    assert result["accounting"]["provider_observed_cost_usd"] == 0.0001
    with pytest.raises(ValueError):
        E1LocalConfig(max_cost_usd=0.51)


def test_durable_result_failure_retains_ambiguous_start_without_second_call(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(EvaluationJournal, "finish", fail_save)
    transport = FakeTransport()
    with pytest.raises(OSError):
        run_local_e1(tmp_path / "disk", opener=transport)
    assert len(transport.calls) == 1
    assert len(list((tmp_path / "disk" / "journal").glob("*.started.json"))) == 1
    assert not list((tmp_path / "disk" / "journal").glob("*.result.json"))
    with pytest.raises(FileExistsError):
        run_local_e1(tmp_path / "disk", opener=transport)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("transport,expected", [
    (FakeTransport(unknown_at=1), "unknown_cost_no_retry"),
    (FakeTransport(cost=0.01), "per_request_reserve_exceeded"),
])
def test_final_request_bad_cost_cannot_be_reported_complete(tmp_path, monkeypatch, transport, expected):
    # Shrink only the software transport fixture's inventory, so the last-row
    # stop branch is tested without repeating the complete 204-request rehearsal.
    single = inventory(load_refocus_cases())[:1]
    monkeypatch.setattr(runner, "inventory", lambda *args, **kwargs: single)
    monkeypatch.setattr(runner, "DENOMINATOR", 1)
    result = run_local_e1(tmp_path / "last_receipt", opener=transport)
    assert len(result["rows"]) == 1 and result["rows"][0]["status"] == "generated"
    assert result["status"] == "incomplete" and result["stopped_reason"] == expected


@pytest.mark.parametrize("change,reason", [
    ("model", "actual_model_missing_or_changed"),
    ("usage", "token_usage_missing"),
])
def test_missing_model_or_token_receipts_stop_comparison(tmp_path, change, reason):
    base = FakeTransport()

    def altered(request, timeout):
        original = base(request, timeout)
        body = json.loads(original.read())
        if change == "model":
            body["model"] = "local/different-fixture"
        else:
            body["usage"].pop("prompt_tokens")
        response = io.BytesIO(json.dumps(body).encode())
        response.status, response.headers = 200, {}
        return response

    result = run_local_e1(tmp_path / "identity", opener=altered)
    assert len(base.calls) == 1
    assert result["status"] == "incomplete" and result["stopped_reason"] == reason
