"""Actual local transport receipts can resume; no model/network quality claim."""
from copy import deepcopy

import pytest

from aria.bench.request_control import BenchmarkRequestControlStopped
from tests.test_bench_request_control import client, controller_fixture, envelope


def _prefix(**kwargs):
    control, _, _ = controller_fixture([envelope()], **kwargs)
    llm = client()
    control.attach(llm)
    llm.complete([])
    return control.accounting()


def test_resume_preserves_actual_prefix_cumulative_cost_and_cap():
    opts = {"max_network_requests": 2, "max_observed_cost_usd": .01}
    saved = _prefix(**opts)
    control, clock, starts = controller_fixture([envelope()], **opts)
    llm = client()
    control.attach(llm)
    control.restore_accounting(saved)
    assert control.accounting() == saved
    assert starts == [] and llm.network_request_count == 0
    llm.complete([])
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    actual = control.accounting()
    assert actual["attempts"][:1] == saved["attempts"]
    assert actual["actual_network_requests"] == 2
    assert actual["provider_observed_cost_usd"] == .002
    assert starts == [3.0] and clock.sleeps == [3.0]
    assert llm.network_request_count == 1
    saved["attempts"][0]["provider_usage"]["cost"] = 99
    assert actual["attempts"][0]["provider_usage"]["cost"] == .001


def test_resume_cost_ceiling_is_not_reset():
    opts = {"max_network_requests": 10, "max_observed_cost_usd": .001}
    saved = _prefix(**opts)
    control, _, starts = controller_fixture([], **opts)
    control.restore_accounting(saved)
    # Neither the input nor a newly exported snapshot may alias live usage.
    exported = control.accounting()
    exported["attempts"][0]["provider_usage"]["cost"] = 0
    exported["attempts"].clear()
    saved["attempts"][0]["provider_usage"]["cost"] = 0
    assert control.accounting()["provider_observed_cost_usd"] == .001
    llm = client()
    control.attach(llm)
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert starts == []
    assert control.accounting()["stopped_reason"] == "observed_cost_budget_exhausted"


@pytest.mark.parametrize("field,value", [
    ("max_network_requests", 100), ("max_observed_cost_usd", 100),
    ("min_request_start_interval_seconds", 0), ("total_wait_seconds", 100),
    ("actual_network_requests", True), ("provider_token_totals", {}),
    ("same_model_retry_network_requests", True), ("http_status_counts", {}),
    ("blocked_before_network", True), ("provider_cost_complete", 1),
    ("wait_seconds_by_role", {"judge": float("nan")}),
])
def test_restore_rejects_changed_config_and_fabricated_summaries(field, value):
    saved = _prefix(max_network_requests=3, max_observed_cost_usd=.1)
    saved[field] = value
    control, _, starts = controller_fixture([], max_network_requests=3, max_observed_cost_usd=.1)
    with pytest.raises(ValueError):
        control.restore_accounting(saved)
    assert starts == [] and control.accounting()["actual_network_requests"] == 0


@pytest.mark.parametrize("field,value", [
    ("retry_index", True), ("network_elapsed_ms", -1), ("role", None),
    ("http_status", True), ("output_limit_reported", 1),
    ("retry_after_seconds", float("inf")), ("error_type", "raw-private-error"),
    ("network_error_kind", "PRIVATE-not-a-kind"), ("failure_phase", "PRIVATE-not-a-phase"),
])
def test_restore_rejects_invalid_attempts(field, value):
    saved = _prefix()
    saved["attempts"][0][field] = value
    control, _, _ = controller_fixture([])
    with pytest.raises(ValueError):
        control.restore_accounting(saved)


def test_restore_unknown_fee_or_stopped_controller_never_reenabled():
    control, _, _ = controller_fixture([{"choices": [{"message": {"content": "ok"}}]}])
    llm = client()
    control.attach(llm)
    llm.complete([])
    saved = control.accounting()
    new, _, starts = controller_fixture([])
    with pytest.raises(ValueError, match="unsafe_or_stopped"):
        new.restore_accounting(saved)
    saved = _prefix()
    saved["stopped_reason"] = "retry_after_exceeds_declared_wait_budget"
    with pytest.raises(ValueError, match="unsafe_or_stopped"):
        new.restore_accounting(saved)
    assert starts == []


def test_restore_is_one_time_fresh_only():
    saved = _prefix()
    control, _, _ = controller_fixture([])
    control.restore_accounting(saved)
    with pytest.raises(ValueError, match="fresh"):
        control.restore_accounting(deepcopy(saved))
    active, _, _ = controller_fixture([envelope()])
    llm = client()
    active.attach(llm)
    llm.complete([])
    with pytest.raises(ValueError, match="fresh"):
        active.restore_accounting(saved)
