import pytest

from aria.bench.request_control import BenchmarkRequestControl, BenchmarkRequestControlStopped
from tests.test_bench_request_control import client, controller_fixture, envelope


def test_observed_cost_limit_blocks_next_call_without_erasing_completed_one():
    control, _, starts = controller_fixture([envelope(), envelope()], max_observed_cost_usd=0.001)
    llm = client()
    control.attach(llm)
    assert llm.complete([]) == "ok"
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1
    report = control.accounting()
    assert report["stopped_reason"] == "observed_cost_budget_exhausted"
    assert report["provider_observed_cost_usd"] == 0.001


def test_cost_guard_stops_on_missing_billing_not_assumed_free():
    missing = envelope()
    del missing["usage"]["cost"]
    control, _, starts = controller_fixture([missing, envelope()], max_observed_cost_usd=1)
    llm = client()
    control.attach(llm)
    llm.complete([])
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == 1
    assert control.accounting()["stopped_reason"] == "provider_cost_unknown_with_cost_guard"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "1"])
def test_cost_guard_rejects_invalid_configuration(value):
    with pytest.raises(ValueError):
        BenchmarkRequestControl(min_interval_seconds=0, max_observed_cost_usd=value)
