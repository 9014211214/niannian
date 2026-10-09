from tests._model_http_stub import stub_model_http
from datetime import datetime, timezone
from email.message import Message
from email.utils import format_datetime
import http.client
import io
import json
import urllib.error

import pytest

from aria.bench.request_control import (
    BenchmarkRequestControl,
    BenchmarkRequestControlStopped,
    retry_after_seconds,
)
from aria.engine.policy.llm import OpenRouterLLMClient


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def wall_time(self):
        return 1_700_000_000 + self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Response:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        if isinstance(self.value, bytes):
            return self.value
        return json.dumps(self.value).encode()


def envelope(*, content="ok", finish_reason="stop"):
    return {
        "model": "fixture/actual",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6, "cost": 0.001,
                  "completion_tokens_details": {"reasoning_tokens": 5},
                  "private": "secret-provider-data"},
    }


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://secret-url.invalid", code, "secret-error-body", headers,
        io.BytesIO(b"private user body"),
    )


def client(role="online_fast", retries=0, fallbacks=None):
    return OpenRouterLLMClient(
        api_key="fixture-not-a-real-key", model="fixture/primary",
        base_url="https://fixture.invalid", role=role, max_retries=retries,
        fallback_models=fallbacks or [],
    )


def controller_fixture(outcomes, **kwargs):
    clock = Clock()
    starts = []
    outcomes = iter(outcomes)

    def opener(_request, **_kwargs):
        starts.append(clock.now)
        clock.now += 0.25
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return Response(outcome)

    control = BenchmarkRequestControl(
        min_interval_seconds=kwargs.pop("min_interval_seconds", 3),
        monotonic=clock.monotonic, wall_time=clock.wall_time,
        sleep=clock.sleep, opener=opener, **kwargs,
    )
    return control, clock, starts


def test_one_gate_paces_roles_retries_and_fallbacks():
    control, clock, starts = controller_fixture([
        http_error(429, 4), envelope(), envelope(), http_error(503), envelope(),
    ])
    online = client(retries=1)
    judge = client("judge")
    reflection = client("reflection", fallbacks=["fixture/fallback"])
    for llm in (online, judge, reflection):
        control.attach(llm)
        assert llm.complete([]) == "ok"
    assert starts == [0.0, 4.25, 7.25, 10.25, 13.25]
    assert online.network_request_count == 2
    assert reflection.network_request_count == 2
    report = control.accounting()
    assert report["actual_network_requests"] == 5
    assert report["same_model_retry_network_requests"] == 1
    assert report["http_status_counts"] == {"429": 1, "200": 3, "503": 1}
    assert report["total_wait_seconds"] == sum(clock.sleeps)
    assert report["provider_observed_cost_usd"] == pytest.approx(0.003)
    assert report["provider_cost_complete"] is False
    assert report["provider_cost_observed_requests"] == 3
    assert all(row["network_elapsed_ms"] == 250 for row in report["attempts"])
    serialized = json.dumps(report)
    for private in ("secret", "private", "fixture-not-a-real-key", "https://"):
        assert private not in serialized


def test_final_rate_limit_still_paces_next_role_without_retry():
    control, _clock, starts = controller_fixture([http_error(429, 7), envelope()])
    online, judge = client(), client("judge")
    control.attach(online)
    control.attach(judge)
    with pytest.raises(urllib.error.HTTPError):
        online.complete([])
    assert judge.complete([]) == "ok"
    assert starts == [0.0, 7.25]
    assert control.accounting()["same_model_retry_network_requests"] == 0


def test_excess_retry_after_is_never_clipped_or_bypassed_by_next_role():
    control, clock, starts = controller_fixture(
        [http_error(429, 31)], max_retry_after_seconds=30,
    )
    online, judge = client(retries=3), client("judge", retries=3)
    control.attach(online)
    control.attach(judge)
    with pytest.raises(urllib.error.HTTPError):
        online.complete([])
    with pytest.raises(BenchmarkRequestControlStopped):
        judge.complete([])
    assert starts == [0.0]
    assert clock.sleeps == []
    assert judge.network_request_count == 0
    assert control.accounting()["stopped_reason"] == "retry_after_exceeds_declared_wait_budget"


def test_hard_request_budget_covers_retries_and_other_roles():
    control, _clock, starts = controller_fixture(
        [http_error(503), http_error(503)], max_network_requests=2,
    )
    llm = client(retries=5)
    control.attach(llm)
    with pytest.raises(BenchmarkRequestControlStopped):
        llm.complete([])
    assert len(starts) == llm.network_request_count == 2
    assert llm.last_diagnostic["network_model_requests"] == 2
    assert control.accounting()["stopped_reason"] == "network_request_budget_exhausted"


@pytest.mark.parametrize("failure", [
    http_error(503), TimeoutError(), urllib.error.URLError("private-network-detail"),
    http.client.IncompleteRead(b"private-partial-body"),
    b"malformed envelope", b"\xff",
])
def test_transient_failures_have_bounded_same_model_retry(failure):
    control, _clock, starts = controller_fixture([failure, envelope()])
    llm = client(retries=1)
    control.attach(llm)
    assert llm.complete([]) == "ok"
    assert starts == [0.0, 3.0]
    assert llm.generation_decision_count == 1
    assert llm.network_request_count == 2
    assert control.accounting()["provider_cost_complete"] is False


def test_same_model_retry_exhaustion_and_auth_errors_do_not_expand_budget():
    control, _clock, starts = controller_fixture([http_error(503)] * 3)
    llm = client(retries=2)
    control.attach(llm)
    with pytest.raises(urllib.error.HTTPError):
        llm.complete([])
    assert len(starts) == 3
    auth_control, _clock, starts = controller_fixture([http_error(401)])
    llm = client(retries=2)
    auth_control.attach(llm)
    with pytest.raises(urllib.error.HTTPError):
        llm.complete([])
    assert len(starts) == 1


def test_output_length_is_accounted_but_never_retried_as_transport_error():
    control, _clock, starts = controller_fixture([
        envelope(content="", finish_reason="length")
    ])
    llm = client("reflection", retries=2)
    control.attach(llm)
    with pytest.raises(ValueError, match="可用文本"):
        llm.complete([])
    assert len(starts) == 1
    report = control.accounting()
    assert report["output_limit_responses"] == 1
    assert report["provider_cost_complete"] is True
    assert report["provider_token_totals"]["reasoning_tokens"] == 5




@pytest.mark.parametrize("value, expected", [
    ("4", 4), ("0", 0), ("-2", 0), ("1.5", 1.5),
    ("NaN", None), ("inf", None), ("junk", None), (None, None),
])
def test_retry_after_parsing(value, expected):
    assert retry_after_seconds(value, wall_time=1_700_000_000) == expected


def test_http_date_uses_injected_wall_clock():
    date = datetime.fromtimestamp(1_700_000_011, tz=timezone.utc)
    assert retry_after_seconds(format_datetime(date, usegmt=True), wall_time=1_700_000_000) == 11


@pytest.mark.parametrize("kwargs", [
    {"min_interval_seconds": -1}, {"min_interval_seconds": float("nan")},
    {"min_interval_seconds": 61}, {"max_retry_after_seconds": float("inf")},
    {"max_network_requests": 0}, {"max_network_requests": True},
])
def test_invalid_controller_policy_fails_before_network(kwargs):
    with pytest.raises(ValueError):
        BenchmarkRequestControl(**{"min_interval_seconds": 0, **kwargs})


def test_unattached_product_client_uses_original_transport(monkeypatch):
    starts = []

    def opener(*_args, **_kwargs):
        starts.append(True)
        return Response(envelope())

    stub_model_http(monkeypatch, opener)
    llm = client()
    assert not hasattr(llm, "benchmark_request_control")
    assert llm.max_completion_tokens == 420
    assert llm.complete([]) == "ok"
    assert starts == [True]


def test_controller_cannot_silently_replace_another_experiment():
    first, *_ = controller_fixture([])
    second, *_ = controller_fixture([])
    llm = client()
    first.attach(llm)
    with pytest.raises(ValueError, match="another"):
        second.attach(llm)








