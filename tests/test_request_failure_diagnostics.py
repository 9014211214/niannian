"""A failed read must remain unknown cost, with safe actionable diagnostics."""
import http.client
import io
import json
import urllib.error

import pytest

from aria.bench.request_control import BenchmarkRequestControl, BenchmarkRequestControlStopped
from aria.engine.policy.llm import OpenRouterLLMClient


@pytest.mark.parametrize("error,kind", [
    (http.client.IncompleteRead(b"PRIVATE-partial"), "incomplete_read"),
    (http.client.RemoteDisconnected("PRIVATE-host"), "remote_disconnected"),
    (ConnectionResetError("PRIVATE-socket"), "connection_error"),
    (TimeoutError("PRIVATE-timeout"), "timeout"),
    (urllib.error.URLError(TimeoutError("PRIVATE-url")), "url_timeout"),
])
def test_read_failure_keeps_http_status_unknown_cost_and_no_secret(error, kind):
    class Response(io.BytesIO):
        status = 200
        def read(self, *args):
            raise error
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        return Response()
    control = BenchmarkRequestControl(min_interval_seconds=0, max_network_requests=3,
                                      max_observed_cost_usd=.1, opener=opener)
    client = OpenRouterLLMClient(model="fixture/model", api_key="PRIVATE-key", role="judge", max_retries=0, fallback_models=[])
    control.attach(client)
    with pytest.raises(type(error)):
        client.complete([])
    account = control.accounting()
    row = account["attempts"][0]
    assert row["http_status"] == 200 and row["failure_phase"] == "read"
    assert row["network_error_kind"] == kind and row["provider_usage"] == {}
    assert client.last_diagnostic["http_status"] == 200
    assert not account["provider_cost_complete"]
    assert account["provider_observed_cost_usd"] is None
    assert "PRIVATE" not in json.dumps(account)
    with pytest.raises(BenchmarkRequestControlStopped):
        client.complete([])
    assert len(calls) == 1


@pytest.mark.parametrize("body,phase", [(b"\xff", "decode"), (b"not JSON PRIVATE", "parse")])
def test_decode_and_json_errors_are_distinct_from_network_failure(body, phase):
    class Response(io.BytesIO):
        status = 200
    control = BenchmarkRequestControl(min_interval_seconds=0, opener=lambda *a, **k: Response(body))
    client = OpenRouterLLMClient(model="fixture/model", api_key="fixture", role="judge", max_retries=0, fallback_models=[])
    control.attach(client)
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        client.complete([])
    row = control.accounting()["attempts"][0]
    assert row["failure_phase"] == phase and row["error_type"] == "invalid_json_response"
    assert "network_error_kind" not in row
    assert "PRIVATE" not in json.dumps(row)
