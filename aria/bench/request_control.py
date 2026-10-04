"""Explicit, process-local transport pacing for reproducible benchmarks.

Not installed by product runtime. All attached roles share one serial request
gate, including retries and explicitly enabled fallbacks. Metadata never retains
request bodies, headers, response text, URLs, or provider error messages.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from email.utils import parsedate_to_datetime
import http.client
import json
import math
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from aria.engine.policy.response_observability import (
    capture_response_headers, observe_http_error, observe_interrupted,
    observe_success, safe_usage as _safe_usage,
)


class BenchmarkRequestControlStopped(RuntimeError):
    """A declared wait/request budget was exceeded; no more network calls."""


def retry_after_seconds(value: Any, *, wall_time: float) -> float | None:
    """Parse delta seconds or an HTTP-date; never retain the header string."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return None
            delay = parsed.timestamp() - wall_time
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def _network_error_kind(exc: BaseException) -> str:
    """Finite labels only: never persist exception text, URLs or partial bodies."""
    if isinstance(exc, http.client.IncompleteRead):
        return "incomplete_read"
    if isinstance(exc, http.client.RemoteDisconnected):
        return "remote_disconnected"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "url_timeout" if isinstance(exc.reason, TimeoutError) else "url_error"
    if isinstance(exc, ConnectionError):
        return "connection_error"
    return "http_protocol_error"


class BenchmarkRequestControl:
    """One opt-in controller per experiment, not per role or logical call.

    A Retry-After greater than the declared wait cap stops this controller for
    the rest of the run. It is NOT clipped and retried early. The same applies
    to the optional global network ceiling. Starting again requires a new run
    and a new output artifact, unless a prospective journal explicitly restores
    a verified completed prefix. Historical reports are never repaired here.
    """

    def __init__(
        self, *, min_interval_seconds: float,
        max_retry_after_seconds: float = 30.0,
        max_network_requests: int | None = None,
        max_observed_cost_usd: float | None = None,
        stop_on_rate_limit: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] | None = None,
    ):
        for name, value in (
            ("min_interval_seconds", min_interval_seconds),
            ("max_retry_after_seconds", max_retry_after_seconds),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 60):
                raise ValueError(f"{name} must be finite and between 0 and 60")
        if max_network_requests is not None and (
            isinstance(max_network_requests, bool)
            or not isinstance(max_network_requests, int)
            or max_network_requests < 1
        ):
            raise ValueError("max_network_requests must be a positive integer")
        if max_observed_cost_usd is not None and (
            type(max_observed_cost_usd) not in (int, float)
            or not math.isfinite(max_observed_cost_usd) or max_observed_cost_usd <= 0
        ):
            raise ValueError("max_observed_cost_usd must be finite and positive")
        if type(stop_on_rate_limit) is not bool:
            raise ValueError("stop_on_rate_limit must be a boolean")
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_retry_after_seconds = float(max_retry_after_seconds)
        self.max_network_requests = max_network_requests
        self.max_observed_cost_usd = max_observed_cost_usd
        self.stop_on_rate_limit = stop_on_rate_limit
        self._monotonic, self._wall_time, self._sleep = monotonic, wall_time, sleep
        self._opener = opener
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._stop_reason: str | None = None
        self._wait_by_role: Counter[str] = Counter()
        self._rows: list[dict] = []
        self._blocked_before_network = 0
        self._accounting_restored = False

    def restore_accounting(self, saved: dict) -> None:
        """Restore an authentic cumulative prefix into a fresh controller only.

        The caller must first validate its journal/profile/material binding and
        explicit resume authority. This only replays accounting, never requests.
        Configuration and every derived counter must agree; an unknown invoice
        or stopped controller cannot be revived. Restart waits conservatively
        for a full declared interval, without guessing elapsed offline time.
        """
        from aria.bench.evaluation_journal import _accounting

        with self._lock:
            if self._rows or self._blocked_before_network or self._stop_reason or self._accounting_restored:
                raise ValueError("accounting_restore_requires_fresh_controller")
            saved, facts = _accounting(saved)
            if facts["unknown_cost_requests"] or saved.get("stopped_reason"):
                raise ValueError("accounting_restore_unsafe_or_stopped_prefix")
            trial = BenchmarkRequestControl(
                min_interval_seconds=self.min_interval_seconds,
                max_retry_after_seconds=self.max_retry_after_seconds,
                max_network_requests=self.max_network_requests,
                max_observed_cost_usd=self.max_observed_cost_usd,
                stop_on_rate_limit=self.stop_on_rate_limit,
            )
            required = {"role", "retry_index", "http_status", "error_type", "retry_after_seconds",
                        "provider_usage", "network_elapsed_ms"}
            def finite(value):
                return type(value) in (int, float) and math.isfinite(value) and value >= 0
            for row in saved["attempts"]:
                if (not required <= set(row) or not isinstance(row["role"], str) or not row["role"]
                        or type(row["retry_index"]) is not int or row["retry_index"] < 0
                        or (row["http_status"] is not None and (
                            type(row["http_status"]) is not int or not 100 <= row["http_status"] <= 599))
                        or row["error_type"] not in {None, "http_error", "network_error", "timeout", "invalid_json_response"}
                        or not finite(row["network_elapsed_ms"])
                        or (row["retry_after_seconds"] is not None and not finite(row["retry_after_seconds"]))
                        or ("output_limit_reported" in row and type(row["output_limit_reported"]) is not bool)
                        or ("network_error_kind" in row and row["network_error_kind"] not in {
                            "incomplete_read", "remote_disconnected", "timeout", "url_timeout",
                            "url_error", "connection_error", "http_protocol_error"})
                        or ("failure_phase" in row and row["failure_phase"] not in {
                            "connect", "read", "decode", "parse"})):
                    raise ValueError("accounting_restore_invalid_attempt")
            waits = saved.get("wait_seconds_by_role")
            blocked = saved.get("blocked_before_network")
            if (not isinstance(waits, dict) or any(not isinstance(k, str) or not finite(v) for k, v in waits.items())
                    or type(blocked) is not int or blocked < 0):
                raise ValueError("accounting_restore_invalid_counters")
            trial._rows = deepcopy(saved["attempts"])
            trial._wait_by_role = Counter(waits)
            trial._blocked_before_network = blocked
            # Canonical JSON distinguishes bool/int and rejects unknown fields,
            # fabricated totals, changed limits, and a different controller ABI.
            if json.dumps(trial.accounting(), sort_keys=True) != json.dumps(saved, sort_keys=True):
                raise ValueError("accounting_restore_configuration_or_summary_mismatch")
            self._rows = trial._rows
            self._wait_by_role = trial._wait_by_role
            self._blocked_before_network = blocked
            self._accounting_restored = True
            delay = self.min_interval_seconds
            if self._rows:
                delay = max(delay, self._rows[-1].get("retry_after_seconds") or 0)
            self._next_allowed = self._monotonic() + delay

    def attach(self, client: Any) -> dict:
        from aria.engine.policy.llm import MockLLMClient, OpenAILLMClient

        if isinstance(client, MockLLMClient):
            return {"attached": False, "reason": "mock_has_no_network"}
        if not isinstance(client, OpenAILLMClient):
            raise TypeError("benchmark request control requires supported transport")
        existing = getattr(client, "benchmark_request_control", None)
        if existing is not None and existing is not self:
            raise ValueError("client already belongs to another request controller")
        client.benchmark_request_control = self
        return {"attached": True, "scope": "shared_across_all_benchmark_roles"}

    def role_wait_seconds(self, role: str) -> float:
        return float(self._wait_by_role[role])

    def _before_request(self, client: Any, role: str) -> None:
        if not self._stop_reason and self.max_observed_cost_usd is not None and self._rows:
            costs = [row["provider_usage"].get("cost") for row in self._rows]
            if any(value is None for value in costs):
                self._stop_reason = "provider_cost_unknown_with_cost_guard"
            elif sum(costs) >= self.max_observed_cost_usd:
                self._stop_reason = "observed_cost_budget_exhausted"
        if not self._stop_reason and self.max_network_requests is not None and len(self._rows) >= self.max_network_requests:
            self._stop_reason = "network_request_budget_exhausted"
        if self._stop_reason:
            self._blocked_before_network += 1
            client._update_diagnostic(
                http_status=None, error_type="benchmark_request_control_stopped"
            )
            raise BenchmarkRequestControlStopped(self._stop_reason)
        delay = max(0.0, self._next_allowed - self._monotonic())
        if delay:
            before = self._monotonic()
            self._sleep(delay)
            self._wait_by_role[role] += max(0.0, self._monotonic() - before)
        self._next_allowed = self._monotonic() + self.min_interval_seconds

    def request_json(self, client: Any, request: urllib.request.Request) -> dict:
        # Serializing the whole transport prevents simultaneous attached roles
        # from racing around a shared Retry-After deadline.
        with self._lock:
            return self._request_json_locked(client, request)

    def _request_json_locked(self, client: Any, request: urllib.request.Request) -> dict:
        role = str(client.role)
        transient = {429, 500, 502, 503, 504}
        for attempt in range(client.max_retries + 1):
            self._before_request(client, role)
            client._record_network_request()
            started = self._monotonic()
            row = {"role": role, "retry_index": attempt, "http_status": None,
                   "error_type": None, "retry_after_seconds": None,
                   "provider_usage": {}}
            can_retry = False
            phase = "connect"
            header_observation = None
            try:
                opener = self._opener or urllib.request.urlopen
                with opener(request, timeout=client.timeout) as response:
                    status = getattr(response, "status", None)
                    if status is None and hasattr(response, "getcode"):
                        status = response.getcode()
                    row["http_status"] = int(status) if status is not None else None
                    client._update_diagnostic(http_status=row["http_status"], error_type=None)
                    header_observation = capture_response_headers(response)
                    phase = "read"
                    raw = response.read()
                phase = "decode"
                decoded = raw.decode("utf-8")
                phase = "parse"
                data = json.loads(decoded)
                row["provider_usage"] = _safe_usage(data)
                row["response_observability"] = observe_success(data, header_observation=header_observation)
                choices = data.get("choices") if isinstance(data, dict) else None
                choice = choices[0] if isinstance(choices, list) and choices else None
                # Only a boolean shape flag; no provider-controlled strings.
                row["output_limit_reported"] = (
                    isinstance(choice, dict) and choice.get("finish_reason") == "length"
                )
                return data
            except urllib.error.HTTPError as exc:
                row.update(http_status=exc.code, error_type="http_error", failure_phase=phase)
                observation, usage = observe_http_error(exc)
                row["response_observability"], row["provider_usage"] = observation, usage
                can_retry = exc.code in transient
                if can_retry:
                    try:
                        retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    except Exception:
                        # Optional observability must not replace the HTTPError.
                        retry_after = None
                    row["retry_after_seconds"] = retry_after_seconds(
                        retry_after,
                        wall_time=self._wall_time(),
                    )
                client._update_diagnostic(http_status=exc.code, error_type="http_error",
                                          failure_phase=phase, response_observability=deepcopy(observation))
                self._schedule_server_wait(row)
                if self.stop_on_rate_limit and exc.code == 429:
                    self._stop_reason = "http_429_requires_new_registration"
                elif self.max_observed_cost_usd is not None and "cost" not in usage:
                    self._stop_reason = "provider_cost_unknown_with_cost_guard"
                if self._stop_reason or not can_retry or attempt >= client.max_retries:
                    raise
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError) as exc:
                kind = "timeout" if isinstance(exc, TimeoutError) else "network_error"
                row.update(error_type=kind, network_error_kind=_network_error_kind(exc), failure_phase=phase)
                if header_observation is not None and phase == "read" and row["http_status"] is not None and 200 <= row["http_status"] <= 299:
                    row["response_observability"] = observe_interrupted(header_observation, phase=phase, error=exc)
                client._update_diagnostic(http_status=row["http_status"], error_type=kind,
                                          network_error_kind=row["network_error_kind"], failure_phase=phase,
                                          **({"response_observability": deepcopy(row["response_observability"])}
                                             if "response_observability" in row else {}))
                if self.max_observed_cost_usd is not None and not self._stop_reason:
                    self._stop_reason = "provider_cost_unknown_with_cost_guard"
                can_retry = True
                if self._stop_reason or attempt >= client.max_retries:
                    raise
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                row["error_type"] = "invalid_json_response"
                row["failure_phase"] = phase
                if header_observation is not None and row["http_status"] is not None and 200 <= row["http_status"] <= 299:
                    row["response_observability"] = observe_interrupted(header_observation, phase=phase, error=exc)
                client._update_diagnostic(error_type="invalid_json_response", failure_phase=phase,
                                          **({"response_observability": deepcopy(row["response_observability"])}
                                             if "response_observability" in row else {}))
                if self.max_observed_cost_usd is not None and not self._stop_reason:
                    self._stop_reason = "provider_cost_unknown_with_cost_guard"
                can_retry = True
                if self._stop_reason or attempt >= client.max_retries:
                    raise
            finally:
                row["network_elapsed_ms"] = round(
                    (self._monotonic() - started) * 1000, 3
                )
                self._rows.append(row)
            if can_retry:
                backoff = min(4.0, 0.5 * (2 ** attempt))
                self._next_allowed = max(
                    self._next_allowed, self._monotonic() + backoff
                )
        raise RuntimeError("unreachable")

    def _schedule_server_wait(self, row: dict) -> None:
        delay = row["retry_after_seconds"]
        if delay is None:
            return
        if delay > self.max_retry_after_seconds:
            self._stop_reason = "retry_after_exceeds_declared_wait_budget"
        else:
            # Even a final failed attempt must delay the NEXT logical call.
            self._next_allowed = max(self._next_allowed, self._monotonic() + delay)

    def accounting(self) -> dict:
        usage_rows = [row["provider_usage"] for row in self._rows]
        costs = [row["cost"] for row in usage_rows if "cost" in row]
        token_rows = [row for row in usage_rows if any(key != "cost" for key in row)]
        tokens: Counter = Counter()
        for row in token_rows:
            tokens.update({key: value for key, value in row.items() if key != "cost"})
        return {
            "enabled": True,
            "version": "1.0",
            "scope": "one_benchmark_process_all_attached_roles_and_network_retries",
            "min_request_start_interval_seconds": self.min_interval_seconds,
            "max_retry_after_seconds": self.max_retry_after_seconds,
            "retry_after_over_budget_policy": "stop_run_transport_without_early_retry",
            "max_network_requests": self.max_network_requests,
            "max_observed_cost_usd": self.max_observed_cost_usd,
            "cost_guard_scope": "stop_before_next_request_after_observed_limit_or_unknown_cost_not_absolute_invoice_cap",
            "actual_network_requests": len(self._rows),
            "same_model_retry_network_requests": sum(row["retry_index"] > 0 for row in self._rows),
            "http_status_counts": dict(Counter(str(row["http_status"]) for row in self._rows)),
            "error_type_counts": dict(Counter(row["error_type"] for row in self._rows if row["error_type"])),
            "output_limit_responses": sum(bool(row.get("output_limit_reported")) for row in self._rows),
            "wait_seconds_by_role": {key: round(value, 6) for key, value in self._wait_by_role.items()},
            "total_wait_seconds": round(sum(self._wait_by_role.values()), 6),
            "stopped_reason": self._stop_reason,
            "blocked_before_network": self._blocked_before_network,
            "attempts": deepcopy(self._rows),
            "provider_usage_scope": "numeric_metadata_from_each_parsed_network_envelope_including_invalid_model_output",
            "provider_token_totals": dict(tokens),
            "provider_token_observed_requests": len(token_rows),
            "provider_cost_observed_requests": len(costs),
            "provider_observed_cost_usd": round(sum(costs), 8) if costs else None,
            "provider_cost_complete": bool(self._rows) and len(costs) == len(self._rows),
            "provider_error_costs_unknown_not_zero": True,
            "network_latency_scope": "attempt_wall_time_excluding_controller_waits_not_product_end_to_end",
            "product_runtime_defaults_changed": False,
            **({"stop_on_rate_limit": True} if self.stop_on_rate_limit else {}),
        }
