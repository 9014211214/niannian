"""A small, durable allowance shared by the three minimal-demo model roles.

Every default physical request rejects HTTP redirection, including when local
connection tracing is disabled. Only bounded transport metadata is retained.
This is a stop-before-next-call
guard, not a provider invoice cap, an authorization issuer, or a retry queue.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

from .response_observability import (
    capture_response_headers, observe_http_error, observe_interrupted,
    observe_success, safe_usage,
)
from .request_pacing import monotonic_value, pacing_interval, wait_until
from .transport_failure import transport_failure
from .observed_transport import (TransportTrace, VERSION as TRANSPORT_TRACE_VERSION,
    DIAGNOSTIC_VERSION as TRANSPORT_DIAGNOSTIC_VERSION, PreparationRecovery, open_observed,
    open_unredirected)
from .bounded_recovery import (
    BoundedRecovery, CONTINUATION_SCHEMA, MAIN_CONTINUATION_SCHEMA,
    validate_recovery_policy, validate_recovery_records,
)

_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_ROLES = {"online_fast", "memory", "reflection"}
MAX_MODEL_RESPONSE_BYTES = 2 * 1024 * 1024
_VALIDATION_FAILURE_CODES = frozenset({
    "model_response_body_limit", "duplicate_response_field", "non_object_response",
    "unexpected_http_status", "response_model_mismatch", "response_provider_mismatch",
    "provider_error_envelope",
    "provider_cost_unknown", "provider_token_usage_missing", "reservation_token_bound_exceeded",
    "reservation_cost_exceeded", "disabled_reasoning_returned", "invalid_or_truncated_response",
    "invalid_response_message", "online_content_invalid_json", "online_content_duplicate_key",
    "online_content_non_finite", "online_content_not_object", "online_response_missing",
    "online_response_not_string", "online_response_empty", "online_decision_schema_invalid",
})


def _record_failure_stage(row, phase, error):
    """Fixed local codes only; never copy an exception's message or response."""
    defaults = {"connect": "connection_failed",  # Historical rows keep their original meaning/code.
                "local_pacing_check": "local_pacing_check_failed",
                "local_stop_check": "local_dispatch_cancelled",
                "local_bookkeeping": "local_bookkeeping_failed",
                "pre_response_open": "response_open_failed",
                "response_context": "response_context_failed",
                "read": "response_read_failed",
                "decode": "response_decode_failed", "parse": "response_json_invalid",
                "envelope_validation": "envelope_validation_failed",
                "receipt_validation": "receipt_validation_failed"}
    row["failure_phase"] = phase if phase in defaults else "receipt_validation"
    code = row.get("validation_error")
    if not isinstance(code, str) or code not in _VALIDATION_FAILURE_CODES:
        # A future hook must register its finite public code before persisting
        # it. Unknown/free-form diagnostics must not leak content through here.
        row.pop("validation_error", None)
        code = ("content_json_invalid" if phase == "receipt_validation" and isinstance(error, json.JSONDecodeError)
                else defaults.get(phase, "receipt_validation_failed"))
    row["failure_code"] = code


class MinimalCallStopped(RuntimeError):
    """A safe reason code; never carries credentials, prompts or provider text."""


def _limit(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("model cost threshold must be finite and positive")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number <= 0 or number > 1000000:
            raise InvalidOperation
        return str(number.normalize())
    except (InvalidOperation, ValueError):
        raise ValueError("model cost threshold must be finite and positive") from None


def validate_limits(max_requests, max_observed_cost_usd):
    if type(max_requests) is not int or not 1 <= max_requests <= 10000:
        raise ValueError("model request limit must be an integer from 1 to 10000")
    return max_requests, _limit(max_observed_cost_usd)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _unique_request_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_request_field")
        value[key] = item
    return value


class MinimalCallGuard:
    allowed_roles = _ROLES

    def __init__(self, path, *, max_requests, max_observed_cost_usd, min_interval_seconds=0, observe_transport=False,
                 recovery_policy=None, transport_diagnostics=False, connection_preparation_recovery=False,
                 pre_application_check=None):
        count, amount = validate_limits(max_requests, max_observed_cost_usd)
        if type(observe_transport) is not bool:
            raise ValueError("invalid_transport_observation_setting")
        self.observe_transport = self._observe_transport = observe_transport
        if type(transport_diagnostics) is not bool or transport_diagnostics and not observe_transport:
            raise ValueError("invalid_transport_diagnostics_setting")
        self.transport_diagnostics = self._transport_diagnostics = transport_diagnostics
        self._trace_version = TRANSPORT_DIAGNOSTIC_VERSION if transport_diagnostics else TRANSPORT_TRACE_VERSION
        self.min_interval_seconds = self._pacing_interval = pacing_interval(min_interval_seconds)
        self.recovery_policy = validate_recovery_policy(recovery_policy)
        if self.recovery_policy is not None and (count != self.recovery_policy["max_logical_requests"] + self.recovery_policy["max_extra_dispatches"] or not observe_transport
                or self._pacing_interval != 10 or not hasattr(self, "policy")):
            raise ValueError("bounded_recovery_requires_observed_reserved_shared_allowance")
        if (type(connection_preparation_recovery) is not bool or connection_preparation_recovery and (
                not transport_diagnostics or not callable(pre_application_check) or not self._preparation_policy_supported())
                or not connection_preparation_recovery and pre_application_check is not None):
            raise ValueError("invalid_connection_preparation_recovery")
        self.connection_preparation_recovery = self._connection_preparation_recovery = connection_preparation_recovery
        self._pre_application_check = pre_application_check
        self._pre_application_check_original = pre_application_check
        self._recovery = BoundedRecovery(self, self.recovery_policy)
        self._recovery_clock = time.perf_counter
        self._pacing_clock, self._pacing_sleep = time.monotonic, time.sleep
        self._last_network_started = None
        self._last_network_request_id = None
        self._reopen_pacing_required = False
        self._stop_requested = threading.Event()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._broken = False
        self._closed = False
        self._closed_snapshot = None
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        try:
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL,
                    max_requests INTEGER NOT NULL, threshold TEXT NOT NULL,
                    halted TEXT);
                CREATE TABLE IF NOT EXISTS roles (role TEXT PRIMARY KEY, binding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL, record TEXT NOT NULL);
            """)
            with self._transaction():
                row = self._db.execute("SELECT version,max_requests,threshold FROM settings WHERE id=1").fetchone()
                if row is None:
                    self._db.execute("INSERT INTO settings VALUES(1,1,?,?,NULL)", (count, amount))
                elif row != (1, count, amount):
                    raise ValueError("existing model allowance differs; retained ledger cannot be reset")
                registered = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                              "AND name='request_pacing'").fetchone()
                if registered is None and self._pacing_interval:
                    if self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                        raise ValueError("retained_allowance_has_no_pacing_contract")
                    self._db.execute("CREATE TABLE request_pacing "
                                     "(id INTEGER PRIMARY KEY CHECK(id=1), min_interval_seconds REAL NOT NULL)")
                    self._db.execute("INSERT INTO request_pacing VALUES(1,?)", (self._pacing_interval,))
                self._check_pacing_configuration()
                transport = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                             "AND name='transport_observation'").fetchone()
                if transport is None and self._observe_transport:
                    if self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                        raise ValueError("retained_allowance_has_no_transport_observation_contract")
                    self._db.execute("CREATE TABLE transport_observation "
                                     "(id INTEGER PRIMARY KEY CHECK(id=1), version TEXT NOT NULL)")
                    self._db.execute("INSERT INTO transport_observation VALUES(1,?)", (self._trace_version,))
                preparation = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='connection_preparation_recovery'").fetchone()
                if preparation is None and self._connection_preparation_recovery:
                    if self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                        raise ValueError("retained_allowance_has_no_preparation_contract")
                    self._db.execute("CREATE TABLE connection_preparation_recovery "
                        "(id INTEGER PRIMARY KEY CHECK(id=1),version TEXT NOT NULL)")
                    self._db.execute("INSERT INTO connection_preparation_recovery VALUES(1,'pre_application_rebuild_v1')")
                self._check_transport_configuration()
                self._recovery.initialize()
                if self._pacing_interval and self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                    # Monotonic timestamps cannot be restored across reboots.
                    # Conservatively wait a full interval after reopening.
                    self._reopen_pacing_required = True
            self.max_requests, self.threshold = count, amount
        except BaseException:
            self._db.close()
            raise

    def _preparation_policy_supported(self):
        return self.recovery_policy is not None and self.recovery_policy["schema"] in {
            CONTINUATION_SCHEMA, MAIN_CONTINUATION_SCHEMA}

    def _observe_http_failure(self, exc):
        evidence = {} if self.recovery_policy is not None else None
        observation, usage = observe_http_error(exc, recovery_evidence=evidence)
        return observation, usage, ({"http_error_recovery_evidence": evidence} if evidence else {})

    def _check_pacing_configuration(self):
        if self.min_interval_seconds != self._pacing_interval:
            raise ValueError("request_pacing_configuration_changed")
        registered = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                      "AND name='request_pacing'").fetchone()
        if registered is not None:
            if self._db.execute("SELECT min_interval_seconds FROM request_pacing WHERE id=1").fetchone() != (self._pacing_interval,):
                raise ValueError("retained_request_pacing_changed")
        elif self._pacing_interval:
            raise ValueError("retained_request_pacing_missing")

    def _check_transport_configuration(self):
        if (self.observe_transport is not self._observe_transport or self.transport_diagnostics is not self._transport_diagnostics
                or self.connection_preparation_recovery is not self._connection_preparation_recovery):
            raise ValueError("transport_observation_configuration_changed")
        registered = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                      "AND name='transport_observation'").fetchone()
        if registered is not None:
            if (not self._observe_transport or self._db.execute(
                    "SELECT version FROM transport_observation WHERE id=1").fetchone() != (self._trace_version,)):
                raise ValueError("retained_transport_observation_changed")
        elif self._observe_transport:
            raise ValueError("retained_transport_observation_missing")
        preparation = self._db.execute("SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='connection_preparation_recovery'").fetchone()
        if preparation is not None:
            if (not self._connection_preparation_recovery or self._db.execute(
                    "SELECT version FROM connection_preparation_recovery WHERE id=1").fetchone() != ("pre_application_rebuild_v1",)):
                raise ValueError("retained_preparation_recovery_changed")
        elif self._connection_preparation_recovery:
            raise ValueError("retained_preparation_recovery_missing")

    @contextmanager
    def _transaction(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def _binding(self, client):
        from .llm import OpenRouterLLMClient
        if (not isinstance(client, OpenRouterLLMClient) or client.role not in self.allowed_roles
                or client.max_retries != 0 or client.fallback_models):
            raise ValueError("minimal allowance requires supported clients without retries or fallback")
        if (client.online_return_format not in {"native", "json_object"}
                or type(client.max_completion_tokens) is not int or client.max_completion_tokens < 64
                or type(client.temperature) not in (int, float) or not math.isfinite(client.temperature)
                or type(client.timeout) not in (int, float) or not math.isfinite(client.timeout)
                or client.timeout <= 0 or not isinstance(client.reasoning_effort, str)):
            raise ValueError("minimal allowance requires valid immutable request options")
        binding = {
            # Old records are never silently rebound to a new transport policy.
            # A mismatch requires accounting review, not a fresh zeroed ledger.
            "request_contract_version": 2, "role": client.role,
            "model": client.model,
            "online_return_format": client.online_return_format,
            "endpoint_sha256": hashlib.sha256(client.base_url.encode()).hexdigest(),
            "max_completion_tokens": client.max_completion_tokens,
            "temperature": client.temperature, "reasoning_effort": client.reasoning_effort,
            "timeout": client.timeout,
        }
        if getattr(client, "bounded_transport", False):
            binding["bounded_transport"] = True
        reflection_format = getattr(client, "reflection_return_format", "native")
        if reflection_format not in {"native", "body_json", "json_schema", "json_schema_v1", "json_schema_v2"}:
            raise ValueError("invalid_reflection_return_format")
        if reflection_format != "native":
            from ..._openrouter import DEFAULT_REFLECTION_MODEL
            from .reflection_json import reflection_request_format, reflection_body_system_prompt
            if (client.role != "reflection" or not client.bounded_transport
                    or client.model != DEFAULT_REFLECTION_MODEL):
                raise ValueError("unsupported_reflection_json_route")
            binding["reflection_return_format"] = reflection_format
            binding["reflection_format_sha256"] = hashlib.sha256(json.dumps(
                ({"system_prompt": reflection_body_system_prompt(), "format_corrections": 1}
                 if reflection_format == "body_json" else reflection_request_format(reflection_format)),
                sort_keys=True, allow_nan=False,
            ).encode()).hexdigest()
        if getattr(client, "reservation_max_price", None) is not None:
            binding["reservation_max_price"] = client.reservation_max_price
        return json.dumps(binding, sort_keys=True, allow_nan=False)

    @staticmethod
    def _validate_payload(client, payload):
        """Match the actual wire to the durable client contract before claim.

        Message content is turn-specific. Every transport option is derived
        from the bound configuration, including an explicitly chosen protocol.
        No actual-payload flag may select a weaker validation branch.
        """
        from ..._openrouter import DEFAULT_ONLINE_FAST_MODEL, DEFAULT_MEMORY_MODEL
        from .online_decision import online_decision_tool_spec, protocol_version

        if type(payload) is not dict:
            raise MinimalCallStopped("model_request_configuration_changed")
        messages = payload.get("messages")
        if (not isinstance(messages, list) or not messages or any(
                type(message) is not dict or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or type(message["content"]) is not str for message in messages)):
            raise MinimalCallStopped("model_request_configuration_changed")
        version = protocol_version(messages) if client.role == "online_fast" else None
        reflection_format = getattr(client, "reflection_return_format", "native")
        reflection_json = reflection_format in {"json_schema", "json_schema_v1", "json_schema_v2"}
        if reflection_json or reflection_format == "body_json":
            from .reflection_json import reflection_request_format, validate_reflection_request
            try:
                validate_reflection_request(client, messages)
            except ValueError:
                raise MinimalCallStopped("model_request_configuration_changed") from None
        json_transport = client.role == "online_fast" and client.online_return_format == "json_object"
        expected = {"model": client.model, "messages": messages, "temperature": client.temperature}
        if json_transport or getattr(client, "bounded_transport", False):
            provider = {"allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}
            if client.model == DEFAULT_ONLINE_FAST_MODEL:
                provider["only"] = ["alibaba"]
            elif client.model == DEFAULT_MEMORY_MODEL:
                provider["only"] = ["anthropic"]
                expected.pop("temperature", None)
            if getattr(client, "reservation_max_price", None) is not None:
                provider["max_price"] = dict(client.reservation_max_price)
            expected.update(max_tokens=client.max_completion_tokens,
                            reasoning={"enabled": False, "exclude": True},
                            stream=False, provider=provider)
            if version and json_transport:
                expected["response_format"] = {"type": "json_object"}
            elif reflection_json:
                expected["response_format"] = reflection_request_format(reflection_format)
            elif version:
                raise MinimalCallStopped("model_request_configuration_changed")
        else:
            expected.update(max_completion_tokens=client.max_completion_tokens,
                            reasoning={"effort": client.reasoning_effort, "exclude": True})
            if version:
                expected.update(tools=[online_decision_tool_spec(version)],
                                tool_choice={"type": "function", "function": {"name": "aria_turn_decision"}},
                                parallel_tool_calls=False)
        # Serialized comparison preserves JSON types: True is not 1 and a
        # floating token ceiling is not the bound integer request option.
        if json.dumps(payload, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise MinimalCallStopped("model_request_configuration_changed")

    def attach(self, client):
        binding = self._binding(client)
        if (getattr(client, "benchmark_request_control", None) is not None
                or getattr(client, "model_request_control", self) is not self):
            raise ValueError("client already has a request controller")
        with self._lock, self._transaction():
            row = self._db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone()
            if row is not None and row[0] != binding:
                raise ValueError("model configuration differs from retained allowance")
            self._db.execute("INSERT OR IGNORE INTO roles VALUES(?,?)", (client.role, binding))
        client.model_request_control = self

    @contextmanager
    def scope(self, kind, operation_id, session_id):
        if (kind not in {"turn", "review"} or any(
                not isinstance(v, str) or not _ID.fullmatch(v) for v in (operation_id, session_id))):
            raise ValueError("invalid model operation identity")
        previous = getattr(self._local, "scope", None)
        self._local.scope = {"kind": kind, "operation_id": operation_id, "session_id": session_id}
        try:
            yield
        finally:
            self._local.scope = previous

    def _snapshot(self):
        self._check_pacing_configuration()
        self._check_transport_configuration()
        self._recovery.check()
        settings = self._db.execute("SELECT max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        if settings is None or settings[:2] != (self.max_requests, self.threshold):
            raise ValueError("model allowance configuration changed")
        rows = [json.loads(row[0]) for row in self._db.execute("SELECT record FROM attempts ORDER BY sequence")]
        costs = [Decimal(str(row["usage"]["cost"])) for row in rows if "cost" in row["usage"]]
        unknown = len(rows) - len(costs)
        total = sum(costs, Decimal(0))
        recovery, allowed, interrupted = (None, set(), False)
        if self.recovery_policy is not None:
            recovery, allowed, interrupted = self._recovery.summary(rows)
        reason = settings[2]
        if self._broken or self._closed:
            reason = "ledger_unavailable"
        elif self._stop_requested.is_set():
            reason = reason or "stop_requested"
        elif any(row["status"] == "pending" for row in rows):
            reason = "unresolved_request"
        elif interrupted:
            reason = "recovery_wait_interrupted"
        elif recovery is not None and unknown >= self.recovery_policy["max_new_unknown"]:
            reason = "recovery_unknown_limit_reached"
        elif any("cost" not in row["usage"] and row["request_id"] not in allowed for row in rows):
            reason = reason or "provider_cost_unknown"
        elif any(row["status"] == "failed" and row["request_id"] not in allowed for row in rows):
            reason = reason or "request_failed"
        elif recovery is not None and recovery["logical_received"] >= self.recovery_policy["max_logical_requests"]:
            reason = "logical_request_limit_reached"
        elif len(rows) >= self.max_requests:
            reason = reason or "request_limit_reached"
        elif total >= Decimal(self.threshold):
            reason = reason or "observed_cost_threshold_reached"
        result = {"configured": True, "calls_used": len(rows), "max_requests": self.max_requests,
                "known_cost_usd": str(total) if costs else None,
                "unknown_cost_requests": unknown, "stopped": bool(reason), "stop_reason": reason,
                "max_observed_cost_usd": self.threshold, "attempts": rows,
                "cost_guard_scope": "stop_before_next_request_not_absolute_invoice_cap"}
        if recovery is not None:
            result["recovery"] = {key: value for key, value in recovery.items() if key != "logical_attempts"}
            result["recovery"].update(policy_sha256=hashlib.sha256(self._recovery.encoded.encode()).hexdigest(),
                historical_rows_rewritten=False, unknown_reserved_cost_usd=self._recovery.cost_held(rows))
        return result

    validate_recovery_records = staticmethod(validate_recovery_records)

    def snapshot(self):
        with self._lock:
            if self._closed_snapshot is not None:
                return deepcopy(self._closed_snapshot)
            try:
                return self._snapshot()
            except Exception:
                self._broken = True
                # Do not turn unreadable accounting into zero or reuse a stale
                # balance as a complete receipt. Care/read-only UI may continue.
                return {"configured": True, "calls_used": None,
                        "max_requests": self.max_requests, "known_cost_usd": None,
                        "unknown_cost_requests": None, "stopped": True,
                        "stop_reason": "ledger_unavailable", "attempts": [],
                        "max_observed_cost_usd": self.threshold,
                        "cost_guard_scope": "stop_before_next_request_not_absolute_invoice_cap"}

    def halt(self, reason="model_or_storage_failed"):
        if reason not in {"model_or_storage_failed", "model_output_failed"}:
            raise ValueError("unsupported stop reason")
        # Signal before taking the transport lock so a waiting sender can stop.
        self._stop_requested.set()
        with self._lock:
            try:
                with self._transaction():
                    self._db.execute("UPDATE settings SET halted=COALESCE(halted,?) WHERE id=1", (reason,))
            except Exception:
                self._broken = True
                raise MinimalCallStopped("ledger_write_failed") from None

    def _begin(self, client, request, *, pacing=None, expected_prior_request=None):
        if (self._connection_preparation_recovery
                and getattr(self._local, "preparation_budget", {}).get("attempts", 2) >= 2):
            raise MinimalCallStopped("logical_connection_preparation_limit")
        scope = getattr(self._local, "scope", None)
        if scope is None or (scope["kind"] == "turn") != (client.role == "online_fast"):
            raise MinimalCallStopped("operation_scope_required")
        binding = self._binding(client)
        with self._transaction():
            state = self._snapshot()
            if state["stopped"]:
                raise MinimalCallStopped(state["stop_reason"])
            if pacing is not None:
                latest = state["attempts"][-1]["request_id"] if state["attempts"] else None
                if latest != expected_prior_request:
                    raise MinimalCallStopped("request_pacing_ledger_changed")
            stored = self._db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone()
            if stored != (binding,):
                raise MinimalCallStopped("model_configuration_changed")
            previous = [row for row in state["attempts"] if row["operation_id"] == scope["operation_id"] and row["role"] == client.role]
            if (request.full_url != client.base_url.rstrip("/") + "/chat/completions"
                    or request.get_method() != "POST"):
                raise MinimalCallStopped("model_endpoint_changed")
            try:
                payload = json.loads(request.data, object_pairs_hook=_unique_request_object)
                self._validate_payload(client, payload)
                product = self._reflection_product_metadata(client, payload, previous)
                if (product.get("product_step") == "format_correction"
                        and previous[0].get("session_id") != scope["session_id"]):
                    raise MinimalCallStopped("reflection_correction_session_changed")
            except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
                raise MinimalCallStopped("model_request_configuration_changed") from None
            if (previous and product.get("product_step") != "format_correction"
                    and (self.recovery_policy is None or self._recovery.active is None)
                    and not self._allow_product_retry(client, request, previous)):
                raise MinimalCallStopped("duplicate_model_operation")
            reservation = self._reserve(client, request, payload, state)
            row = {"request_id": uuid.uuid4().hex, **scope,
                   "role": client.role, "model": client.model, "status": "pending",
                   "started_at": _stamp(), "usage": {}, "observation": {},
                   "request_sha256": hashlib.sha256(request.data).hexdigest(), "elapsed_ms": None,
                   **reservation, **product}
            if self.recovery_policy is not None:
                spec = self.policy["models"][client.model]
                providers = spec["response_providers"]
                if len(providers) != 1 or providers[0] not in {"Alibaba", "Anthropic"}:
                    raise MinimalCallStopped("bounded_recovery_provider_not_explicit")
                row["expected_provider"] = providers[0]
                self._recovery.claim(row, previous)
            if pacing is not None:
                row["pacing"] = pacing
            self._db.execute("INSERT INTO attempts(request_id,record) VALUES(?,?)",
                             (row["request_id"], json.dumps(row, ensure_ascii=False, allow_nan=False)))
        return row

    def _allow_product_retry(self, client, request, previous):
        """Strict/experimental guards never inherit product retry permission."""
        return False

    @staticmethod
    def _reflection_product_metadata(client, payload, previous):
        """Bind one correction to the first received shape error, not a retry."""
        if getattr(client, "reflection_return_format", "native") != "body_json":
            return {}
        from .reflection_json import reflection_product_step, reflection_correction_message
        messages = payload["messages"]
        step = reflection_product_step(messages)
        digest = hashlib.sha256(json.dumps(messages[:2], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        result = {"product_step": step, "reflection_input_sha256": digest}
        if step == "format_correction":
            if (len(previous) != 1 or previous[0].get("status") != "received"
                    or previous[0].get("product_step") != "initial"
                    or previous[0].get("reflection_input_sha256") != digest):
                raise MinimalCallStopped("invalid_reflection_correction_antecedent")
            first = previous[0]
            issues = first.get("reflection_body", {}).get("correction_issues")
            if not issues or messages[2] != reflection_correction_message(issues):
                raise MinimalCallStopped("reflection_correction_diagnostic_mismatch")
            result["format_parent_request_id"] = first["request_id"]
        return result

    @staticmethod
    def _observe_reflection_body(client, row, data):
        if getattr(client, "reflection_return_format", "native") == "body_json":
            from .reflection_json import reflection_body_observation
            row["reflection_body"] = reflection_body_observation(data)

    def _reserve(self, client, request, payload, state):
        return {}

    def _record_receipt(self, client, row, data):
        pass

    def _revalidate_pending_application(self, client, request, row):
        """Read-only check after a connection wait; never make a second claim."""
        if self._stop_requested.is_set() or self._broken or self._closed:
            raise MinimalCallStopped("request_stop_requested")
        if self._pre_application_check is not self._pre_application_check_original:
            raise MinimalCallStopped("pre_application_check_changed")
        self._check_transport_configuration()
        self._check_pacing_configuration()
        self._recovery.check()
        self._check_policy()
        if self._db.execute("SELECT max_requests,threshold,halted FROM settings WHERE id=1").fetchone() != (
                self.max_requests, self.threshold, None):
            raise MinimalCallStopped("model_allowance_changed")
        if self._db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone() != (self._binding(client),):
            raise MinimalCallStopped("model_configuration_changed")
        if (request.full_url != client.base_url.rstrip("/") + "/chat/completions" or request.get_method() != "POST"
                or hashlib.sha256(request.data).hexdigest() != row["request_sha256"]):
            raise MinimalCallStopped("model_request_configuration_changed")
        self._validate_payload(client, json.loads(request.data, object_pairs_hook=_unique_request_object))
        state = self._snapshot()
        pending = [item for item in state["attempts"] if item["status"] == "pending"]
        if (len(pending) != 1 or pending[0]["request_id"] != row["request_id"]
                or pending[0]["request_sha256"] != row["request_sha256"]
                or pending[0].get("reservation") != row.get("reservation")
                or Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) > Decimal(self.threshold)):
            raise MinimalCallStopped("inflight_allowance_changed")
        # The entry owns source/human-authorization validation. Missing callback
        # is rejected at construction, not treated as permission.
        self._pre_application_check()
        if self._stop_requested.is_set():
            raise MinimalCallStopped("request_stop_requested")

    def _finish(self, row):
        with self._transaction():
            self._db.execute("UPDATE attempts SET record=? WHERE request_id=?",
                             (json.dumps(row, ensure_ascii=False, allow_nan=False), row["request_id"]))
            if self.recovery_policy is not None:
                self._recovery.finish(row)

    def request_json(self, client, request):
        if self.recovery_policy is None:
            return self._request_once(client, request)
        # A single logical position can create additional retained physical
        # attempts only through this explicit, globally bounded policy.
        with self._lock:
            self._local.logical_started = self._recovery_clock()
            self._local.preparation_budget = {"attempts": 0}
            while True:
                try:
                    return self._request_once(client, request)
                except MinimalCallStopped:
                    try:
                        if not self._recovery.prepare_wait(client, request):
                            self._recovery.active = None
                            self.halt()
                            raise
                    except MinimalCallStopped:
                        raise
                    except Exception:
                        self._recovery.active = None
                        self.halt()
                        raise MinimalCallStopped("bounded_recovery_wait_failed") from None

    def _request_once(self, client, request):
        # One transport attempt. Other processes see the committed pending row
        # and cannot start another request while this one is unresolved.
        with self._lock:
            try:
                pacing = None
                previous_request = None
                if self._pacing_interval:
                    state = self._snapshot()
                    if state["stopped"]:
                        raise MinimalCallStopped(state["stop_reason"])
                    previous_request = state["attempts"][-1]["request_id"] if state["attempts"] else None
                    try:
                        if previous_request is not None and previous_request != self._last_network_request_id:
                            # Another guard/process may have completed a call.
                            # Wait conservatively; do not compare its clock.
                            deadline = monotonic_value(self._pacing_clock) + self._pacing_interval
                        elif self._last_network_started is not None:
                            deadline = self._last_network_started + self._pacing_interval
                        elif self._reopen_pacing_required:
                            deadline = monotonic_value(self._pacing_clock) + self._pacing_interval
                        else:
                            deadline = None
                        waited = wait_until(deadline, monotonic=self._pacing_clock, sleep=self._pacing_sleep,
                                            cancelled=self._stop_requested.is_set)
                    except BaseException as error:
                        self.halt()
                        if not isinstance(error, Exception):
                            raise
                        raise MinimalCallStopped("request_pacing_wait_failed") from None
                    pacing = {"min_interval_seconds": self._pacing_interval,
                              "wait_ms": round(waited, 3), "start_gap_ms": None,
                              "scope": "one_guard_all_roles", "clock": "monotonic"}
                # No transaction is held during waiting. This claim repeats all
                # allowance/configuration checks before recording an attempt.
                row = self._begin(client, request, pacing=pacing, expected_prior_request=previous_request)
            except MinimalCallStopped:
                client._update_diagnostic(error_type="minimal_call_stopped")
                raise
            except Exception:
                self._broken = True
                raise MinimalCallStopped("ledger_preflight_failed") from None
            started = time.perf_counter()
            phase, headers = "local_pacing_check", None
            opener_invoked, response_returned = False, False
            trace = TransportTrace(diagnostics=self._transport_diagnostics) if self._observe_transport else None
            try:
                if self._connection_preparation_recovery:
                    preparation = PreparationRecovery(check=lambda: self._revalidate_pending_application(client, request, row),
                        timeout=client.timeout, monotonic=self._pacing_clock, budget=self._local.preparation_budget)
                    trace = TransportTrace(diagnostics=True, preparation_recovery=preparation)
                if pacing is not None:
                    current = monotonic_value(self._pacing_clock)
                    if self._last_network_started is not None:
                        gap = current - self._last_network_started
                        if gap < self._pacing_interval:
                            raise MinimalCallStopped("request_pacing_clock_changed")
                        if previous_request == self._last_network_request_id:
                            pacing["start_gap_ms"] = round(gap * 1000, 3)
                    self._last_network_started, self._reopen_pacing_required = current, False
                    self._last_network_request_id = row["request_id"]
                phase = "local_stop_check"
                if self._stop_requested.is_set():
                    raise MinimalCallStopped("request_stop_requested")
                phase = "local_bookkeeping"
                client._record_network_request()
                phase, opener_invoked = "pre_response_open", True
                if self.recovery_policy is not None:
                    row["http_dispatch_started"] = True
                response = (open_observed(request, timeout=client.timeout, trace=trace) if trace is not None
                            else open_unredirected(request, timeout=client.timeout))
                phase, response_returned = "response_context", True
                with response as response:
                    status = getattr(response, "status", None)
                    row["http_status"] = status if type(status) is int else None
                    client._update_diagnostic(http_status=row["http_status"], error_type=None)
                    headers = capture_response_headers(response)
                    phase = "read"
                    raw = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
                    if not isinstance(raw, bytes) or len(raw) > MAX_MODEL_RESPONSE_BYTES:
                        row["validation_error"] = "model_response_body_limit"
                        raise MinimalCallStopped("model_response_body_limit")
                phase = "decode"
                decoded = raw.decode("utf-8")
                phase = "parse"
                duplicate_fields = set()
                def object_pairs(pairs):
                    obj = {}
                    for key, value in pairs:
                        if key in obj:
                            duplicate_fields.add(key)
                        obj[key] = value
                    return obj
                data = json.loads(decoded, object_pairs_hook=object_pairs)
                # HTTP bytes and the outer JSON envelope have been received.
                # Any later schema/content error is not a transport interruption.
                phase = "envelope_validation"
                # Preserve an unambiguous public usage envelope even when a
                # different response field is malformed. Duplicate billing
                # fields remain unknown; picking either value could undercount.
                billing_fields = {"usage", "cost", "prompt_tokens", "completion_tokens", "total_tokens",
                                  "completion_tokens_details", "reasoning_tokens"}
                row["usage"] = {} if duplicate_fields & billing_fields else safe_usage(data)
                observation_data = {"usage": row["usage"]} if duplicate_fields else data
                row["observation"] = observe_success(observation_data, header_observation=headers)
                if duplicate_fields:
                    row["validation_error"] = "duplicate_response_field"
                    raise MinimalCallStopped("duplicate_response_field")
                if not isinstance(data, dict):
                    row["validation_error"] = "non_object_response"
                    raise ValueError("non_object_response")
                phase = "receipt_validation"
                self._observe_reflection_body(client, row, data)
                self._record_receipt(client, row, data)
                row["status"] = "received"
                return data
            except urllib.error.HTTPError as exc:
                if not opener_invoked:
                    _record_failure_stage(row, phase, exc)
                    row["transport_failure"] = transport_failure(exc, opener_invoked=False, response_returned=False,
                        diagnostics=self._transport_diagnostics)
                    client._update_diagnostic(error_type="minimal_request_failed")
                    raise MinimalCallStopped("request_failed") from None
                row["http_status"] = exc.code
                row["observation"], row["usage"], failure_metadata = self._observe_http_failure(exc)
                row.update(failure_metadata)
                row.update(failure_phase="http_status", failure_code="http_error")
                client._update_diagnostic(http_status=exc.code, error_type="http_error")
                raise MinimalCallStopped("request_failed") from None
            except Exception as exc:
                if phase in {"local_pacing_check", "local_stop_check", "local_bookkeeping", "pre_response_open", "response_context"}:
                    row["transport_failure"] = transport_failure(exc, opener_invoked=opener_invoked,
                        response_returned=response_returned, diagnostics=self._transport_diagnostics)
                if headers is not None and phase in {"read", "decode", "parse"}:
                    row["observation"] = observe_interrupted(headers, phase=phase, error=exc)
                _record_failure_stage(row, phase, exc)
                client._update_diagnostic(error_type="minimal_request_failed")
                raise MinimalCallStopped("request_failed") from None
            finally:
                if trace is not None:
                    row["transport_events"] = trace.snapshot()
                if row["status"] == "pending":
                    row["status"] = "failed"
                row["finished_at"] = _stamp()
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                if self.recovery_policy is not None:
                    row["logical_elapsed_ms"] = round((self._recovery_clock() - self._local.logical_started) * 1000, 3)
                try:
                    self._finish(row)
                except Exception:
                    self._broken = True
                    raise MinimalCallStopped("ledger_result_save_failed") from None

    def close(self):
        self._stop_requested.set()
        with self._lock:
            if not self._closed:
                self._closed_snapshot = self.snapshot()
                self._closed_snapshot.update(stopped=True, stop_reason="ledger_closed")
                self._db.close()
                self._closed = True
