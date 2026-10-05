"""Opt-in E1b-R accounting: first content, two no-content retries, fixed holds.

Historical ledgers and the default product allowance are never migrated here.
The formal entry supplies frozen authorization checks, prices and old holds.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from .minimal_calls import MinimalCallStopped, _stamp, _unique_request_object
from .online_decision import protocol_version
from .online_json import parse_json_decision
from .reserved_calls import ReservedCallGuard, validate_reservation_policy
from .request_pacing import monotonic_value, wait_until
from .response_observability import (
    _nonnegative_number, error_envelope_audit, error_envelope_shape, observe_http_error,
    validate_error_envelope_audit, validate_observation,
)
from .transport_failure import validate_transport_failure

SCHEMA = "e1br_no_content_retry_v1"
MAX_REQUESTS = 3153
_SECONDARY_FAILURES = frozenset({"disabled_reasoning_returned", "invalid_or_truncated_response",
                               "invalid_response_message"})


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key(row):
    return row["operation_id"], row["session_id"], row["role"]


def _upstream_error_only(data):
    # An error plus choices/content is never evidence of no model answer.
    shape = error_envelope_shape(data)
    return (shape["object"] and shape["error_object"] and shape["finite"]
            and not shape["unknown_top_key_count"] and not shape["model_content_keys"]
            and shape["user_id_status"] != "invalid" and shape["error_code_status"] == "eligible")


def no_content_failure(row):
    if (row.get("status") != "failed" or row.get("validation_error") not in {None, "provider_error_envelope"}):
        return False
    evidence = row.get("no_content_evidence")
    if "error_envelope_audit" in row:
        try:
            validate_error_envelope_audit(row["error_envelope_audit"])
        except (ValueError, TypeError, KeyError):
            return False
        if row["error_envelope_audit"]["rejection_reasons"]:
            return False
    if type(evidence) is dict and evidence.get("schema") == "e1br_no_content_v1":
        if evidence.get("kind") == "http_error":
            return (set(evidence) == {"schema", "kind", "http_status", "body_status"}
                    and type(evidence["http_status"]) is int
                    and (evidence["http_status"] == 429 or 500 <= evidence["http_status"] <= 599)
                    and evidence["http_status"] == row.get("http_status")
                    and evidence["body_status"] in {"empty", "parsed"}
                    and row.get("failure_phase") == "http_status"
                    and row.get("observation", {}).get("body_status") == evidence["body_status"]
                    and not row["observation"].get("body_close_failed"))
        if evidence.get("kind") == "upstream_error":
            return (set(evidence) == {"schema", "kind", "error_code"}
                    and type(evidence["error_code"]) is int
                    and (evidence["error_code"] in {400, 408, 429} or 500 <= evidence["error_code"] <= 599)
                    and row.get("validation_error") == "provider_error_envelope"
                    and row.get("http_status") == 200)
    failure = row.get("transport_failure", {})
    return (row.get("failure_phase") == "pre_response_open" and not row.get("http_status")
            and failure.get("exception_kind") == "timeout"
            and failure.get("opener_call") == "invoked" and failure.get("response_object") == "not_observed")


def secondary_failure(row, *, no_content_check=None):
    check = no_content_failure if no_content_check is None else no_content_check
    return row.get("role") == "reflection" and row.get("status") == "failed" and (
        check(row) or ("cost" in row.get("usage", {})
            and row.get("validation_error") in _SECONDARY_FAILURES))


def logical_attempts(state):
    """One terminal physical row per product step; keep the correction separate."""
    groups = {}
    for row in state["attempts"]:
        key = (*_key(row), row["product_step"])
        groups[key] = row
    return list(groups.values())


def contract_value(*, stage, registration_sha256, max_requests=MAX_REQUESTS):
    return {"schema": SCHEMA, "stage": stage, "registration_sha256": registration_sha256,
        "infrastructure_retries": 2, "format_corrections": 1, "max_requests": max_requests,
        "unknown_policy": "retain_each_full_reservation", "secondary_role": "reflection"}


def validate_e1br_ledger(path, *, stage, registration_sha256, max_requests=MAX_REQUESTS,
                        reservation_policy=None):
    """Read-only audit of the immutable policy and physical retry relationships."""
    expected = json.dumps(contract_value(stage=stage, registration_sha256=registration_sha256,
                                         max_requests=max_requests), sort_keys=True)
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT contract FROM e1br_policy WHERE id=1").fetchone() != (expected,):
            raise ValueError("e1br_contract_changed")
        rows = [json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")]
        policy = validate_reservation_policy(json.loads(db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()[0]))
    if reservation_policy is not None and policy != validate_reservation_policy(reservation_policy):
        raise ValueError("e1br_frozen_reservation_policy_changed")
    validate_e1br_records(rows, stage=stage, registration_sha256=registration_sha256)
    validate_e1br_reservations(rows, policy)
    return rows


def validate_e1br_reservations(rows, policy):
    policy_sha = hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    for row in rows:
        reservation, spec = row["reservation"], policy["models"][row["model"]]
        bound, output = reservation["input_token_plan_bound"], reservation["output_token_limit"]
        if (type(bound) is not int or not 8192 <= bound <= spec["max_input_tokens"]
                or type(output) is not int or not 64 <= output <= spec["max_output_tokens"]
                or reservation["policy_sha256"] != policy_sha
                or Decimal(reservation["reserved_cost_usd"]) != (
                    Decimal(bound) * Decimal(spec["input_usd_per_million"])
                    + Decimal(output) * Decimal(spec["output_usd_per_million"])) / 1000000):
            raise ValueError("invalid_e1br_retained_reservation")


def validate_e1br_records(rows, *, stage, registration_sha256, no_content_check=None):
    check = no_content_failure if no_content_check is None else no_content_check
    groups = defaultdict(list)
    for row in rows:
        if (row.get("e1b_stage") != stage or row.get("registration_sha256") != registration_sha256
                or row.get("logical_operation_id") != row.get("operation_id")
                or row.get("product_step") not in {"initial", "format_correction"}
                or row.get("status") not in {"pending", "received", "failed"}):
            raise ValueError("invalid_e1br_attempt")
        # These are the already-sanitized, flattened ledger fields. Feeding
        # them to the wire-envelope extractor loses reasoning_tokens (even 0).
        usage = row.get("usage")
        if (type(usage) is not dict or not set(usage) <= {
                "prompt_tokens", "completion_tokens", "total_tokens", "cost", "reasoning_tokens"}
                or any(not _nonnegative_number(value) for value in usage.values())
                or type(row.get("infrastructure_retry_index")) is not int
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("request_sha256", ""))):
            raise ValueError("invalid_e1br_accounting")
        if row.get("observation"):
            validate_observation(row["observation"])
        if "error_envelope_audit" in row:
            audit = row["error_envelope_audit"]
            validate_error_envelope_audit(audit)
            expected_source = "http_error_bytes" if row.get("failure_phase") == "http_status" else "parsed_response_canonical"
            if (audit["http_status"] != row.get("http_status") or audit["source"] != expected_source
                    or audit["body_status"] != row.get("observation", {}).get("body_status")
                    or audit["close_ok"] == bool(row.get("observation", {}).get("body_close_failed"))
                    or "no_content_evidence" in row and audit["rejection_reasons"]):
                raise ValueError("e1br_error_envelope_audit_binding_changed")
        if "transport_failure" in row:
            validate_transport_failure(row["transport_failure"])
        prior = groups[_key(row)]
        same = [item for item in prior if item["product_step"] == row["product_step"]]
        retries = sum("infrastructure_retry_of_request_id" in item for item in prior)
        if same:
            parent = same[-1]
            if (not check(parent) or retries >= 2
                    or row.get("infrastructure_retry_of_request_id") != parent["request_id"]
                    or row.get("request_sha256") != parent["request_sha256"]
                    or row.get("model") != parent["model"]
                    or row.get("infrastructure_retry_index") != retries + 1):
                raise ValueError("invalid_e1br_infrastructure_retry")
        elif "infrastructure_retry_of_request_id" in row or row.get("infrastructure_retry_index") != retries:
            raise ValueError("invalid_e1br_first_attempt")
        if row["product_step"] == "format_correction":
            initial = [item for item in prior if item["product_step"] == "initial" and item["status"] == "received"]
            if (row["role"] != "reflection" or len(initial) != 1
                    or row.get("format_parent_request_id") != initial[0]["request_id"]
                    or row.get("reflection_input_sha256") != initial[0].get("reflection_input_sha256")
                    or not initial[0].get("reflection_body", {}).get("correction_issues")):
                raise ValueError("invalid_e1br_format_correction")
        elif any(item["product_step"] == "format_correction" for item in prior):
            raise ValueError("e1br_initial_after_correction")
        prior.append(row)
    return groups

class E1BRCallGuard(ReservedCallGuard):
    allowed_roles = {"online_fast", "memory", "reflection"}

    def __init__(self, path, *, max_requests=MAX_REQUESTS, max_observed_cost_usd,
                 reservation_policy, stage=None, registration_sha256=None,
                 min_interval_seconds=10, authority_check, **kwargs):
        self.stage = stage or getattr(self, "stage", "main")
        self.registration_sha256 = registration_sha256 or getattr(self, "registration_sha256", None)
        if (self.stage != "main" or not re.fullmatch(r"[a-f0-9]{64}", self.registration_sha256 or "")
                or not callable(authority_check) or min_interval_seconds != 10
                or type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS):
            raise ValueError("invalid_e1br_contract")
        self._authority = self._authority_original = authority_check
        self._contract = json.dumps(contract_value(stage=self.stage,
            registration_sha256=self.registration_sha256, max_requests=max_requests), sort_keys=True)
        prior = Path(path)
        if prior.exists():
            with sqlite3.connect(prior.resolve().as_uri() + "?mode=ro", uri=True) as db:
                try:
                    saved = db.execute("SELECT contract FROM e1br_policy WHERE id=1").fetchone()
                except sqlite3.DatabaseError:
                    raise ValueError("retained_ledger_not_e1br") from None
                if saved != (self._contract,):
                    raise ValueError("retained_e1br_contract_changed")
        # Callers cannot silently weaken observation, pacing or write checks.
        for key, expected in {"observe_transport": True, "transport_diagnostics": True,
                              "connection_preparation_recovery": True, "recovery_policy": None}.items():
            if key in kwargs and kwargs.pop(key) != expected:
                raise ValueError("invalid_e1br_transport_contract")
        super().__init__(path, max_requests=max_requests, max_observed_cost_usd=max_observed_cost_usd,
            reservation_policy=reservation_policy, min_interval_seconds=10,
            observe_transport=True, transport_diagnostics=True, connection_preparation_recovery=True,
            pre_application_check=self._check_authority, **kwargs)
        try:
            with self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_policy (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO e1br_policy VALUES(1,?)", (self._contract,))
            self._snapshot()
        except BaseException:
            self._db.close()
            raise

    def _preparation_policy_supported(self):
        return self.recovery_policy is None and hasattr(self, "_contract")

    def _check_authority(self):
        if self._authority is not self._authority_original:
            raise MinimalCallStopped("e1br_authority_callback_changed")
        self._authority()

    def _validate_history(self, rows):
        return validate_e1br_records(rows, stage=self.stage, registration_sha256=self.registration_sha256,
                                      no_content_check=self._no_content_failure)

    def _no_content_failure(self, row):
        return no_content_failure(row)

    def _secondary_failure(self, row):
        return secondary_failure(row, no_content_check=self._no_content_failure)

    def terminal_primary_failure(self, row):
        """Versioned experiment extension; ordinary E1b-R never accepts one."""
        return False

    def _snapshot(self):
        if self._db.execute("SELECT contract FROM e1br_policy WHERE id=1").fetchone() != (self._contract,):
            raise ValueError("e1br_contract_changed")
        state = super()._snapshot()
        rows = state["attempts"]
        groups = self._validate_history(rows)
        validate_e1br_reservations(rows, self.policy)
        reason = self._db.execute("SELECT halted FROM settings WHERE id=1").fetchone()[0]
        if self._broken or self._closed:
            reason = "ledger_unavailable"
        elif self._stop_requested.is_set():
            reason = reason or "stop_requested"
        elif any(row["status"] == "pending" for row in rows):
            reason = reason or "unresolved_request"
        elif any("cost" not in row["usage"] and not self._no_content_failure(row) for row in rows):
            reason = reason or "provider_cost_unknown"
        elif any(row["status"] == "failed" and not self._no_content_failure(row) and not self._secondary_failure(row)
                 and not self.terminal_primary_failure(row) for row in rows):
            reason = reason or "request_failed"
        elif any(group[-1]["status"] == "failed" and group[-1]["role"] != "reflection"
                 and not self.terminal_primary_failure(group[-1])
                 and sum("infrastructure_retry_of_request_id" in r for r in group) >= 2 for group in groups.values()):
            reason = reason or "infrastructure_retries_exhausted"
        elif len(rows) >= self.max_requests:
            reason = reason or "request_limit_reached"
        elif Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) >= Decimal(self.threshold):
            reason = reason or "inclusive_cost_limit_reached"
        state.update(stopped=bool(reason), stop_reason=reason,
            e1br={"schema": SCHEMA, "infrastructure_retries_used": sum("infrastructure_retry_of_request_id" in r for r in rows),
                  "logical_product_steps": len(logical_attempts(state)),
                  "reflection_secondary_failures": sum(self._secondary_failure(r) for r in logical_attempts(state))})
        return state

    def _begin(self, client, request, *, pacing=None, expected_prior_request=None):
        self._check_authority()
        scope = getattr(self._local, "scope", None)
        if scope is None or (scope["kind"] == "turn") != (client.role == "online_fast"):
            raise MinimalCallStopped("operation_scope_required")
        with self._transaction():
            state = self._snapshot()
            if state["stopped"]:
                raise MinimalCallStopped(state["stop_reason"])
            if pacing is not None and (state["attempts"][-1]["request_id"] if state["attempts"] else None) != expected_prior_request:
                raise MinimalCallStopped("request_pacing_ledger_changed")
            if self._db.execute("SELECT binding FROM roles WHERE role=?", (client.role,)).fetchone() != (self._binding(client),):
                raise MinimalCallStopped("model_configuration_changed")
            if request.full_url != client.base_url.rstrip("/") + "/chat/completions" or request.get_method() != "POST":
                raise MinimalCallStopped("model_endpoint_changed")
            payload = json.loads(request.data, object_pairs_hook=_unique_request_object)
            self._validate_payload(client, payload)
            prior = [r for r in state["attempts"] if r["operation_id"] == scope["operation_id"] and r["role"] == client.role]
            if any(r["session_id"] != scope["session_id"] for r in prior):
                raise MinimalCallStopped("operation_session_changed")
            received_initial = [r for r in prior if r["product_step"] == "initial" and r["status"] == "received"]
            product = {"product_step": "initial", **self._reflection_product_metadata(client, payload, received_initial)}
            same = [r for r in prior if r["product_step"] == product["product_step"]]
            count = sum("infrastructure_retry_of_request_id" in r for r in prior)
            extra = {}
            if same:
                parent = same[-1]
                if not self._no_content_failure(parent) or count >= 2 or parent["request_sha256"] != hashlib.sha256(request.data).hexdigest():
                    raise MinimalCallStopped("duplicate_model_operation")
                extra["infrastructure_retry_of_request_id"] = parent["request_id"]
                count += 1
            elif prior and product["product_step"] != "format_correction":
                raise MinimalCallStopped("duplicate_model_operation")
            # A primary no-content failure must be resolved at its own position.
            for group in self._validate_history(state["attempts"]).values():
                last = group[-1]
                if (last["role"] != "reflection" and last["status"] == "failed"
                        and not self.terminal_primary_failure(last)
                        and _key(last) != (scope["operation_id"], scope["session_id"], client.role)):
                    raise MinimalCallStopped("unresolved_primary_position")
            reservation = self._reserve(client, request, payload, state)
            row = {"request_id": uuid.uuid4().hex, **scope, "logical_operation_id": scope["operation_id"],
                "role": client.role, "model": client.model, "status": "pending", "started_at": _stamp(),
                "usage": {}, "observation": {}, "request_sha256": hashlib.sha256(request.data).hexdigest(),
                "elapsed_ms": None, **reservation, **product, **extra, "infrastructure_retry_index": count}
            if "infrastructure_retry_of_request_id" in extra:
                row["retry_after_wait_ms"] = getattr(self._local, "retry_after_wait_ms", 0)
            if pacing is not None:
                row["pacing"] = pacing
            self._db.execute("INSERT INTO attempts(request_id,record) VALUES(?,?)", (row["request_id"], json.dumps(row, ensure_ascii=False, allow_nan=False)))
            rebuilt = any(len(r.get("transport_events", {}).get("connection_preparations", [])) == 2 for r in prior)
            self._local.preparation_budget = {"attempts": int(rebuilt)}
        return row

    def _reserve(self, client, request, payload, state):
        value = super()._reserve(client, request, payload, state)
        return {**value, "e1b_stage": self.stage, "registration_sha256": self.registration_sha256,
                "reply_return_channel": "json_object" if payload.get("response_format") == {"type": "json_object"} else "text",
                "online_decision_version": protocol_version(payload["messages"]) if client.role == "online_fast" else None}

    def _observe_http_failure(self, exc):
        evidence, audit = {}, {}
        observation, usage = observe_http_error(exc, no_content_evidence=evidence, envelope_audit=audit)
        return observation, usage, {"error_envelope_audit": audit,
            **({"no_content_evidence": evidence} if evidence else {})}

    def _finish(self, row):
        # Error envelopes can contain billable usage too. They never bypass
        # the same reservation overrun checks as a successful answer.
        if row["status"] == "failed":
            usage, reservation = row["usage"], row["reservation"]
            reason = None
            if "cost" in usage and Decimal(str(usage["cost"])) > Decimal(reservation["reserved_cost_usd"]):
                reason = "reservation_cost_exceeded"
            elif (usage.get("prompt_tokens", 0) > reservation["input_token_plan_bound"]
                    or usage.get("completion_tokens", 0) > reservation["output_token_limit"]):
                reason = "reservation_token_bound_exceeded"
            if reason:
                row["validation_error"] = row["failure_code"] = reason
                row.pop("no_content_evidence", None)
        super()._finish(row)

    def _record_receipt(self, client, row, data):
        if "error" in data:
            # This is a canonical parsed envelope, explicitly not original HTTP
            # bytes. The ordinary envelope validator already rejected duplicate
            # keys; the audit only retains hashes, counts and fixed categories.
            captured = json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")
            row["error_envelope_audit"] = error_envelope_audit(data,
                source="parsed_response_canonical", http_status=200, body_status="parsed",
                captured=captured, read_complete=True, close_ok=True)
        if _upstream_error_only(data):
            row["no_content_evidence"] = {"schema": "e1br_no_content_v1", "kind": "upstream_error", "error_code": data["error"]["code"]}
        super()._record_receipt(client, row, data)
        content = data["choices"][0]["message"]["content"]
        row["visible_content_sha256"] = _sha(content)
        channel = client.role == "online_fast" and row["reply_return_channel"] == "json_object"
        raw_body = channel or getattr(client, "reflection_return_format", "native") == "body_json"
        row["client_return_content_sha256"] = _sha(content if raw_body else content.strip())
        if client.role == "online_fast":
            reply = content.strip()
            if channel:
                try:
                    parsed = parse_json_decision(content, row["online_decision_version"])
                except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                    row["validation_error"] = "online_decision_schema_invalid"
                    raise MinimalCallStopped("online_decision_schema_invalid") from None
                # parse_json_decision has already rejected duplicates/nonfinite
                # numbers and validated the complete product response schema.
                reply = parsed.response if hasattr(parsed, "response") else json.loads(content)["response"]
            row["model_reply_sha256"] = _sha(reply.strip())

    def request_json(self, client, request):
        with self._lock:
            self._local.secondary_terminal = None
            self._local.retry_after_wait_ms = 0
            while True:
                before = self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                try:
                    return self._request_once(client, request)
                except MinimalCallStopped:
                    state = self.snapshot()
                    rows = state["attempts"]
                    # Only a newly finished, durable row can authorize a retry
                    # or settle a secondary failure. Preflight errors cannot.
                    if state["stopped"] or len(rows) != before + 1:
                        raise
                    last = rows[-1]
                    group = [r for r in rows if _key(r) == _key(last)]
                    retries = sum("infrastructure_retry_of_request_id" in r for r in group)
                    if self._no_content_failure(last) and retries < 2:
                        observation = last.get("observation", {})
                        if observation.get("retry_after_status") == "accepted":
                            try:
                                self._check_authority()
                                seconds = observation["retry_after_seconds"]
                                deadline = monotonic_value(self._pacing_clock) + seconds
                                self._local.retry_after_wait_ms = wait_until(deadline,
                                    monotonic=self._pacing_clock, sleep=self._pacing_sleep,
                                    cancelled=self._stop_requested.is_set)
                                self._check_authority()
                            except BaseException:
                                self.halt()
                                raise
                        continue
                    if self._secondary_failure(last):
                        self._local.secondary_terminal = (id(client), last["request_id"])
                    raise

    def reflection_failure_is_terminal(self, client):
        marker = getattr(self._local, "secondary_terminal", None)
        if marker is None or marker[0] != id(client) or client.role != "reflection":
            return False
        state = self.snapshot()
        return (not state["stopped"] and bool(state["attempts"])
                and state["attempts"][-1]["request_id"] == marker[1]
                and self._secondary_failure(state["attempts"][-1]))
