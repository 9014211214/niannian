"""Product-only recovery, separate from frozen experimental stop contracts.

Physical request receipts retain their original accounting. Logical operations
and explicit owner recoveries are append-only events; three failed operations,
not three transport attempts, pause the product. Unknown charges, uncertain
storage, and budget limits cannot be resumed through this interface. An explicit
fresh-ledger contract can wait after strict no-content 429s while retaining
unknown-charge holds; it never changes the historical default contract.
The inherited default transport rejects HTTP redirects, and product recovery
does not retry these failures or release their unknown-charge reservations.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from decimal import Decimal
import hashlib
import json
import sqlite3
import uuid

from .minimal_calls import MinimalCallStopped, _ID, _stamp
from .reserved_calls import ReservedCallGuard
from .response_observability import observe_http_error, validate_error_envelope_audit
from .rate_limit import retry_wait_plan, perform_retry_wait
import math

_VERSION = "product_call_recovery_v1"
_HOLD_VERSION = "product_call_recovery_v2_unknown_429_hold"
_HOLD_SCHEMA = "unknown_429_hold_recovery_v1"


def _hold_contract(value):
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {
            "schema", "prior_unknown_429_count", "prior_evidence_sha256", "source_manifest_sha256"}
            or value["schema"] != _HOLD_SCHEMA
            or type(value["prior_unknown_429_count"]) is not int
            or not 0 <= value["prior_unknown_429_count"] <= 6
            or any(type(value[key]) is not str or len(value[key]) != 64
                   or any(c not in "0123456789abcdef" for c in value[key])
                   for key in ("prior_evidence_sha256", "source_manifest_sha256"))):
        raise ValueError("invalid_unknown_429_recovery_contract")
    return json.dumps(value, sort_keys=True, allow_nan=False)

_OUTPUT_FAILURES = frozenset({
    "invalid_or_truncated_response", "invalid_response_message",
    "online_content_invalid_json", "online_content_duplicate_key", "online_content_non_finite",
    "online_content_not_object", "online_response_missing", "online_response_not_string",
    "online_response_empty", "online_decision_schema_invalid", "content_json_invalid",
    "safety_review_schema_invalid",
})
_BUDGET_FAILURES = frozenset({"pre_send_budget_exceeded", "request_limit_reached",
                              "observed_cost_threshold_reached"})
_LOCAL_REJECTIONS = {
    "reservation_token_limit_exceeded": "input_or_output_limit",
    "reservation_provider_mismatch": "configuration",
    "model_configuration_changed": "configuration",
    "model_endpoint_changed": "configuration",
    "model_request_configuration_changed": "configuration",
    "operation_scope_required": "identity",
    "operation_session_changed": "identity",
    "duplicate_model_operation": "identity",
    "reflection_correction_session_changed": "identity",
    "invalid_reflection_correction_antecedent": "identity",
    "reflection_correction_diagnostic_mismatch": "identity",
    "e1cv2_position_required": "identity",
    "e1cv2_unregistered_position": "identity",
    "e1cv2_operation_identity_changed": "identity",
    "e1cv2_position_rebound": "identity",
    "e1c_frozen_source_or_history_changed": "authority",
    "e1c_frozen_source_archive_changed": "authority",
    "e1c_frozen_source_archive_members_changed": "authority",
    "e1cv2_history_original_changed": "authority",
    "local_preflight_rejected": "preflight",
}


class ProductCallGuard(ReservedCallGuard):
    """Product recovery, also usable by separately registered product experiments.

    Frozen experiments retain their originally selected guard and contract.
    """

    allowed_roles = {"online_fast", "memory", "reflection", "safety_review"}

    def __init__(self, path, *, max_requests, max_observed_cost_usd, reservation_policy,
                 min_interval_seconds=0, observe_transport=False, transport_diagnostics=False,
                 unknown_429_recovery=None, recovery_pre_send_check=None):
        self._active_operation_tokens = set()
        self._hold_contract_json = _hold_contract(unknown_429_recovery)
        self._product_version = _HOLD_VERSION if self._hold_contract_json is not None else _VERSION
        if ((self._hold_contract_json is not None) != callable(recovery_pre_send_check)
                or self._hold_contract_json is None and recovery_pre_send_check is not None):
            raise ValueError("unknown_429_recovery_requires_source_check")
        self._recovery_pre_send_check = self._recovery_pre_send_check_original = recovery_pre_send_check
        # Do not silently reinterpret a retained strict experiment/legacy ledger.
        # Even a known charge cannot authorize changing that ledger's contract.
        from pathlib import Path
        if Path(path).exists():
            with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
                present = db.execute("SELECT name FROM sqlite_master WHERE name='product_recovery'").fetchone()
                if present:
                    if db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone() != (self._product_version,):
                        raise ValueError("product_recovery_contract_changed")
                    if self._hold_contract_json is not None and db.execute(
                            "SELECT contract FROM product_hold_recovery WHERE id=1").fetchone() != (self._hold_contract_json,):
                        raise ValueError("unknown_429_recovery_contract_changed")
                elif self._hold_contract_json is not None or db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                    raise ValueError("retained_product_recovery_contract_missing")
        super().__init__(path, max_requests=max_requests, max_observed_cost_usd=max_observed_cost_usd,
                         reservation_policy=reservation_policy, min_interval_seconds=min_interval_seconds,
                         observe_transport=observe_transport, transport_diagnostics=transport_diagnostics)
        try:
            with self._lock, self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS product_recovery "
                                 "(id INTEGER PRIMARY KEY CHECK(id=1), version TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO product_recovery VALUES(1,?)", (self._product_version,))
                if self._hold_contract_json is not None:
                    self._db.execute("CREATE TABLE IF NOT EXISTS product_hold_recovery "
                                     "(id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                    self._db.execute("INSERT OR IGNORE INTO product_hold_recovery VALUES(1,?)", (self._hold_contract_json,))
                self._db.execute("CREATE TABLE IF NOT EXISTS product_events "
                                 "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL)")
                self._db.execute("CREATE TABLE IF NOT EXISTS product_retry_waits "
                                 "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL)")
                if self._db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone() != (self._product_version,):
                    raise ValueError("product_recovery_contract_changed")
        except BaseException:
            self._db.close()
            raise

    def _append(self, event):
        self._db.execute("INSERT INTO product_events(event) VALUES(?)",
                         (json.dumps({"at": _stamp(), **event}, sort_keys=True, allow_nan=False),))

    def _events(self):
        if self._hold_contract_json is not None and self._db.execute(
                "SELECT contract FROM product_hold_recovery WHERE id=1").fetchone() != (self._hold_contract_json,):
            raise ValueError("unknown_429_recovery_contract_changed")
        if self._db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone() != (self._product_version,):
            raise ValueError("product_recovery_contract_changed")
        return [json.loads(row[0]) for row in self._db.execute("SELECT event FROM product_events ORDER BY sequence")]

    @staticmethod
    def _event_state(events):
        operations, resumes, streak, pause, hard = {}, {}, 0, None, None
        rejections = set()
        for event in events:
            if type(event) is not dict or type(event.get("at")) is not str:
                raise ValueError("invalid_product_event")
            kind = event.get("event")
            if kind == "started":
                if (set(event) != {"at", "event", "token", "kind", "operation_id", "session_id"}
                        or event["kind"] not in {"turn", "review"}
                        or any(type(event[k]) is not str or not _ID.fullmatch(event[k])
                               for k in ("token", "operation_id", "session_id"))
                        or event["token"] in operations
                        or any(e["operation_id"] == event["operation_id"] for e in operations.values())):
                    raise ValueError("invalid_product_operation")
                operations[event["token"]] = {**event, "finished": False}
            elif kind == "finished":
                if (set(event) != {"at", "event", "token", "failed"}
                        or type(event["failed"]) is not bool or event["token"] not in operations
                        or operations[event["token"]]["finished"]
                        or (event["token"] in rejections and not event["failed"])):
                    raise ValueError("invalid_product_outcome")
                operations[event["token"]]["finished"] = True
                if event["token"] not in rejections:
                    streak = streak + 1 if event["failed"] else 0
                    pause = event["token"] if streak >= 3 else None
            elif kind == "request_rejected":
                if (set(event) != {"at", "event", "token", "role", "reason", "category",
                                   "request_sha256", "wire_bytes", "input_token_plan_bound"}
                        or event["token"] not in operations or event["token"] in rejections
                        or operations[event["token"]]["finished"]
                        or event["role"] not in {"online_fast", "memory", "reflection", "safety_review", "unrecognized"}
                        or event["reason"] not in _LOCAL_REJECTIONS
                        or event["category"] != _LOCAL_REJECTIONS[event["reason"]]
                        or type(event["request_sha256"]) is not str or len(event["request_sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in event["request_sha256"])
                        or type(event["wire_bytes"]) is not int or event["wire_bytes"] < 0
                        or type(event["input_token_plan_bound"]) is not int
                        or event["input_token_plan_bound"] != event["wire_bytes"] + 8192):
                    raise ValueError("invalid_product_request_rejection")
                rejections.add(event["token"])
                hard = "local_request_rejected"
            elif kind == "resumed":
                if (set(event) != {"at", "event", "request_id", "pause_id", "actor"}
                        or event["actor"] != "owner" or event["pause_id"] != pause
                        or type(event["request_id"]) is not str or not _ID.fullmatch(event["request_id"])
                        or event["request_id"] in resumes or not pause):
                    raise ValueError("invalid_product_resume")
                resumes[event["request_id"]] = event["pause_id"]
                streak, pause = 0, None
            elif kind == "budget_stopped":
                # Older saved events had no operation binding; preserve their
                # original streak interpretation. New refusals bind the scope
                # so zero-send budget checks are not counted as model failures.
                if (set(event) not in ({"at", "event", "reason"}, {"at", "event", "reason", "token"})
                        or event["reason"] not in _BUDGET_FAILURES):
                    raise ValueError("invalid_product_budget_stop")
                if "token" in event:
                    if (event["token"] not in operations or event["token"] in rejections
                            or operations[event["token"]]["finished"]):
                        raise ValueError("invalid_product_budget_stop")
                    rejections.add(event["token"])
                hard = event["reason"]
            else:
                raise ValueError("invalid_product_event")
        return operations, resumes, streak, pause, hard

    def _unknown_429(self, row):
        return (self._hold_contract_json is not None and "cost" not in row.get("usage", {})
                and row.get("http_status") == 429 and self._retryable(row))

    def _retryable(self, row):
        if (row.get("status") != "failed" or row.get("failure_phase") != "http_status"
                or ("cost" not in row.get("usage", {}) and not (
                    self._hold_contract_json is not None and row.get("http_status") == 429))
                or not (row.get("http_status") == 429 or 500 <= (row.get("http_status") or 0) <= 599)):
            return False
        audit = row.get("product_no_content_audit")
        if audit is None:
            return False
        validate_error_envelope_audit(audit)
        return (audit["source"] == "http_error_bytes" and audit["http_status"] == row["http_status"]
                and not audit["rejection_reasons"]
                and ("cost" in row["usage"] or (
                    audit["body_status"] == "parsed" and audit["object"] and audit["error_object"]))
                and ("cost" not in row["usage"] or
                     Decimal(str(row["usage"]["cost"])) <= Decimal(row["reservation"]["reserved_cost_usd"])))

    def _wait_plan(self, row):
        return retry_wait_plan(row["observation"], retry_index=row["product_transport_attempt"] - 1,
                               unknown_429_hold=self._unknown_429(row))

    def _check_recovery_source(self):
        if self._hold_contract_json is not None:
            if self._recovery_pre_send_check is not self._recovery_pre_send_check_original:
                raise MinimalCallStopped("recovery_source_check_changed")
            # This callback returns exactly True after checking the sealed
            # sources and predecessor-ledger manifest. No truthy substitutes.
            if self._recovery_pre_send_check() is not True:
                raise MinimalCallStopped("recovery_source_check_failed")
            if self._stop_requested.is_set():
                raise MinimalCallStopped("request_stop_requested")

    def _append_retry_wait(self, event):
        self._db.execute("INSERT INTO product_retry_waits(event) VALUES(?)",
            (json.dumps({"at": _stamp(), **event}, sort_keys=True, allow_nan=False),))

    def _retry_wait_state(self, attempts):
        rows = [json.loads(row[0]) for row in self._db.execute(
            "SELECT event FROM product_retry_waits ORDER BY sequence")]
        by_id = {row["request_id"]: row for row in attempts}
        pending, seen = {}, set()
        for event in rows:
            if type(event) is not dict or type(event.get("at")) is not str:
                raise ValueError("invalid_product_retry_wait")
            parent = by_id.get(event.get("parent_request_id"))
            if parent is None or not self._retryable(parent):
                raise ValueError("invalid_product_retry_wait_parent")
            if event.get("event") == "planned":
                if (set(event) != {"at", "event", "parent_request_id", "request_sha256", "plan"}
                        or parent["request_id"] in seen or event["request_sha256"] != parent["request_sha256"]
                        or json.dumps(event["plan"], sort_keys=True, allow_nan=False) != json.dumps(
                            self._wait_plan(parent),
                            sort_keys=True, allow_nan=False)):
                    raise ValueError("invalid_product_retry_wait_plan")
                seen.add(parent["request_id"])
                pending[parent["request_id"]] = event["plan"]
            elif event.get("event") == "completed":
                plan = pending.get(parent["request_id"])
                if (set(event) != {"at", "event", "parent_request_id", "wait_ms"}
                        or plan is None or plan["decision"] != "wait"
                        or type(event["wait_ms"]) not in (int, float) or not math.isfinite(event["wait_ms"])
                        or event["wait_ms"] < plan["seconds"] * 1000):
                    raise ValueError("invalid_product_retry_wait_completion")
                del pending[parent["request_id"]]
            else:
                raise ValueError("invalid_product_retry_wait_event")
        active = (getattr(self, "_active_product_retry_wait", None) if self._hold_contract_json is not None
                  else getattr(self._local, "active_retry_wait", None))
        unresolved = any(key != active or plan["decision"] == "blocked" for key, plan in pending.items())
        return rows, unresolved

    def _wait_for_product_retry(self, row):
        parent = row["request_id"]
        plan = self._wait_plan(row)
        try:
            with self._lock, self._transaction():
                self._append_retry_wait({"event": "planned", "parent_request_id": parent,
                    "request_sha256": row["request_sha256"], "plan": plan})
            if plan["decision"] == "blocked":
                raise MinimalCallStopped("product_retry_wait_unavailable")
            self._local.active_retry_wait = parent
            if self._hold_contract_json is not None:
                self._active_product_retry_wait = parent
            waited = perform_retry_wait(plan, monotonic=self._pacing_clock, sleep=self._pacing_sleep,
                cancelled=self._stop_requested.is_set)
            # The next physical claim repeats scope/configuration/fee checks.
            # Completion is not a reservation, receipt, or send authorization.
            with self._lock, self._transaction():
                self._check_recovery_source()
                state = self._snapshot()
                if state["stopped"]:
                    raise MinimalCallStopped(state["stop_reason"])
                self._append_retry_wait({"event": "completed", "parent_request_id": parent,
                    "wait_ms": waited})
        except BaseException as error:
            # Preserve the unresolved started operation as well if persisting
            # the wait failed. A restart cannot turn storage failure into permission.
            self._local.rejection_save_failed = True
            if not isinstance(error, Exception):
                raise
            raise MinimalCallStopped("product_retry_wait_failed") from None
        finally:
            self._local.active_retry_wait = None
            self._active_product_retry_wait = None

    def _snapshot(self):
        state = super()._snapshot()
        events = self._events()
        operations, _resumes, streak, pause, hard = self._event_state(events)
        waits, wait_unresolved = self._retry_wait_state(state["attempts"])
        reason = state["stop_reason"]
        unknown_429 = [row for row in state["attempts"] if self._unknown_429(row)]
        prior_unknown = (json.loads(self._hold_contract_json)["prior_unknown_429_count"]
                         if self._hold_contract_json is not None else 0)
        if reason == "provider_cost_unknown" and self._hold_contract_json is not None:
            if (all("cost" in row["usage"] or self._unknown_429(row) for row in state["attempts"])
                    and all(row["status"] != "failed" or row.get("failure_code") in _OUTPUT_FAILURES
                            or self._retryable(row) for row in state["attempts"])):
                reason = None
        if reason == "request_failed":
            # Only the finite recoverable category stops being permanent. Route,
            # permission, billing, storage and unclassified errors remain hard.
            if all(row["status"] != "failed" or row.get("failure_code") in _OUTPUT_FAILURES
                   or self._retryable(row) for row in state["attempts"]):
                reason = None
        if reason is None:
            if self._hold_contract_json is not None and prior_unknown + len(unknown_429) >= 6:
                reason = "unknown_429_task_limit_reached"
            elif any(row.get("product_transport_attempt") == 3 for row in unknown_429):
                reason = "unknown_429_request_attempts_exhausted"
            elif state["calls_used"] >= self.max_requests:
                reason = "request_limit_reached"
            elif Decimal(state["known_cost_usd"] or "0") >= Decimal(self.threshold):
                reason = "observed_cost_threshold_reached"
            elif wait_unresolved:
                reason = "unresolved_product_retry_wait"
            elif hard:
                reason = hard
            elif any(not row["finished"] and token not in self._active_operation_tokens
                     for token, row in operations.items()):
                reason = "unresolved_product_operation"
            elif pause:
                reason = "consecutive_failures"
        state.update(stopped=bool(reason), stop_reason=reason,
                     recovery_contract=self._product_version, consecutive_failures=streak,
                     can_resume=reason == "consecutive_failures", pause_id=pause,
                     max_extra_attempts_per_request=2, product_events=events, product_retry_waits=waits)
        if self._hold_contract_json is not None:
            state["unknown_429_recovery"] = {
                **json.loads(self._hold_contract_json), "local_unknown_429_count": len(unknown_429),
                "task_unknown_429_count": prior_unknown + len(unknown_429), "max_task_unknown_429": 6,
                "unknown_cost_holds_preserved": True, "minimum_local_cooldown_seconds": 900,
                "maximum_wait_seconds": 3600,
                "wait_in_progress": getattr(self, "_active_product_retry_wait", None) is not None}
        rejected = next((event for event in reversed(events) if event["event"] == "request_rejected"), None)
        if rejected is not None:
            state["last_request_rejection"] = {
                **rejected, "operation_id": operations[rejected["token"]]["operation_id"],
                "request_sent": False,
            }
        return state

    @contextmanager
    def scope(self, kind, operation_id, session_id):
        # Same-instance operations may interleave at physical boundaries. Only
        # tokens registered by this live guard can share its durable allowance;
        # another instance/restart still sees an unresolved started event.
        with super().scope(kind, operation_id, session_id):
            token = uuid.uuid4().hex
            with self._lock, self._transaction():
                state = self._snapshot()
                if state["stopped"]:
                    raise MinimalCallStopped(state["stop_reason"])
                if any(event.get("operation_id") == operation_id for event in state["product_events"]):
                    raise MinimalCallStopped("duplicate_model_operation")
                self._append({"event": "started", "token": token, "kind": kind,
                              "operation_id": operation_id, "session_id": session_id})
                self._active_operation_tokens.add(token)
            self._local.operation_token = token
            self._local.output_failed = False
            self._local.local_rejected = False
            self._local.budget_rejected = False
            self._local.rejection_save_failed = False
            failed = True
            try:
                yield
                with self._lock:
                    rows = [json.loads(row[0]) for row in self._db.execute("SELECT record FROM attempts ORDER BY sequence")]
                last = {r["role"]: r for r in rows if r["operation_id"] == operation_id}
                failed = (self._local.output_failed or self._local.local_rejected
                          or any(r["status"] != "received" for r in last.values()))
            finally:
                try:
                    if self._local.rejection_save_failed:
                        raise MinimalCallStopped("local_rejection_save_failed")
                    with self._lock, self._transaction():
                        self._append({"event": "finished", "token": token, "failed": failed})
                except Exception:
                    self._broken = True
                    raise MinimalCallStopped("product_outcome_save_failed") from None
                finally:
                    with self._lock:
                        self._active_operation_tokens.discard(token)
                    self._local.operation_token = None

    def note_output_failure(self):
        """Engine parsers can return a fallback without raising a transport error."""
        if getattr(self._local, "operation_token", None) is None:
            raise MinimalCallStopped("operation_scope_required")
        self._local.output_failed = True

    def halt(self, reason="model_or_storage_failed"):
        # The common client reports post-receipt parser failures through halt.
        # In the product they fail this operation only; the unchanged strict
        # guards still halt permanently on the same notification.
        if reason == "model_output_failed" and getattr(self._local, "operation_token", None) is not None:
            self.note_output_failure()
            return
        return super().halt(reason)

    def memory_failure_is_terminal(self, client):
        """Seal a settled bad body without reissuing it or running reflection.

        Used only by the production offline wrapper. Missing/unknown receipts
        and storage/identity failures keep the original queue entry blocked.
        """
        scope = getattr(self._local, "scope", {})
        if client.role != "memory" or scope.get("kind") != "review":
            return False
        state = self.snapshot()
        if state["stop_reason"] not in {None, "request_limit_reached", "observed_cost_threshold_reached"}:
            return False
        rows = [row for row in state["attempts"] if row["operation_id"] == scope["operation_id"]
                and row["session_id"] == scope["session_id"] and row["role"] == "memory"]
        if not rows:
            return False
        row = rows[-1]
        if self._retryable(row):
            # A completely settled no-content HTTP failure is terminal for this
            # review too. Keeping its attempted job at the FIFO head would
            # permanently block later sessions after transport retries end.
            return True
        return ("cost" in row["usage"] and row.get("observation", {}).get("envelope_kind") == "success"
                and row.get("observation", {}).get("body_status") == "parsed"
                and (row["status"] == "received" or row.get("failure_code") in _OUTPUT_FAILURES))

    def resume(self, request_id, pause_id, *, actor="owner"):
        if actor != "owner" or type(request_id) is not str or not _ID.fullmatch(request_id):
            raise ValueError("invalid_product_resume")
        with self._lock:
            try:
                with self._transaction():
                    state = self._snapshot()
                    _ops, resumes, _streak, _pause, _hard = self._event_state(state["product_events"])
                    if request_id in resumes:
                        if resumes[request_id] != pause_id:
                            raise MinimalCallStopped("resume_identity_changed")
                        return
                    if not state["can_resume"] or pause_id != state["pause_id"]:
                        raise MinimalCallStopped("product_resume_not_allowed")
                    self._append({"event": "resumed", "request_id": request_id,
                                  "pause_id": pause_id, "actor": actor})
            except MinimalCallStopped:
                raise
            except Exception:
                self._broken = True
                raise MinimalCallStopped("product_resume_save_failed") from None

    def _observe_http_failure(self, exc):
        audit = {}
        observation, usage = observe_http_error(exc, envelope_audit=audit)
        return observation, usage, {"product_no_content_audit": audit}

    def _allow_product_retry(self, client, request, previous):
        if getattr(client, "reflection_return_format", "native") == "body_json":
            from .reflection_json import reflection_product_step
            step = reflection_product_step(json.loads(request.data)["messages"])
            previous = [row for row in previous if row.get("product_step", "initial") == step]
        parent = getattr(self._local, "retry_parent", None)
        return (bool(previous) and len(previous) < 3 and parent == previous[-1]["request_id"]
                and all(self._retryable(row) for row in previous)
                and all(row["request_sha256"] == hashlib.sha256(request.data).hexdigest() for row in previous))

    def _reflection_product_metadata(self, client, payload, previous):
        if getattr(client, "reflection_return_format", "native") != "body_json":
            return super()._reflection_product_metadata(client, payload, previous)
        from .reflection_json import reflection_product_step
        step = reflection_product_step(payload["messages"])
        if step == "initial":
            return super()._reflection_product_metadata(client, payload, previous)
        initial = [row for row in previous if row.get("product_step") == "initial"]
        correction = [row for row in previous if row.get("product_step") == "format_correction"]
        if (not initial or len(initial) > 3 or len(initial) + len(correction) != len(previous)
                or any(not self._retryable(row) or row["request_sha256"] != initial[-1]["request_sha256"]
                       for row in initial[:-1])):
            raise MinimalCallStopped("invalid_reflection_correction_antecedent")
        if correction and (
            len(correction) >= 3 or any(not self._retryable(row) for row in correction)
            or getattr(self._local, "retry_parent", None) != correction[-1]["request_id"]
            or any(row["request_sha256"] != self._local.product_wire_sha for row in correction)
        ):
            raise MinimalCallStopped("invalid_reflection_correction_antecedent")
        # The unchanged strict validator checks the one received initial body
        # and exact correction diagnostics. Infrastructure attempts are not
        # additional received answers or additional business corrections.
        return super()._reflection_product_metadata(client, payload, [initial[-1]])

    def _reserve(self, client, request, payload, state):
        reservation = super()._reserve(client, request, payload, state)
        parent = getattr(self._local, "retry_parent", None)
        reservation["product_transport_attempt"] = 1
        if parent is not None:
            prior = next(row for row in state["attempts"] if row["request_id"] == parent)
            reservation.update(product_retry_parent_request_id=parent,
                               product_transport_attempt=prior["product_transport_attempt"] + 1)
        return reservation

    def _begin(self, client, request, **kwargs):
        # A rejected claim has no physical attempt or provider charge. It must
        # still be durable: otherwise an Engine fallback would misclassify a
        # configuration/identity/authority error as recoverable model output.
        try:
            # _begin runs after the ordinary physical-start pacing wait too.
            self._check_recovery_source()
            if (getattr(self._local, "retry_parent", None) is not None
                    and hashlib.sha256(request.data).hexdigest() != self._local.product_wire_sha):
                raise MinimalCallStopped("model_request_configuration_changed")
            return super()._begin(client, request, **kwargs)
        except Exception as error:
            if isinstance(error, MinimalCallStopped) and str(error) in _BUDGET_FAILURES:
                raise
            token = getattr(self._local, "operation_token", None)
            if token is None:
                raise
            if getattr(self._local, "local_rejected", False):
                raise MinimalCallStopped("local_request_rejected") from None
            if getattr(self._local, "rejection_save_failed", False):
                raise MinimalCallStopped("local_rejection_save_failed") from None
            reason = str(error)
            if reason not in _LOCAL_REJECTIONS:
                reason = "local_preflight_rejected"
            role = getattr(client, "role", "unrecognized")
            if role not in self.allowed_roles:
                role = "unrecognized"
            try:
                with self._transaction():
                    self._append({"event": "request_rejected", "token": token,
                        "role": role, "reason": reason, "category": _LOCAL_REJECTIONS[reason],
                        "request_sha256": hashlib.sha256(request.data).hexdigest(),
                        "wire_bytes": len(request.data), "input_token_plan_bound": len(request.data) + 8192})
                    self._db.execute("UPDATE settings SET halted=COALESCE(halted,?) WHERE id=1",
                                     ("local_request_rejected",))
            except Exception:
                # Preserve the unresolved started operation on disk. A restart
                # must not reinterpret a failed stop write as permission.
                self._broken = True
                self._local.rejection_save_failed = True
                raise MinimalCallStopped("local_rejection_save_failed") from None
            client._update_diagnostic(error_type="local_request_rejected", local_rejection_code=reason)
            self._local.local_rejected = True
            raise MinimalCallStopped("local_request_rejected") from None

    def request_json(self, client, request, *, require_preference_control=False, require_task_projection=False):
        if type(require_preference_control) is not bool or type(require_task_projection) is not bool:
            raise MinimalCallStopped("model_request_configuration_changed")
        options = {**({"require_preference_control": True} if require_preference_control else {}), **({"require_task_projection": True} if require_task_projection else {})}
        with nullcontext():
            self._local.retry_parent = None
            self._local.product_wire_sha = hashlib.sha256(request.data).hexdigest()
            try:
                for attempt in range(3):
                    try:
                        return self._request_once(client, request, **options)
                    except MinimalCallStopped as error:
                        if str(error) == "local_request_rejected":
                            client._update_diagnostic(error_type="local_request_rejected")
                        if str(error) in _BUDGET_FAILURES:
                            if self._local.budget_rejected:
                                raise
                            try:
                                with self._lock, self._transaction():
                                    self._append({"event": "budget_stopped", "reason": str(error),
                                                  "token": self._local.operation_token})
                            except Exception:
                                self._broken = True
                                self._local.rejection_save_failed = True
                                raise MinimalCallStopped("local_rejection_save_failed") from None
                            self._local.budget_rejected = True
                            self._local.local_rejected = True
                            raise
                        state = self.snapshot()
                        rows = state.get("attempts", [])
                        last = rows[-1] if rows else None
                        scope = getattr(self._local, "scope", {})
                        no_retry_role = (client.role == "safety_review"
                                         or getattr(client, "offline_reasoning_route", None) is not None)
                        if no_retry_role and last is not None and self._unknown_429(last):
                            # These roles retain their explicit zero-retry
                            # contract. Do not let a later operation bypass the
                            # required cooldown after their unsettled 429.
                            self.halt()
                        if (no_retry_role
                                or attempt == 2 or state["stopped"] or last is None
                                or last["operation_id"] != scope.get("operation_id")
                                or last["role"] != client.role or not self._retryable(last)):
                            raise
                        self._wait_for_product_retry(last)
                        self._local.retry_parent = last["request_id"]
            finally:
                self._local.retry_parent = None
                self._local.product_wire_sha = None
