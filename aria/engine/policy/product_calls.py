"""Product-only recovery, separate from frozen experimental stop contracts.

Physical request receipts retain their original accounting. Logical operations
and explicit owner recoveries are append-only events; three failed operations,
not three transport attempts, pause the product. Unknown charges, uncertain
storage, and budget limits cannot be resumed through this interface.
The inherited default transport rejects HTTP redirects, and product recovery
does not retry these failures or release their unknown-charge reservations.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import sqlite3
import uuid

from .minimal_calls import MinimalCallStopped, _ID, _stamp
from .reserved_calls import ReservedCallGuard
from .response_observability import observe_http_error, validate_error_envelope_audit

_VERSION = "product_call_recovery_v1"
_OUTPUT_FAILURES = frozenset({
    "invalid_or_truncated_response", "invalid_response_message",
    "online_content_invalid_json", "online_content_duplicate_key", "online_content_non_finite",
    "online_content_not_object", "online_response_missing", "online_response_not_string",
    "online_response_empty", "online_decision_schema_invalid", "content_json_invalid",
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

    allowed_roles = {"online_fast", "memory", "reflection"}

    def __init__(self, path, *, max_requests, max_observed_cost_usd, reservation_policy):
        # Do not silently reinterpret a retained strict experiment/legacy ledger.
        # Even a known charge cannot authorize changing that ledger's contract.
        from pathlib import Path
        if Path(path).exists():
            with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
                present = db.execute("SELECT name FROM sqlite_master WHERE name='product_recovery'").fetchone()
                if not present and db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                    raise ValueError("retained_product_recovery_contract_missing")
        super().__init__(path, max_requests=max_requests, max_observed_cost_usd=max_observed_cost_usd,
                         reservation_policy=reservation_policy)
        try:
            with self._lock, self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS product_recovery "
                                 "(id INTEGER PRIMARY KEY CHECK(id=1), version TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO product_recovery VALUES(1,?)", (_VERSION,))
                self._db.execute("CREATE TABLE IF NOT EXISTS product_events "
                                 "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL)")
                if self._db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone() != (_VERSION,):
                    raise ValueError("product_recovery_contract_changed")
        except BaseException:
            self._db.close()
            raise

    def _append(self, event):
        self._db.execute("INSERT INTO product_events(event) VALUES(?)",
                         (json.dumps({"at": _stamp(), **event}, sort_keys=True, allow_nan=False),))

    def _events(self):
        if self._db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone() != (_VERSION,):
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
                        or event["role"] not in {"online_fast", "memory", "reflection", "unrecognized"}
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

    @staticmethod
    def _retryable(row):
        if (row.get("status") != "failed" or row.get("failure_phase") != "http_status"
                or "cost" not in row.get("usage", {})
                or not (row.get("http_status") == 429 or 500 <= (row.get("http_status") or 0) <= 599)):
            return False
        audit = row.get("product_no_content_audit")
        if audit is None:
            return False
        validate_error_envelope_audit(audit)
        return (audit["source"] == "http_error_bytes" and audit["http_status"] == row["http_status"]
                and not audit["rejection_reasons"]
                and Decimal(str(row["usage"]["cost"])) <= Decimal(row["reservation"]["reserved_cost_usd"]))

    def _snapshot(self):
        state = super()._snapshot()
        events = self._events()
        operations, _resumes, streak, pause, hard = self._event_state(events)
        reason = state["stop_reason"]
        if reason == "request_failed":
            # Only the finite recoverable category stops being permanent. Route,
            # permission, billing, storage and unclassified errors remain hard.
            if all(row["status"] != "failed" or row.get("failure_code") in _OUTPUT_FAILURES
                   or self._retryable(row) for row in state["attempts"]):
                reason = None
        if reason is None:
            if state["calls_used"] >= self.max_requests:
                reason = "request_limit_reached"
            elif Decimal(state["known_cost_usd"] or "0") >= Decimal(self.threshold):
                reason = "observed_cost_threshold_reached"
            elif hard:
                reason = hard
            elif any(not row["finished"] and token != getattr(self._local, "operation_token", None)
                     for token, row in operations.items()):
                reason = "unresolved_product_operation"
            elif pause:
                reason = "consecutive_failures"
        state.update(stopped=bool(reason), stop_reason=reason,
                     recovery_contract=_VERSION, consecutive_failures=streak,
                     can_resume=reason == "consecutive_failures", pause_id=pause,
                     max_extra_attempts_per_request=2, product_events=events)
        rejected = next((event for event in reversed(events) if event["event"] == "request_rejected"), None)
        if rejected is not None:
            state["last_request_rejection"] = {
                **rejected, "operation_id": operations[rejected["token"]]["operation_id"],
                "request_sent": False,
            }
        return state

    @contextmanager
    def scope(self, kind, operation_id, session_id):
        # Serialize a complete interaction locally, while a durable started
        # event blocks another process between physical requests and saving.
        with self._lock, super().scope(kind, operation_id, session_id):
            token = uuid.uuid4().hex
            with self._transaction():
                state = self._snapshot()
                if state["stopped"]:
                    raise MinimalCallStopped(state["stop_reason"])
                if any(event.get("operation_id") == operation_id for event in state["product_events"]):
                    raise MinimalCallStopped("duplicate_model_operation")
                self._append({"event": "started", "token": token, "kind": kind,
                              "operation_id": operation_id, "session_id": session_id})
            self._local.operation_token = token
            self._local.output_failed = False
            self._local.local_rejected = False
            self._local.budget_rejected = False
            self._local.rejection_save_failed = False
            failed = True
            try:
                yield
                rows = [json.loads(row[0]) for row in self._db.execute("SELECT record FROM attempts ORDER BY sequence")]
                last = {r["role"]: r for r in rows if r["operation_id"] == operation_id}
                failed = (self._local.output_failed or self._local.local_rejected
                          or any(r["status"] != "received" for r in last.values()))
            finally:
                try:
                    if self._local.rejection_save_failed:
                        raise MinimalCallStopped("local_rejection_save_failed")
                    with self._transaction():
                        self._append({"event": "finished", "token": token, "failed": failed})
                except Exception:
                    self._broken = True
                    raise MinimalCallStopped("product_outcome_save_failed") from None
                finally:
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

    def request_json(self, client, request):
        with self._lock:
            self._local.retry_parent = None
            self._local.product_wire_sha = hashlib.sha256(request.data).hexdigest()
            try:
                for attempt in range(3):
                    try:
                        return self._request_once(client, request)
                    except MinimalCallStopped as error:
                        if str(error) == "local_request_rejected":
                            client._update_diagnostic(error_type="local_request_rejected")
                        if str(error) in _BUDGET_FAILURES:
                            if self._local.budget_rejected:
                                raise
                            try:
                                with self._transaction():
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
                        if (attempt == 2 or state["stopped"] or last is None
                                or last["operation_id"] != scope.get("operation_id")
                                or last["role"] != client.role or not self._retryable(last)):
                            raise
                        delay = last["observation"].get("retry_after_seconds", 1 + attempt)
                        # An excessive/rejected delay is not silently shortened.
                        if last["observation"].get("retry_after_status") == "rejected" or delay > 30:
                            raise
                        self._pacing_sleep(max(1 + attempt, delay))
                        self._local.retry_parent = last["request_id"]
            finally:
                self._local.retry_parent = None
                self._local.product_wire_sha = None
