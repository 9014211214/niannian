"""Explicit positive prewrite recovery; old policies and ledgers stay unchanged.

This guard preserves the prior initial attempt, unknown fee reservation, and
one already-used connection rebuild. It authorizes at most two new attempts.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reserved_calls import validate_reservation_policy

from aria.engine.policy import e1br_calls as base

from aria.engine.policy.observed_transport import validate_trace
from aria.engine.policy.transport_failure import validate_transport_failure


def transient_prewrite_failure(row):
    """Require positive before-request proof; no billing inference is made."""
    if (row.get("status") != "failed" or row.get("validation_error") is not None
            or row.get("failure_phase") != "pre_response_open"
            or row.get("failure_code") != "response_open_failed"
            or row.get("http_status") is not None
            or row.get("usage") != {} or row.get("observation") != {}
            or any(key in row for key in ("visible_content_sha256", "client_return_content_sha256",
                "model_reply_sha256", "visible_original_path", "visible_original_sha256", "receipt_identity",
                "no_content_evidence", "error_envelope_audit"))):
        return False
    try:
        failure = validate_transport_failure(row["transport_failure"])
        trace = validate_trace(row["transport_events"])
        if (failure["version"] != "transport_failure_v2" or trace["version"] != "connection_trace_v2"
                or failure["opener_call"] != "invoked" or failure["response_object"] != "not_observed"
                or trace["events"] != {"connection_preparation_started": True, "connection_preparation_failed_before_request": True}):
            return False
        preparations = trace.get("connection_preparations")
        if type(preparations) is not list or not 1 <= len(preparations) <= 2:
            return False
        def eligible(value):
            return value["exception_kind"] in {"tls_eof_error", "connection_reset", "connection_aborted"} or (
                value["exception_kind"] == "tls_error"
                and value.get("tls_details", {}).get("reason_code") == "UNEXPECTED_EOF_WHILE_READING")
        if not eligible(failure) or not all("failure" in p and eligible(p["failure"]) for p in preparations):
            return False
        # urllib wraps the same typed local error. Wrapper annotations may vary;
        # classification and every other saved detail must remain identical.
        normalized = lambda value: {k: v for k, v in value.items() if k not in {"exception_wrapper", "reason_status"}}
        return normalized(failure) == normalized(preparations[-1]["failure"])
    except (KeyError, TypeError, ValueError, IndexError):
        return False


SCHEMA = "e1br_positive_prewrite_retry_v1"
SHA = re.compile(r"[a-f0-9]{64}")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def identity(row):
    return row.get("operation_id"), row.get("session_id"), row.get("role")


def no_content(row):
    return base.no_content_failure(row) or transient_prewrite_failure(row)


def validate_contract(contract):
    fields = {"schema", "enabled", "approval_sha256", "parent_campaign", "parent_manifest_sha256",
              "parent_receipt_sha256", "parent_attempt", "connection_rebuild_used", "inherited_physical_attempts",
              "remaining_infrastructure_retries"}
    if (type(contract) is not dict or set(contract) != fields or contract["schema"] != SCHEMA
            or contract["enabled"] is not True or contract["connection_rebuild_used"] is not True
            or type(contract["inherited_physical_attempts"]) is not int or contract["inherited_physical_attempts"] != 1
            or type(contract["remaining_infrastructure_retries"]) is not int or contract["remaining_infrastructure_retries"] != 2
            or not all(type(contract[k]) is str and SHA.fullmatch(contract[k]) for k in
                       ("approval_sha256", "parent_manifest_sha256", "parent_receipt_sha256"))
            or not isinstance(contract["parent_campaign"], str) or not contract["parent_campaign"].startswith("artifacts/local/")
            or ".." in Path(contract["parent_campaign"]).parts):
        raise ValueError("prewrite_optin_contract_invalid")
    parent = contract["parent_attempt"]
    if (not transient_prewrite_failure(parent) or digest(parent) != contract["parent_receipt_sha256"]
            or parent.get("product_step") != "initial" or parent.get("infrastructure_retry_index") != 0
            or "infrastructure_retry_of_request_id" in parent
            or len(parent["transport_events"]["connection_preparations"]) != 2
            or [p["ordinal"] for p in parent["transport_events"]["connection_preparations"]] != [1, 2]
            or parent.get("role") != "online_fast" or len(parent.get("experiment_position", [])) != 6):
        raise ValueError("prewrite_retained_initial_proof_invalid")
    return contract


def link(contract, index):
    return {"schema": "e1br_positive_prewrite_link_v1", "contract_sha256": digest(contract),
            "parent_request_id": contract["parent_attempt"]["request_id"],
            "parent_receipt_sha256": contract["parent_receipt_sha256"],
            "lineage_attempt_index": index, "old_no_content_proven": True,
            "preparation_rebuild_already_used": True,
            "basis": "retained_positive_before_request_proof" if index == 1 else "new_parent_no_content_proof"}


def validate_rows(rows, contract):
    validate_contract(contract)
    parent, matched = contract["parent_attempt"], []
    for number, row in enumerate(rows):
        is_target = identity(row) == identity(parent)
        if (row.get("experiment_position") == parent["experiment_position"]) != is_target:
            raise ValueError("prewrite_position_identity_changed")
        if not is_target:
            if "prewrite_retry" in row or not matched or matched[-1]["status"] != "received":
                raise ValueError("prewrite_original_position_unsettled")
            continue
        index = len(matched) + 1
        preparations = row.get("transport_events", {}).get("connection_preparations", [])
        if (index > 2 or number != index - 1 or row.get("prewrite_retry") != link(contract, index)
                or row.get("product_step") != "initial" or row.get("infrastructure_retry_index") != index - 1
                or row.get("request_sha256") != parent["request_sha256"] or row.get("model") != parent["model"]
                or index == 2 and not no_content(matched[-1])
                or len(preparations) > 1 or preparations and preparations[0]["ordinal"] != 2):
            raise ValueError("prewrite_inherited_retry_or_rebuild_pool_changed")
        matched.append(row)
    return matched


def validate_prewrite_ledger(path, *, contract, stage, registration_sha256, max_requests, reservation_policy):
    validate_contract(contract)
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        if db.execute("SELECT contract FROM e1br_prewrite_policy WHERE id=1").fetchone() != (encoded(contract),):
            raise ValueError("prewrite_ledger_contract_changed")
        original = json.dumps(base.contract_value(stage=stage, registration_sha256=registration_sha256,
                                                  max_requests=max_requests), sort_keys=True)
        if db.execute("SELECT contract FROM e1br_policy WHERE id=1").fetchone() != (original,):
            raise ValueError("e1br_contract_changed")
        policy = validate_reservation_policy(json.loads(db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()[0]))
        rows = [json.loads(row[0]) for row in db.execute("SELECT record FROM attempts ORDER BY sequence")]
    if policy != validate_reservation_policy(reservation_policy):
        raise ValueError("prewrite_reservation_policy_changed")
    base.validate_e1br_records(rows, stage=stage, registration_sha256=registration_sha256, no_content_check=no_content)
    base.validate_e1br_reservations(rows, policy)
    validate_rows(rows, contract)
    return rows


class PrewriteGuardMixin:
    def __init__(self, path, *, prewrite_contract, **kwargs):
        self.prewrite_contract = deepcopy(validate_contract(prewrite_contract))
        self._prewrite_encoded = encoded(self.prewrite_contract)
        if Path(path).exists():
            with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                if db.execute("SELECT contract FROM e1br_prewrite_policy WHERE id=1").fetchone() != (self._prewrite_encoded,):
                    raise ValueError("prewrite_ledger_contract_changed")
        self._prewrite_initializing = True
        super().__init__(path, **kwargs)
        try:
            with self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_prewrite_policy (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO e1br_prewrite_policy VALUES(1,?)", (self._prewrite_encoded,))
            self._prewrite_initializing = False
            self._snapshot()
        except BaseException:
            self._db.close()
            raise

    def _no_content_failure(self, row):
        return no_content(row)

    def _snapshot(self):
        state = super()._snapshot()
        if self._prewrite_initializing:
            return state
        if (encoded(self.prewrite_contract) != self._prewrite_encoded or self._db.execute(
                "SELECT contract FROM e1br_prewrite_policy WHERE id=1").fetchone() != (self._prewrite_encoded,)):
            raise ValueError("prewrite_ledger_contract_changed")
        target = validate_rows(state["attempts"], self.prewrite_contract)
        if len(target) == 2 and target[-1]["status"] == "failed":
            state.update(stopped=True, stop_reason=state.get("stop_reason") or "inherited_retry_pool_exhausted")
        state["prewrite_retry"] = {"schema": SCHEMA, "inherited_initial_physical_attempts": 1,
            "new_physical_attempts_at_position": len(target), "remaining_new_attempts_at_position": 2 - len(target),
            "inherited_connection_rebuild_used": True, "contract_sha256": digest(self.prewrite_contract)}
        return state

    def _reserve(self, client, request, payload, state):
        parent = self.prewrite_contract["parent_attempt"]
        is_target = identity({**self._local.scope, "role": client.role}) == identity(parent)
        prior = [r for r in state["attempts"] if identity(r) == identity(parent)]
        if is_target:
            if (len(prior) >= 2 or len(state["attempts"]) != len(prior) or prior and not no_content(prior[-1])):
                raise MinimalCallStopped("inherited_retry_pool_exhausted")
            if hashlib.sha256(request.data).hexdigest() != parent["request_sha256"] or client.model != parent["model"]:
                raise MinimalCallStopped("prewrite_failed_wire_changed")
        elif not prior or prior[-1]["status"] != "received":
            raise MinimalCallStopped("prewrite_original_position_unsettled")
        row = super()._reserve(client, request, payload, state)
        if is_target:
            if "experiment_position" in row and row["experiment_position"] != parent["experiment_position"]:
                raise MinimalCallStopped("prewrite_failed_position_changed")
            row.update(experiment_position=deepcopy(parent["experiment_position"]), prewrite_retry=link(self.prewrite_contract, len(prior) + 1))
        return row

    def _begin(self, client, request, **kwargs):
        row = super()._begin(client, request, **kwargs)
        if identity(row) == identity(self.prewrite_contract["parent_attempt"]):
            self._local.preparation_budget = {"attempts": 1}
        return row


class PrewriteE1BRCallGuard(PrewriteGuardMixin, base.E1BRCallGuard):
    pass
