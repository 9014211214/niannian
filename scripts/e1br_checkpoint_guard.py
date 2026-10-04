"""One explicitly authorized external failed attempt; never rewrite its proof.

Local retry indices retain the original E1b-R representation. Separate lineage
indices count the old physical attempt and cap its two remaining dispatches.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from aria.engine.policy.e1br_calls import no_content_failure, validate_e1br_ledger
from aria.engine.policy.minimal_calls import MinimalCallStopped


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def identity(row):
    return row.get("operation_id"), row.get("session_id"), row.get("role")


def validate_checkpoint_rows(rows, contract):
    target = identity(contract["parent_attempt"])
    matched = []
    for number, row in enumerate(rows):
        is_target = identity(row) == target
        same_position = row.get("experiment_position") == contract["parent_attempt"]["experiment_position"]
        if same_position != is_target:
            raise ValueError("checkpoint_position_identity_changed")
        if not is_target:
            if "checkpoint_retry" in row:
                raise ValueError("checkpoint_exception_wrong_position")
            if not matched or matched[-1]["status"] != "received":
                raise ValueError("checkpoint_first_position_unsettled")
            continue
        index = len(matched) + 1
        expected = {"schema": "e1br_checkpoint_retry_link_v1",
            "exception_sha256": contract["exception_sha256"],
            "parent_request_id": contract["parent_attempt"]["request_id"],
            "parent_receipt_sha256": contract["parent_receipt_sha256"],
            "parent_manifest_sha256": contract["parent_manifest_sha256"],
            "lineage_attempt_index": index, "old_no_content_proven": False,
            "basis": "explicit_single_position_exception" if index == 1 else "new_parent_no_content_proof"}
        if (index > 2 or number != index - 1 or row.get("checkpoint_retry") != expected
                or row.get("product_step") != "initial" or row.get("infrastructure_retry_index") != index - 1
                or row.get("request_sha256") != contract["parent_attempt"]["request_sha256"]
                or row.get("model") != contract["parent_attempt"]["model"]
                or index == 2 and not no_content_failure(matched[-1])):
            raise ValueError("checkpoint_external_retry_contract_changed")
        matched.append(row)
    return matched


def validate_checkpoint_ledger(path, *, contract, **kwargs):
    rows = validate_e1br_ledger(path, **kwargs)
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        if db.execute("SELECT contract FROM e1br_checkpoint_policy WHERE id=1").fetchone() != (encoded(contract),):
            raise ValueError("checkpoint_ledger_contract_changed")
    validate_checkpoint_rows(rows, contract)
    return rows


class CheckpointGuardMixin:
    """Combine before the formal EntryGuard; ordinary retry policy is unchanged."""

    def __init__(self, path, *, checkpoint_contract, **kwargs):
        self.checkpoint_contract = deepcopy(checkpoint_contract)
        self._checkpoint_encoded = encoded(checkpoint_contract)
        existed = Path(path).exists()
        if existed:
            with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                if db.execute("SELECT contract FROM e1br_checkpoint_policy WHERE id=1").fetchone() != (self._checkpoint_encoded,):
                    raise ValueError("checkpoint_ledger_contract_changed")
        self._checkpoint_initializing = True
        super().__init__(path, **kwargs)
        try:
            with self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_checkpoint_policy (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO e1br_checkpoint_policy VALUES(1,?)", (self._checkpoint_encoded,))
            self._checkpoint_initializing = False
            self._snapshot()
        except BaseException:
            self._db.close()
            raise

    def _snapshot(self):
        state = super()._snapshot()
        if self._checkpoint_initializing:
            return state
        if (encoded(self.checkpoint_contract) != self._checkpoint_encoded
                or self._db.execute("SELECT contract FROM e1br_checkpoint_policy WHERE id=1").fetchone() != (self._checkpoint_encoded,)):
            raise ValueError("checkpoint_ledger_contract_changed")
        target = validate_checkpoint_rows(state["attempts"], self.checkpoint_contract)
        if len(target) == 2 and target[-1]["status"] == "failed":
            state.update(stopped=True, stop_reason=state.get("stop_reason") or "inherited_retry_pool_exhausted")
        state["checkpoint"] = {"schema": "e1br_checkpoint_state_v1",
            "exception_sha256": self.checkpoint_contract["exception_sha256"],
            "retained_old_attempts_at_position": 1, "new_attempts_at_position": len(target),
            "remaining_attempts_at_position": 2 - len(target), "old_no_content_proven": False}
        return state

    def _reserve(self, client, request, payload, state):
        target = self.checkpoint_contract["parent_attempt"]
        scope = {**self._local.scope, "role": client.role}
        is_target = identity(scope) == identity(target)
        previous = [row for row in state["attempts"] if identity(row) == identity(target)]
        if is_target:
            if (len(previous) >= 2 or len(state["attempts"]) != len(previous)
                    or previous and not no_content_failure(previous[-1])):
                raise MinimalCallStopped("inherited_retry_pool_exhausted")
            if hashlib.sha256(request.data).hexdigest() != target["request_sha256"] or client.model != target["model"]:
                raise MinimalCallStopped("checkpoint_failed_wire_changed")
        elif not previous or previous[-1]["status"] != "received":
            raise MinimalCallStopped("checkpoint_first_position_unsettled")
        row = super()._reserve(client, request, payload, state)
        if is_target:
            index = len(previous) + 1
            row["checkpoint_retry"] = {"schema": "e1br_checkpoint_retry_link_v1",
                "exception_sha256": self.checkpoint_contract["exception_sha256"],
                "parent_request_id": target["request_id"],
                "parent_receipt_sha256": self.checkpoint_contract["parent_receipt_sha256"],
                "parent_manifest_sha256": self.checkpoint_contract["parent_manifest_sha256"],
                "lineage_attempt_index": index, "old_no_content_proven": False,
                "basis": "explicit_single_position_exception" if index == 1 else "new_parent_no_content_proof"}
        return row
