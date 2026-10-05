"""Explicit exploratory continuation after a retained, known-cost format failure.

This is an experiment-only policy. It never changes the failed receipt to a
successful model answer and never retries an answer-bearing failed position.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from aria.bench.e1b_runner import E1bExperimentApp
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.engine.policy.online_json import parse_json_decision

SCHEMA = "e1br_known_format_terminal_v1"
SHA = re.compile(r"[a-f0-9]{64}\Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def terminal_format_failure(row):
    """Only complete HTTP-success/schema-rejected first bodies with known usage."""
    try:
        usage, reserve = row["usage"], row["reservation"]
        cost, held = Decimal(str(usage["cost"])), Decimal(reserve["reserved_cost_usd"])
        return bool(
            row.get("role") == "online_fast" and row.get("status") == "failed"
            and row.get("product_step") == "initial" and row.get("http_status") == 200
            and row.get("failure_phase") == "receipt_validation"
            and row.get("validation_error") == row.get("failure_code") == "online_decision_schema_invalid"
            and row.get("reply_return_channel") == "json_object"
            and row.get("observation", {}).get("body_status") == "parsed"
            and row.get("observation", {}).get("envelope_kind") == "success"
            and row.get("receipt_identity") == {"version": "receipt_identity_v1",
                "error_field_status": "absent", "model_status": "recognized", "provider_status": "recognized"}
            and all(SHA.fullmatch(row.get(key, "")) for key in
                    ("request_sha256", "visible_original_sha256", "visible_content_sha256", "client_return_content_sha256"))
            and row["client_return_content_sha256"] == row["visible_content_sha256"]
            and not row.get("model_reply_sha256") and not row.get("no_content_evidence")
            and cost.is_finite() and held.is_finite() and 0 <= cost <= held
            and type(usage.get("prompt_tokens")) is int and 0 <= usage["prompt_tokens"] <= reserve["input_token_plan_bound"]
            and type(usage.get("completion_tokens")) is int and 0 <= usage["completion_tokens"] <= reserve["output_token_limit"]
            and usage.get("reasoning_tokens", 0) == 0)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return False


def _read_evidence(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MinimalCallStopped("invalid_terminal_format_original_path")
    path = Path(root)
    for part in relative.parts:
        path = path / part
        stat = path.lstat()
        if path.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
            raise MinimalCallStopped("invalid_terminal_format_original_path")
    if Path(root).resolve() not in path.resolve().parents:
        raise MinimalCallStopped("invalid_terminal_format_original_path")
    return path.read_bytes()


def _verify_terminal_files(root, row, proof):
    if (not terminal_format_failure(row) or proof.get("schema") != SCHEMA
            or proof.get("receipt_sha256") != digest(row) or proof.get("request_id") != row["request_id"]
            or proof.get("turn_id") != row["operation_id"] or proof.get("session_id") != row["session_id"]
            or proof.get("historical") is not False or proof.get("model_reply_eligible") is not False
            or proof.get("model_metrics", "missing") is not None or proof.get("body_retried") is not False
            or proof.get("fixed_denominator_retained") is not True
            or proof.get("delivered_response") != DETERMINISTIC_MODEL_FAILURE_REPLY
            or proof.get("fallback_sha256") != hashlib.sha256(DETERMINISTIC_MODEL_FAILURE_REPLY.encode()).hexdigest()):
        raise MinimalCallStopped("terminal_format_delivery_binding_changed")
    raw = _read_evidence(root, row["visible_original_path"])
    saved = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != row["visible_original_sha256"]
            or saved.get("request_id") != row["request_id"]
            or hashlib.sha256(saved["content"].encode()).hexdigest() != row["visible_content_sha256"]):
        raise MinimalCallStopped("terminal_format_original_changed")
    if json.loads(_read_evidence(root, proof["delivery_proof_path"])) != proof:
        raise MinimalCallStopped("terminal_format_saved_delivery_changed")
    run = json.loads(_read_evidence(root, proof["run_path"]))
    turns = [turn for turn in run["turns"] if (turn.get("turn_id"), turn.get("session_id")) ==
             (row["operation_id"], row["session_id"])]
    if (len(turns) != 1 or digest(turns[0]) != proof["turn_sha256"] or turns[0]["status"] != "failed"
            or turns[0]["source"] != "model_failed" or turns[0]["response"] != ""):
        raise MinimalCallStopped("terminal_format_saved_turn_changed")
    try:
        parse_json_decision(saved["content"], row["online_decision_version"])
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        pass
    else:
        raise MinimalCallStopped("terminal_format_body_unexpectedly_valid")


class FormatGuardMixin:
    """Durably bind the opt-in and require saved fallback before the next call."""

    def __init__(self, *args, format_contract, format_evidence_root, **kwargs):
        if (type(format_contract) is not dict or format_contract.get("schema") != SCHEMA
                or format_contract.get("retry_answer_bearing_failure") is not False
                or format_contract.get("effect_claim_eligible") is not False):
            raise ValueError("format_continuation_contract_required")
        self._format_contract = json.dumps(format_contract, sort_keys=True, allow_nan=False)
        self._format_root = Path(format_evidence_root).resolve()
        self._format_ready = False
        super().__init__(*args, **kwargs)
        try:
            with self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_format_policy (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO e1br_format_policy VALUES(1,?)", (self._format_contract,))
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_format_delivery (request_id TEXT PRIMARY KEY, proof TEXT NOT NULL)")
            self._format_ready = True
            self._snapshot()
        except BaseException:
            self._db.close()
            raise

    def terminal_primary_failure(self, row):
        return self._format_ready and terminal_format_failure(row)

    def _snapshot(self):
        if self._format_ready and self._db.execute(
                "SELECT contract FROM e1br_format_policy WHERE id=1").fetchone() != (self._format_contract,):
            raise ValueError("format_continuation_contract_changed")
        return super()._snapshot()

    def _begin(self, *args, **kwargs):
        for row in self._snapshot()["attempts"]:
            if terminal_format_failure(row):
                proof = self._db.execute("SELECT proof FROM e1br_format_delivery WHERE request_id=?", (row["request_id"],)).fetchone()
                if proof is None or json.loads(proof[0]).get("receipt_sha256") != digest(row):
                    raise MinimalCallStopped("format_fallback_delivery_not_saved")
                _verify_terminal_files(self._format_root, row, json.loads(proof[0]))
        return super()._begin(*args, **kwargs)

    def record_terminal_delivery(self, row, proof):
        if not terminal_format_failure(row) or proof.get("receipt_sha256") != digest(row):
            raise MinimalCallStopped("invalid_terminal_format_delivery")
        _verify_terminal_files(self._format_root, row, proof)
        with self._transaction():
            current = self._db.execute("SELECT record FROM attempts WHERE request_id=?", (row["request_id"],)).fetchone()
            if current is None or json.loads(current[0]) != row:
                raise MinimalCallStopped("terminal_format_receipt_changed")
            self._db.execute("INSERT INTO e1br_format_delivery VALUES(?,?)",
                             (row["request_id"], json.dumps(proof, sort_keys=True, allow_nan=False)))


def validate_format_ledger(path, *, format_contract, attempts):
    """Read-only opt-in and delivery binding; ordinary ledger checks stay separate."""
    path = Path(path).resolve()
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("format_ledger_not_closed")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        contract = db.execute("SELECT contract FROM e1br_format_policy WHERE id=1").fetchone()
        if contract != (json.dumps(format_contract, sort_keys=True, allow_nan=False),):
            raise ValueError("format_continuation_contract_changed")
        stored = [json.loads(row[0]) for row in db.execute("SELECT record FROM attempts ORDER BY sequence")]
        if stored != attempts:
            raise ValueError("format_ledger_attempts_changed")
        proofs = {key: json.loads(raw) for key, raw in db.execute("SELECT request_id,proof FROM e1br_format_delivery")}
    expected = {row["request_id"]: row for row in attempts if terminal_format_failure(row)}
    if set(proofs) != set(expected):
        raise ValueError("format_terminal_delivery_inventory_changed")
    for key, row in expected.items():
        value = proofs[key]
        _verify_terminal_files(path.parent, row, value)
        if (value.get("schema") != SCHEMA or value.get("receipt_sha256") != digest(row)
                or value.get("request_id") != key or value.get("turn_id") != row["operation_id"]
                or value.get("session_id") != row["session_id"] or value.get("historical") is not False
                or value.get("delivered_response") != DETERMINISTIC_MODEL_FAILURE_REPLY
                or value.get("fallback_sha256") != hashlib.sha256(DETERMINISTIC_MODEL_FAILURE_REPLY.encode()).hexdigest()
                or value.get("model_reply_eligible") is not False or value.get("model_metrics", "missing") is not None
                or value.get("body_retried") is not False or value.get("fixed_denominator_retained") is not True
                or not SHA.fullmatch(value.get("turn_sha256", ""))):
            raise ValueError("format_terminal_delivery_binding_changed")
    return list(proofs.values())


class FormatExperimentApp(E1bExperimentApp):
    """Preserve failed public turn records and the actual deterministic context."""

    def __init__(self, *args, **kwargs):
        self._format_submit_active = False
        self._format_terminal = {}
        super().__init__(*args, **kwargs)

    def _candidate(self):
        if self._storage_failed or not isinstance(self.call_guard, FormatGuardMixin):
            return None
        state = self.call_guard.snapshot()
        if state["stopped"] or not state["attempts"] or not self._run["turns"]:
            return None
        row, turn = state["attempts"][-1], self._run["turns"][-1]
        if (terminal_format_failure(row) and row["operation_id"] == turn["turn_id"]
                and row["session_id"] == turn["session_id"] and turn["status"] == "failed"
                and turn["source"] == "model_failed" and turn["response"] == ""):
            return row
        return None

    def _halt_models(self):
        # Do not clear a halt. Only defer this one expected product-format halt
        # until its already-saved delivery can be recorded below. Storage,
        # unknown, transport, manual stop and all other errors still halt.
        if self._format_submit_active and self._candidate() is not None:
            return
        return super()._halt_models()

    def _unresolved(self):
        return self._storage_failed or any(
            row["status"] == "pending" or row.get("requires_review") and
            (row.get("turn_id"), row.get("session_id")) not in self._format_terminal
            for row in [*self._run["turns"], *self._run["reviews"], *self._run["preference_reviews"]])

    def retain_terminal(self, row, *, historical=False):
        turn = self._run["turns"][-1]
        pair = self._sessions.get("fictional-chen")
        if (not terminal_format_failure(row) or pair is None or self._storage_failed
                or pair[1].turns[-1].agent != DETERMINISTIC_MODEL_FAILURE_REPLY
                or turn["status"] != "failed" or turn["source"] != "model_failed" or turn["response"] != ""
                or row["operation_id"] != turn["turn_id"] or row["session_id"] != turn["session_id"]):
            raise MinimalCallStopped("terminal_format_fallback_context_changed")
        proof = {"schema": SCHEMA, "request_id": row["request_id"], "receipt_sha256": digest(row),
            "turn_id": turn["turn_id"], "session_id": turn["session_id"],
            "turn_sha256": digest(turn), "fallback_sha256": hashlib.sha256(DETERMINISTIC_MODEL_FAILURE_REPLY.encode()).hexdigest(),
            "delivered_response": DETERMINISTIC_MODEL_FAILURE_REPLY,
            "model_reply_eligible": False, "model_metrics": None, "historical": historical,
            "body_retried": False, "fixed_denominator_retained": True}
        path = self.run_dir / f"format-terminal-{row['request_id']}.json"
        if not historical:
            proof.update(delivery_proof_path=path.relative_to(self.call_guard._format_root).as_posix(),
                         run_path=self.run_path.relative_to(self.call_guard._format_root).as_posix())
        with path.open("x", encoding="utf8") as stream:
            json.dump(proof, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        if not historical:
            self.call_guard.record_terminal_delivery(row, proof)
        self._format_terminal[turn["turn_id"], turn["session_id"]] = proof
        return proof

    def submit(self, *args, **kwargs):
        self._format_submit_active = True
        try:
            result = super().submit(*args, **kwargs)
            if result.get("ok") is False:
                row = self._candidate()
                if row is not None:
                    proof = self.retain_terminal(row)
                    # This means the experiment operation durably ended, not
                    # that the model replied successfully. Public row stays failed.
                    return {**result, "ok": True, "model_reply_valid": False,
                            "terminal_format_failure": deepcopy(proof), "state": self.state()}
            return result
        except BaseException:
            self._format_submit_active = False
            self._halt_models()
            raise
        finally:
            self._format_submit_active = False

    def state(self):
        value = super().state()
        value["format_terminal_positions"] = list(deepcopy(self._format_terminal).values())
        value["effect_claim_eligible"] = False
        return value
