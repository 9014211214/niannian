"""Opt-in experimental missing-memory terminals; no retry or successful memory.

Only a complete, durably captured, known-cost JSON syntax failure may reach the
independent reflection stage. Product evidence, semantic and storage failures
remain stops. This module never grants an allowance or clears a halted ledger.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sqlite3

from aria.bench.e1b_runner import _exclusive_json
from aria.engine.memory.consolidator import _parse_json_payload, _strip_fences
from aria.engine.memory.offline_queue import OfflineSessionQueue, _OFFLINE_RESULT_FIELDS
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.minimal_calls import MinimalCallStopped
from scripts.e1br_format_runtime import FormatExperimentApp, FormatGuardMixin, SHA, _read_evidence, digest

SCHEMA = "e1br_memory_terminal_missing_v1"


def require(value, code):
    if not value:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require_unexpired_job(job):
    created, expires = datetime.fromisoformat(job.created_at), datetime.fromisoformat(job.expires_at)
    require(created.tzinfo is not None and expires.tzinfo is not None and created < expires,
            "memory_original_retention_window_invalid")
    require(datetime.now(timezone.utc) < expires, "memory_original_job_expired")


def complete_known_memory(row):
    try:
        usage, reserve = row["usage"], row["reservation"]
        cost, held = Decimal(str(usage["cost"])), Decimal(str(reserve["reserved_cost_usd"]))
        return bool(row.get("role") == "memory" and row.get("product_step") == "initial"
            and row.get("status") == "received" and row.get("http_status") == 200
            and row.get("observation", {}).get("body_status") == "parsed"
            and row.get("observation", {}).get("envelope_kind") == "success"
            and row.get("receipt_identity") == {"version": "receipt_identity_v1",
                "error_field_status": "absent", "model_status": "recognized", "provider_status": "recognized"}
            and all(SHA.fullmatch(row.get(key, "")) for key in
                    ("request_sha256", "visible_original_sha256", "visible_content_sha256", "client_return_content_sha256"))
            and row["client_return_content_sha256"] == row["visible_content_sha256"]
            and not row.get("failure_code") and not row.get("validation_error")
            and cost.is_finite() and held.is_finite() and 0 <= cost <= held
            and type(usage.get("prompt_tokens")) is int and 0 <= usage["prompt_tokens"] <= reserve["input_token_plan_bound"]
            and type(usage.get("completion_tokens")) is int and 0 <= usage["completion_tokens"] <= reserve["output_token_limit"]
            and usage.get("reasoning_tokens", 0) == 0)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return False


def json_syntax_failure(content):
    if type(content) is not str or not _strip_fences(content).startswith(("{", "[")):
        return False
    try:
        _parse_json_payload(content)
    except json.JSONDecodeError:
        return True
    except (ValueError, TypeError, RecursionError):
        return False
    return False


def terminal_memory_evidence(row, visible, first, product, profile, run, position):
    position = tuple(position)
    require(len(position) == 6 and position[0] == "memory" and position[2] in (1, 2)
            and position[3:] == ("shared", 0, None) and tuple(row.get("experiment_position", ())) == position
            and complete_known_memory(row), "memory_complete_known_received_required")
    require(visible.get("request_id") == row["request_id"] and visible.get("role") == "memory"
            and type(visible.get("content")) is str, "memory_visible_identity_changed")
    content_sha = sha(visible["content"].encode())
    require(content_sha == row["visible_content_sha256"] == first.get("visible_response_sha256")
            and first.get("request_sha256") == row["request_sha256"]
            and first.get("request_hash_kind") == "actual_wire", "memory_first_body_binding_changed")
    require(first.get("parse_status") == "parse_error" and first.get("raw_proposals") is None
            and first.get("field_present") is None and product.get("product_status") == "failed"
            and product.get("accepted_count") is None and product.get("accepted_preferences") == []
            and product.get("diagnostics_available") is False,
            "memory_failure_is_not_empty_or_success")
    require(first.get("session") == product.get("session") == position[2], "memory_capture_session_changed")
    require(json_syntax_failure(visible["content"]), "memory_not_json_syntax_failure")
    jobs = profile.get("offline_sessions", [])
    require(len(jobs) == 1 and profile.get("pending_session") is None, "memory_single_sealed_job_required")
    job = jobs[0]
    require(job["session_id"] == row["session_id"] and job["memory_completed"] is False
            and job["reflection_completed"] is False and job["source_completeness"] == "complete"
            and job["active_stage"] == "memory" and job["last_error_code"] == "memory_extraction_failed",
            "memory_original_failed_job_changed")
    n = position[2]
    require(len(run["sessions"]) == n and len(run["turns"]) == n * 12 and len(job["turns"]) == 12
            and all(session["status"] == "sealed" for session in run["sessions"])
            and run["sessions"][-1]["session_id"] == job["session_id"], "memory_sealed_context_changed")
    require(all(old["text"] == turn["user"] for old, turn in zip(run["turns"][-12:], job["turns"], strict=True)),
            "memory_original_inputs_changed")
    review = run["reviews"][-1]
    require(review["status"] == "failed" and review["session_id"] == job["session_id"]
            and review["request_id"] == row["operation_id"], "memory_failed_review_changed")
    return {"schema": SCHEMA, "position": list(position), "request_id": row["request_id"],
        "receipt_sha256": digest(row), "visible_content_sha256": content_sha,
        "first_capture_sha256": digest(first), "product_capture_sha256": digest(product),
        "original_profile_sha256": digest(profile), "original_run_sha256": digest(run),
        "original_job_sha256": digest(job), "old_failed_review_sha256": digest(review),
        "product_status": "failed", "parse_status": "parse_error", "raw_proposals": None,
        "accepted_count": None, "memory_completed": False, "memory_result": "terminal_missing",
        "memory_changed": False, "original_failure_preserved": True, "retry_allowed": False,
        "formation_result": "not_evaluable", "model_effect_claim": False,
        "source_created_at": job["created_at"], "source_expires_at": job["expires_at"]}


def validate_memory_terminal_proof(case_directory, proof, row):
    """Re-read exact saved evidence; paths are relative to the one case directory."""
    require(proof.get("schema") == SCHEMA, "memory_terminal_schema_changed")
    raw = {field: _read_evidence(case_directory, proof[field]) for field in
           ("visible_path", "first_path", "product_path", "baseline_profile_path", "baseline_run_path")}
    values = {key: json.loads(value) for key, value in raw.items()}
    require(sha(raw["visible_path"]) == row["visible_original_sha256"], "memory_visible_file_changed")
    expected = terminal_memory_evidence(row, values["visible_path"], values["first_path"], values["product_path"],
        values["baseline_profile_path"], values["baseline_run_path"], proof["position"])
    require(all(proof.get(key) == value and type(proof.get(key)) is type(value) for key, value in expected.items()),
            "memory_terminal_evidence_changed")
    require(json.loads(_read_evidence(case_directory, proof["proof_path"])) == proof,
            "memory_terminal_saved_proof_changed")
    if proof.get("historical") is True:
        binding = proof["source_baseline"]
        original_run = deepcopy(values["baseline_run_path"])
        original_run["run_id"] = binding["run_id"]
        if binding["model_allowance"] is None:
            original_run.pop("model_allowance", None)
        else:
            original_run["model_allowance"] = binding["model_allowance"]
        require(digest(original_run) == binding["original_run_sha256"]
                and expected["original_profile_sha256"] == binding["original_profile_sha256"]
                and expected["original_job_sha256"] == binding["original_job_sha256"]
                and expected["old_failed_review_sha256"] == binding["old_failed_review_sha256"],
                "memory_historical_baseline_changed")
    else:
        require(proof.get("historical") is False and proof.get("source_baseline") is None,
                "memory_terminal_origin_changed")
    return expected


class MemoryGuardMixin(FormatGuardMixin):
    """Bind the opted-in missing-memory rule and proofs in the shared ledger."""

    def __init__(self, *args, memory_contract, memory_evidence_root, **kwargs):
        require(type(memory_contract) is dict and memory_contract.get("schema") == SCHEMA
                and memory_contract.get("retry_answer_bearing_failure") is False
                and memory_contract.get("effect_claim_eligible") is False,
                "memory_continuation_contract_required")
        self._memory_contract = json.dumps(memory_contract, sort_keys=True, allow_nan=False)
        self._memory_root, self._memory_ready = Path(memory_evidence_root).resolve(), False
        super().__init__(*args, **kwargs)
        try:
            with self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_memory_policy (id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO e1br_memory_policy VALUES(1,?)", (self._memory_contract,))
                self._db.execute("CREATE TABLE IF NOT EXISTS e1br_memory_terminal (request_id TEXT PRIMARY KEY, case_path TEXT NOT NULL, proof TEXT NOT NULL)")
            self._memory_ready = True
            self._snapshot()
        except BaseException:
            self._db.close()
            raise

    def _snapshot(self):
        if self._memory_ready and self._db.execute("SELECT contract FROM e1br_memory_policy WHERE id=1").fetchone() != (self._memory_contract,):
            raise ValueError("memory_continuation_contract_changed")
        return super()._snapshot()

    def _begin(self, *args, **kwargs):
        state = self._snapshot()
        for row in state["attempts"]:
            if complete_known_memory(row):
                visible = json.loads(_read_evidence(self._memory_root, row["visible_original_path"]))
                if json_syntax_failure(visible.get("content")):
                    found = self._db.execute("SELECT case_path,proof FROM e1br_memory_terminal WHERE request_id=?", (row["request_id"],)).fetchone()
                    if found is None:
                        raise MinimalCallStopped("memory_terminal_proof_not_saved")
                    validate_memory_terminal_proof(self._memory_root / found[0], json.loads(found[1]), row)
        return super()._begin(*args, **kwargs)

    def record_memory_terminal(self, case_directory, row, proof):
        require(proof.get("historical") is False, "memory_new_receipt_required")
        validate_memory_terminal_proof(case_directory, proof, row)
        relative = Path(case_directory).resolve().relative_to(self._memory_root).as_posix()
        with self._transaction():
            current = self._db.execute("SELECT record FROM attempts WHERE request_id=?", (row["request_id"],)).fetchone()
            require(current is not None and json.loads(current[0]) == row, "memory_receipt_changed")
            self._db.execute("INSERT INTO e1br_memory_terminal VALUES(?,?,?)",
                (row["request_id"], relative, json.dumps(proof, sort_keys=True, allow_nan=False)))


def validate_memory_ledger(path, *, memory_contract, attempts):
    path = Path(path).resolve()
    require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")), "memory_ledger_not_closed")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require(db.execute("SELECT contract FROM e1br_memory_policy WHERE id=1").fetchone()
                == (json.dumps(memory_contract, sort_keys=True, allow_nan=False),), "memory_continuation_contract_changed")
        require([json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")] == attempts,
                "memory_ledger_attempts_changed")
        proofs = {key: (relative, json.loads(raw)) for key, relative, raw in
                  db.execute("SELECT request_id,case_path,proof FROM e1br_memory_terminal")}
    expected = {}
    for row in attempts:
        if complete_known_memory(row):
            visible = json.loads(_read_evidence(path.parent, row["visible_original_path"]))
            if json_syntax_failure(visible.get("content")):
                expected[row["request_id"]] = row
    require(set(expected) == set(proofs), "memory_terminal_inventory_changed")
    for key, row in expected.items():
        relative, proof = proofs[key]
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "memory_case_path_changed")
        validate_memory_terminal_proof(path.parent / relative, proof, row)
    return [proof for _, proof in proofs.values()]


class MemoryExperimentApp(FormatExperimentApp):
    """Experiment-only route; the ordinary product queue and failed rows stay intact."""

    def __init__(self, *args, **kwargs):
        self._memory_active, self._memory_terminals = False, {}
        super().__init__(*args, **kwargs)

    def _memory_candidate(self):
        if self._storage_failed or not isinstance(self.call_guard, MemoryGuardMixin) or self.proposal_capture is None:
            return None
        state = self.call_guard.snapshot()
        if state["stopped"] or not state["attempts"] or not self._run["reviews"]:
            return None
        row = state["attempts"][-1]
        cap = self.proposal_capture
        try:
            visible_raw = _read_evidence(self.call_guard._memory_root, row["visible_original_path"])
            require(sha(visible_raw) == row["visible_original_sha256"], "memory_visible_file_changed")
            profile = self._profile().model_dump(mode="json")
            first, product = json.loads(cap.first_path.read_bytes()), json.loads(cap.outcome_path.read_bytes())
            terminal_memory_evidence(row, json.loads(visible_raw), first, product, profile, self._run, row["experiment_position"])
            return row, visible_raw, profile
        except (KeyError, TypeError, ValueError, OSError):
            return None

    def _halt_models(self):
        if self._memory_active and self._memory_candidate() is not None:
            return
        return super()._halt_models()

    def _unresolved(self):
        ignored = {proof["old_failed_review_sha256"] for proof in self._memory_terminals.values()}
        return self._storage_failed or any(
            row["status"] == "pending" or row.get("requires_review")
            and (row.get("turn_id"), row.get("session_id")) not in self._format_terminal
            and digest(row) not in ignored
            for row in [*self._run["turns"], *self._run["reviews"], *self._run["preference_reviews"]])

    def retain_memory_terminal(self, row, visible_raw, *, historical=False, source_campaign=None, source_baseline=None):
        require(not self._storage_failed and self.proposal_capture is not None, "memory_capture_required")
        cap, profile = self.proposal_capture, self._profile().model_dump(mode="json")
        proof = terminal_memory_evidence(row, json.loads(visible_raw), json.loads(cap.first_path.read_bytes()),
            json.loads(cap.outcome_path.read_bytes()), profile, self._run, row["experiment_position"])
        require_unexpired_job(self._profile().offline_sessions[0])
        name = f"memory-s{cap.session}.terminal"
        directory = cap.directory
        paths = {"visible_path": name + ".visible.json", "baseline_profile_path": name + ".profile.json",
                 "baseline_run_path": name + ".run.json", "proof_path": name + ".json",
                 "first_path": cap.first_path.name, "product_path": cap.outcome_path.name}
        with (directory / paths["visible_path"]).open("xb") as stream:
            stream.write(visible_raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        _exclusive_json(directory / paths["baseline_profile_path"], profile)
        _exclusive_json(directory / paths["baseline_run_path"], self._run)
        proof.update(paths, historical=historical, source_campaign=source_campaign, source_baseline=deepcopy(source_baseline),
                     fixed_denominator_retained=True, body_retried=False)
        _exclusive_json(directory / paths["proof_path"], proof)
        validate_memory_terminal_proof(directory, proof, row)
        if not historical:
            self.call_guard.record_memory_terminal(directory, row, proof)
        self._memory_terminals[row["session_id"]] = proof
        return proof

    def resume_missing_reflection(self, request_id):
        persona = self._profile()
        require(persona is not None and len(persona.offline_sessions) == 1, "memory_failed_job_required")
        job = persona.offline_sessions[0]
        proof = self._memory_terminals.get(job.session_id)
        require(proof is not None and digest(persona.model_dump(mode="json")) == proof["original_profile_sha256"],
                "memory_baseline_changed_before_reflection")
        require_unexpired_job(job)
        reviewer = persona.model_copy(deep=True)
        ctx = OfflineSessionQueue._context(reviewer, job)
        ctx.external_emotion_experiment = self.experiment
        ctx.working_memory.pop("_cognitive_source_turns", None)
        ctx.working_memory.pop("_cognitive_session_at", None)
        reviewer.cognitive_observations, reviewer.offline_sessions, reviewer.offline_session_audit = [], [], []
        reviewer.pending_session = None
        engine = self._engine_for_persona(persona)
        with self._model_scope("review", request_id, job.session_id):
            proposed, status, error = engine._offline_reflection_stage(reviewer, ctx)
        require(status in {"ok", "skipped", "sealed_failed"}, "memory_following_reflection_not_terminal")
        require_unexpired_job(job)
        queue = OfflineSessionQueue(persona)
        result = queue._merge_result(proposed, stage="reflection") if status in {"ok", "skipped"} else persona.model_copy(deep=True)
        active = result.offline_sessions[0]
        require(active.memory_completed is False, "memory_success_flag_forbidden")
        active.reflection_completed = status in {"ok", "skipped"}
        queue._finish(result, active, datetime.now(timezone.utc), outcome="memory_terminal_missing", error_code="memory_extraction_failed")
        require(result.episodic_memory == persona.episodic_memory and result.stated_preferences == persona.stated_preferences
                and result.relationship_context == persona.relationship_context, "failed_memory_changed_profile")
        self.store.save(result)
        audit = {"schema": "e1br_memory_reflection_completion_v1", "memory_proof_sha256": digest(proof),
                 "memory_completed": False, "reflection_status": status, "reflection_error": error,
                 "offline_audit": result.offline_session_audit[-1].model_dump(mode="json"),
                 "session_completion_semantics": "review_terminal_with_missing_memory_not_memory_success"}
        _exclusive_json(self.proposal_capture.directory / f"memory-s{self.proposal_capture.session}.terminal.reflection.json", audit)
        self._run.setdefault("memory_terminal_positions", []).append(deepcopy(proof))
        self._save_run()
        return {"ok": True, "memory_product_status": "failed", "memory_completed": False,
                "memory_candidates": None, "reflection_status": status, "memory_terminal": deepcopy(proof)}

    def reflect(self, request_id, *, wait=True):
        require(wait is True, "memory_experiment_requires_synchronous_review")
        self._memory_active = True
        before = self._profile()
        try:
            result = super().reflect(request_id, wait=True)
            if result.get("ok") is False:
                selected = self._memory_candidate()
                if selected is not None:
                    row, visible_raw, profile = selected
                    require(before is not None and all(profile[field] == before.model_dump(mode="json")[field]
                            for field in _OFFLINE_RESULT_FIELDS), "failed_memory_modified_existing_knowledge")
                    self.retain_memory_terminal(row, visible_raw)
                    return self.resume_missing_reflection(request_id)
            return result
        except BaseException:
            self._memory_active = False
            self._halt_models()
            raise
        finally:
            self._memory_active = False

    def state(self):
        value = super().state()
        value["memory_terminal_positions"] = list(deepcopy(self._memory_terminals).values())
        return value
