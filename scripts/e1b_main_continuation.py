"""Restore the explicitly authorized v8 main checkpoint in a new experiment.

No model is replayed here. The original complete persona and twelve saved
turns are retained, including the completed memory checkpoint and its source
receipt. The one pending reflection is continued by the ordinary offline
queue, with its original session, attempt count and expiry. Old originals and
the old failed ledger remain read-only. The caller owns authorization, source
freeze, budgets, and the first-dispatch wire check.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from aria.bench.e1b_cases import load_e1b_cases, synthetic_emotion_plan
from aria.bench.e1b_runner import E1bRunner, E1bRunStopped, _exclusive_json
from aria.engine.model.persona import PersonaProfile
from aria.engine.perception.external_emotion import external_emotion_payload
from aria.voice.minimal import _changes, _stamp
from scripts import run_e1b as base
from scripts import run_e1b_repair as common

PARENT = Path("artifacts/local/e1b-formal-v8")
RUN = Path("main/synthetic-emotion-experiments/e1b-main-v1/run")
CASE = "e1b_a1"
SESSION = "bac723749bd3416795ea587dbd201398"
OPERATION = f"e1b-review-{SESSION}"
RECEIPT_SHA256 = "76b372aefdb6c576b3e6950a2145fd661d44307501a5b65ab16308b46f135a7b"
FAILED_REQUEST = "afe96510f7b7437685f5497154ec17b8"
FAILED_WIRE = "3b61060f16b3db62fad67ecb004608de80181a62f158a64012c40142556f26dc"
PINNED_FILES = {
    "shared/live/run.json": "ddd84ca13ca4ce781f8573ffd95d1c1b0b81f7a7c7ae8af243db6736f4278685",
    "shared/live/profiles/fictional-chen.json": "210448e041686d0d6416c37c3c145b33225b0b8b36515348e0466a59bbb3b716",
    "shared/live/care.sqlite3": "e269cbcf0bff953911e1f3547aa1e341c9f3e58304d83b95c64da3e26dc5259e",
    "memory-s1.attempt.json": "35743bb5fd84e59a1e05d086e261eea4be863d2bb719763533c7a54810ff1980",
    "memory-s1.first.json": "f386c0af12487662b9dd62a97e0f1847132f48203362a7efcbf9ee0a434919c4",
    "memory-s1.product.json": "a60beb68c8bd62bd166a6e658c12c2075ef41d00794453b6d5475fe21ccf8064",
    "raw-proposals.json": "21e18bad11308795d961c7abb405a678066f7ca8108e4033e28464d8f38d2584",
    "reflection-s1.first.json": "49476487db392e5a18e588103119c08d68f460a10af1d13b109a2b0392be0d1a",
}


def _require(value, code):
    if not value:
        raise base.RunError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _exclusive_bytes(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fingerprint(paths):
    _require(all(not p.is_symlink() and not any(q.is_symlink() for q in p.parents)
                 for p in paths), "main_prefix_link_rejected")
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def load_prefix(root):
    """Verify the fixed thirteen received answers and full saved checkpoint.

    This is additional to the entry's complete donor-tree/source audit. The
    raw response of memory was not stored: its unchanged processed checkpoint
    and first-answer hashes are retained, never reconstructed as model text.
    """
    root = Path(root).resolve()
    parent = root / PARENT
    files = {}

    def read(relative):
        path = PARENT / relative
        raw = common.read_original(root, path)
        files[path.as_posix()] = _sha(raw)
        return raw

    receipt = json.loads(read(Path("main/execution-receipt.json")))
    _require(receipt.get("receipt_sha256") == RECEIPT_SHA256 == base.digest({
        k: v for k, v in receipt.items() if k != "receipt_sha256"})
        and receipt.get("status") == "not_evaluable", "main_prefix_receipt_changed")
    registration = json.loads(read(Path("main/registration.json")))
    _require(registration["seal_sha256"] == receipt["registration_sha256"]
        == base.digest({k: v for k, v in registration.items() if k != "seal_sha256"}),
        "main_prefix_registration_changed")
    selected = registration["inputs"]["case_ids"]
    _require(selected == [f"e1b_{g}{n}" for g in "abcd" for n in (1, 2, 3)],
             "main_prefix_selection_changed")
    attempts = [r for r in receipt["accounting"]["attempts"] if r.get("e1b_stage") == "main"]
    _require(len(attempts) == 14 and [r["role"] for r in attempts]
        == ["online_fast"] * 12 + ["memory", "reflection"]
        and [r["status"] for r in attempts] == ["received"] * 13 + ["failed"],
        "main_prefix_attempts_changed")
    wires = []
    for index, attempt in enumerate(attempts, 1):
        relative = Path("main/wire-requests") / f"{index:04d}.json"
        raw = read(relative)
        _require(attempt["wire_request_path"] == relative.as_posix()
            and _sha(raw) == attempt["request_sha256"]
            and attempt["registration_sha256"] == registration["seal_sha256"]
            and attempt["session_id"] == SESSION, "main_prefix_wire_changed")
        wires.append(raw)
    failed = attempts[-1]
    _require(failed["request_id"] == FAILED_REQUEST and failed["request_sha256"] == FAILED_WIRE
        and failed["operation_id"] == OPERATION and failed["usage"] == {},
        "main_prefix_failed_position_changed")
    case_relative = RUN / "main" / CASE
    originals = {name: read(case_relative / name) for name in PINNED_FILES}
    _require(all(_sha(raw) == PINNED_FILES[name] for name, raw in originals.items()),
             "main_prefix_checkpoint_changed")
    # Keep the original failed capture as well, never replace it with the new
    # continuation's first answer.
    for name in ("reflection-s1.attempt.json", "reflection-s1.product.json"):
        originals[name] = read(case_relative / name)
    care_path = parent / case_relative / "shared/live/care.sqlite3"
    originals["shared/live/care.sqlite3"] = read(case_relative / "shared/live/care.sqlite3")
    _require(not any(Path(str(care_path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
             "main_prefix_care_database_not_closed")
    with sqlite3.connect(care_path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        _require(tables == {"elders", "users", "assignments", "sessions", "records", "commands"}
            and all(db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == 0 for table in tables),
            "main_prefix_care_database_not_empty")
    saved_run = json.loads(originals["shared/live/run.json"])
    saved_profile = json.loads(originals["shared/live/profiles/fictional-chen.json"])
    persona = PersonaProfile.model_validate(saved_profile)
    jobs = persona.offline_sessions
    _require(persona.user_id == "fictional-chen"
        and persona.profile_origin == "synthetic_emotion_experiment:e1b-main-v1"
        and persona.pending_session is None and len(jobs) == 1
        and jobs[0].session_id == SESSION and jobs[0].memory_completed is True
        and jobs[0].reflection_completed is False and jobs[0].attempts == 1
        and jobs[0].state == "pending" and jobs[0].active_stage == "reflection"
        and jobs[0].source_completeness == "complete"
        and not jobs[0].excluded_turn_indices and not persona.cognitive_health_consent.text_monitoring_enabled,
        "main_prefix_checkpoint_not_reflection_boundary")
    turns = saved_run["turns"]
    suite = load_e1b_cases()
    case = next(c for c in suite.items if c.id == CASE)
    plan = synthetic_emotion_plan(suite, registration["protocol"]["seed"])
    _require(len(turns) == 12 and len(jobs[0].turns) == 12
        and len(saved_run["sessions"]) == 1 and saved_run["sessions"][0]["status"] == "sealed"
        and saved_run["sessions"][0]["session_id"] == SESSION
        and len(saved_run["reviews"]) == 1 and saved_run["reviews"][0]["status"] == "failed"
        and saved_run["reviews"][0]["request_id"] == OPERATION
        and saved_run["preference_reviews"] == [], "main_prefix_run_boundary_changed")
    for index, (turn, attempt, source) in enumerate(zip(turns, attempts[:12], jobs[0].turns)):
        _require(turn["turn_id"] == attempt["operation_id"] == f"e1b-{SESSION}-t{index}"
            and turn["session_id"] == SESSION and turn["status"] == "completed"
            and turn["text"] == case.current_input(1, index) == source.user
            and turn["response"] == source.agent
            and turn["external_emotion"] == external_emotion_payload(plan[(CASE, 1, index)])
            and turn["source"] == "live"
            and base.SHA.fullmatch(attempt.get("model_reply_sha256", "")),
            "main_prefix_turn_or_emotion_changed")
    memory = json.loads(originals["memory-s1.first.json"])
    product = json.loads(originals["memory-s1.product.json"])
    memory_attempt = json.loads(originals["memory-s1.attempt.json"])
    _require(memory["parse_status"] == "present_array" and memory["request_hash_kind"] == "actual_wire"
        and memory["request_sha256"] == attempts[12]["request_sha256"]
        and memory["visible_response_sha256"] == attempts[12]["client_return_content_sha256"]
        and memory_attempt["session_id"] == SESSION
        and product["product_status"] == "ok" and attempts[12]["operation_id"] == OPERATION,
        "main_prefix_memory_capture_changed")
    summary = {"schema": "e1b_retained_main_prefix_v1", "campaign": PARENT.as_posix(),
        "source_receipt_sha256": RECEIPT_SHA256, "registration_sha256": registration["seal_sha256"],
        "case_id": CASE, "session_id": SESSION, "operation_id": OPERATION,
        "retained_received": 13, "retained_roles": {"online_fast": 12, "memory": 1},
        "retained_delivery": {"model_unmodified": sum(base.text_sha(t["response"]) == a["model_reply_sha256"]
            for t, a in zip(turns, attempts[:12])), "guard_modified": sum(
            base.text_sha(t["response"]) != a["model_reply_sha256"] for t, a in zip(turns, attempts[:12]))},
        "retained_request_ids": [r["request_id"] for r in attempts[:13]],
        "failed_request_id": FAILED_REQUEST, "failed_wire_sha256": FAILED_WIRE,
        "restoration": "complete_durable_checkpoint_not_model_response_replay",
        "original_care_database_empty": True, "care_database_or_accounting_not_cloned": True,
        "timestamps_or_expiry_modified": False, "new_calls_for_restore": 0,
        "source_files": [{"path": k, "sha256": v} for k, v in sorted(files.items())]}
    paths = [root / name for name in sorted(files)]
    return {"root": root, "summary": summary, "summary_sha256": base.digest(summary),
        "registration": registration, "receipt": receipt, "retained_attempts": deepcopy(attempts[:13]),
        "failed_attempt": deepcopy(failed), "failed_payload": json.loads(wires[-1]),
        "failed_request_bytes": wires[-1], "selected_case_ids": list(selected),
        "run_root": parent / RUN, "case_root": parent / case_relative,
        "saved_run": saved_run, "saved_profile": saved_profile, "originals": originals,
        "paths": paths, "hashes": files, "fingerprint": _fingerprint(paths)}


def assert_prefix_unchanged(prefix, *, full=False):
    _require(_fingerprint(prefix["paths"]) == prefix["fingerprint"], "retained_main_prefix_changed")
    _require(base.digest(prefix["summary"]) == prefix["summary_sha256"]
        and prefix["saved_run"] == json.loads(prefix["originals"]["shared/live/run.json"])
        and prefix["saved_profile"] == json.loads(prefix["originals"]["shared/live/profiles/fictional-chen.json"])
        and prefix["failed_payload"] == json.loads(prefix["failed_request_bytes"])
        and _sha(prefix["failed_request_bytes"]) == FAILED_WIRE
        and all(_sha(prefix["originals"][name]) == sha for name, sha in PINNED_FILES.items()),
        "retained_main_prefix_memory_changed")
    attempts = [r for r in prefix["receipt"]["accounting"]["attempts"] if r.get("e1b_stage") == "main"]
    _require(prefix["retained_attempts"] == attempts[:13] and prefix["failed_attempt"] == attempts[-1]
        and prefix["receipt"]["receipt_sha256"] == RECEIPT_SHA256 == base.digest({
            k: v for k, v in prefix["receipt"].items() if k != "receipt_sha256"}),
        "retained_main_prefix_receipts_changed")
    if full:
        for name, sha in prefix["hashes"].items():
            _require(_sha(common.read_original(prefix["root"], Path(name))) == sha,
                     "retained_main_prefix_bytes_changed")


def runner_class(prefix, *, base_runner=E1bRunner):
    """Return the ordinary runner with one fixed checkpoint continuation hook.

    The new caller must create a fresh e1b-main-v1 experiment and a new guarded
    ledger that carries the old costs/holds. No registration row is rewritten.
    """
    class MainContinuationRunner(base_runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._prefix_restored = False
            self._prefix_reflection_started = False
            assert_prefix_unchanged(prefix)
            _require(self.suite.source_sha256 == json.loads(
                prefix["originals"]["memory-s1.attempt.json"])["source_sha256"],
                "main_prefix_suite_changed")
            _require(self.seed == prefix["registration"]["protocol"]["seed"], "main_prefix_seed_changed")

        def _session(self, app, case, number, *, arm, repeat):
            if case.id == CASE and number == 1 and arm == "shared" and repeat == 0:
                _require(not self._prefix_restored and self._sequence == 0,
                         "main_prefix_restore_not_first_operation")
                self._prefix_restored = True
                self._operation(app, "restore_retained_prefix", lambda: self._restore(app),
                    detail={"case_id": CASE, "session": 1, "retained_answers": 13,
                            "new_model_calls": 0, "prefix_sha256": prefix["summary_sha256"]})
                return SESSION
            _require(self._prefix_reflection_started, "main_prefix_reflection_must_be_first")
            return super()._session(app, case, number, arm=arm, repeat=repeat)

        def _restore(self, app):
            assert_prefix_unchanged(prefix, full=True)
            _require(app.experiment.profile_origin == prefix["saved_profile"]["profile_origin"]
                and not app._run["turns"] and not app._run["sessions"] and not app._run["reviews"]
                and app._profile() is None, "main_prefix_requires_fresh_isolated_app")
            if app.call_guard is not None:
                _require(app.call_guard.snapshot()["calls_used"] == 0,
                         "main_prefix_requires_no_new_dispatch")
            job = prefix["saved_profile"]["offline_sessions"][0]
            _require(datetime.fromisoformat(job["expires_at"]) > datetime.now(timezone.utc),
                     "main_prefix_original_retention_expired")
            case_root = self.root / "main" / CASE
            originals_root = case_root / "retained-prefix"
            originals_root.mkdir(exist_ok=False)
            for name, raw in prefix["originals"].items():
                target = originals_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _exclusive_bytes(target, raw)
            for name in ("memory-s1.attempt.json", "memory-s1.first.json", "memory-s1.product.json",
                         "raw-proposals.json"):
                _exclusive_bytes(case_root / name, prefix["originals"][name])
            _exclusive_json(case_root / "retained-prefix.json", prefix["summary"])
            # Save the entire original checkpoint, including trust, history,
            # source turns, memory completion, attempt count and fixed TTL.
            destination = app.store._path("fictional-chen")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _exclusive_bytes(destination, prefix["originals"]["shared/live/profiles/fictional-chen.json"])
            restored = deepcopy(prefix["saved_run"])
            restored["retained_prior_reviews"] = restored["reviews"]
            restored["reviews"] = []
            restored["retained_origin"] = {"campaign": PARENT.as_posix(),
                "run_id": restored["run_id"], "model_allowance": restored.get("model_allowance"),
                "prefix_sha256": prefix["summary_sha256"], "answers": 13,
                "new_model_calls": 0, "fees_recounted": False}
            restored["run_id"] = app._run["run_id"]
            if app.call_guard is not None:
                restored["model_allowance"] = deepcopy(app._run["model_allowance"])
            else:
                restored.pop("model_allowance", None)
            app._run = restored
            app._save_run()
            _require(app._profile().model_dump(mode="json") == PersonaProfile.model_validate(
                prefix["saved_profile"]).model_dump(mode="json"), "main_prefix_profile_restore_changed")
            return {"ok": True, "retained": 13, "new_model_calls": 0}

        def _operation(self, app, name, operation, *, detail=None):
            if (name == "reflect" and detail == {"case_id": CASE, "session": 1}):
                _require(self._prefix_restored and not self._prefix_reflection_started,
                         "main_prefix_reflection_already_consumed")
                self._prefix_reflection_started = True
                return super()._operation(app, name, lambda: self._continue_reflection(app),
                    detail={**detail, "continuation_of_request_id": FAILED_REQUEST,
                            "prefix_sha256": prefix["summary_sha256"]})
            return super()._operation(app, name, operation, detail=detail)

        def _continue_reflection(self, app):
            assert_prefix_unchanged(prefix)
            persona = app._profile()
            _require(persona.model_dump(mode="json") == PersonaProfile.model_validate(
                prefix["saved_profile"]).model_dump(mode="json"), "main_prefix_state_changed_before_reflection")
            if app.call_guard is not None:
                _require(app.call_guard.snapshot()["calls_used"] == 0,
                         "main_prefix_reflection_not_first_new_dispatch")
            _require(not app._unresolved() and not app.state()["session_id"],
                     "main_prefix_unresolved_new_operation")
            before = app._understanding(persona)
            row = {"request_id": OPERATION, "session_id": SESSION,
                "status": "pending", "created_at": _stamp(),
                "continuation_of_request_id": FAILED_REQUEST,
                "retained_prefix_sha256": prefix["summary_sha256"]}
            app._run["reviews"].append(row)
            app._save_run()
            try:
                engine = app._engine_for_persona(persona)
                # The normal queue sees memory_completed=True. It preserves
                # the original raw conversation and dispatches only reflection.
                with app._model_scope("review", OPERATION, SESSION):
                    result = engine.recover_offline_sessions(max_jobs=1)
                success = result.get("last_run", {}).get("completed_jobs") == 1
                row.update(status="completed" if success else "failed", result=result,
                    changes=_changes(before, app._understanding(engine.persona), SESSION),
                    finished_at=_stamp())
                if not success:
                    row["requires_review"] = True
                    app._halt_models()
            except Exception:
                row.update(status="failed", requires_review=True, finished_at=_stamp())
                app._halt_models()
                raise
            finally:
                app._save_run()
            if not success:
                raise E1bRunStopped("authorized_reflection_continuation_failed")
            return app._result(True)

    return MainContinuationRunner
