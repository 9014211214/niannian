"""Keep 372 first bodies, including one actual fallback; resume 684 new slots."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from aria.bench.e1br_runner import E1BRRunner
from aria.engine.context import EngineContext, Turn
from aria.engine.model.persona import PersonaProfile
from scripts import e1br_checkpoint as older
from scripts import run_e1br_received_continuation as received_parent
from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts.e1br_format_legacy import MANIFEST, MANIFEST_SHA, FREEZE, FREEZE_SHA, PARENT, RUN, sha
from scripts.e1br_format_runtime import FormatExperimentApp, terminal_format_failure

COMPLETED = (*older.COMPLETED, "e1b_b2")
CASE = "e1b_c1"
NEW_COUNTS = {"main": {"online_fast": 654, "memory": 15, "reflection": 15}}
require = older.require


def assert_originals(checkpoint, *, full=False):
    root = checkpoint["root"]
    members = sorted(p.relative_to(root).as_posix() for p in (root / PARENT).rglob("*") if p.is_file())
    require(members == sorted(checkpoint["membership"]), "format_checkpoint_membership_changed")
    for row in checkpoint["files"]:
        path = root / row["path"]
        require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "format_checkpoint_linked")
        require((path.stat().st_size, path.stat().st_mtime_ns) == checkpoint["fingerprint"][row["path"]],
                "format_checkpoint_original_changed")
        if full:
            require(sha(common.read_original(root, row["path"])) == row["sha256"], "format_checkpoint_bytes_changed")
    older.assert_originals(checkpoint["older"], full=full)


def inherited_files(root, checkpoint):
    archive = checkpoint["source_archive"]
    return list(dict.fromkeys([*older.inherited_files(root, checkpoint["older"]),
        Path(root) / MANIFEST, Path(root) / FREEZE,
        *(Path(root) / row["path"] for row in checkpoint["files"]),
        *(Path(root) / archive["archive"] / row["path"] for row in archive["archive_files"])]))


def validate_archived_source(root):
    allowed = {"path", "systemroot", "windir", "temp", "tmp", "home", "userprofile", "pathext"}
    env = {key: value for key, value in os.environ.items() if key.casefold() in allowed}
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("e1br_format_legacy.py")), str(root)],
        cwd=root, env=env, capture_output=True, text=True, timeout=120)
    require(result.returncode == 0, "format_checkpoint_archived_validation_failed")
    try:
        proof = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise original.RunError("format_checkpoint_archived_validation_invalid") from None
    require(proof.get("status") == "pass" and proof.get("manifest_sha256") == MANIFEST_SHA
            and proof.get("archive_files_verified") == 137 and proof.get("api_calls") == 0
            and proof.get("current_prompt_used_for_old_inputs") is False,
            "format_checkpoint_legacy_proof_changed")
    return proof


def load_checkpoint(root):
    root = Path(root).resolve()
    manifest_raw, freeze_raw = (common.read_original(root, item) for item in (MANIFEST, FREEZE))
    require(sha(manifest_raw) == MANIFEST_SHA and sha(freeze_raw) == FREEZE_SHA, "format_checkpoint_anchors_changed")
    manifest, freeze = json.loads(manifest_raw), json.loads(freeze_raw)
    previous = older.load_checkpoint(root)
    cp = {"root": root, "files": manifest["files"], "membership": [r["path"] for r in manifest["files"]],
        "fingerprint": {r["path"]: ((root / r["path"]).stat().st_size, (root / r["path"]).stat().st_mtime_ns)
                        for r in manifest["files"]}, "older": previous, "source_archive": freeze}
    assert_originals(cp, full=True)
    require(len(cp["files"]) == 368 and len(freeze["archive_files"]) == 137, "format_checkpoint_inventory_changed")
    for row in freeze["archive_files"]:
        require(sha(common.read_original(root, Path(freeze["archive"]) / row["path"])) == row["sha256"],
                "format_checkpoint_archived_source_changed")
    directory = root / PARENT
    registration = original.read_json(directory / "main/registration.json")
    receipt = original.read_json(directory / "main/execution-receipt.json")
    require(registration["seal_sha256"] == freeze["seal_sha256"] == receipt["registration_sha256"]
            and receipt["status"] == "not_evaluable", "format_checkpoint_old_gate_changed")
    rows = receipt["accounting"]["attempts"]
    ledger = directory / "allowance.sqlite3"
    require(not any(Path(str(ledger) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")), "format_checkpoint_open_ledger")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require([json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")] == rows
                and db.execute("SELECT halted FROM settings").fetchone() == ("model_or_storage_failed",),
                "format_checkpoint_closed_ledger_changed")
    require(len(rows) == 72 and Counter(r["status"] for r in rows) == {"received": 71, "failed": 1}
            and all(r["product_step"] == "initial" for r in rows)
            and receipt["accounting"]["unknown_cost_requests"] == 0
            and receipt["accounting"]["known_cost_usd"] == "0.041665508", "format_checkpoint_accounting_changed")
    failed = rows[-1]
    require(terminal_format_failure(failed) and failed["request_id"] == "1cbe1b56a87b412d8226cec3817fcc3f"
            and tuple(failed["experiment_position"]) == ("online_fast", CASE, 2, "shared", 0, 5),
            "format_checkpoint_failure_changed")
    legacy = validate_archived_source(root)
    prefix = received_parent.retained.load_prefix(root)
    first_reflection = received_parent.load_received(root)
    retained = [*deepcopy(prefix["retained_attempts"]), deepcopy(first_reflection["receipt"]),
                *deepcopy(previous["accepted"]), *deepcopy(rows)]
    require(len(retained) == 372 and len({r["request_id"] for r in retained}) == 372
            and Counter(r["role"] for r in retained) == {"online_fast": 354, "memory": 9, "reflection": 9},
            "format_checkpoint_retained_denominator_changed")
    case_root = directory / RUN / "main" / CASE
    run = original.read_json(case_root / "shared/live/run.json")
    profile = original.read_json(case_root / "shared/live/profiles/fictional-chen.json")
    job = profile["offline_sessions"]
    require(len(run["turns"]) == 18 and len(run["sessions"]) == 2
            and [r["status"] for r in run["turns"]] == ["completed"] * 17 + ["failed"]
            and len(job) == 1 and len(job[0]["turns"]) == 6 and job[0]["session_id"] == failed["session_id"]
            and job[0]["attempts"] == 0 and job[0]["state"] == "pending" and job[0]["source_completeness"] == "complete"
            and job[0]["excluded_turn_indices"] == [] and job[0]["memory_completed"] is False
            and job[0]["reflection_completed"] is False and profile["pending_session"] is None,
            "format_checkpoint_saved_context_changed")
    require(all(r["text"] == t["user"] for r, t in zip(run["turns"][12:], job[0]["turns"])),
            "format_checkpoint_context_inputs_changed")
    reports = {}
    for case in COMPLETED:
        path = directory / RUN / "main" / case / "report.json"
        value = original.read_json(path)
        require(value["case_id"] == case and value["scored_reply_count"] == 18 and len(value["branches"]) == 6,
                "format_checkpoint_completed_case_changed")
        reports[case] = sha(path.read_bytes())
    require({p.name for p in (directory / RUN / "main").iterdir() if p.is_dir()} == {*COMPLETED, CASE}
            and not (case_root / "report.json").exists(), "format_checkpoint_unexpected_case_started")
    older.require_empty_care(case_root / "shared/live/care.sqlite3")
    cp.update(registration=registration, receipt=receipt, rows=rows, accepted=rows[:-1], failed=failed,
        retained_attempts=retained, saved_run=run, saved_profile=profile, case_root=case_root, reports=reports,
        legacy_verification=legacy, prefix=prefix, received=first_reflection)
    cp["summary"] = {"schema": "e1br_format_checkpoint_v1", "campaign": PARENT, "manifest_sha256": MANIFEST_SHA,
        "old_gate": "not_evaluable", "total_retained": 372, "valid_received": 371, "known_format_failures": 1,
        "retained_counts_by_version": [13, 1, 286, 72], "completed_report_sha256": reports,
        "remaining_basic_positions": 684, "new_positions": NEW_COUNTS["main"], "first_new_position": ["online_fast", CASE, 2, "shared", 0, 6],
        "failed_request_id": failed["request_id"], "failed_body_replaced": False, "old_halt_cleared": False,
        "legacy_verification": legacy, "source_registration_sha256": registration["seal_sha256"],
        "source_inventory_sha256": original.digest(registration["source"]), "effect_claim_eligible": False}
    return cp


def restore_shared(app, checkpoint):
    """Restore a durable text checkpoint, without re-executing any old input."""
    assert_originals(checkpoint, full=True)
    require(isinstance(app, FormatExperimentApp) and not app._run["turns"] and app._profile() is None,
            "format_checkpoint_requires_fresh_app")
    before = app.call_guard.snapshot() if app.call_guard else None
    persona = PersonaProfile.model_validate(deepcopy(checkpoint["saved_profile"]))
    job = persona.offline_sessions.pop()
    # The old sealed job is untouched. In this new experimental continuation
    # its exact text becomes the active context; no source timestamps change.
    engine = app._engine_for_persona(persona)
    ctx = EngineContext(user_id="fictional-chen", scene_id=app.scene_id, session_id=job.session_id,
                        turns=[Turn(user=t.user, agent=t.agent) for t in job.turns])
    ctx.alliance.trust = persona.alliance.trust
    ctx.external_emotion_experiment = app.experiment
    ctx.working_memory.update(_offline_source_completeness="complete", _sensitive_memory_excluded_turns=[],
                              _last_checkpoint_turn=6, _restored_checkpoint=True, _alliance_seeded=True)
    engine._seed_persistent_tasks(ctx)
    app._sessions["fictional-chen"] = engine, ctx
    app._session_activity["fictional-chen"] = time.monotonic()
    run_id, allowance = app._run["run_id"], deepcopy(app._run.get("model_allowance"))
    app._run = deepcopy(checkpoint["saved_run"])
    app._run["run_id"] = run_id
    if allowance is not None:
        app._run["model_allowance"] = allowance
    app._run["sessions"][-1]["status"] = "active"
    app._run["format_continuation"] = {"source_campaign": PARENT, "source_run_id": checkpoint["saved_run"]["run_id"],
        "manifest_sha256": MANIFEST_SHA, "new_model_calls_during_restore": 0,
        "old_turn_records_changed": False, "old_inputs_replayed": False,
        "restoration_semantics": "durable_text_checkpoint_with_alliance_and_persistent_tasks"}
    app.store.save(persona)
    engine.checkpoint_session(ctx)
    app._save_run()
    app.retain_terminal(checkpoint["failed"], historical=True)
    require(app._run["turns"] == checkpoint["saved_run"]["turns"], "format_checkpoint_turns_changed")
    require(app.call_guard is None or app.call_guard.snapshot() == before, "format_checkpoint_restore_wrote_allowance")
    older.require_empty_care(app.run_dir / "care.sqlite3")
    assert_originals(checkpoint, full=True)
    return {"ok": True, "retained_turns": 18, "active_session_turns": 6, "new_model_calls": 0}


def runner_class(checkpoint):
    class FormatCheckpointRunner(E1BRRunner):
        def run_case(self, case_id, *, phase="main", repeats=3):
            require(phase == "main" and repeats == 3, "format_checkpoint_registered_main_only")
            assert_originals(checkpoint)
            if case_id in COMPLETED:
                target = self.root / phase / case_id
                require(not target.exists(), "format_checkpoint_destination_exists")
                source = checkpoint["root"] / PARENT / RUN / "main" / case_id
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        older.exclusive(target / path.relative_to(source), common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))
                require(sha((target / "report.json").read_bytes()) == checkpoint["reports"][case_id], "format_checkpoint_retained_report_changed")
                self._journal({"event": "retained_complete_case", "case_id": case_id, "new_model_calls": 0,
                               "report_sha256": checkpoint["reports"][case_id], "source_campaign": PARENT})
                return original.read_json(target / "report.json")
            return super().run_case(case_id, phase=phase, repeats=repeats)

        def _session(self, app, case, number, *, arm, repeat):
            if case.id != CASE or arm != "shared":
                return super()._session(app, case, number, arm=arm, repeat=repeat)
            metadata = {"case_id": CASE, "phase": f"s{number}", "arm": arm, "repeat": repeat}
            if number == 1:
                target = self.root / "main" / CASE
                for path in sorted(checkpoint["case_root"].iterdir()):
                    if path.is_file():
                        older.exclusive(target / path.name, common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))
                self._operation(app, "restore_retained_shared", lambda: restore_shared(app, checkpoint), detail=metadata)
                return checkpoint["saved_run"]["sessions"][0]["session_id"]
            session_id = checkpoint["failed"]["session_id"]
            require(len(app._sessions["fictional-chen"][1].turns) == 6, "format_checkpoint_next_turn_changed")
            for turn in range(6, 12):
                self._operation(app, "submit", lambda turn=turn: app.submit(case.current_input(2, turn),
                    f"e1b-{session_id}-t{turn}", session_id, external_emotion=deepcopy(self.plan[(CASE, 2, turn)])),
                    detail={**metadata, "turn": turn})
            self._operation(app, "seal", lambda: app.seal(session_id), detail=metadata)
            return session_id

        def _operation(self, app, name, operation, *, detail=None):
            if name == "reflect" and detail == {"case_id": CASE, "session": 1}:
                self._journal({"event": "retained_complete_review", "case_id": CASE, "session": 1, "new_model_calls": 0})
                return {"ok": True, "retained_original_review": True}
            return super()._operation(app, name, operation, detail=detail)
    return FormatCheckpointRunner
