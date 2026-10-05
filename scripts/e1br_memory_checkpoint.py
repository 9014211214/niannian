"""Restore sealed v4 results without replaying any of its 731 logical positions."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from aria.bench.e1br_runner import E1BRRunner
from aria.engine.memory.diagnosis import build_reflection_messages, _sensitive_memory_excluded_turns
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.e1br_calls import logical_attempts
from scripts import e1br_format_checkpoint as previous
from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts.e1br_memory_runtime import MemoryExperimentApp, digest, require, sha, terminal_memory_evidence

MANIFEST = "artifacts/local/tasks/20261005-02/e1br-close-review/v4-closed-originals-v1.manifest.json"
MANIFEST_SHA = "bf04cb2831d776618abfda2525b010b36e553128d0a2099681aaed0d1009f293"
FREEZE = "artifacts/local/tasks/20261005-02/e1br-v4-root-formal-freeze-v1.json"
FREEZE_SHA = "215585f2c3d9e39b570da14f9e1f62e3fc24c8d56c1e7d575b41475dbbb95914"
PARENT = "artifacts/local/e1br-formal-v4"
RUN = "main/synthetic-emotion-experiments/e1b-main-v1/run"
CASE = "e1b_e1"
COMPLETED = tuple(f"e1b_{letter}{number}" for letter in "abcd" for number in (1, 2))
NEW_COUNTS = {"main": {"online_fast": 312, "memory": 6, "reflection": 7}}


def position_inventory(cases):
    positions = set()
    require(len(cases) == len(set(cases)) == 12, "memory_twelve_cases_required")
    for case in cases:
        for session in (1, 2):
            positions.update(("online_fast", case, session, "shared", 0, turn) for turn in range(12))
            positions.update((role, case, session, "shared", 0, None) for role in ("memory", "reflection"))
        positions.update(("online_fast", case, 3, arm, repeat, turn)
                         for arm in ("A0", "A1") for repeat in (1, 2, 3) for turn in range(10))
    require(len(positions) == 1056, "memory_fixed_denominator_changed")
    return frozenset(positions)


def assert_originals(checkpoint, *, full=False):
    root = checkpoint["root"]
    members = sorted(p.relative_to(root).as_posix() for p in (root / PARENT).rglob("*") if p.is_file())
    require(members == sorted(checkpoint["membership"]), "memory_original_membership_changed")
    for item in checkpoint["files"]:
        path = root / item["path"]
        require((path.stat().st_size, path.stat().st_mtime_ns) == checkpoint["fingerprint"][item["path"]],
                "memory_original_stat_changed")
        if full:
            require(sha(common.read_original(root, item["path"])) == item["sha256"], "memory_original_bytes_changed")
    previous.assert_originals(checkpoint["older"], full=full)


def inherited_files(root, checkpoint):
    archive = checkpoint["source_archive"]
    return list(dict.fromkeys([*previous.inherited_files(root, checkpoint["older"]),
        Path(root) / MANIFEST, Path(root) / FREEZE,
        *(Path(root) / row["path"] for row in checkpoint["files"]),
        *(Path(root) / archive["archive"] / row["path"] for row in archive["archive_files"])]))


def validate_archived_source(root):
    allowed = {"path", "systemroot", "windir", "temp", "tmp", "home", "userprofile", "pathext"}
    env = {key: value for key, value in os.environ.items() if key.casefold() in allowed}
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("e1br_memory_legacy.py")), str(root)],
        cwd=root, env=env, capture_output=True, text=True, timeout=120)
    require(result.returncode == 0, "memory_archived_validation_failed")
    value = json.loads(result.stdout)
    require(value.get("status") == "pass" and value.get("manifest_sha256") == MANIFEST_SHA
            and value.get("archive_files_verified") == 145 and value.get("api_calls") == 0,
            "memory_archived_validation_invalid")
    return value


def load_checkpoint(root):
    root = Path(root).resolve()
    manifest_raw, freeze_raw = (common.read_original(root, name) for name in (MANIFEST, FREEZE))
    require(sha(manifest_raw) == MANIFEST_SHA and sha(freeze_raw) == FREEZE_SHA, "memory_checkpoint_anchors_changed")
    manifest, freeze = json.loads(manifest_raw), json.loads(freeze_raw)
    older = previous.load_checkpoint(root)
    cp = {"root": root, "files": manifest["files"], "membership": [r["path"] for r in manifest["files"]],
          "fingerprint": {r["path"]: ((root / r["path"]).stat().st_size, (root / r["path"]).stat().st_mtime_ns)
                          for r in manifest["files"]}, "older": older, "source_archive": freeze}
    require(len(cp["files"]) == 1149 and len(freeze["archive_files"]) == 145, "memory_checkpoint_inventory_changed")
    assert_originals(cp, full=True)
    for row in freeze["archive_files"]:
        require(sha(common.read_original(root, Path(freeze["archive"]) / row["path"])) == row["sha256"],
                "memory_archived_source_changed")
    directory = root / PARENT
    registration = json.loads(common.read_original(root, Path(PARENT) / "main/registration.json"))
    receipt = json.loads(common.read_original(root, Path(PARENT) / "main/execution-receipt.json"))
    require(registration["seal_sha256"] == freeze["seal_sha256"] == receipt["registration_sha256"]
            and digest({k: v for k, v in registration.items() if k != "seal_sha256"}) == registration["seal_sha256"]
            and digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) == receipt["receipt_sha256"]
            and receipt["status"] == "not_evaluable" and receipt["effect_claim_eligible"] is False,
            "memory_parent_registration_or_gate_changed")
    rows = receipt["accounting"]["attempts"]
    ledger = directory / "allowance.sqlite3"
    require(not any(Path(str(ledger) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")), "memory_source_ledger_not_closed")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require([json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")] == rows
                and db.execute("SELECT halted FROM settings WHERE id=1").fetchone() == ("model_or_storage_failed",),
                "memory_closed_ledger_changed")
    require(len(rows) == 361 and Counter(r["status"] for r in rows) == {"received": 358, "failed": 3}
            and receipt["accounting"]["known_cost_usd"] == "0.363870888"
            and receipt["accounting"]["unknown_cost_requests"] == 1, "memory_closed_accounting_changed")
    initial = [row for row in logical_attempts(receipt["accounting"]) if row["product_step"] == "initial"]
    retained = [*deepcopy(older["retained_attempts"]), *deepcopy(initial)]
    bindings = {row["request_id"]: {"receipt_sha256": digest(row),
        "position": [row["role"], "e1b_a1", 1, "shared", 0, i if row["role"] == "online_fast" else None]}
        for i, row in enumerate(older["prefix"]["retained_attempts"])}
    positions = [tuple(row.get("experiment_position", bindings.get(row["request_id"], {}).get("position", ()))) for row in retained]
    full = position_inventory(registration["inputs"]["case_ids"])
    require(len(retained) == len(set(positions)) == len({r["request_id"] for r in retained}) == 731
            and set(positions) <= full, "memory_retained_positions_changed")
    remaining = full - set(positions)
    require(len(remaining) == 325 and Counter(p[0] for p in remaining) == Counter(NEW_COUNTS["main"]), "memory_remaining_positions_changed")
    failed = rows[-1]
    require(tuple(failed["experiment_position"]) == ("memory", CASE, 2, "shared", 0, None), "memory_old_stop_position_changed")
    case_root = directory / RUN / "main" / CASE
    def read_case(relative):
        return json.loads(common.read_original(root, (case_root / relative).relative_to(root)))
    profile, run = read_case("shared/live/profiles/fictional-chen.json"), read_case("shared/live/run.json")
    visible_raw = common.read_original(root, Path(PARENT) / failed["visible_original_path"])
    require(sha(visible_raw) == failed["visible_original_sha256"], "memory_old_visible_file_changed")
    terminal = terminal_memory_evidence(failed, json.loads(visible_raw), read_case("memory-s2.first.json"),
        read_case("memory-s2.product.json"), profile, run, failed["experiment_position"])
    reports = {}
    for case in COMPLETED:
        raw = common.read_original(root, Path(PARENT) / RUN / "main" / case / "report.json")
        value = json.loads(raw)
        require(value["case_id"] == case and value["scored_reply_count"] == 18 and len(value["branches"]) == 6,
                "memory_completed_report_changed")
        reports[case] = sha(raw)
    require(not (case_root / "report.json").exists()
            and {p.name for p in (directory / RUN / "main").iterdir() if p.is_dir()} == {*COMPLETED, CASE},
            "memory_unexpected_case_started")
    previous.older.require_empty_care(case_root / "shared/live/care.sqlite3")
    physical = [*deepcopy(older["prefix"]["retained_attempts"]), deepcopy(older["received"]["receipt"]),
                *deepcopy(older["older"]["rows"]), *deepcopy(older["rows"]), *deepcopy(rows)]
    require(len(physical) == len({r["request_id"] for r in physical}) == 734, "memory_retained_physical_changed")
    known = sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r.get("usage", {})), Decimal(0))
    hold = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r.get("usage", {})), Decimal(0))
    require(known == Decimal("0.363870888") and hold == Decimal("0.001255602"), "memory_parent_amount_changed")
    legacy = validate_archived_source(root)
    cp.update(registration=registration, receipt=receipt, rows=rows, failed=failed, saved_profile=profile, saved_run=run,
        case_root=case_root, reports=reports, terminal=terminal, retained_attempts=retained,
        retained_physical_attempts=physical, retained_position_bindings=bindings, remaining_positions=frozenset(remaining),
        visible_raw=visible_raw, legacy_verification=legacy)
    cp["summary"] = {"schema": "e1br_memory_checkpoint_v1", "campaign": PARENT, "manifest_sha256": MANIFEST_SHA,
        "total_retained": 731, "retained_physical_attempts": 734, "remaining_basic_positions": 325,
        "new_positions": NEW_COUNTS["main"], "completed_report_sha256": reports,
        "first_new_position": ["reflection", CASE, 2, "shared", 0, None],
        "failed_request_id": failed["request_id"], "failed_body_replaced": False, "old_halt_cleared": False,
        "old_gate": "not_evaluable", "effect_claim_eligible": False,
        "source_registration_sha256": registration["seal_sha256"], "source_inventory_sha256": digest(registration["source"]),
        "legacy_verification": legacy, "e1b_known_usd": "0.880939320", "cumulative_known_usd": "2.273318492",
        "historical_unknown_requests": 10, "historical_unknown_reserved_usd": "0.324948616"}
    cp["summary"]["memory_source_baseline"] = {key: terminal[key] for key in
        ("original_profile_sha256", "original_job_sha256", "original_run_sha256", "old_failed_review_sha256")}
    cp["summary"]["memory_source_baseline"].update(run_id=run["run_id"], model_allowance=deepcopy(run.get("model_allowance")),
                                               source_manifest_sha256=MANIFEST_SHA)
    return cp


def validate_first_request(checkpoint, payload, scope, key):
    persona = PersonaProfile.model_validate(deepcopy(checkpoint["saved_profile"]))
    job = persona.offline_sessions[0]
    ctx = OfflineSessionQueue._context(persona, job)
    excluded = _sensitive_memory_excluded_turns(ctx)
    conversation = "\n".join(f"[turn {i}] 用户:{t.user}\nAria:{t.agent}" for i, t in enumerate(ctx.turns) if i not in excluded)
    expected = build_reflection_messages(persona, conversation, [], True)[1]
    require(tuple(key) == ("reflection", CASE, 2, "shared", 0, None)
            and scope["session_id"] == job.session_id and scope["operation_id"] == f"e1b-review-{job.session_id}"
            and payload.get("response_format") is None and len(payload.get("messages", [])) == 2
            and payload["messages"][1] == expected, "memory_first_reflection_input_changed")
    return {"continued_after_terminal_request_id": checkpoint["failed"]["request_id"],
            "parent_manifest_sha256": MANIFEST_SHA, "old_request_replayed": False, "old_failure_replaced": False}


def restore_shared(app, checkpoint):
    assert_originals(checkpoint, full=True)
    require(isinstance(app, MemoryExperimentApp) and not app._run["turns"] and app._profile() is None,
            "memory_restore_requires_fresh_app")
    before = deepcopy(app.call_guard.snapshot()) if app.call_guard else None
    fresh_id, allowance = app._run["run_id"], deepcopy(app._run.get("model_allowance"))
    app._run = deepcopy(checkpoint["saved_run"])
    app._run["run_id"] = fresh_id
    if allowance is not None:
        app._run["model_allowance"] = allowance
    app.store.save(PersonaProfile.model_validate(deepcopy(checkpoint["saved_profile"])))
    app._save_run()
    require(app.call_guard is None or before == app.call_guard.snapshot(), "memory_restore_changed_new_ledger")
    require(app._run["turns"] == checkpoint["saved_run"]["turns"], "memory_restore_replayed_turns")
    return {"ok": True, "retained_turns": 24, "new_model_calls": 0}


def runner_class(checkpoint):
    class MemoryCheckpointRunner(E1BRRunner):
        def run_case(self, case_id, *, phase="main", repeats=3):
            require(phase == "main" and repeats == 3, "memory_registered_main_only")
            assert_originals(checkpoint)
            if case_id in COMPLETED:
                target = self.root / phase / case_id
                require(not target.exists(), "memory_completed_destination_exists")
                source = checkpoint["root"] / PARENT / RUN / "main" / case_id
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        previous.older.exclusive(target / path.relative_to(source), common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))
                require(sha((target / "report.json").read_bytes()) == checkpoint["reports"][case_id], "memory_old_report_changed")
                self._journal({"event": "retained_complete_case", "case_id": case_id, "new_model_calls": 0,
                    "report_sha256": checkpoint["reports"][case_id], "source_campaign": PARENT})
                return original.read_json(target / "report.json")
            return super().run_case(case_id, phase=phase, repeats=repeats)

        def _session(self, app, case, number, *, arm, repeat):
            if case.id != CASE or arm != "shared":
                return super()._session(app, case, number, arm=arm, repeat=repeat)
            if number == 1:
                target = self.root / "main" / CASE
                for path in sorted(checkpoint["case_root"].iterdir()):
                    if path.is_file():
                        previous.older.exclusive(target / path.name, common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))
                self._operation(app, "restore_retained_shared", lambda: restore_shared(app, checkpoint),
                                detail={"case_id": CASE, "phase": "s1", "arm": "shared", "repeat": 0})
            require(app._run["turns"] == checkpoint["saved_run"]["turns"]
                    and app._run["sessions"] == checkpoint["saved_run"]["sessions"], "memory_old_sessions_changed")
            return checkpoint["saved_run"]["sessions"][number - 1]["session_id"]

        def _operation(self, app, name, operation, *, detail=None):
            if name == "reflect" and detail == {"case_id": CASE, "session": 1}:
                self._journal({"event": "retained_complete_review", "case_id": CASE, "session": 1, "new_model_calls": 0})
                return {"ok": True, "retained_original_review": True}
            if name == "reflect" and detail == {"case_id": CASE, "session": 2}:
                def operation():
                    app.retain_memory_terminal(checkpoint["failed"], checkpoint["visible_raw"], historical=True, source_campaign=PARENT,
                                               source_baseline=checkpoint["summary"]["memory_source_baseline"])
                    return app.resume_missing_reflection(checkpoint["failed"]["operation_id"])
            return super()._operation(app, name, operation, detail=detail)
    return MemoryCheckpointRunner
