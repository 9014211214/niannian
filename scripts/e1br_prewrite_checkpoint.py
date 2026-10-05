"""Preserve v5 first results and resume its unreceived bounded transport position.

The archived source validates old evidence; current source is separately frozen.
This checkpoint grants no credential access or execution authorization.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import stat

from scripts import e1br_memory_checkpoint as parent

PARENT = Path("artifacts/local/e1br-formal-v5")
RUN = Path("main/synthetic-emotion-experiments/e1b-main-v1/run")
MANIFEST = Path("artifacts/local/tasks/20261005-03/v5-close/v5-closed-originals-v1.manifest.json")
MANIFEST_SHA = "90c27f26a314f32975fb1baf770bf32b943466fe026925a4d47ff50bebdf6539"
FREEZE = Path("artifacts/local/tasks/20261005-03/e1br-v5-root-formal-freeze-v1.json")
FREEZE_SHA = "ceae7f411315596eb3aec4358ec919171e0af31f7a0ebdcb709e455c26f9ecc1"
CASE = "e1b_e2"
COMPLETED = (*parent.COMPLETED, "e1b_e1")
FIRST = ("online_fast", CASE, 3, "A0", 1, 4)
NEW_COUNTS = {"online_fast": 214, "memory": 4, "reflection": 4}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())


def require(value, code):
    if not value:
        raise ValueError(code)


def read(root, name):
    path = Path(root) / name
    require(path.resolve().is_relative_to(Path(root).resolve()), "checkpoint_path_outside_root")
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                    for p in (path, *path.parents)), "checkpoint_link_rejected")
    return path.read_bytes()


def require_unexpired(job):
    created, expires = (datetime.fromisoformat(job[k]) for k in ("created_at", "expires_at"))
    require(created.tzinfo is not None and expires.tzinfo is not None and created < expires,
            "original_branch_retention_invalid")
    require(datetime.now(timezone.utc) < expires, "original_branch_retention_expired")


def _closed_members(root):
    """Enumerate exact closed membership without following links or reparses."""
    directory = root / PARENT
    def check(path):
        value = path.lstat()
        require(not stat.S_ISLNK(value.st_mode)
                and not getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                "closed_v5_link_rejected")
        return value
    for path in (directory, *directory.parents):
        check(path)
    pending, observed = [directory], []
    while pending:
        current = pending.pop()
        for path in current.iterdir():
            value = check(path)
            is_directory = stat.S_ISDIR(value.st_mode)
            require(is_directory or stat.S_ISREG(value.st_mode), "closed_v5_nonregular_member")
            observed.append([path.relative_to(root).as_posix(), is_directory])
            if is_directory:
                pending.append(path)
    return sorted(observed)


def assert_originals(cp, *, full=False):
    root = cp["root"]
    require(len({row[0] for row in cp["members"]}) == len(cp["members"])
            and _closed_members(root) == sorted(cp["members"]), "closed_v5_membership_changed")
    for row in cp["files"]:
        path = root / row["path"]
        require((path.stat().st_size, path.stat().st_mtime_ns) == cp["fingerprints"][row["path"]],
                "closed_v5_original_changed")
        if full:
            require(sha(read(root, row["path"])) == row["sha256"], "closed_v5_original_bytes_changed")
    for row in cp["freeze"]["archive_files"]:
        require(sha(read(root, Path(cp["freeze"]["archive"]) / row["path"])) == row["sha256"],
                "formal150_source_changed")
    parent.assert_originals(cp["older"], full=full)


def load_checkpoint(root):
    root = Path(root).resolve()
    raw_manifest, raw_freeze = read(root, MANIFEST), read(root, FREEZE)
    require(sha(raw_manifest) == MANIFEST_SHA and sha(raw_freeze) == FREEZE_SHA, "v5_anchor_changed")
    manifest, freeze = json.loads(raw_manifest), json.loads(raw_freeze)
    cp = {"root": root, "files": manifest["files"], "members": manifest["members"], "freeze": freeze,
          "older": parent.load_checkpoint(root)}
    cp["fingerprints"] = {r["path"]: ((root / r["path"]).stat().st_size, (root / r["path"]).stat().st_mtime_ns)
                          for r in cp["files"]}
    require(len(cp["files"]) == 702 and len(freeze["archive_files"]) == 150, "v5_file_inventory_changed")
    assert_originals(cp, full=True)
    registration = json.loads(read(root, PARENT / "main/registration.json"))
    receipt = json.loads(read(root, PARENT / "main/execution-receipt.json"))
    require(receipt["registration_sha256"] == registration["seal_sha256"] == freeze["seal_sha256"], "v5_seal_changed")
    rows = receipt["accounting"]["attempts"]
    ledger = (root / PARENT / "allowance.sqlite3").resolve()
    require(not any(Path(str(ledger) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")), "v5_ledger_open")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require([json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")] == rows,
                "v5_ledger_receipt_mismatch")
    require(len(rows) == 104 and Counter(r["status"] for r in rows) == {"received": 103, "failed": 1}
            and all(r["product_step"] == "initial" for r in rows), "v5_closed_attempt_inventory_changed")
    require(sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r.get("usage", {})), Decimal(0)) == Decimal("0.111635122")
            and sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r.get("usage", {})), Decimal(0)) == Decimal("0.00076487"),
            "v5_known_cost_or_unknown_hold_changed")
    received, failed = rows[:-1], rows[-1]
    require(tuple(failed["experiment_position"]) == FIRST and not failed.get("visible_content_sha256")
            and not failed.get("usage") and failed["infrastructure_retry_index"] == 0, "v5_stop_changed")
    require(len(failed["transport_events"]["connection_preparations"]) == 2, "old_rebuild_consumption_changed")
    retained = [*deepcopy(cp["older"]["retained_attempts"]), *deepcopy(received)]
    bindings = cp["older"]["retained_position_bindings"]
    positions = [tuple(r.get("experiment_position", bindings.get(r["request_id"], {}).get("position", ()))) for r in retained]
    full = parent.position_inventory(registration["inputs"]["case_ids"])
    remaining = full - set(positions)
    require(len(retained) == len(set(positions)) == 834 and len(remaining) == 222
            and Counter(p[0] for p in remaining) == NEW_COUNTS and FIRST in remaining,
            "v5_logical_position_inventory_changed")
    physical = [*deepcopy(cp["older"]["retained_physical_attempts"]), *deepcopy(rows)]
    require(len(physical) == len({r["request_id"] for r in physical}) == 838, "v5_physical_inventory_changed")
    case_root = root / PARENT / RUN / "main" / CASE
    run = json.loads(read(root, (case_root / "repeat-1-A0/live/run.json").relative_to(root)))
    profile = json.loads(read(root, (case_root / "repeat-1-A0/live/profiles/fictional-chen.json").relative_to(root)))
    require(len(run["turns"]) == 5 and [r["status"] for r in run["turns"]] == ["completed"] * 4 + ["failed"]
            and len(profile["offline_sessions"]) == 1 and profile["pending_session"] is None,
            "v5_partial_branch_shape_changed")
    job = profile["offline_sessions"][0]
    require(job["session_id"] == failed["session_id"] and len(job["turns"]) == 5
            and job["source_completeness"] == "complete" and not job["memory_completed"]
            and not job["reflection_completed"] and job["attempts"] == 0, "v5_partial_branch_job_changed")
    require_unexpired(job)
    wire = read(root, PARENT / failed["wire_request_path"])
    require(sha(wire) == failed["request_sha256"], "v5_failed_wire_changed")
    payload = json.loads(wire)
    messages = payload["messages"]
    require(payload.get("response_format") is None and len(messages) == 10 and messages[0]["role"] == "system",
            "v5_first_plain_text_contract_changed")
    for i in range(4):
        require(messages[1 + 2 * i] == {"role": "user", "content": job["turns"][i]["user"]}
                and messages[2 + 2 * i] == {"role": "assistant", "content": job["turns"][i]["agent"]}
                and run["turns"][i]["response"] == job["turns"][i]["agent"], "v5_success_prefix_does_not_match_wire")
    require(messages[-1] == {"role": "user", "content": job["turns"][4]["user"]}
            and messages[-1]["content"] == run["turns"][4]["text"], "v5_failed_input_changed")
    reports = {}
    for case in COMPLETED:
        path = root / PARENT / RUN / "main" / case / "report.json"
        raw = read(root, path.relative_to(root)); value = json.loads(raw)
        require(value["case_id"] == case and len(value["branches"]) == 6 and value["scored_reply_count"] == 18,
                "v5_completed_report_changed")
        reports[case] = sha(raw)
    require(not (case_root / "report.json").exists(), "v5_partial_case_has_report")
    shared = json.loads(read(root, (case_root / "shared.json").relative_to(root)))
    completed_branch = json.loads(read(root, (case_root / "repeat-1-A1.json").relative_to(root)))
    require(len(shared["state"]["turns"]) == 24 and len(completed_branch["state"]["turns"]) == 10,
            "v5_completed_segment_changed")
    cp.update(registration=registration, receipt=receipt, rows=rows, failed=failed, retained_attempts=retained,
              retained_physical_attempts=physical, retained_position_bindings=bindings,
              remaining_positions=remaining, saved_run=run, saved_profile=profile, saved_job=job,
              wire=wire, payload=payload, case_root=case_root, reports=reports, shared=shared,
              completed_branch=completed_branch)
    cp["summary"] = summary(cp)
    cp["summary"]["memory_source_baseline"] = deepcopy(cp["older"]["summary"]["memory_source_baseline"])
    cp["summary"].update(total_retained=834, retained_physical_attempts=838, remaining_basic_positions=222,
        new_positions=deepcopy(NEW_COUNTS), completed_report_sha256=deepcopy(reports),
        first_new_position=list(FIRST), historical_unknown_reserved_usd="0.325713486",
        source_registration_sha256=registration["seal_sha256"],
        source_inventory_sha256=digest(registration["source"]))
    return cp


def validate_first_request(cp, payload, scope, key):
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    position, operation_id, session_id = key, scope["operation_id"], scope["session_id"]
    require(tuple(position) == FIRST and operation_id == cp["failed"]["operation_id"]
            and session_id == cp["failed"]["session_id"], "retained_first_scope_changed")
    require(request_data == cp["wire"], "retained_first_wire_not_byte_identical")
    require_unexpired(cp["saved_job"])
    return {"wire_sha256": sha(request_data), "old_scope_preserved": True,
            "old_failed_application_in_context": False, "new_model_calls": 0}


def summary(cp):
    return {"schema": "v5_transport_checkpoint_candidate_v1", "status": "read_only_checkpoint_verified",
        "real_execution_enabled": False, "old_logical_first_results": 834, "old_physical_attempts": 838,
        "completed_reports": cp["reports"], "remaining": 222, "remaining_by_role": NEW_COUNTS,
        "first_position": list(FIRST), "first_request_sha256": cp["failed"]["request_sha256"],
        "first_operation_id": cp["failed"]["operation_id"], "first_session_id": cp["failed"]["session_id"],
        "first_wire_saved_byte_identical": True, "wire_conversation_equals_four_successes_plus_current_input": True,
        "current_profile_has_failed_fallback": True, "fallback_must_not_enter_recovered_context": True,
        "original_created_at": cp["saved_job"]["created_at"], "original_expires_at": cp["saved_job"]["expires_at"],
        "old_initial_physical_attempts": 1, "remaining_target_retries": 2, "connection_rebuild_already_used": True,
        "historical_unknown_requests": 11, "historical_hold_usd": "0.325713486",
        "known_e1b_usd": "0.992574442", "known_cumulative_usd": "2.384953614", "allowance_usd": "0.681712072",
        "source_manifest_sha256": MANIFEST_SHA, "freeze_sha256": FREEZE_SHA,
        "formal150_archive_verified": True, "real_model_calls": 0, "key_reads": 0}


def inherited_files(root, checkpoint):
    root = Path(root)
    return list(dict.fromkeys([*parent.inherited_files(root, checkpoint["older"]),
        root / MANIFEST, root / FREEZE, *(root / row["path"] for row in checkpoint["files"]),
        *(root / checkpoint["freeze"]["archive"] / row["path"] for row in checkpoint["freeze"]["archive_files"])]))


def restore_partial_branch(app, checkpoint):
    """New active projection of four received turns; original failure stays sealed."""
    import time
    from aria.engine.context import EngineContext, Turn
    from aria.engine.model.persona import PersonaProfile
    assert_originals(checkpoint, full=True)
    require(not app._run["turns"] and app._profile() is None, "prewrite_restore_requires_fresh_app")
    require_unexpired(checkpoint["saved_job"])
    before_guard = app.call_guard.snapshot() if app.call_guard else None
    profile = PersonaProfile.model_validate(deepcopy(checkpoint["saved_profile"]))
    job = profile.offline_sessions.pop()
    require(job.model_dump(mode="json") == checkpoint["saved_job"], "prewrite_job_projection_changed")
    engine = app._engine_for_persona(profile)
    context = EngineContext(user_id="fictional-chen", scene_id=app.scene_id, session_id=job.session_id,
        turns=[Turn(user=turn.user, agent=turn.agent) for turn in job.turns[:4]])
    context.alliance.trust = profile.alliance.trust
    context.external_emotion_experiment = app.experiment
    context.working_memory.update(_offline_source_completeness="complete", _sensitive_memory_excluded_turns=[],
        _last_checkpoint_turn=4, _restored_checkpoint=True, _alliance_seeded=True)
    engine._seed_persistent_tasks(context)
    app._sessions["fictional-chen"] = engine, context
    app._session_activity["fictional-chen"] = time.monotonic()
    fresh_id, allowance = app._run["run_id"], deepcopy(app._run.get("model_allowance"))
    app._run = deepcopy(checkpoint["saved_run"])
    app._run["run_id"] = fresh_id
    if allowance is not None:
        app._run["model_allowance"] = allowance
    # The original failed turn remains in its immutable source and this explicit
    # side record. It is not a received logical result or an input history turn.
    app._run["transport_continuation"] = {
        "schema": "e1br_prewrite_active_projection_v1", "source_campaign": PARENT.as_posix(),
        "source_manifest_sha256": MANIFEST_SHA, "source_run_id": checkpoint["saved_run"]["run_id"],
        "source_run_sha256": sha(read(checkpoint["root"], (checkpoint["case_root"] / "repeat-1-A0/live/run.json").relative_to(checkpoint["root"]))),
        "retained_failed_turn": deepcopy(checkpoint["saved_run"]["turns"][4]),
        "original_job": deepcopy(checkpoint["saved_job"]), "old_request_id": checkpoint["failed"]["request_id"],
        "old_failed_wire_sha256": checkpoint["failed"]["request_sha256"], "old_failure_replaced": False,
        "active_prefix_received_turns": 4, "new_calls_during_restore": 0,
    }
    app._run["turns"] = deepcopy(checkpoint["saved_run"]["turns"][:4])
    app._run["sessions"][-1]["status"] = "active"
    app.store.save(profile)
    app._save_run()
    require(app.call_guard is None or before_guard == app.call_guard.snapshot(), "prewrite_restore_wrote_ledger")
    require(app._run["turns"] == checkpoint["saved_run"]["turns"][:4], "prewrite_received_prefix_changed")
    require(all(getattr(profile, field) == getattr(PersonaProfile.model_validate(checkpoint["saved_profile"]), field)
                for field in ("episodic_memory", "stated_preferences", "person_hypotheses", "relationship_context", "last_interaction_at")),
            "prewrite_restore_changed_profile_knowledge")
    return {"ok": True, "retained_received_turns": 4, "retained_failed_attempts": 1,
            "old_fallback_in_active_context": False, "model_calls": 0}


def _copy_tree(checkpoint, source, target):
    target = Path(target)
    require(not target.exists(), "prewrite_restore_target_exists")
    for path in sorted(Path(source).rglob("*")):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(read(checkpoint["root"], path.relative_to(checkpoint["root"])))


def runner_class(checkpoint):
    from aria.bench.e1b_runner import _write_json
    from aria.bench.e1br_runner import E1BRRunner
    from aria.engine.model.persona import PersonaProfile

    class PrewriteCheckpointRunner(E1BRRunner):
        def run_case(self, case_id, *, phase="main", repeats=3):
            require(phase == "main" and repeats == 3, "prewrite_registered_main_only")
            assert_originals(checkpoint)
            target = self.root / phase / case_id
            if case_id in COMPLETED:
                source = checkpoint["root"] / PARENT / RUN / "main" / case_id
                _copy_tree(checkpoint, source, target)
                require(sha((target / "report.json").read_bytes()) == checkpoint["reports"][case_id], "retained_report_changed")
                self._journal({"event": "retained_complete_case", "case_id": case_id, "new_model_calls": 0,
                               "report_sha256": checkpoint["reports"][case_id], "source_campaign": PARENT.as_posix()})
                return json.loads((target / "report.json").read_bytes())
            if case_id != CASE:
                return super().run_case(case_id, phase=phase, repeats=repeats)
            require(not target.exists(), "prewrite_partial_target_exists")
            target.mkdir(parents=True)
            for path in sorted(checkpoint["case_root"].iterdir()):
                if path.is_file():
                    with (target / path.name).open("xb") as handle:
                        handle.write(read(checkpoint["root"], path.relative_to(checkpoint["root"])))
            _copy_tree(checkpoint, checkpoint["case_root"] / "shared", target / "shared")
            case = next(case for case in self.suite.items if case.id == CASE)
            shared = deepcopy(checkpoint["shared"])
            shared_profile = PersonaProfile.model_validate(shared["profile"])
            self._assert_profile(shared_profile)
            sessions = {i + 1: row["session_id"] for i, row in enumerate(shared["state"]["sessions"])}
            report = {"version": self.runner_version, "case_id": CASE, "phase": phase,
                "evidence_kind": self.evidence_kind, "synthetic_emotion": True, "frozen": False,
                "effects": "not_evaluated", "shared": shared, "branches": [], "scored_reply_count": 0}
            report.update(self._report_metadata(case, phase))
            for repeat in (1, 2, 3):
                arms = ("A0", "A1") if (repeat + self.seed) % 2 else ("A1", "A0")
                for arm in arms:
                    if repeat == 1 and arm == "A1":
                        _copy_tree(checkpoint, checkpoint["case_root"] / "repeat-1-A1", target / "repeat-1-A1")
                        branch = deepcopy(checkpoint["completed_branch"])
                        report["branches"].append(branch)
                        report["scored_reply_count"] += 3
                        self._journal({"event": "retained_complete_branch", "case_id": CASE, "repeat": repeat,
                                       "arm": arm, "new_model_calls": 0})
                        continue
                    app = self._new_app(target / f"repeat-{repeat}-{arm}")
                    try:
                        profile = shared_profile.model_copy(deep=True)
                        removed = {"preferences": [], "facts": [], "hypotheses": [], "relationships": []}
                        if arm == "A0":
                            profile, removed = self._a0_profile(profile, case, sessions)
                        self._assert_profile(profile)
                        if repeat == 1 and arm == "A0":
                            self._operation(app, "restore_retained_branch", lambda: restore_partial_branch(app, checkpoint))
                            sid = checkpoint["failed"]["session_id"]
                            metadata = {"case_id": CASE, "phase": "s3", "arm": arm, "repeat": repeat}
                            for turn in range(4, 10):
                                require_unexpired(checkpoint["saved_job"])
                                self._operation(app, "submit", lambda turn=turn: app.submit(case.current_input(3, turn),
                                    f"e1b-{sid}-t{turn}", sid, external_emotion=deepcopy(self.plan[(CASE, 3, turn)])),
                                    detail={**metadata, "turn": turn})
                            self._operation(app, "seal", lambda: app.seal(sid), detail=metadata)
                        else:
                            self._operation(app, "import_profile", lambda: app.store.save(profile))
                            self._session(app, case, 3, arm=arm, repeat=repeat)
                        state = app.state()
                        require(len(state["turns"]) == 10, "prewrite_third_session_count_changed")
                        branch = {"arm": arm, "repeat": repeat, "run_id": state.get("run_id"), "removed": removed,
                            "state": state, "scored_turn_ids": [row["turn_id"] for row in state["turns"][7:]],
                            "scored_replies": [{"turn": i, "response": state["turns"][i]["response"],
                                "source": state["turns"][i]["source"], "evidence_kind": self.evidence_kind} for i in (7, 8, 9)]}
                        report["branches"].append(branch)
                        report["scored_reply_count"] += 3
                        self._operation(app, "save_branch", lambda: _write_json(target / f"repeat-{repeat}-{arm}.json", branch))
                    finally:
                        app.close()
            self._operation(app, "save_case_report", lambda: _write_json(target / "report.json", report))
            return report
    return PrewriteCheckpointRunner
