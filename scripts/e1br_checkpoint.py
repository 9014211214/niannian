"""Read-only stopped E1b-R lineage and local restoration of saved first bodies.

No credentials, transport execution, or alteration of the stopped campaign.
The stopped HTTP 429 has no retrospective no-content proof.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from aria.bench.e1b_runner import E1bRunStopped, _write_json, strip_a0_preference_sources
from aria.bench.e1br_runner import E1BRRunner
from aria.engine.model.persona import PersonaProfile
from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts import run_e1br_received_continuation as parent

MANIFEST = Path("artifacts/local/tasks/20261004-07/e1br/v2-main-stop-originals-v1.manifest.json")
MANIFEST_SHA = "99aad2701713dca24dab9a070b6155fc3577eb95a75305ad7bff127a134e681e"
FREEZE = Path("artifacts/local/tasks/20261004-07/e1br-v2-root-formal-freeze-v1.json")
FREEZE_SHA = "49c4b4192e7363f48a90cab7d10a7aa2cd025f4b332079a52119a0dadaf3454d"
RUN = Path("main/synthetic-emotion-experiments/e1b-main-v1/run")
COMPLETED = ("e1b_a1", "e1b_a2", "e1b_b1")
CASE, ARM, REPEAT = "e1b_b2", "A1", 1
SESSION = "b3fb8e96b64249d1b26489038e8e9cae"
FAILED_REQUEST = "0fe8c3151d424775b909818a31355273"
FAILED_WIRE = "b10887cbe6318aff444c170ac2df9395265144e1374e76480c8ec35d774666c5"
NEW_COUNTS = {"main": {"online_fast": 724, "memory": 16, "reflection": 16}}


def require(value, code):
    if not value:
        raise original.RunError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exclusive(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def assert_originals(checkpoint, *, full=False):
    root = checkpoint["root"]
    files = checkpoint["files"]
    actual = sorted(path.relative_to(root).as_posix() for path in (root / parent.CAMPAIGN).rglob("*") if path.is_file())
    require(actual == sorted(checkpoint["membership"]), "checkpoint_original_membership_changed")
    for row in files:
        path = root / row["path"]
        require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "checkpoint_original_linked")
        require((path.stat().st_size, path.stat().st_mtime_ns) == checkpoint["fingerprint"][row["path"]],
                "checkpoint_original_changed")
        if full:
            require(sha(path.read_bytes()) == row["sha256"], "checkpoint_original_bytes_changed")


def require_empty_care(path):
    require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
            "checkpoint_care_not_closed")
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require(tables == {"elders", "users", "assignments", "sessions", "records", "commands"}
                and all(db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == 0 for table in tables),
                "checkpoint_care_side_effects_not_empty")


def load_checkpoint(root):
    from aria.engine.policy.e1br_calls import validate_e1br_ledger

    root = Path(root).resolve()
    manifest_raw, freeze_raw = (common.read_original(root, path) for path in (MANIFEST, FREEZE))
    require(sha(manifest_raw) == MANIFEST_SHA and sha(freeze_raw) == FREEZE_SHA, "checkpoint_anchors_changed")
    manifest, freeze = json.loads(manifest_raw), json.loads(freeze_raw)
    require(len(manifest["files"]) == 763 and manifest["rows"] == 287 and manifest["known_new_cost_usd"] == "0.313798110",
            "checkpoint_stopped_inventory_changed")
    checkpoint = {"root": root, "files": manifest["files"], "membership": manifest["membership"],
        "fingerprint": {row["path"]: ((root / row["path"]).stat().st_size, (root / row["path"]).stat().st_mtime_ns)
                        for row in manifest["files"]}}
    assert_originals(checkpoint, full=True)
    require(len(freeze["archive_files"]) == len({row["path"] for row in freeze["archive_files"]}) == 133,
            "checkpoint_source_archive_count_changed")
    for row in freeze["archive_files"]:
        require(sha(common.read_original(root, Path(freeze["archive"]) / row["path"])) == row["sha256"],
                "checkpoint_source_archive_bytes_changed")
    directory = root / parent.CAMPAIGN
    registration = original.read_json(directory / "main/registration.json")
    receipt = original.read_json(directory / "main/execution-receipt.json")
    require(registration["seal_sha256"] == freeze["seal_sha256"] == receipt["registration_sha256"]
            and receipt["status"] == "not_evaluable", "checkpoint_registration_or_failed_receipt_changed")
    rows = validate_e1br_ledger(directory / "allowance.sqlite3", stage="main", registration_sha256=freeze["seal_sha256"],
        max_requests=parent.MAX_WIRE_REQUESTS, reservation_policy=registration["metadata"]["policy"])
    require(rows == receipt["accounting"]["attempts"] and len(rows) == 287
            and Counter(row["status"] for row in rows) == {"received": 286, "failed": 1}
            and all(row["product_step"] == "initial" for row in rows), "checkpoint_closed_receipt_rows_changed")
    with sqlite3.connect((directory / "allowance.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require(db.execute("SELECT halted FROM settings").fetchone() == ("model_or_storage_failed",),
                "checkpoint_old_stop_changed")
    failed = rows[-1]
    require(failed["request_id"] == FAILED_REQUEST and failed["request_sha256"] == FAILED_WIRE
            and failed["http_status"] == 429 and failed.get("no_content_evidence") is None
            and not failed.get("visible_content_sha256") and failed["infrastructure_retry_index"] == 0
            and failed["reservation"]["reserved_cost_usd"] == "0.001185872"
            and tuple(failed["experiment_position"]) == ("online_fast", CASE, 3, ARM, REPEAT, 8),
            "checkpoint_old_429_evidence_changed")
    for row in rows:
        require(sha(common.read_original(root, parent.CAMPAIGN / row["wire_request_path"])) == row["request_sha256"],
                "checkpoint_wire_changed")
        if row.get("visible_original_path"):
            require(sha(common.read_original(root, parent.CAMPAIGN / row["visible_original_path"])) == row["visible_original_sha256"],
                    "checkpoint_visible_original_changed")
    case_root = directory / RUN / "main" / CASE
    shared = original.read_json(case_root / "shared.json")
    saved_run = original.read_json(case_root / f"repeat-{REPEAT}-{ARM}/live/run.json")
    saved_profile = original.read_json(case_root / f"repeat-{REPEAT}-{ARM}/live/profiles/fictional-chen.json")
    turns = saved_run["turns"]
    require(not (case_root / "report.json").exists() and len(turns) == 9
            and [turn["status"] for turn in turns] == ["completed"] * 8 + ["failed"]
            and turns[-1]["response"] == "" and turns[-1]["source"] == "model_failed"
            and len(saved_run["sessions"]) == 1 and saved_run["sessions"][0]["session_id"] == SESSION
            and saved_run["sessions"][0]["status"] == "sealed", "checkpoint_partial_branch_changed")
    branch_rows = [row for row in rows if row["role"] == "online_fast" and row["session_id"] == SESSION]
    require(len(branch_rows) == 9 and branch_rows[-1] == failed, "checkpoint_branch_receipts_changed")
    originals = []
    for index, (turn, row) in enumerate(zip(turns, branch_rows)):
        require(turn["turn_id"] == row["operation_id"] == f"e1b-{SESSION}-t{index}" and turn["session_id"] == SESSION,
                "checkpoint_turn_identity_changed")
        body = None
        if index < 8:
            visible = original.read_json(directory / row["visible_original_path"])
            body = visible["content"]
            require(sha(body.encode("utf8")) == row["visible_content_sha256"], "checkpoint_first_content_changed")
        originals.append({"receipt": row, "wire": (directory / row["wire_request_path"]).read_bytes(), "content": body})
    reports = {}
    for case in COMPLETED:
        report_path = directory / RUN / "main" / case / "report.json"
        report = original.read_json(report_path)
        require(report["case_id"] == case and report["scored_reply_count"] == 18 and len(report["branches"]) == 6,
                "checkpoint_completed_case_incomplete")
        reports[case] = sha(report_path.read_bytes())
    require({p.name for p in (directory / RUN / "main").iterdir() if p.is_dir()} == {*COMPLETED, CASE},
            "checkpoint_unexpected_case_started")
    require_empty_care(case_root / f"repeat-{REPEAT}-{ARM}/live/care.sqlite3")
    require_empty_care(case_root / "shared/live/care.sqlite3")
    accepted = rows[:-1]
    require(Counter(row["role"] for row in accepted) == {"online_fast": 272, "memory": 7, "reflection": 7},
            "checkpoint_retained_role_count_changed")
    summary = {"schema": "e1br_stopped_checkpoint_v1", "campaign": parent.CAMPAIGN.as_posix(),
        "manifest_sha256": MANIFEST_SHA, "registration_sha256": registration["seal_sha256"],
        "received_retained_from_v2": 286, "earlier_retained": 14, "total_retained": 300,
        "retained_roles": {"online_fast": 284, "memory": 8, "reflection": 8}, "new_positions": NEW_COUNTS["main"],
        "completed_report_sha256": reports, "failed_request_id": FAILED_REQUEST, "failed_wire_sha256": FAILED_WIRE,
        "partial_case": CASE, "partial_arm": ARM, "partial_repeat": REPEAT, "successful_partial_turns": 8,
        "old_no_content_proven": False, "old_failed_attempts_consumed": 1, "max_future_attempts_for_failed_position": 2,
        "execute_authorized": False, "old_halt_cleared": False, "local_restoration_new_model_calls": 0}
    checkpoint.update(summary=summary, registration=registration, receipt=receipt, rows=rows, accepted=accepted,
        failed=failed, shared=shared, saved_run=saved_run, saved_profile=saved_profile, branch_originals=originals,
        reports=reports, case_root=case_root, source_archive=freeze)
    return checkpoint


def inherited_files(root, checkpoint):
    archive = checkpoint["source_archive"]
    return list(dict.fromkeys([*parent.parent_originals(root), Path(root) / MANIFEST, Path(root) / FREEZE,
        *(Path(root) / row["path"] for row in checkpoint["files"]),
        *(Path(root) / archive["archive"] / row["path"] for row in archive["archive_files"])]))


def qualification_lineage(root, prefix, received, checkpoint, frozen, state, logical_attempts, campaign):
    """Private mechanical provenance; never place this mapping in blind items."""
    old_body_registration = original.read_json(Path(root) / parent.PARENT / "main/registration.json")
    groups = [(parent.retained.PARENT, prefix["registration"], prefix["retained_attempts"]),
        (parent.PARENT, old_body_registration, [received["receipt"]]),
        (parent.CAMPAIGN, checkpoint["registration"], checkpoint["accepted"]),
        (campaign, frozen, state["attempts"])]
    sources, physical = {}, []
    for path, registration, rows in groups:
        source = {"campaign": path.as_posix(), "registration_sha256": registration["seal_sha256"],
            "source_inventory_sha256": original.digest(registration["source"])}
        sources[path.as_posix()] = source
        for index, row in enumerate(rows):
            require(row.get("registration_sha256") == source["registration_sha256"], "checkpoint_lineage_registration_changed")
            position = row.get("experiment_position")
            if position is None:
                require(path == parent.retained.PARENT and index < 13, "checkpoint_position_missing")
                position = [row["role"], "e1b_a1", 1, "shared", 0, index if row["role"] == "online_fast" else None]
            physical.append({**source, "request_id": row["request_id"], "role": row["role"],
                "operation_id": row["operation_id"], "session_id": row["session_id"],
                "position": list(position), "product_step": row.get("product_step", "initial"),
                "status": row["status"], "request_sha256": row["request_sha256"],
                "receipt_sha256": original.digest(row), "retained": path != campaign})
    initial_ids = {row["request_id"] for row in [*prefix["retained_attempts"], received["receipt"],
        *checkpoint["accepted"], *(row for row in logical_attempts(state) if row["product_step"] == "initial")]}
    logical = [row for row in physical if row["request_id"] in initial_ids]
    require(len(logical) == len(initial_ids) == 1056 and len({tuple(row["position"]) for row in logical}) == 1056
        and Counter(row["role"] for row in logical) == {"online_fast": 1008, "memory": 24, "reflection": 24},
        "checkpoint_lineage_denominator_changed")
    return {"schema": "e1br_checkpoint_source_lineage_v1", "sources": list(sources.values()),
        "logical_positions": logical, "physical_attempts": physical,
        "retained_counts": [13, 1, 286], "new_basic_positions": 756,
        "prior_failed_position": {"campaign": parent.CAMPAIGN.as_posix(),
            "request_id": FAILED_REQUEST, "receipt_sha256": original.digest(checkpoint["failed"]),
            "no_content_proven": False, "manifest_sha256": MANIFEST_SHA},
        "completed_reports_keep_original_bytes": deepcopy(checkpoint["reports"]),
        "claim_all_answers_current_source": False}


def copy_original_directory(checkpoint, source, destination):
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), "checkpoint_destination_must_be_new")
    directory = checkpoint["root"] / parent.CAMPAIGN / RUN / "main" / source
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            exclusive(destination / path.relative_to(directory), common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))


def restore_branch(app, checkpoint):
    """Rebuild runtime state with saved bodies; never reissue their model calls."""
    assert_originals(checkpoint, full=True)
    require(not app._run["turns"] and not app._run["sessions"] and app._profile() is None,
            "checkpoint_restore_requires_fresh_app")
    require(not app.care.external and not app.care.allow_external, "checkpoint_external_care_forbidden")
    guard = app.call_guard
    before = guard.snapshot() if guard is not None else None
    original_request = app.clients.online._request_json
    matched = []
    source = checkpoint["branch_originals"]

    def saved_only(request):
        index = len(matched)
        require(index < 8 and request.data == source[index]["wire"], "checkpoint_saved_wire_not_identical")
        row = source[index]["receipt"]
        matched.append(row["request_sha256"])
        return {"id": "local-first-body-restoration", "model": row["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": source[index]["content"]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    app.clients.online._request_json = saved_only
    try:
        app.store.save(PersonaProfile.model_validate(checkpoint["shared"]["profile"]))
        _, ctx = app._session("fictional-chen")
        ctx.session_id = SESSION
        require(app.start()["ok"], "checkpoint_local_start_failed")
        for index, turn in enumerate(checkpoint["saved_run"]["turns"][:8]):
            result = app.submit(turn["text"], turn["turn_id"], SESSION,
                external_emotion={key: deepcopy(turn["external_emotion"][key]) for key in ("label", "confidence", "modality", "source")})
            reproduced = app._run["turns"][-1]
            require(result["ok"] and len(matched) == index + 1
                    and reproduced["response"] == turn["response"] and reproduced["source"] == turn["source"],
                    "checkpoint_saved_delivery_not_identical")
        # Restore original successful UI records, including their real timestamps.
        # The failed attempt remains in the donor and its manifest, not erased.
        app._run["turns"] = deepcopy(checkpoint["saved_run"]["turns"][:8])
        app._run["sessions"][0]["started_at"] = checkpoint["saved_run"]["sessions"][0]["started_at"]
        app._run["retained_failed_attempt"] = {"campaign": parent.CAMPAIGN.as_posix(),
            "request_id": FAILED_REQUEST, "wire_sha256": FAILED_WIRE, "local_fallback_not_model_body": True,
            "original_manifest_sha256": MANIFEST_SHA}
        app._run["retained_local_restoration"] = {"visible_first_bodies": 8, "new_model_calls": 0,
            "source_run_id": checkpoint["saved_run"]["run_id"], "processing_times_are_local_restoration": True}
        old_history = {(row["session_id"], row["session_turn"], row["intervention_id"]): row
                       for row in checkpoint["saved_profile"]["intervention_history"]}
        for record in app._profile().intervention_history:
            value = record.model_dump(mode="json")
            old = old_history.get((value["session_id"], value["session_turn"], value["intervention_id"]))
            require(old is not None and {k: v for k, v in old.items() if k != "at"}
                    == {k: v for k, v in value.items() if k != "at"}, "checkpoint_history_semantics_changed")
            record.at = old["at"]
        app.store.save(app._profile())
        app._save_run()
        if guard is not None:
            require(guard.snapshot() == before, "checkpoint_local_restore_must_not_write_ledger")
        require_empty_care(app.run_dir / "care.sqlite3")
        assert_originals(checkpoint, full=True)
        return {"ok": True, "saved_bodies_locally_consumed": 8, "wire_sha256": matched,
                "new_model_calls": 0, "new_ledger_rows": 0}
    finally:
        app.clients.online._request_json = original_request


def runner_class(checkpoint):
    class CheckpointRunner(E1BRRunner):
        def run_case(self, case_id, *, phase="main", repeats=3):
            require(phase == "main" and repeats == 3, "checkpoint_only_registered_main")
            assert_originals(checkpoint)
            if case_id in COMPLETED:
                target = self.root / "main" / case_id
                copy_original_directory(checkpoint, case_id, target)
                require(sha((target / "report.json").read_bytes()) == checkpoint["reports"][case_id],
                        "checkpoint_retained_report_bytes_changed")
                self._journal({"event": "retained_complete_case", "case_id": case_id,
                    "source_campaign": parent.CAMPAIGN.as_posix(), "report_sha256": checkpoint["reports"][case_id],
                    "new_model_calls": 0, "original_source_binding_preserved": True})
                return original.read_json(target / "report.json")
            if case_id != CASE:
                return super().run_case(case_id, phase=phase, repeats=repeats)
            target = self.root / "main" / CASE
            target.mkdir(parents=True, exist_ok=False)
            for path in sorted(checkpoint["case_root"].iterdir()):
                if path.is_file():
                    exclusive(target / path.name, common.read_original(checkpoint["root"], path.relative_to(checkpoint["root"])))
            case = next(row for row in self.suite.items if row.id == CASE)
            shared = deepcopy(checkpoint["shared"])
            shared_profile = PersonaProfile.model_validate(shared["profile"])
            self._assert_profile(shared_profile)
            preference_session = shared["state"]["sessions"][1]["session_id"]
            template = original.read_json(checkpoint["root"] / parent.CAMPAIGN / RUN / "main" / COMPLETED[0] / "report.json")
            report = {name: template[name] for name in ("version", "phase", "evidence_kind", "synthetic_emotion", "frozen", "effects")}
            report.update(case_id=CASE, shared=shared, branches=[], scored_reply_count=0)
            for repeat in range(1, 4):
                for arm in (("A0", "A1") if (repeat + self.seed) % 2 else ("A1", "A0")):
                    app = self._new_app(target / f"repeat-{repeat}-{arm}")
                    try:
                        removed = {"preferences": [], "facts": [], "hypotheses": [], "relationships": []}
                        if (repeat, arm) == (REPEAT, ARM):
                            self._operation(app, "restore_saved_online_prefix", lambda: restore_branch(app, checkpoint),
                                detail={"case_id": CASE, "retained_answers": 8, "new_model_calls": 0})
                            for turn in (8, 9):
                                self._operation(app, "submit", lambda turn=turn: app.submit(case.current_input(3, turn),
                                    f"e1b-{SESSION}-t{turn}", SESSION, external_emotion=deepcopy(self.plan[(CASE, 3, turn)])),
                                    detail={"case_id": CASE, "phase": "s3", "arm": arm, "repeat": repeat, "turn": turn})
                            self._operation(app, "seal", lambda: app.seal(SESSION),
                                detail={"case_id": CASE, "phase": "s3", "arm": arm, "repeat": repeat})
                        else:
                            profile = shared_profile.model_copy(deep=True)
                            if arm == "A0":
                                profile, removed = strip_a0_preference_sources(profile, session_id=preference_session, turn_index=case.pref_turn)
                            self._assert_profile(profile)
                            self._operation(app, "import_profile", lambda: app.store.save(profile))
                            self._session(app, case, 3, arm=arm, repeat=repeat)
                        state = app.state()
                        rows = state["turns"]
                        if len(rows) != 10:
                            raise E1bRunStopped("checkpoint_branch_turn_count_changed")
                        branch = {"arm": arm, "repeat": repeat, "run_id": state["run_id"], "removed": removed, "state": state,
                            "scored_turn_ids": [row["turn_id"] for row in rows[7:]],
                            "scored_replies": [{"turn": index, "response": rows[index]["response"], "source": rows[index]["source"],
                                                "evidence_kind": self.evidence_kind} for index in (7, 8, 9)]}
                        report["branches"].append(branch)
                        report["scored_reply_count"] += 3
                        self._operation(app, "save_branch", lambda: _write_json(target / f"repeat-{repeat}-{arm}.json", branch))
                    finally:
                        app.close()
            _write_json(target / "report.json", report)
            return report
    return CheckpointRunner
