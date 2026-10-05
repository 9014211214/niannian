"""Portable prewrite restore: preserve failure while excluding it from context."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import stat
from types import SimpleNamespace
from pathlib import Path

import pytest

from aria.bench.e1b_runner import E1bLocalFixtureClient
from aria.engine.model.persona import OfflineSessionRecord
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.runtime import MinimalModelClients
from scripts import e1br_prewrite_checkpoint as checkpoint
from scripts.e1br_memory_runtime import MemoryExperimentApp


@pytest.fixture
def restored(tmp_path, monkeypatch):
    token = SyntheticEmotionExperiment.create(tmp_path, "e1b-main-v1")
    clients = MinimalModelClients(*(E1bLocalFixtureClient(role) for role in ("online", "memory", "reflection")), model_calls_enabled=True)
    app = MemoryExperimentApp(token.root / "app", experiment=token, clients=clients)
    at = datetime.now(timezone.utc) - timedelta(minutes=3)
    saved = deepcopy(app._run)
    saved["run_id"] = "old-run"
    saved["sessions"] = [{"session_id": "old-session", "number": 1, "status": "sealed",
                          "started_at": at.isoformat(), "sealed_at": at.isoformat()}]
    saved["turns"] = [{"turn_id": f"e1b-old-session-t{i}", "session_id": "old-session", "session_number": 1,
        "turn_index": i, "status": "completed", "source": "live", "text": f"软件输入{i}",
        "response": f"软件原答{i}", "created_at": at.isoformat()} for i in range(5)]
    saved["turns"][-1].update(status="failed", source="model_failed", response="", requires_review=True)
    persona = app._seed_persona("fictional-chen")
    persona.alliance.trust = .77
    pairs = [{"user": row["text"], "agent": row["response"]} for row in saved["turns"]]
    pairs[-1]["agent"] = DETERMINISTIC_MODEL_FAILURE_REPLY
    persona.offline_sessions = [OfflineSessionRecord(session_id="old-session", scene_id=app.scene_id,
        created_at=at.isoformat(), expires_at=(at + timedelta(days=1)).isoformat(), turns=pairs)]
    case = tmp_path / "old-case"
    old_run_path = case / "repeat-1-A0/live/run.json"
    old_run_path.parent.mkdir(parents=True)
    old_run_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    cp = {"root": tmp_path, "case_root": case, "saved_run": saved,
          "saved_profile": persona.model_dump(mode="json"), "saved_job": persona.offline_sessions[0].model_dump(mode="json"),
          "failed": {"request_id": "software-old-request", "request_sha256": "a" * 64,
                     "session_id": "old-session", "operation_id": "e1b-old-session-t4"}}
    monkeypatch.setattr(checkpoint, "assert_originals", lambda *_args, **_kwargs: None)
    try:
        yield app, clients, cp
    finally:
        app.close()


def test_restore_keeps_success_prefix_profile_dates_and_failed_side_record(restored):
    app, clients, cp = restored
    original = deepcopy(cp)
    result = checkpoint.restore_partial_branch(app, cp)
    assert result["old_fallback_in_active_context"] is False
    assert cp == original
    assert app._run["turns"] == cp["saved_run"]["turns"][:4]
    assert app._run["sessions"][0]["started_at"] == cp["saved_run"]["sessions"][0]["started_at"]
    proof = app._run["transport_continuation"]
    assert proof["retained_failed_turn"] == cp["saved_run"]["turns"][4]
    assert proof["original_job"] == cp["saved_job"] and proof["old_failure_replaced"] is False
    engine, ctx = app._sessions["fictional-chen"]
    assert len(ctx.turns) == 4 and all(turn.agent != DETERMINISTIC_MODEL_FAILURE_REPLY for turn in ctx.turns)
    assert ctx.session_id == "old-session" and engine.persona.alliance.trust == .77
    assert not engine.persona.offline_sessions and engine.persona.pending_session is None
    assert not app._unresolved()
    assert all(client.calls == [] for client in (clients.online, clients.memory, clients.reflection))


def test_expired_restore_rejects_before_mutating_app(restored):
    app, _, cp = restored
    cp["saved_job"]["created_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cp["saved_job"]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    before = deepcopy(app._run)
    with pytest.raises(ValueError, match="original_branch_retention_expired"):
        checkpoint.restore_partial_branch(app, cp)
    assert app._run == before and app._profile() is None


@pytest.mark.parametrize("change", ["position", "operation", "session", "body"])
def test_first_wire_scope_or_body_change_rejected(restored, change):
    _, _, cp = restored
    payload = {"messages": [{"role": "user", "content": "软件专用"}], "model": "software/model"}
    cp["wire"] = json.dumps(payload, ensure_ascii=False).encode()
    scope = {"operation_id": cp["failed"]["operation_id"], "session_id": cp["failed"]["session_id"]}
    key = list(checkpoint.FIRST)
    if change == "position": key[-1] += 1
    elif change == "operation": scope["operation_id"] += "-fresh"
    elif change == "session": scope["session_id"] += "-fresh"
    else: payload["messages"][0]["content"] += "changed"
    with pytest.raises(ValueError, match="retained_first_(scope_changed|wire_not_byte_identical)"):
        checkpoint.validate_first_request(cp, payload, scope, key)


def test_first_wire_identity_preserved_without_dispatch(restored):
    _, _, cp = restored
    payload = {"messages": [{"role": "user", "content": "软件专用"}], "model": "software/model"}
    cp["wire"] = json.dumps(payload, ensure_ascii=False).encode()
    value = checkpoint.validate_first_request(cp, payload,
        {"operation_id": cp["failed"]["operation_id"], "session_id": cp["failed"]["session_id"]}, checkpoint.FIRST)
    assert value["old_scope_preserved"] and value["new_model_calls"] == 0


@pytest.fixture
def closed_source(tmp_path, monkeypatch):
    directory = tmp_path / checkpoint.PARENT
    directory.mkdir(parents=True)
    file = directory / "software.json"
    file.write_bytes(b"software-only")
    row = {"path": file.relative_to(tmp_path).as_posix(), "sha256": checkpoint.sha(file.read_bytes())}
    cp = {"root": tmp_path, "files": [row], "members": [[row["path"], False]],
          "freeze": {"archive_files": []}, "older": None,
          "fingerprints": {row["path"]: (file.stat().st_size, file.stat().st_mtime_ns)}}
    monkeypatch.setattr(checkpoint.parent, "assert_originals", lambda *_args, **_kwargs: None)
    return cp, directory, file


def test_exact_closed_members_and_bytes_pass(closed_source):
    cp, _, _ = closed_source
    checkpoint.assert_originals(cp, full=True)


def test_manifest_traversal_order_does_not_change_exact_membership(closed_source):
    cp, directory, _ = closed_source
    added = directory / "a-empty-directory"
    added.mkdir()
    cp["members"].append([added.relative_to(cp["root"]).as_posix(), True])
    assert cp["members"] != sorted(cp["members"])
    checkpoint.assert_originals(cp, full=True)


def test_duplicate_manifest_path_rejected(closed_source):
    cp, _, _ = closed_source
    cp["members"].append(deepcopy(cp["members"][0]))
    with pytest.raises(ValueError, match="closed_v5_membership_changed"):
        checkpoint.assert_originals(cp, full=True)


@pytest.mark.parametrize("change", ["extra_file", "extra_directory", "missing_file"])
def test_closed_membership_change_rejected(closed_source, change):
    cp, directory, file = closed_source
    if change == "extra_file":
        (directory / "not-in-manifest.json").write_bytes(b"unreviewed software")
    elif change == "extra_directory":
        (directory / "new-empty-directory").mkdir()
    else:
        file.unlink()
    with pytest.raises(ValueError, match="closed_v5_membership_changed"):
        checkpoint.assert_originals(cp, full=True)


@pytest.mark.parametrize("kind", ["symlink", "windows_reparse"])
def test_link_flags_rejected_before_following_or_reading_member(closed_source, monkeypatch, kind):
    cp, _, file = closed_source
    original = Path.lstat
    def observed(path):
        if path == file:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else stat.S_IFREG,
                st_file_attributes=0x400 if kind == "windows_reparse" else 0)
        return original(path)
    # Portable boundary simulation; not a claim of OS symlink privileges.
    monkeypatch.setattr(Path, "lstat", observed)
    with pytest.raises(ValueError, match="closed_v5_link_rejected"):
        checkpoint.assert_originals(cp, full=True)
