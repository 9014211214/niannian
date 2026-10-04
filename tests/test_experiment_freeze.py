"""Offline freeze checks use small authored trees, never private project data."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import verify_experiment_freeze as freeze


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal(source, manifest):
    manifest.pop("seal_sha256", None)
    manifest["seal_sha256"] = freeze.canonical_sha256(manifest)
    _save(source / freeze.DEFAULT_MANIFEST, manifest)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source = tmp_path / "source"
    plan_name = "experiments/t6_20260930/plan.json"
    records = [{"order": i + 1, "case_id": "case", "arm": "B0", "repeat": i + 1,
                "request_sha256": "a" * 64} for i in range(204)]
    monkeypatch.setattr(freeze, "e1_request_records", lambda plan: deepcopy(records))
    _save(source / plan_name, {"e1": "authored fixture"})
    (source / "aria").mkdir()
    (source / "aria/module.py").write_text("value = 1\n", encoding="utf-8")
    entries = [{"path": name, "bytes": (source / name).stat().st_size,
                "sha256": freeze.file_sha256(source / name)}
               for name in ("aria/module.py", plan_name)]
    manifest = {"schema": freeze.SCHEMA, "plan_path": plan_name, "files": entries,
                "e1_requests": deepcopy(records),
                "e1_requests_sha256": freeze.canonical_sha256(records)}
    _seal(source, manifest)
    return source, manifest


def test_verify_is_read_only_and_does_not_read_private_files(frozen, monkeypatch):
    source, manifest = frozen
    private = source / ".env"
    private.write_text("authored do-not-read sentinel", encoding="utf-8")
    before = {path.relative_to(source).as_posix(): path.read_bytes()
              for path in source.rglob("*") if path.is_file()}
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path == private:
            raise AssertionError("private file opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = freeze.verify(source, freeze.DEFAULT_MANIFEST)
    assert result == {"ok": True, "schema": freeze.SCHEMA, "source_files": 1,
                      "verified_files": 2, "e1_requests": 204,
                      "seal_sha256": manifest["seal_sha256"], "network_calls": 0}
    for name, content in before.items():
        if name != ".env":
            assert (source / name).read_bytes() == content
    assert set(before) == {path.relative_to(source).as_posix()
                           for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("change", ["modify", "add", "delete"])
def test_changed_source_rejected(frozen, change):
    source, _ = frozen
    selected = source / "aria/module.py"
    if change == "modify":
        selected.write_text("value = 2\n", encoding="utf-8")
    elif change == "add":
        (source / "aria/new.py").write_text("value = 1\n", encoding="utf-8")
    else:
        selected.unlink()
    with pytest.raises((freeze.FreezeError, FileNotFoundError)):
        freeze.verify(source)


def test_inventory_tampering_rejected_even_with_recomputed_seal(frozen):
    source, manifest = frozen
    manifest["e1_requests"][0]["arm"] = "B3b"
    manifest["e1_requests_sha256"] = freeze.canonical_sha256(manifest["e1_requests"])
    _seal(source, manifest)
    with pytest.raises(freeze.FreezeError, match="request_inventory_mismatch"):
        freeze.verify(source)


def test_broken_seal_rejected(frozen):
    source, manifest = frozen
    manifest["e1_requests"][0]["arm"] = "B3b"
    _save(source / freeze.DEFAULT_MANIFEST, manifest)
    with pytest.raises(freeze.FreezeError, match="seal_mismatch"):
        freeze.verify(source)


@pytest.mark.parametrize("name", ["../outside.json", "C:/outside.json", "/outside.json",
                                  "docs/../../.env", ".env", "data/private.json",
                                  "docs/.env.json", "docs/secrets.json", "aria\\module.py"])
def test_unsafe_file_paths_rejected_before_read(frozen, name):
    source, manifest = frozen
    manifest["files"].append({"path": name, "bytes": 0, "sha256": "0" * 64})
    _seal(source, manifest)
    with pytest.raises(freeze.FreezeError, match="unsafe_manifest_path"):
        freeze.verify(source)


@pytest.mark.parametrize("field", ["manifest", "plan"])
def test_manifest_and_plan_paths_cannot_escape(frozen, field):
    source, manifest = frozen
    if field == "manifest":
        with pytest.raises(freeze.FreezeError):
            freeze.verify(source, "../outside.json")
    else:
        manifest["plan_path"] = "../outside.json"
        _seal(source, manifest)
        with pytest.raises(freeze.FreezeError):
            freeze.verify(source)


def test_plan_must_be_bound_as_a_file(frozen):
    source, manifest = frozen
    manifest["files"] = manifest["files"][:1]
    _seal(source, manifest)
    with pytest.raises(freeze.FreezeError, match="plan_not_bound"):
        freeze.verify(source)


def test_symlink_rejected_without_following(frozen, tmp_path):
    source, manifest = frozen
    outside = tmp_path / "external.json"
    outside.write_text("{}", encoding="utf-8")
    link = source / "experiments/linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("host cannot create symlinks")
    manifest["files"].append({"path": "experiments/linked.json", "bytes": 2, "sha256": "0" * 64})
    _seal(source, manifest)
    with pytest.raises(freeze.FreezeError, match="linked_path_rejected"):
        freeze.verify(source)


def test_bytecode_is_not_part_of_source_inventory(frozen):
    source, _ = frozen
    bytecode = source / "aria/__pycache__/module.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"authored ignored bytecode")
    assert freeze.verify(source)["source_files"] == 1


def test_cli_failure_is_compact_and_does_not_expose_contents(frozen, capsys):
    source, _ = frozen
    (source / "aria/module.py").write_text("private-looking fixture\n", encoding="utf-8")
    assert freeze.main(["--root", str(source)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output == {"ok": False, "reason": "file_content_mismatch"}


def _e1_plan():
    return {"e1": {
        "preparation": {"model": "local/e1-freeze", "temperature": 0.2,
                        "max_completion_tokens": 1024, "seed": 0,
                        "max_cost_usd": 0.5, "per_request_reserve_usd": 0.005},
        "wire": {"model": "qwen/qwen3.7-flash", "temperature": 0.2, "max_tokens": 1024,
                 "reasoning": {"enabled": False, "exclude": True},
                 "provider": {"only": ["authored-provider-fixture"], "allow_fallbacks": False,
                              "require_parameters": True, "data_collection": "deny"},
                 "stream": False}}}


@pytest.mark.parametrize("field", ["messages", "tools", "tool_choice", "max_completion_tokens"])
def test_wire_cannot_replace_authored_messages_or_tools(field):
    plan = _e1_plan()
    plan["e1"]["wire"][field] = "unreviewed substitution"
    with pytest.raises(freeze.FreezeError, match="invalid_e1_wire_fields"):
        freeze.e1_request_records(plan)


def test_actual_e1_reconstruction_is_offline_deterministic_and_content_free(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("network call forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    plan = _e1_plan()
    original = deepcopy(plan)
    first = freeze.e1_request_records(plan)
    assert first == freeze.e1_request_records(plan)
    assert plan == original
    assert len(first) == 204
    assert [row["order"] for row in first] == list(range(1, 205))
    assert len({(row["case_id"], row["arm"]) for row in first}) == 68
    assert all(set(row) == {"order", "case_id", "arm", "repeat", "request_sha256"}
               for row in first)
    changed = deepcopy(plan)
    changed["e1"]["wire"]["provider"]["only"] = ["different-authored-provider"]
    assert all(a["request_sha256"] != b["request_sha256"]
               for a, b in zip(first, freeze.e1_request_records(changed)))
