"""Persona persistence retries only the final replace, using synthetic errors."""
from pathlib import Path

import pytest

from aria.engine.memory import atomic_file, persona_store
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile


def locked(code):
    error = PermissionError("synthetic atomic replacement refusal")
    error.winerror = code
    return error


def prepared(tmp_path, monkeypatch):
    monkeypatch.delenv("ARIA_PROFILE_KEY", raising=False)
    profile = PersonaProfile(user_id="atomic-fixture", profile_origin="fictional_roleplay")
    target = tmp_path / "atomic-fixture.json"
    old = PersonaProfile(user_id="atomic-fixture", profile_origin="blank").model_dump_json().encode()
    target.write_bytes(old)
    pending = target.with_suffix(".json.tmp")
    counts = {"serialized": 0, "encrypted": 0, "written": 0}
    serialize = PersonaProfile.model_dump_json
    encrypt = persona_store.encrypt_if_keyed
    write = Path.write_text
    def serialized(self, *args, **kwargs):
        counts["serialized"] += 1
        return serialize(self, *args, **kwargs)
    def encrypted(*args, **kwargs):
        counts["encrypted"] += 1
        return encrypt(*args, **kwargs)
    def written(path, *args, **kwargs):
        if path == pending:
            counts["written"] += 1
        return write(path, *args, **kwargs)
    monkeypatch.setattr(PersonaProfile, "model_dump_json", serialized)
    monkeypatch.setattr(persona_store, "encrypt_if_keyed", encrypted)
    monkeypatch.setattr(Path, "write_text", written)
    return PersonaStore(tmp_path), profile, target, pending, old, counts


@pytest.mark.parametrize("code", [5, 32, 33])
def test_persona_transient_replace_reuses_written_bytes_without_repeating_work(tmp_path, monkeypatch, code):
    store, profile, target, pending, old, counts = prepared(tmp_path, monkeypatch)
    original_replace = Path.replace
    attempts, pauses = [], []
    def replace(source, destination):
        assert source == pending and destination == target
        attempts.append(source.read_bytes())
        assert target.read_bytes() == old
        if len(attempts) < 3:
            raise locked(code)
        return original_replace(source, destination)
    monkeypatch.setattr(atomic_file.sys, "platform", "win32")
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(atomic_file.time, "sleep", pauses.append)
    store.save(profile)
    assert counts == {"serialized": 1, "encrypted": 1, "written": 1}
    assert attempts[0] == attempts[1] == attempts[2] == target.read_bytes()
    assert pauses == [0.02, 0.04]
    assert not pending.exists()
    restored = store.load(profile.user_id)
    assert restored.profile_origin == "fictional_roleplay"


@pytest.mark.parametrize("platform,code,expected", [
    ("win32", 5, 5), ("win32", 32, 5), ("win32", 33, 5),
    ("win32", 87, 1), ("win32", None, 1), ("linux", 5, 1),
])
def test_persona_replace_exhaustion_and_ineligible_errors_preserve_failure_and_original(
    tmp_path, monkeypatch, platform, code, expected,
):
    store, profile, target, pending, old, counts = prepared(tmp_path, monkeypatch)
    attempts, pauses = [], []
    error = locked(code)
    def replace(source, destination):
        assert source == pending and destination == target
        attempts.append(source.read_bytes())
        assert target.read_bytes() == old
        raise error
    monkeypatch.setattr(atomic_file.sys, "platform", platform)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(atomic_file.time, "sleep", pauses.append)
    with pytest.raises(PermissionError) as caught:
        store.save(profile)
    assert caught.value is error
    assert counts == {"serialized": 1, "encrypted": 1, "written": 1}
    assert len(attempts) == expected
    assert pauses == [0.02 * n for n in range(1, expected)]
    assert target.read_bytes() == old
    assert pending.read_bytes() == attempts[0]
    assert all(raw == attempts[0] for raw in attempts)


def test_persona_pending_write_failure_is_not_retried(tmp_path, monkeypatch):
    store, profile, target, pending, old, counts = prepared(tmp_path, monkeypatch)
    error = locked(5)
    attempts = []
    def write(path, *args, **kwargs):
        assert path == pending
        attempts.append(path)
        raise error
    monkeypatch.setattr(atomic_file.sys, "platform", "win32")
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", lambda *a: pytest.fail("replacement after failed write"))
    monkeypatch.setattr(atomic_file.time, "sleep", lambda *a: pytest.fail("retrying surrounding write"))
    with pytest.raises(PermissionError) as caught:
        store.save(profile)
    assert caught.value is error and attempts == [pending]
    assert counts["serialized"] == counts["encrypted"] == 1
    assert target.read_bytes() == old and not pending.exists()
