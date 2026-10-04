from pathlib import Path

import pytest

from aria.engine.memory import atomic_file


def locked(code=5):
    exc = PermissionError("synthetic locked file")
    exc.winerror = code
    return exc


def test_transient_lock_retries_only_replace_and_preserves_bytes(tmp_path, monkeypatch):
    src, dst = tmp_path / "pending", tmp_path / "saved"
    src.write_bytes(b"new"); dst.write_bytes(b"old")
    original = Path.replace
    attempts, pauses = [], []
    def replace(path, target):
        attempts.append(path.read_bytes())
        if len(attempts) < 3:
            assert dst.read_bytes() == b"old"
            raise locked(32)
        return original(path, target)
    monkeypatch.setattr(atomic_file.sys, "platform", "win32")
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(atomic_file.time, "sleep", pauses.append)
    atomic_file.replace_with_retry(src, dst)
    assert attempts == [b"new"] * 3
    assert pauses == [0.02, 0.04]
    assert dst.read_bytes() == b"new" and not src.exists()


@pytest.mark.parametrize("platform,code,expected", [("win32", 5, 5), ("win32", 32, 5),
                                                     ("win32", 33, 5), ("win32", 87, 1), ("linux", 5, 1)])
def test_real_failure_remains_visible_and_old_target_unchanged(tmp_path, monkeypatch, platform, code, expected):
    src, dst = tmp_path / "pending", tmp_path / "saved"
    src.write_bytes(b"new"); dst.write_bytes(b"old")
    calls = []
    def failed(*args):
        calls.append(1)
        raise locked(code)
    monkeypatch.setattr(atomic_file.sys, "platform", platform)
    monkeypatch.setattr(Path, "replace", failed)
    monkeypatch.setattr(atomic_file.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        atomic_file.replace_with_retry(src, dst)
    assert len(calls) == expected
    assert src.read_bytes() == b"new" and dst.read_bytes() == b"old"


def test_cross_directory_rejected_before_replacement(tmp_path):
    with pytest.raises(ValueError):
        atomic_file.replace_with_retry(tmp_path / "a" / "x", tmp_path / "b" / "x")
