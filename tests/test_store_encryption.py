"""档案静态加密(P5-#20):往返 / 明文迁移 / fail-loud 红线。

需要 cryptography(pip install aria-eldercare-agent[secure]);未安装则跳过。
"""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile


def _store(tmp_path):
    return PersonaStore(base_dir=tmp_path)


def test_encrypted_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_PROFILE_KEY", "correct horse battery")
    store = _store(tmp_path)
    store.save(PersonaProfile(user_id="u1", attachment_style="avoidant"))
    raw = (tmp_path / "u1.json").read_text(encoding="utf-8")
    assert raw.startswith("ARIAENC1:")            # 落盘是密文
    assert "avoidant" not in raw                  # 明文不可见
    p = store.load("u1")
    assert p is not None and p.attachment_style == "avoidant"


def test_plaintext_migrates_on_next_save(tmp_path, monkeypatch):
    monkeypatch.delenv("ARIA_PROFILE_KEY", raising=False)
    store = _store(tmp_path)
    store.save(PersonaProfile(user_id="u1"))      # 无 key:明文(开发默认)
    assert (tmp_path / "u1.json").read_text(encoding="utf-8").lstrip().startswith("{")

    monkeypatch.setenv("ARIA_PROFILE_KEY", "correct horse battery")
    p = store.load("u1")                          # 有 key 读明文:正常
    assert p is not None
    store.save(p)                                 # 下次保存:平滑迁移为密文
    assert (tmp_path / "u1.json").read_text(encoding="utf-8").startswith("ARIAENC1:")


def test_encrypted_without_key_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_PROFILE_KEY", "correct horse battery")
    store = _store(tmp_path)
    store.save(PersonaProfile(user_id="u1"))
    monkeypatch.delenv("ARIA_PROFILE_KEY")
    with pytest.raises(RuntimeError):
        store.load("u1")                          # 绝不静默当新用户(会覆盖档案)


def test_wrong_key_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_PROFILE_KEY", "correct horse battery")
    _store(tmp_path).save(PersonaProfile(user_id="u1"))
    monkeypatch.setenv("ARIA_PROFILE_KEY", "wrong key")
    with pytest.raises(RuntimeError):
        _store(tmp_path).load("u1")
