"""Detached-install defaults and fail-closed profile persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from aria import _env
from aria.bench.provenance import build_reproducibility
from aria.engine.memory import persona_store
from aria.engine.memory.persona_store import PersonaStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_prefers_working_directory_for_detached_install(
    tmp_path, monkeypatch
):
    key = "ARIA_RELEASE_ENV_CWD_TEST"
    (tmp_path / ".env").write_text(f"{key}=from-cwd\n", encoding="utf-8")
    monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_env, "_source_checkout_root", lambda: None)

    _env.load_local_env()

    assert _env.os.environ[key] == "from-cwd"


def test_explicit_env_path_takes_precedence(tmp_path, monkeypatch):
    key = "ARIA_RELEASE_ENV_EXPLICIT_TEST"
    explicit = tmp_path / "named.env"
    explicit.write_text(f"{key}=explicit\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"{key}=cwd\n", encoding="utf-8")
    monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    _env.load_local_env(str(explicit))

    assert _env.os.environ[key] == "explicit"


def test_store_uses_platform_directory_when_not_in_checkout(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ARIA_USER_MEMORY_DIR", raising=False)
    monkeypatch.setattr(
        persona_store, "_source_checkout_memory_dir", lambda: None
    )
    monkeypatch.setattr(
        persona_store, "_platform_user_data_dir", lambda: tmp_path
    )

    assert PersonaStore()._dir == tmp_path


def test_store_preserves_checkout_directory(tmp_path, monkeypatch):
    legacy = tmp_path / "data" / "user_memory"
    monkeypatch.delenv("ARIA_USER_MEMORY_DIR", raising=False)
    monkeypatch.setattr(
        persona_store, "_source_checkout_memory_dir", lambda: legacy
    )

    assert PersonaStore()._dir == legacy


def test_corrupt_profile_fails_closed_without_logging_value(
    tmp_path, caplog
):
    value = "TOPSECRET_PROFILE_VALUE"
    (tmp_path / "u1.json").write_text(
        json.dumps({"user_id": {"private": value}}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="persona profile is corrupt"):
            PersonaStore(tmp_path).load("u1")

    assert value not in caplog.text
    assert "u1" in caplog.text


def test_delete_removes_exact_atomic_temp_file(tmp_path):
    temp = tmp_path / "u1.json.tmp"
    temp.write_text("partial", encoding="utf-8")

    assert PersonaStore(tmp_path).delete("u1") is True
    assert not temp.exists()
    assert PersonaStore(tmp_path).delete("u1") is False


def test_benchmark_provenance_never_serializes_credentials():
    class Client:
        model = "safe-model-name"
        api_key = "TOPSECRET_API_KEY"
        base_url = "https://private.example.invalid"

    rendered = json.dumps(build_reproducibility(Client()))
    assert "safe-model-name" in rendered
    assert "TOPSECRET_API_KEY" not in rendered
    assert "private.example.invalid" not in rendered


def test_source_distribution_excludes_benchmark_report_payloads():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "prune bench_reports" in manifest
    assert "recursive-include bench_reports" not in manifest
    assert "bench_reports *.json" not in manifest


def test_contributor_checks_match_ci_entry_points():
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    expected = (
        'python -m pip install -e ".[dev,secure]" build twine',
        "python -m ruff check aria examples scripts tests",
        "python -m pytest -q", "python -m pip check", "python -m build",
        "python scripts/build_minimal_release.py --check",
    )
    assert all(command in contributing for command in expected)
    assert "\naria-web\n" not in contributing


def test_distribution_name_does_not_collide_with_aria_compute_package():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "aria-eldercare-agent"' in pyproject
    assert 'name = "aria-engine"' not in pyproject


def test_license_notices_describe_only_retained_distribution():
    from scripts import build_minimal_release

    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "MIT" in notices and "CPED" in notices
    assert "CosyVoice samples" in notices and "third-party datasets" in notices
    assert not build_minimal_release._allowed("aria/assets/voice/zero_shot_prompt.wav")
    assert not build_minimal_release._allowed("docs/media/minimal-local-preview.mp4")
    assert (REPO_ROOT / "LICENSES/Apache-2.0.txt").is_file()
