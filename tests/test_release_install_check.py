"""Package verifier boundaries, using authored filenames and local subprocesses."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from scripts import check_release_install as release


def members():
    return [*("aria/" + path for path in release.ASSETS),
            *("aria.dist-info/licenses/" + name for name in release.LICENSE_FILES)]


def test_real_distribution_layout_preserves_assets_and_license_notices():
    assert release.inspect_members(members(), sdist=False)["required_assets"] == 11
    source = ["aria-source/" + name for name in [*members(), *release.SOURCE_ENTRY_FILES]]
    assert release.inspect_members(source, sdist=True)["license_files"] == sorted(release.LICENSE_FILES)


@pytest.mark.parametrize("removed", sorted(release.SOURCE_ENTRY_FILES))
def test_sdist_cannot_silently_omit_current_continuation_entry(removed):
    source = ["aria-source/" + name for name in
              [*members(), *(release.SOURCE_ENTRY_FILES - {removed})]]
    with pytest.raises(release.InstallCheckError, match="source_entry_missing"):
        release.inspect_members(source, sdist=True)


@pytest.mark.parametrize("name", [
    "../escape.py", "/outside.py", "C:/private.py", "aria\\private.py",
    ".env", ".env.production", "data/private.json", "artifacts/local/run.json",
    ".git/config", "docs/media/private.png",
])
def test_package_privacy_and_path_failure_blocks_check(name):
    with pytest.raises(release.InstallCheckError):
        release.inspect_members([*members(), name], sdist=False)
    with pytest.raises(release.InstallCheckError):
        release.inspect_members(["aria-source/" + path for path in [*members(), name]], sdist=True)


@pytest.mark.parametrize("removed", ["LICENSE", "Apache-2.0.txt", "THIRD_PARTY_NOTICES.md",
                                     "minimal.html", "refocus_e1_v1.yaml"])
def test_distribution_cannot_omit_notice_or_runtime_resource(removed):
    names = [name for name in members() if Path(name).name != removed]
    with pytest.raises(release.InstallCheckError, match="missing"):
        release.inspect_members(names, sdist=False)


def test_environment_never_passes_credentials_provider_settings_or_python_injection(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "ARIA_CARE_WEBHOOK_URL", "PRIVATE_TOKEN",
                 "PYTHONPATH", "PIP_EXTRA_INDEX_URL"):
        monkeypatch.setenv(name, "synthetic-placeholder")
    result = release.safe_environment()
    assert not any(name in result for name in (
        "OPENROUTER_API_KEY", "ARIA_CARE_WEBHOOK_URL", "PRIVATE_TOKEN", "PYTHONPATH", "PIP_EXTRA_INDEX_URL"))
    assert result["PIP_CONFIG_FILE"] == os.devnull


def test_ambiguous_artifacts_do_not_select_first_candidate(tmp_path):
    for name in ("one.whl", "two.whl", "one.tar.gz"):
        (tmp_path / name).touch()
    with pytest.raises(release.InstallCheckError, match="exactly_one"):
        release.artifacts(tmp_path)


def test_existing_work_directory_is_preserved_without_installing(tmp_path, monkeypatch):
    work = tmp_path / "existing"
    work.mkdir()
    owner = work / "owner.txt"
    owner.write_text("owner artifact", encoding="utf-8")
    monkeypatch.setattr(release, "artifacts", lambda _path: (Path("wheel.whl"), Path("source.tar.gz")))
    monkeypatch.setattr(release, "inspect_distribution", lambda _path: {})
    monkeypatch.setattr(release, "run", lambda *_args: pytest.fail("must not install into an old directory"))
    with pytest.raises(FileExistsError):
        release.check_installs(tmp_path, work)
    assert owner.read_text(encoding="utf-8") == "owner artifact"


def test_subprocess_failure_does_not_print_credential_text(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess([], 17, b"private stdout", b"private stderr"))
    with pytest.raises(release.InstallCheckError, match="fixture_failed_exit_17"):
        release.run(["synthetic-command"], tmp_path, "fixture")
    assert not capsys.readouterr().out


def test_probe_rejects_source_import_without_opening_env_or_starting_server(monkeypatch):
    import aria
    from aria import _env
    monkeypatch.setattr(_env, "load_local_env", lambda *_args: pytest.fail("must not read env"))
    # The active editable checkout is intentionally not an installed package.
    monkeypatch.setattr(aria, "__file__", str(Path(__file__).resolve().parents[1] / "aria/__init__.py"))
    with pytest.raises(release.InstallCheckError, match="imported_from_source"):
        release.probe()
