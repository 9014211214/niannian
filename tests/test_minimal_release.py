"""Curated archive checks use synthetic files and never read local credentials."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import build_minimal_release as release


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name in release.REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Authored test fixture\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests/test_minimal_runtime.py").write_text("def test_local(): pass\n", encoding="utf-8")
    return root


def test_allowlist_never_reads_credentials_history_runtime_media_or_unlisted_assets(source, tmp_path, monkeypatch):
    omitted = (
        ".env", "data/private.json", ".git/config",
        ".cache/transcript.json", "bench_reports/paid_response.json",
        "aria/assets/voice/zero_shot_prompt.wav", "aria/assets/unreviewed.yaml",
        "docs/unreviewed_notes.md", "tests/__pycache__/private.pyc",
        "docs/media/minimal-stage.png", "docs/media/minimal-local-preview.mp4",
    )
    for name in omitted:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not read or publish")
    original = Path.read_bytes

    def guarded(path):
        if path in {source / name for name in omitted}:
            raise AssertionError("excluded content was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    target = tmp_path / "curated.zip"
    release.build_release(source, target)
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
    assert not names.intersection(omitted)
    assert "aria/runtime.py" in names and "tests/test_minimal_runtime.py" in names
    assert release.GENERATED_FILES <= names


def test_manifest_matches_every_selected_byte_and_build_is_reproducible(source, tmp_path):
    one = tmp_path / "one.zip"
    two = tmp_path / "two.zip"
    first = release.build_release(source, one)
    second = release.build_release(source, two)
    assert one.read_bytes() == two.read_bytes()
    assert first["manifest_sha256"] == second["manifest_sha256"]
    verified = release.verify_release(one)
    assert verified["archive_sha256"] == first["archive_sha256"]
    with zipfile.ZipFile(one) as archive:
        manifest = json.loads(archive.read("RELEASE_MANIFEST.json"))
        assert {row["path"] for row in manifest["files"]} | release.GENERATED_FILES == set(archive.namelist())
        for row in manifest["files"]:
            original = (source / row["path"]).read_bytes()
            assert hashlib.sha256(original).hexdigest() == row["sha256"]
            assert archive.read(row["path"]) == original
        assert str(source) not in archive.read("RELEASE_MANIFEST.json").decode()


@pytest.mark.parametrize("name", ["aria/.env", "aria/local.key", "aria/credentials.py", "tests/run.json"])
def test_dangerous_name_inside_source_is_rejected_before_packing(source, tmp_path, name):
    (source / name).write_text("withheld", encoding="utf-8")
    target = tmp_path / "rejected.zip"
    with pytest.raises(release.ReleaseError, match="dangerous_source_name"):
        release.build_release(source, target)
    assert not target.exists()


@pytest.mark.parametrize("secret", [
    "sk-" + "A7b3" * 12,
    "ghp_" + "Q9w3" * 10,
    "AKIA" + "A1" * 8,
    "-----" + "BEGIN PRIVATE KEY-----\nfixture\n",
    'api_key = "' + "A7b3" * 12 + '"',
    "https://user:" + "A7b3" * 4 + "@example.invalid/",
])
def test_high_confidence_secret_blocks_archive_without_printing_value(source, tmp_path, capsys, secret):
    (source / "aria/runtime.py").write_text(secret, encoding="utf-8")
    target = tmp_path / "rejected.zip"
    assert release.main(["--source", str(source), "--output", str(target)]) == 1
    output = capsys.readouterr().out
    assert "secret_pattern" in output and "aria/runtime.py" in output
    assert secret not in output and "A7b3A7b3" not in output
    assert not target.exists()


def test_existing_output_is_never_overwritten(source, tmp_path):
    target = tmp_path / "existing.zip"
    target.write_bytes(b"owner artifact")
    with pytest.raises(release.ReleaseError, match="output_already_exists"):
        release.build_release(source, target)
    assert target.read_bytes() == b"owner artifact"


def test_explicit_synthetic_password_annotation_is_counted_in_manifest(source, tmp_path):
    text = 'password = "' + "synthetic-local-account-" * 2 + '"  # release-fixture: not-a-deployment-secret\n'
    (source / "tests/test_minimal_runtime.py").write_text(text, encoding="utf-8")
    target = tmp_path / "synthetic.zip"
    result = release.build_release(source, target)
    assert result["synthetic_password_fixture_exemptions"] == 1
    assert release.verify_release(target)["synthetic_password_fixture_exemptions"] == 1
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("RELEASE_MANIFEST.json"))
        assert manifest["synthetic_password_fixture_exemptions"] == [{
            "path": "tests/test_minimal_runtime.py", "line": 1,
            "code": "explicit_synthetic_test_password",
        }]


@pytest.mark.parametrize("name,assignment", [
    ("aria/runtime.py", 'password = "' + "synthetic-local-account-" * 2 + '"'),
    ("tests/test_minimal_runtime.py", 'api_key = "' + "synthetic-local-account-" * 2 + '"'),
    ("tests/test_minimal_runtime.py", 'password = "sk-' + "A7b3" * 12 + '"'),
])
def test_fixture_annotation_cannot_exempt_runtime_passwords_keys_or_provider_tokens(source, tmp_path, name, assignment):
    (source / name).write_text(assignment + '  # release-fixture: not-a-deployment-secret\n', encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="secret_pattern"):
        release.build_release(source, tmp_path / "rejected.zip")


@pytest.mark.parametrize("relative", ["aria/output.zip", "tests/output.zip", "docs/output.zip", ".git/output.zip"])
def test_output_cannot_pollute_source_or_git(source, relative):
    with pytest.raises(release.ReleaseError, match="output_inside_source_rejected"):
        release.build_release(source, source / relative)
    assert not (source / relative).exists()


def test_source_mutation_after_scan_blocks_publication(source, tmp_path, monkeypatch):
    original = release._check_unchanged

    def changed(root, contents):
        (root / "aria/runtime.py").write_text("# changed after scan\n", encoding="utf-8")
        original(root, contents)

    monkeypatch.setattr(release, "_check_unchanged", changed)
    with pytest.raises(release.ReleaseError, match="source_changed_during_build"):
        release.build_release(source, tmp_path / "changed.zip")
    assert not (tmp_path / "changed.zip").exists()
    assert not list(tmp_path.glob(".aria-release-*.tmp"))


def test_publication_failure_leaves_no_partial_zip(source, tmp_path, monkeypatch):
    def fail_link(*args, **kwargs):
        raise OSError("fixture atomic publication failure")

    monkeypatch.setattr(release.os, "link", fail_link)
    with pytest.raises(OSError):
        release.build_release(source, tmp_path / "failed.zip")
    assert not (tmp_path / "failed.zip").exists()
    assert not list(tmp_path.glob(".aria-release-*.tmp"))


def test_verify_detects_changed_archive_member(source, tmp_path):
    original = tmp_path / "original.zip"
    release.build_release(source, original)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(damaged, "w") as new:
        for name in old.namelist():
            new.writestr(name, b"tampered" if name == "aria/runtime.py" else old.read(name))
    with pytest.raises(release.ReleaseError, match="archive_digest_mismatch"):
        release.verify_release(damaged)


@pytest.mark.parametrize("member", ["../escape.py", "aria/.env.py", "aria/../escape.py", "aria/secret.key",
                                    "aria//escape.py", "./aria/escape.py"])
def test_verify_rejects_unsafe_members_before_reading_them(source, tmp_path, member):
    target = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("RELEASE_MANIFEST.json", "{}")
        archive.writestr("RELEASE_SCOPE.md", release.SCOPE_TEXT)
        archive.writestr(member, "private")
    with pytest.raises(release.ReleaseError, match="archive_member_rejected"):
        release.verify_release(target)


def test_linked_source_is_rejected_without_reading_target(source, tmp_path, monkeypatch):
    target = source / "aria/runtime.py"
    original = release._unsafe_link
    monkeypatch.setattr(release, "_unsafe_link", lambda path: path == target or original(path))
    with pytest.raises(release.ReleaseError, match="source_link_rejected"):
        release.build_release(source, tmp_path / "link.zip")


def test_check_mode_creates_no_output(source, capsys):
    before = {str(path) for path in source.rglob("*")}
    assert release.main(["--source", str(source), "--check"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] and output["files"] == len(release.REQUIRED_FILES) + 1
    assert output["scope"] == "selected_current_source_only_not_git_history_or_runtime_data"
    assert {str(path) for path in source.rglob("*")} == before


@pytest.mark.parametrize("name,header", [
    ("minimal-stage.png", b"\x89PNG\r\n\x1a\n"),
    ("minimal-local-preview.mp4", b"\x00\x00\x00\x20ftypisom"),
])
def test_previous_named_media_is_excluded_without_reading(source, tmp_path, name, header, monkeypatch):
    media = source / "docs/media"
    media.mkdir(parents=True)
    (media / name).write_bytes(header + b"\x00local synthetic bytes")
    (media / "unreviewed.mp4").write_bytes(b"must stay excluded")
    original = Path.read_bytes
    def guarded(path):
        if path.parent == media:
            raise AssertionError("excluded media must not be read")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    target = tmp_path / "with-media.zip"
    release.build_release(source, target)
    assert release.verify_release(target)["files"] == len(release.REQUIRED_FILES) + 1
    with zipfile.ZipFile(target) as archive:
        assert "docs/media/" + name not in archive.namelist()
        assert "docs/media/unreviewed.mp4" not in archive.namelist()


@pytest.mark.parametrize("content,code", [
    (b"not an image", "authored_media_header_invalid"),
    (b"\x89PNG\r\n\x1a\n" + b"sk-" + b"A7b3" * 12, "secret_pattern"),
])
def test_legacy_media_does_not_exempt_invalid_format_or_secrets(source, content, code):
    with pytest.raises(release.ReleaseError, match=code):
        release._scan("docs/media/minimal-stage.png", content, release.LEGACY_POLICY_VERSION)


def test_legacy_media_archive_remains_verifiable_with_original_scope(source, tmp_path):
    target = tmp_path / "legacy.zip"
    _, contents, manifest = release.inspect_source(source)
    contents["docs/media/minimal-stage.png"] = b"\x89PNG\r\n\x1a\n\x00legacy fixture"
    manifest["policy_version"] = release.LEGACY_POLICY_VERSION
    manifest["files"] = [{"path": name, "bytes": len(data), "sha256": release._digest(data)}
                         for name, data in sorted(contents.items())]
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.LEGACY_SCOPE_TEXT)
    verified = release.verify_release(target)
    assert verified["policy_version"] == release.LEGACY_POLICY_VERSION
    assert verified["legacy_media_included"] is True


def test_new_policy_cannot_relabel_legacy_media_as_current(source, tmp_path):
    target = tmp_path / "wrong-policy.zip"
    _, contents, manifest = release.inspect_source(source)
    data = b"\x89PNG\r\n\x1a\nlegacy"
    manifest["files"].append({"path": "docs/media/minimal-stage.png", "bytes": len(data),
                              "sha256": release._digest(data)})
    with zipfile.ZipFile(target, "w") as packed:
        for name, content in contents.items():
            packed.writestr(name, content)
        packed.writestr("docs/media/minimal-stage.png", data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_malformed_manifest_reports_safe_error(source, tmp_path, capsys):
    original = tmp_path / "original.zip"
    release.build_release(source, original)
    damaged = tmp_path / "malformed.zip"
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(damaged, "w") as new:
        for name in old.namelist():
            data = old.read(name)
            if name == "RELEASE_MANIFEST.json":
                manifest = json.loads(data)
                del manifest["files"][0]["sha256"]
                data = json.dumps(manifest).encode()
            new.writestr(name, data)
    assert release.main(["--verify", str(damaged)]) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "archive_manifest_invalid"}


def test_v4_explicit_preparation_assets_and_upstream_data_exclusion(source, tmp_path, monkeypatch):
    for name in release.V4_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Authored preparation fixture\n", encoding="utf-8")
    excluded = (
        "artifacts/local/prefeval-source/benchmark_dataset/explicit_preference/example.json",
        "artifacts/local/prefeval-source/error_type/check_violation.txt",
        "artifacts/local/prefeval-run/generated.json", "error_type/check_violation.txt",
        "benchmark_dataset/explicit_preference/example.json", "prefeval-source/source-manifest.json",
        "aria/bench/cases/prefeval_source.json", "tests/fixtures/prefeval_responses.json",
    )
    for name in excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("upstream source must stay private", encoding="utf-8")
    original = Path.read_bytes

    def guarded(path):
        if path in {source / name for name in excluded}:
            raise AssertionError("upstream data was read for release")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    archive = tmp_path / "v4-authored.zip"
    release.build_release(source, archive)
    with zipfile.ZipFile(archive) as packed:
        names = set(packed.namelist())
    assert release.V4_POLICY_FILES <= names
    assert not set(excluded) & names
    assert release.verify_release(archive)["policy_version"] == release.POLICY_VERSION


def test_previous_v3_archive_keeps_original_scope_and_policy(source, tmp_path):
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.PREVIOUS_POLICY_VERSION
    target = tmp_path / "retained-v3.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.PREVIOUS_SCOPE_TEXT)
    before = target.read_bytes()
    result = release.verify_release(target)
    assert result["policy_version"] == "aria-minimal-source-3"
    assert not result["legacy_media_included"]
    assert target.read_bytes() == before


@pytest.mark.parametrize("policy", [release.LEGACY_POLICY_VERSION, release.PREVIOUS_POLICY_VERSION])
def test_v4_files_cannot_be_smuggled_into_an_older_policy(source, tmp_path, policy):
    _, contents, manifest = release.inspect_source(source)
    added = "aria/bench/cases/e1b_v1_draft.yaml"
    contents[added] = b"# authored synthetic fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrongly-labelled.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.PREVIOUS_SCOPE_TEXT if policy == release.PREVIOUS_POLICY_VERSION else release.LEGACY_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v5_execution_tools_included_but_old_v4_scope_stays_verifiable(source, tmp_path):
    for name in release.V5_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Authored E1b execution fixture\n", encoding="utf-8")
    current = tmp_path / "current.zip"
    release.build_release(source, current)
    with zipfile.ZipFile(current) as packed:
        assert release.V5_POLICY_FILES <= set(packed.namelist())
    assert release.verify_release(current)["policy_version"] == release.POLICY_VERSION
    _, contents, manifest = release.inspect_source(source)
    contents = {name: data for name, data in contents.items() if name not in release.V5_POLICY_FILES}
    manifest["files"] = [row for row in manifest["files"] if row["path"] in contents]
    manifest["policy_version"] = release.V4_POLICY_VERSION
    old = tmp_path / "old-v4.zip"
    with zipfile.ZipFile(old, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V4_SCOPE_TEXT)
    before = old.read_bytes()
    assert release.verify_release(old)["policy_version"] == "aria-minimal-source-4"
    assert old.read_bytes() == before
    assert all(not release._allowed(name, release.V4_POLICY_VERSION) for name in release.V5_POLICY_FILES)


def test_e1br_authored_tools_include_no_private_campaign_or_ledger(source, tmp_path, monkeypatch):
    for name in release.V6_POLICY_FILES | release.V7_POLICY_FILES | release.V8_POLICY_FILES | release.V9_POLICY_FILES | release.V10_POLICY_FILES | release.V11_POLICY_FILES | release.V12_POLICY_FILES | release.V13_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Authored E1b-R fixture\n", encoding="utf-8")
    excluded = (
        "artifacts/local/e1br-formal-v1/campaign.json",
        "artifacts/local/e1br-formal-v1/ledger.sqlite3",
        "artifacts/local/tasks/20261004-07/user-approval.json",
        "artifacts/local/tasks/20261004-07/checkpoint-originals.manifest.json",
        "experiments/e1br-formal-v1/raw_response.json",
        "experiments/e1br-formal-v1/fees.json",
        "scripts/unreviewed_e1br_tool.py", "docs/unreviewed_e1br_notes.md",
        "artifacts/local/e1br-formal-v4/visible-first.json",
        "artifacts/local/e1c-formal-v1/first-submission.json",
        "scripts/unreviewed_e1c_tool.py", "docs/unreviewed_e1c_questions.md",
        "artifacts/local/e1br-formal-v6/main/first.json",
        "artifacts/local/tasks/20261005-03/v5-close/v5-closed-originals-v1.manifest.json",
        "scripts/unreviewed_prewrite_tool.py",
    )
    for name in excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private content must not be read")
    original = Path.read_bytes

    def guarded(path):
        if path in {source / name for name in excluded}:
            raise AssertionError("private E1b-R content was read for release")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    target = tmp_path / "e1br-current.zip"
    release.build_release(source, target)
    with zipfile.ZipFile(target) as packed:
        assert release.V6_POLICY_FILES | release.V7_POLICY_FILES | release.V8_POLICY_FILES | release.V9_POLICY_FILES | release.V10_POLICY_FILES | release.V11_POLICY_FILES | release.V12_POLICY_FILES | release.V13_POLICY_FILES <= set(packed.namelist())
        assert not set(excluded) & set(packed.namelist())
    assert release.verify_release(target)["policy_version"] == release.POLICY_VERSION


def test_v5_archive_retains_original_scope_without_new_e1br_files(source, tmp_path):
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V5_POLICY_VERSION
    target = tmp_path / "retained-v5.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V5_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-5"
    assert target.read_bytes() == before


def test_v6_archive_keeps_original_scope_and_excludes_checkpoint_tools(source, tmp_path):
    for name in release.V6_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Authored v6 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V6_POLICY_VERSION
    target = tmp_path / "retained-v6.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V6_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-6"
    assert target.read_bytes() == before


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION, release.V11_POLICY_VERSION, release.V10_POLICY_VERSION, release.V9_POLICY_VERSION, release.V8_POLICY_VERSION, release.V7_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V7_POLICY_FILES))
def test_checkpoint_files_cannot_enter_older_archive_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Authored checkpoint fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-checkpoint-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V6_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


@pytest.mark.parametrize("policy", [release.LEGACY_POLICY_VERSION, release.PREVIOUS_POLICY_VERSION,
                                    release.V4_POLICY_VERSION, release.V5_POLICY_VERSION])
def test_e1br_files_cannot_enter_old_archive_policy(source, tmp_path, policy):
    _, contents, manifest = release.inspect_source(source)
    added = "scripts/run_e1br.py"
    contents[added] = b"# authored E1b-R fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V5_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v7_archive_keeps_exact_scope_without_format_tools(source, tmp_path):
    for name in release.V7_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Software-only authored v7 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V7_POLICY_VERSION
    target = tmp_path / "retained-v7.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V7_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-7"
    assert target.read_bytes() == before


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION, release.V11_POLICY_VERSION, release.V10_POLICY_VERSION, release.V9_POLICY_VERSION, release.V8_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V8_POLICY_FILES))
def test_format_and_e1c_contract_files_cannot_enter_older_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Software-only authored format/E1c fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-format-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V7_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v8_archive_keeps_exact_scope_without_e1c_execution_tools(source, tmp_path):
    for name in release.V8_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Software-only authored v8 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V8_POLICY_VERSION
    target = tmp_path / "retained-v8.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V8_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-8"
    assert target.read_bytes() == before
    assert release.V9_POLICY_FILES == {
        "scripts/e1c_execution.py", "scripts/e1c_calls.py", "scripts/e1c_qualification.py",
        "scripts/e1c_scoring.py", "scripts/run_e1c.py", "docs/E1C_PAID_EXECUTION.md",
    }


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION, release.V11_POLICY_VERSION, release.V10_POLICY_VERSION, release.V9_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V9_POLICY_FILES))
def test_e1c_execution_tools_cannot_enter_older_archive_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Disclosed software-only E1c fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-e1c-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V8_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v9_archive_keeps_exact_scope_without_memory_continuation(source, tmp_path):
    for name in release.V9_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Disclosed software-only v9 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V9_POLICY_VERSION
    target = tmp_path / "retained-v9.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V9_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-9"
    assert target.read_bytes() == before
    assert release.V10_POLICY_FILES == {
        "scripts/run_e1br_memory_continuation.py", "scripts/e1br_memory_checkpoint.py",
        "scripts/e1br_memory_runtime.py", "scripts/e1br_memory_legacy.py",
        "scripts/e1br_memory_qualification.py", "docs/E1BR_MEMORY_CONTINUATION.md",
    }


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION, release.V11_POLICY_VERSION, release.V10_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V10_POLICY_FILES))
def test_memory_continuation_cannot_enter_older_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Disclosed software-only memory-continuation fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-memory-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V9_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v10_archive_keeps_exact_scope_without_prewrite_tools(source, tmp_path):
    for name in release.V10_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Disclosed software-only v10 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V10_POLICY_VERSION
    target = tmp_path / "retained-v10.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V10_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-10"
    assert target.read_bytes() == before
    assert release.V11_POLICY_FILES == {
        "scripts/e1br_prewrite_checkpoint.py", "scripts/e1br_prewrite_runtime.py",
        "scripts/run_e1br_prewrite_continuation.py", "docs/E1BR_PREWRITE_CONTINUATION.md",
    }


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION, release.V11_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V11_POLICY_FILES))
def test_prewrite_continuation_cannot_enter_older_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Disclosed software-only prewrite-continuation fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-prewrite-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V10_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v11_archive_keeps_original_scope_without_later_result_report(source, tmp_path):
    for name in release.V11_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Software-only v11 fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V11_POLICY_VERSION
    target = tmp_path / "retained-v11.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V11_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-11"
    assert target.read_bytes() == before
    assert release.V12_POLICY_FILES == {"docs/E1BR_RESULTS_2026_10_05.md"}


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION, release.V12_POLICY_VERSION}))
def test_later_result_report_cannot_enter_older_policy(source, tmp_path, policy):
    _, contents, manifest = release.inspect_source(source)
    added = "docs/E1BR_RESULTS_2026_10_05.md"
    contents[added] = b"# Aggregate result fixture\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]), "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-result-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V11_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)


def test_v12_archive_keeps_exact_scope_without_design_repair_tools(source, tmp_path):
    for name in release.V12_POLICY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic aggregate result fixture\n", encoding="utf-8")
    _, contents, manifest = release.inspect_source(source)
    manifest["policy_version"] = release.V12_POLICY_VERSION
    target = tmp_path / "retained-v12.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V12_SCOPE_TEXT)
    before = target.read_bytes()
    assert release.verify_release(target)["policy_version"] == "aria-minimal-source-12"
    assert target.read_bytes() == before
    assert release.V13_POLICY_FILES == {
        "scripts/e1c_v2_execution.py", "scripts/e1c_v2_calls.py",
        "scripts/e1c_v2_qualification.py", "scripts/e1c_v2_scoring.py",
        "scripts/run_e1c_v2.py", "docs/E1C_V2_EXECUTION.md",
        "docs/DESIGN_AUDIT.md", "docs/DESIGN_REPAIR_2026_10_06.md",
    }


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V13_POLICY_VERSION}))
@pytest.mark.parametrize("added", sorted(release.V13_POLICY_FILES))
def test_design_repairs_cannot_enter_older_archive_policy(source, tmp_path, policy, added):
    _, contents, manifest = release.inspect_source(source)
    contents[added] = b"# Authored repair tool fixture, no experiment inputs\n"
    manifest["policy_version"] = policy
    manifest["files"].append({"path": added, "bytes": len(contents[added]),
                              "sha256": release._digest(contents[added])})
    target = tmp_path / "wrong-old-repair-policy.zip"
    with zipfile.ZipFile(target, "w") as packed:
        for name, data in contents.items():
            packed.writestr(name, data)
        packed.writestr("RELEASE_MANIFEST.json", release._manifest_bytes(manifest))
        packed.writestr("RELEASE_SCOPE.md", release.V12_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_policy_member_rejected"):
        release.verify_release(target)
