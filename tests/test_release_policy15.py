"""Public document scope and archive compatibility are mechanical local checks."""
from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from scripts import build_minimal_release as release


DOCUMENTS = frozenset({
    "docs/H2_CHAIN_DIAGNOSTIC_2026_10_08.md", "docs/SAFETY_AUTH_DECISION_2026_10_08.md",
    "docs/SAFETY_EVIDENCE_CANDIDATE_2026_10_08.md", "docs/SAFETY_KEYWORD_ISOLATION_2026_10_08.md",
    "docs/PRODUCT_IMPLEMENTATION_REVIEW_2026_10_08.md", "docs/DEVELOPMENT_HANDOFF_2026_10_08.md",
    "docs/CORE_BOUNDARY_REPAIR_2026_10_08.md", "docs/LOCAL_COMPLETION_2026_10_09.md",
})


def fixture_root(tmp_path, extras=()):
    root = tmp_path / "synthetic-source"
    for name in release._required_files(release.POLICY_VERSION) | set(extras):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# authored software fixture only\n", encoding="utf-8")
    return root


def rewrite_archive(source, target, policy, scope, extra=None):
    with zipfile.ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["RELEASE_MANIFEST.json"])
    manifest["policy_version"] = policy
    if policy != release.POLICY_VERSION:
        contents = {n: raw for n, raw in contents.items() if n not in release.V17_POLICY_FILES}
    if policy not in {release.V16_POLICY_VERSION, release.POLICY_VERSION}:
        contents = {n: raw for n, raw in contents.items() if n not in release.V16_REQUIRED_FILES | release.V17_POLICY_FILES}
        manifest["files"] = [row for row in manifest["files"] if row["path"] in contents]
    if extra:
        name, raw = extra
        contents[name] = raw
        manifest["files"].append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    contents["RELEASE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    contents["RELEASE_SCOPE.md"] = scope.encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)


def test_v15_additions_are_exactly_named_public_documents():
    assert release.V15_POLICY_VERSION == "aria-minimal-source-15"
    assert release.V15_POLICY_FILES == DOCUMENTS
    assert all(release._allowed(name) for name in DOCUMENTS)
    assert all(not release._allowed(name, policy) for name in DOCUMENTS
               for policy in release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V16_POLICY_VERSION, release.V15_POLICY_VERSION})
    for name in ("artifacts/local/tasks/report.md", "artifacts/local/sealed.json", ".env",
                 "data/private.json", "docs/private_answers.md", "docs/unlisted_current_report.md"):
        assert not release._allowed(name)


def test_original_v14_scope_bytes_are_preserved():
    assert release.V14_POLICY_VERSION == "aria-minimal-source-14"
    assert hashlib.sha256(release.V14_SCOPE_TEXT.encode()).hexdigest() == (
        "d4b939962729d33b4dc36d54980c1329d9e9e8c5704bf8326af371bba062ff08"
    )


@pytest.mark.parametrize("policy", sorted(release.SUPPORTED_POLICIES - {release.POLICY_VERSION, release.V16_POLICY_VERSION, release.V15_POLICY_VERSION}))
def test_every_older_policy_keeps_v15_documents_out(policy):
    assert all(not release._allowed(name, policy) for name in DOCUMENTS)


@pytest.mark.parametrize("name", sorted(release.V14_POLICY_FILES))
def test_v14_documents_remain_allowed_by_14_15_and_current(name):
    assert release._allowed(name, release.V14_POLICY_VERSION)
    assert release._allowed(name, release.V15_POLICY_VERSION)
    assert release._allowed(name, release.POLICY_VERSION)
    assert all(not release._allowed(name, policy) for policy in release.SUPPORTED_POLICIES
               - {release.V14_POLICY_VERSION, release.V15_POLICY_VERSION, release.V16_POLICY_VERSION, release.POLICY_VERSION})


def test_current_build_retains_v15_named_docs_without_private_outputs(tmp_path):
    root = fixture_root(tmp_path, DOCUMENTS | {"artifacts/local/tasks/private.md"})
    archive = tmp_path / "fixture-current.zip"
    release.build_release(root, archive)
    verified = release.verify_release(archive)
    assert verified["policy_version"] == release.POLICY_VERSION
    with zipfile.ZipFile(archive) as saved:
        assert DOCUMENTS <= set(saved.namelist())
        assert "artifacts/local/tasks/private.md" not in saved.namelist()


def test_v14_original_archive_contract_verifies_with_current_reader(tmp_path):
    root = fixture_root(tmp_path, release.V14_POLICY_FILES)
    current = tmp_path / "synthetic-current.zip"
    release.build_release(root, current)
    original = tmp_path / "synthetic-v14.zip"
    rewrite_archive(current, original, release.V14_POLICY_VERSION, release.V14_SCOPE_TEXT)
    assert release.verify_release(original)["policy_version"] == release.V14_POLICY_VERSION
    illegal = tmp_path / "illegal-v14.zip"
    rewrite_archive(original, illegal, release.V14_POLICY_VERSION, release.V14_SCOPE_TEXT,
                    extra=(next(iter(DOCUMENTS)), b"# newer document\n"))
    with pytest.raises(release.ReleaseError):
        release.verify_release(illegal)


def test_v14_archive_cannot_be_relabelled_with_v15_scope(tmp_path):
    root = fixture_root(tmp_path)
    current = tmp_path / "fixture.zip"
    release.build_release(root, current)
    illegal = tmp_path / "scope-mismatch.zip"
    rewrite_archive(current, illegal, release.V14_POLICY_VERSION, release.V15_SCOPE_TEXT)
    with pytest.raises(release.ReleaseError, match="archive_scope_changed"):
        release.verify_release(illegal)
