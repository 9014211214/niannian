"""Release audit tests use synthetic source/local subprocesses, no scanner/network."""
import json
from pathlib import Path

import pytest

from scripts import audit_release as audit
from scripts import build_minimal_release as release


@pytest.fixture
def source_zip(tmp_path):
    source = tmp_path / "source"
    for name in release._required_files(release.POLICY_VERSION):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n", encoding="utf-8")
    (source / "README.md").write_text("Contact synthetic@example.invalid\n", encoding="utf-8")
    target = tmp_path / "source.zip"
    release.build_release(source, target)
    return target


def empty_detector(directory):
    assert (directory / "RELEASE_MANIFEST.json").exists()
    assert not (directory / ".git").exists()
    return {"version": audit.SCANNER_VERSION, "results": {}, "plugins_used": [], "filters_used": []}


def test_only_verified_zip_is_scanned_and_no_match_is_reported(source_zip):
    result = audit.audit(source_zip, detector=empty_detector)
    assert result["status"] == "review_required"
    finding = next(row for row in result["findings"] if row["kind"] == "email")
    assert finding["path"] == "README.md" and finding["line"] == 1
    assert "synthetic@example.invalid" not in json.dumps(result)
    assert result["scanner"]["network_verification"] is False


def test_review_binds_exact_finding_and_stale_reviews_fail(source_zip):
    first = audit.audit(source_zip, detector=empty_detector)
    reviews = [{"id": row["id"], "category": "synthetic_fixture", "reason": "Authored test contact."}
               for row in first["findings"]]
    assert audit.audit(source_zip, reviews=reviews, detector=empty_detector)["status"] == "pass"
    reviews.append({"id": "a" * 64, "category": "synthetic_fixture", "reason": "Old review."})
    assert audit.audit(source_zip, reviews=reviews, detector=empty_detector)["stale_review_count"] == 1
    assert audit.audit(source_zip, reviews=reviews, detector=empty_detector)["status"] == "review_required"


def test_new_private_value_does_not_inherit_old_review():
    before = audit.privacy_findings({"README.md": "synthetic@example.invalid"})[0]
    after = audit.privacy_findings({"README.md": "another@example.invalid"})[0]
    assert before["id"] != after["id"]


def test_scanner_output_cannot_name_file_outside_verified_tree(source_zip):
    def bad_detector(directory):
        return {"version": audit.SCANNER_VERSION, "results": {"../outside": [{"type": "Secret", "line_number": 1}]}}
    with pytest.raises(audit.AuditError, match="scanner_path_outside_release"):
        audit.audit(source_zip, detector=bad_detector)


@pytest.mark.parametrize("content,kind", [
    ("C:\\Users\\synthetic-user\\project", "local_user_path"),
    ("/home/synthetic-user/project", "posix_user_path"),
    ("13800000000", "phone"),
    ("12345678901", "numeric_account_identifier"),
    ('username = "synthetic-user"', "account_assignment"),
    ("synthetic-user", "local_username"),
])
def test_privacy_patterns_detect_categories_without_values(content, kind):
    rows = audit.privacy_findings({"README.md": content}, username="synthetic-user")
    assert any(row["kind"] == kind for row in rows)
    assert content not in json.dumps(rows)


def test_secret_baseline_is_reduced_to_location_not_hash_or_value(source_zip):
    def scanner(directory):
        return {"version": audit.SCANNER_VERSION, "results": {
            "README.md": [{"type": "Secret Keyword", "line_number": 1, "hashed_secret": "private-hash",
                           "secret_value": "private-value"}]}}
    result = audit.audit(source_zip, detector=scanner)
    assert "private-hash" not in json.dumps(result) and "private-value" not in json.dumps(result)


def test_only_recomputed_generated_manifest_hashes_receive_automatic_explanations(source_zip):
    def scanner(directory):
        lines = (directory / "RELEASE_MANIFEST.json").read_text(encoding="utf-8").splitlines()
        return {"version": audit.SCANNER_VERSION, "results": {
            "RELEASE_MANIFEST.json": [{"type": "Hex High Entropy String", "line_number": number}
                                      for number, line in enumerate(lines, 1) if '"sha256"' in line],
            "README.md": [{"type": "Hex High Entropy String", "line_number": 1}],
        }}
    result = audit.audit(source_zip, detector=scanner)
    manifest_hits = [row for row in result["findings"] if row["path"] == "RELEASE_MANIFEST.json"]
    assert len(manifest_hits) == len(release._required_files(release.POLICY_VERSION))
    assert all(row["review"]["category"] == "public_integrity_digest" for row in manifest_hits)
    assert all(row["review"] is None for row in result["findings"] if row["path"] == "README.md")


def test_subprocess_disables_online_verification(monkeypatch, tmp_path):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: audit.SCANNER_VERSION)
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        class Result:
            returncode = 0
            stdout = json.dumps({"version": audit.SCANNER_VERSION, "results": {}}).encode()
        return Result()
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit.run_detector(tmp_path)
    assert "--no-verify" in calls[0][0] and "--all-files" in calls[0][0]
    assert calls[0][1]["capture_output"] is True


def test_detector_failure_does_not_print_stderr(monkeypatch, tmp_path):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: audit.SCANNER_VERSION)
    class Result:
        returncode = 1
        stderr = b"private scanner text"
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(audit.AuditError, match="^detect_secrets_failed$"):
        audit.run_detector(tmp_path)


@pytest.mark.parametrize("parent_utf8", [None, "0"])
def test_detector_child_reads_nonascii_utf8_when_parent_mode_is_unset_or_disabled(monkeypatch, tmp_path, parent_utf8):
    import hashlib
    import importlib.metadata
    import sys

    if parent_utf8 is None:
        monkeypatch.delenv("PYTHONUTF8", raising=False)
    else:
        monkeypatch.setenv("PYTHONUTF8", parent_utf8)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: audit.SCANNER_VERSION)
    text = "# 中文陪伴🙂\napi_key = 'software-fixture-only'\n"
    fixture = tmp_path / "fixture-中文.py"
    fixture.write_text(text, encoding="utf-8")
    real_run = audit.subprocess.run
    observed = []

    def local_reader(command, **kwargs):
        assert "--all-files" in command and "--no-verify" in command
        observed.append(kwargs["env"]["PYTHONUTF8"])
        # The actual child uses implicit open(), as detect-secrets 1.5.0 does.
        # Its UTF-8 mode must be forced even when the parent explicitly uses 0.
        code = (
            "import hashlib,json,sys; "
            "text=open(sys.argv[1]).read(); "
            "print(json.dumps({'version':'1.5.0','results':{},"
            "'utf8_mode':sys.flags.utf8_mode,"
            "'text_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest()}))"
        )
        return real_run([sys.executable, "-c", code, str(fixture)], **kwargs)

    monkeypatch.setattr(audit.subprocess, "run", local_reader)
    result = audit.run_detector(tmp_path)
    assert observed == ["1"] and result["utf8_mode"] == 1
    assert result["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
