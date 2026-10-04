"""Audit a verified source ZIP; report locations/categories, never matched values.

Run with a separate environment containing detect-secrets==1.5.0. Scanning is
local with verification/network probes disabled. This audits selected bytes,
not Git history, runtime data, or model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_minimal_release as release


SCANNER_VERSION = "1.5.0"
REVIEW_CATEGORIES = frozenset({
    "synthetic_fixture", "public_upstream_reference", "authorized_public_identity",
    "public_integrity_digest", "generic_documentation_example", "detector_configuration",
    "fictional_experiment_identifier",
})
PATTERNS = {
    "local_user_path": re.compile(r"(?i)\b[a-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>]+"),
    "posix_user_path": re.compile(r"/(?:home|Users)/[^/\s\"'<>]+"),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"),
    "phone": re.compile(r"(?<![\w.])(?:\+?86[ -]?)?1[3-9]\d{9}(?![\w.])"),
    "numeric_account_identifier": re.compile(r"(?<![\w.])\d{9,20}(?![\w.])"),
    "account_assignment": re.compile(r"(?i)\b(?:account_id|username|user_name)\s*[=:]\s*[\"'][^\"'\r\n]+[\"']"),
}


class AuditError(ValueError):
    """Codes only; source strings and subprocess stderr must never be printed."""


def _identifier(path: str, line: int, kind: str, source_line: str) -> str:
    payload = json.dumps([path, line, kind, source_line], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finding(path: str, line: int, kind: str, lines: list[str]) -> dict:
    if not (1 <= line <= len(lines)):
        raise AuditError("scanner_location_invalid")
    return {"id": _identifier(path, line, kind, lines[line - 1]),
            "path": path, "line": line, "kind": kind}


def privacy_findings(contents: dict[str, str], username: str | None = None) -> list[dict]:
    patterns = dict(PATTERNS)
    if username and len(username) >= 3:
        patterns["local_username"] = re.compile(r"(?<![\w])" + re.escape(username) + r"(?![\w])", re.I)
    found = []
    for path, content in sorted(contents.items()):
        lines = content.splitlines()
        for number, line in enumerate(lines, 1):
            for kind, pattern in patterns.items():
                if pattern.search(line):
                    found.append(_finding(path, number, kind, lines))
    return found


def run_detector(directory: Path) -> dict:
    try:
        import importlib.metadata
        version = importlib.metadata.version("detect-secrets")
    except importlib.metadata.PackageNotFoundError:
        raise AuditError("install_detect_secrets_in_isolated_environment") from None
    if version != SCANNER_VERSION:
        raise AuditError("scanner_version_not_pinned")
    completed = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", "--all-files", "--no-verify", "."],
        cwd=directory, capture_output=True, timeout=180, check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if completed.returncode:
        raise AuditError("detect_secrets_failed")
    try:
        result = json.loads(completed.stdout)
        if result["version"] != SCANNER_VERSION or not isinstance(result["results"], dict):
            raise ValueError
        return result
    except (ValueError, KeyError, TypeError):
        raise AuditError("detect_secrets_output_invalid") from None


def load_reviews(path: Path | None) -> list[dict]:
    if path is None:
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value["reviews"]
        if (value["schema"] != "aria_release_review_v1" or not isinstance(rows, list)
                or any(set(row) != {"id", "category", "reason"}
                       or not re.fullmatch(r"[0-9a-f]{64}", row["id"])
                       or row["category"] not in REVIEW_CATEGORIES
                       or not isinstance(row["reason"], str) or not row["reason"].strip()
                       for row in rows)
                or len({row["id"] for row in rows}) != len(rows)):
            raise ValueError
        return rows
    except (OSError, ValueError, TypeError, KeyError):
        raise AuditError("review_file_invalid") from None


def audit(archive: Path, *, reviews: list[dict] = (), detector=run_detector,
          username: str | None = None) -> dict:
    verified = release.verify_release(archive)
    if verified["policy_version"] != release.POLICY_VERSION:
        raise AuditError("legacy_archive_requires_new_build")
    contents = {}
    with tempfile.TemporaryDirectory(prefix="aria-release-audit-") as scratch:
        directory = Path(scratch)
        with zipfile.ZipFile(archive) as packed:
            for name in packed.namelist():
                raw = packed.read(name)
                try:
                    contents[name] = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    raise AuditError("audit_requires_text_release") from None
                destination = directory / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        baseline = detector(directory)
    findings = privacy_findings(contents, username)
    for name, matches in baseline["results"].items():
        relative = name.replace("\\", "/").removeprefix("./")
        if relative not in contents:
            raise AuditError("scanner_path_outside_release")
        for match in matches:
            kind = "detect_secrets:" + match["type"]
            findings.append(_finding(relative, match["line_number"], kind,
                                     contents[relative].splitlines()))
    unique = {row["id"]: row for row in findings}
    review_map = {row["id"]: row for row in reviews}
    unknown_reviews = sorted(set(review_map) - set(unique))
    for key, row in unique.items():
        review = review_map.get(key)
        row["review"] = ({"category": review["category"], "reason": review["reason"]}
                         if review else None)
        if (not review and row["path"] == "RELEASE_MANIFEST.json"
                and row["kind"] == "detect_secrets:Hex High Entropy String"
                and re.fullmatch(r'\s*"sha256": "[a-f0-9]{64}",?\s*',
                                 contents[row["path"]].splitlines()[row["line"] - 1])):
            # verify_release has recomputed every listed file digest. This is
            # narrowly limited to its generated manifest, not arbitrary fields.
            row["review"] = {"category": "public_integrity_digest",
                             "reason": "Generated manifest SHA-256 recomputed against archived file bytes."}
    findings = sorted(unique.values(), key=lambda row: (row["path"], row["line"], row["kind"]))
    unresolved = sum(row["review"] is None for row in findings)
    return {
        "schema": "aria_release_audit_v1", "archive_sha256": verified["archive_sha256"],
        "manifest_sha256": verified["manifest_sha256"], "policy_version": verified["policy_version"],
        "scope": "verified_source_zip_including_generated_manifest_excluding_git_history_runtime_and_media",
        "scanner": {"name": "detect-secrets", "version": baseline["version"],
                    "network_verification": False, "plugins": baseline.get("plugins_used", []),
                    "filters": baseline.get("filters_used", [])},
        "privacy_rules": sorted(set(PATTERNS) | ({"local_username"} if username else set())),
        "files": len(contents), "findings": findings, "unresolved_count": unresolved,
        "stale_review_count": len(unknown_reviews),
        "media_files": [], "status": "pass" if not unresolved and not unknown_reviews else "review_required",
        "limitations": ["Not a Git history audit or exhaustive PII detector.",
                        "A reviewed synthetic credential is never deployment configuration.",
                        "Public identity authorization is scoped to the recorded source line.",
                        "Final release bytes must be scanned again after later changes."],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        username = os.environ.get("USERNAME") or Path.home().name
        result = audit(args.archive, reviews=load_reviews(args.reviews), username=username)
        with args.output.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(json.dumps({key: result[key] for key in (
            "status", "archive_sha256", "files", "unresolved_count", "stale_review_count")}, ensure_ascii=False))
        return 0 if result["status"] == "pass" else 2
    except (AuditError, release.ReleaseError, OSError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        code = str(error) if isinstance(error, (AuditError, release.ReleaseError)) else type(error).__name__
        print(json.dumps({"status": "error", "error": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
