"""Build a curated local source ZIP without reading credentials or runtime data.

This does not publish, inspect Git history, download dependencies, or claim that
the result is free of every possible secret or private datum. The manifest binds
the bytes checked by this policy; it is an integrity record, not a signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile


POLICY_VERSION = "aria-minimal-source-11"
V10_POLICY_VERSION = "aria-minimal-source-10"
V9_POLICY_VERSION = "aria-minimal-source-9"
V8_POLICY_VERSION = "aria-minimal-source-8"
V7_POLICY_VERSION = "aria-minimal-source-7"
V6_POLICY_VERSION = "aria-minimal-source-6"
V5_POLICY_VERSION = "aria-minimal-source-5"
V4_POLICY_VERSION = "aria-minimal-source-4"
PREVIOUS_POLICY_VERSION = "aria-minimal-source-3"
LEGACY_POLICY_VERSION = "aria-minimal-source-2"
SUPPORTED_POLICIES = frozenset({POLICY_VERSION, V10_POLICY_VERSION, V9_POLICY_VERSION, V8_POLICY_VERSION, V7_POLICY_VERSION, V6_POLICY_VERSION, V5_POLICY_VERSION, V4_POLICY_VERSION, PREVIOUS_POLICY_VERSION, LEGACY_POLICY_VERSION})
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
LEGACY_MEDIA_FILES = frozenset({"docs/media/minimal-stage.png", "docs/media/minimal-local-preview.mp4"})
NEW_POLICY_FILES = frozenset({
    "scripts/check_release_install.py",
    "scripts/codex_judge_local.py", "docs/E1_CODEX_JUDGE.md",
    "scripts/run_e1_json_schema_mode.py", "scripts/e1_json_schema_mode.py",
    "docs/E1_JSON_SCHEMA_MODE.md", "experiments/t77_20261001/amendment.json",
    "experiments/t77_20261001/requests.json", "experiments/t77_20261001/execution.json",
    "experiments/t77_20261001/outcome.aggregate.json",
    "scripts/run_e1_json_mode.py", "scripts/e1_json_mode.py", "scripts/t76_transport.py",
    "docs/E1_JSON_MODE.md", "experiments/t76_20261001/amendment.json",
    "experiments/t76_20261001/requests.json", "experiments/t76_20261001/execution.json",
    "scripts/run_e1_carrier.py", "scripts/e1_json_carrier.py", "docs/E1_JSON_CARRIER.md",
    "experiments/t75_20261001/amendment.json", "experiments/t75_20261001/execution.json",
    "scripts/run_e1_authorized.py", "experiments/t74_20261001/authorized_run.json",
    "experiments/t74_20261001/authorized_execution.json",
    "scripts/e1_compat.py", "scripts/run_e1_compat.py", "scripts/t74_transport.py",
    "docs/E1_CALL_COMPATIBILITY.md", "experiments/t74_20261001/amendment.json",
    "experiments/t74_20261001/requests.json", "experiments/t74_20261001/execution.json",
    "scripts/check_e1_tool_choice.py",
    "scripts/run_e1_t73.py", "experiments/t73_20261001/authorized_run.json",
    "experiments/t73_20261001/authorized_execution.json",
    "scripts/check_e1_t73.py", "scripts/t73_policy.py", "scripts/t73_execution.py",
    "docs/E1_EXECUTION_T73.md", "experiments/t73_20261001/amendment.json",
    "experiments/t73_20261001/requests.json", "experiments/t73_20261001/execution.json",
    "scripts/check_e1_t72.py", "scripts/t72_policy.py", "scripts/t72_transport.py",
    "docs/E1_EXECUTION_T72.md", "experiments/t72_20260930/amendment.json",
    "experiments/t72_20260930/execution.json",
    "scripts/run_e1_t71.py", "scripts/t71_policy.py", "scripts/audit_release.py",
    "docs/E1_EXECUTION_T71.md", "docs/STATED_PREFERENCE.md", "docs/RELEASE_SECURITY_AUDIT.md",
    "experiments/t71_20260930/amendment.json", "experiments/t71_20260930/execution.json",
    "experiments/t71_20260930/preflight.json", "experiments/t71_20260930/outcome.aggregate.json",
    "experiments/t71_20260930/validation.t6.md",
})
V4_POLICY_FILES = frozenset({
    "scripts/prepare_model_policy.py", "scripts/e1b_rehearsal.py", "scripts/e1b_evaluation.py",
    "scripts/prefeval_prepare.py", "scripts/ctest_rehearsal.py", "scripts/run_ctest.py",
    "scripts/run_ctest_repair.py", "docs/CTEST_REPAIR.md",
    "docs/COGNITIVE_CHANGE.md", "docs/CTEST_EXECUTION.md",
    "aria/bench/cases/e1b_v1_draft.yaml",
    "tests/fixtures/external_emotion_no_input_prompts.json",
})
V5_POLICY_FILES = frozenset({
    "scripts/run_e1b.py", "scripts/e1b_formation.py", "scripts/e1b_secondary.py", "scripts/e1b_assessment.py",
    "scripts/run_e1b_repair.py",
    "scripts/run_e1b_reflection_repair.py",
    "scripts/run_e1b_structure_repair.py",
    "scripts/run_e1b_pacing_repair.py",
    "scripts/run_e1b_observed_repair.py",
    "scripts/run_e1b_bounded_repair.py",
    "scripts/run_e1b_composed_repair.py", "scripts/e1b_composed_evidence.py",
    "scripts/run_e1b_main_continuation.py", "scripts/e1b_main_continuation.py",
    "docs/E1B_EXECUTION.md",
})
V6_POLICY_FILES = frozenset({
    "scripts/run_e1br.py", "scripts/e1br_qualification.py", "scripts/e1br_assessment.py",
    "docs/E1BR_EXECUTION.md",
    "scripts/run_e1br_received_continuation.py", "scripts/e1br_received_body.py",
    "docs/E1BR_RECEIVED_CONTINUATION.md",
})
V7_POLICY_FILES = frozenset({
    "scripts/e1br_checkpoint.py", "scripts/run_e1br_checkpoint_continuation.py",
    "scripts/e1br_checkpoint_guard.py", "docs/E1BR_CHECKPOINT_CONTINUATION.md",
    "scripts/ci_tests.py",
})
V8_POLICY_FILES = frozenset({
    "scripts/run_e1br_format_continuation.py", "scripts/e1br_format_checkpoint.py",
    "scripts/e1br_format_runtime.py", "scripts/e1br_format_legacy.py",
    "scripts/e1br_format_qualification.py", "docs/E1BR_FORMAT_CONTINUATION.md",
    "docs/E1C_EXECUTION.md",
})
V9_POLICY_FILES = frozenset({
    "scripts/e1c_execution.py", "scripts/e1c_calls.py", "scripts/e1c_qualification.py",
    "scripts/e1c_scoring.py", "scripts/run_e1c.py", "docs/E1C_PAID_EXECUTION.md",
})
V10_POLICY_FILES = frozenset({
    "scripts/run_e1br_memory_continuation.py", "scripts/e1br_memory_checkpoint.py",
    "scripts/e1br_memory_runtime.py", "scripts/e1br_memory_legacy.py",
    "scripts/e1br_memory_qualification.py", "docs/E1BR_MEMORY_CONTINUATION.md",
})
V11_POLICY_FILES = frozenset({
    "scripts/e1br_prewrite_checkpoint.py", "scripts/e1br_prewrite_runtime.py",
    "scripts/run_e1br_prewrite_continuation.py", "docs/E1BR_PREWRITE_CONTINUATION.md",
})
FIXED_FILES = frozenset({
    "README.md", "README.zh.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example",
    "SECURITY.md", "CONTRIBUTING.md", "pyproject.toml", "MANIFEST.in", "AGENTS.md",
    "docs/PROGRESS.md", "docs/PRODUCT.md", "docs/EVIDENCE.md", "docs/VALIDATION.md",
    "docs/MINIMAL_DEMO_LOCAL.md", "docs/MINIMAL_RELEASE_CHECKLIST.md",
    "docs/CPED_HOLDOUT_PROTOCOL.md", "docs/CPED_SOURCE_PREPARATION.md",
    "docs/CLEANUP.md", "docs/EXPERIMENT_FREEZE.md", "LICENSES/Apache-2.0.txt",
    "scripts/build_minimal_release.py", "scripts/verify_experiment_freeze.py",
    "experiments/t6_20260930/plan.json", "experiments/t6_20260930/manifest.json",
    "experiments/t6_20260930/judge_contract.json", "experiments/t6_20260930/public_metadata.json",
    "experiments/t6_20260930/e2_preparation.aggregate.json",
    "scripts/run_e1_experiment.py", "scripts/t7_ledger.py", "scripts/t7_transport.py",
    "scripts/t7_judge.py", "docs/E1_EXECUTION.md",
    "experiments/t7_20260930/execution.json", "experiments/t7_20260930/preflight.json",
    "examples/prepare_cped_holdout.py", ".github/workflows/test.yml",
    "aria/assets/strategy_matrix.yaml", "aria/assets/personas/laochen_demo.yaml",
    "aria/scenes/companion_elder/elder_kb.yaml",
    "aria/assets/voice/minimal.html", "aria/assets/voice/care.html",
    "aria/assets/voice/README.md", "tests/fixtures/followthrough_synthetic.json",
}) | NEW_POLICY_FILES | V4_POLICY_FILES | V5_POLICY_FILES | V6_POLICY_FILES | V7_POLICY_FILES | V8_POLICY_FILES | V9_POLICY_FILES | V10_POLICY_FILES | V11_POLICY_FILES
REQUIRED_FILES = frozenset({
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
    "aria/cli.py", "aria/runtime.py", "aria/voice/minimal.py",
    "aria/assets/voice/minimal.html", "aria/assets/voice/care.html", "aria/assets/strategy_matrix.yaml",
    "aria/scenes/companion_elder/elder_kb.yaml",
})
# Explicitly authored case assets/manifests, not downloaded data or run outputs.
CASE_FILES = frozenset({"refocus_e1_v1"})
DATASET_MANIFESTS = frozenset({"cped_v1"})
SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".cache", ".pytest_cache", ".ruff_cache", ".venv",
    "node_modules", "data", "bench_reports", "dist", "build",
})
DANGEROUS_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".keystore", ".sqlite", ".sqlite3", ".db",
    ".log", ".jsonl", ".csv", ".tsv", ".wav", ".mp3", ".mp4", ".webm",
})
DANGEROUS_NAMES = frozenset({
    "id_rsa", "id_ed25519", "credentials.py", "credentials.json", "credentials.yaml",
    "secrets.py", "secrets.json", "secrets.yaml", "run.json", "cookies.json",
})
SECRET_PATTERNS = (
    ("private_key", re.compile(rb"-----" + rb"BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")),
    ("provider_token", re.compile(rb"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{32,}\b")),
    ("github_token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{32,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("slack_token", re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{24,}\b")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("credential_assignment", re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)"
        rb"[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_+/=-]{24,})[\"']")),
    ("url_credentials", re.compile(rb"https?://[^\s/:]+:[^\s/@]{8,}@")),
)
GENERATED_FILES = {"RELEASE_MANIFEST.json", "RELEASE_SCOPE.md"}
LEGACY_SCOPE_TEXT = """# Curated local source archive

This archive was created locally; no publication or model execution is implied.
RELEASE_MANIFEST.json records the checked file bytes and policy version. It is
not a cryptographic signature or proof of safety, licence compliance, or model
quality. Verify it with `python scripts/build_minimal_release.py --verify FILE`.

Included: first-party Python source/tests, explicitly listed HTML/YAML assets,
selected entry documentation, and licence notices. Author-written synthetic
benchmark fixtures are not real-person evidence. The original Git repository,
history, research outputs, downloads, runtime data, credentials, arbitrary media,
and optional legacy voice sample are deliberately not included. Only the exact
named author-created minimal preview PNG/MP4 may be present. Their headers and
bytes are checked, but visible content needs human privacy review; no model
effect or realtime latency claim follows from the recording. The active repository has been reduced to the first-release scope. Historical
research and pre-cleanup source remain in a separate private recovery archive,
which is not distributed. See docs/CLEANUP.md for the scope boundary.

The dependency environment is not bundled or locked by this archive. Install
from pyproject.toml; offline installation requires your own compatible local
dependency cache. Run the focused minimal tests named in MINIMAL_DEMO_LOCAL.md.
All tests remaining in the active repository are included; deleted-feature tests
remain only in the private pre-cleanup snapshot. New pass counts are version-specific.

Content scanning covers only these selected current files. Git history, excluded
files, and all forms of personal information have NOT been comprehensively
audited. A passing token scan is not release approval. See
docs/MINIMAL_RELEASE_CHECKLIST.md and docs/PROGRESS.md for remaining work.
"""
PREVIOUS_SCOPE_TEXT = LEGACY_SCOPE_TEXT.replace(
    "Only the exact\nnamed author-created minimal preview PNG/MP4 may be present. Their headers and\n"
    "bytes are checked, but visible content needs human privacy review; no model\n"
    "effect or realtime latency claim follows from the recording. ",
    "No preview PNG/MP4 or other media is included by policy version 3. "
    "Prior archives retain their original version 2 scope and media; verifying "
    "an old archive does not upgrade its release eligibility. ",
)
V4_SCOPE_TEXT = PREVIOUS_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 3. ",
    "No preview PNG/MP4 or other media is included by policy version 4. "
    "Version 4 adds explicitly listed authored E1b/C-test/E3 preparation scripts, "
    "cognitive-change documentation and an external-emotion regression fixture. "
    "PrefEval upstream data, upstream judge templates and all local experiment "
    "outputs remain excluded. Prior version 3 archives retain their original scope. ",
)
V5_SCOPE_TEXT = V4_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 4. ",
    "No preview PNG/MP4 or other media is included by policy version 5. "
    "Version 5 additionally includes the authored E1b execution entry, "
    "formation-label tool and secondary-judge helper. Their inclusion grants "
    "no model spending permission and makes no experiment result claim. "
    "Prior version 4 archives retain their original scope. ",
)
V6_SCOPE_TEXT = V5_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 5. ",
    "No preview PNG/MP4 or other media is included by policy version 6. "
    "Version 6 adds only the explicitly named authored E1b-R entry, "
    "qualification, assessment and received-body continuation tools, and their execution contracts. Private "
    "campaigns, model responses, approvals and spending ledgers remain excluded. "
    "Prior version 5 archives retain their original scope. ",
)
V7_SCOPE_TEXT = V6_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 6. ",
    "No preview PNG/MP4 or other media is included by policy version 7. "
    "Version 7 adds the explicitly named authored checkpoint reconstruction "
    "and continuation entry/guard tools plus their contract, and the CI test "
    "partition/coverage verifier. Retained responses, "
    "private manifests and campaign ledgers remain excluded. Tool inclusion "
    "does not authorize paid execution. Prior version 6 archives retain their original scope. ",
)
V8_SCOPE_TEXT = V7_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 7. ",
    "No preview PNG/MP4 or other media is included by policy version 8. "
    "Version 8 adds five explicitly named format-continuation tools and their "
    "contract, plus the E1c engineering contract. E1c first-party Python "
    "modules and software fixtures use the existing source/test rules. "
    "Private first answers, retained failure evidence, spending records and "
    "formal holdout questions remain excluded. Inclusion does not qualify "
    "mixed-version results or authorize paid execution. Prior version 7 "
    "archives retain their original scope. ",
)
V9_SCOPE_TEXT = V8_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 8. ",
    "No preview PNG/MP4 or other media is included by policy version 9. "
    "Version 9 adds five explicitly named E1c execution, guarded calls, "
    "qualification, scoring and command-line tools plus their execution contract. "
    "Private authored holdout submissions, model answers, approval files and "
    "spending records remain excluded. Software fixtures do not demonstrate "
    "model effects; tool inclusion does not authorize any paid execution. "
    "Prior version 8 archives retain their original scope. ",
)

V10_SCOPE_TEXT = V9_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 9. ",
    "No preview PNG/MP4 or other media is included by policy version 10. "
    "Version 10 adds five explicitly named memory-continuation tools "
    "and their contract. Private first answers, failure evidence, approvals "
    "and spending records remain excluded. Inclusion does not establish "
    "model reliability or authorize paid execution. Prior version 9 "
    "archives retain their original scope. ",
)


SCOPE_TEXT = V10_SCOPE_TEXT.replace(
    "No preview PNG/MP4 or other media is included by policy version 10. ",
    "No preview PNG/MP4 or other media is included by policy version 11. "
    "Version 11 adds three explicitly named prewrite-continuation tools "
    "and their contract. Private retained answers, failure evidence, approvals "
    "and spending records remain excluded. Inclusion does not grant model "
    "spending permission or establish real-world reliability. Prior version 10 "
    "archives retain their original scope. ",
)


class ReleaseError(ValueError):
    """Only a safe code/path; never include a matched secret or source line."""


def _unsafe_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _allowed(relative: str, policy: str = POLICY_VERSION) -> bool:
    if policy not in SUPPORTED_POLICIES:
        return False
    value = PurePosixPath(relative)
    if (not value.parts or value.is_absolute() or ".." in value.parts or "\\" in relative
            or relative != value.as_posix() or ":" in relative
            or any(part in SKIP_DIRS for part in value.parts)):
        return False
    if relative in V4_POLICY_FILES and policy not in {V4_POLICY_VERSION, V5_POLICY_VERSION, V6_POLICY_VERSION, V7_POLICY_VERSION, V8_POLICY_VERSION, V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V5_POLICY_FILES and policy not in {V5_POLICY_VERSION, V6_POLICY_VERSION, V7_POLICY_VERSION, V8_POLICY_VERSION, V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V6_POLICY_FILES and policy not in {V6_POLICY_VERSION, V7_POLICY_VERSION, V8_POLICY_VERSION, V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V7_POLICY_FILES and policy not in {V7_POLICY_VERSION, V8_POLICY_VERSION, V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V8_POLICY_FILES and policy not in {V8_POLICY_VERSION, V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V9_POLICY_FILES and policy not in {V9_POLICY_VERSION, V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V10_POLICY_FILES and policy not in {V10_POLICY_VERSION, POLICY_VERSION}:
        return False
    if relative in V11_POLICY_FILES and policy != POLICY_VERSION:
        return False
    if relative in FIXED_FILES and not (policy == LEGACY_POLICY_VERSION and relative in NEW_POLICY_FILES):
        return True
    if policy == LEGACY_POLICY_VERSION and relative in LEGACY_MEDIA_FILES:
        return True
    if value.parts[0] == "aria" and value.suffix == ".py":
        return True
    if value.parent.as_posix() == "tests" and value.suffix == ".py":
        return True
    if value.parent.as_posix() == "aria/bench/cases" and value.suffix == ".yaml":
        return value.stem in CASE_FILES
    if value.parent.as_posix() == "aria/bench/datasets" and value.suffix == ".yaml":
        return value.stem in DATASET_MANIFESTS
    return False


def _dangerous_name(relative: str, policy: str = POLICY_VERSION) -> bool:
    if ((policy == LEGACY_POLICY_VERSION and relative in LEGACY_MEDIA_FILES)
            or relative == ".env.example"):
        return False
    path = PurePosixPath(relative.lower())
    return (any(part == ".env" or part.startswith(".env.") for part in path.parts)
            or path.name in DANGEROUS_NAMES or path.suffix in DANGEROUS_SUFFIXES)


def _root(value: str | Path) -> Path:
    path = Path(value).absolute()
    if any(_unsafe_link(part) for part in (path, *path.parents)):
        raise ReleaseError("source_root_link_rejected")
    path = path.resolve()
    if not path.is_dir():
        raise ReleaseError("source_root_missing")
    return path


def collect_files(root: Path) -> list[Path]:
    """Only enumerate code roots/fixed public files; never open .env or data."""
    selected = set()
    for relative in FIXED_FILES:
        path = root / relative
        if path.exists() or path.is_symlink():
            selected.add(path)
    for code_root in (root / "aria", root / "tests"):
        if not code_root.exists():
            continue
        for directory, dirs, names in os.walk(code_root, followlinks=False):
            current = Path(directory)
            for name in list(dirs):
                child = current / name
                if _unsafe_link(child):
                    raise ReleaseError("source_link_rejected:" + child.relative_to(root).as_posix())
                if name in SKIP_DIRS:
                    dirs.remove(name)
            for name in names:
                path = current / name
                relative = path.relative_to(root).as_posix()
                # Optional legacy media is deliberately omitted, never read.
                if relative == "aria/assets/voice/zero_shot_prompt.wav":
                    continue
                if _dangerous_name(relative):
                    raise ReleaseError("dangerous_source_name:" + relative)
                if _allowed(relative):
                    selected.add(path)
    for path in selected:
        relative = path.relative_to(root).as_posix()
        if any(_unsafe_link(part) for part in (path, *path.parents) if part != root.parent):
            raise ReleaseError("source_link_rejected:" + relative)
        if not path.is_file() or not _allowed(relative) or _dangerous_name(relative):
            raise ReleaseError("source_file_rejected:" + relative)
    present = {path.relative_to(root).as_posix() for path in selected}
    missing = REQUIRED_FILES - present
    if missing:
        raise ReleaseError("required_source_missing:" + sorted(missing)[0])
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _scan(relative: str, content: bytes, policy: str = POLICY_VERSION) -> list[dict]:
    if len(content) > MAX_FILE_BYTES:
        raise ReleaseError("source_file_too_large:" + relative)
    if policy == LEGACY_POLICY_VERSION and relative in LEGACY_MEDIA_FILES:
        valid = (content.startswith(b"\x89PNG\r\n\x1a\n") if relative.endswith(".png")
                 else len(content) >= 12 and content[4:8] == b"ftyp")
        if not valid:
            raise ReleaseError("authored_media_header_invalid:" + relative)
    else:
        if b"\x00" in content:
            raise ReleaseError("unexpected_binary_source:" + relative)
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ReleaseError("source_not_utf8:" + relative) from None
    exemptions = []
    for code, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            line_start = content.rfind(b"\n", 0, match.start()) + 1
            line_end = content.find(b"\n", match.end())
            if line_end == -1:
                line_end = len(content)
            test_file = PurePosixPath(relative)
            if (code == "credential_assignment"
                    and test_file.parent.as_posix() == "tests" and test_file.suffix == ".py"
                    and re.match(rb"(?i)(?:password|passwd)\b", match.group(0))
                    and b"# release-fixture: not-a-deployment-secret" in content[match.end():line_end]):
                exemptions.append({"path": relative, "line": content[:line_start].count(b"\n") + 1,
                                   "code": "explicit_synthetic_test_password"})
                continue
            raise ReleaseError("secret_pattern:" + code + ":" + relative)
    return exemptions


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def inspect_source(source_root: str | Path) -> tuple[Path, dict[str, bytes], dict]:
    root = _root(source_root)
    contents = {}
    exemptions = []
    for path in collect_files(root):
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ReleaseError("source_file_too_large:" + relative)
        data = path.read_bytes()
        exemptions.extend(_scan(relative, data))
        contents[relative] = data
    if sum(map(len, contents.values())) > MAX_TOTAL_BYTES:
        raise ReleaseError("source_total_too_large")
    manifest = {
        "policy_version": POLICY_VERSION,
        "scope": "selected_current_source_only_not_git_history_or_runtime_data",
        "model_requests": 0,
        "synthetic_password_fixture_exemptions": exemptions,
        "files": [{"path": name, "sha256": _digest(data), "bytes": len(data)}
                  for name, data in contents.items()],
    }
    return root, contents, manifest


def _manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _check_unchanged(root: Path, contents: dict[str, bytes]) -> None:
    current = collect_files(root)
    if {path.relative_to(root).as_posix() for path in current} != set(contents):
        raise ReleaseError("source_changed_during_build")
    for path in current:
        if path.read_bytes() != contents[path.relative_to(root).as_posix()]:
            raise ReleaseError("source_changed_during_build")


def _output_path(value: str | Path, root: Path) -> Path:
    path = Path(value).absolute()
    if path.suffix.lower() != ".zip":
        raise ReleaseError("output_must_be_zip")
    if any(_unsafe_link(part) for part in (path, *path.parents)):
        raise ReleaseError("output_link_rejected")
    path = path.resolve()
    if path.exists():
        raise ReleaseError("output_already_exists")
    if path.is_relative_to(root):
        relative = path.relative_to(root)
        if relative.parts[0] not in {"dist", ".cache"}:
            raise ReleaseError("output_inside_source_rejected")
    return path


def build_release(source_root: str | Path, output: str | Path) -> dict:
    root, contents, manifest = inspect_source(source_root)
    destination = _output_path(output, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = _manifest_bytes(manifest)
    entries = {**contents, "RELEASE_MANIFEST.json": manifest_data,
               "RELEASE_SCOPE.md": SCOPE_TEXT.encode("utf-8")}
    descriptor, temporary = tempfile.mkstemp(prefix=".aria-release-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    staging = Path(temporary)
    try:
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, content)
        _check_unchanged(root, contents)
        # Same-directory hard-link publication is atomic and cannot overwrite
        # a destination created concurrently. An unsupported filesystem fails.
        _output_path(destination, root)
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return {"archive": str(destination), "files": len(contents),
            "archive_sha256": _digest(destination.read_bytes()),
            "manifest_sha256": _digest(manifest_data), "policy_version": POLICY_VERSION,
            "synthetic_password_fixture_exemptions": len(manifest["synthetic_password_fixture_exemptions"])}


def verify_release(value: str | Path) -> dict:
    path = Path(value)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or not GENERATED_FILES <= set(names):
            raise ReleaseError("archive_member_set_invalid")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES + MAX_FILE_BYTES:
            raise ReleaseError("archive_total_too_large")
        for info in infos:
            if (info.file_size > MAX_FILE_BYTES or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or (info.filename not in GENERATED_FILES and (
                        not any(_allowed(info.filename, policy) for policy in SUPPORTED_POLICIES)
                        or _dangerous_name(info.filename, LEGACY_POLICY_VERSION)))):
                raise ReleaseError("archive_member_rejected")
        manifest_data = archive.read("RELEASE_MANIFEST.json")
        try:
            manifest = json.loads(manifest_data)
            records = manifest["files"]
            policy = manifest["policy_version"]
            if (policy not in SUPPORTED_POLICIES or not isinstance(records, list)
                    or any(not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}
                           or not isinstance(row["path"], str) or type(row["bytes"]) is not int
                           or row["bytes"] < 0 or not isinstance(row["sha256"], str)
                           or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                           for row in records)
                    or len(records) != len({row["path"] for row in records})
                    or not REQUIRED_FILES <= {row["path"] for row in records}
                    or {row["path"] for row in records} | GENERATED_FILES != set(names)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ReleaseError("archive_manifest_invalid") from None
        if any(not _allowed(row["path"], policy) or _dangerous_name(row["path"], policy)
               for row in records):
            raise ReleaseError("archive_policy_member_rejected")
        scope = {LEGACY_POLICY_VERSION: LEGACY_SCOPE_TEXT,
                 PREVIOUS_POLICY_VERSION: PREVIOUS_SCOPE_TEXT,
                 V4_POLICY_VERSION: V4_SCOPE_TEXT,
                 V5_POLICY_VERSION: V5_SCOPE_TEXT,
                 V6_POLICY_VERSION: V6_SCOPE_TEXT,
                 V7_POLICY_VERSION: V7_SCOPE_TEXT,
                 V8_POLICY_VERSION: V8_SCOPE_TEXT,
                 V9_POLICY_VERSION: V9_SCOPE_TEXT,
                 V10_POLICY_VERSION: V10_SCOPE_TEXT,
                 POLICY_VERSION: SCOPE_TEXT}[policy]
        if archive.read("RELEASE_SCOPE.md") != scope.encode("utf-8"):
            raise ReleaseError("archive_scope_changed")
        exemptions = []
        for row in records:
            data = archive.read(row["path"])
            exemptions.extend(_scan(row["path"], data, policy))
            if len(data) != row["bytes"] or _digest(data) != row["sha256"]:
                raise ReleaseError("archive_digest_mismatch:" + row["path"])
        if exemptions != manifest.get("synthetic_password_fixture_exemptions"):
            raise ReleaseError("archive_fixture_audit_mismatch")
    return {"files": len(records), "archive_sha256": _digest(path.read_bytes()),
            "manifest_sha256": _digest(manifest_data), "policy_version": policy,
            "legacy_media_included": any(row["path"] in LEGACY_MEDIA_FILES for row in records),
            "synthetic_password_fixture_exemptions": len(exemptions)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check/build a curated local source ZIP; never publish or call models.")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check", action="store_true", help="scan selected source without creating an archive")
    choice.add_argument("--output", type=Path, help="new .zip path; an existing file is never overwritten")
    choice.add_argument("--verify", type=Path, help="verify a previously built source ZIP")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            result = verify_release(args.verify)
        elif args.output:
            result = build_release(args.source, args.output)
        else:
            _root_path, _contents, manifest = inspect_source(args.source)
            result = {"files": len(manifest["files"]), "policy_version": POLICY_VERSION,
                      "scope": manifest["scope"], "manifest_sha256": _digest(_manifest_bytes(manifest)),
                      "synthetic_password_fixture_exemptions": len(manifest["synthetic_password_fixture_exemptions"])}
    except (ReleaseError, OSError, zipfile.BadZipFile) as error:
        # OS messages could contain a private absolute path. Only policy errors
        # expose a selected relative path; never print source text or matches.
        code = str(error) if isinstance(error, ReleaseError) else type(error).__name__
        print(json.dumps({"ok": False, "error": code}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
