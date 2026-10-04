"""Small, secret-safe reproducibility fingerprints for public benchmarks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


_SAFE_LLM_FIELDS = (
    "source_id",
    "role",
    "model",
    "temperature",
    "reasoning_effort",
    "max_completion_tokens",
    "timeout",
    "max_retries",
    "fallback_models",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return type(value).__name__


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitized_llm_config(llm: Any) -> dict[str, Any]:
    """Capture useful model settings without keys, tokens, or endpoint URLs."""
    if llm is None:
        return {"client_type": "none"}
    config: dict[str, Any] = {"client_type": type(llm).__name__}
    for field in _SAFE_LLM_FIELDS:
        if hasattr(llm, field):
            config[field] = _jsonable(getattr(llm, field))
    return config


def _file_manifest(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    manifest = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            manifest.append({"name": path.name, "sha256": file_fingerprint(path)})
    return sorted(manifest, key=lambda item: (item["name"], item["sha256"]))


def _git_metadata(start: Path) -> tuple[str | None, bool | None]:
    try:
        root = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        ).stdout.strip())
        return commit or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def build_reproducibility(
    llm: Any,
    *,
    case_files: Iterable[str | Path] = (),
    prompt_files: Iterable[str | Path] = (),
    case_material: Any = None,
    config_material: Any = None,
) -> dict[str, Any]:
    """Build provenance metadata for a benchmark report.

    File names and hashes are public; absolute local paths are deliberately
    omitted. Configuration is restricted to a small allowlist so credentials
    can never enter a report through an arbitrary client attribute.
    """
    cases = {"files": _file_manifest(case_files), "selection": case_material}
    prompts = {"files": _file_manifest(prompt_files)}
    config = {
        "llm": sanitized_llm_config(llm),
        "benchmark": _jsonable(config_material),
    }
    commit, dirty = _git_metadata(Path(__file__).resolve().parent)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "case_fingerprint": fingerprint(cases),
        "prompt_fingerprint": fingerprint(prompts),
        "config": config,
        "config_fingerprint": fingerprint(config),
    }


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def invalid_constant(_):
        raise ValueError("non-finite JSON number")
    return json.loads(text, object_pairs_hook=_strict_object,
                      parse_constant=invalid_constant)
