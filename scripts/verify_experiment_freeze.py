"""Verify a T6 freeze offline; never update seals or execute model requests.

The seal establishes consistency with the recorded manifest, not authorship or
permission to spend. Credentials, runtime data and private CPED material are
outside the accepted paths. E1 reconstruction uses authored public fixtures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


SCHEMA = "aria_t6_freeze_v1"
DEFAULT_MANIFEST = "experiments/t6_20260930/manifest.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_ROOTS = {"aria", "docs", "scripts", "tests", "experiments"}
_PUBLIC_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".html", ".toml"}
_PRIVATE_PARTS = {"data", ".git", ".cache", ".venv", "bench_reports", "node_modules"}
_PRIVATE_NAMES = {"credentials", "secrets", "cookies", "id_rsa", "id_ed25519"}
_WIRE_FIELDS = {"model", "temperature", "max_tokens", "reasoning", "provider", "stream"}


class FreezeError(ValueError):
    """A bounded diagnostic that does not disclose file or request contents."""


def canonical_sha256(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _plain_path(path: Path) -> None:
    """Reject symlinks and Windows reparse points before following a path."""
    for candidate in (*reversed(path.parents), path):
        info = candidate.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise FreezeError("linked_path_rejected")


def _private_name(part: str) -> bool:
    lower = part.lower()
    return (lower.startswith(".env") or lower in _PRIVATE_PARTS
            or lower.split(".", 1)[0] in _PRIVATE_NAMES)


def file_sha256(path) -> str:
    selected = Path(path).absolute()
    if any(_private_name(part) for part in selected.parts):
        raise FreezeError("private_path_rejected")
    _plain_path(selected)
    if not selected.is_file():
        raise FreezeError("regular_file_required")
    with selected.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value)):
        raise FreezeError("unsafe_manifest_path")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or ".." in path.parts
            or len(path.parts) < 2 or path.parts[0] not in _PUBLIC_ROOTS
            or any(_private_name(part) for part in path.parts)
            or "__pycache__" in path.parts
            or path.suffix.lower() not in _PUBLIC_SUFFIXES):
        raise FreezeError("unsafe_manifest_path")
    return path


def _public_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*_relative_path(relative).parts)
    _plain_path(path)
    if not path.is_file():
        raise FreezeError("regular_file_required")
    return path


def _json(path: Path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise FreezeError("json_size_limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FreezeError("duplicate_json_key")
            result[key] = value
        return result

    def invalid(_):
        raise FreezeError("nonfinite_json_number")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=invalid)


def _source_inventory(root: Path) -> set[str]:
    source = root / "aria"
    _plain_path(source)
    if not source.is_dir():
        raise FreezeError("aria_source_directory_required")
    files = set()
    for directory, subdirs, names in os.walk(source, followlinks=False):
        parent = Path(directory)
        for name in subdirs:
            _plain_path(parent / name)
        subdirs[:] = [name for name in subdirs if name != "__pycache__"]
        for name in names:
            if name.endswith(".pyc"):
                continue
            relative = (parent / name).relative_to(root).as_posix()
            _public_file(root, relative)
            files.add(relative)
    return files


def e1_request_records(plan) -> list[dict]:
    """Rebuild hashes of all 204 authored requests without exposing their text."""
    from aria.bench.refocus_cases import load_refocus_cases
    from aria.bench.refocus_e1_runner import E1LocalConfig, inventory, prepare_trial

    preparation, wire = plan["e1"]["preparation"], plan["e1"]["wire"]
    if not isinstance(preparation, dict) or not isinstance(wire, dict) or set(wire) != _WIRE_FIELDS:
        raise FreezeError("invalid_e1_wire_fields")
    config = E1LocalConfig(**preparation)
    provider = wire["provider"]
    if (not isinstance(wire["model"], str) or not wire["model"].strip()
            or wire["model"].startswith("local/")
            or type(wire["temperature"]) not in (int, float)
            or not math.isfinite(wire["temperature"])
            or wire["temperature"] != config.temperature
            or type(wire["max_tokens"]) is not int
            or wire["max_tokens"] != config.max_completion_tokens
            or not isinstance(wire["reasoning"], dict)
            or set(wire["reasoning"]) != {"enabled", "exclude"}
            or wire["reasoning"]["enabled"] is not False
            or wire["reasoning"]["exclude"] is not True
            or wire["stream"] is not False
            or not isinstance(provider, dict)
            or set(provider) != {"only", "allow_fallbacks", "require_parameters", "data_collection"}
            or not isinstance(provider["only"], list) or len(provider["only"]) != 1
            or not isinstance(provider["only"][0], str) or not provider["only"][0].strip()
            or provider["allow_fallbacks"] is not False
            or provider["require_parameters"] is not True
            or provider["data_collection"] != "deny"):
        raise FreezeError("invalid_e1_wire_contract")
    suite = load_refocus_cases()
    cases = {case.id: case for case in suite.main_cases()}
    prepared = {}
    records = []
    for order, row in enumerate(inventory(suite, seed=config.seed), 1):
        key = row["case_id"], row["arm"]
        if key not in prepared:
            payload = deepcopy(prepare_trial(cases[key[0]], key[1], config)["payload"])
            payload.pop("max_completion_tokens")
            payload.update(deepcopy(wire))
            prepared[key] = canonical_sha256(payload)
        records.append({"order": order, **row, "request_sha256": prepared[key]})
    return records


def verify(root, manifest_path=DEFAULT_MANIFEST) -> dict:
    root = Path(root).absolute()
    _plain_path(root)
    raw_manifest_path = Path(manifest_path)
    if raw_manifest_path.is_absolute():
        try:
            relative_manifest = raw_manifest_path.relative_to(root).as_posix()
        except ValueError:
            raise FreezeError("manifest_outside_root") from None
    else:
        relative_manifest = str(manifest_path)
    selected = _public_file(root, relative_manifest)
    manifest = _json(selected)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise FreezeError("unknown_freeze_schema")
    sealed = {key: value for key, value in manifest.items() if key != "seal_sha256"}
    if manifest.get("seal_sha256") != canonical_sha256(sealed):
        raise FreezeError("seal_mismatch")
    plan_name = manifest.get("plan_path")
    plan_path = _public_file(root, plan_name)
    if not plan_name.startswith("experiments/") or plan_path.suffix != ".json":
        raise FreezeError("invalid_plan_path")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise FreezeError("file_inventory_required")
    paths, names = [], set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}
                or type(entry["bytes"]) is not int or entry["bytes"] < 0
                or not isinstance(entry["sha256"], str) or not _SHA.fullmatch(entry["sha256"])):
            raise FreezeError("invalid_file_record")
        path = _public_file(root, entry["path"])
        if entry["path"].casefold() in names or entry["path"] == relative_manifest:
            raise FreezeError("duplicate_or_self_bound_file")
        names.add(entry["path"].casefold())
        paths.append((entry, path))
    registered = {entry["path"] for entry, _ in paths}
    if plan_name not in registered:
        raise FreezeError("plan_not_bound")
    sources = _source_inventory(root)
    if sources != {name for name in registered if name.startswith("aria/")}:
        raise FreezeError("source_inventory_mismatch")
    for entry, path in paths:
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise FreezeError("file_content_mismatch")
    records = manifest.get("e1_requests")
    if (not isinstance(records, list) or len(records) != 204
            or manifest.get("e1_requests_sha256") != canonical_sha256(records)):
        raise FreezeError("request_inventory_digest_mismatch")
    expected = e1_request_records(_json(plan_path))
    if records != expected:
        raise FreezeError("request_inventory_mismatch")
    return {"ok": True, "schema": SCHEMA, "source_files": len(sources),
            "verified_files": len(paths), "e1_requests": len(records),
            "seal_sha256": manifest["seal_sha256"], "network_calls": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    # The reconstructed production modules must come from the reviewed root.
    original_search_path = list(sys.path)
    sys.path.insert(0, str(args.root.absolute()))
    try:
        result = verify(args.root, args.manifest)
    except (FreezeError, OSError, ValueError, TypeError, KeyError) as exc:
        reason = str(exc) if isinstance(exc, FreezeError) else type(exc).__name__
        print(json.dumps({"ok": False, "reason": reason}, ensure_ascii=False))
        return 1
    finally:
        sys.path[:] = original_search_path
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
