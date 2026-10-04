"""Local, append-once intake for a separate Codex blind review.

This module reads only the blind manifest/messages and its own output/draft
directories. It performs no inference, network access, credential access or
private experiment mapping lookup. A completed receipt is a local validation
record, never a model-provider receipt or proof of human calibration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SCHEMA = "aria_codex_judge_blind_manifest_v1"
LABELS = ("decides_for_person", "violates_agreement", "leaks_private",
          "lectures_when_quiet")
MAX_BYTES = 2 * 1024 * 1024
OPAQUE = re.compile(r"q_[0-9a-f]{12}\Z")


class IntakeError(ValueError):
    """Fixed, non-sensitive validation failure."""


def _fail(code):
    raise IntakeError(code)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _constant(_):
    _fail("nonfinite_json_value")


def _json(data):
    try:
        text = data.decode("utf-8", errors="strict")
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        # Reject lone surrogates, including those supplied through JSON escapes.
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return result
    except IntakeError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError, OverflowError):
        _fail("invalid_json")


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _no_links(path):
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            _fail("linked_path_forbidden")


def _root(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        _fail("root_must_be_absolute")
    _no_links(path)
    if not path.is_dir():
        _fail("root_not_directory")
    return path.resolve(strict=True)


def _within(root, path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        _fail("path_outside_allowed_directory")
    _no_links(path)
    return path


def _read(root, path):
    path = _within(root, path)
    if not path.is_file():
        _fail("required_file_missing")
    with path.open("rb") as stream:
        value = stream.read(MAX_BYTES + 1)
    if len(value) > MAX_BYTES:
        _fail("file_too_large")
    return value


def _write(root, path, data):
    path = _within(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _within(root, path)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _fail("already_written")


def _message_spec(data, width, count):
    messages = _json(data)
    if (type(messages) is not list or len(messages) != 2 or any(
            type(item) is not dict or set(item) != {"role", "content"}
            or type(item["content"]) is not str for item in messages)
            or [item["role"] for item in messages] != ["system", "user"]):
        _fail("invalid_blind_messages")
    payload = _json(messages[1]["content"].encode("utf-8"))
    keys = ({"comparison", "not_required", "items"} if width == 2 else
            {"context", "confirmed_guidance", "criteria", "items"})
    if type(payload) is not dict or set(payload) != keys:
        _fail("invalid_blind_payload")
    items = payload["items"]
    if type(items) is not list or len(items) != count:
        _fail("invalid_blind_item_count")
    applicable = None
    if width == 5:
        criteria = payload["criteria"]
        if (type(criteria) is not dict or set(criteria) != set(LABELS)
                or any(type(criteria[key]) is not dict
                       or type(criteria[key].get("applicable")) is not bool
                       for key in LABELS)):
            _fail("invalid_applicability")
        applicable = [criteria[key]["applicable"] for key in LABELS]
    expected = {}
    for item in items:
        if width == 5:
            if (type(item) is not list or len(item) != 2
                    or type(item[1]) is not str):
                _fail("invalid_blind_item")
            identity = item[0]
        else:
            if (type(item) is not dict or set(item) != {"opaque_id", "people"}
                    or type(item["people"]) is not list or len(item["people"]) != 2
                    or any(type(person) is not dict or set(person) != {
                        "confirmed_guidance", "context", "criteria", "response"}
                           for person in item["people"])):
                _fail("invalid_blind_pair")
            identity = item["opaque_id"]
        if type(identity) is not str or not OPAQUE.fullmatch(identity) or identity in expected:
            _fail("invalid_blind_id_set")
        expected[identity] = applicable
    return expected


def load_manifest(root):
    """Check every one of the fixed 25 message files before any operation."""
    root = _root(root)
    manifest_bytes = _read(root, root / "blind" / "manifest.json")
    manifest = _json(manifest_bytes)
    if (type(manifest) is not dict or
            manifest.get("schema", manifest.get("schema_version")) != SCHEMA
            or type(manifest.get("batches")) is not list
            or len(manifest["batches"]) != 25):
        _fail("invalid_blind_manifest")
    batches = {}
    all_ids = set()
    for index, item in enumerate(manifest["batches"], 1):
        name = f"batch_{index:03d}"
        if (type(item) is not dict or item.get("file") != name + ".json"
                or type(item.get("sha256")) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or type(item.get("items")) is not int or not 1 <= item["items"] <= 15
                or type(item.get("width")) is not int or item["width"] not in (2, 5)):
            _fail("invalid_blind_manifest_batch")
        data = _read(root, root / "blind" / item["file"])
        if _sha(data) != item["sha256"]:
            _fail("blind_input_hash_mismatch")
        expected = _message_spec(data, item["width"], item["items"])
        if all_ids.intersection(expected):
            _fail("blind_id_collision_across_batches")
        all_ids.update(expected)
        batches[name] = {**item, "expected": expected}
    return root, _sha(manifest_bytes), batches


def _selected(root, batch):
    root, manifest_hash, batches = load_manifest(root)
    if batch not in batches:
        _fail("unknown_batch")
    return root, manifest_hash, batches[batch]


def _claim_record(root, batch, spec, manifest_hash):
    data = _read(root, root / "outputs" / batch / "claim.json")
    value = _json(data)
    if (type(value) is not dict or value.get("schema") != "aria_codex_judge_claim_v1"
            or value.get("batch") != batch or value.get("status") != "started"
            or value.get("input_sha256") != spec["sha256"]
            or value.get("manifest_sha256") != manifest_hash):
        _fail("claim_provenance_mismatch")
    return data


def claim(root, batch):
    """Create an exclusive claim before the task evaluates this batch."""
    root, manifest_hash, spec = _selected(root, batch)
    folder = _within(root, root / "outputs" / batch)
    if folder.exists():
        _fail("batch_already_started")
    state = verify(root)
    if state["counts"]["failed"] or state["counts"]["interrupted"]:
        _fail("prior_batch_incomplete_or_failed")
    next_batch = next((item["batch"] for item in state["batches"]
                       if item["status"] == "not_started"), None)
    if next_batch != batch:
        _fail("batch_order_changed")
    value = {"schema": "aria_codex_judge_claim_v1", "batch": batch,
             "status": "started", "input_sha256": spec["sha256"],
             "manifest_sha256": manifest_hash, "created_utc": _now()}
    _write(root, folder / "claim.json", _bytes(value))
    return {"batch": batch, "status": "started"}


def _validate_response(data, spec):
    value = _json(data)
    if type(value) is not list:
        _fail("response_expected_array")
    seen = set()
    for row in value:
        if type(row) is not list or len(row) != spec["width"] or type(row[0]) is not str:
            _fail("response_invalid_row")
        identity = row[0]
        if identity not in spec["expected"] or identity in seen:
            _fail("response_invalid_id_set")
        seen.add(identity)
        if any(label is not None and (type(label) is not int or label not in (0, 1))
               for label in row[1:]):
            _fail("response_invalid_label")
        if spec["width"] == 5 and any(not applicable and label != 0
                for applicable, label in zip(spec["expected"][identity], row[1:])):
            _fail("response_inapplicable_must_be_zero")
    if seen != set(spec["expected"]):
        _fail("response_invalid_id_set")
    return value


def submit(root, batch, response_file):
    """Retain the first raw answer, including invalid answers; never replace it."""
    root, manifest_hash, spec = _selected(root, batch)
    claim_bytes = _claim_record(root, batch, spec, manifest_hash)
    folder = root / "outputs" / batch
    if any(_within(root, folder / name).exists()
           for name in ("response.json", "receipt.json")):
        _fail("batch_response_already_saved")
    drafts = _within(root, root / "drafts")
    source = _within(drafts, Path(response_file))
    response_bytes = _read(root, source)
    _write(root, folder / "response.json", response_bytes)
    error = None
    try:
        _validate_response(response_bytes, spec)
    except IntakeError as exc:
        error = str(exc)
    value = {"schema": "aria_codex_judge_receipt_v1", "batch": batch,
             "status": "failed" if error else "completed", "error": error,
             "input_sha256": spec["sha256"], "manifest_sha256": manifest_hash,
             "claim_sha256": _sha(claim_bytes), "response_sha256": _sha(response_bytes),
             "finished_utc": _now(), "receipt_kind": "local_output_validation",
             "provider_receipt": False}
    _write(root, folder / "receipt.json", _bytes(value))
    return {"batch": batch, "status": value["status"], "error": error,
            "response_sha256": value["response_sha256"]}


def verify(root):
    """Report missing, interrupted, failed or valid batches without invented zeros."""
    root, manifest_hash, batches = load_manifest(root)
    outputs = _within(root, root / "outputs")
    if outputs.exists() and any(child.name not in batches for child in outputs.iterdir()):
        _fail("unexpected_output_batch")
    states = []
    for batch, spec in batches.items():
        folder = _within(root, outputs / batch)
        if not folder.exists():
            states.append({"batch": batch, "status": "not_started"})
            continue
        if any(child.name not in {"claim.json", "response.json", "receipt.json"}
               for child in folder.iterdir()):
            _fail("unexpected_batch_output_file")
        claim_bytes = _claim_record(root, batch, spec, manifest_hash)
        response_path = _within(root, folder / "response.json")
        receipt_path = _within(root, folder / "receipt.json")
        if not response_path.exists() or not receipt_path.exists():
            states.append({"batch": batch, "status": "interrupted"})
            continue
        response_bytes = _read(root, response_path)
        receipt = _json(_read(root, receipt_path))
        if (type(receipt) is not dict or receipt.get("schema") != "aria_codex_judge_receipt_v1"
                or receipt.get("batch") != batch
                or receipt.get("input_sha256") != spec["sha256"]
                or receipt.get("manifest_sha256") != manifest_hash
                or receipt.get("claim_sha256") != _sha(claim_bytes)
                or receipt.get("response_sha256") != _sha(response_bytes)
                or receipt.get("receipt_kind") != "local_output_validation"
                or receipt.get("provider_receipt") is not False):
            _fail("receipt_provenance_mismatch")
        error = None
        try:
            _validate_response(response_bytes, spec)
        except IntakeError as exc:
            error = str(exc)
        status = "failed" if error else "completed"
        if receipt.get("status") != status or receipt.get("error") != error:
            _fail("receipt_validation_mismatch")
        states.append({"batch": batch, "status": status, "error": error})
    counts = {name: sum(item["status"] == name for item in states)
              for name in ("completed", "failed", "interrupted", "not_started")}
    return {"schema": "aria_codex_judge_local_verification_v1",
            "status": "complete" if counts["completed"] == 25 else "incomplete",
            "denominator": 25, "counts": counts, "batches": states,
            "missing_labels_are_zero": False, "provider_receipt": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("claim", "submit"):
        action = actions.add_parser(name)
        action.add_argument("batch")
        if name == "submit":
            action.add_argument("--response-file", required=True)
    actions.add_parser("verify")
    args = parser.parse_args(argv)
    try:
        if args.action == "claim":
            result = claim(args.root, args.batch)
        elif args.action == "submit":
            result = submit(args.root, args.batch, args.response_file)
        else:
            result = verify(args.root)
    except (OSError, IntakeError) as exc:
        result = {"status": "blocked", "error": str(exc) if isinstance(exc, IntakeError)
                  else "local_file_operation_failed"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in ("started", "completed", "complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
