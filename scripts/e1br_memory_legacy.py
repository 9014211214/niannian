"""Read the archived v4 wire/schema under v4 imports, never replaying a call."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MANIFEST = "artifacts/local/tasks/20261005-02/e1br-close-review/v4-closed-originals-v1.manifest.json"
MANIFEST_SHA = "bf04cb2831d776618abfda2525b010b36e553128d0a2099681aaed0d1009f293"
FREEZE = "artifacts/local/tasks/20261005-02/e1br-v4-root-formal-freeze-v1.json"
FREEZE_SHA = "215585f2c3d9e39b570da14f9e1f62e3fc24c8d56c1e7d575b41475dbbb95914"
PARENT = "artifacts/local/e1br-formal-v4"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_legacy(root):
    root = Path(root).absolute()
    def read(relative):
        relative = Path(relative)
        if relative.is_absolute() or not relative.parts or any(part in {".", ".."} or part.casefold() == ".env" for part in relative.parts):
            raise ValueError("memory_legacy_path_rejected")
        path = root
        for part in relative.parts:
            path /= part
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
                raise ValueError("memory_legacy_link_rejected")
        return path.read_bytes()
    freeze_raw, manifest_raw = read(FREEZE), read(MANIFEST)
    if sha(freeze_raw) != FREEZE_SHA or sha(manifest_raw) != MANIFEST_SHA:
        raise ValueError("memory_legacy_anchors_changed")
    freeze, manifest = json.loads(freeze_raw), json.loads(manifest_raw)
    if len(freeze["archive_files"]) != 145 or len(manifest["files"]) != 1149:
        raise ValueError("memory_legacy_inventory_changed")
    for row in manifest["files"]:
        if sha(read(row["path"])) != row["sha256"]:
            raise ValueError("memory_legacy_original_changed")
    for row in freeze["archive_files"]:
        if sha(read(Path(freeze["archive"]) / row["path"])) != row["sha256"]:
            raise ValueError("memory_legacy_source_changed")
    if any(name == "aria" or name.startswith("aria.") for name in sys.modules):
        raise ValueError("memory_legacy_product_already_loaded")
    archive = root / freeze["archive"]
    sys.path.insert(0, str(archive))
    import aria
    from aria.engine.policy.online_json import parse_json_decision, SCHEMA_MARKER, _SCHEMA_INSTRUCTION
    from aria.engine.policy.online_decision import online_decision_tool_spec
    from aria.engine.memory.consolidator import _parse_json_payload
    if archive not in Path(aria.__file__).resolve().parents:
        raise ValueError("memory_legacy_import_not_archived")
    rows = json.loads(read(Path(PARENT) / "main/execution-receipt.json"))["accounting"]["attempts"]
    if len(rows) != 361:
        raise ValueError("memory_legacy_receipt_count_changed")
    counts = {"wire": 0, "visible": 0, "text": 0, "json_accepted": 0, "json_rejected": 0, "missing_body": 0}
    for row in rows:
        wire_raw = read(Path(PARENT) / row["wire_request_path"])
        if sha(wire_raw) != row["request_sha256"]:
            raise ValueError("memory_legacy_wire_changed")
        counts["wire"] += 1
        if not row.get("visible_original_path"):
            if row["http_status"] != 429 or row["status"] != "failed":
                raise ValueError("memory_legacy_missing_body_changed")
            counts["missing_body"] += 1
            continue
        raw = read(Path(PARENT) / row["visible_original_path"])
        body, wire = json.loads(raw), json.loads(wire_raw)
        if sha(raw) != row["visible_original_sha256"] or sha(body["content"].encode()) != row["visible_content_sha256"]:
            raise ValueError("memory_legacy_first_body_changed")
        counts["visible"] += 1
        if row["role"] == "online_fast":
            if row["reply_return_channel"] == "text":
                if sha(body["content"].strip().encode()) != row["model_reply_sha256"]:
                    raise ValueError("memory_legacy_text_changed")
                counts["text"] += 1
            else:
                version = row["online_decision_version"]
                embedded = json.loads(wire["messages"][0]["content"].rsplit(SCHEMA_MARKER, 1)[1][len(_SCHEMA_INSTRUCTION):])
                if embedded != online_decision_tool_spec(version)["function"]["parameters"]:
                    raise ValueError("memory_legacy_schema_changed")
                try:
                    parse_json_decision(body["content"], version)
                except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                    if row["status"] != "failed":
                        raise ValueError("memory_legacy_original_json_outcome_changed") from None
                    counts["json_rejected"] += 1
                else:
                    if row["status"] != "received":
                        raise ValueError("memory_legacy_original_json_outcome_changed")
                    counts["json_accepted"] += 1
    if counts != {"wire": 361, "visible": 360, "text": 266, "json_accepted": 74, "json_rejected": 2, "missing_body": 1}:
        raise ValueError("memory_legacy_observed_counts_changed")
    failed = rows[-1]
    if tuple(failed["experiment_position"]) != ("memory", "e1b_e1", 2, "shared", 0, None):
        raise ValueError("memory_legacy_stop_position_changed")
    try:
        _parse_json_payload(json.loads(read(Path(PARENT) / failed["visible_original_path"]))["content"])
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError("memory_legacy_bad_body_unexpectedly_valid")
    return {"schema": "e1br_memory_legacy_verification_v1", "status": "pass", "manifest_sha256": MANIFEST_SHA,
            "archive_files_verified": 145, "original_files_verified": 1149, "counts": counts,
            "api_calls": 0, "key_reads": 0, "old_inputs_replayed": 0, "current_prompt_used_for_old_inputs": False}


def main():
    sys.dont_write_bytecode = True
    def audit(event, args):
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.system"}:
            raise RuntimeError("memory_legacy_network_or_process_forbidden")
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes)) and Path(os.fsdecode(path)).name.casefold() == ".env":
                raise RuntimeError("memory_legacy_env_forbidden")
            if isinstance(mode, str) and any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise RuntimeError("memory_legacy_write_forbidden")
    sys.addaudithook(audit)
    print(json.dumps(verify_legacy(sys.argv[1]), sort_keys=True))


if __name__ == "__main__":
    main()
