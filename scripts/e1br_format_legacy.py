"""Read archived v3 schemas in a separate process; no requests or replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

MANIFEST = "artifacts/local/tasks/20261005-01/e1br/v3-main-stop-originals-v1.manifest.json"
MANIFEST_SHA = "29637553cf5ea82f69fd13d615c0bab85450a5f1a43d71a2261925855773e86c"
FREEZE = "artifacts/local/tasks/20261005-01/e1br-v3-root-formal-freeze-v1.json"
FREEZE_SHA = "b5aebe5f1e60ae2edc2d67d7bc37c4628fd5f73e7d6012cad474e9855e3821d2"
PARENT = "artifacts/local/e1br-formal-v3"
RUN = "main/synthetic-emotion-experiments/e1b-main-v1/run"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_legacy(root):
    root = Path(root).resolve()
    def read(relative):
        path = (root / relative).resolve()
        if root not in path.parents or path.name.casefold() == ".env" or path.is_symlink():
            raise ValueError("legacy_input_outside_workspace")
        return path.read_bytes()
    freeze_raw, manifest_raw = read(FREEZE), read(MANIFEST)
    if sha(freeze_raw) != FREEZE_SHA or sha(manifest_raw) != MANIFEST_SHA:
        raise ValueError("legacy_anchors_changed")
    freeze, manifest = json.loads(freeze_raw), json.loads(manifest_raw)
    if len(freeze["archive_files"]) != 137 or len(manifest["files"]) != 368:
        raise ValueError("legacy_inventory_changed")
    for row in manifest["files"]:
        if sha(read(row["path"])) != row["sha256"]:
            raise ValueError("legacy_original_changed")
    archive = root / freeze["archive"]
    for row in freeze["archive_files"]:
        if sha(read(Path(freeze["archive"]) / row["path"])) != row["sha256"]:
            raise ValueError("legacy_source_changed")
    # No current product imports are allowed in this process. The root script
    # is itself new reviewable code, while every product/parser import is old.
    if any(name == "aria" or name.startswith("aria.") for name in sys.modules):
        raise ValueError("legacy_process_already_loaded_product")
    sys.path.insert(0, str(archive))
    from aria.engine.policy.online_json import parse_json_decision, SCHEMA_MARKER, _SCHEMA_INSTRUCTION
    from aria.engine.policy.online_decision import online_decision_tool_spec, DETERMINISTIC_MODEL_FAILURE_REPLY
    import aria
    if archive not in Path(aria.__file__).resolve().parents:
        raise ValueError("legacy_import_not_archived")
    receipt = json.loads(read(Path(PARENT) / "main/execution-receipt.json"))
    rows = receipt["accounting"]["attempts"]
    if len(rows) != 72:
        raise ValueError("legacy_attempt_count_changed")
    accepted, refused, text_returns = [], [], []
    for row in rows:
        wire_raw = read(Path(PARENT) / row["wire_request_path"])
        body_raw = read(Path(PARENT) / row["visible_original_path"])
        body, wire = json.loads(body_raw), json.loads(wire_raw)
        if (sha(wire_raw) != row["request_sha256"] or sha(body_raw) != row["visible_original_sha256"]
                or sha(body["content"].encode()) != row["visible_content_sha256"]):
            raise ValueError("legacy_wire_or_first_body_changed")
        if row["role"] == "online_fast":
            if row["reply_return_channel"] == "text":
                if sha(body["content"].strip().encode()) != row["model_reply_sha256"]:
                    raise ValueError("legacy_text_reply_changed")
                text_returns.append(row["request_id"])
                continue
            version = row["online_decision_version"]
            embedded = json.loads(wire["messages"][0]["content"].rsplit(SCHEMA_MARKER, 1)[1][len(_SCHEMA_INSTRUCTION):])
            if embedded != online_decision_tool_spec(version)["function"]["parameters"]:
                raise ValueError("legacy_wire_schema_changed")
            try:
                parse_json_decision(body["content"], version)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                refused.append(row["request_id"])
            else:
                accepted.append(row["request_id"])
    if len(accepted) + len(text_returns) != 69 or refused != ["1cbe1b56a87b412d8226cec3817fcc3f"]:
        raise ValueError("legacy_original_validation_outcomes_changed")
    live = Path(PARENT) / RUN / "main/e1b_c1/shared/live"
    profile, run = json.loads(read(live / "profiles/fictional-chen.json")), json.loads(read(live / "run.json"))
    queued = profile["offline_sessions"]
    if (len(queued) != 1 or len(queued[0]["turns"]) != 6
            or queued[0]["turns"][-1]["agent"] != DETERMINISTIC_MODEL_FAILURE_REPLY
            or run["turns"][-1]["response"] != "" or run["turns"][-1]["status"] != "failed"):
        raise ValueError("legacy_actual_fallback_changed")
    return {"schema": "e1br_format_legacy_verification_v1", "status": "pass",
        "manifest_sha256": MANIFEST_SHA, "archive_files_verified": 137, "original_files_verified": 368,
        "online_schema_accepted": len(accepted), "online_schema_rejected_request_ids": refused,
        "online_text_returns_verified": len(text_returns),
        "wire_and_first_bodies_verified": len(rows), "api_calls": 0, "key_reads": 0,
        "old_inputs_replayed": 0, "current_prompt_used_for_old_inputs": False,
        "fallback_sha256": sha(DETERMINISTIC_MODEL_FAILURE_REPLY.encode())}


def main():
    # Audit hooks cover accidental new transport and writes, including imports.
    # Bytecode is disabled by the parent and again here before archived imports.
    sys.dont_write_bytecode = True
    def audit(event, args):
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.system"}:
            raise RuntimeError("legacy_validation_network_or_process_forbidden")
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes)) and Path(path).name.casefold() == ".env":
                raise RuntimeError("legacy_validation_env_forbidden")
            if (isinstance(mode, str) and any(c in mode for c in "wa+")) or flags & 3:
                raise RuntimeError("legacy_validation_write_forbidden")
    sys.addaudithook(audit)
    print(json.dumps(verify_legacy(sys.argv[1]), sort_keys=True))


if __name__ == "__main__":
    main()
