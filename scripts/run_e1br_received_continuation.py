"""Versioned E1b-R continuation: locally consume the preserved paid first body.

The halted original is never reopened for writes and its body is never issued
again. Only the remaining fixed positions can consume the inherited allowance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1br as base
from scripts import run_e1b_repair as common
from scripts import e1b_main_continuation as retained
from aria.bench.e1br_runner import E1BRRunner, ProductReflectionCapture
from aria.engine.policy.e1br_calls import validate_e1br_ledger

CAMPAIGN = Path("artifacts/local/e1br-formal-v2")
PARENT = base.CAMPAIGN
MANIFEST = Path("artifacts/local/tasks/20261004-07/e1br-first-stop-originals-v1.manifest.json")
MANIFEST_SHA = "48b5b8fac8a00f6e4de8829a9ca372cdaf16a4b99c7b20ba652309f006f80a02"
FREEZE = Path("artifacts/local/tasks/20261004-07/e1br-root-formal-freeze-v1.json")
RECEIVED_REQUEST = "2e3f49a0944445189925ff8dc73f1108"
PRIOR_E1B, PRIOR_KNOWN = Decimal("0.161604814"), Decimal("1.553983986")
ALLOWANCE = Decimal("1.018088044")
PRIOR_DISPATCHES, RETAINED_MAIN = 557, 14
MAX_WIRE_REQUESTS = 3149
NEW_COUNTS = {"main": {"online_fast": 996, "memory": 23, "reflection": 23}}
RETAINED_UNKNOWN_HOLD, SELECTED = base.RETAINED_UNKNOWN_HOLD, base.SELECTED
BUDGET_CHOICE, RunError = base.BUDGET_CHOICE, base.RunError


def load_received(root):
    root = Path(root)
    raw = common.read_original(root, MANIFEST)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise RunError("e1br_received_parent_manifest_changed")
    manifest = json.loads(raw)
    if (len(manifest["files"]) != 34 or manifest["known_new_cost_usd"] != "0.02671"
            or manifest["received"] != 1 or manifest["new_unknown"] != 0
            or manifest["model_body_reissue_authorized"] is not False):
        raise RunError("e1br_received_parent_inventory_changed")
    for item in manifest["files"]:
        if hashlib.sha256(common.read_original(root, Path(item["path"]))).hexdigest() != item["sha256"]:
            raise RunError("e1br_received_parent_bytes_changed")
    frozen = json.loads(common.read_original(root, FREEZE))
    archived = root / frozen["archive"]
    if len(frozen["archive_files"]) != 131 or len({r["path"] for r in frozen["archive_files"]}) != 130:
        raise RunError("e1br_received_archive_inventory_changed")
    for item in frozen["archive_files"]:
        if hashlib.sha256(common.read_original(root, (archived / item["path"]).relative_to(root))).hexdigest() != item["sha256"]:
            raise RunError("e1br_received_archive_bytes_changed")
    registration = json.loads(common.read_original(root, PARENT / "main/registration.json"))
    if registration["seal_sha256"] != frozen["seal_sha256"] or registration["source"] != [
            {"path": item["path"], "sha256": item["sha256"]} for item in frozen["archive_files"][:130]]:
        raise RunError("e1br_received_registration_changed")
    ledger = root / PARENT / "allowance.sqlite3"
    rows = validate_e1br_ledger(ledger, stage="main", registration_sha256=registration["seal_sha256"],
                               max_requests=3153, reservation_policy=registration["metadata"]["policy"])
    if len(rows) != 1:
        raise RunError("e1br_received_unique_original_required")
    row = rows[0]
    if (row["request_id"] != RECEIVED_REQUEST or row["status"] != "received" or row["role"] != "reflection"
            or row["product_step"] != "initial" or Decimal(str(row["usage"].get("cost"))) != Decimal("0.02671")
            or row["session_id"] != retained.SESSION or row["operation_id"] != retained.OPERATION):
        raise RunError("e1br_received_first_identity_changed")
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        if db.execute("SELECT halted FROM settings WHERE id=1").fetchone() != ("model_or_storage_failed",):
            raise RunError("e1br_parent_halt_changed")
    wire_raw = common.read_original(root, PARENT / row["wire_request_path"])
    visible_raw = common.read_original(root, PARENT / row["visible_original_path"])
    wire, visible = json.loads(wire_raw), json.loads(visible_raw)
    if (hashlib.sha256(wire_raw).hexdigest() != row["request_sha256"]
            or hashlib.sha256(visible_raw).hexdigest() != row["visible_original_sha256"]
            or base.original.text_sha(visible["content"]) != row["client_return_content_sha256"]
            or row["visible_content_sha256"] != row["client_return_content_sha256"]
            or visible["request_id"] != row["request_id"]):
        raise RunError("e1br_received_first_content_changed")
    first_path = PARENT / "main/synthetic-emotion-experiments/e1b-main-v1/run/main/e1b_a1/reflection-s1.first.json"
    first_raw = common.read_original(root, first_path)
    if json.loads(first_raw)["visible_response_sha256"] != row["client_return_content_sha256"]:
        raise RunError("e1br_received_saved_first_changed")
    return {"receipt": row, "content": visible["content"], "messages": wire["messages"],
            "source_campaign": PARENT.as_posix(), "source_first_sha256": hashlib.sha256(first_raw).hexdigest(),
            "source_visible_original_sha256": hashlib.sha256(visible_raw).hexdigest()}


def parent_originals(root):
    load_received(root)
    root = Path(root)
    manifest = base.original.read_json(root / MANIFEST)
    frozen = base.original.read_json(root / FREEZE)
    return list(dict.fromkeys([*base.parent_originals(root), root / MANIFEST, root / FREEZE,
        root / frozen["registration"], *(root / item["path"] for item in manifest["files"]),
        *(root / frozen["archive"] / item["path"] for item in frozen["archive_files"])]))


def build_engine(root):
    received = load_received(root)
    revision = common._load_module("scripts._e1br_received_revision", "run_e1br.py")
    revision.CAMPAIGN, revision.PRIOR_E1B, revision.PRIOR_KNOWN = CAMPAIGN, PRIOR_E1B, PRIOR_KNOWN
    revision.ALLOWANCE, revision.PRIOR_DISPATCHES, revision.RETAINED_MAIN = ALLOWANCE, PRIOR_DISPATCHES, RETAINED_MAIN
    revision.MAX_WIRE_REQUESTS, revision.NEW_COUNTS = MAX_WIRE_REQUESTS, NEW_COUNTS
    revision.PARENT_MANIFEST_SHA = MANIFEST_SHA
    revision.parent_originals = parent_originals
    revision.retained_extra_rows = lambda _: [deepcopy(received["receipt"])]
    revision.EXTRA_FILES = ("scripts/run_e1br_received_continuation.py", "scripts/e1br_received_body.py",
                            "docs/E1BR_RECEIVED_CONTINUATION.md")
    positions = base.position_inventory() - {("reflection", "e1b_a1", 1, "shared", 0, None)}
    revision.position_inventory = lambda: positions
    summary = {"schema": "e1br_received_first_continuation_v1", "source_campaign": PARENT.as_posix(),
        "source_request_id": RECEIVED_REQUEST, "received_content_sha256": received["receipt"]["visible_content_sha256"],
        "source_first_sha256": received["source_first_sha256"],
        "source_visible_original_sha256": received["source_visible_original_sha256"],
        "prior_paid_cost_usd": "0.02671", "local_replay_calls": 0, "no_reflection_reissue": True,
        "expected_local_terminal": "invalid_contract_scope_tags_reflection_failed",
        "accounting_correction": "SQLite unique received row is authoritative; unavailable snapshot is not zero calls"}
    revision.continuation_details = lambda _: {"received_first_continuation": deepcopy(summary)}

    def first(client, payload, scope, key, prefix, current):
        if key != ("online_fast", "e1b_a1", 2, "shared", 0, 0) or client.role != "online_fast":
            raise RunError("e1br_received_next_position_must_be_a1_s2_t0")
        return {"resume_after_received_request_id": RECEIVED_REQUEST,
                "retained_received_content_sha256": received["receipt"]["visible_content_sha256"],
                "retained_prefix_sha256": prefix["summary_sha256"]}
    revision.first_request_lineage = first

    def runner_for(current, prefix):
        from scripts.e1br_received_body import continue_received_reflection
        class ReceivedCapture(ProductReflectionCapture):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if kwargs["case_id"] == "e1b_a1" and kwargs["session"] == 1:
                    self.initial.receipt_rows = lambda: [deepcopy(received["receipt"])]
        class ReceivedBase(E1BRRunner):
            reflection_capture_class = ReceivedCapture
        class Continued(retained.runner_class(prefix, base_runner=ReceivedBase)):
            def _continue_reflection(self, app):
                load_received(current)
                return continue_received_reflection(app, prefix, received)
        return Continued
    revision.runner_for = runner_for
    result = revision.build_engine(root)
    result.received_reflection = received
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("assess")
    report.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
            command.add_argument("--budget-choice", choices=(BUDGET_CHOICE,), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "assess":
        assessment = common._load_module("scripts._e1br_received_assessment", "e1br_assessment.py")
        assessment.revision = sys.modules[__name__]
        result = assessment.assess(ROOT)
        base.original.write_once(args.output, result)
    else:
        entry = build_engine(ROOT)
        if args.command == "prepare":
            value = entry.prepare(ROOT, args.metadata, args.registration, stage="main")
            result = {"stage": "main", "seal_sha256": value["seal_sha256"], "authorization": entry.authorization_template(ROOT), "paid_calls": 0}
        elif args.command == "verify":
            result = {"verified": True, "seal_sha256": entry.verify(ROOT, args.metadata, args.registration)["seal_sha256"], "paid_calls": 0}
        else:
            result = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
                budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
                stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
