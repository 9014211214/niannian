"""Read-only T7.2 qualification; no credentials, inference, or campaign writes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1_t71 as previous  # noqa: E402
from scripts import t71_policy, t72_policy  # noqa: E402
from scripts.t7_transport import TransportError  # noqa: E402

REGISTRATION = "experiments/t72_20260930/amendment.json"
EXECUTION = "experiments/t72_20260930/execution.json"
FILES = frozenset({REGISTRATION, "scripts/check_e1_t72.py", "scripts/t72_policy.py",
    "scripts/t72_transport.py", "tests/test_e1_t72.py", "tests/test_t72_policy.py",
    "tests/test_t72_transport.py", "docs/E1_EXECUTION_T72.md"})
T71_SEAL = "b073aff9657e8889a8e940c7b57e8a5dd247db925fe2e6d02af87dea38d5378d"
INHERITS = {"t6_seal": previous.T6_SEAL, "t7_seal": previous.T7_SEAL, "t71_seal": T71_SEAL}


class QualificationError(ValueError):
    """Only fixed codes; never paths, contents, or exception representations."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise QualificationError("private_link_rejected")
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise QualificationError("private_file_unavailable")
    return path


def verify_repair(bundle):
    root = bundle["root"]
    old = previous.verify_execution(bundle)
    if old["seal_sha256"] != T71_SEAL:
        raise QualificationError("prior_execution_changed")
    registration = previous.legacy.read_json(root / REGISTRATION)
    if (registration.get("schema") != "aria_t72_repair_v1"
            or registration.get("inherits") != INHERITS
            or registration.get("wire_changes") != []
            or registration.get("allows_model_execution") is not False
            or registration.get("unknown_cost_policy") != "retain_and_block"
            or registration.get("budget_usd") != bundle["amendment"]["budget_usd"]):
        raise QualificationError("repair_registration_changed")
    execution = previous.legacy.read_json(root / EXECUTION)
    if (execution.get("schema") != "aria_t72_repair_seal_v1"
            or execution.get("inherits") != INHERITS
            or execution.get("seal_sha256") != previous.freeze.canonical_sha256(
                {k: v for k, v in execution.items() if k != "seal_sha256"})):
        raise QualificationError("repair_seal_changed")
    entries = execution.get("files", [])
    if len(entries) != len(FILES) or {e["path"] for e in entries} != FILES:
        raise QualificationError("repair_inventory_changed")
    for entry in entries:
        path = previous.freeze._public_file(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise QualificationError("repair_file_changed")
    return execution


class _ExistingLedger(t71_policy.CampaignLedger):
    """Use the old validation/summary with an immutable, read-only connection."""

    def __init__(self, path):
        self.path = _regular(path)
        self.campaign_id = previous.legacy.CAMPAIGN_ID
        self.freeze_seal = T71_SEAL
        self._poisoned = False
        self._owned = {}
        self._stop_path = self.path.with_name(self.path.name + ".stopped")
        # immutable=1 is safe only for the stopped, standalone DB being checked.
        # Reject live SQLite journals instead of silently ignoring their data.
        if any(self.path.with_name(self.path.name + suffix).exists()
               for suffix in ("-wal", "-shm", "-journal")):
            raise QualificationError("campaign_has_live_journal")
        with self._read() as conn:
            self._validate(conn)

    def _connect(self, *, readonly=False):
        if not readonly:
            raise QualificationError("qualification_is_read_only")
        conn = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        return conn

    @contextmanager
    def _write(self):
        raise QualificationError("qualification_is_read_only")
        yield  # pragma: no cover


def inspect_campaign(bundle, items):
    """Retain the failed denominator and unknown liabilities; never resolve them."""
    directory = previous.legacy.campaign_directory(bundle["root"])
    marker = _regular(directory / "campaign.json")
    database = _regular(directory / "campaign.sqlite3")
    tracked = {marker: _sha(marker), database: _sha(database)}

    def read_bound(path):
        path = _regular(path)
        digest = _sha(path)
        tracked[path] = digest
        value = previous.legacy.read_json(path)
        if _sha(path) != digest:
            raise QualificationError("campaign_changed_during_check")
        return value

    binding = read_bound(marker)
    if binding != {"campaign_id": previous.legacy.CAMPAIGN_ID,
                   "t6_seal": previous.T6_SEAL, "t7_seal": previous.T7_SEAL,
                   "execution_seal": T71_SEAL}:
        raise QualificationError("campaign_binding_changed")
    state = _ExistingLedger(database).summary()
    stage = state["stages"].get("e1", {})
    actual = stage.get("items", [])
    if len(actual) != 204 or len(items) != 204:
        raise QualificationError("campaign_denominator_changed")
    saved_inventory = read_bound(directory / "e1/inventory.json")
    if saved_inventory != items:
        raise QualificationError("campaign_inventory_changed")
    for saved, expected in zip(actual, items):
        if (saved["id"] != expected["id"]
                or saved["request_sha256"] != expected["request_sha256"]
                or Decimal(saved["reserve_usd"]) != Decimal(expected["reserve_usd"])):
            raise QualificationError("campaign_inventory_changed")
        if saved["status"] in {"failed", "completed"}:
            request = read_bound(directory / "e1" / (saved["id"] + ".request.json"))
            receipt_path = directory / "e1" / (saved["id"] + ".receipt.json")
            read_bound(receipt_path)
            if (previous.freeze.canonical_sha256(request) != expected["request_sha256"]
                    or tracked[receipt_path] != saved["response_sha256"]):
                raise QualificationError("campaign_receipt_changed")
    if any(_sha(path) != digest for path, digest in tracked.items()):
        raise QualificationError("campaign_changed_during_check")
    return {"campaign_id": previous.legacy.CAMPAIGN_ID,
        "e1_counts": stage["counts"], "fixed_positions": len(actual),
        "stopped": state["stopped"], "known_cost_usd": state["known_cost_usd"],
        "unknown_cost_count": state["unknown_cost_count"],
        "outstanding_count": state["outstanding_count"],
        "same_campaign_resume_allowed": False,
        "original_artifacts_verified": len(tracked),
        "artifact_set_sha256": previous.freeze.canonical_sha256(
            {path.relative_to(directory).as_posix(): digest for path, digest in tracked.items()}),
        "reads_are_immutable": True}


def build_report(bundle, *, online=False, opener=None):
    """Assess every actual payload; public GETs are separately opt-in."""
    _, _, prepared, items = previous.prepare_e1(bundle)
    metadata = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    assessments = {}
    for trial in prepared.values():
        assessment = t72_policy.assess_payload_support(model=trial["wire"]["model"],
            frozen_metadata=metadata, payload=trial["wire"],
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"])
        t72_policy.verify_payload_binding(trial["wire"], assessment)
        assessments[assessment["actual_payload_sha256"]] = assessment
    if any(item["request_sha256"] not in assessments for item in items):
        raise QualificationError("actual_payload_inventory_unchecked")
    missing = sorted({field for assessment in assessments.values()
                      for field in assessment["missing_fields"]})
    report = {"schema": "aria_t72_qualification_v1", "mode": "read_only",
        "model_requests": 0, "credentials_read": False, "fixed_positions": len(items),
        "distinct_payloads_checked": len(assessments),
        "payload_inventory_sha256": previous.freeze.canonical_sha256(
            [item["request_sha256"] for item in items]),
        "wire_changes": [], "missing_parameter_evidence": missing,
        "offline_parameter_evidence_sufficient": all(a["supported"] for a in assessments.values()),
        "budget_usd": bundle["amendment"]["budget_usd"],
        "planned_full_e1_reserve_usd": str(sum((Decimal(i["reserve_usd"]) for i in items), Decimal(0))),
        "provider_cost_guarantee": False,
        "live_metadata_checked": False, "ready_to_send": False,
        "execution_entry_enabled": False, "blocking_reasons": []}
    if missing:
        report["blocking_reasons"].append("actual_payload_parameter_support_unverified")
    if online:
        largest = max(prepared.values(), key=lambda t: t["cost_plan"]["input_token_plan_bound"])
        try:
            report["live_metadata"] = t72_policy.fetch_metadata(model=largest["wire"]["model"],
                frozen_metadata=metadata, payload=largest["wire"],
                prompt_token_bound=largest["cost_plan"]["input_token_plan_bound"],
                opener=opener, enable_network=True)
            report["live_metadata_checked"] = True
        except TransportError as exc:
            report["live_metadata"] = exc.receipt
            report["blocking_reasons"].append("live_metadata_gate_failed")
            # Codes originate from validated allowlisted payload fields, not error text.
            report["live_metadata_error"] = exc.code
    else:
        report["blocking_reasons"].append("live_metadata_not_checked")
    try:
        report["prior_campaign"] = inspect_campaign(bundle, items)
        state = report["prior_campaign"]
        if state["stopped"]:
            report["blocking_reasons"].append("prior_campaign_permanently_stopped")
        if state["unknown_cost_count"]:
            report["blocking_reasons"].append("prior_cost_unresolved")
        if state["outstanding_count"]:
            report["blocking_reasons"].append("prior_outcome_unresolved")
    except (ValueError, OSError, KeyError, sqlite3.Error, t71_policy.LedgerError):
        report["blocking_reasons"].append("prior_campaign_evidence_unavailable_or_invalid")
    report["blocking_reasons"].append("linked_rerun_protocol_not_registered")
    report["judge"] = {"eligible": False, "reason": "requires_complete_verified_generation",
        "full_actual_text_reservation_required": True, "budget_usd": "0.5"}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-preflight", action="store_true",
                        help="Only two unauthenticated public metadata GETs; no inference")
    args = parser.parse_args(argv)
    try:
        bundle = previous.load_bundle()
        seal = verify_repair(bundle)
        report = build_report(bundle, online=args.online_preflight)
        report["repair_seal"] = seal["seal_sha256"]
        report["ok"] = True  # The inspection completed; execution is a separate flag.
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 2 if report["blocking_reasons"] else 0
    except Exception:
        print(json.dumps({"ok": False, "ready_to_send": False,
                          "reason": "qualification_check_failed", "model_requests": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
