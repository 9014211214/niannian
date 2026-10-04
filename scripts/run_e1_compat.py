"""E1 compatible execution, preserving all failed attempts and budget gates.

Default is offline/read-only. Public preflight never reads keys. Execute checks
the exact new wire and history before extracting the one authorized local key.
It cannot waive a new unknown cost, reset an old run, or retry a claimed item.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_e1_t72 as prior  # noqa: E402
from scripts import e1_compat as compat, run_e1_t71 as previous  # noqa: E402
from scripts import run_e1_t73 as old_run, t71_policy, t73_policy  # noqa: E402
from scripts.t74_transport import OpenRouterTransport  # noqa: E402

REGISTRATION = "experiments/t74_20261001/amendment.json"
INVENTORY = "experiments/t74_20261001/requests.json"
EXECUTION = "experiments/t74_20261001/execution.json"
CAMPAIGN_ID = "e1_t74_20261001_v1"
FILES = frozenset({REGISTRATION, INVENTORY, "scripts/e1_compat.py",
    "scripts/run_e1_compat.py", "scripts/t74_transport.py", "scripts/check_e1_tool_choice.py",
    "tests/test_t74_execution.py", "tests/test_t74_transport.py", "tests/test_e1_compat.py", "tests/test_t74_run.py",
    "tests/test_e1_tool_choice.py", "docs/E1_CALL_COMPATIBILITY.md"})
HISTORY_IDS = ("t6_20260930_v1", "e1_t73_20261001_v1")
HISTORY_SEALS = (prior.T71_SEAL,
    "ce932d135b4504b18e9627ff2d12817be8bf77ed9a012762bf2a456835ccbfa6")
HISTORY_FILES = (old_run.HISTORY_FILES,
    (old_run.HISTORY_FILES - {"e1-t71-preflight.json"}) | {"e1-t73-preflight.json", "outcome.aggregate.json"})
# Only this original unknown is human-approved so far. No CLI flag expands it.
ACCEPTED_UNKNOWN_ATTEMPTS = (("t6_20260930_v1", "e1_001"),)
RunError = old_run.RunError


def _registration(bundle):
    value = previous.legacy.read_json(bundle["root"] / REGISTRATION)
    if (value.get("schema") != "aria_e1_auto_tool_amendment_v1"
            or value.get("campaign_id") != CAMPAIGN_ID
            or value.get("wire_changes") != compat.WIRE_CHANGES
            or value.get("budget_usd") != bundle["amendment"]["budget_usd"]
            or value.get("unknown_cost_policy") != "retain_and_block_unaccepted"
            or value.get("accepted_unknown_attempts") != [list(a) for a in ACCEPTED_UNKNOWN_ATTEMPTS]
            or value.get("fixed_generation_positions") != 204):
        raise RunError("compatibility_registration_changed")
    return value


def verify_execution(bundle):
    old_run.verify_execution(bundle)
    registration = _registration(bundle)
    inventory = previous.legacy.read_json(bundle["root"] / INVENTORY)
    expected = compat.prepare_e1(bundle)[3]
    if (inventory.get("items") != expected
            or inventory.get("items_sha256") != compat.canonical_sha256(expected)):
        raise RunError("compatible_inventory_changed")
    seal = previous.legacy.read_json(bundle["root"] / EXECUTION)
    if (seal.get("schema") != "aria_e1_auto_tool_execution_v1"
            or seal.get("parent_execution_seal") != HISTORY_SEALS[1]
            or seal.get("seal_sha256") != compat.canonical_sha256(
                {k: v for k, v in seal.items() if k != "seal_sha256"})
            or registration.get("parent_execution_seal") != HISTORY_SEALS[1]):
        raise RunError("compatible_execution_seal_changed")
    entries = seal.get("files", [])
    if len(entries) != len(FILES) or {row["path"] for row in entries} != FILES:
        raise RunError("compatible_execution_inventory_changed")
    for row in entries:
        path = previous.freeze._public_file(bundle["root"], row["path"])
        if path.stat().st_size != row["bytes"] or prior._sha(path) != row["sha256"]:
            raise RunError("compatible_execution_source_changed")
    return seal


class _ReadOnlyLedger(t71_policy.CampaignLedger):
    def __init__(self, path, campaign_id, freeze_seal):
        self.path = prior._regular(path)
        self.campaign_id, self.freeze_seal = campaign_id, freeze_seal
        self._poisoned, self._owned = False, {}
        self._stop_path = self.path.with_name(self.path.name + ".stopped")
        if any(self.path.with_name(self.path.name + suffix).exists()
               for suffix in ("-wal", "-shm", "-journal")):
            raise RunError("historical_campaign_has_live_journal")
        with self._read() as conn:
            self._validate(conn)

    def _connect(self, *, readonly=False):
        if not readonly:
            raise RunError("history_is_read_only")
        conn = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        return conn

    @contextmanager
    def _write(self):
        raise RunError("history_is_read_only")
        yield  # pragma: no cover


def verify_history(bundle):
    registration = _registration(bundle)
    rows = registration.get("prior_campaigns", [])
    if len(rows) != 2 or [row.get("campaign_id") for row in rows] != list(HISTORY_IDS):
        raise RunError("historical_campaign_inventory_changed")
    # This reuses the exact original user-approved seven-file scope verification.
    old_run.verify_history(bundle, old_run._registration(bundle))
    expected_sets = (previous.prepare_e1(bundle)[3], t73_policy.prepare_e1(bundle)[3])
    states, unaccepted = [], []
    for row, expected, identifier, seal, allowed in zip(rows, expected_sets, HISTORY_IDS, HISTORY_SEALS, HISTORY_FILES):
        directory = old_run._no_links(Path(bundle["root"]).parent / "aria-private-experiments" / identifier)
        entries = row.get("artifacts", [])
        actual_paths = set()
        for path in directory.rglob("*"):
            old_run._no_links(path)
            if path.is_file():
                actual_paths.add(path.relative_to(directory).as_posix())
        paths = [entry["path"] for entry in entries]
        if len(entries) != len(allowed) or set(paths) != allowed or actual_paths != allowed:
            raise RunError("historical_artifact_inventory_changed")
        def check_files():
            for entry in entries:
                path = prior._regular(directory / entry["path"])
                if path.stat().st_size != entry["bytes"] or prior._sha(path) != entry["sha256"]:
                    raise RunError("historical_artifact_changed")
        check_files()
        state = _ReadOnlyLedger(directory / "campaign.sqlite3", identifier, seal).summary()
        stage = state["stages"].get("e1", {})
        if (not state["stopped"] or state["outstanding_count"] != 0
                or state["unknown_cost_count"] != 1 or state["known_cost_usd"] != "0"
                or stage.get("counts") != {"not_started": 203, "started": 0, "completed": 0, "failed": 1}
                or previous.legacy.read_json(directory / "e1/inventory.json") != expected):
            raise RunError("historical_campaign_state_changed")
        actual = stage["items"]
        if len(actual) != 204 or any(saved["id"] != planned["id"]
                or saved["request_sha256"] != planned["request_sha256"]
                for saved, planned in zip(actual, expected)):
            raise RunError("historical_campaign_inventory_changed")
        first = actual[0]
        if (compat.canonical_sha256(previous.legacy.read_json(directory / "e1/e1_001.request.json"))
                != first["request_sha256"]
                or prior._sha(directory / "e1/e1_001.receipt.json") != first["response_sha256"]):
            raise RunError("historical_attempt_changed")
        check_files()
        states.append({"campaign_id": identifier, "unknown_cost_count": 1,
                       "known_cost_usd": "0", "stopped": True, "files_verified": len(entries)})
        if (identifier, "e1_001") not in ACCEPTED_UNKNOWN_ATTEMPTS:
            unaccepted.append([identifier, "e1_001"])
    return {"campaigns": states, "unknown_cost_count": 2, "known_cost_usd": "0",
            "unaccepted_unknown_attempts": unaccepted, "total_actual_cost_usd": None}


def campaign_directory(root=ROOT):
    return Path(root).absolute().parent / "aria-private-experiments" / CAMPAIGN_ID


def preflight(bundle, *, online=False, opener=None):
    verify_execution(bundle)
    history = verify_history(bundle)
    report = compat.preflight(bundle, online=online, opener=opener)
    report["parameter_ready_to_send"] = report["ready_to_send"]
    report["history"] = history
    if history["unaccepted_unknown_attempts"]:
        report["blocking_reasons"].append("new_historical_unknown_cost_requires_resolution")
    if old_run._no_links(campaign_directory(bundle["root"])).exists():
        report["blocking_reasons"].append("new_campaign_already_exists")
    if Decimal(report["planned_full_e1_reserve_usd"]) > Decimal("0.5"):
        report["blocking_reasons"].append("new_stage_reservation_failed")
    report["ready_to_send"] = report["parameter_ready_to_send"] and not report["blocking_reasons"]
    report["no_automatic_retry"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-env", action="store_true")
    args = parser.parse_args(argv)
    ledger = None
    try:
        if args.execute and not args.allow_local_env:
            raise RunError("local_env_permission_required")
        bundle = previous.load_bundle(ROOT)
        execution = verify_execution(bundle)
        report = preflight(bundle, online=args.online_preflight or args.execute)
        if not args.execute or not report["ready_to_send"]:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0 if report["ready_to_send"] else 2
        directory = old_run._no_links(campaign_directory(bundle["root"]))
        if directory.exists():
            raise RunError("new_campaign_already_exists")
        key = old_run.credential(bundle["root"])
        directory.mkdir(parents=True, exist_ok=False)
        marker = {"campaign_id": CAMPAIGN_ID, "execution_seal": execution["seal_sha256"],
            "parent_campaign_ids": list(HISTORY_IDS), "historical_unknown_cost_count": 2,
            "accepted_unknown_attempts": [list(a) for a in ACCEPTED_UNKNOWN_ATTEMPTS],
            "linked_inventory_sha256": report["linked_inventory_sha256"]}
        previous.legacy.write_private(directory / "campaign.json", marker)
        previous.legacy.write_private(directory / "preflight.json", report)
        ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3", campaign_id=CAMPAIGN_ID,
                                         freeze_seal=execution["seal_sha256"])
        def source_check():
            current = previous.load_bundle(ROOT)
            if verify_execution(current) != execution or verify_history(current) != report["history"]:
                raise RunError("execution_or_history_changed")
            if previous.legacy.read_json(directory / "campaign.json") != marker:
                raise RunError("campaign_binding_changed")
        result = compat.run_generation(directory, ledger, bundle,
            OpenRouterTransport(key, enable_network=True), source_check=source_check, qualification=report)
        source_check()
        outcome = {"schema": "aria_e1_compatible_outcome_v1", "campaign_id": CAMPAIGN_ID,
            "ok": result["status"] == "complete", "status": result["status"],
            "stop_reason": result["stop_reason"], "denominator": 204,
            "stage_counts": ledger.summary()["stages"]["e1"]["counts"],
            "new_known_cost_usd": ledger.summary()["known_cost_usd"],
            "new_unknown_cost_count": ledger.summary()["unknown_cost_count"],
            "historical_unknown_cost_count": 2, "total_actual_cost_usd": None,
            "model_effect_measured": False, "no_automatic_retry": True}
        previous.legacy.write_private(directory / "outcome.aggregate.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
        return 0 if outcome["ok"] else 2
    except (Exception, KeyboardInterrupt) as error:
        if ledger is not None:
            try:
                ledger.stop("e1", "manual_stop" if isinstance(error, KeyboardInterrupt) else "external_error")
            except Exception:
                pass
        code = error.code if isinstance(error, RunError) else "compatible_run_failed"
        print(json.dumps({"ok": False, "reason": code, "no_automatic_retry": True,
                         "total_actual_cost_usd": None}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
