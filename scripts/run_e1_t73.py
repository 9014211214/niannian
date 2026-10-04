"""Single authorized E1 restart, retaining one exactly bound historical unknown fee.

Default use is read-only. Only --execute --allow-local-env may read the local key
and create the one fixed campaign after every source, history and route gate.
The historical ledger stays stopped; any new failure still stops without retry.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_e1_t72 as prior  # noqa: E402
from scripts import check_e1_t73 as wirecheck  # noqa: E402
from scripts import run_e1_t71 as previous  # noqa: E402
from scripts import t71_policy, t73_execution as execution_adapter  # noqa: E402
from scripts.t72_transport import OpenRouterTransport  # noqa: E402

REGISTRATION = "experiments/t73_20261001/authorized_run.json"
EXECUTION = "experiments/t73_20261001/authorized_execution.json"
FILES = frozenset({REGISTRATION, "scripts/run_e1_t73.py", "tests/test_e1_t73_run.py"})
REPAIR_SEAL = "d9f4ef6d6119265e574100a1f4fa428cde0bb893da9dbfaf7c18d003b6620f7f"
CAMPAIGN_ID = "e1_t73_20261001_v1"
PARENT_CAMPAIGN_ID = "t6_20260930_v1"
AUTHORIZATION_ID = "t73_20261001_unknown_cost_exception_v1"
ORIGINAL_REQUEST_SHA256 = "111189eb00e0b7418c57c3704f3cf4bd7a42c534894715d36ea71acf37930394"
ORIGINAL_RECEIPT_SHA256 = "bbc2db7335f4ab57a2812cba13c1f231d3edbfcf5f8cd5a7da9f44c11021ebe2"
HISTORY_FILES = frozenset({"campaign.json", "campaign.sqlite3", "e1-t71-preflight.json",
    "e1/inventory.json", "e1/e1_001.request.json", "e1/e1_001.receipt.json", "e1/results.private.json"})
AUTHORIZATION = {"user_statement": "接受1笔费用暂时未知", "historical_unknown_cost_count": 1,
    "retain_original_unknown_cost": True, "applies_only_to_campaign": CAMPAIGN_ID,
    "permits_automatic_retry": False}


class RunError(ValueError):
    """Fixed diagnostic codes only; never credentials or exception text."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _same(left, right):
    return previous.freeze.canonical_sha256(left) == previous.freeze.canonical_sha256(right)


def _no_links(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (path, *path.parents)):
        raise RunError("linked_path_rejected")
    return path


def _registration(bundle):
    value = previous.legacy.read_json(bundle["root"] / REGISTRATION)
    if (value.get("schema") != "aria_t73_authorized_run_v1"
            or value.get("id") != AUTHORIZATION_ID
            or value.get("campaign_id") != CAMPAIGN_ID
            or value.get("parent_campaign_id") != PARENT_CAMPAIGN_ID
            or value.get("repair_seal_sha256") != REPAIR_SEAL
            or not _same(value.get("authorization"), AUTHORIZATION)
            or not _same(value.get("budget_usd"), bundle["amendment"]["budget_usd"])
            or not _same(value.get("original_attempt"), {"id": "e1_001",
                "request_sha256": ORIGINAL_REQUEST_SHA256,
                "receipt_sha256": ORIGINAL_RECEIPT_SHA256})):
        raise RunError("authorized_registration_changed")
    return value


def verify_execution(bundle):
    if wirecheck.verify_repair(bundle)["seal_sha256"] != REPAIR_SEAL:
        raise RunError("wire_repair_seal_changed")
    _registration(bundle)
    value = previous.legacy.read_json(bundle["root"] / EXECUTION)
    if (value.get("schema") != "aria_t73_authorized_execution_v1"
            or value.get("repair_seal_sha256") != REPAIR_SEAL
            or value.get("seal_sha256") != previous.freeze.canonical_sha256(
                {k: v for k, v in value.items() if k != "seal_sha256"})):
        raise RunError("authorized_execution_seal_changed")
    entries = value.get("files", [])
    if len(entries) != len(FILES) or {e["path"] for e in entries} != FILES:
        raise RunError("authorized_execution_inventory_changed")
    for entry in entries:
        path = previous.freeze._public_file(bundle["root"], entry["path"])
        if path.stat().st_size != entry["bytes"] or prior._sha(path) != entry["sha256"]:
            raise RunError("authorized_execution_source_changed")
    return value


def verify_history(bundle, registration):
    """Check all seven original artifacts and the exact historical liability."""
    if not _same(registration, _registration(bundle)):
        raise RunError("authorized_registration_changed")
    original = previous.legacy.read_json(bundle["root"] / wirecheck.REGISTRATION)
    if not _same(registration.get("prior_campaign"), original["prior_campaign"]):
        raise RunError("historical_exception_scope_changed")
    directory = _no_links(previous.legacy.campaign_directory(bundle["root"]))
    entries = registration.get("prior_artifacts", [])
    if len(entries) != 7 or {e["path"] for e in entries} != HISTORY_FILES:
        raise RunError("historical_artifact_inventory_changed")
    actual_files = set()
    for path in directory.rglob("*"):
        _no_links(path)
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    if actual_files != HISTORY_FILES:
        raise RunError("historical_artifact_inventory_changed")

    def check_artifacts():
        for entry in entries:
            path = prior._regular(directory / entry["path"])
            if (type(entry["bytes"]) is not int or entry["bytes"] < 0
                    or not isinstance(entry["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                    or path.stat().st_size != entry["bytes"]
                    or prior._sha(path) != entry["sha256"]):
                raise RunError("historical_artifact_changed")

    check_artifacts()
    state = prior.inspect_campaign(bundle, previous.prepare_e1(bundle)[3])
    if (not _same(state, registration["prior_campaign"])
            or state["campaign_id"] != PARENT_CAMPAIGN_ID
            or state["stopped"] is not True
            or state["unknown_cost_count"] != 1
            or state["outstanding_count"] != 0
            or state["known_cost_usd"] != "0"
            or not _same(state["e1_counts"], {
                "failed": 1, "completed": 0, "started": 0, "not_started": 203})):
        raise RunError("historical_exception_scope_changed")
    request = previous.legacy.read_json(directory / "e1/e1_001.request.json")
    if (previous.freeze.canonical_sha256(request) != ORIGINAL_REQUEST_SHA256
            or prior._sha(directory / "e1/e1_001.receipt.json") != ORIGINAL_RECEIPT_SHA256):
        raise RunError("historical_attempt_changed")
    check_artifacts()
    return state


def campaign_directory(root=ROOT):
    return Path(root).absolute().parent / "aria-private-experiments" / CAMPAIGN_ID


def _new_campaign_available(root):
    directory = _no_links(campaign_directory(root))
    if directory.exists():
        raise RunError("new_campaign_already_exists")
    return directory


def preflight(bundle, *, online=False, opener=None):
    registration = _registration(bundle)
    verify_history(bundle, registration)
    report = wirecheck.build_report(bundle, online=online, opener=opener)
    if (report.get("prior_campaign") != registration["prior_campaign"]
            or "prior_cost_unresolved" not in report["blocking_reasons"]):
        raise RunError("historical_exception_scope_changed")
    report["blocking_reasons"] = [reason for reason in report["blocking_reasons"]
                                  if reason != "prior_cost_unresolved"]
    new_campaign_available = True
    try:
        _new_campaign_available(bundle["root"])
    except RunError as error:
        new_campaign_available = False
        report["blocking_reasons"].append(error.code)
    budget_qualified = (Decimal(report["planned_full_e1_reserve_usd"]) <= Decimal("0.5")
                        and report["fixed_positions"] == 204)
    if not budget_qualified:
        report["blocking_reasons"].append("new_stage_reservation_failed")
    report.update(schema="aria_t73_authorized_preflight_v1", campaign_id=CAMPAIGN_ID,
        authorization_id=AUTHORIZATION_ID, historical_unknown_cost_count=1,
        historical_known_cost_usd="0", new_unknown_cost_count=0 if new_campaign_available else None,
        total_actual_cost_usd=None, historical_unknown_cost_retained=True,
        new_stage_budget_qualified=budget_qualified, execution_under_historical_exception=True,
        paid_entry_enabled=True, ready_to_send=bool(online) and not report["blocking_reasons"])
    return report


def credential(root=ROOT):
    """Extract only this one key from local .env; never consult process secrets."""
    path = _no_links(Path(root) / ".env")
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise RunError("local_credential_unavailable")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise RunError("local_credential_unavailable") from None
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=(.*)", line)
        if match:
            value = match[1].strip()
            if value[:1] in {"'", '"'} and len(value) >= 2 and value[-1] == value[0]:
                value = value[1:-1]
            values.append(value)
    if len(values) != 1:
        raise RunError("local_credential_missing_or_duplicate")
    if re.fullmatch(r"[A-Za-z0-9_-]{16,256}", values[0]) is None:
        raise RunError("local_credential_invalid")
    return values[0]


def _marker(execution, registration):
    return {"campaign_id": CAMPAIGN_ID, "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "repair_seal_sha256": REPAIR_SEAL, "execution_seal": execution["seal_sha256"],
        "authorization_id": AUTHORIZATION_ID, "historical_unknown_cost_count": 1,
        "parent_artifacts_sha256": previous.freeze.canonical_sha256(registration["prior_artifacts"]),
        "original_attempt": registration["original_attempt"]}


def open_campaign(bundle, execution):
    if not _same(execution, verify_execution(bundle)):
        raise RunError("authorized_execution_seal_changed")
    registration = _registration(bundle)
    verify_history(bundle, registration)
    directory = _new_campaign_available(bundle["root"])
    # Atomic exclusive creation is the one-shot latch; partial failure consumes it.
    directory.mkdir(parents=True, exist_ok=False)
    previous.legacy.write_private(directory / "campaign.json", _marker(execution, registration))
    ledger = t71_policy.CampaignLedger(directory / "campaign.sqlite3", campaign_id=CAMPAIGN_ID,
                                     freeze_seal=execution["seal_sha256"])
    return directory, ledger


def _compact(result):
    state = result["campaign"]
    counts = state["stages"].get("e1", {}).get("counts", {})
    return {"ok": result["status"] == "complete", "campaign_id": CAMPAIGN_ID,
        "stage": "e1", "status": result["status"], "stop_reason": result["stop_reason"],
        "denominator": 204, "stage_counts": counts, "model_effect_measured": False,
        "new_known_cost_usd": state["known_cost_usd"],
        "new_unknown_cost_count": state["unknown_cost_count"],
        "historical_known_cost_usd": "0", "historical_unknown_cost_count": 1,
        "total_unknown_cost_count": 1 + state["unknown_cost_count"],
        "total_actual_cost_usd": None, "outstanding_count": state["outstanding_count"],
        "no_automatic_retry": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--online-preflight", action="store_true")
    parser.add_argument("--allow-local-env", action="store_true")
    args = parser.parse_args(argv)
    ledger = None
    try:
        if args.execute and not args.allow_local_env:
            raise RunError("local_env_permission_required")
        bundle = previous.load_bundle(ROOT)
        execution = verify_execution(bundle)
        report = preflight(bundle, online=args.online_preflight or args.execute)
        if not args.execute:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0 if report["ready_to_send"] else 2
        if not report["ready_to_send"]:
            raise RunError("preflight_not_ready")
        _new_campaign_available(bundle["root"])
        key = credential(bundle["root"])
        directory, ledger = open_campaign(bundle, execution)
        previous.legacy.write_private(directory / "e1-t73-preflight.json", report)
        expected_marker = _marker(execution, _registration(bundle))

        def source_check():
            current = previous.load_bundle(ROOT)
            if not _same(verify_execution(current), execution):
                raise RunError("authorized_execution_seal_changed")
            verify_history(current, _registration(current))
            if not _same(previous.legacy.read_json(directory / "campaign.json"), expected_marker):
                raise RunError("new_campaign_binding_changed")

        result = execution_adapter.run_generation(directory, ledger, bundle,
            OpenRouterTransport(key, enable_network=True), source_check=source_check)
        source_check()
        report = _compact(result)
        previous.legacy.write_private(directory / "outcome.aggregate.json", report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0 if report["ok"] else 2
    except (Exception, KeyboardInterrupt) as error:
        if ledger is not None:
            try:
                ledger.stop("e1", "manual_stop" if isinstance(error, KeyboardInterrupt) else "external_error")
            except Exception:
                pass
        code = error.code if isinstance(error, RunError) else "authorized_run_failed"
        print(json.dumps({"ok": False, "reason": code, "no_automatic_retry": True,
            "historical_unknown_cost_count": 1, "total_actual_cost_usd": None}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
