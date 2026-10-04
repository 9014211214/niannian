"""Prospective E1 carrier repair; all attempts and their shared budget survive.

Only explicit execute reads the authorized local key. The failed native-tool
campaign is immutable; a new full 204-position campaign uses the same wire and
accepts either a genuine single tool call or strictly validated JSON content.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_json_carrier as compat  # noqa: E402
from scripts import run_e1_authorized as authorized, run_e1_compat as parent  # noqa: E402
from scripts import run_e1_t71 as previous, t71_policy  # noqa: E402
from scripts.t74_transport import OpenRouterTransport  # noqa: E402

REGISTRATION = "experiments/t75_20261001/amendment.json"
EXECUTION = "experiments/t75_20261001/execution.json"
CAMPAIGN_ID = "e1_t75_20261001_v1"
PARENT_CAMPAIGN = "e1_t74_20261001_v1"
PARENT_AUTHORIZATION_SEAL = "490ac7fc64b800024fe4d4d12891c239754cb2d8049e7106c0ff1c777c308351"
CARRY_KNOWN_COST = "0.00171618"
E1_LIMIT = Decimal("0.5")
FILES = frozenset({REGISTRATION, "scripts/run_e1_carrier.py", "scripts/e1_json_carrier.py",
    "tests/test_e1_carrier_run.py", "tests/test_e1_json_carrier.py", "docs/E1_JSON_CARRIER.md"})
RunError = parent.RunError
ACCEPTED_UNKNOWN_ATTEMPTS = authorized.ACCEPTED_UNKNOWN_ATTEMPTS
HISTORY_FILES = frozenset({"campaign.json", "campaign.sqlite3", "preflight.json", "outcome.aggregate.json",
    "e1/inventory.json", "e1/results.private.json"} | {
    f"e1/e1_{index:03d}.{kind}.json" for index in range(1, 13) for kind in ("request", "receipt")})


def expected_registration(bundle, artifacts=None):
    if artifacts is None:
        artifacts = previous.legacy.read_json(bundle["root"] / REGISTRATION)["prior_known_campaign"]["artifacts"]
    return {"schema": "aria_e1_dual_carrier_amendment_v1", "campaign_id": CAMPAIGN_ID,
        "parent_authorization_seal": PARENT_AUTHORIZATION_SEAL,
        "authorization_source": "User instruction 2026-10-01 / PROGRESS task 20261001-04",
        "authorization": deepcopy(authorized.AUTHORIZATION) | {
            "new_private_directory": f"artifacts/local/{CAMPAIGN_ID}"},
        "accepted_unknown_attempts": [list(a) for a in ACCEPTED_UNKNOWN_ATTEMPTS],
        "budget_usd": deepcopy(bundle["amendment"]["budget_usd"]),
        "fixed_generation_positions": 204, "inventory_source": parent.INVENTORY,
        "wire_changes_from_t74": [],
        "response_carriers": ["native_tool", "content_json"],
        "response_contract": "unchanged_tool_parameters_and_production_validation",
        "measurement": "prospective_full_204_no_old_result_upgrade_no_best_retry_selection",
        "unknown_cost_policy": "retain_and_block_unaccepted",
        "prior_known_campaign": {"campaign_id": PARENT_CAMPAIGN,
            "directory": f"artifacts/local/{PARENT_CAMPAIGN}",
            "execution_seal": PARENT_AUTHORIZATION_SEAL, "known_cost_usd": CARRY_KNOWN_COST,
            "unknown_cost_count": 0, "artifacts": deepcopy(artifacts)}}


def _registration(bundle):
    value = previous.legacy.read_json(bundle["root"] / REGISTRATION)
    if value != expected_registration(bundle):
        raise RunError("carrier_registration_changed")
    entries = value["prior_known_campaign"]["artifacts"]
    if (len(entries) != len(HISTORY_FILES) or {row["path"] for row in entries} != HISTORY_FILES
            or any(set(row) != {"path", "bytes", "sha256"} for row in entries)):
        raise RunError("carrier_history_inventory_changed")
    return value


def verify_execution(bundle):
    if authorized.verify_execution(bundle)["seal_sha256"] != PARENT_AUTHORIZATION_SEAL:
        raise RunError("carrier_parent_seal_changed")
    _registration(bundle)
    seal = previous.legacy.read_json(bundle["root"] / EXECUTION)
    if (seal.get("schema") != "aria_e1_dual_carrier_execution_v1"
            or seal.get("parent_authorization_seal") != PARENT_AUTHORIZATION_SEAL
            or seal.get("seal_sha256") != compat.canonical_sha256(
                {k: v for k, v in seal.items() if k != "seal_sha256"})):
        raise RunError("carrier_execution_seal_changed")
    entries = seal.get("files", [])
    if len(entries) != len(FILES) or {row["path"] for row in entries} != FILES:
        raise RunError("carrier_source_inventory_changed")
    for row in entries:
        path = previous.freeze._public_file(bundle["root"], row["path"])
        if path.stat().st_size != row["bytes"] or parent.prior._sha(path) != row["sha256"]:
            raise RunError("carrier_source_changed")
    return seal


def verify_history(bundle):
    registration = _registration(bundle)
    original = authorized.verify_history(bundle)
    directory = parent.old_run._no_links(bundle["root"] / registration["prior_known_campaign"]["directory"])
    actual_paths = set()
    for path in directory.rglob("*"):
        parent.old_run._no_links(path)
        if path.is_file():
            actual_paths.add(path.relative_to(directory).as_posix())
    if actual_paths != HISTORY_FILES:
        raise RunError("carrier_historical_files_changed")

    def check_files():
        for entry in registration["prior_known_campaign"]["artifacts"]:
            path = parent.prior._regular(directory / entry["path"])
            if path.stat().st_size != entry["bytes"] or parent.prior._sha(path) != entry["sha256"]:
                raise RunError("carrier_historical_bytes_changed")

    check_files()
    state = parent._ReadOnlyLedger(directory / "campaign.sqlite3", PARENT_CAMPAIGN,
                                  PARENT_AUTHORIZATION_SEAL).summary()
    items = compat.prepare_e1(bundle)[3]
    stage = state["stages"].get("e1", {})
    if (not state["stopped"] or state["stop_reason"] != "invalid_schema"
            or state["outstanding_count"] != 0 or state["unknown_cost_count"] != 0
            or Decimal(state["known_cost_usd"]) != Decimal(CARRY_KNOWN_COST)
            or stage.get("counts") != {"not_started": 192, "started": 0, "completed": 11, "failed": 1}
            or previous.legacy.read_json(directory / "e1/inventory.json") != items):
        raise RunError("carrier_historical_state_changed")
    saved = stage["items"]
    if len(saved) != 204:
        raise RunError("carrier_historical_inventory_changed")
    for index, (actual, expected) in enumerate(zip(saved, items)):
        if (actual["id"] != expected["id"] or actual["request_sha256"] != expected["request_sha256"]
                or actual["status"] != ("completed" if index < 11 else "failed" if index == 11 else "not_started")):
            raise RunError("carrier_historical_inventory_changed")
        if index < 12:
            prefix = directory / "e1" / actual["id"]
            if (compat.canonical_sha256(previous.legacy.read_json(prefix.with_suffix(".request.json")))
                    != actual["request_sha256"]
                    or parent.prior._sha(prefix.with_suffix(".receipt.json")) != actual["response_sha256"]):
                raise RunError("carrier_historical_attempt_changed")
    check_files()
    return {**original, "known_cost_usd": CARRY_KNOWN_COST,
        "campaigns": original["campaigns"] + [{"campaign_id": PARENT_CAMPAIGN,
            "known_cost_usd": CARRY_KNOWN_COST, "unknown_cost_count": 0, "stopped": True,
            "files_verified": len(HISTORY_FILES), "stage_counts": stage["counts"]}]}


def campaign_directory(root=ROOT):
    return Path(root).absolute() / f"artifacts/local/{CAMPAIGN_ID}"


class CarryLedger(t71_policy.CampaignLedger):
    """Persist the remaining shared E1 cap, not a second $0.50 allowance."""

    def register_stage(self, stage, items, *, limit_usd, max_requests):
        if stage != "e1" or Decimal(str(limit_usd)) != E1_LIMIT or max_requests != 204:
            raise RunError("carrier_stage_scope_changed")
        return super().register_stage(stage, items,
            limit_usd=str(E1_LIMIT - Decimal(CARRY_KNOWN_COST)), max_requests=max_requests)


def preflight(bundle, *, online=False, opener=None):
    verify_execution(bundle)
    history = verify_history(bundle)
    report = compat.preflight(bundle, online=online, opener=opener)
    report["parameter_ready_to_send"] = report["ready_to_send"]
    report["history"] = history
    report["prior_known_e1_cost_usd"] = history["known_cost_usd"]
    remaining = E1_LIMIT - Decimal(history["known_cost_usd"])
    combined = Decimal(history["known_cost_usd"]) + Decimal(report["planned_full_e1_reserve_usd"])
    report.update(remaining_e1_budget_usd=str(remaining), combined_e1_reserve_usd=str(combined))
    if history["unaccepted_unknown_attempts"]:
        report["blocking_reasons"].append("new_historical_unknown_cost_requires_resolution")
    if parent.old_run._no_links(campaign_directory(bundle["root"])).exists():
        report["blocking_reasons"].append("new_campaign_already_exists")
    if combined > E1_LIMIT or remaining <= 0:
        report["blocking_reasons"].append("shared_e1_reservation_failed")
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
        directory = parent.old_run._no_links(campaign_directory(bundle["root"]))
        if directory.exists():
            raise RunError("new_campaign_already_exists")
        key = parent.old_run.credential(bundle["root"])
        directory.mkdir(parents=True, exist_ok=False)
        marker = {"campaign_id": CAMPAIGN_ID, "execution_seal": execution["seal_sha256"],
            "parent_campaign_ids": [*parent.HISTORY_IDS, PARENT_CAMPAIGN],
            "historical_unknown_cost_count": 2,
            "accepted_unknown_attempts": [list(a) for a in ACCEPTED_UNKNOWN_ATTEMPTS],
            "linked_inventory_sha256": report["linked_inventory_sha256"],
            "prior_known_e1_cost_usd": CARRY_KNOWN_COST,
            "remaining_e1_budget_usd": report["remaining_e1_budget_usd"],
            "response_carriers": ["native_tool", "content_json"]}
        previous.legacy.write_private(directory / "campaign.json", marker)
        previous.legacy.write_private(directory / "preflight.json", report)
        ledger = CarryLedger(directory / "campaign.sqlite3", campaign_id=CAMPAIGN_ID,
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
        state = ledger.summary()
        known = Decimal(CARRY_KNOWN_COST) + Decimal(state["known_cost_usd"])
        outcome = {"schema": "aria_e1_dual_carrier_outcome_v1", "campaign_id": CAMPAIGN_ID,
            "ok": result["status"] == "complete", "status": result["status"],
            "stop_reason": result["stop_reason"], "denominator": 204,
            "stage_counts": state["stages"]["e1"]["counts"],
            "new_known_cost_usd": state["known_cost_usd"], "cumulative_known_e1_cost_usd": str(known),
            "new_unknown_cost_count": state["unknown_cost_count"], "historical_unknown_cost_count": 2,
            "total_actual_cost_usd": None, "model_effect_measured": False, "no_automatic_retry": True}
        previous.legacy.write_private(directory / "outcome.aggregate.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
        return 0 if outcome["ok"] else 2
    except (Exception, KeyboardInterrupt) as error:
        if ledger is not None:
            try:
                ledger.stop("e1", "manual_stop" if isinstance(error, KeyboardInterrupt) else "external_error")
            except Exception:
                pass
        code = error.code if isinstance(error, RunError) else "carrier_run_failed"
        print(json.dumps({"ok": False, "reason": code, "no_automatic_retry": True,
                         "total_actual_cost_usd": None}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
