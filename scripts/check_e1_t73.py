"""Qualify the registered E1 wire repair without reading keys or sending inference."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_e1_t72 as prior  # noqa: E402
from scripts import run_e1_t71 as previous  # noqa: E402
from scripts import t73_policy as policy  # noqa: E402
from scripts.t7_transport import TransportError  # noqa: E402

REGISTRATION = "experiments/t73_20261001/amendment.json"
INVENTORY = "experiments/t73_20261001/requests.json"
EXECUTION = "experiments/t73_20261001/execution.json"
FILES = frozenset({REGISTRATION, INVENTORY, "scripts/check_e1_t73.py",
    "scripts/t73_policy.py", "scripts/t73_execution.py", "tests/test_t73_policy.py",
    "tests/test_t73_execution.py", "tests/test_e1_t73.py", "docs/E1_EXECUTION_T73.md"})
T72_SEAL = "814e433fff3ace7a7524b831e9e25567938be8429fbb986d164290a32cc99e33"
INHERITS = {**prior.INHERITS, "t72_seal": T72_SEAL}
WIRE_CHANGES = [{"field": "parallel_tool_calls", "from": False, "to": "omitted"}]


def verify_repair(bundle):
    root = bundle["root"]
    if prior.verify_repair(bundle)["seal_sha256"] != T72_SEAL:
        raise prior.QualificationError("prior_repair_changed")
    registration = previous.legacy.read_json(root / REGISTRATION)
    if (registration.get("schema") != "aria_t73_wire_repair_v1"
            or registration.get("inherits") != INHERITS
            or registration.get("wire_changes") != WIRE_CHANGES
            or registration.get("fixed_generation_positions") != 204
            or registration.get("budget_usd") != bundle["amendment"]["budget_usd"]
            or registration.get("unknown_cost_policy") != "retain_and_block"
            or registration.get("paid_entry_enabled") is not False):
        raise prior.QualificationError("wire_registration_changed")
    execution = previous.legacy.read_json(root / EXECUTION)
    if (execution.get("schema") != "aria_t73_wire_repair_seal_v1"
            or execution.get("inherits") != INHERITS
            or execution.get("seal_sha256") != previous.freeze.canonical_sha256(
                {k: v for k, v in execution.items() if k != "seal_sha256"})):
        raise prior.QualificationError("wire_repair_seal_changed")
    entries = execution.get("files", [])
    if len(entries) != len(FILES) or {e["path"] for e in entries} != FILES:
        raise prior.QualificationError("wire_repair_inventory_changed")
    for entry in entries:
        path = previous.freeze._public_file(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or prior._sha(path) != entry["sha256"]:
            raise prior.QualificationError("wire_repair_source_changed")
    expected = policy.prepare_e1(bundle)[3]
    inventory = previous.legacy.read_json(root / INVENTORY)
    if (inventory.get("schema") != "aria_t73_linked_inventory_v1"
            or inventory.get("items") != expected
            or inventory.get("items_sha256") != previous.freeze.canonical_sha256(expected)):
        raise prior.QualificationError("amended_request_inventory_changed")
    return execution


def build_report(bundle, *, online=False, opener=None):
    _, _, prepared, items = policy.prepare_e1(bundle)
    metadata = previous.legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    assessments = {}
    for trial in prepared.values():
        assessment = policy.assess_payload_support(model=trial["wire"]["model"],
            frozen_metadata=metadata, payload=trial["wire"],
            prompt_token_bound=trial["cost_plan"]["input_token_plan_bound"])
        policy.verify_payload_binding(trial["wire"], assessment)
        assessments[assessment["actual_payload_sha256"]] = assessment
    if any(i["request_sha256"] not in assessments for i in items):
        raise prior.QualificationError("amended_payload_unchecked")
    report = {"schema": "aria_t73_qualification_v1", "model_requests": 0,
        "credentials_read": False, "fixed_positions": len(items),
        "distinct_payloads_checked": len(assessments), "wire_changes": WIRE_CHANGES,
        "linked_inventory_sha256": previous.freeze.canonical_sha256(items),
        "missing_parameter_evidence": sorted({f for a in assessments.values() for f in a["missing_fields"]}),
        "offline_parameter_evidence_sufficient": all(a["supported"] for a in assessments.values()),
        "actual_required_fields": sorted({f for a in assessments.values() for f in a["required_fields"]}),
        "budget_usd": bundle["amendment"]["budget_usd"],
        "planned_full_e1_reserve_usd": str(sum((Decimal(i["reserve_usd"]) for i in items), Decimal(0))),
        "provider_cost_guarantee": False, "live_metadata_attempted": online,
        "live_metadata_passed": False, "parameter_repair_verified": False,
        "ready_to_send": False, "paid_entry_enabled": False, "blocking_reasons": []}
    if not report["offline_parameter_evidence_sufficient"]:
        report["blocking_reasons"].append("amended_payload_metadata_failed")
    if online:
        largest = max(prepared.values(), key=lambda t: t["cost_plan"]["input_token_plan_bound"])
        try:
            report["live_metadata"] = policy.fetch_metadata(model=largest["wire"]["model"],
                frozen_metadata=metadata, payload=largest["wire"],
                prompt_token_bound=largest["cost_plan"]["input_token_plan_bound"],
                opener=opener, enable_network=True)
            report["live_metadata_passed"] = True
        except TransportError as exc:
            report["live_metadata"] = exc.receipt
            report["live_metadata_error"] = exc.code
            report["blocking_reasons"].append("live_metadata_gate_failed")
    else:
        report["blocking_reasons"].append("live_metadata_not_checked")
    report["parameter_repair_verified"] = (report["offline_parameter_evidence_sufficient"]
                                          and report["live_metadata_passed"])
    # The new parameters never replace the inventory used to verify the old attempt.
    try:
        state = prior.inspect_campaign(bundle, previous.prepare_e1(bundle)[3])
        registration = previous.legacy.read_json(bundle["root"] / REGISTRATION)
        if state != registration["prior_campaign"]:
            raise prior.QualificationError("registered_prior_evidence_changed")
        report["prior_campaign"] = state
        if state["unknown_cost_count"]:
            report["blocking_reasons"].append("prior_cost_unresolved")
        if state["outstanding_count"]:
            report["blocking_reasons"].append("prior_outcome_unresolved")
        # A historical NULL is not made safe by editing this read-only report.
        if not state["unknown_cost_count"]:
            report["blocking_reasons"].append("attributable_reconciliation_not_registered")
    except Exception:
        report["blocking_reasons"].append("prior_campaign_evidence_unavailable_or_invalid")
    report["historical_cost_carry_usd"] = None
    report["combined_budget_qualified"] = False
    report["judge"] = {"eligible": False, "reason": "requires_complete_verified_generation",
                       "full_actual_text_reservation_required": True, "budget_usd": "0.5"}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-preflight", action="store_true",
                        help="Two public metadata GETs only; never sends a model request")
    args = parser.parse_args(argv)
    try:
        bundle = previous.load_bundle()
        seal = verify_repair(bundle)
        report = build_report(bundle, online=args.online_preflight)
        report.update(ok=True, repair_seal=seal["seal_sha256"])
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 2 if report["blocking_reasons"] else 0
    except Exception:
        print(json.dumps({"ok": False, "ready_to_send": False, "model_requests": 0,
                          "reason": "wire_repair_check_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
