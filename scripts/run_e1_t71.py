"""T7.1 explicit execution amendment; old T6/T7 files stay unchanged."""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1_experiment as legacy  # noqa: E402
from scripts import t7_judge as judge  # noqa: E402
from scripts import t71_policy as policy  # noqa: E402
from scripts import verify_experiment_freeze as freeze  # noqa: E402
from scripts.t7_transport import OpenRouterTransport, TransportError  # noqa: E402
from scripts.t7_ledger import LedgerError  # noqa: E402

AMENDMENT = "experiments/t71_20260930/amendment.json"
EXECUTION = "experiments/t71_20260930/execution.json"
FILES = {AMENDMENT, "experiments/t71_20260930/validation.t6.md",
         "scripts/run_e1_t71.py", "scripts/t71_policy.py",
         "tests/test_e1_t71.py", "tests/test_t71_policy.py", "docs/E1_EXECUTION_T71.md"}
T6_SEAL = "f020b5079bdd1c6ccaf68409af0728e99cb305d524ff2f1b16b69cc41e3a7212"
T7_SEAL = "e49db651619b6ccb23a87efae1c453f28ad5eef2f6d774160ed7ce9e1d741538"


def load_bundle(root=ROOT):
    """Verify the original T6 bytes, one registered document revision, and T7."""
    root = Path(root).absolute()
    amendment = legacy.read_json(root / AMENDMENT)
    manifest = legacy.read_json(root / legacy.T6_PATH)
    plan = legacy.read_json(root / legacy.PLAN_PATH)
    if (amendment["id"] != "t71_20260930_v1"
            or amendment["campaign_id"] != legacy.CAMPAIGN_ID
            or amendment["inherits"] != {"t6_seal": T6_SEAL, "t7_seal": T7_SEAL}
            or amendment["budget_usd"] != {"e1": "0.5", "judge": "0.5", "pause_total": "2", "total": "5"}
            or manifest["seal_sha256"] != T6_SEAL
            or freeze.canonical_sha256({k: v for k, v in manifest.items() if k != "seal_sha256"}) != T6_SEAL):
        raise legacy.ExecutionError("amendment_or_original_seal_changed")
    revision = amendment["document_revision"]
    if (revision["path"] != "docs/VALIDATION.md"
            or revision["original_reference"] != "experiments/t71_20260930/validation.t6.md"):
        raise legacy.ExecutionError("unregistered_document_revision")
    if (freeze.file_sha256(root / revision["path"]) != revision["current_sha256"]
            or (root / revision["path"]).stat().st_size != revision["current_bytes"]):
        raise legacy.ExecutionError("registered_document_changed")
    entries = manifest["files"]
    if freeze._source_inventory(root) != {e["path"] for e in entries if e["path"].startswith("aria/")}:
        raise legacy.ExecutionError("original_source_inventory_changed")
    for entry in entries:
        selected = revision["original_reference"] if entry["path"] == revision["path"] else entry["path"]
        path = freeze._public_file(root, selected)
        if path.stat().st_size != entry["bytes"] or freeze.file_sha256(path) != entry["sha256"]:
            raise legacy.ExecutionError("original_frozen_file_changed")
        if entry["path"] == revision["path"] and entry["sha256"] != revision["original_sha256"]:
            raise legacy.ExecutionError("original_document_hash_changed")
    records = freeze.e1_request_records(plan)
    if (len(records) != 204 or records != manifest["e1_requests"]
            or freeze.canonical_sha256(records) != manifest["e1_requests_sha256"]):
        raise legacy.ExecutionError("frozen_request_inventory_changed")
    actual = {"python": sys.version.split()[0], "pydantic": importlib.metadata.version("pydantic"),
              "pyyaml": importlib.metadata.version("pyyaml")}
    if any(plan["environment"][key] != value for key, value in actual.items()):
        raise legacy.ExecutionError("frozen_environment_changed")
    contract = legacy.read_json(root / plan["judge"]["contract_path"])
    bundle = {"root": root, "plan": plan, "manifest": manifest, "contract": contract,
              "amendment": amendment, "baseline": {"seal_sha256": T6_SEAL,
              "verified_files": len(entries), "e1_requests": len(records),
              "original_document_reference_used": revision["original_reference"]}}
    old = legacy.verify_execution(bundle)
    if old["seal_sha256"] != T7_SEAL:
        raise legacy.ExecutionError("original_t7_seal_changed")
    return bundle


def verify_execution(bundle):
    value = legacy.read_json(bundle["root"] / EXECUTION)
    if (value.get("schema") != "aria_t71_execution_v1"
            or value.get("inherits") != {"t6_seal": T6_SEAL, "t7_seal": T7_SEAL}
            or value.get("seal_sha256") != freeze.canonical_sha256({k: v for k, v in value.items() if k != "seal_sha256"})):
        raise legacy.ExecutionError("t71_execution_seal_changed")
    entries = value["files"]
    if len(entries) != len(FILES) or {e["path"] for e in entries} != FILES:
        raise legacy.ExecutionError("t71_inventory_changed")
    for entry in entries:
        path = freeze._public_file(bundle["root"], entry["path"])
        if path.stat().st_size != entry["bytes"] or freeze.file_sha256(path) != entry["sha256"]:
            raise legacy.ExecutionError("t71_source_changed")
    return value


def prepare_e1(bundle):
    suite, cases, prepared, items = legacy.prepare_e1(bundle)
    for trial in prepared.values():
        trial["cost_plan"] = policy.estimate_request(trial["wire"], input_price="0.00000003", output_price="0.00000013")
        if (trial["cost_plan"]["input_token_plan_bound"] > bundle["plan"]["e1"]["max_prompt_tokens"]
                or Decimal(trial["cost_plan"]["reserved_cost_usd"]) > Decimal("0.0011")):
            raise legacy.ExecutionError("e1_reservation_insufficient")
    for item in items:
        item["input_token_plan_bound"] = prepared[item["case_id"], item["arm"]]["cost_plan"]["input_token_plan_bound"]
    return suite, cases, prepared, items


def preflight(bundle, *, online=False):
    _, _, prepared, items = prepare_e1(bundle)
    largest = max(prepared.values(), key=lambda p: p["cost_plan"]["input_token_plan_bound"])
    result = {"stage": "e1", "e1_requests": len(items), "model_requests": 0,
              "full_reserved_cost_usd": "0.2244", "stage_limit_usd": "0.5",
              "max_input_token_plan_bound": largest["cost_plan"]["input_token_plan_bound"],
              "bound_status": "authorized_planning_assumption_with_post_usage_stop",
              "provider_hard_bound_verified": False, "live_metadata_checked": False,
              "parallel_tool_calls_support": "unverified_single_native_tool_receipt_required",
              "blocking_reasons": []}
    if online:
        result["metadata"] = policy.fetch_metadata(model=largest["wire"]["model"],
            frozen_metadata=legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json"),
            payload=largest["wire"], prompt_token_bound=largest["cost_plan"]["input_token_plan_bound"], enable_network=True)
        result["live_metadata_checked"] = True
    result["ready_to_send"] = True
    return result


class CheckedTransport:
    def __init__(self, transport, bounds):
        self.transport, self.bounds = transport, bounds

    def request(self, payload):
        digest = freeze.canonical_sha256(payload)
        if digest not in self.bounds:
            raise TransportError("request_changed", outcome_unknown=False)
        envelope = self.transport.request(payload)
        policy.validate_receipt(payload, envelope, prompt_token_bound=self.bounds[digest])
        return envelope


def run_generation(directory, ledger, bundle, transport, *, source_check, sleep=time.sleep):
    # Reuse unchanged T7 control flow with a private globals map, never monkeypatch
    # the original module or its frozen default byte-estimation policy.
    function = legacy.run_generation
    adapted = FunctionType(function.__code__, {**function.__globals__, "prepare_e1": prepare_e1},
                           function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    _, _, _, items = prepare_e1(bundle)
    checked = CheckedTransport(transport, {i["request_sha256"]: i["input_token_plan_bound"] for i in items})
    return adapted(directory, ledger, bundle, checked, source_check=source_check, sleep=sleep)


def open_campaign(bundle, execution):
    directory = legacy.campaign_directory(bundle["root"])
    marker, database = directory / "campaign.json", directory / "campaign.sqlite3"
    for path in (directory, *directory.parents, marker, database, database.with_name(database.name + ".stopped")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise legacy.ExecutionError("private_link_rejected")
    binding = {"campaign_id": legacy.CAMPAIGN_ID, "t6_seal": T6_SEAL,
               "t7_seal": T7_SEAL, "execution_seal": execution["seal_sha256"]}
    if directory.exists():
        if not marker.is_file() or not database.is_file() or legacy.read_json(marker) != binding:
            raise legacy.ExecutionError("existing_campaign_requires_reconciliation")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        legacy.write_private(marker, binding)
    return directory, policy.CampaignLedger(database, campaign_id=legacy.CAMPAIGN_ID,
                                             freeze_seal=execution["seal_sha256"])


def credential(*, allow_local_env=False):
    if allow_local_env:
        return legacy.credential(ROOT)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or any(c.isspace() for c in key):
        raise legacy.ExecutionError("credential_not_in_environment")
    return key


def run_judge(directory, ledger, bundle, prepared, rows, suite, transport, *, source_check, sleep=time.sleep):
    """The T7 loop with only the explicitly amended stage limit supplied."""
    class LimitAdapter:
        def __getattr__(self, name):
            return getattr(ledger, name)

        def register_stage(self, stage, items, *, limit_usd, max_requests):
            if stage != "judge" or limit_usd != "0.3" or max_requests != 35:
                raise legacy.ExecutionError("unexpected_legacy_judge_contract")
            return ledger.register_stage(stage, items, limit_usd="0.5", max_requests=35)

    checked = CheckedTransport(transport, {b["payload_sha256"]: b["input_token_upper_bound"] for b in prepared["batches"]})
    return legacy.run_judge(directory, LimitAdapter(), bundle, prepared, rows, suite, checked,
                            source_check=source_check, sleep=sleep)


def compact_result(result, stage):
    campaign = result["campaign"]
    return {"stage": stage, "status": result["status"], "stop_reason": result["stop_reason"],
            "known_cost_usd": campaign["known_cost_usd"], "unknown_cost_count": campaign["unknown_cost_count"],
            "outstanding_count": campaign["outstanding_count"],
            "stage_counts": {k: v["counts"] for k, v in campaign["stages"].items()},
            "denominator": 204, "model_effect_measured": False}


def judge_stage(bundle, execution, *, execute=False, online=False, allow_local_env=False):
    directory = legacy.campaign_directory(bundle["root"])
    if not (directory / "e1" / "results.private.json").is_file():
        raise legacy.ExecutionError("judge_generation_artifacts_required")
    directory, ledger = open_campaign(bundle, execution)
    rows = legacy.load_generation(directory, ledger, bundle)
    suite, _, _, _ = prepare_e1(bundle)
    prepared = policy.build_batches(rows, suite, bundle["plan"], bundle["contract"], legacy.committed_seeds(bundle))
    sample, empty = legacy.committed_human_records(bundle)
    human = judge.prepare_human_form(rows, suite, sample, empty, prepared)
    report = {"stage": "judge", "planned_batches": len(prepared["batches"]),
              "full_reserved_cost_usd": prepared["total_reserve_usd"], "stage_limit_usd": "0.5",
              "model_requests": 0, "ready_to_send": prepared["status"] == "prepared" and bool(prepared["batches"])}
    if online and report["ready_to_send"]:
        largest = max(prepared["batches"], key=lambda b: b["input_token_upper_bound"])
        report["metadata"] = policy.fetch_metadata(model=largest["payload"]["model"],
            frozen_metadata=legacy.read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json"),
            payload=largest["payload"], prompt_token_bound=largest["input_token_upper_bound"], enable_network=True)
    if execute:
        # Store reviewable full inventory and the empty human form even if budget
        # prevents any judge call. Never overwrite or reroll its fixed IDs.
        path = directory / "judge-t71-preflight.private.json"
        if path.exists():
            raise legacy.ExecutionError("judge_preflight_already_saved")
        legacy.write_private(path, {"prepared": prepared, "human_form": human, "readiness": report})
        if not report["ready_to_send"]:
            return {"ok": False, "status": "blocked_budget", **report}
        result = run_judge(directory, ledger, bundle, prepared, rows, suite,
            OpenRouterTransport(credential(allow_local_env=allow_local_env), enable_network=True),
            source_check=lambda: verify_execution(load_bundle()))
        return {"ok": result["status"] == "complete", **compact_result(result, "judge")}
    return {"ok": True, **report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e1", "judge"), default="e1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--online-preflight", action="store_true")
    parser.add_argument("--allow-local-env", action="store_true", help="Only with explicit user permission to read local .env; never prints its value")
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle()
        execution = verify_execution(bundle)
        if args.stage == "judge":
            report = judge_stage(bundle, execution, execute=args.execute,
                online=args.online_preflight or args.execute, allow_local_env=args.allow_local_env)
        else:
            report = preflight(bundle, online=args.online_preflight or args.execute)
            if args.execute:
                if not report["ready_to_send"]:
                    raise legacy.ExecutionError("preflight_not_ready")
                key = credential(allow_local_env=args.allow_local_env)
                directory, ledger = open_campaign(bundle, execution)
                legacy.write_private(directory / "e1-t71-preflight.json", report)
                result = run_generation(directory, ledger, bundle, OpenRouterTransport(key, enable_network=True),
                    source_check=lambda: verify_execution(load_bundle()))
                report = {"ok": result["status"] == "complete", **compact_result(result, "e1")}
            else:
                report["ok"] = True
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 2
    except (ValueError, OSError, LedgerError, KeyError) as exc:
        code = getattr(exc, "code", str(exc) if isinstance(exc, legacy.ExecutionError) else type(exc).__name__)
        print(json.dumps({"ok": False, "reason": code, "no_automatic_retry": True}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
