"""Explicit E1c prepare, generate, qualify and first-label workflow.

No command invents a source, grants spending authority or expands a budget.
The only credential reader is reached inside the fully authorized real-mode
execute callback, after immutable source/history/price and start checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1c_execution as execution
from scripts import e1c_qualification as qualification
from scripts import e1c_scoring as scoring
from scripts import e1b_evaluation as common


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--campaign", type=Path, required=True)
    command.add_argument("--spec", type=Path, required=True,
                         help="Explicit first-source, closed-history, policy, seed and proposed limits JSON")
    for name in ("verify", "authorize-template", "execute", "qualify", "export", "assess"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name != "authorize-template":
            command.add_argument("--authorization-sha256", required=True)
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
        if name in {"export", "assess"}:
            command.add_argument("--bundle", type=Path, required=True)
        if name == "assess":
            command.add_argument("--bundle-sha256", required=True)
        if name == "export":
            command.add_argument("--blind-seed", type=int, required=True)
    command = commands.add_parser("import-first-labels")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--bundle-sha256", required=True)
    command.add_argument("--kind", choices=("reply", "formation"), required=True)
    command.add_argument("--labels", type=Path, required=True)
    command.add_argument("--batch-id", required=True)
    command.add_argument("--expected-ids", type=Path)
    return result


def run(args):
    if args.command == "prepare":
        spec = common.read_json(execution.regular(args.spec))
        if spec.get("mode", "real_model") != "real_model":
            raise ValueError("e1c_cli_is_not_a_software_transport")
        frozen = execution.prepare(args.campaign, **spec)
        return {"status": "prepared_not_authorized", "registration_sha256": frozen["seal_sha256"],
                "new_allowance_usd": frozen["new_allowance_usd"]}
    if args.command == "authorize-template":
        return execution.authorize_template(common.read_json(args.campaign / "registration.json"))
    if args.command == "import-first-labels":
        return scoring.import_first_batch(args.bundle, args.bundle_sha256,
            execution.regular(args.labels).read_bytes(), kind=args.kind, batch_id=args.batch_id,
            expected_ids=common.read_json(execution.regular(args.expected_ids)) if args.expected_ids else None)
    frozen, source = execution.verify_authority(args.campaign, args.authorization_sha256,
                                               for_dispatch=args.command == "execute")
    if args.command == "verify":
        return {"status": "verified", "registration_sha256": frozen["seal_sha256"],
                "model_calls": 0, "credential_reads": 0}
    if args.command == "execute":
        if frozen["mode"] != "real_model" or not args.allow_local_env:
            raise ValueError("e1c_explicit_real_mode_and_local_key_flag_required")
        # This existing reader extracts only OPENROUTER_API_KEY; it is not run
        # by prepare/verify/qualify/scoring and does not update the environment.
        from scripts.run_ctest import local_key
        value = execution.execute(args.campaign, authorization_sha256=args.authorization_sha256,
                                  credential_loader=lambda: local_key(ROOT / ".env"))
        return {"status": value["status"], "accounting": {
            name: value["accounting"][name] for name in ("calls_used", "known_cost_usd", "unknown_cost_requests")}}
    if args.command == "qualify":
        value = qualification.qualify(args.campaign, authorization_sha256=args.authorization_sha256)
        return {"status": value["status"], "fixed_reply_positions": 216, "fixed_formation_cases": 12,
                "effect_claim_eligible": False}
    if args.command == "export":
        inventory = qualification.qualify(args.campaign, authorization_sha256=args.authorization_sha256, write=False)
        path = args.campaign / "qualification.json"
        if common.read_json(path) != inventory:
            raise ValueError("e1c_saved_qualification_changed")
        receipt = common.read_json(args.campaign / "execution-receipt.json")
        return scoring.export_bundle(receipt["run_root"], source, inventory, args.bundle,
            qualification_path=path, qualification_sha256=execution.sha(path),
            evaluation_protocol=frozen["evaluation_protocol"],
            registered_protocol_sha256=frozen["evaluation_protocol_sha256"], blind_seed=args.blind_seed)
    value = scoring.assess(args.campaign, args.bundle, args.bundle_sha256,
                           authorization_sha256=args.authorization_sha256, write=True)
    return {name: value.get(name) for name in ("status", "main_gate", "formation_gate", "overall_gate", "effect_claim_eligible")}


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
