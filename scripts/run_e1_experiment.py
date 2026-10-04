"""Frozen E1 execution; offline by default, fixed private ledger, no retries.

This entry does not enable the product, CPED, TTS or notifications. Its source
revision is sealed separately from the unchanged T6 experimental protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_experiment_freeze import (  # noqa: E402
    canonical_sha256, file_sha256, verify as verify_t6,
)
from scripts.t7_ledger import CampaignLedger, LedgerError  # noqa: E402
from scripts.t7_transport import (  # noqa: E402
    OpenRouterTransport, TransportError, estimate_request, fetch_metadata,
)

EXECUTION_PATH = "experiments/t7_20260930/execution.json"
PLAN_PATH = "experiments/t6_20260930/plan.json"
T6_PATH = "experiments/t6_20260930/manifest.json"
CAMPAIGN_ID = "t6_20260930_v1"
IMPLEMENTATION_FILES = (
    "scripts/run_e1_experiment.py", "scripts/t7_ledger.py",
    "scripts/t7_transport.py", "scripts/t7_judge.py",
)


class ExecutionError(ValueError):
    pass


def read_json(path):
    from aria.bench.provenance import strict_json
    return strict_json(Path(path).read_text(encoding="utf-8"))


def write_private(path, value):
    """Write an immutable artifact before any following dependent action."""
    path = Path(path)
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (path, *path.parents)):
        raise ExecutionError("private_link_rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def load_bundle(root=ROOT):
    root = Path(root)
    baseline = verify_t6(root)
    plan = read_json(root / PLAN_PATH)
    manifest = read_json(root / T6_PATH)
    contract = read_json(root / plan["judge"]["contract_path"])
    if plan["id"] != CAMPAIGN_ID or contract["freeze_id"] != CAMPAIGN_ID:
        raise ExecutionError("freeze_identity_changed")
    expected = plan["environment"]
    actual = {"python": sys.version.split()[0],
              "pydantic": importlib.metadata.version("pydantic"),
              "pyyaml": importlib.metadata.version("pyyaml")}
    if any(actual[key] != expected[key] for key in actual):
        raise ExecutionError("frozen_environment_changed")
    return {"root": root, "plan": plan, "manifest": manifest,
            "contract": contract, "baseline": baseline}


def verify_execution(bundle):
    root = bundle["root"]
    value = read_json(root / EXECUTION_PATH)
    sealed = {k: v for k, v in value.items() if k != "seal_sha256"}
    if (value.get("schema") != "aria_t7_execution_v1"
            or canonical_sha256(sealed) != value.get("seal_sha256")
            or value.get("t6_seal_sha256") != bundle["baseline"]["seal_sha256"]
            or value.get("campaign_id") != CAMPAIGN_ID):
        raise ExecutionError("execution_seal_mismatch")
    records = value.get("files", [])
    allowed = set(IMPLEMENTATION_FILES) | {
        "tests/test_t7_ledger.py", "tests/test_t7_transport.py",
        "tests/test_t7_judge.py", "tests/test_e1_execution.py",
        "docs/E1_EXECUTION.md",
        "experiments/t7_20260930/preflight.json",
    }
    if (not isinstance(records, list)
            or {r.get("path") for r in records} != allowed
            or len(records) != len(allowed)):
        raise ExecutionError("execution_file_inventory_changed")
    for row in records:
        path = root / row["path"]
        if (path.stat().st_size != row["bytes"]
                or file_sha256(path) != row["sha256"]):
            raise ExecutionError("execution_source_changed")
    return value


def prepare_e1(bundle):
    from aria.bench.refocus_cases import load_refocus_cases
    from aria.bench.refocus_e1_runner import E1LocalConfig, prepare_trial
    plan = bundle["plan"]
    suite = load_refocus_cases()
    cases = {case.id: case for case in suite.main_cases()}
    config = E1LocalConfig(**plan["e1"]["preparation"])
    prepared, items = {}, []
    for record in bundle["manifest"]["e1_requests"]:
        key = record["case_id"], record["arm"]
        if key not in prepared:
            trial = prepare_trial(cases[key[0]], key[1], config)
            wire = deepcopy(trial["payload"])
            wire.pop("max_completion_tokens")
            wire.update(deepcopy(plan["e1"]["wire"]))
            trial["wire"] = wire
            trial["cost_plan"] = estimate_request(
                wire, input_price="0.00000003", output_price="0.00000013")
            if (trial["cost_plan"]["input_token_plan_bound"] > plan["e1"]["max_prompt_tokens"]
                    or Decimal(trial["cost_plan"]["reserved_cost_usd"])
                    > Decimal(str(plan["e1"]["per_request_reserve_usd"]))):
                raise ExecutionError("e1_token_or_cost_reservation_failed")
            prepared[key] = trial
        trial = prepared[key]
        if canonical_sha256(trial["wire"]) != record["request_sha256"]:
            raise ExecutionError("frozen_request_changed")
        items.append({"id": f"e1_{record['order']:03d}", **record,
                      "reserve_usd": str(plan["e1"]["per_request_reserve_usd"]),
                      "input_token_plan_bound": trial["cost_plan"]["input_token_plan_bound"]})
    return suite, cases, prepared, items


def generation_row(record):
    return {k: record[k] for k in ("order", "case_id", "arm", "repeat")} | {
        "status": "not_started", "raw_decision": None,
        "delivered_response": None, "actual_model": None,
        "latency_ms": None, "request_fingerprint": record["request_sha256"],
    }


def parse_generation(envelope, trial, case):
    from aria.bench.refocus_e1_runner import _visible_payload
    from aria.bench.prompt_footprint import measure_output_fields
    from aria.engine.policy.online_decision import apply_single_pass_decision
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ExecutionError("invalid_schema")
    choice = choices[0]
    if choice.get("finish_reason") not in {"tool_calls", "stop"}:
        raise ExecutionError("truncated_output")
    message = choice.get("message") or {}
    calls = message.get("tool_calls")
    if (message.get("content") not in (None, "")
            or not isinstance(calls, list) or len(calls) != 1
            or not isinstance(calls[0], dict)):
        raise ExecutionError("invalid_schema")
    call = calls[0].get("function") or {}
    if call.get("name") != "aria_turn_decision":
        raise ExecutionError("invalid_schema")
    raw = call.get("arguments")
    if isinstance(raw, dict):
        raw = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    if not isinstance(raw, str):
        raise ExecutionError("invalid_schema")
    visible = _visible_payload(raw)
    ctx, persona = deepcopy(trial["ctx"]), deepcopy(trial["persona"])
    delivered = apply_single_pass_decision(raw, ctx, case.stimulus.user_input, persona)
    delivered = guard_external_delivery_response(ctx, delivered, case.stimulus.user_input, persona)
    trace = ctx.working_memory.get("online_turn_decision") or {}
    result = {
        "raw_decision": visible, "delivered_response": delivered,
        "status": "generated" if trace.get("status") == "parsed" else "invalid_protocol",
        "deliberation_status": trace.get("deliberation_status", "not_applicable"),
        "selected_in_deliberated_options": trace.get("selected_in_deliberated_options"),
        "selected_intervention_id": trace.get("selected_intervention_id"),
        "rule_response_changed": delivered != visible["response"],
        "output_measurement": measure_output_fields([visible]),
    }
    return result


def finish_receipt(ledger, stage, identifier, envelope, digest, error=None):
    usage = envelope.get("usage") or {}
    return ledger.finish(stage, identifier,
        cost_usd=usage.get("cost"), prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        actual_model=envelope.get("model"), provider=envelope.get("provider"),
        response_sha256=digest, error=error)


def run_generation(directory, ledger, bundle, transport, *, source_check, sleep=time.sleep):
    suite, cases, prepared, items = prepare_e1(bundle)
    rows = [generation_row(item) for item in items]
    stage_dir = Path(directory) / "e1"
    stage_dir.mkdir(exist_ok=False)
    ledger.register_stage("e1", [{k: item[k] for k in ("id", "request_sha256", "reserve_usd")}
                                for item in items], limit_usd="0.5", max_requests=204)
    try:
        write_private(stage_dir / "inventory.json", items)
    except Exception:
        ledger.stop("e1", "result_save_failed")
        raise
    stopped = None
    for item, row in zip(items, rows):
        if stopped:
            row["reason"] = stopped
            continue
        trial = prepared[item["case_id"], item["arm"]]
        envelope, error, digest = {}, None, None
        try:
            source_check()
            if canonical_sha256(trial["wire"]) != item["request_sha256"]:
                raise ExecutionError("request_changed")
            write_private(stage_dir / f"{item['id']}.request.json", trial["wire"])
            ledger.claim("e1", item["id"])
        except Exception:
            stopped = "source_or_claim_failed"
            ledger.stop("e1", "source_changed")
            row["reason"] = stopped
            continue
        started = time.perf_counter()
        try:
            envelope = transport.request(trial["wire"])
            usage = envelope["usage"]
            if (usage["prompt_tokens"] > item["input_token_plan_bound"]
                    or usage["completion_tokens"] > trial["wire"]["max_tokens"]):
                raise ExecutionError("token_bound_exceeded")
            row.update(parse_generation(envelope, trial, cases[item["case_id"]]))
            if row["status"] != "generated":
                error = "invalid_schema"
        except TransportError as exc:
            envelope = getattr(exc, "receipt", {}) or {}
            error = exc.code
            row.update(status="generation_failed", reason=error)
        except Exception as exc:
            error = str(exc) if isinstance(exc, ExecutionError) else "invalid_schema"
            row.update(status="generation_failed", reason=error)
        row.update(latency_ms=round((time.perf_counter() - started) * 1000, 3),
                   actual_model=envelope.get("model"), provider=envelope.get("provider"),
                   usage=envelope.get("usage", {}), input_measurement=trial["input_measurement"])
        try:
            digest = write_private(stage_dir / f"{item['id']}.receipt.json", {
                "envelope": envelope, "error": error, "row": row})
        except Exception:
            error = "result_save_failed"
        try:
            finish_receipt(ledger, "e1", item["id"], envelope, digest, error)
        except LedgerError as exc:
            error = exc.code
        if error:
            row.update(status="generation_failed", reason=error)
        if error or ledger.summary().get("stopped"):
            stopped = error or ledger.summary().get("stop_reason") or "campaign_stopped"
        else:
            sleep(bundle["plan"]["transport"]["min_interval_seconds"])
    result = {"schema": "aria_e1_live_result_v1", "denominator": 204,
              "status": "complete" if not stopped else "incomplete", "stop_reason": stopped,
              "rows": rows, "campaign": ledger.summary(),
              "model_effect_measured": False, "quality_review": "pending",
              "fixture_origin": "author_exposed_fictional_not_learned_preferences"}
    try:
        write_private(stage_dir / "results.private.json", result)
    except Exception:
        ledger.stop("e1", "result_save_failed")
        raise
    return result


def campaign_directory(root=ROOT):
    return Path(root).parent / "aria-private-experiments" / CAMPAIGN_ID


def open_campaign(bundle, execution):
    directory = campaign_directory(bundle["root"])
    binding = {"campaign_id": CAMPAIGN_ID, "t6_seal": bundle["baseline"]["seal_sha256"],
               "execution_seal": execution["seal_sha256"]}
    marker = directory / "campaign.json"
    database = directory / "campaign.sqlite3"
    paths = (directory, *directory.parents, marker, database,
             database.with_name(database.name + ".stopped"))
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in paths):
        raise ExecutionError("private_link_rejected")
    if directory.exists():
        if not marker.is_file() or not database.is_file() or read_json(marker) != binding:
            raise ExecutionError("existing_campaign_requires_reconciliation")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_private(marker, binding)
    ledger = CampaignLedger(database, campaign_id=CAMPAIGN_ID,
                            freeze_seal=execution["seal_sha256"])
    return directory, ledger


def credential(root=ROOT):
    """Only the explicitly executed stage may read this one credential."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        path = Path(root) / ".env"
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 65536:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                name, sep, value = line.strip().partition("=")
                if sep and name.strip() == "OPENROUTER_API_KEY":
                    key = value.strip().strip('"').strip("'")
                    break
    if not key or any(c.isspace() for c in key):
        raise ExecutionError("credential_unavailable")
    return key


def preflight_e1(bundle, *, online=False):
    _, _, prepared, items = prepare_e1(bundle)
    largest = max(prepared.values(), key=lambda p: p["cost_plan"]["input_token_plan_bound"])
    metadata = read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json")
    result = {"e1_requests": len(items), "model_requests": 0,
              "full_reserved_cost_usd": "0.2244", "stage_limit_usd": "0.5",
              "max_input_token_plan_bound": largest["cost_plan"]["input_token_plan_bound"],
              "provider_hard_token_bound_verified": False,
              "live_metadata_checked": False,
              "blocking_reasons": ["provider_input_token_bound_unverified"]}
    if online:
        try:
            result["metadata"] = fetch_metadata(model=bundle["plan"]["e1"]["wire"]["model"],
                frozen_metadata=metadata, payload=largest["wire"],
                prompt_token_bound=largest["cost_plan"]["input_token_plan_bound"], enable_network=True)
            result["live_metadata_checked"] = True
        except TransportError as exc:
            result["metadata"] = exc.receipt
            result["blocking_reasons"].append(exc.code)
    result["ready_to_send"] = not result["blocking_reasons"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e1", "judge"), default="e1")
    parser.add_argument("--execute", action="store_true", help="Explicit paid stage under frozen limits")
    parser.add_argument("--online-preflight", action="store_true", help="Public unauthenticated metadata only")
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle()
        execution = verify_execution(bundle)
        if args.stage == "judge":
            return judge_main(bundle, execution, execute=args.execute,
                              online=args.online_preflight or args.execute)
        report = preflight_e1(bundle, online=args.online_preflight or args.execute)
        if args.execute:
            if not report["ready_to_send"]:
                print(json.dumps({"ok": False, "status": "blocked_before_credentials", **report}, ensure_ascii=False))
                return 2
            key = credential()
            directory, ledger = open_campaign(bundle, execution)
            write_private(directory / "e1-preflight.json", report)
            transport = OpenRouterTransport(key, enable_network=True)
            result = run_generation(directory, ledger, bundle, transport,
                source_check=lambda: verify_execution(load_bundle()))
            report = {"status": result["status"], "denominator": 204,
                      "generated": sum(r["status"] == "generated" for r in result["rows"]),
                      "stop_reason": result["stop_reason"], "campaign": result["campaign"]}
        print(json.dumps({"ok": report.get("status", "complete") == "complete", **report}, ensure_ascii=False))
        return 0 if report.get("status", "complete") == "complete" else 2
    except (ExecutionError, LedgerError, TransportError, OSError, ValueError, KeyError) as exc:
        code = getattr(exc, "code", str(exc) if isinstance(exc, ExecutionError) else type(exc).__name__)
        print(json.dumps({"ok": False, "reason": code, "no_automatic_retry": True}, ensure_ascii=False))
        return 1


def committed_seeds(bundle):
    path = (bundle["root"].parent / "aria-archive" / "20260930-02-t6"
            / "private" / "blinding-seeds.json")
    if file_sha256(path) != bundle["plan"]["judge"]["private_seed_commitment_sha256"]:
        raise ExecutionError("judge_seed_commitment_changed")
    return read_json(path)


def committed_human_records(bundle):
    private = bundle["root"].parent / "aria-archive" / "20260930-02-t6" / "private"
    records = []
    for name, field in (("human_sample.private.json", "human_sample_positions_commitment_sha256"),
                        ("human_review.empty.json", "human_empty_form_sha256")):
        path = private / name
        if file_sha256(path) != bundle["plan"]["judge"][field]:
            raise ExecutionError("human_sample_commitment_changed")
        records.append(read_json(path))
    return records


def load_generation(directory, ledger, bundle):
    """Bind every judged row to a completed claim and its immutable receipt."""
    source = Path(directory) / "e1"
    result = read_json(source / "results.private.json")
    state = ledger.summary()
    if result.get("status") != "complete" or state["stopped"]:
        raise ExecutionError("judge_requires_complete_generation")
    _, _, _, expected = prepare_e1(bundle)
    items = state.get("stages", {}).get("e1", {}).get("items", [])
    rows = result.get("rows", [])
    if len(items) != 204 or len(rows) != 204:
        raise ExecutionError("judge_requires_all_generation_receipts")
    for item, row, planned in zip(items, rows, expected):
        receipt_path = source / f"{planned['id']}.receipt.json"
        request_path = source / f"{planned['id']}.request.json"
        if (item["id"] != planned["id"] or item["status"] != "completed"
                or item["request_sha256"] != planned["request_sha256"]
                or canonical_sha256(read_json(request_path)) != planned["request_sha256"]
                or file_sha256(receipt_path) != item["response_sha256"]):
            raise ExecutionError("generation_evidence_changed")
        receipt = read_json(receipt_path)
        if receipt.get("error") or receipt.get("row") != row or row.get("status") != "generated":
            raise ExecutionError("generation_evidence_changed")
        if any(row.get(k) != planned[k] for k in ("order", "case_id", "arm", "repeat")):
            raise ExecutionError("generation_coordinates_changed")
    return rows


def judge_content(envelope, batch):
    from scripts.t7_judge import parse_batch
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ExecutionError("invalid_schema")
    choice = choices[0]
    message = choice.get("message") or {}
    if choice.get("finish_reason") != "stop":
        raise ExecutionError("truncated_output")
    if message.get("tool_calls"):
        raise ExecutionError("invalid_schema")
    content = message.get("content")
    parse_batch(batch, content)
    return content


def run_judge(directory, ledger, bundle, prepared, rows, suite, transport, *,
              source_check, sleep=time.sleep):
    """Execute a prepared inventory; CLI additionally enforces readiness gates.

    Local fault tests inject a transport. No option in the CLI bypasses an
    unverified token bound or inadequate reservation.
    """
    from scripts.t7_judge import aggregate, apply_results
    if prepared["status"] != "prepared":
        raise ExecutionError("judge_budget_insufficient")
    stage_dir = Path(directory) / "judge"
    stage_dir.mkdir(exist_ok=False)
    batches = prepared["batches"]
    if not batches:
        raise ExecutionError("judge_no_evaluable_batches")
    ledger.register_stage("judge", [{"id": b["batch_id"], "request_sha256": b["payload_sha256"],
                                    "reserve_usd": b["reserve_usd"]} for b in batches],
                          limit_usd="0.3", max_requests=35)
    try:
        write_private(stage_dir / "preparation.private.json", prepared)
    except Exception:
        ledger.stop("judge", "result_save_failed")
        raise
    contents, stopped = {}, None
    for batch in batches:
        identifier = batch["batch_id"]
        envelope, error, digest, content = {}, None, None, None
        try:
            source_check()
            if canonical_sha256(batch["payload"]) != batch["payload_sha256"]:
                raise ExecutionError("request_changed")
            write_private(stage_dir / f"{identifier}.request.json", batch["payload"])
            ledger.claim("judge", identifier)
        except Exception:
            stopped = "source_or_claim_failed"
            ledger.stop("judge", "source_changed")
            break
        try:
            envelope = transport.request(batch["payload"])
            if envelope["usage"]["prompt_tokens"] > batch["input_token_upper_bound"]:
                raise ExecutionError("token_bound_exceeded")
            content = judge_content(envelope, batch)
        except TransportError as exc:
            envelope, error = exc.receipt, exc.code
        except (ValueError, KeyError, TypeError):
            error = "invalid_schema"
        try:
            digest = write_private(stage_dir / f"{identifier}.receipt.json",
                                   {"envelope": envelope, "error": error})
        except Exception:
            error = "result_save_failed"
        try:
            finish_receipt(ledger, "judge", identifier, envelope, digest, error)
        except LedgerError as exc:
            error = exc.code
        if error or ledger.summary()["stopped"]:
            stopped = error or ledger.summary()["stop_reason"]
            break
        contents[identifier] = content
        sleep(bundle["plan"]["transport"]["min_interval_seconds"])
    try:
        labels = apply_results(prepared, contents)
        report = aggregate(rows, labels["labels"], labels["pairlabels"], suite)
    except Exception:
        ledger.stop("judge", "invalid_schema")
        raise
    result = {"schema": "aria_e1_judge_result_v1", "status": "incomplete" if stopped else "complete",
              "stop_reason": stopped, "completed_batches": len(contents), "planned_batches": len(batches),
              "report": report, "labels": labels, "campaign": ledger.summary()}
    try:
        write_private(stage_dir / "results.private.json", result)
    except Exception:
        ledger.stop("judge", "result_save_failed")
        raise
    return result


def judge_main(bundle, execution, *, execute=False, online=False):
    from scripts.t7_judge import build_batches, prepare_human_form
    directory = campaign_directory(bundle["root"])
    if not (directory / "e1" / "results.private.json").is_file():
        raise ExecutionError("judge_generation_artifacts_required")
    directory, ledger = open_campaign(bundle, execution)
    rows = load_generation(directory, ledger, bundle)
    suite, _, _, _ = prepare_e1(bundle)
    prepared = build_batches(rows, suite, bundle["plan"], bundle["contract"], committed_seeds(bundle))
    sample, empty_form = committed_human_records(bundle)
    human_form = prepare_human_form(rows, suite, sample, empty_form, prepared)
    report = {"stage": "judge", "planned_batches": len(prepared["batches"]),
              "model_requests": 0, "full_reserved_cost_usd": prepared["total_reserve_usd"],
              "stage_limit_usd": "0.3", "blocking_reasons": ["provider_input_token_bound_unverified"]}
    if prepared["status"] != "prepared":
        report["blocking_reasons"].append("judge_budget_insufficient")
    if not prepared["batches"]:
        report["blocking_reasons"].append("judge_no_evaluable_batches")
    if online and prepared["batches"]:
        largest = max(prepared["batches"], key=lambda b: b["input_token_upper_bound"])
        try:
            report["metadata"] = fetch_metadata(model=largest["payload"]["model"],
                frozen_metadata=read_json(bundle["root"] / "experiments/t6_20260930/public_metadata.json"),
                payload=largest["payload"], prompt_token_bound=largest["input_token_upper_bound"],
                enable_network=True)
        except TransportError as exc:
            report["metadata"] = exc.receipt
            report["blocking_reasons"].append(exc.code)
    report["ready_to_send"] = not report["blocking_reasons"]
    if execute:
        # Exact private plan is reviewable even when no paid call is permitted.
        plan_path = directory / "judge-preflight.private.json"
        artifact = {"preparation": prepared, "readiness": report, "human_form": human_form}
        if not plan_path.exists():
            write_private(plan_path, artifact)
        if not report["ready_to_send"]:
            print(json.dumps({"ok": False, "status": "blocked_before_credentials", **report}, ensure_ascii=False))
            return 2
        transport = OpenRouterTransport(credential(bundle["root"]), enable_network=True)
        result = run_judge(directory, ledger, bundle, prepared, rows, suite, transport,
                          source_check=lambda: verify_execution(load_bundle()))
        report = {k: result[k] for k in ("status", "stop_reason", "completed_batches", "planned_batches")}
    print(json.dumps({"ok": report.get("status", "complete") == "complete", **report}, ensure_ascii=False))
    return 0 if report.get("status", "complete") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
