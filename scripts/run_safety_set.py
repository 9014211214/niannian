"""Run fixed safety inputs with the shared product driver and fixed denominator."""
import argparse
from contextlib import closing
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.voice.minimal import MinimalWebApp

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

DEFAULT_INPUTS = Path(__file__).resolve().parents[1] / "aria/bench/cases/fixed_safety_v1/inputs.json"


def safety_story(path):
    sessions = []
    for case in read_json(path)["cases"]:
        item = case["execution_input"]
        operation = {"op": "say", "text": item["messages"][-1]["content"], "execution_input": item}
        sessions.append({"session_id": case["regression_id"], "fresh_profile": True, "operations": [
            {"op": "start_session"}, operation, {"op": "seal"}]})
    return {"independent_cases": True, "sessions": sessions}


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--story", type=Path, required=True)
    value.add_argument("--repeats", type=int, default=1)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--enable-model-calls", action="store_true")
    from aria.release import DEFAULT_RELEASE_PROFILE
    value.add_argument("--release-profile", choices=("release_deepseek_v1", "release_max_v1"),
                       default=DEFAULT_RELEASE_PROFILE)
    from aria.engine.policy.memory_structured import OFFLINE_ROUTES, MODEL, PROVIDER
    value.add_argument("--offline-model", choices=sorted({m for m,_ in OFFLINE_ROUTES}), default=MODEL)
    value.add_argument("--offline-provider", choices=sorted({p for _,p in OFFLINE_ROUTES}), default=PROVIDER)
    value.add_argument("--model-max-requests", type=int)
    value.add_argument("--model-stop-cost-usd")
    value.add_argument("--model-reservation-policy", type=Path)
    for name in ("online-models-metadata", "online-endpoints-metadata", "offline-endpoints-metadata"):
        value.add_argument("--" + name, type=Path)
    value.add_argument("--unit-limit", type=int, help="Stop cleanly after this many independent units.")
    value.add_argument("--continue-from", type=Path, help="Sealed preceding segment of the same batch.")
    value.add_argument("--continuation-policy", type=Path, help="Pre-registered continuation policy, bound at first launch.")
    value.add_argument("--continuation-admission", type=Path, help="Independent, request-specific decision for a stopped segment.")
    value.add_argument("--accept-held-request", action="append", default=[], metavar="ID=USD",
                       help="Operator's explicit instruction for this particular full reservation.")
    return value

def configured_factory(args):
    fields = (args.model_max_requests, args.model_stop_cost_usd, args.model_reservation_policy)
    if not args.enable_model_calls:
        if any(v is not None for v in fields):
            raise ValueError("model_limits_require_enable_model_calls")
        return MinimalWebApp, None
    from aria.engine.policy.minimal_calls import validate_limits
    from aria.engine.policy.reserved_calls import validate_reservation_policy
    from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
    from aria.engine.policy.memory_structured import build_memory_structured_route, MODEL, PROVIDER
    from aria.release import create_max_release_clients, create_deepseek_release_clients, build_release_online_route
    validate_limits(*fields[:2])
    policy = validate_reservation_policy(read_json(args.model_reservation_policy))
    models = args.online_models_metadata.read_bytes()
    online = build_release_online_route(models, args.online_endpoints_metadata.read_bytes(),
        release_profile=args.release_profile)
    memory = build_memory_structured_route(models, args.offline_endpoints_metadata.read_bytes(),
        model=getattr(args,"offline_model",MODEL), provider=getattr(args,"offline_provider",PROVIDER))
    create_clients = (create_deepseek_release_clients if args.release_profile == "release_deepseek_v1"
                      else create_max_release_clients)
    # The shared product ledger keeps every fresh profile under one allowance.
    guard = None
    def factory(path):
        nonlocal guard
        clients = create_clients(model_calls_enabled=True,
            online_structured_route=online, memory_structured_route=memory)
        if guard is None:
            guard = MaxReleaseCallGuard(args.out / "model_calls.sqlite3", max_requests=fields[0],
                max_observed_cost_usd=fields[1], reservation_policy=policy)
        return MinimalWebApp(path, clients=clients, model_call_guard=guard)
    return factory, lambda: guard.close() if guard else None

def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def segment_files(folder):
    return {p.relative_to(folder).as_posix(): checksum(p) for p in folder.rglob("*")
            if p.is_file() and p != folder / "segment.json" and not p.name.endswith("-shm")
            and not (p.name.endswith("-wal") and p.stat().st_size == 0)}

def amount(value):
    from decimal import InvalidOperation
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid_segment_amount") from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("invalid_segment_amount")
    return parsed

def segment_costs(folder, summary, paid):
    if not paid:
        return {"calls": 0, "known": "0", "unknown": {}, "stopped": False, "open_operations": 0}
    path = folder / "model_calls.sqlite3"
    if not path.is_file() or (Path(str(path) + "-wal").exists() and Path(str(path) + "-wal").stat().st_size):
        raise ValueError("segment_requires_closed_ledger")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        rows = [json.loads(row[0]) for row in db.execute("SELECT record FROM attempts ORDER BY sequence")]
        open_operations = db.execute("SELECT COUNT(*) FROM release_operations WHERE finished != 1").fetchone()[0]
    known = sum((amount(r["usage"]["cost"]) for r in rows if "cost" in r["usage"]), Decimal(0))
    unknown = {r["request_id"]: {"hold": str(amount(r["reservation"]["reserved_cost_usd"])),
               "generation": r.get("observation", {}).get("generation_id_candidate")
                    if r.get("observation", {}).get("generation_id_candidate_status") == "accepted" else None,
               "model": r["model"], "provider": r["expected_provider"], "status": r["status"],
               "started_at": r["started_at"], "finished_at": r.get("finished_at"),
               "upstream_failure": upstream_failure(r)}
               for r in rows if "cost" not in r["usage"]}
    fee = summary["fees"]
    held = sum((amount(r["hold"]) for r in unknown.values()), Decimal(0))
    if (len(rows) != fee["calls_used"] or known != amount(fee["known_cost_usd"] or 0)
            or len(unknown) != fee["unknown_cost_requests"] or held != amount(fee["reserved_cost_usd"])):
        raise ValueError("segment_ledger_summary_mismatch")
    return {"calls": len(rows), "known": str(known), "unknown": unknown,
            "stopped": fee["stopped"], "open_operations": open_operations}

def upstream_failure(row):
    transient = {408, 429, 500, 502, 503, 504}
    identity = row.get("receipt_identity") or {}
    if (any(identity.get(k) in {"unrecognized", "wrong_type"} for k in ("model_status", "provider_status"))
            or row.get("returned_provider") not in (None, row["expected_provider"])
            or row.get("returned_model") not in (None, row["model"], row["model"].replace("-1017", "-20261017"))
            or row.get("fee_verification_error") not in {None, "finite_response_identity_unverified", "http_error_cost_unverified"}):
        return False
    kind = row.get("failure_phase"), row.get("failure_code"), row.get("http_status")
    if kind[:2] == ("http_status", "http_error"):
        return kind[2] in transient
    stream = row.get("stream_read", {})
    return (kind == ("read", "response_read_failed", 200) and
            (stream.get("provider_error_code") in transient or row.get("body_read", {}).get("exception_kind")
             in {"incomplete_read", "connection_reset", "timeout", "connection_error"}))

def referenced_evidence(folder, item):
    path = folder / item["path"]
    if checksum(path) != item["sha256"]:
        raise ValueError("continuation_evidence_changed")
    return read_json(path)

def accepted_cost(args, previous, parent_sha):
    costs = previous["costs"]
    if costs["open_operations"]:
        raise ValueError("unclosed_release_operation")
    unknown = costs["unknown"]
    committed = amount(costs["known"])
    if not unknown:
        if previous["stop_reason"] or costs["stopped"] or args.continuation_admission or args.accept_held_request:
            raise ValueError("previous_segment_not_eligible_to_continue")
        return committed
    if previous["stop_reason"] not in {"model_or_storage_failed", "provider_cost_unknown", "unresolved_request",
                                        "unknown_cost", "request_failed", "unresolved_release_operation"}:
        raise ValueError("unverified_or_unrelated_segment_stop")
    if not previous["identity"]["continuation_policy"] or not args.continuation_admission:
        raise ValueError("unknown_fee_requires_registered_continuation_admission")
    packet = read_json(args.continuation_admission)
    if (packet.get("schema") != "aria.independent_segment_admission.v1" or packet.get("approved") is not True
            or not packet.get("reviewer") or packet.get("parent_segment_sha256") != parent_sha
            or packet.get("policy_sha256") != previous["identity"]["continuation_policy"]
            or packet.get("destination") != str(args.out.resolve()) or set(packet["requests"]) != set(unknown)
            or amount(packet["child_cap_usd"]) != amount(args.model_stop_cost_usd)
            or packet["child_max_requests"] != args.model_max_requests):
        raise ValueError("invalid_request_specific_continuation_admission")
    explicit = []
    for rid, row in unknown.items():
        entry = packet["requests"][rid]
        if not row["upstream_failure"] or row["status"] != "failed" or amount(entry["full_hold_usd"]) != amount(row["hold"]):
            raise ValueError("unresolved_or_mismatched_original_reservation")
        evidence = referenced_evidence(args.continuation_admission.parent, entry["evidence"])
        if (evidence.get("local_request_id") != rid
                or evidence.get("generation_id_candidate") != row["generation"]):
            raise ValueError("fee_evidence_identity_mismatch")
        if entry["decision"] == "retain_full_hold":
            if (evidence.get("schema") != "aria.user_hold_instruction.v1" or not evidence.get("user_message_ref")
                    or evidence.get("permission") != "continue_unattempted_units_keep_full_hold"
                    or evidence.get("parent_segment_sha256") != parent_sha
                    or amount(evidence["full_hold_usd"]) != amount(row["hold"])):
                raise ValueError("specific_user_hold_instruction_required")
            explicit.append(f"{rid}={row['hold']}")
            committed += amount(row["hold"])
        elif entry["decision"] == "settled":
            data = evidence.get("data", {})
            parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
            if (not row["generation"] or evidence.get("schema") != "aria.readonly_generation_fee_query.v1"
                    or evidence.get("http_status") != 200 or evidence.get("method") != "GET"
                    or data.get("id") != row["generation"] or data.get("is_byok") is not False
                    or data.get("provider_name") != row["provider"]
                    or data.get("model") not in (row["model"], row["model"].replace("-1017", "-20261017"))
                    or not parse(row["started_at"]) <= parse(data["created_at"]) <= parse(row["finished_at"])
                    or not evidence.get("response_bytes") or len(evidence.get("response_sha256", "")) != 64
                    or evidence.get("model_calls") != 0 or evidence.get("ledger_modified") is not False
                    or amount(data["total_cost"]) != amount(data["usage"])):
                raise ValueError("verified_generation_cost_required")
            committed += amount(data["total_cost"])
        else:
            raise ValueError("unsupported_continuation_disposition")
    if sorted(args.accept_held_request) != sorted(explicit):
        raise ValueError("explicit_live_instruction_required_for_each_retained_hold")
    return committed

def segment_context(args, identity):
    ctx = {"schema": "aria.independent_units.v1", "root": str(args.out.resolve()), "index": 1,
           "identity": identity, "parents": [], "prior_calls": 0, "prior_occupied_usd": "0",
           "initial_cap_usd": args.model_stop_cost_usd, "initial_max_requests": args.model_max_requests,
           "segment_cap_usd": args.model_stop_cost_usd, "segment_max_requests": args.model_max_requests,
           "explicit_retained_requests": args.accept_held_request,
           "admission": ({"path": str(args.continuation_admission.resolve()), "sha256": checksum(args.continuation_admission)}
                         if args.continuation_admission else None)}
    if not args.continue_from:
        if args.continuation_admission or args.accept_held_request:
            raise ValueError("no_parent_for_continuation_admission")
        return ctx, None
    folder = args.continue_from.resolve()
    previous = read_json(folder / "segment.json")
    parent_sha = checksum(folder / "segment.json")
    refs = [*previous["parents"], {"path": str(folder), "sha256": parent_sha}]
    root = Path(previous["root"])
    calls, occupied, predecessor = 0, Decimal(0), None
    for index, ref in enumerate(refs, 1):
        prior_dir = root if index == 1 else Path(f"{root}.part{index}")
        if str(prior_dir) != ref["path"] or checksum(prior_dir / "segment.json") != ref["sha256"]:
            raise ValueError("segment_origin_changed")
        record = read_json(prior_dir / "segment.json")
        started = read_json(prior_dir / "segment-start.json")
        if started != {key: value for key, value in record.items() if key not in {"files", "costs", "stop_reason"}}:
            raise ValueError("segment_start_binding_changed")
        if index == 1 and (record["initial_cap_usd"] != record["segment_cap_usd"]
                           or record["initial_max_requests"] != record["segment_max_requests"]):
            raise ValueError("original_allowance_mismatch")
        if predecessor:
            grant = record["admission"]
            if grant and checksum(grant["path"]) != grant["sha256"]:
                raise ValueError("historical_admission_changed")
            old_args = SimpleNamespace(out=prior_dir, accept_held_request=record["explicit_retained_requests"],
                       continuation_admission=Path(grant["path"]) if grant else None,
                       model_stop_cost_usd=record["segment_cap_usd"], model_max_requests=record["segment_max_requests"])
            occupied += accepted_cost(old_args, predecessor, refs[index-2]["sha256"])
        if record["prior_calls"] != calls or amount(record["prior_occupied_usd"]) != occupied:
            raise ValueError("ancestor_allowance_mismatch")
        if (record["schema"] != ctx["schema"] or record["identity"] != identity or record["index"] != index
                or record["root"] != previous["root"] or record["initial_cap_usd"] != previous["initial_cap_usd"]
                or record["initial_max_requests"] != previous["initial_max_requests"]
                or record["parents"] != refs[:index-1] or record["files"] != segment_files(prior_dir)
                or record["costs"] != segment_costs(prior_dir, read_json(prior_dir / "summary.json"), identity["paid"])):
            raise ValueError("prior_source_plan_or_evidence_changed")
        if identity["paid"] and (amount(record["segment_cap_usd"]) + occupied > amount(record["initial_cap_usd"])
                                 or record["segment_max_requests"] + calls > record["initial_max_requests"]):
            raise ValueError("ancestor_exceeds_original_allowance")
        calls += record["costs"]["calls"]
        predecessor = record
    if args.out.resolve() != Path(f"{root}.part{len(refs)+1}"):
        raise ValueError("unique_next_segment_destination_required")
    cost = occupied + accepted_cost(args, previous, parent_sha)
    if identity["paid"] and (amount(args.model_stop_cost_usd) + cost > amount(previous["initial_cap_usd"])
                            or args.model_max_requests + calls > previous["initial_max_requests"]):
        raise ValueError("continuation_exceeds_original_allowance")
    ctx.update(root=str(root), index=len(refs)+1, parents=refs, prior_calls=calls, prior_occupied_usd=str(cost),
               initial_cap_usd=previous["initial_cap_usd"], initial_max_requests=previous["initial_max_requests"])
    return ctx, read_json(folder / "summary.json")

def seal_segment(args, context, summary):
    context.update(files=segment_files(args.out), costs=segment_costs(args.out, summary, args.enable_model_calls),
                   stop_reason=summary["stop_reason"])
    with (args.out / "segment.json").open("x", encoding="utf-8") as stream:
        json.dump(context, stream, ensure_ascii=False, indent=2)

def main(argv=None):
    from scripts.run_story import main as story_main
    args = list(sys.argv[1:] if argv is None else argv)
    if "--story" not in args:
        args = ["--story", str(DEFAULT_INPUTS), *args]
    return story_main(args, story_loader=safety_story)


if __name__ == "__main__":
    raise SystemExit(main())
