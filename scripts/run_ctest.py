"""Freeze and explicitly execute the authorized C-test component experiment.

Preparation never reads credentials or sends requests. Execution extracts only
the named local key, after verifying source, protocol, metadata and allowance.
There is no resume or automatic retry: an existing attempt remains evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria._openrouter import DEFAULT_MEMORY_MODEL
from aria.bench.ctest_cases import ctest_manifest, load_ctest_cases
from aria.bench.ctest_runner import CTestRunner
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.reserved_calls import ReservedCallGuard
from scripts.prepare_model_policy import derive_policy

SCHEMA = "aria_ctest_execution_v1"
PRIOR_KNOWN = Decimal("0.026719172")
LIMIT = Decimal("2") - PRIOR_KNOWN
FILES = ("scripts/run_ctest.py", "scripts/prepare_model_policy.py", "docs/CTEST_EXECUTION.md")
METADATA = ("models.public-v2.json", "qwen-endpoints.public-v2.json", "sonnet-endpoints.public-v2.json")
OBSERVATIONS = "public-model-metadata-v2.json"
AUTHORIZATION_MARKER = Path("artifacts/local/tasks/20261003-02/ctest-authorization.started.json")
METADATA_URLS = {
    "models": "https://openrouter.ai/api/v1/models",
    "qwen-endpoints": "https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints",
    "sonnet-endpoints": "https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-5/endpoints",
}


class RunError(ValueError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def inventory(root):
    paths = sorted({*(root / "aria").rglob("*.py"), *(root / item for item in FILES)})
    rows = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RunError("source_link_rejected")
        raw = path.read_bytes()
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": digest(raw)})
    return rows


def protocol():
    value = {"schema": SCHEMA, "experiment": "ctest-v1-20261003",
            "authorization": "User approved C-test upper limit USD 2 on 2026-10-03",
            "model": DEFAULT_MEMORY_MODEL, "provider": "anthropic", "max_output_tokens": 1800,
            "timeout_seconds": 120, "retry_count": 0, "fallback_count": 0,
            "denominators": {"people": 12, "change": 6, "control": 6, "sessions": 96,
                             "memory_calls": 96, "online_calls": 0, "reflection_calls": 0},
            "decision": {"control_false_triggers": 0, "change_min_triggers": 4,
                         "all_sessions_required": True, "failed_or_unissued": "not_evaluable",
                         "cognitive_observations": "explicit_array_required_missing_is_not_zero",
                         "best_retry_selection": False},
            "scope": "real_memory_extraction_and_rules_on_authored_fictional_scripts",
            "limitations": ["not clinical validation", "no real online reply evaluation",
                            "functional-impact branch; three-day growth branch covered only by software tests"],
            "clock": "isolated fictional year 2000; actual request times recorded separately",
            "budget": {"user_limit_usd": "2", "prior_known_e1_usd": str(PRIOR_KNOWN),
                       "prior_accepted_unknown_requests": 2, "cumulative_stop_usd": "2",
                       "effective_new_allowance_usd": str(LIMIT),
                       "unknown_new_request": "stop_and_preserve_no_retry",
                       "reserve": "each_actual_wire_utf8_bytes_plus_8192_and_max_output",
                       "full_96_completion_within_budget": "not_guaranteed",
                       "invoice_guarantee": False},
            "case_manifest": ctest_manifest(load_ctest_cases())}
    return json.loads(canonical(value))


def metadata_evidence(directory):
    raws = [(directory / name).read_bytes() for name in METADATA]
    observation_raw = (directory / OBSERVATIONS).read_bytes()
    observations = json.loads(observation_raw)
    observed = []
    for name, raw in zip(("models", "qwen-endpoints", "sonnet-endpoints"), raws):
        record = observations.get(name, {})
        if record.get("sha256") != digest(raw):
            raise RunError("metadata_observation_mismatch")
        stamp = datetime.fromisoformat(record["observed_at"])
        if stamp.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 86400:
            raise RunError("public_metadata_expired")
        if record.get("url") != METADATA_URLS[name]:
            raise RunError("metadata_observation_source_mismatch")
        observed.append({"name": name, "observed_at": record["observed_at"]})
    policy, proof = derive_policy(*raws)
    return {"files": [{"path": name, "sha256": digest(raw)} for name, raw in zip(METADATA, raws)],
            "observation_file": {"path": OBSERVATIONS, "sha256": digest(observation_raw)},
            "policy": policy, "proof": proof, "observations": observed}


def prepare(root, metadata, destination):
    value = {"protocol": protocol(), "source": inventory(root), "metadata": metadata_evidence(metadata),
             "frozen_at": datetime.now(timezone.utc).isoformat()}
    value["seal_sha256"] = digest(canonical(value))
    write_new(destination, value)
    return value


def verify(root, metadata, registration):
    value = json.loads(registration.read_text(encoding="utf-8"))
    if set(value) != {"protocol", "source", "metadata", "frozen_at", "seal_sha256"}:
        raise RunError("registration_shape_changed")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    if value["seal_sha256"] != digest(canonical(unsigned)):
        raise RunError("registration_seal_changed")
    if value["protocol"] != protocol() or value["source"] != inventory(root):
        raise RunError("frozen_source_or_protocol_changed")
    if value["metadata"] != metadata_evidence(metadata):
        raise RunError("frozen_capability_or_price_changed")
    frozen = datetime.fromisoformat(value["frozen_at"])
    if frozen.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - frozen).total_seconds() <= 86400:
        raise RunError("registration_expired")
    return value


def local_key(path):
    # No load_dotenv, environment update, shell interpolation, or unrelated keys.
    matches = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                matches.append(value.strip().strip('"').strip("'"))
    if (len(matches) != 1 or not 16 <= len(matches[0]) <= 512
            or any(ord(ch) <= 32 or ord(ch) >= 127 for ch in matches[0])):
        raise RunError("authorized_local_key_missing_or_invalid")
    return matches[0]


def make_client(key):
    client = OpenRouterLLMClient(api_key=key, base_url="https://openrouter.ai/api/v1",
                                model=DEFAULT_MEMORY_MODEL, fallback_models=[], temperature=0.2,
                                role="memory", timeout=120, max_retries=0, bounded_transport=True)
    client.max_completion_tokens = 1800
    client.reasoning_effort = "none"
    return client


def execute(root, metadata, registration, output, *, allow_local_env):
    if allow_local_env is not True:
        raise RunError("explicit_local_key_permission_required")
    frozen = verify(root, metadata, registration)
    # The authorization is single-use across copied registrations, a newly
    # prepared seal, and output-directory changes. Never automatically reset it.
    authorization_marker = root.resolve() / AUTHORIZATION_MARKER
    if not authorization_marker.resolve().is_relative_to(root.resolve()):
        raise RunError("authorization_marker_outside_workspace")
    authorization_marker.parent.mkdir(parents=True, exist_ok=True)
    write_new(authorization_marker, {"experiment": frozen["protocol"]["experiment"],
              "seal_sha256": frozen["seal_sha256"], "registration": str(registration.resolve()),
              "output": str(output.resolve()), "started_at": datetime.now(timezone.utc).isoformat(),
              "automatic_resume": False})
    write_new(registration.with_suffix(".started.json"), {"seal_sha256": frozen["seal_sha256"],
              "output": str(output.resolve()), "started_at": datetime.now(timezone.utc).isoformat()})
    # Exclusive creation is also the restart gate. Never create another ledger
    # for the same frozen experiment after a timeout or failed result write.
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "STARTED.json", {"seal_sha256": frozen["seal_sha256"],
              "started_at": datetime.now(timezone.utc).isoformat(), "automatic_resume": False})
    guard = ReservedCallGuard(output / "allowance.sqlite3", max_requests=96,
                              max_observed_cost_usd=str(LIMIT),
                              reservation_policy=frozen["metadata"]["policy"])
    status = "not_evaluable"
    failure = None
    try:
        # Configuration validation happens with a placeholder before reading .env.
        client = make_client("local-configuration-placeholder")
        guard.attach(client)
        client.api_key = local_key(root / ".env")
        runner = CTestRunner(output, experiment_id="ctest-v1-20261003", memory_client=client,
                             guard=guard, allow_model_calls=True,
                             registration_sha256=frozen["seal_sha256"])
        result = runner.run()
        status = result["real_model_gate"]
    except Exception as error:
        failure = type(error).__name__  # never provider bodies or key-bearing text
        guard.halt()
    finally:
        state = guard.snapshot()
        if state["calls_used"] is None or state["unknown_cost_requests"] is None:
            status = "not_evaluable"
            failure = failure or "AccountingUnavailable"
        guard.close()
        write_new(output / "execution-receipt.json", {"seal_sha256": frozen["seal_sha256"],
                  "real_model_gate": status, "failure_type": failure, "accounting": state,
                  "old_unknown_requests_retained": 2,
                  "completed_at": datetime.now(timezone.utc).isoformat()})
    return {"real_model_gate": status, "failure_type": failure,
            "calls_used": state["calls_used"], "known_cost_usd": state["known_cost_usd"],
            "unknown_cost_requests": state["unknown_cost_requests"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-local-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare(ROOT, args.metadata, args.registration)
            public = {"status": "prepared", "seal_sha256": result["seal_sha256"]}
        elif args.verify:
            result = verify(ROOT, args.metadata, args.registration)
            public = {"status": "verified", "seal_sha256": result["seal_sha256"]}
        else:
            if args.output is None:
                raise RunError("output_required")
            public = execute(ROOT, args.metadata, args.registration, args.output,
                             allow_local_env=args.allow_local_env)
        print(json.dumps(public, ensure_ascii=False))
        return 2 if public.get("real_model_gate") == "not_evaluable" else 0
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "error": str(error) if isinstance(error, RunError)
                          else type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
