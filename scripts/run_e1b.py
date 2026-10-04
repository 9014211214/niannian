"""Explicit, single-campaign E1b execution. Prepare/verify never read a key.

Authorization files are review artifacts, not authority by themselves: execute
also requires independently supplied hashes and an explicit budget choice. No
command approves a campaign, refreshes metadata, retries, or resets a ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria._openrouter import DEFAULT_MEMORY_MODEL, DEFAULT_ONLINE_FAST_MODEL, DEFAULT_REFLECTION_MODEL
from aria.bench.e1b_cases import load_e1b_cases
from aria.bench.e1b_runner import E1bExperimentApp, E1bRunner
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.online_decision import configured_online_decision_version, online_decision_tool_spec, protocol_version
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.policy.prompt import configured_delivery_block
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.runtime import MinimalModelClients
from scripts import e1b_evaluation as evaluation
from scripts.run_ctest import local_key, metadata_evidence as base_metadata_evidence

SCHEMA = "aria_e1b_execution_v1"
CAMPAIGN = Path("artifacts/local/e1b-formal-v1")
PRIOR_KNOWN = Decimal("1.392379172")
CUMULATIVE_STOP = Decimal("2")
ACCEPTED_OLD_UNKNOWN = 2
RETAINED_UNKNOWN_HOLD = Decimal("0")
MIN_INTERVAL_SECONDS = 0
MAX_WIRE_REQUESTS = 1636
SEED = 20261003
STAGES = ("prescreen", "main", "judge")
COUNTS = {"prescreen": {"online_fast": 540},
          "main": {"online_fast": 1008, "memory": 24, "reflection": 24},
          "judge": {"judge": 40}}
FILES = ("scripts/run_e1b.py", "scripts/run_ctest.py", "scripts/prepare_model_policy.py",
         "scripts/e1b_evaluation.py", "scripts/e1b_formation.py", "scripts/e1b_secondary.py", "scripts/e1b_assessment.py",
         "docs/E1B_EXECUTION.md", "docs/VALIDATION.md",
         "aria/bench/cases/e1b_v1_draft.yaml")
SHA = re.compile(r"[a-f0-9]{64}\Z")
read_json, write_once, digest = evaluation.read_json, evaluation.write_once, evaluation.digest


class RunError(ValueError):
    pass


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def campaign(root):
    root = Path(root).resolve()
    path = root / CAMPAIGN
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise RunError("campaign_path_link_rejected")
    return path


def workspace_identity(root):
    return text_sha(os.path.normcase(str(Path(root).resolve())))


def inventory(root):
    root = Path(root).resolve()
    paths = sorted({*(root / "aria").rglob("*.py"), *(root / "aria").rglob("*.yaml"),
                    *(root / "aria").rglob("*.yml"), *(root / name for name in FILES)})
    rows = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RunError("source_link_rejected")
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": byte_sha(path)})
    return rows


def protocol():
    return {"schema": SCHEMA, "campaign": "e1b-formal-v1", "seed": SEED,
            "case_source_sha256": load_e1b_cases().source_sha256,
            "planned_calls": COUNTS, "planned_total_calls": 1636,
            "prescreen_scoring_positions": 162, "main_scoring_positions": 216,
            "models": {"online_fast": DEFAULT_ONLINE_FAST_MODEL, "memory": DEFAULT_MEMORY_MODEL,
                       "reflection": DEFAULT_REFLECTION_MODEL, "judge": DEFAULT_MEMORY_MODEL},
            "product_configuration": {"online_decision_version": "online_turn_decision_v1", "delivery_block": "mode_v1",
                                      "reflection_return_format": "json_schema"},
            "max_output_tokens": {"online_fast": 1024, "memory": 1800, "reflection": 2800, "judge": 1024},
            "transport": {"base_url": "https://openrouter.ai/api/v1", "reasoning": False,
                          "retries": 0, "fallbacks": 0, "timeout": 120, "reflection_timeout": 240},
            "budget": {"status": "pending", "choice": None, "approved_limit_usd": None,
                       "prior_known_floor_usd": str(PRIOR_KNOWN), "accepted_old_unknown": ACCEPTED_OLD_UNKNOWN,
                       "cumulative_stop_usd": str(CUMULATIVE_STOP),
                       "current_remaining_usd": str(CUMULATIVE_STOP - PRIOR_KNOWN),
                       "completion_within_budget": "not_guaranteed"},
            "ten_case_path": "unregistered_blocked", "failed_missing_rule_only": "not_zero",
            "synthetic": {"people": True, "emotion": "fixed_seed_70_15_15_both_arms_identical"},
            "judging": "first_independent_labels_only_no_author_scoring_no_best_retry"}


def check_age(stamp):
    when = datetime.fromisoformat(stamp)
    if when.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - when).total_seconds() <= 86400:
        raise RunError("registration_expired")


def product_configuration_preflight():
    if configured_online_decision_version() != "online_turn_decision_v1" or configured_delivery_block() != "mode_v1":
        raise RunError("nondefault_product_configuration_requires_separate_registration")


def metadata_evidence(directory):
    """Require the exact reflection route's advertised structured contract."""
    from aria.engine.memory.reflection_contract import reflection_response_schema
    from aria.engine.policy.reflection_json import reflection_wire_schema

    result = base_metadata_evidence(directory)
    path = next(row["path"] for row in result["files"] if row["path"].startswith("sonnet-endpoints."))
    data = read_json(Path(directory) / path)["data"]
    routes = [row for row in data["endpoints"] if row.get("tag") == "anthropic"]
    required = {"response_format", "structured_outputs"}
    if (data.get("id") != DEFAULT_REFLECTION_MODEL or len(routes) != 1
            or type(routes[0].get("status")) is not int or routes[0]["status"] != 0
            or not required <= set(routes[0].get("supported_parameters", []))):
        raise RunError("reflection_structured_output_support_unverified")
    result["reflection_structured_output"] = {
        "provider": "anthropic", "required_parameters": sorted(required),
        "wire_schema_sha256": digest(reflection_wire_schema()),
        "local_full_schema_sha256": digest(reflection_response_schema()),
    }
    return result


def read_receipt(root, stage):
    path = campaign(root) / stage / "execution-receipt.json"
    value = read_json(path)
    if (value.get("stage") != stage or value.get("status") != "completed"
            or value.get("receipt_sha256") != digest({k: v for k, v in value.items() if k != "receipt_sha256"})):
        raise RunError("prior_stage_not_completed")
    state = value["accounting"]
    expected = sum(sum(COUNTS[s].values()) for s in STAGES[:STAGES.index(stage) + 1])
    assert_clean(state, expected_calls=expected, final=stage == "judge")
    return value


def selection_evidence(root):
    receipt = read_receipt(root, "prescreen")
    base = campaign(root) / "prescreen" / "blind-bundle"
    packet, key = evaluation.load_bundle(base)
    if (digest(packet) != receipt["packet_sha256"] or digest(key) != receipt["mapping_sha256"]
            or key["source_report_sha256"] != receipt["qualification"]["source_report_sha256"]):
        raise RunError("prescreen_bundle_not_bound_to_receipt")
    result = evaluation.aggregate(base)
    if (result["status"] != "prescreen_labels_complete" or result["planned_reply_positions"] != 162
            or result["valid_primary_labels"] != 162 or result["missing_generation_positions"] != 0):
        raise RunError("all_18_by_3_first_prescreen_labels_required")
    selection = result["selection"]
    ids = selection.get("case_ids")
    if type(ids) is not list or len(ids) != 12:
        raise RunError("prescreen_shortfall_no_denominator_change")
    labels = sorted((base / "private" / "labels" / "primary").glob("*.json"))
    return {"case_ids": ids, "repeat_violations": {k: list(v) for k, v in result["repeat_violations"].items()},
            "selection": selection, "prescreen_receipt_sha256": receipt["receipt_sha256"],
            "label_files": [{"name": p.name, "sha256": byte_sha(p)} for p in labels]}


def stage_inputs(root, stage):
    if stage == "prescreen":
        return {"case_ids": [c.id for c in load_e1b_cases().items]}
    if stage == "main":
        return selection_evidence(root)
    receipt = read_receipt(root, "main")
    base = campaign(root) / "main" / "blind-bundle"
    packet, key = evaluation.load_bundle(base)
    if digest(packet) != receipt["packet_sha256"] or digest(key) != receipt["mapping_sha256"]:
        raise RunError("main_bundle_not_bound_to_receipt")
    sample = evaluation._load_secondary(base, packet)
    return {"packet_sha256": digest(packet), "sample_sha256": digest(sample),
            "main_receipt_sha256": receipt["receipt_sha256"], "case_ids": key["protocol"]["case_ids"]}


def prepare(root, metadata, destination, *, stage="prescreen"):
    product_configuration_preflight()
    if stage not in STAGES:
        raise RunError("invalid_stage")
    value = {"protocol": protocol(), "workspace_identity": workspace_identity(root),
             "stage": stage, "inputs": stage_inputs(root, stage),
             "source": inventory(root), "metadata": metadata_evidence(Path(metadata)),
             "frozen_at": datetime.now(timezone.utc).isoformat()}
    value["seal_sha256"] = digest(value)
    write_once(destination, value)
    return value


def verify(root, metadata, registration):
    product_configuration_preflight()
    value = read_json(registration)
    if set(value) != {"protocol", "workspace_identity", "stage", "inputs", "source", "metadata", "frozen_at", "seal_sha256"}:
        raise RunError("registration_shape_changed")
    if value["seal_sha256"] != digest({k: v for k, v in value.items() if k != "seal_sha256"}):
        raise RunError("registration_seal_changed")
    check_age(value["frozen_at"])
    if value["workspace_identity"] != workspace_identity(root):
        raise RunError("registration_belongs_to_another_workspace")
    if value["stage"] not in STAGES or value["protocol"] != protocol() or value["source"] != inventory(root):
        raise RunError("frozen_source_or_protocol_changed")
    if value["metadata"] != metadata_evidence(Path(metadata)):
        raise RunError("frozen_capability_or_price_changed")
    if value["inputs"] != stage_inputs(root, value["stage"]):
        raise RunError("frozen_stage_inputs_changed")
    return value


def authorization_template(root=None):
    return {"schema": "e1b_campaign_authorization_v1", "status": "pending", "budget_choice": None,
            "user_limit_usd": None, "prior_known_usd": str(PRIOR_KNOWN),
            "cumulative_stop_usd": str(CUMULATIVE_STOP), "accepted_old_unknown": 2,
            "workspace_identity": workspace_identity(root) if root is not None else None,
            "ledger_identity": text_sha(os.path.normcase(str(campaign(root) / "allowance.sqlite3"))) if root is not None else None,
            "human_authorization_reference": None}


def validate_authorization(root, frozen, *, budget_choice, campaign_authorization_sha256,
                           stage_authorization_sha256):
    if budget_choice != "12_usd1":
        raise RunError("explicit_12_usd1_required_10_case_unregistered")
    if not all(type(s) is str and SHA.fullmatch(s) for s in
               (campaign_authorization_sha256, stage_authorization_sha256)):
        raise RunError("independently_confirmed_authorization_hashes_required")
    base = campaign(root)
    if frozen.get("workspace_identity") != workspace_identity(root):
        raise RunError("registration_belongs_to_another_workspace")
    authorization = read_json(base / "authorization.json")
    if digest(authorization) != campaign_authorization_sha256:
        raise RunError("campaign_authorization_hash_mismatch")
    if (set(authorization) != set(authorization_template()) or authorization["status"] != "approved"
            or authorization["budget_choice"] != budget_choice or authorization["user_limit_usd"] != "1"
            or authorization["accepted_old_unknown"] != 2
            or authorization["workspace_identity"] != workspace_identity(root)
            or authorization["ledger_identity"] != authorization_template(root)["ledger_identity"]
            or authorization["cumulative_stop_usd"] != str(CUMULATIVE_STOP)
            or not isinstance(authorization["human_authorization_reference"], str)
            or not authorization["human_authorization_reference"].strip()):
        raise RunError("campaign_not_explicitly_authorized")
    prior = Decimal(authorization["prior_known_usd"])
    if not prior.is_finite() or not PRIOR_KNOWN <= prior < CUMULATIVE_STOP:
        raise RunError("prior_cost_cannot_be_reset_or_exceed_stop")
    phase = read_json(base / f"{frozen['stage']}.authorization.json")
    expected = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
                "registration_sha256": frozen["seal_sha256"],
                "campaign_authorization_sha256": campaign_authorization_sha256}
    if phase != expected or digest(phase) != stage_authorization_sha256:
        raise RunError("stage_not_explicitly_authorized")
    return authorization, min(Decimal("1"), CUMULATIVE_STOP - prior)


def make_client(role, key):
    config = protocol()
    client = OpenRouterLLMClient(api_key=key, base_url=config["transport"]["base_url"],
        model=config["models"][role], fallback_models=[], temperature=0.2, role=role,
        timeout=config["transport"]["reflection_timeout"] if role == "reflection" else 120,
        max_retries=0, bounded_transport=True,
        online_return_format="json_object" if role == "online_fast" else "native",
        reflection_return_format="json_schema" if role == "reflection" else "native")
    client.max_completion_tokens = config["max_output_tokens"][role]
    client.reasoning_effort = "none"
    return client


class EvidenceGuard(ReservedCallGuard):
    """Response hashes join the existing durable actual-request receipt."""

    def __init__(self, *args, stage, registration_sha256, **kwargs):
        self.stage, self.registration_sha256 = stage, registration_sha256
        super().__init__(*args, **kwargs)

    def _reserve(self, client, request, payload, state):
        count = sum(row.get("e1b_stage") == self.stage and row["role"] == client.role for row in logical_attempts(state))
        if count >= COUNTS[self.stage].get(client.role, 0):
            raise RunError("phase_request_inventory_exhausted")
        value = super()._reserve(client, request, payload, state)
        return {**value, "e1b_stage": self.stage, "registration_sha256": self.registration_sha256,
                "reply_return_channel": "json_object" if payload.get("response_format") == {"type": "json_object"} else "text",
                "online_decision_version": protocol_version(payload.get("messages", []))
                    if client.role == "online_fast" else None}

    def _record_receipt(self, client, row, data):
        super()._record_receipt(client, row, data)
        content = data["choices"][0]["message"]["content"]
        row["visible_content_sha256"] = text_sha(content)
        json_channel = client.role == "online_fast" and row["reply_return_channel"] == "json_object"
        row["client_return_content_sha256"] = text_sha(content if json_channel else content.strip())
        if client.role == "online_fast":
            reply = content.strip()
            if row["reply_return_channel"] == "json_object":
                first = content.lstrip()[:1]
                shape = {"content_utf8_bytes": len(content.encode("utf-8")),
                         "first_token_class": {"{": "object", "[": "array", '"': "string"}.get(first, "other"),
                         "json_valid": False}
                row["reply_shape"] = shape
                # Retain the failed response hash and a fixed category, never
                # exception text, generated field names or hidden reasoning.
                def reject(reason):
                    from aria.engine.policy.online_json_diagnostics import online_shape_diagnostic

                    shape["diagnostic"] = online_shape_diagnostic(content, row["online_decision_version"])
                    row["validation_error"] = reason
                    raise RunError(reason)

                def unique(pairs):
                    value = {}
                    for key, item in pairs:
                        if key in value:
                            reject("online_content_duplicate_key")
                        value[key] = item
                    return value

                try:
                    parsed = json.loads(content, object_pairs_hook=unique,
                        parse_constant=lambda _: reject("online_content_non_finite"))
                except json.JSONDecodeError as error:
                    shape["syntax_error_offset"] = error.pos
                    reject("online_content_invalid_json")
                except RecursionError:
                    reject("online_content_invalid_json")
                shape["json_valid"] = True
                if type(parsed) is not dict:
                    reject("online_content_not_object")
                schema = online_decision_tool_spec(row["online_decision_version"])["function"]["parameters"]
                shape["required_fields_present"] = {key: key in parsed for key in schema["required"]}
                shape["unknown_top_level_fields"] = len(set(parsed) - set(schema["properties"]))
                if "response" not in parsed:
                    reject("online_response_missing")
                reply = parsed["response"]
                if type(reply) is not str:
                    reject("online_response_not_string")
                if not reply.strip():
                    reject("online_response_empty")
                try:
                    # Use the product's exact schema, without repairing or
                    # adding fields, before committing a successful receipt.
                    parse_json_decision(content, row["online_decision_version"])
                except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                    reject("online_decision_schema_invalid")
            row["model_reply_sha256"] = text_sha(reply.strip())


def logical_attempts(state):
    """Legacy executions have exactly one physical attempt per logical call."""
    return state["attempts"]


def prepare_report_view(run_root, frozen, state):
    """Default runs already own every report; versioned compositions opt in."""
    return run_root


def export_protocol(frozen):
    return evaluation.draft_protocol(load_e1b_cases(), phase=frozen["stage"], case_ids=frozen["inputs"]["case_ids"])


def receipt_extensions():
    return {}


def assert_clean(state, *, expected_calls=None, final=False):
    acceptable_final = final and state.get("stop_reason") == "request_limit_reached"
    if ((state.get("stopped") and not acceptable_final) or state.get("unknown_cost_requests") != 0
            or type(state.get("calls_used")) is not int
            or len(state.get("attempts", [])) != state.get("calls_used")
            or (expected_calls is not None and state["calls_used"] != expected_calls)
            or any(row.get("status") != "received" or "cost" not in row.get("usage", {})
                   for row in state.get("attempts", []))):
        raise RunError("accounting_pending_unknown_failed_or_wrong_denominator")


def validate_attempt_registration(row, frozen):
    """Default strict binding; an authorized composition supplies its own join."""
    return row.get("registration_sha256") == frozen["seal_sha256"]


def qualify(run_root, frozen, state):
    stage = frozen["stage"]
    attempts = [r for r in logical_attempts(state) if r.get("e1b_stage") == stage]
    if Counter(r["role"] for r in attempts) != Counter(COUNTS[stage]):
        raise RunError("phase_call_denominator_mismatch")
    if any(not validate_attempt_registration(r, frozen) or not SHA.fullmatch(r.get("visible_content_sha256", ""))
           for r in attempts):
        raise RunError("actual_response_receipt_missing")
    lookup = {(r["operation_id"], r["session_id"]): r for r in attempts if r["role"] == "online_fast"}
    if len(lookup) != COUNTS[stage]["online_fast"]:
        raise RunError("duplicate_online_receipt_identity")
    positions, reports, memory_originals, used = [], {}, [], set()
    for case_id in frozen["inputs"]["case_ids"]:
        path = run_root / stage / case_id / "report.json"
        report = read_json(path)
        reports[case_id] = byte_sha(path)
        if report.get("evidence_kind") != "real_model" or report.get("phase") != stage:
            raise RunError("report_generation_identity_changed")
        if stage == "main":
            raw_capture = read_json(path.with_name("raw-proposals.json"))
            evaluation._validate_raw_capture_v2(raw_capture, path.parent,
                                               case_id, load_e1b_cases().source_sha256)
            if len(raw_capture["sessions"]) != 2 or not all(r["complete"] for r in raw_capture["sessions"]):
                raise RunError("two_complete_memory_originals_required")
            for number in (1, 2):
                original = read_json(path.with_name(f"memory-s{number}.first.json"))
                attempt = read_json(path.with_name(f"memory-s{number}.attempt.json"))
                linked = [r for r in attempts if r["role"] == "memory" and r["session_id"] == attempt["session_id"]]
                reflected = [r for r in attempts if r["role"] == "reflection" and r["session_id"] == attempt["session_id"]]
                if (len(linked) != 1 or original["request_hash_kind"] != "actual_wire"
                        or linked[0]["request_sha256"] != original["request_sha256"]
                        or linked[0]["client_return_content_sha256"] != original["visible_response_sha256"]
                        or len(reflected) != 1 or linked[0]["operation_id"] != reflected[0]["operation_id"]):
                    raise RunError("memory_original_receipt_mismatch")
                memory_originals.append({"case_id": case_id, "session": number,
                    "first_sha256": byte_sha(path.with_name(f"memory-s{number}.first.json")),
                    "receipt_sha256": digest(linked[0]), "reflection_receipt_sha256": digest(reflected[0])})
        branches = report["branches"]
        if stage == "main":
            branches = [{"arm": "shared", "repeat": 0, "state": report["shared"]["state"]}, *branches]
        for branch in branches:
            for index, turn in enumerate(branch["state"]["turns"]):
                row = lookup.get((turn["turn_id"], turn["session_id"]))
                identity = (turn["turn_id"], turn["session_id"])
                if identity in used:
                    raise RunError("online_receipt_reused_for_two_deliveries")
                used.add(identity)
                sha = text_sha(turn["response"])
                status = ("rule_only" if turn.get("source") == "rule_safety" else
                          "missing_receipt" if row is None else
                          "guard_modified" if turn.get("source") == "rule_protection" else
                          "model_unmodified" if sha == row.get("model_reply_sha256") else "guard_modified")
                positions.append({"case_id": case_id, "arm": branch["arm"], "repeat": branch["repeat"],
                    "turn": index, "turn_id": turn["turn_id"], "session_id": turn["session_id"],
                    "delivery": status, "delivered_reply_sha256": sha,
                    "receipt_sha256": digest(row) if row else None,
                    "model_reply_eligible": status == "model_unmodified"})
                if row is None or turn.get("status") != "completed":
                    raise RunError("turn_receipt_or_completed_delivery_missing")
    if set(lookup) != used:
        raise RunError("online_receipts_not_all_bound_to_saved_deliveries")
    return {"schema": "e1b_transport_qualification_v1", "stage": stage,
            "registration_sha256": frozen["seal_sha256"], "ledger_id": state["ledger_id"],
            "source_report_sha256": reports, "actual_counts": dict(Counter(r["role"] for r in attempts)),
            "positions": positions, "memory_originals": memory_originals, "effect_claim_eligible": False,
        "scope": "transport_and_delivery_only_independent_semantic_labels_still_required"}


def execute(root, metadata, registration, *, allow_local_env=False, budget_choice=None,
            campaign_authorization_sha256=None, stage_authorization_sha256=None):
    if allow_local_env is not True:
        raise RunError("explicit_local_key_permission_required")
    frozen = verify(root, metadata, registration)
    authorization, limit = validate_authorization(root, frozen, budget_choice=budget_choice,
        campaign_authorization_sha256=campaign_authorization_sha256,
        stage_authorization_sha256=stage_authorization_sha256)
    base, stage = campaign(root), frozen["stage"]
    binding = {"campaign_authorization_sha256": campaign_authorization_sha256,
               "effective_limit_usd": str(limit), "policy_sha256": digest(frozen["metadata"]["policy"]),
               "prior_known_usd": authorization["prior_known_usd"], "max_requests": MAX_WIRE_REQUESTS,
               "source_sha256": digest(frozen["source"]), "protocol_sha256": digest(frozen["protocol"])}
    ledger = base / "allowance.sqlite3"
    previous = None
    if stage == STAGES[0]:
        if ledger.exists() or (base / "budget-binding.json").exists():
            raise RunError("campaign_already_started_no_new_ledger")
    else:
        previous = read_receipt(root, STAGES[STAGES.index(stage) - 1])
        if not ledger.is_file() or read_json(base / "budget-binding.json") != binding:
            raise RunError("same_durable_budget_required")
    # These markers survive a crash, copied registration, or changed output.
    write_once(base / f"{stage}.started.json", {"registration_sha256": frozen["seal_sha256"],
               "stage_authorization_sha256": stage_authorization_sha256, "automatic_resume": False})
    if stage == STAGES[0]:
        write_once(base / "budget-binding.json", binding)
    output = base / stage
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "registration.json", frozen)
    guard = EvidenceGuard(ledger, max_requests=MAX_WIRE_REQUESTS, max_observed_cost_usd=str(limit),
        reservation_policy=frozen["metadata"]["policy"], stage=stage, registration_sha256=frozen["seal_sha256"],
        min_interval_seconds=MIN_INTERVAL_SECONDS)
    receipt = {"stage": stage, "registration_sha256": frozen["seal_sha256"], "status": "not_evaluable",
               "failure_type": None, "old_unknown_requests_retained": ACCEPTED_OLD_UNKNOWN,
               "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD),
               "cumulative_prior_known_usd": authorization["prior_known_usd"], "effect_claim_eligible": False,
               **receipt_extensions()}
    try:
        if previous is not None and guard.snapshot() != previous["accounting"]:
            raise RunError("retained_accounting_changed")
        assert_clean(guard.snapshot())
        roles = ("judge",) if stage == "judge" else ("online_fast", "memory", "reflection")
        for role in roles:
            guard.attach(make_client(role, "local-configuration-placeholder"))
        key = local_key(Path(root) / ".env")
        if stage == "judge":
            from scripts.e1b_secondary import run_secondary
            client = make_client("judge", key)
            guard.attach(client)
            receipt["judge"] = run_secondary(base / "main" / "blind-bundle", guard, client, output / "judge",
                expected_packet_sha256=frozen["inputs"]["packet_sha256"])
            if receipt["judge"].get("status") != "complete" or receipt["judge"].get("completed_positions") != 40:
                raise RunError("secondary_judgment_not_complete")
        else:
            token = SyntheticEmotionExperiment.create(output, f"e1b-{stage}-v1")
            def factory(path):
                clients = MinimalModelClients(*(make_client(role, key) for role in roles), model_calls_enabled=True)
                return E1bExperimentApp(path, experiment=token, clients=clients,
                    model_call_guard=guard, model_reservation_policy=frozen["metadata"]["policy"])
            runner = E1bRunner(load_e1b_cases(), root=token.root / "run", app_factory=factory, seed=SEED,
                               evidence_kind="real_model", allow_model_calls=True)
            for case_id in frozen["inputs"]["case_ids"]:
                verify(root, metadata, registration)
                validate_authorization(root, frozen, budget_choice=budget_choice,
                    campaign_authorization_sha256=campaign_authorization_sha256,
                    stage_authorization_sha256=stage_authorization_sha256)
                runner.run_case(case_id, phase=stage)
            state = guard.snapshot()
            assert_clean(state)
            report_root = prepare_report_view(runner.root, frozen, state)
            receipt["qualification"] = qualify(report_root, frozen, state)
            write_once(output / "qualification.json", receipt["qualification"])
            proto = export_protocol(frozen)
            proto["status"] = "registered"
            evaluation.export_packet(report_root, output / "blind-bundle", proto,
                                      registered_sha256=digest(proto))
            packet, mapping = evaluation.load_bundle(output / "blind-bundle")
            receipt.update(packet_sha256=digest(packet), mapping_sha256=digest(mapping))
            if stage == "main":
                evaluation.prepare_secondary(output / "blind-bundle")
                from scripts import e1b_formation as formation
                formation.export_packet(runner.root, output / "formation-bundle", proto, registered_sha256=digest(proto))
                formed, formed_key = formation.load_bundle(output / "formation-bundle")
                receipt.update(formation_packet_sha256=digest(formed), formation_mapping_sha256=digest(formed_key))
        state = guard.snapshot()
        expected = sum(sum(COUNTS[s].values()) for s in STAGES[:STAGES.index(stage) + 1])
        assert_clean(state, expected_calls=expected, final=stage == "judge")
        receipt["status"] = "completed"
    except Exception as error:
        receipt["failure_type"] = type(error).__name__
        guard.halt()
    finally:
        receipt["accounting"] = guard.snapshot()
        guard.close()
        receipt["receipt_sha256"] = digest(receipt)
        write_once(output / "execution-receipt.json", receipt)
    return {key: receipt[key] for key in ("stage", "status", "failure_type", "effect_claim_eligible", "receipt_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assessment = commands.add_parser("assess")
    assessment.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--stage", choices=STAGES, default="prescreen")
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
            command.add_argument("--budget-choice", choices=("12_usd1", "10_usd0_5"), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
    args = parser.parse_args()
    if args.command == "assess":
        from scripts.e1b_assessment import assess
        result = assess(ROOT)
        write_once(args.output, result)
    elif args.command == "prepare":
        result = prepare(ROOT, args.metadata, args.registration, stage=args.stage)
        result = {"stage": result["stage"], "seal_sha256": result["seal_sha256"],
                  "authorization": authorization_template(ROOT), "paid_calls": 0}
    elif args.command == "verify":
        result = verify(ROOT, args.metadata, args.registration)
        result = {"verified": True, "stage": result["stage"], "seal_sha256": result["seal_sha256"], "paid_calls": 0}
    else:
        result = execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
            budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
            stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
