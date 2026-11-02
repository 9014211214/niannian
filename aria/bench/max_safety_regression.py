"""Fixed safety evaluation using the ordinary Max/Sonnet product assembly.

Preparation is offline. Execution needs a fresh explicit caller budget, carries
every declared old hold, and reserves each actual wire request before sending.
The optional product safety review is product behavior, never a paid evaluator.
"""
from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import uuid

from aria.bench import safety_regression as fixed
from aria.engine.policy.max_release.guard import MaxReleaseCallGuard, BILLABLE_OUTPUTS
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.engine.policy.qwen_max_structured import build_qwen_max_structured_route, MODEL
from aria.engine.policy.release_configuration import derive_max_policy
from aria.engine.policy.fictional_context import (bind_fictional_conversation_history,
    bind_fictional_scenario_date, fictional_scenario_block)
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.release import create_max_release_clients
from aria.voice.minimal import MinimalWebApp

PROFILE = "release_max_v1"
PLAN_SCHEMA = "aria_max_safety_preparation_v1"
ROLES = ("online_fast", "safety_review")


def transport_contract():
    return {"physical_attempt_limit": 2 * fixed.COUNT, "parallelism": 1,
        "minimum_start_gap_seconds": 120, "automatic_retries": 0, "paid_response_replays": 0,
        "proven_pre_application_tls_extra_limit": 1, "fallback_models": [], "paid_judges": 0,
        "roles_per_case": list(ROLES), "conditional_safety_review": "ordinary_product_policy",
        "ordinary_timeout_seconds": 90, "reflection_timeout_seconds": 240}


def configuration(models_path, online_path, offline_path):
    raw = tuple(Path(p).read_bytes() for p in (models_path, online_path, offline_path))
    online = build_qwen_max_structured_route(raw[0], raw[1])
    offline = build_memory_structured_route(raw[0], raw[2])
    policy, proof = derive_max_policy(*raw)
    hashes = dict(zip(("models_sha256", "endpoints_sha256", "offline_endpoints_sha256"),
                      map(fixed.digest, raw)))
    return online, offline, policy, proof, hashes


def quote(policy, input_bound, role):
    spec = policy["models"][MODEL]
    if type(input_bound) is not int or not 0 < input_bound <= spec["max_input_tokens"]:
        raise ValueError("input_reservation_bound_invalid")
    return (Decimal(input_bound) * fixed.money(spec["input_usd_per_million"])
            + Decimal(BILLABLE_OUTPUTS[role]) * fixed.money(spec["output_usd_per_million"])) / 1000000


class SafetyRunGuard(MaxReleaseCallGuard):
    """Actual production guard, with fixed-case identity and evidence capture."""
    def begin_case(self, case, directory):
        self.case, self.directory = case, directory
        self.request_roles, self.request_bytes, self.scenario = [], 0, None

    def request_json(self, client, request, **options):
        if client.role not in ROLES or client.role in self.request_roles:
            raise MinimalCallStopped("one_physical_request_per_product_role_and_fixed_case")
        if client.role == "safety_review" and self.request_roles != ["online_fast"]:
            raise MinimalCallStopped("safety_review_requires_original_online_request")
        if fixed.source_binding() != self.expected_source:
            raise MinimalCallStopped("prepared_product_source_changed")
        payload = json.loads(request.data)
        self._validate_payload(client, payload, **options)
        self._binding(client)
        if client.role == "online_fast":
            messages = payload["messages"]
            carrier = json.loads(messages[-1]["content"])
            original = self.case["execution_input"]["messages"]
            if carrier != {"schema": "online_json_input_v2", "conversation_history": original[:-1],
                           "current_user_message": original[-1]["content"]}:
                raise ValueError("original_stimulus_changed")
            systems = [m["content"] for m in messages if m["role"] == "system"]
            if sum(len(s.encode("utf-8")) for s in systems) > 12000:
                raise ValueError("system_exceeds_12000_bytes")
            if self.scenario and sum(s.count(self.scenario) for s in systems) != 1:
                raise ValueError("fictional_date_not_preserved")
            self.request_bytes = len(request.data)
            if not self.prepare_only and len(request.data) > self.wire_limit:
                raise ValueError("prepared_online_wire_budget_exceeded")
        self.request_roles.append(client.role)
        path = self.directory / ("request.json" if client.role == "online_fast" else "safety-review-request.json")
        with path.open("xb") as stream:
            stream.write(request.data)
        if self.prepare_only:
            self._reserve(client, request, payload, self._snapshot())
            raise MinimalCallStopped("offline_actual_builder_capture_only")
        before = {r["request_id"] for r in self.snapshot()["attempts"]}
        try:
            return super().request_json(client, request, **options)
        finally:
            new = [r for r in self.snapshot()["attempts"] if r["request_id"] not in before]
            self._capture_first_outputs(client.role, new)

    def _capture_first_outputs(self, role, rows):
        if not rows:
            return  # A genuine pre-send refusal creates no physical response.
        if len(rows) != 1 or rows[0]["role"] != role:
            raise ValueError("unexpected_physical_attempts")
        row = rows[0]
        matches = []
        for binding_path in self.physical_dir.glob("*/attempt-binding.json"):
            binding = fixed.read(binding_path)
            if any(a.get("request_id") == row["request_id"] for a in binding.get("new_attempts", [])):
                matches.append((binding_path.parent, binding))
        if len(matches) != 1:
            raise ValueError("actual_physical_capture_binding_missing")
        directory, binding = matches[0]
        if binding.get("ledger_id") != self.ledger_id or binding.get("role") != role:
            raise ValueError("actual_physical_capture_identity_changed")
        prefix = "" if role == "online_fast" else "safety-review-"
        for name in ("first-body-receipt.json", "first-visible.json"):
            source = directory / name
            if source.is_file():
                raw = source.read_bytes()
                if name == "first-visible.json":
                    receipt = fixed.read(directory / "first-body-receipt.json")
                    if (receipt.get("visible_projection_saved") is not True
                            or receipt.get("public_projection_sha256") != fixed.digest(raw)):
                        raise ValueError("actual_visible_capture_changed")
                with (self.directory / (prefix + name)).open("xb") as stream:
                    stream.write(raw)
        fixed.write_new(self.directory / (prefix + "attempt-binding.json"), {
            "ledger_id": self.ledger_id, "request_id": row["request_id"], "role": role,
            "request_sha256": row["request_sha256"],
            "source_physical_directory": str(directory), "model_generated": True})


def position(case, directory, guard, online, offline, key):
    directory.mkdir()
    selected = create_max_release_clients(model_calls_enabled=True, api_key=key,
        base_url=fixed.BASE_URL, online_structured_route=online, memory_structured_route=offline)
    guard.begin_case(case, directory)
    app, stages = None, []
    try:
        app = MinimalWebApp(directory / "app", clients=selected, model_call_guard=guard)
        initial = app.start()
        if not initial["ok"]:
            raise ValueError("product_start_failed")
        session = initial["state"]["session_id"]
        engine, ctx = app._session("fictional-chen")
        if (ctx.turns or engine.persona.person_hypotheses or engine.persona.stated_preferences
                or engine.persona.episodic_memory or app._run["auto_review"]["enabled"]):
            raise ValueError("fresh_unseeded_product_required")
        data = case["execution_input"]
        bind_fictional_conversation_history(ctx, data["messages"][:-1])
        if "today" in data["setup"]:
            bind_fictional_scenario_date(ctx, data["setup"]["today"])
            guard.scenario = fictional_scenario_block(ctx)
        def prelude(frame):
            turn = next(t for t in frame["state"]["turns"] if t["turn_id"] == case["regression_id"])
            stage = {"kind": "protection", "text": turn["protection_prelude"]["protection"], "model_generated": False}
            fixed.write_new(directory / "protection.json", stage)
            fixed.write_new(directory / "prelude-frame.json", frame)
            stages.append(stage)
            return True
        result = app.submit(data["messages"][-1]["content"], case["regression_id"], session, on_safety_prelude=prelude)
        fixed.write_new(directory / "result-frame.json", result)
        turn = next((t for t in result["state"]["turns"] if t["turn_id"] == case["regression_id"]), {})
        stages.append({"kind": "final_body" if result["ok"] else "error",
                       "text": turn.get("response", "") if result["ok"] else result.get("error", "")})
        output = {"case_id": case["regression_id"], "release_profile": PROFILE,
            "submit_ok": result["ok"], "turn_status": turn.get("status"), "turn_source": turn.get("source"),
            "ordered_visible_stages": stages, "model_body_exposure_requires_independent_review": True,
            "browser_display_observed": False, "human_reading_observed": False,
            "wire_bytes": guard.request_bytes, "client_entered": "online_fast" in guard.request_roles,
            "request_roles": list(guard.request_roles),
            "decision_status": (ctx.working_memory.get("online_turn_decision") or {}).get("status")}
        fixed.write_new(directory / "product-output.json", output)
        if not app.seal(session)["ok"] or app._storage_failed:
            raise ValueError("product_sealing_failed")
        return output
    finally:
        if app is not None:
            app.close()


def prepare(suite_dir, models_path, online_path, offline_path, output):
    manifest, cases = fixed.load_suite(suite_dir)
    online, offline, policy, proof, hashes = configuration(models_path, online_path, offline_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, rows = fixed.source_binding(), []
    (output / "prepared-requests").mkdir()
    with fixed.no_network(), tempfile.TemporaryDirectory(prefix="aria-max-safety-") as scratch_name:
        scratch = Path(scratch_name)
        for case in cases:
            guard = SafetyRunGuard(scratch / (case["regression_id"] + ".sqlite3"), max_requests=2,
                max_observed_cost_usd="1000000", reservation_policy=policy)
            guard.prepare_only, guard.expected_source = True, source
            try:
                result = position(case, scratch / case["regression_id"], guard, online, offline, "inert-offline-builder")
                if not result["client_entered"] or guard.snapshot()["attempts"]:
                    raise ValueError("offline_builder_capture_invalid")
                raw = (scratch / case["regression_id"] / "request.json").read_bytes()
                with (output / "prepared-requests" / (case["regression_id"] + ".json")).open("xb") as stream:
                    stream.write(raw)
                bound = len(raw) + 1024
                rows.append({"case_id": case["regression_id"], "prepared_wire_bytes": len(raw),
                    "prepared_request_sha256": fixed.digest(raw), "wire_byte_limit": bound,
                    "online_reserved_estimate_usd": str(quote(policy, bound + 8192, "online_fast")),
                    "conditional_review_upper_bound_usd": str(quote(policy,
                        policy["models"][MODEL]["max_input_tokens"], "safety_review"))})
            finally:
                guard.close()
    estimate = sum((fixed.money(r["online_reserved_estimate_usd"]) + fixed.money(r["conditional_review_upper_bound_usd"])
                    for r in rows), Decimal(0))
    plan = {"schema": PLAN_SCHEMA, "release_profile": PROFILE, "run_id": uuid.uuid4().hex,
        "preparation_directory": str(output), "created_at": fixed.stamp(), "model_requests": 0,
        "credentials_read": False, "effect_status": "not_evaluated", "source": source,
        "suite_manifest_sha256": fixed.digest((Path(suite_dir) / "manifest.json").read_bytes()),
        **hashes, "policy": policy, "price_coverage": proof, "positions": rows,
        "worst_case_reserved_usd": str(estimate), "estimate_is_execution_admission_gate": False,
        "transport": transport_contract()}
    fixed.write_new(output / "plan.json", plan)
    return plan


def validate_billing(review, plan, cap):
    # Keep the old caller/price/hold checks, but not its all-batch reserve gate.
    # No amount can disappear: the remaining allowance is the global caller cap
    # minus this full known-plus-held baseline. The guard reserves actual wire.
    if review.get("offline_endpoints_sha256") != plan["offline_endpoints_sha256"]:
        raise ValueError("offline_billing_metadata_changed")
    baseline = fixed.validate_billing(review, {**plan, "worst_case_reserved_usd": "0"}, cap)
    if cap <= baseline:
        raise ValueError("no_remaining_caller_allowance")
    return baseline


def validate_preparation(suite_dir, models_path, online_path, offline_path, plan_path, operator):
    manifest, cases = fixed.load_suite(suite_dir)
    plan = fixed.read(plan_path)
    run_id = plan.get("run_id")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("release_profile") != PROFILE
            or plan.get("transport") != transport_contract()
            or type(run_id) is not str or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id)):
        raise ValueError("fresh_max_preparation_required")
    anchor_value = plan.get("preparation_directory")
    if not isinstance(anchor_value, str) or not Path(anchor_value).is_absolute():
        raise ValueError("original_preparation_directory_required")
    anchor = Path(anchor_value)
    if str(anchor.resolve()) != anchor_value or (anchor / "plan.json").read_bytes() != Path(plan_path).read_bytes():
        raise ValueError("original_preparation_identity_changed")
    if manifest.get("fixture_only") is True:
        raise ValueError("fixture_suite_cannot_authorize_paid_execution")
    if not operator.strip() or plan.get("source") != fixed.source_binding():
        raise ValueError("operator_and_unchanged_prepared_source_required")
    if fixed.digest((Path(suite_dir) / "manifest.json").read_bytes()) != plan["suite_manifest_sha256"]:
        raise ValueError("prepared_suite_changed")
    online, offline, policy, proof, hashes = configuration(models_path, online_path, offline_path)
    if any(plan.get(k) != value for k, value in hashes.items()) or plan["policy"] != policy or plan["price_coverage"] != proof:
        raise ValueError("prepared_metadata_or_policy_changed")
    if [p["case_id"] for p in plan["positions"]] != manifest["ordered_case_ids"]:
        raise ValueError("prepared_positions_changed")
    for row in plan["positions"]:
        saved = anchor / "prepared-requests" / (row["case_id"] + ".json")
        if (fixed.digest(saved.read_bytes()) != row["prepared_request_sha256"]
                or saved.stat().st_size != row["prepared_wire_bytes"]
                or row["wire_byte_limit"] != row["prepared_wire_bytes"] + 1024
                or fixed.money(row["online_reserved_estimate_usd"]) != quote(policy, row["wire_byte_limit"] + 8192, "online_fast")
                or fixed.money(row["conditional_review_upper_bound_usd"]) != quote(policy, policy["models"][MODEL]["max_input_tokens"], "safety_review")):
            raise ValueError("prepared_request_or_quote_changed")
    expected_total = sum((fixed.money(r["online_reserved_estimate_usd"]) + fixed.money(r["conditional_review_upper_bound_usd"])
                          for r in plan["positions"]), Decimal(0))
    if fixed.money(plan["worst_case_reserved_usd"]) != expected_total or plan.get("estimate_is_execution_admission_gate") is not False:
        raise ValueError("prepared_estimate_changed")
    return manifest, cases, plan, anchor, online, offline, policy


def execute(suite_dir, models_path, online_path, offline_path, plan_path, review_path, output, cap, operator):
    values = validate_preparation(suite_dir, models_path, online_path, offline_path, plan_path, operator)
    manifest, cases, plan, anchor, online, offline, policy = values
    review = fixed.read(review_path)
    baseline = validate_billing(review, plan, cap)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("output_must_be_new")
    claim = {"created_at": fixed.stamp(), "operator": operator, "pid": os.getpid(), "run_id": plan["run_id"],
        "release_profile": PROFILE, "authorization": "explicit_cli_execute_and_budget",
        "budget_cap_usd": str(cap), "plan_sha256": fixed.digest(Path(plan_path).read_bytes()),
        "inherited_aria_task_authorization": False}
    fixed.write_new(anchor / "execution-claim.json", {**claim, "output": str(output)})
    output.mkdir(parents=True, exist_ok=False)
    fixed.write_new(output / "claim.json", claim)
    guard, error_type = None, None
    results = [{"case_id": c["regression_id"], "status": "not_executed", "reason": "batch_stopped_before_position"} for c in cases]
    try:
        fixed.write_new(output / "plan.json", plan)
        fixed.write_new(output / "billing-review.json", review)
        for name, path in zip(("models.json", "online-endpoints.json", "offline-endpoints.json"),
                              (models_path, online_path, offline_path)):
            with (output / name).open("xb") as stream:
                stream.write(Path(path).read_bytes())
        (output / "suite").mkdir()
        for name in ("manifest.json", *manifest["files"]):
            with (output / "suite" / name).open("xb") as stream:
                stream.write((Path(suite_dir) / name).read_bytes())
        guard = SafetyRunGuard(output / "ledger.sqlite3", max_requests=2 * fixed.COUNT,
            max_observed_cost_usd=str(cap - baseline), reservation_policy=policy)
        guard.prepare_only, guard.expected_source = False, plan["source"]
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("explicit_execution_missing_environment_key")
        for index, (case, planned) in enumerate(zip(cases, plan["positions"])):
            guard.wire_limit = planned["wire_byte_limit"]
            before = len(guard.snapshot()["attempts"])
            results[index] = {"case_id": case["regression_id"], "status": "started", "reason": "result_not_completed"}
            result = position(case, output / case["regression_id"], guard, online, offline, key)
            attempts = guard.snapshot()["attempts"][before:]
            results[index] = {"case_id": case["regression_id"], "status": "completed" if result["submit_ok"] else "failed",
                "attempt_ids": [a["request_id"] for a in attempts], "result": result}
            if (not 1 <= len(attempts) <= 2 or any(a["status"] != "received" or "cost" not in a["usage"] for a in attempts)
                    or guard.snapshot()["stopped"]):
                break
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        snapshot = guard.snapshot() if guard else {"attempts": [], "known_cost_usd": "0", "reserved_cost_usd": "0", "calls_used": 0}
        if guard:
            guard.close()
        # A raised storage/transport error may occur before position() returns.
        # Account for every actual physical request without inventing a reply.
        for result in results:
            if result["status"] == "started":
                result["attempt_ids"] = [a["request_id"] for a in snapshot["attempts"]
                                         if a["operation_id"] == result["case_id"]]
        fixed.write_new(output / "results.json", {"schema": "aria_safety_results_v1", "release_profile": PROFILE,
            "positions": results, "fixed_cases": fixed.COUNT, "fixed_cells": fixed.CELLS,
            "effect_status": "awaiting_independent_scoring", "old_unknown_holds": review["unknown_holds"],
            "old_holds_released_usd": "0", "baseline_usd": str(baseline), "ledger": snapshot, "exception_type": error_type})
        fixed.write_new(output / "scores-template.json", {"suite_version": manifest["version"],
            "results_sha256": fixed.digest((output / "results.json").read_bytes()), "grader": "",
            "grader_is_neither_author_nor_implementer": False,
            "cells": [{**c, "status": None, "reason": "", "evidence": []} for c in manifest["cells"]]})
        fixed.write_new(output / "closure.json", {"closed_at": fixed.stamp(), "can_resume": False,
            "status": "all_positions_recorded" if all(r["status"] in {"completed", "failed"} for r in results) else "stopped_partial",
            "files": fixed.file_map(output)})


def verify_run(directory, scores_path=None):
    directory = Path(directory)
    summary = fixed.verify_run(directory, scores_path)
    plan, results = fixed.read(directory / "plan.json"), fixed.read(directory / "results.json")
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("release_profile") != PROFILE
            or plan.get("transport") != transport_contract()
            or results.get("release_profile") != PROFILE):
        raise ValueError("max_assembly_evidence_required")
    _, _, policy, proof, hashes = configuration(directory / "models.json",
        directory / "online-endpoints.json", directory / "offline-endpoints.json")
    if policy != plan["policy"] or proof != plan["price_coverage"] or any(plan.get(k) != value for k, value in hashes.items()):
        raise ValueError("closed_public_metadata_binding_changed")
    review = fixed.read(directory / "billing-review.json")
    if any(review.get(k) != value for k, value in hashes.items()):
        raise ValueError("closed_billing_metadata_binding_changed")
    attempts = results["ledger"]["attempts"]
    by_id = {row["request_id"]: row for row in attempts}
    seen = []
    for result in results["positions"]:
        ids = result.get("attempt_ids", [])
        if len(ids) > 2 or any(identity not in by_id for identity in ids):
            raise ValueError("case_attempt_binding_changed")
        rows = [by_id[identity] for identity in ids]
        if [r["role"] for r in rows] not in ([], ["online_fast"], list(ROLES)):
            raise ValueError("product_roles_changed")
        if any(r["operation_id"] != result["case_id"] or r["model"] != MODEL for r in rows):
            raise ValueError("case_request_identity_changed")
        seen.extend(ids)
    if len(seen) != len(set(seen)) or set(seen) != set(by_id):
        raise ValueError("unbound_or_duplicate_physical_attempt")
    summary.update(release_profile=PROFILE, conditional_review_is_product_behavior=True,
                   fixed_criteria_unchanged=True, paid_judges=0)
    return summary
