"""Local fixed safety regression. Default: no credentials and no network.

The public fixed suite is a separate, reviewed asset. This runner neither
creates it nor imports historical task code. A fresh explicit paid invocation
authorizes only one new output directory; no old budget is carried implicitly.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import urllib.error
import uuid
from unittest.mock import patch

from aria.engine.policy import minimal_calls
from aria.engine.policy.fictional_context import (bind_fictional_conversation_history,
    bind_fictional_scenario_date, fictional_scenario_block)
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_structured import build_online_structured_route
from aria.engine.policy.reserved_calls import ReservedCallGuard, validate_reservation_policy
from aria.release import ONLINE_MODEL, ONLINE_PROVIDER, create_release_clients
from aria.voice.minimal import MinimalWebApp

SCHEMA = "aria_fixed_safety_suite_v1"
COUNT, CELLS = 87, 696
BASE_URL = "https://openrouter.ai/api/v1"
HIDDEN = {"reasoning", "reasoning_details", "reasoning_content"}


def digest(value):
    return sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_field")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError("nonfinite_json")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite)


def write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def money(value):
    if type(value) not in (str, int, float):
        raise ValueError("invalid_amount")
    text = str(value)
    if len(text) > 64:
        raise ValueError("amount_precision_out_of_bounds")
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise ValueError("invalid_amount") from None
    if not amount.is_finite() or amount < 0 or amount > 1000000:
        raise ValueError("invalid_amount")
    # Bounded input, token and batch limits keep all arithmetic exact within
    # Decimal's 28-digit context. Reject tiny/overspecified values explicitly;
    # never silently round a hold away when adding it to the existing floor.
    parts = amount.as_tuple()
    if len(parts.digits) > 18 or not -12 <= parts.exponent <= 6:
        raise ValueError("amount_precision_out_of_bounds")
    return amount


def file_map(directory):
    return {p.relative_to(directory).as_posix(): digest(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.name != "closure.json"}


def source_binding():
    """Bind the installed/imported product tree, not a task or Git history."""
    import aria
    root = Path(aria.__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): digest(p.read_bytes())
             for p in sorted(root.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {"aria_files": files, "runner_sha256": digest(Path(__file__).read_bytes())}


def load_suite(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("fictional_only") is not True
            or manifest.get("case_count") != COUNT or manifest.get("cell_count") != CELLS
            or not isinstance(manifest.get("version"), str) or not manifest["version"]):
        raise ValueError("fixed87_696_fictional_suite_required")
    if set(manifest.get("files", {})) != {"inputs.json", "references.json", "protocol.json"}:
        raise ValueError("suite_file_set_invalid")
    for name, expected in manifest["files"].items():
        target = directory / name
        if target.is_symlink() or digest(target.read_bytes()) != expected:
            raise ValueError("suite_file_changed")
    inputs = read(directory / "inputs.json")
    cases = inputs.get("cases")
    if inputs.get("fictional_only") is not True or type(cases) is not list or len(cases) != COUNT:
        raise ValueError("complete_fixed_cases_required")
    ids = [row.get("regression_id") for row in cases]
    if ids != [f"FS-{n:03d}" for n in range(1, COUNT + 1)] or manifest.get("ordered_case_ids") != ids:
        raise ValueError("ordered_fixed_case_identity_mismatch")
    for row in cases:
        item = row.get("execution_input", {})
        if set(item) != {"messages", "setup", "context_material"} or item["context_material"] != []:
            raise ValueError("unsupported_execution_context")
        messages, setup = item["messages"], item["setup"]
        if (type(messages) is not list or not messages or any(type(m) is not dict
                or set(m) != {"role", "content"} or m["role"] not in {"user", "assistant"}
                or type(m["content"]) is not str for m in messages)
                or messages[-1]["role"] != "user" or not messages[-1]["content"].strip()
                or not 1 <= len(messages[-1]["content"]) <= 500):
            raise ValueError("original_ordered_messages_required")
        if type(setup) is not dict or set(setup) - {"language", "today"} or setup.get("language", "zh-CN") != "zh-CN":
            raise ValueError("unsupported_fictional_setup")
        if "today" in setup and date.fromisoformat(setup["today"]).isoformat() != setup["today"]:
            raise ValueError("invalid_fictional_date")
    cells = manifest.get("cells", [])
    if (type(cells) is not list or len(cells) != CELLS
            or any(type(c) is not dict or set(c) != {"cell_id", "case_id", "layer", "criterion"}
                   or not all(isinstance(v, str) and v for v in c.values())
                   or c["case_id"] not in ids or c["layer"] not in {"raw", "delivery"} for c in cells)
            or len({c["cell_id"] for c in cells}) != CELLS
            or len({(c["case_id"], c["layer"], c["criterion"]) for c in cells}) != CELLS
            or any(sum(c["case_id"] == id_ and c["layer"] == layer for c in cells) != 4
                   for id_ in ids for layer in ("raw", "delivery"))):
        raise ValueError("fixed_scoring_coordinates_required")
    return manifest, cases


def policy_from_metadata(models_raw, endpoints_raw):
    """All published input tiers and cache prices; no discounts assumed.

    The small rate walk follows prepare_model_policy._all_tier_prices. It is
    local here so an installed wheel has no dependency on checkout scripts.
    Missing non-token terms still require the separate human billing review.
    """
    route = build_online_structured_route(models_raw, endpoints_raw, model=ONLINE_MODEL, provider=ONLINE_PROVIDER)
    endpoint = next(e for e in json.loads(endpoints_raw)["data"]["endpoints"] if e["tag"] == ONLINE_PROVIDER)
    context, maximum = endpoint.get("context_length"), endpoint.get("max_completion_tokens")
    prompt_max = endpoint.get("max_prompt_tokens", context)
    if any(type(v) is not int or v < 4096 for v in (context, maximum, prompt_max)):
        raise ValueError("invalid_endpoint_token_limits")
    price = endpoint.get("pricing")
    if type(price) is not dict or not {"prompt", "completion"} <= price.keys():
        raise ValueError("missing_published_token_prices")
    tiers = price.get("overrides", [])
    if type(tiers) is not list:
        raise ValueError("invalid_price_tiers")
    inputs, outputs, previous = [], [], 0
    input_keys = {"prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h"}
    scalar_keys = input_keys | {"completion", "request", "discount", "web_search"}
    for index, tier in enumerate([price, *tiers]):
        if type(tier) is not dict:
            raise ValueError("invalid_price_tier")
        if index:
            boundary = tier.get("min_prompt_tokens")
            if type(boundary) is not int or not previous < boundary <= context or not (input_keys | {"completion"}) & tier.keys():
                raise ValueError("unsupported_price_tier")
            previous = boundary
        for key, value in tier.items():
            if (not index and key == "overrides") or (index and key == "min_prompt_tokens"):
                continue
            if index and key not in scalar_keys:
                raise ValueError("unsupported_price_tier_dimension")
            amount = money(value)
            if (key == "request" or key not in scalar_keys) and amount != 0:
                raise ValueError("unreserved_additional_charge")
            if key in input_keys | {"completion", "discount"} and amount > 1:
                raise ValueError("invalid_token_rate")
            if key in input_keys:
                inputs.append(amount)
            elif key == "completion":
                outputs.append(amount)
    bound = route.binding()
    policy = validate_reservation_policy({"schema": "aria_reservation_v1", "framing_tokens": 8192,
        "models": {ONLINE_MODEL: {"input_usd_per_million": str(max(inputs) * 1000000),
            "output_usd_per_million": str(max(outputs) * 1000000), "provider": ONLINE_PROVIDER,
            "response_models": list(dict.fromkeys([ONLINE_MODEL, bound["canonical_slug"]])),
            "response_providers": [bound["provider_name"]], "max_input_tokens": min(prompt_max, context - 4096),
            "max_output_tokens": maximum, "metadata_sha256": digest(endpoints_raw)}}})
    return route, policy


@contextmanager
def no_network():
    def denied(*args, **kwargs):
        raise RuntimeError("network_forbidden_in_dry_run")
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", denied))
        stack.enter_context(patch.object(socket, "create_connection", denied))
        stack.enter_context(patch.object(minimal_calls, "open_unredirected", denied))
        yield


def ordered_visible(raw):
    """Keep duplicate visible fields in order; never keep hidden channel fields."""
    class Pairs(list):
        pass
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=Pairs)
    def generic(node):
        if isinstance(node, Pairs):
            return {"object_pairs": [[k, generic(v)] for k, v in node if k not in HIDDEN]}
        if isinstance(node, list):
            return [generic(v) for v in node]
        if isinstance(node, float) and not math.isfinite(node):
            return {"invalid_number": True}
        return node
    def select(node, fields):
        if not isinstance(node, Pairs):
            return {"invalid_object_shape": True}
        return {"object_pairs": [[k, fields[k](v)] for k, v in node if k in fields]}
    def choice(node):
        return select(node, {"index": generic, "finish_reason": generic,
            "message": lambda n: select(n, {k: generic for k in ("role", "content", "refusal", "tool_calls", "function_call")})})
    return select(value, {"choices": lambda n: [choice(c) for c in n] if type(n) is list else {"invalid_choices": True},
        **{k: generic for k in ("content", "message", "messages", "output", "outputs", "response", "completion", "text", "delta", "tool_calls", "function_call", "refusal")}})


def visible_evidence_present(value):
    """Presence only; this never qualifies model-body exposure or success."""
    if isinstance(value, dict):
        for key, item in value.get("object_pairs", []):
            if key in {"content", "refusal", "text", "tool_calls", "function_call"} and item not in (None, "", []):
                return True
            if visible_evidence_present(item):
                return True
    if isinstance(value, list):
        return any(visible_evidence_present(item) for item in value)
    return False


class CapturedResponse:
    def __init__(self, raw, directory):
        self.raw, self.directory, self.captured = raw, directory, False

    def __getattr__(self, key):
        return getattr(self.raw, key)

    def __enter__(self):
        self.raw = self.raw.__enter__()
        return self

    def __exit__(self, *args):
        return self.raw.__exit__(*args)

    def read(self, size=-1):
        try:
            raw = self.raw.read(size)
        except BaseException as error:
            partial = getattr(error, "partial", None)
            if isinstance(partial, bytes):
                self.save(partial, partial=True)
            raise
        self.save(raw)
        return raw

    def save(self, raw, partial=False):
        if self.captured:
            raise ValueError("first_body_already_captured")
        self.captured = True
        receipt = {"body_bytes": len(raw) if isinstance(raw, bytes) else None,
            "body_sha256": digest(raw) if isinstance(raw, bytes) else None,
            "partial": partial, "full_body_stored": False, "hidden_reasoning_stored": False,
            "visible_projection_saved": False}
        if isinstance(raw, bytes) and len(raw) <= minimal_calls.MAX_MODEL_RESPONSE_BYTES:
            try:
                projection = ordered_visible(raw)
            except (ValueError, UnicodeError, RecursionError):
                receipt["visible_capture_missing_reason"] = "outer_json_not_parseable"
            else:
                write_new(self.directory / "first-visible.json", projection)
                receipt["visible_projection_saved"] = True
        write_new(self.directory / "first-body-receipt.json", receipt)


class OfflineDisabled(MockLLMClient):
    def complete(self, messages):
        raise ValueError("offline_roles_disabled_in_safety_regression")


class RunGuard(ReservedCallGuard):
    """One physical attempt per fixed case, no automatic retries or recovery."""
    def request_json(self, client, request, **options):
        if self.request_seen:
            raise MinimalCallStopped("duplicate_model_operation")
        self.request_seen = True
        if source_binding() != self.expected_source:
            raise ValueError("product_source_changed")
        payload = json.loads(request.data)
        self._validate_payload(client, payload, **options)
        messages = payload["messages"]
        carrier = json.loads(messages[-1]["content"])
        if carrier != {"schema": "online_json_input_v2", "conversation_history": self.case["execution_input"]["messages"][:-1],
                       "current_user_message": self.case["execution_input"]["messages"][-1]["content"]}:
            raise ValueError("original_stimulus_changed")
        systems = [m["content"] for m in messages if m["role"] == "system"]
        if sum(len(s.encode("utf-8")) for s in systems) > 12000:
            raise ValueError("system_exceeds_12000_bytes")
        if self.scenario and sum(s.count(self.scenario) for s in systems) != 1:
            raise ValueError("fictional_date_not_preserved")
        with (self.directory / "request.json").open("xb") as stream:
            stream.write(request.data)
        self.request_bytes = len(request.data)
        if self.prepare_only:
            self._reserve(client, request, payload, self._snapshot())
            raise MinimalCallStopped("dry_run_capture_only")
        if len(request.data) > self.wire_limit:
            raise ValueError("prepared_wire_budget_exceeded")
        original_open = minimal_calls.open_unredirected
        def observed(req, **kwargs):
            try:
                response = original_open(req, **kwargs)
            except urllib.error.HTTPError as error:
                # The existing guard remains responsible for parsing billing.
                error.fp = CapturedResponse(error.fp, self.directory) if error.fp is not None else None
                raise
            return CapturedResponse(response, self.directory)
        with patch.object(minimal_calls, "open_unredirected", observed):
            return super().request_json(client, request, **options)

    def _reserve(self, client, request, payload, state):
        reserved = super()._reserve(client, request, payload, state)
        if not self.prepare_only:
            total = money(state["known_cost_usd"] or "0") + money(state["reserved_cost_usd"])
            total += money(reserved["reservation"]["reserved_cost_usd"]) + self.future_reserved
            if total > money(self.threshold):
                raise MinimalCallStopped("pre_send_budget_exceeded")
        return reserved


def position(case, directory, guard, route, key):
    directory.mkdir()
    selected = create_release_clients(model_calls_enabled=True, api_key=key, base_url=BASE_URL,
                                      online_structured_route=route)
    selected = replace(selected, memory=OfflineDisabled(role="memory"), reflection=OfflineDisabled(role="reflection"))
    guard.directory, guard.case = directory, case
    guard.request_seen, guard.request_bytes, guard.scenario = False, 0, None
    app = None
    stages = []
    try:
        app = MinimalWebApp(directory / "app", clients=selected, model_call_guard=guard)
        initial = app.start()
        if not initial["ok"]:
            raise ValueError("product_start_failed")
        session_id = initial["state"]["session_id"]
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
            write_new(directory / "protection.json", stage)
            write_new(directory / "prelude-frame.json", frame)
            stages.append(stage)
            return True
        result = app.submit(data["messages"][-1]["content"], case["regression_id"], session_id, on_safety_prelude=prelude)
        write_new(directory / "result-frame.json", result)
        turn = next((t for t in result["state"]["turns"] if t["turn_id"] == case["regression_id"]), {})
        stages.append({"kind": "final_body" if result["ok"] else "error", "text": turn.get("response", "") if result["ok"] else result.get("error", "")})
        output = {"case_id": case["regression_id"], "submit_ok": result["ok"],
            "turn_status": turn.get("status"), "turn_source": turn.get("source"),
            "ordered_visible_stages": stages, "model_body_exposure_requires_independent_review": True,
            "browser_display_observed": False, "human_reading_observed": False,
            "wire_bytes": guard.request_bytes, "client_entered": guard.request_seen,
            "decision_status": (ctx.working_memory.get("online_turn_decision") or {}).get("status")}
        write_new(directory / "product-output.json", output)
        if not app.seal(session_id)["ok"] or app._storage_failed:
            raise ValueError("product_sealing_failed")
        return output
    finally:
        if app is not None:
            app.close()


def quote(policy, byte_bound):
    spec = policy["models"][ONLINE_MODEL]
    if byte_bound + 8192 > spec["max_input_tokens"]:
        raise ValueError("prepared_input_bound_exceeds_endpoint")
    return (Decimal(byte_bound + 8192) * money(spec["input_usd_per_million"])
            + Decimal(4096) * money(spec["output_usd_per_million"])) / 1000000


def prepare(suite_dir, models_path, endpoints_path, output):
    manifest, cases = load_suite(suite_dir)
    models_raw, endpoints_raw = Path(models_path).read_bytes(), Path(endpoints_path).read_bytes()
    route, policy = policy_from_metadata(models_raw, endpoints_raw)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected_source = source_binding()
    rows = []
    (output / "prepared-requests").mkdir()
    with no_network(), tempfile.TemporaryDirectory(prefix="aria-safety-prepare-") as scratch:
        scratch = Path(scratch)
        for case in cases:
            guard = RunGuard(scratch / (case["regression_id"] + ".sqlite3"), max_requests=1, max_observed_cost_usd="1000000", reservation_policy=policy)
            guard.prepare_only, guard.expected_source = True, expected_source
            try:
                result = position(case, scratch / case["regression_id"], guard, route, "inert-dry-run-no-credential")
                if not result["client_entered"]:
                    raise ValueError("fixed_case_did_not_reach_product_builder")
                # The fresh empty care observation may have a new timestamp.
                # This is a byte budget allowance, not normalization of evidence.
                bound = result["wire_bytes"] + 1024
                request_raw = (scratch / case["regression_id"] / "request.json").read_bytes()
                with (output / "prepared-requests" / (case["regression_id"] + ".json")).open("xb") as stream:
                    stream.write(request_raw)
                rows.append({"case_id": case["regression_id"], "prepared_wire_bytes": result["wire_bytes"],
                    "prepared_request_sha256": digest(request_raw),
                    "wire_byte_limit": bound, "reserved_usd": str(quote(policy, bound))})
                if guard.snapshot()["attempts"]:
                    raise ValueError("dry_run_created_model_attempt")
            finally:
                guard.close()
    plan = {"schema": "aria_safety_preparation_v2", "run_id": uuid.uuid4().hex,
        "preparation_directory": str(output), "created_at": stamp(), "model_requests": 0,
        "credentials_read": False, "effect_status": "not_evaluated", "source": expected_source,
        "suite_manifest_sha256": digest((Path(suite_dir) / "manifest.json").read_bytes()),
        "models_sha256": digest(models_raw), "endpoints_sha256": digest(endpoints_raw),
        "policy": policy, "positions": rows, "worst_case_reserved_usd": str(sum((money(r["reserved_usd"]) for r in rows), Decimal(0))),
        "transport": {"physical_attempt_limit": COUNT, "parallelism": 1, "minimum_start_gap_seconds": 120,
                      "automatic_retries": 0, "fallback_models": [], "paid_judges": 0}}
    write_new(output / "plan.json", plan)
    return plan


def validate_billing(review, plan, cap):
    when = datetime.fromisoformat(review["verified_at"])
    if when.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - when).total_seconds() <= 86400:
        raise ValueError("billing_review_must_be_fresh_within_24_hours")
    if (review.get("fixture") is not False or review.get("billing_mode") != "standard_credits"
            or review.get("account_and_workspace_mode_reviewed") is not True
            or review.get("all_applicable_prices_reviewed") is not True
            or money(review.get("non_token_surcharges_usd")) != 0
            or review.get("models_sha256") != plan["models_sha256"]
            or review.get("endpoints_sha256") != plan["endpoints_sha256"]
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise ValueError("explicit_billing_review_required")
    holds = review.get("unknown_holds")
    if (type(holds) is not list or len(holds) > 10000 or any(type(h) is not dict or set(h) != {"id", "reserved_usd"}
        or not isinstance(h["id"], str) or not h["id"] for h in holds)
        or len({h["id"] for h in holds}) != len(holds)):
        raise ValueError("complete_prior_hold_list_required")
    if any(money(h["reserved_usd"]) <= 0 for h in holds):
        raise ValueError("unknown_fee_requires_positive_reservation")
    baseline = money(review["known_usd"]) + sum((money(h["reserved_usd"]) for h in holds), Decimal(0))
    if cap <= 0 or baseline + money(plan["worst_case_reserved_usd"]) > cap:
        raise ValueError("entire_batch_plus_prior_holds_exceeds_budget")
    return baseline


def execute(suite_dir, models_path, endpoints_path, plan_path, review_path, output, cap, operator):
    """Called only by the explicit CLI action; reads one Key after unique claim."""
    manifest, cases = load_suite(suite_dir)
    plan, review = read(plan_path), read(review_path)
    if (plan.get("schema") != "aria_safety_preparation_v2" or type(plan.get("run_id")) is not str
            or len(plan["run_id"]) != 32 or any(c not in "0123456789abcdef" for c in plan["run_id"])):
        raise ValueError("fresh_preparation_identity_required")
    anchor_value = plan.get("preparation_directory")
    if not isinstance(anchor_value, str):
        raise ValueError("original_preparation_directory_required")
    anchor = Path(anchor_value)
    if not anchor.is_absolute() or str(anchor.resolve()) != anchor_value:
        raise ValueError("original_preparation_directory_required")
    # Copies retain this original absolute anchor and exact plan bytes. Their
    # claim goes to the original directory, never to the copy's parent.
    # Moving/removing the original requires a fresh reviewed preparation.
    if (anchor / "plan.json").read_bytes() != Path(plan_path).read_bytes():
        raise ValueError("original_preparation_identity_changed")
    if manifest.get("fixture_only") is True:
        raise ValueError("fixture_suite_cannot_authorize_paid_execution")
    if not operator.strip() or plan["source"] != source_binding():
        raise ValueError("operator_and_unchanged_prepared_source_required")
    if digest((Path(suite_dir) / "manifest.json").read_bytes()) != plan["suite_manifest_sha256"]:
        raise ValueError("prepared_suite_changed")
    models_raw, endpoints_raw = Path(models_path).read_bytes(), Path(endpoints_path).read_bytes()
    if digest(models_raw) != plan["models_sha256"] or digest(endpoints_raw) != plan["endpoints_sha256"]:
        raise ValueError("prepared_public_metadata_changed")
    route, policy = policy_from_metadata(models_raw, endpoints_raw)
    if policy != plan["policy"] or [p["case_id"] for p in plan["positions"]] != manifest["ordered_case_ids"]:
        raise ValueError("prepared_policy_or_positions_changed")
    for row in plan["positions"]:
        if row["wire_byte_limit"] != row["prepared_wire_bytes"] + 1024 or money(row["reserved_usd"]) != quote(policy, row["wire_byte_limit"]):
            raise ValueError("prepared_quote_changed")
        saved = Path(plan_path).parent / "prepared-requests" / (row["case_id"] + ".json")
        if digest(saved.read_bytes()) != row["prepared_request_sha256"] or saved.stat().st_size != row["prepared_wire_bytes"]:
            raise ValueError("prepared_actual_request_changed")
    if money(plan["worst_case_reserved_usd"]) != sum((money(p["reserved_usd"]) for p in plan["positions"]), Decimal(0)):
        raise ValueError("prepared_total_changed")
    baseline = validate_billing(review, plan, cap)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("output_must_be_new")
    claim = {"created_at": stamp(), "operator": operator, "pid": os.getpid(), "run_id": plan["run_id"],
        "authorization": "explicit_cli_execute_and_budget", "budget_cap_usd": str(cap),
        "plan_sha256": digest(Path(plan_path).read_bytes()), "inherited_aria_task_authorization": False}
    # One preparation authorizes at most one local execution, including copies
    # made before the first claim. Never delete/rewrite this marker.
    write_new(anchor / "execution-claim.json", {**claim, "output": str(output)})
    output.mkdir(parents=True, exist_ok=False)  # Fresh evidence directory; never reopen.
    write_new(output / "claim.json", claim)
    guard, error_type = None, None
    results = [{"case_id": c["regression_id"], "status": "not_executed", "reason": "batch_stopped_before_position"} for c in cases]
    try:
        write_new(output / "plan.json", plan)
        write_new(output / "billing-review.json", review)
        (output / "suite").mkdir()
        for name in ("manifest.json", *manifest["files"]):
            with (output / "suite" / name).open("xb") as stream:
                stream.write((Path(suite_dir) / name).read_bytes())
        guard = RunGuard(output / "ledger.sqlite3", max_requests=COUNT,
            max_observed_cost_usd=str(cap - baseline), reservation_policy=policy, min_interval_seconds=120)
        guard.prepare_only, guard.expected_source = False, plan["source"]
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("explicit_execution_missing_environment_key")
        for i, (case, planned) in enumerate(zip(cases, plan["positions"])):
            guard.wire_limit = planned["wire_byte_limit"]
            guard.future_reserved = sum((money(p["reserved_usd"]) for p in plan["positions"][i + 1:]), Decimal(0))
            before = len(guard.snapshot()["attempts"])
            results[i] = {"case_id": case["regression_id"], "status": "started", "reason": "result_not_completed"}
            result = position(case, output / case["regression_id"], guard, route, key)
            attempts = guard.snapshot()["attempts"][before:]
            results[i] = {"case_id": case["regression_id"], "status": "completed" if result["submit_ok"] else "failed",
                "attempt_ids": [a["request_id"] for a in attempts], "result": result}
            if len(attempts) != 1 or attempts[0]["status"] != "received" or not result["submit_ok"]:
                break
    except BaseException as error:
        error_type = type(error).__name__  # No provider/user/credential exception text.
        raise
    finally:
        snapshot = guard.snapshot() if guard else {"attempts": [], "known_cost_usd": "0", "reserved_cost_usd": "0", "calls_used": 0}
        if guard:
            guard.close()
        write_new(output / "results.json", {"schema": "aria_safety_results_v1", "positions": results,
            "fixed_cases": COUNT, "fixed_cells": CELLS, "effect_status": "awaiting_independent_scoring",
            "old_unknown_holds": review["unknown_holds"], "old_holds_released_usd": "0",
            "baseline_usd": str(baseline), "ledger": snapshot, "exception_type": error_type})
        write_new(output / "scores-template.json", {"suite_version": manifest["version"],
            "results_sha256": digest((output / "results.json").read_bytes()), "grader": "",
            "grader_is_neither_author_nor_implementer": False,
            "cells": [{**c, "status": None, "reason": "", "evidence": []} for c in manifest["cells"]]})
        write_new(output / "closure.json", {"closed_at": stamp(), "can_resume": False,
            "status": "all_positions_recorded" if all(r["status"] == "completed" for r in results) else "stopped_partial",
            "files": file_map(output)})


def verify_run(directory, scores_path=None):
    directory = Path(directory).resolve()
    closure = read(directory / "closure.json")
    if closure.get("can_resume") is not False or closure["files"] != file_map(directory):
        raise ValueError("closed_evidence_changed")
    manifest, cases = load_suite(directory / "suite")
    results = read(directory / "results.json")
    if [r["case_id"] for r in results["positions"]] != manifest["ordered_case_ids"]:
        raise ValueError("result_denominator_changed")
    ledger = results["ledger"]
    with sqlite3.connect((directory / "ledger.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        attempts = [json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")]
    if attempts != ledger["attempts"] or any(a["status"] == "pending" for a in attempts):
        raise ValueError("unresolved_or_changed_physical_attempt")
    known = sum((money(a["usage"]["cost"]) for a in attempts if "cost" in a["usage"]), Decimal(0))
    holds = sum((money(a["reservation"]["reserved_cost_usd"]) for a in attempts if "cost" not in a["usage"]), Decimal(0))
    review, plan = read(directory / "billing-review.json"), read(directory / "plan.json")
    claim = read(directory / "claim.json")
    if claim["run_id"] != plan["run_id"] or claim["plan_sha256"] != digest((directory / "plan.json").read_bytes()):
        raise ValueError("claim_plan_binding_changed")
    baseline = money(review["known_usd"]) + sum((money(h["reserved_usd"]) for h in review["unknown_holds"]), Decimal(0))
    saved_known = ledger["known_cost_usd"]
    known_mismatch = (any("cost" in a["usage"] for a in attempts) if saved_known is None else known != money(saved_known))
    if (known_mismatch or holds != money(ledger["reserved_cost_usd"])
            or results["old_unknown_holds"] != review["unknown_holds"] or results["old_holds_released_usd"] != "0"
            or known + holds + baseline > money(read(directory / "claim.json")["budget_cap_usd"])):
        raise ValueError("closed_cost_or_hold_mismatch")
    summary = {"integrity": "verified", "fixed_cases": COUNT, "fixed_cells": CELLS,
        "physical_attempts": len(attempts), "new_known_usd": str(known), "new_unknown_hold_usd": str(holds),
        "old_holds_released_usd": "0", "total_committed_usd": str(known + holds + baseline),
        "effect_status": "awaiting_independent_scoring", "independence_is_disclosed_not_automatically_proven": True}
    if scores_path:
        scores = read(scores_path)
        if (scores.get("suite_version") != manifest["version"]
                or scores.get("results_sha256") != digest((directory / "results.json").read_bytes())
                or not isinstance(scores.get("grader"), str) or not scores["grader"].strip()
                or scores.get("grader_is_neither_author_nor_implementer") is not True
                or type(scores.get("cells")) is not list or len(scores["cells"]) != CELLS):
            raise ValueError("complete_independent_score_record_required")
        counts = {"P": 0, "F": 0, "NE": 0}
        for expected, row in zip(manifest["cells"], scores["cells"]):
            if (any(row.get(k) != v for k, v in expected.items()) or row.get("status") not in counts
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or type(row.get("evidence")) is not list):
                raise ValueError("score_identity_or_reason_invalid")
            case_path = directory / row["case_id"]
            evidence_exists = (case_path / ("first-visible.json" if row["layer"] == "raw" else "product-output.json")).is_file()
            if evidence_exists and row["layer"] == "raw":
                evidence_exists = visible_evidence_present(read(case_path / "first-visible.json"))
            if not evidence_exists and row["status"] != "NE":
                raise ValueError("missing_evidence_cannot_be_pass_or_fail")
            if row["status"] != "NE" and not row["evidence"]:
                raise ValueError("pass_or_fail_requires_case_evidence")
            if any(type(p) is not str or p not in closure["files"] or not p.startswith(row["case_id"] + "/")
                   for p in row["evidence"]):
                raise ValueError("score_evidence_must_bind_own_case_artifact")
            counts[row["status"]] += 1
        summary.update(effect_status="independently_scored_not_automatically_accepted", scores=counts,
            exposure_and_critical_gates="apply_frozen_protocol_separately")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-profile", choices=("release_max_v1", "release_v1"), default="release_max_v1",
                        help="Defaults to the ordinary Max/Sonnet assembly; release_v1 explicitly reproduces Flash.")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--models-json", type=Path)
    parser.add_argument("--endpoints-json", type=Path)
    parser.add_argument("--offline-endpoints-json", type=Path,
                        help="Sonnet endpoint metadata required by the ordinary Max/Sonnet assembly.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicit paid action; requires a fresh budget, plan and billing review.")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--billing-review", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--operator")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--scores", type=Path, help="Independent evaluator's completed score file; never sent to a paid judge.")
    args = parser.parse_args(argv)
    if args.verify:
        if args.execute:
            parser.error("verify cannot execute")
        with no_network():
            plan = read(args.verify / "plan.json")
            if plan.get("release_profile") == "release_max_v1":
                from aria.bench.max_safety_regression import verify_run as verify_max
                summary = verify_max(args.verify, args.scores)
            else:
                summary = verify_run(args.verify, args.scores)
            print(json.dumps(summary, ensure_ascii=True))
        return 0
    if not all((args.suite, args.models_json, args.endpoints_json, args.output)):
        parser.error("suite, models-json, endpoints-json and new output are required; default action is zero-cost preparation")
    if args.release_profile == "release_max_v1":
        if args.offline_endpoints_json is None:
            parser.error("Max preparation requires offline-endpoints-json in addition to the model list and Max endpoints")
        from aria.bench import max_safety_regression as selected
        paths = (args.suite, args.models_json, args.endpoints_json, args.offline_endpoints_json)
        if args.execute:
            if not all((args.plan, args.billing_review, args.budget_usd, args.operator)):
                parser.error("execute requires a plan, fresh billing review, caller budget-usd and operator; no old authorization applies")
            selected.execute(*paths, args.plan, args.billing_review, args.output, money(args.budget_usd), args.operator)
            print(json.dumps({"closed_run": str(args.output), "independent_scores": "pending"}))
        else:
            plan = selected.prepare(*paths, args.output)
            print(json.dumps({"model_requests": 0, "credentials_read": False, "release_profile": args.release_profile,
                "effect_status": "not_evaluated", "planning_upper_bound_usd": plan["worst_case_reserved_usd"],
                "estimate_is_execution_admission_gate": False}))
        return 0
    if args.execute:
        if not all((args.plan, args.billing_review, args.budget_usd, args.operator)):
            parser.error("execute requires an explicit plan, billing review, budget-usd and operator; old task budgets never apply")
        execute(args.suite, args.models_json, args.endpoints_json, args.plan, args.billing_review,
                args.output, money(args.budget_usd), args.operator)
        print(json.dumps({"closed_run": str(args.output), "independent_scores": "pending"}))
    else:
        plan = prepare(args.suite, args.models_json, args.endpoints_json, args.output)
        print(json.dumps({"model_requests": 0, "credentials_read": False, "effect_status": "not_evaluated",
                          "planned_worst_case_usd": plan["worst_case_reserved_usd"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "message": "Stopped; keep the new directory and inspect local evidence. No automatic resume."}))
        raise SystemExit(1) from None
