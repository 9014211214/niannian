"""Local-only E1 rehearsal: fixed synthetic fixtures, production prompt/parser.

This is a delivery-text use experiment, not a complete Engine/learning test.
It never discovers credentials or offers a paid CLI. The injected opener must
be an in-memory transport; all accounting is explicitly synthetic. A later
registered real run requires its own reviewed transport and price reservation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import re
import secrets
import time
from typing import Callable

from aria.engine.context import EngineContext, Turn
from aria.engine.model.persona import PersonaProfile, PersonHypothesis
from aria.engine.perception.evidence import collect_evidence
from aria.engine.policy.llm import OpenAILLMClient, OpenRouterLLMClient
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2,
    apply_single_pass_decision, online_decision_tool_spec,
    single_pass_messages,
)
from aria.engine.policy.online_return_channel import bind_online_return_channel
from aria.engine.policy.prompt import (
    _untrusted_block, build_messages, render_delivery_block,
)
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from aria.scenes.registry import get_scene
from .evaluation_journal import EvaluationJournal
from .prompt_footprint import measure_messages, measure_output_fields
from .provenance import fingerprint
from .refocus_cases import RefocusCase, RefocusSuite, load_refocus_cases
from .request_control import BenchmarkRequestControl
from .provenance import strict_json

VERSION = "refocus_e1_local_runner_v1"
ARMS = ("B0", "B1", "B2", "B3a", "B3b")
PAIR_ARMS = ("B0", "B2", "B3b")
DENOMINATOR = 204
_VISIBLE_FIELDS = {
    "response", "state_signals", "current_affect", "deliberation",
    "selected_intervention_id", "communication_plan", "uncertainty",
}
_HIDDEN = re.compile(r"reasoning|analysis|thinking|chain.of.thought", re.I)


@dataclass(frozen=True)
class E1LocalConfig:
    model: str = "local/e1-fixture"
    temperature: float = 0.2
    max_completion_tokens: int = 1024
    seed: int = 0
    max_cost_usd: float = 0.5
    per_request_reserve_usd: float = 0.005

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.startswith("local/"):
            raise ValueError("e1_local_fixture_model_required")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("e1_invalid_seed")
        if type(self.max_completion_tokens) is not int or not 64 <= self.max_completion_tokens <= 4096:
            raise ValueError("e1_invalid_output_limit")
        for value in (self.temperature, self.max_cost_usd, self.per_request_reserve_usd):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("e1_invalid_numeric_config")
        if not 0 <= self.temperature <= 2:
            raise ValueError("e1_invalid_temperature")
        if not 0 < self.per_request_reserve_usd <= self.max_cost_usd <= 0.5:
            raise ValueError("e1_invalid_cost_reservation")


def inventory(suite: RefocusSuite, *, seed: int = 0) -> list[dict]:
    """Repeat-major order; adjacent pair members have the same arm rotation."""
    rows = []
    groups = [(case,) for case in suite.traps] + [pair.cases for pair in suite.pairs]
    for repeat in range(3):
        for group_index, group in enumerate(groups):
            arms = ARMS if len(group) == 1 else PAIR_ARMS
            offset = (seed + repeat + group_index) % len(arms)
            for arm in (*arms[offset:], *arms[:offset]):
                for case in group:
                    rows.append({"case_id": case.id, "arm": arm, "repeat": repeat + 1})
    assert len(rows) == DENOMINATOR
    return rows


def _fixture_persona(case: RefocusCase, arm: str) -> PersonaProfile:
    # Compatibility projection ONLY for the existing v2 citation validator.
    # No lifecycle update, store write, fabricated date, or learned-person claim.
    profile = PersonaProfile(user_id="e1_fictional_person", profile_origin="synthetic_demo")
    if arm != "B0":
        profile.person_hypotheses = [PersonHypothesis(
            id=item.id, framework="response", status="working", confidence=1,
            claim="作者给定的本人确认说明：" + item.do,
            scope=item.scope, first_observed_at="",
        ) for item in case.guidance]
    if arm in {"B3a", "B3b"}:
        profile.disc = profile.disc.model_validate(case.persona_fixture.disc.model_dump())
        profile.big_five = profile.big_five.model_validate(case.persona_fixture.big_five.model_dump())
        profile.attachment_style = case.persona_fixture.attachment_style
    return profile


def prepare_trial(case: RefocusCase, arm: str, config: E1LocalConfig) -> dict:
    """Production builders, with ONE experimental delivery-block substitution.

    All candidate scores, contracts and base text are computed without supplied
    personality numbers. B3 numbers affect only delivery text and its citeable
    axes. Common facts/boundaries are identical for every arm, including B0.
    """
    if arm not in ARMS:
        raise ValueError("e1_unknown_arm")
    stimulus = case.model_input()
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id="e1_fictional_person", scene_id=scene.id,
                        session_id="e1_local_fixture")
    for message in stimulus["short_context"]:
        if message["role"] == "user":
            ctx.turns.append(Turn(user=message["content"], agent=""))
        elif ctx.turns and not ctx.turns[-1].agent:
            ctx.turns[-1].agent = message["content"]
    ctx.emotion.label = "unknown"
    scene.state_reasoner().update(
        ctx, None, stimulus["user_input"],
        collect_evidence(stimulus["user_input"], ctx.emotion, turn_index=len(ctx.turns)),
    )
    ctx.working_memory["_benchmark_capture_reply_trace"] = True
    messages = build_messages(scene, None, ctx, stimulus["user_input"],
                              available_tools=[], history_turns=0, delivery_block="mode_v1")
    # history_turns=0 has legacy slice semantics; explicitly retain only the
    # trusted system and current input, then install the exact stimulus window.
    messages = [messages[0], *deepcopy(stimulus["short_context"]), messages[-1]]
    profile = _fixture_persona(case, arm)
    block = "profile_v1" if arm == "B3b" else "mode_v1"
    delivery = render_delivery_block(profile, block=block) if arm in {"B3a", "B3b"} else ""
    neutral = render_delivery_block(None, block="mode_v1")
    if messages[0]["content"].count(neutral) != 1:
        raise ValueError("e1_delivery_block_boundary_changed")
    messages[0]["content"] = messages[0]["content"].replace(neutral, delivery, 1)
    common = {key: stimulus[key] for key in ("facts", "shared_boundaries", "care_context") if key in stimulus}
    messages[0]["content"] += "\n\n" + _untrusted_block(
        "共同事实与边界（作者合成夹具）", "e1_common_synthetic_stimulus", common,
    )
    if arm != "B0":
        # Flatten quote provenance one level so the production untrusted-data
        # depth bound cannot replace the actual elder quotes with truncation.
        guidance = [{
            "id": item.id, "do": item.do, "scope": item.scope,
            "not_implies": list(item.not_implies),
            "quotes": [f"session={q.session},turn={q.turn}: {q.text}" for q in item.quotes],
            "confirmation": (f"session={item.confirmation.session},turn={item.confirmation.turn}: "
                             + item.confirmation.text),
        } for item in case.guidance]
        messages[0]["content"] += "\n\n" + _untrusted_block(
            "本人确认的相处说明（作者合成夹具，非本次学习结果）",
            "e1_author_supplied_stated_fixture", guidance,
        )
    protocol = ONLINE_DECISION_VERSION if arm in {"B0", "B1"} else ONLINE_DECISION_VERSION_V2
    messages = single_pass_messages(messages, ctx, version=protocol,
                                    persona=profile, delivery_block=block)
    if protocol == ONLINE_DECISION_VERSION_V2:
        messages[0]["content"] = messages[0]["content"].replace(
            "人格已由会后离线形成，这里只引用，",
            "这里的人物信息全部由作者预置，不是会后学习结果；仅供本次引用，", 1,
        )
    schema = online_decision_tool_spec(protocol)
    payload = {
        "model": config.model, "messages": bind_online_return_channel(messages, channel="native"),
        "temperature": config.temperature, "max_completion_tokens": config.max_completion_tokens,
        "tools": [schema], "tool_choice": {"type": "function", "function": {"name": "aria_turn_decision"}},
        "parallel_tool_calls": False,
    }
    # Guards must see the same common facts the model sees. This is only a
    # compatibility carrier for author-supplied facts, not a retrieval/learning
    # claim. Install after prompt building to avoid a second facts block.
    # Plain text is required: JSON string quoting would be removed by the
    # production grounding guard's quoted-speech filter.
    ctx.working_memory["recalled_episodes"] = [
        "作者合成事实（非学习结果）：\n" + "\n".join(stimulus["facts"])
    ]
    return {"messages": messages, "payload": payload, "ctx": ctx, "persona": profile,
            "request_fingerprint": fingerprint(payload), "protocol": protocol,
            "input_measurement": measure_messages(payload["messages"], tool_spec=schema)}


class _LocalClient(OpenAILLMClient):
    """No credential lookup; only the attached injected controller is usable."""

    def __init__(self, config: E1LocalConfig, control: BenchmarkRequestControl):
        super().__init__(api_key="e1-local-placeholder", base_url="https://e1.invalid/v1",
                         model=config.model, temperature=config.temperature,
                         role="online_fast", timeout=1, max_retries=0)
        self.config = config
        self.expected_payload = None
        control.attach(self)

    def complete(self, messages):
        # Reuse the production native-response extraction and transport, while
        # keeping the fixed experiment payload explicit and environment-free.
        self._start_decision()
        payload = deepcopy(self.expected_payload)
        if payload is None or payload["messages"] != bind_online_return_channel(messages, channel="native"):
            raise ValueError("e1_prepared_request_changed")
        import urllib.request
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(payload, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
        body = self._request_json(request)
        self.last_model = body.get("model")
        self.last_usage = body.get("usage", {})
        return OpenRouterLLMClient._extract_online_decision(self, body)


def _visible_payload(raw: str) -> dict:
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) - _VISIBLE_FIELDS:
        raise ValueError("e1_invalid_visible_payload")
    if not isinstance(value.get("response"), str) or not value["response"].strip():
        raise ValueError("e1_invalid_visible_response")

    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if _HIDDEN.search(key):
                    raise ValueError("e1_private_reasoning_field_rejected")
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, str) and re.search(r"<(?:think|analysis)\b", item, re.I):
            raise ValueError("e1_private_reasoning_tag_rejected")
    check(value)
    return value


def export_blind_review(rows: list[dict], cases: dict[str, RefocusCase], *, seed: int,
                        stage: str = "raw") -> tuple[list, list]:
    """Use a PRIVATE seed, independent of the public execution-order seed."""
    if stage not in {"raw", "delivered"}:
        raise ValueError("e1_unknown_review_stage")
    order = list(rows)
    rng = random.Random(seed if stage == "raw" else seed ^ 0xA18F9E77)
    rng.shuffle(order)
    blind, key = [], []
    model_names = {row["actual_model"] for row in rows
                   if isinstance(row.get("actual_model"), str) and row["actual_model"]}
    for row in order:
        response = ((row.get("raw_decision") or {}).get("response") if stage == "raw"
                    else row.get("delivered_response"))
        valid_response = isinstance(response, str) and bool(response.strip())
        case = cases[row["case_id"]]
        identifier = "review_" + f"{rng.getrandbits(64):016x}"
        # Preserve the original only in private results, never rewrite leaked
        # identities into apparently blind prose or silently drop the sample.
        unblindable = valid_response and (bool(re.search(
            r"(?<![a-z0-9])B(?:[012]|3[ab])(?![a-z0-9])|"
            r"(?:online_turn_decision|mode|profile)_v\d+|"
            r"prompt[ _-]?version|(?<!\d)(?:1\.12\.0|1\.13\.0)(?!\d)",
            response, re.I,
        )) or any(model.lower() in response.lower() for model in model_names))
        blind.append({
            "review_id": identifier,
            "context": case.model_input(),
            "confirmed_guidance": case.guidance_input(),
            "response": response if valid_response and not unblindable else None,
            "blindability_status": ("not_evaluable" if not valid_response else
                                    "unblindable_marker" if unblindable else "blind"),
            "criteria": case.review.model_dump(mode="json"),
            "labels": {name: None if item.applicable else 0 for name, item in case.review},
        })
        key.append({"review_id": identifier, "case_id": row["case_id"],
                    "arm": row["arm"], "repeat": row["repeat"], "order": row["order"],
                    "stage": stage, "private_blinding_seed": seed})
    return blind, key


def run_local_e1(output_dir: Path | str, *, opener: Callable,
                 config: E1LocalConfig | None = None, suite: RefocusSuite | None = None) -> dict:
    """Run the complete inventory against an explicitly injected local fake.

    Existing destinations are rejected; no automatic retries or resume. A torn
    journal remains ambiguous and cannot become a fresh paid attempt later.
    All 204 coordinates are reported, including not-started rows after a stop.
    """
    if not callable(opener):
        raise ValueError("e1_in_memory_opener_required")
    config = config or E1LocalConfig()
    suite = suite or load_refocus_cases()
    cases = {case.id: case for case in suite.main_cases()}
    schedule = inventory(suite, seed=config.seed)
    prepared = {(case.id, arm): prepare_trial(case, arm, config)
                for case in suite.main_cases()
                for arm in (ARMS if case in suite.traps else PAIR_ARMS)}
    items = [{"coordinate": row, "input_fingerprint": prepared[row["case_id"], row["arm"]]["request_fingerprint"]}
             for row in schedule]
    aria_root = Path(__file__).parents[1]
    source_names = (
        "bench/refocus_e1_runner.py", "bench/refocus_cases.py", "bench/prompt_footprint.py",
        "bench/request_control.py", "bench/evaluation_journal.py", "engine/policy/response_observability.py", "bench/provenance.py",
        "engine/policy/prompt.py", "engine/policy/online_decision.py", "engine/policy/llm.py",
        "engine/policy/online_return_channel.py", "engine/policy/reply_contract.py",
        "engine/policy/strategy.py", "engine/perception/evidence.py",
        "scenes/companion_elder/delivery.py", "scenes/companion_elder/reasoning.py",
    )
    source = {name: hashlib.sha256((aria_root / name).read_bytes()).hexdigest() for name in source_names}
    binding = {"version": VERSION, "config": asdict(config), "items": items,
               "suite_fingerprint": fingerprint(suite.model_dump(mode="json")), "sources": source}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    journal = EvaluationJournal.create(target / "journal", profile_fingerprint=fingerprint(binding),
        source_fingerprint=fingerprint(source), stage="refocus_e1_local", items=items,
        max_network_requests=DENOMINATOR, max_observed_cost_usd=config.max_cost_usd)
    control = BenchmarkRequestControl(min_interval_seconds=0, max_network_requests=DENOMINATOR,
        max_observed_cost_usd=config.max_cost_usd, stop_on_rate_limit=True, opener=opener)
    client = _LocalClient(config, control)
    rows, stopped = [], None
    reserve, maximum = Decimal(str(config.per_request_reserve_usd)), Decimal(str(config.max_cost_usd))
    for order, coordinate in enumerate(schedule, 1):
        item = prepared[coordinate["case_id"], coordinate["arm"]]
        row = {**coordinate, "order": order, "status": "not_started",
               "request_fingerprint": item["request_fingerprint"], "raw_decision": None}
        accounting = control.accounting()
        known = sum((Decimal(str(a["provider_usage"]["cost"])) for a in accounting["attempts"]
                     if "cost" in a["provider_usage"]), Decimal(0))
        if not stopped and known + reserve > maximum:
            stopped = "insufficient_per_request_reserve"
        if stopped:
            row["reason"] = stopped
            rows.append(row)
            continue
        token = journal.start(coordinate, item["request_fingerprint"])
        ctx, persona = deepcopy(item["ctx"]), deepcopy(item["persona"])
        client.expected_payload = item["payload"]
        client.last_model, client.last_usage = None, {}
        before_attempts = len(accounting["attempts"])
        started = time.perf_counter()
        try:
            raw = client.complete(item["messages"])
            payload = _visible_payload(raw)  # Capture declared fields BEFORE any local rewrite.
            row["raw_decision"] = payload
            row["output_measurement"] = measure_output_fields([payload])
            delivered = apply_single_pass_decision(raw, ctx, cases[coordinate["case_id"]].stimulus.user_input, persona)
            delivered = guard_external_delivery_response(ctx, delivered,
                cases[coordinate["case_id"]].stimulus.user_input, persona)
            trace = ctx.working_memory.get("online_turn_decision") or {}
            row.update(status="generated" if trace.get("status") == "parsed" else "invalid_protocol",
                       delivered_response=delivered,
                       deliberation_status=trace.get("deliberation_status", "not_applicable"),
                       selected_in_deliberated_options=trace.get("selected_in_deliberated_options"),
                       selected_intervention_id=trace.get("selected_intervention_id"),
                       rule_response_changed=delivered != payload.get("response"))
        except Exception:
            row.update(status="generation_failed", reason="local_transport_or_protocol_failure")
            stopped = "generation_failed_no_retry"
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["actual_model"] = client.last_model
        row["input_measurement"] = item["input_measurement"]
        accounting = control.accounting()
        new_attempts = accounting["attempts"][before_attempts:]
        attempt = new_attempts[0] if len(new_attempts) == 1 else {}
        row["usage"] = deepcopy(attempt.get("provider_usage", {}))
        if len(new_attempts) != 1:
            stopped = "single_attempt_contract_failed"
        if row["status"] == "generated" and row["actual_model"] != config.model:
            row["status"] = "invalid_model_identity"
            stopped = "actual_model_missing_or_changed"
        row["token_usage_status"] = ("observed" if all(
            name in row["usage"] for name in ("prompt_tokens", "completion_tokens")
        ) else "missing")
        if row["status"] == "generated" and row["token_usage_status"] == "missing":
            row["status"] = "invalid_usage_receipt"
            stopped = "token_usage_missing"
        cost = row["usage"].get("cost")
        row["cost_receipt_status"] = (
            "unknown" if cost is None else "reserve_exceeded"
            if Decimal(str(cost)) > reserve else "within_declared_reserve"
        )
        if cost is None:
            stopped = "unknown_cost_no_retry"
        elif Decimal(str(cost)) > reserve:
            stopped = "per_request_reserve_exceeded"
        if row["status"] == "invalid_protocol":
            stopped = "invalid_protocol_no_retry"
        journal.finish(token, result={"row": row}, accounting=accounting)
        rows.append(row)
    blinding_seed = secrets.randbits(128)
    blind, private_key = export_blind_review(rows, cases, seed=blinding_seed)
    delivered_blind, delivered_key = export_blind_review(rows, cases, seed=blinding_seed, stage="delivered")
    result = {"version": VERSION, "mode": "local_simulated_transport",
              "model_effect_measured": False, "denominator": DENOMINATOR,
              "status": "local_rehearsal_complete" if stopped is None and all(
                  r["status"] == "generated" for r in rows
              ) else "incomplete",
              "stopped_reason": stopped, "binding": binding, "binding_fingerprint": fingerprint(binding),
              "accounting": control.accounting(), "rows": rows,
              "review_targets": {"blind-review.json": "raw_model_response_before_local_guards",
                                 "blind-review-delivered.json": "delivered_response_after_local_guards"},
              "rule_response_changed_count": sum(row.get("rule_response_changed", False) for row in rows),
              "limitations": [
                  "Author-supplied confirmed guidance and numbers, not learned preferences.",
                  "Delivery-text use experiment, not full Engine or cross-session learning.",
                  "Shared production candidate preselection/change constraints may limit person-specific route separation.",
                  "No blind scores or quality conclusions are produced by this rehearsal.",
              ],
              "budget_scope": "synthetic_receipts_and_per_request_reserve_not_provider_invoice_guarantee"}
    for name, content in (("local-results.json", result), ("blind-review.json", blind),
                          ("blind-review-delivered.json", delivered_blind),
                          ("private-review-key.json", private_key + delivered_key)):
        with (target / name).open("x", encoding="utf-8") as output:
            json.dump(content, output, ensure_ascii=False, indent=2, allow_nan=False)
    return result
