"""E1b product-path runner, with explicit clients and isolated fictional data.

There is no credential discovery or paid CLI. The local factory below is a
software fixture. A reviewed caller may inject guarded real clients only after
registration/authorization; it owns one shared budget outside branch stores.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable

from aria.engine.model.persona import PersonaProfile
from aria.engine.memory.atomic_file import replace_with_retry
from aria.engine.memory.consolidator import _strip_fences
from aria.engine.memory.diagnosis import (
    _REFLECTION_ERROR_DETAILS, _reject_nonfinite_json, _unwrap_review_response,
)
from aria.runtime import MinimalModelClients, create_minimal_engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from .e1b_cases import E1bCase, E1bSuite, synthetic_emotion_plan


VERSION = "e1b_product_runner_v2_unfrozen"


class E1bRunStopped(RuntimeError):
    """A failed or unknown operation stops this run without an automatic retry."""


def _first_answer_attempts(guard, state):
    """Only validated no-answer transport recovery can precede a first answer."""
    from ..engine.policy.e1br_calls import E1BRCallGuard, logical_attempts
    if isinstance(guard, E1BRCallGuard):
        guard._validate_history(state["attempts"])
        return logical_attempts(state)
    policy = getattr(guard, "recovery_policy", None)
    if policy is None:
        return state["attempts"]
    from ..engine.policy.bounded_recovery import validate_recovery_records
    return validate_recovery_records(state["attempts"], policy,
        relations=state["recovery"]["relations"], require_complete=True)["logical_attempts"]


def _exclusive_json(path, value):
    """First-attempt originals are never replaced, including partial failures."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_visible_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate visible JSON field")
            result[key] = value
        return result
    decoder = json.JSONDecoder(object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite visible JSON")))
    value = _strip_fences(raw)
    try:
        return decoder.decode(value)
    except json.JSONDecodeError:
        starts = [index for index in (value.find("{"), value.find("[")) if index >= 0]
        if not starts:
            raise
        return decoder.raw_decode(value[min(starts):])[0]


def _proposal_projection(value):
    """Keep every row/type, but only requested public proposal field values."""
    removed = 0
    proposal_fields = {"do", "scope", "quotes", "confirmation", "supersedes", "not_implies"}
    quote_fields = {"turn_index", "text"}
    def project(item, allowed):
        nonlocal removed
        if isinstance(item, dict):
            result = {}
            for name, child in item.items():
                if name not in allowed or re.search(r"reasoning|analysis|thinking|chain.of.thought", name, re.I):
                    removed += 1
                    continue
                result[name] = project(child, quote_fields if name in {"quotes", "confirmation"} else set())
            return result
        if isinstance(item, list):
            return [project(child, allowed) for child in item]
        if isinstance(item, str) and re.search(r"</?(?:think|thinking|analysis|reasoning)(?:\s|>)", item, re.I):
            removed += 1
            return "[unrequested nonpublic content omitted]"
        return item
    projected = project(value, proposal_fields)
    return projected, removed


class _ProposalCapture:
    """One experimental memory response, before product filtering or reflection."""

    def __init__(self, app, directory, *, case_id, session, session_id, source_sha256):
        self.app, self.directory = app, Path(directory)
        if (not isinstance(app, E1bExperimentApp)
                or Path(app.experiment.root).resolve() not in self.directory.resolve().parents):
            raise ValueError("raw proposal capture requires the isolated experiment directory")
        self.case_id, self.session, self.session_id = case_id, session, session_id
        self.source_sha256 = source_sha256
        self.started = False
        self.first = None
        self.first_path = self.directory / f"memory-s{session}.first.json"
        self.outcome_path = self.directory / f"memory-s{session}.product.json"
        self.attempt_path = self.directory / f"memory-s{session}.attempt.json"

    def publish(self):
        sessions = []
        for number in (1, 2):
            first_path = self.directory / f"memory-s{number}.first.json"
            outcome_path = self.directory / f"memory-s{number}.product.json"
            if not first_path.exists():
                continue
            first = json.loads(first_path.read_text(encoding="utf-8"))
            outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else None
            sessions.append({**first, "complete": bool(first["parse_status"] == "present_array" and outcome
                            and outcome["product_status"] == "ok"), "product_outcome": outcome,
                             "first_attempt_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
                             "product_outcome_sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest() if outcome else None})
        _write_json(self.directory / "raw-proposals.json", {
            "schema": "e1b_raw_proposals_v2", "case_id": self.case_id,
            "source_sha256": self.source_sha256, "sessions": sessions})

    def complete(self, client, messages):
        if self.started:
            raise E1bRunStopped("memory first attempt already consumed")
        self.started = True
        _exclusive_json(self.attempt_path, {"session": self.session, "session_id": self.session_id,
                         "status": "started", "source_sha256": self.source_sha256})
        first = {"session": self.session, "request_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                 "request_hash_kind": "software_messages" if self.app.call_guard is None else "input_messages_not_wire",
                 "visible_response_sha256": None,
                 "field_present": None, "raw_proposals": None, "parse_status": "call_failed_or_unknown",
                 "public_projection_dropped_fields": 0}
        try:
            raw = client.complete(messages)
        except Exception:
            _exclusive_json(self.first_path, first)
            self.first = first
            self.publish()
            raise
        identity_missing = False
        if self.app.call_guard is not None:
            attempts = _first_answer_attempts(self.app.call_guard, self.app.call_guard.snapshot())
            rows = [row for row in attempts if row.get("role") == "memory" and row.get("session_id") == self.session_id]
            if len(rows) != 1 or not re.fullmatch(r"[a-f0-9]{64}", rows[0].get("request_sha256", "")):
                identity_missing = True
            else:
                first["request_sha256"] = rows[0]["request_sha256"]
                first["request_hash_kind"] = "actual_wire"
        if isinstance(raw, str):
            first["visible_response_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                parsed = _strict_visible_json(raw)
                first["field_present"] = isinstance(parsed, dict) and "stated_preferences" in parsed
                value = parsed.get("stated_preferences") if isinstance(parsed, dict) else None
                first["raw_proposals"], first["public_projection_dropped_fields"] = _proposal_projection(value)
                first["parse_status"] = ("missing_field" if not first["field_present"] else "null_field"
                                         if value is None else "present_array" if isinstance(value, list) else "not_array")
            except (ValueError, TypeError, RecursionError):
                first["parse_status"] = "parse_error"
        else:
            first["parse_status"] = "non_text_response"
        if identity_missing:
            first["parse_status"] = "request_identity_unavailable"
        _exclusive_json(self.first_path, first)
        self.first = first
        self.publish()
        if first["parse_status"] != "present_array":
            raise E1bRunStopped("missing or malformed raw preference output is not an empty proposal set")
        return raw

    def finish(self, extractor, ctx, result):
        if self.first is None:
            raise E1bRunStopped("memory first response was not durably saved")
        diagnostic = ctx.working_memory.get("stated_preference_extraction")
        _exclusive_json(self.outcome_path, {
            "session": self.session, "product_status": extractor.last_status,
            "stage": "actual_extractor_result_before_merge", "diagnostics_available": isinstance(diagnostic, dict),
            "accepted_preferences": [row.model_dump(mode="json") for row in result.stated_preferences],
            "accepted_count": diagnostic.get("accepted") if isinstance(diagnostic, dict) else None,
            "rejected_codes": diagnostic.get("rejected") if isinstance(diagnostic, dict) else None})
        self.publish()


class _CapturedExtractor:
    def __init__(self, extractor, capture):
        self.extractor, self.capture = extractor, capture
        client = extractor.llm
        class RecordedClient:
            def complete(self, messages):
                return capture.complete(client, messages)
        extractor.llm = RecordedClient()

    def __getattr__(self, name):
        return getattr(self.extractor, name)

    def extract(self, ctx, persona):
        if ctx.session_id != self.capture.session_id:
            raise E1bRunStopped("memory capture session mismatch")
        result = self.extractor.extract(ctx, persona)
        self.capture.finish(self.extractor, ctx, result)
        return result


def _reflection_public_projection(data):
    """Bounded requested public fields only; never store unknown field names."""
    quote = {"turn_index": "number", "evidence_quote": 240}
    review = {"id": 40, "hypothesis_id": 80, "claim_quote": 240, "scope_quote": 240,
              "applicability": 32, "relation": 32, "evidence_basis": 32, "event_kind": 32,
              "prior_event_id": 80, "evidence": [quote], "event_evidence": [quote],
              "reason": 240, "event_summary": 240}
    change = {"review_id": 40, "id": 80, "framework": 32, "dimension": 64, "claim": 240,
              "scope": 240, "scope_tags": [40], "relation": 32,
              "lineage": {"id": 80, "claim_quote": 240, "scope_quote": 240,
                          "mechanism_relation": 32, "scope_relation": 32},
              "atomic_claim": {"aspect": 100, "conditions": 100, "tendency": 100,
                               "outcome": 100, "evidence_basis": 32},
              "target_value": "number", "confidence": "number", "evidence": [quote]}
    removed = 0

    def project(value, spec):
        nonlocal removed
        if value is None:
            return None
        if type(spec) is dict and type(value) is dict:
            result = {}
            for name, child in value.items():
                if name not in spec:
                    removed += 1
                else:
                    result[name] = project(child, spec[name])
            return result
        if type(spec) is list and type(value) is list:
            # More than the wire protocol permits stays a failed shape, not an
            # empty set. Bound the diagnostic copy separately and count drops.
            removed += max(0, len(value) - 64)
            return [project(item, spec[0]) for item in value[:64]]
        if type(spec) is int and type(value) is str and len(value) <= spec:
            if not re.search(r"</?(?:think|thinking|analysis|reasoning)(?:\s|>)", value, re.I):
                return value
        elif spec == "number" and (type(value) is int or type(value) is float and math.isfinite(value)):
            return value
        removed += 1
        return None

    return project(data, {"protocol_version": 16, "reviews": [review], "changes": [change]}), removed


def _reflection_client_diagnostic(client):
    diagnostic = getattr(client, "last_diagnostic", {})
    if type(diagnostic) is not dict:
        return {}
    allowed = {
        "error_type": {"invalid_json_response", "invalid_response_shape", "empty_text_response",
            "reasoning_disabled_violation", "timeout", "network_error", "http_error",
            "minimal_call_stopped", "minimal_request_failed", "invalid_reflection_json_schema"},
        "output_kind": {"text", "empty", "invalid_choice", "missing_choice", "missing_message",
            "content_json", "invalid_content_json", "reasoning_rejected", "reflection_json_schema",
            "invalid_reflection_json_schema"},
        "finish_reason": {"stop", "length", "tool_calls", "content_filter"},
    }
    result = {key: value for key, values in allowed.items()
              if type(value := diagnostic.get(key)) is str and value in values}
    if type(diagnostic.get("http_status")) is int and 100 <= diagnostic["http_status"] <= 599:
        result["http_status"] = diagnostic["http_status"]
    from ..engine.policy.reflection_json import sanitize_reflection_shape_diagnostic
    shape = sanitize_reflection_shape_diagnostic(diagnostic.get("reflection_shape"))
    if shape:
        result["reflection_shape"] = shape
    return result


def _reflection_product_diagnostic(updater):
    result = {"product_status": getattr(updater, "last_status", None)}
    if (type(result["product_status"]) is not str
            or result["product_status"] not in {"ok", "failed", "skipped", "not_run"}):
        result["product_status"] = "unknown"
    for key in ("last_error_type", "last_error_detail"):
        value = getattr(updater, key, None)
        result[key] = value if type(value) is str and value in _REFLECTION_ERROR_DETAILS else None
    transport = getattr(updater, "last_response_transport", None)
    if type(transport) is dict:
        safe = {}
        for key, allowed in {"prompt_contract_version": {"2.0", "2.1", "2.2", "2.3"},
                "format": {"not_evaluated", "bare_json", "single_json_fence"},
                "json_status": {"not_evaluated", "parsed", "failed"}}.items():
            value = transport.get(key)
            if type(value) is str and value in allowed:
                safe[key] = value
        for key in ("wrapper_removed", "content_repaired"):
            if type(transport.get(key)) is bool:
                safe[key] = transport[key]
        result["response_transport"] = safe
    application = getattr(updater, "last_application_summary", None)
    if type(application) is dict:
        result["application_counts"] = {key: application[key] for key in (
            "proposal_count", "proposal_results_count", "omitted_proposal_results_count",
            "applied_count", "deferred_count", "rejected_count")
            if type(application.get(key)) is int and 0 <= application[key] <= 10000}
    return result


class _ReflectionCapture:
    """Experimental first answer and updater result, never a retry or repair."""

    def __init__(self, app, directory, *, case_id, session, session_id, source_sha256):
        self.app, self.directory = app, Path(directory)
        if (not isinstance(app, E1bExperimentApp)
                or Path(app.experiment.root).resolve() not in self.directory.resolve().parents):
            raise ValueError("reflection capture requires the isolated experiment directory")
        self.case_id, self.session, self.session_id = case_id, session, session_id
        self.source_sha256 = source_sha256
        self.started, self.first = False, None
        self.capture_error = None
        self.attempt_path = self.directory / f"reflection-s{session}.attempt.json"
        self.first_path = self.directory / f"reflection-s{session}.first.json"
        self.outcome_path = self.directory / f"reflection-s{session}.product.json"

    def _save(self, path, value):
        try:
            _exclusive_json(path, value)
        except Exception:
            self.capture_error = "reflection_capture_save_failed"
            raise

    def _bind_receipt(self, first, raw):
        if self.app.call_guard is None:
            return True
        try:
            state = self.app.call_guard.snapshot()
            attempts = _first_answer_attempts(self.app.call_guard, state) if raw is not None else state["attempts"]
            rows = [row for row in attempts
                    if row.get("role") == "reflection" and row.get("session_id") == self.session_id]
        except Exception:
            rows = []
        if (len(rows) != 1 or rows[0].get("operation_id") != f"e1b-review-{self.session_id}"
                or type(rows[0].get("request_sha256")) is not str
                or not re.fullmatch(r"[a-f0-9]{64}", rows[0]["request_sha256"])):
            return False
        row = rows[0]
        first.update(request_sha256=row["request_sha256"], request_hash_kind="actual_wire")
        visible_sha = row.get("visible_content_sha256")
        if type(visible_sha) is str and re.fullmatch(r"[a-f0-9]{64}", visible_sha):
            first["wire_visible_content_sha256"] = visible_sha
        response_sha = row.get("client_return_content_sha256")
        if isinstance(response_sha, str) and re.fullmatch(r"[a-f0-9]{64}", response_sha):
            if raw is not None and response_sha != first["visible_response_sha256"]:
                return False
            if raw is None:
                first.update(visible_response_sha256=response_sha, response_hash_source="receipt_only")
        elif raw is not None:
            return False
        first["receipt_status"] = (row["status"] if type(row.get("status")) is str
            and row["status"] in {"received", "failed", "pending"} else "unknown")
        return raw is None or row.get("status") == "received"

    def complete(self, client, messages):
        if self.started:
            self.capture_error = "reflection_first_attempt_already_consumed"
            raise E1bRunStopped(self.capture_error)
        self.started = True
        self._save(self.attempt_path, {"case_id": self.case_id, "session": self.session,
            "session_id": self.session_id, "status": "started", "source_sha256": self.source_sha256})
        first = {"schema": "e1b_reflection_first_v1", "case_id": self.case_id,
            "source_sha256": self.source_sha256, "session": self.session, "session_id": self.session_id,
            "request_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "request_hash_kind": "software_messages" if self.app.call_guard is None else "input_messages_not_wire",
            "visible_response_sha256": None, "wire_visible_content_sha256": None,
            "response_hash_source": "unavailable",
            "parse_status": "call_failed_or_unknown", "public_projection": None,
            "public_projection_dropped_fields": 0, "duplicate_key_count": 0}
        try:
            raw = client.complete(messages)
        except Exception:
            first["client_diagnostic"] = _reflection_client_diagnostic(client)
            if not self._bind_receipt(first, None):
                first["request_identity_missing"] = True
            self._save(self.first_path, first)
            self.first = first
            raise
        first["client_diagnostic"] = _reflection_client_diagnostic(client)
        if isinstance(raw, str):
            first.update(visible_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                         response_hash_source="client_return_content")
            try:
                if len(raw.encode("utf-8")) > 128 * 1024:
                    raise ValueError("reflection_capture_body_limit")
                body, wrapper = _unwrap_review_response(raw)
                def count_duplicates(pairs):
                    value = {}
                    for key, item in pairs:
                        first["duplicate_key_count"] += int(key in value)
                        value[key] = item
                    return value
                parsed = json.loads(body, object_pairs_hook=count_duplicates, parse_constant=_reject_nonfinite_json)
                first["parse_status"] = "object" if type(parsed) is dict else "not_object"
                first["transport_wrapper"] = wrapper
                if type(parsed) is dict:
                    first["known_fields_present"] = {key: key in parsed for key in ("protocol_version", "reviews", "changes")}
                    first["known_field_types"] = {key: type(parsed[key]).__name__ for key in first["known_fields_present"] if key in parsed}
                    first["public_projection"], first["public_projection_dropped_fields"] = _reflection_public_projection(parsed)
            except (ValueError, TypeError, RecursionError):
                first["parse_status"] = "parse_error"
        else:
            first["parse_status"] = "non_text_response"
        identity_ok = self._bind_receipt(first, raw)
        if not identity_ok:
            first["request_identity_missing"] = True
        self._save(self.first_path, first)
        self.first = first
        if not identity_ok:
            self.capture_error = "reflection_request_identity_unavailable"
            raise E1bRunStopped(self.capture_error)
        # Product validation receives the untouched answer. An absent/malformed
        # public field is never converted into a successful empty response.
        return raw

    def finish(self, updater, *, updater_raised=False):
        value = {"schema": "e1b_reflection_product_v1", "session": self.session,
                 "session_id": self.session_id, "first_available": self.first is not None,
                 "first_sha256": hashlib.sha256(self.first_path.read_bytes()).hexdigest() if self.first is not None else None,
                 "capture_error": self.capture_error, "updater_raised": updater_raised,
                 **_reflection_product_diagnostic(updater)}
        self._save(self.outcome_path, value)
        if self.first is None or self.capture_error:
            raise E1bRunStopped(self.capture_error or "reflection_first_response_missing")


class _CapturedReflectionUpdater:
    def __init__(self, updater, capture):
        self.updater, self.capture = updater, capture
        client = updater.llm
        self.reflection_metered_client = client
        class RecordedClient:
            def complete(self, messages):
                return capture.complete(client, messages)

            def __getattr__(self, name):
                return getattr(client, name)
        updater.llm = RecordedClient()

    def __getattr__(self, name):
        return getattr(self.updater, name)

    def update(self, persona, ctx):
        if ctx.session_id != self.capture.session_id:
            raise E1bRunStopped("reflection capture session mismatch")
        try:
            result = self.updater.update(persona, ctx)
        except Exception:
            self.capture.finish(self.updater, updater_raised=True)
            raise
        self.capture.finish(self.updater)
        return result


class E1bExperimentApp(MinimalWebApp):
    """Explicit Python-only synthetic experiment; the ordinary Web cannot opt in."""

    def __init__(self, data_dir: str | Path, *, experiment, clients, **kwargs):
        self.experiment = experiment
        self.external_emotion_experiment = experiment
        self.proposal_capture = None
        self.reflection_capture = None
        root, target = Path(experiment.root).resolve(), Path(data_dir).resolve()
        if target == root or root not in target.parents:
            raise ValueError("E1b app directory must be inside experiment root")
        super().__init__(target, clients=clients, **kwargs)

    def _seed_persona(self, user_id):
        return PersonaProfile(user_id=user_id, profile_origin=self.experiment.profile_origin)

    def _engine_for_persona(self, persona):
        engine = create_minimal_engine(
            get_scene(self.scene_id), persona, clients=self.clients,
            store=self.store, delivery_block=self.delivery_block,
            external_emotion_experiment=self.experiment,
        )
        if self.proposal_capture is not None:
            engine._consolidator.extractor = _CapturedExtractor(engine._consolidator.extractor, self.proposal_capture)
        if self.reflection_capture is not None and engine.hypothesis_updater is not None:
            engine.hypothesis_updater = _CapturedReflectionUpdater(engine.hypothesis_updater, self.reflection_capture)
            engine.diagnoser = engine.hypothesis_updater
        return engine

    def state(self):
        state = super().state()
        fixture = all(isinstance(client, E1bLocalFixtureClient) for client in
                      (self.clients.online, self.clients.memory, self.clients.reflection))
        state["evidence_kind"] = "software_fixture" if fixture else "real_model"
        state["synthetic_emotion"] = True
        state["notice"] = ("E1b 软件替身：没有真实模型效果；情绪读数为固定种子合成。" if fixture else
                           "E1b 虚构人物实验：回复由实际模型生成，情绪读数为固定种子合成。")
        for row in state["turns"]:
            row["evidence_kind"] = state["evidence_kind"]
        return state


def strip_a0_preference_sources(persona: PersonaProfile, *, session_id: str,
                                turn_index: int) -> tuple[PersonaProfile, dict]:
    """Clone and remove full records with any lineage to the preference turn.

    A mixed-source hypothesis is removed as a whole; deleting only one evidence
    entry would retain the conclusion. The original A1 profile is untouched.
    """
    result = persona.model_copy(deep=True)
    removed = {"preferences": [item.id for item in result.stated_preferences],
               "facts": [], "hypotheses": [], "relationships": []}
    result.stated_preferences = []
    removed["facts"] = [item.id for item in result.episodic_memory
                        if item.source_session_id == session_id and item.source_turn_index == turn_index]
    result.episodic_memory = [item for item in result.episodic_memory if item.id not in removed["facts"]]
    removed["hypotheses"] = [item.id for item in result.person_hypotheses
                             if any(row.session_id == session_id and turn_index in row.turn_indices
                                    for row in item.evidence)]
    result.person_hypotheses = [item for item in result.person_hypotheses
                               if item.id not in removed["hypotheses"]]
    # Relationship facts can also carry the same preference. Their schema has
    # record-level lineage, so remove the entire mixed record conservatively.
    removed["relationships"] = [item.id for item in result.relationship_context
                                if any(row.session_id == session_id and turn_index in row.turn_indices
                                       for row in item.evidence)]
    result.relationship_context = [item for item in result.relationship_context
                                   if item.id not in removed["relationships"]]
    return result, removed


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


class E1bRunner:
    """Runs disclosed turns through Engine, actual review, and UI confirmation.

    app_factory receives ONLY a fresh directory, never a case or answer key.
    operation_gate can bind a shared controller before each action. The three
    repeated branches share emotion readings and historical profile bytes but
    receive new session/run IDs; no clock is changed to imply elapsed days.
    """

    reflection_capture_class = _ReflectionCapture

    def __init__(self, suite: E1bSuite, *, root: str | Path,
                 app_factory: Callable[[Path], MinimalWebApp], seed: int,
                 evidence_kind: str = "software_fixture", allow_model_calls: bool = False,
                 operation_gate: Callable[[dict], None] | None = None):
        if evidence_kind not in {"software_fixture", "real_model"}:
            raise ValueError("explicit evidence kind required")
        if evidence_kind == "real_model" and allow_model_calls is not True:
            raise ValueError("real model use requires a separately authorized guarded caller")
        if allow_model_calls and evidence_kind != "real_model":
            raise ValueError("real calls cannot be labeled software fixtures")
        self.suite, self.root, self.factory = suite, Path(root).resolve(), app_factory
        self.seed, self.evidence_kind, self.gate = seed, evidence_kind, operation_gate
        self.plan = synthetic_emotion_plan(suite, seed)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / "operations.jsonl"
        if self.journal.exists():
            raise ValueError("existing E1b run is read-only; no automatic replay or retry")
        self.journal.touch(exist_ok=False)
        self.stopped = False
        self._shared_guard = None
        self._shared_guard_identity = None
        self._sequence = 0
        self._journal({"event": "created", "runner_version": VERSION,
                       "source_sha256": suite.source_sha256, "seed": seed,
                       "evidence_kind": evidence_kind, "synthetic_emotion": True,
                       "frozen": False, "main_gate": "not_evaluated"})
        _write_json(self.root / "emotion-plan.json", {
            "source_sha256": suite.source_sha256, "seed": seed,
            "synthetic": True, "repeat_policy": "identical_across_arms_and_repeats",
            "readings": [{"case_id": case_id, "session": session, "turn": turn, "reading": reading}
                         for (case_id, session, turn), reading in self.plan.items()],
        })

    def _journal(self, row: dict):
        with self.journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _operation(self, app, name: str, operation: Callable, *, detail: dict | None = None):
        if self.stopped:
            raise E1bRunStopped("run already stopped")
        self._sequence += 1
        attempt = {"id": self._sequence, "operation": name, **(detail or {})}
        try:
            # Persist first: if a process dies while an operation is in flight,
            # the old directory cannot be replayed as a fresh paid experiment.
            self._journal({**attempt, "event": "pending"})
            if self.gate:
                self.gate(deepcopy(attempt))
            if self.evidence_kind == "real_model":
                self._bind_shared_guard(app)
            result = operation()
            if isinstance(result, dict) and result.get("ok") is False:
                raise E1bRunStopped("product_operation_failed_or_unknown")
            self._journal({**attempt, "event": "completed"})
            return result
        except Exception as error:
            self.stopped = True
            app._halt_models()
            try:
                self._journal({**attempt, "event": "stopped", "error_type": type(error).__name__})
            except OSError:
                pass
            raise E1bRunStopped("operation failed or result unknown; inspect original records") from error

    def _new_app(self, path: Path):
        if path.exists():
            raise ValueError("branch already exists; refusing to reuse it")
        app = self.factory(path)
        if not isinstance(app, E1bExperimentApp):
            app.close()
            raise ValueError("synthetic emotion requires E1bExperimentApp isolation")
        clients = (app.clients.online, app.clients.memory, app.clients.reflection)
        if self.evidence_kind == "software_fixture":
            if not all(isinstance(client, E1bLocalFixtureClient) for client in clients):
                app.close()
                raise ValueError("software rehearsal accepts only named local fixture clients")
        else:
            from aria.engine.policy.llm import OpenRouterLLMClient
            from aria.engine.policy.reserved_calls import ReservedCallGuard
            if not (isinstance(app.call_guard, ReservedCallGuard)
                    and all(isinstance(client, OpenRouterLLMClient) for client in clients)):
                app.close()
                raise ValueError("real E1b needs bounded clients with a pre-send reservation guard")
        if app.care.external or app.care.allow_external:
            app.close()
            raise ValueError("E1b cannot enable external care delivery")
        return app

    def _bind_shared_guard(self, app):
        guard = app.call_guard
        identity = (guard.ledger_id, guard.policy_sha256, guard.path.resolve())
        if (getattr(app, "_owns_call_guard", True)
                or any(getattr(client, "model_request_control", None) is not guard
                       for client in (app.clients.online, app.clients.memory, app.clients.reflection))
                or (self._shared_guard is not None and (guard is not self._shared_guard or identity != self._shared_guard_identity))):
            self.stopped = True
            if self._shared_guard is not None:
                self._shared_guard.halt()
            raise E1bRunStopped("all real E1b apps must use one externally owned guard and ledger identity")
        guard._check_policy()
        self._shared_guard, self._shared_guard_identity = guard, identity

    def _session(self, app, case: E1bCase, number: int, *, arm: str, repeat: int):
        metadata = {"case_id": case.id, "phase": f"s{number}", "arm": arm, "repeat": repeat}
        started = self._operation(app, "start", app.start, detail=metadata)
        session_id = started["state"]["session_id"]
        for turn in range(len(case.session(number))):
            text = case.current_input(number, turn)
            # Only text + this reading are submitted. Case/category/rubric,
            # expected preference, emotion truth and future turns stay outside.
            payload = deepcopy(self.plan[(case.id, number, turn)])
            self._operation(app, "submit", lambda: app.submit(
                text, f"e1b-{session_id}-t{turn}", session_id, external_emotion=payload),
                detail={**metadata, "turn": turn})
        self._operation(app, "seal", lambda: app.seal(session_id), detail=metadata)
        return session_id

    def _assert_profile(self, persona):
        if persona is None or persona.pending_session or persona.offline_sessions:
            raise E1bRunStopped("shared sessions must be fully sealed and reviewed")
        if persona.cognitive_health_consent.text_monitoring_enabled:
            raise E1bRunStopped("C1 must remain disabled in E1b")

    def run_case(self, case_id: str, *, phase: str = "main", repeats: int = 3) -> dict:
        if phase not in {"main", "prescreen"} or type(repeats) is not int or repeats != 3:
            raise ValueError("registered layout requires main/prescreen and exactly three repeats")
        case = next((item for item in self.suite.items if item.id == case_id), None)
        if case is None:
            raise ValueError("unknown case")
        case_root = self.root / phase / case.id
        case_root.mkdir(parents=True, exist_ok=False)
        report = {"version": VERSION, "case_id": case.id, "phase": phase,
                  "evidence_kind": self.evidence_kind, "synthetic_emotion": True,
                  "frozen": False, "effects": "not_evaluated", "shared": None,
                  "branches": [], "scored_reply_count": 0}
        shared_profile, preference_session = None, None
        if phase == "main":
            app = self._new_app(case_root / "shared")
            try:
                for number in (1, 2):
                    session_id = self._session(app, case, number, arm="shared", repeat=0)
                    app.proposal_capture = _ProposalCapture(app, case_root, case_id=case.id, session=number,
                        session_id=session_id, source_sha256=self.suite.source_sha256)
                    app.reflection_capture = self.reflection_capture_class(app, case_root, case_id=case.id, session=number,
                        session_id=session_id, source_sha256=self.suite.source_sha256)
                    try:
                        self._operation(app, "reflect", lambda: app.reflect(f"e1b-review-{session_id}"),
                                        detail={"case_id": case.id, "session": number})
                    finally:
                        app.proposal_capture = None
                        app.reflection_capture = None
                    if number == 2:
                        preference_session = session_id
                # Deliberately confirm every proposed item, even wrong ones.
                # Matching an expected answer here would bias the experiment.
                proposals = [row.id for row in app._profile().stated_preferences if row.status == "proposed"]
                for preference_id in proposals:
                    self._operation(app, "confirm_preference", lambda: app.preference_action("confirm", preference_id),
                                    detail={"case_id": case.id, "preference_id": preference_id})
                shared_profile = app._profile().model_copy(deep=True)
                self._assert_profile(shared_profile)
                report["shared"] = {"state": app.state(), "profile": shared_profile.model_dump(mode="json"),
                                    "confirmation_ids": proposals, "memory_review_count": 2,
                                    "reflection_review_count": 2}
                self._operation(app, "save_shared", lambda: _write_json(case_root / "shared.json", report["shared"]))
            finally:
                app.close()
        for repeat in range(1, repeats + 1):
            # Balance arm order while keeping all emotion inputs identical.
            arms = ("A0",) if phase == "prescreen" else (("A0", "A1") if (repeat + self.seed) % 2 else ("A1", "A0"))
            for arm in arms:
                app = self._new_app(case_root / f"repeat-{repeat}-{arm}")
                try:
                    removed = {"preferences": [], "facts": [], "hypotheses": [], "relationships": []}
                    if shared_profile is not None:
                        branch_profile = shared_profile.model_copy(deep=True)
                        if arm == "A0":
                            branch_profile, removed = strip_a0_preference_sources(
                                branch_profile, session_id=preference_session, turn_index=case.pref_turn)
                        self._assert_profile(branch_profile)
                        # Only persona data travels across the fork. No care DB,
                        # cost ledger, active session or app run ID is copied.
                        self._operation(app, "import_profile", lambda: app.store.save(branch_profile))
                    self._session(app, case, 3, arm=arm, repeat=repeat)
                    state = app.state()
                    rows = state["turns"]
                    if len(rows) != 10:
                        raise E1bRunStopped("unexpected third-session turn count")
                    branch = {"arm": arm, "repeat": repeat, "run_id": state.get("run_id"),
                              "removed": removed, "state": state,
                              "scored_turn_ids": [row["turn_id"] for row in rows[7:]],
                              "scored_replies": [{"turn": index, "response": rows[index]["response"],
                                                   "source": rows[index]["source"],
                                                   "evidence_kind": self.evidence_kind} for index in (7, 8, 9)]}
                    report["branches"].append(branch)
                    report["scored_reply_count"] += 3
                    self._operation(app, "save_branch", lambda: _write_json(
                        case_root / f"repeat-{repeat}-{arm}.json", branch))
                finally:
                    app.close()
        self._operation(app, "save_case_report", lambda: _write_json(case_root / "report.json", report))
        return report


class E1bLocalFixtureClient:
    """A transparent software double, with no access to cases or answer keys."""

    source_id = "e1b_local_software_fixture"
    model = "local/e1b-software-fixture"

    def __init__(self, role: str):
        self.role, self.calls = role, []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        if self.role == "reflection":
            return '{"protocol_version":"2.0","reviews":[],"changes":[]}'
        if self.role == "memory":
            disclosed = re.findall(r"\[turn (\d+)\] 用户:([^\n]+)", messages[-1]["content"])
            episodes = []
            if disclosed:
                index, quote = disclosed[0]
                episodes.append({"summary": quote, "evidence_quote": quote,
                                 "turn_index": int(index), "keywords": [], "referenceable": False})
            # No desired preference is prefilled. Specific proposal tests may
            # inject a different fixture that derives solely from shown text.
            return json.dumps({"episodes": episodes, "relationships": [], "stated_preferences": []}, ensure_ascii=False)
        digest = hashlib.sha256(messages[-1]["content"].encode()).hexdigest()[:10]
        return f"【软件替身，非真实模型效果】已处理本轮输入 {digest}。"


def local_fixture_factory(experiment, *, recordings: list | None = None):
    def factory(path):
        clients = MinimalModelClients(*(E1bLocalFixtureClient(role) for role in
                                       ("online", "memory", "reflection")), model_calls_enabled=True)
        app = E1bExperimentApp(path, experiment=experiment, clients=clients)
        if recordings is not None:
            recordings.append((app, clients))
        return app
    return factory
