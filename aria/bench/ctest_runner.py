"""C-test offline-memory component runner, isolated fictional clock and people.

Calls the actual LLMExtractor/MemoryConsolidator, including grounded C1 rules.
Online replies and reflection are not called: the authored assistant context is
explicitly synthetic. No effect on a real profile, care service or notification.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from aria.engine.cognitive import capture_cognitive_turn, cognitive_action, cognitive_sensitive
from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator, _parse_json_payload
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.scenes.companion_elder.medical_privacy import guard_cognitive_health_reply, is_cognitive_health_sensitive_utterance
from .ctest_cases import VIRTUAL_EPOCH, ctest_manifest, load_ctest_cases
from .e1b_runner import _write_json


class CTestRunStopped(RuntimeError):
    pass


class _RecordedMemoryCall:
    """Record public structured output, never provider reasoning or prose prefixes."""

    def __init__(self, client, output):
        self.client, self.output = client, output

    def complete(self, messages):
        raw = self.client.complete(messages)
        if not isinstance(raw, str):
            raise CTestRunStopped("memory response is not public text")
        record = {"visible_content_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                  "provider_reasoning_recorded": False}
        try:
            parsed = _parse_json_payload(raw)
            if not isinstance(parsed, dict):
                raise ValueError("unexpected envelope")
            allowed = {"episodes", "relationships", "stated_preferences", "cognitive_observations"}
            # Top-level unrequested fields and non-JSON prose are not copied.
            def visible(value):
                if isinstance(value, dict):
                    return {key: visible(item) for key, item in value.items()
                            if not re.search(r"reasoning|analysis|thinking|chain.of.thought", key, re.I)}
                if isinstance(value, list):
                    return [visible(item) for item in value]
                return value
            record["memory_json"] = {key: visible(value) for key, value in parsed.items() if key in allowed}
            record["status"] = "structured_visible_content"
            observations = parsed.get("cognitive_observations")
            if not isinstance(observations, list) or any(type(row) is not dict for row in observations):
                record["status"] = "cognitive_protocol_missing_or_invalid"
        except (ValueError, TypeError):
            record["status"] = "unstructured_content_hash_only"
        _write_json(self.output, record)
        if record["status"] == "cognitive_protocol_missing_or_invalid":
            raise CTestRunStopped("missing cognitive output is not a measured empty observation set")
        return raw


class CTestLocalMemoryFixture:
    """Overinclusive software extractor; product rules must reject context.

    It sees only the production prompt, never groups, expected triggers, future
    visits or the generated observation store. Not evidence of model accuracy.
    """
    source_id = "ctest_local_software_fixture"
    model = "local/ctest-memory-fixture"

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        content = messages[-1]["content"]
        suffix = content.split("已同意本人原话（全部审阅；只有标注出的认知隐私轮次禁止作为普通事实或人物证据）：\n", 1)
        sources = json.JSONDecoder().raw_decode(suffix[1])[0] if len(suffix) == 2 else []
        rows = []
        for source in sources:
            text = source["text"]
            if re.search(r"电费|水费|账单|算不清|买菜|做饭", text):
                domain = "daily_function"
            elif re.search(r"叫啥|叫什么|说不出", text):
                domain = "language"
            elif re.search(r"日期|星期|礼拜|搞混", text):
                domain = "orientation"
            else:
                domain = "memory"
            rows.append({"domain": domain, "evidence_quote": text, "turn_index": source["turn_index"],
                         "functional_impact": bool(re.search(r"耽误|影响|没法", text))})
        return json.dumps({"episodes": [], "relationships": [], "stated_preferences": [],
                           "cognitive_privacy": [], "cognitive_observations": rows}, ensure_ascii=False)


class CTestRunner:
    def __init__(self, output: str | Path, *, experiment_id: str, memory_client=None,
                 guard=None, allow_model_calls=False, registration_sha256=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment_id):
            raise ValueError("invalid experiment id")
        self.client = memory_client or CTestLocalMemoryFixture()
        self.fixture = isinstance(self.client, CTestLocalMemoryFixture)
        if self.fixture and allow_model_calls:
            raise ValueError("software fixture cannot be labeled real model")
        if registration_sha256 is not None and not (isinstance(registration_sha256, str)
                and re.fullmatch(r"[a-f0-9]{64}", registration_sha256)):
            raise ValueError("invalid externally verified registration SHA")
        if not self.fixture:
            from aria.engine.policy.llm import OpenRouterLLMClient
            from aria.engine.policy.reserved_calls import ReservedCallGuard
            if not (allow_model_calls is True and isinstance(self.client, OpenRouterLLMClient)
                    and isinstance(guard, ReservedCallGuard)):
                raise ValueError("real C-test needs separately authorized bounded client and guard")
            if registration_sha256 is None:
                raise ValueError("real C-test needs externally verified frozen registration")
            guard.attach(self.client)
        self.registration_sha256 = registration_sha256
        self.guard = guard
        self.root = Path(output).resolve() / "synthetic-cognitive-experiments" / experiment_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.cases = load_ctest_cases()
        self.manifest = ctest_manifest(self.cases)
        self._script_sha256 = self.manifest["source_sha256"]
        self.origin = f"synthetic_cognitive_experiment:{experiment_id}"
        self.store = PersonaStore(self.root / "profiles")
        self.journal = self.root / "operations.jsonl"
        self.journal.touch(exist_ok=False)
        self.stopped = False
        _write_json(self.root / "manifest.json", {
            **self.manifest, "frozen": not self.fixture,
            "externally_verified_registration_sha256": self.registration_sha256,
        })

    def _record(self, value):
        with self.journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def run(self):
        if self.stopped or (self.root / "summary.json").exists():
            raise CTestRunStopped("existing run is read-only; no automatic replay")
        reports = []
        try:
            for case in self.cases:
                user_id = f"fictional-{case.id}"
                persona = PersonaProfile(user_id=user_id, profile_origin=self.origin)
                cognitive_action(persona, "enable", actor=user_id, now=VIRTUAL_EPOCH)
                self._record({"event": "synthetic_owner_enabled", "case_id": case.id,
                              "simulated_owner_action": True, "virtual_at": VIRTUAL_EPOCH.isoformat(),
                              "observed_at": datetime.now(timezone.utc).isoformat()})
                self.store.save(persona)
                visits = []
                for visit in case.sessions:
                    if ctest_manifest(self.cases)["source_sha256"] != self._script_sha256:
                        raise CTestRunStopped("script changed after registration; no further calls")
                    ctx = EngineContext(user_id=user_id, scene_id="companion_elder",
                                        session_id=f"{case.id}-s{visit.number}")
                    excluded = []
                    for index, (words, reply) in enumerate(zip(visit.utterances, visit.synthetic_assistant)):
                        if guard_cognitive_health_reply(reply) != reply:
                            raise CTestRunStopped("authored assistant context failed reply guard")
                        capture_cognitive_turn(ctx, persona, words, now=visit.virtual_at)
                        if cognitive_sensitive(words) or is_cognitive_health_sensitive_utterance(words):
                            excluded.append(index)
                        ctx.add_turn(words, reply)
                    ctx.working_memory["_sensitive_memory_excluded_turns"] = excluded
                    ctx.working_memory["_cognitive_review_at"] = visit.virtual_at.isoformat()
                    operation = {"case_id": case.id, "session": visit.number,
                                 "virtual_at": visit.virtual_at.isoformat(), "session_id": ctx.session_id}
                    self._record({**operation, "event": "pending"})
                    candidate = persona.model_copy(deep=True)
                    extractor = LLMExtractor(_RecordedMemoryCall(self.client, self.root / f"{case.id}-s{visit.number}-response.json"))
                    consolidator = MemoryConsolidator(self.store, extractor)
                    scope = self.guard.scope("review", f"ctest-{case.id}-s{visit.number}", ctx.session_id) if self.guard else nullcontext()
                    with scope:
                        consolidator.consolidate(candidate, ctx, persist=False)
                    if extractor.last_status != "ok":
                        raise CTestRunStopped("memory failed or response unknown; do not resend")
                    self.store.save(candidate)
                    persona = candidate
                    result = {**operation, "status": persona.cognitive_trend.status,
                              "counting_observations": sum(row.counting for row in persona.cognitive_observations),
                              "profile": persona.model_dump(mode="json"),
                              "evidence_kind": "synthetic_fixture" if self.fixture else "real_memory_on_synthetic_script"}
                    _write_json(self.root / f"{case.id}-s{visit.number}.json", result)
                    self._record({**operation, "event": "completed"})
                    visits.append({key: result[key] for key in ("session", "virtual_at", "status", "counting_observations")})
                reports.append({"case_id": case.id, "group": case.group, "sessions": visits,
                                "triggered": any(row["status"] == "review_suggested" for row in visits)})
            change = sum(row["triggered"] for row in reports if row["group"] == "change")
            control = sum(row["triggered"] for row in reports if row["group"] == "control")
            passed = control == 0 and change >= 4
            summary = {"evidence_kind": "synthetic_fixture" if self.fixture else "real_memory_on_synthetic_script",
                       "case_count": 12, "session_count": 96,
                       "change_triggered": change, "change_denominator": 6,
                       "control_triggered": control, "control_denominator": 6,
                       "software_rule_check": "pass" if passed else "fail",
                       "real_model_gate": "not_executed" if self.fixture else "pass" if passed else "fail",
                       "memory_calls": 96, "reflection_calls": 0, "online_calls": 0,
                       "scripted_reply_guard_checks": 960, "real_online_reply_guard_checks": 0,
                       "virtual_clock_only": True, "synthetic_assistant_context": True,
                       "source_sha256": self._script_sha256, "frozen": not self.fixture,
                       "externally_verified_registration_sha256": self.registration_sha256,
                       "limitations": ["functional-impact branch only", "not clinical evidence", "not complete Engine path"],
                       "cases": reports}
            _write_json(self.root / "summary.json", summary)
            return summary
        except Exception as error:
            self.stopped = True
            if self.guard:
                self.guard.halt()
            try:
                self._record({"event": "stopped", "error_type": type(error).__name__,
                              "real_model_gate": "not_evaluable"})
            except OSError:
                pass
            raise CTestRunStopped("C-test stopped; original attempt preserved, no automatic retry") from error
