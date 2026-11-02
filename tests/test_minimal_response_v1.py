"""Release state/provenance regressions with forced model labels, not effects."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile, SessionCheckpointRecord
from aria.engine.risk.source_response import KEY, current_source_response
from aria.engine.policy.minimal_calls import MinimalCallGuard
from aria.interfaces.risk import RiskSignal
from aria.runtime import create_minimal_engine
from aria.release import engine_options
from aria.scenes.registry import get_scene
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence
from tests.test_safety_response_instruction import bundle

TEXT = "中性软件来源甲。"
BODY = "中性正文甲。\n中性正文乙？\n中性正文丙？"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def build(monkeypatch, value=None, *, premodel=False):
    clients = bundle()
    value = payload(response=BODY) if value is None else value
    wires, order = [], []
    def receive(request, **kwargs):
        order.append("model")
        wire = json.loads(request.data)
        MinimalCallGuard._validate_payload(clients.online, wire, **kwargs)
        wires.append(wire)
        if isinstance(value, Exception):
            raise value
        return envelope(value, "json_schema_v1")
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
                                   clients=clients, **engine_options())
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="neutral", evidence=[]) if premodel else None)
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires, order


@pytest.mark.parametrize("row", [
    evidence(TEXT, "owner", "historical", "not_immediate"),
    evidence(TEXT, "quotation_or_hypothetical", "current", "not_immediate"),
    evidence(TEXT, "uncertain", "uncertain", "uncertain"),
    evidence("not present", "other", "current", "immediate"),
])
def test_nonacute_labels_preserve_body_but_never_grant_learning_or_actions(monkeypatch, row):
    engine, ctx, wires, _ = build(monkeypatch, payload("fall", row, BODY))
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("care"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *a: effects.append("skill"))
    assert engine.process_turn(ctx, TEXT) == BODY
    assert len(wires) == 1 and not effects and not engine.persona.risk_events
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    audit = ctx.working_memory["safety_delivery"]
    assert audit["status"] == "source_bound_guarded" and audit["segment_ids"] == []
    assert audit["semantic_verified"] is audit["grants_permission"] is False


def test_ordinary_layout_and_questions_are_not_style_rewritten(monkeypatch):
    engine, ctx, wires, _ = build(monkeypatch)
    assert engine.process_turn(ctx, TEXT) == BODY
    assert len(wires) == 1
    assert sum(len(m["content"].encode()) for m in wires[0]["messages"] if m["role"] == "system") <= 12000
    assert "response_format" in wires[0]
    assert not ctx.working_memory.get("_sensitive_memory_excluded_turns")


@pytest.mark.parametrize("row", [evidence(TEXT, "owner", "historical", "not_immediate"),
                                  evidence("not present", "owner", "current", "immediate")])
def test_nonacute_second_guard_does_not_reapply_legacy_style_rewrite(monkeypatch, row):
    body = "不要说‘别想太多’，先听完这段话。"
    engine, ctx, _, _ = build(monkeypatch, payload("fall", row, body))
    assert engine.process_turn(ctx, TEXT) == body
    assert ctx.working_memory["safety_delivery"]["status"] == "source_bound_guarded"


@pytest.mark.parametrize("subject", ["owner", "other"])
def test_current_acute_body_isolated_and_other_not_recorded_as_owner(monkeypatch, subject):
    engine, ctx, _, _ = build(monkeypatch, payload("fall", evidence(TEXT, subject, "current", "immediate"), BODY))
    assert engine.process_turn(ctx, TEXT) == BODY
    assert ctx.working_memory["intervention_delivery"]["reason"] == "semantic_safety_takeover"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    assert bool(engine.persona.risk_events) == (subject == "owner")


@pytest.mark.parametrize("body", ["我已经帮你发给家人了。", "可以服双倍药量。", "你肯定患了阿尔茨海默症。"])
def test_guard_rejection_does_not_add_unrelated_crisis_catalog(monkeypatch, body):
    engine, ctx, _, _ = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "historical", "not_immediate"), body))
    result = engine.process_turn(ctx, TEXT)
    assert result != body
    assert ctx.working_memory["safety_delivery"]["status"] == "guard_replaced_without_crisis_prelude"
    assert ctx.working_memory["safety_delivery"]["segment_ids"] == []
    assert "<action" not in result and not engine.persona.risk_events


@pytest.mark.parametrize("change", ["user", "session", "scene", "turn", "raw", "guarded", "assessment", "audit", "version"])
def test_exact_source_binding_still_required(monkeypatch, change):
    engine, ctx, _, _ = build(monkeypatch, payload("fall", evidence("absent", "other", "current", "immediate"), BODY))
    engine.process_turn(ctx, TEXT)
    ctx.turns.pop()
    assert current_source_response(ctx, TEXT, engine.persona) is not None
    if change in ("user", "session", "scene"):
        setattr(ctx, change + "_id", "other")
    elif change == "turn": ctx.add_turn("other", "other")
    elif change in ("raw", "guarded"):
        ctx.working_memory[KEY] = replace(ctx.working_memory[KEY], **{change: "other"})
    elif change == "assessment": ctx.working_memory["online_turn_decision"]["safety_evidence"]["subject"] = "owner"
    elif change == "audit": ctx.working_memory["online_turn_decision"]["source_response"]["semantic_verified"] = True
    else: ctx.working_memory["_safety_delivery_version"] = "source_response_v4"
    assert current_source_response(ctx, TEXT, engine.persona) is None


@pytest.mark.parametrize("failure", [None, "callback", "client"])
def test_premodel_durable_prelude_before_one_isolated_call(monkeypatch, failure):
    engine, ctx, wires, order = build(monkeypatch, RuntimeError("software") if failure == "client" else None, premodel=True)
    events = []
    def emit(event):
        events.append(deepcopy(event)); order.append("prelude")
        return failure != "callback"
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert events[0]["premodel_safety_version"] == "task_response_v4"
    assert order == (["prelude"] if failure == "callback" else ["prelude", "model"])
    assert len(wires) == (0 if failure == "callback" else 1)
    assert "_premodel_task_inflight" not in ctx.working_memory
    if failure is None: assert result == BODY
    elif failure == "callback": assert result == events[0]["protection"]
    else: assert result != BODY


def test_checkpoint_versions_round_trip():
    record = SessionCheckpointRecord(session_id="s", scene_id="companion_elder", updated_at="2026-10-13T00:00:00Z", **engine_options())
    assert SessionCheckpointRecord.model_validate_json(record.model_dump_json()) == record
