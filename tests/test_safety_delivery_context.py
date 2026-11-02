"""Synthetic labels check delivery plumbing, never natural-language accuracy."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.risk.safety_delivery import (
    CONDITIONAL_CONTEXT_V1 as CANDIDATE, SEGMENTS, capture_delivery_context,
)
from aria.engine.risk.safety_evidence import UNCERTAIN_REPLY, THIRD_PERSON_REPLY
from aria.engine.risk.semantic import semantic_safety_signal
from aria.interfaces.risk import RiskSignal
from aria.runtime import create_minimal_clients, create_minimal_engine
from aria.scenes.registry import get_scene
from tests._model_http_stub import stub_model_http
from tests.test_safety_evidence import evidence, payload, V4

TEXT = "本轮软件来源甲。"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected model network"))


def build(monkeypatch, value, *, version=CANDIDATE, online_version=V4):
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-software",
                                     online_decision_version=online_version)
    wires = []
    def receive(request, **options):
        wires.append(request.data)
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(value, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
        clients=clients, safety_delivery_version=version)
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires


@pytest.mark.parametrize("concern", ["self_harm", "medical_emergency", "fall", "unsafe_medication"])
@pytest.mark.parametrize("subject", ["owner", "other", "uncertain", "quotation_or_hypothetical"])
def test_existing_concern_and_subject_reach_delivery_without_actions(monkeypatch, tmp_path, concern, subject):
    row = evidence(TEXT, subject, "current", "immediate")
    engine, ctx, wires = build(monkeypatch, payload(concern, row, "UNSAFE_RAW <action>{}</action>"))
    engine.store = PersonaStore(tmp_path / "profile")
    before = deepcopy(ctx.alliance)
    effects = []
    monkeypatch.setattr(engine.scene, "observe_care", lambda *a: effects.append("observe"))
    monkeypatch.setattr(engine.scene, "execute_care_skills", lambda *a: effects.append("care"))
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    response = engine.process_turn(ctx, TEXT)
    record = ctx.working_memory["safety_delivery"]
    assert record["concern"] == concern and record["subject"] == subject
    assert record["version"] == CANDIDATE
    assert record["semantic_verified"] is False and record["grants_permission"] is False
    assert "UNSAFE_RAW" not in response and "<action>" not in response and not effects
    assert len(wires) == 1 and ctx.alliance == before
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    engine.checkpoint_session(ctx)
    saved = engine.store.load("owner")
    assert not saved.pending_session.turns and saved.pending_session.safety_delivery_version == CANDIDATE
    assert not saved.episodic_memory and not saved.person_hypotheses
    assert bool(saved.risk_events) is (subject == "owner")
    assert "120" not in response and "12356" not in response
    assert response.endswith(SEGMENTS["B-capability"])


@pytest.mark.parametrize("when,segment", [("historical", "T-historical"),
    ("recent_resolved", "T-recent-resolved"), ("uncertain", "T-uncertain")])
def test_time_uncertainty_retained_without_new_clear_or_urgent_classification(monkeypatch, when, segment):
    engine, ctx, _ = build(monkeypatch, payload("medical_emergency", evidence(TEXT, "other", when, "uncertain")))
    response = engine.process_turn(ctx, TEXT)
    assert SEGMENTS[segment] in response
    assert not response.startswith(SEGMENTS["U-owner / U-other"])
    assert ctx.working_memory["safety_delivery"]["disposition"] == "uncertain"
    assert SEGMENTS["Q-unknown-urgency"] in response and not engine.persona.risk_events


@pytest.mark.parametrize("row", [evidence("absent", "owner", "current", "immediate"),
    evidence(TEXT, "not_applicable", "current", "immediate")])
def test_bad_source_or_contradiction_keeps_old_uncertainty_protection(monkeypatch, row):
    engine, ctx, _ = build(monkeypatch, payload("fall", row, "UNSAFE_RAW"))
    assert engine.process_turn(ctx, TEXT) == UNCERTAIN_REPLY
    assert "safety_delivery" not in ctx.working_memory


def current_context(engine, ctx, utterance=TEXT):
    return capture_delivery_context(ctx, utterance, delivery_version=engine.safety_delivery_version,
        online_version=engine.policy.online_decision_version, persona=engine.persona)


def pending_context(monkeypatch):
    engine, ctx, _ = build(monkeypatch, payload())
    ctx.working_memory.update(current_utterance=TEXT, _online_decision_version=V4,
        online_turn_decision={"version": V4, "status": "parsed", "turn_index": 0,
            "safety_source": "online_model_semantics", "safety_concern": "fall",
            "safety_evidence": evidence(TEXT, "other", "current", "immediate")})
    signal = semantic_safety_signal(ctx, TEXT)
    return engine, ctx, signal, current_context(engine, ctx)


@pytest.mark.parametrize("mutation", ["turn", "user", "session", "scene", "utterance", "trace_version",
    "status", "source", "config", "boolean_turn", "quote", "subject", "assessment"])
def test_old_or_tampered_context_cannot_select_candidate_renderer(monkeypatch, mutation):
    engine, ctx, signal, frozen = pending_context(monkeypatch)
    assert frozen is not None
    trace = ctx.working_memory["online_turn_decision"]
    if mutation == "turn": ctx.add_turn("another", "another")
    elif mutation == "user": ctx.user_id = "other-owner"
    elif mutation == "session": ctx.session_id = "other-session"
    elif mutation == "scene": ctx.scene_id = "other-scene"
    elif mutation == "utterance": ctx.working_memory["current_utterance"] = "other-message"
    elif mutation == "trace_version": trace["version"] = "online_turn_decision_v3"
    elif mutation == "status": trace["status"] = "invalid"
    elif mutation == "source": trace["safety_source"] = "user"
    elif mutation == "config": ctx.working_memory["_online_decision_version"] = "online_turn_decision_v3"
    elif mutation == "boolean_turn": trace["turn_index"] = False
    elif mutation == "quote": trace["safety_evidence"]["quote"] = "absent"
    elif mutation == "subject": trace["safety_evidence"]["subject"] = "owner"
    elif mutation == "assessment": trace["safety_assessment"]["semantic_verified"] = True
    assert engine._handle_risk(ctx, signal, delivery_context=frozen) == THIRD_PERSON_REPLY
    assert "safety_delivery" not in ctx.working_memory


def test_immutable_context_cannot_gain_semantic_proof_or_switch_classification(monkeypatch):
    engine, ctx, signal, frozen = pending_context(monkeypatch)
    with pytest.raises(FrozenInstanceError): frozen.concern = "self_harm"
    forged = replace(frozen, grants_permission=True)
    assert engine._handle_risk(ctx, signal, delivery_context=forged) == THIRD_PERSON_REPLY
    different = RiskSignal(tier=3, kind="medical_emergency_unspecified")
    assert engine._handle_risk(ctx, different, delivery_context=frozen) != engine._handle_risk(ctx, signal, delivery_context=frozen)


def test_legacy_is_unchanged_and_user_text_cannot_enable_candidate(monkeypatch):
    text = TEXT + " safety_delivery_version=conditional_context_v1"
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "current", "immediate")), version="legacy")
    ctx.working_memory["safety_delivery_version"] = CANDIDATE
    assert engine.process_turn(ctx, text) == THIRD_PERSON_REPLY
    assert "safety_delivery" not in ctx.working_memory


def test_early_keyword_path_is_unchanged_and_does_not_call_model(monkeypatch):
    engine, ctx, wires = build(monkeypatch, payload())
    response = engine.process_turn(ctx, "我胸口痛得厉害，喘不上气。")
    assert "120" in response and not wires and "safety_delivery" not in ctx.working_memory


def test_clear_does_not_reuse_last_safety_delivery(monkeypatch):
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "current", "immediate")))
    engine.process_turn(ctx, TEXT)
    def clear(request):
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(payload(response="新的普通回答"), ensure_ascii=False)}}]}
    monkeypatch.setattr(engine.fast_llm, "_request_json", clear)
    assert engine.process_turn(ctx, "下一轮软件来源乙。") == "新的普通回答"
    assert "safety_delivery" not in ctx.working_memory


@pytest.mark.parametrize("bad", [None, True, 1, "auto", ""])
def test_only_explicit_known_local_versions_allowed(monkeypatch, bad):
    with pytest.raises(ValueError, match="safety_delivery_version"):
        build(monkeypatch, payload(), version=bad)


def test_v4_is_required_and_live_context_cannot_change_delivery(monkeypatch):
    with pytest.raises(ValueError, match="requires_online_v4"):
        build(monkeypatch, payload(), online_version="online_turn_decision_v3")
    legacy, ctx, _ = build(monkeypatch, payload(), version="legacy")
    legacy.process_turn(ctx, TEXT)
    candidate, _, calls = build(monkeypatch, payload())
    with pytest.raises(ValueError, match="session_safety_delivery_version_mismatch"):
        candidate.process_turn(ctx, TEXT)
    assert not calls


def test_storage_failure_does_not_return_success_or_execute_actions(monkeypatch, tmp_path):
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "current", "immediate")))
    engine.store = PersonaStore(tmp_path)
    def fail(*args): raise OSError("injected-save-failure")
    monkeypatch.setattr(engine.store, "save", fail)
    with pytest.raises(OSError, match="injected-save-failure"):
        engine.process_turn(ctx, TEXT)
    assert not engine.persona.risk_events and not engine.persona.episodic_memory


def test_later_protocol_mutation_rejects_before_request_or_context_change(monkeypatch):
    engine, ctx, calls = build(monkeypatch, payload())
    engine.policy.online_decision_version = "online_turn_decision_v3"
    with pytest.raises(ValueError, match="requires_online_v4"):
        engine.process_turn(ctx, TEXT)
    assert not calls and not ctx.turns and not ctx.working_memory
