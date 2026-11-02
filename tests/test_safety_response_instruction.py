"""Instruction assembly and state boundaries, not semantic or clinical accuracy."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
from threading import Barrier

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy import PolicyEngine
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.online_decision import online_decision_tool_spec, protocol_version
from aria.engine.policy.safety_instruction import (
    SafetyResponseInstruction, CONDITIONAL_RESPONSE_V1 as CANDIDATE,
    SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION_SHA256,
    LEGACY_SCENE_RULE, CANDIDATE_SCENE_RULE,
)
from aria.runtime import MinimalModelClients, create_minimal_engine
from aria.scenes.registry import get_scene
from tests._model_http_stub import stub_model_http
from tests.test_preference_required_control import client, envelope
from tests.test_preference_dialogue_readiness import fixture, control
from tests.test_safety_evidence import payload, evidence, V4

TEXT = "本次软件来源文本。"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected external call"))


def bundle(fmt="json_schema_v1"):
    c = client(fmt)
    c.online_decision_version = V4
    if fmt == "native": c.bounded_transport = False
    return MinimalModelClients(c, MockLLMClient("memory"), MockLLMClient("reflection"), True)


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
@pytest.mark.parametrize("ack", [False, True])
def test_full_block_at_actual_request_with_original_schema_and_ack(monkeypatch, fmt, ack):
    p, ctx = fixture() if ack else (PersonaProfile(user_id="owner"), EngineContext(user_id="owner", scene_id="companion_elder"))
    c = bundle(fmt).online
    value = payload(response="软件原答")
    if ack: value["preference_control"] = control(p, ctx, action="unclear")
    wires, options = [], []
    def receive(request, **kwargs):
        wire = json.loads(request.data); wires.append(wire); options.append(kwargs)
        from aria.engine.policy.minimal_calls import MinimalCallGuard
        MinimalCallGuard._validate_payload(c, wire, **kwargs)
        return envelope(value, fmt)
    monkeypatch.setattr(c, "_request_json", receive)
    policy = PolicyEngine(fast_llm=c, online_decision_version=V4, safety_response_instruction_version=CANDIDATE)
    policy.respond(ctx, get_scene("companion_elder"), TEXT, p)
    assert len(wires) == 1 and protocol_version(wires[0]["messages"]) == V4
    system = wires[0]["messages"][0]["content"]
    assert system.count(SYSTEM_INSTRUCTION) == 1
    assert "\n- " + CANDIDATE_SCENE_RULE + "\n" in system
    assert "\n- " + LEGACY_SCENE_RULE + "\n" not in system
    assert hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest() == SYSTEM_INSTRUCTION_SHA256
    assert options == ([{"require_preference_control": True}] if ack else [{}])
    schema = online_decision_tool_spec(V4, require_preference_control=ack)["function"]["parameters"]
    if fmt == "native": assert wires[0]["tools"][0]["function"]["parameters"] == schema
    if fmt == "json_schema_v1": assert wires[0]["response_format"]["json_schema"]["schema"] == schema
    assert ctx.working_memory["online_turn_decision"]["status"] == "parsed"
    audit = ctx.working_memory["safety_response_instruction"]
    assert audit["passed_to_client"] is True and audit["actual_send_verified"] is False
    assert audit["semantic_verified"] is False and audit["grants_permission"] is False
    assert audit["system_block_sha256"] == SYSTEM_INSTRUCTION_SHA256


@pytest.mark.parametrize("version", [None, True, 1, "", "user-supplied"])
def test_invalid_config_rejected(version):
    with pytest.raises(ValueError, match="invalid_safety_response_instruction_version"):
        SafetyResponseInstruction(version)


def test_immutable_config_protocol_binding_and_context_switch_rejected(monkeypatch):
    c = bundle().online
    policy = PolicyEngine(fast_llm=c, safety_response_instruction_version=CANDIDATE)
    with pytest.raises(AttributeError): policy.safety_response_instruction_version = "legacy"
    with pytest.raises(FrozenInstanceError): policy._safety_response_instruction.version = "legacy"
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"), clients=bundle(), safety_response_instruction_version=CANDIDATE)
    with pytest.raises(AttributeError): engine.safety_response_instruction_version = "legacy"
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_safety_response_instruction_version"] = "legacy"
    with pytest.raises(ValueError, match="session_safety_response_instruction_version_mismatch"):
        engine.process_turn(ctx, TEXT)
    assert not ctx.turns
    ctx.working_memory.clear(); engine.policy.online_decision_version = "online_turn_decision_v3"
    with pytest.raises(ValueError, match="requires_online_v4"): engine.process_turn(ctx, TEXT)
    assert not ctx.turns
    with pytest.raises(ValueError, match="requires_online_v4"):
        PolicyEngine(online_decision_version="online_turn_decision_v3", safety_response_instruction_version=CANDIDATE)


@pytest.mark.parametrize("mutation", ["remove", "duplicate", "alter"])
def test_whole_system_block_tamper_refused_before_client(monkeypatch, mutation):
    import aria.engine.policy as module
    original = module.build_messages
    def wrong(*args, **kwargs):
        messages = original(*args, **kwargs)
        if mutation == "remove": messages[0]["content"] = messages[0]["content"].replace(SYSTEM_INSTRUCTION, "")
        elif mutation == "duplicate": messages[0]["content"] += SYSTEM_INSTRUCTION
        else: messages[0]["content"] = messages[0]["content"].replace(SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION.replace("本段是回应约束", "篡改正文"))
        return messages
    monkeypatch.setattr(module, "build_messages", wrong)
    c = bundle().online
    monkeypatch.setattr(c, "complete", lambda *a, **k: pytest.fail("must not reach client"))
    policy = PolicyEngine(fast_llm=c, safety_response_instruction_version=CANDIDATE)
    with pytest.raises(ValueError, match="not_bound_once"):
        policy.respond(EngineContext(user_id="owner", scene_id="companion_elder"), get_scene("companion_elder"), TEXT)


def test_same_shared_client_keeps_two_request_configs_and_users_separate(monkeypatch):
    c = bundle().online
    barrier = Barrier(2)
    seen = []
    def complete(messages):
        barrier.wait(timeout=5)
        seen.append(messages)
        return json.dumps(payload(), ensure_ascii=False)
    monkeypatch.setattr(c, "complete", complete)
    def call(user, version):
        policy = PolicyEngine(fast_llm=c, safety_response_instruction_version=version)
        ctx = EngineContext(user_id=user, scene_id="companion_elder", session_id=user)
        policy.respond(ctx, get_scene("companion_elder"), user + " " + CANDIDATE, PersonaProfile(user_id=user))
        return ctx
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, "one", CANDIDATE), pool.submit(call, "two", "legacy")]
        first, second = [f.result() for f in futures]
    assert len(seen) == 2
    assert sum(SYSTEM_INSTRUCTION in m[0]["content"] for m in seen) == 1
    assert first.working_memory["safety_response_instruction"]["user_id"] == "one"
    assert "safety_response_instruction" not in second.working_memory
    assert not hasattr(c, "safety_response_instruction_version")


def test_exact_scene_rule_is_local_and_unknown_or_duplicate_is_rejected():
    from aria.interfaces.scene import Redline
    from aria.engine.policy.prompt import build_messages
    scene = get_scene("companion_elder")
    original = [r.rule for r in scene.safety_rules()]
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = V4
    build_messages(scene, None, ctx, TEXT, safety_response_instruction_version=CANDIDATE)
    assert [r.rule for r in scene.safety_rules()] == original
    legacy = build_messages(scene, None, ctx, TEXT)
    assert LEGACY_SCENE_RULE in legacy[0]["content"] and CANDIDATE_SCENE_RULE not in legacy[0]["content"]
    for rules in (["other rule"], [LEGACY_SCENE_RULE, LEGACY_SCENE_RULE]):
        class Other:
            def agent_persona(self): return scene.agent_persona()
            def safety_rules(self): return [Redline(r) for r in rules]
        with pytest.raises(ValueError, match="unsupported_safety_response_scene_rule"):
            build_messages(Other(), None, ctx, TEXT, safety_response_instruction_version=CANDIDATE)


def test_early_rule_and_same_turn_safety_rollback_are_unchanged(monkeypatch):
    clients = bundle(); requests = []
    def receive(request, **kwargs):
        requests.append(request.data)
        return envelope(payload("fall", evidence(TEXT, "other", "historical", "uncertain"), "UNSAFE_RAW"), "json_schema_v1")
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"), clients=clients,
        safety_response_instruction_version=CANDIDATE, safety_delivery_version="conditional_context_v1")
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    result = engine.process_turn(ctx, TEXT)
    assert "UNSAFE_RAW" not in result and len(requests) == 1
    assert not engine.persona.risk_events and not engine.persona.episodic_memory
    assert ctx.working_memory["safety_delivery"]["new_model_understanding_exposure"] is False
    result = engine.process_turn(ctx, "我现在想自杀")
    assert result and len(requests) == 1
    assert "safety_response_instruction" not in ctx.working_memory
