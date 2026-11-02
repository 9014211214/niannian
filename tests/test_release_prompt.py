"""Release prompt/transport software checks, not model-effect measurements."""
from copy import deepcopy
import hashlib
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy import PolicyEngine
from aria.engine.policy.online_decision import online_decision_tool_spec
from aria.engine.policy.online_json import prepare_json_messages, SCHEMA_MARKER
from aria.engine.policy.release_prompt import INSTRUCTION, SYSTEM_BYTE_LIMIT
from aria.scenes.registry import get_scene
from tests.test_preference_required_control import client, envelope
from tests.test_preference_dialogue_readiness import fixture, control
from tests.test_safety_evidence import payload, V4
from tests.test_stated_preferences import _profile


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def capture(monkeypatch, *, invited=False, profile=None, fmt="json_schema_v1", current="中性软件输入。", extra=None):
    persona, ctx = fixture() if invited else (
        profile or PersonaProfile(user_id="owner"), EngineContext(user_id=(profile.user_id if profile else "owner"), scene_id="companion_elder"))
    if not invited:
        ctx.add_turn("原话甲\n第二行", "旧实际回复甲")
        ctx.add_turn("原话乙  保留空格", "旧实际回复乙")
    if extra:
        ctx.working_memory.update(extra)
    llm = client(fmt)
    llm.online_decision_version = V4
    value = payload(response="  软件首答第一行。\n第二行。  ")
    if invited:
        value["preference_control"] = control(persona, ctx, text=current, action="unclear")
    wires = []
    def receive(request, **options):
        wire = json.loads(request.data)
        from aria.engine.policy.minimal_calls import MinimalCallGuard
        MinimalCallGuard._validate_payload(llm, wire, **options)
        wires.append((wire, options))
        return envelope(value, fmt)
    monkeypatch.setattr(llm, "_request_json", receive)
    policy = PolicyEngine(fast_llm=llm, online_decision_version=V4,
                          safety_response_instruction_version="compact_response_v1")
    result = policy.respond(ctx, get_scene("companion_elder"), current, persona)
    return result, value, persona, ctx, wires


@pytest.mark.parametrize("invited", [False, True])
def test_actual_wire_budget_full_native_schema_exact_history_and_body(monkeypatch, invited):
    current = "  当前原话\n保留全部空白。  "
    result, value, persona, ctx, wires = capture(monkeypatch, invited=invited, current=current)
    (wire, options), = wires
    system = wire["messages"][0]["content"]
    assert len(system.encode()) <= SYSTEM_BYTE_LIMIT
    assert system.count(INSTRUCTION) == 1
    assert SCHEMA_MARKER not in system
    assert "嵌套字段逐项约束" not in system
    assert wire["response_format"]["json_schema"]["schema"] == online_decision_tool_spec(
        V4, require_preference_control=invited)["function"]["parameters"]
    data = json.loads(wire["messages"][1]["content"])
    assert data["current_user_message"] == current
    expected = [] if invited else [
        {"role": "user", "content": "原话甲\n第二行"}, {"role": "assistant", "content": "旧实际回复甲"},
        {"role": "user", "content": "原话乙  保留空格"}, {"role": "assistant", "content": "旧实际回复乙"}]
    assert data["conversation_history"] == expected
    assert result == value["response"]
    assert ctx.working_memory["online_turn_decision"]["parsed_response_sha256"] == hashlib.sha256(result.encode()).hexdigest()
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"] is False
    assert options == ({"require_preference_control": True} if invited else {})
    if invited:
        assert json.dumps(ctx.working_memory["_preference_dialogue_invitation"]["question"], ensure_ascii=False) in system
        assert persona.stated_preferences[0].status == "proposed"


def test_confirmed_preference_scope_and_provenance_are_separate(monkeypatch):
    persona = _profile()
    _, _, _, ctx, wires = capture(monkeypatch, profile=persona)
    system = wires[0][0]["messages"][0]["content"]
    row = persona.stated_preferences[0]
    assert row.do in system and row.scope in system
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["stated_preferences"][0]["quotes"]
    assert snapshot["stated_preferences_prompt_projection"][0]["scope"] == row.scope
    assert not snapshot["preference_invitation_exposed"]
    assert "本人页面已实际展示" not in system


def test_context_over_budget_stops_before_wire_and_does_not_truncate(monkeypatch):
    huge = {f"field_{i}": "中" * 500 for i in range(30)}
    with pytest.raises(ValueError, match="compact_online_system_byte_limit_exceeded"):
        capture(monkeypatch, extra={"minimal_care_context": huge})
    assert all(len(v) == 500 for v in huge.values())


@pytest.mark.parametrize("fmt", ["json_object", "native"])
def test_compact_cannot_remove_schema_without_native_enforcement(monkeypatch, fmt):
    _, _, _, _, wires = capture(monkeypatch, fmt=fmt)
    assert wires == []


def test_compact_carrier_rebinding_is_exact_and_preserves_marker_in_user(monkeypatch):
    _, _, _, _, wires = capture(monkeypatch, current=INSTRUCTION)
    original = wires[0][0]["messages"]
    assert prepare_json_messages(deepcopy(original), V4, compact_schema=True) == original
    assert json.loads(original[1]["content"])["current_user_message"] == INSTRUCTION


def test_dormant_optional_scene_rules_are_not_present(monkeypatch):
    _, _, _, _, wires = capture(monkeypatch)
    system = wires[0][0]["messages"][0]["content"]
    assert "护理记录是读取时点快照" not in system
    assert "可以温和提起的观察" not in system
    assert "外部情绪读数可能错误" not in system
