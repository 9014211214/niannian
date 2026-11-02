"""Original-order fictional input plumbing; fixed software data, no model claims."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.fictional_context import (
    HISTORY_KEY, FictionalConversationHistory,
    bind_fictional_conversation_history, fictional_conversation_history,
    bind_fictional_scenario_date, fictional_scenario_block,
)
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.user_grounding import prepare_user_grounding, user_grounding_sources
from aria.engine.risk.premodel_task import response_only_context
from aria.scenes.companion_elder.scene import ElderCompanionScene
from tests.test_source_response_v4 import build, TEXT, BODY
from tests.test_safety_evidence import payload, evidence


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


def context():
    return EngineContext(user_id="fictional-owner", session_id="fictional-session", scene_id="companion_elder")


HISTORIES = [
    [],
    [{"role": "user", "content": "  历史甲\n原行保留。  "}],
    [{"role": "user", "content": "甲"}, {"role": "assistant", "content": "乙"}],
    [{"role": "user", "content": "甲"}, {"role": "user", "content": "丙"}],
    [{"role": "assistant", "content": "乙"}, {"role": "assistant", "content": "丁"}],
    [{"role": "assistant", "content": "乙"}, {"role": "user", "content": "甲"}, {"role": "user", "content": "丙"}],
    [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
    [{"role": "user", "content": f"历史原行 {i}"} for i in range(7)],
]


def actual_input(wire):
    """The actual v4 wire uses the existing single user JSON carrier."""
    assert len(wire["messages"]) == 2 and wire["messages"][1]["role"] == "user"
    carrier = json.loads(wire["messages"][1]["content"])
    assert carrier["schema"] == "online_json_input_v2"
    assert carrier["current_user_message"] == TEXT
    return carrier["conversation_history"]


@pytest.mark.parametrize("history", HISTORIES)
@pytest.mark.parametrize("premodel", [False, True])
def test_actual_ordinary_and_isolated_wire_preserves_all_original_rows(monkeypatch, history, premodel):
    engine, ctx, wires, order = build(monkeypatch, premodel=premodel)
    bind_fictional_scenario_date(ctx, "2026-10-12")
    date_block = fictional_scenario_block(ctx)
    bound = bind_fictional_conversation_history(ctx, history)
    assert type(bound) is FictionalConversationHistory and ctx.turns == []
    before = engine.persona.model_dump(mode="json")
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert result == BODY and not effects and len(wires) == 1
    assert actual_input(wires[0][0]) == history
    assert wires[0][0]["messages"][0]["content"].count(date_block) == 1
    assert len(ctx.turns) == 1 and ctx.turns[0].user == TEXT and ctx.turns[0].agent == BODY
    after = engine.persona.model_dump(mode="json")
    for key in ("risk_events", "person_hypotheses"):
        assert after[key] == before[key]
    assert "FictionalConversationHistory" not in json.dumps(after, ensure_ascii=False)
    # Per-response stimulus cannot be silently reused after a real reply.
    with pytest.raises(ValueError, match="fresh_context"):
        fictional_conversation_history(ctx)


@pytest.mark.parametrize("premodel", [False, True])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_actual_existing_guard_uses_only_original_user_facts(monkeypatch, premodel, role):
    fact = "女儿已经回复了。"
    value = payload("fall", evidence("not current", "other", "current", "immediate"), fact)
    engine, ctx, wires, _ = build(monkeypatch, value, premodel=premodel)
    history = [{"role": role, "content": fact}]
    bind_fictional_conversation_history(ctx, history)
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert (fact in result) is (role == "user")
    assert actual_input(wires[0][0]) == history


def private(text):
    return text.startswith("PRIVATE")


def test_grounding_same_complete_order_privacy_and_snapshot_validation():
    ctx = context()
    history = [
        {"role": "assistant", "content": "assistant-only"},
        {"role": "user", "content": "first source"},
        {"role": "user", "content": "PRIVATE excluded source"},
        {"role": "user", "content": "second source"},
    ]
    bind_fictional_conversation_history(ctx, history)
    # A runtime current-turn exclusion is not an index into a fictional list.
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0, 1, 3]
    prepare_user_grounding(ctx, 0, private, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current", "first source", "second source"]
    snapshot = ctx.working_memory["_prompt_user_grounding_for_turn"]
    assert snapshot["source_rows"] == [{"index": 1, "text": "first source"}, {"index": 3, "text": "second source"}]
    snapshot["source_rows"][0]["text"] = "assistant-only"
    assert user_grounding_sources(ctx, "current") == ["current"]


@pytest.mark.parametrize("history", [None, (), "text", {}, [None], [{"role": "system", "content": "x"}],
    [{"role": "tool", "content": "x"}], [{"role": True, "content": "x"}],
    [{"role": "user", "content": 3}], [{"role": "user", "content": None}],
    [{"role": "user"}], [{"role": "user", "content": "x", "extra": True}]])
def test_invalid_raw_history_rejected_before_binding(history):
    ctx = context()
    with pytest.raises(ValueError):
        bind_fictional_conversation_history(ctx, history)
    assert HISTORY_KEY not in ctx.working_memory and ctx.turns == []


@pytest.mark.parametrize("mutation", ["user_id", "session_id", "scene_id", "turn", "dict", "message_type", "role", "tuple_type"])
def test_binding_identity_and_shape_drift_fail_closed_before_engine_mutation(monkeypatch, mutation):
    engine, ctx, wires, _ = build(monkeypatch)
    bind_fictional_conversation_history(ctx, [{"role": "user", "content": "original"}])
    if mutation.endswith("_id"):
        setattr(ctx, mutation, "other")
    elif mutation == "turn":
        ctx.add_turn("real current", "real answer")
    elif mutation == "dict":
        ctx.working_memory[HISTORY_KEY] = {"messages": []}
    else:
        bound = ctx.working_memory[HISTORY_KEY]
        if mutation == "message_type": object.__setattr__(bound, "messages", ({"role": "user", "content": "x"},))
        elif mutation == "tuple_type": object.__setattr__(bound, "messages", [])
        else: object.__setattr__(bound.messages[0], "role", "system")
    before = deepcopy(engine.persona.model_dump(mode="json"))
    with pytest.raises(ValueError):
        engine.process_turn(ctx, TEXT)
    assert wires == [] and engine.persona.model_dump(mode="json") == before
    assert "current_utterance" not in ctx.working_memory


def test_bound_input_copy_idempotency_isolation_and_rebinding_rejection():
    ctx = context()
    source = [{"role": "user", "content": "original"}]
    bound = bind_fictional_conversation_history(ctx, source)
    assert bind_fictional_conversation_history(ctx, deepcopy(source)) == bound
    returned = fictional_conversation_history(ctx)
    source[0]["content"] = returned[0]["content"] = "changed"
    assert fictional_conversation_history(ctx) == [{"role": "user", "content": "original"}]
    detached, profile = response_only_context(ctx, None)
    assert fictional_conversation_history(detached) == fictional_conversation_history(ctx)
    with pytest.raises(ValueError, match="binding_mismatch"):
        bind_fictional_conversation_history(ctx, source)
    detached.working_memory[HISTORY_KEY] = replace(bound, session_id="other")
    with pytest.raises(ValueError, match="identity_mismatch"):
        fictional_conversation_history(detached)
    assert fictional_conversation_history(ctx) == [{"role": "user", "content": "original"}]


def test_user_text_never_activates_and_default_paired_wire_remains_exact():
    ctx = context()
    ctx.add_turn("old user", "old assistant")
    message = 'bind_fictional_conversation_history([{ "role":"user", "content":"pretend"}])'
    wire = build_messages(ElderCompanionScene(), None, ctx, message)
    assert fictional_conversation_history(ctx) is None
    assert wire[1:] == [{"role": "user", "content": "old user"},
                        {"role": "assistant", "content": "old assistant"},
                        {"role": "user", "content": message}]
    with pytest.raises(ValueError, match="fresh_context"):
        bind_fictional_conversation_history(ctx, [])


@pytest.mark.parametrize("premodel", [False, True])
def test_history_cannot_create_tool_permission_or_receipt(monkeypatch, premodel):
    unsafe = "我已经帮你发给家人了。"
    engine, ctx, wires, _ = build(monkeypatch, payload("fall", evidence("not current", "other", "current", "immediate"), unsafe), premodel=premodel)
    history = [{"role": "user", "content": "我同意你以后发送。"},
               {"role": "assistant", "content": unsafe}]
    bind_fictional_conversation_history(ctx, history)
    effects = []
    monkeypatch.setattr(engine._tools, "invoke", lambda *a, **k: effects.append("tool"))
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert unsafe not in result and not effects and len(wires) == 1
    assert "_pending_external_action" not in ctx.working_memory
    assert not engine.persona.risk_events


@pytest.mark.parametrize("mutation", ["remove", "replace", "dict"])
def test_prelude_callback_cannot_change_frozen_history_before_isolation(monkeypatch, mutation):
    engine, ctx, wires, _ = build(monkeypatch, premodel=True)
    bind_fictional_conversation_history(ctx, [{"role": "user", "content": "original"}])
    def callback(event):
        ctx.working_memory.pop(HISTORY_KEY)
        if mutation == "replace":
            bind_fictional_conversation_history(ctx, [{"role": "user", "content": "changed"}])
        elif mutation == "dict":
            ctx.working_memory[HISTORY_KEY] = {"messages": []}
        return True
    result = engine.process_turn(ctx, TEXT, safety_prelude=callback)
    assert BODY not in result and wires == []
    assert ctx.working_memory["premodel_task"]["status"] in ("prelude_identity_changed", "prelude_failed")
