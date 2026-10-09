"""T1 switches use local recording clients; they do not measure model quality."""
from __future__ import annotations

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.orchestrator import Engine
from aria.engine.policy import PolicyEngine
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2, single_pass_messages,
)
from aria.engine.policy.prompt import (
    _MODE_GUIDANCE, build_messages,
    configured_delivery_block, confirmed_prompt_hypotheses, delivery_basis_ids, prompt_version_for,
    render_delivery_block, select_delivery_mode,
)
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp


@pytest.fixture(autouse=True)
def default_configuration(monkeypatch):
    monkeypatch.delenv("ARIA_DELIVERY_BLOCK", raising=False)
    monkeypatch.delenv("ARIA_ONLINE_DECISION_VERSION", raising=False)


@pytest.mark.parametrize("mode", list(_MODE_GUIDANCE))
def test_default_is_byte_exact_old_mode_text_for_reply(mode):
    persona = load_demo_persona("laochen_demo")
    persona.delivery_state.committed = mode
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id=persona.user_id, scene_id=scene.id)
    expected = _MODE_GUIDANCE[select_delivery_mode(persona)]
    assert render_delivery_block(persona).encode() == expected.encode()
    assert delivery_basis_ids(persona) == []
    messages = build_messages(scene, persona, ctx, "今天有点闷。")
    assert expected in messages[0]["content"]
    assert "【相处画像:" not in messages[0]["content"]


@pytest.mark.parametrize("value,expected,version", [
    ("mode_v1", "mode_v1", "1.18.0"),
    ("profile_v1", "profile_v1", "1.19.0"),
    ("unknown", "mode_v1", "1.18.0"),
    ("", "mode_v1", "1.18.0"),
])
def test_environment_and_invalid_explicit_values_resolve_safely(
    monkeypatch, value, expected, version,
):
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", value)
    assert configured_delivery_block() == expected
    assert prompt_version_for() == version
    assert PolicyEngine().delivery_block == expected
    assert PolicyEngine().online_decision_version == ONLINE_DECISION_VERSION
    assert configured_delivery_block(value) == expected
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "profile_v1")
    assert PolicyEngine(delivery_block="mode_v1").delivery_block == "mode_v1"
    assert PolicyEngine(delivery_block="invalid").delivery_block == "mode_v1"


class _RecordingClient(MockLLMClient):
    def __init__(self):
        super().__init__()
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return "本地固定测试回复。"


@pytest.mark.parametrize("block", ["mode_v1", "profile_v1"])
def test_engine_reply_keeps_instance_option_after_environment_changes(
    monkeypatch, block,
):
    persona = load_demo_persona("laochen_demo")
    scene = get_scene("companion_elder")
    client = _RecordingClient()
    engine = Engine(scene, persona=persona, llm_client=client, delivery_block=block)
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "profile_v1" if block == "mode_v1" else "mode_v1")
    ctx = EngineContext(user_id=persona.user_id, scene_id=scene.id)
    engine.policy.respond(ctx, scene, "今天有点闷。", persona)
    expected = render_delivery_block(persona, block=block)
    assert all(expected in messages[0]["content"] for messages in client.messages)
    assert len(client.messages) == 1
    assert engine.policy.delivery_block == block
    assert prompt_version_for(engine.policy.delivery_block) == {
        "mode_v1": "1.18.0", "profile_v1": "1.19.0",
    }[block]


def test_v2_cannot_cite_hidden_profile_axes_when_using_mode_block():
    persona = load_demo_persona("laochen_demo")
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id=persona.user_id, scene_id=scene.id)
    messages = build_messages(scene, persona, ctx, "今天有点闷。", delivery_block="mode_v1")
    result = single_pass_messages(
        messages, ctx, version=ONLINE_DECISION_VERSION_V2, persona=persona,
        delivery_block="mode_v1",
    )
    assert ctx.working_memory["_online_decision_basis_ids"] == [
        item.id for item in confirmed_prompt_hypotheses(persona)
    ]
    assert "相处画像维度 []" in result[0]["content"]
    assert "【相处画像:" not in result[0]["content"]


def test_minimal_web_capabilities_and_engine_use_same_frozen_block(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "profile_v1")
    app = MinimalWebApp(tmp_path)
    try:
        monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "mode_v1")
        architecture = app.capabilities()["reasoning_architecture"]
        assert architecture["delivery_block"] == "profile_v1"
        assert architecture["prompt_version"] == "1.19.0"
        assert architecture["online_decision_schema"] == ONLINE_DECISION_VERSION
        engine = app._engine_for_persona(app._seed_persona("fictional-chen"))
        assert engine.policy.delivery_block == "profile_v1"
    finally:
        app.close()


