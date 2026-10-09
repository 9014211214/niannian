"""Minimal runtime assembly checks; all outputs are local software fixtures."""
from __future__ import annotations
from tests._model_http_stub import stub_model_http

from copy import deepcopy
import urllib.error
import urllib.request

import pytest

import aria.runtime as runtime
from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.llm import MockLLMClient, OpenRouterLLMClient
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.scenes.registry import get_scene


def _forbid_call(*args, **kwargs):
    raise AssertionError("the minimal preview must not initialize or request a service")


class _FixtureLLM:
    source_id = "local_test_fixture"
    model = "fixture"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return self.response


def test_preview_ignores_keys_and_factories_and_does_not_learn(tmp_path, monkeypatch):
    for name in (
        "OPENROUTER_API_KEY", "ARIA_LLM_API_KEY", "ARIA_EMBED_API_KEY",
        "DASHSCOPE_API_KEY", "ARIA_VISUAL_API_KEY", "VOLC_TTS_API_KEY",
    ):
        monkeypatch.setenv(name, "not-a-real-key")
    monkeypatch.setenv("ARIA_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ARIA_OMNI_REALTIME_PROVIDER", "bailian")
    monkeypatch.setattr(runtime, "openrouter_api_key", _forbid_call)
    monkeypatch.setattr(runtime, "OpenRouterLLMClient", _forbid_call)
    stub_model_http(monkeypatch, _forbid_call)
    clients = runtime.create_minimal_clients()
    assert not clients.model_calls_enabled
    assert all(isinstance(client, MockLLMClient) for client in (
        clients.online, clients.memory, clients.reflection,
    ))
    persona = PersonaProfile(user_id="minimal-preview")
    engine = runtime.create_minimal_engine(
        get_scene("companion_elder"), persona,
        clients=clients, store=PersonaStore(tmp_path),
    )
    assert not hasattr(engine, "embedder")
    assert engine.state_llm is None
    assert engine.hypothesis_updater is None
    assert engine._world_knowledge is None
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    response = engine.process_turn(ctx, "这纸风车翅膀折反了，怪不得不转。")
    assert "本地流程示意" in response
    assert "没有调用模型" in response
    engine.seal_session(ctx)
    engine.finalize_session(ctx)
    assert not persona.episodic_memory
    assert not persona.person_hypotheses
    assert not persona.relationship_context
    stored = PersonaStore(tmp_path).load(persona.user_id)
    assert stored is not None and not stored.episodic_memory
    assert not stored.person_hypotheses


@pytest.mark.parametrize("value", ["false", "true", 1, None])
def test_permission_flag_cannot_be_accidentally_truthy(value):
    with pytest.raises(TypeError, match="explicit boolean"):
        runtime.create_minimal_clients(model_calls_enabled=value)


def test_minimal_scene_keeps_safety_without_legacy_care_mutation(monkeypatch):
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_URL", "https://example.invalid/care")
    stub_model_http(monkeypatch, _forbid_call)
    persona = PersonaProfile(user_id="local-care")
    engine = runtime.create_minimal_engine(get_scene("companion_elder"), persona)
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    response = engine.process_turn(ctx, "我现在胸痛，还有点喘不上气。")
    assert "120" in response
    assert not persona.care_tasks and not persona.reminder_rules
    assert engine.scene.skills() == [] and engine.scene.tools() == []


def test_disabled_bundle_rejects_a_real_client():
    with pytest.raises(ValueError, match="only accepts local mock"):
        runtime.MinimalModelClients(
            online=_FixtureLLM("test"),
            memory=MockLLMClient("memory"),
            reflection=MockLLMClient("reflection"),
        )


def test_explicit_live_setup_requires_key_and_never_silently_uses_mock():
    with pytest.raises(ValueError, match="API key"):
        runtime.create_minimal_clients(model_calls_enabled=True, api_key="")


def test_live_setup_is_inert_and_retries_and_fallbacks_stay_disabled(monkeypatch):
    stub_model_http(monkeypatch, _forbid_call)
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "some/other-model")
    monkeypatch.setenv("ARIA_LLM_MAX_RETRIES", "9")
    monkeypatch.setenv("OPENROUTER_ONLINE_FAST_MODEL", "unrequested/override")
    for role in ("ONLINE_FAST", "MEMORY", "REFLECTION"):
        monkeypatch.setenv(f"OPENROUTER_{role}_MAX_RETRIES", "7")
        monkeypatch.setenv(f"OPENROUTER_{role}_FALLBACK_MODELS", "fallback/model")
    clients = runtime.create_minimal_clients(
        model_calls_enabled=True, api_key="not-a-real-key",
    )
    assert clients.model_calls_enabled
    assert [client.model for client in (
        clients.online, clients.memory, clients.reflection,
    )] == [
        "qwen/qwen3.7-flash", "anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5",
    ]
    for client in (clients.online, clients.memory, clients.reflection):
        assert isinstance(client, OpenRouterLLMClient)
        assert client.max_retries == 0
        assert client.fallback_models == []
        assert client.network_request_count == 0


def test_live_failure_attempts_once_and_does_not_substitute_story(monkeypatch):
    attempts = []

    def fail(request, **kwargs):
        attempts.append(request)
        raise urllib.error.URLError("local fixture failure")

    stub_model_http(monkeypatch, fail)
    clients = runtime.create_minimal_clients(
        model_calls_enabled=True, api_key="not-a-real-key",
    )
    engine = runtime.create_minimal_engine(
        get_scene("companion_elder"), PersonaProfile(user_id="failure"),
        clients=clients,
    )
    ctx = EngineContext(user_id="failure", scene_id="companion_elder")
    response = engine.process_turn(ctx, "这纸风车翅膀折反了，怪不得不转。")
    assert response == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert len(attempts) == 1
    assert clients.memory.network_request_count == 0
    assert clients.reflection.network_request_count == 0
    assert "本地流程示意" not in response


def test_online_and_sealing_do_not_call_offline_roles(tmp_path, monkeypatch):
    stub_model_http(monkeypatch, _forbid_call)
    fast = _FixtureLLM("本地软件夹具回复，不是模型效果。")
    memory = _FixtureLLM('{"episodes":[],"relationships":[],"cognitive_privacy":[]}')
    reflection = _FixtureLLM('{"protocol_version":"2.0","reviews":[],"changes":[]}')
    clients = runtime.MinimalModelClients(fast, memory, reflection, True)
    persona = PersonaProfile(user_id="two-lines")
    engine = runtime.create_minimal_engine(
        get_scene("companion_elder"), persona,
        clients=clients, store=PersonaStore(tmp_path),
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    inputs = ("这纸风车有一片折反了。", "我把纸风车放在桌上。", "窗外还没有起风。")
    for utterance in inputs:
        engine.process_turn(ctx, utterance)
        engine.checkpoint_session(ctx)
    assert len(fast.calls) == 3
    assert not memory.calls and not reflection.calls
    engine.seal_session(ctx)
    assert not memory.calls and not reflection.calls
    assert engine.offline_status()["pending_count"] == 1
    engine.recover_offline_sessions(max_jobs=1)
    assert len(memory.calls) == 1
    assert len(reflection.calls) == 1
    assert all(utterance in str(reflection.calls) for utterance in inputs)
    assert not persona.person_hypotheses
    engine.recover_offline_sessions(max_jobs=1)
    assert len(memory.calls) == 1 and len(reflection.calls) == 1
