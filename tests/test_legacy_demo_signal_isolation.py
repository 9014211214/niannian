"""Legacy story observations must never become real personal evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import BaselineStat, PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.perception.signals import is_simulated
from aria.engine.policy.llm import MockLLMClient
from aria.scenes.registry import get_scene


class _UnusedModel(MockLLMClient):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        raise AssertionError("real model must not consume a demo observation")


class _UnusedStore:
    def __init__(self):
        self.calls = 0

    def save(self, persona):
        self.calls += 1
        raise AssertionError("demo observation must not touch real storage")


@pytest.mark.parametrize("source", [
    "synthetic_demo_timeline", "synthetic_demo_story", "simulated_acoustic",
])
def test_explicit_demo_source_namespaces_are_recognised_recursively(source):
    assert is_simulated({"source": source})
    assert is_simulated({"behavioral": [{"source": source}]})


@pytest.mark.parametrize("source", [
    "synthetic_model_output", "synthetic_speech_provider", "synthetic_demo",
    "local_wav_prosody", "emotion2vec",
])
def test_unrelated_sources_are_not_reclassified_as_simulation(source):
    assert not is_simulated({"source": source})
    assert not is_simulated({"behavioral": [{"source": source}]})


@pytest.mark.parametrize("channel", ["multimodal", "acoustic", "text_emotion"])
def test_real_engine_rejects_legacy_demo_before_any_state_or_service_change(channel):
    persona = PersonaProfile(user_id="real-control")
    persona.longitudinal_baseline.signals["behavioral:social_participation"] = (
        BaselineStat(count=8, mean=.7, m2=.08)
    )
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    ctx.working_memory["_current_turn_id"] = "real-turn"
    llm = _UnusedModel()
    store = _UnusedStore()
    engine = Engine(
        get_scene("companion_elder"), persona=persona, store=store,
        fast_llm_client=llm,
    )
    row = {
        "source": "synthetic_demo_timeline", "value": .2,
        "baseline_delta": -.65, "confidence": .88, "quality": .95,
        "label": "sad", "score": .88,
    }
    payload = (
        {"behavioral": {"social_participation": row}}
        if channel == "multimodal" else row
    )
    original_payload = deepcopy(payload)
    original_persona = persona.model_dump(mode="json")
    original_context = asdict(ctx)

    with pytest.raises(ValueError, match="isolated simulation engine"):
        engine.process_turn(ctx, "我最近不太想出门。", **{channel: payload})

    assert llm.calls == store.calls == 0
    assert persona.model_dump(mode="json") == original_persona
    assert asdict(ctx) == original_context
    assert payload == original_payload


def test_bound_legacy_story_signal_uses_simulated_baseline_in_isolated_engine():
    persona = PersonaProfile(user_id="fictional", profile_origin="synthetic_demo")
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    ctx.working_memory["_current_turn_id"] = "fictional-turn"
    engine = Engine(
        get_scene("companion_elder"), persona=persona, simulation_mode=True,
        fast_llm_client=MockLLMClient(),
    )
    engine.process_turn(ctx, "我最近不太想出门。", multimodal={
        "behavioral": {"social_participation": {
            "source": "synthetic_demo_timeline", "value": .2,
            "confidence": .88, "quality": .95,
            "turn_id": "fictional-turn", "turn_index": 0,
        }},
    })

    assert len(ctx.turns) == 1
    assert engine.persona.longitudinal_baseline.signals[
        "simulated:behavioral:social_participation"
    ].count == 1
    assert "behavioral:social_participation" not in engine.persona.longitudinal_baseline.signals
    assert persona.longitudinal_baseline.signals == {}  # Original remains untouched.
