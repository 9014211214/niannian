"""Independent boundary probes from the acoustic simulation review.

All models/stores are in-memory doubles. These tests exercise isolation and
reliability rejection, not emotion-recognition or companion-response quality.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import BaselineStat, PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.perception import PerceptionEngine
from aria.engine.perception.fusion import fuse
from aria.engine.perception.gating import route_state_analysis
from aria.engine.perception.longitudinal import enrich_with_personal_baseline
from aria.engine.perception.signals import prepare_acoustic, prepare_multimodal
from tests._signal_fixtures import build_simulated_acoustic
from aria.engine.policy.llm import MockLLMClient
from aria.scenes.registry import get_scene


class _MemoryOnlyStore:
    def __init__(self):
        self.saved = []

    def save(self, persona):
        self.saved.append(deepcopy(persona))


@pytest.mark.parametrize("entry", [
    "process", "checkpoint", "finalize", "persist", "pending_action",
])
def test_simulated_context_cannot_be_reused_by_real_persistence_or_generation(entry):
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id="synthetic-subject", scene_id=scene.id)
    ctx.working_memory["_current_turn_id"] = "sensor-bound"
    simulated = Engine(scene, simulation_mode=True, fast_llm_client=MockLLMClient())
    simulated.process_turn(ctx, "我刚刚看了一会儿书。", acoustic=build_simulated_acoustic(
        "sad", turn_id="sensor-bound", turn_index=0,
    ))
    assert ctx.simulation_mode is True
    original_turns = deepcopy(ctx.turns)
    store = _MemoryOnlyStore()
    real = Engine(
        scene, persona=PersonaProfile(user_id=ctx.user_id), store=store,
        fast_llm_client=MockLLMClient(),
    )
    entries = {
        "process": lambda: real.process_turn(ctx, "你好"),
        "checkpoint": lambda: real.checkpoint_session(ctx),
        "finalize": lambda: real.finalize_session(ctx),
        "persist": lambda: real._persist_runtime_state(ctx),
        "pending_action": lambda: real.stage_pending_action(ctx, "medication_reminder", {}),
    }
    with pytest.raises(ValueError, match="synthetic session"):
        entries[entry]()
    assert store.saved == []
    assert ctx.turns == original_turns
    assert real.persona.pending_session is None


def test_simulation_never_registers_scene_skills_or_tools():
    class SceneProxy:
        def __init__(self):
            self.base = get_scene("companion_elder")
            self.tool_registrations = 0
            self.skill_registrations = 0

        def __getattr__(self, name):
            return getattr(self.base, name)

        def tools(self):
            self.tool_registrations += 1
            return self.base.tools()

        def skills(self):
            self.skill_registrations += 1
            return self.base.skills()

    scene = SceneProxy()
    engine = Engine(scene, simulation_mode=True, fast_llm_client=MockLLMClient())
    assert scene.tool_registrations == scene.skill_registrations == 0
    assert engine._tools.describe() == []
    with pytest.raises(ValueError, match="unknown tool"):
        engine.stage_pending_action(
            EngineContext(user_id="synthetic", scene_id=scene.id),
            "medication_reminder", {},
        )


@pytest.mark.parametrize("extra", ["skills", "tools", "world_knowledge"])
def test_simulation_rejects_explicit_external_extensions(extra):
    value = [object()] if extra in {"skills", "tools"} else object()
    with pytest.raises(ValueError, match="cannot enable external"):
        Engine(get_scene("companion_elder"), simulation_mode=True,
               fast_llm_client=MockLLMClient(), **{extra: value})


def test_removed_extensions_are_not_available_to_the_first_release():
    engine = Engine(get_scene("companion_elder"), fast_llm_client=MockLLMClient())
    for name in ("maybe_initiate", "evaluate_initiation", "compose_initiation"):
        assert not hasattr(engine, name)
    with pytest.raises(TypeError, match="embedder"):
        Engine(get_scene("companion_elder"), fast_llm_client=MockLLMClient(),
               embedder=object())


@pytest.mark.parametrize("child", ["emotion_trace", "prosodic"])
@pytest.mark.parametrize("binding", [{"turn_index": 999}, {"turn_id": "previous"}])
def test_stale_nested_acoustic_observation_cannot_create_current_distress(child, binding):
    raw = {
        "label": "neutral", "score": .9, "quality": .9,
        "turn_id": "current", "turn_index": 0,
        "prosodic": {"confidence": .9, "quality": .9, "pause_ratio": .9},
        "emotion_trace": {
            "pattern": "late_distress", "trace": ["neutral", "sad"],
            "confidence": .9, "quality": .9,
        },
    }
    raw[child].update(binding)
    unchanged = deepcopy(raw)
    prepared = prepare_acoustic(raw, turn_index=0, turn_id="current")
    assert raw == unchanged
    assert any(reason.startswith(child + ":") for reason in prepared["rejection_reasons"])
    fused = fuse(prepared, None, prosodic=prepared["prosodic"], emotion_trace=prepared["emotion_trace"])
    assert fused.primary_label == "unknown"
    assert fused.divergence is False
    assert "长停顿" not in fused.divergence_pattern


@pytest.mark.parametrize("confidence", [0, .1])
def test_low_confidence_prosody_does_not_strengthen_an_independent_divergence(confidence):
    acoustic = {"label": "sad", "score": .9, "quality": .9}
    text = {"label": "neutral", "score": .9, "quality": .9}
    fused = fuse(acoustic, text, prosodic={
        "confidence": confidence, "quality": 1, "pause_ratio": .9,
        "volume_envelope": "decaying", "pitch_jitter": .9,
    })
    assert fused.divergence is True  # Reliable class disagreement still exists.
    assert "声学证据:" not in fused.divergence_pattern
    assert "长停顿" not in fused.divergence_pattern


@pytest.mark.parametrize("binding", [{"turn_index": 999}, {"turn_id": "previous"}])
def test_stale_text_emotion_is_rejected_in_the_actual_engine_path(binding):
    ctx = EngineContext(user_id="synthetic-text-control", scene_id="companion_elder")
    ctx.working_memory["_current_turn_id"] = "current"
    engine = Engine(get_scene("companion_elder"), fast_llm_client=MockLLMClient())
    engine.process_turn(ctx, "你好", text_emotion={
        "label": "sad", "score": .9, "quality": 1, **binding,
    })
    fused = ctx.working_memory["perception_fusion"]
    assert fused["label"] == "unknown"
    assert fused["score"] == 0
    assert fused["divergence"] is False


@pytest.mark.parametrize("delta", ["-inf", float("inf"), float("nan"), 10**1000])
def test_nonfinite_baseline_delta_cannot_route_or_update_a_personal_baseline(delta):
    prepared = prepare_multimodal({"behavioral": {"movement_speed": {
        "value": 1, "baseline_delta": delta, "confidence": 1, "quality": 1,
    }}}, turn_index=0)
    row = prepared["behavioral"]["movement_speed"]
    assert "baseline_delta" not in row
    assert row["quality"] == 0
    assert "invalid_baseline_delta" in row["rejection_reasons"]
    assert route_state_analysis("你好", multimodal=prepared).tier == "skip"
    persona = PersonaProfile(user_id="baseline-control")
    enrich_with_personal_baseline(persona, prepared)
    assert persona.longitudinal_baseline.signals == {}


def test_huge_integer_observation_cannot_crash_preparation_or_baseline_update():
    rows = {"behavioral": {"movement_speed": {
        "value": 10**1000, "quality": 1, "confidence": 1,
    }}}
    prepared = prepare_multimodal(rows, turn_index=0)
    assert prepared["behavioral"]["movement_speed"]["value"] is None
    assert prepared["behavioral"]["movement_speed"]["quality"] == 0
    persona = PersonaProfile(user_id="baseline-control")
    enrich_with_personal_baseline(persona, prepared)
    enrich_with_personal_baseline(persona, rows)  # Legacy direct utility call.
    assert persona.longitudinal_baseline.signals == {}


@pytest.mark.parametrize("acoustic", [None, {"label": "unknown", "score": .9, "quality": .9},
                                      {"label": "sad", "score": .9, "quality": 0}])
def test_missing_unknown_or_rejected_acoustic_is_not_observed_calm(acoustic):
    state = PerceptionEngine().perceive("你好", acoustic=acoustic)
    assert state.label == "unknown"
    assert state.score == 0
    assert state.divergence is False


def test_genuine_neutral_observation_remains_distinct_from_missing_input():
    state = PerceptionEngine().perceive("你好", acoustic={
        "label": "neutral", "score": .9, "quality": .9,
    })
    assert state.label == "neutral"
    assert state.score == pytest.approx(.81)


def test_finite_extreme_sequence_cannot_commit_overflowed_baseline_statistics():
    persona = PersonaProfile(user_id="finite-extreme-sequence")
    first = prepare_multimodal({"behavioral": {"movement_speed": {
        "value": 1e308, "quality": 1, "confidence": 1,
    }}}, turn_index=0)
    enrich_with_personal_baseline(persona, first)
    stat = persona.longitudinal_baseline.signals["behavioral:movement_speed"]
    original = stat.model_dump()
    assert stat.count == 1
    for index, value in enumerate((-1e308, 1.0, 1.0), start=1):
        prepared = prepare_multimodal({"behavioral": {"movement_speed": {
            "value": value, "quality": 1, "confidence": 1,
            "baseline_delta": -.7,
        }}}, turn_index=index)
        enriched = enrich_with_personal_baseline(persona, prepared)
        row = enriched["behavioral"]["movement_speed"]
        assert row["quality"] == 0
        assert "baseline_delta" not in row
        assert stat.model_dump() == original  # Count/timestamp cannot partly commit.
        assert all(math.isfinite(value) for value in (stat.mean, stat.m2))
        assert route_state_analysis("你好", multimodal=enriched).tier == "skip"


@pytest.mark.parametrize(("mean", "m2"), [
    (float("inf"), 0.0), (0.0, float("nan")), (0.0, -1.0),
])
def test_existing_invalid_baseline_is_not_reused_or_silently_repaired(mean, m2):
    persona = PersonaProfile(user_id="invalid-existing-statistic")
    stat = BaselineStat(count=4, mean=mean, m2=m2, updated_at="previous")
    persona.longitudinal_baseline.signals["behavioral:movement_speed"] = stat
    original = repr(stat)
    prepared = prepare_multimodal({"behavioral": {"movement_speed": {
        "value": 1.0, "quality": 1, "confidence": 1, "baseline_delta": -.7,
    }}}, turn_index=0)
    enriched = enrich_with_personal_baseline(persona, prepared)
    row = enriched["behavioral"]["movement_speed"]
    assert row["quality"] == 0
    assert "baseline_delta" not in row
    assert repr(stat) == original
    assert route_state_analysis("你好", multimodal=enriched).tier == "skip"
