"""Quality, provenance, isolation and current-turn contracts (no sensor/API)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.perception import PerceptionEngine, fuse
from aria.engine.perception.evidence import collect_evidence
from aria.engine.perception.gating import route_state_analysis
from aria.engine.perception.longitudinal import enrich_with_personal_baseline
from aria.engine.perception.signals import prepare_acoustic, prepare_multimodal
from tests._signal_fixtures import (
    build_simulated_acoustic, list_emotion_scenarios, simulated_prosodic_observations,
)
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("emotion contract tests cannot call a network service")
    monkeypatch.setattr("urllib.request.urlopen", denied)


def _observation(scenario="sad", turn=0):
    return build_simulated_acoustic(scenario, turn_id=f"turn-{turn}", turn_index=turn)


def _engine():
    profile = PersonaProfile(user_id="fiction", profile_origin="synthetic_demo")
    return Engine(get_scene("companion_elder"), persona=profile, simulation_mode=True)


@pytest.mark.parametrize("scenario", [item["id"] for item in list_emotion_scenarios()])
def test_every_selector_value_is_nonempty_deterministic_and_normalised(scenario):
    first = _observation(scenario)
    assert first == _observation(scenario)
    assert first["source_kind"] == "simulated"
    assert first["source"] != "emotion2vec"
    assert math.isclose(sum(first["all_scores"].values()), 1.0)
    assert first["score"] == first["all_scores"][first["label"]]
    assert not ({"scenario_id", "title", "expected_action", "gold", "observed_at"} & first.keys())
    clean = prepare_acoustic(first, turn_index=0, turn_id="turn-0")
    assert clean["rejection_reasons"] == []
    first["all_scores"]["sad"] = -1
    assert _observation(scenario)["all_scores"]["sad"] >= 0


@pytest.mark.parametrize("field", ["quality", "score"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 2, True, "broken", {}])
def test_invalid_reliability_does_not_create_distress_or_route_upgrade(field, bad):
    acoustic = {"label": "sad", "score": .9, "quality": .9, field: bad}
    text = {"label": "neutral", "score": .8}
    fused = fuse(acoustic, text)
    assert fused.acoustic_label == "unknown"
    assert fused.primary_label == "neutral"
    assert not fused.divergence
    assert math.isfinite(fused.primary_score)
    route = route_state_analysis("你好", acoustic=acoustic, text=text)
    assert "multimodal_conflict" not in route.reasons


def test_low_quality_unknown_is_not_reported_as_observed_calm():
    state = PerceptionEngine().perceive("今天下雨", _observation("low_quality"))
    assert state.label == "unknown"
    assert state.acoustic["acoustic_label"] == "unknown"
    assert state.acoustic["acoustic_quality"] == .08
    assert not state.divergence
    evidence = collect_evidence("今天下雨", state, turn_index=0)
    assert not any(item.modality == "acoustic" for item in evidence)


def test_valid_text_only_emotion_is_not_discarded():
    state = PerceptionEngine().perceive("我难过", text={"label": "sad", "score": .8})
    assert state.label == "sad" and state.score == .8
    assert state.inference_source == "provided_text_emotion"


@pytest.mark.parametrize("extra", [{}, {"confidence": .9}, {"confidence": .9, "quality": .1},
                                   {"confidence": float("nan"), "quality": .9}])
def test_unreliable_trace_cannot_force_divergence(extra):
    trace = {"pattern": "late_distress", "trace": ["neutral", "sad"], **extra}
    fused = fuse({"label": "neutral", "score": .9}, None, emotion_trace=trace)
    assert fused.primary_label == "neutral"
    assert not fused.divergence


def test_parent_quality_also_limits_reliable_segment_trace():
    acoustic = _observation("late_distress")
    acoustic["quality"] = .05
    fused = fuse(acoustic, None, emotion_trace=acoustic["emotion_trace"])
    assert not fused.divergence
    assert fused.primary_label == "unknown"


def test_acoustic_evidence_quality_is_applied_once_and_remains_simulated():
    raw = _observation()
    raw["quality"] = .6
    state = PerceptionEngine().perceive("今天下雨", raw)
    rows = collect_evidence("今天下雨", state, turn_index=0)
    evidence = next(item for item in rows if item.modality == "acoustic")
    assert evidence.confidence == .86 and evidence.quality == .6
    assert evidence.weight == pytest.approx(.516)
    assert state.score == pytest.approx(.516)
    assert evidence.source == "simulated_acoustic" and "合成" in evidence.detail


@pytest.mark.parametrize("update,reason", [
    ({"turn_index": 9}, "turn_index_mismatch"),
    ({"turn_id": "other"}, "turn_id_mismatch"),
    ({"expires_at": "2000-01-01T00:00:00+00:00"}, "expired"),
    ({"observed_at": "2000-01-01T00:00:00+00:00"}, "observation_not_current"),
    ({"observed_at": "2000-01-01"}, "invalid_timestamp"),
    ({"all_scores": {"sad": .1}}, "invalid_label_distribution"),
    ({"source": "emotion2vec"}, "invalid_simulation_provenance"),
    ({"label": {}}, "invalid_label_distribution"),
])
def test_rejected_observations_keep_audit_but_cannot_influence_fusion(update, reason):
    observation = _observation("late_distress")
    observation.update(update)
    clean = prepare_acoustic(observation, turn_index=0, turn_id="turn-0")
    assert clean["quality"] == 0
    assert reason in clean["rejection_reasons"]
    assert clean["prosodic"] == clean["emotion_trace"] == {}
    assert not fuse(clean, {"label": "neutral", "score": .8}).divergence


def test_stale_prosody_does_not_update_personal_baseline():
    persona = PersonaProfile(user_id="fiction")
    raw = simulated_prosodic_observations(_observation())
    clean = prepare_multimodal(raw, turn_index=1, turn_id="turn-1")
    enriched = enrich_with_personal_baseline(persona, clean)
    assert persona.longitudinal_baseline.signals == {}
    assert all(row["quality"] == 0 for row in enriched["prosodic"].values())
    assert raw["prosodic"]["speech_pace"]["quality"] > 0


def test_simulated_and_real_baseline_namespaces_do_not_mix():
    persona = PersonaProfile(user_id="fiction")
    simulated = simulated_prosodic_observations(_observation())
    enrich_with_personal_baseline(persona, simulated)
    enrich_with_personal_baseline(persona, {"prosodic": {"speech_pace": {"value": 4.0}}})
    assert persona.longitudinal_baseline.signals["simulated:prosodic:speech_pace"].count == 1
    assert persona.longitudinal_baseline.signals["prosodic:speech_pace"].count == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_bad_numeric_prosody_cannot_poison_baseline_or_json(bad):
    persona = PersonaProfile(user_id="fiction")
    raw = {"prosodic": {"speech_pace": {"value": bad, "confidence": .9}}}
    clean = prepare_multimodal(raw, turn_index=0)
    enrich_with_personal_baseline(persona, clean)
    assert not persona.longitudinal_baseline.signals
    json.dumps(clean, allow_nan=False)


def test_simulation_rejected_before_real_context_is_mutated():
    profile = PersonaProfile(user_id="real")
    engine = Engine(get_scene("companion_elder"), persona=profile)
    ctx = EngineContext(user_id="real", scene_id="companion_elder")
    before = asdict(ctx), profile.model_dump(mode="json")
    with pytest.raises(ValueError, match="isolated"):
        engine.process_turn(ctx, "今天下雨", acoustic=_observation())
    assert before == (asdict(ctx), profile.model_dump(mode="json"))


def test_simulation_constructor_requires_fiction_and_no_persistence(tmp_path):
    with pytest.raises(ValueError, match="synthetic_demo"):
        Engine(get_scene("companion_elder"), persona=PersonaProfile(user_id="real"), simulation_mode=True)
    with pytest.raises(ValueError, match="memory-only"):
        Engine(get_scene("companion_elder"), store=PersonaStore(base_dir=tmp_path), simulation_mode=True)
    with pytest.raises(ValueError, match="external tools"):
        Engine(get_scene("companion_elder"), capabilities={"notification"}, simulation_mode=True)


def test_engine_clones_fiction_and_resets_current_emotion_and_audit():
    profile = PersonaProfile(user_id="fiction", profile_origin="synthetic_demo")
    engine = Engine(get_scene("companion_elder"), persona=profile, simulation_mode=True)
    ctx = EngineContext(user_id="fiction", scene_id="companion_elder")
    raw = _observation()
    ctx.working_memory["_current_turn_id"] = "turn-0"
    engine.process_turn(ctx, "今天下雨", acoustic=raw, multimodal=simulated_prosodic_observations(raw))
    assert ctx.working_memory["perception_fusion"]["label"] == "sad"
    assert engine.persona.longitudinal_baseline.signals
    assert not profile.longitudinal_baseline.signals
    assert "_current_turn_id" not in ctx.working_memory
    engine.process_turn(ctx, "你好")
    assert ctx.working_memory["perception_input"]["acoustic"] is None
    assert ctx.working_memory["perception_fusion"]["acoustic"] == {}
    assert ctx.emotion.acoustic == {}


def test_prompt_keeps_quality_but_not_scenario_identifiers_or_rejected_label():
    engine = _engine()
    ctx = EngineContext(user_id="fiction", scene_id="companion_elder")
    raw = _observation("low_quality")
    ctx.working_memory["_current_turn_id"] = "turn-0"
    engine.process_turn(ctx, "今天下雨", acoustic=raw)
    prompt = json.dumps(build_messages(engine.scene, engine.persona, ctx, "今天下雨"), ensure_ascii=False)
    assert "acoustic_observation" in prompt
    assert "simulated_acoustic" in prompt
    assert "simulation_id" not in prompt and "low_quality" not in prompt
    assert "turn-0" not in prompt
    assert "0.08" in prompt


def test_safety_early_return_clears_previous_fusion_snapshot():
    engine = _engine()
    ctx = EngineContext(user_id="fiction", scene_id="companion_elder")
    engine.process_turn(ctx, "今天下雨", acoustic=_observation())
    assert ctx.working_memory["perception_fusion"]
    engine.process_turn(ctx, "我胸痛，喘不上气了")
    assert "perception_fusion" not in ctx.working_memory
    assert ctx.emotion.acoustic == {}
