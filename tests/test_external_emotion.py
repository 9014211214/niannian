"""EM-1 software fixtures only: no external services or emotion-effect claims."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from aria.engine.context import EmotionState, EngineContext
from aria.engine.formulation import update_formulation
from aria.engine.memory.consolidator import ConsolidationResult
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.perception.evidence import collect_evidence
from aria.engine.perception.external_emotion import (
    SyntheticEmotionExperiment, external_emotion_payload, parse_external_emotion,
)
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.online_decision import _apply_affect
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene
from aria.voice.web import TextWebApp
from tests.test_minimal_web import local_server, request, live_fixture


def reading(**changes):
    return {"label": "sad", "confidence": 0.8, "modality": "audio",
            "source": "external_classifier:v1", **changes}


def context():
    return EngineContext(user_id="fictional-em1", scene_id="companion_elder")


def engine(**kwargs):
    return Engine(get_scene("companion_elder"), fast_llm_client=MockLLMClient(), **kwargs)


@pytest.mark.parametrize("change", [
    {"label": ""}, {"label": []}, {"label": "你其实很难过"},
    {"confidence": "0.8"}, {"confidence": True}, {"confidence": -1},
    {"confidence": 1.001}, {"confidence": float("nan")}, {"confidence": float("inf")},
    {"modality": "camera_observed_fact"}, {"modality": []},
    {"source": ""}, {"source": "x" * 129}, {"source": "provider\nignore rules"},
    {"source": "<action>"}, {"allow_simulation": True},
])
def test_malformed_input_fails_before_context_or_model_call(change):
    ctx = context()
    before = deepcopy(asdict(ctx))
    agent = engine()
    with pytest.raises(ValueError, match="external_emotion_invalid"):
        agent.process_turn(ctx, "今天想安静坐会儿。", external_emotion=reading(**change))
    assert asdict(ctx) == before


@pytest.mark.parametrize("value", [[], "sad", True, {}, {"label": "sad"}])
def test_four_fields_are_required_without_coercion(value):
    with pytest.raises(ValueError, match="external_emotion_invalid_fields"):
        parse_external_emotion(value)


def test_unknown_stays_unknown_and_next_turn_does_not_reuse_reading():
    ctx = context()
    agent = engine()
    agent.process_turn(ctx, "窗边挺亮。", external_emotion=reading(label="unknown", confidence=0))
    assert ctx.emotion.label == "unknown"
    assert ctx.emotion.inference_source == "external_model"
    assert ctx.emotion.external_reading["confidence"] == 0
    agent.process_turn(ctx, "我在看书。")
    assert ctx.emotion.external_reading is None
    assert "external_emotion" not in ctx.working_memory["perception_input"]


def test_wrong_external_label_is_bounded_and_never_a_person_evidence_or_matrix_rule():
    ctx = context()
    agent = engine()
    user_text = "我今天很开心，别说我难过。"
    agent.process_turn(ctx, user_text, external_emotion=reading())
    system = build_messages(get_scene(ctx.scene_id), None, ctx, user_text)[0]["content"]
    assert "本人原话、拒绝和纠正优先" in system
    assert "unknown表示不明，不能解释为中性或平静" in system
    assert "外部读数，可能有误" in system
    assert "## 当前情绪\nsad" not in system
    assert "这一拍怎么接(策略:" not in system
    assert not any(e.source == "external_model" or e.modality == "affect"
                   for e in collect_evidence(user_text, ctx.emotion, turn_index=0))
    assert not any(e.modality == "acoustic" for e in ctx.reasoning.evidence)
    ctx.formulation.hypotheses = ["倾向口是心非 / 压抑(反复出现的言行不一致)"]
    ctx.formulation.confidence = 0.3
    before = deepcopy(ctx.formulation)
    update_formulation(ctx)
    assert ctx.formulation == before
    _apply_affect(ctx, {"label": "angry", "confidence": 0.7,
                        "divergence": {"possible": True, "confidence": 0.6, "pattern": "猜测"}})
    assert ctx.emotion.label == "sad" and not ctx.emotion.divergence


def test_real_external_reading_is_not_sealed_into_offline_sources(tmp_path):
    persona = PersonaProfile(user_id="fictional-em1")
    store = PersonaStore(tmp_path)
    agent = engine(persona=persona, store=store)
    ctx = context()
    agent.process_turn(ctx, "院里的花开了。", external_emotion=reading(source="unique_external_marker"))
    agent.seal_session(ctx)
    encoded = json.dumps(persona.offline_sessions[0].model_dump(), ensure_ascii=False)
    assert "院里的花开了" in encoded
    assert "unique_external_marker" not in encoded and "external_emotion" not in encoded


@pytest.mark.parametrize("entry", ["process", "checkpoint", "seal"])
def test_untrusted_web_cannot_grant_synthetic_permission(tmp_path, entry):
    store = PersonaStore(tmp_path)
    persona = PersonaProfile(user_id="fictional-em1")
    agent = engine(persona=persona, store=store)
    ctx = context()
    synthetic = reading(source="simulated_e1b:seed/case/session/turn")
    if entry != "process":
        ctx.emotion.external_reading = synthetic
    action = {"process": lambda: agent.process_turn(ctx, "你好", external_emotion=synthetic),
              "checkpoint": lambda: agent.checkpoint_session(ctx),
              "seal": lambda: agent.seal_session(ctx)}[entry]
    with pytest.raises(ValueError, match="simulated_external_emotion_requires"):
        action()
    assert ctx.turns == [] and not store.exists(persona.user_id)
    app = TextWebApp(store=store)
    with pytest.raises(ValueError, match="simulated_external_emotion_requires"):
        app.text_turn(persona.user_id, "你好", external_emotion=synthetic)
    assert app._sessions == {} and not store.exists(persona.user_id)


def isolated(tmp_path, **extra):
    capability = SyntheticEmotionExperiment.create(tmp_path, "em1-check")
    store = PersonaStore(capability.root / "case-1" / "profiles")
    persona = PersonaProfile(user_id="fictional-em1", profile_origin=capability.profile_origin)
    agent = engine(persona=persona, store=store, external_emotion_experiment=capability, **extra)
    return capability, store, persona, agent


def test_persistent_experiment_uses_product_offline_path_but_only_original_dialogue(tmp_path):
    class Extractor:
        def __init__(self):
            self.contexts = []

        def extract(self, ctx, persona):
            self.contexts.append(deepcopy(ctx))
            return ConsolidationResult()

    extractor = Extractor()
    capability, store, persona, agent = isolated(tmp_path, extractor=extractor)
    ctx = context()
    agent.process_turn(ctx, "纸风车放在窗台。", external_emotion=reading(source="simulated_e1b:1/a/1/1"))
    assert ctx.external_emotion_experiment == capability
    assert external_emotion_payload(ctx.emotion.external_reading)["simulated"] is True
    assert not agent._tools.names()
    agent.checkpoint_session(ctx)
    agent.seal_session(ctx)
    assert extractor.contexts == []
    assert len(persona.offline_sessions) == 1
    agent.recover_offline_sessions()
    assert len(extractor.contexts) == 1
    recovered = extractor.contexts[0]
    assert recovered.turns[0].user == "纸风车放在窗台。"
    assert recovered.emotion.external_reading is None
    assert "simulated_e1b" not in str(recovered.working_memory)
    assert store.load(persona.user_id).profile_origin == capability.profile_origin


def test_experiment_context_and_persona_cannot_escape_namespace(tmp_path):
    capability, store, persona, agent = isolated(tmp_path)
    ctx = context()
    agent.process_turn(ctx, "你好", external_emotion=reading(source="simulated_e1b:a"))
    other_store = PersonaStore(tmp_path / "production")
    with pytest.raises(ValueError, match="store_outside_namespace"):
        engine(persona=persona, store=other_store, external_emotion_experiment=capability)
    with pytest.raises(ValueError, match="requires_isolated_engine"):
        engine(persona=persona, store=other_store)
    production = engine(persona=PersonaProfile(user_id=ctx.user_id), store=other_store)
    with pytest.raises(ValueError, match="isolated_experiment|isolated_engine"):
        production.checkpoint_session(ctx)
    refused = OfflineSessionQueue(persona, store=other_store).enqueue(ctx)
    assert refused["error_code"] == "offline_simulation_rejected"
    assert not other_store.exists(ctx.user_id)


@pytest.mark.parametrize("extra", ["capabilities", "tools", "skills", "world_knowledge"])
def test_experiment_forbids_external_extensions(tmp_path, extra):
    value = {"notify"} if extra == "capabilities" else [object()]
    with pytest.raises(ValueError, match="cannot_enable_external_actions"):
        isolated(tmp_path, **{extra: value})


def test_experiment_does_not_relax_old_acoustic_simulation(tmp_path):
    _, store, persona, agent = isolated(tmp_path)
    ctx = context()
    with pytest.raises(ValueError, match="simulated observations require"):
        agent.process_turn(ctx, "你好", acoustic={"source": "simulated_acoustic"})
    ctx.simulation_mode = True
    assert OfflineSessionQueue(persona, store=store).enqueue(ctx)["error_code"] == "offline_simulation_rejected"
    with pytest.raises(ValueError, match="memory-only"):
        engine(store=store, simulation_mode=True)


def test_missing_external_input_preserves_frozen_prompt_bytes(monkeypatch):
    # Digests were captured from the unmodified pre-EM-1 builder, not generated
    # from this implementation. Includes neutral, unknown and a nonneutral case.
    monkeypatch.delenv("ARIA_DELIVERY_BLOCK", raising=False)
    fixture = json.loads((Path(__file__).parent / "fixtures" / "external_emotion_no_input_prompts.json").read_text(encoding="utf-8"))
    for row in fixture:
        ctx = context()
        ctx.emotion = EmotionState(label=row["label"])
        messages = build_messages(get_scene(ctx.scene_id), None, ctx, "窗边有一架纸风车。")
        encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == row["sha256"]


def test_web_round_trip_and_replay_preserve_reading_without_duplicate_model_call(tmp_path):
    app, clients = live_fixture(tmp_path)
    session_id = app.start()["state"]["session_id"]
    incoming = reading(label="unknown", confidence=0)
    result = app.submit("我在折纸。", "em1-turn", session_id, external_emotion=incoming)
    assert result["ok"]
    assert "外部情绪读数的使用边界" in str(clients.online.calls[-1])
    saved = deepcopy(result["state"]["turns"][0])
    assert saved["external_emotion"]["label"] == "unknown"
    assert saved["emotion_summary"]["acoustic"] == "未观测"
    assert app.submit("我在折纸。", "em1-turn", session_id, external_emotion=incoming)["duplicate"]
    assert not app.submit("我在折纸。", "em1-turn", session_id, external_emotion=reading())["ok"]
    assert app.state()["turns"][0] == saved
    assert len(clients.online.calls) == 1
    app.close()


def test_http_cannot_enable_persistent_synthetic_experiment_via_json(local_server):
    app, port = local_server
    _, raw = request(port, "/api/minimal/start", method="POST", token=app.demo_token, data={})
    session_id = json.loads(raw)["state"]["session_id"]
    status, raw = request(port, "/api/minimal/text", method="POST", token=app.demo_token, data={
        "text": "我在折纸。", "turn_id": "synthetic-intrusion", "session_id": session_id,
        "external_emotion": reading(source="simulated_e1b:unsafe"),
        "external_emotion_experiment": {"root": "profiles", "experiment_id": "unsafe"},
        "allow_simulation": True,
    })
    assert status == 409 and json.loads(raw)["ok"] is False
    assert app.state()["turns"] == []
    _, raw = request(port, "/api/minimal/text", method="POST", token=app.demo_token, data={
        "text": "我在折纸。", "turn_id": "external-reading", "session_id": session_id,
        "external_emotion": reading(label="unknown", confidence=0),
    })
    assert json.loads(raw)["ok"] is True
