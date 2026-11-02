"""Neutral forced-risk/forced-output plumbing probes, not semantic evaluation."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile, SessionCheckpointRecord
from aria.engine.risk.premodel_task import validate_premodel_safety_version
from aria.engine.risk.safety_task import KEY
from aria.interfaces.risk import RiskSignal
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from tests._model_http_stub import stub_model_http
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence
from tests.test_safety_response_instruction import bundle

TEXT = "中性软件任务来源甲。"
TASK = "中性软件输出甲\n中性软件输出乙"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


def fixture(monkeypatch, *, version="task_response_v1", value=None):
    clients = bundle()
    wires, order = [], []
    if value is None:
        value = payload("none", evidence("", "not_applicable", "not_applicable", "not_applicable"), "未选旁白。" + TASK)
        value["task_projection"] = {"version": "task_projection_v1", "status": "in_dialogue_task",
                                    "request_quote": TEXT, "response_quote": TASK}
    def receive(request, **kwargs):
        order.append("model")
        wires.append((json.loads(request.data), kwargs))
        if isinstance(value, Exception):
            raise value
        return envelope(value, "json_schema_v1")
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
        clients=clients, safety_delivery_version="task_context_v2",
        safety_response_instruction_version="conditional_response_v2", premodel_safety_version=version)
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="forced_neutral_probe", evidence=[]))
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), wires, order


def test_prelude_precedes_single_model_and_preserves_task_without_whole_raw(monkeypatch):
    engine, ctx, wires, order = fixture(monkeypatch)
    ctx.add_turn("历史来源甲。", "历史输出甲。")
    events = []
    def emit(event):
        order.append("prelude"); events.append(deepcopy(event)); return True
    response = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert order == ["prelude", "model"] and len(wires) == 1
    assert response.startswith(events[0]["protection"]) and response.endswith(TASK)
    assert "未选旁白" not in response
    assert set(events[0]) == {"version", "premodel_safety_version", "user_id", "session_id", "scene_id",
                             "turn_index", "utterance_sha256", "protection", "protection_sha256"}
    assert events[0]["utterance_sha256"] == sha256(TEXT.encode()).hexdigest()
    assert events[0]["protection_sha256"] == sha256(events[0]["protection"].encode()).hexdigest()
    serialized = json.dumps(wires[0][0], ensure_ascii=False)
    assert TEXT in serialized and "历史来源甲" in serialized and "历史输出甲" in serialized
    assert ctx.working_memory["premodel_task"]["status"] == "response_only_complete"
    assert ctx.working_memory["premodel_task"]["client_entered"] is True
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [1]
    assert ctx.working_memory["online_turn_decision"]["safety_concern"] == "none"
    assert ctx.working_memory["safety_delivery"]["safety_context_status"] == "unverified"


@pytest.mark.parametrize("result", [False, None, 1, "true", [], {}])
def test_callback_requires_strict_true(monkeypatch, result):
    engine, ctx, wires, _ = fixture(monkeypatch)
    response = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: result)
    assert not wires and TASK not in response and "急救" in response


@pytest.mark.parametrize("kind", ["missing", "raises", "legacy"])
def test_absent_or_failed_or_legacy_callback_never_calls_model(monkeypatch, kind):
    engine, ctx, wires, order = fixture(monkeypatch, version="legacy" if kind == "legacy" else "task_response_v1")
    def emit(event):
        order.append("prelude")
        if kind == "raises": raise OSError("neutral fixture")
        return True
    response = engine.process_turn(ctx, TEXT, safety_prelude=None if kind == "missing" else emit)
    assert not wires and TASK not in response and "急救" in response
    if kind == "legacy": assert not order


@pytest.mark.parametrize("kind", ["event", "user", "session", "scene", "turn", "utterance", "version"])
def test_callback_identity_changes_stop_before_model(monkeypatch, kind):
    engine, ctx, wires, _ = fixture(monkeypatch)
    def emit(event):
        if kind == "event": event["protection"] = "changed"
        elif kind == "user": ctx.user_id = "other"
        elif kind == "session": ctx.session_id = "other"
        elif kind == "scene": ctx.scene_id = "other"
        elif kind == "turn": ctx.add_turn("other", "other")
        elif kind == "utterance": ctx.working_memory["current_utterance"] = "other"
        elif kind == "version": engine.premodel_safety_version = "legacy"
        return True
    response = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert not wires and TASK not in response


def test_reentry_cannot_generate_before_prelude_finishes(monkeypatch):
    engine, ctx, wires, _ = fixture(monkeypatch)
    def emit(event):
        with pytest.raises(ValueError, match="reentry"):
            engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
        return False
    engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert not wires


def test_reentry_during_actual_client_is_rejected_and_turn_lock_is_released(monkeypatch):
    engine, ctx, wires, _ = fixture(monkeypatch)
    receive = engine.policy.fast_llm._request_json
    attempts = []
    def reentrant(request, **kwargs):
        attempts.append("outer")
        with pytest.raises(ValueError, match="reentry"):
            engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
        return receive(request, **kwargs)
    monkeypatch.setattr(engine.policy.fast_llm, "_request_json", reentrant)
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert result.endswith(TASK) and attempts == ["outer"] and len(wires) == len(ctx.turns) == 1
    assert "_premodel_task_inflight" not in ctx.working_memory
    engine.process_turn(ctx, TEXT)  # a later legitimate turn remains available
    assert len(ctx.turns) == 2 and len(wires) == 1


@pytest.mark.parametrize("failure", ["callback", "client", "commit"])
def test_premodel_lock_clears_after_failures(monkeypatch, failure):
    engine, ctx, wires, _ = fixture(monkeypatch, value=RuntimeError("neutral") if failure == "client" else None)
    def emit(event):
        if failure == "callback": raise RuntimeError("neutral callback")
        return True
    if failure == "commit":
        monkeypatch.setattr(ctx, "add_turn", lambda *a: (_ for _ in ()).throw(OSError("neutral commit")))
        with pytest.raises(OSError, match="neutral commit"):
            engine.process_turn(ctx, TEXT, safety_prelude=emit)
    else:
        engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert "_premodel_task_inflight" not in ctx.working_memory


def test_no_normal_business_mutation_and_pending_authority_not_advertised(monkeypatch):
    engine, ctx, wires, _ = fixture(monkeypatch)
    ctx.working_memory["_pending_external_action"] = {"id": "pending-fixture", "tool": "forbidden_fixture_tool",
        "args": {}, "created_turn": 0, "expires_turn": 2}
    ctx.working_memory["_preference_dialogue_invitation"] = {"marker": "forbidden_invitation_fixture"}
    before_pending = deepcopy(ctx.working_memory["_pending_external_action"])
    before_profile = engine.persona.model_dump(mode="json")
    before_alliance, before_formulation = deepcopy(ctx.alliance), deepcopy(ctx.formulation)
    def forbidden(*args, **kwargs): pytest.fail("normal business hook executed")
    monkeypatch.setattr(engine.scene, "observe_care", forbidden)
    monkeypatch.setattr(engine.scene, "execute_care_skills", forbidden)
    monkeypatch.setattr(engine._skills, "evaluate", forbidden)
    monkeypatch.setattr(engine._tools, "invoke", forbidden)
    monkeypatch.setattr(engine._state_reasoner, "observe_response", forbidden)
    monkeypatch.setattr(engine._state_reasoner, "update", forbidden)
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert result.endswith(TASK) and len(wires) == 1
    after = engine.persona.model_dump(mode="json")
    for allowed in ("safety_screenings",):
        before_profile.pop(allowed, None); after.pop(allowed, None)
    before_profile["person_model_state"]["first_interaction_at"] = after["person_model_state"]["first_interaction_at"]
    assert after == before_profile
    assert ctx.alliance == before_alliance and ctx.formulation == before_formulation
    assert ctx.working_memory["_pending_external_action"] == before_pending
    wire = json.dumps(wires[0][0], ensure_ascii=False)
    assert "forbidden_fixture_tool" not in wire and "forbidden_invitation_fixture" not in wire
    assert "cognitive_control" not in ctx.working_memory["online_turn_decision"]
    assert "preference_control" not in ctx.working_memory["online_turn_decision"]
    assert not engine.persona.risk_events


def test_cloned_business_objects_and_store_are_not_written_by_generated_state(tmp_path, monkeypatch):
    from aria.engine.memory.persona_store import PersonaStore
    value = payload("none", evidence("", "not_applicable", "not_applicable", "not_applicable"), TASK)
    value["task_projection"] = {"version": "task_projection_v1", "status": "in_dialogue_task",
                                "request_quote": TEXT, "response_quote": TASK}
    value["cognitive_control"] = {"version": "cognitive_permission_v1", "action": "disable_observation",
                                  "evidence_quote": TEXT}
    engine, ctx, _, _ = fixture(monkeypatch, value=value)
    store = PersonaStore(tmp_path)
    engine.store = store
    store.save(engine.persona)
    stored_before = (tmp_path / "owner.json").read_bytes()
    original_policy = engine.policy
    engine.scene.neutral_probe_state = {"writes": 0}
    original_agent_persona = type(engine.scene).agent_persona
    def cloned_scene_probe(self):
        self.neutral_probe_state["writes"] += 1
        return original_agent_persona(self)
    monkeypatch.setattr(type(engine.scene), "agent_persona", cloned_scene_probe)
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert result.endswith(TASK)
    assert engine.scene.neutral_probe_state == {"writes": 0}
    assert engine.policy is original_policy
    assert (tmp_path / "owner.json").read_bytes() == stored_before
    assert "cognitive_control" not in ctx.working_memory["online_turn_decision"]
    engine.checkpoint_session(ctx)
    restored = store.load("owner")
    assert restored.pending_session.premodel_safety_version == "task_response_v1"
    assert restored.pending_session.turns == [] and restored.pending_session.turn_indices == []
    assert not restored.episodic_memory and not restored.person_hypotheses


def test_copy_failure_keeps_protection_and_has_no_model_attempt(monkeypatch):
    engine, ctx, wires, _ = fixture(monkeypatch)
    def fail_copy(memo): raise TypeError("not copyable")
    monkeypatch.setattr(engine.scene, "__deepcopy__", fail_copy, raising=False)
    events = []
    response = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert response == events[0]["protection"] and not wires
    assert ctx.working_memory["premodel_task"]["status"] == "response_only_failed"


@pytest.mark.parametrize("kind", ["exception", "missing_projection", "bad_source", "action", "false_completion"])
def test_model_failure_or_guard_rejection_retains_protection(monkeypatch, kind):
    value = RuntimeError("neutral transport failure") if kind == "exception" else payload("none", evidence("", "not_applicable", "not_applicable", "not_applicable"), TASK)
    if isinstance(value, dict) and kind != "missing_projection":
        value["task_projection"] = {"version": "task_projection_v1", "status": "in_dialogue_task",
                                    "request_quote": "absent" if kind == "bad_source" else TEXT,
                                    "response_quote": TASK}
        if kind in ("action", "false_completion"):
            text = '<action>{"tool":"x","args":{}}</action>' if kind == "action" else "我已经帮你发给家人了。"
            value["response"] = value["task_projection"]["response_quote"] = text
    engine, ctx, wires, _ = fixture(monkeypatch, value=value)
    events = []
    response = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert response == events[0]["protection"] and len(wires) == 1
    if kind in ("exception", "missing_projection"):
        assert ctx.working_memory["model_route"]["reply"]["source"] == "deterministic_fallback"
        assert ctx.working_memory["premodel_task"]["status"] == "response_only_failed"
    assert "<action" not in response


@pytest.mark.parametrize("bad", [None, 1, True, {}, "other"])
def test_version_is_strict_and_checkpoint_default_is_legacy(bad):
    with pytest.raises(ValueError): validate_premodel_safety_version(bad)
    assert SessionCheckpointRecord.model_fields["premodel_safety_version"].default == "legacy"


def test_configuration_and_context_versions_cannot_silently_change(monkeypatch):
    with pytest.raises(ValueError, match="premodel_task_requires"):
        create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
                             clients=bundle(), premodel_safety_version="task_response_v1")
    engine, ctx, wires, _ = fixture(monkeypatch)
    ctx.working_memory["_premodel_safety_version"] = "legacy"
    with pytest.raises(ValueError, match="session_premodel"):
        engine.process_turn(ctx, TEXT, safety_prelude=lambda event: True)
    assert not wires and not ctx.turns and KEY not in ctx.working_memory
