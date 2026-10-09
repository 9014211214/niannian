"""Software identity/permission/commit tests; supplied labels are not model accuracy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from aria.engine.cognitive import (
    READINESS_MESSAGE, cognitive_action, online_cognitive_block, stage_cognitive_readiness,
    abandon_cognitive_readiness,
)
from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import (
    acknowledge_preference_invitation, apply_preference_dialogue, current_preference_invitation,
    preference_dialogue_prompt, preference_invitation_view, prepare_preference_invitation,
    validate_preference_control,
)
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.online_decision import apply_single_pass_decision, online_decision_tool_spec
from aria.engine.policy import PolicyEngine
from aria.engine.policy.prompt import build_messages
from aria.runtime import create_minimal_clients, create_minimal_engine
from aria.scenes.registry import get_scene
from tests.test_stated_preferences import _profile
from tests.test_online_json import DECISION
from tests.test_cognitive_change_v2 import calibrated, enabled, record, T0
from tests.test_reserved_calls import client as transport_client, guard as transport_guard, receipt as transport_receipt, stub as transport_stub
from tests._model_http_stub import stub_model_http


def fixture(*, acknowledge=True):
    persona = _profile(confirmed=False)
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="new_dialogue")
    invitation = prepare_preference_invitation(persona, ctx, persona.stated_preferences[0].id, actor=persona.user_id)
    if acknowledge:
        acknowledge_preference_invitation(persona, ctx, invitation["id"], invitation["question_sha256"], actor=persona.user_id)
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v3"
    return persona, ctx


def control(persona, ctx, text="本人当前回答", action="confirm"):
    pending = current_preference_invitation(persona, ctx)
    return {"version": "preference_confirmation_v1", "action": action,
            "invitation_id": pending["id"], "target_id": pending["target_id"],
            "target_sha256": pending["target_sha256"], "evidence_quote": text}


def decide(persona, ctx, value, text="本人当前回答", concern="none"):
    payload = {**deepcopy(DECISION), "safety_concern": concern, "preference_control": value}
    apply_single_pass_decision(json.dumps(payload, ensure_ascii=False), ctx, text, persona)


def test_actual_display_receipt_is_required_and_quote_is_real_owner_source():
    p, ctx = fixture(acknowledge=False)
    assert current_preference_invitation(p, ctx) is None
    assert preference_dialogue_prompt(p, ctx) == ""
    view = preference_invitation_view(p, ctx)
    assert set(view) == {"id", "question", "question_sha256", "preference_id", "session_id", "status"}
    assert view["status"] == "awaiting_display"
    assert p.stated_preferences[0].quotes[0].text in view["question"]
    with pytest.raises(ValueError):
        acknowledge_preference_invitation(p, ctx, view["id"], "wrong", actor=p.user_id)
    with pytest.raises(ValueError):
        acknowledge_preference_invitation(p, ctx, view["id"], view["question_sha256"], actor="caregiver")
    delivered = acknowledge_preference_invitation(p, ctx, view["id"], view["question_sha256"], actor=p.user_id)
    assert delivered["status"] == "delivered"
    assert "preference_control" in preference_dialogue_prompt(p, ctx)


@pytest.mark.parametrize("change", ["owner", "session", "turn", "target", "expire", "simulation"])
def test_current_identity_target_and_ttl_bind_delivery(change):
    p, ctx = fixture()
    if change == "owner": ctx.user_id = "other"
    if change == "session": ctx.session_id = "other"
    if change == "turn": ctx.add_turn("另一次", "回应")
    if change == "target": p.stated_preferences[0].do = "交流时一次只讲一步"
    if change == "expire": ctx.working_memory["_preference_dialogue_invitation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    if change == "simulation": ctx.simulation_mode = True
    assert current_preference_invitation(p, ctx) is None
    assert preference_dialogue_prompt(p, ctx) == ""


@pytest.mark.parametrize("admission", ["rule", "unreviewed", "confirmed", "declined"])
def test_only_model_reviewed_proposals_can_be_invited(admission):
    p, ctx = fixture()
    row = p.stated_preferences[0]
    if admission == "rule": row.extraction_source = "rule"
    if admission == "unreviewed": row.semantic_review = None
    if admission in {"confirmed", "declined"}: row.status = admission
    with pytest.raises(ValueError):
        prepare_preference_invitation(p, ctx, row.id, actor=p.user_id)


@pytest.mark.parametrize("key,value", [
    ("action", []), ("action", True), ("action", "enable"), ("target_id", "pref_other"),
    ("target_sha256", "stale"), ("invitation_id", "old"), ("evidence_quote", "只选了片段"),
    ("evidence_quote", True), ("version", "old"),
])
def test_malformed_control_never_changes_owner_proposal(key, value):
    p, ctx = fixture()
    payload = control(p, ctx)
    payload[key] = value
    assert validate_preference_control(payload, p, ctx, "本人当前回答") is None
    decide(p, ctx, payload)
    reply = apply_preference_dialogue(ctx, p, "本人当前回答")
    assert "还没有确认保存" in reply
    assert p.stated_preferences[0].status == "proposed"


@pytest.mark.parametrize("action,status", [("confirm", "confirmed"), ("decline", "declined")])
def test_commit_precedes_publish_and_retains_exact_owner_confirmation(tmp_path, action, status):
    p, ctx = fixture()
    original = deepcopy(p.stated_preferences[0].model_dump())
    value = control(p, ctx, action=action)
    decide(p, ctx, value)
    store = PersonaStore(tmp_path)
    reply = apply_preference_dialogue(ctx, p, "本人当前回答", store=store)
    assert reply
    assert store.load(p.user_id).stated_preferences[0].status == status
    assert p.stated_preferences[0].do == original["do"] and p.stated_preferences[0].scope == original["scope"]
    assert p.stated_preferences[0].history[-1].quote.text == "本人当前回答"
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]
    assert current_preference_invitation(p, ctx) is None
    assert apply_preference_dialogue(ctx, p, "本人当前回答", store=store) is None


@pytest.mark.parametrize("action", ["defer", "unclear"])
def test_defer_and_correction_do_not_confirm_or_edit_existing_scope(action):
    p, ctx = fixture()
    before = p.model_dump()
    decide(p, ctx, control(p, ctx, action=action))
    assert apply_preference_dialogue(ctx, p, "本人当前回答")
    assert p.model_dump() == before
    assert current_preference_invitation(p, ctx) is None


def test_save_failure_leaves_live_profile_and_invitation_unchanged():
    p, ctx = fixture()
    before = p.model_dump()
    decide(p, ctx, control(p, ctx))
    class Broken:
        def save(self, candidate):
            assert candidate.stated_preferences[0].status == "confirmed"
            raise OSError("injected save failure")
    with pytest.raises(OSError):
        apply_preference_dialogue(ctx, p, "本人当前回答", store=Broken())
    assert p.model_dump() == before
    assert current_preference_invitation(p, ctx) is not None
    assert "preference_dialogue_receipt" not in ctx.working_memory


@pytest.mark.parametrize("change", ["safety", "sha", "trace_turn", "trace_version"])
def test_current_safe_trace_required_even_after_parser_validation(change):
    p, ctx = fixture()
    decide(p, ctx, control(p, ctx))
    trace = ctx.working_memory["online_turn_decision"]
    if change == "safety": trace["safety_concern"] = "medical_emergency"
    if change == "sha": trace["owner_message_sha256"] = "wrong"
    if change == "trace_turn": trace["turn_index"] = 1
    if change == "trace_version": trace["version"] = "online_turn_decision_v1"
    assert "还没有确认保存" in apply_preference_dialogue(ctx, p, "本人当前回答")
    assert p.stated_preferences[0].status == "proposed"


def test_engine_one_same_call_commits_before_returning_confirmation(monkeypatch, tmp_path):
    p, ctx = fixture()
    value = control(p, ctx)
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture")
    calls = []
    def receive(request):
        calls.append(json.loads(request.data))
        payload = {**deepcopy(DECISION), "safety_concern": "none", "preference_control": value}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    store = PersonaStore(tmp_path)
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=clients, store=store)
    reply = engine.process_turn(ctx, "本人当前回答")
    assert len(calls) == 1
    assert value["invitation_id"] in str(calls[0]["messages"])
    assert store.load(p.user_id).stated_preferences[0].status == "confirmed"
    assert reply.startswith("我记下了") and ctx.turns[-1].agent == reply
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


def test_engine_save_failure_stops_reply_and_prevents_false_publish(monkeypatch):
    p, ctx = fixture()
    value = control(p, ctx)
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture")
    calls = []
    def receive(request):
        calls.append(json.loads(request.data))
        payload = {**deepcopy(DECISION), "safety_concern": "none", "preference_control": value}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    class BrokenStore:
        def save(self, candidate):
            raise OSError("injected durable commit failure")
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=clients, store=BrokenStore())
    with pytest.raises(OSError, match="durable commit failure"):
        engine.process_turn(ctx, "本人当前回答")
    assert len(calls) == 1 and ctx.turns == []
    assert p.stated_preferences[0].status == "proposed"
    assert "preference_dialogue_receipt" not in ctx.working_memory
    assert current_preference_invitation(p, ctx) is not None


@pytest.mark.parametrize("edge", ["not_delivered", "expired", "target_changed"])
def test_supplied_invalid_control_cannot_deliver_an_unsaved_confirmation(monkeypatch, tmp_path, edge):
    p, ctx = fixture()
    value = control(p, ctx)
    if edge == "not_delivered": ctx.working_memory["_preference_dialogue_invitation"]["delivered"] = False
    if edge == "expired": ctx.working_memory["_preference_dialogue_invitation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    if edge == "target_changed": p.stated_preferences[0].do = "交流时一次只讲一步"
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture")
    def receive(request):
        payload = {**deepcopy(DECISION), "safety_concern": "none", "preference_control": value,
                   "response": "我记下了，按你刚才确认的范围相处。"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    store = PersonaStore(tmp_path); store.save(p)
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=clients, store=store)
    reply = engine.process_turn(ctx, "本人当前回答")
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["status"] == "parsed" and trace["preference_control_supplied"] is True
    assert trace["preference_control_status"] == "missing_or_invalid"
    assert "还没有确认保存" in reply and not reply.startswith("我记下了")
    assert store.load(p.user_id).stated_preferences[0].status == p.stated_preferences[0].status == "proposed"
    assert "preference_dialogue_receipt" not in ctx.working_memory


def test_omitted_control_keeps_normal_dialogue_open(monkeypatch, tmp_path):
    p, ctx = fixture(acknowledge=False)
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-fixture")
    def receive(request):
        payload = {**deepcopy(DECISION), "safety_concern": "none", "response": "我在听。"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, "_request_json", receive)
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=clients, store=PersonaStore(tmp_path))
    assert engine.process_turn(ctx, "聊聊今天。") == "我在听。"
    assert ctx.working_memory["online_turn_decision"]["preference_control_supplied"] is False
    assert p.stated_preferences[0].status == "proposed"


def cognitive_fixture():
    p = calibrated(enabled())
    record(p, 20, impact=True)
    record(p, 21, domain="daily_function", impact=True)
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder", session_id="ready")
    ctx.add_turn("你好", "你好")
    return p, ctx, T0 + timedelta(days=22)


def test_explicit_text_readiness_prompt_is_repeatable_and_does_not_fake_emotion():
    p, ctx, now = cognitive_fixture()
    assert online_cognitive_block(p, READINESS_MESSAGE, ctx.turns, ctx=ctx, now=now) == ""
    grant = stage_cognitive_readiness(p, ctx, actor=p.user_id, now=now)
    assert grant["message"] == READINESS_MESSAGE
    block = online_cognitive_block(p, READINESS_MESSAGE, ctx.turns, ctx=ctx, now=now)
    assert "可以温和提起的观察" in block
    assert ctx.emotion.external_reading is None
    assert ctx.working_memory["_cognitive_readiness_prepared"]["used_for_prompt"] is True
    assert "_cognitive_text_readiness" in ctx.working_memory
    assert online_cognitive_block(p, READINESS_MESSAGE, ctx.turns, ctx=ctx, now=now) == block
    abandon_cognitive_readiness(ctx, "owner_discarded")
    assert "_cognitive_text_readiness" not in ctx.working_memory
    assert online_cognitive_block(p, READINESS_MESSAGE, ctx.turns, ctx=ctx, now=now) == ""


@pytest.mark.parametrize("change", ["owner", "session", "turn", "message", "source", "consent", "expiry", "simulation"])
def test_readiness_rejects_stale_changed_owner_message_and_source(change):
    p, ctx, now = cognitive_fixture()
    stage_cognitive_readiness(p, ctx, actor=p.user_id, now=now)
    text = READINESS_MESSAGE
    if change == "owner": ctx.user_id = "other"
    if change == "session": ctx.session_id = "other"
    if change == "turn": ctx.add_turn("另一次", "回应")
    if change == "message": text = "先聊别的"
    if change == "source": p.cognitive_observations[0].source_quote = "发生了变化"
    if change == "consent": cognitive_action(p, "disable", actor=p.user_id, delete_records=False, now=now)
    if change == "expiry": now += timedelta(minutes=5)
    if change == "simulation": ctx.simulation_mode = True
    assert online_cognitive_block(p, text, ctx.turns, ctx=ctx, now=now) == ""
    assert "_cognitive_text_readiness" in ctx.working_memory  # Pure preview cannot consume permission.


@pytest.mark.parametrize("condition", ["not_owner", "calibrating", "disabled", "cooldown", "old_window"])
def test_readiness_never_overrides_calibration_consent_trend_or_cooldown(condition):
    p, ctx, now = cognitive_fixture()
    actor = p.user_id
    if condition == "not_owner": actor = "caregiver"
    if condition == "calibrating": p.cognitive_trend.calibration_completed_at = None; p.cognitive_trend.session_times = {}
    if condition == "disabled": cognitive_action(p, "disable", actor=p.user_id, delete_records=False, now=now)
    if condition == "cooldown": cognitive_action(p, "decline", actor=p.user_id, now=now)
    if condition == "old_window": now += timedelta(days=40)
    with pytest.raises((ValueError, PermissionError)):
        stage_cognitive_readiness(p, ctx, actor=actor, now=now)
    assert "_cognitive_text_readiness" not in ctx.working_memory


def test_new_optional_controls_leave_historical_v1_v2_schemas_unchanged():
    for version in ("online_turn_decision_v1", "online_turn_decision_v2"):
        assert "preference_control" not in online_decision_tool_spec(version)["function"]["parameters"]["properties"]
    schema = online_decision_tool_spec("online_turn_decision_v3")["function"]["parameters"]
    assert "preference_control" not in schema["required"]
    assert schema["properties"]["preference_control"]["properties"]["action"]["enum"] == ["confirm", "decline", "defer", "unclear"]


def dispatch_fixture(monkeypatch):
    p, ctx, now = cognitive_fixture()
    monkeypatch.setattr("aria.engine.cognitive._now", lambda value=None: value or now)
    stage_cognitive_readiness(p, ctx, actor=p.user_id, now=now)
    c = transport_client()
    policy = PolicyEngine(fast_llm=c, online_decision_version="online_turn_decision_v3")
    return p, ctx, c, policy


def test_repeatable_previews_then_one_real_fixture_wire_exposure(monkeypatch, tmp_path):
    p, ctx, c, policy = dispatch_fixture(monkeypatch)
    original = deepcopy(ctx.working_memory["_cognitive_text_readiness"])
    scene = get_scene("companion_elder")
    first = build_messages(scene, p, ctx, READINESS_MESSAGE)
    first_snapshot = deepcopy(ctx.working_memory["prompt_evidence_snapshot"])
    second = build_messages(scene, p, ctx, READINESS_MESSAGE)
    assert "可以温和提起的观察" in str(first) and "可以温和提起的观察" in str(second)
    assert not first_snapshot["passed_to_client"] and first_snapshot["cognitive_readiness"]["used_for_prompt"]
    assert ctx.working_memory["_cognitive_text_readiness"] == original
    value = {**deepcopy(DECISION), "safety_concern": "none"}
    rebuilt = []
    def rebuild_same_operation(request):
        # A transport-preparation callback may inspect/rebuild the same prompt
        # while complete() is still in flight. The grant and client snapshot survive.
        rebuilt.append(build_messages(scene, p, ctx, READINESS_MESSAGE))
        assert ctx.working_memory["_cognitive_text_readiness"] == original
        assert "可以温和提起的观察" in str(rebuilt[-1])
    calls = transport_stub(monkeypatch, transport_receipt(c, choices=[{
        "finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]),
        on_open=rebuild_same_operation)
    g = transport_guard(tmp_path / "ready.sqlite3"); g.attach(c)
    try:
        with g.scope("turn", "ready-request", ctx.session_id):
            policy.respond(ctx, scene, READINESS_MESSAGE, p)
        assert len(calls) == c.network_request_count == g.snapshot()["calls_used"] == 1
        assert len(rebuilt) == 1
        assert "可以温和提起的观察" in str(calls[0]["messages"])
        assert p.cognitive_observations[0].source_quote in str(calls[0]["messages"])
        assert ctx.emotion.external_reading is None
        assert "_cognitive_text_readiness" not in ctx.working_memory
        receipt = deepcopy(ctx.working_memory["cognitive_readiness_receipt"])
        assert receipt["client_entered"] and receipt["dispatch_outcome"] == "client_returned"
        assert receipt["provider_attempt"] == "not_inferred_from_client_entry"
        assert ctx.working_memory["prompt_evidence_snapshot"]["passed_to_client"] is True
        # No new grant or second logical dispatch for a failed storage retry of the same turn.
        with pytest.raises(ValueError, match="不能重新准备"):
            stage_cognitive_readiness(p, ctx, actor=p.user_id)
        with g.scope("turn", "different-retry-id", ctx.session_id):
            policy.respond(ctx, scene, READINESS_MESSAGE, p)
        assert len(calls) == c.network_request_count == g.snapshot()["calls_used"] == 1
        assert ctx.working_memory["cognitive_readiness_receipt"] == receipt
        assert ctx.working_memory["prompt_evidence_snapshot"]["passed_to_client"] is False
    finally:
        g.close()


def test_reservation_rejection_has_zero_attempt_and_explicitly_abandons_grant(monkeypatch, tmp_path):
    p, ctx, c, policy = dispatch_fixture(monkeypatch)
    network = []
    def forbidden(*args, **kwargs):
        network.append(1)
        raise AssertionError("A presend rejection must never open the transport")
    stub_model_http(monkeypatch, forbidden)
    g = transport_guard(tmp_path / "presend.sqlite3", budget="0.000001"); g.attach(c)
    try:
        with g.scope("turn", "presend-rejection", ctx.session_id):
            policy.respond(ctx, get_scene("companion_elder"), READINESS_MESSAGE, p)
        assert g.snapshot()["calls_used"] == c.network_request_count == 0 and network == []
        receipt = ctx.working_memory["cognitive_readiness_receipt"]
        assert receipt["client_entered"] and receipt["dispatch_outcome"] == "client_error"
        assert receipt["provider_attempt"] == "not_inferred_from_client_entry"
        assert "_cognitive_text_readiness" not in ctx.working_memory
    finally:
        g.close()


def test_dispatch_unknown_consumes_grant_without_creating_a_retry(monkeypatch, tmp_path):
    import urllib.error
    p, ctx, c, policy = dispatch_fixture(monkeypatch)
    calls = []
    def unknown(request, **kwargs):
        calls.append(json.loads(request.data))
        raise urllib.error.URLError("injected unknown fixture delivery")
    stub_model_http(monkeypatch, unknown)
    g = transport_guard(tmp_path / "unknown.sqlite3"); g.attach(c)
    try:
        with g.scope("turn", "unknown-request", ctx.session_id):
            policy.respond(ctx, get_scene("companion_elder"), READINESS_MESSAGE, p)
        assert len(calls) == c.network_request_count == g.snapshot()["calls_used"] == 1
        assert "可以温和提起的观察" in str(calls[0]["messages"])
        assert g.snapshot()["stopped"]
        assert "_cognitive_text_readiness" not in ctx.working_memory
        assert ctx.working_memory["cognitive_readiness_receipt"]["dispatch_outcome"] == "client_error"
        with g.scope("turn", "unknown-retry", ctx.session_id):
            policy.respond(ctx, get_scene("companion_elder"), READINESS_MESSAGE, p)
        assert len(calls) == c.network_request_count == g.snapshot()["calls_used"] == 1
    finally:
        g.close()
