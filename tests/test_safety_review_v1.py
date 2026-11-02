"""Forced labels/review answers test boundaries, not medical semantic quality."""
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.risk.safety_review import (
    ONLINE_VERSION, current_review_evidence, parse_safety_review,
    response_sentences, safety_review_messages,
)
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.medical import guard_general_medical_reply
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence
from tests.test_safety_response_instruction import bundle

TEXT = "本次软件来源。"
BODY = "保留原句甲。\n删除原句乙。保留原句丙。"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))


class Reviewer:
    role = "safety_review"

    def __init__(self, result=None, *, mutate=None, order=None):
        self.result = {"conflicts": False, "delete_sentence_ids": []} if result is None else result
        self.mutate, self.order, self.messages = mutate, order, []

    def complete(self, messages):
        if self.order is not None:
            self.order.append("review")
        self.messages.append(deepcopy(messages))
        if self.mutate:
            self.mutate()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result if isinstance(self.result, str) else json.dumps(self.result)


def build(monkeypatch, value, reviewer, *, v5=True):
    online = bundle().online
    if v5:
        online.online_decision_version = ONLINE_VERSION
        value = {**deepcopy(value), "applied_preference_ids": []}
    calls = []

    def receive(request, **kwargs):
        calls.append(json.loads(request.data))
        return envelope(value, "json_schema_v1")

    monkeypatch.setattr(online, "_request_json", receive)
    engine = Engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
                    fast_llm_client=online, safety_review_llm_client=reviewer,
                    safety_review_version="safety_review_v1",
                    safety_response_instruction_version="compact_response_v1",
                    safety_delivery_version="minimal_response_v1",
                    premodel_safety_version="task_response_v4")
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: None)
    return engine, EngineContext(user_id="owner", scene_id="companion_elder"), calls


def test_exact_subset_only_and_protection_before_review(monkeypatch):
    order = []
    reviewer = Reviewer({"conflicts": True, "delete_sentence_ids": [1]}, order=order)
    engine, ctx, calls = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), reviewer)
    events = []

    def emit(event):
        order.append("prelude"); events.append(deepcopy(event)); return True

    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert order == ["prelude", "review"] and len(calls) == 1
    assert result.startswith(events[0]["protection"])
    assert "保留原句甲。" in result and "保留原句丙。" in result and "删除原句乙。" not in result
    audit = ctx.working_memory["safety_review"]
    assert audit["delete_sentence_ids"] == [1] and audit["passed_to_client"] is True
    assert audit["actual_send_verified"] is audit["semantic_verified"] is False
    data = json.loads(reviewer.messages[0][1]["content"])
    assert data["protection_already_delivered"] == events[0]["protection"]
    assert data["final_protection"] == events[0]["protection"]
    assert audit["protection_sha256"] == audit["final_protection_sha256"] == events[0]["protection_sha256"]
    assert ctx.working_memory["safety_delivery"]["segment_ids"]
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == [0]


@pytest.mark.parametrize("review_result", [
    TimeoutError("not logged"), "not JSON", {"conflicts": True, "delete_sentence_ids": [99]},
    {"conflicts": False, "delete_sentence_ids": [0]}, {"conflicts": True, "delete_sentence_ids": []},
    {"conflicts": True, "delete_sentence_ids": [False]},
    {"conflicts": True, "delete_sentence_ids": [0, 0]},
    {"conflicts": False, "delete_sentence_ids": [], "body": "injection"},
    '{"conflicts":false,"conflicts":true,"delete_sentence_ids":[0]}',
])
def test_failed_or_invalid_review_never_releases_unreviewed_body(monkeypatch, review_result):
    reviewer = Reviewer(review_result)
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), reviewer)
    events = []
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda e: events.append(e) or True)
    assert result == events[0]["protection"] and "保留原句甲" not in result
    assert len(reviewer.messages) == 1 and ctx.working_memory["safety_review"]["status"] == "review_failed"


@pytest.mark.parametrize("callback", [None, False, "mutate"])
def test_no_confirmed_prelude_no_review(monkeypatch, callback):
    reviewer = Reviewer()
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), reviewer)
    def emit(event):
        if callback == "mutate": event["protection"] = "changed"
        return callback is not False
    result = engine.process_turn(ctx, TEXT, safety_prelude=None if callback is None else emit)
    assert not reviewer.messages and "保留原句甲" not in result


def test_ordinary_and_legacy_turns_have_no_review_or_prelude(monkeypatch):
    for value, v5 in [(payload(response=BODY), True),
                      (payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), False)]:
        reviewer = Reviewer()
        engine, ctx, calls = build(monkeypatch, value, reviewer, v5=v5)
        result = engine.process_turn(ctx, TEXT, safety_prelude=lambda e: pytest.fail("unexpected prelude"))
        assert result == BODY and len(calls) == 1 and not reviewer.messages
        assert "safety_review" not in ctx.working_memory


@pytest.mark.parametrize("mutation", ["user", "session", "turn", "utterance", "evidence", "source"])
def test_current_binding_rechecked_after_review(monkeypatch, mutation):
    reviewer = Reviewer()
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), reviewer)
    def mutate():
        if mutation == "user": ctx.user_id = "other"
        elif mutation == "session": ctx.session_id = "other"
        elif mutation == "turn": ctx.add_turn("new", "new")
        elif mutation == "utterance": ctx.working_memory["current_utterance"] = "new"
        elif mutation == "evidence": ctx.working_memory["online_turn_decision"]["safety_evidence"]["subject"] = "other"
        else: ctx.working_memory.pop("_source_response_for_turn")
    reviewer.mutate = mutate
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda e: True)
    assert "保留原句甲" not in result
    assert ctx.working_memory["safety_review"]["status"] == "source_changed"


def test_quote_failure_does_not_bypass_risk_review(monkeypatch):
    reviewer = Reviewer()
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence("not verbatim", "owner", "current", "immediate"), BODY), reviewer)
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda e: True)
    assert "不要尝试站起" in result and BODY in result and len(reviewer.messages) == 1
    data = json.loads(reviewer.messages[0][1]["content"])
    assert data["safety_evidence"]["verbatim_source"] is False


def test_previous_turn_review_cannot_cross_to_next_turn(monkeypatch):
    reviewer = Reviewer()
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "owner", "current", "immediate"), BODY), reviewer)
    engine.process_turn(ctx, TEXT, safety_prelude=lambda e: True)
    assert current_review_evidence(ctx, TEXT, persona=engine.persona) is None
    engine.process_turn(ctx, TEXT)
    assert len(reviewer.messages) == 1
    assert ctx.working_memory["safety_review"]["status"] == "prelude_unavailable"


@pytest.mark.parametrize("concern", ["fall", "medical_emergency"])
def test_legacy_review_keeps_additional_protection_gate_with_conditional_v4_prelude(monkeypatch, concern):
    from aria.interfaces.risk import RiskSignal
    order = []
    reviewer = Reviewer(order=order)
    engine, ctx, calls = build(monkeypatch, payload(concern, evidence(TEXT, "other", "current", "immediate"), BODY), reviewer)
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="medical_emergency_fall", evidence=[]))
    events = []
    def emit(event):
        order.append("prelude"); events.append(deepcopy(event)); return True
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert len(events) == len(calls) == 1 and not reviewer.messages
    assert order == ["prelude"] and BODY not in result
    from aria.engine.risk.safety_delivery import render_unverified_safety_v2
    assert events[0]["protection"] == render_unverified_safety_v2()[0]
    audit = ctx.working_memory["safety_review"]
    assert audit["prelude_status"] == "already_delivered"
    assert audit["status"] == "additional_protection_required"
    assert audit["review_call_count"] == 0 and audit["model_body_delivered"] is False
    import hashlib
    assert audit["protection_sha256"] == hashlib.sha256(events[0]["protection"].encode()).hexdigest()
    assert audit["final_protection_sha256"] == hashlib.sha256(result.encode()).hexdigest()
    assert audit["protection_sha256"] != audit["final_protection_sha256"]
    if concern == "medical_emergency":
        assert "心肺复苏" in result and "心肺复苏" not in events[0]["protection"]


@pytest.mark.parametrize("field", ["user_id", "session_id", "scene_id", "turn_index", "utterance_sha256", "protection", "protection_sha256", "version", "premodel_safety_version"])
def test_delivered_receipt_is_exactly_bound_to_current_event(field):
    from dataclasses import replace
    from aria.engine.risk.premodel_task import DeliveredSafetyPrelude, current_delivered_prelude, make_prelude
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    event = make_prelude(ctx, TEXT, "软件保护。")
    receipt = DeliveredSafetyPrelude(**event)
    assert current_delivered_prelude(ctx, TEXT, receipt) == event
    bad = replace(receipt, **{field: 3 if field == "turn_index" else "changed"})
    assert current_delivered_prelude(ctx, TEXT, bad) is None
    assert current_delivered_prelude(ctx, TEXT, True) is None
    assert current_delivered_prelude(ctx, TEXT, event) is None


def test_public_saved_callback_keeps_conditional_prelude_and_reviewed_body(tmp_path, monkeypatch):
    import io
    import hashlib
    from aria.engine.policy.llm import MockLLMClient
    from aria.interfaces.risk import RiskSignal
    from aria.runtime import MinimalModelClients
    from aria.voice.minimal import MinimalWebApp
    from tests._model_http_stub import stub_model_http
    from tests.test_online_structured import MODEL, policy
    online = bundle().online
    online.online_decision_version = ONLINE_VERSION
    reviewer = Reviewer({"conflicts": False, "delete_sentence_ids": [],
        "risk_assessment": {"concern": "medical_emergency", "subject": "other",
                            "time": "current", "urgency": "immediate"}})
    clients = MinimalModelClients(online, MockLLMClient("memory"), MockLLMClient("reflection"),
                                  True, safety_review=reviewer)
    wires = []
    value = {**payload("medical_emergency", evidence(TEXT, "other", "current", "immediate"), BODY),
             "applied_preference_ids": []}
    def receive(request, **kwargs):
        wires.append(json.loads(request.data))
        envelope_data = {"id": "gen-callback-software", "model": MODEL, "provider": "Alibaba",
            "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]}
        class Response:
            status = 200
            headers = {}
            def __init__(self): self.body = io.BytesIO(json.dumps(envelope_data).encode())
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1): return self.body.read(size)
        return Response()
    stub_model_http(monkeypatch, receive)
    app = MinimalWebApp(tmp_path, clients=clients, safety_delivery_version="minimal_response_v1",
        safety_response_instruction_version="compact_response_v1", premodel_safety_version="task_response_v4",
        model_max_requests=5, model_stop_cost_usd="1", model_reservation_policy=policy(online))
    frames = []
    try:
        sid = app.start()["state"]["session_id"]
        engine = app._sessions["fictional-chen"][0]
        monkeypatch.setattr(engine._risk, "assess", lambda ctx: RiskSignal(tier=3, kind="medical_emergency_fall", evidence=[]))
        result = app.submit(TEXT, "keyword-relabel", sid,
            on_safety_prelude=lambda frame: frames.append(deepcopy(frame)) or True)
        assert result["ok"] and len(frames) == len(wires) == 1
        shown = frames[0]["state"]["turns"][-1]["protection_prelude"]
        saved = json.loads(app.run_path.read_text(encoding="utf-8"))["turns"][-1]
        assert saved["protection_prelude"]["protection"] == shown["protection"]
        assert saved["protection_prelude"]["status"] == "transport_returned"
        assert "心肺复苏" not in shown["protection"] and "心肺复苏" in saved["response"]
        assert BODY in saved["response"] and len(reviewer.messages) == 1
        assert shown["protection"] not in saved["response"]
        reviewed_input = json.loads(reviewer.messages[0][1]["content"])
        assert reviewed_input["candidate"] == BODY
        assert reviewed_input["protection_already_delivered"] == shown["protection"]
        audit = saved["safety_review"]
        assert audit["status"] == "reviewed" and audit["passed_to_client"] is True
        assert audit["model_body_delivered"] is True
        assert audit["protection_sha256"] == shown["protection_sha256"]
        from aria.engine.risk.safety_delivery import render_review_protection
        reviewed_protection, _ = render_review_protection(reviewer.result["risk_assessment"], reviewed_fields=True)
        final_protection = shown["protection"] + "\n" + reviewed_protection
        assert audit["final_protection_sha256"] == hashlib.sha256(final_protection.encode()).hexdigest()
        assert len(app.call_guard.snapshot()["attempts"]) == 1
    finally:
        app._sessions.clear()
        app.close()


def test_review_gets_exact_unpaired_fictional_history(monkeypatch):
    from aria.engine.policy.fictional_context import bind_fictional_conversation_history
    reviewer = Reviewer()
    engine, ctx, _ = build(monkeypatch, payload("fall", evidence(TEXT, "other", "current", "immediate"), BODY), reviewer)
    history = [{"role": "user", "content": "先前提供的能力背景，仅为软件输入。"}]
    bind_fictional_conversation_history(ctx, history)
    engine.process_turn(ctx, TEXT, safety_prelude=lambda e: True)
    data = json.loads(reviewer.messages[0][1]["content"])
    assert data["safety_evidence"]["conversation_history"] == history


def test_sentence_split_is_lossless_and_indices_are_strict():
    value = "中文。\n\nEnglish first! next? end；尾声"
    pieces = response_sentences(value)
    assert "".join(pieces) == value
    messages = safety_review_messages(TEXT, value, {"concern": "fall"}, "software protection")
    assert json.loads(messages[1]["content"])["sentences"] == [{"id": i, "text": s} for i, s in enumerate(pieces)]
    with pytest.raises(ValueError): parse_safety_review('{"conflicts":true,"delete_sentence_ids":[-1]}', len(pieces))


def test_medical_scope_keeps_safe_body_and_local_refusal_without_rewriting_old_version():
    text = "不能一次吃两片补回来。漏服后擅自加倍服用可能导致血压骤降，引发头晕、乏力甚至跌倒等危险。请核对说明并咨询药师。"
    assert guard_general_medical_reply(text, version="sentence_refusal_v3") == text
    assert guard_general_medical_reply(text, version="local_refusal_v2") != text
    text = "这句普通陪伴保留。可以服双倍药量。最后这句也保留。"
    result = guard_general_medical_reply(text, version="sentence_refusal_v3")
    assert "这句普通陪伴保留。" in result and "最后这句也保留。" in result
    assert "可以服双倍药量。" not in result


@pytest.mark.parametrize("text", ["不能说不能一次吃两片，可以吃双倍药量。", "不要自行停药，但是可以服双倍药量。"])
def test_refusal_does_not_mask_later_unsafe_advice(text):
    assert guard_general_medical_reply(text, version="sentence_refusal_v3") != text
