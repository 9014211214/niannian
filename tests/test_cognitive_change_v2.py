"""C1 local software evidence; synthetic fixtures are not model/medical results."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from aria.engine.cognitive import (
    capture_cognitive_turn, cognitive_action, cognitive_sources,
    evaluate_cognitive_trend, extract_cognitive_rows, online_cognitive_block,
)
from aria.engine.context import EngineContext, Turn
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import CognitiveObservationRecord, PersonaProfile, PersonEvidenceReview
from aria.engine.policy.prompt import build_messages
from aria.engine.orchestrator import Engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from tests.test_minimal_care_web import care_server, call, login

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
READING = {"label": "calm", "confidence": .8, "modality": "text", "source": "external_test"}


class Payload:
    def __init__(self, rows=()):
        self.rows, self.calls = list(rows), []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return json.dumps({"episodes": [], "relationships": [], "stated_preferences": [],
                           "cognitive_privacy": [],
                           "cognitive_observations": self.rows}, ensure_ascii=False)


def enabled():
    persona = PersonaProfile(user_id="owner")
    cognitive_action(persona, "enable", actor="owner", now=T0)
    return persona


def record(persona, day, domain="memory", impact=False, index=0):
    at = (T0 + timedelta(days=day)).isoformat()
    row = CognitiveObservationRecord(
        id=f"obs-{day}-{domain}-{index}", at=at, session_id=f"session-{day}-{index}", domain=domain,
        signal="explicit_change", functional_impact=impact,
        source_quote="我忘了交电费，耽误了事情。" if impact else "我又想不起东西放哪儿了。",
        source_turn_index=0, consent_at=persona.cognitive_health_consent.text_consented_at,
    )
    persona.cognitive_observations.append(row)
    return row


def calibrated(persona):
    persona.cognitive_trend.session_times = {f"s{i}": (T0+timedelta(days=i*2)).isoformat() for i in range(6)}
    evaluate_cognitive_trend(persona, now=T0+timedelta(days=14))
    return persona


def context_with(persona, text="我又想不起钥匙放哪儿了。", *, now=None):
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    capture_cognitive_turn(ctx, persona, text, now=now or T0+timedelta(days=1))
    ctx.add_turn(text, "我听见了。")
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    return ctx


def test_consent_defaults_and_explicit_actor_only():
    p = PersonaProfile(user_id="owner")
    assert not p.cognitive_health_consent.has_text_monitoring_consent()
    with pytest.raises(PermissionError):
        cognitive_action(p, "enable", actor="caregiver", now=T0)
    assert not p.cognitive_health_consent.text_monitoring_enabled
    cognitive_action(p, "enable", actor="owner", now=T0)
    with pytest.raises(ValueError):
        cognitive_action(p, "disable", actor="owner", now=T0)
    assert p.cognitive_health_consent.has_text_monitoring_consent()


def test_enable_does_not_retrofit_earlier_speech():
    p = PersonaProfile(user_id="owner")
    ctx = context_with(p)
    cognitive_action(p, "enable", actor="owner", now=T0)
    assert cognitive_sources(p, ctx) == []


def test_disabled_extractor_prompt_and_online_prompt_unchanged():
    p = PersonaProfile(user_id="owner")
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("我喜欢纸风车。", "真有意思。")
    first, second = Payload(), Payload()
    LLMExtractor(first).extract(ctx, p)
    p.cognitive_observations = [record(enabled(), 20)]
    LLMExtractor(second).extract(ctx, p)
    assert first.calls == second.calls
    assert "cognitive_observations" not in first.calls[0][0]["content"]
    normal = build_messages(get_scene("companion_elder"), p, ctx, "今天挺好。")
    p.cognitive_trend.status = "review_suggested"
    ctx.emotion.external_reading = READING
    # Compare same external input; switching a disabled trend must not add C1.
    with_reading = build_messages(get_scene("companion_elder"), p, ctx, "今天挺好。")
    p.cognitive_trend.status = "disabled"
    assert build_messages(get_scene("companion_elder"), p, ctx, "今天挺好。") == with_reading
    assert "可以温和提起的观察" not in str(normal)


@pytest.mark.parametrize("quote", ["我又想不起钱包放哪儿了。", "钥匙放在哪里了", "又想不起钥匙放哪儿了！", ""])
def test_only_literal_source_quote_is_accepted(quote):
    p = enabled()
    ctx = context_with(p)
    assert extract_cognitive_rows([{"domain": "memory", "evidence_quote": quote, "turn_index": 0}], p, ctx) == []


@pytest.mark.parametrize("text", ["我妈又想不起钥匙放哪儿了。", "如果我忘了交电费怎么办？", "我没有忘交电费。", "护理提醒未确认。", "我又想不起钥匙了，这个别保存。"])
def test_non_owner_negation_care_status_and_do_not_save_are_not_signals(text):
    p = enabled()
    ctx = context_with(p, text)
    rows = [{"domain": "memory", "evidence_quote": text, "turn_index": 0}]
    assert extract_cognitive_rows(rows, p, ctx) == []


@pytest.mark.parametrize("text", ["老了记性差，我一时想不起钥匙放哪。", "我没睡好，想不起东西放哪儿了。"])
def test_context_is_visible_but_does_not_count(text):
    p = enabled()
    ctx = context_with(p, text)
    result = extract_cognitive_rows([{"domain": "memory", "evidence_quote": text, "turn_index": 0}], p, ctx)
    assert len(result) == 1 and result[0].counting is False and not result[0].functional_impact


def test_functional_impact_uses_model_semantics_without_keyword_prerequisite():
    p = enabled()
    ctx = context_with(p)
    result = extract_cognitive_rows([{"domain": "memory", "evidence_quote": ctx.turns[0].user,
                                     "turn_index": 0, "functional_impact": True}], p, ctx)
    assert len(result) == 1 and result[0].functional_impact
    # This tests applying a synthetic semantic annotation, not its correctness.
    assert result[0].source_quote == ctx.turns[0].user


def test_same_event_has_c1_identity_without_personality_ledger():
    p = enabled()
    ctx = context_with(p)
    rows = extract_cognitive_rows([{"domain": "repetition", "evidence_quote": ctx.turns[0].user,
                                   "turn_index": 0, "event_kind": "same_event", "prior_event_id": "invented"}], p, ctx)
    assert len(rows) == 1 and not rows[0].counting
    assert rows[0].event_id != "invented"


def test_repetition_requires_three_known_same_events_within_seven_days():
    p = enabled()
    words = "昨天我在门口捡到了一个蓝色信封。"
    p.evidence_review_ledger.append(PersonEvidenceReview(
        id="old-review", at=T0.isoformat(), session_id="old", applicability="bounded", relation="support",
        evidence_basis="user_quote", event_kind="new_event", event_id="known-envelope", reason="fixture",
        event_summary="fixture event", event_evidence=[{"turn_index": 0, "quote": words}]))
    for day in (1, 3, 5):
        ctx = context_with(p, words, now=T0+timedelta(days=day))
        result = extract_cognitive_rows([{"domain": "repetition", "evidence_quote": words, "turn_index": 0,
                                          "event_kind": "same_event", "prior_event_id": "known-envelope"}], p, ctx)
        assert len(result) == 1
        assert result[0].counting is (day == 5)
        p.cognitive_observations.extend(result)
    old = context_with(p, words, now=T0+timedelta(days=20))
    result = extract_cognitive_rows([{"domain": "repetition", "evidence_quote": words, "turn_index": 0,
                                      "event_kind": "same_event", "prior_event_id": "known-envelope"}], p, old)
    assert not result[0].counting


def test_calibration_requires_both_time_and_sessions():
    p = enabled()
    for day in range(1, 7):
        p.cognitive_trend.session_times[f"s{day}"] = (T0+timedelta(days=day)).isoformat()
        record(p, day, "daily_function", True)
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=13, hours=23)).status == "calibrating"
    p.cognitive_trend.session_times.pop("s6")
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=30)).status == "calibrating"
    p.cognitive_trend.session_times["s6"] = (T0+timedelta(days=7)).isoformat()
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=30)).status == "watch"


def test_growth_gate_and_two_day_impact_gate_do_not_depend_on_model_severity():
    p = calibrated(enabled())
    for day in (15, 16, 17, 18):
        for domain in ("memory", "language"):
            record(p, day, domain)
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=19)).status == "watch"  # 8/30 <2/week
    record(p, 19)
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=19)).status == "review_suggested"
    q = calibrated(enabled())
    record(q, 15, "daily_function", True)
    record(q, 15, "memory", True)
    assert evaluate_cognitive_trend(q, now=T0+timedelta(days=16)).status == "watch"
    record(q, 16, "daily_function", True)
    assert evaluate_cognitive_trend(q, now=T0+timedelta(days=16)).status == "review_suggested"


def test_cooldown_first_30_then_90_days_and_prompt_remains_empty():
    p = calibrated(enabled())
    record(p, 15, "daily_function", True)
    record(p, 16, "daily_function", True)
    evaluate_cognitive_trend(p, now=T0+timedelta(days=16))
    cognitive_action(p, "decline", actor="owner", now=T0+timedelta(days=16))
    assert p.cognitive_trend.cooldown_until == (T0+timedelta(days=46)).isoformat()
    p.cognitive_trend.status = "review_suggested"  # Even a stale status cannot bypass cooling.
    assert online_cognitive_block(p, "今天天气好。", [Turn("你好", "你好")], READING, now=T0+timedelta(days=20)) == ""
    cognitive_action(p, "decline", actor="owner", now=T0+timedelta(days=20))
    assert p.cognitive_trend.cooldown_until == (T0+timedelta(days=110)).isoformat()


def test_owner_arranged_choice_is_only_cooldown_not_a_care_receipt():
    p = calibrated(enabled())
    before = deepcopy(p.care_tasks)
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    capture_cognitive_turn(ctx, p, "我已经设好了体检提醒", now=T0+timedelta(days=16))
    assert p.cognitive_trend.status == "watch" and p.cognitive_trend.cooldown_until
    assert p.care_tasks == before


@pytest.mark.parametrize("words,history,reading", [
    ("今天挺好。", [], READING), ("我很难过。", [Turn("你好", "你好")], READING),
    ("帮我取消提醒。", [Turn("你好", "你好")], READING),
    ("今天挺好。", [Turn("你好", "你好")], None),
    ("今天挺好。", [Turn("你好", "你好")], {**READING, "label": "unknown"}),
    ("今天挺好。", [Turn("你好", "你好")], {**READING, "label": "sad"}),
    ("今天挺好。", [Turn("你好", "你好")], {**READING, "confidence": .1}),
    ("今天挺好。", [Turn("你好", "你好")], {**READING, "source": "simulated_test"}),
])
def test_online_waits_for_conservative_timing(words, history, reading):
    p = calibrated(enabled())
    record(p, 15, "daily_function", True); record(p, 16, "daily_function", True)
    evaluate_cognitive_trend(p, now=T0+timedelta(days=16))
    assert not online_cognitive_block(p, words, history, reading, now=T0+timedelta(days=16))


def test_appropriate_online_block_has_literal_quotes_and_no_score():
    p = calibrated(enabled())
    row = record(p, 15, "daily_function", True); record(p, 16, "daily_function", True)
    evaluate_cognitive_trend(p, now=T0+timedelta(days=16))
    block = online_cognitive_block(p, "今天挺好。", [Turn("你好", "你好")], READING, now=T0+timedelta(days=16))
    assert row.source_quote in block and "可以拒绝" in block and "通知家人" in block


def test_memory_one_call_private_sources_never_enter_reflection_or_ordinary_memory(tmp_path):
    p = enabled()
    ctx = context_with(p)
    client = Payload([{"domain": "memory", "evidence_quote": ctx.turns[0].user, "turn_index": 0}])
    store = PersonaStore(tmp_path)
    queue = OfflineSessionQueue(p, store, now=lambda: T0+timedelta(days=1))
    assert queue.enqueue(ctx)["status"] == "queued"
    assert queue.persona.offline_sessions[0].turns[0].user == ""
    reflection_seen = []

    def memory(profile, offline):
        MemoryConsolidator(store, LLMExtractor(client)).consolidate(profile, offline, persist=False)
        return profile, "ok", None

    def reflection(profile, offline):
        reflection_seen.append(deepcopy((profile, offline)))
        return profile, "ok", None

    assert queue.drain(memory, reflection)["completed_jobs"] == 1
    assert len(client.calls) == 1
    assert len(queue.persona.cognitive_observations) == 1
    assert queue.persona.episodic_memory == []
    reflection_profile, reflection_ctx = reflection_seen[0]
    assert reflection_profile.cognitive_observations == []
    assert "_cognitive_source_turns" not in reflection_ctx.working_memory
    assert all(not turn.user for turn in reflection_ctx.turns)
    assert len(store.load("owner").cognitive_observations) == 1


def test_delete_tombstone_prevents_reextraction_and_disable_clears_raw(tmp_path):
    p = enabled(); ctx = context_with(p)
    rows = [{"domain": "memory", "evidence_quote": ctx.turns[0].user, "turn_index": 0}]
    p.cognitive_observations = extract_cognitive_rows(rows, p, ctx)
    queue = OfflineSessionQueue(p, PersonaStore(tmp_path), now=lambda: T0+timedelta(days=1))
    assert queue.enqueue(ctx)["status"] == "queued"
    p = queue.persona
    cognitive_action(p, "delete", actor="owner", observation_id=p.cognitive_observations[0].id, now=T0+timedelta(days=1))
    assert p.offline_sessions[0].cognitive_turns == []
    assert extract_cognitive_rows(rows, p, ctx) == []
    cognitive_action(p, "disable", actor="owner", delete_records=True, now=T0+timedelta(days=2))
    assert p.cognitive_observations == [] and p.offline_sessions[0].cognitive_turns == []
    assert cognitive_sources(p, ctx) == []


def test_web_owner_controls_reload_no_caregiver_authority_and_read_only(tmp_path):
    app = MinimalWebApp(tmp_path)
    assert app.state()["cognitive"]["enabled"] is False
    with pytest.raises(PermissionError):
        app.cognitive_action("enable", actor="caregiver")
    assert not app.cognitive_action("enable", read_only=True)["ok"]
    assert app.cognitive_action("enable")["ok"]
    assert MinimalWebApp(tmp_path).state()["cognitive"]["enabled"]
    assert not app.cognitive_action("disable")["ok"]
    assert app.cognitive_action("disable", delete_records=False)["ok"]
    assert not MinimalWebApp(tmp_path).state()["cognitive"]["enabled"]
    assert "cognitive" not in app.care_state()
    app.close()


def test_http_care_route_cannot_grant_cognitive_consent_or_read_notes(care_server):
    app, server = care_server
    cookie = login(app, server)
    status, _, _ = call(app, server, "/api/care/action", cookie=cookie, data={
        "elder_id": "fictional-chen", "action": "cognitive.enable", "payload": {"request_id": "c1-rejected"}})
    assert status in {400, 403}
    assert not app.state()["cognitive"]["enabled"]
    assert call(app, server, "/api/minimal/cognitive", token=False, data={"action": "enable"})[0] == 403
    assert call(app, server, "/api/minimal/cognitive", data={"action": "enable", "replay": True})[0] == 409
    assert call(app, server, "/api/minimal/cognitive", data={"action": "enable"})[0] == 200
    p = app._profile()
    p.cognitive_observations.append(CognitiveObservationRecord(
        id="private", at=T0.isoformat(), session_id="old", domain="memory", signal="change", source_quote="这句只在本人面板。"))
    app.store.save(p)
    status, _, body = call(app, server, "/api/care/state", cookie=cookie)
    assert status == 200 and "这句只在本人面板" not in json.dumps(body, ensure_ascii=False)
    assert "cognitive" not in body


def test_consent_checkpoint_restart_restores_private_channel_only(tmp_path):
    app = MinimalWebApp(tmp_path)
    assert app.cognitive_action("enable")["ok"]
    sid = app.start()["state"]["session_id"]
    assert app.submit("我又想不起钥匙放哪儿了。", "one", sid)["ok"]
    saved = app.store.load("fictional-chen")
    assert saved.pending_session.turns == []
    assert len(saved.pending_session.cognitive_turns) == 1
    restored = MinimalWebApp(tmp_path)
    assert restored.seal(sid)["ok"]
    saved = restored.store.load("fictional-chen")
    assert saved.offline_sessions[0].turns[0].user == ""
    assert len(saved.offline_sessions[0].cognitive_turns) == 1
    restored.close()


def test_storage_failure_never_claims_consent_was_enabled(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path)
    monkeypatch.setattr(app.store, "save", lambda *_: (_ for _ in ()).throw(OSError("fixture")))
    result = app.cognitive_action("enable")
    assert not result["ok"] and not result["state"]["cognitive"]["enabled"]
    app.close()


def test_enabled_global_guard_and_emergency_priority():
    p = enabled()
    class Reply:
        def complete(self, _messages):
            return "您有70%概率患阿尔茨海默。"
    engine = Engine(get_scene("companion_elder"), persona=p, llm_client=Reply())
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    response = engine.process_turn(ctx, "今天挺好的。")
    assert "70%" not in response and "阿尔茨海默" not in response
    response = engine.process_turn(ctx, "我刚才突然不知道自己在哪。")
    assert "120" in response and "阿尔茨海默" not in response
    assert [source.text for source in cognitive_sources(p, ctx)] == ["今天挺好的。"]


def test_natural_decline_and_bound_semantic_disable_use_owner_current_words():
    p = calibrated(enabled()); ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    ctx.add_turn("你好", "要不要下次体检时和医生说一句？")
    capture_cognitive_turn(ctx, p, "不用了", now=T0+timedelta(days=20))
    assert p.cognitive_trend.declines == 1
    text = "这种事别提醒我"
    import hashlib
    ctx.working_memory["online_turn_decision"] = {
        "version": "online_turn_decision_v3", "status": "parsed", "turn_index": len(ctx.turns),
        "owner_message_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "cognitive_control": {"version": "cognitive_permission_v1", "action": "disable_observation", "evidence_quote": text},
    }
    capture_cognitive_turn(ctx, p, text, now=T0+timedelta(days=21))
    assert not p.cognitive_health_consent.has_text_monitoring_consent()


def test_cognitive_authorization_never_authorizes_a_family_draft():
    p = enabled()
    record(p, 15, "daily_function", True)
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    safe = scene.guard_delivery_response(ctx, "给女儿的草稿：我忘了交电费，耽误了事情。", "写点什么给女儿吧", p)
    assert "我忘了交电费" not in safe
    assert "cognitive_private_sharing_blocked" in ctx.working_memory["delivery_response_guard"]["violations"]


def test_deleted_observation_recomputes_review_state_and_reading_is_read_only(tmp_path):
    p = calibrated(enabled())
    first = record(p, 15, "daily_function", True)
    record(p, 16, "daily_function", True)
    assert evaluate_cognitive_trend(p, now=T0+timedelta(days=16)).status == "review_suggested"
    cognitive_action(p, "delete", actor="owner", observation_id=first.id, now=T0+timedelta(days=16))
    assert p.cognitive_trend.status == "watch"
    store = PersonaStore(tmp_path)
    store.save(p)
    before = {str(path):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    for _ in range(3):
        assert store.load("owner").cognitive_trend.status == "watch"
    assert before == {str(path):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
