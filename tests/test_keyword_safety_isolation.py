"""Safety routing/privacy with existing backstop phrases; no semantic scores."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import hashlib
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile, SafetyScreeningRecord
from aria.engine.memory.transparency import render_profile
from tests.test_semantic_safety import engine_with_response
from tests.test_minimal_web import live_fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('no external services in software checks')
    stub_model_http(monkeypatch, fail)


@pytest.mark.parametrize('message', [
    '我现在胸口痛，喘不上气。',
    '我邻居现在胸口痛，喘不上气。',
    '不想活了。',
])
def test_keyword_risk_is_excluded_from_checkpoint_and_offline_queue(tmp_path, message):
    app, clients = live_fixture(tmp_path)
    try:
        sid = app.start()['state']['session_id']
        assert app.submit(message, 'keyword-risk-turn', sid)['ok']
        p = app._profile()
        assert len(p.safety_screenings) == 1 and not p.risk_events
        assert p.safety_screenings[0].evidence_sha256 == hashlib.sha256(message.encode()).hexdigest()
        assert p.safety_screenings[0].subject == 'unverified'
        assert p.pending_session is None or p.pending_session.turns == []
        ctx = app._sessions['fictional-chen'][1]
        assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]
        assert ctx.working_memory['intervention_delivery']['status'] == 'not_delivered'
        before = deepcopy(p.safety_screenings)
        app._sessions['fictional-chen'][0].checkpoint_session(ctx)
        assert app._profile().safety_screenings == before
        assert app.seal(sid)['ok']
        job, = app._profile().offline_sessions
        assert job.excluded_turn_indices == [0]
        assert job.turns[0].user == job.turns[0].agent == ''
        assert not any(c.calls for c in (clients.online, clients.memory, clients.reflection))
    finally:
        app.close()


def test_backstop_audit_cannot_promote_subject_or_contain_raw_text():
    value = dict(at='2026-10-08T00:00:00', session_id='test', turn_index=0,
                 tier=3, kind='medical_emergency', evidence_sha256='a'*64)
    row = SafetyScreeningRecord(**value)
    assert 'unverified' in row.model_dump_json()
    for change in ({'subject': 'owner'}, {'raw_utterance': 'PRIVATE'}, {'source': 'confirmed_person'}):
        with pytest.raises(ValueError):
            SafetyScreeningRecord(**{**value, **change})


def test_legacy_profile_load_and_neutral_transparency_do_not_relabel_old_records():
    legacy = {'user_id': 'owner', 'risk_events': [
        {'at': '2026-10-01', 'tier': 2, 'kind': 'historical', 'note': '旧记录'}]}
    p = PersonaProfile.model_validate(legacy)
    assert not p.safety_screenings and p.risk_events[0].note == '旧记录'
    text = render_profile(p)
    assert '不证明本人危险' in text and '我担心过你的时刻' not in text
    loaded = PersonaProfile.model_validate_json(p.model_dump_json())
    assert loaded.risk_events == p.risk_events and loaded.safety_screenings == []


def test_same_turn_dedup_and_later_clear_turn_do_not_inherit_keyword_risk(monkeypatch):
    engine, calls = engine_with_response(monkeypatch, 'none', '我们接着聊风车。')
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert '120' in engine.process_turn(ctx, '我邻居现在胸口痛，喘不上气。')
    assert not calls and len(engine.persona.safety_screenings) == 1
    record = engine.persona.safety_screenings[0]
    signal = engine._risk.assess(ctx)
    # Explicit replay of the same source identity cannot append another row.
    ctx.turns.pop()
    engine._record_safety_screening(ctx, signal, '我邻居现在胸口痛，喘不上气。')
    assert engine.persona.safety_screenings == [record]
    ctx.add_turn('我邻居现在胸口痛，喘不上气。', '既有安全回复')
    assert engine.process_turn(ctx, '那边已由医生处理好了，现在聊风车。') == '我们接着聊风车。'
    assert len(calls) == 1 and 'risk_flag' not in ctx.working_memory
    assert 'safety_screening_audit' not in ctx.working_memory
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]


def test_audit_validation_cannot_prevent_urgent_response(monkeypatch):
    engine, calls = engine_with_response(monkeypatch, 'none')
    ctx = EngineContext(user_id='owner', scene_id='companion_elder', session_id='x'*129)
    assert '120' in engine.process_turn(ctx, '我现在胸口痛，喘不上气。')
    assert not calls and not engine.persona.safety_screenings
    assert ctx.working_memory['safety_screening_audit']['status'] == 'not_recorded'
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]
