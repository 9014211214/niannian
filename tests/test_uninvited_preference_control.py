"""Preset protocol values test state boundaries, not model language accuracy."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.preference_dialogue import apply_preference_dialogue
from aria.engine.policy.online_decision import apply_single_pass_decision
from tests.test_safety_evidence import engine_for, payload, evidence, V4
from tests.test_preference_dialogue_readiness import fixture


def uninvited(action='unclear'):
    return dict(version='preference_confirmation_v1', action=action,
                invitation_id='', target_id='', target_sha256='', evidence_quote='')


def test_v4_uninvited_unclear_preserves_reply_without_profile_authority(monkeypatch):
    value = payload(response='我们接着讨论风筝的做法。')
    value['preference_control'] = uninvited()
    engine, calls = engine_for(monkeypatch, [value])
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, '说说这个手工。') == value['response']
    trace = ctx.working_memory['online_turn_decision']
    assert trace['preference_control_inert'] is True
    assert trace['preference_control_status'] == 'missing_or_invalid'
    assert trace['preference_control'] is None
    assert not engine.persona.stated_preferences
    assert 'preference_dialogue_receipt' not in ctx.working_memory
    assert not ctx.working_memory.get('_sensitive_memory_excluded_turns')
    assert len(calls) == 1


@pytest.mark.parametrize('action', ['confirm', 'decline', 'defer'])
def test_state_actions_without_invitation_still_cannot_publish_confirmation(monkeypatch, action):
    value = payload(response='我已经保存了。')
    value['preference_control'] = uninvited(action)
    engine, calls = engine_for(monkeypatch, [value])
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert '还没有确认保存' in engine.process_turn(ctx, '说说这个手工。')
    assert not engine.persona.stated_preferences and len(calls) == 1
    assert ctx.working_memory['online_turn_decision']['preference_control_inert'] is False


@pytest.mark.parametrize('invitation_state', ['valid', 'unacknowledged', 'expired'])
def test_unclear_with_invitation_context_retains_existing_guard(invitation_state):
    persona, ctx = fixture(acknowledge=invitation_state != 'unacknowledged')
    ctx.working_memory['_online_decision_version'] = V4
    pending = ctx.working_memory['_preference_dialogue_invitation']
    value = uninvited()
    value.update(invitation_id=pending['id'], target_id=pending['target_id'],
                 target_sha256=pending['target_sha256'], evidence_quote='本人当前回答')
    if invitation_state == 'expired': pending['expires_at'] = '2000-01-01T00:00:00+00:00'
    before = deepcopy(persona.model_dump())
    decision = payload()
    decision['preference_control'] = value
    apply_single_pass_decision(json.dumps(decision, ensure_ascii=False), ctx, '本人当前回答', persona)
    trace = ctx.working_memory['online_turn_decision']
    assert trace['preference_control_inert'] is False
    reply = apply_preference_dialogue(ctx, persona, '本人当前回答')
    assert reply and persona.model_dump() == before
    assert 'preference_dialogue_receipt' not in ctx.working_memory


def test_unclear_noop_cannot_clear_safety(monkeypatch):
    value = payload('fall', evidence('此刻有危险', 'owner', 'current', 'immediate'), '未经安全处理')
    value['preference_control'] = uninvited()
    engine, calls = engine_for(monkeypatch, [value])
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, '此刻有危险') != value['response']
    assert len(calls) == 1 and ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]
    assert 'preference_dialogue_receipt' not in ctx.working_memory
