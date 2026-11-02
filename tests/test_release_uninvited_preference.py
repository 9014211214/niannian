"""Forced control labels verify delivery/state invariants, not model semantics."""
from copy import deepcopy
import json

import pytest

from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import apply_preference_dialogue, preference_fingerprint
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from tests.test_minimal_response_v1 import build, TEXT
from tests.test_preference_dialogue_readiness import fixture, control
from tests.test_safety_evidence import payload, evidence, V4
from tests.test_uninvited_preference_control import uninvited


@pytest.mark.parametrize('action', ['confirm', 'decline', 'defer', 'unclear'])
@pytest.mark.parametrize('body', [
    '步骤甲\n步骤乙\n步骤丙',
    '可以把文件保存到你选择的位置；不合适的邀请也可以拒绝。',
    '我先陪着你，按你想说的来。',
])
def test_uninvited_controls_cannot_replace_ordinary_body_or_grant_state(monkeypatch, action, body):
    value = payload(response=body)
    value['preference_control'] = uninvited(action)
    # Values that name an invented target still cannot create a real invitation.
    value['preference_control'].update(invitation_id='invented', target_id='invented',
                                       target_sha256='a' * 64, evidence_quote=TEXT)
    engine, ctx, wires, _ = build(monkeypatch, value)
    assert engine.process_turn(ctx, TEXT) == body
    trace = ctx.working_memory['online_turn_decision']
    assert trace['preference_control_status'] == 'missing_or_invalid'
    assert trace['preference_control'] is None
    assert trace['preference_control_inert'] is True
    assert trace['preference_control_ignored_reason'] == 'no_invitation'
    assert not engine.persona.stated_preferences
    assert 'preference_dialogue_receipt' not in ctx.working_memory
    assert not ctx.working_memory.get('_sensitive_memory_excluded_turns')
    assert len(wires) == 1


@pytest.mark.parametrize('body', [
    '我已经保存了这条相处说明。',
    '这条交流偏好已经启用了。',
    '我已经拒绝了这条相处说明。',
])
def test_uninvited_control_cannot_publish_explicit_preference_completion(monkeypatch, body):
    value = payload(response=body)
    value['preference_control'] = uninvited('confirm')
    engine, ctx, _, _ = build(monkeypatch, value)
    assert engine.process_turn(ctx, TEXT) != body
    assert ctx.working_memory['delivery_response_guard']['violations'] == ['unverified_preference_completion']
    assert not engine.persona.stated_preferences
    assert 'preference_dialogue_receipt' not in ctx.working_memory


@pytest.mark.parametrize('body', [
    '我还没有保存这条相处说明。',
    '如果我保存了这条相处说明，再由你核对。',
    '你可以拒绝这条交流偏好。',
    '我会先解释相处说明。\n保存文件时，请选择你自己的目录。',
])
def test_preference_receipt_backstop_does_not_turn_words_into_a_transaction(monkeypatch, body):
    value = payload(response=body)
    value['preference_control'] = uninvited('decline')
    engine, ctx, _, _ = build(monkeypatch, value)
    assert engine.process_turn(ctx, TEXT) == body
    assert not ctx.working_memory['delivery_response_guard']['activated']


@pytest.mark.parametrize('edge', ['unack', 'expired', 'owner', 'session', 'turn', 'target', 'question', 'simulation'])
def test_existing_invitation_context_cannot_use_uninvited_exception(edge):
    persona, ctx = fixture()
    ctx.working_memory.update(_online_decision_version=V4, _safety_delivery_version='minimal_response_v1')
    value = payload(response='普通正文。')
    value['preference_control'] = control(persona, ctx)
    pending = ctx.working_memory['_preference_dialogue_invitation']
    if edge == 'unack': pending['delivered'] = False
    if edge == 'expired': pending['expires_at'] = '2000-01-01T00:00:00+00:00'
    if edge == 'owner': ctx.user_id = 'other'
    if edge == 'session': ctx.session_id = 'other'
    if edge == 'turn': pending['turn_index'] += 1
    if edge == 'target': pending['target_sha256'] = 'different'
    if edge == 'question': pending['question_sha256'] = 'different'
    if edge == 'simulation': ctx.simulation_mode = True
    before = deepcopy(persona.model_dump())
    apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, '本人当前回答', persona)
    assert '还没有确认保存' in apply_preference_dialogue(ctx, persona, '本人当前回答')
    assert persona.model_dump() == before
    assert 'preference_dialogue_receipt' not in ctx.working_memory
    assert not ctx.working_memory['online_turn_decision'].get('preference_control_ignored_reason')


@pytest.mark.parametrize('action,status', [('confirm', 'confirmed'), ('decline', 'declined')])
def test_real_invitation_still_requires_commit_before_publish(tmp_path, action, status):
    persona, ctx = fixture()
    ctx.working_memory.update(_online_decision_version=V4, _safety_delivery_version='minimal_response_v1')
    value = payload(response='普通正文。')
    value['preference_control'] = control(persona, ctx, action=action)
    apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, '本人当前回答', persona)
    store = PersonaStore(tmp_path)
    assert apply_preference_dialogue(ctx, persona, '本人当前回答', store=store)
    assert store.load(persona.user_id).stated_preferences[0].status == status
    assert ctx.working_memory['preference_dialogue_receipt']['persisted'] is True


@pytest.mark.parametrize('body', ['我已经帮你发给家人了。', '可以服双倍药量。', '你肯定患了阿尔茨海默症。'])
def test_ignored_control_does_not_skip_existing_output_guards(monkeypatch, body):
    value = payload(response=body)
    value['preference_control'] = uninvited('confirm')
    engine, ctx, _, _ = build(monkeypatch, value)
    assert engine.process_turn(ctx, TEXT) != body
    assert not engine.persona.stated_preferences


def test_ignored_control_does_not_clear_urgent_safety(monkeypatch):
    value = payload('fall', evidence(TEXT, 'owner', 'current', 'immediate'), '中性正文。')
    value['preference_control'] = uninvited('confirm')
    engine, ctx, _, _ = build(monkeypatch, value)
    engine.process_turn(ctx, TEXT)
    assert ctx.working_memory['intervention_delivery']['reason'] == 'semantic_safety_takeover'
    assert not ctx.working_memory['online_turn_decision'].get('preference_control_ignored_reason')
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]


@pytest.mark.parametrize('tamper', [None, 'owner', 'session', 'source', 'row', 'action', 'persisted', 'version', 'status'])
def test_named_preference_claim_requires_the_actual_matching_receipt(tmp_path, tamper):
    persona, ctx = fixture()
    ctx.working_memory.update(_online_decision_version=V4, _safety_delivery_version='minimal_response_v1')
    value = payload(response='普通正文。')
    value['preference_control'] = control(persona, ctx, text=TEXT)
    apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, TEXT, persona)
    assert apply_preference_dialogue(ctx, persona, TEXT, store=PersonaStore(tmp_path))
    ctx.working_memory['online_turn_decision'] = dict(turn_index=0,
        preference_control_ignored_reason='no_invitation')
    receipt = ctx.working_memory['preference_dialogue_receipt']
    if tamper == 'owner': receipt['owner_id'] = 'other'
    if tamper == 'session': receipt['session_id'] = 'other'
    if tamper == 'source': receipt['owner_message_sha256'] = 'different'
    if tamper == 'row': receipt['committed_sha256'] = 'different'
    if tamper == 'action': receipt['action'] = 'decline'
    if tamper == 'persisted': receipt['persisted'] = False
    if tamper == 'version': receipt['version'] = 'different'
    if tamper == 'status':
        persona.stated_preferences[0].status = 'proposed'
        receipt['committed_sha256'] = preference_fingerprint(persona.stated_preferences[0])
    body = '我已经保存了这条相处说明。'
    ctx.working_memory['_delivery_response_candidate_for_turn'] = body
    assert (guard_external_delivery_response(ctx, body, TEXT, persona) == body) is (tamper is None)
