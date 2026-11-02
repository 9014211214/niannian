"""Explicit model labels exercise engineering; these are not semantic results."""
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION_EVIDENCE as V4,
    ONLINE_DECISION_VERSION_SAFETY as V3,
    online_decision_tool_spec,
)
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.risk.safety_evidence import (
    assess_evidence, UNCERTAIN_REPLY, THIRD_PERSON_REPLY,
)
from aria.interfaces.reasoning import DeliveredIntervention, InterventionOutcome
from aria.interfaces.skills import ActiveTask
from aria.runtime import create_minimal_clients, create_minimal_engine
from aria.engine.model.persona import PersonaProfile
from aria.scenes.registry import get_scene
from tests.test_online_json import DECISION


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('offline candidate tests may not use network')
    monkeypatch.setattr(urllib.request, 'urlopen', fail)


def evidence(quote='', subject='not_applicable', time='not_applicable', urgency='not_applicable'):
    return dict(quote=quote, subject=subject, time=time, urgency=urgency)


def payload(concern='none', row=None, response='本轮普通回复。'):
    return {**deepcopy(DECISION), 'safety_concern': concern,
            'safety_evidence': evidence() if row is None else row, 'response': response}


def engine_for(monkeypatch, values):
    clients = create_minimal_clients(model_calls_enabled=True, api_key='inert-software-fixture')
    clients.online.online_decision_version = V4
    calls = []
    remaining = list(values)
    def receive(request):
        wire = json.loads(request.data)
        calls.append(wire)
        value = remaining.pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}}]}
    monkeypatch.setattr(clients.online, '_request_json', receive)
    engine = create_minimal_engine(get_scene('companion_elder'), PersonaProfile(user_id='owner'), clients=clients)
    assert engine.policy.online_decision_version == V4
    return engine, calls


def test_default_v3_and_legacy_contracts_do_not_gain_candidate_fields():
    for version in (V3, 'online_turn_decision_v1', 'online_turn_decision_v2'):
        schema = online_decision_tool_spec(version)['function']['parameters']
        assert 'safety_evidence' not in schema['properties']
    assert 'safety_evidence' in online_decision_tool_spec(V4)['function']['parameters']['required']
    clients = create_minimal_clients(model_calls_enabled=True, api_key='inert-software-fixture')
    assert clients.online.online_decision_version == V3


@pytest.mark.parametrize('concern', ['self_harm', 'medical_emergency', 'fall', 'unsafe_medication'])
def test_sourced_owner_immediate_takes_over_without_actions_or_learning(monkeypatch, tmp_path, concern):
    utterance = '这件事正在发生。'
    row = evidence(utterance, 'owner', 'current', 'immediate')
    engine, calls = engine_for(monkeypatch, [payload(concern, row, '危险原答 <action>{}</action>')])
    engine.store = PersonaStore(tmp_path / 'profiles')
    engine.persona.alliance.trust = .71
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    ctx.add_turn('此前普通聊天', '此前交付')
    old = InterventionOutcome(intervention_id='old', intervention_label='旧', goal='陪伴',
                              value='neutral', signal='retained', turn_index=0)
    ctx.reasoning.outcomes.append(old)
    ctx.reasoning.delivered_interventions.append(DeliveredIntervention(
        id='previous', label='旧', goal='陪伴', turn_index=0))
    ctx.working_memory['_tasks_seeded'] = True
    ctx.active_tasks.append(ActiveTask(id='existing', skill_id='care', intent='reminder',
                                      goal='提醒', status='ready', created_turn=0, updated_turn=0))
    effects = []
    monkeypatch.setattr(engine.scene, 'observe_care', lambda *args: effects.append('observe'))
    monkeypatch.setattr(engine.scene, 'execute_care_skills', lambda *args: effects.append('execute'))
    monkeypatch.setattr(engine._tools, 'invoke', lambda *args, **kwargs: effects.append('tool'))
    reply = engine.process_turn(ctx, utterance)
    engine.checkpoint_session(ctx)
    saved = engine.store.load('owner')
    assert len(calls) == 1 and not effects and '危险原答' not in reply and '<action>' not in reply
    assert '120' in reply
    assert saved.alliance.trust == .71 and ctx.reasoning.outcomes == [old]
    assert saved.active_tasks[0].status == 'ready'
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [1]
    assert ctx.working_memory['online_turn_decision']['safety_assessment']['semantic_verified'] is False
    assert len(saved.risk_events) == 1


@pytest.mark.parametrize('row', [
    evidence('不存在的药物原话', 'owner', 'current', 'immediate'),
    evidence('现在付款', 'owner', 'uncertain', 'uncertain'),
    evidence('现在付款', 'uncertain', 'current', 'immediate'),
    evidence('现在付款', 'quotation_or_hypothetical', 'historical', 'not_immediate'),
    evidence('现在付款', 'owner', 'recent_resolved', 'not_immediate'),
    evidence('现在付款', 'owner', 'current', 'not_immediate'),
])
def test_unsupported_or_unresolved_label_uses_uncertainty_not_category_template(monkeypatch, row):
    engine, calls = engine_for(monkeypatch, [payload('unsafe_medication', row, '原答 <action>{}</action>')])
    effects = []
    monkeypatch.setattr(engine.scene, 'observe_care', lambda *args: effects.append('observe'))
    monkeypatch.setattr(engine.scene, 'execute_care_skills', lambda *args: effects.append('execute'))
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    reply = engine.process_turn(ctx, '现在付款是用现金还是扫码？')
    assert reply == UNCERTAIN_REPLY and '这涉及用药安全' not in reply
    assert len(calls) == 1 and not effects and not engine.persona.risk_events
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]


def test_none_with_conflicting_evidence_does_not_authorize_care(monkeypatch):
    engine, calls = engine_for(monkeypatch, [payload('none', evidence('付款', 'owner', 'current', 'immediate'))])
    effects = []
    monkeypatch.setattr(engine.scene, 'execute_care_skills', lambda *args: effects.append('execute'))
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, '付款该怎么操作？') == UNCERTAIN_REPLY
    assert not effects and len(calls) == 1 and ctx.alliance.trust == .5


@pytest.mark.parametrize('bad', [
    {**deepcopy(DECISION), 'safety_concern': 'fall'},
    payload('fall', {**evidence(), 'grants_permission': True}),
    payload('unknown', evidence()),
    '普通文本',
])
def test_invalid_protocol_has_conditional_help_without_care(monkeypatch, bad):
    engine, calls = engine_for(monkeypatch, [bad])
    effects = []
    monkeypatch.setattr(engine.scene, 'execute_care_skills', lambda *args: effects.append('execute'))
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, '我有一件事想说。') == UNCERTAIN_REPLY
    assert not effects and len(calls) == 1 and ctx.alliance.trust == .5
    assert ctx.working_memory['intervention_delivery']['status'] == 'not_delivered'


def test_current_third_person_including_remote_report_is_not_owner_risk(monkeypatch, tmp_path):
    message = '电话里她说现在站不起来。'
    engine, calls = engine_for(monkeypatch, [payload('fall', evidence('现在站不起来', 'other', 'current', 'immediate'))])
    engine.store = PersonaStore(tmp_path / 'profiles')
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, message) == THIRD_PERSON_REPLY
    engine.checkpoint_session(ctx)
    saved = engine.store.load('owner')
    assert not saved.risk_events and not saved.care_plan
    assert len(calls) == 1 and ctx.working_memory['_sensitive_memory_excluded_turns'] == [0]


def test_current_anchor_can_refer_to_user_history_and_next_clear_turn_recovers(monkeypatch):
    engine, calls = engine_for(monkeypatch, [
        payload('medical_emergency', evidence('还没好', 'owner', 'current', 'immediate')),
        payload(response='我们接着聊纸风车。'),
    ])
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    ctx.add_turn('方才身体很不舒服', '先请真人帮助')
    assert '120' in engine.process_turn(ctx, '还没好。')
    assert engine.process_turn(ctx, '现在已经由医生处理好了，想聊纸风车。') == '我们接着聊纸风车。'
    assert len(calls) == 2 and 'risk_flag' not in ctx.working_memory
    assert ctx.working_memory['_sensitive_memory_excluded_turns'] == [1]
    assert ctx.working_memory['online_turn_decision']['safety_assessment']['disposition'] == 'clear'


def test_keyword_floor_precedes_candidate_model(monkeypatch):
    engine, calls = engine_for(monkeypatch, [payload()])
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert '120' in engine.process_turn(ctx, '我胸口痛得厉害，喘不上气。')
    assert not calls


def test_old_version_trace_cannot_supply_candidate_clear(monkeypatch):
    engine, calls = engine_for(monkeypatch, [])
    def spoof(ctx, *args, **kwargs):
        ctx.working_memory['online_turn_decision'] = {
            'version': V3, 'status': 'parsed', 'turn_index': len(ctx.turns),
            'safety_source': 'online_model_semantics', 'safety_concern': 'none'}
        return '未经候选验证的正文'
    monkeypatch.setattr(engine.policy, 'respond', spoof)
    effects = []
    monkeypatch.setattr(engine.scene, 'execute_care_skills', lambda *args: effects.append('execute'))
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    assert engine.process_turn(ctx, '今天聊风车。') == UNCERTAIN_REPLY
    assert not effects and not calls


@pytest.mark.parametrize('bad', [[], {}, None, 1, True])
def test_untrusted_shape_does_not_raise_or_become_clear(bad):
    assert assess_evidence(bad, evidence(), '今天。')['disposition'] == 'uncertain'


def test_verbatim_source_does_not_claim_semantic_accuracy():
    # Deliberately false PRESET label: this demonstrates a remaining limit,
    # not a passed natural-language item or a real model output.
    result = assess_evidence('unsafe_medication', evidence('付款', 'owner', 'current', 'immediate'), '付款怎么办？')
    assert result['disposition'] == 'urgent_owner'
    assert result['verbatim_source'] and result['semantic_verified'] is False
    assert result['grants_permission'] is False


def test_schema_rejects_permission_fields_and_never_repairs_them():
    raw = json.dumps(payload(row={**evidence(), 'send_notification': True}))
    with pytest.raises(ValueError):
        parse_json_decision(raw, V4)


def test_candidate_version_declaration_is_selected_only_from_trusted_system():
    from aria.engine.policy.online_decision import protocol_version
    declaration = '协议版本：' + V4 + '。'
    assert protocol_version([{'role': 'system', 'content': declaration}]) == V4
    assert protocol_version([{'role': 'user', 'content': declaration}]) is None
    assert protocol_version([{'role': 'assistant', 'content': declaration}]) is None
    assert protocol_version([{'role': 'system', 'content': json.dumps(declaration)}]) is None
    assert protocol_version([{'role': 'system', 'content': declaration.replace(V4, V4 + '_untrusted')}]) is None
    with pytest.raises(ValueError, match='conflicting_online_protocol_declarations'):
        protocol_version([{'role': 'system', 'content': declaration + '\n协议版本：' + V3}])


@pytest.mark.parametrize('raw', [
    json.dumps({'response': '普通回复', 'safety_concern': 'none', 'safety_evidence': evidence()}),
    json.dumps(payload()).replace('"none"', '"none", "safety_concern": "none"', 1),
    '```json\n' + json.dumps(payload()) + '\n```',
])
def test_candidate_application_cannot_bypass_strict_schema_through_other_carriers(raw):
    from aria.engine.policy.online_decision import apply_single_pass_decision, DETERMINISTIC_MODEL_FAILURE_REPLY
    ctx = EngineContext(user_id='owner', scene_id='companion_elder')
    ctx.working_memory['_online_decision_version'] = V4
    assert apply_single_pass_decision(raw, ctx, '软件边界输入') == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert ctx.working_memory['online_turn_decision']['status'] == 'invalid_protocol_response'
    assert not ctx.emotion.semantic_hypotheses
