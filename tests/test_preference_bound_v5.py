"""Code binding and persistence with forced labels; not semantic accuracy."""
from copy import deepcopy
import json
import pytest
from aria.engine.memory.preference_dialogue import (
    bind_preference_request, apply_preference_dialogue, validate_preference_control,
    prepare_preference_invitation, acknowledge_preference_invitation,
)
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.policy.online_decision import apply_single_pass_decision, online_decision_tool_spec
from aria.engine.policy.online_json import parse_json_decision
from tests.test_preference_dialogue_readiness import fixture
from tests.test_safety_evidence import payload

V5 = 'online_turn_decision_v5_bound'
TEXT = '本人当前回答'


def setup():
    persona, ctx = fixture()
    ctx.working_memory.update(_online_decision_version=V5, _safety_delivery_version='minimal_response_v1',
                              _safety_response_instruction_version='compact_response_v1')
    bind_preference_request(persona, ctx, TEXT)
    return persona, ctx


def answer(action):
    value = payload(response='我记住了。今天先陪你聊这件事。')
    value.update(preference_control={'action': action}, applied_preference_ids=[],
                 response_segments=[{'text': '我记住了。', 'claims_saved': True},
                                    {'text': '今天先陪你聊这件事。', 'claims_saved': False}])
    return value


@pytest.mark.parametrize('action,status', [('confirm','confirmed'),('decline','declined'),
                                          ('defer','proposed'),('unclear','proposed')])
def test_action_only_binding_preserves_body_and_commits_before_claim(tmp_path, action, status):
    persona,ctx=setup(); value=answer(action)
    body=apply_single_pass_decision(json.dumps(value,ensure_ascii=False),ctx,TEXT,persona)
    result=apply_preference_dialogue(ctx,persona,TEXT,store=PersonaStore(tmp_path),response=body)
    assert result.startswith('今天先陪你聊这件事。\n') and '我记住了' not in result
    assert persona.stated_preferences[0].status == status
    if action in ('confirm','decline'):
        assert PersonaStore(tmp_path).load(persona.user_id).stated_preferences[0].status == status
        assert ctx.working_memory['preference_dialogue_receipt']['persisted'] is True
        assert apply_preference_dialogue(ctx,persona,TEXT,store=PersonaStore(tmp_path),response=body) is None
    else:
        assert '先不记' in result and 'preference_dialogue_receipt' not in ctx.working_memory


@pytest.mark.parametrize('change',['replaced','expired','unack','turn','source','target'])
def test_old_action_cannot_bind_later_invitation(change):
    persona,ctx=setup()
    if change=='replaced':
        inv=prepare_preference_invitation(persona,ctx,persona.stated_preferences[0].id,actor=persona.user_id)
        acknowledge_preference_invitation(persona,ctx,inv['id'],inv['question_sha256'],actor=persona.user_id)
    pending=ctx.working_memory['_preference_dialogue_invitation']
    if change=='expired': pending['expires_at']='2000-01-01T00:00:00+00:00'
    if change=='unack': pending['delivered']=False
    if change=='turn': pending['turn_index']+=1
    if change=='source': ctx.working_memory['_preference_request_binding']['owner_message_sha256']='bad'
    if change=='target': pending['target_sha256']='bad'
    assert validate_preference_control({'action':'confirm'},persona,ctx,TEXT) is None
    assert persona.stated_preferences[0].status=='proposed'


def test_store_failure_never_publishes_success_or_mutates_live_profile():
    persona,ctx=setup(); before=deepcopy(persona.model_dump());value=answer('confirm')
    body=apply_single_pass_decision(json.dumps(value,ensure_ascii=False),ctx,TEXT,persona)
    class Failed:
        def save(self,_): raise OSError('disk full')
    with pytest.raises(OSError):
        apply_preference_dialogue(ctx,persona,TEXT,store=Failed(),response=body)
    assert persona.model_dump()==before and 'preference_dialogue_receipt' not in ctx.working_memory


def test_native_schema_forbids_model_identity_and_requires_sentence_contract():
    value=answer('confirm')
    parse_json_decision(json.dumps(value),V5,require_preference_control=True)
    value['preference_control']['target_id']='invented'
    with pytest.raises(ValueError): parse_json_decision(json.dumps(value),V5,require_preference_control=True)
    value=answer('confirm'); del value['response_segments']
    with pytest.raises(ValueError): parse_json_decision(json.dumps(value),V5,require_preference_control=True)


def test_mismatching_segments_fail_before_any_commit():
    persona,ctx=setup(); value=answer('confirm');value['response_segments'][0]['text']='different'
    apply_single_pass_decision(json.dumps(value),ctx,TEXT,persona)
    assert ctx.working_memory['online_turn_decision']['status']=='invalid_protocol_response'
    assert persona.stated_preferences[0].status=='proposed'


def test_mislabeled_claim_does_not_create_persistence_evidence():
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    persona, ctx = setup()
    value = answer('unclear')
    value['response'] = '我已经保存了这条相处说明。今天可以慢慢聊。'
    value['response_segments'] = [{'text':value['response'], 'claims_saved':False}]
    body = apply_single_pass_decision(json.dumps(value,ensure_ascii=False), ctx, TEXT, persona)
    response = apply_preference_dialogue(ctx,persona,TEXT,response=body)
    ctx.working_memory['_delivery_response_candidate_for_turn'] = response
    delivered = guard_external_delivery_response(ctx,response,TEXT,persona)
    # Deliberately incorrect semantic annotation remains a visible model risk;
    # a lexical recognizer is forbidden for v5 and cannot prove its absence.
    assert '已经保存' in delivered
    assert '今天可以慢慢聊。' in delivered and '先不记' in delivered
    assert persona.stated_preferences[0].status == 'proposed'
    assert ctx.working_memory.get('preference_dialogue_receipt') is None
    assert ctx.working_memory['online_turn_decision']['preference_body_composition']['claims_semantically_verified'] is False


def test_unsolicited_annotated_claim_is_removed_without_status_or_body_rewrite():
    persona,ctx=setup()
    ctx.working_memory.pop('_preference_dialogue_invitation')
    bind_preference_request(persona,ctx,TEXT)
    value=answer('unclear')
    value.pop('preference_control')
    body=apply_single_pass_decision(json.dumps(value,ensure_ascii=False),ctx,TEXT,persona)
    assert apply_preference_dialogue(ctx,persona,TEXT,response=body)=='今天先陪你聊这件事。'
    assert persona.stated_preferences[0].status=='proposed'
