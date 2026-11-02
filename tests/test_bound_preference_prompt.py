"""Prompt placement and non-injection; no model semantic claims."""
from copy import deepcopy

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.online_decision import single_pass_messages
from aria.engine.policy.release_prompt import bound_preference_requirement
from aria.scenes.registry import get_scene
from tests.test_stated_preferences import _profile

V5='online_turn_decision_v5_bound'


def messages(profile,ctx):
    ctx.working_memory.update(_online_decision_version=V5,
        _safety_response_instruction_version='compact_response_v1')
    base=build_messages(get_scene('companion_elder'),profile,ctx,'当前软件输入。',
        safety_response_instruction_version='compact_response_v1')
    return single_pass_messages(base,ctx,version=V5,persona=profile,format_style='schema_guided')


def test_without_confirmed_preferences_proposed_or_withdrawn_changes_no_wire_bytes():
    profile=PersonaProfile(user_id='preference_test')
    ctx=EngineContext(user_id=profile.user_id,scene_id='companion_elder',session_id='fixed')
    blank=messages(profile,ctx)
    profile.stated_preferences=_profile(confirmed=False).stated_preferences
    assert messages(profile,ctx)==blank
    profile.stated_preferences[0].status='withdrawn'
    assert messages(profile,ctx)==blank
    assert bound_preference_requirement(ctx)==''


def test_confirmed_requirements_are_last_once_scope_preserved_and_at_most_four():
    profile=_profile()
    original=deepcopy(profile.stated_preferences[0])
    profile.stated_preferences=[]
    for i in range(5):
        row=deepcopy(original); row.id=f'pref_software_{i}'
        profile.stated_preferences.append(row)
    ctx=EngineContext(user_id=profile.user_id,scene_id='companion_elder',session_id='fixed')
    wire=messages(profile,ctx)
    system=wire[0]['content']
    block=bound_preference_requirement(ctx)
    assert block and system.endswith(block)
    assert system.count('## 本轮必须遵守的本人要求')==1
    projection=ctx.working_memory['prompt_evidence_snapshot']['stated_preferences_prompt_projection']
    assert len(projection)==4
    assert all(r['scope']==original.scope for r in projection)
    assert '优先于一般沟通策略' in block and '安全、隐私和现实行动权限' in block
    assert len(system.encode('utf8'))<=12000
