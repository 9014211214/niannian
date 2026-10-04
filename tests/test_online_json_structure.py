"""Prospective format guidance, not a live-model compliance claim."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import online_decision_tool_spec, single_pass_messages
from aria.engine.policy.online_json import (
    INPUT_VERSION, ONLINE_JSON_STRUCTURE_VERSION, _carrier_contract, _canonical,
    parse_json_decision, prepare_json_messages, schema_field_contract,
)


@pytest.mark.parametrize('version', ['online_turn_decision_v1', 'online_turn_decision_v2'])
@pytest.mark.parametrize('advertised', [[], ['fixture: only on consent']])
def test_generated_json_contract_removes_pseudo_values_preserves_meaning(version, advertised):
    ctx = EngineContext(user_id='structure', scene_id='companion_elder')
    ctx.working_memory['_advertised_tools_for_turn'] = advertised
    dialogue = [{'role': 'system', 'content': '原有规则不可覆盖'},
                {'role': 'user', 'content': '原话包括confidence:0到1和枚举值，也要保留。'}]
    before = deepcopy(dialogue)
    current = single_pass_messages(dialogue, ctx, version=version, format_style='schema_guided')
    wire = prepare_json_messages(current, version)
    text = wire[0]['content']
    assert dialogue == before and current[1:] == dialogue[1:]
    assert '"confidence":0到1' not in text
    assert '"possible":true或false' not in text
    assert '"pace":"枚举值"' not in text
    assert 'stepwise逐步' not in text
    for rule in ('当前或最近三轮用户原话中的逐字短句', '为什么只是当轮候选或反证',
                 '简短、非诊断描述', '为什么字面和具体变化可能不一致',
                 '只能从当前可选干预id中选择', '没有则空字符串',
                 '短句仅指response字段', '情绪/症状本身不授权'):
        assert rule in text
    if advertised:
        assert '当前明确请求/确认并授权' in text and '标签置 response 末尾' in text
    else:
        assert '禁用外部 <action> 标签' in text
    assert json.loads(wire[1]['content'])['current_user_message'] == dialogue[-1]['content']
    assert text.endswith(_canonical(online_decision_tool_spec(version)['function']['parameters']))
    assert prepare_json_messages(wire, version) == wire


@pytest.mark.parametrize('version', ['online_turn_decision_v1', 'online_turn_decision_v2'])
def test_old_v2_prepared_input_rebinds_once_without_losing_any_quoted_text(version):
    prior = [{'role': 'user', 'content': '同名标记## 嵌套字段逐项约束只是原话'},
             {'role': 'assistant', 'content': '已经交付的短句，不是决策对象'}]
    quoted = {'schema': INPUT_VERSION, 'conversation_history': prior, 'current_user_message': '现在的原话'}
    old = [{'role': 'system', 'content': '产品规则原文' + _carrier_contract(version)},
           {'role': 'user', 'content': _canonical(quoted)}]
    before = deepcopy(old)
    new = prepare_json_messages(old, version)
    assert old == before
    assert json.loads(new[-1]['content']) == quoted
    assert new[0]['content'].count('本次输入与返回合同：online_json_protocol_v2') == 1
    assert new[0]['content'].count(ONLINE_JSON_STRUCTURE_VERSION) == 1
    assert prepare_json_messages(new, version) == new


def test_nested_guide_covers_required_paths_without_fixed_strategy_or_answer():
    text = schema_field_contract('online_turn_decision_v2')
    for path in ('$.state_signals[].relation', '$.state_signals[].evidence_quote',
                 '$.current_affect.divergence.possible', '$.current_affect.divergence.confidence',
                 '$.communication_plan.pace', '$.communication_plan.boundary',
                 '$.deliberation.options[].predicted_reaction'):
        assert path + '：' in text
    assert '即使possible为false也必须含confidence和pattern' in text
    assert '不提供本轮答案、默认情绪、默认策略或默认置信度' in text
    assert '"pace":' not in text and '"confidence":' not in text


@pytest.mark.parametrize('mutation', ['boolean_object', 'null_object', 'string_number', 'enum_alias', 'missing_nested'])
def test_structure_guidance_does_not_relax_existing_validation(mutation):
    valid = {'response': '收到。', 'state_signals': [],
             'current_affect': {'label': '未知', 'confidence': .2,
                                'divergence': {'possible': False, 'confidence': .2, 'pattern': ''}},
             'selected_intervention_id': '', 'communication_plan': {'pace': 'open_listening',
                 'support_order': '先听表达', 'boundary': '不代替决定'}, 'uncertainty': .8}
    raw = json.dumps(valid, ensure_ascii=False)
    assert parse_json_decision(raw, 'online_turn_decision_v1') == raw
    if mutation == 'boolean_object':
        valid['current_affect']['divergence'] = False
    elif mutation == 'null_object':
        valid['communication_plan'] = None
    elif mutation == 'string_number':
        valid['uncertainty'] = '0.8'
    elif mutation == 'enum_alias':
        valid['communication_plan']['pace'] = 'open_listening倾听'
    else:
        valid['current_affect']['divergence'].pop('confidence')
    with pytest.raises(ValueError):
        parse_json_decision(json.dumps(valid, ensure_ascii=False), 'online_turn_decision_v1')
