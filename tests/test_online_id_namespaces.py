"""The extra prompt distinguishes server-owned IDs without relaxing validation."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy import online_decision as decision
from aria.engine.policy.online_json import parse_json_decision, parse_v7_payload
from aria.scenes.companion_elder.reasoning import ElderStateReasoner


@pytest.mark.parametrize('version', decision.ONLINE_DECISION_VERSIONS)
@pytest.mark.parametrize('style', ['legacy', 'schema_guided'])
@pytest.mark.parametrize('candidate_count', [0, 7])
def test_namespace_inventory_matches_the_actual_prompt_and_preserves_messages(
    monkeypatch, version, style, candidate_count,
):
    ctx = EngineContext(user_id='software-only', scene_id='companion_elder')
    ctx.reasoning.candidates = ElderStateReasoner()._build_candidates([], None)[:candidate_count]
    messages = [
        {'role': 'system', 'content': '软件原规则'},
        {'role': 'user', 'content': 'software_untrusted_id 原话中的 state_signals'},
    ]
    original = deepcopy(messages)
    rendered = [{'id': row.id} for row in ctx.reasoning.candidates[:6]]
    guide = decision.online_id_namespace_contract(version, rendered)
    after = decision.single_pass_messages(messages, deepcopy(ctx), version=version, format_style=style)
    monkeypatch.setattr(decision, 'online_id_namespace_contract', lambda *_: '')
    before = decision.single_pass_messages(messages, deepcopy(ctx), version=version, format_style=style)
    assert after[0]['content'] == before[0]['content'] + guide
    assert after[1:] == before[1:] == original[1:]
    assert messages == original
    assert 'software_untrusted_id' not in guide
    assert json.dumps([row['id'] for row in rendered], ensure_ascii=False, separators=(',', ':')) in guide
    schema = decision.online_decision_tool_spec(version)['function']['parameters']
    state_ids = schema['properties']['state_signals']['items']['properties']['id']['enum']
    assert json.dumps(state_ids, ensure_ascii=False, separators=(',', ':')) in guide
    assert guide.count(decision.ONLINE_ID_NAMESPACE_VERSION) == 1
    assert ('$.deliberation.options[].intervention_id' in guide) == (version == decision.ONLINE_DECISION_VERSION_V2)


def test_state_inventory_is_read_from_schema_instead_of_hardcoded_aliases(monkeypatch):
    spec = decision.online_decision_tool_spec()
    spec['function']['parameters']['properties']['state_signals']['items']['properties']['id']['enum'] = ['software_state']
    monkeypatch.setattr(decision, 'online_decision_tool_spec', lambda _: spec)
    guide = decision.online_id_namespace_contract(decision.ONLINE_DECISION_VERSION, [{'id': 'software_strategy'}])
    assert '["software_state"]' in guide and '["software_strategy"]' in guide
    assert 'autonomy_threat' not in guide


@pytest.mark.parametrize('version', decision.ONLINE_DECISION_VERSIONS)
def test_strategy_in_state_is_never_accepted_or_repaired(version):
    value = {
        'response': '软件夹具',
        'state_signals': [{'id': 'autonomy_restoration', 'relation': 'candidate',
                           'confidence': 0.2, 'evidence_quote': '软件夹具', 'rationale': '软件结构'}],
        'current_affect': {'label': '软件占位', 'confidence': 0.2,
                           'divergence': {'possible': False, 'confidence': 0.0, 'pattern': ''}},
        'selected_intervention_id': '',
        'communication_plan': {'pace': 'open_listening', 'support_order': '软件占位', 'boundary': '软件占位'},
        'uncertainty': 0.8,
    }
    if version == decision.ONLINE_DECISION_VERSION_V2:
        value['deliberation'] = {'person_read': '软件占位', 'applied_basis': [],
                                'options': [{'intervention_id': '', 'predicted_reaction': '软件占位'}]}
    if version in decision.SAFETY_DECISION_VERSIONS:
        value['safety_concern'] = 'none'
    if version in decision.EVIDENCE_DECISION_VERSIONS:
        value['safety_evidence'] = {
            'quote': '', 'subject': 'not_applicable',
            'time': 'not_applicable', 'urgency': 'not_applicable',
        }
    if version in decision.BOUND_DECISION_VERSIONS:
        value['applied_preference_ids'] = []
    if version in decision.SEMANTIC_DECISION_VERSIONS:
        value['response_segments'] = [{'text': value['response'], 'kind': 'conversation',
                                       'source_quote': '', 'action': 'none', 'target': '', 'state_ref': ''}]
    if version == decision.ONLINE_DECISION_VERSION_V7:
        value.pop('response')
    raw = json.dumps(value, ensure_ascii=False)
    if version == decision.ONLINE_DECISION_VERSION_V7:
        assert parse_json_decision(raw, version) == raw
        parsed, dropped = parse_v7_payload(raw)
        assert 'state_signals' not in parsed
        assert dropped == ['state_signals']
        assert parsed['response'] == ''.join(row['text'] for row in value['response_segments'])
    else:
        with pytest.raises(ValueError, match='invalid_online_json_schema'):
            parse_json_decision(raw, version)
    # The same complete payload is valid with a genuine state; no other fault masks the enum check.
    value['state_signals'][0]['id'] = 'autonomy_threat'
    valid = json.dumps(value, ensure_ascii=False)
    assert parse_json_decision(valid, version) == valid
    if version == decision.ONLINE_DECISION_VERSION_V7:
        parsed, dropped = parse_v7_payload(valid)
        assert parsed['state_signals'] == value['state_signals'] and dropped == []
    assert json.loads(raw)['state_signals'][0]['id'] == 'autonomy_restoration'
