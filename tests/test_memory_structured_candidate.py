"""Explicit memory schema: synthetic transport, no model/semantic qualification."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import io
import json
import urllib.error

import pytest

from aria.engine.memory.memory_contract import FIELDS, memory_response_schema, validate_memory_response
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator, _parse_json_payload
from aria.engine.memory.stated_preferences import extract_preference_rows, confirmed_preference_payloads
from aria.engine.model.persona import (
    PREFERENCE_DO_MAX_CHARACTERS, PREFERENCE_QUOTE_MAX_CHARACTERS,
    PREFERENCE_QUOTE_MAX_COUNT, PREFERENCE_SCOPE_MODEL,
)
from aria.engine.policy.memory_structured import (
    MODEL, build_memory_structured_route, memory_request_format,
    validate_memory_structured_client, validate_memory_structured_response,
)
from aria.engine.policy.minimal_calls import MinimalCallGuard, MinimalCallStopped
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.release import create_qwen_max_memory_structured_candidate_clients
from tests._model_http_stub import stub_model_http
from tests.test_qwen_max_candidate import clients as old_clients, route as max_route
from tests.test_reserved_calls import stub, receipt
from tests.test_memory_prefix_recovery import setup as extractor_setup, preference


def metadata():
    model = {'id': MODEL, 'canonical_slug': MODEL + '-fixture', 'reasoning': {'mandatory': False}}
    endpoint = {'model_id': MODEL, 'name': 'Anthropic fixture', 'provider_name': 'Anthropic',
        'tag': 'anthropic', 'status': 0, 'max_completion_tokens': 128000,
        'supported_parameters': ['structured_outputs', 'response_format', 'max_tokens', 'reasoning'],
        'pricing': {'prompt': '.000002', 'completion': '.00001', 'input_cache_read': '.0000002',
                    'input_cache_write': '.0000025', 'input_cache_write_1h': '.000004', 'web_search': '.01', 'discount': 0}}
    return {'data': [model]}, {'data': {'id': MODEL, 'endpoints': [endpoint]}}


def route(models=None, endpoints=None, **kwargs):
    default = metadata()
    return build_memory_structured_route(json.dumps(models or default[0]).encode(),
        json.dumps(endpoints or default[1]).encode(), **kwargs)


def clients(**kwargs):
    return create_qwen_max_memory_structured_candidate_clients(model_calls_enabled=True,
        api_key='inert-test', base_url='https://fixture.invalid/v1',
        online_structured_route=max_route(), memory_structured_route=kwargs.pop('memory_structured_route', route()), **kwargs)


def policy(c):
    binding = c.memory_structured_route.binding()
    return {'schema': 'aria_reservation_v1', 'framing_tokens': 8192, 'models': {MODEL: {
        'input_usd_per_million': '4', 'output_usd_per_million': '10', 'provider': 'anthropic',
        'response_models': [MODEL, binding['canonical_slug']], 'response_providers': ['Anthropic'],
        'max_input_tokens': 100000, 'max_output_tokens': 4000,
        'metadata_sha256': binding['endpoint_metadata_sha256']}}}


def guard(path, c, prices=None):
    return ReservedCallGuard(path, max_requests=1, max_observed_cost_usd='1', reservation_policy=prices or policy(c))


def empty(): return {name: [] for name in FIELDS}
def messages(): return [{'role': 'system', 'content': '软件结构检查。'}, {'role': 'user', 'content': '软件输入。'}]
def response(c, raw=None):
    return receipt(c, choices=[{'finish_reason': 'stop', 'message': {'content': raw or json.dumps(empty())}}])


def invoke(g, c):
    with g.scope('review', 'software-memory', 'software-session'):
        return c.complete(messages())


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail('network forbidden'))
    monkeypatch.setattr('aria.runtime.openrouter_api_key', lambda: pytest.fail('credential read forbidden'))
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key', lambda: pytest.fail('credential read forbidden'))


def test_explicit_preview_and_disabled_route_never_enable_calls():
    result = create_qwen_max_memory_structured_candidate_clients()
    assert not result.model_calls_enabled and result.safety_review is None
    with pytest.raises(ValueError): clients(memory_structured_route=None)
    with pytest.raises(TypeError): create_qwen_max_memory_structured_candidate_clients(model_calls_enabled=1)


def test_only_memory_candidate_differs_from_old_max_bundle():
    old, new = old_clients(), clients()
    assert old.memory.memory_structured_route is None
    for role in ('online', 'safety_review', 'reflection'):
        a, b = getattr(old, role), getattr(new, role)
        for field in ('model', 'max_completion_tokens', 'timeout', 'online_return_format',
                      'reflection_return_format', 'reasoning_effort', 'memory_prefix_recovery', 'max_retries', 'fallback_models'):
            assert getattr(a, field) == getattr(b, field)
    c = new.memory
    assert (c.model, c.max_completion_tokens, c.timeout, c.max_retries, c.fallback_models) == (MODEL, 4000, 90, 0, [])
    assert c.memory_prefix_recovery and c.network_request_count == 0


@pytest.mark.parametrize('key', ['OPENROUTER_MEMORY_TIMEOUT', 'ARIA_MEMORY_LLM_TIMEOUT', 'ARIA_LLM_TIMEOUT'])
def test_timeout_environment_cannot_raise_explicit_bound(monkeypatch, key):
    monkeypatch.setenv(key, '900')
    # Isolate memory's global fallback from the independently bounded online
    # role, whose original factory correctly rejects a 900-second override.
    monkeypatch.setenv('OPENROUTER_ONLINE_FAST_TIMEOUT', '8')
    c = clients().memory
    assert c.timeout == 90
    validate_memory_structured_client(c)


def test_output_and_retry_environment_cannot_change_factory_contract(monkeypatch):
    monkeypatch.setenv('OPENROUTER_MEMORY_MAX_COMPLETION_TOKENS', '9999')
    monkeypatch.setenv('OPENROUTER_MEMORY_MAX_RETRIES', '4')
    monkeypatch.setenv('OPENROUTER_MEMORY_FALLBACK_MODELS', 'other/model')
    c = clients().memory
    assert (c.max_completion_tokens, c.max_retries, c.fallback_models) == (4000, 0, [])


@pytest.mark.parametrize('fault', ['schema', 'reasoning', 'inactive', 'duplicate', 'provider', 'cap', 'price', 'new_price'])
def test_capability_metadata_cannot_be_incomplete(fault):
    models, endpoints = metadata(); endpoint = endpoints['data']['endpoints'][0]
    if fault == 'schema': endpoint['supported_parameters'].remove('structured_outputs')
    elif fault == 'reasoning': models['data'][0]['reasoning']['mandatory'] = True
    elif fault == 'inactive': endpoint['status'] = -1
    elif fault == 'duplicate': endpoints['data']['endpoints'].append(deepcopy(endpoint))
    elif fault == 'provider': endpoint['provider_name'] = 'Other'
    elif fault == 'cap': endpoint['max_completion_tokens'] = 3999
    elif fault == 'price': endpoint['pricing']['completion'] = 'NaN'
    elif fault == 'new_price': endpoint['pricing']['unknown_charge'] = '1'
    with pytest.raises(ValueError): route(models, endpoints)


def test_schema_required_closed_no_true_defaults_and_same_original_fields():
    schema = memory_response_schema()
    assert tuple(schema['properties']) == FIELDS and schema['required'] == list(FIELDS)
    assert schema['additionalProperties'] is False
    for node in [schema, *schema['$defs'].values()]:
        assert node['additionalProperties'] is False
        assert set(node['properties']) == set(node['required'])
        assert all('default' not in p for p in node['properties'].values())
    review = schema['$defs']['_SemanticReview']['properties']
    for key in ('owner_requested', 'quote_supported', 'scope_preserved', 'no_permission_claim'):
        assert review[key]['type'] == 'boolean' and 'const' not in review[key]
    assert schema['$defs']['_Preference']['properties']['confirmation']['type'] == 'null'
    schema['properties'].clear()
    assert tuple(memory_response_schema()['properties']) == FIELDS


def test_generation_projects_existing_preference_bounds_without_changing_reception():
    from aria.engine.memory.memory_contract import _Response
    schema = memory_response_schema()
    pref = schema['$defs']['_Preference']['properties']
    quote = schema['$defs']['_PreferenceQuote']['properties']
    assert (pref['do']['minLength'], pref['do']['maxLength']) == (1, PREFERENCE_DO_MAX_CHARACTERS)
    assert (pref['quotes']['minItems'], pref['quotes']['maxItems']) == (1, PREFERENCE_QUOTE_MAX_COUNT)
    assert (quote['text']['minLength'], quote['text']['maxLength']) == (1, PREFERENCE_QUOTE_MAX_CHARACTERS)
    assert quote['turn_index']['type'] == 'integer' and quote['turn_index']['minimum'] == 0
    assert 'maxLength' not in pref['scope'] and 'maxItems' not in schema['properties']['stated_preferences']
    reception = _Response.model_json_schema()['$defs']
    assert 'maxLength' not in reception['_Preference']['properties']['do']
    assert 'maxItems' not in reception['_Preference']['properties']['quotes']
    assert 'maxLength' not in reception['_PreferenceQuote']['properties']['text']
    assert 'minimum' not in reception['_PreferenceQuote']['properties']['turn_index']
    pref['do']['maxLength'] = 999
    assert memory_response_schema()['$defs']['_Preference']['properties']['do']['maxLength'] == 60


@pytest.mark.parametrize('field,size,accepted', [
    ('do', 0, False), ('do', 1, True), ('do', 60, True), ('do', 61, False), ('do', 66, False),
    ('quotes', 0, False), ('quotes', 1, True), ('quotes', 3, True), ('quotes', 4, False),
    ('text', 0, False), ('text', 1, True), ('text', 240, True), ('text', 241, False),
    ('turn_index', -1, False), ('turn_index', 0, True),
])
def test_projected_mechanical_boundaries_match_itemwise_extraction(field, size, accepted):
    # Fixed capacity probes, not semantic understanding examples or live results.
    persona, ctx, _, _ = extractor_setup()
    row = preference(ctx)
    row.update(scope='闲聊修理时', revision_basis=None)
    if field == 'do': row['do'] = '等' * size
    elif field == 'quotes': row['quotes'] *= size
    elif field == 'text':
        source = '请等我说完再问。'
        ctx.turns[0].user = source + '嗯' * (240 - len(source))
        assert len(ctx.turns[0].user) == 240
        row['quotes'][0]['text'] = (ctx.turns[0].user + '嗯')[:size]
    else: row['quotes'][0]['turn_index'] = size
    payload = empty(); payload['stated_preferences'] = [row]
    assert validate_memory_response(json.dumps(payload)) == payload
    proposals, rejected = extract_preference_rows([row], ctx, persona, scope_protocol=PREFERENCE_SCOPE_MODEL)
    assert bool(proposals) is accepted and bool(rejected) is not accepted
    defs = memory_response_schema()['$defs']
    owner = '_Preference' if field in ('do', 'quotes') else '_PreferenceQuote'
    bound = defs[owner]['properties'][field]
    if field == 'turn_index': in_generated_bound = size >= bound['minimum']
    else:
        lower, upper = ('minItems', 'maxItems') if field == 'quotes' else ('minLength', 'maxLength')
        in_generated_bound = bound[lower] <= size <= bound[upper]
    assert in_generated_bound is accepted


def test_overlength_proposal_does_not_discard_valid_proposal_or_grounded_fact():
    persona, ctx, client, extractor = extractor_setup()
    client.role = 'memory'; client.preference_scope_protocol = PREFERENCE_SCOPE_MODEL
    good = preference(ctx); good.update(do='等' * 60, revision_basis=None)
    bad = deepcopy(good); bad['do'] += '等'
    payload = empty(); payload['stated_preferences'] = [bad, good]
    payload['episodes'] = [{'summary': ctx.turns[1].user, 'keywords': ['纸风车'],
        'referenceable': True, 'evidence_quote': ctx.turns[1].user, 'turn_index': 1}]
    client.payload = json.dumps(payload)
    assert validate_memory_response(client.payload) == payload
    MemoryConsolidator(extractor=extractor).consolidate(persona, ctx, persist=False)
    assert extractor.last_status == 'ok' and len(client.messages) == 1
    assert len(persona.stated_preferences) == len(persona.episodic_memory) == 1
    assert persona.stated_preferences[0].do == good['do']
    assert persona.stated_preferences[0].status == 'proposed'
    assert persona.stated_preferences[0].confirmation is None
    assert confirmed_preference_payloads(persona) == []
    assert ctx.working_memory['stated_preference_extraction']['rejected'] == ['preference_not_safe_explicit_instruction']


def test_real_local_transport_only_adds_memory_schema_and_preserves_known_fee(tmp_path, monkeypatch):
    c = clients().memory
    calls = stub(monkeypatch, response(c))
    with closing(guard(tmp_path / 'new.sqlite3', c)) as g:
        g.attach(c)
        assert invoke(g, c) == json.dumps(empty())
        state = g.snapshot(); row = state['attempts'][0]
        assert Decimal(state['known_cost_usd']) == Decimal('.00001') and state['reserved_cost_usd'] == '0'
        assert row['memory_body'] == {'finish_reason': 'stop', 'prefix_recovery_permitted': False}
        assert row['reservation']['output_token_limit'] == 4000
    wire = calls[0]
    assert wire['response_format'] == memory_request_format()
    assert wire['reasoning'] == {'enabled': False, 'exclude': True}
    assert 'temperature' not in wire and 'tools' not in wire and 'plugins' not in wire
    assert wire['max_tokens'] == 4000 and wire['provider']['only'] == ['anthropic']
    old = old_clients().memory
    old_calls = stub(monkeypatch, response(old))
    with closing(guard(tmp_path / 'old.sqlite3', c)) as g:
        g.attach(old); invoke(g, old)
    assert {k: v for k, v in wire.items() if k != 'response_format'} == old_calls[0]


@pytest.mark.parametrize('fault', ['same_duplicate', 'conflict', 'missing', 'wrong_type', 'confirmation', 'extra', 'nonfinite',
                                  'fence', 'preamble', 'length', 'tool', 'refusal', 'hidden'])
def test_body_failure_is_one_attempt_and_keeps_verified_cost(tmp_path, monkeypatch, fault):
    c = clients().memory; data = response(c); value = empty()
    if fault in ('same_duplicate', 'conflict'):
        raw = json.dumps(value)[:-1] + ',"episodes":' + ('[]' if fault == 'same_duplicate' else '[{}]') + '}'
    else:
        if fault == 'missing': value.pop('cognitive_privacy')
        elif fault == 'wrong_type': value['episodes'] = False
        elif fault == 'confirmation':
            persona, ctx, _, _ = extractor_setup(); pref = preference(ctx); pref['confirmation'] = True
            value['stated_preferences'] = [pref]
        elif fault == 'extra': value['consent_to_share'] = True
        raw = json.dumps(value)
        if fault == 'nonfinite': raw = raw[:-1] + ',"extra":NaN}'
        elif fault == 'fence': raw = '```json\n' + raw + '\n```'
        elif fault == 'preamble': raw = 'Here is the JSON: ' + raw
    data['choices'][0]['message']['content'] = raw
    if fault == 'length': data['choices'][0]['finish_reason'] = 'length'
    elif fault == 'tool': data['choices'][0]['message']['tool_calls'] = [{'function': {'name': 'unexpected'}}]
    elif fault == 'refusal': data['choices'][0]['message']['refusal'] = 'refused'
    elif fault == 'hidden': data['choices'][0]['message']['reasoning_content'] = 'synthetic hidden'
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / 'calls.sqlite3', c)) as g:
        g.attach(c)
        with pytest.raises((ValueError, MinimalCallStopped)): invoke(g, c)
        state = g.snapshot()
        assert len(calls) == state['calls_used'] == 1
        assert Decimal(state['known_cost_usd']) == Decimal('.00001') and state['reserved_cost_usd'] == '0'
        assert 'unverified_cost_usd' not in state['attempts'][0]


@pytest.mark.parametrize('fault', ['model', 'provider', 'error', 'http429', 'usage', 'overcost'])
def test_unverified_fee_retains_hold_without_double_counting(tmp_path, monkeypatch, fault):
    c = clients().memory; data = response(c)
    if fault in ('model', 'provider'): data[fault] = 'unrecognized'
    elif fault == 'error': data['error'] = {'code': 500, 'message': 'synthetic error'}
    elif fault == 'usage': data['usage']['prompt_tokens'] = False
    elif fault == 'overcost': data['usage']['cost'] = 99
    if fault == 'http429':
        def fail(req, **kwargs): raise urllib.error.HTTPError(req.full_url, 429, 'fixture', {}, io.BytesIO(json.dumps(data).encode()))
        stub_model_http(monkeypatch, fail)
    else: stub(monkeypatch, data)
    with closing(guard(tmp_path / 'calls.sqlite3', c)) as g:
        g.attach(c)
        with pytest.raises(MinimalCallStopped): invoke(g, c)
        state = g.snapshot()
        assert state['calls_used'] == 1 and Decimal(state['reserved_cost_usd']) > 0
        assert Decimal(state['known_cost_usd'] or '0') == 0 and 'cost' not in state['attempts'][0]['usage']


@pytest.mark.parametrize('fault', ['metadata', 'price', 'route_mutation', 'schema_mutation', 'no_reserved_guard'])
def test_before_send_binding_or_price_mismatch_sends_nothing(tmp_path, monkeypatch, fault):
    c = clients().memory; prices = policy(c); calls = stub(monkeypatch, response(c))
    if fault == 'metadata': prices['models'][MODEL]['metadata_sha256'] = '0' * 64
    elif fault == 'price': prices['models'][MODEL]['input_usd_per_million'] = '2'
    g = (MinimalCallGuard(tmp_path / 'calls.sqlite3', max_requests=1, max_observed_cost_usd='1')
         if fault == 'no_reserved_guard' else guard(tmp_path / 'calls.sqlite3', c, prices))
    with closing(g):
        if fault in ('metadata', 'price', 'no_reserved_guard'):
            with pytest.raises(ValueError): g.attach(c)
        else:
            g.attach(c)
            if fault == 'route_mutation': c.timeout = 900
            else:
                from aria.engine.policy import memory_structured
                original = memory_structured.memory_request_format
                def changed(client=None):
                    value = original(client); value['json_schema']['strict'] = False; return value
                monkeypatch.setattr(memory_structured, 'memory_request_format', changed)
            with pytest.raises((ValueError, MinimalCallStopped)): invoke(g, c)
        assert calls == [] and g.snapshot()['calls_used'] == 0


def test_legacy_fences_and_strict_duplicate_rejection_are_unchanged():
    raw = '```json\n' + json.dumps(empty()) + '\n```'
    assert _parse_json_payload(raw) == empty()
    with pytest.raises(ValueError): validate_memory_response(raw)
    with pytest.raises(ValueError, match='memory_json_duplicate_key'):
        _parse_json_payload('{"episodes":[],"episodes":[],"relationships":[]}')


def test_schema_does_not_make_false_semantic_review_true_or_confirm_preferences():
    persona, ctx, client, extractor = extractor_setup()
    value = empty(); value['stated_preferences'] = [preference(ctx)]
    value['stated_preferences'][0]['semantic_review']['owner_requested'] = False
    client.payload = json.dumps(value)
    validate_memory_response(client.payload)
    result = extractor.extract(ctx, persona)
    assert extractor.last_status == 'ok' and result.stated_preferences == []
    assert persona.stated_preferences == []


def test_schema_does_not_bypass_literal_source_or_disabled_cognitive_checks():
    persona, ctx, client, extractor = extractor_setup()
    value = empty(); value['episodes'] = [{'summary': 'synthetic', 'keywords': [], 'referenceable': True,
        'evidence_quote': 'not in the actual source', 'turn_index': 0}]
    client.payload = json.dumps(value); validate_memory_response(client.payload)
    assert extractor.extract(ctx, persona).new_episodes == [] and extractor.last_status == 'ok'
    value['cognitive_observations'] = [{'domain': 'memory', 'evidence_quote': ctx.turns[0].user,
        'turn_index': 0, 'functional_impact': False, 'event_kind': 'new_event', 'prior_event_id': None}]
    client.payload = json.dumps(value); validate_memory_response(client.payload)
    extractor.extract(ctx, persona)
    assert extractor.last_status == 'failed' and not ctx.working_memory.get('_memory_privacy_review_completed')


@pytest.mark.parametrize('fault', ['none', 'format', 'hidden', 'identity'])
def test_unknown_usage_text_never_reaches_public_state_on_success_or_rejection(tmp_path, monkeypatch, fault):
    c = clients().memory; data = response(c)
    data['usage']['provider_metadata'] = {'reasoning': 'NEVER_RETAIN_USAGE_SECRET'}
    if fault == 'format': data['choices'][0]['message']['content'] = '{broken'
    elif fault == 'hidden': data['choices'][0]['message']['reasoning'] = 'NEVER_RETAIN_REASONING_SECRET'
    elif fault == 'identity': data['model'] = 'other/model'
    stub(monkeypatch, data)
    with closing(guard(tmp_path / 'calls.sqlite3', c)) as g:
        g.attach(c)
        if fault == 'none': invoke(g, c)
        else:
            with pytest.raises((ValueError, MinimalCallStopped)): invoke(g, c)
        public = json.dumps({'state': g.snapshot(), 'last_usage': c.last_usage, 'last_diagnostic': c.last_diagnostic})
        assert 'NEVER_RETAIN' not in public and 'provider_metadata' not in public


@pytest.mark.parametrize('field', ['model', 'provider'])
def test_duplicate_outer_fee_identity_is_not_known_zero(tmp_path, monkeypatch, field):
    c = clients().memory; data = response(c); data['usage']['cost'] = 0
    raw = json.dumps(data)[:-1] + ',' + json.dumps(field) + ':' + json.dumps(data[field]) + '}'
    class Response:
        status = 200; headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size=-1): return raw.encode() if size < 0 else raw.encode()[:size]
    stub_model_http(monkeypatch, lambda *a, **kw: Response())
    with closing(guard(tmp_path / 'calls.sqlite3', c)) as g:
        g.attach(c)
        with pytest.raises(MinimalCallStopped): invoke(g, c)
        state = g.snapshot(); row = state['attempts'][0]
        assert row['fee_verification_error'] == 'duplicate_response_identity'
        assert 'cost' not in row['usage'] and Decimal(state['reserved_cost_usd']) > 0
