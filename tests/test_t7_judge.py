"""T7 judge tests use fictional texts, no network and no paid calls."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from aria.bench.refocus_cases import load_refocus_cases, VIOLATION_KEYS
from aria.bench.refocus_e1_runner import inventory
from scripts.t7_judge import aggregate, apply_results, build_batches, parse_batch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def design():
    suite = load_refocus_cases()
    plan = json.loads((ROOT / 'experiments/t6_20260930/plan.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'experiments/t6_20260930/judge_contract.json').read_text(encoding='utf-8'))
    rows = [{**coordinate, 'order': order, 'status': 'generated',
             'raw_decision': {'response': '我听着。'}, 'delivered_response': '我听着。',
             'actual_model': 'qwen/qwen3.7-flash', 'latency_ms': 100,
             'deliberation_status': 'captured' if coordinate['arm'] != 'B0' else 'not_applicable',
             'selected_in_deliberated_options': coordinate['arm'] != 'B0'}
            for order, coordinate in enumerate(inventory(suite), 1)]
    seeds = {name: index for index, name in enumerate(
        ('raw_shuffle', 'delivered_shuffle', 'pair_shuffle', 'human_sample'), 1)}
    return rows, suite, plan, contract, seeds


def _batch_labels(batch, value=0):
    return json.dumps([[key, value] if batch['stage'] == 'pairs'
                       else [key, *[value if a else 0 for a in item['applicable']]]
                       for key, item in batch['private']['items'].items()])


def _complete(design):
    prepared = build_batches(*design)
    result = apply_results(prepared, {b['batch_id']: _batch_labels(b, 1 if b['stage'] == 'pairs' else 0)
                                      for b in prepared['batches']})
    return prepared, result


def _mutate_label(rows, labels, case_id, arm, index, value, repeats=(1, 2, 3), stage='raw'):
    for row in rows:
        if row['case_id'] == case_id and row['arm'] == arm and row['repeat'] in repeats:
            labels[stage][str(row['order'])][index] = value


def test_fixed_inventory_and_payloads_are_blind(design):
    rows, suite, plan, contract, seeds = design
    prepared = build_batches(*design)
    assert len(prepared['slots']) == 408
    assert len(prepared['pair_slots']) == 27
    assert len(prepared['reuse']) == 204
    assert len(prepared['batches']) == 19
    assert len({slot['opaque_id'] for slot in prepared['slots'] + prepared['pair_slots']}) == 435
    all_text = ''
    swaps = set()
    for batch in prepared['batches']:
        payload = batch['payload']
        assert payload['provider'] == contract['model_request']['provider']
        assert payload['reasoning'] == {'enabled': False, 'exclude': True}
        assert payload['max_tokens'] == 512 and payload['stream'] is False
        assert 'temperature' not in payload and 'tools' not in payload
        data = json.loads(payload['messages'][1]['content'])
        text = json.dumps(data)
        all_text += text
        for forbidden in contract['blinding']['forbidden_judge_fields']:
            assert f'"{forbidden}":' not in text
        for case in suite.main_cases():
            assert case.id not in text
            assert all(g.id not in text for g in case.guidance)
        for item in batch['private']['items'].values():
            assert len(item['opaque_id']) == 14
            if batch['stage'] == 'pairs':
                swaps.add(tuple(item['person_order']))
    assert swaps == {(0, 1), (1, 0)}
    assert 'author_supplied_synthetic_not_learned' not in all_text
    assert build_batches(*design) == prepared


def test_stage_randomness_independent_and_no_rebatching_to_meet_budget(design):
    rows, suite, plan, contract, seeds = design
    for row in rows:
        row['delivered_response'] = '我听着，请慢慢说。'
    prepared = build_batches(*design)
    assert len(prepared['batches']) == 35
    assert prepared['status'] == 'blocked_budget'
    assert Decimal(prepared['total_reserve_usd']) > Decimal('.30')
    assert Decimal(prepared['total_reserve_usd']) == sum(Decimal(b['reserve_usd']) for b in prepared['batches'])
    assert prepared['provider_token_bound_verified'] is False
    for batch in prepared['batches']:
        assert batch['input_token_upper_bound'] == batch['payload_utf8_bytes'] + 8192
    other = build_batches(rows, suite, plan, contract, {**seeds, 'delivered_shuffle': 555})
    assert [b for b in prepared['batches'] if b['stage'] == 'raw'] == [b for b in other['batches'] if b['stage'] == 'raw']
    assert [b for b in prepared['batches'] if b['stage'] == 'delivered'] != [b for b in other['batches'] if b['stage'] == 'delivered']


@pytest.mark.parametrize('marker', ['这是 B2 的回答', 'Claude 回答', 'qwen/qwen3.7-flash',
                                     'profile_v1', 'prompt_version=1.12.0', 'e1_p01_left'])
def test_marker_is_not_redacted_or_sent_and_missing_slots_survive(design, marker):
    rows, suite, *_ = design
    rows[0]['raw_decision']['response'] = marker
    rows[0]['delivered_response'] = marker
    rows[1]['status'] = 'not_started'
    prepared = build_batches(*design)
    selected = [slot for slot in prepared['slots'] if slot['order'] <= 2]
    assert [slot['status'] for slot in selected] == ['unblindable', 'not_started'] * 2
    wire = json.dumps([b['payload'] for b in prepared['batches']], ensure_ascii=False)
    if marker != 'qwen/qwen3.7-flash':
        assert marker not in wire
    result = apply_results(prepared, {})
    case = {case.id: case for case in suite.main_cases()}[rows[0]['case_id']]
    assert result['labels']['raw']['1'] == [None if getattr(case.review, key).applicable else 0 for key in VIOLATION_KEYS]
    assert len(result['labels']['raw']) == 204


def test_reuse_requires_identical_utf8_and_valid_source_batch(design):
    rows, *_ = design
    rows[0]['raw_decision']['response'] = '\u00e9'
    rows[0]['delivered_response'] = 'e\u0301'
    prepared = build_batches(*design)
    assert len(prepared['reuse']) == 203
    assert not any(item['source_order'] == 1 for item in prepared['reuse'])
    raw = next(b for b in prepared['batches'] if b['stage'] == 'raw')
    result = apply_results(prepared, {raw['batch_id']: _batch_labels(raw)})
    for item in raw['private']['items'].values():
        order = item['order']
        destination = next(s for s in result['slots'] if s['stage'] == 'delivered' and s['order'] == order)
        if order != 1:
            assert destination['status'] == 'reused_identical_text'
            assert destination['source_batch_id'] == raw['batch_id']
    for item in prepared['reuse']:
        assert item['source_review_sha256'] == item['destination_review_sha256']
    assert any(slot['status'] == 'awaiting_identical_raw_review' for slot in result['slots'])


@pytest.mark.parametrize('bad', ['true', '1.0', '"0"', '2', 'NaN', '-1', '{}', '[]'])
def test_strict_label_types_reject_partial_answer(design, bad):
    prepared = build_batches(*design)
    batch = next(b for b in prepared['batches'] if b['stage'] == 'raw')
    data = _batch_labels(batch).replace(', 0', ', ' + bad, 1)
    with pytest.raises(ValueError):
        parse_batch(batch, data)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'extra', 'width', 'markdown', 'object'])
def test_exact_array_schema_and_complete_ids(design, change):
    batch = build_batches(*design)['batches'][0]
    data = json.loads(_batch_labels(batch))
    if change == 'duplicate':
        data.append(data[0])
    elif change == 'missing':
        data.pop()
    elif change == 'extra':
        data.append(['q_000000000000', 0, 0, 0, 0])
    elif change == 'width':
        data[0].append(0)
    elif change == 'object':
        data = {'labels': data}
    text = json.dumps(data)
    if change == 'markdown':
        text = '```json\n' + text + '\n```'
    with pytest.raises(ValueError):
        parse_batch(batch, text)


def test_inapplicable_null_and_pair_boolean_are_rejected(design):
    batches = build_batches(*design)['batches']
    batch = next(b for b in batches if b['stage'] == 'raw' and any(
        False in item['applicable'] for item in b['private']['items'].values()))
    output = json.loads(_batch_labels(batch))
    target = next(row for row in output if False in batch['private']['items'][row[0]]['applicable'])
    target[batch['private']['items'][target[0]]['applicable'].index(False) + 1] = None
    with pytest.raises(ValueError):
        parse_batch(batch, json.dumps(output))
    batch = next(b for b in batches if b['stage'] == 'pairs')
    with pytest.raises(ValueError):
        parse_batch(batch, _batch_labels(batch).replace(', 0', ', false', 1))


def test_fixed_core_eight_and_no_cross_criterion_compensation(design):
    rows, suite, *_ = design
    _, result = _complete(design)
    for case in suite.traps[:6]:
        dimension = next(i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable)
        _mutate_label(rows, result['labels'], case.id, 'B0', dimension, 1, repeats=(1, 2))
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    assert report['gates']['filming']['passed'] is True
    # Improving only the final two additions does not replace improvements in the original eight.
    for case in (*suite.traps[:2], *suite.traps[8:]):
        dimension = next(i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable)
        _mutate_label(rows, result['labels'], case.id, 'B0', dimension, int(case in suite.traps[8:]))
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['gates']['filming']['passed'] is False
    case = next(case for case in suite.traps if sum(getattr(case.review, k).applicable for k in VIOLATION_KEYS) > 1)
    first, second = [i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable][:2]
    _mutate_label(rows, result['labels'], case.id, 'B0', first, 1)
    _mutate_label(rows, result['labels'], case.id, 'B0', second, 0)
    _mutate_label(rows, result['labels'], case.id, 'B2', first, 0)
    _mutate_label(rows, result['labels'], case.id, 'B2', second, 1)
    item = aggregate(rows, result['labels'], result['pairlabels'], suite)['stages']['raw']['filming_comparisons'][case.id]
    assert item == {'evaluable': True, 'worse': True, 'improved': False}


def test_missing_repeat_blocks_even_with_two_agreeing_repeats(design):
    rows, suite, *_ = design
    _, result = _complete(design)
    case = suite.traps[0]
    dimension = next(i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable)
    _mutate_label(rows, result['labels'], case.id, 'B2', dimension, None, repeats=(3,))
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    assert report['gates']['filming']['passed'] is None
    assert report['gates']['v2']['passed'] is None
    assert report['stages']['raw']['applicable_label_slots'] == 651
    assert report['stages']['raw']['missing_applicable_labels'] == 1
    assert report['stages']['delivered']['missing_applicable_labels'] == 0
    assert report['human_calibration']['assessed'] == 0
    assert report['automatic_default_switch'] is False


def test_profile_needs_all_six_person_stimuli_and_all_three_pairs(design):
    rows, suite, *_ = design
    _, result = _complete(design)
    for case in suite.traps[:2]:
        dimension = next(i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable)
        _mutate_label(rows, result['labels'], case.id, 'B2', dimension, 1)
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    assert len(report['stages']['raw']['profile_comparisons']) == 16
    assert report['gates']['profile']['passed'] is True
    result['pairlabels'][f'{suite.pairs[0].id}:B3b:3'] = None
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['gates']['profile']['passed'] is None
    result['pairlabels'][f'{suite.pairs[0].id}:B3b:3'] = 0
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['gates']['profile']['passed'] is True
    result['pairlabels'][f'{suite.pairs[0].id}:B3b:2'] = 0
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['gates']['profile']['passed'] is False


def test_capture_denominator_is_48_and_latency_uses_common_thirty(design):
    rows, suite, *_ = design
    _, result = _complete(design)
    b2 = [row for row in rows if row['arm'] == 'B2']
    for row in b2[:4]:
        row['deliberation_status'] = 'missing'
        row['selected_in_deliberated_options'] = None
    trap_ids = {case.id for case in suite.traps}
    b1_traps = [row for row in rows if row['arm'] == 'B1']
    b2_traps = [row for row in b2 if row['case_id'] in trap_ids]
    for index, (baseline, treatment) in enumerate(zip(b1_traps, b2_traps), 1):
        baseline['latency_ms'] = index * 100
        treatment['latency_ms'] = index * 100 + (800 if index <= 15 else 1200)
    for row in b2:
        if row['case_id'] not in trap_ids:
            row['latency_ms'] = 10**10
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    assert report['b2_capture']['captured'] == report['b2_capture']['selected_in_candidates'] == 44
    assert report['b2_capture']['passed'] is True
    assert report['latency']['arms']['B1']['p50_ms'] == 1500
    assert report['latency']['arms']['B1']['p95_ms'] == 2900
    assert report['latency']['difference'] == {'p50_ms': 800, 'p95_ms': 1200}
    assert report['gates']['v2']['passed'] is True
    b2_traps[-1]['latency_ms'] = None
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['gates']['v2']['passed'] is None
    b2[-1]['status'] = 'not_started'
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    assert report['b2_capture']['denominator'] == 48 and report['b2_capture']['passed'] is None


def test_cannot_shrink_inventory_or_mix_seeds(design):
    rows, suite, plan, contract, seeds = design
    with pytest.raises(ValueError):
        build_batches(rows[:-1], suite, plan, contract, seeds)
    duplicate = deepcopy(rows)
    duplicate[-1]['order'] = duplicate[-2]['order']
    with pytest.raises(ValueError):
        build_batches(duplicate, suite, plan, contract, seeds)
    for bad in (True, -1, 2**128):
        with pytest.raises(ValueError):
            build_batches(rows, suite, plan, contract, {**seeds, 'raw_shuffle': bad})
    with pytest.raises(ValueError):
        build_batches(rows, suite, plan, contract, {**seeds, 'raw_shuffle': seeds['pair_shuffle']})


def test_pair_judge_pass_cannot_override_either_person_boundary_failure(design):
    rows, suite, *_ = design
    _, result = _complete(design)
    pair = suite.pairs[0]
    case = pair.cases[1]
    dimension = next(i for i, key in enumerate(VIOLATION_KEYS) if getattr(case.review, key).applicable)
    _mutate_label(rows, result['labels'], case.id, 'B3b', dimension, 1, repeats=(1, 2))
    report = aggregate(rows, result['labels'], result['pairlabels'], suite)
    item = report['pairs'][pair.id]['B3b']
    assert item['passed'] is False
    assert item['repeats'] == [False, False, True]
    assert item['comparisons'][0]['judge_pass'] == 1
    assert item['comparisons'][0]['boundaries_all_zero'] is False
    _mutate_label(rows, result['labels'], case.id, 'B3b', dimension, None, repeats=(3,))
    assert aggregate(rows, result['labels'], result['pairlabels'], suite)['profile_pair_gate']['passed'] is None


def _human_sources(rows):
    selected = [{key: row[key] for key in ('case_id', 'arm', 'repeat')} |
                {'review_id': f'h_{index:012x}', 'inventory_order': row['order']}
                for index, row in enumerate(rows[:40], 1)]
    blank = [{'review_id': row['review_id'], 'status': 'awaiting_generation_and_human_review',
              'response': None, 'human_labels': [None] * 4} for row in selected]
    return selected, blank


def test_human_form_preserves_fixed_unavailable_positions_and_never_judge_labels(design):
    from scripts.t7_judge import prepare_human_form
    rows, suite, *_ = design
    rows[0]['raw_decision']['response'] = '我是 Claude。'
    rows[1]['status'] = 'not_started'
    rows[2]['raw_decision']['response'] = ''
    selected, blank = _human_sources(rows)
    prepared = build_batches(*design)
    before = deepcopy((selected, blank, prepared))
    result = prepare_human_form(rows, suite, selected, blank, prepared)
    assert len(result) == 40
    assert [row['review_id'] for row in result] == [row['review_id'] for row in blank]
    assert [row['status'] for row in result[:4]] == ['unblindable', 'not_started', 'not_evaluable', 'awaiting_human_review']
    assert [row['response'] for row in result[:3]] == [None, None, None]
    assert all(row['human_labels'] == [None] * 4 for row in result)
    assert all(row['response'] == '我听着。' for row in result[3:])
    assert before == (selected, blank, prepared)
    wire = json.dumps(result)
    for key in ('case_id', 'arm', 'repeat', 'inventory_order', 'order', 'stage', 'actual_model', 'labels'):
        assert f'"{key}":' not in wire
    assert all('id' not in guidance for row in result for guidance in row['confirmed_guidance'])


@pytest.mark.parametrize('bad', ['missing', 'duplicate_id', 'duplicate_order', 'wrong_order_type',
                                 'coordinate_mismatch', 'judged', 'filled_response', 'filled_labels',
                                 'wrong_label_width', 'id_mismatch', 'auto_id_collision',
                                 'stale_response', 'stale_slot_status'])
def test_human_form_rejects_replacement_answers_mapping_drift_and_id_collisions(design, bad):
    from scripts.t7_judge import prepare_human_form
    rows, suite, *_ = design
    selected, blank = _human_sources(rows)
    prepared = build_batches(*design)
    if bad == 'missing':
        selected.pop()
    elif bad == 'duplicate_id':
        selected[1]['review_id'] = selected[0]['review_id']
        blank[1]['review_id'] = blank[0]['review_id']
    elif bad == 'duplicate_order':
        selected[1]['inventory_order'] = selected[0]['inventory_order']
    elif bad == 'wrong_order_type':
        selected[0]['inventory_order'] = True
    elif bad == 'coordinate_mismatch':
        selected[0]['arm'] = 'B3b'
    elif bad == 'judged':
        blank[0]['status'] = 'human_reviewed'
    elif bad == 'filled_response':
        blank[0]['response'] = '预写回应'
    elif bad == 'filled_labels':
        blank[0]['human_labels'][0] = 0
    elif bad == 'wrong_label_width':
        blank[0]['human_labels'].append(None)
    elif bad == 'id_mismatch':
        blank.reverse()
    elif bad == 'auto_id_collision':
        selected[0]['review_id'] = blank[0]['review_id'] = prepared['slots'][0]['opaque_id']
    elif bad == 'stale_response':
        rows[0]['raw_decision']['response'] = '新增回答'
    elif bad == 'stale_slot_status':
        prepared['slots'][0]['status'] = 'unblindable'
    with pytest.raises(ValueError):
        prepare_human_form(rows, suite, selected, blank, prepared)
