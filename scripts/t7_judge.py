"""Pure T6 judge preparation and aggregation. No credentials, I/O or model calls.

Every returned preparation object is PRIVATE: only ``batch['payload']`` may be
sent. UTF-8 bytes plus fixed framing are a conservative local reservation, not
measured provider tokens or a contractual provider billing bound.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import random
import re

from aria.bench.refocus_cases import VIOLATION_KEYS
from aria.bench.refocus_e1_runner import inventory

FRAMING_TOKEN_RESERVE = 8192
_SEEDS = ('raw_shuffle', 'delivered_shuffle', 'pair_shuffle', 'human_sample')
_LEAK = re.compile(
    r'(?<![a-z0-9])B(?:0|1|2|3a|3b)(?![a-z0-9])|'
    r'(?:online_turn_decision|mode|profile)_v\d+|prompt[ _-]?version|'
    r'(?<!\d)(?:1\.12\.0|1\.13\.0)(?!\d)|'
    r'\b(?:qwen|claude|anthropic|openrouter|gpt[- ]?\d|sonnet)\b|'
    r'\be1_[tp]\d+(?:_[a-z0-9]+)?\b', re.I,
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _rng(seed, namespace):
    value = hashlib.sha256(seed.to_bytes(16, 'big') + b':' + namespace.encode()).digest()
    return random.Random(int.from_bytes(value, 'big'))


def _validated_rows(rows, suite):
    expected = inventory(suite, seed=0)
    if len(rows) != len(expected):
        raise ValueError('judge_inventory_requires_all_204_positions')
    ordered = sorted(rows, key=lambda row: row.get('order', 0))
    for index, (row, coordinate) in enumerate(zip(ordered, expected), 1):
        if type(row.get('order')) is not int or row['order'] != index:
            raise ValueError('judge_invalid_execution_order')
        if any(type(row.get(key)) is not type(value) or row[key] != value
               for key, value in coordinate.items()):
            raise ValueError('judge_coordinate_not_frozen_inventory')
    return ordered


def _projection(case):
    return {
        'context': case.model_input(),
        'confirmed_guidance': [item.model_dump(mode='json', exclude={'id'})
                               for item in case.guidance],
        'criteria': case.review.model_dump(mode='json'),
    }


def _empty(case):
    return [None if getattr(case.review, key).applicable else 0 for key in VIOLATION_KEYS]


def _text(row, stage):
    raw = row.get('raw_decision')
    return ((raw.get('response') if isinstance(raw, dict) else None)
            if stage == 'raw' else row.get('delivered_response'))


def _availability(row, text, model_names):
    if row.get('status') == 'not_started':
        return 'not_started'
    if row.get('status') != 'generated' or not isinstance(text, str) or not text.strip():
        return 'not_evaluable'
    if _LEAK.search(text) or any(model in text.casefold() for model in model_names):
        return 'unblindable'
    return 'pending'


def _validate_contract(plan, contract):
    inv, request = contract['inventory'], contract['model_request']
    if (plan['id'] != contract['freeze_id'] or inv['main_generation_denominator'] != 204
            or inv['raw_review_slots'] != 204 or inv['delivered_review_slots'] != 204
            or inv['paired_comparison_slots'] != 27
            or contract['binary_review']['label_order'] != list(VIOLATION_KEYS)
            or request['max_tokens'] != 512 or request['model'] != 'anthropic/claude-sonnet-5'
            or request['provider'] != {'only': ['anthropic'], 'allow_fallbacks': False,
                                      'require_parameters': True, 'data_collection': 'deny'}
            or request['reasoning'] != {'enabled': False, 'exclude': True}
            or Decimal(str(plan['judge']['max_cost_usd'])) != Decimal('0.3')
            or Decimal(contract['authorization']['judge_stop_cost_usd']) != Decimal('0.3')):
        raise ValueError('judge_contract_not_supported')


def build_batches(rows, suite, plan, contract, seeds):
    """Build all fixed review slots/batches and a whole-stage budget reservation.

    IDs, coordinates, stage and reuse provenance stay in the private map. Payloads
    contain frozen system instructions, review projections and opaque IDs only.
    A blocked result preserves every planned batch; callers must send none.
    """
    rows = _validated_rows(rows, suite)
    _validate_contract(plan, contract)
    if set(seeds) != set(_SEEDS) or any(type(seeds[key]) is not int or
                                      not 0 <= seeds[key] < 2**128 for key in _SEEDS):
        raise ValueError('judge_requires_four_128_bit_seeds')
    if len(set(seeds.values())) != 4:
        raise ValueError('judge_requires_independent_stage_seeds')
    cases = {case.id: case for case in suite.main_cases()}
    model_names = {row['actual_model'].casefold() for row in rows
                   if isinstance(row.get('actual_model'), str) and row['actual_model']}
    used_ids = set()

    def opaque(rng):
        while True:
            value = f'q_{rng.getrandbits(48):012x}'
            if value not in used_ids:
                used_ids.add(value)
                return value

    # Reserve a derived human namespace. The caller must also check the exact
    # frozen human-form IDs; their original spelling is not reconstructed here.
    human_rng = _rng(seeds['human_sample'], 'human_form_ids')
    for _ in range(40):
        opaque(human_rng)
    slots, batches, reuse, by_stage = [], [], [], {}
    request = contract['model_request']

    def make_batch(stage, group, items, payload_data, system):
        payload = {key: deepcopy(request[key]) for key in
                   ('model', 'provider', 'reasoning', 'max_tokens', 'stream')}
        payload['messages'] = [{'role': 'system', 'content': system},
                               {'role': 'user', 'content': canonical(payload_data).decode()}]
        # Count all JSON bytes, not characters, including envelope + both messages.
        size = len(canonical(payload))
        upper = size + FRAMING_TOKEN_RESERVE
        reserve = (Decimal(upper) * Decimal(request['input_price_usd_per_million_tokens'])
                   + Decimal(512) * Decimal(request['output_price_usd_per_million_tokens'])) / Decimal(1000000)
        batch_id = f'{stage}_{len(batches) + 1:02d}'
        batch = {'batch_id': batch_id, 'stage': stage, 'payload': payload,
                 'payload_sha256': _hash(payload), 'reserve_usd': str(reserve),
                 'input_token_upper_bound': upper, 'payload_utf8_bytes': size,
                 'framing_token_reserve': FRAMING_TOKEN_RESERVE,
                 'private': {'batch_id': batch_id, 'stage': stage,
                             'group': group, 'items': items}}
        batches.append(batch)
        return batch

    for stage, seed_key in (('raw', 'raw_shuffle'), ('delivered', 'delivered_shuffle')):
        rng = _rng(seeds[seed_key], f'{stage}_row_order_and_ids')
        grouped = {}
        for row in rows:
            case = cases[row['case_id']]
            text = _text(row, stage)
            projection = _projection(case)
            status = _availability(row, text, model_names)
            slot = {'order': row['order'], 'case_id': row['case_id'], 'arm': row['arm'],
                    'repeat': row['repeat'], 'stage': stage, 'status': status,
                    'labels': _empty(case), 'opaque_id': opaque(rng),
                    'applicable': [getattr(case.review, key).applicable for key in VIOLATION_KEYS],
                    'review_sha256': _hash({**projection, 'response': text}),
                    'response_sha256': hashlib.sha256(text.encode()).hexdigest()
                    if isinstance(text, str) else None, 'projection_sha256': _hash(projection)}
            slots.append(slot)
            by_stage[stage, row['order']] = slot
            if stage == 'delivered' and status == 'pending':
                source = by_stage['raw', row['order']]
                if source['status'] == 'pending' and source['review_sha256'] == slot['review_sha256']:
                    slot['status'] = 'awaiting_identical_raw_review'
                    reuse.append({'source_stage': 'raw', 'source_order': row['order'],
                                  'destination_stage': 'delivered', 'destination_order': row['order'],
                                  'source_review_sha256': source['review_sha256'],
                                  'destination_review_sha256': slot['review_sha256'],
                                  'source_opaque_id': source['opaque_id'],
                                  'destination_opaque_id': slot['opaque_id']})
            if slot['status'] == 'pending':
                grouped.setdefault(case.id, []).append((slot, text))
        pending = []
        for case_id, grouped_items in grouped.items():
            rng.shuffle(grouped_items)
            projection = _projection(cases[case_id])
            data = {**projection, 'items': [[slot['opaque_id'], text] for slot, text in grouped_items]}
            items = {slot['opaque_id']: deepcopy(slot) for slot, _ in grouped_items}
            pending.append(make_batch(stage, case_id, items, data, contract['binary_review']['system_prompt']))
        _rng(seeds[seed_key], f'{stage}_batch_order').shuffle(pending)
        # Only the independent batch shuffle determines the send sequence.
        if pending:
            del batches[-len(pending):]
        batches.extend(pending)

    pair_slots = []
    pair_rng = _rng(seeds['pair_shuffle'], 'pair_row_order_and_ids')
    person_rng = _rng(seeds['pair_shuffle'], 'pair_person_order')
    row_by_coord = {(row['case_id'], row['arm'], row['repeat']): row for row in rows}
    pending = []
    for pair in suite.pairs:
        data_items, private_items = [], {}
        for arm in contract['inventory']['paired_arms']:
            for repeat in contract['inventory']['repeats']:
                members = [row_by_coord[case.id, arm, repeat] for case in pair.cases]
                sources = [by_stage['raw', row['order']] for row in members]
                slot = {'opaque_id': opaque(pair_rng), 'pair_id': pair.id, 'arm': arm,
                        'repeat': repeat, 'key': f'{pair.id}:{arm}:{repeat}',
                        'orders': [row['order'] for row in members], 'label': None,
                        'status': 'pending' if all(s['status'] == 'pending' for s in sources)
                        else 'not_evaluable'}
                pair_slots.append(slot)
                if slot['status'] != 'pending':
                    continue
                people = [{**_projection(case), 'response': _text(row, 'raw')}
                          for case, row in zip(pair.cases, members)]
                order = [0, 1]
                person_rng.shuffle(order)
                slot['person_order'] = order
                data_items.append({'opaque_id': slot['opaque_id'], 'people': [people[i] for i in order]})
                private_items[slot['opaque_id']] = deepcopy(slot)
        if data_items:
            pair_rng.shuffle(data_items)
            data = {'comparison': pair.review.comparison, 'not_required': pair.review.not_required,
                    'items': data_items}
            pending.append(make_batch('pairs', pair.id, private_items, data, contract['pair_review']['system_prompt']))
    _rng(seeds['pair_shuffle'], 'pair_batch_order').shuffle(pending)
    if pending:
        del batches[-len(pending):]
    batches.extend(pending)
    if len(batches) > 35:
        raise ValueError('judge_batch_inventory_exceeds_contract')
    total = sum((Decimal(batch['reserve_usd']) for batch in batches), Decimal(0))
    return {'schema': 'aria_t7_judge_preparation_v1', 'status': 'blocked_budget' if total > Decimal('0.30') else 'prepared',
            'freeze_id': plan['id'], 'batches': batches, 'slots': slots, 'pair_slots': pair_slots,
            'reuse': reuse, 'total_reserve_usd': str(total), 'max_cost_usd': '0.30',
            'budget_method': 'payload_utf8_bytes_plus_8192_framing_tokens_no_cache_discount',
            'provider_token_bound_verified': False, 'provider_billing_guarantee': False,
            'requires_verified_provider_bound_before_send': True,
            'seed_commitment_verified_here': False,
            'notes': ['Caller must verify exact private seed JSON against the frozen commitment.',
                      'Batch-internal contrasts may affect judge decisions; not independent human ratings.']}


def parse_batch(batch, content):
    """Validate an entire batch atomically; malformed answers never yield labels."""
    if type(content) is not str:
        raise ValueError('judge_content_must_be_text')
    try:
        output = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError('judge_invalid_json') from exc
    expected = batch['private']['items']
    is_pair = batch['stage'] == 'pairs'
    width, result = (2 if is_pair else 5), {}
    if type(output) is not list:
        raise ValueError('judge_expected_array')
    for row in output:
        if type(row) is not list or len(row) != width or type(row[0]) is not str:
            raise ValueError('judge_invalid_row')
        key = row[0]
        if key not in expected or key in result:
            raise ValueError('judge_invalid_id_set')
        if any(label is not None and (type(label) is not int or label not in (0, 1)) for label in row[1:]):
            raise ValueError('judge_invalid_label')
        if not is_pair and any(not applicable and label != 0 for applicable, label in
                               zip(expected[key]['applicable'], row[1:])):
            raise ValueError('judge_inapplicable_label_must_be_zero')
        result[key] = row[1] if is_pair else row[1:]
    if set(result) != set(expected):
        raise ValueError('judge_invalid_id_set')
    return result


def apply_results(prepared, batch_results):
    """Apply strict raw content by batch_id; omitted batches stay unevaluated.

    Return label maps keyed by string execution order (JSON round-trip stable),
    paired labels keyed by ``pair_id:arm:repeat`` and private provenance.
    """
    batches = {batch['batch_id']: batch for batch in prepared['batches']}
    if set(batch_results) - set(batches):
        raise ValueError('judge_unknown_result_batch')
    slots = deepcopy(prepared['slots'])
    by_id = {slot['opaque_id']: slot for slot in slots}
    pair_slots = deepcopy(prepared['pair_slots'])
    pair_by_id = {slot['opaque_id']: slot for slot in pair_slots}
    for batch_id, content in batch_results.items():
        parsed = parse_batch(batches[batch_id], content)
        for key, value in parsed.items():
            slot = pair_by_id[key] if batches[batch_id]['stage'] == 'pairs' else by_id[key]
            slot['label' if batches[batch_id]['stage'] == 'pairs' else 'labels'] = value
            slot.update(status='separately_judged', source_batch_id=batch_id)
    for record in prepared['reuse']:
        source, destination = by_id[record['source_opaque_id']], by_id[record['destination_opaque_id']]
        if source['status'] == 'separately_judged':
            destination.update(labels=deepcopy(source['labels']), status='reused_identical_text',
                               source_batch_id=source['source_batch_id'],
                               source_opaque_id=source['opaque_id'])
    return {'labels': {stage: {str(slot['order']): slot['labels'] for slot in slots if slot['stage'] == stage}
                       for stage in ('raw', 'delivered')},
            'pairlabels': {slot['key']: slot['label'] for slot in pair_slots},
            'slots': slots, 'pair_slots': pair_slots, 'reuse': deepcopy(prepared['reuse'])}


def prepare_human_form(rows, suite, sample, empty_form, prepared):
    """Fill the fixed human form with blind raw texts, never judge answers.

    The caller verifies the original sample/form/seed byte commitments first.
    This function preserves all 40 committed IDs and their order, including
    unavailable or unblindable responses, and leaves every human label null.
    """
    rows = _validated_rows(rows, suite)
    if (type(sample) is not list or type(empty_form) is not list
            or len(sample) != 40 or len(empty_form) != 40):
        raise ValueError('human_form_requires_all_40_positions')
    slots = prepared.get('slots')
    pair_slots = prepared.get('pair_slots')
    if (type(slots) is not list or len(slots) != 408
            or type(pair_slots) is not list or len(pair_slots) != 27):
        raise ValueError('human_form_invalid_preparation_inventory')
    auto_ids = [slot.get('opaque_id') for slot in (*slots, *pair_slots)]
    if (any(type(value) is not str or not value for value in auto_ids)
            or len(set(auto_ids)) != 435):
        raise ValueError('human_form_invalid_automatic_ids')
    raw_slots = {slot.get('order'): slot for slot in slots if slot.get('stage') == 'raw'}
    if set(raw_slots) != set(range(1, 205)):
        raise ValueError('human_form_invalid_raw_slot_inventory')
    cases = {case.id: case for case in suite.main_cases()}
    model_names = {row['actual_model'].casefold() for row in rows
                   if isinstance(row.get('actual_model'), str) and row['actual_model']}
    human_ids, coordinates, result = set(), set(), []
    for selected, blank in zip(sample, empty_form):
        if (type(selected) is not dict or set(selected) != {
                'review_id', 'inventory_order', 'case_id', 'arm', 'repeat'}
                or type(blank) is not dict or set(blank) != {
                'review_id', 'status', 'response', 'human_labels'}):
            raise ValueError('human_form_invalid_source_schema')
        identifier = selected['review_id']
        order = selected['inventory_order']
        if (type(identifier) is not str or not identifier or len(identifier) > 64
                or identifier != blank['review_id'] or identifier in human_ids
                or identifier in auto_ids or type(order) is not int
                or not 1 <= order <= 204 or order in coordinates):
            raise ValueError('human_form_invalid_id_or_coordinate')
        if (blank['status'] != 'awaiting_generation_and_human_review'
                or blank['response'] is not None or type(blank['human_labels']) is not list
                or len(blank['human_labels']) != 4
                or any(value is not None for value in blank['human_labels'])):
            raise ValueError('human_form_source_must_be_unreviewed')
        row = rows[order - 1]
        raw_slot = raw_slots[order]
        for key in ('case_id', 'arm', 'repeat'):
            if (type(selected[key]) is not type(row[key]) or selected[key] != row[key]
                    or raw_slot.get(key) != row[key]):
                raise ValueError('human_form_coordinate_mismatch')
        response = _text(row, 'raw')
        status = _availability(row, response, model_names)
        response_hash = hashlib.sha256(response.encode()).hexdigest() if isinstance(response, str) else None
        if raw_slot.get('status') != status or raw_slot.get('response_sha256') != response_hash:
            raise ValueError('human_form_preparation_does_not_match_raw')
        human_ids.add(identifier)
        coordinates.add(order)
        result.append({'review_id': identifier,
                       'status': 'awaiting_human_review' if status == 'pending' else status,
                       **_projection(cases[row['case_id']]),
                       'response': response if status == 'pending' else None,
                       'human_labels': [None, None, None, None]})
    return result


def _decision(value):
    return {'status': 'not_evaluable' if value is None else 'pass' if value else 'fail', 'passed': value}


def _gate(values):
    return None if any(value is None for value in values) else all(values)


def aggregate(rows, labels, pairlabels, suite):
    """Report all fixed denominators. Never promote defaults or invent calibration."""
    rows = _validated_rows(rows, suite)
    cases = {case.id: case for case in suite.main_cases()}
    row_by_coord = {(row['case_id'], row['arm'], row['repeat']): row for row in rows}
    model_names = {row['actual_model'].casefold() for row in rows
                   if isinstance(row.get('actual_model'), str) and row['actual_model']}

    def vector(row, stage):
        values = labels.get(stage, {}).get(str(row['order']), _empty(cases[row['case_id']]))
        if type(values) is not list or len(values) != 4 or any(value is not None and
                (type(value) is not int or value not in (0, 1)) for value in values):
            raise ValueError('judge_invalid_aggregate_labels')
        applicable = [getattr(cases[row['case_id']].review, key).applicable for key in VIOLATION_KEYS]
        if any(not applies and value != 0 for applies, value in zip(applicable, values)):
            raise ValueError('judge_inapplicable_aggregate_label')
        if _availability(row, _text(row, stage), model_names) != 'pending':
            return _empty(cases[row['case_id']])
        return values

    stages = {}
    for stage in ('raw', 'delivered'):
        majority = {}
        missing = 0
        for case_id, case in cases.items():
            arms = ('B0', 'B1', 'B2', 'B3a', 'B3b') if case in suite.traps else ('B0', 'B2', 'B3b')
            majority[case_id] = {}
            for arm in arms:
                values = [vector(row_by_coord[case_id, arm, repeat], stage) for repeat in (1, 2, 3)]
                missing += sum(value is None for item in values for value in item)
                majority[case_id][arm] = [None if any(item[index] is None for item in values)
                                         else int(sum(item[index] for item in values) >= 2)
                                         for index in range(4)]

        def compare(case_ids, treatment, baseline):
            result = {}
            for case_id in case_ids:
                left, right = majority[case_id][treatment], majority[case_id][baseline]
                complete = None not in left and None not in right
                worse = any(a > b for a, b in zip(left, right)) if complete else None
                improved = (not worse and any(a < b for a, b in zip(left, right))) if complete else None
                result[case_id] = {'evaluable': complete, 'worse': worse, 'improved': improved}
            return result

        trap_ids = [case.id for case in suite.traps]
        filming = compare(trap_ids, 'B2', 'B0')
        v2 = compare(trap_ids, 'B2', 'B1')
        profile = compare(list(cases), 'B3b', 'B2')
        complete = lambda result: all(item['evaluable'] for item in result.values())
        filming_pass = (sum(filming[key]['improved'] for key in trap_ids[:8]) >= 6 and
                        not any(item['worse'] for item in filming.values())) if complete(filming) else None
        v2_pass = not any(item['worse'] for item in v2.values()) if complete(v2) else None
        profile_pass = (not any(item['worse'] for item in profile.values()) and
                        sum(profile[key]['improved'] for key in trap_ids) >= 2) if complete(profile) else None
        stages[stage] = {'slots': 204, 'binary_label_slots': 816, 'applicable_label_slots': 651,
                         'inapplicable_label_slots': 165, 'missing_applicable_labels': missing,
                         'majority': majority, 'filming_comparisons': filming, 'v2_comparisons': v2,
                         'profile_comparisons': profile, 'b3a_comparisons': compare(trap_ids, 'B3a', 'B2'),
                         'filming_gate': _decision(filming_pass), 'v2_response_gate': _decision(v2_pass),
                         'profile_response_gate': _decision(profile_pass)}

    pair_results = {}
    for pair in suite.pairs:
        pair_results[pair.id] = {}
        for arm in ('B0', 'B2', 'B3b'):
            repeats, comparisons = [], []
            for repeat in (1, 2, 3):
                label = pairlabels.get(f'{pair.id}:{arm}:{repeat}')
                if label is not None and (type(label) is not int or label not in (0, 1)):
                    raise ValueError('judge_invalid_pair_aggregate_label')
                vectors = [vector(row_by_coord[case.id, arm, repeat], 'raw') for case in pair.cases]
                result = (label == 1 and all(value == 0 for item in vectors for value in item)
                          if label is not None and all(None not in item for item in vectors) else None)
                repeats.append(result)
                comparisons.append({'repeat': repeat, 'judge_pass': label,
                                    'boundaries_all_zero': all(value == 0 for item in vectors for value in item)
                                    if all(None not in item for item in vectors) else None,
                                    'success': result})
            pair_results[pair.id][arm] = {'repeats': repeats, 'comparisons': comparisons, 'denominator': 3,
                                         **_decision(None if None in repeats else sum(repeats) >= 2)}
    pair_pass = _gate([pair_results[pair.id]['B3b']['passed'] for pair in suite.pairs])
    b2 = [row for row in rows if row['arm'] == 'B2']
    capture_count = sum(row.get('status') == 'generated' and row.get('deliberation_status') == 'captured' for row in b2)
    member_count = sum(row.get('status') == 'generated' and row.get('selected_in_deliberated_options') is True for row in b2)
    missing_capture = sum(row.get('status') != 'generated' or row.get('deliberation_status')
                          not in ('captured', 'missing', 'invalid') for row in b2)
    missing_membership = sum(row.get('status') != 'generated' or
                             (row.get('deliberation_status') == 'captured' and
                              type(row.get('selected_in_deliberated_options')) is not bool) or
                             row.get('deliberation_status') not in ('captured', 'missing', 'invalid') for row in b2)
    capture = {'denominator': 48, 'captured': capture_count, 'selected_in_candidates': member_count,
               'missing_capture': missing_capture, 'missing_membership': missing_membership,
               'capture_gate': _decision(None if missing_capture else capture_count >= 44),
               'membership_gate': _decision(None if missing_membership else member_count >= 44),
               **_decision(None if missing_capture or missing_membership else capture_count >= 44 and member_count >= 44)}
    latencies = {}
    for arm in ('B1', 'B2'):
        values = [row.get('latency_ms') for row in rows if row['arm'] == arm and row['case_id'] in trap_ids]
        valid = [value for value in values if type(value) in (int, float) and math.isfinite(value) and value >= 0]
        generated = all(row.get('status') == 'generated' for row in rows
                        if row['arm'] == arm and row['case_id'] in trap_ids)
        valid.sort()
        latencies[arm] = {'denominator': 30, 'observed': len(valid),
                          'p50_ms': valid[math.ceil(.50 * 30) - 1] if len(valid) == 30 and generated else None,
                          'p95_ms': valid[math.ceil(.95 * 30) - 1] if len(valid) == 30 and generated else None}
    delta = {key: latencies['B2'][key] - latencies['B1'][key]
             if latencies['B2'][key] is not None and latencies['B1'][key] is not None else None
             for key in ('p50_ms', 'p95_ms')}
    latency_pass = (delta['p50_ms'] <= 800 and delta['p95_ms'] <= 1200) if None not in delta.values() else None
    return {'schema': 'aria_t7_e1_judge_aggregate_v1', 'generation_denominator': 204,
            'stages': stages, 'pairs': pair_results, 'pair_comparison_denominator': 27,
            'profile_pair_gate': _decision(pair_pass), 'b2_capture': capture,
            'latency': {'arms': latencies, 'difference': delta, **_decision(latency_pass)},
            'gates': {'filming': stages['raw']['filming_gate'],
                      'v2': _decision(_gate([stages['raw']['v2_response_gate']['passed'], capture['passed'], latency_pass])),
                      'profile': _decision(_gate([stages['raw']['profile_response_gate']['passed'], pair_pass]))},
            'human_calibration': {'status': 'not_completed', 'planned_slots': 40, 'assessed': 0},
            'interpretation': 'uncalibrated_model_judge', 'automatic_default_switch': False,
            'selected_intervention_reply_alignment': 'unassessed_requires_separate_semantic_review'}
