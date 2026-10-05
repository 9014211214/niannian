"""Prospective mixed-version continuation after a preserved online format failure.

Existing first bodies and product fallback are immutable. Only undispatched
positions use the clarified prompt. This cannot pass the original frozen gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1br as base
from scripts import run_e1b_repair as common
from scripts import e1br_format_checkpoint as checkpoint_module
from scripts import e1br_format_qualification as qualification
from scripts.e1br_format_runtime import (
    FormatGuardMixin, FormatExperimentApp, terminal_format_failure, validate_format_ledger,
)
from aria.engine.policy.e1br_calls import validate_e1br_ledger
from aria.engine.policy.online_decision import ONLINE_ID_NAMESPACE_VERSION

CAMPAIGN = Path('artifacts/local/e1br-formal-v4')
PRIOR_E1B, PRIOR_KNOWN = Decimal('0.517068432'), Decimal('1.909447604')
RETAINED_UNKNOWN_HOLD = Decimal('0.323693014')
ALLOWANCE = Decimal('0.659238554')
PRIOR_DISPATCHES, RETAINED_MAIN, PRIOR_UNKNOWN = 916, 372, 9
MAX_WIRE_REQUESTS = 2067
NEW_COUNTS = {'main': {'online_fast': 654, 'memory': 15, 'reflection': 15}}
SELECTED, BUDGET_CHOICE, RunError = base.SELECTED, base.BUDGET_CHOICE, base.RunError
APPROVAL_PATH = Path('artifacts/local/tasks/20261005-02/human-continuation-approval-v1.json')
APPROVAL_SHA = 'f3d9d36f2bc386220c79600119c2488abbe3b87e0d012548ca8f601187d9ea66'


def approval(root):
    raw = common.read_original(Path(root), APPROVAL_PATH)
    if hashlib.sha256(raw).hexdigest() != APPROVAL_SHA:
        raise RunError('format_continuation_human_approval_changed')
    value = json.loads(raw)
    if not value.get('human_message_verbatim'):
        raise RunError('format_continuation_human_quote_missing')
    return {'approval_file_sha256': APPROVAL_SHA,
            'human_quote': value['human_message_verbatim'],
            'human_message_reference': '20261005-02',
            'scope': 'preserve_received_body_and_actual_fallback_continue_undispatched_mixed_version'}


def build_engine(root):
    root = Path(root).resolve()
    checkpoint = checkpoint_module.load_checkpoint(root)
    revision = common._load_module('scripts._e1br_format_revision', 'run_e1br.py')
    revision.CAMPAIGN, revision.PRIOR_E1B, revision.PRIOR_KNOWN = CAMPAIGN, PRIOR_E1B, PRIOR_KNOWN
    revision.ALLOWANCE, revision.PRIOR_DISPATCHES = ALLOWANCE, PRIOR_DISPATCHES
    revision.RETAINED_MAIN, revision.PRIOR_UNKNOWN = RETAINED_MAIN, PRIOR_UNKNOWN
    revision.RETAINED_UNKNOWN_HOLD = RETAINED_UNKNOWN_HOLD
    revision.MAX_WIRE_REQUESTS, revision.NEW_COUNTS = MAX_WIRE_REQUESTS, NEW_COUNTS
    revision.PARENT_MANIFEST_SHA = checkpoint_module.MANIFEST_SHA
    revision.approval = approval
    revision.parent_originals = lambda current: checkpoint_module.inherited_files(current, checkpoint)
    # The base builder supplies the original thirteen itself.
    revision.retained_extra_rows = lambda _: deepcopy(checkpoint['retained_attempts'][13:])
    revision.EXTRA_FILES = (
        'scripts/run_e1br_received_continuation.py', 'scripts/e1br_received_body.py',
        'scripts/e1br_checkpoint.py', 'scripts/e1br_checkpoint_guard.py',
        'scripts/run_e1br_checkpoint_continuation.py',
        'scripts/e1br_format_checkpoint.py', 'scripts/e1br_format_runtime.py',
        'scripts/e1br_format_legacy.py',
        'scripts/e1br_format_qualification.py', 'scripts/run_e1br_format_continuation.py',
        'docs/E1BR_FORMAT_CONTINUATION.md',
    )
    positions = (base.position_inventory() -
                 {tuple(row['experiment_position']) for row in checkpoint['retained_attempts']
                  if 'experiment_position' in row})
    if len(positions) != 684 or Counter(row[0] for row in positions) != Counter(NEW_COUNTS['main']):
        raise RunError('format_continuation_remaining_position_count_changed')
    revision.position_inventory = lambda: frozenset(positions)
    revision.continuation_details = lambda _: {
        'stopped_checkpoint': deepcopy(checkpoint['summary']),
        'execution_readiness': 'approved_undispatched_only_mixed_version',
        'online_namespace_prompt_version': ONLINE_ID_NAMESPACE_VERSION,
        'continuation_measurement': 'exploratory_mixed_versions_original_gates_unchanged',
        'retained_known_format_failure': 1,
        'online_format_failure_policy': 'known_cost_complete_first_body_strict_schema_rejected_terminal_fallback_no_retry',
        'remaining_physical_derivation': '684_initial_plus_15_reflection_corrections_plus_1368_no_body_retries',
        'unknown_cost_policy': 'old_nine_unresolved_retained_future_authorized_no_body_attempts_keep_reservations',
        'measurement_denominators': {'logical_positions': 1056, 'reply_positions': 216, 'formation_cases': 12},
        'same_final_configuration_effect_claim_eligible': False,
    }
    revision.runner_for = lambda current, prefix: checkpoint_module.runner_class(checkpoint)

    def first(client, payload, scope, key, prefix, current):
        failed = checkpoint['failed']
        response_format = payload.get('response_format')
        namespace_required = response_format == {'type': 'json_object'}
        if (key != ('online_fast', 'e1b_c1', 2, 'shared', 0, 6)
                or scope['session_id'] != failed['session_id']
                or scope['operation_id'] != f"e1b-{failed['session_id']}-t6"
                or response_format not in (None, {'type': 'json_object'})
                or namespace_required and ONLINE_ID_NAMESPACE_VERSION not in payload['messages'][0]['content']):
            raise RunError('format_continuation_first_undispatched_position_changed')
        return {'continued_after_terminal_request_id': failed['request_id'],
                'parent_manifest_sha256': checkpoint_module.MANIFEST_SHA,
                'first_new_return_path': 'decision_json' if namespace_required else 'plain_text',
                'old_request_replayed': False, 'old_failure_replaced': False}
    revision.first_request_lineage = first
    entry = revision.build_engine(root)

    format_contract = {
        'schema': 'e1br_known_format_terminal_v1',
        'retry_answer_bearing_failure': False, 'effect_claim_eligible': False,
        'approval_sha256': APPROVAL_SHA,
        'parent_manifest_sha256': checkpoint_module.MANIFEST_SHA,
        'prompt_namespace_version': ONLINE_ID_NAMESPACE_VERSION,
    }

    class FormatGuard(FormatGuardMixin, entry.EvidenceGuard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, format_contract=deepcopy(format_contract),
                             format_evidence_root=entry.campaign(root), **kwargs)

    def assert_clean(state, *, expected_calls=None, final=False):
        rows = [row for row in entry.logical_attempts(state) if row['product_step'] == 'initial']
        complete = len(rows) == 684 and Counter(row['role'] for row in rows) == Counter(NEW_COUNTS['main'])
        final_limit = complete and state.get('stop_reason') in {'request_limit_reached', 'inclusive_cost_limit_reached'}
        if ((state.get('stopped') and not final_limit)
                or len(state['attempts']) != state['calls_used']
                or any(row['status'] == 'pending' for row in state['attempts'])
                or any(row['status'] != 'received' and not terminal_format_failure(row)
                       for row in rows if row['role'] != 'reflection')
                or expected_calls is not None and len(rows) != expected_calls):
            raise RunError('format_continuation_primary_or_accounting_incomplete')
        if expected_calls is not None and not complete:
            raise RunError('format_continuation_basic_denominator_changed')
        for row in state['attempts']:
            revision.validate_position_binding(tuple(row.get('experiment_position', ())), row, state['attempts'])
        if Decimal(state['known_cost_usd'] or '0') + Decimal(state['reserved_cost_usd']) > ALLOWANCE:
            raise RunError('format_continuation_inclusive_budget_exceeded')

    def validate_closed_ledger(current, frozen, state):
        if Path(current).resolve() != root:
            raise RunError('format_continuation_workspace_changed')
        rows = validate_e1br_ledger(entry.campaign(root) / 'allowance.sqlite3', stage='main',
            registration_sha256=frozen['seal_sha256'], max_requests=MAX_WIRE_REQUESTS,
            reservation_policy=frozen['metadata']['policy'])
        if rows != state['attempts']:
            raise RunError('format_continuation_closed_ledger_changed')
        validate_format_ledger(entry.campaign(root) / 'allowance.sqlite3',
                               format_contract=format_contract, attempts=rows)
        return rows

    def qualify(run_root, frozen, state):
        checkpoint_module.assert_originals(checkpoint, full=True)
        validate_closed_ledger(root, frozen, state)
        entry.check_wire_evidence(state)
        result = qualification.qualify(run_root, frozen, state,
            retained_attempts=deepcopy(checkpoint['retained_attempts']),
            retained_summary=deepcopy(checkpoint['summary']), logical_attempts=entry.logical_attempts,
            retained_position_bindings={row['request_id']: {
                'receipt_sha256': base.evaluation.digest(row),
                'position': [row['role'], 'e1b_a1', 1, 'shared', 0,
                             index if row['role'] == 'online_fast' else None],
            } for index, row in enumerate(checkpoint['prefix']['retained_attempts'])})
        all_initial = [*checkpoint['retained_attempts'], *(
            row for row in entry.logical_attempts(state) if row['product_step'] == 'initial')]
        registrations = {row['request_id']: row['registration_sha256'] for row in all_initial}
        for position in result['positions']:
            position['generation_registration_sha256'] = registrations[position['request_id']]
        result['source_lineage'] = {
            'schema': 'e1br_format_source_lineage_v1',
            'retained_first_answers': RETAINED_MAIN,
            'retained_original_manifest_sha256': checkpoint_module.MANIFEST_SHA,
            'retained_source_registration': deepcopy(checkpoint['summary']),
            'new_registration_sha256': frozen['seal_sha256'],
            'new_prompt_namespace_version': ONLINE_ID_NAMESPACE_VERSION,
            'initial_positions_by_registration': dict(Counter(registrations.values())),
            'all_initial_request_registrations': registrations,
            'mixed_version': True, 'old_frozen_gates_replaced': False,
        }
        return result

    entry.EvidenceGuard, entry.E1bExperimentApp = FormatGuard, FormatExperimentApp
    entry.assert_clean, entry.qualify = assert_clean, qualify
    entry.validate_closed_ledger, entry.stopped_checkpoint = validate_closed_ledger, checkpoint
    entry.format_contract = format_contract
    inherited_extensions = entry.receipt_extensions
    entry.receipt_extensions = lambda: {
        **inherited_extensions(), 'generation_scope': 'exploratory_mixed_versions',
        'old_frozen_gates_replaced': False, 'same_final_configuration_effect_claim_eligible': False,
    }
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute', 'assess'))
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-local-env', action='store_true')
    parser.add_argument('--budget-choice', choices=(BUDGET_CHOICE,))
    parser.add_argument('--campaign-authorization-sha256')
    parser.add_argument('--stage-authorization-sha256')
    args = parser.parse_args(argv)
    if args.command == 'assess':
        if args.output is None:
            parser.error('assess requires --output')
        value = qualification.assess(ROOT, revision=sys.modules[__name__])
        base.original.write_once(args.output, value)
        print(json.dumps({key: value.get(key) for key in (
            'status', 'main_gate', 'formation_gate', 'overall_gate', 'effect_claim_eligible')}, indent=2))
        return
    if args.metadata is None or args.registration is None:
        parser.error('prepare/verify/execute require --metadata and --registration')
    entry = build_engine(ROOT)
    if args.command == 'prepare':
        value = entry.prepare(ROOT, args.metadata, args.registration, stage='main')
    elif args.command == 'verify':
        value = entry.verify(ROOT, args.metadata, args.registration)
    else:
        value = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
            budget_choice=args.budget_choice,
            campaign_authorization_sha256=args.campaign_authorization_sha256,
            stage_authorization_sha256=args.stage_authorization_sha256)
        print(json.dumps(value, indent=2))
        return
    print(json.dumps({'seal_sha256': value['seal_sha256'], 'paid_calls': 0,
                      'retained_first_answers': RETAINED_MAIN, 'remaining_basic_positions': 684}))


if __name__ == '__main__':
    main()
