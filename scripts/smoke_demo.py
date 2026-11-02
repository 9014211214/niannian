"""One fictional three-session smoke run through the real local product.

No launch review, generated user simulator, scoring service or automatic rerun.
Existing outputs are never reused. --fixture exercises plumbing with labelled
software answers; --live requires an explicit price policy and bounded budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OWNER = 'fictional-chen'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def recovery_binding(prior_path, source_paths, *, prior_occupied, prior_new_spend, stop_path):
    """Bind a fresh allowance to retained task ledgers and an unchanged source set.

    This only opts in to bounded transport recovery; it never resumes an old
    smoke, releases a hold, or treats an absent provider amount as zero.
    """
    import sqlite3
    prior_path = Path(prior_path).resolve()
    prior = read(prior_path)
    bound = {str(prior_path): digest(prior_path.read_bytes())}
    def checked(ref):
        path = Path(ref['path']).resolve()
        if digest(path.read_bytes()) != ref['sha256']:
            raise ValueError('recovery_prior_evidence_changed')
        bound[str(path)] = ref['sha256']
        return path
    floor = read(checked(prior['financial_floor']))
    if (Decimal(floor['committed_usd']) != Decimal(prior_occupied)
            or Decimal(floor['task_incremental_committed_usd']) != Decimal(prior_new_spend)):
        raise ValueError('recovery_prior_budget_mismatch')
    attempts, identities = [], set()
    for ref in prior['task_ledgers']:
        path = checked(ref)
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
            rows = [json.loads(r[0]) for r in db.execute('SELECT record FROM attempts ORDER BY sequence')]
        for row in rows:
            if row['request_id'] in identities or row['status'] == 'pending':
                raise ValueError('recovery_prior_unresolved_or_duplicate')
            identities.add(row['request_id']); attempts.append(row)
    unknown = sum(r.get('http_status') == 429 and 'cost' not in r.get('usage', {}) for r in attempts)
    if unknown != prior['unknown_429_count']:
        raise ValueError('recovery_prior_count_mismatch')
    source = {str(Path(p).resolve()): digest(Path(p).read_bytes()) for p in source_paths}
    source_hash = digest(json.dumps(source, sort_keys=True, separators=(',', ':')).encode())
    expected_aria = {str(p.resolve()) for p in ROOT.joinpath('aria').rglob('*.py')}
    def unchanged():
        if Path(stop_path).exists():
            raise ValueError('requested_stop')
        if expected_aria != {str(p.resolve()) for p in ROOT.joinpath('aria').rglob('*.py')}:
            raise ValueError('recovery_source_set_changed')
        if any(digest(Path(p).read_bytes()) != h for p, h in {**bound, **source}.items()):
            raise ValueError('recovery_source_or_authority_changed')
        return True
    unchanged()
    return ({'schema':'unknown_429_hold_recovery_v1',
             'prior_unknown_429_count':unknown,'prior_evidence_sha256':bound[str(prior_path)],
             'source_manifest_sha256':source_hash}, unchanged, source)


def record_model_outputs(clients, output):
    """Retain each logical public completion, including rejected first outputs.

    The product ledger owns physical sends and billing. These records contain
    fictional prompts and public content only, never headers or hidden reasoning.
    """
    from threading import Lock
    lock, counter = Lock(), [0]
    for client in (clients.online,clients.memory,clients.reflection,clients.safety_review):
        if client is None:
            continue
        original=client.complete
        def recorded(messages, _client=client, _complete=original, **requirements):
            with lock:
                counter[0]+=1; number=counter[0]
            entry={'role':_client.role,'configured_model':_client.model,
                   'started_at':datetime.now(timezone.utc).isoformat(),
                   'messages':deepcopy(messages),'requirements':requirements,
                   'status':'pending','public_output':None}
            try:
                result=_complete(messages,**requirements)
                entry.update(status='returned',public_output=result)
                return result
            except Exception as error:
                entry.update(status='raised',error_class=type(error).__name__)
                raise
            finally:
                entry.update(finished_at=datetime.now(timezone.utc).isoformat(),
                    effective_model=getattr(_client,'last_model',None),
                    usage=deepcopy(getattr(_client,'last_usage',{})),
                    finish_reason=getattr(_client,'last_finish_reason',None))
                write(Path(output)/f'model-completion-{number:03d}.json',entry)
        client.complete=recorded


def display_prelude(output, turn_id, event):
    row=next(r for r in event['state']['turns'] if r['turn_id']==turn_id)
    protection=row['protection_prelude']['protection']
    write(Path(output)/f'prelude-{turn_id}.json',event)
    print(protection,flush=True)
    return True


def display_pending_preference(app, output, session_id, event_number):
    """Render the current owner invitation before issuing its delivery ACK.

    This is surface delivery only: it never supplies an owner answer or changes
    a model classification. The same path handles initial and renewed questions.
    """
    invitation = app.state().get('preference_invitation')
    if not invitation or invitation.get('status') != 'awaiting_display':
        return None
    if (not session_id or invitation.get('session_id') != session_id
            or type(invitation.get('question')) is not str
            or digest(invitation['question'].encode('utf-8')) != invitation.get('question_sha256')):
        raise ValueError('preference_display_binding_invalid')
    write(Path(output)/f'invitation-{event_number}.json', invitation)
    print(invitation['question'], flush=True)
    receipt = app.preference_invitation_ack(
        invitation['id'], invitation['question_sha256'], session_id)
    write(Path(output)/f'preference-delivery-ack-{event_number}.json', receipt)
    if not receipt.get('ok'):
        raise RuntimeError('preference_display_ack_failed')
    return receipt


def select_story(value, index=0, order=None):
    repeats = deepcopy(value.get('repetitions', [value]))
    if order is not None:
        # Only accept an already-authored permutation, with both exact digests.
        for patch in order['patches']:
            target = next(s for r in repeats if r['repeat_id'] == patch['repeat_id']
                          for s in r['sessions'] if s['session_id'] == patch['session_id'])
            canonical = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            original, effective = target['operations'], patch['effective_operations']
            if (digest(canonical(original)) != patch['original_operations_sha256']
                    or digest(canonical(effective)) != patch['effective_operations_sha256']
                    or sorted(map(canonical, original)) != sorted(map(canonical, effective))):
                raise ValueError('story_order_changed_content')
            target['operations'] = deepcopy(effective)
    repeat = repeats[index]
    if len(repeat['sessions']) != 3 or any(not 10 <= sum(
            op['op'] in ('say', 'answer_preference') for op in s['operations']) <= 12
            for s in repeat['sessions']):
        raise ValueError('smoke_requires_three_sessions_of_ten_to_twelve_turns')
    return repeat


def run_story(app, repeat, output, *, fixture=False):
    """Use owner APIs; never seed preferences, facts or person hypotheses."""
    # The formal runner imports this function too. Make the native output
    # contract explicit before printing any invitation or acknowledging it.
    for stream in (sys.stdout, sys.stderr):
        configure = getattr(stream, 'reconfigure', None)
        if configure is not None:
            configure(encoding='utf-8', errors='strict', write_through=True)
    output = Path(output)
    events, reviews, slots, sources, shares = [], {}, {}, {}, {}
    sid = None
    contact = None
    success = False
    initial = app.state()
    if initial['turns'] or any(initial['profile'][k] for k in ('facts','hypotheses','stated_preferences')):
        raise ValueError('smoke_requires_blank_profile')
    write(output/'initial.json', initial)
    try:
        for session in repeat['sessions']:
            key = session['session_id']
            for op in session['operations']:
                name = op['op']; before = app.state(); extra = {}
                if name == 'start_session':
                    result = app.start(); sid = result['state']['session_id']
                elif name in ('say','answer_preference'):
                    turn_id = f"smoke_{key}_{op['turn_id']}"
                    sources[f"{key}.{op['turn_id']}"] = turn_id
                    def prelude(event):
                        # Persist and display before acknowledgement or model review.
                        return display_prelude(output,turn_id,event)
                    result = app.submit(op['text'], turn_id, sid, on_safety_prelude=prelude)
                    state = app.state()
                    rows = [r for r in state['turns'] if r['turn_id'] == turn_id]
                    if rows:
                        print(f"[{key}/{op['turn_id']}] {rows[0].get('response', '')}", flush=True)
                    pair = app._sessions.get(OWNER)
                    if pair:
                        wm = pair[1].working_memory
                        extra['turn_audit'] = {k: deepcopy(wm[k]) for k in (
                            'online_turn_decision','preference_dialogue_receipt','prompt_evidence_snapshot',
                            'safety_review','safety_delivery') if k in wm}
                    if result.get('ok'):
                        delivery = display_pending_preference(app, output, sid, len(events)+1)
                        if delivery is not None:
                            extra['preference_invitation_delivery'] = delivery
                elif name == 'end_session_review':
                    sealed = app.seal(sid)
                    reviewed = app.reflect(f'smoke_{key}_review', wait=True) if sealed.get('ok') else None
                    result = {'ok': bool(sealed.get('ok') and reviewed and reviewed.get('ok')),
                              'seal': sealed, 'review': reviewed}
                    reviews[op['bind_review_as']] = sid
                    sid = None
                elif name == 'invite_preference':
                    candidates = sorted([p for p in app.state()['profile']['stated_preferences']
                        if p['status'] == 'proposed' and p.get('can_confirm') is True
                        and any(q['session_id'] == reviews.get(op['source_review']) for q in p['quotes'])],
                        key=lambda p: (p['created_at'], p['id']))
                    if not candidates:
                        result = {'ok': False, 'missing_reason': 'no_actual_candidate'}
                    else:
                        row = candidates[0]; slots[op['target_slot']] = row['id']
                        result = app.preference_invitation(row['id'], sid)
                        if result.get('ok'):
                            delivery = display_pending_preference(app, output, sid, len(events)+1)
                            if delivery is not None:
                                result = delivery
                elif name == 'revoke_preference':
                    ids = {slots.get(op['target_slot'])}
                    prefs = app.state()['profile']['stated_preferences']
                    while True:
                        children = {p['id'] for p in prefs if p.get('supersedes') in ids}
                        if not children - ids: break
                        ids.update(children)
                    withdrawals = [app.preference_action('withdraw', p['id']) for p in prefs
                                   if p['id'] in ids and p['status'] == 'confirmed']
                    result = {'ok': bool(withdrawals) and all(r.get('ok') for r in withdrawals),
                              'withdrawals': withdrawals}
                elif name == 'sharing_preview':
                    if op.get('mode') != 'local_mock_only' or op['recipient'].get('address') is not None:
                        raise ValueError('external_recipient_forbidden')
                    source_id = sources.get(op['selection']['source'].removesuffix('.exact_user_selected_quote'))
                    source = next((r for r in app.sharing_state()['sources'] if r['turn_id'] == source_id), None)
                    selected = op['selection']['text']
                    if not source or source['text'].count(selected) != 1:
                        result = {'ok': False, 'missing_reason': 'selected_source_unavailable'}
                    else:
                        if contact is None:
                            if not app.care.elder_view(OWNER)['configured']:
                                app.care.bootstrap('smoke_local_admin', 'fictional-smoke-only-2026', [OWNER])
                            app.care_action('contact.create', {'request_id':'smoke_contact',
                                'name':'虚构本地模拟收件箱','relation':'演示','channel':'local'})
                            contact = next(r for r in app.sharing_state()['contacts'] if r['name']=='虚构本地模拟收件箱')
                        payload = {'request_id':'smoke_share_prepare','source_id':source_id,'contact_id':contact['id']}
                        if source['text'] != selected:
                            start = source['text'].index(selected)
                            payload['selection'] = {'start':start,'end':start+len(selected),'unit':'unicode_codepoint'}
                        result = app.sharing_action('prepare', payload, sid)
                        if result.get('ok'): shares[op['bind_artifact_as']] = result['record']
                elif name in ('sharing_approve','sharing_send','sharing_revoke'):
                    prior = shares.get(op['artifact_ref'])
                    if prior is None:
                        result = {'ok':False,'missing_reason':'no_actual_preview'}
                    else:
                        action = name.removeprefix('sharing_')
                        data = {'request_id':f'smoke_share_{len(events)}','id':prior['id']}
                        if action != 'revoke': data['preview_sha256'] = prior['preview_sha256']
                        if action == 'approve': data['confirmed'] = True
                        try:
                            result = app.sharing_action(action,data,sid)
                        except (PermissionError,ValueError) as exc:
                            result = {'ok':False,'error_type':type(exc).__name__,
                                      'expected_denial':op.get('attempt_kind')=='post_revocation_denial_probe'}
                        if result.get('record'): shares[op['artifact_ref']] = result['record']
                else:
                    raise ValueError('unknown_smoke_operation')
                event = {'sequence':len(events)+1,'at':datetime.now(timezone.utc).isoformat(),
                         'session_key':key,'operation':op,'result':result,'before':before,
                         'after':app.state(),**extra}
                events.append(event)
                write(output/f'event-{len(events):03d}.json',event)
                if app.call_guard is not None and app.call_guard.snapshot().get('stopped'):
                    raise RuntimeError('product_budget_or_transport_stopped')
                # Missing model-produced preference is a visible outcome; no fabricated substitute.
        success = True
    finally:
        state = app.state()
        summary = {'evidence_class':'software_fixture' if fixture else 'fictional_model_smoke',
                   'operations_completed':success,'operation_count':len(events),
                   'turns':len(state['turns']),'sessions':len(state['sessions']),
                   'model_usage':state['model_usage'],'preferences':state['profile']['stated_preferences'],
                   'hypotheses':state['profile']['hypotheses'],'offline':state['offline'],
                   'scored':False,'human_evaluation':False,'external_delivery':False}
        write(output/'summary.json',summary)
        write(output/'final-state.json',state)
        content = '\n'.join('<article><pre>'+html.escape(json.dumps({
            'operation':e['operation'],'result':e['result']},ensure_ascii=False,indent=2))+'</pre></article>' for e in events)
        (output/'replay.html').write_text('<!doctype html><meta charset="utf-8"><title>Aria 冒烟回放</title>'
            '<style>body{max-width:900px;margin:30px auto;font:16px sans-serif}pre{white-space:pre-wrap}'
            'article{border-bottom:1px solid #ddd;padding:16px}</style><h1>虚构多会话冒烟回放</h1>'
            '<p>全部步骤保留；软件替身不代表真实模型效果。</p>'+content,encoding='utf8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture',action='store_true'); mode.add_argument('--live',action='store_true')
    parser.add_argument('--story',type=Path,required=True); parser.add_argument('--order',type=Path)
    parser.add_argument('--repeat-index',type=int,default=0);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--online-models',type=Path);parser.add_argument('--online-endpoints',type=Path)
    parser.add_argument('--price-policy',type=Path);parser.add_argument('--budget',default='0.45')
    parser.add_argument('--prior-occupied',default='5.924103542');parser.add_argument('--prior-new-spend',default='0')
    parser.add_argument('--unknown-429-prior',type=Path,
                        help='Explicit retained-ledger binding for a fresh bounded recovery allowance')
    args=parser.parse_args(argv)
    repeat=select_story(read(args.story),args.repeat_index,read(args.order) if args.order else None)
    args.output.mkdir(parents=True,exist_ok=False)
    from aria.voice.minimal import MinimalWebApp
    guard=None
    recovery=None
    sources={}
    if args.fixture:
        from smoke_fixture import fixture_clients
        clients=fixture_clients(repeat)
    else:
        from aria.engine.policy.online_structured import build_online_structured_route
        from aria.release import create_candidate_clients
        from aria.engine.policy.product_calls import ProductCallGuard
        from scripts.run_ctest import local_key
        budget=Decimal(args.budget)
        if (not budget.is_finite() or budget<=0
                or budget+Decimal(args.prior_new_spend)>Decimal('1.5')
                or budget+Decimal(args.prior_occupied)>Decimal('8')):
            raise ValueError('authorized_budget_exceeded')
        route=build_online_structured_route(args.online_models.read_bytes(),args.online_endpoints.read_bytes(),
                                            model='qwen/qwen3.8-flash',provider='alibaba')
        clients=create_candidate_clients(model_calls_enabled=True,online_structured_route=route,
            api_key=local_key(ROOT/'.env'),base_url='https://openrouter.ai/api/v1')
        recovery_options={}
        if args.unknown_429_prior:
            paths=[*ROOT.joinpath('aria').rglob('*.py'),Path(__file__).resolve(),
                   args.story,args.online_models,args.online_endpoints,args.price_policy]
            if args.order: paths.append(args.order)
            recovery,checker,sources=recovery_binding(args.unknown_429_prior,paths,
                prior_occupied=args.prior_occupied,prior_new_spend=args.prior_new_spend,
                stop_path=args.output/'STOP')
            recovery_options={'unknown_429_recovery':recovery,'recovery_pre_send_check':checker}
        guard=ProductCallGuard(args.output/'calls.sqlite3',max_requests=90,max_observed_cost_usd=str(budget),
                               reservation_policy=read(args.price_policy),min_interval_seconds=120,
                               observe_transport=True,transport_diagnostics=True,**recovery_options)
    write(args.output/'run-config.json',{'mode':'fixture' if args.fixture else 'live',
        'story_sha256':digest(args.story.read_bytes()),'repeat_index':args.repeat_index,
        'order_sha256':digest(args.order.read_bytes()) if args.order else None,
        'candidate':'release_v2','no_automatic_rerun':True,'budget':args.budget if args.live else '0',
        'prior_occupied':args.prior_occupied if args.live else None,
        'prior_new_spend':args.prior_new_spend if args.live else None,
        'unknown_429_recovery':recovery,'recovery_source_manifest':sources,
        'models':{c.role:c.model for c in (clients.online,clients.memory,clients.reflection,clients.safety_review)
                  if c is not None and hasattr(c,'model')},
        'source_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes())
                         for p in sorted([*ROOT.joinpath('aria').rglob('*.py'),Path(__file__).resolve()])}})
    app=MinimalWebApp(args.output/'product',clients=clients,model_call_guard=guard)
    record_model_outputs(clients,args.output)
    try:
        result=run_story(app,repeat,args.output,fixture=args.fixture)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        app.close()
        if guard is not None: guard.close()


if __name__=='__main__':
    main()
