"""Settled offline content failures: preserve failed stages, never replay them."""
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
from hashlib import sha256
from pathlib import Path
import json,sqlite3,time
from threading import Event
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.online_structured import DEEPSEEK_MODEL
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from .validation_support import read,ref,check_ref,write_once,require,digest_json
from .reflection_wrapper import contract,reflection_content_projection,VERSION
from .reflection_sse import (structured_receipt_body_complete, structured_stream_marked,
    STRUCTURED_STREAM_VERSION)

FIELD='reflection_validation_contract_binding'
MEMORY_FIELD='memory_validation_contract_binding'
IDENTITY=('request_id','request_sha256','role','operation_id','session_id')

def failure_binding(row):
    return row[MEMORY_FIELD if row['role']=='memory' else FIELD]

def memory_contract(protocol):
    from ...memory.preference_sources import LEGACY,SOURCE_REFS
    from ...memory.memory_contract import FACT_PROTOCOLS,memory_response_schema
    require(protocol in (LEGACY,SOURCE_REFS,*FACT_PROTOCOLS),'unknown_memory_source_protocol')
    return {'version':'settled_memory_content_failure_v1','source_protocol':protocol,
        'schema_sha256':digest_json(memory_response_schema(source_protocol=protocol))}

def receipt_contract(row):
    return (memory_contract(failure_binding(row)['contract']['source_protocol'])
            if row['role']=='memory' else contract(version=_reflection_format(row)))


def _reflection_format(row):
    codec=failure_binding(row)['contract']['domain_codec']
    require(codec in {'reflection_json_schema_v3','reflection_json_schema_v4'},
            'unknown_reflection_validation_codec')
    stream_contract=row['reflection_provider_contract']
    require(stream_contract['local_codec']==codec,'reflection_validation_codec_changed')
    if codec=='reflection_json_schema_v4':
        require(row['model']==DEEPSEEK_MODEL and row['expected_provider']=='Together'
            and stream_contract['provider_schema_applied'] is True
            and stream_contract['response_format_present'] is True,
            'structured_reflection_contract_changed')
    return codec.removeprefix('reflection_')

def _validate_settled_usage(row):
    usage,reserve,identity=row['usage'],row['reservation'],row['receipt_identity']
    require(identity['model_status']==identity['provider_status']=='recognized'
        and identity['error_field_status'] in ('absent','null')
        and all(type(usage[k]) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
        and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
        and usage['prompt_tokens']<=reserve['input_token_plan_bound']
        and usage['completion_tokens']<=reserve['output_token_limit'],'verified_'+row['role']+'_identity_usage_required')
    cost=Decimal(str(usage['cost']))
    require(type(usage['cost']) in (int,float) and cost.is_finite() and 0<=cost<=Decimal(reserve['reserved_cost_usd']),
        'verified_'+row['role']+'_fee_required')

def _validate_memory_receipt_metadata(row):
    require((row['model'],row['expected_provider']) in {
            ('anthropic/claude-sonnet-5','Anthropic'),(DEEPSEEK_MODEL,'Together')}
        and row['returned_model'] in row['memory_receipt_contract'].get('response_models', [row['model']])
        and row['expected_provider']==row['returned_provider']
        and row['http_status']==200 and row['validation_error'] in {
            'invalid_or_truncated_response','invalid_response_message','memory_schema_invalid'}
        and row['failure_phase']=='receipt_validation' and row['failure_code']==row['validation_error']
        and not row.get('fee_verification_error') and not row.get('unverified_cost_usd')
        and row['raw_total_tokens_valid'] is True,'exact_settled_memory_rejection_required')
    _validate_settled_usage(row)
    body,shape,binding=row['body_read'],row['memory_receipt_contract'],failure_binding(row)
    from ...memory.memory_contract import FACT_PROTOCOLS
    if binding['contract']['source_protocol'] in FACT_PROTOCOLS:
        require(row['model']==DEEPSEEK_MODEL and row['expected_provider']=='Together'
                and structured_stream_marked(row),'indexed_memory_requires_selected_stream_route')
    complete = (structured_receipt_body_complete(row) if structured_stream_marked(row)
                else body['complete'] is True and body['capture_status']=='complete')
    require(complete and body['exception_kind'] is None
        and shape['request_sha256']==row['request_sha256'] and shape['plain_text'] is True
        and shape['finish_reason']==('length' if row['validation_error']=='invalid_or_truncated_response' else 'stop'),
        'complete_memory_body_required')
    require(binding['schema']=='aria_actual_memory_validation_binding_v1'
        and all(binding[k]==row[k] for k in IDENTITY) and binding['contract']==receipt_contract(row)
        and binding['bound_before_application_write'] is True,'memory_parser_binding_changed')

def validate_receipt_metadata(row,*,historical=False):
    role = row.get('role') if type(row) is dict else None
    memory = role == 'memory' and not historical
    reason = 'exact_settled_memory_rejection_required' if memory else 'exact_settled_reflection_rejection_required'
    require((memory or role == 'reflection') and row.get('kind') == 'review'
            and row.get('status') == 'failed', reason)
    # Validate persisted JSON at the boundary; a missing or malformed field
    # cannot establish settlement. Keep specific domain rejections intact.
    try:
        if memory:
            return _validate_memory_receipt_metadata(row)
        return _validate_reflection_receipt_metadata(row, historical=historical)
    except (KeyError, TypeError, AttributeError, InvalidOperation):
        raise ValueError(reason) from None

def _validate_reflection_receipt_metadata(row,*,historical):
    # Only the explicitly selected Together route joins the existing actual
    # transaction. Historical receipts retain the original Sonnet contract;
    # DeepSeek must pass the binding and rejected-body checks below as well.
    route = (row.get('model'), row.get('expected_provider'))
    allowed = (route == ('anthropic/claude-sonnet-5', 'Anthropic')
               or not historical and route == (DEEPSEEK_MODEL, 'Together'))
    require(allowed, 'exact_settled_reflection_rejection_required')
    application_receipt = (not historical and _reflection_format(row)=='json_schema_v4'
        and (row.get('reflection_provider_contract', {}).get('receipt_mode') == STRUCTURED_STREAM_VERSION
             or row.get('stream_read', {}).get('receipt_mode') == STRUCTURED_STREAM_VERSION
             or 'structured_stream_contract' in row or 'structured_stream_receipt' in row))
    errors = ({'reflection_schema_invalid', 'invalid_or_truncated_response', 'invalid_response_message'}
              if application_receipt else {'reflection_schema_invalid'})
    models = (row.get('structured_stream_receipt', {}).get('response_models', [])
              if application_receipt else [row['model']])
    require(type(models) is list and row['returned_model'] in models
        and row['expected_provider']==row['returned_provider']
        and row['http_status']==200 and row['validation_error'] in errors
        and row['failure_phase']=='receipt_validation' and row['failure_code']==row['validation_error']
        and not row.get('fee_verification_error') and not row.get('unverified_cost_usd')
        and row['raw_total_tokens_valid'] is True,'exact_settled_reflection_rejection_required')
    _validate_settled_usage(row)
    stream=row['stream_read'];body=row['body_read']
    if application_receipt:
        require(structured_receipt_body_complete(row)
            and stream['finish_reason']==('length' if row['validation_error']=='invalid_or_truncated_response' else 'stop'),
            'complete_structured_reflection_stream_required')
    else:
        require(stream['complete'] is True and stream['application_complete'] is True
            and stream['done_events']==1 and stream['usage_events']==1
            and all(stream[k] is True for k in ('done_observed','usage_observed','finish_observed'))
            and stream['partial_output_submitted'] is False and stream['deadline_extended'] is False
            and body['application_complete'] is True
            and (body['http_message_complete'] is True or body['current_chunk_terminated'] is True)
            and (stream['http_message_complete'] is True or stream['current_chunk_terminated'] is True)
            and body['exception_kind'] is None and body['capture_status']=='complete'
            and body['body_sha256']==stream['stream_sha256'],'complete_reflection_stream_required')
    if not historical:
        binding=row[FIELD]
        require(binding['schema']=='aria_actual_reflection_validation_binding_v1'
            and all(binding[k]==row[k] for k in IDENTITY) and binding['contract']==receipt_contract(row)
            and binding['bound_before_application_write'] is True,'reflection_parser_binding_changed')
        if row['validation_error']=='reflection_schema_invalid':
            require(row['reflection_wrapper']['version']==VERSION and row['reflection_wrapper']['status']=='rejected',
                    'actual_reflection_parser_rejection_required')

def erasure_settlement(row,ledger_id):
    validate_receipt_metadata(row)
    return {'ledger_id':ledger_id,**{key:row[key] for key in IDENTITY},
        'status':row['status'],'model':row['model'],'provider':row['returned_provider'],
        'usage':row['usage'],'reserved_cost_usd':row['reservation']['reserved_cost_usd'],
        'parser_contract':failure_binding(row)['contract'],'body_sha256':row['body_read']['body_sha256'],
        'public_content_sha256':(row['memory_receipt_contract'] if row['role']=='memory' else row['stream_read'])['public_content_sha256'],
        'validation_error':row['validation_error'],'complete_known_verified_before_erasure':True}

def validate_receipt(row,*,historical=False):
    validate_receipt_metadata(row,historical=historical)
    stream=row['memory_receipt_contract'] if row['role']=='memory' else row['stream_read']
    path=check_ref(row['public_first_output']);public=read(path)
    require(all(public[k]==row[k] for k in ('request_id','request_sha256','role'))
        and public['hidden_reasoning_retained'] is False and len(public['public_first_output'])==1,
        'exact_reflection_public_output_required')
    output=public['public_first_output'][0];raw=output['content']
    application_receipt = (row['role']=='reflection' and not historical
                           and _reflection_format(row)=='json_schema_v4'
                           and structured_receipt_body_complete(row))
    require(type(raw) is str and (row['role']=='memory' or application_receipt or raw.strip()) and output['function_calls']==[]
        and sha256(raw.encode()).hexdigest()==stream['public_content_sha256'],
        'complete_original_reflection_content_required')
    if row['role']=='memory':
        if row['validation_error']=='memory_schema_invalid':
            from ...memory.memory_contract import validate_memory_response
            try:validate_memory_response(raw,source_protocol=failure_binding(row)['contract']['source_protocol'])
            except (ValueError,TypeError,KeyError,IndexError,RecursionError):pass
            else:raise ValueError('memory_output_was_not_rejected')
        elif row['validation_error']=='invalid_response_message':
            require(not raw.strip(),'memory_message_rejection_changed')
    elif application_receipt and row['validation_error']=='invalid_or_truncated_response':
        require(row['stream_read']['finish_reason']=='length','reflection_finish_rejection_changed')
    elif application_receipt and row['validation_error']=='invalid_response_message':
        require(not raw.strip(),'reflection_message_rejection_changed')
    elif not historical:
        try:reflection_content_projection(raw, version=_reflection_format(row))
        except (ValueError,TypeError,KeyError,IndexError,RecursionError):pass
        else:raise ValueError('reflection_output_was_not_rejected')
    return raw

def business(profile):
    # Same ownership boundary as held_input and settled_validation: these two
    # fields belong to the active online checkpoint, not the offline result.
    return {k:v for k,v in profile.items() if k not in {'offline_sessions','pending_session','last_interaction_at'}}

def db_digest(path):
    path=Path(path)
    if not path.exists():return None
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        data={name:db.execute('SELECT * FROM "'+name.replace('"','""')+'"').fetchall() for name in tables}
    return sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()

def snapshot(app):
    pair=app._sessions.get(app._settled_owner)
    online=None
    if pair:
        engine,ctx=pair
        checkpoint=engine.persona.pending_session
        online={'session_id':ctx.session_id,'scene_id':ctx.scene_id,'turn_count':len(ctx.turns),
            'checkpoint_sha256':digest_json(checkpoint.model_dump(mode='json') if checkpoint else None),
            'last_interaction_at':engine.persona.last_interaction_at}
    return {'profile':app.store.load(app._settled_owner).model_dump(mode='json'),'run':deepcopy(app._run),
        'care':db_digest(app.run_dir/'care.sqlite3'),'sharing':db_digest(app.run_dir/'sharing.sqlite3'),
        'online_session':online,'waiting_is_live':app._waiting_submission is not None
            and app._waiting_submission is app._run.get('waiting_input')}

def _validate_online_activity(before,after,a,b,offline_session):
    changed=any(before['profile'][key]!=after['profile'][key]
                for key in ('pending_session','last_interaction_at'))
    changed=changed or any(a.get(key)!=b.get(key) for key in ('sessions','turns','waiting_input'))
    if not changed:return
    online=after.get('online_session');pending=after['profile']['pending_session']
    require(type(online) is dict and type(pending) is dict
        and pending['session_id']==online['session_id']!=offline_session
        and pending['scene_id']==online['scene_id']
        and pending['original_turn_count']==online['turn_count']
        and digest_json(pending)==online['checkpoint_sha256']
        and after['profile']['last_interaction_at']==online['last_interaction_at']
        and not pending['offline_finalize_pending'],'offline_online_checkpoint_changed')
    sessions=b['sessions'];current=[s for s in sessions if s['session_id']==online['session_id']]
    require(len(current)==1 and current[0]['status']=='active','offline_online_session_missing')
    if a['sessions']!=sessions:
        added=sessions[-1]
        require(sessions[:-1]==a['sessions'] and added==current[0]
            and set(added)=={'session_id','number','status','started_at'}
            and added['number']==len(sessions),'offline_online_session_history_changed')
        a['sessions']=deepcopy(sessions)
    if a.get('waiting_input')!=b.get('waiting_input'):
        waiting=b.get('waiting_input')
        require(type(waiting) is dict and waiting['session_id']==online['session_id']
            and waiting['session_number']==current[0]['number']
            and waiting['turn_index']==online['turn_count'] and waiting['run_id']==b['run_id']
            and waiting.get('request_sent') is False and type(waiting['text']) is str
            and 0<len(waiting['text'])<=500,'offline_waiting_input_changed')
        if waiting['status']=='waiting':
            require(after.get('waiting_is_live') is True and a['turns']==b['turns'],
                'offline_waiting_input_not_owned')
        else:
            require(waiting['status']=='cancelled' and after.get('waiting_is_live') is False
                and waiting.get('error_code') in {'waiting_delivery_failed','waiting_state_changed','waiting_input_rejected'}
                and b['turns']==a['turns']+[{**waiting,'status':'failed','response':'',
                    'source':'local_request_rejected','requires_review':False,
                    'error':'等待期间状态已改变或等待消息未送达，这句话未发送模型。'}],
                'offline_waiting_cancellation_changed')
            a['turns']=deepcopy(b['turns'])
        a['waiting_input']=deepcopy(waiting)

def validate_state_pair(before,after,request_id,session_id,stage='reflection'):
    require(stage in ('memory','reflection'),'unknown_failed_offline_stage')
    require(business(before['profile'])==business(after['profile']),'reflection_business_halfcommit')
    for key in ('care','sharing'):require(before[key]==after[key],'reflection_permission_halfcommit')
    a,b=deepcopy(before['run']),deepcopy(after['run'])
    _validate_online_activity(before,after,a,b,session_id)
    old=[r for r in a['reviews'] if r['request_id']==request_id]
    new=[r for r in b['reviews'] if r['request_id']==request_id]
    require(len(old)==len(new)==1 and old[0]['status']=='running' and new[0]['status']=='failed'
        and old[0]['session_id']==new[0]['session_id']==session_id and new[0].get('requires_review') is True,
        'honest_failed_reflection_review_required')
    a['reviews']=[r for r in a['reviews'] if r['request_id']!=request_id]
    b['reviews']=[r for r in b['reviews'] if r['request_id']!=request_id]
    # The same installed review can be owner-started or already authorized by
    # auto-review. Only its outcome event may change alongside the failed row;
    # settings, quota attribution, source, and every other event remain fixed.
    oldevents=[r for r in a['auto_review']['events'] if r['request_id']==request_id]
    newevents=[r for r in b['auto_review']['events'] if r['request_id']==request_id]
    if oldevents or newevents:
        require(len(oldevents)==len(newevents)==1
            and oldevents[0]['session_id']==session_id and oldevents[0]['status'] in ('dispatching','running')
            and {**oldevents[0],'status':'failed','reason':'review_not_completed'}==newevents[0],
            'offline_automatic_failure_event_changed')
        a['auto_review']['events']=[r for r in a['auto_review']['events'] if r['request_id']!=request_id]
        b['auto_review']['events']=[r for r in b['auto_review']['events'] if r['request_id']!=request_id]
    require(a==b,'reflection_owner_state_changed')
    oldjobs=before['profile']['offline_sessions'];newjobs=after['profile']['offline_sessions']
    require(len(oldjobs)==len(newjobs) and oldjobs[1:]==newjobs[1:],'reflection_queue_changed')
    prior,job=oldjobs[0],newjobs[0]
    require(prior['session_id']==job['session_id']==session_id and prior['memory_completed'] is (stage=='reflection')
        and prior['memory_privacy_review_completed'] is (stage=='reflection') and prior['reflection_completed'] is False
        and job['reflection_completed'] is False and job['state']=='pending' and job['active_stage']==stage
        and job['attempts']==prior['attempts'] and job['attempts']>0,'uncommitted_reflection_stage_required')
    allowed={'state','last_error_code',stage+'_result'}
    if stage=='memory':allowed.add('preference_scope')
    require(all(prior[k]==job[k] for k in prior if k not in allowed),'reflection_job_identity_changed')
    return new[0],job

def seal_failed_job(app,session_id,stage='reflection'):
    persona=app.store.load(app._settled_owner);candidate=persona.model_copy(deep=True)
    job=candidate.offline_sessions[0]
    require(stage in ('memory','reflection') and job.session_id==session_id
        and job.memory_completed is (stage=='reflection') and job.memory_privacy_review_completed is (stage=='reflection')
        and not job.reflection_completed and job.active_stage==stage,'exact_failed_reflection_job_required')
    original=job.model_dump(mode='json')
    queue=OfflineSessionQueue(persona,store=app.store)
    queue._finish(candidate,job,datetime.now(timezone.utc),outcome=stage+'_failed',error_code=job.last_error_code)
    app.store.save(candidate)
    stored=app.store.load(app._settled_owner)
    require(stored.model_dump(mode='json')==candidate.model_dump(mode='json'),'reflection_terminal_not_durable')
    audit=stored.offline_session_audit[-1].model_dump(mode='json')
    require(audit['outcome']==stage+'_failed' and audit['memory_completed'] is (stage=='reflection')
        and audit['reflection_completed'] is False and audit['created_at']==original['created_at']
        and audit['expires_at']==original['expires_at'] and audit['attempts']==original['attempts'],
        'reflection_terminal_claim_changed')
    pair=app._sessions.get(app._settled_owner)
    if pair:pair[0]._accept_offline_profile(stored)
    return original,stored.model_dump(mode='json'),audit

def validate_terminal_transition(failed_profile,original_job,terminal,audit,stage='reflection'):
    require(failed_profile['offline_sessions'] and failed_profile['offline_sessions'][0]==original_job,
        'original_reflection_job_changed')
    candidate=PersonaProfile.model_validate(deepcopy(failed_profile))
    job=candidate.offline_sessions[0]
    require(stage in ('memory','reflection') and job.memory_completed is (stage=='reflection')
        and job.memory_privacy_review_completed is (stage=='reflection') and not job.reflection_completed
        and job.active_stage==stage and audit['outcome']==stage+'_failed',
        'exact_reflection_failure_transition_required')
    when=datetime.fromisoformat(audit['finished_at'])
    OfflineSessionQueue(candidate)._finish(candidate,job,when,outcome=stage+'_failed',error_code=job.last_error_code)
    require(candidate.model_dump(mode='json')==terminal and candidate.offline_session_audit[-1].model_dump(mode='json')==audit,
        'reflection_terminal_transition_changed')

class SettledOfflineGuardMixin:
    _reflection_transaction=None
    _reflection_resolution_readers=None
    def _erased_reflection_ids(self):
        if not self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='erased_reflection_failures'").fetchone():return set()
        result=set()
        for request_id,raw in self._db.execute('SELECT request_id,record FROM erased_reflection_failures'):
            v=json.loads(raw);row=self._reflection_ledger_row(request_id)
            require(v['schema']=='aria_erased_settled_reflection_v2' and v['request_id']==request_id
                and v['ledger_id']==self.ledger_id and v['ledger_row_sha256']==digest_json(row)
                and v['contract']==receipt_contract(row) and v['state']=='forgotten'
                and v['failed_status_retained'] is True and v['retry_allowed'] is False
                and v['settlement']==erasure_settlement(row,self.ledger_id),
                'unresolved_reflection_forget_lifecycle')
            require(v['content_scope']==self.content_erasure.settled_scope(row),'erased_content_scope_changed')
            self.content_erasure.require_erased(v['content_scope'])
            result.add(request_id)
        return result
    def begin_reflection_forget(self,receipts):
        with self._lock:
            self._db.execute('CREATE TABLE IF NOT EXISTS erased_reflection_failures(request_id TEXT PRIMARY KEY,record TEXT NOT NULL)')
            values=[]
            for receipt,path in receipts:
                row=receipt['ledger_row'];validate_receipt(row)
                require(self._reflection_ledger_row(row['request_id'])==row,'forget_actual_receipt_changed')
                value={'schema':'aria_erased_settled_reflection_v2','request_id':row['request_id'],
                    'ledger_id':self.ledger_id,'ledger_row_sha256':digest_json(row),'contract':receipt_contract(row),
                    'disposition_sha256':ref(path)['sha256'],'terminal_audit_sha256':digest_json(receipt['audit']),
                    'state':'pending','failed_status_retained':True,'retry_allowed':False,
                    'settlement':erasure_settlement(row,self.ledger_id),
                    'content_scope':self.content_erasure.settled_scope(row)}
                self._db.execute('INSERT INTO erased_reflection_failures VALUES(?,?)',
                    (row['request_id'],json.dumps(value,sort_keys=True,allow_nan=False)))
                values.append(value)
            self._db.commit()
            for value in values:
                stored=self._db.execute('SELECT record FROM erased_reflection_failures WHERE request_id=?',(value['request_id'],)).fetchone()
                require(stored and json.loads(stored[0])==value,'forget_marker_not_durable')
            return values
    def complete_reflection_forget(self,values):
        with self._lock:
            for value in values:
                stored=self._db.execute('SELECT record FROM erased_reflection_failures WHERE request_id=?',(value['request_id'],)).fetchone()
                require(stored and json.loads(stored[0])==value,'forget_marker_changed')
                finished={**value,'state':'forgotten'}
                self._db.execute('UPDATE erased_reflection_failures SET record=? WHERE request_id=?',
                    (json.dumps(finished,sort_keys=True,allow_nan=False),value['request_id']))
            self._db.commit()
            require({value['request_id'] for value in values}<=self._erased_reflection_ids(),'forget_not_durable')
    def _reflection_ledger_row(self,request_id,*,read_only=False):
        query='SELECT record FROM attempts WHERE request_id=?'
        rows=(self._read_committed_rows(query,(request_id,)) if read_only else
              self._db.execute(query,(request_id,)).fetchall())
        require(len(rows)==1,'unique_reflection_ledger_identity_required')
        return json.loads(rows[0][0])
    def _snapshot(self):
        state=super()._snapshot()
        erased=self._erased_reflection_ids()
        if state.get('stop_reason')!='request_failed':return state
        allowed=set(erased)
        for reader in self._reflection_resolution_readers or []:allowed|=reader()
        allowed|=self._settled_validation_physical_ids(state)
        allowed|={r['request_id'] for r in state['attempts']
                  if r.get('confirm_judge_disposition')=='unclear_known_cost_format_failure_v1'
                  and self._settled_confirm_format_failure(r)}
        allowed|={r['request_id'] for r in state['attempts']
                  if r.get('safety_review_disposition')=='withheld_known_cost_format_failure_v1'
                  and self._settled_safety_review_format_failure(r)}
        transaction=self._reflection_transaction
        if transaction is not None and not transaction['extra']:
            for row in state['attempts']:
                if (row['request_id'] not in transaction['prior_ids'] and row['role'] in ('memory','reflection')
                    and row['operation_id']==transaction['operation_id'] and row['session_id']==transaction['session_id']):
                    try:validate_receipt(row);check_ref(failure_binding(row)['before_state'])
                    except (ValueError,KeyError,TypeError,OSError):continue
                    allowed.add(row['request_id'])
        failed={r['request_id'] for r in state['attempts'] if r['status']=='failed'}
        if failed and failed<=allowed and state['unknown_cost_requests']==0:
            state.update(stopped=False,stop_reason=None)
            if state['calls_used']>=self.max_requests:
                state.update(stopped=True,stop_reason='request_limit_reached')
            elif Decimal(state['known_cost_usd'] or '0')>=Decimal(self.threshold):
                state.update(stopped=True,stop_reason='observed_cost_threshold_reached')
        return state
    def begin_reflection_review(self,request_id,session_id,before):
        require(self._reflection_transaction is None,'reflection_transaction_reentry')
        self._reflection_transaction={'operation_id':getattr(self,'namespace','')+'__'+request_id if getattr(self,'namespace','') else request_id,
            'session_id':session_id,'prior_ids':{r['request_id'] for r in self.snapshot()['attempts']},
            'before':before,'candidate':None,'clients':{},'extra':False,'settled':Event()}
    def _begin(self,client,request,**kwargs):
        row=super()._begin(client,request,**kwargs)
        transaction=self._reflection_transaction
        if client.role not in ('memory','reflection') or transaction is None:return row
        try:
            require(transaction['clients'].get(client.role) is client and transaction['candidate'] is None,'reflection_request_changed')
            proof=transaction['before'](row)
            from ..memory_structured import memory_schema_protocol
            selected=(memory_contract(memory_schema_protocol(client)) if client.role=='memory'
                      else contract(version=client.reflection_return_format))
            row[MEMORY_FIELD if client.role=='memory' else FIELD]={
                'schema':'aria_actual_'+client.role+'_validation_binding_v1',**{k:row[k] for k in IDENTITY},
                'contract':selected,'before_state':proof,'bound_before_application_write':True}
            ReservedCallGuard._finish(self,row)
            require([r for r in self.snapshot()['attempts'] if r['request_id']==row['request_id']]==[row],
                    'reflection_binding_not_durable')
            return row
        except BaseException:
            self.force_halt_empty_failure();raise
    def request_json(self,client,request,**kwargs):
        t=self._reflection_transaction
        if t is not None and t['candidate'] is not None:
            t['extra']=True;self.force_halt_empty_failure();raise MinimalCallStopped('no_reflection_replay')
        if t is not None and client.role in ('memory','reflection'):
            require(client.role not in t['clients'],'one_offline_request_per_stage_required');t['clients'][client.role]=client
        return super().request_json(client,request,**kwargs)
    def halt(self,reason='model_or_storage_failed'):
        t=self._reflection_transaction
        if t is not None and reason=='model_or_storage_failed' and not t['extra']:
            try:
                state=self.snapshot();rows=[r for r in state['attempts'] if r['request_id'] not in t['prior_ids']
                    and r['operation_id']==t['operation_id'] and r['session_id']==t['session_id']]
                row=rows[-1]
                require(not state['stopped'] and row['operation_id']==t['operation_id']
                    and row['session_id']==t['session_id'] and row['role'] in t['clients'],
                    'current_reflection_transaction_required')
                # Memory may have completed earlier in this same review. Every
                # preceding physical must be received/known, never another F.
                require(all(r['status']=='received' and 'cost' in r['usage'] and r['role']=='memory'
                            for r in rows[:-1]),'unresolved_preceding_stage')
                validate_receipt(row);check_ref(failure_binding(row)['before_state'])
                t['candidate']=deepcopy(row);return
            except (ValueError,KeyError,TypeError,OSError,IndexError):pass
        return super().halt(reason)
    def end_reflection_review(self,resolved):
        t=self._reflection_transaction;self._reflection_transaction=None
        try:
            if t and t['candidate'] is not None and (not resolved or t['extra']):self.force_halt_empty_failure()
        finally:
            if t:t['settled'].set()

    def wait_for_offline_failure_settlement(self):
        # Called only after acquiring the next physical slot, without owner or
        # ledger locks. Successful memory must still yield to an online turn;
        # only a rejected paid body needs its short local terminal commit first.
        t=self._reflection_transaction
        if t is None or t['candidate'] is None:return
        deadline=time.monotonic()+90
        while not t['settled'].wait(.05):
            if self._stop_requested.is_set() or self._closed or self._broken:return
            if time.monotonic()>=deadline:
                self._stop_requested.set()
                return

# Historical imports remain valid; both stages share the same transaction.
SettledReflectionGuardMixin=SettledOfflineGuardMixin

