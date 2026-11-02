"""Pure state boundaries for a complete rejected reply; no response repair."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from .validation_support import require, ref, digest_json
from .reflection_failure import snapshot
from . import validation_contract
from . import online_shape

FIELD = 'online_contract_isolation_binding'
OUTCOME = 'online_contract_isolation_outcome'
IDENTITY = ('request_id', 'request_sha256', 'role', 'operation_id', 'session_id')

def sources():
    return [ref(Path(__file__)), ref(Path(validation_contract.__file__)), ref(Path(online_shape.__file__))]

def summary(origin, pre, row, owner):
    validate_preapplication(origin, pre, row, owner)
    return {'owner_id': owner, 'run_id': pre['run']['run_id'], 'turn_id': pre['turn_id'],
        'session_id': row['session_id'], 'turn_index': pre['turn_index'],
        'business_sha256': digest_json(business(pre['engine_profile'])),
        'permissions_sha256': digest_json({k: pre[k] for k in ('care', 'sharing')}),
        'prior_turns_sha256': digest_json(origin['run']['turns']),
        'run_authority_sha256': digest_json({k: v for k, v in origin['run'].items() if k != 'turns'}),
        'prelude_sha256': digest_json(pre['prelude']),
        'premodel_task_sha256': digest_json(pre['premodel_task']),
        'protection_prelude_delivered': pre['prelude'] is not None}

def row_without_outcome(row):
    return {k: v for k, v in row.items() if k != OUTCOME}

def business(profile):
    return {k:v for k,v in profile.items() if k not in {'pending_session','last_interaction_at'}}

def validate_preapplication(origin,pre,row,owner):
    """Only the existing, source-bound keyword screening audit may predate POST."""
    require(origin['profile']['user_id']==pre['profile']['user_id']==owner,'contract_owner_changed')
    require(origin['run']['run_id']==pre['run']['run_id'],'contract_run_changed')
    for k in ('care','sharing'):
        require(origin[k]==pre[k],'contract_preapplication_permission_changed')
    require(business(origin['profile'])==business(pre['profile']),'contract_preapplication_store_changed')
    before=deepcopy(origin['profile']);engine=deepcopy(pre['engine_profile'])
    prelude=pre['prelude'];audit=pre['premodel_task']
    if prelude is None:
        require(audit is None and business(before)==business(engine),'contract_unprotected_preapplication_changed')
    else:
        require(prelude['status']=='transport_returned' and prelude['model_generated'] is False
            and prelude['user_id']==owner and prelude['session_id']==row['session_id']
            and prelude['turn_index']==pre['turn_index']
            and sha256(prelude['protection'].encode()).hexdigest()==prelude['protection_sha256']
            and prelude['utterance_sha256']==pre['utterance_sha256'], 'contract_delivered_prelude_identity_required')
        require(audit['status']=='prelude_emitted' and audit['client_entered'] is False
            and audit['semantic_verified'] is False and audit['grants_permission'] is False
            and audit['version']==prelude['premodel_safety_version']
            and all(audit[k]==prelude[k] for k in ('session_id','turn_index','protection_sha256','utterance_sha256')),
            'contract_actual_response_only_route_required')
        old=before['safety_screenings'];new=engine['safety_screenings']
        require(new[:len(old)]==old and len(new) in (len(old),len(old)+1),'contract_screening_prefix_changed')
        if len(new)>len(old):
            screening=new[-1];risk=pre['risk_flag']
            require(screening['source']=='keyword_backstop' and screening['subject']=='unverified'
                and screening['session_id']==row['session_id'] and screening['turn_index']==pre['turn_index']
                and screening['evidence_sha256']==pre['utterance_sha256']
                and screening['tier']==risk['tier'] and screening['kind']==risk['kind'],
                'contract_exact_existing_screening_required')
        before['safety_screenings']=new
        require(business(before)==business(engine),'contract_ordinary_preapplication_business_changed')
    pending=pre['run']['turns'][-1]
    require(pending['turn_id']==pre['turn_id'] and pending['session_id']==row['session_id']
        and pending['status']=='pending' and pending.get('protection_prelude')==prelude
        and origin['run']['turns']==pre['run']['turns'][:-1], 'contract_exact_pending_turn_required')
    a,b=deepcopy(origin['run']),deepcopy(pre['run']);a.pop('turns');b.pop('turns')
    require(a==b,'contract_preapplication_run_side_effect')

def validate_states(origin,pre,after,row,failed,owner):
    validate_preapplication(origin,pre,row,owner)
    require(failed['status']=='failed' and failed['response']=='' and failed['source']=='model_failed'
        and failed['requires_review'] is True and failed['turn_id']==pre['turn_id']
        and failed['session_id']==row['session_id'] and failed['turn_index']==pre['turn_index'],
        'contract_actual_failed_model_turn_required')
    require(failed.get('protection_prelude')==pre['prelude'],'contract_delivered_prelude_changed')
    if pre['premodel_task'] is None:
        require(not failed.get('premodel_task'),'contract_unexpected_task_route')
    else:
        expected={**pre['premodel_task'],'status':'response_only_failed','client_entered':True}
        require(failed.get('premodel_task')==expected,'contract_response_only_route_changed')
    require(business(pre['engine_profile'])==business(after['profile']),'contract_model_business_halfcommit')
    require(after['care']==pre['care'] and after['sharing']==pre['sharing'],'contract_permission_halfcommit')
    require(after['run']['turns']==origin['run']['turns']+[failed],'contract_prior_turn_changed')
    a,b=deepcopy(origin['run']),deepcopy(after['run']);a.pop('turns');b.pop('turns')
    require(a==b,'contract_run_halfcommit')

