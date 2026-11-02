"""Keep an undeliverable known-fee reply failed while permitting the next input."""
from copy import deepcopy
from pathlib import Path
from ..engine.policy.max_release.validation_support import (
    read,ref,check_ref,write_once,require,digest_json)
from ..engine.policy.max_release.validation_contract import validate_rejected_receipt
from ..engine.policy.max_release.request_requirements import validate_binding
from ..engine.policy.max_release.contract_state import (
    FIELD,OUTCOME,IDENTITY,sources,snapshot,business,summary,validate_states,row_without_outcome)
from hashlib import sha256

class SettledValidationAppMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if hasattr(self.call_guard,'_validation_resolution_readers'):
            self.call_guard._validation_resolution_readers.append(self._settled_validation_physical_ids)

    def _settled_validation_physical_ids(self,ledger):
        turns=self._settled_validation_ids(ledger=ledger)
        return {read(p)['request_id'] for p in self._settled_validation_directory().glob('*.json')
                if read(p)['turn_id'] in turns}

    def _settled_validation_directory(self):return self.run_dir/'settled-validation-failures'

    def _settled_validation_ids(self,*,ledger=None):
        directory=self._settled_validation_directory()
        if not directory.exists():return set()
        require(hasattr(self.call_guard,'begin_empty_turn'),'settled_validation_guard_required')
        result=set()
        # Historical identity checks must not wait for the transport lock while
        # state() holds the owner lock. These are freshly committed receipts,
        # not a cached set of allowed failures or a fee/admission snapshot.
        if ledger is None:
            ledger={'attempts':self.call_guard.read_committed_attempts()}
        for path in sorted(directory.glob('*.json')):
            value=read(path)
            require(value.get('schema') in {'aria_local_settled_validation_failure_v1',
                    'aria_local_settled_validation_failure_v2'}
                and value.get('run_id')==self._run['run_id']
                and value.get('original_status')=='failed' and value.get('replacement_response') is None
                and value.get('continue_next_input') is True and value.get('effect_success') is False,
                'invalid_settled_validation_metadata')
            rows=[r for r in self._run['turns'] if r['turn_id']==value['turn_id']]
            require(len(rows)==1 and digest_json(rows[0])==value['failed_turn_sha256']
                and rows[0]['status']=='failed' and rows[0].get('requires_review') is True
                and rows[0]['response']=='' and rows[0]['source']=='model_failed',
                'original_failed_turn_changed')
            receipts=[r for r in ledger['attempts'] if r['request_id']==value['request_id']]
            is_bound=value['schema']=='aria_local_settled_validation_failure_v2'
            require(len(receipts)==1 and digest_json(row_without_outcome(receipts[0]) if is_bound else receipts[0])==value['ledger_row_sha256']
                and receipts[0]['public_first_output']==value['first_output'],
                'original_settled_ledger_changed')
            row=receipts[0]
            if is_bound:
                self._held_require_persisted_run()
                binding=row[FIELD];state=binding['state'];outcome=row[OUTCOME]
                require(binding=={'schema':'aria_local_contract_preapplication_v1',
                    **{k:row[k] for k in IDENTITY},'requirements':value['requirements'],
                    'sources':sources(),'state':value['preapplication'],
                    'bound_before_application_write':True},'contract_preapplication_binding_changed')
                for source in binding['sources']:check_ref(source)
                require(state['owner_id']==self._settled_owner
                    and self.store.load(self._settled_owner).user_id==self._settled_owner
                    and state['run_id']==self._run['run_id'] and state['turn_id']==value['turn_id']
                    and state['session_id']==row['session_id']==rows[0]['session_id']
                    and state['turn_index']==rows[0]['turn_index'], 'contract_current_owner_or_failure_changed')
                require(outcome=={'schema':'aria_local_contract_verified_outcome_v1',
                    'request_id':row['request_id'],'disposition_sha256':digest_json(value),
                    'original_receipt_sha256':value['ledger_row_sha256'],'model_reply_delivered':False},
                    'contract_verified_outcome_changed')
                require(value['business_before_sha256']==state['business_sha256']
                    and digest_json(rows[0].get('protection_prelude'))==state['prelude_sha256']
                    and value['protection_prelude_delivered']==state['protection_prelude_delivered'],
                    'contract_business_or_prelude_changed')
            else:
                require(not rows[0].get('protection_prelude') and not rows[0].get('premodel_task'),
                    'legacy_disposition_cannot_grant_protected_failure')
            validate_binding(row,value['requirements'])
            validate_rejected_receipt(row,read(check_ref(value['first_output'])),
                requirements=value['requirements'],request_id=row['request_id'],
                operation_id=row['operation_id'],session_id=rows[0]['session_id'])
            require(value['business_before_sha256']==value['business_after_sha256'],
                'settled_validation_business_mismatch')
            result.add(value['turn_id'])
        return result

    def _safe_settled_validation_ids(self):
        try:return self._settled_validation_ids()
        except Exception:return set()

    def _submit_with_settled_validation(self,text,turn_id,session_id,**kwargs):
        if not hasattr(self.call_guard,'seal_contract_disposition'):
            return self._submit_legacy_settled_validation(text,turn_id,session_id,**kwargs)
        with self._user_lock(self._settled_owner):
            guard=self.call_guard
            if any(r['turn_id']==turn_id for r in self._run['turns']):
                return self._submit_original(text,turn_id,session_id,**kwargs)
            view=self.state()
            if view['requires_review'] or guard.snapshot()['stopped'] or view['session_id']!=session_id:
                return self._submit_original(text,turn_id,session_id,**kwargs)
            self._held_require_persisted_run()
            origin=snapshot(self);captured={}
            def capture(row):
                self._held_require_persisted_run();pair=self._sessions.get(self._settled_owner)
                require(pair is not None and pair[1].session_id==session_id,'contract_live_session_required')
                engine,ctx=pair;pending=self._run['turns'][-1]
                pre={**snapshot(self),'engine_profile':engine.persona.model_dump(mode='json'),
                    'turn_id':turn_id,'turn_index':len(ctx.turns),'utterance_sha256':sha256(text.encode()).hexdigest(),
                    'prelude':deepcopy(pending.get('protection_prelude')),
                    'premodel_task':deepcopy(ctx.working_memory.get('premodel_task')),
                    'risk_flag':deepcopy(ctx.working_memory.get('risk_flag'))}
                state=summary(origin,pre,row,self._settled_owner)
                captured['pre']=pre
                return state
            guard._contract_capture=capture
            guard.begin_empty_turn(turn_id,session_id);candidate=None;resolved=False
            try:
                result=self._submit_original(text,turn_id,session_id,**kwargs)
                candidate=guard.empty_turn_candidate()
                if candidate is None:return result
                self._held_require_persisted_run();after=snapshot(self);failed=self._run['turns'][-1]
                require(not result.get('ok') and not self._storage_failed,'contract_failed_model_result_required')
                pre=captured['pre'];validate_states(origin,pre,after,candidate,failed,self._settled_owner)
                require(candidate[FIELD]['state']==summary(origin,pre,candidate,self._settled_owner),
                    'contract_actual_preapplication_changed')
                after_view=self.state()
                require(after_view['preference_invitation']==view['preference_invitation']
                    or view['preference_invitation'] is not None and view['preference_invitation']['status']=='delivered'
                    and after_view['preference_invitation'] is None,'contract_invitation_changed')
                ctx=self._sessions[self._settled_owner][1];index=failed['turn_index']
                require(index==len(ctx.turns)-1 and index in ctx.working_memory.get('_sensitive_memory_excluded_turns',[])
                    and ctx.working_memory.get('online_turn_decision',{}).get('status')=='model_failure',
                    'contract_failed_slot_not_excluded')
                value={'schema':'aria_local_settled_validation_failure_v2','run_id':self._run['run_id'],
                    'turn_id':turn_id,'request_id':candidate['request_id'],'failed_turn_sha256':digest_json(failed),
                    'ledger_row_sha256':digest_json(candidate),'first_output':candidate['public_first_output'],
                    'requirements':guard.rejected_requirements(),'preapplication':candidate[FIELD]['state'],
                    'business_before_sha256':digest_json(business(pre['engine_profile'])),
                    'business_after_sha256':digest_json(business(after['profile'])),
                    'protection_prelude_delivered':pre['prelude'] is not None,
                    'original_status':'failed','replacement_response':None,'retry_same_input':False,
                    'excluded_original_index':index,'continue_next_input':True,'effect_success':False}
                directory=self._settled_validation_directory();directory.mkdir(exist_ok=True)
                write_once(directory/(digest_json(turn_id)+'.json'),value)
                guard.seal_contract_disposition(candidate,value)
                resolved=turn_id in self._settled_validation_ids();require(resolved,'contract_disposition_not_durable')
                from ..engine.context import Turn
                ctx.turns[index]=Turn(user='',agent='')
                return {**result,'state':self.state(),'settled_validation_failure':True,'request_sent':True,
                    'response_delivered':False,'model_response_delivered':False,
                    'protection_prelude_delivered':pre['prelude'] is not None,'continue_next_input':True}
            except BaseException:
                guard.force_halt_empty_failure()
                raise
            finally:
                guard._contract_capture=None
                guard.end_empty_turn(resolved=resolved)

    def _submit_legacy_settled_validation(self,text,turn_id,session_id,**kwargs):
        with self._user_lock(self._settled_owner):
            guard=self.call_guard
            if not hasattr(guard,'begin_empty_turn') or any(r['turn_id']==turn_id for r in self._run['turns']):
                return self._submit_original(text,turn_id,session_id,**kwargs)
            before=self.state()
            if before['requires_review'] or guard.snapshot()['stopped'] or before['session_id']!=session_id:
                return self._submit_original(text,turn_id,session_id,**kwargs)
            care_before=self.care.elder_view(self._settled_owner)
            profile_before=self.store.load(self._settled_owner).model_dump(mode='json')
            guard.begin_empty_turn(turn_id,session_id);candidate=None
            try:
                result=self._submit_original(text,turn_id,session_id,**kwargs)
                candidate=guard.empty_turn_candidate()
                if candidate is None:return result
                after=self.state();failed=self._run['turns'][-1]
                require(not result.get('ok') and not self._storage_failed and failed['turn_id']==turn_id
                    and failed['session_id']==session_id and failed['status']=='failed' and failed['response']==''
                    and failed['source']=='model_failed' and failed.get('requires_review') is True
                    and before['turns']==after['turns'][:-1]
                    and not failed.get('protection_prelude') and not failed.get('premodel_task'),
                    'exact_current_failed_turn_required')
                for key in ('profile','cognitive','reviews','preference_reviews','auto_review','sessions'):
                    require(before[key]==after[key],'settled_validation_business_state_changed')
                require(self.care.elder_view(self._settled_owner)==care_before,'settled_validation_care_changed')
                require(after['preference_invitation']==before['preference_invitation']
                    or before['preference_invitation'] is not None and before['preference_invitation']['status']=='delivered'
                    and after['preference_invitation'] is None,'unexpected_invitation_change')
                sb,sa=deepcopy(before['sharing']),deepcopy(after['sharing'])
                sb.pop('writable',None);sa.pop('writable',None)
                require(sb==sa,'settled_validation_sharing_changed')
                profile_after=self.store.load(self._settled_owner).model_dump(mode='json')
                business_before={k:v for k,v in profile_before.items() if k not in {'pending_session','last_interaction_at'}}
                business_after={k:v for k,v in profile_after.items() if k not in {'pending_session','last_interaction_at'}}
                require(business_before==business_after,'settled_validation_durable_side_effect')
                pair=self._sessions.get(self._settled_owner)
                require(pair is not None and pair[1].session_id==session_id,'settled_validation_context_missing')
                ctx=pair[1];index=failed['turn_index']
                require(len(ctx.turns)==index+1 and index in ctx.working_memory.get('_sensitive_memory_excluded_turns',[])
                    and ctx.working_memory.get('online_turn_decision',{}).get('status')=='model_failure',
                    'settled_validation_turn_not_excluded')
                from ..engine.context import Turn
                ctx.turns[index]=Turn(user='',agent='')
                value={'schema':'aria_local_settled_validation_failure_v1','run_id':self._run['run_id'],
                    'turn_id':turn_id,'request_id':candidate['request_id'],'failed_turn_sha256':digest_json(failed),
                    'ledger_row_sha256':digest_json(candidate),'first_output':candidate['public_first_output'],
                    'requirements':guard.rejected_requirements(),
                    'business_before_sha256':digest_json(business_before),'business_after_sha256':digest_json(business_after),
                    'original_status':'failed','replacement_response':None,'retry_same_input':False,
                    'excluded_original_index':index,'continue_next_input':True,'effect_success':False}
                directory=self._settled_validation_directory();directory.mkdir(exist_ok=True)
                write_once(directory/(digest_json(turn_id)+'.json'),value)
                require(turn_id in self._settled_validation_ids(),'settled_validation_not_durable')
                return {**result,'state':self.state(),'settled_validation_failure':True,'request_sent':True,
                    'response_delivered':False,'continue_next_input':True}
            except BaseException:
                guard.force_halt_empty_failure()
                raise
            finally:
                resolved=False
                try:resolved=bool(candidate and turn_id in self._settled_validation_ids())
                finally:guard.end_empty_turn(resolved=resolved)

    def _forget_settled_validation_metadata(self):
        directory=self._settled_validation_directory()
        if directory.exists():
            for path in directory.glob('*.json'):path.unlink()
