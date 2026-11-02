"""Same immutable transport, no paid replay; scoped pure rejection disposition."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped
from .validation_support import require,check_ref,read,digest_json
from .validation_contract import validate_rejected_receipt
from .request_requirements import FIELD,make_binding
from .contract_state import FIELD as CONTRACT_FIELD, OUTCOME, IDENTITY, sources, row_without_outcome
class SettledValidationGuardMixin:
    _empty_transaction=None
    _contract_capture=None
    def _persist_requirements_binding(self,row):
        require(row['status']=='pending','pending_annotation_only')
        # Use the base ledger write, not the finite terminal-fee finalizer.
        # This row has not been dispatched and has no provider receipt yet.
        ReservedCallGuard._finish(self,row)
    def _begin(self,client,request,**kwargs):
        row=super()._begin(client,request,**kwargs)
        if client.role!='online_fast' or self._empty_transaction is None:return row
        try:
            require(row['status']=='pending' and FIELD not in row,'new_pending_request_binding_required')
            row[FIELD]=make_binding(row,self.rejected_requirements())
            if self._contract_capture is not None:
                require(callable(self._contract_capture) and CONTRACT_FIELD not in row,
                    'actual_contract_capture_required')
                state=self._contract_capture(row)
                row[CONTRACT_FIELD]={'schema':'aria_local_contract_preapplication_v1',
                    **{k:row[k] for k in IDENTITY},'requirements':self.rejected_requirements(),
                    'sources':sources(),'state':state,'bound_before_application_write':True}
            # This commits the actual request options in its own physical row,
            # before _request_once reaches application-write preparation.
            self._persist_requirements_binding(row)
            stored=[r for r in self.snapshot()['attempts'] if r['request_id']==row['request_id']]
            require(stored==[row],'actual_projection_binding_not_durable')
            return row
        except BaseException:
            self.force_halt_empty_failure()
            raise

    def seal_contract_disposition(self,row,value):
        """Bind the verified, content-free disposition to its existing charge."""
        current=self._reflection_ledger_row(row['request_id'])
        require(current==row and OUTCOME not in row and row['status'] in {'received','failed'}
            and value['ledger_row_sha256']==digest_json(row), 'contract_outcome_receipt_changed')
        sealed=deepcopy(row)
        sealed[OUTCOME]={'schema':'aria_local_contract_verified_outcome_v1',
            'request_id':row['request_id'],'disposition_sha256':digest_json(value),
            'original_receipt_sha256':digest_json(row),'model_reply_delivered':False}
        ReservedCallGuard._finish(self,sealed)
        require(self._reflection_ledger_row(row['request_id'])==sealed,'contract_outcome_not_durable')
        return sealed

    def begin_empty_turn(self,turn_id,session_id):
        require(self._empty_transaction is None,'empty_turn_transaction_active')
        state=self.snapshot()
        require(not state['stopped'],'new_guard_stopped')
        namespace=getattr(self,'namespace','')
        operation_id=namespace+'__'+turn_id if namespace else turn_id
        self._empty_transaction={'operation_id':operation_id,'session_id':session_id,
            'prior_ids':{r['request_id'] for r in state['attempts']},'candidate':None,'attempted_extra':False}
    def empty_turn_candidate(self):
        transaction=self._empty_transaction
        return deepcopy(transaction['candidate']) if transaction else None
    def _settled_validation_physical_ids(self,state):
        allowed=set()
        for reader in self._validation_resolution_readers:allowed|=reader(state)
        transaction=self._empty_transaction
        if transaction and transaction['candidate'] is not None and not transaction['attempted_extra']:
            candidate=transaction['candidate']
            rows=[r for r in state['attempts'] if r['request_id']==candidate['request_id']]
            require(len(rows)==1 and row_without_outcome(rows[0])==candidate,'current_validation_receipt_changed')
            allowed.add(candidate['request_id'])
        return allowed
    def halt(self,reason='model_or_storage_failed'):
        transaction=self._empty_transaction
        if transaction is not None:
            if reason in {'model_output_failed','model_or_storage_failed'} and transaction['candidate'] is None:
                try:
                    state=self.snapshot();new=[r for r in state['attempts'] if r['request_id'] not in transaction['prior_ids']]
                    require(state.get('stop_reason') in (None,'request_failed') and len(new)==1
                        and not transaction['attempted_extra'] and not self._stop_requested.is_set()
                        and not self._broken and not self._closed,
                        'single_settled_attempt_required')
                    row=new[0];path=check_ref(row['public_first_output'])
                    require(self._empty_transaction.get('client') is not None
                        and (row['status']=='failed' and reason=='model_or_storage_failed'
                             and CONTRACT_FIELD in row or reason=='model_output_failed'
                             and self._empty_transaction['client'].last_diagnostic.get('online_response_projection',{}).get('status')=='rejected'),
                        'actual_projection_rejection_required')
                    validate_rejected_receipt(row,read(path),requirements=self.rejected_requirements(),request_id=row['request_id'],
                        operation_id=transaction['operation_id'],session_id=transaction['session_id'])
                    binding=row.get(CONTRACT_FIELD)
                    if row['status']=='failed':
                        require(binding and binding['bound_before_application_write'] is True
                            and all(binding[k]==row[k] for k in IDENTITY)
                            and binding['sources']==sources(), 'actual_contract_binding_required')
                    transaction['candidate']=deepcopy(row)
                    return
                except (ValueError,KeyError,TypeError,OSError):pass
            elif reason=='model_or_storage_failed' and transaction['candidate'] is not None:
                # The submit wrapper is still inside its owner lock. It must
                # now validate and persist a failure disposition before returning.
                return
        return super().halt(reason)
    def rejected_requirements(self):
        value=self._empty_transaction
        require(value is not None and value.get('requirements') is not None,'exact_request_requirements_required')
        return deepcopy(value['requirements'])
    def request_json(self,client,request,**kwargs):
        if self._empty_transaction is not None and self._empty_transaction['candidate'] is not None:
            self._empty_transaction['attempted_extra']=True
            self.force_halt_empty_failure()
            raise MinimalCallStopped('no_second_request_after_empty_reply')
        if self._empty_transaction is not None and client.role=='online_fast':
            require(self._empty_transaction.get('client') is None,'one_online_validation_request_required')
            self._empty_transaction['client']=client
            self._empty_transaction['requirements']={k:kwargs.get(k,False) for k in ('require_preference_control','require_task_projection')}
        return super().request_json(client,request,**kwargs)
    def force_halt_empty_failure(self):
        return super().halt('model_or_storage_failed')
    def end_empty_turn(self,*,resolved):
        value=self._empty_transaction
        self._empty_transaction=None
        if value and value['candidate'] is not None and (not resolved or value['attempted_extra']):
            self.force_halt_empty_failure()

