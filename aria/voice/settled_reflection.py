"""Installed App integration for verified, failed offline-stage disposition."""
from copy import deepcopy
from datetime import datetime
import shutil
from ..engine.memory.offline_queue import OFFLINE_AUDIT_LIMIT
from ..engine.policy.max_release.validation_support import read,ref,check_ref,write_once,require
from ..engine.policy.max_release.reflection_failure import (
    failure_binding,validate_receipt,snapshot,validate_state_pair,seal_failed_job,validate_terminal_transition)

def _current_audit_matches_or_aged_out(audit,current):
    same=[row for row in current if row['session_id']==audit['session_id']]
    if same:return same==[audit]
    # Queue._audit retains only the latest 64 records. A full later window
    # does not revoke a separately persisted, verified terminal transition.
    # Early deletion, a changed current record, or an old/reordered window
    # still fails; absence alone never supplies permission to continue.
    if len(current)!=OFFLINE_AUDIT_LIMIT or len({r['session_id'] for r in current})!=len(current):return False
    times=[datetime.fromisoformat(row['finished_at']) for row in current]
    ended=datetime.fromisoformat(audit['finished_at'])
    return bool(ended.tzinfo and all(value.tzinfo for value in times)
        and times[0]>ended and times==sorted(times))

class SettledOfflineAppMixin:
    def __init__(self,*args,**kwargs):
        self._settled_reflections=None
        super().__init__(*args,**kwargs)
        self._settled_reflections=self.run_dir/'settled-reflection-failures'
        if hasattr(self.call_guard,'begin_reflection_review'):
            if self.call_guard._reflection_resolution_readers is None:
                self.call_guard._reflection_resolution_readers=[]
            self.call_guard._reflection_resolution_readers.append(self._resolved_reflection_physical_ids)
    def _resolved_reflection_physical_ids(self):
        self._resolved_reflection_ids(read_only=False)
        if self._settled_reflections is None or not self._settled_reflections.exists():return set()
        return {read(p)['ledger_row']['request_id'] for p in self._settled_reflections.glob('*.json')}
    def _resolved_reflection_ids(self,*,read_only=True):
        if self._settled_reflections is None or not self._settled_reflections.exists():return set()
        result=set();profile=self.store.load(self._settled_owner).model_dump(mode='json')
        for p in sorted(self._settled_reflections.glob('*.json')):
            receipt=read(p);row=receipt['ledger_row'];review=receipt['failed_review']
            stage=row['role']
            require(stage in ('memory','reflection')
                and receipt['schema']=='aria_settled_'+stage+'_failure_v1' and receipt['reflection_success'] is False
                and receipt['retry_same_request'] is False,'reflection_disposition_changed')
            validate_receipt(row)
            require(self.call_guard._reflection_ledger_row(row['request_id'],read_only=read_only)==row,
                'reflection_ledger_changed')
            require([r for r in self._run['reviews'] if r['request_id']==review['request_id']]==[review],
                'failed_review_changed')
            before=read(check_ref(failure_binding(row)['before_state']));after=read(check_ref(receipt['after_failed_state']))
            validate_state_pair(before,after,review['request_id'],review['session_id'],stage)
            terminal=read(check_ref(receipt['after_terminal_profile']))
            validate_terminal_transition(after['profile'],receipt['original_job'],terminal,receipt['audit'],stage)
            require(receipt['audit'] in terminal['offline_session_audit']
                and _current_audit_matches_or_aged_out(receipt['audit'],profile['offline_session_audit'])
                and receipt['audit']['outcome']==stage+'_failed'
                and not any(j['session_id']==review['session_id'] for j in profile['offline_sessions']),
                'reflection_terminal_audit_changed')
            result.add(review['request_id'])
        return result
    def _run_review_with_settled_reflection(self,request_id,run_id,session_id,persona):
        guard=self.call_guard
        if not hasattr(guard,'begin_reflection_review'):
            return self._run_review_original(request_id,run_id,session_id,persona)
        directory=self._settled_reflections;directory.mkdir(exist_ok=True)
        snapshots=directory/'snapshots';snapshots.mkdir(exist_ok=True)
        def before(row):
            path=snapshots/(row['request_id']+'-before.json');write_once(path,snapshot(self));return ref(path)
        guard.begin_reflection_review(request_id,session_id,before);resolved=False
        try:
            self._run_review_original(request_id,run_id,session_id,persona)
            row=guard._reflection_transaction['candidate']
            if row is None:return
            with self._user_lock(self._settled_owner):
                require(not self._storage_failed and self._run['run_id']==run_id,'reflection_storage_or_owner_changed')
                stage=row['role']
                first=read(check_ref(failure_binding(row)['before_state']));after=snapshot(self)
                review,job=validate_state_pair(first,after,request_id,session_id,stage)
                after_path=snapshots/(row['request_id']+'-after-failed.json');write_once(after_path,after)
                original,terminal,audit=seal_failed_job(self,session_id,stage)
                terminal_path=snapshots/(row['request_id']+'-after-terminal.json');write_once(terminal_path,terminal)
                receipt={'schema':'aria_settled_'+stage+'_failure_v1','ledger_row':row,'failed_review':deepcopy(review),
                    'after_failed_state':ref(after_path),'after_terminal_profile':ref(terminal_path),
                    'original_job':original,'audit':audit,'reflection_success':False,
                    'retry_same_request':False,'continue_new_session':True}
                write_once(directory/(request_id+'.json'),receipt)
                resolved=request_id in self._resolved_reflection_ids()
                require(resolved,'reflection_disposition_not_durable')
        except BaseException:
            guard.force_halt_empty_failure();raise
        finally:guard.end_reflection_review(resolved)

    def _safe_settled_reflection_ids(self):
        try:return self._resolved_reflection_ids()
        except Exception:return set()
    def _forget_settled_reflection_metadata(self):
        directory=self._settled_reflections
        if directory is None or not directory.exists():return
        root=self.run_dir.resolve();target=directory.resolve()
        require(target==root/'settled-reflection-failures' and root in target.parents,
            'invalid_private_reflection_directory')
        shutil.rmtree(target)
    def forget_run(self):
        with self._user_lock(self._settled_owner):
            guard=self.call_guard;receipts=[];markers=[]
            try:
                ids=self._resolved_reflection_ids()
                if self._settled_reflections and self._settled_reflections.exists():
                    for path in sorted(self._settled_reflections.glob('*.json')):
                        value=read(path)
                        require(value['failed_review']['request_id'] in ids,'unverified_forget_disposition')
                        receipts.append((value,path))
                if receipts:markers=guard.begin_reflection_forget(receipts)
                if guard and hasattr(guard,'content_erasure'):guard.content_erasure.erase()
                self._forget_run_original()
                if markers:guard.complete_reflection_forget(markers)
                return self._result()
            except BaseException:
                if guard and hasattr(guard,'force_halt_empty_failure'):guard.force_halt_empty_failure()
                raise

# The historical import and state key remain compatible with installed runs.
SettledReflectionAppMixin=SettledOfflineAppMixin
