"""Owner confirmation for a future input; never replay an unanswered turn."""
from copy import deepcopy
from datetime import datetime, timezone
import re
import json
import sqlite3

from ..engine.policy.max_release.held_input import FIELD, validate_unreceived
from ..engine.policy.max_release.validation_support import require, digest_json

_ACTION = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _sqlite_read_locked(error):
    if hasattr(sqlite3, "SQLITE_BUSY"):
        return getattr(error, "sqlite_errorcode", -1) & 0xff in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    # Python 3.10 exposes only SQLite's error message. A table or schema
    # lock can include its name after the colon.
    message = str(error)
    return message == "database is locked" or message.partition(": ")[0] in (
        "database table is locked", "database schema is locked")


def business(app):
    profile=app.store.load(app._settled_owner).model_dump(mode="json")
    sharing=deepcopy(app.sharing_state());sharing.pop("writable",None)
    return {"profile":{k:v for k,v in profile.items() if k not in {"pending_session","last_interaction_at"}},
        "care":app.care.elder_view(app._settled_owner),"sharing":sharing}


class HeldInputAppMixin:
    def __init__(self,*args,**kwargs):
        self._held_run_guard_ready=False
        self._held_run_integrity_failed=False
        super().__init__(*args,**kwargs)
        if hasattr(self.call_guard,"held_records"):
            self._held_disk_revision=(digest_json(self._held_disk_run()) if self.run_path.exists() else None)
            self._held_run_guard_ready=True
            self.call_guard._held_readers.append(self._held_accepted_physical_ids)
            self.call_guard._held_prewrite_readers.append(self._held_require_persisted_run)
            self.call_guard._online_authority_reader = self._online_dispatch_authority

    def _online_dispatch_authority(self):
        self._held_require_persisted_run()
        return {"owner_id":self._settled_owner,"run_id":self._run["run_id"],
                "business_sha256":digest_json(business(self))}

    def _held_disk_run(self):
        from ..engine.memory.crypto import decrypt_if_needed
        return json.loads(decrypt_if_needed(
            self.run_path.read_text(encoding="utf-8"),self.run_dir,who="minimal-run"))

    def _held_run_failure(self,error):
        self._held_run_integrity_failed=True
        self._storage_failed=True
        self._halt_held_models()
        if isinstance(error,OSError):raise ValueError("held_persisted_run_unreadable") from error
        raise error

    def _held_before_run_save(self):
        if not self._held_run_guard_ready:return
        try:
            require(not self._held_run_integrity_failed,"held_persisted_run_already_failed")
            if self._held_disk_revision is None:
                require(not self.run_path.exists(),"held_run_created_externally")
            else:
                require(digest_json(self._held_disk_run())==self._held_disk_revision,
                    "held_persisted_run_revision_changed")
        except BaseException as error:self._held_run_failure(error)

    def _held_after_run_save(self):
        if not self._held_run_guard_ready:return
        try:
            persisted=self._held_disk_run()
            require(persisted==self._run,"held_persisted_run_write_mismatch")
            self._held_disk_revision=digest_json(persisted)
        except BaseException as error:self._held_run_failure(error)

    def _held_require_persisted_run(self):
        """Read the ordinary durable run; never overwrite an external change.

        Called at explicit consent, new-input entry before any normal save, and
        each actual guard prewrite check after the normal pending save.
        """
        try:
            self._held_before_run_save()
            require(self._held_disk_run()==self._run,"held_persisted_run_mismatch")
        except BaseException as error:self._held_run_failure(error)

    def _held_current_record(self,proof,*,read_only=False):
        self._held_require_persisted_run()
        guard=self.call_guard;row=guard.validate_held_record(proof,read_only=read_only)
        before=proof["before"]
        require(before["owner_id"]==self._settled_owner
            and self.store.load(self._settled_owner).user_id==before["owner_id"]
            and before["run_id"]==self._run["run_id"],"held_current_owner_run_changed")
        turns=[r for r in self._run["turns"] if r["turn_id"]==before["turn_id"]]
        require(len(turns)==1 and digest_json(turns[0])==proof["failed_turn_sha256"]
            and turns[0]["session_id"]==row["session_id"] and turns[0]["status"]=="failed"
            and turns[0]["response"]=="" and turns[0]["source"]=="model_failed"
            and turns[0].get("requires_review") is True,"held_original_failed_turn_changed")
        return row

    def _held_accepted_physical_ids(self, *, read_only=False):
        if not hasattr(self.call_guard,"held_records"):return set()
        result=set()
        for proof in self.call_guard.held_records(read_only=read_only):
            if proof["state"] in {"prepared","accepted"}:
                try:self._held_current_record(proof, read_only=read_only)
                except sqlite3.OperationalError as error:
                    if read_only and _sqlite_read_locked(error):
                        raise
                    self._halt_held_models()
                    return set()
                except Exception:
                    self._halt_held_models()
                    # Preserve the actual readable fee ledger in the stopped
                    # view; a corrupt run is not an unreadable/empty ledger.
                    return set()
                if proof["state"]=="accepted":result.add(proof["request_id"])
            elif proof["state"]=="erasing":
                raise ValueError("held_forget_incomplete")
        return result

    def _held_accepted_turn_ids(self):
        physical=self._held_accepted_physical_ids(read_only=True)
        return {p["before"]["turn_id"] for p in self.call_guard.held_records(read_only=True)
                if p["request_id"] in physical} if physical else set()

    def _safe_held_turn_ids(self):
        try:return sorted(self._held_accepted_turn_ids())
        except Exception:return []

    def _held_view(self):
        guard=self.call_guard
        if not hasattr(guard,"held_records"):return {"available":False}
        try:
            candidates=[]
            for proof in guard.held_records(read_only=True):
                if proof["state"]=="prepared":
                    self._held_current_record(proof, read_only=True);candidates.append(proof)
            if len(candidates)!=1:return {"available":False}
            proof=candidates[0]
            if self._review_is_running():
                return {"available":False,"failure_id":proof["request_id"],
                    "snapshot_status":"unavailable_during_background"}
            # This is presentation only. While review is active, the cached
            # fee view cannot enable consent; the real consent path below
            # requires the fresh guard snapshot and unchanged source checks.
            state=self._model_usage()
            allowed=self._held_accepted_physical_ids()|{proof["request_id"]}|guard._erased_reflection_ids()
            for reader in guard._reflection_resolution_readers or []:allowed|=reader()
            unresolved={r["request_id"] for r in state["attempts"] if r["status"]=="failed" or "cost" not in r["usage"]}
            available=(state["stop_reason"] in {"provider_cost_unknown","request_failed"} and unresolved<=allowed)
            return {"available":available and not self._review_is_running()
                    and not self._storage_failed and not self._closing,
                "failure_id":proof["request_id"],"session_id":proof["before"]["session_id"],
                "held_usd":proof["held_usd"],"cost_status":"unknown_fully_reserved",
                "action":"continue_new_input","response_retry_allowed":False,
                "notice":"这条请求没有收到回复，费用仍不确定。保留全部费用预留；本人确认后仅可提交新的输入，不重发这条请求。"}
        except sqlite3.OperationalError as error:
            if _sqlite_read_locked(error):
                # A concurrent commit is not proof that a source changed. Show
                # unknown without granting consent or caching a valid result.
                return {"available":False,"requires_review":True,
                    "snapshot_status":"historical_receipt_read_unavailable"}
            self._halt_held_models()
            return {"available":False,"requires_review":True}
        except Exception:
            self._halt_held_models()
            return {"available":False,"requires_review":True}

    def _submit_with_held_input(self,text,turn_id,session_id,**kwargs):
        with self._user_lock(self._settled_owner):
            guard=self.call_guard
            if not hasattr(guard,"begin_held_turn"):
                return self._submit_with_settled_validation(text,turn_id,session_id,**kwargs)
            records=guard.held_records()
            if any(p["before"]["turn_id"]==turn_id for p in records):
                if any(t["turn_id"]==turn_id for t in self._run["turns"]):
                    return self._submit_original(text,turn_id,session_id,**kwargs)
                return self._result(False,"该旧请求已保留为失败记录，不能重新发送。",duplicate=True,request_sent=False)
            try:self._held_require_persisted_run()
            except (ValueError, OSError):
                return self._result(False,"持久记录与当前状态不一致或读取失败，已停止发送；原记录和费用预留保留。",request_sent=False)
            view=self.state()
            if (view["requires_review"] or guard.snapshot()["stopped"] or session_id!=view["session_id"]
                    or any(t["turn_id"]==turn_id for t in self._run["turns"])):
                return self._submit_with_settled_validation(text,turn_id,session_id,**kwargs)
            # In-memory snapshots are used only to prove no business commit;
            # persisted proofs contain identities, hashes, amounts and times.
            try:
                original_business=business(self);original_run=deepcopy(self._run)
                original_profile=self.store.load(self._settled_owner).model_dump(mode="json")
                checkpoint=original_profile.get("pending_session")
                before={"owner_id":self._settled_owner,"run_id":self._run["run_id"],"turn_id":turn_id,
                    "session_id":session_id,"business_sha256":digest_json(original_business),
                    "run_prefix_sha256":digest_json(original_run),
                    "checkpoint_expires_at":checkpoint["expires_at"] if checkpoint else None}
                guard.begin_held_turn(before,turn_id,session_id)
            except BaseException:
                self._storage_failed=True;guard.hard_stop_held();raise
            prepared=False
            try:
                result=self._submit_with_settled_validation(text,turn_id,session_id,**kwargs)
                row=guard._held_transaction["candidate"]
                if row is None:return result
                validate_unreceived(row)
                require(not result.get("ok") and not self._storage_failed,"held_actual_failure_required")
                failed=self._run["turns"][-1]
                require(self._run["turns"]==original_run["turns"]+[failed]
                    and failed["turn_id"]==turn_id and failed["session_id"]==session_id
                    and failed["status"]=="failed" and failed["response"]==""
                    and failed["source"]=="model_failed" and failed.get("requires_review") is True
                    and not failed.get("protection_prelude") and not failed.get("premodel_task"),"held_original_failure_required")
                remainder=deepcopy(self._run);remainder["turns"]=original_run["turns"]
                require(remainder==original_run and business(self)==original_business,"held_business_halfcommit")
                pair=self._sessions.get(self._settled_owner)
                require(pair is not None and pair[1].session_id==session_id,"held_context_changed")
                engine,ctx=pair;index=failed["turn_index"]
                require(index==len(ctx.turns)-1 and index in ctx.working_memory.get("_sensitive_memory_excluded_turns",[])
                    and ctx.working_memory.get("online_turn_decision",{}).get("status")=="model_failure",
                    "held_failed_turn_not_excluded")
                after_profile=self.store.load(self._settled_owner).model_dump(mode="json")
                after_checkpoint=after_profile.get("pending_session")
                require(checkpoint is not None and after_checkpoint is not None
                    and after_checkpoint["session_id"]==session_id
                    and index not in after_checkpoint["turn_indices"],"held_checkpoint_changed")
                varying={"updated_at","expires_at","turns","turn_indices","original_turn_count"}
                require({k:v for k,v in checkpoint.items() if k not in varying}
                    =={k:v for k,v in after_checkpoint.items() if k not in varying},"held_checkpoint_business_changed")
                prior={t["turn_index"]:t for t in original_run["turns"] if t["session_id"]==session_id}
                require(len(after_checkpoint["turn_indices"])==len(after_checkpoint["turns"])
                    and all(i in prior and prior[i]["status"]=="completed"
                        and t=={"user":prior[i]["text"],"agent":prior[i]["response"]}
                        for i,t in zip(after_checkpoint["turn_indices"],after_checkpoint["turns"])),"held_checkpoint_prefix_changed")
                # A failed request cannot extend the original retained session.
                expiry=datetime.fromisoformat(checkpoint["expires_at"]).astimezone(timezone.utc)
                current_expiry=datetime.fromisoformat(engine.persona.pending_session.expires_at).astimezone(timezone.utc)
                if current_expiry>expiry:
                    engine.persona.pending_session.expires_at=checkpoint["expires_at"]
                    self.store.save(engine.persona)
                from ..engine.context import Turn
                ctx.turns[index]=Turn(user="",agent="")
                proof={"schema":"aria_held_new_input_disposition_v1","request_id":row["request_id"],
                    "ledger_id":guard.ledger_id,"ledger_row_sha256":digest_json(row),"before":before,
                    "after_business_sha256":digest_json(business(self)),"failed_turn_sha256":digest_json(failed),
                    "held_usd":row["reservation"]["reserved_cost_usd"],"state":"prepared","consent":None,
                    "replacement_response":None,"retry_allowed":False,"effect_success":False}
                guard.write_held_record(proof)
                self._held_current_record(proof);prepared=True
                return {**result,"state":self.state(),"held_unreceived_failure":True,"response_delivered":False,
                        "continue_new_input_requires_confirmation":True}
            except BaseException:
                guard.hard_stop_held();raise
            finally:
                guard.end_held_turn(prepared)

    def continue_new_input(self,request_id,failure_id,session_id,*,confirmed=False,read_only=False,actor=None):
        with self._user_lock(self._settled_owner):
            if actor is None:actor=self._settled_owner
            if actor!=self._settled_owner:raise PermissionError("只有本体验的本人可以确认继续新的输入。")
            guard=self.call_guard
            if (read_only is not False or confirmed is not True or not hasattr(guard,"held_records")
                    or self._storage_failed or self._closing or self._review_is_running()
                    or type(request_id) is not str or not _ACTION.fullmatch(request_id)):
                return self._result(False,"请本人明确确认；只读、处理中或保存失败时不能继续。")
            try:
                records=guard.held_records()
                existing=[p for p in records if p["consent"] and p["consent"]["request_id"]==request_id]
                if existing:
                    require(len(existing)==1 and existing[0]["request_id"]==failure_id
                        and existing[0]["before"]["session_id"]==session_id,"held_action_identity_changed")
                    if existing[0]["state"] in {"erasing","erased"}:
                        return self._result(False,"该旧体验已删除或正在删除，此操作不再适用。",duplicate=True,request_sent=False)
                    self._held_current_record(existing[0])
                    return self._result(duplicate=True,request_sent=False)
                matches=[p for p in records if p["request_id"]==failure_id]
                require(len(matches)==1,"held_failure_identity_missing")
                proof=matches[0]
                if proof["state"] in {"erasing","erased"}:
                    return self._result(False,"该旧体验已删除或正在删除，此操作不再适用。",request_sent=False)
                self._held_current_record(proof)
                require(proof["state"]=="prepared" and proof["before"]["session_id"]==session_id,
                    "held_failure_not_current")
                # The entire unresolved set must be this exact failure or an
                # already independently bound disposition; no global reset.
                state=guard.snapshot()
                require(state["stop_reason"] in {"provider_cost_unknown","request_failed"},"held_not_the_only_stop")
                allowed=self._held_accepted_physical_ids()|{failure_id}
                allowed|=guard._erased_reflection_ids()
                for reader in guard._reflection_resolution_readers or []:allowed|=reader()
                unresolved={r["request_id"] for r in state["attempts"] if r["status"]=="failed" or "cost" not in r["usage"]}
                require(unresolved<=allowed,"other_unresolved_requests")
                accepted={**proof,"state":"accepted","consent":{"request_id":request_id,"failure_id":failure_id,
                    "owner_id":self._settled_owner,"confirmed":True,"at":datetime.now(timezone.utc).isoformat()}}
                guard.write_held_record(accepted,previous=proof)
                require(failure_id in self._held_accepted_physical_ids(),"held_consent_not_durable")
                return self._result(request_sent=False,old_hold_released_usd="0",continue_new_input=True)
            except (ValueError,KeyError,TypeError):
                return self._result(False,"记录或范围已变化，未恢复；请核查本地记录与费用。",request_sent=False)
            except BaseException:
                guard.hard_stop_held();self._storage_failed=True;raise

    def _begin_held_forget(self):
        guard=self.call_guard
        if not hasattr(guard,"held_records"):return []
        self._held_before_run_save()
        values=[]
        for proof in guard.held_records():
            if proof["state"] in {"prepared","accepted"}:
                self._held_current_record(proof)
                marker={**proof,"state":"erasing"};guard.write_held_record(marker,previous=proof);values.append(marker)
        return values

    def _complete_held_forget(self,values):
        for previous in values:
            self.call_guard.write_held_record({**previous,"state":"erased"},previous=previous)
        if self._held_run_guard_ready:
            require(not self.run_path.exists(),"held_deleted_run_still_present")
            self._held_disk_revision=None
