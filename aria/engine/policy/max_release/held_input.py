"""Explicit new-input consent for one fully held, unreceived online request.

No request replay, fee settlement, caller authorization or task state is here.
The original failed physical row is immutable. Consent records only permit a
future separately submitted operation under the same persistent allowance.
"""
from copy import deepcopy
from decimal import Decimal
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import http.client
import json
import math
import ssl
import urllib.error
import uuid

from ..minimal_calls import MinimalCallGuard, MinimalCallStopped
from ..reserved_calls import ReservedCallGuard, FRAMING_TOKENS
from ..transport_failure import transport_failure, validate_transport_failure
from .request_requirements import validate_binding
from .validation_support import require, digest_json, ref, check_ref, read

FIELD = "held_online_before_write"
IDENTITY = ("request_id", "request_sha256", "role", "operation_id", "session_id")


def validate_prewrite_trace(row):
    """Positive bounded TLS-preparation failure; absent events alone never qualify."""
    trace=row['transport_events'];events=trace['events']
    allowed={'connection_preparation_started':True,'connection_preparation_failed_before_request':True}
    require(trace['version']=='connection_trace_v2' and trace['scope']=='local_events_not_delivery_or_billing'
        and trace['absent_events']=='not_observed' and events==allowed,'positive_preapplication_failure_required')
    attempts=trace['connection_preparations']
    require(type(attempts) is list and len(attempts)==2,'original_two_preparations_required')
    for index,item in enumerate(attempts,start=1):
        validate_transport_failure(item['failure']);failure=item['failure']
        require(item['ordinal']==index and item['events']==allowed
            and type(item['elapsed_ms']) in (int,float) and math.isfinite(item['elapsed_ms']) and item['elapsed_ms']>=0
            and failure['opener_call']=='invoked' and failure['response_object']=='not_observed'
            and failure['application_write']=='unobserved' and failure['provider_receipt']=='unobserved'
            and failure['billing_inference']=='none' and failure['exception_kind']=='tls_eof_error',
            'actual_bounded_tls_preparation_failure_required')
    require(row['tls_preparation_candidate']=={'schema':'openrouter_pre_application_tls_v1','physical_preparations':2,
        'logical_extra_before':0,'logical_extra_after':1,'extra_consumed_in_physical':1,
        'original_physical_deadline_shared':True,'application_retry_authorized_by_tls':False,
        'transport_adapter':'explicit_live_endpoint_only'},'original_preparation_allowance_receipt_required')


def source_contract():
    root = Path(__file__).resolve().parents[2]
    names = ("policy/max_release/held_input.py", "policy/max_release/guard.py",
             "policy/max_release/validation_guard.py", "policy/max_release/tls_dispatch.py",
             "policy/max_release/sse_dispatch.py", "policy/response_body.py",
             "policy/max_release/public_capture.py", "../voice/held_input.py", "../voice/minimal.py")
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def validate_partial_response(row):
    from aria.engine.policy.response_body import validate_body_read_observation
    b=row['body_read'];validate_body_read_observation(b)
    require(row['role']=='online_fast' and row['kind']=='turn' and row['status']=='failed'
        and row['model']=='qwen/qwen3.8-max-0902' and row['expected_provider']=='Alibaba'
        and row['usage']=={} and row['http_status']==200 and row['failure_phase']=='read'
        and row['failure_code']=='response_read_failed','actual_online_partial_response_required')
    require(b['complete'] is False and b['framing_checked'] is True and b['framing'] in ('chunked','content_length')
        and 0<b['bytes_received']<b['max_bytes']==2097152 and b['read_mode']=='read1'
        and b['exception_kind']=='incomplete_read' and b['capture_status']=='complete'
        and b['json_status'] in ('invalid_or_incomplete','complete_object','non_object'),'bounded_incomplete_framing_required')
    require(all(row.get(k) is None for k in ('stream_read','public_first_output','returned_model','returned_provider',
        'receipt_identity','unverified_cost_usd','validation_error','transport_failure')),'no_complete_or_offline_output_required')
    events=row['transport_events']['events'];required={'connection_preparation_started','connection_preparation_completed','tls_completed',
        'application_request_started','application_write_attempted','request_write_completed','response_headers_returned'}
    require(row['transport_events']['version']=='connection_trace_v2' and set(events)==required and all(events.values()),'complete_write_and_actual_headers_required')
    obs=row['observation'];require(obs.get('envelope_kind')=='interrupted' and obs.get('body_status')=='read_failed'
        and obs.get('body_read_error_kind')=='incomplete_read','actual_interrupted_observation_required')
    origin=row['actual_partial_body_failure_origin']
    require(origin['schema']=='aria_actual_partial_body_failure_v1' and origin['caught_at_actual_bounded_reader'] is True
        and origin['local_pending_check_failed'] is False and origin['reader_returned_complete_body'] is False
        and all(origin[k]==row[k] for k in IDENTITY) and origin['body_read']==b,'actual_reader_identity_binding_required')
    require(origin['reader_source']==ref(Path(__file__).resolve().parents[1]/'response_body.py')
        and origin['dispatch_source']==ref(Path(__file__).with_name('sse_dispatch.py')),'actual_reader_source_changed')
    receipt=read(check_ref(origin['body_receipt']));expected=deepcopy(b);expected['capture_status']='not_requested'
    require(receipt.get('partial') is True and receipt.get('body_read')==expected
        and receipt.get('visible_projection_saved') is False and receipt.get('full_body_stored') is False
        and receipt.get('hidden_reasoning_stored') is False
        and receipt.get('visible_capture_missing_reason')=='transport_body_incomplete_no_projection','actual_capture_receipt_required')
    directory=Path(origin['body_receipt']['path']).parent
    require(not any((directory/name).exists() for name in ('first-visible.json','stream-public-frames.json')),
        'partial_complete_capture_must_be_absent')
    require(not Path(origin['public_first_output_path']).exists(),'partial_public_output_must_be_absent')

def validate_unreceived(row):
    partial=row.get('failure_phase')=='read'
    if partial:validate_partial_response(row)
    else:
        require(row["role"] == "online_fast" and row["kind"] == "turn" and row["status"] == "failed"
            and row["model"] == "qwen/qwen3.8-max-0902" and row["expected_provider"] == "Alibaba"
            and row["usage"] == {} and row.get("observation") == {}
            and row["failure_phase"] == "pre_response_open" and row["failure_code"] == "response_open_failed",
            "exact_terminal_online_noresponse_required")
        require(all(row.get(k) is None for k in ("http_status", "body_read", "stream_read", "public_first_output",
            "returned_model", "returned_provider", "receipt_identity", "unverified_cost_usd")),
            "observed_or_partial_response_not_eligible")
        failure = row["transport_failure"]
        validate_transport_failure(failure)
        require(failure["opener_call"] == "invoked" and failure["response_object"] == "not_observed"
            and failure["provider_receipt"] == "unobserved" and failure["billing_inference"] == "none",
            "unreceived_transport_evidence_required")
        events = row["transport_events"]["events"]
        prewrite = events.get('connection_preparation_failed_before_request') is True
        required = {"connection_preparation_started", "connection_preparation_completed", "tls_completed",
                    "application_request_started", "application_write_attempted", "request_write_completed"}
        if prewrite:validate_prewrite_trace(row)
        else:require(all(events.get(k) is True for k in required) and not any(v is True for k,v in events.items() if k not in required),
            "full_write_without_response_required")
    require(row["physical_attempt"] == 1 and row.get("recovery_of_request_id") is None
        and row.get("campaign_recovery_ordinal") is None and row["effective_physical_timeout_seconds"] == 90
        and datetime.fromisoformat(row["finished_at"]) >= datetime.fromisoformat(row["started_at"]),
        "original_physical_request_required")
    reserve = row["reservation"]
    amount = Decimal(reserve["reserved_cost_usd"])
    require(amount.is_finite() and amount > 0 and reserve["provider_billing_guarantee"] is False
        and reserve["token_bound_method"] == "full_wire_utf8_bytes_plus_8192_template_plan"
        and reserve["wire_visible_limit"] == 4096 and reserve["billable_completion_limit"] == 8202
        and reserve["output_token_limit"] == 8202 and reserve["gateway_mapping_observed"] is False,
        "complete_unknown_reservation_required")
    bound = row[FIELD]
    validate_binding(row, bound["requirements"])
    require(bound["schema"] == "aria_held_before_application_v1"
        and all(bound[k] == row[k] for k in IDENTITY) and bound["reservation"] == reserve
        and bound["source"] == source_contract() and bound["bound_before_application_write"] is True,
        "held_original_binding_changed")
    if not partial:
        origin = row["held_noresponse_origin"]
        require(all(origin[k] == row[k] for k in IDENTITY)
            and origin["local_pending_check_failed"] is False and origin["response_object_returned"] is False
            and origin["transport_failure"] == failure, "actual_opener_origin_required")
        if origin["origin_kind"] == "tls_preparation_failed":
            require(prewrite and origin['direct_tls_eof_error'] is True and origin['deadline'] is None
                and origin['direct_remote_disconnected'] is False,'actual_prewrite_tls_origin_required')
        elif origin["origin_kind"] == "remote_disconnected":
            require(not prewrite,'fullwrite_origin_cannot_authorize_prewrite')
            require(origin["direct_remote_disconnected"] is True and origin["deadline"] is None,
                "direct_remote_disconnect_required")
        elif origin["origin_kind"] == "local_deadline_expired":
            require(not prewrite,'deadline_origin_cannot_authorize_prewrite')
            value = origin["deadline"]
            require(origin["direct_remote_disconnected"] is False and value["timeout_seconds"] == 90
                and value["check_point"] == "ResponseDeadline.remaining" and value["after_complete_write"] is True
                and all(type(value[k]) in (int,float) and math.isfinite(value[k]) for k in ("expires", "observed"))
                and value["observed"] >= value["expires"], "same_original_deadline_required")
        else:
            raise ValueError("unsupported_noresponse_origin")
    return amount


class HeldInputGuardMixin:
    def __init__(self, *args, **kwargs):
        self._held_ready = False
        self._held_transaction = None
        self._held_readers = []
        self._held_prewrite_readers = []
        self._pending_budget_refusal = None
        self._held_origin = None
        self._held_pending_failure = False
        self._held_pending_identity = None
        self._held_deadline_error = None
        self._held_deadline = None
        super().__init__(*args, **kwargs)
        with self._transaction():
            self._db.execute("CREATE TABLE IF NOT EXISTS held_input_dispositions(request_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS local_budget_refusals(refusal_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        self._held_ready = True

    def budget_refusals(self):
        return [json.loads(r[0]) for r in self._db.execute("SELECT record FROM local_budget_refusals ORDER BY rowid")]

    def _reserve(self,client,request,payload,state):
        self._pending_budget_refusal=None
        try:return super()._reserve(client,request,payload,state)
        except MinimalCallStopped as error:
            if str(error)!="pre_send_budget_exceeded":raise
            # The authoritative parent already validated all limits/prices and
            # refused this exact wire. Retain its finite arithmetic, never text.
            from ..qwen_max_structured import qwen_max_route, validate_qwen_max_client
            spec=self.policy["models"][client.model]
            binding=validate_qwen_max_client(client) if qwen_max_route(client) is not None else None
            bound=len(request.data)+FRAMING_TOKENS
            output=binding["billable_completion_limit"] if binding else payload["max_tokens"]
            amount=(Decimal(bound)*Decimal(spec["input_usd_per_million"])
                +Decimal(output)*Decimal(spec["output_usd_per_million"]))/1000000
            known=Decimal(state["known_cost_usd"] or "0");held=Decimal(state["reserved_cost_usd"])
            record={"schema":"aria_local_budget_refusal_v1","refusal_id":uuid.uuid4().hex,
                "reason":"pre_send_budget_exceeded","at":datetime.now(timezone.utc).isoformat(),
                "ledger_id":self.ledger_id,**deepcopy(getattr(self._local,"scope",{})),
                "role":client.role,"model":client.model,"provider":spec["provider"],
                "request_sha256":sha256(request.data).hexdigest(),"wire_bytes":len(request.data),
                "policy_sha256":self.policy_sha256,"input_token_plan_bound":bound,
                "visible_output_limit":payload["max_tokens"],"billable_output_limit":output,
                "input_usd_per_million":spec["input_usd_per_million"],
                "output_usd_per_million":spec["output_usd_per_million"],
                "known_usd":str(known),"held_usd":str(held),"next_reserved_usd":str(amount),
                "projected_committed_usd":str(known+held+amount),"cap_usd":str(self.threshold),
                "prior_attempts_sha256":digest_json(state["attempts"]),"source":source_contract(),
                "reservation_implementation_sha256":sha256((Path(__file__).resolve().parents[1]/"reserved_calls.py").read_bytes()).hexdigest(),
                "model_request_sent":False,"physical_attempt_created":False,"old_holds_released_usd":"0"}
            require(known+held+amount>Decimal(self.threshold),"budget_refusal_arithmetic_changed")
            self._pending_budget_refusal=(id(error),record)
            raise

    def _persist_budget_refusal(self,error):
        pending=self._pending_budget_refusal;self._pending_budget_refusal=None
        if pending is None or pending[0]!=id(error):return
        record=pending[1]
        # _begin's enclosing reserve transaction has rolled back before here.
        with self._transaction():
            self._db.execute("INSERT INTO local_budget_refusals VALUES(?,?)",
                (record["refusal_id"],json.dumps(record,sort_keys=True,allow_nan=False)))
        require(record in self.budget_refusals(),"budget_refusal_not_durable")

    def held_records(self, *, read_only=False):
        if not self._held_ready:
            return []
        query = "SELECT record FROM held_input_dispositions ORDER BY rowid"
        values = self._read_committed_rows(query) if read_only else self._db.execute(query)
        return [json.loads(v[0]) for v in values]

    def held_row(self, identity, *, read_only=False):
        query = "SELECT record FROM attempts WHERE request_id=?"
        values = (self._read_committed_rows(query, (identity,)) if read_only else
                  self._db.execute(query, (identity,)).fetchall())
        require(len(values) == 1, "unique_original_held_request_required")
        return json.loads(values[0][0])

    def validate_held_record(self, proof, *, read_only=False):
        row = self.held_row(proof["request_id"], read_only=read_only)
        validate_unreceived(row)
        require(proof["schema"] == "aria_held_new_input_disposition_v1" and proof["ledger_id"] == self.ledger_id
            and proof["ledger_row_sha256"] == digest_json(row) and proof["before"] == row[FIELD]["before"]
            and proof["before"]["business_sha256"] == proof["after_business_sha256"]
            and proof["held_usd"] == row["reservation"]["reserved_cost_usd"]
            and proof["replacement_response"] is None and proof["retry_allowed"] is False
            and proof["effect_success"] is False and proof["state"] in {"prepared", "accepted", "erasing", "erased"},
            "held_disposition_changed")
        require(proof["consent"] is None or (type(proof["consent"]) is dict
            and proof["consent"]["confirmed"] is True and proof["consent"]["owner_id"] == proof["before"]["owner_id"]
            and proof["consent"]["failure_id"] == row["request_id"]
            and type(proof["consent"]["request_id"]) is str), "held_consent_changed")
        if proof["state"] == "prepared":require(proof["consent"] is None, "unexpected_held_consent")
        if proof["state"] == "accepted":require(proof["consent"] is not None, "explicit_held_consent_required")
        return row

    def write_held_record(self, proof, *, previous=None):
        self.validate_held_record(proof)
        with self._transaction():
            if previous is None:
                self._db.execute("INSERT INTO held_input_dispositions VALUES(?,?)",
                    (proof["request_id"], json.dumps(proof,sort_keys=True,allow_nan=False)))
            else:
                current = self._db.execute("SELECT record FROM held_input_dispositions WHERE request_id=?", (proof["request_id"],)).fetchone()
                require(current is not None and json.loads(current[0]) == previous, "held_disposition_raced")
                self._db.execute("UPDATE held_input_dispositions SET record=? WHERE request_id=?",
                    (json.dumps(proof,sort_keys=True,allow_nan=False), proof["request_id"]))
        require(proof in self.held_records(), "held_disposition_not_durable")

    def hard_stop_held(self):
        return MinimalCallGuard.halt(self, "model_or_storage_failed")

    def begin_held_turn(self, before, turn_id, session_id):
        require(self._held_transaction is None and not self.snapshot()["stopped"], "held_turn_not_ready")
        self._held_transaction = {"before": deepcopy(before), "turn_id": turn_id, "session_id": session_id,
            "prior_ids": {r["request_id"] for r in self.snapshot()["attempts"]}, "candidate": None, "extra": False}

    def _begin(self, client, request, **kwargs):
        self._held_origin = None;self._held_pending_failure = False
        self._held_pending_identity = None;self._held_deadline_error = None;self._held_deadline = None
        row = super()._begin(client, request, **kwargs)
        transaction = self._held_transaction
        if client.role != "online_fast" or transaction is None:return row
        try:
            require(row["operation_id"] == transaction["turn_id"] and row["session_id"] == transaction["session_id"],
                "actual_held_operation_required")
            row[FIELD] = {"schema":"aria_held_before_application_v1", **{k:row[k] for k in IDENTITY},
                "before": deepcopy(transaction["before"]), "requirements": self.rejected_requirements(),
                "reservation": deepcopy(row["reservation"]), "source": source_contract(), "bound_before_application_write": True}
            ReservedCallGuard._finish(self, row)
            require(self.held_row(row["request_id"]) == row, "held_before_binding_not_durable")
            self._held_pending_identity = {k:row[k] for k in IDENTITY}
            return row
        except BaseException:self.hard_stop_held();raise

    def _held_bind_deadline(self, client, deadline, trace):
        remaining = deadline.remaining
        def observed_remaining():
            try:return remaining()
            except TimeoutError as error:
                if deadline.last is not None and deadline.last >= deadline.expires_at:
                    self._held_deadline_error = id(error)
                    self._held_deadline = {"check_point":"ResponseDeadline.remaining", "timeout_seconds":client.timeout,
                        "expires":deadline.expires_at, "observed":deadline.last,
                        "after_complete_write":"request_write_completed" in trace._events}
                raise
        deadline.remaining = observed_remaining

    def _held_check_pending(self, operation):
        try:
            result=operation()
            for reader in self._held_prewrite_readers:reader()
            return result
        except BaseException:self._held_pending_failure = True;raise

    def _open_observed(self, request, **kwargs):
        try:return super()._open_observed(request, **kwargs)
        except BaseException as error:
            actual = error;seen=set()
            while isinstance(actual, urllib.error.URLError) and id(actual) not in seen and len(seen)<4:
                seen.add(id(actual));actual=object.__getattribute__(actual, "reason")
            disconnected = type(actual) is http.client.RemoteDisconnected
            deadline = id(actual) == self._held_deadline_error
            tls_failed = isinstance(actual,ssl.SSLEOFError)
            if (disconnected or deadline or tls_failed) and self._held_pending_identity is not None:
                self._held_origin = {**self._held_pending_identity,
                    "origin_kind":"local_deadline_expired" if deadline else ("tls_preparation_failed" if tls_failed else "remote_disconnected"),
                    "direct_remote_disconnected":disconnected and not deadline,
                    "deadline":deepcopy(self._held_deadline) if deadline else None,
                    "local_pending_check_failed":self._held_pending_failure, "response_object_returned":False,
                    "transport_failure":transport_failure(error,opener_invoked=True,response_returned=False,diagnostics=True)}
                if tls_failed:self._held_origin['direct_tls_eof_error']=True
            raise

    def _finish(self, row):
        if self._held_origin and self._held_origin["request_id"] == row["request_id"]:
            row["held_noresponse_origin"] = deepcopy(self._held_origin)
        return super()._finish(row)

    def _record_partial_body_failure(self,client,row,error):
        # This hook is called only at the frozen actual bounded-reader exception.
        # Checkers and openers never enter it. Partial bytes are never retained.
        if client.role!='online_fast' or not isinstance(error,http.client.IncompleteRead):return
        require(self._held_pending_identity is not None and self._physical_capture is not None,
            'partial_actual_physical_binding_required')
        require(all(self._held_pending_identity[k]==row[k] for k in IDENTITY),'partial_pending_identity_changed')
        row['actual_partial_body_failure_origin']={'schema':'aria_actual_partial_body_failure_v1',
            **{k:row[k] for k in IDENTITY},'caught_at_actual_bounded_reader':True,
            'local_pending_check_failed':self._held_pending_failure,'reader_returned_complete_body':False,
            'body_read':deepcopy(row['body_read']),'body_receipt':ref(self._physical_capture/'first-body-receipt.json'),
            'reader_source':ref(Path(__file__).resolve().parents[1]/'response_body.py'),
            'dispatch_source':ref(Path(__file__).with_name('sse_dispatch.py')),
            'public_first_output_path':str(self.public_dir/(row['request_id']+'.json'))}

    def _held_current(self):
        transaction = self._held_transaction
        if transaction is None or transaction["extra"]:return None
        rows = [json.loads(v[0]) for v in self._db.execute("SELECT record FROM attempts ORDER BY sequence")]
        rows = [r for r in rows if r["request_id"] not in transaction["prior_ids"]]
        if len(rows) != 1:return None
        row=rows[0]
        try:
            validate_unreceived(row)
            require(row["operation_id"] == transaction["turn_id"] and row["session_id"] == transaction["session_id"], "held_scope_changed")
        except (ValueError,KeyError,TypeError,OSError):return None
        return row

    def halt(self, reason="model_or_storage_failed"):
        if reason in {"model_or_storage_failed", "model_output_failed"}:
            row = self._held_current()
            if row is not None:
                self._held_transaction["candidate"] = deepcopy(row)
                # The unknown row still stops _snapshot until explicit consent.
                return
        return super().halt(reason)

    def request_json(self, client, request, **kwargs):
        transaction=self._held_transaction
        if transaction is not None and transaction["candidate"] is not None:
            transaction["extra"]=True;self.hard_stop_held()
            raise MinimalCallStopped("held_request_replay_forbidden")
        return super().request_json(client,request,**kwargs)

    def end_held_turn(self, prepared):
        transaction=self._held_transaction;self._held_transaction=None
        if transaction and transaction["candidate"] and (not prepared or transaction["extra"]):self.hard_stop_held()

    def _snapshot(self):
        state=super()._snapshot()
        if not self._held_ready:return state
        records=self.held_records()
        state["local_budget_refusals"]=self.budget_refusals()
        for proof in records:self.validate_held_record(proof)
        allowed={p["request_id"] for p in records if p["state"] == "erased" and p["consent"] is not None}
        for reader in self._held_readers:allowed |= reader()
        # Preserve separately qualified known-fee reflection dispositions.
        if state.get("stop_reason") in {"provider_cost_unknown", "request_failed"}:
            allowed |= self._erased_reflection_ids()
            for reader in self._reflection_resolution_readers or []:allowed |= reader()
            unresolved={r["request_id"] for r in state["attempts"] if r["status"] == "failed" or "cost" not in r["usage"]}
            if unresolved and unresolved <= allowed:
                state.update(stopped=False,stop_reason=None)
                if state["calls_used"] >= self.max_requests:state.update(stopped=True,stop_reason="request_limit_reached")
                elif Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) >= Decimal(self.threshold):
                    state.update(stopped=True,stop_reason="observed_cost_threshold_reached")
        state["held_new_input_consents"]=[{"request_id":p["consent"]["request_id"],"failure_id":p["request_id"]}
            for p in records if p["consent"] is not None and p["state"] in {"accepted","erased"}]
        return state
