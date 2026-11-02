"""Frozen one-physical dispatch with isolated trace/opener integration only.

See dispatch-source-binding-v1.json and dispatch.diff for exact derivation.
No global module patching or default-product promotion is used.
"""
import json
import time
import urllib.error
from aria.engine.policy.minimal_calls import (
    MinimalCallStopped, monotonic_value, wait_until, ResponseDeadline, TransportTrace,
    PreparationRecovery, MAX_MODEL_RESPONSE_BYTES, read_response_body, ResponseBodyLimitError,
    capture_response_headers, safe_usage, observe_success, observe_interrupted,
    transport_failure, _record_failure_stage, _stamp, open_unredirected,
)
from .tls_transport import open_candidate as open_observed
from .reflection_sse import read_reflection_sse, STRUCTURED_STREAM_VERSION, structured_sse_contract
from .sse_adapter import StreamingReflectionClient, validate_stream_client, validate_structured_sse_client
from .online_shape import validate_online_sse_client

def _request_once(self, client, request, *, require_preference_control=False, require_task_projection=False):
    # One transport attempt. Other processes see the committed pending row
    # and cannot start another request while this one is unresolved.
    with self._lock:
        try:
            pacing, previous_request = self._pacing_admission(client, request, {
                "require_preference_control":require_preference_control,
                "require_task_projection":require_task_projection})
            # No transaction is held during waiting. This claim repeats all
            # allowance/configuration checks before recording an attempt.
            row = self._begin(client, request, pacing=pacing, expected_prior_request=previous_request,
                              **{**({"require_preference_control": True} if require_preference_control else {}), **({"require_task_projection": True} if require_task_projection else {})})
        except MinimalCallStopped:
            client._update_diagnostic(error_type="minimal_call_stopped")
            raise
        except Exception:
            self._broken = True
            raise MinimalCallStopped("ledger_preflight_failed") from None
        online_stream = (validate_online_sse_client(client)
            if getattr(client, 'online_stream_protocol', None) is not None else None)
        structured_stream = (validate_structured_sse_client(client)
            if getattr(client, 'structured_stream_protocol', None) is not None else None)
        reflection_stream = isinstance(client, StreamingReflectionClient)
        reflection_contract = validate_stream_client(client) if reflection_stream else None
        if reflection_contract and reflection_contract.get('receipt_mode') == STRUCTURED_STREAM_VERSION:
            structured_stream = structured_sse_contract(client.role)
        application_stream = online_stream or structured_stream
        physical_timeout = (reflection_contract['physical_timeout_seconds']
            if reflection_stream else application_stream['physical_timeout_seconds'] if application_stream else client.timeout)
        row['effective_physical_timeout_seconds'] = physical_timeout
        if reflection_stream:
            row['reflection_provider_contract'] = reflection_contract
        if online_stream:
            row['online_stream_contract'] = online_stream
        if structured_stream:
            row['structured_stream_contract'] = structured_stream
        started = time.perf_counter()
        phase, headers = "local_pacing_check", None
        opener_invoked, response_returned = False, False
        body_deadline = ResponseDeadline(started + physical_timeout) if self._transport_diagnostics else None
        trace = TransportTrace(diagnostics=self._transport_diagnostics, response_deadline=body_deadline) if self._observe_transport else None
        try:
            trace = self._candidate_trace(client, request, row, body_deadline,
                require_preference_control=require_preference_control, require_task_projection=require_task_projection)
            if self._connection_preparation_recovery:
                preparation = PreparationRecovery(check=lambda: self._revalidate_pending_application(
                    client, request, row,
                    **{**({"require_preference_control": True} if require_preference_control else {}), **({"require_task_projection": True} if require_task_projection else {})}),
                    timeout=physical_timeout, monotonic=self._pacing_clock, budget=self._local.preparation_budget)
                trace = TransportTrace(diagnostics=True, preparation_recovery=preparation, response_deadline=body_deadline)
            if pacing is not None:
                current = monotonic_value(self._pacing_clock)
                if self._last_network_started is not None:
                    gap = current - self._last_network_started
                    if gap < self._pacing_interval:
                        raise MinimalCallStopped("request_pacing_clock_changed")
                    if previous_request == self._last_network_request_id:
                        pacing["start_gap_ms"] = round(gap * 1000, 3)
                self._last_network_started, self._reopen_pacing_required = current, False
                self._last_network_request_id = row["request_id"]
            phase = "local_stop_check"
            if self._stop_requested.is_set():
                raise MinimalCallStopped("request_stop_requested")
            phase = "local_bookkeeping"
            client._record_network_request()
            phase, opener_invoked = "pre_response_open", True
            if self.recovery_policy is not None:
                row["http_dispatch_started"] = True
            response = (self._open_observed(request, timeout=physical_timeout, trace=trace) if trace is not None
                        else open_unredirected(request, timeout=physical_timeout))
            phase, response_returned = "response_context", True
            with response as response:
                status = getattr(response, "status", None)
                row["http_status"] = status if type(status) is int else None
                client._update_diagnostic(http_status=row["http_status"], error_type=None)
                headers = capture_response_headers(response)
                phase = "read"
                if body_deadline is not None:
                    row["body_read"] = {}
                    try:
                        if reflection_stream or application_stream:
                            if reflection_stream:
                                validate_stream_client(client)
                            elif structured_stream:
                                validate_structured_sse_client(client)
                            else:
                                validate_online_sse_client(client)
                            row["stream_read"] = {}
                            raw = read_reflection_sse(response, limit=MAX_MODEL_RESPONSE_BYTES,
                                deadline=body_deadline, observation=row["body_read"],
                                stream_observation=row["stream_read"], header_observation=headers,
                                receipt_mode=application_stream['receipt_mode'] if application_stream else None)
                        else:
                            try:
                                raw = read_response_body(response, limit=MAX_MODEL_RESPONSE_BYTES,
                                    deadline=body_deadline, observation=row["body_read"])
                            except BaseException as body_error:
                                self._record_partial_body_failure(client,row,body_error)
                                raise
                    except ResponseBodyLimitError:
                        row["validation_error"] = "model_response_body_limit"
                        raise
                else:
                    raw = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
                if not isinstance(raw, bytes) or len(raw) > MAX_MODEL_RESPONSE_BYTES:
                    row["validation_error"] = "model_response_body_limit"
                    raise MinimalCallStopped("model_response_body_limit")
            phase = "decode"
            decoded = raw.decode("utf-8")
            phase = "parse"
            duplicate_fields = set()
            def object_pairs(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        duplicate_fields.add(key)
                    obj[key] = value
                return obj
            data = json.loads(decoded, object_pairs_hook=object_pairs)
            # HTTP bytes and the outer JSON envelope have been received.
            # Any later schema/content error is not a transport interruption.
            phase = "envelope_validation"
            # Preserve an unambiguous public usage envelope even when a
            # different response field is malformed. Duplicate billing
            # fields remain unknown; picking either value could undercount.
            billing_fields = {"usage", "cost", "prompt_tokens", "completion_tokens", "total_tokens",
                              "completion_tokens_details", "reasoning_tokens"}
            row["usage"] = {} if duplicate_fields & billing_fields else safe_usage(data)
            from aria.engine.policy.qwen_max_structured import qwen_max_route
            from aria.engine.policy.memory_structured import memory_structured_route
            if (qwen_max_route(client) is not None or memory_structured_route(client) is not None
                    or getattr(client, "reflection_structured_route", None) is not None) and duplicate_fields & {"model", "provider"}:
                observed = row["usage"].pop("cost", None)
                if observed is not None:
                    row["unverified_cost_usd"] = observed
                row["fee_verification_error"] = "duplicate_response_identity"
            observation_data = {"usage": row["usage"]} if duplicate_fields else data
            row["observation"] = observe_success(observation_data, header_observation=headers)
            if duplicate_fields:
                row["validation_error"] = "duplicate_response_field"
                raise MinimalCallStopped("duplicate_response_field")
            if not isinstance(data, dict):
                row["validation_error"] = "non_object_response"
                raise ValueError("non_object_response")
            phase = "receipt_validation"
            self._observe_reflection_body(client, row, data)
            self._record_receipt(client, row, data)
            row["status"] = "received"
            if online_stream:
                # A complete, attributable receipt is already accounted for.
                # Reuse the same pending product context before giving its
                # content back; a changed context must not erase that receipt.
                phase = "delivery_context_validation"
                row['delivery_context_validation'] = {
                    'context_bound': getattr(self._local, 'online_context', None) is not None,
                    'status': 'rejected', 'receipt_verified': True,
                    'check': 'existing_online_context'}
                self._check_online_context()
                row['delivery_context_validation']['status'] = 'passed'
            return data
        except urllib.error.HTTPError as exc:
            if not opener_invoked:
                _record_failure_stage(row, phase, exc)
                row["transport_failure"] = transport_failure(exc, opener_invoked=False, response_returned=False,
                    diagnostics=self._transport_diagnostics)
                client._update_diagnostic(error_type="minimal_request_failed")
                raise MinimalCallStopped("request_failed") from None
            row["http_status"] = exc.code
            row["observation"], row["usage"], failure_metadata = self._observe_http_failure(exc)
            from aria.engine.policy.qwen_max_structured import qwen_max_route
            from aria.engine.policy.memory_structured import memory_structured_route
            if (qwen_max_route(client) is not None or memory_structured_route(client) is not None
                    or getattr(client, "reflection_structured_route", None) is not None):
                observed = row["usage"].pop("cost", None)
                if observed is not None:
                    row["unverified_cost_usd"] = observed
                row["fee_verification_error"] = "http_error_cost_unverified"
            row.update(failure_metadata)
            row.update(failure_phase="http_status", failure_code="http_error")
            client._update_diagnostic(http_status=exc.code, error_type="http_error")
            raise MinimalCallStopped("request_failed") from None
        except Exception as exc:
            if phase in {"local_pacing_check", "local_stop_check", "local_bookkeeping", "pre_response_open", "response_context"}:
                row["transport_failure"] = transport_failure(exc, opener_invoked=opener_invoked,
                    response_returned=response_returned, diagnostics=self._transport_diagnostics)
            if headers is not None and phase in {"read", "decode", "parse"}:
                row["observation"] = observe_interrupted(headers, phase=phase, error=exc)
            if phase == "delivery_context_validation":
                row.update(failure_phase=phase, failure_code='online_delivery_context_changed')
            else:
                _record_failure_stage(row, phase, exc)
            client._update_diagnostic(error_type="minimal_request_failed")
            raise MinimalCallStopped("request_failed") from None
        finally:
            if trace is not None:
                row["transport_events"] = trace.snapshot()
            if row["status"] == "pending":
                row["status"] = "failed"
            row["finished_at"] = _stamp()
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            if self.recovery_policy is not None:
                row["logical_elapsed_ms"] = round((self._recovery_clock() - self._local.logical_started) * 1000, 3)
            try:
                self._finish(row)
            except Exception:
                self._broken = True
                raise MinimalCallStopped("ledger_result_save_failed") from None
