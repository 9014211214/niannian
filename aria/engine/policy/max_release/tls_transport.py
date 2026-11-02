"""Explicit OpenRouter TLS preparation adapter; no application-request retry.

Standard urllib proxy configuration and verified TLS are inherited unchanged.
The public opener accepts only the exact authorized OpenRouter completion endpoint.
"""
from __future__ import annotations

import http.client
import math
import json
import time
import urllib.parse
import urllib.request

from aria.engine.policy import observed_transport as frozen
from aria.engine.policy.response_body import ResponseDeadline
from aria.engine.policy.transport_failure import transport_failure

SCHEMA = 'openrouter_pre_application_tls_v1'
TLS_REBUILD_DELAY_SECONDS = 1.0


def require_endpoint(request, *, base=False):
    expected = 'https://openrouter.ai/api/v1' + ('' if base else '/chat/completions')
    if request.full_url != expected:
        raise ValueError('exact_openrouter_https_endpoint_required')
    if not base and request.get_method() != 'POST':
        raise ValueError('completion_post_required')


class LogicalTLSAllowance:
    """One extra preparation, distinct from each physical's initial connect."""
    def __init__(self):
        self._used = 0

    @property
    def used(self):
        if type(self._used) is not int or self._used not in (0, 1):
            raise ValueError('invalid_logical_tls_allowance')
        return self._used

    def consume(self):
        if self.used:
            raise ValueError('logical_tls_rebuild_exhausted')
        self._used = 1


class TLSPreparation:
    def __init__(self, *, check, deadline, allowance, bind_headers=lambda request: None, sleep=time.sleep):
        if (not callable(check) or type(deadline) is not ResponseDeadline
                or type(allowance) is not LogicalTLSAllowance or not callable(bind_headers)
                or not callable(sleep)):
            raise ValueError('invalid_candidate_tls_preparation')
        self.check, self.deadline, self.allowance = check, deadline, allowance
        self._sleep = sleep
        self.ordinal = 0
        self.extra_before = allowance.used
        self.rebuild_pending = False
        self._last_time = None
        self._bind_headers = bind_headers
        self._headers_bound = False

    def bind_request_headers(self, request):
        if self._headers_bound:
            raise ValueError('candidate_effective_headers_already_bound')
        self._bind_headers(request)
        self._headers_bound = True

    def now(self):
        value = self.deadline.clock()
        if (type(value) not in (int, float) or not math.isfinite(value)
                or self._last_time is not None and value < self._last_time):
            raise ValueError('invalid_candidate_tls_clock')
        self._last_time = value
        return value

    def before_preparation(self):
        if not self._headers_bound:
            raise ValueError('candidate_effective_headers_not_bound')
        self.check()
        remaining = self.deadline.remaining()
        if self.ordinal and not self.rebuild_pending:
            raise ValueError('unclaimed_candidate_tls_preparation')
        if self.ordinal >= 2:
            raise ValueError('candidate_physical_preparation_limit')
        self.rebuild_pending = False
        self.ordinal += 1
        return remaining, self.ordinal

    def before_application(self, sock):
        self.check()
        sock.settimeout(self.deadline.remaining())

    def claim_rebuild(self, error, trace):
        # Positive evidence comes from the failed connect itself. Merely
        # missing application markers, or matching an error string, is not enough.
        if type(trace) is not CandidateTrace or trace.preparation_recovery is not self:
            return False
        failure = transport_failure(error, opener_invoked=True, response_returned=False, diagnostics=True)
        allowed = (failure['reason_status'] in {'direct', 'typed'}
            and (failure['exception_kind'] == 'tls_eof_error'
                 or failure['exception_kind'] == 'tls_error'
                 and failure.get('tls_details', {}).get('reason_code') == 'UNEXPECTED_EOF_WHILE_READING'))
        preparations = trace._preparations
        proven = (trace._events == {'connection_preparation_started'}
            and len(preparations) == self.ordinal == 1
            and preparations[-1].get('events') == {
                'connection_preparation_started': True,
                'connection_preparation_failed_before_request': True}
            and preparations[-1].get('failure') == failure)
        if not allowed or not proven or self.allowance.used:
            return False
        # A short pause avoids immediately reopening the same failed route.
        # It shares the original deadline and never authorizes a POST replay.
        self.check()
        if self.deadline.remaining() <= TLS_REBUILD_DELAY_SECONDS:
            return False
        self._sleep(TLS_REBUILD_DELAY_SECONDS)
        self.check()
        self.deadline.remaining()
        self.allowance.consume()
        self.rebuild_pending = True
        return True

    def receipt(self):
        return {'schema': SCHEMA, 'physical_preparations': self.ordinal,
            'logical_extra_before': self.extra_before,
            'logical_extra_after': self.allowance.used,
            'extra_consumed_in_physical': self.allowance.used - self.extra_before,
            'original_physical_deadline_shared': True,
            'application_retry_authorized_by_tls': False,
            'transport_adapter': 'explicit_live_endpoint_only'}


class CandidateTrace(frozen.TransportTrace):
    def __init__(self, *, preparation, deadline):
        if type(preparation) is not TLSPreparation or preparation.deadline is not deadline:
            raise ValueError('invalid_candidate_trace')
        # Use the original finite diagnostic shape and strict reader. This
        # isolated type deliberately does not activate legacy PreparationRecovery.
        super().__init__(diagnostics=True, response_deadline=deadline)
        self.preparation_recovery = preparation


class _CandidateConnection(frozen._ObservedConnection):
    def request(self, *args, **kwargs):
        while self.sock is None:
            try:
                self.connect()
            except BaseException as error:
                # Close half-prepared sockets even when cancellation/checks fail.
                self.close()
                preparation = self._trace.preparation_recovery
                if isinstance(error, Exception) and preparation.claim_rebuild(error, self._trace):
                    continue
                if 'connection_preparation_started' in self._trace._events:
                    self._trace._mark('connection_preparation_failed_before_request')
                raise
        # The frozen implementation sees an existing socket, disables auto_open,
        # rechecks before entry/write, and never enters its connect-retry branch.
        return super().request(*args, **kwargs)


class _HTTPSConnection(_CandidateConnection, http.client.HTTPSConnection):
    _https = True


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, trace):
        super().__init__(debuglevel=0)
        self._trace = trace

    def do_open(self, http_class, req, **kwargs):
        # urllib has completed its standard request/proxy headers, but has not
        # created a connection. Bind these effective headers without logging.
        self._trace.preparation_recovery.bind_request_headers(req)
        def connection(*args, **options):
            return _HTTPSConnection(*args, trace=self._trace, **options)
        return super().do_open(connection, req, **kwargs)


def open_candidate(request, *, timeout, trace):
    require_endpoint(request)
    payload = json.loads(request.data)
    if (type(payload) is not dict or type(trace) is not CandidateTrace
            or type(timeout) not in (int, float) or timeout != 90):
        raise ValueError('invalid_candidate_open_contract')
    # No global opener, custom SSL context, alternate trust, proxy override,
    # redirect or application retry handler is installed.
    return urllib.request.build_opener(_HTTPSHandler(trace), frozen._NoRedirect()).open(request, timeout=timeout)
