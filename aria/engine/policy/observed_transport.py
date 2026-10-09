"""Redirect-safe HTTP transport with opt-in local connection events.

Default mode never retries. Explicit preparation recovery can rebuild once
before application request entry; it never retries an application HTTP write.
No global opener installation, custom trust context or billing rule. Absent
events mean unobserved, not proof that no traffic or charge occurred.
"""
from __future__ import annotations

import http.client
import math
import time
import urllib.request

from .transport_failure import transport_failure, validate_transport_failure, preparation_rebuild_eligible


VERSION = "connection_trace_v1"
DIAGNOSTIC_VERSION = "connection_trace_v2"
_EVENTS = frozenset({
    "connection_preparation_started", "connection_preparation_completed", "tls_completed",
    "application_write_attempted", "request_write_completed", "response_headers_returned",
})
_DIAGNOSTIC_EVENTS = _EVENTS | {"connection_preparation_failed_before_request", "application_request_started"}


def validate_trace(value, *, require_complete=False, require_https=True):
    """Validate local observations, never billing, delivery or retry permission.

    v2's failure event is positive evidence from the request method's connect
    exception branch. It does not describe proxy CONNECT traffic or the cause
    of the exception. Absence of any event still means only not observed.
    """
    fields = {"version", "events", "absent_events", "scope"}
    if type(value) is dict and value.get("version") == DIAGNOSTIC_VERSION and "connection_preparations" in value:
        fields.add("connection_preparations")
    if (type(value) is not dict or set(value) != fields
            or type(value.get("version")) is not str or value["version"] not in {VERSION, DIAGNOSTIC_VERSION}
            or value.get("absent_events") != "not_observed"
            or value.get("scope") != "local_events_not_delivery_or_billing"
            or type(value.get("events")) is not dict
            or type(require_complete) is not bool or type(require_https) is not bool):
        raise ValueError("invalid_connection_trace")
    detailed = value["version"] == DIAGNOSTIC_VERSION
    events = value["events"]
    allowed = _DIAGNOSTIC_EVENTS if detailed else _EVENTS
    complete = _EVENTS | ({"application_request_started"} if detailed else set())
    if not require_https:
        complete = complete - {"tls_completed"}
    ready = {"connection_preparation_started", "connection_preparation_completed"}
    if require_https:
        ready.add("tls_completed")
    required = {"connection_preparation_completed": {"connection_preparation_started"},
        "tls_completed": {"connection_preparation_started", "connection_preparation_completed"},
        "application_write_attempted": ready | ({"application_request_started"} if detailed else set()),
        "request_write_completed": {"application_write_attempted"},
        "response_headers_returned": {"request_write_completed"},
        "application_request_started": ready,
        "connection_preparation_failed_before_request": {"connection_preparation_started"}}
    if require_https:
        required["connection_preparation_completed"].add("tls_completed")
    if (not events.keys() <= allowed or any(flag is not True for flag in events.values())
            or any(event in events and not dependencies <= events.keys() for event, dependencies in required.items())
            or "connection_preparation_failed_before_request" in events and events.keys() != {
                "connection_preparation_started", "connection_preparation_failed_before_request"}
            or require_complete and events.keys() != complete):
        raise ValueError("invalid_connection_trace")
    if "connection_preparations" in value:
        preparations = value["connection_preparations"]
        if type(preparations) is not list or len(preparations) > 2:
            raise ValueError("invalid_connection_preparations")
        prior_number = None
        for index, item in enumerate(preparations):
            if (type(item) is not dict or type(item.get("ordinal")) is not int or item["ordinal"] not in {1, 2}
                    or prior_number is not None and item["ordinal"] != prior_number + 1
                    or type(item.get("events")) is not dict):
                raise ValueError("invalid_connection_preparations")
            prior_number = item["ordinal"]
            failed = "failure" in item
            expected_fields = {"ordinal", "events", "elapsed_ms"} | ({"failure"} if failed else set())
            expected_events = {"connection_preparation_started", "connection_preparation_failed_before_request"} if failed else ready
            if (set(item) != expected_fields or item["events"] != dict.fromkeys(expected_events, True)
                    or any(flag is not True for flag in item["events"].values())
                    or type(item["elapsed_ms"]) not in (int, float) or not math.isfinite(item["elapsed_ms"]) or item["elapsed_ms"] < 0):
                raise ValueError("invalid_connection_preparations")
            if failed:
                validate_transport_failure(item["failure"])
                if item["failure"]["version"] != "transport_failure_v2":
                    raise ValueError("invalid_connection_preparations")
            if index < len(preparations) - 1 and (not failed or not _saved_rebuild_eligible(item["failure"])):
                raise ValueError("invalid_connection_preparations")
        if (events and not preparations or preparations and (
                "connection_preparation_completed" in events) == ("failure" in preparations[-1])
                or require_complete and not preparations):
            raise ValueError("invalid_connection_preparations")
    return value


def _saved_rebuild_eligible(failure):
    return failure["exception_kind"] in {"tls_eof_error", "connection_reset", "connection_aborted"} or (
        failure["exception_kind"] == "tls_error"
        and failure["tls_details"].get("reason_code") == "UNEXPECTED_EOF_WHILE_READING")


class PreparationRecovery:
    """One shared logical-call connect budget and one HTTP-open deadline.

    Only standard connect is repeated. All application writes remain single
    attempts, and no connection failure releases a model fee reservation.
    """
    def __init__(self, *, check, timeout, budget, monotonic=time.monotonic):
        if (not callable(check) or not callable(monotonic) or type(timeout) not in (int, float)
                or not math.isfinite(timeout) or timeout <= 0 or type(budget) is not dict
                or set(budget) != {"attempts"} or type(budget["attempts"]) is not int or not 0 <= budget["attempts"] <= 2):
            raise ValueError("invalid_preparation_recovery")
        self.check, self.clock, self.budget = check, monotonic, budget
        self._last_time = None
        self.deadline = self.now() + timeout

    def now(self):
        value = self.clock()
        if (type(value) not in (int, float) or not math.isfinite(value)
                or self._last_time is not None and value < self._last_time):
            raise ValueError("invalid_preparation_clock")
        self._last_time = value
        return value

    def remaining(self):
        remaining = self.deadline - self.now()
        if remaining <= 0:
            raise TimeoutError("preparation_deadline_exhausted")
        return remaining

    def before_preparation(self):
        self.check()
        remaining = self.remaining()
        if self.budget["attempts"] >= 2:
            raise ValueError("logical_connection_preparation_limit")
        self.budget["attempts"] += 1
        return remaining, self.budget["attempts"]

    def before_application(self, sock):
        self.check()
        sock.settimeout(self.remaining())


class TransportTrace:
    def __init__(self, *, diagnostics=False, preparation_recovery=None):
        if type(diagnostics) is not bool:
            raise ValueError("invalid_transport_diagnostics")
        self.diagnostics = diagnostics
        if preparation_recovery is not None and (not diagnostics or type(preparation_recovery) is not PreparationRecovery):
            raise ValueError("invalid_preparation_recovery")
        self.preparation_recovery = preparation_recovery
        self._preparations = []
        self._events = set()

    def _mark(self, event):
        if type(event) is not str or event not in (_DIAGNOSTIC_EVENTS if self.diagnostics else _EVENTS):
            raise ValueError("unknown_transport_event")
        self._events.add(event)

    def snapshot(self):
        result = {"version": DIAGNOSTIC_VERSION if self.diagnostics else VERSION, "events": dict.fromkeys(sorted(self._events), True),
            "absent_events": "not_observed", "scope": "local_events_not_delivery_or_billing"}
        if self.preparation_recovery is not None:
            from copy import deepcopy
            result["connection_preparations"] = deepcopy(self._preparations)
        return result


class _ObservedSocket:
    """Delegate unchanged bytes; the marker precedes the real sendall call."""
    def __init__(self, sock, trace):
        self._socket, self._trace = sock, trace

    def sendall(self, data, *args, **kwargs):
        if self._trace.preparation_recovery is not None:
            self._trace.preparation_recovery.before_application(self._socket)
        self._trace._mark("application_write_attempted")
        return self._socket.sendall(data, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._socket, name)


class _ObservedConnection:
    _https = False

    def __init__(self, *args, trace, **kwargs):
        self._trace = trace
        super().__init__(*args, **kwargs)

    def connect(self):
        recovery = self._trace.preparation_recovery
        preparation = None
        if recovery is not None:
            self.timeout, number = recovery.before_preparation()
            started = recovery.now()
            preparation = {"ordinal": number, "events": {"connection_preparation_started": True}}
            self._trace._preparations.append(preparation)
        self._trace._mark("connection_preparation_started")
        # Standard connect performs TCP, proxy CONNECT (when configured), and
        # HTTPS certificate/hostname verification before returning. The socket
        # is not observed until this completes, so CONNECT is never a POST.
        try:
            super().connect()
        except BaseException as error:
            if preparation is not None:
                preparation["events"]["connection_preparation_failed_before_request"] = True
                preparation["failure"] = transport_failure(error, opener_invoked=True, response_returned=False, diagnostics=True)
                preparation["elapsed_ms"] = round((recovery.now() - started) * 1000, 3)
            raise
        self._trace._mark("connection_preparation_completed")
        if self._https:
            self._trace._mark("tls_completed")
        if preparation is not None:
            preparation["events"]["connection_preparation_completed"] = True
            if self._https:
                preparation["events"]["tls_completed"] = True
            preparation["elapsed_ms"] = round((recovery.now() - started) * 1000, 3)
        self.sock = _ObservedSocket(self.sock, self._trace)

    def request(self, *args, **kwargs):
        while self.sock is None:
            try:
                self.connect()
            except BaseException as error:
                # This positive marker is only reachable before entering the
                # application request method. Proxy CONNECT may have written
                # bytes. No fee reservation or retry policy is changed here.
                recovery = self._trace.preparation_recovery
                # Close any half-prepared TCP/proxy/TLS socket before a fresh
                # standard connect. No parent request/sendall has run here.
                if recovery is not None:
                    self.close()
                    proven = (self._trace._preparations and "failure" in self._trace._preparations[-1]
                        and not self._trace._events & {"application_request_started", "application_write_attempted"})
                    if (isinstance(error, Exception) and proven and recovery.budget["attempts"] < 2
                            and preparation_rebuild_eligible(error)):
                        continue
                if self._trace.diagnostics and "connection_preparation_started" in self._trace._events:
                    self._trace._mark("connection_preparation_failed_before_request")
                raise
        # A later missing socket must not trigger a second automatic connect
        # from HTTPConnection.send after application request writing starts.
        self.auto_open = 0
        if self._trace.preparation_recovery is not None:
            self._trace.preparation_recovery.before_application(self.sock)
        if self._trace.diagnostics:
            self._trace._mark("application_request_started")
        result = super().request(*args, **kwargs)
        self._trace._mark("request_write_completed")
        return result

    def getresponse(self, *args, **kwargs):
        if self._trace.preparation_recovery is not None:
            self._trace.preparation_recovery.before_application(self.sock)
        result = super().getresponse(*args, **kwargs)
        self._trace._mark("response_headers_returned")
        return result


class _ObservedHTTPConnection(_ObservedConnection, http.client.HTTPConnection):
    pass


class _ObservedHTTPSConnection(_ObservedConnection, http.client.HTTPSConnection):
    _https = True


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, trace):
        super().__init__(debuglevel=0)
        self._trace = trace

    def do_open(self, http_class, req, **kwargs):
        def connection(*args, **options):
            return _ObservedHTTPConnection(*args, trace=self._trace, **options)
        return super().do_open(connection, req, **kwargs)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, trace):
        # Let the installed standard library create and configure its verified
        # TLS context; callers cannot pass a different context to this API.
        super().__init__(debuglevel=0)
        self._trace = trace

    def do_open(self, http_class, req, **kwargs):
        def connection(*args, **options):
            return _ObservedHTTPSConnection(*args, trace=self._trace, **options)
        return super().do_open(connection, req, **kwargs)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Let the standard error processor raise the original HTTPError. A
        # redirect must not create an unreserved second application request.
        return None


def open_unredirected(request, *, timeout):
    """Open exactly the registered address, preserving normal proxy/TLS setup.

    A redirect is an HTTP failure of this request, never authority to issue a
    second request with its credentials or consume an unrecorded fee attempt.
    No process-global opener, auth, cookie, or retry handler is installed.
    """
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def open_observed(request, *, timeout, trace):
    if type(trace) is not TransportTrace:
        raise ValueError("invalid_transport_trace")
    # build_opener keeps its normal ProxyHandler. Do not inspect proxy values,
    # install a process-global opener, or add auth/cookie/retry handlers.
    opener = urllib.request.build_opener(_HTTPHandler(trace), _HTTPSHandler(trace), _NoRedirect())
    return opener.open(request, timeout=timeout)
