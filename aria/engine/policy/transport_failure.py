"""Finite exception categories, without claiming an unobserved network phase.

No exception message, argument, class name, URL, address, header, or proxy value
is retained. A local exception category does not establish provider receipt or
billing. In particular, URLError can wrap an error after a partial POST write.
"""
from __future__ import annotations

import errno
import http.client
import socket
import ssl
import urllib.error


VERSION = "transport_failure_v1"
DIAGNOSTIC_VERSION = "transport_failure_v2"
_MAX_URL_REASON_DEPTH = 4
_ERRNO_BUCKETS = {
    "timeout": {errno.ETIMEDOUT, 10060},
    "connection_refused": {errno.ECONNREFUSED, 10061},
    "connection_reset": {errno.ECONNRESET, 10054},
    "connection_aborted": {errno.ECONNABORTED, 10053},
    "network_unreachable": {errno.ENETUNREACH, 10051},
    "host_unreachable": {errno.EHOSTUNREACH, 10065},
    "broken_pipe": {errno.EPIPE},
    "permission_denied": {errno.EACCES, errno.EPERM, 10013},
}
_SSL_KINDS = (
    (ssl.SSLCertVerificationError, "tls_certificate_error"),
    (ssl.SSLEOFError, "tls_eof_error"), (ssl.SSLZeroReturnError, "tls_clean_close"),
    (ssl.SSLSyscallError, "tls_syscall_error"),
    (ssl.SSLWantReadError, "tls_want_read"), (ssl.SSLWantWriteError, "tls_want_write"),
    (ssl.SSLError, "tls_error"),
)
_SSL_ERRNOS = {1: "ssl_error", 2: "want_read", 3: "want_write", 5: "syscall",
               6: "zero_return", 8: "eof", 10: "invalid_error_code"}
# Finite OpenSSL symbols only, never free-form messages or certificate data.
# These describe the local exception, not a transient/retry classification.
_SSL_REASONS = frozenset({
    "UNEXPECTED_EOF_WHILE_READING", "CERTIFICATE_VERIFY_FAILED", "WRONG_VERSION_NUMBER",
    "SSLV3_ALERT_HANDSHAKE_FAILURE", "TLSV1_ALERT_PROTOCOL_VERSION", "TLSV1_ALERT_INTERNAL_ERROR",
    "TLSV1_ALERT_UNKNOWN_CA", "TLSV1_ALERT_CERTIFICATE_EXPIRED", "SSLV3_ALERT_BAD_CERTIFICATE",
    "SSLV3_ALERT_CERTIFICATE_UNKNOWN", "TLSV1_ALERT_DECODE_ERROR", "TLSV1_ALERT_DECRYPT_ERROR",
    "NO_CIPHERS_AVAILABLE", "NO_SHARED_CIPHER", "UNSUPPORTED_PROTOCOL",
})
_SSL_LIBRARIES = frozenset({"SSL", "X509", "X509V3", "PEM", "ASN1", "BIO", "SYS"})
_VERIFY_CODES = frozenset({2, 9, 10, 18, 19, 20, 21, 23, 24, 26, 27, 28, 62, 64, 66, 67, 68})


def _is(error, cls):
    # Avoid an arbitrary object's __class__ property or generated class name.
    return issubclass(type(error), cls)


def _reason(error):
    try:
        return object.__getattribute__(error, "reason")
    except Exception:
        return None


def _kind(error, *, diagnostics=False):
    if diagnostics:
        for cls, label in _SSL_KINDS:
            if _is(error, cls):
                return label
    for cls, label in (
        (ssl.SSLCertVerificationError, "tls_certificate_error"),
        (ssl.SSLError, "tls_error"),
        (socket.gaierror, "name_resolution_error"),
        (socket.herror, "name_resolution_error"),
        (http.client.RemoteDisconnected, "remote_disconnected"),
        (http.client.IncompleteRead, "incomplete_read"),
        (TimeoutError, "timeout"),
        (ConnectionRefusedError, "connection_refused"),
        (ConnectionResetError, "connection_reset"),
        (ConnectionAbortedError, "connection_aborted"),
        (BrokenPipeError, "broken_pipe"),
        (http.client.HTTPException, "http_protocol_error"),
        (OSError, "os_error"),
    ):
        if _is(error, cls):
            return label
    return "other_exception" if _is(error, BaseException) else "untyped_reason"


def _ssl_details(error):
    # Read the built-in instance dictionary and errno slot directly. Do not
    # execute a subclass property, stringify an unknown value, or inspect args.
    attrs = BaseException.__dict__["__dict__"].__get__(error, type(error))
    number = OSError.errno.__get__(error, type(error))
    result = {"ssl_error_category": _SSL_ERRNOS.get(number, "unrecognized") if type(number) is int else "absent"}
    for source, target, allowed, expected_type in (
        ("reason", "reason_code", _SSL_REASONS, str),
        ("library", "library_code", _SSL_LIBRARIES, str),
        ("verify_code", "verify_code", _VERIFY_CODES, int),
    ):
        value = attrs.get(source)
        # Certificate verify codes are meaningful only for that exception.
        present = value is not None and (source != "verify_code" or _is(error, ssl.SSLCertVerificationError))
        recognized = present and type(value) is expected_type and value in allowed
        result[target + "_status"] = "recognized" if recognized else "unrecognized" if present else "absent"
        if recognized:
            result[target] = value
    return result


def _errno_bucket(error):
    if _is(error, (ssl.SSLError, socket.gaierror, socket.herror)):
        # These use different numeric namespaces, not POSIX/Windows errno.
        return "not_applicable"
    if not _is(error, OSError):
        return "not_available"
    # Read the built-in slot, never a subclass property or free-form args.
    try:
        number = OSError.errno.__get__(error, type(error))
    except Exception:
        return "not_available"
    if type(number) is not int:
        return "not_available"
    for label, numbers in _ERRNO_BUCKETS.items():
        if number in numbers:
            return label
    return "other_errno"


def transport_failure(error, *, opener_invoked, response_returned, diagnostics=False):
    """Observe only a local call boundary; application writes remain unknown."""
    if (type(opener_invoked) is not bool or type(response_returned) is not bool or type(diagnostics) is not bool
            or response_returned and not opener_invoked):
        raise ValueError("invalid_transport_boundary")
    is_url = _is(error, urllib.error.URLError)
    current, seen, depth, status = error, set(), 0, "direct"
    while _is(current, urllib.error.URLError):
        if id(current) in seen:
            status, current = "cycle", None
            break
        if depth >= _MAX_URL_REASON_DEPTH:
            status, current = "depth_limit", None
            break
        seen.add(id(current))
        current = _reason(current)
        depth += 1
        status = "typed" if _is(current, BaseException) else "untyped"
    result = {
        "version": DIAGNOSTIC_VERSION if diagnostics else VERSION,
        "exception_wrapper": "url_error" if is_url else "direct",
        "reason_status": status,
        "exception_kind": _kind(current, diagnostics=diagnostics),
        "errno_category": _errno_bucket(current),
        "opener_call": "invoked" if opener_invoked else "not_invoked",
        "response_object": "returned" if response_returned else "not_observed",
        "tcp_connection": "unobserved",
        "tls_handshake": "unobserved",
        "application_write": "unobserved",
        "provider_receipt": "unobserved",
        "billing_inference": "none",
    }
    if diagnostics and _is(current, ssl.SSLError):
        result["tls_details"] = _ssl_details(current)
    return result


def validate_transport_failure(value):
    """Accept original v1 or finite opt-in v2, without migrating old records."""
    choices = {
        "version": {VERSION, DIAGNOSTIC_VERSION}, "exception_wrapper": {"url_error", "direct"},
        "reason_status": {"direct", "cycle", "depth_limit", "typed", "untyped"},
        "exception_kind": {"tls_certificate_error", "tls_error", "name_resolution_error", "remote_disconnected",
            "incomplete_read", "timeout", "connection_refused", "connection_reset", "connection_aborted",
            "broken_pipe", "http_protocol_error", "os_error", "other_exception", "untyped_reason"},
        "errno_category": {*_ERRNO_BUCKETS, "not_applicable", "not_available", "other_errno"},
        "opener_call": {"invoked", "not_invoked"}, "response_object": {"returned", "not_observed"},
        "tcp_connection": {"unobserved"}, "tls_handshake": {"unobserved"}, "application_write": {"unobserved"},
        "provider_receipt": {"unobserved"}, "billing_inference": {"none"},
    }
    if type(value) is not dict:
        raise ValueError("invalid_transport_failure")
    detailed = value.get("version") == DIAGNOSTIC_VERSION
    if detailed:
        choices["exception_kind"].update(label for _, label in _SSL_KINDS)
    tls = detailed and type(value.get("exception_kind")) is str and value["exception_kind"] in {label for _, label in _SSL_KINDS}
    if (set(value) != set(choices) | ({"tls_details"} if tls else set())
            or any(type(value.get(key)) is not str or value[key] not in allowed for key, allowed in choices.items())
            or value["response_object"] == "returned" and value["opener_call"] != "invoked"):
        raise ValueError("invalid_transport_failure")
    if tls:
        detail = value["tls_details"]
        expected = {"ssl_error_category", "reason_code_status", "library_code_status", "verify_code_status"}
        if (type(detail) is not dict or type(detail.get("ssl_error_category")) is not str
                or detail["ssl_error_category"] not in {*_SSL_ERRNOS.values(), "absent", "unrecognized"}):
            raise ValueError("invalid_transport_failure")
        for key, allowed, expected_type in (("reason_code", _SSL_REASONS, str),
                ("library_code", _SSL_LIBRARIES, str), ("verify_code", _VERIFY_CODES, int)):
            status = detail.get(key + "_status")
            if type(status) is not str or status not in {"absent", "recognized", "unrecognized"}:
                raise ValueError("invalid_transport_failure")
            if status == "recognized":
                expected.add(key)
                if type(detail.get(key)) is not expected_type or detail[key] not in allowed:
                    raise ValueError("invalid_transport_failure")
            if key == "verify_code" and value["exception_kind"] != "tls_certificate_error" and status != "absent":
                raise ValueError("invalid_transport_failure")
        if set(detail) != expected or value["errno_category"] != "not_applicable":
            raise ValueError("invalid_transport_failure")
    return value


def preparation_rebuild_eligible(error):
    """Narrow local exception types; caller must separately prove no request."""
    kind = _kind(error, diagnostics=True)
    # The same finite classification is persisted and checked on read. Some
    # HTTP exceptions also inherit ConnectionResetError; they are not a typed
    # transport reset and must not gain permission through that base class.
    if kind in {"tls_eof_error", "connection_reset", "connection_aborted"}:
        return True
    if kind == "tls_error":
        detail = _ssl_details(error)
        return detail.get("reason_code") == "UNEXPECTED_EOF_WHILE_READING"
    return False
