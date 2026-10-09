"""Explicit, opt-in HTTPS transport for the reminder-status outbox.

This is an Aria gateway protocol, not a production-vendor integration. Nothing
reads environment credentials or sends during configuration. The independent
care host owns durable sending/unknown states; the fictional web app rejects it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import ssl
import threading
from typing import Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener,
)

from .care import NotificationDelivery

PROTOCOL = "aria.reminder-status.gateway.v1"
UTC = timezone.utc
MAX_RESPONSE_BYTES = 16_384
_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _identifier(value: object) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("gateway identity must be an opaque identifier")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("gateway clock must return an aware datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class GatewayRoute:
    """Operator-provisioned recipient binding; never derived from display text."""

    subject_id: str
    recipient_id: str
    recipient_channel: str
    route_id: str

    def __post_init__(self):
        for value in (self.subject_id, self.recipient_id, self.recipient_channel, self.route_id):
            _identifier(value)


@dataclass(frozen=True)
class GatewayConfig:
    endpoint: str = field(repr=False)
    bearer_token: str = field(repr=False)
    routes: tuple[GatewayRoute, ...]
    enable_network: bool = False
    timeout_seconds: float = 5.0
    receipt_key: bytes | None = field(default=None, repr=False)

    def __post_init__(self):
        if type(self.endpoint) is not str:
            raise ValueError("gateway endpoint must be an explicit HTTPS URL without credentials or query")
        try:
            endpoint = urlsplit(self.endpoint)
            valid_url = (type(self.endpoint) is str and endpoint.scheme == "https"
                         and bool(endpoint.hostname) and endpoint.username is None
                         and endpoint.password is None and not endpoint.query
                         and not endpoint.fragment and endpoint.port != 0
                         and not any(c.isspace() or ord(c) < 32 for c in self.endpoint))
        except (TypeError, ValueError):
            valid_url = False
        if not valid_url:
            raise ValueError("gateway endpoint must be an explicit HTTPS URL without credentials or query")
        if (type(self.bearer_token) is not str or not 16 <= len(self.bearer_token) <= 4096
                or not all(33 <= ord(c) <= 126 for c in self.bearer_token)):
            raise ValueError("gateway bearer credential is invalid")
        if type(self.enable_network) is not bool:
            raise TypeError("enable_network must be an explicit boolean")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 30):
            raise ValueError("gateway timeout must be finite and between 0 and 30 seconds")
        if self.receipt_key is not None and (type(self.receipt_key) is not bytes or len(self.receipt_key) < 32):
            raise ValueError("gateway receipt key must contain at least 32 bytes")
        if (type(self.routes) is not tuple or not self.routes
                or not all(type(route) is GatewayRoute for route in self.routes)):
            raise ValueError("gateway routes must be an explicit nonempty tuple")
        identities = [(r.subject_id, r.recipient_id, r.recipient_channel) for r in self.routes]
        if len(set(identities)) != len(identities) or len({r.route_id for r in self.routes}) != len(self.routes):
            raise ValueError("gateway routes must have unique recipient bindings and route identities")


@dataclass(frozen=True)
class GatewayRequest:
    endpoint: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    timeout_seconds: float


@dataclass(frozen=True)
class GatewayResponse:
    status_code: int
    body: bytes = field(default=b"", repr=False)
    redirected: bool = False


class GatewayTransport(Protocol):
    def __call__(self, request: GatewayRequest) -> GatewayResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _https_transport(request: GatewayRequest) -> GatewayResponse:
    # Explicit TLS verification, no environment proxies, redirects or retry.
    opener = build_opener(ProxyHandler({}), _NoRedirect(),
                          HTTPSHandler(context=ssl.create_default_context()))
    wire = Request(request.endpoint, data=request.body, headers=request.headers, method="POST")
    try:
        with opener.open(wire, timeout=request.timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            return GatewayResponse(response.status, body,
                                   redirected=response.geturl() != request.endpoint)
    except HTTPError as error:
        # Error bodies may contain credentials or private content. Never read them.
        status = error.code
        error.close()
        return GatewayResponse(status)


def _json_object(body: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate gateway JSON member")
            result[key] = value
        return result

    if type(body) is not bytes or not body or len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("gateway response size is invalid")
    value = json.loads(body.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("gateway response must be an object")
    return value


class HttpsNotificationGatewayAdapter:
    """Single-attempt transport; no 2xx response alone proves delivery.

    An injected transport is an explicit trusted host boundary, useful for
    offline checks. The built-in transport requires enable_network=True.
    Duplicate calls in this process reuse their result; durable interruption
    handling remains the responsibility of MinimalCareService's outbox.
    """

    channel = "https_webhook"

    def __init__(self, config: GatewayConfig, *, transport: GatewayTransport | None = None,
                 clock: Callable[[], datetime] | None = None):
        if type(config) is not GatewayConfig:
            raise TypeError("an explicit GatewayConfig is required")
        if transport is not None and not callable(transport):
            raise TypeError("gateway transport must be callable")
        self._config = config
        self._transport = transport if transport is not None else (_https_transport if config.enable_network else None)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._routes = {(r.subject_id, r.recipient_id, r.recipient_channel): r for r in config.routes}
        self._results: dict[str, tuple[str, NotificationDelivery]] = {}
        self._lock = threading.RLock()

    def send(self, *, recipient: str, message: str, severity: str, idempotency_key: str,
             subject_id: str | None = None, recipient_id: str | None = None,
             recipient_channel: str | None = None) -> NotificationDelivery:
        del recipient  # A display name has no addressing authority.
        key = _identifier(idempotency_key)
        identity = tuple(_identifier(value) for value in (subject_id, recipient_id, recipient_channel))
        route = self._routes.get(identity)
        if route is None:
            raise ValueError("no explicitly provisioned gateway recipient binding")
        if (type(message) is not str or not message.strip() or len(message) > 1000
                or type(severity) is not str or severity != "normal"):
            raise ValueError("gateway permits bounded normal reminder-status notifications only")
        request_fields = {"protocol": PROTOCOL, "scope": "reminder_status", "subject_id": subject_id,
                          "recipient_id": recipient_id, "recipient_channel": recipient_channel,
                          "route_id": route.route_id, "idempotency_key": key,
                          "message": message, "severity": severity}
        fingerprint = hashlib.sha256(_canonical(request_fields)).hexdigest()
        with self._lock:
            previous = self._results.get(key)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise ValueError("gateway idempotency identity was used for different content or recipient")
                return previous[1]
            if self._transport is None:
                return NotificationDelivery("unknown", error="gateway_network_disabled")
            if len(self._results) >= 10_000:
                return NotificationDelivery("unknown", error="gateway_attempt_limit")
            try:
                sent_at = _stamp(self._clock())
                body = _canonical({**request_fields, "sent_at": sent_at})
                request_sha = hashlib.sha256(body).hexdigest()
                request = GatewayRequest(self._config.endpoint,
                    {"Authorization": "Bearer " + self._config.bearer_token,
                     "Content-Type": "application/json; charset=utf-8", "Idempotency-Key": key},
                    body, float(self._config.timeout_seconds))
                # Cache unknown before crossing the transport boundary. Recursive
                # or duplicate calls cannot dispatch a second physical request.
                self._results[key] = (fingerprint, NotificationDelivery("unknown", error="gateway_result_unknown"))
                response = self._transport(request)
                result = self._read_receipt(response, request_fields, request_sha, sent_at)
            except Exception:
                result = NotificationDelivery("unknown", error="gateway_result_unknown")
            self._results[key] = (fingerprint, result)
            return result

    def _read_receipt(self, response, fields, request_sha, sent_at):
        if (type(response) is not GatewayResponse or type(response.status_code) is not int
                or type(response.redirected) is not bool or response.redirected
                or not 200 <= response.status_code <= 299):
            return NotificationDelivery("unknown", error="gateway_result_unknown")
        body = _json_object(response.body)
        identity = {key: fields[key] for key in ("protocol", "subject_id", "recipient_id",
                    "recipient_channel", "route_id", "idempotency_key")}
        expected = {**identity, "request_sha256": request_sha}
        status = body.get("status")
        base_keys = set(expected) | {"provider_id", "status"}
        if (status not in {"accepted", "delivered"}
                or set(body) != base_keys | ({"delivery_receipt"} if status == "delivered" else set())
                or any(type(body.get(key)) is not str or body.get(key) != value for key, value in expected.items())):
            return NotificationDelivery("unknown", error="gateway_result_unknown")
        provider_id = _identifier(body["provider_id"])
        # Even a syntactically opaque upstream id could echo a credential or
        # private ASCII text. Return only a local, request-bound reference.
        provider_reference = "gateway_" + hashlib.sha256(_canonical({
            "gateway": self._config.endpoint, "provider_id": provider_id,
            "request_sha256": request_sha,
        })).hexdigest()
        if status == "accepted":
            return NotificationDelivery("accepted", provider_id=provider_reference)
        receipt = body["delivery_receipt"]
        if not self._trusted_delivery(body, receipt, sent_at):
            return NotificationDelivery("accepted", provider_id=provider_reference,
                                        error="gateway_delivery_receipt_unverified")
        return NotificationDelivery("delivered", provider_id=provider_reference)

    def _trusted_delivery(self, body, receipt, sent_at):
        secret = self._config.receipt_key
        if (secret is None or type(receipt) is not dict
                or set(receipt) != {"status", "delivered_at", "signature"}
                or receipt.get("status") != "delivered" or type(receipt.get("signature")) is not str
                or not _SHA.fullmatch(receipt["signature"]) or type(receipt.get("delivered_at")) is not str):
            return False
        try:
            delivered_at = datetime.fromisoformat(receipt["delivered_at"])
            _stamp(delivered_at)
            start = datetime.fromisoformat(sent_at)
            end = datetime.fromisoformat(_stamp(self._clock()))
            if not start - timedelta(seconds=60) <= delivered_at <= end + timedelta(seconds=60):
                return False
            signed = {**body, "delivery_receipt": {key: value for key, value in receipt.items() if key != "signature"}}
            signature = hmac.new(secret, _canonical(signed), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, receipt["signature"])
        except (TypeError, ValueError):
            return False


def create_gateway_care_service(path: Path, *, config: GatewayConfig, allow_external: bool,
                                transport: GatewayTransport | None = None,
                                clock: Callable[[], datetime] | None = None):
    """Independent authenticated care host; no provisioning or implicit sending.

    The caller must separately provision assignments, verify each real contact,
    and record valid real reminder-status consent. Do not attach to the fictional
    MinimalWebApp, which deliberately rejects external care services.
    """
    if allow_external is not True:
        raise ValueError("real gateway hosting requires explicit allow_external=True")
    from .minimal_care import MinimalCareService

    adapter = HttpsNotificationGatewayAdapter(config, transport=transport, clock=clock)
    return MinimalCareService(path, notifier=adapter, allow_external=True, clock=clock)
