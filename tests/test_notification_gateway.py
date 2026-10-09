"""Injected software gateway checks; no real contacts, network or delivery."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json

import pytest

from aria.engine.minimal_care import MinimalCareService
from aria.engine.notification_gateway import (
    GatewayConfig, GatewayRequest, GatewayResponse, GatewayRoute,
    HttpsNotificationGatewayAdapter, PROTOCOL, create_gateway_care_service,
)

UTC = timezone.utc
START = datetime(2026, 10, 9, 7, tzinfo=UTC)
TOKEN = "gateway-fixture-token-not-deployable"  # release-fixture: not-a-deployment-secret
RECEIPT_KEY = b"offline-fixture-receipt-signing-key"  # release-fixture: not-a-deployment-secret
PASSWORD = "offline-gateway-fixture-password"  # release-fixture: not-a-deployment-secret
ROUTE = GatewayRoute("elder_a", "contact_a", "sms", "route_a")
PARAMS = dict(recipient="姓名不能作为地址", message="提醒未确认，请按约定核查。", severity="normal",
              idempotency_key="delivery_a", subject_id="elder_a", recipient_id="contact_a", recipient_channel="sms")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def config(**changes):
    return GatewayConfig(**{"endpoint": "https://gateway.example.invalid/notify", "bearer_token": TOKEN,
                           "routes": (ROUTE,), "receipt_key": RECEIPT_KEY, **changes})


def response_for(request, *, status="accepted", signed=True, **changes):
    wire = json.loads(request.body)
    response = {key: wire[key] for key in ("protocol", "subject_id", "recipient_id", "recipient_channel",
                "route_id", "idempotency_key")}
    response.update(request_sha256=hashlib.sha256(request.body).hexdigest(), provider_id="provider_a", status=status)
    response.update(changes)
    if status == "delivered":
        response["delivery_receipt"] = {"status": "delivered", "delivered_at": wire["sent_at"]}
        signature = hmac.new(RECEIPT_KEY, canonical(response), hashlib.sha256).hexdigest() if signed else "0" * 64
        response["delivery_receipt"]["signature"] = signature
    return GatewayResponse(200, canonical(response))


class Transport:
    def __init__(self, outcome=None):
        self.requests = []
        self.outcome = outcome or response_for

    def __call__(self, request):
        self.requests.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome(request)


def adapter(transport, **changes):
    return HttpsNotificationGatewayAdapter(config(**changes), transport=transport, clock=lambda: START)


def test_default_disabled_ignores_environment_and_does_not_send(monkeypatch):
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_URL", "https://gateway.example.invalid/secret")
    monkeypatch.setenv("ARIA_CARE_WEBHOOK_TOKEN", TOKEN)
    import aria.engine.notification_gateway as gateway
    monkeypatch.setattr(gateway, "_https_transport", lambda _request: pytest.fail("network forbidden"))
    service = HttpsNotificationGatewayAdapter(config(), clock=lambda: START)
    assert service.send(**PARAMS).status == "unknown"
    assert service.send(**PARAMS).error == "gateway_network_disabled"


def test_injected_accepted_is_not_delivered_and_wire_excludes_display_name():
    transport = Transport()
    result = adapter(transport).send(**PARAMS)
    assert result.status == "accepted" and result.provider_id.startswith("gateway_")
    assert "provider_a" not in result.provider_id
    request = transport.requests[0]
    wire = json.loads(request.body)
    assert wire == {"protocol": PROTOCOL, "scope": "reminder_status", "subject_id": "elder_a",
                    "recipient_id": "contact_a", "recipient_channel": "sms", "route_id": "route_a",
                    "idempotency_key": "delivery_a", "message": PARAMS["message"], "severity": "normal",
                    "sent_at": "2026-10-09T07:00:00+00:00"}
    assert PARAMS["recipient"] not in request.body.decode()
    assert request.headers["Authorization"] == "Bearer " + TOKEN
    assert request.headers["Idempotency-Key"] == wire["idempotency_key"]


def test_only_signed_current_identity_receipt_is_delivered():
    transport = Transport(lambda request: response_for(request, status="delivered"))
    assert adapter(transport).send(**PARAMS).status == "delivered"


@pytest.mark.parametrize("changes", [{"receipt_key": None}, {"receipt_key": b"different-key-not-deployable-12345"}])
def test_untrusted_delivery_receipt_is_only_accepted(changes):
    transport = Transport(lambda request: response_for(request, status="delivered"))
    result = adapter(transport, **changes).send(**PARAMS)
    assert result.status == "accepted" and result.error == "gateway_delivery_receipt_unverified"


def test_invalid_signature_does_not_prove_delivery():
    result = adapter(Transport(lambda request: response_for(request, status="delivered", signed=False))).send(**PARAMS)
    assert result.status == "accepted"


@pytest.mark.parametrize("field,value", [("subject_id", "elder_b"), ("recipient_id", "contact_b"),
    ("recipient_channel", "email"), ("route_id", "route_b"), ("idempotency_key", "delivery_b"),
    ("request_sha256", "0" * 64), ("protocol", "different.v1")])
def test_cross_identity_gateway_receipt_is_unknown(field, value):
    transport = Transport(lambda request: response_for(request, status="delivered", **{field: value}))
    assert adapter(transport).send(**PARAMS).status == "unknown"


@pytest.mark.parametrize("changes", [{"subject_id": "elder_b"}, {"recipient_id": "contact_b"},
    {"recipient_channel": "email"}, {"subject_id": None}, {"recipient_id": True},
    {"recipient_channel": "private@example.invalid"}])
def test_unprovisioned_owner_contact_or_channel_never_reaches_transport(changes):
    transport = Transport()
    with pytest.raises(ValueError):
        adapter(transport).send(**{**PARAMS, **changes})
    assert transport.requests == []


@pytest.mark.parametrize("outcome", [TimeoutError("secret token " + TOKEN), OSError(PARAMS["message"]),
    GatewayResponse(302, b"private response"), GatewayResponse(307), GatewayResponse(400), GatewayResponse(500),
    GatewayResponse(200, b"not json private response"), GatewayResponse(204), GatewayResponse(True),
    GatewayResponse(200, b"{}", redirected=True), GatewayResponse(200, b"x" * 16_385)])
def test_uncertain_results_are_unknown_and_no_duplicate_dispatch(outcome):
    transport = Transport(outcome if isinstance(outcome, BaseException) else lambda _request: outcome)
    current = adapter(transport)
    first = current.send(**PARAMS)
    assert first.status == "unknown"
    assert current.send(**PARAMS) == first
    assert len(transport.requests) == 1
    assert TOKEN not in repr(first) and PARAMS["message"] not in repr(first) and "private response" not in repr(first)


def test_same_id_different_owner_or_content_is_rejected_without_send():
    transport = Transport()
    routes = (ROUTE, GatewayRoute("elder_b", "contact_b", "sms", "route_b"))
    current = adapter(transport, routes=routes)
    current.send(**PARAMS)
    for changes in ({"message": "另一条提醒"}, {"subject_id": "elder_b", "recipient_id": "contact_b"}):
        with pytest.raises(ValueError, match="idempotency"):
            current.send(**{**PARAMS, **changes})
    assert len(transport.requests) == 1


def test_concurrent_duplicate_single_physical_request():
    transport = Transport()
    current = adapter(transport)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _item: current.send(**PARAMS), range(10)))
    assert all(result.status == "accepted" for result in results)
    assert len(transport.requests) == 1


@pytest.mark.parametrize("endpoint", ["http://gateway.example.invalid", "https://" + "user:password" + "@example.invalid/",
    "https://gateway.example.invalid/path?credential=secret", "https://gateway.example.invalid/#secret",
    "https://gateway.example.invalid/\nprivate", "https://gateway.example.invalid:0", "https:///missing", 123])
def test_invalid_endpoint_rejected_without_echo(endpoint):
    with pytest.raises((ValueError, TypeError)) as error:
        config(endpoint=endpoint)
    assert "password" not in str(error.value) and "credential=secret" not in str(error.value)


@pytest.mark.parametrize("changes", [{"enable_network": 1}, {"timeout_seconds": True}, {"timeout_seconds": 0},
    {"timeout_seconds": 31}, {"timeout_seconds": float("nan")}, {"receipt_key": b"short"},
    {"bearer_token": "short"}, {"bearer_token": "long-invalid\r\nheader"}, {"routes": []},
    {"routes": (ROUTE, ROUTE)}, {"routes": (ROUTE, GatewayRoute("elder_b", "contact_b", "sms", "route_a"))}])
def test_invalid_explicit_config(changes):
    with pytest.raises((ValueError, TypeError)):
        config(**changes)


def test_repr_does_not_disclose_credentials_private_content_or_raw_response():
    current = config()
    request = GatewayRequest(current.endpoint, {"Authorization": TOKEN}, PARAMS["message"].encode(), 5)
    response = GatewayResponse(200, b"raw private response")
    assert TOKEN not in repr(current) + repr(request) + repr(response)
    assert PARAMS["message"] not in repr(request)
    assert "raw private response" not in repr(response)
    assert RECEIPT_KEY.decode() not in repr(current)


@pytest.mark.parametrize("echoed", [TOKEN, "private_ascii_message"])
def test_allowed_shape_provider_id_cannot_echo_credentials_or_private_message(echoed):
    transport = Transport(lambda request: response_for(request, provider_id=echoed))
    result = adapter(transport).send(**{**PARAMS, "message": "private_ascii_message"})
    assert result.status == "accepted"
    assert echoed not in repr(result)
    assert result.provider_id.startswith("gateway_") and len(result.provider_id) == 72


def test_duplicate_json_members_and_unrequested_fields_are_unknown():
    def extra(request):
        body = json.loads(response_for(request).body)
        body["message"] = "private echoed content"
        return GatewayResponse(200, canonical(body))
    for transport in (Transport(lambda _request: GatewayResponse(200, b'{"status":"accepted","status":"delivered"}')),
                      Transport(extra)):
        assert adapter(transport).send(**PARAMS).status == "unknown"


def test_signed_stale_or_future_timestamp_is_only_accepted():
    def outdated(request):
        body = json.loads(response_for(request, status="delivered").body)
        body["delivery_receipt"].pop("signature")
        body["delivery_receipt"]["delivered_at"] = "2026-10-08T07:00:00+00:00"
        body["delivery_receipt"]["signature"] = hmac.new(RECEIPT_KEY, canonical(body), hashlib.sha256).hexdigest()
        return GatewayResponse(200, canonical(body))
    assert adapter(Transport(outdated)).send(**PARAMS).status == "accepted"


def test_native_transport_disables_redirects_proxies_and_does_not_read_error_body(monkeypatch):
    import aria.engine.notification_gateway as gateway
    from urllib.error import HTTPError
    from urllib.request import HTTPSHandler, ProxyHandler
    handlers = []
    calls = []

    class PrivateBody:
        def read(self, *args):
            pytest.fail("private error body must not be read")

        def close(self):
            pass

    class Opener:
        def open(self, request, *, timeout):
            calls.append((request, timeout))
            raise HTTPError(request.full_url, 302, "private", {}, PrivateBody())

    def opener(*values):
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(gateway, "build_opener", opener)
    current = HttpsNotificationGatewayAdapter(config(enable_network=True), clock=lambda: START)
    assert current.send(**PARAMS).status == "unknown"
    assert len(calls) == 1
    assert any(isinstance(handler, ProxyHandler) and handler.proxies == {} for handler in handlers)
    assert any(isinstance(handler, HTTPSHandler) for handler in handlers)
    redirect = next(handler for handler in handlers if isinstance(handler, gateway._NoRedirect))
    assert redirect.redirect_request(None, None, 302, None, None, "https://other.example.invalid") is None


def provision(path, clock):
    # Entirely fictional provisioning software seam, never real verification.
    service = MinimalCareService(path, allow_external=True, clock=clock)
    service.bootstrap("carer", PASSWORD, ["elder_a"])
    token = service.login("carer", PASSWORD)
    elder = lambda action, request_id, **values: service.elder_action("elder_a", action,
        {"request_id": request_id, **values})["record"]
    reminder = elder("reminder.create", "rem", label="私密标签不进入通知", schedule="once", date="2026-10-09",
                     time="15:01", timezone="Asia/Shanghai", grace_minutes=1)
    contact = elder("contact.create", "contact", name="虚构护理对象", channel="sms")
    service.mutate(token, "elder_a", "contact.verify", {"request_id": "verify", "contact_id": contact["id"],
                     "evidence_source": "injected fictional software verification"})
    consent = elder("consent.grant", "consent", contact_id=contact["id"], reminder_id=reminder["id"],
                    trigger="unconfirmed", scope="reminder_status", source="real", confirmed=True,
                    expires_at="2026-10-10T07:00:00+00:00")
    service.close()
    return reminder, contact, consent


@pytest.mark.parametrize("response_status", ["accepted", "delivered", "unknown"])
def test_independent_host_outbox_bound_route_and_never_retries_on_restart(tmp_path, response_status):
    now = [START]
    clock = lambda: now[0]
    path = tmp_path / "care.sqlite"
    _reminder, contact, _consent = provision(path, clock)
    route = GatewayRoute("elder_a", contact["id"], "sms", "gateway_route_a")
    cfg = config(routes=(route,))
    transport = Transport(TimeoutError("private") if response_status == "unknown" else
                          lambda request: response_for(request, status=response_status))
    service = create_gateway_care_service(path, config=cfg, allow_external=True, transport=transport, clock=clock)
    assert service.external and service.notifier.channel == "https_webhook"
    now[0] += timedelta(minutes=1)
    assert service.tick()["processed"] == 0
    occurrence = service.elder_view("elder_a")["occurrences"][0]
    service.elder_action("elder_a", "occurrence.displayed", {"request_id": "shown",
                        "occurrence_id": occurrence["id"], "generation": occurrence["generation"]})
    now[0] += timedelta(minutes=1)
    assert service.tick()["processed"] == 1
    row = service.elder_view("elder_a")["deliveries"][0]
    assert row["status"] == response_status
    assert json.loads(transport.requests[0].body)["recipient_id"] == contact["id"]
    assert "私密标签" not in transport.requests[0].body.decode()
    service.tick()
    service.close()
    restored = create_gateway_care_service(path, config=cfg, allow_external=True, transport=transport, clock=clock)
    try:
        restored.tick()
        assert len(transport.requests) == 1
        assert restored.elder_view("elder_a")["deliveries"][0]["status"] == response_status
        with pytest.raises(ValueError, match="真实外发"):
            restored.forget_elder("elder_a")
    finally:
        restored.close()


def test_independent_host_requires_explicit_permission_and_fictional_web_rejects(tmp_path):
    for value in (False, 1, None):
        with pytest.raises(ValueError):
            create_gateway_care_service(tmp_path / "care.sqlite", config=config(), allow_external=value)
    service = create_gateway_care_service(tmp_path / "care.sqlite", config=config(), allow_external=True)
    try:
        from aria.voice.minimal import MinimalWebApp
        with pytest.raises(ValueError, match="虚构Web"):
            MinimalWebApp(tmp_path / "web", care_service=service)
    finally:
        service.close()
