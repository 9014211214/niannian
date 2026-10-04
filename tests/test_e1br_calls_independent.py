"""Independent refusal/unknown/continuation checks: local HTTP fixtures only."""
from contextlib import closing
from hashlib import sha256
import io
import json
import urllib.error

import pytest

from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.e1br_calls import E1BRCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reflection_json import reflection_body_system_prompt
from aria.runtime import create_minimal_clients
from tests.test_reflection_evidence_review import context
from tests.test_reserved_calls import policy, receipt


class Clock:
    def __init__(self): self.value = 100.0
    def now(self): return self.value
    def sleep(self, seconds): self.value += seconds


def configured(tmp_path, monkeypatch, role, values, authority=None):
    client = getattr(create_minimal_clients(model_calls_enabled=True, api_key="local-fixture"), role)
    clock = Clock()
    control = E1BRCallGuard(tmp_path / "ledger.sqlite", max_requests=12,
                           max_observed_cost_usd="10", reservation_policy=policy(),
                           registration_sha256="a" * 64, authority_check=authority or (lambda: None))
    control._pacing_clock, control._pacing_sleep = clock.now, clock.sleep
    control.attach(client)
    sequence, calls = iter(values), []
    class Response:
        status = 200
        headers = {}
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size=-1):
            if isinstance(self.data, Exception): raise self.data
            return json.dumps(self.data).encode()[:size]
    def opened(request, **kwargs):
        calls.append((clock.now(), json.loads(request.data)))
        value = next(sequence)
        if callable(value): value = value(client)
        if isinstance(value, Exception): raise value
        return Response(value)
    monkeypatch.setattr("aria.engine.policy.minimal_calls.open_observed", opened)
    return client, control, calls, clock


def body(content):
    def value(client):
        result = receipt(client)
        result["choices"][0]["message"]["content"] = content
        return result
    return value


def messages():
    return [{"role": "system", "content": reflection_body_system_prompt()},
            {"role": "user", "content": "original local source"}]


def http_error(raw, status=500, headers=None):
    return urllib.error.HTTPError("https://fixture.invalid", status, "local fixture", headers or {}, io.BytesIO(raw))


def test_reflection_return_hash_binds_actual_unstripped_body(tmp_path, monkeypatch):
    raw = ' \n{"protocol_version":"2.0","reviews":[],"changes":[]}\n '
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [body(raw)])
    with closing(guard), guard.scope("review", "r1", "s1"):
        result = client.complete(messages())
        row, = guard.snapshot()["attempts"]
        assert result == raw
        assert row["client_return_content_sha256"] == sha256(result.encode()).hexdigest()
    assert len(calls) == 1


def test_initial_and_correction_share_two_infrastructure_retries_and_all_holds(tmp_path, monkeypatch):
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [
        TimeoutError("local timeout"), body('{"reviews": []}'), http_error(b""), http_error(b""), body("must never send")])
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    with closing(guard), guard.scope("review", "r1", "s1"):
        updater.update(PersonaProfile(user_id="person"), context())
        state = guard.snapshot()
        assert len(calls) == state["calls_used"] == 4
        assert state["e1br"]["infrastructure_retries_used"] == 2
        assert state["unknown_cost_requests"] == 3
        assert float(state["reserved_cost_usd"]) > 0
        assert updater.last_status == "failed" and updater.last_format_correction_count == 1
        assert guard.reflection_failure_is_terminal(client)
        assert not state["stopped"]
        assert [row["product_step"] for row in state["attempts"]] == ["initial", "initial", "format_correction", "format_correction"]


@pytest.mark.parametrize("role", ["memory", "reflection"])
def test_received_body_with_unknown_cost_is_never_infrastructure_retry(tmp_path, monkeypatch, role):
    def unknown(client):
        value = body('{"protocol_version":"2.0","reviews":[],"changes":[]}')(client)
        del value["usage"]["cost"]
        return value
    client, guard, calls, _ = configured(tmp_path, monkeypatch, role, [unknown, body("unapproved second")])
    with closing(guard), guard.scope("review", "r1", "s1"):
        with pytest.raises(MinimalCallStopped):
            client.complete(messages() if role == "reflection" else [{"role": "user", "content": "local input"}])
        state = guard.snapshot()
        assert len(calls) == 1 and state["unknown_cost_requests"] == 1 and state["stopped"]
        assert not guard.reflection_failure_is_terminal(client)


@pytest.mark.parametrize("raw", [b'{"error":', b'{"error":{"code":500},"choices":[{"message":{"content":"body"}}]}',
                                  b'{"error":{"code":500},"error":{"code":500}}'])
def test_incomplete_mixed_or_duplicate_error_does_not_prove_no_content(tmp_path, monkeypatch, raw):
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [http_error(raw), body("unapproved second")])
    with closing(guard), guard.scope("review", "r1", "s1"):
        with pytest.raises(MinimalCallStopped): client.complete(messages())
        assert len(calls) == 1 and guard.snapshot()["stopped"]
        assert not guard.reflection_failure_is_terminal(client)


def test_fresh_preflight_denial_does_not_reuse_secondary_terminal_marker(tmp_path, monkeypatch):
    enabled = [True]
    def authority():
        if not enabled[0]: raise MinimalCallStopped("local_authority_denied")
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection",
        [http_error(b""), http_error(b""), http_error(b"")], authority)
    with closing(guard):
        with guard.scope("review", "r1", "s1"):
            with pytest.raises(MinimalCallStopped): client.complete(messages())
            assert guard.reflection_failure_is_terminal(client)
        enabled[0] = False
        with guard.scope("review", "r2", "s2"):
            with pytest.raises(MinimalCallStopped): client.complete(messages())
            assert not guard.reflection_failure_is_terminal(client)
        assert len(calls) == 3 and guard.snapshot()["calls_used"] == 3


def test_retry_after_positive_hint_is_respected_without_extra_retries(tmp_path, monkeypatch):
    error = http_error(b'{"error":{"code":429}}', 429, {"Retry-After": "60"})
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "memory", [error, body("local answer")])
    with closing(guard), guard.scope("review", "r1", "s1"):
        assert client.complete([{"role": "user", "content": "local input"}]) == "local answer"
        assert guard.snapshot()["e1br"]["infrastructure_retries_used"] == 1
    assert len(calls) == 2 and calls[1][0] - calls[0][0] >= 60


@pytest.mark.parametrize("mutation", ["remove_reservation", "zero_unknown_hold"])
def test_retained_unknown_hold_cannot_be_removed_or_repriced_in_ledger(tmp_path, monkeypatch, mutation):
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [http_error(b"") for _ in range(3)])
    with closing(guard), guard.scope("review", "r1", "s1"):
        with pytest.raises(MinimalCallStopped): client.complete(messages())
        state = guard.snapshot()
        assert not state["stopped"] and state["unknown_cost_requests"] == 3
        row = state["attempts"][0]
        if mutation == "remove_reservation": row.pop("reservation")
        else: row["reservation"]["reserved_cost_usd"] = "0"
        guard._db.execute("UPDATE attempts SET record=? WHERE request_id=?", (json.dumps(row), row["request_id"]))
        assert guard.snapshot()["stopped"], "retained unknown hold was silently dropped"


@pytest.mark.parametrize("raw,status", [
    (b'{"error":{"code":401}}', 503),
    (b'{"error":{"code":403}}', 429),
    (b'{"error":{"code":429},"usage":{"cost":NaN}}', 429),
    (b'{"error":{"code":503},"usage":{"cost":Infinity}}', 503),
    (b'{"error":{"code":503},"usage":{"cost":9,"prompt_tokens":1,"completion_tokens":1}}', 503),
])
def test_auth_invalid_billing_or_overcharge_cannot_hide_in_retryable_status(tmp_path, monkeypatch, raw, status):
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [http_error(raw, status), body("unapproved next")])
    with closing(guard), guard.scope("review", "r1", "s1"):
        with pytest.raises(MinimalCallStopped): client.complete(messages())
        state = guard.snapshot()
        assert len(calls) == 1 and state["stopped"]
        assert not guard.reflection_failure_is_terminal(client)
        if b'"cost":9' in raw:
            assert state["known_cost_usd"] == "9"


def test_metered_client_identity_survives_capture_proxy_at_failed_queue_boundary(tmp_path, monkeypatch):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.runtime import MinimalModelClients, create_minimal_engine
    from aria.scenes.registry import get_scene
    from tests.test_model_routing import _DECISION_OK, _ProtocolAwareFast
    from tests.test_reflection_body_json import Replies
    client, guard, calls, _ = configured(tmp_path, monkeypatch, "reflection", [http_error(b"") for _ in range(3)])
    memory = Replies([{"episodes": [], "relationships": []}])
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="person"),
        clients=MinimalModelClients(_ProtocolAwareFast(_DECISION_OK), memory, client, True),
        store=PersonaStore(tmp_path / "profile"))
    class Proxy:
        def complete(self, messages): return client.complete(messages)
        def __getattr__(self, name): return getattr(client, name)
    engine.hypothesis_updater.reflection_metered_client = client
    engine.hypothesis_updater.llm = Proxy()
    engine.seal_session(context())
    with closing(guard), guard.scope("review", "r1", "session_1"):
        result = engine.recover_offline_sessions(max_jobs=1)
        assert result["last_run"]["completed_jobs"] == 1 and result["pending_count"] == 0
        assert engine.hypothesis_updater.last_status == "failed"
        assert engine.persona.offline_session_audit[-1].outcome == "reflection_failed"
        assert engine.persona.offline_session_audit[-1].memory_completed
        assert not engine.persona.offline_session_audit[-1].reflection_completed
        assert not engine.persona.person_hypotheses
        assert guard.snapshot()["unknown_cost_requests"] == 3
    assert len(calls) == 3 and len(memory.calls) == 1
