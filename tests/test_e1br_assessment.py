"""Closed real guard ledger compatibility, using software HTTP and no network."""
from contextlib import closing
import json
from pathlib import Path

from aria.engine.policy import minimal_calls
from aria.engine.policy.e1br_calls import validate_e1br_ledger
from aria.engine.policy.observed_transport import validate_trace
from aria.engine.policy.reserved_calls import validate_reservation_policy
from aria.engine.model.persona import PersonaProfile
from scripts import e1b_assessment as assessment
from tests.test_e1br_calls import guard, reflection, http_error, body
from tests.test_reflection_evidence_review import context
from tests.test_reserved_calls import policy


def test_closed_ledger_keeps_unknown_holds_and_format_correction(tmp_path, monkeypatch):
    base = tmp_path / "c"
    base.mkdir()
    path = base / "allowance.sqlite3"
    with closing(guard(path)) as g:
        client, updater = reflection(g)
        good = {"protocol_version": "2.0", "reviews": [], "changes": []}
        sequence = iter([http_error(503), body(client, {**good, "reviews": "bad"}),
                         http_error(503), body(client, good)])
        class Response:
            status, headers = 200, {}
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return json.dumps(self.value).encode()[:size]
        def opened(request, *, trace, **kwargs):
            ready = {"connection_preparation_started", "connection_preparation_completed", "tls_completed"}
            trace._preparations.append({"ordinal": 1, "events": dict.fromkeys(ready, True), "elapsed_ms": 0})
            for event in ready | {"application_request_started", "application_write_attempted", "request_write_completed", "response_headers_returned"}:
                trace._mark(event)
            value = next(sequence)
            if isinstance(value, Exception): raise value
            return Response(value)
        monkeypatch.setattr(minimal_calls, "open_observed", opened)
        with g.scope("review", "review1", "session1"):
            updater.update(PersonaProfile(user_id="person"), context())
        state = g.snapshot()
        assert updater.last_status == "ok" and not state["stopped"]
        assert state["unknown_cost_requests"] == 2 and len(state["attempts"]) == 4
    entry = assessment.execution
    monkeypatch.setattr(entry, "CAMPAIGN", Path("c"))
    monkeypatch.setattr(entry, "MAX_WIRE_REQUESTS", 3153)
    monkeypatch.setattr(entry, "MIN_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(entry, "OBSERVE_TRANSPORT", True, raising=False)
    monkeypatch.setattr(entry, "TRANSPORT_VERSION", "connection_trace_v2", raising=False)
    monkeypatch.setattr(entry, "CONNECTION_PREPARATION_RECOVERY", True, raising=False)
    monkeypatch.setattr(entry, "validate_transport_records", lambda rows, protocol: [
        validate_trace(row["transport_events"], require_complete=row["status"] == "received", require_https=True) for row in rows], raising=False)
    frozen = {"metadata": {"policy": validate_reservation_policy(policy())}, "protocol": {"transport": {
        "min_interval_seconds": 10, "connection_preparation_retries": 1}}}
    before = path.read_bytes()
    assert assessment._ledger(tmp_path, state, frozen) == entry.byte_sha(path)
    assert validate_e1br_ledger(path, stage="main", registration_sha256="b" * 64,
        max_requests=3153, reservation_policy=policy()) == state["attempts"]
    assert path.read_bytes() == before and list(base.iterdir()) == [path]
