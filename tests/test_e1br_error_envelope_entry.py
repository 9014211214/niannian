"""Real formal entry/restore/capture/accounting with a synthetic HTTP429 body."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import io
import json
import socket
import urllib.error

import pytest

from aria.engine.policy import minimal_calls
from scripts import run_e1br_received_continuation as revision
from _reservation_fixtures import fixture_reservation_policy
from tests._e1br_entry_fixtures import synthetic_entry_root


def test_actual_entry_http429_metadata_to_saved_retry_and_first_body(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("no network"))
    root = synthetic_entry_root(tmp_path, monkeypatch)
    entry = revision.build_engine(root)
    output = tmp_path / "new-campaign"
    monkeypatch.setattr(entry, "campaign", lambda _: output)
    policy = fixture_reservation_policy()
    for price in policy["models"].values():
        price.update(input_usd_per_million="0.01", output_usd_per_million="0.02")
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "software-fixture-only", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "local_key", lambda _: "software-fixture-no-env-read")
    registration = tmp_path / "registration.json"
    frozen = entry.prepare(root, root, registration, stage="main")
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=revision.BUDGET_CHOICE)
    entry.write_once(output / "authorization.json", auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": "main",
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": entry.digest(auth)}
    entry.write_once(output / "main.authorization.json", phase)
    base_guard = entry.EvidenceGuard

    class ClockGuard(base_guard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clock = [0.0]
            self._pacing_clock = lambda: clock[0]
            self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)

    monkeypatch.setattr(entry, "EvidenceGuard", ClockGuard)
    sent, terminal = [], []
    decision = {"response": "本地替身，仅验证错误包接线。", "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": .3,
            "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
        "selected_intervention_id": "", "uncertainty": .6,
        "communication_plan": {"pace": "open_listening", "support_order": "先听本人说", "boundary": "不强加话题"}}
    raw_error = b'{"error":{"code":429,"metadata":{"provider_name":"Alibaba"}},"user_id":"user_fixture_private"}'

    class Response:
        status, headers = 200, {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return json.dumps(self.value).encode()[:size]

    def opened(request, **kwargs):
        payload = json.loads(request.data)
        sent.append(deepcopy(payload))
        assert payload["model"] == frozen["protocol"]["models"]["online_fast"]
        if len(sent) == 1:
            raise urllib.error.HTTPError("https://fixture.invalid", 429, "synthetic limit", {}, io.BytesIO(raw_error))
        assert len(sent) == 2
        return Response({"model": payload["model"], "provider": "Alibaba",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content":
                json.dumps(decision) if payload.get("response_format") else decision["response"]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": .000001,
                      "completion_tokens_details": {"reasoning_tokens": 0}}})

    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    base_runner = entry.E1bRunner

    class SegmentRunner(base_runner):
        def _operation(self, app, name, operation, *, detail=None):
            if name == "submit" and detail.get("phase") == "s2" and detail.get("turn") == 1:
                terminal.append(app.call_guard.snapshot())
                raise RuntimeError("intentional_software_segment_end")
            return super()._operation(app, name, operation, detail=detail)

    monkeypatch.setattr(entry, "E1bRunner", SegmentRunner)
    result = entry.execute(root, root, registration, allow_local_env=True, budget_choice=revision.BUDGET_CHOICE,
        campaign_authorization_sha256=entry.digest(auth), stage_authorization_sha256=entry.digest(phase))
    assert result["status"] == "not_evaluable"  # Intentional stop after one logical online reply.
    assert len(sent) == 2 and sent[0] == sent[1] and len(terminal) == 1
    state = terminal[0]
    assert not state["stopped"] and state["calls_used"] == 2 and state["unknown_cost_requests"] == 1
    first, second = state["attempts"]
    assert first["status"] == "failed" and second["status"] == "received"
    assert first["experiment_position"] == second["experiment_position"] == ["online_fast", "e1b_a1", 2, "shared", 0, 0]
    assert first["resume_after_received_request_id"] == revision.RECEIVED_REQUEST
    assert second["infrastructure_retry_of_request_id"] == first["request_id"]
    assert second["infrastructure_retry_index"] == 1
    audit = first["error_envelope_audit"]
    assert audit["rejection_reasons"] == [] and audit["read_complete"]
    assert audit["captured_sha256"] == hashlib.sha256(raw_error).hexdigest()
    assert audit["user_id_status"] == "bounded_identifier"
    assert "user_fixture_private" not in json.dumps(state)
    assert not (output / "main/visible-first/0001.json").exists()
    captured = entry.read_json(output / "main/visible-first/0002.json")
    assert captured["request_id"] == second["request_id"]
    assert Decimal(state["reserved_cost_usd"]) == Decimal(first["reservation"]["reserved_cost_usd"])
    assert entry.read_json(output / "main/execution-receipt.json")["accounting"]["attempts"] == state["attempts"]
