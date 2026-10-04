"""Synthetic retained bytes, zero replay dispatches, then one local online fixture."""
from copy import deepcopy
import json
import socket

import pytest

from aria.engine.policy import minimal_calls
from scripts import run_e1br_received_continuation as revision
from _reservation_fixtures import fixture_reservation_policy
from tests._e1br_entry_fixtures import synthetic_entry_root


def test_received_original_restore_then_only_new_s2_online_request(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no network"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("no network"))
    root = synthetic_entry_root(tmp_path, monkeypatch)
    entry = revision.build_engine(root)
    output = tmp_path / "c"
    monkeypatch.setattr(entry, "campaign", lambda _: output)
    policy = fixture_reservation_policy()
    for price in policy["models"].values():
        price.update(input_usd_per_million="0.01", output_usd_per_million="0.02")
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "software-fixture-only", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "local_key", lambda _: "software-fixture-no-env-read")
    registration = tmp_path / "registration.json"
    frozen = entry.prepare(root, root, registration, stage="main")
    assert frozen["protocol"]["planned_total_calls"] == 1042
    assert frozen["protocol"]["max_physical_calls"] == 3149 and frozen["protocol"]["retained_main"] == 14
    assert frozen["protocol"]["budget"]["current_remaining_usd"] == "1.018088044"
    auth = entry.authorization_template(root)
    auth.update(status="approved", budget_choice=revision.BUDGET_CHOICE)
    entry.write_once(output / "authorization.json", auth)
    phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": "main",
        "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": entry.digest(auth)}
    entry.write_once(output / "main.authorization.json", phase)
    original_guard = entry.EvidenceGuard
    class FastGuard(original_guard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clock = [0.0]
            self._pacing_clock = lambda: clock[0]
            self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    monkeypatch.setattr(entry, "EvidenceGuard", FastGuard)
    sent, reached = [], []
    decision = {"response": "本地替身，仅验证实际接线。", "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": .3,
            "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
        "selected_intervention_id": "", "uncertainty": .6,
        "communication_plan": {"pace": "open_listening", "support_order": "先听本人说", "boundary": "不强加话题"}}
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
        return Response({"model": payload["model"], "provider": "Alibaba",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(decision) if payload.get("response_format") else decision["response"]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": .000001,
                      "completion_tokens_details": {"reasoning_tokens": 0}}})
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    base_runner = entry.E1bRunner
    class SegmentRunner(base_runner):
        def _operation(self, app, name, operation, *, detail=None):
            if name == "submit" and detail.get("phase") == "s2" and detail.get("turn") == 1:
                reached.append({"state": app.call_guard.snapshot(), "profile": app._profile().model_dump(mode="json")})
                raise RuntimeError("software_segment_boundary_after_s2_t0")
            return super()._operation(app, name, operation, detail=detail)
    monkeypatch.setattr(entry, "E1bRunner", SegmentRunner)
    result = entry.execute(root, root, registration, allow_local_env=True, budget_choice=revision.BUDGET_CHOICE,
        campaign_authorization_sha256=entry.digest(auth), stage_authorization_sha256=entry.digest(phase))
    assert result["status"] == "not_evaluable"  # intentional one-online fixture
    assert len(sent) == len(reached) == 1
    state = reached[0]["state"]
    assert not state["stopped"] and state["calls_used"] == 1 and state["unknown_cost_requests"] == 0
    row = state["attempts"][0]
    assert row["usage"]["reasoning_tokens"] == 0 and row["status"] == "received"
    assert row["experiment_position"] == ["online_fast", "e1b_a1", 2, "shared", 0, 0]
    assert row["resume_after_received_request_id"] == revision.RECEIVED_REQUEST
    folder = output / "main/synthetic-emotion-experiments/e1b-main-v1/run/main/e1b_a1"
    captured = entry.read_json(folder / "reflection-s1.first.json")
    product = entry.read_json(folder / "reflection-s1.product.json")
    assert captured["request_id"] == revision.RECEIVED_REQUEST and captured["receipt_status"] == "received"
    assert captured["visible_response_sha256"] == entry.received_reflection["receipt"]["client_return_content_sha256"]
    assert product["product_status"] == "failed" and product["format_correction_count"] == 0 and product["capture_error"] is None
    assert reached[0]["profile"]["offline_sessions"] == []
    assert reached[0]["profile"]["offline_session_audit"][-1]["outcome"] == "reflection_failed"
    assert (folder / "memory-s1.first.json").read_bytes() == entry.retained_prefix["originals"]["memory-s1.first.json"]
    assert entry.export_protocol(frozen)["secondary_positions"] == 0
    assert revision.parent_originals(root)
    assert all(path.is_relative_to(root) for path in revision.parent_originals(root))
