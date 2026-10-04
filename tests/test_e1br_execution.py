"""Representative formal-entry segments with an injected, network-free transport.

Real fictional prefix originals are read-only. All new ledgers and authorization
fixtures live in pytest's temporary directory; this is no model effect result.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from scripts import run_e1br as revision
from aria.engine.policy import minimal_calls
from tests.test_e1br_calls import http_error


def test_exact_logical_inventory_and_rebinding_rejection():
    rows = revision.position_inventory()
    assert len(rows) == 1043
    assert Counter(row[0] for row in rows) == Counter(revision.NEW_COUNTS["main"])
    key = revision.position_key("reflection", "reflect", {"case_id": "e1b_a1", "session": 1})
    scope = {"operation_id": "e1b-review-s", "session_id": "s"}
    old = {**scope, "role": "reflection", "experiment_position": list(key)}
    revision.validate_position_binding(key, scope, [old])
    with pytest.raises(revision.RunError, match="rebound"):
        revision.validate_position_binding(key, {"operation_id": "e1b-review-t", "session_id": "t"}, [old])
    for detail in ({"case_id": "e1b_a3", "session": 1}, {"case_id": "e1b_a1", "session": 3}):
        with pytest.raises(revision.RunError):
            revision.position_key("reflection", "reflect", detail)
    with pytest.raises(revision.RunError):
        revision.position_key("memory", "reflect", {"case_id": "e1b_a1", "session": 1})


@pytest.mark.parametrize("kind,physical", [("format_correction", 2), ("exhausted_reflection", 3), ("truncated_content", 1)])
def test_formal_retained_prefix_first_reflection_then_next_session(tmp_path, monkeypatch, kind, physical):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("network forbidden"))
    entry = revision.build_engine(root)
    output = tmp_path / "c"
    monkeypatch.setattr(entry, "campaign", lambda _: output)
    # Frozen software fixture metadata/source are explicitly distinct; ancestor
    # SHA validation, retained restore, entry policy and auth hashes still run.
    from _reservation_fixtures import fixture_reservation_policy
    policy = fixture_reservation_policy()
    for price in policy["models"].values():
        price.update(input_usd_per_million="0.01", output_usd_per_million="0.02")
    monkeypatch.setattr(entry, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    monkeypatch.setattr(entry, "inventory", lambda _: [{"path": "software-fixture-only", "sha256": "a" * 64}])
    monkeypatch.setattr(entry, "local_key", lambda _: "software-fixture-placeholder-no-env-read")
    registration = tmp_path / "registration.json"
    frozen = entry.prepare(root, root, registration, stage="main")
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
    class Response:
        status, headers = 200, {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return json.dumps(self.value).encode()[:size]
    def opened(request, **kwargs):
        payload = json.loads(request.data)
        sent.append(deepcopy(payload))
        assert payload["messages"][1] == entry.retained_prefix["failed_payload"]["messages"][1]
        assert "response_format" not in payload
        if kind == "exhausted_reflection":
            raise http_error(503)
        value = {"protocol_version": "2.0", "reviews": "bad" if len(sent) == 1 else [], "changes": []}
        return Response({"model": payload["model"], "provider": "Anthropic", "choices": [{"finish_reason": "length" if kind == "truncated_content" else "stop",
            "message": {"role": "assistant", "content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.000001}})
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    base_runner = entry.E1bRunner
    class SegmentRunner(base_runner):
        def _operation(self, app, name, operation, *, detail=None):
            if name == "start" and detail.get("phase") == "s2":
                reached.append({"queued": len(app._profile().offline_sessions), "state": app.call_guard.snapshot()})
                raise RuntimeError("software_segment_boundary_after_retained_reflection")
            return super()._operation(app, name, operation, detail=detail)
    monkeypatch.setattr(entry, "E1bRunner", SegmentRunner)
    result = entry.execute(root, root, registration, allow_local_env=True, budget_choice=revision.BUDGET_CHOICE,
        campaign_authorization_sha256=entry.digest(auth), stage_authorization_sha256=entry.digest(phase))
    assert result["status"] == "not_evaluable"  # deliberately partial fixture
    assert len(sent) == physical and len(reached) == 1
    state = reached[0]["state"]
    assert not state["stopped"] and reached[0]["queued"] == 0
    assert all(row["role"] == "reflection" for row in state["attempts"])
    assert state["unknown_cost_requests"] == (3 if kind == "exhausted_reflection" else 0)
    run = output / "main/synthetic-emotion-experiments/e1b-main-v1/run/main/e1b_a1"
    product = entry.read_json(run / "reflection-s1.product.json")
    assert product["product_status"] == ("ok" if kind == "format_correction" else "failed")
    assert product["format_correction_count"] == (1 if kind == "format_correction" else 0)
    assert entry.read_json(run / "retained-prefix.json")["retained_received"] == 13
    assert (run / "memory-s1.first.json").read_bytes() == entry.retained_prefix["originals"]["memory-s1.first.json"]
    assert entry.export_protocol(frozen)["secondary_positions"] == 0
    first = entry.read_json(run / "reflection-s1.first.json")
    if kind == "truncated_content":
        assert first["parse_status"] == "body_received_but_client_rejected" and first["visible_body_received"]
        visible = entry.read_json(output / first["visible_original_path"])
        assert entry.text_sha(visible["content"]) == first["wire_visible_content_sha256"]
        assert state["attempts"][0]["validation_error"] == "invalid_or_truncated_response"
