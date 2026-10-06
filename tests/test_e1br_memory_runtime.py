"""Software transport only: current privacy blocks unreviewed reflection.

The historical frozen experiment used 88 sends. Current code retains its 88
planned positions but skips the failed memory session's reflection position.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from aria.bench.e1b_cases import load_e1b_cases
from aria.bench.e1b_runner import E1bLocalFixtureClient, E1bRunStopped
from aria.bench.e1br_runner import E1BRRunner
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy import minimal_calls
from aria.engine.policy.e1br_calls import E1BRCallGuard
from aria.runtime import MinimalModelClients
from scripts.e1br_memory_runtime import (
    SCHEMA, MemoryExperimentApp, MemoryGuardMixin, complete_known_memory,
    json_syntax_failure, validate_memory_ledger, validate_memory_terminal_proof, require_unexpired_job,
)
from scripts.e1br_format_runtime import SCHEMA as FORMAT_SCHEMA
from scripts.run_e1br import position_key, position_inventory
from tests.test_reserved_calls import client, policy, receipt
from tests.test_e1br_format_runtime import DECISION

CONTRACT = {"schema": SCHEMA, "retry_answer_bearing_failure": False, "effect_claim_eligible": False}
FORMAT_CONTRACT = {**CONTRACT, "schema": FORMAT_SCHEMA}
BAD = '{"episodes":[{"evidence_quote":"software "bad""}],"relationships":[],"stated_preferences":[]}'


class SoftwareGuard(MemoryGuardMixin, E1BRCallGuard):
    def __init__(self, path):
        super().__init__(path, max_observed_cost_usd="2", reservation_policy=policy(),
            registration_sha256="b" * 64, authority_check=lambda: None,
            format_contract=FORMAT_CONTRACT, format_evidence_root=path.parent,
            memory_contract=CONTRACT, memory_evidence_root=path.parent)
        clock = [0.0]
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)

    @contextmanager
    def experiment_position(self, name, detail):
        assert getattr(self._local, "experiment_position", None) is None
        self._local.experiment_position = name, detail
        try:
            yield
        finally:
            self._local.experiment_position = None

    def _reserve(self, c, request, payload, state):
        row = super()._reserve(c, request, payload, state)
        row["experiment_position"] = list(position_key(c.role, *self._local.experiment_position))
        return row

    def _record_receipt(self, c, row, data):
        content = data["choices"][0]["message"]["content"]
        name = f"first-{row['request_id']}.json"
        raw = json.dumps({"request_id": row["request_id"], "role": row["role"], "content": content}).encode()
        (self._memory_root / name).write_bytes(raw)
        row.update(visible_original_path=name, visible_original_sha256=hashlib.sha256(raw).hexdigest(),
                   visible_content_sha256=hashlib.sha256(content.encode()).hexdigest())
        return super()._record_receipt(c, row, data)


def run_software_case(tmp_path, monkeypatch, *, failed_session=1, memory_body=BAD, missing_cost=False, store_failure=False):
    def denied(*args, **kwargs):
        raise AssertionError("unexpected_external_network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    token = SyntheticEmotionExperiment.create(tmp_path, "memory-local")
    guard = SoftwareGuard(tmp_path / "ledger.sqlite3")
    clients = MinimalModelClients(*(client(role) for role in ("online_fast", "memory", "reflection")), model_calls_enabled=True)
    clients.reflection.reflection_return_format = "body_json"
    apps, calls, before_failure = [], [], []
    fixture = E1bLocalFixtureClient("memory")
    counts = {"memory": 0}

    class Response:
        status, headers = 200, {}
        def __init__(self, data):
            self.data = json.dumps(data).encode()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return self.data[:size]

    def opened(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        row = guard.snapshot()["attempts"][-1]
        role = row["role"]
        c = clients.online if role == "online_fast" else getattr(clients, role)
        content = json.dumps(DECISION, ensure_ascii=False) if payload.get("response_format") == {"type": "json_object"} else "软件替身回复。"
        bad = False
        if role == "memory":
            counts["memory"] += 1
            bad = counts["memory"] == failed_session
            if bad:
                before_failure.append(apps[0]._profile().model_dump(mode="json"))
            content = memory_body if bad else fixture.complete(payload["messages"])
        elif role == "reflection":
            content = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
        data = receipt(c)
        data["choices"][0]["message"]["content"] = content
        if bad and missing_cost:
            data["usage"].pop("cost")
        return Response(data)
    monkeypatch.setattr(minimal_calls, "open_observed", opened)

    def factory(path):
        app = MemoryExperimentApp(path, experiment=token, clients=clients,
            model_call_guard=guard, model_reservation_policy=policy())
        apps.append(app)
        if store_failure:
            monkeypatch.setattr(guard, "record_memory_terminal", lambda *args: (_ for _ in ()).throw(OSError("software proof storage failure")))
        return app
    # Exercise the guarded production route with transport replaced above;
    # this flag never permits network because both socket entry points fail.
    runner = E1BRRunner(load_e1b_cases(), root=token.root / "run", app_factory=factory, seed=20261003,
                       evidence_kind="real_model", allow_model_calls=True)
    try:
        result = runner.run_case("e1b_f2")
    except E1bRunStopped:
        result = None
    state = guard.snapshot()
    guard.close()
    return result, state, runner, apps, calls, before_failure


@pytest.mark.parametrize("session", [1, 2])
def test_current_privacy_skips_unreviewed_reflection_and_retains_88_planned_positions(tmp_path, monkeypatch, session):
    result, state, runner, apps, calls, before = run_software_case(tmp_path, monkeypatch, failed_session=session)
    assert result is not None and not runner.stopped, (state["stop_reason"], apps[0]._run["reviews"])
    planned = {position for position in position_inventory() if position[1] == "e1b_f2"}
    observed = {tuple(row["experiment_position"]) for row in state["attempts"]}
    skipped = {("reflection", "e1b_f2", session, "shared", 0, None)}
    assert len(planned) == 88 and planned - observed == skipped
    assert observed <= planned and len(calls) == len(state["attempts"]) == len(observed) == 87
    assert not state["stopped"] and state["unknown_cost_requests"] == 0
    proof = result["shared"]["state"]["memory_terminal_positions"][0]
    directory = runner.root / "main/e1b_f2"
    row = next(r for r in state["attempts"] if r["request_id"] == proof["request_id"])
    assert "cost" in row["usage"] and row["usage"]["cost"] > 0
    assert proof["fixed_denominator_retained"] is True and proof["body_retried"] is False
    assert not any(r["role"] == "reflection" and r["session_id"] == row["session_id"]
                   for r in state["attempts"])
    validate_memory_terminal_proof(directory, proof, row)
    first = json.loads((directory / f"memory-s{session}.first.json").read_bytes())
    product = json.loads((directory / f"memory-s{session}.product.json").read_bytes())
    assert first["raw_proposals"] is None and first["parse_status"] == "parse_error"
    assert product["product_status"] == "failed" and product["accepted_count"] is None
    baseline = json.loads((directory / proof["baseline_profile_path"]).read_bytes())
    failed_job = next(job for job in baseline["offline_sessions"] if job["session_id"] == row["session_id"])
    assert failed_job["memory_privacy_review_completed"] is False
    for field in ("episodic_memory", "stated_preferences", "person_hypotheses", "relationship_context"):
        assert baseline[field] == before[0][field]
    audit = next(r for r in result["shared"]["profile"]["offline_session_audit"] if r["session_id"] == row["session_id"])
    assert audit["memory_completed"] is False and audit["gap"] is True and audit["reflection_completed"] is False
    reflection_audit = json.loads((directory / f"memory-s{session}.terminal.reflection.json").read_bytes())
    assert reflection_audit["reflection_status"] == "sealed_failed"
    assert reflection_audit["reflection_error"] == "memory_privacy_unreviewed"
    assert reflection_audit["offline_audit"] == audit
    other = next(r for r in result["shared"]["profile"]["offline_session_audit"] if r["session_id"] != row["session_id"])
    assert other["outcome"] == "completed" and other["memory_completed"] and other["reflection_completed"]
    assert any(r["role"] == "reflection" and r["session_id"] == other["session_id"] for r in state["attempts"])
    # Both failure positions still allow the later six S3 branches to run;
    # the S1 variant additionally checks the subsequent valid S2 review above.
    assert len(result["branches"]) == 6
    assert all(len(branch["state"]["turns"]) == 10
               and all(turn["status"] == "completed" for turn in branch["state"]["turns"])
               for branch in result["branches"])
    assert result["shared"]["profile"]["offline_sessions"] == []
    assert len(validate_memory_ledger(tmp_path / "ledger.sqlite3", memory_contract=CONTRACT, attempts=state["attempts"])) == 1


@pytest.mark.parametrize("change", ["semantic", "plain_refusal", "unknown_cost", "proof_storage"])
def test_other_memory_failures_still_stop_before_reflection(tmp_path, monkeypatch, change):
    result, state, runner, apps, calls, _ = run_software_case(tmp_path, monkeypatch,
        memory_body='{"episodes":[],"relationships":[],"stated_preferences":null}' if change == "semantic"
        else "软件拒绝示例" if change == "plain_refusal" else BAD,
        missing_cost=change == "unknown_cost", store_failure=change == "proof_storage")
    assert result is None and runner.stopped and state["stopped"]
    assert len(calls) == 13 and all(row["role"] != "reflection" for row in state["attempts"])
    assert apps[0]._profile().offline_sessions[0].memory_completed is False


@pytest.mark.parametrize("content,expected", [(BAD, True), ('{"stated_preferences":null}', False),
    ('{"x":NaN}', False), ('{"x":1,"x":2}', False), ("plaintext refusal", False)])
def test_only_actual_json_decode_errors_are_eligible(content, expected):
    assert json_syntax_failure(content) is expected


def test_complete_memory_gate_rejects_missing_cost_or_truncated_receipt():
    assert not complete_known_memory({"role": "memory", "status": "received"})
    assert not complete_known_memory({"usage": {"cost": "NaN"}, "reservation": {"reserved_cost_usd": ".1"}})


def test_historical_baseline_is_bound_after_new_run_identity(tmp_path, monkeypatch):
    result, state, runner, _, _, _ = run_software_case(tmp_path, monkeypatch)
    proof = deepcopy(result["shared"]["state"]["memory_terminal_positions"][0])
    directory = runner.root / "main/e1b_f2"
    row = next(r for r in state["attempts"] if r["request_id"] == proof["request_id"])
    proof["original_profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="memory_terminal_evidence_changed"):
        validate_memory_terminal_proof(directory, proof, row)


@pytest.mark.parametrize("expires,valid", [("2026-10-06T00:00:00+00:00", True),
    ("2026-10-05T00:00:00+00:00", False), ("2026-10-04T00:00:00+00:00", False)])
def test_current_utc_expiry_cannot_be_extended(monkeypatch, expires, valid):
    from scripts import e1br_memory_runtime as runtime
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(runtime, "datetime", Fixed)
    job = SimpleNamespace(created_at="2026-10-03T00:00:00+00:00", expires_at=expires)
    before = vars(job).copy()
    if valid:
        require_unexpired_job(job)
    else:
        with pytest.raises(ValueError, match="memory_original_job_expired"):
            require_unexpired_job(job)
    assert vars(job) == before
