"""Known visible failure stays failed; only its saved fallback allows progress."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json

import pytest

from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.e1br_calls import E1BRCallGuard
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.runtime import MinimalModelClients
from scripts.e1br_format_runtime import (
    SCHEMA, FormatExperimentApp, FormatGuardMixin, terminal_format_failure,
    validate_format_ledger,
)
from scripts.e1b_evaluation import digest
from tests.test_e1br_calls import stubs, http_error
from tests.test_reserved_calls import client, policy, receipt

CONTRACT = {"schema": SCHEMA, "retry_answer_bearing_failure": False, "effect_claim_eligible": False}
DECISION = {"response": "本地替身，仅检查存储与下一位置。", "state_signals": [],
    "current_affect": {"label": "unknown", "confidence": .3,
        "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
    "selected_intervention_id": "", "uncertainty": .6,
    "communication_plan": {"pace": "open_listening", "support_order": "先听本人说", "boundary": "不强加话题"}}


class Guard(FormatGuardMixin, E1BRCallGuard):
    def __init__(self, path, **kwargs):
        self.evidence = path.parent
        super().__init__(path, max_observed_cost_usd="1", reservation_policy=policy(),
            registration_sha256="b" * 64, authority_check=lambda: None,
            format_contract=CONTRACT, format_evidence_root=path.parent, **kwargs)
        clock = [0.0]
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)

    def _record_receipt(self, c, row, data):
        content = data["choices"][0]["message"]["content"]
        name = f"first-{row['request_id']}.json"
        raw = json.dumps({"request_id": row["request_id"], "content": content}).encode()
        (self.evidence / name).write_bytes(raw)
        row.update(visible_original_path=name, visible_original_sha256=hashlib.sha256(raw).hexdigest(),
                   visible_content_sha256=hashlib.sha256(content.encode()).hexdigest())
        return super()._record_receipt(c, row, data)


def app(tmp_path, g):
    token = SyntheticEmotionExperiment.create(tmp_path, "format-local")
    clients = MinimalModelClients(*(client(role) for role in ("online_fast", "memory", "reflection")), model_calls_enabled=True)
    return FormatExperimentApp(token.root / "app", experiment=token, clients=clients,
        model_call_guard=g, model_reservation_policy=policy())


def envelope(c, value):
    result = receipt(c)
    result["choices"][0]["message"]["content"] = json.dumps(value, ensure_ascii=False)
    return result


@pytest.mark.parametrize("infrastructure_failures", [0, 2])
def test_bad_first_body_then_next_position_keeps_failed_row_and_no_retry(tmp_path, monkeypatch, infrastructure_failures):
    path = tmp_path / "ledger.sqlite3"
    with closing(Guard(path)) as g:
        a = app(tmp_path, g)
        try:
            c = a.clients.online
            calls = stubs(monkeypatch, [*[http_error(503) for _ in range(infrastructure_failures)],
                envelope(c, {**DECISION, "uncertainty": "bad"}), envelope(c, DECISION)])
            session = a.start()["state"]["session_id"]
            first = a.submit("我有点难受，但又说不上来，别人总替我做决定。", "turn1", session)
            assert first["ok"] and first["model_reply_valid"] is False, (first.get("error"), g.snapshot())
            assert len(calls) == 1 + infrastructure_failures and not g.snapshot()["stopped"]
            old = deepcopy(g.snapshot()["attempts"][infrastructure_failures])
            assert terminal_format_failure(old)
            assert a._run["turns"][0]["status"] == "failed"
            assert a._run["turns"][0]["response"] == ""
            proof = a.state()["format_terminal_positions"][0]
            assert proof["receipt_sha256"] == digest(old) and proof["model_metrics"] is None
            second = a.submit("我还是难过，也不知道该怎样说。", "turn2", session)
            assert second["ok"] and len(calls) == 2 + infrastructure_failures
            assert g.snapshot()["attempts"][infrastructure_failures] == old
            assert g.snapshot()["unknown_cost_requests"] == infrastructure_failures
            assert len(a._sessions["fictional-chen"][1].turns) == 2
            with pytest.raises(MinimalCallStopped, match="duplicate_model_operation"):
                with g.scope("turn", "turn1", session):
                    c.complete(calls[0]["messages"])
            assert len(calls) == 2 + infrastructure_failures
            saved = g.snapshot()["attempts"]
        finally:
            a.close()
    assert len(validate_format_ledger(path, format_contract=CONTRACT, attempts=saved)) == 1


@pytest.mark.parametrize("mutation", [
    {"http_status": 500}, {"failure_phase": "read"}, {"status": "received"}, {"role": "memory"},
    {"failure_code": "provider_cost_unknown"}, {"visible_original_sha256": ""},
    {"receipt_identity": {"error_field_status": "object"}}, {"model_reply_sha256": "a" * 64},
    {"observation": {"body_status": "truncated", "envelope_kind": "success"}},
    {"usage": {}}, {"usage": {"cost": "NaN"}}, {"no_content_evidence": {"kind": "upstream_error"}},
])
def test_only_known_complete_strict_format_failure_is_terminal(tmp_path, monkeypatch, mutation):
    with closing(Guard(tmp_path / "ledger.sqlite3")) as g:
        a = app(tmp_path, g)
        try:
            stubs(monkeypatch, [envelope(a.clients.online, {**DECISION, "uncertainty": "bad"})])
            session = a.start()["state"]["session_id"]
            assert a.submit("我有点难受，但又说不上来，别人总替我做决定。", "turn1", session)["ok"]
            row = deepcopy(g.snapshot()["attempts"][0])
            assert terminal_format_failure(row)
            row.update(mutation)
            assert not terminal_format_failure(row)
        finally:
            a.close()


def test_terminal_delivery_save_failure_halts_and_cannot_continue(tmp_path, monkeypatch):
    with closing(Guard(tmp_path / "ledger.sqlite3")) as g:
        a = app(tmp_path, g)
        try:
            calls = stubs(monkeypatch, [envelope(a.clients.online, {**DECISION, "uncertainty": "bad"})])
            session = a.start()["state"]["session_id"]
            monkeypatch.setattr(g, "record_terminal_delivery", lambda *a: (_ for _ in ()).throw(OSError("local fixture")))
            with pytest.raises(OSError):
                a.submit("我有点难受，但又说不上来，别人总替我做决定。", "turn1", session)
            assert g.snapshot()["stopped"] and len(calls) == 1
            assert a.submit("不能继续。", "turn2", session)["ok"] is False
        finally:
            a.close()


def test_stored_contract_change_is_fail_closed(tmp_path):
    with closing(Guard(tmp_path / "ledger.sqlite3")) as g:
        g._db.execute("UPDATE e1br_format_policy SET contract='{}'")
        assert g.snapshot()["stop_reason"] == "ledger_unavailable"


def test_historical_default_never_accepts_a_primary_failure():
    assert E1BRCallGuard.terminal_primary_failure(None, {"status": "failed"}) is False
