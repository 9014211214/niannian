"""Portable durable-context restore, independent of private experiment files."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib

from aria.bench.e1b_runner import E1bLocalFixtureClient
from aria.engine.model.persona import OfflineSessionRecord
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.runtime import MinimalModelClients
from scripts import e1br_format_checkpoint as checkpoint
from scripts.e1br_format_runtime import FormatExperimentApp


def test_restore_keeps_failed_turn_actual_fallback_and_original_timestamps(tmp_path, monkeypatch):
    token = SyntheticEmotionExperiment.create(tmp_path, "e1b-main-v1")
    clients = MinimalModelClients(*(E1bLocalFixtureClient(role) for role in ("online", "memory", "reflection")), model_calls_enabled=True)
    app = FormatExperimentApp(token.root / "app", experiment=token, clients=clients)
    at = datetime.now(timezone.utc)
    saved_run = deepcopy(app._run)
    saved_run["run_id"] = "oldrun"
    saved_run["sessions"] = [{"session_id": f"s{number}", "number": number, "status": "sealed",
                              "started_at": at.isoformat(), "sealed_at": at.isoformat()} for number in (1, 2)]
    saved_run["turns"] = [{"turn_id": f"s{session}t{index}", "session_id": f"s{session}", "session_number": session,
        "turn_index": index, "status": "completed", "source": "live", "response": "软件替身旧交付。", "text": "软件替身老人输入。",
        "created_at": at.isoformat()} for session, count in ((1, 12), (2, 6)) for index in range(count)]
    saved_run["turns"][-1].update(status="failed", source="model_failed", response="", requires_review=True)
    persona = app._seed_persona("fictional-chen")
    persona.alliance.trust = .77
    pairs = [{"user": row["text"], "agent": row["response"]} for row in saved_run["turns"][12:]]
    pairs[-1]["agent"] = DETERMINISTIC_MODEL_FAILURE_REPLY
    persona.offline_sessions = [OfflineSessionRecord(session_id="s2", scene_id=app.scene_id,
        created_at=at.isoformat(), expires_at=(at + timedelta(days=1)).isoformat(), turns=pairs)]
    failed = {"request_id": "f" * 32, "role": "online_fast", "status": "failed", "product_step": "initial", "http_status": 200,
        "failure_phase": "receipt_validation", "validation_error": "online_decision_schema_invalid",
        "failure_code": "online_decision_schema_invalid", "reply_return_channel": "json_object",
        "observation": {"body_status": "parsed", "envelope_kind": "success"},
        "receipt_identity": {"version": "receipt_identity_v1", "error_field_status": "absent", "model_status": "recognized", "provider_status": "recognized"},
        "request_sha256": "a" * 64, "visible_original_sha256": "b" * 64, "visible_content_sha256": "c" * 64,
        "client_return_content_sha256": "c" * 64, "usage": {"cost": .0001, "prompt_tokens": 10, "completion_tokens": 5},
        "reservation": {"reserved_cost_usd": ".001", "input_token_plan_bound": 100, "output_token_limit": 100},
        "operation_id": "s2t5", "session_id": "s2"}
    cp = {"saved_run": saved_run, "saved_profile": persona.model_dump(mode="json"), "failed": failed}
    original = deepcopy(cp)
    checks = []
    monkeypatch.setattr(checkpoint, "assert_originals", lambda cp, **kwargs: checks.append(kwargs))
    try:
        result = checkpoint.restore_shared(app, cp)
        assert result == {"ok": True, "retained_turns": 18, "active_session_turns": 6, "new_model_calls": 0}
        assert cp == original and app._run["turns"] == original["saved_run"]["turns"]
        assert app._run["sessions"][-1]["started_at"] == at.isoformat()
        engine, ctx = app._sessions["fictional-chen"]
        assert len(ctx.turns) == 6 and ctx.turns[-1].agent == DETERMINISTIC_MODEL_FAILURE_REPLY
        assert ctx.alliance.trust == engine.persona.alliance.trust == .77
        assert not engine.persona.offline_sessions and len(engine.persona.pending_session.turns) == 6
        assert not app._unresolved()
        proof = app.state()["format_terminal_positions"][0]
        assert proof["historical"] and not proof["model_reply_eligible"] and proof["model_metrics"] is None
        assert proof["fallback_sha256"] == hashlib.sha256(ctx.turns[-1].agent.encode()).hexdigest()
        assert all(c.calls == [] for c in (clients.online, clients.memory, clients.reflection))
        assert len(checks) == 2
    finally:
        app.close()
