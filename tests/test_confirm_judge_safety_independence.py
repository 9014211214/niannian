"""Product wiring with synthetic HTTP/labels; no model semantic accuracy claim."""
import json
from contextlib import closing

import pytest

from aria.engine.policy.max_release import tls_dispatch
from tests.test_confirm_judge_format_failure import setup, INPUT, BODY, offline
from tests.test_installable_max_release import Response


def inconsistent_setup(tmp_path, monkeypatch, *, outcome="confirm", online_mutation=None, review=None, **limits):
    app, guard, sid, ctx, wires, delivered = setup(tmp_path, monkeypatch, outcome=outcome, **limits)
    opened = tls_dispatch.open_observed
    def with_evidence(request, **kwargs):
        response = opened(request, **kwargs)
        data = json.loads(response.body.getvalue())
        wire = json.loads(request.data)
        if wire["max_tokens"] == 4096:
            value = json.loads(data["choices"][0]["message"]["content"])
            value["safety_evidence"] = {"quote": "", "subject": "other", "time": "uncertain", "urgency": "not_applicable"}
            if online_mutation:
                online_mutation(value)
            data["choices"][0]["message"]["content"] = json.dumps(value, ensure_ascii=False)
        elif wire["max_tokens"] == 1024 and review is not None:
            data["choices"][0]["message"]["content"] = json.dumps(review) if isinstance(review, dict) else review
        delivered[-1] = data
        return Response(data)
    monkeypatch.setattr(tls_dispatch, "open_observed", with_evidence)
    return app, guard, sid, ctx, wires, delivered


def test_none_metadata_inconsistency_cannot_cancel_bound_owner_confirmation(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "confirmed"
        assert [wire["max_tokens"] for wire in wires] == [4096, 60, 1024]
        assert app._run["turns"][0]["response"] == BODY + "\n好，这条记下了。"
        assert "safety_evidence" in ctx.working_memory["online_turn_decision"]["dropped_fields"]
        assert ctx.working_memory["safety_review"]["code_receipts_excluded_from_review"] is True
        assert len(ctx.turns) == 1
        assert guard.snapshot()["unknown_cost_requests"] == 0
    finally:
        app.close()
        guard.close()


def test_invitation_expired_during_judge_pacing_is_not_sent(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch)
    sleep = guard._pacing_sleep
    def expire_before_judge(seconds):
        if len(wires) == 1:
            ctx.working_memory["_preference_dialogue_invitation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
        sleep(seconds)
    guard._pacing_sleep = expire_before_judge
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert [wire["max_tokens"] for wire in wires] == [4096, 1024]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
        assert "记下了" not in app._run["turns"][0]["response"]
        assert guard.snapshot()["unknown_cost_requests"] == 0
        assert not guard.snapshot()["stopped"]
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize("action,status,receipt", [
    ("confirm", "confirmed", "好，这条记下了。"),
    ("decline", "declined", "好，这条不记。"),
    ("defer", "proposed", "好，这条以后再说。"),
    ("unclear", "proposed", "这条我先不记，你想记的时候再说一声。"),
    ("invalid_json", "proposed", "这条我先不记，你想记的时候再说一声。"),
])
def test_uncertain_metadata_keeps_all_transaction_outcomes(tmp_path, monkeypatch, action, status, receipt):
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch, outcome=action)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == status
        assert app._run["turns"][0]["response"] == BODY + "\n" + receipt
        assert [x["max_tokens"] for x in wires] == [4096, 60, 1024]
        attempts = guard.snapshot()["attempts"]
        assert attempts[1]["confirmation_context_admission"]["physical_rebuilds"] == 0
        assert attempts[1]["confirmation_context_admission"]["original_invitation_expires_at"]
        assert ctx.working_memory["intervention_delivery"]["status"] == "not_delivered"
        assert 0 not in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])
        assert len(ctx.turns) == 1
    finally:
        app.close()
        guard.close()


def test_review_deletion_never_reappears_when_receipt_is_kept(tmp_path, monkeypatch):
    from tests.test_safety_review_v2 import answer
    def body(value):
        value["response_segments"][1]["text"] = "保留甲。删除片段。保留乙。"
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch,
        online_mutation=body, review=answer(delete=[0, 2]))
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == "保留甲。保留乙。\n好，这条记下了。"
        sent = json.loads(wires[-1]["messages"][1]["content"])
        assert "好，这条记下了。" not in sent["candidate"]
        assert sent["candidate"] == "我已经记下了。保留甲。删除片段。保留乙。"
        assert ctx.working_memory["safety_review"]["delete_sentence_ids"] == [0, 2]
        assert [x["max_tokens"] for x in wires] == [4096, 60, 1024]
    finally:
        app.close()
        guard.close()


def test_failed_review_blocks_body_without_undoing_saved_receipt(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch, review="not-json")
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        actual = app._run["turns"][0]["response"]
        assert BODY not in actual and actual.endswith("好，这条记下了。")
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "confirmed"
        assert ctx.working_memory["safety_review"]["model_body_delivered"] is False
        assert [x["max_tokens"] for x in wires] == [4096, 60, 1024]
    finally:
        app.close()
        guard.close()


def test_storage_failure_never_claims_saved_after_review(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch)
    save = app.store.save
    def fail_confirmation(persona):
        if any(p.status == "confirmed" for p in persona.stated_preferences):
            raise OSError("synthetic write failure")
        return save(persona)
    monkeypatch.setattr(app.store, "save", fail_confirmation)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == BODY + "\n这条没能保存，我先不记。"
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
        assert "preference_dialogue_receipt" not in ctx.working_memory
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize("concern", ["fall", "medical_emergency", "self_harm", "unsafe_medication"])
def test_non_none_risk_keeps_original_protection_priority(tmp_path, monkeypatch, concern):
    from tests.test_safety_review_v2 import answer
    def risk(value):
        value["safety_concern"] = concern
        value["safety_evidence"] = {"quote": INPUT, "subject": "owner", "time": "current", "urgency": "immediate"}
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch,
        online_mutation=risk, review=answer(concern, subject="owner", time="current", urgency="immediate"))
    events = []
    try:
        assert app.submit(INPUT, "turn1", sid, on_safety_prelude=lambda event: events.append(event) or True)["ok"]
        assert len(events) == 1
        assert [x["max_tokens"] for x in wires] == [4096, 1024]
        assert "confirm_judge" not in ctx.working_memory["online_turn_decision"]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
        assert ctx.working_memory["safety_review"]["prelude_status"] == "delivered_after_review"
        assert ctx.working_memory["safety_review"]["risk_assessment"]["concern"] == concern
        assert len(ctx.turns) == 1
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize("phase", ["tls", "after_headers"])
@pytest.mark.parametrize("change", ["expiry", "owner", "session", "record", "disk"])
def test_judge_application_boundary_rechecks_live_authority(tmp_path, monkeypatch, phase, change):
    from datetime import timedelta
    from tests.test_public_preference_feedback import setup as observed_setup
    def mutate(holder, clock):
        if len(wires) != 2:
            return
        if change == "expiry":
            clock["utc"] += timedelta(seconds=400)
        elif change == "owner":
            holder["ctx"].user_id = "different-owner"
        elif change == "session":
            holder["ctx"].session_id = "different-session"
        elif change == "record":
            holder["persona"].stated_preferences[0].do = "changed private scope"
        else:
            path = holder["app"].run_path
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run_id"] = "external"
            path.write_text(json.dumps(data), encoding="utf-8")
    app, wires, sockets, sid, holder, _ = observed_setup(tmp_path, monkeypatch,
        action="confirm", mutation=mutate, phase=phase)
    try:
        if change == "disk":
            with pytest.raises(RuntimeError):
                app.submit(INPUT, "turn1", sid)
        else:
            # The already received online turn may finish with safe failure
            # feedback, while its paid allowance is stopped without replay.
            app.submit(INPUT, "turn1", sid)
            assert BODY not in app._run["turns"][-1]["response"]
            assert "记下了" not in app._run["turns"][-1]["response"]
        assert len(wires) == 2
        assert sum(len(sock.writes) for sock in sockets) == 2 + (1 if phase == "after_headers" else 0)
        assert app.call_guard.snapshot()["stopped"]
        assert app.call_guard.snapshot()["paid_response_replays"] == 0
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
    finally:
        try:
            app.close()
        except (ValueError, RuntimeError, OSError):
            pass


@pytest.mark.parametrize("change", ["halt", "durable_halt", "disk"])
def test_commit_rechecks_authority_after_candidate_is_built(tmp_path, monkeypatch, change):
    from aria.engine.memory import preference_dialogue
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path, monkeypatch)
    changed = preference_dialogue._changed
    injected = []
    def change_before_store(row, status, **kwargs):
        result = changed(row, status, **kwargs)
        if status == "confirmed":
            injected.append(status)
            assert guard.snapshot()["attempts"][-1]["status"] == "received"
            if change == "halt":
                guard.halt()
            elif change == "durable_halt":
                with guard._transaction():
                    guard._db.execute("UPDATE settings SET halted='model_or_storage_failed' WHERE id=1")
            else:
                data = json.loads(app.run_path.read_text(encoding="utf-8"))
                data["run_id"] = "external-before-save"
                app.run_path.write_text(json.dumps(data), encoding="utf-8")
        return result
    monkeypatch.setattr(preference_dialogue, "_changed", change_before_store)
    try:
        try:
            app.submit(INPUT, "turn1", sid)
        except (ValueError, RuntimeError, OSError):
            pass
        assert injected == ["confirmed"]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
        assert [x["max_tokens"] for x in wires] == [4096, 60]
        assert ctx.working_memory["online_turn_decision"]["preference_commit_diagnostic"]["status"] == "failed"
        assert "preference_dialogue_receipt" not in ctx.working_memory
        assert guard.snapshot()["unknown_cost_requests"] == 0
    finally:
        try:
            app.close()
        except (ValueError, RuntimeError, OSError):
            pass
        guard.close()


@pytest.mark.parametrize("limit", ["requests", "cost"])
def test_last_settled_judge_can_commit_without_an_explicit_stop(tmp_path, monkeypatch, limit):
    from decimal import Decimal
    budget = "1"
    if limit == "cost":
        pilot, pilot_guard, pilot_sid, _, _, _ = inconsistent_setup(tmp_path / "pilot", monkeypatch,
            max_requests=2, bill_reserved=True)
        try:
            pilot.submit(INPUT, "pilot-turn", pilot_sid)
            attempts = pilot_guard.snapshot()["attempts"]
            assert len(attempts) == 2 and all(x["status"] == "received" for x in attempts)
            budget = str(sum((Decimal(x["reservation"]["reserved_cost_usd"]) for x in attempts), Decimal("0")))
        finally:
            pilot.close()
            pilot_guard.close()
    app, guard, sid, ctx, wires, _ = inconsistent_setup(tmp_path / "actual", monkeypatch,
        max_requests=2 if limit == "requests" else 8, budget=budget, bill_reserved=limit == "cost")
    try:
        app.submit(INPUT, "turn1", sid)
        assert [x["max_tokens"] for x in wires] == [4096, 60]
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "confirmed"
        assert ctx.working_memory["preference_dialogue_receipt"]["persisted"] is True
        state = guard.snapshot()
        assert state["unknown_cost_requests"] == 0 and state["reserved_cost_usd"] == "0"
        assert all(x["status"] == "received" for x in state["attempts"])
        assert state["stopped"]
        assert BODY not in app._run["turns"][0]["response"]  # review was not affordable
        assert "好，这条记下了。" in app._run["turns"][0]["response"]
    finally:
        app.close()
        guard.close()
