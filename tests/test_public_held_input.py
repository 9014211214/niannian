"""Actual public App and guard with neutral local transport fixtures only."""
from copy import deepcopy
import http.client
import json
import socket

import pytest

from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.max_release import tls_dispatch
from tests.test_installable_max_release import make_clients,prices,response,Response


@pytest.fixture(autouse=True)
def forbid_network_and_real_keys(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("network_or_real_key_forbidden")
    monkeypatch.setattr(socket,"create_connection",forbidden)
    monkeypatch.setattr(socket.socket,"connect",forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key",forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key",forbidden)


def fixture_client(clients,request):
    carrier=json.loads(json.loads(request.data)["messages"][-1]["content"])
    return clients.online if carrier.get("schema")=="online_json_input_v2" else clients.safety_review


def setup(tmp_path,monkeypatch,failures=(1,2),fault=None,budget="2"):
    clients=make_clients();calls=[]
    def opened(request,**kwargs):
        calls.append(json.loads(request.data));trace=kwargs["trace"]
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        if len(calls) in failures:
            for event in ("connection_preparation_started","connection_preparation_completed","tls_completed",
                          "application_request_started","application_write_attempted","request_write_completed"):
                if fault!="partial_write" or event!="request_write_completed":trace._mark(event)
            if fault=="response_headers":trace._mark("response_headers_received")
            if fault=="untyped_timeout":raise TimeoutError("synthetic local failure")
            if fault=="deadline":
                deadline=trace.response_deadline
                deadline.clock=lambda:deadline.expires_at+1
                deadline.remaining()
            raise http.client.RemoteDisconnected("synthetic local disconnect")
        return Response(response(fixture_client(clients,request)))
    monkeypatch.setattr(tls_dispatch,"open_observed",opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=10,model_stop_cost_usd=budget,model_reservation_policy=prices())
    clock=[1000.0];app.call_guard._pacing_clock=lambda:clock[0]
    app.call_guard._pacing_sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
    return app,calls,app.start()["state"]["session_id"]


def consent(app,sid,index):
    view=app.state()["held_new_input"]
    assert view["available"],app.call_guard.snapshot()
    return app.continue_new_input("action-"+str(index),view["failure_id"],sid,confirmed=True)


def close_corrupt_fixture(app):
    before=app.run_path.read_bytes() if app.run_path.exists() else None
    if app._held_run_integrity_failed:
        # Ordinary close may not falsely seal/overwrite corrupted durable data.
        with pytest.raises(RuntimeError,match="尚未安全封存"):app.close()
        assert (app.run_path.read_bytes() if app.run_path.exists() else None)==before
        app.call_guard.close()
    else:app.close()


def test_two_failures_need_two_owner_consents_then_new_input(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch)
    try:
        for n in (1,2):
            result=app.submit("中性软件输入。","turn-"+str(n),sid)
            assert not result["ok"] and result["held_unreceived_failure"],result
            state=app.call_guard.snapshot();assert state["stopped"] and state["unknown_cost_requests"]==n
            row=deepcopy(state["attempts"][-1]);held=state["reserved_cost_usd"]
            blocked=app.submit("另一个中性输入。","blocked-"+str(n),sid)
            assert not blocked["ok"] and len(calls)==n
            result=consent(app,sid,n);assert result["ok"] and result["request_sent"] is False
            assert not app.call_guard.snapshot()["stopped"]
            assert app.call_guard.snapshot()["attempts"][-1]==row
            assert app.call_guard.snapshot()["reserved_cost_usd"]==held
            old=app.submit("中性软件输入。","turn-"+str(n),sid)
            assert old["duplicate"] and len(calls)==n
        # A new operation may intentionally contain the same text. No text
        # blacklist substitutes for identity-based replay prevention.
        result=app.submit("中性软件输入。","turn-3",sid)
        assert result["ok"] and len(calls)==4
        assert app._run["turns"][-1]["safety_review"]["status"]=="reviewed"
        assert [r["role"] for r in app.call_guard.snapshot()["attempts"]]==[
            "online_fast","online_fast","online_fast","safety_review"]
        assert app.call_guard.snapshot()["unknown_cost_requests"]==2
        assert app.call_guard.snapshot()["paid_response_replays"]==0
    finally:app.close()


@pytest.mark.parametrize("fault",["partial_write","response_headers","untyped_timeout"])
def test_nonqualifying_failures_stay_stopped(tmp_path,monkeypatch,fault):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,),fault=fault)
    try:
        result=app.submit("中性软件输入。","t1",sid)
        assert not result["ok"] and not app.state()["held_new_input"]["available"]
        assert app.call_guard.snapshot()["stopped"] and len(calls)==1
    finally:app.close()


def test_actual_same_deadline_origin_retains_hold(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,),fault="deadline")
    try:
        result=app.submit("中性软件输入。","t1",sid)
        assert not result["ok"] and result["held_unreceived_failure"]
        row=app.call_guard.snapshot()["attempts"][0]
        assert row["held_noresponse_origin"]["origin_kind"]=="local_deadline_expired"
        assert consent(app,sid,1)["ok"]
        assert app.submit("后续中性输入。","t2",sid)["ok"] and len(calls)==3
        assert app._run["turns"][-1]["safety_review"]["status"]=="reviewed"
    finally:app.close()


def test_confirmation_owner_scope_readonly_and_duplicate_are_not_calls(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid)
        failure=app.state()["held_new_input"]["failure_id"]
        with pytest.raises(PermissionError):app.continue_new_input("a1",failure,sid,confirmed=True,actor="other")
        assert not app.continue_new_input("a1",failure,sid,confirmed=False)["ok"]
        assert not app.continue_new_input("a1",failure,sid,confirmed=True,read_only=True)["ok"]
        assert not app.continue_new_input("a1",failure,"other",confirmed=True)["ok"]
        assert app.continue_new_input("a1",failure,sid,confirmed=True)["ok"]
        assert app.continue_new_input("a1",failure,sid,confirmed=True)["duplicate"]
        assert len(calls)==1
    finally:app.close()


@pytest.mark.parametrize("accepted",[False,True])
def test_restart_retains_consent_and_unknown_charge(tmp_path,monkeypatch,accepted):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    app.submit("中性软件输入。","t1",sid)
    if accepted:assert consent(app,sid,1)["ok"]
    original=deepcopy(app.call_guard.snapshot()["attempts"]);app.close()
    app2=MinimalWebApp(tmp_path,clients=make_clients(),model_max_requests=10,model_stop_cost_usd="2",model_reservation_policy=prices())
    try:
        state=app2.state()
        assert app2.call_guard.snapshot()["attempts"]==original
        assert app2.call_guard.snapshot()["stopped"] is not accepted
        assert state["held_new_input"]["available"] is not accepted
        assert not app2.submit("中性软件输入。","t1",sid)["ok"] and len(calls)==1
        if not accepted:assert consent(app2,sid,2)["ok"]
        assert not app2.state()["requires_review"]
    finally:app2.close()


@pytest.mark.parametrize("accepted",[False,True])
def test_forget_minimal_tombstone_does_not_create_consent_or_reset_hold(tmp_path,monkeypatch,accepted):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("此中性内容必须随删除消失。","t1",sid)
        if accepted:assert consent(app,sid,1)["ok"]
        before=deepcopy(app.call_guard.snapshot())
        app.forget_run()
        after=app.call_guard.snapshot();view=app.state()
        assert view["turns"]==[] and view["run_id"]!=before["attempts"][0]["held_online_before_write"]["before"]["run_id"]
        assert after["attempts"]==before["attempts"] and after["reserved_cost_usd"]==before["reserved_cost_usd"]
        assert after["stopped"] is not accepted
        proof,=app.call_guard.held_records()
        assert proof["state"]=="erased" and (proof["consent"] is not None)==accepted
        assert "此中性内容" not in json.dumps(proof,ensure_ascii=False)
        assert not app.submit("此中性内容必须随删除消失。","t1",sid)["ok"] and len(calls)==1
    finally:app.close()


@pytest.mark.parametrize("fault",["owner","run","failed_turn","amount","before_digest","row_identity"])
def test_current_or_persistent_proof_tampering_cannot_unlock(tmp_path,monkeypatch,fault):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid)
        proof,=app.call_guard.held_records();failure=proof["request_id"]
        if fault=="owner":app._settled_owner="other"
        elif fault=="run":app._run["run_id"]="changed"
        elif fault=="failed_turn":app._run["turns"][0]["response"]="forged"
        elif fault=="row_identity":
            row=app.call_guard.held_row(failure);row["request_sha256"]="0"*64
            app.call_guard._db.execute("UPDATE attempts SET record=? WHERE request_id=?",(json.dumps(row),failure));app.call_guard._db.commit()
        else:
            if fault=="amount":proof["held_usd"]="0"
            else:proof["after_business_sha256"]="0"*64
            app.call_guard._db.execute("UPDATE held_input_dispositions SET record=? WHERE request_id=?",(json.dumps(proof),failure));app.call_guard._db.commit()
        assert not app.continue_new_input("tampered",failure,sid,confirmed=True)["ok"]
        assert app.call_guard.snapshot()["stopped"] and len(calls)==1
    finally:
        app._settled_owner="fictional-chen"
        close_corrupt_fixture(app)


@pytest.mark.parametrize("stage",["proof","consent","forget"])
def test_persistent_write_failure_stays_stopped(tmp_path,monkeypatch,stage):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    original=app.call_guard.write_held_record
    def broken(*args,**kwargs):raise OSError("local-fixture-write-failed")
    try:
        if stage=="proof":monkeypatch.setattr(app.call_guard,"write_held_record",broken)
        if stage=="proof":
            with pytest.raises(OSError):app.submit("中性软件输入。","t1",sid)
        else:
            app.submit("中性软件输入。","t1",sid)
            monkeypatch.setattr(app.call_guard,"write_held_record",broken)
            with pytest.raises(OSError):
                if stage=="consent":consent(app,sid,1)
                else:app.forget_run()
        assert app.call_guard.snapshot()["stopped"] and len(calls)==1
    finally:
        monkeypatch.setattr(app.call_guard,"write_held_record",original)
        app.close()


def test_original_checkpoint_deadline_not_extended(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        original=app.store.load(app._settled_owner).pending_session.expires_at
        app.submit("中性软件输入。","t1",sid)
        assert app.store.load(app._settled_owner).pending_session.expires_at==original
        assert consent(app,sid,1)["ok"]
        assert app.store.load(app._settled_owner).pending_session.expires_at==original
    finally:app.close()


def test_full_hold_reduces_next_request_budget_before_dispatch(tmp_path,monkeypatch):
    from decimal import Decimal
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,),budget="0.15")
    try:
        app.submit("中性软件输入。","t1",sid)
        assert consent(app,sid,1)["ok"]
        held=app.call_guard.snapshot()["reserved_cost_usd"]
        result=app.submit("后续中性输入。","t2",sid)
        assert not result["ok"] and len(calls)==1
        state=app.call_guard.snapshot()
        assert state["reserved_cost_usd"]==held and state["stopped"]
        assert state["unknown_cost_requests"]==1
        refusal,=app.call_guard.budget_refusals()
        assert refusal==state["local_budget_refusals"][0]
        assert refusal["reason"]=="pre_send_budget_exceeded" and refusal["operation_id"]=="t2"
        assert refusal["session_id"]==sid and refusal["role"]=="online_fast"
        assert refusal["held_usd"]==held and refusal["cap_usd"]=="0.15"
        assert Decimal(refusal["known_usd"])+Decimal(refusal["held_usd"])+Decimal(refusal["next_reserved_usd"])==Decimal(refusal["projected_committed_usd"])
        assert Decimal(refusal["projected_committed_usd"])>Decimal(refusal["cap_usd"])
        assert len(refusal["request_sha256"])==len(refusal["policy_sha256"])==64
        assert refusal["model_request_sent"] is False and refusal["physical_attempt_created"] is False
        assert "后续中性输入" not in json.dumps(refusal,ensure_ascii=False)
        persisted=json.loads(app.run_path.read_text(encoding="utf-8"))
        assert persisted==app._run and persisted["turns"][-1]["status"]=="failed"
        assert not app._storage_failed and not app._held_run_integrity_failed
    finally:app.close()


@pytest.mark.parametrize("mutation",[False,True])
def test_nonempty_business_preserved_and_halfcommit_rejected(tmp_path,monkeypatch,mutation):
    from aria.engine.model.persona import EpisodicMemoryEntry
    from aria.voice.held_input import business
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        engine,ctx=app._sessions[app._settled_owner]
        engine.persona.episodic_memory.append(EpisodicMemoryEntry(id="neutral-memory",summary="中性软件保留记录。"))
        app.store.save(engine.persona)
        before=business(app)
        if mutation:
            original=tls_dispatch.open_observed
            def changed(*args,**kwargs):
                try:return original(*args,**kwargs)
                finally:
                    engine.persona.episodic_memory[-1].summary="模拟半提交变化。"
                    app.store.save(engine.persona)
            monkeypatch.setattr(tls_dispatch,"open_observed",changed)
            with pytest.raises(ValueError,match="halfcommit"):
                app.submit("中性软件输入。","t1",sid)
            assert not app.state()["held_new_input"]["available"]
            assert app.call_guard.snapshot()["stopped"] and len(calls)==1
        else:
            app.submit("中性软件输入。","t1",sid)
            assert consent(app,sid,1)["ok"] and business(app)==before
    finally:app.close()


@pytest.mark.parametrize("kind",["remote","timeout"])
def test_pending_checker_network_shaped_exception_never_qualifies(tmp_path,monkeypatch,kind):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        def opened(request,**kwargs):
            calls.append(json.loads(request.data));trace=kwargs["trace"]
            trace.preparation_recovery.bind_request_headers(request)
            trace.preparation_recovery.before_preparation()
            for event in ("connection_preparation_started","connection_preparation_completed","tls_completed",
                          "application_request_started","application_write_attempted","request_write_completed"):trace._mark(event)
            def broken():
                if kind=="remote":raise http.client.RemoteDisconnected("local checker")
                raise TimeoutError("local checker")
            app.call_guard._held_check_pending(broken)
        monkeypatch.setattr(tls_dispatch,"open_observed",opened)
        result=app.submit("中性软件输入。","t1",sid)
        assert not result["ok"] and not app.state()["held_new_input"]["available"]
        assert app.call_guard.snapshot()["stopped"] and len(calls)==1
    finally:app.close()


def test_parallel_duplicate_owner_actions_and_late_old_submit_do_not_replay(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid)
        failure=app.state()["held_new_input"]["failure_id"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(app.continue_new_input,"same-action",failure,sid,confirmed=True) for _ in range(3)]
            results=[f.result() for f in futures]
        assert all(r["ok"] for r in results) and sum(r.get("duplicate",False) for r in results)==2
        assert app.submit("后续中性输入。","t2",sid)["ok"]
        assert app.submit("中性软件输入。","t1",sid)["duplicate"] and len(calls)==3
        assert len(app.call_guard.held_records())==1
    finally:app.close()


def test_expired_unfinished_session_has_no_new_paid_dispatch_or_extended_deadline(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    app.submit("中性软件输入。","t1",sid)
    expiry=app.store.load(app._settled_owner).pending_session.expires_at
    app.close()
    app2=MinimalWebApp(tmp_path,clients=make_clients(),model_max_requests=10,model_stop_cost_usd="2",model_reservation_policy=prices())
    try:
        monkeypatch.setattr(app2,"_checkpoint_is_current",lambda pending:False)
        assert app2.state()["session_id"] is None
        assert not app2.submit("中性软件输入。","t2",sid)["ok"] and len(calls)==1
        retained=app2.store.load(app2._settled_owner).pending_session
        assert retained is None or retained.expires_at==expiry
    finally:app2.close()


def test_before_business_snapshot_failure_persistently_stops_without_dispatch(tmp_path,monkeypatch):
    import aria.voice.held_input as module
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        def broken(*args):raise OSError("neutral before snapshot failure")
        monkeypatch.setattr(module,"business",broken)
        with pytest.raises(OSError):app.submit("中性软件输入。","t1",sid)
        assert app.state()["storage_failed"] and app.call_guard.snapshot()["stopped"] and calls==[]
    finally:app.close()


def test_delivered_invitation_consumed_and_existing_preference_unchanged(tmp_path,monkeypatch):
    from tests.test_stated_preferences import _profile
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        engine,ctx=app._sessions[app._settled_owner]
        persona=engine.persona;persona.stated_preferences=deepcopy(_profile(confirmed=False).stated_preferences)
        app.store.save(persona)
        invitation=app.preference_invitation(persona.stated_preferences[0].id,sid)
        assert invitation["ok"]
        value=invitation["preference_invitation"]
        assert app.preference_invitation_ack(value["id"],value["question_sha256"],sid)["ok"]
        before=[p.model_dump(mode="json") for p in persona.stated_preferences]
        result=app.submit("中性软件确认。","t1",sid)
        assert not result["ok"] and result["held_unreceived_failure"]
        assert app.state()["preference_invitation"] is None
        assert consent(app,sid,1)["ok"]
        assert [p.model_dump(mode="json") for p in app.store.load(app._settled_owner).stated_preferences]==before
        assert app.state()["preference_invitation"] is None and len(calls)==1
    finally:app.close()


@pytest.mark.parametrize("accepted",[False,True])
@pytest.mark.parametrize("field",["run_id","response","missing","invalid"])
def test_actual_persisted_run_mismatch_stops_before_confirm_and_next_input(tmp_path,monkeypatch,accepted,field):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid)
        failure=app.state()["held_new_input"]["failure_id"]
        if accepted:assert consent(app,sid,1)["ok"]
        original=deepcopy(app.call_guard.snapshot()["attempts"])
        held=app.call_guard.snapshot()["reserved_cost_usd"]
        raw=app.run_path.read_bytes();value=json.loads(raw)
        if field=="run_id":value["run_id"]="externally-changed-run"
        elif field=="response":value["turns"][0]["response"]="外部改动的软件字段。"
        if field=="missing":app.run_path.unlink();changed=None
        else:
            changed=b"{" if field=="invalid" else json.dumps(value,ensure_ascii=False).encode()
            app.run_path.write_bytes(changed)
        if not accepted:
            assert not app.continue_new_input("a2",failure,sid,confirmed=True)["ok"]
        result=app.submit("下一条中性输入。","t2",sid)
        assert not result["ok"] and len(calls)==1
        state=app.call_guard.snapshot()
        assert state["stopped"] and state["attempts"]==original and state["reserved_cost_usd"]==held
        assert app.run_path.read_bytes()==changed if changed is not None else not app.run_path.exists()
    finally:close_corrupt_fixture(app)


def test_guard_prewrite_rechecks_persisted_run_after_normal_pending_save(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid);assert consent(app,sid,1)["ok"]
        before=app.call_guard.snapshot()["reserved_cost_usd"]
        original=app.call_guard._check_candidate_pending_original
        def changed(*args,**kwargs):
            result=original(*args,**kwargs)
            disk=json.loads(app.run_path.read_text(encoding="utf-8"))
            disk["run_id"]="changed-at-actual-prewrite"
            app.run_path.write_text(json.dumps(disk),encoding="utf-8")
            return result
        monkeypatch.setattr(app.call_guard,"_check_candidate_pending_original",changed)
        with pytest.raises(RuntimeError,match="minimal_run_store_failed"):
            app.submit("下一条中性输入。","t2",sid)
        assert app.call_guard.snapshot()["stopped"]
        rows=app.call_guard.snapshot()["attempts"]
        assert len(rows)==2 and rows[0]["reservation"]["reserved_cost_usd"]==before
        assert rows[1]["transport_events"]["events"].get("request_write_completed") is not True
        assert json.loads(app.run_path.read_text(encoding="utf-8"))["run_id"]=="changed-at-actual-prewrite"
    finally:close_corrupt_fixture(app)


@pytest.mark.parametrize("accepted",[False,True])
@pytest.mark.parametrize("action_id",["action-1","late-new-action"])
def test_late_erased_continuation_cannot_halt_or_unlock_new_run(tmp_path,monkeypatch,accepted,action_id):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit("中性软件输入。","t1",sid)
        failure=app.state()["held_new_input"]["failure_id"]
        if accepted:assert consent(app,sid,1)["ok"]
        app.forget_run()
        before=deepcopy(app.call_guard.snapshot())
        result=app.continue_new_input(action_id,failure,sid,confirmed=True)
        assert not result["ok"] and not app._storage_failed
        after=app.call_guard.snapshot()
        for key in ("calls_used","unknown_cost_requests","reserved_cost_usd","stopped","stop_reason","attempts"):
            assert after[key]==before[key]
        current=app.start()["state"]["session_id"]
        result=app.submit("中性软件输入。","new-turn",current)
        assert result["ok"] is accepted and len(calls)==(3 if accepted else 1)
    finally:app.close()
