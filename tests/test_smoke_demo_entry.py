"""End-to-end local smoke plumbing; scripted labels are not model evaluation."""
import importlib.util
from pathlib import Path
import sys
import socket


def load_runner():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    import smoke_demo
    return smoke_demo


def story():
    sessions=[]
    for s in range(3):
        key=f'S{s+1}'
        ops=[{'op':'start_session','profile_policy':'new_empty_profile' if s==0 else 'continue'}]
        if s==1: ops.append({'op':'invite_preference','source_review':'R1','target_slot':'P1'})
        if s==2: ops.append({'op':'revoke_preference','target_slot':'P1'})
        for t in range(11):
            text=('聊天时先听我说完，一次最多问一个问题。' if s==0 and t==0 else f'软件流程输入{s+1}-{t+1}。')
            op={'op':'answer_preference' if s==1 and t==0 else 'say','turn_id':f'T{t+1}','text':text}
            if op['op']=='answer_preference': op['target_slot']='P1'
            ops.append(op)
        ops.append({'op':'end_session_review','bind_review_as':f'R{s+1}'})
        sessions.append({'session_id':key,'operations':ops})
    return {'repeat_id':'software','sessions':sessions}


def test_one_command_path_runs_all_sessions_and_real_local_preference_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network forbidden')))
    runner=load_runner()
    from smoke_fixture import fixture_clients
    from aria.voice.minimal import MinimalWebApp
    sample=runner.select_story(story())
    output=tmp_path/'output';output.mkdir()
    app=MinimalWebApp(tmp_path/'product',clients=fixture_clients(sample))
    try:
        result=runner.run_story(app,sample,output,fixture=True)
        assert result['evidence_class']=='software_fixture' and result['scored'] is False
        assert result['operations_completed'] and result['turns']==33 and result['sessions']==3
        assert result['preferences'][0]['status']=='withdrawn'
        assert app.store.load('fictional-chen').stated_preferences[0].status=='withdrawn'
        assert (output/'summary.json').exists() and (output/'replay.html').exists()
        assert sum(c.calls for c in (app.clients.online,app.clients.memory,app.clients.reflection))==39
    finally:
        app.close()


def test_prelude_renderer_displays_exact_product_frame_before_ack(tmp_path,capsys):
    runner=load_runner()
    event={'type':'safety_prelude','turn_id':'T1','session_id':'S1',
           'state':{'turns':[{'turn_id':'T1','protection_prelude':{'protection':'实际保护正文。'}}]}}
    assert runner.display_prelude(tmp_path,'T1',event) is True
    assert capsys.readouterr().out=='实际保护正文。\n'
    assert runner.read(tmp_path/'prelude-T1.json')==event


def test_public_completions_are_saved_separately_for_all_roles(tmp_path):
    runner=load_runner()
    from smoke_fixture import fixture_clients
    clients=fixture_clients(story())
    runner.record_model_outputs(clients,tmp_path)
    for client in (clients.online,clients.memory,clients.reflection,clients.safety_review):
        value=client.complete([{'role':'user','content':'软件调用'}])
        assert isinstance(value,str)
    rows=[runner.read(p) for p in sorted(tmp_path.glob('model-completion-*.json'))]
    assert [r['role'] for r in rows]==['online_fast','memory','reflection','safety_review']
    assert all(r['status']=='returned' and r['public_output'] for r in rows)


def test_recovery_binding_preserves_prior_count_and_rejects_source_drift(tmp_path):
    import json
    import sqlite3
    import pytest
    runner=load_runner()
    floor=tmp_path/'floor.json'
    floor.write_text(json.dumps({'committed_usd':'5.93','task_incremental_committed_usd':'.006'}))
    ledger=tmp_path/'calls.sqlite3'
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE attempts(sequence INTEGER, record TEXT)')
        db.execute('INSERT INTO attempts VALUES(1,?)',(json.dumps({
            'request_id':'original','status':'failed','http_status':429,'usage':{}}),))
    ref=lambda p:{'path':str(p),'sha256':runner.digest(p.read_bytes())}
    prior=tmp_path/'prior.json'
    prior.write_text(json.dumps({'financial_floor':ref(floor),'task_ledgers':[ref(ledger)],'unknown_429_count':1}))
    source=tmp_path/'source.py';source.write_text('original')
    contract,check,_=runner.recovery_binding(prior,[source],prior_occupied='5.93',
        prior_new_spend='.006',stop_path=tmp_path/'STOP')
    assert contract['prior_unknown_429_count']==1 and check() is True
    source.write_text('changed')
    with pytest.raises(ValueError,match='source_or_authority_changed'):check()
    source.write_text('original');(tmp_path/'STOP').touch()
    with pytest.raises(ValueError,match='requested_stop'):check()
    with pytest.raises(ValueError,match='budget_mismatch'):
        runner.recovery_binding(prior,[source],prior_occupied='5',prior_new_spend='.006',stop_path=tmp_path/'unused')
