"""Real installed App/guard with neutral local HTTP bytes, zero live calls."""
from copy import deepcopy
from pathlib import Path
import io,json,socket,sqlite3,threading
import pytest
from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.max_release import tls_dispatch,reflection_wrapper as wrapper
from aria.engine.policy.max_release.reflection_failure import FIELD
from aria.engine.policy.max_release.validation_support import read,ref
from tests.test_installable_max_release import make_clients,prices,response,Response,malformed_compact_slot

BARE='{"protocol_version":"2.0","reviews":[],"changes":[]}'

def failure_body(kind):
    if kind=='scope_slot':
        if make_clients().reflection.reflection_return_format=='json_schema_v4':
            # V4 deliberately isolates an illegal item slot; it no longer
            # rejects the whole reply. Exercise the same failure transaction
            # with an invalid collection instead. V4 item isolation itself
            # is covered by the compact codec and actual updater regressions.
            return '{"protocol_version":"2.0","reviews":{},"changes":[]}'
        return malformed_compact_slot('changes','scope',['条件一','条件二'])
    if kind=='version':
        return '{"protocol_version":"v3","contract":"prompt_contract 2.5","reviews":[],"changes":[]}'
    assert kind=='envelope'
    return BARE[:-1]+',"unexpected":1}'

@pytest.fixture(autouse=True, params=['sonnet', 'deepseek'])
def offline(monkeypatch, request):
    def no(*a,**k):raise AssertionError('network or credentials forbidden')
    monkeypatch.setattr(socket,'create_connection',no);monkeypatch.setattr(socket.socket,'connect',no)
    monkeypatch.setattr('aria.runtime.openrouter_api_key',no)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key',no)
    monkeypatch.setattr('aria._env.load_local_env',no)
    monkeypatch.setattr(tls_dispatch,'open_observed',no)
    if request.param == 'deepseek':
        from tests import test_deepseek_offline_route as deepseek
        monkeypatch.setattr(__name__+'.make_clients',deepseek.clients)
        monkeypatch.setattr(__name__+'.prices',deepseek.policy)
        monkeypatch.setattr(__name__+'.response',deepseek.response)

def setup(tmp_path,monkeypatch,raw=None,unknown=False,*,fault=None,failure_role='reflection'):
    clients=make_clients();calls=[]
    def opened(req,**kw):
        wire=json.loads(req.data)
        role=('reflection' if wire['max_tokens']==2800 else 'memory' if wire['max_tokens']==4000
              else 'safety_review' if wire['max_tokens']==1024 else 'online_fast')
        calls.append(role);data=response({'online_fast':clients.online,'memory':clients.memory,
                                        'reflection':clients.reflection,'safety_review':clients.safety_review}[role])
        if role=='safety_review':
            from tests.test_safety_review_v2 import answer
            data['choices'][0]['message']['content']=json.dumps(answer(wire=wire))
        if role=='online_fast' and clients.online.model.startswith('qwen/'):
            data['usage']['total_tokens']=4520
        if role==failure_role:
            data['choices'][0]['message']['content']=raw if raw is not None else BARE[:-1]+',"unexpected":1}'
            if unknown:data['usage'].pop('cost')
            if fault=='model':data['model']='other/model'
            if fault=='provider':data['provider']='Other'
            if fault=='length':
                data['choices'][0]['finish_reason']='length'
                data['usage'].update(completion_tokens=4000,total_tokens=data['usage']['prompt_tokens']+4000)
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        result=Response(data,wire.get('stream',False))
        if role==failure_role and fault=='body':
            result.headers.replace_header('Content-Length',str(len(result.body.getvalue())+20))
        if role=='reflection' and fault in {'done','usage'}:
            raw_stream=result.body.getvalue()
            if fault=='done':raw_stream=raw_stream.replace(b'data: [DONE]\n\n',b'')
            else:raw_stream=b'\n\n'.join(part for part in raw_stream.split(b'\n\n') if b'"usage"' not in part)
            result.body=io.BytesIO(raw_stream)
            result.headers.replace_header('Content-Length',str(len(raw_stream)))
        return result
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=30,
        model_stop_cost_usd='1.00',model_reservation_policy=prices())
    pace(app.call_guard)
    return app,calls

def pace(guard):
    clock=[1000.];guard._pacing_clock=lambda:clock[0]
    guard._pacing_sleep=lambda d:clock.__setitem__(0,clock[0]+d)

def one_session(app,index):
    sid=app.start()['state']['session_id']
    for i in range(3):assert app.submit('中性软件输入。',f'op{index+i}',sid)['ok']
    assert app.seal(sid)['ok']
    return app.reflect(f'op{index+3}',wait=True)

def reopen(path):
    app=MinimalWebApp(path,clients=make_clients(),model_max_requests=30,
        model_stop_cost_usd='1.00',model_reservation_policy=prices())
    pace(app.call_guard);return app

def test_public_repeated_reflection_F_then_next_session(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch)
    try:
        for index in (1,5):
            assert not one_session(app,index)['ok']
            state=app.state();assert not state['requires_review'] and not app.call_guard.snapshot()['stopped']
            review=state['reviews'][-1]
            assert review['status']=='failed' and review['requires_review']
            assert review['request_id'] in state['settled_reflection_failure_ids']
            profile=app.store.load(app._settled_owner)
            assert not profile.offline_sessions and profile.offline_session_audit[-1].outcome=='reflection_failed'
            assert profile.offline_session_audit[-1].memory_completed and not profile.offline_session_audit[-1].reflection_completed
        assert app.start()['ok'] and calls==(['online_fast','safety_review']*3+['memory','reflection'])*2
    finally:app.close()

def test_public_restart_rechecks_failed_receipt_without_replay(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch)
    one_session(app,1);prior=deepcopy(app._run['reviews']);app.close()
    app=reopen(tmp_path)
    try:
        assert not app.state()['requires_review'] and app._run['reviews']==prior
        assert app.start()['ok'] and calls==['online_fast','safety_review']*3+['memory','reflection']
    finally:app.close()

@pytest.mark.parametrize('body,fault,error',[
    ('{"episodes":[','length','invalid_or_truncated_response'),
    ('{"episodes":[',None,'memory_schema_invalid'),
    ('',None,'invalid_response_message'),
])
def test_memory_content_failure_seals_failed_job_and_next_session_survives(
        tmp_path,monkeypatch,body,fault,error):
    app,calls=setup(tmp_path,monkeypatch,body,fault=fault,failure_role='memory')
    try:
        for index in (1,5):
            assert not one_session(app,index)['ok']
            state=app.state();assert not state['requires_review']
            ledger=app.call_guard.snapshot()
            assert not ledger['stopped'] and ledger['unknown_cost_requests']==0
            row=ledger['attempts'][-1]
            assert row['role']=='memory' and row['status']=='failed' and row['validation_error']==error
            assert row['usage']['cost']>0 and row['physical_attempt']==1
            profile=app.store.load(app._settled_owner)
            assert not profile.offline_sessions
            audit=profile.offline_session_audit[-1]
            assert audit.outcome=='memory_failed' and not audit.memory_completed and not audit.reflection_completed
            assert not profile.episodic_memory and not profile.stated_preferences and not profile.person_hypotheses
            receipt=read(app._settled_reflections/(state['reviews'][-1]['request_id']+'.json'))
            assert [turn['user'] for turn in receipt['original_job']['turns']]==['中性软件输入。']*3
            assert read(row['public_first_output']['path'])['public_first_output'][0]['content']==body
        assert calls==(['online_fast','safety_review']*3+['memory'])*2
    finally:app.close()
    app=reopen(tmp_path)
    try:
        assert not app.state()['requires_review'] and not app.call_guard.snapshot()['stopped']
        sid=app.start()['state']['session_id']
        assert app.submit('后续中性软件输入。','after-memory-failure',sid)['ok']
        assert calls.count('memory')==2 and 'reflection' not in calls
    finally:app.close()

@pytest.mark.parametrize('fault',['unknown','model','provider','body'])
def test_memory_unknown_or_incomplete_receipt_keeps_halt_and_job(tmp_path,monkeypatch,fault):
    app,calls=setup(tmp_path,monkeypatch,'{"episodes":[',unknown=fault=='unknown',
        fault=None if fault=='unknown' else fault,failure_role='memory')
    try:
        assert not one_session(app,1)['ok']
        assert app.call_guard.snapshot()['stopped'] and app.state()['requires_review']
        profile=app.store.load(app._settled_owner)
        assert len(profile.offline_sessions)==1 and not profile.offline_session_audit
        assert not profile.episodic_memory and not app.start()['ok']
        assert calls==['online_fast','safety_review']*3+['memory']
    finally:app.close()

def test_memory_failed_forget_preserves_charge_and_never_replays(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch,'{"episodes":[',fault='length',failure_role='memory')
    one_session(app,1);before=app.call_guard.snapshot();app.forget_run()
    after=app.call_guard.snapshot()
    assert not after['stopped'] and before['attempts']==after['attempts']
    assert before['known_cost_usd']==after['known_cost_usd'] and after['unknown_cost_requests']==0
    app.close();app=reopen(tmp_path)
    try:
        assert app.start()['ok'] and not app.call_guard.snapshot()['stopped']
        assert calls==['online_fast','safety_review']*3+['memory']
    finally:app.close()

def test_original_story_driver_continues_after_settled_memory_failure(tmp_path,monkeypatch):
    from scripts.run_story import run_story
    from tests.test_run_story import small_story,records
    created=[]
    def factory(path):
        app,calls=setup(path,monkeypatch,'{"episodes":[',fault='length',failure_role='memory')
        created.append((app,calls));return app
    out=tmp_path/'original-driver'
    summary=run_story(small_story(),1,out,app_factory=factory)
    assert summary['sessions']==2 and summary['turns']==6 and summary['unexecuted_turns']==0
    assert summary['failed_reviews']==2 and summary['stop_reason'] is None
    assert len(created)==1 and created[0][1]==(['online_fast','safety_review']*3+['memory'])*2
    turns=[r for r in records(out) if r['turn_id']]
    assert len(turns)==6 and all(r['ok'] for r in turns)

@pytest.mark.parametrize('stage',['memory','reflection'])
def test_authorized_automatic_review_seals_content_failure_without_changing_permission(tmp_path,monkeypatch,stage):
    from tests.test_minimal_auto_review import enable,wait_worker
    app,calls=setup(tmp_path,monkeypatch,'{"episodes":[' if stage=='memory' else failure_body('envelope'),
        fault='length' if stage=='memory' else None,failure_role=stage)
    try:
        enable(app,2)
        sid=app.start()['state']['session_id']
        for index in range(3):assert app.submit('中性软件输入。',f'auto{index}',sid)['ok']
        assert app.seal(sid)['ok']
        state=wait_worker(app)
        assert not state['requires_review'] and not app.call_guard.snapshot()['stopped']
        assert state['auto_review']['enabled'] and state['auto_review']['today_started']==1
        assert state['auto_review']['events'][0]['status']=='failed'
        assert app.store.load(app._settled_owner).offline_session_audit[-1].outcome==stage+'_failed'
        count=len(calls);assert not app.tick_auto_review()['started'] and len(calls)==count
        sid=app.start()['state']['session_id']
        assert app.submit('新的中性软件输入。','auto-next-session',sid)['ok']
    finally:app.close()

@pytest.mark.parametrize('stage',['memory','reflection'])
def test_cancelled_waiting_delivery_does_not_become_offline_halfcommit(tmp_path,monkeypatch,stage):
    app,calls=setup(tmp_path,monkeypatch,'{"episodes":[' if stage=='memory' else failure_body('envelope'),
        fault='length' if stage=='memory' else None,failure_role=stage)
    entered,release=threading.Event(),threading.Event();opened=tls_dispatch.open_observed
    def delayed(request,**kwargs):
        if json.loads(request.data)['max_tokens']==(4000 if stage=='memory' else 2800):
            entered.set();assert release.wait(8)
        return opened(request,**kwargs)
    monkeypatch.setattr(tls_dispatch,'open_observed',delayed);worker=None
    try:
        sid=app.start()['state']['session_id']
        for index in range(3):assert app.submit('原会话软件输入。',f'cancel-old-{index}',sid)['ok']
        assert app.seal(sid)['ok'] and app.reflect('cancel-background')['ok']
        worker=app._review_thread;assert entered.wait(5)
        new_sid=app.start()['state']['session_id']
        result=app.submit('保存但取消等待发送。','cancel-new',new_sid,on_waiting_input=lambda frame:False)
        assert not result['ok'] and result['request_sent'] is False
        release.set();worker.join(8);assert not worker.is_alive()
        assert not app.call_guard.snapshot()['stopped'] and not app.state()['requires_review']
        failed=app.state()['turns'][-1]
        assert failed['turn_id']=='cancel-new' and failed['request_sent'] is False
        assert failed['source']=='local_request_rejected' and failed['text']=='保存但取消等待发送。'
        assert app.submit('以后新的软件输入。','cancel-next',new_sid)['ok']
        assert calls.count('online_fast')==4 and calls.count(stage)==1
    finally:
        release.set()
        if worker:worker.join(8)
        app.close()

@pytest.mark.parametrize('fault',['business','permission','terminal_write'])
def test_memory_halfcommit_or_terminal_storage_error_never_unlocks(tmp_path,monkeypatch,fault):
    import aria.voice.settled_reflection as integration
    app,calls=setup(tmp_path,monkeypatch,'{"episodes":[',fault='length',failure_role='memory')
    original=app._run_review_original
    def changed(*args):
        original(*args)
        if fault=='business':
            profile=app.store.load(app._settled_owner);profile.disc.dominance=.99;app.store.save(profile)
        elif fault=='permission':
            with sqlite3.connect(app.run_dir/'care.sqlite3') as db:
                db.execute('CREATE TABLE synthetic_halfcommit(value INTEGER)')
    if fault!='terminal_write':monkeypatch.setattr(app,'_run_review_original',changed)
    else:
        prior=integration.write_once
        def failed(path,value):
            if str(path).endswith('-after-terminal.json'):raise OSError('synthetic storage failure')
            return prior(path,value)
        monkeypatch.setattr(integration,'write_once',failed)
    try:
        with pytest.raises((ValueError,OSError)):one_session(app,1)
        assert app.call_guard.snapshot()['stopped'] and not app.start()['ok']
        assert calls==['online_fast','safety_review']*3+['memory']
    finally:app.close()

def test_single_terminal_fence_uses_original_domain(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch,'说明。\n```json\n'+BARE+'\n```')
    try:
        old_format=app.clients.reflection.reflection_return_format=='json_schema_v3'
        assert one_session(app,1)['ok'] is old_format
        assert app.store.load(app._settled_owner).offline_session_audit[-1].reflection_completed is old_format
        assert not app.call_guard.snapshot()['stopped']
    finally:app.close()

@pytest.mark.parametrize('body_kind',['envelope','scope_slot'])
def test_unknown_reflection_never_unlocks_even_after_forget(tmp_path,monkeypatch,body_kind):
    app,calls=setup(tmp_path,monkeypatch,failure_body(body_kind),unknown=True)
    try:
        assert not one_session(app,1)['ok'];before=app.call_guard.snapshot()
        assert before['stopped'] and before['unknown_cost_requests']==1
        app.forget_run()
        assert app.call_guard.snapshot()['stopped'] and app.call_guard.snapshot()['unknown_cost_requests']==1
        sid=app.start()['state']['session_id']
        assert not app.submit('中性软件输入。','afterforget',sid)['ok'] and len(calls)==8
    finally:app.close()

def test_forget_keeps_verified_failed_accounting_without_private_snapshots(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch)
    one_session(app,1);before=app.call_guard.snapshot();directory=app._settled_reflections
    app.forget_run()
    assert not directory.exists() and not app.state()['turns'] and not app.state()['reviews']
    after=app.call_guard.snapshot()
    assert after['attempts']==before['attempts'] and after['calls_used']==8
    assert not after['stopped'] and after['known_cost_usd']==before['known_cost_usd']
    markers=app.call_guard._db.execute('SELECT record FROM erased_reflection_failures').fetchall()
    assert len(markers)==1 and '中性软件输入' not in markers[0][0]
    app.close();app=reopen(tmp_path)
    try:
        assert not app.state()['requires_review'] and app.start()['ok']
        assert app.call_guard.snapshot()['calls_used']==8 and len(calls)==8
    finally:app.close()

@pytest.mark.parametrize('field',['job','terminal','parser'])
def test_tampered_disposition_fails_closed(tmp_path,monkeypatch,field):
    app,calls=setup(tmp_path,monkeypatch);one_session(app,1)
    path=next(app._settled_reflections.glob('*.json'));saved=path.read_bytes();v=read(path)
    try:
        if field=='job':v['original_job']['expires_at']='2099-01-01T00:00:00+00:00'
        elif field=='parser':v['ledger_row'][FIELD]['contract']['version']='weaker'
        else:
            terminal=read(v['after_terminal_profile']['path']);terminal['person_model_state']['sessions_completed']+=1
            new=tmp_path/'changed.json';new.write_text(json.dumps(terminal),encoding='utf8');v['after_terminal_profile']=ref(new)
        path.write_text(json.dumps(v),encoding='utf8')
        state=app.state()
        assert state['requires_review'] and not state['settled_reflection_failure_ids']
        assert not app.start()['ok']
        assert len(calls)==8
    finally:path.write_bytes(saved);app.close()

def test_ui_marks_reflection_failure_without_claiming_success():
    value=Path('aria/assets/voice/minimal.html').read_text(encoding='utf8')
    assert 'settled_reflection_failure_ids' in value and "row?.status==='failed'" in value
    assert 'label(review?.status || offline.status)' in value

@pytest.mark.parametrize('stage',['before','disposition','terminal'])
@pytest.mark.parametrize('body_kind',['envelope','scope_slot'])
def test_durable_proof_failure_cannot_unlock(tmp_path,monkeypatch,stage,body_kind):
    import aria.voice.settled_reflection as integration
    app,calls=setup(tmp_path,monkeypatch,failure_body(body_kind));original=integration.write_once
    def failing(path,value):
        name=Path(path).name
        if (stage=='before' and name.endswith('-before.json')
            and value['profile']['offline_sessions'][0]['active_stage']=='reflection'
            or stage=='terminal' and name.endswith('-after-terminal.json')
            or stage=='disposition' and value.get('schema')=='aria_settled_reflection_failure_v1'):
            raise OSError('local test storage failure')
        return original(path,value)
    monkeypatch.setattr(integration,'write_once',failing)
    try:
        try:one_session(app,1)
        except (OSError,ValueError):pass
        assert app.call_guard.snapshot()['stopped']
        assert calls==['online_fast','safety_review']*3+['memory']+([] if stage=='before' else ['reflection'])
        assert not app.start()['ok']
    finally:app.close()

@pytest.mark.parametrize('stage',['begin','delete','complete'])
def test_forget_failure_does_not_unlock_paid_guard(tmp_path,monkeypatch,stage):
    import aria.voice.settled_reflection as integration
    app,calls=setup(tmp_path,monkeypatch);one_session(app,1)
    before=app.call_guard.snapshot()
    def failed(*a,**k):raise OSError('local test storage failure')
    with monkeypatch.context() as patch:
        if stage=='delete':patch.setattr(integration.shutil,'rmtree',failed)
        else:patch.setattr(app.call_guard,stage+'_reflection_forget',failed)
        with pytest.raises(OSError):app.forget_run()
    # Pending tombstones are themselves an unresolved durable write, never an exemption.
    try:assert app.call_guard.snapshot()['stopped']
    except ValueError:pass
    assert app.call_guard._db.execute('SELECT count(*) FROM attempts').fetchone()[0]==8
    assert len(calls)==8
    app.close()
    app=reopen(tmp_path)
    try:
        try:
            sid=app.start()['state']['session_id']
            assert not app.submit('中性软件输入。','noafterforget',sid)['ok']
        except (ValueError,KeyError):pass
        assert len(calls)==8
    finally:app.close()

@pytest.mark.parametrize('change',['delete','ledger','pending'])
def test_erasure_marker_is_verified_on_restart(tmp_path,monkeypatch,change):
    app,calls=setup(tmp_path,monkeypatch);one_session(app,1);app.forget_run()
    row=app.call_guard._db.execute('SELECT request_id,record FROM erased_reflection_failures').fetchone()
    if change=='delete':app.call_guard._db.execute('DELETE FROM erased_reflection_failures')
    else:
        marker=json.loads(row[1]);marker['ledger_id' if change=='ledger' else 'state']='foreign' if change=='ledger' else 'pending'
        app.call_guard._db.execute('UPDATE erased_reflection_failures SET record=?',(json.dumps(marker),))
    app.call_guard._db.commit();app.close();app=reopen(tmp_path)
    try:
        try:
            sid=app.start()['state']['session_id']
            assert not app.submit('中性软件输入。','markerblocked',sid)['ok']
        except (ValueError,KeyError):pass
        assert len(calls)==8
    finally:app.close()

def test_forget_during_reflection_does_not_revive_old_session(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch,BARE[:-1]+',"unexpected":"PRIVATE_INFLIGHT_NEUTRAL"}')
    sid=app.start()['state']['session_id']
    for i in range(3):assert app.submit('中性软件输入。',f'pre{i}',sid)['ok']
    assert app.seal(sid)['ok']
    original=tls_dispatch.open_observed;entered=threading.Event();release=threading.Event()
    def delayed(req,**kw):
        if json.loads(req.data)['max_tokens']==2800:
            entered.set();assert release.wait(5)
        return original(req,**kw)
    monkeypatch.setattr(tls_dispatch,'open_observed',delayed)
    try:
        assert app.reflect('inflight',wait=False)['ok']
        worker=app._review_thread;assert entered.wait(5)
        app.forget_run();release.set();worker.join(5);assert not worker.is_alive()
        state=app.state()
        assert not state['turns'] and not state['reviews'] and not state['sessions']
        profile=app.store.load(app._settled_owner)
        assert profile is None
        assert app.call_guard.snapshot()['calls_used']==8 and app.call_guard.snapshot()['stopped']
        assert not app._settled_reflections.exists()
        assert not [p for p in tmp_path.rglob('*') if p.is_file() and b'PRIVATE_INFLIGHT_NEUTRAL' in p.read_bytes()]
    finally:release.set();app.close()

@pytest.mark.parametrize('unknown',[False,True])
def test_forget_erases_content_without_dropping_fees_or_stops(tmp_path,monkeypatch,unknown):
    marker='PRIVATE_FORGET_NEUTRAL'
    app,calls=setup(tmp_path,monkeypatch,BARE[:-1]+',"unexpected":"'+marker+'"}',unknown)
    one_session(app,1);before=app.call_guard.snapshot();app.forget_run();app.forget_run()
    after=app.call_guard.snapshot()
    assert before['attempts']==after['attempts'] and before['known_cost_usd']==after['known_cost_usd']
    assert after['stopped'] is unknown
    assert not list(app.call_guard.public_dir.glob('*.json'))
    assert not list(app.call_guard.physical_dir.rglob('first-visible.json'))
    assert not list(app.call_guard.physical_dir.rglob('stream-public-frames.json'))
    assert not [p for p in tmp_path.rglob('*') if p.is_file() and marker.encode() in p.read_bytes()]
    app.close();app=reopen(tmp_path)
    try:
        assert app.call_guard.snapshot()['stopped'] is unknown
        assert app.call_guard.snapshot()['attempts']==before['attempts'] and len(calls)==8
    finally:app.close()

def test_erasure_delete_failure_retains_pending_fence_and_never_unlocks(tmp_path,monkeypatch):
    app,calls=setup(tmp_path,monkeypatch);one_session(app,1)
    original=Path.unlink
    def failed(path,*a,**k):
        if path.name=='first-visible.json':raise OSError('neutral erasure failure')
        return original(path,*a,**k)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',failed)
        with pytest.raises(OSError):app.forget_run()
    with pytest.raises(ValueError):app.call_guard.content_erasure.check()
    assert len(calls)==8 and app.call_guard._db.execute('SELECT count(*) FROM attempts').fetchone()[0]==8
    with pytest.raises(ValueError):app.close()


@pytest.mark.parametrize('raw,kind',[(BARE,'bare_json'),('```json\n'+BARE+'\n```','single_json_fence'),
    ('中性说明。\n```json\n'+BARE+'\n```','terminal_json_fence_after_prose')])
def test_actual_observer_accepts_same_reflection_receipt_as_client(tmp_path,monkeypatch,raw,kind):
    from aria.engine.policy.max_release.public_capture import record_public_model_outputs
    app,calls=setup(tmp_path/'product',monkeypatch,raw);output=tmp_path/'observation';output.mkdir()
    record_public_model_outputs(app.clients,output)
    try:
        version=app.clients.reflection.reflection_return_format
        strict_rejection=version=='json_schema_v4' and kind!='bare_json'
        assert one_session(app,1)['ok'] is not strict_rejection
        entries=[read(p) for p in output.glob('model-completion-*.json')]
        reflection=[v for v in entries if v['role']=='reflection'][0]
        assert reflection['provider_schema_applied'] is (version=='json_schema_v4')
        if strict_rejection:
            assert reflection['status']=='raised'
            failed=app.call_guard.snapshot()['attempts'][-1]
            assert failed['validation_error']=='reflection_schema_invalid'
            assert read(failed['public_first_output']['path'])['public_first_output'][0]['content']==raw
            assert not app.call_guard.snapshot()['stopped']
            assert app.store.load(app._settled_owner).offline_session_audit[-1].outcome=='reflection_failed'
            return
        assert reflection['status']=='returned'
        assert reflection['public_output_representation']=='canonical_decoded_'+version
        assert reflection['public_output_wrapper']['wrapper']==kind
        assert (reflection['public_output_wrapper']['prefix_characters']>0)==(kind=='terminal_json_fence_after_prose')
        assert calls==['online_fast','safety_review']*3+['memory','reflection']
        assert app.store.load(app._settled_owner).offline_session_audit[-1].reflection_completed
    finally:app.close()

@pytest.mark.parametrize('field,value',[('version','other'),('status','rejected'),('wrapper','invented'),
    ('canonical_sha256','0'*64),('serialization_body_sha256','0'*64),('raw_public_sha256','x'),
    ('prefix_sha256','invalid'),('prefix_characters',0),('prefix_characters',True),
    ('representation_only',False),('domain_content_repaired',True),('json_fields_repaired',True),
    ('source_or_identifier_repaired',True),('raw_public_sha256','0'*64),('prefix_sha256','0'*64),('prefix_characters',999)])
def test_actual_observer_rejects_changed_reflection_receipts(tmp_path,monkeypatch,field,value):
    from aria.engine.policy.max_release.public_capture import record_public_model_outputs
    raw=BARE if make_clients().reflection.reflection_return_format=='json_schema_v4' else '说明。\n```json\n'+BARE+'\n```'
    app,calls=setup(tmp_path/'product',monkeypatch,raw)
    client=app.clients.reflection;original=client.complete
    def changed(messages,**requirements):
        result=original(messages,**requirements)
        current=client.last_diagnostic['reflection_wrapper'][field]
        # A bare v4 body starts with zero prefix length; still mutate this
        # case, rather than accidentally testing an unchanged receipt.
        client.last_diagnostic['reflection_wrapper'][field]=(1 if type(value) is int and value==current else value)
        return result
    client.complete=changed;output=tmp_path/'observation';output.mkdir()
    record_public_model_outputs(app.clients,output)
    try:
        assert not one_session(app,1)['ok']
        reflection=[read(p) for p in output.glob('model-completion-*.json') if read(p)['role']=='reflection'][0]
        assert reflection['status']=='raised' and reflection['error_class']=='ValueError'
        assert not app.store.load(app._settled_owner).offline_session_audit
        assert app.call_guard.snapshot()['attempts'][-1]['status']=='received'
        assert app.call_guard.snapshot()['stopped'] and calls==['online_fast','safety_review']*3+['memory','reflection']
    finally:app.close()

def test_actual_observer_capture_failure_keeps_product_failed(tmp_path,monkeypatch):
    from aria.engine.policy.max_release import public_capture as capture
    app,calls=setup(tmp_path/'product',monkeypatch,BARE)
    output=tmp_path/'observation';output.mkdir();capture.record_public_model_outputs(app.clients,output)
    original=capture.write_once
    def fail(path,value):
        if value.get('role')=='reflection':raise OSError('software observer persistence failure')
        return original(path,value)
    monkeypatch.setattr(capture,'write_once',fail)
    try:
        assert not one_session(app,1)['ok']
        assert app.call_guard.snapshot()['attempts'][-1]['status']=='received'
        assert app.call_guard.snapshot()['stopped'] and not app.store.load(app._settled_owner).offline_session_audit
    finally:app.close()


@pytest.mark.parametrize('body_kind',['version','scope_slot'])
def test_complete_format_failure_preserves_profile_fee_and_allows_new_submission(tmp_path,monkeypatch,body_kind):
    from decimal import Decimal
    from aria.engine.model.persona import EpisodicMemoryEntry
    from tests.test_stated_preferences import _profile
    malformed=failure_body(body_kind)
    app,calls=setup(tmp_path,monkeypatch,malformed)
    # Persisted neutral software fixtures, not a claimed learned preference.
    seed=app._seed_persona(app._settled_owner)
    seed.stated_preferences=_profile().stated_preferences
    seed.episodic_memory=[EpisodicMemoryEntry(id='prior-software-fact',summary='已有软件夹具事实。')]
    app.store.save(seed)
    retained={k:seed.model_dump(mode='json')[k] for k in ('stated_preferences','episodic_memory','person_hypotheses')}
    try:
        assert not one_session(app,1)['ok']
        state=app.state();before=app.call_guard.snapshot()
        assert not state['requires_review'] and not before['stopped']
        failed=before['attempts'][-1]
        assert failed['status']=='failed' and failed['validation_error']=='reflection_schema_invalid'
        assert failed['failure_phase']=='receipt_validation' and failed['failure_code']=='reflection_schema_invalid'
        assert 'cost' in failed['usage'] and before['reserved_cost_usd']=='0'
        assert Decimal(before['known_cost_usd'])==sum(Decimal(str(r['usage']['cost'])) for r in before['attempts'])
        profile=app.store.load(app._settled_owner).model_dump(mode='json')
        assert all(profile[k]==v for k,v in retained.items())
        assert not profile['offline_sessions']
        assert profile['offline_session_audit'][-1]['outcome']=='reflection_failed'
        assert profile['offline_session_audit'][-1]['reflection_completed'] is False
        failed_review=deepcopy(state['reviews'][-1]);assert failed_review['status']=='failed'
        original_public=Path(failed['public_first_output']['path']).read_bytes()
        assert json.loads(original_public)['public_first_output'][0]['content']==malformed
        app.close();app=reopen(tmp_path)
        assert not app.state()['requires_review'] and app._run['reviews'][-1]==failed_review
        sid=app.start()['state']['session_id']
        assert app.submit('新的中性软件输入。','after-settled-reflection',sid)['ok']
        after=app.call_guard.snapshot()
        assert after['attempts'][:8]==before['attempts'] and after['calls_used']==10
        assert calls==['online_fast','safety_review']*3+['memory','reflection','online_fast','safety_review']
        assert Path(failed['public_first_output']['path']).read_bytes()==original_public
    finally:app.close()


def test_settled_reflection_cannot_switch_the_decoder_bound_to_its_request(tmp_path,monkeypatch):
    from aria.engine.policy.max_release.reflection_failure import validate_receipt_metadata
    app,calls=setup(tmp_path,monkeypatch)
    try:
        assert not one_session(app,1)['ok']
        row=deepcopy(app.call_guard.snapshot()['attempts'][-1])
        validate_receipt_metadata(row)
        version=app.clients.reflection.reflection_return_format
        replacement='json_schema_v3' if version=='json_schema_v4' else 'json_schema_v4'
        row[FIELD]['contract']=wrapper.contract(version=replacement)
        with pytest.raises(ValueError,match='reflection_validation_codec_changed'):
            validate_receipt_metadata(row)
        assert not app.call_guard.snapshot()['stopped'] and len(calls)==8
    finally:app.close()


@pytest.mark.parametrize('fault',['model','provider','done','usage'])
@pytest.mark.parametrize('body_kind',['envelope','scope_slot'])
def test_unverified_reflection_never_reaches_settled_app_terminal(tmp_path,monkeypatch,fault,body_kind):
    app,calls=setup(tmp_path,monkeypatch,failure_body(body_kind),fault=fault)
    try:
        assert not one_session(app,1)['ok']
        state=app.call_guard.snapshot()
        assert state['stopped'] and not app.start()['ok']
        assert not app.state()['settled_reflection_failure_ids']
        assert not app.store.load(app._settled_owner).offline_session_audit
        assert calls==['online_fast','safety_review']*3+['memory','reflection']
        assert state['calls_used']==8 and state['attempts'][-1]['status']=='failed'
    finally:app.close()


@pytest.mark.parametrize('body_kind',['envelope','scope_slot'])
def test_reflection_before_binding_mismatch_cannot_settle(tmp_path,monkeypatch,body_kind):
    app,calls=setup(tmp_path,monkeypatch,failure_body(body_kind));original=app.call_guard._finish
    def finish(row):
        if row['role']=='reflection':row[FIELD]['request_sha256']='0'*64
        return original(row)
    monkeypatch.setattr(app.call_guard,'_finish',finish)
    try:
        assert not one_session(app,1)['ok']
        assert app.call_guard.snapshot()['stopped'] and not app.start()['ok']
        assert not app.state()['settled_reflection_failure_ids']
        assert not app.store.load(app._settled_owner).offline_session_audit
        assert len(calls)==8
    finally:app.close()


@pytest.mark.parametrize('body_kind',['envelope','scope_slot'])
def test_reflection_business_halfcommit_cannot_settle(tmp_path,monkeypatch,body_kind):
    app,calls=setup(tmp_path,monkeypatch,failure_body(body_kind));original=app._run_review_original
    def halfcommit(*args):
        result=original(*args)
        profile=app.store.load(app._settled_owner)
        profile.person_model_state.sessions_completed+=1
        app.store.save(profile)
        return result
    monkeypatch.setattr(app,'_run_review_original',halfcommit)
    try:
        with pytest.raises(ValueError,match='reflection_business_halfcommit'):one_session(app,1)
        assert app.call_guard.snapshot()['stopped'] and not app.start()['ok']
        assert not app.state()['settled_reflection_failure_ids']
        assert not app.store.load(app._settled_owner).offline_session_audit
        assert len(calls)==8
    finally:app.close()


def test_new_qualification_does_not_unlock_an_already_halted_run(tmp_path,monkeypatch):
    from aria.engine.policy.max_release import reflection_failure
    app,calls=setup(tmp_path,monkeypatch)
    def reject(*args,**kwargs):raise ValueError('synthetic_prior_contract_rejection')
    with monkeypatch.context() as prior:
        prior.setattr(reflection_failure,'validate_receipt_metadata',reject)
        assert not one_session(app,1)['ok']
        before=app.call_guard.snapshot()
        assert before['stopped'] and before['stop_reason']=='model_or_storage_failed'
    app.close();app=reopen(tmp_path)
    try:
        assert app.call_guard.snapshot()['stopped'] and not app.start()['ok']
        assert app.call_guard.snapshot()['attempts']==before['attempts']
        assert len(calls)==8 and not app.state()['settled_reflection_failure_ids']
    finally:app.close()
