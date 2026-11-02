"""Public entry with actual clients and local synthetic HTTP; no live model."""
from copy import deepcopy
from pathlib import Path
import json,socket
import pytest
from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.max_release import tls_dispatch
from aria.engine.policy.qwen_max_structured import build_qwen_max_structured_route
from aria.engine.policy.memory_structured import build_memory_structured_route
from aria.release import create_max_release_clients
from tests.test_installable_max_release import make_clients as default_clients,metadata,prices,response,Response

V6='online_turn_decision_v6_semantic'

def deepseek_role_response(clients,wire):
    from tests import test_deepseek_offline_route as deepseek
    role='safety_review' if wire['max_tokens']==1024 else 'online'
    data=deepseek.response(getattr(clients,role))
    if role=='safety_review':
        from tests.test_safety_review_v2 import answer
        data['choices'][0]['message']['content']=json.dumps(answer(wire=wire))
    return data

def make_clients(version=V6):
    # These rejection fixtures exercise v6's response/segments equality contract.
    if version is None:return default_clients()
    models,online,offline=metadata()
    return create_max_release_clients(model_calls_enabled=True,
        api_key='synthetic-inert-key',base_url='https://openrouter.ai/api/v1',
        online_decision_version=version,
        online_structured_route=build_qwen_max_structured_route(models,online),
        memory_structured_route=build_memory_structured_route(models,offline))

@pytest.fixture(autouse=True)
def no_network_credentials(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('network or credential access forbidden')
    monkeypatch.setattr(socket,'create_connection',forbidden)
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr('aria.runtime.openrouter_api_key',forbidden)
    monkeypatch.setattr('aria.engine.policy.llm.openrouter_api_key',forbidden)
    monkeypatch.setattr(tls_dispatch,'open_observed',forbidden)

def setup(tmp_path,monkeypatch,failures=(1,2),*,unknown=False,empty=False,transform=None,version=V6):
    clients=make_clients(version);calls=[]
    def opened(req,**kw):
        from aria.engine.policy.max_release.request_requirements import validate_binding
        pending=app.call_guard.snapshot()['attempts'][-1]
        if pending['role']=='online_fast':
            validate_binding(pending,app.call_guard.rejected_requirements())
        assert pending['status']=='pending'
        calls.append(json.loads(req.data))
        reviewing=pending['role']=='safety_review'
        data=response(clients.safety_review if reviewing else clients.online)
        value=json.loads(data['choices'][0]['message']['content'])
        if reviewing:
            from tests.test_safety_review_v2 import answer
            value=answer()
        elif transform is not None:transform(value)
        if not reviewing and len(calls) in failures:
            if empty:
                value['response']=''
                for part in value['response_segments']:part['text']=''
            else:value['response']+='结构冲突。'
        data['choices'][0]['message']['content']=json.dumps(value,ensure_ascii=False)
        if unknown:data['usage'].pop('cost')
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        return Response(data,json.loads(req.data).get('stream',False))
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=8,
        model_stop_cost_usd='1.00',model_reservation_policy=prices())
    clock=[1000.0];app.call_guard._pacing_clock=lambda:clock[0]
    app.call_guard._pacing_sleep=lambda delta:clock.__setitem__(0,clock[0]+delta)
    state=app.start()['state']
    return app,calls,state['session_id']

@pytest.mark.parametrize('version,empty',[(V6,False),(V6,True),(None,True)],
    ids=['v6-conflict','v6-empty','default-v7-empty'])
def test_actual_public_entry_twice_failed_then_next_input(tmp_path,monkeypatch,version,empty):
    app,calls,sid=setup(tmp_path,monkeypatch,empty=empty,version=version)
    try:
        before=app.state()['profile']
        for turn in ('local-t1','local-t2'):
            out=app.submit('中性软件输入。',turn,sid)
            assert not out['ok'] and out['settled_validation_failure'] and not out['response_delivered']
            assert out['continue_next_input'] and not out['state']['requires_review']
            row=out['state']['turns'][-1]
            assert row['status']=='failed' and row['requires_review'] and row['response']==''
            assert row['source']=='model_failed' and turn in out['state']['settled_validation_failure_ids']
            assert app.state()['profile']==before
        old=deepcopy(app._run['turns'])
        duplicate=app.submit('中性软件输入。','local-t1',sid)
        assert not duplicate['ok'] and duplicate['duplicate'] and len(calls)==2
        third=app.submit('后续中性软件输入。','local-t3',sid)
        assert third['ok'] and len(calls)==(4 if version is None else 3) and app._run['turns'][:2]==old
        assert app.call_guard.snapshot()['known_cost_usd']==('0.0004' if version is None else '0.0003')
        pair=app._sessions[app._settled_owner][1]
        assert pair.turns[0].user==pair.turns[0].agent==''
        assert pair.working_memory['_sensitive_memory_excluded_turns']==[0,1]
    finally:app.close()

@pytest.mark.parametrize('legacy_response',[False,True],ids=['segments-only','legacy-conflict'])
def test_default_v7_delivers_segments_without_failed_disposition(tmp_path,monkeypatch,legacy_response):
    segments=[{'text':'我在听。\n'},{'text':'你可以接着说。'}]
    expected=''.join(part['text'] for part in segments)
    def transform(value):
        value.clear()
        value.update(response_segments=deepcopy(segments),safety_concern='none')
        if legacy_response:value['response']='与分段正文不同的旧字段。'
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(),transform=transform,version=None)
    try:
        out=app.submit('中性软件输入。','local-t1',sid)
        assert out['ok']
        assert app._run['turns'][-1]['response']==expected
        assert not app._settled_validation_directory().exists()
        assert not out['state']['requires_review']
        schema=calls[0]['response_format']['json_schema']['schema']
        assert schema['required']==['response_segments','safety_concern']
        assert 'response' not in schema['properties']
        duplicate=app.submit('中性软件输入。','local-t1',sid)
        assert duplicate['duplicate'] and len(calls)==2
        assert app.call_guard.snapshot()['known_cost_usd']=='0.0002'
    finally:app.close()

def test_public_metadata_contains_hashes_no_history_or_profile(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit('仅此输入不应进入处置元数据。','local-t1',sid)
        path,=app._settled_validation_directory().glob('*.json')
        raw=path.read_text(encoding='utf8');value=json.loads(raw)
        assert '仅此输入' not in raw and '中性软件正文' not in raw
        assert 'failed_turn' not in value and 'ledger_row' not in value and 'profile_before' not in value
        assert value['business_before_sha256']==value['business_after_sha256']
        assert value['original_status']=='failed' and value['replacement_response'] is None
    finally:app.close()

def test_unknown_fee_never_isolated_or_retried(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,unknown=True)
    try:
        out=app.submit('中性软件输入。','local-t1',sid)
        assert not out['ok'] and out['state']['requires_review']
        assert not app._settled_validation_directory().exists()
        assert app.call_guard.snapshot()['stopped'] and len(calls)==1
        again=app.submit('新的中性软件输入。','local-t2',sid)
        assert not again['ok'] and len(calls)==1
    finally:app.close()

def test_metadata_tamper_keeps_failed_notice_and_locks(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit('中性软件输入。','local-t1',sid)
        path,=app._settled_validation_directory().glob('*.json')
        value=json.loads(path.read_text());value['failed_turn_sha256']='0'*64
        path.write_text(json.dumps(value),encoding='utf8')
        state=app.state()
        assert state['requires_review'] and state['settled_validation_failure_ids']==[]
        assert state['turns'][0]['status']=='failed' and state['turns'][0]['response']==''
        assert not app.submit('另一中性输入。','local-t2',sid)['ok'] and len(calls)==1
    finally:app.close()

def test_forget_clears_new_metadata_and_keeps_allowance(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit('中性软件输入。','local-t1',sid)
        prior=app.call_guard.snapshot()
        app.forget_run()
        assert list(app._settled_validation_directory().glob('*.json'))==[]
        state=app.state()
        assert not state['requires_review'] and state['turns']==[]
        assert app.call_guard.snapshot()['calls_used']==prior['calls_used']==1
        assert app.call_guard.snapshot()['known_cost_usd']==prior['known_cost_usd']=='0.0001'
    finally:app.close()

def test_ui_retains_failure_rendering_but_unlocks_only_verified_failed_ids():
    raw=Path('aria/assets/voice/minimal.html').read_text(encoding='utf8')
    assert 'settled_validation_failure_ids' in raw and "row?.status==='failed'" in raw
    assert "row?.source==='model_failed'" in raw and 'row?.requires_review && !isolated' in raw
    assert '本轮失败，没有可交付回复。' in raw

def test_installable_new_modules_have_no_private_task_dependencies():
    paths=[Path('aria/engine/policy/max_release')/name for name in
        ('validation_support.py','validation_contract.py','validation_guard.py')]
    paths.append(Path('aria/voice/settled_validation.py'))
    raw='\n'.join(p.read_text(encoding='utf8') for p in paths)
    for forbidden in ('sys.path.insert','importlib.util.spec_from_file_location','load_local_env('):
        assert forbidden not in raw

def test_restart_validates_failed_disposition_and_keeps_charge(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    app.submit('中性软件输入。','local-t1',sid)
    original=deepcopy(app._run['turns'][0]);app.close()
    app2=MinimalWebApp(tmp_path,clients=make_clients(),model_max_requests=8,
        model_stop_cost_usd='1.00',model_reservation_policy=prices())
    try:
        state=app2.state()
        assert not state['requires_review'] and state['settled_validation_failure_ids']==['local-t1']
        assert state['turns']==[original] and app2.call_guard.snapshot()['calls_used']==1
        duplicate=app2.submit('中性软件输入。','local-t1',sid)
        assert not duplicate['ok'] and len(calls)==1
    finally:app2.close()

def test_metadata_storage_failure_hard_stops_without_delivery(tmp_path,monkeypatch):
    import aria.voice.settled_validation as module
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    def fail(*a,**kw):raise OSError('explicit local software failure')
    monkeypatch.setattr(module,'write_once',fail)
    try:
        with pytest.raises(OSError):app.submit('中性软件输入。','local-t1',sid)
        state=app.state()
        assert state['requires_review'] and state['turns'][0]['response']==''
        assert state['turns'][0]['status']=='failed' and app.call_guard.snapshot()['stopped']
        assert len(calls)==1 and app.call_guard.snapshot()['known_cost_usd']=='0.0001'
    finally:app.close()

def test_delivered_invitation_consumed_without_confirm_or_reissue(tmp_path,monkeypatch):
    from tests.test_stated_preferences import _profile
    from tests.test_preference_dialogue_readiness import control
    holder={}
    def transform(value):
        value['preference_control']=control(holder['persona'],holder['ctx'],text='中性软件确认。')
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,),transform=transform)
    try:
        engine,ctx=app._sessions[app._settled_owner]
        persona=engine.persona
        persona.stated_preferences=deepcopy(_profile(confirmed=False).stated_preferences)
        app.store.save(persona)
        holder.update(persona=persona,ctx=ctx)
        invitation=app.preference_invitation(persona.stated_preferences[0].id,sid)
        assert invitation['ok']
        value=invitation['preference_invitation']
        ack=app.preference_invitation_ack(value['id'],value['question_sha256'],sid)
        assert ack['ok'] and app.state()['preference_invitation']['status']=='delivered'
        before=[p.model_dump(mode='json') for p in persona.stated_preferences]
        out=app.submit('中性软件确认。','local-t1',sid)
        assert out['settled_validation_failure'] and not out['ok']
        assert app.state()['preference_invitation'] is None
        assert [p.model_dump(mode='json') for p in app.store.load(app._settled_owner).stated_preferences]==before
        assert all(p['status']=='proposed' and p['confirmation'] is None for p in before)
        duplicate=app.submit('中性软件确认。','local-t1',sid)
        assert not duplicate['ok'] and len(calls)==1 and app.state()['preference_invitation'] is None
    finally:app.close()

@pytest.mark.parametrize('field',['require_preference_control','require_task_projection','missing'])
def test_disposition_cannot_rewrite_actual_projection_requirements(tmp_path,monkeypatch,field):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        assert app.submit('中性软件输入。','local-t1',sid)['continue_next_input']
        path,=app._settled_validation_directory().glob('*.json')
        value=json.loads(path.read_text(encoding='utf8'))
        if field=='missing':value.pop('requirements')
        else:value['requirements'][field]=not value['requirements'][field]
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8')
        state=app.state()
        assert state['requires_review'] and state['settled_validation_failure_ids']==[]
        assert state['turns'][0]['status']=='failed' and state['turns'][0]['response']==''
        assert not app.submit('后续中性软件输入。','local-t2',sid)['ok'] and len(calls)==1
    finally:app.close()

def test_actual_requirements_binding_write_failure_sends_no_post(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    def fail(*a,**kw):raise OSError('software_actual_binding_write_failure')
    monkeypatch.setattr(app.call_guard,'_persist_requirements_binding',fail)
    try:
        result=app.submit('中性软件输入。','local-t1',sid)
        assert not result['ok'] and len(calls)==0 and app.call_guard.snapshot()['stopped']
        assert app.state()['requires_review']
    finally:app.close()


def deepseek_setup(tmp_path,monkeypatch,*,fault=None,finish='length',content=None,guard=None):
    from tests import test_deepseek_offline_route as deepseek
    clients=deepseek.clients();calls=[]
    def opened(req,**kw):
        calls.append(json.loads(req.data))
        reviewing=calls[-1]['max_tokens']==1024
        data=deepseek.response(clients.safety_review if reviewing else clients.online)
        if reviewing:
            from tests.test_safety_review_v2 import answer
            data['choices'][0]['message']['content']=json.dumps(answer(wire=calls[-1]))
        if len(calls)==1:
            data['choices'][0]['finish_reason']=finish
            data['choices'][0]['message']['content']=(
                '{"response_segments":[{"text":"未闭合软件正文。"}]'+' '*23305
                if content is None else content)
            data['usage'].update(completion_tokens=4096,total_tokens=4116)
            if fault is not None:fault(data)
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        return Response(data,json.loads(req.data).get('stream',False))
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    options=({'model_call_guard':guard} if guard is not None else {
        'model_max_requests':8,'model_stop_cost_usd':'1.00','model_reservation_policy':deepseek.policy()})
    app=MinimalWebApp(tmp_path,clients=clients,**options)
    clock=[1000.0];app.call_guard._pacing_clock=lambda:clock[0]
    app.call_guard._pacing_sleep=lambda delta:clock.__setitem__(0,clock[0]+delta)
    return app,calls,app.start()['state']['session_id']


@pytest.mark.parametrize('finish,content,physical_status',[
    ('length',None,'failed'),
    ('length','{"response_segments":[{"text":"完整但被截断标记的正文。"}],"safety_concern":"none"}','failed'),
    ('stop','   ','failed'),
    ('stop','{"response_segments":[','received'),
])
def test_deepseek_bad_first_body_is_failed_once_then_next_input(
        tmp_path,monkeypatch,finish,content,physical_status):
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch,finish=finish,content=content)
    try:
        profile=app.state()['profile']
        first=app.submit('中性软件输入。','deepseek-t1',sid)
        assert not first['ok'] and first['settled_validation_failure']
        assert first['continue_next_input'] and not first['model_response_delivered']
        assert app.state()['profile']==profile and not app.state()['requires_review']
        row=app.call_guard.snapshot()['attempts'][0]
        assert row['status']==physical_status and row['usage']['cost']==.00001
        assert row['online_contract_isolation_outcome']['model_reply_delivered'] is False
        original=deepcopy(app._run['turns'][0])
        assert original['status']=='failed' and original['response']==''
        duplicate=app.submit('中性软件输入。','deepseek-t1',sid)
        assert duplicate['duplicate'] and not duplicate['ok'] and len(calls)==1
        following=app.submit('下一条中性软件输入。','deepseek-t2',sid)
        assert following['ok'] and len(calls)==3
        assert app._run['turns'][0]==original
        state=app.call_guard.snapshot()
        assert state['known_cost_usd']=='0.00003' and state['reserved_cost_usd']=='0'
        assert not state['stopped']
    finally:app.close()


@pytest.mark.parametrize('fault',[
    lambda data:data['usage'].pop('cost'),
    lambda data:data.update(provider='Fireworks'),
    lambda data:data.update(model='different/model'),
    lambda data:data['usage'].update(total_tokens=4117),
    lambda data:data['usage'].update(completion_tokens=4097,total_tokens=4117),
    lambda data:data['usage'].update(cost=999),
    lambda data:data['choices'][0]['message'].update(refusal='refused'),
    lambda data:data['choices'][0]['message'].update(tool_calls=[{'type':'function','function':{'name':'x','arguments':'{}'}}]),
],ids=['unknown-cost','provider','model','total','output-bound','cost-bound','refusal','tool'])
def test_deepseek_truncation_does_not_isolate_unverified_receipt(tmp_path,monkeypatch,fault):
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch,fault=fault)
    try:
        first=app.submit('中性软件输入。','deepseek-t1',sid)
        assert not first['ok'] and app.call_guard.snapshot()['stopped']
        assert not app._settled_validation_directory().exists()
        assert not app.submit('下一条中性软件输入。','deepseek-t2',sid)['ok']
        assert len(calls)==1
    finally:app.close()


def test_deepseek_failed_receipt_reopens_without_replay_or_releasing_fee(tmp_path,monkeypatch):
    from tests import test_deepseek_offline_route as deepseek
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch)
    assert app.submit('中性软件输入。','deepseek-t1',sid)['settled_validation_failure']
    row=deepcopy(app.call_guard.snapshot()['attempts'][0]);app.close()
    reopened=MinimalWebApp(tmp_path,clients=deepseek.clients(),model_max_requests=8,
        model_stop_cost_usd='1.00',model_reservation_policy=deepseek.policy())
    try:
        assert not reopened.state()['requires_review']
        assert not reopened.call_guard.snapshot()['stopped']
        assert reopened.call_guard.snapshot()['attempts']==[row]
        assert reopened.call_guard.snapshot()['known_cost_usd']=='0.00001'
        assert reopened.submit('中性软件输入。','deepseek-t1',sid)['duplicate']
        assert len(calls)==1
    finally:reopened.close()


def test_fresh_profile_shared_guard_retains_closed_app_failure_evidence(tmp_path,monkeypatch):
    from tests import test_deepseek_offline_route as deepseek
    guard=deepseek.guard(tmp_path)
    app,calls,sid=deepseek_setup(tmp_path/'case1',monkeypatch,guard=guard)
    assert app.submit('第一条中性软件输入。','case1-turn',sid)['settled_validation_failure']
    failed=deepcopy(guard.snapshot()['attempts'][0]);app.close()
    second=MinimalWebApp(tmp_path/'case2',clients=deepseek.clients(),model_call_guard=guard)
    try:
        sid2=second.start()['state']['session_id']
        assert second.submit('第二条中性软件输入。','case2-turn',sid2)['ok']
        assert len(calls)==3 and guard.snapshot()['attempts'][0]==failed
        assert not guard.snapshot()['stopped'] and guard.snapshot()['known_cost_usd']=='0.00003'
    finally:second.close();guard.close()


def test_deepseek_settlement_write_failure_keeps_failure_and_stops(tmp_path,monkeypatch):
    import aria.voice.settled_validation as module
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise OSError('local durable disposition unavailable')
    monkeypatch.setattr(module,'write_once',fail)
    try:
        with pytest.raises(OSError):app.submit('中性软件输入。','deepseek-t1',sid)
        state=app.call_guard.snapshot()
        assert state['stopped'] and state['known_cost_usd']=='0.00001'
        assert state['attempts'][0]['status']=='failed'
        assert app.state()['requires_review'] and app._run['turns'][0]['response']==''
        assert len(calls)==1
    finally:app.close()


def test_deepseek_forget_retains_charge_without_authorizing_missing_evidence(tmp_path,monkeypatch):
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch)
    try:
        assert app.submit('中性软件输入。','deepseek-t1',sid)['settled_validation_failure']
        app.forget_run()
        assert app._run['turns']==[] and not list(app._settled_validation_directory().glob('*.json'))
        ledger=app.call_guard.snapshot()
        assert ledger['calls_used']==1 and ledger['known_cost_usd']=='0.00001'
        assert ledger['attempts'][0]['status']=='failed' and ledger['stopped']
        assert len(calls)==1
    finally:app.close()


def test_deepseek_receipt_shape_tamper_cannot_unlock(tmp_path,monkeypatch):
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch)
    try:
        assert app.submit('中性软件输入。','deepseek-t1',sid)['settled_validation_failure']
        path,=app._settled_validation_directory().glob('*.json')
        value=json.loads(path.read_text());value['ledger_row_sha256']='0'*64
        path.write_text(json.dumps(value),encoding='utf8')
        assert app.call_guard.snapshot()['stopped']
        assert not app.submit('另一条中性软件输入。','deepseek-t2',sid)['ok']
        assert len(calls)==1
    finally:
        # Restore the saved evidence only to allow fixture teardown; no call is retried.
        value['ledger_row_sha256']=app.call_guard._reflection_ledger_row(value['request_id'])[
            'online_contract_isolation_outcome']['original_receipt_sha256']
        path.write_text(json.dumps(value),encoding='utf8');app.close()


def test_mixed_confirm_failure_and_online_truncation_keep_both_failed(tmp_path,monkeypatch):
    from tests import test_confirm_judge_format_failure as confirm
    from tests import test_deepseek_offline_route as deepseek
    app,guard,sid,ctx,wires,delivered=confirm.setup(tmp_path,monkeypatch,outcome='truncated')
    try:
        assert app.submit('中性软件输入。','confirm-t1',sid)['ok']
        confirm_row=deepcopy(guard.snapshot()['attempts'][1])
        later=[]
        def opened(req,**kw):
            later.append(json.loads(req.data));data=deepseek_role_response(app.clients,later[-1])
            if len(later)==1:
                data['choices'][0].update(finish_reason='length',message={'content':'{"response_segments":['})
            kw['trace'].preparation_recovery.bind_request_headers(req)
            kw['trace'].preparation_recovery.before_preparation()
            return Response(data,json.loads(req.data).get('stream',False))
        monkeypatch.setattr(tls_dispatch,'open_observed',opened)
        assert app.submit('另一条中性软件输入。','mixed-t2',sid)['settled_validation_failure']
        assert app.submit('后续中性软件输入。','mixed-t3',sid)['ok']
        ledger=guard.snapshot()
        assert [r['status'] for r in ledger['attempts']]==['received','failed','received','failed','received','received']
        assert ledger['attempts'][1]==confirm_row and ledger['known_cost_usd']=='0.00006'
        assert not ledger['stopped'] and len(wires)==len(later)==3
    finally:app.close();guard.close()


def test_deepseek_truncation_keeps_protection_prelude_and_discards_partial_body(tmp_path,monkeypatch):
    app,calls,sid=deepseek_setup(tmp_path,monkeypatch)
    try:
        preludes=[]
        out=app.submit('我想自杀。','protected-t1',sid,
            on_safety_prelude=lambda value:preludes.append(value) or True)
        assert out['settled_validation_failure'] and out['protection_prelude_delivered']
        assert out['model_response_delivered'] is False and len(preludes)==1
        failed=deepcopy(app._run['turns'][0])
        assert failed['response']=='' and failed['premodel_task']['status']=='response_only_failed'
        assert app.submit('下一条中性软件输入。','protected-t2',sid)['ok']
        assert len(calls)==3 and app._run['turns'][0]==failed
    finally:app.close()


def test_mixed_reflection_failure_and_online_truncation_keep_both_failed(tmp_path,monkeypatch):
    from tests import test_installable_settled_reflection as reflection
    from tests import test_deepseek_offline_route as deepseek
    for name,value in [('make_clients',deepseek.clients),('prices',deepseek.policy),('response',deepseek.response)]:
        monkeypatch.setattr(reflection,name,value)
    app,calls=reflection.setup(tmp_path,monkeypatch)
    try:
        assert not reflection.one_session(app,1)['ok']
        prior=deepcopy(app.call_guard.snapshot()['attempts']);later=[]
        sid=app.start()['state']['session_id']
        def opened(req,**kw):
            later.append(json.loads(req.data));data=deepseek_role_response(app.clients,later[-1])
            if len(later)==1:
                data['choices'][0].update(finish_reason='length',message={'content':'{"response_segments":['})
            kw['trace'].preparation_recovery.bind_request_headers(req)
            kw['trace'].preparation_recovery.before_preparation()
            return Response(data,json.loads(req.data).get('stream',False))
        monkeypatch.setattr(tls_dispatch,'open_observed',opened)
        assert app.submit('另一条中性软件输入。','mixed-t1',sid)['settled_validation_failure']
        assert app.submit('后续中性软件输入。','mixed-t2',sid)['ok']
        ledger=app.call_guard.snapshot()
        assert ledger['attempts'][:len(prior)]==prior
        assert [r['role'] for r in ledger['attempts'] if r['status']=='failed']==['reflection','online_fast']
        assert not ledger['stopped'] and len(calls)==8 and len(later)==3
    finally:app.close()


@pytest.mark.parametrize('unknown',[False,True],ids=['settled-truncation','unknown-fee-hard-stop'])
@pytest.mark.parametrize('failure_role',['online','safety_review'])
def test_original_safety_driver_closes_each_profile_and_uses_same_guard(tmp_path,monkeypatch,unknown,failure_role):
    from scripts import run_story as driver
    from scripts.run_safety_set import safety_story
    from tests import test_deepseek_offline_route as deepseek
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL
    models,endpoint=deepseek.metadata()
    source=tmp_path/'models.json';source.write_bytes(models)
    route=tmp_path/'route.json';route.write_bytes(endpoint)
    price=tmp_path/'policy.json';price.write_text(json.dumps(deepseek.policy()),encoding='utf8')
    inputs=tmp_path/'software-inputs.json'
    inputs.write_text(json.dumps({'cases':[{'regression_id':f'software-{index}',
        'execution_input':{'messages':[{'role':'user','content':'中性软件输入。'}]}}
        for index in range(3)]}),encoding='utf8')
    out=tmp_path/'run'
    args=driver.parser().parse_args(['--story',str(inputs),'--out',str(out),
        '--enable-model-calls','--model-max-requests','10','--model-stop-cost-usd','1',
        '--model-reservation-policy',str(price),'--online-models-metadata',str(source),
        '--online-endpoints-metadata',str(route),'--offline-endpoints-metadata',str(route),
        '--release-profile','release_deepseek_v1','--offline-model',DEEPSEEK_MODEL,
        '--offline-provider','together'])
    monkeypatch.setattr('aria.runtime.openrouter_api_key',lambda:'inert-software-key')
    factory,close=driver.configured_factory(args);apps=[];calls=[]
    def tracked_factory(path):
        if apps:assert apps[-1]._closing
        app=factory(path)
        if apps:assert app.call_guard is apps[0].call_guard
        else:
            clock=[1000.0];app.call_guard._pacing_clock=lambda:clock[0]
            app.call_guard._pacing_sleep=lambda delta:clock.__setitem__(0,clock[0]+delta)
        apps.append(app);return app
    def opened(req,**kw):
        wire=json.loads(req.data);calls.append(wire)
        role='safety_review' if wire['max_tokens']==1024 else 'online'
        data=deepseek_role_response(apps[-1].clients,wire)
        if len(calls)==1:
            if failure_role=='safety_review':
                data['choices'][0]['message']['content']=json.dumps({
                    'response_segments':[{'text':'未经复核不能交付的软件候选句。'}],'safety_concern':'fall',
                    'safety_evidence':{'quote':'中性软件输入。','subject':'owner','time':'current','urgency':'immediate'}})
        if (len(calls)==1 and failure_role=='online'
                or len(calls)==2 and failure_role=='safety_review' and role=='safety_review'):
            data['choices'][0].update(finish_reason='length',message={'content':'{"response_segments":['})
            if unknown:data['usage'].pop('cost')
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        return Response(data,json.loads(req.data).get('stream',False))
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    try:
        summary=driver.run_story(safety_story(inputs),1,out,app_factory=tracked_factory)
        rows=[json.loads(line) for path in sorted(out.glob('*/*/turns.jsonl'))
            for line in path.read_text(encoding='utf8').splitlines()]
        turns=[row for row in rows if row['turn_id']]
        if failure_role=='online':
            assert turns[0]['failed'] and turns[0]['delivered_response']==''
            assert summary['failed_turns']==1
        else:assert '未经复核不能交付的软件候选句。' not in turns[0]['delivered_response']
        assert all(app._closing for app in apps)
        ledger=apps[-1].call_guard.snapshot()
        failed_index=int(failure_role=='safety_review')
        assert ledger['attempts'][failed_index]['status']=='failed'
        if unknown:
            assert len(calls)==1+failed_index and summary['turns']==1 and summary['unexecuted_turns']==2
            assert ledger['unknown_cost_requests']==1 and ledger['stopped'] and summary['stop_reason']
        else:
            assert len(calls)==5+failed_index and summary['turns']==len(apps)==3 and summary['unexecuted_turns']==0
            assert all(row['ok'] for row in turns[1:]) and summary['stop_reason'] is None
            assert ledger['known_cost_usd']==('0.00006' if failed_index else '0.00005') and not ledger['stopped']
            assert len({row['operation_id'] for row in ledger['attempts']})==3
    finally:close()


def safety_review_setup(tmp_path,monkeypatch,*,mode='length',fault=None,response_fault=None):
    from tests import test_deepseek_offline_route as deepseek
    clients=deepseek.clients();calls=[]
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=10,
        model_stop_cost_usd='1',model_reservation_policy=deepseek.policy())
    clock=[1000.0];app.call_guard._pacing_clock=lambda:clock[0]
    app.call_guard._pacing_sleep=lambda delta:clock.__setitem__(0,clock[0]+delta)
    text='虚构的本轮软件输入。';candidate='这条未经复核的候选软件正文不应交付。'
    def opened(req,**kw):
        wire=json.loads(req.data);role='safety_review' if wire['max_tokens']==1024 else 'online'
        calls.append(role);data=deepseek_role_response(clients,wire)
        if len(calls)==1:
            data['choices'][0]['message']['content']=json.dumps({
                'response_segments':[{'text':candidate}], 'safety_concern':'fall',
                'safety_evidence':{'quote':text,'subject':'owner','time':'current','urgency':'immediate'}})
        if role=='safety_review' and len(calls)==2:
            choice=data['choices'][0]
            if mode=='length':choice.update(finish_reason='length',message={'content':'{"risk_assessment":'})
            elif mode=='empty':choice['message']['content']=' '
            elif mode=='missing_risk':choice['message']['content']='{"keep_each_span":[true]}'
            elif mode=='sentence_bounds':
                choice['message']['content']='{"risk_assessment":[],"keep_each_span":[true,false]}'
            else:choice['message']['content']='{}'
            if fault is not None:fault(data)
        kw['trace'].preparation_recovery.bind_request_headers(req)
        kw['trace'].preparation_recovery.before_preparation()
        result=Response(data,json.loads(req.data).get('stream',False))
        if role=='safety_review' and len(calls)==2 and response_fault is not None:response_fault(result)
        return result
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    return app,calls,app.start()['state']['session_id'],text,candidate


@pytest.mark.parametrize('mode',['length','schema','empty','sentence_bounds','missing_risk'])
def test_safety_review_receipt_and_content_failures_are_independent(tmp_path,monkeypatch,mode):
    app,calls,sid,text,candidate=safety_review_setup(tmp_path,monkeypatch,mode=mode)
    try:
        first=app.submit(text,'review-failure-t1',sid)
        assert calls==['online','safety_review']
        delivered=app._run['turns'][0]['response']
        assert (candidate in delivered) is (mode=='missing_risk')
        audit=app._sessions[app._settled_owner][1].working_memory['safety_review']
        if mode=='missing_risk':
            from aria.engine.risk.safety_delivery import render_unverified_safety_v2
            assert audit['status']=='reviewed' and audit['model_body_delivered'] is True
            assert audit['risk_assessment']['concern']=='uncertain'
            assert 'risk_assessment' in audit['dropped_fields']
            assert render_unverified_safety_v2()[0] in delivered
        else:
            assert audit['status']=='review_failed' and audit['model_body_delivered'] is False
        assert audit['semantic_verified'] is False and audit['grants_permission'] is False
        ledger=app.call_guard.snapshot();row=deepcopy(ledger['attempts'][1])
        if mode in {'length','empty'}:
            assert row['status']=='failed' and row['safety_review_disposition']=='withheld_known_cost_format_failure_v1'
        else:
            assert row['status']=='received' and 'safety_review_disposition' not in row
            assert 'validation_error' not in row
        expected_raw={'length':'{"risk_assessment":','empty':' ','schema':'{}',
            'sentence_bounds':'{"risk_assessment":[],"keep_each_span":[true,false]}',
            'missing_risk':'{"keep_each_span":[true]}'}[mode]
        public=json.loads(Path(row['public_first_output']['path']).read_text(encoding='utf-8'))
        assert public['public_first_output'][0]['content']==expected_raw
        # The original risky-input exclusion is independent of either result.
        ctx=app._sessions[app._settled_owner][1]
        assert 0 in ctx.working_memory['_sensitive_memory_excluded_turns']
        assert ledger['known_cost_usd']=='0.00002' and not ledger['stopped']
        duplicate=app.submit(text,'review-failure-t1',sid)
        assert duplicate['duplicate'] and len(calls)==2
        assert app.submit('下一条中性软件输入。','review-failure-t2',sid)['ok']
        assert calls==['online','safety_review','online','safety_review']
        assert app.call_guard.snapshot()['attempts'][1]==row
    finally:app.close()


@pytest.mark.parametrize('fault',[
    lambda data:data['usage'].pop('cost'),
    lambda data:data.update(provider='Fireworks'),
    lambda data:data.update(model='other/model'),
    lambda data:data['usage'].update(total_tokens=999),
    lambda data:data['usage'].update(cost=999),
    lambda data:data['usage'].update(completion_tokens=1025,total_tokens=1045),
    lambda data:data['choices'][0]['message'].update(refusal='refusal'),
],ids=['unknown','provider','model','totals','cost-bound','token-bound','refusal'])
def test_safety_review_format_never_releases_unverified_or_wrong_channel_failure(tmp_path,monkeypatch,fault):
    app,calls,sid,text,candidate=safety_review_setup(tmp_path,monkeypatch,fault=fault)
    try:
        app.submit(text,'unsafe-review-t1',sid)
        assert candidate not in app._run['turns'][0]['response']
        state=app.call_guard.snapshot()
        assert state['stopped'] and state['attempts'][1]['status']=='failed'
        assert 'safety_review_disposition' not in state['attempts'][1]
        assert not app.submit('下一条中性软件输入。','unsafe-review-t2',sid)['ok']
        assert calls==['online','safety_review']
    finally:app.close()


@pytest.mark.parametrize('fault',['http403','incomplete','capture_storage','cancelled'])
def test_safety_review_transport_storage_or_stop_still_stops(tmp_path,monkeypatch,fault):
    import io
    def response_fault(response):
        if fault=='http403':response.status=403
        elif fault=='incomplete':response.body=io.BytesIO(response.body.getvalue()[:-10])
        elif fault=='cancelled':app.call_guard._stop_requested.set()
    app,calls,sid,text,candidate=safety_review_setup(tmp_path,monkeypatch,response_fault=response_fault)
    if fault=='capture_storage':
        original=app.call_guard.content_erasure.write
        def write(path,value,directory):
            if value.get('role')=='safety_review':raise OSError('local receipt capture unavailable')
            return original(path,value,directory)
        monkeypatch.setattr(app.call_guard.content_erasure,'write',write)
    try:
        app.submit(text,'review-hard-t1',sid)
        assert candidate not in app._run['turns'][0]['response']
        state=app.call_guard.snapshot()
        assert state['stopped']
        assert 'safety_review_disposition' not in state['attempts'][1]
        assert not app.submit('下一条中性软件输入。','review-hard-t2',sid)['ok']
        assert calls==['online','safety_review']
    finally:app.close()


@pytest.mark.parametrize('route',['qwen','fireworks','together'])
def test_review_receipt_does_not_claim_content_validation(tmp_path,monkeypatch,route):
    from tests import test_deepseek_offline_route as deepseek
    from tests import test_deepseek_release_guard as prior
    from tests import test_installable_max_release as old
    from aria.engine.risk.safety_review_v2 import safety_review_messages,parse_review
    if route=='qwen':
        client=old.make_clients().safety_review;guard=old.guard(tmp_path);data=old.response(client)
    else:
        client=prior.clients(provider=route).safety_review;guard=prior.guard(tmp_path,provider=route)
        data=prior.response(client)
    data['choices'][0]['message']['content']='{"conflicts":false,"delete_sentence_ids":[]}'
    seen=deepseek.dispatch(monkeypatch,data)
    original=json.loads(old.messages(client)[1]['content'])
    messages=safety_review_messages(original['utterance'],'软件候选。',original['safety_evidence'],
        None,final_protection='条件保护。')
    try:
        guard.attach(client)
        with guard.scope('turn','version-check','software-session'):
            raw=client.complete(messages)
        state=guard.snapshot();row=state['attempts'][0]
        assert len(seen)==1
        assert raw==data['choices'][0]['message']['content']
        assert row['status']=='received' and not state['stopped']
        assert 'safety_review_disposition' not in row
        reviewed=parse_review(raw,1,allow_partial=True)
        assert reviewed['risk_assessment']['concern']=='uncertain'
        assert reviewed['conflicts'] is False and reviewed['delete_sentence_ids']==[]
        assert 'risk_assessment' in reviewed['dropped_fields']
        if route=='together':
            assert row['safety_review_receipt_contract']['version']=='safety_review_v2'
        else:
            assert 'safety_review_receipt_contract' not in row
    finally:guard.close()


def test_mixed_safety_review_and_online_truncation_failures_preserve_both(tmp_path,monkeypatch):
    from tests import test_deepseek_offline_route as deepseek
    app,calls,sid,text,candidate=safety_review_setup(tmp_path,monkeypatch)
    try:
        app.submit(text,'review-mixed-t1',sid)
        prior=deepcopy(app.call_guard.snapshot()['attempts']);later=[]
        def opened(req,**kw):
            later.append(json.loads(req.data));data=deepseek_role_response(app.clients,later[-1])
            if len(later)==1:data['choices'][0].update(finish_reason='length',message={'content':'{"response_segments":['})
            kw['trace'].preparation_recovery.bind_request_headers(req)
            kw['trace'].preparation_recovery.before_preparation()
            return Response(data,json.loads(req.data).get('stream',False))
        monkeypatch.setattr(tls_dispatch,'open_observed',opened)
        assert app.submit('另一条中性软件输入。','review-mixed-t2',sid)['settled_validation_failure']
        assert app.submit('后续中性软件输入。','review-mixed-t3',sid)['ok']
        state=app.call_guard.snapshot()
        assert state['attempts'][:2]==prior and not state['stopped']
        assert [row['role'] for row in state['attempts'] if row['status']=='failed']==['safety_review','online_fast']
        assert state['known_cost_usd']=='0.00005' and len(calls)==2 and len(later)==3
    finally:app.close()


@pytest.mark.parametrize('fault',[
    'total_missing','total_string','total_bool','total_negative','total_mismatch',
    'hash_both_missing','hash_both_null','hash_both_empty','hash_both_short','hash_both_nonhex',
])
def test_settled_safety_review_requires_complete_usage_and_actual_request_digest(tmp_path,monkeypatch,fault):
    app,calls,sid,text,candidate=safety_review_setup(tmp_path,monkeypatch)
    try:
        app.submit(text,'review-metadata-t1',sid)
        original=deepcopy(app.call_guard.snapshot()['attempts'][1]);row=deepcopy(original)
        eligible=app.call_guard._settled_safety_review_format_failure
        assert eligible(original)
        if fault=='total_missing':row['usage'].pop('total_tokens')
        elif fault=='total_string':row['usage']['total_tokens']=str(row['usage']['total_tokens'])
        elif fault=='total_bool':row['usage']['total_tokens']=True
        elif fault=='total_negative':row['usage']['total_tokens']=-1
        elif fault=='total_mismatch':row['usage']['total_tokens']+=1
        elif fault=='hash_both_missing':
            row.pop('request_sha256');row['safety_review_receipt_contract'].pop('request_sha256')
        else:
            value={'hash_both_null':None,'hash_both_empty':'','hash_both_short':'a'*63,
                'hash_both_nonhex':'z'*64}[fault]
            row['request_sha256']=row['safety_review_receipt_contract']['request_sha256']=value
        assert not eligible(row)
        assert app.call_guard.snapshot()['attempts'][1]==original and len(calls)==2
    finally:app.close()
