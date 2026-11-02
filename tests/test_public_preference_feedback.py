"""Actual App/client/guard and neutral preset actions; not semantic accuracy."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import hashlib,io,json,ssl
from pathlib import Path
import pytest

from aria.voice.minimal import MinimalWebApp
from aria.release import create_max_release_clients
from aria.engine.policy.max_release import tls_dispatch,tls_transport
from aria.engine.memory import preference_dialogue,preference_recovery
from tests.test_installable_max_release import make_clients,prices,response,no_network_credentials
from tests.test_public_partial_input import Socket,TLS
from tests.test_stated_preferences import _profile

BODY='中性软件正文。'
INPUT='中性软件输入。'
V6='online_turn_decision_v6_semantic'
V7='online_turn_decision_v7'

def setup(tmp_path,monkeypatch,*,advance=0,action='unclear',mutation=None,phase=None,false_claim=False,version=V7):
    clock={'utc':datetime.now(timezone.utc),'mono':1000.}
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return clock['utc'] if tz else clock['utc'].replace(tzinfo=None)
    monkeypatch.setattr(preference_dialogue,'datetime',Clock)
    monkeypatch.setattr(preference_recovery,'datetime',Clock)
    clients=make_clients();wires=[];sockets=[];holder={}
    if version == V6:
        clients=create_max_release_clients(model_calls_enabled=True,
            api_key=clients.online.api_key,base_url=clients.online.base_url,
            online_structured_route=clients.online.online_structured_route,
            memory_structured_route=clients.memory.memory_structured_route,
            online_decision_version=V6)
    def change():
        if mutation:mutation(holder,clock)
    def opened(request,**kwargs):
        wires.append(bytes(request.data))
        if phase=='tls':change()
        trace=kwargs['trace'];trace.preparation_recovery.bind_request_headers(request)
        wire=json.loads(request.data)
        schema_name=wire['response_format']['json_schema']['name']
        is_review=schema_name=='aria_safety_review_v2'
        value=response(clients.safety_review if is_review else clients.online)
        raw=json.loads(value['choices'][0]['message']['content'])
        if is_review and false_claim:
            from tests.test_safety_review_v2 import answer
            raw=answer(delete=[1])  # Preset label for this fixture's appended claim.
        if schema_name=='aria_confirm_judge':
            raw={'action':action}
            value['usage'].update(completion_tokens=4106,
                completion_tokens_details={'reasoning_tokens':4096})
        elif not is_review:
            if 'preference_control' in wire['response_format']['json_schema']['schema']['required']:
                raw['preference_control']={'action':action}
            else:raw.pop('preference_control',None)
            if false_claim:
                raw['response']=BODY+'合成状态句。'
                raw['response_segments'].append({'text':'合成状态句。','kind':'preference_state',
                    'source_quote':'','action':'none','target':'','state_ref':''})
        value['choices'][0]['message']['content']=json.dumps(raw,ensure_ascii=False)
        data=json.dumps(value,ensure_ascii=False).encode()
        packet=b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data
        conn=tls_transport._HTTPSConnection('localhost',trace=trace,context=TLS())
        class ObservedSocket(Socket):
            def sendall(self,data,*args,**kw):
                super().sendall(data,*args,**kw)
                if phase=='after_headers' and len(self.writes)==1:change()
        def create(*a,**k):
            sock=ObservedSocket(packet);sockets.append(sock);return sock
        conn._create_connection=create
        conn.request('POST','/chat/completions',body=request.data,headers=dict(request.header_items()))
        return conn.getresponse()
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=10,
        model_stop_cost_usd='2',model_reservation_policy=prices())
    app.call_guard._pacing_clock=lambda:clock['mono']
    def sleep(delta):
        clock['mono']+=delta;clock['utc']+=timedelta(seconds=delta)
        if phase=='waiting':change()
    app.call_guard._pacing_sleep=sleep
    sid=app.start()['state']['session_id'];engine,ctx=app._sessions[app._settled_owner]
    persona=engine.persona;persona.stated_preferences=deepcopy(_profile(confirmed=False).stated_preferences)
    app.store.save(persona)
    invitation=app.preference_invitation(persona.stated_preferences[0].id,sid)['preference_invitation']
    assert app.preference_invitation_ack(invitation['id'],invitation['question_sha256'],sid)['ok']
    pending=deepcopy(ctx.working_memory['_preference_dialogue_invitation'])
    holder.update(app=app,ctx=ctx,persona=persona,pending=pending)
    clock['utc']+=timedelta(seconds=advance)
    return app,wires,sockets,sid,holder,clock

@pytest.mark.parametrize('advance',[0,179,180,181,301])
def test_actual_queued_expiry_completes_same_input_once_with_final_budget(tmp_path,monkeypatch,advance):
    # Retain the exact v6 single-call/body contract as explicit compatibility.
    app,wires,sockets,sid,h,clock=setup(tmp_path,monkeypatch,advance=advance,version=V6)
    try:
        out=app.submit(INPUT,'t1',sid)
        assert out['ok'] and app._run['turns'][-1]['response']==BODY
        assert len(wires)==1 and len(sockets)==1 and len(sockets[0].writes)==2
        row,=app.call_guard.snapshot()['attempts'];wire=json.loads(wires[0])
        assert row['status']=='received' and row['request_sha256']==hashlib.sha256(wires[0]).hexdigest()
        expired=advance>=180
        required=wire['response_format']['json_schema']['schema']['required']
        assert ('preference_control' in required) is (not expired)
        assert ('本人页面已实际展示的唯一待确认对象' in wire['messages'][0]['content']) is (not expired)
        assert json.loads(wire['messages'][-1]['content'])['current_user_message']==INPUT
        assert row['online_projection_requirements_binding']['requirements']['require_preference_control'] is (not expired)
        bound=row['online_context_admission'];assert bound['admission_before_context'] and bound['physical_rebuilds']==0
        assert bound['request_sha256']==row['request_sha256'] and row['physical_attempt']==1
        assert row['pacing']['wait_ms']==120000 and row['effective_physical_timeout_seconds']==90
        assert row['reservation']['input_token_plan_bound']==len(wires[0])+8192
        assert row['reservation']['output_token_limit']==8202
        spec=app.call_guard.policy['models'][row['model']]
        expected=(Decimal(str(spec['input_usd_per_million']))*(len(wires[0])+8192)+Decimal(str(spec['output_usd_per_million']))*8202)/1000000
        assert Decimal(row['reservation']['reserved_cost_usd'])==expected
        assert all(p.status=='proposed' for p in app.store.load(app._settled_owner).stated_preferences)
        if not expired:
            current=h['ctx'].working_memory['_preference_dialogue_invitation']
            assert current['expires_at']==h['pending']['expires_at'] and current['id']!=h['pending']['id']
            assert app.state()['preference_invitation']['status']=='awaiting_display'
            assert 0 in h['ctx'].working_memory['_sensitive_memory_excluded_turns']
        else:
            assert app.state()['preference_invitation'] is None
        assert app.submit(INPUT,'t1',sid)['duplicate'] and len(wires)==1
    finally:app.close()

@pytest.mark.parametrize('action',['confirm','decline'])
@pytest.mark.parametrize('version',[V6,V7])
def test_actual_committed_status_and_persistent_ui_remain(tmp_path,monkeypatch,action,version):
    app,wires,sockets,sid,h,c=setup(tmp_path,monkeypatch,action=action,version=version)
    try:
        result=app.submit(INPUT,'t1',sid);assert result['ok']
        assert app._run['turns'][-1]['response'].startswith(BODY+'\n')
        saved=app.store.load(app._settled_owner).stated_preferences[0]
        assert saved.status==('confirmed' if action=='confirm' else 'declined')
        assert h['ctx'].working_memory['online_turn_decision']['preference_body_composition']['status_from_committed_receipt']
        assert len(wires)==(3 if version==V7 else 1)
        if version==V7:
            assert app._run['turns'][-1]['response']==BODY+'\n'+(
                '好，这条记下了。' if action=='confirm' else '好，这条不记。')
            main,judge,review=map(json.loads,wires)
            assert 'preference_control' not in json.dumps(main,ensure_ascii=False)
            assert judge['response_format']['json_schema']['name']=='aria_confirm_judge'
            assert judge['max_tokens']==60 and judge['temperature']==0
            assert json.loads(judge['messages'][1]['content'])=={
                'invitation':{'do':saved.do,'scope':saved.scope},'utterance':INPUT}
            assert review['response_format']['json_schema']['name']=='aria_safety_review_v2'
            assert review['max_tokens']==1024
            assert [row['role'] for row in app.call_guard.snapshot()['attempts']]==[
                'online_fast','confirm_judge','safety_review']
    finally:app.close()

@pytest.mark.parametrize('advance',[0,59,60,61,179,180,181,301])
def test_v7_queued_expiry_uses_only_current_invitation_and_same_budget(tmp_path,monkeypatch,advance):
    app,wires,sockets,sid,h,c=setup(tmp_path,monkeypatch,advance=advance)
    try:
        out=app.submit(INPUT,'t1',sid)
        assert out['ok']
        eligible_at_online=advance<180
        # Both the original online slot and the judge slot consume 120s.
        # An invitation valid at online capture may expire before judge send.
        eligible_at_judge=advance<60
        expected_calls=3 if eligible_at_judge else 2
        assert len(wires)==len(sockets)==expected_calls
        assert all(len(sock.writes)==2 for sock in sockets)
        assert app._run['turns'][-1]['response']==BODY+(
            '\n这条我先不记，你想记的时候再说一声。' if eligible_at_online else '')
        main=json.loads(wires[0])
        assert 'preference_control' not in json.dumps(main,ensure_ascii=False)
        assert json.loads(main['messages'][-1]['content'])['current_user_message']==INPUT
        attempts=app.call_guard.snapshot()['attempts']
        assert [row['role'] for row in attempts]==(
            ['online_fast','confirm_judge','safety_review'] if eligible_at_judge
            else ['online_fast','safety_review'])
        for index,row in enumerate(attempts):
            assert row['status']=='received' and row['physical_attempt']==1
            assert row['request_sha256']==hashlib.sha256(wires[index]).hexdigest()
            assert row['reservation']['input_token_plan_bound']==len(wires[index])+8192
            assert row['reservation']['output_token_limit']=={
                'online_fast':8202,'confirm_judge':4166,'safety_review':5130}[row['role']]
        assert all(p.status=='proposed' for p in app.store.load(app._settled_owner).stated_preferences)
        # The additional paced body review crosses the original invitation's
        # deadline even at advance=0; it must not extend or redisplay consent.
        assert datetime.fromisoformat(h['pending']['expires_at'])<=c['utc']
        assert '_preference_dialogue_invitation' not in h['ctx'].working_memory
        assert app.state()['preference_invitation'] is None
        assert app.submit(INPUT,'t1',sid)['duplicate'] and len(wires)==expected_calls
    finally:app.close()

def test_false_state_claim_still_removed_and_truthful_feedback_retained(tmp_path,monkeypatch):
    app,wires,sockets,sid,h,c=setup(tmp_path,monkeypatch,false_claim=True)
    try:
        assert app.submit(INPUT,'t1',sid)['ok']
        delivered=app._run['turns'][-1]['response']
        assert '合成状态句。' not in delivered and delivered.startswith(BODY+'\n')
        assert h['ctx'].working_memory['online_turn_decision']['preference_body_composition']['removed_claim_segments']==0
        assert h['ctx'].working_memory['safety_review']['delete_sentence_ids']==[1]
        assert '合成状态句。' in json.loads(json.loads(wires[-1])['messages'][1]['content'])['candidate']
        assert app.store.load(app._settled_owner).stated_preferences[0].status=='proposed'
    finally:app.close()

@pytest.mark.parametrize('phase',['tls','after_headers'])
@pytest.mark.parametrize('change',['expiry','owner','session','record','disk'])
def test_late_authority_race_stops_without_replay(tmp_path,monkeypatch,phase,change):
    def mutate(h,clock):
        if change=='expiry':clock['utc']+=timedelta(seconds=400)
        if change=='owner':h['ctx'].user_id='wrong-owner'
        if change=='session':h['ctx'].session_id='wrong-session'
        if change=='record':h['persona'].stated_preferences[0].do='机械变化。'
        if change=='disk':
            p=h['app'].run_path;x=json.loads(p.read_text(encoding='utf8'));x['run_id']='external';p.write_text(json.dumps(x),encoding='utf8')
    app,wires,sockets,sid,h,c=setup(tmp_path,monkeypatch,mutation=mutate,phase=phase)
    try:
        if change=='disk':
            with pytest.raises(RuntimeError):app.submit(INPUT,'t1',sid)
            assert json.loads(app.run_path.read_text(encoding='utf8'))['run_id']=='external'
        else:assert not app.submit(INPUT,'t1',sid)['ok']
        assert len(wires)==1 and sum(len(s.writes) for s in sockets)==(1 if phase=='after_headers' else 0)
        assert app.call_guard.snapshot()['stopped'] and app._run['turns'][-1]['response']==''
        assert app.call_guard.snapshot()['paid_response_replays']==0
    finally:
        try:app.close()
        except (ValueError,RuntimeError,OSError):pass

def test_durable_profile_mutation_during_wait_cannot_be_overwritten(tmp_path,monkeypatch):
    def mutate(h,c):
        app=h['app'];p=app.store.load(app._settled_owner);p.stated_preferences[0].do='外部机械修改。';app.store.save(p)
    app,wires,sockets,sid,h,c=setup(tmp_path,monkeypatch,mutation=mutate,phase='waiting')
    try:
        assert not app.submit(INPUT,'t1',sid)['ok'] and not wires
        assert app.store.load(app._settled_owner).stated_preferences[0].do=='外部机械修改。'
        assert not app.call_guard.snapshot()['attempts']
    finally:app.close()
