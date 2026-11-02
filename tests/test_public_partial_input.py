"""Actual installed product transport, synthetic socket bytes, no model calls."""
from copy import deepcopy
from pathlib import Path
import io,json,ssl,http.client
import pytest
from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.max_release import tls_dispatch,tls_transport
from aria.engine.policy.max_release.held_input import validate_unreceived
from tests.test_public_held_input import forbid_network_and_real_keys,consent,fixture_client
from tests.test_installable_max_release import make_clients,prices,response

class Socket:
    def __init__(self,body):self.body=body;self.writes=[]
    def sendall(self,data,*a,**k):self.writes.append(bytes(data))
    def makefile(self,*a,**k):return io.BytesIO(self.body)
    def settimeout(self,*a):pass
    def setsockopt(self,*a):pass
    def close(self):pass
class TLS:
    verify_mode=ssl.CERT_REQUIRED;check_hostname=True
    def wrap_socket(self,sock,**kwargs):return sock

def setup(tmp_path,monkeypatch,sequence=('partial','partial','ok')):
    clients=make_clients();calls=[];sockets=[];steps=iter(sequence)
    def opened(request,**kwargs):
        calls.append(bytes(request.data));client=fixture_client(clients,request)
        mode=next(steps) if client.role=='online_fast' else 'ok'
        trace=kwargs['trace'];trace.preparation_recovery.bind_request_headers(request)
        raw=json.dumps(response(client),ensure_ascii=False).encode()
        if mode=='partial':body=b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw[:10]
        elif mode in ('complete_object','non_object'):
            if mode=='non_object':raw=b'[]'
            body=b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'+hex(len(raw))[2:].encode()+b'\r\n'+raw+b'\r\n'
        else:body=b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw
        conn=tls_transport._HTTPSConnection('localhost',trace=trace,context=TLS())
        def create(*a,**k):s=Socket(body);sockets.append(s);return s
        conn._create_connection=create
        conn.request('POST','/chat/completions',body=request.data,headers=dict(request.header_items()))
        return conn.getresponse()
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=10,model_stop_cost_usd='2',model_reservation_policy=prices())
    clock=[1000.];app.call_guard._pacing_clock=lambda:clock[0];app.call_guard._pacing_sleep=lambda n:clock.__setitem__(0,clock[0]+n)
    return app,calls,sockets,app.start()['state']['session_id']

@pytest.mark.parametrize('kind',['partial','complete_object','non_object'])
def test_two_partial_failures_each_require_owner_then_only_new_input(tmp_path,monkeypatch,kind):
    app,calls,sockets,sid=setup(tmp_path,monkeypatch,(kind,kind,'ok'))
    try:
        for n in (1,2):
            result=app.submit('中性软件输入。','t'+str(n),sid)
            assert not result['ok'] and result['held_unreceived_failure']
            state=deepcopy(app.call_guard.snapshot());row=state['attempts'][-1]
            assert state['stopped'] and row['status']=='failed' and row['usage']=={}
            assert row['body_read']['complete'] is False and validate_unreceived(row)>0
            assert app._run['turns'][-1]['response']=='' and app._run['turns'][-1]['status']=='failed'
            assert not app.submit('尚未确认的新输入。','blocked'+str(n),sid)['ok'] and len(calls)==n
            assert consent(app,sid,n)['request_sent'] is False and len(calls)==n
            after=app.call_guard.snapshot()
            assert after['attempts']==state['attempts'] and after['reserved_cost_usd']==state['reserved_cost_usd']
            assert app.submit('中性软件输入。','t'+str(n),sid)['duplicate'] and len(calls)==n
        assert app.submit('中性软件输入。','t3',sid)['ok'] and len(calls)==4
        assert app._run['turns'][-1]['safety_review']['status']=='reviewed'
        assert app.call_guard.snapshot()['unknown_cost_requests']==2
        assert app.call_guard.snapshot()['paid_response_replays']==0
    finally:app.close()

@pytest.mark.parametrize('accepted',[False,True])
def test_partial_restart_and_forget_retain_hold_and_consent(tmp_path,monkeypatch,accepted):
    app,calls,sockets,sid=setup(tmp_path,monkeypatch,('complete_object',))
    app.submit('仅用于软件测试的删除内容。','t1',sid)
    if accepted:assert consent(app,sid,1)['ok']
    before=deepcopy(app.call_guard.snapshot());app.close()
    app=MinimalWebApp(tmp_path,clients=make_clients(),model_max_requests=10,model_stop_cost_usd='2',model_reservation_policy=prices())
    try:
        assert app.call_guard.snapshot()['stopped'] is not accepted
        app.forget_run();after=app.call_guard.snapshot()
        assert after['attempts']==before['attempts'] and after['reserved_cost_usd']==before['reserved_cost_usd']
        assert after['stopped'] is not accepted
        assert all(validate_unreceived(r)>0 for r in after['attempts'])
        assert not app.submit('中性软件输入。','t1',sid)['ok'] and len(calls)==1
    finally:app.close()

@pytest.mark.parametrize('kind',['origin','reader','identity','capture','usage','http','complete','public','run'])
def test_partial_actual_evidence_required(tmp_path,monkeypatch,kind):
    app,calls,sockets,sid=setup(tmp_path,monkeypatch,('partial',))
    try:
        result=app.submit('中性软件输入。','t1',sid);assert result['held_unreceived_failure']
        row=deepcopy(app.call_guard.snapshot()['attempts'][0])
        if kind=='origin':row['actual_partial_body_failure_origin']['caught_at_actual_bounded_reader']=False
        elif kind=='reader':row['actual_partial_body_failure_origin']['reader_source']['sha256']='0'*64
        elif kind=='identity':row['actual_partial_body_failure_origin']['request_id']='other'
        elif kind=='capture':row['actual_partial_body_failure_origin']['body_receipt']['sha256']='0'*64
        elif kind=='usage':row['usage']={'cost':0}
        elif kind=='http':row['http_status']=429
        elif kind=='complete':row['body_read']['complete']=True
        elif kind=='public':Path(row['actual_partial_body_failure_origin']['public_first_output_path']).write_text('{}')
        else:
            path=app.run_dir/'run.json';raw=path.read_bytes();value=json.loads(raw);value['run_id']='other-owner-run';path.write_text(json.dumps(value))
            assert not app.state()['held_new_input']['available']
            assert app.call_guard.snapshot()['stopped']
            assert path.read_bytes()!=raw and len(calls)==1
            return
        with pytest.raises((ValueError,KeyError)):validate_unreceived(row)
    finally:
        try:app.close()
        except (ValueError,RuntimeError,OSError):pass

def test_checker_incomplete_read_cannot_become_actual_reader_origin(tmp_path,monkeypatch):
    app,calls,sockets,sid=setup(tmp_path,monkeypatch,('ok',))
    def fail():raise http.client.IncompleteRead(b'local',1)
    app.call_guard._held_prewrite_readers.append(fail)
    try:
        assert not app.submit('中性软件输入。','t1',sid)['ok']
        assert not app.state()['held_new_input']['available'] and app.call_guard.snapshot()['stopped']
        assert not any('actual_partial_body_failure_origin' in row for row in app.call_guard.snapshot()['attempts'])
    finally:app.call_guard._held_prewrite_readers.remove(fail);app.close()
