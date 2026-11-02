"""Ordinary product continuation after positive local TLS preparation failure."""
from copy import deepcopy
import io,json,ssl
import pytest
from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.max_release import tls_dispatch,tls_transport
from aria.engine.policy.max_release.held_input import validate_unreceived
from tests.test_public_held_input import forbid_network_and_real_keys,consent,fixture_client
from tests.test_installable_max_release import make_clients,prices,response,Response

class Socket:
    def __init__(self):self.writes=[]
    def sendall(self,data,*a,**k):self.writes.append(bytes(data))
    def makefile(self,*a,**k):return io.BytesIO(b'')
    def settimeout(self,*a):pass
    def setsockopt(self,*a):pass
    def close(self):pass

def setup(tmp_path,monkeypatch,failures=(1,2)):
    clients=make_clients();calls=[];sockets=[];preparations=[]
    def opened(request,**kwargs):
        calls.append(bytes(request.data))
        if len(calls) not in failures:return Response(response(fixture_client(clients,request)))
        trace=kwargs['trace'];trace.preparation_recovery.bind_request_headers(request)
        class TLS:
            verify_mode=ssl.CERT_REQUIRED;check_hostname=True
            def wrap_socket(self,sock,**kwargs):
                preparations.append(sock);raise ssl.SSLEOFError(8,'neutral software TLS EOF')
        conn=tls_transport._HTTPSConnection('localhost',trace=trace,context=TLS())
        def create(*a,**k):s=Socket();sockets.append(s);return s
        conn._create_connection=create
        conn.request('POST','/chat/completions',body=request.data,headers=dict(request.header_items()))
        raise AssertionError('TLS preparation must fail before application')
    monkeypatch.setattr(tls_dispatch,'open_observed',opened)
    app=MinimalWebApp(tmp_path,clients=clients,model_max_requests=10,model_stop_cost_usd='2',model_reservation_policy=prices())
    clock=[1000.];app.call_guard._pacing_clock=lambda:clock[0];app.call_guard._pacing_sleep=lambda n:clock.__setitem__(0,clock[0]+n)
    return app,calls,sockets,preparations,app.start()['state']['session_id']

def test_two_prewrite_failures_explicitly_accepted_then_distinct_next_input(tmp_path,monkeypatch):
    app,calls,sockets,preps,sid=setup(tmp_path,monkeypatch)
    try:
        for n in (1,2):
            result=app.submit('中性软件输入。','t'+str(n),sid)
            assert not result['ok'] and result['held_unreceived_failure']
            before=deepcopy(app.call_guard.snapshot());row=before['attempts'][-1]
            assert before['stopped'] and row['usage']=={} and row['status']=='failed'
            assert row['fee_verification_error']=='finite_no_http_response_cost_unverified'
            assert before['stop_diagnostic']['exception_kind']=='tls_eof_error'
            assert before['stop_diagnostic']['request_id']==row['request_id']
            assert row['held_noresponse_origin']['origin_kind']=='tls_preparation_failed'
            assert len(preps)==2*n and all(s.writes==[] for s in sockets)
            assert validate_unreceived(row)>0
            assert not app.submit('尚未确认的新输入。','blocked'+str(n),sid)['ok'] and len(calls)==n
            assert consent(app,sid,n)['request_sent'] is False and len(calls)==n
            after=app.call_guard.snapshot();assert after['attempts']==before['attempts'] and after['reserved_cost_usd']==before['reserved_cost_usd']
            assert not after['stopped'] and 'stop_diagnostic' not in after
            assert app.submit('中性软件输入。','t'+str(n),sid)['duplicate'] and len(calls)==n
        assert app.submit('中性软件输入。','t3',sid)['ok'] and len(calls)==4
        assert app._run['turns'][-1]['safety_review']['status']=='reviewed'
        state=app.call_guard.snapshot();assert state['unknown_cost_requests']==2 and state['paid_response_replays']==0
    finally:app.close()

@pytest.mark.parametrize('accepted',[False,True])
def test_prewrite_restart_and_forget_preserve_explicit_consent_and_hold(tmp_path,monkeypatch,accepted):
    app,calls,sockets,preps,sid=setup(tmp_path,monkeypatch,failures=(1,))
    app.submit('仅用于软件测试的删除内容。','t1',sid)
    if accepted:assert consent(app,sid,1)['ok']
    before=deepcopy(app.call_guard.snapshot());app.close()
    app=MinimalWebApp(tmp_path,clients=make_clients(),model_max_requests=10,model_stop_cost_usd='2',model_reservation_policy=prices())
    try:
        assert app.call_guard.snapshot()['attempts']==before['attempts']
        assert app.call_guard.snapshot()['stopped'] is not accepted
        app.forget_run();after=app.call_guard.snapshot()
        assert after['attempts']==before['attempts'] and after['reserved_cost_usd']==before['reserved_cost_usd']
        assert after['stopped'] is not accepted
        assert not app.submit('中性软件输入。','t1',sid)['ok'] and len(calls)==1
    finally:app.close()

@pytest.mark.parametrize('kind',['SSLEOFError','TimeoutError'])
def test_local_permission_or_storage_checker_exception_is_not_tls_proof(tmp_path,monkeypatch,kind):
    app,calls,sockets,preps,sid=setup(tmp_path,monkeypatch,failures=(1,))
    def fail():
        if kind=='SSLEOFError':raise ssl.SSLEOFError(8,'neutral local check')
        raise TimeoutError('neutral local check')
    app.call_guard._held_prewrite_readers.append(fail)
    try:
        result=app.submit('中性软件输入。','t1',sid)
        assert not result['ok'] and not app.state()['held_new_input']['available']
        assert app.call_guard.snapshot()['stopped'] and all(s.writes==[] for s in sockets)
    finally:app.call_guard._held_prewrite_readers.remove(fail);app.close()

@pytest.mark.parametrize('kind',['missing_attempt','third_attempt','completed_tls','app_write','usage','role','extra','deadline','origin','checker','before'])
def test_prewrite_positive_facts_and_original_binding_are_required(tmp_path,monkeypatch,kind):
    app,calls,sockets,preps,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        app.submit('中性软件输入。','t1',sid);row=deepcopy(app.call_guard.snapshot()['attempts'][0])
        if kind=='missing_attempt':row['transport_events']['connection_preparations'].pop()
        elif kind=='third_attempt':row['transport_events']['connection_preparations'].append(deepcopy(row['transport_events']['connection_preparations'][-1]))
        elif kind=='completed_tls':row['transport_events']['events']['tls_completed']=True
        elif kind=='app_write':row['transport_events']['connection_preparations'][0]['events']['application_write_attempted']=True
        elif kind=='usage':row['usage']={'cost':0}
        elif kind=='role':row['role']='memory'
        elif kind=='extra':row['tls_preparation_candidate']['logical_extra_after']=2
        elif kind=='deadline':row['effective_physical_timeout_seconds']=91
        elif kind=='origin':row['held_noresponse_origin']['origin_kind']='remote_disconnected'
        elif kind=='checker':row['held_noresponse_origin']['local_pending_check_failed']=True
        else:row['held_online_before_write']['bound_before_application_write']=False
        with pytest.raises((ValueError,KeyError)):validate_unreceived(row)
    finally:app.close()
