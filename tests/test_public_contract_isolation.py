"""Mechanical installed entry checks; no model or semantic success claims."""
from copy import deepcopy
import json
from decimal import Decimal
import pytest
from tests.test_installable_settled_validation import setup, no_network_credentials
from aria.engine.policy.max_release.contract_state import FIELD,OUTCOME


@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('empty',[False,True])
def test_two_failed_answers_keep_prelude_and_continue(tmp_path,monkeypatch,protected,empty):
    app,calls,sid=setup(tmp_path,monkeypatch,empty=empty)
    try:
        preludes=[]
        for number in (1,2):
            result=app.submit('我想自杀。' if protected else '中性软件输入。',
                't'+str(number),sid,on_safety_prelude=lambda x:preludes.append(x) or True)
            assert not result['ok'] and result['settled_validation_failure']
            assert result['continue_next_input'] and not result['response_delivered']
            assert result['protection_prelude_delivered'] is protected
            assert not app.state()['requires_review']
            failed=app._run['turns'][-1]
            assert failed['status']=='failed' and failed['response']==''
            assert bool(failed.get('protection_prelude')) is protected
            if protected:
                assert failed['premodel_task']['status']=='response_only_failed'
                assert failed['premodel_task']['client_entered']
        original=deepcopy(app._run['turns'])
        assert app.submit('下一条中性输入。','t3',sid)['ok']
        assert len(calls)==3 and app._run['turns'][:2]==original
        assert len(preludes)==(2 if protected else 0)
        rows=app.call_guard.snapshot()['attempts']
        assert all(r['status']=='received' and Decimal(str(r['usage']['cost']))==Decimal('0.0001') for r in rows)
    finally:app.close()


@pytest.mark.parametrize('field',['business_before_sha256','preapplication','protection_prelude_delivered'])
def test_hash_metadata_is_bound_to_independent_receipt(tmp_path,monkeypatch,field):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        assert app.submit('我想自杀。','t1',sid,on_safety_prelude=lambda x:True)['continue_next_input']
        path,=app._settled_validation_directory().glob('*.json');value=json.loads(path.read_text())
        if field=='preapplication':value[field]['owner_id']='other'
        elif field=='protection_prelude_delivered':value[field]=False
        else:value[field]='0'*64;value['business_after_sha256']='0'*64
        path.write_text(json.dumps(value),encoding='utf-8')
        assert app.state()['requires_review']
        assert not app.submit('中性新输入。','t2',sid)['ok'] and len(calls)==1
    finally:app.close()


def test_hash_only_binding_has_no_private_prelude_or_input(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    try:
        marker='我想自杀。PRIVATE_FIXTURE'
        assert app.submit(marker,'t1',sid,on_safety_prelude=lambda x:True)['continue_next_input']
        row=app.call_guard.snapshot()['attempts'][0]
        path,=app._settled_validation_directory().glob('*.json')
        raw=path.read_text()+json.dumps(row[FIELD])+json.dumps(row[OUTCOME])
        assert marker not in raw and 'protection"' not in raw and 'engine_profile' not in raw
        before=app.call_guard.snapshot()
        app.forget_run()
        assert not list(app._settled_validation_directory().glob('*.json'))
        assert app.call_guard.snapshot()['known_cost_usd']==before['known_cost_usd']
        assert app.state()['turns']==[]
    finally:app.close()


def test_outcome_storage_failure_stops_without_second_call(tmp_path,monkeypatch):
    app,calls,sid=setup(tmp_path,monkeypatch,failures=(1,))
    def fail(*a,**k):raise OSError('local outcome storage failed')
    monkeypatch.setattr(app.call_guard,'seal_contract_disposition',fail)
    try:
        with pytest.raises(OSError):
            app.submit('我想自杀。','t1',sid,on_safety_prelude=lambda x:True)
        assert app.call_guard.snapshot()['stopped'] and app.state()['requires_review']
        assert not app.submit('中性新输入。','t2',sid)['ok'] and len(calls)==1
    finally:app.close()
