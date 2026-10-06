"""Counterexamples for source attribution and preserved failed first bodies."""
from copy import deepcopy
import hashlib
import json

import pytest

from aria.bench.e1c_v2_runner import project_proposals
from aria.engine.context import EngineContext
from aria.engine.memory.stated_preferences import extract_preference_rows
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from scripts.e1c_v2_scoring import accepted_proposal_ids
from scripts.e1c_v2_qualification import bind_memory, bind_online, preflight_evidence
from scripts import e1b_evaluation as common


def candidate():
    return {"do":"报名时先问本人意愿","scope":"报名", "quotes":[{"turn_index":0,"text":"先问问我"}]}


@pytest.mark.parametrize("with_session",[False,True])
def test_expanded_source_quote_maps_to_actual_product_id(with_session):
    ctx=EngineContext(user_id="software",scene_id="companion_elder",session_id="software_session")
    ctx.add_turn("报名前先问问我，别替我决定。","知道了。")
    raw=candidate()
    if with_session:raw["quotes"][0]["session_id"]=ctx.session_id
    accepted,rejected=extract_preference_rows([raw],ctx,PersonaProfile(user_id="software"))
    assert len(accepted)==1 and not rejected
    projected,_=project_proposals([raw])
    consumed=set()
    assert accepted_proposal_ids(projected[0],[p.model_dump(mode="json") for p in accepted],ctx.session_id,consumed)==[accepted[0].id]
    assert accepted_proposal_ids(projected[0],[p.model_dump(mode="json") for p in accepted],ctx.session_id,consumed)==[]


def test_wrong_session_then_valid_candidate_attribution_is_not_reversed():
    ctx=EngineContext(user_id="software",scene_id="companion_elder",session_id="software_session")
    ctx.add_turn("报名前先问问我，别替我决定。","知道了。")
    wrong=candidate(); wrong["quotes"][0]["session_id"]="wrong_session"
    valid=candidate(); valid["quotes"][0]["session_id"]=ctx.session_id
    accepted,rejected=extract_preference_rows([wrong,valid],ctx,PersonaProfile(user_id="software"))
    assert len(accepted)==1 and rejected
    projected,_=project_proposals([wrong,valid])
    saved=[p.model_dump(mode="json") for p in accepted]; consumed=set()
    assert accepted_proposal_ids(projected[0],saved,ctx.session_id,consumed)==[]
    assert accepted_proposal_ids(projected[1],saved,ctx.session_id,consumed)==[accepted[0].id]
    other=deepcopy(valid); other["quotes"][0]["turn_index"]=1
    assert accepted_proposal_ids(other,saved,ctx.session_id,set())==[]


def memory_fixture(tmp_path, raw):
    directory=tmp_path/"capture"; directory.mkdir()
    visible=tmp_path/"visible.json"
    common.write_once(visible,{"content":raw})
    row={"request_id":"r","request_sha256":"a"*64,"session_id":"session", "visible_original_path":"visible.json",
        "client_return_content_sha256":hashlib.sha256(raw.strip().encode()).hexdigest()}
    proposals,_=project_proposals([candidate()])
    first={"session":1,"request_hash_kind":"actual_wire","request_id":"r","request_sha256":"a"*64,
        "physical_request_ids":["r"],"visible_response_sha256":row["client_return_content_sha256"],
        "field_present":True,"parse_status":"present_array","raw_proposals":proposals,"public_projection_dropped_fields":0}
    product={"product_status":"ok","accepted_preferences":[]}
    common.write_once(directory/"memory-s1.first.json",first)
    common.write_once(directory/"memory-s1.product.json",product)
    common.write_once(directory/"memory-s1.attempt.json",{"session_id":"session"})
    return directory,{**first,"complete":True,"product_outcome":product},[row]


@pytest.mark.parametrize("raw",[
    '{"stated_preferences":[],"stated_preferences":[]}',
    '{"stated_preferences":[],"x":1e309}',
    '{"stated_preferences":[]} {"stated_preferences":[]}',
    '{"stated_preferences":[]}',
])
def test_malformed_or_empty_original_cannot_be_relabelled_successful_candidate(tmp_path,raw):
    directory,session,rows=memory_fixture(tmp_path,raw)
    with pytest.raises(ValueError,match="projection_changed"):
        bind_memory(tmp_path,directory,session,rows)


def test_online_bad_body_cannot_be_relabelled_live_or_fake_fallback(tmp_path):
    common.write_once(tmp_path/"visible.json",{"content":"broken"})
    rows=[{"status":"received","visible_original_path":"visible.json"}]
    with pytest.raises(ValueError,match="bad_body"):
        bind_online(tmp_path,{"status":"completed","source":"live","response":"invented"},rows,operation_failed=False)
    with pytest.raises(ValueError,match="fallback_requires"):
        bind_online(tmp_path,{"status":"failed","source":"rule_fallback","response":"invented"},rows,operation_failed=True)
    assert bind_online(tmp_path,{"status":"failed","source":"rule_fallback","response":DETERMINISTIC_MODEL_FAILURE_REPLY},
        rows,operation_failed=True)=="rule_fallback"


def test_no_request_failure_is_not_safety_or_model_success(tmp_path):
    turn={"status":"failed", "source":"rule_fallback", "response":DETERMINISTIC_MODEL_FAILURE_REPLY}
    assert bind_online(tmp_path,turn,[],operation_failed=True)=="local_pre_send_failure"
    with pytest.raises(ValueError,match="fallback_requires"):
        bind_online(tmp_path,turn,[],operation_failed=False)
    with pytest.raises(ValueError,match="fallback_requires"):
        bind_online(tmp_path,{**turn,"response":"invented"},[],operation_failed=True)
    with pytest.raises(ValueError,match="fallback_requires"):
        bind_online(tmp_path,{**turn,"status":"completed"},[],operation_failed=True)


@pytest.mark.parametrize("source,status,failed,response",[
    ("live","completed",False,"unsent model answer"),
    ("rule_protection","completed",False,"unsent protected answer"),
    ("rule_safety","failed",True,"safety cannot hide failed operation"),
    ("rule_safety","completed",True,"safety cannot hide failed operation"),
    ("rule_safety","completed",False,"   "),
    ("model_failed","failed",True,""),
])
def test_no_request_unproven_delivery_is_rejected(tmp_path,source,status,failed,response):
    with pytest.raises(ValueError,match="no_request_requires"):
        bind_online(tmp_path,{"source":source,"status":status,"response":response},[],operation_failed=failed)


def test_no_request_successful_safety_remains_rule_safety(tmp_path):
    assert bind_online(tmp_path,{"source":"rule_safety","status":"completed","response":"local safety reply"},
        [],operation_failed=False)=="rule_safety"


def test_preflight_reason_only_comes_from_matching_persisted_event():
    turn={"turn_id":"op","session_id":"session"}
    start={"event":"started","token":"token","operation_id":"op","session_id":"session"}
    rejected={"event":"request_rejected","token":"token","role":"online_fast",
        "reason":"reservation_token_limit_exceeded","category":"configuration",
        "request_sha256":"a"*64,"wire_bytes":24000,"input_token_plan_bound":32192}
    assert preflight_evidence([start],turn)==[]
    assert preflight_evidence([start,rejected],turn)==[{k:v for k,v in rejected.items() if k!="event"}]
    assert preflight_evidence([start,{**rejected,"token":"another"}],turn)==[]
    assert preflight_evidence([start,{**rejected,"role":"memory"}],turn)==[]
    assert preflight_evidence([{**start,"session_id":"another"},rejected],turn)==[]
    assert preflight_evidence([{**start,"operation_id":"another"},rejected],turn)==[]
