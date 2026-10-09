"""Explicit synthetic transport; no holdout text, network, or credentials."""
from tests._model_http_stub import stub_model_http
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import e1b_evaluation as common
from scripts import e1c_execution as old
from scripts import e1c_v2_execution as entry
from scripts import e1c_v2_qualification as qualification
from scripts import e1c_v2_scoring as scoring
from tests.test_e1c_execution import setup as old_setup, save, no_network_or_env  # noqa: F401
from aria.engine.policy import minimal_calls


def setup(tmp_path, monkeypatch, *, limit="0.5", approve=True, full_source=False):
    campaign, frozen, _auth = old_setup(tmp_path/"previous")
    _,source=old.verify_authority(campaign,_auth)
    with closing(old.guard_code.E1cCallGuard(campaign/"allowance.sqlite3",campaign=campaign,suite=source.suite,
        authority_check=lambda:None, max_requests=3192,max_observed_cost_usd=frozen["new_allowance_usd"],
        reservation_policy=frozen["reservation_policy"],stage="main",registration_sha256=frozen["seal_sha256"])) as guard:
        state=guard.snapshot()
    receipt={"status":"not_evaluable","registration_sha256":frozen["seal_sha256"],
        "inherited_finance":frozen["finance"],"accounting":state}
    receipt["receipt_sha256"]=common.digest(receipt)
    save(campaign/"execution-receipt.json",receipt)
    prior_manifest=Path(frozen["history"]["unknown_manifest"]["path"])
    history={k:{"path":str(p),"sha256":entry.sha(p)} for k,p in (
        ("registration",campaign/"registration.json"),("receipt",campaign/"execution-receipt.json"),
        ("ledger",campaign/"allowance.sqlite3"),("unknown_manifest",prior_manifest))}
    # Active product collaborators can edit unrelated modules during these
    # transport tests; formal source_inventory remains the complete inventory.
    owned=[entry.ROOT/"scripts"/n for n in entry.SCRIPT_DEPENDENCIES if "e1c_v2" in n]
    owned.append(entry.ROOT/"aria/bench/e1c_v2_runner.py")
    if not full_source:
        monkeypatch.setattr(entry,"source_inventory",lambda:[{"path":str(p),"sha256":entry.sha(p)} for p in owned])
    target=tmp_path/"v2"
    frozen=entry.prepare(target,source_directory=source.directory,first_receipt_sha256=source.receipt_sha256,
        history=history,reservation_policy=frozen["reservation_policy"],seed=20261006,
        e1c_limit_usd=limit,cumulative_limit_usd="3",mode="software_fixture")
    human=tmp_path/"SOFTWARE_ONLY_APPROVAL.json"; save(human,{"software_only":True})
    auth=entry.authorize_template(frozen)
    if approve:
        auth.update(status="approved",human_approval_file=str(human),human_approval_file_sha256=entry.sha(human))
    save(target/"authorization.json",auth)
    return target,frozen,common.digest(auth)


class FastGuard(entry.guard_code.E1cV2CallGuard):
    software_fixture_only=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        clock=[0.0]
        self._pacing_clock=lambda:clock[0]
        self._pacing_sleep=lambda n:clock.__setitem__(0,clock[0]+n)


def transport(monkeypatch, mutate=None):
    requests=[]
    class Response:
        status,headers=200,{}
        def __init__(self,value):self.value=value
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,size):return json.dumps(self.value).encode()[:size]
    def opened(request,**kwargs):
        payload=json.loads(request.data); requests.append(deepcopy(payload))
        if "anthropic" not in payload["model"]:
            value={"response":"【软件夹具】我先听您说。","state_signals":[],
                "current_affect":{"label":"unknown","confidence":0.3,"divergence":{"possible":False,"confidence":0.2,"pattern":""}},
                "selected_intervention_id":"","uncertainty":0.6,
                "communication_plan":{"pace":"open_listening","support_order":"先听本人说","boundary":"不强加话题"},
                "safety_concern":"none"}
        elif "protocol_version" in payload["messages"][0]["content"]:
            value={"protocol_version":"2.0","reviews":[],"changes":[]}
        else:
            value={"episodes":[],"relationships":[],"stated_preferences":[],"cognitive_privacy":[]}
        response={"id":f"software-{len(requests)}","model":payload["model"],
            "provider":"Anthropic" if "anthropic" in payload["model"] else "Alibaba",
            "choices":[{"finish_reason":"stop","message":{"role":"assistant","content":json.dumps(value,ensure_ascii=False)}}],
            "usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20,"cost":0.0000005,
                "completion_tokens_details":{"reasoning_tokens":0}}}
        if mutate:mutate(len(requests),response,payload)
        return Response(response)
    opened.software_fixture_only=True
    monkeypatch.setattr(minimal_calls,"open_observed",opened)
    stub_model_http(monkeypatch, opened)
    return requests


def execute(campaign, auth, *, full=False):
    def no_key():raise AssertionError("software must never load credentials")
    return entry.execute(campaign,authorization_sha256=auth,credential_loader=no_key,
        software_first_case_only=not full,guard_factory=FastGuard)


def test_fixed_ten_inventory_inclusive_budget_and_original_unchanged(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch)
    _,source=entry.verify_authority(campaign,auth)
    assert frozen["case_ids"]==[c.id for c in source.suite.items][2:]
    assert frozen["basic_counts"]=={"online_fast":840,"memory":20,"reflection":20}
    assert frozen["reply_positions"]==180 and frozen["formation_cases"]==10
    assert frozen["new_allowance_usd"]=="0.5"
    assert old.guard_code.PHYSICAL_MAX==3192
    assert frozen["finance"]["inherited_unknown_count"]==9


def test_one_case_product_v3_88_requests_no_holdout_no_credentials(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch)
    calls=transport(monkeypatch)
    result=execute(campaign,auth)
    assert result["status"]=="software_segment_complete",result
    assert len(calls)==88
    assert all('online_turn_decision_v3' in c['messages'][0]['content'] for c in calls if 'anthropic' not in c['model'])
    entry.guard_code.closed_ledger(campaign,frozen,result["accounting"])


@pytest.mark.parametrize("failure",["online","memory"])
def test_known_bad_body_is_preserved_product_fallback_or_seal_then_continues(tmp_path,monkeypatch,failure):
    campaign,frozen,auth=setup(tmp_path,monkeypatch)
    def mutate(index,response,payload):
        if index==(1 if failure=="online" else 13):
            response["choices"][0]["message"]["content"]="{invalid software body"
    calls=transport(monkeypatch,mutate)
    result=execute(campaign,auth)
    assert result["status"]=="software_segment_complete",result
    assert len(calls)==(88 if failure=="online" else 87)
    report=common.read_json(Path(result["run_root"])/"main/e1c_b1/report.json")
    if failure=="online":
        first=report["shared"]["state"]["turns"][0]
        assert first["source"]=="rule_fallback" and first["response"]
    else:
        assert report["shared"]["review_outcomes"]==["failed","completed"]
        assert not (Path(result["run_root"])/"main/e1c_b1/reflection-s1.first.json").exists()
    assert sum(e.get("failed") is True for e in result["accounting"]["product_events"])==1
    entry.guard_code.closed_ledger(campaign,frozen,result["accounting"])


def test_three_logical_failures_pause_without_automatic_resume(tmp_path,monkeypatch):
    campaign,_,auth=setup(tmp_path,monkeypatch)
    def mutate(index,response,payload):response["choices"][0]["message"]["content"]="broken"
    calls=transport(monkeypatch,mutate)
    result=execute(campaign,auth)
    assert len(calls)==3 and result["status"]=="not_evaluable"
    assert result["accounting"]["stop_reason"]=="consecutive_failures"
    assert not any(e["event"]=="resumed" for e in result["accounting"]["product_events"])


def test_unknown_fee_stops_and_retains_full_reservation(tmp_path,monkeypatch):
    campaign,_,auth=setup(tmp_path,monkeypatch)
    def mutate(index,response,payload):response["usage"].pop("cost")
    calls=transport(monkeypatch,mutate);result=execute(campaign,auth)
    assert len(calls)==1 and result["status"]=="not_evaluable"
    assert result["accounting"]["unknown_cost_requests"]==1
    assert result["accounting"]["reserved_cost_usd"]!="0"


def test_unapproved_and_modified_authority_never_reads_key(tmp_path,monkeypatch):
    campaign,_,auth=setup(tmp_path,monkeypatch,approve=False)
    transport(monkeypatch)
    with pytest.raises(ValueError):execute(campaign,auth)
    assert not (campaign/"allowance.sqlite3").exists()


def test_complete_880_software_positions_180_deliveries_zero_exposure_and_first_labels(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch)
    calls=transport(monkeypatch)
    result=execute(campaign,auth,full=True)
    assert result["status"]=="generated",result
    assert len(calls)==880
    inventory=qualification.qualify(campaign,authorization_sha256=auth)
    assert inventory["exposure"]["denominator"]==300 and inventory["exposure"]["numerator"]==0
    assert inventory["exposure"]["gate"]=="not_evaluable"
    assert inventory["format"]["fallbacks"]==0
    bundle=tmp_path/"bundle"; created=scoring.export_bundle(campaign,bundle,authorization_sha256=auth,blind_seed=7)
    packet=common.read_json(bundle/"reply/blind/packet.json")
    template=scoring.reply.label_template(packet)
    template["scorer"]={"id":"software_independent_labeler","independent":True,"authored_cases":False}
    for row in template["labels"]:
        row["labels"]={key:0 for key in row["labels"]}
    imported=scoring.import_first_batch(bundle,created["bundle_sha256"],common.canonical(template),
        kind="reply",batch_id="software-first",expected_ids=[i["id"] for i in packet["items"]])
    assert imported["status"]=="first_batch_imported",imported
    assessed=scoring.assess(bundle,created["bundle_sha256"])
    assert assessed["main_gate"]=="software_fixture_only" and assessed["effect_claim_eligible"] is False
    assert assessed["mechanical_gates"]["main_gate"]=="not_evaluable"
    with pytest.raises(ValueError,match="already_reserved"):
        scoring.import_first_batch(bundle,created["bundle_sha256"],common.canonical(template),kind="reply",
            batch_id="software-second",expected_ids=[i["id"] for i in packet["items"]])


def test_complete_product_source_archive_with_both_output_failures(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch,full_source=True)
    def mutate(index,response,payload):
        if index in (1,13):response["choices"][0]["message"]["content"]="broken software first answer"
    calls=transport(monkeypatch,mutate); result=execute(campaign,auth)
    assert result["status"]=="software_segment_complete",result
    assert len(calls)==87
    assert len(frozen["source"])>130
    entry.verify_authority(campaign,auth)
    entry.guard_code.closed_ledger(campaign,frozen,result["accounting"])


def test_full_source_unknown_fee_before_second_wire(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch,full_source=True)
    def mutate(index,response,payload):response["usage"].pop("cost")
    calls=transport(monkeypatch,mutate); result=execute(campaign,auth)
    assert len(calls)==1 and result["accounting"]["unknown_cost_requests"]==1
    assert result["status"]=="not_evaluable"
    entry.guard_code.closed_ledger(campaign,frozen,result["accounting"])
    tampered=deepcopy(result["accounting"]); tampered.update(stopped=False,stop_reason=None)
    with pytest.raises(ValueError,match="unknown_hidden"):
        entry.guard_code.closed_ledger(campaign,frozen,tampered)


def test_product_owner_resume_is_not_frozen_experiment_permission(tmp_path,monkeypatch):
    campaign,frozen,auth=setup(tmp_path,monkeypatch)
    _,source=entry.verify_authority(campaign,auth)
    with closing(FastGuard(campaign/"allowance.sqlite3",campaign=campaign,
        suite=entry.heldout_submission(source).suite,authority_check=lambda:None,
        registration_sha256=frozen["seal_sha256"],max_requests=frozen["physical_max"],
        max_observed_cost_usd=frozen["new_allowance_usd"],reservation_policy=frozen["reservation_policy"])) as guard:
        for index in range(3):
            with guard.scope("turn",f"software-fail-{index}","software-session"):
                guard.note_output_failure()
        state=guard.snapshot()
        guard.resume("software-owner-resume",state["pause_id"])
        state=guard.snapshot()
        assert state["stopped"] is False
    with pytest.raises(ValueError,match="resume_not_registered"):
        entry.guard_code.closed_ledger(campaign,frozen,state)
