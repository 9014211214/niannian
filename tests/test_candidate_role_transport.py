"""New role boundaries using synthetic HTTP and retained durable ledgers only."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import json
import sqlite3
import urllib.error
import io

import pytest

from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.offline_reasoning import build_offline_reasoning_route, MODEL, PROVIDER
from aria.engine.policy.product_calls import ProductCallGuard
from aria.engine.policy.reflection_json import reflection_body_system_prompt, reflection_body_observation
from aria.engine.policy.reserved_calls import ReservedCallGuard
from aria.engine.policy.safety_structured import safety_request_format, validate_safety_review_request
from aria.engine.risk.safety_review import safety_review_messages
from tests.test_online_structured import route as online_route
from tests.test_reserved_calls import stub, receipt, policy as old_policy
from tests._model_http_stub import stub_model_http


def reasoning_metadata():
    return ({"data": {"id": MODEL, "canonical_slug": MODEL + "-fixture",
        "reasoning": {"mandatory": True, "supported_efforts": ["medium", "high"]}}},
        {"data": {"id": MODEL, "endpoints": [{"model_id": MODEL, "tag": PROVIDER,
            "name": "Google | fixture", "provider_name": "Google", "status": 0,
            "max_completion_tokens": 65536,
            "supported_parameters": ["reasoning", "max_tokens", "temperature", "response_format", "structured_outputs"],
            "pricing": {"prompt": "0.000002", "completion": "0.00001", "internal_reasoning": "0.00001"}}]}})


def reasoning_route():
    a, b = reasoning_metadata()
    return build_offline_reasoning_route(json.dumps(a).encode(), json.dumps(b).encode())


def client(role="reflection", **kwargs):
    options = dict(api_key="inert-fixture", base_url="https://fixture.invalid/v1",
                   role=role, bounded_transport=True, max_retries=0, fallback_models=[])
    if role == "reflection":
        options.update(model=MODEL, reflection_return_format="body_json", offline_reasoning_route=reasoning_route(),
                       reflection_itemwise_review=True)
    elif role == "safety_review":
        options.update(model="qwen/qwen3.8-flash", safety_structured_route=online_route(), timeout=90)
    elif role == "memory":
        options.update(model="anthropic/claude-sonnet-5", memory_prefix_recovery=True)
    return OpenRouterLLMClient(**{**options, **kwargs})


def policy(c):
    route = c.offline_reasoning_route or c.safety_structured_route
    if route is None:
        return old_policy()
    binding = route.binding()
    return {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {c.model: {
        "input_usd_per_million": "2", "output_usd_per_million": "10",
        "provider": binding["provider"], "response_models": [c.model, binding["canonical_slug"]],
        "response_providers": [binding["provider_name"]], "max_input_tokens": 100000,
        "max_output_tokens": 4096, "metadata_sha256": binding["endpoint_metadata_sha256"]}}}


def guard(path, c, cls=ReservedCallGuard, **kwargs):
    return cls(path, max_requests=30, max_observed_cost_usd="1", reservation_policy=policy(c), **kwargs)


def messages(c):
    if c.role == "safety_review":
        return safety_review_messages("虚构风险输入", "候选第一句。候选第二句。",
            {"concern": "fall", "subject": "owner", "time": "current", "urgency": "immediate",
             "quote": "虚构风险输入", "verbatim_source": True, "disposition": "urgent_owner",
             "semantic_verified": False, "conversation_history": []}, "先交付必要保护。")
    if c.role == "memory":
        return [{"role": "user", "content": "虚构会话"}]
    return [{"role": "system", "content": reflection_body_system_prompt()},
            {"role": "user", "content": "虚构证据"}]


def response(c):
    data = receipt(c)
    if c.role == "reflection":
        data["provider"] = "Google"
        data["usage"]["completion_tokens_details"] = {"reasoning_tokens": 2}
        data["choices"][0]["message"]["content"] = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
    elif c.role == "safety_review":
        data["provider"] = "Alibaba"
        data["choices"][0]["message"]["content"] = '{"conflicts":true,"delete_sentence_ids":[1]}'
    return data


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unstubbed network"))


def test_offline_real_wire_bound_to_public_metadata_and_inclusive_completion(tmp_path, monkeypatch):
    c = client()
    calls = stub(monkeypatch, response(c))
    with closing(guard(tmp_path / "ledger", c)) as g:
        g.attach(c)
        with g.scope("review", "op1", "session1"):
            raw = c.complete(messages(c))
        row, = g.snapshot()["attempts"]
        assert row["status"] == "received" and row["usage"]["reasoning_tokens"] == 2
        assert row["reservation"]["output_token_limit"] == 2800
        assert raw.startswith('{"protocol_version"')
        binding = json.loads(g._db.execute("SELECT binding FROM roles").fetchone()[0])
        assert binding["offline_reasoning_route"]["completion_bound_includes_reasoning"]
        assert binding["reflection_itemwise_review"] is True
    wire, = calls
    assert wire["reasoning"] == {"effort": "medium", "exclude": True}
    assert wire["provider"]["only"] == [PROVIDER]
    assert wire["max_tokens"] == 2800 and "max_completion_tokens" not in wire
    assert not {"tools", "response_format", "plugins", "models"} & set(wire)


@pytest.mark.parametrize("mutation", ["missing_count", "negative", "over_total", "hidden", "wrong_provider", "unknown_cost", "over_bound"])
def test_reasoning_receipt_rejection_preserves_fee_or_hold_and_no_hidden_text(tmp_path, monkeypatch, mutation):
    c = client(); data = response(c)
    if mutation == "missing_count": data["usage"].pop("completion_tokens_details")
    elif mutation == "negative": data["usage"]["completion_tokens_details"]["reasoning_tokens"] = -1
    elif mutation == "over_total": data["usage"]["completion_tokens_details"]["reasoning_tokens"] = 4
    elif mutation == "hidden": data["choices"][0]["message"]["reasoning_details"] = [{"text": "PRIVATE_CHAIN"}]
    elif mutation == "wrong_provider": data["provider"] = "Google AI Studio"
    elif mutation == "unknown_cost": del data["usage"]["cost"]
    elif mutation == "over_bound": data["usage"]["completion_tokens"] = 2801
    calls = stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger", c)) as g:
        g.attach(c)
        with g.scope("review", "op1", "session1"), pytest.raises(MinimalCallStopped):
            c.complete(messages(c))
        state = g.snapshot()
        assert state["attempts"][0]["status"] == "failed" and len(calls) == 1
        assert "PRIVATE_CHAIN" not in json.dumps(state) + json.dumps(c.last_diagnostic) + json.dumps(c.last_usage)
        if mutation == "unknown_cost":
            assert state["unknown_cost_requests"] == 1 and Decimal(state["reserved_cost_usd"]) > 0
        else:
            assert Decimal(state["known_cost_usd"]) == Decimal("0.00001")


@pytest.mark.parametrize("mutation", ["route", "effort", "price", "mandatory", "missing_parameter", "duplicate"])
def test_offline_capability_cannot_fall_back_or_underprice(mutation):
    a,b = reasoning_metadata(); opts = {}
    ep = b["data"]["endpoints"][0]
    if mutation == "route": opts["provider"] = "google-ai-studio/flex"
    elif mutation == "effort": opts["effort"] = "high"
    elif mutation == "price": ep["pricing"]["internal_reasoning"] = "0.00002"
    elif mutation == "mandatory": a["data"]["reasoning"]["mandatory"] = False
    elif mutation == "missing_parameter": ep["supported_parameters"].remove("max_tokens")
    elif mutation == "duplicate": b["data"]["endpoints"].append(deepcopy(ep))
    with pytest.raises(ValueError): build_offline_reasoning_route(json.dumps(a).encode(), json.dumps(b).encode(), **opts)


def test_new_bound_options_cannot_rebind_retained_ledger(tmp_path):
    c=client(); path=tmp_path/"ledger"
    with closing(guard(path,c)) as g: g.attach(c)
    c=client(reflection_itemwise_review=False)
    with closing(guard(path,c)) as g, pytest.raises(ValueError, match="retained allowance"):
        g.attach(c)


@pytest.mark.parametrize("prefix", [False, True])
def test_only_opted_memory_delivers_settled_length_body_with_finish_reason(tmp_path, monkeypatch, prefix):
    c=client("memory", memory_prefix_recovery=prefix); c.max_completion_tokens=4000
    data=response(c); data["choices"][0].update(finish_reason="length")
    data["choices"][0]["message"]["content"]='{"cognitive_privacy":[],"cognitive_observations":[],"stated_preferences":['
    calls=stub(monkeypatch,data)
    with closing(guard(tmp_path/"ledger",c)) as g:
        g.attach(c)
        with g.scope("review","op1","session1"):
            if prefix:
                assert c.complete(messages(c)) == data["choices"][0]["message"]["content"]
                assert c.last_finish_reason == "length"
                assert g.snapshot()["attempts"][0]["memory_body"]["prefix_recovery_permitted"]
            else:
                with pytest.raises(MinimalCallStopped): c.complete(messages(c))
        assert len(calls)==1


def test_itemwise_bad_item_is_not_permission_for_format_resampling():
    c=client(); data=response(c)
    data["choices"][0]["message"]["content"]='{"protocol_version":"2.0","reviews":[{"bad":"item"}],"changes":[]}'
    observed=reflection_body_observation(data,itemwise_review=True,reasoning_allowed=True)
    assert observed == {"channel_valid":True,"correction_issues":[]}
    assert reflection_body_observation(data,itemwise_review=False,reasoning_allowed=True)["correction_issues"]


def test_safety_review_has_own_role_schema_and_shared_turn_allowance(tmp_path, monkeypatch):
    c=client("safety_review"); calls=stub(monkeypatch,response(c))
    with closing(guard(tmp_path/"ledger",c,ProductCallGuard,min_interval_seconds=120)) as g:
        g.attach(c)
        with g.scope("turn","op1","session1"):
            assert json.loads(c.complete(messages(c)))["delete_sentence_ids"]==[1]
        state=g.snapshot(); row,=state["attempts"]
        assert row["role"]=="safety_review" and state["calls_used"]==1
        assert row["kind"]=="turn" and row["reservation"]["output_token_limit"]==1024
        assert not state["stopped"]
    wire,=calls
    assert wire["response_format"]==safety_request_format()
    assert wire["reasoning"]=={"enabled":False,"exclude":True}
    assert wire["provider"]["only"]==["alibaba"]


@pytest.mark.parametrize("mutation", ["system", "id", "verified", "unknown_key", "too_large", "wrong_schema",
                                     "missing_final", "final_too_large", "shown_too_large"])
def test_safety_request_rejects_forged_or_unbounded_input_before_send(tmp_path, monkeypatch, mutation):
    c=client("safety_review"); calls=stub(monkeypatch,response(c)); m=messages(c)
    payload=json.loads(m[1]["content"])
    if mutation=="system": m[0]["content"]+=" unrelated instruction"
    elif mutation=="id": payload["sentences"][0]["id"]=True
    elif mutation=="verified": payload["safety_evidence"]["semantic_verified"]=True
    elif mutation=="unknown_key": payload["grant_permission"]=True
    elif mutation=="too_large": payload["sentences"]=[{"id":0,"text":"x"*32769}]
    elif mutation=="missing_final": payload.pop("final_protection")
    elif mutation=="final_too_large": payload["final_protection"]="x"*4097
    elif mutation=="shown_too_large": payload["protection_already_delivered"]="x"*4097
    else: payload["version"]="online_turn_decision_v5_bound"
    m[1]["content"]=json.dumps(payload)
    with pytest.raises(ValueError): validate_safety_review_request(m)
    with closing(guard(tmp_path/"ledger",c)) as g:
        g.attach(c)
        with g.scope("turn","op1","session1"),pytest.raises(ValueError): c.complete(m)
        assert g.snapshot()["calls_used"]==0
    assert not calls


@pytest.mark.parametrize("role", ["safety_review", "reflection"])
def test_new_role_429_never_retries_or_releases_hold(tmp_path,monkeypatch,role):
    c=client(role); calls=[]
    def fail(request,**kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url,429,"fixture",{},io.BytesIO(b'{"error":{"code":429,"message":"rate limited"}}'))
    stub_model_http(monkeypatch,fail)
    with closing(guard(tmp_path/"ledger",c,ProductCallGuard)) as g:
        g.attach(c)
        with g.scope("turn" if role=="safety_review" else "review","op1","session1"),pytest.raises(MinimalCallStopped):
            c.complete(messages(c))
        assert len(calls)==1
        assert g.snapshot()["unknown_cost_requests"]==1
        assert Decimal(g.snapshot()["reserved_cost_usd"])>0


@pytest.mark.parametrize("role", ["reflection", "memory", "safety_review"])
def test_new_paid_paths_require_shared_reservation_controller_before_dispatch(monkeypatch, role):
    c=client(role); calls=stub(monkeypatch,response(c))
    with pytest.raises(ValueError, match="candidate_requires_reserved_receipt_guard"):
        c.complete(messages(c))
    assert not calls and c.network_request_count==0


def test_length_receipt_to_memory_extractor_preserves_only_screened_complete_prefix(tmp_path, monkeypatch):
    from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
    from tests.test_memory_prefix_recovery import setup, empty, preference, prefix
    persona, ctx, _, _ = setup()
    value=empty(); value["stated_preferences"]=[preference(ctx)]
    raw=prefix(value)+',"episodes":[{"summary":"unfinished'
    c=client("memory"); c.max_completion_tokens=4000
    data=response(c); data["choices"][0]["finish_reason"]="length"
    data["choices"][0]["message"]["content"]=raw
    calls=stub(monkeypatch,data)
    extractor=LLMExtractor(c,prefix_recovery=True)
    with closing(guard(tmp_path/"ledger",c,ProductCallGuard)) as g:
        g.attach(c)
        with g.scope("review","op1",ctx.session_id):
            MemoryConsolidator(extractor=extractor).consolidate(persona,ctx,persist=False)
        row,=g.snapshot()["attempts"]
        assert row["status"]=="received" and Decimal(str(row["usage"]["cost"]))==Decimal("0.00001")
        assert not g.snapshot()["unknown_cost_requests"] and len(calls)==1
    assert extractor.last_status=="partial" and c.last_finish_reason=="length"
    assert len(persona.stated_preferences)==1 and persona.stated_preferences[0].status=="proposed"
    assert extractor.last_prefix_summary["missing_fields"]==["episodes","relationships"]
    assert not persona.episodic_memory and not persona.relationship_context


@pytest.mark.parametrize("field,value", [("input_usd_per_million","1"), ("output_usd_per_million","9"),
    ("provider","google-ai-studio/flex"), ("metadata_sha256","0"*64)])
def test_reasoning_price_policy_must_cover_exact_selected_endpoint(tmp_path, field, value):
    c=client(); p=policy(c); p["models"][c.model][field]=value
    with closing(ReservedCallGuard(tmp_path/"ledger",max_requests=5,max_observed_cost_usd="1",reservation_policy=p)) as g:
        with pytest.raises(ValueError): g.attach(c)
        assert g.snapshot()["calls_used"]==0


def test_review_role_cannot_run_in_offline_scope(tmp_path, monkeypatch):
    c=client("safety_review"); calls=stub(monkeypatch,response(c))
    with closing(guard(tmp_path/"ledger",c)) as g:
        g.attach(c)
        with g.scope("review","op1","session1"), pytest.raises(MinimalCallStopped,match="operation_scope_required"):
            c.complete(messages(c))
        assert g.snapshot()["calls_used"]==0
    assert not calls


@pytest.mark.parametrize("bad", [[{"role":"system","content":"instruction"}],
    [{"role":"user","content":"text","approved":True}], [{"role":"assistant","content":None}]])
def test_review_history_keeps_only_source_text_roles(bad):
    c=client("safety_review"); m=messages(c); value=json.loads(m[1]["content"])
    value["safety_evidence"]["conversation_history"]=bad
    m[1]["content"]=json.dumps(value)
    with pytest.raises(ValueError): validate_safety_review_request(m)
