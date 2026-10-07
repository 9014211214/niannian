"""Neutral deterministic fixtures: software mechanics, never model accuracy."""
import hashlib
import json
from pathlib import Path
import socket
import urllib.request

import pytest

from aria.engine.context import EngineContext, Turn
from aria.engine.policy import PolicyEngine
from aria.engine.policy.delivery_provenance import delivery_protection, record_reply_stage
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.engine.policy.reply_contract import contract_prompt, reply_quality_contract
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate
from aria.runtime import MinimalModelClients
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.reasoning import ElderStateReasoner
from aria.voice.minimal import MinimalWebApp


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("software fixture attempted network")
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


def context(version="v3", mode="ask", physical=False, risk=False):
    ctx = EngineContext(user_id="fixture", scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_" + version
    configure(ctx, mode, physical)
    if risk:
        ctx.working_memory["risk_flag"] = {"kind": "fixture", "tier": 1}
    return ctx


def configure(ctx, mode="ask", physical=False):
    ctx.reasoning.candidates = [InterventionCandidate(
        id=i, label="软件候选", goal="回应当前内容", guidance="按当前要求回应", score=.5,
    ) for i in ("quiet_companionship", "physical_clarification")]
    ctx.reasoning.selected = ctx.reasoning.candidates[int(physical)]
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode=mode, question="软件预设问题？")
    ctx.working_memory["personality_decision"] = {"current_physical_evidence_qualified": physical, "evidence": []}


def payload(response="软件夹具交付内容。", safety="none"):
    return json.dumps({"response": response, "state_signals": [],
                       "current_affect": {"label": "unknown", "confidence": 0.,
                                          "divergence": {"possible": False, "confidence": 0., "pattern": ""}},
                       "selected_intervention_id": "quiet_companionship",
                       "communication_plan": {"pace": "brief_exchange", "support_order": "交付当前内容", "boundary": "不执行动作"},
                       "uncertainty": .5, "safety_concern": safety}, ensure_ascii=False)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_historical_exact_question_unchanged(version):
    contract = reply_quality_contract(context(version))
    assert contract["version"] == "1.6.0"
    assert contract["question_requirement"] == "exact"
    assert "本轮唯一允许的问句:软件预设问题？" in contract_prompt(contract)


@pytest.mark.parametrize("physical", [False, True])
def test_v3_provisional_question_and_route_do_not_override_valid_output(physical):
    ctx = context(physical=physical)
    contract = reply_quality_contract(ctx)
    assert contract["observation_advisory"] and contract["question_requirement"] == "optional"
    assert "软件预设问题" not in contract_prompt(contract)
    assert apply_single_pass_decision(payload(), ctx, "中性输入。") == "软件夹具交付内容。"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["selection_lock_reason"] is None
    assert not trace["delivery_guard"]["activated"]


@pytest.mark.parametrize("mode,risk,requirement", [("wait", False, "none"), ("ask", True, "exact")])
def test_wait_and_risk_are_not_relaxed(mode, risk, requirement):
    ctx = context(mode=mode, risk=risk)
    contract = reply_quality_contract(ctx)
    assert not contract["observation_advisory"]
    assert contract["question_requirement"] == requirement
    assert apply_single_pass_decision(payload(), ctx, "中性输入。") != "软件夹具交付内容。"


class Client:
    source_id = "software_fixture"
    model = "local-fixture"
    online_decision_version = "online_turn_decision_v3"
    online_return_format = "json_object"

    def __init__(self, response=None):
        self.response = response or payload()
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.response


def test_actual_policy_selects_v3_before_base_prompt():
    ctx, client = context("v1"), Client()
    result = PolicyEngine(fast_llm=client).respond(ctx, get_scene("companion_elder"), "中性输入。")
    assert result == "软件夹具交付内容。" and len(client.calls) == 1
    prompt = client.calls[0][0]["content"]
    assert "不要求使用预设问句" in prompt
    assert "本轮唯一允许的问句" not in prompt
    assert "安全分流 safety_concern 必填" in prompt
    assert "代写不等于发送" in prompt


@pytest.mark.parametrize("first_changed,last_changed", [(True, False), (False, True), (True, True), (False, False)])
def test_stage_merge_does_not_erase_earlier_replacement(first_changed, last_changed):
    ctx = context()
    initial = "software a"
    middle = "software b" if first_changed else initial
    final = "software c" if last_changed else middle
    record_reply_stage(ctx, "online_guard", initial, middle)
    record_reply_stage(ctx, "delivery_guard", middle, final)
    ctx.turns.append(Turn("fixture", final))
    got = delivery_protection(ctx)
    assert got["response_changed"] == (first_changed or last_changed)
    assert got["model_response_sha256"] == hashlib.sha256(initial.encode()).hexdigest()
    assert got["delivered_response_sha256"] == hashlib.sha256(final.encode()).hexdigest()
    assert len(got["stages"]) == 2
    ctx.turns.append(Turn("later", "later"))
    assert delivery_protection(ctx)["status"] == "not_observed"


class Reasoner(ElderStateReasoner):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def update(self, ctx, *args):
        configure(ctx, self.mode)

    def observe_response(self, *args):
        pass


def app_fixture(tmp_path, mode="ask", response=None):
    online = Client(response)
    offline = Client('{"episodes":[],"relationships":[],"stated_preferences":[],"cognitive_privacy":[]}')
    app = MinimalWebApp(tmp_path, clients=MinimalModelClients(online, offline, offline, True))
    sid = app.start()["state"]["session_id"]
    engine, ctx = app._sessions["fictional-chen"]
    engine._state_reasoner = Reasoner(mode)
    return app, sid, engine, ctx, online


@pytest.mark.parametrize("mode,changed", [("ask", False), ("wait", True)])
def test_actual_web_source_survives_later_unchanged_guard(tmp_path, mode, changed):
    app, sid, engine, ctx, online = app_fixture(tmp_path, mode)
    try:
        row = app.submit("中性输入。", "fixture-turn", sid)["state"]["turns"][-1]
        assert len(online.calls) == 1
        assert row["source"] == ("rule_protection" if changed else "live")
        assert row["delivery_protection"]["response_changed"] is changed
        assert [s["stage"] for s in row["delivery_protection"]["stages"][:2]] == [
            "response_normalization", "online_guard"]
        assert row["delivery_protection"]["stages"][-1]["response_changed"] is False
    finally:
        app.close()


def test_scene_and_final_receipt_guards_remain_visible(tmp_path, monkeypatch):
    app, sid, engine, ctx, online = app_fixture(tmp_path, "observe", payload("我已经通知女儿了。"))
    try:
        monkeypatch.setattr(engine.scene, "guard_response", lambda c, r: "软件替换内容。")
        row = app.submit("中性输入。", "fixture-guards", sid)["state"]["turns"][-1]
        protection = row["delivery_protection"]
        assert row["source"] == "rule_protection"
        assert {s["stage"] for s in protection["stages"] if s["response_changed"]} == {"scene_guard", "delivery_guard"}
        assert "unverified_external_action_completion" in protection["violations"]
        assert row["response"] != "我已经通知女儿了。"
        assert not app.state()["profile"]["hypotheses"]
    finally:
        app.close()


def test_same_call_safety_still_takes_over_before_actions(tmp_path):
    app, sid, engine, ctx, online = app_fixture(tmp_path, "ask", payload("软件文本不会交付。", "self_harm"))
    try:
        row = app.submit("软件安全分流输入。", "fixture-safety", sid)["state"]["turns"][-1]
        assert len(online.calls) == 1
        assert row["source"] == "rule_safety"
        assert row["response"] != "软件文本不会交付。"
    finally:
        app.close()


def test_current_product_is_imported():
    import aria.engine.policy.reply_contract as contract
    expected = Path(__file__).resolve().parents[1] / "aria/engine/policy/reply_contract.py"
    assert Path(contract.__file__).resolve() == expected
