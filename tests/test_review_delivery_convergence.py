"""Real App/review wiring with predetermined labels, not semantic accuracy."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy.online_json import parse_v7_payload
from aria.engine.policy.semantic_delivery import delivery_review_pending
from aria.engine.policy.safety_structured import validate_safety_review_request
from tests.test_confirm_judge_safety_independence import inconsistent_setup, offline, INPUT
from tests.test_safety_review_v2 import answer


def review_app_wires(monkeypatch, bundle, first_output):
    from aria.engine.policy.max_release import tls_dispatch
    from tests.test_deepseek_offline_route import response
    from tests.test_installable_max_release import Response

    wires = []
    by_limit = {4096: bundle.online, 1024: bundle.safety_review, 4000: bundle.memory,
                2800: bundle.reflection, 60: bundle.confirm_judge}

    def opened(request, *, trace, **kwargs):
        trace.preparation_recovery.bind_request_headers(request)
        trace.preparation_recovery.before_preparation()
        wire = json.loads(request.data)
        wires.append(wire)
        client = by_limit[wire["max_tokens"]]
        data = response(client)
        result = first_output(client, wire)
        if result is not None:
            data["choices"][0]["message"]["content"] = json.dumps(result, ensure_ascii=False)
        return Response(data, wire["stream"])

    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    return wires


@pytest.mark.parametrize("assessment,excluded,protected", [
    ([], False, False),
    ("no_owner_signal", False, False),
    (answer("uncertain", owner_source_turns=[])["risk_assessment"], False, True),
    (answer("uncertain", owner_source_turns=[0])["risk_assessment"], True, True),
    (answer("uncertain")["risk_assessment"], True, True),
    (answer("uncertain", owner_source_turns=[1])["risk_assessment"], True, True),
    (answer("uncertain", owner_source_turns=[False])["risk_assessment"], True, True),
    (answer("fall", owner_source_turns=[])["risk_assessment"], True, True),
    (answer()["risk_assessment"], True, True),
])
def test_current_review_owner_sources_reach_both_offline_roles(tmp_path, monkeypatch,
                                                             assessment, excluded, protected):
    """Exercise actual App, wire, seal and both clients with predetermined judgments."""
    from contextlib import closing
    from aria.voice.minimal import MinimalWebApp
    from aria.engine.risk.safety_review_v2 import SOURCE_REVIEW_INSTRUCTION
    from tests.test_deepseek_offline_route import clients, guard

    bundle = clients()
    owner = "软件原话甲，我想停一下。"
    def first_output(client, wire):
        if client is bundle.online:
            return {"safety_concern": "none", "response_segments": [{"text": "陪你坐一会儿。"}]}
        if client is bundle.safety_review:
            first = json.loads(wire["messages"][1]["content"])["utterance"] == owner
            return {"risk_assessment": assessment if first else [], "keep_each_span": [True]}
    wires = review_app_wires(monkeypatch, bundle, first_output)
    with closing(guard(tmp_path)) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        try:
            sid = app.start()["state"]["session_id"]
            for i, message in enumerate([owner, "软件原话乙。", "软件原话丙。"]):
                assert app.submit(message, str(i), sid, on_safety_prelude=lambda _: True)["ok"]
            turn = app._run["turns"][0]
            assert turn["response"].endswith("陪你坐一会儿。")
            # The historical string and unnumbered records are not selections
            # under this wire's risk contract, even for a cleared body.
            unbound = assessment != []
            assert bool(turn["safety_review"].get("risk_source_unbound")) is unbound
            assert bool(turn.get("protection_prelude")) is (protected or unbound)
            review_wire = wires[1]
            assert review_wire["messages"][0]["content"] == SOURCE_REVIEW_INSTRUCTION
            schema = review_wire["response_format"]["json_schema"]["schema"]
            assert schema["properties"]["risk_assessment"]["type"] == "array"
            assert app.seal(sid)["ok"]
            job, = app.store.load(app._settled_owner).offline_sessions
            assert job.turns[0].user == ("" if excluded or unbound else owner)
            assert app.reflect("finish", wait=True)["ok"]
            assert [w["max_tokens"] for w in wires[-2:]] == [4000, 2800]
            for w in wires[-2:]:
                assert any(owner in m["content"] for m in w["messages"]) is (not (excluded or unbound))
            assert not allowance.snapshot()["stopped"]
        finally:
            app.close()


def test_pause_source_reaches_both_requests_without_expanding_an_incomplete_proposal(tmp_path, monkeypatch):
    """S1-32 t04–t06 and the incomplete first proposal, with request-local turn indices."""
    from contextlib import closing
    from aria.engine.memory.memory_contract import FIELDS
    from aria.voice.minimal import MinimalWebApp
    from tests.preference_fixtures import communication_review
    from tests.test_deepseek_offline_independent_2313 import guard
    from tests.test_deepseek_offline_route import clients, policy

    bundle = clients()
    request = "说到这些小手工，我有个小习惯：我没开口求办法时，先让我把事儿讲完，别抢着教步骤，也别一口气问一串。"
    pause = "我想停一会儿，你就陪我停一会儿；不用每段话最后都找个问题让我答。"
    limit = "不过只限我闲聊手工的时候。我要是明确问“怎么做”，你直接说办法就好，别还让我等。"
    reply = "我同意记住的是这件事：我闲聊手工、没问办法时，先听我说，少追问，想停就停；我明确问怎么做时，照常给办法。范围就到这儿。"
    do = "在本人闲聊手工、未开口求办法时，先听本人把事儿讲完，不抢着教步骤，也不一口气问一串。"
    scope = "仅限本人闲聊手工的时候；本人明确问“怎么做”时，直接说办法，不让本人等。"

    def first_output(client, wire):
        if client is bundle.safety_review:
            return answer(wire=wire)
        if client is bundle.memory:
            return {**{field: [] for field in FIELDS}, "stated_preferences": [{
                "do": do, "scope": scope, "source_turn_indices": [0, 2], "confirmation": None, "supersedes": None,
                "revision_turn_index": None, "semantic_review": communication_review(),
            }]}
        if client is bundle.confirm_judge:
            return {"action": "unclear"}
    wires = review_app_wires(monkeypatch, bundle, first_output)
    with closing(guard(tmp_path, selected_policy=policy())) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        try:
            sid = app.start()["state"]["session_id"]
            for i, text in enumerate((request, pause, limit)):
                assert app.submit(text, str(i), sid, on_safety_prelude=lambda _: True)["ok"]
            assert app.seal(sid)["ok"]
            job, = app.store.load(app._settled_owner).offline_sessions
            assert job.turns[1].user == pause
            assert app.reflect("finish", wait=True)["ok"]
            assert [w["max_tokens"] for w in wires[-2:]] == [4000, 2800]
            memory, reflection = (w["messages"][1]["content"] for w in wires[-2:])
            entries, _ = json.JSONDecoder().raw_decode(memory)
            assert entries[1]["turn_index"] == 1 and entries[1]["本人原话"] == pause
            assert f"[turn 1] 用户:{pause}\n" in reflection
            pref, = app.store.load(app._settled_owner).stated_preferences
            assert (pref.do, pref.scope, [q.text for q in pref.quotes]) == (do, scope, [request, limit])
            sid = app.start()["state"]["session_id"]
            invitation = app.preference_invitation(pref.id, sid)["preference_invitation"]
            assert app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], sid)["ok"]
            assert app.submit(reply, "answer", sid, on_safety_prelude=lambda _: True)["ok"]
            judge, = (w for w in wires if w["max_tokens"] == 60)
            assert json.loads(judge["messages"][1]["content"]) == {
                "invitation": {"do": do, "scope": scope}, "utterance": reply}
            stored, = app.store.load(app._settled_owner).stated_preferences
            assert stored.status == "proposed" and stored.confirmation is None
            assert not app.preference_action("withdraw", stored.id)["ok"]
        finally:
            app.close()


def run_candidate(tmp_path, monkeypatch, segments, review, *, invited=False, concern="none"):
    def online(value):
        value["response_segments"] = deepcopy(segments)
        value["safety_concern"] = concern
        if concern != "none":
            value["safety_evidence"] = {"quote": INPUT, "subject": "owner",
                                        "time": "current", "urgency": "immediate"}
    app, guard, sid, ctx, wires, _ = inconsistent_setup(
        tmp_path, monkeypatch, online_mutation=online, review=review)
    if not invited:
        ctx.working_memory.pop("_preference_dialogue_invitation", None)
    return app, guard, sid, ctx, wires


@pytest.mark.parametrize("concern,quote,delete,expected", [
    ("unsafe_medication", None, [], "unbound"),  # Previously allowed an unrelated clinical template.
    ("unsafe_medication", "别人说我多吃了药", [], "unbound"),
    ("unsafe_medication", None, [0], "targeted_only"),
    ("other_danger", "守在门外威胁我", [0], "retained"),
    ("other_danger", "守在门外威胁我", [0, 1, 2], "only_protection"),
])
def test_actual_sourced_review_respects_contact_constraints(tmp_path, monkeypatch,
                                                           concern, quote, delete, expected):
    from contextlib import closing
    from aria.engine.policy.max_release import tls_dispatch
    from aria.voice.minimal import MinimalWebApp
    from tests.test_deepseek_offline_route import clients, guard, response
    from tests.test_installable_max_release import Response

    owner = "有人守在门外威胁我。我出声就会被发现。请给我一条能悄悄发出去的求助消息。"
    unsafe = "接通电话后轻声告诉他你的位置。"
    draft = "请替我报警，请别过来。"
    bundle, wires = clients(), []

    def send(req, *, trace, **kwargs):
        trace.preparation_recovery.bind_request_headers(req)
        trace.preparation_recovery.before_preparation()
        wire = json.loads(req.data)
        wires.append(wire)
        reviewing = wire["max_tokens"] == 1024
        data = response(bundle.safety_review if reviewing else bundle.online)
        if reviewing:
            data_in = json.loads(wire["messages"][1]["content"])
            source = next((key for key, value in data_in["owner_sources"].items() if value and quote and quote in value),
                          "bad-source")
            risk = answer(concern, subject="owner", time="current", urgency="immediate", owner_source_id=source)
            result = {"risk_assessment": [risk["risk_assessment"]]}
            if not delete and source == "bad-source":
                # An old blanket clearance on the new wire does not cover any span.
                result.update(conflicts=False, delete_sentence_ids=[])
            else:
                result["keep_each_span"] = [s["id"] not in delete for s in data_in["sentences"]]
        else:
            result = {"response_segments": [{"text": unsafe + draft}], "safety_concern": "none"}
        data["choices"][0]["message"]["content"] = json.dumps(result, ensure_ascii=False)
        return Response(data, wire["stream"])

    monkeypatch.setattr(tls_dispatch, "open_observed", send)
    with closing(guard(tmp_path)) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        try:
            sid = app.start()["state"]["session_id"]
            assert app.submit(owner, "limited-contact", sid, on_safety_prelude=lambda _: True)["ok"]
            online, check = wires
            assert "不能出声就用无声求助" in online["messages"][0]["content"]
            payload = json.loads(check["messages"][1]["content"])
            assert payload["utterance"] == owner and payload["candidate"] == unsafe + draft
            turn = app._run["turns"][0]
            prelude = (turn.get("protection_prelude") or {}).get("protection", "")
            audit = turn["safety_review"]
            assert unsafe not in prelude + turn["response"] and "药" not in prelude
            if expected in {"retained", "targeted_only"}:
                assert turn["response"] == draft
                if expected == "targeted_only":
                    assert audit["risk_source_unbound"] and "安全可行的方式" in prelude
                else:
                    assert not prelude
                assert audit["model_body_delivered"] and audit["delete_sentence_ids"] == [0]
            else:
                assert "安全可行的方式" in prelude and "停一会儿" not in turn["response"]
                assert not audit["model_body_delivered"] and draft not in turn["response"]
                if expected == "unbound":
                    assert audit["failure_type"] == "invalid_safety_review_deletion"
            assert app.seal(sid)["ok"]
            job, = app.store.load(app._settled_owner).offline_sessions
            assert job.turns[0].user == "" and not allowance.snapshot()["stopped"]
        finally:
            app.close()


def test_sourced_review_can_delete_an_instruction_to_the_recipient_and_keep_safe_help(tmp_path, monkeypatch):
    from contextlib import closing
    from aria.voice.minimal import MinimalWebApp
    from tests.test_deepseek_offline_route import clients, guard

    owner = "有人在追我。我已经躲好，请帮我给值班室写条消息。"
    remote_help = "请从值班室联系救援。"
    approach = "你到我们中间来，再想办法引开他。"
    onsite_help = "在安全处给救援人员指路就好。"
    bundle = clients()

    def first_output(client, wire):
        if client is bundle.online:
            return {"safety_concern": "none", "response_segments": [{"text": remote_help + approach + onsite_help}]}
        assert client is bundle.safety_review
        sources = json.loads(wire["messages"][1]["content"])["owner_sources"]
        assert sources["t0.s0"] == "有人在追我。"
        risk = answer("other_danger", subject="owner", time="current", urgency="immediate", owner_source_id="t0.s0")
        return {"risk_assessment": [risk["risk_assessment"]], "keep_each_span": [True, False, False, True]}

    wires = review_app_wires(monkeypatch, bundle, first_output)
    with closing(guard(tmp_path)) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        try:
            sid = app.start()["state"]["session_id"]
            assert app.submit(owner, "recipient", sid, on_safety_prelude=lambda _: True)["ok"]
            online, review = wires
            assert "收信人在安全处代为求援" in online["messages"][0]["content"]
            assert "给收信人的行动建议" in review["messages"][0]["content"]
            parts = json.loads(review["messages"][1]["content"])["sentences"]
            assert "".join(p["text"] for p in parts if p["id"] in [1, 2]) == approach
            payload = json.loads(review["messages"][1]["content"])
            assert "final_protection" not in payload and "subject" not in payload["safety_evidence"]
            assert payload["protection_already_delivered"] is None
            assert payload["utterance"] == owner and payload["candidate"] == remote_help + approach + onsite_help
            result = app._run["turns"][0]
            assert result["response"] == remote_help + onsite_help and not result.get("protection_prelude")
        finally:
            app.close()


@pytest.mark.parametrize("case,verdicts,source_id,expect_body", [
    ("safe", [True, True], "bad-source", "请在安全处代为求援。请不要上门。"),
    ("mixed", [False, True], "bad-source", "请在安全处代为求援。"),
    ("mixed", [False, True], "t0.s0", "请在安全处代为求援。"),
    ("mixed", [True], "t0.s0", None),
    ("mixed", [True, False, True], "bad-source", None),
    ("mixed", [False, 1], "t0.s0", None),
    ("mixed_protocol", [True, True], "bad-source", None),
    ("legacy", None, "bad-source", None),
    ("legacy", None, "t0.s0", None),
])
def test_per_span_review_binds_delivery_separately_from_risk_attribution(
        tmp_path, monkeypatch, case, verdicts, source_id, expect_body):
    from contextlib import closing
    from aria.voice.minimal import MinimalWebApp
    from tests.test_deepseek_offline_route import clients, guard

    owner = "有人守在门口威胁我。我一出声就会被发现。替我写一条求助短信。"
    candidate = ("请在安全处代为求援。请不要上门。" if case == "safe"
                 else "请马上来我门口敲门。请在安全处代为求援。")
    bundle = clients()

    def first_output(client, wire):
        if client is bundle.online:
            return {"safety_concern": "none", "response_segments": [{"text": candidate}]}
        assert client is bundle.safety_review
        data = json.loads(wire["messages"][1]["content"])
        assert data["candidate"] == candidate
        assert [s["id"] for s in data["sentences"]] == [0, 1]
        schema = wire["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == set(schema["required"]) == {"risk_assessment", "keep_each_span"}
        risk = answer("other_danger", subject="owner", time="current", urgency="immediate", owner_source_id=source_id)
        result = {"risk_assessment": [risk["risk_assessment"]]}
        if verdicts is not None:
            result["keep_each_span"] = verdicts
        if case in ("legacy", "mixed_protocol"):
            result.update(conflicts=False, delete_sentence_ids=[])
        return result

    wires = review_app_wires(monkeypatch, bundle, first_output)
    with closing(guard(tmp_path)) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        try:
            sid = app.start()["state"]["session_id"]
            assert app.submit(owner, "review", sid, on_safety_prelude=lambda _: True)["ok"]
            turn = app._run["turns"][0]
            audit = turn["safety_review"]
            assert [w["max_tokens"] for w in wires] == [4096, 1024]
            assert bool(audit.get("span_review_complete")) is (expect_body is not None)
            assert bool(audit.get("model_body_delivered")) is (expect_body is not None)
            assert (audit.get("risk_source_unbound") is True) is (source_id == "bad-source")
            if expect_body:
                assert turn["response"] == expect_body
                if source_id == "bad-source":
                    assert "安全可行的方式" in turn["protection_prelude"]["protection"]
            else:
                assert candidate not in turn["response"] and "敲门" not in turn["response"]
            assert not allowance.snapshot()["stopped"]
        finally:
            app.close()


@pytest.mark.parametrize("kind", ["preference_state", "third_party_fact", "conversation"])
def test_bad_coarse_label_cannot_delete_reviewed_normal_text(tmp_path, monkeypatch, kind):
    text = "我会先听你讲完。想停时就停一会儿。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": kind, **({"source_quote": "错误的辅助引文"} if kind == "third_party_fact" else {})}], answer())
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == text
        assert [wire["max_tokens"] for wire in wires] == [4096, 1024]
        sent = json.loads(wires[-1]["messages"][1]["content"])
        assert sent["candidate"] == text
        assert sent["safety_evidence"]["delivery_facts"]["owner_sources"][0] == INPUT
        assert ctx.working_memory["safety_review"]["model_body_delivered"] is True
        assert not ctx.working_memory.get("_sensitive_memory_excluded_turns")
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("kind,target", [("assistant_action", "daughter"),
    ("conversation", ""), ("assistant_action", "wrong")])
def test_actual_receipt_is_available_independent_of_model_metadata(kind, target):
    from tests.test_semantic_delivery_v6 import context, segment
    from aria.engine.policy.semantic_delivery import bind_current_action_receipt, delivery_review_facts
    persona, ctx = context()
    receipt = {"ok": True, "tool": "notify_family", "args": {"contact": "daughter"},
               "data": {"status": "delivered"}}
    ctx.working_memory["last_tool_result"] = receipt
    bind_current_action_receipt(ctx, INPUT, receipt)
    ctx.working_memory["online_turn_decision"] = {"response_segments": [
        segment("调用结果。", kind, action="notification" if kind == "assistant_action" else "none", target=target)]}
    facts = delivery_review_facts(ctx, INPUT, persona)
    assert facts["current_action_results"] == [{"action": "notification", "target": "daughter",
        "status": "发送服务返回了成功回执；这不证明对方已读、回复或采取行动。"}]
    receipt["data"]["status"] = "pending"
    assert delivery_review_facts(ctx, INPUT, persona)["current_action_results"] == []
    bind_current_action_receipt(ctx, INPUT, receipt)
    assert "尚无完成或送达回执" in delivery_review_facts(ctx, INPUT, persona)["current_action_results"][0]["status"]
    ctx.session_id = "different-session"
    assert delivery_review_facts(ctx, INPUT, persona)["current_action_results"] == []


@pytest.mark.parametrize("kind", ["preference_state", "assistant_action", "conversation"])
def test_only_reviewed_false_clause_is_removed_from_mixed_segment(tmp_path, monkeypatch, kind):
    text = "我已完成这件事。你可以继续说，我在听。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": kind}], answer(delete=[0]))
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == "你可以继续说，我在听。"
        assert json.loads(wires[-1]["messages"][1]["content"])["candidate"] == text
        assert ctx.working_memory["safety_review"]["delete_sentence_ids"] == [0]
        assert all(row.status == "proposed" for row in app.store.load(app._settled_owner).stated_preferences)
        assert len(wires) == 2
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("review", ["not-json", answer(delete=[127])])
def test_deferred_claim_never_escapes_failed_review(tmp_path, monkeypatch, review):
    text = "待审核的状态与交流。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": "preference_state"}], review)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert text not in app._run["turns"][0]["response"]
        assert ctx.working_memory["safety_review"]["model_body_delivered"] is False
        assert len(wires) == 2
    finally:
        app.close(); guard.close()


def test_confirmation_code_receipt_is_not_in_review_deletion_units(tmp_path, monkeypatch):
    text = "我已经记下了。你可以继续说。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": "preference_state"}], answer(delete=[0]), invited=True)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == "你可以继续说。\n好，这条记下了。"
        sent = json.loads(wires[-1]["messages"][1]["content"])
        assert sent["candidate"] == text
        facts = sent["safety_evidence"]["delivery_facts"]
        assert facts["code_status"] == "好，这条记下了。"
        assert facts["preference_records"][0]["status"] == "confirmed"
        assert facts["preference_records"][0]["confirmation"]["text"] == INPUT
        assert [wire["max_tokens"] for wire in wires] == [4096, 60, 1024]
        assert validate_safety_review_request(wires[-1]["messages"]) == 2
    finally:
        app.close(); guard.close()


def test_runtime_review_routing_does_not_survive_a_serialized_working_memory(tmp_path, monkeypatch):
    from aria.engine.context import EngineContext
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": "正常交流。"}], answer())
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        restored = EngineContext(user_id=ctx.user_id, session_id=ctx.session_id, scene_id=ctx.scene_id)
        restored.working_memory = deepcopy(ctx.working_memory)
        restored.working_memory["_semantic_delivery_review_v2"] = True
        assert not delivery_review_pending(restored, INPUT)
        assert not delivery_review_pending(ctx, INPUT)  # completed turn has advanced
    finally:
        app.close(); guard.close()


def test_optional_none_contradiction_is_dropped_not_reinterpreted_as_input_risk():
    raw = {"response_segments": [{"text": "原话应保留。"}], "safety_concern": "none",
           "safety_evidence": {"quote": "", "subject": "other", "time": "uncertain", "urgency": "not_applicable"}}
    original = deepcopy(raw)
    parsed, dropped = parse_v7_payload(json.dumps(raw))
    assert raw == original
    assert parsed["response"] == "原话应保留。" and parsed["safety_concern"] == "none"
    assert "safety_evidence" not in parsed and dropped == ["safety_evidence"]


def test_risk_route_review_also_preserves_safe_adjacent_clause(tmp_path, monkeypatch):
    text = "未核实的完成声明。保留的安全交流。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": "preference_state"}], answer(delete=[0]), concern="fall")
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert app._run["turns"][0]["response"] == "保留的安全交流。"
        assert len(wires) == 2
        assert ctx.working_memory["safety_review"]["model_body_delivered"] is True
        assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("mutation", ["state", "session", "ui_result"])
def test_actual_state_or_identity_change_during_review_blocks_body(tmp_path, monkeypatch, mutation):
    text = "必须绑定当前事实的模型正文。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": "preference_state"}], answer())
    engine, _ = app._sessions[app._settled_owner]
    if mutation == "ui_result":
        target = engine.persona.stated_preferences[0].id
        assert app.preference_action("decline", target)["ok"]
        engine, _ = app._sessions[app._settled_owner]
        assert engine.persona._preference_ui_result is not None
    complete = engine.safety_review_llm.complete
    def changed(messages, **kwargs):
        result = complete(messages, **kwargs)
        if mutation == "state":
            engine.persona.stated_preferences[0].scope += " changed"
        elif mutation == "ui_result":
            engine.persona._preference_ui_result = None
        else:
            ctx.session_id = "another-session"
        return result
    monkeypatch.setattr(engine.safety_review_llm, "complete", changed)
    try:
        app.submit(INPUT, "turn1", sid)
        assert all(text not in row.get("response", "") for row in app._run["turns"])
        assert ctx.working_memory["safety_review"]["model_body_delivered"] is False
        assert len(wires) == 2
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("kind", ["conversation", "preference_state"])
@pytest.mark.parametrize("delete", [(), (1,)])
def test_native_withdrawal_is_in_same_review_without_exempting_body(tmp_path, monkeypatch, kind, delete):
    from aria.engine.memory.preference_status import current_preference_states
    text = "现有状态说明。待复核操作句。继续交流。"
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text, "kind": kind, "state_ref": "model-cannot-authorize"}], answer(delete=delete))
    try:
        target = app._profile().stated_preferences[0].id
        assert app.preference_action("confirm", target)["ok"]
        assert app.preference_action("withdraw", target)["ok"]
        state, = current_preference_states(app._profile())
        assert app.submit(INPUT, "turn1", sid)["ok"]
        sent = json.loads(wires[-1]["messages"][1]["content"])
        facts = sent["safety_evidence"]["delivery_facts"]
        record, = facts["preference_records"]
        assert record["status"] == state["status"] == "withdrawn"
        assert record["applicable"] is False
        assert record["last_ui_result"] == {"action": "withdraw", "persisted": True}
        assert not {"state_ref", "target_id", "source"}.intersection(record)
        assert facts["current_action_results"] == [] and facts["code_status"] is None
        assert validate_safety_review_request(wires[-1]["messages"]) == 3
        assert [wire["max_tokens"] for wire in wires] == [4096, 1024]
        assert app._run["turns"][0]["response"] == (
            "现有状态说明。继续交流。" if delete else text)
        assert ctx.working_memory["safety_review"]["grants_permission"] is False
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "withdrawn"
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("case", ["normal", "confirm", "unclear", "existing_exclusion",
                                 "review_failed", "source_changed", "reviewed_risk"])
def test_deleted_body_does_not_erase_clear_owner_input_from_real_seal(tmp_path, monkeypatch, case):
    from aria.engine.memory.offline_queue import OfflineSessionQueue
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY

    text = "需要删除的整段模型正文。"
    utterance = INPUT + ("我改成只在整理相册时适用，其他话题不这样。" if case == "unclear" else "")
    result = ("not-json" if case == "review_failed" else
              answer("fall", delete=[0], subject="owner", time="current", urgency="immediate")
              if case == "reviewed_risk" else answer(delete=[0]))
    def online(value):
        value["response_segments"] = [{"text": text}]
    app, guard, sid, ctx, wires, _ = inconsistent_setup(
        tmp_path, monkeypatch, outcome="unclear" if case == "unclear" else "confirm",
        online_mutation=online, review=result)
    if case not in {"confirm", "unclear"}:
        ctx.working_memory.pop("_preference_dialogue_invitation", None)
    if case == "existing_exclusion":
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    engine, _ = app._sessions[app._settled_owner]
    if case == "source_changed":
        complete = engine.safety_review_llm.complete
        def changed(messages, **kwargs):
            reviewed = complete(messages, **kwargs)
            engine.persona.stated_preferences[0].do = "新的待确认交流说明"
            return reviewed
        monkeypatch.setattr(engine.safety_review_llm, "complete", changed)
    excluded = case not in {"normal", "confirm", "unclear", "review_failed"}
    try:
        assert app.submit(utterance, "turn1", sid)["ok"]
        delivered = app._run["turns"][0]["response"]
        assert text not in delivered and ctx.turns[0].user == utterance
        if not excluded or case == "existing_exclusion":
            assert DETERMINISTIC_MODEL_FAILURE_REPLY in delivered
        if case == "confirm":
            assert delivered.endswith("好，这条记下了。")
            assert app.store.load(app._settled_owner).stated_preferences[0].status == "confirmed"
        elif case == "unclear":
            assert delivered.endswith("这条我先不记，你想记的时候再说一声。")
            assert app.store.load(app._settled_owner).stated_preferences[0].status == "proposed"
        review = ctx.working_memory["safety_review"]
        assert review["model_body_delivered"] is False
        assert review["status"] == (case if case in {"review_failed", "source_changed"} else "reviewed")
        if review["status"] == "reviewed":
            assert review["risk_assessment"]["concern"] == ("fall" if case == "reviewed_risk" else "none")
        assert (0 in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])) is excluded
        assert ctx.working_memory["intervention_delivery"]["status"] == "not_delivered"
        assert "0" not in ctx.working_memory.get("_person_response_deliveries", {})
        assert len(wires) == (3 if case in {"confirm", "unclear"} else 2)
        assert json.loads(wires[-1]["messages"][1]["content"])["candidate"] == text
        assert app.seal(sid)["ok"]
        saved = app.store.load(app._settled_owner)
        job, = saved.offline_sessions
        assert job.excluded_turn_indices == ([0] if excluded else [])
        assert job.turns[0].user == ("" if excluded else utterance)
        assert job.turns[0].agent == ("" if excluded else delivered)
        assert job.turns[0].person_response_sha256 is None
        recovered = OfflineSessionQueue._context(saved, job)
        assert recovered.turns[0].user == job.turns[0].user
        assert recovered.turns[0].agent == job.turns[0].agent
        assert recovered.working_memory["_person_response_deliveries"] == {}
        assert len(wires) == (3 if case in {"confirm", "unclear"} else 2)
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("delete", [[], [0]])
def test_review_none_other_first_output_preserves_original_input_through_reload(tmp_path, monkeypatch, delete):
    from aria.engine.memory.offline_queue import OfflineSessionQueue
    text = "需要复核的第一句。继续保留这句交流。"
    # Exact field combination observed in the original failed call. The public
    # response stays unchanged; only unused template selectors are discarded.
    raw = {"conflicts": bool(delete), "delete_sentence_ids": delete,
           "risk_assessment": {"concern": "none", "subject": "other",
                               "time": "uncertain", "urgency": "uncertain"}}
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": text}], raw)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        delivered = app._run["turns"][0]["response"]
        assert delivered == ("继续保留这句交流。" if delete else text)
        review = ctx.working_memory["safety_review"]
        assert review["status"] == "reviewed" and review["model_body_delivered"] is True
        assert review["dropped_fields"] == ["risk_assessment.subject"]
        assert not ctx.working_memory.get("_sensitive_memory_excluded_turns")
        assert app.seal(sid)["ok"]
        saved = app.store.load(app._settled_owner)
        job, = saved.offline_sessions
        assert job.excluded_turn_indices == []
        assert job.turns[0].user == INPUT and job.turns[0].agent == delivered
        recovered = OfflineSessionQueue._context(saved, job)
        assert recovered.turns[0].user == INPUT and recovered.turns[0].agent == delivered
        assert len(wires) == 2
    finally:
        app.close(); guard.close()


def test_parse_failure_with_changed_source_cannot_preserve_wrong_learning_input(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch,
        [{"text": "不能归属给变化后的状态。"}], "not-json")
    engine, _ = app._sessions[app._settled_owner]
    complete = engine.safety_review_llm.complete
    def changed(messages, **kwargs):
        result = complete(messages, **kwargs)
        engine.persona.stated_preferences[0].scope += " changed"
        return result
    monkeypatch.setattr(engine.safety_review_llm, "complete", changed)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        assert ctx.working_memory["safety_review"]["status"] == "source_changed"
        assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]
        assert app.seal(sid)["ok"]
        job, = app.store.load(app._settled_owner).offline_sessions
        assert job.turns[0].user == "" and job.turns[0].agent == ""
        assert len(wires) == 2
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("concern", ["none", "uncertain", "fall", "medical_emergency",
                                     "self_harm", "unsafe_medication"])
def test_failed_deletion_keeps_independent_risk_judgment_in_real_app(tmp_path, monkeypatch, concern):
    text = "未经有效删句审核的正文。"
    result = answer(concern, delete=[127], subject="owner", time="current", urgency="immediate")
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": text}], result)
    try:
        assert app.submit(INPUT, "turn1", sid)["ok"]
        delivered = app._run["turns"][0]["response"]
        assert text not in delivered
        audit = ctx.working_memory["safety_review"]
        assert audit["status"] == "review_failed" and audit["model_body_delivered"] is False
        assert audit["risk_assessment"]["concern"] == concern
        assert audit["failure_type"] == "invalid_safety_review_deletion"
        if concern != "none":
            from aria.engine.risk.safety_delivery import render_review_protection, render_unverified_safety_v2
            protection, _ = (render_unverified_safety_v2() if concern == "uncertain" else
                             render_review_protection(audit["risk_assessment"], reviewed_fields=True))
            assert protection in delivered
        assert (0 in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])) is (concern != "none")
        assert app.seal(sid)["ok"]
        job, = app.store.load(app._settled_owner).offline_sessions
        assert job.turns[0].user == (INPUT if concern == "none" else "")
        assert job.turns[0].person_response_sha256 is None
        assert len(wires) == 2 and not guard.snapshot()["stopped"]
    finally:
        app.close(); guard.close()
