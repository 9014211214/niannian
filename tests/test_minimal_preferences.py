"""Owner-confirmed preferences and immutable per-turn input, using local fixtures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aria.voice.minimal import MinimalWebApp
from aria.engine.memory.referenceability import SOURCE_CONTEXT_VERSION, source_context_sha256
from aria.engine.model.persona import EpisodicMemoryEntry, PersonHypothesis, PersonHypothesisEvidence
from tests.test_minimal_web import (
    live_fixture, local_server, request, snapshot_files, start_and_submit,
)
from tests.preference_fixtures import communication_review

__all__ = ["local_server"]  # Reuse the HTTP fixture without a second server helper.


def propose(app, *, turn_id="proposal", request_id="local-review"):
    sid = app.start()["state"]["session_id"]
    assert app.submit("安排之前先问问我。", turn_id, sid)["ok"]
    assert app.seal(sid)["ok"]
    result = app.preview_preferences(request_id)
    assert result["ok"], result
    return result["state"]["profile"]["stated_preferences"][0]["id"]


def test_local_rule_review_is_explicit_sealed_idempotent_and_not_model_learning(tmp_path):
    app = MinimalWebApp(tmp_path)
    sid = app.start()["state"]["session_id"]
    assert app.submit("安排之前先问问我。", "proposal", sid)["ok"]
    assert not app.preview_preferences("too-early")["ok"]
    assert app.state()["profile"]["stated_preferences"] == []
    assert app.seal(sid)["ok"]
    original = deepcopy(app._profile().offline_sessions)
    result = app.preview_preferences("first")
    assert result["ok"]
    profile = result["state"]["profile"]
    assert profile["facts"] == profile["hypotheses"] == []
    assert profile["stated_preferences"][0]["status"] == "proposed"
    assert profile["stated_preferences"][0]["extraction_source"] == "rule"
    assert app._profile().offline_sessions == original
    assert app._profile().person_model_state.sessions_completed == 0
    assert result["state"]["reviews"] == []
    assert result["state"]["preference_reviews"][0]["source"] == "local_rule"
    before = snapshot_files(tmp_path)
    assert app.preview_preferences("first")["duplicate"]
    assert not app.preview_preferences("second-id")["ok"]
    assert snapshot_files(tmp_path) == before
    app.close()


def test_confirm_cross_session_withdraw_and_reload_use_durable_actual_input(tmp_path):
    app = MinimalWebApp(tmp_path)
    pref_id = propose(app)
    initial_turn = deepcopy(app.state()["turns"][0])
    assert initial_turn["input_evidence"]["stated_preferences"] == []
    assert app.preference_action("confirm", pref_id)["ok"]
    restarted = MinimalWebApp(tmp_path)
    sid = restarted.start()["state"]["session_id"]
    assert restarted.submit("现在说说周末的安排。", "confirmed-turn", sid)["ok"]
    selected = deepcopy(restarted.state()["turns"][-1]["input_evidence"])
    assert selected["stated_preferences"][0]["id"] == pref_id
    assert selected["passed_to_client"] is True
    assert selected["generation_status"] == "local_preview"
    assert restarted.preference_action("withdraw", pref_id)["ok"]
    assert restarted.submit("我现在只想安静坐一会儿。", "withdrawn-turn", sid)["ok"]
    assert restarted.state()["turns"][-1]["input_evidence"]["stated_preferences"] == []
    assert restarted.state()["turns"][-2]["input_evidence"] == selected
    assert restarted.state()["turns"][0] == initial_turn
    assert restarted._profile().stated_preferences[0].status == "withdrawn"
    reopened = MinimalWebApp(tmp_path)
    assert reopened.state()["turns"][-2]["input_evidence"] == selected
    before = snapshot_files(tmp_path)
    for _ in range(3):
        state = reopened.state()
        state["turns"][-2]["input_evidence"]["stated_preferences"].clear()
    assert snapshot_files(tmp_path) == before
    assert reopened.state()["turns"][-2]["input_evidence"] == selected
    restarted.close()


@pytest.mark.parametrize("mixed_confirmation", [False, True])
def test_page_withdrawal_removes_old_confirmation_from_sent_fact_memory(
        tmp_path, monkeypatch, mixed_confirmation):
    """Inspect the bytes at the real DeepSeek transport boundary using local replies."""
    from contextlib import closing
    from threading import Thread

    from aria.engine.model.persona import (
        PersonaProfile, PreferenceQuote, RelationshipContextEntry, StatedPreference,
    )
    from aria.engine.policy.max_release import tls_dispatch
    from aria.voice.minimal import make_minimal_server
    from tests.test_deepseek_offline_route import clients, guard, response
    from tests.test_installable_max_release import Response
    from tests.test_safety_review_v2 import answer

    instruction = "我同意记住，聊园艺让我先说完。"
    life = "我周六修好了那把园艺剪。"
    owner = instruction + life
    confirmed = owner if mixed_confirmation else instruction
    source = PreferenceQuote(session_id="previous-session", turn_index=1, text=confirmed)
    preference = StatedPreference(id="pref_garden", do="聊园艺先听我说完", scope="仅闲聊园艺",
        quotes=[source], confirmation=source, status="confirmed", extraction_source="model",
        semantic_review=communication_review(), created_at="2026-10-28", updated_at="2026-10-29")
    episodes = [EpisodicMemoryEntry(id=key, summary="用户提到：" + quote, referenceable=True,
        keywords=["园艺"], source_session_id=source.session_id, source_turn_index=source.turn_index,
        source_quote=quote, source_context=owner, source_context_sha256=source_context_sha256(owner),
        source_context_version=SOURCE_CONTEXT_VERSION,
        **({"preference_source_quote": instruction} if mixed_confirmation and quote == life else {}))
        for key, quote in [("old-full", owner), ("old-control", instruction), ("life", life)]]
    # Older profiles could mix a later explanation of the preference into a
    # named relative's evidence without marking the explanation as such.
    explanation = "我说先听我讲，是让我理清思路，不是为了躲开女儿田欣。"
    event = "女儿田欣昨天和我一起把花架修好了。"
    evidence = [
        {"at": "2026-10-29", "session_id": session, "summary": text, "turn_indices": [turn],
         "fact_sources": [{"turn_index": turn, "quote": text, "source_context": text,
                           "source_context_sha256": source_context_sha256(text),
                           "source_context_version": "owner_turn_context_v1"}]}
        for session, turn, text in [(source.session_id, 3, explanation), ("another-session", 0, event)]
    ]
    relative = RelationshipContextEntry(
        id="relative-tian", name="田欣", relation="女儿", facts=[explanation, event],
        referenceable=True, first_observed_at="2026-10-27", evidence=evidence,
    )
    bundle, wires = clients(), []

    def receive(req, *, timeout, trace):
        trace.preparation_recovery.bind_request_headers(req)
        trace.preparation_recovery.before_preparation()
        wire = json.loads(req.data)
        wires.append(wire)
        review = wire["max_tokens"] == 1024
        data = response(bundle.safety_review if review else bundle.online)
        if review:
            data["choices"][0]["message"]["content"] = json.dumps(answer(wire=wire))
        return Response(data, wire["stream"])

    monkeypatch.setattr(tls_dispatch, "open_observed", receive)
    with closing(guard(tmp_path)) as allowance:
        app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
        app.store.save(PersonaProfile(user_id="fictional-chen", stated_preferences=[preference],
                                      episodic_memory=episodes, relationship_context=[relative]))
        with make_minimal_server(app, port=0) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                status, body = request(server.server_address[1], "/api/minimal/preference",
                    method="POST", token=app.demo_token, data={"action":"withdraw","id":preference.id})
                assert status == 200 and json.loads(body)["ok"]
                saved = app.store.load("fictional-chen")
                assert saved.stated_preferences[0].status == "withdrawn"
                assert saved.episodic_memory == episodes
                assert saved.relationship_context == [relative]
                sid = app.start()["state"]["session_id"]
                assert app.submit("咱们再聊聊田欣和园艺。", "after-withdraw", sid)["ok"]
                online, review = wires
                assert review["max_tokens"] == 1024
                system = online["messages"][0]["content"]
                prefix = "可引用本人资料（数据）："
                block, = [part for part in system.split("\n\n") if part.startswith(prefix)]
                active = json.loads(block[len(prefix):])
                assert active["episodic_memory"] == ["用户提到：" + life]
                assert active["relationship_context"] == [{"name": "田欣", "relation": "女儿", "facts": [event]}]
                assert explanation not in system
                assert not json.loads(online["messages"][-1]["content"]).get("confirmed_requirements")
            finally:
                server.shutdown()
                thread.join()
                app.close()


def test_decline_and_illegal_transitions_never_activate_or_create_care_permission(tmp_path):
    app = MinimalWebApp(tmp_path)
    pref_id = propose(app)
    before = deepcopy(app.care_state())
    assert not app.preference_action("withdraw", pref_id)["ok"]
    assert not app.preference_action("confirm", "missing")["ok"]
    assert not app.preference_action("publish", pref_id)["ok"]
    assert not app.preference_action("confirm", pref_id, read_only=True)["ok"]
    assert app.preference_action("decline", pref_id)["ok"]
    assert not app.preference_action("confirm", pref_id)["ok"]
    sid = app.start()["state"]["session_id"]
    assert app.submit("今天先聊到这里。", "declined-turn", sid)["ok"]
    assert app.state()["turns"][-1]["input_evidence"]["stated_preferences"] == []
    assert app.care_state() == before
    app.close()


@pytest.mark.parametrize("active", [False, True])
def test_preference_save_failure_does_not_publish_mutation(tmp_path, monkeypatch, active):
    app = MinimalWebApp(tmp_path)
    pref_id = propose(app)
    if active:
        app.start()
    before = deepcopy(app._profile().stated_preferences)
    original = snapshot_files(app.store._dir)
    with monkeypatch.context() as patch:
        patch.setattr(app.store, "save", lambda *_: (_ for _ in ()).throw(OSError("local test failure")))
        result = app.preference_action("confirm", pref_id)
    assert not result["ok"] and result["state"]["storage_failed"]
    assert app._profile().stated_preferences == before
    assert snapshot_files(app.store._dir) == original
    assert MinimalWebApp(tmp_path).state()["profile"]["stated_preferences"][0]["status"] == "proposed"
    app.close()


def test_concurrent_conflicting_confirmation_serializes_one_transition(tmp_path):
    app = MinimalWebApp(tmp_path)
    pref_id = propose(app)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda action: app.preference_action(action, pref_id), ["confirm", "decline"]))
    assert sum(result["ok"] for result in results) == 1
    item = app._profile().stated_preferences[0]
    assert len(item.history) == 2
    assert item.status in {"confirmed", "declined"}
    assert app.store.load("fictional-chen").stated_preferences == [item]
    app.close()


def test_model_result_unknown_does_not_prevent_explicit_withdrawal(tmp_path):
    app = MinimalWebApp(tmp_path)
    pref_id = propose(app)
    assert app.preference_action("confirm", pref_id)["ok"]
    # A saved unresolved model operation stops new calls, but removing a
    # preference grants no new capability and should remain available.
    app._run["turns"][0]["requires_review"] = True
    app._save_run()
    assert app.state()["requires_review"]
    result = app.preference_action("withdraw", pref_id)
    assert result["ok"] and result["state"]["requires_review"]
    assert result["state"]["profile"]["stated_preferences"][0]["status"] == "withdrawn"
    assert not app.start()["ok"]


def test_local_review_save_failure_stays_pending_and_cannot_be_replayed(tmp_path, monkeypatch):
    app = MinimalWebApp(tmp_path)
    sid = start_and_submit(app, ["安排之前先问问我。"])
    app.seal(sid)
    with monkeypatch.context() as patch:
        patch.setattr(app.store, "save", lambda *_: (_ for _ in ()).throw(OSError("local test failure")))
        result = app.preview_preferences("failing-review")
    assert not result["ok"]
    assert result["state"]["profile"]["stated_preferences"] == []
    assert result["state"]["preference_reviews"][0]["status"] == "pending"
    restarted = MinimalWebApp(tmp_path)
    assert restarted.state()["requires_review"]
    assert not restarted.preview_preferences("other-id")["ok"]
    assert not restarted.start()["ok"]


def test_live_model_path_cannot_switch_to_local_rule_review(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["安排之前先问问我。"])
    app.seal(sid)
    assert not app.preview_preferences("local-review")["ok"]
    assert not clients.memory.calls and not clients.reflection.calls
    app.close()


def test_actual_recall_is_bounded_and_input_history_survives_profile_changes(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = app.start()["state"]["session_id"]
    persona = app._profile()
    persona.episodic_memory = [
        EpisodicMemoryEntry(id="pinwheel", summary="以前修过纸风车。", keywords=["纸风车"], referenceable=True,
                            source_session_id="source-session", source_turn_index=0, source_quote="我以前修过纸风车。",
                            source_context="我以前修过纸风车。",
                            source_context_sha256=source_context_sha256("我以前修过纸风车。"),
                            source_context_version=SOURCE_CONTEXT_VERSION),
        EpisodicMemoryEntry(id="unrelated", summary="喜欢苹果。", keywords=["苹果"], referenceable=True),
        EpisodicMemoryEntry(id="private", summary="纸风车的私密经历。", keywords=["纸风车"], referenceable=False),
    ]
    persona.person_hypotheses = [
        PersonHypothesis(id="bounded", framework="response", claim="交流时可能更接受放慢一步。", scope="一般交流", status="working",
                         confidence=.6, first_observed_at="2026-10-01", evidence=[PersonHypothesisEvidence(
                             at="2026-10-01", session_id="source-session", relation="support", summary="本人举过一次例子。", turn_indices=[0])]),
        PersonHypothesis(id="candidate", framework="response", claim="未有充分支持的候选理解。", status="candidate",
                         confidence=.3, first_observed_at="2026-10-01"),
    ]
    app.store.save(persona)
    assert app.submit("纸风车要修一修。", "bounded-input", sid)["ok"]
    snapshot = deepcopy(app.state()["turns"][-1]["input_evidence"])
    assert snapshot["recalled_episodes"] == ["以前修过纸风车。"]
    assert snapshot["recalled_provenance_not_injected"][0]["sources"][0]["quote"] == "我以前修过纸风车。"
    assert [row["id"] for row in snapshot["hypotheses"]] == ["bounded"]
    prompt = str(clients.online.calls[-1])
    assert "以前修过纸风车。" in prompt and "交流时可能更接受放慢一步。" in prompt
    assert "纸风车的私密经历。" not in prompt and "未有充分支持的候选理解。" not in prompt
    persona.episodic_memory.clear()
    persona.person_hypotheses.clear()
    app.store.save(persona)
    assert app.state()["turns"][-1]["input_evidence"] == snapshot
    app.close()


def test_grounded_model_preference_proposal_survives_offline_queue_commit(tmp_path):
    app, clients = live_fixture(tmp_path)
    words = ["安排之前先问问我。", "我喜欢和大家一起做风车。", "具体什么时候我还想想。"]
    sid = start_and_submit(app, words)
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": [], "stated_preferences": [{
        "semantic_review": communication_review(),  # Software label, not a model-effect result.
        "do": "涉及本人的安排先询问意愿，再一起商量具体时间",
        "quotes": [{"session_id": sid, "turn_index": 0, "text": words[0]}],
    }]}
    app.seal(sid)
    result = app.reflect("model-fixture-review", wait=True)
    assert result["ok"], result
    item = result["state"]["profile"]["stated_preferences"][0]
    assert item["status"] == "proposed" and item["extraction_source"] == "model"
    assert app.preference_action("confirm", item["id"])["ok"]
    next_sid = app.start()["state"]["session_id"]
    assert app.submit("现在聊聊周末。", "next-session", next_sid)["ok"]
    assert item["do"] in str(clients.online.calls[-1])
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    app.close()


def test_failed_generation_retains_prepared_input_and_safety_turn_does_not_reuse_it(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["今天聊聊纸风车。"])
    assert app.state()["turns"][0]["input_evidence"]["passed_to_client"] is True
    assert app.submit("胸痛得厉害，呼吸困难。", "urgent", sid)["ok"]
    row = app.state()["turns"][-1]
    assert row["input_evidence"]["status"] == "not_built"
    assert row["input_evidence"]["passed_to_client"] is False
    assert len(clients.online.calls) == 1
    app.close()

    failing, failure_clients = live_fixture(tmp_path / "failing")
    failure_clients.online.failure = RuntimeError("fixture failure")
    sid = failing.start()["state"]["session_id"]
    result = failing.submit("纸风车怎么折？", "failed-turn", sid)
    row = result["state"]["turns"][0]
    assert not result["ok"] and row["response"] == ""
    assert row["input_evidence"]["passed_to_client"] is True
    assert row["input_evidence"]["generation_status"] == "model_failed"


def test_failure_before_prompt_build_never_reuses_previous_turn_input(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, ["先聊一会儿。"])
    engine = app._sessions["fictional-chen"][0]
    with monkeypatch.context() as patch:
        patch.setattr(engine, "process_turn", lambda *_: (_ for _ in ()).throw(RuntimeError("before prompt")))
        result = app.submit("换个话题。", "before-prompt-failure", sid)
    assert not result["ok"]
    snapshot = result["state"]["turns"][-1]["input_evidence"]
    assert snapshot["status"] == "not_built" and snapshot["passed_to_client"] is False
    assert len(clients.online.calls) == 1


def test_preference_http_requires_owner_token_and_same_origin(local_server):
    app, port = local_server
    pref_id = propose(app)
    payload = {"action": "confirm", "id": pref_id}
    assert request(port, "/api/minimal/preference", method="POST", data=payload)[0] == 403
    assert request(port, "/api/minimal/preference", method="POST", token=app.demo_token,
                   origin="https://invalid.example", data=payload)[0] == 403
    assert request(port, "/api/minimal/preference", method="POST", token=app.demo_token,
                   data={**payload, "replay": True})[0] == 409
    status, response = request(port, "/api/minimal/preference", method="POST", token=app.demo_token, data=payload)
    assert status == 200
    assert json.loads(response)["state"]["profile"]["stated_preferences"][0]["status"] == "confirmed"
    assert request(port, "/api/minimal/preference", method="POST", token=app.demo_token, data=payload)[0] == 409


def test_ui_preference_and_input_panels_are_safe_dom_and_replay_read_only():
    page = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf-8")
    assert 'id="stated-preferences"' in page and 'id="input-evidence"' in page
    assert "replayMode || !!uncertain || state?.storage_failed || !state || viewingHistory()" in page
    assert "if(replayMode || viewingHistory())return;mutate('preference'" in page
    assert "replayMode?turns.find(turn=>turn.turn_id===frame?.turn_id)" in page
    assert "这一旧回合没有输入快照；不从当前档案反推" in page
    assert ".innerHTML" not in page
