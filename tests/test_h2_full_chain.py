"""Prescribed software responses through Web/Engine/queue/store/request.

These neutral fixtures check engineering wiring, not model semantics, exposure
quality or a rerun of any historical score. No real clients or credentials.
"""
from copy import deepcopy
import json

import pytest

from aria.voice.minimal import MinimalWebApp
from aria.engine.policy.prompt import confirmed_prompt_hypotheses
from tests.test_minimal_web import live_fixture, start_and_submit
from tests.test_reflection_evidence_review import change, context, review, setup, support


def words(number):
    return [f"今天练习的是第{number}项新操作。", "先给我一个动作，我更喜欢自己试完再继续。", "这只是我喜欢的沟通节奏。"]


def reflect_prescribed(app, clients, number, *, relation="support", strength=.9,
                      event_words=None):
    sid = start_and_submit(app, words(number) if event_words is None else event_words)
    current = app._sessions["fictional-chen"][1]
    persona = app._profile()
    old = persona.person_hypotheses[0] if persona.person_hypotheses else None
    item = review(current, old, relation=relation)
    proposal = change(item, old, confidence=strength)
    if relation == "retire":
        item["relation"] = "counter"
        proposal["relation"] = "retire"
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    clients.reflection.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [proposal]}
    assert app.seal(sid)["ok"]
    result = app.reflect(f"chain-review-{number}", wait=True)
    return sid, result


def seed_historical_working(app):
    """Persist previously accepted software evidence, not current model output."""
    updater, llm, _ = setup()
    persona = app._profile() or app._seed_persona("fictional-chen")
    first = support(updater, llm, persona, 1, ctx=context(1, words=words(1)))
    assert first.status == "candidate" and first.confidence == pytest.approx(.48)
    assert len(first.supporting_events()) == 1
    assert confirmed_prompt_hypotheses(persona) == []
    support(updater, llm, persona, 2, ctx=context(2, words=words(2)))
    app.store.save(persona)
    hypothesis, = app.store.load("fictional-chen").person_hypotheses
    assert hypothesis.status == "working"
    assert hypothesis.confidence == pytest.approx(.656)
    assert len(hypothesis.supporting_events()) == 2
    assert len({(e.session_id, i) for e in hypothesis.evidence for i in e.turn_indices}) == 4
    assert not llm.messages  # Seeding is explicitly separate from reflection.
    return hypothesis


def check_next_input(app, clients, *, expected):
    sid = app.start()["state"]["session_id"]
    assert app.submit("现在练习另一个新操作。", f"chain-next-{sid}", sid)["ok"]
    actual = app.state()["turns"][-1]["input_evidence"]
    assert actual["passed_to_client"] is True
    assert [h["id"] for h in actual["hypotheses"]] == [h.id for h in expected]
    serialized = json.dumps(clients.online.calls[-1], ensure_ascii=False)
    for h in expected:
        assert h.claim in serialized and h.scope in serialized
        assert actual["hypotheses"][0]["scope"] == h.scope
    return actual


def test_historical_working_scope_survives_reload_and_repeated_event_has_no_new_vote(tmp_path):
    app, clients = live_fixture(tmp_path)
    learned = seed_historical_working(app)
    before = learned.model_dump()
    reopened = MinimalWebApp(tmp_path, clients=clients)
    _, result = reflect_prescribed(reopened, clients, 3, event_words=words(1))
    assert result["ok"]
    after, = reopened.store.load("fictional-chen").person_hypotheses
    assert after.model_dump() == before
    assert reopened._profile().evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    check_next_input(reopened, clients, expected=[after])
    # Persisted prior identity is in the input; the response does not supply it.
    prior_input = clients.reflection.calls[-1][1]["content"]
    assert learned.id in prior_input and learned.evidence[0].event_id in prior_input
    assert learned.scope in prior_input
    assert not {"event_kind", "prior_event_id"} & clients.reflection.payload["reviews"][0].keys()
    reopened.close()


@pytest.mark.parametrize("relation,strength", [
    ("counter", 1),
    ("retire", .9),
])
def test_grounded_counter_or_retire_changes_durable_prior_and_next_request(tmp_path, relation, strength):
    app, clients = live_fixture(tmp_path)
    before = seed_historical_working(app)
    _, result = reflect_prescribed(app, clients, 3, relation=relation, strength=strength)
    assert result["ok"]
    after, = app.store.load("fictional-chen").person_hypotheses
    assert after.confidence < before.confidence
    assert after.status == ("retired" if relation == "retire" else "candidate")
    assert after.claim == before.claim and after.scope == before.scope
    observed = app._profile().evidence_review_ledger[-1]
    assert observed.relation == "counter"
    assert observed.disposition == "applied"
    reopened = MinimalWebApp(tmp_path, clients=clients)
    check_next_input(reopened, clients, expected=[])
    reopened.close()


@pytest.mark.parametrize("itemwise", [False, True])
def test_invalid_reflection_respects_atomic_or_itemwise_validation(tmp_path, itemwise):
    app, clients = live_fixture(tmp_path)
    clients.reflection.reflection_itemwise_review = itemwise
    sid = start_and_submit(app, words(1))
    ctx = app._sessions["fictional-chen"][1]
    item = review(ctx)
    # Removed model event fields remain illegal. Itemwise processing retains
    # the independent review; legacy atomic processing publishes no half batch.
    invalid = review(ctx, id="bad-context", event_kind="same_event", prior_event_id=None)
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    clients.reflection.payload = {"protocol_version":"2.0", "reviews":[item, invalid], "changes":[change(item)]}
    assert app.seal(sid)["ok"]
    result = app.reflect("chain-invalid", wait=True)
    saved = app.store.load("fictional-chen")
    assert not saved.person_hypotheses
    if itemwise:
        assert result["ok"] and len(saved.evidence_review_ledger) == 1
        assert saved.evidence_review_ledger[0].disposition == "event_identity_unknown_no_update"
        audit = saved.offline_session_audit[-1].reflection_result
        assert audit.status == "partial"
        assert audit.accepted_review_count == audit.rejected_review_count == 1
        assert audit.applied_proposal_count == 0 and audit.deferred_proposal_count == 1
    else:
        assert not result["ok"] and not saved.evidence_review_ledger
        assert sid not in saved.person_model_state.reflected_session_ids
        # The unmetered legacy fixture keeps a failed stage pending.
        assert result["state"]["reviews"][-1]["status"] == "failed"
        assert saved.offline_sessions[0].last_error_code == "invalid_review_fields"
    app.close()


def test_reflection_save_failure_keeps_last_committed_profile_and_blocks_reissue(tmp_path, monkeypatch):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, words(1))
    ctx = app._sessions["fictional-chen"][1]
    item = review(ctx)
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    clients.reflection.payload = {"protocol_version":"2.0", "reviews":[item], "changes":[change(item)]}
    assert app.seal(sid)["ok"]
    original_save = app.store.save

    def reject_adoption(candidate):
        if candidate.evidence_review_ledger:
            raise OSError("synthetic reflection commit failure")
        original_save(candidate)

    with monkeypatch.context() as patch:
        patch.setattr(app.store, "save", reject_adoption)
        result = app.reflect("chain-failed-commit", wait=True)
    assert not result["ok"] and result["state"]["requires_review"]
    # Engine propagates the storage exception; Web retains its pending review
    # instead of inventing a completed/failed semantic result.
    assert app._run["reviews"][-1]["status"] == "pending"
    assert "保存结果不明" in app._run["reviews"][-1]["error"]
    for profile in (app._profile(), app.store.load("fictional-chen")):
        assert not profile.person_hypotheses and not profile.evidence_review_ledger
        assert sid not in profile.person_model_state.reflected_session_ids
        assert profile.offline_sessions[0].memory_completed
        assert not profile.offline_sessions[0].reflection_completed
    before_calls = deepcopy(clients.reflection.calls)
    assert not app.reflect("chain-another-button", wait=True)["ok"]
    assert clients.reflection.calls == before_calls


def test_cold_start_real_product_path_support_correction_retirement_and_next_prompt(tmp_path):
    """Forced semantic labels prove reachability, never model learning accuracy."""
    from aria.runtime import MinimalModelClients
    from tests.test_minimal_web import RecordingClient

    clients = MinimalModelClients(*(RecordingClient(role) for role in ("online", "memory", "reflection")),
                                  model_calls_enabled=True)
    clients.online.online_decision_version = "online_turn_decision_v7"
    clients.online.payload = {"safety_concern": "none", "response_segments": [{"text": "先做第一个动作，做好后我们再往下。"}]}
    clients.reflection.reflection_itemwise_review = True
    app = MinimalWebApp(tmp_path, clients=clients, safety_delivery_version="minimal_response_v1",
                        safety_response_instruction_version="compact_response_v1")
    sessions = [
        ["我们练习画圆。", "照着你刚才说的画好了圆，我愿意再试。", "这种一步一步的节奏让我愿意继续。"],
        ["我们练习折纸。", "照着你刚才说的折好了第一角，我愿意接着折。", "这种逐步说明的节奏让我继续得下去。"],
    ]

    def finish(words, number, relation="support"):
        sid = start_and_submit(app, words)
        ctx = app._sessions["fictional-chen"][1]
        old = next(iter(app._profile().person_hypotheses), None)
        item = review(ctx, old, relation="counter" if relation in {"counter", "retire"} else "support",
                      evidence_basis="observed_response" if relation == "support" else "stated_preference")
        item["event_evidence"] = [deepcopy(item["evidence"][0])]
        proposal = change(item, old, relation="new" if old is None else relation)
        if relation == "support":
            proposal["atomic_claim"]["outcome"] = "本人愿意继续尝试"
            assert {"0", "1"} <= set(ctx.working_memory["_person_response_deliveries"])
        clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
        clients.reflection.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [proposal]}
        assert app.seal(sid)["ok"]
        job = next(job for job in app._profile().offline_sessions if job.session_id == sid)
        if relation == "support":
            assert job.turns[0].person_response_sha256 is not None
        assert app.reflect(f"cold-review-{number}", wait=True)["ok"]
        return app.store.load("fictional-chen").person_hypotheses[0]

    assert not (app._profile() and app._profile().person_hypotheses)
    first = finish(sessions[0], 1)
    assert first.status == "candidate" and first.confidence == pytest.approx(.48)
    assert confirmed_prompt_hypotheses(app._profile()) == []
    repeated = finish(sessions[0], 2)
    assert repeated.model_dump() == first.model_dump()
    working = finish(sessions[1], 3)
    assert working.status == "working" and working.confidence == pytest.approx(.656)
    assert len(working.supporting_events()) == len(working.supporting_sessions()) == 2
    corrected = finish(["我想纠正刚才的理解。", "逐步等候让我没法看清全程，我现在要先看完整路线。", "别把我固定成喜欢一次等一个动作。"], 4, "counter")
    assert corrected.confidence < working.confidence and corrected.status == "candidate"
    assert working.claim in json.dumps(clients.online.calls[-3], ensure_ascii=False)
    retired = finish(["撤掉那个对我的猜测。", "不要再沿用我偏爱一步一步听的猜测。", "以后当场问我就好，那条推断不要再用。"], 5, "retire")
    assert retired.status == "retired" and retired.confidence == 0
    assert retired.claim == working.claim and retired.scope == working.scope
    reopened = MinimalWebApp(tmp_path, clients=clients, safety_delivery_version="minimal_response_v1",
                           safety_response_instruction_version="compact_response_v1")
    check_next_input(reopened, clients, expected=[])
    assert retired.claim not in json.dumps(clients.online.calls[-1], ensure_ascii=False)
    reopened.close()
