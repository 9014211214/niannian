"""Prescribed software responses through Web/Engine/queue/store/request.

These neutral fixtures check engineering wiring, not model semantics, exposure
quality or a rerun of any historical score. No real clients or credentials.
"""
from copy import deepcopy
import json

import pytest

from aria.voice.minimal import MinimalWebApp
from tests.test_minimal_web import live_fixture, start_and_submit
from tests.test_reflection_evidence_review import change, review


def words(number):
    return [f"今天练习的是第{number}项新操作。", "先给我一个动作，我更喜欢自己试完再继续。", "这只是我喜欢的沟通节奏。"]


def reflect_prescribed(app, clients, number, *, relation="support", strength=.9,
                      event_kind="new_event", prior_event_id=None):
    sid = start_and_submit(app, words(number))
    current = app._sessions["fictional-chen"][1]
    persona = app._profile()
    old = persona.person_hypotheses[0] if persona.person_hypotheses else None
    item = review(current, old, relation=relation, event_kind=event_kind,
                  prior_event_id=prior_event_id)
    proposal = change(item, old, confidence=strength)
    if relation == "retire":
        item["relation"] = "counter"
        proposal["relation"] = "retire"
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    clients.reflection.payload = {"protocol_version": "2.0", "reviews": [item], "changes": [proposal]}
    assert app.seal(sid)["ok"]
    result = app.reflect(f"chain-review-{number}", wait=True)
    return sid, result


def learn_working(app, clients):
    _, first = reflect_prescribed(app, clients, 1)
    assert first["ok"] and first["state"]["profile"]["hypotheses"][0]["status"] == "candidate"
    _, second = reflect_prescribed(app, clients, 2)
    assert second["ok"]
    hypothesis, = app.store.load("fictional-chen").person_hypotheses
    assert hypothesis.status == "working"
    assert hypothesis.confidence == pytest.approx(.656)
    assert len(hypothesis.supporting_events()) == 2
    assert len({(e.session_id, i) for e in hypothesis.evidence for i in e.turn_indices}) == 4
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


def test_earned_working_scope_survives_reload_and_repeated_event_has_no_new_vote(tmp_path):
    app, clients = live_fixture(tmp_path)
    learned = learn_working(app, clients)
    before = learned.model_dump()
    reopened = MinimalWebApp(tmp_path, clients=clients)
    _, result = reflect_prescribed(reopened, clients, 3, event_kind="same_event",
                                   prior_event_id=learned.evidence[0].event_id)
    assert result["ok"]
    after, = reopened.store.load("fictional-chen").person_hypotheses
    assert after.model_dump() == before
    assert reopened._profile().evidence_review_ledger[-1].disposition == "same_event_no_fresh_evidence"
    check_next_input(reopened, clients, expected=[after])
    # The provided prior identity is actually in the recorded reflection input.
    prior_input = clients.reflection.calls[-1][1]["content"]
    assert learned.id in prior_input and learned.evidence[0].event_id in prior_input
    assert learned.scope in prior_input
    reopened.close()


@pytest.mark.parametrize("relation,strength,expected_status,expected_confidence", [
    ("counter", 1, "candidate", .456),
    ("retire", .9, "retired", 0),
])
def test_counter_or_explicit_retire_changes_durable_later_use(tmp_path, relation, strength,
                                                            expected_status, expected_confidence):
    app, clients = live_fixture(tmp_path)
    before = learn_working(app, clients)
    _, result = reflect_prescribed(app, clients, 3, relation=relation, strength=strength)
    assert result["ok"]
    after, = app.store.load("fictional-chen").person_hypotheses
    assert after.id == before.id and after.claim == before.claim and after.scope == before.scope
    assert after.status == expected_status and after.confidence == pytest.approx(expected_confidence)
    assert after.evidence[-1].relation == "counter"
    reopened = MinimalWebApp(tmp_path, clients=clients)
    check_next_input(reopened, clients, expected=[])
    assert before.claim not in json.dumps(clients.online.calls[-1], ensure_ascii=False)
    reopened.close()


def test_invalid_reflection_does_not_publish_half_a_review_or_hypothesis(tmp_path):
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, words(1))
    ctx = app._sessions["fictional-chen"][1]
    item = review(ctx)
    # A second invalid context must reject the whole response before adoption.
    invalid = review(ctx, id="bad-context", event_kind="same_event", prior_event_id=None)
    clients.memory.payload = {"episodes": [], "relationships": [], "cognitive_privacy": []}
    clients.reflection.payload = {"protocol_version":"2.0", "reviews":[item, invalid], "changes":[change(item)]}
    assert app.seal(sid)["ok"]
    result = app.reflect("chain-invalid", wait=True)
    assert not result["ok"]
    saved = app.store.load("fictional-chen")
    assert not saved.person_hypotheses and not saved.evidence_review_ledger
    assert sid not in saved.person_model_state.reflected_session_ids
    # This unmetered fixture keeps the failed stage pending; it is not "no evidence".
    assert result["state"]["reviews"][-1]["status"] == "failed"
    assert saved.offline_sessions[0].last_error_code == "same_event_requires_known_reference"
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
        if candidate.person_hypotheses:
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
