"""Version/source/state contracts with explicit labels, not an LLM semantic test."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, heuristic_preferences,
    merge_preferences, preference_admission, transition_preference,
)
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene
from aria.voice.minimal import _USER
from tests.preference_fixtures import communication_review
from tests.test_minimal_web import live_fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Software fixtures cannot call external services")
    stub_model_http(monkeypatch, forbidden)


def source():
    ctx = EngineContext(user_id="semantic-boundary", scene_id="companion_elder", session_id="source-session")
    ctx.add_turn("谈到手工时，先听我说完，再给建议。", "本地软件夹具。")
    return ctx


def row(**changes):
    return {"do": "谈到手工时先听本人说完再给建议", "quotes": [{"turn_index": 0, "text": source().turns[0].user}],
            "semantic_review": communication_review(), **changes}


def profile(*, confirmed=False, legacy=False):
    persona = PersonaProfile(user_id=source().user_id)
    accepted, rejected = extract_preference_rows([row()], source(), persona)
    assert not rejected
    merge_preferences(persona, accepted)
    if confirmed:
        transition_preference(persona, accepted[0].id, "confirm")
    if legacy:
        value = persona.model_dump()
        value["stated_preferences"][0].pop("semantic_review")
        persona = PersonaProfile.model_validate(value)
    return persona


@pytest.mark.parametrize("review", [None, {}, communication_review(version="future"),
    communication_review(owner_requested="true"), communication_review(quote_supported=1),
    communication_review(scope_preserved=None), {**communication_review(), "grant_permission": True}])
def test_missing_or_malformed_semantic_review_is_not_inferred_from_confirmable_words(review):
    accepted, rejected = extract_preference_rows([row(semantic_review=review)], source(), PersonaProfile(user_id="test"))
    assert not accepted and rejected == ["preference_semantic_review_required"]


@pytest.mark.parametrize("change", [
    {"kind": kind} for kind in ("permission_boundary", "action_request", "fact", "person_inference", "unclear")
] + [{key: False} for key in ("owner_requested", "quote_supported", "scope_preserved", "no_permission_claim")])
def test_grammar_is_not_admission_when_explicit_model_label_disqualifies_it(change):
    accepted, rejected = extract_preference_rows([row(semantic_review=communication_review(**change))], source(), PersonaProfile(user_id="test"))
    assert not accepted and rejected == ["preference_not_communication_style"]


def test_full_evidence_and_semantic_review_survive_save_shape_and_confirmation():
    p = profile()
    assert not confirmed_preference_payloads(p)
    pref = p.stated_preferences[0]
    assert pref.scope == source().turns[0].user.rstrip("。")
    assert pref.semantic_review.model_dump() == communication_review()
    transition_preference(p, pref.id, "confirm")
    loaded = PersonaProfile.model_validate_json(p.model_dump_json())
    assert confirmed_preference_payloads(loaded)[0]["id"] == pref.id
    assert loaded.care_plan is None and loaded.person_hypotheses == []
    transition_preference(loaded, pref.id, "withdraw")
    assert confirmed_preference_payloads(loaded) == []


@pytest.mark.parametrize("confirmed", [False, True])
def test_legacy_history_loads_without_promoting_it_and_can_be_reduced(confirmed):
    p = profile(confirmed=confirmed, legacy=True)
    pref = p.stated_preferences[0]
    before = deepcopy(pref.model_dump())
    assert preference_admission(pref) == "needs_review"
    assert not confirmed_preference_payloads(p)
    if not confirmed:
        with pytest.raises(ValueError, match="review_required"):
            transition_preference(p, pref.id, "confirm")
        assert pref.model_dump() == before
    transition_preference(p, pref.id, "withdraw" if confirmed else "decline")
    assert p.stated_preferences[0].history[:-1] == pref.history


def test_replaying_source_does_not_overwrite_legacy_review_or_owner_history():
    p = profile(confirmed=True, legacy=True)
    before = p.model_dump()
    accepted, rejected = extract_preference_rows([row()], source(), p)
    assert not accepted and not rejected and p.model_dump() == before


def test_same_negative_wording_is_decided_by_explicit_semantic_label():
    # Predetermined labels deliberately exercise both routes for the same
    # text. This makes no claim about which label a real model would choose.
    utterance = "我不希望你每次都先问我。"
    ctx = source()
    ctx.turns[0].user = utterance
    value = row(do=utterance, quotes=[{"turn_index": 0, "text": utterance}])
    p = PersonaProfile(user_id=ctx.user_id)
    assert len(extract_preference_rows([value], ctx, p)[0]) == 1
    value["semantic_review"] = communication_review(owner_requested=False)
    assert extract_preference_rows([value], ctx, p) == ([], ["preference_not_communication_style"])


def test_model_source_route_does_not_call_legacy_sentence_parser(monkeypatch):
    import aria.engine.memory.stated_preferences as module
    def legacy_called(*args, **kwargs):
        raise AssertionError("legacy sentence parser called")
    monkeypatch.setattr(module, "_safe_user_instruction", legacy_called)
    assert extract_preference_rows([row()], source(), PersonaProfile(user_id="test"))[0]
    with pytest.raises(AssertionError, match="legacy sentence parser"):
        extract_preference_rows([row()], source(), PersonaProfile(user_id="test"), source="rule")


def test_legacy_preference_does_not_enter_online_or_memory_previous_context():
    p = profile(confirmed=True, legacy=True)
    messages = build_messages(get_scene("companion_elder"), p, source(), "今天想说说新做的椅子。")
    assert p.stated_preferences[0].id not in json.dumps(messages)

    class Memory:
        def complete(self, messages):
            self.messages = messages
            return json.dumps({"episodes": [], "relationships": [], "stated_preferences": [], "cognitive_privacy": []})
    memory = Memory()
    LLMExtractor(memory).extract(source(), p)
    assert p.stated_preferences[0].id not in json.dumps(memory.messages)
    assert "preference_semantics_v1" in memory.messages[0]["content"]


def test_rule_preview_is_explicit_and_cannot_enter_default_live_payload():
    ctx = source()
    ctx.turns[0].user = "说慢一点，别催我。"
    p = PersonaProfile(user_id=ctx.user_id)
    rows = heuristic_preferences(ctx, p)
    assert rows
    merge_preferences(p, rows)
    with pytest.raises(ValueError, match="review_required"):
        transition_preference(p, rows[0].id, "confirm")
    transition_preference(p, rows[0].id, "confirm", allow_rule_preview=True)
    assert not confirmed_preference_payloads(p)
    assert confirmed_preference_payloads(p, allow_rule_preview=True)[0]["admission"] == "local_rule_preview"


@pytest.mark.parametrize("confirmed", [False, True])
def test_live_ui_keeps_legacy_history_disallows_confirmation_and_preserves_care(tmp_path, confirmed):
    app, clients = live_fixture(tmp_path)
    p = profile(confirmed=confirmed, legacy=True)
    p.user_id = _USER
    app.store.save(p)
    state = app.state()
    pref = state["profile"]["stated_preferences"][0]
    assert not pref["can_confirm"] and not pref["usable"] and pref["admission"] == "needs_review"
    care = deepcopy(app.care_state())
    assert not app.preference_action("confirm", pref["id"])["ok"]
    assert app.preference_action("withdraw" if confirmed else "decline", pref["id"])["ok"]
    assert app.care_state() == care
    assert all(not c.calls for c in (clients.online, clients.memory, clients.reflection))
    app.close()


def test_explicitly_imported_rule_profile_does_not_enter_live_request(tmp_path):
    from aria.voice.minimal import MinimalWebApp
    from tests.test_minimal_preferences import propose
    preview = MinimalWebApp(tmp_path)
    pref_id = propose(preview)
    assert preview.preference_action("confirm", pref_id)["ok"]
    rule_profile = preview._profile().model_copy(deep=True)
    preview.close()
    live, clients = live_fixture(tmp_path)
    # Preview and live already use separate stores. Simulate an explicitly
    # imported rule record to verify the additional consumer-side boundary.
    live.store.save(rule_profile)
    sid = live.start()["state"]["session_id"]
    assert live.submit("今天想看看窗台上的小花。", "live-after-preview", sid)["ok"]
    assert pref_id not in json.dumps(clients.online.calls, ensure_ascii=False)
    turn = live.state()["turns"][-1]
    assert turn["input_evidence"]["stated_preferences"] == []
    assert turn["input_evidence"]["stated_preference_protocol_version"] == "2.1"
    item = live.state()["profile"]["stated_preferences"][0]
    assert item["status"] == "confirmed" and not item["usable"]
    assert live.preference_action("withdraw", pref_id)["ok"]
    live.close()
