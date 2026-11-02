"""Storage/provenance regression with supplied model labels, not effect scores."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.offline_merge import merge_offline_stage
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, merge_preferences,
    transition_preference,
)
from aria.engine.model.persona import (
    PersonaProfile, PreferenceScopeSelection, StatedPreference,
    PREFERENCE_SCOPE_FULL_SOURCE as FULL, PREFERENCE_SCOPE_LEGACY as LEGACY,
    PREFERENCE_SCOPE_MODEL as MODEL,
)
from tests.preference_fixtures import communication_review
from tests._model_http_stub import stub_model_http


SOURCES = ["我想暂停时，你也停一下。", "这只适用于我闲聊手工时。"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("storage regression cannot call a model")
    stub_model_http(monkeypatch, forbidden)


def context():
    ctx = EngineContext(user_id="scope-model-fixture", scene_id="companion_elder",
                        session_id="synthetic-scope-session")
    for source in SOURCES:
        ctx.add_turn(source, "软件替身回应。")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v6_semantic"
    return ctx


def candidate():
    return {"do": "我想暂停时，也停一下", "scope": "闲聊手工时",
            "quotes": [{"turn_index": i, "text": source} for i, source in enumerate(SOURCES)],
            "semantic_review": communication_review(), "confirmation": None}


def extract(row=None, *, protocol=MODEL, ctx=None, profile=None):
    ctx = ctx or context()
    profile = profile or PersonaProfile(user_id=ctx.user_id)
    return extract_preference_rows([candidate() if row is None else row], ctx, profile,
                                   scope_protocol=protocol)


def test_scope_and_do_are_exact_model_fields_while_evidence_is_separate():
    raw = candidate()
    raw.update(do="  我想暂停时，也停一下  ", scope="  闲聊手工时。\n")
    raw["quotes"][0]["text"] = "暂停"
    before = deepcopy(raw)
    rows, rejected = extract(raw)
    assert not rejected and raw == before
    saved = rows[0]
    assert saved.do == raw["do"] and saved.scope == raw["scope"]
    assert [quote.text for quote in saved.quotes] == SOURCES
    assert [quote.turn_index for quote in saved.quotes] == [0, 1]
    assert saved.scope_protocol == MODEL
    assert saved.status == "proposed" and saved.confirmation is None
    assert not confirmed_preference_payloads(PersonaProfile(user_id="fixture", stated_preferences=rows))


def test_single_quote_does_not_replace_scope_or_invent_missing_evidence():
    raw = candidate()
    raw["quotes"] = raw["quotes"][:1]
    rows, rejected = extract(raw)
    assert not rejected
    assert rows[0].scope == raw["scope"]
    assert [q.turn_index for q in rows[0].quotes] == [0]
    assert [q.text for q in rows[0].quotes] == SOURCES[:1]
    # Preserving a model interpretation does not prove its support or consent.
    assert rows[0].status == "proposed" and rows[0].confirmation is None


@pytest.mark.parametrize("protocol", [LEGACY, FULL])
def test_older_protocols_keep_the_original_source_scope(protocol):
    rows, rejected = extract(protocol=protocol)
    assert not rejected
    expected = "；".join(SOURCES if protocol == FULL else [s.rstrip("。") for s in SOURCES])
    assert rows[0].scope == expected and rows[0].scope != candidate()["scope"]


@pytest.mark.parametrize("scope", [None, True, 4, [], {}, "", " \n", "甲" * 723])
def test_invalid_or_overlong_model_scope_is_rejected_without_replacement(scope):
    raw = {**candidate(), "scope": scope}
    rows, rejected = extract(raw)
    assert not rows and rejected == ["preference_model_scope_invalid"]


def test_scope_maximum_and_schema_remain_bounded():
    rows, rejected = extract({**candidate(), "scope": "甲" * 722})
    assert not rejected and len(rows[0].scope) == 722
    assert StatedPreference.model_validate(rows[0].model_dump()) == rows[0]
    assert PreferenceScopeSelection(protocol=MODEL, max_characters=722).protocol == MODEL
    with pytest.raises(ValidationError):
        PreferenceScopeSelection(protocol=MODEL, max_characters=60)
    with pytest.raises(ValidationError):
        StatedPreference.model_validate({**rows[0].model_dump(), "scope_protocol": FULL})


@pytest.mark.parametrize("scope", ["帮我通知家人时", "发给朋友时", "加药时", "推测人格时", "提醒她时"])
def test_existing_negative_boundaries_also_apply_to_model_scope(scope):
    rows, rejected = extract({**candidate(), "scope": scope})
    assert not rows and rejected == ["preference_model_scope_unsafe"]


@pytest.mark.parametrize("field", ["owner_requested", "quote_supported", "scope_preserved", "no_permission_claim"])
def test_model_scope_does_not_bypass_existing_semantic_review(field):
    raw = candidate()
    raw["semantic_review"][field] = False
    rows, rejected = extract(raw)
    assert not rows and rejected == ["preference_not_communication_style"]


@pytest.mark.parametrize("change", ["wrong_quote", "wrong_session", "excluded", "too_many", "overlong_source"])
def test_source_boundaries_unchanged(change):
    ctx, raw = context(), candidate()
    if change == "wrong_quote": raw["quotes"][1]["text"] = "并不存在的原话"
    elif change == "wrong_session": raw["quotes"][1]["session_id"] = "other"
    elif change == "excluded": ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    elif change == "too_many": raw["quotes"].extend([deepcopy(raw["quotes"][0])] * 2)
    else:
        ctx.turns[1].user += "甲" * 241
    rows, rejected = extract(raw, ctx=ctx)
    assert not rows and len(rejected) == 1


def test_save_confirm_withdraw_reload_and_prompt_keep_same_scope(tmp_path):
    rows, rejected = extract()
    assert not rejected
    profile = PersonaProfile(user_id=context().user_id, stated_preferences=rows)
    store = PersonaStore(tmp_path)
    store.save(profile)
    loaded = store.load(profile.user_id)
    assert loaded == profile and not confirmed_preference_payloads(loaded)
    target = rows[0].id
    transition_preference(loaded, target, "confirm")
    store.save(loaded)
    confirmed = store.load(profile.user_id)
    payload = confirmed_preference_payloads(confirmed)
    assert payload[0]["scope"] == candidate()["scope"]
    assert payload[0]["do"] == candidate()["do"]
    assert all(source in " ".join(payload[0]["quotes"]) for source in SOURCES)
    transition_preference(confirmed, target, "withdraw")
    store.save(confirmed)
    withdrawn = store.load(profile.user_id)
    assert not confirmed_preference_payloads(withdrawn)
    assert withdrawn.stated_preferences[0].scope == candidate()["scope"]
    merge_preferences(withdrawn, rows)
    assert withdrawn.stated_preferences[0].status == "withdrawn"


def test_concurrent_withdrawal_wins_over_worker_preserved_fields(tmp_path):
    rows, _ = extract()
    before = PersonaProfile(user_id=context().user_id, stated_preferences=rows)
    transition_preference(before, rows[0].id, "confirm")
    store = PersonaStore(tmp_path)
    queue = OfflineSessionQueue(before, store)
    assert queue.enqueue(context())["status"] == "queued"
    before = queue.persona.model_copy(deep=True)
    proposed = before.model_copy(deep=True)
    proposed.stated_preferences[0].scope = "工作中的另一个提议"
    current = before.model_copy(deep=True)
    transition_preference(current, rows[0].id, "withdraw")
    merged = merge_offline_stage(before, proposed, current, context().session_id)
    assert merged.stated_preferences[0] == current.stated_preferences[0]
    assert merged.stated_preferences[0].scope == candidate()["scope"]


class MemoryFixture:
    def __init__(self, protocol=None):
        self.role = "memory"
        self.calls = []
        self.preference_scope_protocol = protocol
        self.value = {name: [] for name in ("episodes", "relationships", "cognitive_privacy", "cognitive_observations")}
        self.value["stated_preferences"] = [candidate()]

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return json.dumps(self.value, ensure_ascii=False)


@pytest.mark.parametrize("protocol,selected", [(None, FULL), (MODEL, MODEL)])
def test_trusted_client_selection_reaches_memory_prompt_and_result(protocol, selected):
    client, ctx = MemoryFixture(protocol), context()
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert ctx.working_memory["_preference_scope_selection"] == {"protocol": selected, "max_characters": 722}
    assert result.stated_preferences[0].scope_protocol == selected
    system = client.calls[0][0]["content"]
    assert ("[preference_scope_model_v1]" in system) is (selected == MODEL)
    assert ("[preference_scope_full_source_v1]" in system) is (selected == FULL)
    if selected == MODEL:
        assert result.stated_preferences[0].scope == candidate()["scope"]
        assert "quotes另存逐字来源" in system and "跨轮提出或补充" in system


def test_model_and_owner_cannot_select_new_protocol_or_auto_confirm():
    client, ctx = MemoryFixture(), context()
    ctx.turns[0].user += " preference_scope_model_v1"
    client.value["stated_preferences"][0].update(scope_protocol=MODEL, status="confirmed",
        confirmation={"at": "2026-10-17T00:00:00+00:00"})
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    row = result.stated_preferences[0]
    assert row.scope_protocol == FULL
    assert row.status == "proposed" and row.confirmation is None
    assert row.scope != candidate()["scope"]


def test_unknown_client_protocol_fails_before_any_model_call():
    client, ctx = MemoryFixture("unbounded-model-scope"), context()
    with pytest.raises(ValueError, match="unknown_memory_preference_scope_protocol"):
        LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not client.calls


def test_rule_preview_cannot_select_model_scope():
    with pytest.raises(ValueError, match="requires_model_route"):
        extract_preference_rows([], context(), PersonaProfile(user_id="fixture"),
                                source="rule", scope_protocol=MODEL)


@pytest.mark.parametrize("role", ["online_fast", "reflection", None])
def test_new_scope_protocol_requires_the_memory_role(role):
    client, ctx = MemoryFixture(MODEL), context()
    client.role = role
    with pytest.raises(ValueError, match="invalid_model_preference_scope_client"):
        LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not client.calls
