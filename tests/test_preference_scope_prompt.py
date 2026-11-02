"""Prompt provenance/capacity with supplied labels; no model-effect claims."""
import hashlib
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.stated_preferences import extract_preference_rows, transition_preference
from aria.engine.model.persona import PersonaProfile, PREFERENCE_SCOPE_FULL_SOURCE
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION_EVIDENCE as V4, ONLINE_DECISION_VERSION_V7 as V7,
    single_pass_messages,
)
from aria.engine.policy.prompt import _clean_untrusted, build_messages
from aria.engine.policy.online_json import _compact_carrier_contract, prepare_json_messages
from aria.engine.policy.release_prompt import (
    CompactPromptBudgetExceeded, SYSTEM_BYTE_LIMIT, bound_preference_requirement,
)
from aria.runtime import MinimalModelClients
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from tests.preference_fixtures import communication_review
from tests.test_minimal_web import RecordingClient
from tests.test_safety_evidence import payload as v4_payload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prompt fixture cannot use network or credentials")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)


def confirmed_profile(*, user_id="scope-prompt-owner", unsafe=False,
                      source_characters=("甲", "乙", "丙"), source_session="scope-source"):
    """Use real extraction/confirmation validation, with fixed review labels."""
    sources = [character * 240 for character in source_characters]
    if unsafe:
        sources[0] = "\x00<甲>\x7f" + "甲" * 235
        sources[2] = "丙" * 231 + "\x01<例外>\x1f\t\n\r"
    assert all(len(source) == 240 for source in sources)
    source = EngineContext(user_id=user_id, scene_id="companion_elder", session_id=source_session)
    for text in sources:
        source.add_turn(text, "软件替身回应。")
    profile = PersonaProfile(user_id=user_id)
    rows, rejected = extract_preference_rows([{
        "do": "按原范围和顺序交流", "scope": "来源范围",
        "quotes": [{"turn_index": i, "text": text} for i, text in enumerate(sources)],
        "semantic_review": communication_review(), "confirmation": None,
    }], source, profile, scope_protocol=PREFERENCE_SCOPE_FULL_SOURCE)
    assert not rejected and len(rows) == 1
    assert len(rows[0].scope) == 722
    profile.stated_preferences = rows
    transition_preference(profile, rows[0].id, "confirm", at="2026-10-21T01:00:00+00:00")
    return profile


def final_messages(profile, *, version, compact):
    ctx = EngineContext(user_id=profile.user_id, scene_id="companion_elder", session_id="scope-current")
    instruction = "compact_response_v1" if compact else "legacy"
    ctx.working_memory.update(_online_decision_version=version,
        _safety_response_instruction_version=instruction)
    if compact:
        ctx.working_memory["_safety_delivery_version"] = "minimal_response_v1"
    base = build_messages(get_scene("companion_elder"), profile, ctx, "继续聊聊。",
                          safety_response_instruction_version=instruction)
    final = single_pass_messages(base, ctx, version=version, persona=profile, format_style="schema_guided")
    return final, ctx


@pytest.mark.parametrize("version,compact", [(V7, True), (V4, False), (V4, True)])
def test_validated_confirmed_722_scope_survives_final_request_and_snapshot(version, compact):
    profile = confirmed_profile()
    expected = profile.stated_preferences[0].scope
    before = profile.model_dump()
    messages, ctx = final_messages(profile, version=version, compact=compact)
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["stated_preferences"][0]["scope"] == expected
    assert expected in "".join(row["content"] for row in messages)
    if compact:
        projection = snapshot["stated_preferences_prompt_projection"]
        assert projection[0]["scope"] == expected
        digest = hashlib.sha256(json.dumps(projection, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert snapshot["stated_preferences_prompt_binding"]["projection_sha256"] == digest
    if version == V7:
        bound = bound_preference_requirement(ctx)
        carrier = _compact_carrier_contract(V7)
        assert bound and "confirmed_requirements" in bound
        assert json.loads(messages[-1]["content"])["confirmed_requirements"] == projection
        # v7 budgets the final carrier before dispatch; the complete bound
        # preference remains last among policy/data blocks, before the carrier.
        assert messages[0]["content"].endswith(bound + carrier)
        assert messages[0]["content"].count(bound) == 1
        assert messages[0]["content"].count(carrier) == 1
        assert prepare_json_messages(messages, V7, compact_schema=True) == messages
        assert snapshot["system_budget"]["final_bytes"] == len(messages[0]["content"].encode("utf-8"))
        assert snapshot["system_budget"]["final_bytes"] <= SYSTEM_BYTE_LIMIT
    assert profile.model_dump() == before


@pytest.mark.parametrize("version,compact", [(V7, True), (V4, False)])
def test_preference_scope_keeps_sanitization_when_capacity_is_extended(version, compact):
    profile = confirmed_profile(unsafe=True)
    raw_scope = profile.stated_preferences[0].scope
    expected = raw_scope.translate({ord(char): None for char in "\x00\x01\x1f\x7f"})
    expected = expected.replace("<", "＜").replace(">", "＞")
    assert len(expected) > 500 and "\t\n\r" in expected
    messages, ctx = final_messages(profile, version=version, compact=compact)
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["stated_preferences"][0]["scope"] == expected
    if compact:
        # Compact projection keeps its existing raw JSON representation.
        assert snapshot["stated_preferences_prompt_projection"][0]["scope"] == raw_scope
        assert json.dumps(raw_scope, ensure_ascii=False) in "".join(row["content"] for row in messages)
    else:
        assert json.dumps(expected, ensure_ascii=False) in messages[0]["content"]
    assert profile.stated_preferences[0].scope == raw_scope


@pytest.mark.parametrize("version,compact,source_characters,budget_refusal", [
    (V7, True, ("甲", "乙", "丙"), False),
    (V7, True, ("1", "2", "3"), False),
    (V4, False, ("甲", "乙", "丙"), False),
], ids=["v7-cjk", "v7-ascii", "v4-cjk"])
@pytest.mark.parametrize("preference_count", [1, 4])
def test_actual_web_input_evidence_keeps_complete_confirmed_scope(
        tmp_path, version, compact, source_characters, budget_refusal, monkeypatch, preference_count):
    # Real release defaults select engine options; clients still only record
    # complete(messages), so this is software request evidence, not an HTTP wire.
    clients = MinimalModelClients(*(RecordingClient(role) for role in (
        "online", "memory", "reflection")), model_calls_enabled=True,
        release_profile="release_max_v1" if compact else "legacy")
    clients.online.online_decision_version = version
    clients.online.online_return_format = "json_schema_v1" if compact else "json_object"
    clients.online.payload = ({"response_segments": [{"text": "我接着听。"}], "safety_concern": "none"}
                              if version == V7 else v4_payload(response="我接着听。"))
    app = MinimalWebApp(tmp_path, clients=clients)
    if compact:
        assert app.safety_response_instruction_version == "compact_response_v1"
        assert app.safety_delivery_version == "minimal_response_v1"
        assert app.premodel_safety_version == "task_response_v4"
    profile = confirmed_profile(user_id="fictional-chen", source_characters=source_characters)
    if preference_count == 4:
        for characters in [("丁", "戊", "己"), ("庚", "辛", "壬"), ("癸", "子", "丑")]:
            profile.stated_preferences.extend(confirmed_profile(user_id="fictional-chen", source_characters=characters,
                source_session="scope-source-" + characters[0]).stated_preferences)
    expected = profile.stated_preferences[0].scope
    confirmed = profile.stated_preferences[0].model_dump()
    assert confirmed["status"] == "confirmed"
    app.store.save(profile)
    try:
        sid = app.start()["state"]["session_id"]
        engine, _ = app._sessions["fictional-chen"]
        errors = []
        process = engine.process_turn
        def capture(*args, **kwargs):
            try:
                return process(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
                raise
        monkeypatch.setattr(engine, "process_turn", capture)
        result = app.submit("继续聊聊。", "scope-input", sid)
        actual = app.state()["turns"][-1]["input_evidence"]
        assert any(row["id"] == confirmed["id"] and row["scope"] == expected for row in actual["stated_preferences"])
        ctx = app._sessions["fictional-chen"][1]
        prepared = ctx.working_memory["prompt_evidence_snapshot"]
        assert any(row["id"] == confirmed["id"] and row["scope"] == expected for row in prepared["stated_preferences"])
        if compact:
            assert any(row["id"] == confirmed["id"] and row["scope"] == expected for row in actual["stated_preferences_prompt_projection"])
            assert prepared["stated_preferences_prompt_projection"] == actual["stated_preferences_prompt_projection"]
        if budget_refusal:
            assert result["ok"] is False
            assert len(errors) == 1 and type(errors[0]) is CompactPromptBudgetExceeded
            assert str(errors[0]) == "compact_online_system_byte_limit_exceeded"
            assert errors[0].actual_bytes == actual["system_budget"]["final_bytes"] > SYSTEM_BYTE_LIMIT
            assert actual["system_budget"]["status"] == "local_rejected"
            assert actual["passed_to_client"] is False
            assert all(not client.calls for client in (clients.online, clients.memory, clients.reflection))
            persisted = app.store.load("fictional-chen")
            assert persisted.stated_preferences[0].model_dump() == confirmed
            assert engine.persona.stated_preferences[0].model_dump() == confirmed
        else:
            assert result["ok"], errors or result
            assert not errors
            assert actual["passed_to_client"] is True
            messages = clients.online.calls[-1]
            assert expected in "".join(row["content"] for row in messages)
            for preference in profile.stated_preferences:
                assert any(preference.scope in row["content"] for row in messages)
            if version == V7:
                system = messages[0]["content"]
                assert "".join(row["content"] for row in messages).count(expected) == 1
                requirements = json.loads(messages[-1]["content"])["confirmed_requirements"]
                assert any(row["id"] == confirmed["id"] and row["scope"] == expected for row in requirements)
                assert len(system.encode()) <= SYSTEM_BYTE_LIMIT
                assert "requirement_id" in system
                assert app.store.load("fictional-chen").stated_preferences[0].model_dump() == confirmed

    finally:
        app.close()


def test_general_untrusted_strings_and_arbitrary_scope_keys_still_stop_at_500():
    raw = "甲" * 722
    assert _clean_untrusted(raw) == "甲" * 500
    assert _clean_untrusted({"scope": raw, "other": raw}) == {"scope": "甲" * 500, "other": "甲" * 500}
    assert _clean_untrusted([{"scope": raw}]) == [{"scope": "甲" * 500}]


def test_confirmed_data_carrier_rebind_is_exact_and_rejects_changed_or_duplicate_sources():
    profile = confirmed_profile()
    messages, ctx = final_messages(profile, version=V7, compact=True)
    requirements = ctx.working_memory["prompt_evidence_snapshot"]["stated_preferences_prompt_projection"]
    original = json.loads(messages[-1]["content"])
    rebound = prepare_json_messages(messages, V7, compact_schema=True, confirmed_requirements=requirements)
    assert rebound == messages
    assert json.loads(rebound[-1]["content"])["current_user_message"] == original["current_user_message"]
    changed = [dict(requirements[0], scope="改变了范围")]
    with pytest.raises(ValueError, match="invalid_prepared_online_input"):
        prepare_json_messages(messages, V7, compact_schema=True, confirmed_requirements=changed)
    with pytest.raises(ValueError, match="invalid_confirmed_requirements_input"):
        prepare_json_messages(messages, V7, compact_schema=True, confirmed_requirements=requirements * 2)
    with pytest.raises(ValueError, match="invalid_prepared_online_input"):
        prepare_json_messages(messages, V4, compact_schema=True)


def test_actual_v7_request_retains_sources_and_full_natural_confirmation(tmp_path):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.engine.memory.preference_dialogue import apply_preference_dialogue, judge_preference_confirmation
    from aria.engine.memory.stated_preferences import confirmed_preference_payloads
    from tests.test_confirm_judge_v7 import Judge, fixture

    utterance = "  就按这个范围记住。\n今天还有别的事情想聊。  "
    profile, confirmation_ctx, raw, _ = fixture(utterance)
    store = PersonaStore(tmp_path)
    judge_preference_confirmation(confirmation_ctx, profile, utterance, client=Judge("confirm"))
    apply_preference_dialogue(confirmation_ctx, profile, utterance, store=store, response=raw)
    profile = store.load(profile.user_id)
    expected, = confirmed_preference_payloads(profile)
    messages, ctx = final_messages(profile, version=V7, compact=True)
    row, = json.loads(messages[-1]["content"])["confirmed_requirements"]
    assert set(row) == {"id", "do", "scope", "not_implies", "quotes", "confirmation"}
    assert row == {key: expected[key] for key in row}
    assert row["confirmation"].endswith(utterance)
    assert row["quotes"] and row["quotes"][0].endswith(profile.stated_preferences[0].quotes[0].text)
    assert ctx.working_memory["prompt_evidence_snapshot"]["stated_preferences_prompt_projection"] == [row]
    for field, replacement in (("quotes", ["altered source"]), ("confirmation", "altered confirmation")):
        with pytest.raises(ValueError, match="invalid_prepared_online_input"):
            prepare_json_messages(messages, V7, compact_schema=True,
                                  confirmed_requirements=[dict(row, **{field: replacement})])
    transition_preference(profile, profile.stated_preferences[0].id, "withdraw")
    store.save(profile)
    after, after_ctx = final_messages(store.load(profile.user_id), version=V7, compact=True)
    assert "confirmed_requirements" not in json.loads(after[-1]["content"])
    assert not after_ctx.working_memory["prompt_evidence_snapshot"].get("stated_preferences_prompt_projection")


@pytest.mark.parametrize("qualification", ["usable", "needs_review", "rule", "source_private", "confirmation_private"])
def test_current_prompt_and_review_keep_confirmation_separate_from_usage(tmp_path, qualification):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.engine.model.persona import PreferenceQuote
    from aria.engine.policy.semantic_delivery import delivery_review_facts

    profile = confirmed_profile()
    preference = profile.stated_preferences[0]
    private = "报名前先问我。我的验证码是 482913。"
    if qualification == "needs_review":
        preference.semantic_review = None
    elif qualification == "rule":
        preference.extraction_source = "rule"
    elif qualification == "source_private":
        preference.quotes[0] = preference.quotes[0].model_copy(update={"text": private})
        preference.scope = "；".join(quote.text for quote in preference.quotes)
    elif qualification == "confirmation_private":
        preference.confirmation = PreferenceQuote(session_id="owner-confirmation", turn_index=0,
                                                  text="同意按原范围记。" + private)
    store = PersonaStore(tmp_path)
    store.save(profile)
    profile = store.load(profile.user_id)
    before = profile.model_dump()
    messages, ctx = final_messages(profile, version=V7, compact=True)
    usable = qualification == "usable"
    wire = "".join(row["content"] for row in messages)
    requirements = json.loads(messages[-1]["content"]).get("confirmed_requirements", [])
    assert len(requirements) == (1 if usable else 0)
    facts = delivery_review_facts(ctx, "继续聊聊。", profile)
    record, = facts["preference_records"]
    assert record["status"] == "confirmed" and record["applicable"] is usable
    if usable:
        assert record["owner_sources"] == [quote.text for quote in preference.quotes]
        assert '"applicable":true' in wire and '"requirement_id"' in wire
        assert "不是其他事实、他人事件或主动回忆的引用依据" in wire
        assert all(quote.text not in facts["owner_sources"] for quote in preference.quotes)
    else:
        assert "owner_sources" not in record and "confirmation" not in record
        assert preference.do not in record["description"] and preference.scope not in record["description"]
        assert '"applicable":false' in wire and '"requirement_id"' not in wire
        assert preference.scope not in wire and private not in wire
    assert profile.model_dump() == before
    ctx.user_id = "a-different-owner"
    assert delivery_review_facts(ctx, "继续聊聊。", profile)["preference_records"] == []


def test_private_confirmation_does_not_remove_another_qualified_preference(tmp_path):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.engine.model.persona import PreferenceQuote
    from aria.engine.policy.semantic_delivery import delivery_review_facts

    profile = confirmed_profile()
    valid_id = profile.stated_preferences[0].id
    second = confirmed_profile(source_session="another-owner-source", source_characters=("丁", "戊", "己"))
    private_row = second.stated_preferences[0]
    private_row.confirmation = PreferenceQuote(session_id="confirmation", turn_index=0,
                                              text="按这个范围记住。我的验证码是 482913。")
    profile.stated_preferences.append(private_row)
    store = PersonaStore(tmp_path)
    store.save(profile)
    profile = store.load(profile.user_id)
    messages, ctx = final_messages(profile, version=V7, compact=True)
    requirement, = json.loads(messages[-1]["content"])["confirmed_requirements"]
    assert requirement["id"] == valid_id
    records = delivery_review_facts(ctx, "继续聊聊。", profile)["preference_records"]
    assert len(records) == 2 and sum(row["applicable"] for row in records) == 1
    assert "482913" not in json.dumps(messages, ensure_ascii=False)
    assert "482913" not in json.dumps(records, ensure_ascii=False)
    assert all(row.status == "confirmed" for row in store.load(profile.user_id).stated_preferences)


def test_general_untrusted_cleanup_still_removes_controls_and_replaces_angle_brackets():
    controls = "".join(chr(index) for index in [*range(0, 9), 11, 12, *range(14, 32), 127])
    assert _clean_untrusted(controls + "<data>\t\n\r</data>") == "＜data＞\t\n\r＜/data＞"
    assert _clean_untrusted(controls + "甲" * 501 + "<tail>") == "甲" * 500


def test_general_untrusted_depth_boundary_stays_at_four():
    assert _clean_untrusted("kept", depth=3) == "kept"
    assert _clean_untrusted("dropped", depth=4) == "[truncated]"
    assert _clean_untrusted({"a": {"b": {"c": {"d": "dropped"}}}}) == {
        "a": {"b": {"c": {"d": "[truncated]"}}}}


def test_general_untrusted_list_and_object_counts_stay_at_thirty():
    assert _clean_untrusted(list(range(31))) == list(range(30))
    assert _clean_untrusted(tuple(range(31))) == list(range(30))
    source = {str(index): index for index in range(31)}
    assert _clean_untrusted(source) == {str(index): index for index in range(30)}
    assert source == {str(index): index for index in range(31)}
    assert _clean_untrusted({"k" * 81: "value"}) == {"k" * 80: "value"}
