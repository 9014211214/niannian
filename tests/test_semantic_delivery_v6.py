"""Forced-label software tests; no test below proves model classification."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.memory.preference_dialogue import bind_preference_request, apply_preference_dialogue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.policy.online_decision import apply_single_pass_decision, online_decision_tool_spec, single_pass_messages
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.policy.semantic_delivery import VERSION, SEGMENT_INSTRUCTION
from aria.engine.policy.prompt import build_messages
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from tests.test_preference_dialogue_readiness import fixture
from tests.test_safety_evidence import payload


def segment(text, kind="conversation", source_quote="", action="none", target="", state_ref=""):
    return dict(text=text, kind=kind, source_quote=source_quote, action=action, target=target, state_ref=state_ref)


def context():
    persona = PersonaProfile(user_id="semantic-test")
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="fixed")
    ctx.working_memory.update(_online_decision_version=VERSION,
        _safety_delivery_version="minimal_response_v1", _safety_response_instruction_version="compact_response_v1")
    return persona, ctx


def decode(persona, ctx, utterance, rows, **extras):
    value = payload(response="".join(row["text"] for row in rows))
    value.update(applied_preference_ids=[], response_segments=rows)
    value.update(extras)
    return apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, utterance, persona)


def deliver(persona, ctx, utterance, rows, **extras):
    response = decode(persona, ctx, utterance, rows, **extras)
    return guard_external_delivery_response(ctx, response, utterance, persona)


def test_ordinary_emotional_acknowledgement_is_not_a_receipt_claim():
    p, c = context()
    text = "她这份心意，您收到了就好。"
    assert deliver(p, c, "她劝我早休息。", [segment(text)]) == text
    assert c.working_memory["delivery_response_guard"]["semantic_verified"] is False


@pytest.mark.parametrize("text", ["我记住了。", "这条已经撤回了。", "以后都按这条相处说明来。"])
def test_uninvited_model_preference_states_removed_without_losing_normal_body(text):
    p, c = context()
    assert deliver(p, c, "现在先聊木工。", [segment(text, "preference_state"),
        segment("你可以接着说。")]) == "你可以接着说。"
    assert c.working_memory.get("preference_dialogue_receipt") is None


def test_wrong_conversation_label_is_a_retained_and_explicit_semantic_risk():
    p, c = context()
    bad = "我记住了，以后都按这个办。"
    assert deliver(p, c, "还没同意保存。", [segment(bad)]) == bad
    assert c.working_memory.get("preference_dialogue_receipt") is None
    assert c.working_memory["delivery_response_guard"]["semantic_verified"] is False


@pytest.mark.parametrize("has_source", [False, True])
def test_third_party_claim_needs_exact_owner_source_without_replacing_other_sentences(has_source):
    p, c = context()
    user = "女儿刚才回复了我的短信。"
    out = deliver(p, c, user, [segment("她回复了。", "third_party_fact", user if has_source else ""),
                                segment("听你这么说，我明白了。")])
    assert out == ("她回复了。" if has_source else "") + "听你这么说，我明白了。"


@pytest.mark.parametrize("status,target,accepted", [("pending", "daughter", False),
    ("unknown", "daughter", False), ("delivered", "wrong", False), ("delivered", "daughter", True)])
def test_receipt_status_and_target_do_not_prove_recipient_reading(status, target, accepted):
    p, c = context()
    c.working_memory["last_tool_result"] = {"ok": True, "tool": "notify_family",
        "args": {"contact": "daughter"}, "data": {"status": status}}
    from aria.engine.policy.semantic_delivery import bind_current_action_receipt
    bind_current_action_receipt(c, "请核对状态。", c.working_memory["last_tool_result"])
    out = deliver(p, c, "请核对状态。", [segment("我已经通知，她已经读到了。", "assistant_action",
                                                   action="notification", target=target), segment("我们继续。")])
    assert "她已经读到了" not in out
    assert ("成功回执" in out) is accepted
    assert out.endswith("我们继续。")


@pytest.mark.parametrize("mutation", ["unbound", "old_turn", "other_input", "changed_content"])
def test_previous_or_mutated_receipt_cannot_support_fresh_completion(mutation):
    from aria.engine.policy.semantic_delivery import bind_current_action_receipt
    p, c = context()
    receipt = {"ok": True, "tool": "notify_family", "args": {"contact": "daughter", "message": "old"},
               "data": {"status": "delivered"}}
    c.working_memory["last_tool_result"] = receipt
    if mutation != "unbound":
        bind_current_action_receipt(c, "本轮要求", receipt)
    if mutation == "old_turn":
        c.working_memory["_semantic_current_action_receipt"]["turn_index"] = -1
    if mutation == "changed_content":
        receipt["args"]["message"] = "new"
    out = deliver(p, c, "另一要求" if mutation == "other_input" else "本轮要求",
                  [segment("新的消息我已经发给她了。", "assistant_action", action="notification", target="daughter"),
                   segment("我们继续。")])
    assert out == "我们继续。"


def test_foreign_source_or_stale_turn_cannot_authorize_delivery():
    p, c = context()
    rows = [segment("她回复了。", "third_party_fact", "她回复了。")]
    body = decode(p, c, "她回复了。", rows)
    c.working_memory["online_turn_decision"]["turn_index"] += 1
    assert guard_external_delivery_response(c, body, "她回复了。", p) != body


@pytest.mark.parametrize("field", ["user_id", "scene_id", "session_id"])
def test_trace_cannot_move_to_another_identity(field):
    p, c = context()
    body = decode(p, c, "当前原话。", [segment("原始正文。")])
    setattr(c, field, "another")
    assert guard_external_delivery_response(c, body, "当前原话。", p) != body


def test_mandatory_segments_and_join_fail_before_any_state_action():
    p, c = context()
    value = payload(response="普通正文。")
    value["applied_preference_ids"] = []
    with pytest.raises(ValueError):
        parse_json_decision(json.dumps(value), VERSION)
    value["response_segments"] = [segment("不同正文。")]
    apply_single_pass_decision(json.dumps(value), c, "当前原话。", p)
    assert c.working_memory["online_turn_decision"]["status"] == "invalid_protocol_response"


def test_v5_schema_retains_optional_legacy_boolean_segments():
    schema = online_decision_tool_spec("online_turn_decision_v5_bound")["function"]["parameters"]
    assert "response_segments" not in schema["required"]
    assert set(schema["properties"]["response_segments"]["items"]["properties"]) == {"text", "claims_saved"}


def test_applied_id_reports_are_separate_from_supplied_verified_ids():
    from tests.test_stated_preferences import _profile
    p = _profile()
    _, c = context()
    c.user_id = p.user_id
    build_messages(get_scene("companion_elder"), p, c, "当前原话。",
                   safety_response_instruction_version="compact_response_v1")
    actual = p.stated_preferences[0].id
    decode(p, c, "当前原话。", [segment("明白。")], applied_preference_ids=["invented", actual])
    trace = c.working_memory["online_turn_decision"]
    assert trace["applied_preference_ids"] == [actual]
    assert trace["rejected_applied_preference_ids"] == ["invented"]
    assert trace["applied_preference_ids_semantically_verified"] is False
    p.stated_preferences[0].status = "withdrawn"
    decode(p, c, "当前原话。", [segment("明白。")], applied_preference_ids=[actual])
    assert c.working_memory["online_turn_decision"]["applied_preference_ids"] == []


@pytest.mark.parametrize("action,status", [("confirm", "confirmed"), ("decline", "declined"), ("unclear", "proposed")])
def test_actual_commit_composition_retains_normal_body_and_code_receipt(tmp_path, action, status):
    p, c = fixture()
    c.working_memory.update(_online_decision_version=VERSION,
        _safety_delivery_version="minimal_response_v1", _safety_response_instruction_version="compact_response_v1")
    user = "本人当前回答"
    bind_preference_request(p, c, user)
    # Use the exact code success phrase too: removing the model's copy must not
    # remove the later identical, genuinely persisted code receipt.
    rows = [segment("我记下了，按你刚才确认的范围相处。以后也可以改或撤回。", "preference_state"), segment("我们继续聊。")]
    body = decode(p, c, user, rows, preference_control={"action": action})
    composed = apply_preference_dialogue(c, p, user, store=PersonaStore(tmp_path), response=body)
    out = guard_external_delivery_response(c, composed, user, p)
    assert out.startswith("我们继续聊。\n")
    assert p.stated_preferences[0].status == status
    if action == "confirm":
        assert "我记下了" in out and PersonaStore(tmp_path).load(p.user_id).stated_preferences[0].status == "confirmed"
        assert apply_preference_dialogue(c, p, user, store=PersonaStore(tmp_path), response=body) is None
        # A repeat that returns no composed body grants no exemption to raw text.
        assert "preference_composed_response_sha256" not in c.working_memory["online_turn_decision"]
        assert guard_external_delivery_response(c, body, user, p) == "我们继续聊。"
    else:
        assert "我记下了" not in out


def test_compact_prompt_has_one_semantic_contract_and_stays_bounded():
    p, c = context()
    base = build_messages(get_scene("companion_elder"), p, c, "当前输入。",
                          safety_response_instruction_version="compact_response_v1")
    messages = single_pass_messages(base, c, version=VERSION, persona=p, format_style="schema_guided")
    system = messages[0]["content"]
    assert system.count(SEGMENT_INSTRUCTION) == 1
    assert len(system.encode("utf-8")) <= 12000


def test_explicit_candidate_assembly_keeps_original_models_and_default_v5():
    from aria.release import create_candidate_clients, create_semantic_candidate_clients
    from tests.test_online_structured import route
    options = dict(model_calls_enabled=True, api_key="inert-software-key",
                   base_url="https://fixture.invalid/v1", online_structured_route=route())
    old, new = create_candidate_clients(**options), create_semantic_candidate_clients(**options)
    assert old.online.online_decision_version == "online_turn_decision_v5_bound"
    assert new.online.online_decision_version == VERSION
    for role in ("online", "memory", "reflection", "safety_review"):
        assert getattr(new, role).model == getattr(old, role).model
        assert getattr(new, role).network_request_count == 0


def test_safety_review_source_and_deletion_path_accepts_v6_without_extra_call(monkeypatch):
    import tests.test_safety_review_v1 as old
    from tests.test_safety_evidence import evidence
    monkeypatch.setattr(old, "ONLINE_VERSION", VERSION)
    value = payload("fall", evidence(old.TEXT, "owner", "current", "immediate"), old.BODY)
    value["response_segments"] = [segment("保留原句甲。\n"), segment("删除原句乙。"), segment("保留原句丙。")]
    reviewer = old.Reviewer({"conflicts": True, "delete_sentence_ids": [1]})
    engine, ctx, calls = old.build(monkeypatch, value, reviewer)
    events = []
    result = engine.process_turn(ctx, old.TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert len(calls) == len(reviewer.messages) == len(events) == 1
    assert result.startswith(events[0]["protection"])
    assert "保留原句甲。" in result and "保留原句丙。" in result and "删除原句乙。" not in result


def test_offline_full_source_preference_and_job_provenance_accepts_v6():
    from tests.test_memory_prefix_recovery import setup, empty, preference
    from aria.engine.memory.consolidator import MemoryConsolidator
    from aria.engine.model.persona import OfflineSessionRecord
    p, c, client, extractor = setup()
    c.working_memory["_online_decision_version"] = VERSION
    value = empty()
    value["stated_preferences"] = [preference(c)]
    client.payload = value
    MemoryConsolidator(extractor=extractor).consolidate(p, c, persist=False)
    assert p.stated_preferences[0].status == "proposed"
    assert c.working_memory["_preference_scope_selection"]["protocol"] == "preference_scope_full_source_v1"
    assert VERSION in str(OfflineSessionRecord.model_fields["origin_online_decision_version"].annotation)


@pytest.mark.parametrize("action", ["confirm", "decline", "withdraw"])
@pytest.mark.parametrize("persisted", [True, False])
def test_v7_review_preserves_native_ui_result_on_its_verified_record(tmp_path, action, persisted):
    from tests.test_minimal_web import live_fixture
    from tests.test_stated_preferences import _profile
    from aria.engine.memory.preference_status import current_preference_states
    from aria.engine.policy.semantic_delivery import V7, delivery_review_facts

    app, clients = live_fixture(tmp_path)
    clients.online.online_decision_version = V7
    profile = _profile(confirmed=action == "withdraw")
    profile.user_id = "fictional-chen"
    app.store.save(profile)
    app.start()
    temporary = app.store._path(profile.user_id).with_suffix(".json.tmp")
    if not persisted:
        temporary.mkdir()  # Exercise the actual native write failure.
    try:
        assert app.preference_action(action, profile.stated_preferences[0].id)["ok"] is persisted
        current = app._profile()
        _, ctx = app._sessions[profile.user_id]
        # No online turn is sent here; normally Engine sets this at turn entry.
        ctx.working_memory["_online_decision_version"] = V7
        original = current.model_dump()
        state, = current_preference_states(current)
        facts = delivery_review_facts(ctx, "核对当前记录。", current)
        record, = facts["preference_records"]
        assert record["status"] == state["status"]
        assert record["last_ui_result"] == {key: state["last_ui_result"][key]
                                             for key in ("action", "persisted")}
        assert record["last_ui_result"] == {"action": action, "persisted": persisted}
        assert not {"state_ref", "target_id", "source"}.intersection(record)
        assert facts["current_action_results"] == [] and facts["code_status"] is None
        assert current.model_dump() == original
        if not persisted:
            assert "保存失败" in record["description"]
        if record["status"] != "confirmed":
            assert record["applicable"] is False
            assert not {"do", "scope", "owner_sources", "confirmation"}.intersection(record)

        # The explicitly selected legacy v6 projection keeps its original shape.
        _, legacy = context()
        legacy.user_id = current.user_id
        old_record, = delivery_review_facts(legacy, "核对当前记录。", current)["preference_records"]
        assert old_record == {key: value for key, value in record.items()
                              if key != "last_ui_result"}
    finally:
        if not persisted:
            temporary.rmdir()
        app.close()


@pytest.mark.parametrize("defect", ["none", "foreign_owner", "changed_profile", "event_owner",
    "event_target", "event_rows", "event_after", "reload"])
def test_v7_review_never_rebinds_stale_or_foreign_ui_result(tmp_path, defect):
    from tests.test_stated_preferences import _profile
    from aria.engine.memory.stated_preferences import transition_preference
    from aria.engine.memory.preference_status import record_preference_ui_result, current_preference_states
    from aria.engine.policy.semantic_delivery import V7, delivery_review_facts

    profile = _profile(confirmed=True)
    sibling = profile.stated_preferences[0].model_copy(deep=True, update={"id": "pref_other_record"})
    profile.stated_preferences.append(sibling)
    store = PersonaStore(tmp_path)
    store.save(profile)
    before = deepcopy(profile)
    target = profile.stated_preferences[0].id
    transition_preference(profile, target, "withdraw")
    store.save(profile)
    _, ctx = context()
    ctx.user_id = profile.user_id
    ctx.working_memory["_online_decision_version"] = V7
    record_preference_ui_result(profile, before, target, "withdraw", persisted=True, ctx=ctx)
    if defect == "foreign_owner":
        ctx.user_id = "someone-else"
    elif defect == "changed_profile":
        profile.stated_preferences[0].scope += "未保存的改变"
    elif defect.startswith("event_"):
        field = {"event_owner": "owner_id", "event_target": "target_id",
                 "event_rows": "rows_sha256", "event_after": "after_sha256"}[defect]
        profile._preference_ui_result[field] = "not-current"
    elif defect == "reload":
        profile = store.load(profile.user_id)
    facts = delivery_review_facts(ctx, "核对当前记录。", profile)
    if defect in {"foreign_owner", "changed_profile"}:
        assert facts["preference_records"] == []
    else:
        # Match native array positions for this software check; no opaque ID
        # is sent to the reviewer as a substitute for semantic association.
        states = current_preference_states(profile)
        records = {state["target_id"]: record for state, record in zip(states, facts["preference_records"], strict=True)}
        assert set(records) == {target, sibling.id}
        assert "last_ui_result" not in records[sibling.id]
        assert ("last_ui_result" in records[target]) is (defect == "none")
        assert records[target]["status"] == "withdrawn"
        assert records[sibling.id]["status"] == "confirmed"
        assert all(not {"state_ref", "target_id", "source"}.intersection(record) for record in records.values())
    assert facts["current_action_results"] == []
