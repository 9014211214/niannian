"""Forced applicability labels verify mechanics, never semantic accuracy."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from aria.engine.risk.safety_review_v2 import parse_review, safety_review_messages
from aria.engine.policy.safety_structured import safety_request_format, validate_safety_review_request
from aria.engine.risk.safety_delivery import render_review_protection
from aria.engine.risk.premodel_task import DeliveredSafetyPrelude, make_prelude
from tests.test_safety_review_v1 import BODY, TEXT, Reviewer, build, no_network
from tests.test_safety_evidence import payload, evidence


def assessment(concern="none", **kwargs):
    return {"concern": concern, "subject": "uncertain", "time": "uncertain", "urgency": "uncertain", **kwargs}


def answer(concern="none", *, delete=(), wire=None, **kwargs):
    risk = assessment(concern, **kwargs)
    if wire is not None and concern == "none":
        props = wire.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        schema = props.get("risk_assessment", {})
        if schema.get("type") == "array":
            risk = []
            if "keep_each_span" in props:
                spans = json.loads(wire["messages"][1]["content"])["sentences"]
                return {"risk_assessment": risk, "keep_each_span": [span["id"] not in delete for span in spans]}
    return {"risk_assessment": risk, "conflicts": bool(delete), "delete_sentence_ids": list(delete)}


def engine_v2(monkeypatch, reviewer, *, label="fall", body=BODY):
    engine, ctx, calls = build(monkeypatch, payload(label, evidence(TEXT, "owner", "current", "immediate"), body), reviewer)
    engine.safety_review_version = "safety_review_v2"
    return engine, ctx, calls


@pytest.mark.parametrize("concern,source", [
    ("fall", "我跌倒了"), ("medical_emergency", "现在呼吸困难"),
    ("self_harm", "我怕自己会伤害自己"), ("unsafe_medication", "我刚才多服了一次"),
])
def test_sourced_concrete_risk_can_correct_online_none_from_an_eligible_owner_turn(concern, source):
    parsed_answer = answer(concern, delete=[1], subject="owner", time="current", urgency="immediate",
                           owner_source_turns=[1], owner_risk_quote=source)
    row = parse_review(json.dumps(parsed_answer), 3, allow_partial=True, owner_texts=["现在还是这样。", source+"。"])
    assert row["risk_assessment"] == assessment(concern, subject="owner", time="current", urgency="immediate")
    assert row["owner_source_turns"] == [1] and row["delete_sentence_ids"] == [1]
    assert render_review_protection(row["risk_assessment"], reviewed_fields=True)[0]
    parsed_answer["risk_assessment"]["owner_source_turns"] = [0]
    unbound = parse_review(json.dumps(parsed_answer), 3, allow_partial=True, owner_texts=["现在还是这样。", source+"。"])
    assert unbound["unbound_risk"] and unbound["risk_assessment"]["concern"] == "uncertain"
    assert unbound["delete_sentence_ids"] == [1] and "deletion_error" not in unbound


def test_other_real_danger_is_bound_to_the_current_owner_source_contract_only():
    current = safety_request_format("safety_review_v2", owner_sources=True)["json_schema"]["schema"]
    report = current["properties"]["risk_assessment"]
    assert report["type"] == "array" and "anyOf" not in report
    schema = report["items"]
    assert "other_danger" in schema["properties"]["concern"]["enum"]
    assert list(schema["properties"]) == schema["required"] == ["owner_source_id", "concern", "subject", "time", "urgency"]
    legacy = safety_request_format("safety_review_v2")["json_schema"]["schema"]["properties"]["risk_assessment"]
    assert "other_danger" not in legacy["properties"]["concern"]["enum"]
    assert "owner_risk_quote" not in legacy["properties"]


@pytest.mark.parametrize("concern", ["fall", "medical_emergency", "self_harm", "unsafe_medication", "other_danger", "uncertain"])
def test_risk_list_can_report_a_sourced_owner_or_third_party_risk(concern):
    source = "他现在还被困在车里"
    risk = answer(concern, subject="other", time="current", urgency="immediate", owner_source_id="t1.s0")
    risk["risk_assessment"] = [risk["risk_assessment"]]
    row = parse_review(json.dumps(risk), 2, allow_partial=True, owner_texts=["是的。", source], clear_online=True)
    assert row["risk_assessment"]["concern"] == concern and not row.get("unbound_risk")
    assert row["owner_source_turns"] == [1] and row["owner_source_id"] == "t1.s0"
    if concern != "uncertain":
        assert row["risk_assessment"]["subject"] == "other"


def test_source_catalog_preserves_history_and_long_turns_and_rejects_wire_tampering():
    from aria.engine.risk.safety_review_v2 import owner_source_catalog
    current, previous = "这里有人。Please help. Stay there!", "x" * 723
    evidence_row = {"concern": "none", "subject": None, "time": None, "urgency": None,
                    "quote": None, "verbatim_source": False, "disposition": None, "semantic_verified": False,
                    "conversation_history": [{"role": "user", "content": "Earlier."}, {"role": "user", "content": previous},
                                             {"role": "assistant", "content": "旁白"}]}
    messages = safety_review_messages(current, BODY, evidence_row, None, final_protection="", owner_sources=True)
    assert validate_safety_review_request(messages) == 3
    payload = json.loads(messages[1]["content"])
    catalog = payload["owner_sources"]
    assert catalog == owner_source_catalog([current, previous, "Earlier."], compact=True)
    assert catalog["t2.s0"] is None and owner_source_catalog([current, previous, "Earlier."])["t2.s0"] == "Earlier."
    assert list(catalog.values())[:3] == ["这里有人。", "Please help.", " Stay there!"]
    assert "".join(catalog[f"t1.s{i}"] for i in range(4)) == previous
    assert max(len(s) for s in catalog.values() if s is not None) == 240 and "旁白" not in catalog.values()
    catalog["t1.s0"] = "旁白"
    messages[1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="invalid_safety_review_owner_sources"):
        validate_safety_review_request(messages)


@pytest.mark.parametrize("shown", [None, "已展示的条件保护。"])
def test_native_review_only_sees_observed_context_and_displayed_protection(shown):
    original = {"concern": "self_harm", "subject": "other", "time": "current", "urgency": "immediate",
                "quote": "旧在线判断抄录。", "verbatim_source": False, "disposition": "previous_guess",
                "semantic_verified": False, "conversation_history": [{"role": "user", "content": "先前本人原话。"}],
                "delivery_facts": {"owner_sources": [], "preference_records": [],
                                   "current_action_results": [], "code_status": None}}
    unshown = "尚未展示的旧类别模板。"
    messages = safety_review_messages(TEXT, BODY, original, shown, final_protection=unshown, owner_sources=True)
    assert validate_safety_review_request(messages) == 3
    data = json.loads(messages[1]["content"])
    assert data["safety_evidence"] == {key: original[key] for key in ("conversation_history", "delivery_facts")}
    assert data["candidate"] == BODY and data["utterance"] == TEXT and data["protection_already_delivered"] == shown
    assert data["owner_sources"]["t1.s0"] is None and "先前本人原话。" in messages[1]["content"]
    assert "final_protection" not in data and unshown not in messages[1]["content"]
    legacy = safety_review_messages(TEXT, BODY, original, shown, final_protection=unshown)
    earlier = json.loads(legacy[1]["content"])
    assert earlier["safety_evidence"] == original and earlier["final_protection"] == unshown
    assert validate_safety_review_request(legacy) == 3
    for key, value in (("concern", "none"), ("subject", "other"), ("quote", "旧在线判断抄录。"),
                       ("final_protection", unshown)):
        injected = deepcopy(data)
        (injected if key == "final_protection" else injected["safety_evidence"])[key] = value
        messages[1]["content"] = json.dumps(injected)
        with pytest.raises(ValueError):
            validate_safety_review_request(messages)


@pytest.mark.parametrize("choice", [
    {"owner_source_id": True}, {"owner_source_id": None}, {"owner_source_id": "t2.s0"},
    {"owner_source_id": "t0.s2"}, {"owner_source_id": "t0.s0", "owner_source_turns": [0], "owner_risk_quote": "我受伤了。"},
])
def test_unbound_or_mixed_source_id_cannot_validate_an_immediate_alert(choice):
    risk = answer("unsafe_medication", subject="owner", time="current", urgency="immediate", **choice)
    risk["risk_assessment"] = [risk["risk_assessment"]]
    parsed = parse_review(json.dumps(risk), 2, allow_partial=True, owner_texts=["我受伤了。"], clear_online=True)
    assert parsed["unbound_risk"] and parsed["risk_assessment"]["concern"] == "uncertain"
    assert "owner_source_id" not in parsed and "owner_source_turns" not in parsed


def test_inflight_change_to_the_full_source_behind_a_whole_turn_reference_withholds_body(monkeypatch):
    from aria.engine.risk import safety_review_v2
    reviewer = Reviewer()
    monkeypatch.setattr(safety_review_v2, "uses_owner_sources", lambda client: True)
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)

    def swap_owner(messages):
        data = json.loads(messages[1]["content"])
        assert data["owner_sources"]["t0.s0"] is None
        data["utterance"] = "别人的输入。"
        messages[1]["content"] = json.dumps(data, ensure_ascii=False)
        risk = assessment("fall", subject="owner", time="current", urgency="immediate", owner_source_id="t0.s0")
        return json.dumps({"risk_assessment": [risk], "conflicts": False, "delete_sentence_ids": []})

    monkeypatch.setattr(reviewer, "complete", swap_owner)
    assert BODY not in engine.process_turn(ctx, TEXT)
    assert ctx.working_memory["safety_review"]["status"] == "source_changed"


@pytest.mark.parametrize("reported", [[], [{}], ["no_owner_signal"], [{}, {}], None])
def test_only_empty_list_is_no_signal_and_body_deletion_is_independent(reported):
    raw = {"risk_assessment": reported, "conflicts": True, "delete_sentence_ids": [1]}
    row = parse_review(json.dumps(raw), 2, allow_partial=True, owner_texts=["软件原话。"], clear_online=True)
    assert row["risk_assessment"]["concern"] == ("none" if reported == [] else "uncertain")
    assert bool(row.get("unbound_risk")) is (reported != [])
    assert row["delete_sentence_ids"] == [1] and "deletion_error" not in row


@pytest.mark.parametrize("risk", ["no_owner_signal", {"concern": "none"}, [
    assessment("unsafe_medication", subject="owner", time="current", urgency="immediate",
               owner_source_turns=[0], owner_risk_quote="软件原话。")]])
def test_current_span_clearance_cannot_use_a_historical_risk_shape(risk):
    row = parse_review(json.dumps({"risk_assessment": risk, "keep_each_span": [True]}), 1,
                       allow_partial=True, require_span_verdicts=True, owner_texts=["软件原话。"])
    assert row["risk_assessment"]["concern"] == "uncertain" and row["unbound_risk"]
    assert row["span_review_complete"] and row["delete_sentence_ids"] == []


def test_native_period_units_preserve_bytes_and_existing_maximum_capacity():
    from aria.engine.risk.safety_review_v2 import MAX_SENTENCES, response_sentences

    text = "Rate 3.5 at example.org. Keep this."
    assert response_sentences(text) == (text,)
    assert response_sentences(text, split_periods=True) == ("Rate 3.5 at example.org.", " Keep this.")
    bounded = "One. Two," * MAX_SENTENCES
    assert len(response_sentences(bounded)) == MAX_SENTENCES
    assert response_sentences(bounded, split_periods=True) == response_sentences(bounded)
    assert "".join(response_sentences(bounded, split_periods=True)) == bounded


def test_maximum_span_review_fits_native_completion_limit_without_repeating_ids():
    from aria.engine.risk.safety_review import MAX_SENTENCES
    risk = assessment("medical_emergency", subject="uncertain", time="recent_resolved",
                      urgency="non_immediate", owner_source_id="t10.s1000")
    result = {"risk_assessment": [risk], "keep_each_span": [False] * MAX_SENTENCES}
    raw = json.dumps(result, separators=(",", ":"))
    assert len(raw.encode("ascii")) < 1024
    row = parse_review(raw, MAX_SENTENCES, allow_partial=True, require_span_verdicts=True, owner_texts=["软件原话。"])
    assert row["span_review_complete"] and row["delete_sentence_ids"] == list(range(MAX_SENTENCES))


@pytest.mark.parametrize("change,clear_online,bad_deletion,applied", [
    ({}, True, False, "none"),
    ({"owner_source_turns": [0], "owner_risk_quote": "别轮才有的文字"}, True, False, "none"),
    ({"owner_source_turns": [0], "owner_risk_quote": "我多吃了一份药"}, True, False, "unsafe_medication"),
    ({"owner_source_turns": [3]}, True, False, "uncertain"),
    ({"owner_risk_quote": None}, True, False, "uncertain"),
    ({"urgency": "immediate"}, True, False, "uncertain"),
    ({"urgency": "uncertain"}, True, False, "uncertain"),
    ({"subject": None}, True, False, "uncertain"),
    ({}, False, False, "uncertain"),
    ({}, True, True, "uncertain"),
])
def test_unsupported_nonurgent_alert_needs_clear_online_and_valid_independent_body_review(
        change, clear_online, bad_deletion, applied):
    review = answer("unsafe_medication", subject="other", time="current", urgency="non_immediate",
                    owner_source_turns=[], owner_risk_quote="")
    review["risk_assessment"].update(change)
    if bad_deletion:
        review["delete_sentence_ids"] = [2]
    row = parse_review(json.dumps(review), 2, allow_partial=True, owner_texts=["我多吃了一份药。"],
                       clear_online=clear_online)
    assert row["risk_assessment"]["concern"] == applied
    assert ("deletion_error" in row) == bad_deletion
    assert bool(row.get("unbound_risk")) == (applied != "unsafe_medication")


@pytest.mark.parametrize("urgency,quote", [
    ("immediate", ""), ("immediate", "没有对应的引文"), ("non_immediate", "没有对应的引文"),
])
def test_contradictory_uncertain_alert_does_not_certify_empty_owner_source(urgency, quote):
    review = answer("uncertain", subject="owner", time="current", urgency=urgency,
                    owner_source_turns=[], owner_risk_quote=quote)
    row = parse_review(json.dumps(review), 2, allow_partial=True, owner_texts=["本人原话。"], clear_online=True)
    assert row["risk_assessment"]["concern"] == "uncertain"
    assert "owner_source_turns" not in row


@pytest.mark.parametrize("label", ["fall", "medical_emergency", "self_harm", "unsafe_medication"])
def test_model_only_none_keeps_body_without_ever_displaying_wrong_template(monkeypatch, label):
    reviewer = Reviewer(answer())
    engine, ctx, calls = engine_v2(monkeypatch, reviewer, label=label)
    events = []
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert result == BODY and events == [] and len(calls) == len(reviewer.messages) == 1
    request = json.loads(reviewer.messages[0][1]["content"])
    assert request["version"] == "safety_review_v2" and request["protection_already_delivered"] is None
    audit = ctx.working_memory["safety_review"]
    assert audit["risk_assessment"]["concern"] == "none" and audit["semantic_verified"] is False
    assert audit["grants_permission"] is False and audit["model_body_delivered"] is True
    assert audit["final_protection_sha256"] is None
    assert ctx.working_memory["safety_delivery"]["segment_ids"] == []


def test_no_applicable_template_still_deletes_other_danger_and_does_not_execute(monkeypatch):
    reviewer = Reviewer(answer(delete=[1]))
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    result = engine.process_turn(ctx, TEXT)
    assert result == "保留原句甲。保留原句丙。"
    assert ctx.working_memory["safety_review"]["conflicts"] is True
    assert not engine.persona.stated_preferences


@pytest.mark.parametrize("concern", ["fall", "medical_emergency", "self_harm", "unsafe_medication"])
def test_explicit_risk_decision_even_without_conflicts_gets_protection_then_body(monkeypatch, concern):
    order, events = [], []
    reviewer = Reviewer(answer(concern, subject="other", time="current", urgency="immediate"), order=order)
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    def emit(event):
        order.append("protection"); events.append(deepcopy(event)); return True
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    expected, ids = render_review_protection(assessment(concern, subject="other", time="current", urgency="immediate"), reviewed_fields=True)
    assert order == ["review", "protection"] and result == BODY
    assert events[0]["protection"] == expected
    assert ctx.working_memory["safety_review"]["prelude_status"] == "delivered_after_review"
    assert ctx.working_memory["safety_delivery"]["segment_ids"] == list(ids)


@pytest.mark.parametrize("invalid,body_kept,protection_count", [
    (TimeoutError("private diagnostic must not be logged"), False, 1),
    ("not-json", False, 1),
    ({"conflicts": False, "delete_sentence_ids": []}, True, 1),
    (answer(delete=[128]), False, 0),
    ({**answer(), "action": "save"}, False, 0),
])
def test_incomplete_review_keeps_independent_body_and_protection(monkeypatch, invalid, body_kept, protection_count):
    reviewer = Reviewer(invalid)
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    events = []
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert len(reviewer.messages) == 1 and len(events) == protection_count and result.strip()
    if events:
        assert events[0]["protection"] not in result
    assert ("保留原句" in result) is body_kept
    assert ctx.working_memory["safety_review"]["model_body_delivered"] is body_kept
    assert "private diagnostic" not in json.dumps(ctx.working_memory["safety_review"])


@pytest.mark.parametrize("bad_callback", ["reject", "mutate", "source"])
def test_callback_binding_failure_never_releases_reviewed_body(monkeypatch, bad_callback):
    reviewer = Reviewer(answer("fall"))
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    def emit(event):
        if bad_callback == "reject": return False
        if bad_callback == "mutate": event["protection"] = "tampered"
        if bad_callback == "source": ctx.session_id = "another"
        return True
    result = engine.process_turn(ctx, TEXT, safety_prelude=emit)
    assert "保留原句" not in result
    assert ctx.working_memory["safety_review"]["model_body_delivered"] is False


def test_source_change_during_review_discards_body_and_assessment(monkeypatch):
    reviewer = Reviewer(answer())
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    reviewer.mutate = lambda: setattr(ctx, "session_id", "changed")
    result = engine.process_turn(ctx, TEXT)
    assert "保留原句" not in result and ctx.working_memory["safety_review"]["status"] == "source_changed"


def test_already_displayed_protection_cannot_be_removed_by_none(monkeypatch):
    from aria.engine.risk.safety_review import current_review_evidence
    from aria.engine.risk.source_response import current_source_response
    reviewer = Reviewer(answer())
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    engine.process_turn(ctx, TEXT)
    # Re-enter only a fresh internal source snapshot; no second model request
    # from production re-entry is permitted. This tests the exact receipt path.
    ctx.turns.clear()
    ctx.working_memory.pop("safety_review", None)
    current = current_review_evidence(ctx, TEXT, persona=engine.persona)
    frozen = current_source_response(ctx, TEXT, engine.persona)
    assert current is not None and frozen is not None
    protection, ids = render_review_protection(current)
    receipt = DeliveredSafetyPrelude(**make_prelude(ctx, TEXT, protection))
    body, delivered, selected = engine._review_source_risk_v2(ctx, TEXT, BODY,
        profile=engine.persona, frozen=frozen, evidence=current, protection=protection,
        segment_ids=ids, safety_prelude=None, delivered_prelude=receipt)
    assert body == BODY and delivered == protection and selected == ("keyword-safety-existing",)
    request = json.loads(reviewer.messages[-1][1]["content"])
    assert request["protection_already_delivered"] == protection


def test_review_empty_body_never_becomes_empty_output(monkeypatch):
    reviewer = Reviewer(answer(delete=[0, 1, 2]))
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    result = engine.process_turn(ctx, TEXT)
    assert result.strip() and "保留原句" not in result


def test_mixed_sentence_can_delete_one_clause_without_losing_adjacent_help(monkeypatch):
    """Forced semantic decision tests the smaller deletion capability only."""
    body = "保留陪伴，删除这个操作，保留求助。"
    engine, ctx, _ = engine_v2(monkeypatch, Reviewer(answer(delete=[1])), body=body)
    result = engine.process_turn(ctx, TEXT)
    assert result == "保留陪伴，保留求助。"
    assert ctx.working_memory["safety_review"]["deletion_unit"] == "punctuation_span_v1"


def test_uncertain_applicability_keeps_separately_reviewed_body(monkeypatch):
    events = []
    engine, ctx, _ = engine_v2(monkeypatch, Reviewer(answer("uncertain", delete=[1])))
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert result == "保留原句甲。保留原句丙。"
    assert len(events) == 1 and events[0]["protection"].startswith("若现实中")
    assert ctx.working_memory["safety_review"]["status"] == "reviewed"


@pytest.mark.parametrize("metadata", [
    assessment(subject="other"), assessment(subject=None, time=17),
    {"concern": "none"}, {**assessment(), "confidence": 1},
])
def test_unused_applicability_fields_do_not_destroy_valid_deletion_review(monkeypatch, metadata):
    original = {**answer(delete=[1]), "risk_assessment": metadata}
    reviewer = Reviewer(original)
    engine, ctx, _ = engine_v2(monkeypatch, reviewer)
    events = []
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert result == "保留原句甲。保留原句丙。" and not events
    audit = ctx.working_memory["safety_review"]
    assert audit["risk_assessment"] == assessment()
    assert audit["dropped_fields"] and audit["semantic_verified"] is False
    assert original["risk_assessment"] == metadata
    assert len(reviewer.messages) == 1


@pytest.mark.parametrize("metadata", [None, [], {}, {"concern": "fictional"},
                                        {"concern": True}])
def test_bad_applicability_is_uncertain_not_safe_and_keeps_reviewed_body(monkeypatch, metadata):
    engine, ctx, _ = engine_v2(monkeypatch,
        Reviewer({**answer(delete=[1]), "risk_assessment": metadata}))
    events = []
    result = engine.process_turn(ctx, TEXT, safety_prelude=lambda event: events.append(event) or True)
    assert result == "保留原句甲。保留原句丙。"
    assert len(events) == 1 and events[0]["protection"].startswith("若现实中")
    audit = ctx.working_memory["safety_review"]
    assert audit["risk_assessment"] == assessment("uncertain")
    assert audit["dropped_fields"] and audit["grants_permission"] is False


@pytest.mark.parametrize("deletion", [
    {"conflicts": False, "delete_sentence_ids": [0]},
    {"conflicts": True, "delete_sentence_ids": []},
    {"conflicts": True, "delete_sentence_ids": [True]},
    {"conflicts": True, "delete_sentence_ids": [0, 0]},
    {"conflicts": True, "delete_sentence_ids": [3]},
])
def test_auxiliary_field_tolerance_never_repairs_invalid_deletion(deletion):
    with pytest.raises(ValueError):
        parse_review(json.dumps({**deletion, "risk_assessment": assessment(subject="other")}), 3)


def test_v2_wire_uses_v2_schema_and_exact_result_with_durable_budget_guard(tmp_path, monkeypatch):
    from tests.test_candidate_role_transport import client, guard, response
    from tests.test_reserved_calls import stub
    c = client("safety_review")
    data = response(c)
    data["choices"][0]["message"]["content"] = json.dumps(answer(delete=[1]))
    calls = stub(monkeypatch, data)
    evidence_row = {"concern": "fall", "subject": "owner", "time": "current", "urgency": "immediate",
        "quote": TEXT, "verbatim_source": True, "disposition": "urgent_owner", "semantic_verified": False,
        "conversation_history": []}
    messages = safety_review_messages(TEXT, BODY, evidence_row, None, final_protection="条件候选。")
    assert validate_safety_review_request(messages) == 3
    with closing(guard(tmp_path / "ledger", c)) as g:
        g.attach(c)
        with g.scope("turn", "op1", "session1"):
            raw = c.complete(messages)
        assert parse_review(raw, 3) == answer(delete=[1])
        row, = g.snapshot()["attempts"]
        assert row["status"] == "received"
    wire, = calls
    from aria.engine.risk.safety_review_v2 import SAFETY_REVIEW_INSTRUCTION
    assert wire["messages"][0] == {"role": "system", "content": SAFETY_REVIEW_INSTRUCTION}
    assert "逐句独立检查每一项操作建议及其前提" in wire["messages"][0]["content"]
    assert "同一正文里有正确的求助或保护句，不能抵消另一句的危险建议" in wire["messages"][0]["content"]
    assert wire["response_format"] == safety_request_format("safety_review_v2")
    assert wire["max_tokens"] == 1024 and wire["reasoning"] == {"enabled": False, "exclude": True}


def test_deletion_keeps_surrounding_support_without_trusting_conversation_kind(monkeypatch):
    """Forced deletion verifies delivery mechanics, not model recognition."""
    from tests.test_safety_core_deduplication import replay
    safe_before = "先别急着起来。"
    dangerous = "如果必须挪动，只做很小幅度的挪动，别强拉疼的那侧腿。"
    safe_after = "告诉接线员你现在站不起来。"
    segments = [{"text": text, "kind": "conversation"}
                for text in (safe_before, dangerous, safe_after)]
    result, ctx, _, reviewer = replay(monkeypatch, segments,
        review=answer("fall", delete=[1, 2, 3], subject="owner", time="current", urgency="immediate"))
    request = json.loads(reviewer.messages[0][1]["content"])
    from aria.engine.risk.safety_review_v2 import response_sentences
    assert [row["text"] for row in request["sentences"]] == list(response_sentences(safe_before + dangerous + safe_after))
    assert request["candidate"] == safe_before + dangerous + safe_after
    assert len(reviewer.messages) == 1
    assert dangerous not in result and safe_before in result and safe_after in result
    assert ctx.working_memory["safety_review"]["delete_sentence_ids"] == [1, 2, 3]
    assert ctx.working_memory["safety_review"]["semantic_verified"] is False
    assert ctx.working_memory["safety_review"]["grants_permission"] is False


def test_historical_review_v1_instruction_is_unchanged():
    from hashlib import sha256
    from aria.engine.risk.safety_review import SAFETY_REVIEW_INSTRUCTION
    assert sha256(SAFETY_REVIEW_INSTRUCTION.encode()).hexdigest() == (
        "f537b892d9815fc4a600051c32b1bbf52843d8c3c2b616dbd612a996677882f6")


@pytest.mark.parametrize("version", ["safety_review_v1", "safety_review_v2"])
def test_transport_response_cannot_substitute_other_protocol(tmp_path, monkeypatch, version):
    from tests.test_candidate_role_transport import client, guard, response, messages as legacy_messages
    from tests.test_reserved_calls import stub
    c = client("safety_review")
    data = response(c)
    messages = legacy_messages(c)
    if version == "safety_review_v1":
        data["choices"][0]["message"]["content"] = json.dumps(answer())
    else:
        payload = json.loads(messages[1]["content"])
        messages = safety_review_messages(payload["utterance"], "候选第一句。候选第二句。",
            payload["safety_evidence"], None, final_protection="候选保护。")
    stub(monkeypatch, data)
    with closing(guard(tmp_path / "ledger", c)) as g:
        g.attach(c)
        if version == "safety_review_v1":
            with g.scope("turn", "op1", "session1"), pytest.raises(ValueError):
                c.complete(messages)
        else:
            with g.scope("turn", "op1", "session1"):
                raw = c.complete(messages)
            # Receipt success does not turn a v1 body into a v2 verdict.
            with pytest.raises(ValueError):
                parse_review(raw, 2)


def provenance_review_messages(record):
    row = {"concern": "none", "subject": None, "time": None, "urgency": None,
        "quote": None, "verbatim_source": False, "disposition": None, "semantic_verified": False,
        "conversation_history": [], "delivery_facts": {
            "owner_sources": [], "preference_records": [record],
            "current_action_results": [], "code_status": None}}
    return safety_review_messages(TEXT, BODY, row, None, final_protection="")


@pytest.mark.parametrize("defect", ["none", "ui_null", "ui_action", "ui_persisted", "ui_extra", "unused_record_id"])
def test_review_input_requires_complete_typed_native_provenance(defect):
    # Mechanical wire validation only; these synthetic records are not evidence
    # that a model understands either the current state or action semantics.
    record = {"status": "withdrawn", "applicable": False, "description": "当前已撤回。",
        "last_ui_result": {"action": "withdraw", "persisted": True}}
    if defect == "ui_null":
        record["last_ui_result"] = None
    elif defect == "ui_action":
        record["last_ui_result"]["action"] = "notify_family"
    elif defect == "ui_persisted":
        record["last_ui_result"]["persisted"] = 1
    elif defect == "ui_extra":
        record["last_ui_result"]["grants_permission"] = True
    elif defect == "unused_record_id":
        record["state_ref"] = "pref:unconsumed"
    messages = provenance_review_messages(record)
    if defect == "none":
        assert validate_safety_review_request(messages) == 3
        assert set(safety_request_format("safety_review_v2")["json_schema"]["schema"]["properties"]) == {
            "risk_assessment", "conflicts", "delete_sentence_ids"}
    else:
        with pytest.raises(ValueError, match="invalid_delivery_review_preference"):
            validate_safety_review_request(messages)
