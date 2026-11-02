"""Software schema/receipt checks and two unchanged public-output replays."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION_V7 as V7, apply_single_pass_decision,
    DETERMINISTIC_MODEL_FAILURE_REPLY, online_decision_tool_spec,
)
from aria.engine.policy.online_json import parse_json_decision, parse_v7_payload
from aria.engine.policy.semantic_delivery import bind_current_action_receipt
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from aria.scenes.registry import get_scene
from tests.test_preference_required_control import client, envelope
from tests.test_safety_evidence import payload as legacy_payload


OPTIONALS = (
    "state_signals", "current_affect", "selected_intervention_id",
    "communication_plan", "uncertainty", "safety_evidence", "applied_preference_ids",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network or credential access forbidden")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)


def full_payload():
    value = legacy_payload(response="旧正文不得作为第二份正文。")
    value.pop("response")
    value.update(applied_preference_ids=[], response_segments=[{"text": "  我在听。\n继续说。  "}])
    return value


def context():
    persona = PersonaProfile(user_id="v7-test")
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="v7-session")
    ctx.working_memory.update(_online_decision_version=V7,
        _safety_delivery_version="minimal_response_v1", _safety_response_instruction_version="compact_response_v1")
    return persona, ctx


def deliver(value, *, persona=None, ctx=None):
    if ctx is None:
        persona, ctx = context()
    body = apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, "软件当前原话。", persona)
    return guard_external_delivery_response(ctx, body, "软件当前原话。", persona), ctx


@pytest.mark.parametrize("preference", [False, True])
@pytest.mark.parametrize("task", [False, True])
def test_v7_schema_has_only_two_required_fields(preference, task):
    schema = online_decision_tool_spec(V7, require_preference_control=preference,
                                     require_task_projection=task)["function"]["parameters"]
    assert schema["required"] == ["response_segments", "safety_concern"]
    assert "response" not in schema["properties"]
    assert "preference_control" not in schema["properties"]
    item = schema["properties"]["response_segments"]["items"]
    assert item["required"] == ["text"]
    assert "safety_core" in item["properties"]["kind"]["enum"]


@pytest.mark.parametrize("field", OPTIONALS)
def test_missing_optional_field_keeps_exact_body(field):
    value = full_payload()
    value.pop(field)
    body, ctx = deliver(value)
    assert body == "  我在听。\n继续说。  "
    assert ctx.working_memory["online_turn_decision"]["dropped_fields"] == []


@pytest.mark.parametrize("field", OPTIONALS)
def test_wrong_optional_type_is_dropped_and_logged_without_content(field):
    value = full_payload()
    value[field] = {"private_invalid_value": ["not retained"]}
    if field in ("current_affect", "communication_plan", "safety_evidence"):
        value[field] = ["private invalid value"]
    parsed, dropped = parse_v7_payload(json.dumps(value))
    assert field not in parsed and dropped == [field]
    body, ctx = deliver(value)
    assert body == "  我在听。\n继续说。  "
    assert ctx.working_memory["online_turn_decision"]["dropped_fields"] == [field]


@pytest.mark.parametrize("field,bad", [
    ("state_signals", [{"id": "invalid"}]),
    ("communication_plan", {"pace": "pause", "support_order": "  ", "boundary": " "}),
    ("uncertainty", float("nan")),
    ("uncertainty", True),
    ("safety_evidence", {"quote": "", "subject": "bad", "time": "current", "urgency": "immediate"}),
    ("applied_preference_ids", [False]),
])
def test_optional_nested_constraints_drop_only_the_bad_field(field, bad):
    value = full_payload()
    value[field] = bad
    body, ctx = deliver(value)
    assert body == "  我在听。\n继续说。  "
    assert ctx.working_memory["online_turn_decision"]["dropped_fields"] == [field]


def test_only_minimal_fields_deliver_without_invented_affect():
    body, ctx = deliver({"response_segments": [{"text": "我在听。"}], "safety_concern": "none"})
    assert body == "我在听。"
    assert ctx.emotion.inference_source != "online_single_pass_decision"
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["state_signals"] == []
    assert trace["safety_evidence"] is None
    assert trace["communication_plan"] is None


@pytest.mark.parametrize("segments", [[], [{"text": ""}], [{"text": " \t\n"}, {"text": ""}]])
def test_empty_or_all_whitespace_body_still_fails(segments):
    raw = json.dumps({"response_segments": segments, "safety_concern": "none"})
    with pytest.raises(ValueError, match="empty_online_response"):
        parse_v7_payload(raw)
    persona, ctx = context()
    assert apply_single_pass_decision(raw, ctx, "当前原话。", persona) == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert ctx.working_memory["online_turn_decision"]["status"] == "invalid_protocol_response"


@pytest.mark.parametrize("extra", [
    {}, {"kind": "invalid"}, {"action": 3}, {"source_quote": []},
    {"target": "x" * 201}, {"state_ref": None}, {"kind": "conversation", "action": "notification"},
])
def test_bad_segment_metadata_downgrades_only_one_item(extra):
    value = {"response_segments": [{"text": "", **extra}, {"text": "甲\n", **extra}, {"text": "\t乙  "}],
             "safety_concern": "none"}
    parsed, _ = parse_v7_payload(json.dumps(value))
    assert parsed["response"] == "甲\n\t乙  "
    assert parsed["response_segments"][1] == {
        "text": "甲\n", "kind": "conversation", "source_quote": "", "action": "", "target": "", "state_ref": "",
    }
    body, _ = deliver(value)
    assert body == parsed["response"]


def test_bad_text_item_does_not_destroy_other_usable_segments():
    parsed, dropped = parse_v7_payload(json.dumps({"response_segments": [
        {"text": "甲"}, {"text": 42}, {}, None, {"text": "乙"}], "safety_concern": "none"}))
    assert parsed["response"] == "甲乙"
    assert dropped == ["response_segments"]


@pytest.mark.parametrize("concern", [None, "other", False, [], ""])
def test_safety_concern_keeps_its_strict_required_enum(concern):
    value = {"response_segments": [{"text": "正文。"}], "safety_concern": concern}
    with pytest.raises(ValueError):
        parse_json_decision(json.dumps(value), V7)


def test_legacy_response_cannot_override_exact_segment_concatenation():
    value = {"response": "另一份互相矛盾的正文。", "response_segments": [{"text": "  甲\n"}, {"text": "\t乙。  "}],
             "safety_concern": "none"}
    body, ctx = deliver(value)
    assert body == "  甲\n\t乙。  "
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["parsed_response_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert trace["dropped_fields"] == ["response"]


@pytest.mark.parametrize("text", ['{"answer":"ok"}', "[1,2,3]", '```json\n{"answer":"ok"}\n```'])
def test_structured_user_body_is_still_usable_text(text):
    body, ctx = deliver({"response_segments": [{"text": text}], "safety_concern": "none"})
    assert body == text
    assert ctx.working_memory["online_turn_decision"]["status"] == "parsed"


@pytest.mark.parametrize("metadata", [{}, {"action": None}, {"action": "notification", "target": []}])
def test_action_with_missing_or_bad_optional_metadata_cannot_bypass_receipts(metadata):
    body, ctx = deliver({"response_segments": [
        {"text": "我已经发给她了。", "kind": "assistant_action", **metadata},
        {"text": "我们继续。"}], "safety_concern": "none"})
    assert body == "我们继续。"
    assert ctx.working_memory["delivery_response_guard"]["activated"] is True


@pytest.mark.parametrize("metadata", [{"state_ref": None}, {"target": 42}, {"action": "notification"}])
def test_preference_state_with_bad_optional_metadata_cannot_claim_saved(metadata):
    value = {"response_segments": [
        {"text": "我已经保存了。", "kind": "preference_state", **metadata},
        {"text": "我们继续。"}], "safety_concern": "none"}
    parsed, _ = parse_v7_payload(json.dumps(value))
    assert parsed["response_segments"][0]["kind"] == "conversation"
    body, ctx = deliver(value)
    assert body == "我们继续。"
    assert ctx.working_memory["online_turn_decision"]["unverified_preference_segment_indexes"] == [0]
    assert ctx.working_memory.get("preference_dialogue_receipt") is None


def test_bad_third_party_source_metadata_does_not_become_a_grounded_claim():
    body, ctx = deliver({"response_segments": [
        {"text": "她已经回复了。", "kind": "third_party_fact", "source_quote": []},
        {"text": "我们继续。"}], "safety_concern": "none"})
    assert body == "我们继续。"
    assert ctx.working_memory["online_turn_decision"]["unverified_fact_segment_indexes"] == [0]


@pytest.mark.parametrize("receipt", [False, True])
def test_action_completion_still_requires_real_current_receipt(receipt):
    persona, ctx = context()
    if receipt:
        result = {"ok": True, "tool": "notify_family", "args": {"contact": "daughter"},
                  "data": {"status": "delivered"}}
        ctx.working_memory["last_tool_result"] = result
        bind_current_action_receipt(ctx, "软件当前原话。", result)
    value = {"response_segments": [
        {"text": "我已经发给她了。", "kind": "assistant_action", "action": "notification", "target": "daughter"},
        {"text": "我们继续。"}], "safety_concern": "none"}
    body, _ = deliver(value, persona=persona, ctx=ctx)
    assert "我已经发给她了。" not in body
    assert ("成功回执" in body) is receipt
    assert body.endswith("我们继续。")


def run_engine(monkeypatch, value, *, instruction="legacy"):
    c = client()
    c.online_decision_version = V7
    wires = []
    def receive(request, **kwargs):
        wires.append(json.loads(request.data))
        return envelope(value, "json_schema_v1")
    monkeypatch.setattr(c, "_request_json", receive)
    persona = PersonaProfile(user_id="engine-v7")
    engine = Engine(get_scene("companion_elder"), persona, fast_llm_client=c,
                    safety_response_instruction_version=instruction,
                    **({"safety_delivery_version": "minimal_response_v1"} if instruction == "compact_response_v1" else {}))
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    monkeypatch.setattr(engine._risk, "assess", lambda ctx: None)
    return engine.process_turn(ctx, "软件当前原话。"), ctx, wires


@pytest.mark.parametrize("bad", [None, "invalid"])
def test_optional_safety_evidence_cannot_replace_ordinary_body_in_engine(monkeypatch, bad):
    value = {"response_segments": [{"text": "我在听。"}], "safety_concern": "none"}
    if bad is not None:
        value["safety_evidence"] = bad
    body, ctx, wires = run_engine(monkeypatch, value)
    assert body == "我在听。"
    assert len(wires) == 1
    assert ctx.working_memory["online_turn_decision"]["safety_evidence"] is None


def test_missing_optional_evidence_does_not_make_explicit_risk_ordinary(monkeypatch):
    body, ctx, wires = run_engine(monkeypatch, {
        "response_segments": [{"text": "普通正文。"}], "safety_concern": "fall"})
    assert len(wires) == 1
    assert body != "普通正文。"
    assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]
    assert ctx.working_memory["online_turn_decision"]["safety_concern"] == "fall"


def test_compact_release_body_and_wire_use_v7(monkeypatch):
    body, _, wires = run_engine(monkeypatch, {"response_segments": [{"text": "我在听。"}], "safety_concern": "none"},
                                instruction="compact_response_v1")
    assert body == "我在听。"
    wire, = wires
    assert wire["response_format"]["json_schema"]["schema"]["required"] == ["response_segments", "safety_concern"]
    assert "preference_control" not in json.dumps(wire, ensure_ascii=False)
    assert "正文只写在 response_segments 里" in wire["messages"][0]["content"]
    assert "response是供本人阅读的完整正文" not in wire["messages"][0]["content"]


@pytest.mark.parametrize("invited", [False, True])
def test_deepseek_final_wire_has_receipt_and_fact_boundaries_without_new_required_fields(
        monkeypatch, tmp_path, invited):
    """Capture the actual product wire before send; this is not semantic accuracy."""
    from contextlib import closing
    from aria.engine.policy.max_release.guard import MaxReleaseCallGuard
    from aria.engine.policy.release_prompt import INSTRUCTION, INSTRUCTION_V7, SYSTEM_BYTE_LIMIT
    from aria.voice.minimal import MinimalWebApp
    from tests.test_deepseek_release import clients, prices
    from tests.test_stated_preferences import _profile

    current = "软件当前原话。"
    wires = []

    class Captured(BaseException):
        pass

    def capture(client, request, **options):
        wires.append(json.loads(request.data))
        raise Captured()

    with closing(MaxReleaseCallGuard(tmp_path / "ledger.sqlite3", max_requests=1,
            max_observed_cost_usd="0.4", reservation_policy=prices())) as budget:
        clock = [1000.0]
        budget._pacing_clock = lambda: clock[0]
        budget._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        app = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=budget)
        monkeypatch.setattr(budget, "_request_once", capture)
        try:
            session = app.start()["state"]["session_id"]
            if invited:
                engine, _ = app._sessions[app._settled_owner]
                engine.persona.stated_preferences = deepcopy(_profile(confirmed=False).stated_preferences)
                app.store.save(engine.persona)
                invitation = app.preference_invitation(engine.persona.stated_preferences[0].id,
                                                       session)["preference_invitation"]
                assert app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], session)["ok"]
            with pytest.raises(Captured):
                app.submit(current, "prompt-boundary", session)
            assert budget.snapshot()["calls_used"] == 0
        finally:
            app.close()
    wire, = wires
    system = wire["messages"][0]["content"]
    assert len(system.encode("utf-8")) <= SYSTEM_BYTE_LIMIT
    assert system.count(INSTRUCTION_V7) == 1 and INSTRUCTION not in system
    assert system.count("保存成功回执只由代码在写入成功后追加") == 1
    assert system.count("复述保留主体、亲属称谓、事件先后") == 1
    assert system.count("严重外伤或疑似骨折立即联系当地急救") == 1
    assert system.count("仅眼前直接环境危险或救命急救确需移动时例外") == 1
    assert "疑似头颈背伤不随意搬动；直接环境危险下先安全脱险。" not in system
    # The explicit historical compact policy remains byte-for-byte unchanged.
    assert hashlib.sha256(INSTRUCTION.encode()).hexdigest() == (
        "40e6969cb0bb095bbe5e8f546b02ceaa84316c53eceee246efbc3cd5777f10e8")
    assert "conversation（普通交流、建议、本人说法的复述）" not in system
    assert wire["response_format"]["json_schema"]["schema"]["required"] == [
        "response_segments", "safety_concern"]
    assert "preference_control" not in json.dumps(wire, ensure_ascii=False)


def test_v7_status_segment_without_save_keeps_exact_ordinary_response(monkeypatch):
    ordinary = "我理解您这次的要求。\n您想继续说的时候，我在听。"
    body, ctx, wires = run_engine(monkeypatch, {"response_segments": [
        {"text": "记下了。", "kind": "preference_state"},
        {"text": ordinary, "kind": "conversation"}], "safety_concern": "none"},
        instruction="compact_response_v1")
    assert body == ordinary
    assert len(wires) == 1
    assert ctx.working_memory.get("preference_dialogue_receipt") is None


def test_v7_incorrect_conversation_label_remains_an_explicit_model_risk():
    """A prompt constraint cannot turn source checks into semantic verification."""
    incorrect = "我已记下这条，以后每次都会这样办。"
    body, ctx = deliver({"response_segments": [{"text": incorrect, "kind": "conversation"}],
                         "safety_concern": "none"})
    assert body == incorrect
    assert ctx.working_memory.get("preference_dialogue_receipt") is None
    assert ctx.working_memory["delivery_response_guard"]["semantic_verified"] is False


def public_fixture(name):
    """Load a public synthetic equivalent. Historical raw provider results stay private."""
    def unpack(value):
        if type(value) is dict and set(value) == {"object_pairs"}:
            return {key: unpack(child) for key, child in value["object_pairs"]}
        if type(value) is list:
            return [unpack(child) for child in value]
        return value
    path = Path(__file__).parent / "fixtures" / name
    return unpack(json.loads(path.read_text(encoding="utf-8")))["choices"][0]["message"]["content"]


def test_saved_mismatched_public_answer_is_deliverable_without_false_save_claim():
    raw = public_fixture("online-v7-mismatch-first-visible.json")
    original = json.loads(raw)
    assert original["response"] != "".join(row["text"] for row in original["response_segments"])
    with pytest.raises(ValueError):
        from aria.engine.policy.max_release.online_shape import online_content_projection
        online_content_projection(raw, "online_turn_decision_v6_semantic")
    parsed, dropped = parse_v7_payload(raw)
    assert parsed["response"] == "".join(row["text"] for row in original["response_segments"])
    persona, ctx = context()
    body = apply_single_pass_decision(raw, ctx, "软件当前原话。", persona)
    assert body == parsed["response"]
    delivered = guard_external_delivery_response(ctx, body, "软件当前原话。", persona)
    assert delivered == "".join(row["text"] for row in original["response_segments"] if row["kind"] == "conversation")
    assert delivered.strip()
    assert dropped == ["preference_control", "response"]


def test_saved_ninth_empty_public_answer_remains_unusable():
    raw = public_fixture("online-v7-empty-first-visible.json")
    with pytest.raises(ValueError, match="empty_online_response"):
        parse_v7_payload(raw)
    persona, ctx = context()
    assert apply_single_pass_decision(raw, ctx, "软件当前原话。", persona) == DETERMINISTIC_MODEL_FAILURE_REPLY


def test_default_max_profile_is_v7_and_explicit_v6_is_retained():
    from aria.release import create_max_release_clients
    from tests.test_installable_max_release import make_clients, metadata
    from aria.engine.policy.qwen_max_structured import build_qwen_max_structured_route
    from aria.engine.policy.memory_structured import build_memory_structured_route
    default = make_clients()
    assert default.online.online_decision_version == V7
    models, online, offline = metadata()
    old = create_max_release_clients(model_calls_enabled=True, api_key="inert-software",
        base_url="https://fixture.invalid/v1", online_decision_version="online_turn_decision_v6_semantic",
        online_structured_route=build_qwen_max_structured_route(models, online),
        memory_structured_route=build_memory_structured_route(models, offline))
    assert old.online.online_decision_version == "online_turn_decision_v6_semantic"


def test_default_max_actual_request_wire_has_only_two_required_fields(monkeypatch, tmp_path):
    from contextlib import closing
    from tests.test_installable_max_release import make_clients, guard, Response, tls_dispatch
    from tests.test_qwen_max_candidate import response
    selected = make_clients().online
    raw_value = {"response_segments": [{"text": "默认Max正文。"}], "safety_concern": "none"}
    data = response(selected)
    data["choices"][0]["message"]["content"] = json.dumps(raw_value, ensure_ascii=False)
    wires = []
    def receive(request, **kwargs):
        wires.append(json.loads(request.data))
        kwargs["trace"].preparation_recovery.bind_request_headers(request)
        kwargs["trace"].preparation_recovery.before_preparation()
        return Response(data)
    monkeypatch.setattr(tls_dispatch, "open_observed", receive)
    with closing(guard(tmp_path)) as budget:
        budget.attach(selected)
        with budget.scope("turn", "v7-wire", "v7-session"):
            result = selected.complete([{"role": "system", "content": "协议版本：" + selected.online_decision_version},
                                        {"role": "user", "content": "软件输入。"}])
    assert json.loads(result) == raw_value
    schema = wires[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["response_segments", "safety_concern"]
    assert "response" not in schema["properties"] and "preference_control" not in schema["properties"]
    assert len(wires) == 1
