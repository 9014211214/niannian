"""Native storage and exact-position software checks; no model-effect evidence."""
from copy import deepcopy
import pytest

from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import apply_preference_dialogue
from aria.engine.memory.stated_preferences import transition_preference
from aria.engine.memory.preference_status import current_preference_states
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.semantic_delivery import VERSION, bind_current_action_receipt
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from tests.test_stated_preferences import _profile
from tests.test_semantic_delivery_v6 import context, segment, decode, deliver
from tests.test_minimal_web import live_fixture


USER = "请核对当前这条交流说明的状态。"


def prepare(p, c):
    messages = build_messages(get_scene("companion_elder"), p, c, USER,
                              safety_response_instruction_version="compact_response_v1")
    assert len(messages[0]["content"].encode("utf-8")) <= 12000
    return c.working_memory["prompt_evidence_snapshot"]["preference_state_sources"]


def render(p, c, ref):
    rows = [segment("模型自由声称刚刚成功了。", "preference_state", state_ref=ref), segment("我们继续聊。")]
    raw = decode(p, c, USER, rows)
    composed = apply_preference_dialogue(c, p, USER, response=raw)
    return guard_external_delivery_response(c, raw if composed is None else composed, USER, p)


@pytest.mark.parametrize("action,status", [("confirm", "已由本人确认并保存"), ("decline", "已拒绝"), ("withdraw", "已撤回")])
def test_real_transition_save_load_has_current_status_not_fresh_action(tmp_path, action, status):
    p = _profile(confirmed=action == "withdraw")
    store = PersonaStore(tmp_path)
    store.save(p)
    transition_preference(p, p.stated_preferences[0].id, action)
    store.save(p)
    p = store.load(p.user_id)
    _, c = context()
    c.user_id = p.user_id
    sources = prepare(p, c)
    output = render(p, c, sources[0]["state_ref"])
    assert status in output and "当前已保存记录中" in output
    assert "刚刚成功" not in output and output.endswith("我们继续聊。")
    assert c.working_memory["delivery_response_guard"]["segments"][0]["new_action_proven"] is False


@pytest.mark.parametrize("action", ["confirm", "decline", "withdraw"])
@pytest.mark.parametrize("active", [False, True])
def test_native_ui_transaction_and_physical_write_failure_report_old_state(tmp_path, action, active):
    app, clients = live_fixture(tmp_path)
    clients.online.online_decision_version = VERSION
    p = _profile(confirmed=action == "withdraw")
    p.user_id = "fictional-chen"
    app.store.save(p)
    if active:
        app.start()
    original = app.store._path(p.user_id).read_bytes()
    temporary = app.store._path(p.user_id).with_suffix(".json.tmp")
    temporary.mkdir()  # Real Windows/native file write failure, not a model mock.
    try:
        outcome = app.preference_action(action, p.stated_preferences[0].id)
        assert outcome["ok"] is False and outcome["state"]["storage_failed"]
        assert app.store._path(p.user_id).read_bytes() == original
        current = app._profile()
        _, c = context()
        c.user_id = current.user_id
        sources = prepare(current, c)
        output = render(current, c, sources[0]["state_ref"])
        assert "保存失败" in output and "不能当作该操作已生效" in output
        assert ("已由本人确认并保存" in output) is (action == "withdraw")
        assert "刚刚成功" not in output and output.endswith("我们继续聊。")
        assert app.store.load(p.user_id)._preference_ui_result is None
    finally:
        temporary.rmdir()
        app.close()


@pytest.mark.parametrize("action", ["confirm", "decline", "withdraw"])
def test_native_ui_success_publishes_actual_committed_profile(tmp_path, action):
    app, clients = live_fixture(tmp_path)
    clients.online.online_decision_version = VERSION
    p = _profile(confirmed=action == "withdraw")
    p.user_id = "fictional-chen"
    app.store.save(p)
    app.start()
    assert app.preference_action(action, p.stated_preferences[0].id)["ok"]
    current = app._profile()
    _, c = context()
    c.user_id = current.user_id
    refs = prepare(current, c)
    assert refs[0]["last_ui_result"]["persisted"] is True
    assert refs[0]["last_ui_result"]["session_id"] == app._sessions[p.user_id][1].session_id
    assert "当前已保存记录中" in render(current, c, refs[0]["state_ref"])
    app.close()


def test_current_native_update_wins_over_earlier_supplied_state(tmp_path):
    p = _profile()
    store = PersonaStore(tmp_path)
    store.save(p)
    _, c = context()
    c.user_id = p.user_id
    refs = prepare(p, c)
    transition_preference(p, p.stated_preferences[0].id, "withdraw")
    store.save(p)
    output = render(p, c, refs[0]["state_ref"])
    assert "已撤回" in output and "已由本人确认并保存" not in output
    assert "已撤回" in render(p, c, refs[0]["state_ref"])


@pytest.mark.parametrize("defect", ["unsaved", "modified_after_save", "fabricated_ref", "changed_input", "changed_session"])
def test_unbound_or_stale_state_never_becomes_verified_success(tmp_path, defect):
    p = _profile()
    _, c = context()
    c.user_id = p.user_id
    if defect != "unsaved":
        PersonaStore(tmp_path).save(p)
    refs = prepare(p, c)
    ref = refs[0]["state_ref"] if refs else "pref:missing"
    if defect == "modified_after_save":
        p.stated_preferences[0].scope = "changed without storage"
    if defect == "fabricated_ref":
        ref = "pref:never-provided"
    if defect == "changed_input":
        c.working_memory["prompt_evidence_snapshot"]["preference_state_binding"]["utterance_sha256"] = "wrong"
    if defect == "changed_session":
        c.session_id = "other"
    assert render(p, c, ref) == "我们继续聊。"


def test_store_source_is_not_a_model_field_and_is_not_loaded_from_json():
    from aria.engine.model.persona import PersonaProfile
    p = PersonaProfile(user_id="neutral", _preference_storage_source={"owner_id": "neutral", "rows_sha256": "x"})
    assert p._preference_storage_source is None
    assert "_preference_storage_source" not in p.model_dump_json()
    assert current_preference_states(p) == []


def repeated_rows():
    return [segment("你写的“"), segment("已发出。"), segment("”只是草稿。"),
            segment("已发出。", "assistant_action", action="notification", target="daughter")]


def test_duplicate_text_removes_the_correct_occurrence_not_the_quote():
    p, c = context()
    assert deliver(p, c, "核对这段话。", repeated_rows()) == "你写的“已发出。”只是草稿。"


def test_ambiguous_upstream_partial_text_is_not_approximately_mapped():
    p, c = context()
    decode(p, c, "核对这段话。", repeated_rows())
    assert guard_external_delivery_response(c, "已发出。", "核对这段话。", p) == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert c.working_memory["delivery_response_guard"]["violations"] == ["semantic_delivery_positions_unavailable"]


def test_medical_deletion_keeps_exact_duplicate_position_and_does_not_resurrect():
    p, c = context()
    rows = [segment("引用里写着已发出。"), segment("你把降压药剂量加倍，应该会好。"),
            segment("已发出。", "assistant_action", action="notification", target="daughter")]
    raw = decode(p, c, "核对文字。", rows)
    safe = get_scene("companion_elder").guard_response(c, raw)
    assert "你把降压药剂量加倍" not in safe
    out = guard_external_delivery_response(c, safe, "核对文字。", p)
    assert out.startswith("引用里写着已发出。") and "你把降压药剂量加倍" not in out
    assert out.count("已发出。") == 1 and out != DETERMINISTIC_MODEL_FAILURE_REPLY


def test_existing_medical_keyword_backstop_is_not_a_semantic_guarantee():
    # Retain the uncovered phrase from the first mechanical check. A local
    # finite backstop pass is not model/safety acceptance or complete coverage.
    p, c = context()
    text = "你应该把药加倍服用。"
    raw = decode(p, c, "中性当前输入。", [segment(text)])
    assert get_scene("companion_elder").guard_response(c, raw) == text


def test_identical_code_medical_boundary_is_not_removed_as_model_state():
    from aria.scenes.companion_elder.medical import SAFE_MEDICAL_BOUNDARY_REPLY_V2
    p, c = context()
    raw = decode(p, c, "核对文字。", [segment("你把降压药剂量加倍，应该会好。"),
        segment(SAFE_MEDICAL_BOUNDARY_REPLY_V2, "preference_state")])
    safe = get_scene("companion_elder").guard_response(c, raw)
    assert safe.count(SAFE_MEDICAL_BOUNDARY_REPLY_V2) == 2
    out = guard_external_delivery_response(c, safe, "核对文字。", p)
    assert out.count(SAFE_MEDICAL_BOUNDARY_REPLY_V2) == 1
    assert "你把降压药剂量加倍" not in out


@pytest.mark.parametrize("status,ok,expected", [("accepted", True, "受理或等待处理"), ("unknown", True, "仍无法核实"),
    ("unknown", False, "报告失败"), ("delivered", True, "成功回执")])
def test_current_executor_nonterminal_states_are_honest_and_keep_normal_body(status, ok, expected):
    p, c = context()
    receipt = {"ok": ok, "tool": "notify_family", "args": {"contact": "daughter", "message": "neutral"},
               "data": {"status": status}}
    c.working_memory["last_tool_result"] = receipt
    bind_current_action_receipt(c, USER, receipt)
    out = deliver(p, c, USER, [segment("我刚刚通知成功，她已经回复了。", "assistant_action", action="notification", target="daughter"),
                               segment("我们继续聊。")])
    assert expected in out and "她已经回复了" not in out and out.endswith("我们继续聊。")
