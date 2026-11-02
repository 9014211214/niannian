"""Binding, storage and delivery tests with supplied labels, not semantic scores."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.preference_candidate import VERSION, CONTEXT_KEY, bind_context
from aria.engine.memory.preference_dialogue import (
    prepare_preference_invitation, acknowledge_preference_invitation,
    bind_preference_request, preference_dialogue_prompt, apply_preference_dialogue,
)
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.policy.online_decision import single_pass_messages
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.semantic_delivery import segment_instruction, SEGMENT_INSTRUCTION
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.delivery import guard_external_delivery_response
from aria.release import create_qwen_max_preference_candidate_clients
from tests.test_memory_structured_candidate import route as memory_route, clients as old_clients
from tests.test_qwen_max_candidate import route as max_route
from tests.test_semantic_delivery_v6 import segment, decode
from tests.test_stated_preferences import _profile
from tests._model_http_stub import stub_model_http

V6 = "online_turn_decision_v6_semantic"
TEXT = "当前本人的软件测试答复"


def clients():
    return create_qwen_max_preference_candidate_clients(model_calls_enabled=True,
        api_key="inert-software-only", base_url="https://fixture.invalid/v1",
        online_structured_route=max_route(), memory_structured_route=memory_route())


def context(p):
    c = EngineContext(user_id=p.user_id, scene_id="companion_elder", session_id="new-synthetic-session")
    c.working_memory.update(_online_decision_version=V6, _safety_delivery_version="minimal_response_v1",
        _safety_response_instruction_version="compact_response_v1", **{CONTEXT_KEY: VERSION})
    return c


def prepared(store, *, others=0):
    p = _profile(confirmed=False)
    p.user_id = "fictional-chen"
    for n in range(others):
        other = p.stated_preferences[0].model_copy(deep=True)
        other.id = f"pref_other-{n}"
        other.do = "这是未被邀请确认的其他说明"
        other.updated_at = f"2030-01-{n+1:02d}T00:00:00+00:00"
        p.stated_preferences.append(other)
    store.save(p)
    c = context(p)
    target = p.stated_preferences[0].id
    invitation = prepare_preference_invitation(p, c, target, actor=p.user_id)
    acknowledge_preference_invitation(p, c, invitation["id"], invitation["question_sha256"], actor=p.user_id)
    bind_preference_request(p, c, TEXT)
    return p, c


def messages(p, c):
    base = build_messages(get_scene("companion_elder"), p, c, TEXT,
                          safety_response_instruction_version="compact_response_v1")
    return single_pass_messages(base, c, version=V6, persona=p)


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setattr("aria.runtime.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))


def test_only_explicit_factory_selects_new_protocol_and_strict_reflection():
    old, new = old_clients(), clients()
    assert getattr(old.online, "preference_protocol_version", None) is None
    assert old.reflection.reflection_return_format == "body_json"
    assert new.online.preference_protocol_version == VERSION
    assert new.reflection.reflection_return_format == "json_schema_v2"
    assert new.reflection.reflection_structured_route is new.memory.memory_structured_route
    assert new.reflection.model == old.reflection.model
    assert new.online.model == old.online.model
    assert new.reflection.timeout == 90 and new.reflection.max_completion_tokens == 2800
    assert all(getattr(new, role).network_request_count == 0 for role in ("online", "memory", "reflection", "safety_review"))
    assert not create_qwen_max_preference_candidate_clients().model_calls_enabled
    with pytest.raises(TypeError): create_qwen_max_preference_candidate_clients(model_calls_enabled=1)


def test_reusing_context_with_old_client_clears_candidate_protocol():
    p = _profile()
    c = context(p)
    bind_context(old_clients().online, c)
    assert CONTEXT_KEY not in c.working_memory
    assert segment_instruction(c) == SEGMENT_INSTRUCTION
    bind_context(clients().online, c)
    assert c.working_memory[CONTEXT_KEY] == VERSION


def test_displayed_object_has_exact_reference_even_beyond_old_four_row_window(tmp_path):
    p, c = prepared(PersonaStore(tmp_path), others=6)
    prompt = messages(p, c)[0]["content"]
    sources = c.working_memory["prompt_evidence_snapshot"]["preference_state_sources"]
    assert len(sources) == 4
    source = sources[0]
    assert source["target_id"] == p.stated_preferences[0].id
    assert source["state_ref"] in preference_dialogue_prompt(p, c)
    assert source["do"] in prompt and source["scope"] in prompt
    assert "这是未被邀请确认的其他说明" in prompt
    assert not c.working_memory["prompt_evidence_snapshot"].get("stated_preferences_prompt_projection")
    assert "普通复述和理解用conversation" in prompt
    assert len(prompt.encode("utf-8")) <= 12000


@pytest.mark.parametrize("action,status", [("confirm", "confirmed"), ("decline", "declined"),
    ("unclear", "proposed"), ("defer", "proposed")])
def test_labels_never_change_target_or_scope_and_restatement_survives(tmp_path, action, status):
    store = PersonaStore(tmp_path)
    p, c = prepared(store, others=2)
    messages(p, c)
    before = deepcopy(p.model_dump())
    text = "我听懂了你对相处方式的意思。"
    raw = decode(p, c, TEXT, [segment(text)], preference_control={"action": action})
    result = apply_preference_dialogue(c, p, TEXT, store=store, response=raw)
    delivered = guard_external_delivery_response(c, result, TEXT, p)
    assert delivered.startswith(text)
    assert p.stated_preferences[0].status == status
    assert p.stated_preferences[0].do == before["stated_preferences"][0]["do"]
    assert p.stated_preferences[0].scope == before["stated_preferences"][0]["scope"]
    assert all(row.status == "proposed" for row in p.stated_preferences[1:])
    persisted = store.load(p.user_id)
    assert persisted.stated_preferences == p.stated_preferences
    if action in {"unclear", "defer"}:
        assert p.model_dump() == before
        assert "我记下了" not in delivered and "你想记的时候" not in delivered
        assert "preference_dialogue_receipt" not in c.working_memory
    else:
        assert c.working_memory["preference_dialogue_receipt"]["persisted"] is True


def test_confirm_commit_native_withdraw_reload_and_input_suppression(tmp_path):
    from tests.test_minimal_web import live_fixture
    app, local_clients = live_fixture(tmp_path)
    local_clients.online.online_decision_version = V6
    p, c = prepared(app.store, others=1)
    messages(p, c)
    raw = decode(p, c, TEXT, [segment("我明白了。")], preference_control={"action": "confirm"})
    apply_preference_dialogue(c, p, TEXT, store=app.store, response=raw)
    target = p.stated_preferences[0].id
    reloaded = app.store.load(p.user_id)
    before = messages(reloaded, context(reloaded))[0]["content"]
    assert target in before
    try:
        app.start()
        result = app.preference_action("withdraw", target)
        assert result["ok"] is True
        after = app._profile()
        assert after.stated_preferences[0].status == "withdrawn"
        assert after.stated_preferences[1].status == "proposed"
        assert app.store.load(p.user_id).stated_preferences[0].status == "withdrawn"
        follow = context(after)
        messages(after, follow)
        snapshot = follow.working_memory["prompt_evidence_snapshot"]
        assert not snapshot.get("stated_preferences_prompt_projection")
        states = snapshot["preference_state_sources"]
        assert len(states) == 2 and states[0]["target_id"] == target
        assert states[0]["last_ui_result"]["persisted"] is True
        text = "软件标签声称撤回已经成功。"
        raw = decode(after, follow, TEXT, [segment(text, "preference_state", state_ref=states[0]["state_ref"])])
        delivered = guard_external_delivery_response(follow, raw, TEXT, after)
        assert "已撤回" in delivered and "软件标签" not in delivered
        restarted = app.store.load(p.user_id)
        assert restarted._preference_ui_result is None
        restarted_ctx = context(restarted)
        restarted_prompt = messages(restarted, restarted_ctx)[0]["content"]
        restarted_snapshot = restarted_ctx.working_memory["prompt_evidence_snapshot"]
        assert not restarted_snapshot.get("stated_preferences_prompt_projection")
        assert next(row for row in restarted_snapshot["preference_state_sources"]
                    if row["target_id"] == target)["status"] == "withdrawn"
        assert '"status":"withdrawn"' in restarted_prompt
        assert "不是当前相处要求" in restarted_prompt
        assert len(restarted_prompt.encode("utf-8")) <= 12000
    finally:
        app.close()


def test_confirmation_write_failure_keeps_invitation_and_never_publishes_success(tmp_path):
    p, c = prepared(PersonaStore(tmp_path))
    messages(p, c)
    before = deepcopy(p.model_dump())
    raw = decode(p, c, TEXT, [segment("我明白了。")], preference_control={"action": "confirm"})
    class Failed:
        def save(self, _): raise OSError("software write failure")
    with pytest.raises(OSError):
        apply_preference_dialogue(c, p, TEXT, store=Failed(), response=raw)
    assert p.model_dump() == before
    assert "_preference_dialogue_invitation" in c.working_memory
    assert "preference_dialogue_receipt" not in c.working_memory
