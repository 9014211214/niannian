"""RF-1 product behavior with synthetic local HTTP and real persistence gates."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy import reflection_json
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.runtime import create_minimal_clients
from tests.test_reflection_evidence_review import context
from tests.test_reflection_structured_contract_v23 import payload
from tests.test_reserved_calls import guard, receipt, stub

EMPTY = {"protocol_version": "2.0", "reviews": [], "changes": []}


class Replies:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []
        self.last_usage = {"prompt_tokens": 2, "completion_tokens": 3, "cost": .00001}

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value if type(value) is str else json.dumps(value, ensure_ascii=False)


def run(values):
    client = Replies(values)
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    persona = PersonaProfile(user_id="person")
    updater.update(persona, context())
    return updater, client, persona


def test_default_body_has_full_schema_and_applies_original_evidence_transaction():
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture").reflection
    assert client.reflection_return_format == "body_json"
    updater, calls, persona = run([payload()])
    assert updater.last_status == "ok" and len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].status == "candidate"
    assert len(calls.calls) == 1 and updater.last_format_correction_count == 0
    prompt = calls.calls[0][0]["content"]
    assert json.dumps(reflection_json.reflection_response_schema(), ensure_ascii=False, sort_keys=True) in prompt
    assert reflection_json.reflection_product_step(calls.calls[0]) == "initial"


def test_one_correction_sends_only_safe_paths_and_types_not_first_body():
    first = {**EMPTY, "reviews": "PRIVATE_GENERATED_VALUE", "PRIVATE_UNKNOWN_KEY": "PRIVATE_BODY"}
    updater, client, persona = run([first, EMPTY])
    assert updater.last_status == "ok" and not persona.person_hypotheses
    assert updater.last_format_correction_count == 1 and len(client.calls) == 2
    assert client.calls[1][:2] == client.calls[0]
    assert reflection_json.reflection_product_step(client.calls[1]) == "format_correction"
    assert "PRIVATE" not in json.dumps(client.calls[1], ensure_ascii=False)
    assert "PRIVATE" not in json.dumps(updater.last_format_attempts)
    assert [row["status"] for row in updater.last_format_attempts] == ["shape_failed", "shape_valid"]
    assert [row["usage"]["cost"] for row in updater.last_format_attempts] == [.00001, .00001]


@pytest.mark.parametrize("first", ['{"reviews":', '{"protocol_version":"2.0","reviews":[],"reviews":[],"changes":[]}',
                                   '{"protocol_version":"2.0","reviews":[],"changes":[],"x":NaN}'])
def test_json_shape_correction_is_bounded_to_one(first):
    updater, client, persona = run([first, first, EMPTY])
    assert updater.last_status == "failed" and len(client.calls) == 2
    assert not persona.person_hypotheses and not persona.evidence_review_ledger
    assert persona.person_model_state.sessions_observed == 0


@pytest.mark.parametrize("failure", ["refusal_text", "wrong_quote", "domain_constraint", "model_error"])
def test_nonformat_failure_never_requests_a_replacement_answer(failure):
    value = payload()
    if failure == "refusal_text": value = "我不能完成这个请求。"
    if failure == "wrong_quote": value["reviews"][0]["evidence"][0]["evidence_quote"] = "模型编造但本人没有说过"
    if failure == "domain_constraint": value["changes"][0]["dimension"] = "openness"
    if failure == "model_error": value = RuntimeError("local model fixture failure")
    updater, client, persona = run([value, EMPTY])
    assert updater.last_status == "failed" and len(client.calls) == 1
    assert updater.last_format_correction_count == 0
    assert not persona.person_hypotheses and not persona.evidence_review_ledger


def test_actual_wire_two_calls_have_separate_receipts_and_original_parent(tmp_path, monkeypatch):
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture").reflection
    values = iter([{**EMPTY, "reviews": "bad"}, EMPTY])
    data = receipt(client)
    def next_body(_):
        data["choices"][0]["message"]["content"] = json.dumps(next(values))
    calls = stub(monkeypatch, data, next_body)
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        g.attach(client)
        with g.scope("review", "review1", "session_1"):
            updater.update(PersonaProfile(user_id="person"), context())
        assert updater.last_status == "ok"
        accounting = g.snapshot()
        assert accounting["calls_used"] == 2 and accounting["known_cost_usd"] == "0.00002"
        assert accounting["unknown_cost_requests"] == 0
        first, second = accounting["attempts"]
        assert first["status"] == second["status"] == "received"
        assert first["request_id"] != second["request_id"]
        assert second["format_parent_request_id"] == first["request_id"]
        assert first["operation_id"] == second["operation_id"] == "review1"
        assert first["reflection_input_sha256"] == second["reflection_input_sha256"]
        with g.scope("review", "review1", "session_1"), pytest.raises(MinimalCallStopped):
            client.complete(calls[-1]["messages"])
        assert g.snapshot()["calls_used"] == 2
    assert len(calls) == 2
    for wire in calls:
        assert not {"response_format", "tools", "tool_choice", "temperature"} & wire.keys()
        assert wire["provider"]["only"] == ["anthropic"]
        assert wire["provider"]["require_parameters"] is True
        assert wire["reasoning"] == {"enabled": False, "exclude": True}


@pytest.mark.parametrize("mutation", ["refusal", "truncation", "tools", "reasoning"])
def test_wrong_channel_is_not_eligible_for_product_format_correction(monkeypatch, mutation):
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture").reflection
    data = receipt(client)
    choice = data["choices"][0]
    choice["message"]["content"] = '{"reviews":'
    if mutation == "refusal": choice["message"]["refusal"] = "no"
    if mutation == "truncation": choice["finish_reason"] = "length"
    if mutation == "tools": choice["message"]["tool_calls"] = [{"id": "x"}]
    if mutation == "reasoning": choice["message"]["reasoning"] = "PRIVATE"
    calls = stub(monkeypatch, data)
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    updater.update(PersonaProfile(user_id="person"), context())
    assert updater.last_status == "failed" and updater.last_format_correction_count == 0
    assert len(calls) == 1
    assert reflection_json.reflection_body_observation(data)["correction_issues"] == []


@pytest.mark.parametrize("mode", ["json_schema_v1", "json_schema_v2", "json_schema"])
def test_archived_strict_modes_are_explicit_and_full_local_gate_remains(mode, monkeypatch):
    from aria.engine.memory.diagnosis import reflection_system_prompt
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture", reflection_return_format=mode).reflection
    data = receipt(client)
    data["choices"][0]["message"]["content"] = json.dumps(EMPTY)
    calls = stub(monkeypatch, data)
    client.complete([{"role": "system", "content": reflection_system_prompt(True)}, {"role": "user", "content": "local fixture"}])
    assert calls[0]["response_format"] == reflection_json.reflection_request_format(mode)
    refs = calls[0]["response_format"]["json_schema"]["schema"]["$defs"]["_Change"]["properties"]["atomic_claim"]["anyOf"]
    assert len(refs) == (3 if mode == "json_schema_v1" else 2)


def test_failed_reflection_seals_without_losing_memory_r1_or_blocking_next_session(tmp_path):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.engine.memory.stated_preferences import transition_preference
    from aria.runtime import MinimalModelClients, create_minimal_engine
    from aria.scenes.registry import get_scene
    from tests.test_model_routing import _DECISION_OK, _ProtocolAwareFast
    from tests.test_stated_preferences import _context, _row

    fast = _ProtocolAwareFast(_DECISION_OK)
    memory = Replies([{"episodes": [], "relationships": [], "stated_preferences": [_row()], "cognitive_privacy": []}] * 2)
    reflection = Replies([{"reviews": []}] * 4)
    reflection.reflection_return_format = "body_json"
    store = PersonaStore(tmp_path)
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="preference_test"),
                                  clients=MinimalModelClients(fast, memory, reflection, True), store=store)
    for session in ("rf1_first", "rf1_second"):
        ctx = _context(session)
        ctx.add_turn("今天就先聊这些。", "好的。")
        engine.seal_session(ctx)
        result = engine.recover_offline_sessions(max_jobs=1)
        assert result["last_run"]["completed_jobs"] == 1 and result["pending_count"] == 0
        saved = store.load("preference_test")
        audit = saved.offline_session_audit[-1]
        assert audit.outcome == "reflection_failed" and audit.gap
        assert audit.memory_completed and not audit.reflection_completed
        new_rows = [row for row in saved.stated_preferences if row.quotes[0].session_id == session]
        assert len(new_rows) == 1 and new_rows[0].status == "proposed"
        transition_preference(engine.persona, new_rows[0].id, "confirm")
        store.save(engine.persona)
        saved = store.load("preference_test")
        assert saved.stated_preferences[0].status == "confirmed"
        assert not saved.person_hypotheses and not saved.evidence_review_ledger
        assert saved.person_model_state.sessions_observed == 0
    assert len(memory.calls) == 2 and len(reflection.calls) == 4
    before = len(memory.calls), len(reflection.calls)
    engine.process_turn(_context("live_next"), "周末安排先跟我商量就好。")
    assert len(fast.calls) == 1 and (len(memory.calls), len(reflection.calls)) == before


def test_failed_reflection_seal_save_error_keeps_original_committed_memory():
    from tests.test_offline_queue import setup, context as queue_context, memory
    queue, store, _ = setup(fail_on=(4,))
    queue.enqueue(queue_context())
    def failed(persona, ctx):
        persona.disc.dominance = .99
        return persona, "sealed_failed", "invalid_json_response"
    result = queue.drain(memory, failed)
    assert result["status"] == "failed" and result["error_code"] == "offline_store_failed"
    assert result["completed_jobs"] == 0
    assert queue.persona.offline_sessions[0].memory_completed
    assert not queue.persona.offline_sessions[0].reflection_completed
    assert queue.persona.offline_session_audit == []
    assert queue.persona.disc.dominance != .99
    assert len(queue.persona.episodic_memory) == 1


@pytest.mark.parametrize("state", [
    {"stopped": True, "attempts": []},
    {"stopped": False, "attempts": [{"status": "pending"}]},
])
def test_budget_or_unresolved_claim_never_becomes_a_completed_optional_stage(tmp_path, state):
    from aria.engine.memory.persona_store import PersonaStore
    from aria.runtime import MinimalModelClients, create_minimal_engine
    from aria.scenes.registry import get_scene
    from tests.test_model_routing import _DECISION_OK, _ProtocolAwareFast
    reflection = Replies([{"reviews": []}] * 2)
    reflection.reflection_return_format = "body_json"
    class Control:
        def snapshot(self): return deepcopy(state)
    reflection.model_request_control = Control()
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="person"),
        clients=MinimalModelClients(_ProtocolAwareFast(_DECISION_OK), Replies([{"episodes": [], "relationships": [], "cognitive_privacy": []}]), reflection, True),
        store=PersonaStore(tmp_path))
    engine.seal_session(context())
    result = engine.recover_offline_sessions(max_jobs=1)
    assert result["last_run"]["completed_jobs"] == 0 and result["pending_count"] == 1
    assert engine.persona.offline_sessions[0].memory_completed
    assert not engine.persona.offline_session_audit


@pytest.mark.parametrize("mutation", ["new_input", "wrong_path", "another_session", "after_valid_first"])
def test_correction_cannot_change_context_or_fabricate_eligibility(tmp_path, monkeypatch, mutation):
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture").reflection
    data = receipt(client)
    data["choices"][0]["message"]["content"] = json.dumps(EMPTY if mutation == "after_valid_first" else {**EMPTY, "reviews": "bad"})
    calls = stub(monkeypatch, data)
    messages = [{"role": "system", "content": reflection_json.reflection_body_system_prompt()},
                {"role": "user", "content": "same source"}]
    with closing(guard(tmp_path / "guard.sqlite")) as g:
        g.attach(client)
        with g.scope("review", "review1", "session1"):
            client.complete(messages)
        issues = [{"path": ["reviews"], "type": "list_type"}]
        if mutation == "wrong_path": issues[0]["path"] = ["changes"]
        correction = [*deepcopy(messages), reflection_json.reflection_correction_message(issues)]
        if mutation == "new_input": correction[1]["content"] = "altered source"
        with g.scope("review", "review1", "other_session" if mutation == "another_session" else "session1"), pytest.raises(MinimalCallStopped):
            client.complete(correction)
        assert g.snapshot()["calls_used"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize("accounting", ["budget", "unknown"])
def test_first_receipt_accounting_failure_prevents_format_correction(tmp_path, monkeypatch, accounting):
    client = create_minimal_clients(model_calls_enabled=True, api_key="local-fixture").reflection
    data = receipt(client)
    data["choices"][0]["message"]["content"] = json.dumps({**EMPTY, "reviews": "bad"})
    if accounting == "budget": data["usage"]["cost"] = .99
    else: del data["usage"]["cost"]
    calls = stub(monkeypatch, data)
    updater = PersonaHypothesisUpdater(client, evidence_review_protocol=True, body_json_protocol=True)
    with closing(guard(tmp_path / "meter.sqlite")) as g:
        g.attach(client)
        with g.scope("review", "review1", "session_1"):
            updater.update(PersonaProfile(user_id="person"), context())
        state = g.snapshot()
        assert state["calls_used"] == 1 and len(calls) == 1
        assert updater.last_status == "failed" and updater.last_format_correction_count == 0
        if accounting == "unknown":
            assert state["unknown_cost_requests"] == 1 and state["stopped"]
        else:
            assert state["known_cost_usd"] == "0.99"
