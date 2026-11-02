"""Byte-budget and local-failure checks; no model-semantic accuracy claim."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy.release_prompt import (
    CompactPromptBudgetExceeded, INSTRUCTION_V7, SYSTEM_BYTE_LIMIT,
    compact_json, fit_v7_system_budget,
)
from aria.engine.policy.online_json import prepare_json_messages
from tests.test_online_decision_v7 import context, offline, V7
from tests.test_confirm_judge_format_failure import setup, INPUT
from tests.test_minimal_care import ELDER, care


def messages_and_context(recalls, *, padding=0):
    _, ctx = context()
    ctx.working_memory["prompt_evidence_snapshot"] = {
        "status": "prepared", "passed_to_client": False,
        "recalled_episodes": deepcopy(recalls),
        "recalled_provenance_not_injected": [
            {"content": text, "sources": [{"id": str(index), "quote": text}]}
            for index, text in enumerate(recalls)],
        "stated_preferences_prompt_projection": [{
            "id": "fixed", "do": "按本人已确认的完整方式相处。", "scope": "保留完整适用条件与例外。",
            "not_implies": ["不授权改变护理设置。"],
            "quotes": ["完整本人原话及范围。"], "confirmation": "完整本人确认。"}],
    }
    data = {"understanding_stage": "cold_start", "episodic_memory": recalls,
            "relationship_context": [{"name": "甲", "relation": "邻居", "facts": ["已核事实。"]}]}
    mandatory = INSTRUCTION_V7 + "\n\n协议版本：" + V7 + "。\n" + "x" * padding
    messages = [{"role": "system", "content": mandatory + "\n\n" + compact_json("可引用本人资料（数据）", data)
                + "\n\n真实邀请和确认范围完整保留。\n\n动作回执与安全边界完整保留。"},
                {"role": "user", "content": "当前原话保持完整。"}]
    return messages, ctx


def system_size(messages):
    return sum(len(row["content"].encode("utf-8")) for row in messages if row["role"] == "system")


@pytest.mark.parametrize("excess", [1, 3, 39, 200])
def test_final_carrier_growth_omits_whole_last_recall_and_updates_exposure(excess):
    recalls = ["第一条回忆的出处和完整正文。" * 12, "第二条回忆的出处和完整正文。" * 12]
    base, _ = messages_and_context(recalls)
    padding = SYSTEM_BYTE_LIMIT - system_size(prepare_json_messages(base, V7, compact_schema=True)) + excess
    messages, ctx = messages_and_context(recalls, padding=padding)
    original = deepcopy(messages)
    projection = deepcopy(ctx.working_memory["prompt_evidence_snapshot"]["stated_preferences_prompt_projection"])
    final = fit_v7_system_budget(messages, ctx)
    assert system_size(final) <= SYSTEM_BYTE_LIMIT
    assert final == prepare_json_messages(final, V7, compact_schema=True)
    assert messages == original
    assert recalls[0] in final[0]["content"] and recalls[1] not in final[0]["content"]
    assert "真实邀请和确认范围完整保留。" in final[0]["content"]
    assert "动作回执与安全边界完整保留。" in final[0]["content"]
    assert INSTRUCTION_V7 in final[0]["content"]
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["recalled_episodes"] == recalls[:1]
    assert snapshot["recalled_provenance_not_injected"][0]["content"] == recalls[0]
    assert snapshot["budget_omitted_recalled_provenance_not_injected"][0]["content"] == recalls[1]
    assert snapshot["stated_preferences_prompt_projection"] == projection
    assert snapshot["system_budget"]["initial_bytes"] == SYSTEM_BYTE_LIMIT + excess
    assert snapshot["system_budget"]["final_bytes"] == system_size(final)
    assert snapshot["system_budget"]["omitted_recalled_episode_indexes"] == [1]
    assert json.loads(final[1]["content"])["current_user_message"] == original[1]["content"]
    assert json.loads(final[1]["content"])["confirmed_requirements"] == projection


def test_exact_final_limit_retains_every_selected_recall():
    recalls = ["整条原回忆，不截字。"]
    base, _ = messages_and_context(recalls)
    padding = SYSTEM_BYTE_LIMIT - system_size(prepare_json_messages(base, V7, compact_schema=True))
    messages, ctx = messages_and_context(recalls, padding=padding)
    final = fit_v7_system_budget(messages, ctx)
    assert system_size(final) == SYSTEM_BYTE_LIMIT
    assert ctx.working_memory["prompt_evidence_snapshot"]["recalled_episodes"] == recalls
    assert ctx.working_memory["prompt_evidence_snapshot"]["system_budget"]["omitted_recalled_episode_indexes"] == []


def test_required_context_still_over_budget_rejects_without_cutting_it():
    messages, ctx = messages_and_context(["可选回忆。"], padding=SYSTEM_BYTE_LIMIT)
    original = deepcopy(messages)
    with pytest.raises(CompactPromptBudgetExceeded) as caught:
        fit_v7_system_budget(messages, ctx)
    assert messages == original
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    assert snapshot["recalled_episodes"] == []
    assert snapshot["system_budget"]["status"] == "local_rejected"
    assert caught.value.actual_bytes == snapshot["system_budget"]["final_bytes"] > SYSTEM_BYTE_LIMIT
    assert snapshot["passed_to_client"] is False


def test_unbound_recall_block_cannot_be_cut_to_make_it_fit():
    messages, ctx = messages_and_context(["完整回忆。"], padding=SYSTEM_BYTE_LIMIT)
    ctx.working_memory["prompt_evidence_snapshot"]["recalled_episodes"] = ["另一份未绑定记录。"]
    with pytest.raises(CompactPromptBudgetExceeded):
        fit_v7_system_budget(messages, ctx)
    assert ctx.working_memory["prompt_evidence_snapshot"]["system_budget"]["omitted_recalled_episode_indexes"] == []


def test_product_presend_overflow_is_durable_local_rejection_and_next_turn_can_run(tmp_path, monkeypatch):
    app, guard, sid, ctx, wires, _ = setup(tmp_path, monkeypatch, outcome="unclear")
    ctx.working_memory["minimal_care_context"] = {f"field{i}": "中" * 500 for i in range(30)}
    try:
        rejected = app.submit(INPUT, "too-large", sid)
        assert not rejected["ok"] and rejected["request_sent"] is False
        assert rejected["error_code"] == "compact_online_system_byte_limit_exceeded"
        row = deepcopy(app._run["turns"][0])
        assert row["status"] == "failed" and row["source"] == "local_request_rejected"
        assert row["error_code"] == "compact_online_system_byte_limit_exceeded"
        assert row["request_sent"] is False and row["requires_review"] is False
        assert row["input_evidence"]["passed_to_client"] is False
        assert row["input_evidence"]["system_budget"]["status"] == "local_rejected"
        assert guard.snapshot()["calls_used"] == 0 and not guard.snapshot()["stopped"]
        assert not wires
        duplicate = app.submit(INPUT, "too-large", sid)
        assert duplicate["duplicate"] and not duplicate["ok"] and not wires
        assert duplicate["error_code"] == rejected["error_code"] and duplicate["request_sent"] is False
        assert app.submit("新的中性输入。", "next-turn", sid)["ok"]
        assert [row["role"] for row in guard.snapshot()["attempts"]] == [
            "online_fast", "confirm_judge", "safety_review"]
        assert len(wires) == 3 and not guard.snapshot()["stopped"]
        assert app._run["turns"][0] == row
        persisted = json.loads(app.run_path.read_text(encoding="utf-8"))
        assert persisted["turns"][0]["request_sent"] is False
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize("error", [ValueError("compact_online_system_byte_limit_exceeded"),
                                   CompactPromptBudgetExceeded(12003)])
def test_error_after_attempt_cannot_be_reclassified_by_name_or_type(tmp_path, monkeypatch, error):
    from aria.engine.policy.max_release import tls_dispatch
    app, guard, sid, ctx, wires, _ = setup(tmp_path, monkeypatch, outcome="unclear")
    previous = tls_dispatch.open_observed
    def failed(request, **options):
        previous(request, **options)
        raise error
    monkeypatch.setattr(tls_dispatch, "open_observed", failed)
    try:
        assert not app.submit(INPUT, "late-error", sid)["ok"]
        state = guard.snapshot()
        assert state["calls_used"] == 1 and state["stopped"]
        assert state["unknown_cost_requests"] == 1
        assert app._run["turns"][0].get("request_sent") is not False
        assert not app.submit("后继输入。", "blocked", sid)["ok"]
        assert len(wires) == 1
    finally:
        app.close()
        guard.close()


@pytest.mark.parametrize("version", ["online_turn_decision_v4_evidence",
    "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic", V7])
def test_only_v7_compacts_empty_care_without_creating_action_authority(care, version):
    from aria.engine.minimal_care_context import build_care_context
    from aria.engine.policy.prompt import build_messages
    from aria.engine.policy.release_prompt import CARE_BOUNDARY
    from aria.engine.policy.semantic_delivery import (
        current_action_receipts, current_action_status, delivery_review_facts,
    )
    from aria.scenes.registry import get_scene
    service, _token, _clock = care
    persona, ctx = context()
    ctx.working_memory["_online_decision_version"] = version
    raw = build_care_context(service, ELDER)
    ctx.working_memory["minimal_care_context"] = deepcopy(raw)
    utterance = "当前完整原话。"
    ctx.add_turn("历史完整原话。", "历史完整回应。")
    before = deepcopy(ctx.working_memory)
    facts_before = delivery_review_facts(ctx, utterance, persona)
    messages = build_messages(get_scene("companion_elder"), persona, ctx, utterance, [],
                              safety_response_instruction_version="compact_response_v1")
    system = messages[0]["content"]
    blocks = system.split("\n\n")
    view = next(json.loads(block.split("：", 1)[1]) for block in blocks
                if block.startswith("只读护理快照："))
    expected = {key: value for key, value in raw.items()
                if version != V7 or key not in {*raw["counts"], "scope_notice"}}
    if version == V7:
        assert {kind: view["counts"]["each"] for kind in view["counts"]["kinds"]} == raw["counts"]
        expected["counts"] = view["counts"]
    assert view == expected
    assert CARE_BOUNDARY in blocks
    assert ctx.working_memory["minimal_care_context"] == before["minimal_care_context"] == raw
    assert messages[1:] == [{"role": "user", "content": "历史完整原话。"},
        {"role": "assistant", "content": "历史完整回应。"}, {"role": "user", "content": utterance}]
    facts_after = delivery_review_facts(ctx, utterance, persona)
    assert facts_after["owner_sources"] == [utterance, "历史完整原话。"]
    assert {key: value for key, value in facts_after.items() if key != "owner_sources"} == {
        key: value for key, value in facts_before.items() if key != "owner_sources"}
    assert facts_after["current_action_results"] == []
    assert current_action_receipts(ctx, utterance, persona) == []
    assert current_action_status(ctx, utterance, {"action": "notification", "target": ""}, persona) is None
    assert SYSTEM_BYTE_LIMIT == 12000
