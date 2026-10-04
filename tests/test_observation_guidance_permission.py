"""Prompt-only permission regressions; no real model or revised quality gold."""
from __future__ import annotations

from copy import deepcopy

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.online_decision import single_pass_messages
from aria.engine.policy import prompt as prompt_module
from aria.engine.policy.prompt import build_messages, permitted_intervention_prompt
from aria.engine.policy.reply_contract import reply_quality_contract
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate
from aria.scenes.registry import get_scene


def _context(mode, *, space=False, physical=False, optional_invitation=False):
    ctx = EngineContext(user_id="synthetic-permission", scene_id="companion_elder")
    candidate = InterventionCandidate(
        id="custom_fact_check", label="待核对方向",
        goal="用一个具体事实问题区分原因",
        guidance="承认不确定，只问一个能核对眼前事实的具体问题。",
        avoid="无依据确定结论", score=.73,
    )
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.selected = candidate
    ctx.reasoning.observation_plan = ActiveObservationPlan(
        mode=mode, question="这个不舒服已经持续多久了？" if mode == "ask" else "",
        hypothesis_ids=["grief_or_transition"] if optional_invitation else [],
    )
    ctx.working_memory["personality_decision"] = {
        "current_physical_evidence_qualified": physical,
        "evidence": [{
            "kind": "response_hypothesis", "status": "working", "dimension": "low_pressure",
            "claim": "先留白，不要求解释。",
        }] if space else [],
    }
    return ctx


@pytest.mark.parametrize(("mode", "space", "optional_invitation"), [
    ("observe", False, False), ("observe", True, False),
    ("wait", False, False), ("ask", False, True),
])
def test_both_prompt_locations_defer_unpermitted_question_directives_without_changing_ledger(
    mode, space, optional_invitation,
):
    ctx = _context(mode, space=space, optional_invitation=optional_invitation)
    before = deepcopy(ctx.reasoning)
    base = build_messages(get_scene("companion_elder"), None, ctx, "今天想安静一会儿。")
    integrated = single_pass_messages(base, ctx)
    for text in (base[0]["content"], integrated[0]["content"]):
        assert "候选方向未获当前提问许可" in text
        assert "只问一个能核对眼前事实" not in text
        assert "用一个具体事实问题区分原因" not in text
    assert '"id":"custom_fact_check"' in integrated[0]["content"]
    assert ctx.reasoning == before
    assert ctx.reasoning.selected.score == .73
    assert ctx.reasoning.selected.guidance == "承认不确定，只问一个能核对眼前事实的具体问题。"


@pytest.mark.parametrize("physical", [False, True])
def test_permitted_fact_and_qualified_physical_clarification_keep_original_guidance(physical):
    ctx = _context("ask", physical=physical)
    fields = permitted_intervention_prompt(ctx.reasoning.selected, ctx)
    assert fields["guidance"] == ctx.reasoning.selected.guidance
    text = single_pass_messages(
        build_messages(get_scene("companion_elder"), None, ctx, "有个事实还没说清。"), ctx,
    )[0]["content"]
    assert "只问一个能核对眼前事实" in text
    assert "候选方向未获当前提问许可" not in text
    assert "本轮唯一允许的问句:这个不舒服已经持续多久了？" in text


def test_non_question_identity_guidance_is_not_removed_just_because_observation_is_quiet():
    ctx = _context("observe", space=True)
    candidate = InterventionCandidate(
        id="custom_identity_direction", label="承接经验",
        goal="尊重本人经历", guidance="接住他已说出的经历，不替他判断能力，也不必找新任务。",
        avoid="不必问一个额外问题", score=.81,
    )
    assert permitted_intervention_prompt(candidate, ctx) == {
        "id": candidate.id, "label": candidate.label, "goal": candidate.goal,
        "guidance": candidate.guidance, "avoid": candidate.avoid,
    }


def test_a_negative_question_instruction_does_not_remove_other_valid_guidance():
    ctx = _context("observe")
    candidate = InterventionCandidate(
        id="custom_companionship", label="陪伴", goal="允许沉默",
        guidance="不要问一个额外问题。承接本人原话，让他自己决定是否继续。",
    )
    assert permitted_intervention_prompt(candidate, ctx)["guidance"] == candidate.guidance


def test_zero_budget_overrides_a_nominal_exact_requirement_in_the_guidance_renderer(monkeypatch):
    ctx = _context("ask", physical=True)
    contract = reply_quality_contract(ctx)
    assert contract["question_requirement"] == "exact"
    contract["question_budget"] = 0
    monkeypatch.setattr(prompt_module, "reply_quality_contract", lambda _: contract)
    assert "候选方向未获当前提问许可" in permitted_intervention_prompt(ctx.reasoning.selected, ctx)["guidance"]


@pytest.mark.parametrize("mode", ["ask", "observe", "wait"])
@pytest.mark.parametrize("space", [False, True])
@pytest.mark.parametrize("physical", [False, True])
def test_current_contract_never_combines_exact_requirement_with_zero_budget(mode, space, physical):
    contract = reply_quality_contract(_context(mode, space=space, physical=physical))
    assert not (contract["question_requirement"] == "exact" and contract["question_budget"] == 0)
