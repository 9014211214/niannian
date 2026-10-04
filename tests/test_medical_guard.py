"""Final-response medical safety for the elder companion scene."""
from __future__ import annotations

import pytest

from aria.engine.context import EngineContext
from aria.engine.orchestrator import Engine
from aria.engine.risk import safety_response
from aria.interfaces.risk import RiskSignal
from aria.scenes.companion_elder.medical import (
    SAFE_MEDICAL_BOUNDARY_REPLY,
    guard_general_medical_reply,
)
from aria.scenes.registry import get_scene


@pytest.mark.parametrize(
    "unsafe",
    [
        "你这就是高血压，不用去看医生了。",
        "你有高血压。",
        "这是典型的心梗。",
        "我不能替医生诊断，不过你这就是抑郁症。",
        "不用咨询医生，先停药观察两天。",
        "停掉降压药就行。",
        "明天吃双倍。",
        "你把降压药剂量加倍，应该会好。",
        "把现在的药换成阿司匹林就行。",
        "这个保健品保证能治好你的病。",
        "根据这几句话已经足以确诊帕金森病。",
    ],
)
def test_general_medical_guard_replaces_unsafe_claims(unsafe: str):
    assert guard_general_medical_reply(unsafe) == SAFE_MEDICAL_BOUNDARY_REPLY


@pytest.mark.parametrize(
    "safe",
    [
        "不要自行停药或加减量，请联系开药医生。",
        "这不是典型心梗，需要由医生评估。",
        "不要停掉药，请先联系开药医生。",
        "明天不要吃双倍，请按原处方联系药师核实。",
        "你不应该先停药，请联系开药医生。",
        "你不可以直接换药，请先问处方医生。",
        "我不能替医生诊断，这些表现不等于高血压。",
        "任何治疗都不能保证治愈。",
        "如果吃药后不舒服，请联系开药医生或药师核实。",
        "这些变化可能有多种原因，是否需要评估由医生判断。",
    ],
)
def test_general_medical_guard_preserves_safe_boundaries(safe: str):
    assert guard_general_medical_reply(safe) == safe


def test_final_guard_runs_even_when_current_turn_was_not_medical():
    class _UnsafeFastModel:
        def complete(self, _messages):
            return "你把药量减半就行，不用问医生。"

    scene = get_scene("companion_elder")
    engine = Engine(scene, llm_client=_UnsafeFastModel())
    output = engine.process_turn(
        EngineContext(user_id="u", scene_id=scene.id),
        "今天天气不错。",
    )
    assert output == SAFE_MEDICAL_BOUNDARY_REPLY


@pytest.mark.parametrize(
    ("kind", "tier"),
    [
        ("medical_emergency_stroke", 3),
        ("medical_emergency_breathing_or_consciousness", 3),
        ("medical_urgent_chest", 2),
        ("immediate_safety_possible_duplicate_medication", 3),
    ],
)
def test_general_medical_guard_preserves_deterministic_emergency_reply(
    kind: str,
    tier: int,
):
    original = safety_response(RiskSignal(tier=tier, kind=kind))
    assert guard_general_medical_reply(original) == original


def test_cognitive_guard_remains_more_specific_for_cognitive_diagnosis():
    scene = get_scene("companion_elder")
    output = scene.guard_response(None, "你这就是阿尔茨海默病。")
    assert "不能通过聊天预测、诊断" in output
    assert output != SAFE_MEDICAL_BOUNDARY_REPLY
