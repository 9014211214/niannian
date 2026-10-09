"""多维相处画像：每个明显偏离中性的特质各自决定一件“怎么说”的事。

只检查本地确定性组装，不代表模型会按画像说得更好。
"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.model.persona import PersonaProfile, PersonHypothesis
from aria.engine.policy.prompt import (
    _MODE_GUIDANCE,
    build_messages,
    confirmed_prompt_hypotheses,
    delivery_basis_ids,
    delivery_profile,
    render_delivery_block,
    select_delivery_mode,
    update_delivery_state,
)
from aria.scenes.registry import get_scene


def _hypothesis(identifier, *, framework="disc", dimension="dominance",
                target=1.0, status="stable", confidence=1.0, tags=None):
    return PersonHypothesis(
        id=identifier, framework=framework, dimension=dimension,
        claim="本地测试假设", target_value=target, confidence=confidence,
        status=status, scope_tags=list(tags or []),
        first_observed_at="2026-09-01T09:00:00+08:00",
    )


def test_default_persona_has_no_profile_and_keeps_neutral_text():
    persona = PersonaProfile(user_id="cold")
    assert delivery_profile(persona) == []
    assert delivery_basis_ids(persona, block="profile_v1") == []
    assert render_delivery_block(persona, block="profile_v1") == _MODE_GUIDANCE["neutral"]
    assert render_delivery_block(None, block="profile_v1") == _MODE_GUIDANCE["neutral"]


def test_laochen_profile_keeps_several_axes_instead_of_one_mode():
    persona = load_demo_persona("laochen_demo")
    assert select_delivery_mode(persona) == "low_pressure"
    axes = [item["axis"] for item in delivery_profile(persona)]
    assert axes == ["pace", "agency", "structure", "closeness", "warmth"]
    block = render_delivery_block(persona, block="profile_v1")
    assert block.startswith("【相处画像:低压】")
    for axis in axes:
        assert f"[{axis}]" in block
    # 画像只说“怎么说”，不向在线模型报人格分数。
    assert "0." not in block
    for item in delivery_profile(persona):
        assert item["drivers"]
        assert all(driver["sources"] == ["baseline"] for driver in item["drivers"])


def test_general_hypothesis_moves_trait_and_is_cited_as_source():
    persona = PersonaProfile(
        user_id="direct",
        person_hypotheses=[_hypothesis("h_direct", tags=["general"])],
    )
    profile = {item["axis"]: item for item in delivery_profile(persona)}
    assert profile["directness"]["level"] == "direct"
    assert profile["agency"]["level"] == "offer_choices"
    driver = profile["directness"]["drivers"][0]
    assert driver["sources"] == ["h_direct"]
    assert driver["value"] == 0.82
    assert delivery_basis_ids(persona, block="profile_v1") == ["directness", "agency"]


def test_scoped_or_candidate_hypothesis_does_not_change_global_profile():
    persona = PersonaProfile(
        user_id="scoped",
        person_hypotheses=[
            _hypothesis("h_family", tags=["family"]),
            _hypothesis("h_candidate", status="candidate", tags=["general"]),
        ],
    )
    assert delivery_profile(persona) == []


def test_multiple_axes_can_disagree_with_the_single_mode_label():
    persona = PersonaProfile(user_id="mixed")
    persona.disc.steadiness = 0.8
    persona.big_five.agreeableness = 0.8
    persona.big_five.neuroticism = 0.8
    persona.big_five.openness = 0.2
    levels = {item["axis"]: item["level"] for item in delivery_profile(persona)}
    assert select_delivery_mode(persona) == "low_pressure"
    assert levels == {
        "pace": "slow_short",
        "reassurance": "acknowledge_first",
        "novelty": "familiar",
        "refusal_exit": "easy_out",
    }


def test_hysteresis_keeps_committed_mode_text_until_switch_is_confirmed():
    persona = PersonaProfile(
        user_id="hysteresis",
        person_hypotheses=[_hypothesis("h_direct", tags=["general"])],
    )
    update_delivery_state(persona)  # bind commitment to the current source pool
    persona.delivery_state.committed = "low_pressure"
    assert render_delivery_block(persona, block="profile_v1") == _MODE_GUIDANCE["low_pressure"]
    assert delivery_basis_ids(persona, block="profile_v1") == []


    persona = load_demo_persona("laochen_demo")
    scene = get_scene("companion_elder")
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    block = render_delivery_block(persona, block="profile_v1")
    reply = build_messages(scene, persona, ctx, "今天有点闷。", delivery_block="profile_v1")
    assert block in reply[0]["content"]


def test_prompt_hypotheses_are_the_last_four_working_or_stable():
    persona = PersonaProfile(
        user_id="many",
        person_hypotheses=[
            _hypothesis(f"h{index}", status="working" if index % 2 else "stable")
            for index in range(6)
        ] + [_hypothesis("h_retired", status="retired")],
    )
    assert [item.id for item in confirmed_prompt_hypotheses(persona)] == [
        "h2", "h3", "h4", "h5"
    ]
    assert confirmed_prompt_hypotheses(None) == []
