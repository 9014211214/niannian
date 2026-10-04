"""安全底线:风险分级筛查 + 确定性接管 + 降速旗 + 安全账本。

全部走 MockLLMClient,无需 key。
"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.model.persona import (
    CarePlan,
    EmergencyContact,
    PersonaProfile,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.llm import MockLLMClient
from aria.engine.risk import HeuristicRiskMonitor, safety_response
from aria.interfaces.reasoning import DeliveredIntervention
from aria.interfaces.risk import RiskSignal
from aria.scenes.base import BaseScene
from aria.scenes.registry import get_scene


def _ctx(uid="u"):
    return EngineContext(user_id=uid, scene_id="companion_elder")


def _engine(persona=None):
    return Engine(get_scene("companion_elder"), persona=persona,
                  llm_client=MockLLMClient())


def _assess(text):
    ctx = _ctx()
    ctx.working_memory["current_utterance"] = text
    return HeuristicRiskMonitor().assess(ctx)


# ── 检测分级 ────────────────────────────────────────────────────
def test_tier3_ideation_detected():
    sig = _assess("有时候真觉得不想活了。")
    assert sig is not None and sig.tier == 3 and sig.kind == "ideation"


def test_missing_you_is_not_ideation():
    assert _assess("哎呀,想死你了,怎么才来。") is None


def test_joking_downgrades_one_tier():
    sig = _assess("一了百了算了——开玩笑的啦。")
    assert sig is not None and sig.tier == 2      # 3 → 2,轻描淡写也值得温和确认


def test_negated_joking_does_not_downgrade_high_distress():
    for utterance in (
        "我不是开玩笑，我真的撑不下去了。",
        "我不是说着玩，我真的熬不下去了。",
    ):
        sig = _assess(utterance)
        assert sig is not None and sig.tier == 2, utterance


def test_tier2_high_distress():
    sig = _assess("我真的撑不下去了。")
    assert sig is not None and sig.tier == 2 and sig.kind == "high_distress"


def test_tier1_needs_accumulation():
    ctx = _ctx()
    mon = HeuristicRiskMonitor()
    ctx.working_memory["current_utterance"] = "最近干什么都没意思。"
    assert mon.assess(ctx) is None                # 单次低落:只计数,不报
    ctx.working_memory["current_utterance"] = "提不起劲。"
    sig = mon.assess(ctx)
    assert sig is not None and sig.tier == 1 and sig.kind == "low_mood"


# ── 引擎接管(tier ≥2):确定性安全回应,含热线与紧急号码 ────────
def test_engine_takeover_contains_hotline():
    persona = PersonaProfile(user_id="u")
    out = _engine(persona).process_turn(_ctx(), "不想活了。")
    assert "12356" in out and "120" in out
    assert not out.startswith("[mock-llm")        # 没走 LLM
    # 落账:安全账本记了一条 tier3
    assert len(persona.risk_events) == 1
    ev = persona.risk_events[0]
    assert ev.tier == 3 and ev.kind == "ideation" and ev.note


def test_urgent_risk_is_not_learned_as_previous_intervention_outcome():
    persona = PersonaProfile(user_id="u")
    engine, ctx = _engine(persona), _ctx()
    ctx.add_turn("最近不太想出门。", "我听见了，咱们慢慢来。")
    ctx.reasoning.delivered_interventions.append(DeliveredIntervention(
        id="quiet_companionship",
        label="安静陪伴",
        goal="降低压力",
        turn_index=0,
    ))

    response = engine.process_turn(ctx, "我现在胸痛，还有点喘不上气。")

    assert "120" in response
    assert ctx.reasoning.outcomes == []


def test_record_once_per_kind_per_session():
    persona = PersonaProfile(user_id="u")
    eng, ctx = _engine(persona), _ctx()
    eng.process_turn(ctx, "不想活了。")
    eng.process_turn(ctx, "真的活不下去。")
    assert len(persona.risk_events) == 1          # 同会话同 kind 只落一条


# ── tier1 不接管:正常回应 + 降速旗 + 落账 ──────────────────────
def test_tier1_does_not_take_over():
    persona = PersonaProfile(user_id="u")
    eng, ctx = _engine(persona), _ctx()
    eng.process_turn(ctx, "最近干什么都没意思。")
    out = eng.process_turn(ctx, "提不起劲。")
    assert "12356" not in out                     # 没被接管
    assert ctx.working_memory["risk_flag"]["tier"] == 1
    assert any(e.kind == "low_mood" for e in persona.risk_events)
    assert ctx.reasoning.observation_plan.mode == "wait"
    assert ctx.reasoning.selected.id == "quiet_companionship"


# ── 安全回应个性化:仅在明确同意过时建议联系紧急联系人 ──────────
def test_safety_response_contact_needs_consent():
    sig = RiskSignal(tier=3, kind="ideation")
    consented = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[
            EmergencyContact(
                name="陈小芳",
                relation="女儿",
                consent_to_notify=True,
                notification_consent_scope="care_and_safety_notifications",
                notification_consent_source="institution_care_plan",
                notification_consented_at="2026-07-01T09:00:00+08:00",
            )]),
    )
    silent = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[
            EmergencyContact(name="陈小芳", relation="女儿", consent_to_notify=False)]),
    )
    assert "陈小芳" in safety_response(sig, consented)
    assert "陈小芳" not in safety_response(sig, silent)


def test_safety_response_never_treats_local_demo_scope_as_real_consent():
    sig = RiskSignal(tier=3, kind="ideation")
    local_demo = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[EmergencyContact(
            name="本地模拟联系人",
            consent_to_notify=True,
            notification_consent_scope="local_demo_care_notifications",
            notification_consent_source="developer_demo_ui_explicit_confirmation",
            notification_consented_at="2026-07-01T09:00:00+08:00",
        )]),
    )
    output = safety_response(sig, local_demo)
    assert "本地模拟联系人" not in output
    assert "你之前同意过" not in output


# ── 默认常驻:BaseScene 的安全网是开着的 ────────────────────────
def test_base_scene_risk_on_by_default():
    assert isinstance(BaseScene().risk_monitor(), HeuristicRiskMonitor)


def test_elder_scene_takes_over_stroke_and_chest_signals_without_llm():
    engine = _engine(PersonaProfile(user_id="u"))
    stroke = engine.process_turn(
        _ctx(), "我刚才突然说话不清，一边手脚没力。"
    )
    assert "120" in stroke and "不要自己开车" in stroke
    assert "12356" not in stroke

    chest = _engine(PersonaProfile(user_id="u")).process_turn(
        _ctx(), "今天突然胸闷，胸口像压着。"
    )
    assert "120" in chest and "不能在聊天里判断" in chest


def test_elder_risk_does_not_take_over_negation_quotes_or_third_person():
    monitor = get_scene("companion_elder").risk_monitor()
    excluded = (
        "我没有胸痛，只是问问胸痛是什么意思。",
        "电视剧里那个老人突然胸痛，演得挺吓人。",
        "我妹妹刚才胸痛，我正陪她等医生。",
        "新闻里说有人活不下去，家人要多留意。",
        "我女儿说她最近真的撑不下去了。",
        "我没有想死，也不会自杀。",
    )
    for utterance in excluded:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is None or signal.tier < 2, utterance


def test_elder_risk_still_takes_over_direct_current_first_person_reports():
    monitor = get_scene("companion_elder").risk_monitor()
    expected = (
        ("我现在胸痛，胸口像压着。", 2),
        ("我现在喘不上气。", 3),
        ("我真的撑不下去了。", 2),
        ("我不想活了。", 3),
        ("我想把药全吃了。", 3),
        ("我已经把绳子准备好了。", 3),
        ("我心口疼得厉害。", 2),
        ("我气都上不来了。", 3),
        ("我突然一边身子不听使唤。", 3),
    )
    for utterance, tier in expected:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is not None and signal.tier == tier, utterance


def test_recent_stroke_signs_remain_emergencies_after_apparent_recovery():
    monitor = get_scene("companion_elder").risk_monitor()
    expected = (
        "我刚才嘴歪了，不过现在已经好了。",
        "刚刚一侧无力、言语不清，过了十分钟缓解了。",
        "我老伴今天突然口角歪，后来恢复了。",
        "刚才嘴角歪、口齿不清，不过这会儿好了。",
    )
    for utterance in expected:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is not None and signal.tier == 3, utterance
        assert signal.kind == "medical_emergency_stroke", utterance


def test_resolved_chest_or_breathing_still_gets_deterministic_medical_route():
    chest = _engine(PersonaProfile(user_id="u")).process_turn(
        _ctx(), "今天胸痛了几分钟，现在已经好了。"
    )
    assert "即使已经缓解" in chest and "尽快联系医生评估" in chest

    breathing = _engine(PersonaProfile(user_id="u")).process_turn(
        _ctx(), "我刚才喘不上气，不过现在已经缓解。"
    )
    assert "即使已经缓解" in breathing and "尽快由医生评估" in breathing
    assert "120" in breathing


def test_resolved_medical_rules_keep_negation_history_and_minor_symptoms_out():
    monitor = get_scene("companion_elder").risk_monitor()
    excluded = (
        "我刚才没有嘴歪，也没有单侧无力，说话一直很清楚。",
        "我没有胸痛，也没有呼吸困难，只是有点累，现在好了。",
        "我去年有过嘴歪和说话不清，早就好了。",
        "我口齿不清那是小时候的事，早就好了。",
        "我上个月胸痛过，已经看过医生，也缓解了。",
        "我老伴前年喘不上气，已经就医了。",
        "刚才走快了有点累，现在已经好了。",
        "刚才有点头晕，现在没事了。",
    )
    for utterance in excluded:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is None or signal.tier < 2, utterance


def test_breathing_that_returns_after_relief_stays_tier_three():
    monitor = get_scene("companion_elder").risk_monitor()
    ctx = _ctx()
    ctx.working_memory["current_utterance"] = (
        "刚才喘不上气，缓解了一会儿，现在又开始喘不上气了。"
    )
    signal = monitor.assess(ctx)
    assert signal is not None and signal.tier == 3
    assert signal.kind == "medical_emergency_breathing_or_consciousness"


def test_old_treated_event_does_not_hide_a_new_event_in_the_same_sentence():
    monitor = get_scene("companion_elder").risk_monitor()
    expected = (
        (
            "我去年胸痛已经看过医生，现在又胸痛了。",
            2,
            "medical_urgent_chest",
        ),
        (
            "我上周嘴歪已经去过医院，今天又嘴歪说话不清了。",
            3,
            "medical_emergency_stroke",
        ),
        (
            "我爸去年嘴歪已经看过医生，现在我嘴歪说话不清。",
            3,
            "medical_emergency_stroke",
        ),
        (
            "我以前呼吸困难已经去医院，现在又喘不上气。",
            3,
            "medical_emergency_breathing_or_consciousness",
        ),
        (
            "我去年胸痛已经看过医生现在又胸痛了。",
            2,
            "medical_urgent_chest",
        ),
    )
    for utterance, tier, kind in expected:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is not None and signal.tier == tier, utterance
        assert signal.kind == kind, utterance


def test_event_local_history_negation_care_and_subject_do_not_leak():
    monitor = get_scene("companion_elder").risk_monitor()
    urgent = (
        "我去年没有胸痛，但现在胸痛得厉害。",
        "我现在胸痛，我女儿已经去医院了。",
        "我爸去年嘴歪已经看过医生，现在我嘴歪说话不清。",
    )
    for utterance in urgent:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is not None and signal.tier >= 2, utterance

    excluded = (
        "我去年胸痛已经看过医生，现在已经没事了。",
        "我爸去年嘴歪已经看过医生，现在我没有嘴歪也没有说话不清。",
        "我妹妹刚才胸痛，我正陪她等医生。",
    )
    for utterance in excluded:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is None or signal.tier < 2, utterance


def test_new_high_recall_phrases_still_respect_negation_and_reference():
    monitor = get_scene("companion_elder").risk_monitor()
    excluded = (
        "我不想把药全吃了，还是按医嘱来。",
        "新闻报道有人割腕，家属及时发现了。",
        "我女儿今天心口疼，已经去医院了。",
        "我不想死，我还想好好活着。",
    )
    for utterance in excluded:
        ctx = _ctx()
        ctx.working_memory["current_utterance"] = utterance
        signal = monitor.assess(ctx)
        assert signal is None or signal.tier < 2, utterance
