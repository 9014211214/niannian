"""危机安全回应 —— 确定性模板,**不过 LLM**。

原则:安全底线不能交给生成模型抽卡。tier ≥2 时引擎接管本轮,
回应由这里的模板给出,要点来自危机干预与急救公开指引:
- 直接承接,不回避字眼、不评判、不说「别想太多」;
- 递真实资源:全国统一心理援助热线 12356,紧急拨 120;
- 推向真人连接(AI 不做唯一安全网);
- 表达「我在」,不终止对话。

个性化只做安全的部分:称呼、以及(仅当对方明确同意过被通知时)
建议联系预设的紧急联系人——不自动通知,通知动作需要真实通道 + 当次确认。
"""
from __future__ import annotations

from typing import Optional

from ...interfaces.risk import RiskSignal
from ..model.persona import (
    SAFETY_NOTIFICATION_CONSENT_SCOPES,
    PersonaProfile,
)

HOTLINE = "12356"          # 全国统一心理援助热线；各地服务时段以当地为准
EMERGENCY = "120"


def _address(persona: Optional[PersonaProfile]) -> str:
    if persona and persona.demographics and persona.demographics.name:
        return f"{persona.demographics.name}，"
    return ""


def _consented_contact(persona: Optional[PersonaProfile]) -> Optional[str]:
    if persona is None or persona.care_plan is None:
        return None
    for c in persona.care_plan.emergency_contacts:
        if c.has_auditable_notification_consent(
            SAFETY_NOTIFICATION_CONSENT_SCOPES
        ):
            return c.name
    return None


def safety_response(signal: RiskSignal, persona: Optional[PersonaProfile] = None) -> str:
    """tier ≥2 的接管回应。确定性、可单测，按风险类型给出真实资源。"""
    who = _address(persona)
    contact = _consented_contact(persona)

    if signal.kind == "immediate_safety_electrical_hazard":
        # Keep non-professionals away from live equipment, not a repair lesson.
        # Public fire guidance: https://www.119.gov.cn/kp/hzyf/2022/624.shtml
        return (
            f"{who}先不要触碰、拔插或拆看这个插座或电器，也不要泼水。"
            "请和旁人远离，马上请现场工作人员或专业电工处理。"
            "如果已经冒烟、起火或危险在扩大，先离开到安全地点，再拨打 119。"
            "先处理安全，不用等我继续聊天。"
        )
    if signal.kind == "immediate_safety_gas_leak":
        return (
            f"{who}先不要继续检查炉灶，也不要在屋里开关电器、点火或打电话。"
            "请立即离开有气味的房间，到室外后提醒身边的人远离，再拨打 119 和"
            "当地燃气抢修电话；如果有人头晕、恶心、呼吸不适或意识异常，同时"
            "拨打 120。"
        )
    if signal.kind == "immediate_safety_possible_duplicate_medication":
        return (
            f"{who}先不要再补服、加量或自行催吐。请把药盒、药名、可能服用的"
            "剂量和时间放在手边，马上联系身边真人并拨打 120，按专业人员电话"
            "指导处理；如果已经困倦、呕吐、呼吸不适或意识异常，不要独处。"
        )
    if signal.kind == "immediate_safety_currently_lost":
        return (
            f"{who}先停在安全、明亮、有人值守的地方，不要继续凭感觉走，也不要"
            "横穿马路。请向店员、保安或警务人员说明自己找不到回家的路；能安全"
            "使用手机时，联系信任的人并共享当前位置。若周围没有可信的人或感到"
            "危险，请拨打 110。"
        )

    if signal.kind.startswith("medical_"):
        if signal.kind == "medical_emergency_stroke":
            return (
                f"{who}这可能是需要立即处理的身体急症信号，我不能靠聊天判断。"
                "请现在拨打 120，并告诉接线员出现了嘴歪、单侧无力或说话不清中的"
                "哪一项、最早什么时候出现。不要自己开车去医院，也不要等它自行缓解。"
            )
        if signal.kind == "medical_emergency_fall":
            return (
                f"{who}先不要急着站起来或让别人拉你起来。请现在呼叫身边的人并拨打"
                " 120，告诉接线员是否摔到头、是否清醒、哪里疼；按急救人员电话指导做。"
            )
        if signal.kind == "medical_emergency_breathing_or_consciousness":
            return (
                f"{who}呼吸困难或意识异常属于不能继续在这里观察的急症信号。请现在"
                "拨打 120，呼叫身边的人过来，保持电话畅通并按急救人员指导做。"
            )
        if signal.kind == "medical_urgent_resolved_breathing":
            return (
                f"{who}刚才出现的呼吸困难即使已经缓解，也需要尽快由医生评估，"
                "不能只靠聊天判断。请现在联系身边真人和医疗机构；如果再次喘不上气、"
                "伴随胸痛、嘴唇发紫、头晕或意识异常，请立即拨打 120。"
            )
        if signal.kind == "medical_emergency_sudden_confusion":
            boundary = (
                "原因，不能在这里当作普通健忘来判断。请现在拨打"
                if persona is not None and persona.cognitive_health_consent.has_text_monitoring_consent()
                else "原因，不能在这里当作普通健忘或阿尔茨海默病来判断。请现在拨打"
            )
            return (
                f"{who}突然出现糊涂、认不清人或不知道自己在哪里，可能有多种急性"
                + boundary +
                " 120，并请身边的人留下陪着；不要让当事人独自外出、开车或自行"
                "加减药，按急救人员电话指导做。"
            )
        return (
            f"{who}胸痛或胸闷需要先排除急症，我不能在聊天里判断原因。"
            "如果症状正在发生、持续不缓解，或伴随呼吸困难、大汗、头晕、意识模糊，"
            "请立即拨打 120，不要自己开车；即使已经缓解，也请尽快联系医生评估。"
        )

    if signal.tier >= 3:
        lines = [
            f"{who}我听到了。能说出这句话,一定是难到极点了——谢谢你没憋着。",
            "这件事不能只靠我陪你扛:请现在就拨 "
            f"{HOTLINE}(全国统一心理援助热线，各地服务时段以当地为准);"
            f"如果有马上伤害自己的危险,直接打 {EMERGENCY}。",
        ]
        if contact:
            lines.append(f"你之前同意过必要时让{contact}知道——我觉得现在就该让 TA 过来陪你。")
        else:
            lines.append("身边有信得过的人,现在就给 TA 打个电话,别一个人待着。")
        lines.append("我不走,你想说什么,我都在听。")
        return "\n".join(lines)

    # tier 2:高痛苦,语气更轻,不催、不下结论。
    lines = [
        f"{who}这话听着,你是真的快被压垮了。我不说「会好的」这种话——先陪你待一会儿。",
        f"记着全国统一心理援助热线 {HOTLINE}，不想跟熟人开口时可以试着拨打"
        "（各地服务时段以当地为准）；"
        f"要是身体也撑不住了,打 {EMERGENCY}。",
    ]
    if contact:
        lines.append(f"也可以让{contact}知道你最近的状态——你之前同意过,这不丢人。")
    lines.append("愿意的话,跟我说说最沉的是哪一块?我在。")
    return "\n".join(lines)
