"""Explicit, bounded v4 delivery candidate; source/shape is not semantic proof.

Fixed Chinese content was independently reviewed with required revisions.
No source-text classification, raw-response concatenation or action permission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ..policy.online_decision import EVIDENCE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_EVIDENCE
from .safety_evidence import assess_evidence

LEGACY = "legacy"
CONDITIONAL_CONTEXT_V1 = "conditional_context_v1"
TASK_CONTEXT_V2 = "task_context_v2"
SOURCE_RESPONSE_V3 = "source_response_v3"
SOURCE_RESPONSE_V4 = "source_response_v4"
MINIMAL_RESPONSE_V1 = "minimal_response_v1"
VERSIONS = (LEGACY, CONDITIONAL_CONTEXT_V1, TASK_CONTEXT_V2, SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1)


def validate_delivery_version(value):
    if type(value) is not str or value not in VERSIONS:
        raise ValueError("invalid_safety_delivery_version")
    return value


@dataclass(frozen=True)
class SafetyDeliveryContext:
    user_id: str
    session_id: str
    scene_id: str
    turn_index: int
    utterance_sha256: str
    online_version: str
    concern: str
    quote: str
    quote_sha256: str
    subject: str
    time: str
    urgency: str
    disposition: str
    reason: str
    source: str = "online_model_semantics"
    semantic_verified: bool = False
    grants_permission: bool = False


def capture_delivery_context(ctx, utterance, *, delivery_version, online_version, persona):
    """Copy only the existing same-turn, sourced v4 assessment; never infer facts."""
    if (delivery_version not in (CONDITIONAL_CONTEXT_V1, TASK_CONTEXT_V2, SOURCE_RESPONSE_V3, SOURCE_RESPONSE_V4, MINIMAL_RESPONSE_V1)
            or online_version not in EVIDENCE_DECISION_VERSIONS
            or type(utterance) is not str
            or any(type(value) is not str or not value
                   for value in (ctx.user_id, ctx.session_id, ctx.scene_id))
            or (persona is not None and persona.user_id != ctx.user_id)):
        return None
    trace = ctx.working_memory.get("online_turn_decision")
    if (type(trace) is not dict or trace.get("version") != online_version
            or trace.get("status") != "parsed"
            or type(trace.get("turn_index")) is not int
            or trace["turn_index"] != len(ctx.turns)
            or trace.get("safety_source") != "online_model_semantics"
            or ctx.working_memory.get("_online_decision_version") != online_version):
        return None
    concern, row = trace.get("safety_concern"), trace.get("safety_evidence")
    assessment = assess_evidence(concern, row, utterance)
    if (concern not in ("self_harm", "medical_emergency", "fall", "unsafe_medication")
            or assessment.get("verbatim_source") is not True
            or assessment["reason"] not in ("model_current_immediate_with_source",
                                            "current_subject_or_urgency_not_established")
            or trace.get("safety_assessment") != assessment):
        return None
    return SafetyDeliveryContext(
        user_id=ctx.user_id, session_id=ctx.session_id, scene_id=ctx.scene_id,
        turn_index=len(ctx.turns), utterance_sha256=hashlib.sha256(utterance.encode("utf-8")).hexdigest(),
        online_version=online_version, concern=concern, quote=row["quote"],
        quote_sha256=assessment["quote_sha256"], subject=row["subject"], time=row["time"],
        urgency=row["urgency"], disposition=assessment["disposition"], reason=assessment["reason"],
    )


# Audited complete paragraphs; only the fixed subject placeholder is substituted.
SEGMENTS = {
    "U-owner / U-other": "如果这涉及现实中需要紧急处理的危险，请立即联系当地急救服务；接下来的聊天不要耽误求助。",
    "T-recent-resolved": "刚缓解不一定意味着风险已经过去；是否仍需紧急处理，要看发生了什么和专业人员的判断。",
    "T-historical": "如果说的是更早的事，不能仅凭已经过去就判断安全；请按下面各类情况的条件求助。过去未评估的异常，也应向专业人员说明。",
    "T-quotation": "如果这里只是在引述或讨论假设，不表示现实中有人正遇险。下面只说明这些情况真实发生时的安全做法。",
    "T-uncertain": "我还不能确定这件事是否正在发生，下面的行动需要按实际情况适用。",
    "Q-unknown-subject": "说的是你、另一个人，还是引述或假设？",
    "Q-unknown-time": "这件事正在发生、刚缓解，还是更早的事？",
    "Q-unknown-urgency": "现在有没有需要马上处理的受伤、身体异常或伤害自己的危险？",
    "B-capability": "我不能在现场查看情况，也没有代你呼叫或通知任何人。",
    "A-self-harm": "如果 S 已经受伤、服下可能有害的东西，或可能马上伤害自己，请立即联系当地急救服务，不要等聊天结束。若有自伤念头但没有眼前危险，也可以尽快联系可信的人或专业心理支持，直接说明正在经历什么。",
    "P-self-harm-owner": "若这些情况此刻涉及你，先远离可能用于伤害自己的物品或危险地点；能安全联系到可信的人时，请其来陪伴或协助求助。若危险迫近，立即联系当地急救服务，不必等有人回复。",
    "P-self-harm-other": "若这些情况此刻涉及现实中的人，你在现场且自身安全时，可以平静地直接询问对方是否有自杀或伤害自己的想法，倾听、陪伴并协助求助；不要为了拿走危险物品而争抢或让自己陷入危险。若你不在现场且对方可能马上伤害自己、已受伤需要急救或疑似中毒，请立即联系当地急救服务；可同时联系能到场的可信的人协助，不要等其回复才求助，也不要假定已经有人照看。",
    "P-self-harm-unknown / quotation": "若涉及现实中的人，能安全做到时，让当事人与危险物品保持距离，并连接能实际提供帮助的人；不要假定你在现场或由你承担陪护。",
    "A-medical-emergency": "若 S 现在呼吸困难、意识不清、明显大量出血，或突然说话不清、身体一侧无力等，请立即联系当地急救服务，说明实际症状和开始时间。突然出现的说话不清、一侧无力等即使刚消退，也应立即联系急救服务；若是更早发生、已经过去且当时未就医，应尽快联系医生安排紧急评估。",
    "P-medical-owner": "若正有急迫身体异常，不要自己开车去就医；能安全联系到附近可信的人时，请其协助。按急救接线员的指导行动，不靠聊天判断已经安全。",
    "P-medical-other / unknown / quotation": "若这些紧急情况真实发生，请立即联系当地急救服务。你在现场且安全时，按接线员指导查看当事人的反应及是否正常呼吸；若无反应且没有正常呼吸，立即告知接线员并按其指导进行心肺复苏，使用可取得的自动体外除颤器；有相应训练时立即按训练施救，不要等聊天回复。你不在现场时，可同时联系能到场的可信的人协助，不要等其回复才求助，只提供自己确知的信息。",
    "A-fall": "如果跌倒后有意识异常、呼吸异常、明显大量出血，或怀疑头部、颈部、背部严重受伤，请立即联系当地急救服务；即使跌倒已过去一段时间，现在出现这些情况也应立即求助。症状刚缓解也不能据此排除伤情；过去跌倒后的其他不适若仍在持续、加重或再次出现，应尽快由专业人员评估。",
    "P-fall-owner": "若你刚跌倒且仍疼痛、头晕或不确定伤在哪里，先不要急着站起，也不要让人强拉你；呼叫能实际到场的人协助。若所在位置有火灾、车流等直接危险，应在不增加危险的前提下优先脱险并立即求助，不要为了等候指示而继续留在危险中。",
    "P-fall-other / unknown / quotation": "若有人真实跌倒且你在现场，先保证环境和自身安全；怀疑头颈背部受伤时，不要随意搬动或扶起当事人。若有火灾、车流等直接危险，或必须移动才能进行必要救命急救，不应机械套用原地不动；立即求助，尽量减少移动且不让自己陷入危险。若你不在现场且存在上述紧急情况，请立即联系当地急救服务，也可联系能到场的人协助，不要等其回复才求助。",
    "A-unsafe-medication": "如果现在怀疑 S 可能误服、过量服药或接触有害物，请立即联系当地中毒咨询专业服务或急救服务，不要因为暂时没有不适或刚缓解就等待。若只是核实常规用药安排，请向医生或药师确认，不要凭这段回复自行停药、补服、加量、混用或改变原来的用药安排。若怀疑误服，下一次如何用药也应及时询问专业人员。",
    "P-medication-owner": "若目前怀疑误服或中毒，不要自行催吐；除非专业人员有明确指导，不要自行用食物、饮水或其他药物处理。能安全取得时保留药盒或包装，并把确知的药名、用量和时间告诉专业人员，不知道的不要猜。若需要就医，不要自己开车；若意识、呼吸出现异常，立即联系急救服务。",
    "P-medication-other / unknown / quotation": "若目前怀疑现实中的人误服或中毒，不要催吐；除非专业人员有明确指导，不要自行给其食物、饮水或其他药物。能安全取得时保留包装，把已知物质、用量和时间交给专业人员。若当事人反应或呼吸异常，请立即联系急救服务；你在现场且安全时按接线员指导施救，有相应训练时立即按训练施救。你不在现场时，可同时联系能到场的人协助，但不要等其回复才联系专业服务，也不要假定当事人清醒或有人陪护。"
}



def render_safety_delivery(context):
    """Called only after Engine rebinds the immutable context to its current turn."""
    ids = []
    if context.disposition in ("urgent_owner", "urgent_other"):
        ids.append("U-owner / U-other")
    if context.subject == "quotation_or_hypothetical":
        ids.append("T-quotation")
    time_segment = {"recent_resolved": "T-recent-resolved", "historical": "T-historical",
                    "uncertain": "T-uncertain"}.get(context.time)
    if time_segment:
        ids.append(time_segment)
    family = {"self_harm": "self-harm", "medical_emergency": "medical-emergency",
              "fall": "fall", "unsafe_medication": "unsafe-medication"}[context.concern]
    ids.append("A-" + family)
    if context.concern == "self_harm":
        protection = {"owner": "P-self-harm-owner", "other": "P-self-harm-other"}.get(
            context.subject, "P-self-harm-unknown / quotation")
    else:
        family = {"medical_emergency": "medical", "fall": "fall",
                  "unsafe_medication": "medication"}[context.concern]
        protection = "P-" + family + ("-owner" if context.subject == "owner"
                                        else "-other / unknown / quotation")
    ids.append(protection)
    for field, segment in (("subject", "Q-unknown-subject"), ("time", "Q-unknown-time"),
                           ("urgency", "Q-unknown-urgency")):
        if getattr(context, field) == "uncertain":
            ids.append(segment)
    ids.append("B-capability")
    subject = {"owner": "你", "other": "对方", "uncertain": "当事人",
               "quotation_or_hypothetical": "假设中的当事人"}[context.subject]
    return "\n".join(SEGMENTS[key].replace(" S ", subject) for key in ids), tuple(ids)


def render_unverified_safety():
    """No inferred subject, location, concern, time or already-completed action."""
    ids = ("U-owner / U-other", "B-capability")
    return "\n".join((SEGMENTS[ids[0]],
        "我还不能确认这里描述的是现实中的当前情况、过去经历，还是引述或假设。",
        SEGMENTS[ids[1]])), ids


def render_unverified_safety_v2(*, safe_contact=False):
    """Immediate conditional protection without asserting unknown user facts."""
    if safe_contact:
        return ("若眼下有危险，请用当前安全可行的方式求助，避免暴露自己或让协助者进入危险。"
                "我不能代为求助。", ("U-safe-contact",))
    return ("若现实中有人正面临紧急危险，请立即联系当地急救服务，不要等聊天回复；我无法现场救援，也未代为呼叫。", ("U-conditional-v2",))


def render_review_protection(evidence, *, reviewed_fields=False):
    """V5 conditional protection; broad medication labels never assert poisoning.

    No keyword classifies the request. These are conditional safety boundaries
    selected by the already-received model label, not claims about a diagnosis.
    Invalid quotations cannot establish the subject or current urgency.
    """
    concern = evidence["concern"]
    # V2 receives independently reviewed subject/time fields, still model
    # judgments. This permits conditional rendering, not a clinical fact claim.
    sourced = reviewed_fields or evidence.get("verbatim_source") is True
    subject = evidence.get("subject") if sourced else "uncertain"
    time = evidence.get("time") if sourced else "uncertain"
    ids, lines = [], []
    if subject == "quotation_or_hypothetical":
        ids.append("T-quotation"); lines.append(SEGMENTS[ids[-1]])
    if time in {"historical", "recent_resolved"}:
        ids.append("T-" + time.replace("_", "-")); lines.append(SEGMENTS[ids[-1]])
    if concern == "unsafe_medication":
        # A proposal to double a missed dose and an actual ingestion require
        # different actions. Preserve that distinction without lexical routing.
        ids.append("V5-medication-conditional")
        lines.append("不要自行加倍补服，具体漏服安排请核对药品说明并向医生或药师确认。"
                     "如果已经误服、过量或怀疑中毒，请立即联系当地中毒咨询专业服务或急救服务，"
                     "不要自行催吐或用其他药物处理。")
    elif concern == "fall":
        ids.append("V5-fall-conditional")
        lines.append("如果跌倒后仍疼痛、头晕或无法安全起身，先不要尝试站起、跪起或让人强拉；"
                     "呼叫能到场的人，并联系当地急救服务说明伤情。"
                     "怀疑头颈背部受伤时不要随意搬动；若现场有火灾、车流等直接危险，"
                     "在不增加危险的前提下优先脱险并求助。")
        ids.append("A-fall"); lines.append(SEGMENTS[ids[-1]])
    elif concern == "medical_emergency":
        ids.append("A-medical-emergency")
        who = {"owner": "你", "other": "对方"}.get(subject, "当事人")
        lines.append(SEGMENTS[ids[-1]].replace(" S ", who))
        if subject == "owner":
            ids.append("P-medical-owner"); lines.append(SEGMENTS[ids[-1]])
        else:
            ids.append("P-medical-other / unknown / quotation"); lines.append(SEGMENTS[ids[-1]])
            ids.append("V5-medical-ability")
            lines.append("如果你无法按压或无法到场，请马上向急救接线员说明，"
                         "并呼叫能实际施救的人；不要把身体受限理解为无需急救。")
    elif concern == "self_harm":
        ids.append("A-self-harm")
        who = {"owner": "你", "other": "对方"}.get(subject, "当事人")
        lines.append(SEGMENTS[ids[-1]].replace(" S ", who))
        ids.append({"owner": "P-self-harm-owner", "other": "P-self-harm-other"}.get(
            subject, "P-self-harm-unknown / quotation"))
        lines.append(SEGMENTS[ids[-1]])
    else:
        raise ValueError("unsupported_review_protection")
    ids.append("B-capability"); lines.append(SEGMENTS[ids[-1]])
    return "\n".join(lines), tuple(ids)
