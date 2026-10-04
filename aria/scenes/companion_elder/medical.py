"""Final-response guard for unsafe medical claims in the elder scene.

The model may discuss symptoms and encourage professional assessment, but it
must not turn a conversational reply into a diagnosis, a prescription change,
or a treatment guarantee.  This guard runs on the final generated text; it is
therefore independent of whether the user mentioned health in the current
turn.
"""
from __future__ import annotations

import re


SAFE_MEDICAL_BOUNDARY_REPLY = (
    "我不能替医生下诊断，也不能建议自行加减、停换药或保证治疗效果。"
    "先不要自行调整；请尽快联系开药医生或药师核实。若有胸痛、呼吸困难、"
    "突然口角歪斜或说话不清等明确急症，请立即拨打 120。"
)

_DISEASE_TERM = (
    r"(?:高血压|低血压|糖尿病|心肌梗死|心梗|脑梗|脑卒中|中风|"
    r"冠心病|心脏病|心衰|肺炎|慢阻肺|抑郁(?:症)?|焦虑(?:症)?|"
    r"双相情感障碍|帕金森(?:病)?|癌症|肿瘤|感染|胃炎|关节炎|"
    r"阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"
)

_DIAGNOSIS_CLAIMS = (
    re.compile(
        r"(?:你|您).{0,8}(?:有|患上|患了).{0,6}" + _DISEASE_TERM
    ),
    re.compile(
        r"(?:这是|这属于|这些表现是|这些症状是|这个情况是)"
        r".{0,5}(?:典型的?|明确的?)?.{0,4}" + _DISEASE_TERM
    ),
    re.compile(
        r"(?:你|您|你的情况|您的情况|这个情况|这些表现|这个症状|这些症状|这)"
        r".{0,12}(?:就是|肯定是|一定是|已经是|属于|确诊为?|诊断为?|"
        r"患有|得了|符合|说明是|证明是).{0,8}" + _DISEASE_TERM
    ),
    re.compile(
        r"(?:根据|从).{0,20}(?:可以|足以|能够).{0,6}(?:确诊|诊断)"
        r".{0,8}" + _DISEASE_TERM
    ),
)

_MEDICATION_ADJUSTMENT_CLAIMS = (
    # Direct imperatives, for example "把降压药剂量加倍".
    re.compile(
        r"^(?:你|您)?(?:现在|马上|先|直接)?把.{0,16}"
        r"(?:药|药物|剂量).{0,8}(?:停掉|停了|换掉|换成|加倍|减半|"
        r"加量|减量|增加|减少|改成)"
    ),
    # Advice or permission to change a medicine or dose.
    re.compile(
        r"(?:建议|应该|可以|需要|最好|不妨|先|直接|赶紧).{0,18}"
        r"(?:停药|停掉.{0,6}药|换药|把.{0,8}药换成|加量|减量|"
        r"剂量.{0,5}(?:加倍|减半|增加|减少)|多吃.{0,5}药|少吃.{0,5}药)"
    ),
    # Negative wording can itself be an unsafe stop instruction.
    re.compile(r"(?:不用|不必|别)(?:再)?(?:吃|服|用).{0,8}(?:药|药物)"),
    re.compile(
        r"(?:停药|停掉.{0,8}药|停用.{0,8}药|停止服用.{0,8}药|"
        r"换药|加量|减量|剂量加倍|剂量减半|药量加倍|药量减半)"
        r".{0,10}(?:就行|即可|看看|观察|试试)"
    ),
    re.compile(
        r"(?:今天|明天|今晚|下次|这次|以后|现在|先|直接)?"
        r"(?:再)?(?:吃|服|用).{0,8}(?:双倍|两倍|加倍|两片|多一片)"
    ),
)

_TREATMENT_GUARANTEES = (
    re.compile(
        r"(?:保证|百分之百|肯定|一定|绝对).{0,14}"
        r"(?:治好|治愈|根治|康复|有效|不会复发|没事)"
    ),
    re.compile(
        r"(?:吃|服|用).{0,10}(?:就能|一定能|肯定能).{0,10}"
        r"(?:治好|治愈|根治|康复)"
    ),
    re.compile(
        r"(?:药|药物|治疗|疗法|保健品).{0,12}"
        r"(?:一定有效|保证有效|包治|能根治)"
    ),
)

_CLAUSE_SPLIT = re.compile(
    r"[。！？；;，,\n]|(?:但|但是|不过|然而|可是|所以|因此|可见|这说明)"
)
_LOCAL_REFUSAL = re.compile(
    r"(?:不要|不能|无法|不可|不应|不可以|不建议|不需要|别|切勿|严禁|"
    r"不可能|不代表|不等于|没有证据(?:表明|证明)?|没有办法)$"
    r"|不$"
)
_NEGATED_DIAGNOSIS = re.compile(
    r"(?:不是|并非|没有|没|未|无|不代表|不等于).{0,10}" + _DISEASE_TERM
)


def _claim_is_explicitly_refused(clause: str, match: re.Match[str]) -> bool:
    """Allow a local safety boundary without licensing a later unsafe claim."""
    matched = match.group(0)
    if _NEGATED_DIAGNOSIS.search(matched):
        return True
    if re.search(
        r"(?:不要|不能|不可|不应|别|切勿|严禁).{0,14}"
        r"(?:自行|擅自|随意|私自|补服|加量|减量|加减量|加倍|减半|"
        r"双倍|两倍|停药|停掉|停用|换药|停换药)",
        matched,
    ):
        return True
    if any(
        phrase in matched
        for phrase in (
            "不要自行", "不能自行", "不可自行", "不应自行", "别自行",
            "不要擅自", "不能擅自", "不可擅自", "不应擅自", "别擅自",
            "不要随意", "不能随意", "不可随意", "不应随意", "别随意",
            "不要私自", "不能私自", "不可私自", "不应私自", "别私自",
            "不能替医生诊断", "无法替医生诊断", "不能诊断", "无法诊断",
            "不能保证", "无法保证", "不可能保证", "不能治愈", "无法治愈",
        )
    ):
        return True
    prefix = clause[max(0, match.start() - 18):match.start()]
    return _LOCAL_REFUSAL.search(prefix) is not None


def guard_general_medical_reply(response: str) -> str:
    """Replace unsafe diagnosis, medication-change, or cure claims.

    Matching is clause-local so a disclaimer before ``但是`` or punctuation
    cannot be used to smuggle an affirmative diagnosis into the same reply.
    """
    compact = " ".join((response or "").strip().split())
    if not compact:
        return compact
    clauses = [
        item.strip()
        for item in _CLAUSE_SPLIT.split(compact)
        if item.strip()
    ]
    patterns = (
        *_DIAGNOSIS_CLAIMS,
        *_MEDICATION_ADJUSTMENT_CLAIMS,
        *_TREATMENT_GUARANTEES,
    )
    for clause in clauses:
        for pattern in patterns:
            match = pattern.search(clause)
            if match is not None and not _claim_is_explicitly_refused(
                clause, match
            ):
                return SAFE_MEDICAL_BOUNDARY_REPLY
    return compact


__all__ = ["SAFE_MEDICAL_BOUNDARY_REPLY", "guard_general_medical_reply"]
