"""养老场景风险筛查：心理危机基线 + 身体急症确定性分流。

它只识别需要立即交给真人/急救的高风险表述，不判断疾病名称。身体急症规则
来自公开急救信号（卒中、胸痛/胸闷、呼吸困难、意识异常、严重跌倒）。
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from ...engine.context import EngineContext
from ...engine.risk.heuristic import (
    HeuristicRiskMonitor,
    _last_subject,
    asserted_self_report_hit,
)
from ...interfaces.risk import RiskSignal

_STROKE = (
    "嘴歪", "嘴角歪", "口角歪", "口角下垂", "半边脸歪",
    "一侧无力", "单侧无力", "半边没力", "一边胳膊没劲",
    "一边手脚没力", "说话不清", "言语不清", "突然说不出话",
    "口齿不清", "吐字不清", "说话含糊",
    "一边身子不听使唤", "半边身子不听使唤",
    "一边身子动不了", "半边身子动不了", "一只手抬不起来",
    "突然看不清东西", "眼前突然看不清", "突然看不见",
    "突然失去平衡", "突然站不稳", "突然走不稳",
    "突然听不懂别人说话", "突然听不懂你说话", "突然听不懂话",
    "突然剧烈头痛", "突然头痛得厉害",
)
_BREATHING = (
    "呼吸困难", "喘不上气", "透不过气", "快喘不过来了",
    "气上不来", "气都上不来", "上不来气", "喘不过来", "喘不过气",
)
_CONSCIOUSNESS = (
    "意识不清", "失去意识", "人事不省", "昏迷", "晕倒了", "叫不醒",
    "神志不清", "突然答非所问", "神志不清答非所问",
)
_SUDDEN_CONFUSION = (
    "突然糊涂", "突然迷糊", "突然搞不清自己在哪", "突然不知道自己在哪",
    "突然认不出家人", "突然认不出人", "今天突然分不清白天晚上",
    "刚才突然不知道这是哪里", "一下子不知道自己在哪", "一下子认不出人",
)
_CHEST = (
    "胸痛", "胸口痛", "胸闷", "胸口压着", "胸口压榨",
    "心口疼", "心口痛", "心口发紧",
)
_SEVERE_FALL = (
    "摔倒站不起来", "跌倒站不起来", "摔了起不来", "跌了起不来",
    "摔倒后头晕", "摔倒后意识", "摔到头",
)
_CURRENTLY_LOST = (
    "我现在找不到回家的路", "我现在找不到家", "现在不知道怎么回家",
    "我迷路了找不到家", "我现在迷路了", "我走丢了",
)
_GAS_DANGER = (
    "闻到煤气味", "闻到燃气味", "煤气泄漏", "燃气泄漏",
    "煤气没关屋里味道很大", "燃气没关屋里味道很大",
    "忘关煤气屋里有味", "忘关燃气屋里有味",
)
# Only explicit electrical facilities + an asserted physical hazard. Emotional
# "冒火花", an old anecdote, or a request for safety knowledge is not this signal.
_ELECTRICAL_HAZARD_RE = re.compile(
    r"(?:插座|插排|排插|接线板|电箱|配电箱|电线|电器)"
    r"(?:现在|正在|刚才|刚刚|突然|又开始|还在|老是|开始|一直|又|在)?"
    r"(?:冒火花|打火花|漏电)"
)
_ELECTRICAL_SECURED_RE = re.compile(
    r"(?:电工|维修人员|专业人员)(?:现在|刚才|刚刚|今天)?(?:已经|已)"
    r"(?:修好|排除故障|排除危险|完成维修)"
)
_ELECTRICAL_INFORMATIONAL = ("科普", "演练", "比喻", "引用", "台词")
_ELECTRICAL_UNCERTAIN_PREFIX_RE = re.compile(
    r"(?:担心|怕会|会不会|要是)(?:会|有一天|哪天)?$"
)
_ELECTRICAL_ONGOING_RE = re.compile(r"(?:还|仍|又)[^。！？；;，,]{0,5}(?:火花|漏电)")
_POSSIBLE_DUPLICATE_MEDICATION = (
    "药吃了两遍", "药吃了两次", "重复吃药", "多吃了一次药",
    "忘了吃没吃又吃了一次", "把药吃重了", "可能重复服药",
)
_ALL_MEDICAL_EVENT_WORDS = (
    _STROKE
    + _BREATHING
    + _CONSCIOUSNESS
    + _SUDDEN_CONFUSION
    + _CHEST
    + _SEVERE_FALL
    + _CURRENTLY_LOST
    + _GAS_DANGER
    + _POSSIBLE_DUPLICATE_MEDICATION
)

_RECENCY_MARKERS = (
    "现在", "正在", "刚才", "刚刚", "今天", "这会儿", "此刻", "目前",
    "又开始", "又出现",
)
_HISTORICAL_MARKERS = (
    "去年", "前年", "上个月", "上周", "昨天", "昨晚", "前几天", "多年前", "以前",
    "曾经", "那次", "早就", "很久以前", "小时候", "中风后一直", "出院以后一直",
)
_SYMPTOM_RESOLVED_MARKERS = (
    "已经好了", "早就好了", "现在好了", "这会儿好了", "后来好了",
    "已经缓解", "缓解了", "恢复了", "现在没事了",
)
_CARE_SECURED_MARKERS = (
    "已经就医", "已就医",
    "已经看过医生", "已经去看医生", "已经去医院", "已经去过医院",
    "已经送去医院", "已经送去看医生", "等医生", "已经出院",
)
_ONGOING_MARKERS = ("现在还", "仍然", "还在", "又开始", "又出现")
_NON_SELF_PHYSICAL_FRAMES = (
    "问我", "跟我说", "对我说", "告诉我", "我问", "我是说",
    "我说的是", "听说", "据说",
)
_FUTURE_OR_UNCERTAIN_PHYSICAL = (
    "如果", "假如", "假设", "以后", "将来", "会不会", "担心", "怕会",
)
_SUBJECTLESS_PHYSICAL_PREFIX_RE = re.compile(
    r"^(?:(?:今天|现在|这会儿|最近|刚刚|刚才|目前|突然|又|还|"
    r"一直|这两天|这几天|早上|上午|下午|晚上|半夜|起床后|走路时|"
    r"一站起来|坐起来|躺下时|总是|总|就是|感觉|觉得|好像|有点|"
    r"有一点|有点儿|特别|很|挺|真|实在))*"
    r"(?:(?:头|胸|胸口|心口|腰|腿|膝盖|脚|手|胳膊|肚子|胃|背|"
    r"肩膀|身体|浑身|哪里|这儿|那儿)(?:有点|有一点|有点儿|特别|"
    r"很|挺|真|一直|又|还|突然|总是|总)*)?$"
)
_EVENT_BOUNDARY_RE = re.compile(r"[。！？；;，,\n]+")
_UNPUNCTUATED_RESTART_RE = re.compile(
    r"(?:但(?:是)?|不过|可是|而)?"
    r"(?:现在|今天|刚才|刚刚|这会儿|目前)(?:我|他|她|又)"
)


@dataclass(frozen=True)
class _MedicalHit:
    word: str
    resolved: bool


def _event_clauses(utterance: str, words: tuple[str, ...]) -> list[str]:
    """切出风险事件；也兼容 ASR 没有补逗号的“旧事 + 现在复发”。"""
    clauses: list[str] = []
    event_end_markers = (
        _HISTORICAL_MARKERS
        + _SYMPTOM_RESOLVED_MARKERS
        + _CARE_SECURED_MARKERS
    )
    for coarse in _EVENT_BOUNDARY_RE.split(utterance):
        coarse = coarse.strip()
        if not coarse:
            continue
        cursor = 0
        for match in _UNPUNCTUATED_RESTART_RE.finditer(coarse):
            if match.start() <= cursor:
                continue
            before = coarse[cursor:match.start()]
            after = coarse[match.start():]
            if (
                any(marker in before for marker in event_end_markers)
                and any(word in after for word in words)
            ):
                clauses.append(before.strip())
                cursor = match.start()
        clauses.append(coarse[cursor:].strip())
    return [clause for clause in clauses if clause]


def _current_medical_hit(
    utterance: str,
    words: tuple[str, ...],
    *,
    allow_resolved: bool = False,
    allow_third_person: bool = True,
) -> Optional[_MedicalHit]:
    """允许现场人代述急症，同时排除否定、引用和已交给医疗人员的事件。

    ``allow_resolved`` 只表示“症状暂时缓解后仍需分流”，不覆盖明确的旧事或
    已经就医的状态。逐分句检查可避免前面的历史事件遮住后面的新近复发。
    """
    clauses = _event_clauses(utterance, words)
    resolved_candidate: Optional[_MedicalHit] = None
    for index, clause in enumerate(clauses):
        hit = asserted_self_report_hit(
            clause, words, allow_third_person=allow_third_person
        )
        if hit is None:
            continue
        if (
            not allow_third_person
            and index > 0
            and clauses[index - 1].startswith(("不是我", "并非我"))
            and any(word in clauses[index - 1] for word in words)
            and clause.startswith("是")
            and _last_subject(clause) != "self"
        ):
            # “不是我头晕，是老周头晕”中后半句没有通用可枚举的人名，
            # 但由前后对照可以证明身体主语已经从本人移开。
            continue

        # 只把紧邻的“已经缓解/已经就医”等状态补语归到当前事件；一旦下一
        # 小分句出现新的同类风险词，就视为另一个事件。主语冲突时也不借用
        # 状态，避免“我胸痛，我女儿已经就医”把本人的新发症状误判为已处理。
        event_text = clause
        hit_at = clause.find(hit)
        event_subject = _last_subject(clause[:hit_at])
        for following in clauses[index + 1:]:
            following_hit = asserted_self_report_hit(
                following, _ALL_MEDICAL_EVENT_WORDS, allow_third_person=True
            )
            if following_hit is not None:
                break
            is_status = any(
                marker in following
                for marker in (
                    _SYMPTOM_RESOLVED_MARKERS
                    + _CARE_SECURED_MARKERS
                    + _ONGOING_MARKERS
                )
            )
            if not is_status:
                break
            status_subject = _last_subject(following)
            if (
                event_subject is not None
                and status_subject is not None
                and event_subject != status_subject
            ):
                break
            event_text = f"{event_text}，{following}"

        recent = any(marker in event_text for marker in _RECENCY_MARKERS)
        historical = any(marker in event_text for marker in _HISTORICAL_MARKERS)
        symptom_resolved = any(
            marker in event_text for marker in _SYMPTOM_RESOLVED_MARKERS
        )
        care_secured = any(
            marker in event_text for marker in _CARE_SECURED_MARKERS
        )
        ongoing = any(marker in event_text for marker in _ONGOING_MARKERS)

        if care_secured and not ongoing:
            continue
        if historical and not recent:
            continue
        # “去年嘴歪，现在已经好了”里的“现在”描述恢复，不代表新发；只有
        # 明确“又开始/又出现”才把这种历史 + 恢复表述视作当下复发。
        if historical and symptom_resolved and not ongoing:
            continue
        resolved = symptom_resolved and not ongoing
        if resolved and not allow_resolved:
            continue
        candidate = _MedicalHit(word=hit, resolved=resolved)
        if not resolved:
            return candidate
        if resolved_candidate is None:
            resolved_candidate = candidate
    return resolved_candidate


def current_self_physical_hit(
    utterance: str,
    words: tuple[str, ...],
) -> Optional[str]:
    """Return a current, non-negated physical expression about the user.

    This reuses the medical event parser's subject, quotation, hypothetical,
    historical and resolved-event boundaries, but does not perform risk
    grading.  It is suitable only as an eligibility gate for clarification.
    """
    for word in words:
        hit = _current_medical_hit(
            utterance,
            (word,),
            allow_resolved=False,
            allow_third_person=False,
        )
        if hit is None:
            continue
        for clause in _event_clauses(utterance, (word,)):
            start = clause.find(word)
            while start >= 0:
                prefix = clause[:start]
                suffix = clause[start + len(word):]
                subject = _last_subject(prefix)
                report_or_object = any(
                    marker in prefix for marker in _NON_SELF_PHYSICAL_FRAMES
                )
                future_or_uncertain = any(
                    marker in prefix
                    for marker in _FUTURE_OR_UNCERTAIN_PHYSICAL
                )
                modal_question = (
                    prefix.rstrip().endswith(("是不是", "有没有"))
                    or suffix.lstrip().startswith(("吗", "呢", "不" + word))
                )
                if (
                    not report_or_object
                    and not future_or_uncertain
                    and not modal_question
                    and (
                        subject == "self"
                        or (
                            subject is None
                            and _SUBJECTLESS_PHYSICAL_PREFIX_RE.fullmatch(
                                prefix.strip()
                            )
                        )
                    )
                    and asserted_self_report_hit(
                        clause,
                        (word,),
                        allow_third_person=False,
                    )
                ):
                    return word
                start = clause.find(word, start + len(word))
    return None


def _current_electrical_hazard(utterance: str) -> bool:
    """Use the existing assertion/time boundary; do not diagnose or infer intent.

    A recent spark becoming quiet is not proof of electrical isolation. Only an
    explicit, non-negated professional repair of this event closes the signal.
    This bounded safeguard is not comprehensive electrical-hazard recognition.
    """
    words = tuple(match.group() for match in _ELECTRICAL_HAZARD_RE.finditer(utterance))
    if not words:
        return False
    clauses = _event_clauses(utterance, words)
    for index, clause in enumerate(clauses):
        for match in _ELECTRICAL_HAZARD_RE.finditer(clause):
            prefix = clause[:match.start()]
            if (
                any(frame in prefix for frame in _ELECTRICAL_INFORMATIONAL)
                or _ELECTRICAL_UNCERTAIN_PREFIX_RE.search(prefix)
            ):
                continue
            event = clause
            if index > 0 and clauses[index - 1] in _HISTORICAL_MARKERS:
                event = clauses[index - 1] + clause
            if _current_medical_hit(event, (match.group(),), allow_resolved=True) is None:
                continue
            after = clause[match.end():]
            if index + 1 < len(clauses) and not _ELECTRICAL_HAZARD_RE.search(clauses[index + 1]):
                after += "，" + clauses[index + 1]
            secured = _ELECTRICAL_SECURED_RE.search(after)
            repair_tail = after[secured.end():].strip("了 。！？；;，,\n") if secured else ""
            still_hazardous = _ELECTRICAL_ONGOING_RE.search("，".join(clauses[index:]))
            repair_is_uncertain = any(word in after for word in (
                "吗", "呢", "吧", "可能", "不确定", "如果", "假如", "假设", "要是",
            )) or any(mark in utterance for mark in ("？", "?"))
            if (
                secured
                and not repair_tail
                and not still_hazardous
                and not repair_is_uncertain
                and not any(word in after for word in (
                    "邻居的", "别人的", "另一个", "另一处", "还在", "仍然", "又开始", "又出现",
                    *_HISTORICAL_MARKERS,
                ))
                and asserted_self_report_hit(after, (secured.group(),), allow_third_person=True)
            ):
                continue
            return True
    return False


class ElderRiskMonitor:
    """优先筛身体急症，其余交给通用心理风险监测。"""

    def __init__(self) -> None:
        self._mental = HeuristicRiskMonitor()

    def assess(self, ctx: EngineContext) -> Optional[RiskSignal]:
        utterance = str(ctx.working_memory.get("current_utterance") or "")
        if _current_medical_hit(utterance, _STROKE, allow_resolved=True):
            return RiskSignal(
                tier=3,
                kind="medical_emergency_stroke",
                evidence=[utterance],
                confidence=0.78,
            )
        breathing = _current_medical_hit(
            utterance, _BREATHING, allow_resolved=True
        )
        if breathing is not None:
            if breathing.resolved:
                return RiskSignal(
                    tier=2,
                    kind="medical_urgent_resolved_breathing",
                    evidence=[utterance],
                    confidence=0.68,
                )
            return RiskSignal(
                tier=3,
                kind="medical_emergency_breathing_or_consciousness",
                evidence=[utterance],
                confidence=0.78,
            )
        if _current_medical_hit(utterance, _CONSCIOUSNESS):
            return RiskSignal(
                tier=3,
                kind="medical_emergency_breathing_or_consciousness",
                evidence=[utterance],
                confidence=0.78,
            )
        if _current_medical_hit(utterance, _SUDDEN_CONFUSION):
            return RiskSignal(
                tier=3,
                kind="medical_emergency_sudden_confusion",
                evidence=[utterance],
                confidence=0.78,
            )
        if _current_medical_hit(utterance, _SEVERE_FALL):
            return RiskSignal(
                tier=3,
                kind="medical_emergency_fall",
                evidence=[utterance],
                confidence=0.74,
            )
        if _current_medical_hit(utterance, _CHEST, allow_resolved=True):
            return RiskSignal(
                tier=2,
                kind="medical_urgent_chest",
                evidence=[utterance],
                confidence=0.66,
            )
        if _current_medical_hit(utterance, _GAS_DANGER):
            return RiskSignal(
                tier=3,
                kind="immediate_safety_gas_leak",
                evidence=[utterance],
                confidence=0.82,
            )
        if _current_electrical_hazard(utterance):
            return RiskSignal(
                tier=3,
                kind="immediate_safety_electrical_hazard",
                evidence=[utterance],
                confidence=0.82,
            )
        if _current_medical_hit(
            utterance, _POSSIBLE_DUPLICATE_MEDICATION
        ):
            return RiskSignal(
                tier=2,
                kind="immediate_safety_possible_duplicate_medication",
                evidence=[utterance],
                confidence=0.72,
            )
        if _current_medical_hit(utterance, _CURRENTLY_LOST):
            return RiskSignal(
                tier=2,
                kind="immediate_safety_currently_lost",
                evidence=[utterance],
                confidence=0.72,
            )
        return self._mental.assess(ctx)
