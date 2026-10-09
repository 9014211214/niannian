"""Existing medical-response and sensitive-memory safeguards; no health monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

@dataclass(frozen=True)
class CognitiveSignal:
    domain: str
    signal: str
    severity: str
    functional_impact: bool
    relative_to_personal_baseline: bool
    confidence: float
    note: str
    confounders: list[str] = field(default_factory=list)


_THIRD_PERSON = (
    "我妈", "我爸", "我老伴", "我爱人", "我女儿", "我儿子", "我爷爷",
    "我奶奶", "我外公", "我外婆", "邻居", "家人", "亲戚", "他说", "她说",
)


_HYPOTHETICAL_OR_QUOTED = (
    "如果", "假如", "会不会", "听说", "新闻", "电视剧", "电影里", "举个例子",
)


_NEGATED = (
    "没有忘", "没忘", "并没有忘", "不是记不住", "没有迷路", "没迷路",
    "不是不会", "没有不会", "不是糊涂", "没有糊涂", "没糊涂",
    "不是认不出", "没有认不出", "不是说不出来", "并非说不出来",
)


_NORMAL_LAPSE = (
    "一时想不起", "刚才想不起后来想起来", "后来想起来了", "钥匙放哪了",
    "眼镜放哪了", "手机放哪了", "名字一下没想起来", "偶尔忘一次",
)


_BASELINE_CHANGE = (
    "以前", "原来", "过去", "从前", "平时", "一直都会", "本来会",
    "最近才", "这阵子开始", "现在却", "现在反而",
)


_CONFOUNDERS = {
    "sleep": ("没睡好", "失眠", "一夜没睡", "睡不着", "熬夜"),
    "hearing": ("听不清", "耳背", "助听器", "没听见"),
    "mood": ("心情低落", "抑郁", "难过", "丧偶", "老伴走了"),
    "medication": ("刚换药", "新药", "药量", "吃药以后"),
    "acute_illness": ("发烧", "感染", "低血糖", "生病", "尿路感染"),
}


_PATTERNS: tuple[tuple[str, str, tuple[str, ...], bool, float], ...] = (
    (
        "memory", "recent_memory_repetition",
        (
            "刚说过就忘", "刚问过又问", "同一个问题问了好几遍",
            "反复问同一件事", "刚发生的事记不住", "刚做完就忘",
            "别人刚告诉我就忘", "最近的事情总记不住",
        ),
        False, 0.90,
    ),
    (
        "orientation", "familiar_place_or_time_confusion",
        (
            "在熟悉的地方迷路", "熟悉的小区也迷路", "回家的路找不到",
            "不知道自己在哪", "分不清白天晚上", "总搞不清今天几号",
            "在家附近找不到路", "认不出这是哪里",
        ),
        True, 0.94,
    ),
    (
        "daily_function", "familiar_daily_task_change",
        (
            "熟悉的事情不会做", "原来会做现在不会", "以前会做现在不会",
            "做饭总忘关火", "煤气忘了关", "账单不会弄了", "不会交水电费了",
            "不会用一直在用的", "买菜算不清账", "忘了怎么回家",
        ),
        True, 0.93,
    ),
    (
        "language", "persistent_word_finding_change",
        (
            "常用的词总说不出来", "最近老是找不到词", "熟人的名字总叫不出来",
            "想说什么总说不出来", "最近说话老卡在词上",
        ),
        False, 0.82,
    ),
    (
        "executive", "planning_or_steps_change",
        (
            "熟悉的步骤也跟不上", "以前会安排现在理不清",
            "按原来的步骤也做不下去", "原来会算的账现在算不清",
            "简单的事情也不知道先做什么",
        ),
        True, 0.88,
    ),
)


_DISEASE_TERMS = (
    "阿尔茨海默", "阿尔兹海默", "阿兹海默", "老年痴呆", "认知障碍", "痴呆症",
)


_COGNITIVE_PRIVACY_MARKERS = (
    "记性变差", "记不住", "刚说过就忘", "刚问过又问", "反复问同一件事",
    "迷路", "找不到回家的路", "不知道自己在哪", "认不出人", "分不清白天晚上",
    "忘关火", "忘关煤气", "常用的词", "找不到词", "熟悉的事情不会做",
    "算不清账", "糊涂", "迷糊", "认知测试", "记忆力测试",
)


_UNSAFE_COGNITIVE_CLAIMS = (
    re.compile(r"(?:你|您).{0,12}(?:患有|得了|就是).{0,8}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症)"),
    re.compile(r"(?:你|您|你的情况|您的情况).{0,12}(?:可以|可|已经)?(?:诊断为|确诊为?|属于).{0,8}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
    re.compile(r"(?:这|这是|这个情况|这些表现).{0,8}(?:是|属于|符合)?(?:典型的?|明确的?).{0,4}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
    re.compile(r"(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍).{0,10}(?:早期|中期|晚期|分期)"),
    re.compile(r"(?:早期|中期|晚期).{0,10}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
    re.compile(r"(?:鱼油|维生素|保健品|营养品|补脑药).{0,12}(?:能|可以|一定会).{0,8}(?:预防|防止|避免).{0,6}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症)"),
    re.compile(r"(?:你|您).{0,12}(?:高度疑似|很可能|大概率|基本确定|八成|多半).{0,8}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
    re.compile(r"(?:表现|症状|对话).{0,8}(?:符合|说明|证明).{0,8}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
    re.compile(r"(?:鱼油|维生素|保健品|营养品|补脑药).{0,12}(?:最有效|有效预防|推荐吃|应该吃|值得长期吃)"),
    re.compile(r"(?:脑力游戏|认知训练|记忆训练).{0,12}(?:可以|能够|能|可).{0,6}(?:治疗|治好|逆转)"),
    re.compile(r"(?:保证|一定|肯定).{0,10}(?:不会得|能预防|可以预防|避免得)"),
    re.compile(r"(?:预防|防止).{0,8}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症).{0,12}(?:最有效|首选).{0,8}(?:鱼油|维生素|保健品|营养品|补脑药)"),
    re.compile(r"(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍).{0,10}(?:可能性|概率).{0,8}(?:极高|很高|较高|高达|不低|大)"),
    re.compile(r"(?:你|您).{0,12}(?:有)?(?:五|六|七|八|九|十)成.{0,8}(?:概率|可能性)?.{0,6}(?:患|得)?.{0,4}(?:阿尔茨海默|阿尔兹海默|阿兹海默|老年痴呆|痴呆症|认知障碍)"),
)


def _probability_claim_is_explicitly_refused(clause: str) -> bool:
    """Return true only when the probability itself is being refused.

    A general disclaimer such as ``不能诊断`` must not license an affirmative
    probability later in the same ASR transcript.  The negation therefore has
    to govern a probability verb, not merely appear somewhere in the clause.
    """
    return re.search(
        r"(?:不能|无法|没法|不应|不可以)"
        r".{0,10}(?:给出|判断|预测|估计|计算|认定|声称)"
        r".{0,16}(?:\d+(?:\.\d+)?\s*%|患病概率|发病概率)",
        clause,
    ) is not None


def _unsafe_match_is_explicitly_negated(
    clause: str, match: re.Match[str]
) -> bool:
    """Distinguish a bounded warning from the unsafe claim it quotes.

    Some unsafe regexes intentionally match substrings such as ``能治疗``;
    without this check they would also match ``不能治疗``.  Negation is kept
    local to the matched claim so an earlier disclaimer cannot mask a later
    diagnosis.
    """
    matched = match.group(0)
    if any(
        phrase in matched
        for phrase in (
            "不能治疗", "无法治疗", "不能预防", "无法预防",
            "不能保证", "无法保证", "不应推荐", "不建议吃",
        )
    ):
        return True
    prefix = clause[max(0, match.start() - 16):match.start()]
    return re.search(
        r"(?:(?:没有|无).{0,4}证据(?:表明|证明)?|"
        r"不代表|不等于|不能判断|无法判断)$",
        prefix,
    ) is not None


def extract_explicit_cognitive_signals(text: str) -> list[CognitiveSignal]:
    """只抽取本人明确表达；高精度优先，缺少表达时返回空。"""
    compact = " ".join((text or "").strip().split())
    if not compact:
        return []
    confounders = [
        name for name, markers in _CONFOUNDERS.items()
        if any(marker in compact for marker in markers)
    ]
    relative = any(marker in compact for marker in _BASELINE_CHANGE)
    clauses = [
        item.strip()
        for item in re.split(
            r"[。！？；;\n]|(?:，|,)?(?:但|但是|不过|然而|可是)", compact
        )
        if item.strip()
    ]
    result: list[CognitiveSignal] = []
    seen: set[tuple[str, str]] = set()
    for clause in clauses:
        if any(marker in clause for marker in _THIRD_PERSON):
            continue
        if any(marker in clause for marker in _HYPOTHETICAL_OR_QUOTED):
            continue
        if any(marker in clause for marker in _NEGATED):
            continue
        strong = any(
            phrase in clause
            for _domain, _signal, phrases, _impact, _confidence in _PATTERNS
            for phrase in phrases
        )
        if not strong and any(marker in clause for marker in _NORMAL_LAPSE):
            continue
        for domain, signal, phrases, impact, confidence in _PATTERNS:
            matched = next(
                (phrase for phrase in phrases if phrase in clause), None
            )
            if matched is None or (domain, signal) in seen:
                continue
            seen.add((domain, signal))
            result.append(CognitiveSignal(
                domain=domain,
                signal=signal,
                severity="functional" if impact else "watch",
                functional_impact=impact,
                relative_to_personal_baseline=relative,
                confidence=max(
                    0.5, confidence - (0.18 if confounders else 0.0)
                ),
                note=f"本人主动描述：{_signal_label(signal)}",
                confounders=list(confounders),
            ))
    return result


def _signal_label(signal: str) -> str:
    return {
        "recent_memory_repetition": "近期信息或提问出现反复",
        "familiar_place_or_time_confusion": "熟悉地点或时间定向出现变化",
        "familiar_daily_task_change": "熟悉日常任务出现变化",
        "persistent_word_finding_change": "持续找词出现变化",
        "planning_or_steps_change": "熟悉计划或步骤出现变化",
    }.get(signal, "认知或日常功能出现变化")


def guard_cognitive_health_reply(response: str) -> str:
    """拦截生成模型越界的疾病概率、诊断、分期或保健品保证。"""
    compact = " ".join((response or "").strip().split())
    if not compact:
        return compact
    clauses = [
        item.strip()
        for item in re.split(
            r"[。！？；;，,\n]|(?:但|不过|然而|可是|所以|因此|可见|这说明)",
            compact,
        )
        if item.strip()
    ]
    unsafe = False
    for clause in clauses:
        unsafe_probability = (
            any(term in clause for term in _DISEASE_TERMS)
            and (
                re.search(r"\d+(?:\.\d+)?\s*%", clause) is not None
                or "患病概率" in clause
                or "发病概率" in clause
                or re.search(
                    r"(?:五|六|七|八|九|十)成.{0,8}(?:概率|可能性|患|得)",
                    clause,
                ) is not None
            )
        )
        if (
            unsafe_probability
            and not _probability_claim_is_explicitly_refused(clause)
        ):
            unsafe = True
            break
        for pattern in _UNSAFE_COGNITIVE_CLAIMS:
            match = pattern.search(clause)
            if match is not None and not _unsafe_match_is_explicitly_negated(
                clause, match
            ):
                unsafe = True
                break
        if unsafe:
            break
    if not unsafe:
        return compact
    return (
        "我不能通过聊天预测、诊断或判断阿尔茨海默病分期，也不能给出个人"
        "患病概率。一次记忆或日常变化可能有睡眠、听力、情绪、药物或身体"
        "不适等多种原因；如果变化反复出现或影响熟悉生活，我可以先听您说"
        "具体发生了什么，再由您决定是否整理时间线请专业人员评估。"
    )


def is_cognitive_health_sensitive_utterance(text: str) -> bool:
    """标记不应进入普通记忆/人格反思的认知健康回合。"""
    compact = " ".join((text or "").strip().split())
    return bool(compact) and (
        any(term in compact for term in _DISEASE_TERMS)
        or any(marker in compact for marker in _COGNITIVE_PRIVACY_MARKERS)
        or bool(extract_explicit_cognitive_signals(compact))
    )


