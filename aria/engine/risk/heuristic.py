"""启发式风险筛查(**非临床**)—— 安全底线的第一道确定性的网。

红线:AI 只筛查 + 升级到人,绝不诊断、绝不做唯一安全网。
词表启发式远非完备;真实产品应叠加分类模型与纵向基线(PHQ / C-SSRS 思路)。
但「先有一道确定性的网」优于「等一张完美的网」——所以它默认常驻(BaseScene)。

分级(与 interfaces.risk 模块文档一致):
  tier 3  明确自伤/轻生意念       → 引擎接管本轮,确定性安全回应(不过 LLM)
  tier 2  高痛苦/被动意念         → 同上,语气更轻
  tier 1  低落/退缩线索(会话内累计 ≥2 次才报)→ 不接管,只给策略层降速旗

设计取舍:
- 危机侧宁可召回偏高:误报的代价是一句多余的关切,漏报的代价不可接受。
- tier 1 特意加了会话内累计门槛:一句「没意思」不该打断对话——那是关系的成本。
- 玩笑语境(「开玩笑」「说着玩」)降一档而不是清零:轻描淡写地说死,
  在危机干预实务里同样值得温和确认。
"""
from __future__ import annotations

import re
from typing import Optional

from ...interfaces.risk import RiskSignal
from ..context import EngineContext

# 明确意念(tier 3)。注意:「想死」有「想死你了」这种思念用法,单独排除。
_TIER3 = (
    "不想活", "活不下去", "活着没意思", "活着没劲", "一了百了",
    "自杀", "想死", "去死", "结束这一切", "结束自己",
    "死了算了", "走了算了", "了结了",
    "想把药全吃", "要把药全吃", "准备把药全吃", "打算把药全吃",
    "把整瓶药吃", "把一瓶药都吃", "把药一口气吃", "吞了一把药",
    "想上吊", "要上吊", "准备上吊", "绳子都套好了", "把绳子准备好了",
    "割腕", "跳楼", "从楼上跳", "烧炭", "喝农药",
)
_TIER3_FALSE_POSITIVES = ("想死你", "想死我了")   # 思念/夸张用法,不算意念

# 高痛苦 / 被动意念(tier 2)—— 还没说要走,但已经在说「撑不住/不如没有我」。
_TIER2 = (
    "撑不下去", "熬不下去", "熬不过去", "扛不住了",
    "我是个累赘", "就是个累赘", "拖累你们", "拖累大家", "没有我你们更好",
)

# 低落 / 退缩线索(tier 1)—— 单轮只计数,会话内累计 ≥2 次才升为信号。
_LOW = ("没意思", "提不起劲", "什么都不想", "没人管我", "孤零零", "白活了")

# 玩笑 / 引用语境 → 降一档(不清零)。
_SOFTEN = ("开玩笑", "说着玩", "说笑", "歌词", "电视里")
_SOFTEN_NEGATION_BEFORE = re.compile(
    r"(?:不是|并非|绝非|绝不是|可不是|没有|没在|不算)\s*$"
)

# 风险词只说明“句子里出现过什么”，不能说明风险正发生在用户本人身上。
# 这里保守排除明确的否定、假设、影视/新闻引用和第三人称转述；未带主语的
# 简短口语仍按本人表达处理，避免漏掉“胸痛”“撑不住了”这类求助。
_THIRD_PERSON_SUBJECTS = (
    "我妈", "我爸", "我老伴", "我爱人", "我女儿", "我儿子",
    "我妹妹", "我妹", "我姐姐", "我姐", "我哥哥", "我哥",
    "我弟弟", "我弟", "我孙子", "我孙女", "我朋友", "我邻居",
    "妈妈", "爸爸", "老伴", "爱人", "女儿", "儿子", "妹妹",
    "姐姐", "哥哥", "弟弟", "孙子", "孙女", "亲属", "亲戚",
    "家人", "邻居", "朋友", "患者", "病人", "那个人", "有人",
    "别人", "他", "她",
)
_SELF_SUBJECT_RE = re.compile(
    r"我(?!妈|爸|老伴|爱人|女儿|儿子|妹妹|妹|姐姐|姐|哥哥|哥|弟弟|弟|"
    r"孙子|孙女|朋友|邻居|家人|亲戚|亲属)"
)
_REFERENCE_CONTEXTS = (
    "电视剧里", "电视里", "剧里", "电影里", "新闻里", "小说里",
    "故事里", "台词里", "视频里", "短视频里", "网上说", "网上看到",
    "新闻说", "新闻报道", "看到新闻", "看新闻",
)
_QUESTION_SUFFIXES = ("是什么意思", "是什么", "有哪些原因", "怎么演", "怎么写")
_NEGATION_BEFORE = re.compile(
    r"(?:没有|没|并未|从未|不会|不想|并非|不是|否认|未曾)"
    r"(?:过|要|会|想过|觉得|感觉|出现|发生|有|到)?[^。！？；，,]{0,3}$"
)

_LOW_HITS_KEY = "_risk_low_mood_hits"
_LOW_THRESHOLD = 2


def _without_tier3_false_positives(utterance: str) -> str:
    cleaned = utterance
    for phrase in _TIER3_FALSE_POSITIVES:
        cleaned = cleaned.replace(phrase, "")
    return cleaned


def _softening_applies(utterance: str, risk_word: str) -> bool:
    """只让风险表达所在分句或紧随其后的澄清分句降低一级。"""
    clauses = re.split(r"(?<=[。！？；;，,\n])", utterance)
    risk_indexes = [
        index for index, clause in enumerate(clauses)
        if risk_word in clause
    ]
    if not risk_indexes:
        return False

    def clause_softens(clause: str) -> bool:
        for marker in _SOFTEN:
            start = 0
            while True:
                at = clause.find(marker, start)
                if at < 0:
                    break
                prefix = clause[:at]
                if not _SOFTEN_NEGATION_BEFORE.search(prefix[-10:]):
                    return True
                start = at + len(marker)
        return False

    # 任一同词风险表达没有玩笑限定时，就按更安全的原等级处理。
    return all(
        clause_softens(clauses[index])
        or (
            index + 1 < len(clauses)
            and clause_softens(clauses[index + 1])
        )
        for index in risk_indexes
    )


def _last_subject(prefix: str) -> str | None:
    """返回命中词前最近的明确主语；亲属称谓中的“我”不算本人主语。"""
    third_at = max((prefix.rfind(item) for item in _THIRD_PERSON_SUBJECTS), default=-1)
    self_at = max((item.start() for item in _SELF_SUBJECT_RE.finditer(prefix)), default=-1)
    if third_at < 0 and self_at < 0:
        return None
    return "self" if self_at > third_at else "third"


def asserted_self_report_hit(
    utterance: str,
    words: tuple[str, ...],
    *,
    allow_third_person: bool = False,
) -> Optional[str]:
    """找出未被否定、引用或第三人称语境包住的风险词。"""
    for word in words:
        start = 0
        while True:
            at = utterance.find(word, start)
            if at < 0:
                break
            clause_start = max(
                (utterance.rfind(mark, 0, at) for mark in "。！？；;，,\n"),
                default=-1,
            ) + 1
            clause_end_candidates = [
                pos for mark in "。！？；;，,\n"
                if (pos := utterance.find(mark, at + len(word))) >= 0
            ]
            clause_end = min(clause_end_candidates, default=len(utterance))
            prefix = utterance[clause_start:at]
            suffix = utterance[at + len(word):clause_end]
            subject = _last_subject(prefix)
            reference_at = max(
                (prefix.rfind(item) for item in _REFERENCE_CONTEXTS),
                default=-1,
            )
            self_after_reference = False
            if reference_at >= 0:
                self_after_reference = _last_subject(prefix[reference_at:]) == "self"

            rejected = (
                bool(_NEGATION_BEFORE.search(prefix[-14:]))
                or (
                    prefix.rstrip().endswith("不")
                    and not word.startswith("不")
                )
                or (subject == "third" and not allow_third_person)
                or (reference_at >= 0 and not self_after_reference)
                or (
                    any(item in prefix for item in ("如果", "假如", "假设"))
                    and not any(item in prefix for item in ("我现在", "我正在", "我刚"))
                )
                or (
                    any(suffix.lstrip().startswith(item) for item in _QUESTION_SUFFIXES)
                    and subject != "self"
                )
            )
            if not rejected:
                return word
            start = at + len(word)
    return None


class HeuristicRiskMonitor:
    """词表分级筛查。无状态判断 + 会话内低落计数(存在 ctx.working_memory)。"""

    def assess(self, ctx: EngineContext) -> Optional[RiskSignal]:
        u = ctx.working_memory.get("current_utterance", "") or ""
        if not u:
            return None

        tier, kind = 0, ""
        tier3_text = _without_tier3_false_positives(u)
        w3 = asserted_self_report_hit(tier3_text, _TIER3)
        if w3:
            tier, kind = 3, "ideation"
        else:
            w2 = asserted_self_report_hit(u, _TIER2)
            if w2:
                tier, kind = 2, "high_distress"

        if tier >= 2:
            risk_text = tier3_text if w3 else u
            risk_word = w3 or w2
            if risk_word and _softening_applies(risk_text, risk_word):
                tier -= 1                      # 玩笑语境降一档,tier2→1 走降速旗
            if tier >= 2:
                return RiskSignal(tier=tier, kind=kind, evidence=[u], confidence=0.6)
            return RiskSignal(tier=1, kind=kind, evidence=[u], confidence=0.4)

        if asserted_self_report_hit(u, _LOW):
            hits = ctx.working_memory.get(_LOW_HITS_KEY, 0) + 1
            ctx.working_memory[_LOW_HITS_KEY] = hits
            if hits >= _LOW_THRESHOLD:
                return RiskSignal(tier=1, kind="low_mood", evidence=[u], confidence=0.3)
        return None
