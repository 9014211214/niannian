"""意图分流(P2-#7)—— 决定在线回复需要快速表达还是单次综合决策。

早期产品验证过的教训:每句话都过 LLM 言外之意探针 = 成本 ×2、语音延迟伤。
这里用零成本规则先分流:
- 寒暄、事实查询、明确任务和短日常陈述走快速表达；
- 嘴硬、关系/角色/自主变化、身体变化、跨轮纠正和多模态冲突走一次综合决策。
综合决策同时产出当轮状态候选、沟通方向和回复，不再串联独立状态模型。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
from .fusion import _normalize_label
from .signals import unit_interval

_GREETINGS = (
    "你好", "您好", "早上好", "下午好", "晚上好", "早安", "晚安",
    "在吗", "在不在", "嗨", "哈喽", "hello", "hi",
)

# 事实查询标志:问的是世界,不是他自己。
_FACT_MARKERS = (
    "几点", "什么时候", "多少", "哪里", "在哪", "怎么去", "怎么做", "怎么用",
    "是什么", "是谁", "查一下", "帮我查", "天气", "星期几", "几号",
    "提醒我", "定个闹钟", "设置", "打开", "关掉", "播放",
)

# 与事实词同句出现时，说明重点可能是个人处境而不是查世界信息。宁可多调用
# 一次，也不能把“他们连几点出门都替我定了”误分成询问时间。
_PERSONAL_STAKES_MARKERS = (
    "替我", "给我定", "给我安排", "不让我", "非让我", "逼我",
    "管着我", "为我好", "俺也去能干啥", "我还能干啥", "我有什么用",
)

# 情绪线索 + 嘴硬高频词：命中任何一个都必须走快速模型的单次综合决策
# （只收歧义低的词；单字「气」会误伤「天气」一类，不收）。
_EMOTION_HINTS = (
    "烦", "累", "哭", "怕", "愁", "委屈", "孤单", "孤独", "想念", "唉",
    "难受", "难过", "低落", "失望", "崩溃", "心累", "睡不着", "没意思",
    "害怕", "担心", "紧张", "压力", "开心", "高兴", "激动",
    "没事", "算了", "都行", "随便", "无所谓", "不想",
)

# 需要上下文、反证或多解释竞争的表达。显式“我很高兴”不一定要花一次状态
# 模型调用；但嘴硬、角色收缩、关系变化、自主受损和跨轮纠正需要。
_SEMANTIC_COMPLEXITY_HINTS = (
    "没事", "算了", "都行", "随便", "无所谓", "不用管", "别问",
    "其实", "反正", "就是", "挺好", "还行", "不想", "不出门",
    "不参加", "帮不上", "插不上", "能干啥", "有什么用", "没用了",
    "一个人", "没人说话", "说不到一块", "空落落", "自从", "走后",
    "替我", "给我定", "给我安排", "不让我", "非让我", "逼我",
    "最近", "这几天", "前几天", "上周", "这周", "以前", "后来",
    "已经", "不是", "纠正", "其实不是", "现在好多了",
    "睡不着", "夜里总醒", "疼", "发黑", "站不稳", "扶着墙",
)

_ROUTINE_STATEMENT_HINTS = (
    "吃过了", "吃了", "喝了", "散步", "看电视", "看书", "听戏",
    "下棋", "晒太阳", "买菜", "做饭", "回来了", "到了", "知道了",
    "好的", "嗯嗯", "谢谢", "不用谢",
)

# 极窄的“证据已经成对出现”通道：不是从一个词猜心理，而是当前原话同时给出
# 字面报平安和正在发生的参与变化。确定性证据层足以先建立低置信候选，省去
# 一次重复状态调用；最终表达仍交给同一个快速模型。
_REASSURANCE_MARKERS = (
    "没事", "我挺好", "都挺好", "还好", "不用担心", "别担心",
)
_CURRENT_PARTICIPATION_CHANGE_MARKERS = (
    "最近不太想出门", "最近不想出门",
    "这几天不太想出门", "这几天不想出门",
    "最近不太想参加", "最近不想参加",
    "这几天不太想参加", "这几天不想参加",
)
_EXPLICIT_TEMPORARY_CAUSE_MARKERS = (
    "等快递", "等人", "下雨", "天气不好", "下午还要", "晚点还要",
    "已经处理好", "事情处理好了",
)


@dataclass(frozen=True)
class StateAnalysisRoute:
    tier: str
    reasons: tuple[str, ...]

    @property
    def needs_integrated_decision(self) -> bool:
        return self.tier == "integrated"

    @property
    def use_probe(self) -> bool:
        """兼容旧调用；现在表示需要综合决策，并不触发独立探针。"""
        return self.needs_integrated_decision


def route_state_analysis(
    utterance: str,
    history: Sequence[Any] | None = None,
    *,
    acoustic: dict[str, Any] | None = None,
    text: dict[str, Any] | None = None,
    multimodal: Mapping[str, Any] | None = None,
) -> StateAnalysisRoute:
    """决定本轮回复是否需要一次快速综合决策；不确定时保留综合路径。

    `skip` 不是“不理解”，而是明确让确定性情绪、风险、Skill 和人物决策继续
    工作；`integrated` 也只会触发快速模型的一次在线综合决策调用。
    """
    s = utterance.strip()
    if not s:
        return StateAnalysisRoute("skip", ("empty",))
    if _multimodal_conflict(acoustic, text):
        return StateAnalysisRoute("integrated", ("multimodal_conflict",))
    if _trusted_multimodal_baseline_change(multimodal):
        return StateAnalysisRoute(
            "integrated", ("trusted_multimodal_baseline_change",)
        )
    if _has_explicit_reassurance_change_pair(s):
        return StateAnalysisRoute(
            "structured", ("explicit_reassurance_change_pair",)
        )
    if any(marker in s for marker in _PERSONAL_STAKES_MARKERS):
        return StateAnalysisRoute("integrated", ("personal_stakes",))
    if any(marker in s for marker in _SEMANTIC_COMPLEXITY_HINTS):
        return StateAnalysisRoute("integrated", ("semantic_complexity",))
    if len(s) <= 8 and any(s.startswith(g) for g in _GREETINGS):
        return StateAnalysisRoute("skip", ("greeting",))
    if any(marker in s for marker in _FACT_MARKERS):
        return StateAnalysisRoute("skip", ("factual_or_task",))
    if any(marker in s for marker in _ROUTINE_STATEMENT_HINTS):
        return StateAnalysisRoute("skip", ("routine_statement",))
    if any(marker in s for marker in _EMOTION_HINTS):
        return StateAnalysisRoute("integrated", ("affect_or_distress",))
    if history and any(
        marker in s for marker in ("还是", "又", "刚才", "那件事", "照旧")
    ):
        return StateAnalysisRoute("integrated", ("context_dependency",))
    # 较长的自由表达更可能同时包含原因、转折或隐含状态；短而无风险的日常陈述
    # 直接交给快速回复，避免“今天猫又来了”也多调一次大模型。
    if len(s) >= 24:
        return StateAnalysisRoute("integrated", ("long_free_expression",))
    return StateAnalysisRoute("skip", ("simple_expression",))


def needs_integrated_decision(utterance: str) -> bool:
    """True 表示快速模型应使用单次综合在线决策协议。"""
    return route_state_analysis(utterance).tier != "skip"


def needs_deep_analysis(utterance: str) -> bool:
    """旧名称兼容；不表示会调用深度模型。"""
    return needs_integrated_decision(utterance)


def _has_explicit_reassurance_change_pair(utterance: str) -> bool:
    if any(marker in utterance for marker in _EXPLICIT_TEMPORARY_CAUSE_MARKERS):
        return False
    return (
        any(marker in utterance for marker in _REASSURANCE_MARKERS)
        and any(
            marker in utterance
            for marker in _CURRENT_PARTICIPATION_CHANGE_MARKERS
        )
    )


def _multimodal_conflict(
    acoustic: dict[str, Any] | None,
    text: dict[str, Any] | None,
) -> bool:
    if not acoustic or not text:
        return False
    left = _normalize_label(acoustic.get("label"))
    right = _normalize_label(text.get("label"))
    if "unknown" in {left, right} or left == right:
        return False
    left_score = unit_interval(acoustic.get("score")) * unit_interval(acoustic.get("quality"), 1.0)
    right_score = unit_interval(text.get("score")) * unit_interval(text.get("quality"), 1.0)
    return min(left_score, right_score) >= 0.55


def _trusted_multimodal_baseline_change(
    multimodal: Mapping[str, Any] | None,
) -> bool:
    """只有高质量、相对本人基线明显变化的视觉/行为信号才升级。"""
    if not isinstance(multimodal, Mapping):
        return False
    for modality in ("visual", "behavioral"):
        payload = multimodal.get(modality)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping) and "signal" in payload:
            rows = [payload]
        elif isinstance(payload, Mapping):
            rows = list(payload.values())
        else:
            continue
        for row in rows:
            if not isinstance(row, Mapping) or bool(row.get("fallback")):
                continue
            confidence = _number(row.get("confidence"), 0.0)
            quality = _number(row.get("quality"), 1.0)
            delta = row.get("baseline_delta")
            if delta is None:
                continue
            if (
                confidence >= 0.70
                and quality >= 0.70
                and abs(_number(delta, 0.0)) >= 0.35
            ):
                return True
    return False


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default
