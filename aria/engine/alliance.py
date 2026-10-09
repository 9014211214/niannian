"""治疗联盟追踪 —— 大师盯的「关系」。**非 LLM**,结构化、可解释、可单测。

这是「理解人」里不依赖 LLM 的一层:关系是涨是裂,用规则维护成显式状态,
裂了引擎会优先修复(而不是把判断含混地丢给提示词)。

破裂判定(P2-#11):优先用言外之意探针顺带给的 rupture 字段(语境化,零额外
成本);探针无判定时(语音模式 / mock)退回关键词兜底 —— 兜底词只收明确
冲着「我」来的表达,「今天天气好烦」不再误伤。
"""
from __future__ import annotations

import re

from .context import EngineContext

# 关键词兜底:只收指向对话者的不耐烦 / 被敷衍 / 拒绝继续。
_RUPTURE = (
    "别管", "够了", "不想说", "你们就会", "敷衍", "烦不烦", "别问了",
)
_DIRECTED_USELESS = (
    "你没用", "你真没用", "你也没用", "你一点用没有",
    "问你也没用", "跟你说也没用", "和你说也没用",
)


def rejection_surface(utterance: str) -> str:
    """Remove only explicitly rejected interpretations of the user's silence.

    This finite inverse-negation handling is shared with observation planning,
    so a correction is not counted as a rupture before that planner runs. It
    does not infer consent, modify the text sent to the model, dismiss a model
    rupture signal, or remove a later real stop after a contrast.
    """
    compact = re.sub(
        r"(?:不是|并非|不代表|不等于)(?:我)?(?:什么都|完全|真的)?不想(?:展开|继续|详细|多|再)?(?:说|聊)(?:话|天)?",
        "", utterance,
    )
    inference = r"(?:当作|当成|看成|当|以为|觉得)(?:我)?(?:什么都)?不想(?:展开|继续|详细|多|再)?(?:说|聊)(?:话|天)?"
    compact = re.sub(
        r"(?:别|不要)(?:把)?(?:我)?(?:话少|沉默|少说话)" + inference,
        "", compact,
    )
    compact = re.sub(r"(?:别|不要)(?:以为|觉得)(?:我)?(?:什么都)?不想(?:展开|继续|详细|多|再)?(?:说|聊)(?:话|天)?", "", compact)
    # A comma may separate the premise from the corrected interpretation, but
    # the final clause is bounded so a later genuine refusal survives.
    return re.sub(
        r"(?:别|不要)因为(?:我)?(?:话少|沉默|少说话)[，,]?(?:就)?" + inference,
        "", compact,
    )


def update_alliance(
    ctx: EngineContext, utterance: str, *, allow_surface_inference: bool = True,
) -> None:
    al = ctx.alliance
    if ctx.emotion.rupture is not None:
        ruptured = ctx.emotion.rupture                      # 探针判定优先(语境化)
    elif not allow_surface_inference:
        # A reported refusal or a long sentence is not relationship evidence.
        # The integrated current-intent model owns ordinary semantics.
        return
    else:
        surface = rejection_surface(utterance)
        ruptured = any(w in surface for w in _RUPTURE) or any(
            phrase in surface for phrase in _DIRECTED_USELESS
        )
    if ruptured:
        if not al.ruptured:
            al.rupture_events += 1                          # 只在新裂开时计数
        al.ruptured = True
        al.trust = max(0.0, al.trust - 0.10)
    else:
        al.ruptured = False
        # 愿意多说一点(不是「嗯/好」)→ 关系微升。
        if allow_surface_inference and len(utterance) >= 12:
            al.trust = min(1.0, al.trust + 0.03)
