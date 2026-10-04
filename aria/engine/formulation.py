"""个案假设更新 —— 把每轮证据累积成「关于这个人此刻的工作假设」(带置信)。

**非 LLM 的轻量在线版**:从 persona 中已经达到使用门槛的可推翻假设，加上本轮
可观察到的表达差异，维护当下工作假设。会后大模型反思位于
``memory/diagnosis.py``；它核验证据与反证，不做临床诊断，也不每轮重算人格。

这一层回答「理解不只是提示词」:假设是引擎自己维护、可解释、会随证据增长的
结构化状态,并且会被 prompt 真正读取(见 policy/prompt.py)、被持久化复利
(见 memory/consolidator.py)。
"""
from __future__ import annotations

from typing import Optional

from .context import EngineContext
from .model.persona import PersonaProfile
from .policy.strategy import normalize_emotion

_H_DIVERGENCE = "倾向口是心非 / 压抑(反复出现的言行不一致)"


def update_formulation(ctx: EngineContext, persona: Optional[PersonaProfile] = None) -> None:
    f = ctx.formulation

    # persona 先验:依恋 + 已沉淀的个案模式(冷启动也有底,老用户带着积累)。
    if persona is not None:
        if persona.attachment_style:
            prior = f"依恋倾向:{persona.attachment_style}"
            if prior not in f.hypotheses:
                f.hypotheses.append(prior)
        for cp in persona.case_conceptualization.core_patterns[:3]:
            if cp not in f.hypotheses:
                f.hypotheses.append(cp)

    # 在线证据:出现口是心非 → 提高「压抑 / 报喜不报忧」假设的置信。
    if ctx.emotion.external_reading is not None:
        return
    if ctx.emotion.divergence:
        if _H_DIVERGENCE not in f.hypotheses:
            f.hypotheses.append(_H_DIVERGENCE)
        f.confidence = min(1.0, f.confidence + 0.15)
    # 反证通道(P2-#12):表里一致地直接表达情绪 = 「压抑」的反面证据,
    # 置信回落;跌破阈值就撤销假设 —— 假设必须可被推翻,否则只是偏见。
    elif normalize_emotion(ctx.emotion.label) != "neutral":
        f.confidence = max(0.0, f.confidence - 0.05)
        if f.confidence < 0.15 and _H_DIVERGENCE in f.hypotheses:
            f.hypotheses.remove(_H_DIVERGENCE)
