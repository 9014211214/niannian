"""情景记忆召回(从旧 MemoryLayer.retrieve_relevant_episodes 移植,关键词版)。

referenceable_only=True 时只返回 Aria 可主动引用的 episode(私密的不召回)。
待升级:第 4 步复利闭环时换成语义(embedding)召回。
"""
from __future__ import annotations

import re

from ..model.persona import EpisodicMemoryEntry, PersonaProfile


def recall_episodes(
    persona: PersonaProfile,
    query: str,
    top_k: int = 3,
    referenceable_only: bool = False,
) -> list[EpisodicMemoryEntry]:
    candidates = (
        persona.referenceable_episodes()
        if referenceable_only
        else list(persona.episodic_memory)
    )
    if not candidates:
        return []

    q = query.lower()
    q_tokens = set(re.findall(r"\w+", q))

    scored: list[tuple[float, EpisodicMemoryEntry]] = []
    for ep in candidates:
        score = 0.0
        for kw in ep.keywords:
            kl = kw.lower()
            if kl and kl in q:          # 子串(中文关键词靠这个命中)
                score += 2.0
            elif kl in q_tokens:
                score += 1.5
        if ep.summary:
            for tok in q_tokens:
                if len(tok) >= 2 and tok in ep.summary.lower():
                    score += 0.5
        scored.append((score, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for s, ep in scored if s > 0][:top_k]
