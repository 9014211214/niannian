"""养老领域知识库 —— KnowledgeSource 的具体实现(P3-#14:结构化 YAML 资产)。

条目在 `elder_kb.yaml`(正常衰老 vs 警示 / 用药 / 回避型沟通禁忌 / 情绪 / 安全),
代码只负责加载 + 关键词检索。将来可换成检索增强(RAG),资产不变。
注意:领域知识 ≠ 关于这个人的知识(后者在 engine/memory 的活档案)。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_KB_PATH = Path(__file__).resolve().parent / "elder_kb.yaml"


@lru_cache(maxsize=1)
def _load() -> list[tuple[list[str], str]]:
    data = yaml.safe_load(_KB_PATH.read_text(encoding="utf-8"))
    return [([str(k) for k in e["keywords"]], str(e["fact"])) for e in data["entries"]]


class ElderKnowledge:
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        q = query.lower()
        scored: list[tuple[int, str]] = []
        for keys, fact in _load():
            score = sum(1 for k in keys if k in q)
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]
