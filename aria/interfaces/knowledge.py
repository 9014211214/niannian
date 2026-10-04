"""KnowledgeSource 协议 —— 领域知识(可检索)。

养老=老年 / 用药 / 警示信号 KB;闲聊=None;客服=产品 KB。
注意:这是「领域知识」,与「关于这个人的知识」(归 engine/memory)分开。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class WorldKnowledgeItem:
    """带来源和有效期的公共现状；绝不能写入某个人的人格或私有记忆。"""

    content: str
    source: str
    observed_at: str
    expires_at: str | None = None

    def is_current(self, now: datetime | None = None) -> bool:
        try:
            observed = datetime.fromisoformat(
                self.observed_at.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        current = now or (
            datetime.now(observed.tzinfo)
            if observed.tzinfo is not None else datetime.now()
        )
        if not _comparable_lte(observed, current):
            return False
        if not self.expires_at:
            return True
        try:
            expiry = datetime.fromisoformat(
                self.expires_at.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        return _comparable_lte(current, expiry)


def _comparable_lte(left: datetime, right: datetime) -> bool:
    """兼容带/不带时区的 ISO 时间；来源时区缺失时按部署本地时间解释。"""
    if left.tzinfo is None and right.tzinfo is not None:
        right = right.replace(tzinfo=None)
    elif left.tzinfo is not None and right.tzinfo is None:
        left = left.replace(tzinfo=None)
    return left <= right


class KnowledgeSource(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[str]: ...


class WorldKnowledgeSource(Protocol):
    def retrieve(
        self, query: str, top_k: int = 3
    ) -> list[WorldKnowledgeItem]: ...
