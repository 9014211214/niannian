"""Channel 协议 —— 触达通道。

语音 / 文本 / 将来的机器人;以及「通知谁」(如养老场景里的照护人)。
"""
from __future__ import annotations

from typing import Protocol


class Channel(Protocol):
    id: str

    def send(self, text: str) -> None: ...
