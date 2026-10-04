"""Bind one online return channel after the client selects its wire transport.

This is a prompt-only protocol revision. It does not change the decision schema,
parse/heal a response, offer external tools, or retry a model request. Keeping the
transformation pure lets prospective benchmarks bind their saved input to the
actual request without changing historical prompt or request artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Literal


ONLINE_RETURN_CHANNEL_VERSION = "online_return_channel_v1"
OnlineReturnChannel = Literal["native", "plain_json"]

# This exact legacy selector remains in the pre-transport prompt for historical
# replay. Remove it from trusted prompt messages, never from quoted user input.
_LEGACY_CHANNEL_SELECTOR = (
    "API有aria_turn_decision时，仅调用一次，JSON放"
    "function.arguments，勿另附content；否则输出JSON正文。"
)
_FORMAT_SCOPE = (
    f"## 客户端指定的唯一返回通道：{ONLINE_RETURN_CHANNEL_VERSION}\n"
    "本条是本次请求的最高优先级返回格式指令；其余短句或自然表达要求仅约束"
    "response字段，不改变内部返回格式。通道已经由客户端选定，不由模型判断或切换。\n"
)
_ACTION_BOUNDARY = (
    "\n此内部返回只提交当轮决策，不执行或授权任何外部动作；"
    "外部<action>仍须遵守原有能力、当前明确请求/确认、授权和真实回执边界。"
)
_CHANNEL_INSTRUCTIONS = {
    "native": (
        _FORMAT_SCOPE
        + "唯一通道=native。必须且仅调用一次aria_turn_decision；完整JSON对象"
        "只放在该函数的function.arguments。message.content必须为空；"
        "不得用正文JSON、普通回复或Markdown代替函数调用，也不得调用其他函数。"
        + _ACTION_BOUNDARY
    ),
    "plain_json": (
        _FORMAT_SCOPE
        + "唯一通道=plain_json。本次未提供返回函数；必须在message.content正文中"
        "只输出一个完整JSON对象。不得输出tool_calls、函数调用、Markdown围栏或"
        "JSON以外的解释；不得仅输出response字段的普通回复。"
        + _ACTION_BOUNDARY
    ),
}


def bind_online_return_channel(
    messages: list[dict], *, channel: OnlineReturnChannel,
) -> list[dict]:
    """Return copied messages with exactly one system-level channel instruction.

    Rebinding is idempotent, including a change of channel. The dedicated system
    instruction follows all existing system messages (and therefore their format
    descriptions), while user/developer text cannot suppress it with a marker.
    The caller decides whether this is an online decision request.
    """
    if channel not in _CHANNEL_INSTRUCTIONS:
        raise ValueError("unknown_online_return_channel")
    instruction = _CHANNEL_INSTRUCTIONS[channel]
    copied = []
    for original in messages:
        message = deepcopy(original)
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                if content in _CHANNEL_INSTRUCTIONS.values():
                    continue
                message["content"] = content.replace(_LEGACY_CHANNEL_SELECTOR, "")
        copied.append(message)
    position = 1 + max(
        (index for index, item in enumerate(copied) if item.get("role") == "system"),
        default=-1,
    )
    copied.insert(position, {"role": "system", "content": instruction})
    return copied
