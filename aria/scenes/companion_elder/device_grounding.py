"""Finite device-capability grounding, not a general factuality classifier.

Owning a phone/album does not establish its version-specific capabilities.
Only explicit recent user observations ground the known capability families
below; assistant history, personality, hypothetical wishes and someone else's
device do not. Genuine capability-presence conditions remain useful guidance.
No model call, device inspection, or external action is performed here.
"""
from __future__ import annotations

import re

from ...engine.context import EngineContext


UNCONFIRMED_DEVICE_CAPABILITY_REPLY = (
    "这个功能我还没有依据确认，先不把猜测当成说明。"
    "可以先看看页面列出的功能，再按你希望的节奏介绍。"
)
_SENTENCES = re.compile(r"[。！？!?；;\n]+|但是|不过|但")
_CLAUSES = re.compile(r"[，,]+|而且|并且|另外")
# A comma is optional before a conditional's consequent. Keep only the
# antecedent's capability conditional: "if A then B" does not establish B.
# Reply-only splitting leaves the source-observation rules unchanged; 就是
# remains part of ordinary definitions rather than a conditional connector.
_DELIVERY_CLAUSES = re.compile(_CLAUSES.pattern + r"|就(?!是)|那么|则|便(?=能|可|会|用|试|点|按)")
_DEVICE = re.compile(r"手机|平板|相册|应用|软件|APP|App|app")
_FEATURES = (
    ("album_location_grouping", re.compile(
        r"(?:按|根据).{0,10}(?:地点|位置|拍摄地).{0,10}(?:分类|归类|整理)|"
        r"(?:地点|位置|拍摄地)(?:自动)?(?:分类|归类|整理)")),
    ("album_person_grouping", re.compile(
        r"(?:按|根据).{0,10}(?:人物|人脸|人名).{0,10}(?:分类|归类|整理)|"
        r"(?:人物|人脸|人名)(?:自动)?(?:分类|归类|整理)")),
    ("album_automatic_creation", re.compile(
        r"(?:一键|自动).{0,6}(?:制作|生成|创建|做成|做).{0,4}"
        r"(?:电子相册|视频相册|音乐相册|回忆视频)")),
    ("album_face_recognition", re.compile(r"人脸识别|识别人脸")),
)
_NONFACT = re.compile(
    r"如果|假如|要是|可能|也许|或许|不确定|不知道|不清楚|有没有|是否|能不能|"
    r"支持吗|可以吗|能.{0,20}吗|想要|希望|听说|据说|广告|演示|举例|例子|电视剧|电影"
)
_NEGATIVE = re.compile(r"不能|不支持|没有|没这|没那个|不具备|不带|取消了|关掉了|已经卸载")
_OTHER_DEVICE = re.compile(r"朋友|别人|他[的那]|她[的那]|(?:女儿|儿子|家人)的|"
                           r"旧手机|以前的手机|另一(?:部|台)")
_RESET = re.compile(r"换了.{0,4}手机|换手机|现在.{0,3}新手机|不是这(?:部|台)|"
                    r"没有这个功能|不支持这个功能|已经卸载")
_PRESENCE_CONDITION = re.compile(
    r"^(?:如果|假如|要是)(?:(?:这个|你的|您的|你这个|您这个)?"
    r"(?:相册|手机|平板|页面|界面|屏幕|应用|软件)(?:里|上|中)?"
    r"(?:确实|真的)?(?:有|显示|支持|带有)|"
    r"(?:你|您)(?:看到|找到|确认有)|有(?!空|时间|兴趣|意愿|需要))"
)
_CAPABILITY_CHECK = re.compile(
    r"有没有|是否(?:有|支持|能)|能不能|不确定.{0,6}(?:有|支持|能)|不一定(?:有|支持|能)"
)
_GENERAL = re.compile(r"^(?:有些|部分|某些|不同的)(?:手机|平板|相册|应用|软件)")
_SPECIFIC_DEVICE = re.compile(r"你(?:的|这)|您(?:的|这)|这(?:个相册|部手机|台平板)")
_DEFINITION = re.compile(r"是指|指的是|的意思是|就是把")
_AFFIRMATION = re.compile(r"有|支持|能|可以|带有|看到|显示|找到|用过|用着|试过")


def _features(clause: str) -> set[str]:
    return {name for name, pattern in _FEATURES if pattern.search(clause)}


def _source_supports(feature: str, sources: list[str]) -> bool:
    # Newer direct denial/uncertainty must not be overridden by an old mention.
    for source in sources:
        clauses = [c.strip() for sentence in _SENTENCES.split(source)
                   for c in _CLAUSES.split(sentence) if c.strip()]
        relevant = [c for c in clauses if feature in _features(c)]
        if relevant:
            for clause in reversed(relevant):
                if (_NONFACT.search(clause) or _NEGATIVE.search(clause)
                        or _OTHER_DEVICE.search(clause) or "?" in source or "？" in source):
                    return False
                if _AFFIRMATION.search(clause):
                    return True
            return False
        if _RESET.search(source):
            return False
    return False


def device_capability_violations(
    response: str, ctx: EngineContext, utterance: str,
) -> list[str]:
    """Check the delivered text against current + last three user turns only.

    Permission conditions ("if you like") and soft suggestions ("try") do not
    establish functionality. An actual presence check/condition covers only
    the named capability and only within that sentence, never its neighbours.
    This finite regression guard intentionally does not certify all phone UI
    advice or infer capability from a tool/visual result without its own typed
    device-identity provenance.
    """
    sources = [str(utterance or ""), *(turn.user for turn in reversed(ctx.turns[-3:]))]
    if not _DEVICE.search(" ".join([response, *sources])):
        return []
    unsupported = False
    for sentence in _SENTENCES.split(response):
        conditional: set[str] = set()
        for raw in _DELIVERY_CLAUSES.split(sentence):
            clause = raw.strip()
            features = _features(clause)
            if not features:
                continue
            if _PRESENCE_CONDITION.search(clause):
                conditional.update(features)
                continue
            # "Whether this album supports X" is a check, not a promise that
            # it does. "If you like, this album can X" has no such check.
            if _CAPABILITY_CHECK.search(clause):
                continue
            if (_GENERAL.search(clause) or _DEFINITION.search(clause)) and not _SPECIFIC_DEVICE.search(clause):
                continue
            if _NEGATIVE.search(clause):
                # This guard concerns invented positive capabilities, not a
                # claim that absence itself has been independently verified.
                continue
            if any(feature not in conditional and not _source_supports(feature, sources)
                   for feature in features):
                unsupported = True
    return ["unconfirmed_device_capability"] if unsupported else []
