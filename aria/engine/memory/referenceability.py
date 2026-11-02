"""Source-bound permission to mention durable facts in owner conversation.

These markers are a conservative privacy fallback, not a semantic classifier.
A selected quote cannot obtain broader permission than its complete source turn.
"""
from __future__ import annotations

import hashlib
import re


SOURCE_CONTEXT_VERSION = "owner_turn_context_v1"
REVIEWED_FACT_CONTEXT = "owner_turn_fact_purpose_v2"

# Keep the existing privacy floor shared by extraction and later consumers.
_SENSITIVE = (
    "老伴", "过世", "去世", "逝", "忌日", "自杀", "自残", "遗嘱", "病危",
    "欠钱", "欠债", "债务", "借款", "贷款", "负债", "转账", "汇款", "存款",
    "账户", "账号", "银行卡", "卡号", "密码", "验证码", "余额",
    "身份证", "证件号", "护照", "社保卡", "医保卡", "户口簿", "户口本",
    "手机号", "电话号码", "微信号", "联系方式", "住址", "门牌号",
    "家庭隐情", "家丑", "婚外情", "出轨", "私生", "收养", "领养", "离婚",
)
_CONFIDENTIALITY_PATTERNS = (
    re.compile(r"(?:别|不要|不许).{0,12}(?:告诉|透露|提起|说给)"),
    re.compile(r"(?:别|不要|不许).{0,12}(?:跟|对|向).{0,10}(?:说|提|讲)"),
    re.compile(r"(?:不想|不能|别|不要).{0,8}让.{0,12}知道"),
    re.compile(r"(?:替我|帮我|给我)?保密"),
    re.compile(r"(?:瞒着|背着).{0,12}(?:家里|家人|家属|孩子|女儿|儿子|老伴)?"),
)
_CONTACT_OR_CREDENTIAL_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)


def must_not_reference(text: str) -> bool:
    """The existing local privacy fallback; it does not grant share permission."""
    value = " ".join(str(text or "").strip().split())
    return (
        any(marker in value for marker in _SENSITIVE)
        or any(pattern.search(value) for pattern in _CONFIDENTIALITY_PATTERNS)
        or any(pattern.search(value) for pattern in _CONTACT_OR_CREDENTIAL_PATTERNS)
    )


def source_context_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reference_context_is_valid(
    text: str | None, digest: str | None, version: str | None, quote: str | None,
) -> bool:
    """Missing legacy context is retained, but cannot authorize active recall."""
    return bool(
        version in (SOURCE_CONTEXT_VERSION, REVIEWED_FACT_CONTEXT)
        and isinstance(text, str) and text.strip()
        and isinstance(quote, str) and quote.strip() and quote in text
        and digest == source_context_sha256(text)
        and not must_not_reference(text)
        and not must_not_reference(quote)
    )


def preference_source_turns(persona, proposals=(), *, retired_only=False) -> dict[tuple[str, int], set[str]]:
    """Index owner evidence, including confirmations retained after withdrawal."""
    from ..model.persona import PreferenceQuote

    turns: dict[tuple[str, int], set[str]] = {}
    for preference in (*persona.stated_preferences, *proposals):
        if retired_only and preference.status not in ("withdrawn", "superseded"):
            continue
        evidence = [*preference.quotes, preference.confirmation,
                    *(transition.quote for transition in preference.history)]
        for quote in evidence:
            if isinstance(quote, PreferenceQuote):
                turns.setdefault((quote.session_id, quote.turn_index), set()).add(quote.text)
    if not retired_only:
        def add_reviewed_pair(session, index, quote, source):
            control = source.preference_source_quote
            if (session and type(index) is int and source.source_context_version == REVIEWED_FACT_CONTEXT
                    and control and reference_context_is_valid(
                        source.source_context, source.source_context_sha256, source.source_context_version, quote,
                    ) and separate_fact_source(source.source_context, quote, control, {source.source_context})):
                turns.setdefault((session, index), set()).add(control)

        for episode in persona.episodic_memory:
            add_reviewed_pair(episode.source_session_id, episode.source_turn_index, episode.source_quote, episode)
        for row in persona.relationship_context:
            for evidence in row.evidence:
                if evidence.source == "user_conversation":
                    for source in evidence.fact_sources:
                        if source.turn_index in evidence.turn_indices:
                            add_reviewed_pair(evidence.session_id, source.turn_index, source.quote, source)
    return turns


def retired_source_starts(persona) -> dict[str, int]:
    """Earliest documented control turn in each source session of a retired instruction."""
    starts = {}
    for session, turn in preference_source_turns(persona, retired_only=True):
        starts[session] = min(turn, starts.get(session, turn))
    return starts


def separate_fact_source(
    context: str, fact: str, preference_part: str | None, evidence: set[str],
) -> bool:
    """A control-bearing turn can still contain a separately sourced life event."""
    if not evidence:
        return True
    if not context or not fact:
        return False
    fact_start = context.find(fact)
    if fact_start < 0 or fact_start != context.rfind(fact):
        return False
    fact_end = fact_start + len(fact)
    for text in evidence:
        # Older records sometimes identify the control span itself. A complete
        # turn alone cannot distinguish its parts; current extraction supplies
        # both literal spans in that case.
        control = preference_part if text == context else text
        if not control:
            return False
        control_start = context.find(control)
        if control_start < 0 or control_start != context.rfind(control):
            return False
        if fact_start < control_start + len(control) and control_start < fact_end:
            return False
    return True
