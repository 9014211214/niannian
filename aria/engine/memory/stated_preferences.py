"""Grounded, revocable communication instructions, separate from person inference.

Extraction is deliberately conservative: an exact user instruction or one of
the small documented rule paraphrases. An LLM cannot grant sharing permission,
invent a trait, or activate a preference by calling its own summary confirmed.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from ..context import EngineContext
from ..model.persona import (
    PersonaProfile, PreferenceQuote, PreferenceTransition,
    PreferenceUIConfirmation, StatedPreference,
)

_NO_PERMISSION = ("分享", "转告", "通知", "联系", "告诉", "保密", "授权", "外发")
_REQUEST = re.compile(r"(?:先问|问问我|别替我|不要替我|不用.{0,8}替我|一步|慢一点|说慢|别催|不要催|不用劝|不要劝|别劝|让我.{0,6}自己|我希望|我更喜欢|请你)")
_AFFIRMATIVE = {"对", "是", "是的", "没错", "可以", "好的", "好", "就这样", "就这么记", "对就这么记", "对这样记", "按这个记", "记下吧", "可以记下", "就是这个意思"}
_CONFIRM_QUESTION = re.compile(r"(?:这样记|这么记|记下|记住|理解对吗|理解得对吗|是这个意思吗)")
_THIRD_PERSON = r"(?:他|她|女儿|儿子|别人|医生|朋友)"
_CLAUSE_BODY = r"[^，,。.;；!！?？:：\n]"
_REPORTED_OR_HYPOTHETICAL = re.compile(
    _THIRD_PERSON + _CLAUSE_BODY + r"{0,8}(?:说|问|希望|要求)"
    # A speaker may pause before the reporting verb. An independent directive
    # such as 'also ask me first' does not start with a reporting verb.
    r"|" + _THIRD_PERSON + r"(?:" + _CLAUSE_BODY + r"{0,8}[，,。.;；!！?？:：]\s*){1,2}(?:说|问|希望|要求)"
    r"|假如|假设|比如|例如|举个例子"
)
_OWNER_STEP_TELL = re.compile(r"告诉我(?:下一步|第一步|一步)")
_RECIPIENT_SPEECH = re.compile(r"(?:跟|向|对|给)(" + _CLAUSE_BODY + r"{1,16}?)(?:提|说|讲|透露|展示|出示)")
_OUTBOUND_TRANSFER = re.compile(r"发给|发到|转发|发送|转给")
_NEGATED_REQUEST = re.compile(
    r"(?:别(?!人)|不要|不用|不必|不想|不希望|不需要|不愿意|没必要|无需|用不着|不准|不许|不能)"
    + _CLAUSE_BODY + r"{0,8}(?:先问|告诉我(?:下一步|第一步|一步))"
)
_UNSAFE_ADVICE = re.compile(r"(?:停药|加药|减药|药量|剂量|双倍|忽略.{0,8}(?:风险|危险|规则)|跳过.{0,8}(?:核验|确认|权限))")
_TRAIT_OR_CAUSE = re.compile(r"disc|big[ _]?five|cbt|依恋|内向|外向|神经质|人格|性格|诊断|抑郁症|焦虑症|缺乏安全感|导致", re.I)


def _sharing_or_permission(text: str) -> bool:
    # Only an explicit step addressed to 'me' is carved out; 'tell my daughter'
    # and every other disclosure/permission marker retain their old rejection.
    bounded = _OWNER_STEP_TELL.sub("", text)
    return (
        any(word in bounded for word in _NO_PERMISSION)
        or bool(_OUTBOUND_TRANSFER.search(text))
        or any(match.group(1).strip() not in {"我", "本人", "你", "您"}
               for match in _RECIPIENT_SPEECH.finditer(text))
    )


def _safe_user_instruction(text: str) -> bool:
    return not (
        _sharing_or_permission(text)
        or _REPORTED_OR_HYPOTHETICAL.search(text)
        or _UNSAFE_ADVICE.search(text)
        or _TRAIT_OR_CAUSE.search(text)
        or _NEGATED_REQUEST.search(text)
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _normalise(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value).lower()


def _rule_instruction(quote: str) -> str | None:
    if not _safe_user_instruction(quote):
        return None
    if re.search(r"先问(?:问)?我|问问我|别替我(?:定|安排|决定)|不要替我(?:定|安排|决定)", quote):
        return "涉及本人的安排先询问意愿，再一起商量具体时间"
    if re.search(r"一次(?:只)?(?:说|教)?一步|一步一步(?:说|教|来)|一次说一件", quote):
        return "说明操作时一次只说一步，等本人回应再继续"
    if re.search(r"(?:别|不要)催我|(?:说|讲)慢一点|说慢点", quote):
        return "交流时放慢节奏，不催促本人回答或行动"
    return None


def _instruction_scope(instruction: str, quotes: list[PreferenceQuote]) -> str:
    if instruction == "涉及本人的安排先询问意愿，再一起商量具体时间":
        return "涉及本人且尚未确认的活动安排和时间商量"
    if instruction == "说明操作时一次只说一步，等本人回应再继续":
        return "本人请求操作讲解时的讲解步骤"
    if instruction == "交流时放慢节奏，不催促本人回答或行动":
        return "本人提出放慢交流或不要催促的当前情境"
    # Retain the actual context instead of inventing a broad reusable scope.
    return "仅限原话所述情境：" + quotes[0].text[:49]


def _quote(raw: object, ctx: EngineContext) -> PreferenceQuote | None:
    if not isinstance(raw, dict):
        return None
    index, text = raw.get("turn_index"), raw.get("text")
    excluded = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
    if (type(index) is not int or not 0 <= index < len(ctx.turns)
            or index in excluded or not isinstance(text, str)
            or not text.strip() or len(text) > 240
            or text not in ctx.turns[index].user
            or raw.get("session_id", ctx.session_id) != ctx.session_id):
        return None
    return PreferenceQuote(session_id=ctx.session_id, turn_index=index, text=text)


def _confirmation(raw: object, quotes: list[PreferenceQuote], instruction: str,
                  ctx: EngineContext) -> PreferenceQuote | None:
    quote = _quote(raw, ctx)
    if quote is None or quote.turn_index <= max(row.turn_index for row in quotes):
        return None
    # A substring '对' inside '不对' is never confirmation. Require the complete
    # user utterance and an immediately preceding assistant paraphrase question.
    full_user = ctx.turns[quote.turn_index].user
    if _normalise(full_user) not in _AFFIRMATIVE or quote.text != full_user:
        return None
    question = ctx.turns[quote.turn_index - 1].agent
    # Exact bounded paraphrase + single confirmation question. This rejects
    # negated or mixed questions such as '不要按 X 记好吗' or 'X, also notify her?'.
    normal_question = _normalise(question)
    prefixes = {"", "你希望", "您希望", "你说", "您说", "我理解为", "你的意思是", "您的意思是", "也就是说", "我把这句话记为"}
    suffixes = {"这样记下好吗", "这样记好吗", "这样记可以吗", "可以这样记下吗", "我可以这样记下吗", "这样理解对吗", "这样理解得对吗", "是这个意思吗", "这样记对吗", "这样记住好吗"}
    same_instruction = False
    for text in [instruction, *(row.text for row in quotes)]:
        needle = _normalise(text)
        if needle and normal_question.count(needle) == 1:
            prefix, suffix = normal_question.split(needle)
            if prefix in prefixes and suffix in suffixes:
                same_instruction = True
                break
    if not same_instruction or not _CONFIRM_QUESTION.search(question):
        return None
    return quote


def extract_preference_rows(rows: object, ctx: EngineContext, persona: PersonaProfile,
                            *, source: str = "model") -> tuple[list[StatedPreference], list[str]]:
    """Return safe proposals/confirmed rows plus bounded diagnostic codes."""
    if not isinstance(rows, list):
        return [], ["preference_rows_not_array"]
    result: list[StatedPreference] = []
    rejected: list[str] = []
    existing = {row.id: row for row in persona.stated_preferences}
    for raw in rows[:12]:
        if not isinstance(raw, dict):
            rejected.append("preference_row_not_object")
            continue
        raw_quotes = raw.get("quotes")
        if not isinstance(raw_quotes, list) or not 1 <= len(raw_quotes) <= 2:
            rejected.append("preference_quotes_invalid")
            continue
        quotes = [_quote(row, ctx) for row in raw_quotes]
        if any(row is None for row in quotes):
            rejected.append("preference_quote_not_verbatim_user_source")
            continue
        quotes = list(quotes)
        # Check full source turns, not only the model-selected substring. A
        # quote cut out of '不要先问我' or '她说先问我' is not affirmative intent.
        if not all(_safe_user_instruction(ctx.turns[row.turn_index].user) for row in quotes):
            rejected.append("preference_full_user_context_unsafe")
            continue
        instruction = raw.get("do")
        if (not isinstance(instruction, str) or not 1 <= len(instruction) <= 60
                or _sharing_or_permission(instruction) or any(_sharing_or_permission(row.text) for row in quotes)
                or _TRAIT_OR_CAUSE.search(instruction)
                or not any(_REQUEST.search(row.text) for row in quotes)
                or not (re.search(r"问|交流|沟通|说|讲|教|催|劝|回答|回应|倾听", instruction)
                        or _OWNER_STEP_TELL.search(instruction))):
            rejected.append("preference_not_safe_explicit_instruction")
            continue
        if not any(instruction in row.text or instruction == _rule_instruction(row.text) for row in quotes):
            rejected.append("preference_instruction_not_grounded")
            continue
        supersedes = raw.get("supersedes") or None
        if supersedes and (supersedes not in existing or existing[supersedes].status != "confirmed"):
            rejected.append("preference_superseded_target_invalid")
            continue
        # Do not let an extracted scope invent an unrelated setting. A quoted
        # scope is accepted; otherwise the bounded generic communication scope.
        scope = raw.get("scope")
        if not isinstance(scope, str) or not any(scope and scope in row.text for row in quotes):
            scope = _instruction_scope(instruction, quotes)
        now = _now()
        identity = f"{ctx.session_id}|{quotes[0].turn_index}|{instruction}"
        preference_id = "pref_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        if preference_id in existing or any(row.id == preference_id for row in result):
            continue  # Never reactivate a declined/withdrawn record on replay.
        confirmed = _confirmation(raw.get("confirmation"), quotes, instruction, ctx)
        history = [PreferenceTransition(at=now, to="proposed", source=source)]
        if confirmed:
            history.append(PreferenceTransition(at=now, from_status="proposed", to="confirmed", source="conversation", quote=confirmed))
        try:
            result.append(StatedPreference(
                id=preference_id, do=instruction, scope=scope,
                not_implies=["不代表所有场景都适用", "不代表同意执行任何现实动作"],
                quotes=quotes, confirmation=confirmed,
                status="confirmed" if confirmed else "proposed", supersedes=supersedes,
                history=history, extraction_source=source, created_at=now, updated_at=now,
            ))
        except ValueError:
            rejected.append("preference_model_validation_failed")
    return result, rejected


def heuristic_preferences(ctx: EngineContext, persona: PersonaProfile) -> list[StatedPreference]:
    rows = []
    for index, turn in enumerate(ctx.turns):
        if not turn.user or len(turn.user) > 240:
            continue
        instruction = _rule_instruction(turn.user)
        if not instruction:
            continue
        raw = {"do": instruction, "quotes": [{"turn_index": index, "text": turn.user}]}
        if index + 1 < len(ctx.turns):
            raw["confirmation"] = {"turn_index": index + 1, "text": ctx.turns[index + 1].user}
        rows.append(raw)
    proposals, rejected = extract_preference_rows(rows, ctx, persona, source="rule")
    ctx.working_memory["stated_preference_extraction"] = {"source": "rule", "accepted": len(proposals), "rejected": rejected}
    return proposals


def _changed(row: StatedPreference, status: str, *, at: str, source: str,
             confirmation=None) -> StatedPreference:
    data = row.model_dump()
    data.update(status=status, updated_at=at)
    if confirmation is not None:
        data["confirmation"] = confirmation.model_dump()
    data["history"].append(PreferenceTransition(at=at, from_status=row.status, to=status, source=source).model_dump())
    return StatedPreference.model_validate(data)


def merge_preferences(persona: PersonaProfile, rows: list[StatedPreference]) -> None:
    """Merge only new grounded extraction rows; no replay-based reactivation."""
    current = {row.id: row for row in persona.stated_preferences}
    for row in rows:
        if row.id in current:
            continue
        if row.supersedes:
            target = current.get(row.supersedes)
            if target is None or target.status != "confirmed":
                continue
            if row.status == "confirmed":
                current[target.id] = _changed(target, "superseded", at=row.updated_at, source="conversation")
        current[row.id] = row
    persona.stated_preferences = list(current.values())


def transition_preference(persona: PersonaProfile, preference_id: str, action: str,
                          *, at: str | None = None) -> StatedPreference:
    """Apply an explicit user's UI action. Caller owns lock and durable commit.

    All validation precedes mutation, including a replacement's old target.
    A failed save must discard the caller's private profile copy.
    """
    current = {row.id: row for row in persona.stated_preferences}
    row = current.get(preference_id)
    transitions = {("proposed", "confirm"): "confirmed", ("proposed", "decline"): "declined", ("confirmed", "withdraw"): "withdrawn"}
    if row is None:
        raise ValueError("preference_not_found")
    status = transitions.get((row.status, action))
    if status is None:
        raise ValueError("invalid_preference_transition")
    now = at or _now()
    datetime.fromisoformat(now)
    replacement = None
    if status == "confirmed" and row.supersedes:
        target = current.get(row.supersedes)
        if target is None or target.status != "confirmed":
            raise ValueError("preference_superseded_target_invalid")
        replacement = _changed(target, "superseded", at=now, source="ui")
    changed = _changed(row, status, at=now, source="ui", confirmation=PreferenceUIConfirmation(at=now) if status == "confirmed" else None)
    if replacement:
        current[replacement.id] = replacement
    current[row.id] = changed
    persona.stated_preferences = list(current.values())
    return changed


def confirmed_preference_payloads(persona: PersonaProfile | None) -> list[dict]:
    """Exact bounded online payload. Withdrawn records never enter context."""
    if persona is None:
        return []
    def confirmation_order(row):
        instant = datetime.fromisoformat(row.updated_at)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        return instant.timestamp(), row.id
    rows = sorted((row for row in persona.stated_preferences if row.status == "confirmed"),
                  key=confirmation_order, reverse=True)[:4]
    def quote_text(row):
        return f"session={row.session_id},turn={row.turn_index}: {row.text}"
    return [{"id": row.id, "do": row.do, "scope": row.scope,
             "not_implies": list(row.not_implies), "quotes": [quote_text(q) for q in row.quotes],
             "confirmation": (quote_text(row.confirmation) if isinstance(row.confirmation, PreferenceQuote)
                              else row.confirmation.model_dump())} for row in rows]
