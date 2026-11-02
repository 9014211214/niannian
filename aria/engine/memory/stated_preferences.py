"""Grounded, revocable communication instructions, separate from person inference.

An LLM proposes a paraphrase; literal owner evidence and negative safety checks
bound admission. Only a separate owner action can activate a proposal. Source
matching is not a semantic proof that the paraphrase is faithful.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from ..context import EngineContext
from .preference_sources import LEGACY as LEGACY_SOURCES, SOURCE_REFS, PreferenceSources
from ..model.persona import (
    PersonaProfile, PreferenceQuote, PreferenceTransition,
    PreferenceUIConfirmation, PreferenceSemanticReview, StatedPreference,
    PREFERENCE_DO_MAX_CHARACTERS, PREFERENCE_QUOTE_MAX_CHARACTERS, PREFERENCE_QUOTE_MAX_COUNT,
    PREFERENCE_SCOPE_LEGACY, PREFERENCE_SCOPE_FULL_SOURCE, PREFERENCE_SCOPE_MODEL,
    preference_scope_max_characters,
)

PREFERENCE_PROTOCOL_VERSION = "2.1"


def _semantic_eligible(review: PreferenceSemanticReview | None) -> bool:
    return bool(review is not None and review.kind == "communication_style"
                and review.owner_requested is True and review.quote_supported is True
                and review.scope_preserved is True and review.no_permission_claim is True)


def preference_admission(row: StatedPreference) -> str:
    """Eligibility is separate from the owner's immutable confirmation history."""
    if row.extraction_source == "rule":
        return "local_rule_preview"
    return "model_reviewed" if _semantic_eligible(row.semantic_review) else "needs_review"


def preference_is_usable(row: StatedPreference, *, allow_rule_preview: bool = False) -> bool:
    admission = preference_admission(row)
    return admission == "model_reviewed" or (allow_rule_preview and admission == "local_rule_preview")

_NO_PERMISSION = ("分享", "转告", "通知", "联系", "告诉", "保密", "授权", "外发")
# Compatibility only: the paused legacy E3 loader imports this old contract.
# Production proposal admission below never calls this positive whitelist.
_REQUEST = re.compile(r"(?:先问|问问我|别替我|不要替我|不用.{0,8}替我|一步|慢一点|说慢|别催|不要催|不用劝|不要劝|别劝|让我.{0,6}自己|我希望|我更喜欢|请你)")
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
_OWNER_DIALOGUE_TELL = re.compile(r"告诉(?:我|你|您)(?=$|[，,。.!！?？;；:：\n])")
_RECIPIENT_SPEECH = re.compile(r"(?:跟(?!上)|向|对(?!方)|给)(" + _CLAUSE_BODY + r"{1,16}?)(?:提(?!示)|说(?!明)|讲|透露|展示|出示|听|看)")
_OUTBOUND_TRANSFER = re.compile(r"发给|发到|转发|发送|转给")
_NEGATED_REQUEST = re.compile(
    r"(?:别(?!人)|不要|不用|不必|不想|不希望|不需要|不愿意|没必要|无需|用不着|不准|不许|不能)"
    + _CLAUSE_BODY + r"{0,8}(?:先问|告诉我(?:下一步|第一步|一步))"
)
_UNSAFE_ADVICE = re.compile(r"(?:停药|加药|减药|药量|剂量|双倍|忽略.{0,8}(?:风险|危险|规则)|跳过.{0,8}(?:核验|确认|权限))")
_MEDICAL_RULE_CONTEXT = re.compile(
    r"药片|吃药|用药|开的药|医嘱"
    r"|(?:不|别|不用|不要|先不|先别)[^。！？]{0,10}(?:就医|复诊|急救|就诊)"
    r"|(?:就医|复诊|急救|就诊)[^。！？]{0,12}(?:不用管|别管|忽略|取消)"
)
_SOURCE_META_CONTEXT = re.compile(
    r"昨天|前天|曾经|发生的事|那是.{0,8}的事|来自|转述|引述|练习册|台词|对白|说明书|读过|读到|念的句子"
    r"|撤回|不确认|不再用|不要再用|别再.{0,8}记|不是.{0,12}(?:要求|意思|让你)|以前你|过去你"
    r"|刚才[^。！？]{0,60}(?:给了|后来)|排小品|我演的|只是练习|随口举例"
)
_TRAIT_OR_CAUSE = re.compile(r"disc|big[ _]?five|cbt|依恋|内向|外向|神经质|人格|性格|诊断|抑郁症|焦虑症|缺乏安全感|导致", re.I)
_REPORTED_CONDITIONAL = re.compile(
    r"(?:小[\u3400-\u9fff]{1,2}|邻居|阿姨|叔叔|主持人|外甥|侄子|同事|老师|老板)"
    + _CLAUSE_BODY + r"{0,12}(?:说|表示|要求|希望|认为|建议|告诉|觉得)"
)
# Negative categories, not a catalogue of permitted ways to ask. These are
# limited syntactic checks, not a replacement for the owner's review.
_NON_OWNER_CONTEXT = re.compile(
    r"[\"“”‘’「」『』]|(?:^|[，,。；;])(?:小|老)[\u3400-\u9fff]{1,2}[，,]你"
    r"|(?:要是|假如).{0,40}(?:你会|你觉得)|(?:他|她|小[\u3400-\u9fff]{1,2})再"
)
_FACT_ONLY = re.compile(
    r"^(?:这|那)[^，,。；;！？]{1,24}每[^，,。；;！？]+[。.]?$"
    r"|^我(?:一向|向来|平常|平时|一直|习惯)[^，,。；;！？]+[。.]?$"
    r"|^(?:下次|以后)?我会[^，,。；;！？]+[。.]?$"
)
_DISCOURSE_PREFIX = re.compile(r"(?:所以)?你(?:记着|记住|听好)|但我这么说吧")
_THIRD_PARTY_ACTION = re.compile(
    r"(?:问(?:问)?|教|催|劝|打断|提醒|提示|回答|回应|倾听|纠正)"
    r"(?:他|她|小[\u3400-\u9fff]{1,2}|女儿|儿子|别人|朋友|医生"
    r"|(?:我|你|您)(?:女儿|儿子|家人|母亲|父亲|朋友|老板|同学))"
)


def _sharing_or_permission(text: str) -> bool:
    # 'Tell me' and 'I tell you' remain inside this dialogue. A recipient such
    # as 'my daughter' must not be mistaken for the owner pronoun prefix.
    bounded = _OWNER_DIALOGUE_TELL.sub("", _OWNER_STEP_TELL.sub("", text))
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
        or _MEDICAL_RULE_CONTEXT.search(text)
        or _SOURCE_META_CONTEXT.search(text)
        or _TRAIT_OR_CAUSE.search(text)
        or _NEGATED_REQUEST.search(text)
        or _REPORTED_CONDITIONAL.search(text)
        or _NON_OWNER_CONTEXT.search(text)
        or _FACT_ONLY.search(text)
        or not _owner_action_targets(text)
    )


def _model_source_safety(text: str) -> bool:
    """Limited negative privacy/medical/diagnostic backstops, not intent parsing.

    Owner intent, reported speech, denial and fact-vs-preference distinctions
    belong to the model's explicit semantic review. A grammatical whitelist
    must not silently replace that review. These checks do not prove safety.
    """
    return not (_sharing_or_permission(text) or _UNSAFE_ADVICE.search(text)
                or _MEDICAL_RULE_CONTEXT.search(text) or _TRAIT_OR_CAUSE.search(text))


def _owner_action_targets(instruction: str) -> bool:
    # Deny explicit third-party targets. Missing a recognised object is not a
    # rejection: natural paraphrases need not fit an affirmative grammar.
    return not _THIRD_PARTY_ACTION.search(instruction)


def _grounded_scope(full_sources: list[str], *, scope_protocol: str = PREFERENCE_SCOPE_LEGACY) -> str | None:
    """Keep all source qualifiers without guessing which words are conditions.

    Short owner turns fit verbatim in scope. Longer turns exceed the existing
    storage bound and are rejected explicitly for length, not semantic support.
    A literal do receives no exception over an equally valid paraphrase.
    """
    limit = preference_scope_max_characters(scope_protocol)
    if scope_protocol == PREFERENCE_SCOPE_FULL_SOURCE:
        full = "；".join(dict.fromkeys(full_sources))
        return full if 1 <= len(full) <= limit else None
    parts = []
    for text in dict.fromkeys(full_sources):
        body = text.strip().rstrip("。.!！;；\n ")
        lead, separator, rest = body.partition("：")
        if separator and _DISCOURSE_PREFIX.fullmatch(lead):
            body = rest  # Evidence still keeps the full unmodified owner turn.
        parts.append(body)
    full = "；".join(parts)
    if 1 <= len(full) <= limit:
        return full
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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


def _quote(raw: object, ctx: EngineContext) -> PreferenceQuote | None:
    if not isinstance(raw, dict):
        return None
    index, text = raw.get("turn_index"), raw.get("text")
    excluded = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
    if (type(index) is not int or not 0 <= index < len(ctx.turns)
            or index in excluded or not isinstance(text, str)
            or not text.strip() or len(text) > PREFERENCE_QUOTE_MAX_CHARACTERS
            or text not in ctx.turns[index].user
            or raw.get("session_id", ctx.session_id) != ctx.session_id):
        return None
    return PreferenceQuote(session_id=ctx.session_id, turn_index=index, text=text)


def extract_preference_rows(rows: object, ctx: EngineContext, persona: PersonaProfile,
                            *, source: str = "model", scope_protocol: str = PREFERENCE_SCOPE_LEGACY,
                            source_protocol: str = LEGACY_SOURCES,
                            source_snapshot: PreferenceSources | None = None,
                            ) -> tuple[list[StatedPreference], list[str]]:
    """Return proposals only; extraction cannot confirm its own interpretation."""
    preference_scope_max_characters(scope_protocol)
    if source_protocol not in (LEGACY_SOURCES, SOURCE_REFS):
        raise ValueError("unknown_preference_source_protocol")
    if source_protocol == SOURCE_REFS and source != "model":
        raise ValueError("preference_source_refs_require_model_route")
    if source != "model" and scope_protocol != PREFERENCE_SCOPE_LEGACY:
        raise ValueError("full_source_scope_requires_model_route")
    if not isinstance(rows, list):
        return [], ["preference_rows_not_array"]
    result: list[StatedPreference] = []
    rejected: list[str] = []
    existing = {row.id: row for row in persona.stated_preferences}
    for raw in rows[:12]:
        if not isinstance(raw, dict):
            rejected.append("preference_row_not_object")
            continue
        if source_protocol == SOURCE_REFS:
            if (set(raw) - {"do", "scope", "source_turn_indices", "confirmation", "supersedes",
                            "revision_turn_index", "semantic_review"}
                    or type(source_snapshot) is not PreferenceSources):
                rejected.append("preference_source_contract_invalid")
                continue
            raw_quotes = raw.get("source_turn_indices")
        else:
            raw_quotes = raw.get("quotes")
        if not isinstance(raw_quotes, list) or not 1 <= len(raw_quotes) <= PREFERENCE_QUOTE_MAX_COUNT:
            rejected.append("preference_quotes_invalid")
            continue
        if source_protocol == SOURCE_REFS:
            quotes = [source_snapshot.quote(index, ctx, persona) for index in raw_quotes]
            if any(row is None for row in quotes) or len({row.turn_index for row in quotes}) != len(quotes):
                rejected.append("preference_source_reference_invalid")
                continue
        else:
            quotes = [_quote(row, ctx) for row in raw_quotes]
        if any(row is None for row in quotes):
            rejected.append("preference_quote_not_verbatim_user_source")
            continue
        quotes = list(quotes)
        # Never check only a selected substring. The model route separates
        # semantic intent from bounded safety checks; the old finite rule
        # parser is retained only for explicitly isolated local previews.
        safe_source = _model_source_safety if source == "model" else _safe_user_instruction
        if not all(safe_source(row.text if source_protocol == SOURCE_REFS else ctx.turns[row.turn_index].user)
                   for row in quotes):
            rejected.append("preference_full_user_context_unsafe")
            continue
        if any(len(row.text if source_protocol == SOURCE_REFS else ctx.turns[row.turn_index].user)
               > PREFERENCE_QUOTE_MAX_CHARACTERS for row in quotes):
            rejected.append("preference_full_user_context_too_long")
            continue
        instruction = raw.get("do")
        if (not isinstance(instruction, str) or not 1 <= len(instruction) <= PREFERENCE_DO_MAX_CHARACTERS
                or not instruction.strip() or _sharing_or_permission(instruction)
                or not _owner_action_targets(instruction)
                or _TRAIT_OR_CAUSE.search(instruction) or _UNSAFE_ADVICE.search(instruction)
                or _MEDICAL_RULE_CONTEXT.search(instruction)):
            rejected.append("preference_not_safe_explicit_instruction")
            continue
        full_sources = [row.text if source_protocol == SOURCE_REFS else ctx.turns[row.turn_index].user
                        for row in quotes]
        supersedes = raw.get("supersedes") or None
        if supersedes:
            if source_protocol == SOURCE_REFS:
                revision_index = raw.get("revision_turn_index")
                revision_valid = (source_snapshot.quote(revision_index, ctx, persona) is not None
                                  and revision_index in raw_quotes)
            else:
                revision_basis = raw.get("revision_basis")
                excluded = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
                revision_valid = (isinstance(revision_basis, str) and revision_basis.strip()
                                  and any(revision_basis in turn.user for index, turn in enumerate(ctx.turns)
                                          if index not in excluded))
            if not revision_valid:
                # The model decides whether this is an explicit revision. Code
                # only checks its current owner source; unsupported replacement
                # intent must not discard an otherwise valid new proposal.
                supersedes = None
        if supersedes and (not isinstance(supersedes, str) or supersedes not in existing
                           or existing[supersedes].status != "confirmed"):
            rejected.append("preference_superseded_target_invalid")
            continue
        semantic_review = None
        if source == "model":
            try:
                semantic_review = PreferenceSemanticReview.model_validate(raw.get("semantic_review"))
            except (ValueError, TypeError):
                rejected.append("preference_semantic_review_required")
                continue
            if not _semantic_eligible(semantic_review):
                rejected.append("preference_not_communication_style")
                continue
        if scope_protocol == PREFERENCE_SCOPE_MODEL:
            # The model's proposed interpretation and its source evidence are
            # different fields. Replacing scope with a quote can remove a
            # condition that the model retained from the wider conversation.
            # Keep its exact interpretation for owner review; do not infer or
            # insert additional source turns, or turn it into confirmation.
            scope = raw.get("scope")
            if (not isinstance(scope, str) or not scope.strip()
                    or not 1 <= len(scope) <= preference_scope_max_characters(scope_protocol)):
                rejected.append("preference_model_scope_invalid")
                continue
            if not _model_source_safety(scope) or not _owner_action_targets(scope):
                rejected.append("preference_model_scope_unsafe")
                continue
        else:
            scope = _grounded_scope(full_sources, scope_protocol=scope_protocol)
            if scope is None:
                rejected.append("preference_full_scope_too_long")
                continue
        now = _now()
        identity = f"{ctx.session_id}|{quotes[0].turn_index}|{instruction}"
        preference_id = "pref_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        if preference_id in existing or any(row.id == preference_id for row in result):
            continue  # Never reactivate a declined/withdrawn record on replay.
        # Keep the complete owner turn, including qualifiers that were outside
        # the selected quote. Never silently truncate evidence to fit the model.
        full_quotes = [PreferenceQuote(session_id=row.session_id, turn_index=row.turn_index,
                                      text=row.text if source_protocol == SOURCE_REFS else ctx.turns[row.turn_index].user)
                       for row in quotes]
        history = [PreferenceTransition(at=now, to="proposed", source=source)]
        try:
            proposal = StatedPreference(
                id=preference_id, do=instruction, scope=scope, scope_protocol=scope_protocol,
                not_implies=["不代表所有场景都适用", "不代表同意执行任何现实动作"],
                quotes=full_quotes, confirmation=None,
                status="proposed", supersedes=supersedes,
                history=history, extraction_source=source, created_at=now, updated_at=now,
                semantic_review=semantic_review,
            )
        except ValueError:
            rejected.append("preference_model_validation_failed")
            continue
        if source_protocol == SOURCE_REFS and any(
                source_snapshot.quote(row.turn_index, ctx, persona) != row for row in quotes):
            # Nothing after source resolution may substitute a later turn.
            # Recheck mutable identity/privacy/source bindings immediately
            # before accepting; stored evidence stays the request snapshot.
            rejected.append("preference_source_reference_changed")
            continue
        result.append(proposal)
    return result, rejected


def heuristic_preferences(ctx: EngineContext, persona: PersonaProfile) -> list[StatedPreference]:
    rows = []
    for index, turn in enumerate(ctx.turns):
        if not turn.user or len(turn.user) > PREFERENCE_QUOTE_MAX_CHARACTERS:
            continue
        instruction = _rule_instruction(turn.user)
        if not instruction:
            continue
        raw = {"do": instruction, "quotes": [{"turn_index": index, "text": turn.user}]}
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
                          *, at: str | None = None, allow_rule_preview: bool = False) -> StatedPreference:
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
    if action == "confirm" and not preference_is_usable(row, allow_rule_preview=allow_rule_preview):
        raise ValueError("preference_review_required_before_confirmation")
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


def confirmed_preference_payloads(persona: PersonaProfile | None, *, allow_rule_preview: bool = False) -> list[dict]:
    """Exact bounded online payload. Withdrawn records never enter context."""
    if persona is None:
        return []
    def confirmation_order(row):
        instant = datetime.fromisoformat(row.updated_at)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        return instant.timestamp(), row.id
    rows = sorted((row for row in persona.stated_preferences if row.status == "confirmed"
                   and preference_is_usable(row, allow_rule_preview=allow_rule_preview)),
                  key=confirmation_order, reverse=True)[:4]
    def quote_text(row):
        return f"session={row.session_id},turn={row.turn_index}: {row.text}"
    return [{"id": row.id, "do": row.do, "scope": row.scope,
             "admission": preference_admission(row), "extraction_source": row.extraction_source,
             "not_implies": list(row.not_implies), "quotes": [quote_text(q) for q in row.quotes],
             "confirmation": (quote_text(row.confirmation) if isinstance(row.confirmation, PreferenceQuote)
                              else row.confirmation.model_dump())} for row in rows]
