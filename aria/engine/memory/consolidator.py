"""复利闭环 —— 对话 → 抽取新记忆 / 更新档案 → 写回 store。**护城河**。

这一层是「理解人」区别于「RAG + 提示词」的根本:对话不只是被检索,而是被
**沉淀回一个会演进的结构化档案**,于是「越用越懂、且无法冷启动」。

抽取器(Extractor)可插拔:
  - HeuristicExtractor:无 LLM 也能跑(把一次会话沉淀成一条 session 摘要记忆),
    用于把「闭环本身」跑通、单测复利。
  - LLMExtractor:会后低温大模型只抽取明确事实、情景记忆与关系上下文。人物假设
    由独立 reflection 职责处理，避免事实抽取器顺手给人下结论。

Consolidator 把抽取结果合并进 persona,交给 PersonaStore 落盘。
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Protocol

from ..context import EngineContext
from ..model.persona import (
    EpisodicMemoryEntry,
    PersonaProfile,
    RelationshipContextEntry,
    RelationshipEvidence,
    RelationshipFactSource,
    StatedPreference,
    PREFERENCE_SCOPE_LEGACY, PREFERENCE_SCOPE_FULL_SOURCE, PREFERENCE_SCOPE_MODEL,
    PreferenceScopeSelection,
    preference_scope_max_characters,
)
from .persona_store import PersonaStore
from .preference_sources import SOURCE_REFS, PreferenceSources, source_protocol as preference_source_protocol
from .referenceability import (
    SOURCE_CONTEXT_VERSION, REVIEWED_FACT_CONTEXT, must_not_reference as _must_not_reference,
    preference_source_turns, separate_fact_source, source_context_sha256,
)
from .stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, heuristic_preferences, merge_preferences,
)
from ..cognitive import (
    cognitive_extraction_prompt, cognitive_sources, cognitive_sensitive,
    extract_cognitive_rows, consolidate_cognitive,
)


@dataclass
class ConsolidationResult:
    new_episodes: list[EpisodicMemoryEntry] = field(default_factory=list)
    formulation_notes: list[str] = field(default_factory=list)
    relationships: list[RelationshipContextEntry] = field(default_factory=list)
    stated_preferences: list[StatedPreference] = field(default_factory=list)
    cognitive_observations: list = field(default_factory=list)


class Extractor(Protocol):
    def extract(self, ctx: EngineContext, persona: PersonaProfile) -> ConsolidationResult: ...


class HeuristicExtractor:
    """无 LLM 的最小抽取:把一次会话沉淀成一条 session 摘要(保守:不可主动引用)。

    目的是把复利「闭环」跑通并可单测;真正的细粒度抽取交给 LLMExtractor。
    """

    def extract(self, ctx: EngineContext, persona: PersonaProfile) -> ConsolidationResult:
        excluded = _sensitive_memory_excluded_turns(ctx)
        sources = [
            (index, turn.user) for index, turn in enumerate(ctx.turns)
            if turn.user and index not in excluded
        ][:3]
        if not sources:
            return ConsolidationResult()
        n = len(persona.episodic_memory) + 1
        ep = EpisodicMemoryEntry(
            id=f"auto_{date.today().isoformat()}_{n}",
            timestamp=date.today().isoformat(),
            type="session_digest",
            summary="本次聊到:" + "；".join(text for _index, text in sources),
            keywords=[],
            sentiment=None,
            referenceable=False,            # 自动抽取保守处理,不主动引用
            source_session_id=ctx.session_id,
            # A multi-turn digest cannot claim a single quoted source turn.
            source_turn_index=sources[0][0] if len(sources) == 1 else None,
            source_quote=sources[0][1] if len(sources) == 1 else None,
        )
        return ConsolidationResult(new_episodes=[ep], stated_preferences=heuristic_preferences(ctx, persona))


_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def _sensitive_memory_excluded_turns(ctx: EngineContext) -> set[int]:
    raw = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
    return {
        item for item in raw
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }


def _normalise_text(value: str) -> str:
    return _NON_WORD.sub("", value or "").lower()


def _grounded_quote(
    value, turn_index, ctx: EngineContext
) -> tuple[str, int] | None:
    quote = " ".join(str(value or "").strip().split())[:240]
    if (
        not isinstance(turn_index, int)
        or isinstance(turn_index, bool)
        or not (0 <= turn_index < len(ctx.turns))
        or turn_index in _sensitive_memory_excluded_turns(ctx)
    ):
        return None
    needle = _normalise_text(quote)
    if len(needle) < 2 or needle not in _normalise_text(
        ctx.turns[turn_index].user
    ):
        return None
    return quote, turn_index


def _verbatim_source_quote(quote: str, source: str) -> str:
    """Recover the user's spelling/punctuation after normalised validation."""
    if quote in source:
        return quote
    indices = [
        index
        for index, character in enumerate(source)
        if _NON_WORD.sub("", character)
        for _normalised in character.lower()
    ]
    needle = _normalise_text(quote)
    start = _normalise_text(source).find(needle)
    if not needle or start < 0:
        raise ValueError("source quote was not grounded")
    return source[indices[start]:indices[start + len(needle) - 1] + 1]


def _strip_fences(raw: str) -> str:
    """剥掉 LLM 常带的 ```json ... ``` 围栏。"""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if "```" in s:
            s = s[: s.rfind("```")]
    return s.strip()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("memory_json_duplicate_key")
        result[key] = value
    return result


def _finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("memory_json_non_finite")
    return value


def _reject_constant(_raw):
    raise ValueError("memory_json_non_finite")


def _parse_json_payload(raw: str):
    """One unambiguous JSON value, optionally after a preamble or in fences.

    Privacy fields must not be overwritten by duplicate keys, including in
    nested objects or the preamble-compatible decoder. Contract errors are
    never retried through a more permissive parser.
    """
    value = _strip_fences(str(raw or ""))
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object,
                               parse_float=_finite_float, parse_constant=_reject_constant)
    try:
        return decoder.decode(value)
    except json.JSONDecodeError:
        starts = [
            index for index in (value.find("{"), value.find("["))
            if index >= 0
        ]
        if not starts:
            raise
        candidate = value[min(starts):]
        parsed, end = decoder.raw_decode(candidate)
        if candidate[end:].strip() not in {"", "```"}:
            raise ValueError("memory_json_trailing_content")
        return parsed


_PREFIX_FIELDS = (
    "stated_preferences", "cognitive_privacy", "cognitive_observations", "episodes", "relationships",
)


class _IncompleteMemoryJSON(Exception):
    pass


def _memory_complete_prefix(raw: str) -> dict:
    """Read complete top-level fields from a syntactically valid JSON prefix.

    No closing character, quote, array member or source text is invented. An
    invalid escape, duplicate key (including inside the unfinished tail), wrong
    field order, trailing content or malformed token rejects the whole prefix.
    This parser is only used after an explicitly observed length termination.
    """
    if not isinstance(raw, str) or len(raw) > 262144:
        raise ValueError("memory_prefix_text_required")
    position, size = 0, len(raw)
    complete = {}
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object,
                               parse_float=_finite_float, parse_constant=_reject_constant)

    def peek():
        nonlocal position
        while position < size and raw[position] in " \r\n\t":
            position += 1
        if position >= size:
            raise _IncompleteMemoryJSON
        return raw[position]

    def expect(character):
        nonlocal position
        if peek() != character:
            raise ValueError("memory_prefix_invalid_syntax")
        position += 1

    def string():
        nonlocal position
        expect('"')
        start = position - 1
        while position < size:
            char = raw[position]
            position += 1
            if char == '"':
                return decoder.decode(raw[start:position])
            if ord(char) < 32:
                raise ValueError("memory_prefix_invalid_string")
            if char == "\\":
                if position == size:
                    raise _IncompleteMemoryJSON
                escaped = raw[position]
                position += 1
                if escaped == "u":
                    for _ in range(4):
                        if position == size:
                            raise _IncompleteMemoryJSON
                        if raw[position] not in "0123456789abcdefABCDEF":
                            raise ValueError("memory_prefix_invalid_escape")
                        position += 1
                elif escaped not in '"\\/bfnrt':
                    raise ValueError("memory_prefix_invalid_escape")
        raise _IncompleteMemoryJSON

    def value(depth=0):
        nonlocal position
        if depth > 64:
            raise ValueError("memory_prefix_excessive_depth")
        char = peek()
        if char == '"':
            return string()
        if char == "{":
            position += 1
            result = {}
            if peek() == "}":
                position += 1
                return result
            while True:
                key = string()
                if key in result:
                    raise ValueError("memory_json_duplicate_key")
                if depth == 0 and (len(result) >= len(_PREFIX_FIELDS) or key != _PREFIX_FIELDS[len(result)]):
                    raise ValueError("memory_prefix_field_order")
                expect(":")
                item = value(depth + 1)
                result[key] = item
                if depth == 0:
                    complete[key] = item
                delimiter = peek()
                if delimiter == "}":
                    position += 1
                    return result
                expect(",")
        if char == "[":
            position += 1
            result = []
            if peek() == "]":
                position += 1
                return result
            while True:
                result.append(value(depth + 1))
                delimiter = peek()
                if delimiter == "]":
                    position += 1
                    return result
                expect(",")
        for literal, decoded in (("true", True), ("false", False), ("null", None)):
            if char == literal[0]:
                remaining = raw[position:position + len(literal)]
                if remaining != literal:
                    if position + len(remaining) == size and literal.startswith(remaining):
                        raise _IncompleteMemoryJSON
                    raise ValueError("memory_prefix_invalid_literal")
                position += len(literal)
                return decoded
        if char in "-0123456789":
            start = position
            while position < size and raw[position] not in " \r\n\t,]}":
                position += 1
            token = raw[start:position]
            if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token):
                if position == size and (
                    token == "-" or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?", token)
                    or re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.", token)
                ):
                    raise _IncompleteMemoryJSON
                raise ValueError("memory_prefix_invalid_number")
            return decoder.decode(token)
        raise ValueError("memory_prefix_invalid_value")

    try:
        if peek() != "{":
            raise ValueError("memory_prefix_object_required")
        value()
        if raw[position:].strip():
            raise ValueError("memory_prefix_trailing_content")
    except _IncompleteMemoryJSON:
        pass
    return complete


def _prefix_memory_rows(data: dict, *, truncated: bool, cognitive_enabled: bool):
    required = set(_PREFIX_FIELDS[:3] if truncated else _PREFIX_FIELDS)
    if not isinstance(data, dict) or not required <= set(data) or set(data) - set(_PREFIX_FIELDS):
        raise ValueError("memory_prefix_required_complete_fields")
    for name, rows in data.items():
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("memory_prefix_object_arrays_required")
    if not cognitive_enabled and data["cognitive_observations"]:
        raise ValueError("cognitive observation requires owner consent")
    return data["cognitive_observations"]


def _strict_grounded_quote(value, turn_index, ctx):
    if (not isinstance(value, str) or not 2 <= len(value) <= 240 or type(turn_index) is not int
            or not 0 <= turn_index < len(ctx.turns)
            or turn_index in _sensitive_memory_excluded_turns(ctx)
            or value not in ctx.turns[turn_index].user):
        return None
    return value, turn_index


def _memory_protocol_rows(data) -> tuple[list[dict], list[dict]]:
    """Validate the extractor envelope before an empty result can be trusted."""
    if isinstance(data, list):
        # 兼容 0.1 版顶层数组；数组元素仍必须是协议条目。
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("legacy episodes must contain JSON objects")
        return data, []
    if not isinstance(data, dict):
        raise ValueError("memory response must be an object or legacy array")
    if "episodes" not in data or "relationships" not in data:
        raise ValueError("memory response is missing protocol fields")
    episode_rows = data["episodes"]
    relationship_rows = data["relationships"]
    if not isinstance(episode_rows, list) or not isinstance(
        relationship_rows, list
    ):
        raise ValueError("memory protocol fields must be arrays")
    if not all(isinstance(item, dict) for item in episode_rows):
        raise ValueError("episodes must contain JSON objects")
    if not all(isinstance(item, dict) for item in relationship_rows):
        raise ValueError("relationships must contain JSON objects")
    return episode_rows, relationship_rows


def _cognitive_protocol_rows(data) -> list[dict]:
    """An absent C1 field is a failed extraction, never zero observations."""
    if not isinstance(data, dict):
        raise ValueError("cognitive memory response must be a JSON object")
    for name in ("episodes", "relationships", "stated_preferences", "cognitive_privacy", "cognitive_observations"):
        rows = data.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("cognitive memory response requires five object arrays")
    return data["cognitive_observations"]


def _cognitive_privacy_indices(data, ctx, persona, observation_rows) -> set[int]:
    """Validate semantic privacy markings before any ordinary extraction.

    This field is required even with monitoring disabled: privacy filtering
    does not grant consent to collect observations. All proposed observations
    also mask their literal source, including non-counting/context proposals.
    """
    if not isinstance(data, dict) or not isinstance(data.get("cognitive_privacy"), list):
        raise ValueError("memory privacy classification is required")
    rows = data["cognitive_privacy"]
    if len(rows) > 120:
        raise ValueError("memory privacy classification exceeds turn limit")
    sources = {index: turn.user for index, turn in enumerate(ctx.turns) if turn.user}
    for row in cognitive_sources(persona, ctx):
        sources[row.turn_index] = row.text
    excluded = {index for index, text in sources.items() if cognitive_sensitive(text)}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"turn_index", "evidence_quote"}
                or type(row["turn_index"]) is not int or row["turn_index"] not in sources
                or not isinstance(row["evidence_quote"], str) or not row["evidence_quote"].strip()
                or len(row["evidence_quote"]) > 2000
                or row["evidence_quote"] not in sources[row["turn_index"]]):
            raise ValueError("memory privacy source is not literal owner speech")
        excluded.add(row["turn_index"])
    for row in observation_rows:
        index, quote = row.get("turn_index"), row.get("evidence_quote")
        if (type(index) is int and index in sources and isinstance(quote, str)
                and quote.strip() and quote in sources[index]):
            excluded.add(index)
    return excluded


def _source_reference_memory_prompt(fact_protocol=None) -> str:
    """Current typed route: one domain instruction, one provider schema.

    Historical text-format routes keep their own examples unchanged. Source
    selection, privacy and persistence still use the existing local validators.
    """
    fact_instruction = """每条事实的evidence_quote必须是同一轮本人原话的连续逐字子串，turn_index沿用输入原编号。"""
    from .memory_contract import FACT_SOURCE_REFS, LEGACY_FACT_SOURCE_REFS
    if fact_protocol == LEGACY_FACT_SOURCE_REFS:
        fact_instruction = """从各轮“事实出处”选择编号：null代表该轮完整本人原话，其余值是可选原文片段。
episodes只选source_id；关键词只选该出处内的原词。relationships选同一人物的source_ids，
每个出处都明确提到此人；称谓要有同一出处支持，未知就留空。代码按编号保存原文，不写摘要或重抄引文。
结合上下文判断谁在说谁的事；引述、假设、否定和纠正不得改成本人经历。片段单独拿出会改变含义时选整轮或不记。"""
    elif fact_protocol == FACT_SOURCE_REFS:
        fact_instruction = """在描述这件事的本人轮中选“事实出处”原键填source_id或source_ids：tN指整轮，tN.sK指该轮的一个原文片段。不要填数字或从别轮数位置。
逐轮分清本人在说明怎样交流，还是在讲自己的生活。解释旧约定、澄清误解或否认一种动机都可能同时讲到值得记的处境、事件或感受，即使无需新提偏好，也要单独选这件生活事实。
混合轮的事实出处选可独立成立的生活句，preference_source_id选同轮与之不重叠的交流要求或解释句；在episodes或relationships选一次即可。不能把被解释的要求、相应否认或确认本身写成经历或关系事实。
本人的经历、感受优先放episodes，关键词取自选句，纯生活事实的preference_source_id填null。relationships只选直接说明该人身份或所做之事的出处：能独立支持关系且不含相处说明时，preference_source_id填independent_relationship。仅在本人解释中出现姓名不证明那是亲属的事；不能把代词猜成某个人。
结合整轮理解引述、假设、否定和纠正。片段脱离语境会失真就不单独选。"""
    turn_label = "输入中本人原话的turn_index" if fact_protocol else "本人原话前[turn N]的原编号N"
    return """你负责会后事实整理，不聊天、推测关系、诊断人格或执行动作。
只提取本人明确表达且值得长期记住的内容。输出一个完整JSON对象，字段和类型遵循请求中的JSON Schema；
不加Markdown或解释，不为凑结构编造条目，无适用内容用空数组。字符串按JSON转义，解码后引文仍须逐字不变。

事实与隐私：
- episodes记经历；relationships只记本人明确提到、已知姓名的真实人物及关系事实，不猜亲属关系。
""" + fact_instruction + """
助手话语、旧摘要及旧说明不是本次本人事实；来源失配或已被隐私排除的条目不保存。
- referenceable是布尔值，仅表示与本人对话时能否主动引用该事实，不是公开或对外分享许可。
按完整语境逐条判断；一次选句分享或外传限制只约束相应对象、内容和条件，不自动禁止所有内部引用。
真正保密、私密、敏感（逝者、家庭隐情、自伤、债务/转账/存款/账户、证件或联系方式）以及本人明确不愿再提的内容仍为false。
- 必须审阅全部本人轮次并输出cognitive_privacy，没有也显式[]。它只标本人认知或日常功能相关私密内容，
不是通用私密标签；结合语境判断记忆、找词、定向、重复和功能变化，不以关键词为准。
普通情绪、生活选择和关系说明本身不构成认知功能变化。标注只含原turn_index与逐字evidence_quote；
该轮不得进入事实、关系、偏好或人物分析。隐私审阅不等于同意变化提醒，未开启时认知观察必须为空。

本人相处说明：
- 只提议本人明确要求助手今后怎样交流。先判断是否有新的要求或明确修改，再写do和scope；
纯解释、感受、回应误解、他人引述、否认/撤回旧要求，不自动变成长久指令。
否定句也可以表达真实交流要求，按实际含义判断，不靠句式。
- 先对照旧已确认说明：同含义同范围的复述、澄清不重复提案，也不充当修改依据。
本人分几轮给同一种相处方式补充做法、再说明范围或例外时，整理成一条完整约定；能独立适用的不同要求分别提取。
承接旧要求时保留未明确改变的条件、对象、时限和例外，没重复旧限定不等于取消它；
无关新要求也不能被强套旧范围。指向或本人出处不明就不提取。
- do写完整、简洁的交流行为摘要；scope写完整适用条件、对象、时限和例外，两项共同表达本人要求。
不必在do重复整段scope；不能截句、删条件来凑长度，也不能增加权限、人格或医疗结论。
- [preference_source_refs_v1]source_turn_indices选择""" + turn_label + """，不重新编号。
行为、跨轮条件和例外的出处都须选入；代码另存完整逐字原话，不输出quotes、text或revision_basis。
- confirmation永远为null，提议不自动生效。只有本人本次明确修改旧说明才引用supersedes，
revision_turn_index必须是所选来源中的实际修改轮；纯澄清不是修改。独立新要求两者都为null。
- [preference_semantics_v1]先区分communication_style、permission_boundary、action_request、fact、person_inference、unclear。
向谁披露什么、允许或禁止代办/参加/代为承诺属于权限，即使是限制也不变成交流方式；本次任务不是长期偏好。
混合句无法独立保留完整交流要求时不提取，不剥去对象把权限要求伪装成相处说明。
只保存communication_style且四项布尔值均为true的候选：owner_requested=本人本次确实新增/明确修改；
quote_supported=本人来源支持do；scope_preserved=结合完整语境及旧范围，do与scope合起来没有扩大或遗漏限定；
no_permission_claim=两项都不授予、限制或推断现实行动/披露权限。
不输出consent_to_share；模型整理、可引用事实及人物判断都不能创建分享或护理授权。
"""


class LLMExtractor:
    """会后事实抽取器，不诊断人格，也不决定如何回应。

    它与在线生成、人物反思使用独立客户端和触发时机。失败不影响主流程；
    隐私由完整本人来源上下文和关系分享授权的确定性边界兜底。
    """

    def __init__(self, llm, *, prefix_recovery: bool = False):
        self.llm = llm
        if type(prefix_recovery) is not bool:
            raise ValueError("memory_prefix_recovery_must_be_boolean")
        self.prefix_recovery = prefix_recovery
        self.last_status = "not_run"
        self.last_error_type: str | None = None
        self.last_prefix_summary: dict | None = None

    def extract(self, ctx: EngineContext, persona: PersonaProfile) -> ConsolidationResult:
        from ..policy.online_decision import EVIDENCE_DECISION_VERSIONS
        self.last_prefix_summary = None
        ctx.working_memory.pop("_memory_prefix_recovery", None)
        scope_protocol = (
            PREFERENCE_SCOPE_FULL_SOURCE
            if ctx.working_memory.get("_online_decision_version") in EVIDENCE_DECISION_VERSIONS
            else PREFERENCE_SCOPE_LEGACY
        )
        # Only the trusted memory-client configuration selects this protocol.
        # Owner text, model fields and saved proposals cannot opt into it.
        configured_scope = getattr(self.llm, "preference_scope_protocol", None)
        if configured_scope is not None:
            if configured_scope != PREFERENCE_SCOPE_MODEL:
                raise ValueError("unknown_memory_preference_scope_protocol")
            from .preference_scope import validate_model_scope_client
            validate_model_scope_client(self.llm)
            scope_protocol = PREFERENCE_SCOPE_MODEL
        scope_selection = PreferenceScopeSelection(
            protocol=scope_protocol, max_characters=preference_scope_max_characters(scope_protocol),
        ).model_dump()
        ctx.working_memory["_preference_scope_selection"] = scope_selection
        source_protocol = preference_source_protocol(self.llm)
        from ..policy.memory_structured import memory_schema_protocol, memory_structured_route
        from .memory_contract import FACT_PROTOCOLS, LEGACY_FACT_SOURCE_REFS
        schema_source_refs = (
            source_protocol == SOURCE_REFS and memory_structured_route(self.llm) is not None
        )
        fact_protocol = memory_schema_protocol(self.llm) if schema_source_refs else None
        indexed_facts = fact_protocol in FACT_PROTOCOLS
        ctx.working_memory["_memory_privacy_review_required"] = True
        ctx.working_memory.pop("_memory_privacy_review_completed", None)
        excluded = _sensitive_memory_excluded_turns(ctx)
        source_snapshot = PreferenceSources(ctx.user_id, ctx.session_id, tuple(
            (index, turn.user, turn.agent) for index, turn in enumerate(ctx.turns) if index not in excluded
        ))
        if indexed_facts:
            turns_with_preference = {index for session, index in preference_source_turns(persona)
                                     if session == ctx.session_id}
            convo, fact_catalog = source_snapshot.fact_input(
                legacy_ids=fact_protocol == LEGACY_FACT_SOURCE_REFS, preference_turns=turns_with_preference)
        else:
            convo = "\n".join(
                f"[turn {index}] 用户:{user}\n小予:{agent}"
                for index, user, agent in source_snapshot.turns
            )
        cognitive_system, cognitive_user = cognitive_extraction_prompt(persona, ctx)
        if cognitive_user:
            convo += cognitive_user
        if not convo.strip():
            self.last_status = "skipped"
            self.last_error_type = None
            ctx.working_memory["_memory_privacy_review_completed"] = True
            return ConsolidationResult()
        if persona.stated_preferences:
            previous = [
                {"id": row["id"], "do": row["do"], "scope": row["scope"]}
                for row in confirmed_preference_payloads(persona)
            ]
            if previous:
                convo += (
                    "\n\n旧已确认相处说明（供比对已有要求及其完整适用范围；不是本次用户原话、"
                    "新证据或新提案的确认，不能代替本次引用；只有明确修改时才供supersedes引用）：\n"
                    + json.dumps(previous, ensure_ascii=False)
                )
        messages = [
            {"role": "system", "content": _source_reference_memory_prompt(fact_protocol if indexed_facts else None) if schema_source_refs else (
                "你负责会后事实抽取，不做人格诊断。基于对话只抽取用户明确表达、"
                "值得长期记住的事实，不推测关系或心理。[memory_json_escaping_v2]只输出一个完整、合法的 JSON 对象，不加 Markdown 围栏或说明文字。"
                "所有字符串中的双引号、反斜杠、换行等字符必须按 JSON 语法转义；"
                "evidence_quote 也必须转义，但 JSON 解码后的引文必须保持用户原话逐字不变。"
                "不要为了凑齐结构而编造事实或偏好；无适用内容的数组使用空数组。输出 JSON："
                '{"episodes":[{"summary":"","keywords":[],"referenceable":false,'
                '"evidence_quote":"逐字用户原话","turn_index":0}],'
                '"relationships":[{"name":"","relation":"","facts":['
                '{"value":"","evidence_quote":"逐字用户原话","turn_index":0}],'
                '"referenceable":false}],"stated_preferences":['
                '{"do":"本人提出的相处方式，60字内；可忠实转述'
                + ("所选本人来源" if source_protocol == SOURCE_REFS else "quotes") +
                '，不新增条件、权限、人格或医疗结论",'
                '"scope":"本人说明的适用场景",'
                + ('"source_turn_indices":[0],' if source_protocol == SOURCE_REFS else
                   '"quotes":[{"turn_index":0,"text":"逐字用户原话"}],') +
                '"confirmation":null,'
                '"supersedes":null,'
                + ('"revision_turn_index":null,' if source_protocol == SOURCE_REFS else '"revision_basis":null,') +
                '"semantic_review":{"version":"preference_semantics_v1",'
                '"kind":"communication_style|permission_boundary|action_request|fact|person_inference|unclear",'
                '"owner_requested":false,"quote_supported":false,"scope_preserved":false,'
                '"no_permission_claim":false}}],"cognitive_privacy":[{"turn_index":0,"evidence_quote":"认知隐私的逐字本人原话"}]'
                + (
                    ',"cognitive_observations":[{"domain":"memory|language|orientation|daily_function|repetition",'
                    '"evidence_quote":"逐字本人原话","turn_index":0,"functional_impact":false,'
                    '"signal":"explicit_change|context",'
                    '"event_kind":"new_event|same_event","prior_event_id":null}]}。'
                    if cognitive_system else '}。'
                ) +
                "关系只记录用户明确提到的真实人物；不知道姓名就不抽取。"
                "每条事实必须给出同一轮用户原话中的逐字 evidence_quote；不能"
                "仅给摘要或模型推断。turn_index 必须沿用输入中的原始编号；系统会"
                "丢弃无法与该轮原话匹配或已被隐私层排除的条目。"
                "referenceable必须为JSON布尔值true或false，不能用字符串或数字。"
                + ("referenceable仅表示助手在与本人对话中可否主动引用该条事实，不是公开或对外分享许可。"
                   "逐条按完整语境判断：一次选句分享或限制外传，只约束相应分享对象、内容和条件；"
                   "仅凭该外传限制，不能推定全部事实或向本人主动引用也受禁止。"
                   "真正保密/私密/敏感（逝者、家庭隐情、自伤、债务/转账/存款/账户、证件或联系方式）"
                   "或本人明确不愿再被提起的内容仍一律referenceable=false。"
                   if schema_source_refs else
                   "私密/敏感（明确要求保密、逝者、家庭隐情、自伤、债务/转账/"
                   "存款/账户、证件或联系方式）一律 referenceable=false。") +
                "必须审阅输入中的全部本人轮次，并输出cognitive_privacy数组；没有认知隐私也须显式输出[]。"
                + ("cognitive_privacy只标本人认知或日常功能方面的私密内容，不是通用私密标签。"
                   "结合完整语境判断记忆、找词、定向、重复或功能变化，不以关键词为准；"
                   "普通情绪、生活选择或关系说明本身不能被推断为认知功能变化。"
                   if schema_source_refs else
                   "用语义判断本人记忆、找词、时间地点、重复或日常功能变化相关的私密内容，不以关键词为必要条件。") +
                "每条标注只有turn_index和evidence_quote，原话须逐字可查。"
                "标注轮次不得进入episodes、relationships、stated_preferences或人物分析；其余普通事实照常抽取。"
                "此隐私排除不等于开启变化提醒，未开启时禁止输出认知观察或累计趋势。"
                "不要输出 consent_to_share，系统永远不会因模型抽取自动授权分享。"
                "相处说明只记录本人明确要求助手今后怎样交流，不记录人格标签、因果推测或分享通知权限。"
                "先判断本次是否真的提出新的交流行为要求，再形成do和scope；"
                "不得先把解释改写成指令，再据此认定本人提出了要求。"
                "先与旧已确认说明比对：含义及适用范围未变的复述、澄清不重复提案，也不是修改依据。"
                "解释原有做法的含义、原因，回应误解或描述当下感受，即使值得回应，也不等于新增长期交流要求；"
                "没有新要求或明确修改，该内容就不进入stated_preferences。"
                "本人明确要求助手以后改变回应方式时仍可提案，不能仅因同时含有解释就漏掉真实的新要求。"
                "同一条相处说明的连续补充条件和例外应整合为一条提案；"
                "先确定共同指向，再合并完整要求，不能仅按动作或轮次拆分，也不能仅因相邻或同场景合并。"
                "承接既有要求的补充必须保留未被本人明确改变的原条件、对象、时限和例外；"
                "本轮未重复旧限定不代表取消限定，不得拆成范围更宽的独立说明。"
                "不同或无关的相处说明仍分别提取，按各自本人要求确定范围，不能强套旧说明的范围。"
                "指向不明、不能确定是否新增要求或缺少完整本人出处时不提取，不补猜。"
                "[preference_semantics_v1]按上述完整语境作semantic_review分类，布尔值只可为true/false。"
                "communication_style只改变本人和助手之间怎样交流；permission_boundary是向谁披露什么、"
                "允许或禁止代办、参加、代为承诺等权限边界，即使是限制性要求也不能记成相处方式。"
                "action_request是本次具体任务，fact是事实，person_inference是人物推断，含糊则unclear。"
                "owner_requested指本人本次确实新增或明确修改了该交流要求，不是对既有要求的纯解释、"
                "他人引述、本人否认或撤回该要求、举例；"
                "本人用否定句提出有效交流要求，与否认该要求不同；肯定或否定句式本身不决定类别，按实际意思判断。"
                "quote_supported指do有逐字本人原话支持；scope_preserved须对照完整语境及旧已确认范围，"
                "确认do与scope没有扩大或遗漏条件、对象、时限和例外，不能只对照当前引用一句话；"
                "no_permission_claim指do与scope都不包含授予、限制或推断现实行动/披露权限的内容。"
                "混合句中如无法独立保留完整的交流要求，不能改写成宽泛说明；不要把权限要求剥去对象伪装成交流方式。"
                "只将communication_style且四项均为true的候选放入stated_preferences；其他类别不在此保存。"
                + ("这一分类不创建护理或分享授权；事实的referenceable仍按上述与本人对话的引用及隐私边界判断。"
                   if schema_source_refs else
                   "这一分类不创建护理授权，也不让隐私要求变成可主动引用的公开事实；仍遵守上述referenceable隐私约束。") +
                "do忠实表达交流行为及其先后条件，scope完整表达适用范围；不能只截取条件之后的动作或为满足长度上限删掉限定。"
                + (("[preference_source_refs_v1]source_turn_indices选支持本条do与scope的用户原话前[turn N]的N，"
                    "不按提案或所选轮次重编号。跨轮条件和例外须一并选入。"
                    "代码保存完整本人原话，每条最多三轮、每轮最多240字；不输出quotes、text或revision_basis。"
                    if schema_source_refs else
                    "[preference_source_refs_v1]source_turn_indices只选实际输入中的原turn编号，不抄写引文；"
                    "代码按这些编号保存该轮完整本人原话，每条最多三轮、每轮最多240字。"
                    "范围条件和例外分属多轮时须逐轮选择，不能只选行为或解释所在轮次；"
                    "不得输出quotes、text或revision_basis。") if source_protocol == SOURCE_REFS else
                   "quotes保留该轮完整本人原话（每轮最多240字），范围条件和例外分属多轮时须逐轮引用，"
                   "不能只引用行为或解释所在轮次；无法完整表达和引用就不提取。") +
                "confirmation一律为null，模型只能提议，必须由本人另行明确操作确认才生效。"
                "除非老人在本次会话中明确要求修改原来的说法，否则不要提出 supersedes；不要把原有条件改宽或删掉。"
                + (("提出supersedes时，revision_turn_index选择本次本人明确表示修改的原turn编号，"
                    "且必须包含在source_turn_indices中；纯澄清不能充当修改依据。"
                    "独立新要求的revision_turn_index和supersedes都为null。") if source_protocol == SOURCE_REFS else
                   "提出supersedes时，revision_basis须逐字引用本次会话中老人表示修改的那句话；"
                   "纯澄清不能充当revision_basis。独立新要求的revision_basis和supersedes都为null。") +
                "不能用助理原话、旧说明或自己的总结充当本次本人提出新要求的证据。"
                "没有符合条件的相处说明就给空数组。"
                "只输出 JSON，不要解释。"
            )},
            {"role": "user", "content": convo},
        ]
        if self.prefix_recovery:
            if not schema_source_refs:
                prompt = messages[0]["content"]
                start = prompt.index("输出 JSON：") + len("输出 JSON：")
                end = prompt.index("关系只记录用户明确提到的真实人物")
                example = json.loads(prompt[start:end].removesuffix("。"))
                example.setdefault("cognitive_observations", [])
                messages[0]["content"] = (
                    prompt[:start] + json.dumps({name: example[name] for name in _PREFIX_FIELDS}, ensure_ascii=False)
                    + "。" + prompt[end:]
                )
            messages[0]["content"] += (
                "[memory_complete_prefix_v1]"
                + ("按请求中Schema的五段顺序输出，" if schema_source_refs else "按示例的五段顺序输出，")
                + "先相处说明，再完整隐私审阅、"
                "完整认知观察、事实、关系。即使未开启认知提醒，也必须在第三段显式输出"
                "cognitive_observations:[]；它不代表取得观察同意。前两项隐私来源审阅完成后才输出普通事实。"
                "关系只写关系身份和新增关系事实；同一原话若用于关系事实，不要再重复放入episodes；"
                "已有事实不在relationships中重复展开。"
                + ("每项新增关系事实选对应本人原文编号。" if indexed_facts else
                   "每项新增关系事实仍必须附逐字出处，不能改写或省略引文。")
            )
        if cognitive_system:
            messages[0]["content"] += cognitive_system
        if scope_protocol == PREFERENCE_SCOPE_FULL_SOURCE:
            messages[0]["content"] += (
                "[preference_scope_full_source_v1]scope须完整保留所有引用来源轮次里的条件、对象和时限，"
                "优先逐字保留完整本人来源；不能用do的短概括代替scope，不能扩大适用范围。"
                "每条最多引用三轮，每轮完整原话最多240字；系统将完整来源原样以分号连接保存，"
                f"scope容量为{scope_selection['max_characters']}字，不剪裁限定或换词来满足容量。"
                "do仍须忠实且不超过60字，不能为凑长度丢失条件；无法完整表达就不提取。"
            )
        elif scope_protocol == PREFERENCE_SCOPE_MODEL:
            messages[0]["content"] += (
                "[preference_scope_model_v1]do与scope分别保存你的提议原文，"
                + ("代码依据source_turn_indices另存逐字来源，" if source_protocol == SOURCE_REFS else "quotes另存逐字来源，") +
                "不会用引文替换scope。scope须保留完整语境中的条件、对象和时限，不能扩大范围；"
                "跨轮提出或补充的限制须一并表达，"
                + ("并引用相应的完整本人轮次。" if source_protocol == SOURCE_REFS else
                   "并在quotes引用相应的完整本人轮次。") +
                f"每条最多引用三轮，每轮最多240字；scope最多{scope_selection['max_characters']}字，do最多60字。"
                "无法在这些边界内完整保留要求和依据就不提取；这些都是待本人另行确认的提议。"
            )
        elif schema_source_refs:
            # Explicit typed legacy-scope configurations remain supported;
            # their schema cannot express the length of selected source text.
            messages[0]["content"] += (
                "每条最多引用三轮，每轮完整本人原话最多240字；无法保留完整来源就不提取。"
            )
        self.last_error_type = None
        try:
            raw = self.llm.complete(messages)
        except Exception:
            self.last_status = "failed"
            self.last_error_type = "model_call_failed"
            return ConsolidationResult()
        truncated = self.prefix_recovery and getattr(self.llm, "last_finish_reason", None) == "length"
        if self.prefix_recovery:
            finish = getattr(self.llm, "last_finish_reason", None)
            self.last_prefix_summary = {
                "protocol": "memory_complete_prefix_v1", "finish_reason": finish if finish in {"stop", "length"} else "unknown",
                "truncated": truncated, "status": "not_parsed", "completed_fields": [],
                "missing_fields": list(_PREFIX_FIELDS), "privacy_sources_validated": False,
            }
        try:
            data = _memory_complete_prefix(raw) if truncated else _parse_json_payload(raw)
            if indexed_facts and isinstance(data, dict):
                for field in ("episodes", "relationships"):
                    if isinstance(data.get(field), list):
                        data[field] = [item for item in data[field] if isinstance(item, dict)]
            if self.prefix_recovery:
                self.last_prefix_summary.update(
                    completed_fields=[name for name in _PREFIX_FIELDS if name in data],
                    missing_fields=[name for name in _PREFIX_FIELDS if name not in data],
                )
                cognitive_rows = _prefix_memory_rows(data, truncated=truncated, cognitive_enabled=bool(cognitive_system))
                # Both completed arrays contribute to the exclusion mask. An
                # invalid observation citation is not silently treated as [] by
                # partial recovery, even if ordinary extraction looks valid.
                sources = {index: turn.user for index, turn in enumerate(ctx.turns) if turn.user}
                sources.update({row.turn_index: row.text for row in cognitive_sources(persona, ctx)})
                for row in cognitive_rows:
                    required_observation = {"domain", "evidence_quote", "turn_index", "functional_impact", "event_kind", "prior_event_id"}
                    if (not required_observation <= set(row) or set(row) - required_observation - {"signal"}
                            or row["domain"] not in {"memory", "language", "orientation", "daily_function", "repetition"}
                            or type(row["turn_index"]) is not int or row["turn_index"] not in sources
                            or not isinstance(row["evidence_quote"], str) or not 4 <= len(row["evidence_quote"]) <= 240
                            or row["evidence_quote"] not in sources[row["turn_index"]]
                            or type(row["functional_impact"]) is not bool
                            or row["event_kind"] not in {"new_event", "same_event"}
                            or row["prior_event_id"] is not None and not isinstance(row["prior_event_id"], str)):
                        raise ValueError("memory_prefix_invalid_observation")
                # Missing tail fields are absent, not falsely reported as empty.
                # Only actually completed arrays are iterated below.
                episode_rows = data["episodes"] if "episodes" in data else ()
                relationship_rows = data["relationships"] if "relationships" in data else ()
            else:
                episode_rows, relationship_rows = _memory_protocol_rows(data)
                cognitive_rows = _cognitive_protocol_rows(data) if cognitive_system else []
            privacy_indices = _cognitive_privacy_indices(data, ctx, persona, cognitive_rows)
            if not cognitive_system and data.get("cognitive_observations"):
                raise ValueError("cognitive observation requires owner consent")
        except (TypeError, ValueError, json.JSONDecodeError):
            self.last_status = "failed"
            self.last_error_type = "invalid_json_response"
            if self.last_prefix_summary is not None:
                self.last_prefix_summary["status"] = "rejected"
                ctx.working_memory["_memory_prefix_recovery"] = dict(self.last_prefix_summary)
            return ConsolidationResult()
        self.last_status = "partial" if truncated else "ok"
        self.last_error_type = "truncated_response" if truncated else None
        if self.last_prefix_summary is not None:
            self.last_prefix_summary["privacy_sources_validated"] = True
            self.last_prefix_summary["status"] = self.last_status
            ctx.working_memory["_memory_prefix_recovery"] = dict(self.last_prefix_summary)
        # Commit the mask before grounding ordinary facts, relationships or R1.
        if indexed_facts:
            excluded.update(_sensitive_memory_excluded_turns(ctx))
        ctx.working_memory["_sensitive_memory_excluded_turns"] = sorted(excluded | privacy_indices)
        ctx.working_memory["_memory_privacy_review_completed"] = True

        preferences, rejected = extract_preference_rows(
            data.get("stated_preferences", []) if isinstance(data, dict) else [], ctx, persona,
            scope_protocol=scope_protocol,
            source_protocol=source_protocol, source_snapshot=source_snapshot,
        )
        preference_sources = preference_source_turns(persona, preferences)
        if indexed_facts:
            episode_rows, relationship_rows = source_snapshot.bind_facts(
                fact_catalog, episode_rows, relationship_rows, ctx, persona, legacy_ids=fact_protocol == LEGACY_FACT_SOURCE_REFS)
            # A clarification can be grounded even when it proposes no new
            # preference. Apply that same exclusion to both fact channels.
            for source in [*episode_rows, *(s for row in relationship_rows for s in row["facts"])]:
                if control := source.get("preference_source_quote"):
                    preference_sources.setdefault((ctx.session_id, source["turn_index"]), set()).add(control)
        eps: list[EpisodicMemoryEntry] = []
        existing_ids = {episode.id for episode in persona.episodic_memory}
        next_number = len(persona.episodic_memory) + 1
        today = date.today().isoformat()
        for item in episode_rows if isinstance(episode_rows, list) else []:
            if not isinstance(item, dict):
                continue
            grounded = (_strict_grounded_quote if self.prefix_recovery else _grounded_quote)(
                item.get("evidence_quote"), item.get("turn_index"), ctx
            )
            if grounded is None:
                continue
            quote, turn_index = grounded
            # 长期记忆正文只保存可追溯原话，模型 summary 仅作为抽取导航，
            # 不能把未被原话支持的改写变成长期事实。
            summary = f"用户提到：{quote}"
            source_context = ctx.turns[turn_index].user
            source_quote = _verbatim_source_quote(quote, source_context)
            preference_quote = item.get("preference_source_quote") if indexed_facts else None
            if not separate_fact_source(source_context, source_quote, preference_quote,
                                        preference_sources.get((ctx.session_id, turn_index), set())):
                continue
            ref = item.get("referenceable") is True
            if _must_not_reference(summary) or _must_not_reference(source_context):
                ref = False                    # 隐私兜底:敏感内容绝不主动引用
            while f"llm_{today}_{next_number}" in existing_ids:
                next_number += 1
            eps.append(EpisodicMemoryEntry(
                id=f"llm_{today}_{next_number}",
                timestamp=today,
                type="extracted",
                summary=summary,
                keywords=[str(k) for k in (item.get("keywords") or [])],
                referenceable=ref,
                source_session_id=ctx.session_id,
                source_turn_index=turn_index,
                source_quote=source_quote,
                source_context=source_context,
                source_context_sha256=source_context_sha256(source_context),
                source_context_version=REVIEWED_FACT_CONTEXT if preference_quote else SOURCE_CONTEXT_VERSION,
                preference_source_quote=preference_quote,
            ))
            next_number += 1
        relationships = self._relationships(
            relationship_rows, ctx, persona, strict_quotes=self.prefix_recovery,
            preference_sources=preference_sources, indexed_facts=indexed_facts,
        )
        ctx.working_memory["stated_preference_extraction"] = {
            "source": "model", "accepted": len(preferences), "rejected": rejected,
            "scope_selection": scope_selection,
            "source_protocol": source_protocol,
        }
        return ConsolidationResult(
            new_episodes=eps,
            relationships=relationships,
            stated_preferences=preferences,
            cognitive_observations=extract_cognitive_rows(
                cognitive_rows, persona, ctx),
        )

    @staticmethod
    def _relationships(
        rows,
        ctx: EngineContext,
        persona: PersonaProfile,
        *, strict_quotes: bool = False,
        preference_sources, indexed_facts: bool,
    ) -> list[RelationshipContextEntry]:
        now = datetime.now().isoformat(timespec="seconds")
        result: list[RelationshipContextEntry] = []
        for item in rows if isinstance(rows, list) else []:
            if not isinstance(item, dict):
                continue
            name = " ".join(str(item.get("name") or "").split())[:60]
            relation = " ".join(
                str(item.get("relation") or "").split()
            )[:40] or None
            raw_facts = item.get("facts") or []
            grounded_facts: list[RelationshipFactSource] = []
            for row in raw_facts if isinstance(raw_facts, list) else []:
                if not isinstance(row, dict):
                    continue
                grounded = (_strict_grounded_quote if strict_quotes else _grounded_quote)(
                    row.get("evidence_quote"), row.get("turn_index"), ctx
                )
                if grounded is not None:
                    quote, index = grounded
                    context = ctx.turns[index].user
                    source_quote = _verbatim_source_quote(quote, context)
                    preference_quote = row.get("preference_source_quote") if indexed_facts else None
                    if separate_fact_source(context, source_quote, preference_quote,
                                            preference_sources.get((ctx.session_id, index), set())):
                        grounded_facts.append(RelationshipFactSource(
                            quote=source_quote, turn_index=index, source_context=context,
                            source_context_sha256=source_context_sha256(context),
                            source_context_version=(row.get("source_context_version", SOURCE_CONTEXT_VERSION)
                                                    if indexed_facts else SOURCE_CONTEXT_VERSION),
                            preference_source_quote=preference_quote,
                        ))
            if not name or not grounded_facts:
                continue
            facts = list(dict.fromkeys(
                source.quote for source in grounded_facts
            ))[:8]
            combined = "".join(facts)
            if _normalise_text(name) not in _normalise_text(combined):
                continue
            if relation and _normalise_text(relation) not in _normalise_text(
                combined
            ):
                continue
            sensitive = _must_not_reference(
                f"{name}{relation or ''}{' '.join(facts)}"
            ) or any(
                _must_not_reference(source.source_context) for source in grounded_facts
            )
            indices = list(dict.fromkeys(
                source.turn_index for source in grounded_facts
            ))[:8]
            raw_id = f"{name.lower()}|{(relation or '').lower()}"
            relationship_id = (
                "rel_"
                + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
            )
            summary = f"{name}（{relation or '关系未说明'}）：{'；'.join(facts)}"
            result.append(RelationshipContextEntry(
                id=relationship_id,
                name=name,
                relation=relation,
                facts=list(dict.fromkeys(facts)),
                referenceable=(
                    item.get("referenceable") is True
                    and not sensitive
                ),
                consent_to_share=False,
                first_observed_at=now,
                last_confirmed_at=now,
                evidence=[RelationshipEvidence(
                    at=now,
                    session_id=ctx.session_id,
                    summary=summary,
                    turn_indices=list(dict.fromkeys(indices)),
                    source="user_conversation",
                    fact_sources=[source for source in grounded_facts if source.quote in facts],
                )],
            ))
        return result


class MemoryConsolidator:
    def __init__(self, store: Optional[PersonaStore] = None, extractor: Optional[Extractor] = None):
        self.store = store or PersonaStore()
        self.extractor = extractor or HeuristicExtractor()

    def consolidate(
        self, persona: PersonaProfile, ctx: EngineContext, *, persist: bool = True,
    ) -> PersonaProfile:
        """抽取本次会话 → 合并进 persona(去重)→ 落盘。返回更新后的 persona。

        这里只沉淀**事实**(episodic)。单次在线工作假设不能直接升级为长期
        个案结论；人物理解由 PersonaHypothesisUpdater 跨会话核验证据。有效
        假设在运行时参与策略，但不回写成不可撤销的静态人格标签。
        """
        result = self.extractor.extract(ctx, persona)
        if getattr(self.extractor, "last_status", "ok") in {"ok", "partial"}:
            consolidate_cognitive(persona, ctx, result.cognitive_observations)
        merge_preferences(persona, result.stated_preferences)
        existing = {e.id for e in persona.episodic_memory}
        for ep in result.new_episodes:
            if ep.id not in existing:
                persona.episodic_memory.append(ep)
                existing.add(ep.id)
        relationships = {
            item.id: item for item in persona.relationship_context
        }
        for incoming in result.relationships:
            current = relationships.get(incoming.id)
            if current is None:
                persona.relationship_context.append(incoming)
                relationships[incoming.id] = incoming
                continue
            current.facts = list(dict.fromkeys(
                [*current.facts, *incoming.facts]
            ))[-20:]
            # 一旦同一关系出现不可主动引用的敏感事实，整条关系退到只内化。
            current.referenceable = (
                current.referenceable and incoming.referenceable
            )
            current.last_confirmed_at = incoming.last_confirmed_at
            for evidence in incoming.evidence:
                duplicate = any(
                    row.session_id == evidence.session_id
                    and row.summary == evidence.summary
                    for row in current.evidence
                )
                if not duplicate:
                    current.evidence.append(evidence)
            current.evidence = current.evidence[-30:]

        # A durable offline queue commits the extracted facts and its stage
        # marker together. Its private transaction must not leak a half-written
        # profile through this legacy convenience save.
        if persist:
            self.store.save(persona)
        return persona
