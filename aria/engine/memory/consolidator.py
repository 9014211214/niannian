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
    StatedPreference,
)
from .persona_store import PersonaStore
from .stated_preferences import extract_preference_rows, heuristic_preferences, merge_preferences
from ..cognitive import cognitive_extraction_prompt, extract_cognitive_rows, consolidate_cognitive


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


# 敏感标记：即便模型判定可引用，也强制压成「内化不可主动提」。这是
# referenceability 地板，不代表不保存；任何向家属/外部分享仍需单独的当轮授权。
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
_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def _must_not_reference(text: str) -> bool:
    """Return true for locally enforceable private/no-reference contexts."""
    value = " ".join(str(text or "").strip().split())
    return (
        any(marker in value for marker in _SENSITIVE)
        or any(pattern.search(value) for pattern in _CONFIDENTIALITY_PATTERNS)
        or any(
            pattern.search(value)
            for pattern in _CONTACT_OR_CREDENTIAL_PATTERNS
        )
    )


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


def _parse_json_payload(raw: str):
    """Accept a JSON object/legacy array inside harmless prose or fences."""
    value = _strip_fences(str(raw or ""))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        starts = [
            index for index in (value.find("{"), value.find("["))
            if index >= 0
        ]
        if not starts:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(value[min(starts):])
        return parsed


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
    for name in ("episodes", "relationships", "stated_preferences", "cognitive_observations"):
        rows = data.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("cognitive memory response requires four object arrays")
    return data["cognitive_observations"]


class LLMExtractor:
    """会后事实抽取器，不诊断人格，也不决定如何回应。

    它与在线生成、人物反思使用独立客户端和触发时机。失败不影响主流程；
    隐私由 `_SENSITIVE` 和关系分享授权的确定性规则兜底。
    """

    def __init__(self, llm):
        self.llm = llm
        self.last_status = "not_run"
        self.last_error_type: str | None = None

    def extract(self, ctx: EngineContext, persona: PersonaProfile) -> ConsolidationResult:
        excluded = _sensitive_memory_excluded_turns(ctx)
        convo = "\n".join(
            f"[turn {index}] 用户:{turn.user}\n小予:{turn.agent}"
            for index, turn in enumerate(ctx.turns)
            if index not in excluded
        )
        cognitive_system, cognitive_user = cognitive_extraction_prompt(persona, ctx)
        if cognitive_user:
            convo += cognitive_user
        if not convo.strip():
            self.last_status = "skipped"
            self.last_error_type = None
            return ConsolidationResult()
        if persona.stated_preferences:
            previous = [
                {"id": row.id, "do": row.do, "scope": row.scope}
                for row in persona.stated_preferences if row.status == "confirmed"
            ][-4:]
            if previous:
                convo += (
                    "\n\n旧相处说明（只供supersedes引用，不是本次用户原话或新证据）：\n"
                    + json.dumps(previous, ensure_ascii=False)
                )
        messages = [
            {"role": "system", "content": (
                "你负责会后事实抽取，不做人格诊断。基于对话只抽取用户明确表达、"
                "值得长期记住的事实，不推测关系或心理。输出 JSON："
                '{"episodes":[{"summary":"","keywords":[],"referenceable":false,'
                '"evidence_quote":"逐字用户原话","turn_index":0}],'
                '"relationships":[{"name":"","relation":"","facts":['
                '{"value":"","evidence_quote":"逐字用户原话","turn_index":0}],'
                '"referenceable":false}],"stated_preferences":['
                '{"do":"本人明确希望的做法，60字内，优先直接引用原话",'
                '"scope":"本人说明的适用场景","quotes":[{"turn_index":0,"text":"逐字用户原话"}],'
                '"confirmation":{"turn_index":1,"text":"本人后续完整肯定答复"},'
                '"supersedes":null}]'
                + (
                    ',"cognitive_observations":[{"domain":"memory|language|orientation|daily_function|repetition",'
                    '"evidence_quote":"逐字本人原话","turn_index":0,"functional_impact":false,'
                    '"event_kind":"new_event|same_event","prior_event_id":null}]}。'
                    if cognitive_system else '}。'
                ) +
                "关系只记录用户明确提到的真实人物；不知道姓名就不抽取。"
                "每条事实必须给出同一轮用户原话中的逐字 evidence_quote；不能"
                "仅给摘要或模型推断。turn_index 必须沿用输入中的原始编号；系统会"
                "丢弃无法与该轮原话匹配或已被隐私层排除的条目。"
                "私密/敏感（明确要求保密、逝者、家庭隐情、自伤、债务/转账/"
                "存款/账户、证件或联系方式）一律 referenceable=false。"
                "不要输出 consent_to_share，系统永远不会因模型抽取自动授权分享。"
                "相处说明只记录本人明确要求怎样交流，不是人格标签、因果推测或分享通知权限。"
                "confirmation只有在助理先复述同一做法并问是否这样记下、本人随后完整肯定时才给出，"
                "否则为null。不能用助理原话或自己的总结充当本人证据；不得扩大原话范围。"
                "没有符合条件的相处说明就给空数组。"
                "只输出 JSON，不要解释。"
            )},
            {"role": "user", "content": convo},
        ]
        if cognitive_system:
            messages[0]["content"] += cognitive_system
        self.last_error_type = None
        try:
            raw = self.llm.complete(messages)
        except Exception:
            self.last_status = "failed"
            self.last_error_type = "model_call_failed"
            return ConsolidationResult()
        try:
            data = _parse_json_payload(raw)
            episode_rows, relationship_rows = _memory_protocol_rows(data)
            cognitive_rows = _cognitive_protocol_rows(data) if cognitive_system else []
        except (TypeError, ValueError, json.JSONDecodeError):
            self.last_status = "failed"
            self.last_error_type = "invalid_json_response"
            return ConsolidationResult()
        self.last_status = "ok"
        self.last_error_type = None

        eps: list[EpisodicMemoryEntry] = []
        base = len(persona.episodic_memory)
        for i, item in enumerate(
            episode_rows if isinstance(episode_rows, list) else []
        ):
            if not isinstance(item, dict):
                continue
            grounded = _grounded_quote(
                item.get("evidence_quote"), item.get("turn_index"), ctx
            )
            if grounded is None:
                continue
            quote, turn_index = grounded
            # 长期记忆正文只保存可追溯原话，模型 summary 仅作为抽取导航，
            # 不能把未被原话支持的改写变成长期事实。
            summary = f"用户提到：{quote}"
            ref = bool(item.get("referenceable", False))
            if _must_not_reference(summary):
                ref = False                    # 隐私兜底:敏感内容绝不主动引用
            eps.append(EpisodicMemoryEntry(
                id=f"llm_{date.today().isoformat()}_{base + i + 1}",
                timestamp=date.today().isoformat(),
                type="extracted",
                summary=summary,
                keywords=[str(k) for k in (item.get("keywords") or [])],
                referenceable=ref,
                source_session_id=ctx.session_id,
                source_turn_index=turn_index,
                source_quote=_verbatim_source_quote(quote, ctx.turns[turn_index].user),
            ))
        relationships = self._relationships(
            relationship_rows, ctx, persona
        )
        preferences, rejected = extract_preference_rows(
            data.get("stated_preferences", []) if isinstance(data, dict) else [], ctx, persona,
        )
        ctx.working_memory["stated_preference_extraction"] = {
            "source": "model", "accepted": len(preferences), "rejected": rejected,
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
            grounded_facts: list[tuple[str, int]] = []
            for row in raw_facts if isinstance(raw_facts, list) else []:
                if not isinstance(row, dict):
                    continue
                grounded = _grounded_quote(
                    row.get("evidence_quote"), row.get("turn_index"), ctx
                )
                if grounded is not None:
                    grounded_facts.append(grounded)
            if not name or not grounded_facts:
                continue
            facts = list(dict.fromkeys(
                quote for quote, _index in grounded_facts
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
            )
            indices = list(dict.fromkeys(
                index for _quote, index in grounded_facts
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
                    bool(item.get("referenceable", False))
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
        if getattr(self.extractor, "last_status", "ok") == "ok":
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
