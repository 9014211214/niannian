"""养老场景的状态推理 —— 关注「生活是否在缩小」，而不是给情绪贴标签。

这是一个可审计的混合推理层:
- 文本、声学、视觉、行为、环境信号都只是证据；
- LLM 只提出带原文锚点的语义候选，规则保留为显式表达和安全降级；
- 假设彼此竞争，低置信时优先问一个区分问题；
- 人格与保护因素改变干预选择，而不只改变措辞；
- 下一轮回应形成结果，跨会话积累后才固化成个人策略。

它不做疾病诊断。涉及身体或持续风险时，只选择澄清/升级路径。
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import re
from typing import Iterable, Optional

from ...engine.context import EngineContext
from ...engine.model.persona import (
    InterventionOutcomeRecord,
    PersonaProfile,
)
from ...engine.policy.prompt import select_delivery_mode
from ...engine.memory.hypothesis_context import online_influencing_hypotheses
from ...interfaces.reasoning import (
    DeliveredIntervention,
    InterventionCandidate,
    InterventionOutcome,
    ObservationExecution,
    ObservationEvidence,
    WorkingHypothesis,
)
from .observation import ElderActiveObservationPolicy
from .risk import current_self_physical_hit

_WITHDRAWAL_WORDS = (
    "不想出门", "懒得出门", "不出门", "不去了", "没意思", "都一样",
    "干坐着", "提不起劲", "不想见", "不想聊", "算了",
)
_ROLE_LOSS_WORDS = (
    "没用了", "没什么用", "帮不上", "用不着我", "不懂了", "跟不上",
    "老了", "累赘", "添麻烦", "碍事",
)
_SOCIAL_WORDS = (
    "一个人", "没人", "不见人", "不想见", "不去棋", "不去活动",
    "都走了", "也不联系",
)
_PHYSICAL_WORDS = (
    "头晕", "胸闷", "喘不上", "疼", "不舒服", "没力气", "站不稳",
    "吃不下", "睡不着",
)
_AUTONOMY_WORDS = (
    "替我决定", "别替我", "我自己来", "不用安排", "管得太多",
    "催来催去", "非让我", "逼着我",
)
_GRIEF_WORDS = (
    "去世", "不在了", "走了以后", "想他", "想她", "舍不得",
    "空落落", "老伴", "送走",
)
_DIRECT_DISCLOSURE_WORDS = (
    "其实", "就是", "因为", "我有点", "我不想", "我怕", "我难受",
    "帮不上", "没用了", "头晕", "不舒服", "一个人",
)
_REJECTION_WORDS = ("别问了", "烦不烦", "不想说", "你别管", "不用你管")
_FEEDBACK_REFERENCE_MARKERS = (
    "电视里", "电视剧", "电影里", "新闻里", "小说里", "视频里",
    "短视频", "网上说", "别人说", "有人说", "他说", "她说", "问我",
    "他对我说", "她对我说", "他跟我说", "她跟我说",
    "他告诉我", "她告诉我", "对我说", "跟我说",
)
_FEEDBACK_THIRD_PERSON = (
    "我妈", "我爸", "我老伴", "我爱人", "我女儿", "我儿子",
    "我孙子", "我孙女", "我朋友", "我邻居", "女儿", "儿子",
    "老伴", "朋友", "邻居",
)
_FEEDBACK_HYPOTHETICAL = ("如果", "假如", "假设", "会不会", "比如")
_THIRD_PERSON_REPORT_RE = re.compile(
    r"(?:别人|有人|对方|他|她|他们|她们|我妈|我爸|我老伴|我爱人|"
    r"我女儿|我儿子|我孙子|我孙女|我朋友|我邻居|女儿|儿子|老伴|"
    r"朋友|邻居).{0,12}(?:对我说|跟我说|告诉我|问我|劝我|让我|说)"
)
_THIRD_PERSON_CONTINUATION_RE = re.compile(
    r"^(?:他|她|他们|她们|对方)(?:又|还|当时|刚才|后来|就|也)?"
)
_DIRECT_SELF_APPROVAL_SPLIT_RE = re.compile(
    r"(?=(?:但|不过|可是|然而)?我(?:倒|就|真)?觉得(?:你|小予|aria))",
    re.IGNORECASE,
)
_FEEDBACK_APPROVAL = (
    re.compile(
        r"(?:你|小予|aria).{0,10}(?:说|问|陪|做)(?:得|的)?.{0,24}"
        r"(?:挺好|很好|不错|很对|有用|合适|舒服)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:谢谢你|多亏你).{0,12}(?:刚才|上次|这样|那么)?",
        re.IGNORECASE,
    ),
)
_WEAK_OUTCOME_SIGNALS = {
    "new_disclosure",
    "continued_engagement",
    "brief_response",
}
# 结果学习只调整可替代的心理社会支持路线。身体澄清与安全升级由当前证据
# 决定，不能因为老人过去不喜欢被问身体状况就被“学没了”；安静陪伴是安全
# 底座，也不作为需要强化或惩罚的技巧。
_OUTCOME_ADAPTIVE_INTERVENTIONS = {
    "low_pressure_clarification",
    "identity_based_reengagement",
    "human_reconnection",
    "autonomy_restoration",
    "meaning_oriented_companionship",
}
_ROUTE_PERSON_DIMENSIONS = {
    "low_pressure_clarification": {"low_pressure", "autonomy"},
    "quiet_companionship": {"low_pressure"},
    "identity_based_reengagement": {"identity_action", "direct_options"},
    "human_reconnection": {"human_connection"},
    "autonomy_restoration": {"autonomy", "direct_options"},
    "meaning_oriented_companionship": {"meaning_memory"},
}
_ROUTE_PERSON_MARKERS = {
    "low_pressure_clarification": (
        "催促", "控制", "决定权", "留白", "空间", "连续追问", "自己决定",
    ),
    "quiet_companionship": ("留白", "空间", "安静", "不追问"),
    "identity_based_reengagement": (
        "请教", "经验", "被需要", "具体事情", "动手", "角色",
    ),
    "human_reconnection": ("熟人", "一起", "陪同", "真实关系", "先联系"),
    "autonomy_restoration": ("控制", "决定权", "自己决定", "选择"),
    "meaning_oriented_companionship": ("回忆", "纪念", "慢慢讲", "离别"),
}
_CURRENT_REOPEN_INTENTS = (
    "我想", "我要", "我愿意", "我可以", "我打算", "我准备",
    "帮我", "替我", "要不我", "让我",
)
_CURRENT_REOPEN_TERMS = {
    "low_pressure_clarification": ("问", "说", "聊"),
    "identity_based_reengagement": ("帮", "看看", "检查", "修", "参加", "做"),
    # 裸“聊/见/约”必须再通过下方的人际对象资格，不能只靠动词出现。
    "human_reconnection": (
        "联系", "问候", "发消息", "打电话", "见面", "见", "约", "聊",
    ),
    "autonomy_restoration": ("自己决定", "自己选", "自己安排"),
    "meaning_oriented_companionship": ("说说", "聊聊", "讲讲", "回忆", "记下来"),
}
_CURRENT_REOPEN_NEGATIONS = (
    "不想", "不要", "不愿意", "不打算", "别", "先不", "暂时不", "不用",
)
_CURRENT_REOPEN_NON_CURRENT = (
    "如果", "假如", "要是", "以前", "之前", "上次", "当时", "曾经", "本来",
    "昨天", "前天", "那天", "等我", "到时候", "的时候",
)
_CURRENT_REOPEN_REPORTED_INTRO_RE = re.compile(
    r"(?:女儿|儿子|老伴|家人|医生|护士|护理员|朋友|邻居|他|她|他们|她们)"
    r".{0,10}(?:说|觉得|以为|问|告诉|让我|叫我)"
)
_CURRENT_SELF_RESET = ("我现在", "现在我", "这次我", "我这次", "今天我")
_CURRENT_REOPEN_EMBEDDED_NON_ACTION = (
    "知道", "确认", "弄清", "有没有", "是否", "会不会",
)
_CURRENT_REOPEN_TRAILING_NEGATION_RE = re.compile(
    r"(?:不(?:再|要|想|愿|打算|必)?|别(?:再)?|勿|莫)\s*$"
)
_CURRENT_REOPEN_MODAL_QUESTION_RE = re.compile(r"(?:吗|呢|是不是)\s*$")
_CURRENT_REOPEN_PERMISSION_QUESTION_RE = re.compile(
    r"(?:可以吗|行吗|好不好)\s*$"
)
_CURRENT_REOPEN_SUBJECT_DENIAL_RE = re.compile(
    r"(?:不是|并非|谁说)我(?:现在|这次|今天)?(?:想|要|愿意|打算|准备)$"
)
_CURRENT_REOPEN_CANCEL_RE = re.compile(
    r"算了|(?:还是)?改天(?:吧)?|下次再说|以后再说|暂时不|只是问问|"
    r"念头.{0,6}(?:没了|打消)|不打算了"
)
_HUMAN_REOPEN_TARGET = re.compile(
    r"(?:女儿|儿子|老伴|家人|亲戚|朋友|邻居|同事|同学|护理员|"
    r"医生|护士|社工|老[\u3400-\u9fff]{1,2}|小[\u3400-\u9fff]{1,2}|"
    r"[\u3400-\u9fff]{1,3}(?:阿姨|叔叔|师傅|老师|大夫))"
)
_QUESTION_SENTENCE = re.compile(r"[^。！!？?]*[？?]")
_OBSERVATION_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
_SEMANTIC_DELIVERY_VERSIONS = {
    "online_turn_decision_v3", "online_turn_decision_v4_evidence",
    "online_turn_decision_v5_bound", "online_turn_decision_v6_semantic",
    "online_turn_decision_v7",
}

_BEHAVIOR_WITHDRAWAL_SIGNALS = {
    "initiative_frequency", "conversation_duration", "social_participation",
    "out_of_home", "contact_frequency", "activity_level", "mobility",
}
_PROSODY_WITHDRAWAL_SIGNALS = {"speech_pace", "volume"}
_PROSODY_INCREASED_STRAIN_SIGNALS = {"pause_ratio", "voice_break_ratio"}
_VISUAL_WITHDRAWAL_SIGNALS = {
    "posture_engagement", "eye_contact", "facial_activation", "movement_energy",
}
_VISUAL_TENSION_SIGNALS = {
    "gaze_avoidance", "lip_tension", "brow_tension", "head_down",
}

# 这些模式不是新的“情绪词典”，而是对关键词降级规则的否决条件：同一句话
# 已明确给出仍在参与、仍有真人连接或直接否定某种解释时，不能截取半句话贴标签。
_CURRENT_CONTEXT_COUNTERS = {
    "reduced_life_engagement": (
        re.compile(
            r"(?:还要|还会|准备|打算|待会|等会|稍后|下午|晚上|明天|照常|已经)"
            r".{0,12}(?:出门|去|参加|活动|下棋|散步|锻炼|买菜|见)"
        ),
        re.compile(r"(?:不是|并非).{0,8}(?:不想|不愿|没兴趣|提不起劲)"),
    ),
    "social_disconnection": (
        re.compile(
            r"(?:和|跟).{1,10}(?:一起|下棋|聊天|散步|见面|活动|吃饭)"
        ),
        re.compile(
            r"(?:每天|常常|经常|这周|今天).{0,10}(?:来找|来看|联系|一起)"
        ),
        re.compile(r"(?:不是|并非).{0,8}(?:没人|不联系|不见人)"),
    ),
    "loss_of_role": (
        re.compile(
            r"(?:不是|并非).{0,8}(?:觉得)?(?:自己)?"
            r"(?:没用|没什么用|帮不上|累赘)"
        ),
        # “杯子放着也不碍事”中的“碍事”描述物件，不是在表达自己是负担。
        re.compile(r"(?:也|并|又)?不(?:碍事|添麻烦)"),
    ),
    "concealed_distress": (
        re.compile(r"(?:不是|并非).{0,8}(?:瞒|藏着|不想说)"),
        re.compile(r"(?:事情|问题).{0,8}(?:已经|早就).{0,5}(?:处理好|解决)"),
    ),
}


def _explicit_response_feedback(text: str) -> str | None:
    """Return only high-precision feedback about Aria's previous response.

    Merely repeating a refusal from a relative, a programme or a hypothetical
    example must not become a durable negative outcome. Ambiguous text is left
    to the weak engagement signals below and therefore cannot create cooldown.
    """
    clauses = _feedback_clauses(text)
    for clause, third_party_context in reversed(clauses):
        lowered = clause.lower()
        if (
            third_party_context
            or any(marker in clause for marker in _FEEDBACK_REFERENCE_MARKERS)
            or any(marker in clause for marker in _FEEDBACK_THIRD_PERSON)
            or any(marker in clause for marker in _FEEDBACK_HYPOTHETICAL)
        ):
            continue
        if any(pattern.search(lowered) for pattern in _FEEDBACK_APPROVAL):
            return "positive"

        for phrase in _REJECTION_WORDS:
            at = clause.find(phrase)
            if at < 0:
                continue
            before = clause[:at]
            after = clause[at + len(phrase):]
            quoted = (
                before.count("“") > before.count("”") and "”" in after
            ) or (
                before.count('"') % 2 == 1 and '"' in after
            ) or (
                before.count("'") % 2 == 1 and "'" in after
            )
            if quoted:
                continue
            prefix = clause[max(0, at - 12):at]
            if any(marker in prefix for marker in (
                "不是", "并非", "没有", "没说", "没让", "不代表",
            )):
                continue
            direct_address = (
                "你" in clause[:at + len(phrase)]
                or "小予" in clause[:at + len(phrase)]
                or "aria" in lowered[:at + len(phrase)]
                or clause.startswith(("别问了", "烦不烦"))
            )
            # “我不想说” alone describes the user's state; it is not a
            # verified evaluation of Aria's intervention.
            if phrase == "不想说" and "你" not in clause:
                direct_address = False
            if direct_address:
                return "negative"
    return None


def _feedback_clauses(text: str) -> list[tuple[str, bool]]:
    """Split feedback events and retain a narrowly scoped reported speaker.

    ASR often drops punctuation, while written dialogue may introduce a person
    in one sentence and continue with ``她对我说…`` in the next. A later,
    explicit first-person evaluation of Aria starts a new event so a quoted
    rejection cannot hide genuine feedback.
    """
    pieces: list[str] = []
    for coarse in re.split(r"[。！？；;，,\n]", text):
        pieces.extend(_DIRECT_SELF_APPROVAL_SPLIT_RE.split(coarse))
    clauses = [
        item.strip(" \t\r\n，,。！？!?；;“”\"'")
        for item in pieces
        if item.strip(" \t\r\n，,。！？!?；;“”\"'")
    ]
    result: list[tuple[str, bool]] = []
    prior_reported_subject = False
    for clause in clauses:
        lowered = clause.lower()
        direct_self_approval = any(
            pattern.search(lowered) for pattern in _FEEDBACK_APPROVAL
        )
        explicit_third = any(
            marker in clause for marker in _FEEDBACK_THIRD_PERSON
        )
        reported = (
            any(marker in clause for marker in _FEEDBACK_REFERENCE_MARKERS)
            or _THIRD_PERSON_REPORT_RE.search(clause) is not None
            or (
                prior_reported_subject
                and _THIRD_PERSON_CONTINUATION_RE.search(clause) is not None
            )
        )
        third_party_context = (
            (reported or explicit_third) and not direct_self_approval
        )
        result.append((clause, third_party_context))
        prior_reported_subject = (
            False
            if direct_self_approval
            else bool(reported or explicit_third)
        )
    return result


class ElderStateReasoner:
    """把养老场景的观测转成竞争假设与本轮干预方向。"""

    def __init__(self) -> None:
        self._observation = ElderActiveObservationPolicy()

    def observe_response(self, ctx: EngineContext, utterance: str) -> None:
        """把当前用户话语视为上一轮干预的后续结果。"""
        delivered = self._previous_delivery(ctx)
        if delivered is None or self._has_outcome(ctx, delivered):
            return
        compact = utterance.strip()
        explicit_feedback = _explicit_response_feedback(compact)
        if explicit_feedback == "negative":
            value, signal, note = (
                "negative",
                "explicit_response_rejection",
                "用户明确拒绝 Aria 上一次回应方式",
            )
        elif explicit_feedback == "positive":
            value, signal, note = (
                "positive",
                "explicit_response_approval",
                "用户明确肯定 Aria 上一次回应方式",
            )
        elif any(word in compact for word in _DIRECT_DISCLOSURE_WORDS):
            value, signal, note = "positive", "new_disclosure", "用户补充了可区分原因的新信息"
        elif len(compact) >= 12:
            value, signal, note = "positive", "continued_engagement", "用户愿意继续展开"
        else:
            value, signal, note = "neutral", "brief_response", "后续回应较短，暂不足以下结论"
        self._append_outcome(ctx, delivered, value, signal, note)

    def record_observation_execution(
        self, ctx: EngineContext, response: str
    ) -> ObservationExecution:
        """只根据最终交付文本记录动作，计划本身绝不冒充已经执行。"""
        plan = ctx.reasoning.observation_plan
        planned_mode = plan.mode if plan is not None else "none"
        planned_question = plan.question.strip() if plan is not None else ""
        online_trace = ctx.working_memory.get("online_turn_decision") or {}
        semantic_delivery = ctx.working_memory.get("_online_decision_version") in _SEMANTIC_DELIVERY_VERSIONS
        delivery_guard = online_trace.get("delivery_guard") or {}
        task_communication = delivery_guard.get("task_communication") or {}
        task_question_executed = bool(
            task_communication.get("included")
            and task_communication.get("mode") == "ask"
        )
        # A controlled task confirmation may coexist with a relationship wait.
        # It is not an emotional probe and must not make the observation ledger
        # claim that the rejected psychological question was asked after all.
        executed_question = (
            ""
            if planned_mode == "wait" and task_question_executed and not semantic_delivery
            else _first_question(response)
        )
        if executed_question:
            executed_mode = "ask"
        elif planned_mode == "wait":
            executed_mode = "wait"
        else:
            executed_mode = "observe"
        if planned_mode == "ask":
            matched_plan = bool(
                planned_question
                and _normalise_observation_text(planned_question)
                in _normalise_observation_text(executed_question)
            )
        else:
            matched_plan = planned_mode == executed_mode
        execution = ObservationExecution(
            turn_index=len(ctx.turns),
            planned_mode=planned_mode,
            executed_mode=executed_mode,
            planned_question=planned_question,
            executed_question=executed_question,
            matched_plan=matched_plan,
            hypothesis_ids=(
                list(plan.hypothesis_ids) if plan is not None else []
            ),
        )
        ctx.reasoning.observation_history.append(execution)
        del ctx.reasoning.observation_history[:-40]
        view = {
            "turn_index": execution.turn_index,
            "planned_mode": execution.planned_mode,
            "executed_mode": execution.executed_mode,
            "planned_question": execution.planned_question,
            "executed_question": execution.executed_question,
            "matched_plan": execution.matched_plan,
            "hypothesis_ids": list(execution.hypothesis_ids),
            "task_question_executed": task_question_executed,
        }
        ctx.working_memory["observation_execution"] = view
        trace = ctx.working_memory.get("online_turn_decision")
        if isinstance(trace, dict):
            trace["observation_execution"] = dict(view)
        return execution

    def record_intervention_delivery(
        self,
        ctx: EngineContext,
        response: str,
        *,
        response_overridden: bool = False,
    ) -> Optional[DeliveredIntervention]:
        """最终文本落定后提交交付；selected 只表示计划，不能用于结果学习。"""
        selected = ctx.reasoning.selected
        reason = ""
        reply_route = (
            (ctx.working_memory.get("model_route") or {}).get("reply") or {}
        )
        guard = (
            ctx.working_memory.get("online_turn_decision") or {}
        ).get("delivery_guard") or {}
        online_trace = ctx.working_memory.get("online_turn_decision") or {}
        online_version = ctx.working_memory.get("_online_decision_version")
        execution = (
            ctx.reasoning.observation_history[-1]
            if ctx.reasoning.observation_history
            and ctx.reasoning.observation_history[-1].turn_index
            == len(ctx.turns)
            else None
        )
        if selected is None:
            reason = "no_selected_intervention"
        elif not str(response).strip():
            reason = "empty_final_response"
        elif response_overridden:
            reason = "final_response_overridden"
        elif (
            reply_route.get("tier") == "deterministic_fallback"
            or reply_route.get("source") == "deterministic_fallback"
        ):
            reason = "model_fallback_response"
        elif (
            reply_route.get("integrated_decision") is True
            and online_trace.get("status") != "parsed"
        ):
            reason = "integrated_decision_not_parsed"
        elif bool(guard.get("activated")):
            reason = "delivery_guard_replaced_response"
        elif "requested_draft_not_delivered" in guard.get("task_delivery_diagnostics", []):
            reason = "requested_draft_not_delivered"
        elif online_version in _SEMANTIC_DELIVERY_VERSIONS and (
            online_trace.get("version") != online_version
            or online_trace.get("status") != "parsed"
            or online_trace.get("turn_index") != len(ctx.turns)
            or online_trace.get("requested_intervention_id") != selected.id
            or online_trace.get("selected_intervention_id") != selected.id
            or selected.selected_turn != len(ctx.turns)
            or selected.id not in {item.id for item in ctx.reasoning.candidates}
            or online_trace.get("parsed_response_sha256") != hashlib.sha256(response.encode("utf-8")).hexdigest()
        ):
            reason = "unbound_final_strategy_delivery"
        elif (
            online_version not in _SEMANTIC_DELIVERY_VERSIONS
            and
            ctx.reasoning.observation_plan is not None
            and ctx.reasoning.observation_plan.mode == "ask"
            and (execution is None or not execution.matched_plan)
        ):
            reason = "planned_probe_not_delivered"

        if reason:
            view = {
                "status": "not_delivered",
                "turn_index": len(ctx.turns),
                "intervention_id": (
                    selected.id if selected is not None else None
                ),
                "reason": reason,
            }
            ctx.working_memory["intervention_delivery"] = view
            trace = ctx.working_memory.get("online_turn_decision")
            if isinstance(trace, dict):
                trace["intervention_delivery"] = dict(view)
            return None

        delivered = DeliveredIntervention(
            id=selected.id,
            label=selected.label,
            goal=selected.goal,
            turn_index=len(ctx.turns),
            hypothesis_ids=list(selected.hypothesis_ids),
            source=str(reply_route.get("source") or "final_response"),
        )
        ctx.reasoning.delivered_interventions.append(delivered)
        del ctx.reasoning.delivered_interventions[:-40]
        view = {
            "status": "delivered",
            "turn_index": delivered.turn_index,
            "intervention_id": delivered.id,
            "source": delivered.source,
        }
        ctx.working_memory["intervention_delivery"] = view
        trace = ctx.working_memory.get("online_turn_decision")
        if isinstance(trace, dict):
            trace["intervention_delivery"] = dict(view)
        return delivered

    def update(
        self,
        ctx: EngineContext,
        persona: Optional[PersonaProfile],
        utterance: str,
        evidence: list[ObservationEvidence],
    ) -> None:
        state = ctx.reasoning
        lexical_fallback = ctx.working_memory.get("_online_decision_version") != "online_turn_decision_v7"
        # Historical outcome records remain available, but a phrase matcher
        # cannot declare that the current owner has or has not reopened a route.
        decision_utterance = utterance if lexical_fallback else ""
        state.evidence.extend(evidence)
        state.evidence = state.evidence[-60:]              # 会话证据有界，长期趋势由上游聚合
        state.hypotheses = self._infer_hypotheses(
            state.evidence, persona, utterance,
            lexical_fallback=lexical_fallback,
        )
        # 先让当前证据、人物假设与已验证结果完成一次有界排序，再决定这一拍
        # 是否还值得追问。否则通用观察题会先锁死 prompt，覆盖已经形成的
        # 人物化路线或“连续失败后先停”的结果学习。
        state.candidates = self._build_candidates(
            state.hypotheses, persona, decision_utterance, lexical_fallback=lexical_fallback,
        )
        state.selected = max(state.candidates, key=lambda c: c.score, default=None)
        selected_id = state.selected.id if state.selected is not None else ""
        current_physical_source = _qualified_current_physical_evidence(
            utterance, evidence, len(ctx.turns)
        )
        person_supported = _person_supports_route(
            persona, state.candidates, selected_id, lexical_fallback=lexical_fallback,
        )
        cooldown_routes = _relevant_verified_cooldowns(
            persona, state.candidates, decision_utterance
        )
        current_user_reopened = lexical_fallback and any(
            _current_explicitly_reopens(utterance, intervention_id)
            for intervention_id in _OUTCOME_ADAPTIVE_INTERVENTIONS
        )
        # 相关冷却来自跨会话真实结果，而不是依赖某个权重是否刚好把 quiet
        # 排到第一。只有可证明的本人当前身体证据可以让身体澄清先行。
        wait_for_cooldown = (
            tuple(cooldown_routes)
            if cooldown_routes and not current_physical_source else ()
        )
        state.observation_plan = self._observation.plan(
            ctx,
            utterance,
            state.hypotheses,
            selected_intervention_id=selected_id,
            person_supported_route=person_supported,
            verified_cooldown_routes=wait_for_cooldown,
            current_user_reopened=current_user_reopened,
            current_physical_evidence_qualified=bool(current_physical_source),
        )
        if (
            state.observation_plan.mode == "wait"
            and state.candidates
        ):
            state.selected = next(
                (
                    item for item in state.candidates
                    if item.id == "quiet_companionship"
                ),
                state.selected,
            )
        if state.selected is not None:
            state.selected.selected_turn = len(ctx.turns)
        ctx.working_memory["personality_decision"] = (
            self._personality_decision_basis(
                persona,
                state.selected,
                state.candidates,
                decision_utterance,
                current_physical_source=current_physical_source,
            )
        )

    def record_outcome(
        self, ctx: EngineContext, value: str, *, signal: str, note: str = ""
    ) -> None:
        """供真实产品显式回写结果，例如完成通话、参加活动或拒绝建议。"""
        delivered = (
            ctx.reasoning.delivered_interventions[-1]
            if ctx.reasoning.delivered_interventions else None
        )
        if delivered is None:
            return
        normalised = value if value in ("positive", "neutral", "negative") else "neutral"
        self._append_outcome(
            ctx, delivered, normalised, signal, note[:120]
        )

    def finalize(self, ctx: EngineContext, persona: Optional[PersonaProfile]) -> None:
        """结果落入长期档案；同一策略多次有效后才沉淀为个人 tactic。"""
        if persona is None:
            return
        now = datetime.now().isoformat(timespec="seconds")
        existing = {
            (r.session_id, r.session_turn, r.intervention_id, r.signal)
            for r in persona.intervention_history
        }
        for outcome in ctx.reasoning.outcomes:
            key = (
                ctx.session_id,
                outcome.turn_index,
                outcome.intervention_id,
                outcome.signal,
            )
            if key in existing:
                continue
            persona.intervention_history.append(InterventionOutcomeRecord(
                at=now,
                session_id=ctx.session_id,
                session_turn=outcome.turn_index,
                intervention_id=outcome.intervention_id,
                intervention_label=outcome.intervention_label,
                goal=outcome.goal,
                outcome=outcome.value,
                signal=outcome.signal,
                note=outcome.note or None,
            ))
            existing.add(key)
        persona.intervention_history = persona.intervention_history[-100:]

        intervention_ids = {
            record.intervention_id
            for record in persona.intervention_history
            if record.signal not in _WEAK_OUTCOME_SIGNALS
        }
        for intervention_id in intervention_ids:
            prefix = f"经多次结果验证有效[{intervention_id}]:"
            # 已验证策略也必须可推翻：同一策略后来连续失败时，旧提示不能永久残留。
            persona.aria_strategy_hints.tactics = [
                item for item in persona.aria_strategy_hints.tactics
                if not item.startswith(prefix)
            ]
            records = _verified_outcome_records(persona, intervention_id)
            summary = _verified_outcome_summary(persona, intervention_id)
            # tactic 也是可逆的：只认最近两个独立会话的一致成功。一次新的
            # 明确成功会先解除旧冷却，但不会立刻把策略重新固化为稳定偏好。
            if (
                summary is None
                or summary["adaptation_state"] != "reinforced"
            ):
                continue
            record = list(records.values())[-1]
            persona.aria_strategy_hints.tactics.append(
                f"{prefix}{record.goal}"
            )

    def _infer_hypotheses(
        self,
        evidence: list[ObservationEvidence],
        persona: Optional[PersonaProfile],
        utterance: str,
        *,
        lexical_fallback: bool = True,
    ) -> list[WorkingHypothesis]:
        hypotheses: list[WorkingHypothesis] = []
        recent = evidence[-30:]

        def text_score(words, score, hypothesis_id):
            # Current online decisions already supply grounded semantic
            # candidates. Everyday vocabulary is not a second classifier.
            # Keep the archived fallback only for explicit older protocols.
            if not lexical_fallback:
                return _semantic_score(recent, hypothesis_id)
            return _text_semantic_score(utterance, words, score, recent, hypothesis_id)
        active_cbt_claims = (
            [
                item.claim
                for item in online_influencing_hypotheses(persona)
                if item.framework == "cbt"
            ]
            if persona is not None else []
        )

        semantic_hidden, semantic_hidden_ids, semantic_hidden_counter = (
            _semantic_score(recent, "concealed_distress")
        )
        if lexical_fallback and _has_current_context_counter(utterance, "concealed_distress"):
            semantic_hidden = 0.0
            semantic_hidden_counter.extend(
                _current_text_ids(utterance, recent)
            )
        hidden_score = semantic_hidden
        hidden_ids = list(semantic_hidden_ids)
        for item in recent:
            if item.signal == "divergence":
                # 纯文本模型的“字面/语义有差异”不等于“在隐藏困扰”。在线模型
                # 若确有这一候选，必须显式输出 concealed_distress 及原话锚点；
                # 这里只让真正观察到的声学×文本分歧提供独立支持，避免把方言、
                # 自嘲、自主受损或普通含蓄表达泛化成“伪装坚强”。
                if item.source in {
                    "text_llm_hypothesis",
                    "online_single_pass_decision",
                }:
                    continue
                candidate = 0.45 * item.weight
                hidden_score = max(hidden_score, candidate)
                hidden_ids.append(item.id)
            elif item.modality == "visual" and item.signal in _VISUAL_TENSION_SIGNALS:
                if _positive_signal(item):
                    hidden_score += 0.12 * item.weight
                    hidden_ids.append(item.id)
        if lexical_fallback and persona and any(
            "压抑" in p or "报喜不报忧" in p
            for p in [
                *persona.case_conceptualization.core_patterns,
                *active_cbt_claims,
            ]
        ):
            hidden_score += 0.08
        self._maybe_add(
            hypotheses, "concealed_distress", "可能在克制或隐藏困扰", "emotional_need",
            hidden_score, hidden_ids,
            "不要揭穿；用一个具体事实问题区分是身体、事情还是关系上的难处。",
            counter_ids=semantic_hidden_counter,
        )

        withdrawal_score, withdrawal_ids, withdrawal_counter = (
            text_score(
                _WITHDRAWAL_WORDS, 0.34,
                "reduced_life_engagement",
            )
        )
        for item in recent:
            if item.modality == "behavioral" and item.signal in _BEHAVIOR_WITHDRAWAL_SIGNALS:
                strength = _negative_delta_strength(item)
                if strength > 0:
                    withdrawal_score += 0.36 * strength * item.weight
                    withdrawal_ids.append(item.id)
            elif item.modality == "prosodic" and item.signal in _PROSODY_WITHDRAWAL_SIGNALS:
                strength = _negative_delta_strength(item)
                if strength > 0:
                    withdrawal_score += 0.13 * strength * item.weight
                    withdrawal_ids.append(item.id)
            elif item.modality == "prosodic" and item.signal in _PROSODY_INCREASED_STRAIN_SIGNALS:
                strength = _positive_delta_strength(item)
                if strength > 0:
                    withdrawal_score += 0.10 * strength * item.weight
                    withdrawal_ids.append(item.id)
            elif item.modality == "visual" and item.signal in _VISUAL_WITHDRAWAL_SIGNALS:
                strength = _negative_delta_strength(item)
                if strength > 0:
                    withdrawal_score += 0.10 * strength * item.weight
                    withdrawal_ids.append(item.id)
        counter_ids = [
            item.id for item in recent
            if item.signal == "topic_activation" and _positive_signal(item)
        ]
        counter_ids.extend(withdrawal_counter)
        withdrawal_score -= min(0.12, sum(item.weight for item in recent if item.id in counter_ids) * 0.08)
        self._maybe_add(
            hypotheses, "reduced_life_engagement", "生活参与可能正在收缩",
            "life_engagement", withdrawal_score, withdrawal_ids,
            "区分身体不便、暂时疲劳、失去兴致，还是觉得参与也没有价值。",
            counter_ids=counter_ids,
        )

        role_score, role_ids, role_counter = text_score(
            _ROLE_LOSS_WORDS, 0.52, "loss_of_role"
        )
        if lexical_fallback and persona and any(
            "负担" in belief or "没用" in belief
            or "不再被需要" in belief
            or "角色" in belief
            for belief in [
                *persona.case_conceptualization.automatic_thoughts_or_beliefs,
                *active_cbt_claims,
            ]
        ):
            role_score += 0.10
        self._maybe_add(
            hypotheses, "loss_of_role", "价值感或“仍然有用”的感觉可能下降",
            "agency_and_identity", role_score, role_ids,
            "从他在意的身份与真实能力切入，确认是不会、不能，还是觉得没人需要。",
            counter_ids=role_counter,
        )

        social_score, social_ids, social_counter = text_score(
            _SOCIAL_WORDS, 0.30, "social_disconnection"
        )
        for item in recent:
            if item.modality == "behavioral" and item.signal in {
                "social_participation", "contact_frequency", "out_of_home"
            }:
                strength = _negative_delta_strength(item)
                if strength > 0:
                    social_score += 0.38 * strength * item.weight
                    social_ids.append(item.id)
        self._maybe_add(
            hypotheses, "social_disconnection", "与真实关系的连接可能在减少",
            "social_connection", social_score, social_ids,
            "先确认他想避开的是活动本身、某个人，还是觉得自己去了也无事可做。",
            counter_ids=social_counter,
        )

        if lexical_fallback and current_self_physical_hit(utterance, _PHYSICAL_WORDS):
            physical_score, physical_ids, physical_counter = (
                _text_semantic_score(
                    utterance, _PHYSICAL_WORDS, 0.48, recent,
                    "physical_discomfort",
                )
            )
        else:
            # 旧事、假设、否定和第三方身体话题不能靠关键词升级为本人的
            # 当前身体假设。开放语义候选只在其引用所属的当前轮作为软证据，
            # 下一轮不会把一次旧猜测伪装成新症状。
            current_turn = max(
                (item.turn_index for item in recent),
                default=0,
            )
            current_semantic = [
                item for item in recent
                if item.turn_index == current_turn
                and item.modality == "semantic"
            ]
            physical_score, physical_ids, physical_counter = (
                _semantic_score(
                    current_semantic, "physical_discomfort"
                )
            )
            if lexical_fallback and any(word in utterance for word in _PHYSICAL_WORDS):
                physical_counter.extend(
                    _current_text_ids(utterance, recent)
                )
        for item in recent:
            if item.modality in ("behavioral", "visual") and item.signal in {
                "movement_speed", "gait_stability", "movement_energy", "mobility"
            }:
                strength = _negative_delta_strength(item)
                if strength > 0:
                    physical_score += 0.16 * strength * item.weight
                    physical_ids.append(item.id)
        self._maybe_add(
            hypotheses, "physical_discomfort", "可能存在需要进一步确认的身体不适",
            "care_need", physical_score, physical_ids,
            "只问具体变化和持续时间，不下诊断；必要时建议联系专业人员。",
            counter_ids=physical_counter,
        )

        autonomy_score, autonomy_ids, autonomy_counter = (
            text_score(
                _AUTONOMY_WORDS, 0.46,
                "autonomy_threat",
            )
        )
        self._maybe_add(
            hypotheses, "autonomy_threat", "自主或决定权可能受到挤压",
            "dignity_and_autonomy", autonomy_score, autonomy_ids,
            "区分他反感的是提醒本身、反复催促，还是别人替他作决定。",
            counter_ids=autonomy_counter,
        )

        grief_score, grief_ids, grief_counter = text_score(
            _GRIEF_WORDS, 0.44, "grief_or_transition"
        )
        self._maybe_add(
            hypotheses, "grief_or_transition", "可能正在经历离别、失落或生活转折",
            "loss_and_transition", grief_score, grief_ids,
            "确认他更想谈发生的事、保留一段记忆，还是暂时只需要有人陪着。",
            counter_ids=grief_counter,
        )

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:5]

    def _build_candidates(
        self,
        hypotheses: list[WorkingHypothesis],
        persona: Optional[PersonaProfile],
        utterance: str = "",
        *,
        lexical_fallback: bool = True,
    ) -> list[InterventionCandidate]:
        by_id = {h.id: h for h in hypotheses}

        def confidence(key: str) -> float:
            return by_id[key].confidence if key in by_id else 0.0

        has_protective_path = bool(
            persona and persona.case_conceptualization.protective_factors
        )
        turn = 0
        candidates = [
            InterventionCandidate(
                id="low_pressure_clarification",
                label="低压澄清",
                goal="用一个不丢面子的事实问题区分原因",
                guidance=(
                    "不要宣布你看穿了他。承认不确定，只问一个能区分身体、事情、"
                    "关系或价值感的具体问题。"
                ),
                avoid="追问感受、直接说“你其实很难过”、连续提问",
                score=0.08 + confidence("concealed_distress") * 0.55
                      + confidence("reduced_life_engagement") * 0.25,
                hypothesis_ids=[
                    key for key in ("concealed_distress", "reduced_life_engagement")
                    if key in by_id
                ],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="identity_based_reengagement",
                label="身份型再参与",
                goal="通过真实能力和角色恢复价值感与生活参与",
                guidance=(
                    "如果已有当前可引用的真实机会，可从职业、手艺或责任感出发"
                    "提出一个很小、可拒绝的参与可能；机会未知或本人需要留白时，"
                    "先承接经历，不必这拍找任务或受益人。不能凭年龄、旧职业或"
                    "活动类型推断当前体力，也不替他认定退出是正确的。"
                    "‘可能帮得上’不能改成‘正缺你、就在等你、非你不可’。"
                    "直接对他说‘你以前……的经验’，不要用姓名把档案复述给他。"
                    "明确允许拒绝，可说‘不想也没关系’；不得用撤回机会、"
                    "让家人失望、‘就当我没说’或制造内疚来推动参与。"
                ),
                avoid=(
                    "泛泛鼓励“你很有用”、把活动当任务摊派、用‘不帮就撤掉’"
                    "之类的损失压力、虚构别人正缺他、用第三人称姓名念档案、"
                    "赌气式退场"
                ),
                score=confidence("loss_of_role") * 0.70
                      + confidence("reduced_life_engagement") * 0.35
                      + (0.10 if has_protective_path else 0.0),
                consent_required=True,
                hypothesis_ids=[
                    key for key in ("loss_of_role", "reduced_life_engagement")
                    if key in by_id
                ],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="human_reconnection",
                label="真实关系重连",
                goal="把他重新连接到一个真实的人或共同活动",
                guidance=(
                    "保留真实的人际连接，只使用已知的人和事情。可按本轮观察许可"
                    "陪他斟酌想说的话，不必立即推动联系。代写不等于发送；同意也"
                    "不等于能力已接通，不承诺替他问对方、打电话或自动跟进。"
                ),
                avoid="伪造家人的请求、未经同意联系他人、制造对 Agent 的依赖",
                score=confidence("social_disconnection") * 0.68
                      + confidence("loss_of_role") * 0.22,
                consent_required=True,
                hypothesis_ids=[
                    key for key in ("social_disconnection", "loss_of_role")
                    if key in by_id
                ],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="physical_clarification",
                label="身体变化澄清",
                goal="确认身体变化是否需要进一步专业帮助",
                guidance=(
                    "问一个具体身体变化、发生时间或影响的问题。不诊断、不建议自行改药；"
                    "信号持续或明显时，转向专业人员。"
                ),
                avoid="把表情或声音直接解释成疾病、替代医生判断",
                score=confidence("physical_discomfort") * 0.90,
                consent_required=True,
                hypothesis_ids=["physical_discomfort"] if "physical_discomfort" in by_id else [],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="autonomy_restoration",
                label="自主恢复",
                goal="把节奏、选择和决定权交还给老人",
                guidance=(
                    "先承认这件事应由他参与决定，再提供一个可拒绝、可修改的小选项；"
                    "如果他只想自己处理，就尊重并约定何时需要帮忙。"
                ),
                avoid="用关心作理由继续催促、替他承诺、把唯一方案包装成选择",
                score=confidence("autonomy_threat") * 0.82
                      + confidence("concealed_distress") * 0.10,
                consent_required=True,
                hypothesis_ids=[
                    key for key in ("autonomy_threat", "concealed_distress")
                    if key in by_id
                ],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="meaning_oriented_companionship",
                label="意义型陪伴",
                goal="给离别或生活转折留下不被催促的表达与纪念空间",
                guidance=(
                    "不要急着转移注意力或劝他想开。跟随他愿意说的具体人、物和"
                    "生活变化；可以提供讲述、记录或安静陪伴三个低压入口。"
                ),
                avoid="泛泛说时间会治愈、强迫回忆、把正常哀伤病理化",
                score=confidence("grief_or_transition") * 0.78
                      + confidence("social_disconnection") * 0.10,
                hypothesis_ids=[
                    key for key in ("grief_or_transition", "social_disconnection")
                    if key in by_id
                ],
                selected_turn=turn,
            ),
            InterventionCandidate(
                id="quiet_companionship",
                label="安静陪伴",
                goal="维持关系安全，不急于改变他",
                guidance="证据不足时接住具体事实，留一点空间，不为了展示能力强行干预。",
                avoid="无证据读心、强行积极、连续建议",
                score=0.12,
                selected_turn=turn,
            ),
        ]
        self._apply_person_modifiers(candidates, persona, utterance, lexical_fallback=lexical_fallback)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _apply_person_modifiers(
        candidates: list[InterventionCandidate],
        persona: Optional[PersonaProfile],
        utterance: str = "",
        *,
        lexical_fallback: bool = True,
    ) -> None:
        """人物模型改变“选哪条路”，不只改变最后一句话的口吻。"""
        if persona is None:
            return
        by_id = {item.id: item for item in candidates}
        receptiveness = max(
            0.0,
            min(
                1.0,
                persona.cognitive_profile.intervention_receptiveness,
            ),
        )
        action_factor = 0.75 + 0.25 * receptiveness
        for candidate_id in (
            "identity_based_reengagement",
            "human_reconnection",
            "physical_clarification",
            "autonomy_restoration",
        ):
            by_id[candidate_id].score *= action_factor

        response_hypotheses = _relevant_response_hypotheses(
            persona, candidates
        )
        active_claims = " ".join(
            item.claim for item in response_hypotheses
        ) if lexical_fallback else ""
        active_dimensions = {
            item.dimension for item in response_hypotheses if item.dimension
        }
        mode = select_delivery_mode(persona)
        low_pressure_evidence = (
            mode == "low_pressure"
            or bool(active_dimensions & {"low_pressure", "autonomy"})
            or any(
                marker in active_claims
                for marker in (
                    "催促", "控制", "决定权", "留白", "空间", "连续追问",
                    "不喜欢被安排", "自己决定",
                )
            )
        )
        if low_pressure_evidence:
            by_id["low_pressure_clarification"].score += 0.08
            by_id["quiet_companionship"].score += 0.06
            if by_id["autonomy_restoration"].hypothesis_ids:
                by_id["autonomy_restoration"].score += 0.05
            if by_id["human_reconnection"].hypothesis_ids:
                by_id["human_reconnection"].score -= 0.02
        elif mode == "firm_with_options":
            if by_id["physical_clarification"].hypothesis_ids:
                by_id["physical_clarification"].score += 0.05
            if by_id["identity_based_reengagement"].hypothesis_ids:
                by_id["identity_based_reengagement"].score += 0.04

        if (
            by_id["identity_based_reengagement"].hypothesis_ids
            and (
                "identity_action" in active_dimensions
                or any(
                    marker in active_claims
                    for marker in (
                        "请教", "经验", "被需要", "具体事情", "动手", "角色"
                    )
                )
            )
        ):
            by_id["identity_based_reengagement"].score += 0.10
        if (
            by_id["human_reconnection"].hypothesis_ids
            and (
                "human_connection" in active_dimensions
                or any(
                    marker in active_claims
                    for marker in (
                        "熟人", "一起", "陪同", "真实关系", "先联系"
                    )
                )
            )
        ):
            by_id["human_reconnection"].score += 0.09
        if (
            by_id["meaning_oriented_companionship"].hypothesis_ids
            and (
                "meaning_memory" in active_dimensions
                or any(
                    marker in active_claims
                    for marker in ("回忆", "纪念", "慢慢讲", "离别")
                )
            )
        ):
            by_id["meaning_oriented_companionship"].score += 0.09
        if (
            by_id["autonomy_restoration"].hypothesis_ids
            and "autonomy" in active_dimensions
        ):
            by_id["autonomy_restoration"].score += 0.09
        if "direct_options" in active_dimensions:
            for candidate_id in (
                "physical_clarification",
                "identity_based_reengagement",
                "autonomy_restoration",
            ):
                if by_id[candidate_id].hypothesis_ids:
                    by_id[candidate_id].score += 0.05

        verified = " ".join(persona.aria_strategy_hints.tactics) if lexical_fallback else ""
        if (
            by_id["identity_based_reengagement"].hypothesis_ids
            and ("身份" in verified or "价值" in verified)
        ):
            by_id["identity_based_reengagement"].score += 0.08
        if (
            by_id["human_reconnection"].hypothesis_ids
            and ("连接" in verified or "关系" in verified)
        ):
            by_id["human_reconnection"].score += 0.08
        if (
            by_id["autonomy_restoration"].hypothesis_ids
            and ("自主" in verified or "选择" in verified)
        ):
            by_id["autonomy_restoration"].score += 0.08
        if (
            by_id["meaning_oriented_companionship"].hypothesis_ids
            and (
                "回忆" in verified
                or "纪念" in verified
                or "离别" in verified
            )
        ):
            by_id["meaning_oriented_companionship"].score += 0.08
        for candidate in candidates:
            if candidate.hypothesis_ids:
                candidate.score += _verified_outcome_adjustment(
                    persona, candidate.id, utterance
                )
        # 当前身体证据是本轮事实，不是沟通偏好。即使另一条心理社会路线曾经
        # 有好结果，也只能改变身体澄清之后怎么说，不能把身体路径挤出首位。
        physical = by_id["physical_clarification"]
        if physical.hypothesis_ids:
            strongest_other = max(
                (
                    candidate.score
                    for candidate in candidates
                    if candidate.id != physical.id
                ),
                default=0.0,
            )
            physical.score = max(
                physical.score,
                min(1.0, strongest_other + 0.05),
            )
        for candidate in candidates:
            candidate.score = max(0.0, min(1.0, candidate.score))

    @staticmethod
    def _personality_decision_basis(
        persona: Optional[PersonaProfile],
        selected: Optional[InterventionCandidate],
        candidates: Optional[list[InterventionCandidate]] = None,
        utterance: str = "",
        *,
        current_physical_source: str = "",
    ) -> dict:
        """给外部审计“哪些人物证据改变了决策”，不是暴露模型思维链。"""
        if persona is None:
            ranked = sorted(
                candidates or [],
                key=lambda item: item.score,
                reverse=True,
            )
            return {
                "delivery_mode": "neutral",
                "selected_intervention": selected.id if selected else None,
                "evidence": [],
                "verified_cooldown_routes": [],
                "current_request_override_routes": [],
                "current_physical_evidence_qualified": bool(
                    current_physical_source
                ),
                "current_physical_evidence_source": (
                    current_physical_source or None
                ),
                "decision_margin": _decision_margin(ranked),
                "candidate_ranking": [
                    {"id": item.id, "score": round(item.score, 3)}
                    for item in ranked[:3]
                ],
            }
        response_evidence = []
        for item in _relevant_response_hypotheses(
            persona, candidates or ([selected] if selected else [])
        ):
            response_evidence.append({
                "kind": "response_hypothesis",
                "hypothesis_id": item.id,
                "dimension": item.dimension,
                "claim": item.claim,
                "scope_tags": item.scope_tags,
                "confidence": round(item.confidence, 2),
                "status": item.status,
            })
        outcome_evidence_items = []
        cooldown_routes: list[str] = []
        override_routes: list[str] = []
        for candidate in candidates or ([selected] if selected else []):
            outcome_summary = _verified_outcome_summary(
                persona, candidate.id
            )
            if outcome_summary is not None:
                outcome_evidence = {
                    "kind": "observed_outcomes",
                    "intervention_id": candidate.id,
                    "score_adjustment": round(
                        _verified_outcome_adjustment(
                            persona, candidate.id, utterance
                        ),
                        3,
                    ),
                    **outcome_summary,
                }
                if _current_explicitly_reopens(utterance, candidate.id):
                    outcome_evidence["current_request_override"] = True
                    override_routes.append(candidate.id)
                elif outcome_summary["adaptation_state"] == "cooldown":
                    cooldown_routes.append(candidate.id)
                outcome_evidence_items.append(outcome_evidence)
        # 人物假设和结果记录分开限额，避免 4 个以上人物假设把真正改变
        # 本轮边界的 cooldown 从 prompt 与审计 trace 中挤掉。
        outcome_evidence_items.sort(
            key=lambda item: (
                item.get("adaptation_state") != "cooldown",
                item.get("intervention_id") or "",
            )
        )
        evidence = [
            *response_evidence[:3],
            *outcome_evidence_items[:3],
        ]
        ranked = sorted(
            candidates or [],
            key=lambda item: item.score,
            reverse=True,
        )
        return {
            "delivery_mode": select_delivery_mode(persona),
            "selected_intervention": selected.id if selected else None,
            "evidence": evidence,
            "verified_cooldown_routes": list(dict.fromkeys(cooldown_routes)),
            "current_request_override_routes": list(
                dict.fromkeys(override_routes)
            ),
            "current_physical_evidence_qualified": bool(
                current_physical_source
            ),
            "current_physical_evidence_source": (
                current_physical_source or None
            ),
            "decision_margin": _decision_margin(ranked),
            "candidate_ranking": [
                {
                    "id": item.id,
                    "score": round(item.score, 3),
                    "hypothesis_ids": item.hypothesis_ids,
                }
                for item in ranked[:3]
            ],
            "principle": (
                "人物假设只调整候选策略与沟通方式；安全阈值和护理事实不随人格变化。"
            ),
        }

    @staticmethod
    def _maybe_add(
        target: list[WorkingHypothesis],
        hypothesis_id: str,
        label: str,
        category: str,
        score: float,
        evidence_ids: Iterable[str],
        clarification: str,
        *,
        counter_ids: Optional[list[str]] = None,
    ) -> None:
        score = max(0.0, min(0.95, score))
        if score < 0.18:
            return
        target.append(WorkingHypothesis(
            id=hypothesis_id,
            label=label,
            category=category,
            confidence=score,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            counterevidence_ids=list(dict.fromkeys(counter_ids or [])),
            clarification_question=clarification,
        ))

    @staticmethod
    def _previous_delivery(
        ctx: EngineContext,
    ) -> Optional[DeliveredIntervention]:
        previous_turn = len(ctx.turns) - 1
        return next(
            (
                item
                for item in reversed(
                    ctx.reasoning.delivered_interventions
                )
                if item.turn_index == previous_turn
            ),
            None,
        )

    @staticmethod
    def _has_outcome(
        ctx: EngineContext, delivered: DeliveredIntervention
    ) -> bool:
        return any(
            o.intervention_id == delivered.id
            and o.turn_index == delivered.turn_index
            for o in ctx.reasoning.outcomes
        )

    def _append_outcome(
        self,
        ctx: EngineContext,
        delivered: DeliveredIntervention,
        value: str,
        signal: str,
        note: str,
    ) -> None:
        if self._has_outcome(ctx, delivered):
            return
        ctx.reasoning.outcomes.append(InterventionOutcome(
            intervention_id=delivered.id,
            intervention_label=delivered.label,
            goal=delivered.goal,
            value=value,
            signal=signal,
            note=note,
            turn_index=delivered.turn_index,
            hypothesis_ids=list(delivered.hypothesis_ids),
        ))


def _first_question(response: str) -> str:
    match = _QUESTION_SENTENCE.search(str(response or ""))
    return match.group(0).strip() if match is not None else ""


def _normalise_observation_text(value: str) -> str:
    return _OBSERVATION_NON_WORD.sub("", str(value or "")).lower()


def _relevant_response_hypotheses(
    persona: PersonaProfile,
    candidates: list[InterventionCandidate],
):
    """只让与当前状态相符的回应偏好进入确定性加权。"""
    active_state_ids = {
        hypothesis_id
        for candidate in candidates
        if candidate is not None
        for hypothesis_id in candidate.hypothesis_ids
    }
    out = []
    for item in online_influencing_hypotheses(persona):
        if (
            item.framework != "response"
            or item.status not in {"working", "stable"}
        ):
            continue
        tags = set(item.scope_tags)
        if tags and "general" not in tags and not (
            tags & active_state_ids
        ):
            continue
        out.append(item)
    return out


def _person_supports_route(
    persona: Optional[PersonaProfile],
    candidates: list[InterventionCandidate],
    intervention_id: str,
    *,
    lexical_fallback: bool = True,
) -> bool:
    """Whether scoped cross-session evidence specifically supports this route."""
    if persona is None or intervention_id not in _ROUTE_PERSON_DIMENSIONS:
        return False
    dimensions = _ROUTE_PERSON_DIMENSIONS[intervention_id]
    markers = _ROUTE_PERSON_MARKERS.get(intervention_id, ())
    return any(
        item.dimension in dimensions
        or lexical_fallback and any(marker in item.claim for marker in markers)
        for item in _relevant_response_hypotheses(persona, candidates)
    )


def _qualified_current_physical_evidence(
    utterance: str,
    evidence: list[ObservationEvidence],
    current_turn: int,
) -> str:
    """Return provenance only for current, personally attributable evidence."""
    if current_self_physical_hit(utterance, _PHYSICAL_WORDS):
        return "current_self_text"
    for item in evidence:
        if (
            item.turn_index == current_turn
            and item.modality == "behavioral"
            and item.signal in {
                "movement_speed", "gait_stability", "mobility"
            }
            and not item.fallback
            and item.weight >= 0.65
            and item.baseline_delta is not None
            and item.baseline_delta <= -0.25
        ):
            return "trusted_current_personal_baseline"
    return ""


def _human_reopen_action_is_specific(
    clause: str,
    term: str,
    start: int,
    intent_at: int,
) -> bool:
    """Require an interpersonal object for otherwise ambiguous verbs."""
    if term in {"联系", "问候", "发消息", "打电话", "见面"}:
        return True
    before = clause[intent_at:start]
    after = clause[start + len(term):start + len(term) + 12]
    if term in {"见", "约"}:
        return _HUMAN_REOPEN_TARGET.search(after) is not None
    if term == "聊":
        return bool(
            any(marker in before for marker in ("和", "跟", "找"))
            and _HUMAN_REOPEN_TARGET.search(before[-12:])
        )
    return False


def _current_explicitly_reopens(
    utterance: str,
    intervention_id: str,
) -> bool:
    """A current first-person request can reopen a previously cooled route.

    This is intentionally narrower than topic mention.  Past, hypothetical or
    negated references do not erase a verified cooldown; the person must state
    a present intention/request in the same clause as the route action.
    """
    terms = _CURRENT_REOPEN_TERMS.get(intervention_id, ())
    if not terms:
        return False
    split_pattern = (
        r"([，,。！？?；;：:\n]+|(?:但|不过|可是|然而))"
    )
    parts = [
        item for item in re.split(split_pattern, str(utterance or ""))
        if item
    ]
    clauses: list[tuple[str, str, str]] = []
    for index, part in enumerate(parts):
        if re.fullmatch(split_pattern, part) or not part.strip():
            continue
        before = (
            parts[index - 1]
            if index > 0 and re.fullmatch(split_pattern, parts[index - 1])
            else ""
        )
        after = (
            parts[index + 1]
            if index + 1 < len(parts)
            and re.fullmatch(split_pattern, parts[index + 1])
            else ""
        )
        clauses.append((part.strip(), before, after))
    decision: bool | None = None
    reported_next = False
    for clause, separator, following_separator in clauses:
        if reported_next:
            # 冒号/逗号后的第一小句仍属于“女儿说……”的转述；句号结束
            # 转述，转折词则明确把话语权切回当前说话人。
            if separator in {"但", "不过", "可是", "然而"} or any(
                marker in separator for marker in "。！？?；;\n"
            ):
                reported_next = False
            else:
                reported_next = False
                continue
        reported_here = bool(
            _CURRENT_REOPEN_REPORTED_INTRO_RE.search(clause)
        )
        if reported_here:
            # 同句后半段和紧随冒号的一小句都仍属于第三方转述。
            reported_next = True
            continue
        cancel_at = max(
            (item.start() for item in _CURRENT_REOPEN_CANCEL_RE.finditer(clause)),
            default=-1,
        )
        last_route_decision_at = -1
        term_occurrences: list[tuple[int, str]] = []
        for term in terms:
            start = clause.find(term)
            while start >= 0:
                term_occurrences.append((start, term))
                start = clause.find(term, start + len(term))
        # 不依赖 terms 在常量里的排列；同一句里最后表达的意向/撤回才生效。
        for start, term in sorted(set(term_occurrences)):
            prefix = clause[max(0, start - 10):start]
            if any(
                negation in prefix
                for negation in _CURRENT_REOPEN_NEGATIONS
            ) or _CURRENT_REOPEN_TRAILING_NEGATION_RE.search(prefix):
                decision = False
                last_route_decision_at = max(last_route_decision_at, start)
                continue
            intent_occurrences: list[tuple[int, str]] = []
            for intent in _CURRENT_REOPEN_INTENTS:
                intent_at = clause.find(intent)
                while 0 <= intent_at <= start:
                    intent_occurrences.append((intent_at, intent))
                    intent_at = clause.find(
                        intent, intent_at + len(intent)
                    )
            if not intent_occurrences:
                continue
            intent_at, intent = max(intent_occurrences)
            between = clause[intent_at + len(intent):start]
            # 自动解除 cooldown 的门槛故意窄：本人意向与当前行动必须
            # 近邻。松散的“想……联系”只作为普通对话内容，不能清除已
            # 验证边界；真实外部动作授权仍由独立工具门判断。
            if (
                len(between) > 16
                or _CURRENT_REOPEN_TRAILING_NEGATION_RE.search(between)
                or (
                    (
                        _CURRENT_REOPEN_MODAL_QUESTION_RE.search(clause)
                        or any(mark in following_separator for mark in "？?")
                    )
                    and not _CURRENT_REOPEN_PERMISSION_QUESTION_RE.search(
                        clause
                    )
                )
                or any(
                    marker in between
                    for marker in _CURRENT_REOPEN_EMBEDDED_NON_ACTION
                )
                or (
                    intent == "我想"
                    and between.startswith(("起", "到"))
                )
            ):
                continue
            intent_prefix = clause[max(0, intent_at - 8):
                                   intent_at + len(intent)]
            if _CURRENT_REOPEN_SUBJECT_DENIAL_RE.search(intent_prefix):
                decision = False
                last_route_decision_at = max(last_route_decision_at, start)
                continue
            intent_context = clause[max(0, intent_at - 10):start]
            non_current_at = max(
                (
                    intent_context.rfind(marker)
                    for marker in _CURRENT_REOPEN_NON_CURRENT
                ),
                default=-1,
            )
            current_reset_at = max(
                (
                    intent_context.rfind(marker)
                    for marker in _CURRENT_SELF_RESET
                ),
                default=-1,
            )
            if non_current_at >= 0 and current_reset_at < non_current_at:
                continue
            if intent == "帮我":
                request_prefix = clause[
                    max(0, intent_at - 8):intent_at
                ]
                if request_prefix and not request_prefix.endswith((
                    "你", "请", "请你", "麻烦", "麻烦你",
                    "能不能", "可以", "可不可以",
                )):
                    # “女儿帮我联系过”描述第三方行为，不是本人当前命令。
                    continue
            if (
                intervention_id == "human_reconnection"
                and not _human_reopen_action_is_specific(
                    clause, term, start, intent_at
                )
            ):
                continue
            decision = True
            last_route_decision_at = max(last_route_decision_at, start)
        if cancel_at > last_route_decision_at:
            decision = False
    return decision is True


def _relevant_verified_cooldowns(
    persona: Optional[PersonaProfile],
    candidates: list[InterventionCandidate],
    utterance: str,
) -> list[str]:
    """Return only cooldowns relevant to current evidence and not reopened now."""
    if persona is None:
        return []
    result: list[str] = []
    for candidate in candidates:
        if not candidate.hypothesis_ids:
            continue
        summary = _verified_outcome_summary(persona, candidate.id)
        if (
            summary is not None
            and summary["adaptation_state"] == "cooldown"
            and not _current_explicitly_reopens(utterance, candidate.id)
        ):
            result.append(candidate.id)
    return result


def _verified_outcome_summary(
    persona: PersonaProfile,
    intervention_id: str,
) -> Optional[dict]:
    """返回可逆的结果学习状态；同一会话只取最后一条高信息结果。"""
    by_session = _verified_outcome_records(persona, intervention_id)
    if len(by_session) < 2:
        return None
    recent_outcomes = [
        record.outcome for record in list(by_session.values())[-2:]
    ]
    if recent_outcomes == ["negative", "negative"]:
        adaptation_state = "cooldown"
    elif recent_outcomes == ["positive", "positive"]:
        adaptation_state = "reinforced"
    else:
        # 新的相反结果立即解除旧的增强/冷却，但必须再跨一个独立会话验证，
        # 才能进入另一稳定状态。因此策略永远可被后续真实结果推翻。
        adaptation_state = "uncertain"
    positive = sum(
        record.outcome == "positive"
        for record in by_session.values()
    )
    negative = sum(
        record.outcome == "negative"
        for record in by_session.values()
    )
    return {
        "independent_sessions": len(by_session),
        "positive": positive,
        "negative": negative,
        "recent_outcomes": recent_outcomes,
        "adaptation_state": adaptation_state,
    }


def _verified_outcome_records(
    persona: PersonaProfile,
    intervention_id: str,
) -> dict[str, InterventionOutcomeRecord]:
    """同一会话只保留最后一条高信息结果。"""
    by_session: dict[str, InterventionOutcomeRecord] = {}
    for record in persona.intervention_history:
        session_id = str(record.session_id or "").strip()
        if (
            record.intervention_id != intervention_id
            or record.signal in _WEAK_OUTCOME_SIGNALS
            or record.outcome not in {"positive", "negative"}
            or not session_id
        ):
            continue
        # 若同一会话后来有显式修正，以最后一次记录为准，并让它处在真实的
        # 最近位置；普通 dict 覆盖值不会改变插入顺序。
        by_session.pop(session_id, None)
        by_session[session_id] = record
    return by_session


def _verified_outcome_adjustment(
    persona: PersonaProfile,
    intervention_id: str,
    utterance: str = "",
) -> float:
    if intervention_id not in _OUTCOME_ADAPTIVE_INTERVENTIONS:
        return 0.0
    summary = _verified_outcome_summary(persona, intervention_id)
    if summary is None:
        return 0.0
    state = summary["adaptation_state"]
    if state == "cooldown":
        if _current_explicitly_reopens(utterance, intervention_id):
            # 历史结果改变默认值，不撤销本人此刻的新选择。新的交付结果会继续
            # 写回账本，因此重新尝试也仍是可逆、可学习的。
            return 0.0
        # 两个独立会话都明确失败时要真的停止重复同一路线，而不是只产生
        # 一个几乎看不见的小权重。它只作用于可替代的心理社会干预，且下一次
        # 相反的高信息结果就会解除；因此这是可逆冷却，不是永久封禁。
        return -0.36
    if state == "reinforced":
        return 0.10
    return 0.0


def _decision_margin(
    ranked: list[InterventionCandidate],
) -> Optional[float]:
    if len(ranked) < 2:
        return None
    return round(ranked[0].score - ranked[1].score, 3)


def _keyword_score(
    utterance: str,
    words: Iterable[str],
    score: float,
    evidence: list[ObservationEvidence],
) -> tuple[float, list[str]]:
    if not any(word in utterance for word in words):
        return 0.0, []
    text_ids = [
        item.id for item in evidence
        if item.modality == "text" and item.value == utterance
    ]
    return score, text_ids[-1:]


def _current_text_ids(
    utterance: str,
    evidence: list[ObservationEvidence],
) -> list[str]:
    return [
        item.id for item in evidence
        if item.modality == "text" and item.value == utterance
    ][-1:]


def _has_current_context_counter(
    utterance: str,
    hypothesis_id: str,
) -> bool:
    """识别同句里的明确反证，避免降级词表做局部截断。"""
    for pattern in _CURRENT_CONTEXT_COUNTERS.get(hypothesis_id, ()):
        matched = pattern.search(utterance)
        if matched is None:
            continue
        segment = matched.group()
        explicit_denial = segment.startswith(("不是", "并非"))
        if (
            not explicit_denial
            and hypothesis_id in {
                "reduced_life_engagement",
                "social_disconnection",
            }
            and any(
                marker in segment
                for marker in (
                    "不想", "不愿", "不能", "不再", "没去", "没有",
                    "取消", "放弃",
                )
            )
        ):
            continue
        return True
    return False


def _semantic_score(
    evidence: list[ObservationEvidence],
    hypothesis_id: str,
) -> tuple[float, list[str], list[str]]:
    """读取经原文校验的 LLM 候选；同类证据取强者，不把转述叠成票数。

    语义候选的权重低于直接观测，也会随轮次衰减。反证可以压低候选，但不会
    删除证据，因此 benchmark 和人工复核仍能看到推理链。
    """
    if not evidence:
        return 0.0, [], []
    current_turn = max(item.turn_index for item in evidence)
    support: list[tuple[float, str]] = []
    counter: list[tuple[float, str]] = []
    for item in evidence:
        if item.modality != "semantic":
            continue
        age = max(0, current_turn - item.turn_index)
        recency = 0.82 ** age
        if item.signal == hypothesis_id:
            support.append((0.72 * item.weight * recency, item.id))
        elif item.signal == f"counter:{hypothesis_id}":
            # 同等置信的原文反证至少应能抵消同等置信的语义候选，
            # 不能让“模型先猜了一次”比用户后来明确澄清更有粘性。
            counter.append((0.72 * item.weight * recency, item.id))
    raw_support = max((value for value, _ in support), default=0.0)
    counter_strength = max((value for value, _ in counter), default=0.0)
    score = max(0.0, raw_support - counter_strength)
    return (
        score,
        [item_id for _, item_id in support],
        [item_id for _, item_id in counter],
    )


def _text_semantic_score(
    utterance: str,
    words: Iterable[str],
    rule_score: float,
    evidence: list[ObservationEvidence],
    hypothesis_id: str,
) -> tuple[float, list[str], list[str]]:
    """把明确表达规则和开放语义候选合并，避免同一句话重复加权。

    规则是模型不可用时的可预测降级；语义候选负责隐晦、长句和方言转述。
    两者高度相关，因此取最大值而非相加。经校验的反证也可以抑制关键词误报。
    """
    explicit, text_ids = _keyword_score(utterance, words, rule_score, evidence)
    semantic, semantic_ids, counter_ids = _semantic_score(
        evidence, hypothesis_id
    )
    if _has_current_context_counter(utterance, hypothesis_id):
        return (
            0.0,
            list(dict.fromkeys([*text_ids, *semantic_ids])),
            list(dict.fromkeys([*counter_ids, *text_ids])),
        )
    if counter_ids and explicit > semantic:
        current_turn = max(
            (item.turn_index for item in evidence),
            default=0,
        )
        counter_penalties = [
            item.weight * (
                1.0 if item.turn_index == current_turn else 0.50
            )
            for item in evidence
            if item.id in counter_ids
        ]
        explicit = max(
            0.0,
            explicit - max(counter_penalties, default=0.0),
        )
    return (
        max(explicit, semantic),
        list(dict.fromkeys([*text_ids, *semantic_ids])),
        counter_ids,
    )


def _negative_delta_strength(item: ObservationEvidence) -> float:
    if item.baseline_delta is None:
        return 0.0
    return max(0.0, min(1.0, -float(item.baseline_delta)))


def _positive_signal(item: ObservationEvidence) -> bool:
    if item.baseline_delta is not None:
        return item.baseline_delta > 0.05
    if isinstance(item.value, (int, float)):
        return float(item.value) > 0.5
    return str(item.value).lower() in ("true", "high", "present", "yes", "positive")


def _positive_delta_strength(item: ObservationEvidence) -> float:
    if item.baseline_delta is None:
        return 0.0
    return max(0.0, min(1.0, float(item.baseline_delta)))
