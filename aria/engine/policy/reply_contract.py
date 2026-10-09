"""Bounded delivery constraints, not an open-domain truth or empathy classifier.

The prompt does the semantic work. These local checks catch a small, explicit
set of unsupported assertions and contractual contradictions without a second
model call. Passing them is not a reply-quality score.
"""
from __future__ import annotations

import re
from typing import Any

from ..context import EngineContext

REPLY_CONTRACT_VERSION = "1.6.0"
REPLY_CONTRACT_HEADING = "## 本轮回复契约"
UNAVAILABLE_CONTACT_REPLY = (
    "我不能直接替你联系对方。若你想说点什么，我可以陪你斟酌说法；要不要联系由你决定。"
)
_QUOTED = re.compile(r'“[^”]*”|‘[^’]*’|「[^」]*」|"[^"\n]*"')
_CLAUSES = re.compile(r"[，,。；;！？?\n]+")
_SELF_POSSESSION = re.compile(
    r"我(?:这儿|这里|手里|身边)(?:正|还|就)?(?:有|放着|拿着)"
)
_NONPHYSICAL = ("一个想法", "个想法", "一个建议", "个建议", "记录", "文字", "消息")
# Concrete demonstratives are referential claims. This is a deliberately finite
# vocabulary, not a claim to recognise every noun, paraphrase or dialect.
_OBJECTS = (
    "木料", "木头", "木盒", "木架", "茶杯", "杯子", "相册", "照片", "图纸",
    "桌子", "椅子", "抽屉", "棋盘", "收音机", "报纸", "毛衣", "围巾",
    "花盆", "工具箱", "螺丝刀", "画册", "手表", "毛笔", "书", "琴",
)
# Directional entailment, not an embedding/synonym model: a tea cup is a cup,
# but an unspecified cup is not necessarily a tea cup. Availability denials
# use the shared family so a newer "杯子丢了" defeats an older tea-cup mention.
_NOUN_SUPPORT_ALIASES = {"杯子": ("杯子", "茶杯")}
_NOUN_AVAILABILITY_FAMILY = {
    "杯子": ("杯子", "茶杯"),
    "茶杯": ("茶杯", "杯子"),
}
_DEMONSTRATIVE = re.compile(
    r"(?:那|这)(?:个|只|块|张|把|本|台|件|条|副)(?:旧|新|小|老)?("
    + "|".join(_OBJECTS) + r")"
)
_DEMEANING = re.compile(
    r"(?:确实|真的|就是|的确|反而|只会|难免|容易)(?:挺|很|有点|会)?"
    r"(?:碍事|添乱|没用|是负担|拖累)"
    r"|(?:活|事|活动)(?:儿)?(?:确实|真的|就是)不适合你(?:插手|做|参加)?"
)
_PROBE = re.compile(
    r"(?:是.{0,30}还是|是不是|为什么|怎么不|什么原因).{0,30}"
    r"(?:不.{0,2}舒服|身体|身上|心里|心事|心情|不想|没意思|怕|累|乏|兴致|动弹|没劲)"
)
_DRAFT_REQUEST = re.compile(
    r"(?:帮我|替我|给我|请你|麻烦你)(?:先|只|就|来|试着|简单|直接){0,3}"
    r"(?:写|拟|起草|润色)(?:一|个|段|条|句|封|消息|短信|一下)"
)
_DRAFT_PROMISE = re.compile(
    r"(?:我)?(?:(?:来|会|可以|先|这就|帮你|替你|给你|为你|帮您|给您)){0,4}"
    r"(?:写|拟|起草|润色)(?:一句|一段|一条|一封|个|一下|好)?(?:消息|短信|话|草稿)?"
)
_DRAFT_META_CLAUSES = (
    re.compile(r"(?:好|好的|行|可以|没问题|明白|放心|别担心)"),
    re.compile(r"(?:你|您)(?:自己)?(?:先)?(?:看完|看看|看一眼|看过|看了)"
               r"(?:觉得)?(?:合适|满意|没问题)?(?:了)?(?:再发|再发送|再决定|再决定发不发)?"),
    re.compile(r"(?:是否|要不要|发不发|什么时候)?(?:发送|发出去)?(?:这一步)?"
               r"(?:由你|你自己|由您)(?:决定|定)"),
    re.compile(r"(?:我)?(?:不会|不|也不)(?:替你)?(?:发送|发出去|报名|告诉(?:孩子|家人|别人|子女))"),
    re.compile(r"(?:不合适|不满意)(?:的话)?(?:咱们|我们)?(?:就|再|可以再)?改"),
)
_PROMOTIONAL_OFFER = re.compile(
    r"(?:领|送|赠)(?:免费)?(?:鸡蛋|礼物|赠品|礼品)|免费(?:体验|旅游)|促销"
)
_OFFER_SETTING = re.compile(r"讲座|活动|课程|体验|旅游|推销|促销")
_OFFER_ASSURANCE = re.compile(
    r"(?:去|参加|听听|看看|体验|试试)(?:一下|也)?(?:无妨|没事|没关系|没有风险|没风险)"
    r"|放心(?:去|参加|听|体验|试)|肯定(?:安全|靠谱|可靠)|不会骗人|没有任何风险"
)
_SPACE_MARKERS = ("留白", "留空间", "不追问", "停止追问", "不要求解释", "不被追问")
_CONTACT_OFFER = re.compile(
    r"我(?P<qualifiers>(?:可以|能|会|来|这就|马上|稍后|现在|准备|打算|将|"
    r"试着|尝试|帮你|替你|为你){0,4})"
    r"(?P<action>联系|通知|打电话|发消息|发送消息|问问|问一下|打听)"
)
_CONTACT_NEGATION = ("不能", "不会", "无法", "没有办法", "并非", "不是", "不代表")
_PHYSICAL_OFFER = re.compile(
    r"我(?:(?:已经|刚刚|刚才|现在|马上|这就|正在|已|刚|可以|能|会|来|试着|尝试|帮你|帮您|替你|替您|为你|为您|给你|给您)){0,5}"
    r"(?:把(?:它|这|那|个|只|旧|新|小|老|茶杯|杯子|木盒|桌子|椅子){1,5})?"
    r"(?P<action>擦干净|擦拭|清洗|搬动|搬走|拿走|端来|倒(?:好|了)?(?:一杯|杯)?(?:温|热|凉|白开)?水|把(?:温|热|凉|白开)?水倒(?:好|完|了)|送水|扶(?:你|您)|抱(?:你|您)|牵(?:你|您))"
)
_QUESTION_NORMALISE = re.compile(r"[\s，,。；;！？?：:\-—]+")
_DURATION_QUESTION = "这个不舒服是刚刚才有，还是已经持续一阵了？"
_PHYSICAL_INTEREST_QUESTIONS = {
    "最近不太想参与，是身体累，还是觉得那些事没什么意思了？",
    "最近不太想出门，是身体不太方便，还是去了也觉得没什么意思？",
}


def _draft_requested(text: str) -> bool:
    for sentence in re.split(r"[。；;！？?\n]+", text):
        for match in _DRAFT_REQUEST.finditer(sentence):
            prefix = sentence[:match.start()]
            suffix = sentence[match.end():]
            local_prefix = re.split(r"[，,]", prefix)[-1]
            if any(marker in prefix for marker in (
                "如果", "假如", "以后", "将来", "她说", "他说", "对方说", "新闻里", "故事里",
            )) or any(marker in local_prefix for marker in ("上次", "昨天", "以前")) or re.search(
                r"(?:不用|不必|不要|不需要|别|不是让)(?:你|您)?(?:再|先|现在)?$", local_prefix,
            ) or re.search(
                r"(?:^|[，,])(?:那|先|今天|现在)?(?:先不用|不用写了|不写了|别写了|算了)", suffix,
            ):
                continue
            return True
    return False


def clarification_question_matches(response: str, preferred_question: str) -> bool:
    """Accept finite, meaning-preserving forms of server-owned fact questions.

    This is a bounded delivery check, not a semantic quality judge. New question
    families remain exact until their permitted information slots and negative
    controls are specified. Quoted questions and a second question do not count.
    """
    questions = re.findall(r"([^。；;！？?\n]+)[？?]", _QUOTED.sub("", response))
    if len(questions) != 1:
        return False
    def normalise(value: str) -> str:
        return _QUESTION_NORMALISE.sub("", value)
    question = normalise(questions[0])
    preferred = normalise(preferred_question)
    # Exact wording is not an escape hatch for an attached diagnosis, invented
    # consent, or cancellation of the question. Only neutral acknowledgement /
    # clarification prefixes may share its sentence; other assertions belong
    # to the independent grounding review, not substring approval here.
    for prefix in ("我想确认一下", "我先确认一下", "想确认一下", "先确认一下", "我听见了", "我听到了"):
        if question.startswith(prefix):
            question = question[len(prefix):]
            break
    if preferred and preferred == question:
        return True
    if preferred == normalise(_DURATION_QUESTION):
        # A start-time or duration query collects the same fact as the original
        # recent/persistent contrast. It must not introduce a specific symptom
        # (e.g. chest pain), a diagnosis, or a new causal explanation.
        subject = r"(?:这个|这种|这次|你说的|现在的)?(?:不舒服|不适|难受|症状|感觉)"
        return bool(re.fullmatch(
            subject + r"(?:是)?(?:从)?(?:什么时候|何时)开始(?:的|了)?(?:呢)?"
            r"|" + subject + r"(?:已经)?(?:持续|有)(?:了)?多久(?:了|呢)?"
            r"|" + subject + r"(?:是)?(?:刚刚才有|刚出现的?|刚开始的?|刚才开始的?|刚刚开始的?)"
            r"(?:还是|或是|还是说)(?:已经)?(?:持续(?:了)?一阵(?:子)?了?|持续(?:了)?一段时间了?|有一阵子了)(?:呢)?",
            question,
        ))
    if preferred in {normalise(item) for item in _PHYSICAL_INTEREST_QUESTIONS}:
        # These two questions contrast a bodily limitation with low interest.
        # Merely asking about fatigue, or substituting fear of others, is not
        # equivalent. Either branch order is permitted; new diagnostic premises
        # and unrelated suffixes do not fit these finite branch patterns.
        question = re.sub(r"^最近不太想(?:参与|参加|出门)", "", question)
        branches = re.split(r"还是说|还是|或是", question)
        if len(branches) != 2:
            return False
        body = re.compile(
            r"(?:是)?(?:最近)?(?:觉得)?(?:最近)?(?:身体|身上)?"
            r"(?:(?:有点|有些)?(?:累|疲惫|疲乏|不舒服|吃不消)|不(?:太)?方便|不太舒服)(?:了|呢)?"
        )
        interest = re.compile(
            r"(?:是)?(?:觉得)?(?:那些事|去了也|坐那儿|坐在那儿|在那儿)?"
            r"(?:插不上话)?(?:没意思|没什么意思|提不起兴趣|没兴趣|没兴致)(?:了|呢)?"
        )
        return bool(
            (body.fullmatch(branches[0]) and interest.fullmatch(branches[1]))
            or (interest.fullmatch(branches[0]) and body.fullmatch(branches[1]))
        )
    return False


def reply_quality_contract(
    ctx: EngineContext, *, utterance: str | None = None,
) -> dict[str, Any]:
    current = str(utterance if utterance is not None else ctx.working_memory.get("current_utterance") or "")
    current_surface = _QUOTED.sub("", current)
    plan = ctx.reasoning.observation_plan
    basis = ctx.working_memory.get("personality_decision") or {}
    leave_space = bool(
        plan is not None and plan.mode == "observe"
        and not basis.get("current_physical_evidence_qualified")
        and any(
            item.get("kind") == "response_hypothesis"
            and item.get("status") in {"working", "stable"}
            and item.get("dimension") in {"low_pressure", "autonomy"}
            and any(marker in str(item.get("claim") or "") for marker in _SPACE_MARKERS)
            for item in basis.get("evidence", []) if isinstance(item, dict)
        )
    )
    # A companionship permission question is not a safety fact query. Apply
    # this equally to every input arm, using the server-owned observation
    # evidence, never the model's replaceable selected route or persona label.
    optional_invitation = bool(
        plan is not None and plan.mode == "ask"
        and plan.hypothesis_ids
        and set(plan.hypothesis_ids) <= {"grief_or_transition"}
        and not basis.get("current_physical_evidence_qualified")
    )
    # In v3 the same call interprets current intent. A pre-model ordinary
    # ask plan is a hypothesis, not authority to force a diagnostic question.
    # Safety takeover and explicit wait remain separate, stronger boundaries.
    semantic_v3 = ctx.working_memory.get("_online_decision_version") == "online_turn_decision_v3"
    advisory = bool(semantic_v3 and plan
                    and not ctx.working_memory.get("risk_flag"))
    return {
        "version": "1.7.3" if semantic_v3 else REPLY_CONTRACT_VERSION,
        "observation_advisory": advisory,
        "observation_mode": plan.mode if plan is not None else "observe",
        "observation_question": plan.question.strip() if plan is not None else "",
        "question_budget": 1 if advisory else (0 if leave_space or (plan and plan.mode == "wait") else 1),
        "question_requirement": (
            "optional" if advisory
            else "none" if leave_space or (plan and plan.mode == "wait")
            else "invitation_optional" if optional_invitation
            else "exact" if plan and plan.mode == "ask"
            else "optional"
        ),
        "person_requires_space": leave_space and not advisory,
        "task_questions_separate": True,
        "draft_requested": _draft_requested(current_surface),
        # This does not classify the activity as fraudulent. Even a low-risk
        # promotional offer is not a verified safety claim. No remote lookup or
        # actual protective action is inferred from these two finite cues.
        "promotional_offer_no_safety_verification": bool(
            _PROMOTIONAL_OFFER.search(current_surface) and _OFFER_SETTING.search(current_surface)
        ),
        # A tool listing only permits an explicitly qualified *attempt*. It is
        # neither a delivery receipt nor proof that a person can reply. Other
        # outbound capabilities must be wired explicitly, not inferred from
        # arbitrary tool descriptions, model text or persona memory.
        "notification_attempt_advertised": any(
            str(item).split(":", 1)[0] == "notify_family"
            for item in ctx.working_memory.get("_advertised_tools_for_turn", [])
        ),
    }


def contract_prompt(contract: dict[str, Any]) -> str:
    """One shared delivery contract for ordinary and integrated generation.

    This is semantic guidance, not a claim that the lexical guard can recognise
    bodily competence, inner experience or every implied real-world capability.
    """
    mode = contract["observation_mode"]
    question = str(contract.get("observation_question") or "")
    if contract.get("observation_advisory"):
        observation = (
            "response 的观察建议：本地候选尚未完成本轮语义判断，不要求使用预设问句。"
            "按当前原话和必要上下文判断本人正在请求什么、纠正什么、停止什么。"
            "可直接交付答案或内容、接受纠正、安静倾听；只有当前需要且缺少必要信息时"
            "才问一个相关问题，不因候选状态或关键词要求本人回答心理二选一。"
            "当前身体求助须认真回应；急症或自伤关注仍按独立安全合同分流。\n"
            "本地等待或冷却建议只限制重复推动对应方向，不取消本人另一个当前请求；"
            "第三人转述不等于本人拒绝，局部停止不等于结束全部交流。"
            "有依据的留白偏好仍按本轮范围尊重，不据此索要解释。\n"
        )
    elif mode == "ask" and contract.get("question_requirement") == "invitation_optional":
        observation = (
            "response 的观察合同=ask：这是陪伴邀请，不是必须回答的事实检查。"
            "可自然给一个允许不答的邀请，也可以不问、安静承接；不强制逐字复述。"
            "不能追问为何失去、何时放下或要求处理物品。\n"
        )
    elif mode == "ask" and question:
        if question in {_DURATION_QUESTION, *_PHYSICAL_INTEREST_QUESTIONS}:
            observation = (
                "response 的观察合同=ask：只收集这一个问题所需的事实，不另加问题。"
                "可保留问句，也可自然近义改写不适的起始/持续时间、身体限制与兴趣的对比，"
                "但不得丢失对比维度、添新症状或变成诱导结论。\n"
                f"本轮事实问题:{question}\n"
            )
        else:
            observation = (
                "response 的观察合同=ask：只问且原样使用这一个事实问题，不另加问题。\n"
                f"本轮唯一允许的问句:{question}\n"
            )
    elif mode == "wait" or contract["person_requires_space"]:
        observation = (
            f"response 的观察合同={mode}：人物证据要求留白；0 个情绪或日常问题，"
            "只承接和保留自主，不要求解释、不换方式推动。\n"
        )
    else:
        observation = (
            "response 的观察合同=observe：先回应原话，不盘问原因、不出二选一诊断题；"
            "最多一个锚定原话且可以不答的日常邀请，也可以不问。\n"
        )
    # This guidance is always visible. The finite draft diagnostic below is
    # not a semantic router for whether the model should complete a task.
    # Let the model understand the current request from context, including
    # negation, reported speech, conditional requests and changed intentions.
    task_delivery = (
        "先按本人当前原话和相关上下文理解正在请求的帮助；转述、否定、未来或假设的任务"
        "不当作现在要做，最新纠正和停止优先。当前明确的问答、选择解释、文字整理等"
        "能在对话内完成时，正文就交付实际答案或内容，不只承诺稍后做，也不以陪伴或隐私提醒代替。"
        "当本轮明确要求起草/改写时，正文必须现在给出一版可看的实际措辞，不只说‘我来写’。"
        "已有相关上下文就沿用本人明确提出的内容和限制，不重复索要；确实缺少内容才问一个必要问题。"
        "草稿用引号或‘草稿：’标明，不把未授权的私人背景写给第三人；草稿不等于发送、报名或同意。\n"
        "本人只问一个问题或限定当前一步时，回答这个范围后停下，不顺带展开后续操作；"
        "按步骤交流不等于每轮追加下一步，简短不等于省略本人明确需要的内容。"
        "准备内容、本人审阅、是否发送、允许谁转发是不同决定，不能彼此推定。"
        "若先前只承诺提供内容而实际尚未给出，本人接着索要或审阅时应补上内容，"
        "不能假装已有可看的正文；也不自动重做历史任务。\n"
    )
    offer_boundary = (
        "本轮提到赠品/体验类活动：赠品、对方热情、尚未发现收费都不是安全核验。"
        "不替未知活动背书，也不单凭这些信息判为诈骗；可保留选择并提出自愿核对主办方或安排，"
        "不能把‘可以随时离开’当作安全保证。\n"
        if contract.get("promotional_offer_no_safety_verification") else ""
    )
    return (
        REPLY_CONTRACT_HEADING + "\n"
        "优先级：安全与真实能力 > 本人当前事实/纠正/明确任务 > 观察合同 > "
        "人物候选做法。观察/留白许可优先，候选示例不是每拍必做的任务。\n"
        + observation
        + task_delivery + offer_boundary
        + "相处说明只确认本轮已实际展示、仍有效的具体复述，不把人物推断、分享或通知权限一并确认。"
        "本人回答的语义判断只是提案；服务器核对并保存前，不声称已经确认或撤回。"
        "本人选择现在聊变化记录，只允许接下来一次当前会话的请求，不证明本人平静，也不是长期提醒或外发授权。\n"
        + "用户明确提出的生活任务仍需交代真实状态；留白不取消任务，必要的任务"
        "确认只围绕任务，不借机问情绪。全轮最多一个问题。\n"
        "承接事实，不代言内心：他说过的感受可按原话回应；你推测的感受只作"
        "低确定性的可能，或不点破。不要把‘也许’升级成‘心里肯定’，也不必"
        "机械地说‘听起来’。人物假设用于选回应方式，不是当前事实。\n"
        "本人已解释噪声、停顿或其他观察的来由时，先尊重这次纠正；否认某种情绪"
        "不是隐瞒该情绪的新证据。恢复聊天不等于同意重问刚才的身体/心事二选一；"
        "没有新的本人求助或具体危险事实，不重开已回应的求证。未来决定也不能说成已经发生。\n"
        "不因年龄、手艺身份或活动类型推断体力；只按本人当前具体限制调整建议，"
        "不替他判定能不能做、适不适合或退出对不对。自责不是能力结论；接住"
        "感受，不赞同没用、碍事或不配参与。\n"
        "只引用当前仍成立的用户事实或可引用记忆，不把历史安排、假设、例子"
        "说成眼前物品或他人需求。机会未知就保留条件或陪着，不编造受益人，"
        "不以善意谎言、内疚或撤回机会催参与。\n"
        "能力仅限对话与本轮外部行动工具，无身体、物品、现场视野或线下动手能力；"
        "‘搭把手’等含混帮助须明确为出主意或斟酌说法。代写不等于发送，征得同意也不会凭空接通能力；"
        "通知须有相应工具，有真实回执才报成功；不保证对方回复或自动跟进。\n"
        "用户可见正文用1–2个自然短句，尽量80字内，不报人物档案、诊断标签或"
        "推理过程；在线决策时仅指response字段。"
        "这是一般交流的默认长度；本人明确请求的具体内容应保留必要完整性，"
        "按其要求简明交付，不为凑短句删掉答案或产物。"
        "邀请可平静拒绝，不用‘就当我没说’施压。"
    )


def _assertion_prefix(prefix: str) -> bool:
    return not any(marker in prefix for marker in (
        "如果", "假如", "假设", "要是", "比如", "例如", "并不是", "不是说",
        "不代表", "不等于", "不能说", "并非", "未必", "不是", "不觉得", "不会",
    ))


def quality_violations(
    response: str, sources: list[str], contract: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    text = _QUOTED.sub("", response)
    violations.extend(contact_offer_violations(response, contract))
    violations.extend(physical_offer_violations(response))
    if contract.get("promotional_offer_no_safety_verification"):
        for clause in re.split(r"[。；;！？?\n]|但是|不过|可是|然而|但", text):
            for match in _OFFER_ASSURANCE.finditer(clause):
                if not any(word in clause[:match.start()] for word in (
                    "不能说", "不能保证", "不代表", "不等于", "不是说", "不要说", "别说",
                    "对方说", "他们说", "他说", "她说", "听说",
                )):
                    violations.append("unverified_promotional_safety_assurance")
    for clause in _CLAUSES.split(text):
        for match in _SELF_POSSESSION.finditer(clause):
            if _assertion_prefix(clause[:match.start()]) and not any(
                clause[match.end():].startswith(word) for word in _NONPHYSICAL
            ):
                violations.append("unsupported_agent_physical_presence")
        for match in _DEMONSTRATIVE.finditer(clause):
            noun = match.group(1)
            if _assertion_prefix(clause[:match.start()]) and not _noun_grounded(noun, sources):
                violations.append("unsupported_concrete_referent")
        for match in _DEMEANING.finditer(clause):
            prefix = clause[:match.start()]
            if _assertion_prefix(prefix) and not any(
                marker in prefix for marker in ("你担心", "你觉得", "你怕", "你说", "他说", "她说")
            ):
                violations.append("dignity_undermining_agreement")
    if (
        contract["observation_mode"] == "observe"
        or contract.get("question_requirement") == "invitation_optional"
    ):
        if contract["person_requires_space"] and (
            "?" in text or "？" in text or _diagnostic_probe(text)
        ):
            violations.append("question_overrode_person_space")
        elif _diagnostic_probe(text):
            violations.append("diagnostic_probe_while_observing")
    return list(dict.fromkeys(violations))


def _diagnostic_probe(text: str) -> bool:
    """Finite body/inner-state probes, excluding explicit refusal to ask one."""
    for sentence in re.split(r"[。；;！？?\n]|但是|不过|可是|然而|但", text):
        for match in _PROBE.finditer(sentence):
            if not any(word in sentence[:match.start()] for word in (
                "不会再问", "不再问", "不追问", "不要再问", "不用回答", "不必回答",
            )):
                return True
    return False


def task_delivery_diagnostics(response: str, contract: dict[str, Any]) -> list[str]:
    """Recognise only a meta-promise with no draft, not grade draft usefulness.

    Do not manufacture a third-party message from arbitrary private history to
    make this diagnostic pass. The actual model reply remains visible to the
    independent assessor; this is an incomplete-task flag, not a model score.
    A natural unquoted draft or necessary content question is not rejected.
    """
    if not contract.get("draft_requested"):
        return []
    clauses = [part.strip() for part in _CLAUSES.split(response) if part.strip()]
    if not clauses or not any(_DRAFT_PROMISE.fullmatch(part) for part in clauses):
        return []
    if all(_DRAFT_PROMISE.fullmatch(part) or any(pattern.fullmatch(part) for pattern in _DRAFT_META_CLAUSES)
           for part in clauses):
        return ["requested_draft_not_delivered"]
    return []


def physical_offer_violations(text: str) -> list[str]:
    """Finite explicit physical-help offers, not general capability inference.

    No physical actuator is currently wired into this delivery contract. A
    user's consent and the mere presence of an object cannot create one. A
    future actuator needs an explicit capability/receipt contract, not a tool
    description which merely mentions a physical verb.
    """
    for sentence in re.split(r"[。；;！？?\n]+", _QUOTED.sub("", text)):
        if any(marker in sentence for marker in ("假设", "举例", "故事里", "电视剧里")):
            continue
        for clause in re.split(r"[，,]|但是|不过|可是|然而|但", sentence):
            for match in _PHYSICAL_OFFER.finditer(clause):
                if any(marker in clause[:match.start()] for marker in (
                    *_CONTACT_NEGATION, "你说", "他说", "她说", "你担心",
                )):
                    continue
                if clause[match.end():].startswith(("数据", "记录", "文字", "文稿")):
                    continue
                return ["unsupported_agent_physical_help"]
    return []


def contact_offer_violations(text: str, contract: dict[str, Any]) -> list[str]:
    """Finite first-person offers, distinct from post-action receipt checking.

    Consent conditionals ("if you agree, I will call") still promise a
    capability. Quoted speech and explicit fiction are not such promises.
    This recognises neither all indirect offers nor open-domain capabilities.
    """
    for sentence in re.split(r"[。；;！？?\n]+", _QUOTED.sub("", text)):
        if any(marker in sentence for marker in ("假设", "举例", "故事里", "电视剧里")):
            continue
        # A denial before a contrast does not negate the new offer after it,
        # including punctuation-free ASR ("不能打电话但我可以替你问").
        for clause in re.split(r"[，,]|但是|不过|可是|然而|但", sentence):
            for match in _CONTACT_OFFER.finditer(clause):
                prefix = clause[:match.start()]
                if any(marker in prefix for marker in (*_CONTACT_NEGATION, "你说", "他说", "她说", "你担心")):
                    continue
                action = match.group("action")
                qualifiers = match.group("qualifiers")
                suffix = clause[match.end():]
                future_marked = any(word in qualifiers for word in (
                    "会", "这就", "马上", "稍后", "准备", "打算", "将", "可以", "能",
                ))
                if not future_marked and (
                    suffix.startswith(("了", "过", "成功", "完"))
                    or (
                        action not in {"问问", "问一下", "打听"}
                        and suffix.endswith("了")
                        and not suffix.endswith(("就行了", "就好了", "算了"))
                    )
                ):
                    continue  # Completion claims belong to the receipt guard.
                if action in {"问问", "问一下", "打听"} and not any(
                    word in qualifiers for word in ("帮你", "替你", "为你")
                ):
                    continue  # Do not confuse an ordinary user question with outreach.
                if action in {"通知", "发消息", "发送消息"} and (
                    contract.get("notification_attempt_advertised")
                    and any(word in qualifiers for word in ("尝试", "试着"))
                ):
                    continue
                return ["unsupported_future_contact_offer"]
    return []


def _noun_grounded(noun: str, sources: list[str]) -> bool:
    # Sources are newest-first. A current denial/disposal cannot be rescued by
    # an older positive mention; merely discussing a hypothetical object is not
    # evidence of possession or current availability either.
    for source in sources:
        # Do not promote quotations/fiction into available physical objects.
        # Keep comma-linked conditionals together; splitting them at a comma
        # drops the antecedent and turns a hypothetical consequent into a fact.
        for clause in reversed(re.split(r"[。；;！？?\n]+", _QUOTED.sub("", source))):
            family = _NOUN_AVAILABILITY_FAMILY.get(noun, (noun,))
            mentions = [clause.find(alias) for alias in family if alias in clause]
            if not mentions:
                continue
            prefix = clause[:min(mentions)]
            if not _assertion_prefix(prefix) or any(marker in prefix for marker in (
                "电视剧", "电视里", "故事里", "小说里", "假想", "举例",
            )) or any(marker in clause for marker in (
                "没有", "没带", "不在", "扔了", "扔掉", "送走", "丢了", "找不到",
            )):
                return False
            return any(
                alias in clause
                for alias in _NOUN_SUPPORT_ALIASES.get(noun, (noun,))
            )
    return False
