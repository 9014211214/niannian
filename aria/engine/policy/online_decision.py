"""在线单次决策：当前状态候选、沟通方向和自然回复在同一次调用中完成。

长期 DISC / Big Five / 依恋 / CBT 不在这里重算。它们由会后 reflection
形成可推翻人物假设，本模块只读取 prompt 中已经确认的人物模型，并对当轮
信号作低置信、可校验的情境判断。
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
import re
from typing import Any

from ...interfaces.reasoning import SemanticStateSignal
from ..context import EngineContext
from ..model.persona import PersonaProfile
from .prompt import (
    confirmed_prompt_hypotheses, delivery_basis_ids, permitted_intervention_prompt,
)
from .reply_contract import (
    REPLY_CONTRACT_HEADING, UNAVAILABLE_CONTACT_REPLY, contract_prompt, quality_violations,
    clarification_question_matches, reply_quality_contract, task_delivery_diagnostics,
)

ONLINE_DECISION_VERSION = "online_turn_decision_v1"
ONLINE_ID_NAMESPACE_VERSION = "online_json_id_namespaces_v1"
# v2: 字段顺序改为“状态 → 情绪 → 轻量 MWM 推演 → 选择 → 计划 → response”，
# 让回复在推演之后生成；推演只能引用离线画像/假设与当前原话，不重算人格。
# v1 保留原样，供历史复现与对照臂使用。
ONLINE_DECISION_VERSION_V2 = "online_turn_decision_v2"
# v3 keeps the v1 decision shape and adds a same-call semantic safety signal.
# Historical v1/v2 schemas remain reproducible; minimal live clients select v3.
ONLINE_DECISION_VERSION_SAFETY = "online_turn_decision_v3"
ONLINE_DECISION_VERSIONS = (ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2, ONLINE_DECISION_VERSION_SAFETY)
SAFETY_CONCERNS = ("none", "self_harm", "medical_emergency", "fall", "unsafe_medication")
ONLINE_DECISION_VERSION_ENV = "ARIA_ONLINE_DECISION_VERSION"
DELIBERATION_VERSION = "turn_deliberation_v1"
DELIBERATION_READ_LIMIT = 40
DELIBERATION_REACTION_LIMIT = 30
DELIBERATION_MAX_OPTIONS = 3
DELIBERATION_MAX_BASIS = 3
_BASIS_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
COMMUNICATION_PLAN_VERSION = "communication_plan_v1"
COMMUNICATION_PLAN_PACES = frozenset({
    "stepwise", "overview_first", "brief_exchange", "open_listening", "pause",
})
COMMUNICATION_PLAN_TEXT_LIMIT = 80
DETERMINISTIC_MODEL_FAILURE_REPLY = (
    "我在，刚才没接稳。你不用重说，我们先停一会儿。"
)

_ALLOWED_STATE_IDS = {
    "concealed_distress",
    "reduced_life_engagement",
    "loss_of_role",
    "social_disconnection",
    "physical_discomfort",
    "autonomy_threat",
    "grief_or_transition",
}
_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
_GENERIC_MINIMISING_PHRASES = (
    "别想太多", "想开点", "没什么大不了", "这没什么", "大家都这样",
)
_DIVERGENCE_MINIMISING_PHRASES = ("没事就好", "那就好")
_PARTICIPATION_MINIMISING_PHRASES = (
    "歇着也挺好", "歇着挺好",
    "不如在家清净", "反正去了也没用", "反正也插不上话", "反正坐在那儿也插不上话",
    "反正坐在那儿也帮不上忙", "以后都别去了", "以后就别去了", "一直不去也挺好",
)


def _minimises_participation(response: str, *, physical_qualified: bool) -> bool:
    """Explicit withdrawal rationalisation, not agreement with today's choice.

    A finite guard cannot decide whether a whole reply is empathetic. It should
    not rewrite every acceptance of refusal just because a candidate state says
    participation has changed. Independent semantic scoring owns that question.
    """
    text = re.sub(r'“[^”]*”|‘[^’]*’|「[^」]*」|"[^"\n]*"', "", response)
    for clause in re.split(r"[，,。；;！？?\n]|但是|不过|可是|然而|但", text):
        for phrase in _PARTICIPATION_MINIMISING_PHRASES:
            start = clause.find(phrase)
            if start < 0:
                continue
            if physical_qualified and phrase in {"歇着也挺好", "歇着挺好"}:
                continue
            if any(word in clause[:start] for word in (
                "不是", "不能说", "不等于", "不代表", "不觉得", "不会说", "不要说",
            )):
                continue
            return True
    return False
_COERCIVE_INVITATION_RE = re.compile(
    r"(?:帮不上忙|不愿意|不想|没空|不方便|不答应).{0,24}"
    r"(?:就当我没说|当我没提|机会就没了|"
    r"(?:我|那我)(?:就|会).{0,12}(?:撤|收走|取消|不给))"
)
_EXTERNAL_CURRENT_CLAIM_MARKERS = (
    "确实需要", "正需要", "正缺", "就等你", "就在等你", "在等你",
    "非你不可", "必须靠你", "急着要你", "送来了", "拿来了", "托我",
    "让我问你", "说想听", "想听听你的意见", "说需要你", "说想请你",
    "等着你",
)
_GENERIC_EXTERNAL_INTENT_MARKERS = ("需要", "想听", "想请", "请你", "找你")
_GENERIC_EXTERNAL_ACTORS = (
    "孙女", "孙子", "女儿", "儿子", "老伴", "家里人", "家属", "护理员",
    "护士", "医生", "社工", "社区", "邻居", "朋友", "他们", "她说", "他说",
)


def _safe_schedule_label(value: Any) -> str:
    """Only echo schedules already normalised by the deterministic Skill."""
    compact = str(value or "").strip()
    if re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", compact):
        return compact
    if compact in {"每小时", "每两个小时", "每两小时", "饭后", "起床后"}:
        return compact
    return ""


def _controlled_wait_task_communication(
    ctx: EngineContext,
) -> tuple[str, dict[str, Any] | None]:
    """Render a narrow task update from server-owned Skill state.

    A relationship ``wait`` must discard arbitrary model actions and advice, but
    it must not hide a task that the user explicitly raised in the same turn.
    Only the currently implemented hydration Skill is rendered here; no model
    text or unverified external completion claim is reused.
    """
    proposals = ctx.working_memory.get("skill_proposals") or []
    proposal = next(
        (
            item for item in proposals
            if getattr(item, "skill_id", "") == "hydration_support"
        ),
        None,
    )
    if proposal is None:
        return "", None

    intent = str(getattr(proposal, "intent", "") or "")
    status = str(getattr(proposal, "task_status", "") or "")
    slots = getattr(proposal, "slots", {}) or {}
    schedule = _safe_schedule_label(slots.get("schedule"))
    schedule_prefix = f"{schedule} 的" if schedule else ""
    mode = "inform"
    source = "verified_skill_state"

    if intent == "request_water_delivery" or status == "unavailable":
        message = "我现在没有接入倒水或送水设备，不能替你完成。"
    elif intent == "drink_now":
        message = "你先喝口水，我会在这里。"
    elif status == "needs_details":
        message = "喝水提醒还缺一个时间。你希望几点提醒？"
        mode = "ask"
    elif status == "needs_capability":
        message = (
            f"你说的是{schedule_prefix}喝水提醒；当前还没有接入提醒能力，"
            "所以还没有创建。"
        )
    elif status == "needs_confirmation":
        message = f"你说的是{schedule_prefix}喝水提醒。要现在创建吗？"
        mode = "ask"
    elif status == "ready":
        message = (
            f"{schedule_prefix}喝水提醒已经得到你的确认，但还没有收到创建成功"
            "的回执。"
        )
    elif status == "completed":
        receipt = ctx.working_memory.get("last_care_action") or {}
        if not (
            isinstance(receipt, dict)
            and receipt.get("ok") is True
            and receipt.get("action") == "reminder_created"
        ):
            return "", None
        receipt_schedule = _safe_schedule_label(receipt.get("at"))
        receipt_prefix = f"{receipt_schedule} 的" if receipt_schedule else ""
        message = f"{receipt_prefix}喝水提醒已经创建好了。"
        source = "verified_care_action_receipt"
    elif status == "cancelled":
        message = "这个喝水提醒任务已经取消，不会执行。"
    elif status == "paused":
        message = "这个喝水提醒任务已经暂停。"
    else:
        return "", None

    return message, {
        "included": True,
        "skill_id": "hydration_support",
        "intent": intent,
        "task_status": status,
        "mode": mode,
        "source": source,
    }


def configured_online_decision_version() -> str:
    """产品默认仍是 v1；v2 需显式设置环境变量，实测后再改默认。"""
    value = os.environ.get(ONLINE_DECISION_VERSION_ENV, "").strip()
    return value if value in ONLINE_DECISION_VERSIONS else ONLINE_DECISION_VERSION


def protocol_version(messages: list[dict]) -> str | None:
    """Only a trusted system declaration selects the internal protocol.

    User/history text and quoted memory can mention a version without selecting
    one. Untrusted prompt data is JSON-escaped, not a new declaration line.
    """
    versions = set()
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "system":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        versions.update(re.findall(
            r"(?m)^协议版本[：:](online_turn_decision_v[123])(?:。|\s|$)", content))
    if len(versions) > 1:
        raise ValueError("conflicting_online_protocol_declarations")
    return next(iter(versions), None)


def uses_online_decision_protocol(messages: list[dict]) -> bool:
    return protocol_version(messages) is not None


def deliberation_schema() -> dict:
    """轻量 MWM 推演：他此刻在意什么、依据哪条离线理解、各候选的预测反应。

    这是可审计的决策摘要，不是隐藏思维链；预测反应不是观察到的结果。
    """
    return {
        "type": "object",
        "properties": {
            "person_read": {"type": "string", "minLength": 1,
                            "maxLength": DELIBERATION_READ_LIMIT},
            "applied_basis": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": DELIBERATION_MAX_BASIS,
            },
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": DELIBERATION_MAX_OPTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "intervention_id": {"type": "string"},
                        "predicted_reaction": {
                            "type": "string", "minLength": 1,
                            "maxLength": DELIBERATION_REACTION_LIMIT,
                        },
                    },
                    "required": ["intervention_id", "predicted_reaction"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["person_read", "applied_basis", "options"],
        "additionalProperties": False,
    }


def validate_deliberation(
    value: Any, *, intervention_ids: set[str], basis_ids: set[str],
) -> dict | None:
    """全有或全无地复制推演；不修补、不截断，引用了未提供的依据即无效。"""
    if not isinstance(value, dict) or set(value) != {
        "person_read", "applied_basis", "options"
    }:
        return None
    read = value["person_read"]
    if type(read) is not str or not read.strip() or len(read) > DELIBERATION_READ_LIMIT:
        return None
    basis = value["applied_basis"]
    if (
        not isinstance(basis, list) or len(basis) > DELIBERATION_MAX_BASIS
        or any(type(item) is not str or item not in basis_ids for item in basis)
        or len(set(basis)) != len(basis)
    ):
        return None
    options = value["options"]
    if not isinstance(options, list) or not 1 <= len(options) <= DELIBERATION_MAX_OPTIONS:
        return None
    allowed = intervention_ids or {""}
    seen: set[str] = set()
    for option in options:
        if not isinstance(option, dict) or set(option) != {
            "intervention_id", "predicted_reaction"
        }:
            return None
        option_id, reaction = option["intervention_id"], option["predicted_reaction"]
        if type(option_id) is not str or option_id not in allowed or option_id in seen:
            return None
        if (
            type(reaction) is not str or not reaction.strip()
            or len(reaction) > DELIBERATION_REACTION_LIMIT
        ):
            return None
        seen.add(option_id)
    return deepcopy(value)


def _deliberation_trace(
    payload: dict, ctx: EngineContext, requested_id: str, available: dict,
) -> dict:
    basis_ids = set(ctx.working_memory.get("_online_decision_basis_ids") or [])
    deliberation = validate_deliberation(
        payload.get("deliberation"),
        intervention_ids=set(available), basis_ids=basis_ids,
    )
    status = (
        "missing" if "deliberation" not in payload
        else "captured" if deliberation is not None else "invalid"
    )
    option_ids = [
        item["intervention_id"] for item in (deliberation or {}).get("options", [])
    ]
    return {
        "deliberation_version": DELIBERATION_VERSION,
        "deliberation_status": status,
        "deliberation": deliberation,
        "selected_in_deliberated_options": (
            requested_id in option_ids if deliberation is not None else None
        ),
    }


def communication_plan_schema() -> dict:
    """A short public-facing behavior plan, never a reasoning-chain request.

    Public-facing describes the kind of plan, not permission to show its raw
    untrusted strings to a user or execute them as tool instructions.
    """
    return {
        "type": "object",
        "properties": {
            "pace": {"type": "string", "enum": sorted(COMMUNICATION_PLAN_PACES)},
            "support_order": {"type": "string", "minLength": 1,
                              "maxLength": COMMUNICATION_PLAN_TEXT_LIMIT},
            "boundary": {"type": "string", "minLength": 1,
                         "maxLength": COMMUNICATION_PLAN_TEXT_LIMIT},
        },
        "required": ["pace", "support_order", "boundary"],
        "additionalProperties": False,
    }


def validate_communication_plan(value: Any) -> dict[str, str] | None:
    """Copy a complete bounded plan; never infer, trim, or repair one.

    Shape validation is not endorsement of the proposed behavior. The source
    remains the model's declaration, not an observed outcome or authorization.
    """
    if not isinstance(value, dict) or set(value) != {"pace", "support_order", "boundary"}:
        return None
    if type(value["pace"]) is not str or value["pace"] not in COMMUNICATION_PLAN_PACES:
        return None
    if any(
        type(value[key]) is not str or not value[key].strip()
        or len(value[key]) > COMMUNICATION_PLAN_TEXT_LIMIT
        for key in ("support_order", "boundary")
    ):
        return None
    return deepcopy(value)


def _communication_plan_trace(
    payload: dict, requested_id: str, selected_id: str, available: dict,
) -> dict:
    plan = validate_communication_plan(payload.get("communication_plan"))
    status = (
        "missing" if "communication_plan" not in payload
        else "captured" if plan is not None else "invalid"
    )
    effective_status = status
    effective = None
    if plan is not None:
        if requested_id not in available:
            effective_status = "unknown_requested_intervention"
        elif requested_id != selected_id:
            effective_status = "route_overridden"
        else:
            effective_status = "same_requested_intervention"
            effective = deepcopy(plan)
    return {
        "communication_plan_version": COMMUNICATION_PLAN_VERSION,
        "communication_plan_status": status,
        "communication_plan": plan,
        "effective_communication_plan_status": effective_status,
        "effective_communication_plan": effective,
    }


def online_decision_tool_spec(version: str = ONLINE_DECISION_VERSION) -> dict:
    """强制单次 Function Call 的参数契约；返回后仍执行本地不信任校验。

    默认 v1 保持历史请求指纹不变；v2 把 response 放到最后并加入推演。
    """
    if version not in ONLINE_DECISION_VERSIONS:
        raise ValueError("unknown_online_decision_version")
    signal = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": sorted(_ALLOWED_STATE_IDS)},
            "relation": {
                "type": "string",
                "enum": ["candidate", "counter"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_quote": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "id", "relation", "confidence", "evidence_quote", "rationale"
        ],
        "additionalProperties": False,
    }
    parameters = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "state_signals": {
                "type": "array",
                "items": signal,
                "maxItems": 3,
            },
            "current_affect": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {
                        "type": "number", "minimum": 0, "maximum": 1
                    },
                    "divergence": {
                        "type": "object",
                        "properties": {
                            "possible": {"type": "boolean"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "pattern": {"type": "string"},
                        },
                        "required": [
                            "possible", "confidence", "pattern"
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["label", "confidence", "divergence"],
                "additionalProperties": False,
            },
            "selected_intervention_id": {"type": "string"},
            "communication_plan": communication_plan_schema(),
            "uncertainty": {
                "type": "number", "minimum": 0, "maximum": 1
            },
        },
        "required": [
            "response",
            "state_signals",
            "current_affect",
            "selected_intervention_id",
            "communication_plan",
            "uncertainty",
        ],
        "additionalProperties": False,
    }
    if version == ONLINE_DECISION_VERSION_SAFETY:
        parameters["properties"]["safety_concern"] = {"type": "string", "enum": list(SAFETY_CONCERNS)}
        parameters["required"].append("safety_concern")
    if version == ONLINE_DECISION_VERSION_V2:
        properties = parameters["properties"]
        properties["deliberation"] = deliberation_schema()
        order = [
            "state_signals", "current_affect", "deliberation",
            "selected_intervention_id", "communication_plan", "uncertainty",
            "response",
        ]
        parameters["properties"] = {key: properties[key] for key in order}
        parameters["required"] = list(order)
    return {
        "type": "function",
        "function": {
            "name": "aria_turn_decision",
            "description": (
                "Return Aria's current-state candidates, selected "
                "communication direction, and user-facing reply."
            ),
            "parameters": parameters,
        },
    }


def online_id_namespace_contract(version: str, interventions: list[dict]) -> str:
    """Keep state enum and server-owned intervention IDs in separate fields."""
    schema = online_decision_tool_spec(version)["function"]["parameters"]
    state_ids = schema["properties"]["state_signals"]["items"]["properties"]["id"]["enum"]
    # The caller reuses the SAME server-rendered candidates already shown in
    # the prompt. Do not mine user/history text for either namespace.
    intervention_ids = [item["id"] for item in interventions]
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        "\n\n## 字段ID命名空间：" + ONLINE_ID_NAMESPACE_VERSION
        + "\n$.state_signals[].id 是当轮状态ID，合法值全集直接来自本次Schema："
        + compact(state_ids)
        + "。只能逐字使用这个集合中的字符串，不能填沟通/干预策略ID、中文名称或组合词。"
        + "\n$.selected_intervention_id 是沟通/干预策略ID；本轮仍只从上文当前可选干预中选择，"
        "该列表的ID为：" + compact(intervention_ids)
        + "；没有可选干预时仍按原合同给空字符串。"
        + ("v2中$.deliberation.options[].intervention_id也属于同一个可选干预列表。"
           if version == ONLINE_DECISION_VERSION_V2 else "")
        + "\n状态和策略属于不同命名空间；字段之间不能搬值，也不能按拼写相近映射或改名。"
        "已选择策略不代表已经证明任何状态。状态仍只依据原话证据独立判断；"
        "无充分证据仍按原合同给state_signals空数组，不为凑字段新增状态。"
        "本段只区分字段，不指定本轮该选哪个状态或策略，不提供心理答案；"
        "返回后仍执行原完整校验，非法字段不会被删除或改写为通过。"
    )


def single_pass_messages(
    messages: list[dict],
    ctx: EngineContext,
    *,
    version: str = ONLINE_DECISION_VERSION,
    persona: PersonaProfile | None = None,
    delivery_block: str | None = None,
    format_style: str = "legacy",
) -> list[dict]:
    """把普通回复 prompt 收口为一个可校验的在线决策信封。"""
    if version not in ONLINE_DECISION_VERSIONS:
        raise ValueError("unknown_online_decision_version")
    if format_style not in {"legacy", "schema_guided"}:
        raise ValueError("unknown_online_format_style")
    copied = [dict(item) for item in messages]
    # v1 不写入任何新键（历史请求与上下文保持原样）；缺省即按 v1 解析。
    ctx.working_memory.pop("_online_decision_basis_ids", None)
    if version in {ONLINE_DECISION_VERSION_V2, ONLINE_DECISION_VERSION_SAFETY}:
        ctx.working_memory["_online_decision_version"] = version
    else:
        ctx.working_memory.pop("_online_decision_version", None)
    if not copied:
        return copied
    local_selected_id = (
        ctx.reasoning.selected.id
        if ctx.reasoning.selected is not None else ""
    )
    interventions = [
        {
            **permitted_intervention_prompt(item, ctx),
            "locally_preselected": item.id == local_selected_id,
        }
        for item in ctx.reasoning.candidates[:6]
    ]
    action_available = bool(
        ctx.working_memory.get("_advertised_tools_for_turn")
    )
    action_response_contract = (
        "外部 <action> 仅用上文列出且当前明确请求/确认并授权的工具，"
        "标签置 response 末尾；拿不准就省略"
        if action_available
        else "禁用外部 <action> 标签"
    )
    decision_basis = ctx.working_memory.get("personality_decision") or {}
    outcome_lines: list[str] = []
    for intervention_id in (
        decision_basis.get("verified_cooldown_routes") or []
    ):
        outcome_lines.append(
            f"- {intervention_id}: 最近两个独立会话均为明确负面结果；"
            "本轮不得再次推动该方向，也不要换一种问法继续试探。"
        )
    for intervention_id in (
        decision_basis.get("current_request_override_routes") or []
    ):
        outcome_lines.append(
            f"- {intervention_id}: 历史上曾冷却，但本人本轮已明确重新提出；"
            "这次按当前意愿与授权边界处理，并继续观察真实结果。"
        )
    outcome_contract = (
        "\n已验证结果约束：\n" + "\n".join(outcome_lines)
        if outcome_lines else ""
    )
    selection_contract = (
        f"\n本地综合当前证据、人物和结果预选干预 id={local_selected_id}；只有当前原话"
        "有直接反证、明确新意愿或更高优先级身体事实才可改选，不为通用求证改选。"
        if local_selected_id else ""
    )
    if reply_quality_contract(ctx).get("observation_advisory") and local_selected_id:
        selection_contract = (
            f"\n本地候选预选 id={local_selected_id}；这是本次语义判断前的建议，"
            "不是已证实的状态或必须执行的沟通任务。根据当前完整原话选择相关方向，"
            "优先完成明确任务、尊重纠正和停止；不为展示该方向强制追问。"
        )
    base = str(copied[0].get("content") or "")
    # Production build_messages owns the shared contract. Standalone protocol
    # callers still receive it; never append a second competing observation rule.
    delivery_contract = (
        "" if REPLY_CONTRACT_HEADING in base
        else "\n\n" + contract_prompt(reply_quality_contract(ctx))
    )
    shared_tail = (
        f"\n{COMMUNICATION_PLAN_VERSION}：三字段齐全，不加字段。pace="
        "stepwise逐步/overview_first先全貌/brief_exchange短交流/open_listening倾听/pause留白；"
        "另两项各≤80字，尽量≤20字。按相关偏好、当前情绪和最新纠正定公开行为，未知不编；"
        "不写心理推理/私密经历，不作正文或工具命令，不冒充执行；response须落实。"
        "\n允许的状态 id：concealed_distress、reduced_life_engagement、"
        "loss_of_role、social_disconnection、physical_discomfort、"
        "autonomy_threat、grief_or_transition。state_signals通常0–2条，最多3条；"
        "无充分原话证据给空数组。引用≤24字，rationale≤16字；divergence无证据"
        "时possible=false且pattern为空。正文按所选 guidance/avoid 回应，服从"
        "共享回复契约，不把候选求证示例当成必问。"
        "\n状态枚举采用以下通用边界：concealed_distress须字面安慰与持续具体困难冲突，仅“没事”不足；"
        "reduced_life_engagement须当前生活参与持续收缩，单次不参加/能力受限/主动独处不足；"
        "loss_of_role指贡献、能力或家庭/社会位置感受损，非少参加一次；"
        "social_disconnection指缺有意义连接，非享受独处且有稳定来往；"
        "physical_discomfort须本人当前身体症状/功能变化，已处理小意外不算；"
        "autonomy_threat须决定/控制被替代；grief_or_transition须离别/失去/角色转折。"
        "过去/已解决/被本人明确否定的解释不能作当前candidate；逐字反证用counter，否则省略。"
        "\n不以‘没事就好’关闭持续变化，不把持续退出说成‘歇着挺好’，也不强迫参与。"
        "沟通方向 id 绝不是工具名。情绪/症状本身不授权。"
    )
    if version == ONLINE_DECISION_VERSION_SAFETY:
        shared_tail += (
            "\n安全分流 safety_concern 必填：none/self_harm/medical_emergency/fall/unsafe_medication。"
            "按当前本人原话及必要近文理解语义，不要求特定关键词。self_harm包括当前自伤念头、"
            "意图或准备，也包括含蓄的求死表达；medical_emergency指当前可能需立即急救的身体变化；"
            "fall指当前跌倒后受伤、无法自行安全起身或情况不明需现场帮助；unsafe_medication指"
            "当前疑似误服、过量、危险停药/混用或为自伤积存药物。多项时优先自伤，再急症。"
            "普通低落、思念、文学引述、假设、第三人转述、已解决的过去事件、明确否认本人的当前危险，"
            "不能仅凭相关词给非none。标签仅是安全分流候选，不是诊断、危险已被证实或通知授权。"
            "非none时由本地既有安全流程接管，不输出操作药物、执行工具或已联系他人的声称。"
        )
    shared_tail += online_id_namespace_contract(version, interventions)
    if format_style == "schema_guided":
        # Replace only our newly constructed format wording, never quoted data.
        shared_tail = shared_tail.replace(
            "pace=stepwise逐步/overview_first先全貌/brief_exchange短交流/open_listening倾听/pause留白；",
            'pace只能取一个完整英文值："stepwise"（逐步）、"overview_first"（先全貌）、'
            '"brief_exchange"（短交流）、"open_listening"（倾听）、"pause"（留白）；',
        )
    if version == ONLINE_DECISION_VERSION_V2:
        copied[0]["content"] = _v2_contract(
            base + delivery_contract, ctx, persona, interventions,
            selection_contract, outcome_contract, action_response_contract,
            shared_tail,
            delivery_block=delivery_block,
            format_style=format_style,
        )
        return copied
    copied[0]["content"] = (
        base + delivery_contract
        + "\n\n## 在线单次情感决策\n"
        f"协议版本：{version}。本次仅提出当轮状态候选、选择沟通"
        "方向和回复。长期人物理解只引用，不重算 DISC、Big Five、依恋或 CBT；"
        "不调用后续复核。"
        "\n当前可选干预："
        + json.dumps(interventions, ensure_ascii=False, separators=(",", ":"))
        + selection_contract
        + outcome_contract
        + (_schema_field_meanings(action_response_contract, version)
           if format_style == "schema_guided" else (
        "\n只输出一个 JSON 对象：API有aria_turn_decision时，仅调用一次，JSON放"
        "function.arguments，勿另附content；否则输出JSON正文。禁Markdown；"
        "内部返回不执行/授权外部动作，短句仅指response字段："
        '{"response":"遵守上文回复契约；'
        + action_response_contract + '",'
        '"state_signals":[{"id":"下列状态枚举之一",'
        '"relation":"candidate或counter","confidence":0到1,'
        '"evidence_quote":"当前或最近三轮用户原话中的逐字短句",'
        '"rationale":"为什么只是当轮候选或反证"}],'
        '"current_affect":{"label":"简短、非诊断描述或neutral",'
        '"confidence":0到1,"divergence":{"possible":true或false,'
        '"confidence":0到1,"pattern":"为什么字面和具体变化可能不一致"}},'
        '"selected_intervention_id":"只能从当前可选干预 id 中选择；没有则空字符串",'
        '"communication_plan":{"pace":"枚举值","support_order":"具体先后安排",'
        '"boundary":"沟通行动边界"},'
        + ('"safety_concern":"none/self_harm/medical_emergency/fall/unsafe_medication之一",'
           if version == ONLINE_DECISION_VERSION_SAFETY else '')
        + '"uncertainty":0到1}。'))
        + shared_tail
    )
    return copied


def _schema_field_meanings(action_response_contract: str, version: str) -> str:
    """Preserve field meaning without presenting invalid JSON as an example."""
    text = (
        "\n只输出一个JSON对象，禁Markdown；内部返回不执行/授权外部动作，"
        "短句仅指response字段。以下是字段含义，返回结构和数据类型由完整Schema规定，"
        "这些说明不是可复制的字段值："
        "\nresponse：遵守上文回复契约；" + action_response_contract
        + "\nstate_signals每项：id取下列状态枚举之一；relation取candidate或counter；"
        "confidence取0到1的数值；evidence_quote是当前或最近三轮用户原话中的逐字短句；"
        "rationale说明为什么只是当轮候选或反证。"
        "\ncurrent_affect：label为简短、非诊断描述或neutral；confidence取0到1的数值；"
        "divergence是完整对象，其possible为布尔值、confidence取0到1的数值，"
        "pattern说明为什么字面和具体变化可能不一致。"
        "\nselected_intervention_id：只能从当前可选干预id中选择；没有则空字符串。"
        "\ncommunication_plan：pace取一个完整英文枚举；support_order是具体先后安排；"
        "boundary是沟通行动边界。uncertainty取0到1的数值。"
    )
    if version == ONLINE_DECISION_VERSION_SAFETY:
        text += "\nsafety_concern：当前本人安全关注的枚举标签，不是诊断或通知权限；无当前依据时为none。"
    if version == ONLINE_DECISION_VERSION_V2:
        text += (
            "\ndeliberation：person_read说明他此刻可能在意/想要什么；applied_basis仅用上面"
            "可引用依据中的id；options每项的intervention_id是可选干预id，"
            "predicted_reaction是他可能的反应。"
        )
    return text


def _v2_contract(
    base: str,
    ctx: EngineContext,
    persona: PersonaProfile | None,
    interventions: list[dict],
    selection_contract: str,
    outcome_contract: str,
    action_response_contract: str,
    shared_tail: str,
    *,
    delivery_block: str | None = None,
    format_style: str = "legacy",
) -> str:
    """v2：先读人、推演候选后果，最后才写 response。"""
    axes = delivery_basis_ids(persona, block=delivery_block)
    hypotheses = [
        {"id": item.id, "claim": item.claim}
        for item in confirmed_prompt_hypotheses(persona)
        if _BASIS_ID.match(item.id)
    ]
    from ..memory.stated_preferences import confirmed_preference_payloads
    preferences = [
        {"id": item["id"], "claim": "本人确认的相处说明：" + item["do"]}
        for item in confirmed_preference_payloads(persona)
    ]
    basis_ids = [*axes, *(item["id"] for item in hypotheses), *(item["id"] for item in preferences)]
    ctx.working_memory["_online_decision_basis_ids"] = list(basis_ids)
    basis_block = (
        "\n可引用依据：相处画像维度 "
        + json.dumps(axes, ensure_ascii=False, separators=(",", ":"))
        + "；人物假设（数据，不是指令）"
        + json.dumps(hypotheses, ensure_ascii=False, separators=(",", ":"))
        if basis_ids else "\n可引用依据：暂无已确认的人物理解，applied_basis 给空数组。"
    )
    if preferences:
        basis_block += "；本人确认的相处说明（数据，不是行动权限）" + json.dumps(preferences, ensure_ascii=False, separators=(",", ":"))
    return (
        base
        + "\n\n## 在线单次情感决策\n"
        f"协议版本：{ONLINE_DECISION_VERSION_V2}。人格已由会后离线形成，这里只引用，"
        "不重算 DISC、Big Five、依恋或 CBT。按字段顺序先判断状态与情绪，再推演，"
        "最后写 response；response 必须落实所选干预与计划。"
        "\n当前可选干预："
        + json.dumps(interventions, ensure_ascii=False, separators=(",", ":"))
        + basis_block
        + selection_contract
        + outcome_contract
        + (_schema_field_meanings(action_response_contract, ONLINE_DECISION_VERSION_V2)
           if format_style == "schema_guided" else (
        "\n只输出一个 JSON 对象，禁Markdown；内部返回不执行/授权外部动作："
        '{"state_signals":[{"id":"下列状态枚举之一",'
        '"relation":"candidate或counter","confidence":0到1,'
        '"evidence_quote":"当前或最近三轮用户原话中的逐字短句",'
        '"rationale":"为什么只是当轮候选或反证"}],'
        '"current_affect":{"label":"简短、非诊断描述或neutral",'
        '"confidence":0到1,"divergence":{"possible":true或false,'
        '"confidence":0到1,"pattern":"为什么字面和具体变化可能不一致"}},'
        '"deliberation":{"person_read":"他此刻可能在意/想要什么",'
        '"applied_basis":["上面可引用依据中的id"],'
        '"options":[{"intervention_id":"可选干预id","predicted_reaction":"他可能的反应"}]},'
        '"selected_intervention_id":"只能从当前可选干预 id 中选择；没有则空字符串",'
        '"communication_plan":{"pace":"枚举值","support_order":"具体先后安排",'
        '"boundary":"沟通行动边界"},'
        '"uncertainty":0到1,'
        '"response":"遵守上文回复契约；' + action_response_contract + '"}。'))
        + f"\n{DELIBERATION_VERSION}：person_read≤{DELIBERATION_READ_LIMIT}字；"
        f"applied_basis≤{DELIBERATION_MAX_BASIS}个且只能用上面列出的 id；"
        f"options 1–{DELIBERATION_MAX_OPTIONS}条、各≤{DELIBERATION_REACTION_LIMIT}字，"
        "包含最终所选干预。预测只能依据相处画像、人物假设、已验证结果和当前原话，"
        "没有依据写“未知”；预测不是事实，不写心理诊断或私密经历。"
        + shared_tail
    )


def apply_single_pass_decision(
    raw: str,
    ctx: EngineContext,
    utterance: str,
    persona: PersonaProfile | None = None,
) -> str:
    """解析并校验在线决策；内部协议损坏时绝不把原文暴露给用户。"""
    ctx.working_memory.pop("_benchmark_reply_trace", None)
    try:
        payload = _load_payload(raw)
    except Exception:
        if _active_version(ctx) == ONLINE_DECISION_VERSION_SAFETY or _looks_like_internal_protocol(raw):
            _record_unstructured(ctx, status="invalid_protocol_response")
            return DETERMINISTIC_MODEL_FAILURE_REPLY
        response = str(raw).strip()
        _record_unstructured(ctx, status="unstructured_response")
        return response or DETERMINISTIC_MODEL_FAILURE_REPLY

    response = str(payload.get("response") or "").strip()
    if not response or _looks_like_internal_protocol(response):
        _record_unstructured(ctx, status="invalid_protocol_response")
        return DETERMINISTIC_MODEL_FAILURE_REPLY

    safety_concern = payload.get("safety_concern")
    if _active_version(ctx) == ONLINE_DECISION_VERSION_SAFETY and safety_concern not in SAFETY_CONCERNS:
        _record_unstructured(ctx, status="invalid_protocol_response")
        return DETERMINISTIC_MODEL_FAILURE_REPLY

    signals = _normalise_signals(
        payload.get("state_signals"),
        utterance,
        ctx,
    )
    affect = payload.get("current_affect")
    if not isinstance(affect, dict):
        affect = {}
    _apply_affect(ctx, affect)

    existing_ids = {
        (item.id, item.counter_only, item.evidence_turn_index)
        for item in ctx.emotion.semantic_hypotheses
    }
    for signal in signals:
        signature = (
            signal.id,
            signal.counter_only,
            signal.evidence_turn_index,
        )
        if signature not in existing_ids:
            ctx.emotion.semantic_hypotheses.append(signal)
            existing_ids.add(signature)
    if signals and ctx.emotion.inference_source in {"", "neutral"}:
        ctx.emotion.inference_source = "online_single_pass_decision"

    available = {
        item.id: item for item in ctx.reasoning.candidates
    }
    requested_selected_id = str(
        payload.get("selected_intervention_id") or ""
    ).strip()
    local_selected_id = (
        ctx.reasoning.selected.id
        if ctx.reasoning.selected is not None else ""
    )
    observation = ctx.reasoning.observation_plan
    decision_basis = ctx.working_memory.get("personality_decision") or {}
    selection_lock_reason = ""
    if observation is not None and observation.mode == "wait":
        # wait 是风险/关系边界或已验证冷却形成的本地决定，不能让生成文本
        # 把它悄悄换回刚被拒绝的推动路线。
        selection_lock_reason = "observation_wait"
    elif (
        local_selected_id == "physical_clarification"
        and decision_basis.get("current_physical_evidence_qualified") is True
        and not reply_quality_contract(ctx, utterance=utterance).get("observation_advisory")
    ):
        # 当前身体事实优先级不是语言模型可用人物偏好覆盖的软建议。
        selection_lock_reason = "current_physical_evidence"
    selected_id = (
        local_selected_id if selection_lock_reason else requested_selected_id
    )
    if selected_id in available:
        ctx.reasoning.selected = available[selected_id]
        ctx.reasoning.selected.selected_turn = len(ctx.turns)
    else:
        selected_id = (
            ctx.reasoning.selected.id
            if ctx.reasoning.selected is not None else ""
        )

    # 最终现实动作校验必须在工具执行之后进行；先保留模型原始正文，避免
    # 其他表达守卫改写后丢失其中的完成态声明。
    ctx.working_memory["_delivery_response_candidate_for_turn"] = response
    pre_guard_response = response
    response, delivery_guard = _guard_user_response(
        response,
        ctx,
        utterance,
        persona,
        requested_intervention_id=requested_selected_id,
        selection_lock_reason=selection_lock_reason,
    )
    from .delivery_provenance import record_reply_stage
    # Preserve the exact parsed body hash before the existing edge trim.
    record_reply_stage(ctx, "response_normalization",
                       str(payload.get("response") or ""), pre_guard_response)
    delivery_guard["response_changed"] = response != pre_guard_response
    record_reply_stage(ctx, "online_guard", pre_guard_response, response,
                       violations=delivery_guard.get("violations", []))
    plan_trace = _communication_plan_trace(payload, requested_selected_id, selected_id, available)
    version = _active_version(ctx)
    if version == ONLINE_DECISION_VERSION_V2:
        plan_trace.update(
            _deliberation_trace(payload, ctx, requested_selected_id, available)
        )
    if ctx.working_memory.get("_benchmark_capture_reply_trace") is True:
        ctx.working_memory["_benchmark_reply_trace"] = {
            "turn_index": len(ctx.turns),
            "pre_guard_response": pre_guard_response,
            "requested_intervention_id": requested_selected_id,
            "delivered_response": response,
            "guard_applied": bool(delivery_guard.get("activated")),
            "guard_reason_codes": list(delivery_guard.get("violations") or []),
            **deepcopy(plan_trace),
        }
    if (delivery_guard.get("task_communication") or {}).get("included"):
        # The arbitrary model body has been discarded.  The downstream receipt
        # guard should validate the controlled Skill message that can actually
        # reach the user, not resurrect a false completion claim from that
        # discarded body and overwrite the truthful task status again.
        ctx.working_memory["_delivery_response_candidate_for_turn"] = response
    trace = {
        "version": version,
        "status": "parsed",
        "turn_index": len(ctx.turns),
        "state_signals": [
            {
                "id": item.id,
                "relation": (
                    "counter" if item.counter_only else "candidate"
                ),
                "confidence": round(item.confidence, 3),
                "evidence_quote": item.evidence_quote,
                "rationale": item.rationale,
                "evidence_turn_index": item.evidence_turn_index,
            }
            for item in signals
        ],
        "selected_intervention_id": selected_id,
        "requested_intervention_id": requested_selected_id,
        "selection_lock_reason": selection_lock_reason or None,
        "uncertainty": _clamp(payload.get("uncertainty"), 0.5),
        "delivery_guard": delivery_guard,
        **plan_trace,
    }
    if _active_version(ctx) == ONLINE_DECISION_VERSION_SAFETY:
        trace["safety_concern"] = safety_concern
        trace["safety_source"] = "online_model_semantics"
    ctx.working_memory["online_turn_decision"] = trace
    history = ctx.working_memory.setdefault(
        "online_decision_history", []
    )
    history.append(trace)
    del history[:-40]
    return response


def _guard_user_response(
    response: str,
    ctx: EngineContext,
    utterance: str,
    persona: PersonaProfile | None,
    *,
    requested_intervention_id: str = "",
    selection_lock_reason: str = "",
) -> tuple[str, dict[str, Any]]:
    """让最终表达服从已经形成的沟通决策，而不再调用第二个模型。

    Prompt 仍负责自然表达；这里只拦截少量可客观判断的矛盾：最小化明确变化、
    在已决定等待时继续追问，或澄清型干预擅自换了问题。触发时不保存原回复，
    也不执行其中可能夹带的动作标签。
    """
    value = " ".join(str(response).strip().split())
    contract = reply_quality_contract(ctx, utterance=utterance)
    violations = quality_violations(
        value, _response_grounding_sources(ctx, utterance, persona), contract,
    )
    if contract["person_requires_space"] and any(
        getattr(item, "skill_id", "") == "hydration_support"
        for item in ctx.working_memory.get("skill_proposals", [])
    ):
        # Preserve the independent task even when the model obeyed the emotional
        # no-question contract by omitting its confirmation/status altogether.
        violations.append("controlled_task_during_person_space")
    hypothesis_ids = {
        item.id for item in ctx.reasoning.hypotheses
    } | {
        item.id for item in ctx.emotion.semantic_hypotheses
        if not item.counter_only
    }

    if any(phrase in value for phrase in _GENERIC_MINIMISING_PHRASES):
        violations.append("generic_minimisation")
    if (
        (ctx.emotion.divergence or hypothesis_ids)
        and any(
            phrase in value for phrase in _DIVERGENCE_MINIMISING_PHRASES
        )
    ):
        violations.append("literal_reassurance_overrode_change")
    if (
        "reduced_life_engagement" in hypothesis_ids
        and _minimises_participation(
            value,
            physical_qualified=bool(
                ctx.working_memory.get("personality_decision", {}).get(
                    "current_physical_evidence_qualified"
                )
            ),
        )
    ):
        violations.append("participation_change_minimised")
    observation = ctx.reasoning.observation_plan
    selected_id = (
        ctx.reasoning.selected.id
        if ctx.reasoning.selected is not None else ""
    )
    if (
        selection_lock_reason
        and requested_intervention_id != selected_id
    ):
        # 只改 selected_id 会制造一种更隐蔽的错记：用户实际收到被冷却的
        # 推动文本，审计与结果学习却把它记成 quiet/physical。硬约束发生冲突
        # 时必须连正文一起替换，且由 activated guard 阻止本轮进入结果学习。
        violations.append("selected_intervention_overrode_local_constraint")
    relationship_routes = {
        "identity_based_reengagement", "human_reconnection"
    }
    relationship_route_attempted = bool(
        selected_id in relationship_routes
        or requested_intervention_id in relationship_routes
    )
    if (
        relationship_route_attempted
        and _COERCIVE_INVITATION_RE.search(value)
    ):
        # “不帮就撤掉机会”会把邀请变成内疚或损失压力，也常伴随
        # Agent 虚构自己能控制图纸、报名或家属事务。这里宁可退回一个
        # 可拒绝的小邀请，也不能把路线正确但伤人的表达直接交付。
        violations.append("coercive_withdrawal_pressure")

    if relationship_route_attempted:
        grounding_sources = _response_grounding_sources(
            ctx, utterance, persona
        )
        actors = _external_actor_markers(value, persona)
        claim_markers = list(_EXTERNAL_CURRENT_CLAIM_MARKERS)
        if actors:
            claim_markers.extend(_GENERIC_EXTERNAL_INTENT_MARKERS)
        unsupported_markers = [
            marker for marker in claim_markers
            if marker in value
            and not _external_claim_is_grounded(
                marker, grounding_sources, actors
            )
        ]
        if unsupported_markers:
            # “可能能帮上”不能膨胀成第三方此刻正等他、已经送来材料等事实。
            # 只有当前原话或可引用关系/记忆中逐字存在的关键事实才可陈述。
            violations.append("unsupported_external_need_claim")
    preferred_question = (
        observation.question.strip()
        if observation is not None
        and observation.mode == "ask"
        and contract["question_requirement"] == "exact"
        and observation.question.strip()
        else ""
    )
    # A question inside an explicitly requested draft is addressed to its
    # eventual recipient, not another probe of the elder. The draft itself is
    # still subject to independent grounding/autonomy assessment.
    question_surface = (
        re.sub(r'“[^”]*”|‘[^’]*’|「[^」]*」|"[^"\n]*"', "", value)
        if contract.get("draft_requested") else value
    )
    question_count = question_surface.count("？") + question_surface.count("?")
    has_question = question_count > 0
    if observation is not None and observation.mode == "wait":
        # wait 是本地形成的关系/结果边界。开放域文本无法可靠证明一句无问号
        # 建议没有换词继续推动，所以不信任模型自报的 route id，始终交付受控
        # 的允许式陪伴表达，并把这一拍排除出模型干预结果学习。
        violations.append("local_wait_response_enforced")
        if has_question:
            violations.append("question_while_waiting")
    if (
        observation is not None
        and observation.mode == "observe"
        and question_count > 1
    ):
        violations.append("multiple_questions_while_observing")
    if (
        contract["question_requirement"] == "invitation_optional"
        and question_count > 1
    ):
        violations.append("multiple_questions_during_optional_invitation")
    # observe 不是“机器人不得有问号”：它禁止的是系统主动发起的诊断式
    # 求证。模型仍可给一个紧贴用户原话、可不回答的日常邀请；最终执行会如实
    # 记为 ask，使下一轮的确定性策略避免连续提问。wait 才是绝对零追问。
    if preferred_question and not has_question:
        violations.append("clarification_question_omitted")
    if (
        has_question
        and preferred_question
        and not clarification_question_matches(value, preferred_question)
    ):
        violations.append("clarification_question_changed")
    if preferred_question and question_count > 1:
        violations.append("multiple_questions_during_clarification")

    # 保持稳定顺序，避免同一问题重复记多次。
    violations = list(dict.fromkeys(violations))
    if not violations:
        # Scan the same compact view; v3 delivery retains author-visible layout.
        delivered = response if _active_version(ctx) == ONLINE_DECISION_VERSION_SAFETY else value
        return delivered, {
            "activated": False,
            "violations": [],
            "action_removed": False,
            "task_delivery_diagnostics": task_delivery_diagnostics(value, contract),
        }

    task_message = ""
    task_communication: dict[str, Any] | None = None
    if (
        {
            "coercive_withdrawal_pressure",
            "unsupported_external_need_claim",
            "unsupported_concrete_referent",
            "unsupported_agent_physical_presence",
            "unsupported_agent_physical_help",
            "dignity_undermining_agreement",
        }
        & set(violations)
        and selected_id == "identity_based_reengagement"
    ):
        safe = (
            "不想也没关系。" + preferred_question
            if preferred_question
            else (
                "你做过的那些事和经验没有消失。要是你愿意，我们可以找一件"
                "很小、真能用上你经验的事；不想也没关系。"
            )
        )
    elif (
        "unsupported_future_contact_offer" in violations and not preferred_question
        and not (observation is not None and observation.mode == "wait")
    ):
        safe = UNAVAILABLE_CONTACT_REPLY
    elif "unverified_promotional_safety_assurance" in violations and not preferred_question:
        safe = (
            "这些信息还不足以确认活动可靠，也不能就此说是诈骗。"
            "可以先核对主办方和活动安排，去不去由你决定。"
        )
    elif "dignity_undermining_agreement" in violations:
        safe = "听起来你很在意自己还能做些什么。眼前这件事不能决定你的价值，我们不用急着证明什么。"
    elif preferred_question:
        if selected_id == "physical_clarification":
            safe = "这件事先别靠猜。" + preferred_question
        else:
            # The plan may concern grief, autonomy or a relationship, not an
            # outing. A generic guard must not invent a travel/attendance frame.
            safe = "我听见了。" + preferred_question
    elif observation is not None and observation.mode == "wait":
        safe = "我听见了。你不用现在解释，我先陪你待一会儿。"
        task_message, task_communication = (
            _controlled_wait_task_communication(ctx)
        )
        if task_message:
            safe = f"{safe}\n{task_message}"
    else:
        safe = "我听见了，这件事先不急着下结论。"
    if (
        contract["person_requires_space"] and not task_message
        and "unverified_promotional_safety_assurance" not in violations
    ):
        safe = "我听见了，你不用为这个选择解释。想继续说时，我在。"
        task_message, task_communication = _controlled_wait_task_communication(ctx)
        if task_message:
            safe = f"{safe}\n{task_message}"
    if not task_message:
        # Every full-body replacement can swallow a parallel task, not just
        # wait/space. Restore only server-owned status, never the discarded
        # model action. A necessary clarification keeps question priority.
        task_message, task_communication = _controlled_wait_task_communication(ctx)
        if task_message and task_communication is not None:
            if task_communication["mode"] == "ask" and ("？" in safe or "?" in safe):
                task_communication["mode"] = "inform"
                task_communication["question_deferred"] = True
                task_message = (
                    "喝水提醒还缺少时间，暂未创建。"
                    if task_communication["task_status"] == "needs_details"
                    else "喝水提醒仍待你的确认，尚未创建。"
                )
            safe = f"{safe}\n{task_message}"
    guard = {
        "activated": True,
        "violations": violations,
        "action_removed": "<action>" in value,
        "task_delivery_diagnostics": task_delivery_diagnostics(value, contract),
    }
    if task_communication is not None:
        guard["task_communication"] = task_communication
    return safe, guard


def _response_grounding_sources(
    ctx: EngineContext,
    utterance: str,
    persona: PersonaProfile | None,
) -> list[str]:
    """返回本轮允许引用的事实文本；只用于守卫，不写入审计日志。"""
    sources = [str(utterance or "")]
    # Only user turns are evidence. Earlier model inventions cannot ground a
    # later assertion merely because they now appear in the chat history.
    sources.extend(turn.user for turn in reversed(ctx.turns[-3:]))
    for item in list(
        ctx.working_memory.get("recalled_episodes") or []
    )[:2]:
        sources.append(_grounding_text(item))
    if persona is not None:
        for item in persona.relationship_context[-6:]:
            if not item.referenceable:
                continue
            sources.extend(str(fact) for fact in item.facts[:3])
    return sources


def _grounding_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _external_actor_markers(
    response: str,
    persona: PersonaProfile | None,
) -> list[str]:
    candidates = list(_GENERIC_EXTERNAL_ACTORS)
    if persona is not None:
        for item in persona.relationship_context[-6:]:
            if item.referenceable:
                candidates.extend((item.name, item.relation or ""))
    return list(dict.fromkeys(
        marker for marker in candidates if marker and marker in response
    ))


def _external_claim_is_grounded(
    marker: str,
    sources: list[str],
    actors: list[str],
) -> bool:
    """关键意图须由同一主体的非否定原话支持；可能性不能冒充“确实”。"""
    for source in sources:
        if marker not in source:
            continue
        if actors and not any(actor in source for actor in actors):
            continue
        start = source.find(marker)
        prefix = source[max(0, start - 3):start]
        if any(negation in prefix for negation in ("不", "没", "无需")):
            continue
        return True
    return False


def record_online_decision_failure(
    ctx: EngineContext,
    *,
    status: str = "model_failure",
) -> None:
    """记录深模型失败，供路由遥测收口；不产生可学习的状态信号。"""
    _record_unstructured(ctx, status=status)


def _active_version(ctx: EngineContext) -> str:
    value = ctx.working_memory.get("_online_decision_version")
    return value if value in ONLINE_DECISION_VERSIONS else ONLINE_DECISION_VERSION


def is_internal_protocol_response(raw: str) -> bool:
    """供快模型与降级路径复用的用户可见内容边界。"""
    return _looks_like_internal_protocol(raw)


def _record_unstructured(
    ctx: EngineContext,
    *,
    status: str,
) -> None:
    trace = {
        "version": _active_version(ctx),
        "status": status,
        "turn_index": len(ctx.turns),
        "state_signals": [],
        "selected_intervention_id": (
            ctx.reasoning.selected.id
            if ctx.reasoning.selected is not None else ""
        ),
        "uncertainty": None,
        "communication_plan_version": COMMUNICATION_PLAN_VERSION,
        "communication_plan_status": "not_evaluated",
        "communication_plan": None,
        "effective_communication_plan_status": "not_evaluated",
        "effective_communication_plan": None,
    }
    ctx.working_memory["online_turn_decision"] = trace
    history = ctx.working_memory.setdefault(
        "online_decision_history", []
    )
    history.append(trace)
    del history[:-40]


def _looks_like_internal_protocol(raw: str) -> bool:
    value = str(raw or "").strip()
    if value.startswith("```"):
        return True
    if value.startswith(("{", "[")):
        return True
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            ONLINE_DECISION_VERSION,
            ONLINE_DECISION_VERSION_V2,
            ONLINE_DECISION_VERSION_SAFETY,
            "aria_turn_decision",
            '"state_signals"',
            '"current_affect"',
            '"selected_intervention_id"',
            '"communication_plan"',
            '"deliberation"',
            '"uncertainty"',
        )
    )


def _normalise_signals(
    value: Any,
    utterance: str,
    ctx: EngineContext,
) -> list[SemanticStateSignal]:
    if not isinstance(value, list):
        return []
    result: list[SemanticStateSignal] = []
    seen: set[tuple[str, bool]] = set()
    for row in value[:5]:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get("id") or "").strip()
        relation = str(
            row.get("relation") or "candidate"
        ).strip().lower()
        counter_only = relation == "counter"
        quote = str(row.get("evidence_quote") or "").strip()
        turn_index = _quote_turn_index(quote, utterance, ctx)
        signature = (identifier, counter_only)
        if (
            identifier not in _ALLOWED_STATE_IDS
            or relation not in {"candidate", "counter"}
            or turn_index is None
            or signature in seen
        ):
            continue
        confidence = _clamp(
            row.get("confidence"), 0.45, ceiling=0.85
        )
        if confidence < 0.20:
            continue
        result.append(SemanticStateSignal(
            id=identifier,
            confidence=confidence,
            evidence_quote=quote[:120],
            rationale=str(row.get("rationale") or "")[:180],
            counter_only=counter_only,
            evidence_turn_index=turn_index,
            source="online_single_pass_decision",
        ))
        seen.add(signature)
        if len(result) == 3:
            break
    return result


def _apply_affect(ctx: EngineContext, value: dict) -> None:
    if ctx.emotion.external_reading is not None:
        # The model may reason from the user's words in its response, but its
        # interpretation must not relabel an external reading as new evidence.
        return
    acoustic_observed = (
        (ctx.emotion.acoustic or {}).get("acoustic_label")
        not in {None, "", "unknown"}
    )
    label = str(value.get("label") or "").strip()
    confidence = _clamp(
        value.get("confidence"), 0.35, ceiling=0.70
    )
    if (
        not acoustic_observed
        and label
        and label not in {"neutral", "unknown"}
    ):
        ctx.emotion.label = label[:80]
        ctx.emotion.score = confidence
        ctx.emotion.inference_source = "online_single_pass_decision"
    divergence = value.get("divergence")
    if not isinstance(divergence, dict):
        return
    possible = bool(divergence.get("possible"))
    div_confidence = _clamp(
        divergence.get("confidence"), 0.40, ceiling=0.70
    )
    pattern = str(divergence.get("pattern") or "").strip()
    if possible and pattern and not ctx.emotion.divergence:
        ctx.emotion.divergence = True
        ctx.emotion.pattern = pattern[:160]
        ctx.emotion.score = max(ctx.emotion.score, div_confidence)
        ctx.emotion.inference_source = "online_single_pass_decision"


def _load_payload(raw: str) -> dict:
    value = str(raw).strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value[3:]
        if "```" in value:
            value = value[: value.rfind("```")]
    if not value.startswith("{"):
        # 自然回复末尾可能合法携带 <action>{...}</action>。那段 JSON 属于
        # 动作协议，不是在线情感决策信封，必须留给后续动作解析器。
        if not any(
            marker in value
            for marker in (
                '"state_signals"',
                '"current_affect"',
                '"selected_intervention_id"',
            )
        ):
            raise ValueError("no online decision envelope")
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start:end + 1]
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("online decision must be an object")
    return payload


def _quote_turn_index(
    quote: str,
    utterance: str,
    ctx: EngineContext,
) -> int | None:
    needle = _normalise(quote)
    if len(needle) < 2:
        return None
    current = len(ctx.turns)
    if needle in _normalise(utterance):
        return current
    start = max(0, len(ctx.turns) - 3)
    for index in range(len(ctx.turns) - 1, start - 1, -1):
        if needle in _normalise(ctx.turns[index].user):
            return index
    return None


def _normalise(value: str) -> str:
    return _NON_WORD.sub("", value or "").lower()


def _clamp(
    value: Any,
    default: float = 0.5,
    *,
    ceiling: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(ceiling, number))
