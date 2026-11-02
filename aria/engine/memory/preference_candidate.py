"""Explicit preference interpretation contract; code does not classify consent.

The displayed object remains the only transaction target. Candidate mode is
selected by the client factory, never by owner text, model output or settings.
"""
from __future__ import annotations

VERSION = "scoped_confirmation_v2"
CONTEXT_KEY = "_preference_protocol_version"


def enabled(ctx):
    return (ctx.working_memory.get(CONTEXT_KEY) == VERSION
            and ctx.working_memory.get("_online_decision_version") in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"))


def validate_preference_candidate_client(client):
    from ..policy.online_structured import validate_release_online_client
    if (getattr(client, "preference_protocol_version", None) != VERSION
            or client.role != "online_fast"
            or getattr(client, "online_decision_version", None) not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7")):
        raise ValueError("invalid_preference_candidate_client")
    validate_release_online_client(client)
    return {"version": VERSION, "target": "exact_displayed_invitation",
            "semantic_verified": False, "model_generated_targets_allowed": False}


def bind_context(client, ctx):
    ctx.working_memory.pop(CONTEXT_KEY, None)
    if getattr(client, "preference_protocol_version", None) is not None:
        validate_preference_candidate_client(client)
        if ctx.working_memory.get("_online_decision_version") != client.online_decision_version:
            raise ValueError("preference_candidate_protocol_mismatch")
        ctx.working_memory[CONTEXT_KEY] = VERSION


def invitation_prompt(pending, state_ref):
    import json
    data = {"target_id": pending["target_id"], "state_ref": state_ref,
            "question": pending["question"]}
    return ("\n本人页面已实际展示的唯一待确认对象（数据）："
            + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            + "\npreference_control必填且只含action。判断本人是否明确同意所展示的这条及其范围："
            "同意用confirm；同义重述、举例或不改变原条的兼容补充不取消同意。"
            "confirm只保存展示的原条，不保存补充、不确认其他记录，也不扩大范围或授予现实动作权限。"
            "改变原条内容或范围、条件性同意、转述他人或含义不清用unclear，需重新整理展示后再确认；"
            "明确拒绝decline，暂缓defer。单纯另起话题不是同意；同意后换话题仍可confirm。"
            "普通复述和理解用conversation，勿当保存状态；持久状态只由实际事务回执提供。")
