"""Explicit v6 candidate: model semantics, code-owned source/receipt checks.

This is not a semantic verifier. A false 'conversation' label can still leak a
false claim; tests retain that failure mode. No word lists or second model call
are used to infer what a sentence means. Historical v5 remains unchanged.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

VERSION = "online_turn_decision_v6_semantic"
V7 = "online_turn_decision_v7"
VERSIONS = (VERSION, V7)
KINDS = ("conversation", "preference_state", "assistant_action", "third_party_fact")
SEGMENT_INSTRUCTION = (
    "\n每轮response_segments必填，逐句text直接连接须逐字等于response，保留空白。"
    "每句kind只能选conversation（普通交流、当下倾听/理解、建议、否定/假设）、"
    "preference_state（相处说明的保存、启用、拒绝、删除或撤回状态，含范围限制）、"
    "assistant_action（本助手已经完成或承诺执行外部动作）、"
    "third_party_fact（他人已做、收到、回复等现实事件）。混合句拆开，不能把状态或动作藏进普通交流。"
    "当下听懂不等于持久记忆，不用保存/记住的成功措辞作礼貌接话；程序独立交付真实偏好状态。"
    "第三方事实的source_quote须逐字引用当前或最近可引用的本人原话，保留否定和归属；"
    "其他kind的source_quote为空。assistant_action的action只能notification/reminder/unsupported，"
    "target为回执中的完整目标；其他kind用action=none及空target。"
    "preference_state的state_ref仅选本次提供的当前记录引用，无可引用记录时为空；其余kind的state_ref为空。"
    "代码按引用交付当前状态，不采用你的自由状态措辞；当前已有记录不等于本轮新保存。"
    "通知成功不证明对方收到、已读、回应或行动；没有成功回执不声称完成，不承诺无能力动作。"
    "所有分类只是本次语义自检，不授予任何权限。"
)


def segment_instruction(ctx):
    from ..memory.preference_candidate import enabled
    if ctx.working_memory.get("_online_decision_version") == V7:
        return (
            "\n正文只写在 response_segments 里，按顺序逐字拼接各项text得到完整正文，保留空白；不要另写response。"
            "只有response_segments和safety_concern必填；其他字段仅在能完整提供时填写。"
            "每段仅text必填，kind缺省conversation，source_quote/action/target/state_ref缺省空字符串。"
            "kind可选conversation（普通交流、当下理解、建议、否定或假设，不含操作完成或他人现实事件的断言）、"
            "preference_state（相处说明的保存、启用、拒绝、删除或撤回状态）、"
            "assistant_action（本助手已完成或承诺外部动作）、third_party_fact（他人的现实事件）、"
            "safety_core（与本轮已交付保护内容重复的整段）。不同性质的句子分段，不把动作或状态藏进普通交流。"
            "复述本人提到的他人现实事件仍是third_party_fact，与陪伴或建议分段；仅否定或假设不算事件已发生。"
            "新增求助号码、陪伴和其他补充须单独写conversation，不归入重复保护段。"
            "third_party_fact的source_quote逐字引用可引用的本人原话；其他kind用空引文。"
            "assistant_action的action只能notification/reminder/unsupported，target取回执完整目标；其他kind留空。"
            "preference_state的state_ref只用本轮提供的记录引用，其余留空；代码交付真实当前状态，已有记录不等于本轮新保存。"
            "没有可信成功回执不能声称完成；通知成功不证明对方已读、回复或行动。"
            "当下听懂只表达本轮理解，不用‘记下了’或‘记住了’作礼貌接话；"
            "保存成功回执只由代码在写入成功后追加，不由模型自行宣告。这些分类不授予权限。"
        )
    if not enabled(ctx):
        return SEGMENT_INSTRUCTION
    return SEGMENT_INSTRUCTION.replace(
        "conversation（普通交流、当下倾听/理解、建议、否定/假设）",
        "conversation（普通交流、复述本人说法/相处方式、当下倾听/理解、建议、否定/假设）",
    ).replace(
        "preference_state（相处说明的保存、启用、拒绝、删除或撤回状态，含范围限制）",
        "preference_state（明确声称相处说明已持久保存、启用、拒绝、删除或撤回）",
    )


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def delivery_review_pending(ctx, utterance):
    """A current Engine routing choice, never a persisted/model permission."""
    return (ctx.working_memory.get("_online_decision_version") == V7
            and getattr(ctx, "_semantic_delivery_review_v2", None)
            == (ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns), digest(utterance)))


def delivery_review_facts(ctx, utterance, persona):
    """Small code-owned facts for the existing deletion-only review.

    Empty evidence means unavailable, not that an external event never occurred.
    No record here authorizes an action, changes a preference, or expands recall.
    """
    from ..memory.preference_status import current_preference_states, render_preference_state
    from ..memory.stated_preferences import preference_is_usable
    from ..memory.referenceability import must_not_reference
    from ..model.persona import PreferenceQuote
    from .user_grounding import user_grounding_sources

    states = (current_preference_states(persona)
              if persona is not None and persona.user_id == ctx.user_id else [])
    records = []
    for state in states:
        source = next(row for row in persona.stated_preferences if row.id == state["target_id"])
        source_texts = [quote.text for quote in source.quotes]
        if isinstance(source.confirmation, PreferenceQuote):
            source_texts.append(source.confirmation.text)
        applicable = (state["status"] == "confirmed" and preference_is_usable(source)
                      and not any(must_not_reference(text) for text in source_texts))
        description = ("记录保留本人历史确认状态；当前不具备作为相处要求或引用原话使用的资格。"
                       if state["status"] == "confirmed" and not applicable
                       else render_preference_state(state))
        item = {"status": state["status"], "applicable": applicable, "description": description}
        if ctx.working_memory.get("_online_decision_version") == V7 and "last_ui_result" in state:
            # The native snapshot binds this owner's UI result to this row;
            # it is not a fresh assistant/executor completion.
            item["last_ui_result"] = {key: state["last_ui_result"][key] for key in ("action", "persisted")}
        if applicable:
            item["owner_sources"] = [quote.text for quote in source.quotes]
            item["confirmation"] = source.confirmation.model_dump(mode="json")
        records.append(item)
    trace = ctx.working_memory.get("online_turn_decision") or {}
    action_states = []
    receipt = _current_action_result(ctx, utterance)
    if receipt is not None:
        category = {"notify_family": "notification", "medication_reminder": "reminder"}.get(receipt.get("tool"))
        if category is not None:
            args = receipt.get("args") if type(receipt.get("args")) is dict else {}
            row = {"action": category, "target": str(args.get("contact") or "").strip()}
            action_states.append({**row, "status": current_action_status(ctx, utterance, row, persona)})
    return {"owner_sources": user_grounding_sources(ctx, utterance),
            "preference_records": records,
            "current_action_results": action_states,
            "code_status": (trace.get("preference_body_composition") or {}).get("code_status")}


def segment_schema(version=VERSION):
    schema = {"type": "array", "minItems": 1, "maxItems": 128, "items": {
        "type": "object", "properties": {
            "text": {"type": "string", "minLength": 1},
            "kind": {"type": "string", "enum": list(KINDS)},
            "source_quote": {"type": "string", "maxLength": 2000},
            "action": {"type": "string", "enum": ["none", "notification", "reminder", "unsupported"]},
            "target": {"type": "string", "maxLength": 200},
            "state_ref": {"type": "string", "maxLength": 64},
        }, "required": ["text", "kind", "source_quote", "action", "target", "state_ref"],
        "additionalProperties": False}}
    if version == V7:
        fields = schema["items"]["properties"]
        fields["text"] = {"type": "string"}
        fields["kind"]["enum"].append("safety_core")
        fields["action"]["enum"].insert(0, "")
        schema["items"]["required"] = ["text"]
    return schema


def valid_segments(segments, version=VERSION):
    if type(segments) is not list or not segments or (version != V7 and len(segments) > 128):
        return False
    for row in segments:
        if (type(row) is not dict or set(row) != {"text", "kind", "source_quote", "action", "target", "state_ref"}
                or any(type(value) is not str for value in row.values())
                or (not row["text"] and version != V7)
                or row["kind"] not in (KINDS + ("safety_core",) if version == V7 else KINDS)
                or len(row["source_quote"]) > 2000 or len(row["target"]) > 200 or len(row["state_ref"]) > 64):
            return False
        if row["kind"] == "assistant_action":
            allowed_actions = {"notification", "reminder", "unsupported"}
            if version == V7:
                allowed_actions.add("")
            if row["action"] not in allowed_actions or row["source_quote"]:
                return False
        elif row["action"] not in ({"", "none"} if version == V7 else {"none"}) or row["target"]:
            return False
        if row["kind"] != "third_party_fact" and row["source_quote"]:
            return False
        if row["kind"] != "preference_state" and row["state_ref"]:
            return False
    return True


def bind_current_action_receipt(ctx, utterance, receipt):
    """Called by the executor after the actual invocation, never by the model."""
    if ctx.working_memory.get("_online_decision_version") not in VERSIONS:
        return
    ctx.working_memory["_semantic_current_action_receipt"] = {
        "user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
        "turn_index": len(ctx.turns), "owner_message_sha256": digest(utterance),
        "receipt_sha256": digest(json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"))),
    }


def _current_action_result(ctx, utterance):
    receipt = ctx.working_memory.get("last_tool_result")
    if type(receipt) is not dict:
        return None
    expected = {"user_id": ctx.user_id, "session_id": ctx.session_id, "scene_id": ctx.scene_id,
        "turn_index": len(ctx.turns), "owner_message_sha256": digest(utterance),
        "receipt_sha256": digest(json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":")))}
    if ctx.working_memory.get("_semantic_current_action_receipt") != expected:
        return None
    return receipt


def current_action_receipts(ctx, utterance, persona):
    """Legacy/previous-turn successes are not evidence for a fresh completion."""
    from .action_receipts import _successful_receipts
    receipt = _current_action_result(ctx, utterance)
    if receipt is None:
        return []
    isolated = deepcopy(ctx)
    isolated.working_memory = {"last_tool_result": deepcopy(receipt)}
    return _successful_receipts(isolated, persona)


def current_action_status(ctx, utterance, row, persona):
    """Render a current executor result, including incomplete/unknown outcomes.

No second action, retry, recipient behavior, or new capability is implied.
"""
    from .action_receipts import _receipt_matches_claim
    receipt = _current_action_result(ctx, utterance)
    if receipt is None:
        return None
    category = {"notify_family": "notification", "medication_reminder": "reminder"}.get(receipt.get("tool"))
    args = receipt.get("args") if type(receipt.get("args")) is dict else {}
    if category != row["action"] or row["target"] != str(args.get("contact") or "").strip():
        return None
    if _receipt_matches_claim({"category": category, "target": row["target"]},
                              current_action_receipts(ctx, utterance, persona)):
        return ("发送服务返回了成功回执；这不证明对方已读、回复或采取行动。" if category == "notification"
                else "提醒安排已有保存回执；这不代表未来已经送达。")
    data = receipt.get("data") if type(receipt.get("data")) is dict else {}
    if data.get("executed") is False:
        return "本次动作回执明确标为未执行，没有完成该动作。"
    if receipt.get("ok") is True and data.get("status") in {"accepted", "pending", "queued"}:
        return "服务记录仍为受理或等待处理，尚无完成或送达回执，也不能确认对方已读或回复。"
    if receipt.get("ok") is False:
        return failed_action_status(data)
    return "本次动作的完成状态仍无法核实；不能确认送达、已读或回复。"


def failed_action_status(data):
    """A failed invocation is not evidence that no external effect occurred."""
    if type(data) is dict and data.get("executed") is False:
        return "本次动作回执明确标为未执行，没有完成该动作。"
    return "本次调用报告失败，未取得完成回执；外部是否产生结果仍需核查。"


def guard_semantic_delivery(ctx, response, utterance, persona, *, privacy_predicate, failed_state=None):
    """Delete only present, source-bound model segments; never resurrect text.

Receipt status is rendered from code after matching a successful receipt. The
model's free claim is not allowed to expand that status into recipient action.
An already replaced safety sentence or a code-produced status is not rewritten.
"""
    from .user_grounding import user_grounding_sources
    from ..memory.preference_status import resolve_preference_state, render_preference_state

    trace = ctx.working_memory.get("online_turn_decision") or {}
    segments = trace.get("response_segments")
    version = ctx.working_memory.get("_online_decision_version")
    audit = {"version": version, "semantic_verified": False, "violations": [],
             "activated": False, "response_changed": False, "segments": []}
    ctx.working_memory["delivery_response_guard"] = audit
    if (version not in VERSIONS or trace.get("version") != version or trace.get("status") != "parsed"
            or type(trace.get("turn_index")) is not int or trace["turn_index"] != len(ctx.turns)
            or trace.get("semantic_delivery_binding") != {
                "user_id": ctx.user_id, "scene_id": ctx.scene_id, "session_id": ctx.session_id}
            or (persona is not None and persona.user_id != ctx.user_id)
            or trace.get("owner_message_sha256") != digest(utterance)
            or not valid_segments(segments, version)
            or digest("".join(row["text"] for row in segments)) != trace.get("parsed_response_sha256")):
        # No raw protocol or unbound prose is promoted through this entry point.
        from .online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
        audit.update(activated=True, response_changed=response != DETERMINISTIC_MODEL_FAILURE_REPLY,
                     violations=["semantic_delivery_source_unavailable"])
        return DETERMINISTIC_MODEL_FAILURE_REPLY
    sources = [] if failed_state else user_grounding_sources(ctx, utterance, privacy_predicate)
    from .delivery_spans import model_positions, publish_positions
    positions = model_positions(ctx, response)
    if positions is None:
        from .online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
        audit.update(activated=True, response_changed=response != DETERMINISTIC_MODEL_FAILURE_REPLY,
                     violations=["semantic_delivery_positions_unavailable"])
        return DETERMINISTIC_MODEL_FAILURE_REPLY
    if delivery_review_pending(ctx, utterance) and not failed_state:
        # Do not let the generating model's coarse kind/quote annotation erase
        # an otherwise useful clause. The configured reviewer sees the actual
        # current state and selects exact harmful/unsupported clauses once.
        # Engine must still review this candidate before any model text leaves.
        audit.update(status="pending_independent_body_review", activated=True)
        return response
    replacements = {}
    for index, row in enumerate(segments):
        observation = {"index": index, "kind": row["kind"], "source_verified": False}
        audit["segments"].append(observation)
        if index not in positions:
            observation["status"] = "already_absent"
            continue
        if failed_state and (row["kind"] != "preference_state" or positions.count(index) != len(row["text"])):
            # The reviewer already selected the surviving clauses of a split
            # segment. Do not erase them on the basis of the coarse original tag.
            observation["status"] = "retained_after_review"
            continue
        replacement = None  # Preserve the actual surviving characters.
        reason = None
        if failed_state:
            # The user may ask about another record. Never attach the failed
            # transaction to that record or repeat either requirement's text.
            state = resolve_preference_state(row["state_ref"], persona, ctx, utterance)
            replacement = render_preference_state(failed_state, include_requirement=False)
            if state and state["target_id"] != failed_state["target_id"]:
                replacement += "另一个被引用的是不同的记录。" + render_preference_state(state, include_requirement=False)
            reason = "preference_status_from_failed_ui"
            observation["source_verified"] = True
        elif version == V7 and index in trace.get("unverified_action_segment_indexes", []):
            replacement, reason = "", "invalid_action_metadata_without_receipt"
        elif version == V7 and index in trace.get("unverified_preference_segment_indexes", []):
            replacement, reason = "", "invalid_preference_metadata_without_current_record"
        elif version == V7 and index in trace.get("unverified_fact_segment_indexes", []):
            replacement, reason = "", "invalid_third_party_metadata_without_source"
        elif row["kind"] == "preference_state":
            state = resolve_preference_state(row["state_ref"], persona, ctx, utterance)
            if state is not None:
                replacement, reason = render_preference_state(state), "preference_status_from_current_record"
                observation.update(source_verified=True, state_ref=row["state_ref"],
                                   current_state_ref=state["state_ref"], new_action_proven=False)
            else:
                replacement, reason = "", "model_preference_status_removed"
        elif row["kind"] == "third_party_fact":
            matches = [i for i, source in enumerate(sources)
                       if row["source_quote"] and row["source_quote"] in source]
            if matches:
                observation.update(source_verified=True, source_indexes=matches,
                                   quote_sha256=digest(row["source_quote"]))
            else:
                replacement, reason = "", "third_party_source_unavailable"
        elif row["kind"] == "assistant_action":
            status = current_action_status(ctx, utterance, row, persona)
            if status is not None:
                observation["source_verified"] = True
                replacement = status
                reason = "action_status_from_receipt"
            else:
                replacement, reason = "", "assistant_action_receipt_unavailable"
        if reason:
            audit["violations"].append(reason)
        observation["status"] = reason or "retained"
        if replacement is not None:
            replacements[index] = replacement
    # Apply by original occurrence index. A code-generated replacement is never
    # searched again as if it were a later, identically worded model sentence.
    parts, output_positions, rendered = [], [], set()
    for char, index in zip(response, positions):
        if index in replacements:
            if index not in rendered:
                parts.append(replacements[index])
                output_positions.extend([None] * len(replacements[index]))
                rendered.add(index)
        else:
            parts.append(char)
            output_positions.append(index)
    result = "".join(parts)
    if not result.strip():
        result = "我还没有可核实的完成状态，可以继续聊你想说的事。"
        output_positions = [None] * len(result)
    publish_positions(ctx, result, output_positions)
    audit["activated"] = bool(audit["violations"])
    audit["response_changed"] = result != response
    return result
