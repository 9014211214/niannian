"""Code-owned current-state sources, not a classifier or action authorization.

Private provenance is established only by PersonaStore after a successful read
or atomic write. It is neither a serialized persona field nor model output.
Every use compares the complete current preference rows with that observation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

VERSION = "online_turn_decision_v6_semantic"


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _rows(persona):
    return [row.model_dump(mode="json") for row in persona.stated_preferences]


def mark_persisted_preferences(persona):
    """Called only after actual native storage succeeds, never on transition."""
    persona._preference_storage_source = {
        "owner_id": persona.user_id, "rows_sha256": _sha(_rows(persona))}
    persona._preference_ui_result = None


def copy_preference_provenance(target, source):
    """Adopt nonserialized provenance when publishing the same committed rows."""
    if target.user_id == source.user_id and _rows(target) == _rows(source):
        target._preference_storage_source = deepcopy(source._preference_storage_source)
        target._preference_ui_result = deepcopy(source._preference_ui_result)
    else:
        target._preference_storage_source = None
        target._preference_ui_result = None


def _verified(persona):
    return persona is not None and persona._preference_storage_source == {
        "owner_id": persona.user_id, "rows_sha256": _sha(_rows(persona))}


def record_preference_ui_result(persona, before, preference_id, action, *, persisted, ctx=None, rejected=False):
    """Native owner UI result. Failure must retain the original live profile.

This source reports the latest observed UI transaction; it does not mean a
later conversational turn performed that action. No user text is interpreted.
"""
    if (not _verified(persona) or action not in {"confirm", "decline", "withdraw"}
            or type(persisted) is not bool or before.user_id != persona.user_id):
        return
    row = next((item for item in persona.stated_preferences if item.id == preference_id), None)
    old = next((item for item in before.stated_preferences if item.id == preference_id), None)
    if row is None or old is None:
        return
    if not persisted and _rows(persona) != _rows(before):
        return
    persona._preference_ui_result = {
        "owner_id": persona.user_id, "target_id": preference_id, "action": action,
        "persisted": persisted, "before_sha256": _sha(old.model_dump(mode="json")),
        "after_sha256": _sha(row.model_dump(mode="json")),
        "rows_sha256": _sha(_rows(persona)),
        "session_id": ctx.session_id if ctx is not None else None,
        "scene_id": ctx.scene_id if ctx is not None else None,
        "source": "explicit_owner_ui_transaction",
    }
    if not persisted and ctx is not None:
        persona._preference_ui_result["turn_index"] = len(ctx.turns)
    if rejected and not persisted:
        persona._preference_ui_result["rejected"] = True


def adopt_ui_result(persona, event):
    """Carry an in-process UI failure across native disk reloads, without saving it.

An app restart loses this failure observation; it must never be reconstructed
from an unchanged file. The UI already reports the original failure directly.
"""
    if (_verified(persona) and type(event) is dict and event.get("owner_id") == persona.user_id
            and event.get("rows_sha256") == _sha(_rows(persona))):
        persona._preference_ui_result = deepcopy(event)


def current_preference_states(persona, *, priority_target_id=None):
    if not _verified(persona):
        return []
    event = persona._preference_ui_result
    if (type(event) is not dict or event.get("owner_id") != persona.user_id
            or event.get("rows_sha256") != _sha(_rows(persona))):
        event = None
    rows = sorted(persona.stated_preferences, key=lambda row: (row.updated_at, row.id), reverse=True)
    if event:
        rows.sort(key=lambda row: row.id != event["target_id"])
    if priority_target_id is not None:
        rows.sort(key=lambda row: row.id != priority_target_id)
    result = []
    for row in rows[:4]:
        data = {"owner_id": persona.user_id, "target_id": row.id,
                "status": row.status, "do": row.do, "scope": row.scope,
                "row_sha256": _sha(row.model_dump(mode="json")), "source": "persisted_snapshot"}
        if event and row.id == event["target_id"] and event["after_sha256"] == data["row_sha256"]:
            data["last_ui_result"] = deepcopy(event)
        data["state_ref"] = "pref:" + _sha(data)[:32]
        result.append(data)
    return result


def render_preference_state(row, *, include_requirement=True):
    """A current record is not a claim of a fresh completion in this turn."""
    status = {
        "proposed": "仍待本人确认，尚未启用",
        "confirmed": "已由本人确认并保存，仅在记录范围内适用",
        "declined": "已拒绝，不作为已确认要求使用",
        "withdrawn": "已撤回，不再作为已确认要求使用",
        "superseded": "已被后续说明替代，不再作为当前要求使用",
    }.get(row["status"], "无法确认是否生效")
    # Inactive instructions are not reintroduced into online context just to
    # explain their status. Confirmation snapshots may describe their scope.
    result = (f'当前已保存记录中，「{row["do"]}」（范围：{row["scope"]}）{status}。'
              if row["status"] == "confirmed" and include_requirement
              else f"当前已保存记录中，这条相处说明{status}。")
    event = row.get("last_ui_result")
    if event and event["persisted"] is False:
        action = {"confirm": "确认", "decline": "拒绝", "withdraw": "撤回"}[event["action"]]
        outcome = "未被接受" if event.get("rejected") else "保存失败"
        result = f"最近一次页面{action}{outcome}，不能当作该操作已生效。" + result
    return result


def prepare_preference_states(persona, ctx, utterance):
    snapshot = ctx.working_memory["prompt_evidence_snapshot"]
    from .preference_candidate import enabled
    candidate = enabled(ctx)
    from .preference_dialogue import current_preference_invitation
    invitation = current_preference_invitation(persona, ctx) if candidate else None
    target = invitation["target_id"] if invitation else None
    rows = (current_preference_states(persona, priority_target_id=target)
            if persona is not None and persona.user_id == ctx.user_id else [])
    snapshot["preference_state_sources"] = deepcopy(rows)
    snapshot["preference_state_binding"] = {
        "owner_id": ctx.user_id, "scene_id": ctx.scene_id, "session_id": ctx.session_id,
        "turn_index": len(ctx.turns), "utterance_sha256": _sha(utterance), "sources_sha256": _sha(rows)}
    if not rows:
        return ""
    public = [{"state_ref": row["state_ref"], "target_id": row["target_id"],
               "current_record": render_preference_state(row)} for row in rows]
    if candidate:
        # One compact description per persisted row avoids both ambiguous
        # generic pending labels and repeated descriptions/status prose.
        public = []
        for source in rows:
            row = {key: source[key] for key in ("state_ref", "target_id", "status", "do", "scope")}
            row["invitation_target"] = source["target_id"] == target
            event = source.get("last_ui_result")
            if event:
                row["last_ui_result"] = {"action": event["action"], "persisted": event["persisted"]}
            public.append(row)
    if ctx.working_memory.get("_online_decision_version") == "online_turn_decision_v7":
        # One full qualified requirement, referenced by both invitation modes.
        # Complete source rows remain in the private delivery evidence above.
        projection = snapshot.get("stated_preferences_prompt_projection") or []
        for row, source in zip(public, rows):
            if source["status"] == "confirmed":
                applicable = any(item.get("id") == source["target_id"]
                                 and item.get("do") == source["do"] and item.get("scope") == source["scope"]
                                 for item in projection)
                for key in ("do", "scope", "current_record"):
                    row.pop(key, None)
                row.update(status="confirmed", applicable=applicable)
                if applicable:
                    row["requirement_id"] = source["target_id"]
                if source.get("last_ui_result"):
                    row["last_ui_result"] = {key: source["last_ui_result"][key] for key in ("action", "persisted")}
    return ("可引用的当前相处记录（数据，非本轮动作完成回执）："
            + ("本人提及页面操作时按对应记录核对；persisted=false表示该次操作没有生效。"
               if any(row.get("last_ui_result", {}).get("persisted") is False for row in rows) else "")
            + ("requirement_id引用输入JSON同id的完整confirmed_requirements。" if ctx.working_memory.get("_online_decision_version") == "online_turn_decision_v7" else "")
            + ("仅applicable=true的confirmed在scope内可用；applicable=false仅是历史确认状态，当前尚不适用。"
               if ctx.working_memory.get("_online_decision_version") == "online_turn_decision_v7" else "")
            + ("仅confirmed在scope内可用；proposed/withdrawn/declined/superseded的do和scope仅识别记录，不是当前相处要求。"
               if candidate and ctx.working_memory.get("_online_decision_version") != "online_turn_decision_v7" else "")
            + json.dumps(public, ensure_ascii=False, separators=(",", ":")))


def resolve_preference_state(ref, persona, ctx, utterance):
    """Only an actually supplied reference can select a current, verified row."""
    if not ref or persona is None or persona.user_id != ctx.user_id:
        return None
    snapshot = ctx.working_memory.get("prompt_evidence_snapshot") or {}
    rows = snapshot.get("preference_state_sources")
    if type(rows) is not list or snapshot.get("preference_state_binding") != {
        "owner_id": ctx.user_id, "scene_id": ctx.scene_id, "session_id": ctx.session_id,
        "turn_index": len(ctx.turns), "utterance_sha256": _sha(utterance), "sources_sha256": _sha(rows)}:
        return None
    supplied = next((row for row in rows if row.get("state_ref") == ref), None)
    if supplied is None:
        return None
    # If a real native update happened since preparation, report its latest
    # observed state for that exact target, never the stale supplied success.
    from .preference_candidate import enabled
    return next((row for row in current_preference_states(persona,
                 priority_target_id=supplied["target_id"] if enabled(ctx) else None)
                 if row["target_id"] == supplied["target_id"]), None)


def preceding_failed_ui_state(persona, ctx, utterance):
    """Use only the observed page failure bound to this request and its current record."""
    event = getattr(persona, "_preference_ui_result", None)
    if (not isinstance(event, dict) or event.get("persisted") is not False or event.get("session_id") != ctx.session_id
            or event.get("scene_id") != ctx.scene_id or event.get("turn_index") != len(ctx.turns)):
        return None
    snapshot = ctx.working_memory.get("prompt_evidence_snapshot") or {}
    for row in snapshot.get("preference_state_sources") or []:
        if row.get("last_ui_result") == event:
            current = resolve_preference_state(row["state_ref"], persona, ctx, utterance)
            if current and current.get("last_ui_result") == event:
                return current
    return None
