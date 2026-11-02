"""Run scripted owner operations through MinimalWebApp; never replay a failed turn."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aria.voice.minimal import MinimalWebApp
from scripts.run_safety_set import checksum, configured_factory, digest, parser, read_json, segment_context, seal_segment
def load_story(path):
    path = Path(path)
    story = read_json(path)
    addendum = path.with_name("operations-order-addendum-v1.private.json")
    if addendum.exists():
        order = read_json(addendum)
        if hashlib.sha256(path.read_bytes()).hexdigest() != order["base_materials_sha256"]:
            raise ValueError("story_addendum_base_mismatch")
        for patch in order["patches"]:
            repeat = next(r for r in story["repetitions"] if r["repeat_id"] == patch["repeat_id"])
            session = next(s for s in repeat["sessions"] if s["session_id"] == patch["session_id"])
            for value, key in ((session["operations"], "original_operations_sha256"),
                               (patch["effective_operations"], "effective_operations_sha256")):
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if hashlib.sha256(encoded.encode()).hexdigest() != patch[key]:
                    raise ValueError("story_addendum_operations_mismatch")
            session["operations"] = patch["effective_operations"]
    return story

def fee_stop(usage):
    reasons = {"pre_send_budget_exceeded", "request_limit_reached", "logical_request_limit_reached",
               "observed_cost_threshold_reached", "provider_cost_unknown", "ledger_unavailable",
               "unresolved_request", "unknown_429_task_limit_reached", "recovery_unknown_limit_reached"}
    if usage.get("stopped") and (usage.get("stop_reason") in reasons or usage.get("unknown_cost_requests")):
        return usage.get("stop_reason") or "unknown_cost"
    return None
def invoke(call):
    try:
        return call()
    except Exception as exc:
        return {"ok": False, "error": "product_exception", "error_type": type(exc).__name__}
class StoryRun:
    def __init__(self, app, summary, stream, sequence):
        self.app, self.summary, self.stream, self.sequence = app, summary, stream, sequence
        self.bindings, self.turn_ids, self.session_id = {}, {}, None
        self.serial = 0
    def record(self, operation, result, *, turn_id=None):
        state = result.get("state") or self.app.state()
        if "profile" not in state:
            state = self.app.state()
        turn = next((t for t in state["turns"] if t["turn_id"] == turn_id), {})
        trace = {}
        for _, ctx in self.app._sessions.values():
            candidate = ctx.working_memory.get("online_turn_decision", {})
            if turn and ctx.session_id == turn["session_id"] and candidate.get("turn_index") == turn.get("turn_index"):
                trace = candidate
        prefs = state["profile"]["stated_preferences"]
        failed = not result.get("ok", False) or turn.get("status") in {"failed", "pending"}
        row = {"repeat": self.repeat, "session": self.label, "session_id": self.session_id,
               "operation": operation, "elder_text": operation.get("text"),
               "delivered_response": turn.get("response", result.get("delivered_response", "")), "ok": result.get("ok", False),
               "error": result.get("error"), "failed": failed, "turn_id": turn_id,
               "source": turn.get("source"), "dropped_fields": trace.get("dropped_fields", []),
               "preferences": prefs, "preference_invitation": state.get("preference_invitation"),
               "result": {k: v for k, v in result.items() if k != "state"},
               "review": state["reviews"][-1] if operation["op"] == "reflect" and state["reviews"] else None}
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        if not self.sequence or prefs != self.sequence[-1]["preferences"]:
            self.sequence.append({"session": self.label, "operation": self.serial, "preferences": deepcopy(prefs)})
        self.summary["fees"] = {k: v for k, v in state["model_usage"].items()
                                if k not in {"attempts", "product_events", "product_retry_waits"}}
        self.summary["stop_reason"] = fee_stop(state["model_usage"])
        if turn_id is not None:
            self.summary["turns"] += 1
            self.summary["failed_turns"] += int(failed)
        elif failed:
            self.summary["failed_operations"] += 1
            self.summary["failed_reviews"] += int(operation["op"] == "reflect")
        return state
    def step(self, op):
        self.serial += 1
        command_id = f"op-{self.repeat}-{self.session}-{self.serial}"
        kind, turn_id = op["op"], None
        if kind in {"say", "answer_preference"}:
            turn_id = command_id
            self.turn_ids[f"{self.label}.{op.get('turn_id', self.serial)}"] = turn_id
            bound = (invoke(lambda: self.app.load_fictional_context(op["execution_input"], self.session_id, fictional=True))
                     if "execution_input" in op else {"ok": True})
            if "execution_input" in op:
                self.record({"op": "load_fictional_context"}, bound)
            def prelude(frame):
                turn = next(t for t in frame["state"]["turns"] if t["turn_id"] == turn_id)
                self.record({"op": "safety_prelude"}, {"ok": True, "state": frame["state"],
                    "delivered_response": turn["protection_prelude"]["protection"]})
                return True
            result = invoke(lambda: self.app.submit(op["text"], turn_id, self.session_id, on_safety_prelude=prelude)) if bound["ok"] else bound
        elif kind == "start_session":
            result = invoke(self.app.start)
            self.session_id = result.get("state", {}).get("session_id")
        elif kind == "end_session_review":
            self.step({"op": "seal"})
            if self.summary["stop_reason"]:
                return
            self.step({"op": "reflect"})
            if op.get("bind_review_as"):
                self.bindings[op["bind_review_as"]] = self.session_id
            return
        elif kind == "seal":
            result = invoke(lambda: self.app.seal(self.session_id))
        elif kind == "reflect":
            result = invoke(lambda: self.app.reflect(command_id, wait=True))
        else:
            result = invoke(lambda: self.owner_operation(op, command_id))
        state = self.record(op, result, turn_id=turn_id)
        invitation = state.get("preference_invitation")
        # Persist the actual displayed question before acknowledging its delivery.
        if invitation and invitation["status"] == "awaiting_display" and not self.summary["stop_reason"]:
            ack = invoke(lambda: self.app.preference_invitation_ack(
                invitation["id"], invitation["question_sha256"], self.session_id))
            self.record({"op": "invitation_display_ack"}, ack)
    def owner_operation(self, op, command_id):
        kind, state = op["op"], self.app.state()
        if kind == "invite_preference":
            source = self.bindings[op["source_review"]]
            candidates = [p for p in state["profile"]["stated_preferences"]
                          if p["status"] == "proposed" and p["can_confirm"]
                          and any(q["session_id"] == source for q in p["quotes"])]
            if not candidates:
                return {"ok": False, "error": "preference_target_missing_or_ambiguous"}
            try:
                order = [(datetime.fromisoformat(p["created_at"]), p["id"]) for p in candidates]
                if any(at.utcoffset() is None or type(pid) is not str or not pid for at, pid in order):
                    raise ValueError("invalid_candidate_metadata")
                target = min(order)[1]
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "preference_target_metadata_invalid"}
            self.bindings[op["target_slot"]] = target
            return self.app.preference_invitation(target, self.session_id)
        if kind in {"confirm_preference", "decline_preference", "revoke_preference"}:
            if op["target_slot"] not in self.bindings:
                return {"ok": False, "error": "preference_target_missing"}
            action = {"confirm_preference": "confirm", "decline_preference": "decline", "revoke_preference": "withdraw"}[kind]
            return self.app.preference_action(action, self.bindings[op["target_slot"]])
        if kind == "sharing_preview":
            recipient = op["recipient"]
            if op["mode"] != "local_mock_only" or recipient["kind"] != "local_mock_inbox" or recipient["address"] is not None:
                return {"ok": False, "error": "only_local_sharing_supported"}
            source_id = self.turn_ids[".".join(op["selection"]["source"].split(".")[:2])]
            source = next(s for s in state["sharing"]["sources"] if s["turn_id"] == source_id)
            text = op["selection"]["text"]
            if not text or source["text"].count(text) != 1:
                return {"ok": False, "error": "selected_quote_missing_or_ambiguous"}
            if "_care_bootstrap" not in self.bindings:
                self.bindings["_care_bootstrap"] = self.app.care.bootstrap(
                    "story-local", secrets.token_urlsafe(32), ["fictional-chen"])
            contact = self.app.care_action("contact.create", {"request_id": command_id + "-contact",
                "name": "虚构本地收件人", "channel": "local"})["record"]
            start = source["text"].index(text)
            result = self.app.sharing_action("prepare", {"request_id": command_id,
                "source_id": source_id, "contact_id": contact["id"],
                "selection": {"start": start, "end": start + len(text), "unit": "unicode_codepoint"}}, self.session_id)
            self.bindings[op["bind_artifact_as"]] = result["record"]
            return {**result, "local_contact": contact}
        if kind in {"sharing_approve", "sharing_send", "sharing_revoke"}:
            row = self.bindings[op["artifact_ref"]]
            action = kind.removeprefix("sharing_")
            payload = {"request_id": command_id, "id": row["id"]}
            if action != "revoke":
                payload["preview_sha256"] = row["preview_sha256"]
            if action == "approve":
                payload["confirmed"] = True
            result = self.app.sharing_action(action, payload, self.session_id)
            self.bindings[op["artifact_ref"]] = result["record"]
            return result
        return {"ok": False, "error": "unsupported_operation"}
def run_story(story, repeats, out, *, app_factory=MinimalWebApp, prior=None, unit_limit=None, context=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("positive_repeats_required")
    if unit_limit is not None and (type(unit_limit) is not int or unit_limit < 1):
        raise ValueError("positive_unit_limit_required")
    plans = story.get("repetitions", [story] * repeats)
    if repeats > len(plans):
        raise ValueError("requested_repeats_exceed_sealed_materials")
    units = []
    for number, plan in enumerate(plans[:repeats], 1):
        indexed = list(enumerate(plan["sessions"], 1))
        if story.get("independent_cases"):
            if not all(session.get("fresh_profile") for _, session in indexed):
                raise ValueError("safety_case_requires_fresh_profile")
            units.extend((number, [item]) for item in indexed)
        else:
            units.append((number, indexed))
    first = prior["last_attempted_unit"] if prior else 0
    if prior and not 0 < first < len(units):
        raise ValueError("no_unattempted_independent_unit")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if context is not None:
        with (out / "segment-start.json").open("x", encoding="utf-8") as stream:
            json.dump(context, stream, ensure_ascii=False, indent=2)
    summary = {"requested_repeats": repeats, "completed_repeats": 0, "sessions": 0,
               "turns": 0, "failed_turns": 0, "failed_operations": 0, "failed_reviews": 0,
               "stop_reason": None, "fees": {}, "preference_changes": {},
               "total_units": len(units), "last_attempted_unit": first, "complete_units": [],
               "planned_turns": sum(sum(op["op"] in {"say", "answer_preference"}
                   for op in s["operations"]) if "operations" in s else len(s["turns"])
                   for p in plans[:repeats] for s in p["sessions"])}
    if prior:
        for field in ("sessions", "turns", "failed_turns", "failed_operations", "failed_reviews",
                      "preference_changes", "complete_units"):
            summary[field] = deepcopy(prior[field])
    try:
        for ordinal, (number, sessions) in enumerate(units[first:first + unit_limit if unit_limit else None], first + 1):
            summary["last_attempted_unit"] = ordinal
            base = out / str(number) / "data"
            first_session = sessions[0][0] if sessions else 1
            app = app_factory(base if first_session == 1 else base / str(first_session))
            sequence = summary["preference_changes"].setdefault(str(number), [])
            runner = StoryRun(app, summary, None, sequence)
            runner.repeat = number
            try:
                for index, session in sessions:
                    if index != first_session and session.get("fresh_profile"):
                        app.close()
                        app = app_factory(out / str(number) / "data" / str(index))
                        runner = StoryRun(app, summary, None, sequence)
                        runner.repeat = number
                    runner.session, runner.label = index, session.get("session_id", str(index))
                    folder = out / str(number) / str(index)
                    folder.mkdir(parents=True)
                    with (folder / "turns.jsonl").open("x", encoding="utf-8") as stream:
                        runner.stream = stream
                        operations = session.get("operations")
                        if operations is None:
                            operations = [{"op": "start_session"}, *({"op": "say", "text": t}
                                for t in session["turns"]), {"op": "end_session_review"}]
                        for op in operations:
                            runner.step(op)
                            if summary["stop_reason"]:
                                break
                    summary["sessions"] += 1
                    if summary["stop_reason"]:
                        break
                if not summary["stop_reason"]:
                    summary["complete_units"].append(ordinal)
            finally:
                app.close()
            if summary["stop_reason"]:
                break
    finally:
        complete = set(summary["complete_units"])
        summary["completed_repeats"] = sum(all(i in complete for i, (r, _) in enumerate(units, 1) if r == number)
                                           for number in range(1, repeats + 1))
        summary["unexecuted_turns"] = summary["planned_turns"] - summary["turns"]
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
def segment_identity(args, story):
    root = Path(__file__).resolve().parents[1]
    files = [root / "pyproject.toml"]
    for folder in ("aria", "scripts", "tests"):
        files.extend(f for f in (root / folder).rglob("*") if f.is_file() and "__pycache__" not in f.parts
                     and f.suffix in {".py", ".json", ".yaml", ".html", ".md", ".toml"})
    config_files = ("model_reservation_policy", "online_models_metadata", "online_endpoints_metadata",
                    "offline_endpoints_metadata", "continuation_policy")
    addendum = args.story.with_name("operations-order-addendum-v1.private.json")
    return {"source": digest({f.relative_to(root).as_posix(): checksum(f) for f in files}),
            "inputs": checksum(args.story), "effective_plan": digest(story), "repeats": args.repeats,
            "operations_addendum": checksum(addendum) if addendum.is_file() else None,
            "paid": args.enable_model_calls, "profile": args.release_profile,
            "offline_model": args.offline_model, "offline_provider": args.offline_provider,
            **{key: checksum(getattr(args, key)) if getattr(args, key) else None for key in config_files}}

def main(argv=None, *, story_loader=load_story):
    cli = parser()
    args = cli.parse_args(argv)
    close = None
    try:
        story = story_loader(args.story)
        context, prior = segment_context(args, segment_identity(args, story))
        factory, close = configured_factory(args)
        summary = run_story(story, args.repeats, args.out, app_factory=factory, prior=prior,
                            unit_limit=args.unit_limit, context=context)
        if close:
            close()
            close = None
        seal_segment(args, context, summary)
        return 2 if summary["stop_reason"] else 3 if summary["unexecuted_turns"] else 0
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, AttributeError, StopIteration):
        cli.error("运行配置或材料无效；请核对路径、材料版本、元数据和费用参数。")
    finally:
        if close:
            close()

if __name__ == "__main__":
    raise SystemExit(main())
