"""Consume one already-paid E1b-R reflection body locally in a new checkpoint.

The composition binds immutable source files and authorization. This helper
never dispatches a model request, changes an old receipt, or invents a new one.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

from aria.bench.e1b_runner import E1bRunStopped, _CapturedReflectionUpdater
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.reflection_json import parse_reflection_json, reflection_correction_issues
from aria.voice.minimal import _changes, _stamp
from scripts import e1b_main_continuation as retained


def _require(value, code):
    if not value:
        raise E1bRunStopped(code)


class SavedContentOnlyClient:
    """One exact local read, with no delegate to the original network client."""

    reflection_return_format = "body_json"

    def __init__(self, received, control):
        self._received = deepcopy(received)
        row = self._received["receipt"]
        content, messages = self._received["content"], self._received["messages"]
        _require(type(content) is str and type(messages) is list and len(messages) == 2,
                 "received_reflection_source_invalid")
        digest = hashlib.sha256(content.encode()).hexdigest()
        _require(row.get("status") == "received" and row.get("role") == "reflection"
                 and row.get("product_step") == "initial"
                 and row.get("session_id") == retained.SESSION and row.get("operation_id") == retained.OPERATION
                 and digest == row.get("visible_content_sha256") == row.get("client_return_content_sha256"),
                 "received_reflection_body_binding_changed")
        _require(hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                 == row.get("reflection_input_sha256"), "received_reflection_input_binding_changed")
        cost = row.get("usage", {}).get("cost")
        _require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                 "received_reflection_known_cost_required")
        for key in ("source_first_sha256", "source_visible_original_sha256"):
            _require(re.fullmatch(r"[a-f0-9]{64}", self._received.get(key, "")),
                     "received_reflection_original_hash_missing")
        # This narrowly scoped continuation consumes the retained semantic
        # failure. A shape correction or a successful response needs its own
        # explicitly frozen continuation, never an accidental second request.
        try:
            parse_reflection_json(content)
        except (ValueError, TypeError, RecursionError) as error:
            _require(not reflection_correction_issues(error, content)
                     and hasattr(error, "errors")
                     and all(item["type"] == "value_error" for item in error.errors()),
                     "received_reflection_not_retained_semantic_failure")
        else:
            raise E1bRunStopped("received_reflection_not_retained_semantic_failure")
        self.model_request_control = control
        self.calls = 0
        self.last_usage = {}  # The old ledger already owns the paid usage.
        self.last_diagnostic = {"http_status": 200, "finish_reason": "stop"}

    def complete(self, messages):
        _require(self.calls == 0 and messages == self._received["messages"],
                 "received_reflection_reuse_or_input_changed")
        self.calls += 1
        return self._received["content"]


def continue_received_reflection(app, prefix, received):
    """Run the real updater/queue on retained bytes, then permit the next session."""
    retained.assert_prefix_unchanged(prefix)
    persona = app._profile()
    _require(persona.model_dump(mode="json") == PersonaProfile.model_validate(
        prefix["saved_profile"]).model_dump(mode="json"), "received_reflection_checkpoint_changed")
    _require(not app._unresolved() and not app.state()["session_id"],
             "received_reflection_unresolved_operation")
    before_accounting = deepcopy(app.call_guard.snapshot()) if app.call_guard else None
    _require(before_accounting is not None and before_accounting.get("stopped") is False
             and before_accounting.get("calls_used") == 0 and before_accounting.get("attempts") == [],
             "received_reflection_requires_fresh_guard")
    saved = SavedContentOnlyClient(received, app.call_guard)
    before = app._understanding(persona)
    original_profile = persona.model_dump(mode="json")
    source = {"campaign": received["source_campaign"], "request_id": received["receipt"]["request_id"],
              "content_sha256": received["receipt"]["client_return_content_sha256"],
              "first_sha256": received["source_first_sha256"],
              "visible_original_sha256": received["source_visible_original_sha256"],
              "new_model_calls": 0, "fees_recounted": False}
    row = {"request_id": retained.OPERATION, "session_id": retained.SESSION,
           "status": "pending", "created_at": _stamp(), "retained_received_source": source}
    app._run["reviews"].append(row)
    app._save_run()
    try:
        engine = app._engine_for_persona(persona)
        wrapper = engine.hypothesis_updater
        _require(isinstance(wrapper, _CapturedReflectionUpdater), "received_reflection_capture_required")
        updater = wrapper.updater
        _require(updater.body_json_protocol and updater.evidence_review_protocol,
                 "received_reflection_product_protocol_changed")
        updater.llm = saved
        engine.hypothesis_updater = _CapturedReflectionUpdater(updater, wrapper.capture)
        engine.diagnoser = engine.hypothesis_updater
        with app._model_scope("review", retained.OPERATION, retained.SESSION):
            result = engine.recover_offline_sessions(max_jobs=1)
        _require(saved.calls == 1 and app.call_guard.snapshot() == before_accounting,
                 "received_reflection_created_new_accounting")
        final = engine.persona.model_dump(mode="json")
        for key in original_profile:
            if key not in {"offline_sessions", "offline_session_audit", "person_model_state"}:
                _require(final[key] == original_profile[key], "received_reflection_changed_retained_memory_or_persona")
        expected_state = deepcopy(original_profile["person_model_state"])
        if retained.SESSION not in expected_state["completed_session_ids"]:
            expected_state["completed_session_ids"] = (expected_state["completed_session_ids"] + [retained.SESSION])[-100:]
            expected_state["sessions_completed"] += 1
        _require(final["person_model_state"] == expected_state,
                 "received_reflection_changed_person_model_state")
        audit = engine.persona.offline_session_audit[-1] if engine.persona.offline_session_audit else None
        success = (result.get("last_run", {}).get("completed_jobs") == 1
                   and not engine.persona.offline_sessions and audit is not None
                   and audit.session_id == retained.SESSION and audit.outcome == "reflection_failed"
                   and audit.memory_completed and not audit.reflection_completed
                   and updater.last_status == "failed" and updater.last_format_correction_count == 0)
        row.update(status="completed" if success else "failed", result=result,
                   changes=_changes(before, app._understanding(engine.persona), retained.SESSION),
                   finished_at=_stamp())
        _require(success, "received_reflection_local_seal_failed")
    except Exception:
        row.update(status="failed", requires_review=True, finished_at=_stamp())
        app._halt_models()
        raise
    finally:
        app._save_run()
    return app._result(True)
