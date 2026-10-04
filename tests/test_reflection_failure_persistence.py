"""Keep safe reflection validation details through the real offline queue."""
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine


class _UntrustedDetail:
    def __str__(self):
        raise AssertionError("untrusted detail must not be stringified")


class _StringSubclass(str):
    def __hash__(self):
        raise AssertionError("untrusted string subclass must not be inspected")


class _FailedUpdater:
    last_status = "failed"

    def __init__(self, detail):
        self.last_error_detail = detail

    def update(self, persona, _ctx):
        return persona


def _engine(updater):
    engine = object.__new__(Engine)
    engine.hypothesis_updater = updater
    return engine


@pytest.mark.parametrize("detail", [
    None, "PRIVATE_PROVIDER_BODY", {"error": "PRIVATE_PROVIDER_BODY"},
    ValueError("PRIVATE_PROVIDER_BODY"), _UntrustedDetail(),
    _StringSubclass("review_protocol_version_required"),
])
def test_untrusted_or_absent_detail_keeps_generic_safe_code(detail):
    persona = PersonaProfile(user_id="fixture")
    ctx = EngineContext(user_id="fixture", scene_id="companion_elder")
    assert _engine(_FailedUpdater(detail))._offline_reflection_stage(persona, ctx) == (
        persona, "failed", "person_reflection_failed",
    )


@pytest.mark.parametrize("detail", [
    "review_protocol_version_required", "review_quote_not_grounded_in_current_user",
    "same_event_requires_known_reference", "model_call_failed",
])
def test_reviewed_error_details_survive_stage_boundary(detail):
    persona = PersonaProfile(user_id="fixture")
    ctx = EngineContext(user_id="fixture", scene_id="companion_elder")
    assert _engine(_FailedUpdater(detail))._offline_reflection_stage(persona, ctx) == (
        persona, "failed", detail,
    )


def test_real_validation_failure_is_saved_by_queue_without_raw_output(tmp_path):
    class Reply:
        calls = 0

        def complete(self, _messages):
            self.calls += 1
            # Numeric 2.0 is not the required string. The safe code should be
            # retained without saving this provider text or inventing a repair.
            return json.dumps({"protocol_version": 2.0, "reviews": [], "changes": []})

    class Store:
        def save(self, persona):
            path.write_text(persona.model_dump_json(), encoding="utf-8")

    path = tmp_path / "profile.json"
    ctx = EngineContext(user_id="fixture", scene_id="companion_elder", session_id="session")
    for word in ["今天去公园。", "坐了一会儿。", "现在回家了。"]:
        ctx.add_turn(word, "我听到了。")
    reply = Reply()
    updater = PersonaHypothesisUpdater(reply, evidence_review_protocol=True)
    engine = _engine(updater)
    queue = OfflineSessionQueue(PersonaProfile(user_id="fixture"), Store())
    assert queue.enqueue(ctx, memory_completed=True)["status"] == "queued"
    result = queue.drain(lambda person, _: (person, "skipped", None),
                         engine._offline_reflection_stage)
    assert result["status"] == "failed"
    assert result["error_code"] == "review_protocol_version_required"
    saved = PersonaProfile.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.offline_sessions[0].last_error_code == "review_protocol_version_required"
    assert saved.offline_sessions[0].attempts == 1
    assert saved.offline_sessions[0].reflection_completed is False
    assert not saved.person_hypotheses and not saved.evidence_review_ledger
    assert reply.calls == 1
    # A fresh queue restores the exact safe detail without another model call.
    restored = OfflineSessionQueue(saved)
    assert restored.status()["jobs"][0]["last_error_code"] == "review_protocol_version_required"
    assert reply.calls == 1
