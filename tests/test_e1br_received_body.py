"""Local retained-body consumption uses no network or original paid artifacts."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import socket

import pytest

from aria.bench.e1b_runner import E1bRunStopped, _CapturedReflectionUpdater
from aria.engine.memory.diagnosis import build_reflection_messages
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.reflection_json import reflection_body_system_prompt
from aria.runtime import MinimalModelClients, create_minimal_engine
from aria.scenes.registry import get_scene
from scripts import e1br_received_body as local
from tests.test_reflection_evidence_review import context
from tests.test_reflection_structured_contract_v23 import payload


def received(messages):
    value = payload()
    value["changes"][0]["scope_tags"] = ["general", "task_request"]
    content = json.dumps(value, ensure_ascii=False)
    sha = hashlib.sha256(content.encode()).hexdigest()
    return {"content": content, "messages": deepcopy(messages), "source_campaign": "software-fixture",
            "source_first_sha256": "a" * 64, "source_visible_original_sha256": "b" * 64,
            "receipt": {"request_id": "old-received", "role": "reflection", "status": "received",
                "session_id": local.retained.SESSION, "operation_id": local.retained.OPERATION,
                "product_step": "initial", "usage": {"cost": .02671, "reasoning_tokens": 0},
                "visible_content_sha256": sha, "client_return_content_sha256": sha,
                "reflection_input_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                    sort_keys=True).encode()).hexdigest()}}


def test_saved_client_consumes_exact_content_once_without_recounting_cost():
    source = received([{"role": "system", "content": reflection_body_system_prompt()},
                       {"role": "user", "content": "fictional source"}])
    client = local.SavedContentOnlyClient(source, None)
    assert client.complete(source["messages"]) == source["content"]
    assert client.calls == 1 and client.last_usage == {}
    with pytest.raises(E1bRunStopped, match="reuse_or_input_changed"):
        client.complete(source["messages"])


@pytest.mark.parametrize("mutation", ["body", "messages", "unknown_cost", "role", "step", "shape"])
def test_changed_body_context_or_billing_and_new_shape_attempt_are_rejected(mutation):
    source = received([{"role": "system", "content": reflection_body_system_prompt()},
                       {"role": "user", "content": "fictional source"}])
    if mutation == "body": source["content"] += " "
    if mutation == "messages": source["messages"][1]["content"] += "new context"
    if mutation == "unknown_cost": source["receipt"]["usage"].pop("cost")
    if mutation == "role": source["receipt"]["role"] = "memory"
    if mutation == "step": source["receipt"]["product_step"] = "format_correction"
    if mutation == "shape":
        source["content"] = '{"protocol_version":"2.0","reviews":"bad","changes":[]}'
        sha = hashlib.sha256(source["content"].encode()).hexdigest()
        source["receipt"].update(visible_content_sha256=sha, client_return_content_sha256=sha)
    with pytest.raises(E1bRunStopped): local.SavedContentOnlyClient(source, None)


@pytest.mark.parametrize("fail_save", [False, True])
def test_real_updater_queue_consumes_old_body_preserves_memory_and_stops_on_save_failure(tmp_path, monkeypatch, fail_save):
    def forbidden(*args, **kwargs): pytest.fail("network or original client invoked")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(local.retained, "assert_prefix_unchanged", lambda _: None)
    class Forbidden:
        reflection_return_format = "body_json"
        complete = forbidden
    clients = MinimalModelClients(Forbidden(), Forbidden(), Forbidden(), True)
    store = PersonaStore(tmp_path / "profiles")
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="person"), clients=clients, store=store)
    ctx = context(); ctx.session_id = local.retained.SESSION
    engine.seal_session(ctx)
    engine.persona.offline_sessions[0].memory_completed = True
    store.save(engine.persona)
    profile = engine.persona.model_copy(deep=True)
    conversation = "\n".join(f"[turn {i}] 用户:{t.user}\nAria:{t.agent}" for i, t in enumerate(ctx.turns))
    messages = build_reflection_messages(profile, conversation, [], True)
    messages[0]["content"] = reflection_body_system_prompt()
    source = received(messages)
    class Guard:
        stopped = False
        def snapshot(self): return {"stopped": self.stopped, "calls_used": 0, "attempts": []}
        def halt(self): self.stopped = True
    class Capture:
        session_id = local.retained.SESSION
        calls = 0
        finished = False
        def complete(self, client, sent):
            self.calls += 1
            return client.complete(sent)
        def finish(self, updater, **kwargs):
            self.finished = True
            assert updater.last_status == "failed" and updater.last_format_correction_count == 0
    capture = Capture()
    class App:
        call_guard = Guard()
        _run = {"reviews": []}
        def _profile(self): return store.load("person")
        def _unresolved(self): return False
        def state(self): return {"session_id": None}
        def _understanding(self, persona): return {"hypotheses": [h.model_dump() for h in persona.person_hypotheses]}
        def _save_run(self): pass
        def _engine_for_persona(self, persona):
            result = create_minimal_engine(get_scene("companion_elder"), persona, clients=clients, store=store)
            result.hypothesis_updater = _CapturedReflectionUpdater(result.hypothesis_updater, capture)
            return result
        def _model_scope(self, *args): return nullcontext()
        def _halt_models(self): self.call_guard.halt()
        def _result(self, ok): return {"ok": ok}
    app = App()
    prefix = {"saved_profile": profile.model_dump(mode="json")}
    original_save = store.save
    if fail_save:
        def save(persona):
            if persona.offline_session_audit: raise OSError("fixture storage failure")
            return original_save(persona)
        monkeypatch.setattr(store, "save", save)
        with pytest.raises(RuntimeError, match="offline_store_failed"):
            local.continue_received_reflection(app, prefix, source)
        assert app.call_guard.stopped and store.load("person").offline_sessions[0].memory_completed
    else:
        assert local.continue_received_reflection(app, prefix, source) == {"ok": True}
        saved = store.load("person")
        assert not saved.offline_sessions and saved.offline_session_audit[-1].outcome == "reflection_failed"
        assert not saved.person_hypotheses and not saved.evidence_review_ledger
        assert app.call_guard.snapshot() == {"stopped": False, "calls_used": 0, "attempts": []}
        assert app._run["reviews"][0]["retained_received_source"]["new_model_calls"] == 0
        with pytest.raises(E1bRunStopped, match="checkpoint_changed"):
            local.continue_received_reflection(app, prefix, source)
    assert capture.calls == 1 and capture.finished
