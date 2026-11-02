"""Public runtime/configuration/guard/save boundaries using local HTTP fixtures."""
import io
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile, SessionCheckpointRecord
from aria.engine.policy.safety_instruction import CONDITIONAL_RESPONSE_V1 as CANDIDATE, SYSTEM_INSTRUCTION
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from aria.voice.web import TextWebApp
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import policy as price_policy
from tests.test_preference_required_control import envelope
from tests.test_safety_evidence import payload, evidence, V4
from tests.test_safety_response_instruction import bundle, TEXT


def options(fmt="json_schema_v1", version=CANDIDATE):
    clients = bundle(fmt)
    prices = price_policy(bundle().online)
    if fmt == "json_object":
        from aria._openrouter import DEFAULT_ONLINE_FAST_MODEL
        clients.online.model = DEFAULT_ONLINE_FAST_MODEL
        spec = prices["models"].pop("qwen/qwen3.8-flash")
        spec["response_models"] = [DEFAULT_ONLINE_FAST_MODEL]
        prices["models"][DEFAULT_ONLINE_FAST_MODEL] = spec
    return dict(clients=clients, safety_response_instruction_version=version,
        safety_delivery_version="conditional_context_v1", model_max_requests=10,
        model_stop_cost_usd="1", model_reservation_policy=prices)


@pytest.mark.parametrize("fmt", ["json_object", "json_schema_v1"])
@pytest.mark.parametrize("save_failure", [False, True])
def test_public_actual_request_guard_resume_and_renderer_independence(tmp_path, monkeypatch, fmt, save_failure):
    requests = []
    def receive(request, **kwargs):
        requests.append(request)
        if save_failure:
            def failed_save(*a, **k): raise OSError("software save failure")
            monkeypatch.setattr(app.store, "save", failed_save)
        value = payload("unsafe_medication", evidence(TEXT, "other", "recent_resolved", "uncertain"), "UNSAFE_RAW")
        data = {"id": "gen-safe-response-fixture", "model": json.loads(request.data)["model"], "provider": "Alibaba",
            "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1}, **envelope(value, fmt)}
        class Response:
            status = 200
            headers = {}
            def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1): return self.body.read(size)
        return Response()
    stub_model_http(monkeypatch, receive)
    app = MinimalWebApp(tmp_path, **options(fmt)); resumed = None
    try:
        sid = app.start()["state"]["session_id"]
        original_save = app.store.save
        result = app.submit(TEXT, "instruction-fixture-1", sid)
        if save_failure:
            monkeypatch.setattr(app.store, "save", original_save)
            assert not result["ok"] and len(requests) == 1
            assert not result["state"]["turns"][-1].get("response")
            assert not app.submit(TEXT, "instruction-fixture-2", sid)["ok"]
            assert len(requests) == 1
            return
        assert result["ok"], result
        row = result["state"]["turns"][-1]
        assert "UNSAFE_RAW" not in row["response"]
        assert row["safety_response_instruction"]["passed_to_client"] is True
        assert row["safety_response_instruction"]["actual_send_verified"] is False
        assert row["safety_delivery_render"]["new_model_understanding_exposure"] is False
        assert json.loads(requests[0].data)["messages"][0]["content"].count(SYSTEM_INSTRUCTION) == 1
        assert len(app.call_guard.snapshot()["attempts"]) == len(requests) == 1
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.safety_response_instruction_version == CANDIDATE
        assert not saved.pending_session.turns and not saved.risk_events
        before = app.run_path.read_bytes()
        with pytest.raises(ValueError, match="minimal_run_safety_response_instruction_version_mismatch"):
            MinimalWebApp(tmp_path, **options(fmt, "legacy"))
        assert app.run_path.read_bytes() == before
        resumed = MinimalWebApp(tmp_path, **options(fmt))
        assert resumed.start()["state"]["session_id"] == sid
        assert resumed._sessions["fictional-chen"][1].working_memory["_safety_response_instruction_version"] == CANDIDATE
        assert resumed.seal(sid)["ok"]
    finally:
        app._sessions.clear(); app.close()
        if resumed: resumed.close()


def test_existing_native_fee_route_is_rejected_without_requests(tmp_path, monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("no request allowed"))
    with pytest.raises(ValueError, match="reservation_requires_bounded_transport"):
        MinimalWebApp(tmp_path, **options("native"))


@pytest.mark.parametrize("saved,current", [(CANDIDATE, "legacy"), ("legacy", CANDIDATE)])
def test_active_checkpoint_version_mismatch_leaves_store_unchanged(tmp_path, saved, current):
    store = PersonaStore(tmp_path)
    p = PersonaProfile(user_id="owner")
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=bundle(), store=store,
        safety_response_instruction_version=saved)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    engine.checkpoint_session(ctx)
    before = store.load("owner").model_dump()
    app = TextWebApp(store=store, clients=bundle(), safety_response_instruction_version=current)
    with pytest.raises(ValueError, match="checkpoint_safety_response_instruction_version_mismatch"):
        app._session("owner")
    assert store.load("owner").model_dump() == before and not app._sessions


def test_old_missing_checkpoint_is_legacy_and_readonly_config():
    record = SessionCheckpointRecord(session_id="legacy", scene_id="companion_elder", updated_at="2020-01-01")
    assert record.safety_response_instruction_version == "legacy"


@pytest.mark.parametrize("arguments", [
    ["--safety-response-instruction-version", CANDIDATE],
    ["--safety-response-instruction-version", CANDIDATE, "--enable-model-calls"],
    ["--safety-response-instruction-version", "user_text"],
])
def test_cli_invalid_before_credentials(monkeypatch, arguments):
    from aria import cli
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("credentials forbidden"))
    with pytest.raises(SystemExit) as outcome: cli.web_main(arguments)
    assert outcome.value.code == 2


def test_cli_explicit_v4_and_both_candidates(monkeypatch, tmp_path):
    from aria import cli
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(options()["model_reservation_policy"]), encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "_minimal_main", lambda args: seen.append(args) or 0)
    assert cli.web_main(["--enable-model-calls", "--model-max-requests", "10", "--model-stop-cost-usd", "1",
        "--model-reservation-policy", str(prices), "--online-decision-version", V4,
        "--safety-response-instruction-version", CANDIDATE,
        "--safety-delivery-version", "conditional_context_v1"]) == 0
    assert seen[0].safety_response_instruction_version == CANDIDATE
    assert seen[0].safety_delivery_version == "conditional_context_v1"
