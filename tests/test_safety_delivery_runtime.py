"""Actual runtime/native/guard paths with synthetic HTTP, zero model requests."""
import io
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.online_decision import online_decision_tool_spec
from aria.engine.risk.safety_delivery import CONDITIONAL_CONTEXT_V1 as CANDIDATE
from aria.runtime import MinimalModelClients, create_minimal_clients, create_minimal_engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from aria.voice.web import TextWebApp
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import MODEL, route, policy
from tests.test_safety_evidence import evidence, payload, V4
from tests.test_safety_delivery_context import TEXT, build


@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected external call"))


def clients():
    c = create_minimal_clients(model_calls_enabled=True, api_key="inert-local-test",
        base_url="https://fixture.invalid/v1", online_model=MODEL,
        online_decision_version=V4, online_return_format="json_schema_v1", online_structured_route=route())
    return MinimalModelClients(c.online, MockLLMClient("memory"), MockLLMClient("reflection"), True)


def options():
    bundle = clients()
    return dict(clients=bundle, safety_delivery_version=CANDIDATE, model_max_requests=10,
                model_stop_cost_usd="1", model_reservation_policy=policy(bundle.online))


def test_public_start_submit_restart_native_fee_guard_and_memory_exclusion(tmp_path, monkeypatch):
    wires = []
    def receive(request, **kwargs):
        wire = json.loads(request.data)
        wires.append(wire)
        value = (payload("unsafe_medication", evidence(TEXT, "other", "recent_resolved", "uncertain"), "UNSAFE_RAW")
                 if len(wires) == 1 else payload(response="软件普通回答"))
        data = {"id": "gen-safetyfixture" + str(len(wires)), "model": MODEL, "provider": "Alibaba",
                "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]}
        class Response:
            status = 200
            headers = {}
            def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1): return self.body.read(size)
        return Response()
    stub_model_http(monkeypatch, receive)
    app = MinimalWebApp(tmp_path, **options())
    resumed = None
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(TEXT, "safety-fixture-1", sid)
        assert result["ok"], result
        response = result["state"]["turns"][-1]["response"]
        audit = result["state"]["turns"][-1]["safety_delivery_render"]
        assert audit["version"] == CANDIDATE and audit["status"] == "rendered"
        assert audit["new_model_understanding_exposure"] is False
        assert "quote" not in audit and audit["session_id"] == sid
        assert "不要自行" in response and "刚缓解" in response and "UNSAFE_RAW" not in response
        assert "除非专业人员有明确指导" in response
        saved = app.store.load("fictional-chen")
        assert saved.pending_session.safety_delivery_version == CANDIDATE
        assert not saved.pending_session.turns and not saved.risk_events
        assert not saved.episodic_memory and not saved.person_hypotheses
        assert json.loads(app.run_path.read_text(encoding="utf-8"))["safety_delivery_version"] == CANDIDATE
        assert len(app.call_guard.snapshot()["attempts"]) == 1
        before = app.run_path.read_bytes()
        wrong = options(); wrong["safety_delivery_version"] = "legacy"
        with pytest.raises(ValueError, match="minimal_run_safety_delivery_version_mismatch"):
            MinimalWebApp(tmp_path, **wrong)
        assert app.run_path.read_bytes() == before and len(wires) == 1
        resumed = MinimalWebApp(tmp_path, **options())
        assert resumed.start()["state"]["session_id"] == sid
        assert resumed._sessions["fictional-chen"][1].working_memory["_safety_delivery_version"] == CANDIDATE
        result = resumed.submit("软件来源乙。", "safety-fixture-2", sid)
        assert result["ok"] and result["state"]["turns"][-1]["response"] == "软件普通回答"
        assert "safety_delivery_render" not in result["state"]["turns"][-1]
        assert len(resumed.call_guard.snapshot()["attempts"]) == 2
        assert len(wires) == 2
        expected = online_decision_tool_spec(V4)["function"]["parameters"]
        assert all(wire["response_format"]["json_schema"]["schema"] == expected for wire in wires)
        assert all("conditional_context_v1" not in json.dumps(wire, ensure_ascii=False) for wire in wires)
        assert resumed.seal(sid)["ok"]
        job = resumed.store.load("fictional-chen").offline_sessions[0]
        assert all(TEXT != turn.user for turn in job.turns)
    finally:
        # Simulated crash: retire the original RAM holder without replaying it.
        app._sessions.clear()
        app.close()
        if resumed is not None: resumed.close()


@pytest.mark.parametrize("saved,current", [(CANDIDATE, "legacy"), ("legacy", CANDIDATE)])
def test_generic_active_checkpoint_cannot_silently_switch_version(tmp_path, monkeypatch, saved, current):
    store = PersonaStore(tmp_path)
    engine, ctx, _ = build(monkeypatch, payload(), version=saved)
    engine.store = store
    engine.checkpoint_session(ctx)
    before = store.load("owner").model_dump()
    app = TextWebApp(store=store, clients=clients(), safety_delivery_version=current)
    with pytest.raises(ValueError, match="checkpoint_safety_delivery_version_mismatch"):
        app._session("owner")
    assert store.load("owner").model_dump() == before and not app._sessions


def test_old_missing_checkpoint_field_stays_legacy(tmp_path):
    from aria.engine.model.persona import SessionCheckpointRecord
    pending = SessionCheckpointRecord(session_id="legacy", scene_id="companion_elder", updated_at="2020-01-01")
    assert pending.safety_delivery_version == "legacy"


@pytest.mark.parametrize("fmt", ["native", "json_object", "json_schema_v1"])
def test_explicit_candidate_and_legacy_v4_send_exact_same_request(monkeypatch, fmt):
    from tests.test_preference_required_control import client, envelope
    def run(version):
        c = client(fmt); c.online_decision_version = V4
        captures = []
        def receive(request, **kwargs):
            captures.append(request.data)
            return envelope(payload("fall", evidence(TEXT, "other", "current", "immediate")), fmt)
        monkeypatch.setattr(c, "_request_json", receive)
        bundle = MinimalModelClients(c, MockLLMClient("memory"), MockLLMClient("reflection"), True)
        engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
            clients=bundle, safety_delivery_version=version)
        ctx = EngineContext(user_id="owner", scene_id="companion_elder", session_id="fixed-software-session")
        engine.process_turn(ctx, TEXT)
        assert len(captures) == 1
        return captures[0]
    assert run("legacy") == run(CANDIDATE)


@pytest.mark.parametrize("arguments", [
    ["--safety-delivery-version", CANDIDATE],
    ["--safety-delivery-version", CANDIDATE, "--enable-model-calls"],
    ["--safety-delivery-version", "user_text"],
])
def test_cli_rejects_incompatible_selection_before_credentials(monkeypatch, arguments):
    from aria import cli
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("credentials must not be read"))
    with pytest.raises(SystemExit) as outcome: cli.web_main(arguments)
    assert outcome.value.code == 2


def test_cli_forwards_only_explicit_configuration(monkeypatch, tmp_path):
    from aria import cli
    selected = options()
    policy_file = tmp_path / "prices.json"
    policy_file.write_text(json.dumps(selected["model_reservation_policy"]), encoding="utf-8")
    observed = []
    monkeypatch.setattr(cli, "_minimal_main", lambda args: observed.append(args) or 0)
    assert cli.web_main(["--enable-model-calls", "--model-max-requests", "10",
        "--model-stop-cost-usd", "1", "--model-reservation-policy", str(policy_file),
        "--safety-delivery-version", CANDIDATE, "--online-decision-version", V4]) == 0
    assert observed[0].safety_delivery_version == CANDIDATE and observed[0].online_decision_version == V4
