"""The normal release entry point and real wire, exercised without network."""
import json

import pytest

from aria import cli
from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.release import create_release_clients, engine_options, ONLINE_MODEL, ONLINE_VERSION
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from tests._model_http_stub import stub_model_http
from tests.test_online_structured import metadata, route
from tests.test_preference_required_control import envelope
from tests.test_reserved_calls import policy
from tests.test_safety_evidence import payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("network forbidden"))


def clients():
    return create_release_clients(model_calls_enabled=True, api_key="inert-software-fixture",
        base_url="https://fixture.invalid/v1", online_structured_route=route())


def release_policy():
    value = policy()
    old_model = next(model for model in value["models"] if model.startswith("qwen/"))
    row = value["models"].pop(old_model)
    structured = route().binding()
    row["response_models"] = [ONLINE_MODEL, structured["canonical_slug"]]
    row["response_providers"] = [structured["provider_name"]]
    row["metadata_sha256"] = structured["endpoint_metadata_sha256"]
    row["input_usd_per_million"] = "0.20"
    row["output_usd_per_million"] = "0.47"
    value["models"][ONLINE_MODEL] = row
    return value


def test_factory_is_inert_and_defaults_reach_the_engine_wire(monkeypatch):
    selected = clients()
    assert selected.release_profile == "release_v1"
    assert selected.online.model == ONLINE_MODEL
    assert selected.online.online_return_format == "json_schema_v1"
    assert selected.online.online_decision_version == ONLINE_VERSION
    assert selected.online.max_completion_tokens == 4096 and selected.online.timeout == 90
    assert selected.reflection.reflection_return_format == "body_json"
    for client in (selected.online, selected.memory, selected.reflection):
        assert client.network_request_count == 0
        assert client.max_retries == 0 and client.fallback_models == []
    engine = create_minimal_engine(get_scene("companion_elder"), PersonaProfile(user_id="owner"),
                                   clients=selected)
    for key, value in engine_options().items():
        assert getattr(engine, key) == value
    raw = "第一行的实际软件首答。\n第二行的实际软件首答。"
    wires = []
    def receive(request, **options):
        wire = json.loads(request.data)
        wires.append(wire)
        return envelope(payload(response=raw), "json_schema_v1")
    monkeypatch.setattr(selected.online, "_request_json", receive)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    assert engine.process_turn(ctx, "中性的软件装配输入。") == raw
    wire, = wires
    assert len(wire["messages"][0]["content"].encode()) <= 12000
    assert wire["response_format"]["type"] == "json_schema"
    assert selected.memory.network_request_count == selected.reflection.network_request_count == 0


def test_web_constructor_inherits_release_bundle_without_calling_models(tmp_path):
    selected = clients()
    app = MinimalWebApp(tmp_path, clients=selected, model_max_requests=4,
                        model_stop_cost_usd="1", model_reservation_policy=release_policy())
    try:
        for key, value in engine_options().items():
            assert getattr(app, key) == value
        assert all(client.network_request_count == 0 for client in (
            selected.online, selected.memory, selected.reflection))
    finally:
        app.close()


@pytest.mark.parametrize("current", ["中性的软件装配输入。", "请核对护理提醒记录。", "软件前置输入：喘不上气。"])
def test_web_actual_dispatch_keeps_budget_and_original_body(tmp_path, monkeypatch, current):
    from tests.test_source_response_web import response, BODY
    selected = clients()
    app = MinimalWebApp(tmp_path, clients=selected, model_max_requests=4,
                        model_stop_cost_usd="1", model_reservation_policy=release_policy())
    wires = []
    def receive(request, **kwargs):
        wires.append(json.loads(request.data))
        return response(request)
    stub_model_http(monkeypatch, receive)
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit(current, "release-software-1", sid, on_safety_prelude=lambda frame: True)
        assert result["ok"] and result["state"]["turns"][-1]["response"] == BODY
        wire, = wires
        assert len(wire["messages"][0]["content"].encode()) <= 12000
        assert json.loads(wire["messages"][1]["content"])["current_user_message"] == current
        assert len(app.call_guard.snapshot()["attempts"]) == 1
        assert selected.memory.network_request_count == selected.reflection.network_request_count == 0
    finally:
        app.close()


def test_cli_explicit_historical_flash_bundle_using_public_metadata(tmp_path, monkeypatch):
    paths = []
    for name, value in zip(("models", "endpoints", "reservation"), (*metadata(), release_policy())):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    seen = []
    monkeypatch.setattr(cli, "_minimal_main", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("must not read credentials"))
    assert cli.web_main(["--enable-model-calls", "--release-profile", "release_v1", "--online-models-metadata", str(paths[0]),
        "--online-endpoints-metadata", str(paths[1]), "--model-reservation-policy", str(paths[2]),
        "--model-max-requests", "4", "--model-stop-cost-usd", "1"]) == 0
    args, = seen
    assert args.release_profile == "release_v1"
    assert args.online_decision_version == ONLINE_VERSION
    for key, value in engine_options().items():
        assert getattr(args, key) == value
    assert args.online_structured_route.model == ONLINE_MODEL


def test_release_metadata_is_required_before_any_credentials(monkeypatch):
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("must not read credentials"))
    with pytest.raises(SystemExit) as raised:
        cli.web_main(["--enable-model-calls"])
    assert raised.value.code == 2


def test_explicit_release_profile_cannot_silently_downgrade(monkeypatch):
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("must not read credentials"))
    with pytest.raises(SystemExit) as raised:
        cli.web_main(["--enable-model-calls", "--release-profile", "release_v1",
                      "--safety-delivery-version", "legacy"])
    assert raised.value.code == 2


def test_explicit_legacy_cli_retains_legacy_defaults(tmp_path, monkeypatch):
    path = tmp_path / "reservation.json"
    path.write_text(json.dumps(policy()), encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "_minimal_main", lambda args: seen.append(args) or 0)
    assert cli.web_main(["--enable-model-calls", "--release-profile", "legacy",
        "--model-reservation-policy", str(path), "--model-max-requests", "4",
        "--model-stop-cost-usd", "1"]) == 0
    args, = seen
    assert args.release_profile == "legacy"
    assert args.online_decision_version == "online_turn_decision_v3"
    assert all(getattr(args, key) == "legacy" for key in engine_options())
