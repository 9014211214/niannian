"""Independent fee-boundary review: no live model or credentials."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import json
import urllib.request

import pytest

from aria._openrouter import DEFAULT_MEMORY_MODEL, OPENROUTER_BASE_URL
from aria.engine.policy.minimal_calls import MAX_MODEL_RESPONSE_BYTES, MinimalCallStopped
from aria.runtime import create_minimal_clients
from aria.voice.minimal import MinimalWebApp
from tests.test_reserved_calls import client, guard, call, receipt, policy, stub, deny_network
from tests.test_prepare_model_policy import inputs, derive


def clients():
    return create_minimal_clients(model_calls_enabled=True, api_key="synthetic-test-only",
                                  base_url="https://fixture.invalid/v1")


def raw_response(monkeypatch, raw):
    reads = []
    class Response:
        status = 200
        headers = {"x-request-id": "req-reviewfixture1234"}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, size):
            reads.append(size)
            return raw[:size]
    stub_model_http(monkeypatch, lambda *_a, **_k: Response())
    return reads


def test_shared_ledger_identity_persists_across_restart_and_close_is_borrowed(tmp_path):
    path = tmp_path / "budget.sqlite3"
    shared = guard(path)
    owner = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=shared)
    owner.start()
    saved_id = owner._run["model_allowance"]["ledger_id"]
    owner.close()
    assert not shared.snapshot()["stopped"]
    shared.close()
    reopened = guard(path)
    assert reopened.ledger_id == saved_id
    owner = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=reopened)
    owner.close(); reopened.close()


def test_different_empty_shared_ledger_cannot_reset_an_existing_run(tmp_path):
    original = guard(tmp_path / "first.sqlite3")
    app = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=original)
    app.start(); app.close()
    other = guard(tmp_path / "second.sqlite3")
    with pytest.raises(ValueError, match="不同调用账本"):
        MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=other)
    assert not other.snapshot()["stopped"]  # App does not own this resource.
    original.close(); other.close()


def test_lost_ledger_recreated_at_same_path_cannot_reset_existing_run(tmp_path):
    path = tmp_path / "budget.sqlite3"
    original = guard(path)
    app = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=original)
    app.start(); app.close(); original.close()
    path.unlink()  # Only a local empty software fixture.
    replacement = guard(path)
    with pytest.raises(ValueError, match="不同调用账本"):
        MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=replacement)
    replacement.close()


def test_forget_retains_shared_accounting_identity_without_chat_content(tmp_path):
    original = guard(tmp_path / "first.sqlite3")
    app = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=original)
    app.start()
    assert app.forget_run()["ok"]
    retained = json.loads((app.run_dir / "model_allowance.json").read_text())
    assert set(retained) == {"ledger_id", "policy_sha256"}
    assert not app.run_path.exists()
    app.close()
    other = guard(tmp_path / "second.sqlite3")
    with pytest.raises(ValueError, match="不同调用账本"):
        MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=other)
    restored = MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=original)
    restored.start(); restored.close()
    other.close(); original.close()


def test_post_transport_json_failure_halts_even_without_web(tmp_path, monkeypatch):
    from tests.test_online_json import MESSAGES
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    stub(monkeypatch, receipt(c))  # Text cannot satisfy the integrated JSON protocol.
    with g.scope("turn", "op1", "session1"):
        with pytest.raises(ValueError):
            c.complete(MESSAGES)
    assert g.snapshot()["stopped"] and g.snapshot()["stop_reason"] == "model_output_failed"
    assert g.snapshot()["known_cost_usd"] == "0.00001"
    g.close()


def test_explicit_limits_cannot_be_ignored_with_shared_guard(tmp_path):
    shared = guard(tmp_path / "budget.sqlite3")
    with pytest.raises(ValueError, match="显式限额"):
        MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=shared,
                      model_max_requests=30, model_stop_cost_usd="0.5")
    modified = policy()
    modified["models"][DEFAULT_MEMORY_MODEL]["input_usd_per_million"] = "4"
    with pytest.raises(ValueError, match="显式费用预留策略"):
        MinimalWebApp(tmp_path / "app", clients=clients(), model_call_guard=shared,
                      model_reservation_policy=modified)
    shared.close()


def test_fixed_request_fees_and_unknown_surcharges_cannot_be_silently_omitted():
    for field in ("request", "unexpected_fixed_fee"):
        data = inputs(); data[2]["data"]["endpoints"][0]["pricing"][field] = "0.01"
        with pytest.raises(ValueError, match="unreserved"):
            derive(data)


def test_high_cache_write_price_is_reserved_and_plain_price_recorded():
    data = inputs()
    pricing = data[2]["data"]["endpoints"][0]["pricing"]
    pricing.update(input_cache_write="0.0000025", input_cache_write_1h="0.000004", web_search="0.01")
    prepared, proof = derive(data)
    assert prepared["models"][DEFAULT_MEMORY_MODEL]["input_usd_per_million"] == "4"
    assert proof[1]["plain_input_usd_per_million"] == "2.000000"
    assert proof[1]["reserved_input_usd_per_million"] == "4.000000"


def test_success_response_read_is_bounded_and_oversized_result_is_unknown(tmp_path, monkeypatch):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    reads = raw_response(monkeypatch, b" " * (MAX_MODEL_RESPONSE_BYTES+2))
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    assert reads == [MAX_MODEL_RESPONSE_BYTES+1]
    state = g.snapshot()
    assert state["stopped"] and state["unknown_cost_requests"] == 1
    assert state["attempts"][0]["validation_error"] == "model_response_body_limit"
    g.close()


@pytest.mark.parametrize("billing_duplicate", [True, False])
def test_duplicate_json_is_rejected_without_silently_choosing_a_fee(tmp_path, monkeypatch, billing_duplicate):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    raw = json.dumps(receipt(c))
    if billing_duplicate:
        raw = raw.replace('"cost": 1e-05', '"cost": 0.1, "cost": 0')
    else:
        raw = raw.replace('"finish_reason": "stop"', '"finish_reason": "length", "finish_reason": "stop"')
    raw_response(monkeypatch, raw.encode())
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot()
    assert state["attempts"][0]["validation_error"] == "duplicate_response_field"
    assert state["unknown_cost_requests"] == int(billing_duplicate)
    if not billing_duplicate:
        assert state["known_cost_usd"] == "0.00001"
    assert state["stopped"]
    g.close()


@pytest.mark.parametrize("message", [{"content": ""}, {"content": 123}, {"content": "text", "refusal": "refused"},
                                    {"content": "text", "tool_calls": [{"function": {"name": "unexpected"}}]}])
def test_invalid_success_message_is_stopped_in_guard_not_only_web(tmp_path, monkeypatch, message):
    g = guard(tmp_path / "calls.sqlite3"); c = client(); g.attach(c)
    data = receipt(c); data["choices"][0]["message"] = message
    stub(monkeypatch, data)
    with pytest.raises(MinimalCallStopped):
        call(g, c)
    state = g.snapshot()
    assert state["stopped"] and state["known_cost_usd"] == "0.00001"
    g.close()


def test_default_minimal_endpoint_does_not_follow_ambient_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://unrelated.invalid")
    selection = create_minimal_clients(model_calls_enabled=True, api_key="synthetic-test-only")
    assert {c.base_url for c in (selection.online, selection.memory, selection.reflection)} == {OPENROUTER_BASE_URL}
