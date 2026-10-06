"""Real Engine/client assembly with synthetic responses, never real models."""
from copy import deepcopy
import hashlib
import json
import urllib.request

import pytest

from aria import cli
from aria.runtime import create_minimal_clients
from aria.voice.minimal import MinimalWebApp
from _reservation_fixtures import fixture_reservation_policy


DECISION = {
    "response": "我在听，可以慢慢说。", "state_signals": [],
    "current_affect": {"label": "unknown", "confidence": .3,
                       "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
    "selected_intervention_id": "", "uncertainty": .6,
    "communication_plan": {"pace": "open_listening", "support_order": "先听本人说",
                           "boundary": "不强加话题或行动"},
}


class Response:
    status = 200
    headers = {}

    def __init__(self, output, usage):
        if isinstance(output, dict):
            message = {"tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                "name": "aria_turn_decision", "arguments": json.dumps(output, ensure_ascii=False)}}]}
        else:
            message = {"content": output}
        self.data = {"id": "gen-fixtureintegration", "model": "fixture/model", "usage": usage,
                     "choices": [{"finish_reason": "stop", "message": message}]}

    def read(self, size=-1):
        raw = json.dumps(self.data, ensure_ascii=False).encode()
        return raw if size < 0 else raw[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("unstubbed network"))


def app_at(path, *, maximum=21):
    clients = create_minimal_clients(model_calls_enabled=True, api_key="fixture-inert-key",
                                     base_url="https://fixture.invalid/v1")
    return MinimalWebApp(path, clients=clients, model_max_requests=maximum, model_stop_cost_usd="1",
                         model_reservation_policy=fixture_reservation_policy())


def responses(monkeypatch, *rows):
    pending = iter(rows)
    calls = []

    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        response = next(pending)
        # The existing Engine may choose plain text for a simple turn. Match
        # the declared wire channel instead of inventing a native tool reply.
        message = response.data["choices"][0]["message"]
        if "tool_calls" in message and not payload.get("tools"):
            decision = json.loads(message["tool_calls"][0]["function"]["arguments"])
            from aria.engine.policy.online_decision import protocol_version
            if protocol_version(payload["messages"]) == "online_turn_decision_v3":
                decision["safety_concern"] = "none"
            response = Response(
                json.dumps(decision, ensure_ascii=False)
                if payload.get("response_format") == {"type": "json_object"}
                else decision["response"], response.data["usage"])
        response.data["model"] = payload["model"]
        response.data["provider"] = "Alibaba" if payload["provider"]["only"] == ["alibaba"] else "Anthropic"
        response.data["usage"].setdefault("prompt_tokens", 10)
        response.data["usage"].setdefault("completion_tokens", 12)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", receive)
    return calls


def send_one(app):
    sid = app.start()["state"]["session_id"]
    result = app.submit("这纸风车翅膀折反了，怪不得不转。", "first", sid)
    assert result["ok"], result
    return sid


def test_missing_or_invalid_live_limits_reject_before_loading_credentials(monkeypatch):
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("credentials read before validation"))
    for flags in ([], ["--model-max-requests", "21"],
                  ["--model-max-requests", "21", "--model-stop-cost-usd", "NaN"]):
        with pytest.raises(SystemExit) as result:
            cli.web_main(["--minimal", "--enable-model-calls", *flags])
        assert result.value.code == 2
    with pytest.raises(SystemExit):
        cli.web_main(["--minimal", "--model-max-requests", "21", "--model-stop-cost-usd", "1"])


def test_unknown_cost_rejects_reply_and_seal_forget_restart_cannot_reset_allowance(tmp_path, monkeypatch):
    calls = responses(monkeypatch, Response(DECISION, {}))
    app = app_at(tmp_path)
    sid = app.start()["state"]["session_id"]
    assert not app.submit("这纸风车翅膀折反了，怪不得不转。", "first", sid)["ok"]
    state = app.state()
    assert state["turns"][0]["response"] == ""
    assert state["model_usage"]["unknown_cost_requests"] == 1
    assert state["model_usage"]["stopped"] and state["requires_review"]
    assert not app.submit("再说一遍", "second", sid)["ok"]
    assert len(app.state()["turns"]) == 1
    assert app.seal(sid)["ok"]
    assert not app.reflect("review")["ok"]
    assert app.tick_care_all() == ["fictional-chen"]
    before = hashlib.sha256(app.call_guard.path.read_bytes()).hexdigest()
    for _ in range(3):
        app.state()
    assert hashlib.sha256(app.call_guard.path.read_bytes()).hexdigest() == before
    forgotten = app.forget_run()["state"]
    assert not forgotten["turns"] and forgotten["model_usage"]["calls_used"] == 1
    assert hashlib.sha256(app.call_guard.path.read_bytes()).hexdigest() == before
    app.close()
    assert app.state()["model_usage"]["calls_used"] == 1
    reopened = app_at(tmp_path)
    sid = reopened.start()["state"]["session_id"]
    assert not reopened.submit("重新开始也不能清费用", "third", sid)["ok"]
    assert len(calls) == 1
    assert "fixture-inert-key" not in json.dumps(reopened.state())
    assert "纸风车".encode() not in reopened.call_guard.path.read_bytes()
    reopened.close()


def test_three_roles_share_guard_and_explicit_review_context(tmp_path, monkeypatch):
    calls = responses(monkeypatch, *(Response(DECISION, {"cost": .01}) for _ in range(3)),
                      Response('{"episodes":[],"relationships":[],"cognitive_privacy":[]}', {"cost": .02}),
                      Response('{"protocol_version":"2.0","reviews":[],"changes":[]}', {"cost": .03}))
    app = app_at(tmp_path, maximum=5)
    sid = send_one(app)
    assert app.submit("囡囡她妈小时候，什么坏了都拿来找我。", "second", sid)["ok"]
    assert app.submit("她对我好，我知道。可有些事我自己愿意做。", "third", sid)["ok"]
    assert app.seal(sid)["ok"] and len(calls) == 3
    result = app.reflect("review-one")
    assert result["ok"], result
    usage = result["state"]["model_usage"]
    assert [r["role"] for r in usage["attempts"]] == ["online_fast"] * 3 + ["memory", "reflection"]
    assert [r["operation_id"] for r in usage["attempts"]] == ["first", "second", "third", "review-one", "review-one"]
    assert {r["session_id"] for r in usage["attempts"]} == {sid}
    assert usage["calls_used"] == len(calls) == 5 and usage["stopped"]
    assert app.reflect("review-one")["duplicate"] and len(calls) == 5
    next_sid = app.start()["state"]["session_id"]
    assert not app.submit("下一次", "next", next_sid)["ok"]
    assert app.seal(next_sid)["ok"]
    app.close()


def test_memory_protocol_failure_keeps_original_and_allows_new_conversation(tmp_path, monkeypatch):
    calls = responses(monkeypatch, Response(DECISION, {"cost": .01}), Response('{}', {"cost": .02}),
                      Response(DECISION, {"cost": .01}), Response('{"episodes":[],"relationships":[],"cognitive_privacy":[]}', {"cost": .02}))
    app = app_at(tmp_path)
    sid = send_one(app)
    app.seal(sid)
    result = app.reflect("invalid-memory")
    assert not result["ok"]
    assert not result["state"]["reviews"][0]["requires_review"]
    assert not result["state"]["model_usage"]["stopped"]
    assert result["state"]["model_usage"]["consecutive_failures"] == 1
    assert result["state"]["model_usage"]["unknown_cost_requests"] == 0
    assert result["state"]["offline"]["pending_count"] == 0
    audit = result["state"]["offline"]["audit"][-1]
    assert audit["outcome"] == "memory_failed" and not audit["memory_completed"] and not audit["reflection_completed"]
    assert not app.reflect("do-not-retry-memory")["ok"]
    assert len(calls) == 2
    new_sid = app.start()["state"]["session_id"]
    assert app.submit("换一个话题", "another", new_sid)["ok"]
    assert len(calls) == 3
    assert len(app.state()["reviews"]) == 1
    assert app.seal(new_sid)["ok"]
    assert app.reflect("new-session-memory")["ok"] and len(calls) == 4
    assert app.state()["offline"]["audit"][0] == audit
    app.close()


def test_old_live_transcript_without_accounting_cannot_start_from_zero(tmp_path):
    root = tmp_path / "live"
    root.mkdir()
    row = {"version": 1, "mode": "live", "run_id": "old", "sessions": [],
           "turns": [{"turn_id": "old"}], "reviews": []}
    (root / "run.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少调用台账"):
        app_at(tmp_path)
    assert not (root / "model_calls.sqlite3").exists()
    assert json.loads((root / "run.json").read_text()) == row


def test_unreadable_guard_stops_new_calls_but_status_and_sealing_remain_available(tmp_path, monkeypatch):
    app = app_at(tmp_path)
    sid = app.start()["state"]["session_id"]
    original = app.call_guard._snapshot
    monkeypatch.setattr(app.call_guard, "_snapshot", lambda: (_ for _ in ()).throw(OSError("private path")))
    state = deepcopy(app.state())
    usage = state["model_usage"]
    assert usage["stopped"] and usage["known_cost_usd"] is None and usage["calls_used"] is None
    assert "private path" not in json.dumps(state)
    assert not app.submit("不会发出", "blocked", sid)["ok"]
    assert app.seal(sid)["ok"]
    monkeypatch.setattr(app.call_guard, "_snapshot", original)
    assert app.state()["model_usage"]["stopped"]
    app.close()


@pytest.mark.parametrize("route", ["fast", "fast_integrated"])
@pytest.mark.parametrize("violation", ["text", "details", "tokens"])
def test_reasoning_contract_violation_preserves_fee_and_persistently_stops(route, violation, tmp_path, monkeypatch):
    secret = "synthetic-hidden-content-must-not-persist"
    calls = []
    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        assert payload["reasoning"] == {"enabled": False, "exclude": True}
        structured = payload.get("response_format") == {"type": "json_object"}
        from aria.engine.policy.online_decision import protocol_version
        is_v3 = protocol_version(payload["messages"]) == "online_turn_decision_v3"
        assert structured == (is_v3 or route == "fast_integrated")
        decision = {**DECISION, **({"safety_concern": "none"} if is_v3 else {})}
        response = Response(json.dumps(decision, ensure_ascii=False) if structured else "我在听。",
                            {"cost": .0123, "prompt_tokens": 10, "completion_tokens": 12})
        response.data["model"] = payload["model"]
        response.data["provider"] = "Alibaba"
        message = response.data["choices"][0]["message"]
        if violation == "text": message["reasoning"] = secret
        if violation == "details": message["reasoning_details"] = [{"type": "reasoning.text", "text": secret}]
        if violation == "tokens": response.data["usage"]["completion_tokens_details"] = {"reasoning_tokens": 3}
        return response
    monkeypatch.setattr(urllib.request, "urlopen", receive)
    app = app_at(tmp_path)
    sid = app.start()["state"]["session_id"]
    app._sessions["fictional-chen"][1].working_memory["_force_reply_model"] = route
    result = app.submit("今天想说说纸风车。", "first", sid)
    assert not result["ok"]
    state = result["state"]
    assert state["turns"][0]["response"] == ""
    assert state["turns"][0]["source"] == "model_failed"
    assert state["model_usage"]["calls_used"] == 1
    assert state["model_usage"]["known_cost_usd"] == "0.0123"
    assert state["model_usage"]["unknown_cost_requests"] == 0
    assert state["model_usage"]["stopped"]
    if violation == "tokens":
        assert state["model_usage"]["attempts"][0]["usage"]["reasoning_tokens"] == 3
    assert not app.submit("不能再调用", "second", sid)["ok"]
    assert len(calls) == 1
    assert secret not in json.dumps(state)
    app.forget_run()
    app.close()
    reopened = app_at(tmp_path)
    new_sid = reopened.start()["state"]["session_id"]
    assert not reopened.submit("重启也不能再调用", "third", new_sid)["ok"]
    assert reopened.state()["model_usage"]["known_cost_usd"] == "0.0123"
    assert len(calls) == 1
    reopened.close()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
