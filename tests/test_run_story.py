"""Driver checks use fictional software responses; no model effects are measured."""
from copy import deepcopy
from decimal import Decimal
import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from aria.runtime import MinimalModelClients
from aria.voice.minimal import MinimalWebApp
from scripts import run_story as driver
from scripts.run_safety_set import safety_story
from tests._model_http_stub import stub_model_http
from tests.test_minimal_call_integration import DECISION, Response, app_at, responses


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr("aria._env.load_local_env", lambda: pytest.fail(".env forbidden"))


def small_story():
    return {"sessions": [
        {"session_id": "first", "turns": ["今天想聊聊风车。", "纸可以折成翅膀。", "现在转起来了。"]},
        {"session_id": "second", "turns": ["今天又来试一试。", "慢慢折就可以。", "今天先聊到这里。"]},
    ]}


def records(out):
    return [json.loads(line) for path in sorted(out.glob("*/*/turns.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()]


def offline_responses():
    return [Response('{"episodes":[],"relationships":[],"cognitive_privacy":[]}', {"cost": .00001}),
            Response('{"protocol_version":"2.0","reviews":[],"changes":[]}', {"cost": .00001})]


def online_response():
    return Response(deepcopy(DECISION), {"cost": .00001})


def invitation_candidate(identifier, created_at, **changes):
    return {"id": identifier, "created_at": created_at, "status": "proposed",
            "can_confirm": True, "quotes": [{"session_id": "review-a"}],
            "do": "虚构说明，内容不用于选择", **changes}


def invitation_runner(preferences):
    class App:
        _sessions = {}
        def __init__(self):
            self.calls = []
            self.current = {"profile": {"stated_preferences": deepcopy(preferences)},
                            "turns": [], "reviews": [], "model_usage": {}}
        def state(self):
            return deepcopy(self.current)
        def preference_invitation(self, target, session_id):
            self.calls.append(("invite", target, session_id))
            self.current["preference_invitation"] = {
                "id": "software-invitation", "question_sha256": "software-question-hash",
                "status": "awaiting_display"}
            return {"ok": True, "selected_id": target, "state": self.state()}
        def preference_invitation_ack(self, invitation_id, question_sha256, session_id):
            self.calls.append(("ack", invitation_id, question_sha256, session_id))
            self.current["preference_invitation"]["status"] = "awaiting_answer"
            return {"ok": True, "state": self.state()}
    app, stream = App(), StringIO()
    summary = {"failed_operations": 0, "failed_reviews": 0}
    runner = driver.StoryRun(app, summary, stream, [])
    runner.repeat, runner.session, runner.label, runner.session_id = 1, 2, "s2", "current-session"
    runner.bindings["source"] = "review-a"
    return runner, app, stream


INVITE = {"op": "invite_preference", "source_review": "source", "target_slot": "primary"}


@pytest.mark.parametrize("kind,action", [
    ("confirm_preference", "confirm"), ("decline_preference", "decline"),
    ("revoke_preference", "withdraw"),
])
def test_missing_preference_target_is_recorded_without_call_or_guess(kind, action):
    preferences = [invitation_candidate("later-unbound", "2026-10-24T00:00:00+00:00")]
    runner, app, stream = invitation_runner(preferences)
    def apply(action, target):
        app.calls.append((action, target))
        return {"ok": True}
    app.preference_action = apply
    original_bindings = dict(runner.bindings)
    runner.step({"op": kind, "target_slot": "primary"})
    first = json.loads(stream.getvalue().splitlines()[0])
    assert first["result"] == {"ok": False, "error": "preference_target_missing"}
    assert first["failed"] and first["preferences"] == preferences
    assert app.calls == [] and runner.bindings == original_bindings
    assert runner.summary["failed_operations"] == 1
    assert runner.summary["stop_reason"] is None
    # A separate already-bound operation still runs once; no missing slot is repaired.
    runner.bindings["existing"] = "original-target"
    runner.step({"op": kind, "target_slot": "existing"})
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == 2 and rows[1]["ok"] and not rows[1]["failed"]
    assert app.calls == [(action, "original-target")]
    assert "primary" not in runner.bindings and runner.summary["failed_operations"] == 1


@pytest.mark.parametrize("kind", ["confirm_preference", "decline_preference", "revoke_preference"])
def test_missing_preference_target_preserves_existing_fee_stop(kind):
    runner, app, stream = invitation_runner([])
    app.current["model_usage"] = {"stopped": True, "stop_reason": "provider_cost_unknown",
                                  "unknown_cost_requests": 1}
    app.preference_action = lambda *args: pytest.fail("missing target must not reach product")
    runner.step({"op": kind, "target_slot": "primary"})
    assert runner.summary["failed_operations"] == 1
    assert runner.summary["stop_reason"] == "provider_cost_unknown"
    assert json.loads(stream.getvalue())["error"] == "preference_target_missing"


@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_candidates_select_chronologically_and_keep_every_candidate(reverse):
    # 10:00+08 is earlier than 03:00Z, despite their lexical order.
    preferences = [invitation_candidate("pref-a", "2026-10-22T03:00:00+00:00"),
                   invitation_candidate("pref-z", "2026-10-22T10:00:00+08:00")]
    if reverse:
        preferences.reverse()
    runner, app, stream = invitation_runner(preferences)
    runner.step(INVITE)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert app.calls[0] == ("invite", "pref-z", "current-session")
    assert app.calls[1][0] == "ack"
    assert runner.bindings["primary"] == "pref-z"
    assert [row["operation"]["op"] for row in rows] == ["invite_preference", "invitation_display_ack"]
    assert all(row["ok"] and row["preferences"] == preferences for row in rows)
    assert app.state()["profile"]["stated_preferences"] == preferences
    assert all(p["status"] == "proposed" for p in preferences)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_creation_instant_uses_id_not_list_order_or_content(reverse):
    preferences = [invitation_candidate("pref-z", "2026-10-22T02:00:00+00:00", do="看似更合题意"),
                   invitation_candidate("pref-a", "2026-10-22T10:00:00+08:00", do="不可按题意择优")]
    if reverse:
        preferences.reverse()
    runner, app, _ = invitation_runner(preferences)
    runner.step(INVITE)
    assert app.calls[0] == ("invite", "pref-a", "current-session")
    assert app.state()["profile"]["stated_preferences"] == preferences


def test_candidate_source_status_and_confirmability_filters_are_preserved():
    preferences = [
        invitation_candidate("confirmed", "2026-10-01T00:00:00+00:00", status="confirmed"),
        invitation_candidate("withdrawn", "2026-10-01T00:00:00+00:00", status="withdrawn"),
        invitation_candidate("not-admissible", None, can_confirm=False),
        invitation_candidate("other-review", "2026-10-01T00:00:00+00:00", quotes=[{"session_id": "other"}]),
        invitation_candidate("selected", "2026-10-22T00:00:00+00:00"),
    ]
    runner, app, stream = invitation_runner(preferences)
    runner.step(INVITE)
    assert app.calls[0] == ("invite", "selected", "current-session")
    assert json.loads(stream.getvalue().splitlines()[0])["preferences"] == preferences


def test_no_candidate_is_recorded_without_stopping_later_operations():
    runner, app, stream = invitation_runner([])
    runner.step(INVITE)
    first = json.loads(stream.getvalue().splitlines()[0])
    assert not first["ok"] and first["error"] == "preference_target_missing_or_ambiguous"
    assert app.calls == [] and "primary" not in runner.bindings
    assert runner.summary["failed_operations"] == 1 and runner.summary["stop_reason"] is None
    app.current["profile"]["stated_preferences"] = [
        invitation_candidate("later", "2026-10-22T00:00:00+00:00")]
    runner.step(INVITE)
    assert app.calls[0] == ("invite", "later", "current-session")
    assert runner.summary["failed_operations"] == 1


@pytest.mark.parametrize("mutation", ["missing_time", "none_time", "bad_time", "date_only",
                                     "naive_time", "missing_id", "none_id", "empty_id", "numeric_id"])
def test_missing_or_invalid_order_metadata_never_guesses_a_candidate(mutation):
    uncertain = invitation_candidate("pref-a", "2026-10-21T00:00:00+00:00")
    if mutation == "missing_time": del uncertain["created_at"]
    if mutation == "none_time": uncertain["created_at"] = None
    if mutation == "bad_time": uncertain["created_at"] = "not-a-time"
    if mutation == "date_only": uncertain["created_at"] = "2026-10-21"
    if mutation == "naive_time": uncertain["created_at"] = "2026-10-21T00:00:00"
    if mutation == "missing_id": del uncertain["id"]
    if mutation == "none_id": uncertain["id"] = None
    if mutation == "empty_id": uncertain["id"] = ""
    if mutation == "numeric_id": uncertain["id"] = 7
    preferences = [invitation_candidate("valid", "2026-10-22T00:00:00+00:00"), uncertain]
    runner, app, stream = invitation_runner(preferences)
    runner.step(INVITE)
    row = json.loads(stream.getvalue().splitlines()[0])
    assert not row["ok"] and row["error"] == "preference_target_metadata_invalid"
    assert app.calls == [] and "primary" not in runner.bindings
    assert row["preferences"] == preferences
    assert runner.summary["stop_reason"] is None


def test_two_sessions_three_turns_continue_after_second_model_failure(tmp_path, monkeypatch, capsys):
    calls = responses(monkeypatch, online_response(), Response("invalid JSON", {"cost": .00001}),
        online_response(), *offline_responses(), *(online_response() for _ in range(3)), *offline_responses())
    apps = []
    def factory(path):
        app = app_at(path)
        apps.append(app)
        return app
    out = tmp_path / "run"
    summary = driver.run_story(small_story(), 1, out, app_factory=factory)
    rows = records(out)
    turns = [row for row in rows if row["turn_id"]]
    assert len(calls) == 10
    assert summary["turns"] == 6 and summary["failed_turns"] == 1
    assert summary["sessions"] == 2 and summary["completed_repeats"] == 1
    assert summary["stop_reason"] is None and summary["unexecuted_turns"] == 0
    assert turns[1]["ok"] is False and turns[1]["source"] == "rule_fallback"
    assert turns[1]["delivered_response"]
    assert all(row["ok"] for row in turns[2:])
    assert [row["elder_text"] for row in turns] == sum((s["turns"] for s in small_story()["sessions"]), [])
    assert len([row for row in rows if row["operation"]["op"] == "reflect"]) == 2
    assert all(row["review"]["status"] == "completed" for row in rows if row["operation"]["op"] == "reflect")
    assert Decimal(summary["fees"]["known_cost_usd"]) == Decimal(".00010")
    actual = apps[0].state()["turns"]
    assert [row["delivered_response"] for row in turns] == [row["response"] for row in actual]
    assert all("dropped_fields" in row and "preferences" in row for row in turns)
    assert capsys.readouterr().out == ""
    assert not list(out.rglob("*.http")) and not list(out.rglob("*response-body*"))


def test_unknown_cost_stops_without_replacing_missing_turns(tmp_path, monkeypatch):
    calls = responses(monkeypatch, Response(deepcopy(DECISION), {}))
    summary = driver.run_story(small_story(), 1, tmp_path / "run", app_factory=app_at)
    assert len(calls) == summary["turns"] == 1
    assert summary["failed_turns"] == 1 and summary["unexecuted_turns"] == 5
    assert summary["stop_reason"] == "provider_cost_unknown"
    assert summary["fees"]["unknown_cost_requests"] == 1


def test_request_limit_stops_all_remaining_sessions(tmp_path, monkeypatch):
    calls = responses(monkeypatch, online_response(), online_response())
    summary = driver.run_story(small_story(), 1, tmp_path / "run",
                               app_factory=lambda path: app_at(path, maximum=2))
    assert len(calls) == 2 and summary["turns"] == 2 and summary["failed_turns"] == 0
    assert summary["stop_reason"] == "request_limit_reached"
    assert summary["unexecuted_turns"] == 4


def test_one_product_allowance_covers_fresh_repeats(tmp_path, monkeypatch):
    from aria.engine.policy.product_calls import ProductCallGuard
    from aria.runtime import create_minimal_clients
    from _reservation_fixtures import fixture_reservation_policy
    guard = ProductCallGuard(tmp_path / "shared.sqlite3", max_requests=11,
        max_observed_cost_usd="1", reservation_policy=fixture_reservation_policy())
    calls = responses(monkeypatch, *(online_response() for _ in range(3)), *offline_responses(),
        *(online_response() for _ in range(3)), *offline_responses(), online_response())
    def factory(path):
        clients = create_minimal_clients(model_calls_enabled=True, api_key="fixture-inert-key",
                                        base_url="https://fixture.invalid/v1")
        return MinimalWebApp(path, clients=clients, model_call_guard=guard)
    try:
        summary = driver.run_story(small_story(), 2, tmp_path / "run", app_factory=factory)
        assert len(calls) == summary["fees"]["calls_used"] == 11
        assert summary["completed_repeats"] == 1 and summary["turns"] == 7
        assert summary["stop_reason"] == "request_limit_reached"
        assert Decimal(summary["fees"]["known_cost_usd"]) == Decimal(".00011")
    finally:
        guard.close()


def test_nonfinancial_product_pause_does_not_end_batch(tmp_path, monkeypatch):
    calls = responses(monkeypatch, *(Response("invalid JSON", {"cost": .00001}) for _ in range(3)))
    summary = driver.run_story(small_story(), 1, tmp_path / "run", app_factory=app_at)
    assert len(calls) == 3 and summary["turns"] == 6
    assert summary["failed_turns"] == 6 and summary["stop_reason"] is None
    assert summary["fees"]["stopped"] and summary["fees"]["stop_reason"] == "consecutive_failures"


def test_review_failure_is_recorded_and_next_session_is_attempted(tmp_path, monkeypatch):
    calls = responses(monkeypatch, *(online_response() for _ in range(3)),
        Response("invalid JSON", {"cost": .00001}), *(online_response() for _ in range(3)), *offline_responses())
    summary = driver.run_story(small_story(), 1, tmp_path / "run", app_factory=app_at)
    rows = records(tmp_path / "run")
    assert summary["sessions"] == 2 and summary["turns"] == 6
    reviews = [row for row in rows if row["operation"]["op"] == "reflect"]
    assert len(reviews) == 2 and not reviews[0]["ok"]
    assert reviews[0]["review"]["status"] == "failed"
    assert summary["stop_reason"] is None
    assert len(calls) >= 4


def test_dropped_optional_fields_use_current_product_trace(tmp_path):
    class Client:
        model = "software-fixture"
        source_id = "fixture"
        online_decision_version = "online_turn_decision_v7"
        def complete(self, messages):
            return json.dumps({"response_segments": [{"text": "我在听。"}],
                "safety_concern": "none", "uncertainty": "invalid",
                "safety_evidence": {"quote": "", "subject": "not_applicable",
                                    "time": "not_applicable", "urgency": "not_applicable"}}, ensure_ascii=False)
    online = Client()
    clients = MinimalModelClients(online, Client(), Client(), model_calls_enabled=True)
    story = {"sessions": [{"session_id": "s1", "operations": [
        {"op": "start_session"}, {"op": "say", "text": "今天过得不错。"}, {"op": "seal"}]}]}
    driver.run_story(story, 1, tmp_path / "run",
                     app_factory=lambda path: MinimalWebApp(path, clients=clients))
    row = next(row for row in records(tmp_path / "run") if row["turn_id"])
    assert row["ok"] and row["delivered_response"] == "我在听。"
    assert row["dropped_fields"] == ["uncertainty"]


def test_fresh_repeat_profiles_and_existing_output_are_not_reused(tmp_path):
    out = tmp_path / "run"
    summary = driver.run_story(small_story(), 2, out)
    assert summary["turns"] == 12 and summary["completed_repeats"] == 2
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in out.glob("*/data/local_preview/run.json")]
    assert len({run["run_id"] for run in runs}) == 2
    before = (out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        driver.run_story(small_story(), 1, out)
    assert (out / "summary.json").read_bytes() == before


def test_addendum_is_applied_in_memory_and_hash_mismatch_is_rejected(tmp_path):
    story = {"repetitions": [{"repeat_id": "r1", "sessions": [
        {"session_id": "s1", "operations": [{"op": "start_session"}, {"op": "seal"}]}]}]}
    source = tmp_path / "materials.private.json"
    source.write_text(json.dumps(story), encoding="utf-8")
    original = source.read_bytes()
    before = story["repetitions"][0]["sessions"][0]["operations"]
    after = [before[0], {"op": "say", "text": "虚构台词"}, before[1]]
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    patch = {"base_materials_sha256": hashlib.sha256(original).hexdigest(), "patches": [
        {"repeat_id": "r1", "session_id": "s1", "original_operations_sha256": digest(before),
         "effective_operations_sha256": digest(after), "effective_operations": after}]}
    (tmp_path / "operations-order-addendum-v1.private.json").write_text(json.dumps(patch), encoding="utf-8")
    assert driver.load_story(source)["repetitions"][0]["sessions"][0]["operations"] == after
    assert source.read_bytes() == original
    source.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="base_mismatch"):
        driver.load_story(source)


def test_sharing_uses_actual_product_selection_and_revoke_blocks_second_send(tmp_path):
    story = {"sessions": [{"session_id": "s1", "operations": [
        {"op": "start_session"}, {"op": "say", "turn_id": "t1", "text": "今天愿意一起折纸。"},
        {"op": "sharing_preview", "mode": "local_mock_only", "bind_artifact_as": "share",
         "selection": {"source": "s1.t1.exact_user_selected_quote", "text": "愿意一起折纸"},
         "recipient": {"kind": "local_mock_inbox", "id": "local", "address": None}},
        {"op": "sharing_approve", "artifact_ref": "share"},
        {"op": "sharing_send", "artifact_ref": "share"},
        {"op": "sharing_revoke", "artifact_ref": "share"},
        {"op": "sharing_send", "artifact_ref": "share"},
        {"op": "say", "text": "继续说说今天的事。"}, {"op": "seal"}]}]}
    summary = driver.run_story(story, 1, tmp_path / "run")
    rows = records(tmp_path / "run")
    sent = [row for row in rows if row["operation"]["op"] == "sharing_send"]
    assert sent[0]["ok"] and sent[0]["result"]["record"]["status"] == "delivered_local"
    assert not sent[1]["ok"] and summary["turns"] == 2
    preview = next(row for row in rows if row["operation"]["op"] == "sharing_preview")
    assert preview["result"]["record"]["message"] == "愿意一起折纸"
    assert preview["result"]["local_contact"]["channel"] == "local"


def test_safety_loads_original_context_and_continues_single_message_cases(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"cases": [
        {"regression_id": "F1", "execution_input": {"messages": [{"role": "user", "content": "第一句"}]}},
        {"regression_id": "F2", "execution_input": {"messages": [
            {"role": "assistant", "content": "旧语境"}, {"role": "user", "content": "本次输入"}]}},
        {"regression_id": "F3", "execution_input": {"messages": [{"role": "user", "content": "第三句"}]}},
    ]}), encoding="utf-8")
    summary = driver.run_story(safety_story(path), 1, tmp_path / "run")
    assert summary["planned_turns"] == 3 and summary["sessions"] == 3
    assert summary["turns"] == 3 and summary["unexecuted_turns"] == 0
    assert summary["failed_operations"] == 0
    assert [row["elder_text"] for row in records(tmp_path / "run") if row["turn_id"]] == ["第一句", "本次输入", "第三句"]
    context_rows = [r for r in records(tmp_path / "run") if r["operation"]["op"] == "load_fictional_context"]
    assert all(row["ok"] for row in context_rows)
    assert context_rows[1]["result"]["fictional_input"]["messages"] == [
        {"role": "assistant", "content": "旧语境"}, {"role": "user", "content": "本次输入"}]


def test_fictional_context_preserves_actual_wire_and_no_persona_seeding(tmp_path, monkeypatch):
    calls = responses(monkeypatch, online_response())
    app = app_at(tmp_path)
    try:
        sid = app.start()["state"]["session_id"]
        original = {"messages": [{"role": "user", "content": "  历史甲\n保留空白。 "},
            {"role": "user", "content": "历史乙。"}, {"role": "assistant", "content": "旧回复。"},
            {"role": "user", "content": "  这次聊风车。\n"}],
            "setup": {"language": "zh-CN", "today": "2026-10-12"}, "context_material": []}
        before = app._profile().model_dump(mode="json")
        assert app.load_fictional_context(original, sid, fictional=True)["ok"]
        assert calls == [] and app._profile().model_dump(mode="json") == before
        assert app.state()["turns"] == []
        assert app.submit(original["messages"][-1]["content"], "actual", sid)["ok"]
        carrier = json.loads(calls[0]["messages"][-1]["content"])
        assert carrier["conversation_history"] == original["messages"][:-1]
        assert carrier["current_user_message"] == original["messages"][-1]["content"]
        assert "2026-10-12" in calls[0]["messages"][0]["content"]
        assert app.state()["sessions"][0]["fictional_input"] == original
        assert len(app.state()["turns"]) == 1
        assert not app.submit("另一句", "different", sid)["ok"]
        assert len(calls) == 1
    finally:
        app.close()


def test_safety_prelude_is_logged_before_stub_model_and_keeps_original_context(tmp_path, monkeypatch):
    from tests.test_premodel_web_stream import opts, response, TEXT, TASK
    out, calls = tmp_path / "run", []
    original = {"messages": [{"role": "assistant", "content": "虚构历史回复。"},
        {"role": "user", "content": TEXT}], "setup": {"today": "2026-10-12", "language": "zh-CN"}}
    def receive(request, **kwargs):
        wire = json.loads(request.data)
        calls.append(wire)
        before = records(out)
        assert before[-1]["operation"]["op"] == "safety_prelude" and before[-1]["delivered_response"]
        carrier = json.loads(wire["messages"][-1]["content"])
        assert carrier["conversation_history"] == original["messages"][:-1]
        assert carrier["current_user_message"] == TEXT
        assert "2026-10-12" in wire["messages"][0]["content"]
        return response(request)
    stub_model_http(monkeypatch, receive)
    summary = driver.run_story({"sessions": [{"session_id": "safety", "operations": [
        {"op": "start_session"}, {"op": "say", "text": TEXT, "execution_input": original}, {"op": "seal"}]}]},
        1, out, app_factory=lambda path: MinimalWebApp(path, **opts()))
    turn = next(row for row in records(out) if row["turn_id"])
    assert len(calls) == summary["turns"] == 1 and summary["failed_operations"] == 0
    assert turn["ok"] and turn["delivered_response"].endswith(TASK)


@pytest.mark.parametrize("options", [{}, {"fictional": False}, {"fictional": True, "read_only": True},
                                    {"fictional": True, "actor": "caregiver"}])
def test_fictional_context_is_explicit_owner_only(tmp_path, options):
    app = MinimalWebApp(tmp_path)
    try:
        sid = app.start()["state"]["session_id"]
        before = app.state()
        assert not app.load_fictional_context({"messages": [{"role": "user", "content": "当前原句"}]}, sid, **options)["ok"]
        assert app.state() == before
    finally:
        app.close()


@pytest.mark.parametrize("mutation", ["role", "date", "language", "material", "changed", "stale", "after_turn"])
def test_invalid_or_changed_fictional_context_is_rejected_without_partial_binding(tmp_path, mutation):
    from aria.engine.policy.fictional_context import HISTORY_KEY, KEY
    app = MinimalWebApp(tmp_path)
    try:
        sid = app.start()["state"]["session_id"]
        value = {"messages": [{"role": "user", "content": "当前原句"}], "setup": {"today": "2026-10-12"}}
        if mutation == "role": value["messages"][0]["role"] = "system"
        if mutation == "date": value["setup"]["today"] = "not-a-date"
        if mutation == "language": value["setup"]["language"] = "unrecognized"
        if mutation == "material": value["context_material"] = ["unsupported"]
        if mutation == "changed":
            assert app.load_fictional_context(deepcopy(value), sid, fictional=True)["ok"]
            value["messages"][0]["content"] = "换过的句子"
        if mutation == "after_turn": assert app.submit("已发生对话", "first", sid)["ok"]
        before = deepcopy(app._sessions["fictional-chen"][1].working_memory)
        result = app.load_fictional_context(value, "wrong" if mutation == "stale" else sid, fictional=True)
        assert not result["ok"]
        after = app._sessions["fictional-chen"][1].working_memory
        assert after.get(HISTORY_KEY) == before.get(HISTORY_KEY) and after.get(KEY) == before.get(KEY)
    finally:
        app.close()


def test_all_fixed_87_inputs_are_loadable_and_submitted_once(tmp_path):
    from scripts.run_safety_set import DEFAULT_INPUTS
    original = driver.read_json(DEFAULT_INPUTS)
    before = DEFAULT_INPUTS.read_bytes()
    summary = driver.run_story(safety_story(DEFAULT_INPUTS), 1, tmp_path / "run")
    rows = records(tmp_path / "run")
    inputs = [r for r in rows if r["operation"]["op"] == "load_fictional_context"]
    turns = [r for r in rows if r["turn_id"]]
    assert len(inputs) == len(turns) == summary["planned_turns"] == summary["turns"] == 87
    assert all(row["ok"] for row in inputs)
    assert {row["session"] for row in turns} == {c["regression_id"] for c in original["cases"]}
    bound = {row["session"]: row["result"]["fictional_input"] for row in inputs}
    assert bound == {case["regression_id"]: case["execution_input"] for case in original["cases"]}
    assert summary["unexecuted_turns"] == 0 and DEFAULT_INPUTS.read_bytes() == before


def test_safety_cases_have_fresh_products_without_offline_queue_overflow(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"cases": [
        {"regression_id": f"F{i}", "execution_input": {"messages": [{"role": "user", "content": "虚构测试句。"}]}}
        for i in range(12)]}), encoding="utf-8")
    summary = driver.run_story(safety_story(path), 1, tmp_path / "run")
    assert summary["turns"] == 12 and summary["failed_turns"] == 0
    assert summary["failed_operations"] == 0 and summary["unexecuted_turns"] == 0
    for row in records(tmp_path / "run"):
        assert row["preferences"] == []


def test_cli_preview_never_loads_env_and_limits_do_not_enable_models(tmp_path, capsys):
    path = tmp_path / "story.json"
    path.write_text(json.dumps(small_story()), encoding="utf-8")
    assert driver.main(["--story", str(path), "--out", str(tmp_path / "run")]) == 0
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit) as error:
        driver.main(["--story", str(path), "--out", str(tmp_path / "other"), "--model-max-requests", "2"])
    assert error.value.code == 2 and not (tmp_path / "other").exists()


def test_scripts_are_small_and_do_not_import_historical_executors():
    for module in (driver, __import__("scripts.run_safety_set", fromlist=["__file__"])):
        text = Path(module.__file__).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 300
        assert "artifacts/local" not in text and "load_local_env" not in text


def _segment_material(tmp_path, safety):
    texts = [["first"], ["failed receipt", "missing in interrupted repeat"], ["third"], ["fourth"]]
    if safety:
        material = {"cases": [{"regression_id": f"F{i}", "execution_input": {
            "messages": [{"role": "user", "content": turns[0]}]}} for i, turns in enumerate(texts)]}
    else:
        material = {"repetitions": [{"sessions": [{"session_id": "s", "operations": [
            {"op": "start_session"}, *({"op": "say", "text": t} for t in turns), {"op": "seal"}]}]}
            for turns in texts]}
    story = tmp_path / "story.json"
    story.write_text(json.dumps(material), encoding="utf8")
    return ["--story", str(story), "--repeats", "1" if safety else "4"]


@pytest.mark.parametrize("safety", [False, True])
def test_clean_segment_uses_only_next_independent_units_and_preserves_fixed_denominator(tmp_path, safety):
    common = _segment_material(tmp_path, safety)
    run = (lambda argv: driver.main(argv, story_loader=safety_story)) if safety else driver.main
    out = tmp_path / "batch"
    assert run([*common, "--out", str(out), "--unit-limit", "1"]) == 3
    before = {f: f.read_bytes() for f in out.rglob("*") if f.is_file()}
    after = tmp_path / "batch.part2"
    evidence = next(out.glob("*/*/turns.jsonl"))
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        run([*common, "--continue-from", str(out), "--out", str(after)])
    assert not after.exists()
    evidence.write_bytes(before[evidence])
    story = Path(common[1])
    original = story.read_bytes()
    story.write_bytes(original.replace(b"first", b"other"))
    with pytest.raises(SystemExit):
        run([*common, "--continue-from", str(out), "--out", str(after)])
    assert not after.exists()
    story.write_bytes(original)
    assert run([*common, "--continue-from", str(out), "--out", str(after)]) == 0
    result = driver.read_json(after / "summary.json")
    assert result["complete_units"] == [1, 2, 3, 4] and result["last_attempted_unit"] == 4
    assert result["planned_turns"] == result["turns"] == (4 if safety else 5)
    assert "first" not in [r["elder_text"] for r in records(after) if r["turn_id"]]
    assert all(f.read_bytes() == v for f,v in before.items())
    with pytest.raises(SystemExit):
        run([*common, "--continue-from", str(out), "--out", str(after)])


def _mock_paid_segment_guard(monkeypatch, called, *, fault=None, open_operation=False):
    """Product-driver boundary with a deterministic physical receipt ledger, never a remote call."""
    import sqlite3
    def configured(args):
        db = None
        def factory(path):
            nonlocal db
            if db is None:
                db = sqlite3.connect(args.out / "model_calls.sqlite3")
                db.execute("CREATE TABLE attempts(sequence INTEGER PRIMARY KEY,record TEXT)")
                db.execute("CREATE TABLE release_operations(finished INTEGER)")
            class FakeApp:
                _sessions = {}
                def __init__(self):
                    self.turns = []
                def state(self):
                    rows = [json.loads(v[0]) for v in db.execute("SELECT record FROM attempts")]
                    missing = sum("cost" not in r["usage"] for r in rows)
                    fees = {"calls_used": len(rows), "unknown_cost_requests": missing,
                            "known_cost_usd": str(sum((Decimal(str(r["usage"].get("cost", 0))) for r in rows), Decimal(0))),
                            "reserved_cost_usd": str(Decimal(".009") * missing),
                            "stopped": bool(missing), "stop_reason": "model_or_storage_failed" if missing else None}
                    return {"session_id": "local", "turns": self.turns, "profile": {"stated_preferences": []},
                            "reviews": [], "model_usage": fees}
                def start(self):
                    return {"ok": True, "state": self.state()}
                def submit(self, text, tid, sid, **kwargs):
                    called.append(text)
                    failed = text == "failed receipt"
                    row = {"request_id": tid, "model": "deepseek/deepseek-v4.1-flash-1017",
                           "expected_provider": "Together", "status": "failed" if failed else "received",
                           "reservation": {"reserved_cost_usd": ".009"}, "usage": {} if failed else {"cost": ".001"},
                           "started_at":"2026-11-01T00:00:00+00:00", "finished_at":"2026-11-01T00:00:30+00:00",
                           "failure_phase":"read","failure_code":"response_read_failed","http_status":200,
                           "stream_read":{"provider_error_code":504},
                           "observation": {"generation_id_candidate": "gen-test" if failed else None,
                                           "generation_id_candidate_status":"accepted"}}
                    if failed and fault:
                        row.update(fault)
                    if failed and open_operation:
                        db.execute("INSERT INTO release_operations VALUES(0)")
                    db.execute("INSERT INTO attempts(record) VALUES (?)", (json.dumps(row),))
                    db.commit()
                    self.turns.append({"turn_id": tid, "status": "completed", "response": text, "source": "fixture"})
                    return {"ok": True, "state": self.state()}
                def seal(self, sid):
                    return {"ok": True}
                def load_fictional_context(self, *args, **kw):
                    return {"ok": True}
                def close(self):
                    pass
            return FakeApp()
        return factory, lambda: db.close() if db else None
    monkeypatch.setattr(driver, "configured_factory", configured)


@pytest.mark.parametrize("safety,settled", [(False, False), (True, True)])
def test_unknown_fee_stops_then_only_bound_admission_can_run_untouched_suffix(tmp_path, monkeypatch, safety, settled):
    called = []
    _mock_paid_segment_guard(monkeypatch, called)
    policy = tmp_path / "continuation-policy.json"
    policy.write_text('{"unknown_fee":"stop","continuation":"only_settled_or_user_retained"}', encoding="utf8")
    common = [*_segment_material(tmp_path, safety), "--enable-model-calls", "--continuation-policy", str(policy)]
    run = (lambda argv: driver.main(argv, story_loader=safety_story)) if safety else driver.main
    out = tmp_path / "batch"
    assert run([*common, "--out", str(out), "--model-max-requests", "8", "--model-stop-cost-usd", ".020"]) == 2
    assert called == ["first", "failed receipt"]
    seal = out / "segment.json"
    old_sha = hashlib.sha256(seal.read_bytes()).hexdigest()
    old = driver.read_json(seal)
    rid, = old["costs"]["unknown"]
    after = tmp_path / "batch.part2"
    child_cap = ".017" if settled else ".010"
    args = [*common, "--continue-from", str(out), "--out", str(after), "--model-max-requests", "6",
            "--model-stop-cost-usd", child_cap, "--unit-limit", "1"]
    with pytest.raises(SystemExit):
        run(args)
    assert not after.exists() and called == ["first", "failed receipt"]
    evidence = tmp_path / "request-disposition.json"
    proof = {"local_request_id": rid, "generation_id_candidate": "gen-test"}
    if settled:
        proof.update(schema="aria.readonly_generation_fee_query.v1", http_status=404, method="GET", model_calls=0,
                     ledger_modified=False, response_bytes=1024, response_sha256="a"*64,
                     data={"id":"gen-test","provider_name":"Together","model":"deepseek/deepseek-v4.1-flash-20261017",
                           "is_byok":False,"usage":.002,"total_cost":.002,"created_at":"2026-11-01T00:00:01Z"})
    else:
        proof.update(schema="aria.user_hold_instruction.v1", user_message_ref="fixture:owner-message-2",
                     permission="continue_unattempted_units_keep_full_hold", parent_segment_sha256=old_sha,
                     full_hold_usd=".009")
    def dump(path, obj):
        path.write_text(json.dumps(obj), encoding="utf8")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"decision":"settled" if settled else "retain_full_hold", "full_hold_usd":".009",
             "evidence":{"path":evidence.name,"sha256":dump(evidence,proof)}}
    receipt = tmp_path / "admission.json"
    packet = {"schema":"aria.independent_segment_admission.v1","approved":True,"reviewer":"fixture reviewer",
              "parent_segment_sha256":old_sha,"policy_sha256":hashlib.sha256(policy.read_bytes()).hexdigest(),
              "destination":str(after.resolve()),"child_cap_usd":child_cap,"child_max_requests":6,
              "requests":{rid:entry}}
    dump(receipt,packet)
    args += ["--continuation-admission", str(receipt)]
    # A 404 is not a settlement, and the external hold record alone does not express live operator intent.
    with pytest.raises(SystemExit):
        run(args)
    if settled:
        proof["http_status"] = 200
        proof["data"]["created_at"] = "2026-11-02T00:00:01Z"
        entry["evidence"]["sha256"] = dump(evidence,proof)
        dump(receipt,packet)
        with pytest.raises(SystemExit):
            run(args)
        proof["data"]["created_at"] = "2026-11-01T00:00:01Z"
        entry["evidence"]["sha256"] = dump(evidence,proof)
        dump(receipt,packet)
    else:
        args += ["--accept-held-request", rid + "=0.009"]
    inflated = args.copy()
    inflated[inflated.index("--model-stop-cost-usd")+1] = ".020"
    packet["child_cap_usd"] = ".020"
    dump(receipt,packet)
    with pytest.raises(SystemExit):
        run(inflated)
    assert not after.exists() and called == ["first", "failed receipt"]
    packet["child_cap_usd"] = child_cap
    dump(receipt,packet)
    assert run(args) == 3 and called == ["first", "failed receipt", "third"]
    assert hashlib.sha256(seal.read_bytes()).hexdigest() == old_sha
    final = tmp_path / "batch.part3"
    tail = [*common, "--continue-from", str(after), "--out", str(final), "--model-max-requests","5",
            "--model-stop-cost-usd", ".017" if settled else ".010"]
    # This would fit the preceding segment by itself but overcommits the ancestor's charge/hold.
    with pytest.raises(SystemExit):
        run(tail)
    tail[-1] = ".016" if settled else ".009"
    assert run(tail) == (0 if safety else 3)
    assert called == ["first", "failed receipt", "third", "fourth"]
    result = driver.read_json(final / "summary.json")
    assert result["complete_units"] == [1,3,4] and result["turns"] == 4
    assert result["planned_turns"] == (4 if safety else 5)
    assert result["unexecuted_turns"] == (0 if safety else 1)
    assert len(driver.read_json(final / "segment.json")["parents"]) == 2


@pytest.mark.parametrize("fault,open_op", [
    ({"failure_phase":"receipt_validation","failure_code":"response_provider_mismatch"},False),
    ({"returned_provider":"unexpected"},False),
    ({"fee_verification_error":"duplicate_response_identity"},False),
    ({},True),
])
def test_explicit_hold_cannot_bypass_other_hard_stops(tmp_path, monkeypatch, fault, open_op):
    called = []
    _mock_paid_segment_guard(monkeypatch,called,fault=fault,open_operation=open_op)
    policy = tmp_path / "policy.json"
    policy.write_text("{}",encoding="utf8")
    common = [*_segment_material(tmp_path,False),"--enable-model-calls","--continuation-policy",str(policy)]
    out = tmp_path / "run"
    assert driver.main([*common,"--out",str(out),"--model-stop-cost-usd",".02","--model-max-requests","8"]) == 2
    seg = driver.read_json(out/"segment.json")
    rid,row = next(iter(seg["costs"]["unknown"].items()))
    parent_sha = hashlib.sha256((out/"segment.json").read_bytes()).hexdigest()
    child = tmp_path / "run.part2"
    instruction = tmp_path / "owner.json"
    instruction.write_text(json.dumps({"schema":"aria.user_hold_instruction.v1","user_message_ref":"fixture:owner-2",
        "permission":"continue_unattempted_units_keep_full_hold","parent_segment_sha256":parent_sha,
        "full_hold_usd":row["hold"],"local_request_id":rid,"generation_id_candidate":"gen-test"}),encoding="utf8")
    grant = tmp_path/"grant.json"
    grant.write_text(json.dumps({"schema":"aria.independent_segment_admission.v1","approved":True,"reviewer":"fixture",
        "parent_segment_sha256":parent_sha,"policy_sha256":hashlib.sha256(policy.read_bytes()).hexdigest(),
        "destination":str(child.resolve()),"child_cap_usd":".01","child_max_requests":6,"requests":{rid:{
            "decision":"retain_full_hold","full_hold_usd":row["hold"],"evidence":{"path":instruction.name,
            "sha256":hashlib.sha256(instruction.read_bytes()).hexdigest()}}}}),encoding="utf8")
    with pytest.raises(SystemExit):
        driver.main([*common,"--out",str(child),"--continue-from",str(out),"--continuation-admission",str(grant),
                     "--accept-held-request",f"{rid}={row['hold']}","--model-stop-cost-usd",".01","--model-max-requests","6"])
    assert not child.exists() and called == ["first","failed receipt"]
