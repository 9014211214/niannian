"""Product-path E1b integration using named software fixtures, never models."""
from copy import deepcopy
import json
import re
import urllib.request

import pytest

from aria.bench.e1b_cases import load_e1b_cases
from aria.bench.e1b_runner import (
    E1bExperimentApp, E1bLocalFixtureClient, E1bRunner, E1bRunStopped,
    local_fixture_factory, strip_a0_preference_sources,
)
from aria.engine.model.persona import (
    EpisodicMemoryEntry, PersonaProfile, PersonHypothesis, PersonHypothesisEvidence,
    RelationshipContextEntry, RelationshipEvidence,
)
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.runtime import MinimalModelClients


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("E1b software tests must never use network")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def setup_runner(tmp_path, *, factory=None):
    token = SyntheticEmotionExperiment.create(tmp_path, "e1b-test")
    records = []
    app_factory = factory(token, records) if factory else local_fixture_factory(token, recordings=records)
    runner = E1bRunner(load_e1b_cases(), root=token.root / "run", app_factory=app_factory, seed=20261003)
    return runner, records


def test_strip_a0_removes_mixed_source_claim_whole_preserves_other_facts():
    profile = PersonaProfile(user_id="fictional")
    profile.episodic_memory = [
        EpisodicMemoryEntry(id="pref", summary="偏好事实", source_session_id="s2", source_turn_index=5),
        EpisodicMemoryEntry(id="other", summary="花开了", source_session_id="s2", source_turn_index=9),
    ]
    def evidence(session, turn):
        return PersonHypothesisEvidence(at="2026-10-03", session_id=session,
                                        turn_indices=[turn], relation="support", summary="逐字依据")
    profile.person_hypotheses = [
        PersonHypothesis(id="mixed", framework="response", claim="混合依据", first_observed_at="2026-10-03",
                         evidence=[evidence("s1", 0), evidence("s2", 5)]),
        PersonHypothesis(id="other", framework="response", claim="无关依据", first_observed_at="2026-10-03",
                         evidence=[evidence("s1", 5)]),
    ]
    profile.relationship_context = [RelationshipContextEntry(
        id="relation-pref", name="女儿", facts=["先问本人"], first_observed_at="2026-10-03",
        evidence=[RelationshipEvidence(at="2026-10-03", session_id="s2", summary="来源原话", turn_indices=[5])])]
    original = profile.model_dump()
    stripped, removed = strip_a0_preference_sources(profile, session_id="s2", turn_index=5)
    assert profile.model_dump() == original
    assert [row.id for row in stripped.episodic_memory] == ["other"]
    assert [row.id for row in stripped.person_hypotheses] == ["other"]
    assert removed["facts"] == ["pref"] and removed["hypotheses"] == ["mixed"]
    assert removed["relationships"] == ["relation-pref"]
    assert stripped.relationship_context == []


def test_three_sessions_actual_product_path_forks_and_no_future_answers(tmp_path):
    runner, records = setup_runner(tmp_path)
    case = runner.suite.items[0]
    report = runner.run_case(case.id)
    assert report["evidence_kind"] == "software_fixture"
    assert report["effects"] == "not_evaluated" and report["scored_reply_count"] == 18
    assert len(records) == 7 and len(report["branches"]) == 6
    shared_app, clients = records[0]
    assert len(clients.online.calls) == 24
    assert len(clients.memory.calls) == len(clients.reflection.calls) == 2
    assert len(report["shared"]["profile"]["episodic_memory"]) == 2
    assert report["shared"]["profile"]["person_model_state"]["sessions_completed"] == 2
    # Author annotations and future dialogue do not enter online or offline clients.
    disclosed = [text for session in (1, 2) for text, _ in case.session(session)]
    for index, messages in enumerate(clients.online.calls):
        wire = json.dumps(messages, ensure_ascii=False)
        assert case.expected_preference.do not in wire and case.category not in wire
        assert all(future not in wire for future in disclosed[index + 1:])
        assert case.s3.trap[0] not in wire
    for client in (clients.memory, clients.reflection):
        assert all(case.s3.trap[0] not in str(call) for call in client.calls)
        assert all("simulated_e1b" not in str(call) for call in client.calls)
    run_ids = {shared_app._run["run_id"]}
    for app, branch_clients in records[1:]:
        run_ids.add(app._run["run_id"])
        assert len(branch_clients.online.calls) == 10
        assert not branch_clients.memory.calls and not branch_clients.reflection.calls
        assert app._profile().cognitive_health_consent.text_monitoring_enabled is False
        assert all("软件替身" in row["response"] for row in app._run["turns"])
    assert len(run_ids) == 7
    branch_emotions = [[row["external_emotion"] for row in branch["state"]["turns"]]
                       for branch in report["branches"]]
    assert all(value == branch_emotions[0] for value in branch_emotions)
    assert not list(runner.root.rglob("model_calls.sqlite3"))
    plan = json.loads((runner.root / "emotion-plan.json").read_text(encoding="utf-8"))
    assert len(plan["readings"]) == 612 and plan["synthetic"] is True
    assert all(branch["state"]["evidence_kind"] == "software_fixture" for branch in report["branches"])


def test_prescreen_does_not_generate_history_or_run_memory(tmp_path):
    runner, records = setup_runner(tmp_path)
    report = runner.run_case(runner.suite.items[0].id, phase="prescreen")
    assert report["shared"] is None and report["scored_reply_count"] == 9
    assert {branch["arm"] for branch in report["branches"]} == {"A0"}
    assert sum(len(clients.online.calls) for _, clients in records) == 30
    assert all(not clients.memory.calls and not clients.reflection.calls for _, clients in records)


class ProposingClient(E1bLocalFixtureClient):
    def complete(self, messages):
        result = super().complete(messages)
        if self.role != "memory":
            return result
        rows = re.findall(r"\[turn (\d+)\] 用户:([^\n]+)", messages[-1]["content"])
        # This test fixture derives a proposal from the already shown utterance,
        # not from the YAML's desired answer. The product validates/activates it.
        for index, text in rows:
            if "先问问我愿不愿意" in text:
                payload = json.loads(result)
                payload["stated_preferences"] = [{
                    "do": "先问问我愿不愿意", "quotes": [{"turn_index": int(index), "text": text}]}]
                return json.dumps(payload, ensure_ascii=False)
        return result


def test_actual_r1_confirmation_in_a1_and_absent_in_a0(tmp_path):
    def factory(token, records):
        def create(path):
            clients = MinimalModelClients(*(ProposingClient(role) for role in ("online", "memory", "reflection")),
                                          model_calls_enabled=True)
            app = E1bExperimentApp(path, experiment=token, clients=clients)
            records.append((app, clients))
            return app
        return create
    runner, _ = setup_runner(tmp_path, factory=factory)
    report = runner.run_case(runner.suite.items[0].id)
    assert len(report["shared"]["confirmation_ids"]) == 1
    preference = report["shared"]["profile"]["stated_preferences"][0]
    assert preference["status"] == "confirmed"
    for branch in report["branches"]:
        selected = branch["state"]["turns"][0]["input_evidence"]["stated_preferences"]
        assert bool(selected) is (branch["arm"] == "A1")


@pytest.mark.parametrize("failure", ["model", "memory", "save", "journal", "budget"])
def test_failure_unknown_or_save_stops_without_fallback_replay(tmp_path, monkeypatch, failure):
    runner, records = setup_runner(tmp_path)
    original_factory = runner.factory
    def factory(path):
        app = original_factory(path)
        if failure in {"model", "memory"}:
            client = getattr(app.clients, "online" if failure == "model" else "memory")
            def fail(messages):
                client.calls.append(deepcopy(messages))
                raise TimeoutError("injected unknown result")
            client.complete = fail
        if failure == "save":
            app.store.save = lambda *_: (_ for _ in ()).throw(OSError("injected write failure"))
        return app
    runner.factory = factory
    if failure == "budget":
        runner.gate = lambda *_: (_ for _ in ()).throw(RuntimeError("reservation denied"))
    if failure == "journal":
        runner._journal = lambda *_: (_ for _ in ()).throw(OSError("injected journal failure"))
    with pytest.raises((E1bRunStopped, RuntimeError)):
        runner.run_case(runner.suite.items[0].id)
    assert runner.stopped
    assert not (runner.root / "main" / runner.suite.items[0].id / "report.json").exists()
    counts = [len(clients.online.calls) for _, clients in records]
    if failure != "memory":
        assert sum(counts) <= 1
    with pytest.raises(ValueError, match="read-only"):
        E1bRunner(runner.suite, root=runner.root, app_factory=original_factory, seed=20261003)
    assert counts == [len(clients.online.calls) for _, clients in records]


def test_real_mode_needs_explicit_separate_authorized_caller(tmp_path):
    with pytest.raises(ValueError, match="separately authorized"):
        E1bRunner(load_e1b_cases(), root=tmp_path, app_factory=lambda _: None,
                  seed=0, evidence_kind="real_model")


def memory_response_factory(response):
    def factory(token, records):
        base = local_fixture_factory(token, recordings=records)
        def make(path):
            app = base(path)
            memory = app.clients.memory
            def complete(messages):
                memory.calls.append(deepcopy(messages))
                return response
            memory.complete = complete
            return app
        return make
    return factory


@pytest.mark.parametrize("raw,expected", [
    ('{"episodes":[],"relationships":[]}', "missing_field"),
    ('{"episodes":[],"relationships":[],"stated_preferences":null}', "null_field"),
    ('{"episodes":[],"relationships":[],"stated_preferences":{}}', "not_array"),
    ('not JSON at all', "parse_error"),
    ('{"episodes":[],"relationships":[],"stated_preferences":[],"stated_preferences":[]}', "parse_error"),
    (None, "non_text_response"),
])
def test_first_missing_null_or_parse_error_is_saved_and_stops_before_reflection(tmp_path, raw, expected):
    runner, records = setup_runner(tmp_path, factory=memory_response_factory(raw))
    case_id = runner.suite.items[0].id
    with pytest.raises(E1bRunStopped):
        runner.run_case(case_id)
    folder = runner.root / "main" / case_id
    first = json.loads((folder / "memory-s1.first.json").read_text(encoding="utf-8"))
    assert first["parse_status"] == expected
    assert (first["field_present"] is False) if expected == "missing_field" else True
    assert first["field_present"] is True if expected == "null_field" else True
    capture = json.loads((folder / "raw-proposals.json").read_text(encoding="utf-8"))
    assert capture["sessions"][0]["complete"] is False
    assert sum(len(c.memory.calls) for _, c in records) == 1
    assert sum(len(c.reflection.calls) for _, c in records) == 0
    assert sum(len(c.online.calls) for _, c in records) == 12
    assert not (folder / "report.json").exists()


def test_all_raw_candidates_types_and_over_limit_preserved_but_hidden_fields_excluded(tmp_path):
    import hashlib
    candidates = [None, "bad row", 3, ["bad nested"], {"do": "先问问我", "analysis": "HIDDEN_NESTED",
                  "quotes": [{"turn_index": 0, "text": "not grounded", "thinking": "HIDDEN_QUOTE"}]}]
    candidates += [{"do": f"无依据候选 {i}", "quotes": []} for i in range(11)]
    raw = '<think>HIDDEN_PREFIX</think>' + json.dumps({"episodes": [], "relationships": [],
           "reasoning": "HIDDEN_TOP", "stated_preferences": candidates, "cognitive_privacy": []}, ensure_ascii=False)
    runner, records = setup_runner(tmp_path, factory=memory_response_factory(raw))
    report = runner.run_case(runner.suite.items[0].id)
    folder = runner.root / "main" / runner.suite.items[0].id
    capture = json.loads((folder / "raw-proposals.json").read_text(encoding="utf-8"))
    assert len(capture["sessions"]) == 2
    for row in capture["sessions"]:
        assert row["complete"] is True and row["field_present"] is True
        assert len(row["raw_proposals"]) == 16
        assert row["raw_proposals"][:4] == candidates[:4]
        assert row["public_projection_dropped_fields"] == 2
        assert row["visible_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        assert row["product_outcome"]["accepted_count"] == 0
        assert len(row["product_outcome"]["rejected_codes"]) == 12
    assert not report["shared"]["profile"]["stated_preferences"]
    for path in folder.rglob("*.json"):
        content = path.read_text(encoding="utf-8")
        assert all(secret not in content for secret in ("HIDDEN_NESTED", "HIDDEN_QUOTE", "HIDDEN_PREFIX", "HIDDEN_TOP"))
    assert sum(len(c.memory.calls) for _, c in records) == 2
    assert sum(len(c.reflection.calls) for _, c in records) == 2


def test_actual_product_acceptance_recorded_before_ui_confirmation(tmp_path):
    def factory(token, records):
        def make(path):
            clients = MinimalModelClients(E1bLocalFixtureClient("online"), ProposingClient("memory"),
                                          E1bLocalFixtureClient("reflection"), model_calls_enabled=True)
            app = E1bExperimentApp(path, experiment=token, clients=clients)
            records.append((app, clients))
            return app
        return make
    runner, _ = setup_runner(tmp_path, factory=factory)
    runner.run_case(runner.suite.items[0].id)
    path = runner.root / "main" / runner.suite.items[0].id / "raw-proposals.json"
    capture = json.loads(path.read_text(encoding="utf-8"))
    outcome = capture["sessions"][1]["product_outcome"]
    assert outcome["accepted_count"] == 1
    assert outcome["accepted_preferences"][0]["status"] == "proposed"
    assert outcome["rejected_codes"] == []


@pytest.mark.parametrize("stage", ["first", "publish", "outcome"])
def test_raw_original_write_failure_stops_further_models_and_preserves_partial_file(tmp_path, monkeypatch, stage):
    import aria.bench.e1b_runner as module
    runner, records = setup_runner(tmp_path)
    original = module._exclusive_json
    def failed(path, value):
        if (stage == "first" and str(path).endswith(".first.json")) or (stage == "outcome" and str(path).endswith(".product.json")):
            path.write_text("partial retained", encoding="utf-8")
            raise OSError("injected raw original save failure")
        return original(path, value)
    monkeypatch.setattr(module, "_exclusive_json", failed)
    if stage == "publish":
        original_publish = module._write_json
        def failed_publish(path, value):
            if path.name == "raw-proposals.json":
                raise OSError("injected aggregate save failure")
            return original_publish(path, value)
        monkeypatch.setattr(module, "_write_json", failed_publish)
    with pytest.raises(E1bRunStopped):
        runner.run_case(runner.suite.items[0].id)
    first = runner.root / "main" / runner.suite.items[0].id / "memory-s1.first.json"
    if stage == "first":
        assert first.read_text(encoding="utf-8") == "partial retained"
    else:
        assert json.loads(first.read_text(encoding="utf-8"))["parse_status"] == "present_array"
    assert sum(len(c.memory.calls) for _, c in records) == 1
    assert sum(len(c.reflection.calls) for _, c in records) == 0
    assert sum(len(c.online.calls) for _, c in records) == 12


def test_operation_ids_unique_across_shared_sessions_and_all_forks(tmp_path):
    runner, records = setup_runner(tmp_path)
    runner.run_case(runner.suite.items[0].id)
    turn_ids = [row["turn_id"] for app, _ in records for row in app._run["turns"]]
    review_ids = [row["request_id"] for app, _ in records for row in app._run["reviews"]]
    assert len(turn_ids) == len(set(turn_ids)) == 84
    assert len(review_ids) == len(set(review_ids)) == 2


@pytest.mark.parametrize("same_ledger", [False, True])
def test_real_apps_must_share_external_guard_instance_and_identity(tmp_path, same_ledger):
    from aria.engine.policy.reserved_calls import ReservedCallGuard
    from aria.runtime import create_minimal_clients
    models = ("qwen/qwen3.7-flash", "anthropic/claude-sonnet-5")
    policy = {"schema": "aria_reservation_v1", "framing_tokens": 8192, "models": {model: {
        "input_usd_per_million": "4", "output_usd_per_million": "10",
        "provider": "alibaba" if index == 0 else "anthropic", "response_models": [model],
        "response_providers": ["Alibaba" if index == 0 else "Anthropic"],
        "max_input_tokens": 100000, "max_output_tokens": 4000, "metadata_sha256": "a" * 64}
        for index, model in enumerate(models)}}
    guards = [ReservedCallGuard(tmp_path / f"guard-{0 if same_ledger else i}.sqlite3", max_requests=200,
              max_observed_cost_usd="1", reservation_policy=policy) for i in range(2)]
    token = SyntheticEmotionExperiment.create(tmp_path, "shared-real-guard-test")
    def factory(path):
        clients = create_minimal_clients(model_calls_enabled=True, api_key="not-a-real-key",
                                          base_url="https://openrouter.ai/api/v1")
        return E1bExperimentApp(path, experiment=token, clients=clients, model_call_guard=guards[0])
    runner = E1bRunner(load_e1b_cases(), root=token.root / "run", app_factory=factory, seed=0,
                      evidence_kind="real_model", allow_model_calls=True)
    apps = []
    try:
        for index in range(2):
            app = runner._new_app(token.root / f"app-{index}")
            apps.append(app)
            runner._operation(app, "local-no-call", lambda: {"ok": True})
        assert runner._shared_guard is guards[0]
        assert guards[0].snapshot()["calls_used"] == 0
        bad = E1bExperimentApp(token.root / "app-bad", experiment=token,
              clients=create_minimal_clients(model_calls_enabled=True, api_key="not-a-real-key",
                                             base_url="https://openrouter.ai/api/v1"), model_call_guard=guards[1])
        apps.append(bad)
        with pytest.raises(E1bRunStopped, match="one externally owned guard"):
            runner._bind_shared_guard(bad)
        assert runner.stopped
        assert all(guard.snapshot()["calls_used"] == 0 for guard in guards)
    finally:
        for app in apps:
            app.close()
        for guard in guards:
            guard.close()


def test_actual_capture_integrates_formation_without_claiming_semantic_score(tmp_path):
    from scripts.e1b_evaluation import aggregate, draft_protocol, export_packet, load_bundle
    runner, _ = setup_runner(tmp_path)
    runner.run_case(runner.suite.items[0].id)
    protocol = draft_protocol(runner.suite, phase="main", case_ids=[c.id for c in runner.suite.items[:12]])
    output = tmp_path / "blind-capture"
    export_packet(runner.root, output, protocol)
    _, key = load_bundle(output)
    formed = key["formation"]["cases"][0]
    assert formed["raw_capture"]["schema"] == "e1b_raw_proposals_v2"
    assert formed["status"] == "independent_formation_labels_missing"
    assert len(formed["raw_capture"]["sessions"]) == 2
    assert all(row["complete"] for row in formed["raw_capture"]["sessions"])
    result = aggregate(output)
    assert result["formation_gate"] == "not_evaluable" and result["formation"]["proposed"] is None
    assert result["formation"]["incomplete_raw_cases"] == []
    assert len(result["formation"]["missing_raw_cases"]) == 11
