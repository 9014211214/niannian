"""C-test software fixtures; no real model or clinical effect conclusions."""
from datetime import datetime, timezone
import json
import urllib.request

import pytest

from aria.bench.ctest_cases import DAY_OFFSETS, ctest_manifest, load_ctest_cases
from aria.bench.ctest_runner import CTestLocalMemoryFixture, CTestRunner, CTestRunStopped, _RecordedMemoryCall


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("C-test unit tests cannot use network")))


def test_script_layout_fixed_clock_and_assistant_provenance():
    cases = load_ctest_cases()
    manifest = ctest_manifest(cases)
    assert len(cases) == 12 and manifest["groups"] == {"change": 6, "control": 6}
    assert sum(len(case.sessions) for case in cases) == 96
    assert sum(len(visit.utterances) for case in cases for visit in case.sessions) == 960
    assert manifest["synthetic_assistant_context"] and manifest["online_model_reply_count"] == 0
    assert manifest["source_sha256"] == ctest_manifest(load_ctest_cases())["source_sha256"]
    assert DAY_OFFSETS == (0, 2, 4, 7, 10, 12, 18, 24)
    for case in cases:
        for visit in case.sessions:
            assert 10 <= len(visit.utterances) <= 12
            assert len(visit.utterances) == len(visit.synthetic_assistant)
            assert visit.virtual_at.year == 2000
            assert all("预期" not in line and "对照组" not in line for line in visit.utterances)


def test_all_96_actual_extractor_and_rules_with_synthetic_clock(tmp_path):
    before = datetime.now(timezone.utc)
    runner = CTestRunner(tmp_path, experiment_id="all12-local")
    summary = runner.run()
    assert len(runner.client.calls) == 96
    assert summary["case_count"] == 12 and summary["session_count"] == 96
    assert summary["change_triggered"] == 6 and summary["control_triggered"] == 0
    assert summary["software_rule_check"] == "pass"
    assert summary["real_model_gate"] == "not_executed"
    assert summary["reflection_calls"] == summary["online_calls"] == summary["real_online_reply_guard_checks"] == 0
    assert summary["scripted_reply_guard_checks"] == 960
    for case in summary["cases"]:
        assert all(row["status"] == "calibrating" for row in case["sessions"][:6])
        assert case["sessions"][6]["status"] == "watch"
    first_prompt = json.dumps(runner.client.calls[0], ensure_ascii=False)
    assert "我又忘了交电费，账单过期了" not in first_prompt
    for messages in runner.client.calls:
        serialized = json.dumps(messages, ensure_ascii=False)
        assert "control_denominator" not in serialized and "change_min_triggered" not in serialized
        assert '"group"' not in serialized and "virtual_day" not in serialized
    records = [json.loads(line) for line in runner.journal.read_text(encoding="utf-8").splitlines()]
    consent = [row for row in records if row["event"] == "synthetic_owner_enabled"]
    assert len(consent) == 12 and all(row["simulated_owner_action"] for row in consent)
    assert all(datetime.fromisoformat(row["observed_at"]) >= before for row in consent)
    profile = runner.store.load("fictional-c01")
    assert profile.profile_origin == "synthetic_cognitive_experiment:all12-local"
    assert profile.cognitive_health_consent.has_text_monitoring_consent()
    assert profile.cognitive_observations and not profile.person_hypotheses
    assert not profile.notification_receipts
    assert not list(runner.root.rglob("run.json"))  # no fabricated production Web logs
    calls = len(runner.client.calls)
    with pytest.raises(CTestRunStopped, match="read-only"):
        runner.run()
    assert len(runner.client.calls) == calls
    with pytest.raises(FileExistsError):
        CTestRunner(tmp_path, experiment_id="all12-local")


@pytest.mark.parametrize("failure", ["unknown", "invalid_json", "save_after_call", "journal"])
def test_failure_preserves_attempt_and_never_retries(tmp_path, monkeypatch, failure):
    client = CTestLocalMemoryFixture()
    runner = CTestRunner(tmp_path, experiment_id="failure", memory_client=client)
    original = client.complete
    if failure in {"unknown", "invalid_json"}:
        def complete(messages):
            original(messages)
            if failure == "unknown":
                raise TimeoutError("fixture result unknown")
            return "invalid fixture response"
        client.complete = complete
    elif failure == "save_after_call":
        original_save = runner.store.save
        def save(profile):
            if client.calls:
                raise OSError("injected store error")
            return original_save(profile)
        runner.store.save = save
    else:
        runner._record = lambda *_: (_ for _ in ()).throw(OSError("injected journal error"))
    with pytest.raises(CTestRunStopped):
        runner.run()
    assert runner.stopped and len(client.calls) <= 1
    assert not (runner.root / "summary.json").exists()
    with pytest.raises(CTestRunStopped):
        runner.run()
    assert len(client.calls) <= 1


def test_non_fixture_cannot_invoke_without_explicit_real_guard(tmp_path):
    class Client:
        def complete(self, _):
            raise AssertionError("must never be called")
    with pytest.raises(ValueError, match="separately authorized"):
        CTestRunner(tmp_path, experiment_id="not-authorized", memory_client=Client())
    with pytest.raises(ValueError, match="cannot be labeled"):
        CTestRunner(tmp_path, experiment_id="bad-label", allow_model_calls=True)


@pytest.mark.parametrize("observations", [None, "omitted", "wrong-type", ["not-an-object"]])
def test_missing_cognitive_protocol_is_not_scored_as_zero(tmp_path, observations):
    client = CTestLocalMemoryFixture()
    calls = []
    def incomplete(messages):
        calls.append(messages)
        body = {"episodes": [], "relationships": [], "stated_preferences": []}
        if observations != "omitted":
            body["cognitive_observations"] = observations
        return json.dumps(body)
    client.complete = incomplete
    runner = CTestRunner(tmp_path, experiment_id="missing-cognitive-field", memory_client=client)
    with pytest.raises(CTestRunStopped):
        runner.run()
    assert len(calls) == 1
    assert not (runner.root / "summary.json").exists()
    recorded = json.loads((runner.root / "c01-s1-response.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "cognitive_protocol_missing_or_invalid"


def test_fixed_denominator_or_script_cannot_change_after_creation(tmp_path):
    runner = CTestRunner(tmp_path, experiment_id="changed-plan")
    runner.cases = runner.cases[:-1]
    with pytest.raises(CTestRunStopped):
        runner.run()
    assert runner.client.calls == [] and not (runner.root / "summary.json").exists()
    with pytest.raises(ValueError, match="twelve unique"):
        ctest_manifest(load_ctest_cases()[:-1])


def test_visible_json_record_omits_unrequested_analysis_and_preserves_source_hash(tmp_path):
    import hashlib
    raw = 'unrequested prose before JSON ' + json.dumps({
        "analysis": "unrequested top level", "episodes": [], "relationships": [],
        "cognitive_observations": [{"domain": "memory", "evidence_quote": "逐字原话", "turn_index": 0,
                                     "reasoning_details": [{"text": "unrequested nested"}]}]}, ensure_ascii=False)
    class Client:
        def complete(self, _):
            return raw
    path = tmp_path / "response.json"
    assert _RecordedMemoryCall(Client(), path).complete([]) == raw
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["visible_content_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "unrequested" not in json.dumps(record)
    assert record["memory_json"]["cognitive_observations"][0]["evidence_quote"] == "逐字原话"


def test_registration_sha_must_be_a_separate_valid_digest(tmp_path):
    with pytest.raises(ValueError, match="registration SHA"):
        CTestRunner(tmp_path, experiment_id="bad-registration", registration_sha256="not-a-sha")
