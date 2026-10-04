"""Only authored dummy data and clients; no downloaded source or real models."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from aria.bench.prefeval_cases import CHECKS, call_inventory, distraction_plan, load_suite, qualification_audit, sha
from aria.bench.prefeval_judge import aggregate, export_materials, validate_labels
from aria.bench.prefeval_runner import (
    GENERAL_SYSTEM, PrefEvalClients, PrefEvalFixtureClient, PrefEvalRunner, PrefEvalStopped, arm_messages,
)
import aria.bench.prefeval_runner as runner_module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "dummy-upstream"
    root.mkdir()
    files = []

    def save(name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode()
        path.write_bytes(raw)
        files.append({"path": name, "sha256": sha(raw)})

    save("LICENSE", "Dummy license header: Attribution-NonCommercial 4.0 International")
    for index in range(20):
        topic = f"dummy_topic_{index:02d}"
        save(f"benchmark_dataset/explicit_preference/{topic}.json", json.dumps([
            {"preference": f"I prefer fictional option {index}-{i}.",
             "question": f"What example fits topic {index} item {i}?",
             "explanation": "Dummy expected context, never pass this to a reply generator."}
            for i in range(50)]))
    for check in CHECKS:
        save(f"error_type/check_{check}.txt", f"Dummy {check} criteria for software checks only.")
    manifest = {"repository": "https://github.com/amazon-science/PrefEval", "revision": "0" * 40,
                "use": "local_noncommercial_evaluation_only", "redistribute_source": False, "files": files}
    (root / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def fixture_clients():
    return PrefEvalClients(PrefEvalFixtureClient("online"), PrefEvalFixtureClient("memory"))


def small_suite(source):
    suite = load_suite(source)
    return replace(suite, cases=(suite.cases[0], suite.cases[-1]))


def test_fixed_stratified_sample_has_full_denominator_and_no_data_filtering(source):
    suite = load_suite(source)
    assert len(suite.pool) == 1000 and len(suite.cases) == 48
    counts = Counter(c.topic for c in suite.cases)
    assert len(counts) == 20 and Counter(counts.values()) == {2: 12, 3: 8}
    assert suite.frozen_selection() == load_suite(source).frozen_selection()
    assert suite.frozen_selection() != load_suite(source, seed=123).frozen_selection()
    audit = qualification_audit(suite)
    assert audit["structurally_excluded"] == 48
    assert audit["real_model_effect"] == "not_run"
    assert audit["status"] == "needs_protocol_review"
    assert "score" not in audit


def test_source_changes_fail_before_sampling(source):
    changed = source / "benchmark_dataset/explicit_preference/dummy_topic_00.json"
    changed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        load_suite(source)


def test_distractions_exclude_own_topic_and_selected_rows_and_count_all_calls(source):
    suite = load_suite(source)
    plan = distraction_plan(suite)
    selected = {case.id for case in suite.cases}
    assert plan == distraction_plan(suite)
    assert all(len(rows) == 10 for rows in plan.values())
    assert all(row["source_topic"] != topic and row["source_case_id"] not in selected
               for topic, rows in plan.items() for row in rows)
    inventory = call_inventory(suite)
    assert inventory["generated_distraction_online"] == 200
    assert inventory["calls_before_judging"] == 584
    assert inventory["claude_calls_if_four_official_checks_separate"] == 160
    assert inventory["budget_reestimated"] is False


def test_four_arms_share_generic_prompt_and_p2_cannot_receive_gold_preference(source):
    case = load_suite(source).cases[0]
    first = [{"user": text, "assistant": "Fixture answer"} for text in (case.preference, "one", "two")]
    noise = [{"user": f"Other question {i}", "assistant": "Other response"} for i in range(10)]
    arms = {arm: arm_messages(case, arm, first_session=first, distractions=noise, confirmed=[])
            for arm in ("P0", "P1", "P2", "P3")}
    assert all(rows[0]["content"] == GENERAL_SYSTEM for rows in arms.values())
    assert len(arms["P0"]) == 2
    assert len(arms["P1"]) == 28
    assert len(arms["P2"]) == 10
    assert case.preference in str(arms["P1"]) and case.preference in str(arms["P3"])
    assert case.preference not in str(arms["P0"]) and case.preference not in str(arms["P2"])
    assert all(case.explanation not in str(rows) for rows in arms.values())


def test_actual_extractor_and_confirmation_pipeline_with_no_invented_preferences(source, tmp_path):
    suite, clients = small_suite(source), fixture_clients()
    result = PrefEvalRunner(suite, root=tmp_path / "run", clients=clients).run()
    assert len(clients.memory.calls) == 2
    assert "stated_preferences" in clients.memory.calls[0][0]["content"]
    assert "_expected" not in str(clients.memory.calls)
    assert all(item["preferences"] == [] for item in result["formations"])
    assert len(result["outputs"]) == 8
    assert result["calls_before_judging"] == 36
    assert result["metrics"] == "not_evaluated"
    assert (tmp_path / "run/generated-distractions.frozen.json").exists()
    with pytest.raises(FileExistsError):
        PrefEvalRunner(suite, root=tmp_path / "run", clients=fixture_clients())


def test_real_run_refused_before_clients_or_credentials_when_contract_cannot_apply(source, tmp_path):
    clients = fixture_clients()
    with pytest.raises(ValueError, match="applicability_review_required"):
        PrefEvalRunner(load_suite(source), root=tmp_path / "should-not-exist", clients=clients,
                       evidence_kind="real_model")
    assert not (tmp_path / "should-not-exist").exists()
    assert not clients.online.calls and not clients.memory.calls


def test_model_failure_and_saved_result_failure_are_not_retried(source, tmp_path, monkeypatch):
    clients = fixture_clients()
    runner = PrefEvalRunner(small_suite(source), root=tmp_path / "failed", clients=clients)
    calls = []

    def fail(messages):
        calls.append(messages)
        raise RuntimeError("local-fixture-failure")

    monkeypatch.setattr(clients.online, "complete", fail)
    with pytest.raises(PrefEvalStopped, match="no_automatic_retry"):
        runner.run()
    with pytest.raises(PrefEvalStopped, match="fresh_run_only"):
        runner.run()
    assert len(calls) == 1
    assert "stopped" in runner.journal.read_text(encoding="utf-8")


def test_save_failure_stops_after_generated_noise_without_starting_memory(source, tmp_path, monkeypatch):
    clients = fixture_clients()
    runner = PrefEvalRunner(small_suite(source), root=tmp_path / "save-failed", clients=clients)

    def fail_save(*args):
        raise OSError("local-save-failure")

    monkeypatch.setattr(runner_module, "_write", fail_save)
    with pytest.raises(PrefEvalStopped, match="result_save_failed"):
        runner.run()
    assert runner.stopped and not clients.memory.calls
    assert len(clients.online.calls) == 20


def test_product_confirmation_is_used_for_a_valid_authored_communication_fixture(source, tmp_path, monkeypatch):
    suite = small_suite(source)
    quote = "请你先问问我再做安排"
    case = replace(suite.cases[0], preference=quote)
    suite = replace(suite, cases=(case,))
    clients = fixture_clients()
    captured = []

    def memory(messages):
        captured.append(messages)
        return json.dumps({"episodes": [], "relationships": [], "stated_preferences": [{
            "do": quote, "quotes": [{"turn_index": 0, "text": quote}], "confirmation": None}]}, ensure_ascii=False)

    monkeypatch.setattr(clients.memory, "complete", memory)
    generated = PrefEvalRunner(suite, root=tmp_path / "valid-authored-fixture", clients=clients).run()
    preferences = generated["formations"][0]["preferences"]
    assert len(preferences) == 1 and preferences[0]["do"] == quote
    assert preferences[0]["confirmation"]["source"] == "ui"
    assert case.question not in str(captured) and case.explanation not in str(captured)
    assert generated["evidence_kind"] == "software_fixture"


def valid_labels(materials):
    return [{"position_id": row["position_id"], "response_sha256": row["response_sha256"],
             "violation": "No", "hallucination": "No", "acknowledge": "No", "helpful": "Yes", "restatement": ""}
            for row in materials["packet"]["cards"]]


def test_judge_materials_are_blind_and_blank_not_fake_scores(source, tmp_path):
    suite = small_suite(source)
    result = PrefEvalRunner(suite, root=tmp_path / "judge", clients=fixture_clients()).run()
    materials = export_materials(suite, result)
    assert len(materials["packet"]["cards"]) == 8
    assert all("arm" not in card and "case_id" not in card for card in materials["packet"]["cards"])
    assert materials["blank_labels"][0]["violation"] is None
    with pytest.raises(ValueError, match="unscored"):
        validate_labels(materials["packet"], materials["blank_labels"])
    labels = valid_labels(materials)
    assert len(validate_labels(materials["packet"], labels)) == 8
    with pytest.raises(ValueError, match="missing_or_extra"):
        validate_labels(materials["packet"], labels[:-1])
    wrong = deepcopy(labels)
    wrong[0]["response_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stale"):
        validate_labels(materials["packet"], wrong)
    wrong = deepcopy(labels)
    wrong[0].update(acknowledge="Yes", restatement="invented quote")
    with pytest.raises(ValueError, match="not_verbatim"):
        validate_labels(materials["packet"], wrong)
    summary = aggregate(suite, result, materials, labels,
                        [{"case_id": case.id, "formed_correctly": False} for case in suite.cases])
    assert summary["gate"] == "not_eligible_for_effect_conclusion"
    assert summary["input_token_saving_P2_vs_P1"] is None
