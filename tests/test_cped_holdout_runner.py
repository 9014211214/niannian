"""Original synthetic fixtures and in-memory fakes; no CPED or model access."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import urllib.request

import pytest

from aria.bench.cped_holdout import prepare_cped_holdout
from aria.bench.cped_holdout_runner import (
    CONDITIONS, LETTERS, HoldoutExecutionConfig, HoldoutStageConfig,
    _bootstrap, _empty_measure, _exact_mcnemar, _fingerprints, _formation_metrics,
    _preflight, _report, _unknown_persona,
    assert_aggregate_only_execution, behavior_request, condition_order,
    formation_request, parse_formation, run_cped_holdout, wrong_persona_assignment,
)
from aria.bench.evaluation_journal import EvaluationJournal, JournalError
from aria.bench.cped_source import CPEDRow, CPED_TRAITS


def _source():
    # Authored here; the artificial two-character suffix ensures unique turns.
    activities = ("公园散步", "书房阅读", "院子种花", "厨房做菜", "广场听歌", "河边钓鱼", "窗前画画", "客厅下棋")
    feelings = ("让我安心", "带来乐趣", "缓解烦闷", "使我舒坦", "令人放松", "唤起回忆", "增添兴致", "带动心情")
    rows = []
    for i in range(64):
        for dialogue in range(4):
            for turn in range(6):
                suffix = chr(0x4E00 + i) + chr(0x5200 + dialogue * 4 + max(turn - 2, 0))
                target = turn >= 2
                rows.append(CPEDRow(
                    tv_id=f"synthetic-show-{i}", dialogue_id=str(dialogue), utterance_id=f"utt_{turn}",
                    speaker=f"synthetic-name-{i}" if target else f"synthetic-partner-{i}",
                    age="elderly" if i < 13 else "middle-aged",
                    traits={trait: "high" for trait in CPED_TRAITS}, sentiment="neutral", emotion="neutral",
                    dialogue_act="statement-opinion", split="authored",
                    utterance=f"今天{activities[i % 8]}{feelings[i // 8]}{suffix}。" if target else
                    f"愿听您的另一段经历{chr(0x6500 + i)}{chr(0x6800 + dialogue * 2 + turn)}。",
                ))
    return rows


@pytest.fixture(scope="module")
def prepared():
    result = prepare_cped_holdout(_source(), terms_review_acknowledged=True, aggregate_only_accepted=True)
    assert result.report["status"] == "prepared_pending_identity_review"
    return result


def _config(prepared):
    return HoldoutExecutionConfig(
        formation=HoldoutStageConfig("synthetic/former", "synthetic/former-v1", 800, .005),
        behavior=HoldoutStageConfig("synthetic/predictor", "synthetic/predictor-v1", 80, .001),
        identity_reviewed_fingerprint=prepared.private_cohort_fingerprint,
        data_file_sha256=(hashlib.sha256(b"original synthetic corpus, no CPED").hexdigest(),),
        bootstrap_samples=200,
    )


def _fake(payload):
    body = json.loads(payload["messages"][1]["content"])
    formation = "dimensions" in body
    return {"model": "synthetic/former-v1" if formation else "synthetic/predictor-v1",
            "usage": {"prompt_tokens": 110, "completion_tokens": 70 if formation else 12, "cost": .00001},
            "output": {trait: {"label": "high", "confidence": .8, "evidence": [body["history"][0]],
                               "counterevidence": []} for trait in CPED_TRAITS}
            if formation else {"ranking": list(LETTERS)}}


@pytest.fixture(scope="module")
def completed(prepared, tmp_path_factory):
    destination = tmp_path_factory.mktemp("holdout-complete") / "private-ledger"
    payloads = []
    def fake(payload):
        payloads.append(payload)
        return _fake(payload)
    report = run_cped_holdout(prepared, config=_config(prepared), fake_model=fake, ledger_path=destination)
    return report, payloads, destination


def test_full_288_call_rehearsal_is_complete_software_not_effect_evidence(completed):
    report, payloads, directory = completed
    assert len(payloads) == report["attempted_calls"] == report["expected_calls"] == 288
    assert report["not_attempted_calls"] == report["retry_calls"] == 0
    assert report["status"] == "inconclusive"
    assert report["official_experiment_complete"] is False
    assert report["cloud_submission_allowed"] is False
    assert report["stop_reason"] is None
    assert report["formation"]["valid"] == 48
    assert report["formation_metrics"]["evidence_grounding_rate"] == 1
    assert report["single_actual_model_per_stage_observed"] is True
    assert report["all_conditions_valid_at_least_90_percent"] is True
    assert len(list(directory.glob("*.result.json"))) == 288
    assert report["elderly_slice"]["role_count"] < 24
    assert set(report["elderly_slice"]) == {"role_count", "metrics_allowed"}
    assert_aggregate_only_execution(report)


def test_blinding_same_candidates_and_context_and_no_source_ids(completed, prepared):
    report, payloads, directory = completed
    for i, case in enumerate(prepared.cases):
        formed = json.loads(payloads[i]["messages"][1]["content"])
        assert set(formed) == {"dimensions", "history"}
        assert not (set(formed["history"]) & set(case.candidates))
        group = payloads[48 + 5 * i:48 + 5 * i + 5]
        contents = [json.loads(p["messages"][1]["content"]) for p in group]
        assert all(c["context"] == contents[0]["context"] and c["candidates"] == contents[0]["candidates"] for c in contents)
        for payload in [payloads[i], *group]:
            encoded = json.dumps(payload, ensure_ascii=False)
            assert all(condition not in encoded for condition in CONDITIONS)
            assert all(part not in encoded for part in case.role_key)
            assert "correct_position" not in encoded and "gold_traits" not in encoded
        for c, condition in zip(contents, condition_order(i)):
            if condition in {"wrong_persona", "inferred_persona"}:
                assert all(set(v) == {"label", "confidence"} for v in c["person"].values())
    public = json.dumps(report, ensure_ascii=False)
    stored = "".join(p.read_text(encoding="utf-8") for p in directory.glob("*.json"))
    for case in prepared.cases:
        assert all(text not in public and text not in stored for text in case.formation_texts)
        assert case.role_key[0] not in public and case.role_key[0] not in stored
    assert '"ranking"' not in stored and '"evidence"' not in stored


def test_fixed_denominators_rotation_and_paired_character_statistics(completed):
    report = completed[0]
    for condition in CONDITIONS:
        row = report["conditions"][condition]
        assert row["valid"] == row["expected"] == 48
        assert row["accuracy"] == .25
        assert set(row["planned_order_position_counts"]) == {9, 10}
        assert sum(row["planned_order_position_counts"]) == 48
        assert row["planned_order_position_counts"] == row["actual_order_position_counts"]
        assert len(row["accuracy_ci95"]) == len(row["mrr_ci95"]) == 2
    assert len(report["comparisons"]) == 5
    primary = report["comparisons"]["inferred_persona_vs_wrong_persona"]
    assert primary["primary"] is True and primary["role_denominator"] == 48
    assert primary["exact_mcnemar_p"] == 1 and primary["top1_difference"] == 0
    assert report["task_sensitivity_review_required"] is True


def test_repeat_run_rejected_before_callback(completed, prepared):
    before = {p.name: p.read_bytes() for p in completed[2].glob("*.json")}
    with pytest.raises(JournalError, match="already_exists"):
        run_cped_holdout(prepared, config=_config(prepared), fake_model=lambda _: pytest.fail("repeated"),
                         ledger_path=completed[2])
    assert before == {p.name: p.read_bytes() for p in completed[2].glob("*.json")}


@pytest.mark.parametrize("reason", ["transport_failed", "provider_cost_unknown", "provider_tokens_unknown",
                                   "actual_model_missing_or_changed", "maximum_request_cost_exceeded"])
def test_failure_unknown_cost_and_model_drift_stop_without_retry(reason, prepared, tmp_path, monkeypatch):
    stub_model_http(monkeypatch, lambda *_a, **_k: pytest.fail("network access"))
    calls = []
    def fake(payload):
        calls.append(payload)
        if reason == "transport_failed":
            raise TimeoutError("private provider text must not escape")
        envelope = _fake(payload)
        if reason == "provider_cost_unknown":
            del envelope["usage"]["cost"]
        elif reason == "provider_tokens_unknown":
            del envelope["usage"]["prompt_tokens"]
        elif reason == "actual_model_missing_or_changed":
            envelope["model"] = "untrusted/identity"
        else:
            envelope["usage"]["cost"] = .01
        return envelope
    destination = tmp_path / "ledger"
    report = run_cped_holdout(prepared, config=_config(prepared), fake_model=fake, ledger_path=destination)
    assert len(calls) == report["attempted_calls"] == 1
    assert report["stop_reason"] == reason
    assert report["not_attempted_calls"] == 287
    assert report["all_conditions_valid_at_least_90_percent"] is False
    assert all(row["accuracy"] is None and row["accuracy_ci95"] is None for row in report["conditions"].values())
    assert all(row["exact_mcnemar_p"] is None for row in report["comparisons"].values())
    assert report["formation"]["invalid"] == 1
    assert "private provider" not in json.dumps(report)
    assert "untrusted/identity" not in json.dumps(report)
    with pytest.raises(JournalError):
        run_cped_holdout(prepared, config=_config(prepared), fake_model=fake, ledger_path=destination)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["origin", "review", "cohort", "surface", "ceiling", "reserve"])
def test_prerequisites_reject_before_callback_and_ledger_creation(change, prepared, tmp_path):
    config = _config(prepared)
    source = prepared
    if change == "origin":
        config = replace(config, data_origin="real_cped")
        source = object()  # Not touched when actual data is prohibited.
    elif change == "review":
        config = replace(config, identity_reviewed_fingerprint="0" * 64)
    elif change == "cohort":
        source = replace(prepared, cases=prepared.cases[:-1])
    elif change == "surface":
        source = replace(prepared, report=deepcopy(prepared.report))
        source.report["surface_correct_counts"]["length"] = 48
    elif change == "ceiling":
        config = replace(config, cost_ceiling_usd=2.01)
    else:
        config = replace(config, formation=replace(config.formation, max_request_cost_usd=.05))
    with pytest.raises(ValueError):
        run_cped_holdout(source, config=config, fake_model=lambda _: pytest.fail("not admitted"),
                         ledger_path=tmp_path / "ledger")
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("failure_type", [JournalError, OSError])
def test_journal_write_failure_never_calls_model_and_result_failure_never_retries(prepared, tmp_path, monkeypatch, failure_type):
    def fail(*_a, **_k):
        raise failure_type("synthetic disk failure")
    calls = []
    with monkeypatch.context() as scoped:
        scoped.setattr(EvaluationJournal, "start", fail)
        report = run_cped_holdout(prepared, config=_config(prepared), fake_model=lambda p: calls.append(p),
                                 ledger_path=tmp_path / "before")
    assert calls == [] and report["stop_reason"] == "ledger_write_failed"
    def fake(payload):
        calls.append(payload)
        return _fake(payload)
    monkeypatch.setattr(EvaluationJournal, "finish", fail)
    report = run_cped_holdout(prepared, config=_config(prepared), fake_model=fake, ledger_path=tmp_path / "after")
    assert len(calls) == 1 and report["stop_reason"] == "ledger_result_save_failed"
    assert len(list((tmp_path / "after").glob("*.started.json"))) == 1
    assert len(list((tmp_path / "after").glob("*.result.json"))) == 0


def test_wrong_persona_permutation_is_cross_show_without_self_loops_and_deterministic(prepared):
    personas = [_unknown_persona() for _ in prepared.cases]
    a = wrong_persona_assignment(prepared.cases, personas, seed=19)
    assert a == wrong_persona_assignment(prepared.cases, personas, seed=19)
    assert set(a) == set(range(48))
    assert all(i != j and prepared.cases[i].role_key[0] != prepared.cases[j].role_key[0] for i, j in enumerate(a))
    same_show = [replace(c, role_key=("same-show", str(i))) for i, c in enumerate(prepared.cases)]
    with pytest.raises(ValueError, match="no_cross_show"):
        wrong_persona_assignment(same_show, personas, seed=19)


def test_requests_are_stable_and_oracle_is_only_deliberate_weak_gold_input(prepared):
    case = prepared.cases[0]
    config = _config(prepared)
    unknown = _unknown_persona()
    assert formation_request(case, config.formation) == formation_request(case, config.formation)
    for condition in CONDITIONS:
        a = behavior_request(case, condition, unknown, unknown, config.behavior)
        b = behavior_request(case, condition, unknown, unknown, config.behavior)
        assert a == b
        parsed = json.loads(a["messages"][1]["content"])
        expected = case.gold_traits if condition == "oracle_big_five" else ("unknown",) * 5
        assert tuple(parsed["person"][t]["label"] for t in CPED_TRAITS) == expected


def test_formation_unknown_grounding_and_invalid_denominator(prepared):
    personas = [_unknown_persona() for _ in prepared.cases]
    metric = _formation_metrics(prepared.cases, personas, [False] * 48)
    assert metric["valid_roles"] == 0 and metric["dimension_denominator"] == 240
    assert metric["unknown_coverage"] == 0 and metric["accuracy"] == 0
    assert metric["unavailable_dimensions"] == 240
    assert metric["evidence_grounding_rate"] is None
    personas[0] = {t: {"label": "high", "confidence": .9, "evidence": ["未经提供的原话"],
                      "counterevidence": []} for t in CPED_TRAITS}
    metric = _formation_metrics(prepared.cases, personas, [True] + [False] * 47)
    assert metric["evidence_quotes"] == 5 and metric["grounded_quotes"] == 0
    assert metric["high_confidence_ungrounded_rate"] == 5 / 240
    assert metric["accuracy"] == 5 / 240
    assert parse_formation(personas[0]) is not None
    malformed = deepcopy(personas[0])
    malformed[CPED_TRAITS[0]]["confidence"] = True
    assert parse_formation(malformed) is None
    assert parse_formation({}) is None


def test_invalid_schema_is_accounted_then_next_unknown_stops_without_repair(prepared, tmp_path):
    calls = []
    def fake(payload):
        calls.append(payload)
        result = _fake(payload)
        if len(calls) == 1:
            result["output"] = {"wrong_shape": True}
        else:
            del result["usage"]["cost"]
        return result
    report = run_cped_holdout(prepared, config=_config(prepared), fake_model=fake, ledger_path=tmp_path / "ledger")
    assert len(calls) == 2
    assert report["formation"]["failure_counts"] == {"invalid_model_output": 1, "provider_cost_unknown": 1}
    assert report["formation_metrics"]["unavailable_dimensions"] == 240
    assert report["formation_metrics"]["unknown_coverage"] == 0


@pytest.mark.parametrize("valid_count,allowed", [(43, False), (44, True)])
def test_ninety_percent_validity_uses_all_48_expected_roles(valid_count, allowed, prepared):
    config = _config(prepared)
    measures = {c: _empty_measure() for c in ("formation", *CONDITIONS)}
    for measure in measures.values():
        measure.update(attempted=48, not_attempted=0, valid=valid_count, invalid=48-valid_count)
    scores = {c: [1] * valid_count + [0] * (48-valid_count) for c in CONDITIONS}
    report = _report(prepared.cases, [_unknown_persona() for _ in prepared.cases], [True] * 48,
                     measures, scores, scores, config, _fingerprints(prepared, config),
                     _preflight(prepared, config), None, False, preparation_seed=prepared.report["seed"])
    assert report["all_conditions_valid_at_least_90_percent"] is allowed
    assert report["conditions"][CONDITIONS[0]]["accuracy"] == valid_count / 48
    assert_aggregate_only_execution(report)


def test_partial_conditions_never_create_order_artifact_paired_advantage(prepared):
    config = _config(prepared)
    measures = {c: _empty_measure() for c in ("formation", *CONDITIONS)}
    measures["inferred_persona"].update(attempted=1, not_attempted=47, valid=1)
    measures["wrong_persona"].update(attempted=1, not_attempted=47, invalid=1)
    scores = {c: [0] * 48 for c in CONDITIONS}
    scores["inferred_persona"][0] = 1
    report = _report(prepared.cases, [_unknown_persona() for _ in prepared.cases], [False] * 48,
                     measures, scores, scores, config, _fingerprints(prepared, config),
                     _preflight(prepared, config), "provider_cost_unknown", False,
                     preparation_seed=prepared.report["seed"])
    row = report["conditions"]["inferred_persona"]
    assert row["metric_scope"] == "incomplete_conservative_lower_bound"
    assert row["accuracy"] == 1 / 48 and row["accuracy_ci95"] is None
    comparison = report["comparisons"]["inferred_persona_vs_wrong_persona"]
    assert comparison["top1_difference"] is None and comparison["exact_mcnemar_p"] is None
    assert report["formation_metrics"]["accuracy"] is None
    assert report["formation_metrics"]["unknown_coverage"] is None
    assert report["preparation_seed"] == 20260909 and report["seed"] == 20260929
    assert_aggregate_only_execution(report)


def test_bootstrap_and_mcnemar_resample_independent_roles():
    assert _bootstrap([0] * 48, seed=1, repetitions=200) == [0, 0]
    assert _bootstrap([1] * 48, seed=1, repetitions=200) == [1, 1]
    assert _exact_mcnemar([1] * 6, [0] * 6) == (6, 0, .03125)


@pytest.mark.parametrize("location", ["root", "condition", "formation", "fingerprints", "elderly", "failure", "ci"])
def test_aggregate_export_rejects_private_derivatives_at_every_depth(location, completed):
    report = deepcopy(completed[0])
    if location == "root":
        report["cases"] = [{"quote": "private source"}]
    elif location == "condition":
        report["conditions"][CONDITIONS[0]]["role_ids"] = ["private-source-id"]
    elif location == "formation":
        report["formation_metrics"]["evidence"] = "private quote"
    elif location == "fingerprints":
        report["fingerprints"]["data_file_sha256"] = ["private path"]
    elif location == "elderly":
        report["elderly_slice"]["accuracy"] = .5
    elif location == "failure":
        report["formation"]["failure_counts"]["private error text"] = 1
    else:
        report["conditions"][CONDITIONS[0]]["accuracy_ci95"] = [0, "private text"]
    with pytest.raises(ValueError, match="aggregate_only"):
        assert_aggregate_only_execution(report)
