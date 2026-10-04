"""Draft schema and synthetic inputs, independent of any paid transport."""
from collections import Counter
from copy import deepcopy
import json

import pytest
import yaml

from aria.bench.e1b_cases import (
    DEFAULT_CASES, EMOTIONS, load_e1b_cases, select_prescreen, synthetic_emotion_plan,
)


def test_complete_draft_has_fixed_denominators_and_safe_projection():
    suite = load_e1b_cases()
    assert len(suite.items) == 18
    assert sum(len(case.session(s)) for case in suite.items for s in (1, 2, 3)) == 612
    for case in suite.items:
        assert len(suite.rubric(case)) == 6
        assert case.current_input(3, 7) == case.s3.trap[0]
        payload = case.current_input(1, 0)
        assert case.expected_preference.do not in payload
        assert case.category not in payload
    with pytest.raises(ValueError):
        suite.items[0].current_input(1, -1)


@pytest.mark.parametrize("mutation", ["extra", "missing_followup", "duplicate", "bool_index", "unknown_emotion", "leaking_answer"])
def test_invalid_draft_rejected_without_changing_original(tmp_path, mutation):
    original = DEFAULT_CASES.read_bytes()
    raw = yaml.safe_load(original)
    first = raw["items"][0]
    if mutation == "extra":
        first["prewritten_assistant"] = "never use this"
    elif mutation == "missing_followup":
        first["s3"]["followups"].pop()
    elif mutation == "duplicate":
        raw["items"][1]["id"] = first["id"]
    elif mutation == "bool_index":
        first["pref_turn"] = True
    elif mutation == "unknown_emotion":
        first["s1"][0][1] = "unrecognized"
    else:
        first["s1"][0][0] += " A1 expected_preference rubric"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_e1b_cases(path)
    assert DEFAULT_CASES.read_bytes() == original


def test_emotions_repeat_exactly_and_do_not_expose_ground_truth_or_answers():
    suite = load_e1b_cases()
    plan = synthetic_emotion_plan(suite, 20261003)
    assert plan == synthetic_emotion_plan(suite, 20261003)
    assert plan != synthetic_emotion_plan(suite, 20261004)
    counts = Counter()
    for case in suite.items:
        for session in (1, 2, 3):
            for index, (_, truth_code) in enumerate(case.session(session)):
                payload = plan[(case.id, session, index)]
                assert set(payload) == {"label", "confidence", "source", "modality"}
                truth = EMOTIONS[truth_code]
                truth = "sad" if truth == "low" else truth
                if payload["label"] == "unknown":
                    counts["unknown"] += 1
                    assert payload["confidence"] == 0
                elif payload["label"] == truth:
                    counts["truth"] += 1
                    assert .6 <= payload["confidence"] <= .9
                else:
                    counts["wrong"] += 1
                    assert .4 <= payload["confidence"] <= .7
                assert payload["source"].startswith("simulated_e1b:")
                assert "expected" not in json.dumps(payload)
    assert counts == {"truth": 428, "wrong": 92, "unknown": 92}


def test_prescreen_missing_scores_not_zero_and_shortfall_not_pass():
    suite = load_e1b_cases()
    labels = {case.id: (False, False, False) for case in suite.items}
    labels[suite.items[0].id] = (False, True, False)
    result = select_prescreen(suite, labels)
    assert result["case_ids"] == [suite.items[0].id]
    assert result["planned_denominator"] == 12 and result["actual_denominator"] == 1
    assert result["main_gate_registered"] is False
    assert result["status"] == "denominator_rule_required"
    for invalid in ({}, {**labels, suite.items[0].id: (None, False, False)},
                    {**labels, suite.items[0].id: (0, False, False)}):
        with pytest.raises(ValueError):
            select_prescreen(suite, invalid)
    with pytest.raises(ValueError):
        select_prescreen(suite, labels, target=10)
    all_fail = {case.id: (True, False, False) for case in suite.items}
    assert select_prescreen(suite, all_fail)["case_ids"] == [case.id for case in suite.items[:12]]
