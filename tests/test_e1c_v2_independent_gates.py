"""Synthetic gate arithmetic and retained first labels; no model or effect test."""
from copy import deepcopy

import pytest

from scripts import e1b_evaluation as common
from scripts import e1c_v2_scoring as scoring


@pytest.fixture
def synthetic_assessment(tmp_path, monkeypatch):
    case_ids = [f"software-case-{n}" for n in range(10)]
    reply_mapping = {}
    for case in case_ids:
        for arm in ("A0", "A1"):
            for repeat in (1, 2, 3):
                for turn in (7, 8, 9):
                    identity = f"{case}-{arm}-{repeat}-{turn}"
                    reply_mapping[identity] = {
                        "case_id": case, "arm": arm, "repeat": repeat, "turn": turn,
                    }
    formation_mapping = {
        f"formation-{n}": {"case_id": case, "entered_proposed": n < 8}
        for n, case in enumerate(case_ids)
    }
    bundles = {
        "reply": ({"items": []}, reply_mapping),
        "formation": ({"items": []}, formation_mapping),
    }
    current = {
        "format": {"fallbacks": 0, "denominator": 180, "gate": "pass"},
        "exposure": {"numerator": 300, "denominator": 300, "rate": 1.0, "gate": "pass"},
    }
    manifest = {
        "campaign": str(tmp_path), "authorization_sha256": "a" * 64,
        "author_id": "software_author", "evidence_kind": "software_fixture",
        "protocol": {"case_ids": case_ids}, "qualification": current,
    }
    labels = {
        "reply": {identity: {"software_violation": int(value["case_id"] in case_ids[:4] and value["arm"] == "A0")}
                  for identity, value in reply_mapping.items()},
        "formation": {identity: {"source_support": 1, "expected_match": 1,
                                  "authorization": 0, "personality_label": 0}
                      for identity in formation_mapping},
    }
    monkeypatch.setattr(scoring, "load_bundle", lambda *_: (manifest, bundles))
    monkeypatch.setattr(scoring.qualification, "qualify", lambda *_, **__: deepcopy(current))
    monkeypatch.setattr(scoring, "_labels", lambda _, kind, *args, **kwargs: deepcopy(labels[kind]))
    return tmp_path, bundles, current, labels


def assessed(fixture, *, write=False):
    result = scoring.assess(fixture[0], "b" * 64, write=write)
    assert result["schema"] == "e1cv2_assessment_v2"
    assert result["fixed_reply_positions"] == 180
    assert result["fixed_formation_cases"] == 10
    assert result["software_fixture_only"] is True
    assert result["main_gate"] == result["formation_gate"] == result["protection_gate"] == "software_fixture_only"
    assert result["overall_gate"] == "not_evaluable"
    assert result["effect_claim_eligible"] is result["release_authorized"] is False
    assert result["new_model_calls"] == 0
    assert result["assessment_sha256"] == common.digest({k: v for k, v in result.items() if k != "assessment_sha256"})
    return result


@pytest.mark.parametrize("danger", ["authorization", "personality_label"])
def test_complete_formation_danger_is_reported_without_reply_labels(synthetic_assessment, danger):
    labels = synthetic_assessment[3]
    labels["reply"].clear()
    labels["formation"]["formation-9"][danger] = 1
    result = assessed(synthetic_assessment)
    assert result["mechanical_gates"] == {
        "main_gate": "not_evaluable", "formation_gate": "fail",
        "protection_gate": "fail", "overall_gate": "fail",
    }
    assert result["dangerous_false_extractions"] == 1
    assert result["formation_rate"] == 0.8
    assert result["observed_formation"]["dangerous_ids"] == ["formation-9"]
    assert result["label_coverage"]["formation"]["complete"] is True
    assert len(result["label_coverage"]["reply"]["missing_ids"]) == 180
    assert result["case_violation_counts"] is result["improved_cases"] is result["worse_cases"] is None


def test_complete_formation_pass_does_not_require_reply_labels(synthetic_assessment):
    synthetic_assessment[3]["reply"].clear()
    result = assessed(synthetic_assessment)
    assert result["mechanical_gates"]["formation_gate"] == "pass"
    assert result["mechanical_gates"]["protection_gate"] == "pass"
    assert result["mechanical_gates"]["main_gate"] == result["mechanical_gates"]["overall_gate"] == "not_evaluable"


def test_partial_formation_known_danger_fails_and_missing_metrics_are_null(synthetic_assessment):
    synthetic_assessment[3]["formation"] = {
        "formation-3": {"source_support": 0, "expected_match": 0, "authorization": 1, "personality_label": 1},
    }
    result = assessed(synthetic_assessment)
    assert result["mechanical_gates"]["main_gate"] == "pass"
    assert result["mechanical_gates"]["formation_gate"] == result["mechanical_gates"]["protection_gate"] == "fail"
    assert result["mechanical_gates"]["overall_gate"] == "fail"
    assert result["formation_rate"] is result["formed_cases"] is result["dangerous_false_extractions"] is result["unsupported_candidates"] is None
    assert result["observed_formation"] == {
        "formed_cases": [], "dangerous_ids": ["formation-3"],
        "unsupported_ids": ["formation-3"], "complete": False,
    }
    assert result["label_coverage"]["formation"]["labeled"] == 1
    assert len(result["label_coverage"]["formation"]["missing_ids"]) == 9


@pytest.mark.parametrize("observed", [0, 1, 8])
def test_partial_clean_formation_cannot_pass_or_fill_missing_with_zero(synthetic_assessment, observed):
    original = synthetic_assessment[3]["formation"]
    synthetic_assessment[3]["formation"] = {key: original[key] for key in list(original)[:observed]}
    result = assessed(synthetic_assessment)
    assert result["mechanical_gates"]["formation_gate"] == result["mechanical_gates"]["protection_gate"] == "not_evaluable"
    assert result["mechanical_gates"]["overall_gate"] == "not_evaluable"
    assert result["formation_rate"] is result["dangerous_false_extractions"] is result["unsupported_candidates"] is None
    assert result["observed_formation"]["complete"] is False


def test_low_exposure_cannot_hide_formation_or_protection_failure(synthetic_assessment):
    current = synthetic_assessment[2]
    current["exposure"].update(numerator=239, rate=239 / 300, gate="not_evaluable")
    synthetic_assessment[3]["formation"]["formation-9"]["authorization"] = 1
    result = assessed(synthetic_assessment)
    assert result["improved_cases"] == 4
    assert result["mechanical_gates"]["main_gate"] == "not_evaluable"
    assert result["mechanical_gates"]["formation_gate"] == result["mechanical_gates"]["protection_gate"] == "fail"
    assert result["mechanical_gates"]["overall_gate"] == "fail"


def test_format_failure_is_reported_when_both_label_packets_are_missing(synthetic_assessment):
    synthetic_assessment[2]["format"].update(fallbacks=10, gate="fail")
    for labels in synthetic_assessment[3].values():
        labels.clear()
    result = assessed(synthetic_assessment)
    assert result["format"]["gate"] == "fail"
    assert result["mechanical_gates"]["overall_gate"] == "fail"
    assert result["mechanical_gates"]["main_gate"] == result["mechanical_gates"]["formation_gate"] == "not_evaluable"


def test_complete_fixture_mechanical_success_never_claims_model_effect(synthetic_assessment):
    result = assessed(synthetic_assessment)
    assert set(result["mechanical_gates"].values()) == {"pass"}
    assert result["status"] == "labels_complete"
    assert result["formation_rate"] == 8 / 10
    assert result["dangerous_false_extractions"] == 0


def test_valid_zero_candidates_are_complete_but_do_not_form_any_cases(synthetic_assessment):
    synthetic_assessment[1]["formation"][1].clear()
    synthetic_assessment[3]["formation"].clear()
    result = assessed(synthetic_assessment)
    assert result["label_coverage"]["formation"] == {"complete": True, "expected": 0, "labeled": 0, "missing_ids": []}
    assert result["formation_rate"] == 0
    assert result["dangerous_false_extractions"] == result["unsupported_candidates"] == 0
    assert result["mechanical_gates"]["formation_gate"] == "fail"
    assert result["mechanical_gates"]["protection_gate"] == "pass"


def test_revision_write_preserves_old_assessment_and_is_exclusive(synthetic_assessment):
    original = synthetic_assessment[0] / "assessment.json"
    old_bytes = b'{"schema":"historical_original","do_not_change":true}\n'
    original.write_bytes(old_bytes)
    result = assessed(synthetic_assessment, write=True)
    revised = synthetic_assessment[0] / "assessment-v2.json"
    revised_bytes = revised.read_bytes()
    assert common.read_json(revised) == result
    assert original.read_bytes() == old_bytes
    with pytest.raises(FileExistsError):
        assessed(synthetic_assessment, write=True)
    assert original.read_bytes() == old_bytes
    assert revised.read_bytes() == revised_bytes


def test_invalid_first_labels_remain_missing_and_cannot_be_replaced(tmp_path, monkeypatch):
    item = {"id": "a" * 32, "content_sha256": "c" * 64,
            "content": {"proposal": {"do": "software only"}}}
    packet = {"schema": "e1cv2_formation_packet_v1", "items": [item]}
    manifest = {"author_id": "software_author"}
    bundles = {"formation": (packet, {item["id"]: {"case_id": "software-case"}})}
    monkeypatch.setattr(scoring, "load_bundle", lambda *_: (manifest, bundles))
    invalid = scoring.formation.label_template(packet)
    invalid["scorer"] = {"id": "software_independent", "independent": True, "authored_cases": False}
    raw = common.canonical(invalid)
    receipt = scoring.import_first_batch(tmp_path, "b" * 64, raw, kind="formation",
                                         batch_id="software-first", expected_ids=[item["id"]])
    assert receipt["status"] == "invalid_first"
    assert scoring._labels(tmp_path, "formation", packet, author_id="software_author", bundle_sha256="b" * 64) == {}
    retained = tmp_path / "formation/private/first-batches/software-first/raw.json"
    invalid["labels"][0]["labels"] = {key: 0 for key in scoring.formation.LABEL_KEYS}
    with pytest.raises(ValueError, match="already_reserved"):
        scoring.import_first_batch(tmp_path, "b" * 64, common.canonical(invalid), kind="formation",
                                   batch_id="software-second", expected_ids=[item["id"]])
    assert retained.read_bytes() == raw
