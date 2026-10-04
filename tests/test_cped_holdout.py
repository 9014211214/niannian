"""Authored source-shaped fixtures, not CPED or extracted TV dialogue."""
from dataclasses import replace
import json
import random

import pytest

from aria.bench.cped_holdout import (
    assert_aggregate_only_preparation,
    prepare_cped_holdout,
    surface_baselines,
)
from aria.bench.cped_preflight import REQUIRED_CASE_AUDITS
from aria.bench.cped_source import CPEDRow, CPED_TRAITS


def _fixture(count=64):
    activities = ("公园散步", "书房阅读", "院子种花", "厨房做菜", "广场听歌", "河边钓鱼", "窗前画画", "客厅下棋")
    feelings = ("让我安心", "带来乐趣", "缓解烦闷", "使我舒坦", "令人放松", "唤起回忆", "增添兴致", "带动心情")
    rows = []
    for index in range(count):
        for dialogue in range(4):
            for turn in range(6):
                target = turn >= 2
                # Distinct source content across all fragments, but shared
                # generic vocabulary permits matched hard distractors.
                variation = chr(0x4E00 + index) + chr(0x5200 + dialogue * 4 + max(turn - 2, 0))
                text = (
                    f"今天{activities[index % 8]}{feelings[index // 8 % 8]}{variation}。"
                    if target else f"聊聊最近的不同经历{variation}{turn}。"
                )
                rows.append(CPEDRow(
                    tv_id=f"show-{index}", dialogue_id=str(dialogue),
                    utterance_id=f"utt_{turn}", speaker=f"name-{index}" if target else f"partner-{index}",
                    age="elderly" if index < 13 else "middle-aged",
                    traits={trait: "high" for trait in CPED_TRAITS},
                    sentiment="neutral", emotion="neutral", dialogue_act="statement-opinion",
                    utterance=text, split="train",
                ))
    return rows


def _run(rows=None, **kwargs):
    return prepare_cped_holdout(
        _fixture() if rows is None else rows,
        terms_review_acknowledged=True, aggregate_only_accepted=True, **kwargs,
    )


def test_source_builder_constructs_independent_heldout_cases_and_real_counts():
    result = _run()
    assert result.report["selected_role_count"] == 48
    assert len({case.role_key for case in result.cases}) == 48
    assert result.report["position_counts"] == [12] * 4
    assert result.report["surface_correct_counts"] == surface_baselines(result.cases)
    assert result.report["surface_correct_counts"]["position"] == 12
    assert result.report["executed_model_calls"] == 0
    assert result.report["cloud_submission_allowed"] is False
    assert result.report["identity_review_required"] is True
    for case in result.cases:
        assert len(case.formation_texts) >= 12
        assert case.holdout_dialogue not in case.formation_dialogues
        assert len(set(case.candidates)) == 4
        assert case.target_source_key not in case.context_source_keys
        assert not (set(case.formation_source_keys) & set(case.context_source_keys))
        assert all(key[0] != case.role_key[0] for key in case.distractor_source_keys)
        assert len({key[:2] for key in case.distractor_source_keys}) == 3
        assert all(int(key[2].split("_")[-1]) < int(case.target_source_key[2].split("_")[-1]) for key in case.context_source_keys)
        # A mechanical builder must not impersonate a complete entity review.
        assert case.metadata().completed_audits < REQUIRED_CASE_AUDITS


def test_same_input_in_a_different_order_gives_identical_cohort_and_fingerprint():
    rows = _fixture()
    shuffled = list(rows)
    random.Random(81).shuffle(shuffled)
    a, b = _run(rows), _run(shuffled)
    assert a.cases == b.cases
    assert a.report == b.report
    assert a.private_cohort_fingerprint == b.private_cohort_fingerprint


@pytest.mark.parametrize("field", ["terms_review_acknowledged", "aggregate_only_accepted"])
def test_terms_gate_never_touches_source_or_inventory(field):
    class NoRead:
        def __iter__(self):
            raise AssertionError("read before terms acknowledgement")
    kwargs = {"terms_review_acknowledged": True, "aggregate_only_accepted": True}
    kwargs[field] = False
    result = prepare_cped_holdout(NoRead(), extra_identity_terms=NoRead(), **kwargs)
    assert result.report["status"] == "stopped_terms"
    assert not result.cases


def test_identical_cross_split_rows_cannot_inflate_evidence():
    rows = _fixture()
    original = _run(rows)
    duplicate = _run(rows + [replace(row, split="test") for row in rows])
    assert duplicate.cases == original.cases
    assert duplicate.report["duplicate_source_row_count"] == len(rows)
    assert duplicate.report["unique_source_row_count"] == len(rows)


def test_conflicting_duplicate_source_identity_fails_closed_without_private_output():
    rows = _fixture(1)
    result = _run(rows + [replace(rows[0], utterance="sensitive conflicting text")])
    assert result.report["status"] == "stopped_source_integrity"
    assert not result.cases
    assert result.report["failure_counts"]["conflicting_source_identity"] == 1
    assert "sensitive" not in json.dumps(result.report)


@pytest.mark.parametrize("utterance_id", ["non-numeric", "other_1"])
def test_ambiguous_or_nonnumeric_turn_order_stops(utterance_id):
    rows = _fixture(1)
    rows[0] = replace(rows[0], utterance_id=utterance_id)
    assert _run(rows).report["status"] == "stopped_source_integrity"


def test_corrupt_rows_and_traits_do_not_get_silently_filtered_into_a_good_run():
    for row in (object(), replace(_fixture(1)[0], traits={}), replace(_fixture(1)[0], tv_id="")):
        result = _run([row])
        assert result.report["status"] == "stopped_source_integrity"
        assert result.report["failure_counts"]["invalid_source_row"] == 1


def test_insufficient_role_count_and_elderly_slice_are_not_filled_by_repeats():
    result = _run(_fixture(13))
    assert result.report["status"] == "stopped_feasibility"
    assert result.report["selected_role_count"] <= 13
    assert result.report["elderly_role_count"] <= 13
    assert "accuracy" not in json.dumps(result.report)


def test_inconsistent_labels_reject_role_before_selection():
    rows = _fixture()
    rows[2] = replace(rows[2], traits={trait: "low" for trait in CPED_TRAITS})
    result = _run(rows)
    assert ("show-0", "name-0") not in {case.role_key for case in result.cases}
    assert result.report["failure_counts"]["inconsistent_role_labels"] == 1


def test_punctuation_only_history_duplicates_cannot_satisfy_twelve_utterances():
    rows = _fixture(1)
    rows = [replace(row, utterance="同一句合成原话" + "。" * (int(row.utterance_id[-1]) + 1)) if row.speaker == "name-0" else row for row in rows]
    result = _run(rows)
    assert not result.cases
    assert result.report["failure_counts"]["insufficient_unique_formation"] >= 1


def test_known_entities_masked_uniformly_without_claiming_entity_completeness():
    rows = [replace(row, utterance="name-0虚构城" + row.utterance) for row in _fixture()]
    result = _run(rows, extra_identity_terms=("虚构城",))
    assert result.cases
    for case in result.cases:
        model_material = " ".join((*case.formation_texts, *(text for _, text in case.context), *case.candidates))
        assert "name-0" not in model_material and "虚构城" not in model_material
        assert "[名称]" in model_material
    assert result.report["identity_review_required"] is True


def test_matching_requires_other_shows_and_actual_label_matches():
    rows = _fixture()
    # All source participants in one show cannot supply other-show distractors.
    same_show = [replace(row, tv_id="one-show", dialogue_id=f"{row.tv_id}/{row.dialogue_id}") for row in rows]
    result = _run(same_show)
    assert not result.cases
    assert result.report["failure_counts"]["no_matched_distractor_triplet"] > 0
    unique_emotions = [replace(row, emotion=row.tv_id) for row in rows]
    assert not _run(unique_emotions).cases


def test_three_other_speakers_in_one_foreign_fragment_are_not_independent_distractors():
    rows = _fixture(8)
    # Keep the first role's four complete fragments. All other source rows
    # become distinct speakers in just one foreign fragment, with unique turn
    # order. There are enough matched texts/roles, but not three fragments.
    target = [row for row in rows if row.tv_id == "show-0"]
    foreign = [replace(row, tv_id="foreign", dialogue_id="only-one",
                       speaker=f"foreign-person-{i}", utterance_id=f"utt_{i}")
               for i, row in enumerate(row for row in rows if row.tv_id != "show-0")]
    result = _run(target + foreign)
    assert not result.cases
    assert result.report["failure_counts"]["no_matched_distractor_triplet"] >= 1


def test_surface_baselines_do_not_see_correct_position_and_detect_obvious_shortcut():
    case = _run().cases[0]
    case = replace(case, context=(("对话者1", "短句"),), candidates=("短句", "更长的不同句子", "还有很长的句子内容", "最后一个不相关表达"), correct_position=0)
    scores = surface_baselines([case])
    assert scores["length"] == scores["lexical"] == 1
    different_gold = replace(case, correct_position=3)
    assert surface_baselines([different_gold])["length"] == 0


@pytest.mark.parametrize("key,value", [
    ("cases", ["private"]), ("seed", "private"), ("status", "private"),
    ("failure_counts", {"private-role": 1}), ("private_digest", "0" * 64),
    ("surface_correct_counts", {"length": True}), ("source_row_count", True),
    ("cloud_submission_allowed", True), ("executed_model_calls", 1),
])
def test_public_report_rejects_private_derivatives_arbitrary_metadata_and_wrong_types(key, value):
    report = _run([]).report
    report[key] = value
    with pytest.raises(ValueError, match="aggregate-only"):
        assert_aggregate_only_preparation(report)


def test_repr_of_preparation_or_cases_does_not_dump_private_text_or_ids():
    result = _run()
    text = repr(result) + repr(result.cases[0])
    assert "今天" not in text and "show-" not in text and "name-" not in text
    assert result.private_cohort_fingerprint not in text
    assert_aggregate_only_preparation(result.report)


@pytest.mark.parametrize("seed", [True, -1, 2**32, "17"])
def test_invalid_seeds_cannot_change_or_hide_selection_policy(seed):
    with pytest.raises(ValueError, match="seed"):
        _run([], seed=seed)
