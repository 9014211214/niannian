"""Synthetic CSV tests for the retained CPED source loader; no real dataset."""
from __future__ import annotations

import csv
import pytest

from aria.bench.cped_source import CPED_TRAITS, load_cped_corpus, load_cped_manifest


_FIELDS = [
    "TV_ID", "Dialogue_ID", "Utterance_ID", "Speaker", "Gender", "Age",
    *CPED_TRAITS, "Scene", "FacePosition_LU", "FacePosition_RD",
    "Sentiment", "Emotion", "DA", "Utterance",
]


def _row(
    dialogue: str,
    index: int,
    speaker: str,
    utterance: str,
    *,
    emotion: str = "neutral",
    age: str = "elderly",
) -> dict[str, str]:
    return {
        "TV_ID": "1",
        "Dialogue_ID": dialogue,
        "Utterance_ID": f"{dialogue}_{index:03d}",
        "Speaker": speaker,
        "Gender": "unknown",
        "Age": age,
        "Neuroticism": "low",
        "Extraversion": "high",
        "Openness": "high",
        "Agreeableness": "high",
        "Conscientiousness": "high",
        "Scene": "home",
        "FacePosition_LU": "0_0",
        "FacePosition_RD": "0_0",
        "Sentiment": (
            "positive" if emotion == "happy"
            else ("neutral" if emotion == "neutral" else "negative")
        ),
        "Emotion": emotion,
        "DA": "statement-opinion",
        "Utterance": utterance,
    }


def _write_fixture(path):
    rows = []
    for dialogue_index in range(4):
        dialogue = f"1_{dialogue_index:03d}"
        rows.extend([
            _row(dialogue, 0, "邻居", "今天活动室人不少"),
            _row(dialogue, 1, "角色甲", "我先听听大家怎么说"),
            _row(
                dialogue, 2, "角色甲",
                ["这事真让我高兴", "我还是有点担心",
                 "别再催我了", "今天心里挺难受"][dialogue_index],
                emotion=["happy", "worried", "anger", "sadness"][
                    dialogue_index
                ],
            ),
            # 作为人格证据保留，但不参与情绪标签抽样。
            _row(
                dialogue, 3, "角色甲", "我总是主动把事情安排好",
                emotion="",
            ),
        ])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)



def test_local_cped_loader_preserves_source_and_license_metadata(tmp_path):
    source = tmp_path / "test_split.csv"
    _write_fixture(source)
    corpus = load_cped_corpus(source, acknowledge_unverified_data_license=True)
    assert len(corpus.rows) == 16
    assert corpus.source_files == ["test_split.csv"]
    assert len(corpus.sha256["test_split.csv"]) == 64
    assert corpus.rows[1].utterance == "我先听听大家怎么说"
    assert corpus.data_license == "NOASSERTION"
    assert corpus.data_license_status == "unverified"
    assert corpus.code_license == "Apache-2.0"
    assert corpus.license_acknowledged is True


def test_read_requires_explicit_terms_acknowledgement_before_source_access(tmp_path):
    with pytest.raises(PermissionError, match="不授予任何权利"):
        load_cped_corpus(tmp_path / "missing.csv")
    manifest = load_cped_manifest()
    assert manifest["data_license"]["spdx"] == "NOASSERTION"
    assert manifest["data_license"]["status"] == "unverified"
    assert manifest["code_license"]["spdx"] == "Apache-2.0"
