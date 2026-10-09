"""Character accounting is software evidence, never token or model evidence."""
import json

import pytest

from aria.bench.prompt_footprint import (
    fixture_report, measure_messages, measure_output_fields,
)


def test_partition_covers_whitespace_unicode_and_nested_headings_exactly():
    content = "身份😀\n\n## 规则\n内容\n\n【相处画像:自然】\n条目\n\n## 数据\n尾\n"
    result = measure_messages([{"role": "system", "content": content},
                               {"role": "user", "content": "输入"}])
    blocks = result["system_blocks"]
    assert sum(row["characters"] for row in blocks) == len(content)
    assert blocks[0]["start"] == 0
    assert blocks[-1]["end"] == len(content)
    assert all(a["end"] == b["start"] for a, b in zip(blocks, blocks[1:]))
    assert result["message_content_characters"] == len(content) + 2
    assert "身份" not in json.dumps(result, ensure_ascii=False)


def test_missing_is_not_zero_and_private_reasoning_never_enters_report():
    result = measure_output_fields([
        {"response": "😀", "deliberation": None, "reasoning_content": "PRIVATE"},
        {"response": "abc", "deliberation": {"person_read": "秘密"}},
        {"response": float("nan")}, {},
    ])
    response = result["fields"]["response"]
    assert response["observed"] == 2
    assert response["mean"] == 4  # JSON strings: 3 and 5 Unicode code points.
    assert response["missing"] == 1
    assert response["invalid_json_value"] == 1
    assert result["fields"]["deliberation"]["null"] == 1
    encoded = json.dumps(result, ensure_ascii=False)
    assert "PRIVATE" not in encoded and "秘密" not in encoded
    assert "reasoning_content" not in encoded
    assert measure_output_fields([])["fields"]["response"]["mean"] is None


def test_fixture_measurements_are_reproducible_and_explicitly_scoped(monkeypatch):
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "profile_v1")
    first = fixture_report()
    monkeypatch.setenv("ARIA_DELIVERY_BLOCK", "mode_v1")
    second = fixture_report()
    assert first == second
    assert len(first["input_measurements"]) == 8
    assert first["output_measurements"]["samples"] == 0
    assert first["model_calls"] == 0
    for row in first["input_measurements"]:
        assert sum(b["characters"] for b in row["system_blocks"]) == row["content_characters_by_role"]["system"]
        assert row["prompt_version"] == ("1.18.0" if row["delivery_block"] == "mode_v1" else "1.19.0")


def test_non_text_messages_are_rejected_instead_of_undercounted():
    with pytest.raises(ValueError, match="text_messages"):
        measure_messages([{"role": "user", "content": [{"type": "image"}]}])


def test_nested_field_distributions_count_elements_and_keep_raw_text_private():
    result = measure_output_fields([
        {"state_signals": [{"rationale": "甲乙"}, {"rationale": "丙"}],
         "communication_plan": {"support_order": "先听再问"}},
        {"state_signals": []}, {},
    ])
    rationale = result["fields"]["state_signals.*.rationale"]
    assert rationale["observed"] == 2
    assert rationale["missing"] == 2
    assert rationale["string_content_characters"]["mean"] == 1.5
    assert result["fields"]["communication_plan.support_order"]["string_content_characters"]["mean"] == 4
    assert "先听再问" not in json.dumps(result, ensure_ascii=False)
