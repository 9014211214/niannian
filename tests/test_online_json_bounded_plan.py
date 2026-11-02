"""Bounded plan guidance and strict parsing, using synthetic schema fixtures."""
from copy import deepcopy
import json

import pytest

from aria.engine.policy import online_json
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSIONS, ONLINE_DECISION_VERSION_V7, online_decision_tool_spec,
    validate_communication_plan,
)


def _schema(version):
    return online_decision_tool_spec(version)["function"]["parameters"]


def _synthetic_value(schema):
    """Type-only data; never an answer or a prompt example."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema["type"]
    if kind == "object":
        return {key: _synthetic_value(schema["properties"][key]) for key in schema["required"]}
    if kind == "array":
        return [_synthetic_value(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "x" * max(schema.get("minLength", 0), 1)
    if kind == "number":
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    raise AssertionError(kind)


def _value_with_plan(version):
    schema = _schema(version)
    value = _synthetic_value(schema)
    # The v7 plan is optional; include it explicitly when testing its bounds.
    value["communication_plan"] = _synthetic_value(schema["properties"]["communication_plan"])
    return value


def _messages():
    return [{"role": "system", "content": "fixture product instruction"},
            {"role": "user", "content": "fixture history input"},
            {"role": "assistant", "content": "fixture history output"},
            {"role": "user", "content": "fixture current input"}]


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
def test_plan_guidance_exposes_existing_bounds_without_sample_decisions(version):
    schema = _schema(version)
    before = deepcopy(schema)
    guidance = online_json.bounded_plan_contract(version)
    for key in ("support_order", "boundary"):
        field = schema["properties"]["communication_plan"]["properties"][key]
        assert (f"$.communication_plan.{key}：至少{field['minLength']}个字符、"
                f"至多{field['maxLength']}个字符（含标点和空格）。") in guidance
    assert "非空且非纯空白" in guidance
    assert "不是通用政策、安全规则或系统提示的复述；不要粘贴整段提示" in guidance
    assert "本轮简短先后安排和沟通行动边界" in guidance
    assert online_json.ONLINE_JSON_PLAN_VERSION in guidance
    assert schema == before == _schema(version)
    for sample in ('"pace":', '"support_order":', '"boundary":', '"response":'):
        assert sample not in guidance


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
def test_plan_guidance_reads_each_field_limit_independently(version, monkeypatch):
    def different_bounds(requested):
        assert requested == version
        spec = online_decision_tool_spec(requested)
        fields = spec["function"]["parameters"]["properties"]["communication_plan"]["properties"]
        fields["support_order"].update(minLength=2, maxLength=17)
        fields["boundary"].update(minLength=3, maxLength=29)
        return spec
    monkeypatch.setattr(online_json, "online_decision_tool_spec", different_bounds)
    guidance = online_json.bounded_plan_contract(version)
    assert "support_order：至少2个字符、至多17个字符" in guidance
    assert "boundary：至少3个字符、至多29个字符" in guidance


@pytest.mark.parametrize("source_version", ONLINE_DECISION_VERSIONS)
@pytest.mark.parametrize("target_version", ONLINE_DECISION_VERSIONS)
@pytest.mark.parametrize("source_kind", ["carrier", "structure_v1", "bounded_plan"])
def test_plan_contract_upgrade_and_version_rebind_preserve_history(
    source_version, target_version, source_kind,
):
    messages = _messages()
    before = deepcopy(messages)
    prepared = online_json.prepare_json_messages(messages, source_version)
    if source_kind != "bounded_plan":
        contract = online_json._carrier_contract(source_version)
        if source_kind == "structure_v1":
            suffix = (online_json.SCHEMA_MARKER + online_json._SCHEMA_INSTRUCTION
                      + online_json._canonical(_schema(source_version)))
            assert contract.endswith(suffix)
            contract = (contract[:-len(suffix)]
                        + online_json.schema_field_contract(source_version) + suffix)
        prepared[0]["content"] = messages[0]["content"] + contract
    prepared_before = deepcopy(prepared)
    rebound = online_json.prepare_json_messages(prepared, target_version)
    assert rebound == online_json.prepare_json_messages(messages, target_version)
    assert prepared == prepared_before and messages == before
    assert online_json.prepare_json_messages(rebound, target_version) == rebound
    assert json.loads(rebound[1]["content"]) == json.loads(prepared[1]["content"])
    system = rebound[0]["content"]
    assert system.count(online_json.ONLINE_JSON_PLAN_VERSION) == 1
    assert system.count(online_json.ONLINE_JSON_STRUCTURE_VERSION) == 1
    assert system.endswith(online_json._canonical(_schema(target_version)))


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
def test_plan_markers_and_quoted_contract_do_not_erase_user_or_system_content(version):
    messages = _messages()
    marker = "## 当轮沟通计划长度合同：" + online_json.ONLINE_JSON_PLAN_VERSION
    messages[0]["content"] += "\n" + marker + "\nfixture rule after marker"
    messages[1]["content"] = online_json._format_contract(version)
    messages[-1]["content"] = marker
    result = online_json.prepare_json_messages(messages, version)
    assert result[0]["content"].startswith(messages[0]["content"])
    quoted = json.loads(result[1]["content"])
    assert quoted["conversation_history"] == messages[1:-1]
    assert quoted["current_user_message"] == marker
    assert online_json.prepare_json_messages(result, version) == result


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
@pytest.mark.parametrize("field", ["support_order", "boundary"])
@pytest.mark.parametrize("length", [0, 1, 80, 81])
def test_plan_parser_keeps_length_boundaries_without_repair(version, field, length):
    value = _value_with_plan(version)
    value["communication_plan"][field] = "界" * length
    raw = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if 1 <= length <= 80:
        assert online_json.parse_json_decision(raw, version) == raw
        assert validate_communication_plan(value["communication_plan"]) == value["communication_plan"]
        if version == ONLINE_DECISION_VERSION_V7:
            parsed, dropped = online_json.parse_v7_payload(raw)
            assert parsed["communication_plan"] == value["communication_plan"] and dropped == []
    else:
        if version == ONLINE_DECISION_VERSION_V7:
            assert online_json.parse_json_decision(raw, version) == raw
            parsed, dropped = online_json.parse_v7_payload(raw)
            assert "communication_plan" not in parsed and dropped == ["communication_plan"]
            assert parsed["response"] == "".join(row["text"] for row in value["response_segments"])
        else:
            with pytest.raises(ValueError, match="invalid_online_json_schema"):
                online_json.parse_json_decision(raw, version)
        assert validate_communication_plan(value["communication_plan"]) is None
    assert value["communication_plan"][field] == "界" * length


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
@pytest.mark.parametrize("field", ["support_order", "boundary"])
def test_plan_whitespace_still_rejected_by_existing_product_gate(version, field):
    value = _value_with_plan(version)
    value["communication_plan"][field] = " \t\n" * 20
    raw = json.dumps(value, ensure_ascii=False)
    # JSON minLength counts whitespace; the existing product gate rejects it.
    assert online_json.parse_json_decision(raw, version) == raw
    assert validate_communication_plan(value["communication_plan"]) is None
    if version == ONLINE_DECISION_VERSION_V7:
        parsed, dropped = online_json.parse_v7_payload(raw)
        assert "communication_plan" not in parsed and dropped == ["communication_plan"]
        assert parsed["response"] == "".join(row["text"] for row in value["response_segments"])


@pytest.mark.parametrize("version", ONLINE_DECISION_VERSIONS)
def test_plan_limits_count_punctuation_spaces_and_unicode_characters(version):
    value = _value_with_plan(version)
    text = "界 .🙂" * 20
    assert len(text) == 80 and len(text.encode("utf-8")) > 80
    value["communication_plan"].update(support_order=text, boundary=text)
    raw = json.dumps(value, ensure_ascii=False)
    assert online_json.parse_json_decision(raw, version) == raw
    if version == ONLINE_DECISION_VERSION_V7:
        parsed, dropped = online_json.parse_v7_payload(raw)
        assert parsed["communication_plan"] == value["communication_plan"] and dropped == []
    value["communication_plan"]["boundary"] += " "
    raw = json.dumps(value, ensure_ascii=False)
    if version == ONLINE_DECISION_VERSION_V7:
        assert online_json.parse_json_decision(raw, version) == raw
        parsed, dropped = online_json.parse_v7_payload(raw)
        assert "communication_plan" not in parsed and dropped == ["communication_plan"]
        assert parsed["response"] == "".join(row["text"] for row in value["response_segments"])
    else:
        with pytest.raises(ValueError, match="invalid_online_json_schema"):
            online_json.parse_json_decision(raw, version)
