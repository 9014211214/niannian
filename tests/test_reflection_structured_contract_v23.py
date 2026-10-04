"""Local shape guarantees, separate from semantic or provider effectiveness."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from aria.engine.memory.diagnosis import reflection_response_schema
from aria.engine.memory.reflection_contract import validate_reflection_response_shape
from tests.test_reflection_evidence_review import apply, change, context, review, setup
from tests.test_reflection_belief_change_v23 import R8_RETIRE_RAW


def payload():
    ctx = context()
    item = review(ctx)
    return {"protocol_version": "2.0", "reviews": [item], "changes": [change(item)]}


def test_saved_r8_low_retire_shape_is_legal_without_rewriting_proposal():
    raw = json.loads(R8_RETIRE_RAW)
    before = deepcopy(raw)
    validate_reflection_response_shape(raw)
    assert raw == before
    assert raw["changes"][0]["confidence"] == .5


def test_schema_is_fresh_closed_and_all_nullable_fields_still_required():
    schema = reflection_response_schema()
    assert schema["title"] == "AriaOfflineEvidenceReview2_3"
    assert schema["properties"]["changes"]["maxItems"] == 1
    assert schema["properties"]["reviews"]["maxItems"] == 6
    for obj in [schema, *schema["$defs"].values()]:
        if obj.get("type") == "object":
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
    assert schema["$defs"]["_StatedPreference"]["properties"]["outcome"]["type"] == "null"
    assert schema["$defs"]["_ObservedResponse"]["properties"]["outcome"]["type"] == "string"
    assert len(schema["$defs"]["_Change"]["properties"]["atomic_claim"]["anyOf"]) == 3
    assert "task_request" in schema["$defs"]["_Change"]["properties"]["scope_tags"]["items"]["enum"]
    schema["$defs"].clear()
    assert reflection_response_schema()["$defs"]


@pytest.mark.parametrize("mutation", [
    "extra_root", "extra_review", "extra_atomic", "extra_quote", "missing_atomic", "empty_event_quotes",
    "mistyped_turn", "boolean_turn", "boolean_confidence", "nan_confidence", "infinite_confidence",
    "negative_confidence", "large_confidence", "preference_with_outcome", "response_without_outcome",
    "missing_nullable_outcome", "wrong_relation_enum", "wrong_scope_tag", "duplicate_scope_tag",
    "general_plus_context", "dimension_for_wrong_framework", "target_for_response", "extra_lineage",
    "non_json_null", "whitespace_condition", "overlong_condition", "unknown_atomic_basis",
])
def test_schema_locally_rejects_observed_structural_failures_without_repair(mutation):
    raw = payload()
    item, proposal = raw["reviews"][0], raw["changes"][0]
    atomic = proposal["atomic_claim"]
    if mutation == "extra_root":
        raw["comment"] = "extra"
    elif mutation == "extra_review":
        item["event_summary_note"] = "extra"
    elif mutation == "extra_atomic":
        atomic["allow_share"] = True
    elif mutation == "extra_quote":
        proposal["evidence"][0]["turn_index2"] = 1
    elif mutation == "missing_atomic":
        proposal["atomic_claim"] = None
    elif mutation == "empty_event_quotes":
        item["event_evidence"] = []
    elif mutation == "mistyped_turn":
        item["evidence"][0]["turn_index"] = "1"
    elif mutation == "boolean_turn":
        item["evidence"][0]["turn_index"] = True
    elif mutation == "boolean_confidence":
        proposal["confidence"] = True
    elif mutation == "nan_confidence":
        proposal["confidence"] = float("nan")
    elif mutation == "infinite_confidence":
        proposal["confidence"] = float("inf")
    elif mutation == "negative_confidence":
        proposal["confidence"] = -.001
    elif mutation == "large_confidence":
        proposal["confidence"] = 1.001
    elif mutation == "preference_with_outcome":
        atomic["outcome"] = "没发生但我猜会开心"
    elif mutation == "response_without_outcome":
        atomic["evidence_basis"] = "observed_response"
    elif mutation == "missing_nullable_outcome":
        del atomic["outcome"]
    elif mutation == "wrong_relation_enum":
        item["relation"] = "same_event"
    elif mutation == "wrong_scope_tag":
        proposal["scope_tags"] = ["low_pressure"]
    elif mutation == "duplicate_scope_tag":
        proposal["scope_tags"] = ["task_request", "task_request"]
    elif mutation == "general_plus_context":
        proposal["scope_tags"] = ["general", "task_request"]
    elif mutation == "dimension_for_wrong_framework":
        proposal["dimension"] = "openness"
    elif mutation == "target_for_response":
        proposal["target_value"] = .9
    elif mutation == "extra_lineage":
        proposal["lineage"] = {
            "id": "old", "claim_quote": "claim", "scope_quote": "scope",
            "mechanism_relation": "same", "scope_relation": "within", "claim_quote_check": True,
        }
    elif mutation == "non_json_null":
        atomic["outcome"] = "null"
    elif mutation == "whitespace_condition":
        atomic["conditions"] = " \t\n "
    elif mutation == "overlong_condition":
        atomic["conditions"] = "场" * 101
    else:
        atomic["evidence_basis"] = "context_only"
    before = deepcopy(raw)
    with pytest.raises(ValidationError):
        validate_reflection_response_shape(raw)
    assert raw == before


@pytest.mark.parametrize("basis,outcome", [
    ("stated_preference", None), ("observed_response", "本人说这种讲法听起来舒服"),
    ("repeated_pattern", None), ("repeated_pattern", "本人描述每次都有类似反应"),
])
def test_each_atomic_evidence_basis_has_an_explicit_legal_outcome_shape(basis, outcome):
    raw = payload()
    raw["reviews"][0]["evidence_basis"] = basis
    raw["changes"][0]["atomic_claim"].update(evidence_basis=basis, outcome=outcome)
    validate_reflection_response_shape(raw)


def test_structural_schema_pass_does_not_assert_quote_grounding_or_scope_truth():
    raw = payload()
    raw["reviews"][0]["evidence"][0]["evidence_quote"] = "合法字符串但当前用户根本没讲过"
    validate_reflection_response_shape(raw)  # syntax is not source entailment
    updater, llm, persona = setup()
    llm.payload = raw
    updater.update(persona, context())
    assert updater.last_status == "failed" and not persona.person_hypotheses
    assert updater.last_belief_change_results is None


def test_unknown_is_structurally_valid_without_promoting_to_support():
    raw = payload()
    raw["reviews"][0].update(applicability="unknown", relation="unknown", event_kind="unknown")
    raw["changes"] = []
    validate_reflection_response_shape(raw)
    updater, llm, persona = setup()
    llm.payload = raw
    updater.update(persona, context())
    assert updater.last_status == "ok" and not persona.person_hypotheses
    assert updater.last_belief_change_results == []


def test_new_generation_one_change_does_not_retroactively_break_old_multi_proposal_wire():
    updater, llm, persona = setup()
    ctx = context()
    first = review(ctx, id="one")
    second = review(ctx, id="two")
    one, two = change(first), change(second)
    two["atomic_claim"]["tendency"] = "更偏好由自己控制讲解速度"
    raw = {"protocol_version": "2.0", "reviews": [first, second], "changes": [one, two]}
    with pytest.raises(ValidationError):
        validate_reflection_response_shape(raw)
    apply(updater, llm, persona, ctx, [first, second], [one, two])
    assert updater.last_status == "ok" and len(persona.person_hypotheses) == 1
    assert len(updater.last_belief_change_results) == 1
    assert updater.last_application_summary["applied_count"] == 1
    assert updater.last_application_summary["deferred_count"] == 1
