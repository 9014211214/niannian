"""An equivalent wire grammar reduction keeps the full local admission gate."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from aria.engine.memory.reflection_contract import reflection_response_schema
from aria.engine.policy import reflection_json
from tests.test_reflection_structured_contract_v23 import payload

OLD_NAMES = ("_StatedPreference", "_ObservedResponse", "_RepeatedPattern")
NEW_NAMES = ("_AtomicClaimNullOutcome", "_AtomicClaimObservedOutcome")


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_exact_reviewed_wire_and_unchanged_full_local_schema():
    # These hashes identify the approved candidate and the independently
    # frozen original contract, not a schema read from local experiment files.
    assert reflection_json.REFLECTION_JSON_PROTOCOL_VERSION == "reflection_json_schema_v2"
    assert digest(reflection_json.reflection_wire_schema()) == "15f3553324acf254b4bd184db3eaec072084e238ddbde3b61a57412421022012"
    assert digest(reflection_response_schema()) == "b5ea411892d9e12c442e8bfafb0b762a036c5aaeb63e3646e083108a704cde9f"
    format_ = reflection_json.reflection_request_format()
    assert format_["type"] == "json_schema"
    assert format_["json_schema"]["name"] == "aria_reflection_v1" and format_["json_schema"]["strict"] is True


def test_only_atomic_subgraph_changes_and_every_call_is_fresh():
    full = reflection_response_schema()
    original_full = deepcopy(full)
    original_wire = reflection_json._project_wire_keywords(full)
    assert digest(original_wire) == "50fc2d963a2c901ce27dd027650609dbf3854fe2c28917a429205d17a184b9a4"
    expected_old = deepcopy(original_wire)
    reduced = reflection_json._factor_atomic_outcomes(original_wire)
    restored = deepcopy(reduced)
    for name in NEW_NAMES:
        del restored["$defs"][name]
    for name in OLD_NAMES:
        restored["$defs"][name] = deepcopy(original_wire["$defs"][name])
    restored["$defs"]["_Change"]["properties"]["atomic_claim"] = deepcopy(
        original_wire["$defs"]["_Change"]["properties"]["atomic_claim"])
    assert restored == original_wire == expected_old
    assert full == original_full == reflection_response_schema()
    reduced["$defs"].clear()
    assert digest(reflection_json.reflection_wire_schema()) == "15f3553324acf254b4bd184db3eaec072084e238ddbde3b61a57412421022012"


@pytest.mark.parametrize("mutation", ["required", "open_object", "common_text", "nullable_shape",
    "basis", "atomic_reference", "outside_reference", "name_collision"])
def test_incompatible_future_atomic_contract_is_not_silently_factored(mutation):
    wire = reflection_json._project_wire_keywords(reflection_response_schema())
    defs = wire["$defs"]
    if mutation == "required":
        defs["_RepeatedPattern"]["required"].remove("conditions")
    elif mutation == "open_object":
        defs["_RepeatedPattern"]["additionalProperties"] = True
    elif mutation == "common_text":
        defs["_RepeatedPattern"]["properties"]["conditions"]["pattern"] = "different"
    elif mutation == "nullable_shape":
        defs["_RepeatedPattern"]["properties"]["outcome"]["anyOf"].append({"type": "number"})
    elif mutation == "basis":
        defs["_StatedPreference"]["properties"]["evidence_basis"]["const"] = "unreviewed_basis"
    elif mutation == "atomic_reference":
        defs["_Change"]["properties"]["atomic_claim"]["anyOf"].reverse()
    elif mutation == "outside_reference":
        wire["properties"]["unreviewed"] = {"$ref": "#/$defs/_ObservedResponse"}
    else:
        defs[NEW_NAMES[0]] = {"type": "object"}
    before = deepcopy(wire)
    with pytest.raises(ValueError, match="incompatible_reflection_atomic_schema"):
        reflection_json._factor_atomic_outcomes(wire)
    assert wire == before


@pytest.mark.parametrize("basis,outcome,accepted", [
    ("stated_preference", None, True), ("stated_preference", "observed text", False),
    ("observed_response", "observed text", True), ("observed_response", None, False),
    ("repeated_pattern", None, True), ("repeated_pattern", "observed text", True),
])
def test_factored_wire_preserves_original_atomic_relation_and_raw_return(basis, outcome, accepted):
    reflection_json.reflection_wire_schema()
    value = payload()
    value["reviews"][0]["evidence_basis"] = basis
    value["changes"][0]["atomic_claim"].update(evidence_basis=basis, outcome=outcome)
    raw = json.dumps(value, ensure_ascii=False)
    before = deepcopy(value)
    if accepted:
        assert reflection_json.parse_reflection_json(raw) == raw
    else:
        with pytest.raises(ValueError):
            reflection_json.parse_reflection_json(raw)
    assert value == before
