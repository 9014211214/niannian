"""Later retained prefixes preserve fixed evidence and missing-memory semantics."""
import pytest

from scripts import e1br_memory_qualification as subject
from scripts.run_e1b import RunError
from tests.test_e1br_format_qualification import full_metadata  # noqa: F401
from tests.test_e1br_memory_qualification import memory_metadata, fail_memory  # noqa: F401


def test_explicit_later_prefix_keeps_missing_memory_and_original_inventory(memory_metadata):
    data = memory_metadata
    data.retained = data.rows[:834]
    data.state["attempts"] = data.rows[834:]
    fail_memory(data)
    arguments = dict(retained_attempts=data.retained, retained_summary=data.retained_summary,
                     logical_attempts=lambda state: state["attempts"], retained_position_bindings=data.legacy)
    with pytest.raises(RunError, match="memory_retained_logical_denominator_changed"):
        subject.qualify(data.root, data.frozen, data.state, **arguments)
    result = subject.qualify(data.root, data.frozen, data.state, expected_retained=834, **arguments)
    assert result["retained_first_answers"] == result["source_lineage"]["retained_first_answers"] == 834
    assert result["new_physical_attempts"] == 222
    assert result["actual_counts"] == {"online_fast": 1008, "memory": 24, "reflection": 24}
    assert len(result["source_lineage"]["initial_positions"]) == 1056
    assert sum(row["retained"] for row in result["source_lineage"]["initial_positions"]) == 834
    assert len(result["scored_position_inventory"]) == 216
    assert result["missing_memory_positions"][0]["raw_proposals"] is None
    assert result["fixed_case_denominator"] == 12
    assert result["original_overall_gate"] == "not_evaluable"


@pytest.mark.parametrize("declared", [True, 834.0, 0, 833, 835, 1056])
def test_later_prefix_does_not_accept_changed_declared_count(memory_metadata, declared):
    data = memory_metadata
    with pytest.raises(RunError, match="memory_retained_logical_denominator_changed"):
        subject.qualify(data.root, data.frozen, data.state, expected_retained=declared,
            retained_attempts=data.rows[:834], retained_summary=data.retained_summary,
            logical_attempts=lambda state: state["attempts"])
