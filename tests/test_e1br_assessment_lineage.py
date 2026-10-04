"""Continuation accounting is derived from sealed lineage, never a fixed count."""
from copy import deepcopy
from decimal import Decimal

import pytest

from scripts import e1br_assessment as assessment


def fixture(count):
    hold = "0.320307142" if count == 8 else "0.321493014"
    budget = {"accepted_old_unknown": count, "historical_unknown_reserved_usd": hold}
    return {"protocol": {"budget": budget}}, deepcopy(budget), {
        "inherited_unknown_requests": count, "historical_unknown_reserved_usd": hold}


@pytest.mark.parametrize("count", [8, 9])
def test_each_version_reports_its_own_retained_unknown_count(monkeypatch, count):
    frozen, auth, receipt = fixture(count)
    hold = auth["historical_unknown_reserved_usd"]
    monkeypatch.setattr(assessment.revision, "RETAINED_UNKNOWN_HOLD", Decimal(hold))
    assert assessment.retained_accounting(frozen, auth, receipt) == (count, hold)


@pytest.mark.parametrize("mutation", ["receipt", "authorization", "float", "bool", "negative", "hold", "receipt_hold", "prior_hold"])
def test_changed_lineage_or_cleared_unknown_is_rejected(monkeypatch, mutation):
    frozen, auth, receipt = fixture(9)
    monkeypatch.setattr(assessment.revision, "RETAINED_UNKNOWN_HOLD", Decimal("0.321493014"))
    if mutation == "receipt":
        receipt["inherited_unknown_requests"] = 8
    elif mutation == "authorization":
        auth["accepted_old_unknown"] = 8
    elif mutation == "float":
        receipt["inherited_unknown_requests"] = 9.0
    elif mutation == "bool":
        frozen["protocol"]["budget"]["accepted_old_unknown"] = True
    elif mutation == "negative":
        frozen["protocol"]["budget"]["accepted_old_unknown"] = -1
    elif mutation == "hold":
        auth["historical_unknown_reserved_usd"] = "0"
    elif mutation == "receipt_hold":
        receipt["historical_unknown_reserved_usd"] = "0"
    else:
        monkeypatch.setattr(assessment.revision, "RETAINED_UNKNOWN_HOLD", Decimal("0"))
    with pytest.raises(assessment.revision.RunError, match="retained_accounting_binding_changed"):
        assessment.retained_accounting(frozen, auth, receipt)


@pytest.mark.parametrize("flags", [{"software_fixture_only": True}, {"software_fixture_only": "false"},
    {"software_fixture_only": 0}, {"new_calls_are_real_models": False}, {"new_calls_are_real_models": None},
    {"new_calls_are_real_models": 1}])
def test_software_or_ambiguous_generation_cannot_claim_real_effects(tmp_path, flags):
    with pytest.raises(assessment.revision.RunError, match="software_receipt"):
        assessment.require_real_generation(tmp_path, flags)


def test_software_directory_marker_blocks_even_when_receipt_flags_are_removed(tmp_path):
    (tmp_path / "SOFTWARE_ONLY.json").write_text("{}", encoding="utf-8")
    with pytest.raises(assessment.revision.RunError, match="software_receipt"):
        assessment.require_real_generation(tmp_path, {})


@pytest.mark.parametrize("flags", [{}, {"software_fixture_only": False, "new_calls_are_real_models": True}])
def test_real_generation_still_requires_the_remaining_full_assessment_gates(tmp_path, flags):
    # Passing this metadata gate is not evidence of completed labels or effects.
    assert assessment.require_real_generation(tmp_path, flags) is None
