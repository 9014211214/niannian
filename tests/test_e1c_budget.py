"""Explicit software prices; no live prices, ledger access or spending approval."""
from decimal import Decimal

import pytest

from aria.bench.e1c_registration import FinancialInheritance, RoleCostEnvelope, estimate_e1c_budget


def inherited():
    return FinancialInheritance.model_validate({
        "schema_version": "e1c_finance_v1", "prior_e1b_closed": True,
        "prior_assessment_sha256": "a" * 64, "prior_ledger_sha256": "b" * 64,
        "known_cumulative_usd": "2", "known_e1b_usd": "1",
        "unknown_holds": [{"campaign": "SOFTWARE_ONLY", "request_id": f"SOFTWARE-ONLY-{i}",
                           "request_sha256": f"{i:064x}", "amount": "0.01"} for i in range(9)],
    })


def envelopes():
    return {role: RoleCostEnvelope.model_validate({"model": f"SOFTWARE_ONLY/{role}",
            "price_metadata_sha256": "c" * 64, "max_input_tokens": 1000, "max_output_tokens": 100,
            "input_usd_per_million": "1", "output_usd_per_million": "2"})
            for role in ("online", "memory", "reflection")}


def test_complete_cost_scenario_preserves_prior_holds_and_shared_retry_pool():
    result = estimate_e1c_budget(inherited(), envelopes(), reflection_format_correction=True,
                                infrastructure_retries=2)
    assert result["physical_maximum"] == 3192
    assert result["base_counts"] == {"online": 1008, "memory": 24, "reflection": 24}
    assert Decimal(result["additional_maximum_usd"]) == Decimal(3192) * Decimal("0.0012")
    assert result["inherited_unknown_count"] == 9 and result["inherited_unknown_hold_usd"] == "0.09"
    assert Decimal(result["committed_cumulative_plus_scenario_usd"]) == Decimal("2.09") + Decimal("3.8304")
    assert result["execution_authorized"] is False and result["judge_included"] is False


def test_default_scenario_neither_invents_retry_authorization_nor_judge_budget():
    result = estimate_e1c_budget(inherited(), envelopes())
    assert result["physical_maximum"] == 1056 and result["format_correction_count"] == 0
    assert result["shared_infrastructure_retries_per_logical"] == 0
    assert result["execution_authorized"] is False
    assert result["judge_configuration_requires_registration"] is True


@pytest.mark.parametrize("kwargs", [{"infrastructure_retries": 3}, {"infrastructure_retries": True},
                                    {"reflection_format_correction": 1}])
def test_implicit_or_unbounded_retry_scenarios_rejected(kwargs):
    with pytest.raises(ValueError):
        estimate_e1c_budget(inherited(), envelopes(), **kwargs)


def test_missing_role_and_unclosed_prior_rejected():
    roles = envelopes()
    roles.pop("memory")
    with pytest.raises(ValueError):
        estimate_e1c_budget(inherited(), roles)
    prior = inherited().model_copy(update={"prior_e1b_closed": False})
    with pytest.raises(ValueError, match="final_e1b_accounting"):
        estimate_e1c_budget(prior, envelopes())


def test_same_original_id_across_campaigns_keeps_two_distinct_unknown_holds():
    # Synthetic financial entries exercising the historical ID-collision shape.
    from aria.bench.e1c_registration import financial_preparation
    payload = inherited().model_dump(mode="json")
    payload["unknown_holds"] = [
        {"campaign": "e1_t73_20261001_v1", "request_id": "e1/e1_001",
         "request_sha256": "a" * 64, "amount": "0.0011"},
        {"campaign": "t6_20260930_v1", "request_id": "e1/e1_001",
         "request_sha256": "b" * 64, "amount": "0.0011"}]
    result = financial_preparation(FinancialInheritance.model_validate(payload))
    assert result["inherited_unknown_count"] == 2
    assert Decimal(result["inherited_unknown_hold_usd"]) == Decimal("0.0022")
    assert result["inherited_unknown_holds"] == payload["unknown_holds"]
    assert [row["request_id"] for row in result["inherited_unknown_holds"]] == ["e1/e1_001"] * 2
    assert result["execution_authorized"] is False


def test_same_campaign_id_cannot_be_duplicated_by_changing_wire_hash():
    payload = inherited().model_dump(mode="json")
    changed = {**payload["unknown_holds"][0], "request_sha256": "f" * 64}
    payload["unknown_holds"].append(changed)
    with pytest.raises(ValueError, match="invalid_inherited_financial_history"):
        FinancialInheritance.model_validate(payload)
