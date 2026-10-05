"""E1c inventory and carried finance preparation, never a spending authority."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from .e1c_cases import E1cSuite, Money, SHA, Strict


def planned_positions(suite: E1cSuite, *, seed: int):
    """No prior experimental answers; finance alone is inherited."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("explicit_uint32_seed_required")
    positions = []
    for case in suite.items:
        for session in (1, 2):
            positions.extend((case.id, session, "shared", 0, "online", turn) for turn in range(12))
            positions.extend((case.id, session, "shared", 0, role, None) for role in ("memory", "reflection"))
        for repeat in (1, 2, 3):
            arms = ("A0", "A1") if (repeat + seed) % 2 else ("A1", "A0")
            positions.extend((case.id, 3, arm, repeat, "online", turn) for arm in arms for turn in range(10))
    return tuple(positions)


class UnknownHold(Strict):
    campaign: StrictStr = Field(min_length=1)
    request_id: StrictStr = Field(min_length=1)
    request_sha256: SHA
    amount: Money


class FinancialInheritance(Strict):
    schema_version: Literal["e1c_finance_v1"]
    prior_e1b_closed: StrictBool
    prior_assessment_sha256: SHA
    prior_ledger_sha256: SHA
    known_cumulative_usd: Money
    known_e1b_usd: Money
    unknown_holds: tuple[UnknownHold, ...]

    @model_validator(mode="after")
    def consistent(self):
        if (Decimal(self.known_e1b_usd) > Decimal(self.known_cumulative_usd)
                or len({(row.campaign, row.request_id) for row in self.unknown_holds}) != len(self.unknown_holds)
                or any(Decimal(row.amount) <= 0 for row in self.unknown_holds)):
            raise ValueError("invalid_inherited_financial_history")
        return self


def financial_preparation(snapshot: FinancialInheritance):
    """Pure arithmetic, not approval or validation of the referenced ledger."""
    if snapshot.prior_e1b_closed is not True:
        raise ValueError("final_e1b_accounting_required_before_registration")
    holds = sum((Decimal(row.amount) for row in snapshot.unknown_holds), Decimal(0))
    return {"known_cumulative_usd": snapshot.known_cumulative_usd,
            "inherited_unknown_count": len(snapshot.unknown_holds),
            "inherited_unknown_holds": [row.model_dump(mode="json") for row in snapshot.unknown_holds],
            "inherited_unknown_hold_usd": str(holds),
            "committed_cumulative_usd": str(Decimal(snapshot.known_cumulative_usd) + holds),
            "e1c_paid_calls": 0, "retained_experimental_answers": 0,
            "basic_positions": 1056, "reply_denominator": 216, "formation_denominator": 12,
            "conditional_physical_max_if_separately_approved": 3192,
            "execution_authorized": False, "next": "verify_ledger_and_obtain_separate_e1c_limits"}


class RoleCostEnvelope(Strict):
    """Supplied maximum prices and token bounds, never fetched or inferred."""

    model: StrictStr = Field(min_length=1)
    price_metadata_sha256: SHA
    max_input_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)
    input_usd_per_million: Money
    output_usd_per_million: Money

    def maximum_usd(self):
        return (Decimal(self.input_usd_per_million) * self.max_input_tokens
                + Decimal(self.output_usd_per_million) * self.max_output_tokens) / Decimal(1000000)


def estimate_e1c_budget(inherited: FinancialInheritance, envelopes: dict[str, RoleCostEnvelope], *,
                        reflection_format_correction: bool = False, infrastructure_retries: int = 0):
    """Conservative scenario arithmetic; no supplied limit becomes approval.

    Reflection bounds must cover either product step. Infrastructure retries
    share a single per-logical-position pool across initial and correction.
    Public metadata and the historical ledger still require independent review.
    """
    if (set(envelopes) != {"online", "memory", "reflection"}
            or any(not isinstance(value, RoleCostEnvelope) for value in envelopes.values())
            or type(reflection_format_correction) is not bool
            or type(infrastructure_retries) is not int or not 0 <= infrastructure_retries <= 2):
        raise ValueError("explicit_three_roles_and_bounded_scenario_required")
    prior = financial_preparation(inherited)
    counts = {"online": 1008, "memory": 24, "reflection": 24}
    costs = {role: envelope.maximum_usd() for role, envelope in envelopes.items()}
    basic = sum((counts[role] * costs[role] for role in counts), Decimal(0))
    correction_count = 24 if reflection_format_correction else 0
    corrections = correction_count * costs["reflection"]
    infrastructure = infrastructure_retries * basic
    added = basic + corrections + infrastructure
    return {**prior, "schema": "e1c_budget_scenario_v1", "execution_authorized": False,
            "role_envelopes": {role: item.model_dump(mode="json") for role, item in envelopes.items()},
            "base_counts": counts, "base_maximum_usd": str(basic),
            "format_correction_count": correction_count, "format_correction_maximum_usd": str(corrections),
            "shared_infrastructure_retries_per_logical": infrastructure_retries,
            "infrastructure_maximum_usd": str(infrastructure),
            "physical_maximum": 1056 + correction_count + 1056 * infrastructure_retries,
            "additional_maximum_usd": str(added),
            "committed_cumulative_plus_scenario_usd": str(Decimal(prior["committed_cumulative_usd"]) + added),
            "judge_included": False, "judge_configuration_requires_registration": True,
            "cost_kind": "input_bounds_scenario_not_price_verification_or_spending_approval"}


def execute(*_args, **_kwargs):
    raise PermissionError("e1c_paid_entry_requires_separate_frozen_authorization_and_guard")
