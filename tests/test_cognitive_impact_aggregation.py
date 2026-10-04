"""Functional-impact days survive same-session frequency de-duplication."""
from copy import deepcopy
from datetime import timedelta

import pytest

from aria.engine.cognitive import evaluate_cognitive_trend
from tests.test_cognitive_change_v2 import T0, calibrated, enabled, record


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_same_session_observations_preserve_impact_in_either_order(reverse):
    persona = calibrated(enabled())
    first = record(persona, 15, "memory", True)
    second = record(persona, 15, "memory", False, index=1)
    second.session_id = first.session_id
    record(persona, 16, "memory", True)
    if reverse:
        persona.cognitive_observations.reverse()
    trend = evaluate_cognitive_trend(persona, now=T0 + timedelta(days=16))
    assert trend.status == "review_suggested"
    assert trend.supporting_days == ["2026-08-16", "2026-08-17"]


def test_duplicate_frequency_cannot_create_increase_or_another_impact_day():
    persona = calibrated(enabled())
    # Eight unique session/domain observations remain below the unchanged 2/week
    # threshold. All repetitions must keep the original day and frequency slot.
    for day in (15, 16, 17, 18):
        for domain in ("memory", "language"):
            record(persona, day, domain, impact=(day == 15))
    duplicates = deepcopy(persona.cognitive_observations)
    persona.cognitive_observations.extend(duplicates * 4)
    trend = evaluate_cognitive_trend(persona, now=T0 + timedelta(days=18))
    assert trend.status == "watch"
    assert len(trend.supporting_days) == 4


@pytest.mark.parametrize("invalid", ["old_consent", "non_counting", "stale", "future", "baseline"])
def test_only_qualified_recent_observations_add_impact_days(invalid):
    persona = calibrated(enabled())
    record(persona, 15, "memory", True)
    second = record(persona, 16, "memory", True)
    if invalid == "old_consent":
        second.consent_at = (T0 - timedelta(days=1)).isoformat()
    elif invalid == "non_counting":
        second.counting = False
    elif invalid == "stale":
        second.at = (T0 - timedelta(days=40)).isoformat()
    elif invalid == "future":
        second.at = (T0 + timedelta(days=17)).isoformat()
    else:
        second.at = (T0 + timedelta(days=1)).isoformat()
    assert evaluate_cognitive_trend(persona, now=T0 + timedelta(days=16)).status == "watch"


def test_same_natural_day_across_sessions_is_one_impact_day():
    persona = calibrated(enabled())
    record(persona, 15, "memory", True)
    second = record(persona, 15, "language", True, index=1)
    second.at = (T0 + timedelta(days=15, hours=3)).isoformat()
    assert evaluate_cognitive_trend(persona, now=T0 + timedelta(days=16)).status == "watch"
