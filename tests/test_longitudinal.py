"""个人时间基线的有限聚合与来源边界。"""
from __future__ import annotations

import math

from aria.engine.model.persona import BaselineStat, PersonaProfile
from aria.engine.perception.longitudinal import (
    enrich_with_personal_baseline,
)


def test_personal_baseline_adds_delta_after_history_and_only_keeps_aggregates():
    persona = PersonaProfile(user_id="u")
    persona.longitudinal_baseline.signals["prosodic:speech_pace"] = BaselineStat(
        count=5, mean=4.0, m2=0.4
    )
    original = {
        "prosodic": {
            "speech_pace": {
                "value": 2.0,
                "confidence": 0.8,
                "quality": 0.9,
            }
        }
    }
    enriched = enrich_with_personal_baseline(persona, original)
    row = enriched["prosodic"]["speech_pace"]
    assert row["baseline_delta"] < -0.4
    assert "个人历史 5 次" in row["detail"]
    assert "baseline_delta" not in original["prosodic"]["speech_pace"]
    stat = persona.longitudinal_baseline.signals["prosodic:speech_pace"]
    assert stat.count == 6 and stat.mean < 4.0
    assert not hasattr(stat, "raw_samples")
