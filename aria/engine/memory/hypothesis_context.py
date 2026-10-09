"""One bounded set of active hypotheses for online influence and later review."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.persona import PersonaProfile

ACTIVE_HYPOTHESIS_LIMIT = 30


def online_influencing_hypotheses(persona: PersonaProfile | None) -> list:
    if persona is None:
        return []
    return [item for item in persona.person_hypotheses
            if item.status in {"working", "stable"}][-ACTIVE_HYPOTHESIS_LIMIT:]


def reflection_hypotheses(persona: PersonaProfile) -> list:
    active = online_influencing_hypotheses(persona)
    identities = {item.id for item in active}
    remaining = [item for item in persona.person_hypotheses if item.id not in identities]
    slots = ACTIVE_HYPOTHESIS_LIMIT - len(active)
    return [*active, *(remaining[-slots:] if slots else [])]
