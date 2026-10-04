"""Author-exposed E1 development fixtures, with separate stimulus and review views.

These fictional preferences and personality numbers are supplied by the author,
not learned by Aria. This loader checks structure and quote provenance, not the
psychological validity of a fixture. It has no model, network, or runtime writes.
Only ``model_input`` and, in the designated conditions, ``guidance_input`` may
enter a generation request. Historical sessions and review contracts are kept
outside both projections. Review criteria describe behavior, not a gold reply.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator
import yaml


DEFAULT_REFOCUS_CASES = Path(__file__).parent / "cases" / "refocus_e1_v1.yaml"
REFOCUS_CASE_VERSION = "refocus_e1_v1"
VIOLATION_KEYS = (
    "decides_for_person", "violates_agreement", "leaks_private", "lectures_when_quiet",
)
_ARM_NAME = re.compile(r"(?<![A-Za-z0-9])B(?:0|1|2|3a|3b|3)(?:['′])?(?![A-Za-z0-9])")
Text = Annotated[StrictStr, Field(min_length=1, max_length=2000)]
Identifier = Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Turn(_Record):
    role: Literal["user", "assistant"]
    content: Text


class PriorSession(_Record):
    session: Literal[1, 2]
    turns: Annotated[tuple[Turn, ...], Field(min_length=1, max_length=8)]


class Quote(_Record):
    session: Literal[1, 2]
    turn: Annotated[StrictInt, Field(ge=0, le=7)]
    text: Text


class Guidance(_Record):
    id: Identifier
    do: Text
    scope: Text
    not_implies: Annotated[tuple[Text, ...], Field(min_length=1, max_length=5)]
    quotes: Annotated[tuple[Quote, ...], Field(min_length=1, max_length=5)]
    confirmation: Quote


class DISCFixture(_Record):
    dominance: Unit
    influence: Unit
    steadiness: Unit
    conscientiousness: Unit


class BigFiveFixture(_Record):
    openness: Unit
    conscientiousness: Unit
    extraversion: Unit
    agreeableness: Unit
    neuroticism: Unit


class PersonaFixture(_Record):
    source: Literal["author_supplied_synthetic_not_learned"]
    disc: DISCFixture
    big_five: BigFiveFixture
    attachment_style: Literal["secure", "anxious", "avoidant", "disorganized"]


class CareSnapshot(_Record):
    """Common factual stimulus, shared by every condition; no inferred intent."""

    source: Literal["synthetic_local_care_records"]
    reminder_status: Literal["unconfirmed"]
    consent_status: Literal["active"]
    authorized_trigger: Literal["reminder_unconfirmed_after_60_minutes"]
    shared_fields: tuple[Literal["reminder_confirmation_status"]]
    notification_status: Literal["local_inbox_accepted"]
    notification_text: Text
    real_delivery: Literal[False]
    family_acknowledged: Literal[False]


class Stimulus(_Record):
    facts: Annotated[tuple[Text, ...], Field(min_length=1, max_length=6)]
    short_context: Annotated[tuple[Turn, ...], Field(max_length=4)]
    shared_boundaries: Annotated[tuple[Text, ...], Field(max_length=5)]
    session: Literal[3]
    user_input: Text
    care_context: CareSnapshot | None = None


class BinaryViolation(_Record):
    applicable: StrictBool
    violation_if: Text
    not_violation_if: Text


class ReviewContract(_Record):
    decides_for_person: BinaryViolation
    violates_agreement: BinaryViolation
    leaks_private: BinaryViolation
    lectures_when_quiet: BinaryViolation


class RefocusCase(_Record):
    id: Identifier
    category: Literal["arrangement", "takeover", "privacy", "quiet", "correction",
                      "indirect_expression", "authorized_notification", "paired", "same_control"]
    prior_sessions: Annotated[tuple[PriorSession, ...], Field(min_length=1, max_length=2)]
    stimulus: Stimulus
    guidance: Annotated[tuple[Guidance, ...], Field(min_length=1, max_length=4)]
    persona_fixture: PersonaFixture
    review: ReviewContract

    @model_validator(mode="after")
    def _check_sources_and_projection(self) -> RefocusCase:
        sessions = {item.session: item for item in self.prior_sessions}
        if len(sessions) != len(self.prior_sessions):
            raise ValueError("duplicate prior session")
        if len({item.id for item in self.guidance}) != len(self.guidance):
            raise ValueError("duplicate guidance id")
        for item in self.guidance:
            for quote in (*item.quotes, item.confirmation):
                session = sessions.get(quote.session)
                if session is None or quote.turn >= len(session.turns):
                    raise ValueError("quote source does not exist")
                turn = session.turns[quote.turn]
                if turn.role != "user" or quote.text not in turn.content:
                    raise ValueError("quote must be verbatim elder speech")
            confirmation = item.confirmation
            turns = sessions[confirmation.session].turns
            if confirmation.turn == 0 or turns[confirmation.turn - 1].role != "assistant":
                raise ValueError("confirmation must follow an explicit restatement")
            if any((q.session, q.turn) >= (confirmation.session, confirmation.turn)
                   for q in item.quotes):
                raise ValueError("confirmation must follow its evidence")
            if item.do in self.stimulus.user_input or any(
                q.text in self.stimulus.user_input for q in item.quotes
            ):
                raise ValueError("third-session input must not repeat a prior preference")
        # An exact-field projection prevents accidental metadata leakage. Text
        # checks are a backstop, not a claim to detect all semantic answer leaks.
        for value in _strings([self.model_input(), self.guidance_input()]):
            if _ARM_NAME.search(value) or any(
                marker in value for marker in ("预期答案", "标准答案", "正确答案", "评分标准")
            ):
                raise ValueError("condition labels or review answers in model-visible stimulus")
        return self

    def model_input(self) -> dict[str, Any]:
        """Fresh common stimulus only: never all prior conversations or review."""
        return self.stimulus.model_dump(mode="json", exclude_none=True)

    def guidance_input(self) -> list[dict[str, Any]]:
        """Fresh author-written confirmed instructions; no score/personality data."""
        return [item.model_dump(mode="json") for item in self.guidance]


class PairReview(_Record):
    relation: Literal["different", "same"]
    comparison: Text
    not_required: Text


class RefocusPair(_Record):
    id: Identifier
    cases: tuple[RefocusCase, RefocusCase]
    review: PairReview

    @model_validator(mode="after")
    def _same_stimulus(self) -> RefocusPair:
        first, second = self.cases
        if first.model_input() != second.model_input():
            raise ValueError("paired cases require byte-equivalent common stimulus")
        if first.guidance == second.guidance or first.persona_fixture == second.persona_fixture:
            raise ValueError("paired people require different guidance and persona fixtures")
        expected = "same_control" if self.review.relation == "same" else "paired"
        if any(case.category != expected for case in self.cases):
            raise ValueError("pair category must match review relation")
        return self


class RefocusSuite(_Record):
    version: Literal["refocus_e1_v1"]
    provenance: Literal["author_exposed_fictional_development_cases_not_effect_evidence"]
    traps: Annotated[tuple[RefocusCase, ...], Field(min_length=10, max_length=10)]
    pairs: tuple[RefocusPair, RefocusPair, RefocusPair]
    same_response_control: RefocusPair

    @model_validator(mode="after")
    def _complete_suite(self) -> RefocusSuite:
        all_cases = (*self.traps, *(case for pair in self.pairs for case in pair.cases),
                     *self.same_response_control.cases)
        if len({case.id for case in all_cases}) != 18:
            raise ValueError("all 18 case ids must be unique")
        pair_ids = [pair.id for pair in (*self.pairs, self.same_response_control)]
        if len(set(pair_ids)) != 4:
            raise ValueError("all pair ids must be unique")
        expected = {"arrangement": 2, "takeover": 2, "privacy": 1, "quiet": 1,
                    "correction": 2, "indirect_expression": 1, "authorized_notification": 1}
        if {key: sum(case.category == key for case in self.traps) for key in expected} != expected:
            raise ValueError("the ten traps must retain their declared category coverage")
        if any(pair.review.relation != "different" for pair in self.pairs):
            raise ValueError("three main pairs must test contextually appropriate differences")
        if self.same_response_control.review.relation != "same":
            raise ValueError("supplemental control must allow the same response")
        return self

    def main_cases(self) -> tuple[RefocusCase, ...]:
        """16 stimuli in the fixed 204-call design, excluding the extra control."""
        return (*self.traps, *(case for pair in self.pairs for case in pair.cases))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    values: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in values:
            raise ValueError("E1 requires unique string mapping keys")
        values[key] = loader.construct_object(value_node, deep=deep)
    return values


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_refocus_cases(path: Path | str | None = None) -> RefocusSuite:
    """Load validated, fresh development fixtures; never repair or score them."""
    selected = DEFAULT_REFOCUS_CASES if path is None else Path(path)
    if selected.stat().st_size > 500_000:
        raise ValueError("E1 fixture file exceeds size limit")
    content = selected.read_text(encoding="utf-8")
    try:
        if any(isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
               for token in yaml.scan(content)):
            raise ValueError("E1 aliases and anchors are forbidden")
        payload = yaml.load(content, Loader=_UniqueLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("invalid E1 fixture YAML") from exc
    return RefocusSuite.model_validate(payload)
