"""Dedicated E1c holdout structure; no product-output filtering or model calls.

An authored first submission is distinct from the explicit software fixture.
Author annotations and future turns never enter the current-input projection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from aria.bench.e1b_cases import (
    CATEGORIES, EMOTIONS, RUBRIC_KEYS, E1bExpectedPreference, E1bPersona,
    E1bRubric, E1bTestSession, Utterance,
)

VERSION = "e1c_format_v1"
SHA = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
Money = Annotated[StrictStr, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]{1,12})?$")]
CATEGORY_ORDER = (
    "family_arranges", "signed_up_by_others", "sharing_boundary",
    "wants_quiet", "one_step_help", "let_me_try",
)
CASE_CATEGORIES = {f"e1c_{letter}{n}": category
                   for letter, category in zip("abcdef", CATEGORY_ORDER) for n in (1, 2)}
ANNOTATION = re.compile(
    r"\b(?:A0|A1|e1[bc]_[a-f][1-3]|expected_preference|rubric|pref_turn|"
    r"preference_sources)\b|预期答案|评分标准"
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreferenceSource(Strict):
    """Complete utterance location; two sentences may live in one location."""

    session: StrictInt = Field(ge=1, le=2)
    turn: StrictInt = Field(ge=0, le=11)
    quote: StrictStr = Field(min_length=1, max_length=500)


class E1cCase(Strict):
    id: StrictStr = Field(pattern=r"^e1c_[a-f][12]$")
    category: StrictStr
    persona: E1bPersona
    s1: tuple[Utterance, ...] = Field(min_length=12, max_length=12)
    s2: tuple[Utterance, ...] = Field(min_length=12, max_length=12)
    s3: E1bTestSession
    preference_sources: tuple[PreferenceSource, ...] = Field(min_length=1, max_length=24)
    expected_preference: E1bExpectedPreference
    rubric: dict[str, E1bRubric]

    @model_validator(mode="after")
    def structural_only(self):
        if CASE_CATEGORIES.get(self.id) != self.category or set(self.rubric) - RUBRIC_KEYS:
            raise ValueError("category_id_or_rubric_mismatch")
        coordinates = [(source.session, source.turn) for source in self.preference_sources]
        if coordinates != sorted(set(coordinates)):
            raise ValueError("source_coordinates_must_be_unique_and_chronological")
        for source in self.preference_sources:
            if self.current_input(source.session, source.turn) != source.quote:
                raise ValueError("source_must_quote_entire_original_utterance")
        for session in (1, 2, 3):
            for text, _ in self.session(session):
                if not text.strip() or len(text) > 500 or ANNOTATION.search(text):
                    raise ValueError("invalid_utterance_or_author_annotation")
        return self

    def session(self, number):
        if type(number) is not int or number not in (1, 2, 3):
            raise ValueError("invalid_session")
        return self.s1 if number == 1 else self.s2 if number == 2 else (
            *self.s3.warmup, self.s3.trap, *self.s3.followups)

    def current_input(self, session, turn):
        if type(turn) is not int or not 0 <= turn < len(self.session(session)):
            raise ValueError("invalid_turn")
        return self.session(session)[turn][0]

    @property
    def source_layout(self):
        if len(self.preference_sources) == 1:
            return "single_utterance"
        if len({source.session for source in self.preference_sources}) == 1:
            return "multiple_turns_one_session"
        return "multiple_sessions"


class SuiteFile(Strict):
    version: Literal["e1c_format_v1"]
    provenance: Literal["independent_fictional_holdout", "software_fixture_only"]
    emotion_codes: dict[str, StrictStr]
    defaults: dict[str, E1bRubric]
    items: tuple[E1cCase, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def fixed_inventory(self):
        if (self.emotion_codes != EMOTIONS or set(self.defaults) - RUBRIC_KEYS
                or set(item.id for item in self.items) != set(CASE_CATEGORIES)
                or Counter(item.category for item in self.items) != Counter(dict.fromkeys(CATEGORIES, 2))):
            raise ValueError("e1c_twelve_six_by_two_required")
        for item in self.items:
            if set(self.defaults) | set(item.rubric) != RUBRIC_KEYS:
                raise ValueError("all_six_rubrics_required")
        return self


@dataclass(frozen=True)
class E1cSuite:
    items: tuple[E1cCase, ...]
    defaults: dict[str, E1bRubric]
    source_sha256: str
    provenance: str
    version: str = VERSION

    def rubric(self, case):
        return {**self.defaults, **case.rubric}


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    def reject(_):
        raise ValueError("nonfinite_json")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)


def parse_suite(raw: bytes, *, allow_software_fixture=False):
    parsed = SuiteFile.model_validate(strict_json(raw))
    if parsed.provenance == "software_fixture_only" and allow_software_fixture is not True:
        raise ValueError("software_fixture_not_formal_holdout")
    return E1cSuite(parsed.items, parsed.defaults, hashlib.sha256(raw).hexdigest(), parsed.provenance)


def load_e1c_cases(path, *, expected_sha256: str, allow_software_fixture=False):
    raw = Path(path).read_bytes()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("first_submission_sha_mismatch")
    return parse_suite(raw, allow_software_fixture=allow_software_fixture)


