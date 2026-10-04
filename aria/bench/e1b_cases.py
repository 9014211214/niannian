"""Strict E1b draft loading and disclosed-turn input projection.

Author annotations are evaluation material, never generation context. All cases
and emotional readings are fictional; this module makes no service calls.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator
import yaml


DEFAULT_CASES = Path(__file__).with_name("cases") / "e1b_v1_draft.yaml"
CATEGORIES = {"family_arranges", "signed_up_by_others", "sharing_boundary",
              "wants_quiet", "one_step_help", "let_me_try"}
RUBRIC_KEYS = {"leaks_private", "lectures_when_quiet", "evades_request",
               "fabricates_unknown", "decides_for_person", "violates_agreement"}
EMOTIONS = {"平": "calm", "喜": "happy", "低": "low", "焦": "anxious",
            "怒": "angry", "疲": "tired"}
ADJACENT = {"calm": "tired", "tired": "calm", "happy": "calm",
            "low": "tired", "anxious": "angry", "angry": "anxious"}
_ANNOTATION = re.compile(r"\b(?:A0|A1|expected_preference|rubric|pref_turn)\b|预期答案|评分标准")
Utterance = tuple[StrictStr, Literal["平", "喜", "低", "焦", "怒", "疲"]]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E1bRubric(_StrictModel):
    applicable: StrictBool
    violation_if: StrictStr | None = None
    not_violation_if: StrictStr | None = None

    @model_validator(mode="after")
    def applicable_has_rules(self):
        if self.applicable and not (self.violation_if and self.not_violation_if):
            raise ValueError("applicable rubric needs both rules")
        return self


class E1bPersona(_StrictModel):
    name: StrictStr
    age: StrictInt = Field(ge=60, le=120)
    facts: tuple[StrictStr, ...]


class E1bExpectedPreference(_StrictModel):
    do: StrictStr
    scope: StrictStr
    not_implies: tuple[StrictStr, ...]


class E1bTestSession(_StrictModel):
    warmup: tuple[Utterance, ...] = Field(min_length=7, max_length=7)
    trap: Utterance
    followups: tuple[Utterance, ...] = Field(min_length=2, max_length=2)


class E1bCase(_StrictModel):
    id: StrictStr = Field(pattern=r"^e1b_[a-f][1-3]$")
    category: StrictStr
    persona: E1bPersona
    s1: tuple[Utterance, ...] = Field(min_length=12, max_length=12)
    s2: tuple[Utterance, ...] = Field(min_length=12, max_length=12)
    pref_turn: StrictInt = Field(ge=0, le=11)
    s3: E1bTestSession
    expected_preference: E1bExpectedPreference
    rubric: dict[str, E1bRubric]

    @model_validator(mode="after")
    def validate_case(self):
        if self.category not in CATEGORIES or set(self.rubric) - RUBRIC_KEYS:
            raise ValueError("unknown category or rubric")
        for number in (1, 2, 3):
            for text, _ in self.session(number):
                if not text.strip() or len(text) > 500 or _ANNOTATION.search(text):
                    raise ValueError("invalid user utterance or evaluator annotation in dialogue")
        return self

    def session(self, number: int) -> tuple[Utterance, ...]:
        if number == 1:
            return self.s1
        if number == 2:
            return self.s2
        if number == 3:
            return (*self.s3.warmup, self.s3.trap, *self.s3.followups)
        raise ValueError("session must be 1, 2, or 3")

    def current_input(self, session: int, turn: int) -> str:
        """Only this elder utterance; biography and expected answer stay out."""
        if type(turn) is not int or turn < 0:
            raise ValueError("invalid turn")
        return self.session(session)[turn][0]


class _SuiteFile(_StrictModel):
    version: Literal["e1b_v1_draft"]
    provenance: Literal["author_written_fictional_candidates_not_effect_evidence"]
    emotion_codes: dict[str, str]
    defaults: dict[str, E1bRubric]
    items: tuple[E1bCase, ...] = Field(min_length=18, max_length=18)


@dataclass(frozen=True)
class E1bSuite:
    items: tuple[E1bCase, ...]
    defaults: dict[str, E1bRubric]
    source_sha256: str
    version: str = "e1b_v1_draft"

    def rubric(self, case: E1bCase) -> dict[str, E1bRubric]:
        return {**self.defaults, **case.rubric}


def load_e1b_cases(path: str | Path = DEFAULT_CASES) -> E1bSuite:
    raw = Path(path).read_bytes()
    parsed = _SuiteFile.model_validate(yaml.safe_load(raw))
    if parsed.emotion_codes != EMOTIONS or set(parsed.defaults) - RUBRIC_KEYS:
        raise ValueError("emotion mapping or defaults changed; register a new version")
    if len({item.id for item in parsed.items}) != 18:
        raise ValueError("case ids must be unique")
    if Counter(item.category for item in parsed.items) != Counter(dict.fromkeys(CATEGORIES, 3)):
        raise ValueError("E1b draft must contain six categories of three cases")
    for item in parsed.items:
        if set(parsed.defaults) | set(item.rubric) != RUBRIC_KEYS:
            raise ValueError("every case must resolve all six rubric dimensions")
    return E1bSuite(parsed.items, parsed.defaults, hashlib.sha256(raw).hexdigest())


def synthetic_emotion_plan(suite: E1bSuite, seed: int) -> dict[tuple[str, int, int], dict]:
    """Freeze one 70/15/15 plan, shared by both arms and every repeat.

    Whole-suite counts use nearest integers (428/92/92 for 612 turns). No
    truth/category/rubric field is present in the payload supplied to EM-1.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an explicit uint32")
    coordinates = [(case.id, session, turn, EMOTIONS[entry[1]])
                   for case in suite.items for session in (1, 2, 3)
                   for turn, entry in enumerate(case.session(session))]
    rng = random.Random(seed)
    correct, wrong = round(len(coordinates) * .70), round(len(coordinates) * .15)
    modes = ["truth"] * correct + ["adjacent_error"] * wrong
    modes += ["unknown"] * (len(coordinates) - len(modes))
    rng.shuffle(modes)
    result = {}
    for (case_id, session, turn, truth), mode in zip(coordinates, modes):
        label = truth if mode == "truth" else ADJACENT[truth] if mode == "adjacent_error" else "unknown"
        confidence = (round(rng.uniform(.6, .9) if mode == "truth" else rng.uniform(.4, .7), 4)
                      if mode != "unknown" else 0.)
        result[(case_id, session, turn)] = {
            "label": "sad" if label == "low" else label,
            "confidence": confidence, "modality": "text",
            "source": f"simulated_e1b:{seed}/{case_id}/{session}/{turn}",
        }
    return result


def select_prescreen(suite: E1bSuite, violations: dict[str, tuple[bool, bool, bool]],
                     *, target: int = 12) -> dict:
    """Select in original order; absent/invalid labels never become zero scores."""
    if target != 12:
        raise ValueError("a changed denominator requires a separately registered rule")
    expected = {case.id for case in suite.items}
    if set(violations) != expected or any(
        len(values) != 3 or any(type(value) is not bool for value in values)
        for values in violations.values()
    ):
        raise ValueError("prescreen requires all 18 x 3 explicit judgments")
    selected = [case.id for case in suite.items if any(violations[case.id])][:target]
    return {"case_ids": selected, "planned_denominator": target,
            "actual_denominator": len(selected), "prescreen_only": True,
            "main_gate_registered": len(selected) == target,
            "status": "ready_for_registration" if len(selected) == target else "denominator_rule_required"}


def select_prescreen_balanced(suite: E1bSuite, violations: dict[str, tuple[bool, bool, bool]]) -> dict:
    """Approved E1b-R selection; identities fixed before any A1 generation."""
    select_prescreen(suite, violations)  # Reuse the complete 18 x 3 label checks.
    if not all(any(violations[case.id]) for case in suite.items):
        raise ValueError("e1br_requires_original_all18_eligible_evidence")
    selected = [f"e1b_{group}{number}" for group in "abcdef" for number in (1, 2)]
    selected_cases = [case for case in suite.items if case.id in selected]
    if [case.id for case in selected_cases] != selected or Counter(case.category for case in selected_cases) != Counter(dict.fromkeys(CATEGORIES, 2)):
        raise ValueError("e1br_fixed_balanced_selection_changed")
    return {"schema": "e1br_balanced_selection_v1", "case_ids": selected,
            "planned_denominator": 12, "actual_denominator": 12,
            "per_category": 2, "all18_originally_eligible": True,
            "selection_uses_a1_outcomes": False, "selection_uses_violation_magnitude": False,
            "status": "ready_for_registration", "main_gate_registered": False}
