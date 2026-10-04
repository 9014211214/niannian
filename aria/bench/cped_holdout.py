"""Local, model-free CPED holdout construction; no automatic data acquisition.

Private cases stay in memory and are not public report objects. This implements
mechanical source checks, not complete entity recognition or rights clearance.
An independently retained identity/plot review is still required before sending
any prepared text to a model. The protocol's cohort and shortcut gates stay fixed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import json
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from .cped_preflight import COHORT_SIZE, HoldoutCaseAudit, SURFACE_BASELINES
from .cped_source import CPEDRow, CPED_TRAITS


BUILDER_VERSION = "1.0"
DEFAULT_SEED = 20260909
_MECHANICAL_AUDITS = frozenset({
    "consistent_big_five", "target_is_next_speaker_turn",
    "three_distractors_other_roles_and_dialogues",
    "matched_emotion_act_length_punctuation_topic",
    "no_duplicate_text_or_fragment_overlap",
})
_FAILURES = frozenset({
    "invalid_source_row", "conflicting_source_identity", "ambiguous_turn_order",
    "inconsistent_role_labels", "insufficient_independent_dialogues",
    "insufficient_unique_formation", "no_eligible_holdout_target",
    "no_matched_distractor_triplet", "cohort_below_48",
    "surface_shortcut_above_35_percent",
})


@dataclass(frozen=True, repr=False)
class PreparedHoldoutCase:
    """Restricted working material: do not serialize into a public artifact."""

    role_key: tuple[str, str]
    formation_dialogues: tuple[str, ...]
    holdout_dialogue: str
    formation_texts: tuple[str, ...]
    context: tuple[tuple[str, str], ...]  # Local labels, never original speakers.
    candidates: tuple[str, ...]
    correct_position: int
    gold_traits: tuple[str, ...]
    elderly: bool
    # Provenance remains private for source audit. No per-case hash is public.
    formation_source_keys: tuple[tuple[str, str, str], ...]
    context_source_keys: tuple[tuple[str, str, str], ...]
    target_source_key: tuple[str, str, str]
    distractor_source_keys: tuple[tuple[str, str, str], ...]

    def metadata(self) -> HoldoutCaseAudit:
        return HoldoutCaseAudit(
            role_key=self.role_key,
            formation_dialogues=self.formation_dialogues,
            holdout_dialogue=self.holdout_dialogue,
            formation_utterances=len(self.formation_texts),
            context_turns=len(self.context),
            correct_position=self.correct_position,
            elderly=self.elderly,
            # Do not assert complete anonymity merely because known names were
            # masked. The existing preflight therefore still fails closed.
            completed_audits=_MECHANICAL_AUDITS,
        )


@dataclass(frozen=True)
class HoldoutPreparation:
    cases: tuple[PreparedHoldoutCase, ...] = field(repr=False)
    report: dict
    private_cohort_fingerprint: str = field(repr=False)


def _hash(seed: int, *parts: object) -> str:
    return hashlib.sha256(json.dumps(
        [seed, *parts], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _key(row: CPEDRow) -> tuple[str, str, str]:
    return row.tv_id, row.dialogue_id, row.utterance_id


def _role(row: CPEDRow) -> tuple[str, str]:
    return row.tv_id, row.speaker


def _normal(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def _content(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


def _duplicate_key(text: str) -> str:
    # Punctuation/spacing-only differences cannot create independent evidence.
    return _content(_normal(text)).casefold()


def _grams(text: str) -> set[str]:
    text = _content(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _similarity(left: str, right: str) -> float:
    a, b = _grams(left), _grams(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def _punctuation(text: str) -> tuple[int, int, int, int]:
    # NFKC-normalized Chinese/Latin punctuation and sentence-count bucket.
    return (
        int("?" in text), int("!" in text), min(2, text.count(",") + text.count("、")),
        min(3, len(re.findall(r"[。.!?;；]+", text))),
    )


def _match_bucket(row: CPEDRow, text: str) -> tuple:
    return row.emotion, row.dialogue_act, len(_content(text)) // 6, _punctuation(text)


def _valid_row(row: CPEDRow) -> bool:
    return (
        isinstance(row, CPEDRow)
        and all(isinstance(value, str) and value.strip() for value in (
            row.tv_id, row.dialogue_id, row.utterance_id, row.speaker,
            row.emotion, row.dialogue_act, row.utterance, row.age,
        ))
        and isinstance(row.traits, dict) and set(row.traits) == set(CPED_TRAITS)
        and all(isinstance(value, str) and value in {"high", "low", "unknown"}
                for value in row.traits.values())
    )


def prepare_cped_holdout(
    rows: Iterable[CPEDRow], *, terms_review_acknowledged: bool = False,
    aggregate_only_accepted: bool = False,
    extra_identity_terms: Sequence[str] = (), seed: int = DEFAULT_SEED,
) -> HoldoutPreparation:
    """Deterministic source-derived cohort, with no file or model access.

    The acknowledgement is checked before touching the iterable. Known speaker
    names and supplied entities are masked uniformly in *all* model material;
    completeness of that inventory is not assumed. No automatic retry with a
    different seed or weaker gates is performed when fewer than 48 roles pass.
    """
    if terms_review_acknowledged is not True or aggregate_only_accepted is not True:
        return _result((), "stopped_terms", Counter(), 0, 0, 0, {}, 0)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("CPED holdout seed must be an unsigned 32-bit integer")
    if not isinstance(extra_identity_terms, (list, tuple)) or any(
        not isinstance(term, str) or not term.strip() for term in extra_identity_terms
    ):
        raise ValueError("CPED identity inventory must contain nonempty strings")

    failures: Counter[str] = Counter()
    source: dict[tuple[str, str, str], CPEDRow] = {}
    duplicate_rows = 0
    total_rows = 0
    for row in rows:
        total_rows += 1
        if not _valid_row(row):
            failures["invalid_source_row"] += 1
            continue
        key = _key(row)
        previous = source.get(key)
        if previous is not None:
            # Split membership is acquisition metadata, not another utterance.
            if replace(previous, split="") != replace(row, split=""):
                failures["conflicting_source_identity"] += 1
            else:
                duplicate_rows += 1
        else:
            source[key] = row
    if failures:
        return _result((), "stopped_source_integrity", failures, total_rows,
                       len(source), duplicate_rows, {}, 0)

    terms = {_normal(row.speaker) for row in source.values()}
    terms.update(_normal(term) for term in extra_identity_terms)
    mask = re.compile("|".join(re.escape(term) for term in sorted(
        terms, key=lambda value: (-len(value), value),
    ))) if terms else None
    texts = {
        key: mask.sub("[名称]", _normal(row.utterance)) if mask else _normal(row.utterance)
        for key, row in source.items()
    }
    by_dialogue: dict[tuple[str, str], list[CPEDRow]] = defaultdict(list)
    by_role: dict[tuple[str, str], list[CPEDRow]] = defaultdict(list)
    buckets: dict[tuple, list[CPEDRow]] = defaultdict(list)
    for key, row in source.items():
        by_dialogue[(row.tv_id, row.dialogue_id)].append(row)
        by_role[_role(row)].append(row)
        if 4 <= len(_content(texts[key])) <= 120:
            buckets[_match_bucket(row, texts[key])].append(row)
    for dialogue in by_dialogue.values():
        ordinals = [re.search(r"(\d+)$", row.utterance_id) for row in dialogue]
        if any(match is None for match in ordinals) or len({
            int(match.group(1)) for match in ordinals if match
        }) != len(dialogue):
            failures["ambiguous_turn_order"] += 1
        else:
            dialogue.sort(key=lambda row: int(re.search(r"(\d+)$", row.utterance_id).group(1)))
    if failures:
        return _result((), "stopped_source_integrity", failures, total_rows,
                       len(source), duplicate_rows, {}, 0)

    eligible: list[PreparedHoldoutCase] = []
    for role in sorted(by_role, key=lambda value: _hash(seed, "role", value)):
        role_rows = by_role[role]
        if any(len({row.traits[trait] for row in role_rows}) != 1 for trait in CPED_TRAITS):
            failures["inconsistent_role_labels"] += 1
            continue
        dialogues = sorted({row.dialogue_id for row in role_rows},
                           key=lambda value: _hash(seed, "fragment", role, value))
        if len(dialogues) < 4:
            failures["insufficient_independent_dialogues"] += 1
            continue
        # Fixed partition, not a search for a holdout that raises later scores.
        formation_dialogues, heldout = tuple(dialogues[:3]), dialogues[3]
        formation: list[CPEDRow] = []
        seen: set[str] = set()
        for dialogue_id in formation_dialogues:
            for row in by_dialogue[(role[0], dialogue_id)]:
                text = texts[_key(row)]
                duplicate = _duplicate_key(text)
                if _role(row) == role and 4 <= len(_content(text)) <= 120 and duplicate not in seen:
                    formation.append(row)
                    seen.add(duplicate)
        if len(formation) < 12 or len({row.dialogue_id for row in formation}) < 3:
            failures["insufficient_unique_formation"] += 1
            continue
        # All eligible target utterances in the three fixed fragments are used;
        # no target-dependent history selection or truncation is allowed here.
        heldout_rows = by_dialogue[(role[0], heldout)]
        candidate_targets: list[tuple[CPEDRow, list[CPEDRow]]] = []
        for index, target in enumerate(heldout_rows):
            if _role(target) != role or index < 2 or _role(heldout_rows[index - 1]) == role:
                continue
            context = heldout_rows[max(0, index - 6):index]
            context_keys = [_duplicate_key(texts[_key(row)]) for row in context]
            target_text = texts[_key(target)]
            target_key = _duplicate_key(target_text)
            if (4 <= len(_content(target_text)) <= 120
                    and target_key not in seen and target_key not in context_keys
                    and len(set(context_keys)) == len(context_keys)
                    and not (set(context_keys) & seen)):
                candidate_targets.append((target, context))
        if not candidate_targets:
            failures["no_eligible_holdout_target"] += 1
            continue
        # Fix target before inspecting distractor availability. An unavailable
        # triplet rejects this role, not a switch to an easier target.
        target, context = min(candidate_targets, key=lambda pair: _hash(seed, "target", _key(pair[0])))
        target_text = texts[_key(target)]
        excluded = seen | {_duplicate_key(texts[_key(row)]) for row in context} | {_duplicate_key(target_text)}
        distractors: list[CPEDRow] = []
        distractor_roles: set[tuple[str, str]] = set()
        distractor_dialogues: set[tuple[str, str]] = set()
        for row in sorted(buckets[_match_bucket(target, target_text)],
                          key=lambda value: _hash(seed, "distractor", role, _key(value))):
            text = texts[_key(row)]
            # Other shows is stricter than other roles/dialogues and reduces
            # same-plot hints. Matching is a lexical proxy, not topic truth.
            if (row.tv_id == role[0] or _role(row) in distractor_roles
                    or (row.tv_id, row.dialogue_id) in distractor_dialogues
                    or _duplicate_key(text) in excluded
                    or not 0.05 <= _similarity(target_text, text) <= 0.65):
                continue
            distractors.append(row)
            distractor_roles.add(_role(row))
            distractor_dialogues.add((row.tv_id, row.dialogue_id))
            excluded.add(_duplicate_key(text))
            if len(distractors) == 3:
                break
        if len(distractors) != 3:
            failures["no_matched_distractor_triplet"] += 1
            continue
        labels: dict[str, str] = {role[1]: "目标人物"}
        labelled_context = []
        for row in context:
            if row.speaker not in labels:
                labels[row.speaker] = f"对话者{len(labels)}"
            labelled_context.append((labels[row.speaker], texts[_key(row)]))
        eligible.append(PreparedHoldoutCase(
            role_key=role, formation_dialogues=formation_dialogues,
            holdout_dialogue=heldout,
            formation_texts=tuple(texts[_key(row)] for row in formation),
            context=tuple(labelled_context),
            candidates=(target_text, *(texts[_key(row)] for row in distractors)),
            correct_position=0,
            gold_traits=tuple(target.traits[trait] for trait in CPED_TRAITS),
            elderly=all(row.age == "elderly" for row in role_rows),
            formation_source_keys=tuple(_key(row) for row in formation),
            context_source_keys=tuple(_key(row) for row in context),
            target_source_key=_key(target),
            distractor_source_keys=tuple(_key(row) for row in distractors),
        ))

    selected = []
    for index, case in enumerate(eligible[:COHORT_SIZE]):
        position = index % 4  # Exactly twelve per position in the fixed cohort.
        options = list(case.candidates[1:])
        options.insert(position, case.candidates[0])
        selected.append(replace(case, candidates=tuple(options), correct_position=position))
    if len(selected) < COHORT_SIZE:
        failures["cohort_below_48"] += 1
    surface = surface_baselines(selected)
    if any(count * 100 > 35 * len(selected) for count in surface.values()) and selected:
        failures["surface_shortcut_above_35_percent"] += 1
    status = "prepared_pending_identity_review"
    if len(selected) < COHORT_SIZE or "surface_shortcut_above_35_percent" in failures:
        status = "stopped_feasibility"
    return _result(tuple(selected), status, failures, total_rows, len(source),
                   duplicate_rows, surface, len(eligible), seed=seed)


def surface_baselines(cases: Sequence[PreparedHoldoutCase]) -> dict[str, int]:
    """Four prespecified, weak non-model probes; never an anti-leak guarantee.

    First three match the last preceding utterance. Ties use the first option,
    fixed before answers are known. Position baseline is always option A.
    """
    correct = dict.fromkeys(sorted(SURFACE_BASELINES), 0)
    for case in cases:
        reference = case.context[-1][1]
        candidates = case.candidates
        predictions = {
            "length": min(range(4), key=lambda i: abs(len(_content(candidates[i])) - len(_content(reference)))),
            "punctuation": min(range(4), key=lambda i: sum(abs(a - b) for a, b in zip(
                _punctuation(candidates[i]), _punctuation(reference),
            ))),
            "lexical": max(range(4), key=lambda i: _similarity(reference, candidates[i])),
            "position": 0,
        }
        for name, prediction in predictions.items():
            correct[name] += int(prediction == case.correct_position)
    return correct


def _result(cases, status, failures, total, unique, duplicates, surface, eligible, *, seed=DEFAULT_SEED):
    report = {
        "suite": "cped_holdout_source_preparation_v1",
        "builder_version": BUILDER_VERSION,
        "status": status,
        "evidence_status": "mechanical_preparation_not_identity_clearance_or_effect_evidence",
        "artifact_scope": "aggregate_only_no_case_level_derivatives",
        "source_row_count": total, "unique_source_row_count": unique,
        "duplicate_source_row_count": duplicates,
        "eligible_role_count": eligible, "selected_role_count": len(cases),
        "elderly_role_count": sum(case.elderly for case in cases),
        "position_counts": [sum(case.correct_position == i for case in cases) for i in range(4)],
        "surface_correct_counts": surface, "failure_counts": dict(sorted(failures.items())),
        "seed": seed, "executed_model_calls": 0,
        "identity_review_required": True, "cloud_submission_allowed": False,
    }
    assert_aggregate_only_preparation(report)
    # Private digest binds review to the exact texts, positions and provenance.
    # It is not returned inside the aggregate-only report.
    private_digest = _hash(seed, [case.__dict__ for case in cases])
    return HoldoutPreparation(cases, report, private_digest)


def assert_aggregate_only_preparation(report: Mapping) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("Invalid aggregate-only CPED preparation report")
    constants = {
        "suite": "cped_holdout_source_preparation_v1", "builder_version": BUILDER_VERSION,
        "evidence_status": "mechanical_preparation_not_identity_clearance_or_effect_evidence",
        "artifact_scope": "aggregate_only_no_case_level_derivatives",
        "identity_review_required": True, "cloud_submission_allowed": False,
        "executed_model_calls": 0,
    }
    counts = {
        "source_row_count", "unique_source_row_count", "duplicate_source_row_count",
        "eligible_role_count", "selected_role_count", "elderly_role_count", "seed",
    }
    expected = set(constants) | counts | {"status", "position_counts", "surface_correct_counts", "failure_counts"}
    valid = (
        isinstance(report, Mapping) and set(report) == expected
        and all(type(report[key]) is type(value) and report[key] == value for key, value in constants.items())
        and all(type(report[key]) is int and report[key] >= 0 for key in counts)
        and isinstance(report["status"], str) and report["status"] in {
            "stopped_terms", "stopped_source_integrity", "stopped_feasibility",
            "prepared_pending_identity_review",
        }
        and isinstance(report["position_counts"], list) and len(report["position_counts"]) == 4
        and all(type(value) is int and value >= 0 for value in report["position_counts"])
    )
    for key, allowed in (("surface_correct_counts", SURFACE_BASELINES), ("failure_counts", _FAILURES)):
        value = report.get(key)
        valid = valid and isinstance(value, dict) and set(value) <= allowed and all(
            type(count) is int and count >= 0 for count in value.values()
        )
    if not valid:
        raise ValueError("Invalid aggregate-only CPED preparation report")
