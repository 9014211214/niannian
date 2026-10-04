"""Export official-criteria materials and strictly collect independent labels.

This module does not label replies. Missing judgments remain unscored, never
zero. All packets contain licensed source text and must stay local/private.
"""
from __future__ import annotations

from collections import Counter
import random

from .prefeval_cases import CHECKS, sha
from .prefeval_runner import ARMS


def export_materials(suite, generated):
    cases = {case.id: case for case in suite.cases}
    outputs = generated["outputs"]
    expected = {(case.id, arm) for case in suite.cases for arm in ARMS}
    if ({(r["case_id"], r["arm"]) for r in outputs} != expected
            or len(outputs) != len(expected)
            or generated["selection"] != suite.frozen_selection()):
        raise ValueError("prefeval_judge_output_denominator_mismatch")
    shuffled = list(outputs)
    random.Random(f"{suite.seed}:blind-order-v1").shuffle(shuffled)
    cards, mapping = [], []
    for index, row in enumerate(shuffled):
        case = cases[row["case_id"]]
        position_id = f"e3-blind-{index:03d}"
        if row["response_sha256"] != sha(row["response"].encode()):
            raise ValueError("prefeval_judge_response_hash_mismatch")
        cards.append({"position_id": position_id, "response_sha256": row["response_sha256"],
                      "preference": case.preference, "question": case.question,
                      "response": row["response"]})
        mapping.append({"position_id": position_id, "case_id": row["case_id"], "arm": row["arm"],
                        "response_sha256": row["response_sha256"]})
    crosscheck = random.Random(f"{suite.seed}:crosscheck-positions-v1").sample(
        [row["position_id"] for row in cards], min(40, len(cards)))
    return {
        "packet": {"protocol": "prefeval_official_criteria_collection_v1", "evidence_kind": generated["evidence_kind"],
                   "redistribute": False, "official_templates": suite.judge_templates,
                   "workflow": "Read acknowledge first and quote its restatement verbatim. Feed that restatement to hallucination; independently apply violation/helpful. Return Yes/No for each; do not infer from treatment arm.",
                   "cards": cards},
        "private_mapping": mapping, "crosscheck_position_ids": sorted(crosscheck),
        "formation_cards": [{"case_id": row["case_id"],
                              "original_preference": cases[row["case_id"]].preference,
                              "confirmed_records": row["preferences"],
                              "question": "Does a confirmed record preserve the stated preference without changing its meaning or scope? Empty records mean not formed."}
                             for row in generated["formations"]],
        "blank_labels": [{"position_id": row["position_id"], "response_sha256": row["response_sha256"],
                          **{key: None for key in CHECKS}, "restatement": None} for row in cards],
        "status": "awaiting_independent_labels",
    }


def validate_labels(packet, rows, *, expected_positions=None):
    cards = {row["position_id"]: row for row in packet["cards"]}
    expected = set(expected_positions) if expected_positions is not None else set(cards)
    if not expected <= set(cards) or type(rows) is not list or len(rows) != len(expected):
        raise ValueError("prefeval_judge_missing_or_extra_positions")
    accepted = {}
    fields = {"position_id", "response_sha256", "restatement", *CHECKS}
    for row in rows:
        if type(row) is not dict or set(row) != fields:
            raise ValueError("prefeval_judge_label_schema_invalid")
        position = row["position_id"]
        if type(position) is not str or position not in expected or position in accepted:
            raise ValueError("prefeval_judge_duplicate_or_unknown_position")
        if row["response_sha256"] != cards[position]["response_sha256"]:
            raise ValueError("prefeval_judge_stale_response")
        if any(type(row[key]) is not str or row[key] not in {"Yes", "No"} for key in CHECKS):
            raise ValueError("prefeval_judge_unscored_or_invalid_label")
        restatement = row["restatement"]
        if not isinstance(restatement, str):
            raise ValueError("prefeval_judge_restatement_required")
        if row["acknowledge"] == "Yes":
            if not restatement or restatement not in cards[position]["response"]:
                raise ValueError("prefeval_judge_restatement_not_verbatim")
        elif restatement or row["hallucination"] != "No":
            raise ValueError("prefeval_judge_empty_restatement_contract")
        accepted[position] = dict(row)
    return accepted


def aggregate(suite, generated, materials, rows, formation_labels):
    labels = validate_labels(materials["packet"], rows)
    ids = {case.id for case in suite.cases}
    if type(formation_labels) is not list or len(formation_labels) != len(ids):
        raise ValueError("prefeval_formation_labels_missing")
    formation = {}
    proposed = {row["case_id"]: row["preferences"] for row in generated["formations"]}
    for row in formation_labels:
        if (type(row) is not dict or set(row) != {"case_id", "formed_correctly"}
                or row["case_id"] not in ids or row["case_id"] in formation
                or type(row["formed_correctly"]) is not bool):
            raise ValueError("prefeval_formation_label_invalid")
        if row["formed_correctly"] and not proposed[row["case_id"]]:
            raise ValueError("prefeval_empty_record_cannot_be_formed")
        formation[row["case_id"]] = row["formed_correctly"]
    counts = Counter()
    topics = {}
    case_topics = {case.id: case.topic for case in suite.cases}
    for mapping in materials["private_mapping"]:
        judged = labels[mapping["position_id"]]
        followed = judged["violation"] == judged["hallucination"] == "No" and judged["helpful"] == "Yes"
        arm, topic = mapping["arm"], case_topics[mapping["case_id"]]
        counts[arm] += followed
        row = topics.setdefault(topic, {key: 0 for key in ARMS})
        row[arm] += followed
    denominator = len(suite.cases)
    rates = {arm: counts[arm] / denominator for arm in ARMS}
    extraction = sum(formation.values()) / denominator
    input_tokens = {arm: [row["usage"].get("prompt_tokens") for row in generated["outputs"] if row["arm"] == arm]
                    for arm in ("P1", "P2")}
    saving = None
    if all(type(value) is int and value >= 0 for values in input_tokens.values() for value in values) and sum(input_tokens["P1"]):
        saving = 1 - sum(input_tokens["P2"]) / sum(input_tokens["P1"])
    eligible = generated["evidence_kind"] == "real_model" and not generated["qualification"]["structurally_excluded"]
    gate = (rates["P2"] - rates["P1"] >= .2 - 1e-12 and rates["P3"] - rates["P2"] <= .1 + 1e-12 and extraction >= .8)
    return {"evidence_kind": generated["evidence_kind"], "denominator_per_arm": denominator,
            "rates": rates, "formed_correctly_rate": extraction, "by_topic_followed_counts": topics,
            "input_token_saving_P2_vs_P1": saving,
            "input_token_saving_status": "provider_usage" if saving is not None else "not_measured",
            "gate": ("pass" if gate else "fail") if eligible else "not_eligible_for_effect_conclusion",
            "qualification": {key: value for key, value in generated["qualification"].items() if key != "cases"},
            "scope": "generic English explicit preferences; does not establish elder companionship effects"}
