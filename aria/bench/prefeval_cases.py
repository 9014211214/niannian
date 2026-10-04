"""Local-only PrefEval E3 source verification, fixed sampling and qualification.

The dataset and official judge prompts are not bundled or redistributed. Load
the explicitly supplied, ignored local snapshot of amazon-science/PrefEval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from aria.engine.memory.stated_preferences import _REQUEST, _safe_user_instruction

VERSION = "prefeval_e3_preparation_v1"
DEFAULT_SEED = 20261003
CHECKS = ("violation", "hallucination", "acknowledge", "helpful")


def sha(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PrefEvalCase:
    id: str
    topic: str
    index: int
    preference: str
    question: str
    explanation: str


@dataclass(frozen=True)
class PrefEvalSuite:
    source_revision: str
    source_manifest_sha256: str
    cases: tuple[PrefEvalCase, ...]
    pool: tuple[PrefEvalCase, ...]
    seed: int
    judge_templates: dict[str, str]

    def frozen_selection(self):
        return {"version": VERSION, "source_revision": self.source_revision,
                "source_manifest_sha256": self.source_manifest_sha256, "seed": self.seed,
                "sample_count": len(self.cases), "topic_counts": dict(Counter(c.topic for c in self.cases)),
                "case_ids": [c.id for c in self.cases],
                "selected_content_sha256": sha([asdict(c) for c in self.cases]),
                "source_rows_missing_explanation": [c.id for c in self.pool if not c.explanation],
                "judge_template_sha256": {key: sha(value.encode()) for key, value in self.judge_templates.items()},
                "redistribute_source": False, "public_output": "aggregates_only"}


def load_suite(root, *, seed: int = DEFAULT_SEED) -> PrefEvalSuite:
    root = Path(root).resolve()
    manifest_bytes = (root / "source-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("repository") != "https://github.com/amazon-science/PrefEval"
            or manifest.get("redistribute_source") is not False
            or manifest.get("use") != "local_noncommercial_evaluation_only"):
        raise ValueError("prefeval_source_manifest_invalid")
    indexed = {}
    for item in manifest["files"]:
        if not (item["path"] in {"LICENSE", "README.md", *(f"error_type/check_{name}.txt" for name in CHECKS)}
                or item["path"].startswith("benchmark_dataset/explicit_preference/") and item["path"].endswith(".json")):
            raise ValueError("prefeval_unregistered_source_file_forbidden")
        path = root / item["path"]
        if root not in path.resolve().parents or item["path"] in indexed:
            raise ValueError("prefeval_source_path_invalid")
        raw = path.read_bytes()
        if sha(raw) != item["sha256"]:
            raise ValueError("prefeval_source_hash_mismatch")
        indexed[item["path"]] = raw
    if b"Attribution-NonCommercial 4.0 International" not in indexed.get("LICENSE", b""):
        raise ValueError("prefeval_source_license_not_verified")
    files = sorted(name for name in indexed if name.startswith("benchmark_dataset/explicit_preference/")
                   and name.endswith(".json"))
    if len(files) != 20:
        raise ValueError("prefeval_expected_twenty_topics")
    grouped = {}
    for name in files:
        rows = json.loads(indexed[name])
        if type(rows) is not list or len(rows) < 3:
            raise ValueError("prefeval_topic_has_insufficient_rows")
        topic = Path(name).stem
        cases = []
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or any(type(row.get(k)) is not str or not row[k].strip()
                                                 for k in ("preference", "question"))
                    or ("explanation" in row and type(row["explanation"]) is not str)):
                raise ValueError("prefeval_invalid_explicit_case")
            cases.append(PrefEvalCase(f"{topic}-{index:02d}", topic, index,
                                     row["preference"], row["question"], row.get("explanation", "")))
        grouped[topic] = cases
    if sum(len(rows) for rows in grouped.values()) != 1000:
        raise ValueError("prefeval_expected_thousand_source_rows")
    topics = sorted(grouped)
    rng = random.Random(seed)
    three_topics = set(rng.sample(topics, 8))
    selected = []
    for topic in topics:
        selected.extend(rng.sample(grouped[topic], 3 if topic in three_topics else 2))
    templates = {name: indexed[f"error_type/check_{name}.txt"].decode("utf-8") for name in CHECKS}
    return PrefEvalSuite(manifest["revision"], sha(manifest_bytes), tuple(selected),
                        tuple(c for topic in topics for c in grouped[topic]), seed, templates)


def qualification_audit(suite: PrefEvalSuite) -> dict:
    """Necessary product rules, not model outputs or a semantic quality score.

    Every extracted quote must be a verbatim substring of the source and contain
    the product request pattern. If the full source lacks that pattern, no
    choice of a shorter quote or model-written instruction can pass that gate.
    """
    rows = []
    for case in suite.cases:
        issues = []
        if not _REQUEST.search(case.preference):
            issues.append("no_product_communication_request_pattern_in_verbatim_source")
        if not _safe_user_instruction(case.preference):
            issues.append("product_full_user_instruction_safety_gate")
        rows.append({"case_id": case.id, "topic": case.topic,
                     "structurally_excluded": bool(issues), "reasons": issues,
                     "meaning": "protocol_applicability_only_not_model_score"})
    excluded = sum(row["structurally_excluded"] for row in rows)
    return {"status": "needs_protocol_review" if excluded else "necessary_rules_passed_not_semantic_validation",
            "sample_count": len(rows), "structurally_excluded": excluded,
            "extraction_success_upper_bound": (len(rows) - excluded) / len(rows),
            "formed_success_threshold": 0.8, "real_model_effect": "not_run",
            "reason_counts": dict(Counter(reason for row in rows for reason in row["reasons"])),
            "by_topic": {topic: {"sample_count": sum(r["topic"] == topic for r in rows),
                                    "excluded": sum(r["topic"] == topic and r["structurally_excluded"] for r in rows)}
                         for topic in sorted({c.topic for c in suite.cases})}, "cases": rows}


def distraction_plan(suite: PrefEvalSuite) -> dict[str, list[dict]]:
    """Ten new response generations per topic; no upstream LMSYS conversation."""
    selected_ids = {c.id for c in suite.cases}
    result = {}
    for topic in sorted({c.topic for c in suite.cases}):
        candidates = [c for c in suite.pool if c.topic != topic and c.id not in selected_ids]
        rng = random.Random(f"{suite.seed}:{topic}:new-distractions-v1")
        result[topic] = [{"source_case_id": c.id, "source_topic": c.topic, "question": c.question}
                         for c in rng.sample(candidates, 10)]
    return result


def call_inventory(suite: PrefEvalSuite) -> dict:
    n, topics = len(suite.cases), len({c.topic for c in suite.cases})
    return {"first_session_online": n * 3, "generated_distraction_online": topics * 10,
            "four_arm_final_online": n * 4, "memory_extraction": n,
            "claude_crosscheck_positions": 40, "claude_calls_if_four_official_checks_separate": 160,
            "claude_calls_if_registered_joint_protocol": 40,
            "calls_before_judging": n * 3 + topics * 10 + n * 4 + n,
            "budget_usd": "1.00", "budget_reestimated": False,
            "budget_note": "Include distraction generation and register judge batching before any send. Existing 0.9 estimate is not a reservation.",
            "tokens": "unknown_until_actual_prompts_and_provider_receipts; bytes_are_not_tokens"}
