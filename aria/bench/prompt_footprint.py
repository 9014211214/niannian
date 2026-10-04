"""Local prompt/trace character accounting; never opens a model client.

Counts Unicode code points, not tokens or latency. Public reports contain sizes
only. Provider fields outside the declared application decision are ignored.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Iterable

from aria.engine.context import EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.online_decision import (
    ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2,
    online_decision_tool_spec, single_pass_messages,
)
from aria.engine.policy.prompt import build_messages, prompt_version_for
from aria.scenes.registry import get_scene


VERSION = "prompt_footprint_v1"
_BOUNDARY = re.compile(r"(?m)^(?:## |【(?:交付模式|相处画像)[:：])")
# Only application protocol fields; deliberately excludes provider reasoning.
_FIELDS = (
    "response", "state_signals", "current_affect", "deliberation",
    "selected_intervention_id", "communication_plan", "uncertainty",
    "state_signals.*.evidence_quote", "state_signals.*.rationale",
    "current_affect.label", "current_affect.divergence.pattern",
    "deliberation.person_read", "deliberation.applied_basis",
    "deliberation.options.*.predicted_reaction",
    "communication_plan.pace", "communication_plan.support_order",
    "communication_plan.boundary",
)
_LABELS = {
    "## 安全红线": "safety_rules",
    "## 数据与指令边界": "data_instruction_boundary",
    "【交付模式": "delivery_mode",
    "【相处画像": "delivery_profile",
    "## 真实冷启动": "cold_start",
    "## 认识阶段": "observing_stage",
    "## 对这个人的方式": "person_context",
    "## 经本人对话确认的关系上下文": "relationship_context",
    "## 这一拍怎么接": "strategy",
    "## 本轮回复契约": "reply_contract",
    "## 在线单次情感决策": "online_decision_contract",
    "## 在线先推演后回复": "online_deliberation_contract",
}


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def measure_messages(messages: list[dict], *, tool_spec: dict | None = None) -> dict:
    """Partition system text at generated headings; keep every separator.

    IDs are positional so arbitrary prompt/heading contents never enter output.
    This is a textual partition, not a claim about semantic independence. Nested
    headings (e.g. care records) are separate rows and remain in original order.
    """
    blocks = []
    by_role: dict[str, int] = {}
    for index, message in enumerate(messages):
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("footprint_requires_text_messages")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("footprint_unknown_role")
        by_role[role] = by_role.get(role, 0) + len(content)
        if role != "system":
            continue
        offsets = sorted({0, len(content), *[m.start() for m in _BOUNDARY.finditer(content)]})
        for number, (start, end) in enumerate(zip(offsets, offsets[1:]), 1):
            blocks.append({
                "block_id": f"system_{index}_block_{number:02d}",
                "kind": next((label for prefix, label in _LABELS.items()
                              if content[start:end].startswith(prefix)),
                             "identity" if start == 0 else "other_section"),
                "start": start, "end": end, "characters": end - start,
            })
    return {
        "unit": "unicode_code_points_not_tokens",
        "message_content_characters": sum(by_role.values()),
        "content_characters_by_role": by_role,
        "system_blocks": blocks,
        "top_five_system_blocks": sorted(blocks, key=lambda row: -row["characters"])[:5],
        "tool_spec_compact_json_characters": _json_size(tool_spec) if tool_spec else 0,
        "message_envelope_compact_json_characters": _json_size(messages),
    }


def _distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "observed": len(values),
        "min": min(values) if values else None,
        "mean": mean(values) if values else None,
        "p50": median(values) if values else None,
        "p95_nearest_rank": ordered[math.ceil(.95 * len(values)) - 1] if values else None,
        "max": max(values) if values else None,
    }


def _field_values(value: object, path: list[str]) -> list:
    if not path:
        return [value]
    first, *rest = path
    if first == "*" and isinstance(value, list):
        return [found for item in value for found in _field_values(item, rest)]
    if isinstance(value, dict) and first in value:
        return _field_values(value[first], rest)
    return []


def measure_output_fields(payloads: Iterable[dict]) -> dict:
    """Count declared output fields, keeping missing/null/invalid separate.

    Input is decoded model decision objects, BEFORE local repair/guard. The
    caller must identify whether they are synthetic or real and bind versions.
    No fabricated sample or zero-filled missing field is used by the CLI.
    Declared objects are measured as received, including extra nested keys;
    length accounting does not validate their schema. Contents are never stored
    in the report. Nested path rows select only the explicitly listed leaves.
    """
    rows = list(payloads)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("footprint_requires_decoded_objects")
    fields = {}
    for field in _FIELDS:
        sizes, text_sizes, missing, null, invalid = [], [], 0, 0, 0
        for row in rows:
            values = _field_values(row, field.split("."))
            if not values:
                missing += 1
            for value in values:
                if value is None:
                    null += 1
                    continue
                try:
                    size = _json_size(value)
                except (ValueError, TypeError, OverflowError):
                    invalid += 1
                else:
                    sizes.append(size)
                    if isinstance(value, str):
                        text_sizes.append(len(value))
        fields[field] = {
            **_distribution(sizes), "missing": missing, "null": null,
            "invalid_json_value": invalid,
            "string_content_characters": _distribution(text_sizes),
        }
    return {
        "samples": len(rows), "unit": "compact_json_unicode_code_points_not_tokens",
        "includes_json_quotes_and_punctuation": True, "fields": fields,
        "nested_array_scope": "Observed counts array elements; missing counts samples with no matching leaf.",
    }


def fixture_report() -> dict:
    """Two no-history fixtures, each with both delivery blocks and protocols.

    No orchestrator/perception/recall is simulated. These are the real production
    builders with empty per-turn state; long runtime care/history inputs must be
    measured separately. The Lao Chen persona is author-supplied, not learned.
    """
    scene = get_scene("companion_elder")
    rows = []
    for fixture in ("default", "laochen_demo"):
        for block in ("mode_v1", "profile_v1"):
            for protocol in (ONLINE_DECISION_VERSION, ONLINE_DECISION_VERSION_V2):
                persona = (PersonaProfile(user_id="footprint_default") if fixture == "default"
                           else load_demo_persona("laochen_demo"))
                ctx = EngineContext(user_id=persona.user_id, scene_id=scene.id,
                                    session_id="local-footprint")
                ctx.emotion.label = "unknown"
                messages = build_messages(scene, persona, ctx, "今天有点闷。", delivery_block=block)
                messages = single_pass_messages(messages, ctx, version=protocol,
                                                persona=persona, delivery_block=block)
                rows.append({
                    "fixture": fixture, "delivery_block": block,
                    "prompt_version": prompt_version_for(block),
                    "online_decision_version": protocol,
                    **measure_messages(messages, tool_spec=online_decision_tool_spec(protocol)),
                })
    return {
        "version": VERSION, "source": "local_production_builders_with_synthetic_fixtures",
        "model_calls": 0, "model_effect_measured": False,
        "scope": "No history, recalled facts, care records, or reasoning candidates; not a runtime maximum.",
        "input_measurements": rows,
        "output_measurements": measure_output_fields([]),
        "output_status": "not_run_no_current_model_samples",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fixture_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an earlier measurement silently.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
