"""Exact character-origin projection for the explicit semantic candidate.

No fuzzy matching or text search: each output character is either retained at
its original position or produced by a trusted code transformation. Only the
current trace owns a projection; its caller still verifies owner/turn/input.
"""
from __future__ import annotations

import hashlib


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_positions(ctx, text):
    trace = ctx.working_memory.get("online_turn_decision") or {}
    if trace.get("version") not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
        return None
    source = trace.get("response_segments")
    if type(source) is not list:
        return None
    original = "".join(row["text"] for row in source)
    if _sha(original) != trace.get("parsed_response_sha256"):
        return None
    original_positions = [index for index, row in enumerate(source) for _ in row["text"]]
    projection = trace.get("semantic_span_projection") or {}
    if (projection.get("output_sha256") != _sha(text)
            or projection.get("source_sha256") != trace.get("parsed_response_sha256")):
        if text == original:
            return original_positions
        # Edge stripping is an exact transformation with unique original offsets.
        if text == original.strip():
            start = len(original) - len(original.lstrip())
            return original_positions[start:start + len(text)]
        return None
    positions = [None] * len(text)
    previous = 0
    for row in projection.get("spans", []):
        if (type(row) is not dict or set(row) != {"start", "end", "index"}
                or any(type(row[key]) is not int for key in row)
                or not 0 <= row["index"] < len(source)
                or not previous <= row["start"] < row["end"] <= len(text)):
            return None
        positions[row["start"]:row["end"]] = [row["index"]] * (row["end"] - row["start"])
        previous = row["end"]
    return positions


def publish_positions(ctx, text, positions):
    trace = ctx.working_memory.get("online_turn_decision") or {}
    if trace.get("version") not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
        return
    if len(text) != len(positions):
        raise ValueError("delivery_position_length_mismatch")
    spans = []
    start = 0
    while start < len(positions):
        index = positions[start]
        end = start + 1
        while end < len(positions) and positions[end] == index:
            end += 1
        if index is not None:
            spans.append({"start": start, "end": end, "index": index})
        start = end
    trace["semantic_span_projection"] = {"source_sha256": trace.get("parsed_response_sha256"),
        "output_sha256": _sha(text), "spans": spans}


def project_exact_transform(ctx, before, after, source_positions):
    """The transforming code supplies real before offsets, None for code text."""
    current = model_positions(ctx, before)
    if current is None or len(source_positions) != len(after):
        return False
    positions = []
    last = -1
    for char, index in zip(after, source_positions):
        if index is None:
            positions.append(None)
        elif (type(index) is not int or not last < index < len(before) or index < 0
              or before[index] != char):
            raise ValueError("delivery_position_source_mismatch")
        else:
            last = index
            positions.append(current[index])
    publish_positions(ctx, after, positions)
    return True


def publish_code_response(ctx, text):
    publish_positions(ctx, text, [None] * len(text))


def publish_preference_composition(ctx, text, retained_indexes):
    trace = ctx.working_memory.get("online_turn_decision") or {}
    rows = trace.get("response_segments") or []
    body = "".join(rows[index]["text"] for index in retained_indexes)
    if not text.startswith(body):
        raise ValueError("preference_composer_position_mismatch")
    positions = [index for index in retained_indexes for _ in rows[index]["text"]]
    publish_positions(ctx, text, positions + [None] * (len(text) - len(body)))


def parse_action_projected(ctx, response):
    from .actions import parse_action
    if ctx.working_memory.get("_online_decision_version") not in ("online_turn_decision_v6_semantic", "online_turn_decision_v7"):
        return parse_action(response)
    positions = []
    text, action = parse_action(response, source_positions=positions)
    project_exact_transform(ctx, response, text, positions)
    return text, action
