"""Permitted owner facts projected from the same bounded reply history."""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json

from ..context import EngineContext
from ..memory.referenceability import must_not_reference

DEFAULT_HISTORY_TURNS = 4
GROUNDING_SCHEMA = "owner_user_grounding_v2"
PrivacyPredicate = Callable[[str], bool]
_REUSE_SNAPSHOT = object()


def _predicate_key(predicate: PrivacyPredicate | None) -> str | None:
    if not callable(predicate):
        return None
    module = getattr(predicate, "__module__", None)
    name = getattr(predicate, "__qualname__", None)
    return f"{module}.{name}" if isinstance(module, str) and isinstance(name, str) else None


def _permitted(text: str, predicate: PrivacyPredicate | None) -> bool:
    if must_not_reference(text) or _predicate_key(predicate) is None:
        return False
    try:
        private = predicate(text)
    except Exception:
        return False
    # A malformed hook cannot grant factual grounding permission.
    return type(private) is bool and not private


def _window(ctx: EngineContext, count: int) -> tuple[list[tuple[int, str]], str]:
    from .fictional_context import fictional_conversation_history
    fictional_history = fictional_conversation_history(ctx)
    if fictional_history is not None:
        # This explicit one-response stimulus is the complete actual wire
        # history, not a paired Turn log or person evidence. Use original row
        # indexes; current runtime exclusions do not refer to these rows.
        # Existing privacy predicates still filter every user source below.
        turns = [(index, row["content"]) for index, row in enumerate(fictional_history)
                 if row["role"] == "user"]
        fingerprint = hashlib.sha256(json.dumps(
            {"fictional_conversation_history_v1": fictional_history},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return turns, fingerprint
    start = max(0, len(ctx.turns) - count)
    excluded = {item for item in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])
                if type(item) is int and item >= 0}
    turns = [(index, turn.user) for index, turn in enumerate(ctx.turns[start:], start)]
    fingerprint = hashlib.sha256(json.dumps(
        {"turns": turns, "excluded": sorted(excluded)},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return [(index, text) for index, text in turns if index not in excluded], fingerprint


def _bound_window(ctx: EngineContext, snapshot: object) -> int | None:
    if not (isinstance(snapshot, dict)
            and snapshot.get("schema") == GROUNDING_SCHEMA
            and snapshot.get("user_id") == ctx.user_id
            and snapshot.get("scene_id") == ctx.scene_id
            and snapshot.get("session_id") == ctx.session_id
            and type(snapshot.get("history_turns")) is int
            and snapshot["history_turns"] >= 0):
        return None
    return snapshot["history_turns"]


def _selection_sha256(snapshot: dict) -> str:
    # Internal cache consistency only; an actor able to rewrite every field
    # and recompute hashes is outside this in-process provenance boundary.
    return hashlib.sha256(json.dumps(
        {"predicate_key": snapshot.get("predicate_key"), "source_rows": snapshot.get("source_rows")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def prepare_user_grounding(
    ctx: EngineContext, history_turns: int,
    privacy_predicate: PrivacyPredicate | None = None,
    *, current_utterance: str | None = None,
) -> None:
    count = history_turns if type(history_turns) is int and history_turns >= 0 else 0
    turns, fingerprint = _window(ctx, count)
    ctx.working_memory["_prompt_user_grounding_for_turn"] = {
        "schema": GROUNDING_SCHEMA, "user_id": ctx.user_id,
        "scene_id": ctx.scene_id, "session_id": ctx.session_id,
        "turn_index": len(ctx.turns), "history_turns": count,
        "predicate_key": _predicate_key(privacy_predicate),
        "window_sha256": fingerprint,
        "utterance_sha256": hashlib.sha256(current_utterance.encode("utf-8")).hexdigest()
        if isinstance(current_utterance, str) else None,
        "source_rows": [{"index": index, "text": text} for index, text in turns
                        if _permitted(text, privacy_predicate)],
    }
    snapshot = ctx.working_memory["_prompt_user_grounding_for_turn"]
    snapshot["sources"] = [row["text"] for row in snapshot["source_rows"]]
    snapshot["selection_sha256"] = _selection_sha256(snapshot)


def _current_sources_are_bound(ctx: EngineContext, snapshot: dict, count: int, utterance: str) -> bool:
    turns, fingerprint = _window(ctx, count)
    rows = snapshot.get("source_rows")
    return bool(
        type(snapshot.get("turn_index")) is int
        and snapshot.get("turn_index") == len(ctx.turns)
        and snapshot.get("window_sha256") == fingerprint
        and snapshot.get("utterance_sha256") == hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        and isinstance(snapshot.get("predicate_key"), str)
        and isinstance(rows, list)
        and all(isinstance(row, dict) and set(row) == {"index", "text"}
                and type(row["index"]) is int and isinstance(row["text"], str)
                and (row["index"], row["text"]) in turns
                and not must_not_reference(row["text"]) for row in rows)
        and len({row["index"] for row in rows}) == len(rows)
        and snapshot.get("sources") == [row["text"] for row in rows]
        and snapshot.get("selection_sha256") == _selection_sha256(snapshot)
    )


def user_grounding_sources(
    ctx: EngineContext, utterance: str,
    privacy_predicate: PrivacyPredicate | None | object = _REUSE_SNAPSHOT,
) -> list[str]:
    snapshot = ctx.working_memory.get("_prompt_user_grounding_for_turn")
    count = _bound_window(ctx, snapshot)
    if privacy_predicate is not _REUSE_SNAPSHOT:
        # A guard may rebuild after configuration/history changed, but never
        # broadens a registered reply window to the default. Old/unbound
        # snapshots fail closed; direct scene callers with no snapshot use 4.
        count = count if count is not None else DEFAULT_HISTORY_TURNS if snapshot is None else 0
        prepare_user_grounding(ctx, count, privacy_predicate, current_utterance=utterance)
        snapshot = ctx.working_memory["_prompt_user_grounding_for_turn"]
    elif not (count is not None and _current_sources_are_bound(ctx, snapshot, count, utterance)):
        return [utterance]
    # Current wording permits a response to this owner, never proactive recall
    # or third-party sharing. No callable is persisted in checkpoint data.
    return [utterance, *snapshot["sources"]]
