"""Scene hook and source-window mechanics; fixed software inputs only."""
from copy import deepcopy
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.user_grounding import prepare_user_grounding, user_grounding_sources
from aria.scenes.companion_elder.delivery import _grounding_sources
from aria.scenes.companion_elder.scene import ElderCompanionScene


def permitted(text):
    return text.startswith("PRIVATE")


def other_permitted(text):
    return False


def context():
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("first owner source", "assistant-only source")
    ctx.add_turn("PRIVATE second owner source", "assistant-only second")
    ctx.add_turn("third owner source", "assistant-only third")
    return ctx


@pytest.mark.parametrize("count,expected", [(0, []), (1, ["third owner source"]), (3, ["first owner source", "third owner source"]), (True, []), (-1, []), (float('inf'), [])])
def test_exact_registered_window_and_strict_count(count, expected):
    ctx = context()
    prepare_user_grounding(ctx, count, permitted, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current", *expected]
    assert user_grounding_sources(ctx, "current", other_permitted) == ["current", *([] if type(count) is not int or count < 0 else [turn.user for turn in (ctx.turns[-count:] if count else [])])]
    json.dumps(ctx.working_memory, ensure_ascii=False)


@pytest.mark.parametrize("mutation", ["schema", "owner", "scene", "session", "bool_count", "negative_count"])
def test_unbound_snapshot_never_rebuilds_default_window(mutation):
    ctx = context()
    prepare_user_grounding(ctx, 1, permitted, current_utterance="current")
    snap = ctx.working_memory["_prompt_user_grounding_for_turn"]
    key, value = {"schema": ("schema", "old"), "owner": ("user_id", "other"), "scene": ("scene_id", "other"), "session": ("session_id", "other"), "bool_count": ("history_turns", True), "negative_count": ("history_turns", -1)}[mutation]
    snap[key] = value
    assert user_grounding_sources(ctx, "current") == ["current"]
    assert user_grounding_sources(ctx, "current", other_permitted) == ["current"]
    assert ctx.working_memory["_prompt_user_grounding_for_turn"]["history_turns"] == 0


@pytest.mark.parametrize("mutation", ["foreign_source", "assistant_source", "false_index", "duplicate", "text_only", "current_utterance", "excluded", "history_changed"])
def test_consumers_reject_tampered_or_stale_source_rows(mutation):
    ctx = context()
    prepare_user_grounding(ctx, 1, permitted, current_utterance="current")
    snap = ctx.working_memory["_prompt_user_grounding_for_turn"]
    if mutation in {"foreign_source", "assistant_source"}:
        snap["source_rows"][0]["text"] = "foreign" if mutation == "foreign_source" else ctx.turns[-1].agent
        snap["sources"] = [snap["source_rows"][0]["text"]]
    elif mutation == "false_index":
        snap["source_rows"][0]["index"] = True
    elif mutation == "duplicate":
        snap["source_rows"] *= 2
        snap["sources"] *= 2
    elif mutation == "text_only":
        snap["sources"] = ["foreign"]
    elif mutation == "excluded":
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [2]
    elif mutation == "history_changed":
        ctx.turns[-1].user = "edited owner source"
    assert user_grounding_sources(ctx, "other" if mutation == "current_utterance" else "current") == ["other" if mutation == "current_utterance" else "current"]


@pytest.mark.parametrize("result", [None, 0, 1, "false", [], {}])
def test_non_boolean_hook_cannot_grant_history(result):
    def invalid(text):
        return result
    ctx = context()
    prepare_user_grounding(ctx, 3, invalid, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current"]


def test_missing_and_raising_hook_fail_closed_without_persisting_callable():
    ctx = context()
    assert user_grounding_sources(ctx, "current") == ["current"]
    prepare_user_grounding(ctx, 3, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current"]
    def raising(text):
        raise RuntimeError("software hook failure")
    prepare_user_grounding(ctx, 3, raising, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current"]
    json.dumps(ctx.working_memory)


def test_direct_scene_keeps_privacy_hook_and_existing_zero_window():
    ctx = context()
    ctx.add_turn("最近记不住事情，女儿已经回复了。", "收到。")
    assert "最近记不住事情，女儿已经回复了。" not in _grounding_sources(ctx, "current", None)
    build_messages(ElderCompanionScene(), None, ctx, "current", history_turns=0)
    before = deepcopy(ctx.working_memory["_prompt_user_grounding_for_turn"])
    assert _grounding_sources(ctx, "current", None) == ["current"]
    assert ctx.working_memory["_prompt_user_grounding_for_turn"] == before


@pytest.mark.parametrize("source", ["这件事别提起，女儿已经回复了。", "最近记不住事情，女儿已经回复了。"])
def test_filtered_actual_window_row_cannot_be_spliced_back_into_online_cache(source):
    from aria.engine.policy.online_decision import _response_grounding_sources
    ctx = context()
    ctx.add_turn(source, "收到。")
    build_messages(ElderCompanionScene(), None, ctx, "current", history_turns=1)
    snap = ctx.working_memory["_prompt_user_grounding_for_turn"]
    assert snap["sources"] == []
    snap["source_rows"] = [{"index": 3, "text": source}]
    snap["sources"] = [source]
    assert _response_grounding_sources(ctx, "current", None) == ["current"]
    assert _grounding_sources(ctx, "current", None) == ["current"]


def test_boolean_current_turn_header_is_not_a_real_index():
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("owner source", "assistant source")
    prepare_user_grounding(ctx, 1, permitted, current_utterance="current")
    ctx.working_memory["_prompt_user_grounding_for_turn"]["turn_index"] = True
    assert user_grounding_sources(ctx, "current") == ["current"]


@pytest.mark.parametrize("configuration", ["none", "raising", "strict"])
def test_scene_configuration_remains_identical_in_prompt_online_and_final(configuration):
    from aria.engine.policy.online_decision import _response_grounding_sources
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    ctx.add_turn("女儿已经回复了。", "收到。")
    scene = ElderCompanionScene()
    def private(text):
        if configuration == "raising":
            raise RuntimeError("software hook failure")
        return True
    scene.grounding_privacy_predicate = None if configuration == "none" else private
    build_messages(scene, None, ctx, "current", history_turns=1)
    assert _response_grounding_sources(ctx, "current", None) == ["current"]
    assert scene.guard_delivery_response(ctx, "女儿已经回复了。", "current") != "女儿已经回复了。"
    assert _response_grounding_sources(ctx, "current", None) == ["current"]
    assert ctx.working_memory["_prompt_user_grounding_for_turn"]["history_turns"] == 1
    json.dumps(ctx.working_memory)


def test_explicit_missing_hook_does_not_reuse_previous_more_permissive_filter():
    ctx = context()
    prepare_user_grounding(ctx, 1, permitted, current_utterance="current")
    assert user_grounding_sources(ctx, "current") == ["current", "third owner source"]
    assert user_grounding_sources(ctx, "current", None) == ["current"]
