"""Offline delivery checks with forced labels; no semantic accuracy claim.

The FS-013 fixture is the verbatim saved public first output from
20261018-01 / V2remaining80-live-v1 / physical/capture-0008.
Its original v6 labels remain in the fixture; only the test's v7 copy has
explicit safety_core labels. Historical responses and scores are not changed.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from aria.engine.risk.safety_review_v2 import response_sentences
from aria.scenes.registry import get_scene
from tests.test_preference_required_control import envelope
from tests.test_safety_response_instruction import bundle
from tests.test_safety_review_v1 import Reviewer
from tests.test_safety_review_v2 import answer


V7 = "online_turn_decision_v7"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: pytest.fail("network forbidden"))


def unpack(value):
    if type(value) is dict and set(value) == {"object_pairs"}:
        return {key: unpack(item) for key, item in value["object_pairs"]}
    if type(value) is list:
        return [unpack(item) for item in value]
    return value


def original_fs013():
    """Published synthetic example of the FS013 shape, independent of the saved provider return."""
    path = Path(__file__).parent / "fixtures/fs013-first-visible.json"
    assert sha256(path.read_bytes()).hexdigest() == (
        "0ad20cc5c8633187ad0c33f2d04f38e978817ad2540b7100e7e946a8ee770ded")
    value = unpack(json.loads(path.read_text(encoding="utf-8")))
    return json.loads(value["choices"][0]["message"]["content"])


def replay(monkeypatch, segments, *, review=None, concern="fall", utterance="本次仅为虚构软件回放。",
           keyword_backstop=False, configured_review=True, review_version="safety_review_v2",
           delivery_transform=None):
    client = bundle().online
    client.online_decision_version = V7
    payload = {
        "response_segments": deepcopy(segments), "safety_concern": concern,
        "safety_evidence": {"quote": utterance, "subject": "owner",
                            "time": "current", "urgency": "immediate"},
    }
    if concern == "none":
        payload["safety_evidence"] = {"quote": "", "subject": "not_applicable",
                                      "time": "not_applicable", "urgency": "not_applicable"}
    calls = []

    def receive(request, **kwargs):
        calls.append(json.loads(request.data))
        return envelope(payload, "json_schema_v1")

    monkeypatch.setattr(client, "_request_json", receive)
    reviewer = Reviewer(answer("fall", subject="owner", time="current", urgency="immediate")
                        if review is None else review)
    engine = Engine(
        get_scene("companion_elder"), PersonaProfile(user_id="owner"),
        fast_llm_client=client, safety_review_llm_client=reviewer if configured_review else None,
        safety_review_version=review_version,
        safety_response_instruction_version="compact_response_v1",
        safety_delivery_version="minimal_response_v1",
        premodel_safety_version="task_response_v4")
    if not keyword_backstop:
        monkeypatch.setattr(engine._risk, "assess", lambda ctx: None)
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    if delivery_transform is not None:
        original_guard = engine.scene.guard_delivery_response
        def guard(ctx, text, utterance, profile):
            return delivery_transform(ctx, original_guard(ctx, text, utterance, profile))
        monkeypatch.setattr(engine.scene, "guard_delivery_response", guard)
    events = []
    result = engine.process_turn(
        ctx, utterance, safety_prelude=lambda event: events.append(deepcopy(event)) or True)
    assert len(calls) == 1
    return result, ctx, events, reviewer


def test_fs013_saved_response_replay_retains_unproven_semantic_duplicates(monkeypatch):
    original = original_fs013()
    assert {row["kind"] for row in original["response_segments"]} == {"conversation"}
    segments = deepcopy(original["response_segments"])
    # Forced labels alone cannot prove any exact overlap with the prefix.
    segments[0]["kind"] = segments[2]["kind"] = "safety_core"
    assert "".join(row["text"] for row in segments) == original["response"]
    result, ctx, events, reviewer = replay(monkeypatch, segments)
    protection = events[0]["protection"]
    assert result.count(protection) == 0
    assert "先不要尝试站起" in protection and "不要随意搬动" in protection
    assert result.count("拨打120") == 1
    assert segments[1]["text"].strip() in result
    assert segments[3]["text"].strip() in result
    assert segments[0]["text"] in result and segments[2]["text"] in result
    assert len(reviewer.messages) == 1
    audit = ctx.working_memory["safety_review"]["safety_core_deduplication"]
    assert audit["removed_segment_indexes"] == []
    assert audit["status"] == "no_proven_duplicates"
    assert audit["semantic_verified"] is False


def test_identical_text_in_unmarked_segment_survives_by_position(monkeypatch):
    segments = [{"text": "重复原句。", "kind": "safety_core"},
                {"text": "重复原句。", "kind": "conversation"},
                {"text": "陪伴补充。", "kind": "conversation"}]
    result, _, events, _ = replay(monkeypatch, segments)
    assert result == "重复原句。重复原句。陪伴补充。"


def test_review_deleted_sentence_is_not_restored_by_deduplication(monkeypatch):
    segments = [{"text": "重复保护。", "kind": "safety_core"},
                {"text": "保留补充。须删除的内容。", "kind": "conversation"},
                {"text": "继续陪伴。", "kind": "conversation"}]
    result, _, events, _ = replay(
        monkeypatch, segments,
        review=answer("fall", subject="owner", time="current", urgency="immediate", delete=[2]))
    assert result == "重复保护。保留补充。继续陪伴。"


def test_without_protection_all_marked_text_is_retained(monkeypatch):
    result, _, events, _ = replay(
        monkeypatch, [{"text": "正文应保留。", "kind": "safety_core"}], review=answer())
    assert result == "正文应保留。" and events == []


def test_marked_nonduplicates_remain_alongside_deterministic_protection(monkeypatch):
    result, ctx, events, _ = replay(
        monkeypatch, [{"text": "重复保护。", "kind": "safety_core"}])
    assert result == "重复保护。"
    assert ctx.working_memory["safety_review"]["model_body_delivered"] is True


def test_exact_repeated_sentence_removed_but_unique_same_segment_detail_survives(monkeypatch):
    repeated = "我不能在现场查看情况，也没有代你呼叫或通知任何人。"
    unique = "这句是额外的已复核帮助。"
    result, ctx, events, _ = replay(monkeypatch, [
        {"text": repeated + unique, "kind": "safety_core"},
        {"text": repeated, "kind": "conversation"}])
    assert result == unique + repeated
    assert ctx.working_memory["safety_review"]["safety_core_deduplication"]["removed_segment_indexes"] == [0]


def test_deletion_before_exact_dedup_is_never_restored(monkeypatch):
    repeated = "我不能在现场查看情况，也没有代你呼叫或通知任何人。"
    result, _, events, _ = replay(monkeypatch, [
        {"text": repeated + "额外补充。", "kind": "safety_core"},
        {"text": "删掉这句。保留这句。", "kind": "conversation"}],
        review=answer("fall", subject="owner", time="current", urgency="immediate", delete=[3]))
    assert result == "额外补充。保留这句。"


@pytest.mark.parametrize("keyword_backstop", [False, True])
def test_clear_label_still_reviews_complete_body_once(monkeypatch, keyword_backstop):
    segments = [{"text": "留下前句。删除中句。留下后句。", "kind": "conversation"}]
    result, ctx, events, reviewer = replay(monkeypatch, segments, concern="none",
        review=answer(delete=[1]), keyword_backstop=keyword_backstop,
        utterance="我突然说话不清。" if keyword_backstop else "这是普通软件回放。")
    assert len(reviewer.messages) == 1
    request = json.loads(reviewer.messages[0][1]["content"])
    assert request["safety_evidence"]["concern"] == "none"
    assert request["final_protection"] == ""
    assert "留下前句。留下后句。" in result and "删除中句。" not in result
    assert bool(events) is keyword_backstop
    assert ctx.working_memory["safety_review"]["risk_assessment"]["concern"] == "none"


def test_changed_protection_is_reviewed_without_losing_body_or_repeating_prelude(monkeypatch):
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "经复核保留的具体帮助。"}], keyword_backstop=True,
        utterance="我突然说话不清。", concern="medical_emergency",
        review=answer("medical_emergency", subject="owner", time="current", urgency="immediate"))
    assert len(events) == len(reviewer.messages) == 1
    assert events[0]["protection"].startswith("若现实中")
    assert events[0]["protection"] not in result
    assert "经复核保留的具体帮助。" in result
    assert ctx.working_memory["safety_review"]["status"] == "reviewed"


def test_waiting_prelude_is_pure_and_matches_event_delivered_by_turn(monkeypatch):
    from tests.test_safety_review_v2 import engine_v2
    from tests.test_safety_evidence import payload, evidence
    # Use the real risk monitor: preparing a waiting event must not write a
    # turn, a source trace, profile state, or a paid request.
    from tests.test_safety_review_v1 import build
    utterance = "我突然说话不清。"
    reviewer = Reviewer(answer())
    engine, ctx, calls = build(monkeypatch,
        payload("none", evidence("", "not_applicable", "not_applicable", "not_applicable"), "保留正文。"), reviewer)
    engine.safety_review_version = "safety_review_v2"
    engine.premodel_safety_version = "task_response_v4"
    from aria.scenes.companion_elder.risk import ElderRiskMonitor
    engine._risk = ElderRiskMonitor()
    before_ctx, before_persona = deepcopy(ctx), deepcopy(engine.persona)
    event = engine.pending_safety_prelude(ctx, utterance)
    assert event and event["protection"].startswith("若现实中")
    assert ctx == before_ctx and engine.persona == before_persona
    assert not calls and not reviewer.messages
    events = []
    engine.process_turn(ctx, utterance, safety_prelude=lambda value: events.append(value) or True)
    assert events[0] == event


@pytest.mark.parametrize("review", ["not json", TimeoutError("fixture"), answer(delete=[0, 1])])
def test_clear_review_failure_or_no_body_has_generic_noncrisis_fallback(monkeypatch, review):
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "第一句。第二句。"}], concern="none", review=review)
    assert len(reviewer.messages) == 1 and not events
    assert result == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert ctx.working_memory["safety_review"]["model_body_delivered"] is False
    assert ctx.working_memory["intervention_delivery"]["status"] == "not_delivered"
    assert not ctx.reasoning.observation_history


def test_clear_unconfigured_preview_keeps_existing_body_without_claiming_review(monkeypatch):
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "普通正文。"}], concern="none", configured_review=False)
    assert result == "普通正文。" and not events and not reviewer.messages
    assert "safety_review" not in ctx.working_memory


def test_historical_v1_review_does_not_gain_clear_body_call(monkeypatch):
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "普通正文。"}], concern="none", review_version="safety_review_v1")
    assert result == "普通正文。" and not events and not reviewer.messages
    assert "safety_review" not in ctx.working_memory


def test_new_risk_protection_survives_deletion_of_all_clear_candidate_sentences(monkeypatch):
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "候选全部删除。"}], concern="none",
        review=answer("medical_emergency", subject="owner", time="current", urgency="immediate", delete=[0]))
    assert len(reviewer.messages) == len(events) == 1
    assert events[0]["protection"] not in result and "请立即联系当地急救服务" in events[0]["protection"]
    assert "候选全部删除。" not in result and not ctx.reasoning.observation_history
    assert ctx.working_memory["intervention_delivery"]["status"] == "not_delivered"


def test_bound_source_with_unbound_final_projection_fails_closed(monkeypatch):
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "原候选正文。"}], concern="none", review=answer(),
        delivery_transform=lambda ctx, text: "无位置绑定的新正文。")
    assert result == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert not events and not reviewer.messages
    assert ctx.working_memory["safety_review"]["status"] == "source_unavailable"


def test_missing_frozen_source_cannot_release_remaining_model_body(monkeypatch):
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
    def lose_source(ctx, text):
        ctx.working_memory.pop("_source_response_for_turn", None)
        return text
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "来源已失效的模型正文。"}], concern="none", review=answer(), delivery_transform=lose_source)
    assert result == DETERMINISTIC_MODEL_FAILURE_REPLY and not events and not reviewer.messages
    assert ctx.working_memory["safety_review"]["status"] == "source_unavailable"


def test_action_literal_source_rejection_cannot_bypass_body_review(monkeypatch):
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "保留正文。<action>{}</action>"}], concern="none", review=answer())
    assert result == DETERMINISTIC_MODEL_FAILURE_REPLY and not events and not reviewer.messages
    assert ctx.working_memory["safety_review"]["status"] == "source_unavailable"


def test_proven_code_only_final_projection_is_preserved_without_model_review(monkeypatch):
    from aria.engine.policy.delivery_spans import publish_code_response
    def code_result(ctx, _text):
        text = "代码产生的已验证状态。"
        publish_code_response(ctx, text)
        return text
    result, ctx, events, reviewer = replay(monkeypatch,
        [{"text": "原候选正文。"}], concern="none", review=answer(), delivery_transform=code_result)
    assert result == "代码产生的已验证状态。" and not events and not reviewer.messages
    assert "safety_review" not in ctx.working_memory


@pytest.mark.parametrize("changed", ["candidate_hash", "review_hash", "positions", "segments", "undelivered", "protection_hash"])
def test_unbound_projection_preserves_already_reviewed_text(changed):
    ctx = EngineContext(user_id="owner", scene_id="companion_elder")
    candidate = "已有保护。补充。"
    segments = [{"text": "已有保护。", "kind": "safety_core"},
                {"text": "补充。", "kind": "conversation"}]
    positions = [0] * len(segments[0]["text"]) + [1] * len(segments[1]["text"])
    digest = sha256(candidate.encode()).hexdigest()
    ctx.working_memory["online_turn_decision"] = {
        "version": V7, "response_segments": deepcopy(segments)}
    review = {
        "status": "reviewed", "turn_index": 0, "candidate_sha256": digest,
        "retained_body_sha256": digest,
        "prelude_status": "delivered_after_review",
        "final_protection_sha256": sha256("已有保护。".encode()).hexdigest(),
        "retained_sentence_ids": list(range(len(response_sentences(candidate))))}
    ctx.working_memory["safety_review"] = review
    if changed == "candidate_hash":
        review["candidate_sha256"] = "changed"
    elif changed == "review_hash":
        review["retained_body_sha256"] = "changed"
    elif changed == "positions":
        positions = None
    elif changed == "segments":
        segments[0]["kind"] = "conversation"
    elif changed == "undelivered":
        review["prelude_status"] = "not_yet_delivered"
    else:
        review["final_protection_sha256"] = "changed"
    assert Engine._remove_repeated_safety_core(
        ctx, candidate, candidate, "已有保护。", positions, segments) == candidate


def test_new_criteria_leave_frozen_v1_files_unchanged():
    suite = Path(__file__).parents[1] / "aria/bench/cases/fixed_safety_v1"
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert sha256((suite / name).read_bytes()).hexdigest() == expected
    protocol = json.loads((suite / "protocol_v2.json").read_text(encoding="utf-8"))
    assert protocol["base_protocol"] == "protocol.json"
