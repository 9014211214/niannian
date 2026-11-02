"""Independent authored replies drive fixed judge labels, never semantic scoring."""
from contextlib import closing
from dataclasses import replace
import json
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.preference_dialogue import (
    CONFIRM_JUDGE_INSTRUCTION, JUDGED_ONLINE,
    acknowledge_preference_invitation, apply_preference_dialogue, bind_preference_request,
    confirm_judge_messages, confirm_judge_request_format, current_preference_invitation,
    judge_preference_confirmation, parse_confirm_judge, prepare_preference_invitation,
    validate_confirm_judge_messages,
)
from aria.engine.memory.stated_preferences import extract_preference_rows, merge_preferences
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.engine.policy.qwen_max_structured import validate_qwen_max_client
from aria.runtime import create_minimal_engine
from aria.scenes.registry import get_scene
from tests.preference_fixtures import communication_review
from tests.test_installable_max_release import make_clients, guard, Response
from aria.engine.policy.max_release import tls_dispatch

DO = "记住您喜欢晚饭后在小区慢慢散步这一项偏好。"
SCOPE = "仅在我们以后的聊天中，当您询问活动建议时参考这项偏好；不分享给其他人，也不自动安排活动。"
BODY = "说起今天的事，我愿意接着听。"

# Written by /root/arch_t2_author before viewing implementation or prior tests.
# These labels are supplied by a stub, so this is software behavior evidence only.
REPLIES = [
    ("confirm", "愿意，你就按刚才说的记下来吧。"),
    ("confirm", "可以呀，以后聊活动的时候你心里就有数了。"),
    ("confirm", "行，这个习惯你帮我记着。"),
    ("confirm", "就这么办吧，我同意。"),
    ("confirm", "没问题，记下吧，省得我下回又从头说。"),
    ("confirm", "行，你记着吧。对了，我今天买的橘子可甜了。"),
    ("confirm", "我愿意，就照你说的记。哎，你知道豆腐汤怎么烧才鲜吗？"),
    ("confirm", "可以，把这条记下来。刚才我孙女打电话，说学校要开运动会呢。"),
    ("confirm", "好，这个散步习惯你记住就行。我那盆兰花终于冒新芽了。"),
    ("confirm", "同意，你按刚才讲的办吧。咱们再聊聊老电影，我昨晚看了半部，还挺有意思。"),
    ("decline", "不用记了，下次聊到我再告诉你。"),
    ("decline", "这个我不愿意让你留下来记着。"),
    ("decline", "算啦，我就是随口说说，别给我记上。"),
    ("decline", "你有这个心挺好，不过我的生活习惯还是不要保存了。"),
    ("decline", "不行，这件事我不同意，咱们直接聊天就好。"),
    ("decline", "留着这条对我没什么用，就别记了吧。"),
    ("defer", "让我想一想，过两天再决定要不要记。"),
    ("defer", "这会儿先搁着吧，我正忙着收衣服，晚些再说。"),
    ("defer", "我还没拿定主意呢，等下回聊的时候再问我。"),
    ("defer", "先别急着记，等我看看这阵子能不能坚持散步，再跟你说。"),
    ("unclear", "要记就只记我喜欢散步，晚饭后这个时间别写死。"),
    ("unclear", "地点改成附近公园吧，我现在不大在小区里走了。"),
    ("unclear", "只能这回聊天用，聊完就忘掉，这样我才愿意。"),
    ("unclear", "那就保留到这个月底，下个月就别再留着了。"),
    ("unclear", "你得加上一条，天气凉快的时候我才喜欢出去走，热天我可不爱出门。"),
    ("unclear", "我们小区那棵桂花树今年开得晚。"),
    ("unclear", "你说人老了怎么就醒得这么早，我今天五点就睁眼了。"),
    ("unclear", "昨天隔壁老陈送了我一袋红薯，还带着土呢。"),
    ("unclear", "外头是不是又下雨了？我听见窗台上噼里啪啦的。"),
    ("unclear", "我那副老花镜又找不着了，刚才明明还拿在手里的。"),
]


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("real network and credential access forbidden")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(tls_dispatch, "open_observed", forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)


class Judge:
    def __init__(self, action="confirm", raw=None, error=False):
        self.raw = json.dumps({"action": action}) if raw is None else raw
        self.error = error
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if self.error:
            raise OSError("fixture failure")
        return self.raw


def fixture(utterance, *, delivered=True, decode=True, claims=True):
    persona = PersonaProfile(user_id="confirm-test")
    source = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="source")
    source.add_turn("我喜欢晚饭后在小区慢慢散步。", "我听到了。")
    rows, rejected = extract_preference_rows([{
        "do": DO, "scope": SCOPE, "semantic_review": communication_review(),
        "quotes": [{"turn_index": 0, "text": "我喜欢晚饭后在小区慢慢散步。"}],
        "confirmation": None,
    }], source, persona)
    assert not rejected
    merge_preferences(persona, rows)
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="actual-confirm")
    ctx.add_turn("以前我会一早去买菜。", "这是之前的一件事。")
    ctx.working_memory.update(_online_decision_version=JUDGED_ONLINE,
        _safety_delivery_version="minimal_response_v1")
    pending = prepare_preference_invitation(persona, ctx, rows[0].id, actor=persona.user_id)
    if delivered:
        acknowledge_preference_invitation(persona, ctx, pending["id"], pending["question_sha256"], actor=persona.user_id)
    bind_preference_request(persona, ctx, utterance)
    segments = ([{"text": "我已经记下了。", "kind": "preference_state"}] if claims else []) + [{"text": BODY}]
    value = {"response_segments": segments, "safety_concern": "none"}
    raw = apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, utterance, persona) if decode else None
    return persona, ctx, raw, value


@pytest.mark.parametrize("action,utterance", REPLIES, ids=[f"author-T2-{i:03}" for i in range(1, 31)])
def test_author_original_invitation_isolated_request_and_fixed_label(action, utterance):
    messages = confirm_judge_messages(DO, SCOPE, utterance)
    validate_confirm_judge_messages(messages)
    assert json.loads(messages[1]["content"]) == {
        "invitation": {"do": DO, "scope": SCOPE}, "utterance": utterance}
    assert len(messages) == 2 and messages[0]["content"] == CONFIRM_JUDGE_INSTRUCTION
    assert parse_confirm_judge(Judge(action).complete(messages)) == {"action": action}


@pytest.mark.parametrize("action,utterance", REPLIES, ids=[f"T2-{i:03}" for i in range(1, 31)])
def test_independent_fixed_labels_commit_and_preserve_body(tmp_path, action, utterance):
    persona, ctx, raw, _ = fixture(utterance)
    pending = current_preference_invitation(persona, ctx)
    judge = Judge(action)
    judge_preference_confirmation(ctx, persona, utterance, client=judge)
    store = PersonaStore(tmp_path)
    reply = apply_preference_dialogue(ctx, persona, utterance, store=store, response=raw)
    assert reply.startswith(BODY + "\n")
    assert "我已经记下了。" not in reply
    assert ("好，这条记下了。" in reply) is (action == "confirm")
    expected = {"confirm": "confirmed", "decline": "declined"}.get(action, "proposed")
    assert persona.stated_preferences[0].status == expected
    if action in {"confirm", "decline"}:
        saved = store.load(persona.user_id).stated_preferences[0]
        assert saved.status == expected
        assert saved.history[-1].quote.text == utterance
        assert saved.history[-1].quote.turn_index == 1
        assert saved.history[-1].quote.session_id == ctx.session_id
    if action == "unclear":
        assert reply.endswith("这条我先不记，你想记的时候再说一声。")
    # Persistence exercises the existing admitted source scope; the author's
    # original privacy sentence remains intact in SCOPE for wire-only tests.
    assert judge.calls == [confirm_judge_messages(
        persona.stated_preferences[0].do, persona.stated_preferences[0].scope, utterance)]
    control = ctx.working_memory["online_turn_decision"]["preference_control"]
    assert control["invitation_id"] == pending["id"]
    assert control["target_id"] == pending["target_id"]
    assert control["target_sha256"] == pending["target_sha256"]
    assert control["evidence_quote"] == utterance
    for identifier in (pending["id"], pending["target_id"], pending["target_sha256"], persona.user_id):
        assert identifier not in json.dumps(judge.calls)


@pytest.mark.parametrize("raw", ["", "not json", '{"action":"confirm","extra":0}',
    '{"action":"confirm","action":"confirm"}', '{"action":true}', '{"action":"save"}', "[]", "null"])
def test_invalid_judge_is_unclear_once_without_replacing_body(tmp_path, raw):
    utterance = "对，就这样。"
    p, ctx, body, _ = fixture(utterance)
    judge = Judge(raw=raw)
    judge_preference_confirmation(ctx, p, utterance, client=judge)
    judge_preference_confirmation(ctx, p, utterance, client=judge)
    reply = apply_preference_dialogue(ctx, p, utterance, store=PersonaStore(tmp_path), response=body)
    assert reply == BODY + "\n这条我先不记，你想记的时候再说一声。"
    assert p.stated_preferences[0].status == "proposed"
    assert len(judge.calls) == 1
    assert ctx.working_memory["online_turn_decision"]["confirm_judge"]["status"] == "failed"


@pytest.mark.parametrize("judge", [None, Judge(error=True)])
def test_judge_failure_keeps_body(tmp_path, judge):
    p, ctx, raw, _ = fixture("行。")
    judge_preference_confirmation(ctx, p, "行。", client=judge)
    reply = apply_preference_dialogue(ctx, p, "行。", store=PersonaStore(tmp_path), response=raw)
    assert reply == BODY + "\n这条我先不记，你想记的时候再说一声。"
    assert p.stated_preferences[0].status == "proposed"


@pytest.mark.parametrize("action,utterance", [
    ("confirm", "就按原来的范围记住。今天还有件小事想聊。"),
    ("decline", "这条不记了，我们说眼下的事。"),
    ("defer", "晚些再决定，现在先聊聊今天。"),
    ("unclear", "我要改成仅在整理相册时适用，其他时候不照这个做。"),
])
def test_confirmation_transaction_preserves_owner_turn_through_seal_and_memory(tmp_path, action, utterance):
    # Labels are explicit fixtures, not claims about the judge's understanding.
    from aria.engine.memory.consolidator import LLMExtractor
    from aria.engine.memory.offline_queue import OfflineSessionQueue
    from tests.test_preference_source_refs import Reply

    persona, ctx, raw, _ = fixture(utterance)
    private_text = ctx.turns[0].user
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    judge_preference_confirmation(ctx, persona, utterance, client=Judge(action))
    store = PersonaStore(tmp_path)
    reply = apply_preference_dialogue(ctx, persona, utterance, store=store, response=raw)
    ctx.add_turn(utterance, reply)
    queue = OfflineSessionQueue(persona, store)
    assert queue.enqueue(ctx)["status"] == "queued"
    saved = store.load(persona.user_id)
    job, = saved.offline_sessions
    assert job.excluded_turn_indices == [0]
    assert job.turns[0].user == job.turns[0].agent == ""
    assert job.turns[1].user == utterance and job.turns[1].agent == reply
    recovered = OfflineSessionQueue._context(saved, job)
    memory = Reply(rows=[])
    extractor = LLMExtractor(memory, prefix_recovery=True)
    extractor.extract(recovered, saved)
    assert extractor.last_status == "ok"
    actual_input = memory.calls[0][1]["content"]
    assert f"[turn 1] 用户:{utterance}\n小予:{reply}" in actual_input
    assert private_text not in actual_input and "[turn 0]" not in actual_input


def test_v7_unclear_reoffer_does_not_require_erasing_the_owner_correction(tmp_path):
    from aria.engine.memory import preference_candidate, preference_recovery

    utterance = "我改成只在整理相册时这样，其他话题不这样。"
    persona, ctx, _, value = fixture(utterance, decode=False)
    value["safety_evidence"] = {"quote": "", "subject": "not_applicable",
        "time": "not_applicable", "urgency": "not_applicable"}
    raw = apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, utterance, persona)
    ctx.working_memory.update({preference_candidate.CONTEXT_KEY: preference_candidate.VERSION,
                              preference_recovery.CONTEXT_KEY: preference_recovery.VERSION})
    old = current_preference_invitation(persona, ctx)
    judge_preference_confirmation(ctx, persona, utterance, client=Judge("unclear"))
    reply = apply_preference_dialogue(ctx, persona, utterance,
                                     store=PersonaStore(tmp_path), response=raw)
    assert persona.stated_preferences[0].status == "proposed"
    ctx.add_turn(utterance, reply)
    shown = preference_recovery.complete_unclear_turn(persona, ctx)
    assert shown["status"] == "awaiting_display"
    pending = ctx.working_memory["_preference_dialogue_invitation"]
    assert pending["expires_at"] == old["expires_at"] and pending["id"] != old["id"]
    assert pending["target_sha256"] == old["target_sha256"]
    assert current_preference_invitation(persona, ctx) is None  # No new display ACK.
    assert ctx.turns[-1].user == utterance
    assert 1 not in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])


@pytest.mark.parametrize("action,code_status", [
    ("confirm", "好，这条记下了。"),
    ("decline", "好，这条不记。"),
    ("defer", "好，这条以后再说。"),
    ("unclear", "这条我先不记，你想记的时候再说一声。"),
])
@pytest.mark.parametrize("review_binding", ["current", "absent", "working_memory", "other_utterance"])
def test_composer_keeps_model_spans_only_for_bound_pending_review(tmp_path, action, code_status, review_binding):
    from aria.engine.policy.delivery_spans import model_positions
    from aria.engine.policy.semantic_delivery import digest

    utterance = "本轮供固定标签测试的回答。"
    persona, ctx, _, value = fixture(utterance, decode=False)
    mixed = "我已经记下了。我们可以慢慢聊今天的事。"
    value["response_segments"][0]["text"] = mixed
    raw = apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, utterance, persona)
    binding = (ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns), digest(utterance))
    if review_binding == "current":
        ctx._semantic_delivery_review_v2 = binding
    elif review_binding == "working_memory":
        ctx.working_memory["_semantic_delivery_review_v2"] = binding
    elif review_binding == "other_utterance":
        ctx._semantic_delivery_review_v2 = (*binding[:-1], digest("别轮输入。"))
    judge_preference_confirmation(ctx, persona, utterance, client=Judge(action))
    # This stops at composition. The engine must still execute the configured
    # review before any preserved model claim can become a delivered reply.
    composed = apply_preference_dialogue(ctx, persona, utterance,
                                        store=PersonaStore(tmp_path), response=raw)
    kept = review_binding == "current"
    assert composed == (raw if kept else BODY) + "\n" + code_status
    trace = ctx.working_memory["online_turn_decision"]
    assert trace["preference_body_composition"]["code_status"] == code_status
    assert trace["preference_body_composition"]["removed_claim_segments"] == (0 if kept else 1)
    expected_positions = ([0] * len(mixed) if kept else []) + [1] * len(BODY)
    assert model_positions(ctx, composed) == expected_positions + [None] * (len(code_status) + 1)


def test_pending_review_does_not_invent_status_for_uninvited_model_claim():
    from aria.engine.policy.semantic_delivery import digest

    utterance = "聊聊今天的事情。"
    persona, ctx, raw, _ = fixture(utterance, delivered=False)
    ctx._semantic_delivery_review_v2 = (
        ctx.user_id, ctx.session_id, ctx.scene_id, len(ctx.turns), digest(utterance))
    assert apply_preference_dialogue(ctx, persona, utterance, response=raw) is None
    assert "preference_dialogue_receipt" not in ctx.working_memory
    assert "preference_body_composition" not in ctx.working_memory["online_turn_decision"]
    assert persona.stated_preferences[0].status == "proposed"


def test_storage_failure_cannot_emit_success_and_preserves_body():
    p, ctx, raw, _ = fixture("行。")
    judge_preference_confirmation(ctx, p, "行。", client=Judge())
    class Broken:
        def save(self, value):
            raise OSError("fixture storage failure")
    reply = apply_preference_dialogue(ctx, p, "行。", store=Broken(), response=raw)
    assert reply == BODY + "\n这条没能保存，我先不记。"
    assert p.stated_preferences[0].status == "proposed"
    assert "preference_dialogue_receipt" not in ctx.working_memory


def test_changed_invitation_during_judge_never_commits(tmp_path):
    p, ctx, raw, _ = fixture("行。")
    class Changed(Judge):
        def complete(self, messages):
            ctx.working_memory["_preference_dialogue_invitation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            return super().complete(messages)
    judge_preference_confirmation(ctx, p, "行。", client=Changed())
    reply = apply_preference_dialogue(ctx, p, "行。", store=PersonaStore(tmp_path), response=raw)
    assert reply == BODY + "\n这条我先不记，你想记的时候再说一声。"
    assert p.stated_preferences[0].status == "proposed"


@pytest.mark.parametrize("mutation", ["undelivered", "expired", "input_changed", "profile_changed", "no_invitation"])
def test_no_current_delivered_bound_invitation_means_no_judge(mutation):
    p, ctx, raw, _ = fixture("行。", delivered=mutation != "undelivered")
    if mutation == "expired":
        ctx.working_memory["_preference_dialogue_invitation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    if mutation == "input_changed":
        ctx.working_memory["_preference_request_binding"]["owner_message_sha256"] = "stale"
    if mutation == "profile_changed":
        p.stated_preferences[0].do = "新的内容"
    if mutation == "no_invitation":
        ctx.working_memory.pop("_preference_dialogue_invitation")
    judge = Judge()
    judge_preference_confirmation(ctx, p, "行。", client=judge)
    assert not judge.calls


def judge_response(client, content='{"action":"confirm"}'):
    return {"id": "gen-confirmjudgefixture123456", "model": client.model, "provider": "Alibaba",
        "usage": {"cost": .0001, "prompt_tokens": 20, "completion_tokens": 4106,
                  "total_tokens": 4126, "completion_tokens_details": {"reasoning_tokens": 4096}},
        "choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def test_max_role_wire_reserved_in_existing_guard(tmp_path, monkeypatch):
    selected = make_clients()
    judge = selected.confirm_judge
    assert judge.model == selected.online.model
    assert judge.temperature == 0 and judge.max_completion_tokens == 60
    assert judge.max_retries == 0 and judge.fallback_models == []
    assert validate_qwen_max_client(judge)["billable_completion_limit"] == 4166
    calls = []
    def opened(req, **kw):
        calls.append(json.loads(req.data))
        kw["trace"].preparation_recovery.bind_request_headers(req)
        kw["trace"].preparation_recovery.before_preparation()
        return Response(judge_response(judge))
    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    with closing(guard(tmp_path)) as g:
        g.attach(judge)
        with g.scope("turn", "turn1", "session1"):
            assert judge.complete(confirm_judge_messages(DO, SCOPE, "行。")) == '{"action":"confirm"}'
        state = g.snapshot()
        row, = state["attempts"]
        assert row["role"] == "confirm_judge" and row["kind"] == "turn"
        assert row["reservation"]["wire_visible_limit"] == 60
        assert row["reservation"]["output_token_limit"] == 4166
        assert state["known_cost_usd"] == "0.0001" and state["reserved_cost_usd"] == "0"
    wire, = calls
    assert wire["max_tokens"] == 60 and wire["temperature"] == 0
    assert wire["response_format"] == confirm_judge_request_format()
    assert wire["provider"]["only"] == ["alibaba"]
    assert wire["messages"] == confirm_judge_messages(DO, SCOPE, "行。")


@pytest.mark.parametrize("failure", ["budget", "duplicate", "missing_cost"])
def test_existing_guard_blocks_unfunded_duplicate_and_unknown_cost(tmp_path, monkeypatch, failure):
    from aria.engine.policy.minimal_calls import MinimalCallStopped
    judge = make_clients().confirm_judge
    calls = []
    data = judge_response(judge)
    if failure == "missing_cost":
        del data["usage"]["cost"]
    def opened(req, **kw):
        calls.append(req)
        kw["trace"].preparation_recovery.bind_request_headers(req)
        kw["trace"].preparation_recovery.before_preparation()
        return Response(data)
    monkeypatch.setattr(tls_dispatch, "open_observed", opened)
    with closing(guard(tmp_path, budget="0.000001" if failure == "budget" else "1")) as g:
        g.attach(judge)
        with g.scope("turn", "same-turn", "session"):
            if failure == "duplicate":
                judge.complete(confirm_judge_messages(DO, SCOPE, "行。"))
            with pytest.raises(MinimalCallStopped):
                judge.complete(confirm_judge_messages(DO, SCOPE, "行。"))
        assert len(calls) == (0 if failure == "budget" else 1)
        snapshot = g.snapshot()
        assert snapshot["stopped"]
        if failure == "missing_cost":
            assert float(snapshot["reserved_cost_usd"]) > 0
            assert snapshot["known_cost_usd"] is None


def test_web_attaches_fifth_role_to_same_existing_guard(tmp_path):
    from aria.voice.minimal import MinimalWebApp
    selected = make_clients()
    with closing(guard(tmp_path)) as g:
        app = MinimalWebApp(data_dir=tmp_path / "app", clients=selected, model_call_guard=g)
        try:
            assert all(client.model_request_control is g for client in (
                selected.online, selected.memory, selected.reflection,
                selected.safety_review, selected.confirm_judge))
            assert g.snapshot()["calls_used"] == 0
        finally:
            app.close()


def test_engine_order_main_wire_has_no_control_and_only_committed_receipt(tmp_path):
    from tests.test_safety_review_v1 import Reviewer
    from tests.test_safety_review_v2 import answer
    selected = make_clients()
    p, ctx, _, value = fixture("行。今天橘子真甜。", decode=False)
    calls = []
    class Online:
        online_decision_version = JUDGED_ONLINE
        def complete(self, messages, **kwargs):
            calls.append("online")
            assert "preference_control" not in json.dumps(messages)
            return json.dumps(value, ensure_ascii=False)
    class Isolated(Judge):
        def complete(self, messages):
            assert calls == ["online"]
            calls.append("judge")
            return super().complete(messages)
    judge = Isolated()
    class Safety(Reviewer):
        def complete(self, messages):
            calls.append("safety_review")
            assert "好，这条记下了。" not in json.loads(messages[1]["content"])["sentences"][0]["text"]
            return super().complete(messages)
    reviewer = Safety(answer())
    selected = replace(selected, online=Online(), confirm_judge=judge, safety_review=reviewer)
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=selected,
        store=PersonaStore(tmp_path))
    reply = engine.process_turn(ctx, "行。今天橘子真甜。")
    assert calls == ["online", "judge", "safety_review"]
    assert BODY in reply and reply.endswith("好，这条记下了。")
    assert p.stated_preferences[0].confirmation.text == "行。今天橘子真甜。"
    assert p.stated_preferences[0].confirmation.turn_index == 1


@pytest.mark.parametrize("review_result,retained_body", [
    ("not-json", None),
    ({"conflicts": True, "delete_sentence_ids": [0, 1, 2],
      "risk_assessment": {"concern": "none", "subject": "uncertain", "time": "uncertain", "urgency": "uncertain"}}, None),
    ({"conflicts": True, "delete_sentence_ids": [0, 1],
      "risk_assessment": {"concern": "none", "subject": "uncertain", "time": "uncertain", "urgency": "uncertain"}}, "我愿意接着听。"),
])
def test_safety_review_cannot_delete_successful_confirmation_receipt(tmp_path, review_result, retained_body):
    from tests.test_safety_review_v1 import Reviewer
    from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
    selected = make_clients()
    utterance = "行。今天橘子真甜。"
    p, ctx, _, value = fixture(utterance, decode=False)
    class Online:
        online_decision_version = JUDGED_ONLINE
        def complete(self, messages, **kwargs):
            return json.dumps(value, ensure_ascii=False)
    reviewer = Reviewer(review_result)
    selected = replace(selected, online=Online(), confirm_judge=Judge(), safety_review=reviewer)
    engine = create_minimal_engine(get_scene("companion_elder"), p, clients=selected,
        store=PersonaStore(tmp_path))
    reply = engine.process_turn(ctx, utterance)
    assert len(reviewer.messages) == 1
    data = json.loads(reviewer.messages[0][1]["content"])
    assert data["candidate"] == "我已经记下了。" + BODY
    assert data["sentences"] == [{"id": 0, "text": "我已经记下了。"},
                                  {"id": 1, "text": "说起今天的事，"},
                                  {"id": 2, "text": "我愿意接着听。"}]
    assert data["safety_evidence"]["delivery_facts"]["code_status"] == "好，这条记下了。"
    assert reply.endswith("好，这条记下了。")
    if retained_body is None:
        assert DETERMINISTIC_MODEL_FAILURE_REPLY in reply
    else:
        assert retained_body in reply and DETERMINISTIC_MODEL_FAILURE_REPLY not in reply
    assert BODY not in reply and p.stated_preferences[0].status == "confirmed"
    assert ctx.working_memory["safety_review"]["code_receipts_excluded_from_review"] is True


def test_exact_judge_criterion_is_present():
    assert "老人表示同意记下（如‘对’‘就这么记’‘行’‘记下吧’），即使后面接着说别的事，也是 confirm；明确不要记是 decline；说以后再说是 defer；只有在他修改了内容或条件、否认复述、或者完全没有回应这件事时，才是 unclear。" in CONFIRM_JUDGE_INSTRUCTION


@pytest.mark.parametrize("selected", ["target", "missing", "sibling", "review_deleted_claim", "risk_inline", "risk_prelude"])
def test_failed_page_withdrawal_projects_state_and_keeps_separate_conversation(tmp_path, monkeypatch, selected):
    from aria.engine.memory.preference_status import current_preference_states
    from aria.voice.minimal import MinimalWebApp
    from tests.test_deepseek_offline_route import clients, guard as route_guard
    from tests.test_review_delivery_convergence import review_app_wires
    from tests.test_stated_preferences import _profile

    claim = "你已经撤回了。"
    body = "你想谈今天的事，我听着。"
    segments = [{"text": claim, "kind": "preference_state"}, {"text": body}]
    clean = {"risk_assessment": [], "keep_each_span": [True, True, True]}
    risk = selected.startswith("risk_")
    owner = "我摔倒了，腿疼得动不了。" if risk else "我以为刚撤回了。"
    bundle = clients()
    def first_output(client, wire):
        if client is bundle.online:
            if risk and len(wires) == 1:
                return {"safety_concern": "fall", "safety_evidence": {
                    "quote": owner, "subject": "owner", "time": "current", "urgency": "immediate"},
                    "response_segments": segments}
            return {"safety_concern": "none", "response_segments": segments}
        if client is bundle.safety_review:
            if risk and len(wires) == 2:
                return {**clean, "risk_assessment": [{"concern": "fall", "owner_source_id": "t0.s0",
                        "subject": "owner", "time": "current", "urgency": "immediate"}]}
            return clean
    wires = review_app_wires(monkeypatch, bundle, first_output)
    allowance = route_guard(tmp_path)
    app = MinimalWebApp(tmp_path / "app", clients=bundle, model_call_guard=allowance)
    try:
        profile = _profile(confirmed=False)
        profile.user_id = "fictional-chen"
        if selected == "sibling":
            sibling = _profile().stated_preferences[0].model_copy(update={"id": "pref_different"})
            profile.stated_preferences.append(sibling)
        app.store.save(profile)
        sid = app.start()["state"]["session_id"]
        profile = app._profile()
        target = profile.stated_preferences[0].id
        assert not app.preference_action("withdraw", target)["ok"]
        states = current_preference_states(profile)
        state = next(row for row in states if row["target_id"] == target)
        assert state["status"] == "proposed" and state["last_ui_result"]["persisted"] is False
        ref = state["state_ref"]
        if selected == "sibling":
            ref = next(row["state_ref"] for row in states if row["target_id"] == sibling.id)
        elif selected == "missing":
            ref = "pref:missing"
        elif selected == "review_deleted_claim":
            segments[:] = [{"text": claim + body, "kind": "preference_state"}]
            clean["keep_each_span"] = [False, True, True]
        segments[0]["state_ref"] = ref
        preludes = []
        def acknowledge(event):
            preludes.append(event)
            return True
        assert app.submit(owner, "after-failed-page", sid,
                          on_safety_prelude=acknowledge if selected == "risk_prelude" else None)["ok"]
        delivered = app._run["turns"][-1]["response"]
        assert claim not in delivered and delivered.endswith(body)
        if selected == "review_deleted_claim":
            assert delivered == body
        else:
            assert "仍待本人确认" in delivered and "不能当作该操作已生效" in delivered
        if selected == "sibling":
            assert "另一" in delivered and "已由本人确认并保存" in delivered
            assert sibling.do not in delivered and sibling.scope not in delivered
        if risk:
            if selected == "risk_inline":
                assert "先不要尝试站起" in delivered
            else:
                assert len(preludes) == 1 and preludes[0]["type"] == "safety_prelude"
                assert "先不要尝试站起" in app._run["turns"][-1]["protection_prelude"]["protection"]
                assert "先不要尝试站起" not in delivered
            assert app._run["turns"][-1]["safety_review"]["risk_assessment"]["concern"] == "fall"
        assert len(wires) == 2
        assert '"persisted":false' in wires[0]["messages"][0]["content"]
        review = json.loads(wires[1]["messages"][1]["content"])
        assert review["candidate"] == claim + body
        record, = [row for row in review["safety_evidence"]["delivery_facts"]["preference_records"]
                   if "last_ui_result" in row]
        assert record["last_ui_result"] == {"action": "withdraw", "persisted": False}
        assert "仍待本人确认" in record["description"] and "未被接受" in record["description"]
        # A later turn does not acquire a fresh failure or override its ordinary text.
        clean["keep_each_span"] = [True]
        segments[:] = [{"text": "我们慢慢聊。", "kind": "preference_state", "state_ref": state["state_ref"]}]
        assert app.submit("我想随便聊聊。", "later-talk", sid)["ok"]
        assert app._run["turns"][-1]["response"] == "我们慢慢聊。"
        assert app.store.load(profile.user_id).stated_preferences[0].status == "proposed"
    finally:
        app.close()
        allowance.close()


def state_reply(persona, ctx, utterance, ref, *, body="我们继续聊。"):
    from aria.scenes.companion_elder.delivery import guard_external_delivery_response
    value = {"safety_concern": "none", "response_segments": [
        {"text": "我刚刚已经替你完成撤回了。", "kind": "preference_state", "state_ref": ref},
        {"text": body},
    ]}
    raw = apply_single_pass_decision(json.dumps(value, ensure_ascii=False), ctx, utterance, persona)
    composed = apply_preference_dialogue(ctx, persona, utterance, response=raw)
    return guard_external_delivery_response(ctx, raw if composed is None else composed, utterance, persona)


def state_fixture(tmp_path, *, action="withdraw", persisted=True):
    from aria.engine.memory.stated_preferences import transition_preference
    from aria.engine.policy.prompt import build_messages
    from tests.test_stated_preferences import _profile
    persona = _profile(confirmed=action == "withdraw")
    store = PersonaStore(tmp_path)
    if persisted:
        store.save(persona)
    transition_preference(persona, persona.stated_preferences[0].id, action)
    if persisted:
        store.save(persona)
        persona = store.load(persona.user_id)
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", session_id="state-receipt")
    ctx.working_memory.update(_online_decision_version=JUDGED_ONLINE,
        _safety_delivery_version="minimal_response_v1")
    utterance = "核对一下这条相处说明现在的状态。"
    build_messages(get_scene("companion_elder"), persona, ctx, utterance,
                   safety_response_instruction_version="compact_response_v1")
    sources = ctx.working_memory["prompt_evidence_snapshot"]["preference_state_sources"]
    return persona, ctx, utterance, sources[0]["state_ref"] if sources else "pref:missing"


@pytest.mark.parametrize("action,status", [
    ("withdraw", "已撤回"), ("confirm", "已由本人确认并保存"), ("decline", "已拒绝"),
])
def test_v7_current_state_reaches_existing_receipt_guard(tmp_path, action, status):
    persona, ctx, utterance, ref = state_fixture(tmp_path, action=action)
    out = state_reply(persona, ctx, utterance, ref)
    assert status in out and "当前已保存记录中" in out
    assert "我刚刚已经替你完成撤回了。" not in out
    assert out.endswith("我们继续聊。")
    state = ctx.working_memory["delivery_response_guard"]["segments"][0]
    assert state["source_verified"] and state["new_action_proven"] is False
    assert "confirm_judge" not in ctx.working_memory["online_turn_decision"]
    assert "preference_dialogue_receipt" not in ctx.working_memory


@pytest.mark.parametrize("defect", [
    "unsaved", "fabricated_ref", "empty_ref", "changed_input", "changed_session", "modified_after_save",
])
def test_v7_state_composer_cannot_bypass_current_source_checks(tmp_path, defect):
    persona, ctx, utterance, ref = state_fixture(tmp_path, persisted=defect != "unsaved")
    if defect == "fabricated_ref":
        ref = "pref:never-supplied"
    elif defect == "empty_ref":
        ref = ""
    elif defect == "changed_input":
        ctx.working_memory["prompt_evidence_snapshot"]["preference_state_binding"]["utterance_sha256"] = "stale"
    elif defect == "changed_session":
        ctx.session_id = "new-session"
    elif defect == "modified_after_save":
        persona.stated_preferences[0].scope = "unsaved change"
    assert state_reply(persona, ctx, utterance, ref) == "我们继续聊。"
    assert "preference_dialogue_receipt" not in ctx.working_memory


def test_v7_native_ui_withdrawal_single_state_segment_is_delivered(tmp_path):
    from aria.engine.memory.preference_status import current_preference_states
    from aria.runtime import MinimalModelClients
    from aria.voice.minimal import MinimalWebApp
    from tests.test_minimal_web import RecordingClient
    from tests.test_stated_preferences import _profile
    clients = MinimalModelClients(*(RecordingClient(role) for role in ("online", "memory", "reflection")),
                                  model_calls_enabled=True)
    clients.online.online_decision_version = JUDGED_ONLINE
    app = MinimalWebApp(tmp_path, clients=clients, safety_delivery_version="minimal_response_v1",
                        safety_response_instruction_version="compact_response_v1")
    persona = _profile()
    persona.user_id = "fictional-chen"
    app.store.save(persona)
    try:
        started = app.start()
        assert app.preference_action("withdraw", persona.stated_preferences[0].id)["ok"]
        sources = current_preference_states(app._profile())
        clients.online.payload = {"safety_concern": "none", "response_segments": [{
            "text": "这条相处说明已经撤回了，以后按当时说的话来。",
            "kind": "preference_state", "state_ref": sources[0]["state_ref"],
        }]}
        out = app.submit("刚才我在页面撤回了这条说明，核对一下。", "withdraw-check", started["state"]["session_id"])
        assert out["ok"]
        delivered = app._run["turns"][-1]["response"]
        assert "已撤回，不再作为已确认要求使用" in delivered
        assert "可以继续聊你想说的事。" not in delivered
        assert app.store.load(persona.user_id).stated_preferences[0].status == "withdrawn"
        assert len(clients.online.calls) == 1
    finally:
        app.close()
