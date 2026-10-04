"""Post-r1 development fixtures, not unseen model quality measurements.

Frozen failed model text is replayed through the real runtime without network
calls. Transfer wording tests the finite boundary, not open-domain recall.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aria.engine.context import EngineContext, Turn
from aria.engine.model.persona import PersonaProfile
from aria.engine.orchestrator import Engine
from tests._signal_fixtures import build_simulated_acoustic, simulated_prosodic_observations
from aria.engine.policy.online_decision import apply_single_pass_decision
from aria.engine.policy.prompt import build_messages
from aria.engine.policy.reply_contract import (
    contract_prompt, quality_violations, reply_quality_contract, task_delivery_diagnostics,
)
from aria.interfaces.reasoning import ActiveObservationPlan, InterventionCandidate, WorkingHypothesis
from aria.scenes.companion_elder.observation import (
    ElderActiveObservationPolicy, _direct_situational_correction,
)
from aria.scenes.registry import get_scene


def _synthetic_fixture():
    return json.loads((Path(__file__).parent / "fixtures/followthrough_synthetic.json").read_text(encoding="utf-8"))


class _SyntheticEstablishedProfile(PersonaProfile):
    def understanding_stage(self):
        return "established"


class _Forbidden:
    def complete(self, messages):
        pytest.fail("No deep model, acoustic service or extra model review may be called")


class _Replay:
    source_id = "offline_measured_failure_replay"
    model = last_model = "fixture/frozen-r1-body"
    last_latency_ms = 0.0

    def __init__(self, outputs):
        self.outputs = outputs
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return json.dumps(self.outputs[len(self.messages) - 1], ensure_ascii=False)


def _output(response, signals=(), route="quiet_companionship"):
    return {"response": response, "state_signals": list(signals),
            "selected_intervention_id": route, "uncertainty": 0.4,
            "current_affect": {"label": "neutral", "confidence": 0.8,
                               "divergence": {"possible": False, "confidence": 0, "pattern": ""}}}


def _engine_case(case_id, outputs):
    # Original authored test inputs, never learned evidence or an effect score.
    row = _synthetic_fixture()["cases"][case_id]
    case = SimpleNamespace(history=row["history"], utterance=row["utterance"])
    persona = _SyntheticEstablishedProfile.model_validate(row["persona"])
    client = _Replay(outputs)
    engine = Engine(get_scene("companion_elder"), persona=persona, simulation_mode=True,
                    fast_llm_client=client, llm_client=_Forbidden(), state_llm_client=_Forbidden())
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder", simulation_mode=True)
    ctx.turns = [Turn(**item) for item in case.history]
    ctx.working_memory["_force_reply_model"] = "fast_integrated"
    ctx.working_memory["_benchmark_capture_reply_trace"] = True
    return engine, ctx, client, case


def _turn(engine, ctx, utterance, condition):
    turn_id = f"replay-{len(ctx.turns)}"
    ctx.working_memory["_current_turn_id"] = turn_id
    acoustic = build_simulated_acoustic(condition, turn_id=turn_id, turn_index=len(ctx.turns))
    return engine.process_turn(ctx, utterance, acoustic=acoustic,
                               multimodal=simulated_prosodic_observations(acoustic))


def _context(utterance, *, mode="observe", question=""):
    ctx = EngineContext(user_id="synthetic-followthrough", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = utterance
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode=mode, question=question)
    candidate = InterventionCandidate(id="quiet_companionship", label="fixture", goal="陪伴", guidance="自然回应")
    ctx.reasoning.candidates = [candidate]
    ctx.reasoning.selected = candidate
    return ctx


def test_all_six_measured_turns_keep_context_and_do_not_reopen_body_mind_probe():
    # Exact raw text/state from the original report, not a new model response.
    raw_probe = "好，那咱们接着聊。你刚才说暂时不想展开，是身体哪里不太舒服，还是心里有什么事？"
    outputs = [
        _output("刚坐下歇会儿挺好。我就在旁边，有事随时叫我。"),
        _output("看书挺好，慢慢看。"),
        _output("窗边安静，正好能静下心。我就在旁边，不打扰你。"),
        _output("好，那就不展开。我就在旁边陪着你。", [{
            "id": "concealed_distress", "relation": "candidate", "confidence": 0.65,
            "evidence_quote": "今天暂时不想展开说", "rationale": "回避深入交流暗示有未言明困扰",
        }]),
        _output("原来是这样，难怪刚才话少了。那现在安静些了吗？", [{
            "id": "physical_discomfort", "relation": "candidate", "confidence": 0.6,
            "evidence_quote": "刚才窗外施工很吵", "rationale": "噪音导致的不适或烦躁",
        }], route="low_pressure_clarification"),
        _output(raw_probe, [
            {"id": "physical_discomfort", "relation": "candidate", "confidence": 0.6,
             "evidence_quote": "今天暂时不想展开说", "rationale": "排除噪音后主动恢复对话"},
            {"id": "concealed_distress", "relation": "candidate", "confidence": 0.5,
             "evidence_quote": "不是我生气", "rationale": "刻意撇清情绪以保护自主"},
        ], route="low_pressure_clarification"),
    ]
    outputs[3]["current_affect"]["divergence"] = {
        "possible": True, "confidence": 0.74, "pattern": "句内轨迹末段出现 sad 线索，与表面平静不一致",
    }
    engine, ctx, client, _ = _engine_case("quiet_is_baseline", outputs)
    trajectory = SimpleNamespace(**_synthetic_fixture()["trajectory"])
    final = []
    for utterance, condition in zip(trajectory.utterances, trajectory.scenario_ids, strict=True):
        final.append(_turn(engine, ctx, utterance, condition))
    assert len(client.messages) == 6 and len(ctx.turns) == 8
    assert "？" not in final[3]  # Explicit pause is not an invitation to question.
    assert "身体" not in final[-1] and "心里" not in final[-1]
    assert ctx.reasoning.observation_plan.mode == "observe"
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert "diagnostic_probe_while_observing" in guard["violations"]
    assert ctx.working_memory["_benchmark_reply_trace"]["pre_guard_response"] == raw_probe
    # Actual prior deliveries, not future scripted text, reach subsequent turns.
    last_messages = client.messages[-1]
    assert any(item["content"] == trajectory.utterances[4] for item in last_messages)
    assert any(item["content"] == final[4] for item in last_messages)
    assert "本人已解释噪声" in last_messages[0]["content"]
    assert engine.store is None


@pytest.mark.parametrize("reply", [
    "是身体哪里不太舒服，还是心里有什么事？",
    "现在是身上有些难受，还是心情不好？",
    "是不是心里有事，才刚才不愿意细聊？",
    "你是不是有心事？",
])
def test_heldout_probe_wording_is_not_allowed_by_observe(reply):
    contract = reply_quality_contract(_context("刚才是背景音，现在可以聊了。"))
    assert "diagnostic_probe_while_observing" in quality_violations(reply, [], contract)


@pytest.mark.parametrize("reply", [
    "是继续看书还是听一段评书？",
    "刚才是外面吵，你已经说明白了。咱们接着聊。",
    "我不再问是身体不舒服还是心里有事；你刚才的解释我听明白了。",
    "可以写成：‘你是不是有心事？’",  # Quoted draft, not a probe of this user.
])
def test_probe_guard_preserves_daily_choice_acknowledgement_and_quoted_drafts(reply):
    assert quality_violations(reply, [], reply_quality_contract(_context("现在接着说。"))) == []


def test_real_current_body_fact_keeps_necessary_question_after_old_noise_explanation():
    question = "这个不舒服是刚刚才有，还是已经持续一阵了？"
    engine, ctx, client, _ = _engine_case("quiet_is_baseline", [_output(question, route="physical_clarification")])
    ctx.turns.append(Turn(user="刚才是外面吵，不是我生气。", agent="你的解释我听明白了。"))
    reply = _turn(engine, ctx, "现在我肚子不舒服。", "neutral")
    assert question in reply and len(client.messages) == 1
    assert ctx.working_memory["personality_decision"]["current_physical_evidence_qualified"]


def test_current_emergency_still_bypasses_model_despite_old_signal_correction():
    engine = Engine(get_scene("companion_elder"), persona=PersonaProfile(user_id="acute-check"),
                    fast_llm_client=_Forbidden(), llm_client=_Forbidden(), state_llm_client=_Forbidden())
    ctx = _context("我现在胸痛，喘不上气。")
    ctx.turns.append(Turn(user="刚才窗外很吵，不是我生气。", agent="明白了。"))
    assert "120" in engine.process_turn(ctx, "我现在胸痛，喘不上气。")


@pytest.mark.parametrize("utterance", [
    "帮我写一句问活动室的话，我自己看完再决定发不发，也别告诉孩子。",
    "请你起草一条消息，先给我看看。",
    "替我简单写一句，就说我愿意提供建议但不主持。",
    "麻烦你拟一下短信。",
    "别替我发，帮我写一句问活动室的话。",
    "不要告诉孩子，帮我写一句消息。",
    "昨天没想好，今天帮我写一句。",
])
def test_explicit_current_draft_request_adds_actual_text_contract(utterance):
    ctx = _context(utterance)
    contract = reply_quality_contract(ctx)
    assert contract["draft_requested"] is True
    prompt = contract_prompt(contract)
    assert "现在给出一版" in prompt and "不重复索要" in prompt
    assert "不把未授权的私人背景写给第三人" in prompt
    assert "正文必须现在" in build_messages(get_scene("companion_elder"), None, ctx, utterance)[0]["content"]


@pytest.mark.parametrize("utterance", [
    "不用帮我写一句话了。", "别替我起草消息。", "我不是让你帮我写一句。",
    "她说‘帮我写一句’，可我今天只想聊天。", "以后可能需要写消息，今天先聊聊。",
    "如果需要我会请你写一句话，今天不写。", "以后我可能请你写一句话，现在先不用。",
    "帮我写一句，算了，今天先不写了。", "上次她说请你写一句，今天只是转述。",
])
def test_draft_contract_does_not_invent_current_task_from_negation_quote_or_future(utterance):
    assert reply_quality_contract(_context(utterance))["draft_requested"] is False


@pytest.mark.parametrize("reply", [
    "好，我帮你拟一句。你看完觉得合适再发，放心，不告诉孩子。",
    "可以，我来拟一条，你先看一眼，不合适咱们再改。",
    "好的，我会写一条消息。是否发送由你决定。",
])
def test_only_meta_promise_is_recorded_as_incomplete_not_silently_given_a_draft(reply):
    contract = reply_quality_contract(_context("帮我写一句消息。"))
    assert task_delivery_diagnostics(reply, contract) == ["requested_draft_not_delivered"]


@pytest.mark.parametrize("reply", [
    "可以这样说：我只提供两张照片，不上台，行吗？",
    "‘我可以参加前半小时，后面还有安排。’ 你看合不合适。",
    "我帮你拟一句：‘请问能只提供照片，不上台吗？’",
    "我只提供照片，不上台，可以吗？你看完再决定。",
    "你想写给谁，主要问哪件事？",  # Necessary clarification, not proof of task completion.
    "我帮你写一句，先告诉我主要想表达什么？",
])
def test_natural_draft_or_content_question_is_not_a_meta_only_failure(reply):
    contract = reply_quality_contract(_context("帮我写一句消息。"))
    assert task_delivery_diagnostics(reply, contract) == []


def test_measured_empty_draft_remains_visible_and_is_not_learned_as_completed_intervention():
    raw = "好，我帮你拟一句。你看完觉得合适再发，放心，不告诉孩子。"
    engine, ctx, client, case = _engine_case("draft_not_send", [_output(raw)])
    final = _turn(engine, ctx, case.utterance, "sad")
    assert final == raw  # Do not hide this known model failure with invented private-history text.
    guard = ctx.working_memory["online_turn_decision"]["delivery_guard"]
    assert guard["activated"] is False
    assert guard["task_delivery_diagnostics"] == ["requested_draft_not_delivered"]
    assert ctx.working_memory["intervention_delivery"]["reason"] == "requested_draft_not_delivered"
    assert "不上台讲" in json.dumps(client.messages, ensure_ascii=False)
    assert len(client.messages) == 1


def test_user_requested_draft_question_is_not_a_probe_even_when_person_prefers_space():
    ctx = _context("帮我写一句问活动室的话。")
    ctx.working_memory["personality_decision"] = {"evidence": [{
        "kind": "response_hypothesis", "status": "working", "dimension": "autonomy", "claim": "喜欢留空间",
    }]}
    raw = "可以先写：‘请问能只提供两张照片，不上台吗？’ 发不发由你决定。"
    assert apply_single_pass_decision(json.dumps(_output(raw), ensure_ascii=False), ctx,
                                      ctx.working_memory["current_utterance"]) == raw


def test_current_draft_precedence_does_not_authorize_sending_or_disclosing():
    raw = "我已经告诉孩子并发送消息了。"
    engine, ctx, client, case = _engine_case("draft_not_send", [_output(raw)])
    final = _turn(engine, ctx, case.utterance, "sad")
    assert final != raw and ctx.working_memory["delivery_response_guard"]["activated"]
    assert len(client.messages) == 1


def test_measured_promotion_all_clear_is_replaced_but_not_classified_as_fraud():
    raw = "没看到要付钱，心里大概有数了。明天去听听也无妨，就当看看热闹；要是觉得不对劲，随时走人就好。"
    engine, ctx, client, case = _engine_case("uncertain_promotion", [_output(raw)])
    final = _turn(engine, ctx, case.utterance, "low_quality")
    assert final != raw and "不足以确认活动可靠" in final and "不能就此说是诈骗" in final
    assert "unverified_promotional_safety_assurance" in ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"]
    assert "fraud_assessment" not in ctx.working_memory  # Removed monitoring feature.
    assert len(client.messages) == 1 and "尚未发现收费都不是安全核验" in client.messages[0][0]["content"]


@pytest.mark.parametrize("reply", [
    "送礼品的讲座可以放心参加。", "去听听也没风险。", "这种活动肯定靠谱。",
    "我不是催你，但参加也无妨。",
])
def test_promotion_backing_boundary_transfers_beyond_original_eggs_phrase(reply):
    ctx = _context("下周有个讲座送礼品，暂时没提收费。")
    assert "unverified_promotional_safety_assurance" in quality_violations(reply, [], reply_quality_contract(ctx))


@pytest.mark.parametrize("reply", [
    "还不能说放心参加；先核对主办方和安排。",
    "没有收费不等于没风险，也不足以判断是诈骗。",
    "他们说可以放心参加，这是对方的说法，还没有核实。",
    "广告写着‘去听听也无妨’，我们可以核对一下主办方。",
    "你愿意的话可以一起看看安排，去不去由你自己决定。",
])
def test_promotion_check_preserves_uncertainty_attribution_and_user_choice(reply):
    contract = reply_quality_contract(_context("下周的活动送礼品，暂时没提收费。"))
    assert "unverified_promotional_safety_assurance" not in quality_violations(reply, [], contract)


def test_no_promotional_context_does_not_activate_all_clear_guard():
    ctx = _context("明天想去公园看看花。")
    assert not reply_quality_contract(ctx)["promotional_offer_no_safety_verification"]
    assert quality_violations("去看看也无妨。", [], reply_quality_contract(ctx)) == []


# This second measured run is also now development material. It exposed a
# different history-dependent guard failure at step 5, not a new held-out score.
def test_post_20260910_all_six_actual_bodies_do_not_inject_probe_after_explanation():
    raw_five = "原来是这样，那确实挺闹心的。既然不想说，就先静静心，我在这儿陪着你。"
    raw_six = "好，那您慢慢说。刚才施工确实挺扰人的，现在安静了正好歇口气。"
    outputs = [
        _output("好，那先歇会儿。我就在旁边，有事随时叫我。"),
        _output("看书挺好的，正好能静下心。您看的是什么书？"),
        _output("那正好，安静地待会儿也挺好。"),
        _output("好，那就不展开。您先安静待着，我在这儿陪着。", [{
            "id": "concealed_distress", "relation": "candidate", "confidence": 0.35,
            "evidence_quote": "今天暂时不想展开说", "rationale": "回避交流可能隐含未言明困扰",
        }]),
        _output(raw_five, [{
            "id": "physical_discomfort", "relation": "counter", "confidence": 0.9,
            "evidence_quote": "不是我生气", "rationale": "用户明确否认情绪困扰",
        }]),
        _output(raw_six, [
            {"id": "concealed_distress", "relation": "candidate", "confidence": 0.4,
             "evidence_quote": "不是我生气", "rationale": "否认情绪但主动解释"},
            {"id": "physical_discomfort", "relation": "counter", "confidence": 0.8,
             "evidence_quote": "刚才窗外施工很吵", "rationale": "已明确是环境噪音"},
        ]),
    ]
    outputs[3]["current_affect"]["divergence"] = {
        "possible": True, "confidence": 0.74, "pattern": "句内轨迹末段出现 sad 线索，与字面平静不一致",
    }
    engine, ctx, client, _ = _engine_case("quiet_is_baseline", outputs)
    trajectory = SimpleNamespace(**_synthetic_fixture()["trajectory"])
    replies = []
    for step, (utterance, condition) in enumerate(zip(trajectory.utterances, trajectory.scenario_ids, strict=True)):
        replies.append(_turn(engine, ctx, utterance, condition))
        if step >= 4:
            assert ctx.reasoning.observation_plan.mode == "observe"
            assert "声音解释为环境" in ctx.reasoning.observation_plan.reason
            assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["activated"] is False
    assert replies[4:] == [raw_five, raw_six]
    assert len(client.messages) == 6 and len(ctx.turns) == 8
    # The semantic interpretation is intentionally NOT repaired in the trace:
    # an assessor must still see the model wrongly used a denial as a candidate.
    signal = ctx.working_memory["online_turn_decision"]["state_signals"][0]
    assert signal["id"] == "concealed_distress" and signal["evidence_quote"] == "不是我生气"


def _residual_hypothesis():
    return [WorkingHypothesis(id="concealed_distress", label="未证实困扰", category="state", confidence=0.6)]


@pytest.mark.parametrize("utterance", [
    "刚才窗外施工很吵，不是我生气。",
    "隔壁装修太吵了，我并没有生气。",
    "麦克风收音有问题，我不是难过。",
    "刚才电视太响，不是我紧张。",
])
def test_situational_first_person_explanation_does_not_force_residual_emotion_probe(utterance):
    ctx = _context(utterance)
    ctx.emotion.label = "sad"  # Explanation does not erase or relabel this signal.
    plan = ElderActiveObservationPolicy().plan(ctx, utterance, _residual_hypothesis())
    assert plan.mode == "observe" and not plan.question
    assert ctx.emotion.label == "sad"


@pytest.mark.parametrize("utterance", [
    "刚才窗外施工很吵。", "不是我生气。", "我没事。",
    "他说刚才窗外施工很吵，不是我生气。",
    "如果刚才窗外施工很吵，不是我生气。",
    "昨天窗外施工很吵，不是我生气。",
    "窗外施工很吵，她不是生气。",
    "我不是说窗外施工很吵，不是我生气。",
    "窗外施工很吵，并不是我没有生气。",
    "‘刚才窗外施工很吵，不是我生气。’ 这是电视剧台词。",
])
def test_sound_correction_requires_current_direct_self_evidence_not_generic_denial(utterance):
    assert _direct_situational_correction(utterance) is False


@pytest.mark.parametrize("utterance", [
    "刚才窗外施工很吵，不是我生气，但我确实难过。",
    "刚才窗外施工很吵，不是我生气，是我担心明天的事。",
])
def test_correction_of_one_affect_does_not_cancel_new_distress_disclosure(utterance):
    plan = ElderActiveObservationPolicy().plan(_context(utterance), utterance, _residual_hypothesis())
    assert plan.mode == "ask"


def test_new_current_body_fact_and_explicit_request_override_situational_exception():
    utterance = "刚才窗外施工很吵，不是我生气，我现在肚子不舒服。"
    plan = ElderActiveObservationPolicy().plan(
        _context(utterance), utterance, [WorkingHypothesis(
            id="physical_discomfort", label="身体不适", category="physical", confidence=0.6,
        )], current_physical_evidence_qualified=True, selected_intervention_id="physical_clarification",
    )
    assert plan.mode == "ask" and "不舒服" in plan.question
    question = "我不急着追问，是身体有点不舒服，还是最近有件事压在心里？"
    utterance = "刚才外面施工很吵，不是我生气。我现在想说说别的烦心事。"
    plan = ElderActiveObservationPolicy().plan(
        _context(utterance), utterance, _residual_hypothesis(), current_user_reopened=True,
    )
    assert plan.mode == "ask" and plan.question == question


def test_recovery_confirmation_only_reuses_immediately_previous_direct_correction():
    ctx = _context("现在安静了，可以继续说。")
    ctx.turns = [Turn(user="刚才窗外施工很吵，不是我生气。", agent="明白了。")]
    policy = ElderActiveObservationPolicy()
    assert policy.plan(ctx, "现在安静了，可以继续说。", _residual_hypothesis()).mode == "observe"
    ctx.turns.append(Turn(user="今天想聊活动。", agent="我听着。"))
    assert policy.plan(ctx, "现在安静了，可以继续说。", _residual_hypothesis()).mode == "ask"


def test_same_turn_new_body_need_is_not_hidden_by_sound_correction_in_real_engine():
    question = "这个不舒服是刚刚才有，还是已经持续一阵了？"
    engine, ctx, client, _ = _engine_case("quiet_is_baseline", [_output(question, route="physical_clarification")])
    final = _turn(engine, ctx, "刚才窗外施工很吵，不是我生气。我现在肚子不舒服。", "neutral")
    assert question in final
    assert ctx.working_memory["personality_decision"]["current_physical_evidence_qualified"]
    assert len(client.messages) == 1


def test_same_turn_new_hydration_task_question_is_not_hidden_by_sound_correction():
    question = "你希望几点收到喝水提醒？"
    engine, ctx, client, _ = _engine_case("quiet_is_baseline", [_output(question)])
    final = _turn(engine, ctx, "刚才窗外施工很吵，不是我生气。帮我设置喝水提醒。", "neutral")
    assert "几点" in final and "喝水提醒" in final
    assert len(client.messages) == 1
    assert not ctx.working_memory.get("last_care_action")
