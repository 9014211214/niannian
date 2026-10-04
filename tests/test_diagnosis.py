"""会后人物假设更新：跨会话证据，而不是单次人格重算。"""
from __future__ import annotations

import json
import re

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import (
    PersonHypothesis,
    PersonHypothesisEvidence,
    PersonaProfile,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.prompt import select_delivery_mode
from aria.interfaces.reasoning import ObservationEvidence
from aria.scenes.registry import get_scene


class _AvoidantDiagnosisLLM:
    def complete(self, messages):
        conversation = messages[1]["content"]
        matches = re.findall(
            r"\[turn (\d+)\] 用户:(.*?)\nAria:", conversation
        )
        assert len(matches) >= 2
        return json.dumps({
            "changes": [{
                "framework": "attachment",
                "dimension": "avoidant",
                "claim": "在普遍交流中更接受低压、可留白的靠近方式",
                "scope": "",
                "scope_tags": ["general"],
                "relation": "support",
                "confidence": 0.7,
                "evidence": [
                    {
                        "turn_index": int(turn_index),
                        "evidence_quote": quote,
                    }
                    for turn_index, quote in matches[:2]
                ],
            }],
        }, ensure_ascii=False)


class _LegacyProfileLLM:
    def complete(self, _messages):
        return json.dumps({
            "disc": {"steadiness": 0.9},
            "big_five": {"extraversion": 0.1},
            "attachment_style": "avoidant",
            "confidence": 0.9,
        })


def _ctx(n: int) -> EngineContext:
    ctx = EngineContext(user_id="u", scene_id="s")
    for i in range(n):
        ctx.add_turn(f"嗯，还行。{i}", "在的。")
    return ctx


def test_diagnosis_builds_personality_and_changes_delivery():
    persona = PersonaProfile(user_id="u")
    # 冷启动:没有性格模型 → 交付永远中性
    assert select_delivery_mode(persona) == "neutral"

    PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()).update(persona, _ctx(4))

    # 单次会话只形成候选，不改人格、不改变交付。
    assert persona.attachment_style is None
    assert persona.disc.steadiness == 0.5
    assert all(item.status == "candidate" for item in persona.person_hypotheses)
    assert select_delivery_mode(persona) == "neutral"

    # 第二个独立会话重复支持后升级为工作假设；静态画像仍保持基线，
    # 运行时策略只读取当前有效假设，因此反证后可以真正撤销影响。
    PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()).update(persona, _ctx(4))
    assert persona.attachment_style is None
    assert persona.disc.steadiness == 0.5
    assert persona.big_five.extraversion == 0.5
    assert select_delivery_mode(persona) == "low_pressure"
    assert not persona.case_conceptualization.core_patterns
    assert any(item.status == "working" for item in persona.person_hypotheses)
    # 更多独立会话只增强证据，不把人永久写成一个分数。
    PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()).update(persona, _ctx(4))
    assert persona.disc.steadiness == 0.5


def test_diagnosis_skips_when_evidence_too_thin():
    persona = PersonaProfile(user_id="u")
    PersonaHypothesisUpdater(
        _AvoidantDiagnosisLLM(), min_turns=3
    ).update(persona, _ctx(1))
    assert persona.disc.steadiness == 0.5          # 证据不足,不乱改
    assert persona.attachment_style is None


def test_diagnosis_graceful_on_bad_json():
    class Bad:
        def complete(self, m):
            return "我无法输出 JSON"
    persona = PersonaProfile(user_id="u")
    updater = PersonaHypothesisUpdater(Bad())
    updater.update(persona, _ctx(4))
    assert persona.attachment_style is None        # 失败不污染
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_json_response"


def test_reflection_accepts_json_after_harmless_preamble_and_fence():
    class Preamble:
        def complete(self, _messages):
            return "说明如下：\n```json\n{\"changes\":[]}\n```"

    updater = PersonaHypothesisUpdater(Preamble())
    updater.update(PersonaProfile(user_id="u"), _ctx(4))

    assert updater.last_status == "ok"
    assert updater.last_error_type is None


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"change":[]}',
        '{"changes":{}}',
        '{"changes":["not-a-change"]}',
    ],
)
def test_reflection_rejects_valid_json_with_wrong_protocol(payload):
    class WrongProtocol:
        def complete(self, _messages):
            return payload

    persona = PersonaProfile(user_id="u")
    ctx = _ctx(4)
    updater = PersonaHypothesisUpdater(WrongProtocol())

    updater.update(persona, ctx)

    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_json_response"
    assert ctx.session_id not in persona.person_model_state.reflected_session_ids
    assert persona.person_model_state.sessions_observed == 0


def test_reflection_prompt_caps_and_deduplicates_hypothesis_changes():
    class Capture:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    llm = Capture()
    PersonaHypothesisUpdater(llm).update(
        PersonaProfile(user_id="u"), _ctx(4)
    )

    system = llm.messages[0]["content"]
    assert "只返回一个" in system
    assert "不要另建同义假设" in system
    assert "必须同时有“方式”和“结果”" in system
    assert "不同任务或情境中重复出现" in system
    assert "情境/触发 → 信念或解释 → 情绪或行为" in system
    assert "relation 必须是 counter" in system
    assert "不能把限制条件一律当成反证" in system
    assert "缺失或被反向实施时，可能支持原倾向" in system
    assert "只谈范围以外的不同情况" in system
    assert "不要先从本轮某个词重新选择人格标签" in system
    assert "沿用原 framework、dimension、id" in system
    assert "人物身份是已限定的主张与机制，不是 framework/dimension 标签" in system
    assert "逐一对照既有主张（包括 candidate），再分类" in system
    assert "support/counter/retire 还须填写 lineage" in system
    assert "无法确定同一机制时不要强行复用" in system
    assert "同一\n机制下产生相反结果" in system
    assert '"mechanism_relation":"same"' in system
    assert '"scope_relation":"within"' in system


def test_reflection_receives_compact_hypothesis_lifecycle_summary():
    persona = PersonaProfile(user_id="u")
    PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()).update(
        persona, _ctx(4)
    )

    class Capture:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    capture = Capture()
    PersonaHypothesisUpdater(capture).update(persona, _ctx(4))
    user = capture.messages[1]["content"]

    assert '"support_sessions": 1' in user
    assert '"counter_sessions": 0' in user
    assert '"status": "candidate"' in user
    assert '"recent_relations": [' in user
    assert '"relation": "support"' in user
    assert '"lineage_anchors": [' in user


def test_offline_lineage_anchors_keep_original_support_after_recent_counters():
    evidence = [
        PersonHypothesisEvidence(
            at=f"2026-08-{index + 1:02d}T10:00:00",
            session_id=f"historical-{index}",
            relation="support" if index < 2 else "counter",
            summary=f"用户原话：历史证据 {index}",
            turn_indices=[index],
            scope="过去已限定的情境",
            activation_epoch=0,
        )
        for index in range(7)
    ]
    hypothesis = PersonHypothesis(
        id="existing-lineage",
        framework="response",
        dimension="direct_options",
        claim="选项少且清楚时更容易决定",
        scope="需要作决定时",
        first_observed_at="2026-08-01T10:00:00",
        evidence=evidence,
    )
    persona = PersonaProfile(user_id="u", person_hypotheses=[hypothesis])
    before = hypothesis.model_dump()

    class Capture:
        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    capture = Capture()
    PersonaHypothesisUpdater(capture).update(persona, _ctx(4))
    payload = capture.messages[1]["content"]
    existing = json.loads(
        payload.split("当前人物假设：\n", 1)[1].split("\n\n本次会话：", 1)[0]
    )[0]

    assert len(existing["recent_relations"]) == 4
    assert all(row["relation"] == "counter" for row in existing["recent_relations"])
    assert [row["purpose"] for row in existing["lineage_anchors"]] == [
        "earliest_retained_support", "latest_support", "latest_counter",
    ]
    assert [row["session_id"] for row in existing["lineage_anchors"]] == [
        "historical-0", "historical-1", "historical-6",
    ]
    assert all(row["activation_epoch"] == 0 for row in existing["lineage_anchors"])
    assert "不能把历史本身当作本次新证据" in capture.messages[0]["content"]
    assert hypothesis.model_dump() == before


def test_reflection_can_reconsider_a_retired_hypothesis_from_its_audit_trail():
    persona = PersonaProfile(user_id="u")
    PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()).update(persona, _ctx(4))
    hypothesis = persona.person_hypotheses[0]
    hypothesis.status = "retired"
    hypothesis.confidence = 0.0

    class Capture:
        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    capture = Capture()
    PersonaHypothesisUpdater(capture).update(persona, _ctx(4))
    system, user = [message["content"] for message in capture.messages]
    assert hypothesis.id in user
    assert '"status": "retired"' in user
    assert '"activation_epoch": 0' in user
    assert '"summary":' in user
    assert "引用原 id 返回 support" in system
    assert hypothesis.status == "retired"


def test_offline_reflection_does_not_receive_online_candidates():
    class Capture:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    ctx = _ctx(3)
    ctx.working_memory["online_decision_history"] = [{
        "status": "parsed",
        "turn_index": 1,
        "state_signals": [{
            "id": "concealed_distress",
            "relation": "candidate",
            "confidence": 0.66,
            "evidence_quote": "嗯，还行。1",
            "evidence_turn_index": 1,
        }],
        "selected_intervention_id": "quiet_companionship",
        "uncertainty": 0.4,
    }]
    capture = Capture()
    PersonaHypothesisUpdater(capture).update(
        PersonaProfile(user_id="u"), ctx
    )
    system = capture.messages[0]["content"]
    user = capture.messages[1]["content"]
    assert "在线状态、情绪候选和干预选择不会提供给会后反思" in system
    assert "在线状态、情绪候选和干预选择不会提供" in user
    assert "concealed_distress" not in user
    assert "quiet_companionship" not in user


def test_reflection_only_receives_real_nonline_observations():
    class Capture:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return '{"changes":[]}'

    ctx = _ctx(3)
    ctx.reasoning.evidence.extend([
        ObservationEvidence(
            id="semantic-self",
            modality="semantic",
            signal="concealed_distress",
            value="ONLINE_SEMANTIC_SELF_SIGNAL",
            source="online_single_pass_decision",
            confidence=0.9,
        ),
        ObservationEvidence(
            id="affect-self",
            modality="affect",
            signal="emotion",
            value="ONLINE_AFFECT_SELF_SIGNAL",
            source="online_single_pass_decision",
            confidence=0.9,
        ),
        ObservationEvidence(
            id="visual-real",
            modality="visual",
            signal="posture_engagement",
            value="REAL_VISUAL_SIGNAL",
            source="camera_opt_in",
            confidence=0.9,
            quality=0.9,
        ),
    ])
    capture = Capture()
    PersonaHypothesisUpdater(capture).update(
        PersonaProfile(user_id="u"), ctx
    )
    user = capture.messages[1]["content"]
    assert "REAL_VISUAL_SIGNAL" in user
    assert "ONLINE_SEMANTIC_SELF_SIGNAL" not in user
    assert "ONLINE_AFFECT_SELF_SIGNAL" not in user


def test_legacy_full_profile_without_grounded_changes_is_rejected():
    persona = PersonaProfile(user_id="legacy")
    ctx = _ctx(3)
    updater = PersonaHypothesisUpdater(_LegacyProfileLLM())
    updater.update(persona, ctx)
    assert not persona.person_hypotheses
    assert updater.last_status == "failed"
    assert updater.last_error_type == "invalid_json_response"
    assert ctx.session_id not in persona.person_model_state.reflected_session_ids


def test_engine_diagnoses_on_session_close_and_persists(tmp_path):
    """端到端:单次只存候选；跨会话重复后才让人物假设影响交付。"""
    store = PersonaStore(base_dir=tmp_path)
    persona = PersonaProfile(user_id="u1")
    engine = Engine(
        get_scene("companion_elder"), persona=persona, store=store,
        hypothesis_updater=PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()),
    )
    ctx = EngineContext(user_id="u1", scene_id="companion_elder")
    for _ in range(4):
        engine.process_turn(ctx, "嗯，没事，还行。")
    engine.finalize_session(ctx)

    reloaded = store.load("u1")
    assert reloaded is not None and reloaded.attachment_style is None
    assert reloaded.person_hypotheses

    engine2 = Engine(
        get_scene("companion_elder"), persona=reloaded, store=store,
        hypothesis_updater=PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()),
    )
    ctx2 = EngineContext(user_id="u1", scene_id="companion_elder")
    for _ in range(4):
        engine2.process_turn(ctx2, "我还是更愿意自己消化，不想被连续追问。")
    engine2.finalize_session(ctx2)
    reloaded2 = store.load("u1")
    assert reloaded2 is not None
    assert reloaded2.attachment_style is None
    assert any(
        item.framework == "attachment"
        and item.dimension == "avoidant"
        and item.status == "working"
        for item in reloaded2.person_hypotheses
    )
    assert reloaded2.delivery_state.candidate == "low_pressure"

    engine3 = Engine(
        get_scene("companion_elder"), persona=reloaded2, store=store,
        hypothesis_updater=PersonaHypothesisUpdater(_AvoidantDiagnosisLLM()),
    )
    ctx3 = EngineContext(user_id="u1", scene_id="companion_elder")
    for _ in range(4):
        engine3.process_turn(ctx3, "我更喜欢你别催，给我一点空间。")
    engine3.finalize_session(ctx3)
    assert select_delivery_mode(store.load("u1")) == "low_pressure"
