"""人物理解必须靠跨会话证据演进，而不是每次重算人格分数。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.diagnosis import PersonaHypothesisUpdater
from aria.engine.model.persona import (
    PersonHypothesis,
    PersonHypothesisEvidence,
    PersonaProfile,
)
from aria.engine.policy.prompt import select_delivery_mode


class _ChangeLLM:
    def __init__(self, relation: str = "support"):
        self.relation = relation

    def complete(self, _messages):
        return json.dumps({
            "changes": [{
                "framework": "big_five",
                "dimension": "extraversion",
                "claim": "在不熟悉的人群中更倾向先观察而不是主动表达",
                "scope": "不熟悉的群体活动",
                "relation": self.relation,
                "target_value": 0.2,
                "confidence": 0.9,
                "evidence": [
                    {
                        "turn_index": 0,
                        "evidence_quote": "第 0 轮可观察内容",
                    },
                    {
                        "turn_index": 2,
                        "evidence_quote": "第 2 轮可观察内容",
                    },
                ],
            }],
        }, ensure_ascii=False)


class _NoChangeLLM:
    def complete(self, _messages):
        return '{"changes":[]}'


class _PayloadLLM:
    def __init__(self, changes):
        self.changes = changes

    def complete(self, _messages):
        return json.dumps({"changes": self.changes}, ensure_ascii=False)


class _ResponsePreferenceLLM:
    def complete(self, _messages):
        return json.dumps({
            "changes": [{
                "framework": "response",
                "dimension": "human_connection",
                "claim": "谈到与人疏远时，具体联系熟人比泛泛安慰更容易接受",
                "scope": "真实关系连接减少时",
                "scope_tags": [
                    "social_disconnection",
                    "not_a_valid_tag",
                ],
                "relation": "new",
                "confidence": 0.82,
                "evidence": [{
                    "turn_index": 1,
                    "evidence_quote": "第 1 轮可观察内容",
                }],
            }],
        }, ensure_ascii=False)


def _ctx() -> EngineContext:
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    for index in range(3):
        ctx.add_turn(f"第 {index} 轮可观察内容", "收到")
    return ctx


def test_candidate_does_not_change_trait_until_cross_session_support():
    persona = PersonaProfile(user_id="u")
    updater = PersonaHypothesisUpdater(_ChangeLLM())

    updater.update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "candidate"
    assert persona.big_five.extraversion == 0.5

    updater.update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "working"
    assert len(hypothesis.supporting_sessions()) == 2
    assert persona.big_five.extraversion == 0.5
    assert [
        f"{item.from_status}->{item.to_status}"
        for item in hypothesis.status_history
    ] == ["none->candidate", "candidate->working"]


def test_no_change_keeps_existing_hypothesis_untouched():
    persona = PersonaProfile(user_id="u")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    snapshot = persona.person_hypotheses[0].model_dump()

    PersonaHypothesisUpdater(_NoChangeLLM()).update(persona, _ctx())
    assert persona.person_hypotheses[0].model_dump() == snapshot
    assert persona.person_model_state.sessions_observed == 2


def test_counterevidence_demotes_and_can_retire_hypothesis():
    persona = PersonaProfile(user_id="u")
    support = PersonaHypothesisUpdater(_ChangeLLM())
    support.update(persona, _ctx())
    support.update(persona, _ctx())
    assert persona.person_hypotheses[0].status == "working"

    counter = PersonaHypothesisUpdater(_ChangeLLM("counter"))
    counter.update(persona, _ctx())
    assert persona.person_hypotheses[0].counter_sessions()
    assert persona.person_hypotheses[0].status == "candidate"
    counter.update(persona, _ctx())
    counter.update(persona, _ctx())
    assert persona.person_hypotheses[0].status == "retired"
    transitions = [
        f"{item.from_status}->{item.to_status}"
        for item in persona.person_hypotheses[0].status_history
    ]
    assert "working->candidate" in transitions
    assert transitions[-1] == "candidate->retired"


def test_retired_hypothesis_reuses_lineage_but_requires_fresh_cross_session_support():
    persona = PersonaProfile(user_id="u")
    support = PersonaHypothesisUpdater(_ChangeLLM())
    for _ in range(2):
        support.update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    original_id = hypothesis.id
    counter = PersonaHypothesisUpdater(_ChangeLLM("counter"))
    for _ in range(3):
        counter.update(persona, _ctx())
    assert hypothesis.status == "retired"
    original_evidence = [item.model_dump() for item in hypothesis.evidence]

    # The model omitted id: an exact retired hypothesis must not become a second
    # object with the same hash. Old supports may not bypass cold-start gates.
    support.update(persona, _ctx())
    assert len(persona.person_hypotheses) == 1
    assert hypothesis.id == original_id
    assert hypothesis.status == "candidate"
    assert hypothesis.activation_epoch == 1
    assert len({e.session_id for e in hypothesis.activation_evidence()}) == 1
    assert [e.model_dump() for e in hypothesis.evidence[:len(original_evidence)]] == (
        original_evidence
    )

    # Epoch boundaries survive saving and loading the profile.
    restored = PersonaProfile.model_validate_json(persona.model_dump_json())
    support.update(restored, _ctx())
    renewed = restored.person_hypotheses[0]
    assert renewed.id == original_id
    assert renewed.status == "working"
    assert len({e.session_id for e in renewed.activation_evidence()}) == 2
    assert [t.to_status for t in renewed.status_history][-2:] == [
        "candidate", "working",
    ]


def test_retirement_does_not_reuse_old_days_to_make_recovery_stable():
    persona = PersonaProfile(user_id="u")
    day = datetime(2026, 7, 1, 10)
    for index in range(6):
        PersonaHypothesisUpdater(
            _ChangeLLM(), now_provider=lambda i=index: day + timedelta(days=i),
        ).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.status == "stable"
    PersonaHypothesisUpdater(_ChangeLLM("retire")).update(persona, _ctx())
    assert hypothesis.status == "retired"

    support = PersonaHypothesisUpdater(
        _ChangeLLM(), now_provider=lambda: day + timedelta(days=10),
    )
    for _ in range(6):
        support.update(persona, _ctx())
    assert hypothesis.confidence >= 0.75
    assert hypothesis.status == "working"
    for index in (11, 12):
        PersonaHypothesisUpdater(
            _ChangeLLM(), now_provider=lambda i=index: day + timedelta(days=i),
        ).update(persona, _ctx())
    assert hypothesis.status == "stable"


def test_counterevidence_records_exception_without_expanding_claim_scope_or_target():
    persona = PersonaProfile(user_id="u")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    scope_before = hypothesis.scope
    target_before = hypothesis.target_value
    PersonaHypothesisUpdater(_PayloadLLM([{
        "id": hypothesis.id,
        "framework": "big_five",
        "dimension": "extraversion",
        "claim": "熟悉家人来访时也会主动讲述经历",
        "scope": "熟悉家人来访时",
        "scope_tags": ["autonomy_threat"],
        "relation": "counter",
        "target_value": 0.95,
        "confidence": 0.8,
        "evidence": [{
            "turn_index": 0, "evidence_quote": "第 0 轮可观察内容",
        }],
    }])).update(persona, _ctx())

    assert hypothesis.scope == scope_before
    assert hypothesis.scope_tags == []
    assert hypothesis.target_value == target_before
    assert hypothesis.evidence[-1].scope == "熟悉家人来访时"
    assert hypothesis.evidence[-1].scope_tags == ["autonomy_threat"]
    assert hypothesis.evidence[-1].relation == "counter"


def test_duplicate_substrings_of_one_turn_cannot_inflate_activation():
    def updater(repeated):
        rows = [{"turn_index": 0, "evidence_quote": "第 0 轮可观察内容"}]
        if repeated:
            rows += rows * 3 + [
                {"turn_index": 0, "evidence_quote": "轮可观察内容"},
            ]
        return PersonaHypothesisUpdater(_PayloadLLM([{
            "framework": "response", "dimension": "low_pressure",
            "claim": "沟通中留白后更愿意表达", "scope": "反复被追问时",
            "scope_tags": ["autonomy_threat"], "relation": "support",
            "confidence": 0.9, "evidence": rows,
        }]))

    single = PersonaProfile(user_id="single")
    repeated = PersonaProfile(user_id="repeated")
    for _ in range(2):
        updater(False).update(single, _ctx())
        updater(True).update(repeated, _ctx())
    clean = single.person_hypotheses[0]
    attempted = repeated.person_hypotheses[0]
    assert attempted.confidence == clean.confidence
    assert len(attempted.evidence) == len(clean.evidence) == 2
    assert attempted.status == clean.status == "candidate"


def test_legacy_duplicate_evidence_rows_cannot_supply_extra_support_turns():
    hypothesis = PersonHypothesis(
        id="legacy-duplicates", framework="response", dimension="low_pressure",
        claim="留白后更愿意交流", first_observed_at="2026-07-01T10:00:00",
        confidence=0.8,
        evidence=[
            PersonHypothesisEvidence(
                at="2026-07-01T10:00:00", session_id=session,
                relation="support", summary=quote, turn_indices=[0],
            )
            for session in ("session-1", "session-2")
            for quote in ("有可观察的回应", "可观察的回应")
        ],
    )
    PersonaHypothesisUpdater._refresh_status(hypothesis)
    assert hypothesis.status == "candidate"


def test_same_session_reflection_is_idempotent():
    persona = PersonaProfile(user_id="u")
    updater = PersonaHypothesisUpdater(_ChangeLLM())
    context = _ctx()
    updater.update(persona, context)
    updater.update(persona, context)
    assert persona.person_model_state.sessions_observed == 1
    assert len(persona.person_hypotheses[0].supporting_sessions()) == 1


def test_five_sessions_on_one_day_cannot_become_stable():
    observed_at = datetime(2026, 7, 1, 10, 0)
    persona = PersonaProfile(user_id="u")
    updater = PersonaHypothesisUpdater(
        _ChangeLLM(),
        now_provider=lambda: observed_at,
    )
    for _ in range(5):
        updater.update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    assert len(hypothesis.supporting_sessions()) == 5
    assert len(hypothesis.supporting_days()) == 1
    assert hypothesis.status == "working"

    day_index = 0

    def _next_day():
        nonlocal day_index
        day_index += 1
        return observed_at + timedelta(days=day_index)

    updater = PersonaHypothesisUpdater(
        _ChangeLLM(),
        now_provider=_next_day,
    )
    updater.update(persona, _ctx())
    updater.update(persona, _ctx())
    assert len(persona.person_hypotheses[0].supporting_days()) >= 3
    assert persona.person_hypotheses[0].status == "stable"


def test_response_preference_keeps_structured_dimension_and_scope_tags():
    persona = PersonaProfile(user_id="response-u")
    PersonaHypothesisUpdater(_ResponsePreferenceLLM()).update(
        persona, _ctx()
    )

    hypothesis = persona.person_hypotheses[0]
    assert hypothesis.framework == "response"
    assert hypothesis.dimension == "human_connection"
    assert hypothesis.scope_tags == ["social_disconnection"]


def test_hallucinated_quote_cannot_become_personality_evidence():
    class Ungrounded:
        def complete(self, _messages):
            return json.dumps({
                "changes": [{
                    "framework": "big_five",
                    "dimension": "extraversion",
                    "claim": "更喜欢独处",
                    "scope": "general",
                    "relation": "new",
                    "confidence": 0.9,
                    "evidence": [{
                        "turn_index": 0,
                        "evidence_quote": "我从来不参加任何活动",
                    }],
                }],
            }, ensure_ascii=False)

    persona = PersonaProfile(user_id="ungrounded")
    PersonaHypothesisUpdater(Ungrounded()).update(persona, _ctx())
    assert not persona.person_hypotheses


def test_invalid_framework_dimension_is_rejected():
    class InvalidDimension:
        def complete(self, _messages):
            return json.dumps({
                "changes": [{
                    "framework": "big_five",
                    "dimension": "avoidant",
                    "claim": "混用了依恋和大五维度",
                    "relation": "new",
                    "confidence": 0.9,
                    "evidence": [{
                        "turn_index": 0,
                        "evidence_quote": "第 0 轮可观察内容",
                    }],
                }],
            }, ensure_ascii=False)

    persona = PersonaProfile(user_id="invalid-dimension")
    PersonaHypothesisUpdater(InvalidDimension()).update(persona, _ctx())
    assert not persona.person_hypotheses


def test_requested_id_must_match_framework_and_dimension():
    persona = PersonaProfile(user_id="mismatched-id")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    existing = persona.person_hypotheses[0]
    before = existing.model_dump()

    class MismatchedId:
        def complete(self, _messages):
            return json.dumps({
                "changes": [{
                    "id": existing.id,
                    "framework": "attachment",
                    "dimension": "avoidant",
                    "claim": "试图把同一 id 改成依恋假设",
                    "relation": "support",
                    "confidence": 0.9,
                    "evidence": [{
                        "turn_index": 0,
                        "evidence_quote": "第 0 轮可观察内容",
                    }],
                }],
            }, ensure_ascii=False)

    PersonaHypothesisUpdater(MismatchedId()).update(persona, _ctx())
    assert len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].model_dump() == before


def test_reflection_applies_at_most_one_highest_ranked_new_change():
    evidence = [{
        "turn_index": 0,
        "evidence_quote": "第 0 轮可观察内容",
    }]
    changes = [{
        "framework": "big_five",
        "dimension": "conscientiousness",
        "claim": "在日常任务中倾向提前安排",
        "scope": "日常任务",
        "scope_tags": ["task_request"],
        "relation": "new",
        "confidence": 0.8,
        "evidence": evidence,
    }, {
        "framework": "response",
        "dimension": "autonomy",
        "claim": "有选择时更愿意回应",
        "scope": "需要作决定时",
        "scope_tags": ["autonomy_threat"],
        "relation": "new",
        "confidence": 0.9,
        "evidence": evidence,
    }]

    persona = PersonaProfile(user_id="one-change")
    PersonaHypothesisUpdater(_PayloadLLM(changes)).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].framework == "big_five"
    assert persona.person_hypotheses[0].dimension == "conscientiousness"


def test_existing_counterevidence_outranks_an_adjacent_new_label():
    persona = PersonaProfile(user_id="counter-priority")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    existing = persona.person_hypotheses[0]
    evidence = [{
        "turn_index": 0,
        "evidence_quote": "第 0 轮可观察内容",
    }]
    changes = [{
        "framework": "response",
        "dimension": "autonomy",
        "claim": "保留选择权时更愿意回应",
        "scope": "需要作决定时",
        "scope_tags": ["autonomy_threat"],
        "relation": "new",
        "confidence": 0.9,
        "evidence": evidence,
    }, {
        # Deliberately omit id and use contradictory wording.  The only
        # same-dimension/scope lifecycle target can still be resolved safely.
        "framework": "big_five",
        "dimension": "extraversion",
        "claim": "在不熟悉群体中也会直接表达",
        "scope": "不熟悉的人群活动",
        "relation": "counter",
        "confidence": 0.9,
        "evidence": evidence,
    }]

    PersonaHypothesisUpdater(_PayloadLLM(changes)).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].id == existing.id
    assert len(persona.person_hypotheses[0].counter_sessions()) == 1


@pytest.mark.parametrize(
    "invalid_counter",
    [
        {"evidence": [{"turn_index": 0, "evidence_quote": "本轮未发生的反证"}]},
        {"evidence": []},
        {"id": "nonexistent-lineage"},
        {"framework": "response", "dimension": "autonomy"},
        {"relation": "no_change"},
    ],
)
def test_invalid_lifecycle_change_cannot_hide_grounded_support(invalid_counter):
    persona = PersonaProfile(user_id="ground-before-rank")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    counter = {
        "id": hypothesis.id,
        "framework": hypothesis.framework,
        "dimension": hypothesis.dimension,
        "claim": hypothesis.claim,
        "scope": hypothesis.scope,
        "relation": "counter",
        "confidence": 0.9,
        "evidence": [{"turn_index": 0, "evidence_quote": "第 0 轮可观察内容"}],
        **invalid_counter,
    }
    support = {
        "id": hypothesis.id,
        "framework": hypothesis.framework,
        "dimension": hypothesis.dimension,
        "claim": hypothesis.claim,
        "scope": hypothesis.scope,
        "relation": "support",
        "confidence": 0.9,
        "evidence": [{"turn_index": 1, "evidence_quote": "第 1 轮可观察内容"}],
    }

    PersonaHypothesisUpdater(_PayloadLLM([counter, support])).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 1
    assert len(hypothesis.supporting_sessions()) == 2
    assert not hypothesis.counter_sessions()
    assert hypothesis.status == "working"


def test_existing_support_outranks_new_adjacent_interpretation():
    persona = PersonaProfile(user_id="continuity-before-adjacent-label")
    PersonaHypothesisUpdater(_ResponsePreferenceLLM()).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    claim_before = hypothesis.claim
    evidence = [{"turn_index": 2, "evidence_quote": "第 2 轮可观察内容"}]
    adjacent = {
        "framework": "response", "dimension": "autonomy",
        "claim": "自行决定联系形式时更愿意回应",
        "scope": "真实关系连接减少时", "relation": "new",
        "confidence": 0.9, "evidence": evidence,
    }
    support = {
        "id": hypothesis.id,
        "framework": hypothesis.framework,
        "dimension": hypothesis.dimension,
        "claim": "与原有熟人建立合适的联系仍有帮助",
        "scope": hypothesis.scope,
        "scope_tags": hypothesis.scope_tags,
        "relation": "support", "confidence": 0.8, "evidence": evidence,
    }

    PersonaHypothesisUpdater(_PayloadLLM([adjacent, support])).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.claim == claim_before
    assert hypothesis.dimension == "human_connection"
    assert len(hypothesis.supporting_sessions()) == 2
    # Two separate one-turn observations still do not meet the three-turn gate.
    assert hypothesis.status == "candidate"


def test_historical_anchor_quote_cannot_be_recounted_as_current_support():
    persona = PersonaProfile(user_id="history-is-not-new-evidence")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    before = hypothesis.model_dump()
    current = EngineContext(user_id="u", scene_id="companion_elder")
    for message in ["今天想歇一会儿", "没有新的事情", "先不说了"]:
        current.add_turn(message, "收到")

    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, current)

    assert hypothesis.model_dump() == before


def test_missing_id_paraphrase_merges_only_same_scoped_hypothesis():
    first = [{
        "framework": "big_five",
        "dimension": "extraversion",
        "claim": "面对陌生的多人活动时倾向先观察再决定是否加入",
        "scope": "不熟悉且人数较多的社交活动",
        "scope_tags": ["social_disconnection"],
        "relation": "new",
        "confidence": 0.8,
        "evidence": [{
            "turn_index": 0,
            "evidence_quote": "第 0 轮可观察内容",
        }],
    }]
    paraphrase = [{
        "framework": "big_five",
        "dimension": "extraversion",
        "claim": "在不熟悉的多人场合会先观察，再决定要不要参与",
        "scope": "陌生而人数多的社交场合",
        "scope_tags": ["social_disconnection"],
        "relation": "support",
        "confidence": 0.85,
        "evidence": [{
            "turn_index": 1,
            "evidence_quote": "第 1 轮可观察内容",
        }],
    }]
    persona = PersonaProfile(user_id="paraphrase")
    PersonaHypothesisUpdater(_PayloadLLM(first)).update(persona, _ctx())
    original_id = persona.person_hypotheses[0].id

    PersonaHypothesisUpdater(_PayloadLLM(paraphrase)).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 1
    assert persona.person_hypotheses[0].id == original_id
    assert len(persona.person_hypotheses[0].supporting_sessions()) == 2


def test_missing_id_does_not_merge_different_meaning_in_same_dimension():
    first = [{
        "framework": "big_five",
        "dimension": "conscientiousness",
        "claim": "日常出门前倾向提前列出需要携带的物品",
        "scope": "日常出行准备",
        "scope_tags": ["task_request"],
        "relation": "new",
        "confidence": 0.8,
        "evidence": [{
            "turn_index": 0,
            "evidence_quote": "第 0 轮可观察内容",
        }],
    }]
    different = [{
        "framework": "big_five",
        "dimension": "conscientiousness",
        "claim": "照料植物时倾向每天记录光照和土壤变化",
        "scope": "长期照料植物",
        "scope_tags": ["task_request"],
        "relation": "new",
        "confidence": 0.8,
        "evidence": [{
            "turn_index": 2,
            "evidence_quote": "第 2 轮可观察内容",
        }],
    }]
    persona = PersonaProfile(user_id="distinct-meaning")
    PersonaHypothesisUpdater(_PayloadLLM(first)).update(persona, _ctx())
    PersonaHypothesisUpdater(_PayloadLLM(different)).update(persona, _ctx())

    assert len(persona.person_hypotheses) == 2
    assert len({item.id for item in persona.person_hypotheses}) == 2


def _lineage_seed():
    ctx = _ctx()
    ctx.turns[0].user = "你先别接着追问，让我缓一缓，我自己会往下说。"
    ctx.turns[2].user = "刚才你没有催，我现在愿意把后面的事讲了。"
    change = {
        "framework": "response", "dimension": "low_pressure",
        "claim": "谈及个人困扰时，少追问并留出整理时间更容易继续表达",
        "scope": "谈及个人困扰时", "scope_tags": ["concealed_distress"],
        "relation": "new", "confidence": 0.8,
        "evidence": [
            {"turn_index": index, "evidence_quote": ctx.turns[index].user}
            for index in [0, 2]
        ],
    }
    persona = PersonaProfile(user_id="explicit-claim-lineage")
    PersonaHypothesisUpdater(_PayloadLLM([change])).update(persona, ctx)
    return persona


def _lineage_proposal(hypothesis, relation="support"):
    ctx = _ctx()
    ctx.turns[1].user = "你留了这一会儿让我理一理，我就愿意继续说了。"
    if relation == "counter":
        ctx.turns[1].user = "还是同一件烦心事，你一直等着不问，我反而更不敢开口。"
    return ctx, {
        # The model changed only its classification. The explicit comparison is
        # the authority to use the original claim, not similarity of these words.
        "framework": "response", "dimension": "autonomy",
        "claim": "自行决定何时开口有帮助",
        "scope": "再次谈到同一件个人烦心事时",
        "scope_tags": ["autonomy_threat"],
        "relation": relation, "confidence": 0.8,
        "lineage": {
            "id": hypothesis.id,
            "claim_quote": hypothesis.claim,
            "scope_quote": hypothesis.scope or "",
            "mechanism_relation": "same", "scope_relation": "within",
        },
        "evidence": [{"turn_index": 1, "evidence_quote": ctx.turns[1].user}],
    }


def test_explicit_claim_lineage_preserves_identity_despite_adjacent_label():
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    claim, scope, tags = hypothesis.claim, hypothesis.scope, hypothesis.scope_tags[:]
    first_evidence = [item.model_dump() for item in hypothesis.evidence]
    ctx, proposal = _lineage_proposal(hypothesis)
    updater = PersonaHypothesisUpdater(_PayloadLLM([proposal]))

    updater.update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert (hypothesis.framework, hypothesis.dimension) == ("response", "low_pressure")
    assert (hypothesis.claim, hypothesis.scope, hypothesis.scope_tags) == (claim, scope, tags)
    assert len(hypothesis.supporting_sessions()) == 2
    assert hypothesis.status == "working"  # Existing two-session/three-turn gate.
    assert [item.model_dump() for item in hypothesis.evidence[:2]] == first_evidence
    assert hypothesis.evidence[-1].source == "conversation_quote:lineage:response/autonomy"
    assert hypothesis.evidence[-1].scope == proposal["scope"]
    assert hypothesis.evidence[-1].scope_tags == ["autonomy_threat"]
    assert hypothesis.evidence[-1].session_id == ctx.session_id
    assert proposal["framework"] == "response"  # Never mutate model output.
    assert proposal["dimension"] == "autonomy"


def test_explicit_lineage_keeps_real_counter_and_does_not_rename_the_claim():
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    ctx, support = _lineage_proposal(hypothesis)
    PersonaHypothesisUpdater(_PayloadLLM([support])).update(persona, ctx)
    original_claim, original_scope = hypothesis.claim, hypothesis.scope
    before = hypothesis.confidence
    ctx, counter = _lineage_proposal(hypothesis, "counter")

    PersonaHypothesisUpdater(_PayloadLLM([counter])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.confidence < before
    assert hypothesis.status == "candidate"
    assert hypothesis.evidence[-1].relation == "counter"
    assert hypothesis.evidence[-1].session_id == ctx.session_id
    assert (hypothesis.claim, hypothesis.scope) == (original_claim, original_scope)


@pytest.mark.parametrize("field,value", [
    ("mechanism_relation", "different"),
    ("mechanism_relation", "uncertain"),
    ("scope_relation", "outside"),
    ("scope_relation", "uncertain"),
    ("claim_quote", "原主张的一个相似改写，不是原文"),
    ("scope_quote", "所有情况"),
    ("claim_quote", True),
    ("id", "unknown-id"),
])
def test_explicit_lineage_requires_exact_identity_and_scope_comparison(field, value):
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    before = hypothesis.model_dump()
    ctx, proposal = _lineage_proposal(hypothesis)
    proposal["lineage"][field] = value

    PersonaHypothesisUpdater(_PayloadLLM([proposal])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.model_dump() == before


@pytest.mark.parametrize("mutation", [
    {"id": "a-different-requested-id"},
    {"relation": "new"},
    {"lineage": "not-an-object"},
    {"evidence": [{"turn_index": 1, "evidence_quote": "本轮从未出现的句子"}]},
])
def test_inconsistent_or_ungrounded_lineage_never_creates_an_adjacent_candidate(mutation):
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    before = hypothesis.model_dump()
    ctx, proposal = _lineage_proposal(hypothesis)
    proposal.update(mutation)

    PersonaHypothesisUpdater(_PayloadLLM([proposal])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.model_dump() == before


def test_no_cross_dimension_guess_when_lineage_is_absent():
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    before = hypothesis.model_dump()
    ctx, proposal = _lineage_proposal(hypothesis)
    proposal.pop("lineage")
    proposal.update({
        "relation": "new", "claim": "自己选择活动而不是被安排时更愿意参加",
        "scope": "参加社区活动的决定", "scope_tags": ["autonomy_threat"],
    })
    ctx.turns[1].user = "选什么活动我自己决定，我才愿意参加，追问多少倒没关系。"
    proposal["evidence"][0]["evidence_quote"] = ctx.turns[1].user

    PersonaHypothesisUpdater(_PayloadLLM([proposal])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 2
    assert hypothesis.model_dump() == before
    assert persona.person_hypotheses[-1].dimension == "autonomy"
    assert persona.person_hypotheses[-1].status == "candidate"


def test_explicit_lineage_cannot_transfer_a_target_value_between_frameworks():
    persona = PersonaProfile(user_id="cross-framework")
    PersonaHypothesisUpdater(_ChangeLLM()).update(persona, _ctx())
    hypothesis = persona.person_hypotheses[0]
    target_before = hypothesis.target_value
    ctx, proposal = _lineage_proposal(hypothesis)
    proposal.update({"framework": "disc", "dimension": "influence", "target_value": 0.95})
    proposal["evidence"] = [{"turn_index": 0, "evidence_quote": ctx.turns[0].user}]

    PersonaHypothesisUpdater(_PayloadLLM([proposal])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.framework == "big_five"
    assert hypothesis.dimension == "extraversion"
    assert hypothesis.target_value == target_before
    assert hypothesis.evidence[-1].source == "conversation_quote:lineage:disc/influence"


def test_explicit_lineage_retirement_reuse_still_needs_fresh_independent_support():
    persona = _lineage_seed()
    hypothesis = persona.person_hypotheses[0]
    hypothesis.status = "retired"
    hypothesis.confidence = 0.0
    ctx, proposal = _lineage_proposal(hypothesis)

    PersonaHypothesisUpdater(_PayloadLLM([proposal])).update(persona, ctx)

    assert len(persona.person_hypotheses) == 1
    assert hypothesis.activation_epoch == 1
    assert hypothesis.status == "candidate"
    assert len(hypothesis.activation_evidence()) == 1
    assert len(hypothesis.supporting_sessions()) == 2


def test_contextual_hypothesis_cannot_lock_global_delivery_mode():
    contextual = PersonHypothesis(
        id="contextual-avoidance",
        framework="attachment",
        dimension="avoidant",
        claim="被家属连续催促时更想结束交流",
        scope="被家属催促时",
        scope_tags=["autonomy_threat"],
        confidence=0.9,
        status="stable",
        first_observed_at="2026-07-01T09:00:00",
    )
    persona = PersonaProfile(
        user_id="contextual-mode",
        person_hypotheses=[contextual],
    )
    assert select_delivery_mode(persona) == "neutral"

    contextual.scope = ""
    contextual.scope_tags = ["general"]
    assert select_delivery_mode(persona) == "low_pressure"
