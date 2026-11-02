"""Preset memory rows test source capacity/storage, never semantic merge quality."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.memory_contract import FIELDS, validate_memory_response
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.stated_preferences import (
    confirmed_preference_payloads, extract_preference_rows, transition_preference,
)
from aria.engine.model.persona import (
    PersonaProfile, PreferenceScopeSelection, StatedPreference,
    PREFERENCE_SCOPE_FULL_SOURCE as FULL, PREFERENCE_SCOPE_LEGACY as LEGACY,
    PREFERENCE_SCOPE_MODEL as MODEL,
)
from tests.preference_fixtures import communication_review
from tests.test_confirm_judge_v7 import deny_network


SOURCES = [
    "聊我做饭的往事时，先让我说完，不要马上教我做法。",
    "我停一会儿时，请陪着停一会儿，别接连提问。",
    "这只限闲聊；如果我问具体怎么做，就直接讲方法。",
]
DO = "闲聊做饭往事时先听完，停顿时也停；明确问做法时直接讲方法"
SCOPE = "仅限闲聊做饭往事且未询问方法时；本人停顿时也暂停，不接连提问。明确询问做法时直接讲方法。"


def context(sources=SOURCES):
    ctx = EngineContext(user_id="continuation-fixture", scene_id="companion_elder",
                        session_id="continuation-session")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v7"
    for source in sources:
        ctx.add_turn(source, "软件替身回应。")
    return ctx


def candidate(sources=SOURCES):
    return {"do": DO, "scope": SCOPE,
            "quotes": [{"turn_index": i, "text": source} for i, source in enumerate(sources)],
            "confirmation": None, "supersedes": None, "revision_basis": None,
            "semantic_review": communication_review()}


class MemoryReply:
    role = "memory"

    def __init__(self, rows, protocol=MODEL):
        self.rows = rows
        self.preference_scope_protocol = MODEL if protocol == MODEL else None
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        data = {field: [] for field in FIELDS}
        data["stated_preferences"] = self.rows
        raw = json.dumps(data, ensure_ascii=False)
        validate_memory_response(raw)
        return raw


@pytest.mark.parametrize("protocol", [FULL, MODEL])
def test_three_turn_continuation_keeps_every_source_exception_and_owner_confirmation(tmp_path, protocol):
    ctx = context()
    profile = PersonaProfile(user_id=ctx.user_id)
    client = MemoryReply([candidate()], protocol)
    rows = LLMExtractor(client).extract(ctx, profile).stated_preferences
    assert len(rows) == 1
    row = rows[0]
    assert row.do == DO
    assert row.scope == (SCOPE if protocol == MODEL else "；".join(SOURCES))
    assert [(q.session_id, q.turn_index, q.text) for q in row.quotes] == [
        (ctx.session_id, i, text) for i, text in enumerate(SOURCES)]
    assert row.status == "proposed" and row.confirmation is None
    assert ctx.working_memory["_preference_scope_selection"] == {"protocol": protocol, "max_characters": 722}
    prompt = client.messages[0]["content"]
    assert "同一条相处说明的连续补充条件和例外应整合为一条提案" in prompt
    assert "不同或无关的相处说明仍分别提取" in prompt
    assert "每条最多引用三轮" in prompt and "722字" in prompt
    assert "每条最多引用两轮" not in prompt
    profile.stated_preferences = rows
    store = PersonaStore(tmp_path)
    store.save(profile)
    loaded = store.load(profile.user_id)
    assert loaded.stated_preferences[0] == row
    transition_preference(loaded, row.id, "confirm", at="2026-10-21T03:00:00+00:00")
    payload = confirmed_preference_payloads(loaded)[0]
    assert payload["scope"] == row.scope
    assert all(text in quote for text, quote in zip(SOURCES, payload["quotes"]))


@pytest.mark.parametrize("protocol", [FULL, MODEL])
def test_memory_request_groups_complete_requirements_before_semantic_classification(protocol):
    ctx = context()
    client = MemoryReply([], protocol)
    LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    prompt = client.messages[0]["content"]
    grouping_instructions = (
        "先判断本次是否真的提出新的交流行为要求，再形成do和scope",
        "先与旧已确认说明比对：含义及适用范围未变的复述、澄清不重复提案，也不是修改依据。",
        "同一条相处说明的连续补充条件和例外应整合为一条提案",
        "先确定共同指向，再合并完整要求，不能仅按动作或轮次拆分，也不能仅因相邻或同场景合并。",
    )
    positions = [prompt.index(instruction) for instruction in grouping_instructions]
    assert positions == sorted(positions)
    classification = "[preference_semantics_v1]按上述完整语境作semantic_review分类"
    assert positions[-1] < prompt.index(classification)
    assert "quotes保留该轮完整本人原话（每轮最多240字），范围条件和例外分属多轮时须逐轮引用" in prompt
    assert "不能只引用行为或解释所在轮次" in prompt
    assert "不同或无关的相处说明仍分别提取，按各自本人要求确定范围，不能强套旧说明的范围。" in prompt


@pytest.mark.parametrize("protocol", [FULL, MODEL])
@pytest.mark.parametrize("prefix_recovery", [False, True])
def test_confirmed_scope_context_and_clarification_instructions_reach_memory(
        protocol, prefix_recovery):
    # This checks request wiring and immutable state with an empty stub reply,
    # not whether a model correctly recognizes a clarification.
    source = context()
    profile = PersonaProfile(user_id=source.user_id)
    rows, rejected = extract_preference_rows([candidate()], source, profile,
                                             scope_protocol=protocol)
    assert not rejected
    profile.stated_preferences = rows
    transition_preference(profile, rows[0].id, "confirm", at="2026-10-22T01:00:00+00:00")
    before = profile.model_dump()
    current = context()
    current.session_id = "clarification-context"
    client = MemoryReply([], protocol)
    result = LLMExtractor(client, prefix_recovery=prefix_recovery).extract(current, profile)

    system, user = (message["content"] for message in client.messages)
    context_header = (
        "旧已确认相处说明（供比对已有要求及其完整适用范围；不是本次用户原话、"
        "新证据或新提案的确认，不能代替本次引用；只有明确修改时才供supersedes引用）：\n"
    )
    supplied = json.loads(user.split(context_header, 1)[1])
    assert supplied == [{"id": rows[0].id, "do": DO, "scope": rows[0].scope}]
    assert "只供supersedes引用" not in user
    assert "含义及适用范围未变的复述、澄清不重复提案" in system
    assert "本人明确要求助手以后改变回应方式时仍可提案" in system
    assert "必须保留未被本人明确改变的原条件、对象、时限和例外" in system
    assert "没有新要求或明确修改，该内容就不进入stated_preferences" in system
    assert "不得先把解释改写成指令，再据此认定本人提出了要求" in system
    assert "本轮未重复旧限定不代表取消限定，不得拆成范围更宽的独立说明" in system
    assert "scope_preserved须对照完整语境及旧已确认范围" in system
    assert "revision_basis和supersedes都为null，作为独立的新提案" not in system
    assert "每条最多引用三轮" in system and "722字" in system
    assert result.stated_preferences == []
    assert profile.model_dump() == before


@pytest.mark.parametrize("classification,owner_requested,scope_preserved", [
    ("fact", False, True),
    ("unclear", False, False),
    ("communication_style", False, True),
    ("communication_style", True, False),
])
def test_preset_explanation_or_missing_scope_does_not_change_confirmed_preference(
        classification, owner_requested, scope_preserved):
    # The supplied classifications check admission and source wiring only.
    # They do not measure whether a model recognises the distinction.
    previous = context()
    profile = PersonaProfile(user_id=previous.user_id)
    profile.stated_preferences, rejected = extract_preference_rows(
        [candidate()], previous, profile, scope_protocol=MODEL)
    assert not rejected
    transition_preference(profile, profile.stated_preferences[0].id, "confirm")
    before = profile.model_dump()
    current = context(["我之前说停一停，是想理顺思路，刚才那件事没有别的意思。"])
    current.session_id = "explanation-only"
    proposed = {
        **candidate(), "do": "停一停是为了理顺思路",
        "scope": "说停一停时",
        "quotes": [{"turn_index": 0, "text": current.turns[0].user}],
        "semantic_review": {
            **communication_review(), "kind": classification,
            "owner_requested": owner_requested, "scope_preserved": scope_preserved,
        },
    }
    client = MemoryReply([proposed])
    result = LLMExtractor(client).extract(current, profile)
    assert result.stated_preferences == []
    assert current.working_memory["stated_preference_extraction"]["rejected"] == [
        "preference_not_communication_style"]
    assert "[turn 0] 用户:" + current.turns[0].user in client.messages[1]["content"]
    assert SCOPE in client.messages[1]["content"]
    assert profile.model_dump() == before


def test_preset_explicit_revision_and_independent_request_retain_their_own_scopes():
    previous = context()
    profile = PersonaProfile(user_id=previous.user_id)
    profile.stated_preferences, rejected = extract_preference_rows(
        [candidate()], previous, profile, scope_protocol=MODEL)
    assert not rejected
    original = profile.stated_preferences[0]
    transition_preference(profile, original.id, "confirm")
    before = profile.model_dump()
    texts = [
        "原来只在闲聊做饭时先听我说完，现在改成闲聊旅行时也这样；问办法时还是直接讲。",
        "另外，以后核对账目时，请把金额逐项念一遍。",
    ]
    current = context(texts)
    current.session_id = "explicit-revision-and-independent"
    revised = {
        **candidate(), "do": "闲聊做饭或旅行时先听完；主动问办法时直接讲",
        "scope": "闲聊做饭或旅行时，主动询问办法除外",
        "quotes": [{"turn_index": 0, "text": texts[0]}],
        "supersedes": original.id, "revision_basis": texts[0],
    }
    independent = {
        **candidate(), "do": "核对账目时逐项念金额", "scope": "核对账目时",
        "quotes": [{"turn_index": 1, "text": texts[1]}],
    }
    client = MemoryReply([revised, independent])
    result = LLMExtractor(client).extract(current, profile)
    assert len(result.stated_preferences) == 2
    assert [row.scope for row in result.stated_preferences] == [revised["scope"], independent["scope"]]
    assert [row.supersedes for row in result.stated_preferences] == [original.id, None]
    assert all(row.status == "proposed" and row.confirmation is None for row in result.stated_preferences)
    assert [[q.text for q in row.quotes] for row in result.stated_preferences] == [[texts[0]], [texts[1]]]
    assert profile.model_dump() == before


@pytest.mark.parametrize("action", ["proposed", "withdrawn"])
def test_clarification_context_never_projects_unconfirmed_or_withdrawn_instruction(action):
    ctx = context()
    profile = PersonaProfile(user_id=ctx.user_id)
    profile.stated_preferences, rejected = extract_preference_rows(
        [candidate()], ctx, profile, scope_protocol=MODEL)
    assert not rejected
    if action == "withdrawn":
        target = profile.stated_preferences[0].id
        transition_preference(profile, target, "confirm", at="2026-10-22T01:00:00+00:00")
        transition_preference(profile, target, "withdraw", at="2026-10-22T01:01:00+00:00")
    before = profile.model_dump()
    client = MemoryReply([])
    LLMExtractor(client).extract(ctx, profile)
    assert "旧已确认相处说明" not in client.messages[1]["content"]
    assert profile.model_dump() == before
    assert not confirmed_preference_payloads(profile)


@pytest.mark.parametrize("protocol", [LEGACY, FULL, MODEL])
@pytest.mark.parametrize("count", [1, 2])
def test_one_and_two_quote_proposals_remain_compatible(protocol, count):
    sources = ["闲聊时先听我说完。", "停顿时也暂停一下。"][:count]
    raw = {**candidate(sources), "do": "闲聊时先听完，停顿时也暂停", "scope": "闲聊时"}
    ctx = context(sources)
    rows, rejected = extract_preference_rows([raw], ctx, PersonaProfile(user_id=ctx.user_id),
                                             scope_protocol=protocol)
    assert not rejected and len(rows) == 1
    assert [q.text for q in rows[0].quotes] == sources
    assert StatedPreference.model_validate_json(rows[0].model_dump_json()) == rows[0]


def test_four_quote_row_is_rejected_without_discarding_an_independent_valid_proposal():
    sources = SOURCES + ["我们核对菜单时，请把菜名一个一个念。"]
    ctx = context(sources)
    result = LLMExtractor(MemoryReply([candidate(sources), candidate()])).extract(
        ctx, PersonaProfile(user_id=ctx.user_id))
    assert len(result.stated_preferences) == 1
    assert [q.text for q in result.stated_preferences[0].quotes] == SOURCES
    assert ctx.working_memory["stated_preference_extraction"]["rejected"] == ["preference_quotes_invalid"]
    data = result.stated_preferences[0].model_dump()
    data["quotes"].append({"session_id": ctx.session_id, "turn_index": 3, "text": sources[3]})
    with pytest.raises(ValidationError):
        StatedPreference.model_validate(data)


def test_unrelated_preference_remains_a_separate_model_proposal():
    unrelated = "我们核对菜单时，请把菜名一个一个念。"
    ctx = context(SOURCES + [unrelated])
    separate = {**candidate(), "do": "核对菜单时逐个念菜名", "scope": "核对菜单时",
                "quotes": [{"turn_index": 3, "text": unrelated}]}
    result = LLMExtractor(MemoryReply([candidate(), separate])).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert len(result.stated_preferences) == 2
    assert [[q.turn_index for q in row.quotes] for row in result.stated_preferences] == [[0, 1, 2], [3]]
    assert [row.scope for row in result.stated_preferences] == [SCOPE, separate["scope"]]
    assert len({row.id for row in result.stated_preferences}) == 2
    assert all(row.status == "proposed" and row.confirmation is None for row in result.stated_preferences)


@pytest.mark.parametrize("excluded", [0, 1, 2])
def test_merged_proposal_cannot_strip_a_private_source_or_exception(excluded):
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [excluded]
    rows, rejected = extract_preference_rows([candidate()], ctx, PersonaProfile(user_id=ctx.user_id),
                                             scope_protocol=MODEL)
    assert not rows and rejected == ["preference_quote_not_verbatim_user_source"]


@pytest.mark.parametrize("protocol", [FULL, MODEL])
@pytest.mark.parametrize("count", [1, 2])
def test_legacy_481_capacity_profile_records_load_without_rewriting(tmp_path, protocol, count):
    sources = SOURCES[:count]
    ctx = context(sources)
    rows, rejected = extract_preference_rows([candidate(sources)], ctx, PersonaProfile(user_id=ctx.user_id),
                                             scope_protocol=protocol)
    assert not rejected
    data = PersonaProfile(user_id=ctx.user_id, stated_preferences=rows).model_dump()
    selection = {"protocol": protocol, "max_characters": 481}
    record = {"session_id": "historical-session", "scene_id": ctx.scene_id,
              "created_at": "2026-10-17T00:00:00+00:00", "expires_at": "2026-10-24T00:00:00+00:00",
              "source_completeness": "complete", "preference_scope": selection}
    data["offline_sessions"] = [record]
    data["offline_session_audit"] = [{**deepcopy(record), "finished_at": "2026-10-17T00:01:00+00:00",
                                      "outcome": "completed", "memory_completed": True}]
    path = tmp_path / (ctx.user_id + ".json")
    original = json.dumps(data, ensure_ascii=False)
    path.write_text(original, encoding="utf-8")
    loaded = PersonaStore(tmp_path).load(ctx.user_id)
    assert path.read_text(encoding="utf-8") == original
    assert [q.text for q in loaded.stated_preferences[0].quotes] == sources
    assert loaded.offline_sessions[0].preference_scope.model_dump() == selection
    assert loaded.offline_session_audit[0].preference_scope.model_dump() == selection
    LLMExtractor(MemoryReply([candidate(sources)], protocol)).extract(ctx, loaded)
    assert ctx.working_memory["_preference_scope_selection"] == {"protocol": protocol, "max_characters": 722}


@pytest.mark.parametrize("protocol,capacity", [
    (FULL, 480), (FULL, 482), (FULL, 723), (MODEL, 60), (MODEL, 721), (MODEL, 723),
    (LEGACY, 481), (LEGACY, 722), (MODEL, True),
])
def test_capacity_compatibility_accepts_no_other_values(protocol, capacity):
    with pytest.raises(ValidationError):
        PreferenceScopeSelection(protocol=protocol, max_characters=capacity)
