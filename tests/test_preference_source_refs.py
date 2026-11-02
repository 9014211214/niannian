"""Mechanical provenance fixtures; prescribed labels are not semantic scores."""
from copy import deepcopy
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.memory_contract import FIELDS, memory_response_schema, validate_memory_response
from aria.engine.memory.preference_sources import LEGACY, SOURCE_REFS, PreferenceSources
from aria.engine.memory.stated_preferences import extract_preference_rows, merge_preferences, transition_preference
from aria.engine.model.persona import PersonaProfile, PREFERENCE_SCOPE_MODEL
from aria.engine.policy.memory_structured import memory_request_format, validate_memory_structured_client
from tests.preference_fixtures import communication_review


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network or credentials forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("aria._env.load_local_env", forbidden)
    monkeypatch.setattr("aria.runtime.openrouter_api_key", forbidden)
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", forbidden)


def context():
    ctx = EngineContext(user_id="source_owner", scene_id="companion_elder", session_id="source_session")
    ctx.add_turn("  聊家常时，请先听我说完。\n再回应我。  ", "助理内容不能作本人出处。")
    ctx.add_turn("今天我整理了阳台。", "我听着。")
    ctx.add_turn("只在闲聊的时候这样。", "明白。")
    return ctx


def candidate(**changes):
    return {"do": "闲聊时先听本人说完再回应", "scope": "仅限闲聊", "source_turn_indices": [0, 2],
            "confirmation": None, "supersedes": None, "revision_turn_index": None,
            "semantic_review": communication_review(), **changes}


def snapshot(ctx, omitted=()):
    return PreferenceSources(ctx.user_id, ctx.session_id, tuple(
        (index, turn.user, turn.agent) for index, turn in enumerate(ctx.turns) if index not in omitted))


def extract(rows=None, *, ctx=None, persona=None, sources=None):
    ctx = ctx or context()
    return extract_preference_rows(rows if rows is not None else [candidate()], ctx,
        persona or PersonaProfile(user_id=ctx.user_id), scope_protocol=PREFERENCE_SCOPE_MODEL,
        source_protocol=SOURCE_REFS, source_snapshot=sources if sources is not None else snapshot(ctx))


def embedded_json_objects(text):
    """Find actual format examples without matching a prescribed prompt sentence."""
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


class Reply:
    role = "memory"
    preference_source_protocol = SOURCE_REFS
    preference_scope_protocol = PREFERENCE_SCOPE_MODEL
    last_finish_reason = "stop"

    def __init__(self, rows=None, privacy=None, mutate=None):
        self.payload = {field: [] for field in FIELDS}
        self.payload["stated_preferences"] = rows if rows is not None else [candidate()]
        self.payload["cognitive_privacy"] = privacy or []
        self.calls = []
        self.mutate = mutate

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        if self.mutate:
            self.mutate()
        raw = json.dumps(self.payload, ensure_ascii=False)
        validate_memory_response(raw, source_protocol=self.preference_source_protocol)
        return raw


def test_actual_request_sources_are_stored_completely_and_only_proposed():
    ctx, client = context(), Reply()
    result = LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    row, = result.stated_preferences
    assert [(q.turn_index, q.text) for q in row.quotes] == [(i, ctx.turns[i].user) for i in (0, 2)]
    assert row.do == candidate()["do"] and row.scope == candidate()["scope"]
    assert row.status == "proposed" and row.confirmation is None
    system, actual = client.calls[0]
    for q in row.quotes:
        assert f"[turn {q.turn_index}] 用户:{q.text}\n" in actual["content"]
        assert q.session_id == ctx.session_id
    example, _ = json.JSONDecoder().raw_decode(system["content"].split("输出 JSON：", 1)[1])
    assert "source_turn_indices" in example["stated_preferences"][0]
    assert "quotes" not in example["stated_preferences"][0]
    assert "revision_turn_index" in example["stated_preferences"][0]
    assert len(client.calls) == 1
    assert ctx.working_memory["stated_preference_extraction"]["source_protocol"] == SOURCE_REFS


@pytest.mark.parametrize("indices", [None, [], [0, 1, 2, 0], [0, 0], [-1], [3], [True], [0.0], ["0"], [{}]])
def test_invalid_sources_reject_whole_candidate(indices):
    rows, rejected = extract([candidate(source_turn_indices=indices)])
    assert rows == [] and rejected


@pytest.mark.parametrize("omitted", [(0,), (2,)])
def test_unexposed_source_rejects_whole_candidate_without_partial_evidence(omitted):
    ctx = context()
    rows, rejected = extract(ctx=ctx, sources=snapshot(ctx, omitted))
    assert rows == [] and rejected == ["preference_source_reference_invalid"]


@pytest.mark.parametrize("change", ["user", "persona", "session", "text", "appended"])
def test_source_snapshot_cannot_follow_mutated_or_other_context(change):
    ctx = context()
    sources = snapshot(ctx)
    profile = PersonaProfile(user_id=ctx.user_id)
    row = candidate()
    if change == "user": ctx.user_id = "other"
    elif change == "persona": profile.user_id = "other"
    elif change == "session": ctx.session_id = "another_session"
    elif change == "text": ctx.turns[0].user += "后来改过。"
    else:
        ctx.add_turn("后来新增的要求。", "好。")
        row["source_turn_indices"] = [3]
    assert extract([row], ctx=ctx, persona=profile, sources=sources)[0] == []


@pytest.mark.parametrize("stage", ["source", "scope"])
@pytest.mark.parametrize("change", ["user", "persona", "session", "text", "privacy", "removed"])
def test_sources_and_identity_are_rechecked_after_proposal_validation(monkeypatch, stage, change):
    from aria.engine.memory import stated_preferences as preferences
    ctx, profile = context(), PersonaProfile(user_id="source_owner")
    sources = snapshot(ctx)
    original = [ctx.turns[i].user for i in (0, 2)]
    safety = preferences._model_source_safety
    inspected = []

    def mutate_after_resolution(text):
        inspected.append(text)
        if len(inspected) == (1 if stage == "source" else 3):
            if change == "user": ctx.user_id = "later_owner"
            elif change == "persona": profile.user_id = "later_owner"
            elif change == "session": ctx.session_id = "later_session"
            elif change == "text": ctx.turns[2].user = "模型没有收到的新文字。"
            elif change == "privacy": ctx.working_memory["_sensitive_memory_excluded_turns"] = [2]
            else: ctx.turns.clear()
        return safety(text)

    monkeypatch.setattr(preferences, "_model_source_safety", mutate_after_resolution)
    accepted, rejected = extract(ctx=ctx, persona=profile, sources=sources)
    assert not accepted and rejected == ["preference_source_reference_changed"]
    assert inspected[:2] == original  # Safety also sees only the actual sent sources.


def test_preexcluded_sources_keep_original_sparse_indices_and_cannot_reenter():
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    client = Reply(mutate=lambda: ctx.working_memory.update(_sensitive_memory_excluded_turns=[]))
    result = LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert [q.turn_index for q in result.stated_preferences[0].quotes] == [0, 2]
    assert "[turn 1] 用户:" not in client.calls[0][1]["content"]
    assert "[turn 2] 用户:" in client.calls[0][1]["content"]
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [1]
    client = Reply([candidate(source_turn_indices=[1])], mutate=lambda: ctx.working_memory.update(_sensitive_memory_excluded_turns=[]))
    assert not LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id)).stated_preferences


def test_same_response_privacy_exclusion_applies_before_source_resolution():
    ctx = context()
    client = Reply(privacy=[{"turn_index": 2, "evidence_quote": ctx.turns[2].user}])
    result = LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not result.stated_preferences
    assert 2 in ctx.working_memory["_sensitive_memory_excluded_turns"]


@pytest.mark.parametrize("kind,limit", [("source", 240), ("do", 60), ("scope", 722)])
def test_existing_capacity_boundaries_are_not_truncated(kind, limit):
    for count, accepted in ((limit, True), (limit + 1, False)):
        ctx, row = context(), candidate(source_turn_indices=[0])
        if kind == "source": ctx.turns[0].user = "字" * count
        else: row[kind] = "字" * count
        rows, rejected = extract([row], ctx=ctx)
        assert bool(rows) is accepted
        if accepted:
            value = rows[0].quotes[0].text if kind == "source" else getattr(rows[0], kind)
            assert len(value) == count
        else: assert rejected


@pytest.mark.parametrize("extra", [{"quotes": [{"turn_index": 0, "text": "伪文"}]}, {"revision_basis": "伪文"},
                                   {"source_protocol": SOURCE_REFS}, {"status": "confirmed"}])
def test_new_contract_never_repairs_legacy_quotes_or_admits_extra_claims(extra):
    assert extract([candidate(**extra)])[0] == []


def test_legacy_remains_strict_and_cannot_be_enabled_by_model_fields():
    ctx, profile = context(), PersonaProfile(user_id="source_owner")
    old = candidate(quotes=[{"turn_index": 0, "text": "改写的原话"}], revision_basis=None)
    assert extract_preference_rows([old], ctx, profile)[1] == ["preference_quote_not_verbatim_user_source"]
    assert not extract_preference_rows([candidate()], ctx, profile)[0]
    old["quotes"][0]["text"] = ctx.turns[0].user
    assert extract_preference_rows([old], ctx, profile)[0]


@pytest.mark.parametrize("protocol,role", [("unknown", "memory"), (SOURCE_REFS, "reflection")])
def test_untrusted_protocol_or_role_fails_before_call(protocol, role):
    client, ctx = Reply(), context()
    client.preference_source_protocol, client.role = protocol, role
    with pytest.raises(ValueError):
        LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not client.calls


@pytest.mark.parametrize("revision", [None, True, "0", -1, 3, 1])
def test_invalid_or_unselected_revision_removes_only_supersedes(revision):
    ctx, profile = context(), PersonaProfile(user_id="source_owner")
    old = extract([candidate()], ctx=ctx, persona=profile)[0][0]
    merge_preferences(profile, [old])
    transition_preference(profile, old.id, "confirm")
    row = candidate(do="闲聊时等本人讲完", supersedes=old.id, revision_turn_index=revision)
    proposals, rejected = extract([row], ctx=ctx, persona=profile)
    assert not rejected and proposals[0].supersedes is None
    assert profile.stated_preferences[0].status == "confirmed"


def test_valid_revision_and_withdrawal_still_require_owner_actions():
    ctx, profile = context(), PersonaProfile(user_id="source_owner")
    old = extract(ctx=ctx, persona=profile)[0][0]
    merge_preferences(profile, [old])
    transition_preference(profile, old.id, "confirm")
    row = candidate(do="闲聊时等本人讲完", supersedes=old.id, revision_turn_index=0)
    proposal, = extract([row], ctx=ctx, persona=profile)[0]
    merge_preferences(profile, [proposal])
    assert [p.status for p in profile.stated_preferences] == ["confirmed", "proposed"]
    transition_preference(profile, proposal.id, "confirm")
    assert [p.status for p in profile.stated_preferences] == ["superseded", "confirmed"]
    transition_preference(profile, proposal.id, "withdraw")
    assert extract([row], ctx=ctx, persona=profile)[0] == []


def test_bad_reference_does_not_discard_valid_sibling_after_native_schema_reception():
    ctx, client = context(), Reply([candidate(source_turn_indices=[True]), candidate()])
    result = LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert len(result.stated_preferences) == 1
    assert ctx.working_memory["stated_preference_extraction"]["rejected"] == ["preference_source_reference_invalid"]


def test_new_schema_and_release_selection_are_explicit_and_old_schema_stays_legacy():
    from tests.test_deepseek_release import clients
    selected = clients().memory
    assert selected.preference_source_protocol == SOURCE_REFS
    schema = memory_request_format(selected)
    assert schema["json_schema"]["name"] == "aria_memory_source_refs_schema_v1"
    properties = schema["json_schema"]["schema"]["$defs"]["_PreferenceRefs"]["properties"]
    assert "quotes" not in properties and "revision_basis" not in properties
    assert properties["source_turn_indices"]["maxItems"] == 3
    assert validate_memory_structured_client(selected)["preference_source_protocol"] == SOURCE_REFS
    old = memory_response_schema()
    assert "quotes" in old["$defs"]["_Preference"]["properties"]
    selected.preference_source_protocol = LEGACY
    assert memory_request_format(selected) == memory_request_format()


@pytest.mark.parametrize("review", [None, communication_review(owner_requested=False),
                                  communication_review(scope_preserved=False)])
def test_source_binding_never_substitutes_for_model_review(review):
    assert extract([candidate(semantic_review=review)])[0] == []


def test_missing_request_snapshot_is_not_reconstructed_from_current_context():
    ctx = context()
    rows, rejected = extract_preference_rows([candidate()], ctx, PersonaProfile(user_id=ctx.user_id),
        scope_protocol=PREFERENCE_SCOPE_MODEL, source_protocol=SOURCE_REFS)
    assert not rows and rejected == ["preference_source_contract_invalid"]


def test_three_sources_and_no_source_normalization():
    ctx = context()
    ctx.turns[1].user = "好，先这样。\r\n🙂 e\u0301"
    rows, rejected = extract([candidate(source_turn_indices=[0, 1, 2])], ctx=ctx)
    assert not rejected and len(rows[0].quotes) == 3
    assert [q.text for q in rows[0].quotes] == [t.user for t in ctx.turns]


@pytest.mark.parametrize("excluded", [(), (1,)])
def test_native_client_wire_guard_and_provenance_are_the_same_contract(tmp_path, monkeypatch, excluded):
    import aria
    from tests.test_deepseek_offline_route import clients, guard, response, dispatch
    assert Path(aria.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]
    client = clients().memory
    assert client.preference_source_protocol == SOURCE_REFS
    assert client.preference_scope_protocol == PREFERENCE_SCOPE_MODEL
    payload = {field: [] for field in FIELDS}
    payload["stated_preferences"] = [candidate(source_turn_indices=[True]), candidate()]
    reply = response(client)
    reply["choices"][0]["message"]["content"] = json.dumps(payload, ensure_ascii=False)
    calls = dispatch(monkeypatch, reply)
    ctx = context()
    ctx.working_memory["_sensitive_memory_excluded_turns"] = list(excluded)
    profile = PersonaProfile(user_id=ctx.user_id)
    before_profile = profile.model_dump(mode="json")
    with closing(guard(tmp_path)) as guarded:
        guarded.attach(client)
        with guarded.scope("review", "software-memory", ctx.session_id):
            result = LLMExtractor(client, prefix_recovery=True).extract(ctx, profile)
        attempt, = guarded.snapshot()["attempts"]
        assert attempt["status"] == "received"
        assert attempt["expected_provider"] == attempt["returned_provider"] == "Together"
    assert len(calls) == len(result.stated_preferences) == 1
    wire, timeout = calls[0]
    assert timeout == 90 and wire["stream"] is True
    assert wire["provider"]["require_parameters"] is True
    assert wire["response_format"] == memory_request_format(client)
    assert wire["response_format"]["json_schema"]["strict"] is True
    # Reviewed current typed prompt; this checks actual wire selection only,
    # never the semantic correctness of a model's privacy or scope classification.
    assert "source_id" in wire["messages"][0]["content"]
    assert wire["response_format"]["json_schema"]["name"] == "aria_memory_fact_sources_v3"
    assert list(wire["response_format"]["json_schema"]["schema"]["properties"]) == list(FIELDS)
    assert not any({"episodes", "relationships", "stated_preferences"} <= value.keys()
                   for value in embedded_json_objects(wire["messages"][0]["content"]))
    wire_turns = json.loads(wire["messages"][1]["content"])
    assert [(r["turn_index"], r["本人原话"], r["助手回复"]) for r in wire_turns] == [
        (i, turn.user, turn.agent) for i, turn in enumerate(ctx.turns) if i not in excluded]
    row, = result.stated_preferences
    assert [(q.turn_index, q.text) for q in row.quotes] == [(i, ctx.turns[i].user) for i in (0, 2)]
    assert all(q.session_id == ctx.session_id for q in row.quotes)
    assert (row.do, row.scope) == (candidate()["do"], candidate()["scope"])
    assert row.status == "proposed" and row.confirmation is None and row.supersedes is None
    assert row.semantic_review.no_permission_claim is True
    assert profile.model_dump(mode="json") == before_profile
    print(json.dumps({"aria_file": str(Path(aria.__file__).resolve()),
                      "cwd": str(Path.cwd()), "fixture_wire_bytes": len(json.dumps(wire, ensure_ascii=False).encode()),
                      "fixture_system_utf8_bytes": len(wire["messages"][0]["content"].encode()),
                      "model_calls": 0, "transport": "local_python_stub"}))


def test_source_protocol_changes_after_guard_attachment_cannot_send(tmp_path, monkeypatch):
    from tests.test_memory_structured_candidate import clients, guard, response, invoke
    from tests.test_reserved_calls import stub
    from aria.engine.policy.minimal_calls import MinimalCallStopped
    client = clients().memory
    calls = stub(monkeypatch, response(client))
    with closing(guard(tmp_path / "calls.sqlite3", client)) as guarded:
        guarded.attach(client)
        client.preference_source_protocol = SOURCE_REFS
        with pytest.raises(MinimalCallStopped, match="configuration_changed"):
            invoke(guarded, client)
    assert calls == []


def test_current_memory_fact_ids_bind_literal_owner_sources_through_native_wire(tmp_path, monkeypatch):
    from tests.test_deepseek_offline_route import clients, guard, response, dispatch
    ctx = EngineContext(user_id="owner", session_id="fact-session", scene_id="companion_elder")
    texts = [
        "有人说我搬走了。这不对，我还住在旧街。",
        "姑姑方桐来家里喝茶。",
        "我女儿肖梅在车站。",
        "我昨天走了很久" * 45 + "。后来我搬了花盆。",
        "我忘了回家的路。",
        "存款的事我自己知道。",
        "不应出现在请求中的轮次。",
    ]
    for text in texts:
        ctx.add_turn(text, "这是助手说的话，不能当作本人原话。")
    ctx.working_memory["_sensitive_memory_excluded_turns"] = [6]
    profile, client = PersonaProfile(user_id=ctx.user_id), clients().memory
    payload = {field: [] for field in FIELDS}
    episode = lambda source_id, **kw: {"source_id": source_id, "keywords": [],
                                      "referenceable": True, **kw}
    payload["episodes"] = [
        episode("t0", keywords=["旧街", "不存在的词", ""]), episode("t3.s2"), episode("t4"), episode("t5"),
        episode(True), episode(-1), episode(99), episode("t1", evidence_quote="伪造的原话"),
        None, "不是事实对象", ["失实记录"],
    ]
    payload["relationships"] = [
        {"name": "方桐", "relation": "姑姑", "source_ids": ["t1", "t2"], "referenceable": True},
        {"name": "方桐", "relation": "女儿", "source_ids": ["t1", "t2"], "referenceable": True},
        {"name": "编造名字", "relation": "", "source_ids": ["t0"], "referenceable": True},
        None, False,
    ]
    payload["cognitive_privacy"] = [{"turn_index": 4, "evidence_quote": texts[4]}]
    data = response(client)
    data["choices"][0]["message"]["content"] = json.dumps(payload, ensure_ascii=False)
    calls = dispatch(monkeypatch, data)
    with closing(guard(tmp_path)) as guarded:
        guarded.attach(client)
        with guarded.scope("review", "fact-wire", ctx.session_id):
            result = LLMExtractor(client, prefix_recovery=True).extract(ctx, profile)
        receipt, = guarded.snapshot()["attempts"]
        assert receipt["status"] == "received"
    wire, _ = calls[0]
    assert wire["stream"] is True and wire["response_format"] == memory_request_format(client)
    schema = wire["response_format"]["json_schema"]["schema"]["$defs"]
    assert set(schema["_SelectedEpisode"]["properties"]) == {"source_id", "keywords", "referenceable", "preference_source_id"}
    assert set(schema["_SelectedRelationship"]["properties"]) == {"name", "relation", "source_ids", "referenceable", "preference_source_id"}
    source_rows = json.loads(wire["messages"][1]["content"])
    assert [r["本人原话"] for r in source_rows] == texts[:6]
    assert source_rows[0]["事实出处"] == {"t0": None, "t0.s1": "有人说我搬走了。", "t0.s2": "这不对，我还住在旧街。", "t0.s3": "这不对，", "t0.s4": "我还住在旧街。"}
    assert source_rows[3]["事实出处"] == {"t3.s2": "后来我搬了花盆。"}
    assert [(ep.source_quote, ep.source_turn_index) for ep in result.new_episodes] == [
        (texts[0], 0), ("后来我搬了花盆。", 3), (texts[5], 5)]
    assert all(ep.source_context == texts[ep.source_turn_index] and ep.source_session_id == ctx.session_id
               for ep in result.new_episodes)
    assert result.new_episodes[0].keywords == ["旧街"] and result.new_episodes[0].can_reference()
    assert not result.new_episodes[-1].can_reference()
    relationship, = result.relationships
    assert (relationship.name, relationship.relation, relationship.facts) == ("方桐", "姑姑", [texts[1]])
    assert relationship.evidence[0].fact_sources[0].source_context == texts[1]
    assert not relationship.consent_to_share and relationship.can_reference()


@pytest.mark.parametrize("privacy", [[None], [{"turn_index": 0, "evidence_quote": "本人从未说过这句。"}]])
def test_bad_privacy_review_still_blocks_indexed_facts(tmp_path, monkeypatch, privacy):
    from tests.test_deepseek_offline_route import clients, guard, response, dispatch
    ctx, client = context(), clients().memory
    payload = {field: [] for field in FIELDS}
    payload.update(episodes=[None, {"source_id": "t0", "keywords": [], "referenceable": True}],
                   cognitive_privacy=privacy)
    data = response(client)
    data["choices"][0]["message"]["content"] = json.dumps(payload, ensure_ascii=False)
    dispatch(monkeypatch, data)
    extractor = LLMExtractor(client, prefix_recovery=True)
    with closing(guard(tmp_path)) as guarded:
        guarded.attach(client)
        with guarded.scope("review", "bad-privacy", ctx.session_id):
            result = extractor.extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert extractor.last_status == "failed" and not result.new_episodes
    assert not ctx.working_memory.get("_memory_privacy_review_completed")


@pytest.mark.parametrize("change", ["context_owner", "profile_owner", "session", "text", "mask"])
def test_fact_source_cannot_be_rebound_after_generation(change):
    ctx = context()
    profile = PersonaProfile(user_id=ctx.user_id)
    sources = snapshot(ctx)
    request, catalog = sources.fact_input()
    assert json.loads(request)[0]["本人原话"] == ctx.turns[0].user
    if change == "context_owner":
        ctx.user_id = "other"
    elif change == "profile_owner":
        profile.user_id = "other"
    elif change == "session":
        ctx.session_id = "other"
    elif change == "text":
        ctx.turns[0].user = "另一段本人原话。"
    else:
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    assert sources.bind_facts(catalog, [{"source_id": "t0", "keywords": [], "referenceable": True}],
                              [], ctx, profile) == ([], [])


def test_long_owner_turn_offers_literal_english_sentences_with_full_context():
    ctx = context()
    ctx.turns[0].user = ("Yesterday we met at lane 1.5 in the swimming pool. " * 6
                         + "Today I kept the red scarf.")
    sources = snapshot(ctx)
    request, catalog = sources.fact_input()
    row = json.loads(request)[0]
    quote = " Today I kept the red scarf."
    idx = next(key for key, span in row["事实出处"].items() if span == quote)
    episodes, _ = sources.bind_facts(catalog, [{"source_id": idx, "keywords": [], "referenceable": True}],
                                     [], ctx, PersonaProfile(user_id=ctx.user_id))
    assert row["本人原话"] == ctx.turns[0].user and episodes[0]["evidence_quote"] == quote
    assert all(not span or not span.endswith("lane 1.") for span in row["事实出处"].values())


def test_fact_choice_does_not_reinterpret_turn_number_or_save_a_mismatched_source():
    ctx = context()
    ctx.turns[0].user = "我同意先听我讲，别急着问。"
    ctx.turns[1].user = "我和舅舅在门口种了茉莉。"
    sources = snapshot(ctx)
    _, catalog = sources.fact_input()
    row = {"keywords": ["茉莉"], "referenceable": True, "preference_source_id": None}
    episodes, _ = sources.bind_facts(catalog, [
        {**row, "source_id": 1}, {**row, "source_id": "1"}, {**row, "source_id": "t0"},
        {**row, "source_id": "t1"},
    ], [], ctx, PersonaProfile(user_id=ctx.user_id))
    assert episodes == [{"turn_index": 1, "evidence_quote": ctx.turns[1].user,
                         "keywords": ["茉莉"], "referenceable": True}]


def test_mixed_turn_can_identify_two_disjoint_verbatim_sources():
    ctx = context()
    ctx.turns[0].user = "以后先听我讲，今天舅舅于川带来一盆茉莉。"
    sources = snapshot(ctx)
    wire, catalog = sources.fact_input(preference_turns={0})
    exposed = json.loads(wire)[0]
    assert exposed["相处说明来源"] is True
    refs = {value: key for key, value in exposed["事实出处"].items()}
    fact, control = refs["今天舅舅于川带来一盆茉莉。"], refs["以后先听我讲，"]
    row = {"source_id": fact, "preference_source_id": control, "keywords": ["茉莉"], "referenceable": True}
    relation = {"name": "于川", "relation": "舅舅", "source_ids": [fact], "referenceable": True,
                "preference_source_id": control}
    episodes, relationships = sources.bind_facts(catalog, [row], [relation], ctx, PersonaProfile(user_id=ctx.user_id))
    assert episodes[0]["evidence_quote"] == relationships[0]["facts"][0]["evidence_quote"] == "今天舅舅于川带来一盆茉莉。"
    assert episodes[0]["preference_source_quote"] == relationships[0]["facts"][0]["preference_source_quote"] == "以后先听我讲，"
    for invalid in (fact, "t0", "t1", "t0.s999", 0):
        episodes, _ = sources.bind_facts(catalog, [{**row, "preference_source_id": invalid}], [],
                                         ctx, PersonaProfile(user_id=ctx.user_id))
        assert "preference_source_quote" not in episodes[0]


def test_shared_keyword_does_not_turn_a_different_turns_facts_into_this_one():
    ctx = context()
    ctx.turns[0].user = "先听我说，院里的茉莉开了。"
    ctx.turns[1].user = "我和小柯把茉莉送给学校。"
    sources = snapshot(ctx)
    wire, catalog = sources.fact_input()
    refs = {value: key for key, value in json.loads(wire)[0]["事实出处"].items()}
    flower = refs["院里的茉莉开了。"]
    control = refs["先听我说，"]
    valid = {"source_id": flower, "keywords": ["茉莉", "院里"], "referenceable": True,
             "preference_source_id": control}
    # Both mention jasmine. Only turn 1 supports all three keywords together.
    wrong = {**valid, "keywords": ["小柯", "茉莉", "学校"], "preference_source_id": None}
    for order in ([wrong, valid], [valid, wrong]):
        episodes, _ = sources.bind_facts(catalog, order, [], ctx, PersonaProfile(user_id=ctx.user_id))
        assert episodes == [{"turn_index": 0, "evidence_quote": "院里的茉莉开了。", "keywords": ["茉莉", "院里"],
                             "referenceable": True, "preference_source_quote": "先听我说，"}]
    # A shared word on its own is supported, and a bad duplicate may not lend
    # its control proof to the independently valid row or erase its privacy.
    episodes, _ = sources.bind_facts(catalog, [
        {**wrong, "preference_source_id": control, "referenceable": False},
        {**valid, "keywords": ["茉莉"], "preference_source_id": None},
    ], [], ctx, PersonaProfile(user_id=ctx.user_id))
    assert episodes == [{"turn_index": 0, "evidence_quote": "院里的茉莉开了。",
                         "keywords": ["茉莉"], "referenceable": False}]


@pytest.mark.parametrize("life_channel", ["episodes", "relationships"])
def test_one_selection_keeps_mixed_life_fact_and_rejects_explanation_in_either_channel(
        tmp_path, monkeypatch, life_channel):
    from aria.engine.memory.referenceability import SOURCE_CONTEXT_VERSION
    from aria.engine.model.persona import PreferenceQuote, StatedPreference
    from tests.test_deepseek_offline_route import clients, guard, response, dispatch

    client = clients().memory
    ctx = context()
    explanation = "我说想停一下，是想想刚才的话，并没有疏远小严的意思。"
    life = "他们陪我看完了那场电影。"
    ctx.turns[0].user = explanation + life
    ctx.turns[1].user = "朋友小严和我一起看电影。"
    wire, _ = snapshot(ctx).fact_input()
    refs = {value: key for key, value in json.loads(wire)[0]["事实出处"].items()}
    pair = {"referenceable": True, "preference_source_id": refs[explanation]}
    wrong = {"source_id": refs[explanation], "keywords": [], "referenceable": True,
             "preference_source_id": None}
    relationship = {"name": "小严", "relation": "", "source_ids": [refs["并没有疏远小严的意思。"]],
                    "referenceable": True, "preference_source_id": "independent_relationship"}
    payload = {k: [] for k in FIELDS}
    payload["episodes"] = [wrong]
    payload["relationships"] = [relationship, dict(relationship, relation="朋友", source_ids=["t1"])]
    if life_channel == "episodes":
        payload["episodes"].append({**pair, "source_id": refs[life], "keywords": ["电影"]})
    else:
        # Claiming a person's identity fails; the life fact was still selected once.
        payload["relationships"].append({**pair, "name": "小严", "relation": "朋友",
                                         "source_ids": [refs[life]]})
    data = response(client)
    data["choices"][0]["message"]["content"] = json.dumps(payload, ensure_ascii=False)
    calls = dispatch(monkeypatch, data)
    profile = PersonaProfile(user_id=ctx.user_id)
    with closing(guard(tmp_path)) as allowance:
        allowance.attach(client)
        with allowance.scope("review", "mixed-life", ctx.session_id):
            result = LLMExtractor(client, prefix_recovery=True).extract(ctx, profile)
    assert len(calls) == 1
    assert not result.stated_preferences
    episode, = result.new_episodes
    assert (episode.source_quote, episode.preference_source_quote) == (life, explanation)
    relation, = result.relationships
    assert relation.name == "小严" and relation.facts == [ctx.turns[1].user]
    assert relation.consent_to_share is False
    old = PreferenceQuote(session_id=ctx.session_id, turn_index=0, text=explanation)
    profile.stated_preferences = [StatedPreference(
        id="pref_former", do="聊天先等我说完", scope="闲聊", status="withdrawn",
        quotes=[old], confirmation=old, created_at="2026-11-01", updated_at="2026-11-01",
    )]
    profile.relationship_context = [relation]
    profile.episodic_memory = [episode]
    assert profile.referenceable_relationships()[0].facts == [ctx.turns[1].user]
    assert profile.referenceable_episodes()[0].source_quote == life
    # A v8 name match or an invalid purpose token could not establish this
    # later statement's use independently of the retired communication rule.
    relation.evidence[0].fact_sources[0].source_context_version = SOURCE_CONTEXT_VERSION
    assert profile.referenceable_relationships() == []
    assert relation.facts == [ctx.turns[1].user]


def test_withdrawal_honors_a_later_v8_relationships_own_verified_pair_only():
    from aria.engine.memory.referenceability import SOURCE_CONTEXT_VERSION, source_context_sha256
    from aria.engine.model.persona import PreferenceQuote, StatedPreference

    control = "我说先让我讲，是指把手工这件事讲完。"
    life = "朋友小林和我昨天修好了桌脚。"
    context = control + life
    earlier = PreferenceQuote(session_id="craft", turn_index=0, text="聊手工让我先说完。")
    retired = StatedPreference(id="pref_wood", do=earlier.text, scope="手工", quotes=[earlier],
                               confirmation=earlier, status="withdrawn", created_at="2026-11-01",
                               updated_at="2026-11-01")
    pair = dict(quote=life, turn_index=1, source_context=context, source_context_version=SOURCE_CONTEXT_VERSION,
                source_context_sha256=source_context_sha256(context), preference_source_quote=control)
    profile = PersonaProfile(user_id="owner", stated_preferences=[retired], relationship_context=[{
        "id": "lin", "name": "小林", "relation": "朋友", "referenceable": True, "facts": [life, control],
        "first_observed_at": "2026-11-01", "evidence": [{"at": "2026-11-01", "session_id": "craft",
            "turn_indices": [1], "summary": context, "fact_sources": [
                pair, dict(pair, quote=control, preference_source_quote=None),
            ]}],
    }])
    row, = profile.relationship_context
    source = row.evidence[0].fact_sources[0]
    assert [item.facts for item in profile.referenceable_relationships()] == [[life]]
    assert row.facts == [life, control] and row.consent_to_share is False
    for invalid in (None, "原话里没有的说明", life, context):
        source.preference_source_quote = invalid
        assert profile.referenceable_relationships() == []
    source.preference_source_quote = control
    source.source_context_sha256 = "mismatch"
    assert profile.referenceable_relationships() == []


def test_v2_integer_contract_preserves_its_frozen_shape_but_cannot_start_a_fresh_release(tmp_path):
    import hashlib
    from aria.engine.memory.memory_contract import LEGACY_FACT_SOURCE_REFS
    from tests.test_deepseek_offline_route import clients, guard
    client = clients().memory
    client.memory_fact_source_protocol = LEGACY_FACT_SOURCE_REFS
    contract = memory_request_format(client)["json_schema"]
    encoded = json.dumps(contract["schema"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert contract["name"] == "aria_memory_fact_sources_v2"
    assert hashlib.sha256(encoded).hexdigest() == "37bd0b66e8fe1c78de33bd6ce6299e7418dcbe5b203dced1b88ada1721b1de3d"
    with closing(guard(tmp_path)) as guarded:
        with pytest.raises(ValueError, match="retained_memory_binding_required"):
            guarded.attach(client)
        assert guarded.snapshot()["calls_used"] == 0


def test_retained_fact_schema_binding_cannot_be_used_for_new_sources(tmp_path, monkeypatch):
    from tests.test_deepseek_offline_route import clients, guard
    old, current = clients().memory, clients().memory
    old.memory_fact_source_protocol = None
    assert memory_request_format(old)["json_schema"]["name"] == "aria_memory_source_refs_schema_v1"
    assert memory_request_format(current)["json_schema"]["name"] == "aria_memory_fact_sources_v3"
    with monkeypatch.context() as historical:
        historical.setattr("aria.engine.policy.max_release.guard._release_pacing_interval", lambda *args: 120)
        with closing(guard(tmp_path)) as guarded:
            guarded.attach(old)
    with closing(guard(tmp_path)) as guarded:
        with pytest.raises(ValueError, match="model configuration differs from retained allowance"):
            guarded.attach(current)
        assert guarded.snapshot()["calls_used"] == 0
    current.structured_stream_protocol = None
    with pytest.raises(ValueError, match="unsupported_memory_fact_source_protocol"):
        memory_request_format(current)


def test_same_synthetic_context_protocol_size_comparison(tmp_path, monkeypatch):
    from tests.test_memory_structured_candidate import clients, guard, response
    from tests.test_reserved_calls import stub
    metrics = {"evidence_kind": "software_constructed_same_context", "model_calls": 0, "protocols": {}}
    for protocol in (LEGACY, SOURCE_REFS):
        ctx, client = context(), clients().memory
        client.preference_source_protocol = protocol
        client.preference_scope_protocol = PREFERENCE_SCOPE_MODEL
        row = candidate()
        if protocol == LEGACY:
            row["quotes"] = [{"turn_index": index, "text": ctx.turns[index].user}
                             for index in row.pop("source_turn_indices")]
            row["revision_basis"] = row.pop("revision_turn_index")
        payload = {field: [] for field in FIELDS}
        payload["stated_preferences"] = [row]
        raw = json.dumps(payload, ensure_ascii=False)
        calls = stub(monkeypatch, response(client, raw))
        with closing(guard(tmp_path / (protocol + ".sqlite3"), client)) as guarded:
            guarded.attach(client)
            with guarded.scope("review", "software-memory", ctx.session_id):
                result = LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
        assert len(calls) == len(result.stated_preferences) == 1
        wire = calls[0]
        examples = [value for value in embedded_json_objects(wire["messages"][0]["content"])
                    if {"episodes", "relationships", "stated_preferences"} <= value.keys()]
        if protocol == LEGACY:
            example, = examples
            assert "quotes" in example["stated_preferences"][0]
            assert "source_turn_indices" not in example["stated_preferences"][0]
        else:
            assert not examples
        metrics["protocols"][protocol] = {
            "system_utf8_bytes": len(wire["messages"][0]["content"].encode()),
            "user_utf8_bytes": len(wire["messages"][1]["content"].encode()),
            "schema_utf8_bytes": len(json.dumps(wire["response_format"], ensure_ascii=False).encode()),
            "wire_utf8_bytes": len(json.dumps(wire, ensure_ascii=False).encode()),
            "fixture_response_utf8_bytes": len(raw.encode()),
        }
    print(json.dumps(metrics, ensure_ascii=False))


@pytest.mark.parametrize("typed,protocol,expected_sha", [
    (False, LEGACY, "2137d0a61b12b0c25460f925fbc5871f91e004aa59eeac4842846b1f9d99a53b"),
    (True, LEGACY, "2137d0a61b12b0c25460f925fbc5871f91e004aa59eeac4842846b1f9d99a53b"),
    (False, SOURCE_REFS, "1f5a298c7d3a0c4cb21d53447204fc37da79e779dd7e3076753c9f6a6521d6d7"),
])
def test_privacy_prompt_change_keeps_explicit_old_routes_exact(typed, protocol, expected_sha):
    from tests.test_memory_structured_candidate import route
    client, ctx = Reply(rows=[]), context()
    client.preference_source_protocol = protocol
    if typed:
        client.memory_structured_route = route()
    LLMExtractor(client, prefix_recovery=True).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert len(client.calls) == 1
    assert sha256(client.calls[0][0]["content"].encode()).hexdigest() == expected_sha
    assert client.calls[0][1]["content"] == "\n".join(
        f"[turn {index}] 用户:{turn.user}\n小予:{turn.agent}"
        for index, turn in enumerate(ctx.turns))


@pytest.mark.parametrize("referenceable", [False, True])
def test_current_scope_prompt_preserves_model_referenceability_and_separate_share_permission(referenceable):
    """Prescribed flags check storage boundaries, not semantic classification."""
    from tests.test_memory_structured_candidate import route
    client, ctx = Reply(rows=[]), context()
    client.memory_structured_route = route()
    source = "今天我和邻居周岚一起修了花架。"
    ctx.turns[1].user = source
    client.payload["episodes"] = [{
        "summary": source, "keywords": [], "referenceable": referenceable,
        "evidence_quote": source, "turn_index": 1,
    }]
    client.payload["relationships"] = [{
        "name": "周岚", "relation": "邻居", "referenceable": referenceable,
        "facts": [{"value": source, "evidence_quote": source, "turn_index": 1}],
    }]
    profile = PersonaProfile(user_id=ctx.user_id)
    before = profile.model_dump(mode="json")
    result = LLMExtractor(client, prefix_recovery=True).extract(ctx, profile)
    episode, = result.new_episodes
    relationship, = result.relationships
    assert episode.referenceable is referenceable
    assert relationship.referenceable is referenceable
    assert relationship.consent_to_share is False
    assert episode.source_quote == episode.source_context == source
    fact, = relationship.evidence[0].fact_sources
    assert fact.quote == fact.source_context == source
    assert ctx.working_memory["_sensitive_memory_excluded_turns"] == []
    assert profile.model_dump(mode="json") == before
