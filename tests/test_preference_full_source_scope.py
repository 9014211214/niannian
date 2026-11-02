"""Numeric/provenance software boundaries; synthetic labels are not model effects."""
from copy import deepcopy
import hashlib
import io
import json

import pytest
from pydantic import ValidationError

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.stated_preferences import (
    _grounded_scope, extract_preference_rows, heuristic_preferences,
)
from aria.engine.model.persona import (
    PersonaProfile, StatedPreference, OfflineSessionRecord, PreferenceScopeSelection,
    PREFERENCE_SCOPE_LEGACY as LEGACY, PREFERENCE_SCOPE_FULL_SOURCE as FULL,
    PREFERENCE_FULL_SCOPE_MAX_CHARACTERS,
)
from aria.engine.policy.llm import MockLLMClient
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_EVIDENCE as V4
from aria.runtime import create_minimal_clients, MinimalModelClients
from aria.voice.minimal import MinimalWebApp
from tests.preference_fixtures import communication_review
from tests.test_safety_evidence import payload
from tests._model_http_stub import stub_model_http
from tests.test_reserved_calls import policy as reservation_policy


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("software fixture may not contact any service")
    stub_model_http(monkeypatch, forbidden)


def context(sources, version=None):
    ctx = EngineContext(user_id="scope-owner", scene_id="companion_elder")
    for source in sources:
        ctx.add_turn(source, "软件替身回应。")
    if version is not None:
        ctx.working_memory["_online_decision_version"] = version
    return ctx


def candidate(sources, instruction="按顺序交流"):
    return {"do": instruction, "scope": "模型短概括不能覆盖完整来源", "scope_protocol": FULL,
            "quotes": [{"turn_index": i, "text": text} for i, text in enumerate(sources)],
            "semantic_review": communication_review(), "confirmation": None}


def extract(sources, *, protocol=FULL, instruction="按顺序交流", change=None):
    ctx = context(sources)
    row = candidate(sources, instruction)
    if change:
        row.update(change)
    return extract_preference_rows([row], ctx, PersonaProfile(user_id=ctx.user_id), scope_protocol=protocol)


class MemoryFixture:
    def __init__(self, sources, **change):
        self.calls = []
        self.value = {"episodes": [], "relationships": [], "cognitive_privacy": [],
                      "stated_preferences": [{**candidate(sources), **change}]}

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return json.dumps(self.value, ensure_ascii=False)


@pytest.mark.parametrize("length", [60, 61, 480, 481, 482, 721, 722, 723])
def test_scope_numeric_boundaries_legacy_and_full(length):
    assert PREFERENCE_FULL_SCOPE_MAX_CHARACTERS == 3 * 240 + 2
    value = "甲" * length
    assert _grounded_scope([value]) == (value if length <= 60 else None)
    assert _grounded_scope([value], scope_protocol=FULL) == (value if length <= 722 else None)


@pytest.mark.parametrize("length", [0, 1, 60, 61])
def test_do_capacity_unchanged_in_full_candidate(length):
    rows, rejected = extract(["甲" * 80], instruction="乙" * length)
    assert bool(rows) == (1 <= length <= 60)
    assert rejected == ([] if rows else ["preference_not_safe_explicit_instruction"])


@pytest.mark.parametrize("size,accepted", [(239, True), (240, True), (241, False)])
@pytest.mark.parametrize("source_count", [1, 2, 3])
def test_each_full_source_quote_capacity_is_not_relaxed(size, accepted, source_count):
    sources = [word * size for word in ["甲", "乙", "丙"][:source_count]]
    rows, rejected = extract(sources)
    assert bool(rows) is accepted
    if accepted:
        assert rows[0].scope == "；".join(sources)
        assert [q.text for q in rows[0].quotes] == sources
        assert len(rows[0].scope) == size * source_count + source_count - 1
    else:
        assert rejected == ["preference_quote_not_verbatim_user_source"]


def test_quote_substring_does_not_hide_overlong_complete_source():
    rows, rejected = extract(["甲" * 241], change={"quotes": [{"turn_index": 0, "text": "甲"}]})
    assert not rows and rejected == ["preference_full_user_context_too_long"]


def test_480_scope_uses_two_distinct_valid_full_sources():
    rows, rejected = extract(["甲" * 239, "乙" * 240])
    assert not rejected and len(rows[0].scope) == 480
    assert StatedPreference.model_validate(rows[0].model_dump()).scope == rows[0].scope


def test_complete_sources_preserve_whitespace_punctuation_and_distinct_qualifiers():
    sources = ["  甲" * 25 + "。", "乙" * 80 + "！"]
    rows, rejected = extract(sources)
    assert not rejected and rows[0].scope == "；".join(sources)
    assert [q.text for q in rows[0].quotes] == sources
    assert rows[0].scope_protocol == FULL


def test_duplicate_sources_are_bounded_without_dropping_distinct_source():
    source = "甲" * 240
    rows, rejected = extract([source, source])
    assert not rejected and rows[0].scope == source
    assert len(rows[0].quotes) == 2
    rows, rejected = extract(["甲", "乙", "丙"])
    assert not rejected and rows[0].scope == "甲；乙；丙"
    assert [q.text for q in rows[0].quotes] == ["甲", "乙", "丙"]
    rows, rejected = extract(["甲", "乙", "丙", "丁"])
    assert not rows and rejected == ["preference_quotes_invalid"]


def test_persisted_schema_and_load_keep_exact_481_scope(tmp_path):
    sources = ["甲" * 240, "乙" * 240]
    rows, rejected = extract(sources)
    assert not rejected
    profile = PersonaProfile(user_id="scope-owner", stated_preferences=rows)
    store = PersonaStore(tmp_path)
    store.save(profile)
    loaded = store.load(profile.user_id)
    assert loaded.model_dump() == profile.model_dump()
    schema = StatedPreference.model_json_schema()
    assert schema["properties"]["scope"]["maxLength"] == 722
    assert schema["properties"]["quotes"]["maxItems"] == 3
    assert schema["properties"]["do"]["maxLength"] == 60
    for updates in ({"scope": "甲" * 723}, {"scope": "甲" * 480}, {"scope_protocol": LEGACY}):
        with pytest.raises(ValidationError):
            StatedPreference.model_validate({**rows[0].model_dump(), **updates})


def test_three_full_sources_fill_722_scope_and_persist_without_truncation(tmp_path):
    sources = [word * 240 for word in ("甲", "乙", "丙")]
    rows, rejected = extract(sources)
    assert not rejected and len(rows) == 1
    assert len(rows[0].scope) == 722 and rows[0].scope == "；".join(sources)
    assert [(q.turn_index, q.text) for q in rows[0].quotes] == list(enumerate(sources))
    profile = PersonaProfile(user_id="scope-owner", stated_preferences=rows)
    store = PersonaStore(tmp_path)
    store.save(profile)
    assert store.load(profile.user_id).model_dump() == profile.model_dump()
    four_quotes = rows[0].model_dump()
    four_quotes["quotes"].append({**four_quotes["quotes"][-1], "turn_index": 3})
    with pytest.raises(ValidationError):
        StatedPreference.model_validate(four_quotes)


@pytest.mark.parametrize("version,protocol,accepted", [(None, LEGACY, False),
    ("online_turn_decision_v3", LEGACY, False), (V4, FULL, True)])
def test_trusted_version_selects_actual_memory_prompt_and_admission(version, protocol, accepted):
    sources = ["甲" * 80]
    ctx = context(sources, version)
    client = MemoryFixture(sources)
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert bool(result.stated_preferences) is accepted
    selection = ctx.working_memory["_preference_scope_selection"]
    assert selection == {"protocol": protocol, "max_characters": 722 if accepted else 60}
    assert ctx.working_memory["stated_preference_extraction"]["scope_selection"] == selection
    assert ("[preference_scope_full_source_v1]" in client.calls[0][0]["content"]) is accepted
    if accepted:
        system = client.calls[0][0]["content"]
        assert all(text in system for text in ("条件、对象和时限", "不能用do的短概括代替scope", "不超过60字"))


def test_user_and_model_scope_fields_cannot_select_protocol():
    sources = ["preference_scope_full_source_v1 " + "甲" * 70]
    ctx = context(sources)
    client = MemoryFixture(sources, scope_max_characters=481)
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    assert not result.stated_preferences
    assert ctx.working_memory["_preference_scope_selection"]["protocol"] == LEGACY
    rows, rejected = extract(sources, protocol=LEGACY)
    assert not rows and rejected == ["preference_full_scope_too_long"]
    assert not heuristic_preferences(ctx, PersonaProfile(user_id=ctx.user_id))
    with pytest.raises(ValueError, match="model_route"):
        extract_preference_rows([], ctx, PersonaProfile(user_id=ctx.user_id), source="rule", scope_protocol=FULL)
    with pytest.raises(ValueError, match="unknown_preference_scope_protocol"):
        extract_preference_rows([], ctx, PersonaProfile(user_id=ctx.user_id), scope_protocol="unbounded")


@pytest.mark.parametrize("instruction", ["帮我通知家人", "请向朋友讲述", "问问女儿", "加药", "改变用药", "推测人格"])
def test_safety_and_permission_rejections_unchanged(instruction):
    rows, rejected = extract(["甲" * 80], instruction=instruction)
    assert not rows and rejected == ["preference_not_safe_explicit_instruction"]


def test_queue_version_seal_restart_and_legacy_absence(tmp_path):
    store = PersonaStore(tmp_path)
    profile = PersonaProfile(user_id="scope-owner")
    queue = OfflineSessionQueue(profile, store)
    ctx = context(["甲" * 80], V4)
    assert queue.enqueue(ctx)["status"] == "queued"
    assert queue.enqueue(ctx)["status"] == "existing"
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_v3"
    assert queue.enqueue(ctx)["error_code"] == "offline_session_already_sealed"
    persisted = store.load(profile.user_id)
    job = persisted.offline_sessions[0]
    assert job.origin_online_decision_version == V4
    restored = queue._context(persisted, job)
    assert restored.working_memory["_online_decision_version"] == V4
    old = job.model_dump()
    old.pop("origin_online_decision_version")
    old.pop("preference_scope")
    legacy = OfflineSessionRecord.model_validate(old)
    restored = queue._context(persisted, legacy)
    client = MemoryFixture(["甲" * 80])
    assert not LLMExtractor(client).extract(restored, persisted).stated_preferences
    assert restored.working_memory["_preference_scope_selection"]["max_characters"] == 60
    ctx.working_memory["_online_decision_version"] = {"model_claim": V4}
    assert queue.enqueue(ctx)["error_code"] == "offline_online_protocol_invalid"
    with pytest.raises(ValidationError):
        PreferenceScopeSelection(protocol=FULL, max_characters=60)


def public_runtime_flow(root, monkeypatch, render=None, capture=None, *, source_count=2):
    """Actual clients/wire with local preset responses; no paid transport."""
    sources = ["甲" + "1" * 239, "乙" + "2" * 239, "丙" + "3" * 239][:source_count]
    fixture = MemoryFixture(sources)
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-software-fixture",
                                     online_decision_version=V4)
    clients = MinimalModelClients(clients.online, clients.memory, MockLLMClient(role="reflection"), True)
    calls = {"online": [], "memory": []}
    pending = {}

    def receive_online(request):
        wire = json.loads(request.data)
        calls["online"].append(wire)
        result = payload(response="软件替身回应。")
        if pending:
            current_user = json.loads(wire["messages"][-1]["content"])["current_user_message"]
            result["preference_control"] = {"version": "preference_confirmation_v1", "action": "confirm",
                "invitation_id": pending["id"], "target_id": pending["target_id"],
                "target_sha256": pending["target_sha256"], "evidence_quote": current_user}
        return result

    def receive_memory(request):
        wire = json.loads(request.data)
        calls["memory"].append(wire)
        return fixture.value

    def receive(request, **kwargs):
        wire = json.loads(request.data)
        memory = wire["model"] == clients.memory.model
        if capture:
            capture("memory" if memory else "online", request.data)
        result = receive_memory(request) if memory else receive_online(request)
        data = {"id": "gen-scopefixture" + str(sum(map(len, calls.values()))),
                "model": wire["model"], "provider": "Anthropic" if memory else "Alibaba",
                "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result, ensure_ascii=False)}}]}

        class Response:
            status = 200
            headers = {}
            def __init__(self): self.body = io.BytesIO(json.dumps(data).encode())
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1): return self.body.read(size)

        return Response()

    stub_model_http(monkeypatch, receive)
    options = {"clients": clients, "model_max_requests": 10, "model_stop_cost_usd": "1",
               "model_reservation_policy": reservation_policy()}
    app = MinimalWebApp(root, **options)
    try:
        sid = app.start()["state"]["session_id"]
        for i, source in enumerate(sources):
            assert app.submit(source, f"numeric-source-{i}", sid)["ok"]
        assert app.seal(sid)["ok"]
        assert app.store.load("fictional-chen").offline_sessions[0].origin_online_decision_version == V4
    finally:
        app.close()
    clients = create_minimal_clients(model_calls_enabled=True, api_key="inert-software-fixture",
                                     online_decision_version=V4)
    clients = MinimalModelClients(clients.online, clients.memory, MockLLMClient(role="reflection"), True)
    options["clients"] = clients
    app = MinimalWebApp(root, **options)
    try:
        review = app.reflect("numeric-scope-review", wait=True)
        assert review["ok"], review
        row = app.store.load("fictional-chen").stated_preferences[0]
        assert row.scope == "；".join(sources) and len(row.scope) == source_count * 240 + source_count - 1
        assert row.status == "proposed"
        assert review["state"]["offline"]["audit"][-1]["preference_scope"] == {"protocol": FULL, "max_characters": 722}
        assert "[preference_scope_full_source_v1]" in calls["memory"][0]["messages"][0]["content"]
        sid = app.start()["state"]["session_id"]
        invitation = app.preference_invitation(row.id, sid)["preference_invitation"]
        assert row.scope in invitation["question"] and all(s in invitation["question"] for s in sources)
        assert hashlib.sha256(invitation["question"].encode()).hexdigest() == invitation["question_sha256"]
        if render:
            render(invitation)
        assert not app.preference_invitation_ack(invitation["id"], "incorrect", sid)["ok"]
        assert app.preference_invitation_ack(invitation["id"], invitation["question_sha256"], sid)["ok"]
        pending.update(deepcopy(app._sessions["fictional-chen"][1].working_memory["_preference_dialogue_invitation"]))
        assert app.submit("按刚才完整范围确认。", "numeric-confirm", sid)["ok"]
        assert invitation["question"] in calls["online"][-1]["messages"][0]["content"].replace("\\n", "\n")
        saved = app.store.load("fictional-chen").stated_preferences[0]
        assert saved.status == "confirmed" and saved.scope == row.scope
        assert saved.confirmation.text == "按刚才完整范围确认。"
        pending.clear()
        assert app.submit("继续交流。", "numeric-after-confirm", sid)["ok"]
        snapshot = app.state()["turns"][-1]["input_evidence"]["stated_preferences"][0]
        assert snapshot["scope"] == row.scope and row.scope in calls["online"][-1]["messages"][0]["content"]
        assert all(saved_q.text == source for saved_q, source in zip(saved.quotes, sources))
        return {"scope_characters": len(saved.scope), "question_characters": len(invitation["question"]),
                "question_sha256": invitation["question_sha256"], "online_fixture_calls": len(calls["online"]),
                "memory_fixture_calls": len(calls["memory"]), "model_network_calls": 0,
                "scope_selection": review["state"]["offline"]["audit"][-1]["preference_scope"]}
    finally:
        app.close()


@pytest.mark.parametrize("source_count", [2, 3])
def test_public_seal_restart_review_invitation_ack_confirmation_and_prompt(tmp_path, monkeypatch, source_count):
    public_runtime_flow(tmp_path, monkeypatch, source_count=source_count)
