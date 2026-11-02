"""Mechanical codec and actual guarded payload tests; no live effect evidence."""
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
import re

import pytest

from aria.engine.policy.reflection_compact import (
    compact_reflection_schema, compact_reflection_request_format,
    compact_reflection_system_prompt, compact_reflection_messages,
    encode_compact_reflection, decode_compact_reflection,
    indexed_compact_reflection_schema, indexed_compact_reflection_request_format,
    indexed_compact_reflection_system_prompt, encode_indexed_compact_reflection,
    decode_indexed_compact_reflection,
)
from aria.engine.policy.reflection_json import reflection_request_format
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.release import create_qwen_max_compact_reflection_clients
from tests.test_reflection_structured_candidate import guard, client as old_client
from tests.test_memory_structured_candidate import route
from tests.test_qwen_max_candidate import route as max_route
from tests.test_reflection_structured_contract_v23 import payload
from tests.test_reflection_json_transport import MESSAGES, response
from tests.test_reserved_calls import stub
from tests._model_http_stub import stub_model_http


def client():
    return create_qwen_max_compact_reflection_clients(model_calls_enabled=True,
        api_key="inert-test", base_url="https://fixture.invalid/v1",
        online_structured_route=max_route(), memory_structured_route=route()).reflection


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setattr("aria.runtime.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))
    monkeypatch.setattr("aria.engine.policy.llm.openrouter_api_key", lambda: pytest.fail("credential read forbidden"))


def test_schema_is_union_free_required_closed_and_fresh():
    schema = compact_reflection_schema()
    def walk(node):
        if type(node) is dict:
            assert not {"anyOf", "oneOf", "allOf", "title", "description"} & node.keys()
            assert type(node.get("type")) is not list
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["properties"]) == set(node["required"])
            for value in node.values(): walk(value)
        elif type(node) is list:
            for value in node: walk(value)
    walk(schema)
    assert len(json.dumps(compact_reflection_request_format(), ensure_ascii=False,
                          separators=(",", ":")).encode()) == 3869
    assert len(schema["$defs"]) == 5
    schema["$defs"].clear()
    assert compact_reflection_schema()["$defs"]


def test_v4_bookkeeping_is_code_owned_and_legacy_v3_remains_explicit():
    old = compact_reflection_schema()
    current = indexed_compact_reflection_schema()
    assert "id" in old["$defs"]["_Review"]["required"]
    assert "review_id" in old["$defs"]["_Change"]["required"]
    assert "id" not in current["$defs"]["_Review"]["properties"]
    assert "review_id" not in current["$defs"]["_Change"]["properties"]
    assert current["$defs"]["_Change"]["properties"]["review_index"] == {
        "type": "integer", "enum": list(range(6))}
    assert reflection_request_format("json_schema_v4") == indexed_compact_reflection_request_format()
    prompt = indexed_compact_reflection_system_prompt()
    assert '"id":"r1"' not in prompt and "review.id 用本次唯一" not in prompt
    assert '"$defs"' not in prompt and "response_format Schema" in prompt


def test_current_prompt_does_not_rewrite_a_legacy_transport_prompt(monkeypatch):
    from aria.engine.policy.max_release import sse_adapter
    before = indexed_compact_reflection_system_prompt()
    monkeypatch.setattr(sse_adapter, "schema_free_reflection_system_prompt",
                        lambda: (_ for _ in ()).throw(AssertionError("legacy prompt used")))
    assert indexed_compact_reflection_system_prompt() == before
    assert compact_reflection_schema() != indexed_compact_reflection_schema()


@pytest.mark.parametrize("index", [0, 1, 5])
def test_v4_change_binding_is_original_array_position(index):
    original = payload()
    original["reviews"] = [deepcopy(original["reviews"][0]) for _ in range(6)]
    for number, row in enumerate(original["reviews"]):
        row["id"] = f"model-name-{number}"
    original["changes"][0]["review_id"] = original["reviews"][index]["id"]
    wire = encode_indexed_compact_reflection(original)
    assert wire["changes"][0]["review_index"] == index
    decoded = json.loads(decode_indexed_compact_reflection(json.dumps(wire)))
    assert [row["id"] for row in decoded["reviews"]] == [f"r{i + 1}" for i in range(6)]
    assert decoded["changes"][0]["review_id"] == f"r{index + 1}"
    for left, right in zip(original["reviews"], decoded["reviews"]):
        assert {k: v for k, v in left.items() if k != "id"} == {k: v for k, v in right.items() if k != "id"}


@pytest.mark.parametrize("index", [True, False, -1, 6, 1.0, "0", None])
def test_v4_illegal_indices_never_bind_or_discard_valid_reviews(index):
    wire = encode_indexed_compact_reflection(payload())
    wire["changes"][0]["review_index"] = index
    raw = json.dumps(wire)
    with pytest.raises(ValueError):
        decode_indexed_compact_reflection(raw)
    decoded = json.loads(decode_indexed_compact_reflection(raw, itemwise_review=True))
    assert decoded["reviews"][0]["id"] == "r1"
    assert "review_id" not in decoded["changes"][0]
    assert decoded["changes"][0]["review_index"] == index


@pytest.mark.parametrize("fault", ["missing_index", "bad_slot", "missing_review_field", "non_object_review"])
def test_v4_bad_first_item_cannot_rebind_later_change(fault):
    from tests.test_reflection_itemwise import setup
    from tests.test_reflection_evidence_review import context
    original = payload()
    first, second = deepcopy(original["reviews"][0]), deepcopy(original["reviews"][0])
    first["id"], second["id"] = "first", "second"
    original["reviews"] = [first, second]
    original["changes"][0]["review_id"] = "second"
    wire = encode_indexed_compact_reflection(original)
    if fault == "missing_index":
        wire["changes"][0].pop("review_index")
    elif fault == "bad_slot":
        wire["reviews"][0]["hypothesis_id"] = [None]
    elif fault == "missing_review_field":
        wire["reviews"][0].pop("reason")
    else:
        wire["reviews"][0] = None
    canonical = json.loads(decode_indexed_compact_reflection(json.dumps(wire), itemwise_review=True))
    assert len(canonical["reviews"]) == 2 and canonical["reviews"][1]["id"] == "r2"
    updater, client, persona = setup()
    client.payload = canonical
    updater.update(persona, context())
    assert updater.last_review_results[1]["status"] == "accepted"
    if fault == "missing_index":
        assert updater.last_proposal_results[0]["reason"] == "invalid_change_shape"
    else:
        assert updater.last_review_results[0]["status"] == "rejected"
        assert canonical["changes"][0]["review_id"] == "r2"
        assert updater.last_proposal_results[0]["reason"] == "event_identity_unknown_no_update"


@pytest.mark.parametrize("collection,field", [("reviews", "hypothesis_id"), ("changes", "scope")])
@pytest.mark.parametrize("invalid", [None, "domain-valid-scalar", [None], ["one", "two"]])
def test_v4_illegal_nullable_wire_cannot_masquerade_as_valid_domain(collection, field, invalid):
    from tests.test_reflection_itemwise import setup
    from tests.test_reflection_evidence_review import context
    wire = encode_indexed_compact_reflection(payload())
    wire[collection][0][field] = invalid
    canonical = json.loads(decode_indexed_compact_reflection(json.dumps(wire), itemwise_review=True))
    required = "id" if collection == "reviews" else "review_id"
    assert required not in canonical[collection][0]
    updater, client, persona = setup()
    client.payload = canonical
    updater.update(persona, context())
    if collection == "reviews":
        assert updater.last_review_results[0]["reason"] == "invalid_review_shape"
        assert updater.last_proposal_results[0]["reason"] == "bound_review_rejected"
    else:
        assert updater.last_review_results[0]["status"] == "accepted"
        assert updater.last_proposal_results[0]["reason"] == "invalid_change_shape"


def test_v4_all_null_wire_slots_do_not_become_a_valid_legacy_review():
    from tests.test_reflection_itemwise import setup
    from tests.test_reflection_evidence_review import context
    wire = encode_indexed_compact_reflection(payload())
    for field in ("hypothesis_id", "claim_quote", "scope_quote"):
        wire["reviews"][0][field] = None
    canonical = json.loads(decode_indexed_compact_reflection(json.dumps(wire), itemwise_review=True))
    updater, client, persona = setup()
    client.payload = canonical
    updater.update(persona, context())
    assert updater.last_review_results[0]["reason"] == "invalid_review_shape"
    assert not persona.evidence_review_ledger


@pytest.mark.parametrize("fault", ["duplicate", "extra_object", "fence", "seven_reviews"])
def test_v4_does_not_reinterpret_legacy_ids_or_repair_whole_json(fault):
    wire = encode_indexed_compact_reflection(payload())
    if fault == "seven_reviews": wire["reviews"] *= 7
    raw = json.dumps(wire)
    if fault == "duplicate": raw = raw[:-1] + ',"changes":[]}'
    if fault == "extra_object": raw += '\n{"protocol_version":"2.0","reviews":[],"changes":[]}'
    if fault == "fence": raw = '```json\n' + raw + '\n```'
    with pytest.raises(ValueError):
        decode_indexed_compact_reflection(raw, itemwise_review=True)


@pytest.mark.parametrize("collection", ["reviews", "changes"])
@pytest.mark.parametrize("bound_index", [0, 1])
def test_v4_legacy_id_is_rejected_without_poisoning_legal_sibling(collection, bound_index):
    from tests.test_reflection_itemwise import setup
    from tests.test_reflection_evidence_review import context
    wire = encode_indexed_compact_reflection(payload())
    wire["reviews"].append(deepcopy(wire["reviews"][0]))
    wire["changes"][0]["review_index"] = bound_index
    field = "id" if collection == "reviews" else "review_id"
    # The model's illegal ID deliberately collides with the second review's
    # generated ID. It must never enter the old duplicate-ID counter.
    wire[collection][0][field] = "r2"
    original_bad = deepcopy(wire[collection][0])
    canonical = json.loads(decode_indexed_compact_reflection(json.dumps(wire), itemwise_review=True))
    assert canonical[collection][0] == {"invalid_wire_item": original_bad}
    updater, client, persona = setup()
    client.payload = canonical
    updater.update(persona, context())
    assert updater.last_review_results[1]["status"] == "accepted"
    if collection == "reviews":
        assert updater.last_review_results[0]["reason"] == "invalid_review_shape"
        assert updater.last_proposal_results[0]["reason"] == (
            "bound_review_rejected" if bound_index == 0 else "event_identity_unknown_no_update")
    else:
        assert updater.last_proposal_results[0]["reason"] == "invalid_change_shape"
    with pytest.raises(ValueError):
        decode_indexed_compact_reflection(json.dumps(wire))


@pytest.mark.parametrize("field", ["hypothesis_id", "claim_quote", "scope_quote"])
@pytest.mark.parametrize("value", [None, "", "带中文的精确来源"])
def test_nullable_review_slots_roundtrip_without_value_invention(field, value):
    original = payload(); original["reviews"][0][field] = value
    before = deepcopy(original)
    encoded = encode_compact_reflection(original)
    assert encoded["reviews"][0][field] == ([] if value is None else [value])
    assert json.loads(decode_compact_reflection(json.dumps(encoded))) == original == before


@pytest.mark.parametrize("field,value", [
    ("id", None), ("id", ""), ("id", "existing-id"),
    ("scope", None), ("scope", ""), ("scope", "原范围"),
    ("target_value", None), ("target_value", 0), ("target_value", 1),
    ("lineage", None), ("lineage", {"id": "existing", "claim_quote": "原主张", "scope_quote": "",
       "mechanism_relation": "same", "scope_relation": "within"}),
])
def test_nullable_change_slots_roundtrip(field, value):
    original = payload(); original["changes"][0][field] = value
    if field == "target_value":
        original["changes"][0].update(framework="big_five", dimension="openness")
    encoded = encode_compact_reflection(original)
    assert json.loads(decode_compact_reflection(json.dumps(encoded))) == original


@pytest.mark.parametrize("basis,outcome,valid", [
    ("stated_preference", None, True), ("stated_preference", "假设的反应", False),
    ("observed_response", None, False), ("observed_response", "有据的反应", True),
    ("repeated_pattern", None, True), ("repeated_pattern", "有据的反应", True),
])
def test_local_atomic_cross_field_contract_still_rejects_invalid_combinations(basis, outcome, valid):
    original = payload(); original["changes"][0]["atomic_claim"].update(evidence_basis=basis, outcome=outcome)
    raw = json.dumps(encode_compact_reflection(original))
    if valid:
        assert json.loads(decode_compact_reflection(raw)) == original
    else:
        with pytest.raises(ValueError): decode_compact_reflection(raw)


@pytest.mark.parametrize("value", [None, "", "x", [None], ["one", "two"], False, {}])
def test_malformed_slot_never_guesses_discards_or_fills(value):
    encoded = encode_compact_reflection(payload()); encoded["reviews"][0]["hypothesis_id"] = value
    with pytest.raises(ValueError): decode_compact_reflection(json.dumps(encoded), itemwise_review=True)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "nonfinite", "fence", "prefix"])
def test_json_and_original_field_validation_stays_strict(fault):
    encoded = encode_compact_reflection(payload())
    if fault == "missing": del encoded["changes"][0]["atomic_claim"]["outcome"]
    if fault == "extra": encoded["reviews"][0]["invented_permission"] = True
    raw = json.dumps(encoded, ensure_ascii=False)
    if fault == "duplicate": raw = raw[:-1] + ',"protocol_version":"2.0"}'
    if fault == "nonfinite": raw = raw[:-1] + ',"x":NaN}'
    if fault == "fence": raw = '```json\n' + raw + '\n```'
    if fault == "prefix": raw = '结果：' + raw
    with pytest.raises(ValueError): decode_compact_reflection(raw)


def test_itemwise_policy_and_multiple_proposals_are_not_silently_dropped():
    original = payload(); original["changes"] *= 3
    decoded = json.loads(decode_compact_reflection(json.dumps(encode_compact_reflection(original)), itemwise_review=True))
    assert decoded == original and len(decoded["changes"]) == 3
    with pytest.raises(ValueError): decode_compact_reflection(json.dumps(encode_compact_reflection(original)))
    original["reviews"][0].pop("reason")
    decoded = json.loads(decode_compact_reflection(json.dumps(encode_compact_reflection(original)), itemwise_review=True))
    assert "reason" not in decoded["reviews"][0]


def test_actual_guarded_wire_decodes_once_and_binds_both_representations(tmp_path, monkeypatch):
    c = client(); original = payload(); encoded = json.dumps(encode_compact_reflection(original), ensure_ascii=False)
    calls = stub(monkeypatch, response(c, encoded))
    with closing(guard(tmp_path / "ledger.sqlite", c)) as g:
        g.attach(c)
        with g.scope("review", "codec", "synthetic"):
            decoded = c.complete(MESSAGES)
        assert json.loads(decoded) == original
        assert g.snapshot()["calls_used"] == 1
        assert g.snapshot()["known_cost_usd"] == "0.00001"
    wire, = calls
    assert wire["messages"] == compact_reflection_messages(MESSAGES)
    assert wire["messages"][1] == MESSAGES[1]
    assert wire["response_format"] == compact_reflection_request_format()
    assert wire["max_tokens"] == 2800 and c.timeout == 90
    assert wire["reasoning"] == {"enabled": False, "exclude": True}
    assert wire["provider"]["only"] == ["anthropic"]
    assert c.last_diagnostic["reflection_wire_codec"] == {
        "version": "reflection_json_schema_v3", "public_wire_sha256": sha256(encoded.encode()).hexdigest(),
        "canonical_sha256": sha256(decoded.encode()).hexdigest(), "representation_only": True,
        "content_repaired": False}


@pytest.mark.parametrize("fault", ["old_prompt", "changed_prompt", "schema", "reasoning", "timeout"])
def test_wire_or_client_tamper_stops_before_send(tmp_path, monkeypatch, fault):
    c = client(); calls = stub(monkeypatch, response(c))
    original = c._request_for_model
    def alter(payload, *, model):
        value = deepcopy(payload)
        if fault == "old_prompt": value["messages"] = deepcopy(MESSAGES)
        if fault == "changed_prompt": value["messages"][0]["content"] += "变动"
        if fault == "schema": value["response_format"]["json_schema"]["strict"] = False
        if fault == "reasoning": value["reasoning"] = {"effort": "low", "exclude": True}
        return original(value, model=model)
    monkeypatch.setattr(c, "_request_for_model", alter)
    with closing(guard(tmp_path / "ledger.sqlite", c)) as g:
        g.attach(c)
        if fault == "timeout": c.timeout = 91
        with g.scope("review", "codec", "synthetic"), pytest.raises((ValueError, MinimalCallStopped)):
            c.complete(MESSAGES)
        assert g.snapshot()["calls_used"] == 0
    assert calls == []


def test_old_wire_and_prompt_remain_unchanged(tmp_path, monkeypatch):
    old = old_client(); calls = stub(monkeypatch, response(old))
    with closing(guard(tmp_path / "old.sqlite", old)) as g:
        g.attach(old)
        with g.scope("review", "old", "synthetic"): old.complete(MESSAGES)
    assert calls[0]["messages"] == MESSAGES
    assert calls[0]["response_format"] == reflection_request_format("json_schema_v2")
    assert "严格 JSON 形状" not in compact_reflection_system_prompt()
    assert '空字符串必须写成 [""]' in compact_reflection_system_prompt()
    preview = create_qwen_max_compact_reflection_clients()
    assert not preview.model_calls_enabled


def test_decoded_quotes_still_face_unchanged_source_validation():
    from tests.test_reflection_evidence_review import context, setup
    original = payload(); original["reviews"][0]["evidence"][0]["evidence_quote"] = "从未出现在会话的伪造引文"
    decoded = decode_compact_reflection(json.dumps(encode_compact_reflection(original)))
    updater, llm, persona = setup(); llm.payload = json.loads(decoded)
    updater.update(persona, context())
    assert updater.last_status == "failed" and not persona.person_hypotheses


def test_v3_cannot_use_unverified_plain_reflection_client():
    from aria.runtime import create_minimal_clients
    from aria.engine.policy.reflection_json import validate_reflection_request
    c = create_minimal_clients(model_calls_enabled=True, api_key="inert-test",
        reflection_return_format="json_schema_v3").reflection
    with pytest.raises(ValueError, match="unsupported_reflection_json_route"):
        validate_reflection_request(c, MESSAGES)


def test_scope_protocol_is_durable_and_cannot_be_removed_after_attach(tmp_path):
    from tests.test_memory_structured_candidate import guard as memory_guard
    clients = create_qwen_max_compact_reflection_clients(model_calls_enabled=True,
        api_key="inert-test", base_url="https://fixture.invalid/v1",
        online_structured_route=max_route(), memory_structured_route=route())
    c = clients.memory
    with closing(memory_guard(tmp_path / "memory.sqlite", c)) as g:
        g.attach(c)
        assert json.loads(g._binding(c))["preference_scope"]["protocol"] == "preference_scope_model_v1"
        del c.preference_scope_protocol
        with pytest.raises(ValueError, match="differs from retained"): g.attach(c)


def test_serialization_instruction_separates_internal_labels_from_data_version():
    from aria.engine.memory.diagnosis import reflection_system_prompt
    prompt=compact_reflection_system_prompt()
    assert '顶层仅有 protocol_version、reviews、changes 三个键' in prompt
    assert 'protocol_version 的值固定为字符串 "2.0"' in prompt
    assert '传输版本 v3 和提示文档版本 2.5 都不是输出值' in prompt
    assert '不得增加 contract 字段' in prompt
    marker='顶层形状示例：'
    example,_=json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])
    assert example=={'protocol_version':'2.0','reviews':[],'changes':[]}
    # Domain rules remain byte-for-byte before the existing serialization split.
    domain=reflection_system_prompt(True).split('严格 JSON 形状（即使不更新也必须有protocol_version和reviews）：',1)[0]
    assert domain in prompt
    schema=compact_reflection_schema()
    assert schema['properties']['protocol_version']=={'const':'2.0','type':'string'}
    assert set(schema['properties'])=={'protocol_version','reviews','changes'}
    assert json.loads(decode_compact_reflection(json.dumps(example),itemwise_review=True))==example


@pytest.mark.parametrize('version,extra',[('v3',False),('2.5',False),('2.0',True),('v3',True)])
def test_documentation_labels_are_rejected_without_repair(version,extra):
    value={'protocol_version':version,'reviews':[],'changes':[]}
    if extra:value['contract']='prompt_contract 2.5'
    raw=json.dumps(value);original=deepcopy(value)
    with pytest.raises(ValueError):decode_compact_reflection(raw,itemwise_review=True)
    assert json.loads(raw)==original


@pytest.mark.parametrize("text", ["", " \t\r\n", "本", "完整的多字文本。",
    " \n完整的跨行文本。\n ", '含"引号"和\\反斜线', "é🙂e\u0301", "甲" * 241])
def test_v4_nonblank_patterns_match_whole_text_without_changing_local_bounds(text):
    from pydantic import TypeAdapter, ValidationError
    from aria.engine.memory.reflection_contract import _Text100, _Text240
    schema = indexed_compact_reflection_schema()["$defs"]
    legacy = compact_reflection_schema()["$defs"]
    nodes = [(schema["_Review"]["properties"][key], legacy["_Review"]["properties"][key], _Text240, 240)
             for key in ("reason", "event_summary")]
    atomic, old_atomic = schema["_AtomicClaim"]["properties"], legacy["_AtomicClaim"]["properties"]
    nodes.extend((atomic[key], old_atomic[key], _Text100, 100)
                 for key in ("aspect", "conditions", "tendency"))
    nodes.append((atomic["outcome"]["items"], old_atomic["outcome"]["items"], _Text100, 100))
    for node, old_node, local_type, limit in nodes:
        assert old_node == {"type": "string", "pattern": r"\S"}
        assert set(node) == set(old_node) and node["type"] == "string"
        assert bool(re.fullmatch(node["pattern"], text)) == bool(text.strip())
        assert bool(re.search(old_node["pattern"], text)) == bool(re.fullmatch(node["pattern"], text))
        validator = TypeAdapter(local_type)
        if text.strip() and len(text) <= limit:
            assert validator.validate_python(text) == text
        else:
            with pytest.raises(ValidationError):
                validator.validate_python(text)
    wire = encode_indexed_compact_reflection(payload())
    wire["reviews"][0].update(reason=text, event_summary=text)
    if text.strip() and len(text) <= 240:
        decoded = json.loads(decode_indexed_compact_reflection(json.dumps(wire)))
        assert decoded["reviews"][0]["reason"] == decoded["reviews"][0]["event_summary"] == text
    else:
        with pytest.raises(ValueError):
            decode_indexed_compact_reflection(json.dumps(wire))
