"""Safe, durable failure evidence from the real client and local HTTP envelopes."""
from copy import deepcopy
import json

import pytest

from scripts import run_e1b as run
from aria.engine.policy.minimal_calls import MinimalCallStopped
from tests.test_online_json import DECISION, MESSAGES
from tests.test_reserved_calls import policy, receipt, stub


@pytest.mark.parametrize("content,reason", [
    ('a private fixture without JSON', "online_content_invalid_json"),
    ('{"response":"private fixture","response":"other"}', "online_content_duplicate_key"),
    ('{"response":NaN}', "online_content_non_finite"),
    ('["private fixture"]', "online_content_not_object"),
    ('{"private fixture":"not a response"}', "online_response_missing"),
    ('{"response":42}', "online_response_not_string"),
    ('{"response":"  "}', "online_response_empty"),
    ('{"response":"private fixture"}', "online_decision_schema_invalid"),
])
def test_inner_failure_retains_known_fee_and_outer_receipt_without_body(
    tmp_path, monkeypatch, content, reason,
):
    path = tmp_path / "allowance.sqlite3"
    guard = run.EvidenceGuard(path, max_requests=20, max_observed_cost_usd="1",
        reservation_policy=policy(), stage="prescreen", registration_sha256="a" * 64)
    client = run.make_client("online_fast", "inert-unit-test-key")
    guard.attach(client)
    data = receipt(client)
    data["choices"][0]["message"]["content"] = content
    calls = stub(monkeypatch, data)
    with guard.scope("turn", "first", "synthetic-session"):
        with pytest.raises(MinimalCallStopped):
            client.complete(MESSAGES)
    state = guard.snapshot()
    row, = state["attempts"]
    assert row["status"] == "failed"
    assert row["validation_error"] == reason
    assert row["observation"]["body_status"] == "parsed"
    assert row["observation"]["envelope_kind"] == "success"
    assert row["visible_content_sha256"] == run.text_sha(content)
    assert row["client_return_content_sha256"] == run.text_sha(content.strip())
    assert "model_reply_sha256" not in row
    assert row["reply_shape"]["content_utf8_bytes"] == len(content.encode("utf-8"))
    if reason == "online_content_invalid_json":
        assert row["reply_shape"]["json_valid"] is False
        assert type(row["reply_shape"]["syntax_error_offset"]) is int
    if reason == "online_response_missing":
        assert row["reply_shape"]["required_fields_present"]["response"] is False
        assert row["reply_shape"]["unknown_top_level_fields"] == 1
    assert row["online_decision_version"] == "online_turn_decision_v1"
    assert state["known_cost_usd"] == "0.00001"
    assert state["unknown_cost_requests"] == 0 and state["reserved_cost_usd"] == "0"
    assert "private fixture" not in json.dumps(state)
    assert "inert-unit-test-key" not in json.dumps(state)
    with guard.scope("turn", "second", "synthetic-session"):
        with pytest.raises(MinimalCallStopped):
            client.complete(MESSAGES)
    assert len(calls) == 1
    first = deepcopy(row)
    guard.close()
    retained = run.EvidenceGuard(path, max_requests=20, max_observed_cost_usd="1",
        reservation_policy=policy(), stage="prescreen", registration_sha256="a" * 64)
    assert retained.snapshot()["attempts"] == [first]
    assert retained.snapshot()["stopped"]
    retained.close()


@pytest.mark.parametrize("integrated", [False, True])
def test_valid_text_and_json_still_return_exact_client_content(tmp_path, monkeypatch, integrated):
    guard = run.EvidenceGuard(tmp_path / "valid.sqlite3", max_requests=20,
        max_observed_cost_usd="1", reservation_policy=policy(), stage="prescreen",
        registration_sha256="b" * 64)
    client = run.make_client("online_fast", "inert-unit-test-key")
    guard.attach(client)
    content = json.dumps(DECISION, ensure_ascii=False) if integrated else DECISION["response"]
    data = receipt(client)
    data["choices"][0]["message"]["content"] = "\n " + content + " \n"
    calls = stub(monkeypatch, data)
    with guard.scope("turn", "first", "synthetic-session"):
        returned = client.complete(MESSAGES if integrated else MESSAGES[1:])
    assert returned == (data["choices"][0]["message"]["content"] if integrated else content)
    row, = guard.snapshot()["attempts"]
    assert row["status"] == "received" and "validation_error" not in row
    assert row["model_reply_sha256"] == run.text_sha(DECISION["response"])
    assert row["client_return_content_sha256"] == run.text_sha(returned)
    assert len(calls) == 1
    guard.close()
