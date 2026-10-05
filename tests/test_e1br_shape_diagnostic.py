"""Persist safe failure locations without admitting or retrying bad bodies."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import hashlib
import json

import pytest

from aria.engine.policy.e1br_calls import validate_e1br_ledger, validate_e1br_records
from aria.engine.policy.minimal_calls import MinimalCallStopped
from tests.test_e1br_calls import guard, stubs
from tests.test_online_json import DECISION, MESSAGES
from tests.test_reserved_calls import client, receipt


def invoke(g, c, operation="first"):
    with g.scope("turn", operation, "fictional-session"):
        return c.complete(MESSAGES)


def failed_row(tmp_path, monkeypatch, value, *, cost=0.00001):
    path = tmp_path / "calls.sqlite"
    content = json.dumps(value, ensure_ascii=False)
    with closing(guard(path)) as g:
        c = client(); g.attach(c)
        data = receipt(c)
        data["usage"]["cost"] = cost
        data["choices"][0]["message"]["content"] = content
        calls = stubs(monkeypatch, [data, receipt(c)])
        with pytest.raises(MinimalCallStopped):
            invoke(g, c)
        state = g.snapshot()
        assert state["stopped"] and state["unknown_cost_requests"] == 0
        assert len(state["attempts"]) == len(calls) == 1
        row = state["attempts"][0]
        assert Decimal(str(row["usage"]["cost"])) == Decimal(str(cost))
        assert row["visible_content_sha256"] == hashlib.sha256(content.encode()).hexdigest()
        assert "model_reply_sha256" not in row
        with pytest.raises(MinimalCallStopped):
            invoke(g, c, "next")
        assert len(calls) == 1
    rows = validate_e1br_ledger(path, stage="main", registration_sha256="b" * 64)
    assert rows == state["attempts"]
    with closing(guard(path)) as reopened:
        assert reopened.snapshot()["attempts"] == rows
        assert reopened.snapshot()["stopped"]
    return row


@pytest.mark.parametrize("kind", ["length81", "length567", "enum", "missing", "unknown_key"])
def test_bad_schema_retains_only_safe_paths_and_never_retries(tmp_path, monkeypatch, kind):
    value = deepcopy(DECISION)
    secret = "PRIVATE_SYNTHETIC_VALUE"
    plan = value["communication_plan"]
    if kind.startswith("length"):
        plan["boundary"] = "秘" * int(kind[6:])
        expected = ("max_length", ["communication_plan", "boundary"])
    elif kind == "enum":
        plan["pace"] = secret
        expected = ("enum", ["communication_plan", "pace"])
    elif kind == "missing":
        del plan["boundary"]
        expected = ("missing", ["communication_plan", "boundary"])
    else:
        plan[secret] = secret
        expected = ("extra_forbidden", ["communication_plan", "unknown_field"])
    row = failed_row(tmp_path, monkeypatch, value)
    diagnostic = row["online_shape"]
    assert diagnostic["kind"] == "schema"
    assert [(i["type"], i["path"]) for i in diagnostic["issues"]] == [expected]
    assert secret not in json.dumps(row, ensure_ascii=False)
    assert "秘" not in json.dumps(row, ensure_ascii=False)


@pytest.mark.parametrize("mutation", ["value", "path", "empty", "kind", "status", "role", "error"])
def test_unsafe_or_misbound_saved_diagnostics_are_rejected(tmp_path, monkeypatch, mutation):
    value = deepcopy(DECISION)
    value["communication_plan"]["boundary"] = "x" * 81
    row = failed_row(tmp_path, monkeypatch, value)
    legacy = deepcopy(row)
    del legacy["online_shape"]
    validate_e1br_records([legacy], stage="main", registration_sha256="b" * 64)
    if mutation == "value":
        row["online_shape"]["raw"] = "private data"
    elif mutation == "path":
        row["online_shape"]["issues"][0]["path"] = ["private generated key"]
    elif mutation == "empty":
        row["online_shape"] = {}
    elif mutation == "kind":
        row["online_shape"]["kind"] = "private generated text"
    elif mutation == "status":
        row["status"] = "received"
    elif mutation == "role":
        row["role"] = "memory"
    else:
        row["validation_error"] = "provider_error_envelope"
    with pytest.raises(ValueError, match="invalid_e1br_online_shape"):
        validate_e1br_records([row], stage="main", registration_sha256="b" * 64)


def test_reservation_overrun_is_rejected_before_shape_observation(tmp_path, monkeypatch):
    value = deepcopy(DECISION)
    value["communication_plan"]["boundary"] = "x" * 81
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client(); g.attach(c)
        data = receipt(c)
        data["usage"]["cost"] = 0.2
        data["choices"][0]["message"]["content"] = json.dumps(value)
        calls = stubs(monkeypatch, [data])
        with pytest.raises(MinimalCallStopped):
            invoke(g, c)
        state = g.snapshot()
    row = state["attempts"][0]
    assert len(calls) == 1 and state["stopped"]
    assert row["usage"]["cost"] == 0.2
    assert row["validation_error"] == "reservation_cost_exceeded"
    assert "online_shape" not in row


def test_valid_body_has_no_failure_diagnostic(tmp_path, monkeypatch):
    with closing(guard(tmp_path / "calls.sqlite")) as g:
        c = client(); g.attach(c)
        data = receipt(c)
        data["choices"][0]["message"]["content"] = json.dumps(DECISION)
        calls = stubs(monkeypatch, [data])
        assert json.loads(invoke(g, c)) == DECISION
        state = g.snapshot()
        assert len(calls) == 1 and not state["stopped"]
        assert "online_shape" not in state["attempts"][0]
