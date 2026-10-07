"""Independent neutral software checks; no model or sealed semantic cases."""
import hashlib
import json
import socket
import urllib.request

import pytest

from test_observation_delivery_v3 import app_fixture, payload
from aria.engine.policy.delivery_provenance import delivery_protection


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network is forbidden during independent software review")
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(urllib.request, "urlopen", fail)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("mode", ["ask", "wait"])
def test_semantic_safety_projection_matches_persisted_delivery(tmp_path, mode):
    original = "软件安全原候选。"
    app, sid, engine, ctx, online = app_fixture(tmp_path, mode, payload(original, "self_harm"))
    try:
        row = app.submit("中性软件输入。", "review-safety", sid)["state"]["turns"][-1]
        assert row["source"] == "rule_safety"
        persisted = json.loads((tmp_path / "live" / "run.json").read_text("utf-8"))["turns"][-1]
        assert persisted["response"] == row["response"]
        assert persisted["delivery_protection"] == row["delivery_protection"]
        projection = row["delivery_protection"]
        assert projection["model_response_sha256"] == digest(original)
        assert projection["delivered_response_sha256"] == digest(row["response"])
        assert projection["response_changed"] is True
    finally:
        app.close()


def test_next_turn_does_not_reuse_safety_or_wait_stage(tmp_path):
    app, sid, engine, ctx, online = app_fixture(tmp_path, "wait")
    try:
        first = app.submit("中性一。", "review-first", sid)["state"]["turns"][-1]
        assert first["delivery_protection"]["response_changed"] is True
        engine._state_reasoner.mode = "ask"
        second = app.submit("中性二。", "review-second", sid)["state"]["turns"][-1]
        assert second["source"] == "live"
        assert second["delivery_protection"]["response_changed"] is False
        assert second["delivery_protection"]["delivered_response_sha256"] == digest(second["response"])
        assert delivery_protection(ctx) == second["delivery_protection"]
    finally:
        app.close()


@pytest.mark.parametrize("invalid", ["not json", '{"response":"软件无安全字段。"}'])
def test_invalid_protocol_has_no_fabricated_observed_stage(tmp_path, invalid):
    app, sid, engine, ctx, online = app_fixture(tmp_path, "ask", invalid)
    try:
        row = app.submit("中性协议输入。", "review-protocol", sid)["state"]["turns"][-1]
        assert row["source"] == "model_failed"
        assert row["delivery_protection"]["status"] == "not_observed"
        assert row["delivery_protection"]["delivered_response_sha256"] is None
    finally:
        app.close()
