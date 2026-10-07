"""Mechanical normalization metadata checks; no semantic accuracy claims."""
import hashlib
import json
from pathlib import Path
import socket
import urllib.request

import pytest

from aria.engine.context import Turn
from aria.engine.policy import online_decision
from aria.engine.policy.delivery_provenance import delivery_protection
from test_observation_delivery_v3 import app_fixture, context, payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    assert Path(online_decision.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    def forbidden(*args, **kwargs):
        raise AssertionError("mechanical fixture attempted network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("text", ["alpha\nbeta", "alpha\nbeta\n", "\r\nalpha\r\nbeta\r\n", "  alpha\t beta  ", "\u2003alpha\nbeta\u3000"])
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_complete_exact_raw_hash_without_changing_reply(text, version):
    modified_ctx = context(version=version, mode="observe")
    # Existing layout contract: v3 preserves internal layout; legacy versions
    # retain their compact scan/output view. No historical source is imported.
    expected = text.strip() if version == "v3" else " ".join(text.strip().split())
    actual = online_decision.apply_single_pass_decision(payload(text), modified_ctx, "中性软件输入。")
    assert actual == expected
    modified_ctx.turns.append(Turn("中性软件输入。", actual))
    observed = delivery_protection(modified_ctx)
    assert observed["version"] == "delivery_protection_v3"
    assert observed["model_response_sha256"] == digest(text)
    assert observed["delivered_response_sha256"] == digest(actual)
    stages = observed["stages"]
    assert [s["stage"] for s in stages[:2]] == ["response_normalization", "online_guard"]
    assert stages[0]["before_sha256"] == digest(text)
    assert stages[0]["after_sha256"] == digest(text.strip())
    assert all(a["after_sha256"] == b["before_sha256"] for a, b in zip(stages, stages[1:]))
    assert observed["normalization_response_changed"] == (text != text.strip())
    assert observed["protection_response_changed"] == any(s["response_changed"] for s in stages[1:])


@pytest.mark.parametrize("text", ["alpha\nbeta\n", "\r\nalpha\r\nbeta\r\n", "\u2003alpha\t beta\u3000"])
def test_web_live_source_is_preserved_for_only_outer_trim(tmp_path, text):
    app, sid, engine, ctx, client = app_fixture(tmp_path, "ask", payload(text))
    try:
        row = app.submit("中性软件输入。", "normalization-first", sid)["state"]["turns"][-1]
        assert row["source"] == "live" and row["response"] == text.strip()
        protection = row["delivery_protection"]
        assert protection["response_changed"] and protection["normalization_response_changed"]
        assert not protection["protection_response_changed"]
        assert protection["model_response_sha256"] == digest(text)
        assert protection["delivered_response_sha256"] == digest(row["response"])
        stored = json.loads((tmp_path / "live/run.json").read_text("utf-8"))["turns"][-1]
        assert stored["delivery_protection"] == protection
        assert stored["source"] == "live"
        assert len(client.calls) == 1 and not app._run["reviews"]
    finally:
        app.close()


def test_changed_guard_stays_rule_protection_and_next_turn_resets(tmp_path):
    text = "alpha\nbeta\n"
    app, sid, engine, ctx, client = app_fixture(tmp_path, "wait", payload(text))
    try:
        first = app.submit("中性软件输入。", "normalization-guard", sid)["state"]["turns"][-1]
        assert first["source"] == "rule_protection"
        assert first["delivery_protection"]["normalization_response_changed"]
        assert first["delivery_protection"]["protection_response_changed"]
        engine._state_reasoner.mode = "ask"
        client.response = payload("alpha\nbeta")
        second = app.submit("中性软件输入。", "normalization-reset", sid)["state"]["turns"][-1]
        assert second["source"] == "live"
        assert not second["delivery_protection"]["normalization_response_changed"]
        assert not second["delivery_protection"]["response_changed"]
        assert len(client.calls) == 2
    finally:
        app.close()


@pytest.mark.parametrize("invalid", ["bad json", '{"response":" \\n "}', '{"response":"alpha"}'])
def test_failed_protocol_does_not_claim_a_normalized_valid_reply(tmp_path, invalid):
    app, sid, engine, ctx, client = app_fixture(tmp_path, "ask", invalid)
    try:
        row = app.submit("中性软件输入。", "normalization-invalid", sid)["state"]["turns"][-1]
        assert row["source"] == "model_failed"
        assert row["delivery_protection"]["status"] == "not_observed"
        assert row["delivery_protection"]["model_response_sha256"] is None
        assert not row["delivery_protection"]["normalization_response_changed"]
    finally:
        app.close()
