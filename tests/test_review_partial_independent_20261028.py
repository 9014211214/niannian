"""Independent App/receipt/seal regressions; fixed outputs are not semantic scores."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aria.engine.policy.max_release import tls_dispatch
from aria.engine.risk.safety_delivery import render_unverified_safety_v2
from tests.test_confirm_judge_safety_independence import INPUT, offline
from tests.test_installable_max_release import Response
from tests.test_review_delivery_convergence import run_candidate
from tests.test_safety_review_v2 import answer


BODY = "独立软件检查正文甲。独立软件检查正文乙。"


@pytest.mark.parametrize("field,value", [("dropped_fields", 1), ("dropped_fields", ["forged-diagnostic"]),
    ("dropped_fields", {"forged-diagnostic": "x"}), ("deletion_error", "forged-diagnostic")])
@pytest.mark.parametrize("concern", ["none", "fall"])
def test_model_extra_internal_keys_cannot_escape_local_failure_or_forge_audit(tmp_path, monkeypatch, field, value, concern):
    raw = answer(concern)
    raw[field] = value
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw)
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        audit = ctx.working_memory["safety_review"]
        assert audit["status"] == "review_failed" and BODY not in app._run["turns"][0]["response"]
        assert audit["risk_assessment"]["concern"] == concern
        assert "forged-diagnostic" not in json.dumps(audit)
        assert not guard.snapshot()["stopped"]
        assert app.submit("另一句独立检查原话。", "two", sid)["ok"]
        assert len(wires) == 4
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("concern", ["fall", "uncertain", "none"])
@pytest.mark.parametrize("missing", ["conflicts", "delete_sentence_ids"])
def test_missing_deletion_member_keeps_bound_risk_projection(tmp_path, monkeypatch, concern, missing):
    raw = answer(concern, subject="owner", time="current", urgency="immediate")
    raw.pop(missing)
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw)
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        audit = ctx.working_memory["safety_review"]
        assert audit["status"] == "review_failed" and BODY not in app._run["turns"][0]["response"]
        assert audit.get("risk_assessment", {}).get("concern") == concern
        assert (0 in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])) == (concern != "none")
        assert app.seal(sid)["ok"]
        job, = app.store.load(app._settled_owner).offline_sessions
        assert job.turns[0].user == (INPUT if concern == "none" else "")
        assert len(wires) == 2 and not guard.snapshot()["stopped"]
    finally:
        app.close(); guard.close()


def test_missing_assessment_is_unknown_with_reviewed_body(tmp_path, monkeypatch):
    raw = {"conflicts": False, "delete_sentence_ids": []}
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw)
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        audit = ctx.working_memory["safety_review"]
        assert audit["status"] == "reviewed"
        assert audit["risk_assessment"]["concern"] == "uncertain"
        assert BODY in app._run["turns"][0]["response"]
        assert render_unverified_safety_v2()[0] in app._run["turns"][0]["response"]
        assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("ids,conflicts", [([True], True), ([0.0], True), ([-1], True),
    ([0, 0], True), ([0], False), ([], True), ("0", True), ([0], "true")])
def test_bad_deletion_types_with_valid_risk_preserve_original_and_continue(tmp_path, monkeypatch, ids, conflicts):
    raw = answer("fall", subject="owner", time="current", urgency="immediate")
    raw.update(delete_sentence_ids=ids, conflicts=conflicts)
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw)
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        first = deepcopy(app._run["turns"][0])
        audit = ctx.working_memory["safety_review"]
        assert audit["status"] == "review_failed" and audit["risk_assessment"]["concern"] == "fall"
        assert BODY not in first["response"] and "跌倒" in first["response"]
        row = guard.snapshot()["attempts"][-1]
        assert row["status"] == "received" and row["usage"]["cost"] > 0
        saved = json.loads(Path(row["public_first_output"]["path"]).read_text(encoding="utf-8"))
        assert json.loads(saved["public_first_output"][0]["content"]) == raw
        assert app.submit("另一句中性原话。", "two", sid)["ok"]
        assert len(wires) == 4 and app._run["turns"][0] == first
        assert not guard.snapshot()["stopped"]
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("raw", ["not-json", '{"risk_assessment":{"concern":"fall","concern":"none"},"conflicts":false,"delete_sentence_ids":[]}'])
@pytest.mark.parametrize("existing", [False, True])
def test_unparseable_review_keeps_bound_input_but_never_removes_old_exclusion(tmp_path, monkeypatch, raw, existing):
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw, invited=True)
    if existing:
        ctx.working_memory["_sensitive_memory_excluded_turns"] = [0]
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        text = app._run["turns"][0]["response"]
        assert BODY not in text and text.endswith("好，这条记下了。")
        assert app.store.load(app._settled_owner).stated_preferences[0].status == "confirmed"
        assert ctx.working_memory["safety_review"]["status"] == "review_failed"
        assert "risk_assessment" not in ctx.working_memory["safety_review"]
        assert (0 in ctx.working_memory.get("_sensitive_memory_excluded_turns", [])) == existing
        assert app.seal(sid)["ok"]
        job, = app.store.load(app._settled_owner).offline_sessions
        assert job.turns[0].user == ("" if existing else INPUT)
        assert job.turns[0].person_response_sha256 is None
        assert len(wires) == 3 and not guard.snapshot()["stopped"]
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("raw", ["not-json", answer("fall", delete=[127], subject="owner", time="current", urgency="immediate")])
def test_failed_review_with_changed_source_does_not_apply_partial_judgment(tmp_path, monkeypatch, raw):
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], raw)
    engine, _ = app._sessions[app._settled_owner]
    complete = engine.safety_review_llm.complete
    def changed(messages, **options):
        result = complete(messages, **options)
        engine.persona.stated_preferences[0].scope += "新增条件"
        return result
    monkeypatch.setattr(engine.safety_review_llm, "complete", changed)
    try:
        assert app.submit(INPUT, "one", sid)["ok"]
        assert ctx.working_memory["safety_review"]["status"] == "source_changed"
        assert 0 in ctx.working_memory["_sensitive_memory_excluded_turns"]
        assert BODY not in app._run["turns"][0]["response"]
        assert "跌倒" not in app._run["turns"][0]["response"]
        assert app.seal(sid)["ok"]
        job, = app.store.load(app._settled_owner).offline_sessions
        assert job.turns[0].user == ""
    finally:
        app.close(); guard.close()


@pytest.mark.parametrize("failure", ["missing_cost", "wrong_provider", "bad_totals", "over_budget"])
def test_receipt_failures_still_stop_further_dispatch(tmp_path, monkeypatch, failure):
    app, guard, sid, ctx, wires = run_candidate(tmp_path, monkeypatch, [{"text": BODY}], answer())
    opened = tls_dispatch.open_observed
    def failed(request, **options):
        response = opened(request, **options)
        data = json.loads(response.body.getvalue())
        if json.loads(request.data)["max_tokens"] == 1024:
            if failure == "missing_cost": data["usage"].pop("cost")
            if failure == "wrong_provider": data["provider"] = "invalid-provider"
            if failure == "bad_totals": data["usage"]["total_tokens"] += 1
            if failure == "over_budget": data["usage"]["cost"] = 99
        return Response(data)
    monkeypatch.setattr(tls_dispatch, "open_observed", failed)
    try:
        app.submit(INPUT, "one", sid)
        assert guard.snapshot()["stopped"]
        assert BODY not in app._run["turns"][0]["response"]
        app.submit("不可再次发出的输入。", "two", sid)
        assert len(wires) == 2
    finally:
        app.close(); guard.close()
