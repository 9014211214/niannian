"""No network: prospective checkpoints preserve failures, cost evidence and order."""
from __future__ import annotations

from copy import deepcopy
import json
import shutil

import pytest

from aria.bench.evaluation_journal import EvaluationJournal, JournalBlocked, JournalError
from aria.bench.provenance import fingerprint


@pytest.fixture
def binding():
    return {"profile_fingerprint": fingerprint({"run_id": "new-prospective-registration"}),
            "source_fingerprint": fingerprint("frozen-source"), "stage": "calibrate",
            "items": [{"coordinate": {"case_id": f"case-{i}"},
                       "input_fingerprint": fingerprint({"input": i})} for i in range(3)],
            "max_network_requests": 3, "max_observed_cost_usd": 1.0}


def account(*costs):
    known = [c for c in costs if c is not None]
    return {"enabled": True, "actual_network_requests": len(costs),
            "attempts": [{"role": "judge", "retry_index": 0, "http_status": 200,
                          "error_type": "network_error" if cost is None else None,
                          "network_error_kind": "incomplete_read" if cost is None else None,
                          "failure_phase": "response_read" if cost is None else None,
                          "provider_usage": {} if cost is None else {"cost": cost}}
                         for cost in costs],
            "provider_observed_cost_usd": round(sum(known), 8) if known else None,
            "provider_cost_observed_requests": len(known),
            "provider_cost_complete": bool(costs) and len(known) == len(costs)}


def start(journal, binding, index=0):
    item = binding["items"][index]
    return journal.start(item["coordinate"], item["input_fingerprint"])


def complete(journal, binding, index=0, accounting=None, result=None):
    return journal.finish(start(journal, binding, index),
                          result=result if result is not None else {"status": "scored", "score": index},
                          accounting=accounting if accounting is not None else account(0.1))


def test_started_persisted_before_control_returns_and_no_overwrite(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    token = start(journal, binding)
    record = json.loads((journal.path / "000000.started.json").read_text(encoding="utf-8"))
    assert record["fingerprint"] == token
    assert record["manifest_fingerprint"] == journal.manifest["fingerprint"]
    assert record["item"] == binding["items"][0]
    assert (journal.path / "000000.claim.json").read_bytes() == (journal.path / "000000.started.json").read_bytes()
    original = (journal.path / "000000.started.json").read_bytes()
    with pytest.raises(JournalBlocked):
        start(journal, binding)
    assert original == (journal.path / "000000.started.json").read_bytes()


def test_explicit_resume_replays_exact_original_zero_calls_and_continues(tmp_path, binding):
    path = tmp_path / "run"
    journal = EvaluationJournal.create(path, **binding)
    original = {"status": "invalid_judgment", "diagnostic": {"code": "invalid_json_syntax"}}
    complete(journal, binding, result=original)
    readonly = EvaluationJournal.open(path, **binding)
    with pytest.raises(JournalBlocked, match="explicit_resume"):
        start(readonly, binding, 1)
    assert readonly.plan_resume()["safe_pending"] == [i["coordinate"] for i in binding["items"][1:]]
    resumed = EvaluationJournal.open(path, resume=True, **binding)
    item = binding["items"][0]
    assert resumed.replay(item["coordinate"], item["input_fingerprint"]) == original
    # Returned objects cannot mutate the stored original result or account.
    replay = resumed.replay(item["coordinate"], item["input_fingerprint"])
    replay["status"] = "scored"
    assert resumed.replay(item["coordinate"], item["input_fingerprint"]) == original
    assert resumed.aggregate()["actual_network_requests"] == 1
    complete(resumed, binding, 1, accounting=account(0.1, 0.2))
    assert resumed.aggregate()["known_cost_usd"] == 0.3
    assert resumed.aggregate()["denominator"] == 3


def test_missing_unstarted_item_is_not_replayed_as_zero_score(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    item = binding["items"][0]
    assert journal.replay(item["coordinate"], item["input_fingerprint"]) is None
    assert journal.aggregate()["completed_items"] == 0


def test_new_directory_cannot_be_created_over_existing_checkpoint(tmp_path, binding):
    path = tmp_path / "run"
    EvaluationJournal.create(path, **binding)
    with pytest.raises(JournalError, match="already_exists"):
        EvaluationJournal.create(path, **binding)


@pytest.mark.parametrize("field,value", [
    ("profile_fingerprint", fingerprint("renamed-profile")),
    ("source_fingerprint", fingerprint("changed-code")),
    ("stage", "transfer"), ("max_network_requests", 99), ("max_observed_cost_usd", 9.0),
])
def test_cannot_change_registration_on_resume(tmp_path, binding, field, value):
    path = tmp_path / "run"
    EvaluationJournal.create(path, **binding)
    changed = {**binding, field: value}
    with pytest.raises(JournalError, match="mismatch"):
        EvaluationJournal.open(path, resume=True, **changed)


@pytest.mark.parametrize("change", ["reorder", "reduce", "input", "coordinate"])
def test_full_frozen_denominator_and_order_required(tmp_path, binding, change):
    path = tmp_path / "run"
    EvaluationJournal.create(path, **binding)
    changed = deepcopy(binding)
    if change == "reorder":
        changed["items"].reverse()
    elif change == "reduce":
        changed["items"].pop()
    elif change == "input":
        changed["items"][0]["input_fingerprint"] = fingerprint("other-input")
    else:
        changed["items"][0]["coordinate"] = "other-coordinate"
    with pytest.raises(JournalError, match="mismatch"):
        EvaluationJournal.open(path, resume=True, **changed)


def test_cannot_resume_copied_or_moved_directory(tmp_path, binding):
    original, copied = tmp_path / "run", tmp_path / "renamed"
    EvaluationJournal.create(original, **binding)
    shutil.copytree(original, copied)
    with pytest.raises(JournalError, match="location_mismatch"):
        EvaluationJournal.open(copied, resume=True, **binding)


def test_duplicate_coordinate_rejected_even_with_different_input(tmp_path, binding):
    binding["items"][1]["coordinate"] = deepcopy(binding["items"][0]["coordinate"])
    with pytest.raises(JournalError, match="duplicate_coordinate"):
        EvaluationJournal.create(tmp_path / "run", **binding)
    assert not (tmp_path / "run").exists()


def test_out_of_order_and_repeat_are_not_a_best_of_rerun(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    with pytest.raises(JournalBlocked, match="original_order"):
        start(journal, binding, 1)
    complete(journal, binding)
    with pytest.raises(JournalBlocked, match="original_order"):
        start(journal, binding, 0)


def test_wrong_input_and_unregistered_coordinate_block_before_started(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    with pytest.raises(JournalError, match="input_fingerprint"):
        journal.start(binding["items"][0]["coordinate"], fingerprint("changed"))
    with pytest.raises(JournalError, match="not_registered"):
        journal.start("outside", fingerprint("changed"))
    assert list(journal.path.iterdir()) == [journal.path / "manifest.json"]


def test_crash_leaves_ambiguous_never_assumed_zero_or_retried(tmp_path, binding):
    path = tmp_path / "run"
    original = EvaluationJournal.create(path, **binding)
    token = start(original, binding)
    resumed = EvaluationJournal.open(path, resume=True, **binding)
    plan, totals = resumed.plan_resume(), resumed.aggregate()
    assert plan["ambiguous"] == [binding["items"][0]["coordinate"]]
    assert not plan["safe_pending"] and plan["blocked"]
    assert totals["total_cost_usd"] is None
    assert not totals["cost_complete"] and not totals["actual_network_requests_complete"]
    with pytest.raises(JournalBlocked, match="ambiguous"):
        start(resumed, binding, 1)
    with pytest.raises(JournalBlocked, match="same_process"):
        resumed.finish(token, result={"status": "scored"}, accounting=account(0.1))


def test_crash_between_claim_and_started_blocks_even_without_network(tmp_path, binding, monkeypatch):
    from aria.bench import evaluation_journal as module

    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    write = module._write_exclusive

    def torn(path, record):
        if path.name.endswith(".started.json"):
            raise OSError("simulated disk interruption")
        write(path, record)

    monkeypatch.setattr(module, "_write_exclusive", torn)
    with pytest.raises(OSError):
        start(journal, binding)
    assert journal.plan_resume()["ambiguous"]
    assert not journal.plan_resume()["safe_pending"]


def test_unknown_fee_is_durable_and_blocks_whole_remaining_stage(tmp_path, binding):
    path = tmp_path / "run"
    journal = EvaluationJournal.create(path, **binding)
    complete(journal, binding, accounting=account(None), result={"status": "judge_error"})
    resumed = EvaluationJournal.open(path, resume=True, **binding)
    totals = resumed.aggregate()
    assert totals["unknown_cost_requests"] == 1 and totals["known_cost_usd"] == 0
    assert totals["total_cost_usd"] is None and not totals["cost_complete"]
    assert resumed.plan_resume()["completed_count"] == 1
    assert len(resumed.plan_resume()["not_started"]) == 2
    assert resumed.plan_resume()["safe_pending"] == []
    with pytest.raises(JournalBlocked, match="unknown"):
        start(resumed, binding, 1)


@pytest.mark.parametrize("limit,kind", [(1, "requests"), (0.1, "cost")])
def test_budget_is_stage_cumulative_after_explicit_reopen(tmp_path, binding, limit, kind):
    binding["max_network_requests" if kind == "requests" else "max_observed_cost_usd"] = limit
    path = tmp_path / "run"
    journal = EvaluationJournal.create(path, **binding)
    complete(journal, binding)
    resumed = EvaluationJournal.open(path, resume=True, **binding)
    with pytest.raises(JournalBlocked, match="cumulative"):
        start(resumed, binding, 1)


def test_account_reset_and_balance_fee_repair_rejected(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding)
    token = start(journal, binding, 1)
    for changed in (account(), account(0.2), account(0.2, 0.1)):
        with pytest.raises(JournalError, match="reset_or_history"):
            journal.finish(token, result={"status": "scored"}, accounting=changed)
    journal.finish(token, result={"status": "scored"}, accounting=account(0.1, 0.2))
    assert journal.aggregate()["actual_network_requests"] == 2


@pytest.mark.parametrize("changed", [
    {"provider_cost_complete": True}, {"provider_observed_cost_usd": 0},
    {"actual_network_requests": 0}, {"provider_cost_observed_requests": 1},
])
def test_unknown_fee_cannot_be_declared_complete_or_zero(tmp_path, binding, changed):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    token = start(journal, binding)
    with pytest.raises(JournalError, match="evidence_mismatch"):
        journal.finish(token, result={"status": "judge_error"}, accounting={**account(None), **changed})


def test_no_network_is_known_no_charge_not_unknown_invoice(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding, accounting=account(), result={"status": "not_assessed"})
    assert journal.aggregate()["total_cost_usd"] == 0
    assert journal.aggregate()["unknown_cost_requests"] == 0
    assert len(journal.plan_resume()["safe_pending"]) == 2


@pytest.mark.parametrize("filename", ["manifest.json", "000000.claim.json", "000000.started.json", "000000.result.json"])
def test_record_tamper_fails_closed(tmp_path, binding, filename):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding)
    path = journal.path / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    value["created_at"] = "edited"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(JournalError, match="fingerprint_mismatch"):
        journal.validate()


@pytest.mark.parametrize("filename", ["manifest.json", "000000.claim.json", "000000.started.json"])
def test_missing_required_record_rejected(tmp_path, binding, filename):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding)
    (journal.path / filename).unlink()
    with pytest.raises(JournalError):
        journal.validate()


def test_missing_result_becomes_ambiguous_not_pending(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding)
    (journal.path / "000000.result.json").unlink()
    assert journal.plan_resume()["ambiguous"] == [binding["items"][0]["coordinate"]]
    assert not journal.plan_resume()["safe_pending"]


def test_gap_or_extra_record_rejected(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    complete(journal, binding)
    complete(journal, binding, 1, accounting=account(0.1, 0.2))
    for kind in ("claim", "started", "result"):
        (journal.path / f"000000.{kind}.json").unlink()
    with pytest.raises(JournalError, match="non_prefix"):
        journal.validate()


def test_extra_unregistered_record_rejected(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    (journal.path / "000099.result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(JournalError, match="unregistered"):
        journal.validate()


def test_duplicate_json_keys_rejected(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    path = journal.path / "manifest.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('{', '{"version":"spoof",', 1), encoding="utf-8")
    with pytest.raises(JournalError, match="invalid_record_json"):
        journal.validate()


@pytest.mark.parametrize("unsafe", [
    {"api_key": "credential_value"},
    {"receipt": {"authorization": "credential_value"}},
    {"receipt": {"reasoning_content": "provider hidden chain"}},
    {"receipt": {"reasoning_details": [{"text": "provider hidden chain"}]}},
    {"reason": "Bearer abcdef0123456789"},
    {"reason": "sk-or-v1-abcdef0123456789"},
    {"response_body": '{"choices":[{"message":{"reasoning_content":"hidden"}}]}'},
    {"response_body": '{"reasoning_content":"hidden","reasoning_content":"other"}'},
    {"response_body": '{"reasoning_content":"hidden'},
    {"receipt": {"analysis": "provider hidden chain"}},
])
def test_secret_and_hidden_reasoning_never_written(tmp_path, binding, unsafe):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    token = start(journal, binding)
    with pytest.raises(JournalError):
        journal.finish(token, result=unsafe, accounting=account(0.1))
    assert not (journal.path / "000000.result.json").exists()
    stored = "".join(p.read_text(encoding="utf-8") for p in journal.path.iterdir())
    assert "credential_value" not in stored and "provider hidden chain" not in stored


def test_public_verdict_and_schema_definition_preserved_unchanged(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    result = {"status": "scored", "dimensions": {
        "support_entailment": {"score": 2, "reason": "e1 supports the bounded claim.", "evidence_ids": ["e1"]}},
        "receipt": {"request_payloads": [{"reasoning": {"effort": "low", "exclude": True},
                                          "schema": {"properties": {"analysis": {"type": "string"}}}}],
                    "response_body": '{"dimensions":{"support_entailment":{"score":2,"reason":"e1"}}}'}}
    complete(journal, binding, result=result)
    item = binding["items"][0]
    assert journal.replay(item["coordinate"], item["input_fingerprint"]) == result


def test_finish_cannot_overwrite_completed_record(tmp_path, binding):
    journal = EvaluationJournal.create(tmp_path / "run", **binding)
    token = start(journal, binding)
    journal.finish(token, result={"status": "judge_error"}, accounting=account(0.1))
    original = (journal.path / "000000.result.json").read_bytes()
    with pytest.raises(JournalBlocked, match="same_process"):
        journal.finish(token, result={"status": "scored"}, accounting=account(0.1))
    assert (journal.path / "000000.result.json").read_bytes() == original
