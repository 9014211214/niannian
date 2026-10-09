"""Secondary dispatch uses fake HTTP envelopes; no live network or credentials."""
from tests._model_http_stub import stub_model_http
import json
import urllib.request

import pytest

from scripts import e1b_evaluation as evaluation
from scripts.e1b_secondary import SecondaryStopped, run_secondary
from scripts.run_e1b import EvidenceGuard
from test_e1b_evaluation import make_run
from test_reserved_calls import client, policy, receipt


def setup(tmp_path, *, real=True):
    _, protocol, run = make_run(tmp_path)
    protocol["status"] = "registered"
    for case in protocol["case_ids"]:
        path = run / f"main/{case}/report.json"
        report = evaluation.read_json(path)
        report.update(evidence_kind="real_model" if real else "software_fixture",
                      frozen=True, registration_sha256=evaluation.digest(protocol))
        path.write_bytes(evaluation.canonical(report))
    bundle = tmp_path / "bundle"
    evaluation.export_packet(run, bundle, protocol, registered_sha256=evaluation.digest(protocol))
    evaluation.prepare_secondary(bundle)
    packet, _ = evaluation.load_bundle(bundle)
    guard = EvidenceGuard(tmp_path / "allowance.sqlite3", max_requests=100,
                          max_observed_cost_usd="2", reservation_policy=policy(),
                          stage="judge", registration_sha256="a" * 64)
    judge = client("judge")
    guard.attach(judge)
    return bundle, guard, judge, evaluation.digest(packet)


def stub(monkeypatch, judge, bundle, *, change=None):
    calls = []
    class Response:
        status = 200
        headers = {}
        def __init__(self, data):
            self.data = data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size=-1):
            return json.dumps(self.data).encode()[:size]
    def opened(request, **kwargs):
        wire = json.loads(request.data)
        item = json.loads(wire["messages"][1]["content"])
        # The first dispatch marker and per-position pending row precede HTTP.
        assert (bundle / "private/secondary-dispatch.started.json").exists()
        state = judge.model_request_control.snapshot()
        assert state["attempts"][-1]["status"] == "pending"
        assert state["attempts"][-1]["operation_id"] == "e1b-judge-" + item["id"]
        row = {"id": item["id"], "content_sha256": item["content_sha256"],
               "labels": {key: 0 for key in item["content"]["rubric"]}}
        raw = json.dumps(row)
        envelope = receipt(judge, choices=[{"finish_reason": "stop", "message": {"content": raw}}])
        calls.append(wire)
        if change:
            change(envelope, row)
        return Response(envelope)
    stub_model_http(monkeypatch, opened)
    return calls


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unmocked network is forbidden")
    stub_model_http(monkeypatch, fail)


def test_forty_joint_calls_use_existing_guard_and_only_blind_content(tmp_path, monkeypatch):
    bundle, guard, judge, sha = setup(tmp_path)
    calls = stub(monkeypatch, judge, bundle)
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert result["status"] == "complete" and result["completed_positions"] == 40
    assert len(calls) == 40 and result["accounting"]["calls_used"] == 40
    assert result["ledger_id"] == guard.ledger_id and not guard.snapshot()["stopped"]
    packet, _ = evaluation.load_bundle(bundle)
    labels, _, _ = evaluation._labels(bundle, packet, "secondary")
    assert len(labels) == 40
    for wire in calls:
        assert len(wire["messages"]) == 2 and wire["model"] == judge.model
        content = wire["messages"][1]["content"]
        assert not any(token in content for token in ("case_id", '"arm"', "expected_preference", "private_mapping"))
    with pytest.raises(SecondaryStopped, match="previously_dispatched"):
        run_secondary(bundle, guard, judge, tmp_path / "different", expected_packet_sha256=sha)
    assert len(calls) == 40
    guard.close()


@pytest.mark.parametrize("bad", ["wrong_id", "unknown", "extra_reasoning", "bool", "duplicate", "broken"])
def test_invalid_first_reply_blocks_retry_and_preserves_dispatched_position(tmp_path, monkeypatch, bad):
    bundle, guard, judge, sha = setup(tmp_path)
    def change(envelope, row):
        if bad == "wrong_id":
            row["id"] = "f" * 32
        elif bad == "unknown":
            row = {"id": row["id"], "not_evaluable": True}
        elif bad == "extra_reasoning":
            row["analysis"] = "HIDDEN_TEXT_NOT_TO_SAVE"
        elif bad == "bool":
            row["labels"]["fabricates_unknown"] = False
        raw = json.dumps(row)
        if bad == "duplicate":
            raw = '{"id":"duplicate",' + raw[1:]
        elif bad == "broken":
            raw = "HIDDEN_TEXT_NOT_TO_SAVE malformed JSON"
        envelope["choices"][0]["message"]["content"] = raw
    calls = stub(monkeypatch, judge, bundle, change=change)
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert result["status"] == "not_evaluable" and result["completed_positions"] == 0
    assert len(calls) == 1 and guard.snapshot()["stopped"]
    first = bundle / "private/labels/secondary" / (result["failed_position"] + ".json")
    assert evaluation.read_json(first)["schema"] == "e1b_invalid_first_label_v1"
    assert "HIDDEN_TEXT_NOT_TO_SAVE" not in first.read_text()
    with pytest.raises(SecondaryStopped):
        run_secondary(bundle, guard, judge, tmp_path / "retry", expected_packet_sha256=sha)
    assert len(calls) == 1
    guard.close()


def test_unknown_cost_stops_after_one_and_does_not_treat_reply_as_label(tmp_path, monkeypatch):
    bundle, guard, judge, sha = setup(tmp_path)
    calls = stub(monkeypatch, judge, bundle, change=lambda envelope, _: envelope["usage"].pop("cost"))
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert len(calls) == 1 and result["accounting"]["unknown_cost_requests"] == 1
    assert result["completed_positions"] == 0 and result["status"] == "not_evaluable"
    guard.close()


@pytest.mark.parametrize("bad", ["fixture", "wrong_packet", "changed_sample", "unbound", "wrong_role"])
def test_preflight_rejects_without_sending_or_creating_dispatch_marker(tmp_path, bad):
    bundle, guard, judge, sha = setup(tmp_path, real=bad != "fixture")
    if bad == "wrong_packet":
        sha = "0" * 64
    elif bad == "changed_sample":
        path = bundle / "secondary-blind/sample.json"
        sample = evaluation.read_json(path)
        sample["ids"].reverse()
        path.write_bytes(evaluation.canonical(sample))
    elif bad == "unbound":
        judge.model_request_control = None
    elif bad == "wrong_role":
        judge.role = "memory"
    with pytest.raises(ValueError):
        run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert guard.snapshot()["calls_used"] == 0
    assert not (bundle / "private/secondary-dispatch.started.json").exists()
    guard.close()


def test_failed_first_save_stops_before_second_call_without_recovering(tmp_path, monkeypatch):
    bundle, guard, judge, sha = setup(tmp_path)
    calls = stub(monkeypatch, judge, bundle)
    original = evaluation.write_once
    def fail(path, value):
        if str(path).endswith(".first.json"):
            raise OSError("synthetic save failure")
        return original(path, value)
    monkeypatch.setattr(evaluation, "write_once", fail)
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert result["completed_positions"] == 0 and guard.snapshot()["stopped"]
    assert len(calls) == 1 and list((tmp_path / "out").glob("*.attempt.json"))
    guard.close()


@pytest.mark.parametrize("bad", ["no_receipt", "wrong_visible_hash"])
def test_labels_without_matching_actual_response_receipt_remain_invalid(tmp_path, monkeypatch, bad):
    bundle, guard, judge, sha = setup(tmp_path)
    calls = stub(monkeypatch, judge, bundle)
    if bad == "no_receipt":
        def no_receipt(messages):
            item = json.loads(messages[1]["content"])
            return json.dumps({"id": item["id"], "content_sha256": item["content_sha256"],
                               "labels": {name: 0 for name in item["content"]["rubric"]}})
        monkeypatch.setattr(judge, "complete", no_receipt)
    else:
        original = guard._record_receipt
        def wrong_hash(client, row, data):
            original(client, row, data)
            row["client_return_content_sha256"] = "0" * 64
        monkeypatch.setattr(guard, "_record_receipt", wrong_hash)
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert result["status"] == "not_evaluable" and result["completed_positions"] == 0
    assert len(calls) == (0 if bad == "no_receipt" else 1)
    assert guard.snapshot()["stopped"]
    guard.close()


def test_provider_outer_whitespace_matches_actual_client_return_without_false_failure(tmp_path, monkeypatch):
    bundle, guard, judge, sha = setup(tmp_path)
    def surround(envelope, row):
        envelope["choices"][0]["message"]["content"] = " \n" + json.dumps(row) + "\n "
    calls = stub(monkeypatch, judge, bundle, change=surround)
    result = run_secondary(bundle, guard, judge, tmp_path / "out", expected_packet_sha256=sha)
    assert result["status"] == "complete" and len(calls) == 40
    assert all(row["visible_content_sha256"] != row["client_return_content_sha256"]
               for row in result["accounting"]["attempts"])
    guard.close()
