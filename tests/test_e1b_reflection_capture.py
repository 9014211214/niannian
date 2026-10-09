"""Fictional E1b first-answer evidence; no network, retries or semantic scoring."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import urllib.request

import pytest

import aria.bench.e1b_runner as module
from aria.bench.e1b_runner import (
    E1bRunStopped, _ReflectionCapture, _reflection_product_diagnostic,
    local_fixture_factory,
)
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from tests.test_e1b_runner import setup_runner


VALID = '{"protocol_version":"2.0","reviews":[],"changes":[]}'


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reflection capture tests must never access the network")
    stub_model_http(monkeypatch, forbidden)


@pytest.fixture
def capture(tmp_path_factory):
    token = SyntheticEmotionExperiment.create(tmp_path_factory.mktemp("r"), "capture")
    app = local_fixture_factory(token)(token.root / "app")
    folder = token.root / "capture"
    folder.mkdir()
    value = _ReflectionCapture(app, folder, case_id="fixture", session=1,
        session_id="fixture-s1", source_sha256="b" * 64)
    try:
        yield value
    finally:
        app.call_guard = None
        app.close()


class Client:
    def __init__(self, result=VALID, *, error=False):
        self.result, self.error, self.calls = result, error, []
        self.last_diagnostic = {}

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        if self.error:
            raise ValueError("PRIVATE_EXCEPTION_MUST_NOT_BE_SAVED")
        return self.result


def receipt(capture, raw=VALID):
    return {"role": "reflection", "session_id": capture.session_id,
        "operation_id": f"e1b-review-{capture.session_id}", "request_sha256": "a" * 64,
        "visible_content_sha256": sha(raw), "client_return_content_sha256": sha(raw.strip()),
        "status": "received"}


def bind_guard(capture, rows):
    capture.app.call_guard = SimpleNamespace(snapshot=lambda: {"attempts": deepcopy(rows)})


def test_projection_keeps_requested_public_fields_and_no_unknown_names_or_hidden_text(capture):
    payload = {"protocol_version": "2.0", "reviews": [{"id": "r1", "reason": "公开证据说明",
        "evidence": [{"turn_index": 0, "evidence_quote": "本人原话", "PRIVATE_NAME": "PRIVATE_QUOTE"}],
        "PRIVATE_REVIEW": "PRIVATE_REVIEW_TEXT"}], "changes": [{"claim": "有范围的公开假设",
        "confidence": 0.2, "atomic_claim": {"conditions": "当前同类交流", "PRIVATE_ATOMIC": "PRIVATE_VALUE"},
        "scope": "<think>PRIVATE_TAGGED</think>"}], "PRIVATE_TOP": "PRIVATE_BODY"}
    raw = json.dumps(payload, ensure_ascii=False)
    client = Client(raw)
    assert capture.complete(client, [{"role": "user", "content": "fixture"}]) == raw
    first = read(capture.first_path)
    assert first["visible_response_sha256"] == sha(raw)
    assert first["request_hash_kind"] == "software_messages"
    assert first["public_projection_dropped_fields"] == 5
    assert first["public_projection"]["changes"][0]["scope"] is None
    assert first["public_projection"]["reviews"][0]["reason"] == "公开证据说明"
    assert "PRIVATE" not in capture.first_path.read_text(encoding="utf-8")
    assert len(client.calls) == 1


@pytest.mark.parametrize("raw,status", [
    ("PRIVATE_MALFORMED", "parse_error"),
    ('<think>PRIVATE_PREFIX</think>' + VALID, "parse_error"),
    (None, "non_text_response"),
    ('["PRIVATE_ARRAY"]', "not_object"),
])
def test_unparseable_first_is_hashed_without_saving_arbitrary_text(capture, raw, status):
    assert capture.complete(Client(raw), []) == raw
    first = read(capture.first_path)
    assert first["parse_status"] == status and first["public_projection"] is None
    assert first["visible_response_sha256"] == (sha(raw) if isinstance(raw, str) else None)
    assert "PRIVATE" not in capture.first_path.read_text(encoding="utf-8")


def test_no_field_repair_and_duplicate_count_is_diagnostic_only(capture):
    raw = '{"protocol_version":"2.0","protocol_version":"2.0","reviews":[]}'
    assert capture.complete(Client(raw), []) == raw
    first = read(capture.first_path)
    assert first["duplicate_key_count"] == 1
    assert first["known_fields_present"]["changes"] is False
    assert "changes" not in first["public_projection"]


def test_projection_bounds_nonfinite_and_large_values_without_mutating_raw(capture):
    raw = '{"protocol_version":"2.0","reviews":[],"changes":[{"confidence":1e309,"scope":"' + "X" * 241 + '"}]}'
    assert capture.complete(Client(raw), []) == raw
    first = read(capture.first_path)
    assert first["public_projection_dropped_fields"] == 2
    assert first["public_projection"]["changes"] == [{"confidence": None, "scope": None}]


def test_actual_wire_and_client_trimmed_sha_are_separate_and_first_is_exclusive(capture):
    wire_raw = " \n" + VALID + "\n "
    bind_guard(capture, [receipt(capture, wire_raw)])
    client = Client(wire_raw.strip())
    assert capture.complete(client, []) == wire_raw.strip()
    first = read(capture.first_path)
    assert first["request_hash_kind"] == "actual_wire" and first["request_sha256"] == "a" * 64
    assert first["wire_visible_content_sha256"] == sha(wire_raw)
    assert first["visible_response_sha256"] == sha(wire_raw.strip())
    original = capture.first_path.read_bytes()
    with pytest.raises(E1bRunStopped, match="already_consumed"):
        capture.complete(client, [])
    assert capture.first_path.read_bytes() == original and len(client.calls) == 1


@pytest.mark.parametrize("bad", ["operation", "session", "role", "request_sha", "response_sha", "duplicate", "failed"])
def test_missing_or_mismatched_wire_identity_stops_but_retains_first(capture, bad):
    row = receipt(capture)
    fields = {"operation": ("operation_id", "another"), "session": ("session_id", "another"),
        "role": ("role", "memory"), "request_sha": ("request_sha256", None),
        "response_sha": ("client_return_content_sha256", "c" * 64), "failed": ("status", "failed")}
    if bad in fields:
        key, value = fields[bad]
        row[key] = value
    bind_guard(capture, [row, row] if bad == "duplicate" else [row])
    with pytest.raises(E1bRunStopped, match="identity_unavailable"):
        capture.complete(Client(), [])
    assert read(capture.first_path)["request_identity_missing"] is True
    assert capture.capture_error == "reflection_request_identity_unavailable"


def test_strict_client_failure_has_only_safe_diagnostic_and_receipt_hash(capture):
    bind_guard(capture, [receipt(capture)])
    client = Client(error=True)
    client.last_diagnostic = {"error_type": "invalid_reflection_json_schema",
        "output_kind": "invalid_reflection_json_schema", "finish_reason": "stop", "http_status": 200,
        "PRIVATE_NAME": "PRIVATE_REASONING"}
    with pytest.raises(ValueError):
        capture.complete(client, [])
    first = read(capture.first_path)
    assert first["public_projection"] is None and first["parse_status"] == "call_failed_or_unknown"
    assert first["response_hash_source"] == "receipt_only"
    assert first["visible_response_sha256"] == sha(VALID)
    assert first["wire_visible_content_sha256"] == sha(VALID)
    assert first["client_diagnostic"]["error_type"] == "invalid_reflection_json_schema"
    assert "PRIVATE" not in capture.first_path.read_text(encoding="utf-8")
    capture.finish(SimpleNamespace(last_status="failed", last_error_type="model_call_failed",
        last_error_detail="model_call_failed"))
    assert read(capture.outcome_path)["last_error_type"] == "model_call_failed"


def test_outcome_saves_only_safe_codes_and_commits_first_file_hash(capture):
    capture.complete(Client(), [])
    updater = SimpleNamespace(last_status="failed", last_error_type="invalid_evidence_review_protocol",
        last_error_detail="review_quote_not_grounded_in_current_user",
        last_response_transport={"prompt_contract_version": "2.3", "format": "bare_json",
            "json_status": "parsed", "wrapper_removed": False, "content_repaired": False, "PRIVATE": "PRIVATE"},
        last_application_summary={"proposal_count": 1, "rejected_count": 1, "PRIVATE": "PRIVATE"})
    capture.finish(updater)
    outcome = read(capture.outcome_path)
    assert outcome["last_error_detail"] == "review_quote_not_grounded_in_current_user"
    assert outcome["first_sha256"] == hashlib.sha256(capture.first_path.read_bytes()).hexdigest()
    assert outcome["application_counts"] == {"proposal_count": 1, "rejected_count": 1}
    assert "PRIVATE" not in capture.outcome_path.read_text(encoding="utf-8")
    assert _reflection_product_diagnostic(SimpleNamespace(last_status=[],
        last_error_type="PRIVATE", last_error_detail="PRIVATE")) == {
            "product_status": "unknown", "last_error_type": None, "last_error_detail": None}
    with pytest.raises(FileExistsError):
        capture.finish(updater)


@pytest.mark.parametrize("stage", ["attempt", "first", "product"])
def test_fsync_failure_prevents_send_or_next_stage_without_replacing_original(capture, monkeypatch, stage):
    client = Client()
    original_fsync = module.os.fsync
    count = 0
    def fail_at_stage(fd):
        nonlocal count
        count += 1
        if count == {"attempt": 1, "first": 2, "product": 3}[stage]:
            raise OSError("PRIVATE_FS_ERROR")
        return original_fsync(fd)
    monkeypatch.setattr(module.os, "fsync", fail_at_stage)
    with pytest.raises(OSError):
        capture.complete(client, [])
        capture.finish(SimpleNamespace(last_status="ok"))
    assert len(client.calls) == (0 if stage == "attempt" else 1)
    path = {"attempt": capture.attempt_path, "first": capture.first_path, "product": capture.outcome_path}[stage]
    original = path.read_bytes()
    assert original and capture.capture_error == "reflection_capture_save_failed"
    with pytest.raises(E1bRunStopped):
        capture.complete(client, [])
    assert path.read_bytes() == original


def test_capture_cannot_target_normal_app_or_outside_experiment(capture, tmp_path):
    kwargs = dict(case_id="c", session=1, session_id="s", source_sha256="b" * 64)
    with pytest.raises(ValueError, match="isolated"):
        _ReflectionCapture(SimpleNamespace(), capture.directory, **kwargs)
    with pytest.raises(ValueError, match="isolated"):
        _ReflectionCapture(capture.app, tmp_path, **kwargs)


@pytest.mark.parametrize("raw,error_type,error_detail", [
    ('{"protocol_version":"2.0","reviews":[]}', "invalid_json_response", "changes_array_required"),
    ("PRIVATE_NOT_JSON", "invalid_json_response", "invalid_json_response"),
    ('{"protocol_version":"WRONG","reviews":[],"changes":[]}',
        "invalid_evidence_review_protocol", "review_protocol_version_required"),
])
def test_runner_retains_updater_failure_and_never_continues_to_second_session(
    tmp_path_factory, raw, error_type, error_detail,
):
    def factory(token, records):
        base = local_fixture_factory(token, recordings=records)
        def create(path):
            app = base(path)
            client = app.clients.reflection
            def complete(messages):
                client.calls.append(deepcopy(messages))
                return raw
            client.complete = complete
            return app
        return create
    runner, records = setup_runner(tmp_path_factory.mktemp("i"), factory=factory)
    case_id = runner.suite.items[0].id
    with pytest.raises(E1bRunStopped):
        runner.run_case(case_id)
    folder = runner.root / "main" / case_id
    outcome = read(folder / "reflection-s1.product.json")
    assert outcome["product_status"] == "failed"
    assert outcome["last_error_type"] == error_type and outcome["last_error_detail"] == error_detail
    assert outcome["first_available"] is True
    assert sum(len(clients.online.calls) for _, clients in records) == 12
    assert sum(len(clients.memory.calls) for _, clients in records) == 1
    assert sum(len(clients.reflection.calls) for _, clients in records) == 1
    assert not (folder / "reflection-s2.attempt.json").exists()
    assert not (folder / "report.json").exists()
    assert "PRIVATE" not in (folder / "reflection-s1.first.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("stage", ["attempt", "first", "product"])
def test_runner_capture_write_failure_stops_later_calls_and_keeps_partial(
    tmp_path_factory, monkeypatch, stage,
):
    runner, records = setup_runner(tmp_path_factory.mktemp("i"))
    original = module._exclusive_json
    def fail(path, value):
        if path.name == f"reflection-s1.{stage}.json":
            path.write_text("partial retained", encoding="utf-8")
            raise OSError("PRIVATE_WRITE_ERROR")
        return original(path, value)
    monkeypatch.setattr(module, "_exclusive_json", fail)
    case_id = runner.suite.items[0].id
    with pytest.raises(E1bRunStopped):
        runner.run_case(case_id)
    folder = runner.root / "main" / case_id
    assert (folder / f"reflection-s1.{stage}.json").read_text(encoding="utf-8") == "partial retained"
    assert sum(len(clients.online.calls) for _, clients in records) == 12
    assert sum(len(clients.memory.calls) for _, clients in records) == 1
    assert sum(len(clients.reflection.calls) for _, clients in records) == (0 if stage == "attempt" else 1)
    assert runner.stopped and not (folder / "report.json").exists()
