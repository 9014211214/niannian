"""Approved selection and two-step capture; no live model or credential access."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import urllib.request

import pytest

from aria.bench.e1b_cases import load_e1b_cases, select_prescreen_balanced
from aria.bench.e1b_runner import E1bRunStopped, local_fixture_factory
from aria.bench.e1br_runner import ProductReflectionCapture
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.reflection_json import reflection_body_system_prompt, reflection_correction_message
from scripts import e1b_evaluation as evaluation

VALID = '{"protocol_version":"2.0","reviews":[],"changes":[]}'
SELECTED = [f"e1b_{g}{n}" for g in "abcdef" for n in (1, 2)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("no network in E1b-R contract tests"))


def test_fixed_balanced_selection_uses_neither_magnitude_nor_a1():
    suite = load_e1b_cases()
    first = {case.id: (True, False, False) for case in suite.items}
    second = {case.id: (False, True, True) for case in suite.items}
    assert select_prescreen_balanced(suite, first) == select_prescreen_balanced(suite, second)
    assert select_prescreen_balanced(suite, first)["case_ids"] == SELECTED


@pytest.mark.parametrize("change", ("missing", "not_eligible", "not_bool"))
def test_selection_rejects_unregistered_evidence_changes(change):
    suite = load_e1b_cases()
    labels = {case.id: (True, False, False) for case in suite.items}
    if change == "missing":
        labels.pop("e1b_f3")
    else:
        labels["e1b_f3"] = (False, False, False) if change == "not_eligible" else (1, False, False)
    with pytest.raises(ValueError):
        select_prescreen_balanced(suite, labels)


def test_new_evaluation_revision_keeps_denominators_and_old_contract():
    suite = load_e1b_cases()
    old = evaluation.draft_protocol(suite, phase="main", case_ids=SELECTED)
    new = evaluation.draft_protocol(suite, phase="main", case_ids=SELECTED, revision="e1br_v1")
    assert old["secondary_positions"] == 40 and new["secondary_positions"] == 0
    assert new["planned_denominator"] == 12 and new["min_improved"] == 4 and new["max_worse"] == 0
    evaluation.validate_protocol(new, suite)
    new["secondary_positions"] = 40
    with pytest.raises(ValueError, match="layout"):
        evaluation.validate_protocol(new, suite)


@pytest.fixture
def capture(tmp_path):
    token = SyntheticEmotionExperiment.create(tmp_path, "e1br-capture")
    app = local_fixture_factory(token)(token.root / "app")
    folder = token.root / "capture"
    folder.mkdir()
    value = ProductReflectionCapture(app, folder, case_id="fixture", session=1,
        session_id="fixture-s1", source_sha256="b" * 64)
    yield value
    app.call_guard = None
    app.close()


def messages(correction=False):
    value = [{"role": "system", "content": reflection_body_system_prompt()},
             {"role": "user", "content": "fictional transcript"}]
    if correction:
        value.append(reflection_correction_message([{"path": [], "type": "json_syntax"}]))
    return value


def test_initial_bad_shape_and_correction_are_distinct_immutable_firsts(capture):
    bodies = iter(("PRIVATE invalid JSON", VALID))
    client = SimpleNamespace(complete=lambda value: next(bodies), last_diagnostic={})
    assert capture.complete(client, messages()) == "PRIVATE invalid JSON"
    initial = capture.initial.first_path.read_bytes()
    assert capture.complete(client, messages(True)) == VALID
    capture.finish(SimpleNamespace(last_status="ok"))
    assert capture.initial.first_path.read_bytes() == initial
    assert b"PRIVATE" not in initial
    value = json.loads(capture.initial.outcome_path.read_text(encoding="utf-8"))
    assert value["format_correction_count"] == 1
    assert [row["product_step"] for row in value["product_steps"]] == ["initial", "format_correction"]
    with pytest.raises(E1bRunStopped, match="order"):
        capture.complete(client, messages(True))


def test_correction_cannot_be_first(capture):
    with pytest.raises(E1bRunStopped, match="order"):
        capture.complete(SimpleNamespace(), messages(True))


def test_failed_no_body_is_retained_but_never_format_corrected(capture):
    def fail(_):
        raise ValueError("PRIVATE failure")
    with pytest.raises(ValueError):
        capture.complete(SimpleNamespace(complete=fail, last_diagnostic={}), messages())
    with pytest.raises(E1bRunStopped, match="requires_first_model_content"):
        capture.complete(SimpleNamespace(), messages(True))
    capture.finish(SimpleNamespace(last_status="failed"))
    assert json.loads(capture.initial.outcome_path.read_text())["product_status"] == "failed"


def test_capture_binds_all_infrastructure_attempts_to_unique_body(capture):
    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()
    rows = [{"role": "reflection", "session_id": "fixture-s1", "operation_id": "e1b-review-fixture-s1",
             "product_step": "initial", "request_id": f"fixture-{n}", "request_sha256": "a" * 64,
             "status": "failed" if n == 1 else "received"} for n in (1, 2)]
    rows[-1].update(visible_content_sha256=sha(VALID), client_return_content_sha256=sha(VALID))
    capture.initial.app.call_guard = SimpleNamespace(snapshot=lambda: {"attempts": deepcopy(rows)})
    assert capture.complete(SimpleNamespace(complete=lambda _: VALID, last_diagnostic={}), messages()) == VALID
    assert capture.initial.first["physical_request_ids"] == ["fixture-1", "fixture-2"]
    assert capture.initial.first["request_id"] == "fixture-2"


def test_capture_rejects_retry_after_content(capture):
    digest = hashlib.sha256(VALID.encode()).hexdigest()
    first = {"role": "reflection", "session_id": "fixture-s1", "operation_id": "e1b-review-fixture-s1",
             "product_step": "initial", "request_id": "first", "request_sha256": "a" * 64,
             "status": "received", "visible_content_sha256": digest, "client_return_content_sha256": digest}
    later = {**first, "request_id": "later", "status": "failed"}
    later.pop("visible_content_sha256")
    later.pop("client_return_content_sha256")
    capture.initial.app.call_guard = SimpleNamespace(snapshot=lambda: {"attempts": [first, later]})
    with pytest.raises(E1bRunStopped, match="identity"):
        capture.complete(SimpleNamespace(complete=lambda _: VALID, last_diagnostic={}), messages())
