"""Independent formation denominators and immutable local labels, no models."""
from copy import deepcopy
import hashlib
import json
import urllib.request

import pytest

from aria.bench.e1b_cases import load_e1b_cases
from scripts.e1b_evaluation import canonical, digest, draft_protocol, read_json, write_once
from scripts.e1b_formation import (
    OMITTED, RULES, aggregate, export_packet, import_labels, label_template,
    load_bundle, main,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("formation tooling must not use network or model")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def proposal(case):
    return {"do": case.expected_preference.do, "scope": case.expected_preference.scope,
            "quotes": [{"turn_index": case.pref_turn, "text": case.current_input(2, case.pref_turn)}]}


def write_capture(directory, case, suite, arrays=None, *, product_status="ok", dropped=0):
    arrays = [[], [proposal(case)]] if arrays is None else arrays
    rows = []
    for number, raw in enumerate(arrays, 1):
        first = {"session": number, "request_sha256": str(number) * 64, "request_hash_kind": "software_messages",
                 "visible_response_sha256": "3" * 64, "field_present": True, "raw_proposals": raw,
                 "parse_status": "present_array" if type(raw) is list else "null_field" if raw is None else "not_array",
                 "public_projection_dropped_fields": dropped}
        outcome = {"session": number, "product_status": product_status, "accepted_preferences": [],
                   "accepted_count": 0, "rejected_codes": ["product_rejects_even_a_correct_candidate"],
                   "private_result": "never_show_product_result"}
        first_path, outcome_path = directory / f"memory-s{number}.first.json", directory / f"memory-s{number}.product.json"
        write_once(first_path, first)
        write_once(outcome_path, outcome)
        rows.append({**first, "complete": first["parse_status"] == "present_array" and product_status == "ok",
                     "product_outcome": outcome, "first_attempt_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
                     "product_outcome_sha256": hashlib.sha256(outcome_path.read_bytes()).hexdigest()})
    write_once(directory / "raw-proposals.json", {"schema": "e1b_raw_proposals_v2", "case_id": case.id,
               "source_sha256": suite.source_sha256, "sessions": rows})


def make_run(tmp_path, *, count=12, arrays_for=None, product_status="ok", dropped=0):
    suite = load_e1b_cases()
    protocol = draft_protocol(suite, phase="main", case_ids=[case.id for case in suite.items[:count]])
    run = tmp_path / "run"
    for case in suite.items[:count]:
        write_capture(run / "main" / case.id, case, suite,
                      None if arrays_for is None else arrays_for(case), product_status=product_status, dropped=dropped)
    return suite, protocol, run


def bundle(tmp_path, **kwargs):
    suite, protocol, run = make_run(tmp_path, **kwargs)
    output = tmp_path / "bundle"
    exported = export_packet(run, output, protocol)
    return output, exported, suite, protocol, run


def labels_for(root, callback=None):
    packet, key = load_bundle(root)
    submission = label_template(packet)
    submission["scorer"]["id"] = "independent_software_fixture_scorer"
    items = {item["id"]: item for item in packet["items"]}
    for row in submission["labels"]:
        item = items[row["id"]]
        row["labels"] = {"source_support": 1, "expected_match": 1, "authorization": 0, "personality_label": 0}
        if callback:
            callback(row["labels"], item, key["mapping"][row["id"]])
    return submission


def test_blind_content_full_raw_denominators_and_product_rejection_separated(tmp_path):
    def arrays(case):
        return [[], [proposal(case)] * 14 + [None, "a malformed candidate"]]
    out, exported, suite, _, _ = bundle(tmp_path, arrays_for=arrays)
    packet, key = load_bundle(out)
    assert exported["observed_raw_candidates"] == 12 * 16
    assert len(packet["items"]) == 12 * 16
    assert key["inventory"][0]["raw_capture"]["sessions"][1]["product_outcome"]["accepted_count"] == 0
    text = canonical(packet).decode()
    for hidden in ("A0", "A1", "e1b_a1", "accepted_count", "rejected_codes", "never_show_product_result", "case_id"):
        assert hidden not in text
    assert all(item["content_sha256"] == digest(item["content"]) for item in packet["items"])
    assert suite.items[0].current_input(3, 0) not in text
    def label(labels, item, _):
        if not isinstance(item["content"]["proposal"], dict):
            labels.update(source_support=0, expected_match=0)
    import_labels(out, labels_for(out, label))
    result = aggregate(out)
    assert result["proposed"] == 12 and result["proposal_rate"] == 1
    assert result["candidate_denominator"] == 192
    assert result["false_extractions"] == 24
    assert result["verbatim_candidates"] == 168 and result["verbatim_quote_rate"] == 168 / 192
    assert result["mechanical_formation_gate"] == "fail"
    assert result["formation_gate"] == "not_registered" and not result["effect_claim_eligible"]


@pytest.mark.parametrize("product_status", ["ok", "failed"])
def test_correct_raw_candidate_is_not_failed_because_product_rejected_or_failed(tmp_path, product_status):
    out, *_ = bundle(tmp_path, product_status=product_status)
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["proposed"] == 12 and result["false_extractions"] == 0
    assert result["mechanical_formation_gate"] == "pass"
    assert result["product_results_used_for_model_labels"] is False


def test_legal_empty_arrays_are_zero_proposals_not_missing_or_false_and_zero_quote_rate_is_null(tmp_path):
    out, exported, *_ = bundle(tmp_path, arrays_for=lambda case: [[], []])
    assert exported["available_label_positions"] == 0
    result = aggregate(out)
    assert result["status"] == "local_labels_complete"
    assert result["proposed"] == 0 and result["proposal_rate"] == 0
    assert result["candidate_denominator"] == 0 and result["false_extractions"] == 0
    assert result["verbatim_candidates"] == 0 and result["verbatim_quote_rate"] is None
    assert result["mechanical_formation_gate"] == "fail"


@pytest.mark.parametrize("failure", ["missing_capture", "missing_first", "corrupt", "duplicate_key", "changed_first", "one_session", "null_array", "object_array", "absent_array"])
def test_incomplete_originals_never_become_empty_success_or_smaller_denominator(tmp_path, failure):
    suite, protocol, run = make_run(tmp_path, arrays_for=(lambda c: [[], None] if c.id == "e1b_a1" else [[], [proposal(c)]])
                                    if failure == "null_array" else
                                    (lambda c: [[], {}] if c.id == "e1b_a1" else [[], [proposal(c)]])
                                    if failure == "object_array" else None)
    directory = run / "main" / suite.items[0].id
    capture_path = directory / "raw-proposals.json"
    if failure == "missing_capture":
        capture_path.unlink()
    elif failure == "missing_first":
        (directory / "memory-s1.first.json").unlink()
    elif failure == "corrupt":
        capture_path.write_text('{"truncated":', encoding="utf-8")
    elif failure == "duplicate_key":
        capture_path.write_text('{"schema":"x","schema":"e1b_raw_proposals_v2"}', encoding="utf-8")
    elif failure == "changed_first":
        (directory / "memory-s1.first.json").write_text('{}', encoding="utf-8")
    elif failure == "one_session":
        capture = read_json(capture_path)
        capture["sessions"] = capture["sessions"][:1]
        capture_path.write_bytes(canonical(capture))
    elif failure == "absent_array":
        first_path = directory / "memory-s2.first.json"
        first = read_json(first_path)
        first.update(field_present=False, parse_status="missing_field", raw_proposals=None)
        first_path.write_bytes(canonical(first))
        capture = read_json(capture_path)
        capture["sessions"][1].update(first)
        capture["sessions"][1].update(complete=False, first_attempt_sha256=hashlib.sha256(first_path.read_bytes()).hexdigest())
        capture_path.write_bytes(canonical(capture))
    out = tmp_path / "bundle"
    export_packet(run, out, protocol)
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["planned_expected_preferences"] == 12
    assert result["status"] == "not_evaluable" and result["proposal_rate"] is None
    assert result["candidate_denominator"] is None and result["false_extractions"] is None
    assert list(result["raw_problem_cases"]) == [suite.items[0].id]


@pytest.mark.parametrize("failure", ["wrong_sha", "unknown", "bool", "missing_field", "malformed_supported"])
def test_invalid_first_labels_are_retained_exclusively_and_never_imputed(tmp_path, failure):
    out, *_ = bundle(tmp_path, arrays_for=(lambda case: [[], [None]]) if failure == "malformed_supported" else None)
    submission = labels_for(out)
    row = submission["labels"][0]
    if failure == "wrong_sha":
        row["content_sha256"] = "0" * 64
    elif failure == "unknown":
        row["labels"]["source_support"] = None
    elif failure == "bool":
        row["labels"]["source_support"] = True
    elif failure == "missing_field":
        row["labels"].pop("authorization")
    result = import_labels(out, submission)
    assert result["saved_invalid_first_labels"] >= 1
    first_path = out / "private/labels" / f"{row['id']}.json"
    initial = first_path.read_bytes()
    with pytest.raises(FileExistsError):
        import_labels(out, labels_for(out))
    assert first_path.read_bytes() == initial
    result = aggregate(out)
    assert result["status"] == "not_evaluable" and result["false_extractions"] is None


@pytest.mark.parametrize("failure", ["author", "not_independent", "placeholder"])
def test_author_and_unidentified_scorer_are_rejected(tmp_path, failure):
    out, *_ = bundle(tmp_path)
    submission = labels_for(out)
    if failure == "author":
        submission["scorer"]["authored_cases"] = True
    elif failure == "not_independent":
        submission["scorer"]["independent"] = False
    else:
        submission["scorer"]["id"] = "REPLACE_INDEPENDENT_SCORER"
    with pytest.raises(ValueError, match="independent_non_author"):
        import_labels(out, submission)
    assert not list((out / "private/labels").glob("*.json"))


@pytest.mark.parametrize("quotes", [None, [], [{"turn_index": True, "text": "我"}], [{"turn_index": 99, "text": "我"}],
                                   [{"turn_index": 0, "text": "not in user source"}], [{"turn_index": 0, "text": ""}]])
def test_verbatim_failure_is_diagnostic_not_independent_semantic_failure(tmp_path, quotes):
    def arrays(case):
        raw = proposal(case)
        if quotes is None:
            raw.pop("quotes")
        else:
            raw["quotes"] = quotes
        return [[], [raw]]
    out, *_ = bundle(tmp_path, arrays_for=arrays)
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["candidate_denominator"] == 12 and result["verbatim_quote_rate"] == 0
    assert result["false_extractions"] == 0 and result["proposed"] == 12
    assert result["mechanical_formation_gate"] == "pass"
    assert RULES["verbatim_role"].startswith("diagnostic_only")


def test_session_one_cannot_quote_future_session_two_and_both_sessions_count(tmp_path):
    out, *_ = bundle(tmp_path, arrays_for=lambda case: [[proposal(case)], [proposal(case)]])
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["candidate_denominator"] == 24
    assert result["verbatim_candidates"] == 12 and result["verbatim_quote_rate"] == .5


def test_error_categories_overlap_only_once_and_unmatched_supported_candidate_is_not_error(tmp_path):
    out, *_ = bundle(tmp_path)
    def labels(row, _, loc):
        if loc["case_id"] == "e1b_a1":
            row.update(authorization=1, personality_label=1, source_support=0, expected_match=0)
        elif loc["case_id"] == "e1b_a2":
            row.update(expected_match=0)
    import_labels(out, labels_for(out, labels))
    result = aggregate(out)
    assert result["proposed"] == 10 and result["false_extractions"] == 1
    assert result["false_category_counts"] == {"authorization": 1, "personality_label": 1, "unsupported": 1}
    assert result["mechanical_formation_gate"] == "fail"


@pytest.mark.parametrize("matched,expected", [(9, "fail"), (10, "pass")])
def test_original_twelve_case_eighty_percent_boundary(tmp_path, matched, expected):
    out, _, _, protocol, _ = bundle(tmp_path)
    allowed = set(protocol["case_ids"][:matched])
    import_labels(out, labels_for(out, lambda row, item, loc: row.update(expected_match=int(loc["case_id"] in allowed))))
    result = aggregate(out)
    assert result["false_extractions"] == 0 and result["proposed"] == matched
    assert result["mechanical_formation_gate"] == expected
    assert result["formation_gate"] == "not_registered" and not result["effect_claim_eligible"]


def test_ten_cases_never_reuse_twelve_gate(tmp_path):
    out, *_ = bundle(tmp_path, count=10)
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["proposed"] == 10 and result["proposal_rate"] == 1
    assert result["mechanical_formation_gate"] is None
    assert result["formation_gate"] == "unregistered_denominator_rule"


@pytest.mark.parametrize("tamper", ["source", "packet", "label", "label_seal_missing"])
def test_post_export_integrity_failures_are_not_evaluable(tmp_path, tamper):
    out, _, suite, _, run = bundle(tmp_path)
    submission = labels_for(out)
    import_labels(out, submission)
    if tamper == "source":
        target = run / "main" / suite.items[0].id / "memory-s2.first.json"
    elif tamper == "packet":
        target = out / "blind/packet.json"
    else:
        target = out / "private/labels" / f"{submission['labels'][0]['id']}.json"
        if tamper == "label_seal_missing":
            target.with_suffix(".seal.json").unlink()
    if tamper != "label_seal_missing":
        target.write_text('{}', encoding="utf-8")
    result = aggregate(out)
    assert result["status"] == "not_evaluable" and result["formation_gate"] == "not_evaluable"


def test_privacy_projection_extra_fields_are_safe_but_lost_required_content_is_ne(tmp_path):
    out, *_ = bundle(tmp_path / "complete", dropped=3)
    import_labels(out, labels_for(out))
    assert aggregate(out)["mechanical_formation_gate"] == "pass"
    def arrays(case):
        value = proposal(case)
        value["do"] = OMITTED
        return [[], [value]]
    out, result, *_ = bundle(tmp_path / "lost", arrays_for=arrays, dropped=1)
    assert result["observed_raw_candidates"] == 12 and result["available_label_positions"] == 0
    result = aggregate(out)
    assert result["status"] == "not_evaluable" and result["observed_raw_candidates"] == 12


def test_missing_label_is_ne_and_later_batch_error_does_not_erase_first_unknown(tmp_path):
    out, *_ = bundle(tmp_path)
    assert aggregate(out)["missing_or_invalid_labels"] == 12
    submission = labels_for(out)
    row = deepcopy(submission["labels"][0])
    row["labels"]["source_support"] = None
    submission["labels"] = [row, deepcopy(row)]
    with pytest.raises(ValueError, match="retained"):
        import_labels(out, submission)
    assert read_json(out / "private/labels" / f"{row['id']}.json")["schema"] == "e1b_formation_invalid_first_label_v1"
    assert aggregate(out)["missing_or_invalid_labels"] == 12


def test_cli_local_export_import_and_aggregate_does_not_register_or_call_models(tmp_path, capsys):
    _, protocol, run = make_run(tmp_path)
    protocol_path, bundle_path = tmp_path / "protocol.json", tmp_path / "bundle"
    write_once(protocol_path, protocol)
    assert main(["export", "--run-root", str(run), "--protocol", str(protocol_path), "--output", str(bundle_path)]) == 0
    submission_path = tmp_path / "labels.json"
    write_once(submission_path, labels_for(bundle_path))
    assert main(["import-labels", "--bundle", str(bundle_path), "--submission", str(submission_path)]) == 0
    output = tmp_path / "result.json"
    assert main(["aggregate", "--bundle", str(bundle_path), "--output", str(output)]) == 0
    result = read_json(output)
    assert result["new_model_calls"] == 0 and not result["effect_claim_eligible"]
    assert len(capsys.readouterr().out.splitlines()) == 3
    with pytest.raises(FileExistsError):
        main(["aggregate", "--bundle", str(bundle_path), "--output", str(output)])


def test_nonempty_do_missing_candidate_cannot_be_labeled_supported(tmp_path):
    out, *_ = bundle(tmp_path, arrays_for=lambda case: [[], [{"quotes": proposal(case)["quotes"]}]])
    result = import_labels(out, labels_for(out))
    assert result["saved_invalid_first_labels"] == 12
    assert aggregate(out)["status"] == "not_evaluable"
