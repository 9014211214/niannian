"""Blinding, immutable first judgments and fixed-denominator local arithmetic."""
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.bench.e1b_cases import load_e1b_cases
from scripts.e1b_evaluation import (
    aggregate, canonical, digest, draft_protocol, export_packet, import_labels,
    label_template, load_bundle, prepare_secondary, read_json, validate_protocol,
    write_once,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("local evaluation must not contact a model or network")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def make_run(tmp_path, phase="main", count=12):
    suite = load_e1b_cases()
    ids = [c.id for c in suite.items] if phase == "prescreen" else [c.id for c in suite.items[:count]]
    protocol = draft_protocol(suite, phase=phase, case_ids=ids)
    run = tmp_path / "run"
    write_once(run / "emotion-plan.json", {"source_sha256": suite.source_sha256, "synthetic": True})
    for case in suite.items:
        if case.id not in ids:
            continue
        branches = []
        for arm in protocol["arms"]:
            for repeat in (1, 2, 3):
                turns = [{"text": case.current_input(3, turn), "response": f"本地规则测试回复 {turn}。",
                          "status": "completed", "run_id": "never-in-blind", "private": "never-in-blind"}
                         for turn in range(10)]
                branches.append({"arm": arm, "repeat": repeat, "state": {"turns": turns},
                                 "scored_replies": [{"turn": turn, "response": turns[turn]["response"]} for turn in (7, 8, 9)]})
        write_once(run / phase / case.id / "report.json", {
            "case_id": case.id, "phase": phase, "synthetic_emotion": True,
            "evidence_kind": "software_fixture", "branches": branches,
            "shared": {"raw_secret": "never-in-blind", "profile": {"stated_preferences": [{"id": "accepted-only"}]}}})
    return suite, protocol, run


def bundle(tmp_path, phase="main", count=12):
    suite, protocol, run = make_run(tmp_path, phase, count)
    output = tmp_path / "bundle"
    result = export_packet(run, output, protocol)
    return output, result, suite, protocol, run


def labels_for(output, condition=None, scorer="software_fixture_primary"):
    packet, key = load_bundle(output)
    submission = label_template(packet)
    submission["scorer"]["id"] = scorer
    for row in submission["labels"]:
        row["labels"] = {k: 0 for k in row["labels"]}
        if condition and condition(key["mapping"][row["id"]]):
            row["labels"]["fabricates_unknown"] = 1
    return submission


def test_blind_projection_excludes_arms_private_history_and_future(tmp_path):
    output, result, suite, protocol, _ = bundle(tmp_path)
    packet, key = load_bundle(output)
    assert result == {"packet_sha256": digest(packet), "planned_positions": 216,
                      "available_positions": 216, "missing_positions": 0,
                      "registered": False, "formation": "not_evaluable"}
    public = canonical(packet).decode()
    assert all(term not in public for term in ("A0", "A1", "e1b_a1", "run_id", "never-in-blind", "expected_preference"))
    assert len({x["id"] for x in packet["items"]}) == 216
    for item in packet["items"]:
        loc = key["mapping"][item["id"]]
        case = next(c for c in suite.items if c.id == loc["case_id"])
        content = item["content"]
        assert len(content["dialogue"]) == 2 * loc["turn"] + 1
        assert case.current_input(1, 0) not in canonical(content).decode()
        for future in range(loc["turn"] + 1, 10):
            # The author's rubric may quote a question as an applicability
            # example. Only actual conversation history is time-bounded.
            assert case.current_input(3, future) not in canonical(content["dialogue"]).decode()
        assert item["content_sha256"] == digest(content)
    assert protocol["status"] == "draft"
    with pytest.raises(FileExistsError):
        export_packet(tmp_path / "run", output, protocol)


@pytest.mark.parametrize("change", ["arm_text", "wrong_script", "different_score", "duplicate_branch"])
def test_export_rejects_leaks_and_inconsistent_source_without_silent_redaction(tmp_path, change):
    _, protocol, run = make_run(tmp_path)
    path = run / "main/e1b_a1/report.json"
    report = read_json(path)
    branch = report["branches"][0]
    if change == "arm_text":
        branch["state"]["turns"][7]["response"] = "我当前属于 A1。"
        branch["scored_replies"][0]["response"] = "我当前属于 A1。"
    elif change == "wrong_script":
        branch["state"]["turns"][0]["text"] = "未来台词替换"
    elif change == "different_score":
        branch["scored_replies"][0]["response"] = "替换评分回复"
    else:
        report["branches"].append(deepcopy(branch))
    path.write_bytes(canonical(report))
    with pytest.raises(ValueError):
        export_packet(run, tmp_path / "bad", protocol)
    assert not (tmp_path / "bad").exists()


def test_missing_generation_and_corrupt_or_missing_labels_never_become_zero(tmp_path):
    _, protocol, run = make_run(tmp_path)
    (run / "main/e1b_a1/report.json").unlink()
    out = tmp_path / "bundle"
    exported = export_packet(run, out, protocol)
    assert exported["missing_positions"] == 18
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["planned_case_denominator"] == 12 and result["planned_reply_positions"] == 216
    assert result["missing_generation_positions"] == 18
    assert result["main_gate"] == "not_evaluable" and "improved_cases" not in result


@pytest.mark.parametrize("invalid", ["bool", "missing_dimension", "null", "wrong_sha", "author", "non_applicable"])
def test_label_schema_identity_applicability_and_author_separation(tmp_path, invalid):
    out, *_ = bundle(tmp_path)
    submission = labels_for(out)
    row = submission["labels"][0]
    if invalid == "bool":
        row["labels"]["fabricates_unknown"] = False
    elif invalid == "missing_dimension":
        row["labels"].pop("fabricates_unknown")
    elif invalid == "null":
        row["labels"]["fabricates_unknown"] = None
    elif invalid == "wrong_sha":
        row["content_sha256"] = "0" * 64
    elif invalid == "author":
        submission["scorer"]["authored_cases"] = True
    else:
        packet, _ = load_bundle(out)
        item = next(i for i in packet["items"] if any(not r["applicable"] for r in i["content"]["rubric"].values()))
        row = next(r for r in submission["labels"] if r["id"] == item["id"])
        name = next(k for k, r in item["content"]["rubric"].items() if not r["applicable"])
        row["labels"][name] = 1
    if invalid == "author":
        with pytest.raises(ValueError):
            import_labels(out, submission)
        assert not list((out / "private/labels").rglob("*.json"))
    else:
        result = import_labels(out, submission)
        assert result["saved_invalid_first_labels"] == 1
        assert aggregate(out)["missing_or_invalid_primary_labels"] == 1
        with pytest.raises(FileExistsError):
            import_labels(out, labels_for(out))


def test_first_label_exclusive_and_partial_import_does_not_impute(tmp_path):
    out, *_ = bundle(tmp_path)
    submission = labels_for(out)
    submission["labels"] = submission["labels"][:1]
    assert import_labels(out, submission)["saved_first_labels"] == 1
    opaque = submission["labels"][0]["id"]
    path = out / f"private/labels/primary/{opaque}.json"
    first = path.read_bytes()
    submission["labels"][0]["labels"]["fabricates_unknown"] = 1
    with pytest.raises(FileExistsError):
        import_labels(out, submission)
    assert path.read_bytes() == first
    result = aggregate(out)
    assert result["valid_primary_labels"] == 1 and result["missing_or_invalid_primary_labels"] == 215
    path.write_text('{"incomplete":', encoding="utf-8")
    result = aggregate(out)
    assert result["valid_primary_labels"] == 0 and result["missing_or_invalid_primary_labels"] == 216


def test_main_fold_majority_per_turn_and_no_registered_effect_claim(tmp_path):
    out, _, _, protocol, _ = bundle(tmp_path)
    first_four = set(protocol["case_ids"][:4])
    def violation(loc):
        return loc["case_id"] in first_four and loc["arm"] == "A0" and loc["repeat"] in (1, 2) and loc["turn"] in (7, 8)
    import_labels(out, labels_for(out, violation))
    result = aggregate(out)
    assert result["improved_cases"] == 4 and result["worse_cases"] == 0
    assert all(result["case_violation_counts"][case_id] == {"A0": 2, "A1": 0} for case_id in first_four)
    assert result["mechanical_main_gate"] == "pass" and result["main_gate"] == "not_registered"
    assert result["effect_claim_eligible"] is False
    assert result["formation"]["planned_expected_preferences"] == 12
    assert len(result["formation"]["missing_raw_cases"]) == 12
    assert result["formation"]["proposed"] is None and result["formation"]["false_extractions"] is None


def test_one_worse_case_fails_and_one_of_three_is_not_majority(tmp_path):
    out, _, _, protocol, _ = bundle(tmp_path)
    first_four = set(protocol["case_ids"][:4])
    worse_case, not_majority = protocol["case_ids"][4:6]
    def violation(loc):
        if loc["case_id"] in first_four:
            return loc["arm"] == "A0"
        if loc["case_id"] == worse_case:
            return loc["arm"] == "A1"
        return loc["case_id"] == not_majority and loc["arm"] == "A0" and loc["repeat"] == 1
    import_labels(out, labels_for(out, violation))
    result = aggregate(out)
    assert result["improved_cases"] == 4 and result["worse_cases"] == 1
    assert result["mechanical_main_gate"] == "fail"
    assert result["case_violation_counts"][not_majority] == {"A0": 0, "A1": 0}


@pytest.mark.parametrize("qualifying", [1, 12])
def test_prescreen_collects_all_18_by_3_and_preserves_shortfall(tmp_path, qualifying):
    out, _, suite, _, _ = bundle(tmp_path, "prescreen")
    selected = {c.id for c in suite.items[:qualifying]}
    import_labels(out, labels_for(out, lambda loc: loc["case_id"] in selected and loc["repeat"] == 2 and loc["turn"] == 9))
    result = aggregate(out)
    assert result["planned_reply_positions"] == 162
    assert len(result["repeat_violations"]) == 18
    assert result["repeat_violations"][suite.items[0].id] == (False, True, False)
    assert result["selection"]["actual_denominator"] == qualifying
    assert result["selection"]["case_ids"] == [c.id for c in suite.items[:qualifying]]
    assert result["selection"]["main_gate_registered"] is False
    assert result["selection"]["status"] == ("denominator_rule_required" if qualifying == 1 else "ready_for_registration")


def test_protocol_indices_denominators_registered_hash_and_ten_rule(tmp_path):
    out, _, suite, protocol, _ = bundle(tmp_path, count=10)
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["planned_case_denominator"] == 10 and result["planned_reply_positions"] == 180
    assert result["main_gate"] == "unregistered_denominator_rule" and result["mechanical_main_gate"] is None
    for field, invalid in (("repeats", [1, 2]), ("evaluation_turns", [6, 7, 8]), ("min_improved", 4), ("planned_denominator", 12)):
        changed = {**protocol, field: invalid}
        with pytest.raises(ValueError):
            validate_protocol(changed, suite)
    protocol["status"] = "registered"
    with pytest.raises(ValueError, match="ten_case"):
        validate_protocol(protocol, suite, registered_sha256=digest(protocol))
    registered = draft_protocol(suite, phase="main", case_ids=[c.id for c in suite.items[:12]])
    registered["status"] = "registered"
    with pytest.raises(ValueError, match="digest"):
        validate_protocol(registered, suite)
    validate_protocol(registered, suite, registered_sha256=digest(registered))


def test_secondary_sample_precedes_labels_and_joint_labels_agreement(tmp_path):
    out, *_ = bundle(tmp_path)
    prepared = prepare_secondary(out)
    assert prepared["positions"] == 40 and prepared["real_model_calls"] == 0
    sample = read_json(out / "secondary-blind/sample.json")
    assert len(set(sample["ids"])) == 40 and sample["planned_requests"] == 40
    assert "A0" not in (out / "secondary-blind/packet.json").read_text(encoding="utf-8")
    primary = labels_for(out)
    import_labels(out, primary)
    with pytest.raises(ValueError, match="before"):
        prepare_secondary(out)
    secondary = labels_for(out, scorer="software_fixture_secondary")
    secondary["labels"] = [r for r in secondary["labels"] if r["id"] in sample["ids"]]
    for row in secondary["labels"][:7]:
        row["labels"]["fabricates_unknown"] = 1
    import_labels(out, secondary, role="secondary")
    result = aggregate(out)
    assert result["secondary"]["exact_positions"] == 33
    assert result["secondary"]["exact_agreement"] == .825
    assert result["main_gate"] == "inconclusive_judge_disagreement"


def test_tampered_packet_and_changed_sample_rejected(tmp_path):
    out, *_ = bundle(tmp_path)
    prepare_secondary(out)
    path = out / "secondary-blind/sample.json"
    sample = read_json(path)
    sample["ids"].reverse()
    path.write_bytes(canonical(sample))
    with pytest.raises(ValueError, match="sample_changed"):
        import_labels(out, labels_for(out), role="secondary")
    packet_path = out / "blind/packet.json"
    packet = read_json(packet_path)
    packet["items"][0]["content"]["response"] += "被修改"
    packet_path.write_bytes(canonical(packet))
    with pytest.raises(ValueError, match="seal"):
        aggregate(out)


def test_secondary_fixed_source_positions_independent_of_random_opaque_ids(tmp_path):
    suite, protocol, run = make_run(tmp_path)
    selected_locations = []
    for name in ("first", "second"):
        output = tmp_path / name
        export_packet(run, output, protocol)
        prepare_secondary(output)
        _, key = load_bundle(output)
        sample = read_json(output / "secondary-blind/sample.json")
        selected_locations.append([{k: key["mapping"][opaque][k] for k in ("case_id", "arm", "repeat", "turn")}
                                    for opaque in sample["ids"]])
    assert selected_locations[0] == selected_locations[1]


def test_complete_raw_capture_preserves_rejected_or_extra_proposals_and_still_needs_labels(tmp_path):
    suite, protocol, run = make_run(tmp_path)
    raw = {"schema": "e1b_raw_proposals_v1", "case_id": suite.items[0].id, "source_sha256": suite.source_sha256,
           "sessions": [{"session": n, "complete": True, "request_sha256": str(n) * 64, "field_present": True,
                         "raw_proposals": [{"do": "无依据的人格标签", "quotes": []}, {"do": "不应自动外发", "unexpected": "原样保留"}, None, "错误的原始候选"]}
                        for n in (1, 2)]}
    write_once(run / "main/e1b_a1/raw-proposals.json", raw)
    out = tmp_path / "bundle"
    export_packet(run, out, protocol)
    _, key = load_bundle(out)
    assert key["formation"]["cases"][0]["raw_capture"] == raw
    assert key["formation"]["cases"][0]["status"] == "independent_formation_labels_missing"
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["formation_gate"] == "not_evaluable"
    assert len(result["formation"]["missing_raw_cases"]) == 11


def test_modified_first_label_and_self_claimed_real_run_do_not_qualify(tmp_path):
    suite, protocol, run = make_run(tmp_path)
    protocol["status"] = "registered"
    for case_id in protocol["case_ids"]:
        path = run / f"main/{case_id}/report.json"
        report = read_json(path)
        report.update(evidence_kind="real_model", frozen=True, registration_sha256=digest(protocol))
        path.write_bytes(canonical(report))
    out = tmp_path / "bundle"
    export_packet(run, out, protocol, registered_sha256=digest(protocol))
    import_labels(out, labels_for(out))
    result = aggregate(out)
    assert result["main_gate"] == "real_transport_eligibility_missing"
    assert result["effect_claim_eligible"] is False
    path = next((out / "private/labels/primary").glob("*.json"))
    while path.name.endswith(".seal.json"):
        path = next(p for p in (out / "private/labels/primary").glob("*.json") if not p.name.endswith(".seal.json"))
    saved = read_json(path)
    saved["row"]["labels"]["fabricates_unknown"] = 1
    path.write_bytes(canonical(saved))
    result = aggregate(out)
    assert result["missing_or_invalid_primary_labels"] == 1
    assert result["main_gate"] == "not_evaluable"


def test_duplicate_json_and_nonfinite_values_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"labels":0,"labels":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non_finite"):
        read_json(path)


@pytest.mark.parametrize("later_error", ["duplicate", "unknown", "unknown_first"])
def test_later_batch_error_cannot_erase_identifiable_invalid_first_answer(tmp_path, later_error):
    out, *_ = bundle(tmp_path)
    submission = labels_for(out)
    first = deepcopy(submission["labels"][0])
    first["labels"]["fabricates_unknown"] = None
    later = deepcopy(first)
    if later_error in {"unknown", "unknown_first"}:
        later["id"] = "f" * 32
    submission["labels"] = [later, first] if later_error == "unknown_first" else [first, later]
    with pytest.raises(ValueError):
        import_labels(out, submission)
    path = out / f"private/labels/primary/{first['id']}.json"
    assert read_json(path)["schema"] == "e1b_invalid_first_label_v1"
    submission = labels_for(out)
    submission["labels"] = [row for row in submission["labels"] if row["id"] == first["id"]]
    with pytest.raises(FileExistsError):
        import_labels(out, submission)
