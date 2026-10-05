"""Full twelve-case metadata arithmetic plus one actual 88-call software product case."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tests.e1c_evaluation_fixture import data, bound, scoring, save, no_network, fixture  # noqa: F401
from scripts import e1b_evaluation as reply


@pytest.fixture(autouse=True)
def arithmetic_rubric_fixture(monkeypatch):
    original = fixture.fixture_payload
    def payload():
        value = original()
        name = sorted(value["defaults"])[0]
        value["defaults"][name] = {"applicable": True, "violation_if": "软件标签算术占位：1",
                                    "not_violation_if": "软件标签算术占位：0"}
        return value
    monkeypatch.setattr(fixture, "fixture_payload", payload)


def ready(data, tmp_path, *, qualification_name="SOFTWARE-ONLY-qualification.json"):
    inventory = bound(data)
    inventory.update(status="qualified", evidence_kind="software_fixture",
                     registration_sha256="a" * 64, ledger_sha256="d" * 64, authorization_sha256="e" * 64)
    protocol = scoring.protocol(data.submission, status="registered")
    inventory.update(evaluation_protocol=protocol, evaluation_protocol_sha256=reply.digest(protocol))
    inventory.update(finance={"notice": "SOFTWARE ONLY synthetic accounting"},
                     new_accounting={"known_cost_usd": "0"},
                     source_lineage={"single_registered_source": True})
    path = tmp_path / qualification_name
    sha = save(path, inventory)
    output = tmp_path / "first-blind-bundle"
    result = scoring.export_bundle(data.root, data.submission, inventory, output,
        qualification_path=path, qualification_sha256=sha, evaluation_protocol=protocol,
        registered_protocol_sha256=reply.digest(protocol), blind_seed=123)
    return output, result["bundle_sha256"], inventory


def labels(root, sha, kind):
    _, bundles = scoring.load_bundle(root, sha)
    packet, key = bundles[kind]
    module = reply if kind == "reply" else scoring.formation
    value = module.label_template(packet)
    value["scorer"]["id"] = "SOFTWARE_ONLY_INDEPENDENT"
    for row in value["labels"]:
        row["labels"] = {name: 0 for name in row["labels"]}
        meta = key["mapping"][row["id"]]
        if kind == "reply":
            item = next(item for item in packet["items"] if item["id"] == row["id"])
            applicable = next(name for name, rule in item["content"]["rubric"].items() if rule["applicable"])
            if meta["arm"] == "A0" and meta["case_id"] in {"e1c_a1", "e1c_a2", "e1c_b1", "e1c_b2"}:
                row["labels"][applicable] = 1
        else:
            item = next(item for item in packet["items"] if item["id"] == row["id"])
            if not scoring.formation._uninterpretable(item["content"]["proposal"]):
                row["labels"].update(source_support=1, expected_match=1)
    return value


def import_both(root, sha):
    for kind in ("reply", "formation"):
        value = labels(root, sha, kind)
        assert scoring.import_first_batch(root, sha, json.dumps(value).encode(), kind=kind,
                                           batch_id="software-first")["status"] == "first_batch_imported"


def test_full_216_and_12_exact_old_arithmetic_retains_nonobject_false_extractions(data, tmp_path):
    root, sha, _ = ready(data, tmp_path)
    import_both(root, sha)
    result = scoring.aggregate(root, sha)
    assert result["status"] == "software_labels_complete"
    assert result["reply_evaluation"]["improved_cases"] == 4
    assert result["reply_evaluation"]["worse_cases"] == 0
    assert result["reply_evaluation"]["mechanical_main_gate"] == "pass"
    assert result["reply_evaluation"]["secondary"]["planned_positions"] == 0
    formed = result["formation_evaluation"]
    assert formed["proposed"] == 12 and formed["candidate_denominator"] == 48
    assert formed["false_extractions"] == 24 and formed["mechanical_formation_gate"] == "fail"
    assert result["mechanical_overall_gate"] == "fail"
    assert result["overall_gate"] == "not_evaluable" and not result["effect_claim_eligible"]
    assert reply.SCHEMA == "e1b_blind_packet_v1"  # no global monkeypatch


@pytest.mark.parametrize("raw", [b'{"bad":"first"}', b"[]", b" " * (256 * 1024 + 1)], ids=["malformed", "nonobject", "oversized"])
def test_invalid_first_entire_batch_cannot_be_relabelled_via_new_batch_id(data, tmp_path, raw):
    root, sha, _ = ready(data, tmp_path)
    failed = scoring.import_first_batch(root, sha, raw, kind="reply", batch_id="bad-first")
    assert failed["status"] == "invalid_first"
    assert (root / "reply/private/first-batches/bad-first/raw.json").read_bytes() == raw
    later = scoring.import_first_batch(root, sha, json.dumps(labels(root, sha, "reply")).encode(),
                                       kind="reply", batch_id="later-batch")
    assert later["status"] == "invalid_first"
    result = scoring.aggregate(root, sha)
    assert result["reply_evaluation"]["valid_primary_labels"] == 0
    assert result["reply_evaluation"]["missing_or_invalid_primary_labels"] == 216
    assert result["mechanical_overall_gate"] is None


def test_partial_batches_preserve_first_unknown_as_missing_not_zero(data, tmp_path):
    root, sha, _ = ready(data, tmp_path)
    value = labels(root, sha, "reply")
    one, rest = deepcopy(value), deepcopy(value)
    one["labels"], rest["labels"] = value["labels"][:1], value["labels"][1:]
    ident = one["labels"][0]["id"]
    name = next(iter(one["labels"][0]["labels"]))
    one["labels"][0]["labels"][name] = None
    bad = scoring.import_first_batch(root, sha, json.dumps(one).encode(), kind="reply",
        batch_id="one", expected_ids=[ident])
    assert bad["result"]["saved_invalid_first_labels"] == 1
    good = scoring.import_first_batch(root, sha, json.dumps(rest).encode(), kind="reply", batch_id="rest",
        expected_ids=[r["id"] for r in rest["labels"]])
    assert good["result"]["saved_first_labels"] == 215
    one["labels"][0]["labels"][name] = 0
    assert scoring.import_first_batch(root, sha, json.dumps(one).encode(), kind="reply",
        batch_id="correction", expected_ids=[ident])["status"] == "invalid_first"
    result = scoring.aggregate(root, sha)["reply_evaluation"]
    assert result["valid_primary_labels"] == 215 and result["missing_or_invalid_primary_labels"] == 1
    assert "mechanical_main_gate" not in result


@pytest.mark.parametrize("identity", ["author", "wrong_type"])
def test_known_author_and_bad_scorer_shape_stay_invalid_first(data, tmp_path, identity):
    root, sha, _ = ready(data, tmp_path)
    value = labels(root, sha, "reply")
    if identity == "author":
        value["scorer"]["id"] = data.submission.author.author_id
    else:
        value["scorer"] = "not_an_identity_object"
    result = scoring.import_first_batch(root, sha, json.dumps(value).encode(), kind="reply", batch_id="first")
    assert result["status"] == "invalid_first"
    assert scoring.aggregate(root, sha)["reply_evaluation"]["valid_primary_labels"] == 0


def test_raw_storage_failure_does_not_allow_replacement_batch(data, tmp_path, monkeypatch):
    root, sha, _ = ready(data, tmp_path)
    raw = json.dumps(labels(root, sha, "reply")).encode()
    original_open = Path.open
    def fail_raw(path, *args, **kwargs):
        if path.name == "raw.json" and path.parent.name == "failed-storage":
            raise OSError("SOFTWARE disk-write failure")
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(Path, "open", fail_raw)
        with pytest.raises(OSError, match="disk-write"):
            scoring.import_first_batch(root, sha, raw, kind="reply", batch_id="failed-storage")
    intent = reply.read_json(root / "reply/private/first-batches/failed-storage/intent.json")
    assert len(intent["expected_ids"]) == 216
    later = scoring.import_first_batch(root, sha, raw, kind="reply", batch_id="second")
    assert later["status"] == "invalid_first"
    assert scoring.aggregate(root, sha)["reply_evaluation"]["valid_primary_labels"] == 0


@pytest.mark.parametrize("changed", [False, True])
def test_assessment_rechecks_closed_qualification_before_gate(data, tmp_path, monkeypatch, changed):
    """Only wrapper control flow: the genuine closed-ledger test is entry's chain."""
    from types import SimpleNamespace
    root, sha, inventory = ready(data, tmp_path, qualification_name="qualification.json")
    save(tmp_path / "registration.json", {"notice": "SOFTWARE wrapper fixture, no authorization"})
    import_both(root, sha)
    calls = []
    def qualify(campaign, *, authorization_sha256, write):
        calls.append((campaign, authorization_sha256, write))
        value = deepcopy(inventory)
        if changed:
            value["ledger_sha256"] = "f" * 64
        return value
    monkeypatch.setattr(scoring, "_qualification_module", lambda: SimpleNamespace(qualify=qualify))
    if changed:
        with pytest.raises(ValueError, match="closed_evidence_changed"):
            scoring.assess(tmp_path, root, sha, authorization_sha256="e" * 64)
    else:
        result = scoring.assess(tmp_path, root, sha, authorization_sha256="e" * 64)
        assert result["closed_generation_revalidated"] and result["overall_gate"] == "not_evaluable"
        assert result["main_gate"] == result["formation_gate"] == "software_fixture_only"
        assert not result["effect_claim_eligible"] and not result["release_authorized"]
    assert calls == [(tmp_path, "e" * 64, False)]


@pytest.mark.parametrize("target", ["packet", "mapping", "qualification", "original_source", "first_raw"])
def test_immutable_inputs_and_raw_first_byte_changes_fail_closed(data, tmp_path, target):
    root, sha, _ = ready(data, tmp_path)
    import_both(root, sha)
    choices = {"packet": root / "reply/blind/packet.json", "mapping": root / "reply/private/mapping.json",
               "qualification": tmp_path / "SOFTWARE-ONLY-qualification.json",
               "original_source": data.source / "author-original.json",
               "first_raw": root / "reply/private/first-batches/software-first/raw.json"}
    path = choices[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        scoring.aggregate(root, sha)


def test_formation_never_supplies_future_user_utterances(data, tmp_path):
    root, sha, _ = ready(data, tmp_path)
    _, bundles = scoring.load_bundle(root, sha)
    packet, key = bundles["formation"]
    for item in packet["items"]:
        meta = key["mapping"][item["id"]]
        case = next(c for c in data.submission.suite.items if c.id == meta["case_id"])
        assert item["content"]["user_utterances"] == [{"turn_index": n, "text": text}
                                                       for n, (text, _) in enumerate(case.session(meta["session"]))]
        earlier = item["content"]["earlier_user_utterances"]
        assert earlier == ([] if meta["session"] == 1 else [{"turn_index": n, "text": text}
                                                        for n, (text, _) in enumerate(case.session(1))])
        assert "case_id" not in item["content"] and "product_outcome" not in item["content"]


def test_unknown_qualification_and_changed_protocol_cannot_open_first_bundle(data, tmp_path):
    inventory = bound(data)
    path = tmp_path / "q.json"
    protocol = scoring.protocol(data.submission, status="registered")
    with pytest.raises(ValueError, match="complete_qualification"):
        scoring.export_bundle(data.root, data.submission, inventory, tmp_path / "out",
            qualification_path=path, qualification_sha256="f" * 64,
            evaluation_protocol=protocol, registered_protocol_sha256=reply.digest(protocol), blind_seed=1)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["missing", "different", "wrong_digest"])
def test_scoring_protocol_must_match_generation_registration(data, tmp_path, change):
    inventory = bound(data)
    p = scoring.protocol(data.submission, status="registered")
    inventory.update(status="qualified", evidence_kind="software_fixture",
        registration_sha256="a" * 64, ledger_sha256="d" * 64, authorization_sha256="e" * 64,
        evaluation_protocol=deepcopy(p), evaluation_protocol_sha256=reply.digest(p))
    if change == "missing":
        inventory.pop("evaluation_protocol")
    elif change == "different":
        inventory["evaluation_protocol"]["secondary_positions"] = 40
        inventory["evaluation_protocol_sha256"] = reply.digest(inventory["evaluation_protocol"])
    else:
        inventory["evaluation_protocol_sha256"] = "f" * 64
    path = tmp_path / "qualified-original.json"
    sha = save(path, inventory)
    with pytest.raises(ValueError, match="not_frozen_before_generation"):
        scoring.export_bundle(data.root, data.submission, inventory, tmp_path / "bundle",
            qualification_path=path, qualification_sha256=sha, evaluation_protocol=p,
            registered_protocol_sha256=reply.digest(p), blind_seed=123)
    assert not (tmp_path / "bundle").exists()


def test_actual_one_case_88_calls_produce_partial_capture_not_fake_full_qualification(tmp_path):
    from tests.test_e1c_runner import setup
    from aria.bench.e1c_runner import E1cRunner
    args, records = setup(tmp_path)
    runner = E1cRunner(**args)
    runner.run_case("e1c_a1")
    partial = scoring.inspect_case_capture(args["root"], runner.submission, "e1c_a1")
    assert sum(len(c.online.calls) + len(c.memory.calls) + len(c.reflection.calls) for _, c in records) == 88
    assert partial["reply_positions"] == 18 and partial["fixed_reply_positions"] == 216
    assert partial["fixed_formation_cases"] == 12 and not partial["effect_claim_eligible"]
    assert partial["formal_gate"] == "not_evaluable"
    for candidate in partial["formation_candidates"]:
        assert len(candidate["earlier_user_utterances"]) == (0 if candidate["session"] == 1 else 12)


@pytest.mark.parametrize("proposed,gate", [(0, "fail"), (9, "fail"), (10, "pass")])
def test_original_eighty_percent_gate_and_empty_array_semantics(data, tmp_path, proposed, gate):
    for index, case in enumerate(data.submission.suite.items):
        directory = data.root / "main" / case.id
        capture = reply.read_json(directory / "raw-proposals.json")
        for session in capture["sessions"]:
            number = session["session"]
            # Disclosed arithmetic fixtures: no preferred product outcome filtering.
            replacements = [session["raw_proposals"][1]] if index < proposed else []
            first = reply.read_json(directory / f"memory-s{number}.first.json")
            first["raw_proposals"] = replacements
            session["raw_proposals"] = replacements
            session["first_attempt_sha256"] = save(directory / f"memory-s{number}.first.json", first)
        save(directory / "raw-proposals.json", capture)
    root, sha, _ = ready(data, tmp_path)
    value = labels(root, sha, "reply")
    assert scoring.import_first_batch(root, sha, json.dumps(value).encode(), kind="reply", batch_id="first")["status"] == "first_batch_imported"
    if proposed:
        value = labels(root, sha, "formation")
        assert scoring.import_first_batch(root, sha, json.dumps(value).encode(), kind="formation", batch_id="first")["status"] == "first_batch_imported"
    result = scoring.aggregate(root, sha)
    formed = result["formation_evaluation"]
    assert result["status"] == "software_labels_complete"
    assert formed["proposed"] == proposed and formed["false_extractions"] == 0
    assert formed["candidate_denominator"] == proposed * 2
    assert formed["mechanical_formation_gate"] == gate
    assert result["mechanical_overall_gate"] == gate
    if not proposed:
        assert formed["verbatim_quote_rate"] is None
    assert result["overall_gate"] == "not_evaluable" and not result["effect_claim_eligible"]
