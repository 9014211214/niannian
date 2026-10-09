"""Local, staged, explicitly synthetic structural checks. No holdout semantics."""
from tests._model_http_stub import stub_model_http
from collections import Counter
from copy import deepcopy
import hashlib
import json
import socket
import urllib.request

import pytest

from aria.bench.e1c_cases import load_e1c_cases, parse_suite
from aria.bench.e1c_source import seal_first_submission
from aria.bench.e1c_runner import strip_a0_all_sources
from aria.bench.e1c_registration import (
    FinancialInheritance, execute, financial_preparation, planned_positions,
)
from e1c_software_fixture import fixture_author, fixture_payload


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("software_preparation_network_forbidden")
    stub_model_http(monkeypatch, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def suite():
    return parse_suite(encode(fixture_payload()), allow_software_fixture=True)


def test_dedicated_loader_and_no_legacy_pref_turn():
    loaded = suite()
    assert len(loaded.items) == 12 and all(case.id.startswith("e1c_") for case in loaded.items)
    assert set(Counter(case.category for case in loaded.items).values()) == {2}
    assert not hasattr(loaded.items[0], "pref_turn")
    assert loaded.items[0].source_layout == "multiple_sessions"
    assert loaded.items[0].current_input(1, 1) == "软件夹具，会话1轮1。"
    with pytest.raises(ValueError, match="software_fixture"):
        parse_suite(encode(fixture_payload()))
    # Existing E1b contract remains strict, without reading its actual questions.
    from aria.bench.e1b_cases import _SuiteFile
    with pytest.raises(ValueError):
        _SuiteFile.model_validate(fixture_payload())


@pytest.mark.parametrize("fault", ["count", "duplicate", "id", "category", "s1_count", "bool_turn",
                                   "future", "quote", "duplicate_source", "reorder", "annotation", "missing_rubric"])
def test_rejects_structural_faults_without_product_prescreen(fault):
    payload = fixture_payload()
    case = payload["items"][0]
    if fault == "count":
        payload["items"].pop()
    elif fault == "duplicate":
        payload["items"][1] = deepcopy(case)
    elif fault == "id":
        case["id"] = "e1b_a1"
    elif fault == "category":
        case["category"] = "wants_quiet"
    elif fault == "s1_count":
        case["s1"].pop()
    elif fault == "bool_turn":
        case["preference_sources"][0]["turn"] = True
    elif fault == "future":
        case["preference_sources"][0]["session"] = 3
    elif fault == "quote":
        case["preference_sources"][0]["quote"] = "部分"
    elif fault == "duplicate_source":
        case["preference_sources"].append(deepcopy(case["preference_sources"][0]))
    elif fault == "reorder":
        case["preference_sources"].reverse()
    elif fault == "annotation":
        case["s3"]["trap"][0] = "rubric 软件夹具"
    elif fault == "missing_rubric":
        payload["defaults"].pop("leaks_private")
    with pytest.raises(ValueError):
        parse_suite(encode(payload), allow_software_fixture=True)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'\xef\xbb\xbf{}'])
def test_json_ambiguity_rejected(raw):
    with pytest.raises(ValueError):
        parse_suite(raw, allow_software_fixture=True)


def test_natural_unrecognized_text_is_not_screened_and_layout_supports_three_forms():
    from aria.bench.e1c_cases import E1cCase
    case = fixture_payload()["items"][0]
    case["s1"][1][0] = "软件夹具。第二句仅测结构。"
    case["preference_sources"] = [{"session": 1, "turn": 1, "quote": case["s1"][1][0]}]
    assert E1cCase.model_validate(case).source_layout == "single_utterance"
    case["preference_sources"].append({"session": 1, "turn": 2, "quote": case["s1"][2][0]})
    assert E1cCase.model_validate(case).source_layout == "multiple_turns_one_session"
    assert suite().items[0].source_layout == "multiple_sessions"


def test_invalid_first_source_preserved_and_cannot_be_replaced(tmp_path):
    original = b'{"invalid_first":true}'
    receipt = seal_first_submission(original, fixture_author(), tmp_path / "first")
    assert receipt["status"] == "invalid_first_submission"
    assert (tmp_path / "first/author-original.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        seal_first_submission(encode(fixture_payload()), fixture_author(), tmp_path / "first")
    assert receipt["execution_authorized"] is False


def test_first_source_ids_and_sha_no_best_subset_or_changed_source(tmp_path):
    payload = fixture_payload()
    payload["items"].reverse()  # Preserve author order, not extractor-based reorder.
    original = encode(payload)
    receipt = seal_first_submission(original, fixture_author(), tmp_path / "first")
    assert receipt["case_ids"] == [case["id"] for case in payload["items"]]
    source = tmp_path / "first/author-original.json"
    assert load_e1c_cases(source, expected_sha256=receipt["source_sha256"],
                              allow_software_fixture=True).source_sha256 == hashlib.sha256(original).hexdigest()
    other = tmp_path / "changed.json"
    other.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="sha_mismatch"):
        load_e1c_cases(other, expected_sha256=receipt["source_sha256"], allow_software_fixture=True)


def test_author_scope_expansion_and_numeric_attestation_rejected(tmp_path):
    for field, value in (("provided_materials", {"product_prompts": "b" * 64}), ("first_submission", 1)):
        author = fixture_author()
        author[field] = value
        raw = encode(fixture_payload())
        receipt = seal_first_submission(raw, author, tmp_path / field)
        assert receipt["status"] == "invalid_first_submission"
        assert receipt["error_stage"] == "author_provenance"
        assert (tmp_path / field / "author-original.json").read_bytes() == raw
        with pytest.raises(FileExistsError):
            seal_first_submission(b"{}", fixture_author(), tmp_path / field)


def test_a0_removes_every_cross_session_source_whole_record_and_preserves_a1():
    from aria.engine.model.persona import (
        EpisodicMemoryEntry, PersonaProfile, PersonHypothesis, PersonHypothesisEvidence,
        RelationshipContextEntry, RelationshipEvidence,
    )
    persona = PersonaProfile(user_id="software-fixture")
    persona.episodic_memory = [EpisodicMemoryEntry(
        id=name, summary="软件夹具", source_session_id=s, source_turn_index=t)
        for name, s, t in (("first", "actual-s1", 1), ("second", "actual-s2", 4), ("keep", "actual-s1", 4))]
    persona.person_hypotheses = [PersonHypothesis(id="mixed", framework="response", claim="软件夹具",
        first_observed_at="2026-10-05", evidence=[PersonHypothesisEvidence(
            at="2026-10-05", session_id=s, turn_indices=[t], relation="support", summary="软件夹具")
            for s, t in (("actual-s1", 1), ("actual-s2", 8))])]
    persona.relationship_context = [RelationshipContextEntry(id="relation-second", name="软件夹具",
        facts=["软件夹具"], first_observed_at="2026-10-05", evidence=[RelationshipEvidence(
            at="2026-10-05", session_id="actual-s2", turn_indices=[4], summary="软件夹具")])]
    original = persona.model_dump()
    cleared, removed = strip_a0_all_sources(persona, suite().items[0], {1: "actual-s1", 2: "actual-s2"})
    assert persona.model_dump() == original
    assert [row.id for row in cleared.episodic_memory] == ["keep"]
    assert not cleared.person_hypotheses and not cleared.relationship_context
    assert removed["facts"] == ["first", "second"] and removed["hypotheses"] == ["mixed"]
    assert len(removed["per_source"]) == 2
    with pytest.raises(ValueError):
        strip_a0_all_sources(persona, suite().items[0], {1: "same", 2: "same"})


def test_planned_denominators_do_not_retain_e1b_answers():
    positions = planned_positions(suite(), seed=20261005)
    assert len(positions) == len(set(positions)) == 1056
    assert Counter(row[4] for row in positions) == {"online": 1008, "memory": 24, "reflection": 24}
    assert len([row for row in positions if row[1] == 3 and row[5] >= 7]) == 216
    assert 1056 + 24 + 1056 * 2 == 3192
    assert positions[28][2:4] == ("A1", 1)
    assert positions[48][2:4] == ("A0", 2)


def financial_fixture():
    return {"schema_version": "e1c_finance_v1", "prior_e1b_closed": True,
            "prior_assessment_sha256": "a" * 64, "prior_ledger_sha256": "b" * 64,
            "known_cumulative_usd": "1.2", "known_e1b_usd": "0.4",
            "unknown_holds": [{"campaign": "SOFTWARE-CAMPAIGN", "request_id": "SOFTWARE-UNKNOWN-1",
                               "request_sha256": "c" * 64, "amount": "0.1"},
                              {"campaign": "SOFTWARE-CAMPAIGN", "request_id": "SOFTWARE-UNKNOWN-2",
                               "request_sha256": "d" * 64, "amount": "0.02"}]}


def test_finance_inherits_every_unknown_but_never_authorizes_or_resets_history():
    report = financial_preparation(FinancialInheritance.model_validate(financial_fixture()))
    assert report["inherited_unknown_count"] == 2 and report["inherited_unknown_hold_usd"] == "0.12"
    assert report["committed_cumulative_usd"] == "1.32"
    assert report["retained_experimental_answers"] == 0 and report["execution_authorized"] is False
    with pytest.raises(PermissionError):
        execute(approval=True, budget="1.1")
    incomplete = financial_fixture()
    incomplete["prior_e1b_closed"] = False
    with pytest.raises(ValueError):
        financial_preparation(FinancialInheritance.model_validate(incomplete))


@pytest.mark.parametrize("fault", ["duplicate", "negative", "nan", "numeric", "inconsistent"])
def test_invalid_financial_inheritance_rejected(fault):
    value = financial_fixture()
    if fault == "duplicate":
        value["unknown_holds"].append(deepcopy(value["unknown_holds"][0]))
    elif fault == "inconsistent":
        value["known_e1b_usd"] = "2"
    else:
        value["known_cumulative_usd"] = {"negative": "-1", "nan": "NaN", "numeric": 1.2}[fault]
    with pytest.raises(ValueError):
        FinancialInheritance.model_validate(value)


