"""Independent synthetic checks; no holdout questions or model calls."""
import json

import pytest

from aria.bench.e1c_cases import parse_suite
from aria.bench.e1c_source import seal_first_submission
from aria.bench.e1c_runner import strip_a0_all_sources
from aria.bench.e1c_registration import FinancialInheritance, execute, financial_preparation
from e1c_software_fixture import fixture_author, fixture_payload


def encode(payload):
    return json.dumps(payload, ensure_ascii=False).encode()


@pytest.mark.parametrize("bad_provenance", [None, {}, {"first_submission": 1}])
def test_invalid_provenance_preserves_first_bytes_and_blocks_replacement(tmp_path, bad_provenance):
    first = b'{"invalid_original":"SOFTWARE_ONLY"}'
    directory = tmp_path / "first"
    receipt = seal_first_submission(first, bad_provenance, directory)
    assert receipt["status"] == "invalid_first_submission"
    assert receipt["execution_authorized"] is False
    assert (directory / "author-original.json").read_bytes() == first
    with pytest.raises(FileExistsError):
        seal_first_submission(encode(fixture_payload()), fixture_author(), directory)
    assert (directory / "author-original.json").read_bytes() == first


def test_a0_removes_partial_preferences_and_all_cross_session_sources(tmp_path):
    from aria.engine.model.persona import (
        EpisodicMemoryEntry, PersonaProfile, PreferenceQuote, StatedPreference,
    )
    from datetime import datetime, timezone
    at = datetime.now(timezone.utc).isoformat()
    profile = PersonaProfile(user_id="SOFTWARE_ONLY")
    profile.stated_preferences = [StatedPreference(
        id="pref_SOFTWARE", do="软件占位", scope="软件占位", created_at=at, updated_at=at,
        quotes=[PreferenceQuote(session_id="S1", turn_index=1, text="软件占位")])]
    profile.episodic_memory = [EpisodicMemoryEntry(id=f"SOFTWARE_FACT_{n}", summary="软件占位",
        source_session_id=s, source_turn_index=t) for n, (s,t) in enumerate([("S1",1),("S2",4),("S2",5)])]
    before = profile.model_dump()
    suite = parse_suite(encode(fixture_payload()), allow_software_fixture=True)
    cleared, removed = strip_a0_all_sources(profile, suite.items[0], {1:"S1", 2:"S2"})
    assert cleared.stated_preferences == [] and profile.model_dump() == before
    assert [entry.id for entry in cleared.episodic_memory] == ["SOFTWARE_FACT_2"]
    assert len(removed["per_source"]) == 2


def test_all_nine_unknowns_carried_only_as_arithmetic_and_execute_always_refuses():
    snapshot = FinancialInheritance.model_validate({
        "schema_version":"e1c_finance_v1", "prior_e1b_closed": True,
        "prior_assessment_sha256":"a"*64, "prior_ledger_sha256":"b"*64,
        "known_cumulative_usd":"1.909447604", "known_e1b_usd":"0.517068432",
        "unknown_holds":[{"campaign": "SOFTWARE_ONLY", "request_id":f"SOFTWARE_{n}",
                          "request_sha256": f"{n:064x}", "amount":"0.01"} for n in range(9)],
    })
    result = financial_preparation(snapshot)
    assert result["inherited_unknown_count"] == 9 and result["inherited_unknown_hold_usd"] == "0.09"
    assert result["execution_authorized"] is False and result["retained_experimental_answers"] == 0
    with pytest.raises(PermissionError):
        execute(status="approved", budget="100", api_key="SOFTWARE-NOT-A-KEY")
