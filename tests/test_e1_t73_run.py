"""Authorized-run tests use temporary histories, synthetic keys and fake HTTP."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import check_e1_t72 as prior
from scripts import check_e1_t73 as wire_check
from scripts import run_e1_t71 as previous
from scripts import run_e1_t73 as runner
from scripts import t72_transport
from tests.test_e1_t72 import failed_campaign, tree_hashes  # noqa: F401
from tests.test_e1_t73 import static_metadata


REAL_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_KEY = "synthetic-only-target-credential"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("real_network_or_credentials_forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(previous, "credential", forbidden)
    monkeypatch.setattr(previous.legacy, "credential", forbidden)
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def guarded_text(path, *args, **kwargs):
        if path.absolute() == REAL_ROOT / ".env":
            forbidden()
        return read_text(path, *args, **kwargs)

    def guarded_bytes(path, *args, **kwargs):
        if path.absolute() == REAL_ROOT / ".env":
            forbidden()
        return read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)


@pytest.fixture(scope="module")
def bundle():
    return previous.load_bundle()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def artifact_rows(directory):
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(directory.rglob("*")) if path.is_file()]


@pytest.fixture
def authorized(failed_campaign, monkeypatch):
    bundle, directory, _, _, _ = failed_campaign
    write_json(directory / "e1-t71-preflight.json", {"synthetic": True, "model_requests": 0})
    state = prior.inspect_campaign(bundle, previous.prepare_e1(bundle)[3])
    repair = previous.legacy.read_json(REAL_ROOT / wire_check.REGISTRATION)
    repair["prior_campaign"] = state
    write_json(bundle["root"] / wire_check.REGISTRATION, repair)
    registration = {
        "schema": "aria_t73_authorized_run_v1",
        "id": "t73_20261001_unknown_cost_exception_v1",
        "campaign_id": "e1_t73_20261001_v1", "parent_campaign_id": "t6_20260930_v1",
        "repair_seal_sha256": "d9f4ef6d6119265e574100a1f4fa428cde0bb893da9dbfaf7c18d003b6620f7f",
        "budget_usd": deepcopy(bundle["amendment"]["budget_usd"]),
        "authorization": {"user_statement": "接受1笔费用暂时未知",
            "historical_unknown_cost_count": 1, "retain_original_unknown_cost": True,
            "applies_only_to_campaign": "e1_t73_20261001_v1", "permits_automatic_retry": False},
        "prior_campaign": state, "prior_artifacts": artifact_rows(directory),
        "original_attempt": {"id": "e1_001",
            "request_sha256": "111189eb00e0b7418c57c3704f3cf4bd7a42c534894715d36ea71acf37930394",
            "receipt_sha256": hashlib.sha256((directory / "e1/e1_001.receipt.json").read_bytes()).hexdigest()},
    }
    # Synthetic HTTP fixture has a different receipt from the one human-approved
    # production failure; bind this temporary fixture without touching that file.
    monkeypatch.setattr(runner, "ORIGINAL_RECEIPT_SHA256",
                        registration["original_attempt"]["receipt_sha256"])
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    return bundle, directory, registration


def patch_main(monkeypatch, bundle):
    monkeypatch.setattr(runner, "ROOT", bundle["root"])
    monkeypatch.setattr(previous, "load_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(runner, "verify_execution", lambda _: {"seal_sha256": "a" * 64})


def test_exact_history_exception_is_verified_without_modifying_old_files(authorized):
    bundle, directory, registration = authorized
    before = tree_hashes(directory)
    runner.verify_history(bundle, registration)
    assert len(registration["prior_artifacts"]) == 7
    assert tree_hashes(directory) == before
    with sqlite3.connect((directory / "campaign.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT cost_usd FROM items WHERE id='e1_001'").fetchone() == (None,)
        assert conn.execute("SELECT stopped FROM campaign").fetchone() == (1,)


@pytest.mark.parametrize("relative", ["e1/e1_001.receipt.json", "e1-t71-preflight.json",
    "e1/results.private.json", "campaign.json"])
def test_changed_bound_history_is_never_covered_by_exception(authorized, relative):
    bundle, directory, registration = authorized
    with (directory / relative).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(runner.RunError):
        runner.verify_history(bundle, registration)


def test_second_unknown_is_rejected_even_if_artifact_hashes_are_updated(authorized):
    bundle, directory, registration = authorized
    with sqlite3.connect(directory / "campaign.sqlite3") as conn:
        conn.execute("UPDATE items SET status='failed',claim_token=?,error='external_error' WHERE id='e1_002'", ("c" * 64,))
    registration["prior_artifacts"] = artifact_rows(directory)
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    with pytest.raises((runner.RunError, ValueError)):
        runner.verify_history(bundle, registration)


def test_offline_preflight_never_reads_credentials_or_claims_live_support(authorized, monkeypatch):
    bundle, directory, _ = authorized
    monkeypatch.setattr(runner, "credential", lambda *args: pytest.fail("offline read key"))
    before = tree_hashes(directory)
    report = runner.preflight(bundle)
    assert not report["ready_to_send"] and not report["credentials_read"]
    assert report["model_requests"] == 0 and report["historical_unknown_cost_count"] == 1
    assert report["new_unknown_cost_count"] == 0 and report["total_actual_cost_usd"] is None
    assert tree_hashes(directory) == before


def test_two_public_gets_can_qualify_only_the_exact_prior_exception(authorized):
    bundle, directory, _ = authorized
    before = tree_hashes(directory)
    calls, opener = static_metadata(bundle)
    report = runner.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and report["ready_to_send"]
    assert report["historical_unknown_cost_count"] == 1 and report["new_unknown_cost_count"] == 0
    assert report["total_actual_cost_usd"] is None
    assert tree_hashes(directory) == before


def test_exception_does_not_waive_other_metadata_gates(authorized):
    bundle, _, _ = authorized
    calls, opener = static_metadata(bundle, failure=True)
    report = runner.preflight(bundle, online=True, opener=opener)
    assert len(calls) == 2 and not report["ready_to_send"]
    assert "live_metadata_gate_failed" in report["blocking_reasons"]


def test_existing_campaign_blocks_main_before_credential_read(authorized, monkeypatch, capsys):
    bundle, _, _ = authorized
    patch_main(monkeypatch, bundle)
    destination = runner.campaign_directory(bundle["root"])
    destination.mkdir(parents=True)
    calls, opener = static_metadata(bundle)
    real_preflight = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda bundle, online=False, **kw:
                        real_preflight(bundle, online=online, opener=opener))
    monkeypatch.setattr(runner, "credential", lambda *args: pytest.fail("existing campaign read key"))
    assert runner.main(["--execute", "--allow-local-env"]) != 0
    output = json.loads(capsys.readouterr().out)
    assert output.get("ready_to_send") is not True
    assert len(calls) == 2 and output["reason"] == "preflight_not_ready"


def test_execute_without_env_flag_never_reads_key(authorized, monkeypatch, capsys):
    bundle, _, _ = authorized
    patch_main(monkeypatch, bundle)
    monkeypatch.setattr(runner, "credential", lambda *args: pytest.fail("missing flag read key"))
    assert runner.main(["--execute"]) != 0
    capsys.readouterr()
    assert not runner.campaign_directory(bundle["root"]).exists()


def test_credential_reads_only_target_local_assignment_and_ignores_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-process-value-must-be-ignored")
    (tmp_path / ".env").write_text("UNRELATED_BROKEN='unterminated\nOTHER_TOKEN=private-sentinel\n"
        + "OPENROUTER_API_KEY=" + SYNTHETIC_KEY + "\n", encoding="utf-8")
    assert runner.credential(tmp_path) == SYNTHETIC_KEY
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("text", ["OPENROUTER_API_KEY=\n", "OTHER=value\n",
    "OPENROUTER_API_KEY=synthetic\nOPENROUTER_API_KEY=duplicate\n",
    "OPENROUTER_API_KEY=has space\n", "OPENROUTER_API_KEY=非ASCII\n",
    "OPENROUTER_API_KEY=synthetic\x00value\n"])
def test_local_credential_rejects_duplicates_missing_and_invalid_values(tmp_path, text):
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    with pytest.raises(runner.RunError) as caught:
        runner.credential(tmp_path)
    assert "synthetic" not in str(caught.value) and "非ASCII" not in str(caught.value)


def test_local_credential_rejects_link_before_read(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    target.write_text("OPENROUTER_API_KEY=" + SYNTHETIC_KEY, encoding="utf-8")
    is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: True if path == target else is_symlink(path))
    with pytest.raises(runner.RunError):
        runner.credential(tmp_path)


def test_main_first_new_unknown_stops_and_second_execute_cannot_retry(authorized, monkeypatch, capsys):
    bundle, old_directory, _ = authorized
    before = tree_hashes(old_directory)
    patch_main(monkeypatch, bundle)
    calls, opener = static_metadata(bundle)
    real_preflight = runner.preflight
    monkeypatch.setattr(runner, "preflight", lambda bundle, online=False, **kw:
                        real_preflight(bundle, online=online, opener=opener))
    key_reads, posts = [], []

    def fake_credential(*args):
        key_reads.append(True)
        return SYNTHETIC_KEY

    def failed_http(request, *, timeout):
        posts.append(json.loads(request.data))
        raise HTTPError(request.full_url, 404, "private provider text", {}, io.BytesIO(b"{}"))

    def fake_transport(key, **kwargs):
        assert key == SYNTHETIC_KEY
        return t72_transport.OpenRouterTransport(key, opener=failed_http)

    monkeypatch.setattr(runner, "credential", fake_credential)
    monkeypatch.setattr(runner, "OpenRouterTransport", fake_transport)
    assert runner.main(["--execute", "--allow-local-env"]) != 0
    first = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and len(posts) == len(key_reads) == 1
    assert "parallel_tool_calls" not in posts[0]
    assert first["historical_unknown_cost_count"] == first["new_unknown_cost_count"] == 1
    assert first["total_actual_cost_usd"] is None
    directory = runner.campaign_directory(bundle["root"])
    result = previous.legacy.read_json(directory / "e1/results.private.json")
    assert len(result["rows"]) == result["denominator"] == 204
    assert result["campaign"]["stages"]["e1"]["counts"] == {
        "failed": 1, "completed": 0, "started": 0, "not_started": 203}
    assert tree_hashes(old_directory) == before
    assert runner.main(["--execute", "--allow-local-env"]) != 0
    capsys.readouterr()
    assert len(posts) == len(key_reads) == 1
    assert tree_hashes(old_directory) == before


def make_seal(bundle, monkeypatch):
    for relative in runner.FILES - {runner.REGISTRATION}:
        destination = bundle["root"] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / relative, destination)
    seal = {"schema": "aria_t73_authorized_execution_v1",
        "repair_seal_sha256": "d9f4ef6d6119265e574100a1f4fa428cde0bb893da9dbfaf7c18d003b6620f7f",
        "files": [{"path": relative,
            "bytes": (bundle["root"] / relative).stat().st_size,
            "sha256": hashlib.sha256((bundle["root"] / relative).read_bytes()).hexdigest()}
            for relative in sorted(runner.FILES)]}
    seal["seal_sha256"] = previous.freeze.canonical_sha256(seal)
    write_json(bundle["root"] / runner.EXECUTION, seal)
    monkeypatch.setattr(wire_check, "verify_repair", lambda _: {
        "seal_sha256": seal["repair_seal_sha256"]})
    return seal


def test_authorized_seal_validates_local_registration_and_source(authorized, monkeypatch):
    bundle, _, _ = authorized
    seal = make_seal(bundle, monkeypatch)
    assert runner.verify_execution(bundle)["seal_sha256"] == seal["seal_sha256"]


@pytest.mark.parametrize("relative", ["scripts/run_e1_t73.py", "tests/test_e1_t73_run.py",
    "experiments/t73_20261001/authorized_run.json"])
def test_each_authorized_sealed_file_change_is_rejected(authorized, monkeypatch, relative):
    bundle, _, _ = authorized
    make_seal(bundle, monkeypatch)
    with (bundle["root"] / relative).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(runner.RunError):
        runner.verify_execution(bundle)


@pytest.mark.parametrize("change", ["another_unknown", "auto_retry", "larger_budget", "different_campaign"])
def test_resealed_registration_cannot_expand_exact_user_exception(authorized, monkeypatch, change):
    bundle, _, registration = authorized
    if change == "another_unknown":
        registration["authorization"]["historical_unknown_cost_count"] = 2
    elif change == "auto_retry":
        registration["authorization"]["permits_automatic_retry"] = True
    elif change == "larger_budget":
        registration["budget_usd"]["e1"] = "1"
    else:
        registration["authorization"]["applies_only_to_campaign"] = "other_campaign"
    write_json(bundle["root"] / runner.REGISTRATION, registration)
    make_seal(bundle, monkeypatch)
    with pytest.raises(runner.RunError):
        runner.verify_execution(bundle)
