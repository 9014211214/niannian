"""E1c actual product assembly with named local stubs, no holdout questions."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from aria.bench.e1b_runner import E1bLocalFixtureClient, E1bRunner
from aria.bench.e1c_runner import E1cExperimentApp, E1cRunner
from aria.bench.e1c_source import load_first_submission, seal_first_submission
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.runtime import MinimalModelClients
from e1c_software_fixture import fixture_author, fixture_payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("E1c software integration cannot access network")
    stub_model_http(monkeypatch, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def setup(tmp_path):
    original = json.dumps(fixture_payload(), ensure_ascii=False).encode()
    source = tmp_path / "first"
    seal_first_submission(original, fixture_author(), source)
    sha = hashlib.sha256((source / "receipt.json").read_bytes()).hexdigest()
    token = SyntheticEmotionExperiment.create(tmp_path, "e1c-software-only")
    records = []
    def factory(path):
        clients = MinimalModelClients(*(E1bLocalFixtureClient(role) for role in
            ("online", "memory", "reflection")), model_calls_enabled=True)
        clients.reflection.reflection_return_format = "body_json"
        app = E1cExperimentApp(path, experiment=token, clients=clients)
        records.append((app, clients))
        return app
    return {"source_directory": source, "source_receipt_sha256": sha,
            "root": token.root / "run", "seed": 20261005, "app_factory": factory}, records


def test_e1b_default_single_source_semantics_remain():
    from aria.engine.model.persona import EpisodicMemoryEntry, PersonaProfile
    profile = PersonaProfile(user_id="SOFTWARE_ONLY", episodic_memory=[EpisodicMemoryEntry(
        id=name, summary="软件夹具", source_session_id=session, source_turn_index=4)
        for name, session in (("keep", "S1"), ("remove", "S2"))])
    result, removed = E1bRunner._a0_profile(None, profile, SimpleNamespace(pref_turn=4), {1: "S1", 2: "S2"})
    assert [row.id for row in result.episodic_memory] == ["keep"]
    assert removed["facts"] == ["remove"]


@pytest.mark.parametrize("configuration", [{"allow_model_calls": True}, {"evidence_kind": "real_model"}])
def test_paid_path_refuses_before_factory_or_output_creation(tmp_path, configuration):
    args, records = setup(tmp_path)
    with pytest.raises(PermissionError, match="separate_frozen_authorization"):
        E1cRunner(**args, **configuration)
    assert not records and not args["root"].exists()


def test_first_source_change_and_order_rejected_before_any_product_call(tmp_path):
    args, records = setup(tmp_path)
    runner = E1cRunner(**args)
    with pytest.raises(ValueError, match="no_prescreen"):
        runner.run_case("e1c_a1", phase="prescreen")
    with pytest.raises(ValueError, match="first_author_order"):
        runner.run_case("e1c_a2")
    source = args["source_directory"] / "author-original.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="first_source_sha_mismatch"):
        runner.run_case("e1c_a1")
    assert not records


@pytest.mark.parametrize("path", ["author-original.json", "author-provenance.json", "receipt.json"])
def test_each_first_source_file_is_hash_bound(tmp_path, path):
    args, _ = setup(tmp_path)
    target = args["source_directory"] / path
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha_mismatch"):
        load_first_submission(args["source_directory"], expected_receipt_sha256=args["source_receipt_sha256"],
                              allow_software_fixture=True)


def test_first_source_reparse_flag_is_rejected_before_read(tmp_path, monkeypatch):
    # Windows attribute simulation, not a claim of OS-level symlink coverage.
    from pathlib import Path
    args, _ = setup(tmp_path)
    original = Path.lstat
    def flagged(path):
        value = original(path)
        if path.name == "author-provenance.json":
            return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=0x400)
        return value
    monkeypatch.setattr(Path, "lstat", flagged)
    with pytest.raises(ValueError, match="links_forbidden"):
        load_first_submission(args["source_directory"], expected_receipt_sha256=args["source_receipt_sha256"],
                              allow_software_fixture=True)


def test_failed_source_persistence_cannot_be_replaced_by_later_success(tmp_path, monkeypatch):
    from aria.bench import e1c_source
    raw = b'{"SOFTWARE_ONLY":"first"}'
    original = e1c_source._write_once
    def fail_provenance(path, value):
        if path.name == "author-provenance.json":
            raise OSError("SOFTWARE_ONLY simulated disk failure")
        original(path, value)
    monkeypatch.setattr(e1c_source, "_write_once", fail_provenance)
    source = tmp_path / "first"
    with pytest.raises(OSError):
        seal_first_submission(raw, fixture_author(), source)
    assert (source / "author-original.json").read_bytes() == raw
    assert not (source / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        seal_first_submission(b'{}', fixture_author(), source)


def test_one_case_full_product_path_and_explicit_e1c_report(tmp_path):
    args, records = setup(tmp_path)
    runner = E1cRunner(**args)
    frozen_plan = deepcopy(runner.plan)
    assert all(row["source"].startswith("simulated_e1c:") for row in frozen_plan.values())
    report = runner.run_case("e1c_a1")
    assert report["version"] == "e1c_product_runner_v1_unfrozen"
    assert report["experiment"] == "e1c" and report["suite_version"] == "e1c_format_v1"
    assert report["evidence_kind"] == "software_fixture" and report["effects"] == "not_evaluated"
    assert report["source_sha256"] == runner.suite.source_sha256
    assert report["first_receipt_sha256"] == args["source_receipt_sha256"]
    assert report["scored_reply_count"] == 18 and len(report["branches"]) == 6
    assert report["planned_reply_denominator"] == 216 and report["planned_basic_positions"] == 1056
    assert report["retained_experimental_answers"] == 0 and runner.plan == frozen_plan
    assert sum(len(clients.online.calls) for _, clients in records) == 84
    assert sum(len(clients.memory.calls) for _, clients in records) == 2
    assert sum(len(clients.reflection.calls) for _, clients in records) == 2
    assert all(len(branch["removed"]["per_source"]) == 2 for branch in report["branches"] if branch["arm"] == "A0")
    assert all(not app.care.external and not app.care.allow_external for app, _ in records)
    for _, clients in records:
        for client in (clients.online, clients.memory, clients.reflection):
            assert "软件标注占位，不是语义答案" not in str(client.calls)
            assert "preference_sources" not in str(client.calls)
    saved = json.loads((args["root"] / "main/e1c_a1/report.json").read_text(encoding="utf-8"))
    assert saved == report and saved["shared"]["state"]["experiment"] == "e1c"
    with pytest.raises(ValueError, match="first_author_order"):
        runner.run_case("e1c_a1")
