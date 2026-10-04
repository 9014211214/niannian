"""CLI boundary checks use synthetic in-memory corpus only."""
import json
from types import SimpleNamespace

import pytest

from examples import prepare_cped_holdout as cli


def _allowed():
    return ["--data", "private-source", "--acknowledge-unverified-data-license", "--accept-aggregate-only"]


def test_cli_requires_both_acknowledgements_before_data_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected source access")
    monkeypatch.setattr(cli, "load_cped_corpus", forbidden)
    for args in (["--data", "private"], _allowed()[:-1]):
        with pytest.raises(SystemExit) as exc:
            cli.main(args)
        assert exc.value.code == 2


def test_cli_outputs_only_aggregate_failure_and_safe_fingerprints(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_cped_corpus", lambda *args, **kwargs: SimpleNamespace(rows=[], sha256={"private-source.csv": "a" * 64}))
    assert cli.main(_allowed()) == 2
    output = capsys.readouterr().out
    report = json.loads(output)
    assert "private-source" not in output
    assert report["preparation"]["executed_model_calls"] == 0
    assert report["preparation"]["status"] == "stopped_feasibility"
    assert report["private_working_cases_exported"] is False
    assert report["source_file_sha256"] == ["a" * 64]


def test_cli_never_overwrites_an_existing_report_or_source(tmp_path, monkeypatch):
    existing = tmp_path / "existing.json"
    existing.write_text("existing material", encoding="utf-8")
    monkeypatch.setattr(cli, "load_cped_corpus", lambda *args, **kwargs: pytest.fail("must not load"))
    with pytest.raises(SystemExit):
        cli.main(_allowed() + ["--output", str(existing)])
    assert existing.read_text(encoding="utf-8") == "existing material"


def test_source_exception_does_not_echo_restricted_text(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("private character and original dialogue")
    monkeypatch.setattr(cli, "load_cped_corpus", fail)
    with pytest.raises(SystemExit):
        cli.main(_allowed())
    output = capsys.readouterr()
    assert "private character" not in output.err + output.out


def test_cli_invalid_seed_stops_before_source_access(monkeypatch):
    monkeypatch.setattr(cli, "load_cped_corpus", lambda *args, **kwargs: pytest.fail("must not load"))
    with pytest.raises(SystemExit):
        cli.main(_allowed() + ["--seed", "-1"])
