from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.ci_tests import select_nodes, summarize, verify_cell


NODES = ["tests/test_e1b_pacing_repair.py::test_full_chain",
         "tests/test_care.py::test_access", "tests/test_other.py::test_skipped"]


def records(windows=True):
    rows = []
    for part in ("A", "B") if windows else ("all",):
        selected = select_nodes(NODES, part)
        outcomes = {node: "skipped" if node.endswith("test_skipped") else "passed" for node in selected}
        rows.append({"partition": part,
                     "collection": {"partition": part, "all_nodes": NODES[:],
                                    "selected_nodes": selected, "collect_only": False},
                     "execution": {"partition": part, "exit_code": 0, "collect_only": False,
                                   "outcomes": outcomes},
                     "xml_totals": {"tests": len(selected), "failures": 0, "errors": 0,
                                    "skipped": list(outcomes.values()).count("skipped")},
                     "install": {"status": "pass"} if part != "A" else None})
    return rows


def test_full_collection_partition_is_disjoint_and_complete():
    left, right = select_nodes(NODES, "A"), select_nodes(NODES, "B")
    assert not set(left) & set(right)
    assert set(left) | set(right) == set(select_nodes(NODES, "all"))
    assert select_nodes(["tests/test_new_file.py::test_new"], "B") == ["tests/test_new_file.py::test_new"]


@pytest.mark.parametrize("windows", [True, False])
def test_cell_summary_counts_skips_without_counting_them_as_passes(windows):
    result = verify_cell(records(windows), windows=windows)
    assert result["tests"] == 3 and result["passed"] == 2 and result["skipped"] == 1
    assert result["package_install_runs"] == 1 and result["disjoint_and_complete"]


@pytest.mark.parametrize("corruption", ["missing_partition", "overlap", "collection_changed", "missing_execution",
                                        "failed_execution", "only_collected", "junit_missing", "skip_mismatch",
                                        "no_install", "double_install", "failed_install"])
def test_missing_failed_or_duplicate_evidence_cannot_pass(corruption):
    rows = deepcopy(records())
    if corruption == "missing_partition":
        rows.pop()
    elif corruption == "overlap":
        rows[1]["collection"]["selected_nodes"].append(NODES[0])
    elif corruption == "collection_changed":
        rows[1]["collection"]["all_nodes"].append("tests/test_new.py::test_new")
    elif corruption == "missing_execution":
        rows[0]["execution"]["outcomes"].clear()
    elif corruption == "failed_execution":
        rows[0]["execution"]["outcomes"][NODES[0]] = "failed"
    elif corruption == "only_collected":
        rows[0]["execution"]["collect_only"] = True
    elif corruption == "junit_missing":
        rows[0]["xml_totals"]["tests"] = 0
    elif corruption == "skip_mismatch":
        rows[1]["xml_totals"]["skipped"] = 0
    elif corruption == "no_install":
        rows[1]["install"] = None
    elif corruption == "double_install":
        rows[0]["install"] = {"status": "pass"}
    else:
        rows[1]["install"]["status"] = "fail"
    with pytest.raises(ValueError):
        verify_cell(rows, windows=True)


def test_four_platform_summary_rejects_other_commit_and_missing_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    commit = "a" * 40
    for os_name in ("Windows", "Linux"):
        for version in ("3.10", "3.12"):
            for record in records(os_name == "Windows"):
                part = record["partition"]
                directory = tmp_path / f"{os_name}-{version}-{part}"
                directory.mkdir()
                context = {"source_commit": commit, "run_id": "123", "run_attempt": "1",
                           "python": version + ".1", "runner_os": os_name, "partition": part}
                for name, data in (("context", context), ("collection", record["collection"]),
                                   ("execution", record["execution"])):
                    (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
                totals = record["xml_totals"]
                attrs = " ".join(f'{key}="{value}"' for key, value in totals.items())
                (directory / "pytest.xml").write_text(f"<testsuites><testsuite {attrs}/></testsuites>", encoding="utf-8")
                if record["install"]:
                    (directory / "install").mkdir()
                    (directory / "install/result.json").write_text(json.dumps(record["install"]), encoding="utf-8")
    assert summarize(tmp_path, tmp_path / "four-platforms.json", commit)["status"] == "pass"
    with pytest.raises(ValueError, match="artifact_commit"):
        summarize(tmp_path, tmp_path / "bad.json", "b" * 40)
    (tmp_path / "Windows-3.10-A/context.json").unlink()
    with pytest.raises(ValueError, match="missing_or_duplicate_partition"):
        summarize(tmp_path, tmp_path / "bad.json", commit)


def test_pytest_plugin_records_real_execution_without_losing_skips(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_e1b_pacing_repair.py").write_text(
        "import pytest\ndef test_pass(): assert True\n"
        "@pytest.mark.skip(reason='synthetic platform skip')\ndef test_skip(): assert False\n", encoding="utf-8")
    (tests / "test_other.py").write_text("def test_other(): assert True\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/ci_tests.py"
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE"}}
    env["PYTHONUTF8"] = "1"
    records = []
    for part in ("A", "B"):
        output = tmp_path / "output" / part
        completed = subprocess.run([sys.executable, "-I", str(script), "run", "--partition", part,
                                    "--output", str(output)], cwd=tmp_path, env=env,
                                   capture_output=True, timeout=30)
        assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
        xml = ET.parse(output / "pytest.xml")
        records.append({"partition": part,
                        "collection": json.loads((output / "collection.json").read_text("utf-8")),
                        "execution": json.loads((output / "execution.json").read_text("utf-8")),
                        "xml_totals": {key: sum(int(s.get(key, "0")) for s in xml.iter("testsuite"))
                                       for key in ("tests", "failures", "errors", "skipped")},
                        "install": {"status": "pass"} if part == "B" else None})
    actual = verify_cell(records, windows=True)
    assert actual["full_collection"] == 3 and actual["passed"] == 2 and actual["skipped"] == 1
