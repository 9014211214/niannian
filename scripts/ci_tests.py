"""Partition Windows software tests and verify all four CI platform results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

WINDOWS_A_FILES = frozenset({
    "tests/test_e1b_pacing_repair.py", "tests/test_e1b_structure_repair.py",
    "tests/test_e1b_reflection_repair.py",
})
PARTITIONS = frozenset({"all", "A", "B"})


def partition_for(nodeid: str) -> str:
    return "A" if nodeid.split("::", 1)[0] in WINDOWS_A_FILES else "B"


def select_nodes(nodes: list[str], partition: str) -> list[str]:
    if partition not in PARTITIONS or len(nodes) != len(set(nodes)):
        raise ValueError("invalid_partition_or_duplicate_nodeid")
    return [node for node in nodes if partition == "all" or partition_for(node) == partition]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_tests(partition: str, output: Path, *, collect_only: bool = False) -> int:
    import pytest

    output.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, str] = {}

    class Evidence:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, session, config, items):
            all_nodes = [item.nodeid for item in items]
            selected = select_nodes(all_nodes, partition)
            selected_set = set(selected)
            excluded = [item for item in items if item.nodeid not in selected_set]
            items[:] = [item for item in items if item.nodeid in selected_set]
            if excluded:
                config.hook.pytest_deselected(items=excluded)
            write_json(output / "collection.json", {
                "schema": "aria_ci_collection_v1", "partition": partition,
                "all_nodes": all_nodes, "selected_nodes": selected,
                "partition_files": sorted(WINDOWS_A_FILES),
                "collect_only": collect_only,
            })

        def pytest_runtest_logreport(self, report):
            if report.failed:
                outcomes[report.nodeid] = "failed"
            elif outcomes.get(report.nodeid) != "failed":
                if report.skipped:
                    outcomes[report.nodeid] = "skipped"
                elif report.when == "call":
                    outcomes[report.nodeid] = "passed"

        def pytest_sessionfinish(self, session, exitstatus):
            write_json(output / "execution.json", {
                "schema": "aria_ci_execution_v1", "partition": partition,
                "exit_code": int(exitstatus), "collect_only": collect_only,
                "outcomes": outcomes,
            })

    args = ["-q", "-ra", "--durations=15"]
    args += ["--collect-only"] if collect_only else ["--junitxml=" + str(output / "pytest.xml")]
    sys.path.insert(0, str(Path.cwd()))
    return int(pytest.main(args, plugins=[Evidence()]))


def verify_cell(records: list[dict], *, windows: bool) -> dict:
    expected = {"A", "B"} if windows else {"all"}
    if len(records) != len(expected) or {r["partition"] for r in records} != expected:
        raise ValueError("missing_or_duplicate_partition")
    universe = records[0]["collection"]["all_nodes"]
    if len(universe) != len(set(universe)):
        raise ValueError("duplicate_collected_nodeid")
    selected_union: set[str] = set()
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for record in records:
        collection, execution = record["collection"], record["execution"]
        part = record["partition"]
        if (collection["all_nodes"] != universe or collection["partition"] != part
                or execution["partition"] != part or collection["collect_only"]
                or execution["collect_only"] or execution["exit_code"] != 0):
            raise ValueError("collection_or_execution_invalid")
        selected = collection["selected_nodes"]
        if selected != select_nodes(universe, part) or selected_union.intersection(selected):
            raise ValueError("partition_overlap_or_selection_changed")
        if set(execution["outcomes"]) != set(selected):
            raise ValueError("unexecuted_or_extra_test")
        if any(value not in {"passed", "skipped"} for value in execution["outcomes"].values()):
            raise ValueError("test_execution_failed")
        xml_totals = record["xml_totals"]
        if (xml_totals["tests"] != len(selected) or xml_totals["failures"]
                or xml_totals["errors"] or xml_totals["skipped"] != sum(
                    value == "skipped" for value in execution["outcomes"].values())):
            raise ValueError("junit_coverage_or_outcome_mismatch")
        selected_union.update(selected)
        for key in totals:
            totals[key] += xml_totals[key]
    if selected_union != set(universe):
        raise ValueError("partition_union_incomplete")
    installers = [r for r in records if r.get("install") is not None]
    if (len(installers) != 1 or installers[0]["partition"] != ("B" if windows else "all")
            or installers[0]["install"].get("status") != "pass"):
        raise ValueError("clean_install_missing_or_duplicate")
    return {**totals, "passed": totals["tests"] - totals["skipped"],
            "partitions": sorted(expected), "full_collection": len(universe),
            "disjoint_and_complete": True, "package_install_runs": 1}


def summarize(directory: Path, output: Path, commit: str) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(directory.glob("*/context.json")):
        context = json.loads(path.read_text("utf-8"))
        if (context["source_commit"] != commit or context["run_id"] != os.environ["GITHUB_RUN_ID"]
                or context["run_attempt"] != os.environ["GITHUB_RUN_ATTEMPT"]):
            raise ValueError("artifact_commit_or_run_changed")
        runtime = ".".join(context["python"].split(".")[:2])
        record = {"partition": context["partition"]}
        for name in ("collection", "execution"):
            record[name] = json.loads((path.parent / (name + ".json")).read_text("utf-8"))
        xml = ET.parse(path.parent / "pytest.xml")
        record["xml_totals"] = {key: sum(int(s.get(key, "0")) for s in xml.iter("testsuite"))
                                for key in ("tests", "failures", "errors", "skipped")}
        install = path.parent / "install/result.json"
        record["install"] = json.loads(install.read_text("utf-8")) if install.is_file() else None
        groups.setdefault((context["runner_os"], runtime), []).append(record)
    expected = {(os_name, py) for os_name in ("Windows", "Linux") for py in ("3.10", "3.12")}
    if set(groups) != expected:
        raise ValueError("four_platform_matrix_incomplete")
    result = {"schema": "aria_ci_four_platform_summary_v1", "source_commit": commit,
              "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
              "status": "pass", "model_effect_claim": False,
              "platforms": {" / ".join(key): verify_cell(rows, windows=key[0] == "Windows")
                            for key, rows in sorted(groups.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--partition", choices=sorted(PARTITIONS), required=True)
    run.add_argument("--output", type=Path, default=Path("artifacts/ci"))
    run.add_argument("--collect-only", action="store_true")
    summary = sub.add_parser("summary")
    summary.add_argument("--directory", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_tests(args.partition, args.output, collect_only=args.collect_only)
    try:
        result = summarize(args.directory, args.output, args.commit)
    except (KeyError, ValueError, OSError, ET.ParseError):
        # Never turn missing/failed/cancelled partitions into a partial pass.
        print("Four-platform software evidence is incomplete or inconsistent.")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
