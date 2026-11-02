"""Check wheel/sdist installs on Windows and Linux; never call a model or publish.

The default creates two clean environments and lets pip resolve dependencies.
--reuse-local-dependencies is an explicitly weaker, offline local check: it
exposes the invoking interpreter's installed dependencies, but the Aria package
must still come from each newly installed artifact. Neither mode reads .env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import venv
import zipfile


ASSETS = (
    "assets/strategy_matrix.yaml", "assets/personas/laochen_demo.yaml",
    "assets/voice/minimal.html", "assets/voice/care.html",
    "bench/cases/refocus_e1_v1.yaml", "bench/datasets/cped_v1.yaml",
    "scenes/companion_elder/elder_kb.yaml",
    "bench/cases/fixed_safety_v1/manifest.json",
    "bench/cases/fixed_safety_v1/inputs.json",
    "bench/cases/fixed_safety_v1/references.json",
    "bench/cases/fixed_safety_v1/protocol.json",
)
LICENSE_FILES = {"LICENSE", "Apache-2.0.txt", "THIRD_PARTY_NOTICES.md"}
SOURCE_ENTRY_FILES = {
    "docs/FIXED_SAFETY_REGRESSION.md",
    "docs/E1BR_RESULTS_2026_10_05.md",
    "scripts/run_e1b_main_continuation.py", "scripts/e1b_main_continuation.py",
    "scripts/run_e1br.py", "scripts/e1br_qualification.py", "scripts/e1br_assessment.py",
    "scripts/run_e1br_received_continuation.py", "scripts/e1br_received_body.py",
    "scripts/e1br_checkpoint.py", "scripts/run_e1br_checkpoint_continuation.py",
    "scripts/e1br_checkpoint_guard.py", "scripts/ci_tests.py",
    "scripts/run_e1br_format_continuation.py", "scripts/e1br_format_checkpoint.py",
    "scripts/e1br_format_runtime.py", "scripts/e1br_format_legacy.py",
    "scripts/e1br_format_qualification.py",
    "scripts/e1c_execution.py", "scripts/e1c_calls.py", "scripts/e1c_qualification.py",
    "scripts/e1c_scoring.py", "scripts/run_e1c.py",
    "scripts/e1c_v2_execution.py", "scripts/e1c_v2_calls.py",
    "scripts/e1c_v2_qualification.py", "scripts/e1c_v2_scoring.py", "scripts/run_e1c_v2.py",
    "docs/E1C_V2_EXECUTION.md", "docs/DESIGN_AUDIT.md", "docs/DESIGN_REPAIR_2026_10_06.md",
    "scripts/run_e1br_memory_continuation.py", "scripts/e1br_memory_checkpoint.py",
    "scripts/e1br_memory_runtime.py", "scripts/e1br_memory_legacy.py",
    "scripts/e1br_memory_qualification.py", "docs/E1BR_MEMORY_CONTINUATION.md",
    "scripts/e1br_prewrite_checkpoint.py", "scripts/e1br_prewrite_runtime.py",
    "scripts/run_e1br_prewrite_continuation.py", "docs/E1BR_PREWRITE_CONTINUATION.md",
}


class InstallCheckError(ValueError):
    """Safe diagnostic codes only, without subprocess text or local paths."""


def python_in(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def safe_environment() -> dict[str, str]:
    # Do not access inherited credential values, even though no model is used.
    environment = {
        name: os.environ[name] for name in os.environ
        if not name.startswith(("ARIA_", "OPENROUTER_", "DASHSCOPE_", "ALIYUN_", "VOLC_"))
        and not name.endswith(("_TOKEN", "_KEY", "_SECRET", "_PASSWORD"))
        and not name.startswith(("PYTHON", "PIP_"))
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CONFIG_FILE=os.devnull)
    return environment


def run(arguments: list[str], directory: Path, phase: str) -> bytes:
    result = subprocess.run(arguments, cwd=directory, env=safe_environment(),
                            capture_output=True, timeout=300, check=False)
    if result.returncode:
        raise InstallCheckError(f"{phase}_failed_exit_{result.returncode}")
    return result.stdout


def artifacts(directory: Path) -> tuple[Path, Path]:
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise InstallCheckError("require_exactly_one_wheel_and_one_sdist")
    return wheels[0].resolve(), sources[0].resolve()


def inspect_members(names: list[str], *, sdist: bool) -> dict:
    relative = []
    for name in names:
        value = PurePosixPath(name)
        if (value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name
                or value.as_posix() != name):
            raise InstallCheckError("unsafe_distribution_member")
        parts = value.parts[1:] if sdist else value.parts
        if not parts:
            continue
        if any(part in {".git", ".venv", "data", "bench_reports", ".cache"} for part in parts):
            raise InstallCheckError("private_distribution_member")
        if parts[:2] in {("artifacts", "local"), ("docs", "media")}:
            raise InstallCheckError("private_distribution_member")
        if any((part == ".env" or part.startswith(".env.")) and part != ".env.example"
               for part in parts):
            raise InstallCheckError("credential_distribution_member")
        relative.append("/".join(parts))
    included = {PurePosixPath(name).name for name in relative}
    if not LICENSE_FILES <= included:
        raise InstallCheckError("distribution_license_missing")
    if any("aria/" + name not in relative for name in ASSETS):
        raise InstallCheckError("distribution_asset_missing")
    if sdist and not SOURCE_ENTRY_FILES <= set(relative):
        raise InstallCheckError("distribution_source_entry_missing")
    return {"files": len(relative), "required_assets": len(ASSETS),
            "license_files": sorted(LICENSE_FILES)}


def inspect_distribution(path: Path) -> dict:
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            names = [row.filename for row in archive.infolist() if not row.is_dir()]
        result = inspect_members(names, sdist=False)
    else:
        with tarfile.open(path, "r:gz") as archive:
            rows = archive.getmembers()
            if any(not (row.isfile() or row.isdir()) for row in rows):
                raise InstallCheckError("sdist_link_or_special_member")
            names = [row.name for row in rows if row.isfile()]
            if len({PurePosixPath(name).parts[0] for name in names}) != 1:
                raise InstallCheckError("sdist_root_invalid")
        result = inspect_members(names, sdist=True)
    return {"artifact": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result}



def probe_fixed_safety_suite(installed: Path) -> dict:
    """Read the actual installed runner and fixed assets; never execute models."""
    from importlib.resources import files
    from aria.bench import safety_regression

    if not Path(safety_regression.__file__).resolve().is_relative_to(installed):
        raise InstallCheckError("safety_runner_not_from_installed_package")
    suite = Path(str(files("aria").joinpath("bench/cases/fixed_safety_v1"))).resolve()
    if not suite.is_relative_to(installed):
        raise InstallCheckError("safety_suite_not_from_installed_package")
    manifest, cases = safety_regression.load_suite(suite)
    return {"safety_runner_installed_origin": True, "fixed_safety_cases": len(cases),
            "fixed_safety_cells": len(manifest["cells"]), "fixed_safety_version": manifest["version"]}


def probe() -> dict:
    """Exercise the installed package through actual loopback HTTP and storage."""
    import http.client
    from importlib.resources import files
    from datetime import datetime, timedelta, timezone
    import threading
    import urllib.request

    import aria
    from aria import _env
    from aria.engine.minimal_care import MinimalCareService
    from aria.voice.minimal import MinimalWebApp, make_minimal_server

    installed = Path(sysconfig.get_path("purelib")).resolve()
    if not Path(aria.__file__).resolve().is_relative_to(installed):
        raise InstallCheckError("aria_imported_from_source_or_dependency_reuse")
    for name in ASSETS:
        if not files("aria").joinpath(name).is_file():
            raise InstallCheckError("installed_resource_missing")

    def forbidden(*_args, **_kwargs):
        raise InstallCheckError("external_request_or_env_read_forbidden")

    _env.load_local_env = forbidden
    urllib.request.urlopen = forbidden
    safety_suite = probe_fixed_safety_suite(installed)
    current = [datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)]
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="preview-") as scratch:
        directory = Path(scratch)
        care = MinimalCareService(directory / "care.sqlite3", clock=lambda: current[0])
        password = "synthetic-install-check"  # release-fixture: not-a-deployment-secret
        care.bootstrap("fixture-carer", password, ["fictional-chen"])
        care.bootstrap("fixture-other", password, ["other-elder"])
        app = MinimalWebApp(directory / "app", care_service=care)
        server = make_minimal_server(app, port=0)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()

        def request(route, data=None, *, token=True, cookie=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-Aria-Demo-Token"] = app.demo_token
            if cookie:
                headers["Cookie"] = cookie
            connection.request("GET" if data is None else "POST", route,
                               json.dumps(data) if data is not None else None, headers)
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result

        try:
            for page in ("/", "/care"):
                status, _, body = request(page)
                assert status == 200 and b"<html" in body.lower()
            assert request("/api/minimal/state", token=False)[0] == 403
            for index in range(3):
                status, _, body = request("/api/minimal/start", {})
                assert status == 200
                session = json.loads(body)["state"]["session_id"]
                assert app.submit("今天先安静坐一会儿。", f"install-turn-{index}", session)["ok"]
                assert app.seal(session)["ok"]
            state = app.state()
            assert not state["model_calls_enabled"] and not any(state["services"].values())
            assert len(state["sessions"]) == len(state["turns"]) == 3
            assert all(row["source"] == "local_preview" for row in state["turns"])
            assert not state["profile"]["facts"] and not state["profile"]["hypotheses"]
            assert request("/api/care/state")[0] == 403
            status, headers, _ = request("/api/care/login", {
                "username": "fixture-carer", "password": password})
            assert status == 200 and "HttpOnly" in headers["Set-Cookie"]
            cookie = headers["Set-Cookie"].split(";", 1)[0]
            assert request("/api/care/state?elder_id=other-elder", cookie=cookie)[0] == 403
            rule = app.care_action("reminder.create", {
                "request_id": "install-reminder", "label": "准备纸和本子", "schedule": "once",
                "date": "2026-10-01", "time": "15:30", "timezone": "Asia/Shanghai",
                "grace_minutes": 30})["record"]
            contact = app.care_action("contact.create", {
                "request_id": "install-contact", "name": "虚构女儿"})["record"]
            consent = app.care_action("consent.grant", {
                "request_id": "install-consent", "contact_id": contact["id"],
                "reminder_id": rule["id"], "trigger": "unconfirmed", "scope": "reminder_status",
                "source": "simulation", "confirmed": True,
                "expires_at": "2026-11-01T00:00:00+00:00"})["record"]
            current[0] += timedelta(minutes=30)
            care.tick()
            app.care_action("consent.revoke", {
                "request_id": "install-revoke", "consent_id": consent["id"]})
            current[0] += timedelta(minutes=30)
            care.tick()
            assert care.elder_view("fictional-chen")["deliveries"] == []
            before = app.state()
            for _ in range(3):
                assert request("/api/minimal/state")[0] == 200
            assert app.state() == before
        finally:
            server.shutdown()
            serving.join(timeout=5)
            server.server_close()
    return {"version": aria.__version__, "installed_origin": True, "assets": len(ASSETS),
            **safety_suite,
            "loopback_pages": 2, "preview_sessions": 3, "care_scope_and_revocation": True,
            "readonly_replay_basis": True, "model_requests": 0, "dotenv_reads": 0}


def check_installs(dist: Path, work: Path, *, reuse: bool = False) -> dict:
    wheel, source = artifacts(dist)
    summaries = [inspect_distribution(path) for path in (wheel, source)]
    # mkdir(exist_ok=False) prevents reuse of an old environment or result.
    work.mkdir(parents=True, exist_ok=False)
    run([sys.executable, "-m", "twine", "check", "--strict", str(wheel), str(source)],
        work, "metadata")
    for kind, package, summary in zip(("wheel", "sdist"), (wheel, source), summaries):
        environment = work / kind
        # Unix symlinks retain the base runtime when invoked inside a managed
        # Python 3.10 venv; copied binaries can otherwise lose their stdlib path.
        # Windows keeps its standard copied launchers and needs no symlink rights.
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
        executable = str(python_in(environment))
        if reuse:
            # A path line does not execute the parent's editable-install .pth files.
            site = environment / ("Lib/site-packages" if os.name == "nt" else
                                  f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
            (site / "_aria_local_dependencies.pth").write_text(
                sysconfig.get_path("purelib") + "\n", encoding="utf-8")
        options = (["--no-index", "--no-deps", "--no-build-isolation"] if reuse else
                   ["--no-cache-dir", "--index-url", "https://pypi.org/simple"])
        run([executable, "-m", "pip", "install", *options, str(package)], work, kind + "_install")
        run([executable, "-m", "pip", "check"], work, kind + "_dependencies")
        summary["installed_packages"] = json.loads(run(
            [executable, "-m", "pip", "list", "--format=json"], work, kind + "_versions"))
        cli = environment / ("Scripts/aria-web.exe" if os.name == "nt" else "bin/aria-web")
        run([str(cli), "--help"], work, kind + "_cli")
        result = run([executable, "-I", str(Path(__file__).resolve()), "--probe"],
                     work, kind + "_runtime")
        summary["runtime"] = json.loads(result)
    return {"schema": "aria_install_check_v1", "status": "pass", "python": sys.version.split()[0],
            "platform": sys.platform, "dependency_mode": "local_reused_not_cold" if reuse else "clean_resolved",
            "package_index": None if reuse else "https://pypi.org/simple",
            "distributions": summaries, "model_requests": 0, "published": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--reuse-local-dependencies", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.probe:
            result = probe()
        else:
            work = (args.work_dir.resolve() if args.work_dir else
                    Path(tempfile.gettempdir()) / ("aria-install-" + os.urandom(8).hex()))
            result = check_installs(args.dist_dir.resolve(), work, reuse=args.reuse_local_dependencies)
            with (work / "result.json").open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2)
                output.write("\n")
    except (InstallCheckError, OSError, ValueError, AssertionError, subprocess.TimeoutExpired,
            zipfile.BadZipFile, tarfile.TarError) as error:
        result = {"status": "error", "error": str(error) if isinstance(error, InstallCheckError)
                  else type(error).__name__}
        print(json.dumps(result))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
