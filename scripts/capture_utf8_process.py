"""Capture a noninteractive child without shell text decoding.

Use this around the smoke or formal entry point instead of PowerShell text
redirection. stdout.txt and stderr.txt contain the exact child bytes, even if
they are invalid UTF-8. A fresh output directory is required for every process.
This launcher neither retries nor changes the child's evaluation/budget rules.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def _stream_receipt(path):
    data = path.read_bytes()
    try:
        data.decode("utf-8", errors="strict")
        valid, invalid_at = True, None
    except UnicodeDecodeError as error:
        valid, invalid_at = False, error.start
    return {
        "path": str(path), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "utf8_valid": valid, "invalid_byte_offset": invalid_at,
        "transcoded": False,
    }


def capture_process(command, output, *, cwd=None):
    """Run once, preserving separate native streams and the actual exit code."""
    if not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("a nonempty child command is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    stdout_path, stderr_path = output / "stdout.txt", output / "stderr.txt"
    returncode, error_class = None, None
    environment = dict(os.environ)
    environment.update(PYTHONIOENCODING="utf-8:strict", PYTHONUNBUFFERED="1")
    # Binary file handles bypass PowerShell's native-output decoding entirely.
    # No PIPEs also means neither stream can deadlock while the other is full.
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            result = subprocess.run(
                command, cwd=cwd, env=environment, shell=False,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                check=False,
            )
            returncode = result.returncode
        except OSError as error:
            error_class = type(error).__name__
    receipt = {
        "schema": "aria_raw_utf8_process_capture_v1",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "command_sha256": hashlib.sha256(json.dumps(
            command, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "shell": False, "attempts": 1,
        "child_exit_code": returncode, "launch_error_class": error_class,
        "stdout": _stream_receipt(stdout_path),
        "stderr": _stream_receipt(stderr_path),
    }
    receipt["capture_utf8_valid"] = (
        receipt["stdout"]["utf8_valid"] and receipt["stderr"]["utf8_valid"]
    )
    with (output / "process-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    receipt = capture_process(command, args.output, cwd=args.cwd)
    # ASCII only: the invoking shell may still use an unrelated code page.
    # The UTF-8 child output is read directly from the two captured files.
    print(json.dumps({"receipt": str(args.output.resolve() / "process-receipt.json"),
                      "child_exit_code": receipt["child_exit_code"],
                      "capture_utf8_valid": receipt["capture_utf8_valid"]},
                     ensure_ascii=True), flush=True)
    if receipt["launch_error_class"]:
        return 126
    if not receipt["capture_utf8_valid"]:
        return receipt["child_exit_code"] or 2
    return receipt["child_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
