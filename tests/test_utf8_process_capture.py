"""Real local subprocess checks; no model or network calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.capture_utf8_process import capture_process


ROOT = Path(__file__).resolve().parents[1]


def test_original_bytes_and_exit_are_preserved_under_conflicting_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    out_text, err_text = "陈师傅，这样记下对吗？\n", "本地测试错误：拒绝\n"
    code = ("import sys; print(" + repr(out_text) + ", end='', flush=True); "
            "print(" + repr(err_text) + ", end='', file=sys.stderr, flush=True); "
            "raise SystemExit(7)")
    receipt = capture_process([sys.executable, "-c", code], tmp_path / "capture")
    for key, text in (("stdout", out_text), ("stderr", err_text)):
        expected = text.replace("\n", os.linesep).encode("utf-8")
        assert Path(receipt[key]["path"]).read_bytes() == expected
        assert receipt[key]["sha256"] == hashlib.sha256(expected).hexdigest()
        assert receipt[key]["utf8_valid"] and not receipt[key]["transcoded"]
    assert receipt["child_exit_code"] == 7
    assert receipt["launch_error_class"] is None
    assert receipt["attempts"] == 1


def test_invalid_utf8_is_retained_and_reported_without_replacement(tmp_path):
    receipt = capture_process([sys.executable, "-c",
        "import os; os.write(1,b'first\\xfflast'); os.write(2,b'\\xfe')"], tmp_path / "bad")
    assert Path(receipt["stdout"]["path"]).read_bytes() == b"first\xfflast"
    assert Path(receipt["stderr"]["path"]).read_bytes() == b"\xfe"
    assert not receipt["capture_utf8_valid"]
    assert receipt["stdout"]["invalid_byte_offset"] == 5
    assert receipt["child_exit_code"] == 0


def test_existing_capture_is_never_reused_or_child_rerun(tmp_path):
    output = tmp_path / "kept"
    output.mkdir()
    marker = output / "original"
    marker.write_bytes(b"old")
    child_marker = tmp_path / "child-ran"
    with pytest.raises(FileExistsError):
        capture_process([sys.executable, "-c",
            f"from pathlib import Path; Path({str(child_marker)!r}).touch()"], output)
    assert marker.read_bytes() == b"old" and not child_marker.exists()


def test_launch_failure_is_recorded_without_exception_text(tmp_path):
    receipt = capture_process([str(tmp_path / "missing-executable")], tmp_path / "failure")
    assert receipt["child_exit_code"] is None
    assert receipt["launch_error_class"] in ("FileNotFoundError", "PermissionError")
    assert Path(receipt["stdout"]["path"]).read_bytes() == b""
    assert "missing-executable" not in json.dumps(receipt)


def test_cli_keeps_child_exit_and_captures_large_separate_streams(tmp_path):
    code = "import os; os.write(1, '邀请'.encode('utf-8')*100000); os.write(2,b'e'*800000); raise SystemExit(9)"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/capture_utf8_process.py"),
        "--output", str(tmp_path / "cli"), "--", sys.executable, "-c", code],
        capture_output=True, timeout=20)
    assert result.returncode == 9
    receipt = json.loads((tmp_path / "cli/process-receipt.json").read_text(encoding="utf-8"))
    assert Path(receipt["stdout"]["path"]).read_bytes() == "邀请".encode("utf-8") * 100000
    assert Path(receipt["stderr"]["path"]).read_bytes() == b"e" * 800000
    assert json.loads(result.stdout)["child_exit_code"] == 9


def test_cli_invalid_encoding_is_failure_while_child_status_remains_zero(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/capture_utf8_process.py"),
        "--output", str(tmp_path / "cli-bad"), "--", sys.executable,
        "-c", "import os; os.write(1,b'\\xff')"], capture_output=True, timeout=20)
    assert result.returncode == 2
    receipt = json.loads((tmp_path / "cli-bad/process-receipt.json").read_text(encoding="utf-8"))
    assert receipt["child_exit_code"] == 0 and not receipt["capture_utf8_valid"]


def test_smoke_fixture_subprocess_emits_actual_invitation_before_confirmation(tmp_path):
    # Scripted software labels exercise the real local store and capture path.
    # They are not model evaluation or a reclassification of any old live run.
    from test_smoke_demo_entry import story
    story_path = tmp_path / "software-story.json"
    story_path.write_text(json.dumps(story(), ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "smoke"
    code = (
        "import runpy, socket, sys; "
        "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden')); "
        "sys.path.insert(0,'scripts'); "
        "sys.argv=['scripts/smoke_demo.py','--fixture','--story'," + repr(str(story_path)) +
        ",'--output'," + repr(str(output)) + "]; runpy.run_path('scripts/smoke_demo.py',run_name='__main__')"
    )
    receipt = capture_process([sys.executable, "-c", code], tmp_path / "smoke-capture", cwd=ROOT)
    assert receipt["child_exit_code"] == 0 and receipt["capture_utf8_valid"]
    raw = Path(receipt["stdout"]["path"]).read_bytes()
    invitations = list(output.glob("invitation-*.json"))
    assert len(invitations) == 1
    invitation = json.loads(invitations[0].read_text(encoding="utf-8"))
    canonical = invitation["question"].replace("\n", os.linesep).encode("utf-8")
    assert canonical in raw
    assert raw.index(canonical) < raw.index("[S2/T1]".encode())
    assert "\ufffd".encode() not in raw
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_class"] == "software_fixture"
    assert summary["operations_completed"] and summary["turns"] == 33
    assert summary["preferences"][0]["status"] == "withdrawn"
    assert not summary["scored"] and not summary["human_evaluation"]
