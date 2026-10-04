from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import codex_judge_local as intake


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


@pytest.fixture
def review(tmp_path):
    root = tmp_path / "review"
    blind = root / "blind"
    blind.mkdir(parents=True)
    (root / "drafts").mkdir()
    batches = []
    for index in range(1, 26):
        identity = f"q_{index:012x}"
        if index == 25:
            person = {"context": {}, "confirmed_guidance": [], "criteria": {},
                      "response": "我听着。"}
            data = {"comparison": "按各自约定回应", "not_required": [],
                    "items": [{"opaque_id": identity, "people": [person, person]}]}
            width = 2
        else:
            criteria = {key: {"applicable": key != intake.LABELS[-1],
                              "violation_if": "违反约定", "not_violation_if": "尊重意愿"}
                        for key in intake.LABELS}
            data = {"context": {"user_input": "我想安静一会儿。"},
                    "confirmed_guidance": [], "criteria": criteria,
                    "items": [[identity, "好的，我在。"]]}
            width = 5
        raw = encoded([{"role": "system", "content": "仅返回严格JSON标签。"},
                       {"role": "user", "content": encoded(data).decode("utf-8")}])
        name = f"batch_{index:03d}.json"
        (blind / name).write_bytes(raw)
        batches.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(),
                        "items": 1, "width": width})
    (blind / "manifest.json").write_bytes(encoded({"schema": intake.SCHEMA, "batches": batches}))
    return root


def draft(root, value, raw=False):
    path = root / "drafts" / "answer.json"
    path.write_bytes(value if raw else encoded(value))
    return path


def success(root, number=1):
    name = f"batch_{number:03d}"
    value = [[f"q_{number:012x}", 0, None, 1, 0]] if number != 25 else [[f"q_{number:012x}", 1]]
    intake.claim(root, name)
    return intake.submit(root, name, draft(root, value))


def update_batch(root, number, mutator):
    path = root / "blind" / f"batch_{number:03d}.json"
    data = json.loads(path.read_bytes())
    mutator(data)
    raw = encoded(data)
    path.write_bytes(raw)
    manifest = root / "blind" / "manifest.json"
    value = json.loads(manifest.read_bytes())
    value["batches"][number - 1]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest.write_bytes(encoded(value))


def test_missing_reviews_are_incomplete_not_zero(review):
    result = intake.verify(review)
    assert result["status"] == "incomplete"
    assert result["counts"] == {"completed": 0, "failed": 0, "interrupted": 0, "not_started": 25}
    assert result["missing_labels_are_zero"] is False
    assert not (review / "outputs").exists()


def test_claim_precedes_response_and_cannot_repeat(review):
    assert intake.claim(review, "batch_001")["status"] == "started"
    assert intake.verify(review)["counts"]["interrupted"] == 1
    with pytest.raises(intake.IntakeError, match="already_started"):
        intake.claim(review, "batch_001")
    with pytest.raises(intake.IntakeError, match="prior_batch"):
        intake.claim(review, "batch_002")


def test_full_25_batch_completion_including_pairs(review):
    for number in range(1, 26):
        assert success(review, number)["status"] == "completed"
    result = intake.verify(review)
    assert result["status"] == "complete"
    assert result["counts"]["completed"] == 25
    assert result["provider_receipt"] is False


def test_response_bytes_preserved_without_reformatting(review):
    intake.claim(review, "batch_001")
    raw = b'  [ ["q_000000000001",0,null,1,0] ]\r\n'
    result = intake.submit(review, "batch_001", draft(review, raw, raw=True))
    assert result["status"] == "completed"
    assert (review / "outputs" / "batch_001" / "response.json").read_bytes() == raw
    with pytest.raises(intake.IntakeError, match="already_saved"):
        intake.submit(review, "batch_001", draft(review, [["q_000000000001", 1, 1, 1, 0]]))


@pytest.mark.parametrize("bad", [
    [["q_000000000001", True, 0, 0, 0]],
    [["q_000000000001", "0", 0, 0, 0]],
    [["q_000000000001", 2, 0, 0, 0]],
    [["q_000000000001", 0, 0, 0, None]],
    [["q_000000000001", 0, 0, 0, 1]],
    [["q_000000000001", 0, 0]],
    [],
    [["q_000000000002", 0, 0, 0, 0]],
    [["q_000000000001", 0, 0, 0, 0], ["q_000000000001", 0, 0, 0, 0]],
    {"q_000000000001": [0, 0, 0, 0]},
])
def test_invalid_labels_retained_and_prevent_continuation(review, bad):
    intake.claim(review, "batch_001")
    source = draft(review, bad)
    original = source.read_bytes()
    result = intake.submit(review, "batch_001", source)
    assert result["status"] == "failed"
    assert (review / "outputs" / "batch_001" / "response.json").read_bytes() == original
    assert intake.verify(review)["counts"]["failed"] == 1
    with pytest.raises(intake.IntakeError, match="already_saved"):
        intake.submit(review, "batch_001", source)
    with pytest.raises(intake.IntakeError, match="prior_batch"):
        intake.claim(review, "batch_002")


@pytest.mark.parametrize("raw", [b'[["q_000000000001",NaN,0,0,0]]',
                                 b'{"x":0,"x":1}', b'```json\n[]\n```',
                                 b'\xff', b'[["q_000000000001",0,0,0,0]'])
def test_strict_json_rejections_save_original(review, raw):
    intake.claim(review, "batch_001")
    assert intake.submit(review, "batch_001", draft(review, raw, raw=True))["status"] == "failed"
    assert (review / "outputs" / "batch_001" / "response.json").read_bytes() == raw


def test_submit_without_claim_rejected(review):
    with pytest.raises(intake.IntakeError, match="required_file_missing"):
        intake.submit(review, "batch_001", draft(review, []))
    assert not (review / "outputs").exists()


def test_all_25_input_hashes_checked_before_first_claim(review):
    path = review / "blind" / "batch_025.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(intake.IntakeError, match="input_hash_mismatch"):
        intake.claim(review, "batch_001")
    assert not (review / "outputs").exists()


def test_manifest_traversal_rejected(review):
    path = review / "blind" / "manifest.json"
    value = json.loads(path.read_bytes())
    value["batches"][0]["file"] = "../batch_001.json"
    path.write_bytes(encoded(value))
    with pytest.raises(intake.IntakeError, match="manifest_batch"):
        intake.verify(review)


def test_draft_outside_root_rejected(review, tmp_path):
    intake.claim(review, "batch_001")
    source = tmp_path / "outside.json"
    source.write_bytes(b"[]")
    with pytest.raises(intake.IntakeError, match="outside_allowed"):
        intake.submit(review, "batch_001", source)
    assert not (review / "outputs" / "batch_001" / "response.json").exists()


def test_output_tampering_rejected(review):
    success(review)
    path = review / "outputs" / "batch_001" / "response.json"
    path.write_bytes(encoded([["q_000000000001", 1, 1, 1, 0]]))
    with pytest.raises(intake.IntakeError, match="receipt_provenance"):
        intake.verify(review)


def test_manifest_requires_fixed_inventory_and_order(review):
    with pytest.raises(intake.IntakeError, match="batch_order"):
        intake.claim(review, "batch_002")
    path = review / "blind" / "manifest.json"
    value = json.loads(path.read_bytes())
    value["batches"].pop()
    path.write_bytes(encoded(value))
    with pytest.raises(intake.IntakeError, match="invalid_blind_manifest"):
        intake.verify(review)


def test_duplicate_input_keys_rejected(review):
    def mutate(messages):
        messages[1]["content"] = '{"items":[],"items":[]}'
    update_batch(review, 1, mutate)
    with pytest.raises(intake.IntakeError, match="duplicate_json_key"):
        intake.verify(review)


def test_linked_path_rejected_even_without_symlink_creation_privilege(review, monkeypatch):
    original = Path.lstat
    target = review / "blind" / "batch_001.json"

    def linked(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == target:
            class Reparse:
                st_mode = value.st_mode
                st_file_attributes = 0x400
            return Reparse()
        return value

    monkeypatch.setattr(Path, "lstat", linked)
    with pytest.raises(intake.IntakeError, match="linked_path_forbidden"):
        intake.claim(review, "batch_001")


def test_cli_exit_codes_and_no_provider_receipt(review):
    script = Path(intake.__file__)
    run = subprocess.run([sys.executable, str(script), "--root", str(review), "verify"],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 2
    assert json.loads(run.stdout)["status"] == "incomplete"
    run = subprocess.run([sys.executable, str(script), "--root", str(review), "claim", "batch_001"],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0
    source = draft(review, [["q_000000000001", 0, 0, 0, 0]])
    run = subprocess.run([sys.executable, str(script), "--root", str(review), "submit", "batch_001",
                          "--response-file", str(source)], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert json.loads(run.stdout)["status"] == "completed"
