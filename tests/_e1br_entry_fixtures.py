"""Portable synthetic donor checkpoints for the public E1b-R entry tests.

The real restore/hash/authentication paths run against temporary artifacts.
No developer campaign, original model answer, authorization, or .env is read.
"""
from copy import deepcopy
import hashlib

from aria.bench.e1b_cases import load_e1b_cases
from aria.bench.e1b_runner import E1bRunner, local_fixture_factory
from aria.engine.memory.diagnosis import build_reflection_messages
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.reflection_json import reflection_body_system_prompt
from scripts import e1b_main_continuation as retained
from scripts import run_e1b as original
from scripts.e1b_evaluation import canonical
from scripts import run_e1br
from scripts import run_e1br_received_continuation as received_entry
from tests.test_e1br_received_body import received


def synthetic_entry_root(tmp_path, monkeypatch):
    root = tmp_path / "isolated-originals"
    root.mkdir()
    suite = load_e1b_cases()
    token = SyntheticEmotionExperiment.create(root / "fixture", "e1b-main-v1")
    runner = E1bRunner(suite, root=token.root / "run",
                      app_factory=local_fixture_factory(token), seed=original.SEED)
    app = runner._new_app(token.root / "source-app")
    session = runner._session(app, suite.items[0], 1, arm="shared", repeat=0)
    operation = f"e1b-review-{session}"
    profile = app._profile()
    job = profile.offline_sessions[0]
    job.memory_completed, job.reflection_completed = True, False
    job.attempts, job.state, job.active_stage = 1, "pending", "reflection"
    app.store.save(profile)
    saved_run = deepcopy(app._run)
    saved_run["reviews"] = [{"request_id": operation, "session_id": session, "status": "failed"}]
    saved_profile = profile.model_dump(mode="json")
    app.close()
    conversation = "\n".join(f"[turn {i}] 用户:{t.user}\nAria:{t.agent}" for i, t in enumerate(job.turns))
    messages = build_reflection_messages(profile, conversation, [], True)
    messages[0]["content"] = reflection_body_system_prompt()
    failed_wire = canonical({"messages": messages, "model": "software/reflection-fixture"})
    failed_wire_sha = hashlib.sha256(failed_wire).hexdigest()
    monkeypatch.setattr(retained, "SESSION", session)
    monkeypatch.setattr(retained, "OPERATION", operation)
    monkeypatch.setattr(retained, "FAILED_REQUEST", "synthetic-failed-request")
    monkeypatch.setattr(retained, "FAILED_WIRE", failed_wire_sha)

    parent = root / retained.PARENT
    case = parent / retained.RUN / "main" / retained.CASE
    case.mkdir(parents=True)
    registration = {"inputs": {"case_ids": [f"e1b_{g}{n}" for g in "abcd" for n in (1, 2, 3)]},
                    "protocol": {"seed": original.SEED}, "evidence_kind": "software_fixture"}
    registration["seal_sha256"] = original.digest(registration)
    original.write_once(parent / "main/registration.json", registration)
    attempts = []
    for index in range(14):
        role = "online_fast" if index < 12 else "memory" if index == 12 else "reflection"
        wire = failed_wire if index == 13 else canonical({"software_fixture": role, "index": index})
        path = f"main/wire-requests/{index + 1:04d}.json"
        target = parent / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(wire)
        row = {"e1b_stage": "main", "role": role, "status": "failed" if index == 13 else "received",
               "request_id": "synthetic-failed-request" if index == 13 else f"synthetic-retained-{index}",
               "session_id": session, "operation_id": f"e1b-{session}-t{index}" if index < 12 else operation,
               "wire_request_path": path, "request_sha256": hashlib.sha256(wire).hexdigest(),
               "registration_sha256": registration["seal_sha256"], "usage": {} if index == 13 else {"cost": .000001}}
        if index < 12:
            row["model_reply_sha256"] = original.text_sha(saved_run["turns"][index]["response"])
        if index == 12:
            row["client_return_content_sha256"] = original.text_sha("synthetic memory with no proposals")
        attempts.append(row)
    receipt = {"status": "not_evaluable", "registration_sha256": registration["seal_sha256"],
               "accounting": {"attempts": attempts}, "evidence_kind": "software_fixture"}
    receipt["receipt_sha256"] = original.digest(receipt)
    monkeypatch.setattr(retained, "RECEIPT_SHA256", receipt["receipt_sha256"])
    original.write_once(parent / "main/execution-receipt.json", receipt)
    values = {
        "shared/live/run.json": saved_run,
        "shared/live/profiles/fictional-chen.json": saved_profile,
        "memory-s1.attempt.json": {"session_id": session, "source_sha256": suite.source_sha256},
        "memory-s1.first.json": {"parse_status": "present_array", "request_hash_kind": "actual_wire",
                                  "request_sha256": attempts[12]["request_sha256"],
                                  "visible_response_sha256": attempts[12]["client_return_content_sha256"],
                                  "raw_stated_preferences": [], "evidence_kind": "software_fixture"},
        "memory-s1.product.json": {"product_status": "ok"},
        "raw-proposals.json": {"evidence_kind": "software_fixture", "sessions": []},
        "reflection-s1.first.json": {"request_id": "synthetic-failed-request", "parse_status": "not_received"},
        "reflection-s1.attempt.json": {"session_id": session},
        "reflection-s1.product.json": {"product_status": "failed"},
    }
    for name, value in values.items():
        original.write_once(case / name, value)
    care = case / "shared/live/care.sqlite3"
    # The application created this empty schema; copy only the synthetic store.
    care.write_bytes((app.run_dir / "care.sqlite3").read_bytes())
    pins = {name: hashlib.sha256((case / name).read_bytes()).hexdigest()
            for name in retained.PINNED_FILES}
    monkeypatch.setattr(retained, "PINNED_FILES", pins)
    prefix = retained.load_prefix(root)  # Run the original byte/shape/turn checks.
    source = received(messages)
    source["receipt"]["request_id"] = "synthetic-received-reflection"
    source["receipt"]["request_sha256"] = failed_wire_sha
    monkeypatch.setattr(received_entry, "RECEIVED_REQUEST", source["receipt"]["request_id"])
    monkeypatch.setattr(received_entry, "load_received", lambda current: deepcopy(source))
    monkeypatch.setattr(received_entry, "parent_originals", lambda current: list(prefix["paths"]))

    def configure(module):
        monkeypatch.setattr(module, "approval", lambda current: {
            "approval_file_sha256": "a" * 64, "human_quote": "Synthetic test authorization; no external calls",
            "human_message_reference": "software_fixture"})
        monkeypatch.setattr(module, "parent_originals", lambda current: list(prefix["paths"]))
        monkeypatch.setattr(module, "selection_evidence", lambda current: {
            "case_ids": list(module.SELECTED), "selection": {"software_fixture": True},
            "retained_prescreen_positions": 540, "retained_first_labels": 162,
            "original_aggregate_sha256": "b" * 64, "first_label_files": []})
    configure(run_e1br)
    load_module = run_e1br.common._load_module
    def load(name, filename):
        module = load_module(name, filename)
        if filename == "run_e1br.py":
            configure(module)
        return module
    monkeypatch.setattr(run_e1br.common, "_load_module", load)
    return root
