"""Actual client/guard/product software checks. No network or credential files."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import socket
import sys
import urllib.error

import pytest

from aria.bench.e1b_runner import E1bLocalFixtureClient
from aria.bench.e1c_source import seal_first_submission
from aria.engine.policy import minimal_calls, e1br_calls
from scripts import e1b_evaluation as common
from tests._reservation_fixtures import fixture_reservation_policy
from tests.e1c_software_fixture import fixture_author, fixture_payload

from scripts import e1c_execution as entry
from scripts import e1c_qualification as qualification


@pytest.fixture(autouse=True)
def no_network_or_env(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("software candidate cannot use network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    original = Path.open
    def checked(path, *args, **kwargs):
        if path.name.casefold() == ".env":
            raise AssertionError("software candidate cannot read .env")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(common.canonical(value))
    return entry.sha(path)


def previous(tmp_path):
    root = tmp_path / "SOFTWARE_PRIOR"
    root.mkdir()
    policy = fixture_reservation_policy()
    # Lower disclosed synthetic prices make a short software budget practical.
    for model in policy["models"].values():
        model.update(input_usd_per_million="0.01", output_usd_per_million="0.05")
    frozen = {"metadata": {"policy": policy}, "protocol": {"budget": {
        "prior_known_floor_usd": "2", "prior_e1b_known_usd": "1",
        "historical_unknown_reserved_usd": "0.09", "accepted_old_unknown": 9}}}
    frozen["seal_sha256"] = common.digest(frozen)
    save(root / "registration.json", frozen)
    with closing(e1br_calls.E1BRCallGuard(root / "ledger.sqlite3", max_requests=3153,
            max_observed_cost_usd="1", reservation_policy=policy,
            stage="main", registration_sha256=frozen["seal_sha256"], authority_check=lambda: None)) as guard:
        state = guard.snapshot()
    receipt = {"status": "completed", "software_fixture_only": True, "registration_sha256": frozen["seal_sha256"], "accounting": state}
    receipt["receipt_sha256"] = common.digest(receipt)
    save(root / "receipt.json", receipt)
    unknowns = [{"campaign": "SOFTWARE_PRIOR_ANCESTOR", "request_id": f"unknown-{n}",
                 "request_sha256": f"{n:064x}", "amount": "0.01"} for n in range(9)]
    save(root / "unknowns.json", unknowns)
    history = {key: {"path": str(root / filename), "sha256": entry.sha(root / filename)} for key, filename in
        (("registration", "registration.json"), ("receipt", "receipt.json"), ("ledger", "ledger.sqlite3"), ("unknown_manifest", "unknowns.json"))}
    history["financial_snapshot"] = {"schema_version": "e1c_finance_v1", "prior_e1b_closed": True,
        "prior_assessment_sha256": history["receipt"]["sha256"], "prior_ledger_sha256": history["ledger"]["sha256"],
        "known_cumulative_usd": "2", "known_e1b_usd": "1", "unknown_holds": unknowns}
    return history, policy


def setup(tmp_path, *, approve=True, e1c_limit="0.5"):
    source = tmp_path / "first"
    seal_first_submission(json.dumps(fixture_payload()).encode(), fixture_author(), source)
    history, policy = previous(tmp_path)
    campaign = tmp_path / "campaign"
    frozen = entry.prepare(campaign, source_directory=source, first_receipt_sha256=entry.sha(source / "receipt.json"),
        history=history, reservation_policy=policy, seed=20261005, e1c_limit_usd=e1c_limit,
        cumulative_limit_usd="3", mode="software_fixture")
    human = tmp_path / "SOFTWARE_AUTHORITY_ONLY.json"
    save(human, {"software_fixture": True, "not_real_authority": True})
    auth = entry.authorize_template(frozen)
    if approve:
        auth.update(status="approved", human_approval_file=str(human), human_approval_file_sha256=entry.sha(human))
    save(campaign / "authorization.json", auth)
    return campaign, frozen, common.digest(auth)


class FastGuard(entry.guard_code.E1cCallGuard):
    software_fixture_only = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        clock = [0.0]
        self._pacing_clock = lambda: clock[0]
        self._pacing_sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)


def fake_http(monkeypatch, before=None):
    requests = []
    class Response:
        status, headers = 200, {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return json.dumps(self.value).encode()[:size]
    def opened(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(deepcopy(payload))
        messages = payload["messages"]
        if "response_format" in payload:
            text = json.dumps({"response": "【软件替身】我先听您说。", "state_signals": [],
                "current_affect": {"label": "unknown", "confidence": 0.3,
                    "divergence": {"possible": False, "confidence": 0.2, "pattern": ""}},
                "selected_intervention_id": "", "uncertainty": 0.6,
                "communication_plan": {"pace": "open_listening", "support_order": "先听本人说", "boundary": "不强加话题"}})
        elif "anthropic" in payload["model"]:
            role = "reflection" if "protocol_version" in messages[0]["content"] else "memory"
            text = E1bLocalFixtureClient(role).complete(messages)
        else:
            text = E1bLocalFixtureClient("online").complete(messages)
        value = {"id": f"software-{len(requests)}", "model": payload["model"],
            "provider": "Anthropic" if "anthropic" in payload["model"] else "Alibaba",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.0000005,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}
        if before is not None:
            before(len(requests), value)
        return Response(value)
    opened.software_fixture_only = True
    monkeypatch.setattr(minimal_calls, "open_observed", opened)
    return requests


def test_dedicated_cap_and_fixed_inventory_without_old_guard_mutation(tmp_path):
    campaign, frozen, auth = setup(tmp_path)
    _, source = entry.verify_authority(campaign, auth)
    assert e1br_calls.MAX_REQUESTS == 3153 and e1br_calls.SCHEMA == "e1br_no_content_retry_v1"
    assert entry.guard_code.CALLS.MAX_REQUESTS == 3192
    assert len(entry.guard_code.positions(source.suite)) == 1056
    assert frozen["finance"]["inherited_unknown_count"] == 9
    assert frozen["finance"]["inherited_unknown_hold_usd"] == "0.09"
    assert frozen["new_allowance_usd"] == "0.5"


def test_unapproved_entry_refuses_before_credentials_or_ledger(tmp_path):
    campaign, _, auth = setup(tmp_path, approve=False)
    calls = []
    with pytest.raises(ValueError, match="authorization_changed"):
        entry.execute(campaign, authorization_sha256=auth, credential_loader=lambda: calls.append(True))
    assert not calls and not (campaign / "allowance.sqlite3").exists()


@pytest.mark.parametrize("kind", ["source", "history", "approval", "archive"])
def test_changed_authority_refuses_before_credential_callback(tmp_path, kind):
    campaign, frozen, auth = setup(tmp_path)
    target = (Path(frozen["source_archive"]) / "aria/__init__.py" if kind == "archive" else
              Path(frozen["source_directory"]) / "author-original.json" if kind == "source" else
              Path(frozen["history"]["unknown_manifest"]["path"]) if kind == "history" else
              Path(common.read_json(campaign / "authorization.json")["human_approval_file"]))
    target.write_bytes(target.read_bytes() + b" ")
    calls = []
    with pytest.raises(ValueError):
        entry.execute(campaign, authorization_sha256=auth, credential_loader=lambda: calls.append(True))
    assert not calls and not (campaign / "allowance.sqlite3").exists()


def test_complete_one_case_88_actual_client_product_calls_and_closed_wire(tmp_path, monkeypatch):
    campaign, frozen, auth = setup(tmp_path)
    requests = fake_http(monkeypatch)
    def forbidden_key():
        raise AssertionError("Software mode must never invoke a credential reader")
    receipt = entry.execute(campaign, authorization_sha256=auth, credential_loader=forbidden_key,
        software_first_case_only=True, guard_factory=FastGuard)
    assert receipt["status"] == "software_segment_complete", receipt.get("error_type")
    assert len(requests) == len(receipt["accounting"]["attempts"]) == 88
    assert receipt["accounting"]["unknown_cost_requests"] == 0
    rows = receipt["accounting"]["attempts"]
    assert all(row["status"] == "received" for row in rows)
    assert all(row["registration_sha256"] == frozen["seal_sha256"] for row in rows)
    assert Decimal(receipt["accounting"]["known_cost_usd"]) == Decimal("0.000044")
    entry.guard_code.closed_ledger(campaign, frozen, receipt["accounting"])
    _, source = entry.verify_authority(campaign, auth)
    directory = Path(receipt["run_root"]) / "main/e1c_a1"
    report, _ = qualification.check_case_source(directory, source, source.suite.items[0])
    assert len(report["branches"]) == 6
    for number in (1, 2):
        product = common.read_json(directory / f"reflection-s{number}.product.json")
        assert product["product_status"] == "ok" and product["format_correction_count"] == 0
        first = common.read_json(directory / f"reflection-s{number}.first.json")
        bound = next(r for r in rows if r["request_id"] == first["request_id"])
        assert first["request_sha256"] == bound["request_sha256"]
        assert first["visible_response_sha256"] == bound["client_return_content_sha256"]
    # One case cannot be exported as the fixed 12-case experiment.
    with pytest.raises(ValueError, match="complete_receipt_required"):
        qualification.qualify(campaign, authorization_sha256=auth)
    firstwire = campaign / rows[0]["wire_request_path"]
    firstwire.write_bytes(firstwire.read_bytes() + b" ")
    with pytest.raises(ValueError, match="wire_original_changed"):
        entry.guard_code.closed_ledger(campaign, frozen, receipt["accounting"])


def test_budget_reservation_stops_before_first_wire(tmp_path, monkeypatch):
    campaign, _, auth = setup(tmp_path, e1c_limit="0.00000001")
    requests = fake_http(monkeypatch)
    receipt = entry.execute(campaign, authorization_sha256=auth, credential_loader=lambda: "SOFTWARE_PLACEHOLDER",
        software_first_case_only=True, guard_factory=FastGuard)
    assert receipt["status"] == "not_evaluable" and not requests
    assert receipt["accounting"]["calls_used"] == 0
    assert not list((campaign / "wire").glob("*.json"))


def test_approved_source_revocation_between_calls_stops_before_second_wire(tmp_path, monkeypatch):
    campaign, _, auth = setup(tmp_path)
    def revoke(count, value):
        if count == 1:
            source = campaign / "authorization.json"
            source.write_bytes(source.read_bytes() + b" ")
            # Whitespace changes canonical auth equally: alter the explicit value.
            data = common.read_json(source); data["status"] = "revoked"
            save(source, data)
    requests = fake_http(monkeypatch, revoke)
    receipt = entry.execute(campaign, authorization_sha256=auth, credential_loader=lambda: "SOFTWARE_PLACEHOLDER",
        software_first_case_only=True, guard_factory=FastGuard)
    assert receipt["status"] == "not_evaluable" and len(requests) == 1


@pytest.mark.parametrize("failures", [1, 3])
def test_private_guard_retains_unknowns_and_shared_bounded_retry(tmp_path, monkeypatch, failures):
    from tests.test_e1br_calls import stubs
    from tests.test_reserved_calls import receipt as provider_receipt
    campaign, frozen, auth = setup(tmp_path)
    _, source = entry.verify_authority(campaign, auth)
    with closing(FastGuard(campaign / "allowance.sqlite3", campaign=campaign, suite=source.suite,
            authority_check=lambda: entry.verify_authority(campaign, auth), max_requests=3192,
            max_observed_cost_usd=frozen["new_allowance_usd"], reservation_policy=frozen["reservation_policy"],
            stage="main", registration_sha256=frozen["seal_sha256"])) as guard:
        client = entry.original.make_client("memory", "SOFTWARE_PLACEHOLDER")
        guard.attach(client)
        success = provider_receipt(client)
        success["usage"]["cost"] = 0.0000005
        sequence = [urllib.error.HTTPError("https://software.invalid", 429, "software", {},
                    io.BytesIO(b'{"error":{"code":429}}')) for _ in range(failures)] + [success]
        calls = stubs(monkeypatch, sequence)
        def dispatch():
            with guard.experiment_position("reflect", {"case_id": "e1c_a1", "session": 1}):
                with guard.scope("review", "e1b-review-software-s1", "software-s1"):
                    return client.complete([{"role": "user", "content": "software memory"}])
        if failures == 1:
            assert dispatch() == "local fixture"
        else:
            with pytest.raises(minimal_calls.MinimalCallStopped):
                dispatch()
        state = guard.snapshot()
        assert state["unknown_cost_requests"] == failures
        assert len(calls) == (2 if failures == 1 else 3)
        assert len(entry.guard_code.CALLS.logical_attempts(state)) == 1
        assert Decimal(state["reserved_cost_usd"]) == sum(
            Decimal(row["reservation"]["reserved_cost_usd"]) for row in state["attempts"] if "cost" not in row["usage"])
        with pytest.raises(minimal_calls.MinimalCallStopped):
            dispatch()
        assert len(calls) == (2 if failures == 1 else 3)
    if failures == 1:
        entry.guard_code.closed_ledger(campaign, frozen, state)
        bad = deepcopy(state); bad["unknown_cost_requests"] = 0
        with pytest.raises(ValueError, match="cost_totals_changed"):
            entry.guard_code.closed_ledger(campaign, frozen, bad)


def test_online_bad_first_body_is_preserved_and_strictly_stops_without_retry(tmp_path, monkeypatch):
    from tests.test_online_json import MESSAGES
    campaign, frozen, auth = setup(tmp_path)
    _, source = entry.verify_authority(campaign, auth)
    def corrupt_first(count, value):
        body = json.loads(value["choices"][0]["message"]["content"])
        body["communication_plan"]["boundary"] = ""
        value["choices"][0]["message"]["content"] = json.dumps(body)
    requests = fake_http(monkeypatch, corrupt_first)
    with closing(FastGuard(campaign / "allowance.sqlite3", campaign=campaign, suite=source.suite,
            authority_check=lambda: entry.verify_authority(campaign, auth), max_requests=3192,
            max_observed_cost_usd=frozen["new_allowance_usd"], reservation_policy=frozen["reservation_policy"],
            stage="main", registration_sha256=frozen["seal_sha256"])) as guard:
        client = entry.original.make_client("online_fast", "SOFTWARE_PLACEHOLDER")
        guard.attach(client)
        with pytest.raises(minimal_calls.MinimalCallStopped):
            with guard.experiment_position("submit", {"case_id": "e1c_a1", "phase": "s1", "arm": "shared", "repeat": 0, "turn": 0}):
                with guard.scope("turn", "e1b-software-s1-t0", "software-s1"):
                    client.complete(MESSAGES)
        state = guard.snapshot()
        assert len(requests) == 1
        row = state["attempts"][0]
        assert row["status"] == "failed" and row["validation_error"] == "online_decision_schema_invalid"
        assert row["usage"]["cost"] == 0.0000005 and row["visible_content_sha256"]
        assert (campaign / row["visible_original_path"]).exists()
        assert state["stopped"] is True


def test_software_mode_rejects_unmarked_transport_before_start_or_credential(tmp_path):
    campaign, _, auth = setup(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="software_requires_explicit_local_transport"):
        entry.execute(campaign, authorization_sha256=auth, credential_loader=lambda: calls.append(True))
    assert not calls and not (campaign / "started.json").exists()


def test_cli_does_not_execute_software_source_or_read_key(tmp_path, monkeypatch):
    from scripts import run_e1c, run_ctest
    campaign, _, auth = setup(tmp_path)
    def forbidden(*args):
        raise AssertionError("CLI must not read a key for software material")
    monkeypatch.setattr(run_ctest, "local_key", forbidden)
    args = run_e1c.parser().parse_args(["execute", "--campaign", str(campaign),
        "--authorization-sha256", auth, "--allow-local-env"])
    with pytest.raises(ValueError, match="explicit_real_mode"):
        run_e1c.run(args)
    assert not (campaign / "allowance.sqlite3").exists()


def test_source_freeze_is_explicit_and_excludes_release_tools_and_progress(tmp_path, monkeypatch):
    root = tmp_path / "source-layout"
    for name in entry.SCRIPT_DEPENDENCIES:
        p = root / "scripts" / name
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# software layout", encoding="utf-8")
    (root / "pyproject.toml").write_text("# software config", encoding="utf-8")
    monkeypatch.setattr(entry, "ROOT", root)
    before = entry.source_inventory()
    (root / "scripts/build_minimal_release.py").write_text("# unrelated publishing", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs/PROGRESS.md").write_text("status", encoding="utf-8")
    assert entry.source_inventory() == before
    assert {Path(r["path"]).name for r in before} == set(entry.SCRIPT_DEPENDENCIES) | {"pyproject.toml"}


def test_public_metadata_link_is_rejected_before_shared_reader(tmp_path, monkeypatch):
    from scripts.run_ctest import METADATA, OBSERVATIONS
    for name in (*METADATA, OBSERVATIONS):
        (tmp_path / name).write_text("SOFTWARE metadata", encoding="utf-8")
    # Portable metadata injection; this does not claim OS symlink creation.
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == METADATA[0] or original(p))
    reads = []
    monkeypatch.setattr(entry.original, "base_metadata_evidence", lambda p: reads.append(p))
    with pytest.raises(ValueError, match="evidence_path_forbidden"):
        entry.metadata_evidence(tmp_path)
    assert not reads
