"""Formal entry tests: injected local HTTP only, never model effect evidence."""
from tests._model_http_stub import stub_model_http
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
import urllib.request

import pytest

from scripts import run_e1b as run
from aria.bench.e1b_runner import E1bLocalFixtureClient
from _reservation_fixtures import fixture_reservation_policy


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    stub_model_http(monkeypatch, lambda *a, **k: pytest.fail("unexpected network"))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    # Avoid Windows MAX_PATH in deeply nested pytest-only fictional profiles.
    monkeypatch.setattr(run, "CAMPAIGN", run.Path("c"))
    policy = fixture_reservation_policy()
    for spec in policy["models"].values():
        spec.update(input_usd_per_million="0.01", output_usd_per_million="0.02")
    monkeypatch.setattr(run, "inventory", lambda _: [{"path": "fixture-source", "sha256": "a" * 64}])
    monkeypatch.setattr(run, "metadata_evidence", lambda _: {"policy": policy, "software_fixture": True})
    registration = root / "registration.json"
    value = run.prepare(root, root, registration)
    return root, registration, value


def approve(root, frozen):
    auth = run.authorization_template(root)
    auth.update(status="approved", budget_choice="12_usd1", user_limit_usd="1",
                human_authorization_reference="synthetic unit-test authorization, no real user approval")
    base = run.campaign(root)
    if not (base / "authorization.json").exists():
        run.write_once(base / "authorization.json", auth)
    stage = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": frozen["stage"],
             "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": run.digest(auth)}
    run.write_once(base / f"{frozen['stage']}.authorization.json", stage)
    return {"allow_local_env": True, "budget_choice": "12_usd1",
            "campaign_authorization_sha256": run.digest(auth), "stage_authorization_sha256": run.digest(stage)}


def test_pending_inventory_budget_and_no_key_read(prepared, monkeypatch):
    root, registration, expected = prepared
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("offline cannot read key"))
    assert run.verify(root, root, registration) == expected
    assert expected["protocol"]["planned_total_calls"] == 1636
    assert run.COUNTS["prescreen"] == {"online_fast": 540}
    assert sum(run.COUNTS["main"].values()) == 1056
    assert expected["protocol"]["budget"]["current_remaining_usd"] == "0.607620828"
    assert run.authorization_template()["status"] == "pending"
    with pytest.raises(FileExistsError):
        run.prepare(root, root, registration)
    with pytest.raises(run.RunError, match="permission"):
        run.execute(root, root, registration)
    with pytest.raises(run.RunError, match="10_case"):
        run.execute(root, root, registration, allow_local_env=True, budget_choice="10_usd0_5")
    assert not run.campaign(root).exists()


@pytest.mark.parametrize("change", ["source", "metadata", "expired", "authorization_hash", "reset_cost", "raise_stop"])
def test_preflight_rejects_changes_before_key(prepared, monkeypatch, change):
    root, registration, value = prepared
    args = approve(root, value)
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("invalid preparation must not read key"))
    if change == "source":
        monkeypatch.setattr(run, "inventory", lambda _: [])
    elif change == "metadata":
        monkeypatch.setattr(run, "metadata_evidence", lambda _: {})
    elif change == "expired":
        value["frozen_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        value["seal_sha256"] = run.digest({k: v for k, v in value.items() if k != "seal_sha256"})
        registration.write_text(json.dumps(value), encoding="utf-8")
    elif change == "authorization_hash":
        args["stage_authorization_sha256"] = "f" * 64
    else:
        path = run.campaign(root) / "authorization.json"
        auth = run.read_json(path)
        auth["prior_known_usd" if change == "reset_cost" else "cumulative_stop_usd"] = "0" if change == "reset_cost" else "3"
        path.write_text(json.dumps(auth), encoding="utf-8")
        args["campaign_authorization_sha256"] = run.digest(auth)
    with pytest.raises(run.RunError):
        run.execute(root, root, registration, **args)
    assert not (run.campaign(root) / "prescreen.started.json").exists()


def test_explicit_role_configuration_ignores_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MEMORY_MAX_COMPLETION_TOKENS", "9000")
    monkeypatch.setenv("OPENROUTER_REFLECTION_MAX_RETRIES", "6")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://invalid.example")
    for role, limit in run.protocol()["max_output_tokens"].items():
        client = run.make_client(role, "local-test-placeholder")
        assert client.max_completion_tokens == limit and client.max_retries == 0
        assert client.reasoning_effort == "none" and client.fallback_models == []
        assert client.timeout == (240 if role == "reflection" else 120)
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.reflection_return_format == ("json_schema" if role == "reflection" else "native")


@pytest.mark.parametrize("name,value", [("ARIA_ONLINE_DECISION_VERSION", "online_turn_decision_v2"),
                                        ("ARIA_DELIVERY_BLOCK", "profile_v1")])
def test_experimental_product_environment_is_not_silently_frozen(prepared, monkeypatch, name, value):
    root, registration, _ = prepared
    monkeypatch.setenv(name, value)
    with pytest.raises(run.RunError, match="nondefault_product"):
        run.verify(root, root, registration)


def test_same_authorization_cannot_move_to_another_checkout(prepared, tmp_path, monkeypatch):
    root, registration, frozen = prepared
    args = approve(root, frozen)
    other = tmp_path / "other-checkout"
    other.mkdir()
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("copied campaign cannot read key"))
    for path in run.campaign(root).glob("*.json"):
        run.write_once(run.campaign(other) / path.name, run.read_json(path))
    with pytest.raises(run.RunError, match="another_workspace"):
        run.verify(other, other, registration)
    with pytest.raises(run.RunError, match="another_workspace"):
        run.validate_authorization(other, frozen, **{k: v for k, v in args.items() if k != "allow_local_env"})
    # Preparing a fresh seal in another root also cannot reuse the campaign owner.
    moved = run.prepare(other, other, other / "new-registration.json")
    with pytest.raises(run.RunError, match="not_explicitly_authorized"):
        run.validate_authorization(other, moved, **{k: v for k, v in args.items() if k != "allow_local_env"})


def test_product_yaml_resource_change_blocks_before_key(tmp_path, monkeypatch):
    root = tmp_path / "resource-repo"
    (root / "aria/assets").mkdir(parents=True)
    source = root / "aria/product.py"
    source.write_text("# synthetic source\n", encoding="utf-8")
    resource = root / "aria/assets/strategy.yaml"
    resource.write_text("strategy: original\n", encoding="utf-8")
    monkeypatch.setattr(run, "FILES", ())
    monkeypatch.setattr(run, "metadata_evidence", lambda _: {"policy": {}})
    monkeypatch.setattr(run, "local_key", lambda _: pytest.fail("changed resources cannot read key"))
    registration = root / "freeze.json"
    frozen = run.prepare(root, root, registration)
    assert any(r["path"] == "aria/assets/strategy.yaml" for r in frozen["source"])
    resource.write_text("strategy: changed\n", encoding="utf-8")
    with pytest.raises(run.RunError, match="source_or_protocol_changed"):
        run.execute(root, root, registration, allow_local_env=True)


@pytest.mark.parametrize("status", ["pending", "failed", "unknown", "wrong_count", "halted"])
def test_clean_resume_denies_unresolved_or_changed_accounting(status):
    state = {"stopped": False, "unknown_cost_requests": 0, "calls_used": 540,
             "attempts": [{"status": "received", "usage": {"cost": 0.01}}]}
    if status in {"pending", "failed"}:
        state["attempts"][0]["status"] = status
    elif status == "unknown":
        state["unknown_cost_requests"] = 1
    elif status == "wrong_count":
        state["calls_used"] = 539
    else:
        state["stopped"] = True
    with pytest.raises(run.RunError, match="accounting"):
        run.assert_clean(state, expected_calls=540)


def fake_http(monkeypatch, *, failure=None):
    calls = []
    decision = {"response": "这是本地软件替身回复，不能作为模型效果。", "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": .3,
            "divergence": {"possible": False, "confidence": .2, "pattern": ""}},
        "selected_intervention_id": "", "uncertainty": .6,
        "communication_plan": {"pace": "open_listening", "support_order": "先听本人说", "boundary": "不强加话题"}}
    class Response:
        status = 200
        headers = {}
        def __init__(self, raw):
            self.stream = io.BytesIO(raw)
        def read(self, size=-1):
            return self.stream.read(size)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def receive(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        if failure == "timeout":
            raise TimeoutError("synthetic timeout")
        if payload["model"] == run.DEFAULT_MEMORY_MODEL:
            if payload["max_tokens"] == 1024:
                item = json.loads(payload["messages"][-1]["content"])
                content = json.dumps({"id": item["id"], "content_sha256": item["content_sha256"],
                    "labels": {name: 0 for name in run.evaluation.RUBRIC_KEYS}})
            else:
                role = "memory" if payload["max_tokens"] == 1800 else "reflection"
                content = E1bLocalFixtureClient(role).complete(payload["messages"])
        else:
            content = json.dumps(decision, ensure_ascii=False) if payload.get("response_format") else decision["response"]
        # Client trimming must not break memory/judge or delivery qualification.
        content = "\n " + content + " \n"
        usage = {"prompt_tokens": 30, "completion_tokens": 50, "cost": .000001}
        if failure == "unknown":
            usage.pop("cost")
        return Response(json.dumps({"model": payload["model"], "provider": payload["provider"]["only"][0].title(), "usage": usage,
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())
    stub_model_http(monkeypatch, receive)
    return calls, decision


@pytest.mark.parametrize("failure", ["timeout", "unknown", "save"])
def test_failure_unknown_save_and_changed_output_cannot_restart(prepared, monkeypatch, failure):
    root, registration, value = prepared
    args = approve(root, value)
    monkeypatch.setattr(run, "local_key", lambda _: "local-unit-test-inert-key")
    calls, _ = fake_http(monkeypatch, failure=failure)
    if failure == "save":
        original = run.EvidenceGuard._finish
        def fail_finish(self, row):
            if row["role"] == "online_fast":
                raise OSError("synthetic save failure")
            return original(self, row)
        monkeypatch.setattr(run.EvidenceGuard, "_finish", fail_finish)
    result = run.execute(root, root, registration, **args)
    assert result["status"] == "not_evaluable" and len(calls) == 1
    receipt = run.read_json(run.campaign(root) / "prescreen" / "execution-receipt.json")
    assert receipt["accounting"]["stopped"]
    assert receipt["accounting"]["calls_used"] == 1
    with pytest.raises(run.RunError, match="already_started"):
        run.execute(root, root, registration, **args)
    with pytest.raises(run.RunError, match="not_completed"):
        run.prepare(root, root, root / "main.json", stage="main")
    assert len(calls) == 1


def test_full_18_by_3_by_10_prescreen_real_assembly_fake_http(prepared, monkeypatch):
    root, registration, value = prepared
    args = approve(root, value)
    monkeypatch.setattr(run, "local_key", lambda _: "local-unit-test-inert-key")
    calls, decision = fake_http(monkeypatch)
    clients_seen = []
    original_client = run.make_client
    def record_client(role, key):
        client = original_client(role, key)
        clients_seen.append(client)
        return client
    monkeypatch.setattr(run, "make_client", record_client)
    result = run.execute(root, root, registration, **args)
    assert result["status"] == "completed", (result, [c.last_diagnostic for c in clients_seen])
    assert len(calls) == 540
    receipt = run.read_receipt(root, "prescreen")
    assert receipt["accounting"]["known_cost_usd"] == "0.000540"
    positions = receipt["qualification"]["positions"]
    assert len(positions) == 540
    assert all(p["model_reply_eligible"] == (p["delivery"] == "model_unmodified") for p in positions)
    assert all(r["model_reply_sha256"] == run.text_sha(decision["response"]) for r in receipt["accounting"]["attempts"])
    assert all("content" not in row and "response" not in row for row in receipt["accounting"]["attempts"])
    with pytest.raises(run.RunError, match="first_prescreen"):
        run.prepare(root, root, root / "main-unlabeled.json", stage="main")
    base = run.campaign(root) / "prescreen" / "blind-bundle"
    packet, key = run.evaluation.load_bundle(base)
    labels = run.evaluation.label_template(packet)
    labels["scorer"]["id"] = "independent_software_test_labels"
    for row in labels["labels"]:
        row["labels"] = {name: int(spec["applicable"]) for name, spec in
                         next(i["content"]["rubric"] for i in packet["items"] if i["id"] == row["id"]).items()}
    run.evaluation.import_labels(base, labels)
    selected = run.selection_evidence(root)
    assert len(selected["case_ids"]) == 12 and len(selected["repeat_violations"]) == 18
    main_registration = root / "main.json"
    main = run.prepare(root, root, main_registration, stage="main")
    assert main["inputs"] == selected
    main_args = approve(root, main)
    continued = run.execute(root, root, main_registration, **main_args)
    assert continued["status"] == "completed", continued
    assert len(calls) == 1596
    main_receipt = run.read_receipt(root, "main")
    assert main_receipt["accounting"]["ledger_id"] == receipt["accounting"]["ledger_id"]
    assert main_receipt["accounting"]["attempts"][:540] == receipt["accounting"]["attempts"]
    assert len(main_receipt["qualification"]["memory_originals"]) == 24
    judge_registration = root / "judge.json"
    judge = run.prepare(root, root, judge_registration, stage="judge")
    judged = run.execute(root, root, judge_registration, **approve(root, judge))
    assert judged["status"] == "completed", judged
    assert len(calls) == 1636
    final = run.read_receipt(root, "judge")
    assert final["accounting"]["stop_reason"] == "request_limit_reached"
    assert final["accounting"]["ledger_id"] == receipt["accounting"]["ledger_id"]
    assert final["effect_claim_eligible"] is False


def test_qualification_distinguishes_rule_and_changed_deliveries(tmp_path, monkeypatch):
    monkeypatch.setitem(run.COUNTS, "prescreen", {"online_fast": 3})
    ids = [run.load_e1b_cases().items[0].id]
    frozen = {"stage": "prescreen", "inputs": {"case_ids": ids}, "seal_sha256": "a" * 64}
    attempts = [{"role": "online_fast", "operation_id": f"turn{i}", "session_id": "s", "e1b_stage": "prescreen",
                 "registration_sha256": "a" * 64, "visible_content_sha256": "b" * 64,
                 "model_reply_sha256": run.text_sha("model")} for i in range(3)]
    rows = [{"turn_id": f"turn{i}", "session_id": "s", "response": response,
             "source": source, "status": "completed"} for i, (response, source) in enumerate(
                 [("model", "live"), ("changed", "live"), ("rule", "rule_safety")])]
    report = {"evidence_kind": "real_model", "phase": "prescreen",
              "branches": [{"arm": "A0", "repeat": 1, "state": {"turns": rows}}]}
    run.write_once(tmp_path / "prescreen" / ids[0] / "report.json", report)
    value = run.qualify(tmp_path, frozen, {"attempts": attempts, "ledger_id": "fixture"})
    assert [r["delivery"] for r in value["positions"]] == ["model_unmodified", "guard_modified", "rule_only"]
    assert [r["model_reply_eligible"] for r in value["positions"]] == [True, False, False]
    missing = deepcopy(attempts)
    missing[0]["operation_id"] = "other"
    with pytest.raises(run.RunError, match="receipt"):
        run.qualify(tmp_path, frozen, {"attempts": missing, "ledger_id": "fixture"})
