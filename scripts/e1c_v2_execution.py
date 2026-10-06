"""Separately frozen E1c-v2 authority: original ten unseen cases, actual product outcomes.

No credential discovery; root supplies a callback only after all checks.
"""
from copy import deepcopy
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3

from aria.bench.e1c_source import load_first_submission
from aria.bench.e1c_v2_runner import E1cV2Runner, E1cV2ExperimentApp, heldout_submission
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.reserved_calls import validate_reservation_policy
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_SAFETY
from aria.runtime import MinimalModelClients
from scripts import e1b_evaluation as common
from scripts import run_e1b as original
from scripts import e1c_execution as previous
from scripts import e1c_v2_calls as guard_code

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DEPENDENCIES = (*previous.SCRIPT_DEPENDENCIES, "run_e1c_v2.py", "e1c_v2_execution.py",
    "e1c_v2_calls.py", "e1c_v2_qualification.py", "e1c_v2_scoring.py")
sha, regular, metadata_evidence = previous.sha, previous.regular, previous.metadata_evidence
def source_inventory():
    files = [*sorted((ROOT / "aria").rglob("*.py")),
             *sorted((ROOT / "aria").rglob("*.yaml")), *sorted((ROOT / "aria").rglob("*.yml")),
             *(ROOT / "scripts" / name for name in SCRIPT_DEPENDENCIES), ROOT / "pyproject.toml"]
    return [{"path": str(p), "sha256": sha(regular(p))} for p in files]
def verify_source_archive(frozen):
    archive = regular(frozen["source_archive"])
    expected = {}
    for row in frozen["source"]:
        relative = Path(row["path"]).relative_to(ROOT)
        expected[relative.as_posix()] = row["sha256"]
        if sha(regular(archive / relative)) != row["sha256"]:
            raise ValueError("e1c_frozen_source_archive_changed")
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("e1c_frozen_source_archive_members_changed")

def protocol(source):
    return {"schema": "e1cv2_scoring_protocol_v1", "source_sha256": source.suite.source_sha256,
        "case_ids": [c.id for c in heldout_submission(source).suite.items],
        "excluded_development_cases": ["e1c_a1", "e1c_a2"],
        "case_denominator": 10, "reply_denominator": 180, "formation_denominator": 10,
        "main_gate": {"minimum_improved_cases": 4, "maximum_worse_cases": 0},
        "formation_gate": {"minimum_proposed_expected_cases": 8, "authorization_or_personality_false_positives": 0},
        "format_gate": {"maximum_scored_fallback_fraction": "0.05", "denominator": 180},
        "exposure_gate": {"minimum_fraction": "0.8", "denominator": 300,
            "numerator": "A1_S3_unique_planned_positions_whose_actual_wire_contains_confirmed_preferences",
            "missing": "zero_numerator_fixed_300_denominator_main_not_evaluable_below_80_percent"},
        "scoring": "actual_delivered_reply_including_fallback_first_independent_codex_labels",
        "formation_failure": "failed_or_missing_memory_remains_in_ten_case_denominator",
        "confirmation": "all_proposed_explicitly_confirmed_by_script_no_expected_answer_selection",
        "pause": "three_logical_failures_stop_no_automatic_owner_resume",
        "paid_judge_calls": 0, "author_cannot_score": source.author.author_id,
        "unknown_fee": "stop_before_next_request_retain_full_reservation",
        "scope": "fictional_scripts_synthetic_emotion_single_judge_no_real_elder_claim"}


def verify_history(history):
    """Bind closed E1c-v1 plus its independently retained E1b ancestry."""
    for kind in ("registration", "receipt", "ledger", "unknown_manifest"):
        if sha(regular(history[kind]["path"])) != history[kind]["sha256"]:
            raise ValueError("e1cv2_history_original_changed")
    frozen = common.read_json(history["registration"]["path"])
    receipt = common.read_json(history["receipt"]["path"])
    if (frozen.get("schema") != "e1c_registered_execution_v1"
            or frozen["seal_sha256"] != common.digest({k:v for k,v in frozen.items() if k != "seal_sha256"})
            or receipt["registration_sha256"] != frozen["seal_sha256"]
            or receipt["receipt_sha256"] != common.digest({k:v for k,v in receipt.items() if k != "receipt_sha256"})
            or receipt["status"] not in {"generated", "not_evaluable", "software_segment_complete"}
            or receipt["inherited_finance"] != frozen["finance"]):
        raise ValueError("e1cv2_history_receipt_binding_changed")
    if previous.verify_history(frozen["history"]) != frozen["finance"]:
        raise ValueError("e1cv2_ancestral_finance_changed")
    ledger = regular(history["ledger"]["path"])
    if any(Path(str(ledger)+s).exists() for s in ("-wal", "-shm", "-journal")):
        raise ValueError("e1cv2_history_ledger_not_closed")
    rows = previous.guard_code.CALLS.validate_e1br_ledger(ledger, stage="main",
        registration_sha256=frozen["seal_sha256"], max_requests=frozen["physical_max"],
        reservation_policy=frozen["reservation_policy"])
    state = receipt["accounting"]
    if rows != state["attempts"] or any(r["status"] == "pending" for r in rows):
        raise ValueError("e1cv2_history_rows_changed")
    known = sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r["usage"]), Decimal(0))
    held = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r["usage"]), Decimal(0))
    if (known != Decimal(state["known_cost_usd"] or "0") or held != Decimal(state["reserved_cost_usd"])
            or state["calls_used"] != len(rows)
            or state["unknown_cost_requests"] != sum("cost" not in r["usage"] for r in rows)):
        raise ValueError("e1cv2_history_cost_totals_changed")
    inherited = common.read_json(frozen["history"]["unknown_manifest"]["path"])
    expected = [*inherited, *[{"campaign": str(ledger.parent.resolve()), "request_id":r["request_id"],
        "request_sha256":r["request_sha256"], "amount":r["reservation"]["reserved_cost_usd"]}
        for r in rows if "cost" not in r["usage"]]]
    actual = common.read_json(history["unknown_manifest"]["path"])
    if sorted(actual, key=common.canonical) != sorted(expected, key=common.canonical):
        raise ValueError("e1cv2_unknown_identity_or_reservation_changed")
    ancestor = frozen["finance"]
    cumulative_known = Decimal(ancestor["known_cumulative_usd"]) + known
    cumulative_held = sum((Decimal(r["amount"]) for r in actual), Decimal(0))
    return {"inherited_known_cumulative_usd": str(cumulative_known),
        "inherited_unknown_hold_usd": str(cumulative_held), "inherited_unknown_count": len(actual),
        "committed_cumulative_usd": str(cumulative_known+cumulative_held),
        "inherited_e1c_known_usd": str(known), "inherited_e1c_hold_usd": str(held),
        "committed_e1c_usd": str(known+held), "prior_ledger_sha256": history["ledger"]["sha256"],
        "unknown_manifest_sha256": history["unknown_manifest"]["sha256"]}
def prepare(campaign, *, source_directory, first_receipt_sha256, history, reservation_policy,
            seed, e1c_limit_usd, cumulative_limit_usd, mode="real_model", metadata_directory=None):
    if mode not in {"real_model", "software_fixture"} or type(seed) is not int:
        raise ValueError("e1c_explicit_mode_and_seed_required")
    source = load_first_submission(source_directory, expected_receipt_sha256=first_receipt_sha256,
                                   allow_software_fixture=mode == "software_fixture")
    if (source.suite.provenance == "software_fixture_only") != (mode == "software_fixture"):
        raise ValueError("e1c_source_mode_changed")
    finance = verify_history(history)
    heldout = heldout_submission(source)
    policy = validate_reservation_policy(reservation_policy)
    metadata = None
    if mode == "real_model":
        if metadata_directory is None:
            raise ValueError("e1c_public_metadata_required")
        metadata = metadata_evidence(Path(metadata_directory))
        if metadata["policy"] != policy:
            raise ValueError("e1c_price_policy_not_bound_to_public_metadata")
        metadata_directory = str(Path(metadata_directory).resolve())
    elif metadata_directory is not None:
        raise ValueError("e1c_software_metadata_must_be_disclosed_fixture")
    from aria.bench.e1c_cases import Money
    from pydantic import TypeAdapter
    limits = [Decimal(TypeAdapter(Money).validate_python(v)) for v in (e1c_limit_usd, cumulative_limit_usd)]
    allowance = min(limits[0] - Decimal(finance["committed_e1c_usd"]), limits[1] - Decimal(finance["committed_cumulative_usd"]))
    if allowance <= 0:
        raise ValueError("e1c_no_inclusive_budget_remaining")
    evaluation_protocol = protocol(source)
    product = original.protocol()
    frozen = {"schema": "e1cv2_registered_execution_v1", "campaign": str(Path(campaign).resolve()),
        "source_archive": str(Path(campaign).resolve() / "frozen-source"),
        "source_directory": str(source.directory), "first_receipt_sha256": first_receipt_sha256,
        "source_sha256": source.suite.source_sha256, "case_ids": [c.id for c in heldout.suite.items],
        "source": source_inventory(), "history": deepcopy(history), "finance": finance,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "metadata_directory": metadata_directory, "metadata": metadata,
        "reservation_policy": policy, "seed": seed, "mode": mode,
        "client_configuration": {name: product[name] for name in ("models", "max_output_tokens", "transport")},
        "e1c_limit_usd": str(limits[0]), "cumulative_limit_usd": str(limits[1]),
        "new_allowance_usd": str(allowance), "physical_max": guard_code.PHYSICAL_MAX,
        "basic_counts": {"online_fast": 840, "memory": 20, "reflection": 20},
        "reply_positions": 180, "formation_cases": 10, "retained_experimental_answers": 0,
        "infrastructure_retries_per_position": 2, "reflection_corrections_per_position": 1,
        "unknown_rule": "retain_each_full_reservation_including_inherited_manifest",
        "online_first_body_failure": "actual_product_fallback_scored_no_body_retry",
        "evaluation_protocol": evaluation_protocol, "evaluation_protocol_sha256": common.digest(evaluation_protocol),
        "online_decision_version": ONLINE_DECISION_VERSION_SAFETY,
        "judge": "independent_codex_only_no_api_judge", "effect_claim_eligible": False}
    frozen["seal_sha256"] = common.digest(frozen)
    target = Path(campaign)
    target.mkdir(parents=True, exist_ok=False)
    for row in frozen["source"]:
        source_path = regular(row["path"])
        raw = source_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("e1c_source_changed_during_freeze")
        saved = target / "frozen-source" / source_path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open("xb") as stream:
            stream.write(raw)
    verify_source_archive(frozen)
    common.write_once(target / "registration.json", frozen)
    if mode == "software_fixture":
        common.write_once(target / "SOFTWARE_ONLY.json", {"real_calls": False})
    return frozen
def authorize_template(frozen):
    return {"schema": "e1cv2_execution_authorization_v1", "status": "draft",
        "registration_sha256": frozen["seal_sha256"], "mode": frozen["mode"],
        "new_allowance_usd": frozen["new_allowance_usd"], "history_sha256": common.digest(frozen["history"]),
        "finance_sha256": common.digest(frozen["finance"]), "human_approval_file_sha256": None,
        "human_approval_file": None, "credentials": "explicit_callback_only",
        "physical_max": guard_code.PHYSICAL_MAX, "unknown_rule": frozen["unknown_rule"]}
def verify_authority(campaign, authorization_sha256, *, for_dispatch=False):
    campaign = Path(campaign).resolve()
    frozen = common.read_json(campaign / "registration.json")
    auth = common.read_json(campaign / "authorization.json")
    if (common.digest(auth) != authorization_sha256 or auth["status"] != "approved"
            or frozen["seal_sha256"] != common.digest({k: v for k, v in frozen.items() if k != "seal_sha256"})
            or frozen["campaign"] != str(campaign)):
        raise ValueError("e1c_registration_or_authorization_changed")
    expected = authorize_template(frozen)
    expected.update(status="approved", human_approval_file=auth.get("human_approval_file"),
                    human_approval_file_sha256=auth.get("human_approval_file_sha256"))
    if auth != expected or sha(regular(auth["human_approval_file"])) != auth["human_approval_file_sha256"]:
        raise ValueError("e1c_explicit_human_approval_required")
    if verify_history(frozen["history"]) != frozen["finance"] or source_inventory() != frozen["source"]:
        raise ValueError("e1c_frozen_source_or_history_changed")
    verify_source_archive(frozen)
    original.product_configuration_preflight()
    product = original.protocol()
    if frozen["client_configuration"] != {name: product[name] for name in ("models", "max_output_tokens", "transport")}:
        raise ValueError("e1c_registered_client_configuration_changed")
    if frozen["mode"] == "real_model":
        if for_dispatch:
            original.check_age(frozen["frozen_at"])
            if metadata_evidence(Path(frozen["metadata_directory"])) != frozen["metadata"]:
                raise ValueError("e1c_public_metadata_changed")
        else:
            for item in [*frozen["metadata"]["files"], frozen["metadata"]["observation_file"]]:
                if sha(regular(Path(frozen["metadata_directory"]) / item["path"])) != item["sha256"]:
                    raise ValueError("e1c_public_metadata_original_changed")
    source = load_first_submission(frozen["source_directory"], expected_receipt_sha256=frozen["first_receipt_sha256"],
                                   allow_software_fixture=frozen["mode"] == "software_fixture")
    if source.suite.source_sha256 != frozen["source_sha256"] or [c.id for c in heldout_submission(source).suite.items] != frozen["case_ids"]:
        raise ValueError("e1c_first_source_changed")
    if (frozen["evaluation_protocol"] != protocol(source)
            or common.digest(frozen["evaluation_protocol"]) != frozen["evaluation_protocol_sha256"]):
        raise ValueError("e1c_registered_scoring_protocol_changed")
    if (campaign / "SOFTWARE_ONLY.json").exists() != (frozen["mode"] == "software_fixture"):
        raise ValueError("e1c_software_mode_marker_changed")
    return frozen, source
def execute(campaign, *, authorization_sha256, credential_loader, software_first_case_only=False,
            guard_factory=guard_code.E1cV2CallGuard):
    frozen, source = verify_authority(campaign, authorization_sha256, for_dispatch=True)
    campaign = Path(campaign).resolve()
    if software_first_case_only and frozen["mode"] != "software_fixture":
        raise ValueError("e1c_formal_subset_forbidden")
    if guard_factory is not guard_code.E1cV2CallGuard and frozen["mode"] != "software_fixture":
        raise ValueError("e1c_formal_guard_override_forbidden")
    if frozen["mode"] == "software_fixture":
        from aria.engine.policy import minimal_calls
        if (getattr(guard_factory, "software_fixture_only", False) is not True
                or getattr(minimal_calls.urllib.request.urlopen, "software_fixture_only", False) is not True):
            raise ValueError("e1c_software_requires_explicit_local_transport")
    if not callable(credential_loader):
        raise ValueError("e1c_explicit_credential_loader_required")
    started = {"registration_sha256": frozen["seal_sha256"],
        "authorization_sha256": authorization_sha256, "automatic_resume": False}
    common.write_once(campaign / "started.json", started)
    binding = {"registration_sha256": frozen["seal_sha256"], "authorization_sha256": authorization_sha256,
        "history_sha256": common.digest(frozen["history"]), "new_allowance_usd": frozen["new_allowance_usd"]}
    common.write_once(campaign / "budget-binding.json", binding)
    def authority():
        verify_authority(campaign, authorization_sha256, for_dispatch=True)
        if (common.read_json(campaign / "budget-binding.json") != binding
                or common.read_json(campaign / "started.json") != started):
            raise ValueError("e1c_durable_budget_binding_changed")
    guard = guard_factory(campaign / "allowance.sqlite3", campaign=campaign, suite=heldout_submission(source).suite,
        authority_check=authority, max_requests=guard_code.PHYSICAL_MAX, max_observed_cost_usd=frozen["new_allowance_usd"],
        reservation_policy=frozen["reservation_policy"], registration_sha256=frozen["seal_sha256"])
    receipt = {"schema": "e1cv2_execution_receipt_v1", "status": "not_evaluable", "mode": frozen["mode"],
        "registration_sha256": frozen["seal_sha256"], "authorization_sha256": authorization_sha256,
        "inherited_finance": deepcopy(frozen["finance"]), "retained_experimental_answers": 0,
        "software_fixture_only": frozen["mode"] == "software_fixture", "effect_claim_eligible": False}
    try:
        authority()
        # A software exercise can never read a real credential by accident.
        key = "SOFTWARE_PLACEHOLDER" if frozen["mode"] == "software_fixture" else credential_loader()
        token = SyntheticEmotionExperiment.create(campaign, "e1c-v2-main")
        def factory(path):
            clients = MinimalModelClients(*(original.make_client(role, key) for role in ("online_fast", "memory", "reflection")), model_calls_enabled=True)
            clients.reflection.reflection_return_format = "body_json"
            clients.online.online_decision_version = ONLINE_DECISION_VERSION_SAFETY
            return E1cV2ExperimentApp(path, experiment=token, clients=clients, model_call_guard=guard,
                                   software_fixture_only=frozen["mode"] == "software_fixture",
                                   model_reservation_policy=frozen["reservation_policy"])
        runner = E1cV2Runner(source, evidence_kind="real_model", allow_model_calls=True, root=token.root / "run", app_factory=factory, seed=frozen["seed"])
        targets = runner.suite.items[:1] if software_first_case_only else runner.suite.items
        receipt["run_root"] = str(token.root / "run")
        for case in targets:
            runner.run_case(case.id)
        receipt.update(status="software_segment_complete" if software_first_case_only else "generated",
            run_root=str(token.root / "run"), complete_cases=len(targets))
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        receipt["stop_reason"] = guard.snapshot()["stop_reason"]
    finally:
        files = {p.relative_to(campaign).as_posix(): sha(regular(p))
                 for directory in (campaign / "wire", campaign / "visible-first", Path(receipt.get("run_root", campaign / "absent-run")))
                 for p in sorted(directory.rglob("*.json"))}
        common.write_once(campaign / "generation-files.json", files)
        receipt["generation_inventory_sha256"] = sha(campaign / "generation-files.json")
        receipt["accounting"] = guard.snapshot()
        guard.close()
        receipt["receipt_sha256"] = common.digest(receipt)
        common.write_once(campaign / "execution-receipt.json", receipt)
    return receipt
