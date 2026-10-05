"""E1c-specific immutable registration, authorization and shared paid guard.

This module has no environment/key reader. A supplied credential callback is
invoked only after all explicit authority/source/history checks and one-time
start markers. Tests replace transport, not these checks.
"""
from contextlib import nullcontext
from copy import deepcopy
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

from aria.bench.e1b_runner import E1bRunner
from aria.bench.e1c_runner import E1cRunner, E1cExperimentApp
from aria.bench.e1c_source import load_first_submission
from aria.bench.e1c_registration import FinancialInheritance, financial_preparation
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment
from aria.engine.policy.e1br_calls import validate_e1br_ledger
from aria.engine.policy.reserved_calls import validate_reservation_policy
from aria.runtime import MinimalModelClients
from scripts import e1b_evaluation as common
from scripts import run_e1b as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT_DEPENDENCIES = (
    "run_e1c.py", "e1c_execution.py", "e1c_calls.py", "e1c_qualification.py", "e1c_scoring.py",
    "run_e1b.py", "run_ctest.py", "prepare_model_policy.py", "run_e1b_repair.py",
    "e1b_evaluation.py", "e1b_formation.py", "e1b_secondary.py",
    "e1br_qualification.py", "e1br_format_qualification.py", "e1br_format_runtime.py",
)
def sibling(name):
    import importlib
    return importlib.import_module(f"scripts.{name}")


guard_code = sibling("e1c_calls")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular(path):
    path = Path(path).absolute()
    if (path.name.casefold() == ".env" or any(p.is_symlink() or
            getattr(p.lstat(), "st_file_attributes", 0) & 0x400 for p in (path, *path.parents) if p.exists())):
        raise ValueError("e1c_evidence_path_forbidden")
    return path


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


def metadata_evidence(directory):
    """Check saved public inputs before the shared parser reads any bytes."""
    from scripts.run_ctest import METADATA, OBSERVATIONS
    directory = regular(directory)
    for name in (*METADATA, OBSERVATIONS):
        regular(directory / name)
    return original.base_metadata_evidence(directory)


def verify_history(history):
    """Recheck the final E1b ledger and carry its entire authorized fee baseline.

    Old unknowns remain unknown: their individually identified manifest is
    sealed by the new human authorization, not declared provider-settled here.
    """
    for kind in ("registration", "receipt", "ledger", "unknown_manifest"):
        item = history[kind]
        if sha(regular(item["path"])) != item["sha256"]:
            raise ValueError("e1c_history_original_changed")
    frozen, receipt = (common.read_json(history[k]["path"]) for k in ("registration", "receipt"))
    if (frozen["seal_sha256"] != common.digest({k: v for k, v in frozen.items() if k != "seal_sha256"})
            or receipt["registration_sha256"] != frozen["seal_sha256"]
            or receipt.get("status") not in {"completed", "not_evaluable"}
            or receipt.get("receipt_sha256") != common.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})):
        raise ValueError("e1c_history_registration_binding_changed")
    path = regular(history["ledger"]["path"])
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("e1c_history_ledger_not_closed")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        max_requests = db.execute("SELECT max_requests FROM settings WHERE id=1").fetchone()[0]
    rows = validate_e1br_ledger(path, stage="main", registration_sha256=frozen["seal_sha256"],
        max_requests=max_requests, reservation_policy=frozen["metadata"]["policy"])
    state = receipt["accounting"]
    if rows != state["attempts"] or any(row["status"] == "pending" for row in rows):
        raise ValueError("e1c_history_receipt_or_pending_changed")
    known = sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r["usage"]), Decimal(0))
    held = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r["usage"]), Decimal(0))
    if (known != Decimal(state["known_cost_usd"] or "0") or held != Decimal(state["reserved_cost_usd"])
            or state["calls_used"] != len(rows)
            or state["unknown_cost_requests"] != sum("cost" not in r["usage"] for r in rows)):
        raise ValueError("e1c_history_cost_totals_changed")
    inherited = FinancialInheritance.model_validate(history["financial_snapshot"])
    budget = frozen["protocol"]["budget"]
    records = common.read_json(history["unknown_manifest"]["path"])
    if (records != [r.model_dump(mode="json") for r in inherited.unknown_holds]
            or Decimal(inherited.known_cumulative_usd) != Decimal(budget["prior_known_floor_usd"]) + known
            or Decimal(inherited.known_e1b_usd) != Decimal(budget["prior_e1b_known_usd"]) + known
            or sum((Decimal(r.amount) for r in inherited.unknown_holds), Decimal(0)) !=
                Decimal(budget["historical_unknown_reserved_usd"]) + held
            or len(inherited.unknown_holds) != budget["accepted_old_unknown"] + state["unknown_cost_requests"]
            or inherited.prior_ledger_sha256 != history["ledger"]["sha256"]
            or inherited.prior_assessment_sha256 != history["receipt"]["sha256"]):
        raise ValueError("e1c_history_inherited_totals_or_identity_changed")
    # New unresolved rows must appear individually, including their original
    # request SHA and full reservation; equal aggregate totals are insufficient.
    for row in rows:
        if "cost" not in row["usage"]:
            matches = [item for item in inherited.unknown_holds if item.request_id == row["request_id"]
                       and item.request_sha256 == row["request_sha256"]
                       and Decimal(item.amount) == Decimal(row["reservation"]["reserved_cost_usd"])
                       and item.campaign == str(path.parent.resolve())]
            if len(matches) != 1:
                raise ValueError("e1c_history_new_unknown_identity_missing")
    return financial_preparation(inherited)


def prepare(campaign, *, source_directory, first_receipt_sha256, history, reservation_policy,
            seed, e1c_limit_usd, cumulative_limit_usd, mode="real_model", metadata_directory=None):
    if mode not in {"real_model", "software_fixture"} or type(seed) is not int:
        raise ValueError("e1c_explicit_mode_and_seed_required")
    source = load_first_submission(source_directory, expected_receipt_sha256=first_receipt_sha256,
                                   allow_software_fixture=mode == "software_fixture")
    if (source.suite.provenance == "software_fixture_only") != (mode == "software_fixture"):
        raise ValueError("e1c_source_mode_changed")
    finance = verify_history(history)
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
    allowance = min(limits[0], limits[1] - Decimal(finance["committed_cumulative_usd"]))
    if allowance <= 0:
        raise ValueError("e1c_no_inclusive_budget_remaining")
    evaluation_protocol = sibling("e1c_scoring").protocol(source, status="registered")
    product = original.protocol()
    frozen = {"schema": "e1c_registered_execution_v1", "campaign": str(Path(campaign).resolve()),
        "source_archive": str(Path(campaign).resolve() / "frozen-source"),
        "source_directory": str(source.directory), "first_receipt_sha256": first_receipt_sha256,
        "source_sha256": source.suite.source_sha256, "case_ids": [c.id for c in source.suite.items],
        "source": source_inventory(), "history": deepcopy(history), "finance": finance,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "metadata_directory": metadata_directory, "metadata": metadata,
        "reservation_policy": policy, "seed": seed, "mode": mode,
        "client_configuration": {name: product[name] for name in ("models", "max_output_tokens", "transport")},
        "e1c_limit_usd": str(limits[0]), "cumulative_limit_usd": str(limits[1]),
        "new_allowance_usd": str(allowance), "physical_max": 3192,
        "basic_counts": {"online_fast": 1008, "memory": 24, "reflection": 24},
        "reply_positions": 216, "formation_cases": 12, "retained_experimental_answers": 0,
        "infrastructure_retries_per_position": 2, "reflection_corrections_per_position": 1,
        "unknown_rule": "retain_each_full_reservation_including_inherited_manifest",
        "online_first_body_failure": "strict_stop_no_format_retry_no_fallback_effect_credit",
        "evaluation_protocol": evaluation_protocol, "evaluation_protocol_sha256": common.digest(evaluation_protocol),
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
    return {"schema": "e1c_execution_authorization_v1", "status": "draft",
        "registration_sha256": frozen["seal_sha256"], "mode": frozen["mode"],
        "new_allowance_usd": frozen["new_allowance_usd"], "history_sha256": common.digest(frozen["history"]),
        "finance_sha256": common.digest(frozen["finance"]), "human_approval_file_sha256": None,
        "human_approval_file": None, "credentials": "explicit_callback_only",
        "physical_max": 3192, "unknown_rule": frozen["unknown_rule"]}


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
    if source.suite.source_sha256 != frozen["source_sha256"] or [c.id for c in source.suite.items] != frozen["case_ids"]:
        raise ValueError("e1c_first_source_changed")
    if (frozen["evaluation_protocol"] != sibling("e1c_scoring").protocol(source, status="registered")
            or common.digest(frozen["evaluation_protocol"]) != frozen["evaluation_protocol_sha256"]):
        raise ValueError("e1c_registered_scoring_protocol_changed")
    if (campaign / "SOFTWARE_ONLY.json").exists() != (frozen["mode"] == "software_fixture"):
        raise ValueError("e1c_software_mode_marker_changed")
    return frozen, source


class AuthorizedE1cRunner(E1cRunner):
    """Actual shared product runner, opened only by the dedicated authority."""
    def __init__(self, submission, **kwargs):
        self.submission, self._next_case = submission, 0
        E1bRunner.__init__(self, submission.suite, evidence_kind="real_model", allow_model_calls=True, **kwargs)

    def _operation(self, app, name, operation, *, detail=None):
        scope = (app.call_guard.experiment_position(name, detail)
                 if app.call_guard is not None and name in {"submit", "reflect"} else nullcontext())
        with scope:
            return super()._operation(app, name, operation, detail=detail)


def execute(campaign, *, authorization_sha256, credential_loader, software_first_case_only=False,
            guard_factory=guard_code.E1cCallGuard):
    frozen, source = verify_authority(campaign, authorization_sha256, for_dispatch=True)
    campaign = Path(campaign).resolve()
    if software_first_case_only and frozen["mode"] != "software_fixture":
        raise ValueError("e1c_formal_subset_forbidden")
    if guard_factory is not guard_code.E1cCallGuard and frozen["mode"] != "software_fixture":
        raise ValueError("e1c_formal_guard_override_forbidden")
    if frozen["mode"] == "software_fixture":
        from aria.engine.policy import minimal_calls
        if (getattr(guard_factory, "software_fixture_only", False) is not True
                or getattr(minimal_calls.open_observed, "software_fixture_only", False) is not True):
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
    guard = guard_factory(campaign / "allowance.sqlite3", campaign=campaign, suite=source.suite,
        authority_check=authority, max_requests=3192, max_observed_cost_usd=frozen["new_allowance_usd"],
        reservation_policy=frozen["reservation_policy"], stage="main", registration_sha256=frozen["seal_sha256"])
    receipt = {"schema": "e1c_execution_receipt_v1", "status": "not_evaluable", "mode": frozen["mode"],
        "registration_sha256": frozen["seal_sha256"], "authorization_sha256": authorization_sha256,
        "inherited_finance": deepcopy(frozen["finance"]), "retained_experimental_answers": 0,
        "software_fixture_only": frozen["mode"] == "software_fixture", "effect_claim_eligible": False}
    try:
        authority()
        # A software exercise can never read a real credential by accident.
        key = "SOFTWARE_PLACEHOLDER" if frozen["mode"] == "software_fixture" else credential_loader()
        token = SyntheticEmotionExperiment.create(campaign, "e1c-main")
        def factory(path):
            clients = MinimalModelClients(*(original.make_client(role, key) for role in ("online_fast", "memory", "reflection")), model_calls_enabled=True)
            clients.reflection.reflection_return_format = "body_json"
            return E1cExperimentApp(path, experiment=token, clients=clients, model_call_guard=guard,
                                   model_reservation_policy=frozen["reservation_policy"])
        runner = AuthorizedE1cRunner(source, root=token.root / "run", app_factory=factory, seed=frozen["seed"])
        targets = source.suite.items[:1] if software_first_case_only else source.suite.items
        for case in targets:
            runner.run_case(case.id)
        receipt.update(status="software_segment_complete" if software_first_case_only else "generated",
            run_root=str(token.root / "run"), complete_cases=len(targets))
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        guard.halt()
    finally:
        receipt["accounting"] = guard.snapshot()
        guard.close()
        receipt["receipt_sha256"] = common.digest(receipt)
        common.write_once(campaign / "execution-receipt.json", receipt)
    return receipt
