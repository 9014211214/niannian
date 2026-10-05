"""Approved E1b-R main continuation, fixed selection and shared paid allowance.

Preparation is local. Root alone binds human approval, preserved originals and
the final source before calling execute; no automatic resume or zero-cost reset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.bench.e1b_cases import load_e1b_cases, select_prescreen_balanced
from aria.bench.e1br_runner import E1BRRunner
from aria.engine.memory.reflection_contract import reflection_response_schema
from aria.engine.policy.reflection_json import reflection_body_system_prompt
from aria.engine.policy.observed_transport import validate_trace
from scripts import run_e1b as original
from scripts import run_e1b_repair as common
from scripts import e1b_evaluation as evaluation
from scripts import e1b_main_continuation as retained
from scripts import run_e1b_main_continuation as v9
from scripts import e1br_qualification

CAMPAIGN = Path("artifacts/local/e1br-formal-v1")
TASK = Path("artifacts/local/tasks/20261004-07/e1br")
APPROVAL_PATH = Path("artifacts/local/tasks/20261004-07/user-approval.json")
PARENT_MANIFEST = Path("artifacts/local/tasks/20261004-03/v9-main-failure-originals-v1.manifest.json")
APPROVAL_SHA = "b8b6e1eeb28bf6e20e5557e041a7808639390e1bdfea1cb320d06c7f6e928cd5"
PARENT_MANIFEST_SHA = "f665b097515a1e339dcf3057672344cafc7275d441f10f77b15993666741daec"
PRIOR_E1B = Decimal("0.134894814")
PRIOR_KNOWN = Decimal("1.527273986")
RETAINED_UNKNOWN_HOLD = Decimal("0.320307142")
ALLOWANCE = Decimal("1.044798044")
E1B_LIMIT = Decimal("1.5")
CUMULATIVE_STOP = Decimal("3")
PRIOR_UNKNOWN = 8
PRIOR_DISPATCHES = 556
RETAINED_MAIN = 13
EXTRA_FILES = ()
NEW_COUNTS = {"main": {"online_fast": 996, "memory": 23, "reflection": 24}}
MAX_WIRE_REQUESTS = 3153
BUDGET_CHOICE = "e1br_12_codex_only_usd1_5_cumulative3"
SELECTED = [f"e1b_{group}{number}" for group in "abcdef" for number in (1, 2)]
RunError = original.RunError


def retained_extra_rows(root):
    return []


def continuation_details(root):
    return {}


def runner_for(root, prefix):
    return retained.runner_class(prefix, base_runner=E1BRRunner)


def first_request_lineage(client, payload, scope, key, prefix, root):
    if (client.role != "reflection" or scope["operation_id"] != retained.OPERATION
            or scope["session_id"] != retained.SESSION
            or payload["messages"][1] != prefix["failed_payload"]["messages"][1]):
        raise RunError("e1br_retained_first_reflection_changed")
    return {"continuation_of_request_id": retained.FAILED_REQUEST,
            "retained_prefix_sha256": prefix["summary_sha256"], "new_product_protocol": "rf1_body_json"}


@lru_cache(maxsize=1)
def position_inventory():
    """The exact new logical positions; retained a1 S1 content is excluded."""
    positions = set()
    for case in SELECTED:
        for number in (1, 2):
            if (case, number) != (SELECTED[0], 1):
                positions.update(("online_fast", case, number, "shared", 0, turn) for turn in range(12))
                positions.add(("memory", case, number, "shared", 0, None))
            positions.add(("reflection", case, number, "shared", 0, None))
        positions.update(("online_fast", case, 3, arm, repeat, turn)
                         for arm in ("A0", "A1") for repeat in (1, 2, 3) for turn in range(10))
    return frozenset(positions)


def position_key(role, name, detail):
    if type(detail) is not dict:
        raise RunError("e1br_experiment_position_missing")
    if name == "submit" and role == "online_fast":
        if detail.get("phase") not in {"s1", "s2", "s3"} or type(detail.get("turn")) is not int:
            raise RunError("e1br_experiment_position_invalid")
        key = (role, detail.get("case_id"), int(detail["phase"][1:]), detail.get("arm"), detail.get("repeat"), detail["turn"])
    elif name == "reflect" and role in {"memory", "reflection"} and type(detail.get("session")) is int:
        key = (role, detail.get("case_id"), detail["session"], "shared", 0, None)
    else:
        raise RunError("e1br_role_outside_experiment_position")
    if type(key[4]) is not int or key not in position_inventory():
        raise RunError("e1br_unregistered_logical_position")
    return key


def validate_position_binding(key, scope, rows):
    if key not in position_inventory():
        raise RunError("e1br_unregistered_logical_position")
    expected_operation = (f"e1b-{scope['session_id']}-t{key[5]}" if key[0] == "online_fast"
                          else f"e1b-review-{scope['session_id']}")
    if scope["operation_id"] != expected_operation:
        raise RunError("e1br_position_operation_changed")
    for row in rows:
        saved = tuple(row.get("experiment_position", ()))
        if saved not in position_inventory() or saved[0] != row["role"]:
            raise RunError("e1br_ledger_position_missing")
        same_operation = (row["role"], row["operation_id"], row["session_id"]) == (key[0], scope["operation_id"], scope["session_id"])
        if (saved == key) != same_operation:
            raise RunError("e1br_logical_position_rebound")


def approval(root):
    if not isinstance(APPROVAL_SHA, str):
        raise RunError("e1br_human_approval_not_bound")
    raw = common.read_original(root, APPROVAL_PATH)
    if hashlib.sha256(raw).hexdigest() != APPROVAL_SHA:
        raise RunError("e1br_human_approval_changed")
    value = json.loads(raw)
    if not value.get("user_approval_text"):
        raise RunError("e1br_human_quote_missing")
    return {"approval_file_sha256": APPROVAL_SHA, "human_quote": value["user_approval_text"],
            "human_message_reference": value.get("human_message_reference", "20261004-07")}


def parent_originals(root):
    if not isinstance(PARENT_MANIFEST_SHA, str):
        raise RunError("e1br_parent_originals_not_bound")
    raw = common.read_original(root, PARENT_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != PARENT_MANIFEST_SHA:
        raise RunError("e1br_parent_manifest_changed")
    manifest = json.loads(raw)
    rows = [*manifest["files"], *manifest["frozen_archive_files"]]
    if len(manifest["files"]) != 33 or len(manifest["frozen_archive_files"]) != 125:
        raise RunError("e1br_v9_parent_inventory_changed")
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise RunError("e1br_parent_inventory_invalid")
    for row in rows:
        if hashlib.sha256(common.read_original(root, Path(row["path"]))).hexdigest() != row["sha256"]:
            raise RunError("e1br_parent_bytes_changed")
    ancestors, _ = v9.parent_originals(root)
    return list(dict.fromkeys([*ancestors, Path(root) / PARENT_MANIFEST,
                             *(Path(root) / row["path"] for row in rows)]))


def selection_evidence(root):
    bundle = Path(root) / retained.PARENT / "prescreen/blind-bundle"
    result = evaluation.aggregate(bundle)
    if (result["status"] != "prescreen_labels_complete" or result["valid_primary_labels"] != 162
            or result["planned_reply_positions"] != 162 or result["missing_generation_positions"] != 0):
        raise RunError("e1br_original_162_first_labels_required")
    selected = select_prescreen_balanced(load_e1b_cases(), result["repeat_violations"])
    labels = sorted((bundle / "private/labels/primary").glob("*.json"))
    return {"case_ids": selected["case_ids"], "selection": selected,
            "retained_prescreen_positions": 540, "retained_first_labels": 162,
            "original_aggregate_sha256": original.digest(result),
            "first_label_files": [{"path": path.relative_to(root).as_posix(), "sha256": original.byte_sha(path)} for path in labels]}


def build_engine(root):
    from aria.engine.policy.e1br_calls import E1BRCallGuard, logical_attempts

    root = Path(root).resolve()
    anchor = approval(root)
    parents = parent_originals(root)
    fingerprint = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in parents)
    prefix = retained.load_prefix(root)
    extra_rows = retained_extra_rows(root)
    new_total = sum(NEW_COUNTS["main"].values())
    selection = selection_evidence(root)
    entry = common._load_module("scripts._e1br_execution", "run_e1b.py")
    entry.CAMPAIGN, entry.PRIOR_KNOWN, entry.CUMULATIVE_STOP = CAMPAIGN, PRIOR_KNOWN, CUMULATIVE_STOP
    entry.STAGES, entry.COUNTS, entry.MAX_WIRE_REQUESTS = ("main",), NEW_COUNTS, MAX_WIRE_REQUESTS
    entry.ACCEPTED_OLD_UNKNOWN, entry.RETAINED_UNKNOWN_HOLD = PRIOR_UNKNOWN, RETAINED_UNKNOWN_HOLD
    entry.MIN_INTERVAL_SECONDS, entry.OBSERVE_TRANSPORT = 10, True
    entry.TRANSPORT_VERSION = "connection_trace_v2"
    entry.CONNECTION_PREPARATION_RECOVERY = True
    entry.RECOVERY_POLICY = None  # E1b-R has its own versioned durable policy.
    entry.FILES = (*entry.FILES, "scripts/run_e1br.py", "scripts/e1br_qualification.py",
                   "scripts/e1br_assessment.py", "scripts/e1b_main_continuation.py",
                   "scripts/run_e1b_repair.py", "scripts/run_e1b_main_continuation.py",
                   "scripts/run_e1b_reflection_repair.py", "docs/E1BR_EXECUTION.md", *EXTRA_FILES)
    base_protocol, base_preflight = entry.protocol, entry.product_configuration_preflight

    def check_parent(*, full=False):
        if approval(root) != anchor:
            raise RunError("e1br_approval_changed_during_run")
        retained.assert_prefix_unchanged(prefix, full=full)
        if (any(path.is_symlink() or any(parent.is_symlink() for parent in path.parents) for path in parents)
                or tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in parents) != fingerprint):
            raise RunError("e1br_originals_changed_during_run")
        if full:
            parent_originals(root)

    def preflight():
        base_preflight()
        if ALLOWANCE != min(E1B_LIMIT - PRIOR_E1B, CUMULATIVE_STOP - PRIOR_KNOWN) - RETAINED_UNKNOWN_HOLD:
            raise RunError("e1br_inherited_budget_changed")
        if selection["case_ids"] != SELECTED:
            raise RunError("e1br_balanced_selection_changed")
        check_parent()

    def protocol():
        value = base_protocol()
        value.update(schema="aria_e1br_execution_v1", campaign=CAMPAIGN.name,
                     planned_calls=NEW_COUNTS, planned_total_calls=new_total,
                     max_physical_calls=MAX_WIRE_REQUESTS, max_format_corrections=NEW_COUNTS["main"]["reflection"],
                     max_infrastructure_retries_per_position=2, max_infrastructure_retries_total=2 * new_total,
                     measurement_calls={"prescreen": {"online_fast": 540}, "main": e1br_qualification.COUNTS},
                     prior_lineage_dispatches=PRIOR_DISPATCHES, lineage_physical_upper_bound=PRIOR_DISPATCHES + MAX_WIRE_REQUESTS,
                     retained_prescreen=540, retained_main=RETAINED_MAIN, continuation=prefix["summary"], approval=anchor,
                     selection=selection, parent_manifest_sha256=PARENT_MANIFEST_SHA,
                     request_evidence="exact_fictional_body_before_send_no_headers")
        value["models"].pop("judge")
        value["max_output_tokens"].pop("judge")
        value["product_configuration"].update(reflection_return_format="body_json",
            reflection_body_prompt_sha256=original.text_sha(reflection_body_system_prompt()),
            reflection_local_schema_sha256=original.digest(reflection_response_schema()),
            reflection_first_answer_capture="e1br_reflection_product_v1",
            reflection_shape_corrections=1, reflection_is_secondary=True)
        value["transport"].update(min_interval_seconds=10, observe_transport=True,
            observation_version="connection_trace_v2", transport_diagnostics=True,
            connection_preparation_retries=1,
            application_request_retries=2, retry_pool="one_shared_pool_per_logical_position_including_format_correction",
            retry_content_rule="no_model_body_only", exhausted_online_or_memory="stop_not_evaluable",
            exhausted_reflection="secondary_failure_if_durable_and_guard_permits")
        value["budget"].update(status="human_decision_anchored", choice=BUDGET_CHOICE,
            approved_limit_usd=str(E1B_LIMIT), cumulative_stop_usd=str(CUMULATIVE_STOP), prior_e1b_known_usd=str(PRIOR_E1B),
            prior_known_floor_usd=str(PRIOR_KNOWN), historical_unknown_reserved_usd=str(RETAINED_UNKNOWN_HOLD),
            accepted_old_unknown=PRIOR_UNKNOWN, effective_new_allowance_usd=str(ALLOWANCE),
            current_remaining_usd=str(ALLOWANCE), new_unknown_rule="each_authorized_no_body_attempt_retains_its_own_reservation")
        value["judging"] = "independent_codex_first_labels_only_no_secondary_model_calibration"
        value.update(continuation_details(root))
        return value

    def authorization_template(current=None):
        return {"schema": "e1br_authorization_v1", "status": "pending", "budget_choice": None,
            "campaign": CAMPAIGN.as_posix(), "user_limit_usd": str(E1B_LIMIT), "cumulative_stop_usd": str(CUMULATIVE_STOP),
            "prior_known_usd": str(PRIOR_KNOWN), "prior_e1b_known_usd": str(PRIOR_E1B),
            "historical_unknown_reserved_usd": str(RETAINED_UNKNOWN_HOLD), "accepted_old_unknown": PRIOR_UNKNOWN,
            "effective_new_allowance_usd": str(ALLOWANCE), "max_physical_calls": MAX_WIRE_REQUESTS,
            "max_infrastructure_retries_per_position": 2, "max_format_corrections_per_reflection": 1,
            "new_unknown_rule": "retain_each_authorized_attempt_reservation_within_inclusive_budget",
            "approval": anchor, "parent_manifest_sha256": PARENT_MANIFEST_SHA,
            "workspace_identity": entry.workspace_identity(current) if current is not None else None,
            "ledger_identity": entry.text_sha(os.path.normcase(str(entry.campaign(current) / "allowance.sqlite3"))) if current is not None else None}

    def validate_authorization(current, frozen, *, budget_choice, campaign_authorization_sha256, stage_authorization_sha256):
        preflight()
        expected = authorization_template(root)
        expected.update(status="approved", budget_choice=BUDGET_CHOICE)
        auth = entry.read_json(entry.campaign(root) / "authorization.json")
        phase = {"schema": "e1b_stage_authorization_v1", "status": "approved", "stage": "main",
                 "registration_sha256": frozen["seal_sha256"], "campaign_authorization_sha256": campaign_authorization_sha256}
        if (Path(current).resolve() != root or budget_choice != BUDGET_CHOICE or auth != expected
                or original.digest(auth) != campaign_authorization_sha256
                or entry.read_json(entry.campaign(root) / "main.authorization.json") != phase
                or original.digest(phase) != stage_authorization_sha256
                or frozen["workspace_identity"] != entry.workspace_identity(root)
                or frozen["protocol"] != protocol() or frozen["source"] != entry.inventory(root)):
            raise RunError("e1br_specific_frozen_authorization_required")
        return auth, ALLOWANCE

    def stage_inputs(current, stage):
        check_parent()
        if Path(current).resolve() != root or stage != "main":
            raise RunError("e1br_only_main_no_new_prescreen_or_judge")
        return {**selection, "retained": prefix["summary"], **continuation_details(root)}

    def assert_clean(state, *, expected_calls=None, final=False):
        rows = [row for row in logical_attempts(state) if row["product_step"] == "initial"]
        completed_inventory = len(rows) == new_total and Counter(row["role"] for row in rows) == Counter(NEW_COUNTS["main"])
        final_limit = completed_inventory and state.get("stop_reason") in {"request_limit_reached", "inclusive_cost_limit_reached"}
        if ((state.get("stopped") and not final_limit) or len(state["attempts"]) != state["calls_used"]
                or any(row["status"] == "pending" for row in state["attempts"])
                or any(row["status"] != "received" for row in rows if row["role"] != "reflection")
                or expected_calls is not None and len(rows) != expected_calls):
            raise RunError("e1br_primary_or_accounting_incomplete")
        if expected_calls is not None and Counter(row["role"] for row in rows) != Counter(NEW_COUNTS["main"]):
            raise RunError("e1br_basic_role_denominator_changed")
        for row in state["attempts"]:
            validate_position_binding(tuple(row.get("experiment_position", ())), row, state["attempts"])
        if Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"]) > ALLOWANCE:
            raise RunError("e1br_inclusive_budget_exceeded")

    def check_wire_evidence(state):
        for number, row in enumerate(state["attempts"], 1):
            relative = Path("main/wire-requests") / f"{number:04d}.json"
            if (row.get("e1b_stage") != "main" or row.get("wire_request_path") != relative.as_posix()
                    or hashlib.sha256(common.read_original(root, CAMPAIGN / relative)).hexdigest() != row["request_sha256"]):
                raise RunError("e1br_actual_wire_changed")
            if row.get("visible_content_sha256"):
                visible_path = Path("main/visible-first") / f"{number:04d}.json"
                raw = common.read_original(root, CAMPAIGN / visible_path)
                visible = json.loads(raw)
                if (row.get("visible_original_path") != visible_path.as_posix()
                        or row.get("visible_original_sha256") != hashlib.sha256(raw).hexdigest()
                        or visible != {"schema": "e1br_visible_content_original_v1", "request_id": row["request_id"],
                            "role": row["role"], "content": visible.get("content")}
                        or type(visible["content"]) is not str
                        or original.text_sha(visible["content"]) != row["visible_content_sha256"]):
                    raise RunError("e1br_visible_first_original_changed")
        validate_transport_records(state["attempts"], protocol())

    def validate_transport_records(records, frozen_protocol):
        if frozen_protocol["transport"]["observe_transport"] is not True:
            raise RunError("e1br_transport_observation_required")
        for row in records:
            validate_trace(row.get("transport_events", {}), require_complete=row["status"] == "received", require_https=True)

    def qualify(run_root, frozen, state):
        check_parent(full=True)
        check_wire_evidence(state)
        return e1br_qualification.qualify(run_root, frozen, state, retained_attempts=[*prefix["retained_attempts"], *extra_rows],
            retained_summary=prefix["summary"], logical_attempts=logical_attempts)

    class EntryGuard(E1BRCallGuard):
        def __init__(self, *args, stage, registration_sha256, **kwargs):
            self.stage, self.registration_sha256 = stage, registration_sha256
            super().__init__(*args, authority_check=self.check_authority, **kwargs)

        @contextmanager
        def experiment_position(self, name, detail):
            if getattr(self._local, "experiment_position", None) is not None:
                raise RunError("e1br_nested_experiment_position")
            self._local.experiment_position = (name, dict(detail))
            try:
                yield
            finally:
                self._local.experiment_position = None

        def check_authority(self):
            base = entry.campaign(root)
            frozen = entry.read_json(base / "main/registration.json")
            binding, started = entry.read_json(base / "budget-binding.json"), entry.read_json(base / "main.started.json")
            validate_authorization(root, frozen, budget_choice=BUDGET_CHOICE,
                campaign_authorization_sha256=binding["campaign_authorization_sha256"],
                stage_authorization_sha256=started["stage_authorization_sha256"])

        def _record_receipt(self, client, row, data):
            # Preserve only the requested visible content channel, never the
            # provider envelope's reasoning or reasoning_details fields.
            choices = data.get("choices") if type(data) is dict else None
            message = choices[0].get("message") if type(choices) is list and choices and type(choices[0]) is dict else None
            content = message.get("content") if type(message) is dict else None
            if type(content) is str:
                relative = Path("main/visible-first") / Path(row["wire_request_path"]).name
                path = entry.campaign(root) / relative
                entry.write_once(path, {"schema": "e1br_visible_content_original_v1",
                    "request_id": row["request_id"], "role": row["role"], "content": content})
                row.update(visible_original_path=relative.as_posix(), visible_original_sha256=original.byte_sha(path),
                           visible_content_sha256=original.text_sha(content))
            return super()._record_receipt(client, row, data)

        def _reserve(self, client, request, payload, state):
            self.check_authority()
            position = getattr(self._local, "experiment_position", None)
            if position is None:
                raise RunError("e1br_experiment_position_missing")
            key = position_key(client.role, *position)
            validate_position_binding(key, self._local.scope, state["attempts"])
            first = not state["attempts"]
            lineage = {}
            if first:
                lineage = first_request_lineage(client, payload, self._local.scope, key, prefix, root)
            row = super()._reserve(client, request, payload, state)
            relative = Path("main/wire-requests") / f"{len(state['attempts']) + 1:04d}.json"
            common.write_wire_body(entry.campaign(root) / relative, request.data)
            row.update(e1b_stage="main", registration_sha256=self.registration_sha256,
                       wire_request_path=relative.as_posix(), experiment_position=list(key))
            row.update(lineage)
            return row

    def make_client(role, key):
        client = original.make_client(role, key)
        if role == "reflection":
            client.reflection_return_format = "body_json"
        return client

    entry.protocol, entry.product_configuration_preflight = protocol, preflight
    entry.authorization_template, entry.validate_authorization = authorization_template, validate_authorization
    entry.stage_inputs, entry.assert_clean, entry.qualify = stage_inputs, assert_clean, qualify
    entry.metadata_evidence = entry.base_metadata_evidence
    entry.EvidenceGuard, entry.make_client = EntryGuard, make_client
    entry.logical_attempts, entry.check_wire_evidence = logical_attempts, check_wire_evidence
    entry.export_protocol = lambda frozen: evaluation.draft_protocol(load_e1b_cases(), phase="main", case_ids=SELECTED, revision="e1br_v1")
    entry.E1bRunner = runner_for(root, prefix)
    entry.receipt_extensions = lambda: {"inherited_unknown_requests": PRIOR_UNKNOWN,
        "retained_main_prefix_sha256": prefix["summary_sha256"], "prior_main_first_answers": RETAINED_MAIN,
        "second_judge": "not_planned_by_approved_revision", "reflection_is_secondary": True}
    entry.retained_prefix = prefix
    entry.validate_transport_records = validate_transport_records
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("assess")
    report.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "verify", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--registration", type=Path, required=True)
        if name == "execute":
            command.add_argument("--allow-local-env", action="store_true")
            command.add_argument("--budget-choice", choices=(BUDGET_CHOICE,), required=True)
            command.add_argument("--campaign-authorization-sha256", required=True)
            command.add_argument("--stage-authorization-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "assess":
        from scripts.e1br_assessment import assess
        result = assess(ROOT)
        original.write_once(args.output, result)
    else:
        entry = build_engine(ROOT)
        if args.command == "prepare":
            value = entry.prepare(ROOT, args.metadata, args.registration, stage="main")
            result = {"stage": "main", "seal_sha256": value["seal_sha256"], "authorization": entry.authorization_template(ROOT), "paid_calls": 0}
        elif args.command == "verify":
            value = entry.verify(ROOT, args.metadata, args.registration)
            result = {"verified": True, "seal_sha256": value["seal_sha256"], "paid_calls": 0}
        else:
            result = entry.execute(ROOT, args.metadata, args.registration, allow_local_env=args.allow_local_env,
                budget_choice=args.budget_choice, campaign_authorization_sha256=args.campaign_authorization_sha256,
                stage_authorization_sha256=args.stage_authorization_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
