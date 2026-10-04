"""Prospective, append-only item checkpoints; never a historical-run repair tool.

The caller supplies a frozen profile/source and the *entire ordered inventory*
on every open. Only an explicit ``resume=True`` enables new starts. A start with
no durable result is ambiguous, even if a caller believes the request was never
sent. Unknown invoice amounts stop the whole stage; balance changes cannot fill
them in. This module never calls a model, reads credentials, or retries a call.

Hashes detect changed/missing records relative to the caller's trusted inputs;
they are not signatures against an attacker able to rewrite the whole journal.
The absolute location binding prevents treating a copied directory as a resume.
Keep the journal and its fixed profile together in the experiment's provenance.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


VERSION = "1.0"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_BYTES = 16 * 1024 * 1024
_FORBIDDEN = {
    "api_key", "apikey", "authorization", "proxy_authorization", "password",
    "secret", "access_token", "refresh_token", "cookie", "set_cookie",
    "reasoning_content", "reasoning_details", "hidden_reasoning", "chain_of_thought",
    "raw_response", "raw_request", "headers", "private_data",
}
_SECRET_TEXT = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{8,}|\bBearer\s+[A-Za-z0-9._-]{8,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", re.IGNORECASE,
)
_PRIVATE_JSON_KEY = re.compile(
    r'"(?:api[_-]?key|authorization|reasoning_content|reasoning_details|hidden_reasoning|chain_of_thought)"\s*:',
    re.IGNORECASE,
)
_RESULT_FIELDS = {
    "axis", "rubric_version", "rubric_fingerprint", "packet_fingerprint", "metadata",
    "applicability", "status", "dimensions", "critical_issues", "axis_pass",
    "diagnostic", "assessment_transport", "judgment_schema_version",
    "judgment_schema_fingerprint", "judge_provider_diagnostic", "claim_checks",
    "judge_effective_model", "judge_configured_model", "producer_model",
    "assessment_protocol", "formation_protocol_binding", "effective_rubric_fingerprint",
    "judge_instruction_contract", "judge_instructions_fingerprint",
    "applicability_decision", "attribution_scope", "independence_scope",
    "review_coordinate", "coordinate", "reason", "score", "result", "row", "receipt",
    "unavailable", "transport", "request_payloads", "response_body", "capture",
}
_ACCOUNT_FIELDS = {
    "enabled", "version", "scope", "min_request_start_interval_seconds",
    "max_retry_after_seconds", "retry_after_over_budget_policy", "max_network_requests",
    "max_observed_cost_usd", "cost_guard_scope", "actual_network_requests",
    "same_model_retry_network_requests", "http_status_counts", "error_type_counts",
    "output_limit_responses", "wait_seconds_by_role", "total_wait_seconds",
    "stopped_reason", "blocked_before_network", "attempts", "provider_usage_scope",
    "provider_token_totals", "provider_token_observed_requests",
    "provider_cost_observed_requests", "provider_observed_cost_usd",
    "provider_cost_complete", "provider_error_costs_unknown_not_zero",
    "network_latency_scope", "product_runtime_defaults_changed",
    "network_error_kind_counts", "failure_phase_counts", "journal_resume",
    "stop_on_rate_limit",
}
_ATTEMPT_FIELDS = {
    "role", "retry_index", "http_status", "error_type", "retry_after_seconds",
    "provider_usage", "output_limit_reported", "network_elapsed_ms",
    "network_error_kind", "failure_phase",
    "response_observability",
}
_USAGE_FIELDS = {"cost", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"}


class JournalError(ValueError):
    """Invalid source, unsafe payload, or damaged checkpoint (no raw data in errors)."""


class JournalBlocked(JournalError):
    """A journal cannot safely start another item."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise JournalError("journal_non_json_value") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_json(value: Any, *, depth: int = 0) -> None:
    if depth > 64:
        raise JournalError("journal_payload_too_deep")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise JournalError("journal_non_finite_number")
        return
    if isinstance(value, str):
        if _SECRET_TEXT.search(value):
            raise JournalError("journal_credential_like_text_rejected")
        if _PRIVATE_JSON_KEY.search(value):
            raise JournalError("journal_private_field_in_text_rejected")
        # A public receipt may contain JSON strings. Inspect their actual object
        # keys too; a literal JSON-schema property definition is not reasoning.
        if value.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(value, object_pairs_hook=_unique_keys)
            except json.JSONDecodeError:
                pass
            else:
                _safe_json(nested, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _safe_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JournalError("journal_non_string_key")
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN:
                raise JournalError("journal_private_field_rejected")
            if normalized in {"reasoning", "analysis"} and not (
                isinstance(item, dict) and (
                    item.get("type") in {"string", "object", "array", "null"}
                    or set(item) <= {"effort", "exclude", "enabled", "max_tokens"}
                )
            ):
                raise JournalError("journal_private_reasoning_rejected")
            _safe_json(item, depth=depth + 1)
        return
    raise JournalError("journal_non_json_value")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JournalError("journal_duplicate_json_key")
        result[key] = value
    return result


def _seal(value: dict) -> dict:
    return {**deepcopy(value), "fingerprint": _digest(value)}


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise JournalError("journal_missing_or_unsafe_record")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError, UnicodeError) as exc:
        raise JournalError("journal_invalid_record_json") from exc
    _safe_json(value)
    if (not isinstance(value, dict) or "fingerprint" not in value
            or value["fingerprint"] != _digest({k: v for k, v in value.items() if k != "fingerprint"})):
        raise JournalError("journal_record_fingerprint_mismatch")
    return value


def _write_exclusive(path: Path, value: dict) -> None:
    _safe_json(value)
    data = _canonical(value) + b"\n"
    if len(data) > _MAX_BYTES:
        raise JournalError("journal_record_too_large")
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except FileExistsError as exc:
        raise JournalError("journal_record_already_exists_no_overwrite") from exc


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise JournalError("journal_invalid_numeric_budget_or_usage")
    return Decimal(str(value))


def _accounting(value: dict) -> tuple[dict, dict]:
    """Check the cumulative invoice evidence, retaining only public metadata."""
    if not isinstance(value, dict) or not set(value) <= _ACCOUNT_FIELDS:
        raise JournalError("journal_invalid_accounting_fields")
    _safe_json(value)
    if "stop_on_rate_limit" in value and value["stop_on_rate_limit"] is not True:
        raise JournalError("journal_invalid_rate_limit_stop_policy")
    count, attempts = value.get("actual_network_requests"), value.get("attempts")
    if (type(count) is not int or count < 0 or not isinstance(attempts, list)
            or count != len(attempts)):
        raise JournalError("journal_request_count_evidence_mismatch")
    costs = []
    for attempt in attempts:
        if not isinstance(attempt, dict) or not set(attempt) <= _ATTEMPT_FIELDS:
            raise JournalError("journal_invalid_attempt_fields")
        if "response_observability" in attempt:
            from aria.engine.policy.response_observability import validate_attempt_observation
            try:
                validate_attempt_observation(attempt)
            except ValueError as exc:
                raise JournalError("journal_invalid_response_observability") from exc
        usage = attempt.get("provider_usage")
        if not isinstance(usage, dict) or not set(usage) <= _USAGE_FIELDS:
            raise JournalError("journal_invalid_usage_fields")
        for amount in usage.values():
            _number(amount)
        if "cost" in usage:
            costs.append(_number(usage["cost"]))
    if value.get("stop_on_rate_limit") is True:
        limited = [index for index, attempt in enumerate(attempts) if attempt.get("http_status") == 429]
        if (limited and (limited != [len(attempts) - 1]
                         or value.get("stopped_reason") != "http_429_requires_new_registration")
                or not limited and value.get("stopped_reason") == "http_429_requires_new_registration"):
            raise JournalError("journal_rate_limit_stop_evidence_mismatch")
    if value.get("max_observed_cost_usd") is not None:
        # Only the new interrupted-observation ABI promises an immediate stop.
        # Historical rows (notably r15) may have been stopped by the journal
        # before the older controller populated its own stopped_reason.
        interrupted = [index for index, attempt in enumerate(attempts)
                       if isinstance(attempt.get("response_observability"), dict)
                       and attempt["response_observability"].get("version") == "1.1"
                       and attempt["response_observability"].get("envelope_kind") == "interrupted"]
        if interrupted and (interrupted != [len(attempts) - 1]
                            or value.get("stopped_reason") != "provider_cost_unknown_with_cost_guard"):
            raise JournalError("journal_interrupted_cost_stop_evidence_mismatch")
    known = sum(costs, Decimal(0))
    rounded = round(float(known), 8)
    expected_observed = rounded if costs else None
    if (value.get("provider_observed_cost_usd") != expected_observed
            or value.get("provider_cost_complete") is not (bool(count) and len(costs) == count)):
        raise JournalError("journal_cost_summary_evidence_mismatch")
    if ("provider_cost_observed_requests" in value
            and value["provider_cost_observed_requests"] != len(costs)):
        raise JournalError("journal_cost_count_evidence_mismatch")
    return deepcopy(value), {
        "actual_network_requests": count,
        "known_cost_usd": rounded,
        "unknown_cost_requests": count - len(costs),
        "cost_complete": len(costs) == count,
        "total_cost_usd": rounded if len(costs) == count else None,
    }


def _empty_accounting():
    return {"actual_network_requests": 0, "attempts": [],
            "provider_observed_cost_usd": None, "provider_cost_complete": False}


class EvaluationJournal:
    """Single-writer, fail-closed, ordered stage checkpoint.

    ``finish`` takes an authentic *cumulative stage* controller snapshot, not a
    new process's zero-based accounting. Prefix changes, fee repair and resets
    are rejected. ``aggregate()['accounting']`` exposes the saved prefix for a
    caller to combine with newly observed attempts and enforce remaining caps.
    """

    def __init__(self, path: Path, manifest: dict, *, writable: bool):
        self.path, self.manifest = path, manifest
        self._writable = writable
        self._active: dict[str, int] = {}

    @staticmethod
    def _binding(path, *, profile_fingerprint, source_fingerprint, stage, items,
                 max_network_requests, max_observed_cost_usd):
        for value in (profile_fingerprint, source_fingerprint):
            if not isinstance(value, str) or _HASH.fullmatch(value) is None:
                raise JournalError("journal_invalid_source_fingerprint")
        if not isinstance(stage, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", stage) is None:
            raise JournalError("journal_invalid_stage")
        if type(max_network_requests) is not int or max_network_requests < 0:
            raise JournalError("journal_invalid_network_budget")
        _number(max_observed_cost_usd)
        if not isinstance(items, list) or not items:
            raise JournalError("journal_requires_complete_nonempty_inventory")
        seen = set()
        for item in items:
            if (not isinstance(item, dict) or set(item) != {"coordinate", "input_fingerprint"}
                    or not isinstance(item["input_fingerprint"], str)
                    or _HASH.fullmatch(item["input_fingerprint"]) is None):
                raise JournalError("journal_invalid_inventory_item")
            _safe_json(item["coordinate"])
            identity = _digest(item["coordinate"])
            if identity in seen:
                raise JournalError("journal_duplicate_coordinate")
            seen.add(identity)
        return {"version": VERSION, "profile_fingerprint": profile_fingerprint,
                "source_fingerprint": source_fingerprint, "stage": stage,
                "items": deepcopy(items), "denominator": len(items),
                "max_network_requests": max_network_requests,
                "max_observed_cost_usd": max_observed_cost_usd,
                "location_fingerprint": _digest(os.path.normcase(str(Path(path).resolve())))}

    @classmethod
    def create(cls, path, **binding):
        target = Path(path).absolute()
        fixed = cls._binding(target, **binding)
        if target.is_symlink():
            raise JournalError("journal_symlink_location_rejected")
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise JournalError("journal_already_exists_explicit_resume_required") from exc
        manifest = _seal({**fixed, "created_at": datetime.now(timezone.utc).isoformat()})
        _write_exclusive(target / "manifest.json", manifest)
        return cls(target, manifest, writable=True)

    @classmethod
    def open(cls, path, *, resume=False, **binding):
        if type(resume) is not bool:
            raise JournalError("journal_resume_requires_explicit_boolean")
        target = Path(path).absolute()
        if target.is_symlink():
            raise JournalError("journal_symlink_location_rejected")
        expected = cls._binding(target, **binding)
        manifest = _read(target / "manifest.json")
        if ({k: v for k, v in manifest.items() if k not in {"created_at", "fingerprint"}} != expected):
            raise JournalError("journal_profile_source_inventory_budget_or_location_mismatch")
        journal = cls(target, manifest, writable=resume)
        journal.validate()
        return journal

    def _record_path(self, index, kind):
        return self.path / f"{index:06d}.{kind}.json"

    def validate(self):
        """Re-read and verify all evidence; never trust in-memory completion."""
        if _read(self.path / "manifest.json") != self.manifest:
            raise JournalError("journal_manifest_changed")
        allowed = {"manifest.json"}
        completed, ambiguous = [], []
        previous = self.manifest["fingerprint"]
        accounting = _empty_accounting()
        stopped = False
        for index, item in enumerate(self.manifest["items"]):
            paths = {k: self._record_path(index, k) for k in ("claim", "started", "result")}
            present = {k for k, path in paths.items() if path.exists()}
            allowed.update(path.name for path in paths.values())
            if not present:
                stopped = True
                continue
            if stopped:
                raise JournalError("journal_non_prefix_or_missing_record")
            if "claim" not in present:
                raise JournalError("journal_missing_write_ahead_claim")
            claim = _read(paths["claim"])
            expected = {"kind": "started", "index": index, "item": item,
                        "manifest_fingerprint": self.manifest["fingerprint"],
                        "previous_fingerprint": previous,
                        "accounting_before_fingerprint": _digest(accounting)}
            if {k: v for k, v in claim.items() if k not in {"created_at", "fingerprint"}} != expected:
                raise JournalError("journal_start_binding_mismatch")
            if "started" not in present:
                if "result" in present:
                    raise JournalError("journal_result_missing_started_record")
                ambiguous.append(deepcopy(item["coordinate"]))
                stopped = True
                continue
            started = _read(paths["started"])
            if started != claim:
                raise JournalError("journal_started_claim_mismatch")
            if "result" not in present:
                ambiguous.append(deepcopy(item["coordinate"]))
                stopped = True
                continue
            result = _read(paths["result"])
            if (set(result) != {"kind", "index", "started_fingerprint", "result", "accounting",
                               "created_at", "fingerprint"}
                    or result["kind"] != "result" or result["index"] != index
                    or result["started_fingerprint"] != started["fingerprint"]):
                raise JournalError("journal_result_binding_mismatch")
            self._check_result(result["result"])
            next_accounting, _ = _accounting(result["accounting"])
            self._check_prefix(accounting, next_accounting)
            accounting = next_accounting
            completed.append(result)
            previous = result["fingerprint"]
        if any(p.name not in allowed or p.is_symlink() for p in self.path.iterdir()):
            raise JournalError("journal_unregistered_record_or_symlink")
        _, totals = _accounting(accounting)
        return {"completed_records": completed, "ambiguous": ambiguous,
                "accounting": accounting, **totals}

    @staticmethod
    def _check_result(result):
        if not isinstance(result, dict) or not set(result) <= _RESULT_FIELDS:
            raise JournalError("journal_only_public_score_or_receipt_allowed")
        _safe_json(result)

    @staticmethod
    def _check_prefix(before, after):
        count = before["actual_network_requests"]
        if (after["actual_network_requests"] < count
                or after["attempts"][:count] != before["attempts"]):
            raise JournalError("journal_accounting_reset_or_history_changed")

    def aggregate(self):
        state = self.validate()
        return {k: deepcopy(v) for k, v in state.items() if k != "completed_records"} | {
            "completed_items": len(state["completed_records"]),
            "denominator": self.manifest["denominator"],
            "result_fingerprints": [r["fingerprint"] for r in state["completed_records"]],
            "unknown_in_flight_requests": bool(state["ambiguous"]),
            "actual_network_requests_complete": not state["ambiguous"],
            "cost_complete": state["cost_complete"] and not state["ambiguous"],
            "total_cost_usd": None if state["ambiguous"] else state["total_cost_usd"],
        }

    def plan_resume(self):
        state = self.validate()
        completed = len(state["completed_records"])
        blockers = []
        if state["ambiguous"]:
            blockers.append("started_without_durable_result_request_and_fee_ambiguous")
        if state["unknown_cost_requests"]:
            blockers.append("provider_cost_unknown_no_retry_or_balance_backfill")
        if state["accounting"].get("stopped_reason"):
            blockers.append("original_transport_controller_stopped_requires_new_registration")
        if state["actual_network_requests"] >= self.manifest["max_network_requests"]:
            blockers.append("cumulative_network_request_budget_exhausted")
        known = sum((Decimal(str(a["provider_usage"]["cost"]))
                     for a in state["accounting"]["attempts"] if "cost" in a["provider_usage"]), Decimal(0))
        if known >= Decimal(str(self.manifest["max_observed_cost_usd"])):
            blockers.append("cumulative_observed_cost_budget_exhausted")
        untouched = self.manifest["items"][completed + len(state["ambiguous"]):]
        pending = [deepcopy(i["coordinate"]) for i in untouched]
        return {"version": VERSION, "stage": self.manifest["stage"],
                "denominator": self.manifest["denominator"],
                "completed": [deepcopy(i["coordinate"]) for i in self.manifest["items"][:completed]],
                "ambiguous": deepcopy(state["ambiguous"]), "not_started": pending,
                "safe_pending": [] if blockers else pending, "blockers": blockers,
                "completed_count": completed, "pending": pending, "blocked": bool(blockers),
                "resume_requires_explicit_flag": True, "network_calls_made_by_plan": 0,
                "historical_run_import_supported": False}

    def replay(self, coordinate, input_fingerprint):
        state = self.validate()
        index = self._index(coordinate, input_fingerprint)
        if index >= len(state["completed_records"]):
            return None
        return deepcopy(state["completed_records"][index]["result"])

    def _index(self, coordinate, input_fingerprint):
        for index, item in enumerate(self.manifest["items"]):
            if _digest(item["coordinate"]) == _digest(coordinate):
                if item["input_fingerprint"] != input_fingerprint:
                    raise JournalError("journal_input_fingerprint_changed")
                return index
        raise JournalError("journal_coordinate_not_registered")

    def start(self, coordinate, input_fingerprint):
        if not self._writable:
            raise JournalBlocked("journal_read_only_explicit_resume_required")
        plan = self.plan_resume()
        if plan["blockers"]:
            raise JournalBlocked(plan["blockers"][0])
        index = self._index(coordinate, input_fingerprint)
        if index != len(plan["completed"]):
            raise JournalBlocked("journal_original_order_no_repeat_required")
        state = self.validate()
        previous = (state["completed_records"][-1]["fingerprint"] if index
                    else self.manifest["fingerprint"])
        record = _seal({"kind": "started", "index": index,
                        "item": self.manifest["items"][index],
                        "manifest_fingerprint": self.manifest["fingerprint"],
                        "previous_fingerprint": previous,
                        "accounting_before_fingerprint": _digest(state["accounting"]),
                        "created_at": datetime.now(timezone.utc).isoformat()})
        # Both durable records precede returning control to the model caller.
        # A torn write, or deletion of either copy, cannot become a fresh item.
        _write_exclusive(self._record_path(index, "claim"), record)
        _write_exclusive(self._record_path(index, "started"), record)
        token = record["fingerprint"]
        self._active[token] = index
        return token

    def finish(self, token, *, result, accounting):
        if token not in self._active:
            raise JournalBlocked("journal_finish_requires_same_process_active_start")
        index = self._active[token]
        self._check_result(result)
        snapshot, _ = _accounting(accounting)
        state = self.validate()
        self._check_prefix(state["accounting"], snapshot)
        record = _seal({"kind": "result", "index": index, "started_fingerprint": token,
                        "result": result, "accounting": snapshot,
                        "created_at": datetime.now(timezone.utc).isoformat()})
        _write_exclusive(self._record_path(index, "result"), record)
        del self._active[token]
        return deepcopy(result)
