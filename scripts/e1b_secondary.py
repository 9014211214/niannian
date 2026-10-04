"""Forty joint secondary judgments through the caller's existing allowance.

This is not an authorization or a standalone paid entry. The campaign owner
must first verify the frozen main run and its presampled bundle. This helper
never obtains credentials, constructs a client, resets a ledger, or retries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aria._openrouter import DEFAULT_MEMORY_MODEL
from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.reserved_calls import ReservedCallGuard
from scripts import e1b_evaluation as evaluation


class SecondaryStopped(ValueError):
    """Safe diagnostic code; response text is never included."""


def _parse(raw):
    if not isinstance(raw, str):
        raise SecondaryStopped("secondary_nontext_reply")
    try:
        return json.loads(raw, object_pairs_hook=evaluation._unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, RecursionError):
        raise SecondaryStopped("secondary_invalid_json") from None


def _save_invalid(bundle, packet, item, scorer, raw, reason):
    """The dispatched position is known even if its returned ID is wrong."""
    record = {"schema": "e1b_invalid_first_label_v1", "packet_sha256": evaluation.digest(packet),
              "scorer": scorer, "role": "secondary", "id": item["id"],
              "expected_content_sha256": item["content_sha256"],
              "submission_row_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()
              if isinstance(raw, str) else None, "error": reason}
    path = Path(bundle) / "private" / "labels" / "secondary" / (item["id"] + ".json")
    evaluation.write_once(path, record)
    evaluation.write_once(path.with_suffix(".seal.json"), {"first_label_sha256": evaluation.digest(record)})


def run_secondary(bundle, guard, client, output, *, expected_packet_sha256):
    """Consume each frozen position once, returning only a bounded receipt.

    On a failed call, malformed label or save failure, preserve the dispatched
    position and halt the shared campaign. A different output path cannot resume
    this bundle. The caller retains ownership of the guard and its final receipt.
    """
    packet, key = evaluation.load_bundle(bundle)
    sample = evaluation._load_secondary(bundle, packet)
    if (evaluation.digest(packet) != expected_packet_sha256
            or packet["evidence_kind"] != "real_model"
            or key["protocol"]["status"] != "registered"
            or key["protocol"]["planned_denominator"] != 12):
        raise SecondaryStopped("secondary_registered_real_packet_required")
    if (not isinstance(guard, ReservedCallGuard)
            or not isinstance(client, OpenRouterLLMClient)
            or client.role != "judge" or client.model != DEFAULT_MEMORY_MODEL
            or getattr(client, "model_request_control", None) is not guard
            or client.max_retries != 0 or client.fallback_models
            or not client.bounded_transport):
        raise SecondaryStopped("secondary_shared_guarded_judge_required")
    guard._check_policy()
    state = guard.snapshot()
    if state["stopped"] or any(row["role"] == "judge" for row in state["attempts"]):
        raise SecondaryStopped("secondary_ledger_stopped_or_previously_dispatched")
    scorer = {"id": "claude_secondary_api", "independent": True, "authored_cases": False}
    _, _, primary_scorers = evaluation._labels(bundle, packet, "primary")
    if scorer["id"] in primary_scorers:
        raise SecondaryStopped("secondary_primary_scorer_overlap")
    if any((Path(bundle) / "private" / "labels" / "secondary").glob("*.json")):
        raise SecondaryStopped("secondary_first_labels_already_exist")
    items = {item["id"]: item for item in packet["items"]}
    output = Path(output)
    if output.exists():
        raise SecondaryStopped("secondary_output_already_exists")
    # Bundle marker is exclusive and independent of caller-selected output.
    evaluation.write_once(Path(bundle) / "private" / "secondary-dispatch.started.json", {
        "schema": "e1b_secondary_dispatch_v1", "packet_sha256": expected_packet_sha256,
        "sample_sha256": evaluation.digest(sample), "ledger_id": guard.ledger_id,
        "reservation_policy_sha256": guard.policy_sha256, "planned_positions": 40,
        "automatic_resume": False})
    output.mkdir(parents=True, exist_ok=False)
    completed = 0
    failure = None
    failed_position = None
    try:
        for opaque in sample["ids"]:
            item = items[opaque]
            failed_position = opaque
            # Only public third-session dialogue, reply and rubric are sent.
            messages = [{"role": "system", "content": evaluation.CLAUDE_PROTOCOL},
                        {"role": "user", "content": evaluation.canonical(item).decode("utf-8")}]
            operation = "e1b-judge-" + opaque
            evaluation.write_once(output / (opaque + ".attempt.json"), {
                "id": opaque, "content_sha256": item["content_sha256"],
                "operation_id": operation, "ledger_id": guard.ledger_id,
                "messages_sha256": evaluation.digest(messages), "status": "started"})
            raw = None
            try:
                before = guard.snapshot()
                with guard.scope("review", operation, "e1b-secondary"):
                    raw = client.complete(messages)
                after = guard.snapshot()
                from aria.bench.e1b_runner import _first_answer_attempts
                attempts = _first_answer_attempts(guard, after)
                prior_attempts = _first_answer_attempts(guard, before)
                if (len(attempts) != len(prior_attempts) + 1
                        or attempts[:-1] != prior_attempts
                        or after["attempts"][:len(before["attempts"])] != before["attempts"]):
                    raise SecondaryStopped("secondary_receipt_inventory_mismatch")
                received = attempts[-1]
                if (received.get("status") != "received" or received.get("role") != "judge"
                        or received.get("operation_id") != operation
                        or received.get("session_id") != "e1b-secondary"
                        or not isinstance(raw, str)
                        or received.get("client_return_content_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()
                        or "cost" not in received.get("usage", {})):
                    raise SecondaryStopped("secondary_visible_response_receipt_mismatch")
                row = _parse(raw)
                evaluation._validate_label(row, item)
            except Exception as error:
                reason = "secondary_call_or_label_invalid"
                if isinstance(error, SecondaryStopped):
                    reason = str(error)
                _save_invalid(bundle, packet, item, scorer, raw, reason)
                raise SecondaryStopped(reason) from None
            # Strict schema excludes hidden reasoning and arbitrary extra fields.
            evaluation.write_once(output / (opaque + ".first.json"), {
                "id": opaque, "visible_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "visible_response": raw, "row": row})
            evaluation.import_labels(bundle, {"schema": "e1b_labels_v1", "packet_sha256": expected_packet_sha256,
                                            "scorer": scorer, "labels": [row]}, role="secondary")
            completed += 1
            failed_position = None
    except Exception as error:
        failure = type(error).__name__
        guard.halt()
    receipt = {"schema": "e1b_secondary_execution_v1", "packet_sha256": expected_packet_sha256,
               "sample_sha256": evaluation.digest(sample), "ledger_id": guard.ledger_id,
               "planned_positions": 40, "completed_positions": completed,
               "status": "complete" if completed == 40 and failure is None else "not_evaluable",
               "failed_position": failed_position, "failure_type": failure,
               "joint_six_labels_per_request": True, "automatic_resume": False,
               "accounting": guard.snapshot()}
    try:
        evaluation.write_once(output / "receipt.json", receipt)
    except Exception:
        guard.halt()
        raise SecondaryStopped("secondary_receipt_save_failed") from None
    return receipt
