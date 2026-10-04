"""E1b-R capture: immutable initial reflection plus one product correction.

Infrastructure retries belong to the shared ledger; they never select a better
model answer. The correction is an explicit product step, reported separately.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import nullcontext

from .e1b_runner import E1bRunner, E1bRunStopped, _ReflectionCapture, _reflection_product_diagnostic
from ..engine.policy.reflection_json import reflection_product_step


class _StepCapture(_ReflectionCapture):
    def __init__(self, *args, product_step, **kwargs):
        super().__init__(*args, **kwargs)
        self.product_step = product_step
        if product_step == "format_correction":
            self.attempt_path = self.directory / f"reflection-s{self.session}.correction.attempt.json"
            self.first_path = self.directory / f"reflection-s{self.session}.correction.first.json"

    def receipt_rows(self):
        return self.app.call_guard.snapshot()["attempts"]

    def _bind_receipt(self, first, raw):
        first["product_step"] = self.product_step
        if self.app.call_guard is None:
            return True
        rows = [row for row in self.receipt_rows()
                if row.get("role") == "reflection" and row.get("session_id") == self.session_id
                and row.get("product_step", "initial") == self.product_step]
        visible = [row for row in rows if row.get("visible_content_sha256")]
        if len(visible) > 1 or not rows or visible and visible[0] is not rows[-1]:
            return False
        row = visible[0] if visible else rows[-1]
        if (row.get("operation_id") != f"e1b-review-{self.session_id}"
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("request_sha256", ""))):
            return False
        first.update(request_id=row["request_id"], request_sha256=row["request_sha256"],
                     request_hash_kind="actual_wire", physical_attempt_count=len(rows),
                     physical_request_ids=[item["request_id"] for item in rows],
                     receipt_status=row["status"])
        if row.get("visible_content_sha256"):
            first["wire_visible_content_sha256"] = row["visible_content_sha256"]
            first["visible_body_received"] = True
            if row.get("visible_original_path"):
                first.update(visible_original_path=row["visible_original_path"],
                             visible_original_sha256=row["visible_original_sha256"])
            if raw is None:
                first["parse_status"] = "body_received_but_client_rejected"
        else:
            first["visible_body_received"] = False
        returned = row.get("client_return_content_sha256")
        if raw is not None:
            return row["status"] == "received" and returned == first["visible_response_sha256"]
        if returned:
            first.update(visible_response_sha256=returned, response_hash_source="receipt_only")
        return row["status"] == "failed"


class ProductReflectionCapture:
    """Two separately saved product steps; captures cannot conceal any failure."""

    def __init__(self, *args, **kwargs):
        self.initial = _StepCapture(*args, product_step="initial", **kwargs)
        self.correction = _StepCapture(*args, product_step="format_correction", **kwargs)
        self.steps = []
        self.session_id = self.initial.session_id

    def complete(self, client, messages):
        step = reflection_product_step(messages)
        expected = "initial" if not self.steps else "format_correction"
        if len(self.steps) >= 2 or step != expected:
            raise E1bRunStopped("e1br_reflection_product_step_order_changed")
        capture = self.initial if step == "initial" else self.correction
        if step == "format_correction" and not (self.initial.first or {}).get("visible_response_sha256"):
            raise E1bRunStopped("e1br_correction_requires_first_model_content")
        self.steps.append(step)
        return capture.complete(client, messages)

    def finish(self, updater, *, updater_raised=False):
        captures = [self.initial] + ([self.correction] if len(self.steps) == 2 else [])
        invalid = [cap for cap in captures if cap.first is None or cap.capture_error
                   or cap.first.get("request_identity_missing")]
        value = {"schema": "e1br_reflection_product_v1", "session": self.initial.session,
                 "session_id": self.initial.session_id, "first_available": self.initial.first is not None,
                 "first_sha256": hashlib.sha256(self.initial.first_path.read_bytes()).hexdigest()
                     if self.initial.first is not None else None,
                 "format_correction_count": max(0, len(self.steps) - 1),
                 "product_steps": [{"product_step": cap.product_step,
                                    "first_path": cap.first_path.name,
                                    "first_sha256": hashlib.sha256(cap.first_path.read_bytes()).hexdigest()
                                        if cap.first is not None else None} for cap in captures],
                 "capture_error": "capture_missing_or_unbound" if invalid else None,
                 "updater_raised": updater_raised, "secondary_indicator_only": True,
                 "format_attempts": getattr(updater, "last_format_attempts", []),
                 **_reflection_product_diagnostic(updater)}
        self.initial._save(self.initial.outcome_path, value)
        if invalid:
            raise E1bRunStopped("e1br_reflection_capture_missing_or_unbound")


class E1BRRunner(E1bRunner):
    reflection_capture_class = ProductReflectionCapture

    def _operation(self, app, name, operation, *, detail=None):
        scope = nullcontext()
        if app.call_guard is not None and name in {"submit", "reflect"}:
            scope = app.call_guard.experiment_position(name, detail)
        with scope:
            return super()._operation(app, name, operation, detail=detail)
