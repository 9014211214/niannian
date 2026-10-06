"""E1c-v2 experiment identity layered on the unchanged interactive guard."""
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3

from aria.engine.policy.product_calls import ProductCallGuard
from scripts import e1b_evaluation as common
from scripts.e1c_calls import positions, position_key
from scripts.run_e1b_repair import write_wire_body

PHYSICAL_MAX = (880 + 20) * 3


class E1cV2CallGuard(ProductCallGuard):
    def __init__(self, path, *, campaign, suite, authority_check, registration_sha256, **kwargs):
        self.campaign = Path(campaign).resolve()
        self.registered_positions = positions(suite)
        self.authority_check = authority_check
        self.registration_sha256 = registration_sha256
        super().__init__(path, **kwargs)

    @contextmanager
    def experiment_position(self, name, detail):
        if getattr(self._local, "experiment_position", None) is not None:
            raise ValueError("e1cv2_nested_position")
        self._local.experiment_position = name, dict(detail)
        try:
            yield
        finally:
            self._local.experiment_position = None

    def _reserve(self, client, request, payload, state):
        self.authority_check()
        scoped = getattr(self._local, "experiment_position", None)
        if scoped is None:
            raise ValueError("e1cv2_position_required")
        key = position_key(client.role, *scoped)
        if key not in self.registered_positions:
            raise ValueError("e1cv2_unregistered_position")
        scope = self._local.scope
        expected = (f"e1b-{scope['session_id']}-t{key[5]}" if client.role == "online_fast"
                    else f"e1b-review-{scope['session_id']}")
        if scope["operation_id"] != expected:
            raise ValueError("e1cv2_operation_identity_changed")
        for previous in state["attempts"]:
            same = (previous["role"], previous["operation_id"], previous["session_id"]) == (
                client.role, scope["operation_id"], scope["session_id"])
            if (tuple(previous["experiment_position"]) == key) != same:
                raise ValueError("e1cv2_position_rebound")
        record = super()._reserve(client, request, payload, state)
        relative = Path("wire") / f"{len(state['attempts']) + 1:04d}.json"
        write_wire_body(self.campaign / relative, request.data)
        record.update(experiment_id="e1c_v2", experiment_position=list(key),
                      registration_sha256=self.registration_sha256,
                      wire_request_path=relative.as_posix())
        return record

    def _record_receipt(self, client, row, data):
        choices = data.get("choices") if type(data) is dict else None
        message = choices[0].get("message") if type(choices) is list and choices and type(choices[0]) is dict else None
        content = message.get("content") if type(message) is dict else None
        if type(content) is str:
            relative = Path("visible-first") / Path(row["wire_request_path"]).name
            common.write_once(self.campaign / relative, {"schema": "e1cv2_visible_first_v1",
                "request_id": row["request_id"], "role": row["role"], "content": content})
            row.update(visible_original_path=relative.as_posix(),
                visible_original_sha256=hashlib.sha256((self.campaign / relative).read_bytes()).hexdigest(),
                visible_content_sha256=hashlib.sha256(content.encode()).hexdigest())
        super()._record_receipt(client, row, data)
        raw = client.role == "online_fast" or getattr(client, "reflection_return_format", "native") == "body_json"
        row["client_return_content_sha256"] = hashlib.sha256((content if raw else content.strip()).encode()).hexdigest()


def closed_ledger(campaign, frozen, state):
    """Read only; retain failed rows, unknown reservations and pause events."""
    root = Path(campaign).resolve()
    path = root / "allowance.sqlite3"
    if not path.is_file() or path.is_symlink() or any(Path(str(path) + s).exists() for s in ("-wal", "-shm", "-journal")):
        raise ValueError("e1cv2_ledger_not_closed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        settings = db.execute("SELECT max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        rows = [json.loads(r[0]) for r in db.execute("SELECT record FROM attempts ORDER BY sequence")]
        events = [json.loads(r[0]) for r in db.execute("SELECT event FROM product_events ORDER BY sequence")]
        policy = json.loads(db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()[0])
        version = db.execute("SELECT version FROM product_recovery WHERE id=1").fetchone()[0]
    if (settings[:2] != (frozen["physical_max"], frozen["new_allowance_usd"])
            or policy != frozen["reservation_policy"] or rows != state["attempts"]
            or events != state["product_events"] or version != state["recovery_contract"]
            or state["calls_used"] != len(rows) or any(r["status"] == "pending" for r in rows)):
        raise ValueError("e1cv2_ledger_binding_changed")
    if settings[2] is not None and not state["stopped"]:
        raise ValueError("e1cv2_durable_halt_disagrees_with_receipt")
    operations, resumes, streak, pause, hard = ProductCallGuard._event_state(events)
    if resumes:
        raise ValueError("e1cv2_automatic_or_owner_resume_not_registered")
    if (any(not r["finished"] for r in operations.values()) or streak != state["consecutive_failures"]
            or pause != state["pause_id"]):
        raise ValueError("e1cv2_operation_events_incomplete")
    known = sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r["usage"]), Decimal(0))
    held = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r["usage"]), Decimal(0))
    if (known != Decimal(state["known_cost_usd"] or "0") or held != Decimal(state["reserved_cost_usd"])
            or state["unknown_cost_requests"] != sum("cost" not in r["usage"] for r in rows)
            or known + held > Decimal(frozen["new_allowance_usd"])):
        raise ValueError("e1cv2_cost_totals_changed")
    if ((pause or hard or state["unknown_cost_requests"] or settings[2]
            or len(rows) >= frozen["physical_max"] or known >= Decimal(frozen["new_allowance_usd"]))
            and not state["stopped"]):
        raise ValueError("e1cv2_stop_event_or_unknown_hidden")
    from aria.engine.policy.e1br_calls import validate_e1br_reservations
    validate_e1br_reservations(rows, policy)
    for index, row in enumerate(rows):
        if row["registration_sha256"] != frozen["seal_sha256"] or row["experiment_id"] != "e1c_v2":
            raise ValueError("e1cv2_registration_binding_changed")
        for name, expected in (("wire_request_path", row["request_sha256"]),
                               ("visible_original_path", row.get("visible_original_sha256"))):
            if name not in row:
                continue
            file = root / row[name]
            if not file.resolve().is_relative_to(root) or file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError("e1cv2_request_or_visible_original_changed")
        wire = (root / row["wire_request_path"]).read_bytes()
        if row["reservation"]["input_token_plan_bound"] != len(wire) + 8192:
            raise ValueError("e1cv2_reservation_not_bound_to_wire")
        if row.get("visible_original_path"):
            original = common.read_json(root / row["visible_original_path"])
            if (original["request_id"] != row["request_id"] or original["role"] != row["role"]
                    or hashlib.sha256(original["content"].encode()).hexdigest() != row["visible_content_sha256"]):
                raise ValueError("e1cv2_visible_binding_changed")
        attempt = row["product_transport_attempt"]
        if attempt != 1:
            parent = next((r for r in rows[:index] if r["request_id"] == row.get("product_retry_parent_request_id")), None)
            if (parent is None or not ProductCallGuard._retryable(parent) or attempt != parent["product_transport_attempt"] + 1
                    or attempt > 3 or row["request_sha256"] != parent["request_sha256"]
                    or row["experiment_position"] != parent["experiment_position"]):
                raise ValueError("e1cv2_retry_chain_changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("e1cv2_ledger_mutated_during_review")
    return digest
