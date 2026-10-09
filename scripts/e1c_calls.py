"""Private E1c instance of the existing reservation/retry guard, no old mutation."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

from aria.engine.policy import e1br_calls as original_calls
from scripts import e1b_evaluation as common
from scripts import run_e1b_repair as files

PHYSICAL_MAX = 3192


def parameterized_calls():
    """Isolated module globals supply a separately versioned E1c contract.

    The original class and its MAX_REQUESTS=3153 remain unchanged. Integration
    can turn these two constants into explicit factory parameters; all retry,
    reservation, response and pacing algorithms stay exactly the shared code.
    """
    name = "aria.engine.policy._e1c_registered_calls"
    spec = importlib.util.spec_from_file_location(name, Path(original_calls.__file__))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.MAX_REQUESTS = PHYSICAL_MAX
    module.SCHEMA = "e1c_no_content_retry_v1"
    return module


CALLS = parameterized_calls()


def positions(suite):
    values = set()
    for case in suite.items:
        for number in (1, 2):
            values.update(("online_fast", case.id, number, "shared", 0, i) for i in range(12))
            values.update((role, case.id, number, "shared", 0, None) for role in ("memory", "reflection"))
        values.update(("online_fast", case.id, 3, arm, repeat, i)
                      for arm in ("A0", "A1") for repeat in (1, 2, 3) for i in range(10))
    return frozenset(values)


def position_key(role, name, detail):
    if name == "submit" and role == "online_fast":
        return role, detail["case_id"], int(detail["phase"][1:]), detail["arm"], detail["repeat"], detail["turn"]
    if name == "reflect" and role in {"memory", "reflection"}:
        return role, detail["case_id"], detail["session"], "shared", 0, None
    raise ValueError("e1c_role_outside_registered_action")


class E1cCallGuard(CALLS.E1BRCallGuard):
    def __init__(self, path, *, campaign, suite, authority_check, **kwargs):
        self.campaign = Path(campaign).resolve()
        self.registered_positions = positions(suite)
        super().__init__(path, authority_check=authority_check, **kwargs)

    @contextmanager
    def experiment_position(self, name, detail):
        if getattr(self._local, "experiment_position", None) is not None:
            raise ValueError("e1c_nested_position")
        self._local.experiment_position = name, dict(detail)
        try:
            yield
        finally:
            self._local.experiment_position = None

    def _reserve(self, client, request, payload, state):
        self._check_authority()
        scoped = getattr(self._local, "experiment_position", None)
        if scoped is None:
            raise ValueError("e1c_registered_position_required")
        key = position_key(client.role, *scoped)
        if key not in self.registered_positions:
            raise ValueError("e1c_unregistered_position")
        scope = self._local.scope
        expected = (f"e1b-{scope['session_id']}-t{key[5]}" if key[0] == "online_fast"
                    else f"e1b-review-{scope['session_id']}")
        if scope["operation_id"] != expected:
            raise ValueError("e1c_operation_identity_changed")
        for prior in state["attempts"]:
            same = (prior["role"], prior["operation_id"], prior["session_id"]) == (client.role, scope["operation_id"], scope["session_id"])
            if (tuple(prior["experiment_position"]) == key) != same:
                raise ValueError("e1c_position_rebound")
        record = super()._reserve(client, request, payload, state)
        relative = Path("wire") / f"{len(state['attempts']) + 1:04d}.json"
        files.write_wire_body(self.campaign / relative, request.data)
        record.update(experiment_id="e1c", experiment_position=list(key), wire_request_path=relative.as_posix())
        return record

    def _record_receipt(self, client, row, data):
        choices = data.get("choices") if type(data) is dict else None
        message = choices[0].get("message") if type(choices) is list and choices and type(choices[0]) is dict else None
        content = message.get("content") if type(message) is dict else None
        if type(content) is str:
            relative = Path("visible-first") / Path(row["wire_request_path"]).name
            path = self.campaign / relative
            common.write_once(path, {"schema": "e1c_visible_first_v1", "request_id": row["request_id"],
                                     "role": row["role"], "content": content})
            row.update(visible_original_path=relative.as_posix(),
                visible_original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                visible_content_sha256=hashlib.sha256(content.encode()).hexdigest())
        return super()._record_receipt(client, row, data)


def closed_ledger(campaign, frozen, state):
    """Read-only exact persisted settings, fee reservations and retry chain."""
    root = Path(campaign).resolve()
    path = root / "allowance.sqlite3"
    if not path.is_file() or path.is_symlink() or any(Path(str(path) + s).exists() for s in ("-wal", "-shm", "-journal")):
        raise ValueError("e1c_ledger_not_closed")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = CALLS.validate_e1br_ledger(path, stage="main", registration_sha256=frozen["seal_sha256"],
        max_requests=frozen["physical_max"], reservation_policy=frozen["reservation_policy"])
    if rows != state["attempts"] or state["calls_used"] != len(rows) or any(r["status"] == "pending" for r in rows):
        raise ValueError("e1c_saved_ledger_rows_changed_or_pending")
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        settings = db.execute("SELECT max_requests,threshold,halted FROM settings WHERE id=1").fetchone()
        if settings != (frozen["physical_max"], frozen["new_allowance_usd"], None):
            raise ValueError("e1c_closed_budget_or_stop_changed")
    from decimal import Decimal
    known = sum((Decimal(str(r["usage"]["cost"])) for r in rows if "cost" in r["usage"]), Decimal(0))
    held = sum((Decimal(r["reservation"]["reserved_cost_usd"]) for r in rows if "cost" not in r["usage"]), Decimal(0))
    if (known != Decimal(state["known_cost_usd"] or "0") or held != Decimal(state["reserved_cost_usd"])
            or state["unknown_cost_requests"] != sum("cost" not in r["usage"] for r in rows)
            or state["max_requests"] != frozen["physical_max"]
            or Decimal(state["max_observed_cost_usd"]) != Decimal(frozen["new_allowance_usd"])
            or known + held > Decimal(frozen["new_allowance_usd"])):
        raise ValueError("e1c_closed_cost_totals_changed")
    for row in rows:
        wire = root / row["wire_request_path"]
        if not wire.resolve().is_relative_to(root) or wire.is_symlink() or hashlib.sha256(wire.read_bytes()).hexdigest() != row["request_sha256"]:
            raise ValueError("e1c_wire_original_changed")
        if row.get("visible_original_path"):
            content_path = root / row["visible_original_path"]
            if (not content_path.resolve().is_relative_to(root) or content_path.is_symlink()
                    or hashlib.sha256(content_path.read_bytes()).hexdigest() != row["visible_original_sha256"]):
                raise ValueError("e1c_visible_original_changed")
            first = json.loads(content_path.read_bytes())
            if (first["request_id"] != row["request_id"] or first["role"] != row["role"]
                    or hashlib.sha256(first["content"].encode()).hexdigest() != row["visible_content_sha256"]):
                raise ValueError("e1c_visible_binding_changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError("e1c_ledger_changed_during_review")
    return before
