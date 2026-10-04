"""Small, fail-closed T7 campaign ledger. No transport or environment access.

The caller must keep one fixed ledger path for the experiment, independent of run
and output IDs. An unfinished durable claim is an ambiguous attempt, never a
retry ticket. Monetary values in public receipts are decimal strings.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
import uuid


SCHEMA_VERSION = "aria_t7_campaign_ledger_v1"
TOTAL_LIMIT = Decimal("5")
PAUSE_LIMIT = Decimal("2")
STAGES = {
    "e1": {
        "limit": Decimal("0.5"), "requests": 204,
        "models": {"qwen/qwen3.7-flash", "qwen/qwen3.7-flash-20260727"},
        "provider": "alibaba", "output_tokens": 1024,
    },
    "judge": {
        "limit": Decimal("0.3"), "requests": 35,
        "models": {"anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-20260630"},
        "provider": "anthropic", "output_tokens": 512,
    },
}
ERROR_CODES = frozenset({
    "transport_error", "http_error", "rate_limited", "invalid_response",
    "invalid_schema", "truncated_output", "receipt_missing",
    "token_bound_exceeded", "request_changed", "source_changed",
    "metadata_changed", "price_changed", "result_save_failed",
    "ledger_save_failed", "unknown_result", "unknown_cost", "manual_stop",
    "external_error", "unfinished_claim", "already_started", "out_of_order",
    "inventory_changed", "model_changed", "provider_changed", "invalid_receipt",
    "stage_budget_exceeded", "campaign_budget_exceeded", "checkpoint_reached",
    "reservation_exceeded", "reservation_insufficient",
})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,179}\Z")


class LedgerError(RuntimeError):
    """A safe code, deliberately excluding provider/error text."""

    def __init__(self, code: str, *, _persist_stop: bool = False):
        super().__init__(code)
        self.code = code
        self._persist_stop = _persist_stop


def _money(value: Any, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise LedgerError("invalid_amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise LedgerError("invalid_amount") from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise LedgerError("invalid_amount")
    # Reject pathological numeric inputs before serializing or doing arithmetic.
    if result > Decimal("1000000") or not -18 <= result.as_tuple().exponent <= 6:
        raise LedgerError("invalid_amount")
    return result


def _amount(value: Decimal) -> str:
    return "0" if value == 0 else format(value.normalize(), "f")


def _safe_error(value: Any) -> str:
    return value if isinstance(value, str) and value in ERROR_CODES else "external_error"


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _inventory_hash(items: list[dict], limit: str, maximum: int) -> str:
    payload = {"items": items, "limit_usd": limit, "max_requests": maximum}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


class CampaignLedger:
    """One immutable campaign, all allowed stages, at most one outstanding call."""

    def __init__(self, path: str | Path, *, campaign_id: str, freeze_seal: str):
        if not _valid_id(campaign_id) or not _valid_hash(freeze_seal):
            raise LedgerError("invalid_campaign")
        self.path = Path(path).resolve()
        self.campaign_id = campaign_id
        self.freeze_seal = freeze_seal
        self._poisoned = False
        self._owned: dict[tuple[str, str], str] = {}
        self._stop_path = self.path.with_name(self.path.name + ".stopped")
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        # Opening or inspecting an existing journal never resolves or alters it.
        with self._read() as conn:
            self._validate(conn)

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)
            conn.execute("PRAGMA query_only=ON")
        else:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.execute("PRAGMA synchronous=FULL")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""CREATE TABLE IF NOT EXISTS campaign (
                    id INTEGER PRIMARY KEY CHECK(id=1), schema_version TEXT NOT NULL,
                    campaign_id TEXT NOT NULL, freeze_seal TEXT NOT NULL,
                    total_limit TEXT NOT NULL, pause_limit TEXT NOT NULL,
                    stopped INTEGER NOT NULL CHECK(stopped IN (0,1)), stop_reason TEXT
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS stages (
                    name TEXT PRIMARY KEY, inventory_hash TEXT NOT NULL,
                    limit_usd TEXT NOT NULL, max_requests INTEGER NOT NULL,
                    actual_model TEXT, provider TEXT
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS items (
                    stage TEXT NOT NULL REFERENCES stages(name), id TEXT NOT NULL,
                    position INTEGER NOT NULL, request_sha256 TEXT NOT NULL,
                    reserve_usd TEXT NOT NULL, status TEXT NOT NULL,
                    claim_token TEXT, cost_usd TEXT, prompt_tokens INTEGER,
                    completion_tokens INTEGER, actual_model TEXT, provider TEXT,
                    response_sha256 TEXT, error TEXT,
                    PRIMARY KEY(stage,id), UNIQUE(stage,position)
                )""")
                conn.execute("""INSERT OR IGNORE INTO campaign
                    VALUES (1,?,?,?,?,?,0,NULL)""", (
                    SCHEMA_VERSION, self.campaign_id, self.freeze_seal,
                    _amount(TOTAL_LIMIT), _amount(PAUSE_LIMIT),
                ))
        except (sqlite3.Error, OSError):
            self._poison()
            raise LedgerError("ledger_save_failed") from None
        finally:
            if "conn" in locals():
                conn.close()

    def _poison(self) -> None:
        self._poisoned = True
        # A second durable marker protects a failed DB update/commit. If the
        # medium itself fails, the already committed claim still blocks retries.
        try:
            fd = os.open(self._stop_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(b"ledger_save_failed\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = self._connect(readonly=True)
            yield conn
        except sqlite3.Error:
            raise LedgerError("ledger_read_failed") from None
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if self._poisoned or self._stop_path.exists():
            raise LedgerError("ledger_save_failed")
        conn = None
        try:
            conn = self._connect()
            conn.execute("BEGIN IMMEDIATE")
            self._validate(conn)
            try:
                yield conn
            except LedgerError as exc:
                if exc._persist_stop:
                    conn.commit()
                else:
                    conn.rollback()
                raise
            conn.commit()
        except (sqlite3.Error, OSError):
            self._poison()
            raise LedgerError("ledger_save_failed") from None
        finally:
            if conn is not None:
                conn.close()

    def _validate(self, conn: sqlite3.Connection) -> None:
        expected_columns = {
            "campaign": ("id", "schema_version", "campaign_id", "freeze_seal", "total_limit", "pause_limit", "stopped", "stop_reason"),
            "stages": ("name", "inventory_hash", "limit_usd", "max_requests", "actual_model", "provider"),
            "items": ("stage", "id", "position", "request_sha256", "reserve_usd", "status", "claim_token", "cost_usd", "prompt_tokens", "completion_tokens", "actual_model", "provider", "response_sha256", "error"),
        }
        for table, columns in expected_columns.items():
            actual = tuple(row["name"] for row in conn.execute(f"PRAGMA table_info({table})"))
            if actual != columns:
                raise LedgerError("invalid_ledger_schema")
        rows = conn.execute("SELECT * FROM campaign").fetchall()
        if len(rows) != 1:
            raise LedgerError("invalid_ledger")
        row = rows[0]
        if (
            row["schema_version"] != SCHEMA_VERSION
            or row["campaign_id"] != self.campaign_id
            or row["freeze_seal"] != self.freeze_seal
            or row["total_limit"] != _amount(TOTAL_LIMIT)
            or row["pause_limit"] != _amount(PAUSE_LIMIT)
            or type(row["stopped"]) is not int or row["stopped"] not in (0, 1)
            or (row["stopped"] and row["stop_reason"] not in ERROR_CODES)
            or (not row["stopped"] and row["stop_reason"] is not None)
        ):
            raise LedgerError("campaign_mismatch")
        for stage in conn.execute("SELECT * FROM stages").fetchall():
            name = stage["name"]
            if name not in STAGES:
                raise LedgerError("invalid_ledger")
            spec = STAGES[name]
            limit = _money(stage["limit_usd"], positive=True)
            maximum = stage["max_requests"]
            if limit > spec["limit"] or type(maximum) is not int or not 1 <= maximum <= spec["requests"]:
                raise LedgerError("invalid_ledger")
            if (stage["actual_model"] is None) != (stage["provider"] is None):
                raise LedgerError("invalid_ledger")
            if stage["actual_model"] is not None and (
                stage["actual_model"] not in spec["models"] or stage["provider"] != spec["provider"]
            ):
                raise LedgerError("invalid_ledger")
            items = conn.execute("SELECT * FROM items WHERE stage=? ORDER BY position", (name,)).fetchall()
            if not 1 <= len(items) <= maximum:
                raise LedgerError("invalid_ledger")
            inventory = []
            for index, item in enumerate(items):
                reserve = _money(item["reserve_usd"], positive=True)
                if item["position"] != index or not _valid_id(item["id"]) or not _valid_hash(item["request_sha256"]):
                    raise LedgerError("invalid_ledger")
                if item["status"] not in {"not_started", "started", "completed", "failed"}:
                    raise LedgerError("invalid_ledger")
                if item["status"] == "not_started" and any(item[key] is not None for key in (
                    "claim_token", "cost_usd", "prompt_tokens", "completion_tokens",
                    "actual_model", "provider", "response_sha256", "error",
                )):
                    raise LedgerError("invalid_ledger")
                if item["status"] != "not_started" and not _valid_hash(item["claim_token"]):
                    raise LedgerError("invalid_ledger")
                if item["status"] == "started" and any(item[key] is not None for key in (
                    "cost_usd", "prompt_tokens", "completion_tokens", "actual_model",
                    "provider", "response_sha256", "error",
                )):
                    raise LedgerError("invalid_ledger")
                if item["status"] == "failed" and item["error"] is None:
                    raise LedgerError("invalid_ledger")
                if item["cost_usd"] is not None:
                    _money(item["cost_usd"])
                if item["error"] is not None and item["error"] not in ERROR_CODES:
                    raise LedgerError("invalid_ledger")
                if item["status"] == "completed" and (
                    item["error"] is not None or item["cost_usd"] is None
                    or not _valid_hash(item["response_sha256"])
                    or item["actual_model"] != stage["actual_model"]
                    or item["provider"] != stage["provider"]
                    or any(type(item[key]) is not int or item[key] < 0 for key in ("prompt_tokens", "completion_tokens"))
                ):
                    raise LedgerError("invalid_ledger")
                inventory.append({"id": item["id"], "request_sha256": item["request_sha256"], "reserve_usd": _amount(reserve)})
            if _inventory_hash(inventory, _amount(limit), maximum) != stage["inventory_hash"]:
                raise LedgerError("invalid_ledger")
        if conn.execute("SELECT COUNT(*) FROM items WHERE stage NOT IN (SELECT name FROM stages)").fetchone()[0]:
            raise LedgerError("invalid_ledger")

    @staticmethod
    def _halt(conn: sqlite3.Connection, reason: str) -> None:
        conn.execute("UPDATE campaign SET stopped=1,stop_reason=? WHERE stopped=0", (reason,))
        raise LedgerError(reason, _persist_stop=True)

    @staticmethod
    def _require_active(conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT stopped,stop_reason FROM campaign").fetchone()
        if row["stopped"]:
            raise LedgerError(row["stop_reason"])

    @staticmethod
    def _costs(conn: sqlite3.Connection, stage: str | None = None) -> tuple[Decimal, Decimal]:
        query = "SELECT cost_usd,reserve_usd,status FROM items"
        params: tuple = ()
        if stage is not None:
            query += " WHERE stage=?"
            params = (stage,)
        known, remaining = Decimal(0), Decimal(0)
        for item in conn.execute(query, params):
            if item["cost_usd"] is not None:
                known += _money(item["cost_usd"])
            if item["status"] in {"not_started", "started"}:
                remaining += _money(item["reserve_usd"])
        return known, remaining

    def _check_budget(self, conn: sqlite3.Connection) -> None:
        known, remaining = self._costs(conn)
        if known >= TOTAL_LIMIT or known + remaining > TOTAL_LIMIT:
            self._halt(conn, "campaign_budget_exceeded")
        if known >= PAUSE_LIMIT or known + remaining > PAUSE_LIMIT:
            self._halt(conn, "checkpoint_reached")
        for stage in conn.execute("SELECT name,limit_usd FROM stages").fetchall():
            cost, reserve = self._costs(conn, stage["name"])
            limit = _money(stage["limit_usd"])
            if cost >= limit or cost + reserve > limit:
                self._halt(conn, "stage_budget_exceeded")

    def register_stage(self, stage: str, items: list[dict], *, limit_usd: Any, max_requests: int) -> dict:
        if not isinstance(stage, str) or stage not in STAGES:
            raise LedgerError("stage_not_authorized")
        spec = STAGES[stage]
        limit = _money(limit_usd, positive=True)
        if limit > spec["limit"] or type(max_requests) is not int or not 1 <= max_requests <= spec["requests"]:
            raise LedgerError("invalid_stage_limits")
        if type(items) is not list or not 1 <= len(items) <= max_requests:
            raise LedgerError("invalid_inventory")
        inventory, ids = [], set()
        for item in items:
            if type(item) is not dict or set(item) != {"id", "request_sha256", "reserve_usd"}:
                raise LedgerError("invalid_inventory")
            if not _valid_id(item["id"]) or not _valid_hash(item["request_sha256"]) or item["id"] in ids:
                raise LedgerError("invalid_inventory")
            reserve = _money(item["reserve_usd"], positive=True)
            ids.add(item["id"])
            inventory.append({"id": item["id"], "request_sha256": item["request_sha256"], "reserve_usd": _amount(reserve)})
        fingerprint = _inventory_hash(inventory, _amount(limit), max_requests)
        with self._write() as conn:
            self._require_active(conn)
            previous = conn.execute("SELECT inventory_hash FROM stages WHERE name=?", (stage,)).fetchone()
            if previous is not None:
                if previous["inventory_hash"] != fingerprint:
                    self._halt(conn, "inventory_changed")
                return {"stage": stage, "inventory_sha256": fingerprint, "items": len(items)}
            conn.execute("INSERT INTO stages VALUES (?,?,?,?,NULL,NULL)", (stage, fingerprint, _amount(limit), max_requests))
            conn.executemany("""INSERT INTO items
                (stage,id,position,request_sha256,reserve_usd,status)
                VALUES (?,?,?,?,?,'not_started')""", [
                (stage, item["id"], index, item["request_sha256"], item["reserve_usd"])
                for index, item in enumerate(inventory)
            ])
            if conn.execute("SELECT COUNT(*) FROM items WHERE status='started'").fetchone()[0]:
                self._halt(conn, "unfinished_claim")
            self._check_budget(conn)
        return {"stage": stage, "inventory_sha256": fingerprint, "items": len(items)}

    def claim(self, stage: str, item_id: str) -> dict:
        if not isinstance(stage, str) or stage not in STAGES or not _valid_id(item_id):
            raise LedgerError("invalid_item")
        token = uuid.uuid4().hex + uuid.uuid4().hex
        with self._write() as conn:
            self._require_active(conn)
            if conn.execute("SELECT COUNT(*) FROM items WHERE status='started'").fetchone()[0]:
                self._halt(conn, "unfinished_claim")
            item = conn.execute("SELECT * FROM items WHERE stage=? AND id=?", (stage, item_id)).fetchone()
            if item is None:
                raise LedgerError("unregistered_item")
            if item["status"] != "not_started":
                self._halt(conn, "already_started")
            first = conn.execute("SELECT id FROM items WHERE stage=? AND status='not_started' ORDER BY position LIMIT 1", (stage,)).fetchone()
            if first["id"] != item_id:
                self._halt(conn, "out_of_order")
            if stage == "judge":
                e1 = conn.execute("SELECT COUNT(*) FROM items WHERE stage='e1'").fetchone()[0]
                unfinished = conn.execute("SELECT COUNT(*) FROM items WHERE stage='e1' AND status!='completed'").fetchone()[0]
                if not e1 or unfinished:
                    self._halt(conn, "unfinished_claim")
            self._check_budget(conn)
            conn.execute("UPDATE items SET status='started',claim_token=? WHERE stage=? AND id=?", (token, stage, item_id))
            result = {
                "stage": stage, "id": item_id, "request_sha256": item["request_sha256"],
                "reserve_usd": item["reserve_usd"], "claim_token": token,
            }
        self._owned[(stage, item_id)] = token
        return result

    def finish(
        self, stage: str, item_id: str, *, cost_usd: Any,
        prompt_tokens: Any, completion_tokens: Any, actual_model: Any,
        provider: Any, response_sha256: Any, error: str | None = None,
    ) -> dict:
        if not isinstance(stage, str) or stage not in STAGES or not _valid_id(item_id):
            raise LedgerError("invalid_item")
        problem = _safe_error(error) if error is not None else None
        try:
            cost = _money(cost_usd)
        except LedgerError:
            cost = None
            problem = problem or "unknown_cost"
        valid_tokens = all(type(value) is int and 0 <= value <= 2**63 - 1 for value in (prompt_tokens, completion_tokens))
        if not valid_tokens:
            problem = problem or "invalid_receipt"
        elif completion_tokens > STAGES[stage]["output_tokens"] or (stage == "e1" and prompt_tokens > 31999):
            problem = problem or "token_bound_exceeded"
        if not isinstance(actual_model, str) or actual_model not in STAGES[stage]["models"]:
            problem = problem or "model_changed"
        if not isinstance(provider, str) or provider != STAGES[stage]["provider"]:
            problem = problem or "provider_changed"
        if not _valid_hash(response_sha256):
            problem = problem or "invalid_receipt"
        with self._write() as conn:
            item = conn.execute("SELECT * FROM items WHERE stage=? AND id=?", (stage, item_id)).fetchone()
            if item is None or item["status"] != "started" or self._owned.get((stage, item_id)) != item["claim_token"]:
                self._halt(conn, "unknown_result")
            stage_row = conn.execute("SELECT * FROM stages WHERE name=?", (stage,)).fetchone()
            if stage_row["actual_model"] is not None and actual_model != stage_row["actual_model"]:
                problem = problem or "model_changed"
            if stage_row["provider"] is not None and provider != stage_row["provider"]:
                problem = problem or "provider_changed"
            if cost is not None and cost > _money(item["reserve_usd"]):
                problem = problem or "reservation_exceeded"
            # Never store arbitrary provider/error strings in the journal.
            stored_model = actual_model if isinstance(actual_model, str) and actual_model in STAGES[stage]["models"] else None
            stored_provider = provider if isinstance(provider, str) and provider == STAGES[stage]["provider"] else None
            conn.execute("""UPDATE items SET status=?,cost_usd=?,prompt_tokens=?,
                completion_tokens=?,actual_model=?,provider=?,response_sha256=?,error=?
                WHERE stage=? AND id=?""", (
                "failed" if problem else "completed", _amount(cost) if cost is not None else None,
                prompt_tokens if valid_tokens else None, completion_tokens if valid_tokens else None,
                stored_model, stored_provider, response_sha256 if _valid_hash(response_sha256) else None,
                problem, stage, item_id,
            ))
            if not problem and stage_row["actual_model"] is None:
                conn.execute("UPDATE stages SET actual_model=?,provider=? WHERE name=?", (actual_model, provider, stage))
            # Keep the observed cost even when it overruns every budget.
            self._check_budget(conn)
            if problem:
                self._halt(conn, problem)
        self._owned.pop((stage, item_id), None)
        return {"stage": stage, "id": item_id, "status": "completed", "cost_usd": _amount(cost)}

    def stop(self, stage: str, reason: str) -> None:
        if not isinstance(stage, str) or stage not in STAGES:
            raise LedgerError("stage_not_authorized")
        with self._write() as conn:
            conn.execute("UPDATE campaign SET stopped=1,stop_reason=? WHERE stopped=0", (_safe_error(reason),))

    def summary(self) -> dict:
        with self._read() as conn:
            self._validate(conn)
            campaign = conn.execute("SELECT * FROM campaign").fetchone()
            known, remaining = self._costs(conn)
            outstanding = conn.execute("SELECT COUNT(*) FROM items WHERE status='started'").fetchone()[0]
            unknown = conn.execute("SELECT COUNT(*) FROM items WHERE status!='not_started' AND cost_usd IS NULL").fetchone()[0]
            poisoned = self._poisoned or self._stop_path.exists()
            result = {
                "schema": SCHEMA_VERSION, "campaign_id": self.campaign_id,
                "freeze_seal": self.freeze_seal,
                "stopped": bool(campaign["stopped"] or outstanding or poisoned),
                "stop_reason": campaign["stop_reason"] or ("ledger_save_failed" if poisoned else "unfinished_claim" if outstanding else None),
                "known_cost_usd": _amount(known), "remaining_reserve_usd": _amount(remaining),
                "unknown_cost_count": unknown, "outstanding_count": outstanding,
                "total_limit_usd": _amount(TOTAL_LIMIT), "pause_limit_usd": _amount(PAUSE_LIMIT),
                "stages": {},
            }
            for stage in conn.execute("SELECT * FROM stages ORDER BY name").fetchall():
                items = [dict(row) for row in conn.execute("SELECT * FROM items WHERE stage=? ORDER BY position", (stage["name"],))]
                # Claims are private tokens, not useful for a summary or export.
                for item in items:
                    item.pop("claim_token")
                cost, reserve = self._costs(conn, stage["name"])
                result["stages"][stage["name"]] = {
                    "inventory_sha256": stage["inventory_hash"], "limit_usd": stage["limit_usd"],
                    "max_requests": stage["max_requests"], "actual_model": stage["actual_model"],
                    "provider": stage["provider"], "known_cost_usd": _amount(cost),
                    "remaining_reserve_usd": _amount(reserve),
                    "counts": {status: sum(item["status"] == status for item in items) for status in ("not_started", "started", "completed", "failed")},
                    "items": items,
                }
            return result
