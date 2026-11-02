"""Owner-approved exact utterance sharing, limited to a local simulation.

This store is separate from reminders, model context and person memory. The
trusted host supplies owner identity and current saved sources; HTTP role
claims are not authentication. SQLite is local plaintext, as is the care demo.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading


_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OPEN = {"draft", "approved", "reserved"}
_UNCERTAIN = {"reserved", "sending", "unknown"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("转达标识无效。")
    return value


@dataclass(frozen=True)
class SharingContext:
    """An internal host snapshot, never accepted from a client JSON body."""

    owner_id: str
    run_id: str
    session_id: str | None
    writable: bool
    sources: tuple[dict, ...]
    contacts: tuple[dict, ...]
    contacts_available: bool = True


@dataclass(frozen=True)
class LocalSharingReceipt:
    status: str
    sharing_id: str
    preview_sha256: str
    receipt_id: str


class LocalSharingAdapter:
    """No network, recipient account or model; the durable row is the receipt."""

    channel = "local_sharing_inbox"
    external = False

    def send(self, *, sharing_id, preview_sha256, message, recipient):
        # Do not retain an additional message copy or imply human reading.
        return LocalSharingReceipt("delivered_local", sharing_id, preview_sha256,
                                   "local_" + secrets.token_hex(12))


class ExplicitSharingService:
    """A send needs a fresh preview, explicit approval and a distinct send.

    Committed reserved/sending records cannot be retried, even after a crash.
    The final dispatch claim is the attempt boundary: withdrawal before this
    claim prevents dispatch; afterwards the receipt may remain unknown.
    """

    def __init__(self, path, *, owner_id, adapter=None, clock=None):
        self.owner_id = _id(owner_id)
        self.adapter = adapter if adapter is not None else LocalSharingAdapter()
        if (not isinstance(self.adapter, LocalSharingAdapter)
                or self.adapter.external is not False
                or self.adapter.channel != "local_sharing_inbox"):
            raise ValueError("具体内容转达目前只接受独立本地模拟通道。")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._epoch = secrets.token_hex(16)
        self._lock = threading.RLock()
        self.failed = False
        self._closed = False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA secure_delete=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sharing(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
              run_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sharing_commands(owner_id TEXT NOT NULL,
              request_id TEXT NOT NULL, digest TEXT NOT NULL, action TEXT NOT NULL,
              sharing_id TEXT NOT NULL, PRIMARY KEY(owner_id,request_id));
        """)
        with self._lock, self._tx():
            for row in self._rows():
                if row["status"] in _UNCERTAIN - {"unknown"}:
                    row.update(status="unknown", reason="process_interrupted_no_retry")
                    self._put(row)
                elif row["status"] in {"draft", "approved"}:
                    row.update(status="expired", reason="server_restarted")
                    self._put(row)

    def _now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("转达时钟必须包含时区。")
        return value.astimezone(timezone.utc)

    @contextmanager
    def _tx(self):
        if self.failed or self._closed:
            raise RuntimeError("sharing_store_stopped")
        try:
            self._db.execute("BEGIN IMMEDIATE")
            yield
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def _rows(self):
        return [json.loads(row[0]) for row in self._db.execute(
            "SELECT data FROM sharing WHERE owner_id=? ORDER BY rowid", (self.owner_id,))]

    def _put(self, row):
        self._db.execute("INSERT OR REPLACE INTO sharing VALUES(?,?,?,?)",
                         (row["id"], self.owner_id, row["run_id"], _json(row)))

    def _get(self, sharing_id, context):
        raw = self._db.execute("SELECT data FROM sharing WHERE id=? AND owner_id=? AND run_id=?",
                               (_id(sharing_id), self.owner_id, context.run_id)).fetchone()
        if raw is None:
            raise ValueError("转达记录不属于当前本人体验。")
        return json.loads(raw[0])

    def _authorize(self, context, actor, read_only, *, reduction=False):
        if (not isinstance(context, SharingContext) or context.owner_id != self.owner_id
                or actor != self.owner_id):
            raise PermissionError("仅本人可以选择、批准、发送或撤回具体内容。")
        if type(read_only) is not bool or read_only:
            raise PermissionError("保存回放和历史浏览只能读取转达记录。")
        if self._closed or self.failed:
            raise RuntimeError("sharing_store_stopped")
        _id(context.run_id)
        if not reduction and (context.writable is not True or not context.session_id):
            raise ValueError("请先回到本人当前可写会话。")

    def _selection(self, context, source_id, contact_id):
        sources = [row for row in context.sources if row.get("turn_id") == source_id]
        contacts = [row for row in context.contacts if row.get("id") == contact_id]
        if len(sources) != 1 or len(contacts) != 1:
            raise ValueError("原话或联系人已变化，请重新选择。")
        source, contact = sources[0], contacts[0]
        if (source.get("status") != "completed" or not isinstance(source.get("text"), str)
                or not source["text"].strip() or len(source["text"]) > 500
                or source.get("session_id") != context.session_id
                or type(source.get("turn_index")) is not int):
            raise ValueError("只能选择当前活动会话已保存完成的一整句本人原话。")
        if contact.get("elder_id") != self.owner_id or contact.get("channel") != "local":
            raise ValueError("只能选择本人的本地演示联系人。")
        selected_source = {key: source[key] for key in ("turn_id", "session_id", "turn_index", "text")}
        selected_contact = {key: contact.get(key) for key in ("id", "elder_id", "name", "relation", "channel")}
        if not isinstance(selected_contact["name"], str) or not selected_contact["name"].strip():
            raise ValueError("联系人称呼无效。")
        return selected_source, selected_contact

    def _validate_preview(self, row, context, digest):
        if (row.get("forgotten") or row["epoch"] != self._epoch or row["run_id"] != context.run_id
                or row["session_id"] != context.session_id
                or datetime.fromisoformat(row["expires_at"]) <= self._now()
                or digest != row["preview_sha256"]):
            raise ValueError("预览已失效，请重新选择并批准。")
        source, contact = self._selection(context, row["source"]["turn_id"], row["recipient"]["id"])
        if source != row["source"] or contact != row["recipient"]:
            raise ValueError("原话或收件人已变化，原批准不能沿用。")
        selection = row.get("selection")
        message = self._selected_message(source, selection)
        preview = {key: row[key] for key in ("source", "recipient", "channel", "message", "run_id", "session_id")}
        if selection is not None:
            preview["selection"] = selection
        if message != row["message"] or _sha(preview) != row["preview_sha256"]:
            raise ValueError("选定内容或预览已变化，请重新核对并批准。")

    @staticmethod
    def _selected_message(source, selection):
        """An owner-selected exact span; no extraction, paraphrase or inference."""
        if selection is None:
            return source["text"]
        if (type(selection) is not dict or set(selection) != {"start", "end", "unit"}
                or selection["unit"] != "unicode_codepoint"
                or type(selection["start"]) is not int or type(selection["end"]) is not int
                or not 0 <= selection["start"] < selection["end"] <= len(source["text"])):
            raise ValueError("请在当前原话中选定一个完整连续片段。")
        message = source["text"][selection["start"]:selection["end"]]
        if not message.strip():
            raise ValueError("选定内容不能为空。")
        return message

    def view(self, context):
        if not isinstance(context, SharingContext) or context.owner_id != self.owner_id:
            raise PermissionError("无权读取转达记录。")
        with self._lock:
            all_rows = self._rows()
            rows = [row for row in all_rows if row["run_id"] == context.run_id]
            holds = [{key: row[key] for key in ("id", "status", "attempts")}
                     for row in all_rows if row["status"] in _UNCERTAIN]
            commands = [dict(row) for row in self._db.execute(
                "SELECT request_id,action,sharing_id FROM sharing_commands WHERE owner_id=?",
                (self.owner_id,)) if any(value["id"] == row["sharing_id"] for value in rows)]
            return {"channel": self.adapter.channel, "external_enabled": False,
                    "records": deepcopy(rows), "commands": commands, "holds": holds,
                    "storage_failed": self.failed, "requires_review": bool(holds),
                    "notice": "仅本地模拟转达；回执不证明真实发送、真人阅读或模型理解。"}

    def action(self, context, action, payload, *, actor, read_only=False, current_context=None):
        self._authorize(context, actor, read_only, reduction=action == "revoke")
        allowed = {"prepare": {"source_id", "contact_id", "selection"}, "approve": {"id", "preview_sha256", "confirmed"},
                   "send": {"id", "preview_sha256"}, "revoke": {"id"}}
        if (not isinstance(action, str) or action not in allowed or not isinstance(payload, dict)
                or set(payload) - allowed[action] - {"request_id"}):
            raise ValueError("转达表单格式无效；不能提交改写内容或权限。")
        request_id = _id(payload.get("request_id"))
        digest = _sha({"action": action, "payload": payload, "run_id": context.run_id,
                       "session_id": context.session_id})
        try:
            with self._lock, self._tx():
                duplicate = self._db.execute("SELECT * FROM sharing_commands WHERE owner_id=? AND request_id=?",
                                             (self.owner_id, request_id)).fetchone()
                if duplicate:
                    if duplicate["digest"] != digest:
                        raise ValueError("请求编号已经用于其他转达操作。")
                    return {"record": self._get(duplicate["sharing_id"], context), "duplicate": True}
                if action == "prepare":
                    if any(row["status"] in _UNCERTAIN for row in self._rows()):
                        raise ValueError("已有转达结果未知；禁止换编号重新发送。")
                    source, recipient = self._selection(context, payload.get("source_id"), payload.get("contact_id"))
                    selection = payload.get("selection")
                    message = self._selected_message(source, selection)
                    for row in self._rows():
                        if row["run_id"] == context.run_id and row["status"] in {"draft", "approved"}:
                            row.update(status="superseded", reason="new_selection")
                            self._put(row)
                    preview = {"source": source, "recipient": recipient, "channel": self.adapter.channel,
                               "message": message, "run_id": context.run_id,
                               "session_id": context.session_id}
                    if selection is not None:
                        preview["selection"] = deepcopy(selection)
                    row = {"id": "share_" + secrets.token_hex(12), "owner_id": self.owner_id,
                           **preview, "preview_sha256": _sha(preview), "epoch": self._epoch,
                           "status": "draft", "created_at": self._now().isoformat(),
                           "expires_at": (self._now() + timedelta(minutes=5)).isoformat(),
                           "attempts": 0, "receipt": None}
                else:
                    row = self._get(payload.get("id"), context)
                    if action == "revoke":
                        row["withdrawn_at"] = self._now().isoformat()
                        if row["status"] in _OPEN:
                            row.update(status="revoked", reason="owner_withdrawal_before_attempt")
                        elif row["status"] in {"sending", "unknown"}:
                            row["reason"] = "withdrawn_after_attempt_result_requires_review"
                    else:
                        self._validate_preview(row, context, payload.get("preview_sha256"))
                        if action == "approve":
                            if row["status"] != "draft" or payload.get("confirmed") is not True:
                                raise ValueError("必须先核对原话、收件人和本地通道，再明确批准。")
                            row.update(status="approved", approved_at=self._now().isoformat())
                        elif row["status"] != "approved" or row.get("withdrawn_at"):
                            raise ValueError("该转达未获当前本人批准，或者已经尝试；不会重发。")
                        else:
                            row.update(status="reserved", send_request_id=request_id)
                self._put(row)
                self._db.execute("INSERT INTO sharing_commands VALUES(?,?,?,?,?)",
                                 (self.owner_id, request_id, digest, action, row["id"]))
            if action == "send":
                return {"record": self._dispatch(row["id"], current_context or (lambda: context)),
                        "duplicate": False}
            return {"record": deepcopy(row), "duplicate": False}
        except (sqlite3.Error, OSError, RuntimeError):
            self.failed = True
            raise RuntimeError("sharing_store_failed_no_retry") from None

    def _dispatch(self, sharing_id, current_context):
        context = current_context()
        with self._lock, self._tx():
            raw = self._db.execute("SELECT data FROM sharing WHERE id=? AND owner_id=?",
                                   (_id(sharing_id), self.owner_id)).fetchone()
            if raw is None:
                raise ValueError("转达记录已失效，未发送。")
            row = json.loads(raw[0])
            if row["status"] != "reserved" or row.get("withdrawn_at"):
                return deepcopy(row)
            try:
                self._authorize(context, self.owner_id, False)
                self._validate_preview(row, context, row["preview_sha256"])
            except (ValueError, PermissionError):
                row.update(status="revoked", reason="current_source_or_permission_changed_before_attempt")
                self._put(row)
                return deepcopy(row)
            row.update(status="sending", attempts=1, attempted_at=self._now().isoformat())
            self._put(row)
        try:
            receipt = self.adapter.send(sharing_id=row["id"], preview_sha256=row["preview_sha256"],
                                        message=row["message"], recipient=deepcopy(row["recipient"]))
            valid = (isinstance(receipt, LocalSharingReceipt)
                     and receipt.status in {"accepted", "delivered_local"}
                     and receipt.sharing_id == row["id"]
                     and receipt.preview_sha256 == row["preview_sha256"]
                     and isinstance(receipt.receipt_id, str)
                     and re.fullmatch(r"local_[a-f0-9]{24}", receipt.receipt_id))
            status = receipt.status if valid else "unknown"
        except Exception:
            status, receipt = "unknown", None
        with self._lock, self._tx():
            current = self._get(sharing_id, context)
            if current["status"] == "sending":
                current.update(status=status, completed_at=self._now().isoformat(),
                               receipt=({"status": status, "id": receipt.receipt_id,
                                         "source": "local_simulation", "preview_sha256": row["preview_sha256"]}
                                        if status != "unknown" else None))
                if status == "unknown":
                    current["reason"] = "adapter_or_receipt_unknown_no_retry"
                self._put(current)
            return deepcopy(current)

    def forget(self):
        """Erase local bodies and recipients, retain opaque no-retry tombstones."""
        with self._lock, self._tx():
            for row in self._rows():
                if row["status"] in _OPEN:
                    row.update(status="revoked", reason="owner_forgot_run")
                row.pop("message", None)
                row.pop("source", None)
                row.pop("recipient", None)
                row.pop("receipt", None)
                row["forgotten"] = True
                self._put(row)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._db.close()
