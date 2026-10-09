"""Small deterministic care service for the fictional, local companion demo.

SQLite owns commands, schedule instances and the notification outbox. No model,
environment lookup or background thread is created here. The host runs tick()
independently of chat; views never tick or write. External delivery requires an
explicit adapter and permission plus a verified contact and real consent.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading

from .care import LocalInboxNotificationAdapter

UTC = timezone.utc
TIMEZONES = {"UTC": UTC, "Asia/Shanghai": timezone(timedelta(hours=8))}
_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_TIME = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_KINDS = ("reminders", "occurrences", "contacts", "consents", "tasks", "deliveries")
_ELDER_ACTIONS = {
    "reminder.create", "reminder.cancel", "occurrence.ack", "occurrence.decline",
    "occurrence.snooze", "occurrence.displayed", "contact.create", "consent.grant", "consent.revoke",
}
_ELDER_RESPONSES = {"occurrence.ack", "occurrence.decline", "occurrence.snooze", "occurrence.displayed"}
_CAREGIVER_ACTIONS = (_ELDER_ACTIONS - _ELDER_RESPONSES) | {
    "contact.verify", "task.follow_up", "task.close", "receipt.record",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stamp(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("必须提供带时区的时间")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _parse(value):
    try:
        parsed = datetime.fromisoformat(value)
        _stamp(parsed)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        raise ValueError("时间必须是包含时区的ISO日期时间") from None


def _text(value, name, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name}不能为空且不能超过{limit}字")
    return value.strip()


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("标识无效")
    return value


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}必须是{minimum}至{maximum}的整数")
    return value


def _stable(prefix, *values):
    return prefix + "_" + hashlib.sha256(_json(values).encode()).hexdigest()[:32]


class MinimalCareService:
    """One management role with explicit assignments; one notification adapter.

    elder_action is a trusted host integration point, not an authentication API.
    HTTP callers must bind it to the fixed fictional identity or authenticate an
    elder separately. This local prototype's SQLite data is not encrypted.
    """

    def __init__(self, path: Path, notifier=None, allow_external=False, clock=None):
        if type(allow_external) is not bool:
            raise TypeError("allow_external must be an explicit boolean")
        self.notifier = notifier if notifier is not None else LocalInboxNotificationAdapter()
        self.external = not isinstance(self.notifier, LocalInboxNotificationAdapter)
        if self.external and not allow_external:
            raise ValueError("外部通知通道必须显式允许")
        self.allow_external = allow_external
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._closed = False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS elders(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT NOT NULL,
              password_hash TEXT NOT NULL, failures INTEGER NOT NULL DEFAULT 0, locked_until TEXT);
            CREATE TABLE IF NOT EXISTS assignments(username TEXT NOT NULL REFERENCES users(username),
              elder_id TEXT NOT NULL REFERENCES elders(id), PRIMARY KEY(username,elder_id));
            CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,
              username TEXT NOT NULL REFERENCES users(username), expires_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL, id TEXT NOT NULL,
              elder_id TEXT NOT NULL REFERENCES elders(id), data TEXT NOT NULL, PRIMARY KEY(kind,id));
            CREATE INDEX IF NOT EXISTS records_elder ON records(elder_id,kind);
            CREATE TABLE IF NOT EXISTS commands(actor TEXT NOT NULL, elder_id TEXT NOT NULL,
              request_id TEXT NOT NULL, digest TEXT NOT NULL, result TEXT NOT NULL,
              PRIMARY KEY(actor,elder_id,request_id));
        """)
        # A send may have reached the gateway before the previous process died.
        # Never turn this state back into queued, even for an idempotent gateway.
        with self._lock:
            inflight = [row for row in self._rows("deliveries") if row["status"] == "sending"]
            if inflight:
                with self._tx():
                    for row in inflight:
                        row.update(status="unknown", error="进程中断，发送结果未知；禁止自动重发")
                        self._put("deliveries", row)
                        for task in self._rows("tasks", row["elder_id"]):
                            if task["occurrence_id"] == row["occurrence_id"] and task["status"] != "closed":
                                task["delivery_attention"] = "通知发送结果未知；请核查回执，禁止自动重发。"
                                self._put("tasks", task)
            # Old offered_local meant a database row was created, not that an
            # elder page displayed it. Keep that historical state as evidence,
            # but do not continue its grace window or unsent notifications.
            legacy = [row for row in self._rows("occurrences")
                      if row["status"] in {"pending", "unconfirmed", "snoozed"}
                      and row.get("display_contract") != "elder_page_receipt_v1"]
            if legacy:
                with self._tx():
                    for row in self._rows("occurrences"):
                        if (row["status"] not in {"pending", "unconfirmed", "snoozed"}
                                or row.get("display_contract") == "elder_page_receipt_v1"):
                            continue
                        row["legacy_visibility"] = {key: row.get(key) for key in
                            ("status", "delivery_status", "offered_at", "deadline_at", "generation")}
                        row["generation"] = row.get("generation", 0) + 1
                        row.update(display_contract="elder_page_receipt_v1", delivery_status="awaiting_display",
                                   displayed_at=None, displayed_by=None, displayed_generation=None,
                                   display_source=None, offered_at=None, deadline_at=None,
                                   display_history=[])
                        if row["status"] == "unconfirmed":
                            row["status"] = "pending"
                        self._put("occurrences", row)
                        self._cancel_queued(row["elder_id"], occurrence_id=row["id"],
                            reason="旧生成记录没有本人页面展示回执；禁止发送。")
                        for task in self._rows("tasks", row["elder_id"]):
                            if task["occurrence_id"] == row["id"]:
                                task.update(visibility_evidence="legacy_unverified",
                                    delivery_attention="旧记录仅证明提醒已生成，不能证明本人页面展示；未发送通知已停止。")
                                self._put("tasks", task)

    @contextmanager
    def _tx(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _now(self, value=None):
        return _parse(_stamp(self.clock() if value is None else value))

    def _rows(self, kind, elder_id=None):
        sql = "SELECT data FROM records WHERE kind=?"
        args = [kind]
        if elder_id is not None:
            sql += " AND elder_id=?"
            args.append(elder_id)
        return [json.loads(row[0]) for row in self._db.execute(sql + " ORDER BY rowid", args)]

    def _get(self, kind, row_id, elder_id):
        row = self._db.execute("SELECT data FROM records WHERE kind=? AND id=? AND elder_id=?",
                               (kind, row_id, elder_id)).fetchone()
        if row is None:
            raise ValueError("记录不存在或不在负责范围内")
        return json.loads(row[0])

    def _put(self, kind, row):
        self._db.execute("INSERT INTO records(kind,id,elder_id,data) VALUES(?,?,?,?) "
                         "ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data",
                         (kind, row["id"], row["elder_id"], _json(row)))

    def _elder_exists(self, elder_id):
        return self._db.execute("SELECT 1 FROM elders WHERE id=?", (elder_id,)).fetchone() is not None

    def bootstrap(self, username, password, elder_ids):
        """Operator-only provisioning, never expose as an unauthenticated route."""
        username = _identifier(username)
        if not isinstance(password, str) or not 12 <= len(password) <= 256:
            raise ValueError("密码长度必须为12至256字符")
        if not isinstance(elder_ids, (list, tuple)) or not elder_ids:
            raise ValueError("必须明确负责老人范围")
        ids = list(dict.fromkeys(_identifier(value) for value in elder_ids))
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()
        with self._lock, self._tx():
            if self._db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                raise ValueError("账号已存在；不会覆盖密码或负责范围")
            self._db.execute("INSERT INTO users(username,salt,password_hash) VALUES(?,?,?)",
                             (username, salt, digest))
            for elder_id in ids:
                self._db.execute("INSERT OR IGNORE INTO elders(id) VALUES(?)", (elder_id,))
                self._db.execute("INSERT INTO assignments(username,elder_id) VALUES(?,?)", (username, elder_id))
        return {"ok": True, "username": username, "elders": [{"id": value} for value in ids]}

    def login(self, username, password):
        now = self._now()
        with self._lock:
            row = self._db.execute("SELECT * FROM users WHERE username=?", (str(username),)).fetchone()
            salt = bytes.fromhex(row["salt"]) if row else b"minimal-care-dummy"
            value = password if isinstance(password, str) and len(password) <= 256 else ""
            digest = hashlib.pbkdf2_hmac("sha256", value.encode(), salt, 600_000).hex()
            valid = bool(row and hmac.compare_digest(digest, row["password_hash"]))
            locked = bool(row and row["locked_until"] and _parse(row["locked_until"]) > now)
            if not valid or locked:
                if row and not locked:
                    failures = row["failures"] + 1
                    with self._tx():
                        self._db.execute("UPDATE users SET failures=?,locked_until=? WHERE username=?",
                                         (failures, _stamp(now + timedelta(minutes=15)) if failures >= 5 else None,
                                          row["username"]))
                raise PermissionError("登录失败，请核对账号密码或稍后再试")
            token = secrets.token_urlsafe(32)
            with self._tx():
                self._db.execute("UPDATE users SET failures=0,locked_until=NULL WHERE username=?", (username,))
                self._db.execute("INSERT INTO sessions(token_hash,username,expires_at) VALUES(?,?,?)",
                                 (hashlib.sha256(token.encode()).hexdigest(), username, _stamp(now + timedelta(hours=8))))
            return token

    def _authorize(self, token, elder_id=None):
        if not isinstance(token, str):
            raise PermissionError("请先登录护理后台")
        row = self._db.execute("SELECT username,expires_at FROM sessions WHERE token_hash=?",
                               (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if row is None or _parse(row["expires_at"]) <= self._now():
            raise PermissionError("登录已失效，请重新登录")
        ids = [r[0] for r in self._db.execute("SELECT elder_id FROM assignments WHERE username=? ORDER BY elder_id",
                                            (row["username"],))]
        if elder_id is not None and elder_id not in ids:
            raise PermissionError("无权访问该老人")
        return row["username"], ids

    def logout(self, token):
        with self._lock, self._tx():
            if isinstance(token, str):
                self._db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
        return {"ok": True}

    def _view(self, ids):
        return {"ok": True, "configured": bool(ids), "elders": [{"id": value} for value in ids],
                "channel": self.notifier.channel, "external_enabled": self.external,
                "timezones": list(TIMEZONES), "notice": "未确认只表示没有收到确认，不表示没吃药或发生危险。",
                **{kind: [row for elder in ids for row in self._rows(kind, elder)] for kind in _KINDS}}

    def view(self, token, elder_id=None):
        with self._lock:
            _username, ids = self._authorize(token, elder_id)
            return self._view([elder_id] if elder_id is not None else ids)

    def elder_view(self, elder_id):
        with self._lock:
            _identifier(elder_id)
            return self._view([elder_id] if self._elder_exists(elder_id) else [])

    def mutate(self, token, elder_id, action, payload):
        with self._lock, self._tx():
            username, _ids = self._authorize(token, elder_id)
            return self._command("caregiver:" + username, elder_id, action, payload, caregiver=True)

    def elder_action(self, elder_id, action, payload):
        with self._lock, self._tx():
            _identifier(elder_id)
            if not self._elder_exists(elder_id):
                raise PermissionError("护理尚未配置，请由护理员明确初始化负责范围")
            return self._command("elder:" + elder_id, elder_id, action, payload, caregiver=False)

    def _command(self, actor, elder_id, action, payload, *, caregiver):
        # Check the current actor's authority before either applying a command
        # or replaying a result saved under an older permissions contract.
        if type(action) is not str:
            raise ValueError("护理操作名称必须是字符串")
        if caregiver and action in _ELDER_RESPONSES:
            raise PermissionError("提醒展示回执、确认、拒绝和延后只能由本人操作；护理员请记录跟进。")
        if action not in (_CAREGIVER_ACTIONS if caregiver else _ELDER_ACTIONS):
            raise PermissionError("该角色无权执行此护理操作")
        if not isinstance(payload, dict):
            raise ValueError("操作参数必须是对象")
        request_id = _identifier(payload.get("request_id"))
        digest = hashlib.sha256(_json([action, payload]).encode()).hexdigest()
        previous = self._db.execute("SELECT digest,result FROM commands WHERE actor=? AND elder_id=? AND request_id=?",
                                    (actor, elder_id, request_id)).fetchone()
        if previous:
            if previous["digest"] != digest:
                raise ValueError("请求标识已用于不同操作；未再次执行")
            return {**json.loads(previous["result"]), "duplicate": True}
        now = self._now()
        result = self._apply(elder_id, action, payload, actor, caregiver, now)
        result = {"ok": True, "action": action, "record": result}
        self._db.execute("INSERT INTO commands(actor,elder_id,request_id,digest,result) VALUES(?,?,?,?,?)",
                         (actor, elder_id, request_id, digest, _json(result)))
        return result

    def _apply(self, elder, action, p, actor, caregiver, now):
        if action == "reminder.create":
            schedule = p.get("schedule")
            zone = p.get("timezone")
            if schedule not in {"once", "daily"} or zone not in TIMEZONES:
                raise ValueError("提醒频率仅支持once/daily；时区仅支持UTC或Asia/Shanghai")
            if not isinstance(p.get("time"), str) or not _TIME.fullmatch(p["time"]):
                raise ValueError("时间必须为24小时制HH:MM")
            try:
                day = date.fromisoformat(p["date"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("日期必须为YYYY-MM-DD") from None
            if day.year < 1992:
                raise ValueError("本地固定时区实现仅支持1992年及以后日期")
            due = datetime.combine(day, time.fromisoformat(p["time"]), TIMEZONES[zone])
            if due <= now:
                raise ValueError("首次提醒日期时间必须在未来")
            row = {"id": "rem_" + secrets.token_hex(12), "elder_id": elder,
                   "label": _text(p.get("label"), "提醒内容", 120), "schedule": schedule,
                   "date": day.isoformat(), "time": p["time"], "timezone": zone,
                   "grace_minutes": _integer(p.get("grace_minutes", 30), "等待时间", 1, 1440),
                   "next_due_at": _stamp(due), "status": "active", "created_at": _stamp(now), "created_by": actor}
            self._put("reminders", row)
            return row
        if action == "reminder.cancel":
            row = self._get("reminders", p.get("reminder_id"), elder)
            if row["status"] == "completed":
                raise ValueError("已结束的一次性提醒不能取消")
            row["status"] = "cancelled"
            row["cancelled_at"] = _stamp(now)
            self._put("reminders", row)
            for occurrence in self._rows("occurrences", elder):
                if occurrence["reminder_id"] == row["id"] and occurrence["status"] in {"pending", "unconfirmed", "snoozed"}:
                    occurrence["status"] = "cancelled"
                    self._put("occurrences", occurrence)
                    self._cancel_queued(elder, occurrence_id=occurrence["id"])
            return row
        if action.startswith("occurrence."):
            row = self._get("occurrences", p.get("occurrence_id"), elder)
            if action == "occurrence.displayed":
                generation = p.get("generation")
                if type(generation) is not int or generation != row.get("generation", 0):
                    raise ValueError("提醒展示回执必须对应当前提醒代次")
                if self._displayed(row):
                    return row
                if row["status"] != "pending":
                    raise ValueError("不允许的提醒状态转换")
                reminder = self._get("reminders", row["reminder_id"], elder)
                stamp = _stamp(now)
                event = {"generation": generation, "at": stamp, "actor": actor,
                         "source": "elder_page_receipt_v1"}
                row.setdefault("display_history", []).append(event)
                # Client timestamps and role claims are never trusted. This
                # receipt records the visible-page callback, not human reading.
                row.update(display_contract="elder_page_receipt_v1", delivery_status="displayed_local",
                           displayed_at=stamp, displayed_by=actor, displayed_generation=generation,
                           display_source="elder_page_receipt_v1", offered_at=stamp,
                           deadline_at=_stamp(now + timedelta(minutes=reminder["grace_minutes"])))
                self._put("occurrences", row)
                return row
            statuses = {"occurrence.ack": "acknowledged", "occurrence.decline": "declined", "occurrence.snooze": "snoozed"}
            if action not in statuses or row["status"] not in {"pending", "unconfirmed", "snoozed"}:
                raise ValueError("不允许的提醒状态转换")
            if action == "occurrence.snooze":
                minutes = _integer(p.get("minutes"), "延后分钟", 1, 1440)
                row["snoozed_until"] = _stamp(now + timedelta(minutes=minutes))
                row["generation"] = row.get("generation", 0) + 1
                row.update(delivery_status="awaiting_display", displayed_at=None, displayed_by=None,
                           displayed_generation=None, display_source=None, offered_at=None, deadline_at=None)
            row.update(status=statuses[action], responded_at=_stamp(now), responded_by=actor)
            self._put("occurrences", row)
            self._cancel_queued(elder, occurrence_id=row["id"])
            reminder = self._get("reminders", row["reminder_id"], elder)
            if reminder["schedule"] == "once" and row["status"] in {"acknowledged", "declined"}:
                reminder["status"] = "completed"
                self._put("reminders", reminder)
            return row
        if action == "contact.create":
            row = {"id": "contact_" + secrets.token_hex(12), "elder_id": elder,
                   "name": _text(p.get("name"), "联系人", 80), "relation": str(p.get("relation", ""))[:80],
                   "channel": _text(p.get("channel", "local"), "联系人通道", 100), "verified": False,
                   "verification_source": None, "created_at": _stamp(now)}
            self._put("contacts", row)
            return row
        if action == "contact.verify" and caregiver:
            row = self._get("contacts", p.get("contact_id"), elder)
            row.update(verified=True, verification_source=_text(p.get("evidence_source"), "核验来源"),
                       verified_by=actor, verified_at=_stamp(now))
            self._put("contacts", row)
            return row
        if action == "consent.grant":
            contact = self._get("contacts", p.get("contact_id"), elder)
            reminder = self._get("reminders", p.get("reminder_id"), elder)
            if reminder["status"] != "active":
                raise ValueError("仅可授权仍有效的提醒")
            if p.get("confirmed") is not True or p.get("trigger") != "unconfirmed" or p.get("scope") != "reminder_status":
                raise ValueError("必须明确确认仅分享提醒未确认状态")
            source = p.get("source")
            if source not in {"simulation", "real"}:
                raise ValueError("必须明确授权来源")
            if source == "real" and (not self.allow_external or not contact["verified"]):
                raise ValueError("真实授权需要显式外发配置和已核验联系人")
            expiry = _parse(p.get("expires_at"))
            start = _parse(p["valid_from"]) if p.get("valid_from") else now
            if expiry <= max(now, start) or expiry > now + timedelta(days=366):
                raise ValueError("授权必须在未来到期且有效期不超过366天")
            for old in self._rows("consents", elder):
                if old["contact_id"] == contact["id"] and old["reminder_id"] == reminder["id"] and old["status"] == "active" and _parse(old["expires_at"]) > now:
                    raise ValueError("该联系人已有有效规则；请先撤回旧授权")
            row = {"id": "consent_" + secrets.token_hex(12), "elder_id": elder,
                   "contact_id": contact["id"], "reminder_id": reminder["id"], "trigger": "unconfirmed",
                   "scope": "reminder_status", "source": source, "valid_from": _stamp(start), "expires_at": _stamp(expiry),
                   "status": "active", "recorded_by": actor, "confirmed_at": _stamp(now)}
            self._put("consents", row)
            return row
        if action == "consent.revoke":
            row = self._get("consents", p.get("consent_id"), elder)
            row.update(status="revoked", revoked_at=_stamp(now), revoked_by=actor)
            self._put("consents", row)
            self._cancel_queued(elder, consent_id=row["id"])
            return row
        if action in {"task.follow_up", "task.close"} and caregiver:
            row = self._get("tasks", p.get("task_id"), elder)
            target = "following_up" if action == "task.follow_up" else "closed"
            if row["status"] == "closed" or (target == "closed" and row["status"] != "following_up"):
                raise ValueError("任务须先跟进后关闭，已关闭不能再次修改")
            note = _text(p.get("note"), "跟进记录", 500)
            row["history"].append({"from": row["status"], "to": target, "at": _stamp(now), "actor": actor, "note": note})
            row.update(status=target, note=note)
            self._put("tasks", row)
            return row
        if action == "receipt.record" and caregiver:
            row = self._get("deliveries", p.get("delivery_id"), elder)
            target = p.get("status")
            allowed = {"accepted": {"delivered", "family_ack"}, "delivered": {"family_ack"},
                       "delivered_local": {"family_ack"}, "unknown": {"delivered", "family_ack", "failed"}}
            if target not in allowed.get(row["status"], set()):
                raise ValueError("不允许的通知回执转换；不能手工伪造网关受理")
            source = _text(p.get("evidence_source"), "人工核查来源")
            note = _text(p.get("note"), "核查记录", 500)
            if row["channel"] == "local_care_inbox" and target == "delivered":
                raise ValueError("本地模拟记录不能改成真实送达")
            row["receipt_history"].append({"status": target, "source": "caregiver_report",
                                            "evidence_source": source, "note": note, "actor": actor, "at": _stamp(now)})
            row.update(status=target, receipt_source="caregiver_report")
            self._put("deliveries", row)
            return row
        raise ValueError("不支持的护理操作")

    def _cancel_queued(self, elder, *, occurrence_id=None, consent_id=None,
                       reason="确认、拒绝、延后、取消或撤回阻止了尚未发送的通知"):
        for row in self._rows("deliveries", elder):
            if row["status"] == "queued" and (occurrence_id is None or row["occurrence_id"] == occurrence_id) and (consent_id is None or row["consent_id"] == consent_id):
                row.update(status="cancelled", error=reason)
                self._put("deliveries", row)

    def _eligible(self, consent, occurrence, now):
        return (consent["status"] == "active" and consent["reminder_id"] == occurrence["reminder_id"]
                and _parse(consent["valid_from"]) <= now < _parse(consent["expires_at"])
                and (not self.external or consent["source"] == "real"))

    @staticmethod
    def _displayed(occurrence):
        try:
            displayed_at = _parse(occurrence.get("displayed_at"))
            deadline_at = _parse(occurrence.get("deadline_at"))
        except ValueError:
            return False
        return (occurrence.get("display_contract") == "elder_page_receipt_v1"
                and occurrence.get("delivery_status") == "displayed_local"
                and occurrence.get("display_source") == "elder_page_receipt_v1"
                and occurrence.get("displayed_by") == "elder:" + occurrence["elder_id"]
                and type(occurrence.get("displayed_generation")) is int
                and type(occurrence.get("generation")) is int
                and occurrence["displayed_generation"] == occurrence.get("generation", 0)
                and deadline_at > displayed_at)

    def tick(self, now=None):
        """Generate due reminders, queue shown-and-unconfirmed events, dispatch once.

        A delayed daily rule offers only its most recent due instance. Skipped
        dates are counted explicitly, never marked delivered or unconfirmed.
        Confirmation windows start only on an elder visible-page receipt.
        """
        current = self._now(now)
        with self._lock, self._tx():
            for reminder in self._rows("reminders"):
                if reminder["status"] != "active" or not reminder["next_due_at"]:
                    continue
                due = _parse(reminder["next_due_at"])
                if due <= current:
                    if reminder["schedule"] == "daily":
                        for old in self._rows("occurrences", reminder["elder_id"]):
                            if old["reminder_id"] == reminder["id"] and old["status"] in {"pending", "snoozed", "unconfirmed"}:
                                old.update(status="superseded", superseded_at=_stamp(current),
                                           superseded_reason="新的每日提醒已到期；旧提醒停止等待，不补发旧日通知")
                                self._put("occurrences", old)
                                self._cancel_queued(reminder["elder_id"], occurrence_id=old["id"])
                    skipped = (current - due).days if reminder["schedule"] == "daily" else 0
                    if skipped:
                        reminder["catchup"] = {"skipped_count": skipped, "first_skipped_due_at": _stamp(due),
                            "last_skipped_due_at": _stamp(due + timedelta(days=skipped - 1)),
                            "recorded_at": _stamp(current), "status": "not_offered"}
                        reminder["skipped_total"] = reminder.get("skipped_total", 0) + skipped
                        due += timedelta(days=skipped)
                    occ_id = _stable("occ", reminder["id"], _stamp(due))
                    self._put("occurrences", {"id": occ_id, "elder_id": reminder["elder_id"],
                        "reminder_id": reminder["id"], "label": reminder["label"], "due_at": _stamp(due),
                        "timezone": reminder["timezone"], "status": "pending", "generated_at": _stamp(current),
                        "generation": 0, "delivery_status": "awaiting_display", "offered_at": None,
                        "display_contract": "elder_page_receipt_v1", "displayed_at": None, "displayed_by": None,
                        "displayed_generation": None, "display_source": None, "display_history": [],
                        "deadline_at": None})
                    if reminder["schedule"] == "once":
                        reminder["next_due_at"] = None
                    else:
                        reminder["next_due_at"] = _stamp(due + timedelta(days=1))
                    self._put("reminders", reminder)
            for occurrence in self._rows("occurrences"):
                elder = occurrence["elder_id"]
                reminder = self._get("reminders", occurrence["reminder_id"], elder)
                if occurrence["status"] == "snoozed" and _parse(occurrence["snoozed_until"]) <= current:
                    occurrence.update(status="pending", generated_at=_stamp(current), delivery_status="awaiting_display",
                        displayed_at=None, displayed_by=None, displayed_generation=None, display_source=None,
                        offered_at=None, deadline_at=None)
                    self._put("occurrences", occurrence)
                if (occurrence["status"] != "pending" or not self._displayed(occurrence)
                        or _parse(occurrence["deadline_at"]) > current):
                    continue
                occurrence["status"] = "unconfirmed"
                self._put("occurrences", occurrence)
                task_id = _stable("task", occurrence["id"], occurrence.get("generation", 0))
                if not self._db.execute("SELECT 1 FROM records WHERE kind='tasks' AND id=?", (task_id,)).fetchone():
                    self._put("tasks", {"id": task_id, "elder_id": elder, "occurrence_id": occurrence["id"],
                        "status": "open", "summary": "提醒未确认，请核查是否需要帮助；不能据此认定未服药或遇险。",
                        "created_at": _stamp(current), "note": "", "history": [],
                        "visibility_evidence": occurrence["display_source"],
                        "displayed_at": occurrence["displayed_at"], "generation": occurrence.get("generation", 0)})
                for consent in self._rows("consents", elder):
                    if not self._eligible(consent, occurrence, current):
                        continue
                    contact = self._get("contacts", consent["contact_id"], elder)
                    if self.external and not contact["verified"]:
                        continue
                    delivery_id = _stable("delivery", occurrence["id"], occurrence.get("generation", 0), contact["id"])
                    if self._db.execute("SELECT 1 FROM records WHERE kind='deliveries' AND id=?", (delivery_id,)).fetchone():
                        continue
                    # Never include the label, conversations, personality or medical inference.
                    message = f"提醒未确认。原定时间：{occurrence['due_at']}。这只表示尚未收到确认；请按约定核查是否需要帮助。"
                    self._put("deliveries", {"id": delivery_id, "elder_id": elder, "occurrence_id": occurrence["id"],
                        "contact_id": contact["id"], "consent_id": consent["id"], "status": "queued",
                        "channel": self.notifier.channel, "message": message, "event_id": delivery_id,
                        "created_at": _stamp(current), "provider_id": None, "receipt_history": [], "error": None,
                        "source": consent["source"], "occurrence_generation": occurrence.get("generation", 0),
                        "visibility_evidence": occurrence["display_source"], "displayed_at": occurrence["displayed_at"]})
        return self.dispatch_pending(now=now)

    def dispatch_pending(self, now=None):
        """Public for a scheduler/test boundary; does not requeue unknown or failed."""
        sent = []
        with self._lock:
            ids = [row["id"] for row in self._rows("deliveries") if row["status"] == "queued"][:32]
            for row_id in ids:
                # A previous gateway call may have crossed another rule's
                # expiry. Explicit test clocks are separate from runtime time.
                current = self._now(now)
                with self._tx():
                    raw = self._db.execute("SELECT elder_id FROM records WHERE kind='deliveries' AND id=?", (row_id,)).fetchone()
                    if raw is None:
                        continue
                    elder = raw[0]
                    row = self._get("deliveries", row_id, elder)
                    if row["status"] != "queued":
                        continue
                    occurrence = self._get("occurrences", row["occurrence_id"], elder)
                    consent = self._get("consents", row["consent_id"], elder)
                    contact = self._get("contacts", row["contact_id"], elder)
                    if (occurrence["status"] != "unconfirmed" or not self._displayed(occurrence)
                            or type(row.get("occurrence_generation")) is not int
                            or row["occurrence_generation"] != occurrence.get("generation", 0)
                            or not self._eligible(consent, occurrence, current)
                            or (self.external and not contact["verified"])):
                        row.update(status="cancelled", error="发送前复核：规则已失效或老人已回应")
                        self._put("deliveries", row)
                        continue
                    row.update(status="sending", attempted_at=_stamp(current))
                    self._put("deliveries", row)
                # The durable sending state is committed before touching the adapter.
                try:
                    receipt = self.notifier.send(recipient=contact["name"], message=row["message"], severity="normal",
                        idempotency_key=row_id, subject_id=elder, recipient_id=contact["id"], recipient_channel=contact["channel"])
                    status = receipt.status
                    # The legacy webhook collapses network failures to failed.
                    # Treat all failed/unsupported external results conservatively as unknown.
                    if status not in {"accepted", "delivered", "delivered_local"} or (self.external and status == "delivered_local"):
                        status = "unknown"
                    if not self.external and status != "delivered_local":
                        status = "unknown"
                    provider_id = str(receipt.provider_id)[:200] if receipt.provider_id else None
                except Exception:
                    status, provider_id = "unknown", None
                with self._tx():
                    row = self._get("deliveries", row_id, elder)
                    # Another process may already have conservatively marked an interrupted send unknown.
                    if row["status"] != "sending":
                        continue
                    row.update(status=status, provider_id=provider_id,
                               error="发送结果未知，禁止自动重发" if status == "unknown" else None)
                    row["receipt_history"].append({"status": status, "source": "local_adapter" if not self.external else "gateway",
                                                    "at": _stamp(current), "provider_id": provider_id})
                    self._put("deliveries", row)
                    if status == "unknown":
                        for task in self._rows("tasks", elder):
                            if task["occurrence_id"] == row["occurrence_id"] and task["status"] != "closed":
                                task["delivery_attention"] = "通知发送结果未知；请核查回执，禁止自动重发。"
                                self._put("tasks", task)
                    sent.append(row_id)
        return {"ok": True, "processed": len(sent), "delivery_ids": sent}

    def forget_elder(self, elder_id):
        """Clear only local fictional artifacts; retain account and assignments."""
        if self.external or self.allow_external:
            raise ValueError("真实外发账本不可通过清除体验删除")
        with self._lock, self._tx():
            _identifier(elder_id)
            self._db.execute("DELETE FROM records WHERE elder_id=?", (elder_id,))
            self._db.execute("DELETE FROM commands WHERE elder_id=?", (elder_id,))
        return {"ok": True}

    def close(self):
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
