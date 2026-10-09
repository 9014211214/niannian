"""Explicit automatic-review metadata and a durable daily job quota.

The quota contains only opaque IDs and dates, separately from deletable chat.
It counts first dispatch attempts, including failed or uncertain starts, rather
than money. The host retains its ordinary per-request reservation and stop gates.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

VERSION = "minimal_auto_review_v1"
_LOCAL_TZ = timezone(timedelta(hours=8))
_HASH = re.compile(r"^[a-f0-9]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_STATES = {"queued", "deferred", "dispatching", "running", "completed", "failed", "pending", "cancelled", "blocked"}


def default_settings():
    return {"version": VERSION, "enabled": False, "daily_limit": 1,
            "enabled_at": None, "events": [], "commands": []}


def local_day(now=None):
    clock = now or datetime.now(timezone.utc)
    if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("auto_review_clock_requires_timezone")
    return clock.astimezone(_LOCAL_TZ).date().isoformat()


def stable_request_id(run_id, session_id):
    identity = json.dumps([run_id, session_id], ensure_ascii=False, separators=(",", ":"))
    return "auto_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:48]


def command_digest(enabled, daily_limit):
    value = json.dumps({"enabled": enabled, "daily_limit": daily_limit}, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_digest(job, owner):
    # Hash the sealed owner source, rather than mutable stage/attempt results.
    fields = ("session_id", "scene_id", "created_at", "expires_at", "source_completeness",
              "turns", "excluded_turn_indices", "cognitive_turns", "cognitive_session_at")
    source = ({key: job.model_dump(mode="json")[key] for key in fields}
              if job is not None else {"source": None})
    source["owner"] = owner
    return hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("invalid_auto_review_timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("invalid_auto_review_timestamp")


def validate_settings(value, run_id, sessions):
    """Reject malformed retained state; never infer permission from old runs."""
    if (type(value) is not dict or set(value) != set(default_settings())
            or value.get("version") != VERSION or type(value.get("enabled")) is not bool
            or type(value.get("daily_limit")) is not int or not 1 <= value["daily_limit"] <= 8
            or type(value.get("events")) is not list or type(value.get("commands")) is not list):
        raise ValueError("invalid_auto_review_settings")
    if value["enabled_at"] is not None:
        _timestamp(value["enabled_at"])
    if value["enabled"] and value["enabled_at"] is None:
        raise ValueError("auto_review_consent_missing")
    ids = set()
    command_ids = set()
    for command in value["commands"]:
        if (type(command) is not dict or set(command) != {"request_id", "digest", "at"}
                or not isinstance(command.get("request_id"), str) or not _ID.fullmatch(command["request_id"])
                or command["request_id"] in command_ids or not isinstance(command.get("digest"), str)
                or not _HASH.fullmatch(command["digest"])):
            raise ValueError("invalid_auto_review_command")
        _timestamp(command["at"])
        command_ids.add(command["request_id"])
    if value["commands"]:
        last = value["commands"][-1]
        if (last["digest"] != command_digest(value["enabled"], value["daily_limit"])
                or (value["enabled"] and value["enabled_at"] != last["at"])):
            raise ValueError("auto_review_setting_command_mismatch")
    elif value["enabled"] or value["enabled_at"] is not None or value["daily_limit"] != 1:
        raise ValueError("auto_review_setting_command_missing")
    for row in value["events"]:
        fields = {"request_id", "session_id", "source_sha256", "authorized_at", "created_at",
                  "status", "reason", "started_on"}
        if (type(row) is not dict or set(row) != fields or row.get("session_id") not in sessions
                or row.get("session_id") in ids
                or row.get("request_id") != stable_request_id(run_id, row["session_id"])
                or not isinstance(row.get("source_sha256"), str)
                or not _HASH.fullmatch(row["source_sha256"])
                or row.get("status") not in _STATES
                or not isinstance(row.get("reason"), str) or len(row["reason"]) > 80):
            raise ValueError("invalid_auto_review_event")
        _timestamp(row["created_at"])
        _timestamp(row["authorized_at"])
        if not any(command["at"] == row["authorized_at"]
                   and command["digest"] in {command_digest(True, limit) for limit in range(1, 9)}
                   for command in value["commands"]):
            raise ValueError("auto_review_event_authorization_missing")
        if row["started_on"] is not None:
            try:
                if datetime.strptime(row["started_on"], "%Y-%m-%d").date().isoformat() != row["started_on"]:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("invalid_auto_review_start_day") from None
        if row["status"] in {"dispatching", "running", "completed", "failed", "pending", "cancelled"} and row["started_on"] is None:
            raise ValueError("auto_review_start_missing")
        ids.add(row["session_id"])
    return value


def view(settings, *, running=False, dispatching=None, now=None, quota_used=None):
    day = local_day(now)
    events = deepcopy(settings["events"])
    uncertain = False
    for row in events:
        if (row["status"] in {"dispatching", "running", "pending"}
                and not running and row["request_id"] != dispatching):
            row["status"], row["reason"] = "pending", "interrupted_requires_review"
            uncertain = True
    started = (sum(row["started_on"] == day for row in events) if quota_used is None else quota_used)
    blocked = sum(row["status"] == "blocked" for row in events)
    last = settings["commands"][-1] if settings["commands"] else {}
    label = "已开启" if settings["enabled"] else "已关闭"
    return {"version": VERSION, "enabled": settings["enabled"], "daily_limit": settings["daily_limit"],
            "enabled_at": settings["enabled_at"], "day": day, "timezone": "Asia/Shanghai",
            "day_timezone": "Asia/Shanghai", "day_label": day,
            "quota_scope": "current_data_directory_and_mode_not_account_wide",
            "today_started": started, "blocked_count": blocked, "requires_review": uncertain,
            "count_scope": "reserved_automatic_job_attempts_including_failed_or_uncertain_starts",
            "last_command_id": last.get("request_id"), "last_command_digest": last.get("digest"),
            "events": events,
            "message": f"自动会后整理{label}；当前资料目录今日已占用{started}/{settings['daily_limit']}个任务名额，{blocked}个已阻止。按上海自然日计；启动失败或结果不明也占名额。任务数上限不是费用或全账号上限，费用停发仍生效。"}


class AutoReviewQuota:
    """Atomic cross-process quota; disabled/read-only use creates no file."""

    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def _check_day(day):
        if type(day) is not str or datetime.strptime(day, "%Y-%m-%d").date().isoformat() != day:
            raise ValueError("invalid_auto_review_quota_day")

    def _read(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=3)
        try:
            if connection.execute("SELECT version FROM auto_metadata").fetchall() != [(VERSION,)]:
                raise ValueError("invalid_auto_review_quota_version")
            rows = connection.execute("SELECT request_id, day FROM auto_jobs").fetchall()
            if any(not _ID.fullmatch(row[0]) for row in rows):
                raise ValueError("invalid_auto_review_quota_id")
            for _, day in rows:
                self._check_day(day)
            return rows
        finally:
            connection.close()

    def validate_requests(self, expected):
        rows = dict(self._read()) if self.path.exists() else {}
        if any(rows.get(request_id) != day for request_id, day in expected.items()):
            raise ValueError("automatic_dispatch_quota_missing_or_changed")

    def used(self, day):
        self._check_day(day)
        return sum(value == day for _, value in self._read()) if self.path.exists() else 0

    def reserve(self, request_id, day, limit):
        if type(request_id) is not str or not _ID.fullmatch(request_id):
            raise ValueError("invalid_auto_review_quota_id")
        self._check_day(day)
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("invalid_auto_review_quota_limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=3, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS auto_metadata(version TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS auto_jobs(request_id TEXT PRIMARY KEY, day TEXT NOT NULL)")
            versions = connection.execute("SELECT version FROM auto_metadata").fetchall()
            if not versions:
                connection.execute("INSERT INTO auto_metadata(version) VALUES(?)", (VERSION,))
            elif versions != [(VERSION,)]:
                raise ValueError("invalid_auto_review_quota_version")
            for old_id, old_day in connection.execute("SELECT request_id,day FROM auto_jobs"):
                if type(old_id) is not str or not _ID.fullmatch(old_id):
                    raise ValueError("invalid_auto_review_quota_id")
                self._check_day(old_day)
            if connection.execute("SELECT 1 FROM auto_jobs WHERE request_id=?", (request_id,)).fetchone():
                result = "duplicate"
            elif connection.execute("SELECT COUNT(*) FROM auto_jobs WHERE day=?", (day,)).fetchone()[0] >= limit:
                result = "daily_limit"
            else:
                connection.execute("INSERT INTO auto_jobs(request_id,day) VALUES(?,?)", (request_id, day))
                result = "reserved"
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
