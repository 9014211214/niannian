"""Explicit, durable recovery of at most two complete upstream HTTP429 errors.

Failed records and unknown-cost reservations are never rewritten or released.
This policy is opt-in and cannot infer permission from an HTTP status alone.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import re

from .request_pacing import monotonic_value, wait_until
from .observed_transport import validate_trace

SCHEMA = "bounded_http429_recovery_v1"
_FIXED = {"schema": SCHEMA, "max_extra_dispatches": 2, "max_new_unknown": 2,
          "backoff_seconds": [60, 120], "max_attempts_per_operation": 3, "max_logical_requests": 1636}
CONTINUATION_SCHEMA = "bounded_http429_continuation_v1"
_CONTINUATION_FIXED = {"schema": CONTINUATION_SCHEMA, "max_extra_dispatches": 1, "max_new_unknown": 1,
    "backoff_seconds": [120], "max_attempts_per_operation": 2, "max_logical_requests": 1246}
MAIN_CONTINUATION_SCHEMA = "bounded_main_continuation_v1"
_MAIN_CONTINUATION_FIXED = {"schema": MAIN_CONTINUATION_SCHEMA, "max_extra_dispatches": 0,
    "max_new_unknown": 1, "backoff_seconds": [], "max_attempts_per_operation": 1,
    "max_logical_requests": 1083}
_EVIDENCE = {"schema": "http429_error_envelope_v1", "complete_error_envelope": True, "no_model_answer": True}


def validate_recovery_policy(value):
    if value is None:
        return None
    fixed = (_MAIN_CONTINUATION_FIXED if type(value) is dict and value.get("schema") == MAIN_CONTINUATION_SCHEMA
             else _CONTINUATION_FIXED if type(value) is dict and value.get("schema") == CONTINUATION_SCHEMA else _FIXED)
    if (type(value) is not dict or set(value) != {*fixed, "authorization_sha256"}
            or any(json.dumps(value.get(key), sort_keys=True) != json.dumps(expected, sort_keys=True)
                   for key, expected in fixed.items())
            or type(value.get("authorization_sha256")) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", value["authorization_sha256"])):
        raise ValueError("invalid_bounded_recovery_policy")
    return deepcopy(value)


def recovery_eligible(row):
    observed = row.get("observation", {})
    proof = row.get("http_error_recovery_evidence")
    trace = row.get("transport_events", {})
    try:
        validate_trace(trace, require_complete=True, require_https=True)
    except ValueError:
        return False
    return (row.get("status") == "failed" and type(row.get("http_status")) is int
        and row["http_status"] == 429 and row.get("failure_phase") == "http_status"
        and row.get("failure_code") == "http_error" and proof == _EVIDENCE
        and json.dumps(proof, sort_keys=True) == json.dumps(_EVIDENCE, sort_keys=True)
        and observed.get("envelope_kind") == "http_error" and observed.get("body_status") == "parsed"
        and observed.get("error_category") == "rate_limited" and observed.get("provider_error_code") == 429
        and observed.get("provider_name_status") == "recognized"
        and observed.get("provider_reported_name") == row.get("expected_provider")
        and row.get("expected_provider") in {"Alibaba", "Anthropic"}
        and not observed.get("body_close_failed") and row.get("http_dispatch_started") is True)


def _key(row):
    return row["role"], row["session_id"], row["operation_id"]


def validate_recovery_records(records, policy, *, relations=None, require_complete=False):
    policy = validate_recovery_policy(policy)
    if policy is None or type(records) is not list:
        raise ValueError("bounded_recovery_evidence_required")
    groups, by_id, extras = {}, {}, []
    for row in records:
        if (type(row) is not dict or not row.get("request_id") or row["request_id"] in by_id
                or row.get("logical_operation_id") != row.get("operation_id")
                or type(row.get("physical_attempt")) is not int
                or row.get("status") not in {"pending", "received", "failed"}
                or type(row.get("http_dispatch_started")) is not bool
                or row.get("status") == "received" and row.get("http_dispatch_started") is not True):
            raise ValueError("bounded_recovery_record_invalid")
        by_id[row["request_id"]] = row
        group = groups.setdefault(_key(row), [])
        if row["physical_attempt"] != len(group) + 1 or len(group) >= policy["max_attempts_per_operation"]:
            raise ValueError("bounded_recovery_attempt_order_changed")
        if group:
            prior = group[-1]
            if (len(extras) >= policy["max_extra_dispatches"] or not recovery_eligible(prior) or row.get("recovery_of_request_id") != prior["request_id"]
                    or row.get("request_sha256") != prior["request_sha256"]
                    or row.get("model") != prior.get("model") or row.get("expected_provider") != prior.get("expected_provider")
                    or row.get("e1b_stage") != prior.get("e1b_stage")
                    or row.get("registration_sha256") != prior.get("registration_sha256")
                    or row.get("recovery_ordinal") != len(extras) + 1
                    or row.get("recovery_wait_seconds") != policy["backoff_seconds"][len(extras)]
                    or type(row.get("recovery_wait_ms")) not in (int, float)
                    or not row["recovery_wait_ms"] >= row["recovery_wait_seconds"] * 1000):
                raise ValueError("bounded_recovery_relation_changed")
            extras.append(row)
        elif any(key in row for key in ("recovery_of_request_id", "recovery_ordinal", "recovery_wait_seconds", "recovery_wait_ms")):
            raise ValueError("bounded_recovery_unexpected_relation")
        group.append(row)
    for group in groups.values():
        settled = [row for row in group if row["status"] != "pending"]
        if any("connection_preparations" in row.get("transport_events", {}) for row in settled):
            ordinals = []
            for row in settled:
                trace = row.get("transport_events", {})
                if "connection_preparations" not in trace:
                    raise ValueError("bounded_recovery_preparation_evidence_missing")
                validate_trace(trace, require_complete=row["status"] == "received")
                ordinals.extend(item["ordinal"] for item in trace["connection_preparations"])
            if ordinals != list(range(1, len(ordinals) + 1)) or len(ordinals) > 2:
                raise ValueError("bounded_recovery_preparation_limit_exceeded")
    if (len(groups) > policy["max_logical_requests"] or len(extras) > policy["max_extra_dispatches"]
            or len(records) > policy["max_logical_requests"] + policy["max_extra_dispatches"]):
        raise ValueError("bounded_recovery_inventory_exceeded")
    if relations is not None:
        if type(relations) is not list or len(relations) > policy["max_extra_dispatches"]:
            raise ValueError("bounded_recovery_relations_invalid")
        children = set()
        for number, relation in enumerate(relations, 1):
            parent = by_id.get(relation.get("parent_request_id"))
            child = by_id.get(relation.get("child_request_id"))
            if (set(relation) != {"ordinal", "parent_request_id", "child_request_id", "status", "wait_seconds", "wait_ms"}
                    or relation["ordinal"] != number or relation["wait_seconds"] != policy["backoff_seconds"][number - 1]
                    or parent is None or not recovery_eligible(parent)):
                raise ValueError("bounded_recovery_relations_invalid")
            if child is None:
                if relation["child_request_id"] is not None or relation["status"] != "waiting" or relation["wait_ms"] is not None:
                    raise ValueError("bounded_recovery_wait_changed")
            else:
                if (child["request_id"] in children or child.get("recovery_of_request_id") != parent["request_id"]
                        or child.get("recovery_ordinal") != number or child.get("recovery_wait_ms") != relation["wait_ms"]
                        or relation["status"] != {"pending": "dispatched", "received": "completed", "failed": "failed"}[child["status"]]):
                    raise ValueError("bounded_recovery_saved_relation_changed")
                children.add(child["request_id"])
        if children != {row["request_id"] for row in extras}:
            raise ValueError("bounded_recovery_missing_relation")
    complete = [group[-1] for group in groups.values() if group[-1]["status"] == "received"]
    authorized = [row for group in groups.values() if group[-1]["status"] == "received" for row in group[:-1]]
    unknown = [row for row in records if "cost" not in row.get("usage", {})]
    if len(unknown) > policy["max_new_unknown"]:
        raise ValueError("bounded_recovery_unknown_limit_exceeded")
    if require_complete and len(complete) != len(groups):
        raise ValueError("bounded_recovery_logical_calls_incomplete")
    return {"logical_attempts": complete, "logical_calls_used": len(groups), "logical_received": len(complete),
        "extra_dispatches": len(extras), "new_unknown_requests": len(unknown),
        "authorized_unknown_requests": sum("cost" not in row.get("usage", {}) for row in authorized),
        "authorized_failed_request_ids": [row["request_id"] for row in authorized],
        "http_dispatches": sum(row["http_dispatch_started"] for row in records),
        "relations": deepcopy(relations) if relations is not None else None}


class BoundedRecovery:
    def __init__(self, guard, policy):
        self.guard, self.policy = guard, policy
        self.encoded = json.dumps(policy, sort_keys=True, allow_nan=False) if policy is not None else None
        self.active = None
        self.wait_ms = None

    def initialize(self):
        g = self.guard
        table = g._db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bounded_recovery_policy'").fetchone()
        if table is None and self.policy is not None:
            if g._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                raise ValueError("retained_ledger_has_no_recovery_policy")
            g._db.execute("CREATE TABLE bounded_recovery_policy(id INTEGER PRIMARY KEY CHECK(id=1),policy TEXT NOT NULL)")
            g._db.execute("INSERT INTO bounded_recovery_policy VALUES(1,?)", (self.encoded,))
            g._db.execute("CREATE TABLE authorized_recoveries(ordinal INTEGER PRIMARY KEY,parent_request_id TEXT UNIQUE NOT NULL,"
                "child_request_id TEXT UNIQUE,status TEXT NOT NULL,wait_seconds INTEGER NOT NULL,wait_ms REAL)")
        self.check()

    def check(self):
        g = self.guard
        table = g._db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bounded_recovery_policy'").fetchone()
        if table is None:
            if self.policy is not None:
                raise ValueError("retained_recovery_policy_missing")
        elif (self.policy is None or json.dumps(g.recovery_policy, sort_keys=True, allow_nan=False) != self.encoded
              or g._db.execute("SELECT policy FROM bounded_recovery_policy WHERE id=1").fetchone() != (self.encoded,)):
            raise ValueError("retained_recovery_policy_changed")

    def relations(self):
        fields = ("ordinal", "parent_request_id", "child_request_id", "status", "wait_seconds", "wait_ms")
        return [dict(zip(fields, row)) for row in self.guard._db.execute(
            "SELECT ordinal,parent_request_id,child_request_id,status,wait_seconds,wait_ms FROM authorized_recoveries ORDER BY ordinal")]

    def summary(self, records):
        self.check()
        relations = self.relations()
        summary = validate_recovery_records(records, self.policy, relations=relations)
        allowed = set(summary["authorized_failed_request_ids"])
        if self.active is not None:
            relation = relations[self.active - 1]
            if relation["child_request_id"] is None:
                parent = next(row for row in records if row["request_id"] == relation["parent_request_id"])
                allowed.update(row["request_id"] for row in records if _key(row) == _key(parent) and recovery_eligible(row))
        interrupted = any(relation["status"] == "waiting" and relation["ordinal"] != self.active for relation in relations)
        return summary, allowed, interrupted

    def prepare_wait(self, client, request):
        g = self.guard
        if g._broken or g._closed or g._stop_requested.is_set():
            return False
        if (g._connection_preparation_recovery
                and getattr(g._local, "preparation_budget", {}).get("attempts", 2) >= 2):
            # No connection remains for a second HTTP attempt. Stop before a
            # waiting relation, new fee hold or physical claim can be created.
            return False
        with g._transaction():
            self.check()
            if g._db.execute("SELECT halted FROM settings WHERE id=1").fetchone()[0] is not None:
                return False
            rows = [json.loads(row[0]) for row in g._db.execute("SELECT record FROM attempts ORDER BY sequence")]
            relations = self.relations()
            summary = validate_recovery_records(rows, self.policy, relations=relations)
            scope = getattr(g._local, "scope", None)
            if (not rows or scope is None or rows[-1]["role"] != client.role
                    or rows[-1]["operation_id"] != scope["operation_id"]
                    or rows[-1]["session_id"] != scope["session_id"]
                    or rows[-1].get("e1b_stage") != getattr(g, "stage", None)
                    or rows[-1]["request_sha256"] != hashlib.sha256(request.data).hexdigest()
                    or not recovery_eligible(rows[-1]) or rows[-1]["physical_attempt"] >= self.policy["max_attempts_per_operation"]
                    or len(relations) >= self.policy["max_extra_dispatches"]
                    or summary["new_unknown_requests"] >= self.policy["max_new_unknown"]
                    or any(relation["status"] == "waiting" for relation in relations)):
                return False
            number = len(relations) + 1
            seconds = self.policy["backoff_seconds"][number - 1]
            g._db.execute("INSERT INTO authorized_recoveries VALUES(?,?,NULL,'waiting',?,NULL)",
                          (number, rows[-1]["request_id"], seconds))
            self.active, self.wait_ms = number, None
        started = monotonic_value(g._pacing_clock)
        self.wait_ms = wait_until(started + seconds, monotonic=g._pacing_clock,
                                 sleep=g._pacing_sleep, cancelled=g._stop_requested.is_set)
        return True

    def claim(self, row, previous):
        row.update(logical_operation_id=row["operation_id"], physical_attempt=len(previous) + 1,
                   http_dispatch_started=False)
        if self.active is None:
            if previous:
                raise ValueError("duplicate_model_operation")
            return
        relation = self.relations()[self.active - 1]
        if (self.wait_ms is None or not previous or row["session_id"] != previous[-1]["session_id"]
                or row.get("e1b_stage") != previous[-1].get("e1b_stage")
                or row.get("registration_sha256") != previous[-1].get("registration_sha256")
                or relation["parent_request_id"] != previous[-1]["request_id"]
                or relation["status"] != "waiting" or relation["child_request_id"] is not None
                or row["request_sha256"] != previous[-1]["request_sha256"]):
            raise ValueError("bounded_recovery_claim_changed")
        row.update(recovery_of_request_id=previous[-1]["request_id"], recovery_ordinal=self.active,
                   recovery_wait_seconds=relation["wait_seconds"], recovery_wait_ms=round(self.wait_ms, 3))
        self.guard._db.execute("UPDATE authorized_recoveries SET child_request_id=?,status='dispatched',wait_ms=? WHERE ordinal=?",
                              (row["request_id"], row["recovery_wait_ms"], self.active))
        self.active, self.wait_ms = None, None

    def finish(self, row):
        if "recovery_ordinal" in row:
            self.guard._db.execute("UPDATE authorized_recoveries SET status=? WHERE ordinal=? AND child_request_id=?",
                ("completed" if row["status"] == "received" else "failed", row["recovery_ordinal"], row["request_id"]))

    def cost_held(self, records):
        return str(sum((Decimal(row["reservation"]["reserved_cost_usd"])
                       for row in records if "cost" not in row.get("usage", {})), Decimal(0)))
