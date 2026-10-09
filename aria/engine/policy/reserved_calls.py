"""Durable, pre-send reservations for explicitly priced model requests.

The token allowance is a conservative planning convention, not a provider
invoice guarantee. Unknown outcomes remain charged against the allowance and
halt the run. No function here obtains credentials or refreshes prices.
The inherited physical transport rejects HTTP redirects in every mode; a
redirect never creates a second request outside this reservation ledger.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from .minimal_calls import MinimalCallGuard, MinimalCallStopped
from .response_observability import observe_response_error, receipt_identity, validate_receipt_identity

SCHEMA = "aria_reservation_v1"
FRAMING_TOKENS = 8192
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def _decimal(value):
    if type(value) not in (str, int, float, Decimal):
        raise ValueError("invalid_reservation_price")
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result < 0 or result > 1000000:
            raise InvalidOperation
        return result
    except (InvalidOperation, ValueError):
        raise ValueError("invalid_reservation_price") from None


def validate_reservation_policy(value):
    if (type(value) is not dict or set(value) != {"schema", "framing_tokens", "models"}
            or value["schema"] != SCHEMA or type(value["framing_tokens"]) is not int
            or value["framing_tokens"] != FRAMING_TOKENS
            or type(value["models"]) is not dict or not value["models"]):
        raise ValueError("invalid_reservation_policy")
    result = deepcopy(value)
    fields = {"input_usd_per_million", "output_usd_per_million", "provider",
              "response_models", "response_providers", "max_input_tokens",
              "max_output_tokens", "metadata_sha256"}
    for model, spec in result["models"].items():
        if (not isinstance(model, str) or not 1 <= len(model) <= 160
                or type(spec) is not dict or set(spec) != fields):
            raise ValueError("invalid_reservation_model")
        for key in ("input_usd_per_million", "output_usd_per_million"):
            spec[key] = str(_decimal(spec[key]).normalize())
        for key in ("max_input_tokens", "max_output_tokens"):
            if type(spec[key]) is not int or not 64 <= spec[key] <= 10000000:
                raise ValueError("invalid_reservation_token_limit")
        if (not isinstance(spec["metadata_sha256"], str)
                or not _SHA.fullmatch(spec["metadata_sha256"])
                or not isinstance(spec["provider"], str)
                or not re.fullmatch(r"[a-z0-9_/-]{1,80}", spec["provider"])):
            raise ValueError("invalid_reservation_metadata")
        for key in ("response_models", "response_providers"):
            if (type(spec[key]) is not list or not 1 <= len(spec[key]) <= 16
                    or any(not isinstance(v, str) or not 1 <= len(v) <= 160
                           or any(ord(c) < 32 for c in v) for v in spec[key])
                    or len(set(spec[key])) != len(spec[key])):
                raise ValueError("invalid_reservation_response_identity")
    return result


class ReservedCallGuard(MinimalCallGuard):
    """One shared allowance across product roles and separately scoped judging."""

    allowed_roles = {"online_fast", "memory", "reflection", "judge"}

    def __init__(self, path, *, max_requests, max_observed_cost_usd, reservation_policy, min_interval_seconds=0,
                 observe_transport=False, recovery_policy=None, transport_diagnostics=False,
                 connection_preparation_recovery=False, pre_application_check=None):
        self.policy = validate_reservation_policy(reservation_policy)
        self._policy_json = json.dumps(self.policy, sort_keys=True, allow_nan=False)
        self.policy_sha256 = hashlib.sha256(self._policy_json.encode()).hexdigest()
        # Never add a new policy to an old allowance, even if no fee is visible.
        prior = Path(path)
        if prior.exists():
            with sqlite3.connect(prior.resolve().as_uri() + "?mode=ro", uri=True) as db:
                try:
                    row = db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
                except sqlite3.DatabaseError:
                    raise ValueError("retained_ledger_has_no_reservation_policy") from None
                if row != (self._policy_json,):
                    raise ValueError("retained_reservation_policy_changed")
        super().__init__(path, max_requests=max_requests,
                         max_observed_cost_usd=max_observed_cost_usd, min_interval_seconds=min_interval_seconds,
                         observe_transport=observe_transport, recovery_policy=recovery_policy,
                         transport_diagnostics=transport_diagnostics,
                         connection_preparation_recovery=connection_preparation_recovery, pre_application_check=pre_application_check)
        try:
            with self._lock, self._transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS reservation_policy "
                                 "(id INTEGER PRIMARY KEY CHECK(id=1), policy TEXT NOT NULL)")
                self._db.execute("INSERT OR IGNORE INTO reservation_policy VALUES(1,?)",
                                 (self._policy_json,))
                self._db.execute("CREATE TABLE IF NOT EXISTS reservation_identity "
                                 "(id INTEGER PRIMARY KEY CHECK(id=1), ledger_id TEXT NOT NULL)")
                identity = self._db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
                if identity is None:
                    if self._db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                        raise ValueError("retained_reservation_identity_missing")
                    identity = (uuid.uuid4().hex,)
                    self._db.execute("INSERT INTO reservation_identity VALUES(1,?)", identity)
                if not re.fullmatch(r"[a-f0-9]{32}", identity[0]):
                    raise ValueError("invalid_reservation_identity")
                self.ledger_id = identity[0]
                self._check_policy()
        except BaseException:
            self._db.close()
            raise

    def _check_policy(self):
        if (json.dumps(self.policy, sort_keys=True, allow_nan=False) != self._policy_json
                or self._db.execute("SELECT policy FROM reservation_policy WHERE id=1").fetchone()
                != (self._policy_json,)
                or self._db.execute("SELECT ledger_id FROM reservation_identity WHERE id=1").fetchone()
                != (self.ledger_id,)):
            raise ValueError("reservation_policy_changed")

    def _binding(self, client):
        if not getattr(client, "bounded_transport", False):
            raise ValueError("reservation_requires_bounded_transport")
        spec = self.policy["models"].get(client.model)
        if spec is None or client.max_completion_tokens > spec["max_output_tokens"]:
            raise ValueError("model_not_in_reservation_policy")
        binding = json.loads(super()._binding(client))
        binding["reservation_policy_sha256"] = self.policy_sha256
        return json.dumps(binding, sort_keys=True, allow_nan=False)

    def attach(self, client):
        self._check_policy()
        spec = self.policy["models"].get(client.model)
        if spec is None:
            raise ValueError("model_not_in_reservation_policy")
        ceiling = {"prompt": float(spec["input_usd_per_million"]),
                   "completion": float(spec["output_usd_per_million"])}
        if (getattr(client, "reservation_max_price", None) is not None
                and client.reservation_max_price != ceiling):
            raise ValueError("client_price_ceiling_changed")
        client.reservation_max_price = ceiling
        super().attach(client)

    def _snapshot(self):
        self._check_policy()
        state = super()._snapshot()
        for row in state["attempts"]:
            if "receipt_identity" in row:
                validate_receipt_identity(row["receipt_identity"])
        reserved = sum((Decimal(row["reservation"]["reserved_cost_usd"])
                        for row in state["attempts"] if "cost" not in row["usage"]), Decimal(0))
        state.update(reserved_cost_usd=str(reserved),
                     ledger_id=self.ledger_id,
                     reservation_policy_sha256=self.policy_sha256,
                     cost_guard_scope="pre_send_reservation_with_usage_check_not_provider_invoice_guarantee")
        return state

    def _reserve(self, client, request, payload, state):
        spec = self.policy["models"][client.model]
        if payload.get("provider", {}).get("only") != [spec["provider"]]:
            raise MinimalCallStopped("reservation_provider_mismatch")
        bound = len(request.data) + FRAMING_TOKENS
        output = payload.get("max_tokens")
        if (bound > spec["max_input_tokens"] or type(output) is not int
                or not 64 <= output <= spec["max_output_tokens"]):
            raise MinimalCallStopped("reservation_token_limit_exceeded")
        amount = (Decimal(bound) * Decimal(spec["input_usd_per_million"])
                  + Decimal(output) * Decimal(spec["output_usd_per_million"])) / 1000000
        committed = Decimal(state["known_cost_usd"] or "0") + Decimal(state["reserved_cost_usd"])
        if committed + amount > Decimal(self.threshold):
            raise MinimalCallStopped("pre_send_budget_exceeded")
        return {"reservation": {"input_token_plan_bound": bound,
                "output_token_limit": output, "reserved_cost_usd": str(amount),
                "policy_sha256": self.policy_sha256,
                "token_bound_method": "full_wire_utf8_bytes_plus_8192_template_plan",
                "provider_billing_guarantee": False}}

    def _record_receipt(self, client, row, data):
        spec, reservation, usage = self.policy["models"][client.model], row["reservation"], row["usage"]

        def reject(reason):
            row["validation_error"] = reason
            raise MinimalCallStopped(reason)

        if row.get("http_status") != 200:
            reject("unexpected_http_status")
        identity = receipt_identity(data, models=spec["response_models"], providers=spec["response_providers"])
        row["receipt_identity"] = identity
        error_status = identity["error_field_status"]
        if error_status == "object":
            row["observation"] = observe_response_error(data, row["observation"], error_field_status=error_status)
            reject("provider_error_envelope")
        if data.get("model") not in spec["response_models"]:
            reject("response_model_mismatch")
        row["returned_model"] = data["model"]
        if data.get("provider") not in spec["response_providers"]:
            reject("response_provider_mismatch")
        row["returned_provider"] = data["provider"]
        if "cost" not in usage:
            reject("provider_cost_unknown")
        if any(type(usage.get(key)) is not int or usage[key] < 0
               for key in ("prompt_tokens", "completion_tokens")):
            reject("provider_token_usage_missing")
        if (usage["prompt_tokens"] > reservation["input_token_plan_bound"]
                or usage["completion_tokens"] > reservation["output_token_limit"]):
            reject("reservation_token_bound_exceeded")
        if Decimal(str(usage["cost"])) > Decimal(reservation["reserved_cost_usd"]):
            reject("reservation_cost_exceeded")
        try:
            client._reject_disabled_reasoning(data)
        except Exception:
            reject("disabled_reasoning_returned")
        choices = data.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], dict) or choices[0].get("finish_reason") != "stop"):
            reject("invalid_or_truncated_response")
        message = choices[0].get("message")
        if (type(message) is not dict or type(message.get("content")) is not str
                or not message["content"].strip() or message.get("tool_calls") not in (None, [])
                or message.get("function_call") is not None or message.get("refusal") not in (None, "")):
            reject("invalid_response_message")

