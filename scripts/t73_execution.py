"""T7.3 generation adapter for a caller with completed reconciliation/gates.

This is not a paid CLI, campaign factory, authorization or reconciliation gate.
The caller supplies its already-qualified ledger, source check and explicit T7.2
transport. Historical unknown costs and stopped campaigns are never resolved or
reopened here. The original T7 loop durably claims before the one-attempt POST,
persists safe HTTP diagnostics, and stops on the first failure.
"""
from __future__ import annotations

from copy import deepcopy
import time
from types import FunctionType, MappingProxyType

from scripts import run_e1_experiment as legacy
from scripts import t73_policy as policy
from scripts.t72_transport import OpenRouterTransport, TransportError
from scripts.verify_experiment_freeze import canonical_sha256


class CheckedTransport:
    """Bind the sent payload and observed usage to the amended inventory."""

    def __init__(self, transport: OpenRouterTransport, bounds):
        self.transport = transport
        self.bounds = MappingProxyType(dict(bounds))

    def request(self, payload):
        # Use a private copy so caller mutation cannot change the prepared wire.
        snapshot = deepcopy(payload)
        digest = canonical_sha256(snapshot)
        if digest not in self.bounds:
            raise TransportError("request_changed", outcome_unknown=False)
        bound = self.bounds[digest]
        if type(bound) is not int or bound < 0:
            raise TransportError("invalid_prompt_bound", outcome_unknown=False)
        policy._snapshot(snapshot)
        envelope = self.transport.request(snapshot)
        if canonical_sha256(snapshot) != digest:
            raise TransportError("request_changed", receipt=envelope)
        return policy.validate_receipt(snapshot, envelope, prompt_token_bound=bound)


def run_generation(directory, ledger, bundle, transport, *, source_check, sleep=time.sleep):
    """Run the unchanged durable loop using only the amended preparation.

    Creating a qualified ledger and clearing historical budget liabilities are
    explicitly the caller's responsibility; no bypass or reset is implemented.
    """
    function = legacy.run_generation
    adapted = FunctionType(function.__code__,
        {**function.__globals__, "prepare_e1": policy.prepare_e1},
        function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    _, _, _, items = policy.prepare_e1(bundle)
    bounds = {item["request_sha256"]: item["input_token_plan_bound"] for item in items}
    checked = CheckedTransport(transport, bounds)
    return adapted(directory, ledger, bundle, checked, source_check=source_check, sleep=sleep)
