"""The user's exact two historical exceptions, using the sealed T7.4 sender.

The old unaccepted-cost entry stays unchanged. This new entry binds the explicit
continuation authorization and writes new private artifacts inside this project.
The same public capability, source, ledger, usage and one-attempt rules apply.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_e1_compat as previous  # noqa: E402

REGISTRATION = "experiments/t74_20261001/authorized_run.json"
EXECUTION = "experiments/t74_20261001/authorized_execution.json"
FILES = frozenset({REGISTRATION, "scripts/run_e1_authorized.py", "tests/test_e1_authorized.py"})
COMPATIBILITY_SEAL = "24bc1b9373b8ee80e4dbf61a43988b258dd035c488d6bf4d365ef09a66e03192"
ACCEPTED_UNKNOWN_ATTEMPTS = tuple((identifier, "e1_001") for identifier in previous.HISTORY_IDS)
AUTHORIZATION = {
    "user_statement": "接受新增这1笔费用暂时未知（合计2笔），按原0.5美元上限继续E1",
    "historical_unknown_cost_count": 2, "retain_original_null_costs": True,
    "permits_future_unknown_cost_exception": False, "permits_automatic_retry": False,
    "new_private_directory": "artifacts/local/e1_t74_20261001_v1",
}
RunError = previous.RunError


def expected_registration(bundle):
    parent = previous.previous.legacy.read_json(bundle["root"] / previous.REGISTRATION)
    return {**deepcopy(parent), "accepted_unknown_attempts": [list(a) for a in ACCEPTED_UNKNOWN_ATTEMPTS],
        "authorization_source": "User instruction 2026-10-01 / PROGRESS task 20261001-04",
        "compatibility_seal_sha256": COMPATIBILITY_SEAL, "authorization": deepcopy(AUTHORIZATION)}


def _registration(bundle):
    value = previous.previous.legacy.read_json(bundle["root"] / REGISTRATION)
    if value != expected_registration(bundle):
        raise RunError("exact_continuation_authorization_changed")
    return value


def verify_execution(bundle):
    parent = previous.verify_execution(bundle)
    if parent["seal_sha256"] != COMPATIBILITY_SEAL:
        raise RunError("compatibility_parent_seal_changed")
    _registration(bundle)
    seal = previous.previous.legacy.read_json(bundle["root"] / EXECUTION)
    if (seal.get("schema") != "aria_t74_exact_continuation_v1"
            or seal.get("compatibility_seal_sha256") != COMPATIBILITY_SEAL
            or seal.get("seal_sha256") != previous.compat.canonical_sha256(
                {k: v for k, v in seal.items() if k != "seal_sha256"})):
        raise RunError("continuation_execution_seal_changed")
    entries = seal.get("files", [])
    if len(entries) != len(FILES) or {row["path"] for row in entries} != FILES:
        raise RunError("continuation_source_inventory_changed")
    for row in entries:
        path = previous.previous.freeze._public_file(bundle["root"], row["path"])
        if path.stat().st_size != row["bytes"] or previous.prior._sha(path) != row["sha256"]:
            raise RunError("continuation_source_changed")
    return seal


def campaign_directory(root=ROOT):
    return Path(root).absolute() / AUTHORIZATION["new_private_directory"]


def _namespace():
    # Private function globals retain the reviewed code. No old module mutation.
    namespace = {**vars(previous), "ROOT": ROOT, "_registration": _registration,
        "verify_execution": verify_execution, "campaign_directory": campaign_directory,
        "ACCEPTED_UNKNOWN_ATTEMPTS": ACCEPTED_UNKNOWN_ATTEMPTS}
    for name in ("verify_history", "preflight", "main"):
        function = getattr(previous, name)
        cloned = FunctionType(function.__code__, namespace, function.__name__,
                              function.__defaults__, function.__closure__)
        cloned.__kwdefaults__ = dict(function.__kwdefaults__ or {})
        namespace[name] = cloned
    return namespace


def verify_history(bundle):
    return _namespace()["verify_history"](bundle)


def preflight(bundle, *, online=False, opener=None):
    return _namespace()["preflight"](bundle, online=online, opener=opener)


def main(argv=None):
    return _namespace()["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
