"""Read-only E1 tool-choice capability diagnosis; no keys or inference requests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_e1_t73 as wirecheck  # noqa: E402
from scripts import run_e1_t71 as previous  # noqa: E402
from scripts import t7_transport, t73_policy  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints"
MODEL = "qwen/qwen3.7-flash"
CANONICAL_MODEL = "qwen/qwen3.7-flash-20260727"
PROVIDER = "alibaba"
MAX_METADATA_BYTES = 2 * 1024 * 1024


class CheckError(ValueError):
    """A fixed code, never the contents of a file or remote error."""


def assess_tool_choice(payload, endpoint_document):
    """Check both the parameter name and the exact requested tool-choice mode."""
    result = {"supported": False, "required_mode": None, "advertised_support": None,
              "actual_payload_sha256": previous.freeze.canonical_sha256(payload),
              "blocking_reason": "invalid_tool_choice_request", "ready_to_send": False}
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        return result
    if (not isinstance(payload.get("provider"), dict)
            or payload["provider"].get("only") != [PROVIDER]):
        return result
    choice = payload.get("tool_choice")
    if isinstance(choice, str) and choice in {"none", "auto", "required"}:
        mode = choice
    elif (isinstance(choice, dict) and set(choice) == {"type", "function"}
          and choice.get("type") == "function" and isinstance(choice.get("function"), dict)
          and set(choice["function"]) == {"name"} and isinstance(choice["function"]["name"], str)
          and choice["function"]["name"]):
        mode = "function"
    else:
        return result
    result["required_mode"] = mode
    result["blocking_reason"] = "endpoint_identity_unverified"
    try:
        data = endpoint_document["data"]
        selected = [row for row in data["endpoints"] if row["tag"] == PROVIDER]
        if data["id"] != MODEL or len(selected) != 1:
            return result
        endpoint = selected[0]
        if (endpoint["model_id"] != MODEL or endpoint["provider_name"].lower() != PROVIDER
                or not isinstance(endpoint.get("name"), str)
                or not endpoint["name"].endswith(" | " + CANONICAL_MODEL)):
            return result
        fields = endpoint["supported_parameters"]
        if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
            raise ValueError
        result["blocking_reason"] = "tool_choice_parameter_support_unverified"
        if "tool_choice" not in fields or "tools" not in fields:
            return result
        result["blocking_reason"] = "tool_choice_mode_support_unverified"
        modes = endpoint.get("supports_tool_choice")
        supported = modes.get(mode) if isinstance(modes, dict) else None
        if type(supported) is not bool:
            return result
        result["advertised_support"] = supported
        if not supported:
            result["blocking_reason"] = "tool_choice_mode_not_supported"
            return result
        result.update(supported=True, blocking_reason=None)
        return result
    except (ValueError, TypeError, KeyError, AttributeError):
        result["blocking_reason"] = "invalid_endpoint_metadata"
        return result


def check_inventory(bundle, endpoint_document):
    """Bind the diagnosis to all 204 unchanged T7.3 positions, without sending."""
    wirecheck.verify_repair(bundle)
    _, _, prepared, items = t73_policy.prepare_e1(bundle)
    assessed = {}
    for trial in prepared.values():
        result = assess_tool_choice(trial["wire"], endpoint_document)
        assessed[result["actual_payload_sha256"]] = result
    if (len(items) != 204 or len(assessed) != 65
            or any(item["request_sha256"] not in assessed for item in items)):
        raise CheckError("amended_inventory_changed")
    positions = [assessed[item["request_sha256"]] for item in items]
    reasons = Counter(row["blocking_reason"] for row in positions if not row["supported"])
    return {"schema": "aria_e1_tool_choice_diagnosis_v1", "fixed_positions": 204,
        "distinct_payloads_checked": len(assessed),
        "linked_inventory_sha256": previous.freeze.canonical_sha256(items),
        "requested_modes": sorted({row["required_mode"] for row in positions
                                    if row["required_mode"] is not None}),
        "supported_positions": sum(row["supported"] for row in positions),
        "blocked_positions": sum(not row["supported"] for row in positions),
        "blocking_reason_counts": dict(reasons), "ready_to_send": False,
        "diagnosis_only": True, "wire_changes": [], "model_requests": 0,
        "credentials_read": False, "historical_http_cause_confirmed": False}


def read_metadata_file(path):
    path = Path(path).absolute()
    if path.suffix.lower() != ".json":
        raise CheckError("metadata_json_file_required")
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (path, *path.parents)):
        raise CheckError("metadata_link_rejected")
    if not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise CheckError("metadata_file_unavailable")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise CheckError("metadata_file_too_large")
    return t7_transport._decode(raw), hashlib.sha256(raw).hexdigest()


def fetch_endpoint(*, opener=None, enable_network=False):
    if opener is None and enable_network is not True:
        raise CheckError("network_disabled")
    request = Request(ENDPOINT, method="GET", headers={
        "Accept": "application/json", "Accept-Encoding": "identity"})
    raw = t7_transport._exchange(request, opener=opener or t7_transport._opener(), timeout=20,
                                 limit=MAX_METADATA_BYTES)
    return t7_transport._decode(raw), hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--metadata-file", help="Saved public endpoint JSON; no automatic network")
    source.add_argument("--online-preflight", action="store_true", help="One public GET, no credentials")
    args = parser.parse_args(argv)
    try:
        if not args.metadata_file and not args.online_preflight:
            raise CheckError("metadata_file_required")
        bundle = previous.load_bundle(ROOT)
        wirecheck.verify_repair(bundle)
        if args.online_preflight:
            document, digest = fetch_endpoint(enable_network=True)
        else:
            document, digest = read_metadata_file(args.metadata_file)
        report = check_inventory(bundle, document)
        report.update(ok=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
            metadata_response_sha256=digest, source_url=ENDPOINT,
            metadata_scope="live_public_get" if args.online_preflight else "saved_public_snapshot")
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 2 if report["blocked_positions"] else 0
    except Exception as error:
        reason = str(error) if isinstance(error, CheckError) else "tool_choice_check_failed"
        print(json.dumps({"ok": False, "reason": reason, "ready_to_send": False,
                          "credentials_read": False, "model_requests": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
