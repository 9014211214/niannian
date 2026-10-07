"""Transport acceptance and unknown results are not confirmed notification."""
import pytest

from aria.engine.context import EngineContext
from aria.scenes.companion_elder.delivery import (
    UNGROUNDED_EXTERNAL_EVENT_REPLY,
    UNVERIFIED_ACTION_REPLY,
    guard_external_delivery_response,
)


def _reply(receipt, text="我已经通知女儿了。"):
    ctx = EngineContext(user_id="receipt-fixture", scene_id="companion_elder")
    ctx.working_memory["last_tool_result"] = receipt
    return guard_external_delivery_response(ctx, text, "嗯。", None)


def _receipt(data):
    return {"tool": "notify_family", "ok": True, "args": {"contact": "女儿"}, "data": data}


@pytest.mark.parametrize("status", [
    "unknown", "failed", "accepted", "queued", "pending", "sent", "sent_local", "delivered_local",
    "unrecognized", None, "", False, 1,
])
def test_nonterminal_or_invalid_status_cannot_claim_delivery_even_with_true_flag(status):
    assert _reply(_receipt({"status": status, "notified": True})) == UNVERIFIED_ACTION_REPLY


@pytest.mark.parametrize("data", [{}, {"notified": "true"}, {"delivered": 1}, {"executed": True}])
def test_bare_ok_or_nonterminal_flags_do_not_prove_notification(data):
    assert _reply(_receipt(data)) == UNVERIFIED_ACTION_REPLY


@pytest.mark.parametrize("data", [
    {"status": "delivered"}, {"status": "notified"}, {"notified": True}, {"delivered": True},
])
def test_terminal_status_or_explicit_legacy_boolean_still_supports_matching_target(data):
    assert _reply(_receipt(data)) == "我已经通知女儿了。"


@pytest.mark.parametrize("outer,inner", [("unknown", "delivered"), ("delivered", "unknown"), ("delivered", "notified")])
def test_conflicting_status_layers_fail_closed(outer, inner):
    receipt = _receipt({"status": inner, "notified": True})
    receipt["status"] = outer
    assert _reply(receipt) == UNVERIFIED_ACTION_REPLY


def test_terminal_receipt_still_cannot_prove_relative_read_or_action():
    assert _reply(_receipt({"status": "delivered"}), "女儿已经回复了。") == UNGROUNDED_EXTERNAL_EVENT_REPLY


def test_terminal_status_does_not_override_explicit_failure():
    assert _reply(_receipt({"status": "delivered", "delivered": False})) == UNVERIFIED_ACTION_REPLY


def test_consistent_statuses_allow_case_and_surrounding_space_normalization():
    receipt = _receipt({"status": " delivered "})
    receipt["status"] = "DELIVERED"
    assert _reply(receipt) == "我已经通知女儿了。"
