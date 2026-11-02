"""Exact owner spans and bound previews, without model semantic claims."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tests.test_explicit_sharing import fixture, action, send


def selected(service, context):
    return action(service, context, "prepare", "partial", source_id="turn_1", contact_id="contact_1",
                  selection={"start": 2, "end": 5, "unit": "unicode_codepoint"})


def test_only_selected_unicode_span_reaches_adapter(fixture):
    service, context, adapter, *_ = fixture
    context = replace(context, sources=({**context.sources[0], "text": "甲🪁乙丙丁未选择的私事。"},))
    row = selected(service, context)
    assert row["message"] == "乙丙丁" and not adapter.calls
    assert row["source"]["text"] == context.sources[0]["text"]
    with pytest.raises(ValueError): send(service, context, row)
    approved = action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    assert approved["status"] == "approved" and not adapter.calls
    result = send(service, context, row)
    assert result["status"] == "delivered_local" and result["attempts"] == 1
    assert adapter.calls[0]["message"] == "乙丙丁"
    assert not any("source" in name for name in adapter.calls[0])


@pytest.mark.parametrize("selection", [
    {}, {"start": 0, "end": 2}, {"start": True, "end": 2, "unit": "unicode_codepoint"},
    {"start": 1, "end": 1, "unit": "unicode_codepoint"},
    {"start": -1, "end": 2, "unit": "unicode_codepoint"},
    {"start": 0, "end": 999, "unit": "unicode_codepoint"},
    {"start": 0, "end": 2, "unit": "byte"},
    {"start": 0, "end": 2, "unit": "unicode_codepoint", "message": "another"},
    {"start": 0.0, "end": 2, "unit": "unicode_codepoint"}, "0:2", [0, 2],
])
def test_malformed_or_widened_spans_never_create_authority(fixture, selection):
    service, context, adapter, *_ = fixture
    with pytest.raises(ValueError):
        action(service, context, "prepare", "bad", source_id="turn_1", contact_id="contact_1", selection=selection)
    assert not service.view(context)["records"] and not adapter.calls


@pytest.mark.parametrize("tamper", ["message", "selection", "source", "recipient"])
def test_changed_preview_cannot_reuse_approval_at_final_dispatch(fixture, tamper):
    service, context, adapter, *_ = fixture
    row = selected(service, context)
    action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    original = service._dispatch
    def changed(sharing_id, current_context):
        saved = deepcopy(service._get(sharing_id, context))
        if tamper == "message": saved["message"] = context.sources[0]["text"]
        elif tamper == "selection": saved["selection"]["end"] += 1
        elif tamper == "source": saved["source"]["text"] += "新内容"
        else: saved["recipient"]["name"] = "新收件人"
        service._put(saved)
        return original(sharing_id, current_context)
    service._dispatch = changed
    result = send(service, context, row)
    assert result["status"] == "revoked" and result["attempts"] == 0 and not adapter.calls


def test_edit_unselected_source_also_expires_prior_preview(fixture):
    service, context, adapter, *_ = fixture
    row = selected(service, context)
    action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    changed = replace(context, sources=({**context.sources[0], "text": context.sources[0]["text"] + "新内容"},))
    with pytest.raises(ValueError): send(service, changed, row)
    assert not adapter.calls


def test_changed_selection_requires_new_preview_and_new_approval(fixture):
    service, context, adapter, *_ = fixture
    row = selected(service, context)
    action(service, context, "approve", "approve", id=row["id"], preview_sha256=row["preview_sha256"], confirmed=True)
    new = action(service, context, "prepare", "changed", source_id="turn_1", contact_id="contact_1",
                 selection={"start": 0, "end": 6, "unit": "unicode_codepoint"})
    assert new["preview_sha256"] != row["preview_sha256"]
    with pytest.raises(ValueError): send(service, context, row)
    with pytest.raises(ValueError): send(service, context, new, "newsend")
    action(service, context, "revoke", "revoke", id=new["id"])
    with pytest.raises(ValueError): send(service, context, new, "withdrawn")
    assert not adapter.calls
