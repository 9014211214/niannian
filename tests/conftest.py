"""Global test isolation: local developer credentials never affect pytest.

Every network-facing test supplies an explicit fake key/endpoint and a local stub.
Removing inherited credentials also prevents an offline test from accidentally making
a paid request merely because the developer has configured the project ``.env``.
"""
from __future__ import annotations

import pytest


# These tests bind the exact T6/T7 runtime AND its recorded Python/dependency
# versions. They are exercised against the preserved source, never re-sealed
# against new product code. Pure component tests in these modules still run.
_FROZEN_INTEGRATION_MODULES = {
    "test_e1_execution.py", "test_e1_t71.py", "test_e1_t72.py",
    "test_e1_t73.py", "test_e1_t73_run.py", "test_e1_tool_choice.py",
    "test_e1_compat.py", "test_e1_authorized.py", "test_e1_carrier_run.py",
    "test_e1_json_carrier.py", "test_e1_json_mode.py", "test_e1_json_mode_run.py",
    "test_e1_json_schema_mode.py", "test_e1_json_schema_mode_run.py",
    "test_t73_policy.py", "test_t73_execution.py", "test_t74_execution.py",
    "test_t74_run.py",
}


def pytest_collection_modifyitems(items):
    for item in items:
        implicit_frozen_cli = (item.path.name == "test_e1_tool_choice.py"
            and item.name == "test_cli_saved_capabilities_block_without_printing_source_text")
        if (implicit_frozen_cli or (item.path.name in _FROZEN_INTEGRATION_MODULES
                and {"bundle", "source_bundle"}.intersection(item.fixturenames))):
            item.add_marker(pytest.mark.skip(reason=(
                "Historical T6/T7 source-bound integration: run the preserved frozen "
                "source with its recorded environment; not a current-product test.")))


_EXTERNAL_CONFIGURATION = {
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ARIA_LLM_API_KEY",
    "ARIA_LLM_BASE_URL",
    "ARIA_LLM_PROVIDER",
    "ALIYUN_DASHSCOPE_API_KEY",
    "DASHSCOPE_API_KEY",
    "ALIYUN_BAILIAN_WORKSPACE_ID",
    "ALIYUN_WORKSPACE_ID",
    "DASHSCOPE_WORKSPACE_ID",
    "VOLC_APP_ID",
    "VOLC_APP_TOKEN",
    "VOLC_TTS_ACCESS_KEY",
    "VOLC_TTS_API_KEY",
    "VOLC_TTS_APP_ID",
    "VOLC_API_KEY",
    "VOLCENGINE_API_KEY",
    "ARIA_ASR_BASE_URL",
    "ARIA_ASR_PROVIDER",
    "ARIA_TTS_BASE_URL",
    "ARIA_TTS_PROVIDER",
    "ARIA_EMOTION_BASE_URL",
    "ARIA_VISUAL_API_KEY",
    "ARIA_VISUAL_BASE_URL",
    "ARIA_VISUAL_PROVIDER",
    "ARIA_EMBED_API_KEY",
    "ARIA_EMBED_BASE_URL",
    "ARIA_EMBED_PROVIDER",
    "ARIA_CARE_WEBHOOK_TOKEN",
    "ARIA_CARE_WEBHOOK_URL",
    "ARIA_OMNI_REALTIME_PROVIDER",
    "ARIA_PROFILE_KEY",
    "ARIA_VOICE_SERVICE_TOKEN",
    "ARIA_WEB_DEMO_TOKEN",
}


@pytest.fixture(autouse=True)
def _isolate_external_configuration(monkeypatch):
    """Start every test without inherited credentials or service endpoints."""
    for name in _EXTERNAL_CONFIGURATION:
        monkeypatch.delenv(name, raising=False)
    yield
