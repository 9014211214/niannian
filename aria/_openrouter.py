"""Aria 的 OpenRouter 统一默认值与请求头。

模型名集中在这里，避免聊天、视觉、ASR、TTS 各写一套容易漂移的配置。
所有默认值都可以用对应的 ``OPENROUTER_*_MODEL`` 环境变量覆盖。
"""
from __future__ import annotations

import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ONLINE_FAST_MODEL = "qwen/qwen3.7-flash"
# online 是旧兼容/独立评测角色；产品 process_turn 只装配 online_fast。
# 即使旧调用仍创建 online，默认也不能悄悄升级到会后深度模型。
DEFAULT_ONLINE_MODEL = DEFAULT_ONLINE_FAST_MODEL
# 迁移兼容/独立实验角色；产品在线主链路不创建或调用它。
DEFAULT_STATE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_MEMORY_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_REFLECTION_MODEL = "anthropic/claude-sonnet-5"
# 在线回复由 Qwen 生成；默认 judge 用不同模型家族，避免同族自评偏差。
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ROLE_FALLBACK_MODEL = "qwen/qwen3.5-plus-20260420"

# 兼容既有导入和环境变量命名。
DEFAULT_CHAT_MODEL = DEFAULT_ONLINE_MODEL
DEFAULT_CHAT_FALLBACK_MODEL = DEFAULT_ROLE_FALLBACK_MODEL
DEFAULT_VISION_MODEL = "qwen/qwen3.5-plus-20260420"
DEFAULT_ASR_MODEL = "qwen/qwen3-asr-flash-2026-02-10"
DEFAULT_TTS_MODEL = "qwen/qwen-audio-3.0-tts-plus"
DEFAULT_TTS_VOICE = "longanlingxin"
DEFAULT_EMBED_MODEL = "qwen/qwen3-embedding-8b"


def api_key() -> str:
    configured = os.getenv("OPENROUTER_API_KEY", "").strip()
    if configured:
        return configured
    # 早期 Aria 版本把同一 OpenRouter 凭据放在 ARIA_LLM_*。只有端点
    # 明确属于 OpenRouter 才继承，避免把任意自建服务密钥误发给云端音频、
    # 视觉或向量接口。
    legacy_base = os.getenv("ARIA_LLM_BASE_URL", "").strip()
    if "openrouter.ai" in legacy_base.lower():
        return os.getenv("ARIA_LLM_API_KEY", "").strip()
    return ""


def base_url() -> str:
    configured = os.getenv("OPENROUTER_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    legacy = os.getenv("ARIA_LLM_BASE_URL", "").strip()
    if "openrouter.ai" in legacy.lower():
        return legacy.rstrip("/")
    return OPENROUTER_BASE_URL


def headers(key: str, *, content_type: str = "application/json") -> dict[str, str]:
    result = {
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
    }
    referer = os.getenv("OPENROUTER_SITE_URL", "").strip()
    if referer:
        result["HTTP-Referer"] = referer
    result["X-OpenRouter-Title"] = (
        os.getenv("OPENROUTER_APP_NAME") or "Aria Elder Companion"
    ).strip()
    return result
