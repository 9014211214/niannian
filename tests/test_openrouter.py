"""Text model protocol, transport and explicit role configuration."""
from __future__ import annotations

import http.client
import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aria.engine.memory.persona_store import PersonaStore
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.llm import (
    OpenAILLMClient,
    OpenRouterLLMClient,
    create_llm_client,
)
class _OpenRouterStub(BaseHTTPRequestHandler):

    def do_POST(self):  # noqa: N802
        payload = json.loads(
            self.rfile.read(int(self.headers["Content-Length"])).decode()
        )
        assert self.headers["Authorization"] == "Bearer test-key"
        assert self.headers["X-OpenRouter-Title"]
        if self.path == "/chat/completions":
            assert payload["model"] == "qwen/chat-test"
            assert "models" not in payload
            assert payload["reasoning"] == {
                "effort": "low", "exclude": True
            }
            assert payload["max_completion_tokens"] == 700
            integrated = any(
                "online_turn_decision_v1" in str(
                    item.get("content") or ""
                )
                for item in payload["messages"]
            )
            if integrated:
                assert payload["tools"][0]["function"]["name"] == (
                    "aria_turn_decision"
                )
                assert payload["tool_choice"]["function"]["name"] == (
                    "aria_turn_decision"
                )
                assert payload["parallel_tool_calls"] is False
                assert "provider" not in payload
            else:
                assert "tools" not in payload
            message = (
                {
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": "aria_turn_decision",
                            "arguments": '{"response":"我在。"}',
                        }
                    }],
                }
                if integrated else {"content": "我在，慢慢说。"}
            )
            self._json({
                "model": "qwen/chat-test",
                "choices": [{"message": message}],
            })
        else:
            self.send_error(404)

    def _json(self, value):
        self._bytes(json.dumps(value).encode(), "application/json")

    def _bytes(self, value: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)

    def log_message(self, *args):
        pass


@pytest.fixture()
def openrouter_url():
    server = HTTPServer(("127.0.0.1", 0), _OpenRouterStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_openrouter_chat_and_structured_decision(openrouter_url):
    llm = OpenRouterLLMClient(
        api_key="test-key", base_url=openrouter_url, model="qwen/chat-test",
        fallback_models=["qwen/fallback-test"],
    )
    assert llm.complete([{"role": "user", "content": "你好"}]) == "我在，慢慢说。"
    assert llm.last_model == "qwen/chat-test"
    assert llm.observed_models == ["qwen/chat-test"]
    integrated = llm.complete([
        {
            "role": "system",
            "content": "协议版本：online_turn_decision_v1",
        },
        {"role": "user", "content": "测试"},
    ])
    assert json.loads(integrated)["response"] == "我在。"
    assert llm.call_count == 2
    assert llm.network_request_count == 2
    assert llm.generation_decision_count == 2


def test_openrouter_explicit_fallback_starts_primary_and_counts_requests():
    captured: list[dict] = []

    class CaptureClient(OpenRouterLLMClient):
        def _request_json(self, request):
            self._record_network_request()
            payload = json.loads(request.data.decode("utf-8"))
            captured.append(payload)
            if payload["model"] == "qwen/primary":
                self._update_diagnostic(
                    http_status=503, error_type="http_error"
                )
                raise urllib.error.HTTPError(
                    request.full_url, 503, "unavailable", {}, None
                )
            return {
                "model": "qwen/fallback",
                "choices": [{"message": {
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": "aria_turn_decision",
                            "arguments": '{"response":"我在。"}',
                        }
                    }],
                }}],
            }

    with_fallback = CaptureClient(
        api_key="x",
        base_url="https://example.test",
        model="qwen/primary",
        fallback_models=[
            "qwen/primary", "qwen/fallback", "qwen/fallback"
        ],
    )
    with_fallback.complete([{
        "role": "system",
        "content": "协议版本：online_turn_decision_v1",
    }])
    assert with_fallback.fallback_models == ["qwen/fallback"]
    assert [item["model"] for item in captured] == [
        "qwen/primary", "qwen/fallback"
    ]
    assert all("models" not in item for item in captured)
    assert with_fallback.call_count == 2
    assert with_fallback.network_request_count == 2
    assert with_fallback.generation_decision_count == 1
    diagnostic = with_fallback.last_diagnostic
    assert diagnostic["primary_failed"] is True
    assert diagnostic["fallback_used"] is True
    assert diagnostic["effective_model"] == "qwen/fallback"
    assert diagnostic["network_model_requests"] == 2
    assert [item["status"] for item in diagnostic["attempts"]] == [
        "failed", "succeeded"
    ]


@pytest.mark.parametrize("status", [401, 402, 403, 422])
def test_openrouter_does_not_fallback_for_account_or_request_errors(status):
    client = OpenRouterLLMClient(
        api_key="x",
        base_url="https://example.test",
        model="qwen/primary",
        fallback_models=["qwen/fallback"],
        role="online_fast",
    )
    client.last_diagnostic = {
        "http_status": status,
        "error_type": "http_error",
    }
    assert client._fallback_allowed() is False


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503, 529])
def test_openrouter_allows_model_related_http_fallback(status):
    client = OpenRouterLLMClient(
        api_key="x",
        base_url="https://example.test",
        model="qwen/primary",
        fallback_models=["qwen/fallback"],
        role="online_fast",
    )
    client.last_diagnostic = {
        "http_status": status,
        "error_type": "http_error",
    }
    assert client._fallback_allowed() is True


def test_online_fast_defaults_bound_timeout_tail_without_http_retry(
    monkeypatch,
):
    for name in (
        "ARIA_ONLINE_FAST_LLM_TIMEOUT",
        "ARIA_ONLINE_FAST_LLM_MAX_RETRIES",
        "ARIA_LLM_TIMEOUT",
        "ARIA_LLM_MAX_RETRIES",
        "OPENROUTER_ONLINE_FAST_TIMEOUT",
        "OPENROUTER_ONLINE_FAST_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    client = OpenRouterLLMClient(
        api_key="x",
        base_url="https://example.test",
        model="qwen/primary",
        fallback_models=["qwen/fallback"],
        role="online_fast",
    )
    assert client.timeout == 8
    assert client.max_retries == 0
    client.last_diagnostic = {
        "http_status": None,
        "error_type": "timeout",
    }
    assert client._fallback_allowed() is False


def test_llm_retries_only_transient_http_errors(monkeypatch):
    calls = 0

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "model": "vendor/state",
                "choices": [{"message": {"content": "{}"}}],
            }).encode()

    def fake_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 9
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://example.test", 429, "limited", {}, None
            )
        return _Response()

    import aria.engine.policy.llm as llm_module

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = OpenAILLMClient(
        api_key="x", base_url="https://example.test", model="vendor/state",
        role="state", timeout=9, max_retries=2,
    )
    assert client.complete([{"role": "user", "content": "x"}]) == "{}"
    assert calls == 2
    assert client.call_count == 2
    assert client.network_request_count == 2
    assert client.generation_decision_count == 1


def test_llm_retries_incomplete_provider_response_without_retaining_body(
    monkeypatch,
):
    reads = 0

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise http.client.IncompleteRead(b"private partial body")
            return json.dumps({
                "model": "vendor/state",
                "choices": [{"message": {"content": "{}"}}],
            }).encode()

    import aria.engine.policy.llm as llm_module

    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(),
    )
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    client = OpenAILLMClient(
        api_key="x",
        base_url="https://example.test",
        model="vendor/state",
        role="state",
        timeout=9,
        max_retries=2,
    )

    assert client.complete([{"role": "user", "content": "x"}]) == "{}"
    assert reads == 2
    assert client.network_request_count == 2
    assert client.last_diagnostic["error_type"] is None
    assert "partial" not in json.dumps(client.last_diagnostic)


def test_shared_llm_route_metrics_are_thread_local():
    barrier = threading.Barrier(2)

    class ConcurrentClient(OpenAILLMClient):
        def _request_json(self, _request):
            self._record_network_request()
            name = threading.current_thread().name
            barrier.wait(timeout=3)
            return {
                "model": f"vendor/{name}",
                "choices": [{"message": {"content": "我在。"}}],
            }

    client = ConcurrentClient(
        api_key="x", base_url="https://example.test", model="configured"
    )

    def run():
        client.complete([{"role": "user", "content": "你好"}])
        return threading.current_thread().name, client.last_model

    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="aria-route"
    ) as pool:
        rows = [future.result() for future in [pool.submit(run), pool.submit(run)]]
    assert all(model == f"vendor/{name}" for name, model in rows)
    assert client.call_count == 2
    assert client.network_request_count == 2
    assert client.generation_decision_count == 2


def test_failed_decision_does_not_reuse_previous_usage_or_model():
    class SuccessThenFailure(OpenAILLMClient):
        def __init__(self):
            super().__init__(
                api_key="x",
                base_url="https://example.test",
                model="configured",
                max_retries=0,
            )
            self.responses = 0

        def _request_json(self, _request):
            self._record_network_request()
            self.responses += 1
            if self.responses > 1:
                raise RuntimeError("fixture failure")
            return {
                "model": "vendor/success",
                "choices": [{"message": {"content": "我在。"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

    client = SuccessThenFailure()
    assert client.complete([{"role": "user", "content": "你好"}]) == "我在。"
    assert client.last_usage["total_tokens"] == 12
    assert client.last_model == "vendor/success"

    with pytest.raises(RuntimeError, match="fixture failure"):
        client.complete([{"role": "user", "content": "再试一次"}])

    assert client.last_usage == {}
    assert client.last_model is None


def test_openrouter_records_safe_empty_response_diagnostic():
    client = OpenRouterLLMClient(
        api_key="x",
        base_url="https://example.test",
        model="qwen/flash",
        fallback_models=[],
    )
    client.last_diagnostic = {"http_status": 200}

    with pytest.raises(ValueError, match="没有返回有效工具参数"):
        client._extract_online_decision({
            "model": "qwen/flash",
            "choices": [{
                "finish_reason": "length",
                "native_finish_reason": "length",
                "message": {"content": None, "tool_calls": []},
            }],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 420,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        })

    assert client.last_diagnostic == {
        "http_status": 200,
        "finish_reason": "length",
        "native_finish_reason": "length",
        "output_kind": "empty",
        "error_type": "missing_online_decision_tool_arguments",
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 420,
            "reasoning_tokens": 0,
        },
    }


def test_openrouter_key_selects_only_llm_factory(monkeypatch):
    for key in (
        "ARIA_LLM_API_KEY", "ALIYUN_DASHSCOPE_API_KEY", "ARIA_ASR_BASE_URL",
        "VOLC_APP_ID", "VOLC_APP_TOKEN", "ARIA_TTS_BASE_URL",
        "ARIA_VISUAL_BASE_URL", "ARIA_LLM_PROVIDER", "ARIA_ASR_PROVIDER",
        "ARIA_TTS_PROVIDER", "ARIA_VISUAL_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    default_llm = create_llm_client()
    assert isinstance(default_llm, OpenRouterLLMClient)
    assert default_llm.role == "online_fast"

def test_legacy_aria_env_still_uses_openrouter_routing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ARIA_LLM_PROVIDER", raising=False)
    for name in (
        "ARIA_ASR_PROVIDER",
        "ARIA_TTS_PROVIDER",
        "ARIA_VISUAL_PROVIDER",
        "ARIA_EMBED_PROVIDER",
        "ARIA_ASR_BASE_URL",
        "ARIA_TTS_BASE_URL",
        "ARIA_VISUAL_BASE_URL",
        "ARIA_EMBED_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARIA_LLM_API_KEY", "legacy-key")
    monkeypatch.setenv(
        "ARIA_LLM_BASE_URL", "https://openrouter.ai/api/v1"
    )
    monkeypatch.setenv(
        "ARIA_LLM_MODEL", "anthropic/claude-sonnet-4.5"
    )
    client = create_llm_client("state")
    fast = create_llm_client("online_fast")
    assert isinstance(client, OpenRouterLLMClient)
    assert client.api_key == "legacy-key"
    assert client.model == "anthropic/claude-sonnet-4.5"
    assert client.fallback_models
    assert fast.model == "qwen/qwen3.7-flash"

def test_openrouter_model_roles_are_configured_independently(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_ONLINE_MODEL", "vendor/online")
    monkeypatch.setenv(
        "OPENROUTER_ONLINE_FAST_MODEL", "vendor/online-fast"
    )
    monkeypatch.setenv("OPENROUTER_STATE_MODEL", "vendor/state")
    monkeypatch.setenv("OPENROUTER_MEMORY_MODEL", "vendor/memory")
    monkeypatch.setenv("OPENROUTER_REFLECTION_MODEL", "vendor/reflection")
    monkeypatch.setenv("OPENROUTER_JUDGE_MODEL", "vendor/judge")
    monkeypatch.setenv(
        "OPENROUTER_REFLECTION_FALLBACK_MODELS", "vendor/reflection-fallback"
    )

    online = create_llm_client("online")
    online_fast = create_llm_client("online_fast")
    state = create_llm_client("state")
    memory = create_llm_client("memory")
    reflection = create_llm_client("reflection")
    judge = create_llm_client("judge")
    assert (online.role, online.model) == ("online", "vendor/online")
    assert (online_fast.role, online_fast.model) == (
        "online_fast", "vendor/online-fast"
    )
    assert online_fast.reasoning_effort == "none"
    assert online_fast.max_completion_tokens == 420
    assert (state.role, state.model) == ("state", "vendor/state")
    assert (memory.role, memory.model) == ("memory", "vendor/memory")
    assert (reflection.role, reflection.model) == (
        "reflection", "vendor/reflection"
    )
    assert reflection.fallback_models == ["vendor/reflection-fallback"]
    assert (judge.role, judge.model) == ("judge", "vendor/judge")


def test_default_judge_is_a_different_family_from_online_fast(monkeypatch):
    for key in (
        "OPENROUTER_ONLINE_FAST_MODEL",
        "OPENROUTER_JUDGE_MODEL",
        "ARIA_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    online_fast = create_llm_client("online_fast")
    judge = create_llm_client("judge")

    assert online_fast.model.startswith("qwen/")
    assert judge.model.startswith("anthropic/")
    assert judge.fallback_models == []




