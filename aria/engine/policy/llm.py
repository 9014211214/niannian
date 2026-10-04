"""LLM 客户端 —— 策略层的「说什么」最终由它生成。

- `MockLLMClient`:无 key 也能跑(测试 / CI)。返回占位,不真生成。
- `OpenRouterLLMClient`:一个 ``OPENROUTER_API_KEY``，按 online_fast /
  memory / reflection / judge 职责独立选择主模型和回退。online 只作旧兼容；
  产品 ``process_turn`` 不装配它。
  state 角色仅保留给独立评测或迁移兼容，不进入产品在线主链路。
- `OpenAILLMClient`:其他兼容端点 / 自部署 vLLM,仅用 stdlib,
  无额外依赖。必须显式设置 ARIA_LLM_BASE_URL 与 ARIA_LLM_API_KEY。
- `create_llm_client(role)`:OpenRouter 优先；无 key 回退对应职责的 mock。

策略层只依赖 `LLMClient` 协议;换模型 / 换厂商不改引擎。
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Literal, Protocol

from ..._openrouter import (
    DEFAULT_CHAT_FALLBACK_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ONLINE_FAST_MODEL,
    DEFAULT_STATE_MODEL,
    DEFAULT_MEMORY_MODEL,
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_JUDGE_MODEL,
    api_key as openrouter_api_key,
    base_url as openrouter_base_url,
    headers as openrouter_headers,
)
from .online_decision import (
    online_decision_tool_spec,
    protocol_version,
    uses_online_decision_protocol,
)
from .online_return_channel import OnlineReturnChannel, bind_online_return_channel
from .online_json import parse_json_decision, prepare_json_messages

LLMRole = Literal[
    "online", "online_fast", "state", "memory", "reflection", "judge"
]


class LLMClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class MockLLMClient:
    """确定性占位:无 LLM key 时用。证明管线通,不代表真实回应。"""

    source_id = "mock"

    def __init__(self, role: LLMRole = "online"):
        self.role = role
        self.model = "mock"
        self.last_model = "mock"
        self.observed_models = ["mock"]

    def complete(self, messages: list[dict]) -> str:
        return "[mock-llm:未配置 OPENROUTER_API_KEY / ARIA_LLM_API_KEY,返回占位]"


class OpenAILLMClient:
    """OpenAI 兼容 chat/completions(stdlib urllib,无第三方依赖)。"""

    source_id = "configured_llm"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, temperature: float | None = None,
                 role: LLMRole = "online", timeout: float | None = None,
                 max_retries: int | None = None):
        self.role = role
        self.api_key = api_key or os.getenv("ARIA_LLM_API_KEY")
        self.base_url = (base_url or os.getenv("ARIA_LLM_BASE_URL")
                         or "http://127.0.0.1:8000/v1").rstrip("/")
        role_prefix = f"ARIA_{role.upper()}_LLM"
        self.model = (
            model
            or os.getenv(f"{role_prefix}_MODEL")
            or os.getenv("ARIA_LLM_MODEL")
            or "local-model"
        )
        # 温度可控(可复现的前提之一);bench 用低温跑,聊天默认 0.7。
        if temperature is None:
            configured = (
                os.getenv(f"{role_prefix}_TEMPERATURE")
                or os.getenv("ARIA_LLM_TEMPERATURE")
            )
            temperature = float(
                configured if configured is not None
                else (
                    "0.2" if role == "online_fast"
                    else ("0.7" if role == "online" else "0.2")
                )
            )
        self.temperature = temperature
        self._metrics_local = threading.local()
        self._metrics_lock = threading.Lock()
        self._last_model_global: str | None = None
        self._last_latency_global: float | None = None
        self._last_usage_global: dict = {}
        self._last_diagnostic_global: dict = {}
        self.observed_models: list[str] = []
        # call_count 保留历史公开名，但语义收紧为实际 HTTP 尝试次数；
        # generation_decision_count 才是上层 complete() 决策次数。
        self.call_count = 0
        self.network_request_count = 0
        self.generation_decision_count = 0
        if timeout is None:
            configured_timeout = (
                os.getenv(f"{role_prefix}_TIMEOUT")
                or os.getenv("ARIA_LLM_TIMEOUT")
            )
            timeout = float(
                configured_timeout
                if configured_timeout is not None
                else ("8" if role == "online_fast" else (
                    "45" if role == "online" else "90"
                ))
            )
        self.timeout = timeout
        if max_retries is None:
            configured_retries = (
                os.getenv(f"{role_prefix}_MAX_RETRIES")
                or os.getenv("ARIA_LLM_MAX_RETRIES")
            )
            max_retries = int(
                configured_retries
                if configured_retries is not None
                else (0 if role == "online_fast" else (
                    1 if role == "online" else 2
                ))
            )
        self.max_retries = max(0, int(max_retries))

    @property
    def last_model(self) -> str | None:
        return getattr(
            self._metrics_local, "last_model", self._last_model_global
        )

    @last_model.setter
    def last_model(self, value: str | None) -> None:
        self._metrics_local.last_model = value
        with self._metrics_lock:
            self._last_model_global = value

    @property
    def last_latency_ms(self) -> float | None:
        return getattr(
            self._metrics_local,
            "last_latency_ms",
            self._last_latency_global,
        )

    @last_latency_ms.setter
    def last_latency_ms(self, value: float | None) -> None:
        self._metrics_local.last_latency_ms = value
        with self._metrics_lock:
            self._last_latency_global = value

    @property
    def last_usage(self) -> dict:
        return getattr(
            self._metrics_local, "last_usage", self._last_usage_global
        )

    @last_usage.setter
    def last_usage(self, value: dict) -> None:
        clean = dict(value)
        self._metrics_local.last_usage = clean
        with self._metrics_lock:
            self._last_usage_global = clean

    @property
    def last_diagnostic(self) -> dict:
        return getattr(
            self._metrics_local,
            "last_diagnostic",
            self._last_diagnostic_global,
        )

    @last_diagnostic.setter
    def last_diagnostic(self, value: dict) -> None:
        clean = dict(value)
        self._metrics_local.last_diagnostic = clean
        with self._metrics_lock:
            self._last_diagnostic_global = clean

    def _update_diagnostic(self, **values) -> None:
        current = dict(self.last_diagnostic)
        current.update(values)
        self.last_diagnostic = current

    def _start_decision(self) -> None:
        self._metrics_local.current_network_requests = 0
        # A failed request has no response-level usage or effective model.
        # Clear both at the decision boundary so callers cannot accidentally
        # attribute the previous successful response to a later failure.
        self.last_usage = {}
        self.last_model = None

    def _record_network_request(self) -> None:
        current = int(getattr(
            self._metrics_local, "current_network_requests", 0
        ))
        self._metrics_local.current_network_requests = current + 1
        with self._metrics_lock:
            self.call_count += 1
            self.network_request_count += 1

    def _current_network_requests(self) -> int:
        return int(getattr(
            self._metrics_local, "current_network_requests", 0
        ))

    def _record_decision(self, elapsed_ms: float) -> None:
        self.last_latency_ms = elapsed_ms
        with self._metrics_lock:
            self.generation_decision_count += 1

    def complete(self, messages: list[dict]) -> str:
        self._start_decision()
        self.last_diagnostic = {}
        body = json.dumps({
            "model": self.model,
            "messages": self._prepare_online_messages(messages, channel="plain_json"),
            "temperature": self.temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            data = self._request_json(req)
        finally:
            self._record_decision((time.perf_counter() - started) * 1000)
        return self._extract_response(data)

    def _prepare_online_messages(
        self, messages: list[dict], *, channel: OnlineReturnChannel,
    ) -> list[dict]:
        # Offline reflection/judging and ordinary text requests retain their
        # prompts. Binding is also exposed as a pure function so prospective
        # benchmarks can reconstruct the exact current wire request.
        if (
            self.role not in {"online", "online_fast"}
            or not uses_online_decision_protocol(messages)
        ):
            return messages
        return bind_online_return_channel(messages, channel=channel)

    def _extract_response(self, data: dict) -> str:
        self._record_effective_model(data.get("model"))
        usage = data.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else {}
        choice = self._first_choice(data)
        message = choice.get("message")
        if not isinstance(message, dict):
            self._record_response_shape(
                choice, output_kind="missing_message",
                error_type="invalid_response_shape",
            )
            raise ValueError("模型响应缺少 message")
        content = message.get("content")
        if content is None or not str(content).strip():
            self._record_response_shape(
                choice, output_kind="empty",
                error_type="empty_text_response",
            )
            raise ValueError("模型没有返回可用文本")
        self._record_response_shape(
            choice, output_kind="text", error_type=None
        )
        return str(content).strip()

    def _first_choice(self, data: dict) -> dict:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            self._update_diagnostic(
                output_kind="missing_choice",
                error_type="invalid_response_shape",
            )
            raise ValueError("模型响应缺少 choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            self._update_diagnostic(
                output_kind="invalid_choice",
                error_type="invalid_response_shape",
            )
            raise ValueError("模型响应 choice 格式无效")
        return choice

    def _record_response_shape(
        self,
        choice: dict,
        *,
        output_kind: str,
        error_type: str | None,
    ) -> None:
        details = self.last_usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        self._update_diagnostic(
            finish_reason=choice.get("finish_reason"),
            native_finish_reason=choice.get("native_finish_reason"),
            output_kind=output_kind,
            error_type=error_type,
            usage={
                "prompt_tokens": self.last_usage.get("prompt_tokens"),
                "completion_tokens": self.last_usage.get(
                    "completion_tokens"
                ),
                "reasoning_tokens": details.get("reasoning_tokens"),
            },
        )

    def _request_json(self, request: urllib.request.Request) -> dict:
        """只重试幂等推理请求的瞬时错误；认证/参数错误立即暴露。"""
        minimal_control = getattr(self, "model_request_control", None)
        if minimal_control is not None:
            if getattr(self, "benchmark_request_control", None) is not None:
                raise ValueError("multiple model request controllers are not allowed")
            return minimal_control.request_json(self, request)
        # Only the benchmark CLI installs this process-local shared controller.
        # Product clients retain their existing transport and latency defaults.
        controller = getattr(self, "benchmark_request_control", None)
        if controller is not None:
            return controller.request_json(self, request)
        transient_statuses = {429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            self._record_network_request()
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    status = getattr(response, "status", None)
                    if status is None and hasattr(response, "getcode"):
                        status = response.getcode()
                    self._update_diagnostic(
                        http_status=(int(status) if status is not None else None),
                        error_type=None,
                    )
                    try:
                        return json.loads(response.read().decode("utf-8"))
                    except http.client.IncompleteRead:
                        # A provider/proxy may close an otherwise valid HTTP
                        # response mid-body. Retry the same model within this
                        # role's budget and never retain the partial body,
                        # which can contain user text or model output.
                        self._update_diagnostic(
                            http_status=None,
                            error_type="network_error",
                        )
                        if attempt >= self.max_retries:
                            raise
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # This is the provider envelope, not model-authored
                        # JSON. A malformed envelope is a transport failure.
                        self._update_diagnostic(
                            error_type="invalid_json_response"
                        )
                        if attempt >= self.max_retries:
                            raise
            except urllib.error.HTTPError as exc:
                self._update_diagnostic(
                    http_status=exc.code,
                    error_type="http_error",
                )
                retryable = exc.code in transient_statuses
                if not retryable or attempt >= self.max_retries:
                    raise
            except TimeoutError:
                self._update_diagnostic(
                    http_status=None, error_type="timeout"
                )
                if attempt >= self.max_retries:
                    raise
            except urllib.error.URLError:
                self._update_diagnostic(
                    http_status=None, error_type="network_error"
                )
                if attempt >= self.max_retries:
                    raise
            time.sleep(min(4.0, 0.5 * (2 ** attempt)))
        raise RuntimeError("unreachable")

    def _record_effective_model(self, model: object) -> None:
        if model is None or not str(model).strip():
            return
        effective = str(model)
        self.last_model = effective
        with self._metrics_lock:
            if effective not in self.observed_models:
                self.observed_models.append(effective)


class OpenRouterLLMClient(OpenAILLMClient):
    """OpenRouter 职责模型；该职责的主模型失败时才进入模型路由回退。"""

    source_id = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        temperature: float | None = None,
        role: LLMRole = "online",
        timeout: float | None = None,
        max_retries: int | None = None,
        online_return_format: Literal["native", "json_object"] = "native",
        bounded_transport: bool = False,
        reflection_return_format: Literal["native", "json_schema"] = "native",
    ):
        if online_return_format not in {"native", "json_object"}:
            raise ValueError("invalid_online_return_format")
        self.online_return_format = online_return_format
        if reflection_return_format not in {"native", "json_schema"}:
            raise ValueError("invalid_reflection_return_format")
        self.reflection_return_format = reflection_return_format
        if type(bounded_transport) is not bool:
            raise ValueError("invalid_bounded_transport")
        self.bounded_transport = bounded_transport
        self.reservation_max_price = None
        model_env = {
            "online": "OPENROUTER_ONLINE_MODEL",
            "online_fast": "OPENROUTER_ONLINE_FAST_MODEL",
            "state": "OPENROUTER_STATE_MODEL",
            "memory": "OPENROUTER_MEMORY_MODEL",
            "reflection": "OPENROUTER_REFLECTION_MODEL",
            "judge": "OPENROUTER_JUDGE_MODEL",
        }[role]
        default_model = {
            "online": DEFAULT_CHAT_MODEL,
            "online_fast": DEFAULT_ONLINE_FAST_MODEL,
            "state": DEFAULT_STATE_MODEL,
            "memory": DEFAULT_MEMORY_MODEL,
            "reflection": DEFAULT_REFLECTION_MODEL,
            "judge": DEFAULT_JUDGE_MODEL,
        }[role]
        selected_model = model or os.getenv(model_env)
        if role == "online":
            selected_model = (
                selected_model or os.getenv("OPENROUTER_CHAT_MODEL")
            )
        selected_model = selected_model or default_model
        if temperature is None:
            configured_temperature = os.getenv(
                f"OPENROUTER_{role.upper()}_TEMPERATURE"
            )
            if configured_temperature is not None:
                temperature = float(configured_temperature)
        if timeout is None:
            configured_timeout = os.getenv(
                f"OPENROUTER_{role.upper()}_TIMEOUT"
            )
            if configured_timeout is not None:
                timeout = float(configured_timeout)
        super().__init__(
            api_key=api_key or openrouter_api_key(),
            base_url=base_url or openrouter_base_url(),
            model=selected_model,
            temperature=temperature,
            role=role,
            timeout=timeout,
            max_retries=max_retries,
        )
        configured_retries = os.getenv(
            f"OPENROUTER_{role.upper()}_MAX_RETRIES"
        )
        if max_retries is None and configured_retries is not None:
            self.max_retries = max(0, int(configured_retries))
        role_fallbacks = os.getenv(
            f"OPENROUTER_{role.upper()}_FALLBACK_MODELS"
        )
        if fallback_models is not None:
            configured_fallbacks = fallback_models
        elif role_fallbacks is not None:
            # 显式空值表示禁用该职责回退。
            configured_fallbacks = [
                item.strip()
                for item in role_fallbacks.split(",")
                if item.strip()
            ]
        elif role == "judge":
            # judge 失败应判为 inconclusive，不能静默回退在线 Qwen 自评。
            configured_fallbacks = []
        else:
            configured_fallbacks = [
                item.strip()
                for item in os.getenv(
                    "OPENROUTER_FALLBACK_MODELS",
                    DEFAULT_CHAT_FALLBACK_MODEL,
                ).split(",")
                if item.strip()
            ]
        # 保持用户声明顺序，同时去重且禁止把主模型再次放进 fallback。
        # 这样显式两阶段请求不会重复尝试同一个模型。
        self.fallback_models = []
        for item in configured_fallbacks:
            candidate = str(item).strip()
            if (
                not candidate
                or candidate == self.model
                or candidate in self.fallback_models
            ):
                continue
            self.fallback_models.append(candidate)
        default_max_tokens = {
            # 综合决策的 Function Call 比普通短回复长，但仍只调用一次快模型。
            "online_fast": 420,
            "online": 700,
            "state": 900,
            "memory": 1800,
            "reflection": 2800,
            "judge": 1600,
        }[role]
        configured_max_tokens = os.getenv(
            f"OPENROUTER_{role.upper()}_MAX_COMPLETION_TOKENS"
        )
        self.max_completion_tokens = max(
            64,
            int(configured_max_tokens or default_max_tokens),
        )
        default_reasoning_effort = {
            "online_fast": "none",
            "online": "low",
            "state": "low",
            "memory": "low",
            "reflection": "high",
            "judge": "low",
        }[role]
        self.reasoning_effort = (
            os.getenv(f"OPENROUTER_{role.upper()}_REASONING_EFFORT")
            or default_reasoning_effort
        ).strip().lower()

    def complete(self, messages: list[dict]) -> str:
        self._start_decision()
        self.last_diagnostic = {}
        reflection_json = self.reflection_return_format == "json_schema"
        if reflection_json:
            from .reflection_json import validate_reflection_request
            validate_reflection_request(self, messages)
        base_payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning": {
                "effort": self.reasoning_effort,
                "exclude": True,
            },
        }
        integrated = (protocol_version(messages)
                      if self.role in {"online", "online_fast"} else None)
        bounded_online = (self.role in {"online", "online_fast"}
                          and self.online_return_format == "json_object")
        json_mode = bool(integrated and bounded_online)
        if bounded_online or self.bounded_transport:
            base_payload["max_tokens"] = base_payload.pop("max_completion_tokens")
            base_payload["reasoning"] = {"enabled": False, "exclude": True}
            base_payload["stream"] = False
            base_payload["provider"] = {
                "allow_fallbacks": False, "require_parameters": True,
                "data_collection": "deny",
            }
            # The successful experiment covered this exact default route.
            # An explicitly configured different model must not inherit Alibaba.
            if self.model == DEFAULT_ONLINE_FAST_MODEL:
                base_payload["provider"]["only"] = ["alibaba"]
            elif self.model == DEFAULT_MEMORY_MODEL:
                base_payload["provider"]["only"] = ["anthropic"]
                # The selected Sonnet endpoint does not advertise temperature.
                # Omit the unsupported field rather than weaken parameter checks.
                base_payload.pop("temperature", None)
            if self.reservation_max_price is not None:
                base_payload["provider"]["max_price"] = dict(self.reservation_max_price)
        if json_mode:
            base_payload["response_format"] = {"type": "json_object"}
        elif reflection_json:
            from .reflection_json import reflection_request_format
            base_payload["response_format"] = reflection_request_format()
        elif integrated:
            base_payload["tools"] = [online_decision_tool_spec(integrated)]
            base_payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "aria_turn_decision"},
            }
            base_payload["parallel_tool_calls"] = False
        model_chain = [self.model, *self.fallback_models]
        attempts: list[dict] = []
        started = time.perf_counter()
        try:
            for index, model in enumerate(model_chain):
                self.last_diagnostic = {}
                before_requests = self._current_network_requests()
                response_received = False
                try:
                    data = self._request_for_model(
                        base_payload, model=model
                    )
                    response_received = True
                    if bounded_online or self.bounded_transport:
                        self._reject_disabled_reasoning(data)
                    if json_mode:
                        result = self._extract_json_decision(data, integrated)
                    elif reflection_json:
                        result = self._extract_reflection_json(data)
                    else:
                        result = (self._extract_online_decision(data)
                                  if integrated else self._extract_response(data))
                except Exception:
                    control = getattr(self, "model_request_control", None)
                    if response_received and control is not None:
                        # The transport receipt can be valid while model JSON
                        # or its return channel is unusable. Direct callers
                        # must stop too, even without a Web error wrapper.
                        control.halt("model_output_failed")
                    attempts.append(self._attempt_diagnostic(
                        model=model,
                        status="failed",
                        network_requests=(
                            self._current_network_requests()
                            - before_requests
                        ),
                    ))
                    if (
                        index >= len(model_chain) - 1
                        or not self._fallback_allowed()
                    ):
                        self._finalize_attempt_diagnostic(
                            attempts, succeeded=False
                        )
                        raise
                    continue
                attempts.append(self._attempt_diagnostic(
                    model=model,
                    status="succeeded",
                    network_requests=(
                        self._current_network_requests() - before_requests
                    ),
                ))
                self._finalize_attempt_diagnostic(
                    attempts, succeeded=True
                )
                return result
            raise RuntimeError("unreachable")
        finally:
            self._record_decision(
                (time.perf_counter() - started) * 1000
            )

    def _request_for_model(self, payload: dict, *, model: str) -> dict:
        selected = dict(payload)
        selected["model"] = model
        # Bind after the final tool configuration, including an independently
        # selected no-tool benchmark arm. Never switch channel after a response.
        selected["messages"] = self._prepare_online_messages(
            selected["messages"],
            channel="native" if selected.get("tools") else "plain_json",
        )
        if (self.role in {"online", "online_fast"}
                and selected.get("response_format") == {"type": "json_object"}
                and (version := protocol_version(selected["messages"]))):
            selected["messages"] = prepare_json_messages(selected["messages"], version)
        # 每个实际请求只命名一个模型，避免 OpenRouter 的多模型 Function
        # Call 路由跳过主模型；fallback 顺序由本客户端显式执行。
        selected.pop("models", None)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                selected, ensure_ascii=False
            ).encode("utf-8"),
            headers=openrouter_headers(self.api_key),
        )
        return self._request_json(request)

    def _fallback_allowed(self) -> bool:
        error_type = self.last_diagnostic.get("error_type")
        if error_type == "http_error":
            status = self.last_diagnostic.get("http_status")
            # 认证、余额和请求权限错误对同一 OpenRouter key 不会因换模型
            # 消失；继续请求只会增加延迟。模型不存在/限流/供应商故障才允许
            # 进入同职责回退。
            return bool(
                status in {400, 404, 408, 409, 429}
                or isinstance(status, int) and status >= 500
            )
        if (
            self.role == "online_fast"
            and error_type in {"network_error", "timeout"}
        ):
            # 快速在线职责已经使用短超时。网络本身不可达或主请求已超时
            # 时，再向同一网关发第二次模型请求会放大尾延迟；直接走本地兜底。
            return False
        return error_type in {
            "network_error",
            "timeout",
            "invalid_json_response",
            "invalid_response_shape",
            "empty_text_response",
            "empty_online_decision_response",
            "invalid_online_decision_arguments",
            "missing_online_decision_tool_arguments",
        }

    def _attempt_diagnostic(
        self,
        *,
        model: str,
        status: str,
        network_requests: int,
    ) -> dict:
        diagnostic = self.last_diagnostic
        return {
            "model": model,
            "status": status,
            "network_requests": max(0, int(network_requests)),
            "http_status": diagnostic.get("http_status"),
            "error_type": diagnostic.get("error_type"),
            "finish_reason": diagnostic.get("finish_reason"),
            "output_kind": diagnostic.get("output_kind"),
        }

    def _finalize_attempt_diagnostic(
        self, attempts: list[dict], *, succeeded: bool
    ) -> None:
        current = dict(self.last_diagnostic)
        primary_failed = bool(
            attempts and attempts[0].get("status") == "failed"
        )
        current.update({
            "attempts": [dict(item) for item in attempts],
            "primary_failed": primary_failed,
            "fallback_used": len(attempts) > 1,
            "generation_decisions": 1,
            "network_model_requests": sum(
                int(item.get("network_requests") or 0)
                for item in attempts
            ),
            "effective_model": (
                self.last_model if succeeded else None
            ),
        })
        self.last_diagnostic = current

    def _reject_disabled_reasoning(self, data: dict) -> None:
        """Enforce the bounded online request's disabled-reasoning contract.

        Run before either structured or short-text extraction. Keep usage for
        accounting, but never copy reasoning text/details into diagnostics.
        Native and offline clients do not use this response boundary.
        """
        if not isinstance(data, dict):
            return  # Existing extraction owns malformed envelope diagnostics.
        self._record_effective_model(data.get("model"))
        usage = data.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else {}
        details = self.last_usage.get("completion_tokens_details")
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        violation = reasoning_tokens is not None and (
            type(reasoning_tokens) is not int or reasoning_tokens != 0
        )
        choices = data.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            reasoning = message.get("reasoning")
            reasoning_details = message.get("reasoning_details")
            if (reasoning is not None and reasoning != ""
                    or reasoning_details is not None and reasoning_details != [] and reasoning_details != ""):
                violation = True
        if violation:
            def safe_count(value):
                return value if type(value) is int and value >= 0 else None
            self._update_diagnostic(
                output_kind="reasoning_rejected",
                error_type="reasoning_disabled_violation",
                usage={
                    "prompt_tokens": safe_count(self.last_usage.get("prompt_tokens")),
                    "completion_tokens": safe_count(self.last_usage.get("completion_tokens")),
                    "reasoning_tokens": safe_count(reasoning_tokens),
                },
            )
            raise ValueError("在线回执违反禁用推理约定") from None

    def _extract_json_decision(self, data: dict, version: str) -> str:
        from .online_json_diagnostics import online_shape_diagnostic

        self._record_effective_model(data.get("model"))
        usage = data.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else {}
        choice = self._first_choice(data)
        content_checked = False
        try:
            message = choice.get("message")
            if (len(data["choices"]) != 1 or type(message) is not dict
                    or choice.get("finish_reason") != "stop"
                    or message.get("tool_calls") not in (None, [])
                    or message.get("function_call") is not None
                    or message.get("refusal") not in (None, "")):
                raise ValueError("invalid_json_return_channel")
            content_checked = True
            raw = parse_json_decision(message.get("content"), version)
        except (ValueError, TypeError, KeyError, RecursionError):
            self._record_response_shape(choice, output_kind="invalid_content_json",
                                        error_type="invalid_online_json_schema")
            diagnostic = (online_shape_diagnostic(message.get("content"), version)
                          if content_checked else {"kind": "return_channel", "issues": []})
            self._update_diagnostic(online_shape=diagnostic)
            raise ValueError("在线决策正文不满足完整JSON字段契约") from None
        self._record_response_shape(choice, output_kind="content_json", error_type=None)
        return raw

    def _extract_reflection_json(self, data: dict) -> str:
        from .reflection_json import parse_reflection_json, reflection_shape_diagnostic

        self._record_effective_model(data.get("model"))
        usage = data.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else {}
        choice = self._first_choice(data)
        try:
            message = choice.get("message")
            if (len(data["choices"]) != 1 or type(message) is not dict
                    or choice.get("finish_reason") != "stop"
                    or message.get("tool_calls") not in (None, [])
                    or message.get("function_call") is not None
                    or message.get("refusal") not in (None, "")):
                raise ValueError("invalid_reflection_return_channel")
            raw = parse_reflection_json(message.get("content"))
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            self._record_response_shape(choice, output_kind="invalid_reflection_json_schema",
                                        error_type="invalid_reflection_json_schema")
            self._update_diagnostic(reflection_shape=reflection_shape_diagnostic(error))
            raise ValueError("invalid_reflection_json_schema") from None
        self._record_response_shape(choice, output_kind="reflection_json_schema", error_type=None)
        return raw

    def _extract_online_decision(self, data: dict) -> str:
        """只取强制 Function Call 参数；正文不能补成有效原生决策。"""
        self._record_effective_model(data.get("model"))
        usage = data.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else {}
        choice = self._first_choice(data)
        message = choice.get("message")
        if not isinstance(message, dict):
            self._record_response_shape(
                choice, output_kind="missing_message",
                error_type="invalid_response_shape",
            )
            raise ValueError("在线决策响应缺少 message")
        content = message.get("content")
        has_content = content is not None and (
            not isinstance(content, str) or bool(content.strip())
        )
        calls = message.get("tool_calls")
        if calls is not None and (not isinstance(calls, list) or len(calls) > 1):
            self._record_response_shape(
                choice, output_kind="ambiguous_tool_call",
                error_type="ambiguous_online_decision_tool_calls",
            )
            raise ValueError("在线决策必须且只能返回一个工具调用")
        if calls and has_content:
            self._record_response_shape(
                choice, output_kind="mixed_return_channels",
                error_type="mixed_online_decision_response",
            )
            raise ValueError("在线决策工具调用不能混合正文返回")
        for call in calls or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                self._record_response_shape(
                    choice, output_kind="invalid_tool_call",
                    error_type="invalid_online_decision_arguments",
                )
                raise ValueError("在线决策工具调用格式无效")
            function = call.get("function") or {}
            if function.get("name") != "aria_turn_decision":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                if not str(arguments.get("response") or "").strip():
                    self._record_response_shape(
                        choice, output_kind="invalid_tool_call",
                        error_type="invalid_online_decision_arguments",
                    )
                    raise ValueError("在线决策工具参数无有效 response")
                self._record_response_shape(
                    choice, output_kind="tool_call", error_type=None
                )
                return json.dumps(arguments, ensure_ascii=False)
            if arguments is not None and str(arguments).strip():
                try:
                    parsed_arguments = json.loads(str(arguments))
                except json.JSONDecodeError:
                    self._record_response_shape(
                        choice, output_kind="invalid_tool_call",
                        error_type="invalid_online_decision_arguments",
                    )
                    raise ValueError("在线决策工具参数不是有效 JSON")
                if (
                    not isinstance(parsed_arguments, dict)
                    or not str(
                        parsed_arguments.get("response") or ""
                    ).strip()
                ):
                    self._record_response_shape(
                        choice, output_kind="invalid_tool_call",
                        error_type="invalid_online_decision_arguments",
                    )
                    raise ValueError("在线决策工具参数缺少有效 response")
                self._record_response_shape(
                    choice, output_kind="tool_call", error_type=None
                )
                return json.dumps(parsed_arguments, ensure_ascii=False)
        self._record_response_shape(
            choice,
            output_kind=(
                "text_without_tool_call"
                if content is not None and str(content).strip()
                else "empty"
            ),
            error_type="missing_online_decision_tool_arguments",
        )
        raise ValueError("在线决策没有返回有效工具参数")


def create_llm_client(role: LLMRole = "online_fast") -> LLMClient:
    """按职责创建模型客户端。

    普通表达与在线综合决策统一使用 online_fast；会后记忆和人物反思可独立
    配置。复杂回合的“本轮状态候选”和“当前怎么回复”必须在 online_fast
    的同一次调用中完成；
    “会后怎么理解这个人”由 reflection 独立完成。state 仅为旧版迁移和
    独立实验保留。judge 只用于 Benchmark，默认使用不同模型家族。
    """
    provider = os.getenv("ARIA_LLM_PROVIDER", "").strip().lower()
    if provider in {"off", "disabled", "mock"}:
        return MockLLMClient(role=role)
    if openrouter_api_key() and provider not in {"generic", "local"}:
        legacy_model = None
        if (
            not os.getenv("OPENROUTER_API_KEY", "").strip()
            and "openrouter.ai" in os.getenv(
                "ARIA_LLM_BASE_URL", ""
            ).lower()
        ):
            legacy_model = os.getenv(f"ARIA_{role.upper()}_LLM_MODEL")
            if legacy_model is None and role != "online_fast":
                legacy_model = os.getenv("ARIA_LLM_MODEL")
        return OpenRouterLLMClient(model=legacy_model, role=role)
    configured_key = os.getenv("ARIA_LLM_API_KEY")
    configured_base = os.getenv("ARIA_LLM_BASE_URL")
    if (
        configured_key
        and configured_base
        and "openrouter.ai" in configured_base.lower()
        and provider not in {"generic", "local"}
    ):
        # 兼容旧 .env：只要端点明确是 OpenRouter，就仍启用它的模型回退、
        # 来源头和职责配置；不要求用户为了升级代码而移动现有密钥。
        role_prefix = f"ARIA_{role.upper()}_LLM"
        legacy_role_model = os.getenv(f"{role_prefix}_MODEL")
        # 通用旧变量继续服务深度职责；快速表达必须显式配置或使用新的
        # OpenRouter Flash 默认，不能被旧的“所有职责同一模型”悄悄覆盖。
        if legacy_role_model is None and role != "online_fast":
            legacy_role_model = os.getenv("ARIA_LLM_MODEL")
        return OpenRouterLLMClient(
            api_key=configured_key,
            base_url=configured_base,
            model=legacy_role_model,
            role=role,
        )
    if configured_key and configured_base:
        return OpenAILLMClient(role=role)
    return MockLLMClient(role=role)
