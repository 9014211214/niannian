"""生产/演示共用装配：在线快速决策与会后人物反思两条时间线。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._openrouter import (
    DEFAULT_MEMORY_MODEL,
    DEFAULT_ONLINE_FAST_MODEL,
    DEFAULT_REFLECTION_MODEL,
    OPENROUTER_BASE_URL,
    api_key as openrouter_api_key,
)
from .engine.memory.consolidator import ConsolidationResult, LLMExtractor
from .engine.memory.diagnosis import PersonaHypothesisUpdater
from .engine.memory.persona_store import PersonaStore
from .engine.model.persona import PersonaProfile
from .engine.orchestrator import Engine
from .engine.perception.external_emotion import SyntheticEmotionExperiment
from .engine.policy.llm import (
    LLMClient,
    MockLLMClient,
    OpenRouterLLMClient,
)
from .interfaces.scene import Scene
from .interfaces.skills import Skill
from .interfaces.tools import Tool


class MinimalPreviewLLMClient(MockLLMClient):
    """Explicit local placeholder, regardless of credentials in the environment."""

    def complete(self, messages: list[dict]) -> str:
        return "【本地流程示意】本轮没有调用模型，尚未生成真实陪伴回复。"


@dataclass(frozen=True)
class MinimalModelClients:
    """The only three model roles accepted by the minimal text assembly.

    This flag describes permission to invoke these clients, not evidence that a
    real model was run. Test doubles still need to be identified as fixtures by
    their caller. There is no state model, fallback model, or media client here.
    """

    online: LLMClient
    memory: LLMClient
    reflection: LLMClient
    model_calls_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_calls_enabled, bool):
            raise TypeError("model_calls_enabled must be an explicit boolean")
        if not self.model_calls_enabled and not all(
            isinstance(client, MockLLMClient)
            for client in (self.online, self.memory, self.reflection)
        ):
            raise ValueError("disabled minimal assembly only accepts local mock clients")


def create_minimal_clients(
    *,
    model_calls_enabled: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    online_model: str = DEFAULT_ONLINE_FAST_MODEL,
    memory_model: str = DEFAULT_MEMORY_MODEL,
    reflection_model: str = DEFAULT_REFLECTION_MODEL,
    reflection_return_format: str = "json_schema",
) -> MinimalModelClients:
    """Create inert local clients unless model calls are explicitly enabled.

    Merely having a key, provider, or model in the environment cannot enable a
    cloud service. Live creation only configures clients; it performs no request.
    The host must obtain its own model/fee authorization before enabling it.
    Client retries and multi-model fallback are disabled even if environment
    variables request them. Failures remain visible and require a new action.
    """
    if not isinstance(model_calls_enabled, bool):
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if not model_calls_enabled:
        return MinimalModelClients(
            online=MinimalPreviewLLMClient(role="online_fast"),
            memory=MinimalPreviewLLMClient(role="memory"),
            reflection=MinimalPreviewLLMClient(role="reflection"),
        )
    key = (api_key if api_key is not None else openrouter_api_key()).strip()
    if not key:
        raise ValueError("explicit model calls require an OpenRouter API key")
    models = (online_model, memory_model, reflection_model)
    if not all(isinstance(model, str) and model.strip() for model in models):
        raise ValueError("minimal assembly requires explicit nonempty model names")
    if reflection_return_format not in {"native", "json_schema"}:
        raise ValueError("invalid_reflection_return_format")
    if reflection_return_format == "json_schema" and reflection_model.strip() != DEFAULT_REFLECTION_MODEL:
        raise ValueError("structured reflection requires the verified default model; native must be explicit")
    clients = [
        OpenRouterLLMClient(
            role=role,
            api_key=key,
            base_url=base_url if base_url is not None else OPENROUTER_BASE_URL,
            model=model.strip(),
            fallback_models=[],
            max_retries=0,
            online_return_format="json_object" if role == "online_fast" else "native",
            bounded_transport=True,
            reflection_return_format=reflection_return_format if role == "reflection" else "native",
            timeout=240 if role == "reflection" else None,
        )
        for role, model in zip(("online_fast", "memory", "reflection"), models)
    ]
    # The integrated JSON includes affect, strategy and response fields. Keep
    # the explicit minimal default at the validated experiment's output ceiling.
    clients[0].max_completion_tokens = 1024
    clients[0].reasoning_effort = "none"
    return MinimalModelClients(
        online=clients[0], memory=clients[1], reflection=clients[2],
        model_calls_enabled=True,
    )


class _PreviewMemoryExtractor:
    """A preview must not manufacture learned facts through heuristic fallback."""

    def extract(self, ctx, persona) -> ConsolidationResult:
        return ConsolidationResult()


def create_minimal_engine(
    scene: Scene,
    persona: Optional[PersonaProfile],
    *,
    clients: MinimalModelClients | None = None,
    delivery_block: str | None = None,
    store: Optional[PersonaStore] = None,
    capabilities: Optional[set[str]] = None,
    skills: Optional[list[Skill]] = None,
    tools: Optional[list[Tool]] = None,
    external_emotion_experiment: SyntheticEmotionExperiment | None = None,
) -> Engine:
    """Reuse Engine with text inference only and explicitly bounded capabilities.

    Media, perception services, embeddings and world-knowledge adapters are not
    created here. The caller separately controls any care service it explicitly
    supplies. Online turns and sealing never invoke the two after-session roles;
    the host must reserve finalize/recover for an explicit processing action.
    Preview extraction intentionally creates no facts or personality hypotheses.
    """
    selected = clients if clients is not None else create_minimal_clients()
    memory_enabled = selected.model_calls_enabled and not isinstance(
        selected.memory, MockLLMClient
    )
    reflection_enabled = selected.model_calls_enabled and not isinstance(
        selected.reflection, MockLLMClient
    )
    return Engine(
        scene,
        persona=persona,
        llm_client=selected.online,
        delivery_block=delivery_block,
        fast_llm_client=selected.online,
        state_llm_client=None,
        store=store,
        extractor=(
            LLMExtractor(selected.memory)
            if memory_enabled else _PreviewMemoryExtractor()
        ),
        hypothesis_updater=(
            PersonaHypothesisUpdater(selected.reflection, evidence_review_protocol=True)
            if reflection_enabled and store is not None else None
        ),
        world_knowledge=None,
        capabilities=set(capabilities or ()),
        skills=skills,
        tools=tools,
        external_emotion_experiment=external_emotion_experiment,
    )

