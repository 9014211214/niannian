"""One explicit first-release assembly; historical clients remain reproducible."""
from __future__ import annotations

from dataclasses import replace

RELEASE_VERSION = "release_v1"
ONLINE_MODEL = "qwen/qwen3.8-flash"
ONLINE_PROVIDER = "alibaba"
ONLINE_VERSION = "online_turn_decision_v4_evidence"
INSTRUCTION_VERSION = "compact_response_v1"
DELIVERY_VERSION = "minimal_response_v1"
PREMODEL_VERSION = "task_response_v4"
QWEN_MAX_CANDIDATE_MODEL = "qwen/qwen3.8-max-0902"
DEFAULT_RELEASE_PROFILE = "release_deepseek_v1"


def build_release_online_route(models_raw, endpoints_raw, *, release_profile):
    """Select one explicit saved route; never infer a model from an old ledger."""
    if release_profile == "release_max_v1":
        from .engine.policy.qwen_max_structured import build_qwen_max_structured_route
        return build_qwen_max_structured_route(models_raw, endpoints_raw)
    if release_profile == "release_deepseek_v1":
        from .engine.policy.online_structured import (
            build_online_structured_route, DEEPSEEK_MODEL, DEEPSEEK_PROVIDER)
        return build_online_structured_route(models_raw, endpoints_raw,
            model=DEEPSEEK_MODEL, provider=DEEPSEEK_PROVIDER)
    raise ValueError("unsupported_release_online_route")


def create_deepseek_release_clients(*, model_calls_enabled=False, api_key=None, base_url=None,
                                   online_structured_route=None, memory_structured_route=None):
    """DeepSeek v7 online roles, with explicit same-route offline selection.

    Configuration only: fresh public capability/pricing evidence and the
    existing product guard are still required before any model call.
    """
    from .runtime import create_minimal_clients, create_confirm_judge_client
    from .engine.policy.llm import OpenRouterLLMClient
    from .engine.policy.online_structured import (
        OnlineStructuredRoute, DEEPSEEK_MODEL, DEEPSEEK_PROVIDER_NAMES)
    from .engine.policy.memory_structured import MemoryStructuredRoute, validate_memory_structured_client
    from .engine.policy.reflection_structured import validate_reflection_structured_client
    from .engine.memory.preference_scope import validate_model_scope_client
    from .engine.memory.preference_sources import SOURCE_REFS
    from .engine.model.persona import PREFERENCE_SCOPE_MODEL
    from .engine.memory.preference_candidate import VERSION as PREFERENCE_VERSION
    from .engine.memory.preference_recovery import VERSION as RECOVERY_VERSION
    from .engine.policy.max_release.sse_adapter import (
        make_provider_schema_free_reflection_client, make_provider_structured_reflection_client, enable_structured_sse)
    from .engine.policy.max_release.online_shape import make_segments_projected_online_client, enable_online_sse

    if type(model_calls_enabled) is not bool:
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if not model_calls_enabled:
        return create_minimal_clients(model_calls_enabled=False)
    if (type(online_structured_route) is not OnlineStructuredRoute
            or online_structured_route.model != DEEPSEEK_MODEL
            or online_structured_route.provider not in DEEPSEEK_PROVIDER_NAMES
            or type(memory_structured_route) is not MemoryStructuredRoute):
        raise ValueError("deepseek_release_requires_verified_deepseek_and_sonnet_routes")
    offline_options = {}
    if memory_structured_route.model == DEEPSEEK_MODEL:
        if (online_structured_route.provider != "together"
                or memory_structured_route.provider != "together"
                or online_structured_route.models_raw != memory_structured_route.models_raw
                or online_structured_route.endpoints_raw != memory_structured_route.endpoints_raw):
            raise ValueError("deepseek_offline_requires_same_together_metadata")
        offline_options = dict(memory_model=DEEPSEEK_MODEL, reflection_model=DEEPSEEK_MODEL,
                               reflection_structured_route=memory_structured_route)
    clients = create_minimal_clients(model_calls_enabled=True, api_key=api_key, base_url=base_url,
        online_model=DEEPSEEK_MODEL, online_return_format="json_schema_v1",
        online_structured_route=online_structured_route, online_decision_version="online_turn_decision_v7",
        reflection_return_format="json_schema_v3" if offline_options else "body_json",
        memory_prefix_recovery=True, reflection_itemwise_review=True, **offline_options)
    clients.online.max_completion_tokens = 4096
    clients.online.timeout = 90
    clients.online.preference_protocol_version = PREFERENCE_VERSION
    clients.online.preference_recovery_version = RECOVERY_VERSION
    clients.memory.max_completion_tokens = 4000
    clients.memory.timeout = 90
    clients.memory.reasoning_effort = "none"
    clients.memory.memory_structured_route = memory_structured_route
    clients.memory.preference_scope_protocol = PREFERENCE_SCOPE_MODEL
    clients.memory.preference_source_protocol = SOURCE_REFS
    validate_memory_structured_client(clients.memory)
    validate_model_scope_client(clients.memory)
    clients.reflection.reflection_structured_route = memory_structured_route
    clients.reflection.reflection_return_format = "json_schema_v3"
    clients.reflection.max_completion_tokens = 2800
    clients.reflection.timeout = 90
    clients.reflection.reasoning_effort = "none"
    validate_reflection_structured_client(clients.reflection)
    review = OpenRouterLLMClient(role="safety_review", model=DEEPSEEK_MODEL,
        api_key=clients.online.api_key, base_url=clients.online.base_url, bounded_transport=True,
        safety_structured_route=online_structured_route, max_retries=0, fallback_models=[], timeout=90)
    review.max_completion_tokens = 1024
    review.reasoning_effort = "none"
    online = make_segments_projected_online_client(clients.online)
    reflection = (make_provider_structured_reflection_client(clients.reflection) if offline_options
                  else make_provider_schema_free_reflection_client(clients.reflection))
    confirm = create_confirm_judge_client(clients.online)
    memory = clients.memory
    if offline_options:
        from .engine.memory.memory_contract import FACT_SOURCE_REFS
        memory.memory_fact_source_protocol = FACT_SOURCE_REFS
        online = enable_online_sse(online)
        memory, review, confirm = map(enable_structured_sse, (memory, review, confirm))
    return replace(clients, release_profile="release_deepseek_v1", safety_review=review,
        reflection=reflection, online=online, memory=memory, confirm_judge=confirm)


def create_qwen_max_compact_reflection_clients(*, model_calls_enabled=False, api_key=None,
                                              base_url=None, online_structured_route=None,
                                              memory_structured_route=None):
    """Explicit v3 wire and model-authored preference scope; no default promotion.

    Uses the same Max/Sonnet model, provider and fixed physical limits. The
    caller's guarded real executor must reserve its actual new wire payload.
    Construction performs no request and grants no live effect qualification.
    """
    from .engine.policy.reflection_structured import validate_reflection_structured_client
    clients = create_qwen_max_preference_candidate_clients(
        model_calls_enabled=model_calls_enabled, api_key=api_key, base_url=base_url,
        online_structured_route=online_structured_route, memory_structured_route=memory_structured_route)
    if not model_calls_enabled:
        return clients
    clients.reflection.reflection_return_format = "json_schema_v3"
    from .engine.model.persona import PREFERENCE_SCOPE_MODEL
    from .engine.memory.preference_scope import validate_model_scope_client
    clients.memory.preference_scope_protocol = PREFERENCE_SCOPE_MODEL
    validate_model_scope_client(clients.memory)
    validate_reflection_structured_client(clients.reflection)
    return clients


def create_qwen_max_preference_candidate_clients(*, model_calls_enabled=False, api_key=None,
                                               base_url=None, online_structured_route=None,
                                               memory_structured_route=None):
    """Explicit scoped confirmation and strict reflection; no default promotion."""
    from .engine.memory.preference_candidate import VERSION, validate_preference_candidate_client
    from .engine.policy.reflection_structured import validate_reflection_structured_client
    clients = create_qwen_max_memory_structured_candidate_clients(
        model_calls_enabled=model_calls_enabled, api_key=api_key, base_url=base_url,
        online_structured_route=online_structured_route, memory_structured_route=memory_structured_route)
    if not model_calls_enabled:
        return clients
    clients.online.preference_protocol_version = VERSION
    validate_preference_candidate_client(clients.online)
    clients.reflection.reflection_structured_route = memory_structured_route
    clients.reflection.reflection_return_format = "json_schema_v2"
    clients.reflection.max_completion_tokens = 2800
    clients.reflection.timeout = 90
    clients.reflection.reasoning_effort = "none"
    validate_reflection_structured_client(clients.reflection)
    return clients


def engine_options():
    return {"safety_delivery_version": DELIVERY_VERSION,
            "safety_response_instruction_version": INSTRUCTION_VERSION,
            "premodel_safety_version": PREMODEL_VERSION}


def create_release_clients(*, model_calls_enabled=False, api_key=None, base_url=None,
                           online_structured_route=None):
    """Configure the reviewed bundle. No fetch or model call occurs here.

    A normal CLI live launch supplies the public metadata it reviewed with its
    reservation policy; an embedded host passes the same typed route directly.
    Preview keeps its explicitly labelled, inert local flow.
    """
    from .runtime import create_minimal_clients
    if type(model_calls_enabled) is not bool:
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if not model_calls_enabled:
        return create_minimal_clients(model_calls_enabled=False)
    from .engine.policy.online_structured import OnlineStructuredRoute
    if (type(online_structured_route) is not OnlineStructuredRoute
            or online_structured_route.model != ONLINE_MODEL
            or online_structured_route.provider != ONLINE_PROVIDER):
        raise ValueError("release_v1_requires_verified_qwen38_alibaba_route")
    clients = create_minimal_clients(
        model_calls_enabled=True, api_key=api_key, base_url=base_url,
        online_model=ONLINE_MODEL, online_return_format="json_schema_v1",
        online_structured_route=online_structured_route, online_decision_version=ONLINE_VERSION,
        reflection_return_format="body_json",
    )
    clients.online.max_completion_tokens = 4096
    clients.online.timeout = 90
    return replace(clients, release_profile=RELEASE_VERSION)


def create_candidate_clients(*, model_calls_enabled=False, api_key=None, base_url=None,
                             online_structured_route=None, offline_reasoning_route=None,
                             online_decision_version="online_turn_decision_v5_bound"):
    """Explicit release-v2 candidate; construction never starts a model call.

    Historical release_v1 remains reproducible. The optional Argon reflection
    route is selected from verified public metadata, not inferred from a name.
    """
    from .runtime import create_minimal_clients
    from .engine.policy.llm import OpenRouterLLMClient
    from .engine.policy.online_structured import OnlineStructuredRoute
    from .engine.policy.online_decision import BOUND_DECISION_VERSIONS
    from ._openrouter import DEFAULT_REFLECTION_MODEL
    if type(model_calls_enabled) is not bool:
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if online_decision_version not in BOUND_DECISION_VERSIONS:
        raise ValueError("candidate_online_version_not_supported")
    if not model_calls_enabled:
        return create_minimal_clients(model_calls_enabled=False)
    if (type(online_structured_route) is not OnlineStructuredRoute
            or online_structured_route.model != ONLINE_MODEL
            or online_structured_route.provider != ONLINE_PROVIDER):
        raise ValueError("release_v2_requires_verified_qwen38_alibaba_route")
    reflection_model = (offline_reasoning_route.model if offline_reasoning_route is not None
                        else DEFAULT_REFLECTION_MODEL)
    clients = create_minimal_clients(
        model_calls_enabled=True, api_key=api_key, base_url=base_url,
        online_model=ONLINE_MODEL, online_return_format="json_schema_v1",
        online_structured_route=online_structured_route, online_decision_version=online_decision_version,
        reflection_model=reflection_model, reflection_return_format="body_json",
        offline_reasoning_route=offline_reasoning_route,
        memory_prefix_recovery=True, reflection_itemwise_review=True)
    clients.online.max_completion_tokens = 4096
    clients.online.timeout = 90
    clients.memory.max_completion_tokens = 4000
    review = OpenRouterLLMClient(
        role="safety_review", model=ONLINE_MODEL, api_key=clients.online.api_key,
        base_url=clients.online.base_url, bounded_transport=True,
        safety_structured_route=online_structured_route, max_retries=0, fallback_models=[], timeout=90)
    review.max_completion_tokens = 1024
    review.reasoning_effort = "none"
    return replace(clients, release_profile="release_v2", safety_review=review)


def create_qwen_max_memory_structured_candidate_clients(*, model_calls_enabled=False, api_key=None,
                                                       base_url=None, online_structured_route=None,
                                                       memory_structured_route=None):
    """Explicit same-Sonnet memory schema; original Max/default factories stay reproducible.

    Metadata proves advertised support only. No request or live qualification
    is created here. Online, safety and reflection use the unchanged Max bundle.
    """
    from .engine.policy.llm import OpenRouterLLMClient
    from .engine.policy.memory_structured import MemoryStructuredRoute, validate_memory_structured_client
    if type(model_calls_enabled) is not bool:
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if not model_calls_enabled:
        return create_qwen_max_candidate_clients()
    if (type(memory_structured_route) is not MemoryStructuredRoute
            or memory_structured_route.model != "anthropic/claude-sonnet-5"):
        raise ValueError("memory_candidate_requires_verified_sonnet_anthropic_route")
    clients = create_qwen_max_candidate_clients(model_calls_enabled=True, api_key=api_key,
        base_url=base_url, online_structured_route=online_structured_route)
    memory = OpenRouterLLMClient(role="memory", model=memory_structured_route.model,
        api_key=clients.memory.api_key, base_url=clients.memory.base_url, bounded_transport=True,
        max_retries=0, fallback_models=[], timeout=90, memory_prefix_recovery=True)
    memory.max_completion_tokens = 4000
    memory.reasoning_effort = "none"  # Wire remains enabled:false, never effort:none.
    memory.memory_structured_route = memory_structured_route
    validate_memory_structured_client(memory)
    return replace(clients, memory=memory)


def create_semantic_candidate_clients(**kwargs):
    """Explicit, unqualified v6 protocol on the original v2 model/route bundle.

    No default is promoted, model changed, or call made by construction.
    """
    from .engine.policy.semantic_delivery import VERSION
    if "online_decision_version" in kwargs:
        raise ValueError("semantic_candidate_version_is_fixed")
    return create_candidate_clients(**kwargs, online_decision_version=VERSION)


def create_qwen_max_candidate_clients(*, model_calls_enabled=False, api_key=None,
                                      base_url=None, online_structured_route=None):
    """Explicit Max/v6 candidate; no default promotion or live qualification.

    The supplied public metadata must bind the dated Max model to Alibaba.
    Max requires minimal reasoning. The visible wire limits stay 4096/1024;
    the reservation guard separately allows for documented reasoning overhead.
    Both online roles use that exact route; the original Sonnet offline roles,
    protocols and limits remain unchanged. Construction never sends a request;
    the caller still needs a fresh reservation policy and execution admission.
    """
    from .runtime import create_minimal_clients
    from .engine.policy.llm import OpenRouterLLMClient
    from .engine.policy.qwen_max_structured import QwenMaxStructuredRoute, validate_qwen_max_client
    from .engine.policy.semantic_delivery import VERSION

    if type(model_calls_enabled) is not bool:
        raise TypeError("model_calls_enabled must be an explicit boolean")
    if not model_calls_enabled:
        return create_minimal_clients(model_calls_enabled=False)
    if (type(online_structured_route) is not QwenMaxStructuredRoute
            or online_structured_route.model != QWEN_MAX_CANDIDATE_MODEL
            or online_structured_route.provider != ONLINE_PROVIDER):
        raise ValueError("max_candidate_requires_verified_qwen38_max_0902_alibaba_route")
    clients = create_minimal_clients(
        model_calls_enabled=True, api_key=api_key, base_url=base_url,
        online_model=QWEN_MAX_CANDIDATE_MODEL, online_return_format="json_schema_v1",
        online_structured_route=online_structured_route, online_decision_version=VERSION,
        reflection_return_format="body_json",
        memory_prefix_recovery=True, reflection_itemwise_review=True)
    clients.online.max_completion_tokens = 4096
    clients.online.timeout = 90
    clients.memory.max_completion_tokens = 4000
    review = OpenRouterLLMClient(
        role="safety_review", model=QWEN_MAX_CANDIDATE_MODEL, api_key=clients.online.api_key,
        base_url=clients.online.base_url, bounded_transport=True,
        safety_structured_route=online_structured_route, max_retries=0, fallback_models=[], timeout=90)
    review.max_completion_tokens = 1024
    review.reasoning_effort = online_structured_route.effort
    validate_qwen_max_client(clients.online)
    validate_qwen_max_client(review)
    return replace(clients, release_profile="release_v2", safety_review=review)


def create_max_release_clients(*, model_calls_enabled=False, api_key=None, base_url=None,
                               online_structured_route=None, memory_structured_route=None,
                               online_decision_version="online_turn_decision_v7"):
    """Explicit installable assembly; construction never grants effects or spending."""
    from .engine.policy.online_decision import SEMANTIC_DECISION_VERSIONS
    if online_decision_version not in SEMANTIC_DECISION_VERSIONS:
        raise ValueError("unsupported_max_release_online_version")
    clients=create_qwen_max_compact_reflection_clients(model_calls_enabled=model_calls_enabled,
        api_key=api_key,base_url=base_url,online_structured_route=online_structured_route,
        memory_structured_route=memory_structured_route)
    if not model_calls_enabled:return clients
    from .engine.policy.max_release.sse_adapter import make_provider_schema_free_reflection_client
    from .engine.policy.max_release.online_shape import make_segments_projected_online_client
    from .engine.memory.preference_recovery import VERSION as RECOVERY_VERSION
    clients.online.online_decision_version = online_decision_version
    clients=replace(clients,reflection=make_provider_schema_free_reflection_client(clients.reflection),
        online=make_segments_projected_online_client(clients.online),
        release_profile="release_max_v1")
    clients.online.preference_recovery_version=RECOVERY_VERSION
    if online_decision_version == "online_turn_decision_v7":
        from .runtime import create_confirm_judge_client
        clients = replace(clients, confirm_judge=create_confirm_judge_client(clients.online))
    for client in (clients.online,clients.memory,clients.reflection,clients.safety_review):client.timeout=90
    return clients
