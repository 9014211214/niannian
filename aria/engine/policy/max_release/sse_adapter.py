"""Explicit provider-schema-free reflection with unchanged v3 local validation.

Endpoint capability/price evidence is retained as a reference; it never means
this transport sent a provider schema. Historical factories/guards stay exact.
"""
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import json

from aria.engine.policy.llm import OpenRouterLLMClient
from aria.engine.policy.minimal_calls import MinimalCallStopped
from aria.engine.policy.reflection_structured import validate_reflection_structured_client
from aria.engine.policy.reflection_compact import (
    compact_reflection_request_format, compact_reflection_schema,
    indexed_compact_reflection_request_format, indexed_compact_reflection_schema,
    indexed_compact_reflection_system_prompt,
)
from aria.engine.policy.reflection_json import validate_reflection_request
from .reflection_sse import (VERSION as STREAM_VERSION, SSEPublicCapture,
    STRUCTURED_STREAM_VERSION, STRUCTURED_ROLES, structured_sse_contract)
from .reflection_wrapper import VERSION as WRAPPER_VERSION, reflection_content_projection

VERSION = 'provider_schema_free_reflection_v3'
PROMPT_VERSION = 'reflection_compact_v3_visible_schema_v3'
STRUCTURED_VERSION = 'provider_structured_reflection_v4'
STRUCTURED_PROMPT_VERSION = 'reflection_compact_v4_provider_schema_v1'


def model_visible_schema_text():
    return json.dumps(compact_reflection_schema(), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def schema_free_reflection_system_prompt():
    """Describe the current wire once, without translating a historical prompt.

    The compact codec, local validators and historical explicit prompts are
    unchanged. This route's visible schema remains the sole field/type source.
    """
    prompt = """你只做会后人物证据审核，不聊天、诊断、执行动作或授予权限。
依据本次非排除用户原话和实际交付记录，审核已有推断或提出有范围、可推翻的候选。
明确交流要求由 memory 提议并经本人确认；偏好、分享授权、撤销和护理规则不自动成为人物推断。

只输出一个完整 JSON 对象，顶层仅有 protocol_version、reviews、changes。
protocol_version 固定为字符串 "2.0"。reviews 为数组，可为空，最多 6 项；changes 为数组，最多 1 项。
字段、类型、枚举以末尾 Schema 为准；同一对象的每个键只写一次，不加解释、Markdown或额外字段。
没有可审核的人物证据时输出 {"protocol_version":"2.0","reviews":[],"changes":[]}。
不要为填满数组写占位 review；有真实但关系不明的证据可以记录 unknown，不补造 change。

先完成 reviews，再决定 changes：
1. 优先审核本人对旧主张的纠正和反证，其次审核本次实际互动反应。普通事实交给 memory，
无需每句话都建 review。保留原话的事情、关系、紧急性和阶段，不外推为所有场景或稳定人格；
不从年龄、性别或一次情绪推人格，不把他人或影视角色的偏好算作本人的证据。
历史已确认偏好只读，不替本人改写或撤回，也不重复计分。操作声明通常为 context_only。
2. evidence 引用支持判断的本人原话，event_evidence 引用识别事件或当前纠正的原话。
两者各含 1..3 项，turn_index 是本次非排除用户轮次的整数索引，evidence_quote 是该轮
连续逐字子串（2..240 字符）。不得改写、拼接、引用助手话语或在同一数组重复。
纠正旧事时，event_evidence 可引用本次重述的旧事，evidence 必须包含本次新纠正。
旧 reason/summary 不是本人事实；重贴或重解释旧句不是新事件。事件身份、去重和计数由代码维护，
不输出事件分类、事件 ID 或签名。reason 和 event_summary 各写一句非空说明，最多 240 字符。
3. applicability 比较证据与主张条件：相符为 within_scope，不同为 outside_scope，
主体或条件不明为 unknown。新主张比较 atomic_claim.conditions，没有旧身份不等于 unknown。
忙碌、噪声中的短答不是“安静且空闲时不愿交流”的反例。
relation 使用 support/counter/neutral/unknown。本人明确否定旧主张的一部分，为原主张记范围内
counter；不因另一部分成立而改判 support，不把旧 claim 改成相反观点。范围外、不知道、没试过
不等于反例。同一主张有真正反例和支持时优先反例。有据反证或明确纠正不要求证明旧事日期。
4. 正向新增支持只来自本会话真实交付后的 observed_response：核对紧邻前一轮 Aria 的实际回复，
evidence 和 event_evidence 引用随后的本人响应，不把对他人的反应、往事或计划当作本次反应。
只记观察到的关联，不声称因果。原独立事件数量与状态阈值保持。
stated_preference 可保留偏好观察，但不据此新增正向人物 change；首次往事、repeated_pattern
猜测只作有范围观察。同一事件的多种讲述不是多个独立行为。

对象填写：
- 仅以下八处使用零或一个元素的数组：review.hypothesis_id/claim_quote/scope_quote，
  change.id/scope/lineage/target_value，atomic_claim.outcome。无值用 []，有值用 [值]；
  空字符串用 [""]、数字零用 [0]。不写 null、不放多个元素，其余字段按 Schema 类型填写。
- review.id 用本次唯一的 1..40 个 ASCII 字母、数字、下划线或短横线。新观察的三个身份槽
  都用 []；已有主张分别用 [原 id]、[完整原 claim]、[完整原 scope]，不改身份、原文和范围。
- change.review_id 绑定实际 review.id；change.evidence 只选绑定 review.evidence 中的原项。
  只有 within_scope 的 support/counter 可附 change。context_only、unknown、范围外或旧事复述
  不附 change，不修改证据标签来凑更新。新 change 为 relation=new、id=[]、lineage=[]，绑定
  support，claim="" 交代码生成；已有 change 的 id/claim 保持原主张，lineage=[完整原身份对象]，
  mechanism_relation=same、scope_relation=within。已有 support 对应 support，counter 对应
  counter 或 retire；只有本人明确要求停止沿用旧主张时选 retire。
- atomic_claim 恰有五个字段：aspect、conditions、tendency、outcome、evidence_basis。
  前三项是非空字符串，各最多 100 字符。evidence_basis 与绑定 review 相同；
  stated_preference 的 outcome=[]，observed_response 的 outcome=[本人有据的实际反应]，
  repeated_pattern 按证据用 [] 或单个反应字符串（非空，最多 100 字符），不得伪造结果。
  新 claim 生成“在{conditions}的情境下，可能{tendency}”，有 outcome 再接
  “；本人描述的伴随反应是{outcome}”；全句不超过 180 字符，不能删必要条件来缩短。
- confidence 是 0..1 有限数字，只表示本次证据关系强度，不是累计人物分或人格概率，
  不复制旧分或为过门槛抬分。retire 达原 0.7 门槛才直接撤回，否则按有效 counter 处理。
  target_value 仅 big_five/disc 可填 [0..1数字]，其他或不确定时用 []。
- scope_tags 选 Schema 中 1..8 个不同标签，general 只能单独用。framework/dimension 对应：
  response：low_pressure/direct_options/identity_action/human_connection/meaning_memory/autonomy；
  big_five：openness/conscientiousness/extraversion/agreeableness/neuroticism；
  disc：dominance/influence/steadiness/conscientiousness；attachment：secure/anxious/avoidant/disorganized；
  cbt：core_pattern/trigger/belief/maintaining_factor/protective_factor。
  big_five/disc 需相对稳定行为，attachment 需重要关系中的重复模式；cbt 需本人可见的
  情境—解释/信念—情绪/行为联系，不从表面行为编造深层心理。

格式示例（仅展示一个完整观察的形状，不是本次证据）：
{"protocol_version":"2.0","reviews":[{"id":"r1","hypothesis_id":[],"claim_quote":[],"scope_quote":[],"applicability":"unknown","relation":"unknown","evidence_basis":"context_only","evidence":[{"turn_index":0,"evidence_quote":"我想先歇一会儿。"}],"event_evidence":[{"turn_index":0,"evidence_quote":"刚整理完桌子，我想先歇一会儿。"}],"reason":"只说明当前需要休息，不据此推断稳定性格。","event_summary":"本人整理桌子后表示想休息。"}],"changes":[]}
提交前按 Schema 检查每个对象的必填字段与唯一键、逐字引用及条数；没有变化也保留 changes=[]。
"""
    return (prompt
        + '\n字段与类型 Schema（仅说明正文形状，未启用供应商格式约束；不要输出 Schema 本身）：\n'
        + model_visible_schema_text())


def model_visible_contract_binding(client=None):
    if client is not None and client.reflection_return_format == 'json_schema_v4':
        schema=json.dumps(indexed_compact_reflection_schema(),ensure_ascii=False,sort_keys=True,separators=(',',':'))
        return {'model_visible_schema_protocol':STRUCTURED_PROMPT_VERSION,
            'model_visible_schema_sha256':sha256(schema.encode()).hexdigest(),
            'model_visible_schema_bytes':len(schema.encode()),
            'model_visible_system_sha256':sha256(indexed_compact_reflection_system_prompt().encode()).hexdigest()}
    return {'model_visible_schema_protocol': PROMPT_VERSION,
        'model_visible_schema_sha256': sha256(model_visible_schema_text().encode()).hexdigest(),
        'model_visible_schema_bytes': len(model_visible_schema_text().encode()),
        'model_visible_system_sha256': sha256(schema_free_reflection_system_prompt().encode()).hexdigest()}


def schema_free_reflection_messages(client, messages):
    """Transform only a verified old v3 system; evidence remains exact."""
    validate_stream_client(client)
    validate_reflection_request(client, messages, wire=True)
    result = deepcopy(messages)
    result[0]['content'] = schema_free_reflection_system_prompt()
    return result


class SchemaFreePublicCapture(SSEPublicCapture):
    """Bind the actual provider-format choice to each first physical receipt."""
    @staticmethod
    def _mark(observation):
        observation['provider_schema_applied'] = False
        observation['provider_schema_protocol'] = VERSION
        observation.update(model_visible_contract_binding())

    def observe_complete_stream(self, assembled, frames, stream_observation):
        self._mark(stream_observation)
        return super().observe_complete_stream(assembled, frames, stream_observation)

    def observe_failed_stream(self, observation):
        self._mark(observation)
        return super().observe_failed_stream(observation)


class StructuredPublicCapture(SchemaFreePublicCapture):
    @staticmethod
    def _mark(observation):
        observation['provider_schema_applied']=True
        observation['provider_schema_protocol']=STRUCTURED_VERSION
        schema=json.dumps(indexed_compact_reflection_schema(),ensure_ascii=False,sort_keys=True,separators=(',',':'))
        observation.update(model_visible_schema_protocol=STRUCTURED_PROMPT_VERSION,
            model_visible_schema_sha256=sha256(schema.encode()).hexdigest(),
            model_visible_schema_bytes=len(schema.encode()),
            model_visible_system_sha256=sha256(indexed_compact_reflection_system_prompt().encode()).hexdigest())


def validate_stream_client(client):
    reference = validate_reflection_structured_client(client)
    if type(client) is ProviderStructuredReflectionClient:
        from aria.engine.policy.online_structured import DEEPSEEK_MODEL
        if (client.model != DEEPSEEK_MODEL or reference['provider'] != 'together'
                or client.reflection_return_format != 'json_schema_v4'
                or getattr(client,'reflection_stream_protocol',None) != STREAM_VERSION
                or getattr(client,'reflection_provider_schema_protocol',None) != STRUCTURED_VERSION
                or getattr(client,'reflection_public_wrapper_protocol',None) != WRAPPER_VERSION
                or type(getattr(client,'reflection_physical_timeout_seconds',None)) is not int
                or client.reflection_physical_timeout_seconds != 90):
            raise ValueError('invalid_provider_structured_reflection_client')
        return {'schema':STRUCTURED_VERSION,'stream_protocol':STREAM_VERSION,'stream':True,
            'physical_timeout_seconds':90,'provider_schema_applied':True,'response_format_present':True,
            'local_codec':'reflection_json_schema_v4','local_itemwise_validation':True,
            'receipt_mode':STRUCTURED_STREAM_VERSION,'terminal_receipt_finishes':['stop','length'],
            'required_terminal_usage':True,'required_done':True,'partial_submission':False,
            'public_wrapper_protocol':WRAPPER_VERSION,'native_json_only':True,
            'reference_schema_sha256':reference['reflection_schema_sha256'],
            **model_visible_contract_binding(client)}
    if (type(client) is not ProviderSchemaFreeReflectionClient
            or client.reflection_return_format != 'json_schema_v3'
            or getattr(client, 'reflection_stream_protocol', None) != STREAM_VERSION
            or getattr(client, 'reflection_provider_schema_protocol', None) != VERSION
            or getattr(client, 'reflection_public_wrapper_protocol', None) != WRAPPER_VERSION
            or client.timeout != 90
            or type(getattr(client, 'reflection_physical_timeout_seconds', None)) is not int
            or client.reflection_physical_timeout_seconds != 90):
        raise ValueError('invalid_provider_schema_free_reflection_client')
    return {'schema': VERSION, 'stream_protocol': STREAM_VERSION, 'stream': True,
        'physical_timeout_seconds': 90, 'provider_schema_applied': False,
        'response_format_present': False, 'local_codec': 'reflection_json_schema_v3',
        'local_itemwise_validation': True, 'required_terminal_usage': True,
        'required_done': True, 'partial_submission': False,
        'public_wrapper_protocol': WRAPPER_VERSION,
        'reference_schema_sha256': reference['reflection_schema_sha256'],
        **model_visible_contract_binding()}


def expected_provider_schema_free_payload(client, messages):
    """Construct the ACTUAL expected wire, with no invented response_format."""
    validate_stream_client(client)
    if (type(messages) is not list or len(messages) != 2
            or messages[0] != {'role': 'system', 'content': schema_free_reflection_system_prompt()}
            or type(messages[1]) is not dict or set(messages[1]) != {'role', 'content'}
            or messages[1]['role'] != 'user' or type(messages[1]['content']) is not str):
        raise ValueError('provider_schema_free_visible_contract_required')
    reference = validate_reflection_structured_client(client)
    provider = {'allow_fallbacks': False, 'require_parameters': True,
                'data_collection': 'deny', 'only': [reference['provider']]}
    if getattr(client, 'reservation_max_price', None) is not None:
        provider['max_price'] = deepcopy(client.reservation_max_price)
    return {'model': client.model, 'messages': deepcopy(messages),
        'max_tokens': 2800, 'reasoning': {'enabled': False, 'exclude': True},
        'stream': True, 'provider': provider}


def expected_provider_reflection_payload(client,messages):
    if type(client) is not ProviderStructuredReflectionClient:
        return expected_provider_schema_free_payload(client,messages)
    validate_stream_client(client)
    validate_reflection_request(client,messages,wire=True)
    reference=validate_reflection_structured_client(client)
    provider={'allow_fallbacks':False,'require_parameters':True,
              'data_collection':'deny','only':[reference['provider']]}
    if getattr(client,'reservation_max_price',None) is not None:
        provider['max_price']=deepcopy(client.reservation_max_price)
    return {'model':client.model,'messages':deepcopy(messages),'max_tokens':2800,
        'reasoning':{'enabled':False,'exclude':True},'stream':True,'provider':provider,
        'response_format':indexed_compact_reflection_request_format()}


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


class ProviderSchemaFreeReflectionClient(OpenRouterLLMClient):
    def _extract_reflection_json(self,data):
        validate_stream_client(self)
        # Shared with receipt validation; no JSON/body repair or source edits.
        try:
            projected=reflection_content_projection(data['choices'][0]['message']['content'])
        except (ValueError,TypeError,KeyError,IndexError,RecursionError):
            self._update_diagnostic(reflection_wrapper={'version':WRAPPER_VERSION,'status':'rejected',
                'representation_only':True,'domain_content_repaired':False})
            # Preserve original strict error reporting and rejection.
            return super()._extract_reflection_json(data)
        view=deepcopy(data);view['choices'][0]['message']['content']=projected['body']
        result=super()._extract_reflection_json(view)
        if result!=projected['canonical']:
            raise ValueError('reflection_wrapper_canonical_mismatch')
        self._update_diagnostic(reflection_wrapper=deepcopy(projected['metadata']))
        return result

    def _request_for_model(self, payload, *, model):
        validate_stream_client(self)
        if (not isinstance(getattr(self, 'model_request_control', None), ReflectionSSEGuardMixin)
                or not hasattr(self.model_request_control, 'policy')):
            raise ValueError('provider_schema_free_reserved_guard_required')
        if type(payload) is not dict or model != self.model:
            raise ValueError('invalid_provider_schema_free_input')
        original_messages = payload.get('messages')
        actual_messages = schema_free_reflection_messages(self, original_messages)
        expected = expected_provider_schema_free_payload(self, actual_messages)
        # Verify what the unchanged v3 builder generated before deleting exactly
        # its provider format. This is an input assertion, never actual-wire
        # validation, request capture, price reservation or execution evidence.
        source = deepcopy(expected); source.pop('model'); source['stream'] = False
        source['messages'] = deepcopy(original_messages)
        source['response_format'] = compact_reflection_request_format()
        if not _same(payload, source):
            raise ValueError('provider_schema_free_builder_changed')
        value = deepcopy(payload); del value['response_format']; value['stream'] = True
        value['messages'] = actual_messages
        self._update_diagnostic(provider_schema_applied=False,
            reflection_provider_schema_protocol=VERSION, **model_visible_contract_binding())
        return super()._request_for_model(value, model=model)

    def _record_response_shape(self, choice, *, output_kind, error_type):
        names = {'reflection_json_schema': 'compact_v3_local_json',
                 'invalid_reflection_json_schema': 'invalid_compact_v3_local_json'}
        super()._record_response_shape(choice, output_kind=names.get(output_kind, output_kind),
                                      error_type=names.get(error_type, error_type))
        self._update_diagnostic(provider_schema_applied=False,
            reflection_provider_schema_protocol=VERSION, **model_visible_contract_binding())


class ProviderStructuredReflectionClient(OpenRouterLLMClient):
    """Current DeepSeek wire: existing strict schema, native JSON and SSE."""
    def _extract_reflection_json(self,data):
        validate_stream_client(self)
        try:
            projected=reflection_content_projection(data['choices'][0]['message']['content'],
                                                     version='json_schema_v4')
        except (ValueError,TypeError,KeyError,IndexError,RecursionError):
            self._update_diagnostic(reflection_wrapper={'version':WRAPPER_VERSION,'status':'rejected',
                'representation_only':True,'domain_content_repaired':False})
            return super()._extract_reflection_json(data)
        result=super()._extract_reflection_json(data)
        if result!=projected['canonical']:
            raise ValueError('reflection_wrapper_canonical_mismatch')
        self._update_diagnostic(reflection_wrapper=deepcopy(projected['metadata']))
        return result

    def _request_for_model(self,payload,*,model):
        validate_stream_client(self)
        if (not isinstance(getattr(self,'model_request_control',None),ReflectionSSEGuardMixin)
                or not hasattr(self.model_request_control,'policy')
                or type(payload) is not dict or model!=self.model):
            raise ValueError('provider_structured_reserved_guard_required')
        expected=expected_provider_reflection_payload(self,payload.get('messages'))
        source=deepcopy(expected);source.pop('model');source['stream']=False
        if not _same(payload,source):
            raise ValueError('provider_structured_builder_changed')
        value=deepcopy(payload);value['stream']=True
        self._update_diagnostic(provider_schema_applied=True,
            reflection_provider_schema_protocol=STRUCTURED_VERSION,**model_visible_contract_binding(self))
        return super()._request_for_model(value,model=model)

    def _record_response_shape(self,choice,*,output_kind,error_type):
        super()._record_response_shape(choice,output_kind=output_kind,error_type=error_type)
        self._update_diagnostic(provider_schema_applied=True,
            reflection_provider_schema_protocol=STRUCTURED_VERSION,**model_visible_contract_binding(self))


# Both are existing reflection calls through the same SSE dispatch/guard.
StreamingReflectionClient = (ProviderSchemaFreeReflectionClient,ProviderStructuredReflectionClient)


def make_provider_schema_free_reflection_client(client):
    validate_reflection_structured_client(client)
    if (type(client) is not OpenRouterLLMClient or client.reflection_return_format != 'json_schema_v3'
            or getattr(client, 'model_request_control', None) is not None
            or any(getattr(client, key, 0) != 0 for key in
                   ('call_count', 'network_request_count', 'generation_decision_count'))):
        raise ValueError('fresh_compact_reflection_client_required')
    result = copy(client); result.__class__ = ProviderSchemaFreeReflectionClient
    result.reflection_stream_protocol = STREAM_VERSION
    result.reflection_provider_schema_protocol = VERSION
    result.reflection_public_wrapper_protocol = WRAPPER_VERSION
    result.reflection_physical_timeout_seconds = 90
    validate_stream_client(result)
    return result


def make_provider_structured_reflection_client(client):
    validate_reflection_structured_client(client)
    if (type(client) is not OpenRouterLLMClient or client.reflection_return_format not in {'json_schema_v3','json_schema_v4'}
            or getattr(client,'model_request_control',None) is not None
            or any(getattr(client,key,0)!=0 for key in ('call_count','network_request_count','generation_decision_count'))):
        raise ValueError('fresh_compact_reflection_client_required')
    result=copy(client);result.__class__=ProviderStructuredReflectionClient
    result.reflection_return_format='json_schema_v4'
    result.reflection_stream_protocol=STREAM_VERSION
    result.reflection_provider_schema_protocol=STRUCTURED_VERSION
    result.reflection_public_wrapper_protocol=WRAPPER_VERSION
    result.reflection_physical_timeout_seconds=90
    validate_stream_client(result)
    return result


def create_qwen_max_schema_free_reflection_clients(**kwargs):
    """Explicit opt-in assembly; all non-reflection clients are unchanged."""
    from aria.release import create_qwen_max_compact_reflection_clients
    clients = create_qwen_max_compact_reflection_clients(**kwargs)
    if not kwargs.get('model_calls_enabled', False): return clients
    return replace(clients, reflection=make_provider_schema_free_reflection_client(clients.reflection))


def validate_structured_sse_client(client):
    """Reuse native role validators; transport changes no prompt, schema or role."""
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL, validate_release_online_client
    from aria.engine.policy.memory_structured import validate_memory_structured_client
    from aria.engine.policy.safety_structured import validate_safety_structured_client
    if (type(client) is not OpenRouterLLMClient or client.role not in STRUCTURED_ROLES
            or getattr(client, 'structured_stream_protocol', None) != STRUCTURED_STREAM_VERSION
            or client.model != DEEPSEEK_MODEL or client.timeout != 90
            or type(client.max_retries) is not int or client.max_retries != 0
            or client.fallback_models != [] or client.bounded_transport is not True
            or client.max_completion_tokens != {'memory': 4000, 'safety_review': 1024, 'confirm_judge': 60}[client.role]
            or getattr(client, 'online_stream_protocol', None) is not None):
        raise ValueError('invalid_structured_sse_client')
    route = (validate_memory_structured_client(client) if client.role == 'memory' else
             validate_safety_structured_client(client) if client.role == 'safety_review' else
             validate_release_online_client(client))
    if route['provider'] != 'together':
        raise ValueError('invalid_structured_sse_route')
    return structured_sse_contract(client.role)


def enable_structured_sse(client):
    if (getattr(client, 'structured_stream_protocol', None) is not None
            or getattr(client, 'model_request_control', None) is not None
            or any(getattr(client, key, 0) != 0 for key in
                ('call_count', 'network_request_count', 'generation_decision_count'))):
        raise ValueError('fresh_structured_sse_client_required')
    result = copy(client)
    result.structured_stream_protocol = STRUCTURED_STREAM_VERSION
    validate_structured_sse_client(result)
    return result


class ReflectionSSEGuardMixin:
    def _binding(self, client):
        value = json.loads(super()._binding(client))
        if (client.role == 'reflection' or isinstance(client, ProviderSchemaFreeReflectionClient)
                or getattr(client, 'reflection_provider_schema_protocol', None) is not None):
            contract = validate_stream_client(client)
            # super() has already checked exact endpoint/provider/model/prices
            # through ReservedCallGuard. Rename only the evidence description,
            # never remove the client's reference route or its receipt checks.
            reference = value.pop('reflection_structured_route')
            capability = {k: v for k, v in reference.items()
                          if k not in {'version', 'format', 'reflection_schema_sha256'}}
            value['reflection_endpoint_capability_reference'] = capability
            value['reflection_local_return_format'] = value.pop('reflection_return_format')
            value['reflection_local_contract_sha256'] = value.pop('reflection_format_sha256')
            value['reflection_provider_contract'] = contract
            if contract.get('receipt_mode') == STRUCTURED_STREAM_VERSION:
                value['structured_stream_contract'] = structured_sse_contract(client.role)
        if getattr(client,'online_stream_protocol',None) is not None:
            from .online_shape import validate_online_sse_client
            value['online_stream_contract']=validate_online_sse_client(client)
        if getattr(client, 'structured_stream_protocol', None) is not None:
            value['structured_stream_contract'] = validate_structured_sse_client(client)
        return json.dumps(value, sort_keys=True, allow_nan=False)

    def _validate_payload(self, client, payload, **options):
        if (client.role == 'reflection' or isinstance(client, ProviderSchemaFreeReflectionClient)
                or getattr(client, 'reflection_provider_schema_protocol', None) is not None):
            try:
                validate_stream_client(client)
                if (set(options) - {'require_preference_control', 'require_task_projection'}
                        or any(type(v) is not bool or v for v in options.values())
                        or type(payload) is not dict):
                    raise ValueError('invalid_reflection_request_options')
                expected = expected_provider_reflection_payload(client, payload.get('messages'))
                if not _same(payload, expected): raise ValueError('actual_payload_changed')
            except (ValueError, TypeError, KeyError, RecursionError):
                raise MinimalCallStopped('provider_schema_free_wire_changed') from None
            return
        if (getattr(client, 'online_stream_protocol', None) is not None
                or getattr(client, 'structured_stream_protocol', None) is not None):
            from .online_shape import validate_online_sse_client
            try:
                (validate_structured_sse_client(client) if getattr(client, 'structured_stream_protocol', None) is not None
                 else validate_online_sse_client(client))
                if type(payload) is not dict or payload.get('stream') is not True:
                    raise ValueError('online_stream_required')
                original=deepcopy(payload);original['stream']=False
                return super()._validate_payload(client,original,**options)
            except (ValueError,TypeError,KeyError,RecursionError):
                raise MinimalCallStopped('online_sse_wire_changed') from None
        return super()._validate_payload(client, payload, **options)

    def _open_observed(self, request, **kwargs):
        from . import tls_dispatch
        response = tls_dispatch.open_observed(request, **kwargs)
        directory = self._physical_capture
        if directory is not None:
            payload=json.loads(request.data)
            if payload.get('stream') is True:
                response_format=payload.get('response_format')
                if type(response_format) is dict:
                    if response_format.get('json_schema',{}).get('name')=='aria_reflection_compact_v4':
                        return StructuredPublicCapture(response,directory)
                    return SSEPublicCapture(response,directory)
                return SchemaFreePublicCapture(response,directory)
            from .public_capture import PublicFirstReadCapture
            return PublicFirstReadCapture(response, directory)
        return response
