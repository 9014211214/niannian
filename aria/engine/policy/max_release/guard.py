"""Installable, explicitly enabled online/Sonnet guard.

Caller limits and the persistent local ledger are the only accounting scope.
This module contains no experiment identities, old fees, shared slots or grants.
Paid response replay is disabled. The proven one-extra pre-application TLS
preparation keeps the same physical request, reservation and deadline.
"""
from contextlib import closing, contextmanager
from copy import deepcopy
from decimal import Decimal
import hashlib,json,sqlite3,uuid
import urllib.request
from pathlib import Path
from ..minimal_calls import MinimalCallStopped, _unique_request_object, validate_limits
from ..reserved_calls import ReservedCallGuard, validate_reservation_policy
from .sse_adapter import ReflectionSSEGuardMixin
from .reflection_sse import structured_receipt_body_complete, structured_stream_marked
from .sse_dispatch import _request_once
from .tls_transport import CandidateTrace, LogicalTLSAllowance, TLSPreparation, require_endpoint
from .public_capture import write_once
from .validation_guard import SettledValidationGuardMixin
from .reflection_failure import SettledOfflineGuardMixin
from .content_erasure import ContentErasure
from .held_input import HeldInputGuardMixin
from .context_admission import ContextAdmissionMixin

OUTPUTS={'online_fast':4096,'memory':4000,'reflection':2800,'safety_review':1024,'confirm_judge':60}
BILLABLE_OUTPUTS={'online_fast':8202,'memory':4000,'reflection':2800,'safety_review':5130,'confirm_judge':4166}
MODELS={'online_fast':'qwen/qwen3.8-max-0902','safety_review':'qwen/qwen3.8-max-0902',
        'memory':'anthropic/claude-sonnet-5','reflection':'anthropic/claude-sonnet-5','confirm_judge':'qwen/qwen3.8-max-0902'}
ONLINE_FORMATS={'online_fast':'json_schema_v1','memory':'native','reflection':'native','safety_review':'native','confirm_judge':'native'}
REFLECTION_FORMATS={'online_fast':'native','memory':'native','reflection':'json_schema_v3','safety_review':'native','confirm_judge':'native'}
COUNTS=OUTPUTS  # Role names only; public installation has no experiment position counts.


def _release_pacing_interval(path, policy):
    from ..online_structured import DEEPSEEK_MODEL

    if set(policy['models']) != {DEEPSEEK_MODEL}:
        return 120
    # The single-model policy has already been validated as Together-only.
    # Five seconds is a local throttle. Each retained ledger keeps its own rate.
    path = Path(path)
    if not path.exists():
        return 5
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        try:
            registered = db.execute('SELECT min_interval_seconds FROM request_pacing WHERE id=1').fetchone()
            if registered not in ((5,), (120,)) or db.execute(
                "SELECT 1 FROM attempts WHERE json_extract(record, '$.pacing.min_interval_seconds') IS NOT ? LIMIT 1",
                registered,
            ).fetchone():
                raise ValueError('retained_request_pacing_changed')
        except sqlite3.DatabaseError as error:
            raise ValueError('retained_request_pacing_unavailable') from error
    return registered[0]


class MaxReleaseCallGuard(ContextAdmissionMixin,HeldInputGuardMixin,SettledOfflineGuardMixin,SettledValidationGuardMixin,ReflectionSSEGuardMixin,ReservedCallGuard):
    allowed_roles=set(OUTPUTS)
    def __init__(self,path,*,max_requests,max_observed_cost_usd,reservation_policy):
        max_requests,max_observed_cost_usd=validate_limits(max_requests,max_observed_cost_usd)
        reservation_policy = validate_reservation_policy(reservation_policy)
        self._policy_online_model(reservation_policy)
        self.cap=Decimal(str(max_observed_cost_usd))
        self.public_dir=Path(path).parent/'model-public-first-outputs';self.public_dir.mkdir(parents=True,exist_ok=True)
        self.physical_dir=Path(path).parent/'model-physical-receipts';self.physical_dir.mkdir(parents=True,exist_ok=True)
        self._physical_capture=None;self._tls_logical=None;self._candidate_preparation=None
        self._validation_resolution_readers=[]
        self._active_operation_tokens=set()
        self.checker=self._check_runtime_contract
        self._candidate_checker_original=self.checker
        self._candidate_threshold=str(max_observed_cost_usd)
        self._candidate_max_requests=max_requests
        self._candidate_pacing_interval = _release_pacing_interval(path, reservation_policy)
        super().__init__(path,max_requests=max_requests,max_observed_cost_usd=max_observed_cost_usd,
            reservation_policy=reservation_policy,min_interval_seconds=self._candidate_pacing_interval,observe_transport=True,
            transport_diagnostics=True,recovery_policy=None,connection_preparation_recovery=False)
        with self._lock, self._transaction():
            self._db.execute("CREATE TABLE IF NOT EXISTS release_operations "
                "(token TEXT PRIMARY KEY, kind TEXT NOT NULL, operation_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, finished INTEGER NOT NULL CHECK(finished IN (0,1)))")
        if self._pacing_interval == 120:
            self._reopen_pacing_required = True
        self.content_erasure=ContentErasure(path,self.ledger_id,self.public_dir,self.physical_dir)

    def _check_runtime_contract(self):
        if (self.checker is not self._candidate_checker_original or self.threshold!=self._candidate_threshold
            or self.max_requests!=self._candidate_max_requests or self._pacing_interval!=self._candidate_pacing_interval
            or self._connection_preparation_recovery is not False or self.recovery_policy is not None):
            raise MinimalCallStopped('max_release_runtime_contract_changed')
        return True

    def _candidate_contract(self):
        return self._check_runtime_contract()

    def _new_tls_allowance(self,client,request):return LogicalTLSAllowance()

    def _online_model(self):
        return self._policy_online_model(self.policy)

    @staticmethod
    def _policy_online_model(policy):
        from ..online_structured import DEEPSEEK_MODEL, DEEPSEEK_PROVIDER_NAMES
        models = policy['models']
        if DEEPSEEK_MODEL not in models:
            return MODELS['online_fast']
        # A new exact policy selects DeepSeek. Historical Max allowances cannot
        # be reused or silently mixed with the replacement online model.
        spec = models[DEEPSEEK_MODEL]
        if (set(models) not in ({DEEPSEEK_MODEL, MODELS['memory']}, {DEEPSEEK_MODEL})
                or spec['provider'] not in DEEPSEEK_PROVIDER_NAMES
                or spec['response_providers'] != [DEEPSEEK_PROVIDER_NAMES[spec['provider']]]
                or set(models) == {DEEPSEEK_MODEL} and spec['provider'] != 'together'):
            raise ValueError('deepseek_release_policy_mismatch')
        return DEEPSEEK_MODEL

    def _offline_model(self):
        from ..online_structured import DEEPSEEK_MODEL
        return DEEPSEEK_MODEL if set(self.policy['models']) == {DEEPSEEK_MODEL} else MODELS['memory']

    def _binding(self, client):
        require_endpoint(urllib.request.Request(client.base_url),base=True)
        online_model = self._online_model()
        online_role = client.role in ('online_fast', 'safety_review', 'confirm_judge')
        if (client.role not in COUNTS or client.timeout != 90
                or type(client.max_retries) is not int or client.max_retries != 0
                or client.fallback_models != [] or client.max_completion_tokens != OUTPUTS[client.role]
                or client.model != (online_model if online_role else self._offline_model())
                or client.bounded_transport is not True
                or client.online_return_format != ONLINE_FORMATS[client.role]
                or client.reflection_return_format not in (
                    {'json_schema_v3', 'json_schema_v4'} if client.role == 'reflection'
                    and self._offline_model() != MODELS['reflection']
                    else {REFLECTION_FORMATS[client.role]})
                or getattr(client, 'offline_reasoning_route', None) is not None):
            raise ValueError('finite_chain_role_contract_changed')
        if online_role:
            from ..online_structured import validate_release_online_client, DEEPSEEK_MODEL
            value = validate_release_online_client(client)
            if online_model == DEEPSEEK_MODEL:
                from ..online_structured import _metadata
                from ..release_configuration import _all_tier_prices
                route = getattr(client, {'online_fast':'online_structured_route',
                    'safety_review':'safety_structured_route',
                    'confirm_judge':'confirm_structured_route'}[client.role])
                endpoint, = [row for row in _metadata(route.endpoints_raw)['data']['endpoints']
                             if row.get('tag') == value['provider']]
                rates = _all_tier_prices(endpoint['pricing'], endpoint['context_length'])
                spec = self.policy['models'][online_model]
                if (Decimal(spec['input_usd_per_million']) < rates[0] * 1000000
                        or Decimal(spec['output_usd_per_million']) < rates[1] * 1000000
                        or spec['max_output_tokens'] != OUTPUTS['online_fast']
                        or endpoint['max_completion_tokens'] < OUTPUTS['online_fast']):
                    raise ValueError('deepseek_release_reservation_understated')
            elif (value['wire_visible_limit'] != OUTPUTS[client.role]
                    or value['billable_completion_limit'] != BILLABLE_OUTPUTS[client.role]
                    or value['reasoning_token_plan'] != 4096 or value['shared_token_margin'] != 10
                    or value['gateway_mapping_observed'] is not False):
                raise ValueError('finite_max_token_accounting_contract_changed')
        if client.role == 'memory':
            from aria.engine.policy.memory_structured import validate_memory_structured_client
            validate_memory_structured_client(client)
            from aria.engine.memory.preference_scope import validate_model_scope_client
            validate_model_scope_client(client)
        if client.role in {'safety_review', 'memory'}:
            from ..online_structured import DEEPSEEK_MODEL
            from ...risk.safety_review_v2 import uses_owner_sources
            from ..memory_structured import memory_schema_protocol
            from ...memory.memory_contract import FACT_PROTOCOLS, LEGACY_FACT_SOURCE_REFS
            if client.role == 'safety_review':
                source_bound = uses_owner_sources(client)
            else:
                protocol = memory_schema_protocol(client)
                source_bound = protocol in FACT_PROTOCOLS
                if protocol == LEGACY_FACT_SOURCE_REFS and not self._db.execute(
                        "SELECT 1 FROM roles WHERE role='memory'").fetchone():
                    raise ValueError('retained_memory_binding_required')
            same_route = self._offline_model() == DEEPSEEK_MODEL
            if (source_bound and not same_route) or (self._pacing_interval == 5 and not source_bound):
                raise ValueError('release_owner_source_contract_mismatch')
        if client.role == 'reflection':
            from aria.engine.policy.reflection_structured import validate_reflection_structured_client
            validate_reflection_structured_client(client)
        if client.role == 'online_fast':
            from aria.engine.memory.preference_candidate import validate_preference_candidate_client
            validate_preference_candidate_client(client)
        result=super()._binding(client)
        if client.role=='online_fast':
            from .online_shape import validate_online_projection_client
            value=json.loads(result);value['online_response_projection']=validate_online_projection_client(client)
            result=json.dumps(value,sort_keys=True,allow_nan=False)
        elif getattr(client,'online_response_projection_protocol',None) is not None:
            raise ValueError('online_projection_wrong_role')
        return result

    def request_json(self, client, request, **kwargs):
        with self.dispatch_slot(), self._lock:
            self._candidate_contract()
            if self._tls_logical is not None:
                raise MinimalCallStopped('candidate_logical_reentry_forbidden')
            require_endpoint(request)
            identity = {'wire': bytes(request.data), 'url': request.full_url,
                'method': request.get_method(), 'role': client.role, 'model': client.model,
                'binding': self._binding(client), 'scope': deepcopy(getattr(self._local, 'scope', None)),
                'headers': {name.lower(): value for name, value in request.header_items()}}
            self._tls_logical = {'allowance': self._new_tls_allowance(client, request), 'identity': identity, 'effective_headers_sha256': None}
            try:
                return super().request_json(client, request, **kwargs)
            finally:
                self._candidate_preparation = None
                self._tls_logical = None

    def _candidate_trace(self, client, request, row, deadline, **options):
        if self._tls_logical is None:
            raise MinimalCallStopped('candidate_logical_context_missing')
        preparation = TLSPreparation(check=lambda: self._check_candidate_pending(client, request, row, **options),
            deadline=deadline, allowance=self._tls_logical['allowance'], bind_headers=self._bind_candidate_headers)
        self._candidate_preparation = (row['request_id'], preparation)
        trace=CandidateTrace(preparation=preparation, deadline=deadline)
        self._held_bind_deadline(client,deadline,trace)
        return trace

    @staticmethod
    def _headers_digest(request):
        headers = sorted((name.lower(), value) for name, value in request.header_items())
        return hashlib.sha256(json.dumps(headers, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

    def _bind_candidate_headers(self, request):
        if self._tls_logical is None:
            raise MinimalCallStopped('candidate_logical_context_missing')
        digest = self._headers_digest(request)
        current = self._tls_logical['effective_headers_sha256']
        if current is not None and current != digest:
            raise MinimalCallStopped('candidate_effective_headers_changed')
        self._tls_logical['effective_headers_sha256'] = digest

    def _check_candidate_pending(self, client, request, row, *,
                                 require_preference_control=False, require_task_projection=False):
        return self._held_check_pending(lambda:self._check_candidate_pending_original(client,request,row,
            require_preference_control=require_preference_control,require_task_projection=require_task_projection))

    def _check_candidate_pending_original(self, client, request, row, *,
                                 require_preference_control=False, require_task_projection=False):
        """Real finite-ledger checks before each preparation, entry and write."""
        self._candidate_contract()
        self._check_online_context()
        if self._stop_requested.is_set() or self._broken or self._closed:
            raise MinimalCallStopped('request_stop_requested')
        self.checker()
        self._check_transport_configuration()
        self._check_pacing_configuration()
        self._recovery.check()
        self._check_policy()
        if self._db.execute('SELECT max_requests,threshold,halted FROM settings WHERE id=1').fetchone() != (
                self.max_requests, self.threshold, None):
            raise MinimalCallStopped('model_allowance_changed')
        if self._tls_logical is None:
            raise MinimalCallStopped('candidate_logical_context_missing')
        identity = self._tls_logical['identity']
        self._tls_logical['allowance'].used
        headers = {name.lower(): value for name, value in request.header_items()}
        binding = self._binding(client)
        if (request.data != identity['wire'] or request.full_url != identity['url']
                or request.get_method() != identity['method'] or request.get_method() != 'POST'
                or client.role != identity['role'] or client.model != identity['model']
                or binding != identity['binding']
                or getattr(self._local, 'scope', None) != identity['scope']
                or any(headers.get(name) != value for name, value in identity['headers'].items())
                or self._headers_digest(request) != self._tls_logical['effective_headers_sha256']
                or request.full_url != client.base_url.rstrip('/') + '/chat/completions'
                or hashlib.sha256(request.data).hexdigest() != row['request_sha256']
                or any(row.get(name) != identity['scope'].get(name) for name in ('kind', 'session_id', 'operation_id'))
                or row['role'] != client.role or row['model'] != client.model
                or row['expected_provider'] != self.policy['models'][client.model]['response_providers'][0]):
            raise MinimalCallStopped('candidate_request_identity_changed')
        if self._db.execute('SELECT binding FROM roles WHERE role=?', (client.role,)).fetchone() != (binding,):
            raise MinimalCallStopped('model_configuration_changed')
        self._validate_payload(client, json.loads(request.data, object_pairs_hook=_unique_request_object),
            require_preference_control=require_preference_control, require_task_projection=require_task_projection)
        state = self._snapshot()
        pending = [item for item in state['attempts'] if item['status'] == 'pending']
        comparable = ('request_id', 'request_sha256', 'role', 'model', 'expected_provider',
            'kind', 'session_id', 'operation_id', 'reservation', 'physical_attempt', 'recovery_of_request_id', 'online_context_admission',
            'confirmation_context_admission')
        occupied = Decimal(state['known_cost_usd'] or '0') + Decimal(state['reserved_cost_usd'])
        if (len(pending) != 1 or any(pending[0].get(key) != row.get(key) for key in comparable)
                or occupied > self.cap):
            raise MinimalCallStopped('inflight_allowance_changed')
        if self._stop_requested.is_set():
            raise MinimalCallStopped('request_stop_requested')

    def _begin(self,client,request,**kwargs):
        try:
            self.checker();state=self._snapshot();scope=getattr(self._local,'scope',{})
            if scope.get('kind')!=('turn' if client.role in ('online_fast','safety_review','confirm_judge') else 'review'):
                raise MinimalCallStopped('finite_role_operation_kind_mismatch')
            if any(r['role']==client.role and r['operation_id']==scope.get('operation_id') for r in state['attempts']):
                raise MinimalCallStopped('finite_no_duplicate_or_format_correction')
            row=super()._begin(client,request,**kwargs)
            row.update(physical_attempt=1,expected_provider=self.policy['models'][client.model]['response_providers'][0])
            self._bind_online_admission_row(row)
            with self._transaction():self._db.execute('UPDATE attempts SET record=? WHERE request_id=?',
                (json.dumps(row,ensure_ascii=False,allow_nan=False),row['request_id']))
            return row
        except BaseException as error:
            try:self._persist_budget_refusal(error)
            finally:self.halt()
            raise

    @staticmethod
    def _reflection_product_metadata(client, payload, previous):
        if previous:
            raise MinimalCallStopped('finite_no_duplicate_or_format_correction')
        if client.role == 'reflection':
            from .sse_adapter import expected_provider_reflection_payload
            # Validate the actual complete model-visible contract. Do not
            # reconstruct an old system prompt to pass a different protocol.
            expected = expected_provider_reflection_payload(client, payload['messages'])
            if json.dumps(payload, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
                raise MinimalCallStopped('reflection_metadata_wire_changed')
        return ReservedCallGuard._reflection_product_metadata(client, payload, previous)

    @staticmethod
    def _fee_attribution_error(row):
        identity, usage = row.get('receipt_identity', {}), row.get('usage', {})
        reservation = row.get('reservation', {})
        if row.get('raw_total_tokens_valid') is False:
            return 'finite_raw_token_totals_invalid'
        if row.get('http_status') is None:
            return 'finite_no_http_response_cost_unverified'
        if row.get('http_status') != 200:
            return 'finite_http_error_cost_unverified'
        if identity.get('model_status') != 'recognized' or identity.get('provider_status') != 'recognized':
            return 'finite_response_identity_unverified'
        if identity.get('error_field_status') not in ('absent', 'null'):
            return 'finite_error_envelope_cost_unverified'
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens')):
            return 'finite_token_usage_unverified'
        if (usage['prompt_tokens'] > reservation['input_token_plan_bound']
                or usage['completion_tokens'] > reservation['output_token_limit']):
            return 'finite_token_bound_exceeded'
        if 'total_tokens' in usage and (type(usage['total_tokens']) is not int
                or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']):
            return 'finite_token_totals_inconsistent'
        if 'cost' in usage and Decimal(str(usage['cost'])) > Decimal(reservation['reserved_cost_usd']):
            return 'finite_cost_bound_exceeded'
        return None

    def _record_receipt(self, client, row, data):
        # Preserve only actual public first content/arguments. Never serialize
        # full envelopes, headers, keys, reasoning or reasoning_details.
        public = []
        choices = data.get('choices') if type(data) is dict else None
        for choice in choices if type(choices) is list else []:
            if type(choice) is not dict or type(choice.get('message')) is not dict:
                continue
            message = choice['message']
            value = {'content': message.get('content') if type(message.get('content')) is str else None,
                     'function_calls': []}
            for call in message.get('tool_calls', []) or []:
                function = call.get('function') if type(call) is dict else None
                if type(function) is dict:
                    value['function_calls'].append({k: function.get(k) for k in ('name', 'arguments')
                                                   if type(function.get(k)) is str})
            public.append(value)
        path = self.public_dir / (row['request_id'] + '.json')
        saved=self.content_erasure.write(path, {'request_id': row['request_id'], 'role': row['role'],
            'request_sha256': row['request_sha256'], 'public_first_output': public,
            'hidden_reasoning_retained': False},self._physical_capture)
        row['public_first_output'] = {'path': str(path), 'sha256': saved['sha256']}
        if not saved['saved']:row['public_content_erased_by_owner']=True
        if client.role == 'memory':
            choice=choices[0] if type(choices) is list and len(choices)==1 and type(choices[0]) is dict else {}
            message=choice.get('message')
            text=message.get('content') if type(message) is dict else None
            row['memory_receipt_contract']={'request_sha256':row['request_sha256'],
                'response_models':list(self.policy['models'][client.model]['response_models']),
                'finish_reason':choice.get('finish_reason'),
                'plain_text':(type(text) is str and message.get('tool_calls') in (None, [])
                    and message.get('function_call') is None and message.get('refusal') in (None, '')),
                'public_content_sha256':hashlib.sha256(text.encode()).hexdigest() if type(text) is str else None}
        if client.role == 'online_fast' and type(choices) is list and len(choices) == 1:
            choice=choices[0];message=choice.get('message') if type(choice) is dict else None
            if (type(message) is dict and type(message.get('content')) is str
                    and message.get('tool_calls') in (None, [])
                    and message.get('function_call') is None and message.get('refusal') in (None, '')):
                row['online_receipt_shape']={'decision_version':client.online_decision_version,
                    'finish_reason':choice.get('finish_reason'),
                    'public_content_sha256':hashlib.sha256(message['content'].encode()).hexdigest()}
        from ..online_structured import DEEPSEEK_MODEL
        if (client.role == 'safety_review' and row.get('model')==DEEPSEEK_MODEL
                and row.get('expected_provider')=='Together'):
            from ..safety_structured import validate_safety_review_request
            messages=json.loads(self._tls_logical['identity']['wire'])['messages']
            count=validate_safety_review_request(messages,client=client)
            choice=choices[0] if type(choices) is list and len(choices)==1 and type(choices[0]) is dict else {}
            message=choice.get('message')
            row['safety_review_receipt_contract']={'request_sha256':row['request_sha256'],
                'response_models':list(self.policy['models'][client.model]['response_models']),
                'version':json.loads(messages[1]['content'])['version'],'sentence_count':count,
                'finish_reason':choice.get('finish_reason'),
                'plain_text':(type(message) is dict and type(message.get('content')) is str
                    and message.get('tool_calls') in (None, []) and message.get('function_call') is None
                    and message.get('refusal') in (None, ''))}
        if 'structured_stream_contract' in row:
            choice = choices[0] if type(choices) is list and len(choices) == 1 and type(choices[0]) is dict else {}
            message = choice.get('message')
            content = message.get('content') if type(message) is dict else None
            row['structured_stream_receipt'] = {'request_sha256': row['request_sha256'],
                'response_models': list(self.policy['models'][client.model]['response_models']),
                'finish_reason': choice.get('finish_reason'),
                'plain_text': (type(content) is str and message.get('tool_calls') in (None, [])
                    and message.get('function_call') is None and message.get('refusal') in (None, '')),
                'public_content_sha256': hashlib.sha256(content.encode()).hexdigest() if type(content) is str else None}
            if not structured_receipt_body_complete(row, content):
                raise MinimalCallStopped('structured_stream_receipt_invalid')
        # Validate a supplied total before allowlisting can drop its bad type.
        raw_usage = data.get('usage') if type(data) is dict else None
        if type(raw_usage) is dict and 'total_tokens' in raw_usage:
            row['raw_total_tokens_valid'] = (
                all(type(raw_usage.get(k)) is int and raw_usage[k] >= 0
                    for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                and raw_usage['total_tokens'] == raw_usage['prompt_tokens'] + raw_usage['completion_tokens'])
        view = data
        if client.role == 'reflection':
            from .reflection_wrapper import reflection_receipt_projection
            view = reflection_receipt_projection(client, row, data)
        if (row.get('online_stream_contract', {}).get('length_projection')
                and row.get('online_receipt_shape', {}).get('finish_reason') == 'length'
                and row.get('raw_total_tokens_valid') is True):
            from .online_shape import online_envelope_projection, online_receipt_body_complete
            from .request_requirements import FIELD, validate_binding
            if online_receipt_body_complete(row, choices[0]['message']['content']):
                requirements = row[FIELD]['requirements']
                validate_binding(row, requirements)
                try:
                    _, projection = online_envelope_projection(data, client.online_decision_version,
                        allow_padded_length=True, **requirements)
                except ValueError:
                    pass  # Keep the original truncation rejection and settled charge.
                else:
                    row['online_length_projection'] = projection
        super()._record_receipt(client, row, view)
        reason = self._fee_attribution_error(row)
        if reason is not None:
            row['validation_error'] = reason
            raise MinimalCallStopped(reason)
        if 'online_length_projection' in row:
            # The assembled native choice remains intact. Only this settled
            # receipt authorizes the corresponding local parser view.
            data['online_length_projection'] = row['online_length_projection']
        # The itemwise envelope check does not decode compact nullable slots.
        # Keep the existing projection's rejection and the verified charge;
        # classify it before the transport can mark this receipt received.
        if client.role == 'reflection' and row['reflection_wrapper']['status'] == 'rejected':
            row['validation_error'] = 'reflection_schema_invalid'
            raise MinimalCallStopped('reflection_schema_invalid')
        if client.role == 'safety_review' and 'safety_review_receipt_contract' in row:
            from ...risk.safety_review import parse_safety_review
            contract=row['safety_review_receipt_contract']
            # v2's bound input assessment and deletion verdict have separate
            # product effects. The product owns that validation once; a paid
            # receipt being received never claims its safety verdict passed.
            if contract['version']=='safety_review_v1':
                try:parse_safety_review(data['choices'][0]['message']['content'],contract['sentence_count'])
                except (ValueError,TypeError,KeyError,RecursionError):
                    row['validation_error']='safety_review_schema_invalid'
                    raise MinimalCallStopped('safety_review_schema_invalid') from None
        if client.role == 'memory' and data['choices'][0]['finish_reason'] != 'stop':
            raise MinimalCallStopped('finite_memory_must_be_untruncated')

    def _finish(self,row):
        if self._candidate_preparation is not None and self._candidate_preparation[0]==row['request_id']:
            row['tls_preparation_candidate']=self._candidate_preparation[1].receipt()
        reason=self._fee_attribution_error(row)
        if reason is not None:
            observed=row['usage'].pop('cost',None)
            if observed is not None:row['unverified_cost_usd']=observed
            row.setdefault('fee_verification_error',reason)
        # T2 consumes a failed confirmation as unclear. Only a fully received,
        # attributable format rejection can take that path; retain the failed
        # physical row and its charge instead of turning it into a success.
        unclear = (not self._stop_requested.is_set() and not self._broken and not self._closed
                   and self._settled_confirm_format_failure(row))
        if unclear:
            row['confirm_judge_disposition'] = 'unclear_known_cost_format_failure_v1'
        review_failed = (not self._stop_requested.is_set() and not self._broken and not self._closed
                         and self._settled_safety_review_format_failure(row))
        if review_failed:
            # The existing risk reviewer catches this failure and withholds
            # the candidate body. This is not a successful safety verdict.
            row['safety_review_disposition']='withheld_known_cost_format_failure_v1'
        super()._finish(row)
        if not (unclear or review_failed) and (row.get('status')!='received' or 'cost' not in row.get('usage',{})):self.halt()

    @classmethod
    def _settled_safety_review_format_failure(cls,row):
        from ..online_structured import DEEPSEEK_MODEL
        error=row.get('validation_error');contract=row.get('safety_review_receipt_contract',{})
        body=row.get('body_read',{});usage=row.get('usage',{});request_sha256=row.get('request_sha256')
        if (row.get('role')!='safety_review' or row.get('kind')!='turn' or row.get('status')!='failed'
                or row.get('model')!=DEEPSEEK_MODEL
                or row.get('receipt_identity', {}).get('model_status')!='recognized'
                or row.get('returned_model') not in contract.get('response_models', [DEEPSEEK_MODEL])
                or row.get('expected_provider')!='Together' or row.get('returned_provider')!='Together'
                or row.get('failure_phase')!='receipt_validation' or row.get('failure_code')!=error
                or error not in {'invalid_or_truncated_response','invalid_response_message','safety_review_schema_invalid'}
                or type(request_sha256) is not str or len(request_sha256)!=64
                or any(char not in '0123456789abcdef' for char in request_sha256)
                or contract.get('request_sha256')!=request_sha256
                or contract.get('version') not in {'safety_review_v1','safety_review_v2'}
                or type(contract.get('sentence_count')) is not int or not 1<=contract['sentence_count']<=128
                or contract.get('plain_text') is not True
                or contract.get('finish_reason')!=('length' if error=='invalid_or_truncated_response' else 'stop')
                or not (structured_receipt_body_complete(row) if structured_stream_marked(row)
                        else body.get('complete') is True and body.get('capture_status')=='complete')
                or body.get('exception_kind') is not None or row.get('raw_total_tokens_valid') is not True
                or not all(type(usage.get(key)) is int and usage[key]>=0
                           for key in ('prompt_tokens','completion_tokens','total_tokens'))
                or usage['prompt_tokens']+usage['completion_tokens']!=usage['total_tokens']
                or row.get('fee_verification_error') is not None or row.get('unverified_cost_usd') is not None
                or cls._fee_attribution_error(row) is not None or type(usage.get('cost')) not in (int,float)):
            return False
        cost=Decimal(str(usage['cost']))
        return cost.is_finite() and cost>=0

    @classmethod
    def _settled_confirm_format_failure(cls, row):
        error = row.get('validation_error')
        if (row.get('role') != 'confirm_judge' or row.get('kind') != 'turn'
                or row.get('status') != 'failed' or row.get('failure_phase') != 'receipt_validation'
                or error not in {'invalid_or_truncated_response', 'invalid_response_message'}
                or row.get('failure_code') != error or row.get('fee_verification_error') is not None
                or not (structured_receipt_body_complete(row) if structured_stream_marked(row)
                        else row.get('body_read', {}).get('complete') is True)
                or row.get('body_read', {}).get('exception_kind') is not None
                or 'cost' not in row.get('usage', {}) or cls._fee_attribution_error(row) is not None):
            return False
        cost = Decimal(str(row['usage']['cost']))
        return cost.is_finite() and cost >= 0

    def _snapshot(self):
        state = super()._snapshot()
        # A live instance may have its registered review and online turn open.
        # No other guard, process, or restart may inherit that authorization.
        if any(token not in self._active_operation_tokens for (token,) in self._db.execute(
                "SELECT token FROM release_operations WHERE finished=0")):
            state.update(stopped=True, stop_reason=state.get('stop_reason') or 'unresolved_release_operation')
            return state
        # Never clear an explicit halt, unknown fee, pending request or storage
        # failure. Old failed rows have no disposition and remain stopped.
        if state.get('stop_reason') != 'request_failed' or state['unknown_cost_requests'] != 0:
            return state
        failed = [row for row in state['attempts'] if row['status'] == 'failed']
        if failed and all(row.get('confirm_judge_disposition') == 'unclear_known_cost_format_failure_v1'
                          and self._settled_confirm_format_failure(row) for row in failed):
            state.update(stopped=False, stop_reason=None)
            if state['calls_used'] >= self.max_requests:
                state.update(stopped=True, stop_reason='request_limit_reached')
            elif Decimal(state['known_cost_usd'] or '0') >= Decimal(self.threshold):
                state.update(stopped=True, stop_reason='observed_cost_threshold_reached')
        return state

    @contextmanager
    def scope(self,kind,operation_id,session_id):
        with super().scope(kind,operation_id,session_id):
            token=uuid.uuid4().hex
            with self._lock,self._transaction():
                self.checker()
                if self._snapshot()['stopped']:raise MinimalCallStopped('max_release_stopped')
                self._db.execute("INSERT INTO release_operations VALUES(?,?,?,?,0)",
                    (token,kind,operation_id,session_id))
                self._active_operation_tokens.add(token)
            try:
                yield
            except BaseException:
                self.halt()
                raise
            finally:
                try:
                    with self._lock,self._transaction():
                        self._db.execute("UPDATE release_operations SET finished=1 WHERE token=?",(token,))
                except BaseException:
                    self._broken=True
                    self.halt()
                    raise
                finally:
                    self._active_operation_tokens.discard(token)

    def _request_once(self,client,request,**options):
        # Keep capture/TLS state and its final binding inside the same physical
        # slot as transport. Review yields this slot before its next request.
        with self.dispatch_slot(),self._lock:
            directory=self.physical_dir/uuid.uuid4().hex;directory.mkdir()
            before={r['request_id'] for r in self._snapshot()['attempts']}
            self._physical_capture=directory
            try:return _request_once(self,client,request,**options)
            finally:
                self._physical_capture=None
                try:write_once(directory/'attempt-binding.json',{'role':client.role,'ledger_id':self.ledger_id,
                    'new_attempts':[r for r in self.snapshot()['attempts'] if r['request_id'] not in before]})
                except BaseException:self._broken=True;self.halt();raise

    def _open_observed(self,request,**kwargs):
        response=super()._open_observed(request,**kwargs)
        directory=self._physical_capture
        response.content_writer=lambda path,value:self.content_erasure.write(path,value,directory)
        return response

    def snapshot(self):
        if hasattr(self,'content_erasure'):self.content_erasure.check()
        state=super().snapshot()
        # Keep halt reasons compatible with the existing held-input and
        # validation dispositions. Explain the latest recorded request failure
        # separately; its local exception is not proof of a remote root cause.
        if (state.get('stopped') and state.get('stop_reason') in {
                'model_or_storage_failed', 'provider_cost_unknown', 'request_failed'}
                and state.get('attempts')):
            row=state['attempts'][-1]
            if row.get('status') == 'failed' and row.get('failure_phase'):
                diagnostic={'source':'latest_recorded_request', **{key:row.get(key) for key in (
                    'request_id','failure_phase','failure_code','http_status','fee_verification_error')}}
                failure=row.get('transport_failure')
                if failure is not None:
                    from ..transport_failure import validate_transport_failure
                    try:validate_transport_failure(failure)
                    except (ValueError,KeyError,TypeError):pass
                    else:diagnostic.update({key:failure[key] for key in ('exception_kind','errno_category')})
                state['stop_diagnostic']=diagnostic
        state.update(runtime_profile=('release_max_v1' if self._online_model() == MODELS['online_fast']
                                      else 'release_deepseek_v1'),paid_response_replays=0,
            logical_tls_extra_limit=1,connection_preparations_are_physical_model_attempts=False,
            no_experiment_budget_or_recovery_slots_inherited=True)
        return state
