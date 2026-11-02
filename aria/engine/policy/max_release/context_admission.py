"""Wait before context capture; never rewrite an already bound request."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

from ..minimal_calls import MinimalCallStopped, monotonic_value, wait_until
from .validation_support import require, digest_json


def wait_for_slot(guard):
    """The original pacing algorithm, before any reservation or physical row."""
    pacing = None
    previous = None
    if guard._pacing_interval:
        state = guard._snapshot()
        if state['stopped']:
            raise MinimalCallStopped(state['stop_reason'])
        previous = state['attempts'][-1]['request_id'] if state['attempts'] else None
        try:
            if previous is not None and previous != guard._last_network_request_id:
                deadline = monotonic_value(guard._pacing_clock) + guard._pacing_interval
            elif guard._last_network_started is not None:
                deadline = guard._last_network_started + guard._pacing_interval
            elif guard._reopen_pacing_required:
                deadline = monotonic_value(guard._pacing_clock) + guard._pacing_interval
            else:
                deadline = None
            waited = wait_until(deadline, monotonic=guard._pacing_clock,
                                sleep=guard._pacing_sleep, cancelled=guard._stop_requested.is_set)
        except BaseException as error:
            guard.halt()
            if not isinstance(error, Exception):
                raise
            raise MinimalCallStopped('request_pacing_wait_failed') from None
        pacing = {'min_interval_seconds':guard._pacing_interval, 'wait_ms':round(waited,3),
                  'start_gap_ms':None, 'scope':'one_guard_all_roles', 'clock':'monotonic'}
    return pacing, previous


class ContextAdmissionMixin:
    @contextmanager
    def online_context_admission(self, client, ctx, persona, utterance):
        with self._context_admission(client, ctx, persona, utterance, role='online_fast'):
            yield

    @contextmanager
    def confirm_context_admission(self, client, ctx, persona, utterance):
        """Recheck an existing invitation after pacing, before reserving a call."""
        with self._context_admission(client, ctx, persona, utterance, role='confirm_judge'):
            token = self._local.online_context
            def before_commit():
                # A local capability for this transaction, never prompt/state data.
                with self._lock:
                    self._check_online_context(token=token)
            yield before_commit

    @contextmanager
    def _context_admission(self, client, ctx, persona, utterance, *, role):
        """Hold the same local dispatch lock while waiting/building/dispatching.

        The SQLite expected-prior check still rejects a competing process. No
        paid row, request identity, reservation or physical deadline exists yet.
        """
        with self._lock:
            require(getattr(self._local,'online_context',None) is None, 'online_context_reentry')
            require(role in {'online_fast', 'confirm_judge'} and client.role==role
                    and client.model_request_control is self,
                    'online_context_wrong_client')
            scope = deepcopy(getattr(self._local,'scope',None))
            require(type(scope) is dict and scope['kind']=='turn' and scope['session_id']==ctx.session_id,
                    'online_context_wrong_scope')
            authority = getattr(self, '_online_authority_reader', None)
            require(callable(authority), 'online_context_authority_missing')
            token = {'client':client, 'ctx':ctx, 'persona':persona, 'utterance':utterance, 'role':role,
                     'identity':(ctx.user_id,ctx.session_id,ctx.scene_id,len(ctx.turns)),
                     'scope':scope, 'authority_reader':authority, 'authority':deepcopy(authority()),
                     'binding':self._binding(client), 'bound':False, 'consumed':False,
                     'original_invitation':deepcopy(ctx.working_memory.get('_preference_dialogue_invitation'))}
            self._local.online_context = token
            try:
                self._check_online_context(check_invitation=False)
                token['pacing'], token['previous'] = wait_for_slot(self)
                self._check_online_context(check_invitation=False)
                yield
                if role == 'confirm_judge':
                    self._check_online_context()
            finally:
                self._local.online_context = None

    def _check_online_context(self, *, check_invitation=True, token=None):
        if token is None:
            token = getattr(self._local,'online_context',None)
        if token is None:
            return
        self._candidate_contract()
        require(not self._broken and not self._closed and not self._stop_requested.is_set(),
                'online_context_stopped')
        ctx, persona = token['ctx'], token['persona']
        require((ctx.user_id,ctx.session_id,ctx.scene_id,len(ctx.turns))==token['identity']
                and persona is not None and persona.user_id==ctx.user_id
                and getattr(self._local,'scope',None)==token['scope']
                and self._binding(token['client'])==token['binding'], 'online_context_identity_changed')
        require(getattr(self,'_online_authority_reader',None) is token['authority_reader']
                and token['authority_reader']()==token['authority'], 'online_context_authority_changed')
        require(ctx.working_memory.get('_preference_dialogue_invitation')==token['original_invitation'],
                'online_context_invitation_mutated')
        from ...memory.preference_dialogue import current_preference_invitation, current_judged_preference_invitation
        if token['role'] == 'confirm_judge':
            # Explicit durable stop/configuration changes veto a transaction.
            # A naturally exhausted count/cost cap only blocks the next call;
            # it does not invalidate this already received judge result.
            require(self._db.execute('SELECT max_requests,threshold,halted FROM settings WHERE id=1').fetchone()
                    == (self.max_requests, self.threshold, None), 'confirmation_context_stopped_or_changed')
            current = current_judged_preference_invitation(ctx, persona, token['utterance'])
            require(current is not None and current==token['original_invitation'],
                    'confirmation_context_invitation_changed')
        else:
            current = current_preference_invitation(persona, ctx)
        if check_invitation:
            require(token['bound'] and current==token['invitation'],
                    'online_context_invitation_changed')

    def bind_online_context(self, client, ctx, persona, utterance, messages, requirements, invitation):
        token = getattr(self._local,'online_context',None)
        require(token is not None and token['role']=='online_fast' and not token['bound'] and token['client'] is client
                and token['ctx'] is ctx and token['persona'] is persona and token['utterance']==utterance,
                'online_context_binding_mismatch')
        from ..online_json import prepare_json_messages
        from ..online_decision import protocol_version
        from ..release_prompt import uses_compact_prompt
        prepared = client._prepare_online_messages(deepcopy(messages), channel="plain_json")
        wire = prepare_json_messages(prepared, protocol_version(prepared), **requirements,
                                     compact_schema=uses_compact_prompt(prepared))
        token.update(bound=True, invitation=deepcopy(invitation), messages=wire,
                     requirements=deepcopy(requirements))
        self._check_online_context()

    def bind_confirm_context(self, client, ctx, persona, utterance, messages):
        token = getattr(self._local,'online_context',None)
        require(token is not None and token['role']=='confirm_judge' and not token['bound']
                and token['client'] is client and token['ctx'] is ctx
                and token['persona'] is persona and token['utterance']==utterance,
                'confirmation_context_binding_mismatch')
        from ...memory.preference_dialogue import current_judged_preference_invitation, validate_confirm_judge_messages
        validate_confirm_judge_messages(messages)
        token.update(bound=True, invitation=current_judged_preference_invitation(ctx,persona,utterance),
                     messages=deepcopy(messages), requirements={})
        self._check_online_context()

    def _pacing_admission(self, client, request, requirements):
        token = getattr(self._local,'online_context',None)
        if token is None:
            return wait_for_slot(self)
        self._check_online_context()
        supplied = {k:v for k,v in requirements.items() if v}
        require(not token['consumed'] and token['client'] is client
                and supplied==token['requirements']
                and json.loads(request.data)['messages']==token['messages'], 'online_context_wire_mismatch')
        token['consumed']=True
        token['request_sha256']=hashlib.sha256(request.data).hexdigest()
        return deepcopy(token['pacing']), token['previous']

    def _bind_online_admission_row(self, row):
        token = getattr(self._local,'online_context',None)
        if token is None:
            return
        require(token['consumed'] and row['request_sha256']==token['request_sha256'],
                'online_context_final_request_changed')
        field = 'confirmation_context_admission' if token['role']=='confirm_judge' else 'online_context_admission'
        row[field] = {
            'schema':'aria_confirmation_context_admission_v1' if token['role']=='confirm_judge' else 'aria_online_context_admission_v1',
            'admission_before_context':True, 'physical_rebuilds':0,
            'request_id':row['request_id'], 'request_sha256':row['request_sha256'],
            'owner_id':token['identity'][0], 'session_id':token['identity'][1],
            'turn_index':token['identity'][3],
            'owner_message_sha256':hashlib.sha256(token['utterance'].encode()).hexdigest(),
            'messages_sha256':digest_json(token['messages']),
            'requirements':deepcopy(token['requirements']),
            'invitation_sha256':digest_json(token['invitation']),
            'original_invitation_expires_at':(token['invitation'] or {}).get('expires_at'),
            'authority_sha256':digest_json(token['authority']),
        }
