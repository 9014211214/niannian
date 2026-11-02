"""Bounded SSE framing for one reflection; no partial output or raw stream log."""
from copy import deepcopy
from hashlib import sha256
import json
import http.client
from pathlib import Path

from aria.engine.policy.response_body import (ResponseDeadline, CHUNK_BYTES, _framing, _kind,
    ResponseFramingError, ResponseBodyLimitError, ResponseCaptureError)
from aria.engine.policy.response_observability import safe_usage, safe_generation_id

VERSION = 'reflection_sse_incremental_v2'
STRUCTURED_STREAM_VERSION = 'structured_sse_application_v1'
STRUCTURED_ROLES = frozenset({'memory', 'safety_review', 'confirm_judge'})
HIDDEN = {'reasoning', 'reasoning_details', 'reasoning_content', 'analysis', 'thinking'}
PROTOCOL_CODES = frozenset({
    'bytes_required', 'event_type', 'unknown_sse_field', 'unterminated_event',
    'data_after_done', 'early_done', 'nonfinite', 'error_event', 'object_type',
    'hidden_channel', 'identity_shape', 'identity_changed', 'choices',
    'data_after_usage', 'empty_nonaccounting_event', 'choice_identity', 'delta',
    'unknown_delta_field', 'non_content_channel', 'role', 'content_type',
    'nonstop_finish', 'usage_order', 'usage_shape', 'usage_incomplete',
    'hidden_token_usage', 'data_after_finish', 'incomplete_stream',
    'identity_missing', 'header_identity_conflict', 'empty_content',
    'duplicate_field', 'capture_failed', 'incremental_reader_required',
    'unconsumed_event_tail', 'chunk_validator_missing', 'finish_changed',
})


class SSEProtocolError(ValueError):
    def __init__(self, code):
        if code not in PROTOCOL_CODES: raise ValueError('invalid_sse_failure_code')
        self.code = code
        super().__init__('reflection_sse_' + code)


def _reject(code):
    raise SSEProtocolError(code)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: _reject('duplicate_field')
        result[key] = value
    return result


class IncrementalReflectionSSE:
    """Dispatch only blank-line-terminated SSE events; keep partial bytes private."""
    def __init__(self, *, observation=None, header_observation=None, receipt_mode=None):
        if receipt_mode not in (None, 'online_sse_application_v1', STRUCTURED_STREAM_VERSION):
            raise ValueError('invalid_sse_receipt_mode')
        self.receipt_mode = receipt_mode
        self.finish_reasons = ('stop', 'length') if receipt_mode else ('stop',)
        self.finish_reason = None
        self.stats = observation if observation is not None else {}
        self.stats.update(schema=VERSION, complete=False, comments=0, data_events=0,
            usage_events=0, done_events=0, content_characters=0,
            finish_observed=False, usage_observed=False, done_observed=False,
            partial_output_submitted=False, usage_is_verified_fee=False, deadline_extended=False)
        self.header_observation = header_observation
        self.identity, self.parts, self.public_frames = {}, [], []
        self.finished, self.used, self.done, self.usage = False, False, False, None
        self.line = bytearray(); self.data_lines = []; self.event_type = ''
        self.skip_lf = False; self.first_line = True
        if receipt_mode:
            self.stats.update(receipt_mode=receipt_mode, finish_reason=None)

    def _progress(self):
        self.stats.update(finish_observed=self.finished, usage_observed=self.used,
                          done_observed=self.done)
        if self.receipt_mode:
            self.stats['finish_reason'] = self.finish_reason
        candidate = self.identity.get('id')
        if safe_generation_id(candidate) is not None: self.stats['generation_id'] = candidate

    def feed(self, raw):
        if type(raw) is not bytes: _reject('bytes_required')
        for value in raw:
            if self.skip_lf:
                self.skip_lf = False
                if value == 10: continue
            if value in (10, 13):
                self._line(bytes(self.line)); self.line.clear()
                self.skip_lf = value == 13
            else: self.line.append(value)

    def _line(self, raw):
        line = raw.decode('utf-8-sig' if self.first_line else 'utf-8')
        self.first_line = False
        if line == '':
            if self.data_lines:
                if self.event_type not in ('', 'message'): _reject('event_type')
                payload = '\n'.join(self.data_lines); self.data_lines = []
                self._event(payload)
            self.event_type = ''
        elif line.startswith(':'): self.stats['comments'] += 1
        else:
            field, sep, value = line.partition(':')
            if sep and value.startswith(' '): value = value[1:]
            if field == 'data': self.data_lines.append(value)
            elif field == 'event': self.event_type = value
            elif field not in ('id', 'retry'): _reject('unknown_sse_field')

    def _event(self, payload):
        stats = self.stats
        if self.done: _reject('data_after_done')
        if payload.strip() == '[DONE]':
            if not self.finished or not self.used: _reject('early_done')
            self.done = True; stats['done_events'] += 1
            self._progress(); return
        stats['data_events'] += 1
        obj = json.loads(payload, object_pairs_hook=_unique, parse_constant=lambda _: _reject('nonfinite'))
        if type(obj) is not dict: _reject('error_event')
        if obj.get('error') is not None:
            error = obj['error']
            code = error.get('code') if type(error) is dict else None
            if type(code) is int and 100 <= code <= 599:
                stats['provider_error_code'] = code
            _reject('error_event')
        if obj.get('object') not in (None, 'chat.completion.chunk'): _reject('object_type')
        if any(obj.get(key) not in (None, '', []) for key in HIDDEN): _reject('hidden_channel')
        for key in ('id', 'model', 'provider'):
            if key not in obj: continue
            value = obj[key]
            if type(value) is not str or not value or len(value) > 200 or any(ord(c) < 32 for c in value):
                _reject('identity_shape')
            if key in self.identity and self.identity[key] != value: _reject('identity_changed')
            self.identity[key] = value
        choices = obj.get('choices')
        if type(choices) is not list or len(choices) > 1: _reject('choices')
        is_usage = 'usage' in obj and obj['usage'] is not None
        if self.used: _reject('data_after_usage')
        if not choices and not is_usage: _reject('empty_nonaccounting_event')
        content, finish, role = '', None, None
        if choices:
            choice = choices[0]
            if type(choice) is not dict or type(choice.get('index')) is not int or choice['index'] != 0:
                _reject('choice_identity')
            if choice.get('error') is not None: _reject('error_event')
            delta = choice.get('delta')
            if type(delta) is not dict: _reject('delta')
            if set(delta) - HIDDEN - {'role', 'content', 'refusal', 'tool_calls', 'function_call'}:
                _reject('unknown_delta_field')
            if any(delta.get(key) not in (None, '', []) for key in HIDDEN): _reject('hidden_channel')
            if (delta.get('refusal') not in (None, '') or delta.get('tool_calls') not in (None, [])
                    or delta.get('function_call') is not None): _reject('non_content_channel')
            role = delta.get('role')
            if role not in (None, 'assistant'): _reject('role')
            content = delta.get('content')
            if content is None: content = ''
            if type(content) is not str: _reject('content_type')
            finish = choice.get('finish_reason')
            if finish is not None and finish not in self.finish_reasons: _reject('nonstop_finish')
            if self.finished and finish is not None and finish != self.finish_reason:
                _reject('finish_changed')
        if is_usage:
            if content or (not self.finished and finish not in self.finish_reasons): _reject('usage_order')
            if finish in self.finish_reasons:
                self.finished = True; self.finish_reason = finish
            source = obj['usage']
            if type(source) is not dict: _reject('usage_shape')
            self.usage = safe_usage({'usage': source})
            if ('cost' not in self.usage or any(type(self.usage.get(key)) is not int or self.usage[key] < 0
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                    or self.usage['total_tokens'] != self.usage['prompt_tokens'] + self.usage['completion_tokens']):
                _reject('usage_incomplete')
            details = source.get('completion_tokens_details')
            if type(details) is dict and details.get('reasoning_tokens') not in (None, 0):
                _reject('hidden_token_usage')
            self.used = True; stats['usage_events'] += 1
        else:
            if self.finished: _reject('data_after_finish')
            self.parts.append(content); stats['content_characters'] += len(content)
            if finish in self.finish_reasons:
                self.finished = True; self.finish_reason = finish
        self.public_frames.append({**{key: obj[key] for key in ('id', 'model', 'provider') if key in obj},
            'content': content, 'finish_reason': finish, 'role': role,
            'accounting': is_usage, **({'usage': deepcopy(self.usage)} if is_usage else {})})
        self._progress()

    def finish(self, raw):
        if self.line or self.data_lines or self.event_type: _reject('unterminated_event')
        stats = self.stats
        if not self.done or not self.finished or not self.used: _reject('incomplete_stream')
        if set(self.identity) != {'id', 'model', 'provider'} or safe_generation_id(self.identity['id']) is None:
            _reject('identity_missing')
        candidate = self.header_observation or {}
        if candidate.get('generation_id_candidate_status') == 'conflicting': _reject('header_identity_conflict')
        if (candidate.get('generation_id_candidate_status') == 'accepted'
                and candidate.get('generation_id_candidate') != self.identity['id']): _reject('header_identity_conflict')
        content = ''.join(self.parts)
        if not content.strip() and not self.receipt_mode: _reject('empty_content')
        data = {**self.identity, 'object': 'chat.completion', 'usage': self.usage,
                'choices': [{'index': 0, 'finish_reason': self.finish_reason,
                             'message': {'role': 'assistant', 'content': content}}]}
        assembled = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        stats.update(complete=True, stream_bytes=len(raw), stream_sha256=sha256(raw).hexdigest(),
                     assembled_sha256=sha256(assembled).hexdigest(), public_content_sha256=sha256(content.encode()).hexdigest(),
                     generation_id=self.identity['id'], partial_output_submitted=False,
                     usage_is_verified_fee=False, deadline_extended=False)
        return assembled, self.public_frames


def parse_reflection_sse(raw, *, observation=None, header_observation=None, receipt_mode=None):
    parser = IncrementalReflectionSSE(observation=observation, header_observation=header_observation,
                                     receipt_mode=receipt_mode)
    parser.feed(raw)
    return parser.finish(raw)



BODY_SCHEMA = 'bounded_sse_application_body_v2'
BODY_FIELDS = frozenset({'schema', 'max_bytes', 'bytes_received', 'body_sha256', 'framing',
    'content_length', 'framing_checked', 'complete', 'application_complete', 'http_message_complete',
    'http_eof_observed', 'current_chunk_terminated', 'exception_kind', 'json_status',
    'read_mode', 'capture_status'})
BODY_EXCEPTIONS = frozenset({'capture_failed', 'body_limit', 'framing_invalid', 'incomplete_read',
    'remote_disconnected', 'connection_reset', 'timeout', 'http_protocol_error', 'other_read_error'})


def validate_stream_body_observation(value):
    """Strict evidence validation only; does not itself grant a retry or known fee."""
    import re
    if (type(value) is not dict or set(value) != BODY_FIELDS or value['schema'] != BODY_SCHEMA
        or type(value['max_bytes']) is not int or value['max_bytes'] < 1
        or type(value['bytes_received']) is not int or not 0 <= value['bytes_received'] <= value['max_bytes'] + 1
        or type(value['body_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', value['body_sha256'])
        or value['framing'] not in ('unobserved', 'chunked', 'content_length', 'close_delimited')
        or value['content_length'] is not None and (type(value['content_length']) is not int or not 0 <= value['content_length'] < 10**20)
        or any(type(value[k]) is not bool for k in ('framing_checked', 'complete', 'application_complete',
            'http_message_complete', 'http_eof_observed', 'current_chunk_terminated'))
        or value['exception_kind'] is not None and (type(value['exception_kind']) is not str or value['exception_kind'] not in BODY_EXCEPTIONS)
        or value['json_status'] != 'not_inspected'
        or value['read_mode'] not in ('unobserved', 'read1')
        or value['capture_status'] not in ('not_requested', 'complete', 'failed')):
        raise ValueError('invalid_stream_body_observation')
    if (value['complete'] != value['http_message_complete']
        or value['framing_checked'] != (value['framing'] != 'unobserved')
        or (value['framing'] == 'content_length') != (value['content_length'] is not None)
        or value['bytes_received'] == 0 and value['body_sha256'] != sha256(b'').hexdigest()
        or value['current_chunk_terminated'] and value['framing'] != 'chunked'
        or value['http_eof_observed'] and not value['http_message_complete']
        or value['http_message_complete'] and (not value['framing_checked']
            or value['content_length'] is not None and value['bytes_received'] != value['content_length'])
        or value['application_complete'] and (not value['framing_checked'] or value['read_mode'] != 'read1'
            or value['bytes_received'] > value['max_bytes']
            or value['exception_kind'] not in (None, 'capture_failed')
            or value['framing'] == 'chunked' and not value['current_chunk_terminated'])
        or (value['capture_status'] == 'failed') != (value['exception_kind'] == 'capture_failed')):
        raise ValueError('invalid_stream_body_observation')
    return value


def structured_sse_contract(role):
    from aria.engine.policy.online_structured import DEEPSEEK_MODEL
    if role not in STRUCTURED_ROLES | {'reflection'}:
        raise ValueError('invalid_structured_stream_role')
    return {'schema': STRUCTURED_STREAM_VERSION, 'reader_protocol': VERSION,
        'receipt_mode': STRUCTURED_STREAM_VERSION, 'role': role,
        'model': DEEPSEEK_MODEL, 'provider': 'together', 'physical_timeout_seconds': 90,
        'response_format_preserved': True, 'terminal_receipt_finishes': ['stop', 'length'],
        'required_delivery_finish': 'stop', 'required_terminal_usage': True, 'required_done': True}


def structured_stream_marked(row):
    """Missing one SSE evidence field must never downgrade to finite HTTP."""
    return (any(key in row for key in ('structured_stream_contract', 'structured_stream_receipt', 'stream_read'))
            or row.get('body_read', {}).get('schema') == BODY_SCHEMA)


def structured_receipt_body_complete(row, public_content=None):
    """One application-terminal proof for these roles; never grants known fees."""
    try:
        if (json.dumps(row.get('structured_stream_contract'), sort_keys=True, allow_nan=False)
                != json.dumps(structured_sse_contract(row['role']), sort_keys=True, allow_nan=False)
                or row['model'] != row['structured_stream_contract']['model']
                or row['expected_provider'] != 'Together'):
            return False
        body = validate_stream_body_observation(row['body_read'])
        stream, shape = row['stream_read'], row['structured_stream_receipt']
        if (body['application_complete'] is not True or body['capture_status'] != 'complete'
                or body['exception_kind'] is not None
                or stream.get('schema') != VERSION or stream.get('receipt_mode') != STRUCTURED_STREAM_VERSION
                or any(stream.get(k) is not True for k in ('complete', 'application_complete',
                    'finish_observed', 'usage_observed', 'done_observed', 'closed_after_done'))
                or any(stream.get(k) is not False for k in ('partial_output_submitted',
                    'usage_is_verified_fee', 'deadline_extended'))
                or any(type(stream.get(k)) is not int or stream[k] != 1 for k in ('usage_events', 'done_events'))
                or any(stream.get(k) is not body[k] for k in
                    ('http_message_complete', 'http_eof_observed', 'current_chunk_terminated'))
                or stream.get('stream_bytes') != body['bytes_received']
                or stream.get('stream_sha256') != body['body_sha256']
                or safe_generation_id(stream.get('generation_id')) is None
                or stream['generation_id'] != row.get('observation', {}).get('generation_id')
                or shape.get('request_sha256') != row['request_sha256'] or shape.get('plain_text') is not True
                or shape.get('finish_reason') not in ('stop', 'length')
                or shape['finish_reason'] != stream.get('finish_reason')
                or shape.get('public_content_sha256') != stream.get('public_content_sha256')):
            return False
        for key in ('stream_sha256', 'assembled_sha256', 'public_content_sha256'):
            value = stream.get(key)
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                return False
        return public_content is None or (type(public_content) is str
            and sha256(public_content.encode()).hexdigest() == stream['public_content_sha256']
            and stream['content_characters'] == len(public_content))
    except (KeyError, TypeError, ValueError, RecursionError, UnicodeError):
        return False


def read_reflection_sse(response, *, limit, deadline, observation, stream_observation, header_observation,
                        receipt_mode=None):
    """Return at verified SSE DONE; never wait for the next HTTP chunk or EOF.

    The current declared chunk and its CRLF must be complete. The original
    observed HTTP reader validates earlier chunk framing; no transport decoder
    is replaced. Same absolute deadline and raw body byte cap apply throughout.
    """
    if (type(limit) is not int or limit < 1 or type(deadline) is not ResponseDeadline
            or type(observation) is not dict or type(stream_observation) is not dict):
        raise ValueError('invalid_stream_reader_configuration')
    observation.update(schema=BODY_SCHEMA, max_bytes=limit, bytes_received=0,
        body_sha256=sha256(b'').hexdigest(), framing='unobserved', content_length=None,
        framing_checked=False, complete=False, application_complete=False, http_message_complete=False,
        http_eof_observed=False, current_chunk_terminated=False, exception_kind=None,
        json_status='not_inspected', read_mode='unobserved', capture_status='not_requested')
    parser = IncrementalReflectionSSE(observation=stream_observation, header_observation=header_observation,
                                     receipt_mode=receipt_mode)
    body = bytearray(); digest = sha256()

    def consume(part):
        if type(part) is not bytes: raise ResponseFramingError('non_bytes_response_body')
        accounted = part[:max(0, limit + 1 - len(body))]
        body.extend(accounted); digest.update(accounted)
        observation.update(bytes_received=len(body), body_sha256=digest.hexdigest())
        if len(body) > limit or len(part) > limit: raise ResponseBodyLimitError('response_body_limit')
        deadline.remaining()
        parser.feed(part)

    def read_part(size):
        deadline.remaining()
        part = response.read1(min(size, limit + 1 - len(body)))
        consume(part)
        return part

    try:
        deadline.remaining()
        headers = getattr(response, 'headers', {})
        getter = getattr(headers, 'get_all', None)
        types = getter('Content-Type', []) if callable(getter) else [headers.get('Content-Type')]
        if (len(types) != 1 or type(types[0]) is not str
                or types[0].split(';', 1)[0].strip().lower() != 'text/event-stream'):
            _reject('content_type')
        framing, length = _framing(response)
        observation.update(framing=framing, content_length=length, framing_checked=True)
        if length is not None and length > limit: raise ResponseBodyLimitError('response_body_limit')
        if not callable(getattr(response, 'read1', None)): _reject('incremental_reader_required')
        observation['read_mode'] = 'read1'
        while not parser.done:
            part = read_part(CHUNK_BYTES)
            if not part:
                if length is not None and len(body) != length:
                    raise http.client.IncompleteRead(b'', max(0, length - len(body)))
                observation.update(http_eof_observed=True, http_message_complete=True, complete=True)
                break
        if parser.done:
            if framing == 'chunked':
                # read1 leaves a consumed chunk's CRLF for the NEXT read call.
                # Verify only this known boundary, never request next chunk size.
                if (getattr(response, '_strict_framing', None) is not True
                        or not callable(getattr(response, '_safe_read', None))):
                    _reject('chunk_validator_missing')
                remaining = getattr(response, 'chunk_left', None)
                if type(remaining) is not int or remaining < 0:
                    raise ResponseFramingError('invalid_current_chunk_state')
                while remaining:
                    part = read_part(min(CHUNK_BYTES, remaining))
                    if not part: raise http.client.IncompleteRead(b'', remaining)
                    next_remaining = getattr(response, 'chunk_left', None)
                    if type(next_remaining) is not int or next_remaining != remaining - len(part):
                        raise ResponseFramingError('invalid_current_chunk_state')
                    remaining = next_remaining
                deadline.remaining()
                if response._safe_read(2) != b'\r\n':
                    raise ResponseFramingError('invalid_chunk_terminator')
                deadline.remaining()
                observation['current_chunk_terminated'] = True
            elif length is not None:
                while len(body) < length:
                    if not read_part(min(CHUNK_BYTES, length - len(body))):
                        raise http.client.IncompleteRead(b'', length - len(body))
                if len(body) != length: raise ResponseFramingError('response_length_mismatch')
                observation.update(http_message_complete=True, complete=True)
        assembled, frames = parser.finish(bytes(body))
        deadline.remaining()
        close = getattr(response, 'close', None)
        if callable(close): close()
        deadline.remaining()
        observation['application_complete'] = True
        stream_observation.update(application_complete=True, http_eof_observed=observation['http_eof_observed'],
            http_message_complete=observation['http_message_complete'],
            current_chunk_terminated=observation['current_chunk_terminated'], closed_after_done=callable(close))
        capture_body = getattr(response, 'observe_application_body', None)
        capture_stream = getattr(response, 'observe_complete_stream', None)
        try:
            if callable(capture_body): capture_body(observation=deepcopy(observation))
            if callable(capture_stream): capture_stream(assembled, frames, stream_observation)
        except Exception:
            observation.update(capture_status='failed', exception_kind='capture_failed')
            raise ResponseCaptureError('response_capture_failed') from None
        if callable(capture_body) or callable(capture_stream): observation['capture_status'] = 'complete'
        return assembled
    except Exception as error:
        code = (error.code if type(error) is SSEProtocolError else
                'utf8_invalid' if isinstance(error, UnicodeDecodeError) else
                'json_invalid' if isinstance(error, json.JSONDecodeError) else
                'deadline_exhausted' if isinstance(error, TimeoutError) else
                'transport_or_capture_failure')
        stream_observation.update(schema=VERSION, complete=False, application_complete=False,
            partial_output_submitted=False, usage_is_verified_fee=False, deadline_extended=False, failure_code=code)
        observation.update(bytes_received=len(body), body_sha256=sha256(body).hexdigest(), exception_kind=_kind(error))
        if observation['capture_status'] != 'failed':
            capture_body = getattr(response, 'observe_failed_body', None)
            capture_stream = getattr(response, 'observe_failed_stream', None)
            try:
                if callable(capture_body): capture_body(observation=deepcopy(observation))
                if callable(capture_stream): capture_stream(stream_observation)
            except Exception:
                observation.update(capture_status='failed', exception_kind='capture_failed')
                raise ResponseCaptureError('response_capture_failed') from None
            if callable(capture_body) or callable(capture_stream): observation['capture_status'] = 'complete'
        raise


class SSEPublicCapture:
    """Persist only validated public frames; raw/hidden stream bytes stay in RAM."""
    def __init__(self, raw, directory):
        self.raw, self.directory, self.captured, self.body = raw, Path(directory), False, None

    def __getattr__(self, name): return getattr(self.raw, name)
    def __enter__(self): self.raw = self.raw.__enter__(); return self
    def __exit__(self, *args): return self.raw.__exit__(*args)
    def observe_application_body(self, *, observation): self.body = deepcopy(observation)
    def observe_failed_body(self, *, observation): self.body = deepcopy(observation)

    def observe_complete_stream(self, assembled, frames, stream_observation):
        from .public_capture import write_once, projection,write_visible
        if self.captured: raise ValueError('stream_capture_reused')
        self.captured = True
        visible, obs = projection(assembled)
        saved=write_visible(self,self.directory / 'first-visible.json', visible)
        write_once(self.directory / 'public-response-observation.json', obs)
        frames_saved=write_visible(self,self.directory / 'stream-public-frames.json', {'schema': VERSION, 'frames': frames,
            'raw_stream_saved': False, 'hidden_reasoning_saved': False})
        write_once(self.directory / 'first-body-receipt.json', {'transformation': VERSION,
            'partial': False, 'body_read': self.body, 'stream_read': deepcopy(stream_observation),
            'raw_stream_scope': 'observed_sse_prefix_through_current_chunk',
            'unread_http_tail_possible': not self.body['http_message_complete'],
            'full_body_stored': False, 'hidden_reasoning_stored': False,
            'byte_identical_response_stored': False, 'visible_projection_saved': saved['saved'] and frames_saved['saved'],
            'assembled_envelope_sha256': sha256(assembled).hexdigest(),
            'public_projection_sha256': saved['sha256'],
            'public_frames_sha256': frames_saved['sha256']})

    def observe_failed_stream(self, observation):
        from .public_capture import write_once
        if self.captured: return
        self.captured = True
        write_once(self.directory / 'first-body-receipt.json', {'transformation': VERSION,
            'partial': True, 'body_read': self.body, 'stream_read': deepcopy(observation),
            'raw_stream_scope': 'observed_prefix_before_failure',
            'unread_http_tail_possible': True,
            'full_body_stored': False, 'hidden_reasoning_stored': False,
            'byte_identical_response_stored': False, 'visible_projection_saved': False,
            'visible_capture_missing_reason': 'stream_not_complete_or_protocol_rejected'})
