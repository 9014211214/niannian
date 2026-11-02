"""Bounded first-response projection; original response bytes never reach disk."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import Lock

from aria.engine.policy.minimal_calls import MAX_MODEL_RESPONSE_BYTES
from aria.engine.policy.response_observability import safe_usage, receipt_identity

VERSION = "aria_max_public_response_projection_v1"
HIDDEN = frozenset({"reasoning", "reasoning_details", "reasoning_content", "analysis", "thinking"})


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def write_visible(capture,path,value):
    writer=getattr(capture,'content_writer',None)
    if writer is not None:return writer(path,value)
    write_once(path,value)
    return {'saved':True,'sha256':sha256(Path(path).read_bytes()).hexdigest()}


def projection(raw):
    class Pairs(list):
        pass
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=Pairs)
    removed = {key: 0 for key in HIDDEN}
    def generic(node):
        if isinstance(node, Pairs):
            result = []
            for key, item in node:
                if key in HIDDEN:
                    removed[key] += 1
                else:
                    result.append([key, generic(item)])
            return {"object_pairs": result}
        if isinstance(node, list):
            return [generic(item) for item in node]
        if isinstance(node, float) and not math.isfinite(node):
            return {"invalid_number": True}
        return node
    def selected(node, fields):
        if not isinstance(node, Pairs):
            return {"invalid_object_shape": True}
        result = []
        for key, item in node:
            if key in HIDDEN:
                removed[key] += 1
            elif key in fields:
                result.append([key, fields[key](item)])
        return {"object_pairs": result}
    def choice(node):
        return selected(node, {"index": generic, "finish_reason": generic,
            "message": lambda message: selected(message, {key: generic for key in
                ("role", "content", "refusal", "tool_calls", "function_call")})})
    visible = selected(value, {"choices": lambda choices: [choice(item) for item in choices]
        if type(choices) is list else {"invalid_choices": True}})
    # These are observations only; the durable guard owns identity/usage/fee
    # verification, including duplicate-key handling. Never quote error bodies.
    ordinary = json.loads(raw.decode("utf-8"))
    usage = safe_usage(ordinary) if type(ordinary) is dict else {}
    identity = receipt_identity(ordinary, models=["qwen/qwen3.8-max-0902", "qwen/qwen3.8-max-20260902",
        "deepseek/deepseek-v4.1-flash-1017", "deepseek/deepseek-v4.1-flash-20261017",
        "anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-20260630"], providers=["Alibaba", "Anthropic", "Together", "Fireworks"]
        ) if type(ordinary) is dict else None
    return visible, {"usage_observation": usage, "usage_is_verified_fee": False,
                     "identity_states": identity, "hidden_fields_removed": removed}


class PublicFirstReadCapture:
    def __init__(self, raw, directory):
        self.raw, self.directory, self.captured = raw, Path(directory), False

    def __getattr__(self, key):
        return getattr(self.raw, key)

    def __enter__(self):
        self.raw = self.raw.__enter__()
        return self

    def __exit__(self, *args):
        return self.raw.__exit__(*args)

    def read(self, size=-1):
        # Legacy path still supports one complete bounded read. Never project a
        # partial exception buffer, even if its bytes happen to be valid JSON.
        raw = self.raw.read(size)
        return raw

    def observe_complete_body(self, raw, *, observation):
        self.save(raw)

    def observe_failed_body(self, *, observation):
        if self.captured:
            return
        self.captured = True
        write_once(self.directory / "first-body-receipt.json", {
            "transformation": VERSION, "partial": True,
            "body_read": deepcopy(observation), "full_body_stored": False,
            "hidden_reasoning_stored": False, "byte_identical_response_stored": False,
            "visible_projection_saved": False,
            "visible_capture_missing_reason": "transport_body_incomplete_no_projection"})

    def save(self, raw, partial=False):
        if self.captured:
            return
        self.captured = True
        receipt = {"transformation": VERSION, "body_bytes": len(raw) if type(raw) is bytes else None,
            "body_sha256": sha256(raw).hexdigest() if type(raw) is bytes else None,
            "partial": partial, "full_body_stored": False, "hidden_reasoning_stored": False,
            "byte_identical_response_stored": False, "visible_projection_saved": False}
        if not partial and type(raw) is bytes and len(raw) <= MAX_MODEL_RESPONSE_BYTES:
            try:
                visible, observation = projection(raw)
            except (ValueError, UnicodeError, TypeError, RecursionError):
                receipt["visible_capture_missing_reason"] = "outer_json_not_parseable"
            else:
                saved=write_visible(self,self.directory / "first-visible.json", visible)
                write_once(self.directory / "public-response-observation.json", observation)
                receipt["visible_projection_saved"] = saved['saved']
                receipt["public_projection_sha256"] = saved['sha256']
                if not saved['saved']:receipt['visible_capture_missing_reason']='owner_erased'
        write_once(self.directory / "first-body-receipt.json", receipt)


def begin_reflection_observation(client):
    from aria.engine.policy.reserved_calls import ReservedCallGuard
    guard=getattr(client,'model_request_control',None)
    if not isinstance(guard,ReservedCallGuard):raise ValueError('actual_reflection_guard_required')
    with guard._lock:
        scope=deepcopy(getattr(guard._local,'scope',None))
        if not scope or scope.get('kind')!='review':raise ValueError('actual_reflection_scope_required')
        rows=[json.loads(v[0]) for v in guard._db.execute('SELECT record FROM attempts ORDER BY sequence')]
        return {'guard':guard,'ledger_id':guard.ledger_id,'scope':scope,
            'prior_ids':{r['request_id'] for r in rows},'binding':guard._binding(client)}

def verify_reflection_observation(client,before,result):
    from .reflection_wrapper import reflection_content_projection
    guard=before['guard']
    if getattr(client,'model_request_control',None) is not guard or guard.ledger_id!=before['ledger_id']:
        raise ValueError('actual_reflection_guard_changed')
    with guard._lock:
        rows=[json.loads(v[0]) for v in guard._db.execute('SELECT record FROM attempts ORDER BY sequence')]
        newer=[r for r in rows if r['request_id'] not in before['prior_ids']]
        if (len(newer)!=1 or guard._binding(client)!=before['binding']
            or getattr(guard._local,'scope',None)!=before['scope']):
            raise ValueError('actual_unique_reflection_request_required')
        row=newer[0]
        if (row.get('status')!='received' or row.get('role')!='reflection' or row.get('model')!=client.model
            or any(row.get(k)!=v for k,v in before['scope'].items())
            or row.get('http_status')!=200 or row.get('fee_verification_error')
            or 'cost' not in row.get('usage',{})):
            raise ValueError('actual_received_reflection_required')
        source=row.get('public_first_output',{});path=Path(source.get('path','')).resolve()
        if path!=(Path(guard.public_dir)/(row['request_id']+'.json')).resolve():
            raise ValueError('actual_reflection_public_path_changed')
        raw_bytes=path.read_bytes()
        if sha256(raw_bytes).hexdigest()!=source.get('sha256'):
            raise ValueError('actual_reflection_public_source_changed')
        public=json.loads(raw_bytes)
        if (any(public.get(k)!=row[k] for k in ('request_id','request_sha256','role'))
            or public.get('hidden_reasoning_retained') is not False
            or type(public.get('public_first_output')) is not list or len(public['public_first_output'])!=1):
            raise ValueError('actual_reflection_public_identity_changed')
        content=public['public_first_output'][0]
        if type(content.get('content')) is not str or content.get('function_calls')!=[]:
            raise ValueError('actual_reflection_public_content_required')
        raw=content['content'];stream=row.get('stream_read',{})
        if (stream.get('public_content_sha256')!=sha256(raw.encode('utf8')).hexdigest()
            or stream.get('complete') is not True or stream.get('done_observed') is not True
            or stream.get('usage_observed') is not True or stream.get('finish_observed') is not True):
            raise ValueError('actual_reflection_complete_stream_required')
        projected=reflection_content_projection(raw, version=client.reflection_return_format)
        if (projected['canonical']!=result or projected['metadata']!=row.get('reflection_wrapper')
            or projected['metadata']!=getattr(client,'last_diagnostic',{}).get('reflection_wrapper')):
            raise ValueError('actual_reflection_wrapper_source_mismatch')
        return {'ledger_id':guard.ledger_id,'request_id':row['request_id'],'request_sha256':row['request_sha256'],
            'operation_id':row['operation_id'],'session_id':row['session_id'],'public_first_output':deepcopy(source),
            'raw_public_sha256':projected['metadata']['raw_public_sha256'],
            'canonical_sha256':projected['metadata']['canonical_sha256'],'same_actual_guard_receipt':True}


def record_public_model_outputs(clients, output):
    """Same logical observer as smoke_demo, with allowlisted usage retention."""
    lock, counter = Lock(), [0]
    for client in (clients.online, clients.memory, clients.reflection, clients.safety_review):
        if client is None:
            continue
        if client.role == 'reflection':
            from .sse_adapter import validate_stream_client
            validate_stream_client(client)
        if client.role == 'online_fast':
            from .online_shape import validate_online_projection_client
            validate_online_projection_client(client)
        original = client.complete
        def recorded(messages, _client=client, _complete=original, **requirements):
            with lock:
                counter[0] += 1
                number = counter[0]
            entry = {"role": _client.role, "configured_model": _client.model,
                "started_at": datetime.now(timezone.utc).isoformat(), "messages": deepcopy(messages),
                "requirements": requirements, "status": "pending", "public_output": None}
            if _client.role == 'reflection':
                from .sse_adapter import model_visible_contract_binding, validate_stream_client
                stream_contract = validate_stream_client(_client)
                entry.update(provider_schema_applied=stream_contract['provider_schema_applied'],
                    reflection_provider_schema_protocol=stream_contract['schema'],
                    local_codec_and_itemwise_validation=True)
                entry.update(model_visible_contract_binding(_client))
            try:
                reflection_before = begin_reflection_observation(_client) if _client.role == 'reflection' else None
                result = _complete(messages, **requirements)
                entry.update(status="returned", public_output=result)
                if _client.role == 'online_fast':
                    from .online_shape import OUTER_CLOSE, PADDED_OBJECT, completion_projection_metadata
                    projection_receipt = completion_projection_metadata(_client, result)
                    entry['online_response_projection'] = projection_receipt
                    entry['public_output_representation'] = {
                        'empty_response_derived': 'canonical_from_empty_response_segments',
                        OUTER_CLOSE: 'canonical_with_outer_close',
                    }.get(projection_receipt['mode'], 'unmodified_public_json')
                    entry['original_raw_contract_failure_preserved'] = (
                        projection_receipt['raw_response_contract'] in ('fail_empty_response', OUTER_CLOSE, PADDED_OBJECT))
                if _client.role == "reflection" and _client.reflection_return_format in {"json_schema_v3", "json_schema_v4"}:
                    codec = getattr(_client, "last_diagnostic", {}).get("reflection_wire_codec")
                    if (type(codec) is not dict or codec.get("version") != "reflection_" + _client.reflection_return_format
                            or codec.get("canonical_sha256") != sha256(result.encode("utf-8")).hexdigest()
                            or type(codec.get("public_wire_sha256")) is not str
                            or len(codec["public_wire_sha256"]) != 64
                            or any(c not in "0123456789abcdef" for c in codec["public_wire_sha256"])
                            or codec.get("representation_only") is not True
                            or codec.get("content_repaired") is not False):
                        raise ValueError("reflection_codec_receipt_missing_or_changed")
                    entry["public_output_representation"] = "canonical_decoded_" + _client.reflection_return_format
                    entry["public_output_codec"] = {k: codec[k] for k in (
                        "version", "public_wire_sha256", "canonical_sha256", "representation_only", "content_repaired")}
                    wrapper = getattr(_client, "last_diagnostic", {}).get("reflection_wrapper")
                    from .reflection_wrapper import VERSION as WRAPPER_VERSION
                    if (type(wrapper) is not dict or wrapper.get('version') != WRAPPER_VERSION
                            or wrapper.get('status') != 'validated'
                            or wrapper.get('wrapper') not in ('bare_json', 'single_json_fence', 'terminal_json_fence_after_prose')
                            or (_client.reflection_return_format == 'json_schema_v4' and wrapper.get('wrapper') != 'bare_json')
                            or wrapper.get('canonical_sha256') != codec['canonical_sha256']
                            or wrapper.get('serialization_body_sha256') != codec['public_wire_sha256']
                            or any(type(wrapper.get(k)) is not str or len(wrapper[k]) != 64
                                   or any(c not in '0123456789abcdef' for c in wrapper[k])
                                   for k in ('raw_public_sha256', 'serialization_body_sha256', 'canonical_sha256','prefix_sha256'))
                            or type(wrapper.get('prefix_characters')) is not int or wrapper['prefix_characters']<0
                            or (wrapper['wrapper']=='terminal_json_fence_after_prose') != (wrapper['prefix_characters']>0)
                            or wrapper.get('representation_only') is not True
                            or any(wrapper.get(k) is not False for k in
                                   ('domain_content_repaired', 'json_fields_repaired', 'source_or_identifier_repaired'))
                            or wrapper['wrapper'] == 'bare_json' and wrapper['raw_public_sha256'] != wrapper['serialization_body_sha256']):
                        raise ValueError('reflection_wrapper_codec_receipt_missing_or_changed')
                    entry['public_output_codec_input'] = 'serialization_body_after_explicit_outer_wrapper'
                    entry['public_output_wrapper'] = {k: wrapper[k] for k in (
                        'version', 'status', 'wrapper', 'raw_public_sha256', 'serialization_body_sha256',
                        'canonical_sha256','prefix_sha256','prefix_characters', 'representation_only', 'domain_content_repaired',
                        'json_fields_repaired', 'source_or_identifier_repaired')}
                    entry["actual_reflection_source_binding"] = verify_reflection_observation(_client, reflection_before, result)
                return result
            except BaseException as error:
                entry.update(status="raised", error_class=type(error).__name__)
                raise
            finally:
                _client.last_usage = safe_usage({"usage": getattr(_client, "last_usage", {})})
                entry.update(finished_at=datetime.now(timezone.utc).isoformat(),
                    effective_model=getattr(_client, "last_model", None),
                    usage=deepcopy(_client.last_usage),
                    usage_is_verified_fee=False, finish_reason=getattr(_client, "last_finish_reason", None))
                write_once(Path(output) / f"model-completion-{number:03d}.json", entry)
        client.complete = recorded
