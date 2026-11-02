"""Unambiguous public representation only; never repair domain fields or evidence."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json,re
from aria.engine.policy.reflection_compact import decode_compact_reflection, decode_indexed_compact_reflection

VERSION='reflection_terminal_single_json_fence_v2'

def contract(version='json_schema_v3'):
    if version not in {'json_schema_v3','json_schema_v4'}:
        raise ValueError('invalid_reflection_projection_version')
    return {'version':VERSION,'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'domain_codec':'reflection_'+version,'itemwise_review':True}

def select_serialization_body(raw):
    """Return the unchanged complete body and wrapper metadata, without domain decoding."""
    if type(raw) is not str or raw.strip()!=raw.strip(' \t\r\n'):
        raise ValueError('invalid_reflection_public_wrapper')
    value=raw.strip(' \t\r\n');body=raw;wrapper='bare_json';prefix=''
    # Literal fence characters inside a complete JSON string are data. Domain
    # validation (including duplicate keys) remains in the unchanged codec.
    try:json.loads(raw)
    except (ValueError,TypeError):pass
    else:return {'body':body,'wrapper':wrapper,'prefix':prefix}
    if '```' in value:
        match=re.fullmatch(r'([\s\S]*?)(?m:^```json[ \t]*\r?\n)([\s\S]*?)\r?\n```',value)
        if match is None or len(re.findall(r'(?m)^[ \t]*```',value))!=2:
            raise ValueError('ambiguous_reflection_public_wrapper')
        prefix,body=match.group(1),match.group(2)
        if any(char in prefix for char in '{}[]`'):
            raise ValueError('ambiguous_reflection_public_prefix')
        if prefix.strip():
            try:json.loads(prefix)
            except (ValueError,TypeError):pass
            else:raise ValueError('second_json_candidate_forbidden')
        wrapper='terminal_json_fence_after_prose' if prefix.strip() else 'single_json_fence'
    return {'body':body,'wrapper':wrapper,'prefix':prefix}

def reflection_content_projection(raw,version='json_schema_v3'):
    if version not in {'json_schema_v3','json_schema_v4'}:
        raise ValueError('invalid_reflection_projection_version')
    # Structured v4 accepts only the whole native JSON. The legacy explicit
    # wrapper policy remains available solely to its historical v3 route.
    selected=({'body':raw,'wrapper':'bare_json','prefix':''} if version=='json_schema_v4'
              else select_serialization_body(raw))
    body,wrapper,prefix=selected['body'],selected['wrapper'],selected['prefix']
    decoder=decode_indexed_compact_reflection if version=='json_schema_v4' else decode_compact_reflection
    canonical=decoder(body,itemwise_review=True)
    metadata={'version':VERSION,'status':'validated','wrapper':wrapper,
        'raw_public_sha256':sha256(raw.encode()).hexdigest(),
        'serialization_body_sha256':sha256(body.encode()).hexdigest(),
        'canonical_sha256':sha256(canonical.encode()).hexdigest(),
        'prefix_sha256':sha256(prefix.encode()).hexdigest(),'prefix_characters':len(prefix),
        'representation_only':True,'domain_content_repaired':False,
        'json_fields_repaired':False,'source_or_identifier_repaired':False}
    if version=='json_schema_v4':
        metadata['domain_codec']='reflection_json_schema_v4'
    return {'body':body,'canonical':canonical,'metadata':metadata}

def reflection_receipt_projection(client,row,data):
    from .sse_adapter import validate_stream_client
    validate_stream_client(client)
    try:projection=reflection_content_projection(data['choices'][0]['message']['content'],
                                                version=client.reflection_return_format)
    except (ValueError,TypeError,KeyError,IndexError,RecursionError):
        row['reflection_wrapper']={'version':VERSION,'status':'rejected',
            'representation_only':True,'domain_content_repaired':False}
        return data
    row['reflection_wrapper']=deepcopy(projection['metadata'])
    view=deepcopy(data);view['choices'][0]['message']['content']=projection['body'];return view
