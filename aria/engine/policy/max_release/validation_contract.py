"""Pure existing projection rejection plus exact settled receipt, not semantics."""
from decimal import Decimal
from hashlib import sha256
from .validation_support import require
from .online_shape import online_content_projection,online_receipt_body_complete,DECISION_VERSION
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_V7
from aria.engine.policy.online_structured import DEEPSEEK_MODEL

def validate_rejected_receipt(row,output,*,requirements,request_id,operation_id,session_id):
    require(row.get('request_id')==request_id and row.get('operation_id')==operation_id
        and row.get('session_id')==session_id and row.get('role')=='online_fast','exact_online_failure_identity_required')
    cost=row.get('usage',{}).get('cost')
    require(type(cost) in (int,float,str) and Decimal(str(cost)).is_finite()
        and Decimal(str(cost))>=0,'known_finite_cost_required')
    failed=row.get('status')=='failed'
    route=(row.get('model'),row.get('expected_provider'))
    require((route==('qwen/qwen3.8-max-0902','Alibaba')
             or route==(DEEPSEEK_MODEL,'Together'))
        and row.get('status') in {'received','failed'} and row.get('http_status')==200
        and (online_receipt_body_complete(row) if 'online_stream_contract' in row or 'stream_read' in row
             else row.get('body_read',{}).get('complete') is True)
        and row.get('body_read',{}).get('capture_status')=='complete'
        and row.get('body_read',{}).get('exception_kind') is None
        and row.get('returned_model')==row.get('model')
        and row.get('returned_provider')==row.get('expected_provider')
        and row.get('receipt_identity',{}).get('model_status')=='recognized'
        and row.get('receipt_identity',{}).get('provider_status')=='recognized'
        and row.get('receipt_identity',{}).get('error_field_status') in ('absent','null')
        and (failed or not row.get('validation_error')) and not row.get('fee_verification_error')
        and row.get('raw_total_tokens_valid') is not False,'settled_complete_receipt_required')
    require(output.get('request_id')==request_id and output.get('role')=='online_fast'
        and output.get('request_sha256')==row.get('request_sha256')
        and output.get('hidden_reasoning_retained') is False,'public_output_identity_required')
    values=output.get('public_first_output')
    require(type(values) is list and len(values)==1 and type(values[0]) is dict
        and values[0].get('function_calls')==[] and type(values[0].get('content')) is str,'single_public_body_without_tool_required')
    require(type(requirements) is dict and set(requirements)=={'require_preference_control','require_task_projection'}
        and all(type(v) is bool for v in requirements.values()),'exact_request_requirements_required')
    raw=values[0]['content']
    if 'online_stream_contract' in row or 'stream_read' in row:
        require(online_receipt_body_complete(row,raw),'settled_online_stream_content_changed')
    shape=row.get('online_receipt_shape',{})
    version=shape.get('decision_version',DECISION_VERSION)
    if failed:
        usage=row['usage'];reserve=row['reservation']
        require(route==(DEEPSEEK_MODEL,'Together') and version==ONLINE_DECISION_VERSION_V7
            and row.get('kind')=='turn' and row.get('failure_phase')=='receipt_validation'
            and row.get('failure_code')==row.get('validation_error')
            and ((shape.get('finish_reason')=='length' and row['validation_error']=='invalid_or_truncated_response')
                 or (shape.get('finish_reason')=='stop' and not raw.strip()
                     and row['validation_error']=='invalid_response_message'))
            and shape.get('public_content_sha256')==sha256(raw.encode()).hexdigest()
            and row.get('raw_total_tokens_valid') is True
            and not row.get('unverified_cost_usd'), 'actual_online_format_rejection_required')
        require(all(type(usage.get(k)) is int and usage[k]>=0
                    for k in ('prompt_tokens','completion_tokens','total_tokens'))
            and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
            and usage['prompt_tokens']<=reserve['input_token_plan_bound']
            and usage['completion_tokens']<=reserve['output_token_limit']
            and Decimal(str(cost))<=Decimal(reserve['reserved_cost_usd']),
            'verified_online_usage_and_fee_required')
        return {'request_id':request_id,'cost_usd':str(Decimal(str(cost))),
            'validator':'actual_online_receipt','validator_result':row['validation_error'],
            'requirements':dict(requirements),'original_public_sha256':sha256(raw.encode()).hexdigest(),
            'original_output_status':'failed','semantic_fields_applied':False,'replacement_response':None}
    try:online_content_projection(raw,version,**requirements)
    except ValueError as exc:
        require(version==ONLINE_DECISION_VERSION_V7 or str(exc)=='invalid_online_segments_projection',
                'specific_projection_rejection_required')
        error=str(exc)
    else:raise ValueError('valid_output_or_later_failure_not_eligible')
    return {'request_id':request_id,'cost_usd':str(Decimal(str(cost))),
        'validator':'online_content_projection','validator_result':error,
        'requirements':dict(requirements),'original_public_sha256':sha256(raw.encode()).hexdigest(),
        'original_output_status':'failed','semantic_fields_applied':False,'replacement_response':None}
