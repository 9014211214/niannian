"""Bind immutable request identity and projection options; no semantic policy."""
from copy import deepcopy
from .validation_support import require

FIELD='online_projection_requirements_binding'
IDENTITY=('request_id','request_sha256','role','operation_id','session_id')
OPTIONS=('require_preference_control','require_task_projection')

def make_binding(row,requirements):
    require(set(requirements)==set(OPTIONS) and all(type(requirements[k]) is bool for k in OPTIONS),
        'exact_actual_projection_options_required')
    require(row['role']=='online_fast' and all(isinstance(row[k],str) and row[k] for k in IDENTITY),
        'exact_online_request_identity_required')
    return {'schema':'aria_actual_projection_requirements_binding_v1',
        **{k:row[k] for k in IDENTITY},'requirements':deepcopy(requirements),'bound_before_application_write':True}

def validate_binding(row,requirements):
    require(row.get(FIELD)==make_binding(row,requirements),'actual_request_projection_requirements_changed')

