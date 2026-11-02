"""Small local validation helpers; no experiment or account configuration."""
from hashlib import sha256
import json
from pathlib import Path
from .public_capture import write_once

def require(value,reason):
    if not value:raise ValueError(reason)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def digest_json(value):
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()

def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}

def check_ref(value):
    path=Path(value['path']).resolve()
    require(ref(path)==value,'settled_validation_evidence_changed')
    return path
