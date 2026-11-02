"""Durable owner erasure fence for private response projections, including in-flight ones."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
import json,re,sqlite3
from .validation_support import require

def serialized(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')

class ContentErasure:
    def __init__(self,ledger_path,ledger_id,public_dir,physical_dir):
        self.path=Path(ledger_path);self.ledger_id=ledger_id
        self.public=Path(public_dir).resolve();self.physical=Path(physical_dir).resolve();self.lock=RLock()
        require(self.public.parent==self.path.resolve().parent==self.physical.parent,'invalid_content_erasure_root')
        with self.db() as db:db.execute('CREATE TABLE IF NOT EXISTS content_erasure_fences(scope TEXT PRIMARY KEY,record TEXT NOT NULL)')
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=10)
        try:
            with db:yield db
        finally:db.close()
    def _scope(self,directory):
        p=Path(directory).resolve()
        require(p.parent==self.physical and re.fullmatch('[0-9a-f]{32}',p.name),'invalid_content_erasure_scope')
        return p.name
    def _read(self,db,scope):
        found=db.execute('SELECT record FROM content_erasure_fences WHERE scope=?',(scope,)).fetchone()
        if not found:return None
        row=json.loads(found[0]);require(row['schema']=='aria_private_content_erasure_v1' and row['ledger_id']==self.ledger_id
            and row['scope']==scope and row['state'] in ('pending','erased'),'invalid_content_erasure_fence')
        return row
    def check(self):
        with self.lock,self.db() as db:
            for (scope,) in db.execute('SELECT scope FROM content_erasure_fences'):
                require(self._read(db,scope)['state']=='erased','content_erasure_incomplete')
    def settled_scope(self,row):
        matches=[]
        for directory in self.physical.iterdir():
            binding=directory/'attempt-binding.json'
            if not binding.is_file():continue
            value=json.loads(binding.read_text(encoding='utf8'))
            if any(r.get('request_id')==row['request_id'] for r in value.get('new_attempts',[])):
                require(value['ledger_id']==self.ledger_id and value['new_attempts']==[row],
                    'content_erasure_physical_binding_changed')
                matches.append(self._scope(directory))
        require(len(matches)==1,'unique_content_erasure_physical_scope_required')
        return matches[0]
    def require_erased(self,scope):
        with self.lock,self.db() as db:
            value=self._read(db,scope)
            require(value is not None and value['state']=='erased','content_erasure_proof_missing')
    def write(self,path,value,directory):
        path=Path(path).resolve();scope=self._scope(directory);raw=serialized(value)
        require(path.parent==self.public or path.parent==self.physical/scope,'content_write_outside_capture')
        with self.lock,self.db() as db:
            if self._read(db,scope) is not None:return {'sha256':sha256(raw).hexdigest(),'saved':False}
            with path.open('xb') as out:out.write(raw)
        return {'sha256':sha256(raw).hexdigest(),'saved':True}
    def erase(self):
        with self.lock:
            directories=list(self.physical.iterdir())
            scopes=[self._scope(p) for p in directories]
            with self.db() as db:
                for scope in scopes:
                    old=self._read(db,scope)
                    value=old or {'schema':'aria_private_content_erasure_v1','ledger_id':self.ledger_id,
                        'scope':scope,'requested_at':datetime.now(timezone.utc).isoformat(),'state':'pending'}
                    value['state']='pending'
                    db.execute('INSERT OR REPLACE INTO content_erasure_fences VALUES(?,?)',(scope,json.dumps(value,sort_keys=True)))
            # Scope fences are durable before any deletion; callbacks can no longer persist content.
            for p in self.public.iterdir():
                require(p.resolve().parent==self.public and re.fullmatch(r'[0-9a-f]{32}\.json',p.name),'invalid_public_capture_path')
                p.unlink()
            for directory in directories:
                for name in ('first-visible.json','stream-public-frames.json'):
                    p=directory/name;require(p.resolve().parent==directory.resolve(),'invalid_physical_content_path')
                    p.unlink(missing_ok=True)
            with self.db() as db:
                for scope in scopes:
                    value=self._read(db,scope);value['state']='erased'
                    db.execute('UPDATE content_erasure_fences SET record=? WHERE scope=?',(json.dumps(value,sort_keys=True),scope))
            self.check()
