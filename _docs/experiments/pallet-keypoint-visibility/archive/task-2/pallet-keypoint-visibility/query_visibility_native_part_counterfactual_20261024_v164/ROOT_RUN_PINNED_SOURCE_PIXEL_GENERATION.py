"""Root's frozen-runtime guard for the V164 source-only one-shot generation.
No pixel body is opened until every pinned dependency and code binding passes.
"""
import sys,json,hashlib,runpy
from pathlib import Path
import numpy as np
import torch,PIL,scipy
V=Path(__file__).resolve().parent
entry=V/'ROOT_ACTUAL_NATIVE_PART_PIXEL_ENTRY.json'
e=json.loads(entry.read_text())
def verified(b):
 p=Path(b['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==b['sha256'],str(p); return p
verified(e['execution_wrapper'])
p=json.loads(verified(e['protocol']).read_text())
actual=dict(executable=sys.executable,numpy=np.__version__,torch=torch.__version__,Pillow=PIL.__version__,scipy=scipy.__version__)
assert actual==p['runtime'],dict(actual=actual,expected=p['runtime'])
for name in ('root_static_review','independent_review_a','independent_review_b'):
 verified(e[name])
code=verified(e['code'])
assert not Path(e['output']).exists(),'Actual output already exists: do not restart'
print(json.dumps(dict(runtime_guard_PASS=True,runtime=actual,entry=str(entry))),flush=True)
mod=runpy.run_path(str(code),run_name='V164_ROOT_VERIFIED_LIBRARY_NO_MAIN')
mod['run'](entry)
