"""Pure bindings/math for V127. Import performs no source/body/model I/O."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
V=Path(__file__).resolve().parent;P=V.parent
POLICY=V/'ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json';POLICY_SHA='2e49c665897bd4cdcd80506335b1489465677802be9b8bb16fb9124a8c3238a2'
ADDENDUM=V/'ROOT_PREEXECUTION_PRECISION_ADDENDUM.json';ADDENDUM_SHA='7b68155ff46e47acee408456c8660ada1d1990310604d2a8e2cb508865aaa1b6'
AXIS=(3,7,11);OUTER=(0,1,2,3,5,6,7,8);N=4896;U=4428
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def policy():
 assert sha(POLICY)==POLICY_SHA and sha(ADDENDUM)==ADDENDUM_SHA
 p=read(POLICY);assert p['root_authorized_source_only']and all(not p['constraints'][k]for k in('current223_read_fit_normalizer_threshold_checkpoint_armchoice','V112_access','PnP','q_L_xy_correction'))
 return p
def pure_functions(path,names,namespace):
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pinned_pure_functions','exec'),namespace)
 return namespace
def normalize_np(x):
 return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),np.float32(1e-12))
def stencil_numpy(grid):
 """Generated-only scalar reference; actual encoder arithmetic is explicit Torch."""
 raw=np.stack([grid[:,y:y+2,x:x+2].mean((1,2))for y in AXIS for x in AXIS],1)
 pooled=np.repeat(normalize_np(raw.mean(1))[:,None,:],8,axis=1).reshape(len(grid),-1)
 ordered=np.stack([normalize_np(raw[:,i])for i in OUTER],1).reshape(len(grid),-1)
 return raw,pooled,ordered
def state_sha(model):
 h=hashlib.sha256()
 for k,t in sorted(model.state_dict().items()):
  a=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()
